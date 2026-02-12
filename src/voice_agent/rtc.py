"""WebSocket 音声セッション管理

ブラウザから int16 PCM 音声チャンク（48kHz mono）を WebSocket で受信し、
VAD で区切って既存パイプライン（STT→LLM→TTS）を実行。
結果は JSON テキスト + WAV バイナリで返す。
"""

from __future__ import annotations

import asyncio
import json
import logging

import numpy as np
from fastapi import WebSocket

from voice_agent.main import trim_for_speech
from voice_agent.vad import SpeechDetector
from voice_agent.viseme import extract_visemes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_SAMPLE_RATE = 48000
_TARGET_SAMPLE_RATE = 16000
_VAD_FRAME_MS = 20


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間でリサンプルする。"""
    if src_rate == dst_rate:
        return audio
    target_len = int(len(audio) * dst_rate / src_rate)
    indices = np.round(np.linspace(0, len(audio) - 1, target_len)).astype(int)
    return audio[indices]


def _run_pipeline(pcm_48k: bytes) -> dict:
    """ブロッキングの AI パイプラインを実行する（スレッドプール用）。"""
    from voice_agent.llm import chat
    from voice_agent.stt import transcribe
    from voice_agent.tts import synthesize

    audio_48k = np.frombuffer(pcm_48k, dtype=np.int16)
    audio_16k = _resample(audio_48k, _SAMPLE_RATE, _TARGET_SAMPLE_RATE)

    logger.info(f"STT input: {len(audio_16k)} samples ({len(audio_16k)/16000:.1f}s)")
    user_text = transcribe(audio_16k)
    if not user_text:
        logger.info("STT returned empty text")
        return {"user_text": "", "coach_text": "", "wav_bytes": b"", "visemes": []}

    logger.info(f"STT: {user_text}")
    coach_text = chat(user_text)
    coach_text = trim_for_speech(coach_text)
    logger.info(f"LLM: {coach_text}")

    wav_bytes, audio_query = synthesize(coach_text, return_query=True)
    visemes = extract_visemes(audio_query)
    logger.info(f"TTS: {len(wav_bytes)} bytes, {len(visemes)} visemes")

    return {
        "user_text": user_text,
        "coach_text": coach_text,
        "wav_bytes": wav_bytes,
        "visemes": visemes,
    }


def _to_vad_frames(pcm_bytes: bytes, sample_rate: int) -> list[bytes]:
    """PCM バイト列を VAD 用の 20ms フレームに分割する。"""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)

    if sample_rate != _SAMPLE_RATE:
        audio = _resample(audio, sample_rate, _SAMPLE_RATE)

    frame_samples = _SAMPLE_RATE * _VAD_FRAME_MS // 1000  # 960
    frames = []
    for i in range(0, len(audio) - frame_samples + 1, frame_samples):
        frames.append(audio[i:i + frame_samples].tobytes())
    return frames


def _transcribe_partial(pcm_48k: bytes) -> str:
    """部分認識用 STT。スレッドプールで実行される。"""
    from voice_agent.stt import transcribe

    audio_48k = np.frombuffer(pcm_48k, dtype=np.int16)
    audio_16k = _resample(audio_48k, _SAMPLE_RATE, _TARGET_SAMPLE_RATE)
    return transcribe(audio_16k)


async def _send_event(ws: WebSocket, event: dict) -> None:
    """WebSocket が接続中なら JSON を送信する。"""
    try:
        await ws.send_text(json.dumps(event))
    except Exception:
        logger.debug(f"WebSocket send failed, dropping: {event.get('type')}")


async def handle_ws_session(websocket: WebSocket) -> None:
    """WebSocket ベースの音声セッションを処理する。

    プロトコル:
    - クライアント→サーバー: バイナリ (int16 PCM, 48kHz mono)
    - サーバー→クライアント: テキスト JSON (state/visemes/result)
    - サーバー→クライアント: バイナリ (WAV 音声)
    """
    detector = SpeechDetector(sample_rate=_SAMPLE_RATE)
    processing = False
    chunk_count = 0
    last_partial_time = 0.0
    partial_interval = 1.0  # 部分認識の間隔（秒）
    partial_task: asyncio.Task | None = None

    logger.info("WebSocket audio session started")

    async def _do_partial(speech_bytes: bytes) -> None:
        """部分認識を実行してクライアントに送信する。"""
        nonlocal partial_task
        try:
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _transcribe_partial, speech_bytes)
            if text:
                logger.info(f"Partial STT: {text}")
                await _send_event(websocket, {"type": "partial_text", "text": text})
        except Exception as e:
            logger.debug(f"Partial STT error: {e}")
        finally:
            partial_task = None

    try:
        while True:
            pcm_bytes = await websocket.receive_bytes()
            chunk_count += 1

            if processing:
                continue

            vad_frames = _to_vad_frames(pcm_bytes, _SAMPLE_RATE)

            for vf in vad_frames:
                if processing:
                    break

                speech_data = detector.process_frame(vf)

                if detector.is_speaking and not processing and chunk_count % 25 == 0:
                    await _send_event(websocket, {"type": "state", "state": "LISTENING"})

                # 発話中の部分認識（1秒ごと）
                if (detector.is_speaking
                        and not processing
                        and partial_task is None):
                    now = asyncio.get_event_loop().time()
                    if now - last_partial_time >= partial_interval:
                        last_partial_time = now
                        speech_so_far = detector.get_speech_buffer()
                        # 最低0.3秒分の音声がないとスキップ
                        min_bytes = int(_SAMPLE_RATE * 2 * 0.3)
                        if len(speech_so_far) > min_bytes:
                            partial_task = asyncio.ensure_future(
                                _do_partial(speech_so_far)
                            )

                if speech_data is not None:
                    processing = True
                    last_partial_time = 0.0
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        partial_task = None
                    speech_duration = len(speech_data) / (_SAMPLE_RATE * 2)
                    logger.info(f"Speech detected: {speech_duration:.1f}s ({len(speech_data)} bytes)")
                    await _send_event(websocket, {"type": "state", "state": "THINKING"})

                    loop = asyncio.get_event_loop()
                    try:
                        result = await loop.run_in_executor(None, _run_pipeline, speech_data)
                    except Exception as e:
                        logger.error(f"Pipeline error: {e}", exc_info=True)
                        await _send_event(websocket, {"type": "state", "state": "ERROR"})
                        processing = False
                        continue

                    if not result["user_text"]:
                        await _send_event(websocket, {"type": "state", "state": "IDLE"})
                        processing = False
                        continue

                    await _send_event(websocket, {
                        "type": "result",
                        "user_text": result["user_text"],
                        "coach_text": result["coach_text"],
                    })
                    await _send_event(websocket, {"type": "visemes", "data": result["visemes"]})

                    if result["wav_bytes"]:
                        await _send_event(websocket, {"type": "state", "state": "SPEAKING"})
                        await websocket.send_bytes(result["wav_bytes"])
                    else:
                        await _send_event(websocket, {"type": "state", "state": "IDLE"})

                    processing = False

    except Exception as e:
        logger.info(f"WebSocket session ended: {e}")

    logger.info(f"Session ended after {chunk_count} audio chunks")
