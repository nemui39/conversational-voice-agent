"""WebRTC セッション管理

WebSocket シグナリングで RTCPeerConnection を確立し、
受信音声を VAD で区切って既存パイプライン（STT→LLM→TTS）を実行する。
結果は WebRTC オーディオトラック + DataChannel で返す。
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import wave

import numpy as np
from aiortc import RTCPeerConnection, RTCSessionDescription
from fastapi import WebSocket

from voice_agent.main import trim_for_speech
from voice_agent.rtc_tracks import TTSPlaybackTrack
from voice_agent.vad import SpeechDetector
from voice_agent.viseme import extract_visemes

logger = logging.getLogger(__name__)

# aiortc は 48kHz で音声を扱う
_RTC_SAMPLE_RATE = 48000
_TARGET_SAMPLE_RATE = 16000


def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """線形補間でリサンプルする。"""
    if src_rate == dst_rate:
        return audio
    target_len = int(len(audio) * dst_rate / src_rate)
    indices = np.round(np.linspace(0, len(audio) - 1, target_len)).astype(int)
    return audio[indices]


def _run_pipeline(pcm_48k: bytes) -> dict:
    """ブロッキングの AI パイプラインを実行する（スレッドプール用）。

    Args:
        pcm_48k: 48kHz int16 モノラル PCM バイト列

    Returns:
        {user_text, coach_text, wav_bytes, audio_query, visemes}
    """
    from voice_agent.llm import chat
    from voice_agent.stt import transcribe
    from voice_agent.tts import synthesize

    # 48kHz → 16kHz にリサンプル
    audio_48k = np.frombuffer(pcm_48k, dtype=np.int16)
    audio_16k = _resample(audio_48k, _RTC_SAMPLE_RATE, _TARGET_SAMPLE_RATE)

    # STT
    user_text = transcribe(audio_16k)
    if not user_text:
        return {"user_text": "", "coach_text": "", "wav_bytes": b"", "audio_query": {}, "visemes": []}

    # LLM
    coach_text = chat(user_text)
    coach_text = trim_for_speech(coach_text)

    # TTS
    wav_bytes, audio_query = synthesize(coach_text, return_query=True)

    # ビセム
    visemes = extract_visemes(audio_query)

    return {
        "user_text": user_text,
        "coach_text": coach_text,
        "wav_bytes": wav_bytes,
        "audio_query": audio_query,
        "visemes": visemes,
    }


def _wav_to_pcm_48k(wav_bytes: bytes) -> np.ndarray:
    """WAV バイト列を 48kHz int16 PCM に変換する。"""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        channels = wf.getnchannels()

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio[::channels]

    return _resample(audio, rate, _RTC_SAMPLE_RATE)


async def handle_rtc_session(websocket: WebSocket) -> None:
    """1 クライアントの WebRTC セッションを管理する。

    1. WebSocket で SDP offer 受信 → answer 返送
    2. 受信音声を VAD で監視
    3. 発話検出 → パイプライン実行 → 結果送信
    """
    pc = RTCPeerConnection()
    detector = SpeechDetector(sample_rate=_RTC_SAMPLE_RATE)
    playback_track = TTSPlaybackTrack()
    dc = pc.createDataChannel("events")
    processing = False

    def send_event(event: dict) -> None:
        """DataChannel が open なら JSON を送信する。"""
        if dc.readyState == "open":
            dc.send(json.dumps(event))

    @pc.on("track")
    async def on_track(track):
        nonlocal processing
        if track.kind != "audio":
            return

        logger.info("Audio track received from browser")
        send_event({"type": "state", "state": "IDLE"})

        while True:
            try:
                frame = await track.recv()
            except Exception:
                break

            if processing:
                continue

            # AudioFrame → PCM bytes
            pcm_bytes = bytes(frame.planes[0])

            # フレームサンプル数を確認して 20ms に切り出す
            # aiortc は通常 960 samples @ 48kHz = 20ms
            expected_bytes = _RTC_SAMPLE_RATE * 2 * 20 // 1000  # 1920 bytes
            if len(pcm_bytes) != expected_bytes:
                # フレームサイズが異なる場合はスキップ
                continue

            speech_data = detector.process_frame(pcm_bytes)

            # 発話中の状態通知
            if detector.is_speaking and not processing:
                send_event({"type": "state", "state": "LISTENING"})

            if speech_data is not None:
                processing = True
                send_event({"type": "state", "state": "THINKING"})

                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(None, _run_pipeline, speech_data)
                except Exception as e:
                    logger.error(f"Pipeline error: {e}")
                    send_event({"type": "state", "state": "ERROR"})
                    processing = False
                    continue

                if not result["user_text"]:
                    send_event({"type": "state", "state": "IDLE"})
                    processing = False
                    continue

                # テキスト結果送信
                send_event({
                    "type": "result",
                    "user_text": result["user_text"],
                    "coach_text": result["coach_text"],
                })

                # ビセム送信
                send_event({"type": "visemes", "data": result["visemes"]})

                # TTS 音声を WebRTC で再生
                if result["wav_bytes"]:
                    pcm_48k = _wav_to_pcm_48k(result["wav_bytes"])
                    send_event({"type": "state", "state": "SPEAKING"})
                    done_event = playback_track.play(pcm_48k)
                    await done_event.wait()

                send_event({"type": "state", "state": "IDLE"})
                processing = False

    # TTS 再生用トラックを追加
    pc.addTrack(playback_track)

    # WebSocket シグナリングループ
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)

            if msg.get("type") == "offer":
                offer = RTCSessionDescription(sdp=msg["sdp"], type="offer")
                await pc.setRemoteDescription(offer)

                answer = await pc.createAnswer()
                await pc.setLocalDescription(answer)

                await websocket.send_json({
                    "type": "answer",
                    "sdp": pc.localDescription.sdp,
                })
                logger.info("SDP answer sent")

    except Exception as e:
        logger.info(f"WebSocket closed: {e}")
    finally:
        await pc.close()
        logger.info("RTCPeerConnection closed")
