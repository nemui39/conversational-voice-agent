"""オーディオ入出力モジュール

マイクからの音声録音とスピーカーへの音声再生を担当する。
"""

from __future__ import annotations

import io
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd

from voice_agent.config import BLOCK_SIZE, CHANNELS, DTYPE, SAMPLE_RATE


def record_until_silence(
    silence_threshold: float = 500.0,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
) -> np.ndarray:
    """マイクから音声を録音し、無音を検知したら停止する。

    録音開始後、音声が入ってから silence_duration 秒間
    無音が続くと停止する。音声が一度も入らなかった場合は
    max_duration で打ち切る。

    Args:
        silence_threshold: 無音と判定するRMS振幅の閾値
        silence_duration: 無音がこの秒数続いたら録音停止
        max_duration: 最大録音時間（秒）

    Returns:
        録音された音声データ (int16 numpy array)
    """
    chunks: list[np.ndarray] = []
    silent_chunks = 0
    has_voice = False
    max_silent_chunks = int(silence_duration * SAMPLE_RATE / BLOCK_SIZE)
    max_chunks = int(max_duration * SAMPLE_RATE / BLOCK_SIZE)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        blocksize=BLOCK_SIZE,
    ) as stream:
        for _ in range(max_chunks):
            data, overflowed = stream.read(BLOCK_SIZE)
            chunk = data[:, 0] if data.ndim > 1 else data.flatten()
            chunks.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2))

            if rms > silence_threshold:
                has_voice = True
                silent_chunks = 0
            else:
                silent_chunks += 1

            if has_voice and silent_chunks >= max_silent_chunks:
                break

    if not chunks:
        return np.array([], dtype=np.int16)

    return np.concatenate(chunks)


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """numpy配列をWAVバイト列に変換する。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


def save_wav(audio: np.ndarray, path: str | Path = "record.wav") -> Path:
    """デバッグ用：録音データをWAVファイルに保存する。"""
    path = Path(path)
    wav_bytes = audio_to_wav_bytes(audio)
    path.write_bytes(wav_bytes)
    duration = len(audio) / SAMPLE_RATE
    print(f"  保存: {path} ({duration:.1f}秒, {len(audio)} samples)")
    return path


def play_wav_bytes(wav_bytes: bytes) -> None:
    """WAVバイト列をスピーカーで再生する。"""
    buf = io.BytesIO(wav_bytes)
    with wave.open(buf, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        channels = wf.getnchannels()

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sd.play(audio, samplerate=rate)
    sd.wait()
