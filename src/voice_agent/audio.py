"""オーディオ入出力モジュール

マイクからの音声録音とスピーカーへの音声再生を担当する。
"""

from __future__ import annotations

import io
import wave

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
DTYPE = "int16"


def record_until_silence(
    silence_threshold: float = 500.0,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
) -> np.ndarray:
    """マイクから音声を録音し、無音を検知したら停止する。

    Args:
        silence_threshold: 無音と判定する振幅の閾値
        silence_duration: 無音がこの秒数続いたら録音停止
        max_duration: 最大録音時間（秒）

    Returns:
        録音された音声データ (int16 numpy array)
    """
    # TODO: Day 2 で実装
    raise NotImplementedError


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """numpy配列をWAVバイト列に変換する。"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # int16 = 2 bytes
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    return buf.getvalue()


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
