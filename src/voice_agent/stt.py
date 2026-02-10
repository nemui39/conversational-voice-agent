"""Speech-to-Text モジュール

faster-whisper を使って音声をテキストに変換する。
"""

from __future__ import annotations

import numpy as np
from faster_whisper import WhisperModel

from voice_agent.config import SAMPLE_RATE

_model: WhisperModel | None = None

MODEL_SIZE = "small"


def _get_model() -> WhisperModel:
    """モデルをシングルトンでロードする（初回のみ時間がかかる）。"""
    global _model
    if _model is None:
        print(f"  Whisper モデル '{MODEL_SIZE}' をロード中...")
        _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        print("  ロード完了")
    return _model


def transcribe(audio: np.ndarray, language: str = "ja") -> str:
    """音声データをテキストに変換する。

    Args:
        audio: 音声データ (int16 numpy array, 16kHz mono)
        language: 認識言語コード

    Returns:
        認識されたテキスト
    """
    if len(audio) == 0:
        return ""

    # faster-whisper は float32 (-1.0 ~ 1.0) を期待する
    audio_f32 = audio.astype(np.float32) / 32768.0

    model = _get_model()
    segments, info = model.transcribe(
        audio_f32,
        language=language,
        beam_size=5,
        vad_filter=True,
    )

    text = "".join(seg.text for seg in segments).strip()
    return text
