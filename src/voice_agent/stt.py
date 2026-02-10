"""Speech-to-Text モジュール

faster-whisper を使って音声をテキストに変換する。
"""

from __future__ import annotations

import numpy as np


def transcribe(audio: np.ndarray, language: str = "ja") -> str:
    """音声データをテキストに変換する。

    Args:
        audio: 音声データ (int16 numpy array, 16kHz mono)
        language: 認識言語コード

    Returns:
        認識されたテキスト
    """
    # TODO: Day 2 で実装 (faster-whisper)
    raise NotImplementedError
