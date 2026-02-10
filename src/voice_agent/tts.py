"""Text-to-Speech モジュール

VOICEVOX Engine API を使ってテキストを音声に変換する。
"""

from __future__ import annotations

import os

VOICEVOX_HOST = os.getenv("VOICEVOX_HOST", "http://localhost:50021")
DEFAULT_SPEAKER_ID = 1


def synthesize(text: str, speaker_id: int = DEFAULT_SPEAKER_ID) -> bytes:
    """テキストを音声WAVデータに変換する。

    Args:
        text: 読み上げるテキスト
        speaker_id: VOICEVOX話者ID

    Returns:
        WAV形式の音声バイト列
    """
    # TODO: Day 4 で実装 (VOICEVOX REST API)
    raise NotImplementedError
