"""Text-to-Speech モジュール

VOICEVOX Engine API を使ってテキストを音声に変換する。
"""

from __future__ import annotations

import os

import httpx

VOICEVOX_HOST = os.getenv("VOICEVOX_HOST", "http://localhost:50021")
DEFAULT_SPEAKER_ID = 1

_TIMEOUT = 30.0


def synthesize(
    text: str,
    speaker_id: int = DEFAULT_SPEAKER_ID,
    return_query: bool = False,
) -> bytes | tuple[bytes, dict]:
    """テキストを音声WAVデータに変換する。

    VOICEVOX の 2段階API を呼ぶ:
      1. POST /audio_query  → 音声合成用クエリJSON
      2. POST /synthesis    → WAVバイト列

    Args:
        text: 読み上げるテキスト
        speaker_id: VOICEVOX話者ID
        return_query: Trueの場合 (wav_bytes, audio_query) タプルを返す

    Returns:
        WAV形式の音声バイト列。return_query=True の場合は (wav_bytes, audio_query) タプル。
    """
    try:
        # 1. audio_query
        aq_resp = httpx.post(
            f"{VOICEVOX_HOST}/audio_query",
            params={"text": text, "speaker": speaker_id},
            timeout=_TIMEOUT,
        )
        aq_resp.raise_for_status()
        audio_query = aq_resp.json()

        # 2. synthesis
        syn_resp = httpx.post(
            f"{VOICEVOX_HOST}/synthesis",
            params={"speaker": speaker_id},
            json=audio_query,
            timeout=_TIMEOUT,
        )
        syn_resp.raise_for_status()

    except httpx.ConnectError:
        raise RuntimeError(
            f"VOICEVOX に接続できません ({VOICEVOX_HOST})。起動しているか確認してください。"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"VOICEVOX API エラー: {e.response.status_code} {e.response.text[:200]}") from e

    if return_query:
        return syn_resp.content, audio_query
    return syn_resp.content
