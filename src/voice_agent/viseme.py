"""Viseme extraction from VOICEVOX audio_query mora data.

VOICEVOX audio_query の accent_phrases/moras 構造からビセムタイムラインを生成する。
各モーラの母音(a/i/u/e/o/N)をビセムタイプ(A/I/U/E/O/N)にマッピングし、
consonant_length + vowel_length から絶対時刻を計算する。
"""

from __future__ import annotations

# 日本語母音 → ビセムタイプ
# 大文字は無声母音（例: "U" = 無声の u）
_VOWEL_TO_VISEME = {
    "a": "A", "i": "I", "u": "U", "e": "E", "o": "O",
    "A": "A", "I": "I", "U": "U", "E": "E", "O": "O",
    "N": "N",
}


def extract_visemes(audio_query: dict) -> list[dict]:
    """VOICEVOX audio_query JSON からビセムタイムラインを抽出する。

    Args:
        audio_query: VOICEVOX /audio_query エンドポイントの応答 JSON

    Returns:
        ビセムイベントのリスト。各要素は:
        - t: 開始時刻（秒）
        - v: ビセムタイプ（"A"/"I"/"U"/"E"/"O"/"N"）
        - dur: 持続時間（秒）
        - unvoiced: 無声母音かどうか
    """
    visemes: list[dict] = []
    t = audio_query.get("prePhonemeLength", 0.0)

    for phrase in audio_query.get("accent_phrases", []):
        for mora in phrase.get("moras", []):
            # 子音フェーズ: 時刻を進める（口の遷移期間）
            consonant_len = mora.get("consonant_length") or 0.0
            t += consonant_len

            # 母音フェーズ: ビセムイベントを生成
            vowel = mora.get("vowel", "")
            vowel_len = mora.get("vowel_length") or 0.0
            viseme_type = _VOWEL_TO_VISEME.get(vowel, "N")

            if vowel_len > 0:
                visemes.append({
                    "t": round(t, 4),
                    "v": viseme_type,
                    "dur": round(vowel_len, 4),
                    "unvoiced": vowel.isupper() and vowel != "N",
                })
            t += vowel_len

        # アクセント句間のポーズ
        pause = phrase.get("pause_mora")
        if pause:
            pause_len = pause.get("vowel_length") or 0.0
            if pause_len > 0:
                visemes.append({
                    "t": round(t, 4),
                    "v": "N",
                    "dur": round(pause_len, 4),
                    "unvoiced": False,
                })
            t += pause_len

    return visemes
