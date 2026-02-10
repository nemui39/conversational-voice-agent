"""LLM モジュール

Anthropic Claude API を使ってユーザー入力に応答する。
"""

from __future__ import annotations


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """ユーザーメッセージに対する応答を生成する。

    Args:
        user_message: ユーザーの発話テキスト
        history: 会話履歴 (Anthropic messages形式)

    Returns:
        アシスタントの応答テキスト
    """
    # TODO: Day 3 で実装 (anthropic SDK)
    raise NotImplementedError
