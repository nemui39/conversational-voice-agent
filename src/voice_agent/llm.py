"""LLM モジュール

Anthropic Claude API を使ってユーザー入力に応答する。
"""

from __future__ import annotations

import os
import sys

import anthropic

SYSTEM_PROMPT = """\
あなたは情熱型AIコーチです。以下のルールを厳守してください。

【口調】
- 短く、力強く、肯定から入る
- 「いいね！」「最高だ！」「その調子！」など前向きな一言で始める
- 敬語は使わない。タメ口で熱く語る

【構成（必ずこの順）】
1. 肯定・承認（1文）
2. 核心のアドバイスまたは回答（1〜2文）
3. 「次の一歩」を1つだけ提示（1文）

【制約】
- 最大3文。絶対に4文以上にしない
- 句読点を多めに入れて、音声で聞きやすくする
- 著名人の名前や決め台詞は使わない
- 質問には的確に答えた上で、前向きに背中を押す
"""

MAX_HISTORY_TURNS = 6  # user+assistantで1ターン = 最大12メッセージ
MODEL = "claude-sonnet-4-5-20250929"


def _get_client() -> anthropic.Anthropic:
    """APIクライアントを取得する。キー未設定なら明確にエラーを出す。"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("エラー: ANTHROPIC_API_KEY が設定されていません。")
        print("  cp .env.example .env して API キーを記入してください。")
        sys.exit(1)
    return anthropic.Anthropic(api_key=api_key)


def chat(user_message: str, history: list[dict] | None = None) -> str:
    """ユーザーメッセージに対する応答を生成する。

    Args:
        user_message: ユーザーの発話テキスト
        history: 会話履歴 (Anthropic messages形式)。この関数が末尾に追記する。

    Returns:
        アシスタントの応答テキスト
    """
    if history is None:
        history = []

    history.append({"role": "user", "content": user_message})

    # 履歴が長くなりすぎたら古いものを削る
    max_messages = MAX_HISTORY_TURNS * 2
    if len(history) > max_messages:
        history[:] = history[-max_messages:]

    client = _get_client()

    try:
        response = client.messages.create(
            model=MODEL,
            max_tokens=256,
            system=SYSTEM_PROMPT,
            messages=history,
        )
    except anthropic.BadRequestError as e:
        if "credit balance" in str(e):
            raise RuntimeError(
                "APIクレジット不足です。https://console.anthropic.com/settings/billing でチャージしてください。"
            ) from e
        raise RuntimeError(f"API リクエストエラー: {e}") from e
    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "API キーが無効です。.env の ANTHROPIC_API_KEY を確認してください。"
        ) from e
    except anthropic.APIError as e:
        raise RuntimeError(f"Claude API エラー: {e}") from e

    assistant_text = response.content[0].text
    history.append({"role": "assistant", "content": assistant_text})

    return assistant_text
