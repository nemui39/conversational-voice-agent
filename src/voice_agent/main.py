"""メインエントリポイント

音声会話パイプライン全体を統合する。
マイク → STT → LLM → TTS → スピーカー のループを実行する。
"""

from __future__ import annotations

from dotenv import load_dotenv


def main() -> None:
    """音声会話エージェントのメインループ。"""
    load_dotenv()

    print("=== Conversational Voice Agent ===")
    print("マイクに向かって話しかけてください。Ctrl+C で終了します。")
    print()

    # TODO: Day 5 でパイプラインを結合
    # conversation_history = []
    # while True:
    #     1. record_until_silence() でマイク入力を取得
    #     2. transcribe() で音声→テキスト
    #     3. chat() でLLM応答を生成
    #     4. synthesize() でテキスト→音声
    #     5. play_wav_bytes() でスピーカー出力

    print("パイプラインは現在開発中です。Day 5 で統合予定。")


if __name__ == "__main__":
    main()
