"""メインエントリポイント

音声会話パイプライン全体を統合する。
マイク → STT → LLM → TTS → スピーカー のループを実行する。
"""

from __future__ import annotations

import argparse
import sys

from dotenv import load_dotenv


def run_stt_loop(save_wav: bool = False) -> None:
    """Day2 デバッグモード: 録音 → STT → print のループ。"""
    from voice_agent.audio import record_until_silence, save_wav as _save_wav
    from voice_agent.stt import transcribe

    print("=== Day2: 録音 → STT テストモード ===")
    print("マイクに向かって話してください。Ctrl+C で終了。")
    print()

    round_num = 0
    while True:
        round_num += 1
        print(f"--- [{round_num}] 録音中... (話し終わると自動停止) ---")
        try:
            audio = record_until_silence()
        except Exception as e:
            print(f"  録音エラー: {e}")
            print("  マイクデバイスを確認してください。")
            sys.exit(1)

        if len(audio) == 0:
            print("  音声が取れませんでした。もう一度試してください。")
            continue

        duration = len(audio) / 16000
        print(f"  録音完了: {duration:.1f}秒")

        if save_wav:
            _save_wav(audio, f"record_{round_num:03d}.wav")

        print("  STT処理中...")
        try:
            text = transcribe(audio)
        except Exception as e:
            print(f"  STTエラー: {e}")
            continue

        if not text:
            print("  (テキストなし — 音声が短すぎるかも)")
        else:
            print(f"  >> {text}")
        print()


def main() -> None:
    """音声会話エージェントのメインループ。"""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Conversational Voice Agent")
    parser.add_argument(
        "--mode",
        choices=["stt", "full"],
        default="full",
        help="実行モード: stt=録音→STTのみ, full=全パイプライン",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="録音データをWAVファイルに保存する (デバッグ用)",
    )
    args = parser.parse_args()

    if args.mode == "stt":
        try:
            run_stt_loop(save_wav=args.save_wav)
        except KeyboardInterrupt:
            print("\n終了します。")
        return

    # full モード (Day5 で実装予定)
    print("=== Conversational Voice Agent ===")
    print("full モードは Day5 で実装予定です。")
    print("今は --mode stt で STT テストができます:")
    print("  python -m voice_agent.main --mode stt")


if __name__ == "__main__":
    main()
