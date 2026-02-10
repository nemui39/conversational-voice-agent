"""メインエントリポイント

音声会話パイプライン全体を統合する。
マイク → STT → LLM → TTS → スピーカー のループを実行する。
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from dotenv import load_dotenv


def _load_wav(file_path: str) -> "np.ndarray":
    """WAVファイルを16kHz mono int16で読み込む。"""
    import wave

    import numpy as np

    with wave.open(file_path, "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        rate = wf.getframerate()
        channels = wf.getnchannels()
        duration = wf.getnframes() / rate

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio[::channels]

    print(f"  {rate}Hz / {channels}ch / {duration:.1f}秒")

    if rate != 16000:
        target_len = int(len(audio) * 16000 / rate)
        indices = np.round(np.linspace(0, len(audio) - 1, target_len)).astype(int)
        audio = audio[indices]
        print(f"  → 16kHz にリサンプル ({target_len} samples)")

    return audio


def trim_for_speech(text: str, max_sentences: int = 3) -> str:
    """音声向けにテキストを整形する。最大 max_sentences 文に切り詰める。"""
    # 句点・感嘆符・疑問符で文を分割
    sentences = re.split(r'(?<=[。！？!?])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= max_sentences:
        return text.strip()

    return "".join(sentences[:max_sentences])


def run_pipeline(
    file_path: str,
    out_path: str = "outputs/reply.wav",
    speaker_id: int = 1,
    no_play: bool = False,
) -> None:
    """WAVファイル → STT → LLM → TTS → 出力 のフルパイプライン。"""
    from voice_agent.llm import chat
    from voice_agent.stt import transcribe
    from voice_agent.tts import synthesize

    print(f"=== Conversational Voice Agent ===")
    print(f"  入力: {file_path}")

    audio = _load_wav(file_path)

    print("  STT処理中...")
    text = transcribe(audio)

    if not text:
        print("  (テキストなし — 音声が短すぎるかも)")
        return

    print(f"  あなた: {text}")
    print("  コーチ応答中...")

    reply = chat(text)
    reply = trim_for_speech(reply)
    print(f"  コーチ: {reply}")

    print(f"  音声合成中... (speaker={speaker_id})")
    wav_bytes = synthesize(reply, speaker_id=speaker_id)

    # 出力ディレクトリを作成
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(wav_bytes)
    print(f"  保存: {out_path} ({len(wav_bytes):,} bytes)")

    if not no_play:
        try:
            from voice_agent.audio import play_wav_bytes
            print("  再生中...")
            play_wav_bytes(wav_bytes)
        except Exception:
            print("  (再生デバイスなし — WAVファイルを直接開いてください)")


def run_stt_loop(save_wav: bool = False) -> None:
    """録音 → STT → print のループ（マイクテスト用）。"""
    from voice_agent.audio import record_until_silence, save_wav as _save_wav
    from voice_agent.stt import transcribe

    print("=== 録音 → STT テストモード ===")
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
            print("  WSL2の場合: --file test.wav でファイル入力テストできます")
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

    parser = argparse.ArgumentParser(
        description="Conversational Voice Agent — 熱血AIコーチ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
使用例:
  python -m voice_agent.main --file input.wav
  python -m voice_agent.main --file input.wav --out reply.wav --no-play
  python -m voice_agent.main --file input.wav --speaker 3
  python -m voice_agent.main --mode stt
""",
    )
    parser.add_argument(
        "--file",
        help="入力WAVファイル（音声→STT→LLM→TTS）",
    )
    parser.add_argument(
        "--out",
        default="outputs/reply.wav",
        help="出力WAVファイルパス (default: outputs/reply.wav)",
    )
    parser.add_argument(
        "--speaker",
        type=int,
        default=1,
        help="VOICEVOX話者ID (default: 1)",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="音声再生をスキップ（WAV保存のみ）",
    )
    parser.add_argument(
        "--mode",
        choices=["stt"],
        help="stt=録音→STTのみ（マイクテスト用）",
    )
    parser.add_argument(
        "--save-wav",
        action="store_true",
        help="録音データをWAVに保存 (--mode stt 用)",
    )
    args = parser.parse_args()

    if args.file:
        run_pipeline(
            file_path=args.file,
            out_path=args.out,
            speaker_id=args.speaker,
            no_play=args.no_play,
        )
        return

    if args.mode == "stt":
        try:
            run_stt_loop(save_wav=args.save_wav)
        except KeyboardInterrupt:
            print("\n終了します。")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
