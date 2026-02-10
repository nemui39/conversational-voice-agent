# Conversational Voice Agent

音声入力からリアルタイムで会話できる熱血AIコーチ。

## アーキテクチャ

```
🎤 音声入力 → [STT: Whisper] → [LLM: Claude API] → [TTS: VOICEVOX] → 🔊 音声出力
```

## 技術スタック

| コンポーネント | 技術 |
|-------------|------|
| Speech-to-Text | faster-whisper (small) |
| LLM | Anthropic Claude API (Sonnet) |
| Text-to-Speech | VOICEVOX |
| オーディオI/O | sounddevice |
| 言語 | Python 3.12+ |

## セットアップ

### 前提条件

- Python 3.12+
- PortAudio（`sudo apt install libportaudio2` / `brew install portaudio`）
- [VOICEVOX](https://voicevox.hiroshiba.jp/) がローカルで起動していること（デフォルト: `http://localhost:50021`）
- Anthropic API Key

### インストール

```bash
git clone https://github.com/nemui39/conversational-voice-agent.git
cd conversational-voice-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 環境変数の設定

```bash
cp .env.example .env
# .env を編集して ANTHROPIC_API_KEY を設定
```

## 使い方

### デモ（サンプル音声で実行）

```bash
# やる気が出ないときのコーチ応答
python -m voice_agent.main --file samples/in_motivation.wav --no-play

# プレゼン前の緊張へのコーチ応答
python -m voice_agent.main --file samples/in_anxiety.wav --no-play

# 出力先を指定
python -m voice_agent.main --file samples/in_motivation.wav --out outputs/reply.wav --no-play

# VOICEVOX 話者を変更（ID=3 など）
python -m voice_agent.main --file input.wav --speaker 3 --no-play
```

出力WAVは `outputs/reply.wav` に保存されます。

### マイク入力テスト

```bash
python -m voice_agent.main --mode stt
python -m voice_agent.main --mode stt --save-wav
```

## ライセンス

MIT
