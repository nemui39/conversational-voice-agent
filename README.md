# Conversational Voice Agent

マイク入力からリアルタイムで音声会話できるAIエージェント。

## アーキテクチャ

```
🎤 マイク → [STT: Whisper] → [LLM: Claude API] → [TTS: VOICEVOX] → 🔊 スピーカー
```

## 技術スタック

| コンポーネント | 技術 |
|-------------|------|
| Speech-to-Text | faster-whisper |
| LLM | Anthropic Claude API |
| Text-to-Speech | VOICEVOX |
| オーディオI/O | sounddevice |
| 言語 | Python 3.12+ |

## セットアップ

### 前提条件

- Python 3.12+
- [VOICEVOX](https://voicevox.hiroshiba.jp/) がローカルで起動していること（デフォルト: `http://localhost:50021`）
- Anthropic API Key

### インストール

```bash
git clone https://github.com/nemui39/conversational-voice-agent.git
cd conversational-voice-agent
pip install -e .
```

### 環境変数の設定

```bash
cp .env.example .env
# .env を編集して API キーを設定
```

### 実行

```bash
voice-agent
```

## 1週間ロードマップ

| Day | マイルストーン |
|-----|-------------|
| 1 | プロジェクトセットアップ + ファーストコミット |
| 2 | STT実装 (Whisper) + マイク入力 |
| 3 | LLM統合 (Claude API) |
| 4 | TTS実装 (VOICEVOX連携) |
| 5 | パイプライン結合 + リアルタイム化 |
| 6 | エラーハンドリング + 会話履歴管理 |
| 7 | テスト + ドキュメント + デモ |

## ライセンス

MIT
