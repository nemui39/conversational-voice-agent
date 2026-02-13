# Conversational Voice Agent

ブラウザ上の3Dアバターとリアルタイムで音声会話できる熱血AIコーチ。

VRMアバターがリップシンク付きで応答し、ユーザーの発話をリアルタイムに認識・返答する。

## デモ

```
🎤 ブラウザ(マイク) → WebSocket(PCM 48kHz) → VAD → STT → LLM → TTS → WebSocket(WAV) → 🔊 ブラウザ(再生+リップシンク)
```

## アーキテクチャ

```
┌─────────────────────────────────────────────────────┐
│  Browser (static/index.html)                        │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ AudioWork│  │ WebSocket│  │ Three.js + VRM    │  │
│  │ let(mic) │→ │ (PCM)    │  │ アバター+リップ   │  │
│  └──────────┘  └────┬─────┘  │ シンク+表情制御   │  │
│                     │        └───────────────────┘  │
└─────────────────────┼───────────────────────────────┘
                      │ WebSocket (int16 PCM / JSON / WAV)
┌─────────────────────┼───────────────────────────────┐
│  Server (FastAPI)   │                               │
│  ┌──────────────────▼──────────────────────────┐    │
│  │ VAD (webrtcvad) → 発話区間検出              │    │
│  │       ↓                                     │    │
│  │ STT (faster-whisper) → テキスト化           │    │
│  │       ↓                                     │    │
│  │ LLM (Claude Sonnet 4.5) → 応答生成         │    │
│  │       ↓                                     │    │
│  │ TTS (VOICEVOX) → WAV + viseme生成          │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## 技術スタック

| コンポーネント | 技術 | 詳細 |
|-------------|------|------|
| LLM | Anthropic Claude API | Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`) |
| STT | faster-whisper | small モデル, CPU int8量子化, polyphase リサンプル |
| TTS | VOICEVOX Engine API | ローカル起動, 話者ID=1 |
| VAD | webrtcvad | aggressiveness=2, 無音閾値0.4秒 |
| 3Dアバター | Three.js + @pixiv/three-vrm | VRMモデル描画, BlendShape制御 |
| リップシンク | VOICEVOX音素タイミング | `audio_query` の母音+長さ → visemeマッピング → BlendShape |
| 音声通信 | WebSocket | AudioWorkletでPCM 48kHz送信, WAVバイナリ返送 |
| サーバー | FastAPI + uvicorn | 非同期WebSocket, スレッドプールでSTT/LLM/TTS実行 |
| 言語 | Python 3.12+ / JavaScript (ES Modules) | |

## 主な機能

- **リアルタイム音声会話**: ブラウザのマイクから話しかけるとアバターが音声で応答
- **3D VRMアバター**: 待機中の呼吸・瞬きアニメーション、状態に応じた表情変化
- **リップシンク**: VOICEVOX音素データに基づくリアルタイム口パク
- **STT疑似ストリーミング**: 発話中に1秒間隔で部分認識テキストを表示
- **エコー防止**: TTS再生中のマイクミュート + 500msクールダウン
- **ハルシネーション対策**: Whisper `no_speech_prob` フィルタ + 既知パターンブロック
- **音声前処理**: DC除去, RMS正規化, 無音ゲートでSTT精度を向上

## AIコーチのプロンプト設計

LLMには「情熱型AIコーチ」のシステムプロンプトを設定:

- **口調**: タメ口で熱く、「いいね！」「最高だ！」など肯定から入る
- **構成**: 肯定・承認(1文) → 核心のアドバイス(1〜2文) → 次の一歩(1文)
- **制約**: 最大3文、句読点多め（音声で聞きやすく）、会話履歴は直近6ターン保持

## セットアップ

### 前提条件

- Python 3.12+
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

### Webアプリ（アバター + 音声会話）

```bash
# VOICEVOX を先に起動しておく
uvicorn voice_agent.api:app --reload --host 0.0.0.0 --port 8000

# ブラウザで http://localhost:8000 を開く
# 「Connect」ボタンを押してマイクを許可 → 話しかける
```

### CLIモード（デバッグ用）

```bash
# サンプル音声で実行
python -m voice_agent.main --file samples/in_motivation.wav --no-play

# マイク入力テスト
python -m voice_agent.main --mode stt
```

## ライセンス

MIT
