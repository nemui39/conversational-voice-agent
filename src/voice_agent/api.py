"""FastAPI Web API

POST /api/coach に WAV をアップロードすると、
コーチの返信音声 + ビセムタイムラインを JSON で返す。
"""

from __future__ import annotations

import base64
import io
import wave

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from voice_agent.llm import chat
from voice_agent.main import trim_for_speech
from voice_agent.stt import transcribe
from voice_agent.tts import synthesize
from voice_agent.viseme import extract_visemes

load_dotenv()

app = FastAPI(title="Conversational Voice Agent", version="0.3.0")

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """Web UI を返す。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Conversational Voice Agent</h1><p>POST /api/coach with a WAV file.</p>"


@app.post("/api/coach")
def coach(
    file: UploadFile = File(...),
    speaker: int = Query(1, description="VOICEVOX話者ID"),
):
    """音声ファイルを受け取り、コーチの返信音声+ビセムデータを返す。"""
    # 入力WAVを読み込み
    try:
        raw = file.file.read()
        buf = io.BytesIO(raw)
        with wave.open(buf, "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            rate = wf.getframerate()
            channels = wf.getnchannels()

        audio = np.frombuffer(frames, dtype=np.int16)
        if channels > 1:
            audio = audio[::channels]

        # 16kHz にリサンプル
        if rate != 16000:
            target_len = int(len(audio) * 16000 / rate)
            indices = np.round(np.linspace(0, len(audio) - 1, target_len)).astype(int)
            audio = audio[indices]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"WAVファイルの読み込みに失敗: {e}")

    # STT
    text = transcribe(audio)
    if not text:
        raise HTTPException(status_code=422, detail="音声からテキストを認識できませんでした")

    # LLM
    reply = chat(text)
    reply = trim_for_speech(reply)

    # TTS (audio_query も取得してビセム抽出に使う)
    wav_bytes, audio_query = synthesize(reply, speaker_id=speaker, return_query=True)

    # ビセムタイムライン抽出
    visemes = extract_visemes(audio_query)

    # 音声を base64 エンコード
    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

    return JSONResponse({
        "status": "ok",
        "user_text": text,
        "coach_text": reply,
        "audio_base64": audio_b64,
        "sample_rate": audio_query.get("outputSamplingRate", 24000),
        "visemes": visemes,
    })


@app.websocket("/ws/rtc")
async def rtc_ws(websocket: WebSocket):
    """WebRTC シグナリング用 WebSocket エンドポイント。"""
    from voice_agent.rtc import handle_rtc_session

    await websocket.accept()
    try:
        await handle_rtc_session(websocket)
    except WebSocketDisconnect:
        pass


# 静的ファイル配信（VRMモデル等）— ルート定義の後にマウント
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
