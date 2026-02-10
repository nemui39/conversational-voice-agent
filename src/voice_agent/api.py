"""FastAPI Web API

POST /api/coach に WAV をアップロードすると、
熱血コーチの返信 WAV が返る。
"""

from __future__ import annotations

import io
import tempfile
import wave
from urllib.parse import quote

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from voice_agent.llm import chat
from voice_agent.main import trim_for_speech
from voice_agent.stt import transcribe
from voice_agent.tts import synthesize

load_dotenv()

app = FastAPI(title="Conversational Voice Agent", version="0.1.0")

STATIC_DIR = Path(__file__).resolve().parent.parent.parent / "static"


@app.get("/", response_class=HTMLResponse)
def index():
    """簡易 Web UI を返す。"""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Conversational Voice Agent</h1><p>POST /api/coach with a WAV file.</p>"


@app.post("/api/coach")
def coach(
    file: UploadFile = File(...),
    speaker: int = Query(1, description="VOICEVOX話者ID"),
):
    """音声ファイルを受け取り、コーチの返信音声を返す。"""
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

    # TTS
    wav_bytes = synthesize(reply, speaker_id=speaker)

    # 一時ファイルに書き出して返す
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(wav_bytes)
    tmp.close()

    return FileResponse(
        tmp.name,
        media_type="audio/wav",
        filename="coach_reply.wav",
        headers={
            "X-User-Text": quote(text),
            "X-Coach-Text": quote(reply),
        },
    )
