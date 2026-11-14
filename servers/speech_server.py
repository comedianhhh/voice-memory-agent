"""Local, free, OpenAI-compatible speech server: Whisper STT + Kokoro TTS on the GPU.

LiveKit's openai plugin talks to any server with the same routes, so the agent points
`openai.STT(base_url=...)` and `openai.TTS(base_url=...)` here instead of api.openai.com.

Routes:
  POST /v1/audio/transcriptions   multipart: file (wav), model, language, response_format
  POST /v1/audio/speech           json: input, voice, response_format (pcm only), speed
  GET  /health

Run:  .venv/Scripts/python.exe speech_server.py   (port 8880)
Env:  WHISPER_MODEL (default large-v3-turbo), KOKORO_VOICE (default af_heart)
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time

# CTranslate2 (faster-whisper) needs cuBLAS/cuDNN; torch's wheel already ships them.
import torch

if sys.platform == "win32":
    os.add_dll_directory(os.path.join(os.path.dirname(torch.__file__), "lib"))

import numpy as np
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel
from kokoro import KPipeline
from pydantic import BaseModel

log = logging.getLogger("speech")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3-turbo")
DEFAULT_VOICE = os.getenv("KOKORO_VOICE", "af_heart")
TTS_SAMPLE_RATE = 24000  # Kokoro's native rate, also LiveKit openai.TTS's default

log.info("loading whisper %s on cuda", WHISPER_MODEL)
whisper = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
log.info("loading kokoro")
kokoro = KPipeline(lang_code="a", device="cuda", repo_id="hexgrad/Kokoro-82M")

app = FastAPI()


@app.get("/health")
def health() -> dict:
    return {"ok": True, "whisper": WHISPER_MODEL, "voice": DEFAULT_VOICE}


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form("whisper-1"),
    language: str | None = Form(None),
    response_format: str = Form("json"),
    prompt: str | None = Form(None),
) -> dict:
    audio, sr = sf.read(io.BytesIO(await file.read()), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:  # whisper wants 16 kHz mono
        n = int(len(audio) * 16000 / sr)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False), np.arange(len(audio)), audio)
        audio = audio.astype(np.float32)

    t0 = time.perf_counter()
    segments, info = whisper.transcribe(
        audio, language=language or None, beam_size=1, vad_filter=False, initial_prompt=prompt
    )
    text = "".join(s.text for s in segments).strip()
    log.info("stt %.0f ms  %.1fs audio  %r", (time.perf_counter() - t0) * 1000, len(audio) / 16000, text)
    out = {"text": text}
    if response_format == "verbose_json":
        out |= {"language": info.language, "duration": len(audio) / 16000}
    return out


class SpeechRequest(BaseModel):
    input: str
    model: str = "kokoro"
    voice: str | None = None
    response_format: str = "pcm"
    speed: float = 1.0
    instructions: str | None = None  # sent by gpt-4o-mini-tts clients; ignored


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> StreamingResponse:
    if req.response_format != "pcm":
        raise HTTPException(400, "only response_format=pcm (24 kHz s16le mono) is supported")
    voice = req.voice if req.voice and req.voice not in {"alloy", "ash"} else DEFAULT_VOICE

    def gen():
        t0 = time.perf_counter()
        first = True
        # Kokoro splits on sentences; each chunk is sent as soon as it is synthesized,
        # so time-to-first-audio is one sentence, not the whole reply.
        for result in kokoro(req.input, voice=voice, speed=req.speed):
            if result.audio is None:
                continue
            pcm = (result.audio.numpy().clip(-1, 1) * 32767).astype("<i2").tobytes()
            if first:
                log.info("tts first chunk %.0f ms  %r", (time.perf_counter() - t0) * 1000, req.input[:60])
                first = False
            yield pcm

    return StreamingResponse(gen(), media_type="audio/pcm")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8880")))
