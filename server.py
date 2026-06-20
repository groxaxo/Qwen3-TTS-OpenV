#!/usr/bin/env python3
"""
FastAPI server for Qwen3-TTS OpenVINO inference.

Wraps OVQwen3TTS engine with HTTP endpoints.
Default: Vivian voice on Intel iGPU, code predictor on CPU.

Usage:
    uv run server.py                        # defaults: GPU + Vivian
    uv run server.py --device CPU           # force CPU-only
    uv run server.py --port 8765            # custom port
    uv run server.py --ov-dir /path/to/model
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field

# Add src to path so ov_infer can import its siblings
SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "openvino"))

from ov_infer import (
    OVQwen3TTS,
    ModelLoadConfig,
    ModelType,
    CustomVoiceRequest,
    VoiceDesignRequest,
    SamplingParams,
    Speaker,
    Language,
    SPEAKERS,
    SPEECH_DECODER_SR,
    NUM_CODE_GROUPS,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_OV_DIR = os.environ.get(
    "OV_DIR", str(Path.home() / "models" / "customvoice-1.7b-int8-ov")
)
DEFAULT_DEVICE = os.environ.get("OV_DEVICE", "GPU")
DEFAULT_CP_DEVICE = os.environ.get("OV_CP_DEVICE", "CPU")
DEFAULT_SD_DEVICE = os.environ.get("OV_SD_DEVICE")  # None = follow --device
DEFAULT_PORT = int(os.environ.get("PORT", "8765"))

# Global engine singleton
engine: OVQwen3TTS | None = None
_load_config: ModelLoadConfig | None = None
_load_time: float = 0.0


# ---------------------------------------------------------------------------
# Lifespan — load model at startup, release on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine, _load_config, _load_time

    t0 = time.perf_counter()
    engine = OVQwen3TTS()
    _load_config = ModelLoadConfig(
        ov_dir=app.state.ov_dir,
        device=app.state.device,
        cp_device=app.state.cp_device,
        sd_device=app.state.sd_device,
        model_type=ModelType.CUSTOM_VOICE,
    )
    # Load in a thread so uvicorn internals stay responsive
    await asyncio.to_thread(engine.load_model, _load_config)
    _load_time = time.perf_counter() - t0
    print(f"[server] model ready in {_load_time:.1f}s")

    yield

    if engine and engine.loaded:
        await engine.unload_model()
    print("[server] shutdown complete")


app = FastAPI(
    title="Qwen3-TTS OpenVINO Server",
    description="Text-to-speech with Qwen3-TTS on Intel iGPU via OpenVINO",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class SynthesizeRequest(BaseModel):
    """Request body for speech synthesis."""
    text: str = Field(description="Text to synthesize")
    speaker: str = Field(default="vivian", description="Speaker name")
    language: str | None = Field(default=None, description="Language (null=auto-detect)")
    instruct: str | None = Field(default=None, description="Optional style instruction")
    temperature: float = Field(default=0.9, ge=0.01, le=2.0)
    top_k: int = Field(default=50, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.05, ge=1.0, le=2.0)
    max_new_tokens: int = Field(default=2048, ge=10, le=4096, description="Max codec frames. Model emits EOS naturally; 2048 is just a safety ceiling.")
    seed: int | None = Field(default=None, description="Random seed for reproducibility")


class PerfMetrics(BaseModel):
    """Performance metrics."""
    total_time_s: float
    audio_duration_s: float
    rtf: float
    num_frames: int
    tokens_per_sec: float


class SynthesizeJsonResponse(BaseModel):
    """JSON response with base64 audio and metrics."""
    audio_b64: str
    sample_rate: int
    metrics: PerfMetrics


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    if engine is None or not engine.loaded:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "detail": "Model not loaded"},
        )
    return {
        "status": "ready",
        "model_type": engine.model_type.value,
        "device": _load_config.device if _load_config else "unknown",
        "cp_device": _load_config.cp_device if _load_config else "unknown",
        "sd_device": _load_config.sd_device or _load_config.device if _load_config else "unknown",
        "load_time_s": round(_load_time, 2),
    }


@app.get("/speakers")
async def list_speakers():
    return {
        "speakers": [
            {"name": s.value, "codec_id": SPEAKERS[s].codec_id}
            for s in Speaker
        ],
        "default": "vivian",
    }


def _parse_request(req: SynthesizeRequest):
    """Validate and build a CustomVoiceRequest from the HTTP request."""
    try:
        speaker = Speaker(req.speaker.lower())
    except ValueError:
        raise HTTPException(
            400,
            f"Unknown speaker '{req.speaker}'. "
            f"Available: {', '.join(s.value for s in Speaker)}",
        )

    lang = None
    if req.language:
        try:
            lang = Language(req.language.lower())
        except ValueError:
            raise HTTPException(
                400,
                f"Unknown language '{req.language}'. "
                f"Available: auto, {', '.join(l.value for l in Language)}",
            )

    if req.seed is not None:
        np.random.seed(req.seed)

    sampling = SamplingParams(
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        max_new_tokens=req.max_new_tokens,
    )

    return CustomVoiceRequest(
        text=req.text,
        speaker=speaker,
        language=lang,
        instruct=req.instruct,
        sampling=sampling,
    )


def _compute_metrics(wav: np.ndarray, sr: int, t_total: float) -> PerfMetrics:
    audio_dur = len(wav) / sr
    rtf = t_total / audio_dur if audio_dur > 0 else float("inf")
    num_frames = max(1, int(len(wav) / (sr / 50)))
    tps = (num_frames * NUM_CODE_GROUPS) / t_total if t_total > 0 else 0.0
    return PerfMetrics(
        total_time_s=round(t_total, 3),
        audio_duration_s=round(audio_dur, 3),
        rtf=round(rtf, 3),
        num_frames=num_frames,
        tokens_per_sec=round(tps, 1),
    )


@app.post("/synthesize")
async def synthesize(req: SynthesizeRequest):
    """
    Synthesize speech → returns audio/wav.

    Performance metrics in response headers:
      X-RTF, X-Total-Time, X-Audio-Duration, X-Tokens-Per-Sec
    """
    if engine is None or not engine.loaded:
        raise HTTPException(503, "Model not loaded")

    request = _parse_request(req)

    t_start = time.perf_counter()
    wav, sr = await asyncio.to_thread(engine.generate, request)
    t_total = time.perf_counter() - t_start

    if wav.size == 0:
        raise HTTPException(500, "Generation produced empty audio")

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    m = _compute_metrics(wav, sr, t_total)

    print(
        f"[server] {len(req.text)} chars -> {m.audio_duration_s:.2f}s audio | "
        f"RTF={m.rtf:.3f} | {m.tokens_per_sec:.1f} tok/s | total={m.total_time_s:.2f}s"
    )

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-RTF": str(m.rtf),
            "X-Total-Time": str(m.total_time_s),
            "X-Audio-Duration": str(m.audio_duration_s),
            "X-Tokens-Per-Sec": str(m.tokens_per_sec),
        },
    )


@app.post("/synthesize/json")
async def synthesize_json(req: SynthesizeRequest):
    """Synthesize speech → returns JSON with base64 audio + metrics."""
    if engine is None or not engine.loaded:
        raise HTTPException(503, "Model not loaded")

    request = _parse_request(req)

    t_start = time.perf_counter()
    wav, sr = await asyncio.to_thread(engine.generate, request)
    t_total = time.perf_counter() - t_start

    if wav.size == 0:
        raise HTTPException(500, "Generation produced empty audio")

    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")

    return SynthesizeJsonResponse(
        audio_b64=base64.b64encode(buf.getvalue()).decode(),
        sample_rate=sr,
        metrics=_compute_metrics(wav, sr, t_total),
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="Qwen3-TTS OpenVINO FastAPI server")
    parser.add_argument("--ov-dir", default=DEFAULT_OV_DIR, help="OpenVINO model directory")
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Main OV device (default: GPU)")
    parser.add_argument("--cp-device", default=DEFAULT_CP_DEVICE, help="Code predictor device (default: CPU)")
    parser.add_argument("--sd-device", default=DEFAULT_SD_DEVICE, help="Speech decoder device (default: same as --device)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    app.state.ov_dir = args.ov_dir
    app.state.device = args.device
    app.state.cp_device = args.cp_device
    app.state.sd_device = args.sd_device

    print(f"[server] starting on {args.host}:{args.port}")
    print(f"[server] model: {args.ov_dir}")
    print(f"[server] device={args.device}  cp_device={args.cp_device}  sd_device={args.sd_device or args.device}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
