#!/usr/bin/env python3
# ruff: noqa: E402
"""
OpenVINO inference engine for Qwen3-TTS.

Single engine class supporting three request types:
  CustomVoiceRequest   — predefined speaker name + optional style instruction
  VoiceDesignRequest   — free-form voice description (no speaker)
  VoiceCloneRequest    — reference audio + optional transcript (ICL)

Usage (library):
    engine = OVQwen3TTS()
    engine.load_model(ModelLoadConfig(
        ov_dir="./ov_output", model_type=ModelType.VOICE_CLONE,
    ))

    wav, sr = engine.generate(CustomVoiceRequest(
        text="Hello world", speaker=Speaker.VIVIAN,
    ))

    wav, sr = engine.generate(VoiceCloneRequest(
        text="Hello world", ref_audio_path="./ref.wav", ref_text="Reference.",
    ))

    await engine.unload_model()

Usage (CLI):
    python ov_infer.py --mode custom_voice \\
        --ov-dir ./ov_output --text "Hello world" --speaker vivian --output hello.wav

    python ov_infer.py --mode voice_clone \\
        --ov-dir ./ov_output --text "Hello world" \\
        --ref-audio ./ref.wav --ref-text "Reference." --output hello.wav
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gc
import io
import sys
import time
import wave as wave_mod
from dataclasses import dataclass
from enum import StrEnum
import librosa
from pathlib import Path
from typing import Literal, Union

import numpy as np
import openvino as ov
import soundfile as sf
from pydantic import BaseModel, Field, model_validator
from transformers import AutoTokenizer

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# ---------------------------------------------------------------------------
# Constants — fixed for the Qwen3-TTS OpenVINO checkpoint family
# ---------------------------------------------------------------------------

# Special token IDs
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671
CODEC_BOS_ID = 2149
CODEC_EOS_ID = 2150
CODEC_PAD_ID = 2148
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

# Language and Speaker enums + registries
class Language(StrEnum):
    CHINESE = "chinese"
    ENGLISH = "english"
    GERMAN = "german"
    ITALIAN = "italian"
    PORTUGUESE = "portuguese"
    SPANISH = "spanish"
    JAPANESE = "japanese"
    KOREAN = "korean"
    FRENCH = "french"
    RUSSIAN = "russian"
    BEIJING_DIALECT = "beijing_dialect"
    SICHUAN_DIALECT = "sichuan_dialect"


@dataclass(frozen=True, slots=True)
class LanguageInfo:
    codec_id: int


LANGUAGES: dict[Language, LanguageInfo] = {
    Language.CHINESE:          LanguageInfo(codec_id=2055),
    Language.ENGLISH:          LanguageInfo(codec_id=2050),
    Language.GERMAN:           LanguageInfo(codec_id=2053),
    Language.ITALIAN:          LanguageInfo(codec_id=2070),
    Language.PORTUGUESE:       LanguageInfo(codec_id=2071),
    Language.SPANISH:          LanguageInfo(codec_id=2054),
    Language.JAPANESE:         LanguageInfo(codec_id=2058),
    Language.KOREAN:           LanguageInfo(codec_id=2064),
    Language.FRENCH:           LanguageInfo(codec_id=2061),
    Language.RUSSIAN:          LanguageInfo(codec_id=2069),
    Language.BEIJING_DIALECT:  LanguageInfo(codec_id=2074),
    Language.SICHUAN_DIALECT:  LanguageInfo(codec_id=2062),
}


class Speaker(StrEnum):
    SERENA = "serena"
    VIVIAN = "vivian"
    UNCLE_FU = "uncle_fu"
    RYAN = "ryan"
    AIDEN = "aiden"
    ONO_ANNA = "ono_anna"
    SOHEE = "sohee"
    ERIC = "eric"
    DYLAN = "dylan"


@dataclass(frozen=True, slots=True)
class SpeakerInfo:
    codec_id: int
    dialect: Language | None = None


SPEAKERS: dict[Speaker, SpeakerInfo] = {
    Speaker.SERENA:   SpeakerInfo(codec_id=3066),
    Speaker.VIVIAN:   SpeakerInfo(codec_id=3065),
    Speaker.UNCLE_FU: SpeakerInfo(codec_id=3010),
    Speaker.RYAN:     SpeakerInfo(codec_id=3061),
    Speaker.AIDEN:    SpeakerInfo(codec_id=2861),
    Speaker.ONO_ANNA: SpeakerInfo(codec_id=2873),
    Speaker.SOHEE:    SpeakerInfo(codec_id=2864),
    Speaker.ERIC:     SpeakerInfo(codec_id=2875, dialect=Language.SICHUAN_DIALECT),
    Speaker.DYLAN:    SpeakerInfo(codec_id=2878, dialect=Language.BEIJING_DIALECT),
}


class ModelType(StrEnum):
    CUSTOM_VOICE = "custom_voice"
    VOICE_DESIGN = "voice_design"
    VOICE_CLONE = "voice_clone"

# Talker architecture
NUM_CODE_GROUPS = 16
HIDDEN_SIZE = 2048
HEAD_DIM = 128
VOCAB_SIZE = 3072
TALKER_MAX_POS = 32768
TALKER_ROPE_THETA = 1_000_000.0
MROPE_SECTION = (24, 20, 20)

# Code predictor architecture
CP_HEAD_DIM = 128
CP_MAX_POS = 65536
CP_ROPE_THETA = 1_000_000.0

# Speech decoder
SPEECH_DECODER_SR = 24000

# Speaker encoder mel-spectrogram params
SE_SR = 24000
SE_N_FFT = 1024
SE_HOP = 256
SE_WIN = 1024
SE_N_MELS = 128
SE_FMIN = 0.0
SE_FMAX = 12000.0

# Speech encoder
ENC_INPUT_SR = 24000

# Prompt templates
_INSTRUCT_TMPL = "<|im_start|>user\n{instruct}<|im_end|>\n"
_SYNTH_TMPL = "<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
_REF_TEXT_TMPL = "<|im_start|>assistant\n{ref_text}<|im_end|>\n"

# Suppress mask: block last 1024 codec IDs except EOS
SUPPRESS_MASK = np.zeros(VOCAB_SIZE, dtype=bool)
for _i in range(VOCAB_SIZE - 1024, VOCAB_SIZE):
    if _i != CODEC_EOS_ID:
        SUPPRESS_MASK[_i] = True


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class ModelLoadConfig(BaseModel):
    """Configuration for loading Qwen3-TTS OV models."""

    ov_dir: str = Field(description="Path to the converted OpenVINO model directory.")
    device: str = Field(default="CPU", description="OpenVINO device (e.g. 'CPU', 'GPU').")
    cp_device: str | None = Field(default=None, description="Override device for code predictor (e.g. 'CPU' when GPU produces NaN).")
    cp_f32: bool = Field(default=False, description="Force f32 precision for code predictor on GPU (fixes NaN with small models).")
    model_type: ModelType = Field(description="Which model variant to load.")


class SamplingParams(BaseModel):
    """Controls generation behaviour for both talker and code predictor."""

    max_new_tokens: int = Field(default=2048, description="Maximum codec frames to generate.")
    do_sample: bool = Field(default=True, description="Sample from logits (False = greedy).")
    top_k: int = Field(default=50, description="Top-k filter for talker logits.")
    top_p: float = Field(default=1.0, description="Nucleus filter for talker logits (1.0 = off).")
    temperature: float = Field(default=0.9, description="Temperature scaling for talker logits.")
    repetition_penalty: float = Field(
        default=1.05, description="Repetition penalty on first-codebook history (1.0 = off)."
    )
    non_streaming_mode: bool = Field(
        default=True,
        description="True = all text tokens in prefill; False = drip-fed during decode.",
    )

    # Code predictor (sub-talker) sampler
    subtalker_do_sample: bool = Field(default=True, description="Sample sub-codebook logits.")
    subtalker_top_k: int = Field(default=50, description="Top-k for code predictor.")
    subtalker_top_p: float = Field(default=1.0, description="Nucleus filter for code predictor.")
    subtalker_temperature: float = Field(default=0.9, description="Temperature for code predictor.")


class CustomVoiceRequest(BaseModel):
    """Synthesise speech with a predefined speaker and optional style instruction."""

    mode: Literal[ModelType.CUSTOM_VOICE] = ModelType.CUSTOM_VOICE
    text: str
    speaker: Speaker
    language: Language | None = Field(default=None, description="None = auto-detect.")
    instruct: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)


class VoiceDesignRequest(BaseModel):
    """Synthesise speech shaped by a free-form voice description."""

    mode: Literal[ModelType.VOICE_DESIGN] = ModelType.VOICE_DESIGN
    text: str
    voice_description: str
    language: Language | None = Field(default=None, description="None = auto-detect.")
    sampling: SamplingParams = Field(default_factory=SamplingParams)


class VoiceCloneRequest(BaseModel):
    """Clone a voice from reference audio, optionally with ICL transcript."""

    mode: Literal[ModelType.VOICE_CLONE] = ModelType.VOICE_CLONE
    text: str
    ref_audio_path: str | None = Field(default=None, description="WAV file path.")
    ref_audio_b64: str | None = Field(default=None, description="Base64-encoded WAV bytes.")
    ref_text: str | None = Field(
        default=None, description="Reference audio transcript (enables ICL conditioning)."
    )
    x_vector_only: bool = Field(
        default=False, description="Use only x-vector embedding, skip ICL even with ref_text."
    )
    language: Language | None = Field(default=None, description="None = auto-detect.")
    instruct: str | None = None
    sampling: SamplingParams = Field(default_factory=SamplingParams)

    @model_validator(mode="after")
    def _require_audio_source(self):
        if not self.ref_audio_path and not self.ref_audio_b64:
            raise ValueError("Provide ref_audio_path or ref_audio_b64")
        return self


OVQwen3TTSRequest = Union[CustomVoiceRequest, VoiceDesignRequest, VoiceCloneRequest]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class OVQwen3TTSHelpers:
    """Static utility methods for sampling, RoPE, OV dispatch, and audio I/O."""

    # ---- Sampling -----------------------------------------------------------

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        x = x - np.max(x)
        e = np.exp(x)
        return e / e.sum()

    @staticmethod
    def sample_token(
        logits: np.ndarray,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
    ) -> int:
        logits = logits.copy().astype(np.float32)
        if not do_sample:
            return int(np.argmax(logits))
        if temperature != 1.0:
            logits /= temperature
        if top_k > 0:
            top_k = min(top_k, logits.shape[-1])
            threshold = np.partition(logits, -top_k)[-top_k]
            logits[logits < threshold] = -np.inf
        if top_p < 1.0:
            idx = np.argsort(logits)[::-1]
            sl = logits[idx]
            probs = OVQwen3TTSHelpers.softmax(sl)
            cutoff = np.searchsorted(np.cumsum(probs), top_p) + 1
            sl[cutoff:] = -np.inf
            logits[idx] = sl
        probs = OVQwen3TTSHelpers.softmax(logits)
        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def apply_repetition_penalty(
        logits: np.ndarray, past_tokens: list[int], penalty: float
    ) -> np.ndarray:
        for tid in set(past_tokens):
            if logits[tid] > 0:
                logits[tid] /= penalty
            else:
                logits[tid] *= penalty
        return logits

    # ---- RoPE ---------------------------------------------------------------

    @staticmethod
    def precompute_mrope(max_len: int, head_dim: int, theta: float = TALKER_ROPE_THETA):
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(pos, inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    @staticmethod
    def precompute_standard_rope(max_len: int, head_dim: int, theta: float = 10_000.0):
        inv = 1.0 / (theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
        pos = np.arange(max_len, dtype=np.float32)
        freqs = np.outer(pos, inv)
        emb = np.concatenate([freqs, freqs], axis=-1)
        return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)

    @staticmethod
    def slice_rope(cos, sin, start: int, length: int):
        c = cos[start : start + length][np.newaxis, np.newaxis]
        s = sin[start : start + length][np.newaxis, np.newaxis]
        return c, s

    # ---- OV dispatch --------------------------------------------------------

    @staticmethod
    def ov_call(compiled_model, inputs: dict) -> dict:
        result = compiled_model(inputs)
        return {
            out.get_any_name(): result[out] for out in compiled_model.outputs
        }

    @staticmethod
    def ov_stateful_infer(request, inputs: dict) -> dict:
        request.infer(inputs)
        return {
            out.get_any_name(): request.get_tensor(out.get_any_name()).data.copy()
            for out in request.model_outputs
        }

    # ---- Audio I/O ----------------------------------------------------------

    @staticmethod
    def load_audio_wav(path: str) -> tuple[np.ndarray, int]:
        with wave_mod.open(path, "r") as wf:
            n_ch = wf.getnchannels()
            sw = wf.getsampwidth()
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        if sw == 2:
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sw == 4:
            samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
        elif sw == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
        else:
            raise ValueError(f"Unsupported sample width: {sw}")
        if n_ch > 1:
            samples = samples.reshape(-1, n_ch).mean(axis=1)
        return samples, sr

    @staticmethod
    def decode_audio_b64(b64: str) -> tuple[np.ndarray, int]:
        data, sr = sf.read(io.BytesIO(base64.b64decode(b64)), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr

    @staticmethod
    def mel_spectrogram(
        audio: np.ndarray,
        sr: int,
        target_sr: int = SE_SR,
        n_fft: int = SE_N_FFT,
        hop_size: int = SE_HOP,
        win_size: int = SE_WIN,
        n_mels: int = SE_N_MELS,
        fmin: float = SE_FMIN,
        fmax: float = SE_FMAX,
    ) -> np.ndarray:
        """Log-mel spectrogram -> (n_mels, T) float32."""

        audio = audio.astype(np.float32)
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        pad = (n_fft - hop_size) // 2
        audio = np.pad(audio, (pad, pad), mode="reflect")
        stft = librosa.stft(
            audio, n_fft=n_fft, hop_length=hop_size, win_length=win_size,
            window="hann", center=False,
        )
        mag = np.sqrt(stft.real ** 2 + stft.imag ** 2 + 1e-9).astype(np.float32)
        mel_basis = librosa.filters.mel(
            sr=target_sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax,
        ).astype(np.float32)
        return np.log(np.clip(mel_basis @ mag, 1e-5, None)).astype(np.float32)


H = OVQwen3TTSHelpers  # short alias used inside the engine


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OVQwen3TTS:
    """Single engine that serves all three Qwen3-TTS modes.

    Lifecycle:
        engine = OVQwen3TTS()
        engine.load_model(ModelLoadConfig(
            ov_dir="./ov_output", model_type=ModelType.VOICE_CLONE,
        ))
        wav, sr = engine.generate(request)
        await engine.unload_model()
    """

    # Maps ModelType → set of request types it can serve
    _SUPPORTED_REQUESTS: dict[ModelType, tuple[type, ...]] = {
        ModelType.CUSTOM_VOICE: (CustomVoiceRequest,),
        ModelType.VOICE_DESIGN: (VoiceDesignRequest,),
        ModelType.VOICE_CLONE: (VoiceCloneRequest,),
    }

    def __init__(self):
        self._loaded = False
        self._model_type: ModelType | None = None

    # ---- Lifecycle ----------------------------------------------------------

    def load_model(self, config: ModelLoadConfig):
        """Load OV models according to *config*.

        Core models (text_model, codec_embedding, talker, code_predictor,
        speech_decoder) are loaded for every model type.  Voice-clone models
        (speaker_encoder, speech_encoder) are loaded only when
        ``config.model_type == ModelType.VOICE_CLONE``.
        """
        if self._loaded:
            self._release_models()

        p = Path(config.ov_dir)
        device = config.device
        core = ov.Core()

        self.tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)

        # RoPE tables
        self._mrope_cos, self._mrope_sin = H.precompute_mrope(
            TALKER_MAX_POS, HEAD_DIM, TALKER_ROPE_THETA,
        )
        self._cp_cos, self._cp_sin = H.precompute_standard_rope(
            CP_MAX_POS, CP_HEAD_DIM, CP_ROPE_THETA,
        )

        # Core models (all modes)
        self._text_model_c = core.compile_model(str(p / "text_model.xml"), device)
        self._codec_emb_c = core.compile_model(str(p / "codec_embedding.xml"), device)
        self._cp_codec_emb_c = core.compile_model(str(p / "cp_codec_embedding.xml"), device)
        self._decoder_c = core.compile_model(
            str(p / "speech_tokenizer" / "speech_decoder.xml"), device,
        )
        self._decoder_input_name = self._decoder_c.input(0).get_any_name()

        talker_c = core.compile_model(str(p / "talker.xml"), device)
        self._talker_req = talker_c.create_infer_request()
        cp_device = config.cp_device or device
        cp_props = {"INFERENCE_PRECISION_HINT": "f32"} if config.cp_f32 else {}
        cp_c = core.compile_model(str(p / "code_predictor.xml"), cp_device, cp_props)
        self._cp_req = cp_c.create_infer_request()

        # Voice-clone models (only when needed)
        self._speaker_enc_c = None
        self._speech_enc_c = None
        if config.model_type == ModelType.VOICE_CLONE:
            self._speaker_enc_c = core.compile_model(
                str(p / "speaker_encoder.xml"), device,
            )
            self._speech_enc_c = core.compile_model(
                str(p / "speech_tokenizer" / "speech_encoder.xml"), device,
            )

        self._model_type = config.model_type
        self._loaded = True
        print(
            f"[engine] loaded from {p}  device={device}  "
            f"model_type={config.model_type.value}"
        )

    async def unload_model(self):
        """Release all compiled models and force garbage collection off the event loop."""
        await asyncio.to_thread(self._release_and_gc)

    def _release_models(self):
        """Drop references to all OV models and tokenizer."""
        for a in (
            "_text_model_c", "_codec_emb_c", "_cp_codec_emb_c", "_decoder_c",
            "_talker_req", "_cp_req", "_speaker_enc_c", "_speech_enc_c",
            "tokenizer",
        ):
            if hasattr(self, a):
                delattr(self, a)
        self._model_type = None
        self._loaded = False

    def _release_and_gc(self):
        """Release models then run a full gc pass (called via to_thread)."""
        self._release_models()
        gc.collect()
        print("[engine] unloaded")

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    def model_type(self) -> ModelType | None:
        return self._model_type

    # ---- Public API ---------------------------------------------------------

    def generate(self, request: OVQwen3TTSRequest) -> tuple[np.ndarray, int]:
        """Dispatch to the appropriate generation path.

        Validates that *request* is compatible with the loaded model type.

        Returns:
            (wav: float32 ndarray, sample_rate: int)
        """
        if not self._loaded:
            raise RuntimeError("Call load_model() before generate()")

        supported = self._SUPPORTED_REQUESTS[self._model_type]
        if not isinstance(request, supported):
            raise TypeError(
                f"Model loaded as {self._model_type.value} — cannot serve "
                f"{type(request).__name__}. Expected one of: "
                f"{', '.join(t.__name__ for t in supported)}"
            )

        if isinstance(request, VoiceCloneRequest):
            return self._generate_voice_clone(request)
        return self._generate_standard(request)

    # ---- Internal: standard generation (custom_voice / voice_design) --------

    def _generate_standard(
        self, request: CustomVoiceRequest | VoiceDesignRequest,
    ) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        sp = request.sampling

        if isinstance(request, CustomVoiceRequest):
            build_kw = dict(
                text=request.text,
                speaker=request.speaker,
                language=request.language,
                instruct=request.instruct,
            )
        else:  # VoiceDesignRequest
            build_kw = dict(
                text=request.text,
                speaker=None,
                language=request.language,
                instruct=request.voice_description,
            )

        t0 = time.perf_counter()
        inp = self._build_inputs(**build_kw, non_streaming_mode=sp.non_streaming_mode)
        print(f"[perf] build_inputs:         {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, sp)

        if not codes:
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        t0 = time.perf_counter()
        wav = self._decode_codes(codes)
        print(f"[perf] speech decoder (OV):  {time.perf_counter() - t0:.3f}s")

        self._log_summary(codes, wav, t_total)
        return wav, SPEECH_DECODER_SR

    # ---- Internal: voice clone generation -----------------------------------

    def _generate_voice_clone(self, request: VoiceCloneRequest) -> tuple[np.ndarray, int]:
        t_total = time.perf_counter()
        sp = request.sampling

        # Load reference audio
        if request.ref_audio_path:
            audio, audio_sr = H.load_audio_wav(request.ref_audio_path)
        else:
            audio, audio_sr = H.decode_audio_b64(request.ref_audio_b64)

        # Speaker embedding (ECAPA-TDNN)
        t0 = time.perf_counter()
        speaker_embed = self._extract_speaker_embedding(audio, audio_sr)
        print(f"[perf] speaker encoder:      {time.perf_counter() - t0:.3f}s")

        # ICL reference codes
        use_icl = request.ref_text is not None and not request.x_vector_only
        ref_codes = None
        if use_icl:
            t0 = time.perf_counter()
            ref_codes = self._encode_audio(audio, audio_sr)
            print(f"[perf] speech encoder (OV):  {time.perf_counter() - t0:.3f}s")
            print(f"[info] ref_codes shape:      {ref_codes.shape}")

        t0 = time.perf_counter()
        inp = self._build_inputs(
            text=request.text,
            speaker_embed=speaker_embed,
            language=request.language,
            instruct=request.instruct,
            non_streaming_mode=sp.non_streaming_mode,
            ref_text=request.ref_text if use_icl else None,
            ref_codes=ref_codes,
        )
        print(f"[perf] build_inputs:         {time.perf_counter() - t0:.3f}s")

        codes = self._run_loop(inp, sp)

        if not codes:
            return np.zeros(0, dtype=np.float32), SPEECH_DECODER_SR

        t0 = time.perf_counter()
        if use_icl and ref_codes is not None:
            wav = self._decode_icl(codes, ref_codes)
        else:
            wav = self._decode_codes(codes)
        print(f"[perf] speech decoder (OV):  {time.perf_counter() - t0:.3f}s")

        self._log_summary(codes, wav, t_total)
        return wav, SPEECH_DECODER_SR

    def _decode_icl(
        self,
        gen_codes: list[list[int]],
        ref_codes: np.ndarray,
    ) -> np.ndarray:
        """Decode with reference prefix, then trim the ref portion from output."""
        ref_2d = ref_codes[0]  # (T_ref, n_q)
        gen_2d = np.asarray(gen_codes, dtype=np.int64)
        combined = np.concatenate([ref_2d, gen_2d], axis=0)
        decoder_in = combined.T[np.newaxis]  # (1, n_q, T)
        result = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        full_wav = np.clip(result["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)

        cut = int(ref_2d.shape[0] / combined.shape[0] * len(full_wav))
        return full_wav[cut:]

    # ---- OV model wrappers --------------------------------------------------

    def _text_model(self, ids: np.ndarray) -> np.ndarray:
        return H.ov_call(self._text_model_c, {"token_ids": ids})["projected"]

    def _codec_embed(self, ids: np.ndarray) -> np.ndarray:
        return H.ov_call(self._codec_emb_c, {"token_ids": ids})["embeddings"]

    def _cp_codec_embed(self, ids: np.ndarray, step_idx: int) -> np.ndarray:
        return H.ov_call(self._cp_codec_emb_c, {
            "token_ids": ids,
            "step_idx": np.array(step_idx, dtype=np.int64),
        })["embeddings"]

    def _talker_infer(self, embeds, cos, sin):
        r = H.ov_stateful_infer(self._talker_req, {
            "inputs_embeds": embeds, "cos": cos, "sin": sin,
            "beam_idx": np.array([0], dtype=np.int32),
        })
        return r["logits"], r["hidden"]

    def _cp_infer(self, embeds, cos, sin, gen_steps: int):
        r = H.ov_stateful_infer(self._cp_req, {
            "inputs_embeds": embeds, "cos": cos, "sin": sin,
            "generation_steps": np.array(gen_steps, dtype=np.int64),
            "beam_idx": np.array([0], dtype=np.int32),
        })
        return r["logits"], r["hidden"]

    def _decode_codes(self, codes: list[list[int]]) -> np.ndarray:
        arr = np.asarray(codes, dtype=np.int64)
        decoder_in = arr.T[np.newaxis]
        r = H.ov_call(self._decoder_c, {self._decoder_input_name: decoder_in})
        return np.clip(r["waveform"].squeeze(), -1.0, 1.0).astype(np.float32)

    # ---- Voice-clone specific OV calls --------------------------------------

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        mels = H.mel_spectrogram(audio, sr)  # (n_mels, T)
        mels_in = mels.T[np.newaxis].astype(np.float32)  # (1, T, n_mels)
        r = H.ov_call(self._speaker_enc_c, {"mels": mels_in})
        return r["embedding"][:, np.newaxis, :]  # (1, 1, D)

    def _encode_audio(self, audio: np.ndarray, sr: int) -> np.ndarray:
        audio = audio.astype(np.float32)
        if sr != ENC_INPUT_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=ENC_INPUT_SR)
        r = H.ov_call(self._speech_enc_c, {"audio": audio[np.newaxis]})
        return r["codes"]  # (1, T_ref, n_q)

    # ---- Prefill assembly ---------------------------------------------------

    def _get_special_embeds(self):
        ids = np.array([[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]], dtype=np.int64)
        e = self._text_model(ids)
        return e[:, 0:1, :], e[:, 1:2, :], e[:, 2:3, :]

    def _resolve_language_id(self, language: Language | None, speaker: Speaker | None) -> int | None:
        lang_id = LANGUAGES[language].codec_id if language is not None else None

        if language in (Language.CHINESE, None) and speaker is not None:
            dialect = SPEAKERS[speaker].dialect
            if dialect is not None:
                lang_id = LANGUAGES[dialect].codec_id
        return lang_id

    def _build_codec_control(
        self,
        language_id: int | None,
        speaker_embed: np.ndarray | None = None,
        speaker: Speaker | None = None,
    ) -> np.ndarray:
        if language_id is None:
            prefix_ids = np.array(
                [[CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]], dtype=np.int64,
            )
        else:
            prefix_ids = np.array(
                [[CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID]],
                dtype=np.int64,
            )

        emb_prefix = self._codec_embed(prefix_ids)
        emb_suffix = self._codec_embed(
            np.array([[CODEC_PAD_ID, CODEC_BOS_ID]], dtype=np.int64),
        )

        spk = None
        if speaker_embed is not None:
            spk = speaker_embed
        elif speaker is not None:
            spk = self._codec_embed(
                np.array([[SPEAKERS[speaker].codec_id]], dtype=np.int64),
            )

        parts = [emb_prefix] + ([spk] if spk is not None else []) + [emb_suffix]
        return np.concatenate(parts, axis=1)

    def _build_inputs(
        self,
        text: str,
        speaker: Speaker | None = None,
        speaker_embed: np.ndarray | None = None,
        language: Language | None = None,
        instruct: str | None = None,
        non_streaming_mode: bool = True,
        ref_text: str | None = None,
        ref_codes: np.ndarray | None = None,
    ) -> dict:
        formatted = _SYNTH_TMPL.format(text=text)
        input_ids = self.tokenizer(formatted, return_tensors="np", padding=False)["input_ids"]

        tts_bos, tts_eos, tts_pad = self._get_special_embeds()
        lang_id = self._resolve_language_id(language, speaker)
        codec_ctrl = self._build_codec_control(lang_id, speaker_embed, speaker)

        # Role prefix:  <|im_start|>assistant\n  (first 3 tokens)
        role = self._text_model(input_ids[:, :3])

        # Control signal: text-side padding + bos  summed with codec-side embeddings
        n_codec = codec_ctrl.shape[1]
        text_side = np.concatenate(
            [np.tile(tts_pad, (1, n_codec - 2, 1)), tts_bos], axis=1,
        )
        control = text_side + codec_ctrl[:, :-1, :]
        talker = np.concatenate([role, control], axis=1)

        # Prepend instruct if provided
        if instruct:
            inst_ids = self.tokenizer(
                _INSTRUCT_TMPL.format(instruct=instruct), return_tensors="np", padding=False,
            )["input_ids"]
            talker = np.concatenate([self._text_model(inst_ids), talker], axis=1)

        # Text token assembly — three branches
        use_icl = ref_codes is not None and ref_text is not None

        if use_icl:
            ref_ids = self.tokenizer(
                _REF_TEXT_TMPL.format(ref_text=ref_text), return_tensors="np", padding=False,
            )["input_ids"]
            ref_text_ids = ref_ids[:, 3:-2]
            target_ids = input_ids[:, 3:-5]
            all_text_ids = np.concatenate([ref_text_ids, target_ids], axis=1)

            text_emb = self._text_model(all_text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)

            codec_bos_emb = self._codec_embed(np.array([[CODEC_BOS_ID]], dtype=np.int64))
            ref_emb = self._embed_ref_codes(ref_codes[0])
            codec_bos_ref = np.concatenate([codec_bos_emb, ref_emb], axis=1)

            text_block = text_eos + self._codec_embed(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            codec_block = codec_bos_ref + np.tile(tts_pad, (1, codec_bos_ref.shape[1], 1))

            final_bos = tts_pad + self._codec_embed(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_block, codec_block, final_bos], axis=1)
            trailing = tts_pad

        elif non_streaming_mode:
            text_ids = input_ids[:, 3:-5]
            text_emb = self._text_model(text_ids)
            text_eos = np.concatenate([text_emb, tts_eos], axis=1)
            codec_pad_seq = self._codec_embed(
                np.full((1, text_eos.shape[1]), CODEC_PAD_ID, dtype=np.int64),
            )
            final_bos = tts_pad + self._codec_embed(
                np.array([[CODEC_BOS_ID]], dtype=np.int64),
            )
            talker = np.concatenate([talker, text_eos + codec_pad_seq, final_bos], axis=1)
            trailing = tts_pad

        else:
            first = self._text_model(input_ids[:, 3:4])
            talker = np.concatenate([talker, first + codec_ctrl[:, -1:, :]], axis=1)
            remaining = self._text_model(input_ids[:, 4:-5])
            trailing = np.concatenate([remaining, tts_eos], axis=1)

        return {"inputs_embeds": talker, "trailing_text_hidden": trailing, "tts_pad_embed": tts_pad}

    def _embed_ref_codes(self, codes_2d: np.ndarray) -> np.ndarray:
        T = codes_2d.shape[0]
        result = self._codec_embed(codes_2d[:, 0].reshape(1, T).astype(np.int64))
        for i in range(1, codes_2d.shape[1]):
            result = result + self._cp_codec_embed(
                codes_2d[:, i].reshape(1, T).astype(np.int64), step_idx=i - 1,
            )
        return result

    # ---- Sub-code generation ------------------------------------------------

    def _generate_sub_codes(
        self,
        first_code_embed: np.ndarray,
        past_hidden: np.ndarray,
        sp: SamplingParams,
    ) -> tuple[list[int], np.ndarray]:
        num_sub = NUM_CODE_GROUPS - 1
        self._cp_req.reset_state()

        prefill = np.concatenate([past_hidden, first_code_embed], axis=1)
        cos, sin = H.slice_rope(self._cp_cos, self._cp_sin, 0, 2)

        logits, _ = self._cp_infer(prefill, cos, sin, gen_steps=0)

        tid = H.sample_token(
            logits[0, -1, :],
            sp.subtalker_do_sample, sp.subtalker_top_k,
            sp.subtalker_top_p, sp.subtalker_temperature,
        )
        sub_codes = [tid]

        code_emb = self._cp_codec_embed(np.array([[tid]], dtype=np.int64), step_idx=0)
        embeds_sum = first_code_embed + code_emb
        cache_pos = 2

        for step in range(1, num_sub):
            cos, sin = H.slice_rope(self._cp_cos, self._cp_sin, cache_pos, 1)
            logits, _ = self._cp_infer(code_emb, cos, sin, gen_steps=step)

            tid = H.sample_token(
                logits[0, -1, :],
                sp.subtalker_do_sample, sp.subtalker_top_k,
                sp.subtalker_top_p, sp.subtalker_temperature,
            )
            sub_codes.append(tid)

            code_emb = self._cp_codec_embed(np.array([[tid]], dtype=np.int64), step_idx=step)
            embeds_sum = embeds_sum + code_emb
            cache_pos += 1

        return sub_codes, embeds_sum

    # ---- Core generation loop -----------------------------------------------

    def _run_loop(
        self, inp: dict, sp: SamplingParams,
    ) -> list[list[int]]:
        """Run the autoregressive talker + code-predictor loop.

        Returns:
            List of codec frame codes (each frame is a list of NUM_CODE_GROUPS ints).
        """
        embeds = inp["inputs_embeds"]
        trailing = inp["trailing_text_hidden"]
        pad_emb = inp["tts_pad_embed"]

        self._talker_req.reset_state()
        S = embeds.shape[1]
        cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, 0, S)

        t0 = time.perf_counter()
        logits, hidden = self._talker_infer(embeds, cos, sin)
        print(f"[perf] talker prefill ({S}t): {time.perf_counter() - t0:.3f}s")

        cache_pos = S
        first_logits = logits[0, -1, :].copy()
        first_logits[SUPPRESS_MASK] = -np.inf
        first_code = H.sample_token(
            first_logits, sp.do_sample, sp.top_k, sp.top_p, sp.temperature,
        )

        all_codes: list[list[int]] = []
        past_first: list[int] = []
        past_hidden = hidden[:, -1:, :]
        t_cp = t_talk = 0.0

        step = 0
        while step < sp.max_new_tokens:
            if first_code == CODEC_EOS_ID:
                break

            past_first.append(first_code)
            fc_emb = self._codec_embed(np.array([[first_code]], dtype=np.int64))

            t0 = time.perf_counter()
            subs, emb_sum = self._generate_sub_codes(fc_emb, past_hidden, sp)
            t_cp += time.perf_counter() - t0

            all_codes.append([first_code] + subs)

            next_emb = emb_sum
            if step < trailing.shape[1]:
                next_emb = next_emb + trailing[:, step : step + 1, :]
            else:
                next_emb = next_emb + pad_emb

            cos, sin = H.slice_rope(self._mrope_cos, self._mrope_sin, cache_pos, 1)
            t0 = time.perf_counter()
            logits, hidden = self._talker_infer(next_emb, cos, sin)
            t_talk += time.perf_counter() - t0

            cache_pos += 1
            step += 1

            sl = logits[0, -1, :].copy()
            sl[SUPPRESS_MASK] = -np.inf
            if sp.repetition_penalty != 1.0 and past_first:
                sl = H.apply_repetition_penalty(sl, past_first, sp.repetition_penalty)
            first_code = H.sample_token(sl, sp.do_sample, sp.top_k, sp.top_p, sp.temperature)
            past_hidden = hidden[:, -1:, :]

        n = step
        if n > 0:
            dt = t_cp + t_talk
            pf = dt / n
            print(f"[perf] decode loop ({n} frames):")
            print(f"[perf]   code predictor:  total={t_cp:.3f}s  avg={t_cp/n:.3f}s")
            print(f"[perf]   talker decode:   total={t_talk:.3f}s  avg={t_talk/n:.3f}s")
            print(f"[perf]   per frame:       {pf:.3f}s  ({1/pf:.1f} fps)")
            print(f"[perf]   throughput:      {n * NUM_CODE_GROUPS / dt:.1f} tokens/s")

        return all_codes

    # ---- Logging ------------------------------------------------------------

    @staticmethod
    def _log_summary(codes: list, wav: np.ndarray, t_total_start: float):
        sr = SPEECH_DECODER_SR
        print(f"[perf] total:                {time.perf_counter() - t_total_start:.3f}s")
        print(f"[info] {len(codes)} frames -> {len(wav)} samples ({len(wav)/sr:.2f}s audio)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_language(value: str) -> Language | None:
    """Convert CLI language string to Language enum or None (auto)."""
    if value.lower() == "auto":
        return None
    try:
        return Language(value.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Unknown language: {value!r}. "
            f"Choose from: auto, {', '.join(m.value for m in Language)}"
        )


def _parse_speaker(value: str) -> Speaker:
    """Convert CLI speaker string to Speaker enum."""
    try:
        return Speaker(value.lower())
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Unknown speaker: {value!r}. "
            f"Choose from: {', '.join(m.value for m in Speaker)}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS OpenVINO inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
mode usage:

  custom_voice:
    args: --text, --speaker (required), [--language], [--instruct]
    example:
      python ov_infer.py --mode custom_voice \\
        --ov-dir ./ov_output --text "Hello world" --speaker vivian --output hello.wav

  voice_design:
    args: --text, --voice-description (required), [--language]
    example:
      python ov_infer.py --mode voice_design \\
        --ov-dir ./ov_output --text "Hello world" \\
        --voice-description "Warm female narration" --output hello.wav

  voice_clone:
    args: --text, --ref-audio (required), [--ref-text], [--x-vector-only], [--language]
    example:
      python ov_infer.py --mode voice_clone \\
        --ov-dir ./ov_output --text "Hello world" \\
        --ref-audio ./ref.wav --ref-text "Reference text." --output hello.wav
""",
    )
    parser.add_argument("--ov-dir", required=True, help="Path to converted OV model directory")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in ModelType],
        required=True,
    )
    parser.add_argument("--text", required=True, help="Text to synthesise")
    parser.add_argument("--output", default="output.wav", help="Output WAV path")
    parser.add_argument("--device", default="CPU", help="OpenVINO device")
    parser.add_argument("--cp-device", default=None, help="Override device for code predictor (e.g. CPU when GPU produces NaN)")
    parser.add_argument("--cp-f32", action="store_true", help="Force f32 precision for code predictor on GPU (fixes NaN with small models)")

    # custom_voice
    cv = parser.add_argument_group("custom_voice mode")
    cv.add_argument("--speaker", type=_parse_speaker, default=None, help="Speaker name")
    cv.add_argument("--instruct", default=None, help="Optional style instruction")

    # voice_design
    vd = parser.add_argument_group("voice_design mode")
    vd.add_argument("--voice-description", default=None, help="Voice description / instruction")

    # voice_clone
    vc = parser.add_argument_group("voice_clone mode")
    vc.add_argument("--ref-audio", default=None, help="Reference audio WAV path")
    vc.add_argument("--ref-text", default=None, help="Reference audio transcript (enables ICL)")
    vc.add_argument(
        "--x-vector-only", action="store_true",
        help="Use only x-vector, skip ICL even with --ref-text",
    )

    # Shared generation
    parser.add_argument(
        "--language", type=_parse_language, default=None,
        help="Language (default: auto). Options: auto, "
             + ", ".join(m.value for m in Language),
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-sample", action="store_true", help="Greedy decoding")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)

    model_type = ModelType(args.mode)

    # Build sampling params
    sp = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Build request
    if model_type == ModelType.CUSTOM_VOICE:
        if not args.speaker:
            parser.error("--speaker is required for custom_voice mode")
        request = CustomVoiceRequest(
            text=args.text, speaker=args.speaker,
            language=args.language, instruct=args.instruct, sampling=sp,
        )
    elif model_type == ModelType.VOICE_DESIGN:
        if not args.voice_description:
            parser.error("--voice-description is required for voice_design mode")
        request = VoiceDesignRequest(
            text=args.text, voice_description=args.voice_description,
            language=args.language, sampling=sp,
        )
    elif model_type == ModelType.VOICE_CLONE:
        if not args.ref_audio:
            parser.error("--ref-audio is required for voice_clone mode")
        request = VoiceCloneRequest(
            text=args.text, ref_audio_path=args.ref_audio,
            ref_text=args.ref_text, x_vector_only=args.x_vector_only,
            language=args.language, instruct=args.instruct, sampling=sp,
        )
    else:
        parser.error(f"Unknown mode: {args.mode}")

    # Run
    engine = OVQwen3TTS()
    engine.load_model(ModelLoadConfig(
        ov_dir=args.ov_dir, device=args.device, cp_device=args.cp_device,
        cp_f32=args.cp_f32, model_type=model_type,
    ))
    wav, sr = engine.generate(request)
    asyncio.run(engine.unload_model())

    sf.write(args.output, wav, sr)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()