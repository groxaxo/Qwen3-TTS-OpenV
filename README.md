# Qwen3-TTS OpenVINO

> **OpenVINO inference for [Qwen3-TTS](https://arxiv.org/abs/2601.15621)** —
> built without `transformers`, runs on Intel CPUs / GPUs / NPUs.

---

## 🙏 Credits

This project is a fork of
**[SearchSavior/Qwen3-TTS-OpenV](https://github.com/SearchSavior/Qwen3-TTS-OpenV)**
by **Emerson Tatelbaum** ([@SearchSavior](https://github.com/SearchSavior)).
The original work — including the PyTorch → OpenVINO IR export pipeline, the
`nn.Module`-based refactor of Qwen3-TTS, and the three-task reference
implementations — is what made this possible.

Emerson's goal was to support Qwen3-TTS in
[OpenArc](https://github.com/SearchSavior/OpenArc). This fork adds a FastAPI
HTTP server and the **`sd_device` override** that makes the speech decoder
work correctly on Intel Iris Xe iGPUs (see [Fix story](#-fix-story-sd_device-on-x30-w)
below).

Paper:
> Hu et al., *Qwen3-TTS Technical Report*, arXiv:2601.15621, 2026.
> [arxiv.org/abs/2601.15621](https://arxiv.org/abs/2601.15621)

---

## ✨ What's in this fork

| Feature | Source | Notes |
|---|---|---|
| PyTorch → OpenVINO IR export (all three tasks) | original | INT8 / FP16 / FP32, weight-format selectable per submodule |
| `nn.Module`-based reference implementation | original | tracing-friendly, no `transformers` |
| OpenVINO inference engine (`ov_infer.py`) | original | custom-voice, voice-design, voice-clone |
| **FastAPI HTTP server (`server.py`)** | **this fork** | `POST /synthesize`, `POST /synthesize/json`, `GET /health`, `GET /speakers` |
| **`sd_device` override** | **this fork** | speech decoder can run on a different OpenVINO device than the talker |

---

## 🚀 Quick start

### 1. Export a Qwen3-TTS checkpoint to OpenVINO IR

```bash
uv sync
uv run src/openvino/ov_convert.py \
  --model-path ./Qwen3-TTS-1.7B \
  --new ./customvoice-1.7b-int8-ov \
  --model-type custom_voice \
  --weight-format int8
```

### 2. Run the FastAPI server

```bash
uv run server.py \
  --host 0.0.0.0 --port 8765 \
  --device GPU --cp-device CPU --sd-device CPU \
  --ov-dir ./customvoice-1.7b-int8-ov
```

`--sd-device CPU` is **required on x30w-k (Iris Xe iGPU)** — see the fix story
below. On other Intel hardware the default (`= --device`) usually works.

### 3. Synthesize speech

```bash
curl -X POST http://localhost:8765/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hola Facu, soy Qwen3 TTS.","speaker":"vivian","language":"spanish"}' \
  -o out.wav
```

Or use the CLI mode for one-off renders (no server):

```bash
uv run src/openvino/ov_infer.py \
  --ov-dir ./customvoice-1.7b-int8-ov \
  --mode custom_voice \
  --text "Hola Facu" \
  --speaker vivian \
  --output hola.wav
```

### 4. Health check

```bash
curl http://localhost:8765/health
# → {"status":"ready","model_type":"custom_voice","device":"GPU","cp_device":"CPU","sd_device":"CPU",...}
```

---

## 🔧 Fix story: `sd_device` on x30w-k

The original repo's `ov_infer.py` exposed only `--device` (talker) and
`--cp-device` (code predictor). The third component, the **speech decoder**
(vocoder), inherited `--device`.

On **x30w-k** (Intel i5-1240P with Iris Xe iGPU, OpenVINO INT8), compiling the
speech decoder on GPU produces **broken audio**: RMS ~0.015, peak ~0.08,
near-flat temporal modulation. The same codec codes, decoded on CPU, produce
clean audio (RMS ~0.105, peak ~0.70) — and CPU is also **15× faster** than
the iGPU path (1.5 s vs 23.4 s for the same input).

Diagnosed via same-codes re-decode experiment on 2026-06-20:

| Decoder device | RMS | peak | mod_var | time | STT works? |
|---|---|---|---|---|---|
| GPU (default) | 0.0156 | 0.0795 | 0.0009 | 23.4 s | ❌ empty |
| **CPU (correct)** | **0.1052** | **0.6988** | **0.0718** | **1.5 s** | **✅** |

The fix:

1. New `sd_device` field in `ModelLoadConfig` (default = same as `device`).
2. New CLI flag `--sd-device`.
3. New env var `OV_SD_DEVICE` for the FastAPI server.
4. `/health` reports `sd_device` so you can verify the configuration.

Total RTF dropped from ~6.9 to ~2.1 with this fix on x30w-k.

---

## 🧬 Project structure

```text
.
├── README.md                              ← you are here
├── pyproject.toml                         ← deps (fastapi, uvicorn, openvino, librosa, …)
├── server.py                              ← FastAPI HTTP server (this fork)
│
├── src/
│   ├── openvino/
│   │   ├── ov_convert.py                  ← PyTorch → OpenVINO IR exporter
│   │   └── ov_infer.py                    ← OpenVINO inference engine
│   │
│   ├── torch_functional/                  ← tensor-only reference impl
│   ├── torch_modules/                     ← nn.Module-based impl (for tracing)
│   └── test_modules.py                    ← inference script for module-based stack
│
├── openivno_notebook_reference/           ← upstream OpenVINO notebook helper, kept for context
├── qwen3_tts_transformers_reference/      ← upstream transformers impl, kept for context
│
├── CLAUDE_convert_from_functional_to_module.md
└── snip_snip.py                           ← dev scratch
```

---

## 📚 API reference

### `POST /synthesize`

```json
{
  "text": "Hola Facu",
  "speaker": "vivian",
  "language": "spanish",
  "temperature": 0.9,
  "top_k": 50,
  "top_p": 1.0,
  "repetition_penalty": 1.05,
  "max_new_tokens": 2048
}
```

Response: `audio/wav` PCM 16-bit @ 24 kHz mono.

Headers:
- `X-RTF`
- `X-Total-Time`
- `X-Audio-Duration`
- `X-Tokens-Per-Sec`

### `POST /synthesize/json`

Same request, returns `{audio_b64, sample_rate, metrics}` — useful when you
want to inline the audio without writing a temp file.

### `GET /speakers`

Lists available speakers with codec IDs. Default: `vivian`.

### `GET /health`

Reports model status, device routing, and load time.

---

## 🎙️ Voices

`serena`, **vivian** (default), `uncle_fu`, `ryan`, `aiden`, `ono_anna`,
`sohee`, `eric`, `dylan`. Languages: `auto`, `english`, `chinese`, `spanish`,
`german`, `french`, `italian`, `japanese`, `korean`, `portuguese`, `russian`,
plus `beijing_dialect` and `sichuan_dialect`.

Use `language: null` for auto-detect.

---

## ⚙️ Device routing

OpenVINO lets you put each submodel on a different device. On mixed hardware
this can matter a lot:

| Submodel | Typical device | Why |
|---|---|---|
| `text_model` | GPU or CPU | small, either works |
| `codec_embedding`, `cp_codec_embedding` | GPU or CPU | tiny lookup tables |
| `talker` | **GPU** | large transformer, GPU is much faster |
| `code_predictor` | **CPU** | empirically faster on x30w-k; the OV compiler picks better CPU kernels for sequential ops |
| `speech_decoder` (vocoder) | **CPU on iGPU hardware** | INT8 iGPU path is broken on x30w-k; CPU is 15× faster *and* correct |

---

## 🔭 Verification: STT correlation

After generating audio, verify it actually says what you asked:

```bash
curl -X POST http://your-parakeet-server:5092/v1/audio/transcriptions \
  -F model=parakeet-tdt-0.6b-v3 \
  -F file=@out.wav
```

Metrics like RMS / peak / zero-crossings only tell you about audio energy —
they don't tell you the model said what you asked. STT correlation is the
ground-truth check.

---

## 🤝 Contributing back to upstream

Emerson's policy (from the original README):

> PRs not related to OpenArc development may not be accepted, as this
> implementation must remain *mostly* canonical.

If you find a fix or improvement that should go to the original repo, open
a PR there first. This fork is mainly for FastAPI serving and the
`sd_device` workaround.

---

## 📜 Acknowledgements

- **[Emerson Tatelbaum (@SearchSavior)](https://github.com/SearchSavior)** —
  original Qwen3-TTS OpenVINO port, the OpenVINO IR export pipeline, the
  `nn.Module` refactor, and the reference implementations.
- **[OpenArc](https://github.com/SearchSavior/OpenArc)** — the project
  this work was built to support.
- **[OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)**
  — `qwen_3_tts_helper.py` was a reference for the original export work.
- **[Qwen team @ Alibaba](https://arxiv.org/abs/2601.15621)** — for the
  Qwen3-TTS Technical Report and the open-weight models.

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and
          Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and
          Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and
          Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```