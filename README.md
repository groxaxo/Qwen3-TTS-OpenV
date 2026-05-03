
# Converting to OpenVINO --> Pytorch from scratch

Without transformers. 


This repo contains **three** implementations of Qwen3-TTS I made over two months in early 2026 as a way to get inside the complex process of building an OpenVINO IR from scratch, without transformers. 

AI assistance was used during development; however, even Opus 4.5 struggled to apply OpenVINO conventions I have learned from developing OpenArc, studying the src, examples etc. The long timeline was because I took time to study the code, test it, and optimize heavily. 

Optimization includes 

- making choices in the export design which anticpate where kernel fusions happen during compile before and during inference time 

- Correctly assessing tradeoffs of stateful pipeline, which basically means passing hidden states with logits between subgraphs

- through testing I discovered that the code predictor was faster on CPU ie, the compiler chooses better kernels. Even with copy it's much faster.

That's what I remember from development; another learning from this project was to take better notes.


This repo contains end to end qwen3-tts:

- Once in `torch.nn.functional as F`
  - So tensors only, without the nn.module class-like approach. No bueno for openvino export.

- Again using `import torch.nn as nn`
  - This ended up being neccessary for the OpenVINO pytorch trace to work properly
  - `nn.Module` does a better job of keeping things organized

- And finally a complete OpenVINO implementation of all three tasks, validated for 1.7B on CPUs and GPUs. 

At the time I used an A770 and Xeon W2255 but since deployment in OpenArc there have been no portability issues to other hardware yet.

## Designing an Export to OpenVINO IR

Intel has done very little to document the actual procedure around building IR in a from scratch way; almost all the examples import from `transformers` and inherit all `transformers` complexity.

Here is the procedure for converting *any* pytorch model to OpenVINO IR;

- Define your operations as `nn.Modules` that contain some logic, and end in a `forward` that returns some data the next step needs.

- Make a thin `nn.Module` on top of each `forward` call

- In that way making an IR requires
  - knowing your models data flow
  - making choices about what device makes sense to use for the ops required for that part of the pipeline by testing; compiling an openvino model is like 
  - through profiling and ensuring correctness vs pytorch on CPU slowly work in the openvino details like `fuse_cache_reorder`
  - I used the example from openvino-dev-samples in the [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen3-tts/qwen_3_tts_helper.py) for inspiration, but my implementation diverges and makes some different choices around submodel design.


## Usage

Clone the repo and run 

```
uv sync
```



`ov_convert` (PyTorch -> OpenVINO IR)

Converts a Qwen3-TTS checkpoint directory into an OpenVINO model directory.

```bash
uv run src/openvino/ov_convert.py \
  --model-path <qwen3_tts_model_dir> \
  --new <output_ov_dir> \
  --weight-format <fp16|int8|int4> \
  [--model-type <voice_clone|voice_design|custom_voice>] \
  [--cp-weight-format <fp16|int8|int4>]
```

Argument notes:
- `--model-path`: path to the source Qwen3-TTS model directory.
- `--new`: output directory for converted OpenVINO IR files.
- `--weight-format`: default precision/compression for exported modules.
- `--model-type`: optional override for `tts_model_type` from `config.json`.
- `--cp-weight-format`: optional code predictor override; use `fp16` if quantized CP causes NaNs on GPU.

Examples:

```bash
# voice_design export (INT8)
uv run src/openvino/ov_convert.py \
  --model-path ./Qwen3-TTS-1.7B \
  --new ./voicedesign-1.7b-int8-ov \
  --model-type voice_design \
  --weight-format int8

# custom_voice export (INT8)
uv run src/openvino/ov_convert.py \
  --model-path ./Qwen3-TTS-1.7B \
  --new ./customvoice-1.7b-int8-ov \
  --model-type custom_voice \
  --weight-format int8

# voice_clone export (INT8, but keep code predictor in FP16 for GPU stability)
uv run src/openvino/ov_convert.py \
  --model-path ./Qwen3-TTS-1.7B \
  --new ./voiceclone-1.7b-int8-ov \
  --model-type voice_clone \
  --weight-format int8 \
  --cp-weight-format fp16
```







Run these commands from the repository root.

`voice_design`
```bash
uv run src/openvino/ov_infer.py \
  --ov-dir ./voicedesign-1.7b-int8-ov \
  --mode voice_design \
  --text "This is a test of Echo9Zulu's Qwen3 TTS OpenVINO implementation for the voice design task." \
  --voice-description "Speak like an announcer over an intercom" \
  --output voicedesign_int8.wav
```

`custom_voice`
```bash
uv run src/openvino/ov_infer.py \
  --ov-dir ./customvoice-1.7b-int8-ov \
  --mode custom_voice \
  --text "This is a test of Echo9Zulu's Qwen3 TTS OpenVINO implementation for the custom voice task." \
  --speaker uncle_fu \
  --output customvoice_int8.wav \
  --device GPU.0
```

`voice_clone`
```bash
uv run src/openvino/ov_infer.py \
  --ov-dir ./voiceclone-1.7b-int8-ov \
  --mode voice_clone \
  --text "This is a test of Echo9Zulu's Qwen3 TTS OpenVINO implementation for the voice clone task." \
  --ref-audio ./elmo_sample.wav \
  --ref-text "Color? Red! [laughs] Or, or who's your best friend? Um, Elmo's pet goldfish, Dorothy..." \
  --output voiceclone_elmo_int8.wav \
  --device GPU.0
```

