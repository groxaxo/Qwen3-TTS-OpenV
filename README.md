
# Converting Qwen3-TTS Pytorch to OpenVINO from scratch

Without transformers. 


This repo contains **three** implementations of Qwen3-TTS I made over two months in early 2026 as a way to get inside the complex process of building an OpenVINO IR from scratch, without transformers to then implement in [OpenArc](https://github.com/SearchSavior/OpenArc).

AI assistance was used during development; however, even Opus 4.5 struggled to apply OpenVINO conventions I have learned from developing OpenArc, studying the src, examples etc. I made effort to study the code, test it, and optimize heavily. An awesome way to learn the architecture from zero, with a highly optimized inference implementation included. Pushing performance further would require authoring custom opencl GPU kernels for slow ops, a procedure left to future work.

>!NOTE
> To use a finetuned or otherwise modified version of Qwen3-TTS in OpenArc, you need to export using ov_convert.py


## Optimizations 

- making choices in the export design which anticpate where kernel fusions happen during compile before and during inference time 

- Assessing tradeoffs of stateful pipeline, which basically means passing hidden states with logits between subgraphs

- through testing I discovered that the code predictor was faster on CPU ie, the compiler chooses better kernels. Even with copy it's much faster; most ops which are sequential are faster on CPU; in general this is true, but OV has long history of utilizing hardware features

That's what I remember from development; another learning from this project was to take better notes.


This repo contains end to end qwen3-tts for all tasks:

- Once in `torch.nn.functional as F`
  - So tensors only, without the nn.module class-like approach. No bueno for openvino export, discovered the long way.

- Again using `import torch.nn as nn`
  - This ended up being neccessary for the OpenVINO pytorch trace to work properly
  - `nn.Module` does a better job of keeping things organized

- And finally a complete OpenVINO implementation of all three tasks, validated for 1.7B on CPUs and GPUs. 

At the time I used an A770 and Xeon W2255 but since deployment in OpenArc there have been no portability issues to other hardware yet.

## Designing an Export to OpenVINO IR

Intel has done very little to document the actual procedure around building IR in a from scratch way; almost all the examples import from `transformers` and inherit all `transformers` complexity which makes makes the code intel does publish quite terse; to make sense of how they 

Here is the procedure for converting *any* pytorch model to OpenVINO IR;

- Define your operations as `nn.Modules` that contain some logic, and end in a `forward` that returns some data the next step needs.

- Make a thin `nn.Module` on top of each `forward` call

- In that way making an IR requires
  - knowing your models data flow
  - making choices about what device makes sense to use for the ops required for that part of the pipeline by testing; compiling an openvino model is like 
  - through profiling and ensuring correctness vs pytorch on CPU slowly work in the openvino details like `fuse_cache_reorder`
  - I used the example from openvino-dev-samples in the [notebook](https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/qwen3-tts/qwen_3_tts_helper.py) for inspiration, but my implementation diverges and makes some different choices around submodel design.

## Project Structure

```text
.
├── CLAUDE_convert_from_functional_to_module.md # Notes from functional-to-module refactoring work.
├── snip_snip.py                                # Local scratch/helper script from development experiments.

├── openivno_notebook_reference/                # OpenVINO notebook reference material used for comparison.
│   └── qwen_3_tts_helper.py                    # Upstream helper code that informed export/inference design.
├── 
qwen3_tts_transformers_reference/           # Embedded Transformers-based reference implementation.
│   ├── README.md                               # Upstream reference docs.
│   ├── pyproject.toml                          # Upstream package metadata/dependencies.
│   ├── assets/                                 # Reference assets (including technical report PDF).
│   └── qwen_tts/                               # Upstream implement. Not installed or used; context for the agent
└── src/                                        # Primary source code for this project.
    
    ├── openvino/                               # OpenVINO conversion and runtime entrypoints.
    │   ├── ov_convert.py                       # Exports PyTorch checkpoints into OpenVINO IR submodels.
    │   └── ov_infer.py                         # Runs OpenVINO inference for supported voice modes.
    
    ├── torch_functional/                       # Tensor/functional-style PyTorch implementation.
    │   ├── qwen3_tts.py                        # Functional model flow and generation logic.
    │   ├── qwen3_tts_tokenizer.py              # Tokenization utilities for functional implementation.
    │   └── test_tts.py                         # Functional implementation tests/examples.
    
    ├── torch_modules/                          # nn.Module-based PyTorch implementation for tracing/export.
    │   ├── talker.py                           # Main orchestration module for TTS generation.
    │   ├── speech_decoder.py                   # Speech decoder model components.
    │   ├── code_predictor.py                   # Acoustic code prediction module.
    │   ├── speaker_encoder.py                  # Speaker conditioning and embedding module.
    │   ├── generate.py                         # Shared generation helpers for module-based stack.
    │   └── constants.py                        # Shared constants/config values.
    └── test_modules.py                         # inference script for module-based implementation.
```

## How to learn from this project


Unlike other from scratch implementation repos this one encourages using AI tools to help you learn. Have the agent explore and explain the architecture of qwen-tts while you read the paper; then interrogate the code to understand it's business. 

This was my first attempt to do a reasonably hard architecture, which was intentional; I needed to prove out that making bespoke implementions outside of what's offic

Lessons in this codebase can be used to design an export for ANY pytorch model in all of transformers, documenting a deeper dive you can follow `outside` transformers abstraction.


## Usage

Clone the repo and run 

```
uv sync
```

obtain the Qwen3-TTS pytorch models from the [hub](https://huggingface.co/collections/Qwen/qwen3-tts)

- 1.7B export is fully supported
- 0.6B export had some issues I didn't finish working out >:D
- the pytorch code covers both

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

## Future work

- PRs not related to OpenArc development may not be accepted, as this implementation must remain *mostly* cannonical. 

- PRs which implement an NPU pathway are most welcome! The foundation is laid, but I don't have an NPU device to test with.




## Acknowledgements 

```
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

[OpenVINO Notebooks](https://github.com/openvinotoolkit/openvino_notebooks)

OpenArc community for their support