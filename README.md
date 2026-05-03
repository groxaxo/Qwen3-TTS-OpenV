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
