voice_design

```
uv run ov_infer.py --ov-dir voicedesign-1.7b-int8-ov  --mode voice_design --text "This is a test of Echo9Zulu's Qwen3 TTS openvino implementation for the voice design task!" --voice-description "Speak like an announcer over an intercom" --output voicedesign_int8.wav
```

custom_voice
```
uv run ov_infer.py --ov-dir customvoice-1.7b-int8-ov  --mode custom_voice --text "This is a test of Echo9Zulu's Qwen3 TTS openvino implementation for the custom voice task!" --speaker uncle_fu --output customvoice_int8.wav --device GPU.0
```

voice_clone
```
uv run ov_infer.py --ov-dir base-1.7b-int8-ov  --mode voice_clone --text "This is a test of Echo9Zulu's Qwen3 TTS openvino implementation for the voice clone task!" --ref-audio /home/echo/Projects/ov_qwen3_tts/audio/elmo_sample.wav --ref-text "Color? Red! [laughs] Or, or who's your best friend? Um, Elmo's pet goldfish, Dorothy. Is it like... what is it like living on Sesame Street? That's a good question. Awesome, baby! [laughs] Wait... Elmo's not supposed to be answering these yet. [laughs] Sorry! [laughs] Well... now, you can ask Elmo any question you want right here on YouTube using this..." --output voiceclone_elmo_int8.wav --device GPU.0
```