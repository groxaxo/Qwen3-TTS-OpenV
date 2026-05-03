"""End-to-end test: load model, generate speech, save to WAV."""
import argparse
import wave
import numpy as np
import torch
from torch_functional import qwen3_tts


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS pure PyTorch inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
model_type usage and examples:

  voice_design:
    args: --text, --language, --voice-description (required)
    example:
      python test_tts.py --model-path ./Qwen3-TTS-12Hz-1.7B-VoiceDesign \\
        --mode voice_design --text "Hello world" --voice-description "Warm female narration" \\
        --max-new-tokens 2048 --temperature 0.9 --top-k 50 --top-p 0.95 --repetition-penalty 1.0

  custom_voice:
    args: --text, --language, --speaker (required), [--style-instruction]
    example:
      python test_tts.py --model-path ./Qwen3-TTS-12Hz-1.7B-CustomVoice \\
        --mode custom_voice --text "Hello world" --speaker Chelsie --style-instruction "Speak with energy" \\
        --max-new-tokens 2048 --temperature 0.9 --top-k 50 --top-p 0.95 --repetition-penalty 1.0

  voice_clone:
    args: --text, --language, --ref-audio (required), [--ref-text], [--x-vector-only]
    example:
      python test_tts.py --model-path ./Qwen3-TTS-12Hz-1.7B-VoiceClone \\
        --mode voice_clone --text "Hello world" --ref-audio ./sample_ref.wav --ref-text "Hello world" \\
        --max-new-tokens 2048 --temperature 0.9 --top-k 50 --top-p 0.95 --repetition-penalty 1.0

  base (handled via voice_clone path):
    args: --text, --language, --ref-audio (required), [--ref-text], [--x-vector-only]
    example:
      python test_tts.py --model-path ./Qwen3-TTS-12Hz-0.6B-Base \\
        --text "Hello world" --ref-audio ./sample_ref.wav \\
        --max-new-tokens 2048 --temperature 0.9 --top-k 50 --top-p 0.95 --repetition-penalty 1.0
""",
    )
    parser.add_argument("--model-path", default="./Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                        help="Path to model directory")
    parser.add_argument("--mode", choices=["voice_design", "custom_voice", "voice_clone"],
                        default=None,
                        help="Generation mode (auto-detected from model config if omitted)")
    parser.add_argument("--output", "-o", default="test_output.wav",
                        help="Output WAV file path")

    # Common arguments (all modes)
    common = parser.add_argument_group("common (all modes)")
    common.add_argument("--text", default="Hello, this is a test of the Qwen3 text to speech system.",
                        help="Text to synthesize")
    common.add_argument("--language", default="English",
                        help="Language (default: English)")

    # voice_design arguments
    vd = parser.add_argument_group("voice_design mode")
    vd.add_argument("--voice-description", default=None,
                    help="Voice description / instruction (required for voice_design)")

    # custom_voice arguments
    cv = parser.add_argument_group("custom_voice mode")
    cv.add_argument("--speaker", default=None,
                    help="Speaker name (required for custom_voice)")
    cv.add_argument("--style-instruction", default=None,
                    help="Optional style instruction for custom_voice")

    # voice_clone arguments
    vc = parser.add_argument_group("voice_clone mode")
    vc.add_argument("--ref-audio", default=None,
                    help="Reference audio WAV path (required for voice_clone)")
    vc.add_argument("--ref-text", default=None,
                    help="Optional transcription of the reference audio")
    vc.add_argument("--x-vector-only", action="store_true",
                    help="Use only x-vector speaker embedding, ignore --ref-text")

    # Generation parameters
    gen = parser.add_argument_group("generation parameters")
    gen.add_argument("--max-new-tokens", type=int, default=2048,
                     help="Maximum new tokens to generate")
    gen.add_argument("--temperature", type=float, default=0.9,
                     help="Sampling temperature")
    gen.add_argument("--top-k", type=int, default=50,
                     help="Top-k sampling")
    gen.add_argument("--top-p", type=float, default=0.95,
                     help="Top-p (nucleus) sampling")
    gen.add_argument("--no-sample", action="store_true",
                     help="Disable sampling (use greedy decoding)")
    gen.add_argument("--repetition-penalty", type=float, default=1.0,
                     help="Repetition penalty (1.0 = no penalty)")

    args = parser.parse_args()

    print("Loading model...")
    state = qwen3_tts.load_model(args.model_path)
    model_type = state["config"].get("tts_model_type", "unknown")
    print(f"Model loaded. Type: {model_type}")
    print(f"Main model weights: {len(state['weights'])} tensors")
    print(f"Speech tokenizer weights: {len(state['speech_tokenizer']['weights'])} tensors")

    # Auto-detect mode from model config if not specified
    mode = args.mode
    if mode is None:
        if args.ref_audio is not None:
            mode = "voice_clone"
        else:
            mode = model_type
        print(f"Auto-detected mode: {mode}")

    # Validate required arguments per mode
    if mode == "voice_design":
        if args.voice_description is None:
            parser.error("--voice-description is required for voice_design mode")
    elif mode == "custom_voice":
        if args.speaker is None:
            parser.error("--speaker is required for custom_voice mode")
    elif mode in ("voice_clone", "base"):
        if args.ref_audio is None:
            parser.error("--ref-audio is required for voice_clone mode")
        if args.x_vector_only and args.ref_text is not None:
            print("Warning: --x-vector-only set, ignoring --ref-text")
            args.ref_text = None

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    print(f"\nMode: {mode}")
    print(f"Text: {args.text}")
    print(f"Language: {args.language}")
    if mode == "voice_design":
        print(f"Voice description: {args.voice_description}")
    elif mode == "custom_voice":
        print(f"Speaker: {args.speaker}")
        if args.style_instruction:
            print(f"Style instruction: {args.style_instruction}")
    elif mode in ("voice_clone", "base"):
        print(f"Reference audio: {args.ref_audio}")
        if args.ref_text:
            print(f"Reference text: {args.ref_text}")
        print(f"X-vector only: {args.x_vector_only or args.ref_text is None}")
    print(f"Generation params: {gen_kwargs}")

    print("\nGenerating speech...")
    with torch.inference_mode():
        if mode == "voice_design":
            wav, sr = qwen3_tts.generate_voice_design(
                state, text=args.text, instruct=args.voice_description,
                language=args.language, **gen_kwargs)
        elif mode == "custom_voice":
            wav, sr = qwen3_tts.generate_custom_voice(
                state, text=args.text, speaker=args.speaker,
                language=args.language, instruct=args.style_instruction,
                **gen_kwargs)
        elif mode in ("voice_clone", "base"):
            wav, sr = qwen3_tts.generate_voice_clone(
                state, text=args.text, ref_audio=args.ref_audio,
                ref_text=args.ref_text, x_vector_only=args.x_vector_only,
                language=args.language, **gen_kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    print(f"Generated {len(wav)} samples at {sr}Hz ({len(wav)/sr:.2f}s)")

    with wave.open(args.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((wav * 32767).astype(np.int16).tobytes())

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
