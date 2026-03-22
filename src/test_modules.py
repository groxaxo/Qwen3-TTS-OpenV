"""Module-backed TTS smoke test script.

This mirrors `test_tts.py` but uses the refactor modules in `src/modules/*`
for inference. Optionally compare module output against the existing functional
pipeline.
"""

import argparse
import time
import wave

import numpy as np
import torch

import qwen3_tts
from modules import (
    CodePredictorBackbone,
    SpeechDecoder,
    TalkerBackbone,
    generate,
)


def _build_modules(state):
    cfg = state["config"]
    talker_config = cfg["talker_config"]
    cp_config = talker_config["code_predictor_config"]
    st_cfg = state["speech_tokenizer"]["config"]
    speech_decoder_cfg = st_cfg.get("decoder_config", st_cfg)

    t0 = time.perf_counter()
    talker = TalkerBackbone(state["weights"], talker_config)
    print(f"[perf] TalkerBackbone init:       {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    code_predictor = CodePredictorBackbone(state["weights"], cp_config)
    print(f"[perf] CodePredictorBackbone init:{time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    speech_decoder = SpeechDecoder(state["speech_tokenizer"]["weights"], speech_decoder_cfg)
    print(f"[perf] SpeechDecoder init:        {time.perf_counter() - t0:.3f}s")

    return talker, code_predictor, speech_decoder


def _build_voice_clone_inputs(args, state):
    se_weights = state.get("speaker_encoder_weights")
    if not se_weights:
        raise ValueError("Model does not have speaker encoder weights. Voice clone requires a base model.")

    if isinstance(args.ref_audio, str):
        audio_np, audio_sr = qwen3_tts._load_audio_wav(args.ref_audio)
    else:
        audio_np, audio_sr = args.ref_audio

    speaker_embed = qwen3_tts.extract_speaker_embedding(
        audio_np,
        audio_sr,
        se_weights,
        state["config"].get("speaker_encoder_config", {}),
    )

    use_icl = args.ref_text is not None and not args.x_vector_only
    ref_codes = None
    if use_icl:
        audio_tensor = torch.from_numpy(audio_np).float()
        ref_codes = qwen3_tts.encode(state["speech_tokenizer"], audio_tensor, audio_sr)[0]

    return speaker_embed, ref_codes


def _run_modules_generate(state, args):
    talker, code_predictor, speech_decoder = _build_modules(state)

    kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=not args.no_sample,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
    )
    kwargs["subtalker_dosample"] = not args.no_sample

    if args.mode == "voice_design":
        return generate(
            text=args.text,
            talker=talker,
            cp=code_predictor,
            speech_decoder=speech_decoder,
            tokenizer=state["tokenizer"],
            weights=state["weights"],
            config=state["config"],
            speech_tokenizer_state=state["speech_tokenizer"],
            speaker=None,
            instruct=args.voice_description,
            language=args.language,
            non_streaming_mode=args.non_streaming_mode,
            **kwargs,
        )

    if args.mode == "custom_voice":
        return generate(
            text=args.text,
            talker=talker,
            cp=code_predictor,
            speech_decoder=speech_decoder,
            tokenizer=state["tokenizer"],
            weights=state["weights"],
            config=state["config"],
            speech_tokenizer_state=state["speech_tokenizer"],
            speaker=args.speaker,
            instruct=args.style_instruction,
            language=args.language,
            non_streaming_mode=args.non_streaming_mode,
            **kwargs,
        )

    if args.mode in ("voice_clone", "base"):
        speaker_embed, ref_codes = _build_voice_clone_inputs(args, state)
        return generate(
            text=args.text,
            talker=talker,
            cp=code_predictor,
            speech_decoder=speech_decoder,
            tokenizer=state["tokenizer"],
            weights=state["weights"],
            config=state["config"],
            speech_tokenizer_state=state["speech_tokenizer"],
            speaker=None,
            instruct=args.voice_instruction,
            language=args.language,
            non_streaming_mode=args.non_streaming_mode,
            speaker_embed=speaker_embed,
            ref_text=args.ref_text,
            ref_codes=ref_codes,
            **kwargs,
        )

    raise ValueError(f"Unknown mode: {args.mode}")


def _run_functional_generate(state, args):
    if args.mode == "voice_design":
        return qwen3_tts.generate_voice_design(
            state,
            text=args.text,
            instruct=args.voice_description,
            language=args.language,
            non_streaming_mode=args.non_streaming_mode,
            do_sample=not args.no_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.no_sample,
            max_new_tokens=args.max_new_tokens,
        )

    if args.mode == "custom_voice":
        return qwen3_tts.generate_custom_voice(
            state,
            text=args.text,
            speaker=args.speaker,
            language=args.language,
            instruct=args.style_instruction,
            non_streaming_mode=args.non_streaming_mode,
            do_sample=not args.no_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.no_sample,
            max_new_tokens=args.max_new_tokens,
        )

    if args.mode in ("voice_clone", "base"):
        return qwen3_tts.generate_voice_clone(
            state,
            text=args.text,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only=args.x_vector_only,
            language=args.language,
            instruct=args.voice_instruction,
            non_streaming_mode=args.non_streaming_mode,
            do_sample=not args.no_sample,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            subtalker_dosample=not args.no_sample,
            max_new_tokens=args.max_new_tokens,
        )

    raise ValueError(f"Unknown mode: {args.mode}")


def _compare_outputs(module_wave, functional_wave):
    min_len = min(len(module_wave), len(functional_wave))
    if min_len == 0:
        print("[compare] One output is empty; cannot compute diff.")
        return
    diff = np.abs(module_wave[:min_len] - functional_wave[:min_len])
    print(
        f"[compare] len(module)={len(module_wave)} len(func)={len(functional_wave)} "
        f"max_abs_diff={diff.max():.6f} mean_abs_diff={diff.mean():.6f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS module-backed test script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:

  custom_voice:
    python test_modules.py --model-path ./Qwen3-TTS-12Hz-1.7B-CustomVoice \\
      --mode custom_voice --text "Hello world" --speaker Chelsie --top_k 50

  voice_design:
    python test_modules.py --model-path ./Qwen3-TTS-12Hz-1.7B-VoiceDesign \\
      --mode voice_design --text "Hello world" --voice-description "warm narration"

  compare mode:
    python test_modules.py --model-path ./Qwen3-TTS-12Hz-1.7B-VoiceDesign \\
      --mode voice_design --text "Hello world" --voice-description "warm" --compare
""",
    )

    parser.add_argument("--model-path", default="./Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    parser.add_argument("--mode", choices=["voice_design", "custom_voice", "voice_clone", "base"], default="voice_design")
    parser.add_argument("--output", default="test_modules_output.wav")
    parser.add_argument("--output-functional", default=None, help="Optional output path for functional baseline")

    parser.add_argument("--text", default="Hello, this is a test of the Qwen3-TTS module path.")
    parser.add_argument("--language", default="English")

    vd = parser.add_argument_group("voice_design mode")
    vd.add_argument("--voice-description", default=None)

    cv = parser.add_argument_group("custom_voice mode")
    cv.add_argument("--speaker", default=None)
    cv.add_argument("--style-instruction", default=None)

    vc = parser.add_argument_group("voice_clone mode")
    vc.add_argument("--ref-audio", default=None)
    vc.add_argument("--ref-text", default=None)
    vc.add_argument("--x-vector-only", action="store_true")
    vc.add_argument("--voice-instruction", default=None)

    gen = parser.add_argument_group("generation params")
    gen.add_argument("--max-new-tokens", type=int, default=2048)
    gen.add_argument("--temperature", type=float, default=0.9)
    gen.add_argument("--top-k", type=int, default=50)
    gen.add_argument("--top-p", type=float, default=0.95)
    gen.add_argument("--no-sample", action="store_true")
    gen.add_argument("--repetition-penalty", type=float, default=1.0)
    gen.add_argument("--non-streaming-mode", type=bool, default=True)

    parser.add_argument("--compare", action="store_true", help="Run both module + functional paths and compare.")
    parser.add_argument("--use-modules", action="store_true", help="Force module-only path when compare is false.")

    args = parser.parse_args()

    if args.mode == "voice_design" and not args.voice_description:
        parser.error("--voice-description required for voice_design mode")
    if args.mode == "custom_voice" and not args.speaker:
        parser.error("--speaker required for custom_voice mode")
    if args.mode in ("voice_clone", "base") and not args.ref_audio:
        parser.error("--ref-audio required for voice_clone mode")
    if args.mode in ("voice_clone", "base") and args.x_vector_only and args.ref_text is not None:
        print("Ignoring --ref-text due --x-vector-only")
        args.ref_text = None

    t_load = time.perf_counter()
    state = qwen3_tts.load_model(args.model_path)
    print(f"[perf] load_model:               {time.perf_counter() - t_load:.3f}s")

    if args.compare:
        print("Running functional baseline...")
        t0 = time.perf_counter()
        func_wave, func_sr = _run_functional_generate(state, args)
        t_func = time.perf_counter() - t0
        audio_s = len(func_wave) / func_sr
        print(f"Functional: {len(func_wave)} samples @ {func_sr}Hz ({audio_s:.2f}s audio)")
        print(f"[perf] functional pipeline:      {t_func:.3f}s  (RTF {t_func/audio_s:.2f}x)")

        print("Running module-backed pipeline...")
        t0 = time.perf_counter()
        mod_wave, mod_sr = _run_modules_generate(state, args)
        t_mod = time.perf_counter() - t0
        audio_s = len(mod_wave) / mod_sr
        print(f"Module: {len(mod_wave)} samples @ {mod_sr}Hz ({audio_s:.2f}s audio)")
        print(f"[perf] module pipeline:          {t_mod:.3f}s  (RTF {t_mod/audio_s:.2f}x)")
        _compare_outputs(mod_wave, func_wave)
    else:
        if not args.use_modules:
            t0 = time.perf_counter()
            mod_wave, mod_sr = _run_functional_generate(state, args)
            t_func = time.perf_counter() - t0
            audio_s = len(mod_wave) / mod_sr
            print(f"Functional: {len(mod_wave)} samples @ {mod_sr}Hz ({audio_s:.2f}s audio)")
            print(f"[perf] functional pipeline:      {t_func:.3f}s  (RTF {t_func/audio_s:.2f}x)")
        else:
            t0 = time.perf_counter()
            mod_wave, mod_sr = _run_modules_generate(state, args)
            t_mod = time.perf_counter() - t0
            audio_s = len(mod_wave) / mod_sr
            print(f"Module: {len(mod_wave)} samples @ {mod_sr}Hz ({audio_s:.2f}s audio)")
            print(f"[perf] module pipeline:          {t_mod:.3f}s  (RTF {t_mod/audio_s:.2f}x)")
            if args.output_functional:
                with wave.open(args.output_functional, "w") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(mod_sr)
                    wf.writeframes((np.clip(mod_wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    if args.compare:
        if args.output_functional:
            with wave.open(args.output_functional, "w") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(func_sr)
                wf.writeframes((np.clip(func_wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes())

    # Default save module output (or functional output if modules disabled)
    if args.compare or args.use_modules:
        out_wave, out_sr = mod_wave, mod_sr
    else:
        out_wave, out_sr = mod_wave, mod_sr

    with wave.open(args.output, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(out_sr)
        wf.writeframes((np.clip(out_wave, -1.0, 1.0) * 32767).astype(np.int16).tobytes())
    print(f"Saved output to {args.output}")


if __name__ == "__main__":
    main()
