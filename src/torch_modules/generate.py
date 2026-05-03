"""Pure-Python generation orchestration using module backbones."""

from typing import Dict, Optional
import time

import numpy as np
import torch

from torch_functional.qwen3_tts import apply_repetition_penalty, build_talker_input, sample_token
from torch_functional.qwen3_tts_tokenizer import split_rvq_decode
from .talker import TalkerBackbone
from .code_predictor import CodePredictorBackbone
from .speech_decoder import SpeechDecoder
from .constants import TTSConstants


def slice_mrope(cos: torch.Tensor, sin: torch.Tensor, start: int, length: int):
    return cos[start:start + length].unsqueeze(0).unsqueeze(0), sin[start:start + length].unsqueeze(
        0
    ).unsqueeze(0)


def slice_rope(cos: torch.Tensor, sin: torch.Tensor, start: int, length: int):
    return cos[start:start + length].unsqueeze(0).unsqueeze(0), sin[start:start + length].unsqueeze(
        0
    ).unsqueeze(0)


def build_empty_cache_kwargs(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    kwargs = {}
    for i in range(n_layers):
        kwargs[f"past_key_{i}"] = torch.zeros(1, n_kv_heads, 0, head_dim, dtype=dtype, device=device)
        kwargs[f"past_value_{i}"] = torch.zeros(1, n_kv_heads, 0, head_dim, dtype=dtype, device=device)
    return kwargs


def extract_present_cache_kwargs(outputs, n_layers: int) -> dict[str, torch.Tensor]:
    kwargs = {}
    for i in range(n_layers):
        kwargs[f"past_key_{i}"] = getattr(outputs, f"present_key_{i}")
        kwargs[f"past_value_{i}"] = getattr(outputs, f"present_value_{i}")
    return kwargs


def generate_sub_codes(
    first_code_embed: torch.Tensor,
    past_hidden: torch.Tensor,
    cp_backbone: CodePredictorBackbone,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    consts: TTSConstants,
    sampling_kwargs: dict,
):
    num_sub_codes = consts.num_code_groups - 1

    empty_kvs = build_empty_cache_kwargs(
        consts.cp_n_layers,
        consts.cp_n_kv_heads,
        consts.cp_head_dim,
        cp_backbone._weight_dtype,
        past_hidden.device,
    )

    prefill = torch.cat([past_hidden, first_code_embed], dim=1)
    cos, sin = slice_rope(rope_cos, rope_sin, start=0, length=2)
    outputs = cp_backbone(prefill, cos, sin, **empty_kvs)
    cp_hidden = outputs.hidden

    step_logits = cp_backbone.lm_head(cp_hidden[:, -1:, :], 0)[:, 0, :]
    token_id = sample_token(step_logits, **sampling_kwargs)[:, -1]
    token_id = token_id.view(1, 1)

    sub_codes = [token_id]
    code_embed = cp_backbone.codec_embedding(token_id, 0)
    embeds_sum = first_code_embed + code_embed
    cp_kvs = extract_present_cache_kwargs(outputs, consts.cp_n_layers)

    cache_pos = 2
    for step in range(1, num_sub_codes):
        cos, sin = slice_rope(rope_cos, rope_sin, start=cache_pos, length=1)
        outputs = cp_backbone(code_embed, cos, sin, **cp_kvs)
        cp_hidden = outputs.hidden

        step_logits = cp_backbone.lm_head(cp_hidden[:, -1:, :], step)[:, 0, :]
        token_id = sample_token(step_logits, **sampling_kwargs)[:, -1]
        token_id = token_id.view(1, 1)
        sub_codes.append(token_id)

        code_embed = cp_backbone.codec_embedding(token_id, step)
        embeds_sum = embeds_sum + code_embed
        cp_kvs = extract_present_cache_kwargs(outputs, consts.cp_n_layers)
        cache_pos += 1

    sub_codes = torch.cat(sub_codes, dim=-1)
    return sub_codes, embeds_sum


def generate_sub_codes_from_state(
    first_code_embed: torch.Tensor,
    past_hidden: torch.Tensor,
    cp_backbone: CodePredictorBackbone,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    consts: TTSConstants,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
) -> tuple[torch.Tensor, torch.Tensor]:
    sampling_kwargs = dict(do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
    return generate_sub_codes(first_code_embed, past_hidden, cp_backbone, rope_cos, rope_sin, consts, sampling_kwargs)


def build_input(
    text: str,
    tokenizer,
    weights: dict,
    config: dict,
    language: str = "auto",
    speaker=None,
    instruct: Optional[str] = None,
    speaker_embed=None,
    non_streaming_mode: bool = True,
    ref_text=None,
    ref_codes=None,
) -> Dict[str, torch.Tensor]:
    return build_talker_input(
        text,
        weights,
        config,
        tokenizer,
        language=language,
        speaker=speaker,
        instruct=instruct,
        speaker_embed=speaker_embed,
        non_streaming_mode=non_streaming_mode,
        ref_text=ref_text,
        ref_codes=ref_codes,
    )


def generate(
    text: str,
    talker: TalkerBackbone,
    cp: CodePredictorBackbone,
    speech_decoder: SpeechDecoder,
    tokenizer,
    weights: dict,
    config: dict,
    speech_tokenizer_state: dict,
    talker_input: Optional[Dict[str, torch.Tensor]] = None,
    speaker: Optional[str] = None,
    instruct: Optional[str] = None,
    speaker_embed: Optional[torch.Tensor] = None,
    language: str = "Auto",
    non_streaming_mode: bool = True,
    ref_text: Optional[str] = None,
    ref_codes=None,
    max_new_tokens: int = 2048,
    do_sample: bool = True,
    top_k: int = 50,
    top_p: float = 1.0,
    temperature: float = 0.9,
    repetition_penalty: float = 1.05,
    subtalker_dosample: bool = True,
    subtalker_top_k: int = 50,
    subtalker_top_p: float = 1.0,
    subtalker_temperature: float = 0.9,
):
    t0 = time.perf_counter()
    consts = TTSConstants.from_config_and_weights(config, weights)
    print(f"[perf] TTSConstants init:    {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    if talker_input is None:
        talker_input = build_input(
            text=text,
            tokenizer=tokenizer,
            weights=weights,
            config=config,
            language=language,
            speaker=speaker,
            instruct=instruct,
            speaker_embed=speaker_embed,
            non_streaming_mode=non_streaming_mode,
            ref_text=ref_text,
            ref_codes=ref_codes,
        )
    print(f"[perf] build_input:          {time.perf_counter() - t0:.3f}s")

    inputs_embeds = talker_input["inputs_embeds"]
    trailing_text_hidden = talker_input["trailing_text_hidden"]
    tts_pad_embed = talker_input["tts_pad_embed"]

    mrope_cos = consts.talker_mrope_cos
    mrope_sin = consts.talker_mrope_sin
    rope_cos = consts.cp_rope_cos
    rope_sin = consts.cp_rope_sin

    # prefill talker with all input
    S = inputs_embeds.shape[1]
    empty_talker_kvs = build_empty_cache_kwargs(
        consts.talker_n_layers,
        consts.talker_n_kv_heads,
        consts.talker_head_dim,
        talker._weight_dtype,
        inputs_embeds.device,
    )

    print(f"[perf] talker prefill ({S} tokens)...")
    t0 = time.perf_counter()
    cos, sin = slice_mrope(mrope_cos, mrope_sin, start=0, length=S)
    outputs = talker(inputs_embeds, cos, sin, **empty_talker_kvs)
    print(f"[perf] talker prefill:       {time.perf_counter() - t0:.3f}s")
    hidden = outputs.hidden
    talker_kvs = extract_present_cache_kwargs(outputs, consts.talker_n_layers)
    cache_position = S

    logits = outputs.logits[:, -1, :].clone()
    logits[:, consts.suppress_mask] = float("-inf")
    first_code = sample_token(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)

    all_step_codes = []
    past_first_codes = []
    past_hidden = hidden[:, -1:, :]

    t_cp_total = 0.0
    t_talker_total = 0.0
    t_step0_cp = None
    t_step0_talker = None

    step = 0
    while step < max_new_tokens:
        first_code_val = int(first_code.item())
        if first_code_val == consts.codec_eos_id:
            break

        past_first_codes.append(first_code_val)
        first_code_embed = talker.codec_embedding(first_code)

        t_cp0 = time.perf_counter()
        sub_codes, all_embeds_sum = generate_sub_codes(
            first_code_embed,
            past_hidden,
            cp,
            rope_cos,
            rope_sin,
            consts,
            dict(
                do_sample=subtalker_dosample,
                top_k=subtalker_top_k,
                top_p=subtalker_top_p,
                temperature=subtalker_temperature,
            ),
        )
        t_cp = time.perf_counter() - t_cp0
        t_cp_total += t_cp
        if step == 0:
            t_step0_cp = t_cp

        all_step_codes.append(torch.cat([first_code, sub_codes], dim=-1))

        next_embed = all_embeds_sum
        if step < trailing_text_hidden.shape[1]:
            next_embed = next_embed + trailing_text_hidden[:, step : step + 1, :]
        else:
            next_embed = next_embed + tts_pad_embed

        cos, sin = slice_mrope(mrope_cos, mrope_sin, start=cache_position, length=1)
        t_td0 = time.perf_counter()
        outputs = talker(next_embed, cos, sin, **talker_kvs)
        t_talker = time.perf_counter() - t_td0
        t_talker_total += t_talker
        if step == 0:
            t_step0_talker = t_talker

        hidden = outputs.hidden
        talker_kvs = extract_present_cache_kwargs(outputs, consts.talker_n_layers)

        cache_position += 1
        step += 1

        logits = outputs.logits[:, -1, :].clone()
        logits[:, consts.suppress_mask] = float("-inf")
        if repetition_penalty != 1.0 and len(past_first_codes) > 0:
            past_tensor = torch.tensor([past_first_codes], dtype=torch.long, device=logits.device)
            logits = apply_repetition_penalty(logits, past_tensor, repetition_penalty)
        first_code = sample_token(logits, do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
        past_hidden = hidden[:, -1:, :]

    n_steps = step
    if n_steps > 0:
        print(f"[perf] decode loop ({n_steps} frames):")
        print(f"[perf]   code_predictor   step0={t_step0_cp:.3f}s  avg={t_cp_total/n_steps:.3f}s  total={t_cp_total:.3f}s")
        print(f"[perf]   talker decode    step0={t_step0_talker:.3f}s  avg={t_talker_total/n_steps:.3f}s  total={t_talker_total:.3f}s")
        per_frame = (t_cp_total + t_talker_total) / n_steps
        print(f"[perf]   per frame:       {per_frame:.3f}s  ({1/per_frame:.1f} frames/s)")

    if len(all_step_codes) == 0:
        all_codes = torch.empty((1, 0, consts.num_code_groups), dtype=torch.long, device=inputs_embeds.device)
    else:
        all_codes = torch.stack(all_step_codes, dim=1)  # (1, steps, num_code_groups)

    speech_cfg = speech_tokenizer_state["config"]
    dec_cfg = speech_cfg.get("decoder_config", speech_cfg)
    st_weights = speech_tokenizer_state["weights"]

    if all_codes.numel() == 0:
        empty_output = np.zeros((0,), dtype=np.float32)
        return empty_output, dec_cfg.get("output_sample_rate", 24000)

    # split_rvq_decode expects (batch, num_quantizers, T)
    t0 = time.perf_counter()
    rvq_codes = all_codes[0].transpose(0, 1).unsqueeze(0).long()
    latent = split_rvq_decode(
        rvq_codes,
        st_weights,
        "decoder.quantizer",
        dec_cfg.get("num_semantic_quantizers", 1),
        dec_cfg.get("num_quantizers", 16),
        dec_cfg,
    )
    print(f"[perf] RVQ decode:           {time.perf_counter() - t0:.3f}s")

    t0 = time.perf_counter()
    wav = speech_decoder(latent.to(hidden.device))
    print(f"[perf] speech_decoder:       {time.perf_counter() - t0:.3f}s")

    return wav.squeeze(0).squeeze(0).detach().cpu().numpy(), dec_cfg.get("output_sample_rate", 24000)
