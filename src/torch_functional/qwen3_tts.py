import json
import os
import math
import wave as wave_mod
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer

from qwen3_tts_tokenizer import rms_norm, swiglu_ffn, decode, encode

def list_weight_keys(state):
    """Print all weight keys grouped by prefix for debugging."""
    keys = sorted(state["weights"].keys())
    prefixes = {}
    for k in keys:
        prefix = k.rsplit(".", 1)[0] if "." in k else k
        top = ".".join(prefix.split(".")[:3])
        prefixes.setdefault(top, []).append(k)
    for prefix, ks in sorted(prefixes.items()):
        print(f"\n{prefix} ({len(ks)} params):")
        for k in ks[:5]:
            print(f"  {k}: {state['weights'][k].shape}")
        if len(ks) > 5:
            print(f"  ... and {len(ks) - 5} more")


def load_model(model_path: str) -> dict:
    """Load TTS model weights, config, text tokenizer, and speech tokenizer.

    Args:
        model_path: Path to the Qwen3-TTS model directory.

    Returns:
        dict with keys: 'weights', 'config', 'tokenizer' (AutoTokenizer),
        'speech_tokenizer' (from load_tokenizer), 'generate_config'.
    """
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    weights = {}
    for fname in sorted(os.listdir(model_path)):
        if fname.endswith(".safetensors"):
            shard = load_file(os.path.join(model_path, fname), device="cpu")
            weights.update(shard)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    gen_config_path = os.path.join(model_path, "generation_config.json")
    generate_config = {}
    if os.path.exists(gen_config_path):
        with open(gen_config_path, "r") as f:
            generate_config = json.load(f)

    # Load speech tokenizer
    from qwen3_tts_tokenizer import load_tokenizer
    speech_tokenizer = load_tokenizer(model_path)

    # Extract speaker encoder weights if present (Base model for voice clone)
    se_prefix = "speaker_encoder."
    speaker_encoder_weights = {}
    for k, v in weights.items():
        if k.startswith(se_prefix):
            speaker_encoder_weights[k[len(se_prefix):]] = v

    return {
        "weights": weights,
        "config": config,
        "tokenizer": tokenizer,
        "speech_tokenizer": speech_tokenizer,
        "generate_config": generate_config,
        "speaker_encoder_weights": speaker_encoder_weights if speaker_encoder_weights else None,
    }


# ---------------------------------------------------------------------------
# Talker Transformer Forward Pass
# ---------------------------------------------------------------------------



def multimodal_rotary_embedding(seq_len, head_dim, mrope_section, theta=1000000.0):
    """Compute cos/sin for 3D multi-modal RoPE.

    Mirrors Qwen3TTSTalkerRotaryEmbedding with default rope_type and
    identical position_ids across all 3 modalities (text-only case).

    Args:
        seq_len: sequence length.
        head_dim: attention head dimension.
        mrope_section: list of ints defining section sizes for each modality.
        theta: RoPE base frequency.

    Returns:
        cos (3, seq_len, head_dim), sin (3, seq_len, head_dim).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)

    cos = emb.cos().unsqueeze(0).expand(3, -1, -1)
    sin = emb.sin().unsqueeze(0).expand(3, -1, -1)
    return cos, sin


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, mrope_interleaved=False):
    """Apply 3D RoPE to q, k using section-based splitting.

    Matches upstream apply_multimodal_rotary_pos_emb exactly, supporting
    both interleaved and non-interleaved modes.

    Args:
        q, k: (batch, heads, seq, head_dim).
        cos, sin: (3, seq, head_dim) or (3, batch, seq, head_dim).
        mrope_section: list of ints (e.g. [24, 20, 20]).
        mrope_interleaved: whether to use interleaved merging.
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    if mrope_interleaved:
        dim = cos.shape[-1]
        modality_num = len(mrope_section)

        def apply_interleaved_rope(x, mod_num):
            x_t = x[0].clone()
            for i, n in enumerate(mrope_section[1:], 1):
                beg_idx = i
                end_idx = n * mod_num
                x_t[..., beg_idx:end_idx:mod_num] = x[i, ..., beg_idx:end_idx:mod_num]
            return x_t

        cos = torch.cat(
            [apply_interleaved_rope(cos[..., :dim // 2], modality_num)] * 2, dim=-1
        )
        sin = torch.cat(
            [apply_interleaved_rope(sin[..., :dim // 2], modality_num)] * 2, dim=-1
        )
    else:
        sections = mrope_section * 2  # double for cos/sin halves
        cos = torch.cat(
            [m[i % 3] for i, m in enumerate(cos.split(sections, dim=-1))], dim=-1
        )
        sin = torch.cat(
            [m[i % 3] for i, m in enumerate(sin.split(sections, dim=-1))], dim=-1
        )

    # After merge: (seq, head_dim). Unsqueeze to (1, 1, seq, head_dim) for
    # broadcasting with q, k of shape (batch, heads, seq, head_dim).
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def talker_attention(hidden, weights, prefix, config, cos, sin, kv_cache=None, cache_position=None):
    """Talker attention layer with QK-norm and 3D RoPE.

    Mirrors Qwen3TTSTalkerAttention.forward.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: flat weight dict.
        prefix: e.g. 'talker.model.layers.0.self_attn'.
        config: talker_config dict.
        cos, sin: (3, max_len, head_dim) precomputed RoPE.
        kv_cache: optional (k, v) tuple for incremental decoding.
        cache_position: int or tensor, position offset for KV cache.

    Returns:
        (batch, seq, hidden_size), updated (k, v) cache.
    """
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // n_heads)
    eps = config.get("rms_norm_eps", 1e-6)
    rope_scaling = config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [])
    mrope_interleaved = rope_scaling.get("interleaved", False)

    q = F.linear(hidden, weights[f"{prefix}.q_proj.weight"])
    k = F.linear(hidden, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden, weights[f"{prefix}.v_proj.weight"])

    batch, seq, _ = hidden.shape
    q = q.view(batch, seq, n_heads, head_dim)
    k = k.view(batch, seq, n_kv_heads, head_dim)

    # QK-norm (RMSNorm on head_dim, matching upstream q_norm / k_norm)
    q = rms_norm(q, weights[f"{prefix}.q_norm.weight"], eps)
    k = rms_norm(k, weights[f"{prefix}.k_norm.weight"], eps)

    q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
    k = k.transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    # Select RoPE for current positions
    if cache_position is not None:
        if isinstance(cache_position, int):
            pos_slice = slice(cache_position, cache_position + seq)
        else:
            pos_slice = cache_position
        cur_cos = cos[:, pos_slice]
        cur_sin = sin[:, pos_slice]
    else:
        cur_cos = cos[:, :seq]
        cur_sin = sin[:, :seq]

    # Apply 3D RoPE
    q, k = apply_multimodal_rotary_pos_emb(q, k, cur_cos, cur_sin, mrope_section, mrope_interleaved)

    # KV cache
    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=2)
        v = torch.cat([kv_cache[1], v], dim=2)
    new_cache = (k, v)

    # GQA expansion
    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    full_seq = k.shape[2]
    if seq > 1:
        mask = torch.triu(
            torch.full((seq, full_seq), float('-inf'), device=attn.device),
            diagonal=full_seq - seq + 1,
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
    out = F.linear(out, weights[f"{prefix}.o_proj.weight"])

    return out, new_cache


def talker_layer(hidden, weights, prefix, config, cos, sin, kv_cache=None, cache_position=None):
    """Single Talker decoder layer.

    Mirrors Qwen3TTSTalkerDecoderLayer.forward.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: flat weight dict.
        prefix: e.g. 'talker.model.layers.0'.
        config: talker_config dict.
        cos, sin: precomputed RoPE.
        kv_cache: optional (k, v) tuple.
        cache_position: position offset for KV cache.

    Returns:
        hidden, updated (k, v) cache.
    """
    eps = config.get("rms_norm_eps", 1e-6)

    # Pre-norm + attention
    normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)
    attn_out, new_cache = talker_attention(
        normed, weights, f"{prefix}.self_attn", config, cos, sin, kv_cache, cache_position)
    hidden = hidden + attn_out

    # Pre-norm + SwiGLU FFN
    normed = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
    ffn_out = swiglu_ffn(
        normed,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )
    hidden = hidden + ffn_out

    return hidden, new_cache


def talker_forward(inputs_embeds, weights, config, kv_caches=None, cache_position=None):
    """Full Talker transformer forward pass.

    Mirrors Qwen3TTSTalkerModel.forward: runs all decoder layers with
    3D multi-modal RoPE and final RMSNorm.

    Args:
        inputs_embeds: (batch, seq, hidden_size) input embeddings.
        weights: flat weight dict (keys like 'talker.model.layers.{i}...').
        config: talker_config dict from config.json.
        kv_caches: optional list of (k, v) tuples, one per layer.
        cache_position: int or tensor, position offset for KV cache.

    Returns:
        hidden: (batch, seq, hidden_size) final hidden states.
        new_caches: list of (k, v) tuples, one per layer.
    """
    n_layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    rope_scaling = config.get("rope_scaling", {})
    mrope_section = rope_scaling.get("mrope_section", [])
    theta = config.get("rope_theta", 1000000.0)

    # Precompute RoPE for max possible length
    max_len = config.get("max_position_embeddings", 32768)
    cos, sin = multimodal_rotary_embedding(max_len, head_dim, mrope_section, theta)

    # Match input dtype to weights
    weight_dtype = weights[f"talker.model.layers.0.input_layernorm.weight"].dtype
    hidden = inputs_embeds.to(weight_dtype)
    new_caches = []
    for i in range(n_layers):
        cache = kv_caches[i] if kv_caches else None
        hidden, new_cache = talker_layer(
            hidden, weights, f"talker.model.layers.{i}", config,
            cos, sin, cache, cache_position)
        new_caches.append(new_cache)

    eps = config.get("rms_norm_eps", 1e-6)
    hidden = rms_norm(hidden, weights["talker.model.norm.weight"], eps)

    return hidden, new_caches


# ---------------------------------------------------------------------------
# Code Predictor (Sub-Talker) — generates codebooks 2..N given codebook 1
# ---------------------------------------------------------------------------

def standard_rotary_embedding(seq_len, head_dim, theta=1000000.0):
    """Compute cos/sin for standard 1-D RoPE (not 3-D multimodal).

    Used by the code predictor, which uses Qwen3TTSRotaryEmbedding
    (standard positional RoPE, not the talker's multimodal variant).

    Args:
        seq_len: maximum sequence length.
        head_dim: per-head dimension.
        theta: RoPE base frequency.

    Returns:
        cos (seq_len, head_dim), sin (seq_len, head_dim).
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)  # (seq_len, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)   # (seq_len, head_dim)
    return emb.cos(), emb.sin()


def apply_standard_rotary_pos_emb(q, k, cos, sin):
    """Apply standard 1-D RoPE to q and k.

    Matches upstream ``apply_rotary_pos_emb`` with ``unsqueeze_dim=1``.

    Args:
        q, k: (batch, heads, seq, head_dim).
        cos, sin: (seq, head_dim) — will be broadcast to match q/k.
    """
    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    # Unsqueeze to (1, 1, seq, head_dim) for broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def code_predictor_attention(hidden, weights, prefix, config, cos, sin,
                             kv_cache=None, cache_position=None):
    """Code predictor attention with QK-norm and standard 1-D RoPE.

    Mirrors Qwen3TTSAttention.forward (used by the code predictor's
    Qwen3TTSDecoderLayer), which differs from talker attention in using
    standard RoPE instead of 3-D multimodal RoPE.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: flat weight dict.
        prefix: e.g. 'talker.code_predictor.model.layers.0.self_attn'.
        config: code_predictor_config dict.
        cos, sin: (max_len, head_dim) precomputed standard RoPE.
        kv_cache: optional (k, v) tuple for incremental decoding.
        cache_position: int or tensor, position offset for KV cache.

    Returns:
        (batch, seq, hidden_size), updated (k, v) cache.
    """
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // n_heads)
    eps = config.get("rms_norm_eps", 1e-6)

    q = F.linear(hidden, weights[f"{prefix}.q_proj.weight"])
    k = F.linear(hidden, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden, weights[f"{prefix}.v_proj.weight"])

    batch, seq, _ = hidden.shape
    q = q.view(batch, seq, n_heads, head_dim)
    k = k.view(batch, seq, n_kv_heads, head_dim)

    # QK-norm (RMSNorm per head, matching upstream q_norm / k_norm)
    q = rms_norm(q, weights[f"{prefix}.q_norm.weight"], eps)
    k = rms_norm(k, weights[f"{prefix}.k_norm.weight"], eps)

    q = q.transpose(1, 2)  # (batch, heads, seq, head_dim)
    k = k.transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    # Select RoPE for current positions
    if cache_position is not None:
        if isinstance(cache_position, int):
            pos_slice = slice(cache_position, cache_position + seq)
        else:
            pos_slice = cache_position
        cur_cos = cos[pos_slice]
        cur_sin = sin[pos_slice]
    else:
        cur_cos = cos[:seq]
        cur_sin = sin[:seq]

    # Apply standard 1-D RoPE
    q, k = apply_standard_rotary_pos_emb(q, k, cur_cos, cur_sin)

    # KV cache
    if kv_cache is not None:
        k = torch.cat([kv_cache[0], k], dim=2)
        v = torch.cat([kv_cache[1], v], dim=2)
    new_cache = (k, v)

    # GQA expansion
    if n_kv_heads < n_heads:
        repeats = n_heads // n_kv_heads
        k = k.repeat_interleave(repeats, dim=1)
        v = v.repeat_interleave(repeats, dim=1)

    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    full_seq = k.shape[2]
    if seq > 1:
        mask = torch.triu(
            torch.full((seq, full_seq), float('-inf'), device=attn.device),
            diagonal=full_seq - seq + 1,
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
    out = torch.matmul(attn, v)
    out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
    out = F.linear(out, weights[f"{prefix}.o_proj.weight"])

    return out, new_cache


def code_predictor_layer(hidden, weights, prefix, config, cos, sin,
                         kv_cache=None, cache_position=None):
    """Single code predictor decoder layer.

    Mirrors Qwen3TTSDecoderLayer.forward (used by the code predictor model).

    Args:
        hidden: (batch, seq, hidden_size).
        weights: flat weight dict.
        prefix: e.g. 'talker.code_predictor.model.layers.0'.
        config: code_predictor_config dict.
        cos, sin: precomputed standard RoPE.
        kv_cache: optional (k, v) tuple.
        cache_position: position offset for KV cache.

    Returns:
        hidden, updated (k, v) cache.
    """
    eps = config.get("rms_norm_eps", 1e-6)

    # Pre-norm + attention
    normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)
    attn_out, new_cache = code_predictor_attention(
        normed, weights, f"{prefix}.self_attn", config, cos, sin,
        kv_cache, cache_position)
    hidden = hidden + attn_out

    # Pre-norm + SwiGLU FFN
    normed = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
    ffn_out = swiglu_ffn(
        normed,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )
    hidden = hidden + ffn_out

    return hidden, new_cache


def code_predictor_forward(inputs_embeds, weights, config, generation_step=0,
                           kv_caches=None, cache_position=None):
    """Code predictor forward pass.

    Mirrors Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward.
    Runs inputs_embeds through small_to_mtp_projection, then N decoder layers
    with standard 1-D RoPE, final RMSNorm, and a per-step lm_head.

    Args:
        inputs_embeds: (batch, seq, talker_hidden_size) — in talker's
            embedding space (e.g. dim 2048).
        weights: flat weight dict.
        config: code_predictor_config dict (from config.json).
        generation_step: which codebook is being predicted. 0 = codebook 2,
            1 = codebook 3, etc. Used to select the correct lm_head.
        kv_caches: list of (k, v) caches, one per layer.
        cache_position: int or tensor, position offset for KV cache.

    Returns:
        logits: (batch, seq, vocab_size).
        new_kv_caches: list of (k, v) tuples, one per layer.
    """
    prefix = "talker.code_predictor"
    n_layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
    theta = config.get("rope_theta", 1000000.0)
    eps = config.get("rms_norm_eps", 1e-6)

    # Project from talker hidden_size to code predictor hidden_size (if sizes differ)
    proj_key = f"{prefix}.small_to_mtp_projection.weight"
    if proj_key in weights:
        hidden = F.linear(
            inputs_embeds,
            weights[proj_key],
            weights[f"{prefix}.small_to_mtp_projection.bias"],
        )
    else:
        hidden = inputs_embeds

    # Match dtype to layer weights
    weight_dtype = weights[f"{prefix}.model.layers.0.input_layernorm.weight"].dtype
    hidden = hidden.to(weight_dtype)

    # Precompute standard RoPE
    max_len = config.get("max_position_embeddings", 32768)
    cos, sin = standard_rotary_embedding(max_len, head_dim, theta)

    new_caches = []
    for i in range(n_layers):
        cache = kv_caches[i] if kv_caches else None
        hidden, new_cache = code_predictor_layer(
            hidden, weights, f"{prefix}.model.layers.{i}", config,
            cos, sin, cache, cache_position)
        new_caches.append(new_cache)

    # Final RMSNorm
    hidden = rms_norm(hidden, weights[f"{prefix}.model.norm.weight"], eps)

    # Per-step lm_head (no bias)
    logits = F.linear(hidden, weights[f"{prefix}.lm_head.{generation_step}.weight"])

    return logits, new_caches


def apply_repetition_penalty(logits, past_tokens, penalty=1.05):
    """Apply repetition penalty to logits based on past tokens.

    Args:
        logits: (batch, vocab_size).
        past_tokens: (batch, seq) int64.
        penalty: float > 1.0.
    """
    if penalty == 1.0 or past_tokens.numel() == 0:
        return logits
    score = torch.gather(logits, 1, past_tokens)
    score = torch.where(score < 0, score * penalty, score / penalty)
    logits.scatter_(1, past_tokens, score)
    return logits


def sample_token(logits, do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
    """Sample a token from logits.

    Placeholder implementation — will be fully implemented in Task 10
    (Sampling Utilities). Supports greedy and basic top-k/top-p sampling.

    Args:
        logits: (batch, vocab_size) unnormalized logits for the last position.
        do_sample: if False, use greedy (argmax).
        top_k: keep only top-k logits before sampling.
        top_p: nucleus sampling threshold.
        temperature: softmax temperature.

    Returns:
        token_ids: (batch, 1) sampled token indices.
    """
    if not do_sample:
        return logits.argmax(dim=-1, keepdim=True)

    # Temperature scaling
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0 and top_k < logits.shape[-1]:
        top_k_vals, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_vals[:, -1:]
        logits = logits.masked_fill(logits < threshold, float('-inf'))

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float('-inf')
        # Scatter back to original order
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    token_ids = torch.multinomial(probs, num_samples=1)
    return token_ids


def generate_sub_codes(first_code_embed, past_hidden, weights, talker_config,
                       do_sample=True, top_k=50, top_p=1.0, temperature=0.9):
    """Generate codebooks 2-N autoregressively using the code predictor.

    Mirrors the code predictor generation flow from
    ``Qwen3TTSTalkerForConditionalGeneration.forward`` (generation branch).

    The code predictor is called with an initial input of
    ``cat(past_hidden, first_code_embed)`` (both in talker embedding space),
    then autoregressively generates num_code_groups-1 additional codebook
    tokens.

    Args:
        first_code_embed: (batch, 1, talker_hidden_size) — embedding of the
            first codebook code (from talker's embed_tokens).
        past_hidden: (batch, 1, talker_hidden_size) — last hidden state from
            the talker transformer (before codec_head, after final norm).
        weights: flat weight dict.
        talker_config: talker config dict (the full talker config section
            from config.json, which contains code_predictor_config and
            num_code_groups).

    Returns:
        all_codes: (batch, num_code_groups-1) int64 tensor — predicted codes
            for codebooks 2..N.
        all_embeds_sum: (batch, 1, talker_hidden_size) — sum of all codebook
            embeddings (first code + predicted codes), used to form the next
            talker input.
    """
    cp_config = talker_config["code_predictor_config"]
    num_code_groups = talker_config.get("num_code_groups", 16)
    num_sub_codes = num_code_groups - 1  # codebooks to predict (e.g. 15)
    prefix = "talker.code_predictor"

    # --- Prefill: feed [past_hidden, first_code_embed] ------------------
    # Both are in talker's hidden space (e.g. 2048 dim).
    # The code predictor's forward() will project them through
    # small_to_mtp_projection internally.
    prefill_embeds = torch.cat([past_hidden, first_code_embed], dim=1)  # (B, 2, talker_hidden)

    # Prefill: generation_step for the first prediction is 0 (predicting codebook 2)
    logits, kv_caches = code_predictor_forward(
        prefill_embeds, weights, cp_config,
        generation_step=0, kv_caches=None, cache_position=None)

    # The prefill output has shape (batch, 2, vocab). We take the last position.
    step_logits = logits[:, -1, :]  # (batch, vocab_size)

    all_codes = []
    # Accumulate sum of embeddings in talker's space (for next talker input)
    all_embeds_sum = first_code_embed.clone()  # start with first code embed

    # Sample first sub-code (codebook 2)
    token_id = sample_token(step_logits, do_sample, top_k, top_p, temperature)
    all_codes.append(token_id)  # (batch, 1)

    # Embed the sampled token using codec_embedding[0] (codebook 2's embedding)
    # codec_embedding maps to talker hidden space (2048)
    token_embed = F.embedding(
        token_id, weights[f"{prefix}.model.codec_embedding.0.weight"]
    )  # (batch, 1, talker_hidden_size)
    all_embeds_sum = all_embeds_sum + token_embed

    # --- Autoregressive generation for codebooks 3..N -------------------
    cache_pos = 2  # next position after the 2-token prefill
    for step in range(1, num_sub_codes):
        # Embed the previous token using codec_embedding[step-1]
        # (the forward() method uses generation_steps-1 as the embedding index)
        # token_embed is already computed from the previous iteration
        step_embeds = token_embed  # (batch, 1, talker_hidden_size)

        logits, kv_caches = code_predictor_forward(
            step_embeds, weights, cp_config,
            generation_step=step, kv_caches=kv_caches,
            cache_position=cache_pos)

        step_logits = logits[:, -1, :]  # (batch, vocab_size)
        token_id = sample_token(step_logits, do_sample, top_k, top_p, temperature)
        all_codes.append(token_id)  # (batch, 1)

        # Embed the sampled token for the next step and accumulate
        token_embed = F.embedding(
            token_id, weights[f"{prefix}.model.codec_embedding.{step}.weight"]
        )  # (batch, 1, talker_hidden_size)
        all_embeds_sum = all_embeds_sum + token_embed

        cache_pos += 1

    all_codes = torch.cat(all_codes, dim=-1)  # (batch, num_sub_codes)
    return all_codes, all_embeds_sum


# ---------------------------------------------------------------------------
# Control Signal Construction + Autoregressive Generation Loop
# ---------------------------------------------------------------------------


def _text_projection(hidden, weights):
    """Apply the talker's text_projection ResizeMLP.

    ResizeMLP: Linear(fc1) -> SiLU -> Linear(fc2), both with bias.

    Mirrors ``Qwen3TTSTalkerResizeMLP.forward``.

    Args:
        hidden: (batch, seq, text_hidden_size).
        weights: flat weight dict.

    Returns:
        (batch, seq, hidden_size).
    """
    h = F.linear(hidden, weights["talker.text_projection.linear_fc1.weight"],
                 weights["talker.text_projection.linear_fc1.bias"])
    h = F.silu(h)
    h = F.linear(h, weights["talker.text_projection.linear_fc2.weight"],
                 weights["talker.text_projection.linear_fc2.bias"])
    return h


def _embed_text_ids(token_ids, weights):
    """Look up text_embedding and apply text_projection.

    Equivalent to:
        self.talker.text_projection(self.talker.get_text_embeddings()(token_ids))

    Args:
        token_ids: (batch, seq) int64 — text token IDs from the tokenizer.
        weights: flat weight dict.

    Returns:
        (batch, seq, hidden_size).
    """
    text_emb = F.embedding(token_ids, weights["talker.model.text_embedding.weight"])
    return _text_projection(text_emb, weights)


def _embed_codec_ids(token_ids, weights):
    """Look up codec_embedding (the talker's input embedding for codec tokens).

    Equivalent to:
        self.talker.get_input_embeddings()(token_ids)

    Args:
        token_ids: (batch, seq) int64 — codec token IDs.
        weights: flat weight dict.

    Returns:
        (batch, seq, hidden_size).
    """
    return F.embedding(token_ids, weights["talker.model.codec_embedding.weight"])


def _embed_ref_codes(ref_codes, weights, num_code_groups):
    """Sum all codebook group embeddings per timestep for reference codes.

    Args:
        ref_codes: (T, num_code_groups) int64 — reference audio codes.
        weights: flat weight dict.
        num_code_groups: int (typically 16).

    Returns:
        (1, T, hidden_size) float tensor.
    """
    embeds = []
    # Group 0: talker's codec_embedding
    embeds.append(F.embedding(ref_codes[:, 0:1],
                              weights["talker.model.codec_embedding.weight"]))
    # Groups 1..N-1: code_predictor's codec_embedding
    for i in range(1, num_code_groups):
        embeds.append(F.embedding(
            ref_codes[:, i:i+1],
            weights[f"talker.code_predictor.model.codec_embedding.{i-1}.weight"]))
    # (T, num_code_groups, hidden) -> sum over groups -> (T, hidden)
    return torch.cat(embeds, dim=1).sum(dim=1).unsqueeze(0)  # (1, T, hidden)


def generate_icl_prompt(text_ids, ref_text_ids, ref_codes, tts_pad_embed,
                        tts_eos_embed, weights, config, non_streaming_mode=True):
    """Construct ICL (In-Context Learning) prompt embeddings.

    Mirrors upstream lines 1968-2019. Non-streaming mode:
    1. Text: embed(cat(ref_text_ids, target_text_ids)) + tts_eos
    2. Codec: codec_bos + embed_ref_codes(ref_codes)
    3. ICL embed: interleave text+codec_pad and codec+tts_pad

    Args:
        text_ids: (1, T_text) int64 — target text token IDs (already sliced).
        ref_text_ids: (1, R) int64 — reference text token IDs (already sliced).
        ref_codes: (T_code, num_code_groups) int64 — reference audio codes.
        tts_pad_embed: (1, 1, hidden) — tts_pad embedding.
        tts_eos_embed: (1, 1, hidden) — tts_eos embedding.
        weights: flat weight dict.
        config: top-level config dict.
        non_streaming_mode: bool.

    Returns:
        (icl_input_embed, trailing_text_hidden):
            icl_input_embed: (1, seq, hidden) — combined text+codec ICL embedding.
            trailing_text_hidden: (1, 1, hidden) — for generation loop.
    """
    talker_config = config["talker_config"]
    num_code_groups = talker_config.get("num_code_groups", 16)

    # 1. Text side: embed(cat(ref_text, target_text)) + tts_eos
    all_text_ids = torch.cat([ref_text_ids, text_ids], dim=1)  # (1, R+T)
    text_embeds = _embed_text_ids(all_text_ids, weights)  # (1, R+T, hidden)
    text_with_eos = torch.cat([text_embeds, tts_eos_embed], dim=1)  # (1, R+T+1, hidden)

    # 2. Codec side: codec_bos + embed_ref_codes
    codec_bos_embed = _embed_codec_ids(
        torch.tensor([[talker_config["codec_bos_id"]]], dtype=torch.long), weights
    )  # (1, 1, hidden)
    ref_code_embeds = _embed_ref_codes(ref_codes, weights, num_code_groups)  # (1, T_code, hidden)
    codec_with_bos = torch.cat([codec_bos_embed, ref_code_embeds], dim=1)  # (1, T_code+1, hidden)

    # 3. Build ICL embed: text rows get codec_pad added, codec rows get tts_pad added
    text_len = text_with_eos.shape[1]
    codec_len = codec_with_bos.shape[1]

    # Pad text side with codec_pad embeddings
    codec_pad_for_text = _embed_codec_ids(
        torch.full((1, text_len), talker_config["codec_pad_id"], dtype=torch.long), weights
    )  # (1, text_len, hidden)
    text_block = text_with_eos + codec_pad_for_text  # (1, text_len, hidden)

    # Pad codec side with tts_pad embeddings
    tts_pad_for_codec = tts_pad_embed.expand(-1, codec_len, -1)  # (1, codec_len, hidden)
    codec_block = codec_with_bos + tts_pad_for_codec  # (1, codec_len, hidden)

    # Concatenate: text block then codec block
    icl_input_embed = torch.cat([text_block, codec_block], dim=1)  # (1, text_len+codec_len, hidden)

    # Trailing text hidden: just tts_pad (all text is in prefill for non-streaming)
    trailing_text_hidden = tts_pad_embed  # (1, 1, hidden)

    return icl_input_embed, trailing_text_hidden


def build_talker_input(
    text,
    weights,
    config,
    tokenizer,
    language="auto",
    speaker=None,
    instruct=None,
    speaker_embed=None,
    non_streaming_mode=True,
    ref_text=None,
    ref_codes=None,
):
    """Construct the full input embedding sequence for the Talker model.

    This mirrors ``Qwen3TTSForConditionalGeneration.generate()`` lines 2068-2234
    in the upstream code.  It:

    1. Tokenizes the text using the chat template format:
       ``<|im_start|>assistant\\n{text}<|im_end|>\\n<|im_start|>assistant\\n``
    2. Optionally tokenizes instruct text (for VoiceDesign/CustomVoice):
       ``<|im_start|>user\\n{instruct}<|im_end|>\\n``
    3. Creates special token embeddings (tts_bos, tts_eos, tts_pad)
    4. Creates codec control token embeddings (think/nothink, language_id,
       optional speaker embedding)
    5. Combines everything in the correct order

    Args:
        text: str — the text to synthesize.
        weights: flat weight dict.
        config: top-level config dict (config.json).
        tokenizer: AutoTokenizer instance.
        language: language name (str), e.g. "auto", "chinese", "english".
        speaker: speaker name (str) or None.
        instruct: optional instruction text (str) or None.
        speaker_embed: optional (hidden_size,) tensor for voice clone speaker.
        non_streaming_mode: if True, prepend all text tokens in the prefill.
        ref_text: optional str — reference text for ICL mode voice cloning.
        ref_codes: optional (T, num_code_groups) int64 — reference audio codes.

    Returns:
        dict with:
            'inputs_embeds': (1, seq_len, hidden_size) — full talker input.
            'trailing_text_hidden': (1, T, hidden_size) — remaining text to
                feed one-by-one during generation.
            'tts_pad_embed': (1, 1, hidden_size) — cached for use in gen loop.
    """
    talker_config = config["talker_config"]

    # ── 1. Tokenize the main text ──────────────────────────────────────
    assistant_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_id = tokenizer(assistant_text, return_tensors="pt")["input_ids"]  # (1, L)

    # ── 2. Optionally tokenize instruct text ───────────────────────────
    instruct_embed = None
    if instruct is not None and instruct != "":
        instruct_text = f"<|im_start|>user\n{instruct}<|im_end|>\n"
        instruct_ids = tokenizer(instruct_text, return_tensors="pt")["input_ids"]
        instruct_embed = _embed_text_ids(instruct_ids, weights)

    # ── 3. Create special TTS token embeddings ─────────────────────────
    tts_bos_id = config["tts_bos_token_id"]
    tts_eos_id = config["tts_eos_token_id"]
    tts_pad_id = config["tts_pad_token_id"]

    special_ids = torch.tensor([[tts_bos_id, tts_eos_id, tts_pad_id]], dtype=torch.long)
    special_embeds = _embed_text_ids(special_ids, weights)  # (1, 3, hidden)
    tts_bos_embed = special_embeds[:, 0:1, :]  # (1, 1, hidden)
    tts_eos_embed = special_embeds[:, 1:2, :]  # (1, 1, hidden)
    tts_pad_embed = special_embeds[:, 2:3, :]  # (1, 1, hidden)

    # ── 4. Resolve language ID ─────────────────────────────────────────
    codec_language_id_map = talker_config.get("codec_language_id", {})
    spk_is_dialect = talker_config.get("spk_is_dialect", {})

    if language.lower() == "auto":
        language_id = None
    else:
        lang_key = language.lower()
        if lang_key not in codec_language_id_map:
            raise ValueError(f"Language '{language}' not supported. "
                             f"Available: {list(codec_language_id_map.keys())}")
        language_id = codec_language_id_map[lang_key]

    # Handle dialect override for Chinese speakers
    if (language.lower() in ["chinese", "auto"]
            and speaker is not None and speaker != ""
            and spk_is_dialect.get(speaker.lower(), False) is not False):
        dialect = spk_is_dialect[speaker.lower()]
        language_id = codec_language_id_map[dialect]

    # ── 5. Resolve speaker embedding ───────────────────────────────────
    spk_embed_tensor = None
    if speaker_embed is not None:
        # Voice clone mode — speaker_embed is already a (hidden_size,) tensor
        spk_embed_tensor = speaker_embed.view(1, 1, -1)
    elif speaker is not None and speaker != "":
        spk_id_map = talker_config.get("spk_id", {})
        spk_key = speaker.lower()
        if spk_key not in spk_id_map:
            raise ValueError(f"Speaker '{speaker}' not supported. "
                             f"Available: {list(spk_id_map.keys())}")
        spk_id_val = spk_id_map[spk_key]
        spk_embed_tensor = _embed_codec_ids(
            torch.tensor([[spk_id_val]], dtype=torch.long), weights
        )  # (1, 1, hidden)

    # ── 6. Build codec control token sequence ──────────────────────────
    if language_id is None:
        # No-think mode
        codec_prefill_ids = [
            talker_config["codec_nothink_id"],
            talker_config["codec_think_bos_id"],
            talker_config["codec_think_eos_id"],
        ]
    else:
        # Think mode with language ID
        codec_prefill_ids = [
            talker_config["codec_think_id"],
            talker_config["codec_think_bos_id"],
            language_id,
            talker_config["codec_think_eos_id"],
        ]

    codec_embedding_0 = _embed_codec_ids(
        torch.tensor([codec_prefill_ids], dtype=torch.long), weights
    )  # (1, 3or4, hidden)

    codec_embedding_1 = _embed_codec_ids(
        torch.tensor([[talker_config["codec_pad_id"],
                       talker_config["codec_bos_id"]]], dtype=torch.long), weights
    )  # (1, 2, hidden)

    if spk_embed_tensor is None:
        codec_input_embedding = torch.cat([codec_embedding_0, codec_embedding_1], dim=1)
    else:
        codec_input_embedding = torch.cat(
            [codec_embedding_0, spk_embed_tensor, codec_embedding_1], dim=1
        )
    # codec_input_embedding shape: (1, N, hidden) where N = 5..7

    # ── 7. Build the role prefix ───────────────────────────────────────
    # <|im_start|>assistant\n
    role_embed = _embed_text_ids(input_id[:, :3], weights)  # (1, 3, hidden)

    # ── 8. Build text-side + codec-side control signal ─────────────────
    # Text side: [tts_pad * (N-2), tts_bos]
    # These get ADDED to codec_input_embedding[:, :-1] (all but the last codec token)
    n_codec = codec_input_embedding.shape[1]
    text_side = torch.cat([
        tts_pad_embed.expand(-1, n_codec - 2, -1),
        tts_bos_embed,
    ], dim=1)  # (1, N-1, hidden)

    control_embed = text_side + codec_input_embedding[:, :-1, :]  # (1, N-1, hidden)

    talker_input_embed = torch.cat([role_embed, control_embed], dim=1)

    # ── 9. Handle text tokens ──────────────────────────────────────────
    # input_id[:, 3:-5] = actual text tokens (excluding role prefix and suffix)
    # input_id[:, 3:4]  = first text token
    # The last 5 tokens are: [<|im_end|>, \n, <|im_start|>, assistant, \n]

    if ref_codes is not None and ref_text is not None:
        # ── ICL mode: ref_text + ref_codes conditioning ───────────────
        # Tokenize ref_text: "<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_text_str = f"<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ref_text_ids_full = tokenizer(ref_text_str, return_tensors="pt")["input_ids"]
        # Slice to get just the text content: skip first 3 (<|im_start|>assistant\n)
        # and last 2 (<|im_end|>\n)
        ref_text_ids = ref_text_ids_full[:, 3:-2]  # (1, R)

        # Target text IDs (same slice as normal path)
        target_text_ids = input_id[:, 3:-5]  # (1, T)

        # Generate ICL prompt
        icl_embed, trailing_text_hidden = generate_icl_prompt(
            target_text_ids, ref_text_ids, ref_codes,
            tts_pad_embed, tts_eos_embed, weights, config, non_streaming_mode)

        # Final codec_bos position: tts_pad + codec_bos
        final_bos = tts_pad_embed + _embed_codec_ids(
            torch.tensor([[talker_config["codec_bos_id"]]], dtype=torch.long), weights
        )  # (1, 1, hidden)

        talker_input_embed = torch.cat([
            talker_input_embed,
            icl_embed,
            final_bos,
        ], dim=1)

    elif non_streaming_mode:
        # Non-streaming: all text tokens go into the prefill.
        # Text tokens (3:-5) + tts_eos, all added with codec_pad
        text_token_ids = input_id[:, 3:-5]  # (1, T)
        num_text = text_token_ids.shape[1]

        text_embeds = _embed_text_ids(text_token_ids, weights)  # (1, T, hidden)
        text_with_eos = torch.cat([text_embeds, tts_eos_embed], dim=1)  # (1, T+1, hidden)

        codec_pad_repeated = _embed_codec_ids(
            torch.full((1, num_text + 1), talker_config["codec_pad_id"], dtype=torch.long),
            weights,
        )  # (1, T+1, hidden)

        text_codec_block = text_with_eos + codec_pad_repeated  # (1, T+1, hidden)

        # Final codec_bos position: tts_pad + codec_bos
        final_bos = tts_pad_embed + _embed_codec_ids(
            torch.tensor([[talker_config["codec_bos_id"]]], dtype=torch.long), weights
        )  # (1, 1, hidden)

        talker_input_embed = torch.cat([
            talker_input_embed,
            text_codec_block,
            final_bos,
        ], dim=1)

        # In non-streaming mode, all text is already in the prefill,
        # so trailing_text_hidden is just tts_pad repeated
        trailing_text_hidden = tts_pad_embed  # (1, 1, hidden)

    else:
        # Streaming mode: only first text token goes into the prefill.
        # First text token + last codec_embedding position (codec_bos)
        first_text_embed = _embed_text_ids(input_id[:, 3:4], weights)  # (1, 1, hidden)
        first_text_with_codec = first_text_embed + codec_input_embedding[:, -1:, :]

        talker_input_embed = torch.cat([
            talker_input_embed,
            first_text_with_codec,
        ], dim=1)

        # Remaining text tokens (4:-5) + tts_eos for trailing
        remaining_text_ids = input_id[:, 4:-5]  # (1, T-1)
        if remaining_text_ids.shape[1] > 0:
            remaining_embeds = _embed_text_ids(remaining_text_ids, weights)
            trailing_text_hidden = torch.cat([remaining_embeds, tts_eos_embed], dim=1)
        else:
            trailing_text_hidden = tts_eos_embed  # (1, 1, hidden)

    # ── 10. Prepend instruct embed if present ──────────────────────────
    if instruct_embed is not None:
        talker_input_embed = torch.cat([instruct_embed, talker_input_embed], dim=1)

    return {
        "inputs_embeds": talker_input_embed,        # (1, seq, hidden)
        "trailing_text_hidden": trailing_text_hidden, # (1, T, hidden)
        "tts_pad_embed": tts_pad_embed,              # (1, 1, hidden)
    }


def _generate_loop(
    inputs_embeds,
    trailing_text_hidden,
    tts_pad_embed,
    weights,
    config,
    max_new_tokens=2048,
    do_sample=True,
    top_k=50,
    top_p=1.0,
    temperature=0.9,
    repetition_penalty=1.05,
    subtalker_dosample=True,
    subtalker_top_k=50,
    subtalker_top_p=1.0,
    subtalker_temperature=0.9,
):
    """Autoregressive generation loop for the Talker model.

    This mirrors the ``Qwen3TTSTalkerForConditionalGeneration.forward()``
    generation branch (lines 1664-1744) and the outer
    ``Qwen3TTSTalkerForConditionalGeneration.generate()`` loop.

    The loop:
    1. Prefills the talker with the full input embeddings (all at once).
    2. At each step, samples the first codebook token from the talker's
       ``codec_head`` logits.
    3. Calls the code predictor to generate remaining codebooks (2..N).
    4. Sums all codebook embeddings and adds the trailing text embedding
       for the current step, forming the next input.
    5. Stops when the sampled first-codebook token is the codec EOS token,
       or when ``max_new_tokens`` is reached.

    Args:
        inputs_embeds: (1, prefill_len, hidden_size) — from build_talker_input.
        trailing_text_hidden: (1, T, hidden_size) — trailing text embeddings.
        tts_pad_embed: (1, 1, hidden_size) — tts_pad embedding.
        weights: flat weight dict.
        config: top-level config dict.
        max_new_tokens: maximum codec tokens to generate.
        do_sample, top_k, top_p, temperature: sampling params for talker.
        repetition_penalty: penalty for repeated first-codebook tokens.
        subtalker_dosample, subtalker_top_k, subtalker_top_p,
        subtalker_temperature: sampling params for code predictor.

    Returns:
        all_codes: (1, num_steps, num_code_groups) int64 — generated codec
            codes for all steps until EOS (EOS step excluded).
    """
    talker_config = config["talker_config"]
    codec_eos_id = talker_config["codec_eos_token_id"]
    num_code_groups = talker_config.get("num_code_groups", 16)
    vocab_size = talker_config["vocab_size"]

    # Compute suppress_tokens: last 1024 codec IDs except EOS
    suppress_tokens = [
        i for i in range(vocab_size - 1024, vocab_size)
        if i != codec_eos_id
    ]
    suppress_mask = torch.zeros(vocab_size, dtype=torch.bool)
    suppress_mask[suppress_tokens] = True

    # ── Prefill ────────────────────────────────────────────────────────
    hidden, kv_caches = talker_forward(
        inputs_embeds, weights, talker_config,
        kv_caches=None, cache_position=None,
    )

    # Get logits from the last position
    logits = F.linear(hidden[:, -1:, :], weights["talker.codec_head.weight"])  # (1, 1, vocab)
    logits = logits[:, 0, :]  # (1, vocab)

    # Apply suppress mask
    logits[:, suppress_mask] = float('-inf')

    # Sample first codebook token
    past_first_codes = []
    first_code = sample_token(logits, do_sample, top_k, top_p, temperature)  # (1, 1)

    all_step_codes = []
    generation_step = 0
    cache_position = inputs_embeds.shape[1]  # next position after prefill

    past_hidden = hidden[:, -1:, :]  # (1, 1, hidden) — last hidden state

    while generation_step < max_new_tokens:
        first_code_val = first_code.item()

        # Check EOS
        if first_code_val == codec_eos_id:
            break

        past_first_codes.append(first_code_val)

        # Embed the first codebook token using codec_embedding
        first_code_embed = _embed_codec_ids(first_code, weights)  # (1, 1, hidden)

        # Generate sub-codes (codebooks 2..N) using code predictor
        sub_codes, all_embeds_sum = generate_sub_codes(
            first_code_embed, past_hidden, weights, talker_config,
            do_sample=subtalker_dosample,
            top_k=subtalker_top_k,
            top_p=subtalker_top_p,
            temperature=subtalker_temperature,
        )
        # sub_codes: (1, num_code_groups-1)
        # all_embeds_sum: (1, 1, hidden) — sum of ALL codebook embeddings

        # Record this step's codes: [first_code, sub_code_0, ..., sub_code_N-2]
        step_codes = torch.cat([first_code, sub_codes], dim=-1)  # (1, num_code_groups)
        all_step_codes.append(step_codes)

        # Build next talker input: embeds_sum + trailing text
        next_embed = all_embeds_sum  # (1, 1, hidden)

        if generation_step < trailing_text_hidden.shape[1]:
            next_embed = next_embed + trailing_text_hidden[:, generation_step:generation_step+1, :]
        else:
            next_embed = next_embed + tts_pad_embed

        # Forward one step through talker
        hidden, kv_caches = talker_forward(
            next_embed, weights, talker_config,
            kv_caches=kv_caches, cache_position=cache_position,
        )

        cache_position += 1
        generation_step += 1

        # Get logits for next first-codebook token
        logits = F.linear(hidden[:, -1:, :], weights["talker.codec_head.weight"])
        logits = logits[:, 0, :]  # (1, vocab)

        # Apply suppress mask
        logits[:, suppress_mask] = float('-inf')

        # Apply repetition penalty on past first-codebook tokens
        if repetition_penalty != 1.0 and len(past_first_codes) > 0:
            past_tensor = torch.tensor([past_first_codes], dtype=torch.long)
            logits = apply_repetition_penalty(logits, past_tensor, repetition_penalty)

        # Sample next first codebook token
        first_code = sample_token(logits, do_sample, top_k, top_p, temperature)

        past_hidden = hidden[:, -1:, :]

    all_codes = torch.stack(all_step_codes, dim=1)  # (1, num_steps, num_code_groups)
    return all_codes


# ---------------------------------------------------------------------------
# Speaker Encoder (ECAPA-TDNN) for Voice Clone
# ---------------------------------------------------------------------------


def _reflect_pad_conv1d(x, weight, bias, dilation=1):
    """Conv1d with reflect padding (padding='same' equivalent).

    Args:
        x: (batch, in_channels, time)
        weight: (out_channels, in_channels, kernel_size)
        bias: (out_channels,) or None
        dilation: int
    """
    kernel_size = weight.shape[2]
    effective_k = dilation * (kernel_size - 1)
    pad_left = effective_k // 2
    pad_right = effective_k - pad_left
    x = F.pad(x, (pad_left, pad_right), mode="reflect")
    return F.conv1d(x, weight, bias, dilation=dilation)


def _tdnn_block(x, w, prefix, dilation=1):
    """TimeDelayNetBlock: Conv1d(reflect pad) + ReLU."""
    weight = w[f"{prefix}.conv.weight"]
    bias = w[f"{prefix}.conv.bias"]
    return F.relu(_reflect_pad_conv1d(x, weight, bias, dilation=dilation))


def _res2net_block(x, w, prefix, scale=8, dilation=1):
    """Res2NetBlock: split into chunks, sequential residual processing."""
    chunks = torch.chunk(x, scale, dim=1)
    outputs = []
    output_part = None
    for i, chunk in enumerate(chunks):
        if i == 0:
            output_part = chunk
        elif i == 1:
            output_part = _tdnn_block(chunk, w, f"{prefix}.blocks.{i - 1}", dilation=dilation)
        else:
            output_part = _tdnn_block(chunk + output_part, w, f"{prefix}.blocks.{i - 1}", dilation=dilation)
        outputs.append(output_part)
    return torch.cat(outputs, dim=1)


def _squeeze_excitation_block(x, w, prefix):
    """SE block: global avg pool -> conv1 -> relu -> conv2 -> sigmoid -> scale."""
    x_mean = x.mean(dim=2, keepdim=True)  # (B, C, 1)
    x_mean = F.relu(_reflect_pad_conv1d(x_mean, w[f"{prefix}.conv1.weight"], w[f"{prefix}.conv1.bias"]))
    x_mean = torch.sigmoid(_reflect_pad_conv1d(x_mean, w[f"{prefix}.conv2.weight"], w[f"{prefix}.conv2.bias"]))
    return x * x_mean


def _se_res2net_block(x, w, prefix, scale=8, kernel_size=3, dilation=1):
    """SqueezeExcitationRes2NetBlock: tdnn1 -> res2net -> tdnn2 -> SE + residual."""
    residual = x
    x = _tdnn_block(x, w, f"{prefix}.tdnn1")  # kernel=1, dilation=1
    x = _res2net_block(x, w, f"{prefix}.res2net_block", scale=scale, dilation=dilation)
    x = _tdnn_block(x, w, f"{prefix}.tdnn2")  # kernel=1, dilation=1
    x = _squeeze_excitation_block(x, w, f"{prefix}.se_block")
    return x + residual


def _attentive_statistics_pooling(x, w, prefix):
    """Attentive statistical pooling.

    Args:
        x: (batch, channels, time)
    Returns:
        (batch, channels*2, 1)
    """
    eps = 1e-12
    B, C, T = x.shape
    # Full mask (no padding in our single-sample case)
    mask = torch.ones(B, 1, T, dtype=x.dtype, device=x.device)
    total = mask.sum(dim=2, keepdim=True)

    # First statistics
    mean = (mask * x).sum(dim=2) / total.squeeze(2)
    std = torch.sqrt(((mask * (x - mean.unsqueeze(2))).pow(2)).sum(dim=2) / total.squeeze(2) + eps)

    mean_expanded = mean.unsqueeze(2).expand(-1, -1, T)
    std_expanded = std.unsqueeze(2).expand(-1, -1, T)
    attn_input = torch.cat([x, mean_expanded, std_expanded], dim=1)  # (B, 3*C, T)

    # TDNN + tanh + conv for attention
    attn = _tdnn_block(attn_input, w, f"{prefix}.tdnn")
    attn = torch.tanh(attn)
    attn = _reflect_pad_conv1d(attn, w[f"{prefix}.conv.weight"], w[f"{prefix}.conv.bias"])

    # Softmax attention
    attn = attn.masked_fill(mask == 0, float("-inf"))
    attn = F.softmax(attn, dim=2)

    # Weighted statistics
    w_mean = (attn * x).sum(dim=2)
    w_std = torch.sqrt(((attn * (x - w_mean.unsqueeze(2))).pow(2)).sum(dim=2) + eps)

    pooled = torch.cat([w_mean, w_std], dim=1).unsqueeze(2)  # (B, 2*C, 1)
    return pooled


def speaker_encoder_forward(mels, w, se_config):
    """ECAPA-TDNN speaker encoder forward pass.

    Args:
        mels: (batch, time, mel_dim=128) — mel spectrogram
        w: dict of speaker_encoder weights (keys without 'speaker_encoder.' prefix)
        se_config: speaker_encoder_config dict

    Returns:
        (batch, enc_dim) speaker embedding
    """
    enc_channels = se_config.get("enc_channels", [512, 512, 512, 512, 1536])
    enc_kernel_sizes = se_config.get("enc_kernel_sizes", [5, 3, 3, 3, 1])
    enc_dilations = se_config.get("enc_dilations", [1, 2, 3, 4, 1])
    scale = se_config.get("enc_res2net_scale", 8)
    se_channels = se_config.get("enc_se_channels", 128)

    # Transpose: (B, T, mel_dim) -> (B, mel_dim, T)
    x = mels.transpose(1, 2)

    hidden_list = []

    # Block 0: initial TDNN
    x = _tdnn_block(x, w, "blocks.0", dilation=enc_dilations[0])
    hidden_list.append(x)

    # Blocks 1..(N-2): SE-Res2Net blocks
    for i in range(1, len(enc_channels) - 1):
        x = _se_res2net_block(
            x, w, f"blocks.{i}",
            scale=scale,
            kernel_size=enc_kernel_sizes[i],
            dilation=enc_dilations[i],
        )
        hidden_list.append(x)

    # Multi-layer feature aggregation: cat blocks[1:]
    x = torch.cat(hidden_list[1:], dim=1)
    x = _tdnn_block(x, w, "mfa", dilation=enc_dilations[-1])

    # Attentive statistical pooling
    x = _attentive_statistics_pooling(x, w, "asp")

    # Final FC
    x = _reflect_pad_conv1d(x, w["fc.weight"], w["fc.bias"])

    return x.squeeze(-1)  # (batch, enc_dim)


def mel_spectrogram(y, n_fft=1024, num_mels=128, sampling_rate=24000,
                    hop_size=256, win_size=1024, fmin=0, fmax=12000):
    """Compute log mel spectrogram.

    Args:
        y: (batch, samples) or (samples,) waveform tensor
        Other args: STFT/mel parameters

    Returns:
        (batch, num_mels, time) log-mel spectrogram
    """
    from librosa.filters import mel as librosa_mel_fn

    if y.dim() == 1:
        y = y.unsqueeze(0)

    mel_basis = torch.from_numpy(
        librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
    ).float().to(y.device)
    hann_window = torch.hann_window(win_size).to(y.device)

    padding = (n_fft - hop_size) // 2
    y = F.pad(y.unsqueeze(1), (padding, padding), mode="reflect").squeeze(1)

    spec = torch.stft(
        y, n_fft, hop_length=hop_size, win_length=win_size,
        window=hann_window, center=False, pad_mode="reflect",
        normalized=False, onesided=True, return_complex=True,
    )
    spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

    mel_spec = torch.matmul(mel_basis, spec)
    mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

    return mel_spec


def extract_speaker_embedding(audio, sr, speaker_encoder_weights, se_config):
    """Extract speaker embedding from audio waveform.

    Args:
        audio: np.ndarray — mono float32 waveform
        sr: int — sample rate (will resample to 24kHz if different)
        speaker_encoder_weights: dict of weights (keys without 'speaker_encoder.' prefix)
        se_config: speaker_encoder_config dict

    Returns:
        (enc_dim,) speaker embedding tensor
    """
    target_sr = se_config.get("sample_rate", 24000)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(y=audio.astype(np.float32), orig_sr=int(sr), target_sr=target_sr)

    audio_tensor = torch.from_numpy(audio).float()
    mels = mel_spectrogram(audio_tensor, sampling_rate=target_sr)  # (1, 128, T)
    mels = mels.transpose(1, 2)  # (1, T, 128)

    # Cast to model dtype
    dtype = speaker_encoder_weights["fc.weight"].dtype
    mels = mels.to(dtype)

    embedding = speaker_encoder_forward(mels, speaker_encoder_weights, se_config)
    return embedding[0]  # (enc_dim,)


def _load_audio_wav(path):
    """Load a WAV file and return (np.ndarray float32, sample_rate)."""
    with wave_mod.open(path, "r") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    elif sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, sr


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_custom_voice(state, text, speaker, language="Auto", instruct=None,
                          non_streaming_mode=True, **kwargs):
    """Generate speech with CustomVoice model using predefined speaker.

    Args:
        state: dict from load_model().
        text: str — text to synthesize.
        speaker: str — speaker name.
        language: str — language (default "Auto").
        instruct: str or None — optional instruction.
        non_streaming_mode: bool.
        **kwargs: generation params (do_sample, top_k, top_p, temperature,
                  repetition_penalty, max_new_tokens, etc.)

    Returns:
        (np.ndarray waveform, int sample_rate).
    """
    weights = state["weights"]
    config = state["config"]
    tokenizer = state["tokenizer"]

    result = build_talker_input(
        text, weights, config, tokenizer,
        language=language, speaker=speaker, instruct=instruct,
        non_streaming_mode=non_streaming_mode,
    )

    codes = _generate_loop(
        result["inputs_embeds"],
        result["trailing_text_hidden"],
        result["tts_pad_embed"],
        weights, config, **kwargs,
    )

    # Decode to audio — codes shape is (1, num_steps, num_code_groups)
    speech_tok = state["speech_tokenizer"]
    wavs, sr = decode(speech_tok, {"audio_codes": codes[0]})  # codes[0]: (steps, n_code_groups)

    return wavs[0], sr


def generate_voice_design(state, text, instruct, language="Auto",
                          non_streaming_mode=True, **kwargs):
    """Generate speech with VoiceDesign model using instruction text.

    Args:
        state: dict from load_model().
        text: str — text to synthesize.
        instruct: str — voice/style instruction.
        language: str — language (default "Auto").
        non_streaming_mode: bool.
        **kwargs: generation params.

    Returns:
        (np.ndarray waveform, int sample_rate).
    """
    weights = state["weights"]
    config = state["config"]
    tokenizer = state["tokenizer"]

    result = build_talker_input(
        text, weights, config, tokenizer,
        language=language, speaker=None, instruct=instruct,
        non_streaming_mode=non_streaming_mode,
    )

    codes = _generate_loop(
        result["inputs_embeds"],
        result["trailing_text_hidden"],
        result["tts_pad_embed"],
        weights, config, **kwargs,
    )

    speech_tok = state["speech_tokenizer"]
    wavs, sr = decode(speech_tok, {"audio_codes": codes[0]})

    return wavs[0], sr


def generate_voice_clone(state, text, ref_audio, ref_text=None,
                         x_vector_only=False, language="Auto", instruct=None,
                         non_streaming_mode=True, **kwargs):
    """Generate speech with voice cloned from reference audio (Base model).

    Supports two modes:
    - x-vector only: extracts speaker embedding from reference audio only.
    - ICL mode: also encodes reference audio to codes and uses ref_text for
      richer in-context learning conditioning.

    Args:
        state: dict from load_model().
        text: str — text to synthesize.
        ref_audio: str (WAV file path) or tuple (np.ndarray, sample_rate).
        ref_text: str or None — transcription of reference audio (enables ICL).
        x_vector_only: bool — if True, skip ICL even when ref_text is provided.
        language: str — language (default "Auto").
        instruct: str or None — optional instruction.
        non_streaming_mode: bool.
        **kwargs: generation params (do_sample, top_k, top_p, temperature,
                  repetition_penalty, max_new_tokens, etc.)

    Returns:
        (np.ndarray waveform, int sample_rate).
    """
    se_weights = state.get("speaker_encoder_weights")
    if not se_weights:
        raise ValueError("Model does not have speaker encoder weights. "
                         "Voice clone requires a Base model.")

    se_config = state["config"].get("speaker_encoder_config", {})

    # Load reference audio
    if isinstance(ref_audio, str):
        audio, audio_sr = _load_audio_wav(ref_audio)
    elif isinstance(ref_audio, (tuple, list)) and len(ref_audio) == 2:
        audio, audio_sr = ref_audio[0], ref_audio[1]
    else:
        raise ValueError("ref_audio must be a WAV file path or (np.ndarray, sample_rate) tuple")

    # Extract speaker embedding
    spk_embed = extract_speaker_embedding(audio, audio_sr, se_weights, se_config)

    weights = state["weights"]
    config = state["config"]
    tokenizer = state["tokenizer"]
    speech_tok = state["speech_tokenizer"]

    # ICL mode: encode reference audio to codes
    use_icl = ref_text is not None and not x_vector_only
    ref_codes = None
    if use_icl:
        audio_tensor = torch.from_numpy(audio).float()
        ref_codes = encode(speech_tok, audio_tensor, audio_sr)  # (1, T, n_q)
        ref_codes = ref_codes[0]  # (T, n_q)

    result = build_talker_input(
        text, weights, config, tokenizer,
        language=language, speaker=None, instruct=instruct,
        speaker_embed=spk_embed,
        non_streaming_mode=non_streaming_mode,
        ref_text=ref_text if use_icl else None,
        ref_codes=ref_codes,
    )

    codes = _generate_loop(
        result["inputs_embeds"],
        result["trailing_text_hidden"],
        result["tts_pad_embed"],
        weights, config, **kwargs,
    )

    # Decode to audio
    if use_icl:
        # Prepend ref_codes to generated codes before decode
        # codes: (1, num_steps, num_code_groups), ref_codes: (T_ref, n_q)
        combined_codes = torch.cat([ref_codes.unsqueeze(0), codes[0:1]], dim=1)
        wavs, sr = decode(speech_tok, {"audio_codes": combined_codes[0]})
        wav = wavs[0]
        # Trim reference audio portion from the output
        ref_len = ref_codes.shape[0]
        total_len = combined_codes.shape[1]
        cut = int(ref_len / total_len * len(wav))
        wav = wav[cut:]
    else:
        wavs, sr = decode(speech_tok, {"audio_codes": codes[0]})
        wav = wavs[0]

    return wav, sr
