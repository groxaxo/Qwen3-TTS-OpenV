import math
import json
import os
import torch
import torch.nn.functional as F
import numpy as np
from safetensors.torch import load_file


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


def load_tokenizer(model_path: str) -> dict:
    """Load speech tokenizer weights and config from model_path/speech_tokenizer/.

    Args:
        model_path: Path to the Qwen3-TTS model directory.

    Returns:
        dict with keys: 'weights' (flat tensor dict), 'config' (parsed JSON dict),
        'model_type' ('qwen3_tts_tokenizer_25hz' or 'qwen3_tts_tokenizer_12hz').
    """
    tokenizer_path = os.path.join(model_path, "speech_tokenizer")
    config_path = os.path.join(tokenizer_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    model_type = config.get("model_type", "")

    # Load all safetensors files
    weights = {}
    for fname in sorted(os.listdir(tokenizer_path)):
        if fname.endswith(".safetensors"):
            shard = load_file(os.path.join(tokenizer_path, fname), device="cpu")
            weights.update(shard)

    return {
        "weights": weights,
        "config": config,
        "model_type": model_type,
    }


# ---------------------------------------------------------------------------
# Functional building blocks (shared by tokenizer decoder & main model)
# ---------------------------------------------------------------------------


def rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x^2) + eps)."""
    dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(dtype)


def layer_norm(x, weight, bias, eps=1e-5):
    """Standard LayerNorm."""
    return F.layer_norm(x, weight.shape, weight, bias, eps)


def silu_linear(x, w1, w2, b1=None, b2=None):
    """SwiGLU-style gated MLP: down(silu(gate(x)) * up(x))."""
    h = F.linear(x, w1, b1)
    h = F.silu(h)
    return F.linear(h, w2, b2)


def swiglu_ffn(x, gate_w, up_w, down_w, gate_b=None, up_b=None, down_b=None):
    """SwiGLU FFN: down(silu(gate(x)) * up(x))."""
    gate = F.silu(F.linear(x, gate_w, gate_b))
    up = F.linear(x, up_w, up_b)
    return F.linear(gate * up, down_w, down_b)


def conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Functional conv1d wrapper that handles dtype casting."""
    return F.conv1d(x.to(weight.dtype), weight, bias, stride, padding, dilation, groups)


def conv_transpose1d(x, weight, bias=None, stride=1, padding=0, output_padding=0):
    """Functional conv_transpose1d wrapper."""
    return F.conv_transpose1d(x.to(weight.dtype), weight, bias, stride, padding, output_padding)


def snake_beta(x, alpha_param, beta_param):
    """SnakeBeta activation: x + (1/beta) * sin^2(alpha * x)."""
    alpha = torch.exp(alpha_param).unsqueeze(0).unsqueeze(-1)
    beta = torch.exp(beta_param).unsqueeze(0).unsqueeze(-1)
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


# ---------------------------------------------------------------------------
# VQ codebook decode (V2 / 12Hz tokenizer)
# ---------------------------------------------------------------------------

def vq_codebook_decode(codes, cluster_usage, embedding_sum, eps=1e-5):
    """Decode code indices to embeddings using EuclideanCodebook.

    Args:
        codes: (T,) int64 tensor of code indices.
        cluster_usage: (codebook_size,) tensor.
        embedding_sum: (codebook_size, dim) tensor.
        eps: Epsilon for numerical stability.

    Returns:
        (T, dim) tensor of decoded embeddings.
    """
    embedding = embedding_sum / cluster_usage.clamp(min=eps).unsqueeze(-1)
    return F.embedding(codes, embedding)


def vq_decode(codes, weights, prefix, codebook_dim, dim):
    """Single VectorQuantization decode.

    Args:
        codes: (T,) int64.
        weights: weight dict.
        prefix: weight key prefix (e.g. 'decoder.quantizer.rvq_first.vq.layers.0').
        codebook_dim: codebook dimension.
        dim: output dimension.

    Returns:
        (dim, T) tensor.
    """
    cluster_usage = weights[f"{prefix}._codebook.cluster_usage"]
    embedding_sum = weights[f"{prefix}._codebook.embedding_sum"]
    quantized = vq_codebook_decode(codes, cluster_usage, embedding_sum)

    # project_out (if codebook_dim != dim)
    proj_key = f"{prefix}.project_out.weight"
    if proj_key in weights:
        quantized = F.linear(quantized, weights[proj_key],
                             weights.get(f"{prefix}.project_out.bias"))

    return quantized.transpose(0, 1)  # (dim, T)


def rvq_decode(codes, weights, prefix, n_q, codebook_dim, dim):
    """ResidualVectorQuantization decode -- sum across quantizer layers.

    Args:
        codes: (n_q, T) int64 -- one row per quantizer.
        weights: weight dict.
        prefix: e.g. 'decoder.quantizer.rvq_first.vq'.
        n_q: number of quantizers.
        codebook_dim: codebook dimension.
        dim: output dimension.

    Returns:
        (dim, T) tensor.
    """
    result = torch.zeros(dim, codes.shape[1], dtype=torch.float32)
    for i in range(n_q):
        layer_prefix = f"{prefix}.layers.{i}"
        result = result + vq_decode(codes[i], weights, layer_prefix, codebook_dim, dim)
    return result


def split_rvq_decode(codes, weights, prefix, n_q_semantic, n_q_total, config):
    """SplitResidualVectorQuantizer decode.

    Args:
        codes: (batch, n_q_total, T) int64.
        weights: weight dict.
        prefix: e.g. 'decoder.quantizer'.
        n_q_semantic: number of semantic quantizers (typically 1).
        n_q_total: total quantizers (typically 16).
        config: tokenizer config dict.

    Returns:
        (batch, codebook_dim, T) tensor.
    """
    codebook_dim = config.get("codebook_dim", 512) // 2  # dimension per quantizer
    output_dim = config.get("codebook_dim", 512)

    results = []
    for b in range(codes.shape[0]):
        # Semantic quantizer (first)
        semantic_codes = codes[b, :n_q_semantic]  # (n_q_semantic, T)
        semantic = rvq_decode(semantic_codes, weights,
                              f"{prefix}.rvq_first.vq", n_q_semantic,
                              codebook_dim, codebook_dim)

        # Output projection for rvq_first (Conv1d: codebook_dim -> output_dim)
        proj_key = f"{prefix}.rvq_first.output_proj.weight"
        if proj_key in weights:
            semantic = conv1d(semantic.unsqueeze(0), weights[proj_key]).squeeze(0)

        result = semantic

        # Acoustic quantizers (rest)
        if n_q_total > n_q_semantic:
            acoustic_codes = codes[b, n_q_semantic:n_q_total]  # (n_q_acoustic, T)
            n_q_acoustic = n_q_total - n_q_semantic
            acoustic = rvq_decode(acoustic_codes, weights,
                                  f"{prefix}.rvq_rest.vq", n_q_acoustic,
                                  codebook_dim, codebook_dim)
            proj_key = f"{prefix}.rvq_rest.output_proj.weight"
            if proj_key in weights:
                acoustic = conv1d(acoustic.unsqueeze(0),
                                  weights[proj_key]).squeeze(0)
            result = result + acoustic

        results.append(result)

    return torch.stack(results)  # (batch, output_dim, T)


# ---------------------------------------------------------------------------
# V2 decoder transformer (12Hz tokenizer decode path)
# ---------------------------------------------------------------------------


def rotary_embedding(seq_len, head_dim, theta=10000.0):
    """Compute cos/sin for RoPE.

    Returns:
        (seq_len, head_dim) cos, (seq_len, head_dim) sin.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def apply_rotary_pos_emb_simple(q, k, cos, sin):
    """Apply rotary position embeddings to q, k.

    Args:
        q, k: (batch, heads, seq, head_dim).
        cos, sin: (seq, head_dim).
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    def rotate_half(x):
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def v2_decoder_attention(hidden, weights, prefix, config, cos, sin, kv_cache=None):
    """Single attention layer for V2 decoder transformer.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: weight dict.
        prefix: e.g. 'decoder.pre_transformer.layers.0.self_attn'.
        config: dict with num_attention_heads, num_key_value_heads, head_dim, hidden_size.
        cos, sin: rotary embeddings.
        kv_cache: optional (k, v) tuple for incremental decoding.

    Returns:
        (batch, seq, hidden_size), updated kv_cache.
    """
    n_heads = config["num_attention_heads"]
    n_kv_heads = config["num_key_value_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // n_heads)

    q = F.linear(hidden, weights[f"{prefix}.q_proj.weight"])
    k = F.linear(hidden, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden, weights[f"{prefix}.v_proj.weight"])

    batch, seq, _ = hidden.shape
    q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, n_kv_heads, head_dim).transpose(1, 2)

    # Apply RoPE
    q, k = apply_rotary_pos_emb_simple(q, k, cos[:seq], sin[:seq])

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

    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    # Causal mask
    if seq > 1:
        mask = torch.triu(torch.full((seq, k.shape[2]), float('-inf')), diagonal=k.shape[2] - seq + 1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0).to(attn.device)

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
    out = F.linear(out, weights[f"{prefix}.o_proj.weight"])

    return out, new_cache


def v2_decoder_transformer_layer(hidden, weights, prefix, config, cos, sin, kv_cache=None):
    """Single V2 decoder transformer layer with pre-norm + layer scale.

    Returns:
        hidden, updated kv_cache.
    """
    eps = config.get("rms_norm_eps", 1e-5)

    # Pre-norm attention
    normed = rms_norm(hidden, weights[f"{prefix}.input_layernorm.weight"], eps)
    attn_out, new_cache = v2_decoder_attention(
        normed, weights, f"{prefix}.self_attn", config, cos, sin, kv_cache)

    # Layer scale
    ls_key = f"{prefix}.self_attn_layer_scale.scale"
    if ls_key in weights:
        attn_out = attn_out * weights[ls_key]
    hidden = hidden + attn_out

    # Pre-norm FFN
    normed = rms_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"], eps)
    ffn_out = swiglu_ffn(
        normed,
        weights[f"{prefix}.mlp.gate_proj.weight"],
        weights[f"{prefix}.mlp.up_proj.weight"],
        weights[f"{prefix}.mlp.down_proj.weight"],
    )

    ls_key = f"{prefix}.mlp_layer_scale.scale"
    if ls_key in weights:
        ffn_out = ffn_out * weights[ls_key]
    hidden = hidden + ffn_out

    return hidden, new_cache


def v2_decoder_transformer(hidden, weights, prefix, config):
    """Full V2 decoder transformer.

    Args:
        hidden: (batch, seq, latent_dim) -- already decoded from VQ.
        weights: weight dict.
        prefix: e.g. 'decoder.pre_transformer'.
        config: decoder config dict.

    Returns:
        (batch, seq, latent_dim).
    """
    n_layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", 64)
    theta = config.get("rope_theta", 10000.0)
    seq_len = hidden.shape[1]

    # Input projection: latent_dim -> hidden_size
    inp_key = f"{prefix}.input_proj.weight"
    if inp_key in weights:
        hidden = F.linear(hidden, weights[inp_key],
                          weights.get(f"{prefix}.input_proj.bias"))

    # Rotary embeddings
    cos, sin = rotary_embedding(seq_len, head_dim, theta)
    cos, sin = cos.to(hidden.device), sin.to(hidden.device)

    # Transformer layers
    for i in range(n_layers):
        layer_prefix = f"{prefix}.layers.{i}"
        hidden, _ = v2_decoder_transformer_layer(hidden, weights, layer_prefix, config, cos, sin)

    # Final norm
    hidden = rms_norm(hidden, weights[f"{prefix}.norm.weight"],
                      config.get("rms_norm_eps", 1e-5))

    # Output projection: hidden_size -> latent_dim
    out_key = f"{prefix}.output_proj.weight"
    if out_key in weights:
        hidden = F.linear(hidden, weights[out_key],
                          weights.get(f"{prefix}.output_proj.bias"))

    return hidden


# ---------------------------------------------------------------------------
# V2 conv decoder blocks (12Hz tokenizer decode path)
# ---------------------------------------------------------------------------


def causal_conv1d(x, weight, bias=None, stride=1, dilation=1):
    """Causal 1D convolution with left padding.

    Args:
        x: (batch, channels, time).
        weight: (out_channels, in_channels_per_group, kernel_size).
    """
    kernel_size = weight.shape[2]
    pad = (kernel_size - 1) * dilation
    x = F.pad(x, (pad, 0))
    # Infer groups from input channels and weight shape
    groups = x.shape[1] // weight.shape[1]
    return F.conv1d(x, weight.to(x.dtype),
                    bias.to(x.dtype) if bias is not None else None,
                    stride=stride, dilation=dilation, groups=groups)


def causal_conv_transpose1d(x, weight, bias=None, stride=1):
    """Causal transposed 1D convolution.

    Args:
        x: (batch, channels, time).
        weight: (in_channels, out_channels, kernel_size).
    """
    out = F.conv_transpose1d(x, weight.to(x.dtype),
                             bias.to(x.dtype) if bias is not None else None,
                             stride=stride)
    # Remove right padding to make causal
    pad = weight.shape[2] - stride
    if pad > 0:
        out = out[..., :-pad]
    return out


def convnext_block(x, weights, prefix):
    """ConvNeXt block: depthwise conv -> norm -> pointwise up -> GELU -> pointwise down.

    Args:
        x: (batch, channels, time).
    """
    # Depthwise conv
    dw_w = weights[f"{prefix}.dwconv.conv.weight"]
    dw_b = weights.get(f"{prefix}.dwconv.conv.bias")
    h = causal_conv1d(x, dw_w, dw_b)

    h = h.transpose(1, 2)  # (batch, time, channels)
    h = layer_norm(h, weights[f"{prefix}.norm.weight"], weights[f"{prefix}.norm.bias"])
    h = F.linear(h, weights[f"{prefix}.pwconv1.weight"], weights[f"{prefix}.pwconv1.bias"])
    h = F.gelu(h)
    h = F.linear(h, weights[f"{prefix}.pwconv2.weight"], weights[f"{prefix}.pwconv2.bias"])
    h = h.transpose(1, 2)  # (batch, channels, time)

    # Layer scale
    gamma_key = f"{prefix}.gamma"
    if gamma_key in weights:
        h = h * weights[gamma_key].unsqueeze(0).unsqueeze(-1)

    return x + h


def decoder_residual_unit_with_dilation(x, weights, prefix, dilation):
    """Residual unit with explicit dilation.

    Args:
        x: (batch, channels, time).
    """
    h = snake_beta(x, weights[f"{prefix}.act1.alpha"], weights[f"{prefix}.act1.beta"])

    conv1_w = weights[f"{prefix}.conv1.conv.weight"]
    conv1_b = weights.get(f"{prefix}.conv1.conv.bias")
    h = causal_conv1d(h, conv1_w, conv1_b, dilation=dilation)

    h = snake_beta(h, weights[f"{prefix}.act2.alpha"], weights[f"{prefix}.act2.beta"])

    conv2_w = weights[f"{prefix}.conv2.conv.weight"]
    conv2_b = weights.get(f"{prefix}.conv2.conv.bias")
    h = causal_conv1d(h, conv2_w, conv2_b)

    return x + h


def v2_decoder_block(x, weights, prefix, upsample_rate):
    """Decoder block: SnakeBeta -> TransConv upsample -> 3 residual units.

    Args:
        x: (batch, in_channels, time).
    """
    # SnakeBeta + TransConv upsample
    x = snake_beta(x, weights[f"{prefix}.block.0.alpha"], weights[f"{prefix}.block.0.beta"])
    x = causal_conv_transpose1d(x, weights[f"{prefix}.block.1.conv.weight"],
                                weights.get(f"{prefix}.block.1.conv.bias"),
                                stride=upsample_rate)

    # 3 residual units with dilations [1, 3, 9]
    for i, dilation in enumerate([1, 3, 9]):
        x = decoder_residual_unit_with_dilation(x, weights, f"{prefix}.block.{i + 2}", dilation)

    return x


# ---------------------------------------------------------------------------
# V2 full decode pipeline (12Hz tokenizer)
# ---------------------------------------------------------------------------

def v2_decode(state, audio_codes):
    """Decode V2 (12Hz) codes to waveform.

    Args:
        state: dict from load_tokenizer().
        audio_codes: (batch, code_length, num_quantizers) int64 tensor.

    Returns:
        list of 1D float32 numpy arrays, sample_rate int.
    """
    weights = state["weights"]
    config = state["config"]
    dec_cfg = config.get("decoder_config", config)

    n_q = dec_cfg.get("num_quantizers", 16)
    n_q_semantic = dec_cfg.get("num_semantic_quantizers", 1)
    upsample_rates = dec_cfg.get("upsample_rates", [8, 5, 4, 3])
    upsampling_ratios = dec_cfg.get("upsampling_ratios", [2, 2])

    # Total upsample factor (product of all rates)
    total_upsample = 1
    for r in upsample_rates + upsampling_ratios:
        total_upsample *= r

    # Compute expected audio lengths from non-padding codes
    audio_lengths = (audio_codes[..., 0] > -1).sum(1) * total_upsample

    # Clamp codes (padding was -1, clamp to 0)
    audio_codes = torch.clamp(audio_codes, min=0)

    # Transpose: (batch, code_length, n_q) -> (batch, n_q, code_length)
    codes = audio_codes.transpose(1, 2)

    # 1. Split RVQ decode: codes -> embeddings
    #    hidden: (batch, codebook_dim, T)
    hidden = split_rvq_decode(codes, weights, "decoder.quantizer",
                              n_q_semantic, n_q, dec_cfg)

    # 2. Pre-conv: (batch, codebook_dim, T) -> (batch, latent_dim, T)
    hidden = causal_conv1d(hidden,
                           weights["decoder.pre_conv.conv.weight"],
                           weights.get("decoder.pre_conv.conv.bias"))

    # 3. Transformer: (batch, T, latent_dim)
    hidden = hidden.transpose(1, 2)
    hidden = v2_decoder_transformer(hidden, weights, "decoder.pre_transformer",
                                    dec_cfg)
    hidden = hidden.transpose(1, 2)  # (batch, latent_dim, T)

    # 4. Upsampling blocks (transposed conv + ConvNeXt, one per ratio)
    for i, ratio in enumerate(upsampling_ratios):
        hidden = causal_conv_transpose1d(
            hidden,
            weights[f"decoder.upsample.{i}.0.conv.weight"],
            weights.get(f"decoder.upsample.{i}.0.conv.bias"),
            stride=ratio)
        hidden = convnext_block(hidden, weights, f"decoder.upsample.{i}.1")

    # 5. Main decoder conv chain
    # First conv: latent_dim -> decoder_dim
    hidden = causal_conv1d(hidden,
                           weights["decoder.decoder.0.conv.weight"],
                           weights.get("decoder.decoder.0.conv.bias"))

    # Decoder blocks with upsampling (one per upsample_rate)
    for i, rate in enumerate(upsample_rates):
        hidden = v2_decoder_block(hidden, weights,
                                  f"decoder.decoder.{i + 1}", rate)

    # Final: SnakeBeta activation -> conv to mono -> clamp
    n_final = 1 + len(upsample_rates)
    hidden = snake_beta(hidden,
                        weights[f"decoder.decoder.{n_final}.alpha"],
                        weights[f"decoder.decoder.{n_final}.beta"])
    hidden = causal_conv1d(hidden,
                           weights[f"decoder.decoder.{n_final + 1}.conv.weight"],
                           weights.get(f"decoder.decoder.{n_final + 1}.conv.bias"))
    hidden = hidden.clamp(-1.0, 1.0)

    # 6. Extract waveforms per batch item and trim to expected length
    wav = hidden.squeeze(1)  # (batch, audio_length)
    wavs = [wav[i, :audio_lengths[i]].float().detach().cpu().numpy()
            for i in range(wav.shape[0])]

    output_sr = config.get("output_sample_rate", 24000)
    return wavs, output_sr

# ---------------------------------------------------------------------------
# V1 decode pipeline (25Hz tokenizer — DiT + BigVGAN)
# ---------------------------------------------------------------------------

def dit_sinusoidal_embedding(timesteps, dim, scale=1000):
    """Sinusoidal position embedding for diffusion timesteps.

    Args:
        timesteps: (batch,) float tensor of timestep values in [0, 1].
        dim: embedding dimension (typically freq_embed_dim=256).
        scale: scaling factor (default 1000).

    Returns:
        (batch, dim) float tensor.
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = scale * timesteps.unsqueeze(1).float() * emb.unsqueeze(0)
    emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
    return emb.to(timesteps.dtype)

def dit_timestep_embedding(timesteps, weights, prefix):
    """Full timestep embedding: sinusoidal -> Linear -> SiLU -> Linear.

    Args:
        timesteps: (batch,) float tensor.
        weights: weight dict.
        prefix: e.g. 'decoder.dit.time_embed'.

    Returns:
        (batch, hidden_size) float tensor.
    """
    # SinusPositionEmbedding has dim stored in time_mlp.0.weight's input dim
    freq_dim = weights[f"{prefix}.time_mlp.0.weight"].shape[1]
    h = dit_sinusoidal_embedding(timesteps, freq_dim)
    h = h.to(timesteps.dtype)
    # time_mlp: Linear -> SiLU -> Linear (indices 0, 1=SiLU, 2)
    h = F.linear(h, weights[f"{prefix}.time_mlp.0.weight"],
                 weights[f"{prefix}.time_mlp.0.bias"])
    h = F.silu(h)
    h = F.linear(h, weights[f"{prefix}.time_mlp.2.weight"],
                 weights[f"{prefix}.time_mlp.2.bias"])
    return h

def dit_codec_embedding(codes, weights, prefix, repeats, drop_code=False):
    """Embed discrete codes and repeat interleave to mel resolution.

    Args:
        codes: (batch, code_len) int64.
        weights: weight dict.
        prefix: e.g. 'decoder.dit.text_embed'.
        repeats: repeat factor (default 2).
        drop_code: if True, zero out codes for CFG unconditioned branch.

    Returns:
        (batch, code_len * repeats, emb_dim) tensor.
    """
    if drop_code:
        codes = torch.zeros_like(codes)
    embed_weight = weights[f"{prefix}.codec_embed.weight"]
    code_embed = F.embedding(codes, embed_weight)
    code_embed = torch.repeat_interleave(code_embed, repeats=repeats, dim=1)
    return code_embed

def dit_rotary_embedding(seq_len, head_dim, batch_size, dtype, device, base=10000):
    """Compute rotary position embeddings for the DiT model.

    The DiT uses a different rotate pattern (interleaved) compared to V2.
    Returns cos, sin shaped (batch, seq_len, head_dim).

    Args:
        seq_len: sequence length.
        head_dim: dimension per head.
        batch_size: batch size.
        dtype: tensor dtype.
        device: tensor device.
        base: RoPE base (default 10000).

    Returns:
        (cos, sin) each (batch, seq_len, head_dim).
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)  # (seq_len, head_dim//2)
    # Stack pairs: (f0, f0, f1, f1, ...) to match the interleaved rotate pattern
    freqs = torch.stack((freqs, freqs), dim=-1)  # (seq_len, head_dim//2, 2)
    freqs = freqs.reshape(seq_len, -1)  # (seq_len, head_dim)
    freqs = freqs.repeat(batch_size, *([1] * freqs.dim()))  # (batch, seq_len, head_dim)
    cos = freqs.cos().to(dtype=dtype, device=device)
    sin = freqs.sin().to(dtype=dtype, device=device)
    return cos, sin


def dit_apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings using the interleaved (codec) rotate pattern.

    Args:
        q, k: (batch, heads, seq, head_dim).
        cos, sin: (batch, seq, head_dim) -- will be unsqueezed for heads dim.

    Returns:
        Rotated (q, k).
    """
    def rotate_half_codec(x):
        # Interleaved rotation: reshape to (..., d, 2), swap & negate, reshape back
        x = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.reshape(*x.shape[:-2], -1)

    cos = cos.unsqueeze(1)  # (batch, 1, seq, head_dim)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (rotate_half_codec(q) * sin)
    k_embed = (k * cos) + (rotate_half_codec(k) * sin)
    return q_embed, k_embed


def dit_attention(hidden, weights, prefix, config, cos, sin, attn_mask=None):
    """DiT self-attention layer.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: weight dict.
        prefix: e.g. 'decoder.dit.transformer_blocks.0.attn'.
        config: DiT config dict.
        cos, sin: rotary embeddings (batch, seq, head_dim).
        attn_mask: optional (batch, heads, seq, seq) bool mask.

    Returns:
        (batch, seq, hidden_size) attention output.
    """
    n_heads = config.get("num_attention_heads", 16)
    head_dim = config.get("head_dim", 64)
    inner_dim = n_heads * head_dim

    q = F.linear(hidden, weights[f"{prefix}.to_q.weight"],
                 weights.get(f"{prefix}.to_q.bias"))
    k = F.linear(hidden, weights[f"{prefix}.to_k.weight"],
                 weights.get(f"{prefix}.to_k.bias"))
    v = F.linear(hidden, weights[f"{prefix}.to_v.weight"],
                 weights.get(f"{prefix}.to_v.bias"))

    batch, seq, _ = hidden.shape
    q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, n_heads, head_dim).transpose(1, 2)

    # Apply RoPE (interleaved rotation)
    q, k = dit_apply_rotary_pos_emb(q, k, cos, sin)

    # Scaled dot product attention
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale

    if attn_mask is not None:
        # attn_mask is True where attention is allowed
        attn = attn.masked_fill(~attn_mask, float('-inf'))

    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, seq, inner_dim)

    # to_out: Linear + Dropout (skip dropout at inference)
    out = F.linear(out, weights[f"{prefix}.to_out.0.weight"],
                   weights.get(f"{prefix}.to_out.0.bias"))

    return out


def dit_block(hidden, timestep_emb, weights, prefix, config, cos, sin, block_diff,
              look_ahead_block=0, look_backward_block=0):
    """Single DiT decoder layer with AdaLN-Zero + attention + FFN.

    Args:
        hidden: (batch, seq, hidden_size).
        timestep_emb: (batch, hidden_size) time step embedding.
        weights: weight dict.
        prefix: e.g. 'decoder.dit.transformer_blocks.0'.
        config: DiT config dict.
        cos, sin: rotary embeddings.
        block_diff: (batch, heads, seq, seq) block difference tensor.
        look_ahead_block: how many blocks ahead attention can attend.
        look_backward_block: how many blocks backward attention can attend.

    Returns:
        (batch, seq, hidden_size).
    """
    hidden_size = config.get("hidden_size", 1024)

    # --- AdaLayerNormZero for attention ---
    # linear: silu(emb) -> 6 * hidden_size
    ada_emb = F.silu(timestep_emb)
    ada_emb = F.linear(ada_emb,
                       weights[f"{prefix}.attn_norm.linear.weight"],
                       weights[f"{prefix}.attn_norm.linear.bias"])
    # Split into 6 modulation parameters
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
        torch.chunk(ada_emb, 6, dim=1)

    # Normalize hidden (elementwise_affine=False LayerNorm)
    norm = F.layer_norm(hidden, (hidden_size,), eps=1e-6)
    norm = norm * (1 + scale_msa[:, None]) + shift_msa[:, None]

    # --- Attention with block-sparse mask ---
    attn_mask = (block_diff >= -float(look_backward_block)) & \
                (block_diff <= float(look_ahead_block))

    attn_out = dit_attention(norm, weights, f"{prefix}.attn", config,
                             cos, sin, attn_mask=attn_mask)

    hidden = hidden + gate_msa.unsqueeze(1) * attn_out

    # --- FFN with AdaLN modulation ---
    ff_norm = F.layer_norm(hidden, (hidden_size,), eps=1e-6)
    ff_norm = ff_norm * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

    # DiTMLP: Linear -> GELU(tanh) -> Linear (with dropout skipped at inference)
    h = F.linear(ff_norm,
                 weights[f"{prefix}.ff.ff.0.weight"],
                 weights[f"{prefix}.ff.ff.0.bias"])
    h = F.gelu(h, approximate='tanh')
    h = F.linear(h,
                 weights[f"{prefix}.ff.ff.3.weight"],
                 weights[f"{prefix}.ff.ff.3.bias"])

    hidden = hidden + gate_mlp.unsqueeze(1) * h

    return hidden


# ---------------------------------------------------------------------------
# ECAPA-TDNN speaker condition encoder (inside DiT input_embed)
# ---------------------------------------------------------------------------


def ecapa_tdnn_block(x, weights, prefix):
    """TimeDelayNetBlock: Conv1d(same padding, reflect) + ReLU.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix.

    Returns:
        (batch, out_channels, time).
    """
    w = weights[f"{prefix}.conv.weight"]
    b = weights.get(f"{prefix}.conv.bias")
    # Same padding with reflect mode
    kernel_size = w.shape[2]
    dilation = 1
    # Check if this is a dilated conv (for SERes2Net blocks)
    # The dilation is baked into the weight shape via padding="same"
    # For nn.Conv1d with padding="same", PyTorch handles it internally
    # We need to replicate: pad = ((kernel_size - 1) * dilation) // 2
    # But for "same" with reflect, we need to manually pad
    pad_total = (kernel_size - 1) * dilation
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    x = F.pad(x, (pad_left, pad_right), mode='reflect')
    out = F.conv1d(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None)
    out = F.relu(out)
    return out


def ecapa_tdnn_block_dilation(x, weights, prefix, dilation):
    """TimeDelayNetBlock with explicit dilation for Res2Net sub-blocks.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix.
        dilation: dilation factor.

    Returns:
        (batch, out_channels, time).
    """
    w = weights[f"{prefix}.conv.weight"]
    b = weights.get(f"{prefix}.conv.bias")
    kernel_size = w.shape[2]
    pad_total = (kernel_size - 1) * dilation
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    x = F.pad(x, (pad_left, pad_right), mode='reflect')
    out = F.conv1d(x, w.to(x.dtype), b.to(x.dtype) if b is not None else None,
                   dilation=dilation)
    out = F.relu(out)
    return out


def ecapa_res2net_block(x, weights, prefix, scale):
    """Res2NetBlock: split channels, apply TDNNs with cumulative addition.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix for blocks ModuleList.
        scale: number of splits.

    Returns:
        (batch, channels, time).
    """
    chunks = torch.chunk(x, scale, dim=1)
    outputs = []
    output_part = None
    for i, hidden_part in enumerate(chunks):
        if i == 0:
            output_part = hidden_part
        elif i == 1:
            output_part = ecapa_tdnn_block_dilation(
                hidden_part, weights, f"{prefix}.{i - 1}", dilation=1)
        else:
            output_part = ecapa_tdnn_block_dilation(
                hidden_part + output_part, weights, f"{prefix}.{i - 1}", dilation=1)
        outputs.append(output_part)
    return torch.cat(outputs, dim=1)


def ecapa_se_block(x, weights, prefix):
    """Squeeze-Excitation block: mean -> conv1d -> relu -> conv1d -> sigmoid -> scale.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix.

    Returns:
        (batch, channels, time).
    """
    # Global average pooling
    x_mean = x.mean(dim=2, keepdim=True)

    # conv1 + relu
    w1 = weights[f"{prefix}.conv1.weight"]
    b1 = weights.get(f"{prefix}.conv1.bias")
    # "same" padding with reflect for kernel_size=1 is just no padding
    h = F.conv1d(x_mean, w1.to(x.dtype), b1.to(x.dtype) if b1 is not None else None)
    h = F.relu(h, inplace=False)

    # conv2 + sigmoid
    w2 = weights[f"{prefix}.conv2.weight"]
    b2 = weights.get(f"{prefix}.conv2.bias")
    h = F.conv1d(h, w2.to(x.dtype), b2.to(x.dtype) if b2 is not None else None)
    h = torch.sigmoid(h)

    return x * h


def ecapa_se_res2net_block(x, weights, prefix, res2net_scale, dilation):
    """SqueezeExcitationRes2NetBlock: TDNN -> Res2Net -> TDNN -> SE, with residual.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix.
        res2net_scale: Res2Net scale factor.
        dilation: dilation for the Res2Net internal TDNNs.

    Returns:
        (batch, channels, time).
    """
    residual = x

    # tdnn1 (kernel_size=1, dilation=1)
    h = ecapa_tdnn_block(x, weights, f"{prefix}.tdnn1")

    # res2net_block
    h = ecapa_res2net_block(h, weights, f"{prefix}.res2net_block.blocks", res2net_scale)

    # tdnn2 (kernel_size=1, dilation=1)
    h = ecapa_tdnn_block(h, weights, f"{prefix}.tdnn2")

    # se_block
    h = ecapa_se_block(h, weights, f"{prefix}.se_block")

    return h + residual


def ecapa_attentive_stat_pooling(x, weights, prefix, channels):
    """Attentive Statistical Pooling.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: weight key prefix.
        channels: number of channels.

    Returns:
        (batch, channels * 2, 1).
    """
    eps = 1e-12
    batch, C, T = x.shape

    # Mask (all ones for inference)
    mask = torch.ones(batch, 1, T, device=x.device, dtype=x.dtype)
    total = mask.sum(dim=2, keepdim=True)

    # Compute mean and std
    mean = (mask * x).sum(dim=2) / total.squeeze(2)  # (batch, C)
    std = torch.sqrt(((mask * (x - mean.unsqueeze(2)).pow(2)).sum(dim=2) / total.squeeze(2)).clamp(eps))

    mean_expanded = mean.unsqueeze(2).repeat(1, 1, T)
    std_expanded = std.unsqueeze(2).repeat(1, 1, T)
    attention_input = torch.cat([x, mean_expanded, std_expanded], dim=1)  # (batch, 3*C, T)

    # TDNN for attention: Conv1d(3*C, attention_channels, 1) + ReLU
    h = ecapa_tdnn_block(attention_input, weights, f"{prefix}.tdnn")
    # Tanh
    h = torch.tanh(h)
    # Conv1d(attention_channels, C, 1)
    w = weights[f"{prefix}.conv.weight"]
    b = weights.get(f"{prefix}.conv.bias")
    h = F.conv1d(h, w.to(x.dtype), b.to(x.dtype) if b is not None else None)

    # Softmax attention
    attention = F.softmax(h, dim=2)  # (batch, C, T)

    # Weighted mean and std
    w_mean = (attention * x).sum(dim=2)  # (batch, C)
    w_std = torch.sqrt(((attention * (x - w_mean.unsqueeze(2)).pow(2)).sum(dim=2)).clamp(eps))

    pooled = torch.cat([w_mean, w_std], dim=1).unsqueeze(2)  # (batch, 2*C, 1)
    return pooled


def ecapa_tdnn_forward(x, weights, prefix, config):
    """ECAPA-TimeDelayNet forward pass for reference mel conditioning.

    Takes reference mel (batch, mel_T, mel_dim), returns (batch, enc_dim).

    Args:
        x: (batch, mel_T, mel_dim).
        weights: weight dict.
        prefix: e.g. 'decoder.dit.input_embed.spk_encoder'.
        config: DiT config dict with enc_channels, enc_kernel_sizes, etc.

    Returns:
        (batch, enc_dim).
    """
    enc_channels = config.get("enc_channels", [256, 256, 256, 256, 768])
    enc_kernel_sizes = config.get("enc_kernel_sizes", [5, 3, 3, 3, 1])
    enc_dilations = config.get("enc_dilations", [1, 2, 3, 4, 1])
    enc_res2net_scale = config.get("enc_res2net_scale", 2)
    enc_se_channels = config.get("enc_se_channels", 64)
    enc_attention_channels = config.get("enc_attention_channels", 64)

    # Transpose: (batch, mel_T, mel_dim) -> (batch, mel_dim, mel_T)
    h = x.transpose(1, 2)

    hidden_states_list = []

    # blocks.0: initial TDNN layer
    block_prefix = f"{prefix}.blocks.0"
    h = ecapa_tdnn_block(h, weights, block_prefix)
    hidden_states_list.append(h)

    # blocks.1 to blocks.(N-2): SE-Res2Net layers
    for i in range(1, len(enc_channels) - 1):
        block_prefix = f"{prefix}.blocks.{i}"
        h = ecapa_se_res2net_block(h, weights, block_prefix,
                                   enc_res2net_scale, enc_dilations[i])
        hidden_states_list.append(h)

    # Multi-layer feature aggregation: cat all except first, then TDNN
    cat_hidden = torch.cat(hidden_states_list[1:], dim=1)
    h = ecapa_tdnn_block(cat_hidden, weights, f"{prefix}.mfa")

    # Attentive Statistical Pooling
    h = ecapa_attentive_stat_pooling(h, weights, f"{prefix}.asp",
                                     enc_channels[-1])

    # Final Conv1d (kernel_size=1)
    fc_w = weights[f"{prefix}.fc.weight"]
    fc_b = weights.get(f"{prefix}.fc.bias")
    h = F.conv1d(h, fc_w.to(x.dtype), fc_b.to(x.dtype) if fc_b is not None else None)

    return h.squeeze(-1)  # (batch, enc_dim)


def dit_input_embedding(noised_mel, speaker_embedding, ref_mel,
                        code_embed, weights, prefix, config,
                        code_embed_uncond=None, apply_cfg=True):
    """DiT input embedding: concatenate features and project.

    For CFG: doubles the batch (conditioned + unconditioned).

    Args:
        noised_mel: (batch, mel_T, mel_dim) noised mel spectrogram.
        speaker_embedding: (batch, mel_T, enc_emb_dim) xvector repeated.
        ref_mel: (batch, ref_mel_T, mel_dim) reference mel.
        code_embed: (batch, mel_T, emb_dim) code embeddings.
        weights: weight dict.
        prefix: e.g. 'decoder.dit.input_embed'.
        config: DiT config dict.
        code_embed_uncond: (batch, mel_T, emb_dim) unconditioned code embed.
        apply_cfg: whether to apply classifier-free guidance doubling.

    Returns:
        (batch*2 if CFG else batch, mel_T, hidden_size).
    """
    if apply_cfg:
        noised_mel = torch.cat([noised_mel, noised_mel], dim=0)
        speaker_embedding = torch.cat([speaker_embedding,
                                       torch.zeros_like(speaker_embedding)], dim=0)
        ref_mel = torch.cat([ref_mel, torch.zeros_like(ref_mel)], dim=0)
        code_embed = torch.cat([code_embed, code_embed_uncond], dim=0)

    # ECAPA-TDNN speaker encoder on reference mel -> (batch, enc_dim)
    cond_vec = ecapa_tdnn_forward(ref_mel, weights, f"{prefix}.spk_encoder", config)
    cond_vec = cond_vec.unsqueeze(1).repeat(1, noised_mel.size(1), 1)  # (batch, mel_T, enc_dim)

    # Concatenate: noised_mel + cond_vec + code_embed + speaker_embedding
    combined = torch.cat([noised_mel, cond_vec, code_embed, speaker_embedding], dim=-1)

    # Project to hidden_size
    h = F.linear(combined,
                 weights[f"{prefix}.proj.weight"],
                 weights[f"{prefix}.proj.bias"])
    return h


def dit_create_block_diff(seq_len, block_size, num_heads, batch_size, device):
    """Create blockwise difference matrix for block-sparse attention.

    Args:
        seq_len: sequence length.
        block_size: block size for attention masking.
        num_heads: number of attention heads.
        batch_size: batch size.
        device: torch device.

    Returns:
        (batch, heads, seq, seq) int tensor of block differences.
    """
    block_indices = torch.arange(seq_len, device=device) // block_size
    block_i = block_indices.unsqueeze(1)
    block_j = block_indices.unsqueeze(0)
    block_diff = block_j - block_i  # (seq, seq)
    return block_diff.expand(batch_size, num_heads, seq_len, seq_len)


def dit_forward(noised_mel, codes, xvectors, ref_mels, timesteps,
                weights, config, apply_cfg=True):
    """Full DiT forward pass.

    Args:
        noised_mel: (batch, mel_T, mel_dim) noised mel input.
        codes: (batch, code_len) int64 discrete codes.
        xvectors: (batch, mel_T, enc_emb_dim) speaker embeddings (already repeated).
        ref_mels: (batch, ref_mel_T, mel_dim) reference mels.
        timesteps: (batch*2,) float timestep values for both CFG branches.
        weights: weight dict.
        config: DiT config dict.
        apply_cfg: whether to apply classifier-free guidance.

    Returns:
        (batch*2 if CFG else batch, mel_T, mel_dim) predicted velocity.
    """
    dit_prefix = "decoder.dit"
    hidden_size = config.get("hidden_size", 1024)
    num_layers = config.get("num_hidden_layers", 22)
    num_heads = config.get("num_attention_heads", 16)
    head_dim = config.get("head_dim", 64)
    block_size = config.get("block_size", 24)
    repeats = config.get("repeats", 2)
    look_ahead_layers = config.get("look_ahead_layers", [10])
    look_backward_layers = config.get("look_backward_layers", [0, 20])

    # 1. Timestep embedding
    time_emb = dit_timestep_embedding(timesteps, weights, f"{dit_prefix}.time_embed")

    # 2. Code embedding (with unconditioned version for CFG)
    code_embed = dit_codec_embedding(codes, weights, f"{dit_prefix}.text_embed",
                                     repeats, drop_code=False)
    code_embed_uncond = None
    if apply_cfg:
        code_embed_uncond = dit_codec_embedding(codes, weights,
                                                f"{dit_prefix}.text_embed",
                                                repeats, drop_code=True)

    # 3. Input embedding (concatenation + projection + CFG doubling)
    hidden = dit_input_embedding(noised_mel, xvectors, ref_mels,
                                 code_embed, weights,
                                 f"{dit_prefix}.input_embed",
                                 config,
                                 code_embed_uncond=code_embed_uncond,
                                 apply_cfg=apply_cfg)

    batch, seq_len, _ = hidden.shape

    # 4. Rotary embeddings
    cos, sin = dit_rotary_embedding(seq_len, head_dim, batch, hidden.dtype, hidden.device)

    # 5. Block-sparse attention mask
    block_diff = dit_create_block_diff(seq_len, block_size, num_heads, batch, hidden.device)

    # 6. Transformer blocks
    for i in range(num_layers):
        layer_prefix = f"{dit_prefix}.transformer_blocks.{i}"
        look_ahead = 1 if i in look_ahead_layers else 0
        look_backward = 1 if i in look_backward_layers else 0
        hidden = dit_block(hidden, time_emb, weights, layer_prefix, config,
                           cos, sin, block_diff,
                           look_ahead_block=look_ahead,
                           look_backward_block=look_backward)

    # 7. Final norm (AdaLayerNormZero_Final)
    ada_emb = F.silu(time_emb)
    ada_emb = F.linear(ada_emb,
                       weights[f"{dit_prefix}.norm_out.linear.weight"],
                       weights[f"{dit_prefix}.norm_out.linear.bias"])
    scale, shift = torch.chunk(ada_emb, 2, dim=1)
    hidden = F.layer_norm(hidden, (hidden_size,), eps=1e-6)
    hidden = hidden * (1 + scale)[:, None, :] + shift[:, None, :]

    # 8. Output projection -> mel_dim
    output = F.linear(hidden,
                      weights[f"{dit_prefix}.proj_out.weight"],
                      weights[f"{dit_prefix}.proj_out.bias"])

    return output


def dit_sample(codes, xvectors, ref_mels, weights, config,
               num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0):
    """ODE sampling loop for DiT (Euler method).

    Args:
        codes: (batch, code_len) int64 discrete codes.
        xvectors: (batch, enc_emb_dim) speaker xvectors.
        ref_mels: (batch, ref_mel_T, mel_dim) reference mels.
        weights: weight dict.
        config: DiT config dict.
        num_steps: number of ODE steps (default 10).
        guidance_scale: CFG scale (default 0.5).
        sway_coefficient: time schedule sway (default -1.0).

    Returns:
        (batch, mel_dim, mel_T) generated mel spectrogram.
    """
    mel_dim = config.get("mel_dim", 80)
    repeats = config.get("repeats", 2)
    batch_size = codes.shape[0]
    max_duration = codes.shape[1] * repeats

    # Initialize from noise
    noise = torch.randn(batch_size, 30000, mel_dim, dtype=ref_mels.dtype)
    initial_state = noise[:, :max_duration].to(codes.device)

    # Expand xvectors to mel_T
    xvectors_expanded = xvectors.unsqueeze(1).repeat(1, max_duration, 1)

    def ode_fn(t, x):
        batch_2 = batch_size * 2
        if t.ndim == 0:
            t = t.repeat(batch_2)

        if guidance_scale < 1e-5:
            pred = dit_forward(x, codes, xvectors_expanded, ref_mels,
                               t, weights, config, apply_cfg=False)
            return pred

        model_output = dit_forward(x, codes, xvectors_expanded, ref_mels,
                                   t, weights, config, apply_cfg=True)
        cond_pred, uncond_pred = torch.chunk(model_output, 2, dim=0)

        return cond_pred + (cond_pred - uncond_pred) * guidance_scale

    # Time schedule
    t_schedule = torch.linspace(0, 1, num_steps,
                                device=codes.device,
                                dtype=xvectors.dtype)

    if sway_coefficient is not None:
        t_schedule = t_schedule + sway_coefficient * (
            torch.cos(torch.pi / 2 * t_schedule) - 1 + t_schedule)

    # Euler integration
    values = initial_state.clone()
    for t0, t1 in zip(t_schedule[:-1], t_schedule[1:]):
        dt = t1 - t0
        vt = ode_fn(t0, values)
        values = values + vt * dt

    # (batch, mel_T, mel_dim) -> (batch, mel_dim, mel_T)
    return values.permute(0, 2, 1)


# ---------------------------------------------------------------------------
# BigVGAN vocoder (V1 decoder — mel to waveform)
# ---------------------------------------------------------------------------


def bigvgan_kaiser_sinc_filter(cutoff, half_width, kernel_size):
    """Generate a Kaiser-windowed sinc filter for up/down sampling.

    Returns:
        (1, 1, kernel_size) float32 tensor.
    """
    is_even = kernel_size % 2 == 0
    half_size = kernel_size // 2

    delta_f = 4 * half_width
    attenuation = 2.285 * (half_size - 1) * math.pi * delta_f + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0

    kaiser_window = torch.kaiser_window(kernel_size, beta=beta, periodic=False,
                                         dtype=torch.float32)
    if is_even:
        time_indices = torch.arange(-half_size, half_size) + 0.5
    else:
        time_indices = torch.arange(kernel_size) - half_size

    if cutoff == 0:
        return torch.zeros(1, 1, kernel_size, dtype=torch.float32)

    sinc_filter = torch.sinc(2 * cutoff * time_indices)
    normalized = 2 * cutoff * kaiser_window * sinc_filter
    normalized = normalized / normalized.sum()
    return normalized.view(1, 1, kernel_size)


def bigvgan_upsample1d(x, ratio, filt):
    """Anti-aliased upsampling with Kaiser-sinc filter.

    Args:
        x: (batch, channels, time).
        ratio: upsample ratio.
        filt: (1, 1, kernel_size) filter tensor.

    Returns:
        (batch, channels, time * ratio) approximately.
    """
    channels = x.shape[1]
    kernel_size = filt.shape[2]
    stride = ratio
    pad = kernel_size // ratio - 1
    pad_left = pad * stride + (kernel_size - stride) // 2
    pad_right = pad * stride + (kernel_size - stride + 1) // 2

    x = F.pad(x, (pad, pad), mode='replicate')
    filt_expanded = filt.expand(channels, -1, -1).to(x.dtype)
    x = ratio * F.conv_transpose1d(x, filt_expanded, stride=stride, groups=channels)
    x = x[..., pad_left:-pad_right]
    return x


def bigvgan_downsample1d(x, ratio, filt):
    """Anti-aliased downsampling with Kaiser-sinc filter.

    Args:
        x: (batch, channels, time).
        ratio: downsample ratio.
        filt: (1, 1, kernel_size) filter tensor.

    Returns:
        (batch, channels, time // ratio) approximately.
    """
    channels = x.shape[1]
    kernel_size = filt.shape[2]
    even = kernel_size % 2 == 0
    pad_left = kernel_size // 2 - int(even)
    pad_right = kernel_size // 2

    x = F.pad(x, (pad_left, pad_right), mode='replicate')
    filt_expanded = filt.expand(channels, -1, -1).to(x.dtype)
    return F.conv1d(x, filt_expanded, stride=ratio, groups=channels)


def bigvgan_activation1d(x, alpha, beta, up_ratio=2, down_ratio=2):
    """Anti-aliased SnakeBeta activation: upsample -> snake_beta -> downsample.

    Uses precomputed Kaiser-sinc filters.

    Args:
        x: (batch, channels, time).
        alpha: SnakeBeta alpha parameter.
        beta: SnakeBeta beta parameter.
        up_ratio: upsample ratio.
        down_ratio: downsample ratio.

    Returns:
        (batch, channels, time).
    """
    # Compute filters on the fly (they're small)
    up_kernel_size = 12
    down_kernel_size = 12
    up_filt = bigvgan_kaiser_sinc_filter(0.5 / up_ratio, 0.6 / up_ratio,
                                          up_kernel_size).to(x.device)
    down_filt = bigvgan_kaiser_sinc_filter(0.5 / down_ratio, 0.6 / down_ratio,
                                            down_kernel_size).to(x.device)
    x = bigvgan_upsample1d(x, up_ratio, up_filt)
    x = snake_beta(x, alpha, beta)
    x = bigvgan_downsample1d(x, down_ratio, down_filt)
    return x


def bigvgan_causal_conv1d(x, weight, bias=None, dilation=1):
    """Causal 1D convolution (left-padded) for BigVGAN.

    Args:
        x: (batch, channels, time).
        weight: conv weight tensor.
        bias: optional bias tensor.

    Returns:
        (batch, out_channels, time).
    """
    kernel_size = weight.shape[2]
    causal_padding = dilation * (kernel_size - 1)
    x = F.pad(x, [causal_padding, 0])
    groups = x.shape[1] // weight.shape[1]
    return F.conv1d(x, weight.to(x.dtype),
                    bias.to(x.dtype) if bias is not None else None,
                    dilation=dilation, groups=groups)


def bigvgan_amp_block(x, weights, prefix, kernel_size, dilations, causal_type):
    """AMPBlock residual block for BigVGAN.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: e.g. 'decoder.bigvgan.resblocks.0'.
        kernel_size: convolution kernel size.
        dilations: tuple of 3 dilation values, e.g. (1, 3, 5).
        causal_type: '1' (non-causal convs2 + Identity pre_conv) or
                     '2' (causal convs2 + Conv1d pre_conv with SnakeBeta pre_act).

    Returns:
        (batch, channels, time).
    """
    # Pre-conv and pre-act (only for causal_type='2')
    if causal_type == '2':
        # pre_conv is a regular Conv1d with same padding
        pre_w = weights[f"{prefix}.pre_conv.weight"]
        pre_b = weights.get(f"{prefix}.pre_conv.bias")
        pre_ks = pre_w.shape[2]
        pre_pad = (pre_ks - 1) // 2
        hidden = F.conv1d(x, pre_w.to(x.dtype),
                          pre_b.to(x.dtype) if pre_b is not None else None,
                          padding=pre_pad)
        # pre_act is TorchActivation1d(SnakeBeta)
        hidden = bigvgan_activation1d(
            hidden,
            weights[f"{prefix}.pre_act.act.alpha"],
            weights[f"{prefix}.pre_act.act.beta"])
    else:
        # Identity
        hidden = x

    # Main residual loop: 3 iterations
    # activations: [act0, act1, act2, act3, act4, act5]
    # acts1 = activations[0, 2, 4], acts2 = activations[1, 3, 5]
    for i in range(3):
        act1_idx = i * 2
        act2_idx = i * 2 + 1

        # Activation 1 (TorchActivation1d with SnakeBeta)
        hidden = bigvgan_activation1d(
            hidden,
            weights[f"{prefix}.activations.{act1_idx}.act.alpha"],
            weights[f"{prefix}.activations.{act1_idx}.act.beta"])

        # CausalConv1d (convs1.{i})
        hidden = bigvgan_causal_conv1d(
            hidden,
            weights[f"{prefix}.convs1.{i}.weight"],
            weights.get(f"{prefix}.convs1.{i}.bias"),
            dilation=dilations[i])

        # Activation 2
        hidden = bigvgan_activation1d(
            hidden,
            weights[f"{prefix}.activations.{act2_idx}.act.alpha"],
            weights[f"{prefix}.activations.{act2_idx}.act.beta"])

        # convs2.{i} - CausalConv1d (causal_type='2') or regular Conv1d (causal_type='1')
        if causal_type == '1':
            # Regular Conv1d with same padding (dilation=1)
            w2 = weights[f"{prefix}.convs2.{i}.weight"]
            b2 = weights.get(f"{prefix}.convs2.{i}.bias")
            ks = w2.shape[2]
            pad = (ks - 1) // 2
            hidden = F.conv1d(hidden, w2.to(x.dtype),
                              b2.to(x.dtype) if b2 is not None else None,
                              padding=pad)
        else:
            # CausalConv1d (dilation=1)
            hidden = bigvgan_causal_conv1d(
                hidden,
                weights[f"{prefix}.convs2.{i}.weight"],
                weights.get(f"{prefix}.convs2.{i}.bias"),
                dilation=1)

        x = x + hidden

    return x


def bigvgan_process_mel(mel):
    """Process mel spectrogram for BigVGAN: exp -> amplitude_to_db -> normalize.

    Args:
        mel: (batch, mel_dim, mel_T) mel spectrogram.

    Returns:
        (batch, mel_dim, mel_T) processed mel.
    """
    min_db_level = -115

    # amplitude_to_db
    amplitude = torch.exp(mel)
    min_level = torch.exp(
        torch.tensor(min_db_level / 20.0 * np.log(10),
                     device=mel.device, dtype=mel.dtype))
    db = 20 * torch.log10(torch.clamp(amplitude, min=min_level.item()))
    db = db - 20

    # normalize_spectrogram
    max_value = 1
    return torch.clamp(
        (2 * max_value) * ((db - min_db_level) / (-min_db_level)) - max_value,
        -max_value, max_value)


def bigvgan_forward(mel, weights, config):
    """Full BigVGAN vocoder forward pass.

    Args:
        mel: (batch, mel_dim, mel_T) mel spectrogram.
        weights: weight dict.
        config: BigVGAN config dict.

    Returns:
        (batch, audio_T) waveform.
    """
    bigvgan_prefix = "decoder.bigvgan"

    upsample_rates = config.get("upsample_rates", [5, 3, 2, 2, 2, 2])
    upsample_kernel_sizes = config.get("upsample_kernel_sizes", [11, 7, 4, 4, 4, 4])
    upsample_initial_channel = config.get("upsample_initial_channel", 1536)
    resblock_kernel_sizes = config.get("resblock_kernel_sizes", [3, 7, 11])
    resblock_dilation_sizes = config.get("resblock_dilation_sizes",
                                          [[1, 3, 5], [1, 3, 5], [1, 3, 5]])

    num_upsample_layers = len(upsample_rates)
    num_residual_blocks = len(resblock_kernel_sizes)

    # Process mel spectrogram
    h = bigvgan_process_mel(mel)

    # conv_pre: Conv1d(mel_dim, initial_channel, 5, padding=2)
    h = F.conv1d(h,
                 weights[f"{bigvgan_prefix}.conv_pre.weight"].to(h.dtype),
                 weights[f"{bigvgan_prefix}.conv_pre.bias"].to(h.dtype),
                 padding=2)

    # Upsample layers
    for layer_idx in range(num_upsample_layers):
        channels = upsample_initial_channel // (2 ** (layer_idx + 1))
        stride = upsample_rates[layer_idx]
        kernel_size = upsample_kernel_sizes[layer_idx]

        # ConvTranspose1d upsample (inside ups.{layer_idx}.0)
        up_w = weights[f"{bigvgan_prefix}.ups.{layer_idx}.0.weight"]
        up_b = weights.get(f"{bigvgan_prefix}.ups.{layer_idx}.0.bias")
        pad = (kernel_size - stride) // 2
        h = F.conv_transpose1d(h, up_w.to(h.dtype),
                                up_b.to(h.dtype) if up_b is not None else None,
                                stride=stride, padding=pad)

        # Sum residual blocks
        causal_type = '2' if layer_idx <= 1 else '1'
        residual_sum = None
        for block_idx in range(num_residual_blocks):
            rb_idx = layer_idx * num_residual_blocks + block_idx
            rb_prefix = f"{bigvgan_prefix}.resblocks.{rb_idx}"
            rb_out = bigvgan_amp_block(
                h, weights, rb_prefix,
                resblock_kernel_sizes[block_idx],
                resblock_dilation_sizes[block_idx],
                causal_type)
            if residual_sum is None:
                residual_sum = rb_out
            else:
                residual_sum = residual_sum + rb_out
        h = residual_sum / num_residual_blocks

    # activation_post: TorchActivation1d(SnakeBeta)
    h = bigvgan_activation1d(
        h,
        weights[f"{bigvgan_prefix}.activation_post.act.alpha"],
        weights[f"{bigvgan_prefix}.activation_post.act.beta"])

    # conv_post: Conv1d(channels, 1, 7, padding=3, bias=False)
    conv_post_w = weights[f"{bigvgan_prefix}.conv_post.weight"]
    h = F.conv1d(h, conv_post_w.to(h.dtype), padding=3)

    h = torch.clamp(h, min=-1.0, max=1.0)
    return h.squeeze(1)  # (batch, audio_T)


# ---------------------------------------------------------------------------
# V1 full decode pipeline (25Hz tokenizer)
# ---------------------------------------------------------------------------


def v1_decode(state, codes_dict):
    """Decode V1 (25Hz) codes to waveform using DiT + BigVGAN.

    Args:
        state: dict from load_tokenizer().
        codes_dict: dict with:
            'audio_codes': (batch, code_length) or list of int64 tensors.
            'xvectors': (batch, enc_emb_dim) or list of float tensors.
            'ref_mels': (batch, ref_mel_T, mel_dim) or list of float tensors.

    Returns:
        (list of 1D float32 numpy arrays, int sample_rate).
    """
    weights = state["weights"]
    config = state["config"]

    # Extract DiT and BigVGAN configs
    dec_cfg = config.get("decoder_config", {})
    dit_cfg = dec_cfg.get("dit_config", dec_cfg)
    bigvgan_cfg = dec_cfg.get("bigvgan_config", dec_cfg)

    # Get parameters
    repeats = dit_cfg.get("repeats", 2)
    decode_upsample_rate = config.get("decode_upsample_rate", 1920)

    # Prepare inputs
    audio_codes = codes_dict["audio_codes"]
    xvectors = codes_dict["xvectors"]
    ref_mels = codes_dict["ref_mels"]

    # Handle list inputs: pad to batch tensors
    if isinstance(audio_codes, list):
        max_code_len = max(c.shape[0] for c in audio_codes)
        padded_codes = torch.full((len(audio_codes), max_code_len), -1,
                                  dtype=torch.long)
        for i, c in enumerate(audio_codes):
            padded_codes[i, :c.shape[0]] = c
        audio_codes = padded_codes

    if isinstance(xvectors, list):
        xvectors = torch.stack(xvectors)

    if isinstance(ref_mels, list):
        # Pad ref_mels to same length
        max_ref_len = max(m.shape[0] for m in ref_mels)
        mel_dim = ref_mels[0].shape[1] if ref_mels[0].dim() == 2 else ref_mels[0].shape[-1]
        padded_refs = torch.zeros(len(ref_mels), max_ref_len, mel_dim,
                                  dtype=ref_mels[0].dtype)
        for i, m in enumerate(ref_mels):
            padded_refs[i, :m.shape[0]] = m
        ref_mels = padded_refs

    # Compute expected audio lengths
    audio_lengths = (audio_codes > -1).sum(1) * decode_upsample_rate

    # Clamp codes (padding was -1)
    audio_codes = torch.clamp(audio_codes, min=0)

    # 1. DiT sampling: codes -> mel spectrogram
    mel = dit_sample(audio_codes, xvectors, ref_mels, weights, dit_cfg,
                     num_steps=10, guidance_scale=0.5, sway_coefficient=-1.0)
    # mel: (batch, mel_dim, mel_T)

    # 2. BigVGAN: mel -> waveform
    wav = bigvgan_forward(mel, weights, bigvgan_cfg)
    # wav: (batch, audio_T)

    # 3. Trim to expected lengths
    wavs = [wav[i, :audio_lengths[i]].float().detach().cpu().numpy()
            for i in range(wav.shape[0])]

    output_sr = config.get("output_sample_rate", 24000)
    return wavs, output_sr


# ---------------------------------------------------------------------------
# V2 encoder pipeline (12Hz tokenizer — audio to codes)
# ---------------------------------------------------------------------------


def encoder_resblock(x, weights, prefix):
    """Encoder residual block: ELU -> Conv(C->C//2, k=3) -> ELU -> Conv(C//2->C, k=1) + residual.

    Args:
        x: (batch, channels, time).
        weights: weight dict.
        prefix: e.g. 'encoder.encoder.layers.1'.

    Returns:
        (batch, channels, time).
    """
    residual = x
    h = F.elu(x)
    h = causal_conv1d(h, weights[f"{prefix}.block.1.conv.weight"],
                      weights.get(f"{prefix}.block.1.conv.bias"))
    h = F.elu(h)
    h = causal_conv1d(h, weights[f"{prefix}.block.3.conv.weight"],
                      weights.get(f"{prefix}.block.3.conv.bias"))
    return h + residual


def encoder_conv_forward(x, weights, prefix):
    """Encoder convolutional backbone.

    Input: (batch, 1, samples).
    Hardcoded layer map from weight keys:
    - Layer 0: CausalConv(1->64, k=7)
    - Layer 1: ResBlock(64), Layer 2: ELU
    - Layer 3: CausalConv(64->128, k=8, stride=4)
    - Layer 4: ResBlock(128), Layer 5: ELU
    - Layer 6: CausalConv(128->256, k=10, stride=5)
    - Layer 7: ResBlock(256), Layer 8: ELU
    - Layer 9: CausalConv(256->512, k=12, stride=6)
    - Layer 10: ResBlock(512), Layer 11: ELU
    - Layer 12: CausalConv(512->1024, k=16, stride=8)
    - Layer 13: ELU
    - Layer 14: CausalConv(1024->512, k=3)

    Returns:
        (batch, 512, T').
    """
    # Layer 0: initial conv
    h = causal_conv1d(x, weights[f"{prefix}.layers.0.conv.weight"],
                      weights.get(f"{prefix}.layers.0.conv.bias"))

    # Downsampling stages: (resblock, ELU, strided conv) x 4
    strides = [4, 5, 6, 8]
    resblock_layers = [1, 4, 7, 10]
    conv_layers = [3, 6, 9, 12]

    for rb_idx, conv_idx, stride in zip(resblock_layers, conv_layers, strides):
        h = encoder_resblock(h, weights, f"{prefix}.layers.{rb_idx}")
        h = F.elu(h)
        h = causal_conv1d(h, weights[f"{prefix}.layers.{conv_idx}.conv.weight"],
                          weights.get(f"{prefix}.layers.{conv_idx}.conv.bias"),
                          stride=stride)

    # Layer 13: ELU
    h = F.elu(h)
    # Layer 14: final conv
    h = causal_conv1d(h, weights[f"{prefix}.layers.14.conv.weight"],
                      weights.get(f"{prefix}.layers.14.conv.bias"))

    return h


def encoder_transformer_attention(hidden, weights, prefix, config, cos, sin):
    """Single attention layer for encoder transformer (bidirectional, no QK-norm).

    Args:
        hidden: (batch, seq, hidden_size).
        weights: weight dict.
        prefix: e.g. 'encoder.encoder_transformer.layers.0.self_attn'.
        config: encoder_config dict.
        cos, sin: rotary embeddings (seq, head_dim).

    Returns:
        (batch, seq, hidden_size).
    """
    n_heads = config["num_attention_heads"]
    head_dim = config.get("head_dim", config["hidden_size"] // n_heads)

    q = F.linear(hidden, weights[f"{prefix}.q_proj.weight"])
    k = F.linear(hidden, weights[f"{prefix}.k_proj.weight"])
    v = F.linear(hidden, weights[f"{prefix}.v_proj.weight"])

    batch, seq, _ = hidden.shape
    q = q.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    k = k.view(batch, seq, n_heads, head_dim).transpose(1, 2)
    v = v.view(batch, seq, n_heads, head_dim).transpose(1, 2)

    # Apply RoPE (standard 1D)
    q, k = apply_rotary_pos_emb_simple(q, k, cos[:seq], sin[:seq])

    # Scaled dot product attention — NO causal mask (bidirectional)
    scale = head_dim ** -0.5
    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
    out = torch.matmul(attn, v)

    out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
    out = F.linear(out, weights[f"{prefix}.o_proj.weight"])
    return out


def encoder_transformer_layer(hidden, weights, prefix, config, cos, sin):
    """Single encoder transformer layer: LayerNorm + attn + layer_scale + LayerNorm + GELU FFN + layer_scale.

    Args:
        hidden: (batch, seq, hidden_size).
        weights: weight dict.
        prefix: e.g. 'encoder.encoder_transformer.layers.0'.
        config: encoder_config dict.
        cos, sin: rotary embeddings.

    Returns:
        (batch, seq, hidden_size).
    """
    eps = config.get("norm_eps", 1e-5)

    # Pre-norm attention (LayerNorm with bias)
    normed = layer_norm(hidden, weights[f"{prefix}.input_layernorm.weight"],
                        weights[f"{prefix}.input_layernorm.bias"], eps)
    attn_out = encoder_transformer_attention(
        normed, weights, f"{prefix}.self_attn", config, cos, sin)

    # Layer scale
    ls_key = f"{prefix}.self_attn_layer_scale.scale"
    if ls_key in weights:
        attn_out = attn_out * weights[ls_key]
    hidden = hidden + attn_out

    # Pre-norm FFN (GELU, not SwiGLU)
    normed = layer_norm(hidden, weights[f"{prefix}.post_attention_layernorm.weight"],
                        weights[f"{prefix}.post_attention_layernorm.bias"], eps)
    ffn_out = F.linear(normed, weights[f"{prefix}.mlp.fc1.weight"])
    ffn_out = F.gelu(ffn_out)
    ffn_out = F.linear(ffn_out, weights[f"{prefix}.mlp.fc2.weight"])

    ls_key = f"{prefix}.mlp_layer_scale.scale"
    if ls_key in weights:
        ffn_out = ffn_out * weights[ls_key]
    hidden = hidden + ffn_out

    return hidden


def encoder_transformer_forward(hidden, weights, prefix, config):
    """Full encoder transformer (8 layers, bidirectional).

    Args:
        hidden: (batch, seq, hidden_size=512).
        weights: weight dict.
        prefix: e.g. 'encoder.encoder_transformer'.
        config: encoder_config dict.

    Returns:
        (batch, seq, hidden_size).
    """
    n_layers = config["num_hidden_layers"]
    head_dim = config.get("head_dim", 64)
    theta = config.get("rope_theta", 10000.0)
    seq_len = hidden.shape[1]

    # No input/output projection (hidden=512 matches conv output)
    cos, sin = rotary_embedding(seq_len, head_dim, theta)
    cos, sin = cos.to(hidden.device), sin.to(hidden.device)

    for i in range(n_layers):
        hidden = encoder_transformer_layer(
            hidden, weights, f"{prefix}.layers.{i}", config, cos, sin)

    # No final norm (encoder transformer has none)
    return hidden


def vq_codebook_encode(x, cluster_usage, embed_sum, eps=1e-5):
    """Encode vectors to nearest codebook indices using Euclidean distance.

    Args:
        x: (T, dim) float tensor.
        cluster_usage: (codebook_size,) tensor.
        embed_sum: (codebook_size, dim) tensor.

    Returns:
        codes: (T,) int64 tensor.
        quantized: (T, dim) tensor.
    """
    embedding = embed_sum / cluster_usage.clamp(min=eps).unsqueeze(-1)
    dists = torch.cdist(x[None], embedding[None], p=2)[0]  # (T, codebook_size)
    codes = dists.argmin(dim=-1)  # (T,)
    quantized = F.embedding(codes, embedding)
    return codes, quantized


def split_rvq_encode(hidden, weights, prefix, n_q_semantic, n_q_total, config):
    """SplitResidualVectorQuantizer encode.

    Args:
        hidden: (batch, hidden_dim, T) float tensor (e.g. 512-dim from downsample).
        weights: weight dict.
        prefix: e.g. 'encoder.quantizer'.
        n_q_semantic: number of semantic quantizers (typically 1).
        n_q_total: total quantizers to use (typically 16).
        config: encoder_config dict.

    Returns:
        codes: (batch, T, n_q_total) int64 tensor.
    """
    n_q_acoustic = n_q_total - n_q_semantic

    all_codes = []
    for b in range(hidden.shape[0]):
        h = hidden[b:b+1]  # (1, hidden_dim, T)

        # --- Semantic RVQ ---
        sem_prefix = f"{prefix}.semantic_residual_vector_quantizer"
        # input_proj: Conv1d (hidden_dim -> codebook_dim, k=1)
        sem_h = conv1d(h, weights[f"{sem_prefix}.input_proj.weight"])  # (1, codebook_dim, T)
        residual = sem_h.clone()

        sem_codes_list = []
        for i in range(n_q_semantic):
            vq_prefix = f"{sem_prefix}.layers.{i}"
            # permute to (T, codebook_dim) for VQ encode
            r_t = residual[0].permute(1, 0)  # (T, codebook_dim)
            codes_i, quantized_i = vq_codebook_encode(
                r_t,
                weights[f"{vq_prefix}.codebook.cluster_usage"],
                weights[f"{vq_prefix}.codebook.embed_sum"])
            sem_codes_list.append(codes_i)
            residual = residual - quantized_i.permute(1, 0).unsqueeze(0)

        # --- Acoustic RVQ ---
        aco_prefix = f"{prefix}.acoustic_residual_vector_quantizer"
        aco_h = conv1d(h, weights[f"{aco_prefix}.input_proj.weight"])  # (1, codebook_dim, T)
        residual = aco_h.clone()

        aco_codes_list = []
        for i in range(n_q_acoustic):
            vq_prefix = f"{aco_prefix}.layers.{i}"
            r_t = residual[0].permute(1, 0)  # (T, codebook_dim)
            codes_i, quantized_i = vq_codebook_encode(
                r_t,
                weights[f"{vq_prefix}.codebook.cluster_usage"],
                weights[f"{vq_prefix}.codebook.embed_sum"])
            aco_codes_list.append(codes_i)
            residual = residual - quantized_i.permute(1, 0).unsqueeze(0)

        # Stack: semantic codes first, then acoustic
        step_codes = torch.stack(sem_codes_list + aco_codes_list, dim=-1)  # (T, n_q_total)
        all_codes.append(step_codes)

    return torch.stack(all_codes)  # (batch, T, n_q_total)


def v2_encode(state, audio, sample_rate):
    """Encode audio to V2 (12Hz) codes.

    Full pipeline: resample -> conv_encoder -> transpose -> transformer -> transpose -> downsample -> quantize.

    Args:
        state: dict from load_tokenizer().
        audio: (batch, samples) or (samples,) float tensor.
        sample_rate: int, input sample rate.

    Returns:
        codes: (batch, T, n_q_total) int64 tensor.
    """
    weights = state["weights"]
    config = state["config"]
    enc_cfg = config.get("encoder_config", {})

    target_sr = config.get("input_sample_rate", 24000)
    n_q_total = config.get("encoder_valid_num_quantizers", 16)
    n_q_semantic = enc_cfg.get("num_semantic_quantizers", 1)

    if audio.dim() == 1:
        audio = audio.unsqueeze(0)

    # Resample if needed
    if sample_rate != target_sr:
        import torchaudio.functional as AF
        audio = AF.resample(audio, sample_rate, target_sr)

    # (batch, samples) -> (batch, 1, samples)
    x = audio.unsqueeze(1).float()

    # 1. Conv encoder
    x = encoder_conv_forward(x, weights, "encoder.encoder")  # (batch, 512, T')

    # 2. Transformer (expects (batch, seq, hidden))
    x = x.transpose(1, 2)  # (batch, T', 512)
    x = encoder_transformer_forward(x, weights, "encoder.encoder_transformer", enc_cfg)
    x = x.transpose(1, 2)  # (batch, 512, T')

    # 3. Compress downsample: CausalConv(512->512, k=4, stride=2)
    x = causal_conv1d(x, weights["encoder.downsample.conv.weight"],
                      weights.get("encoder.downsample.conv.bias"),
                      stride=enc_cfg.get("compress", 2))

    # 4. Quantize
    codes = split_rvq_encode(x, weights, "encoder.quantizer",
                             n_q_semantic, n_q_total, enc_cfg)

    return codes  # (batch, T, n_q_total)


def encode(state, audio, sample_rate):
    """Top-level encode dispatcher.

    Args:
        state: dict from load_tokenizer().
        audio: float tensor of audio samples.
        sample_rate: int, input sample rate.

    Returns:
        codes tensor (shape depends on model type).
    """
    model_type = state["model_type"]

    if model_type == "qwen3_tts_tokenizer_12hz":
        return v2_encode(state, audio, sample_rate)
    else:
        raise ValueError(f"Encode not supported for model_type: {model_type}")


def decode(state, codes_dict):
    """Top-level decode dispatcher.

    Args:
        state: dict from load_tokenizer().
        codes_dict: dict with 'audio_codes' tensor.
            For V1: also 'xvectors', 'ref_mels'.

    Returns:
        (list of np.ndarray waveforms, int sample_rate).
    """
    model_type = state["model_type"]

    if model_type == "qwen3_tts_tokenizer_12hz":
        audio_codes = codes_dict["audio_codes"]
        if isinstance(audio_codes, list):
            # Pad variable-length codes into a single tensor
            max_len = max(c.shape[0] for c in audio_codes)
            n_q = audio_codes[0].shape[1] if audio_codes[0].dim() == 2 else 1
            padded = torch.full((len(audio_codes), max_len, n_q), -1,
                                dtype=torch.long)
            for i, c in enumerate(audio_codes):
                padded[i, :c.shape[0]] = c
            audio_codes = padded
        elif isinstance(audio_codes, torch.Tensor) and audio_codes.dim() == 2:
            # (code_length, num_quantizers) -> (1, code_length, num_quantizers)
            audio_codes = audio_codes.unsqueeze(0)
        return v2_decode(state, audio_codes)

    elif model_type == "qwen3_tts_tokenizer_25hz":
        return v1_decode(state, codes_dict)

    else:
        raise ValueError(f"Unknown tokenizer model_type: {model_type}")
