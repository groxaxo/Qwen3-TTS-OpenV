import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_functional.qwen3_tts_tokenizer import layer_norm, rms_norm, swiglu_ffn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rotary_pos_emb_simple(q, k, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


def _causal_conv1d(x, weight, bias=None, stride=1, dilation=1):
    kernel_size = weight.shape[2]
    pad = (kernel_size - 1) * dilation
    x = F.pad(x, (pad, 0))
    groups = x.shape[1] // weight.shape[1]
    return F.conv1d(
        x,
        weight.to(x.dtype),
        bias.to(x.dtype) if bias is not None else None,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )


def _conv1d(x, weight, bias=None):
    return F.conv1d(
        x.to(weight.dtype),
        weight,
        bias.to(weight.dtype) if bias is not None else None,
    )


def _causal_conv_transpose1d(x, weight, bias=None, stride=1):
    out = F.conv_transpose1d(
        x,
        weight.to(x.dtype),
        bias.to(x.dtype) if bias is not None else None,
        stride=stride,
    )
    pad = weight.shape[2] - stride
    if pad > 0:
        out = out[..., :-pad]
    return out


def _snake_beta(x, alpha_param, beta_param):
    alpha = torch.exp(alpha_param).unsqueeze(0).unsqueeze(-1)
    beta = torch.exp(beta_param).unsqueeze(0).unsqueeze(-1)
    return x + (1.0 / (beta + 1e-9)) * torch.pow(torch.sin(x * alpha), 2)


class VectorQuantizerDecode(nn.Module):
    def __init__(self, weights: dict, prefix: str, eps: float = 1e-5):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("cluster_usage", weights[f"{prefix}._codebook.cluster_usage"])
        self.register_buffer("embedding_sum", weights[f"{prefix}._codebook.embedding_sum"])
        self.register_buffer("project_out_w", weights.get(f"{prefix}.project_out.weight"))
        self.register_buffer("project_out_b", weights.get(f"{prefix}.project_out.bias"))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding_sum / self.cluster_usage.clamp(min=self.eps).unsqueeze(-1)
        quantized = F.embedding(codes.to(torch.long), embedding)
        if self.project_out_w is not None:
            quantized = F.linear(quantized, self.project_out_w, self.project_out_b)
        return quantized.transpose(1, 2)


class ResidualVectorQuantizerDecode(nn.Module):
    def __init__(self, weights: dict, prefix: str, num_layers: int, eps: float = 1e-5):
        super().__init__()
        self.layers = nn.ModuleList(
            [VectorQuantizerDecode(weights, f"{prefix}.layers.{i}", eps=eps) for i in range(int(num_layers))]
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        decoded = [layer(codes[:, i, :]) for i, layer in enumerate(self.layers)]
        return torch.stack(decoded, dim=0).sum(dim=0)


class SplitRVQDecoder(nn.Module):
    def __init__(self, weights: dict, prefix: str, config: dict):
        super().__init__()
        self.n_q_semantic = int(config.get("num_semantic_quantizers", 1))
        self.n_q_total = int(config.get("num_quantizers", 16))

        self.semantic_decoder = ResidualVectorQuantizerDecode(
            weights,
            f"{prefix}.rvq_first.vq",
            self.n_q_semantic,
        )
        self.register_buffer("semantic_proj_w", weights.get(f"{prefix}.rvq_first.output_proj.weight"))

        n_q_acoustic = self.n_q_total - self.n_q_semantic
        self.acoustic_decoder = None
        if n_q_acoustic > 0:
            self.acoustic_decoder = ResidualVectorQuantizerDecode(
                weights,
                f"{prefix}.rvq_rest.vq",
                n_q_acoustic,
            )
        self.register_buffer("acoustic_proj_w", weights.get(f"{prefix}.rvq_rest.output_proj.weight"))

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        semantic = self.semantic_decoder(codes[:, : self.n_q_semantic, :])
        if self.semantic_proj_w is not None:
            semantic = _conv1d(semantic, self.semantic_proj_w)

        result = semantic
        if self.acoustic_decoder is not None:
            acoustic = self.acoustic_decoder(codes[:, self.n_q_semantic : self.n_q_total, :])
            if self.acoustic_proj_w is not None:
                acoustic = _conv1d(acoustic, self.acoustic_proj_w)
            result = result + acoustic
        return result


class SpeechDecoderTransformerLayer(nn.Module):
    def __init__(self, weights: dict, prefix: str, config: dict):
        super().__init__()
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config.get("head_dim", 64))
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.scale = self.head_dim ** -0.5
        self.gqa_reps = self.n_heads // self.n_kv_heads

        self.register_buffer("ln1", weights[f"{prefix}.input_layernorm.weight"])
        self.register_buffer("ln2", weights[f"{prefix}.post_attention_layernorm.weight"])
        self.register_buffer("q_proj", weights[f"{prefix}.self_attn.q_proj.weight"])
        self.register_buffer("k_proj", weights[f"{prefix}.self_attn.k_proj.weight"])
        self.register_buffer("v_proj", weights[f"{prefix}.self_attn.v_proj.weight"])
        self.register_buffer("o_proj", weights[f"{prefix}.self_attn.o_proj.weight"])
        self.register_buffer("gate_proj", weights[f"{prefix}.mlp.gate_proj.weight"])
        self.register_buffer("up_proj", weights[f"{prefix}.mlp.up_proj.weight"])
        self.register_buffer("down_proj", weights[f"{prefix}.mlp.down_proj.weight"])

        reference_dtype = self.ln1.dtype
        self.register_buffer(
            "attn_scale",
            weights.get(f"{prefix}.self_attn_layer_scale.scale", torch.ones(1, dtype=reference_dtype)),
        )
        self.register_buffer(
            "mlp_scale",
            weights.get(f"{prefix}.mlp_layer_scale.scale", torch.ones(1, dtype=reference_dtype)),
        )

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        normed = rms_norm(hidden, self.ln1, self.eps)
        q = F.linear(normed, self.q_proj)
        k = F.linear(normed, self.k_proj)
        v = F.linear(normed, self.v_proj)

        batch, seq, _ = hidden.shape
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = _apply_rotary_pos_emb_simple(q, k, cos[:seq], sin[:seq])
        if self.n_kv_heads < self.n_heads:
            k = k.repeat_interleave(self.gqa_reps, dim=1)
            v = v.repeat_interleave(self.gqa_reps, dim=1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.full((seq, seq), float("-inf"), device=attn.device), diagonal=1)
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = F.linear(out, self.o_proj)
        hidden = hidden + (out * self.attn_scale)

        normed = rms_norm(hidden, self.ln2, self.eps)
        ffn = swiglu_ffn(normed, self.gate_proj, self.up_proj, self.down_proj)
        hidden = hidden + (ffn * self.mlp_scale)
        return hidden


class ConvNeXtBlock(nn.Module):
    def __init__(self, weights: dict, prefix: str):
        super().__init__()
        self.register_buffer("dwconv_w", weights[f"{prefix}.dwconv.conv.weight"])
        self.register_buffer("dwconv_b", weights.get(f"{prefix}.dwconv.conv.bias"))
        self.register_buffer("norm_w", weights[f"{prefix}.norm.weight"])
        self.register_buffer("norm_b", weights[f"{prefix}.norm.bias"])
        self.register_buffer("pwconv1_w", weights[f"{prefix}.pwconv1.weight"])
        self.register_buffer("pwconv1_b", weights.get(f"{prefix}.pwconv1.bias"))
        self.register_buffer("pwconv2_w", weights[f"{prefix}.pwconv2.weight"])
        self.register_buffer("pwconv2_b", weights.get(f"{prefix}.pwconv2.bias"))

        default_gamma = torch.ones(self.pwconv2_w.shape[0], dtype=self.pwconv2_w.dtype)
        self.register_buffer("gamma", weights.get(f"{prefix}.gamma", default_gamma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _causal_conv1d(x, self.dwconv_w, self.dwconv_b)
        h = layer_norm(h.transpose(1, 2), self.norm_w, self.norm_b)
        h = F.linear(h, self.pwconv1_w, self.pwconv1_b)
        h = F.gelu(h)
        h = F.linear(h, self.pwconv2_w, self.pwconv2_b)
        h = h.transpose(1, 2)
        h = h * self.gamma.unsqueeze(0).unsqueeze(-1)
        return x + h


class UpsampleBlock(nn.Module):
    def __init__(self, weights: dict, prefix: str, ratio: int):
        super().__init__()
        self.ratio = int(ratio)
        self.register_buffer("conv_w", weights[f"{prefix}.0.conv.weight"])
        self.register_buffer("conv_b", weights.get(f"{prefix}.0.conv.bias"))
        self.convnext = ConvNeXtBlock(weights, f"{prefix}.1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _causal_conv_transpose1d(x, self.conv_w, self.conv_b, stride=self.ratio)
        return self.convnext(x)


class ResidualUnit(nn.Module):
    def __init__(self, weights: dict, prefix: str, dilation: int):
        super().__init__()
        self.dilation = int(dilation)
        self.register_buffer("act1_alpha", weights[f"{prefix}.act1.alpha"])
        self.register_buffer("act1_beta", weights[f"{prefix}.act1.beta"])
        self.register_buffer("conv1_w", weights[f"{prefix}.conv1.conv.weight"])
        self.register_buffer("conv1_b", weights.get(f"{prefix}.conv1.conv.bias"))
        self.register_buffer("act2_alpha", weights[f"{prefix}.act2.alpha"])
        self.register_buffer("act2_beta", weights[f"{prefix}.act2.beta"])
        self.register_buffer("conv2_w", weights[f"{prefix}.conv2.conv.weight"])
        self.register_buffer("conv2_b", weights.get(f"{prefix}.conv2.conv.bias"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = _snake_beta(x, self.act1_alpha, self.act1_beta)
        h = _causal_conv1d(h, self.conv1_w, self.conv1_b, dilation=self.dilation)
        h = _snake_beta(h, self.act2_alpha, self.act2_beta)
        h = _causal_conv1d(h, self.conv2_w, self.conv2_b)
        return x + h


class VocoderBlock(nn.Module):
    def __init__(self, weights: dict, prefix: str, upsample_rate: int, dilations=(1, 3, 9)):
        super().__init__()
        self.upsample_rate = int(upsample_rate)
        self.register_buffer("alpha0", weights[f"{prefix}.block.0.alpha"])
        self.register_buffer("beta0", weights[f"{prefix}.block.0.beta"])
        self.register_buffer("conv1_w", weights[f"{prefix}.block.1.conv.weight"])
        self.register_buffer("conv1_b", weights.get(f"{prefix}.block.1.conv.bias"))
        self.residual_units = nn.ModuleList(
            [
                ResidualUnit(weights, f"{prefix}.block.{i + 2}", dilation)
                for i, dilation in enumerate(dilations)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _snake_beta(x, self.alpha0, self.beta0)
        x = _causal_conv_transpose1d(x, self.conv1_w, self.conv1_b, stride=self.upsample_rate)
        for residual_unit in self.residual_units:
            x = residual_unit(x)
        return x


class SpeechDecoder(nn.Module):
    """nn.Module wrapper for the V2 speech decoder after RVQ decode."""

    def __init__(self, weights: dict, config: dict):
        super().__init__()

        self.upsample_rates = list(config.get("upsample_rates", [8, 5, 4, 3]))
        self.upsampling_ratios = list(config.get("upsampling_ratios", [2, 2]))
        self.n_layers = int(config["num_hidden_layers"])
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config.get("head_dim", 64))
        self.eps = float(config.get("rms_norm_eps", 1e-5))
        self.theta = float(config.get("rope_theta", 10000.0))
        self._weight_dtype = weights["decoder.pre_conv.conv.weight"].dtype

        self.register_buffer("pre_conv_w", weights["decoder.pre_conv.conv.weight"])
        self.register_buffer("pre_conv_b", weights.get("decoder.pre_conv.conv.bias"))
        self.register_buffer("pre_trans_input_proj_w", weights["decoder.pre_transformer.input_proj.weight"])
        self.register_buffer("pre_trans_input_proj_b", weights.get("decoder.pre_transformer.input_proj.bias"))
        self.register_buffer("pre_trans_output_proj_w", weights["decoder.pre_transformer.output_proj.weight"])
        self.register_buffer("pre_trans_output_proj_b", weights.get("decoder.pre_transformer.output_proj.bias"))
        self.register_buffer("pre_trans_norm_w", weights["decoder.pre_transformer.norm.weight"])

        self.pre_transformer_layers = nn.ModuleList(
            [
                SpeechDecoderTransformerLayer(weights, f"decoder.pre_transformer.layers.{i}", config)
                for i in range(self.n_layers)
            ]
        )
        self.upsample_blocks = nn.ModuleList(
            [
                UpsampleBlock(weights, f"decoder.upsample.{i}", ratio)
                for i, ratio in enumerate(self.upsampling_ratios)
            ]
        )

        self.register_buffer("decoder0_w", weights["decoder.decoder.0.conv.weight"])
        self.register_buffer("decoder0_b", weights.get("decoder.decoder.0.conv.bias"))
        self.decoder_blocks = nn.ModuleList(
            [
                VocoderBlock(weights, f"decoder.decoder.{i + 1}", rate)
                for i, rate in enumerate(self.upsample_rates)
            ]
        )

        final_idx = 1 + len(self.upsample_rates)
        self.register_buffer("final_alpha", weights[f"decoder.decoder.{final_idx}.alpha"])
        self.register_buffer("final_beta", weights[f"decoder.decoder.{final_idx}.beta"])
        self.register_buffer("final_conv_w", weights[f"decoder.decoder.{final_idx + 1}.conv.weight"])
        self.register_buffer("final_conv_b", weights.get(f"decoder.decoder.{final_idx + 1}.conv.bias"))

    def _pre_transformer(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(hidden, self.pre_trans_input_proj_w, self.pre_trans_input_proj_b)
        seq = hidden.shape[1]
        inv_freq = 1.0 / (
            self.theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=hidden.device) / self.head_dim)
        )
        positions = torch.arange(seq, dtype=torch.float32, device=hidden.device)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(hidden.device)
        sin = emb.sin().to(hidden.device)

        for layer in self.pre_transformer_layers:
            hidden = layer(hidden, cos, sin)

        hidden = rms_norm(hidden, self.pre_trans_norm_w, self.eps)
        hidden = F.linear(hidden, self.pre_trans_output_proj_w, self.pre_trans_output_proj_b)
        return hidden

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = latent.to(self._weight_dtype)
        hidden = _causal_conv1d(hidden, self.pre_conv_w, self.pre_conv_b)
        hidden = hidden.transpose(1, 2)
        hidden = self._pre_transformer(hidden)
        hidden = hidden.transpose(1, 2)

        for upsample_block in self.upsample_blocks:
            hidden = upsample_block(hidden)

        hidden = _causal_conv1d(hidden, self.decoder0_w, self.decoder0_b)
        for decoder_block in self.decoder_blocks:
            hidden = decoder_block(hidden)

        hidden = _snake_beta(hidden, self.final_alpha, self.final_beta)
        hidden = _causal_conv1d(hidden, self.final_conv_w, self.final_conv_b)
        return torch.clamp(hidden, -1.0, 1.0)


class IntegerInputSpeechDecoder(nn.Module):
    """Traceable wrapper that decodes integer RVQ codes before vocoding."""

    def __init__(self, weights: dict, config: dict):
        super().__init__()
        self.num_quantizers = int(config.get("num_quantizers", 16))
        self.rvq_decoder = SplitRVQDecoder(weights, "decoder.quantizer", config)
        self.speech_decoder = SpeechDecoder(weights, config)

    def _validate_codes(self, codes: torch.Tensor) -> None:
        if codes.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise TypeError("codes must have an integer dtype")
        if codes.ndim != 3 or codes.shape[1] != self.num_quantizers:
            raise ValueError(f"codes must have shape (batch, {self.num_quantizers}, time)")

    def decode_latent(self, codes: torch.Tensor) -> torch.Tensor:
        self._validate_codes(codes)
        return self.rvq_decoder(codes)

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.speech_decoder(self.decode_latent(codes))
