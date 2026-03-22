from collections import namedtuple
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen3_tts_tokenizer import rms_norm, swiglu_ffn


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _make_explicit_forward(n_layers: int):
    cache_args = []
    passthrough = []
    for i in range(n_layers):
        cache_args.extend([f"past_key_{i}", f"past_value_{i}"])
        passthrough.extend([f"past_key_{i}", f"past_value_{i}"])

    signature = ", ".join(["self", "hidden", "cos", "sin", *cache_args])
    call_args = ", ".join(["hidden", "cos", "sin", *passthrough])
    namespace = {}
    exec(
        "def forward(" + signature + "):\n"
        "    return self._forward_impl(" + call_args + ")\n",
        namespace,
    )
    return namespace["forward"]


def _count_sequential_weights(weights: dict, prefix: str) -> int:
    count = 0
    while f"{prefix}.{count}.weight" in weights:
        count += 1
    return count


class _LinearHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return torch.matmul(hidden, self.weight.T)


class _EmbeddingHead(nn.Module):
    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.register_buffer("weight", weight)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(token_ids, self.weight)


class CPCodecEmbedding(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        n_heads = _count_sequential_weights(weights, "talker.code_predictor.model.codec_embedding")
        self.heads = nn.ModuleList(
            [
                _EmbeddingHead(weights[f"talker.code_predictor.model.codec_embedding.{i}.weight"])
                for i in range(n_heads)
            ]
        )

    def forward(self, token_ids: torch.Tensor, step_idx: int) -> torch.Tensor:
        return self.heads[step_idx](token_ids)


class CPLMHead(nn.Module):
    def __init__(self, weights: dict):
        super().__init__()
        n_heads = _count_sequential_weights(weights, "talker.code_predictor.lm_head")
        self.heads = nn.ModuleList(
            [_LinearHead(weights[f"talker.code_predictor.lm_head.{i}.weight"]) for i in range(n_heads)]
        )

    def forward(self, hidden: torch.Tensor, step_idx: int) -> torch.Tensor:
        return self.heads[step_idx](hidden)


class CodePredictorLayer(nn.Module):
    def __init__(self, weights: dict, layer_prefix: str, config: dict):
        super().__init__()
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.n_heads))
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self.gqa_reps = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5

        self.register_buffer("ln1", weights[f"{layer_prefix}.input_layernorm.weight"])
        self.register_buffer("ln2", weights[f"{layer_prefix}.post_attention_layernorm.weight"])
        self.register_buffer("q_proj", weights[f"{layer_prefix}.self_attn.q_proj.weight"])
        self.register_buffer("k_proj", weights[f"{layer_prefix}.self_attn.k_proj.weight"])
        self.register_buffer("v_proj", weights[f"{layer_prefix}.self_attn.v_proj.weight"])
        self.register_buffer("o_proj", weights[f"{layer_prefix}.self_attn.o_proj.weight"])
        self.register_buffer("q_norm", weights[f"{layer_prefix}.self_attn.q_norm.weight"])
        self.register_buffer("k_norm", weights[f"{layer_prefix}.self_attn.k_norm.weight"])
        self.register_buffer("gate_proj", weights[f"{layer_prefix}.mlp.gate_proj.weight"])
        self.register_buffer("up_proj", weights[f"{layer_prefix}.mlp.up_proj.weight"])
        self.register_buffer("down_proj", weights[f"{layer_prefix}.mlp.down_proj.weight"])

    def forward(
        self,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normed = rms_norm(hidden, self.ln1, self.eps)

        q = F.linear(normed, self.q_proj)
        k = F.linear(normed, self.k_proj)
        v = F.linear(normed, self.v_proj)

        batch, seq, _ = hidden.shape
        q = q.view(batch, seq, self.n_heads, self.head_dim)
        k = k.view(batch, seq, self.n_kv_heads, self.head_dim)

        q = rms_norm(q, self.q_norm, self.eps)
        k = rms_norm(k, self.k_norm, self.eps)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        present_key = torch.cat([past_key, k], dim=2)
        present_value = torch.cat([past_value, v], dim=2)

        k_exp = present_key.repeat_interleave(self.gqa_reps, dim=1)
        v_exp = present_value.repeat_interleave(self.gqa_reps, dim=1)

        attn = torch.matmul(q, k_exp.transpose(-2, -1)) * self.scale
        full_seq = present_key.shape[2]
        mask = torch.triu(
            torch.full((seq, full_seq), float("-inf"), device=attn.device),
            diagonal=full_seq - seq + 1,
        )
        attn = attn + mask.unsqueeze(0).unsqueeze(0)
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
        out = torch.matmul(attn, v_exp)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = F.linear(out, self.o_proj)
        hidden = hidden + out

        normed = rms_norm(hidden, self.ln2, self.eps)
        hidden = hidden + swiglu_ffn(normed, self.gate_proj, self.up_proj, self.down_proj)
        return hidden, present_key, present_value


class CodePredictorBackbone(nn.Module):
    """Code predictor backbone with per-layer modules and explicit cache IO."""

    def __init__(self, weights: dict, config: dict):
        super().__init__()

        self.n_layers = int(config["num_hidden_layers"])
        self.n_heads = int(config["num_attention_heads"])
        self.n_kv_heads = int(config["num_key_value_heads"])
        self.hidden_size = int(config["hidden_size"])
        self.head_dim = int(config.get("head_dim", self.hidden_size // self.n_heads))
        self.eps = float(config.get("rms_norm_eps", 1e-6))
        self._weight_dtype = weights["talker.code_predictor.model.layers.0.input_layernorm.weight"].dtype

        self._has_projection = "talker.code_predictor.small_to_mtp_projection.weight" in weights
        if self._has_projection:
            self.register_buffer("mtp_proj_w", weights["talker.code_predictor.small_to_mtp_projection.weight"])
            self.register_buffer("mtp_proj_b", weights["talker.code_predictor.small_to_mtp_projection.bias"])

        self.codec_embedding = CPCodecEmbedding(weights)
        self.lm_head = CPLMHead(weights)
        self.layers = nn.ModuleList(
            [
                CodePredictorLayer(weights, f"talker.code_predictor.model.layers.{i}", config)
                for i in range(self.n_layers)
            ]
        )
        self.register_buffer("final_norm", weights["talker.code_predictor.model.norm.weight"])

        output_fields = ["hidden"]
        for i in range(self.n_layers):
            output_fields.extend([f"present_key_{i}", f"present_value_{i}"])
        self.Outputs = namedtuple("CodePredictorBackboneOutputs", output_fields)
        self.forward = types.MethodType(_make_explicit_forward(self.n_layers), self)

    def _forward_impl(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, *past_kvs: torch.Tensor):
        hidden = hidden.to(self._weight_dtype)
        if self._has_projection:
            hidden = F.linear(hidden, self.mtp_proj_w, self.mtp_proj_b)

        present_kvs = []
        for i, layer in enumerate(self.layers):
            hidden, present_key, present_value = layer(
                hidden,
                cos,
                sin,
                past_kvs[2 * i],
                past_kvs[2 * i + 1],
            )
            present_kvs.extend([present_key, present_value])

        hidden = rms_norm(hidden, self.final_norm, self.eps)
        return self.Outputs(hidden, *present_kvs)
