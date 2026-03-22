import dataclasses
from typing import Tuple

import torch


def precompute_mrope(
    max_len: int,
    head_dim: int,
    mrope_section,
    theta: float = 1_000_000.0,
    mrope_interleaved: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute M-RoPE cos/sin tables (max_len, head_dim).

    In TTS-only mode all 3 M-RoPE modalities share identical positions, so the
    section merge is an identity op; a simple outer-product + cos/sin suffices.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def precompute_standard_rope(
    max_len: int,
    head_dim: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute standard 1-D RoPE cos/sin tables (max_len, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


@dataclasses.dataclass(frozen=True)
class TTSConstants:
    """All config-derived scalars and precomputed tensors for generation."""

    # Talker config scalars
    codec_eos_id: int
    num_code_groups: int
    vocab_size: int
    talker_n_layers: int
    talker_n_kv_heads: int
    talker_head_dim: int

    # Code predictor config scalars
    cp_n_layers: int
    cp_n_kv_heads: int
    cp_head_dim: int

    # RoPE config
    mrope_section: list
    mrope_interleaved: bool
    talker_rope_theta: float
    talker_max_positions: int
    cp_rope_theta: float
    cp_max_positions: int

    # Suppress mask (precomputed)
    suppress_mask: torch.Tensor

    # RoPE tables — precomputed once, shape (max_positions, head_dim)
    talker_mrope_cos: torch.Tensor
    talker_mrope_sin: torch.Tensor
    cp_rope_cos: torch.Tensor
    cp_rope_sin: torch.Tensor

    @staticmethod
    def from_config_and_weights(config: dict, weights: dict) -> "TTSConstants":
        tc = config["talker_config"]
        cp = tc["code_predictor_config"]
        rs = tc.get("rope_scaling", {})

        codec_eos = tc["codec_eos_token_id"]
        ncg = tc.get("num_code_groups", 16)
        vs = tc["vocab_size"]

        suppress = torch.zeros(vs, dtype=torch.bool)
        suppress[[i for i in range(vs - 1024, vs) if i != codec_eos]] = True

        t_head_dim = tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"])
        cp_head_dim_val = cp.get("head_dim", cp["hidden_size"] // cp["num_attention_heads"])

        t_mrope_cos, t_mrope_sin = precompute_mrope(
            tc.get("max_position_embeddings", 32768),
            t_head_dim,
            rs.get("mrope_section", []),
            tc.get("rope_theta", 1_000_000.0),
            rs.get("interleaved", False),
        )
        cp_rcos, cp_rsin = precompute_standard_rope(
            cp.get("max_position_embeddings", 32768),
            cp_head_dim_val,
            cp.get("rope_theta", 1_000_000.0),
        )

        return TTSConstants(
            codec_eos_id=codec_eos,
            num_code_groups=ncg,
            vocab_size=vs,
            talker_n_layers=tc["num_hidden_layers"],
            talker_n_kv_heads=tc["num_key_value_heads"],
            talker_head_dim=t_head_dim,
            cp_n_layers=cp["num_hidden_layers"],
            cp_n_kv_heads=cp["num_key_value_heads"],
            cp_head_dim=cp_head_dim_val,
            mrope_section=rs.get("mrope_section", []),
            mrope_interleaved=rs.get("interleaved", False),
            talker_rope_theta=tc.get("rope_theta", 1_000_000.0),
            talker_max_positions=tc.get("max_position_embeddings", 32768),
            cp_rope_theta=cp.get("rope_theta", 1_000_000.0),
            cp_max_positions=cp.get("max_position_embeddings", 32768),
            suppress_mask=suppress,
            talker_mrope_cos=t_mrope_cos,
            talker_mrope_sin=t_mrope_sin,
            cp_rope_cos=cp_rcos,
            cp_rope_sin=cp_rsin,
        )
