# ruff: noqa: E402

"""
Convert refactored Qwen3-TTS PyTorch modules to OpenVINO IR.

Produces five core submodels:
  1. talker.xml              — stateful TalkerBackbone (includes codec_head)
  2. code_predictor.xml      — stateful CodePredictorBackbone (includes step-indexed LM heads)
  3. text_model.xml          — TextEmbedding + TextProjection combined
  4. codec_embedding.xml     — talker codec embedding lookup
  5. cp_codec_embedding.xml  — step-indexed code predictor codec embedding

Plus optional models:
  - speech_tokenizer/speech_decoder.xml
  - speaker_encoder.xml  (base model only)
"""

import argparse
import gc
import os
import shutil
import sys
import types
from pathlib import Path

import nncf
import numpy as np
import openvino as ov
import torch
import torch.nn as nn
from openvino._offline_transformations import apply_make_stateful_transformation
from openvino.frontend.pytorch.patch_model import __make_16bit_traceable
from openvino.op.util import Variable, VariableInfo


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import qwen3_tts
from modules import (
    CPCodecEmbedding,
    CodePredictorBackbone,
    IntegerInputSpeechDecoder,
    SpeakerEncoder,
    TalkerBackbone,
    TextEmbedding,
    TextProjection,
)


# ---------------------------------------------------------------------------
# Forward-signature code generation
# ---------------------------------------------------------------------------


def _make_talker_export_forward(n_layers: int):
    """Generate forward(self, inputs_embeds, cos, sin, past_key_0, ...) for talker."""
    args = ["self", "inputs_embeds", "cos", "sin"]
    passthrough = ["inputs_embeds", "cos", "sin"]
    for i in range(n_layers):
        args.extend([f"past_key_{i}", f"past_value_{i}"])
        passthrough.extend([f"past_key_{i}", f"past_value_{i}"])
    sig = ", ".join(args)
    call = ", ".join(passthrough)
    ns = {}
    exec(f"def forward({sig}):\n    return self._forward_impl({call})\n", ns)
    return ns["forward"]


def _make_cp_export_forward(n_layers: int):
    """Generate forward(self, inputs_embeds, cos, sin, generation_steps, past_key_0, ...) for code predictor."""
    args = ["self", "inputs_embeds", "cos", "sin", "generation_steps"]
    passthrough = ["inputs_embeds", "cos", "sin", "generation_steps"]
    for i in range(n_layers):
        args.extend([f"past_key_{i}", f"past_value_{i}"])
        passthrough.extend([f"past_key_{i}", f"past_value_{i}"])
    sig = ", ".join(args)
    call = ", ".join(passthrough)
    ns = {}
    exec(f"def forward({sig}):\n    return self._forward_impl({call})\n", ns)
    return ns["forward"]


# ---------------------------------------------------------------------------
# Export wrappers — thin modules that reshape backbone I/O for OV tracing
# ---------------------------------------------------------------------------


class TalkerForExport(nn.Module):
    """
    Wraps TalkerBackbone for OV export.

    Input:  inputs_embeds, cos, sin, past_key_0, past_value_0, ...
    Output: (logits, hidden, present_key_0, present_value_0, ...)
    """

    def __init__(self, backbone: TalkerBackbone):
        super().__init__()
        self.backbone = backbone
        self.n_layers = backbone.n_layers
        self.forward = types.MethodType(_make_talker_export_forward(self.n_layers), self)

    def _forward_impl(self, inputs_embeds, cos, sin, *past_kvs):
        return tuple(self.backbone._forward_impl(inputs_embeds, cos, sin, *past_kvs))


class CodePredictorForExport(nn.Module):
    """
    Wraps CodePredictorBackbone for OV export.

    Folds the step-indexed LM heads into the forward pass so the exported IR
    accepts generation_steps as an input and returns logits directly.

    Input:  inputs_embeds, cos, sin, generation_steps, past_key_0, past_value_0, ...
    Output: (logits, hidden, present_key_0, present_value_0, ...)
    """

    def __init__(self, backbone: CodePredictorBackbone):
        super().__init__()
        self.backbone = backbone
        self.lm_heads = backbone.lm_head.heads
        self.n_layers = backbone.n_layers
        self.forward = types.MethodType(_make_cp_export_forward(self.n_layers), self)

    def _forward_impl(self, inputs_embeds, cos, sin, generation_steps, *past_kvs):
        outputs = tuple(self.backbone._forward_impl(inputs_embeds, cos, sin, *past_kvs))
        hidden = outputs[0]
        # present caches are outputs[1:]

        # Step-indexed LM head: compute all heads, select by generation_steps tensor
        all_logits = torch.stack([head(hidden) for head in self.lm_heads], dim=0)
        logits = torch.index_select(all_logits, 0, generation_steps.reshape(1)).squeeze(0)

        return (logits, hidden) + outputs[1:]


class SpeechEncoderForOV(nn.Module):
    """Full audio -> discrete-codes encoder for OV export (ICL reference encoding).

    Pipeline mirrors v2_encode() in qwen3_tts_tokenizer.py:
        audio (1, samples) ->
        conv encoder (1, 512, T') ->
        transformer (bidirectional, 1, 512, T') ->
        downsample (1, 512, T) ->
        split RVQ quantize (1, T, n_q_total)

    All weights are stored as registered buffers (float32) so that
    __make_16bit_traceable operates correctly. The encoder transformer RoPE
    is precomputed at max length and sliced dynamically in forward().
    """

    def __init__(self, speech_tokenizer_state: dict):
        super().__init__()
        config = speech_tokenizer_state["config"]
        enc_cfg = config.get("encoder_config", {})

        self._n_q_total = int(config.get("encoder_valid_num_quantizers", 16))
        self._n_q_semantic = int(enc_cfg.get("num_semantic_quantizers", 1))
        self._hidden_size = int(enc_cfg.get("hidden_size", 512))
        self._compress = int(enc_cfg.get("compress", 2))
        self._n_attn_heads = int(enc_cfg.get("num_attention_heads", 8))
        self._n_kv_heads = int(enc_cfg.get("num_key_value_heads", self._n_attn_heads))
        self._head_dim = int(enc_cfg.get("head_dim", self._hidden_size // self._n_attn_heads))
        self._n_layers = int(enc_cfg.get("num_hidden_layers", 6))
        self._norm_eps = float(enc_cfg.get("norm_eps", 1e-5))

        # Cast all weights to float32 upfront
        w = {k: v.float() for k, v in speech_tokenizer_state["weights"].items()}
        self._w = w

        # --- Conv encoder buffers: layers.0, 4 downsampling stages, layers.14 ---
        self.register_buffer("conv_enc_l0_w", w["encoder.encoder.layers.0.conv.weight"])
        self.register_buffer("conv_enc_l0_b", w.get("encoder.encoder.layers.0.conv.bias", torch.zeros(0)))

        strides = [4, 5, 6, 8]
        resblock_indices = [1, 4, 7, 10]
        conv_indices = [3, 6, 9, 12]
        for rb_idx, conv_idx in zip(resblock_indices, conv_indices):
            p = f"encoder.encoder.layers.{rb_idx}"
            self.register_buffer(f"rb_{rb_idx}_b1_w", w[f"{p}.block.1.conv.weight"])
            self.register_buffer(f"rb_{rb_idx}_b1_b", w.get(f"{p}.block.1.conv.bias", torch.zeros(0)))
            self.register_buffer(f"rb_{rb_idx}_b3_w", w[f"{p}.block.3.conv.weight"])
            self.register_buffer(f"rb_{rb_idx}_b3_b", w.get(f"{p}.block.3.conv.bias", torch.zeros(0)))
            p2 = f"encoder.encoder.layers.{conv_idx}"
            self.register_buffer(f"ds_{conv_idx}_w", w[f"{p2}.conv.weight"])
            self.register_buffer(f"ds_{conv_idx}_b", w.get(f"{p2}.conv.bias", torch.zeros(0)))

        self.register_buffer("conv_enc_l14_w", w["encoder.encoder.layers.14.conv.weight"])
        self.register_buffer("conv_enc_l14_b", w.get("encoder.encoder.layers.14.conv.bias", torch.zeros(0)))

        # --- Encoder transformer buffers ---
        for i in range(self._n_layers):
            p = f"encoder.encoder_transformer.layers.{i}"
            self.register_buffer(f"tr_{i}_ln1_w", w[f"{p}.input_layernorm.weight"])
            self.register_buffer(f"tr_{i}_ln1_b", w[f"{p}.input_layernorm.bias"])
            self.register_buffer(f"tr_{i}_q_w", w[f"{p}.self_attn.q_proj.weight"])
            self.register_buffer(f"tr_{i}_k_w", w[f"{p}.self_attn.k_proj.weight"])
            self.register_buffer(f"tr_{i}_v_w", w[f"{p}.self_attn.v_proj.weight"])
            self.register_buffer(f"tr_{i}_o_w", w[f"{p}.self_attn.o_proj.weight"])
            ls1_key = f"{p}.self_attn_layer_scale.scale"
            self.register_buffer(f"tr_{i}_ls1", w[ls1_key] if ls1_key in w else torch.zeros(0))
            self.register_buffer(f"tr_{i}_ln2_w", w[f"{p}.post_attention_layernorm.weight"])
            self.register_buffer(f"tr_{i}_ln2_b", w[f"{p}.post_attention_layernorm.bias"])
            self.register_buffer(f"tr_{i}_fc1_w", w[f"{p}.mlp.fc1.weight"])
            self.register_buffer(f"tr_{i}_fc2_w", w[f"{p}.mlp.fc2.weight"])
            ls2_key = f"{p}.mlp_layer_scale.scale"
            self.register_buffer(f"tr_{i}_ls2", w[ls2_key] if ls2_key in w else torch.zeros(0))

        # --- Downsample buffer ---
        self.register_buffer("downsample_w", w["encoder.downsample.conv.weight"])
        self.register_buffer("downsample_b", w.get("encoder.downsample.conv.bias", torch.zeros(0)))

        # --- Quantizer buffers ---
        n_q_aco = self._n_q_total - self._n_q_semantic
        sem = "encoder.quantizer.semantic_residual_vector_quantizer"
        aco = "encoder.quantizer.acoustic_residual_vector_quantizer"
        self.register_buffer("sem_inp_w", w[f"{sem}.input_proj.weight"])
        self.register_buffer("aco_inp_w", w[f"{aco}.input_proj.weight"])
        for i in range(self._n_q_semantic):
            self.register_buffer(f"sem_cu_{i}", w[f"{sem}.layers.{i}.codebook.cluster_usage"])
            self.register_buffer(f"sem_es_{i}", w[f"{sem}.layers.{i}.codebook.embed_sum"])
        for i in range(n_q_aco):
            self.register_buffer(f"aco_cu_{i}", w[f"{aco}.layers.{i}.codebook.cluster_usage"])
            self.register_buffer(f"aco_es_{i}", w[f"{aco}.layers.{i}.codebook.embed_sum"])

        # Precomputed RoPE for transformer: max 4096 frames (~86 s at 12 Hz)
        head_dim = self._head_dim
        theta = float(enc_cfg.get("rope_theta", 10000.0))
        max_frames = 4096
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_frames, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("rope_cos", emb.cos())
        self.register_buffer("rope_sin", emb.sin())

        self._strides = strides
        self._resblock_indices = resblock_indices
        self._conv_indices = conv_indices

    # --- Internal helpers ---

    @staticmethod
    def _causal_conv1d(x, w, b, stride=1, dilation=1):
        kernel_size = w.shape[2]
        pad = (kernel_size - 1) * dilation
        x = torch.nn.functional.pad(x, (pad, 0))
        groups = x.shape[1] // w.shape[1]
        b_arg = b if b.numel() > 0 else None
        return torch.nn.functional.conv1d(
            x, w.to(x.dtype), b_arg.to(x.dtype) if b_arg is not None else None,
            stride=stride, dilation=dilation, groups=groups,
        )

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _attn_layer(self, hidden, layer_idx, cos, sin):
        """Single encoder transformer attention (bidirectional, no causal mask)."""
        n_heads = self._n_attn_heads
        n_kv = self._n_kv_heads
        hd = self._head_dim
        q_w = getattr(self, f"tr_{layer_idx}_q_w")
        k_w = getattr(self, f"tr_{layer_idx}_k_w")
        v_w = getattr(self, f"tr_{layer_idx}_v_w")
        o_w = getattr(self, f"tr_{layer_idx}_o_w")

        q = torch.nn.functional.linear(hidden, q_w)
        k = torch.nn.functional.linear(hidden, k_w)
        v = torch.nn.functional.linear(hidden, v_w)

        batch, seq, _ = hidden.shape
        q = q.view(batch, seq, n_heads, hd).transpose(1, 2)
        k = k.view(batch, seq, n_kv, hd).transpose(1, 2)
        v = v.view(batch, seq, n_kv, hd).transpose(1, 2)

        cos_b = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, hd)
        sin_b = sin.unsqueeze(0).unsqueeze(0)
        q = q * cos_b + self._rotate_half(q) * sin_b
        k = k * cos_b + self._rotate_half(k) * sin_b

        if n_kv < n_heads:
            reps = n_heads // n_kv
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)

        scale = hd ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.nn.functional.softmax(attn, dim=-1, dtype=torch.float32).to(hidden.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        return torch.nn.functional.linear(out, o_w)

    def _transformer_layer(self, hidden, layer_idx, cos, sin):
        ln1_w = getattr(self, f"tr_{layer_idx}_ln1_w")
        ln1_b = getattr(self, f"tr_{layer_idx}_ln1_b")
        ln2_w = getattr(self, f"tr_{layer_idx}_ln2_w")
        ln2_b = getattr(self, f"tr_{layer_idx}_ln2_b")
        fc1_w = getattr(self, f"tr_{layer_idx}_fc1_w")
        fc2_w = getattr(self, f"tr_{layer_idx}_fc2_w")
        ls1 = getattr(self, f"tr_{layer_idx}_ls1")
        ls2 = getattr(self, f"tr_{layer_idx}_ls2")

        eps = self._norm_eps
        H = hidden.shape[-1]

        normed = torch.nn.functional.layer_norm(hidden, (H,), ln1_w, ln1_b, eps)
        attn_out = self._attn_layer(normed, layer_idx, cos, sin)
        if ls1.numel() > 0:
            attn_out = attn_out * ls1
        hidden = hidden + attn_out

        normed = torch.nn.functional.layer_norm(hidden, (H,), ln2_w, ln2_b, eps)
        ffn = torch.nn.functional.gelu(torch.nn.functional.linear(normed, fc1_w))
        ffn = torch.nn.functional.linear(ffn, fc2_w)
        if ls2.numel() > 0:
            ffn = ffn * ls2
        return hidden + ffn

    def _vq_encode_one(self, x_t, cluster_usage, embed_sum):
        """x_t: (T, dim) -> (T,) int64 codes, (T, dim) quantized."""
        eps = 1e-5
        embedding = embed_sum / cluster_usage.clamp(min=eps).unsqueeze(-1)
        dists = torch.cdist(x_t.unsqueeze(0), embedding.unsqueeze(0), p=2).squeeze(0)
        codes = dists.argmin(dim=-1)
        quantized = torch.nn.functional.embedding(codes, embedding)
        return codes, quantized

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio: (1, samples) float32 -> codes: (1, T, n_q_total) int64"""
        x = audio.float().unsqueeze(1)  # (1, 1, samples)

        # === Conv encoder ===
        b0_w = self.conv_enc_l0_w
        b0_b = self.conv_enc_l0_b
        x = self._causal_conv1d(x, b0_w, b0_b)

        for rb_idx, conv_idx, stride in zip(
            self._resblock_indices, self._conv_indices, self._strides
        ):
            rb_b1_w = getattr(self, f"rb_{rb_idx}_b1_w")
            rb_b1_b = getattr(self, f"rb_{rb_idx}_b1_b")
            rb_b3_w = getattr(self, f"rb_{rb_idx}_b3_w")
            rb_b3_b = getattr(self, f"rb_{rb_idx}_b3_b")
            ds_w = getattr(self, f"ds_{conv_idx}_w")
            ds_b = getattr(self, f"ds_{conv_idx}_b")

            # Residual block
            residual = x
            h = torch.nn.functional.elu(x)
            h = self._causal_conv1d(h, rb_b1_w, rb_b1_b)
            h = torch.nn.functional.elu(h)
            h = self._causal_conv1d(h, rb_b3_w, rb_b3_b)
            x = h + residual

            x = torch.nn.functional.elu(x)
            x = self._causal_conv1d(x, ds_w, ds_b, stride=stride)

        x = torch.nn.functional.elu(x)
        x = self._causal_conv1d(x, self.conv_enc_l14_w, self.conv_enc_l14_b)

        # === Encoder transformer ===
        x = x.transpose(1, 2)  # (1, T', hidden)
        seq_len = x.shape[1]
        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]
        for i in range(self._n_layers):
            x = self._transformer_layer(x, i, cos, sin)
        x = x.transpose(1, 2)  # (1, hidden, T')

        # === Downsample ===
        x = self._causal_conv1d(x, self.downsample_w, self.downsample_b, stride=self._compress)

        # === Split RVQ encode ===
        h = x  # (1, hidden, T)
        n_q_semantic = self._n_q_semantic
        n_q_acoustic = self._n_q_total - n_q_semantic

        # Semantic
        sem_h = torch.nn.functional.conv1d(h, self.sem_inp_w.to(h.dtype))  # (1, cdim, T)
        residual = sem_h
        sem_codes = []
        for i in range(n_q_semantic):
            cu = getattr(self, f"sem_cu_{i}")
            es = getattr(self, f"sem_es_{i}")
            r_t = residual[0].permute(1, 0)  # (T, cdim)
            codes_i, quantized_i = self._vq_encode_one(r_t, cu, es)
            sem_codes.append(codes_i)
            residual = residual - quantized_i.permute(1, 0).unsqueeze(0)

        # Acoustic
        aco_h = torch.nn.functional.conv1d(h, self.aco_inp_w.to(h.dtype))
        residual = aco_h
        aco_codes = []
        for i in range(n_q_acoustic):
            cu = getattr(self, f"aco_cu_{i}")
            es = getattr(self, f"aco_es_{i}")
            r_t = residual[0].permute(1, 0)
            codes_i, quantized_i = self._vq_encode_one(r_t, cu, es)
            aco_codes.append(codes_i)
            residual = residual - quantized_i.permute(1, 0).unsqueeze(0)

        all_codes = torch.stack(sem_codes + aco_codes, dim=-1)  # (T, n_q_total)
        return all_codes.unsqueeze(0)  # (1, T, n_q_total)


class TextModelForExport(nn.Module):
    """
    Combines TextEmbedding + TextProjection into one traced graph.

    Input:  token_ids
    Output: projected embeddings in talker hidden space
    """

    def __init__(self, text_embedding: TextEmbedding, text_projection: TextProjection):
        super().__init__()
        self.text_embedding = text_embedding
        self.text_projection = text_projection

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.text_projection(self.text_embedding(token_ids))


class CPCodecEmbeddingForExport(nn.Module):
    """
    Wraps CPCodecEmbedding for OV export with tensor-based step selection.

    Input:  token_ids, step_idx (scalar int64 tensor)
    Output: embeddings for the selected codebook
    """

    def __init__(self, module: CPCodecEmbedding):
        super().__init__()
        self.heads = module.heads

    def forward(self, token_ids: torch.Tensor, step_idx: torch.Tensor) -> torch.Tensor:
        all_embeds = torch.stack([head(token_ids) for head in self.heads], dim=0)
        selected = torch.index_select(all_embeds, 0, step_idx.reshape(1))
        return selected.squeeze(0)


# ---------------------------------------------------------------------------
# NNCF compression configuration
# ---------------------------------------------------------------------------


def get_compression_config(weight_format: str):
    if weight_format == "int4":
        return {
            "transformer": {
                "mode": nncf.CompressWeightsMode.INT4_SYM,
                "group_size": 128,
                "ratio": 1.0,
            },
            "conv": {
                "mode": nncf.CompressWeightsMode.INT8_SYM,
            },
        }
    if weight_format == "int8":
        return {
            "transformer": {
                "mode": nncf.CompressWeightsMode.INT8_SYM,
            },
            "conv": {
                "mode": nncf.CompressWeightsMode.INT8_SYM,
            },
        }
    return None


# ---------------------------------------------------------------------------
# Stateful KV-cache transformation helpers
# ---------------------------------------------------------------------------


def set_input_names(ov_model: ov.Model, names: list[str]):
    for i, name in enumerate(names):
        ov_model.input(i).get_tensor().set_names({name})


def set_output_names(ov_model: ov.Model, names: list[str]):
    for i, name in enumerate(names):
        ov_model.output(i).get_tensor().set_names({name})


def fuse_cache_reorder(
    ov_model: ov.Model,
    reference_input_name: str,
    key_value_input_names: list[str],
):
    """Insert beam_idx Gather on each KV cache input for beam search support."""
    batch_dim = ov_model.input(reference_input_name).get_partial_shape()[0]
    beam_idx = ov.opset13.parameter([batch_dim], dtype=ov.Type.i32, name="beam_idx")
    beam_idx.output(0).get_tensor().set_names({"beam_idx"})
    ov_model.add_parameters([beam_idx])

    axis = np.array(0, dtype=np.int64)
    for input_name in key_value_input_names:
        port = ov_model.input(input_name)
        consumers = list(port.get_target_inputs())
        gather = ov.opset13.gather(port, beam_idx.output(0), axis)
        for consumer in consumers:
            consumer.replace_source_output(gather.output(0))

    ov_model.validate_nodes_and_infer_types()


def build_cache_state_initializer(
    ov_model: ov.Model,
    reference_input_name: str,
    tail_dims_by_input_name: dict[str, list[int]],
):
    """Create ShapeOf-based dynamic initializer for each ReadValue state variable."""
    shape_node = ov.opset13.shape_of(ov_model.input(reference_input_name), output_type="i64")
    batch = ov.opset13.gather(shape_node, np.array([0], dtype=np.int64), np.array(0, dtype=np.int64))

    read_values = {}
    assigns = {}
    for op in ov_model.get_ops():
        if op.get_type_name() == "ReadValue":
            read_values[op.get_variable_id()] = op
        elif op.get_type_name() == "Assign":
            assigns[op.get_variable_id()] = op

    for input_name, tail_dims in tail_dims_by_input_name.items():
        matched_id = None
        for var_id in read_values:
            if input_name in var_id:
                matched_id = var_id
                break
        if matched_id is None:
            raise RuntimeError(f"No ReadValue found for cache input '{input_name}'")
        if matched_id not in assigns:
            raise RuntimeError(f"No Assign found for cache variable '{matched_id}'")

        old_read = read_values[matched_id]
        old_assign = assigns[matched_id]
        elem_type = old_read.output(0).get_element_type()
        partial_shape = old_read.output(0).get_partial_shape()

        dims = [batch] + [np.array([d], dtype=np.int64) for d in tail_dims]
        target_shape = ov.opset13.concat(dims, axis=0)
        zero = ov.opset13.convert(ov.opset13.constant(np.array([0.0], dtype=np.float32)), elem_type)
        init = ov.opset13.broadcast(zero, target_shape)

        info = VariableInfo()
        info.variable_id = matched_id
        info.data_type = elem_type
        info.data_shape = partial_shape
        variable = Variable(info)

        new_read = ov.opset6.read_value(init, variable, name=old_read.get_friendly_name())
        new_assign = ov.opset6.assign(
            old_assign.input_value(0).get_node(), variable, old_assign.get_friendly_name()
        )

        for consumer in list(old_read.output(0).get_target_inputs()):
            consumer.replace_source_output(new_read.output(0))

        ov_model.remove_sink(old_assign)
        ov_model.add_sinks([new_assign])

    ov_model.validate_nodes_and_infer_types()


# ---------------------------------------------------------------------------
# Shared conversion helpers
# ---------------------------------------------------------------------------


def sync_cached_weight_dtypes(module: nn.Module):
    """After __make_16bit_traceable mutates buffers, re-sync _weight_dtype fields."""
    for child in module.modules():
        if not hasattr(child, "_weight_dtype"):
            continue
        for _, buf in child.named_buffers(recurse=False):
            if isinstance(buf, torch.Tensor):
                child._weight_dtype = buf.dtype
                break


def convert_and_save(
    module: nn.Module,
    output_xml: Path,
    example_input,
    input_spec,
    input_names: list[str],
    output_names: list[str],
    compression_args=None,
    stateful_config=None,
):
    """
    Trace, name, optionally make stateful, optionally compress, save.

    stateful_config: dict with keys
        - state_pairs: {past_key_0: present_key_0, ...}
        - cache_tail_dims: {past_key_0: [n_kv_heads, 0, head_dim], ...}
        - reference_input: name of the input to derive batch dim from
    """
    if output_xml.exists():
        print(f"  [skip] {output_xml.name}")
        return

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    module.eval()
    __make_16bit_traceable(module)
    sync_cached_weight_dtypes(module)

    ov_model = ov.convert_model(module, input=input_spec, example_input=example_input, share_weights=True)
    set_input_names(ov_model, input_names)
    set_output_names(ov_model, output_names)

    if stateful_config is not None:
        pairs = stateful_config["state_pairs"]
        ref = stateful_config["reference_input"]
        tails = stateful_config["cache_tail_dims"]
        fuse_cache_reorder(ov_model, ref, list(pairs.keys()))
        apply_make_stateful_transformation(ov_model, pairs)
        build_cache_state_initializer(ov_model, ref, tails)

    if compression_args is not None:
        ov_model = nncf.compress_weights(ov_model, **compression_args)

    ov.save_model(ov_model, output_xml, compress_to_fp16=False)
    print(f"  [saved] {output_xml.name}")
    del ov_model
    gc.collect()


# ---------------------------------------------------------------------------
# Submodel 1: Talker (stateful)
# ---------------------------------------------------------------------------


def convert_talker(talker: TalkerBackbone, config: dict, compression_config, output_dir: Path):
    print("[1/5] Talker backbone (stateful)")
    wrapper = TalkerForExport(talker)

    n_layers = int(config["num_hidden_layers"])
    n_kv_heads = int(config["num_key_value_heads"])
    hidden_size = int(config["hidden_size"])
    head_dim = int(config.get("head_dim", hidden_size // int(config["num_attention_heads"])))

    example_input = {
        "inputs_embeds": torch.randn(1, 2, hidden_size),
        "cos": torch.randn(1, 1, 2, head_dim),
        "sin": torch.randn(1, 1, 2, head_dim),
    }
    input_spec = {
        "inputs_embeds": ov.PartialShape([-1, -1, hidden_size]),
        "cos": ov.PartialShape([1, 1, -1, head_dim]),
        "sin": ov.PartialShape([1, 1, -1, head_dim]),
    }
    input_names = ["inputs_embeds", "cos", "sin"]
    output_names = ["logits", "hidden"]
    state_pairs = {}
    cache_tail_dims = {}

    for i in range(n_layers):
        pk, pv = f"past_key_{i}", f"past_value_{i}"
        rk, rv = f"present_key_{i}", f"present_value_{i}"
        example_input[pk] = torch.randn(1, n_kv_heads, 2, head_dim)
        example_input[pv] = torch.randn(1, n_kv_heads, 2, head_dim)
        input_spec[pk] = ov.PartialShape([-1, n_kv_heads, -1, head_dim])
        input_spec[pv] = ov.PartialShape([-1, n_kv_heads, -1, head_dim])
        state_pairs[pk] = rk
        state_pairs[pv] = rv
        cache_tail_dims[pk] = [n_kv_heads, 0, head_dim]
        cache_tail_dims[pv] = [n_kv_heads, 0, head_dim]
        input_names.extend([pk, pv])
        output_names.extend([rk, rv])

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "talker.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=input_names,
        output_names=output_names,
        compression_args=compression_config["transformer"] if compression_config else None,
        stateful_config={
            "state_pairs": state_pairs,
            "cache_tail_dims": cache_tail_dims,
            "reference_input": "inputs_embeds",
        },
    )


# ---------------------------------------------------------------------------
# Submodel 2: Code predictor (stateful KV cache, with step-indexed LM heads)
# ---------------------------------------------------------------------------


def convert_code_predictor(
    cp: CodePredictorBackbone, talker_config: dict, cp_config: dict, compression_config, output_dir: Path
):
    print("[2/5] Code predictor backbone (stateful + LM heads)")
    wrapper = CodePredictorForExport(cp)

    n_layers = int(cp_config["num_hidden_layers"])
    n_kv_heads = int(cp_config["num_key_value_heads"])
    cp_hidden = int(cp_config["hidden_size"])
    head_dim = int(cp_config.get("head_dim", cp_hidden // int(cp_config["num_attention_heads"])))
    input_dim = int(talker_config["hidden_size"]) if cp._has_projection else cp_hidden
    state_pairs = {}
    cache_tail_dims = {}

    example_input = {
        "inputs_embeds": torch.randn(1, 2, input_dim),
        "cos": torch.randn(1, 1, 2, head_dim),
        "sin": torch.randn(1, 1, 2, head_dim),
        "generation_steps": torch.tensor(0, dtype=torch.int64),
    }
    input_spec = {
        "inputs_embeds": ov.PartialShape([-1, -1, input_dim]),
        "cos": ov.PartialShape([1, 1, -1, head_dim]),
        "sin": ov.PartialShape([1, 1, -1, head_dim]),
        "generation_steps": (ov.PartialShape([]), ov.Type.i64),
    }
    input_names = ["inputs_embeds", "cos", "sin", "generation_steps"]
    output_names = ["logits", "hidden"]

    for i in range(n_layers):
        pk, pv = f"past_key_{i}", f"past_value_{i}"
        rk, rv = f"present_key_{i}", f"present_value_{i}"
        example_input[pk] = torch.randn(1, n_kv_heads, 2, head_dim)
        example_input[pv] = torch.randn(1, n_kv_heads, 2, head_dim)
        input_spec[pk] = ov.PartialShape([-1, n_kv_heads, -1, head_dim])
        input_spec[pv] = ov.PartialShape([-1, n_kv_heads, -1, head_dim])
        state_pairs[pk] = rk
        state_pairs[pv] = rv
        cache_tail_dims[pk] = [n_kv_heads, 0, head_dim]
        cache_tail_dims[pv] = [n_kv_heads, 0, head_dim]
        input_names.extend([pk, pv])
        output_names.extend([rk, rv])

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "code_predictor.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=input_names,
        output_names=output_names,
        compression_args=compression_config["transformer"] if compression_config else None,
        stateful_config={
            "state_pairs": state_pairs,
            "cache_tail_dims": cache_tail_dims,
            "reference_input": "inputs_embeds",
        },
    )


# ---------------------------------------------------------------------------
# Submodel 3: Text model (TextEmbedding + TextProjection combined)
# ---------------------------------------------------------------------------


def convert_text_model(talker: TalkerBackbone, output_dir: Path):
    print("[3/5] Text model (embedding + projection)")
    wrapper = TextModelForExport(talker.text_embedding, talker.text_projection)

    example_input = {"token_ids": torch.ones(1, 2, dtype=torch.int64)}
    input_spec = {"token_ids": (ov.PartialShape([-1, -1]), ov.Type.i64)}

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "text_model.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["token_ids"],
        output_names=["projected"],
    )


# ---------------------------------------------------------------------------
# Submodel 4: Codec embedding (talker)
# ---------------------------------------------------------------------------


def convert_codec_embedding(talker: TalkerBackbone, output_dir: Path):
    print("[4/5] Codec embedding")

    example_input = {"token_ids": torch.ones(1, 2, dtype=torch.int64)}
    input_spec = {"token_ids": (ov.PartialShape([-1, -1]), ov.Type.i64)}

    convert_and_save(
        module=talker.codec_embedding,
        output_xml=output_dir / "codec_embedding.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["token_ids"],
        output_names=["embeddings"],
    )


# ---------------------------------------------------------------------------
# Submodel 5: CP codec embedding (step-indexed)
# ---------------------------------------------------------------------------


def convert_cp_codec_embedding(cp: CodePredictorBackbone, output_dir: Path):
    print("[5/5] Code predictor codec embedding (step-indexed)")
    wrapper = CPCodecEmbeddingForExport(cp.codec_embedding)

    example_input = {
        "token_ids": torch.ones(1, 1, dtype=torch.int64),
        "step_idx": torch.tensor(0, dtype=torch.int64),
    }
    input_spec = {
        "token_ids": (ov.PartialShape([-1, -1]), ov.Type.i64),
        "step_idx": (ov.PartialShape([]), ov.Type.i64),
    }

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "cp_codec_embedding.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["token_ids", "step_idx"],
        output_names=["embeddings"],
    )


# ---------------------------------------------------------------------------
# Additional models: Speech decoder, Speaker encoder
# ---------------------------------------------------------------------------


def convert_speech_decoder(speech_tokenizer_state: dict, compression_config, output_dir: Path):
    print("[+] Speech decoder")
    dec_cfg = speech_tokenizer_state["config"].get("decoder_config", speech_tokenizer_state["config"])
    num_quantizers = int(dec_cfg.get("num_quantizers", 16))
    wrapper = IntegerInputSpeechDecoder(speech_tokenizer_state["weights"], dec_cfg)

    example_input = {"codes": torch.zeros(1, num_quantizers, 100, dtype=torch.int64)}
    input_spec = {"codes": (ov.PartialShape([1, num_quantizers, -1]), ov.Type.i64)}

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "speech_tokenizer" / "speech_decoder.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["codes"],
        output_names=["waveform"],
        compression_args=compression_config["conv"] if compression_config else None,
    )


def convert_speech_encoder_ov(speech_tokenizer_state: dict, compression_config, output_dir: Path):
    """Export the full audio->codes speech encoder for ICL reference encoding."""
    print("[+] Speech encoder (audio -> discrete codes, for ICL)")
    wrapper = SpeechEncoderForOV(speech_tokenizer_state)

    # 3 s at 24 kHz as example input; actual inference uses variable-length audio
    example_input = {"audio": torch.randn(1, 72000)}
    input_spec = {"audio": ov.PartialShape([1, -1])}

    convert_and_save(
        module=wrapper,
        output_xml=output_dir / "speech_tokenizer" / "speech_encoder.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["audio"],
        output_names=["codes"],
        # No weight compression: quantization tables must remain exact integers
    )


def convert_speaker_encoder(module: SpeakerEncoder, config: dict, compression_config, output_dir: Path):
    print("[+] Speaker encoder")
    mel_dim = int(config.get("mel_dim", 128))

    example_input = {"mels": torch.randn(1, 100, mel_dim)}
    input_spec = {"mels": ov.PartialShape([1, -1, mel_dim])}

    convert_and_save(
        module=module,
        output_xml=output_dir / "speaker_encoder.xml",
        example_input=example_input,
        input_spec=input_spec,
        input_names=["mels"],
        output_names=["embedding"],
        compression_args=compression_config["conv"] if compression_config else None,
    )


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def copy_metadata_files(model_path: Path, output_dir: Path):
    """Copy config/tokenizer JSON files (not safetensors) to output dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for item in model_path.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, output_dir / item.name)

    src_tok = model_path / "speech_tokenizer"
    dst_tok = output_dir / "speech_tokenizer"
    if src_tok.exists():
        dst_tok.mkdir(parents=True, exist_ok=True)
        for item in src_tok.iterdir():
            if item.is_file() and item.suffix != ".safetensors":
                shutil.copy2(item, dst_tok / item.name)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def convert_pipeline(
    model_path: Path,
    weight_format: str,
    output_dir: Path,
    model_type_override: str = None,
):
    """Convert all submodels for a Qwen3-TTS model variant.

    Args:
        model_path:          Path to the HuggingFace model directory.
        weight_format:       Weight compression: 'int4', 'int8', or 'fp16'.
        output_dir:          Destination directory for IR files.
        model_type_override: If provided, overrides the tts_model_type field in
                             config.json.  Accepted values: 'voice_clone',
                             'voice_design', 'custom_voice'.
    """
    state = qwen3_tts.load_model(str(model_path))
    config = state["config"]
    talker_config = config["talker_config"]
    cp_config = talker_config["code_predictor_config"]
    speech_tokenizer_state = state["speech_tokenizer"]

    # Resolve model type — CLI arg wins over config
    model_type = model_type_override or config.get("tts_model_type", "custom_voice")
    compression_config = get_compression_config(weight_format)

    print(f"Model type: {model_type}")
    print(f"Weight format: {weight_format}")
    print()

    # Instantiate backbone modules
    talker = TalkerBackbone(state["weights"], talker_config)
    code_predictor = CodePredictorBackbone(state["weights"], cp_config)

    # Core five submodels (shared by all task types)
    convert_talker(talker, talker_config, compression_config, output_dir)
    convert_code_predictor(code_predictor, talker_config, cp_config, compression_config, output_dir)
    convert_text_model(talker, output_dir)
    convert_codec_embedding(talker, output_dir)
    convert_cp_codec_embedding(code_predictor, output_dir)

    # Speech decoder (shared by all task types)
    convert_speech_decoder(speech_tokenizer_state, compression_config, output_dir)

    # Voice-clone-specific: speaker encoder + speech encoder for ICL
    if model_type in ("base", "voice_clone"):
        se_weights = state.get("speaker_encoder_weights")
        if not se_weights:
            raise ValueError(
                f"Model type '{model_type}' requires speaker encoder weights, "
                "but none were found in the model directory."
            )
        se_config = config.get("speaker_encoder_config", {})
        speaker_encoder = SpeakerEncoder(se_weights, se_config)
        convert_speaker_encoder(speaker_encoder, se_config, compression_config, output_dir)

    if model_type == "voice_clone":
        convert_speech_encoder_ov(speech_tokenizer_state, compression_config, output_dir)

    # Copy HF config / tokenizer JSON files so the OV dir is self-contained
    copy_metadata_files(model_path, output_dir)
    print()
    print(f"Done. Output: {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3-TTS modules to OpenVINO IR.")
    parser.add_argument("--model-path", required=True, help="Path to the Qwen3-TTS model directory.")
    parser.add_argument(
        "--weight-format",
        choices=["int4", "int8", "fp16"],
        default="fp16",
        help="Weight compression: int4, int8, or fp16 (no compression).",
    )
    parser.add_argument(
        "--new",
        dest="output_dir",
        required=True,
        help="Output directory for IR files.",
    )
    parser.add_argument(
        "--model-type",
        choices=["voice_clone", "voice_design", "custom_voice"],
        default=None,
        help=(
            "Override the tts_model_type from config.json.  "
            "'voice_clone' also exports speaker_encoder.xml and "
            "speech_tokenizer/speech_encoder.xml for ICL support."
        ),
    )
    args = parser.parse_args()

    model_path = Path(args.model_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    os.makedirs(output_dir, exist_ok=True)
    convert_pipeline(model_path, args.weight_format, output_dir, args.model_type)


if __name__ == "__main__":
    main()