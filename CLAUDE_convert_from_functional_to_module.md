## Project: Qwen3-TTS OpenVINO Pipeline

Refactoring functional PyTorch TTS code (`qwen3_tts.py`, `qwen3_tts_tokenizer.py`) into `nn.Module` classes suitable for OpenVINO export via `ov.convert_model()`.

### 10 Things to Know

1. **The codebase is entirely functional — no nn.Module classes exist yet.** Every operation is a standalone function taking a flat `weights` dict and string key prefixes. The refactor wraps these into modules with `register_buffer` for weights, but the math stays identical.

2. **TalkerBackbone and CodePredictorBackbone are structurally almost identical.** Both are pre-norm causal decoder transformers with QK-norm (RMSNorm on Q and K after projection, before RoPE), GQA (`repeat_interleave`), and SwiGLU FFN. The only differences: Talker uses 3D M-RoPE, Code Predictor uses standard 1D RoPE and has a `small_to_mtp_projection` linear layer at the input.

3. **RoPE cos/sin must be precomputed outside the module and passed in as inputs.** The M-RoPE merging logic (section-based split/reassemble per `mrope_section`) is complex control flow. Precompute once in Python, slice per call, pass as `(1, 1, S, head_dim)` tensors. The module just does `q * cos + rotate_half(q) * sin`.

4. **The modules must have no value-dependent branching in forward().** For OV tracing: always apply the causal mask (a 1×1 mask is a no-op), always concat KV cache (concat with 0-length past works fine), always do GQA expand (store `gqa_reps` as a fixed int). Dynamic shapes are used — one model handles both prefill (S>1) and decode (S=1).

5. **Embedding lookups and output heads stay outside the modules, in Python/NumPy.** `text_embedding`, `text_projection`, `codec_embedding`, `codec_head`, CP `lm_head.{0-14}`, and CP `codec_embedding.{0-14}` are all handled by the orchestrator. The per-step `lm_head` selection is control flow that can't be traced.

6. **The generation loop has two nested autoregressive levels.** Outer loop: Talker generates one first-codebook token per audio frame. Inner loop: Code Predictor generates 15 sub-codebook tokens given that first token. The CP's KV cache is created fresh each outer step (max 17 tokens), while the Talker's KV cache grows across the whole utterance.

7. **The SpeechDecoder module starts from continuous latents, not discrete codes.** The RVQ codebook decode (codes → latent vectors via table lookup + sum) stays in Python. The module's input is `(1, 512, T)` and it runs: causal_conv → transformer → upsample → conv chain → waveform `(1, 1, T*1920)`.

8. **All three generation modes (CustomVoice, VoiceDesign, VoiceClone) share the same Talker, Code Predictor, and SpeechDecoder.** They only differ in how `build_talker_input()` constructs the prefill sequence. The SpeakerEncoder is only needed for VoiceClone.

9. **Every module must be validated against the original functional code before proceeding.** Create identical inputs, run both paths, assert `allclose`. The original functions are the ground truth. Fix mismatches before moving to the next module.

10. **Weight naming convention**: the flat dict uses dotted paths like `talker.model.layers.{i}.self_attn.q_proj.weight`. When registering buffers, replace dots with underscores. Keep a clear mapping comment so the correspondence is auditable.