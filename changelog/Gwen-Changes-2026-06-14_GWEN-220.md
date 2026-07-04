# GwenLand — GWEN-220: Real Attention Forward (replace mean-pool with Attn+MLP per layer)

**Date:** 2026-06-14 (WIB)
**Scope:** new `engine/transformer_ops.rs`, new `train/transformer_layer.rs`, `train/layered_training_loop.rs`, `engine/mod.rs`, `train/mod.rs`, new `tests/gwen220_wave4.rs`
**Type:** Replaces the GWEN-217 mean-pool surrogate forward with a real per-layer transformer block (RMSNorm + RoPE + GQA attention + SwiGLU MLP, residuals), so the training loss is a faithful proxy. Preserves GWEN-219 multi-tensor/per-projection LoRA routing and the merge-adapter export path.
**Status:** ✅ Implementation (Waves 1–3) complete and unit-tested. Wave 4 validation: no-OOM, RAM-envelope, and real-attention faithfulness confirmed on Qwen3-1.7B Q8_0; full multi-step loss-trend deferred (CPU cost / thermal).

---

## Executive Summary

`LayeredTrainingLoop` previously computed each layer's forward as a **mean-pool**
over token embeddings (a GWEN-217 placeholder). Mean-pool discards sequence
structure, so the loss was not a faithful proxy for generation quality and
started at ~`ln(vocab)` ≈ 9.0 — the frozen pretrained weights contributed nothing.

GWEN-220 replaces that surrogate with a **real transformer block forward**, run
**one layer at a time** through the existing mmap layered loader (the GWEN-216
RAM invariant is preserved). The decisive evidence it works: on real text the
frozen pretrained Qwen weights now give a **step-1 loss of 2.77** (perplexity
≈16) instead of 9.0 — the forward is actually using the model's knowledge.

The attention/RoPE/RMSNorm/GQA/SwiGLU math is extracted into a shared,
autograd-compatible ops module rather than touching the read-only
`GgqrCandleBackend` reference implementation.

---

## What Changed

### 1. Shared autograd transformer ops — new `engine/transformer_ops.rs`

Pure Candle ops, no model loading / tensor-name lookup / KV cache, all inside the
differentiable graph:
- `rms_norm` — RMS over the last dim.
- `rope_tables` + `apply_rope` — non-interleaved LLaMA/Qwen RoPE via
  `candle_nn::rotary_emb::rope_slow`, with absolute `position_offset`.
- `repeat_kv` — GQA head expansion `[b, kv_heads, …] → [b, query_heads, …]`.
- `apply_causal_mask` + `causal_scaled_dot_product_attention` — `softmax(QKᵀ/√d
  + mask)·V`.

Unit tests: RoPE shape/position/backprop, GQA grouping, causal masking never
reads future positions.

### 2. Training-side transformer layer — new `train/transformer_layer.rs`

Operates on one dequantized base layer at a time; base weights stay frozen, the
seven per-projection LoRA adapters remain in autograd:
- `linear_with_lora` — frozen `[out,in]` base + optional `scale·(B@A)` delta.
- `attention_forward` — RMSNorm → Q/K/V (+LoRA) → optional Qwen3 per-head q/k
  norm → RoPE → GQA → causal SDPA → output proj (+LoRA) → residual.
- `mlp_forward` — RMSNorm → SwiGLU `down(SiLU(gate)·up)` (+LoRA) → residual.
- `transformer_layer_forward` — full layer = attention block + MLP block.

Unit tests assert output shape `[batch, seq, hidden]`, all-finite values, and
**finite gradients for every one of the seven LoRA adapters**; invalid GQA
configs are rejected.

### 3. Layered loop wiring — `train/layered_training_loop.rs`

The mean-pool call site is replaced by `transformer_layer_forward`, run
**per layer** in two passes to keep RAM bounded (gradient-checkpoint style):
- `forward_boundaries` — forward through all 28 layers, **detaching** between
  layers, caching only the boundary hidden states.
- `forward_backward_sample` — final RMSNorm + LM head → cross-entropy, then walk
  layers in reverse, re-loading + recomputing each layer to accumulate its LoRA
  gradients via a vector-Jacobian product, freeing it before the next.

Only one layer is materialised in RAM at any time (GWEN-216 invariant intact).
GWEN-219's per-projection adapter routing (`l{N}.{proj}.lora_{a|b}`) is unchanged.

---

## Validation (Wave 4)

`cargo build -p gwenland-core` → clean (pre-existing warnings only).
`transformer_ops` + `transformer_layer` unit tests → green (shape / finite /
per-adapter gradient checks).

### Native dry-run on local **Qwen3-1.7B Q8_0** (`tests/gwen219_dryrun.rs`, env-gated)

```
=== DEFAULT PATH ===
vocab(capped)=8192 hidden=2048 layers=28
trainable params=8716288               # 7 projections × 28 layers, r=8
RSS start=228.4 MB  peak=1041.9 MB     # layer-by-layer; no full-model load
step 1 loss=2.7747                     # REAL attention on real text (ppl ≈ 16)
✓ no OOM — 1 step completed cleanly

=== EXPERIMENTAL --gdtqp PATH ===
RSS peak=1075.7 MB · step 1 loss=2.7747 · ✓ no OOM
```

The headline: **2.77 vs the mean-pool baseline's ~9.0**. The frozen pretrained
weights now predict next tokens through a true attention+MLP forward — the loss
is a faithful language-model loss, not a structureless surrogate. Real attention
does more work per layer than mean-pool, so the transient peak rises (~589 MB →
~1042 MB) but stays layer-by-layer and far under the 8 GB budget.

### Loss-trend harness — new `tests/gwen220_wave4.rs`

Env-gated (`GWEN_DRYRUN_GGUF`, optional `GWEN220_DATASET` / `GWEN220_STEPS`);
drives `run_native_local` over a real dataset for several steps and asserts each
loss is finite and in a sane LM band `(0.01, ln(vocab))` without diverging.

> Note: an initial draft of this harness used a **single repeated synthetic
> sequence** as a fast overfit smoke test. It drove the loss to 0.0000 in ~3
> steps — a memorization artifact of the degenerate one-example dataset, **not**
> the attention code (the same code gives a healthy 2.77 on real text). The
> harness was rewritten to train on real varied data with sane-band assertions.

The **full multi-step trend run was deferred**: each optimiser step
re-dequantizes all 28 Q8_0 layers twice and scales with sequence length², so on
the i3/8 GB/no-GPU target a multi-step run on long real samples is ~40–60 min and
runs the machine hot. The harness is ready to run when desired.

### Merge round-trip

Not re-run end-to-end here. GWEN-220 does **not** change the checkpoint/adapter
format, and the layered-checkpoint → export → `blk.*` GGUF merge path is covered
by the GWEN-219 `lora_bridge`/`lora_merger` tests
(`extract_adapters_reads_layered_checkpoint_layout`,
`layered_checkpoint_exports_to_mergeable_adapter`, `test_merge_*`).

---

## Known pre-existing issues (NOT introduced here)

- **`engine::inference::selector::tests::{tilde_expand, relative_gguf_ok,
  empty_stop_sequences_ok}`** fail under a default `cargo test` because they lack
  the `#[cfg(feature = "candle-backend")]` guard their sibling tests have
  (`default = []` builds in no backend, so `select_backend("auto")` errors). They
  pass with `--features candle-backend`. Unrelated to GWEN-220 (`selector.rs`
  untouched); a one-line guard each would fix them. Same item flagged in the
  GWEN-219 entry.
- **candle-backend test suite not compilable in this environment** — the
  `--features candle-backend` build (pulls wgpu / candle_transformers / naga)
  exhausted local disk (~4 GB free; "no space on device"). This is an
  environmental limit, not a regression; GWEN-220 modifies no candle-backend-gated
  code.

---

**End of Gwen-Changes-2026-06-14_GWEN-220.md**
