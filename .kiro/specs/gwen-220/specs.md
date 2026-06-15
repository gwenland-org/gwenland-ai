# GWEN-220 — Real Attention Forward (Replace Mean-Pool with Attn+MLP per Layer)

**Project:** GwenLand (pure Rust, local-first AI dev toolkit — load → inference → train → publish)
**Branch:** `jinxsuperdev/gwen-220-real-attention-forward-replace-mean-pool-with-actual-attnmlp`
**Priority:** High
**Target hardware:** i3 11th gen, 8GB RAM, no discrete GPU (mmap-based, OOM-safe — this constraint is non-negotiable)

> This spec is self-contained for handoff to ChatGPT Codex. No prior conversation context is assumed. Point Codex at the actual repo before Wave 1.

---

## 1. Glossary (for context)

- **GGQR-CF-mmap**: GwenLand's quantized tensor reader — mmap-based, AVX2-accelerated, OOM-safe. Stable, proven (full_dequant 4.3 GiB/s).
- **GgqrCandleBackend**: A *stable, tested (233 tests)* inference backend that does zero-copy handoff from GGQR-dequantized tensors into `candle` for inference. **This is the canonical reference implementation for attention/RoPE/RMSNorm/GQA math** — treat as read-only, do not modify.
- **mistral.rs backend**: A separate, parallel inference path using the `mistralrs` crate directly. Not the focus of this task — do not conflate with GgqrCandleBackend.
- **LayeredTrainingLoop / LayerLoader**: Training-side components (from GWEN-216/217/219) that mmap-load one layer at a time to keep RAM usage low (~100MB peak vs 1.92GB full-load).
- **GDTQP**: A separate, **EXPERIMENTAL/unvalidated** quantization formula behind a `--gdtqp` flag. Out of scope for this task — do not touch.
- **`build_model_config`**: Existing function that parses GGUF KV metadata into a config struct, already exposing `n_heads`, `n_kv_heads`, `rope_theta`, `hidden_size`, `intermediate_size`, `n_layer`, `rms_norm_eps` (verify exact field names in code).

---

## 2. Background / Current State

`LayeredTrainingLoop` (file: `packages/core/src/train/layered_training_loop.rs`) currently computes its forward pass per layer using **mean-pooling** over token embeddings as a surrogate for the real transformer block. This was a deliberate placeholder from GWEN-217 to get a working end-to-end cross-entropy training loop (loss trending ≈9.0 → ~8.92 over 1200 steps on Qwen3-1.7B Q8_0, no OOM — this is the regression baseline).

GWEN-219 (just completed) extended the loop to iterate **all per-layer projections** (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`) and route per-projection LoRA adapters, with an export bridge to `gwen train merge-adapter`. **GWEN-220 must preserve this multi-tensor/per-projection LoRA routing** — it is not being replaced, just plugged into a real forward pass instead of mean-pool.

Mean-pooling discards sequence structure entirely, so the loss is currently *not* a faithful proxy for generation quality, and adapters trained this way may not transfer well to actual inference via GgqrCandleBackend.

---

## 3. Goal

Replace the mean-pool surrogate in `LayeredTrainingLoop` with a real **per-layer transformer block forward**: Attention (with RoPE + GQA) + MLP (SwiGLU), each with RMSNorm and residual connections — implemented or adapted from the existing logic inside `GgqrCandleBackend`.

---

## 4. Non-Goals

- Do **not** modify `GgqrCandleBackend` itself (stable, 233 tests pass — read-only reference for math/structure).
- Do **not** touch `--gdtqp` / GDTQP code paths (separate, unvalidated track).
- Do **not** change the GGUF checkpoint/adapter file format beyond what's strictly needed (checkpoint resume is GWEN-221, vocab cap lift is GWEN-222 — both separate issues).
- Do **not** regress the GWEN-219 multi-tensor/per-projection LoRA routing or the `merge-adapter` export path.

---

## 5. Target Forward Pass (per loaded layer)

For each layer `i` in `LayerLoader`, given input hidden states `[batch, seq_len, hidden_size]`:

1. **RMSNorm** (pre-attention norm, `rms_norm_eps` from GGUF metadata)
2. **QKV projections** — read `q_proj`/`k_proj`/`v_proj` weights for layer `i` (with LoRA delta applied per GWEN-219 routing)
3. **Reshape into heads** using `n_heads` (query heads) and `n_kv_heads` (GQA — repeat/broadcast KV heads to match query head count)
4. **RoPE** applied to Q and K using `rope_theta` and absolute position indices
5. **Scaled dot-product attention** with causal mask (`softmax(QK^T / sqrt(head_dim) + causal_mask) @ V`)
6. **Output projection** (`o_proj`, with LoRA)
7. **Residual add** (input + attention output)
8. **RMSNorm** (pre-MLP norm)
9. **MLP** — SwiGLU: `down_proj(SiLU(gate_proj(x)) * up_proj(x))`, each with LoRA per GWEN-219
10. **Residual add** (output of step 7 + MLP output)
11. Pass resulting hidden states to the next layer (final layer's output feeds the LM head for cross-entropy loss)

---

## 6. Code Reuse Strategy

- Locate the RoPE / RMSNorm / GQA-expansion / attention math currently used inside `GgqrCandleBackend` (likely under `packages/core/src/engine/`).
- Determine what can be **extracted into a shared module** (e.g. `packages/core/src/engine/attention.rs` or similar — exact location TBD by Codex during Wave 1 audit) that both:
  - `GgqrCandleBackend` (inference, no autograd needed) and
  - `LayeredTrainingLoop` (training, needs `candle` autograd-compatible ops)

  can use, **without breaking GgqrCandleBackend's existing 233 tests**.
- If inference-only ops (e.g. KV-cache specific code) aren't reusable as-is for training, that's expected — extract only the parts that are shared (RoPE rotation math, RMSNorm formula, GQA head-repeat logic, SwiGLU formula), and write training-specific wrappers around them.
- LoRA injection points (per GWEN-219) must remain at the same projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.

---

## 7. Implementation Plan (Waves)

Each wave ends with a gate: **do not proceed to the next wave until the gate passes.**

### Wave 1 — Audit & Extraction Plan
- Read `GgqrCandleBackend`'s attention/RoPE/RMSNorm/GQA/MLP implementation.
- Read `layered_training_loop.rs` and confirm exactly how GWEN-219's multi-tensor/per-projection LoRA routing is structured (so Wave 2/3 plug into it correctly).
- Confirm available GGUF KV metadata fields via `build_model_config` (n_heads, n_kv_heads, rope_theta, hidden_size, intermediate_size, rms_norm_eps, head_dim if present).
- Produce a short written plan: what gets extracted to a shared module, what stays inference-only, what's new for training.
- **Gate:** `cargo build` clean, plan written, no source changes yet beyond possibly module scaffolding.

### Wave 2 — Attention Block
- Implement the attention sub-block (steps 1–7 above) as a function/struct usable from `LayeredTrainingLoop`.
- RMSNorm, QKV proj (+ LoRA), GQA head-repeat, RoPE, causal scaled-dot-product attention, output proj (+ LoRA), residual.
- Unit test: given a fixed small input tensor + fixed weights, verify output shape `[batch, seq_len, hidden_size]` and that values are finite (no NaN/Inf).
- **Gate:** `cargo build` clean, new unit test passes, existing test suite still green.

### Wave 3 — MLP Block + Full Layer Assembly
- Implement MLP sub-block (steps 8–10): RMSNorm → SwiGLU (gate/up/down, + LoRA) → residual.
- Assemble full transformer layer = attention block (Wave 2) + MLP block, replacing the mean-pool call site in `LayeredTrainingLoop` for **all** loaded layers.
- Unit test: full-layer forward shape + finite-value check; ideally a small numerical sanity check (e.g. output differs from input, magnitude in a reasonable range).
- **Gate:** `cargo build` clean, tests pass, existing test suite still green.

### Wave 4 — Validation & Regression
- Run `gwen train --dry-run` on **Qwen3-1.7B Q8_0** (the standard regression model).
- Compare loss trend against the GWEN-217 mean-pool baseline (start ≈ `ln(vocab)` ≈ 9.0, trending to ~8.92 over 1200 steps, no OOM). Real attention should still show a sensible downward trend — if it looks *worse* than mean-pool, flag this as a likely bug (not an expected outcome) for review before going further.
- Run a small training pass → `gwen train export-adapter` / `merge-adapter`, confirm output is still drop-in compatible with the GGQR-Candle inference path (no format break).
- Confirm peak RAM stays within the GWEN-216 layered-loading envelope (no full-model materialization).
- **Gate:** dry-run completes without OOM on target hardware, loss trend is sane, merge-adapter round-trip works, full existing test suite green.

---

## 8. Acceptance Criteria (from Linear GWEN-220)

- [ ] Per-layer forward uses real attention (not mean-pool)
- [ ] RoPE + GQA + RMSNorm correctly applied
- [ ] Loss is a faithful proxy for generation quality
- [ ] Adapters trained here are compatible with the GGQR-Candle inference path
- [ ] No regression to GWEN-219 multi-tensor/per-projection LoRA routing
- [ ] `cargo build` clean at every wave gate; existing test suite (233+ tests) stays green
- [ ] Dry-run validated on Qwen3-1.7B Q8_0 on i3/8GB/no-GPU, no OOM

---

## 9. Notes for Codex

- This is a **Rust** project using `candle` for training and a separate `mistralrs`-based path for inference — don't introduce Python or other-language dependencies.
- Be conservative with new crate dependencies; if a hallucinated/unfamiliar crate name comes up, verify it exists on crates.io before adding it to `Cargo.toml`.
- Stop at each wave gate and report status before continuing — don't auto-chain all four waves in one shot.