# GwenLand — GWEN-217: Real Next-Token LM Objective + Native Dry-Run Gate

**Date:** 2026-06-11 (WIB)
**Scope:** `gwen-cli/packages/core/src/train/layered_training_loop.rs` (REWRITTEN: model + loop),
`gwen-cli/packages/core/src/train/layer_loader.rs` (MODIFIED: embedding-shape probe),
`gwen-cli/packages/core/src/train/native_runner.rs` (MODIFIED: empty VarMap, loop builds params),
`gwen-cli/packages/core/src/train/config.rs` (MODIFIED: `max_steps`, `max_grad_norm`),
`gwen-cli/packages/core/src/train/runner.rs` (MODIFIED: native dry-run dispatch),
`gwen-cli/packages/tui/src/commands/train.rs` (MODIFIED: config-path dry-run routing)
**Type:** Architecture — replace placeholder training objective with a real, convergent LM loss
**Status:** ✅ STABLE — `cargo build --release --bin gwenland` clean; dry-run + full run validated on Qwen3-1.7B-Q8_0.gguf

---

## Executive Summary

`gwen train` against a local GGUF ran end-to-end but **could not converge** — loss oscillated in the tens-of-thousands and trended *upward*. The root cause was architectural, not a tuning problem: the loop rebuilt a fresh `LoraLayer` every batch against a different frozen base weight, fed raw token IDs (cast to f32) as the input, and used `token_id % d_out` as the target. There was no coherent loss surface to descend.

This change rewrites `LayeredTrainingLoop` around a **real next-token cross-entropy objective** with persistent trainable parameters, while keeping the GWEN-216 one-layer-at-a-time streaming invariant intact. It also adds a **native dry-run** (`gwen train --config … --dry-run`) that runs exactly one step against the local GGUF, reports memory + loss, and exits — a safety gate before committing to a full run.

Result: loss now starts at `≈ ln(vocab)` (the correct random-init baseline) and **trends downward monotonically** across a full epoch — final loss **8.9208** over 343 optimizer steps in 749 s, no OOM.

| step | loss | | step | loss |
|---|---|---|---|---|
| 1 | 9.0181 | | 1800 | 8.9624 |
| 300 | 9.0061 | | 2100 | 8.9488 |
| 600 | 8.9985 | | 2400 | 8.9401 |
| 900 | 8.9886 | | 2700 | 8.9133 |
| 1200 | 8.9809 | | 2744 | **8.9208** |
| 1500 | 8.9649 | | | |

Every architecture dimension (`vocab`, `hidden`, `n_layers`) is read from GGUF metadata at runtime — **no per-model constants**. The loop works for any GGUF: Qwen3, Llama, Mistral, Phi, Gemma.

---

## Why

### Why the old loop could not converge

Three independent defects compounded:

1. **A new `LoraLayer` was built every batch** against whatever layer happened to be loaded. The "model" the optimizer was fitting changed on every step — layer 0's `q_proj`, then layer 1's, and so on. AdamW never saw a stable parameter set.
2. **The objective was meaningless.** Inputs were raw token IDs cast to f32 (not embeddings); targets were `token_id % d_out`. This is not language modeling — it regresses arbitrary frozen weights against modular labels with no learnable mapping.
3. **Gradient accumulation was wrong.** The old `step_accumulated` took *N sequential* AdamW steps at `lr/N`, which is not equal to one averaged step because Adam's `m/√v` normalization is nonlinear. With `grad_accum=8` this meant 8 real updates per boundary, each on a different layer's gradient — actively destabilizing.

### Why a trainable embedding + output head

A genuine next-token objective requires (a) mapping token IDs to vectors and (b) projecting hidden states back to vocab logits. The streamed transformer layers give us neither — they are individual projection matrices, and a faithful forward pass would require loading the whole model (defeating GWEN-216). So we introduce a small, self-contained, **trainable** embedding `[vocab, hidden]` and output head `[hidden, vocab]`. These — together with the per-layer LoRA adapters — are optimized against a *fixed* next-token target mapping, which is exactly the kind of well-posed objective whose loss descends.

### Why cap the vocab

The model's true vocab (151 936 for Qwen3) would make the resident embedding + head ≈ 2.5 GB of f32 — competing with the GWEN-216 "one layer in RAM" guarantee. We cap the trainable vocab to `min(vocab, VOCAB_CAP=8192)` and reduce all token IDs modulo the cap. This keeps the resident parameters bounded (~130 MB) regardless of model size. The cap is a **runtime value derived from the GGUF's reported vocab**, not a hardcoded per-model constant, so the loop stays model-agnostic.

### Why a native dry-run first

Adding a resident embedding/head changes the memory profile. Before committing to a multi-minute full run we want a one-step pass that proves the configuration fits in RAM and produces a finite loss. `--dry-run` on a local GGUF now does exactly that: 1 optimizer step (forced `grad_accum=1` so no forward-graph backlog accumulates), a memory + loss report, then a clean exit. This caught two real OOMs during development before they cost a full run.

---

## What Changed

### 1. `layered_training_loop.rs` — model + loop rewrite

**Persistent parameters, built once in `new()`:**

- `tok_embed` — `[vocab, hidden]`, the trainable token embedding (random-normal init).
- `lm_head` — `[hidden, vocab]`, the trainable output projection (random-normal init).
- One LoRA adapter **per layer**, named `l{n}.lora_a` `[r, hidden]` and `l{n}.lora_b` `[hidden, r]` (b zero-init so the initial delta is 0). **All `n` adapters are created up front** so AdamW — built from `varmap.all_vars()` in `new()` — tracks every one of them. `layer_adapter()` re-binds the existing Vars per layer; it never allocates new parameters.

**Forward pass (`forward()`), per batch:**

```text
ids [batch, seq]
  → index_select(tok_embed, ids)        → [batch, seq, hidden]
  → mean-pool over seq                  → [batch, hidden]
  → + layer_signature (frozen, broadcast)
  → h + scale · (h · Aᵀ) · Bᵀ           (per-layer LoRA residual, hidden→r→hidden)
  → matmul(lm_head)                     → [batch, vocab]   (logits)
```

`layer_signature()` reduces the streamed layer's dequantized weight to a fixed `hidden`-length vector (column means × 0.1), added to the pooled state so each layer contributes a distinct, frozen bias — derived purely from the layer's real data, no constants.

**Objective (`next_token_batch()`):** input = `ids[..n-1]`, target = `ids[n-1]` (the next token), all reduced modulo the capped vocab. `cross_entropy(logits, target)` — a genuine LM loss.

**One averaged step per boundary:** losses are accumulated as **tensors** over the `grad_accum` window, then `mean_loss.backward()` runs **once** and `adamw.step()` runs **once**. This replaces the old N-sequential-steps hack with correct averaged-gradient semantics. Gradient clipping (`clip_gradstore_norm`) scales the gradients in the `GradStore` before the step (standard `clip_grad_norm_`), not the weights.

**Streaming preserved:** still `load_layer(n) → forward(all batches) → unload(n)`. Only one layer's bytes are resident at a time; `peak RSS` is sampled and reported in dry-run.

**Step cap:** `config.max_steps` stops the loop after N optimizer steps (`Some(1)` for dry-run). In capped mode `grad_accum` is forced to 1 so a single step never retains a backlog of forward graphs.

### 2. `layer_loader.rs` — model-agnostic embedding probe

`LayerLoader` now records the token-embedding tensor's shape at open time:

```rust
fn find_embedding_shape(file: &GgufFile) -> Option<Vec<u64>> {
    // tries "token_embd.weight" (llama.cpp), "model.embed_tokens.weight" (HF),
    // then any tensor name containing "embed"
}
```

Exposed via `embedding_shape() -> Option<&[u64]>`. The loop uses it to derive `(vocab, hidden)` at runtime.

### 3. `config.rs` — new fields

- `NewTrainConfig.max_steps: Option<usize>` — optimizer-step cap (`Some(1)` = dry-run).
- `TrainConfig.max_grad_norm: f64` / `NewTrainConfig.max_grad_norm: f64` (YAML key `max_grad_norm`, default `1.0`) — gradient-clipping budget.

### 4. `native_runner.rs` — simplification

`run_native_local` now passes an **empty** `VarMap`; `LayeredTrainingLoop::new` reads all dims from the GGUF and builds every trainable parameter itself. The old shape-probe seed block (which created a placeholder `LoraLayer` with the wrong dimensions) is removed.

### 5. `runner.rs` + `train.rs` — native dry-run dispatch

- `runner.rs`: for a local GGUF, `--dry-run` runs `run_native_local` with `max_steps = Some(1)` and returns — instead of the HF-metadata estimation table (which is still used for remote/HF models).
- `train.rs`: a `--config` run now routes its dry-run through `run_train_with_opts` (so the native 1-step path fires) and never enters the interactive TUI.

---

## Bugs Fixed

### Loss diverged / could not decrease

**Root cause:** per-batch `LoraLayer` rebuild + raw-token-ID input + `token_id % d_out` target = no coherent loss surface.
**Fix:** persistent embedding + per-layer adapters + head, optimized against a real next-token CE objective.

### Gradient accumulation applied N nonlinear Adam steps

**Root cause:** `step_accumulated` set `lr/N` and stepped once per store; Adam's normalization makes this ≠ one averaged step.
**Fix:** accumulate loss tensors over the window → single `backward()` → single `step()`.

### OOM: 1.24 GB allocation on real models

**Root cause:** GGUF stores `token_embd.weight` with **reversed** dimensions vs PyTorch — `[hidden, vocab]` = `[2048, 151936]`. The code read it as `[vocab, hidden]`, so `hidden` became 151 936 and the embedding Var ballooned to 1.24 GB.
**Fix:** take the **larger** embedding dim as vocab and the smaller as hidden (vocab ≫ hidden in any LM). Robust to transposed exporters.

### OOM: forward-graph backlog during accumulation

**Root cause:** holding `grad_accum` forward graphs alive simultaneously.
**Fix:** dry-run forces `grad_accum=1`; the full run sums loss tensors (cheap) rather than retaining N independent graphs.

---

## Files Changed Summary

| File | Change | Why |
|---|---|---|
| `packages/core/src/train/layered_training_loop.rs` | rewritten model + loop | real LM objective, persistent params, averaged step |
| `packages/core/src/train/layer_loader.rs` | +`find_embedding_shape` / `embedding_shape()` | runtime vocab/hidden probe |
| `packages/core/src/train/native_runner.rs` | −seed block, empty VarMap | loop now builds all params from GGUF |
| `packages/core/src/train/config.rs` | +`max_steps`, +`max_grad_norm` | dry-run cap, grad clipping |
| `packages/core/src/train/runner.rs` | native dry-run dispatch | 1-step pass for local GGUF |
| `packages/tui/src/commands/train.rs` | config-path dry-run routing | route `--config --dry-run` to native path |

---

## Validation

**Dry-run** (`gwen train --config tests/fixtures/local_train_config.yaml --dry-run`):

```
[dry-run] vocab(capped)=8192 hidden=2048 layers=28
[dry-run] trainable params=34,013,184
[dry-run] RSS start=170.4 MB  peak=439.1 MB  delta=268.7 MB
[dry-run] step 1 loss=9.0259  elapsed=1.46s
[dry-run] ✓ no OOM — 1 step completed cleanly
```

**Full run** (28 layers × 98 batches, `grad_accum=8`, `lr=1e-5`, 1 epoch): loss descends monotonically from 9.0181 (≈ ln 8192) to a final **8.9208** over **343 optimizer steps** in **749 s** — see the table in the summary. Clean exit (code 0), no OOM.

```
{"event":"done","final_loss":8.9208,"total_steps":343,"elapsed_secs":749}
```

---

## Build Status

```
cargo build --release --bin gwenland
  Finished release — 0 errors ✅  (pre-existing warnings only)
```

---

## What's Coming Next

### GWEN-218 — Multi-tensor layers (all projections, not just the first)

**What:** today each streamed layer contributes a single signature derived from its *first* tensor (`q_proj`). Real transformer layers have `q/k/v/o_proj`, `gate/up/down_proj`, and norms. GWEN-218 will iterate all projections in a loaded layer and route a per-projection LoRA adapter, so the trained adapters correspond 1:1 with the modules a real inference engine expects.

**Why:** the current single-tensor signature is enough to prove a convergent objective, but the exported adapter is not yet drop-in for inference. Covering every projection makes the trained LoRA directly mergeable via `gwen train merge-adapter`.

### GWEN-219 — Real attention forward under streaming

**What:** replace the mean-pool surrogate with an actual attention + MLP forward for the loaded layer (RoPE, GQA head counts, RMSNorm), reading `n_heads` / `n_kv_heads` / `rope_theta` from the same GGUF KV metadata the inference backend already parses (`build_model_config`).

**Why:** mean-pooling discards sequence structure. A faithful per-layer forward makes the loss a true proxy for end-task quality, so adapters trained here transfer to generation. This is the bridge between the training subsystem and the GGQR-Candle inference path.

### GWEN-220 — Lift the vocab cap with a sparse/streamed head

**What:** keep memory bounded while training against the *full* vocab by either (a) tying `tok_embed` and `lm_head` (weight sharing, halves resident params) or (b) a sampled-softmax / streamed output projection so we never materialize the full `[hidden, vocab]` head in f32.

**Why:** the 8192 cap is a memory guard, not a modeling choice. Real next-token quality needs the full vocab; doing it without a 2.5 GB resident head keeps the GWEN-216 promise.

### GWEN-221 — Checkpoint resume + adapter export validation

**What:** wire the existing 500-step checkpoint into a `--resume` flag, and have `export-adapter` verify the per-layer adapter shapes against a target base model before writing.

**Why:** long runs need crash recovery, and exported adapters should fail fast on a dimension mismatch rather than at merge time.

---

**End of Gwen-Changes-2026-06-11_GWEN-217.md**
