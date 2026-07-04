# GwenLand - GWEN-223: Checkpoint Resume with AdamW Optimizer State

**Date:** 2026-06-16 (WIB)
**Scope:** new `train/adamw_state.rs`, `train/layered_training_loop.rs`, `train/native_runner.rs`, `train/mod.rs`, `.kiro/specs/gwen-223-adamw-optimizer-state-serialization/tasks.md`
**Type:** Resume-quality fix for native LoRA training. Checkpoints now get a companion AdamW state file so `--resume` can restore optimizer moments instead of restarting momentum from zero.
**Status:** Implementation and validation complete. Required Wave 1-5 tasks are checked off and the focused train test suite is green. Optional property/E2E extras remain unchecked by design.

---

## Executive Summary

GWEN-222 made resume work for LoRA weights. That was necessary, but it still
left one rough edge: AdamW came back cold. The adapter weights resumed from the
checkpoint, while `m1`, `m2`, and the optimizer step counter started fresh. In
practice that means the first few steps after a resume behave like a small
optimizer warm-up, even though the weights themselves are already mid-run.

GWEN-223 closes that gap. Every periodic weight checkpoint can now be paired
with:

```text
checkpoint_000500_adamw.safetensors
```

The sidecar stores:

- first moment tensors (`m1`);
- second moment tensors (`m2`);
- the AdamW step counter (`step`).

On resume, GwenLand loads the LoRA adapter weights first, then tries to load the
matching AdamW sidecar. If the sidecar is missing or unreadable, training falls
back to the GWEN-222 behavior and continues with fresh optimizer state. A bad
optimizer file should never turn a usable weight checkpoint into a dead end.

---

## Why This Was Needed

Candle's `AdamW` keeps its internal moment tensors private. There is no public
accessor for the optimizer's `m1`, `m2`, or step counter, so the implementation
uses a small mirror alongside the real optimizer:

```rust
type MomentStore = HashMap<String, (Tensor, Tensor)>;
```

After each optimizer step, the training loop applies the same AdamW moment
formula to this mirror:

```text
m1 = 0.9   * m1 + 0.1   * grad
m2 = 0.999 * m2 + 0.001 * grad^2
```

That mirror is what gets serialized. It is deliberately boring, which is what we
want for checkpoint state: easy to inspect, easy to skip, and easy to keep out
of the LoRA weight file itself.

---

## What Changed

### 1. New AdamW state module

`train/adamw_state.rs` is the new home for optimizer-state serialization.

It contains:

- `MomentStore`, keyed by the existing VarMap adapter key;
- VarMap key to AdamW sidecar key translators;
- `varmap_key_for`, which resolves a `Var` by matching Candle tensor IDs;
- `adamw_state_path`, which derives `checkpoint_XXXXXX_adamw.safetensors`;
- `save_adamw_state`;
- `load_adamw_state`.

Sidecar tensor keys follow the stable checkpoint naming shape:

```text
l0.attn_q.lora_a      -> layer_0.attn_q.lora_a.m1
l0.attn_q.lora_a      -> layer_0.attn_q.lora_a.m2
l12.ffn_down.lora_b   -> layer_12.ffn_down.lora_b.m1
```

Fallback single-adapter test fixtures are also supported:

```text
l0.lora_a -> layer_0.lora_a.m1
```

### 2. `LayeredTrainingLoop` now mirrors AdamW moments

The loop now owns:

```rust
moment_store: MomentStore
step_t: usize
```

After Candle applies `self.adamw.step(&grads)`, the loop updates the mirror from
the exact gradient store used by the optimizer. Moment tensors are initialized
as zeros on first use and are shape-checked on every update. If a tensor cannot
be mapped back to a VarMap key, or if a stored moment shape no longer matches
the gradient, the bad entry is skipped with a warning instead of stopping the
run.

Checkpoint writing was promoted from "save weights only" to:

```rust
save_checkpoint_and_adamw_state(step)
```

The weight checkpoint is still the primary artifact. If the weight save fails,
the method returns early. If the AdamW sidecar save fails, the warning is logged
and training continues.

### 3. Resume now restores the sidecar when available

`run_native_local` now does the right ordering:

1. build `LayeredTrainingLoop`;
2. load adapter weights into the already-created VarMap;
3. load the matching AdamW sidecar.

That order matters because Candle's `VarMap::load` only refreshes variables that
already exist.

The resume path is intentionally forgiving:

| Sidecar state | Behavior |
|---|---|
| Present and valid | restore `moment_store` and `step_t` |
| Missing | warn and continue with fresh AdamW state |
| Corrupt or missing `step` | warn, clear moment store, continue |
| Shape mismatch | drop only the mismatched key and keep the rest |

This keeps old GWEN-222 checkpoints compatible. A user can resume from an older
weight-only checkpoint and still train.

---

## Important Implementation Note: `step` Uses `I64`, Not `U64`

The task plan asked for a `U64` step tensor. Candle 0.9.2 does not expose a
`DType::U64` safetensors path, so GWEN-223 stores `step_t` as a one-element
`I64` tensor instead:

```text
step: [i64; 1]
```

The save path checks that `usize` fits into `i64`. The load path rejects negative
values and then converts back to `usize`. This avoids the precision loss that a
float fallback would have introduced.

---

## Validation

New and affected tests cover:

- VarMap <-> AdamW sidecar key translation;
- key round-trip property;
- `MomentStore` initialization and update formula;
- `step_t` increments;
- sidecar path derivation;
- save creates the expected `_adamw.safetensors` file;
- save writes exactly `2N + 1` tensors;
- empty stores still save a valid `step`;
- load returns `Ok(None)` for missing sidecars;
- load rejects corrupt files and files without `step`;
- save/load round-trips tensors and `step_t`;
- resume-time shape filtering drops only mismatched entries;
- independent moment-reference validation against the gradients used for AdamW.

Commands run:

```text
cargo test -p gwenland-core --lib load_adamw_state
-> 6 passed

cargo test -p gwenland-core --lib test_moment_values_match_adamw_internal
-> 1 passed

cargo test -p gwenland-core --lib train
-> 137 passed

cargo check -p gwenland-core
-> passed
```

The warnings shown by Cargo are the existing project warnings
(`Q*_K` naming, a couple unused fields/imports). No new compile error or test
failure was introduced by GWEN-223.

---

## Deliberate Boundaries

The AdamW sidecar is not embedded into the LoRA weight checkpoint. Keeping it as
a companion file preserves the existing adapter checkpoint shape and makes the
new state optional for readers.

The implementation does not try to mutate Candle's private optimizer internals.
Instead, it mirrors the moment math in GwenLand and uses the mirror for
serialization. That is less magical, easier to audit, and avoids depending on
private Candle layout.

The optional property tests listed in the task file are still unchecked. The
required implementation and validation wave are complete, and the high-signal
tests are in place. The optional items can still be useful later, but they are
not blocking the feature.

---

## Files Changed

- **`packages/core/src/train/adamw_state.rs`**
  - new optimizer-state module;
  - key translators;
  - save/load sidecar functions;
  - unit tests for pathing, save, load, corrupt/missing cases, and tensor
    round-trip.
- **`packages/core/src/train/mod.rs`**
  - registered `adamw_state`.
- **`packages/core/src/train/layered_training_loop.rs`**
  - added `moment_store` and `step_t`;
  - updated moments after optimizer steps;
  - saved AdamW sidecars alongside periodic checkpoints;
  - restored and shape-filtered sidecars on resume;
  - added formula, shape, count, load-filter, and independent validation tests.
- **`packages/core/src/train/native_runner.rs`**
  - calls `load_adamw_state` after adapter weights are loaded.
- **`.kiro/specs/gwen-223-adamw-optimizer-state-serialization/tasks.md`**
  - Waves 1-5 and the final checkpoint marked complete.

---

## Result

Resume is now "weights plus optimizer state" when the sidecar exists, and still
"weights only" when it does not. That gives current checkpoints a real
continuation path while keeping older checkpoints and partial artifacts usable.

This is the kind of resume behavior training code should have: if the full state
is there, use it; if it is not, keep moving and tell the operator exactly what
happened.

---

**End of Gwen-Changes-2026-06-16_GWEN-223.md**
