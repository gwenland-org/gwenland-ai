# Stummañ

**The from-scratch training framework.** Pure Rust, built on `glcore` + `glproc`.

Breton *stummañ* — "to shape, to form".

## What level does it work at?

**Gradient and parameter level.** Stummañ is the only crate in the workspace
that runs a backward pass. Where `glproc` computes a forward pass and stops,
Stummañ records it on a tape and walks it backwards.

| Level | Sub-system | What it owns |
|---|---|---|
| **Tensor + autograd** | Kevrin | `Tensor<B>`, the tape, backward closures |
| **Gradient** | Kevskrid | `VLGradStore` — flat `Vec<f32>` per tensor id |
| **Parameter** | Karg | `TPParameter<B>`, LoRA `A`/`B`, in-place updates |
| **Step** | Deskiñ | `Trainer`, the optimizer step, the observer hook |
| **Optimizer state** | Gwellaer | `OPAdamW`, `OPAdafactor`, `OPLion` moments |

~13,500 lines. **327 tests.**

## ⛔ Not `gltrain`

Two different things, and confusing them wastes an afternoon:

| | Stummañ | `gltrain/` |
|---|---|---|
| Built on | `glcore` + `glproc` | **candle** (~35 crates.io deps) |
| Dependencies | `anyhow`, `thiserror` | `candle-*`, `wgpu`, `actix-web`, `tokio`, `reqwest`, … |
| Crates in tree | ~20 | **381** |
| Status | active, this is the direction | legacy, excluded from the workspace |

Stummañ exists because rewriting `gltrain` off candle was the alternative, and
building fresh on the GL engines turned out cheaper than unpicking it.

## ⛔ Not a workspace member

Stummañ declares its own `[workspace]` and is `exclude`d from the root, so the
inference tree never builds it.

**This means `cargo test -p stumman` from the repo root FAILS** with
`package ID specification 'stumman' did not match any packages`. Run it from
inside the crate:

```bash
cd stumman && cargo test        # 317 unit + 9 integration + 1 doc-test = 327
```

## What it trains today (M2)

Honestly: **one linear layer.**

`VLTrainerConfig` is `{d_in, d_out, r, alpha, lr, weight_decay, adapter_seed,
base_seed}`. `Trainer` holds one `ABLinear` and one `LRLora`. The dataset is
`Vec<(Vec<f32>, Vec<f32>)>` with `[1, d_in]` inputs, built in memory. The loss
is MSE.

There is **no tokenizer in the loop, no batching beyond one sample, no
multi-layer model, and no text dataset.** Every token-denominated metric
therefore has no subject, which is exactly what `glbench`'s null-semantics
vocabulary exists to express.

## Observability (M2.5)

`Trainer::set_observer(Box<dyn StepObserver>)` delivers one `VLTrainingStep` per
step: loss, phase timings, gradient statistics, learning rate.

Two things a consumer must know:

- **`VLGradStore` holds activations too.** `Tape::finish_step` returns a
  gradient for *every tensor the tape touched* — nine entries for two
  parameters on the M2 trainer. `VLTrainingStep`'s gradient fields are filtered
  to **parameters only**, because that is what `clip_grad_norm_` means
  everywhere else. The full store still reaches `on_tensors`.
- **`clear_observer` cannot be used to read results.** It hands back
  `Box<dyn StepObserver>`, which cannot be downcast without `Any`. Use shared
  ownership (`Rc<RefCell<…>>`); `tests/observer_boundary.rs` demonstrates the
  pattern from outside the crate.

### Measured overhead

| Layer | Baseline | Observed |
|---|---|---|
| 64×64 | 14,857 ns/step | +4.3% |
| 256×256 | 35,542 ns/step | +11.5% |
| 512×512 | 78,905 ns/step | **+14.2%** |

**Overhead grows with layer width.** The "fixed cost" hypothesis was refuted by
the sweep. With no observer installed, `train_step` reads `is_some()` once and
runs byte-identically to M2 — asserted by test, not assumed.

## Naming

Type names carry two-character semantic prefixes (`TPTensor`, `AGTape`,
`BEGlProc`, `OPAdamW`, `VLGradStore`). Traits take **no** prefix.
See `gl-agent-skills/stumman-naming/SKILL.md` — and read it before adding a
type, because a name is cheapest to fix before it has callers.

## Known issues

`KNOWN_ISSUES.md` is the decision record, not a bug list. KL-001 (`Backend` is
not dyn-compatible) and KL-006 (backward closures capture forward values) both
shaped the API you see.
