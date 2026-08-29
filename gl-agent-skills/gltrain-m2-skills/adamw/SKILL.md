# Implementing the Stummañ Optimizer Family

> **Domain:** gltrain-m2-skills
> **Applies to:** `gltrain/src/optim/` (does not exist yet)
> **Scope:** `OPAdamW` (**FULL**), `OPLion`, `OPAdafactor`, `OPAdamW8bit` (all
> **STUB**, with real state shapes)
> **Prerequisite reading:** `gltrain/M2_RESEARCH.md` §4, §7-B, §7-E, §8.3 — the
> full research and architecture record this skill distills.
> **Last updated:** 2026-08-17

## BEFORE YOU START

- [ ] I have read `gltrain/KNOWN_ISSUES.md` KL-005 and KL-006. KL-006 is not
      optional background — it is the reason `step()` cannot take a live `Tape`.
- [ ] I know `OP` is an approved prefix (`gl-agent-skills/gwenland-naming-convention/SKILL.md`,
      decided 2026-08-17). Types are `OPAdamW`, `OPLion`, `OPAdafactor`,
      `OPAdamW8bit` — never `AdamW`/`ABAdamW`/etc.
- [ ] The optimizers' **helper** types are prefixed too: `OPAdamWMoments` (not
      `AdamWMoments`) and `ENAdafactorMoment` (not `AdafactorMoment`). Sketches
      in an earlier revision of this file showed them bare; that was wrong.
      A moments struct only makes sense inside `optim/`, so the domain prefix
      wins over the plain-data `VL` (the `AGNode` precedent in
      `gltrain-naming/SKILL.md`), and the Adafactor one takes `EN` because a
      closed set of variants is its whole job. Landed M2 Wave 3.
- [ ] I have read `gltrain/src/nn/param.rs` (`TPParameter`) and
      `gltrain/src/nn/module.rs` (`Module`, `trainable_parameters`) — the
      optimizer updates these, it does not invent its own parameter type.
- [ ] I have read `gltrain/src/autograd/tape.rs`'s `Tape::finish_step` — the
      only sanctioned way to get gradients out of a tape.
- [ ] Before writing `OPAdafactor` or `OPAdamW8bit` specifically: I have read
      Rules 3 and 4 below in full. Both are stubs whose *shape* is genuinely
      different from AdamW's, not just "AdamW with different numbers" — get the
      state layout wrong and the eventual real implementation inherits a bug.

## Context

Four optimizers, one FULL. AdamW's math is settled — the paper
(Loshchilov & Hutter 2019, arXiv:1711.05101) and PyTorch's implementation agree
exactly, verified term-by-term, so there is no reference/production divergence
to design around. The difficulty there is entirely architectural: KL-006, F1
(gradients live on the tape, not the tensor), F2 (`TensorId` cannot be
persisted). Get those three right and the arithmetic is the easy part — see
Rules 1–7.

The three stubs are not interchangeable placeholders. Each fails to reduce to
AdamW in a *specific, researched* way, and a stub that hides that (by
allocating AdamW-shaped state "for now") teaches the next wave nothing. Rules
8–10 exist so each stub's state shape is correct on day one, even though its
`step()` still refuses to compute.

## Rules — `OPAdamW` (FULL)

1. **The exact update, decoupled weight decay, same-step form** (verified
   equivalent to the paper):

   ```text
   m_t = β1·m_{t-1} + (1-β1)·g_t
   v_t = β2·v_{t-1} + (1-β2)·g_t²
   m̂_t = m_t / (1 - β1^t)
   v̂_t = v_t / (1 - β2^t)
   θ_t = θ_{t-1} - lr·( m̂_t/(√v̂_t + ε) + λ·θ_{t-1} )
   ```

   Note `θ_{t-1}` in the decay term, not the post-update `θ_t`. Applying decay
   *after* the adaptive step (`θ *= (1 - lr·wd)`) is a real bug that appeared in
   this crate's own planning doc (`GLTRAIN_PLAN.md` §3.4) — it leaves a spurious
   `+lr²·wd·u` term. Do not copy that sketch.

2. **`step()` never takes a live `Tape`.** It cannot: the gradients it needs
   live in `grad_store`, and `Tape::finish_step()` already clears the tape
   atomically when it returns them. Signature:

   ```rust
   fn step(&mut self, params: &mut [&mut TPParameter<B>], grads: &VLGradStore) -> Result<()>;
   ```

   The call site looks like:

   ```rust
   tape_guard.backward()?;
   let grads = tape_guard.finish_step();   // tape is now empty, guaranteed
   drop(tape_guard);
   optimizer.step(&mut module.parameters_mut_filtered_trainable(), &grads)?;
   ```

   There is no ordering for a caller to get wrong: you cannot obtain a
   `VLGradStore` without the tape already being cleared. This *is* KL-006
   option (a) ("require `tape.clear()` before `optimizer.step()`, enforced with
   a guard") — implemented as a type-level guarantee instead of a runtime
   `if tape.is_empty() { } else { return Err(...) }` check, which would have
   nothing to check against (the grads it needs would already be gone).

3. **Weight mutation goes through `TPParameter::set_data`, never through a new
   `&mut` accessor on `Tensor`.** `set_data` already refuses a frozen parameter
   and a length mismatch. Do not add a second write path.

4. **Optimizer state is keyed by `TensorId` in memory, by name on disk.**
   `HashMap<TensorId, OPAdamWMoments<B>>` internally — IDs are live and unique for
   the process. But `tensor.rs`'s own doc comment is explicit: *"Do not persist
   or compare IDs across process restarts... Checkpoints must key on parameter
   names, never on these."* So `state_tensors`/`load_state` take the current
   `&[&TPParameter<B>]` list to resolve `TensorId ↔ name` at the serialization
   boundary — the in-memory map never needs to change shape.

5. **Parameter groups exist from the first commit, not as a later refactor.**
   This is what makes LoRA+ (M3) a config change instead of a rewrite (§7-B in
   the research doc: LoRA+ is *only* `η_B = λ·η_A`, no new adapter type — see
   `lora.md` Rule 1). A `VLParamGroup { name, lr_multiplier }` list, resolved
   per-parameter by name, defaulting everything to one group at multiplier 1.0,
   is enough for M2 and is the exact shape M3 needs.

6. **Do arithmetic through `Backend`, not hand-rolled loops.** `Backend::div`,
   `::sqrt`, `::add_scalar` were added to the trait (and both `GlProc`/
   `SisdBackend`) specifically for this optimizer. Route `m`, `v`, and the
   update through them (`B::from_vec`/`B::to_vec` at the boundary with the raw
   `Vec<f32>` gradient and parameter data). This keeps the optimizer
   backend-generic the same way the rest of the crate is, and it is what a
   future GPU backend's in-place kernels would replace transparently.

7. **This runs off-tape.** Do not call any `Tensor` op (`mul_scalar`, `add`,
   ...) inside `step()`. Those record to whatever tape is currently attached to
   their operands, and a parameter tensor is tracked — every optimizer-internal
   op would silently grow the tape by one more node per step, forever (finding
   F4 in the research doc). Operate on `B::Storage`/`Vec<f32>` directly.

## Rules — the three stubs

8. **`OPLion` is sign-based with a single momentum buffer; it is not an AdamW
   approximation.** Chen et al. 2023 (arXiv:2302.06675):

   ```text
   update = sign( β1·m_{t-1} + (1-β1)·g_t )
   θ_t    = θ_{t-1} - lr·( update + λ·θ_{t-1} )
   m_t    = β2·m_{t-1} + (1-β2)·g_t
   ```

   Only `m` is stored — **half** of AdamW's state, one buffer per parameter
   instead of two. The stub must allocate exactly that:
   `HashMap<TensorId, Vec<f32>>` (or `B::Storage`), never a second buffer "in
   case it's needed later" — that would misrepresent the memory claim the
   optimizer exists to make. Defaults differ from Adam's on purpose:
   `β1=0.9, β2=0.99` (Adam's are `0.9, 0.999`) — `β1` shapes the *update*,
   `β2` shapes what the *momentum remembers*, and they are not the same knob
   wearing two names. Capability record: `state_shape: "1x params (momentum
   only)"`, and a note that `lr` should default 3–10x smaller than AdamW's with
   `weight_decay` correspondingly 3–10x larger (effective decay is `lr·λ`, so
   the two must move together or the actual regularization strength silently
   changes when someone swaps optimizers).

9. **`OPAdafactor`'s state *shape depends on the parameter's rank*. This is the
   one genuinely new architectural fact among the three stubs.** Shazeer &
   Stern 2018 (arXiv:1804.04235):

   - For a **≥2-D** parameter `[n, m]`: no full `[n,m]` second moment. Store row
     sums `R: [n,1]` and column sums `C: [1,m]`, reconstruct
     `V̂ = R·C / (1ᵀR)` on demand. Memory `O(n+m)` instead of `O(n·m)`.
   - For a **1-D** parameter (a bias vector): the factorization does not apply
     at all. The full second moment is kept, same shape as the parameter:
     `V̂_t = β̂2_t·V̂_{t-1} + (1-β̂2_t)(g_t² + ε1)`.
   - `β̂2_t = 1 - t^{-0.8}` — a **schedule**, not a fixed constant the way
     AdamW's `β2` is. Do not add a `beta2: f64` config field and treat it as
     settled at construction.
   - Update clipping: `Û = U / max(1, RMS(U)/d)` with `d = 1`.
   - `ε1 = 1e-30`, `ε2 = 1e-3` (the latter is a floor on `RMS(θ)` for the
     relative step size, not an epsilon in a denominator the way AdamW's `ε` is).

   Because of the rank split, the stub's state cannot be one `HashMap<TensorId,
   FixedShapeBuffer>` the way AdamW's is. Allocate an enum from day one:

   ```rust
   enum ENAdafactorMoment<B: Backend> {
       Factored { row: B::Storage, col: B::Storage },  // rank >= 2
       Full(B::Storage),                                // rank == 1
   }
   ```

   picked per-parameter at construction from `param.shape().len()`, even though
   `step()` still refuses. A stub that allocates `OPAdamWMoments`-shaped state
   "as a placeholder" would need a breaking rewrite the day this is actually
   implemented — allocate the right enum now and only the `step()` body is new
   work later.

   Flag, do not resolve: Adafactor's headline feature is a **relative step
   size** (`αₜ = max(ε2, RMS(θ_{t-1}))·ρₜ` — the learning rate itself adapts to
   the parameter's own magnitude). `Optimizer::step`'s signature assumes a
   fixed `lr` per group. Whether that needs a trait change or Adafactor computes
   its own effective `lr` internally per parameter is an open question for
   whoever lands the real implementation — record it in the stub's doc comment,
   don't silently decide it by shipping a signature that only fits AdamW.

10. **`OPAdamW8bit` is a codec over `OPAdamW`, not a fourth update rule.**
    Dettmers et al. 2022 (arXiv:2110.02861) quantizes AdamW's `m` and `v` to
    8 bits — the *update rule is unchanged*, only the storage is. This is why
    `M2_RESEARCH.md`'s cost matrix (§6) calls it an `OptimizerStateCodec`, not a
    fifth entry in the same family as Lion/Adafactor. Concretely:

    - **Both** `m` and `v` quantized (unlike Lion, which drops `v` entirely).
    - **Block-wise**: 2048-element blocks, independent absmax per block. Not a
      single global scale — that is what isolates outliers to one block.
    - **Dynamic tree quantization**, not linear/uniform 8-bit — covers the ~7
      orders of magnitude optimizer states span. A stub allocating `Vec<u8>`
      buckets with a naive linear scale would be a different (worse) codec,
      not a simplified version of this one.
    - **The update itself runs in 32-bit.** Dequantize the 8-bit state to f32,
      run AdamW's ordinary update (Rule 1, unchanged), requantize. Never do
      arithmetic on the quantized bytes directly.
    - **Stable-embedding exception**: embedding-layer optimizer state stays
      32-bit even when everything else is 8-bit. A real implementation needs a
      way to exempt parameters by name pattern; the stub's capability record
      should note this rather than let a later implementer discover it only
      after training an embedding layer badly.

    Architecturally: prefer `OPAdamW8bit` **wrapping** a real `OPAdamW`
    internally and intercepting only `state_tensors`/`load_state` (where the
    quantize/dequantize actually happens) over reimplementing the update math a
    second time. The registry still gives it its own id (`"adamw8bit"`) —
    composition is an implementation detail, not something the registry caller
    needs to know.

## ✅ Correct Pattern

```rust
pub struct OPAdamW<B: Backend> {
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    groups: Vec<VLParamGroup>,
    group_of: HashMap<String, usize>,
    state: HashMap<TensorId, OPAdamWMoments<B>>,
    step_count: usize,
}

impl<B: Backend> Optimizer<B> for OPAdamW<B> {
    fn step(&mut self, params: &mut [&mut TPParameter<B>], grads: &VLGradStore) -> Result<()> {
        self.step_count += 1;
        let bc1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bc2 = 1.0 - self.beta2.powi(self.step_count as i32);
        for p in params.iter_mut() {
            let Some((g, _shape)) = grads.get(p.id()) else { continue }; // no grad this step, skip
            let lr = self.effective_lr(p.name());
            // ... m,v update via B::*, then p.set_data(new_theta)?
        }
        Ok(())
    }
}

// Adafactor's state is rank-dependent from construction, even as a stub:
enum ENAdafactorMoment<B: Backend> {
    Factored { row: B::Storage, col: B::Storage },
    Full(B::Storage),
}
fn new_state<B: Backend>(param: &TPParameter<B>) -> Result<ENAdafactorMoment<B>> {
    if param.shape().len() >= 2 {
        Ok(ENAdafactorMoment::Factored { row: B::zeros(param.shape()[0])?, col: B::zeros(param.shape()[1])? })
    } else {
        Ok(ENAdafactorMoment::Full(B::zeros(param.n_elems())?))
    }
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ takes the live tape — cannot both hold grads and require it empty
fn step(&mut self, tape: &mut Tape) -> Result<()> { ... }

// ❌ decay applied after the adaptive step — leaves a +lr²·wd·u error
param.mul_scalar_inplace(1.0 - lr * wd)?;   // AFTER the update, wrong order

// ❌ Lion allocating a second (v) buffer "to match AdamW's shape"
struct LionState<B: Backend> { m: B::Storage, v: B::Storage }  // v does not exist in Lion

// ❌ Adafactor's stub allocating AdamW-shaped state as a placeholder
struct ENAdafactorMoment<B: Backend> { m: B::Storage, v: B::Storage }
// -- hides the rank-dependent shape; a real impl needs a rewrite, not a fill-in

// ❌ "AdamW8bit" as AdamW + a blanket cast to u8
state.m = state.m.iter().map(|f| *f as u8).collect();
// -- no blocking, no dynamic type, no 32-bit update window, no embedding exception:
//    four separate things this gets wrong at once

// ❌ optimizer state keyed only by TensorId, with no name-resolution path
// (cannot be saved: IDs are not persistable)
pub struct OPAdamW { state: HashMap<TensorId, (Vec<f32>, Vec<f32>)> }
// and no `state_tensors(&self, params: &[&TPParameter<B>])` to re-key it

// ❌ optimizer math via tracked Tensor ops — grows the tape every step
let new_m = state.m.mul_scalar(beta1)?.add(&grad.mul_scalar(1.0 - beta1)?)?;
```

## GwenLand-Specific Notes

- Default hyperparameters match PyTorch: `lr=1e-3, betas=(0.9, 0.999),
  eps=1e-8, weight_decay=1e-2`. No reason to deviate without a measured reason.
- `Backend::div`/`::sqrt` reject a zero divisor / negative input as an error,
  not silently producing `inf`/`NaN` — this is deliberate (see the doc comments
  on those trait methods) and the optimizer should let those errors propagate
  rather than catching and defaulting.
- A `VLOptimizerCapability` record (mirroring `VLAdapterCapability` in
  `lora.md`) should carry: `id`, `status`, `state_shape` (`Fixed(multiplier)`
  for AdamW/Lion, `RankDependent` for Adafactor), `memory_multiplier`
  (2.0 AdamW, 1.0 Lion, ~O(n+m)/O(nm) Adafactor, ~0.25 AdamW8bit), `source`.
- Registry pattern: `OptimizerRegistry<B>` should copy
  `glictus-caliburni/src/plugin.rs`'s `PluginRegistry` exactly — refuse on
  duplicate id, `resolve()` → `Option`, `require()` → `Result`,
  `with_builtins()` preloads all four. See `checkpoint.md` for the same
  pattern applied there.

## Related Skills

- [lora.md](lora.md) — how the adapter this optimizer updates is shaped; where LoRA+ actually lives
- [checkpoint.md](checkpoint.md) — where `state_tensors`/`load_state` get written
- [../rust-skills/trait-design.md](../rust-skills/trait-design.md) — object-safety rules the `Optimizer` trait must follow for the registry
- [../gltrain-naming/SKILL.md](../gltrain-naming/SKILL.md) — the `OP` prefix decision and full M2 name map
