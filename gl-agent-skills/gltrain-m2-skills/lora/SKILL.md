# Implementing and Extending the LoRA Adapter Family

> **Domain:** gltrain-m2-skills
> **Applies to:** `gltrain/src/nn/adapter/` (implemented: `lora.rs`; stubs:
> `dora.rs`, `qlora.rs`, `loha.rs`, `vera.rs`, `locon.rs`)
> **Status:** `LRLora` is **landed and tested** (153/153 tests pass in
> `gltrain/`, including numerical anchors against hand-computed values). This
> skill covers both maintaining it and extending the other five.
> **Prerequisite reading:** `gltrain/M2_RESEARCH.md` §2, §7-A — the full
> per-paper research this skill distills.
> **Last updated:** 2026-08-17

## BEFORE YOU START

- [ ] I have read `gltrain/src/nn/adapter/mod.rs`'s module doc comment — it
      states *why* this is not a `LoRA -> {variants}` subtype tree, with the
      specific paper-level reason each variant breaks that shape.
- [ ] I know `LR` is the approved prefix for adapters (`LRLora`, `LRDora`, ...)
      and there is **no `LRLoraPlus`** — see Rule 1.
- [ ] I have read `gltrain/src/nn/linear.rs`'s module doc comment for the
      **row-vector shape convention** (`y = x @ W`, `W: [d_in, d_out]`) — it is
      the transpose of the LoRA paper's and of PyTorch/candle's, and getting it
      backwards is nearly silent (square projections pass every shape check).
- [ ] If touching `LRLora` itself: I have run
      `cargo test --lib -p gltrain adapter::lora` and it is green before I
      start.

## Context

Reading all seven "LoRA family" items in the original task literally
(LoRA, LoRA+, DoRA, QLoRA, LoCon, LoHa, VeRA) as siblings in one inheritance
tree produces a tree that cannot hold four of the seven. This skill exists so
that fact survives past the person who found it.

## Rules

1. **LoRA+ is not an adapter.** Hayou et al. 2024 (arXiv:2402.12354): the
   *only* change is `η_B = λ·η_A` between the A and B parameter groups. Zero
   shape, init, or forward-pass difference from plain LoRA. It lives on
   `OPAdamW`'s parameter groups (see `adamw.md` Rule 5). If you find yourself
   writing `LRLoraPlus`, stop — you are duplicating `LRLora` field-for-field.

2. **`Adapter::forward` takes the base weight's *values*, not its output**,
   specifically so DoRA can exist without a trait change:

   ```rust
   fn forward(&self, x: &Tensor<B>, base_weight: &Tensor<B>, tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>>;
   ```

   DoRA (Liu et al. 2024, arXiv:2402.09353) computes
   `W' = m · (W0 + BA) / ‖W0 + BA‖_c` — a renormalization of the **combined**
   weight, not `base_out + delta_out`. An adapter trait shaped as "produce a
   delta to add" cannot express this at all. Do not narrow the signature back
   to a base *output* to save a matmul in the additive case; the additive
   adapters lose nothing by taking the weight instead (`LRLora::forward` is the
   reference — it still computes `x.matmul(base_weight)` in one line).

3. **QLoRA is a composition, not a new parameterization.** Its forward
   equation (Dettmers et al. 2023, arXiv:2305.14314) is
   `Y = X·dequant(W_NF4) + X·L1·L2` — the adapter term `+ X·L1·L2` **is**
   `LRLora`, unchanged. `LRQLora` must own a real `LRLora` internally (see
   `qlora.rs`'s `inner_lora()`) rather than reimplementing the math. What is
   actually missing is an NF4 codec and a quantized `BaseWeightSource`, which
   are base-weight-storage concerns, not adapter-math ones. **Never** let
   `LRQLora::forward` silently delegate to the inner LoRA against a dense base
   — that produces a plausible-looking answer that is not QLoRA (no quantized
   memory behavior at all), which is exactly the "requested X, got Y" failure
   `gl-agent-skills` forbids for stubs. It must return
   `GlTrainError::Unsupported` even though the inner LoRA works.

4. **LoHa materializes the full delta; do not assume "low-rank ⇒ cheap".**
   `ΔW = (B1·A1) ⊙ (B2·A2)` (Yeh et al. 2023, arXiv:2309.14859) does not factor
   through `x` the way `x·A·B` does — the Hadamard product has no equivalent
   regrouping. The `[d_in, d_out]` delta has to be formed on every forward pass.
   Set `materializes_delta: true` on its capability record and do not size a
   memory budget for it the way you would for LoRA.

5. **VeRA's random pair is shared across every adapted layer**, not owned per
   layer (Kopiczko et al. 2024, arXiv:2310.11454): "A and B are frozen, random,
   and shared across layers." A tree walk that returns each layer's parameters
   will return the shared pair once per layer if you are not careful — dedupe
   by **name** (`crate::nn::module::trainable_parameters` already does this;
   any new traversal must too, or a shared parameter's effective learning rate
   silently doubles). The two vector lengths are easy to swap: `Λ_d` has length
   **r** (the rank axis), `Λ_b` has length **d_out** (the output axis) — check
   against the parameter-count formula `|Θ| = L_tuned·(d_model + r)` if unsure.

6. **The checkpoint format must be able to say "generated from this seed", not
   only "here are the bytes".** This falls directly out of Rule 5: VeRA's paper
   states the frozen matrices "do not need to be stored in memory" and "can be
   regenerated from a random number generator (RNG) seed". A checkpoint schema
   designed only around dense tensors will discover this requirement late and
   have to widen. See `checkpoint.md` Rule 6.

7. **LoCon is blocked at the tensor layer, not the adapter layer — say so.**
   `Tensor::matmul`'s `check_matmul_shapes` rejects anything but rank 2
   outright. There is no conv op, no im2col, no 4-D tensor anywhere in the
   crate. Scoping "implement LoCon" as adapter work misjudges it by an order of
   magnitude; it is a convolution-support project that happens to end with an
   adapter. It is also worth flagging that LoCon exists for diffusion U-Nets —
   GwenLand tests against Llama/Qwen2/Qwen3, none of which have a convolution —
   so the real open question is whether diffusion-model support is in scope at
   all, not "when do we implement LoCon".

8. **Init: LoRA's `A` is `N(0, std²)`, `B` is exactly zero.** The paper says
   Gaussian for A without stating a variance; PEFT's default is actually
   `kaiming_uniform_(a=√5)`, with Gaussian as an opt-in (`std = 1/r`). This
   implementation follows the paper with `std = 1/r`. **Record which one you
   used** — a checkpoint trained under one init is not numerically comparable
   to one trained under the other, and the divergence is silent (both start the
   adapter at `ΔW = 0`, so early loss curves look identical).

9. **Scale is `alpha/r`, with an `rslora` flag for `alpha/√r`.** Both are real
   reference-implementation behaviors (LoRA paper vs. PEFT's `use_rslora`); pick
   one as default, expose the other as a flag, never hardcode just one.

10. **Every stub allocates its *real* researched parameter shapes.** A stub
    that allocates nothing, or allocates a made-up placeholder shape, teaches
    the next wave nothing. `LRDora` allocates `A`, `B`, and a `[1, d_out]`
    magnitude vector at construction and only fails on `forward`/`merge_into`;
    `LRLoHa` allocates all four real matrices with correctly-differing inits
    (one zeroed factor *per branch*, not all four — zeroing everything kills
    every gradient). `LRLoCon` is the deliberate exception: it allocates
    nothing, because a rank-4 conv-kernel shape has no honest representation in
    a crate with no 4-D tensors, and a placeholder flattening would be mistaken
    for a decision later.

## ✅ Correct Pattern

```rust
// DoRA's forward signature needs base VALUES — this is why the trait takes them:
fn forward(&self, x: &Tensor<B>, base_weight: &Tensor<B>, tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>> {
    // additive (LoRA): base_out + scale * (x @ A @ B)     -- only reads base_weight for one matmul
    // DoRA (M3): needs base_weight's *values* to build W0 + BA before renormalizing
}

// QLoRA composes rather than reimplements:
pub struct LRQLora<B: Backend> {
    inner: LRLora<B>,          // the adapter math IS LoRA
    base_format: ENBaseWeightFormat,  // what's actually new
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ LoRA+ as a sibling type — duplicates LRLora for a learning-rate difference
pub struct LRLoraPlus<B: Backend> { a: TPParameter<B>, b: TPParameter<B>, /* ... */ }

// ❌ narrowing the trait to a base OUTPUT — makes DoRA structurally impossible
fn forward(&self, x: &Tensor<B>, base_out: &Tensor<B>, tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>>;

// ❌ QLoRA silently falling back to a dense-base LoRA forward pass
impl Adapter<B> for LRQLora<B> {
    fn forward(&self, x, base_weight, tape) -> Result<Tensor<B>> {
        self.inner.forward(x, base_weight, tape)   // looks right, is NOT QLoRA
    }
}

// ❌ LoHa's capability record implying it's as cheap as LoRA
materializes_delta: false,   // WRONG for LoHa — it must form the full delta

// ❌ a shared-parameter tree walk with no dedup, silently 2x'ing VeRA's LR
fn parameters(&self) -> Vec<&TPParameter<B>> {
    layers.iter().flat_map(|l| l.parameters()).collect()   // shared A/B counted N times
}
```

## GwenLand-Specific Notes

- Numerical tests for any adapter need a **hand-computed anchor**, not just
  "loss goes down". `lora.rs`'s
  `forward_matches_a_hand_computed_example` is the template: pick `r=1`,
  compute the expected output with a calculator, assert against it at
  `TOL_MATMUL = 1e-4`.
- Every adapter test file should include a test that the frozen base weight
  **never receives a gradient** (`the_frozen_base_weight_never_receives_a_gradient`
  in `lora.rs`) — this is the property that makes the whole method's memory
  argument true, and it is exactly the kind of thing that regresses silently if
  someone "simplifies" the forward pass later.
- Registry: `AdapterRegistry<B>` follows `glictus-caliburni/src/plugin.rs`'s
  `PluginRegistry` — refuse duplicate ids, `Option` for `resolve`, `Result` for
  `require`, `with_builtins()` preloads all six (one full, five stub).

## Related Skills

- [adamw.md](adamw.md) — where LoRA+ actually lives (parameter groups)
- [checkpoint.md](checkpoint.md) — VeRA's seed-not-bytes requirement, transposed-shape validation
- [../gltrain-naming/SKILL.md](../gltrain-naming/SKILL.md) — the `LR` prefix and full M2 name map
- [../architecture-skills/inference-first.md](../architecture-skills/inference-first.md) — Rule 1 (reference impl first) and Rule 5 (no speculative complexity), both load-bearing on how much a stub should build
