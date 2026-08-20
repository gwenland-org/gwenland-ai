# Stummañ M2 — Research, Comparative Analysis, and Proposed Architecture

> **Status:** RESEARCH / DESIGN. No implementation code has been written against
> this document yet.
> **Scope:** the LoRA-variant, checkpoint, and optimizer "skill" families.
> **Baseline measured:** `cargo test --lib` in `gltrain/` → **62 passed, 0 failed**
> (Windows 11, rustc via the repo toolchain, 2026-08-17).
> **Date:** 2026-08-17

---

## 0. Premise corrections before anything else

Three premises in the task framing do not survive contact with the repository.
They are corrected here because every downstream decision depends on them.

### 0.1 "gltrain" is two different things, and the one that matters is `gltrain/`

| | `gltrain/` | `gltrain/` |
|---|---|---|
| Crate name | `gwenland-core` | `gltrain` |
| Workspace | **excluded** from root workspace | own workspace root (`[workspace]` in its manifest) |
| Deps | candle-core/nn/transformers, wgpu, tokio, actix-web, reqwest, safetensors, ~40 more | `glcore`, `glproc`, `anyhow`, `thiserror` |
| Autograd | candle's | its own, hand-written |
| Status | legacy monolith, still builds | active, M1 complete |

The root `Cargo.toml` states the exclusion reason directly: gltrain "mandatorily
depends on candle … which violates this workspace's 'Inference First' rule of
zero external ML dependencies".

`gltrain/GLTRAIN_PLAN.md` is literally titled **"Stummañ — gltrain Planning
Document"**, and its **Milestone 2** is *"LoRA Training on glproc"* with
deliverables `nn/lora.rs`, `optim/adamw.rs`, `checkpoint/saver.rs`. That is the
M2 in this task. **All work below targets `gltrain/`.** `gltrain/` is treated
as prior art to read, not as a codebase to extend.

### 0.2 The plan's own M2 sketches are stale in three places

`GLTRAIN_PLAN.md` was written before M1 existed. Verified against the tree:

1. **§3.8 says "Reuse existing LoRA logic from gltrain/src/train/lora.rs".**
   That file exists but is 69 lines of candle `VarBuilder`/`VarMap` glue. There
   is nothing to reuse; the API it is built on does not exist in gltrain.
2. **§3.4's AdamW sketch calls methods that do not exist** — `Parameter`,
   `param.grad()`, `Tensor::zeros_like`, `div_scalar`, `sqrt`, `add_scalar`,
   `sub_inplace`, `mul_scalar_inplace`. None are in the crate. §3.8 also calls
   `Tensor::randn`, which does not exist and cannot without an RNG.
3. **§3.4's AdamW sketch applies weight decay incorrectly.** It does the
   adaptive step first, then `param *= (1 - lr*wd)`. That multiplies the decay
   into the update term as well, leaving an `+lr²·wd·u` error. The correct
   decoupled form subtracts both from θ_{t-1}. See §5.1.

Treating these sketches as specifications would ship all three defects. This is
the failure mode `KNOWN_ISSUES.md` already documents twice ("wave specs shipped
real bugs; only running the tests caught them").

### 0.3 There is no runtime "skill system" in the repo to plug into

`gl-agent-skills/` is a **documentation** directory of coding rules for agents.
It is not a runtime registry. The repo's actual registry pattern is
`glictus-caliburni/src/plugin.rs` (`PluginRegistry`), and it is a good one —
§8.4 adopts its exact shape. The word "skill" in this task maps onto that
pattern, not onto `gl-agent-skills/`.

---

## 1. Repository analysis

### 1.1 What `gltrain` actually is, as of M1

```
gltrain/src/
    lib.rs          re-exports
    error.rs        GlTrainError, Result<T>
    tensor/         Kevrin  — Tensor<B>, Backend trait
    backend/        Karg    — GlProc (glproc kernels), SisdBackend (scalar oracle)
    autograd/       Kevskrid — Tape, ComputationNode, VLGradStore, ops
```

**The complete `Tensor<B>` op surface** (`tensor/tensor.rs`):

| Category | Ops |
|---|---|
| Construct | `zeros`, `ones`, `from_vec` |
| Shape | `shape`, `n_elems`, `ndim` |
| Data | `to_vec`, `item` |
| Autograd | `with_grad`, `id`, `requires_grad`, `tape`, `detach` |
| Math | `matmul`, `transpose`, `add`, `sub`, `mul`, `mul_scalar`, `relu` |
| Reduce | `sum`, `mean` (→ shape `[1]`, tape-recording), `sum_scalar`, `mean_scalar` (raw f32, no node) |

**The complete `Backend` trait**: `zeros`, `ones`, `from_vec`, `to_vec`,
`matmul`, `transpose`, `add`, `sub`, `mul`, `mul_scalar`, `relu`, `sum`, `mean`.

**What does not exist:** `Parameter`, `Module`, `Optimizer`, `Dataset`, any
`nn/`, `optim/`, or `checkpoint/` module, `div`, `sqrt`, `add_scalar`,
`div_scalar`, `neg`, `randn`, any RNG, any conv op, any tensor of rank > 2 in
`matmul` (`check_matmul_shapes` rejects non-2-D outright), any serialization.

### 1.2 The five facts that constrain the design

**F1 — Gradients live on the tape, not on tensors.**
`VLGradStore` maps `TensorId → (Vec<f32>, Vec<usize>)`. There is no `grad` field
on `Tensor`. Any `Parameter` abstraction must read gradients *through the tape*,
so a parameter is `(name, Tensor<B>)` plus a tape handle, not a self-contained
cell. The plan's `param.grad()` implies the opposite and would need a redesign
of the tape to work.

**F2 — `TensorId` is process-global and explicitly not persistable.**
`tensor.rs:18-35` is unambiguous:

> **Do not persist or compare IDs across process restarts.** … Checkpoints must
> key on parameter names, never on these.

So the *name* is the primary key of every serialized artifact, and optimizer
state must be re-keyed by name on load. This kills any design that stores
optimizer moments keyed by tensor identity.

**F3 — KL-006 blocks in-place weight update, which is exactly what `step()` does.**
`matmul`, `mul` and `relu` capture `Vec<f32>` snapshots of their forward inputs
at record time. If the optimizer mutates weight storage in place, a live tape
holds pre-update values and the next backward pass computes gradients against
weights that no longer exist — silently, with a plausible loss curve.
`KNOWN_ISSUES.md` states this must be resolved **with** the in-place update, not
after. This is M2's single highest-severity risk.

**F4 — Any tensor op on a tracked parameter records a tape node.**
`record_op` returns early only when *no* operand requires grad. A parameter is
tracked by definition, so `param.mul_scalar(0.9)` inside `step()` would append a
node to the tape every step, growing it without bound and polluting the next
backward. **The optimizer must therefore run off-tape**, on raw buffers, not on
tracked `Tensor` values. The plan's §3.4 sketch does the opposite.

**F5 — The dependency budget is four crates.**
`glcore`, `glproc`, `anyhow`, `thiserror`. Inference-First Rule 6: "no external
ML dependencies, no C bindings". gltrain is the training arm and is allowed
*more* latitude than the inference tree, but nothing here needs it (§7).

### 1.3 Reusable prior art found in-tree

| Need | Existing asset | Verdict |
|---|---|---|
| Registry with capability metadata | `glictus-caliburni/src/plugin.rs` + `traits/plugin.rs` | **Adopt the pattern verbatim** (§8.4) |
| Safetensors *reading* | `glcore/src/format/safetensors.rs` (164 lines, mmap, from scratch) | **Reuse as the round-trip oracle** |
| Safetensors *writing* | **does not exist anywhere in the repo** | Must write (~100 lines, §8.5) |
| Deterministic RNG | `glproc/src/sampler.rs` xorshift64*, "no `rand` dependency" | **Copy the approach** for LoRA init |
| Versioned validation | `glictus-caliburni/src/manifest/validator.rs`, rules V01–V17, `ValidationResult{errors,warnings}`, major=error / minor=warning | **Adopt the pattern** |
| Optimizer-state serialization | `gltrain/src/train/adamw_state.rs` — saves to a **sidecar** `{stem}_adamw.safetensors` | Confirms the §7-E split |
| LoRA math | `gltrain/src/train/lora.rs` | Read for shape conventions only; candle-bound |

Note the last row of prior art contradicts `GLTRAIN_PLAN.md` Q8, which decided
"no optimizer state in checkpoints". gltrain ships optimizer-state save/load
anyway. Q8 is a *scoping* decision, not an architecture one — §7-E treats it as
such.

---

## 2. Research findings — LoRA family

All formulations below are from the primary papers; sources listed in §12.

### 2.1 LoRA (Hu et al., 2021)

> `h = W₀x + ΔWx = W₀x + BAx`, with `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, `r ≪ min(d,k)`.

- **Init:** "random Gaussian initialization for A and zero for B, so ΔW = BA is
  zero at the beginning of training."
- **Scaling:** "We then scale ΔWx by α/r, where α is a constant in r… This
  scaling helps to reduce the need to retune hyperparameters when we vary r."
- **Merge:** "we can explicitly compute and store W = W₀ + BA … this guarantees
  that we do not introduce any additional latency during inference."
- **Dropout:** not in the paper. It comes from the reference/PEFT
  implementation, applied to the *input*: `lora_B(lora_A(dropout(x))) * scaling`.
- **PEFT divergences worth recording:** `lora_A` init is
  `kaiming_uniform_(a=√5)` by default, *not* Gaussian (Gaussian is an opt-in
  `init_lora_weights="gaussian"` with `std=1/r`); and `use_rslora` switches the
  scaling to `α/√r`. Two implementations, two defaults — documented, not
  silently picked.

### 2.2 LoRA+ (Hayou et al., 2024)

> "set the learning rates for A,B such that **η_B = λ·η_A** with λ>1 fixed and
> tune η_A."

- **The architecture is byte-for-byte identical to LoRA.** A and B keep their
  shapes, init, and scaling.
- Theorem 1: efficient feature learning needs `η_A = Θ(n⁻¹)`, `η_B = Θ(1)` for
  width `n`; so the ratio scales as `Θ(n)`.
- Empirical λ: `≈2⁴` for RoBERTa under one init, `2²–2³` under another, `2¹–2²`
  for Llama. No single universal constant.
- **Conclusion: LoRA+ is not an adapter. It is a learning-rate policy.** See §7-B.

### 2.3 DoRA (Liu et al., 2024)

> `W = m · V/‖V‖_c`, where `‖·‖_c` is the vector-wise norm **across each column**.
> Fine-tuning: `W′ = m · (W₀ + BA)/‖W₀ + BA‖_c`.

- **Trainable:** `m` (one scalar per output column), `A`, `B`. `W₀` frozen.
- `m` initialised to `‖W₀‖_c`.
- **Memory trick:** "treat `‖V+ΔV‖_c` as a constant, thereby detaching it from
  the gradient graph" — ~24.4% gradient-memory reduction on LLaMA.
- Mergeable before inference, no added latency.

**This is the structurally important one.** DoRA is *not* `base_out + adapter_out`.
It renormalises the **combined** weight, so:
- it needs the base weight **values** at forward time, not just the base output;
- the adapter output cannot be computed independently and added;
- it needs a column-norm op and a detach-from-graph op that gltrain lacks.

Any abstraction shaped as "frozen base output + additive adapter delta" cannot
express DoRA. This is the single strongest argument against a flat
`Adapter`-subtype tree.

### 2.4 QLoRA (Dettmers et al., 2023)

> `Y^BF16 = X^BF16 · doubleDequant(c₁^FP32, c₂^{k-bit}, W^NF4) + X^BF16 L₁^BF16 L₂^BF16`

- **NF4:** quantile-based 4-bit type, `qᵢ = ½(Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1)))`
  with `Q_X` the standard-normal quantile function; "information theoretically
  optimal for zero-centered normally distributed data"; block-wise absmax
  normalisation.
- **Double Quantization:** quantises the quantisation constants themselves,
  FP8 with second-level blocksize 256 → **0.5 → 0.127 bits/param**, saving 0.373.
- **Compute dtype:** weights "dequantized from storage to BFloat16, then perform
  matrix multiplication in 16-bit". Storage precision ≠ compute precision.
- **Paged optimizers:** NVIDIA unified memory; optimizer state evicted to CPU RAM
  on OOM and paged back. A *memory-management* feature, unrelated to the adapter.

**The adapter term `+ X L₁ L₂` is plain LoRA, unchanged.** QLoRA is a
composition of three independent things (§7-C).

### 2.5 LoCon / LyCORIS (Yeh et al., 2023)

- Extends LoRA to **Conv2d**. A conv with kernel `(out, in, kh, kw)` factors into
  a down-conv `Conv(in, dim, ksize, stride, padding)` then an up-conv
  `Conv(dim, out, 1)`; with the optional Tucker form
  `Conv(in,dim,1×1) → Conv(dim,dim,kh×kw,stride,padding) → Conv(dim,out,1×1)`.
- `rank(ΔW) ≤ dim`.

**Blocked below the adapter layer.** gltrain has no conv op, no 4-D tensors, and
`check_matmul_shapes` rejects anything that is not 2-D. LoCon is not gated on the
adapter abstraction at all; it is gated on the tensor layer. Recording that is
more useful than a stub that pretends otherwise.

### 2.6 LoHa (LyCORIS, from FedPara)

> `ΔW = (B₁A₁) ⊙ (B₂A₂)`, `B₁,B₂ ∈ ℝ^{p×r}`, `A₁,A₂ ∈ ℝ^{r×q}`, `⊙` = Hadamard.
> Forward: `h′ = W₀h + b + γ[(B₁A₁) ⊙ (B₂A₂)]h`.

- **Four** trainable matrices, not two.
- Rank up to `r²` versus LoRA's `2r` for comparable parameter count; "2r < r²"
  for r > 2.

**Critical memory consequence.** `(B₁A₁) ⊙ (B₂A₂)` does not factor through `x`:
you cannot compute it as a chain of matrix–vector products the way LoRA can. The
full `p×q` `ΔW` **must be materialised** every forward pass. LoRA's headline
memory property — never forming `ΔW` during training — does not hold for LoHa.
LyCORIS works around it with "custom backward which will reconstruct B and A when
actually needed". Any abstraction that assumes an adapter is cheap because it is
low-rank is wrong for LoHa.

### 2.7 VeRA (Kopiczko et al., 2024)

> `h = W₀x + Λ_b B Λ_d A x`

- `A` and `B` are **frozen, random, and shared across all adapted layers**.
- Trainable: the scaling vectors only — `d` of length **r**, `b` of length
  **d_out**. (The parameter count `|Θ| = L_tuned × (d_model + r)` is the internal
  check: `d_model` for `b`, `r` for `d`.)
- Init: Kaiming for `A`,`B`; `Λ_d` to a constant `d_init` (e.g. 10⁻¹); `Λ_b` to
  **zeros**, matching LoRA's zero-init of B.
- The frozen matrices "do not need to be stored in memory" — they "can be
  regenerated from a random number generator (RNG) seed".

**Two structural consequences.** (1) Parameters are *not owned per layer*: a
`Module::parameters()` that assumes each layer owns its weights double-counts the
shared `A`/`B` or misses them. (2) The checkpoint stores an **RNG seed**, not
tensors, for the largest arrays — so the serialization format must be able to
express "this tensor is generated, here is its seed", not just "here are bytes".
That is a format-level requirement, discoverable only from the paper.

---

## 3. Research findings — checkpoints

### 3.1 Format primitives

**safetensors** (official spec): `8 bytes LE u64 header length` → `N bytes UTF-8
JSON` → raw byte buffer. Header maps `name → {dtype, shape, data_offsets:[BEGIN,END)}`,
offsets **relative to the start of the byte buffer**. Reserved key `__metadata__`
is a free-form **string→string** map (arbitrary JSON is *not* allowed). Duplicate
keys disallowed; the buffer "needs to be entirely indexed, and cannot contain
holes"; little-endian; row-major; header capped at 100 MB against DoS.

`glcore/src/format/safetensors.rs` already implements the reader, including the
out-of-bounds offset check. **Only the writer is missing.**

**GGUF** (ggml spec): magic `0x46554747`, `u32` version (3), `u64 tensor_count`,
`u64 metadata_kv_count`, then typed KV metadata (13 value types), then tensor
info (`name` ≤ **64 bytes**, `u32 n_dimensions` ≤ 4, `u64[] dims`, `ggml_type`,
`u64 offset`), then the aligned data section. Alignment comes from
`general.alignment` (multiple of 8), **default 32** when absent; every tensor
offset is a multiple of it and inter-tensor space is `0x00`-padded. LLM tensor
names follow `blk.N.attn_q.weight` etc.

### 3.2 Sharded checkpoints

HuggingFace convention: shards named `model-00001-of-00006.safetensors`, plus an
index `model.safetensors.index.json`:

```json
{ "metadata": { "total_size": 28966928384 },
  "weight_map": { "lm_head.weight": "model-00006-of-00006.safetensors", … } }
```

Default max shard size 5 GB. The index is a pure **name → file** map: sharding is
a *layout* over a logical tensor bundle, and the bundle's identity is unchanged.

### 3.3 Incremental / delta checkpoints

Check-N-Run (Eisenman et al., NSDI '22) is the reference system. Its two
techniques are **differential checkpointing** ("tracks and checkpoints the
modified part of the model") and quantization of the checkpoint. Two distinct
incremental modes exist: *baseline-relative* (everything modified since the full
checkpoint) and *consecutive* (only what changed in the last interval) — the
second is smaller per file but makes reconstruction depend on the whole chain.

**The finding that matters:** Check-N-Run's differential mode is effective
because recommendation-model embedding tables are *sparsely* updated. The paper's
own delta-compression scheme is noted as having **limited applicability to
traditional deep learning models**, where every parameter changes every step.

For LoRA specifically, *every* adapter parameter receives a gradient every step,
so a naive "changed tensor detection" delta saves nothing. Incremental
checkpointing for gltrain would have to be value-delta + compression, not
presence-delta. This substantially lowers its priority (§10).

---

## 4. Research findings — optimizers

### 4.1 AdamW (Loshchilov & Hutter, 2019)

Algorithm 2, decoupled form — weight decay is **removed from the gradient** and
applied directly to the parameter:

```
g_t  ← ∇f_t(θ_{t-1})                        (NO λθ term — that is the L2 variant)
m_t  ← β₁ m_{t-1} + (1-β₁) g_t
v_t  ← β₂ v_{t-1} + (1-β₂) g_t²
m̂_t  ← m_t / (1-β₁^t)
v̂_t  ← v_t / (1-β₂^t)
θ_t  ← θ_{t-1} − η_t ( α·m̂_t/(√v̂_t + ε) + λ·θ_{t-1} )
```

**Paper vs PyTorch — checked, and they agree.** PyTorch documents the decay as a
separate earlier line (`θ ← θ − γλθ`, then the adaptive step). Both forms
subtract `γλθ_{t-1}` and `γ·m̂/(√v̂+ε)` from `θ_{t-1}`, so they are algebraically
identical. Likewise `m̂/(√v̂+ε)` and PyTorch's coded
`(lr/bc₁)·m/(√v/√bc₂ + ε)` are the same expression. **No divergence to
document.** PyTorch defaults: `lr=1e-3, betas=(0.9,0.999), eps=1e-8,
weight_decay=1e-2`.

The one thing that *is* easy to get wrong, and that `GLTRAIN_PLAN.md` §3.4 gets
wrong, is applying decay **multiplicatively after** the adaptive step
(`θ ← (θ − lr·u)(1 − lr·wd)`), which leaves a spurious `+lr²·wd·u`.

### 4.2 Lion (Chen et al., 2023)

```
update ← sign( β₁·m_{t-1} + (1-β₁)·g_t )
θ_t    ← θ_{t-1} − lr·( update + λ·θ_{t-1} )
m_t    ← β₂·m_{t-1} + (1-β₂)·g_t
```

- **One** momentum buffer, not two → ~50% of AdamW's optimizer memory.
- Defaults `β₁=0.9, β₂=0.99` (Adam's are 0.9/0.999); the two betas serve
  different roles — the update interpolates, the state EMA remembers "a ~10x
  longer history".
- lr "typically 3–10x smaller than AdamW"; λ correspondingly 3–10x **larger**,
  since effective decay is `lr·λ`.
- The update is `sign(...)` — magnitude information is discarded entirely. It is
  **not** an AdamW approximation and shares no state layout with it.

### 4.3 Adafactor (Shazeer & Stern, 2018)

- Factored second moment: keep row sums `R` and column sums `C`, reconstruct
  `V̂ = R·C / (1ᵀR)`. Memory `O(n+m)` instead of `O(nm)`.
- `β̂₂ₜ = 1 − t^{-0.8}`.
- Update clipping: `Û = U / max(1, RMS(U)/d)` with `d = 1`.
- Relative step size: `αₜ = max(ε₂, RMS(X_{t-1}))·ρₜ` — the LR scales with the
  *parameter's own magnitude*.
- `ε₁ = 1e-30`, `ε₂ = 1e-3`.
- **Dimensionality is load-bearing:** factorisation applies only to ≥2-D
  parameters. For vectors, the full second moment is kept
  (`V̂ₜ = β̂₂ₜV̂_{t-1} + (1-β̂₂ₜ)(G²ₜ + ε₁)`). So Adafactor's state *shape depends
  on the parameter's rank* — the only optimizer here for which that is true, and
  the reason the optimizer-state abstraction cannot assume "state is the same
  shape as the parameter".

### 4.4 8-bit AdamW (Dettmers et al., 2022)

- **Both** `m` and `v` quantised to 8 bit. Parameters and gradients stay at their
  original precision (typically 16-bit mixed).
- **Block-wise:** blocks of **2048** elements, independent absmax per block.
  Isolates outliers to one block and needs no cross-core synchronisation.
- **Dynamic tree quantization**, extended for non-negative tensors by
  repurposing the sign bit; covers the ~7 orders of magnitude optimizer states
  span.
- **The update itself is done in 32-bit**: "dequantize the 8-bit optimizer states
  to 32-bit, perform the update, and then quantize the states back to 8-bit",
  element-wise in registers, no temporary buffer.
- **Stable embedding layer**: embedding optimizer states stay **32-bit** while
  other layers use 8-bit.
- Memory: 8 GB → 2 GB per 1B params.

So `AdamW + cast state to u8` is wrong on four counts: no blocking, no dynamic
type, no 32-bit update window, no per-layer precision exemption.

---

## 5. Research matrix I — mechanism

| Skill | Core mathematics | Trainable state | Base-model state | Optimizer interaction | Serialization | Quantization | Mergeable | Architecture type |
|---|---|---|---|---|---|---|---|---|
| **LoRA** | `h = W₀x + (α/r)·BAx` | `A(r×k)`, `B(d×r)` | frozen dense | none | 2 tensors/site + `r`,`α` | none (fp32 adapter) | yes, `W₀+BA` | **Adapter parameterization** |
| **LoRA+** | identical to LoRA | identical | identical | **η_B = λη_A**, λ≈2¹–2⁴ | identical + λ | none | yes | **Training policy** (param groups) |
| **DoRA** | `W′ = m·(W₀+BA)/‖W₀+BA‖_c` | `m(1×k)`, `A`, `B` | frozen, **values needed in fwd** | none | 3 tensors/site | none | yes | **Weight composition** + LoRA |
| **QLoRA** | `Y = X·dequant(W^NF4) + XL₁L₂` | `A`, `B` (plain LoRA) | **NF4 + double-quant** | paged optimizer (memory strategy) | adapter unchanged; base is a different artifact | **NF4, blockwise, DQ** | adapter yes; base no (lossy) | **Composition**: quant base × LoRA × optim memory |
| **LoCon** | conv factorisation, `rank(ΔW) ≤ dim` | down/up conv kernels | frozen conv | none | 2–3 kernels/site | none | yes | **Adapter param.** (needs 4-D + conv) |
| **LoHa** | `ΔW = (B₁A₁)⊙(B₂A₂)` | `A₁,A₂,B₁,B₂` | frozen dense | none | 4 tensors/site | none | yes | **Adapter param.** (materializes ΔW) |
| **VeRA** | `h = W₀x + Λ_b BΛ_d Ax` | `d(r)`, `b(d_out)` only | frozen dense | none | vectors + **RNG seed** | none | yes | **Adapter param.** + shared frozen params |
| **LoraCheckpoint** | — | adapter tensors | identity ref only | optional sidecar | safetensors + metadata | none | — | **Storage** |
| **FullCheckpoint** | — | all params | **contains** them | **contains** state | safetensors bundle | none | — | **Storage** |
| **ShardedCheckpoint** | — | same as content | same | same | + `index.json` weight_map | none | — | **Storage layout** |
| **IncrementalCheckpoint** | — | delta vs base | base ref | same | + chain metadata | delta compression | — | **Storage layout + recovery** |
| **GgufMerge** | `W₀ + (α/r)BA` → requantize | none (consumes) | reads + rewrites | none | **GGUF out** | **re-quantization, lossy** | one-way | **Export / transformation** |
| **AdamW** | decoupled decay + bias-corrected Adam | — | — | `m`,`v` per param (2×) | 2 tensors/param + `t` | none | — | **Optimizer** |
| **Lion** | `sign(interp(g,m,β₁))` | — | — | `m` per param (1×) | 1 tensor/param | none | — | **Optimizer** |
| **Adafactor** | factored `V̂ = RC/1ᵀR` | — | — | `R`,`C` (2-D) *or* full `v` (1-D) | **rank-dependent** | none | — | **Optimizer** (rank-dependent state) |
| **AdamW8bit** | AdamW, state in 8-bit blocks | — | — | quantized `m`,`v` + per-block absmax | state + scales | **blockwise dynamic 8-bit** | — | **Optimizer state codec** over AdamW |

## 6. Research matrix II — cost and feasibility

| Skill | Complexity | Memory impact | Dependencies (on gltrain work) | M2 feasibility | Recommended abstraction |
|---|---|---:|---|---|---|
| **LoRA** | Low | +2·r·(d+k) fp32 | `matmul`,`add`,`mul_scalar` (all exist) + RNG | ✅ **FULL** | `Adapter` impl |
| **LoRA+** | Very low | none | AdamW **parameter groups** | Stub; ~1 day once groups exist | `LrPolicy` on the optimizer, **not** an adapter |
| **DoRA** | Medium | +k, plus base values live in fwd | `col_norm`, `div`, detach-in-graph | Stub | `WeightComposition` + LoRA adapter |
| **QLoRA** | High | base ÷ ~4 | NF4 codec, dequant kernel, quantized base source | Stub | **Composition**, no new adapter |
| **LoCon** | High | ≈LoRA per site | **4-D tensors + conv fwd/bwd** — absent | Stub (blocked at tensor layer) | `Adapter` impl, gated on tensor work |
| **LoHa** | Medium | **+p·q materialized ΔW** | `hadamard` (= `mul`, exists) | Stub | `Adapter` impl |
| **VeRA** | Medium | +(r + d_out) only | seeded RNG + **shared param ownership** | Stub | `Adapter` impl + shared-param registry |
| **LoraCheckpoint** | Low | — | safetensors **writer** | ✅ **FULL** | `CheckpointStore` impl |
| **FullCheckpoint** | Low–Med | 2–3× params on disk | needs a real `Module` tree | Stub | `CheckpointStore` impl |
| **ShardedCheckpoint** | Medium | bounded RSS | index writer/reader | Stub | `CheckpointStore` impl (layout) |
| **IncrementalCheckpoint** | High | **saves ~0 for dense LoRA** (§3.3) | delta+compression | Stub, **lowest value** | `CheckpointStore` impl (layout) |
| **GgufMerge** | High | — | GGUF **writer** (absent), requantization | Stub | `Exporter` — **separate trait** |
| **AdamW** | Low | 2× params | `div`,`sqrt`,`add_scalar` on `Backend`; KL-006 fix | ✅ **FULL** | `Optimizer` impl |
| **Lion** | Low | 1× params | `sign` | Stub; cheapest of the three | `Optimizer` impl |
| **Adafactor** | Medium | O(n+m) | row/col reductions; **rank-dependent state** | Stub | `Optimizer` impl |
| **AdamW8bit** | High | 0.25× state | blockwise codec + dynamic tree type | Stub | **`OptimizerStateCodec`** over AdamW, not a new optimizer |

---

## 7. The critical architecture questions, answered

### A — Are all LoRA variants the same abstraction? **No.**

The inheritance tree in the task brief fails on four independent counts, each
traceable to a specific paper:

1. **LoRA+ has no structure of its own** (§2.2). Modelling it as a sibling of
   LoRA creates a type whose fields are identical to its parent's and whose only
   difference lives in the optimizer.
2. **DoRA is not an additive delta** (§2.3). `m·(W₀+BA)/‖W₀+BA‖_c` cannot be
   written as `base_out + adapter_out`; it needs the base weight's *values*.
   A tree rooted at "adapter that produces a delta" cannot hold it.
3. **QLoRA's adapter is literally LoRA** (§2.4). Its novelty is entirely in base
   storage and optimizer paging. A `QLoRA` adapter type would duplicate `LoRA`
   and put quantization in the wrong layer.
4. **VeRA's parameters are shared across layers** (§2.7). Per-layer parameter
   ownership — which a flat adapter tree assumes — is wrong for it.

**Adopted: the compositional model**, with four orthogonal axes:

```
TrainableSite
├── BaseWeightSource     dense fp32 │ NF4-quantized │ GGUF-quantized   ← QLoRA lives here
├── AdapterParam         LoRA │ LoHa │ VeRA │ LoCon                    ← the ΔW parameterization
├── WeightComposition    Additive │ MagnitudeDirection                 ← DoRA lives here
└── (training policy)    uniform LR │ per-group LR                     ← LoRA+ lives here, on the optimizer
```

Every researched variant is a point in this product: LoRA = (dense, LoRA,
Additive, uniform). DoRA = (dense, LoRA, **MagnitudeDirection**, uniform).
QLoRA = (**NF4**, LoRA, Additive, uniform). LoRA+ = (dense, LoRA, Additive,
**per-group**). LoHa/VeRA/LoCon swap only the second axis. Nothing is left over,
and no axis is a subtype of another.

### B — Where does LoRA+ belong? **The optimizer, via parameter groups.**

The paper changes exactly one thing: `η_B = λ·η_A`. It changes no shape, no
init, no forward, no serialized tensor. Putting it in the adapter would mean an
adapter type that must reach into the optimizer to have any effect — inverted
ownership, and it would have to be re-done for every adapter that has an
"A-like" and "B-like" half (LoHa has four).

**Consequence for M2:** `AdamW` gets **parameter groups** now, with a per-group
`lr` override. That is one struct and a loop, it is how every reference
optimizer is built anyway, and it is what makes LoRA+ a config change later
rather than a rewrite. It is not speculative: M2 needs a parameter collection
regardless, and grouping is the shape that collection takes.

### C — Where does QLoRA belong? **A composition of three, none of them an adapter.**

From the forward equation: `Y = X·doubleDequant(c₁,c₂,W^NF4) + X L₁L₂`.

| QLoRA component | Correct home |
|---|---|
| NF4 + double quantization | `BaseWeightSource` — a quantized weight provider |
| `+ X L₁L₂` | the existing **LoRA** adapter, unchanged |
| Paged optimizers | optimizer **memory strategy** (orthogonal; also applies to AdamW alone) |
| BF16 compute dtype | a **precision policy** on the backend |

Modelling QLoRA as an adapter variant would place a quantization format inside
an adapter type, and would make "QLoRA + DoRA" (a real, published combination —
QDoRA) inexpressible. The composition model gets it for free.

### D — Are checkpoints really one thing? **No — two traits, not five.**

Splitting by what the operation actually *is*:

| | Storage (`CheckpointStore`) | Export (`Exporter`) |
|---|---|---|
| Members | `LoraCheckpoint`, `FullCheckpoint`, `ShardedCheckpoint`, `IncrementalCheckpoint` | `GgufMerge` |
| Round-trips | **yes**, bit-exact | **no**, lossy by design |
| Has `load()` | yes | **meaningless** |
| Purpose | resume training | produce a deployment artifact |
| Consumer | gltrain itself | glproc / glictus-caliburni / llama.cpp |

`GgufMerge` reads an adapter checkpoint plus a base model, computes
`W₀ + (α/r)BA`, and **requantizes** to a GGUF type. That is one-way and lossy
(§3.1: GGUF's alignment and quantized block types are a deployment format, not a
training one). Forcing it behind a `Checkpoint` trait means giving it a `load()`
that can never be implemented — the exact "silent fallback / meaningless method"
shape the task forbids.

The other four *do* share one interface, because they are all layouts over the
same logical content (a named tensor bundle + typed metadata). Sharding is a
file-layout decision (§3.2 — the index is a pure name→file map). Incremental adds
a base-checkpoint dependency and a reconstruction step, so it also needs
`resolve_chain()`, but that is one extra method on the same trait, not a new one.

**Recovery** is not a separate trait either: it is `validate()` returning the
caliburni-style `ValidationResult { errors, warnings }`.

### E — Should optimizer state be in the checkpoint? **Separate segments under one manifest.**

Six state categories, and they have genuinely different lifetimes and consumers:

| Segment | Needed to resume? | Needed to deploy? | Size |
|---|---|---|---|
| Model (base) weights | no (referenced by identity) | yes | 1× |
| **Adapter state** | **yes** | **yes** | small |
| **Optimizer state** | **yes** | **no** | 1–2× adapter |
| Scheduler state | yes | no | tiny |
| Training progress (step, epoch, RNG) | yes | no | tiny |
| Metadata / config | yes | yes | tiny |

The split follows from the two columns disagreeing. gltrain's own prior art
already lands here: `adamw_state_path()` writes a **sidecar**
`{stem}_adamw.safetensors` beside the weight checkpoint. That is the right
instinct, and it generalises.

**Adopted:** a checkpoint is a **directory** containing `manifest.json` plus one
file per segment. `save_adapter_only()` writes two files; `save_full()` writes
all of them. Loading a segment is independent, so resume reads everything and
export reads only the adapter — without either path parsing bytes it does not
need. `GLTRAIN_PLAN.md` Q8's "no optimizer state in v1" then becomes a *default*
(`save_optimizer_state: false`), not a format limitation.

**F2 makes this mandatory, not stylistic:** optimizer state is keyed by
`TensorId`, which is process-global and non-persistable. Serializing it means
re-keying by **name**, which only works if names are the checkpoint's primary key
throughout.

---

## 8. Proposed architecture

### 8.1 Module layout

```
gltrain/src/
    nn/              Gwiskadur   — Parameter, Module, Linear, adapters
        param.rs         named, tape-aware parameter
        module.rs        Module trait, parameter collection
        linear.rs        Linear
        adapter/
            mod.rs       Adapter trait + AdapterRegistry
            lora.rs      FULL
            dora.rs      loha.rs  vera.rs  locon.rs   (stubs)
    optim/           Gwellaer    — optimizers
        mod.rs           Optimizer trait, ParamGroup, OptimizerRegistry
        adamw.rs         FULL
        lion.rs  adafactor.rs  adamw8bit.rs           (stubs)
    checkpoint/      Pik         — persistence
        mod.rs           CheckpointStore + Exporter traits, registries
        manifest.rs      versioned manifest + ValidationResult
        safetensors.rs   the missing WRITER
        lora_ckpt.rs     FULL
        full.rs  sharded.rs  incremental.rs           (stubs)
        export_gguf.rs   GgufMerge                     (stub, Exporter)
    rng.rs           deterministic xorshift64* + Box-Muller
```

Every new file opens with its Breton sub-system header, per gltrain-naming.

### 8.2 The three traits

Traits take **no prefix** (naming rule 2). Sketches, not final code:

```rust
/// One adapter parameterization. Produces the trainable delta for a site.
pub trait Adapter<B: Backend> {
    fn forward(&self, x: &Tensor<B>, base: &BaseWeight<B>) -> Result<Tensor<B>>;
    fn parameters(&self) -> Vec<&Parameter<B>>;
    fn capability(&self) -> &AdapterCapability;   // ← metadata, see §8.4
    fn merge_into(&self, base: &mut Tensor<B>) -> Result<()>;
}

pub trait Optimizer<B: Backend> {
    fn step(&mut self, tape: &mut Tape) -> Result<()>;
    fn zero_grad(&mut self, tape: &mut Tape);
    fn groups(&self) -> &[ParamGroup];            // ← LoRA+ hooks here
    fn state_tensors(&self) -> Vec<(String, &[f32], Vec<usize>)>;
    fn load_state(&mut self, named: &NamedTensors) -> Result<()>;
}

pub trait CheckpointStore { /* save, load, validate, resolve_chain */ }
pub trait Exporter        { /* export only — no load(), by design (§7-D) */ }
```

`Adapter::forward` takes the **base weight**, not the base *output*. That single
signature choice is what makes DoRA expressible (§2.3) without changing the trait
later — and it costs additive adapters nothing.

### 8.3 Resolving KL-006 (mandatory, §1.2-F3)

Adopting `KNOWN_ISSUES.md`'s **option (a)**: `Optimizer::step()` takes
`&mut Tape` and returns `GlTrainError::InvalidOp` if the tape still holds nodes,
requiring `tape.clear()` first. Rationale: the graph is dead once the step is
taken, so the discipline is correct anyway; it mirrors the KL-005 guard exactly;
and it is the option `KNOWN_ISSUES.md` itself recommends. It ships **with** the
in-place update, in the same commit, with a regression test — as KL-006 requires.

Also from §1.2-F4: `step()` reads gradients via `VLGradStore::take()` and writes
weights through a new **off-tape** mutation path. Optimizer arithmetic never
touches tape-recording `Tensor` ops.

### 8.4 Registry — adopting the caliburni pattern

`glictus-caliburni/src/plugin.rs` already solved this, and its decisions are the
ones the task asks for:

- `register()` **refuses on collision** rather than overwriting ("two plugins
  disagreeing about one layer type is a wiring bug").
- `resolve()` → `Option`, `require()` → `Result`. No silent fallback anywhere.
- `with_builtins()` preloads the shipped set.
- Capability metadata is a trait method returning a struct, and
  `supports_dtype()` returning `false` "is a normal outcome" — not an error.

Applied here, `AdapterCapability` carries: `id`, `status:
{Full, Stub{reason, tracking}}`, `trainable_params_formula`, `mergeable`,
`requires_base_values`, `materializes_delta`, `shares_params_across_layers`.
Every field is a fact this research established, so the metadata is a
*machine-readable summary of §5* rather than invented ceremony.

A stub's `forward()` returns `GlTrainError::Unsupported { skill, reason,
milestone }` — never a fallback to LoRA.

### 8.5 Serialization

Write a safetensors **writer** in-tree (~100 lines): the header JSON is a flat,
fully-specified structure (§3.1), so it needs no serde — and `glcore`'s existing
**reader** then serves as an independent round-trip oracle, which a third-party
crate could not. Zero new dependencies, and it satisfies Inference-First Rule 1
(reference implementation first).

---

## 9. Architecture risks

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| **R1** | **KL-006**: in-place update staleness — silent wrong gradients, plausible loss curve | **High** | §8.3 guard, lands with the update, regression test |
| **R2** | `TensorId` not persistable (F2) | High | Names are the primary key everywhere; optimizer state re-keyed on load; test that IDs differ across runs |
| **R3** | `matmul` is **2-D only**; no batch dim | High for M3 | M2 targets a micro-model; batched matmul is M3 scope, recorded not hidden |
| **R4** | Optimizer ops on tracked params pollute the tape (F4) | High | Optimizer runs off-tape on raw buffers |
| **R5** | `Backend` lacks `div`/`sqrt`/`add_scalar` | Medium | Add to the trait — 2 impls, both in-crate |
| **R6** | LoHa materializes a full `p×q` ΔW (§2.6) | Medium | Recorded in capability metadata now, so no one assumes low-rank ⇒ cheap |
| **R7** | VeRA's cross-layer shared params break per-layer ownership (§2.7) | Medium | `Module::parameters()` must dedupe by identity from day one |
| **R8** | Incremental checkpointing saves ~nothing for dense LoRA (§3.3) | Low | Deprioritised to last in §10 with the evidence |
| **R9** | **`OP` prefix is not approved**; no prefix category exists for checkpoints | **Blocking** | Needs JinXSuper's sign-off — see §11 |
| **R10** | 13 stubs vs Inference-First **Rule 5** ("no speculative complexity") | **Design tension** | See below |

**On R10, explicitly.** Rule 5 forbids "abstraction for engines that don't exist
yet". A literal reading forbids most of this task. The reconciliation is that
Rule 4 makes *measured limits* a deliverable, and §5/§6 are exactly that: each
stub records a researched fact (DoRA needs base values; LoHa materializes ΔW;
VeRA shares parameters; LoCon is blocked on 4-D tensors) that would otherwise be
rediscovered the expensive way. **The stubs are justified only to the extent the
research pins down a real constraint.** Where it does not, the honest output is a
documented gap, not a type. That is why `IncrementalCheckpoint` is last in §10
and why LoCon's stub says "blocked at the tensor layer" rather than pretending
the adapter abstraction is what gates it.

---

## 10. M3+ implementation order, derived from dependencies

Ordered by what unblocks what, not by perceived importance:

| # | Item | Unblocked by | Why here |
|---|---|---|---|
| 1 | **LoRA+** | AdamW param groups (M2) | Config-only once groups exist. Cheapest real capability. |
| 2 | **Lion** | `sign` op | Smallest optimizer; validates the `Optimizer` trait against a *non-Adam* shape early, which is the point. |
| 3 | **FullCheckpoint** | `Module` tree + segments (M2) | Same machinery as LoraCheckpoint over more tensors. |
| 4 | **DoRA** | `col_norm`, `div`, graph-detach | First real test of `WeightComposition`; highest quality/effort ratio of the adapters. |
| 5 | **LoHa** | `mul` (exists) | Pure `AdapterParam` swap; validates that axis. |
| 6 | **VeRA** | seeded RNG + shared-param dedupe (R7) | Forces the parameter-ownership question; do it before the model tree hardens. |
| 7 | **Adafactor** | row/col reductions | Rank-dependent state — the case that proves the optimizer-state abstraction. |
| 8 | **ShardedCheckpoint** | index writer | Needed once models exceed one file; pure layout. |
| 9 | **AdamW8bit** | blockwise codec | `OptimizerStateCodec` over a working AdamW; needs 1–7 stable. |
| 10 | **QLoRA** | NF4 codec + quantized `BaseWeightSource` + dequant kernel | Largest surface; wants glproc quant work (see the quant/bandwidth-wall notes). |
| 11 | **GgufMerge** | **GGUF writer** (absent repo-wide) + requantization | Blocked on a whole new writer, not on training. |
| 12 | **LoCon** | **4-D tensors + conv fwd/bwd** | Blocked at the tensor layer; unrelated to adapters. |
| 13 | **IncrementalCheckpoint** | delta + compression | §3.3: near-zero benefit for dense LoRA. Last on evidence. |

---

## 11. Open decisions that need JinXSuper before implementation

1. **`OP` prefix approval (blocking for `optim/`).** `gltrain-naming/SKILL.md`:
   "Before M2 writes a line of optimizer code, `OP` gets proposed … and added to
   the table." The case to make: an optimizer carries **mutable state across
   steps**, whereas `AB` (Algorithm Block) is pure — which is the exact objection
   the skill says must be answered.
2. **No prefix category exists for checkpoints.** Options: extend `VL` (they are
   largely plain data), reuse `RS` (they own file handles and have a `Drop`
   story), or propose `CK`. Needs the same sign-off process.
3. **Confirm M2 targets `gltrain/`, not `gltrain/`** (§0.1).

---

## 12. Sources

**LoRA family** — LoRA: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) ·
LoRA+: [arXiv:2402.12354](https://arxiv.org/abs/2402.12354) ·
DoRA: [arXiv:2402.09353](https://arxiv.org/abs/2402.09353) ·
QLoRA: [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) ·
LyCORIS/LoCon/LoHa: [arXiv:2309.14859](https://arxiv.org/abs/2309.14859) and
[LyCORIS Algo-Details](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md) ·
VeRA: [arXiv:2310.11454](https://arxiv.org/abs/2310.11454) ·
reference impl: [huggingface/peft `lora/layer.py`](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)

**Optimizers** — AdamW: [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) and
[torch.optim.AdamW](https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py) ·
Lion: [arXiv:2302.06675](https://arxiv.org/abs/2302.06675) ·
Adafactor: [arXiv:1804.04235](https://arxiv.org/abs/1804.04235) ·
8-bit optimizers: [arXiv:2110.02861](https://arxiv.org/abs/2110.02861)

**Formats** — safetensors: [huggingface/safetensors](https://github.com/huggingface/safetensors) ·
GGUF: [ggml docs/gguf.md](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) ·
sharded checkpoints: [HF big_models](https://huggingface.co/docs/transformers/main/en/big_models) ·
Check-N-Run: [USENIX NSDI '22](https://www.usenix.org/conference/nsdi22/presentation/eisenman)

**In-repo** — `gltrain/GLTRAIN_PLAN.md`, `gltrain/KNOWN_ISSUES.md`,
`glcore/src/format/safetensors.rs`, `glictus-caliburni/src/plugin.rs`,
`glictus-caliburni/src/manifest/validator.rs`, `gltrain/src/train/lora.rs`,
`gltrain/src/train/adamw_state.rs`, `gl-agent-skills/**`
