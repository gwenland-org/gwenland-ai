# GATE — Constraint Engine
## Paper §6

> Read [`GATE-concepts.md`](GATE-concepts.md) first for the `Constraint` /
> `Validator` / `ValidationResult` types this file assumes.

Each candidate plan passes through a sequential chain of constraint checks
with early-exit: the first violation short-circuits the chain. The seven
default constraint types below are ordered as GATE's own default validation
order (§6.3) — cheapest and most-rejective first.

---

## The Seven Constraint Types

### 1. Shape
**Checks:** for each operation `o` with inputs `t₁,...,tₙ` and output
`t_out`, the shape-compatibility predicate `φₒ(shape(t₁),...,shape(tₙ)) →
shape(t_out)` holds (e.g. inner dimensions of a matmul agree).
**Complexity:** `O(|V|)` per plan. **Backend-independent** (a shape
mismatch is wrong on every backend). **Rust:** `ShapeConstraint`.

*GwenLand today:* `glcore::tensor::Tensor::reshape` ([`tensor.rs:70`](../../glcore/src/tensor.rs))
already does a shape-compatibility check — it compares `numel()` and
returns `GlError::ShapeMismatch { expected, got }` on mismatch — but it
validates one reshape call, not a whole op-DAG. There is no `TensorGraph`
to walk yet (see `GATE-mapping.md`), so a real `ShapeConstraint` has
nothing to iterate over until that type exists.

### 2. Tensor Layout
**Checks:** the layout assignment `Λ: 𝒯 → {NC, CN, NCHW, NHWC, ...}` is
compatible with every consuming operation and the target backend.
**Complexity:** `O(|V|)`, static. **Backend-dependent.** **Rust:**
`TensorLayoutConstraint`.

*GwenLand today:* nothing. Every engine's `Tensor` is row-major, always
(`tensor.rs`: "Data is row-major: the last dimension is contiguous") — there
is exactly one layout in the whole codebase, so there has never been a
layout *choice* to validate. This constraint has no work to do until a
backend introduces a second layout (e.g. a tiled or blocked layout for a
GPU kernel).

### 3. Memory
**Checks:** `C_mem(P) = 1` iff `M(P) ≤ M_max(β)`, the peak of live
allocations over the schedule. **Complexity:** `O(|V|)`, static, budget is
user-configurable. **Rust:** `MemoryConstraint`.

*GwenLand today:* this one **already exists in spirit**, twice:
- [`glictus-caliburni`'s ARTX05 runtime](../../glictus-caliburni/src/runtime) clamps `max_seq_len` down
  when a full-context KV cache would exceed the machine's RAM (the "KV
  cache is the memory trap" finding — Qwen3-1.7B full-ctx needs 8.75 GiB on
  an 8 GB machine).
- [`glbench/src/validation/memory.rs`](../../glbench/src/validation/memory.rs) generalizes exactly that finding into a
  reusable check: `model_bytes + kv_cache_bytes` vs. `available_bytes`,
  flagging `Severity::Error` if the configuration wouldn't fit and
  `Severity::Warning` above an 80%-of-RAM tight-fit threshold.

Both are **post-hoc / advisory** — they run after load or as a benchmark
report, not as a pre-dispatch gate that blocks a plan from being selected.
A real `MemoryConstraint` would move this same arithmetic (already proven
correct by the ARTX05 incident) *before* dispatch instead of after.

### 4. Backend Capability
**Checks:** the capability predicate `κ: 𝒪 × ℬ → {0,1}` accepts every
operation (op-level and dtype-level support). **Complexity:** `O(|V|)`,
static. **Backend-dependent.** **Rust:** `BackendCapabilityConstraint`.

*GwenLand today:* `glcore::engine_trait::EngineSpec { available: bool }`
(returned by every `GlEngine::capabilities()`) is the entire existing
capability surface — one boolean per whole engine ("can this engine run at
all right now"), not per-operation or per-dtype. glvulkan/glmetal report
`available: false` unconditionally (stubs); glcuda's is a real driver
probe (`driver::cuda_available()`). A real `BackendCapabilityConstraint`
needs a much finer predicate than exists today — this is the single
biggest capability gap among the seven (see `GATE-mapping.md`).

### 5. Numerical Error
**Checks:** estimated cumulative error satisfies `‖ŷ − y_ref‖₂ ≤ ε`
(default `ε = 10⁻⁴`), bounded analytically (interval arithmetic over
dtype conversions, reorderings, layout transforms) since the true reference
output isn't available at validation time. **Complexity:** `O(|V|·d)`,
tolerance user-configurable. **Rust:** `NumericalErrorConstraint`.

*GwenLand today:* the closest existing analog is
[`glbench/src/validation/numerical.rs`](../../glbench/src/validation/numerical.rs)'s `compare_tokens` +
[`parity.rs`](../../glbench/src/validation/parity.rs)'s `validate_against_oracle` — greedy-decode a candidate
engine and glproc (the project's numerical oracle, per `DESIGN.md`) on the
same prompt/seed and report the matching token prefix. It differs from the
paper's constraint in two ways: it's an **exact discrete token match**, not
a continuous relative-`L₂` bound, and it's a manual/benchmark-time
comparison (`glbench validate --against`), not a pre-dispatch gate.

**Why this constraint matters here, concretely:** GwenLand already lived
through the exact failure mode Finding 3 describes (see below) — the Q6_K
dequant corruption bug (fixed 2026-07-23): `glcore`'s Q6_K dequant used the
wrong nibble order, silently corrupting `ffn_down.weight` in *every*
layer. Output was fluent, plausible-looking text — wrong, but not
obviously broken — and was only caught by comparing actual generated
output against expectation, not by any per-kernel unit test (every
individual dequant path "worked"). That is the TVM story, independently
rediscovered in this codebase.

### 6. Determinism
**Checks:** rejects plans containing non-deterministic operations when
`require_determinism` is set — atomic parallel reductions with unstable
summation order, floating-point reassociation, non-deterministic
algorithms (top-k tie-breaking, sampling). **Complexity:** `O(|V|)`,
static. **Rust:** `DeterminismConstraint`.

*GwenLand today:* **do not confuse with**
[`glbench/src/validation/deterministic.rs`](../../glbench/src/validation/deterministic.rs) — that module checks whether a
*benchmark run's methodology* was deterministic (was a seed pinned, was
warmup done, is temperature 0), which is a completely different question
from whether an *execution plan's operations* are algorithmically
deterministic. Engines do carry a `seed: Option<u64>` for reproducible
sampling (`GlprocConfig`/`GlcudaConfig`), which is adjacent but is about RNG
determinism, not about non-deterministic parallel reduction order. No
existing code inspects an op sequence for non-deterministic reduction
patterns.

### 7. Safety
**Checks:** (1) every tensor read is within allocated bounds, (2) every
write targets a valid buffer, (3) buffer lifetimes cover all uses, (4) no
aliasing creates data races. **Complexity:** `O(|V|²)` worst case
(alias/liveness analysis), `O(|V|)` in practice. **Rust:**
`SafetyConstraint`.

*GwenLand today:* Rust's ownership/borrow checker already enforces most of
this *at compile time* for safe code — which is most of the codebase. The
real risk surface is the `unsafe` blocks in `glcuda`'s FFI layer (CUDA
driver calls, raw device pointers), governed today by
[`gl-agent-skills/rust-skills/unsafe-rules.md`](../../gl-agent-skills/rust-skills/unsafe-rules.md) (manual invariant
comments, safe wrapper discipline) — a code-review convention, not a
runtime-checkable constraint. A `SafetyConstraint` in GATE's sense would
need plan-level bounds/aliasing analysis that doesn't exist as an
independent, invokable check anywhere today.

---

## Default Validation Order (§6.3)

Constraints run **cheapest-and-most-rejective first**, to maximize
early-exit savings: **shape → tensor layout → memory → backend capability
→ numerical error → determinism → safety**. Empirically (paper §9) this
ordering reduces effective validation cost by 2.5–3.5×: shape and
backend-capability checks alone reject 30–50% of candidates, so on average
only 2–3 of the 7 constraints ever run per plan.

`Validator::register` (see `GATE-concepts.md`) does not enforce this
order — it's caller discipline, same as the paper's reference
interfaces (§5.1/§5.2) register constraints in exactly this sequence.
Phase 3's boilerplate does not reorder or validate registration order; a
future wave may want a lint or a fixed-order constructor if this proves
easy to get wrong in practice.

---

## Composability (§6.2)

Adding a constraint `C_{k+1}` yields the validator
`V'(P) = V(P) · C_{k+1}(P)` with **zero modification of existing
constraints** — open for extension, closed for modification. Each
constraint is an independent object behind one interface (`Constraint` in
Rust); Theorem 10.1 (soundness, see `GATE-algorithm.md`) holds for any `k`.
Concretely: `Validator::register(Box::new(MyNewConstraint))` is the entire
integration surface — no existing `Constraint` impl, and no code in
`Validator::validate`, needs to change.

This is also why `glcore/src/gate/` stays constraint-*protocol*-only
(the `Constraint` trait, `Validator`, `ValidationResult`) while every
concrete constraint (`ShapeConstraint`, `MemoryConstraint`, ...) belongs to
the backend crate that can actually evaluate it — see
[`gl-agent-skills/architecture-skills/glcore-rules.md`](../../gl-agent-skills/architecture-skills/glcore-rules.md): glcore is
"shared foundation + orchestration, zero inference compute," and a real
`ShapeConstraint` needs to inspect actual tensor shapes from an actual
graph, which is compute-adjacent, backend-specific work.

---

## Finding 3: TVM 0.25.0's Silent Miscompilation

The paper's motivating case for `NumericalErrorConstraint` (§12.2): an
official TVM 0.25.0 win_amd64 wheel compiled a ResNet-50 into a module
whose output diverged from four other independent implementations by
relative `L₂` of 1.24–4.03 — a **different top-1 class** — while every
per-operator micro-test passed at ~10⁻⁷. The isolated cause: any graph
containing more than one `BatchNormalization` node was lowered incorrectly,
while the imported IR remained correct. Under GATE's protocol, such a plan
is rejected by the numerical-error constraint *before* dispatch; under
TVM's own pipeline it executed silently.

GwenLand's own Q6_K dequant corruption bug (cited under Constraint 5 above)
is the same failure shape: correct-looking per-component tests, wrong
composed output, caught only by checking actual numbers. It is the concrete
reason this constraint is not optional decoration in this codebase's
context — it already would have caught a real, shipped bug.
