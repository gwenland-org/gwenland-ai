# GATE — GwenLand Mapping 🔥
## GwenLand-specific: crate ownership, gaps, integration points, scope

> Not from the paper. This file exists because the paper's own §11
> (Implementation) describes an implementation that does not match this
> repository — see "What Already Exists" below. Read this file before
> starting any GATE implementation wave.

---

## 1. Concept → Crate Mapping

Every type the paper defines, and where it lives (or will live) in
GwenLand:

| Paper concept | Rust name | Crate / file | Status |
|---|---|---|---|
| Tensor type `𝒯` | `DType` | `glcore::tensor` | **exists**, reused as-is |
| Execution plan `P` | `ExecutionPlan` | `glcore::gate::plan` | net-new (Phase 3) |
| Backend `β ∈ ℬ` | `BackendKind` | `glcore::gate::plan` | net-new — closed enum of GwenLand's 4 engines, not an open trait-object set like the paper's `ℬ` |
| Memory layout `𝓛` | `MemoryLayout` | `glcore::gate::plan` | net-new, stub-only |
| Tensor op `𝒪` member | `TensorOp` | `glcore::gate::plan` | net-new **marker stub only** — see §3 |
| Computation graph `𝒢` | `TensorGraph` | — | **does not exist**, not created this sprint — see §3 |
| Constraint `C: 𝒫→{0,1}` | `Constraint` trait | `glcore::gate::constraint` | net-new |
| Validation outcome | `ValidationResult` | `glcore::gate::constraint` | net-new |
| Validator `V = ∏Cᵢ` | `Validator` | `glcore::gate::validator` | net-new |
| Metric vector `m ∈ ℝ⁵` | `MetricVector` | `glcore::gate::metrics` | net-new |
| Weight vector `w ∈ Δ⁴` | `WeightVector` | `glcore::gate::metrics` | net-new |
| Normalization strategy | `NormalizationStrategy` | `glcore::gate::metrics` | net-new |
| Execution policy `π` | `ExecutionPolicy` | `glcore::gate::policy` | net-new |
| Plan generator | `Planner` | `glcore::gate::planner` | net-new, protocol-only (see §3) |
| Dispatcher | `Dispatcher` | `glcore::gate::dispatcher` | net-new, protocol-only |
| Error (`ConstraintViolation` etc.) | `GateError` | `glcore::gate::error` | net-new — **not merged into `GlError`**, see §4 |
| `Backend` trait (paper §11) | — | — | **does not exist** — see §2 |
| `ShapeConstraint`, `MemoryConstraint`, etc. (concrete constraints) | — | `glproc::gate_constraints::*` (future) | **out of scope this sprint** — see §3 and `GATE-impl-plan.md` Wave G1 |

---

## 2. What Already Exists

Short list, because it *is* short:

- **`glcore::tensor::DType`** — the tensor-type domain `𝒯`. Reused directly;
  GATE's shape/layout constraints will read it, nothing about it needs to
  change.
- **`glcore::engine_trait::GlEngine`** — GwenLand's actual backend
  abstraction today. It is *not* the paper's `Backend` trait (see §4,
  first gap) but it is the trait every `BackendKind` variant ultimately
  corresponds to at runtime.
- **`glcore::engine_trait::EngineSpec`** — the (very coarse) existing
  capability report (`{name, backend, available}`). The seed of
  `BackendCapabilityConstraint`, not a replacement for it.
- **`glcore::error::GlError`** — the existing error type for everything
  inference-related. `GateError` is deliberately separate (§4).
- **Advisory checks that do today, ad hoc, what a constraint would do
  formally** — cited with file paths in `GATE-constraints.md`:
  `glbench/src/validation/memory.rs` (memory), `glbench/src/validation/numerical.rs`
  + `parity.rs` (numerical error, via oracle token comparison), the ARTX05
  KV-cache clamp in `glictus-caliburni`.

Everything else in the paper's §11 implementation description
(`Constraint` trait, `Backend` trait with `supports_op`/`supports_dtype`,
`TensorOp`, `ExecutionPlan`, `MetricVector`, `WeightVector`, `GateError`)
**is not in this codebase.** Phase 3's boilerplate is the first time any of
it exists in Rust source, anywhere in this repo.

---

## 3. What Must Be Created

Net-new, in dependency order:

1. `glcore::gate::error::GateError` — no dependencies.
2. `glcore::gate::plan::{ExecutionPlan, BackendKind, OpId, MemoryLayout}`,
   plus a **marker-only `TensorOp`** — a zero-field or minimal struct just
   sufficient for `ExecutionPlan::ordering: Vec<TensorOp>` to compile. This
   is *not* the paper's `𝒪` (a partial function between tensor types) — it
   has no shape-checking behavior yet. That behavior needs a real
   `TensorGraph`, which is explicitly **not built this sprint** (see
   Gap 1 below).
3. `glcore::gate::constraint::{Constraint, ValidationResult}` — depends on
   `plan`.
4. `glcore::gate::validator::Validator` — depends on `constraint`.
5. `glcore::gate::metrics::{MetricVector, WeightVector, NormalizationStrategy}` —
   depends on `plan` (for `normalize(plans: &[ExecutionPlan], ...)`).
6. `glcore::gate::policy::ExecutionPolicy` — depends on `metrics`.
7. `glcore::gate::planner::Planner` — depends on `validator`, `metrics`,
   `plan`. `generate_candidates` and `select_best` are `todo!()` — there is
   no `TensorGraph` to generate candidates *from* yet, so this cannot be
   real even in principle until Gap 1 is resolved.
8. `glcore::gate::dispatcher::Dispatcher` — depends on `plan`, `error`.
   `dispatch` is `todo!()` — there is no bridge from `ExecutionPlan` to an
   actual `Box<dyn GlEngine>` call yet (see Integration Points, §5).

**Explicitly not created this sprint:** `TensorGraph`, any concrete
`Constraint` impl (`ShapeConstraint` et al.), any real cost estimator, any
code that calls a `GlEngine` from inside `glcore::gate`. All of §3's types
compile and their tests pass; none of them do anything yet.

---

## 4. Gaps and Decisions

**Gap 1 — no `TensorGraph`/op-DAG exists anywhere in GwenLand.** Every
engine (`glproc`, `glcuda`) executes a *fixed, hand-written layer walk*
(`Runner::generate`, `GpuModel::generate`) — there is no intermediate
representation of "the graph of ops this model requires" that a plan
generator could enumerate strategies over. This is the largest structural
gap between the paper and this codebase, bigger than any single missing
type. **Decision:** do not fabricate a `TensorGraph` this sprint just to
make `Planner::generate_candidates` look complete — a stub signature that
takes `&TensorGraph` and returns `todo!()` documents the target shape
honestly; inventing a graph representation without a real caller would be
speculative design the boilerplate rules explicitly forbid ("zero
inference logic," "no design for hypothetical future requirements").
Building `TensorGraph` for real is Wave G2+ territory (`GATE-impl-plan.md`)
and is its own decision about how much of glproc's/glcuda's fixed layer
walk it should replace — deliberately not decided here.

**Gap 2 — `GateError` is not merged into `GlError`.** The paper's
`GlEngine`-adjacent error path and its `GateError` are one thing in the
paper's hypothetical glcore; here they're kept as two separate enums.
**Decision:** keep them separate. `gl-agent-skills/architecture-skills/glcore-rules.md`
Rule 6 treats changes to shared types (`GlError` explicitly named) as
"cross-backend events" requiring every implementor to be checked in the
same PR — folding `GateError`'s variants into `GlError` today would force
a cross-backend change for a feature (GATE) that nothing yet calls. A
future wave (Wave G4, dispatcher integration) will need an explicit
`From<GateError> for GlError` or similar bridge at the one point they
actually meet — deferred, not forgotten.

**Gap 3 — `BackendKind` is a closed enum, not an open capability
registry.** The paper's `ℬ` is an open set ("the set of hardware
backends"); GwenLand's `BackendKind` is a 4-variant enum
(`Glproc/Glcuda/Glvulkan/Glmetal`) matching the fixed set of engines this
project ever builds (per `backend-independence.md`, engines are not
pluggable third-party things). **Decision:** closed enum, deliberately —
matches how `EngineSpec.backend: &'static str` already works today, and
GwenLand has never had (or wanted) a dynamic backend registry.

**Gap 4 — the paper's default validation order is caller discipline, not
enforced.** `Validator::register` doesn't check or reorder what's
registered (see `GATE-constraints.md`). **Decision:** match the paper's
own reference implementations, which also rely on registration order
rather than a sorting step — revisit only if a real ordering bug is
observed (per `rejected-optimizations.md`'s general philosophy: don't
pre-optimize a problem that hasn't been measured).

**Gap 5 — `NormalizationStrategy` defaults to `MaxNorm`, matching the
paper's reference implementations, not GwenLand's own preference.** No
GwenLand-specific reason exists yet to prefer threshold-relative
normalization; `MaxNorm`'s conservative bias is a reasonable default given
this project's own numerical-correctness history (the Q6_K bug). Revisit
once a real cost estimator exists and can be measured against both
strategies.

---

## 5. Integration Points

None of these are wired this sprint — `glcore::gate` compiles standalone
with zero external callers. This section is a map of *where* future waves
connect it, so Wave G4+ in `GATE-impl-plan.md` has concrete anchors instead
of vague intent.

**`glproc` engine dispatch** (`glproc/src/engine.rs`, `GlprocEngine::run`) —
today calls `Runner::generate` directly with no plan selection step at
all. A GATE-integrated future would insert `Planner::select_best` between
`GlprocEngine::infer` receiving an `InferInput` and `Runner::generate`
being called — but only once `TensorGraph` (Gap 1) exists to generate
candidates from. Not before.

**`glbench`'s `engine/adapter.rs`** — `EngineAdapter::load` /
`build_engine()` is today's *entire* engine-selection mechanism: match on
an explicit `--engine` string, hard error on unknown names, no
auto-selection at all (see `README.md`'s fallback-chain section). This is
the natural landing spot for a future `--policy` flag
(`GATE-impl-plan.md` Wave G6): `build_engine()` would ask a `Planner` to
pick among available engines under the requested `ExecutionPolicy` instead
of requiring the caller to already know which engine they want. Until
then, `EngineAdapter` and `glcore::gate` do not reference each other.

**Future `glcuda`/`glvulkan` backends** — `BackendCapabilityConstraint`
(Wave G5) is where each backend's real capability surface
(`glcuda::driver::cuda_available()`, kernel-set coverage, dtype support)
would get exposed to GATE. glvulkan/glmetal being stubs today
(`available: false` unconditionally) means `BackendCapabilityConstraint`
would reject every candidate plan on those backends until real kernels
land — which is *correct* GATE behavior (Theorem 10.2), not a bug to work
around.

**`GllmEngine` (`glictus-caliburni::runtime::gllm_engine`)** — already
implements `GlEngine` and is driven by `glbench` through the same
`build_engine`-adjacent path (`EngineAdapter::load_gllm`, gated behind the
`gllm-bench` feature). It also exposes `vocab_size()` and
`score_sequence()` beyond the `GlEngine` trait — methods `glbench` calls
directly today, which is itself a small boundary leak per
`backend-independence.md` Rule 5 ("backends expose `GlEngine` and nothing
else"). Any future GATE integration touching GllmEngine inherits that
pre-existing tension; it is not something this sprint introduces or
resolves.

---

## 6. NOT in Scope (Yet)

Directly from the paper's own Future Work (§13.4), plus GwenLand-specific
exclusions:

- **Distributed / multi-device execution.** Paper explicitly single-device
  only; GwenLand is explicitly single-GPU (`ArchGLML_X2.md`'s invariants
  table). `glictus-caliburni/src/runtime/distributed.rs` exists in the repo
  for the GLLM format's own reasons but is unrelated to GATE.
- **Native TPU backend.** The paper's TPU datapoint runs through XLA/JAX,
  not a native backend; GwenLand has no TPU backend at all, native or
  otherwise.
- **ML-guided plan generation.** Paper's own future work, not attempted.
- **Runtime constraint adaptation under changing system load.** Future
  work in the paper; no mechanism for it exists or is planned this sprint.
- **Formal verification of constraint implementations themselves.** Noted
  by the paper as future work; today's constraints (once Wave G1 writes
  them) are ordinary Rust, tested the ordinary way.
- **Threshold-relative cost normalization as the default.** Exists as a
  documented alternative (`NormalizationStrategy::ThresholdRelative`) but
  `MaxNorm` is default — see Gap 5.
- **Any concrete `Constraint` implementation.** `ShapeConstraint`,
  `MemoryConstraint`, etc. are Wave G1 (`glproc::gate_constraints::*`), not
  this sprint.
- **Any change to `glproc`/`glcuda`/`glvulkan`/`glmetal` production code.**
  This sprint touches `glcore` only.
