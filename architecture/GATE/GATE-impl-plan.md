# GATE — Implementation Wave Plan
## Turning `glcore::gate`'s boilerplate into a working constraint engine

> Read [`GATE-mapping.md`](GATE-mapping.md) first — every wave below exists
> to close one of its five gaps or wire one of its integration points.
> Per [`gl-agent-skills/before-coding/wave-confirmation-gates.md`](../../gl-agent-skills/before-coding/wave-confirmation-gates.md):
> **STOP after every wave.** Each wave below is its own branch/PR (per
> [`branch-strategy.md`](../../gl-agent-skills/before-coding/branch-strategy.md)'s "one PR = one topic") and needs an
> explicit go-ahead before starting — this document authorizes the *shape*
> of the work, not permission to run through all six waves unattended.

---

## Sequencing note (read before starting Wave G1)

The wave numbering below is fixed by the original sprint brief, but it
creates one real dependency tension worth naming rather than silently
resolving: **Wave G1 (constraints) is numbered before Wave G2 (which is
where a real `TensorGraph` gets built — see `GATE-mapping.md` Gap 1)**.
`ShapeConstraint` genuinely needs per-op shape information to do more than
a trivial check, and that information doesn't exist until G2 lands.

This is not a reason to renumber. It's a reason to scope G1 honestly: its
highest-value, fully-implementable-today deliverables are
`BackendCapabilityConstraint` and `MemoryConstraint` (both work at a
whole-plan/whole-engine granularity using data that already exists —
`EngineSpec`, `MemoryTelemetry`). `ShapeConstraint` ships in G1 too, but as
a deliberately coarse check (e.g., input/output tensor rank and dtype
sanity from GGUF config), documented as provisional, and gets upgraded
when G2's real graph exists. Do not build a throwaway `TensorGraph` inside
G1 just to make `ShapeConstraint` feel complete — that duplicates G2's job
and risks two incompatible graph representations.

---

## Wave G1 — Constraint Implementations in glproc

**Scope:** `ShapeConstraint`, `BackendCapabilityConstraint`,
`MemoryConstraint` as real `glcore::gate::constraint::Constraint`
implementations, living in `glproc` (per
[`glcore-rules.md`](../../gl-agent-skills/architecture-skills/glcore-rules.md): concrete constraints are
compute-adjacent, backend-specific — they do not belong in `glcore`).

- `MemoryConstraint`: port the arithmetic already proven correct in
  [`glbench/src/validation/memory.rs`](../../glbench/src/validation/memory.rs) (`model_bytes + kv_cache_bytes` vs.
  available RAM) from an advisory post-hoc check into a real pre-dispatch
  `Constraint::validate`.
- `BackendCapabilityConstraint`: wraps `GlEngine::capabilities().available`
  today (coarse, whole-engine); does not yet need per-op/per-dtype
  granularity since glproc supports everything it loads.
- `ShapeConstraint`: coarse form only, per the sequencing note above.

**Files touched:** new `glproc/src/gate_constraints/{mod,shape,backend_capability,memory}.rs`.
No changes to `glcore::gate` (constraints are pure consumers of its
`Constraint` trait).

**Tests required:** one accept + one reject case per constraint (e.g.
`MemoryConstraint` rejecting a plan whose backend budget is exceeded, using
a real `MemoryTelemetry`-shaped fixture); one test wiring all three into a
`Validator` and confirming early-exit order matches
[`GATE-constraints.md`](GATE-constraints.md)'s default ordering.

**Confirmation gate:** report which constraints are real vs. coarse-stub,
with the reasoning above restated against actual code (not just this
plan), plus `cargo test -p glproc` and `cargo clippy -p glproc -- -D warnings`
output.

---

## Wave G2 — PlanGenerator for the glproc Backend

**Scope:** two deliverables, not one:
1. Upgrade `glcore::gate::plan::TensorGraph` from its Phase 3 marker stub
   (`{ ops: Vec<TensorOp> }`, no edges) to a real minimal graph — built
   from glproc's already-loaded `GlprocModel` structure, not invented
   fresh.
2. A real candidate generator for glproc. **Concrete anchor, not
   speculative:** glproc already makes multiple real kernel-path choices
   per weight matrix today — see `GlprocEngine::backend_telemetry`'s
   `kernel_of` closure in [`glproc/src/engine.rs`](../../glproc/src/engine.rs) (F32 dense vs. native
   Q4_K integer-dot vs. Q8_0-bridge, chosen today by a fixed gate, not a
   validated+costed choice). A real `generate_candidates` for glproc can
   start by treating *these already-existing choices* as candidate plans,
   rather than inventing new strategies — turning today's single hardcoded
   pick into an enumerable, GATE-validated one.

**Open design question this wave must resolve** (deliberately not decided
here, per "no speculative design ahead of need"): does `glcore::gate::Planner`
gain a pluggable generator (a trait, mirroring how `Validator` takes
`Box<dyn Constraint>`), or does `glproc` own its own generator that feeds
`Planner::select_best` directly? Whichever is chosen, `Planner`'s public
API (`GATE-concepts.md`) should not need to special-case glproc by name —
consistent with backend independence.

**Files touched:** `glcore::gate::plan::TensorGraph` (real fields, still
no compute); new `glproc/src/gate_planner.rs`; possibly
`glcore::gate::planner` (if the generator becomes pluggable).

**Tests required:** `generate_candidates` on a real loaded model produces
≥2 distinct candidate plans reflecting real kernel-path alternatives;
`TensorGraph` construction round-trips a real `GlprocModel`'s layer
structure without loss the constraints from G1 would need.

**Confirmation gate:** report the generator-ownership decision and why,
plus example candidate plans generated from a real GGUF file.

---

## Wave G3 — CostEvaluator + Metric Estimators (Analytical)

**Scope:** analytical estimators for `MetricVector`'s five dimensions,
grounded in this project's own measurements rather than first-principles
FLOP counting where a measurement already exists — e.g. the DDR4 bandwidth
ceiling in [`cpu-skills/memory-bandwidth.md`](../../gl-agent-skills/cpu-skills/memory-bandwidth.md) and glproc's
existing per-stage `PhaseProfile`/`StageTiming` telemetry are real data to
calibrate against, not a reason to build a fresh roofline model from
scratch.

**Files touched:** new `glproc/src/gate_metrics.rs` (estimators); wires
into `glcore::gate::metrics::CostEvaluator` (already real, from Phase 3 —
this wave supplies what feeds it, not the dot product itself).

**Tests required:** at least one estimator validated against a real
measured run on the reference tier (i3-1115G4) — matching the paper's own
methodology (Finding 1/2) of grounding analytical claims in a real
measurement rather than trusting the model in isolation.

**Confirmation gate:** report estimated vs. measured deltas per metric
dimension; flag any dimension where the estimator is not yet trustworthy
(honest "not measured" beats a confident wrong number — the same
discipline [`bench-skills/measurement-discipline.md`](../../gl-agent-skills/bench-skills/measurement-discipline.md) already
requires of every benchmark in this repo).

---

## Wave G4 — Dispatcher Integration (Replaces the Fallback-Chain Convention)

**Scope:** the first wave that touches production dispatch. Per
[`README.md`](README.md)'s fallback-chain section: there is no existing
chain-walking code to "replace" — `glcli` hardcodes `GlprocEngine::new()`
today. This wave makes `Dispatcher::dispatch` real (bridge
`ExecutionPlan` → an actual `Box<dyn GlEngine>` call) and resolves
`GATE-mapping.md` Gap 2 (the `GateError`/`GlError` bridge — an explicit
`From<GateError> for GlError` or equivalent at the one point they meet).

**This is the highest-risk wave** — it changes what a real CLI invocation
actually runs. Per this repo's UI/CLI-testing convention (don't just trust
the test suite for a behavior change): run `glcli` against a real model
before and after, on the golden path, and confirm output is unchanged when
only one backend is available (the common case today).

**Files touched:** `glcore::gate::dispatcher::Dispatcher` (real impl);
`glcore::gate::error` (bridge to `GlError`); `glcli/src/main.rs` (or a new
opt-in path alongside the existing hardcoded one — backward compatibility
during rollout is this wave's call to make, not preempted here).

**Tests required:** dispatch on a valid single-candidate plan returns the
same output as calling the engine directly; dispatch on an all-rejected
candidate set returns `GateError::NoValidPlan`, never a panic or a silent
fallback (Key Invariant 3).

**Confirmation gate:** report the `GlError` bridge decision, a real
before/after `glcli` transcript on the reference hardware, and explicit
confirmation that an unavailable-engine case still fails the way
`fallback-chain.md` Rule 6 requires (loud, not silent).

---

## Wave G5 — glcuda / glvulkan Constraint Implementations

**Scope:** real `BackendCapabilityConstraint` for `glcuda` (backed by
`glcuda::driver::cuda_available()` and `KernelSet` coverage — real
per-dtype/per-op support, not just the whole-engine boolean) and
`glvulkan` (which, being a stub, correctly rejects every candidate via
this same constraint — per `GATE-mapping.md` §5, that is *correct* GATE
behavior under Theorem 10.2, not a bug to special-case around).

**Files touched:** new `glcuda/src/gate_constraints.rs`, `glvulkan/src/gate_constraints.rs`.
Per [`backend-independence.md`](../../gl-agent-skills/architecture-skills/backend-independence.md): each backend's
constraint impl stays self-contained, no cross-backend imports.

**Tests required:** on a machine without CUDA, `BackendCapabilityConstraint`
for glcuda rejects every plan with a clear reason (not a panic); on a
machine with CUDA, it accepts plans within actual kernel/dtype coverage
and rejects those outside it.

**Confirmation gate:** report constraint behavior on both an available and
an unavailable backend (glvulkan's stub covers the latter for free).

---

## Wave G6 — glbench Integration

**Scope:** a `--policy` flag on `glbench`'s workload spec, wired through
`EngineAdapter`/`build_engine()` (see `GATE-mapping.md` §5) so a benchmark
run can request `Latency`/`Memory`/`Energy`/`Balanced`/`Custom` instead of
hardcoding an engine name; a report section showing constraint pass/reject
rates per candidate (mirroring the paper's own "constraint pass rate"
analytical finding, §12.1).

**Files touched:** `glbench/src/core/workload.rs` (add `policy` field);
`glbench/src/engine/adapter.rs` (`build_engine`/`EngineAdapter::load` grow
a GATE-driven path, additive — the existing explicit `--engine` path per
`fallback-chain.md` Rule 6 must keep working, since an explicit engine
request must never silently fall back to policy-driven selection);
`glbench/src/export/{json,markdown}.rs` (constraint pass-rate reporting).

**Tests required:** existing `glbench` engine-selection tests
(`known_engines_lists_gllm` etc.) stay green; a new test confirms
`--policy` selection and an explicit `--engine` request are mutually
exclusive paths that never silently blend.

**Confirmation gate:** report a real `glbench run --policy memory` (or
equivalent) transcript on the reference hardware, plus full existing
`glbench` test suite output to confirm no regression.

---

## After Wave G6

No wave beyond G6 is scoped here. Revisit `GATE-mapping.md` §6 (explicitly
out of scope) before adding one — distributed execution, a native TPU
backend, ML-guided plan generation, and threshold-relative normalization
as default all remain deliberately unstarted until there is a concrete
GwenLand need for them, not because the paper mentions them as future
work.
