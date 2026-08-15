# GATE — Core Concepts
## Mathematical foundation (paper §4) mapped to Rust

> Read [`README.md`](README.md) first for what GATE is/is not and the
> terminology note distinguishing it from `gate-integration.md`'s runtime
> gates.

Every definition below is reproduced from the paper's §4 (Mathematical
Foundation) and mapped to the Rust type that carries it in
`glcore::gate` (see [`GATE-mapping.md`](GATE-mapping.md) for the
crate-ownership rationale — short version: all of it is net-new, nothing
in today's `glcore` overlaps). Types shown here are the target signatures
Phase 3's boilerplate implements; every method body there is `todo!()`.

---

## Domains

The paper defines five base domains a tensor computation draws on:

| Symbol | Domain | Meaning |
|---|---|---|
| `𝒯` | tensor types | rank, shape, dtype |
| `𝒪` | tensor operations | partial functions input-types → output-type |
| `ℬ` | hardware backends | each with supported ops, dtypes, memory capacity, alignment |
| `𝓛` | memory layouts | mappings from logical tensor indices to physical offsets |
| `𝒢 = (V, E)` | tensor computation graph | a DAG of operations (`V`) and data dependencies (`E`) |

GwenLand mapping:

```rust
// glcore::gate::plan — net-new
pub enum BackendKind { Glproc, Glcuda, Glvulkan, Glmetal }   // ℬ, restricted to
                                                              // GwenLand's four
                                                              // engines rather
                                                              // than an open set

pub struct OpId(pub usize);          // an index into a TensorGraph's nodes (V)

pub enum MemoryLayout { /* stub */ } // 𝓛 — no layout enum exists in glcore
                                      // today; DType (glcore::tensor) is the
                                      // closest existing concept, but it
                                      // describes element encoding, not
                                      // index→offset layout, so it is
                                      // extended here, not reused.
```

`𝒯` (tensor types) already exists as `glcore::tensor::DType` — reused, not
wrapped: rank/shape is carried per-tensor already (`Tensor::shape`), and
`DType` is the dtype axis GATE's shape/layout constraints will read.

`𝒪` (tensor operations) and `𝒢` (the DAG) have **no existing counterpart** —
`glcore` has no op-graph representation at all (see the "What must be
created" table in `GATE-mapping.md`). Phase 3 stubs a `TensorOp` marker type
in `plan.rs` sufficient for `ExecutionPlan::ordering: Vec<TensorOp>` to
compile; a real `TensorGraph` type is out of scope for the boilerplate wave
(see `GATE-mapping.md` §6, not in scope).

---

## Execution Plans

> **Definition (Execution plan).** An execution plan for `𝒢` is a tuple
> `P = (σ, β, 𝓛map, m)`, where `σ = (o₁, ..., oₖ)` is a topological ordering
> of the operations of `𝒢`; `β ∈ ℬ` is the target backend; `𝓛map: V → 𝓛`
> assigns a memory layout to each intermediate tensor; and `m ∈ ℝᵈ` is the
> plan's metric vector.

```rust
// glcore::gate::plan
pub struct ExecutionPlan {
    pub ordering: Vec<TensorOp>,          // σ
    pub backend:  BackendKind,            // β
    pub layouts:  HashMap<OpId, MemoryLayout>, // 𝓛map
    pub metrics:  MetricVector,           // m ∈ ℝ⁵ (d = 5, see below)
}
```

The set of all plans is `𝒫`; a finite candidate subset generated for one
request is `𝒫cand ⊆ 𝒫` — `Vec<ExecutionPlan>`, produced by
`Planner::generate_candidates` (see [`GATE-algorithm.md`](GATE-algorithm.md)).

---

## Constraints and Validation

> **Definition (Constraint).** A constraint is a *total* function
> `C: 𝒫 → {0, 1}`. Plan `P` satisfies `C` iff `C(P) = 1`.

Totality matters: every plan gets a definitive pass/fail, never an
undefined case — this is what makes the soundness proof (Theorem 10.1)
go through for any registered constraint set.

```rust
// glcore::gate::constraint
pub trait Constraint: Send + Sync {
    fn validate(&self, plan: &ExecutionPlan) -> ValidationResult;
    fn name(&self) -> &'static str;
}

pub enum ValidationResult { Pass, Reject { reason: String } }
```

`ValidationResult` is the Rust encoding of `{0, 1}` plus a *reason* — the
paper's `C(P) ∈ {0,1}` is a boolean, but §5 Step 3 requires "rejected plans
are logged with the identity of the first violated constraint and the
specific reason." `Reject { reason }` carries that; `Constraint::name()`
carries the constraint identity.

> **Definition (Validator).** A validator over constraints `C₁, ..., Cₖ` is
> the product `V(P) = ∏ᵢ Cᵢ(P)`. `V(P) = 1` iff every constraint accepts
> `P`; a single violation invalidates the plan.

```rust
// glcore::gate::validator
pub struct Validator {
    constraints: Vec<Box<dyn Constraint>>,
}
impl Validator {
    pub fn new() -> Self { /* empty */ }
    pub fn register(&mut self, c: Box<dyn Constraint>) { /* push */ }
    pub fn validate(&self, plan: &ExecutionPlan) -> ValidationResult {
        /* conjunctive, early-exit on first Reject — the product ∏Cᵢ(P),
           computed lazily rather than as k separate booleans multiplied
           together, since early-exit is the whole complexity argument
           (§9.2) */
        todo!()
    }
}
```

> **Definition (Valid execution set).**
> `𝒫valid = { P ∈ 𝒫cand | V(P) = 1 }`.

Always well-defined, possibly empty — `Vec<ExecutionPlan>` filtered by
`Validator::validate`, computed inside `Planner::select_best`. It is the
**only** domain GATE's optimizer ever ranges over (Key Invariant 1 in
`README.md`).

---

## Metrics, Weights, and Cost

GATE uses a five-dimensional metric vector `m(P) = (m₁, ..., m₅)(P)`:

| Dim | Symbol | Description | Unit |
|---|---|---|---|
| Latency | `m₁` | estimated total execution time | ms |
| Peak memory | `m₂` | maximum live allocation | MB |
| Sync overhead | `m₃` | kernel launches, transfers, barriers | ms |
| Energy | `m₄` | total energy consumed | mJ |
| Numerical error | `m₅` | cumulative `L₂` error estimate | — (dimensionless) |

All five are non-negative and estimated analytically from plan structure —
cost evaluation itself adds no execution overhead.

```rust
// glcore::gate::metrics
pub struct MetricVector {
    pub latency_ms: f64,        // m₁
    pub peak_memory_mb: f64,    // m₂
    pub sync_overhead_ms: f64,  // m₃
    pub energy_mj: f64,         // m₄
    pub numerical_error: f64,   // m₅ — dimensionless relative L₂
}
```

> **Definition (Weight vector).** `w ∈ ℝᵈ≥0` with `Σwᵢ = 1`; the set of
> valid weight vectors is the standard simplex `Δᵈ⁻¹`.

```rust
// glcore::gate::metrics
pub struct WeightVector {
    pub weights: [f64; 5],   // must sum to 1.0 — see policy_weight_vectors_sum_to_one test
}
impl WeightVector {
    pub fn cost(&self, m: &MetricVector) -> f64 {
        // wᵀ·m(P) = Σ wᵢ·mᵢ(P)
        todo!()
    }
}
```

> **Definition (Cost function).**
> `𝒞(P, w) = wᵀ·m(P) = Σᵢ wᵢ·mᵢ(P)`.

`𝒞` is `WeightVector::cost`, above — a plain dot product. The paper's
Monotonicity lemma (Lemma 4.1: `𝒞` is strictly increasing in each metric
component, all else fixed) has no separate Rust encoding — it's a
mathematical property of the dot product with `wᵢ > 0`, not something the
type system needs to enforce.

---

## Normalization

Because the five dimensions carry incompatible units (ms, MB, ms, mJ,
dimensionless), every real cost comparison **must** normalize each
dimension first. The paper names two strategies (§13.2):

```rust
// glcore::gate::metrics
pub enum NormalizationStrategy {
    /// Divide each dimension by its maximum value across the candidate set
    /// (the paper's default/reference-implementation choice). Conservative:
    /// a tiny absolute spread in one dimension (e.g. 10⁻⁶ relative L₂
    /// between an exact and a BatchNorm-folded plan) still normalizes to
    /// 1.0 if it's the largest value in its column, so that dimension can
    /// dominate selection despite being practically insignificant. This is
    /// what makes GATE's empirical selection observably correctness-first
    /// (paper Finding 2) — see GATE-policy.md.
    MaxNorm,
    /// Divide the numerical-error dimension by the constraint's own
    /// tolerance ε instead of the candidate maximum, so deviations far
    /// below tolerance contribute proportionally little cost. A principled
    /// alternative the paper leaves to future work (§13.4) rather than
    /// adopting as the default.
    ThresholdRelative { epsilon: f64 },
}

pub fn normalize(plans: &[ExecutionPlan], strategy: NormalizationStrategy)
    -> Vec<MetricVector>
{
    todo!()
}
```

`MaxNorm` is the default because it is what both empirical reference
implementations (`gate_resnet`, `gate_py`) actually used — see
`GATE-algorithm.md`'s empirical validation section for the numbers this
produced.
