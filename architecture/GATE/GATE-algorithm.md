# GATE — The Algorithm
## Paper §5 (algorithm), §9 (complexity), §10 (correctness), §12.2 (empirical)

> Read [`GATE-concepts.md`](GATE-concepts.md) first — this file uses
> `ExecutionPlan`, `Validator`, `MetricVector`, `WeightVector` as defined
> there without re-deriving them.

---

## The Seven Steps, Annotated

GATE proceeds in four phases — **generate, validate, evaluate, dispatch** —
refined into seven steps (paper §5):

1. **Generate candidate plans.** The Plan Generator explores execution
   strategies for the tensor graph across available backends, producing a
   finite candidate set `𝒫cand` (default bound: 8 candidates, maximum 64).
   Each candidate fixes backend assignment, layout configuration, operation
   ordering, and fusion decisions.
   — GwenLand: `Planner::generate_candidates(&self, graph, policy) -> Vec<ExecutionPlan>`.
2. **Validate.** Each candidate is submitted to the Constraint Engine
   (`V(P) = ∏Cᵢ(P)`) with early-exit: the first violated constraint
   short-circuits the remaining checks.
   — GwenLand: `Validator::validate(&self, plan) -> ValidationResult`.
3. **Reject invalid plans.** Rejected plans are logged with the identity of
   the first violated constraint and the specific reason — diagnostics, not
   silent fallback.
   — GwenLand: `ValidationResult::Reject { reason }` plus `Constraint::name()`.
4. **Error on empty valid set.** If `𝒫valid = ∅`, terminate with
   `ConstraintViolation`. GATE never silently substitutes a potentially
   incorrect plan.
   — GwenLand: `GateError::NoValidPlan { candidates_tried }` (see
   [`GATE-mapping.md`](GATE-mapping.md) for why the Rust error type splits
   this from the paper's single `ConstraintViolation` name).
5. **Evaluate cost.** Compute `𝒞(P, w) = wᵀ·m(P)` for every valid plan, with
   `w` resolved from the execution policy.
   — GwenLand: `WeightVector::cost(&self, m: &MetricVector) -> f64`.
6. **Select.** `P* = argmin_{P ∈ 𝒫valid} 𝒞(P, w)`.
   — GwenLand: `Planner::select_best(&self, candidates) -> Result<ExecutionPlan, GateError>`.
7. **Dispatch.** Execute `P*` on its target backend and return the output
   tensors.
   — GwenLand: `Dispatcher::dispatch(&self, plan) -> Result<ExecutionResult, GateError>`.

### Algorithm 1 (paper, reproduced)

```
Input:  graph G; backends B; constraints C₁,...,Cₖ; policy π; context κ
Output: execution result of the optimal valid plan

Pcand ← GeneratePlans(G, B)
Pvalid ← ∅
foreach P ∈ Pcand:
    valid ← ⊤
    for i ← 1 to k:
        if Cᵢ(P) = 0:
            valid ← ⊥; LogRejection(P, Cᵢ); break
    if valid: Pvalid ← Pvalid ∪ {P}
if Pvalid = ∅:
    return ConstraintViolation
w ← π(κ)
P* ← argmin_{P ∈ Pvalid} wᵀ·m(P)
return Dispatch(P*)
```

---

## Reference Interface — Python (paper §5.1)

```python
def execute(
    tensor_graph: Sequence[TensorOp],
    policy: ExecutionPolicy = ExecutionPolicy.BALANCED,
    *,
    memory_budget_mb: float = 4096.0,
    max_numerical_error: float = 1e-4,
    require_determinism: bool = False,
    custom_weights: Optional[WeightVector] = None,
    num_candidates: int = 8,
) -> ExecutionResult:
    validator = Validator(constraints=[
        ShapeConstraint(tensor_graph),
        TensorLayoutConstraint(),
        MemoryConstraint(memory_budget_mb),
        BackendCapabilityConstraint(),
        NumericalErrorConstraint(max_numerical_error),
        DeterminismConstraint(require_determinism),
        SafetyConstraint(),
    ])
    weights = custom_weights or WeightVector.from_policy(policy)
    planner = Planner(validator=validator,
                      cost_evaluator=CostEvaluator(weights),
                      num_candidates=num_candidates)
    candidates = planner.generate_candidates(tensor_graph, policy)
    best_plan = planner.select_best(candidates)
    return Dispatcher().dispatch(best_plan)
```

## Reference Interface — Rust (paper §5.2)

```rust
fn execute(
    graph:  &TensorGraph,
    policy: ExecutionPolicy,
    config: &GateConfig,
) -> Result<ExecutionResult, PlanError> {
    let mut validator = Validator::new();
    validator.register(Box::new(ShapeConstraint { graph: graph.ops.clone() }));
    validator.register(Box::new(TensorLayoutConstraint::new()));
    validator.register(Box::new(MemoryConstraint { budget_mb: config.memory_budget_mb }));
    validator.register(Box::new(BackendCapabilityConstraint::new()));
    validator.register(Box::new(NumericalErrorConstraint { max_error: config.max_numerical_error }));
    validator.register(Box::new(DeterminismConstraint::new(config.require_determinism)));
    validator.register(Box::new(SafetyConstraint::new()));

    let weights = config.custom_weights
        .unwrap_or_else(|| WeightVector::from_policy(policy));
    let planner = Planner::new(validator, CostEvaluator::new(weights),
                               config.num_candidates);
    let mut candidates = planner.generate_candidates(graph, policy);
    let best_plan = planner.select_best(&mut candidates)?;
    Dispatcher::new().dispatch(&best_plan)
}
```

This is the paper's own reference Rust interface, not GwenLand-adjusted —
Phase 3's boilerplate constructs the equivalent types (`Planner`,
`Validator`, `Dispatcher`, `WeightVector`, `GateError` in place of
`PlanError`), but nothing calls `execute()` yet: no `TensorGraph` type
exists to pass it (see `GATE-mapping.md`, "what must be created"). This
snippet is the target shape for a future wave, not code that compiles
today.

---

## Empty Valid Set Semantics

When `𝒫valid = ∅` the argmin is undefined, and GATE raises an explicit
`ConstraintViolation` rather than falling through to any plan. Formally,
dispatch is the *partial* function

```
Φ(𝒫cand, V, w) = { P* = argmin_{P∈𝒫valid} 𝒞(P,w)   if 𝒫valid ≠ ∅
                  { ⊥ (ConstraintViolation)          if 𝒫valid = ∅
```

GATE either returns a valid, cost-minimal plan or raises an explicit error;
it never returns an invalid plan. In GwenLand this is
`Planner::select_best` returning `Result<ExecutionPlan, GateError>`, with
`GateError::NoValidPlan { candidates_tried }` as the `⊥` case.

---

## Correctness Theorems (paper §10)

> **Theorem 10.1 (Constraint soundness).** If `V(P) = 1`, then `P` satisfies
> all `k` constraints: `∀i ∈ {1,...,k}: Cᵢ(P) = 1`.
>
> *Proof.* `V(P) = ∏ᵢ Cᵢ(P)` with each factor in `{0,1}`. The product equals
> 1 iff every factor equals 1. ∎

Acceptance by the validator is a *sufficient* condition for constraint
satisfaction — every plan surviving validation is semantically correct
with respect to all registered constraints.

> **Theorem 10.2 (Dispatch correctness).** GATE never dispatches an invalid
> plan: if GATE dispatches `P*`, then `V(P*) = 1`.
>
> *Proof.* The dispatch phase receives input exclusively from
> `𝒫valid = {P ∈ 𝒫cand | V(P) = 1}`. Hence `P* ∈ 𝒫valid` and `V(P*) = 1`. ∎

This safety property holds unconditionally — for any constraint set,
backend inventory, or computation graph — because it relies only on the
structural fact that dispatch operates over `𝒫valid`. This is why Key
Invariant 1 in `README.md` ("constraint before optimization") is described
as structural, not procedural: it is *what makes Theorem 10.2 true*, not a
convention layered on top of it.

**Corollary (constraint satisfaction).** Every dispatched plan satisfies all
seven default constraint types: shape, tensor layout, memory, backend
capability, numerical error, determinism, and safety.

> **Theorem 10.3 (Optimality).** Among valid plans, GATE selects the
> cost-minimal one: `∀P ∈ 𝒫valid: 𝒞(P*, w) ≤ 𝒞(P, w)`.
>
> *Proof.* `𝒞(·, w)` is a real-valued function on the finite, non-empty set
> `𝒫valid`; such a function attains its minimum, and GATE computes it by
> linear scan. ∎

GATE does not trade optimality for correctness — it restricts the *search
space* to valid plans and is exactly optimal within it. The guarantee is
relative to candidate-set quality (§13.3, Limitations): if the true optimal
plan was never generated into `𝒫cand`, GATE cannot select it.

**Additional properties** (§10.1): *Termination* — `𝒫cand` is finite, each
constraint check terminates, dispatch delegates to backends that terminate
on valid plans. *Completeness* — if at least one valid plan exists in
`𝒫cand`, GATE finds it (the validation loop examines every candidate).
*Safety* — independent of constraint set, cost model, weight vector, and
backend, the algorithm never dispatches an invalid plan (Theorem 10.2).

*Remark (completeness of the constraint set).* GATE guarantees satisfaction
of *registered* constraints. The seven defaults cover the most common error
sources, but novel sources may exist; the composable design admits new
constraints, and Theorem 10.1 holds for any `k`.

---

## Complexity (paper §9.1, Table 3)

| Component | Complexity | Notes |
|---|---|---|
| Single constraint (per plan) | `O(n)` | Safety `O(n²)` worst case, `O(n)` observed |
| Validation (total) | `O(knp)` | `k = 7` by default |
| Plan generation | `O(N_max · n)` | bounded by `N_max = 64` |
| Cost evaluation | `O(dp)` | `d = 5` |
| Dispatch | `O(n)` | single plan, one-time |
| **Overall** | `O(knp + dp + n)` | dominated by `O(knp)` |

(`n` = operations in the graph, `k` = constraints, `p` = candidate plans,
`d` = metric dimensions.)

The worst case assumes every plan survives all `k` constraints. With the
default ordering (cheap, high-rejection constraints first — see
`GATE-constraints.md` §6.3), early-exit reduces effective cost by
**2.5–3.5×**: shape and backend-capability checks reject 30–50% of
candidates, and on average only 2–3 of 7 constraints execute per plan.
Memory: `O(p(n+d))` for candidate plans and metric vectors — linear in
graph size for `p ≤ 64`, `d = 5`.

---

## Empirical Validation (paper §12.2)

To ground the analytical complexity model, the paper's authors implemented
the full algorithm twice — independently, from scratch — and measured it
end-to-end on real hardware (an Intel Tiger Lake i3-1115G4, plus one TPU
v5e datapoint via Colab), running a pretrained ResNet-50.

**Finding 1: constraint validation is empirically free.** For `n = 6`
candidates and the full `k = 7` constraint set, validation plus
cost-minimal selection completes in **1.4–8.2 µs per policy** — seven
orders of magnitude below a single inference (376 ms) — consistent with
the `O(knp)` bound above. The dominant planning cost in practice is not the
algorithm but optional per-candidate *calibration* (one real execution per
plan, when empirically-grounded metrics are desired), which amortizes over
any realistic deployment.

**Finding 2** (selection is observably correctness-first) and **Finding 3**
(a production compiler — TVM 0.25.0 — silently miscompiled the same
ResNet-50, changing the top-1 prediction, while every per-operator
micro-test passed) are discussed in `GATE-policy.md` and
`GATE-constraints.md` respectively, where they motivate specific design
choices (normalization strategy; the numerical-error constraint).

This is why the `AlwaysReject` test constraint and the empty-validator test
in Phase 3's boilerplate matter even though they contain no real logic:
they are the scaffolding the *first* real constraint (Wave G1, see
`GATE-impl-plan.md`) will be measured against, in the same spirit as the
paper's own dependency-free reference implementations existing to validate
the production crates.
