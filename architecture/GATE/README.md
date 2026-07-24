# GATE Architecture — Index
## GwenLand AI · Gwen Algorithm for Tensor Execution

> "Numbers are not considered trustworthy until logical constraints validate them."
> — GATE paper, Section 1

---

## What This Folder Is

Living architecture documentation for implementing GATE in the GwenLand AI
engine ecosystem. Derived from the GATE paper (Gwen Research Group, revised
July 17, 2026) and mapped to the actual GwenLand codebase.

**Rule**: Before touching any GATE-related code, read the relevant doc here
first. These docs are the single source of truth for design decisions.

---

## Files

| File | Contents |
|------|----------|
| `README.md` | This index |
| `GATE-concepts.md` | Core mathematical concepts — ExecPlan, Constraint, Validator, MetricVector, WeightVector, Cost |
| `GATE-algorithm.md` | The 4-phase algorithm (generate→validate→evaluate→dispatch), pseudocode, correctness theorems |
| `GATE-constraints.md` | All 7 constraint types — definition, complexity, backend-dependence, default ordering |
| `GATE-policy.md` | ExecutionPolicy, WeightVector presets, normalization, Policy Independence Theorem |
| `GATE-mapping.md` | 🔥 GwenLand-specific — maps every GATE concept to actual crate/file/struct, identifies gaps |
| `GATE-impl-plan.md` | Wave-gated implementation plan — the megaprompt for Claude Code |

---

## Quick Reference — The Algorithm in One Block

```
Input:  TensorGraph G, backends B, constraints C₁…C₇, policy π
Output: ExecutionResult of optimal valid plan

1. Pcand ← GeneratePlans(G, B)          // up to 64 candidates
2. Pvalid ← ∅
3. for each P in Pcand:
     for i = 1..k:
       if Cᵢ(P) = 0 → log rejection, break   // early-exit
     if all passed → Pvalid ∪= {P}
4. if Pvalid = ∅ → ConstraintViolation error  // never silent fallback
5. w ← π(context)                        // policy → weight vector
6. P* ← argmin_{P ∈ Pvalid} wᵀ·m(P)    // cost-minimal valid plan
7. return Dispatch(P*)
```

Complexity: O(knp + dp + n), dominated by O(knp).
Validation overhead empirically: **1.4–8.2 µs** — 7 orders below one inference.

---

## Key Invariants (Never Violate These)

1. **Constraint before optimization** — `Pvalid` is built before any cost is
   evaluated. The argmin is defined over `Pvalid`, never `Pcand`.
2. **Never dispatch invalid** — if `Dispatch(P*)` is called, `V(P*) = 1` by
   construction (Theorem 10.2).
3. **No silent fallback** — `Pvalid = ∅` → explicit `ConstraintViolation`,
   never a "best effort" invalid plan.
4. **Policy independence of validity** — changing `w` can never make an
   invalid plan valid (Theorem 4.2).
5. **Composability** — adding constraint C_{k+1} requires zero modification
   to existing constraints.