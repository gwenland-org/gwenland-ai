# GATE Architecture — Index
## GwenLand AI · Gwen Algorithm for Tensor Execution

> "Numbers are not considered trustworthy until logical constraints validate them."
> — GATE paper, Section 1

---

## What This Folder Is

Living architecture documentation for implementing GATE in the GwenLand AI
engine ecosystem. Derived from the GATE paper (`GATE.tex`/`GATE.pdf`, Gwen
Research Group, revised edition) and mapped to the actual GwenLand codebase
as of 2026-07-24.

**Rule**: Before touching any GATE-related code, read the relevant doc here
first. These docs are the single source of truth for design decisions.

**Terminology note — do not confuse with `gate-integration.md`.** GwenLand
already uses the lowercase word "gate" for something else:
[`gl-agent-skills/architecture-skills/gate-integration.md`](../../gl-agent-skills/architecture-skills/gate-integration.md)
documents *measured-policy runtime decision points* (the AVX-512 decline,
the ≥4-KV-head threading threshold, the Q4_K→Q8_0 repack, ...) — small,
named `if` branches justified by a benchmark. **GATE** (this folder, always
capitalized) is the *Gwen Algorithm for Tensor Execution*: a formal
generate→validate→evaluate→dispatch protocol for selecting a whole execution
plan. The two are related in spirit (both pick a path from measurement, not
vibes) but are not the same mechanism — GATE's constraints are logical
predicates proven before cost is ever considered, not empirical if/else
gates. `gate-integration.md` has been updated to point here rather than
paraphrase this paper.

---

## What GATE Is — and Is Not

(Paper §1.1.) GATE is **not** a neural-network framework (no layers, losses,
or training loops), **not** a compiler (parses no source, emits no machine
code), **not** an optimizer (applies no loop transformations or fusion
itself), **not** a scheduler, and **not** a tensor library. GATE **is** the
protocol layer that sits above all of these: it accepts candidate execution
plans as input, validates them against a composable set of logical
constraints, evaluates their cost under a policy-driven objective, and
selects the optimal valid plan for dispatch. GATE defines the *protocol* by
which optimization and correctness interact, not the implementation of
either half.

Concretely for GwenLand: GATE does not replace glproc's kernels, glcuda's
PTX, or glvulkan's SPIR-V — it decides, given a set of already-possible
execution plans across those backends, which one is provably correct and
cheapest under the active policy.

---

## Files

| File | Contents |
|------|----------|
| `README.md` | This index |
| [`GATE-concepts.md`](GATE-concepts.md) | Mathematical definitions (§4) — domains, `ExecutionPlan`, `Constraint`, `Validator`, `MetricVector`, `WeightVector`, cost — each mapped to the Rust type Phase 3 introduces |
| [`GATE-algorithm.md`](GATE-algorithm.md) | The 7-step algorithm (§5) annotated line by line, both reference interfaces (Python §5.1 / Rust §5.2), the correctness theorems (§10), the complexity table (§9.1), empirical validation (Finding 1) |
| [`GATE-constraints.md`](GATE-constraints.md) | All 7 constraint types (§6.1) — definition, complexity, backend-dependence, what GwenLand code already does today; default validation order (§6.3); composability (§6.2); the TVM 0.25.0 miscompilation case |
| [`GATE-policy.md`](GATE-policy.md) | The 5 preset policies and their weight vectors (§7/Table 2), the Policy Independence theorem (Thm. 4.2), normalization sensitivity (§13.2), the `ExecutionPolicy` Rust design, when to use which policy in GwenLand |
| [`GATE-mapping.md`](GATE-mapping.md) 🔥 | GwenLand-specific — concept→crate mapping, what already exists vs what's net-new, gaps and decisions, integration points, explicitly out of scope |
| [`GATE-impl-plan.md`](GATE-impl-plan.md) | Wave-gated implementation plan for turning the boilerplate into a working constraint engine |

**License** (for `GATE.tex`/`GATE.pdf` and its `sections/`): Dual — [CC
BY-NC-ND 4.0](LICENSE) (non-commercial) + commercial license available on
request (see [LICENSE](LICENSE)). The architecture-mapping docs listed
above (`GATE-*.md`) are GwenLand project documentation, not part of the
paper, and follow the repository's own code license.

---

## Quick Reference — The Algorithm in One Block

Algorithm 1 from the paper (§5), generate→validate→evaluate→dispatch:

```
Input:  tensor graph G; backends B; constraints C₁,...,Cₖ; policy π; context κ
Output: ExecutionResult of the optimal valid plan, or ConstraintViolation

Pcand  ← GeneratePlans(G, B)                 // Step 1 — up to 64 candidates
Pvalid ← ∅
for each P in Pcand:                          // Step 2 — validate
    valid ← true
    for i = 1..k:
        if Cᵢ(P) = 0:
            valid ← false
            LogRejection(P, Cᵢ)               // Step 3 — reject with diagnostics
            break                              //          (early-exit)
    if valid: Pvalid ← Pvalid ∪ {P}
if Pvalid = ∅:
    return ConstraintViolation                // Step 4 — never silent fallback
w  ← π(κ)                                     // Step 5 — resolve policy → weights
P* ← argmin_{P ∈ Pvalid} wᵀ·m(P)               // Step 6 — cost-minimal VALID plan
return Dispatch(P*)                           // Step 7
```

Complexity: `O(knp + dp + n)`, dominated by `O(knp)` (Table 3, §9.1).
Validation overhead empirically: **1.4–8.2 µs** per policy — seven orders
of magnitude below one model inference (§12.2, Finding 1).

---

## Key Invariants (Never Violate These)

1. **Constraint before optimization** — `Pvalid` is built before any cost is
   evaluated. The argmin is defined over `Pvalid`, never `Pcand`. This is
   structural (the selection criterion is *defined* this way, §4.5), not a
   procedural convention an implementation could skip.
2. **Never dispatch invalid** — if `Dispatch(P*)` is called, `V(P*) = 1` by
   construction (Theorem 10.2, dispatch correctness).
3. **No silent fallback** — `Pvalid = ∅` → explicit `ConstraintViolation`,
   never a "best effort" invalid plan (§4.6, empty valid set semantics).
4. **Policy independence of validity** — changing the weight vector `w` can
   never change which plans are valid, only which valid plan wins (Theorem
   4.2).
5. **Composability** — adding constraint `C_{k+1}` requires zero
   modification to existing constraints; open for extension, closed for
   modification (§6.2).

---

## How GATE Relates to GwenLand's Fallback Chain

`gl-agent-skills/architecture-skills/fallback-chain.md` documents the
intended engine order **glcuda → glvulkan → glmetal → glproc**, with glproc
as the unconditional floor. That file is honest about its own scope: it
describes a *convention*, not a mechanism. Confirmed by reading the actual
code (2026-07-24): `glcli` hard-codes `GlprocEngine::new()`, and glbench's
`build_engine()` matches on an explicit `--engine` name string with a hard
error on an unknown name. **There is no code today that walks the chain and
auto-selects an engine** — the order is enforced only by caller discipline
and documentation.

GATE does not replace an existing selection mechanism, then — it would be
the **first real implementation** of automatic engine selection, and a
strictly more rigorous one: instead of trying backends in a fixed preference
order until one reports available, GATE generates concrete candidate plans
across whichever backends are capable, proves each one valid (shape, layout,
memory, backend capability, numerical error, determinism, safety), and picks
the cost-minimal valid plan under the active policy. The fixed order in
`fallback-chain.md` becomes one *possible outcome* of `BackendCapabilityConstraint`
plus cost evaluation, not the mechanism itself. Wiring this in (replacing
today's per-caller hardcoding) is Wave G4 in
[`GATE-impl-plan.md`](GATE-impl-plan.md) — it does not happen as part of this
sprint's boilerplate, which only introduces the types.
