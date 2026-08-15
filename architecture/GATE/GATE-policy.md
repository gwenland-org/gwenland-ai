# GATE — Execution Policies
## Paper §4.4 (policy definition), §7 (cost evaluation & policy table), §13.2 (normalization)

> Read [`GATE-concepts.md`](GATE-concepts.md) first for `MetricVector` /
> `WeightVector` / the cost function this file builds on.

---

## The Five Preset Policies (Table 2)

> **Definition (Execution policy).** A policy is a mapping
> `π: 𝒦 → Δᵈ⁻¹` from a deployment context `κ` to a weight vector.

| Policy | Weight vector `w` = (latency, memory, sync, energy, error) | Recommended context |
|---|---|---|
| Latency | `(0.6, 0.1, 0.1, 0.1, 0.1)` | real-time, interactive services |
| Memory | `(0.1, 0.6, 0.1, 0.1, 0.1)` | edge devices, multi-model batching |
| Energy | `(0.1, 0.1, 0.1, 0.6, 0.1)` | mobile, battery, green computing |
| Balanced | `(0.2, 0.2, 0.2, 0.2, 0.2)` | general-purpose, benchmarking |
| Custom | user-supplied | domain-specific requirements |

Policy resolution is a constant-time lookup plus normalization — no search,
no per-request cost.

```rust
// glcore::gate::policy
pub enum ExecutionPolicy {
    Latency,
    Memory,
    Energy,
    Balanced,
    Custom(WeightVector),
}
impl ExecutionPolicy {
    pub fn weight_vector(&self) -> WeightVector {
        // Latency  = [0.6, 0.1, 0.1, 0.1, 0.1]
        // Memory   = [0.1, 0.6, 0.1, 0.1, 0.1]
        // Energy   = [0.1, 0.1, 0.1, 0.6, 0.1]
        // Balanced = [0.2, 0.2, 0.2, 0.2, 0.2]
        // Custom(w) = w, returned as-is
        todo!()
    }
}
```

The `policy_weight_vectors_sum_to_one` test (Phase 3) asserts
`weights.iter().sum::<f64>() ≈ 1.0` for all four presets — the simplex
constraint from the `WeightVector` definition, checked once so it's never
silently violated by a typo in the table above.

---

## Theorem 4.2 — Policy Independence of Validity

> For any `w, w' ∈ Δᵈ⁻¹`, `𝒫valid(w) = 𝒫valid(w')`.
>
> *Proof.* `𝒫valid` is defined by `V` alone (`V(P) = ∏Cᵢ(P)`); no
> constraint reads `w`. Hence the valid set is invariant under any change
> of weights. ∎

**Implication:** the policy affects only *which valid plan is selected*,
never *whether the selected plan is valid*. Switching from `Latency` to
`Memory` mid-deployment can change the winning plan; it can never turn an
invalid plan valid, or vice versa. This is Key Invariant 4 in
[`README.md`](README.md), and it's what makes `ExecutionPolicy` a
configuration knob rather than a correctness lever — a user (or `--policy`
CLI flag, see `GATE-mapping.md`'s glbench integration point) can never
accidentally weaken correctness by picking a policy.

---

## Normalization Sensitivity (§13.2)

Because the five metric dimensions carry incompatible units, every real
comparison normalizes first (see `NormalizationStrategy` in
`GATE-concepts.md`). The choice of strategy is **behavior-relevant**, not
cosmetic — the paper's own empirical study demonstrates this:

**Finding 2 (paper §12.2):** under the Latency policy, GATE selected the
exact-numerics plan `t4` (4 threads, unfolded BatchNorm) over the ~6%
*faster* `t4_bnfold`: BatchNorm-folding perturbs output by relative `L₂` of
`1.1×10⁻⁶`, and under **max-normalization** this nonzero `m₅` outweighs the
latency gain, because max-normalization makes the *largest* candidate value
in a dimension the unit — so a tiny absolute spread (10⁻⁶ vs 0) still
normalizes to 1.0 and can dominate selection. This is conservative
(exactness-preserving) but can forgo real, safe performance. The paper's
`gate_py` reference implementation, whose candidate family normalizes
differently, accepted the folding — both behaviors are correct instances
of Theorem 10.3 (optimality is relative to normalization + candidate set),
illustrating that **the weight vector and normalization strategy — not
ad-hoc heuristics — arbitrate the performance-vs-exactness frontier.**

`ThresholdRelative { epsilon }` (normalize `m₅` by the constraint's own
tolerance `ε` rather than the candidate maximum) is the principled
alternative the paper names but leaves to future work (§13.4) rather than
adopting as default. `MaxNorm` is Phase 3's default for the same reason
it was the paper's reference-implementation default: it is the
conservative, exactness-preserving choice, and GwenLand's own numerical
history (the Q6_K corruption bug, `GATE-constraints.md` §5) is a strong
argument for staying conservative until there's a reason not to.

---

## When to Use Which Policy in GwenLand

| Context | Policy | Why |
|---|---|---|
| `glbench` benchmarking | **Balanced** | equal weight across dimensions is the right default for "how does this engine behave generally" — matches the paper's own analytical-study configuration (Balanced policy, `k=7`). |
| Edge/consumer devices (the project's reference tier: Tiger Lake i3-1115G4, 8 GB DDR4) | **Memory** | this machine is the one where the ARTX05 KV-cache trap (8.75 GiB needed, 8 GB available) actually happened — memory headroom is the binding constraint here, not raw speed. |
| Real-time inference (interactive chat/completion) | **Latency** | matches the paper's own recommended context; GwenLand's `InferOutput` already separates `prefill_ms`/`generation_ms` specifically because interactive latency is judged on decode speed, not blended throughput. |
| Future TPU/cloud path | **Energy** | the paper's own TPU v5e datapoint notes energy metrology is platform-gated (no RAPL on Windows, no power telemetry on a Colab guest) — Energy policy only becomes meaningful once GwenLand has a deployment target that can actually measure `m₄`. Not actionable today; recorded here as the reason the preset exists ahead of the hardware. |

No context in GwenLand today calls for **Custom** — it exists for the case
where a deployment's tradeoff doesn't match any preset (e.g. a hard
memory ceiling *and* a hard latency SLA at once). Nothing in the current
codebase needs it yet; it's here because the paper's `WeightVector` type
already supports it at zero extra cost.
