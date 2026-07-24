//! Metric vectors, weights, cost, and normalization — paper §4.4 (metrics,
//! weights, cost function) and §13.2 (normalization sensitivity).

use crate::gate::plan::ExecutionPlan;

/// The five-dimensional per-plan metric vector — paper's
/// `m(P) = (m₁,...,m₅)` (§4.4). All components are non-negative and
/// estimated analytically from plan structure, so cost evaluation itself
/// adds no execution overhead.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct MetricVector {
    /// `m₁` — estimated total execution time, ms.
    pub latency_ms: f64,
    /// `m₂` — maximum live allocation, MB.
    pub peak_memory_mb: f64,
    /// `m₃` — kernel launches, transfers, barriers, ms.
    pub sync_overhead_ms: f64,
    /// `m₄` — total energy consumed, mJ.
    pub energy_mj: f64,
    /// `m₅` — cumulative relative `L₂` error estimate, dimensionless.
    pub numerical_error: f64,
}

/// A point on the weight simplex `Δ⁴` — paper's `w ∈ ℝ⁵≥0, Σwᵢ = 1`
/// (§4.4).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightVector {
    /// Per-dimension weights, in the same order as [`MetricVector`]'s
    /// fields: latency, peak memory, sync overhead, energy, numerical
    /// error.
    pub weights: [f64; 5],
}

impl WeightVector {
    /// The cost function `𝒞(P, w) = wᵀ·m(P) = Σᵢ wᵢ·mᵢ(P)` (paper §4.4).
    pub fn cost(&self, m: &MetricVector) -> f64 {
        self.weights[0] * m.latency_ms
            + self.weights[1] * m.peak_memory_mb
            + self.weights[2] * m.sync_overhead_ms
            + self.weights[3] * m.energy_mj
            + self.weights[4] * m.numerical_error
    }
}

/// Evaluates plan cost under a fixed weight vector — the paper's reference
/// interfaces' `CostEvaluator` (§5.1 `CostEvaluator(weights)`, §5.2
/// `CostEvaluator::new(weights)`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEvaluator {
    weights: WeightVector,
}

impl CostEvaluator {
    /// Construct an evaluator for a fixed weight vector.
    pub fn new(weights: WeightVector) -> Self {
        CostEvaluator { weights }
    }

    /// Evaluate `𝒞(P, w)` for one plan.
    pub fn evaluate(&self, plan: &ExecutionPlan) -> f64 {
        self.weights.cost(&plan.metrics)
    }
}

/// How to normalize heterogeneous metric dimensions before applying
/// weights (paper §13.2) — behavior-relevant, not cosmetic: see
/// `architecture/GATE/GATE-policy.md` (Finding 2) for why the choice
/// changes which plan gets selected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormalizationStrategy {
    /// Divide each dimension by its maximum value across the candidate
    /// set (the paper's reference-implementation default).
    MaxNorm,
    /// Divide the numerical-error dimension by the constraint tolerance
    /// `epsilon` instead of the candidate maximum (paper §13.2, a
    /// principled alternative left to future work).
    ThresholdRelative {
        /// The numerical-error constraint's tolerance, `ε`.
        epsilon: f64,
    },
}

/// Normalize each candidate plan's metric vector under `strategy`. Stub:
/// needs a real multi-candidate cost-evaluation pipeline to be meaningful
/// — see `architecture/GATE/GATE-mapping.md` §3.
pub fn normalize(_plans: &[ExecutionPlan], _strategy: NormalizationStrategy) -> Vec<MetricVector> {
    todo!()
}
