//! Metric vectors, weights, cost, and normalization — paper §4.4 (metrics,
//! weights, cost function) and §13.2 (normalization sensitivity).

use crate::gate::plan::ExecutionPlan;

/// The five-dimensional per-plan metric vector — paper's
/// `m(P) = (m₁,...,m₅)` (§4.4). All components are non-negative and
/// estimated analytically from plan structure, so cost evaluation itself
/// adds no execution overhead.
///
/// # Undetermined values
///
/// A field's `Undetermined State` (per `architecture/GateCostModel/
/// ARTX06-Terminology.md`) — a measurement that was attempted and failed,
/// or was never attempted — is represented as `f64::INFINITY`, not `0.0`
/// and not a producer-chosen sentinel. This is a deliberate, load-bearing
/// choice, not a type-system enforcement (each field stays a plain `f64`
/// rather than `Option<f64>`, so existing candidate-producing code is not
/// forced to unwrap a field it always measures): `INFINITY` composes
/// correctly with [`WeightVector::cost`]'s summation — it can never win an
/// `argmin` search under any positive weight, and it does not poison
/// dimensions the candidate's weight is `0.0` for, which a producer
/// silently defaulting to `0.0` instead would get backwards (an
/// unmeasured dimension would look *free*, not *unknown*). A producer
/// setting this MUST log the fact (see `glproc::gate`'s calibration path
/// for the pattern) — `INFINITY` is not meant to pass silently.
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
    ///
    /// Each term is computed via [`weighted_term`] rather than a plain
    /// `wᵢ * mᵢ`: IEEE 754 defines `0.0 * f64::INFINITY = NaN`, so a naive
    /// product would let an `Undetermined State` dimension (see
    /// [`MetricVector`]'s doc) poison the whole sum even when its weight is
    /// exactly `0.0` — silently turning "this dimension doesn't matter for
    /// this policy" into "this plan's cost is unknowable." `weighted_term`
    /// special-cases a zero weight to contribute exactly `0.0`, matching
    /// what "this dimension doesn't matter" actually means.
    pub fn cost(&self, m: &MetricVector) -> f64 {
        weighted_term(self.weights[0], m.latency_ms)
            + weighted_term(self.weights[1], m.peak_memory_mb)
            + weighted_term(self.weights[2], m.sync_overhead_ms)
            + weighted_term(self.weights[3], m.energy_mj)
            + weighted_term(self.weights[4], m.numerical_error)
    }
}

/// `weight * value`, except an exactly-zero weight always contributes
/// `0.0` regardless of `value` — including `f64::INFINITY` (an
/// `Undetermined State` metric, see [`MetricVector`]'s doc). Without this,
/// `0.0 * f64::INFINITY` is `NaN` per IEEE 754, and a single `NaN` term
/// poisons the whole cost sum, making an unmeasured-but-irrelevant
/// dimension look unrankable instead of simply ignored.
fn weighted_term(weight: f64, value: f64) -> f64 {
    if weight == 0.0 {
        0.0
    } else {
        weight * value
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

#[cfg(test)]
mod tests {
    use super::*;

    fn latency_only() -> WeightVector {
        WeightVector { weights: [1.0, 0.0, 0.0, 0.0, 0.0] }
    }

    fn plan_with_latency(latency_ms: f64) -> ExecutionPlan {
        ExecutionPlan {
            metrics: MetricVector { latency_ms, ..Default::default() },
            ..Default::default()
        }
    }

    #[test]
    fn cost_evaluator_increases_with_latency() {
        let evaluator = CostEvaluator::new(latency_only());
        let cheap = evaluator.evaluate(&plan_with_latency(5.0));
        let expensive = evaluator.evaluate(&plan_with_latency(50.0));
        assert!(expensive > cheap, "higher latency_ms must cost more");
    }

    #[test]
    fn cost_evaluator_increases_with_each_dimension_independently() {
        let uniform = WeightVector { weights: [0.2, 0.2, 0.2, 0.2, 0.2] };
        let baseline = MetricVector::default();
        let base_cost = CostEvaluator::new(uniform)
            .evaluate(&ExecutionPlan { metrics: baseline, ..Default::default() });

        let bumped = [
            MetricVector { latency_ms: 10.0, ..baseline },
            MetricVector { peak_memory_mb: 10.0, ..baseline },
            MetricVector { sync_overhead_ms: 10.0, ..baseline },
            MetricVector { energy_mj: 10.0, ..baseline },
            MetricVector { numerical_error: 10.0, ..baseline },
        ];
        for m in bumped {
            let cost = CostEvaluator::new(uniform).evaluate(&ExecutionPlan {
                metrics: m,
                ..Default::default()
            });
            assert!(cost > base_cost, "bumping {m:?} alone must raise cost");
        }
    }

    /// `Undetermined State` — `f64::INFINITY` per this module's doc — must
    /// make a candidate strictly worse under a positive weight (never win
    /// `argmin`), and must NOT be `NaN` (a `NaN` cost breaks every
    /// `argmin` comparison silently, since every `NaN` comparison is
    /// `false`) or a finite value that could look like a real measurement.
    #[test]
    fn cost_evaluator_handles_undetermined_state_explicitly() {
        let evaluator = CostEvaluator::new(latency_only());
        let undetermined_cost = evaluator.evaluate(&plan_with_latency(f64::INFINITY));
        assert!(undetermined_cost.is_infinite(), "must stay infinite, not collapse to NaN or 0");
        assert!(!undetermined_cost.is_nan());
        let known_cost = evaluator.evaluate(&plan_with_latency(1_000_000.0));
        assert!(
            undetermined_cost > known_cost,
            "Undetermined must lose to even a very bad known measurement"
        );
    }

    /// The dimension-poisoning failure mode this module's doc warns
    /// against: an Undetermined `latency_ms` must NOT make the plan's cost
    /// infinite when the policy assigns latency a weight of `0.0` — only
    /// the dimensions actually weighted should determine the outcome.
    #[test]
    fn undetermined_dimension_does_not_poison_zero_weighted_evaluation() {
        let memory_only = WeightVector { weights: [0.0, 1.0, 0.0, 0.0, 0.0] };
        let evaluator = CostEvaluator::new(memory_only);
        let plan = ExecutionPlan {
            metrics: MetricVector {
                latency_ms: f64::INFINITY, // undetermined, but unweighted here
                peak_memory_mb: 42.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let cost = evaluator.evaluate(&plan);
        assert!(cost.is_finite(), "zero-weighted Undetermined dimension must not poison the cost");
        assert_eq!(cost, 42.0);
    }
}
