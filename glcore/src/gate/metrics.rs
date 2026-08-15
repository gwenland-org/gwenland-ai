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

/// Normalize each candidate plan's metric vector under `strategy` (paper
/// §13.2), returning one normalized vector per plan, in order.
///
/// Normalization is what makes five incommensurable units (ms, MB, mJ, a
/// dimensionless error) addable under one [`WeightVector`], and it is
/// behavior-relevant, not cosmetic: `GATE-policy.md` Finding 2 records a case
/// where max-normalization makes a `1.1×10⁻⁶` numerical perturbation outweigh
/// a 6% latency win, because the *largest candidate value* becomes the unit
/// and a tiny absolute spread still normalizes to `1.0`.
///
/// # Undetermined values
///
/// `f64::INFINITY` means "measurement attempted and failed, or never
/// attempted" ([`MetricVector`]'s doc), and this function preserves that
/// meaning rather than arithmetic on it:
///
/// * A dimension's maximum is taken over **finite** values only — an
///   unmeasured candidate must not silently become the unit every other
///   candidate is divided by.
/// * An `INFINITY` entry stays `INFINITY`, so it can still never win an
///   `argmin`. Dividing it by the max would give `NaN` (or `0.0` against an
///   infinite max), and a `NaN` cost compares `false` against everything,
///   which breaks selection silently — the exact failure
///   [`WeightVector::cost`] already guards against.
/// * A dimension whose finite maximum is `0.0` (or which has no finite
///   values) normalizes its finite entries to `0.0` rather than `0.0/0.0`.
///   Every candidate scoring zero there means the dimension cannot
///   discriminate, which is what `0.0` correctly encodes.
///
/// An empty `plans` yields an empty `Vec`.
pub fn normalize(plans: &[ExecutionPlan], strategy: NormalizationStrategy) -> Vec<MetricVector> {
    /// Largest finite value of one dimension across the candidate set;
    /// `None` when the dimension holds no finite value at all.
    fn finite_max(plans: &[ExecutionPlan], get: impl Fn(&MetricVector) -> f64) -> Option<f64> {
        plans
            .iter()
            .map(|p| get(&p.metrics))
            .filter(|v| v.is_finite())
            .fold(None, |acc: Option<f64>, v| Some(acc.map_or(v, |a| a.max(v))))
    }

    /// `value / unit`, preserving Undetermined and never emitting `NaN`.
    fn scale(value: f64, unit: Option<f64>) -> f64 {
        if !value.is_finite() {
            return value; // Undetermined stays Undetermined.
        }
        match unit {
            Some(u) if u > 0.0 => value / u,
            // No finite maximum, or a maximum of zero: the dimension does not
            // discriminate between candidates.
            _ => 0.0,
        }
    }

    let max_latency = finite_max(plans, |m| m.latency_ms);
    let max_memory = finite_max(plans, |m| m.peak_memory_mb);
    let max_sync = finite_max(plans, |m| m.sync_overhead_ms);
    let max_energy = finite_max(plans, |m| m.energy_mj);

    // `m₅` is the one dimension the strategies disagree about: `MaxNorm` uses
    // the candidate maximum like every other dimension, `ThresholdRelative`
    // uses the constraint's own tolerance `ε` so that "1.0" means "exactly at
    // the tolerance" instead of "worst of whatever happened to be generated".
    // A non-positive or non-finite `epsilon` is not a usable unit, so it falls
    // back to the candidate maximum rather than producing infinities.
    let error_unit = match strategy {
        NormalizationStrategy::MaxNorm => finite_max(plans, |m| m.numerical_error),
        NormalizationStrategy::ThresholdRelative { epsilon } if epsilon.is_finite() && epsilon > 0.0 => {
            Some(epsilon)
        }
        NormalizationStrategy::ThresholdRelative { .. } => finite_max(plans, |m| m.numerical_error),
    };

    plans
        .iter()
        .map(|p| MetricVector {
            latency_ms: scale(p.metrics.latency_ms, max_latency),
            peak_memory_mb: scale(p.metrics.peak_memory_mb, max_memory),
            sync_overhead_ms: scale(p.metrics.sync_overhead_ms, max_sync),
            energy_mj: scale(p.metrics.energy_mj, max_energy),
            numerical_error: scale(p.metrics.numerical_error, error_unit),
        })
        .collect()
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

    fn plan_with(m: MetricVector) -> ExecutionPlan {
        ExecutionPlan { metrics: m, ..Default::default() }
    }

    /// Happy path: each dimension is divided by that dimension's own largest
    /// candidate value, so the worst candidate scores exactly 1.0 there.
    #[test]
    fn max_norm_divides_each_dimension_by_its_own_candidate_maximum() {
        let plans = vec![
            plan_with(MetricVector {
                latency_ms: 25.0,
                peak_memory_mb: 400.0,
                sync_overhead_ms: 1.0,
                energy_mj: 50.0,
                numerical_error: 0.0,
            }),
            plan_with(MetricVector {
                latency_ms: 100.0,
                peak_memory_mb: 100.0,
                sync_overhead_ms: 4.0,
                energy_mj: 200.0,
                numerical_error: 2e-6,
            }),
        ];
        let out = normalize(&plans, NormalizationStrategy::MaxNorm);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].latency_ms, 0.25);
        assert_eq!(out[0].peak_memory_mb, 1.0, "400 is the max, so it is the unit");
        assert_eq!(out[1].latency_ms, 1.0);
        assert_eq!(out[1].peak_memory_mb, 0.25);
        assert_eq!(out[1].sync_overhead_ms, 1.0);
        assert_eq!(out[1].energy_mj, 1.0);
        // Finding 2's effect: a tiny absolute error still normalizes to 1.0.
        assert_eq!(out[1].numerical_error, 1.0);
        assert_eq!(out[0].numerical_error, 0.0);
    }

    /// `ThresholdRelative` re-bases only `m₅`, onto the constraint tolerance:
    /// an error at exactly `ε` scores 1.0 no matter what the other candidates
    /// happen to score, which is the whole point of the alternative.
    #[test]
    fn threshold_relative_scales_error_by_epsilon_not_by_the_candidate_max() {
        let plans = vec![
            plan_with(MetricVector { numerical_error: 1e-6, latency_ms: 10.0, ..Default::default() }),
            plan_with(MetricVector { numerical_error: 2e-6, latency_ms: 20.0, ..Default::default() }),
        ];
        let out = normalize(&plans, NormalizationStrategy::ThresholdRelative { epsilon: 1e-5 });
        // Approximate: `1e-6 / 1e-5` is 0.09999999999999999 in IEEE 754, not
        // an exact 0.1. The ratios below (0.5, 1.0, 0.25) *are* exact in
        // binary, so those stay exact comparisons.
        assert!(
            (out[0].numerical_error - 0.1).abs() < 1e-12,
            "1e-6 is a tenth of the tolerance, got {}",
            out[0].numerical_error
        );
        assert!((out[1].numerical_error - 0.2).abs() < 1e-12);
        // Under MaxNorm the same pair would be 0.5 / 1.0 — the strategies must
        // actually differ, otherwise the enum is decorative.
        let max_norm = normalize(&plans, NormalizationStrategy::MaxNorm);
        assert_eq!(max_norm[0].numerical_error, 0.5);
        assert_eq!(max_norm[1].numerical_error, 1.0);
        // Dimensions other than m₅ are max-normalized under both.
        assert_eq!(out[1].latency_ms, 1.0);
    }

    /// Degenerate input: a dimension where every candidate scores `0.0` has no
    /// usable unit. It must normalize to `0.0`, never `0.0/0.0 = NaN` — a
    /// single `NaN` makes every later comparison `false` and breaks `argmin`
    /// silently, the same hazard [`WeightVector::cost`] guards against.
    #[test]
    fn all_zero_dimension_normalizes_to_zero_rather_than_nan() {
        let plans = vec![
            plan_with(MetricVector { latency_ms: 5.0, ..Default::default() }),
            plan_with(MetricVector { latency_ms: 10.0, ..Default::default() }),
        ];
        let out = normalize(&plans, NormalizationStrategy::MaxNorm);
        for v in &out {
            assert!(!v.energy_mj.is_nan(), "an all-zero dimension must not produce NaN");
            assert_eq!(v.energy_mj, 0.0);
            assert!(!v.numerical_error.is_nan());
        }
    }

    /// Undetermined (`INFINITY`) must survive normalization as Undetermined,
    /// and must not become the unit that finite candidates are measured
    /// against — otherwise one unmeasured candidate would silently rescale
    /// every other candidate's score in that dimension to ~0.
    #[test]
    fn undetermined_survives_normalization_and_is_not_used_as_the_unit() {
        let plans = vec![
            plan_with(MetricVector { latency_ms: f64::INFINITY, ..Default::default() }),
            plan_with(MetricVector { latency_ms: 40.0, ..Default::default() }),
            plan_with(MetricVector { latency_ms: 10.0, ..Default::default() }),
        ];
        let out = normalize(&plans, NormalizationStrategy::MaxNorm);
        assert!(out[0].latency_ms.is_infinite(), "Undetermined must stay Undetermined");
        assert!(!out[0].latency_ms.is_nan(), "INFINITY/INFINITY would be NaN — must not happen");
        assert_eq!(out[1].latency_ms, 1.0, "40 is the largest *finite* value, so it is the unit");
        assert_eq!(out[2].latency_ms, 0.25);
    }

    /// A non-positive `epsilon` is not a usable unit (it would divide by zero
    /// or flip the sign), so the error dimension falls back to the candidate
    /// maximum instead of emitting infinities.
    #[test]
    fn non_positive_epsilon_falls_back_to_max_norm_for_the_error_dimension() {
        let plans = vec![
            plan_with(MetricVector { numerical_error: 1e-6, ..Default::default() }),
            plan_with(MetricVector { numerical_error: 4e-6, ..Default::default() }),
        ];
        let out = normalize(&plans, NormalizationStrategy::ThresholdRelative { epsilon: 0.0 });
        assert!(out.iter().all(|v| v.numerical_error.is_finite()));
        assert_eq!(out[0].numerical_error, 0.25);
        assert_eq!(out[1].numerical_error, 1.0);
    }

    #[test]
    fn normalize_of_no_candidates_is_empty() {
        assert!(normalize(&[], NormalizationStrategy::MaxNorm).is_empty());
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
