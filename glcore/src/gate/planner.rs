//! `Planner` — orchestrates candidate generation and cost-minimal
//! selection (paper §5 Steps 1, 5, 6; Algorithm 1).

use crate::gate::constraint::ValidationResult;
use crate::gate::error::GateError;
use crate::gate::metrics::CostEvaluator;
use crate::gate::plan::{ExecutionPlan, TensorGraph};
use crate::gate::policy::ExecutionPolicy;
use crate::gate::validator::Validator;

/// Produces real `ExecutionPlan` candidates for a `TensorGraph`'s ops.
///
/// `glcore::gate` cannot itself know which backend-specific strategies
/// (weight formats, kernel choices, ...) exist for a given op — only the
/// backend crate that implements them does, the same reason concrete
/// `Constraint`s live in backend crates rather than here (see
/// `architecture/GATE/GATE-constraints.md`'s Composability section). A
/// `CandidateSource` is that seam: a backend supplies one, `Planner`
/// drives it, and `glcore::gate` stays protocol-only.
pub trait CandidateSource {
    /// Every candidate `ExecutionPlan` this source can produce for `op`,
    /// each already carrying its own `metrics` (paper's `m(P)`, produced
    /// however the backend obtains it — measured or estimated; this trait
    /// takes no position on which).
    fn candidates_for(&self, op: &crate::gate::plan::TensorOp) -> Vec<ExecutionPlan>;
}

/// The orchestration component implementing Algorithm 1 end-to-end (paper
/// §8, "Planner"): generates candidates, validates them, and selects the
/// cost-minimal valid one.
#[derive(Debug)]
pub struct Planner {
    validator: Validator,
    cost_evaluator: CostEvaluator,
    /// Candidate bound: default 8, maximum 64 (paper §5 Step 1).
    num_candidates: usize,
}

impl Planner {
    /// Construct a planner from a validator, cost evaluator, and candidate
    /// bound (paper's reference interfaces, §5.2: `Planner::new(...)`).
    pub fn new(validator: Validator, cost_evaluator: CostEvaluator, num_candidates: usize) -> Self {
        Planner { validator, cost_evaluator, num_candidates }
    }

    /// Generate up to `num_candidates` execution plans for `graph`'s ops,
    /// by asking `source` for each op's candidates in turn (paper §5 Step
    /// 1). `policy` is accepted per the paper's reference signature but
    /// unused here: candidate *generation* enumerates strategies, it does
    /// not weigh them — weighing happens at Step 5 (`select_best`), via
    /// whatever `WeightVector` the caller built `cost_evaluator` from.
    ///
    /// Truncates to `num_candidates` (the paper's own bound, §5 Step 1) —
    /// a graph or source that offers more than the bound is truncated, not
    /// rejected, since Step 1 exists to keep Step 2's validation work
    /// bounded, not to enforce a hard error on an over-generous source.
    pub fn generate_candidates(
        &self,
        graph: &TensorGraph,
        _policy: ExecutionPolicy,
        source: &dyn CandidateSource,
    ) -> Vec<ExecutionPlan> {
        let mut candidates = Vec::new();
        for op in &graph.ops {
            candidates.extend(source.candidates_for(op));
            if candidates.len() >= self.num_candidates {
                break;
            }
        }
        candidates.truncate(self.num_candidates);
        candidates
    }

    /// Validate every candidate and select the cost-minimal valid plan
    /// (paper §5 Steps 2-6; Theorem 10.3, optimality). Returns
    /// [`GateError::NoValidPlan`] if `𝒫valid = ∅` (paper §4.6).
    pub fn select_best(
        &self,
        candidates: &mut Vec<ExecutionPlan>,
    ) -> Result<ExecutionPlan, GateError> {
        let candidates_tried = candidates.len();
        let mut best: Option<(ExecutionPlan, f64)> = None;
        for plan in candidates.drain(..) {
            match self.validator.validate(&plan) {
                ValidationResult::Pass => {
                    let cost = self.cost_evaluator.evaluate(&plan);
                    if best.as_ref().is_none_or(|(_, best_cost)| cost < *best_cost) {
                        best = Some((plan, cost));
                    }
                }
                ValidationResult::Reject { .. } => {}
            }
        }
        best.map(|(plan, _)| plan)
            .ok_or(GateError::NoValidPlan { candidates_tried })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate::metrics::{MetricVector, WeightVector};
    use crate::gate::plan::{BackendKind, TensorOp};

    /// Two candidates per op: a "fast" and a "slow" plan, distinguishable
    /// only by `latency_ms` — enough to exercise real `argmin` selection
    /// without inventing backend-specific format logic in a test.
    struct FastSlowSource;
    impl CandidateSource for FastSlowSource {
        fn candidates_for(&self, op: &TensorOp) -> Vec<ExecutionPlan> {
            let make = |latency_ms: f64| ExecutionPlan {
                ordering: vec![op.clone()],
                backend: BackendKind::Glproc,
                layouts: Default::default(),
                metrics: MetricVector { latency_ms, ..Default::default() },
            };
            vec![make(10.0), make(5.0)]
        }
    }

    fn latency_only_evaluator() -> CostEvaluator {
        CostEvaluator::new(WeightVector { weights: [1.0, 0.0, 0.0, 0.0, 0.0] })
    }

    #[test]
    fn generate_candidates_asks_source_per_op() {
        let planner = Planner::new(Validator::new(), latency_only_evaluator(), 8);
        let graph = TensorGraph { ops: vec![TensorOp::new("a"), TensorOp::new("b")] };
        let candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        assert_eq!(candidates.len(), 4, "2 ops x 2 candidates each");
    }

    #[test]
    fn generate_candidates_truncates_to_bound() {
        let planner = Planner::new(Validator::new(), latency_only_evaluator(), 3);
        let graph = TensorGraph { ops: vec![TensorOp::new("a"), TensorOp::new("b")] };
        let candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        assert_eq!(candidates.len(), 3, "bounded by num_candidates");
    }

    #[test]
    fn select_best_picks_minimum_cost_among_valid() {
        let planner = Planner::new(Validator::new(), latency_only_evaluator(), 8);
        let graph = TensorGraph { ops: vec![TensorOp::new("a")] };
        let mut candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        let best = planner.select_best(&mut candidates).unwrap();
        assert_eq!(best.metrics.latency_ms, 5.0, "must pick the lower-latency candidate");
    }

    #[test]
    fn select_best_skips_rejected_candidates() {
        struct RejectFast;
        impl crate::gate::constraint::Constraint for RejectFast {
            fn validate(&self, plan: &ExecutionPlan) -> ValidationResult {
                if plan.metrics.latency_ms < 6.0 {
                    ValidationResult::Reject { reason: "too fast to trust".into() }
                } else {
                    ValidationResult::Pass
                }
            }
            fn name(&self) -> &'static str {
                "RejectFast"
            }
        }
        let mut validator = Validator::new();
        validator.register(Box::new(RejectFast));
        let planner = Planner::new(validator, latency_only_evaluator(), 8);
        let graph = TensorGraph { ops: vec![TensorOp::new("a")] };
        let mut candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        let best = planner.select_best(&mut candidates).unwrap();
        assert_eq!(best.metrics.latency_ms, 10.0, "the 5.0ms candidate was rejected");
    }

    #[test]
    fn select_best_errors_when_every_candidate_rejected() {
        struct AlwaysReject;
        impl crate::gate::constraint::Constraint for AlwaysReject {
            fn validate(&self, _plan: &ExecutionPlan) -> ValidationResult {
                ValidationResult::Reject { reason: "always rejects".into() }
            }
            fn name(&self) -> &'static str {
                "AlwaysReject"
            }
        }
        let mut validator = Validator::new();
        validator.register(Box::new(AlwaysReject));
        let planner = Planner::new(validator, latency_only_evaluator(), 8);
        let graph = TensorGraph { ops: vec![TensorOp::new("a")] };
        let mut candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        let candidates_tried = candidates.len();
        match planner.select_best(&mut candidates) {
            Err(GateError::NoValidPlan { candidates_tried: n }) => {
                assert_eq!(n, candidates_tried)
            }
            other => panic!("expected NoValidPlan, got {other:?}"),
        }
    }

    #[test]
    fn empty_graph_yields_no_candidates() {
        let planner = Planner::new(Validator::new(), latency_only_evaluator(), 8);
        let graph = TensorGraph::default();
        let candidates =
            planner.generate_candidates(&graph, ExecutionPolicy::Balanced, &FastSlowSource);
        assert!(candidates.is_empty());
    }
}
