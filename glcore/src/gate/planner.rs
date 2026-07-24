//! `Planner` — orchestrates candidate generation and cost-minimal
//! selection (paper §5 Steps 1, 5, 6; Algorithm 1).

use crate::gate::error::GateError;
use crate::gate::metrics::CostEvaluator;
use crate::gate::plan::{ExecutionPlan, TensorGraph};
use crate::gate::policy::ExecutionPolicy;
use crate::gate::validator::Validator;

/// The orchestration component implementing Algorithm 1 end-to-end (paper
/// §8, "Planner"): generates candidates, validates them, and selects the
/// cost-minimal valid one.
///
/// Fields go unread until `generate_candidates`/`select_best` are
/// implemented for real (see `architecture/GATE/GATE-mapping.md` Gap 1) —
/// `#[allow(dead_code)]` documents that as intentional stub state, not an
/// oversight (rustc does not count a derived `Debug` impl as a field read).
#[derive(Debug)]
#[allow(dead_code)]
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

    /// Generate up to `num_candidates` execution plans for `graph` under
    /// `policy` (paper §5 Step 1). Stub: needs a real [`TensorGraph`] with
    /// actual topology to enumerate strategies over — see
    /// `architecture/GATE/GATE-mapping.md` Gap 1.
    pub fn generate_candidates(
        &self,
        _graph: &TensorGraph,
        _policy: ExecutionPolicy,
    ) -> Vec<ExecutionPlan> {
        todo!()
    }

    /// Validate every candidate and select the cost-minimal valid plan
    /// (paper §5 Steps 2-6; Theorem 10.3, optimality). Returns
    /// [`GateError::NoValidPlan`] if `𝒫valid = ∅` (paper §4.6).
    pub fn select_best(
        &self,
        _candidates: &mut Vec<ExecutionPlan>,
    ) -> Result<ExecutionPlan, GateError> {
        todo!()
    }
}
