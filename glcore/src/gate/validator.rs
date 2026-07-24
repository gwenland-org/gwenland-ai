//! `Validator` — the conjunction of registered constraints (paper §4.3,
//! `V(P) = ∏Cᵢ(P)`) with early-exit (paper §5 Step 2).

use std::fmt;

use crate::gate::constraint::{Constraint, ValidationResult};
use crate::gate::plan::ExecutionPlan;

/// A validator: an ordered set of constraints applied conjunctively with
/// early-exit. See `architecture/GATE/GATE-constraints.md` for the default
/// ordering rationale (cheapest, most-rejective first).
#[derive(Default)]
pub struct Validator {
    constraints: Vec<Box<dyn Constraint>>,
}

impl Validator {
    /// An empty validator — accepts every plan (`V(P) = 1` vacuously, the
    /// empty product).
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a constraint to the chain. Order matters: constraints run in
    /// registration order (paper §6.3's default order is caller
    /// discipline, not enforced here — see `architecture/GATE/GATE-mapping.md`
    /// Gap 4).
    pub fn register(&mut self, constraint: Box<dyn Constraint>) {
        self.constraints.push(constraint);
    }

    /// Evaluate `V(P) = ∏Cᵢ(P)` with early-exit: the first `Reject` short-
    /// circuits the remaining checks (paper §5 Step 2).
    pub fn validate(&self, plan: &ExecutionPlan) -> ValidationResult {
        for constraint in &self.constraints {
            let result = constraint.validate(plan);
            if matches!(result, ValidationResult::Reject { .. }) {
                return result;
            }
        }
        ValidationResult::Pass
    }
}

impl fmt::Debug for Validator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Validator").field("constraints", &self.constraints.len()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AlwaysReject;
    impl Constraint for AlwaysReject {
        fn validate(&self, _plan: &ExecutionPlan) -> ValidationResult {
            ValidationResult::Reject { reason: "always rejects".to_string() }
        }
        fn name(&self) -> &'static str {
            "AlwaysReject"
        }
    }

    #[test]
    fn validator_empty_accepts_any_plan() {
        let validator = Validator::new();
        let plan = ExecutionPlan::default();
        assert_eq!(validator.validate(&plan), ValidationResult::Pass);
    }

    #[test]
    fn validator_single_reject_constraint_rejects() {
        let mut validator = Validator::new();
        validator.register(Box::new(AlwaysReject));
        let plan = ExecutionPlan::default();
        match validator.validate(&plan) {
            ValidationResult::Reject { reason } => assert_eq!(reason, "always rejects"),
            ValidationResult::Pass => panic!("expected rejection"),
        }
    }
}
