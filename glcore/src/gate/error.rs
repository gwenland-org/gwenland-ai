//! `GateError` — GATE's own error type (paper §5 Step 4, §4.6). Kept
//! separate from [`crate::error::GlError`] — see
//! `architecture/GATE/GATE-mapping.md` Gap 2 for why.

use std::fmt;

/// The error GATE raises when a plan cannot be validated or dispatched.
#[derive(Debug, Clone)]
pub enum GateError {
    /// A candidate plan violated a registered constraint (paper §5 Step 3).
    ConstraintViolation {
        /// The name of the constraint that rejected the plan.
        constraint: &'static str,
        /// The specific reason the plan was rejected.
        reason: String,
        /// Index of the rejected plan within the candidate set.
        plan_index: usize,
    },
    /// No candidate plan survived validation — `𝒫valid = ∅` (paper §4.6,
    /// the `⊥` case of the dispatch partial function).
    NoValidPlan {
        /// How many candidates were generated and tried.
        candidates_tried: usize,
    },
    /// The dispatcher failed to execute an otherwise-valid plan.
    BackendError(String),
}

impl fmt::Display for GateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GateError::ConstraintViolation { constraint, reason, plan_index } => write!(
                f,
                "constraint violation: plan {plan_index} rejected by `{constraint}`: {reason}"
            ),
            GateError::NoValidPlan { candidates_tried } => {
                write!(f, "no valid plan found among {candidates_tried} candidates")
            }
            GateError::BackendError(msg) => write!(f, "backend error: {msg}"),
        }
    }
}

impl std::error::Error for GateError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_error_is_display() {
        let e = GateError::NoValidPlan { candidates_tried: 8 };
        assert_eq!(e.to_string(), "no valid plan found among 8 candidates");
    }
}
