//! The `Constraint` trait — GATE's atomic correctness check (paper §4.3,
//! §6). See `architecture/GATE/GATE-concepts.md` for the full mapping.

use crate::gate::plan::ExecutionPlan;

/// A logical constraint over execution plans — paper's `C: 𝒫 → {0,1}`
/// (Definition, §4.3). Concrete constraints (`ShapeConstraint`,
/// `MemoryConstraint`, ...) live in each backend crate, not here — see
/// `architecture/GATE/GATE-constraints.md`'s Composability section.
pub trait Constraint: Send + Sync {
    /// Evaluate this constraint against a candidate plan.
    fn validate(&self, plan: &ExecutionPlan) -> ValidationResult;
    /// The constraint's name, used in rejection diagnostics (paper §5
    /// Step 3).
    fn name(&self) -> &'static str;
}

/// The outcome of one constraint check — the Rust encoding of the paper's
/// `{0, 1}` codomain, extended with a rejection reason (paper §5 Step 3:
/// "rejected plans are logged with ... the specific reason").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationResult {
    /// The constraint accepted the plan (`C(P) = 1`).
    Pass,
    /// The constraint rejected the plan (`C(P) = 0`), with a diagnostic.
    Reject {
        /// Why the plan was rejected.
        reason: String,
    },
}
