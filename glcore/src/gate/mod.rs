//! GATE — Gwen Algorithm for Tensor Execution: protocol boilerplate.
//!
//! See `architecture/GATE/README.md` for what GATE is/is not and its
//! relationship to `gl-agent-skills/architecture-skills/gate-integration.md`'s
//! runtime gates, and `architecture/GATE/GATE-mapping.md` for exactly what
//! in this module is real logic vs. a stub. Every type here is either pure
//! protocol data (a struct/enum with no compute) or orchestration control
//! flow ([`Validator::validate`]'s early-exit loop,
//! [`ExecutionPolicy::weight_vector`]'s table lookup) fully specified by
//! the paper's own definitions — nothing under `glcore::gate` performs
//! inference compute, and nothing here is wired into any engine yet.

pub mod constraint;
pub mod dispatcher;
pub mod error;
pub mod metrics;
pub mod plan;
pub mod planner;
pub mod policy;
pub mod validator;

pub use constraint::{Constraint, ValidationResult};
pub use dispatcher::{Dispatcher, ExecutionResult};
pub use error::GateError;
pub use metrics::{normalize, CostEvaluator, MetricVector, NormalizationStrategy, WeightVector};
pub use plan::{BackendKind, ExecutionPlan, MemoryLayout, OpId, TensorGraph, TensorOp};
pub use planner::Planner;
pub use policy::ExecutionPolicy;
pub use validator::Validator;
