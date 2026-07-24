//! `Dispatcher` — executes the selected plan on its target backend (paper
//! §5 Step 7; §8 "Dispatcher").

use crate::gate::error::GateError;
use crate::gate::plan::ExecutionPlan;

/// Executes a validated, cost-minimal [`ExecutionPlan`] and returns its
/// output. Paper §8: "allocates device memory per the plan's layout
/// assignments, schedules kernels, manages transfers ... translating
/// abstract plan operations into concrete backend API calls."
#[derive(Debug, Clone, Copy, Default)]
pub struct Dispatcher;

impl Dispatcher {
    /// Construct a dispatcher (paper's reference interfaces, §5.2:
    /// `Dispatcher::new()`).
    pub fn new() -> Self {
        Self
    }

    /// Execute `plan` on its target backend (paper §5 Step 7). Stub: no
    /// bridge from `ExecutionPlan` to a real `Box<dyn GlEngine>` call
    /// exists yet — see `architecture/GATE/GATE-mapping.md` §5
    /// (Integration Points).
    pub fn dispatch(&self, _plan: &ExecutionPlan) -> Result<ExecutionResult, GateError> {
        todo!()
    }
}

/// The result of dispatching a plan — paper's reference interfaces'
/// `ExecutionResult` (§5.1/§5.2). Marker stub only, matching `TensorOp`
/// (`plan.rs`): nothing populates this yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ExecutionResult;
