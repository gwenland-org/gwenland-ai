//! Stummañ Kevskrid — Computation graph node.
//!
//! A ComputationNode records one operation in the forward pass:
//! which tensors went in, which tensor came out, and the backward
//! function to call during the replay pass (Wave 3).
//!
//! Wave 2: nodes are recorded but backward_fn is never called yet.
//! Wave 3 will add `Tape::backward()` that replays them in reverse.

use std::sync::Arc;

/// Unique identifier for a tensor within the tape.
/// Monotonically increasing; assigned at tensor creation.
/// Stable across moves (unlike raw pointers).
pub type TensorId = usize;

/// Unique identifier for a computation node within the tape.
pub type NodeId = usize;

/// Placeholder type for a node's backward function.
///
/// # This is deliberately inert in Wave 2
///
/// The plan doc's signature, `Box<dyn Fn(&Tensor, &Tape) -> Result<Vec<Tensor>>>`,
/// cannot be written as-is: `Tensor<B>` is generic over its backend, but
/// [`crate::autograd::tape::Tape`] must stay backend-agnostic or every tape
/// becomes generic too — and a single tape could then never hold nodes from
/// mixed-backend graphs.
///
/// Wave 2 does not solve that. It stores a closure that takes nothing and
/// returns nothing, purely so the field has a type that compiles and can be
/// moved into the tape. **Nothing calls it.** Wave 3 owns the real design.
pub type BackwardFn = Arc<dyn Fn() + Send + Sync>;

/// One recorded operation in the forward pass.
///
/// # Memory model
/// - `inputs`: TensorIds of operands (cheap copy, no Arc clone)
/// - `output`: TensorId of result tensor
/// - `backward_fn`: heap-allocated closure capturing input TensorIds
///   and any data needed to compute gradients. Stored as a boxed trait
///   object for polymorphism — no giant match statement on op type.
///
/// # Wave 2 note
/// `backward_fn` is stored but never called in Wave 2.
/// Wave 3 adds `Tape::backward()` which calls these in reverse order.
pub struct ComputationNode {
    /// Unique node ID (monotonically increasing within this tape).
    pub id: NodeId,

    /// Human-readable op name for debugging and gate reports.
    /// Uses `&'static str` (string literal), not `String`.
    /// Examples: "Matmul", "Add", "ReLU", "Transpose"
    pub op_name: &'static str,

    /// TensorIds of all input tensors to this op (in argument order).
    pub inputs: Vec<TensorId>,

    /// TensorId of the output tensor produced by this op.
    pub output: TensorId,

    /// Backward function. See [`BackwardFn`] — inert placeholder in Wave 2,
    /// redesigned in Wave 3 when something finally calls it.
    pub backward_fn: BackwardFn,
}

impl std::fmt::Debug for ComputationNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComputationNode")
            .field("id", &self.id)
            .field("op_name", &self.op_name)
            .field("inputs", &self.inputs)
            .field("output", &self.output)
            .field("backward_fn", &"<fn>")
            .finish()
    }
}
