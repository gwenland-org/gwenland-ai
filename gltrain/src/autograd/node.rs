//! Stummañ Kevskrid — Computation graph node.
//!
//! A ComputationNode records one operation in the forward pass:
//! which tensors went in, which tensor came out, and the backward
//! function to call during the replay pass (Wave 3).
//!
//! `Tape::backward()` replays them in reverse recording order.

use crate::error::Result;
use std::sync::Arc;

/// Unique identifier for a tensor within the tape.
/// Monotonically increasing; assigned at tensor creation.
/// Stable across moves (unlike raw pointers).
pub type TensorId = usize;

/// Unique identifier for a computation node within the tape.
pub type NodeId = usize;

/// One input's gradient: the data plus the shape it belongs to.
///
/// `None` means the input is frozen and wants no gradient (KL-003).
pub type InputGrad = Option<(Vec<f32>, Vec<usize>)>;

/// A node's backward function: given the gradient flowing into the op's
/// output, produce the gradient for each of its inputs.
///
/// Arguments are `grad_output` (dL/d(output), flattened row-major) and
/// `output_shape`. The return has one entry per [`ComputationNode::inputs`]
/// entry, in the same order.
///
/// # Why `Vec<f32>` instead of tensors
///
/// The plan's signature, `Box<dyn Fn(&Tensor, &Tape) -> Result<Vec<Tensor>>>`,
/// cannot be written: `Tensor<B>` is generic over its backend, so a tape
/// holding one would become `Tape<B>` and could never span a mixed-backend
/// graph, which is exactly what M4 needs.
///
/// Raw `f32` buffers are the common currency that sidesteps it. Every backward
/// closure captures the forward values it needs at record time and returns
/// plain buffers, so nothing in `autograd/` ever names a backend type. The
/// cost is that backward math runs on the scalar helpers in
/// [`crate::autograd::ops`] rather than dispatching to AVX2.
///
/// # Frozen inputs
///
/// Returning `None` at index `i` says input `i` is frozen: no gradient is
/// computed and none is accumulated. That is normal, not an error. It is the
/// LoRA shape, a frozen base weight consumed by a trainable activation.
pub type BackwardFn = Arc<dyn Fn(&[f32], &[usize]) -> Result<Vec<InputGrad>> + Send + Sync>;

/// One recorded operation in the forward pass.
///
/// # Memory model
/// - `inputs`: TensorIds of operands (cheap copy, no Arc clone)
/// - `output`: TensorId of result tensor
/// - `backward_fn`: heap-allocated closure capturing input TensorIds
///   and any data needed to compute gradients. Stored as a boxed trait
///   object for polymorphism — no giant match statement on op type.
///
/// `Tape::backward()` calls these in reverse recording order.
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

    /// Backward function for this op. See [`BackwardFn`].
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
