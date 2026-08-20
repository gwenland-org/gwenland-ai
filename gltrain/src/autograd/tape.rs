//! Stummañ Kevskrid — Autograd tape.
//!
//! The Tape records computation nodes during the forward pass in
//! append-only order. Reversing this list gives a valid backward
//! order (topological sort is implicit for define-by-run graphs).
//!
//! # Wave 2 scope
//! - `push()`: record a node
//! - `len()`: query recorded node count
//! - `op_names()` / `node_ids()`: inspect recorded ops (for testing)
//! - `register_tensor()`: store metadata for a participating tensor
//! - `get_tensor_meta()`: retrieve shape of a recorded tensor by ID
//!
//! # Wave 3 will add
//! - `backward()`: reverse traversal + gradient computation
//! - `grad_store`: a map from TensorId to accumulated gradient

use crate::autograd::grad_store::VLGradStore;
use crate::autograd::node::{ComputationNode, NodeId, TensorId};
use crate::error::{GlTrainError, Result};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, MutexGuard};

/// Lightweight tensor metadata stored in the tape.
///
/// We store the shape (needed for backward shape checks) but NOT the
/// storage — that stays in the `Tensor`, owned by the user. The tape
/// therefore never keeps tensor data alive.
#[derive(Debug, Clone)]
pub struct TensorMeta {
    /// The tensor this metadata describes.
    pub id: TensorId,
    /// Its shape at the time it was registered.
    pub shape: Vec<usize>,
    /// Whether it was tracked for gradients.
    pub requires_grad: bool,
}

/// The autograd tape — append-only record of the forward pass.
///
/// # Thread safety
/// Tape is wrapped in `Arc<Mutex<Tape>>` when shared across tensor ops.
/// The Mutex ensures sequential append during forward pass.
/// In practice, training is single-threaded in Wave 1–3; the Mutex
/// overhead is negligible.
///
/// # Memory
/// - `nodes`: Vec grows during forward pass, cleared between steps
/// - `tensors`: map of [`TensorMeta`] (shape only, no storage)
///
/// # Not every input ID resolves (KL-003)
///
/// A node's `inputs` list every operand of the op, but only *tracked* tensors
/// are ever registered. Calling [`Tape::get_tensor_meta`] on an input ID can
/// therefore return `None`, and that is normal.
///
/// **A `None` input ID means a frozen/untracked operand: no gradient is
/// computed for it, and this is not an error.** It is the ordinary LoRA
/// shape — a frozen base weight consumed by a tracked activation. Wave 3's
/// `backward()` must treat an unresolvable input as a place to stop
/// propagating, never as a failure; erroring there would break the primary
/// training path.
///
/// # All operands of one op share one tape (KL-002)
///
/// A tensor op rejects operands carrying different tapes, so every input ID on
/// a node either belongs to this tape or belongs to no tape at all. A node can
/// never reference a tensor that is live on some *other* tape.
pub struct Tape {
    /// Recorded computation nodes in forward-pass order.
    /// Backward pass reverses this vec (Wave 3).
    nodes: Vec<ComputationNode>,

    /// Tensor metadata indexed by TensorId.
    /// Populated when tensors are registered via `register_tensor()`.
    tensors: HashMap<TensorId, TensorMeta>,

    /// Monotonically increasing node ID counter.
    next_node_id: NodeId,

    /// Gradients accumulated by the most recent `backward()`.
    grad_store: VLGradStore,
}

impl Tape {
    /// Lock a shared tape, recovering the guard if the mutex was poisoned.
    ///
    /// A tape is almost always held as `Arc<Mutex<Tape>>`, so this is an
    /// associated function taking that handle rather than a `&self` method:
    /// you would need the lock already to call a method on the tape itself.
    ///
    /// ```
    /// use std::sync::{Arc, Mutex};
    /// use gltrain::Tape;
    ///
    /// let tape = Arc::new(Mutex::new(Tape::new()));
    /// let guard = Tape::lock(&tape);
    /// assert!(guard.is_empty());
    /// ```
    ///
    /// # Why it never fails
    ///
    /// `Mutex::lock` only errors when another thread panicked while holding
    /// the lock. A tape is a `Vec` plus a `HashMap` with no cross-field
    /// invariant a partial write could break, so reclaiming the guard is sound
    /// and strictly better than the alternatives: this crate forbids `unwrap`
    /// outside tests, and making every caller write
    /// `unwrap_or_else(|p| p.into_inner())` by hand just moves the same
    /// decision somewhere it will eventually be got wrong.
    ///
    /// A poisoned tape does mean some earlier forward pass died halfway, so
    /// the recorded graph may be incomplete. It will not be corrupt.
    pub fn lock(shared: &Arc<Mutex<Tape>>) -> MutexGuard<'_, Tape> {
        shared
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            tensors: HashMap::new(),
            next_node_id: 0,
            grad_store: VLGradStore::new(),
        }
    }

    /// Append a computation node to the tape.
    /// Called by tensor ops (matmul, add, ...) during the forward pass.
    pub fn push(&mut self, node: ComputationNode) {
        self.nodes.push(node);
    }

    /// Allocate and return the next node ID (monotonically increasing).
    pub fn next_node_id(&mut self) -> NodeId {
        let id = self.next_node_id;
        self.next_node_id += 1;
        id
    }

    /// Number of recorded nodes (used in tests to verify recording).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if no nodes have been recorded yet.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Register tensor metadata so the tape can reference it by ID.
    /// Called when a new tensor is created with `requires_grad = true`
    /// or when it participates in a tracked op.
    pub fn register_tensor(&mut self, meta: TensorMeta) {
        self.tensors.insert(meta.id, meta);
    }

    /// Retrieve tensor metadata by ID.
    /// Returns None if the tensor was not registered (e.g. a no-grad tensor).
    pub fn get_tensor_meta(&self, id: TensorId) -> Option<&TensorMeta> {
        self.tensors.get(&id)
    }

    /// Number of registered tensors. Lets a caller check registration
    /// bookkeeping without exposing the map itself.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Return the op names of all recorded nodes, in forward order.
    /// Used in tests to verify which ops were recorded.
    pub fn op_names(&self) -> Vec<&'static str> {
        self.nodes.iter().map(|n| n.op_name).collect()
    }

    /// Return all recorded node IDs in forward order.
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().map(|n| n.id).collect()
    }

    /// Read-only view of the recorded nodes, in forward order.
    /// Wave 3's `backward()` walks this in reverse.
    pub fn nodes(&self) -> &[ComputationNode] {
        &self.nodes
    }

    /// Clear recorded nodes, tensor metadata and gradients.
    /// Called between training steps, after `optimizer.step()`.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.tensors.clear();
        self.next_node_id = 0;
        self.grad_store.clear();
    }

    // ── Backward pass ────────────────────────────────────────────────────

    /// Run the backward pass over everything recorded so far.
    ///
    /// Nodes are walked in reverse. Forward-order append gives topological
    /// order for a define-by-run graph, so reversing it is a valid backward
    /// order with no sort needed.
    ///
    /// For each node: read the output gradient out of the store, call the
    /// node's [`crate::autograd::node::BackwardFn`], and accumulate each
    /// returned input gradient. A `None` at index `i` means input `i` is
    /// frozen and is skipped, per KL-003. That is not an error.
    ///
    /// # The seed
    ///
    /// The last recorded node's output is seeded with all ones, so `backward()`
    /// computes the gradient of `sum(output)`. When the graph ends in a scalar,
    /// which is the loss case, shape `[1]` makes that seed exactly `[1.0]` and
    /// the usual dL/dL = 1 falls out. Seeding a single `1.0` regardless of
    /// shape would be wrong for every non-scalar tail: the buffer length has to
    /// match the output it belongs to or the first backward closure reads past
    /// its end.
    ///
    /// # One backward pass per gradient store
    ///
    /// Returns [`GlTrainError::InvalidOp`] if gradients are already present.
    /// Calling this twice in a row does not double the gradient, it triples it:
    /// the seed accumulates on top of itself and then propagates, measured as
    /// `[1,1,1,1]` after one call and `[3,3,3,3]` after two. Rather than leave
    /// that as a footgun, a dirty store is rejected. Call
    /// [`Tape::zero_grad`] first.
    ///
    /// Multi-batch gradient accumulation is a real need and is Wave 4's job.
    /// It gets an explicit entry point rather than falling out of calling this
    /// twice by accident.
    ///
    /// Read results with [`Tape::grad`]. Reset with [`Tape::zero_grad`].
    pub fn backward(&mut self) -> Result<()> {
        if !self.grad_store.is_empty() {
            return Err(GlTrainError::InvalidOp(
                "call zero_grad() before backward()".into(),
            ));
        }
        if self.nodes.is_empty() {
            return Ok(());
        }

        // Seed the final output with ones.
        let last_output = self.nodes[self.nodes.len() - 1].output;
        let seed_shape = self
            .tensors
            .get(&last_output)
            .ok_or_else(|| {
                GlTrainError::InvalidOp(format!(
                    "output tensor {last_output} not registered in tape"
                ))
            })?
            .shape
            .clone();
        let seed_len: usize = seed_shape.iter().product();
        self.grad_store
            .accumulate(last_output, vec![1.0f32; seed_len], seed_shape)?;

        for node in self.nodes.iter().rev() {
            // No gradient reached this node, so nothing downstream needs it.
            let Some((grad_output, output_shape)) = self.grad_store.get(node.output).cloned()
            else {
                continue;
            };

            let input_grads = (node.backward_fn)(&grad_output, &output_shape)?;

            if input_grads.len() != node.inputs.len() {
                return Err(GlTrainError::InvalidOp(format!(
                    "op '{}' returned {} gradients but the node has {} inputs",
                    node.op_name,
                    input_grads.len(),
                    node.inputs.len()
                )));
            }

            for (input_id, maybe_grad) in node.inputs.iter().zip(input_grads) {
                // KL-003: None means a frozen operand. Skip it.
                let Some((grad_data, grad_shape)) = maybe_grad else {
                    continue;
                };
                self.grad_store
                    .accumulate(*input_id, grad_data, grad_shape)?;
            }
        }

        Ok(())
    }

    /// Read a gradient accumulated by [`Tape::backward`].
    /// `None` means the tensor never received one, which is what a frozen
    /// operand looks like.
    pub fn grad(&self, id: TensorId) -> Option<&(Vec<f32>, Vec<usize>)> {
        self.grad_store.get(id)
    }

    /// Borrow the whole gradient store.
    pub fn grad_store(&self) -> &VLGradStore {
        &self.grad_store
    }

    /// Drop all gradients but keep the recorded graph.
    /// Call after the optimizer step.
    pub fn zero_grad(&mut self) {
        self.grad_store.clear();
    }

    /// End a training step: take the accumulated gradients and clear the tape,
    /// in one call.
    ///
    /// # This is the KL-006 guard
    ///
    /// `matmul`/`mul`/`relu`'s backward closures snapshot their forward
    /// operands at record time. `KNOWN_ISSUES.md`'s recommended fix (option a)
    /// is "require `tape.clear()` before `optimizer.step()`", enforced the way
    /// KL-005 guards double-backward.
    ///
    /// A `&mut Tape` check cannot enforce that by itself: the gradients the
    /// optimizer needs *live in* `grad_store`, so a guard that simply refused a
    /// non-empty tape would make it impossible to ever read them. This method
    /// resolves that by doing both atomically: the returned [`VLGradStore`] is
    /// the only way to get gradients out, and by the time the caller holds it,
    /// `self` is already fully cleared. There is no ordering for a caller to
    /// get wrong, and no separate boolean to check.
    ///
    /// [`crate::optim::Optimizer::step`] takes the result of this call, never
    /// the tape itself, which is what makes weight mutation impossible to
    /// perform against a still-live tape.
    pub fn finish_step(&mut self) -> VLGradStore {
        let grads = std::mem::take(&mut self.grad_store);
        self.nodes.clear();
        self.tensors.clear();
        self.next_node_id = 0;
        grads
    }
}

impl Default for Tape {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Tape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tape")
            .field("nodes", &self.nodes)
            .field("tensor_count", &self.tensors.len())
            .field("next_node_id", &self.next_node_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::node::ComputationNode;
    use std::sync::Arc;

    /// A node whose backward returns one `None` per input: structurally valid,
    /// but contributes no gradient. These tests only exercise recording.
    fn dummy_node(
        id: usize,
        op_name: &'static str,
        inputs: Vec<usize>,
        output: usize,
    ) -> ComputationNode {
        let arity = inputs.len();
        ComputationNode {
            id,
            op_name,
            inputs,
            output,
            backward_fn: Arc::new(move |_: &[f32], _: &[usize]| Ok(vec![None; arity])),
        }
    }

    #[test]
    fn new_tape_is_empty() {
        let tape = Tape::new();
        assert_eq!(tape.len(), 0);
        assert!(tape.is_empty());
    }

    #[test]
    fn push_single_node_increments_len() {
        let mut tape = Tape::new();
        tape.push(dummy_node(0, "Matmul", vec![0, 1], 2));
        assert_eq!(tape.len(), 1);
        assert!(!tape.is_empty());
    }

    #[test]
    fn push_multiple_nodes_records_in_order() {
        let mut tape = Tape::new();
        tape.push(dummy_node(0, "Matmul", vec![0, 1], 2));
        tape.push(dummy_node(1, "Add", vec![2, 3], 4));
        tape.push(dummy_node(2, "ReLU", vec![4], 5));
        assert_eq!(tape.len(), 3);
        assert_eq!(tape.op_names(), vec!["Matmul", "Add", "ReLU"]);
    }

    #[test]
    fn next_node_id_is_monotonically_increasing() {
        let mut tape = Tape::new();
        let id0 = tape.next_node_id();
        let id1 = tape.next_node_id();
        let id2 = tape.next_node_id();
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);
    }

    #[test]
    fn register_and_retrieve_tensor_meta() {
        let mut tape = Tape::new();
        tape.register_tensor(TensorMeta {
            id: 42,
            shape: vec![4, 8],
            requires_grad: true,
        });
        let meta = tape
            .get_tensor_meta(42)
            .expect("tensor 42 should be registered");
        assert_eq!(meta.shape, vec![4, 8]);
        assert!(meta.requires_grad);
    }

    #[test]
    fn clear_resets_tape_to_empty() {
        let mut tape = Tape::new();
        tape.push(dummy_node(0, "Matmul", vec![0, 1], 2));
        tape.register_tensor(TensorMeta {
            id: 0,
            shape: vec![2, 2],
            requires_grad: true,
        });
        tape.clear();
        assert_eq!(tape.len(), 0);
        assert!(tape.get_tensor_meta(0).is_none());
    }

    // ── M2 Wave 1: finish_step, the KL-006 resolution ────────────────────

    /// The property KL-006 needs: after `finish_step` the caller holds the
    /// gradients AND the tape is empty, so there is no window in which an
    /// in-place weight update could invalidate a live backward closure.
    #[test]
    fn finish_step_clears_the_tape_and_returns_the_gradients() {
        use crate::backend::GlProc;
        use crate::tensor::Tensor;
        use std::sync::Mutex;

        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])
            .unwrap()
            .with_grad(tape.clone());
        let x_id = x.id();
        let w = Tensor::<GlProc>::ones(&[2, 2]).unwrap();
        let _y = x.matmul(&w).unwrap();

        let grads = {
            let mut guard = Tape::lock(&tape);
            guard.backward().unwrap();
            assert!(
                !guard.is_empty(),
                "the tape must have nodes before finish_step"
            );
            guard.finish_step()
        };

        // The gradients survived the clear.
        assert!(
            grads.contains(x_id),
            "finish_step dropped the gradient it was supposed to hand over"
        );

        // The tape did not.
        let guard = Tape::lock(&tape);
        assert_eq!(guard.len(), 0, "nodes must be gone");
        assert_eq!(guard.tensor_count(), 0, "tensor registrations must be gone");
        assert!(guard.grad_store().is_empty(), "grad store must be gone");
    }

    /// `finish_step` without a `backward()` yields nothing rather than
    /// failing. An optimizer stepping on an empty store simply skips every
    /// parameter, which is the correct no-op.
    #[test]
    fn finish_step_returns_an_empty_store_when_backward_was_never_called() {
        let mut tape = Tape::new();
        tape.push(dummy_node(0, "Noop", vec![], 0));
        let grads = tape.finish_step();
        assert!(grads.is_empty(), "no backward ran, so there are no gradients");
        assert_eq!(tape.len(), 0, "the tape is still cleared");
    }

    /// Node IDs restart after `finish_step`, so a second training step records
    /// against the same ID space rather than growing it forever.
    #[test]
    fn finish_step_resets_the_node_id_counter() {
        let mut tape = Tape::new();
        let first = tape.next_node_id();
        tape.push(dummy_node(first, "Noop", vec![], 0));
        let _ = tape.finish_step();
        assert_eq!(
            tape.next_node_id(),
            first,
            "IDs must restart, or a long run counts up without bound"
        );
    }
}
