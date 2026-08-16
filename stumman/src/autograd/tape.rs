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

use crate::autograd::node::{ComputationNode, NodeId, TensorId};
use std::collections::HashMap;

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
}

impl Tape {
    /// Create a new empty tape.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            tensors: HashMap::new(),
            next_node_id: 0,
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

    /// Clear all recorded nodes and tensor metadata.
    /// Called between training steps (after `optimizer.step()`).
    /// Wave 3: also clears the gradient store.
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.tensors.clear();
        self.next_node_id = 0;
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

    fn dummy_node(
        id: usize,
        op_name: &'static str,
        inputs: Vec<usize>,
        output: usize,
    ) -> ComputationNode {
        ComputationNode {
            id,
            op_name,
            inputs,
            output,
            backward_fn: Arc::new(|| ()),
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
}
