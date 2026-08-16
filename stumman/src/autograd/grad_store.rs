//! Stummañ Kevskrid: gradient accumulator.
//!
//! `VLGradStore` holds gradients keyed by `TensorId`. It accumulates: calling
//! `accumulate` twice for the same ID adds the gradients together, which is
//! what a tensor used in more than one op needs (a shared weight, or `x` in
//! `x.matmul(&x)`).
//!
//! Gradient data is `Vec<f32>`, not `B::Storage`, so this type stays
//! backend-agnostic and `Tape` never has to become generic. See
//! [`crate::autograd::node::BackwardFn`] for why that matters.

use crate::autograd::node::TensorId;
use crate::error::{GlTrainError, Result};
use std::collections::HashMap;

/// Maps `TensorId` to accumulated gradient data plus its shape.
///
/// Accumulation rule: a tensor that receives gradients from several paths gets
/// them summed. That is the correct rule for shared parameters, and it is why
/// `accumulate` adds instead of overwriting.
#[derive(Debug, Default)]
pub struct VLGradStore {
    grads: HashMap<TensorId, (Vec<f32>, Vec<usize>)>,
}

impl VLGradStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Accumulate a gradient for `id`, adding to whatever is already there.
    ///
    /// Returns [`GlTrainError::ShapeMismatch`] if the incoming gradient has a
    /// different element count than the one already stored. Silently summing
    /// mismatched buffers would produce a plausible-looking wrong gradient.
    pub fn accumulate(
        &mut self,
        id: TensorId,
        grad_data: Vec<f32>,
        shape: Vec<usize>,
    ) -> Result<()> {
        match self.grads.get_mut(&id) {
            Some((existing, existing_shape)) => {
                if existing.len() != grad_data.len() {
                    return Err(GlTrainError::ShapeMismatch {
                        expected: existing_shape.clone(),
                        got: shape,
                    });
                }
                for (acc, incoming) in existing.iter_mut().zip(&grad_data) {
                    *acc += incoming;
                }
            }
            None => {
                self.grads.insert(id, (grad_data, shape));
            }
        }
        Ok(())
    }

    /// Borrow the gradient for `id`, if one was accumulated.
    pub fn get(&self, id: TensorId) -> Option<&(Vec<f32>, Vec<usize>)> {
        self.grads.get(&id)
    }

    /// Remove and return the gradient for `id`. The optimizer uses this.
    pub fn take(&mut self, id: TensorId) -> Option<(Vec<f32>, Vec<usize>)> {
        self.grads.remove(&id)
    }

    /// How many tensors currently have a gradient.
    pub fn len(&self) -> usize {
        self.grads.len()
    }

    /// True when no gradients have been accumulated.
    pub fn is_empty(&self) -> bool {
        self.grads.is_empty()
    }

    /// Drop every gradient. Call after the optimizer step.
    pub fn clear(&mut self) {
        self.grads.clear();
    }

    /// Whether `id` has a gradient.
    pub fn contains(&self, id: TensorId) -> bool {
        self.grads.contains_key(&id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Gradients are exact small sums here, so any drift is a real bug.
    const TOL_GRAD: f32 = 1e-6;

    #[test]
    fn new_store_is_empty() {
        let store = VLGradStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(!store.contains(0));
    }

    #[test]
    fn first_accumulate_inserts() {
        let mut store = VLGradStore::new();
        store.accumulate(7, vec![1.0, 2.0], vec![2]).unwrap();
        let (data, shape) = store.get(7).expect("id 7 must be present");
        assert_eq!(shape, &vec![2]);
        assert!((data[0] - 1.0).abs() < TOL_GRAD);
        assert!((data[1] - 2.0).abs() < TOL_GRAD);
    }

    /// The whole reason this type exists: a tensor reached from two paths must
    /// end up with the sum, not the last write.
    #[test]
    fn second_accumulate_sums_instead_of_overwriting() {
        let mut store = VLGradStore::new();
        store.accumulate(7, vec![1.0, 2.0], vec![2]).unwrap();
        store.accumulate(7, vec![10.0, 20.0], vec![2]).unwrap();
        let (data, _) = store.get(7).expect("id 7 must be present");
        assert!((data[0] - 11.0).abs() < TOL_GRAD, "got {}", data[0]);
        assert!((data[1] - 22.0).abs() < TOL_GRAD, "got {}", data[1]);
        assert_eq!(store.len(), 1, "accumulation must not add an entry");
    }

    #[test]
    fn accumulate_rejects_length_mismatch() {
        let mut store = VLGradStore::new();
        store.accumulate(7, vec![1.0, 2.0], vec![2]).unwrap();
        let err = store.accumulate(7, vec![1.0], vec![1]);
        assert!(err.is_err(), "mismatched gradient length must be rejected");
    }

    #[test]
    fn take_removes_the_entry() {
        let mut store = VLGradStore::new();
        store.accumulate(7, vec![1.0], vec![1]).unwrap();
        assert!(store.take(7).is_some());
        assert!(store.take(7).is_none(), "take must consume the entry");
        assert!(store.is_empty());
    }

    #[test]
    fn clear_drops_everything() {
        let mut store = VLGradStore::new();
        store.accumulate(1, vec![1.0], vec![1]).unwrap();
        store.accumulate(2, vec![1.0], vec![1]).unwrap();
        store.clear();
        assert!(store.is_empty());
    }
}
