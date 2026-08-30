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

    /// Every gradient in the store: id, values, shape.
    ///
    /// Added for the M2.5 observer, which holds a `&VLGradStore` and otherwise
    /// has no way to reach anything: every other accessor needs a `TensorId`
    /// the caller already knows.
    ///
    /// Iteration order is **unspecified**. The backing store is a `HashMap`, so
    /// the order changes between runs. A consumer that needs a stable order
    /// sorts by `TensorId` itself. This matters more than it looks: a consumer
    /// that quietly assumes insertion order gets a plausible wrong answer on
    /// some runs and the right one on others.
    pub fn iter(&self) -> impl Iterator<Item = (TensorId, &[f32], &[usize])> {
        self.grads
            .iter()
            .map(|(id, (data, shape))| (*id, data.as_slice(), shape.as_slice()))
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

    /// M2.5: `iter` is the observer's only way in, so it has to reach every
    /// entry. Counting alone would pass on an iterator that yielded the same
    /// entry three times, so the ids are checked as a set too.
    #[test]
    fn iter_yields_every_entry_exactly_once() {
        let mut store = VLGradStore::new();
        store.accumulate(3, vec![1.0, 2.0], vec![2]).unwrap();
        store.accumulate(1, vec![3.0], vec![1]).unwrap();
        store.accumulate(9, vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let mut seen: Vec<TensorId> = store.iter().map(|(id, _, _)| id).collect();
        assert_eq!(seen.len(), store.len(), "iter must yield len() entries");
        seen.sort_unstable();
        assert_eq!(seen, vec![1, 3, 9], "iter must reach every accumulated id");

        // Data and shape must arrive intact, not just the keys.
        let total: usize = store.iter().map(|(_, data, _)| data.len()).sum();
        assert_eq!(total, 6, "iter must expose every gradient element");
        for (id, data, shape) in store.iter() {
            let (want_data, want_shape) = store.get(id).unwrap();
            assert_eq!(data, want_data.as_slice(), "id {id}: data disagrees with get");
            assert_eq!(shape, want_shape.as_slice(), "id {id}: shape disagrees with get");
        }
    }

    /// An empty store yields nothing rather than panicking. The observer calls
    /// `iter` unconditionally on any observed step, including one where a
    /// frozen-only forward produced no gradients at all.
    #[test]
    fn iter_on_an_empty_store_yields_nothing() {
        let store = VLGradStore::new();
        assert_eq!(store.iter().count(), 0);
    }
}
