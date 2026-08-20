//! Stummañ Deskiñ: the training observer layer.
//!
//! Records what a training step did, without changing what it does. The whole
//! sub-system is one struct and one trait.
//!
//! # Why nothing here is generic over `B`
//!
//! KL-001 says `Backend` is not dyn-compatible, so a trait carrying `B` could
//! not be boxed. It does not have to: the two types that actually cross this
//! boundary are already backend-free. [`VLGradStore`] holds `Vec<f32>` plus a
//! shape, and [`VLNamedTensor`] holds the same. So [`StepObserver`] takes no
//! type parameter, `Box<dyn StepObserver>` compiles, and KL-001 never reaches
//! this file.
//!
//! # Where the observer runs, and why it is after the update
//!
//! [`crate::train::Trainer::train_step`] calls the observer **after**
//! `optimizer.step()`, not between `finish_step()` and it. Two reasons:
//!
//! 1. `optimizer_ns` cannot be reported by a callback that runs before the
//!    optimizer does. A step record delivered early would have to carry a
//!    zero or a lie in that field.
//! 2. Reading gradients late is safe. `grads` is a local owned by
//!    `train_step`, and `Optimizer::step` borrows it shared, so it is still
//!    fully alive afterwards.
//!
//! KL-006 is untouched either way: its guarantee is that the tape is empty
//! before any weight write, and `finish_step()` has already emptied it well
//! before the observer is called. The observer never sees a tape at all.

use crate::autograd::grad_store::VLGradStore;
use crate::optim::VLNamedTensor;

/// One training step, as plain facts.
///
/// `VL` because it is data with derived traits and nothing else. Every field is
/// a scalar or a count, which is what keeps [`StepObserver`] object-safe.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTrainingStep {
    /// Global step index, zero-based. The value `Trainer::step_count` had
    /// before this step incremented it.
    pub index: usize,
    /// Epoch this step belongs to. Zero for a bare `train_step` call, since
    /// only `Trainer::train` runs epochs.
    pub epoch: usize,

    /// The loss `train_step` returns. The same value, not a recomputation.
    pub loss: f32,

    /// Nanoseconds in the forward pass.
    pub forward_ns: u64,
    /// Nanoseconds in backward, including `finish_step`.
    pub backward_ns: u64,
    /// Nanoseconds in the optimizer update.
    pub optimizer_ns: u64,
    /// Nanoseconds from the top of the step to the end of the update.
    ///
    /// Deliberately excludes the observer's own cost: it is measured before
    /// the observer window opens, so installing an observer does not inflate
    /// the number the observer reports.
    pub total_ns: u64,

    /// How many **trainable parameters** received a gradient this step.
    ///
    /// Not the size of the gradient store. `Tape::finish_step` returns a
    /// gradient for every tensor the tape touched, activations included: on the
    /// M2 trainer that is nine entries for two parameters. Every field below
    /// counts parameters only, for the reason given on `grad_l2_norm`. An
    /// observer that wants the activations too reads them from the
    /// [`VLGradStore`] handed to [`StepObserver::on_tensors`].
    pub grad_count: usize,
    /// Total gradient elements across those parameters.
    pub grad_elements: usize,
    /// Global L2 norm over the parameter gradients, accumulated in f64.
    ///
    /// Parameters only, deliberately. This is the quantity every framework's
    /// `clip_grad_norm_` computes and the one gradient-health analysis reads;
    /// mixing activation gradients in would produce a number that moves with
    /// the shape of the graph rather than with the health of the update.
    ///
    /// Non-finite elements are excluded from the sum and counted in `grad_nan`
    /// / `grad_inf` instead. One NaN would otherwise make the whole norm NaN
    /// and destroy the only signal the field carries.
    pub grad_l2_norm: f64,
    /// NaN gradient elements. Usually 0/0.
    pub grad_nan: usize,
    /// Infinite gradient elements. Usually overflow. Counted separately from
    /// `grad_nan` because the two mean different things.
    pub grad_inf: usize,

    /// Base learning rate at this step, before any group multiplier. Read from
    /// the optimizer, not from the trainer config, so it stays correct when
    /// M3 adds scheduling.
    pub lr: f64,
}

/// Watches a training run, step by step.
///
/// Traits take no prefix (naming rule 2). Object-safe by construction: no type
/// parameter, no `Self`-by-value receiver, no `Clone` supertrait.
///
/// Every method is called from inside `Trainer::train_step`. An implementation
/// that panics takes the run down with it, which is deliberate: an observer
/// that swallows its own failure is worse than one that stops.
pub trait StepObserver {
    /// Called once per step, after the parameter update has been applied.
    fn on_step(&mut self, step: &VLTrainingStep);

    /// Whether this observer wants the O(n) tensor payload.
    ///
    /// Default `false` so the expensive path is opt-in. When this returns
    /// false, `Trainer` does not call `Optimizer::state_tensors`, which
    /// allocates a full copy of the optimizer state.
    fn wants_tensors(&self) -> bool {
        false
    }

    /// Gradients and optimizer state, as flat f32.
    ///
    /// Called after [`StepObserver::on_step`], and only when
    /// [`StepObserver::wants_tensors`] returned true. Both borrows last
    /// exactly this call.
    fn on_tensors(&mut self, _grads: &VLGradStore, _opt_state: &[VLNamedTensor]) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    /// If `StepObserver` ever loses object safety, this fails to compile.
    /// `Trainer` stores `Box<dyn StepObserver>`, so that would be a hard break
    /// rather than a style regression. KL-001 is the reason it is worth an
    /// explicit guard: the sibling `Backend` trait is *not* dyn-compatible, and
    /// the difference is easy to erase by accident.
    #[test]
    fn step_observer_is_object_safe() {
        fn _assert_object_safe(_: Box<dyn StepObserver>) {}
        fn _assert_ref_safe(_: &dyn StepObserver) {}
    }

    /// A NaN and an Inf in the same store must land in different counters. A
    /// single `is_finite()` check would collapse them, and the two point at
    /// different bugs: NaN is usually 0/0, Inf is usually overflow.
    #[test]
    fn nan_and_inf_are_separate_fields_on_the_record() {
        let step = VLTrainingStep {
            index: 0,
            epoch: 0,
            loss: 1.0,
            forward_ns: 0,
            backward_ns: 0,
            optimizer_ns: 0,
            total_ns: 0,
            grad_count: 1,
            grad_elements: 2,
            grad_l2_norm: 0.0,
            grad_nan: 1,
            grad_inf: 1,
            lr: 1e-3,
        };
        assert_eq!(step.grad_nan, 1);
        assert_eq!(step.grad_inf, 1);
    }
}
