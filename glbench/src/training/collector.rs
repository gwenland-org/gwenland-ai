//! [`VLStepCollector`] — glbench's implementation of stumman's `StepObserver`.
//!
//! # D-05: glbench observes, it does not drive
//!
//! This type is passive. It receives what `Trainer::train_step` hands it and
//! writes it down. It never calls `train_step`, never touches optimizer state,
//! and never writes a parameter — the boundary `glbench/DESIGN.md` §1 draws
//! between measuring and doing.
//!
//! # Why the results come back through an `Rc`, not through `clear_observer`
//!
//! `Trainer::set_observer` takes `Box<dyn StepObserver>` and `clear_observer`
//! hands the same trait object back. A trait object cannot be downcast without
//! `Any`, so **the returned box is not a way to read the results.** The
//! collector therefore writes into an `Rc<RefCell<…>>` the caller also holds.
//!
//! This is not a guess: `stumman/tests/observer_boundary.rs` demonstrates the
//! pattern from outside the crate specifically so Wave 4 would not have to
//! discover it at implementation time.
//!
//! # D-19 sampling
//!
//! With `--step-sample N`, steps where `index % N == 0` are archived, **plus
//! the first and last unconditionally**. Time-to-target, plateau detection and
//! stability all read the endpoints, and a run whose last step does not land on
//! a multiple of `N` would otherwise lose the one step that says how training
//! ended. `steps_observed` is counted separately from `steps_archived`, so a
//! consumer can never mistake a thinned series for a complete one.

use std::cell::RefCell;
use std::rc::Rc;

use crate::numerical::bitprof;
use crate::numerical::scope::{ENBitScope, VLBitScope};
use crate::training::step::VLTrainingStep;

/// Everything a collector accumulates, shared with whoever installed it.
#[derive(Debug, Default)]
pub struct VLCollected {
    /// Archived steps, after D-19 sampling.
    pub steps: Vec<VLTrainingStep>,
    /// Steps the observer was called for, before sampling.
    pub steps_observed: usize,
    /// Bit profiles of gradients and optimizer state, when a bit scope asked
    /// for them, each tagged with the step it came from.
    pub bit_profiles: Vec<VLStepBitProfile>,
    /// Total optimizer-state elements seen, for the memory record. `None` when
    /// the tensor payload was never requested — distinct from zero.
    pub optimizer_state_elements: Option<usize>,

    /// The most recent step, held back until it is known not to be the last.
    ///
    /// Lives in the **shared** state rather than in [`VLStepCollector`] on
    /// purpose. `Trainer::clear_observer` hands back a `Box<dyn StepObserver>`,
    /// which cannot be downcast without `Any` — so anything the flush needs
    /// must be reachable from the handle, not from the collector.
    pending_last: Option<VLTrainingStep>,
    /// D-19 `N`, kept here for the same reason.
    sample_n: usize,
}

/// One tensor's bit profile, tagged with the step it was taken at.
///
/// The step index matters: gradients change every step, so a profile without
/// one is a measurement of an unknown moment. Weights need no such tag, which
/// is why [`VLBitScope`] does not carry it — this pairing is training-only.
#[derive(Debug, Clone)]
pub struct VLStepBitProfile {
    /// The training step this profile was taken at.
    pub step_index: usize,
    /// The profiled tensor.
    pub scope: VLBitScope,
}

/// A shared handle to a collector's results.
pub type VLCollectedHandle = Rc<RefCell<VLCollected>>;

/// Collects training steps into [`VLCollected`].
pub struct VLStepCollector {
    out: VLCollectedHandle,
    /// Which tensor families to bit-profile, if any.
    bit_scopes: Vec<ENBitScope>,
}

impl VLStepCollector {
    /// Build a collector and return it with the handle to read results from.
    ///
    /// `sample_n` of 0 is treated as 1: archiving nothing is never what a
    /// caller means, and erroring here would push the check into every call
    /// site.
    pub fn new(sample_n: usize, bit_scopes: Vec<ENBitScope>) -> (VLStepCollector, VLCollectedHandle) {
        let out: VLCollectedHandle = Rc::new(RefCell::new(VLCollected {
            sample_n: sample_n.max(1),
            ..VLCollected::default()
        }));
        let collector = VLStepCollector { out: Rc::clone(&out), bit_scopes };
        (collector, out)
    }

    /// Whether this collector wants the O(n) tensor payload.
    fn wants_tensors_inner(&self) -> bool {
        self.bit_scopes
            .iter()
            .any(|s| matches!(s, ENBitScope::Gradients | ENBitScope::Optimizer))
    }
}

impl stumman::StepObserver for VLStepCollector {
    fn on_step(&mut self, step: &stumman::VLTrainingStep) {
        let archived = VLTrainingStep::from(step);
        let mut out = self.out.borrow_mut();
        out.steps_observed += 1;

        // The endpoint rule needs one step of lookahead: a step is only known
        // to be the last one once no further step arrives. So every step is
        // held back one round, and whatever is still pending at the end is the
        // last step — flushed by `finish`.
        let sample_n = out.sample_n;
        if let Some(previous) = out.pending_last.take() {
            if previous.index % sample_n == 0 {
                out.steps.push(previous);
            }
        }
        out.pending_last = Some(archived);
    }

    fn wants_tensors(&self) -> bool {
        self.wants_tensors_inner()
    }

    fn on_tensors(
        &mut self,
        grads: &stumman::VLGradStore,
        opt_state: &[stumman::VLNamedTensor],
    ) {
        let mut out = self.out.borrow_mut();

        // `on_tensors` fires after `on_step`, so the step just delivered is the
        // one still pending. Its index is what the payload belongs to.
        let step_index = match out.pending_last.as_ref() {
            Some(step) => step.index,
            None => return,
        };

        // The optimizer-state footprint is a scalar and is recorded every step
        // regardless of sampling — it is one number, not a payload.
        let elements: usize = opt_state.iter().map(|t| t.data.len()).sum();
        *out.optimizer_state_elements.get_or_insert(0) = elements;

        // Profiles follow the same D-19 sampling as the steps. Without this a
        // 192-step run archived 2,688 profiles and a 1.1 MB file while
        // `--step-sample` thinned nothing — measured, not hypothetical.
        //
        // The endpoint rule deliberately does NOT extend here: knowing a step
        // is the last one needs a lookahead, and holding a full tensor payload
        // back for a round costs far more than the one extra profile is worth.
        if step_index % out.sample_n != 0 {
            return;
        }

        if self.bit_scopes.contains(&ENBitScope::Gradients) {
            // `VLGradStore::iter` documents its order as unspecified — it is a
            // HashMap. Sorting by id makes the profile list reproducible across
            // runs, which a consumer diffing two archives depends on.
            let mut entries: Vec<_> = grads.iter().collect();
            entries.sort_by_key(|(id, _, _)| format!("{id:?}"));
            for (id, data, _shape) in entries {
                out.bit_profiles.push(VLStepBitProfile {
                    step_index,
                    scope: VLBitScope {
                        scope: ENBitScope::Gradients,
                        tensor_name: format!("grad/{id:?}"),
                        profile: bitprof::profile(data),
                    },
                });
            }
        }

        if self.bit_scopes.contains(&ENBitScope::Optimizer) {
            for tensor in opt_state {
                out.bit_profiles.push(VLStepBitProfile {
                    step_index,
                    scope: VLBitScope {
                        scope: ENBitScope::Optimizer,
                        tensor_name: tensor.name.clone(),
                        profile: bitprof::profile(&tensor.data),
                    },
                });
            }
        }
    }
}

/// Flush the held-back final step into the results.
///
/// Must be called once training finishes. The collector cannot do it itself:
/// nothing tells an observer that a run has ended, so the last step is still
/// pending when `Trainer::train` returns.
///
/// Takes only the handle. That is the whole reason `pending_last` lives in the
/// shared state — `clear_observer` returns a trait object this function could
/// not open.
pub fn finish(handle: &VLCollectedHandle) {
    let mut out = handle.borrow_mut();
    let Some(last) = out.pending_last.take() else {
        return;
    };
    // Unconditional: this is the endpoint D-19 always keeps. Guarded against a
    // double push when the last index also happened to be a multiple of N and
    // was already archived by the lookahead.
    if out.steps.last().map(|s| s.index) != Some(last.index) {
        out.steps.push(last);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use stumman::StepObserver as _;

    fn wire_step(index: usize, epoch: usize, loss: f32) -> stumman::VLTrainingStep {
        stumman::VLTrainingStep {
            index,
            epoch,
            loss,
            forward_ns: 1_000,
            backward_ns: 2_000,
            optimizer_ns: 500,
            total_ns: 3_600,
            grad_count: 2,
            grad_elements: 16,
            grad_l2_norm: 0.5,
            grad_nan: 0,
            grad_inf: 0,
            lr: 1e-3,
        }
    }

    /// Drive `count` steps through a collector and return the archived indices.
    fn archived_indices(count: usize, sample_n: usize) -> (Vec<usize>, usize) {
        let (mut collector, handle) = VLStepCollector::new(sample_n, Vec::new());
        for i in 0..count {
            collector.on_step(&wire_step(i, 0, 1.0 - i as f32 * 0.01));
        }
        drop(collector);
        finish(&handle);
        let out = handle.borrow();
        (out.steps.iter().map(|s| s.index).collect(), out.steps_observed)
    }

    #[test]
    fn sample_n_of_one_archives_every_step() {
        let (indices, observed) = archived_indices(5, 1);
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(observed, 5);
    }

    /// D-19's endpoint rule: 9 is not a multiple of 4, and must be kept anyway.
    #[test]
    fn thinning_keeps_the_multiples_plus_both_endpoints() {
        let (indices, observed) = archived_indices(10, 4);
        assert_eq!(indices, vec![0, 4, 8, 9]);
        assert_eq!(observed, 10, "observed counts every step, not the archived ones");
    }

    #[test]
    fn the_last_step_is_not_archived_twice_when_it_lands_on_a_multiple() {
        // 9 steps, N=4: indices 0,4,8 are multiples and 8 is also the last.
        let (indices, _) = archived_indices(9, 4);
        assert_eq!(indices, vec![0, 4, 8], "8 must appear once, not twice");
    }

    #[test]
    fn a_single_step_run_archives_that_step() {
        let (indices, observed) = archived_indices(1, 10);
        assert_eq!(indices, vec![0]);
        assert_eq!(observed, 1);
    }

    #[test]
    fn a_run_with_no_steps_archives_nothing_and_does_not_panic() {
        let (indices, observed) = archived_indices(0, 4);
        assert!(indices.is_empty());
        assert_eq!(observed, 0);
    }

    /// A thinned series must never be mistakable for a complete one.
    #[test]
    fn observed_and_archived_counts_diverge_under_sampling() {
        let (indices, observed) = archived_indices(100, 10);
        assert_eq!(observed, 100);
        assert!(indices.len() < observed, "sampling must actually thin");
        assert_eq!(indices.first(), Some(&0));
        assert_eq!(indices.last(), Some(&99), "the endpoint survives thinning");
    }

    /// Zero would archive nothing, which is never what a caller means.
    #[test]
    fn a_sample_n_of_zero_is_treated_as_one() {
        let (indices, _) = archived_indices(3, 0);
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[test]
    fn the_tensor_payload_is_requested_only_for_the_training_scopes() {
        let (weights_only, _) = VLStepCollector::new(1, vec![ENBitScope::Weights]);
        assert!(
            !weights_only.wants_tensors(),
            "weights come from the package, not from the step"
        );

        let (none, _) = VLStepCollector::new(1, Vec::new());
        assert!(!none.wants_tensors());

        for scope in [ENBitScope::Gradients, ENBitScope::Optimizer] {
            let (collector, _) = VLStepCollector::new(1, vec![scope]);
            assert!(collector.wants_tensors(), "{scope:?} needs the payload");
        }
    }

    /// Regression: bit profiles used to ignore `--step-sample` entirely.
    ///
    /// Measured before the fix: a 192-step run with `--bit-scope
    /// gradients,optimizer` archived 2,688 profiles into a 1.1 MB file while
    /// `--step-sample` thinned nothing. Profiles now follow the same `N` as the
    /// steps.
    #[test]
    fn bit_profiles_follow_the_same_sampling_as_the_steps() {
        use stumman::{VLGradStore, VLNamedTensor};

        let (mut collector, handle) =
            VLStepCollector::new(4, vec![ENBitScope::Optimizer]);
        let state = vec![VLNamedTensor {
            name: "lora_a.m".to_string(),
            data: vec![0.1, 0.2, 0.3],
            shape: vec![3],
        }];

        for i in 0..10 {
            collector.on_step(&wire_step(i, 0, 1.0));
            collector.on_tensors(&VLGradStore::new(), &state);
        }
        finish(&handle);

        let out = handle.borrow();
        let profiled: Vec<usize> = out.bit_profiles.iter().map(|p| p.step_index).collect();
        assert_eq!(
            profiled,
            vec![0, 4, 8],
            "one profile per sampled step, not one per step"
        );
        assert_eq!(out.steps_observed, 10, "every step is still observed");
    }

    /// The scalar footprint is one number, so it is recorded every step even
    /// when the payload itself is skipped.
    #[test]
    fn the_optimizer_footprint_is_recorded_even_on_unsampled_steps() {
        use stumman::{VLGradStore, VLNamedTensor};

        let (mut collector, handle) = VLStepCollector::new(100, vec![ENBitScope::Optimizer]);
        let state = vec![VLNamedTensor {
            name: "lora_a.m".to_string(),
            data: vec![0.0; 7],
            shape: vec![7],
        }];
        // Step 1 is not a multiple of 100, so no profile is taken.
        collector.on_step(&wire_step(1, 0, 1.0));
        collector.on_tensors(&VLGradStore::new(), &state);
        finish(&handle);

        let out = handle.borrow();
        assert!(out.bit_profiles.is_empty(), "the payload was skipped");
        assert_eq!(
            out.optimizer_state_elements,
            Some(7),
            "but the footprint is a scalar and is still known"
        );
    }

    #[test]
    fn the_archived_step_carries_the_values_the_wire_step_had() {
        let (mut collector, handle) = VLStepCollector::new(1, Vec::new());
        collector.on_step(&wire_step(7, 2, 0.25));
        drop(collector);
        finish(&handle);

        let out = handle.borrow();
        let step = &out.steps[0];
        assert_eq!(step.index, 7);
        assert_eq!(step.epoch, 2);
        assert!((step.loss - 0.25).abs() < 1e-6);
        assert_eq!(step.grad_count, 2);
        assert!((step.lr - 1e-3).abs() < 1e-12);
    }
}
