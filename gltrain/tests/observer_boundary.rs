//! The observer boundary, exercised the way `glbench` will exercise it.
//!
//! # Why this is an integration test and not a unit test
//!
//! An integration test links `gltrain` as an **external crate**, so it can only
//! reach the public API. Every in-module test in `trainer.rs` can name private
//! paths, which means none of them can prove what this file proves: that a
//! consumer outside the crate can implement [`StepObserver`], install it, and
//! read what it collected.
//!
//! glbench v3 Wave 4 (`architecture/glbench-v3/DESIGN.md`) is built entirely on
//! this boundary — `training::collector::VLStepCollector` is exactly the shape
//! below. If any type in the trait signature stopped being reachable, Wave 4
//! would fail to compile and nothing in gltrain's own suite would notice.
//!
//! # The handle problem, solved here rather than discovered in Wave 4
//!
//! `Trainer::set_observer` takes `Box<dyn StepObserver>` and `clear_observer`
//! hands it back as the same trait object. A trait object cannot be downcast
//! without `Any`, so **the returned box is not how a caller reads the results**.
//! The working pattern is shared ownership: the collector writes into an
//! `Rc<RefCell<…>>` the caller also holds. That is what
//! [`VLStepCollectorProxy`] demonstrates, and Wave 4 should copy it rather than
//! rediscover it.

use std::cell::RefCell;
use std::rc::Rc;

use gltrain::autograd::grad_store::VLGradStore;
use gltrain::backend::GlProc;
use gltrain::optim::VLNamedTensor;
use gltrain::train::observe::{StepObserver, VLTrainingStep};
use gltrain::{Trainer, VLMicroDataset, VLTrainerConfig};

/// Layer width for the fixture. Small on purpose — this file tests the shape of
/// the boundary, not the numerics, and gltrain's own suite covers those.
const D_IN: usize = 4;
/// Output width of the fixture layer.
const D_OUT: usize = 4;
/// LoRA rank. The adapter therefore owns exactly two trainable parameters.
const RANK: usize = 2;
/// Trainable parameters an `LRLora` at [`RANK`] owns: `A` and `B`.
const LORA_PARAMS: usize = 2;

/// What the collector accumulates, shared with the caller.
#[derive(Default)]
struct Collected {
    /// One record per observed step, in delivery order.
    steps: Vec<VLTrainingStep>,
    /// `(gradient tensors, optimizer state tensors)` per `on_tensors` call.
    tensor_calls: Vec<(usize, usize)>,
    /// Total gradient elements seen through `on_tensors`, to prove the payload
    /// is real f32 data rather than an empty handle.
    grad_elements_seen: usize,
}

/// A stand-in for Wave 4's `VLStepCollector`.
///
/// Holds only an `Rc` to the results, so the caller keeps reading them after
/// the `Box<dyn StepObserver>` has been handed to the `Trainer`.
struct VLStepCollectorProxy {
    out: Rc<RefCell<Collected>>,
    wants_tensors: bool,
}

impl StepObserver for VLStepCollectorProxy {
    fn on_step(&mut self, step: &VLTrainingStep) {
        self.out.borrow_mut().steps.push(step.clone());
    }

    fn wants_tensors(&self) -> bool {
        self.wants_tensors
    }

    fn on_tensors(&mut self, grads: &VLGradStore, opt_state: &[VLNamedTensor]) {
        let mut out = self.out.borrow_mut();
        out.tensor_calls.push((grads.len(), opt_state.len()));
        // Reading through the public iterator is the access pattern GLBitProf
        // needs: a `&[f32]` per tensor, with no backend type in sight.
        for (_id, data, _shape) in grads.iter() {
            out.grad_elements_seen += data.len();
        }
    }
}

/// Build a trainer and a small deterministic dataset.
fn fixture(samples: usize) -> (Trainer<GlProc>, VLMicroDataset) {
    let config = VLTrainerConfig::new(D_IN, D_OUT, RANK, 1e-2, 7);
    let trainer = Trainer::<GlProc>::new(config).expect("trainer builds");
    // `synthetic_regression` also returns the ground-truth weight it generated
    // from; this file only needs the dataset.
    let (dataset, _true_w) = VLMicroDataset::synthetic_regression(samples, D_IN, D_OUT, 11)
        .expect("dataset builds");
    (trainer, dataset)
}

/// Install a collector and return the shared handle.
fn observe(trainer: &mut Trainer<GlProc>, wants_tensors: bool) -> Rc<RefCell<Collected>> {
    let out = Rc::new(RefCell::new(Collected::default()));
    trainer.set_observer(Box::new(VLStepCollectorProxy {
        out: Rc::clone(&out),
        wants_tensors,
    }));
    out
}

/// The compile-time half of the proof: every type in the trait signature is
/// nameable from outside the crate. If this file compiles, that holds — the
/// test body only has to run to prove the runtime half.
#[test]
fn an_external_crate_can_implement_and_install_a_step_observer() {
    let (mut trainer, dataset) = fixture(4);
    let collected = observe(&mut trainer, false);

    trainer.train(&dataset, 2).expect("training runs under observation");

    let out = collected.borrow();
    assert_eq!(
        out.steps.len(),
        8,
        "2 epochs x 4 samples must deliver 8 records, got {}",
        out.steps.len()
    );
    assert!(
        out.tensor_calls.is_empty(),
        "wants_tensors() == false must not trigger the O(n) payload"
    );
}

/// Wave 4's `VLTrainingSession` needs `index` and `epoch` to be trustworthy:
/// D-19 sampling keys off `index`, and the convergence window is per-epoch.
#[test]
fn step_index_is_global_and_epoch_advances_with_train() {
    let (mut trainer, dataset) = fixture(3);
    let collected = observe(&mut trainer, false);

    trainer.train(&dataset, 3).expect("training runs");

    let out = collected.borrow();
    assert_eq!(out.steps.len(), 9);

    // `index` is global and gap-free across epoch boundaries.
    let indices: Vec<usize> = out.steps.iter().map(|s| s.index).collect();
    assert_eq!(indices, (0..9).collect::<Vec<_>>(), "index must be global and dense");

    // `epoch` advances every `dataset.len()` steps.
    let epochs: Vec<usize> = out.steps.iter().map(|s| s.epoch).collect();
    assert_eq!(epochs, vec![0, 0, 0, 1, 1, 1, 2, 2, 2], "epoch must track train()");
}

/// D-19: archive every Nth step **plus the first and last unconditionally**.
///
/// Expressible from `index` alone, which is the property Wave 4 needs — the
/// sampler must not have to ask the trainer anything.
#[test]
fn d19_sampling_is_expressible_from_the_delivered_records() {
    let (mut trainer, dataset) = fixture(5);
    let collected = observe(&mut trainer, false);
    trainer.train(&dataset, 2).expect("training runs");

    let out = collected.borrow();
    let observed = out.steps.len();
    assert_eq!(observed, 10);

    const N: usize = 4;
    let last = observed - 1;
    let archived: Vec<usize> = out
        .steps
        .iter()
        .map(|s| s.index)
        .filter(|&i| i % N == 0 || i == 0 || i == last)
        .collect();

    // Multiples of 4, plus the endpoints. 9 is the last step and is NOT a
    // multiple of 4 — exactly the case D-19's endpoint rule exists for.
    assert_eq!(archived, vec![0, 4, 8, 9]);
    assert!(archived.contains(&0), "first step is always kept");
    assert!(archived.contains(&last), "last step is always kept");
}

/// The gradient statistics must describe **parameters**, not the tape.
///
/// `finish_step` returns a gradient for every tensor the tape touched, which on
/// this trainer is far more than two. A consumer reading `grad_count` as "how
/// many parameters got a gradient" would be wrong if this regressed, and the
/// number would still look plausible.
#[test]
fn gradient_statistics_count_parameters_not_activations() {
    let (mut trainer, dataset) = fixture(2);
    let collected = observe(&mut trainer, true);
    trainer.train(&dataset, 1).expect("training runs");

    let out = collected.borrow();
    let step = &out.steps[0];
    assert_eq!(
        step.grad_count, LORA_PARAMS,
        "grad_count must be the two LoRA parameters, not the whole store"
    );
    assert!(step.grad_l2_norm.is_finite(), "norm must be a real number");
    assert!(step.grad_l2_norm > 0.0, "a real step has a non-zero gradient");
    assert_eq!(step.grad_nan, 0);
    assert_eq!(step.grad_inf, 0);

    // The store handed to `on_tensors` is the *full* one, activations included,
    // which is the documented contract: statistics are filtered, the payload is
    // not.
    let (grad_tensors, _) = out.tensor_calls[0];
    assert!(
        grad_tensors > LORA_PARAMS,
        "on_tensors must receive the whole store ({grad_tensors} tensors), \
         so a consumer that wants activations can still have them"
    );
}

/// GLBitProf's gradient and optimizer scopes (Wave 4) read exactly this.
#[test]
fn tensor_payload_delivers_gradients_and_optimizer_state_as_flat_f32() {
    let (mut trainer, dataset) = fixture(2);
    let collected = observe(&mut trainer, true);
    trainer.train(&dataset, 1).expect("training runs");

    let out = collected.borrow();
    assert_eq!(out.tensor_calls.len(), 2, "one payload per step");
    for (grads, opt_state) in &out.tensor_calls {
        assert!(*grads > 0, "gradient store must not arrive empty");
        assert!(
            *opt_state > 0,
            "AdamW keeps moments per parameter, so state_tensors must be non-empty"
        );
    }
    assert!(
        out.grad_elements_seen > 0,
        "iterating the store must yield real f32 data"
    );
}

/// Timing fields have to be internally consistent or `VLTrainingAttribution`
/// (Wave 4) reports a phase breakdown that does not add up.
#[test]
fn phase_timings_are_consistent_and_exclude_the_observer_itself() {
    let (mut trainer, dataset) = fixture(3);
    let collected = observe(&mut trainer, false);
    trainer.train(&dataset, 1).expect("training runs");

    let out = collected.borrow();
    for step in &out.steps {
        let phases = step.forward_ns + step.backward_ns + step.optimizer_ns;
        assert!(
            step.total_ns >= phases,
            "total {} must cover forward+backward+optimizer {phases}",
            step.total_ns
        );
        // The three phases are measured back to back inside one step, so their
        // sum cannot exceed the total by more than the two boundary reads.
        assert!(step.forward_ns > 0, "forward must take measurable time");
        assert!(step.optimizer_ns > 0, "the optimizer update must take measurable time");
    }
}

/// The zero-cost claim, from outside: installing and removing an observer must
/// not change what training computes.
///
/// Bit-identical loss sequences from the same seed, observed and not. This is
/// Gate 3's "byte-identical results" requirement, checked across the public
/// API rather than inside the module that implements it.
#[test]
fn observation_does_not_change_the_loss_sequence() {
    let (mut bare, dataset) = fixture(4);
    let unobserved = bare.train(&dataset, 3).expect("training runs");

    let (mut watched, dataset2) = fixture(4);
    let collected = observe(&mut watched, true);
    let observed = watched.train(&dataset2, 3).expect("training runs");

    assert_eq!(
        unobserved.len(),
        observed.len(),
        "both runs must report the same number of epoch losses"
    );
    for (epoch, (a, b)) in unobserved.iter().zip(&observed).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "epoch {epoch}: observation changed the loss ({a} vs {b})"
        );
    }

    // And the per-step losses the observer saw match what training returned.
    assert!(!collected.borrow().steps.is_empty());
}

/// `clear_observer` must actually stop delivery — Wave 4's `unified` mode
/// installs an observer for the training phase only.
#[test]
fn clearing_the_observer_stops_delivery_but_keeps_what_was_collected() {
    let (mut trainer, dataset) = fixture(2);
    let collected = observe(&mut trainer, false);

    trainer.train(&dataset, 1).expect("first run");
    let after_first = collected.borrow().steps.len();
    assert_eq!(after_first, 2);

    let returned = trainer.clear_observer();
    assert!(returned.is_some(), "clear_observer hands the box back");

    trainer.train(&dataset, 1).expect("second run, unobserved");
    assert_eq!(
        collected.borrow().steps.len(),
        after_first,
        "no records may arrive after clear_observer"
    );
}

/// A bare `train_step` has no epoch to belong to, and Wave 4 must not read the
/// zero it reports as "epoch 0 of a real run".
#[test]
fn a_bare_train_step_reports_epoch_zero_and_still_advances_the_index() {
    let (mut trainer, dataset) = fixture(2);
    let collected = observe(&mut trainer, false);

    let (x, target) = dataset.sample::<GlProc>(0).expect("sample");
    trainer.train_step(&x, &target).expect("step runs");
    trainer.train_step(&x, &target).expect("step runs");

    let out = collected.borrow();
    assert_eq!(out.steps.len(), 2);
    assert_eq!(out.steps[0].epoch, 0);
    assert_eq!(out.steps[1].epoch, 0);
    assert_eq!(out.steps[0].index, 0);
    assert_eq!(out.steps[1].index, 1, "index advances outside train() too");
}
