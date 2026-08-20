//! Stummañ Deskiñ: measures what the M2.5 observer costs.
//!
//! Gate 3 requires a measured overhead number rather than an estimate, and a
//! timing assertion inside `#[test]` would be a flaky test wearing a
//! measurement's clothes. So it lives here.
//!
//! ```text
//! cargo run --release --example observed_training
//! ```
//!
//! Three configurations per layer size, over the same seed and step count:
//!
//! - no observer            — the path every existing caller is on
//! - observer, no tensors   — phase timing plus one pass over the parameters
//! - observer, with tensors — the above plus `Optimizer::state_tensors`
//!
//! Swept across three layer widths, because a single percentage would be a
//! lie. Measured 2026-08-19 on this machine, `observer, no tensors`:
//!
//! ```text
//!   64x64    +4.3%
//!  256x256  +11.5%
//!  512x512  +14.2%
//! ```
//!
//! Note the direction. The first draft of this file asserted the overhead was
//! a fixed per-step charge whose ratio would *fall* as the step grew. The sweep
//! says the opposite, and the reason is that the cost is not fixed: the
//! observer walks the parameter gradients, and a LoRA adapter has `2·r·d`
//! parameters, so its work grows linearly with the layer width while the step
//! itself does not grow as fast as `d²` on this backend. The claim was wrong
//! and the measurement is what caught it, which is the entire reason Gate 3
//! requires a number rather than an argument.
//!
//! Do not extrapolate these to M3's model. Re-run it there.
//!
//! The loss printed per size must be identical across all three configurations.
//! If it is not, the observer is perturbing the run and the timings are the
//! least of the problem.

use std::time::Instant;

use gltrain::autograd::grad_store::VLGradStore;
use gltrain::backend::GlProc;
use gltrain::optim::VLNamedTensor;
use gltrain::train::observe::{StepObserver, VLTrainingStep};
use gltrain::{Tensor, Trainer, VLTrainerConfig};

/// Does the least an observer can do, so what is measured is the hook itself
/// and not whatever a consumer builds on top of it.
struct NullObserver {
    wants: bool,
    seen: usize,
    last_norm: f64,
}

impl StepObserver for NullObserver {
    fn on_step(&mut self, step: &VLTrainingStep) {
        self.seen += 1;
        self.last_norm = step.grad_l2_norm;
    }
    fn wants_tensors(&self) -> bool {
        self.wants
    }
    fn on_tensors(&mut self, _grads: &VLGradStore, _opt_state: &[VLNamedTensor]) {}
}

const STEPS: usize = 1000;
const REPEATS: usize = 5;
const SIZES: [usize; 3] = [64, 256, 512];

fn null(wants: bool) -> NullObserver {
    NullObserver {
        wants,
        seen: 0,
        last_norm: 0.0,
    }
}

/// One run of `STEPS` steps at width `d`. Returns wall nanoseconds and the
/// final loss.
fn run(d: usize, observer: Option<NullObserver>) -> (u128, f32) {
    let cfg = VLTrainerConfig::new(d, d, 8, 1e-4, 42);
    let base = Tensor::<GlProc>::randn(&[d, d], 1.0, 1234).unwrap();
    let mut trainer = Trainer::<GlProc>::with_base(cfg, base).unwrap();
    let x = Tensor::<GlProc>::randn(&[1, d], 1.0, 7).unwrap();
    let y = Tensor::<GlProc>::randn(&[1, d], 1.0, 8).unwrap();

    if let Some(obs) = observer {
        trainer.set_observer(Box::new(obs));
    }

    // Warm up outside the timed region: the first step allocates the optimizer
    // moments, and charging that to the observer would measure the wrong thing.
    let mut loss = trainer.train_step(&x, &y).unwrap();

    let t0 = Instant::now();
    for _ in 0..STEPS {
        loss = trainer.train_step(&x, &y).unwrap();
    }
    (t0.elapsed().as_nanos(), loss)
}

fn sweep(d: usize) {
    // Best-of rather than mean. This machine has known cross-session drift, so
    // the minimum is the estimator least contaminated by whatever else the OS
    // was doing. Interleaved rather than run in blocks, so a thermal ramp
    // partway through hits all three configurations equally.
    let mut best = [u128::MAX; 3];
    let mut losses = [0f32; 3];

    for _ in 0..REPEATS {
        let (ns, l) = run(d, None);
        best[0] = best[0].min(ns);
        losses[0] = l;

        let (ns, l) = run(d, Some(null(false)));
        best[1] = best[1].min(ns);
        losses[1] = l;

        let (ns, l) = run(d, Some(null(true)));
        best[2] = best[2].min(ns);
        losses[2] = l;
    }

    let labels = ["no observer", "observer, no tensors", "observer, with tensors"];
    let baseline = best[0] as f64;

    println!("--- {d}x{d} layer, LoRA r=8 ---");
    println!(
        "{:<24} {:>10} {:>12} {:>10}",
        "configuration", "total ms", "ns/step", "overhead"
    );
    for i in 0..3 {
        println!(
            "{:<24} {:>10.2} {:>12.0} {:>9.1}%",
            labels[i],
            best[i] as f64 / 1e6,
            best[i] as f64 / STEPS as f64,
            (best[i] as f64 / baseline - 1.0) * 100.0
        );
    }

    if losses[0].to_bits() == losses[1].to_bits() && losses[0].to_bits() == losses[2].to_bits() {
        println!("loss identical across all three: {:.9}\n", losses[0]);
    } else {
        println!(
            "FINAL LOSS DIVERGED: {:.9} / {:.9} / {:.9}",
            losses[0], losses[1], losses[2]
        );
        std::process::exit(1);
    }
}

fn main() {
    println!("stummañ M2.5 — observer overhead");
    println!("{STEPS} steps, best of {REPEATS}, interleaved");
    println!("the observer walks parameter gradients, so its cost grows with the layer\n");

    for d in SIZES {
        sweep(d);
    }
}
