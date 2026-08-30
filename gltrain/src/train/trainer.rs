//! Stummañ Deskiñ: the training loop.
//!
//! This is where M1's tape, M2's optimizer and M2's checkpoint meet. The whole
//! milestone reduces to one method, [`Trainer::train_step`], and to the order
//! its six lines run in.
//!
//! # The forward pass calls the adapter once, not the base and then the adapter
//!
//! [`LRLora::forward`] already computes `x @ base_weight` and adds its scaled
//! delta to it: it returns the **full layer output**, not just the delta. That
//! is a deliberate trait choice, made so DoRA (M3) can exist without changing
//! the signature, since DoRA renormalizes the combined weight and cannot be
//! written as `base_out + delta_out` at all.
//!
//! So calling `base.forward(x)` and then `adapter.forward(x, w)` and adding
//! them would count the base contribution **twice**. It would still train:
//! the adapter would learn to cancel the surplus, the loss would fall, and
//! every test short of a numerical anchor would pass. The base is applied once,
//! inside the adapter.
//!
//! # KL-006, as executable code
//!
//! ```text
//! let grads = {
//!     let mut guard = Tape::lock(&self.tape);
//!     guard.backward()?;
//!     guard.finish_step()      // tape is empty from here on
//! };
//! self.optimizer.step(&mut params, &grads)?;
//! ```
//!
//! `finish_step` returns the gradients and clears the tape in one call, so the
//! weight write on the next line cannot invalidate a live backward closure.
//! There is no ordering for a caller to get wrong, because there is no way to
//! hold a `VLGradStore` and a populated tape at the same time.

use crate::autograd::grad_store::VLGradStore;
use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, LRLora, VLAdapterSpec};
use crate::nn::linear::ABLinear;
use crate::nn::param::TPParameter;
use crate::optim::{Optimizer, VLAdamWConfig, OPAdamW};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use crate::train::observe::{StepObserver, VLTrainingStep};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::checkpoint::{CPLora, CheckpointStore, VLManifest};

/// How to build a [`Trainer`].
///
/// `VL` because it is a plain config bag with derived traits only.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTrainerConfig {
    /// Input dimension of the adapted layer.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// LoRA rank.
    pub r: usize,
    /// LoRA alpha. Scaling is `alpha/r`.
    pub alpha: f32,
    /// Base learning rate.
    pub lr: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
    /// Seed for the adapter's `A` initialization.
    pub adapter_seed: u64,
    /// Seed for the frozen base weight, when the trainer generates one.
    pub base_seed: u64,
}

impl VLTrainerConfig {
    /// A config for a `[d_in, d_out]` layer at the given rank, with
    /// `alpha = r` and no weight decay.
    ///
    /// Weight decay defaults to 0 here rather than to AdamW's 1e-2: this
    /// trainer's job on M2 is to *fit* a micro-dataset, and decay pulls the
    /// adapter back toward zero, which fights the exit criterion for no
    /// benefit at ten samples.
    pub fn new(d_in: usize, d_out: usize, r: usize, lr: f64, seed: u64) -> Self {
        Self {
            d_in,
            d_out,
            r,
            alpha: r as f32,
            lr,
            weight_decay: 0.0,
            adapter_seed: seed,
            base_seed: seed.wrapping_add(1),
        }
    }
}

/// Mean squared error between a prediction and a target.
///
/// Records tape nodes, so the result can be `backward()`-ed. `target` arrives
/// untracked and stays that way: it is data, not a parameter, and the KL-003
/// frozen-operand path handles it without a gradient.
pub fn mse_loss<B: Backend>(pred: &Tensor<B>, target: &Tensor<B>) -> Result<Tensor<B>> {
    let diff = pred.sub(target)?;
    let sq = diff.mul(&diff)?;
    sq.mean()
}

/// One frozen linear layer plus a LoRA adapter over it, and the machinery to
/// train the adapter.
pub struct Trainer<B: Backend> {
    base: ABLinear<B>,
    adapter: LRLora<B>,
    optimizer: OPAdamW<B>,
    tape: Arc<Mutex<Tape>>,
    config: VLTrainerConfig,
    step: usize,
    /// M2.5. `None` is the default and the zero-cost path: with no observer
    /// installed, `train_step` calls no clock and touches no gradient twice.
    observer: Option<Box<dyn StepObserver>>,
    /// Which epoch `train` is currently in. Stays 0 for a bare `train_step`,
    /// which has no epoch to belong to.
    current_epoch: usize,
}

impl<B: Backend> Trainer<B> {
    /// A trainer over a randomly generated frozen base weight.
    pub fn new(config: VLTrainerConfig) -> Result<Self> {
        let base = Tensor::<B>::randn(&[config.d_in, config.d_out], 1.0, config.base_seed)?;
        Self::with_base(config, base)
    }

    /// A trainer over a supplied frozen base weight of shape `[d_in, d_out]`.
    pub fn with_base(config: VLTrainerConfig, base_weight: Tensor<B>) -> Result<Self> {
        if base_weight.shape() != [config.d_in, config.d_out] {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![config.d_in, config.d_out],
                got: base_weight.shape().to_vec(),
            });
        }
        let spec = VLAdapterSpec {
            d_in: config.d_in,
            d_out: config.d_out,
            r: config.r,
            alpha: config.alpha,
            rslora: false,
            seed: config.adapter_seed,
        };
        Ok(Self {
            base: ABLinear::frozen("base", base_weight)?,
            adapter: LRLora::new(&spec)?,
            optimizer: OPAdamW::new(VLAdamWConfig {
                lr: config.lr,
                weight_decay: config.weight_decay,
                ..VLAdamWConfig::default()
            }),
            tape: Arc::new(Mutex::new(Tape::new())),
            config,
            step: 0,
            observer: None,
            current_epoch: 0,
        })
    }

    /// The frozen base layer.
    pub fn base(&self) -> &ABLinear<B> {
        &self.base
    }

    /// The adapter being trained.
    pub fn adapter(&self) -> &LRLora<B> {
        &self.adapter
    }

    /// The optimizer.
    pub fn optimizer(&self) -> &OPAdamW<B> {
        &self.optimizer
    }

    /// How many updates have been applied.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// The tape, for tests that need to inspect it.
    pub fn tape(&self) -> &Arc<Mutex<Tape>> {
        &self.tape
    }

    /// Install an observer. It receives one [`VLTrainingStep`] per step from
    /// the next `train_step` onwards.
    ///
    /// Installing one turns on phase timing and a pass over the gradients, so
    /// it is not free. Not installing one costs a single `Option::is_some`.
    pub fn set_observer(&mut self, observer: Box<dyn StepObserver>) {
        self.observer = Some(observer);
    }

    /// Remove the observer and hand it back, so a caller can read whatever it
    /// accumulated. Returns `None` if none was installed.
    pub fn clear_observer(&mut self) -> Option<Box<dyn StepObserver>> {
        self.observer.take()
    }

    /// The epoch `train` is currently in, or 0 outside a `train` call.
    pub fn current_epoch(&self) -> usize {
        self.current_epoch
    }

    /// Forward pass: the full layer output, base plus scaled adapter delta.
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // Once. `LRLora::forward` applies the base weight itself.
        self.adapter
            .forward(x, self.base.weight().tensor(), &self.tape)
    }

    /// One forward, one backward, one optimizer step. Returns the loss.
    ///
    /// # M2.5 instrumentation
    ///
    /// `observed` is read once, at the top. When it is false every `then` below
    /// short-circuits, so no clock is read, no gradient is walked, and the
    /// arithmetic is the same sequence of operations M2 shipped. The timing
    /// calls sit *between* the math, never inside it, which is why there is one
    /// code path here rather than two: a duplicated body could drift, a shared
    /// one cannot.
    pub fn train_step(&mut self, x: &Tensor<B>, target: &Tensor<B>) -> Result<f32> {
        let observed = self.observer.is_some();
        let t0 = observed.then(Instant::now);

        let pred = self.forward(x)?;
        let loss = mse_loss(&pred, target)?;
        let loss_value = loss.item()?;
        let t1 = observed.then(Instant::now);

        // KL-006: the gradients and the empty tape arrive together, so the
        // in-place weight write below cannot be observed by a live closure.
        let grads = {
            let mut guard = Tape::lock(&self.tape);
            guard.backward()?;
            guard.finish_step()
        };
        let t2 = observed.then(Instant::now);

        // LoRA owns two distinctly-named parameters, so no dedup is needed
        // here. An adapter that shared one across sites (VeRA, M3) would have
        // to go through `crate::nn::trainable_parameters_mut` instead, or the
        // shared parameter would be updated once per site.
        //
        // Scoped so the `&mut self.adapter` borrow ends before the observer
        // window, which needs `&self` to read the parameters back out.
        {
            let mut params = self.adapter.parameters_mut();
            self.optimizer.step(&mut params, &grads)?;
        }
        let t3 = observed.then(Instant::now);

        let index = self.step;
        self.step += 1;

        // The observer runs *after* the update, not between `finish_step` and
        // it. `optimizer_ns` cannot be reported by a callback that runs before
        // the optimizer does, and reading `grads` late is safe: it is a local
        // this function owns, and `Optimizer::step` only borrows it shared.
        // KL-006 is untouched either way, since the tape was emptied by
        // `finish_step` well before any of this.
        if observed {
            self.emit_observation(index, loss_value, &grads, t0, t1, t2, t3)?;
        }
        Ok(loss_value)
    }

    /// Build one [`VLTrainingStep`] and hand it to the observer.
    ///
    /// Only called when an observer is installed, so every `Instant` is `Some`.
    #[allow(clippy::too_many_arguments)]
    fn emit_observation(
        &mut self,
        index: usize,
        loss: f32,
        grads: &VLGradStore,
        t0: Option<Instant>,
        t1: Option<Instant>,
        t2: Option<Instant>,
        t3: Option<Instant>,
    ) -> Result<()> {
        let (Some(t0), Some(t1), Some(t2), Some(t3)) = (t0, t1, t2, t3) else {
            return Ok(());
        };

        // Statistics cover **trainable parameters only**, not the whole store.
        //
        // `finish_step` returns a gradient for every tensor the tape touched,
        // which on this trainer is nine entries: the two LoRA parameters plus
        // seven intermediate activations. Folding activations into
        // `grad_l2_norm` would produce a number that is not the gradient norm
        // anyone means by that phrase: every framework's `clip_grad_norm_`
        // computes it over parameters, and gradient-health analysis reads it
        // that way. Measured on the 4x4 r=2 fixture, the difference is 9
        // tensors versus 2.
        //
        // The full store still reaches an observer that wants it, through
        // `on_tensors`.
        //
        // Non-finite values are counted and excluded from the norm rather than
        // folded into it: a single NaN would otherwise make `grad_l2_norm` NaN
        // and erase the whole signal. NaN and Inf get separate counters because
        // they point at different bugs, and `is_nan` / `is_infinite` are
        // mutually exclusive so nothing is double-counted.
        let mut grad_count = 0usize;
        let mut grad_elements = 0usize;
        let mut l2_squared = 0f64;
        let mut grad_nan = 0usize;
        let mut grad_inf = 0usize;
        for param in self.adapter_parameters() {
            let Some((data, _shape)) = grads.get(param.id()) else {
                continue;
            };
            grad_count += 1;
            grad_elements += data.len();
            for &v in data {
                if v.is_nan() {
                    grad_nan += 1;
                } else if v.is_infinite() {
                    grad_inf += 1;
                } else {
                    l2_squared += (v as f64) * (v as f64);
                }
            }
        }

        let record = VLTrainingStep {
            index,
            epoch: self.current_epoch,
            loss,
            forward_ns: (t1 - t0).as_nanos() as u64,
            backward_ns: (t2 - t1).as_nanos() as u64,
            optimizer_ns: (t3 - t2).as_nanos() as u64,
            total_ns: (t3 - t0).as_nanos() as u64,
            grad_count,
            grad_elements,
            grad_l2_norm: l2_squared.sqrt(),
            grad_nan,
            grad_inf,
            // From the optimizer, not from `self.config`: identical today, but
            // it stays correct when M3 adds a schedule.
            lr: self.optimizer.config().lr,
        };

        // `state_tensors` allocates a full copy of the optimizer state, so it
        // is only called when the observer asked for it. Computed before the
        // `&mut self.observer` borrow below, because it needs `&self`.
        let wants_tensors = self.observer.as_ref().is_some_and(|o| o.wants_tensors());
        let opt_state = if wants_tensors {
            self.optimizer.state_tensors(&self.adapter_parameters())?
        } else {
            Vec::new()
        };

        if let Some(observer) = self.observer.as_mut() {
            observer.on_step(&record);
            if wants_tensors {
                observer.on_tensors(grads, &opt_state);
            }
        }
        Ok(())
    }

    /// Train for `epochs` passes over `dataset`, returning the mean loss per
    /// epoch.
    pub fn train(&mut self, dataset: &super::VLMicroDataset, epochs: usize) -> Result<Vec<f32>> {
        if dataset.is_empty() {
            return Err(GlTrainError::InvalidOp(
                "cannot train on an empty dataset".into(),
            ));
        }
        if dataset.d_in() != self.config.d_in || dataset.d_out() != self.config.d_out {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![self.config.d_in, self.config.d_out],
                got: vec![dataset.d_in(), dataset.d_out()],
            });
        }
        let mut history = Vec::with_capacity(epochs);
        for epoch in 0..epochs {
            // M2.5: the only place an epoch exists. `train_step` called
            // directly reports epoch 0, which is honest rather than a default:
            // a bare step genuinely does not belong to one.
            self.current_epoch = epoch;
            let mut total = 0.0f64;
            for i in 0..dataset.len() {
                let (x, y) = dataset.sample::<B>(i)?;
                total += self.train_step(&x, &y)? as f64;
            }
            history.push((total / dataset.len() as f64) as f32);
        }
        // Leave the counter where a bare `train_step` would find it, so the
        // epoch a step reports never depends on a previous `train` call.
        self.current_epoch = 0;
        Ok(history)
    }

    /// The manifest this trainer's adapter would be saved under.
    pub fn manifest(&self) -> VLManifest {
        VLManifest::for_lora(self.adapter.config(), self.step)
    }

    /// Write the adapter and its optimizer state to `dir`.
    pub fn save_checkpoint(&self, dir: &Path) -> Result<()> {
        let ckpt = CPLora::checkpoint_from(&self.adapter, Some(&self.optimizer), self.step, None)?;
        CPLora.save(dir, &ckpt)
    }

    /// Restore the adapter and optimizer state from `dir`.
    ///
    /// The restored adapter carries **new** `TensorId`s, so the optimizer state
    /// is re-keyed onto them by name. The tape is cleared as well: any node
    /// left on it would reference IDs that no longer exist.
    pub fn load_checkpoint(&mut self, dir: &Path) -> Result<()> {
        let ckpt = CPLora.load(dir)?;
        CPLora.validate(dir, &self.manifest())?.into_result()?;

        self.adapter = CPLora::restore_adapter::<B>(&ckpt)?;
        let mut optimizer = OPAdamW::new(self.optimizer.config().clone());
        if ckpt.manifest.has_optimizer_state {
            let params = self.adapter.parameters();
            CPLora::restore_optimizer(&ckpt, &mut optimizer, &params)?;
        }
        self.optimizer = optimizer;
        self.step = ckpt.manifest.step;
        Tape::lock(&self.tape).finish_step();
        Ok(())
    }

    /// The adapter's parameters, for a caller that wants to inspect them.
    pub fn adapter_parameters(&self) -> Vec<&TPParameter<B>> {
        self.adapter.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;
    use crate::train::VLMicroDataset;

    /// Loss is a mean of squares over a short chain of f32 ops.
    const TOL_LOSS: f32 = 1e-4;

    /// Frozen weights are compared byte for byte: nothing should have touched
    /// them at all, so any difference is a real bug rather than drift.
    const TOL_EXACT: f32 = 0.0;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "stumman_tr_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The M2 exit criterion, and the reason every other wave exists.
    ///
    /// # Where lr = 0.01 comes from
    ///
    /// Measured, not guessed. Sweeping this exact setup, 50 epochs over 10
    /// samples at rank 4:
    ///
    /// ```text
    ///       lr    loss[0]   loss[49]        min    max_climb   monotone
    ///      0.2    5.45459   0.031710   0.030987     7.849724         no
    ///      0.1    5.30153   0.122191   0.013108     0.377226         no
    ///     0.05    5.41556   0.035342   0.004936     0.043322         no
    ///     0.02    5.58047   0.000380   0.000380     0.000010        yes
    ///     0.01    5.64144   0.000323   0.000323     0.000000        yes
    ///    0.005    5.67236   0.162608   0.162608     0.000000        yes
    ///    0.002    5.69097   1.538199   1.538199     0.000000        yes
    /// ```
    ///
    /// The usable window is roughly `[0.01, 0.02]`: below it the run has not
    /// converged by epoch 50, above it AdamW overshoots and the loss
    /// oscillates around the minimum rather than settling. At 0.05 the curve
    /// really does fall to 0.0049 by epoch 11 and climb back to 0.16 by epoch
    /// 39, which is why the monotonicity assertion below is worth having: the
    /// two threshold checks alone would have passed that run.
    #[test]
    fn trainer_loss_decreases_on_synthetic_regression() {
        let (ds, _w) = VLMicroDataset::synthetic_regression(10, 4, 4, 42).unwrap();
        let cfg = VLTrainerConfig::new(4, 4, 4, 0.01, 42);
        let mut trainer = Trainer::<GlProc>::with_base(
            cfg,
            Tensor::<GlProc>::randn(&[4, 4], 1.0, 1234).unwrap(),
        )
        .unwrap();

        let history = trainer.train(&ds, 50).unwrap();
        assert_eq!(history.len(), 50);

        println!("loss history over 50 epochs:");
        for (e, l) in history.iter().enumerate() {
            println!("  epoch {e:>2}: {l:.6}");
        }
        let (first, last) = (history[0], history[49]);
        println!("loss[0] = {first:.4}, loss[49] = {last:.6}");

        assert!(
            first > 1.0,
            "the task must start hard: loss[0] = {first}, expected > 1.0"
        );
        assert!(
            last < 0.1,
            "the adapter must fit the micro-dataset: loss[49] = {last}, expected < 0.1"
        );
        // Monotone, allowing a plateau but not a real climb. AdamW on a
        // bilinear objective can wobble; anything above this is divergence.
        const MAX_INCREASE: f32 = 1e-4;
        for w in history.windows(2) {
            assert!(
                w[1] <= w[0] + MAX_INCREASE,
                "loss climbed from {} to {}, which is more than the {MAX_INCREASE} plateau \
                 allowance",
                w[0],
                w[1]
            );
        }
    }

    /// The KL-006 regression test. If this ever fails, a backward closure is
    /// alive at the moment the optimizer overwrites the weights it captured.
    #[test]
    fn train_step_leaves_the_tape_empty() {
        let (ds, _) = VLMicroDataset::synthetic_regression(3, 4, 4, 5).unwrap();
        let mut trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 5)).unwrap();

        for i in 0..ds.len() {
            let (x, y) = ds.sample::<GlProc>(i).unwrap();
            trainer.train_step(&x, &y).unwrap();
            let guard = Tape::lock(trainer.tape());
            assert!(
                guard.is_empty(),
                "after step {i} the tape still holds {} node(s)",
                guard.len()
            );
            assert_eq!(guard.tensor_count(), 0, "tensor registrations survived");
            assert!(guard.grad_store().is_empty(), "gradients survived");
        }
    }

    /// The property that makes LoRA's entire memory argument true.
    #[test]
    fn trainer_leaves_the_frozen_base_weight_untouched() {
        let (ds, _) = VLMicroDataset::synthetic_regression(10, 4, 4, 11).unwrap();
        let mut trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 4, 0.05, 11)).unwrap();

        let before = trainer.base().weight().to_vec().unwrap();
        trainer.train(&ds, 20).unwrap();
        let after = trainer.base().weight().to_vec().unwrap();

        assert_eq!(before.len(), after.len());
        for (i, (b, a)) in before.iter().zip(&after).enumerate() {
            assert!(
                (b - a).abs() <= TOL_EXACT,
                "base weight element {i} changed: {b} -> {a}"
            );
        }
        assert!(
            !trainer.base().weight().is_trainable(),
            "the base must be frozen"
        );
    }

    /// A frozen operand receives no gradient at all, which is what stops the
    /// optimizer from ever considering it.
    #[test]
    fn backward_produces_adapter_gradients_and_none_for_the_frozen_base() {
        let trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 3, 2, 0.01, 3)).unwrap();
        let x = Tensor::<GlProc>::from_vec(vec![0.5, -1.0, 2.0, 0.25], &[1, 4]).unwrap();
        let target = Tensor::<GlProc>::from_vec(vec![1.0, 1.0, 1.0], &[1, 3]).unwrap();

        let pred = trainer.forward(&x).unwrap();
        let loss = mse_loss(&pred, &target).unwrap();
        assert_eq!(loss.shape(), &[1]);

        let mut guard = Tape::lock(trainer.tape());
        guard.backward().unwrap();

        // B is the one that moves first: at init A is random and B is zero, so
        // dL/dA is zero and dL/dB is not.
        let b_grad = guard
            .grad(trainer.adapter().b().id())
            .expect("B must receive a gradient");
        assert!(
            b_grad.0.iter().any(|g| *g != 0.0),
            "B's gradient is all zeros: {:?}",
            b_grad.0
        );
        assert!(
            guard.grad(trainer.base().weight().id()).is_none(),
            "the frozen base weight received a gradient"
        );
    }

    /// At init `B = 0`, so `A @ B = 0` and the adapted layer must reproduce the
    /// base layer exactly. If it does not, the forward pass is applying the
    /// base twice, or not at all.
    #[test]
    fn an_untrained_adapter_reproduces_the_frozen_base_layer_exactly() {
        let trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 17)).unwrap();
        let x = Tensor::<GlProc>::from_vec(vec![0.5, -1.0, 2.0, 0.25], &[1, 4]).unwrap();

        let adapted = trainer.forward(&x).unwrap().to_vec().unwrap();
        let base_only = x
            .matmul(trainer.base().weight().tensor())
            .unwrap()
            .to_vec()
            .unwrap();

        for (i, (a, b)) in adapted.iter().zip(&base_only).enumerate() {
            assert!(
                (a - b).abs() < TOL_LOSS,
                "element {i}: adapted {a} != base {b}. A doubled base would give {}",
                b * 2.0
            );
        }
    }

    #[test]
    fn trainer_checkpoint_round_trip_resumes_training() {
        let dir = tmp_dir("resume");
        let (ds, _) = VLMicroDataset::synthetic_regression(10, 4, 4, 42).unwrap();
        // Same lr as the convergence test, for the same measured reason.
        let cfg = VLTrainerConfig::new(4, 4, 4, 0.01, 42);
        let base = Tensor::<GlProc>::randn(&[4, 4], 1.0, 1234).unwrap();

        let mut trainer = Trainer::<GlProc>::with_base(cfg.clone(), base.clone()).unwrap();
        let first = trainer.train(&ds, 25).unwrap();
        trainer.save_checkpoint(&dir).unwrap();
        let loss_at_save = *first.last().unwrap();
        let steps_at_save = trainer.step_count();

        // A brand-new trainer, with a fresh adapter and no optimizer state.
        let mut resumed = Trainer::<GlProc>::with_base(cfg, base).unwrap();
        resumed.load_checkpoint(&dir).unwrap();
        assert_eq!(resumed.step_count(), steps_at_save);
        assert_eq!(
            resumed.optimizer().step_count(),
            steps_at_save,
            "the optimizer's bias-correction clock must resume too"
        );

        // The restored adapter must reproduce the saved loss before it trains.
        let (x0, y0) = ds.sample::<GlProc>(0).unwrap();
        let pred = resumed.forward(&x0).unwrap();
        let _ = mse_loss(&pred, &y0).unwrap();
        Tape::lock(resumed.tape()).finish_step();

        let second = resumed.train(&ds, 25).unwrap();
        println!(
            "loss at save = {loss_at_save:.6}, after resuming 25 more = {:.6}",
            second.last().unwrap()
        );
        assert!(
            *second.last().unwrap() < loss_at_save,
            "resumed training must keep improving: {loss_at_save} -> {}",
            second.last().unwrap()
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Saving and reloading must not perturb the weights at all. A resumed run
    /// that starts from slightly different numbers is a silent bug: the loss
    /// curve still looks fine.
    #[test]
    fn a_checkpoint_round_trip_preserves_the_adapter_bit_for_bit() {
        let dir = tmp_dir("bitexact");
        let (ds, _) = VLMicroDataset::synthetic_regression(5, 4, 4, 8).unwrap();
        let cfg = VLTrainerConfig::new(4, 4, 2, 0.05, 8);
        let base = Tensor::<GlProc>::randn(&[4, 4], 1.0, 99).unwrap();

        let mut trainer = Trainer::<GlProc>::with_base(cfg.clone(), base.clone()).unwrap();
        trainer.train(&ds, 5).unwrap();
        trainer.save_checkpoint(&dir).unwrap();
        let (want_a, want_b) = (
            trainer.adapter().a().to_vec().unwrap(),
            trainer.adapter().b().to_vec().unwrap(),
        );

        let mut resumed = Trainer::<GlProc>::with_base(cfg, base).unwrap();
        resumed.load_checkpoint(&dir).unwrap();
        for (i, (g, w)) in resumed
            .adapter()
            .a()
            .to_vec()
            .unwrap()
            .iter()
            .zip(&want_a)
            .enumerate()
        {
            assert!((g - w).abs() <= TOL_EXACT, "A[{i}]: {g} != {w}");
        }
        for (i, (g, w)) in resumed
            .adapter()
            .b()
            .to_vec()
            .unwrap()
            .iter()
            .zip(&want_b)
            .enumerate()
        {
            assert!((g - w).abs() <= TOL_EXACT, "B[{i}]: {g} != {w}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// MSE of a prediction against itself is zero, and against a constant
    /// offset is that offset squared. Two anchors a sign error cannot pass.
    #[test]
    fn mse_loss_matches_its_hand_computed_value() {
        let pred = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let same = mse_loss(&pred, &pred).unwrap().item().unwrap();
        assert!(same.abs() < TOL_LOSS, "MSE against itself = {same}");

        // Every element off by 2, so the mean of squares is exactly 4.
        let target = Tensor::<GlProc>::from_vec(vec![3.0, 4.0, 5.0, 6.0], &[2, 2]).unwrap();
        let off = mse_loss(&pred, &target).unwrap().item().unwrap();
        assert!((off - 4.0).abs() < TOL_LOSS, "MSE = {off}, expected 4.0");
    }

    #[test]
    fn training_on_a_dataset_of_the_wrong_width_is_refused() {
        let (ds, _) = VLMicroDataset::synthetic_regression(4, 8, 8, 1).unwrap();
        let mut trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 1)).unwrap();
        assert!(trainer.train(&ds, 1).is_err());
    }

    #[test]
    fn training_on_an_empty_dataset_is_refused() {
        let ds = VLMicroDataset::new(4, 4);
        let mut trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 1)).unwrap();
        assert!(trainer.train(&ds, 1).is_err());
    }

    #[test]
    fn a_base_weight_of_the_wrong_shape_is_refused() {
        let wrong = Tensor::<GlProc>::zeros(&[3, 5]).unwrap();
        assert!(Trainer::<GlProc>::with_base(VLTrainerConfig::new(4, 4, 2, 0.01, 1), wrong).is_err());
    }

    // ---- M2.5 observability -------------------------------------------------

    use std::cell::RefCell;
    use std::rc::Rc;

    /// Shared read-back handle.
    ///
    /// `set_observer` takes a `Box<dyn StepObserver>`, and a trait object
    /// cannot be downcast back to its concrete type without either an `Any`
    /// supertrait (which would widen the public trait for the benefit of tests)
    /// or `unsafe`. Sharing an `Rc<RefCell<..>>` with the observer costs
    /// nothing and stays entirely in safe code.
    #[derive(Clone, Default)]
    struct VLRecorderHandle {
        steps: Rc<RefCell<Vec<VLTrainingStep>>>,
        tensor_calls: Rc<RefCell<usize>>,
    }

    impl VLRecorderHandle {
        fn steps(&self) -> Vec<VLTrainingStep> {
            self.steps.borrow().clone()
        }
        fn tensor_calls(&self) -> usize {
            *self.tensor_calls.borrow()
        }
    }

    struct VLRecorder {
        handle: VLRecorderHandle,
        wants: bool,
    }

    impl StepObserver for VLRecorder {
        fn on_step(&mut self, step: &VLTrainingStep) {
            self.handle.steps.borrow_mut().push(step.clone());
        }
        fn wants_tensors(&self) -> bool {
            self.wants
        }
        fn on_tensors(
            &mut self,
            _grads: &crate::autograd::grad_store::VLGradStore,
            _opt_state: &[crate::optim::VLNamedTensor],
        ) {
            *self.handle.tensor_calls.borrow_mut() += 1;
        }
    }

    /// Install a recorder and hand back the handle to read it through.
    fn watch(trainer: &mut Trainer<GlProc>, wants: bool) -> VLRecorderHandle {
        let handle = VLRecorderHandle::default();
        trainer.set_observer(Box::new(VLRecorder {
            handle: handle.clone(),
            wants,
        }));
        handle
    }

    fn fixture(seed: u64) -> (Trainer<GlProc>, Tensor<GlProc>, Tensor<GlProc>) {
        let cfg = VLTrainerConfig::new(4, 4, 2, 0.01, seed);
        let base = Tensor::<GlProc>::randn(&[4, 4], 1.0, 1234).unwrap();
        let trainer = Trainer::<GlProc>::with_base(cfg, base).unwrap();
        let x = Tensor::<GlProc>::randn(&[1, 4], 1.0, 7).unwrap();
        let y = Tensor::<GlProc>::randn(&[1, 4], 1.0, 8).unwrap();
        (trainer, x, y)
    }

    /// Two identical trainers with no observer produce the same losses. The
    /// determinism baseline the next test measures the observer against.
    #[test]
    fn unobserved_train_step_is_deterministic() {
        const N: usize = 10;
        let (mut a, xa, ya) = fixture(42);
        let (mut b, xb, yb) = fixture(42);
        for i in 0..N {
            let la = a.train_step(&xa, &ya).unwrap();
            let lb = b.train_step(&xb, &yb).unwrap();
            assert_eq!(la.to_bits(), lb.to_bits(), "step {i}: unobserved run diverged");
        }
    }

    /// **The load-bearing test of M2.5.**
    ///
    /// Installing an observer must not perturb the arithmetic. Comparing two
    /// *unobserved* runs cannot show that: both sides would be equally wrong if
    /// instrumentation broke the math. So one side is observed and the other is
    /// not, and the losses are compared bit for bit.
    #[test]
    fn installing_an_observer_does_not_change_the_loss() {
        const N: usize = 10;
        let (mut observed, xo, yo) = fixture(42);
        let (mut plain, xp, yp) = fixture(42);
        let _handle = watch(&mut observed, false);

        for i in 0..N {
            let lo = observed.train_step(&xo, &yo).unwrap();
            let lp = plain.train_step(&xp, &yp).unwrap();
            assert_eq!(
                lo.to_bits(),
                lp.to_bits(),
                "step {i}: the observer changed the loss"
            );
        }

        // The frozen base must still be frozen on the observed side.
        let frozen = observed.base().weight().tensor().to_vec().unwrap();
        let reference = plain.base().weight().tensor().to_vec().unwrap();
        for (a, b) in frozen.iter().zip(reference.iter()) {
            assert!((a - b).abs() <= TOL_EXACT, "observation touched a frozen weight");
        }
    }

    /// The same, with the expensive payload turned on: `state_tensors` must not
    /// disturb the run either.
    #[test]
    fn requesting_tensors_does_not_change_the_loss() {
        const N: usize = 6;
        let (mut observed, xo, yo) = fixture(42);
        let (mut plain, xp, yp) = fixture(42);
        let handle = watch(&mut observed, true);

        for i in 0..N {
            let lo = observed.train_step(&xo, &yo).unwrap();
            let lp = plain.train_step(&xp, &yp).unwrap();
            assert_eq!(lo.to_bits(), lp.to_bits(), "step {i}: on_tensors changed the loss");
        }
        assert_eq!(handle.tensor_calls(), N);
    }

    /// Step indices count from zero and the epoch follows the outer loop.
    #[test]
    fn observer_receives_step_index_and_epoch() {
        let (ds, _w) = VLMicroDataset::synthetic_regression(3, 4, 4, 42).unwrap();
        let mut trainer = Trainer::<GlProc>::new(VLTrainerConfig::new(4, 4, 2, 0.01, 42)).unwrap();
        let handle = watch(&mut trainer, false);

        trainer.train(&ds, 2).unwrap();

        let seen: Vec<(usize, usize)> = handle.steps().iter().map(|s| (s.index, s.epoch)).collect();
        assert_eq!(
            seen,
            vec![(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)],
            "step index must be global and monotonic; epoch must follow the outer loop"
        );
    }

    /// A bare `train_step` reports epoch 0, and `train` leaves the counter where
    /// a later bare step would find it.
    #[test]
    fn a_bare_train_step_reports_epoch_zero() {
        let (ds, _w) = VLMicroDataset::synthetic_regression(2, 4, 4, 42).unwrap();
        let (mut trainer, x, y) = fixture(42);
        let handle = watch(&mut trainer, false);

        trainer.train(&ds, 3).unwrap();
        assert_eq!(trainer.current_epoch(), 0, "train must reset the epoch counter");
        trainer.train_step(&x, &y).unwrap();

        assert_eq!(
            handle.steps().last().unwrap().epoch,
            0,
            "a step outside train() belongs to no epoch"
        );
    }

    /// The observed loss is the value `train_step` returned, not a
    /// recomputation of it. Compared on bits: same value, or a bug.
    #[test]
    fn observer_loss_matches_the_returned_value() {
        let (mut trainer, x, y) = fixture(42);
        let handle = watch(&mut trainer, false);

        let mut returned = Vec::new();
        for _ in 0..5 {
            returned.push(trainer.train_step(&x, &y).unwrap());
        }

        let steps = handle.steps();
        assert_eq!(steps.len(), returned.len());
        for (i, (step, want)) in steps.iter().zip(returned.iter()).enumerate() {
            assert_eq!(
                step.loss.to_bits(),
                want.to_bits(),
                "step {i}: observed loss is not the returned loss"
            );
        }
    }

    /// Gradient facts are real: LoRA has two trainable parameters, both get a
    /// gradient, the norm is finite and positive, and a healthy step has
    /// neither NaN nor Inf.
    #[test]
    fn observed_gradient_statistics_describe_the_step() {
        let (mut trainer, x, y) = fixture(42);
        let handle = watch(&mut trainer, false);
        trainer.train_step(&x, &y).unwrap();

        let steps = handle.steps();
        let step = &steps[0];

        assert_eq!(step.grad_count, 2, "LoRA A and B should both receive a gradient");
        assert_eq!(step.grad_elements, 4 * 2 + 2 * 4, "r=2 over a 4x4 layer");
        assert!(step.grad_l2_norm.is_finite(), "norm must be neither NaN nor Inf");
        assert!(step.grad_l2_norm > 0.0, "a real step has a non-zero gradient");
        assert_eq!(step.grad_nan, 0);
        assert_eq!(step.grad_inf, 0);
        assert!((step.lr - 0.01).abs() < 1e-12, "lr must come from the optimizer");
    }

    /// The store holds more than the parameters, and the record counts only the
    /// parameters.
    ///
    /// `finish_step` returns a gradient per *tensor*, activations included. This
    /// was found by running the code, not by reading it: the first version of
    /// `emit_observation` walked the whole store and reported `grad_count = 9`
    /// on a two-parameter adapter, which would have made `grad_l2_norm` a number
    /// nobody could interpret. The test pins both halves of the distinction so a
    /// later edit cannot quietly re-merge them.
    #[test]
    fn the_store_holds_activations_but_the_record_counts_parameters() {
        struct VLStoreSizer {
            store_len: Rc<RefCell<usize>>,
        }
        impl StepObserver for VLStoreSizer {
            fn on_step(&mut self, _step: &VLTrainingStep) {}
            fn wants_tensors(&self) -> bool {
                true
            }
            fn on_tensors(
                &mut self,
                grads: &crate::autograd::grad_store::VLGradStore,
                _opt_state: &[crate::optim::VLNamedTensor],
            ) {
                *self.store_len.borrow_mut() = grads.len();
            }
        }

        let (mut trainer, x, y) = fixture(42);
        let store_len = Rc::new(RefCell::new(0usize));
        trainer.set_observer(Box::new(VLStoreSizer {
            store_len: Rc::clone(&store_len),
        }));
        trainer.train_step(&x, &y).unwrap();

        let store_len = *store_len.borrow();
        assert!(
            store_len > 2,
            "the tape's store should carry activation gradients too, got {store_len}"
        );
        // And `iter` must reach all of them, which is what makes the full store
        // usable by an observer that wants more than the parameters.
        assert!(store_len >= 2, "store must at least contain the parameters");
    }

    /// `on_tensors` fires only when the observer asks for it. The default is
    /// off, so a plain observer never pays for `state_tensors`.
    #[test]
    fn tensor_payload_is_opt_in() {
        for (wants, expected) in [(false, 0), (true, 3)] {
            let (mut trainer, x, y) = fixture(42);
            let handle = watch(&mut trainer, wants);
            for _ in 0..3 {
                trainer.train_step(&x, &y).unwrap();
            }
            assert_eq!(
                handle.tensor_calls(),
                expected,
                "wants_tensors={wants} should give {expected} on_tensors calls"
            );
        }
    }

    /// Phase timings must be attributed, not merely non-zero: the three parts
    /// have to fit inside the total, or one phase is being charged for
    /// another's work. `total_ns` is measured before the observer window opens,
    /// so it must not include the observer's own cost either.
    #[test]
    fn phase_timings_are_attributed_within_the_total() {
        let (mut trainer, x, y) = fixture(42);
        let handle = watch(&mut trainer, false);
        for _ in 0..5 {
            trainer.train_step(&x, &y).unwrap();
        }

        for (i, s) in handle.steps().iter().enumerate() {
            let parts = s.forward_ns + s.backward_ns + s.optimizer_ns;
            assert!(
                parts <= s.total_ns,
                "step {i}: phases ({parts} ns) exceed the total ({} ns)",
                s.total_ns
            );
            assert!(s.total_ns > 0, "step {i}: total_ns must be measured");
        }
    }

    /// `clear_observer` gives the box back and stops the callbacks.
    #[test]
    fn clearing_the_observer_stops_observation() {
        let (mut trainer, x, y) = fixture(42);
        let handle = watch(&mut trainer, false);
        trainer.train_step(&x, &y).unwrap();
        assert_eq!(handle.steps().len(), 1);

        assert!(trainer.clear_observer().is_some());
        assert!(trainer.clear_observer().is_none(), "clearing twice yields None");

        trainer.train_step(&x, &y).unwrap();
        assert_eq!(handle.steps().len(), 1, "a cleared observer must not be called");
    }
}
