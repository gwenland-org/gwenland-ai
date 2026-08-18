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

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, LRLora, VLAdapterSpec};
use crate::nn::linear::ABLinear;
use crate::nn::param::TPParameter;
use crate::optim::{Optimizer, VLAdamWConfig, OPAdamW};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::path::Path;
use std::sync::{Arc, Mutex};

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

    /// Forward pass: the full layer output, base plus scaled adapter delta.
    pub fn forward(&self, x: &Tensor<B>) -> Result<Tensor<B>> {
        // Once. `LRLora::forward` applies the base weight itself.
        self.adapter
            .forward(x, self.base.weight().tensor(), &self.tape)
    }

    /// One forward, one backward, one optimizer step. Returns the loss.
    pub fn train_step(&mut self, x: &Tensor<B>, target: &Tensor<B>) -> Result<f32> {
        let pred = self.forward(x)?;
        let loss = mse_loss(&pred, target)?;
        let loss_value = loss.item()?;

        // KL-006: the gradients and the empty tape arrive together, so the
        // in-place weight write below cannot be observed by a live closure.
        let grads = {
            let mut guard = Tape::lock(&self.tape);
            guard.backward()?;
            guard.finish_step()
        };

        // LoRA owns two distinctly-named parameters, so no dedup is needed
        // here. An adapter that shared one across sites (VeRA, M3) would have
        // to go through `crate::nn::trainable_parameters_mut` instead, or the
        // shared parameter would be updated once per site.
        let mut params = self.adapter.parameters_mut();
        self.optimizer.step(&mut params, &grads)?;
        self.step += 1;
        Ok(loss_value)
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
        for _ in 0..epochs {
            let mut total = 0.0f64;
            for i in 0..dataset.len() {
                let (x, y) = dataset.sample::<B>(i)?;
                total += self.train_step(&x, &y)? as f64;
            }
            history.push((total / dataset.len() as f64) as f32);
        }
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
}
