//! Stummañ Gwellaer: AdamW optimizer. **FULL implementation.**
//!
//! Loshchilov & Hutter 2019 (arXiv:1711.05101). The paper and PyTorch's
//! implementation agree term for term, so there is no reference/production
//! divergence to design around and the arithmetic is settled:
//!
//! ```text
//! m_t = b1*m_{t-1} + (1-b1)*g_t
//! v_t = b2*v_{t-1} + (1-b2)*g_t^2
//! m^_t = m_t / (1 - b1^t)
//! v^_t = v_t / (1 - b2^t)
//! theta_t = theta_{t-1} - lr*( m^_t/(sqrt(v^_t) + eps) + lambda*theta_{t-1} )
//! ```
//!
//! # The decay term reads the pre-update weight
//!
//! `theta_{t-1}`, not the post-update `theta_t`. Applying decay *after* the
//! adaptive step, as `theta *= (1 - lr*wd)`, is a real bug that shipped in this
//! crate's own planning document (`STUMMAN_PLAN.md` §3.4): it multiplies the
//! decay into the update term too, leaving a spurious `+lr^2*wd*u` error. The
//! sketch there is stale and must not be copied.
//!
//! # What is actually hard here
//!
//! Not the math. Three architectural facts, each of which would produce a
//! silently wrong training run:
//!
//! - **KL-006**: the update writes weights in place, so no tape may be live.
//!   [`Optimizer::step`] takes a [`VLGradStore`], which cannot be obtained
//!   while a tape still holds nodes. See [`crate::optim`]'s module docs.
//! - **F1**: gradients live on the tape, not on the tensor. There is no
//!   `param.grad()`; the store is looked up by the parameter's `TensorId`.
//! - **F2**: `TensorId` is process-global and not persistable. State is keyed
//!   by ID in memory and by **name** on disk, translated only at the
//!   save/load boundary.

use crate::autograd::grad_store::VLGradStore;
use crate::autograd::node::TensorId;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use crate::nn::param::TPParameter;
use crate::optim::{
    ENOptimizerStateShape, GroupTable, Optimizer, VLNamedTensor, VLOptimizerCapability,
    VLOptimizerSpec, VLParamGroup,
};
use crate::tensor::backend::Backend;
use std::collections::HashMap;

/// AdamW's capability record.
pub static CAPABILITY: &VLOptimizerCapability = &VLOptimizerCapability {
    id: "adamw",
    status: ENSkillStatus::Full,
    state_shape: ENOptimizerStateShape::Fixed {
        buffers_per_param: 2,
    },
    memory_multiplier: 2.0,
    source: "Loshchilov & Hutter 2019, arXiv:1711.05101",
};

/// AdamW's hyperparameters.
///
/// `VL` because it is a plain config bag with derived traits only. Defaults
/// match PyTorch's, which is what a reader will assume unless told otherwise.
#[derive(Debug, Clone, PartialEq)]
pub struct VLAdamWConfig {
    /// Base learning rate, before any group multiplier.
    pub lr: f64,
    /// First moment decay.
    pub beta1: f64,
    /// Second moment decay.
    pub beta2: f64,
    /// Added to the denominator after the square root, never inside it.
    pub eps: f64,
    /// Decoupled weight decay coefficient, `lambda` in the update above.
    pub weight_decay: f64,
}

impl Default for VLAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 1e-2,
        }
    }
}

/// The first and second moments for one parameter.
///
/// `OP` rather than `VL`: this is meaningless outside the optimizer sub-system,
/// and the naming convention says a domain prefix beats the plain-data one in
/// exactly that case (the `AGNode` precedent). Both buffers are the
/// parameter's own size, which is where AdamW's 2x memory cost comes from.
pub struct OPAdamWMoments<B: Backend> {
    /// First moment, `m`.
    pub m: B::Storage,
    /// Second moment, `v`.
    pub v: B::Storage,
}

impl<B: Backend> OPAdamWMoments<B> {
    /// Zeroed moments sized for an `n_elems` parameter.
    fn zeros(n_elems: usize) -> Result<Self> {
        Ok(Self {
            m: B::zeros(n_elems)?,
            v: B::zeros(n_elems)?,
        })
    }
}

/// AdamW with decoupled weight decay and parameter groups.
pub struct OPAdamW<B: Backend> {
    config: VLAdamWConfig,
    groups: GroupTable,
    /// Keyed by `TensorId`, which is live and unique for this process. Never
    /// serialized in this form: see [`Optimizer::state_tensors`].
    state: HashMap<TensorId, OPAdamWMoments<B>>,
    step_count: usize,
}

impl<B: Backend> OPAdamW<B> {
    /// An optimizer with the given hyperparameters and only the default group.
    pub fn new(config: VLAdamWConfig) -> Self {
        Self {
            config,
            groups: GroupTable::new(),
            state: HashMap::new(),
            step_count: 0,
        }
    }

    /// PyTorch's defaults with a learning rate override.
    pub fn with_lr(lr: f64) -> Self {
        Self::new(VLAdamWConfig {
            lr,
            ..VLAdamWConfig::default()
        })
    }

    /// The hyperparameters in use.
    pub fn config(&self) -> &VLAdamWConfig {
        &self.config
    }

    /// How many steps have been applied. Drives the bias correction, so it has
    /// to survive a checkpoint or a resumed run takes a badly-scaled first step.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// The moments held for a parameter, if it has taken a step yet.
    pub fn moments(&self, id: TensorId) -> Option<&OPAdamWMoments<B>> {
        self.state.get(&id)
    }

    /// The learning rate this parameter will actually get, after its group's
    /// multiplier.
    pub fn effective_lr(&self, param_name: &str) -> f64 {
        self.groups.effective_lr(self.config.lr, param_name)
    }

    /// The key optimizer state is saved under: `"{param}.{slot}"`.
    fn state_key(param_name: &str, slot: &str) -> String {
        format!("{param_name}.{slot}")
    }
}

/// The key the step counter is saved under.
///
/// Double-underscored so it cannot collide with a parameter named `step`: a
/// real parameter's key always contains a `.` separator, and this one does not.
pub const STEP_COUNT_KEY: &str = "__step_count__";

impl<B: Backend> Optimizer<B> for OPAdamW<B> {
    fn step(&mut self, params: &mut [&mut TPParameter<B>], grads: &VLGradStore) -> Result<()> {
        self.step_count += 1;
        let t = self.step_count as i32;
        // Bias correction. At t=1 these are 1-0.9 = 0.1 and 1-0.999 = 0.001,
        // which is what stops the first step from being ~1000x too small.
        let bc1 = 1.0 - self.config.beta1.powi(t);
        let bc2 = 1.0 - self.config.beta2.powi(t);
        if bc1 == 0.0 || bc2 == 0.0 {
            return Err(GlTrainError::InvalidOp(
                "AdamW bias correction is zero; beta1 and beta2 must be < 1".into(),
            ));
        }

        for p in params.iter_mut() {
            // No gradient this step is normal, not an error: a frozen base
            // weight never gets one, and that is the ordinary LoRA shape.
            let Some((grad, grad_shape)) = grads.get(p.id()) else {
                continue;
            };
            if grad.len() != p.n_elems() {
                return Err(GlTrainError::ShapeMismatch {
                    expected: p.shape().to_vec(),
                    got: grad_shape.clone(),
                });
            }
            // A frozen parameter must never be written, even if something
            // upstream produced a gradient for it.
            if !p.is_trainable() {
                continue;
            }

            let n = p.n_elems();
            let lr = self.groups.effective_lr(self.config.lr, p.name());

            let state = match self.state.entry(p.id()) {
                std::collections::hash_map::Entry::Occupied(e) => e.into_mut(),
                // Lazily allocated: a parameter that never receives a gradient
                // never costs 2x its size.
                std::collections::hash_map::Entry::Vacant(e) => {
                    e.insert(OPAdamWMoments::<B>::zeros(n)?)
                }
            };

            // Everything below runs through `Backend`, on raw storage. Using
            // tracked `Tensor` ops here would append a node per parameter per
            // step to whatever tape the parameter carries (finding F4).
            let g = B::from_vec(grad.clone())?;

            // m = b1*m + (1-b1)*g
            let m = B::add(
                &B::mul_scalar(&state.m, self.config.beta1 as f32, n)?,
                &B::mul_scalar(&g, (1.0 - self.config.beta1) as f32, n)?,
                n,
            )?;
            // v = b2*v + (1-b2)*g^2
            let g2 = B::mul(&g, &g, n)?;
            let v = B::add(
                &B::mul_scalar(&state.v, self.config.beta2 as f32, n)?,
                &B::mul_scalar(&g2, (1.0 - self.config.beta2) as f32, n)?,
                n,
            )?;

            let m_hat = B::mul_scalar(&m, (1.0 / bc1) as f32, n)?;
            let v_hat = B::mul_scalar(&v, (1.0 / bc2) as f32, n)?;

            // denom = sqrt(v^) + eps. `eps` is outside the root, matching the
            // paper and PyTorch. `B::sqrt` rejects a negative input, which for
            // a sum of squares means the state is already corrupt.
            let denom = B::add_scalar(&B::sqrt(&v_hat, n)?, self.config.eps as f32, n)?;
            let adaptive = B::div(&m_hat, &denom, n)?;

            // theta_t = theta_{t-1} - lr*(adaptive + lambda*theta_{t-1}).
            // The decay reads the PRE-update weight. Scaling theta after the
            // adaptive step instead leaves a spurious +lr^2*wd*u term.
            let theta = B::from_vec(p.to_vec()?)?;
            let decay = B::mul_scalar(&theta, self.config.weight_decay as f32, n)?;
            let update = B::mul_scalar(&B::add(&adaptive, &decay, n)?, lr as f32, n)?;
            let new_theta = B::sub(&theta, &update, n)?;

            // Commit the moments only once the update itself succeeded, so a
            // failed step leaves the optimizer where it was.
            state.m = m;
            state.v = v;

            // The one write path. `set_data` refuses a frozen parameter and a
            // length mismatch, so there is no second `&mut` accessor to keep in
            // sync.
            p.set_data(B::to_vec(&new_theta)?)?;
        }
        Ok(())
    }

    fn groups(&self) -> &[VLParamGroup] {
        self.groups.groups()
    }

    fn add_group(&mut self, group: VLParamGroup) -> Result<()> {
        self.groups.add_group(group)
    }

    fn assign_group(&mut self, param_name: &str, group_name: &str) -> Result<()> {
        self.groups.assign(param_name, group_name)
    }

    fn state_tensors(&self, params: &[&TPParameter<B>]) -> Result<Vec<VLNamedTensor>> {
        let mut out = Vec::with_capacity(params.len() * 2 + 1);
        for p in params {
            // A parameter with no state has not stepped yet. Skipping it is
            // correct: a zeroed entry would be indistinguishable on load from
            // one that had genuinely converged to zero moments.
            let Some(state) = self.state.get(&p.id()) else {
                continue;
            };
            let shape = p.shape().to_vec();
            out.push(VLNamedTensor::new(
                Self::state_key(p.name(), "m"),
                B::to_vec(&state.m)?,
                shape.clone(),
            ));
            out.push(VLNamedTensor::new(
                Self::state_key(p.name(), "v"),
                B::to_vec(&state.v)?,
                shape,
            ));
        }
        // Without this, a resumed run restarts the bias correction at t=1 and
        // scales its first update by roughly 0.3x for no visible reason.
        out.push(VLNamedTensor::new(
            STEP_COUNT_KEY,
            vec![self.step_count as f32],
            vec![1],
        ));
        Ok(out)
    }

    fn load_state(&mut self, params: &[&TPParameter<B>], named: &[VLNamedTensor]) -> Result<()> {
        let by_name: HashMap<&str, &VLNamedTensor> =
            named.iter().map(|t| (t.name.as_str(), t)).collect();

        for p in params {
            let m_key = Self::state_key(p.name(), "m");
            let v_key = Self::state_key(p.name(), "v");
            let (Some(m), Some(v)) = (
                by_name.get(m_key.as_str()),
                by_name.get(v_key.as_str()),
            ) else {
                // Half a pair is a corrupt file, not a parameter that has not
                // stepped. Say which half is missing.
                if by_name.contains_key(m_key.as_str()) != by_name.contains_key(v_key.as_str()) {
                    return Err(GlTrainError::Checkpoint(format!(
                        "optimizer state for '{}' has only one of its two moments",
                        p.name()
                    )));
                }
                continue;
            };
            // Shape, not element count: a transposed [896, 4864] and a correct
            // [4864, 896] have the same length and would load silently.
            for (slot, t) in [("m", m), ("v", v)] {
                if t.shape != p.shape() {
                    return Err(GlTrainError::ShapeMismatch {
                        expected: p.shape().to_vec(),
                        got: t.shape.clone(),
                    });
                }
                if t.data.len() != p.n_elems() {
                    return Err(GlTrainError::Checkpoint(format!(
                        "optimizer state '{}.{slot}' holds {} values for a {}-element parameter",
                        p.name(),
                        t.data.len(),
                        p.n_elems()
                    )));
                }
            }
            // Re-keyed here, and only here: name on disk, TensorId in memory.
            self.state.insert(
                p.id(),
                OPAdamWMoments {
                    m: B::from_vec(m.data.clone())?,
                    v: B::from_vec(v.data.clone())?,
                },
            );
        }

        if let Some(sc) = by_name.get(STEP_COUNT_KEY) {
            let raw = *sc.data.first().ok_or_else(|| {
                GlTrainError::Checkpoint(format!("'{STEP_COUNT_KEY}' entry is empty"))
            })?;
            if raw < 0.0 {
                return Err(GlTrainError::Checkpoint(format!(
                    "'{STEP_COUNT_KEY}' is negative ({raw})"
                )));
            }
            self.step_count = raw as usize;
        }
        Ok(())
    }

    fn capability(&self) -> &'static VLOptimizerCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>> {
    let d = VLAdamWConfig::default();
    Ok(Box::new(OPAdamW::<B>::new(VLAdamWConfig {
        lr: spec.lr.unwrap_or(d.lr),
        beta1: spec.beta1.unwrap_or(d.beta1),
        beta2: spec.beta2.unwrap_or(d.beta2),
        eps: spec.eps.unwrap_or(d.eps),
        weight_decay: spec.weight_decay.unwrap_or(d.weight_decay),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::tape::Tape;
    use crate::backend::GlProc;
    use crate::tensor::Tensor;
    use std::sync::{Arc, Mutex};

    /// Optimizer arithmetic is a short chain of f32 elementwise ops with no
    /// accumulation over a long axis, so only rounding separates it from the
    /// f64 hand computation.
    const TOL_OPTIM: f32 = 1e-5;

    /// A one-element trainable parameter holding `value`.
    fn scalar_param(name: &str, value: f32) -> TPParameter<GlProc> {
        TPParameter::trainable(name, Tensor::<GlProc>::from_vec(vec![value], &[1]).unwrap())
    }

    /// A gradient store with one entry, as `finish_step` would return it.
    fn grads_for(id: TensorId, g: f32) -> VLGradStore {
        let mut s = VLGradStore::new();
        s.accumulate(id, vec![g], vec![1]).unwrap();
        s
    }

    /// The anchor test. Every term computed by hand in f64 for one step on a
    /// single scalar, so a sign error or a misplaced bias correction cannot
    /// hide behind a plausible loss curve.
    ///
    /// theta_0 = 1.0, g = 0.5, lr = 0.1, b1 = 0.9, b2 = 0.999,
    /// eps = 1e-8, lambda = 0.01
    ///
    ///   m_1  = 0.9*0 + 0.1*0.5                 = 0.05
    ///   v_1  = 0.999*0 + 0.001*0.25            = 0.00025
    ///   bc1  = 1 - 0.9^1                       = 0.1
    ///   bc2  = 1 - 0.999^1                     = 0.001
    ///   m^   = 0.05 / 0.1                      = 0.5
    ///   v^   = 0.00025 / 0.001                 = 0.25
    ///   sqrt(v^) + eps                         = 0.50000001
    ///   adaptive = 0.5 / 0.50000001            ~ 0.99999998
    ///   decay    = 0.01 * 1.0                  = 0.01
    ///   theta_1  = 1.0 - 0.1*(0.99999998+0.01) = 0.899000002
    ///
    /// Cross-checked against an independent f64 evaluation rather than trusted
    /// from the implementation, which would only prove the code agrees with
    /// itself.
    #[test]
    fn adamw_update_matches_the_hand_computed_first_step() {
        let mut p = scalar_param("w", 1.0);
        let g = grads_for(p.id(), 0.5);
        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        });

        opt.step(&mut [&mut p], &g).unwrap();

        // Compared in f64: the derivation above is exact to more digits than an
        // f32 literal can carry, and clippy is right to reject one that long.
        let got = p.to_vec().unwrap()[0] as f64;
        let want = 0.899_000_002_f64;
        assert!(
            (got - want).abs() < TOL_OPTIM as f64,
            "theta_1 = {got}, hand-computed {want}"
        );
    }

    /// The decay reads theta_{t-1}. The stale sketch in STUMMAN_PLAN.md §3.4
    /// scales theta *after* the adaptive step, which multiplies the decay into
    /// the update as well and leaves a +lr^2*wd*u error.
    ///
    /// With the numbers above, the buggy order gives
    /// `(1.0 - 0.1*0.99999998) * (1 - 0.1*0.01) = 0.899100...`, which differs
    /// from the correct 0.898999998 by ~1e-4: far above TOL_OPTIM, so this
    /// test genuinely separates the two.
    #[test]
    fn adamw_weight_decay_uses_the_pre_update_theta() {
        let mut p = scalar_param("w", 1.0);
        let g = grads_for(p.id(), 0.5);
        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
            lr: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        });
        opt.step(&mut [&mut p], &g).unwrap();

        let got = p.to_vec().unwrap()[0] as f64;
        let correct = 0.899_000_002_f64;
        let decay_applied_after = (1.0_f64 - 0.1 * 0.999_999_98) * (1.0 - 0.1 * 0.01);

        // The two orders differ by lr^2*wd*u, which is 9.99999980e-5 here, and
        // TOL_OPTIM is 1e-5. The boundary is the geometric mean of the two, so
        // both assertions clear it by ~3x and neither decides on a rounding
        // edge. (A first pass used 5e-5 and the separability guard below
        // rejected it: 2*5e-5 is 1e-4, a hair above the actual separation.)
        const WRONG_ORDER_MARGIN: f64 = 3e-5;
        assert!(
            (correct - decay_applied_after).abs() > 2.0 * WRONG_ORDER_MARGIN,
            "the two orders must be separable at all: they differ by {}",
            (correct - decay_applied_after).abs()
        );
        assert!(
            (got - correct).abs() < TOL_OPTIM as f64,
            "got {got}, correct order gives {correct}"
        );
        assert!(
            (got - decay_applied_after).abs() > WRONG_ORDER_MARGIN,
            "got {got}, which matches the WRONG decay-after order {decay_applied_after}"
        );
    }

    /// Without bias correction the first step would be scaled by
    /// `m_1/sqrt(v_1) = 0.05/0.0158 ~ 3.16` instead of `~1.0`. The corrected
    /// update moves theta by about `lr`; the uncorrected one by about `3*lr`.
    #[test]
    fn adamw_applies_bias_correction_on_the_first_step() {
        let mut p = scalar_param("w", 1.0);
        let g = grads_for(p.id(), 0.5);
        // No decay, so the whole move is the adaptive term.
        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..VLAdamWConfig::default()
        });
        opt.step(&mut [&mut p], &g).unwrap();

        let moved = 1.0 - p.to_vec().unwrap()[0];
        // Corrected: m^/(sqrt(v^)+eps) = 1.0, so the move is exactly lr.
        assert!(
            (moved - 0.1).abs() < TOL_OPTIM,
            "moved {moved}; bias-corrected AdamW moves by lr on step 1"
        );
        // Uncorrected would be ~0.316, which this comfortably excludes.
        assert!(moved < 0.2, "moved {moved}, looks uncorrected");
    }

    #[test]
    fn adamw_bias_correction_denominators_are_one_minus_beta_to_the_t() {
        // bc1 = 1 - 0.9^1 = 0.1 exactly; bc2 = 1 - 0.999^1 = 0.001 exactly.
        // Stated as a test so the constants in the anchor above are pinned.
        let c = VLAdamWConfig::default();
        assert!((1.0 - c.beta1.powi(1) - 0.1).abs() < 1e-12);
        assert!((1.0 - c.beta2.powi(1) - 0.001).abs() < 1e-12);
    }

    /// LoRA+ in one test: the `lora_b` group gets 2x the base rate, and nothing
    /// about the parameter itself differs.
    #[test]
    fn adamw_param_groups_scale_the_learning_rate() {
        let mut a = scalar_param("lora_a", 1.0);
        let mut b = scalar_param("lora_b", 1.0);
        let mut g = VLGradStore::new();
        g.accumulate(a.id(), vec![0.5], vec![1]).unwrap();
        g.accumulate(b.id(), vec![0.5], vec![1]).unwrap();

        let mut opt = OPAdamW::<GlProc>::new(VLAdamWConfig {
            lr: 0.1,
            weight_decay: 0.0,
            ..VLAdamWConfig::default()
        });
        opt.add_group(VLParamGroup::new("lora_b", 2.0)).unwrap();
        opt.assign_group("lora_b", "lora_b").unwrap();
        assert_eq!(opt.effective_lr("lora_a"), 0.1);
        assert_eq!(opt.effective_lr("lora_b"), 0.2);

        opt.step(&mut [&mut a, &mut b], &g).unwrap();

        let moved_a = 1.0 - a.to_vec().unwrap()[0];
        let moved_b = 1.0 - b.to_vec().unwrap()[0];
        assert!(
            (moved_b - 2.0 * moved_a).abs() < TOL_OPTIM,
            "lora_b moved {moved_b}, expected 2x lora_a's {moved_a}"
        );
    }

    /// A frozen parameter is skipped even when a gradient exists for its ID.
    /// `set_data` would refuse anyway; this checks `step` never gets that far,
    /// so a frozen base weight cannot turn a training run into an error.
    #[test]
    fn adamw_never_updates_a_frozen_parameter() {
        let mut p = TPParameter::frozen(
            "base",
            Tensor::<GlProc>::from_vec(vec![1.0], &[1]).unwrap(),
        );
        let g = grads_for(p.id(), 0.5);
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        opt.step(&mut [&mut p], &g).expect("a frozen param is skipped, not an error");
        assert_eq!(p.to_vec().unwrap()[0], 1.0, "frozen weight was written");
    }

    /// A parameter with no gradient this step is skipped, not an error, and
    /// costs no state. This is the ordinary LoRA shape, not an edge case.
    #[test]
    fn adamw_skips_a_parameter_with_no_gradient_and_allocates_no_state_for_it() {
        let mut p = scalar_param("w", 1.0);
        let empty = VLGradStore::new();
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        opt.step(&mut [&mut p], &empty).unwrap();
        assert_eq!(p.to_vec().unwrap()[0], 1.0);
        assert!(
            opt.moments(p.id()).is_none(),
            "state was allocated for a parameter that never stepped"
        );
    }

    /// F4: the optimizer must not run on tracked `Tensor` ops. If it did, the
    /// tape would grow by a node per parameter per step, forever, and pollute
    /// the next backward pass.
    #[test]
    fn adamw_step_records_nothing_on_the_tape() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let mut p = scalar_param("w", 1.0);
        // Make the parameter genuinely tape-tracked, the state F4 warns about.
        let _tracked = p.tracked(&tape);
        let before = Tape::lock(&tape).len();

        let g = grads_for(p.id(), 0.5);
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        opt.step(&mut [&mut p], &g).unwrap();

        let after = Tape::lock(&tape).len();
        assert_eq!(after, before, "step() appended {} node(s)", after - before);
    }

    /// F2: `TensorId` is process-global and not persistable, so state has to
    /// survive a round trip through **names**. The reloading parameters here
    /// carry different IDs, exactly as they would after a restart.
    #[test]
    fn adamw_state_round_trips_keyed_by_name_not_by_tensor_id() {
        let mut p = scalar_param("w", 1.0);
        let g = grads_for(p.id(), 0.5);
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        opt.step(&mut [&mut p], &g).unwrap();

        let saved = opt.state_tensors(&[&p]).unwrap();
        assert!(saved.iter().any(|t| t.name == "w.m"));
        assert!(saved.iter().any(|t| t.name == "w.v"));
        assert!(
            !saved.iter().any(|t| t.name.parse::<usize>().is_ok()),
            "a bare numeric key would be a TensorId, which is not persistable"
        );

        // A fresh parameter with the same name and a different ID.
        let fresh = scalar_param("w", 1.0);
        assert_ne!(fresh.id(), p.id(), "the test needs a genuinely new ID");

        let mut reloaded = OPAdamW::<GlProc>::with_lr(0.1);
        reloaded.load_state(&[&fresh], &saved).unwrap();

        let orig = opt.moments(p.id()).unwrap();
        let back = reloaded
            .moments(fresh.id())
            .expect("state must be found under the new ID");
        assert!((GlProc::to_vec(&orig.m).unwrap()[0] - GlProc::to_vec(&back.m).unwrap()[0]).abs() < TOL_OPTIM);
        assert!((GlProc::to_vec(&orig.v).unwrap()[0] - GlProc::to_vec(&back.v).unwrap()[0]).abs() < TOL_OPTIM);
        assert_eq!(reloaded.step_count(), 1, "the step counter must survive");
    }

    /// A resumed run must continue the bias-correction schedule. Restarting at
    /// t=1 would scale the first update by roughly 0.3x with nothing to show
    /// for it in the logs.
    #[test]
    fn adamw_step_count_survives_a_state_round_trip() {
        let mut p = scalar_param("w", 1.0);
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        let g = grads_for(p.id(), 0.5);
        for _ in 0..5 {
            opt.step(&mut [&mut p], &g).unwrap();
        }
        assert_eq!(opt.step_count(), 5);

        let saved = opt.state_tensors(&[&p]).unwrap();
        let mut reloaded = OPAdamW::<GlProc>::with_lr(0.1);
        reloaded.load_state(&[&p], &saved).unwrap();
        assert_eq!(reloaded.step_count(), 5);
    }

    /// Shape, not element count. A transposed save has the same length and
    /// would otherwise load silently, which is a real bug class in this repo.
    #[test]
    fn adamw_load_state_rejects_a_transposed_shape() {
        let p = TPParameter::trainable(
            "w",
            Tensor::<GlProc>::from_vec(vec![0.0; 6], &[2, 3]).unwrap(),
        );
        let saved = vec![
            VLNamedTensor::new("w.m", vec![0.0; 6], vec![3, 2]),
            VLNamedTensor::new("w.v", vec![0.0; 6], vec![3, 2]),
        ];
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        let err = opt.load_state(&[&p], &saved);
        assert!(
            matches!(err, Err(GlTrainError::ShapeMismatch { .. })),
            "a [3,2] state for a [2,3] parameter must be rejected, got {err:?}"
        );
    }

    /// Half a moment pair is a corrupt file. Silently treating it as "has not
    /// stepped yet" would resume from a half-initialized optimizer.
    #[test]
    fn adamw_load_state_rejects_a_half_written_moment_pair() {
        let p = scalar_param("w", 1.0);
        let saved = vec![VLNamedTensor::new("w.m", vec![0.05], vec![1])];
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        assert!(opt.load_state(&[&p], &saved).is_err());
    }

    /// Two steps must not equal one step twice as large. This pins that the
    /// moments actually carry between calls rather than being rebuilt.
    #[test]
    fn adamw_moments_accumulate_across_steps() {
        let mut p = scalar_param("w", 1.0);
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        let g = grads_for(p.id(), 0.5);
        opt.step(&mut [&mut p], &g).unwrap();
        let m_after_1 = GlProc::to_vec(&opt.moments(p.id()).unwrap().m).unwrap()[0];
        opt.step(&mut [&mut p], &g).unwrap();
        let m_after_2 = GlProc::to_vec(&opt.moments(p.id()).unwrap().m).unwrap()[0];

        // m_1 = 0.05, m_2 = 0.9*0.05 + 0.1*0.5 = 0.095
        assert!((m_after_1 - 0.05).abs() < TOL_OPTIM, "m_1 = {m_after_1}");
        assert!((m_after_2 - 0.095).abs() < TOL_OPTIM, "m_2 = {m_after_2}");
        assert_eq!(opt.step_count(), 2);
    }

    #[test]
    fn adamw_rejects_a_gradient_of_the_wrong_length() {
        let mut p = TPParameter::trainable(
            "w",
            Tensor::<GlProc>::from_vec(vec![1.0, 2.0], &[2]).unwrap(),
        );
        let mut g = VLGradStore::new();
        g.accumulate(p.id(), vec![0.5], vec![1]).unwrap();
        let mut opt = OPAdamW::<GlProc>::with_lr(0.1);
        assert!(opt.step(&mut [&mut p], &g).is_err());
    }

    #[test]
    fn adamw_defaults_match_pytorch() {
        let c = VLAdamWConfig::default();
        assert_eq!(c.lr, 1e-3);
        assert_eq!(c.beta1, 0.9);
        assert_eq!(c.beta2, 0.999);
        assert_eq!(c.eps, 1e-8);
        assert_eq!(c.weight_decay, 1e-2);
    }
}
