//! Stummañ Gwellaer: Lion optimizer. **STUB, with the real state shape.**
//!
//! Chen et al. 2023 (arXiv:2302.06675), EvoLved Sign Momentum:
//!
//! ```text
//! update  = sign( b1*m_{t-1} + (1-b1)*g_t )
//! theta_t = theta_{t-1} - lr*( update + lambda*theta_{t-1} )
//! m_t     = b2*m_{t-1} + (1-b2)*g_t
//! ```
//!
//! # This is not AdamW with different numbers
//!
//! Two things make it a genuinely different shape, and a stub that blurred
//! either would teach the next wave nothing:
//!
//! **One buffer, not two.** There is no second moment. Lion stores only `m`,
//! which is half of AdamW's state and the entire memory argument for using it.
//! [`OPLion::allocate_state`] allocates exactly one buffer per parameter. A
//! placeholder `v` "in case it is needed later" would misrepresent the claim
//! the optimizer exists to make.
//!
//! **`beta1` and `beta2` are not the same knob AdamW's are.** `b1` shapes the
//! *update* (the interpolation that gets sign-ed); `b2` shapes what the
//! *momentum remembers*. The defaults are `0.9 / 0.99`, not AdamW's
//! `0.9 / 0.999`.
//!
//! # The learning rate and the decay have to move together
//!
//! Because the update is a `sign`, every step has magnitude `lr` regardless of
//! the gradient. The published guidance is an `lr` 3-10x smaller than AdamW's
//! with `weight_decay` correspondingly 3-10x larger: the effective decay is
//! `lr * lambda`, so changing one alone silently changes the actual
//! regularization strength when someone swaps optimizers.

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

/// Why this is a stub, in the error a caller sees.
const STUB_REASON: &str =
    "the update rule is written and Backend::sign exists, but no measured comparison against \
     OPAdamW has been run on this crate's workloads; shipping a second update rule unvalidated \
     would make a divergence impossible to attribute";
const STUB_MILESTONE: &str = "M3";

/// Lion's capability record.
pub static CAPABILITY: &VLOptimizerCapability = &VLOptimizerCapability {
    id: "lion",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    state_shape: ENOptimizerStateShape::Fixed {
        buffers_per_param: 1,
    },
    memory_multiplier: 1.0,
    source: "Chen et al. 2023, arXiv:2302.06675",
};

/// Lion's hyperparameters.
///
/// The defaults are Lion's own, not AdamW's. See the module docs.
#[derive(Debug, Clone, PartialEq)]
pub struct VLLionConfig {
    /// Base learning rate. Should be 3-10x smaller than AdamW's.
    pub lr: f64,
    /// Interpolation for the *update* that gets sign-ed.
    pub beta1: f64,
    /// Decay for what the *momentum* remembers. `0.99`, not AdamW's `0.999`.
    pub beta2: f64,
    /// Decoupled weight decay. Should be 3-10x larger than AdamW's, since the
    /// effective decay is `lr * weight_decay` and `lr` went down.
    pub weight_decay: f64,
}

impl Default for VLLionConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.99,
            weight_decay: 1e-1,
        }
    }
}

/// Sign-momentum optimizer. Constructs and introspects; refuses to compute.
pub struct OPLion<B: Backend> {
    config: VLLionConfig,
    groups: GroupTable,
    /// **One** buffer per parameter. Lion has no second moment.
    state: HashMap<TensorId, B::Storage>,
}

impl<B: Backend> OPLion<B> {
    /// An optimizer with the given hyperparameters.
    pub fn new(config: VLLionConfig) -> Self {
        Self {
            config,
            groups: GroupTable::new(),
            state: HashMap::new(),
        }
    }

    /// The hyperparameters in use.
    pub fn config(&self) -> &VLLionConfig {
        &self.config
    }

    /// Allocate the real state for these parameters.
    ///
    /// [`Optimizer::step`] refuses, so nothing allocates lazily the way
    /// [`crate::optim::OPAdamW`] does. This exists so the state shape is a
    /// testable fact today rather than a claim in a doc comment: one buffer
    /// per parameter, the parameter's own size.
    pub fn allocate_state(&mut self, params: &[&TPParameter<B>]) -> Result<()> {
        for p in params {
            self.state.insert(p.id(), B::zeros(p.n_elems())?);
        }
        Ok(())
    }

    /// The momentum buffer for a parameter, once allocated.
    pub fn momentum(&self, id: TensorId) -> Option<&B::Storage> {
        self.state.get(&id)
    }

    /// How many buffers of parameter size this optimizer holds per parameter.
    /// One. AdamW holds two.
    pub fn buffers_per_param(&self) -> usize {
        1
    }
}

impl<B: Backend> Optimizer<B> for OPLion<B> {
    fn step(&mut self, _params: &mut [&mut TPParameter<B>], _grads: &VLGradStore) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "lion",
            reason: STUB_REASON,
            milestone: STUB_MILESTONE,
        })
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
        // Serializing is sound even though stepping is not: the state shape is
        // final, so a file written now stays readable when `step` lands.
        let mut out = Vec::with_capacity(params.len());
        for p in params {
            let Some(m) = self.state.get(&p.id()) else {
                continue;
            };
            out.push(VLNamedTensor::new(
                format!("{}.m", p.name()),
                B::to_vec(m)?,
                p.shape().to_vec(),
            ));
        }
        Ok(out)
    }

    fn load_state(&mut self, params: &[&TPParameter<B>], named: &[VLNamedTensor]) -> Result<()> {
        let by_name: HashMap<&str, &VLNamedTensor> =
            named.iter().map(|t| (t.name.as_str(), t)).collect();
        for p in params {
            let key = format!("{}.m", p.name());
            let Some(t) = by_name.get(key.as_str()) else {
                continue;
            };
            if t.shape != p.shape() {
                return Err(GlTrainError::ShapeMismatch {
                    expected: p.shape().to_vec(),
                    got: t.shape.clone(),
                });
            }
            self.state.insert(p.id(), B::from_vec(t.data.clone())?);
        }
        Ok(())
    }

    fn capability(&self) -> &'static VLOptimizerCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>> {
    let d = VLLionConfig::default();
    Ok(Box::new(OPLion::<B>::new(VLLionConfig {
        lr: spec.lr.unwrap_or(d.lr),
        beta1: spec.beta1.unwrap_or(d.beta1),
        beta2: spec.beta2.unwrap_or(d.beta2),
        weight_decay: spec.weight_decay.unwrap_or(d.weight_decay),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;
    use crate::tensor::Tensor;

    fn param(name: &str, n: usize) -> TPParameter<GlProc> {
        TPParameter::trainable(name, Tensor::<GlProc>::zeros(&[n]).unwrap())
    }

    /// A stub must refuse, never quietly do something else. Returning AdamW's
    /// update here would be "asked for Lion, got Adam" with no error anywhere.
    #[test]
    fn lion_step_returns_unsupported() {
        let mut opt = OPLion::<GlProc>::new(VLLionConfig::default());
        let mut p = param("w", 4);
        let err = opt.step(&mut [&mut p], &VLGradStore::new());
        assert!(matches!(
            err,
            Err(GlTrainError::Unsupported { skill: "lion", .. })
        ));
    }

    /// The entire memory argument for Lion. A second buffer here would be a
    /// factual misstatement about the optimizer, not a harmless placeholder.
    #[test]
    fn lion_allocates_one_buffer_per_parameter_not_two() {
        let mut opt = OPLion::<GlProc>::new(VLLionConfig::default());
        let p = param("w", 8);
        opt.allocate_state(&[&p]).unwrap();

        let m = opt.momentum(p.id()).expect("momentum must be allocated");
        assert_eq!(GlProc::to_vec(m).unwrap().len(), 8);
        assert_eq!(opt.buffers_per_param(), 1);
        assert_eq!(
            CAPABILITY.state_shape,
            ENOptimizerStateShape::Fixed {
                buffers_per_param: 1
            }
        );
        assert_eq!(CAPABILITY.memory_multiplier, 1.0);
    }

    /// `beta2` is 0.99, not AdamW's 0.999. They are not the same knob: AdamW's
    /// decays a second moment, Lion's decays the momentum itself.
    #[test]
    fn lion_defaults_are_its_own_not_adamws() {
        let lion = VLLionConfig::default();
        let adamw = crate::optim::VLAdamWConfig::default();
        assert_eq!(lion.beta1, 0.9);
        assert_eq!(lion.beta2, 0.99);
        assert_ne!(lion.beta2, adamw.beta2);
        assert!(
            lion.lr < adamw.lr,
            "a sign-based update needs a smaller lr: every step has magnitude lr"
        );
        assert!(
            lion.weight_decay > adamw.weight_decay,
            "effective decay is lr*lambda, so lambda has to rise as lr falls"
        );
    }

    /// The state shape is final, so state written today stays readable when
    /// `step` lands. Only the update rule is missing.
    #[test]
    fn lion_state_round_trips_by_name() {
        let mut opt = OPLion::<GlProc>::new(VLLionConfig::default());
        let p = param("w", 3);
        opt.allocate_state(&[&p]).unwrap();

        let saved = opt.state_tensors(&[&p]).unwrap();
        assert_eq!(saved.len(), 1, "one buffer, so one entry: {saved:?}");
        assert_eq!(saved[0].name, "w.m");
        assert!(
            !saved.iter().any(|t| t.name == "w.v"),
            "Lion has no second moment to save"
        );

        let mut fresh = OPLion::<GlProc>::new(VLLionConfig::default());
        fresh.load_state(&[&p], &saved).unwrap();
        assert!(fresh.momentum(p.id()).is_some());
    }

    #[test]
    fn lion_reports_itself_as_a_stub_naming_the_milestone() {
        assert!(!CAPABILITY.status.is_full());
        assert!(matches!(
            CAPABILITY.status,
            ENSkillStatus::Stub {
                milestone: "M3",
                ..
            }
        ));
    }
}
