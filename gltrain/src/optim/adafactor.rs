//! Stummañ Gwellaer: Adafactor optimizer. **STUB, with the real state shape.**
//!
//! Shazeer & Stern 2018 (arXiv:1804.04235).
//!
//! # The state's shape depends on the parameter's rank
//!
//! This is the one genuinely new architectural fact among M2's three stub
//! optimizers, and it is why [`ENAdafactorMoment`] is an enum rather than a
//! struct:
//!
//! - **rank >= 2**, parameter `[n, m]`: there is no full `[n, m]` second
//!   moment. Adafactor keeps row sums `R: [n]` and column sums `C: [m]` and
//!   reconstructs `V^ = R*C / (1^T R)` on demand. Memory is `O(n+m)` instead of
//!   `O(n*m)`, which is the entire point of the optimizer.
//! - **rank 1**, a bias vector: the factorization does not apply at all. The
//!   full second moment is kept, the same shape as the parameter.
//!
//! A stub that allocated [`crate::optim::OPAdamWMoments`]-shaped state "as a
//! placeholder" would need a breaking rewrite the day this is implemented.
//! Allocating the right enum now leaves only the `step` body as new work.
//!
//! # `beta2` is a schedule, not a constant
//!
//! `b2_t = 1 - t^-0.8`. Deliberately **not** a `beta2: f64` config field: it
//! changes every step, and a field would invite a caller to set it once and
//! assume it held. [`VLAdafactorConfig`] has no such field.
//!
//! Other published constants, recorded so the eventual implementation does not
//! have to re-derive them: update clipping `U^ = U / max(1, RMS(U)/d)` with
//! `d = 1`; `eps1 = 1e-30`; `eps2 = 1e-3`. Note `eps2` is a **floor on
//! `RMS(theta)`** for the relative step size, not a denominator epsilon the way
//! AdamW's `eps` is.
//!
//! # An open question this stub deliberately does not settle
//!
//! Adafactor's headline feature is a **relative step size**:
//! `alpha_t = max(eps2, RMS(theta_{t-1})) * rho_t`. The learning rate adapts to
//! each parameter's own magnitude, while [`Optimizer::step`] assumes a fixed
//! `lr` per group. Whether that needs a trait change, or Adafactor computes its
//! own effective `lr` per parameter internally, belongs to whoever lands the
//! real implementation. Recording it beats silently deciding it by shipping a
//! signature that only ever fitted AdamW.

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
    "the factored second moment needs row/column reductions, which the Backend trait does not \
     have, and the relative step size does not fit Optimizer::step's fixed per-group lr";
const STUB_MILESTONE: &str = "M3";

/// Adafactor's capability record.
pub static CAPABILITY: &VLOptimizerCapability = &VLOptimizerCapability {
    id: "adafactor",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    state_shape: ENOptimizerStateShape::RankDependent,
    // Nominal. The true figure is O(n+m)/O(n*m) for a rank-2 parameter, so it
    // falls with size rather than being a constant: for a [1024, 1024] weight
    // it is ~0.002, for a rank-1 bias it is 1.0. RankDependent is the field
    // that carries the real answer.
    memory_multiplier: 0.01,
    source: "Shazeer & Stern 2018, arXiv:1804.04235",
};

/// Adafactor's hyperparameters.
///
/// No `beta2`: it is the schedule `1 - t^-0.8`, not a constant. See the module
/// docs.
#[derive(Debug, Clone, PartialEq)]
pub struct VLAdafactorConfig {
    /// Base learning rate, `rho_t` in the paper's relative step size.
    pub lr: f64,
    /// Regularization added to the squared gradient. `1e-30`.
    pub eps1: f64,
    /// Floor on `RMS(theta)` for the relative step size. `1e-3`. Not a
    /// denominator epsilon.
    pub eps2: f64,
    /// Update-clipping threshold `d`. `1.0`.
    pub clip_threshold: f64,
    /// Decoupled weight decay.
    pub weight_decay: f64,
}

impl Default for VLAdafactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            eps1: 1e-30,
            eps2: 1e-3,
            clip_threshold: 1.0,
            weight_decay: 0.0,
        }
    }
}

/// Adafactor's per-parameter second moment, whose shape follows the rank.
///
/// `EN` because a closed set of variants is this type's whole job, and the
/// variants are the fact the type exists to record.
pub enum ENAdafactorMoment<B: Backend> {
    /// Rank >= 2: row and column sums only, `O(n+m)`.
    Factored {
        /// Row sums, length `shape[0]`.
        row: B::Storage,
        /// Column sums, length `shape[1]`.
        col: B::Storage,
    },
    /// Rank 1: the factorization does not apply, so the full moment is kept.
    Full(B::Storage),
}

impl<B: Backend> ENAdafactorMoment<B> {
    /// The right variant for this parameter, picked from its rank.
    pub fn for_param(param: &TPParameter<B>) -> Result<Self> {
        let shape = param.shape();
        if shape.len() >= 2 {
            Ok(Self::Factored {
                row: B::zeros(shape[0])?,
                col: B::zeros(shape[1])?,
            })
        } else {
            Ok(Self::Full(B::zeros(param.n_elems())?))
        }
    }

    /// Elements actually stored. `n + m` when factored, `n * m` when not.
    pub fn stored_elems(&self) -> Result<usize> {
        Ok(match self {
            Self::Factored { row, col } => B::to_vec(row)?.len() + B::to_vec(col)?.len(),
            Self::Full(v) => B::to_vec(v)?.len(),
        })
    }

    /// Whether this parameter's moment is factored.
    pub fn is_factored(&self) -> bool {
        matches!(self, Self::Factored { .. })
    }
}

/// Sublinear-memory optimizer. Constructs and introspects; refuses to compute.
pub struct OPAdafactor<B: Backend> {
    config: VLAdafactorConfig,
    groups: GroupTable,
    state: HashMap<TensorId, ENAdafactorMoment<B>>,
}

impl<B: Backend> OPAdafactor<B> {
    /// An optimizer with the given hyperparameters.
    pub fn new(config: VLAdafactorConfig) -> Self {
        Self {
            config,
            groups: GroupTable::new(),
            state: HashMap::new(),
        }
    }

    /// The hyperparameters in use.
    pub fn config(&self) -> &VLAdafactorConfig {
        &self.config
    }

    /// Allocate the real, rank-dependent state for these parameters.
    ///
    /// Done at construction time rather than lazily in `step`, because the
    /// variant is a property of the parameter's shape and is knowable before
    /// any gradient exists. This is what makes the rank split a testable fact
    /// on M2 rather than a promise.
    pub fn allocate_state(&mut self, params: &[&TPParameter<B>]) -> Result<()> {
        for p in params {
            self.state.insert(p.id(), ENAdafactorMoment::for_param(p)?);
        }
        Ok(())
    }

    /// The moment held for a parameter, once allocated.
    pub fn moment(&self, id: TensorId) -> Option<&ENAdafactorMoment<B>> {
        self.state.get(&id)
    }
}

impl<B: Backend> Optimizer<B> for OPAdafactor<B> {
    fn step(&mut self, _params: &mut [&mut TPParameter<B>], _grads: &VLGradStore) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "adafactor",
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
        let mut out = Vec::new();
        for p in params {
            let Some(moment) = self.state.get(&p.id()) else {
                continue;
            };
            // The slot names differ by variant on purpose: a `.row`/`.col` pair
            // and a `.full` buffer are not interchangeable, and a loader that
            // saw the wrong one for a parameter's rank has read the wrong file.
            match moment {
                ENAdafactorMoment::Factored { row, col } => {
                    let r = B::to_vec(row)?;
                    let c = B::to_vec(col)?;
                    out.push(VLNamedTensor::new(
                        format!("{}.row", p.name()),
                        r.clone(),
                        vec![r.len()],
                    ));
                    out.push(VLNamedTensor::new(
                        format!("{}.col", p.name()),
                        c.clone(),
                        vec![c.len()],
                    ));
                }
                ENAdafactorMoment::Full(v) => {
                    out.push(VLNamedTensor::new(
                        format!("{}.full", p.name()),
                        B::to_vec(v)?,
                        p.shape().to_vec(),
                    ));
                }
            }
        }
        Ok(out)
    }

    fn load_state(&mut self, params: &[&TPParameter<B>], named: &[VLNamedTensor]) -> Result<()> {
        let by_name: HashMap<&str, &VLNamedTensor> =
            named.iter().map(|t| (t.name.as_str(), t)).collect();
        for p in params {
            let factored = p.shape().len() >= 2;
            if factored {
                let (Some(row), Some(col)) = (
                    by_name.get(format!("{}.row", p.name()).as_str()),
                    by_name.get(format!("{}.col", p.name()).as_str()),
                ) else {
                    continue;
                };
                if row.data.len() != p.shape()[0] || col.data.len() != p.shape()[1] {
                    return Err(GlTrainError::Checkpoint(format!(
                        "adafactor state for '{}' has row/col lengths {}/{}, expected {}/{}",
                        p.name(),
                        row.data.len(),
                        col.data.len(),
                        p.shape()[0],
                        p.shape()[1]
                    )));
                }
                self.state.insert(
                    p.id(),
                    ENAdafactorMoment::Factored {
                        row: B::from_vec(row.data.clone())?,
                        col: B::from_vec(col.data.clone())?,
                    },
                );
            } else {
                let Some(full) = by_name.get(format!("{}.full", p.name()).as_str()) else {
                    continue;
                };
                if full.shape != p.shape() {
                    return Err(GlTrainError::ShapeMismatch {
                        expected: p.shape().to_vec(),
                        got: full.shape.clone(),
                    });
                }
                self.state
                    .insert(p.id(), ENAdafactorMoment::Full(B::from_vec(full.data.clone())?));
            }
        }
        Ok(())
    }

    fn capability(&self) -> &'static VLOptimizerCapability {
        CAPABILITY
    }
}

/// Registry constructor.
///
/// `spec.beta1` and `spec.beta2` are ignored: Adafactor has no first moment by
/// default, and its second-moment decay is a schedule rather than a constant.
pub fn build<B: Backend>(spec: &VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>> {
    let d = VLAdafactorConfig::default();
    Ok(Box::new(OPAdafactor::<B>::new(VLAdafactorConfig {
        lr: spec.lr.unwrap_or(d.lr),
        weight_decay: spec.weight_decay.unwrap_or(d.weight_decay),
        ..d
    })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;
    use crate::tensor::Tensor;

    fn param(name: &str, shape: &[usize]) -> TPParameter<GlProc> {
        TPParameter::trainable(name, Tensor::<GlProc>::zeros(shape).unwrap())
    }

    #[test]
    fn adafactor_step_returns_unsupported() {
        let mut opt = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        let mut p = param("w", &[4, 4]);
        let err = opt.step(&mut [&mut p], &VLGradStore::new());
        assert!(matches!(
            err,
            Err(GlTrainError::Unsupported {
                skill: "adafactor",
                ..
            })
        ));
    }

    /// The whole point of the optimizer: a `[n, m]` weight costs `n + m`, not
    /// `n * m`. Getting this wrong now would mean a breaking rewrite later.
    #[test]
    fn adafactor_allocates_factored_state_for_a_rank_two_parameter() {
        let mut opt = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        let p = param("w", &[8, 32]);
        opt.allocate_state(&[&p]).unwrap();

        let m = opt.moment(p.id()).expect("state must be allocated");
        assert!(m.is_factored(), "a rank-2 parameter must factor");
        assert_eq!(m.stored_elems().unwrap(), 8 + 32);
        assert!(
            m.stored_elems().unwrap() < p.n_elems(),
            "factored state must be smaller than the parameter it tracks"
        );
    }

    /// The factorization does not apply to a bias vector, so the full moment
    /// is kept. Treating every parameter as factorable would be wrong here.
    #[test]
    fn adafactor_allocates_full_state_for_a_rank_one_parameter() {
        let mut opt = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        let p = param("b", &[16]);
        opt.allocate_state(&[&p]).unwrap();

        let m = opt.moment(p.id()).expect("state must be allocated");
        assert!(!m.is_factored(), "a rank-1 parameter cannot factor");
        assert_eq!(m.stored_elems().unwrap(), 16);
    }

    /// The two variants must coexist in one optimizer, which is exactly what a
    /// single fixed-shape state buffer could not express.
    #[test]
    fn adafactor_holds_both_variants_at_once() {
        let mut opt = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        let w = param("w", &[4, 6]);
        let b = param("b", &[6]);
        opt.allocate_state(&[&w, &b]).unwrap();
        assert!(opt.moment(w.id()).unwrap().is_factored());
        assert!(!opt.moment(b.id()).unwrap().is_factored());
        assert_eq!(CAPABILITY.state_shape, ENOptimizerStateShape::RankDependent);
    }

    /// `beta2` is the schedule `1 - t^-0.8`. A config field would invite a
    /// caller to set it once and assume it held.
    #[test]
    fn adafactor_config_has_no_beta2_field() {
        let c = VLAdafactorConfig::default();
        assert_eq!(c.eps1, 1e-30);
        assert_eq!(c.eps2, 1e-3);
        assert_eq!(c.clip_threshold, 1.0);
        // Compile-time evidence: the struct literal below names every field
        // there is, and beta2 is not among them.
        let _exhaustive = VLAdafactorConfig {
            lr: c.lr,
            eps1: c.eps1,
            eps2: c.eps2,
            clip_threshold: c.clip_threshold,
            weight_decay: c.weight_decay,
        };
    }

    /// The saved slot names differ by variant, so a file written for a rank-2
    /// parameter cannot be silently loaded into a rank-1 one.
    #[test]
    fn adafactor_state_names_record_which_variant_was_saved() {
        let mut opt = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        let w = param("w", &[4, 6]);
        let b = param("b", &[6]);
        opt.allocate_state(&[&w, &b]).unwrap();

        let saved = opt.state_tensors(&[&w, &b]).unwrap();
        let names: Vec<&str> = saved.iter().map(|t| t.name.as_str()).collect();
        assert!(names.contains(&"w.row"), "{names:?}");
        assert!(names.contains(&"w.col"), "{names:?}");
        assert!(names.contains(&"b.full"), "{names:?}");
        assert!(!names.contains(&"w.full"), "{names:?}");

        let mut fresh = OPAdafactor::<GlProc>::new(VLAdafactorConfig::default());
        fresh.load_state(&[&w, &b], &saved).unwrap();
        assert!(fresh.moment(w.id()).unwrap().is_factored());
        assert!(!fresh.moment(b.id()).unwrap().is_factored());
    }

    #[test]
    fn adafactor_reports_itself_as_a_stub_naming_the_milestone() {
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
