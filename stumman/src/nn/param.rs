//! Stummañ Gwiskadur: named trainable parameter.
//!
//! A `TPParameter<B>` is a tensor plus the two things the tensor itself cannot
//! carry: a stable **name**, and a frozen/trainable flag.

use crate::autograd::node::TensorId;
use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// A named tensor the optimizer is allowed to update.
///
/// # Why the name is not optional
///
/// `TensorId` is process-global and explicitly not persistable
/// (`tensor.rs`: "Checkpoints must key on parameter names, never on these").
/// Every serialized artifact in this crate is keyed by name, so a parameter
/// without one cannot be saved, restored, or matched against a checkpoint. The
/// name is therefore a constructor argument, not a setter.
///
/// # Frozen parameters
///
/// A frozen parameter is a real parameter that simply never joins a tape. It
/// still has a name and shape, so a checkpoint can reference it and a validator
/// can check it, which is what a LoRA base weight needs. Freezing is not the
/// same as deleting.
#[derive(Clone)]
pub struct TPParameter<B: Backend> {
    name: String,
    tensor: Tensor<B>,
    trainable: bool,
}

impl<B: Backend> TPParameter<B> {
    /// A trainable parameter.
    pub fn trainable(name: impl Into<String>, tensor: Tensor<B>) -> Self {
        Self {
            name: name.into(),
            // A parameter owns its data, so it must not share a tape identity
            // with whatever produced the tensor.
            tensor: tensor.detach(),
            trainable: true,
        }
    }

    /// A frozen parameter. Never gets a gradient, never gets updated.
    pub fn frozen(name: impl Into<String>, tensor: Tensor<B>) -> Self {
        Self {
            name: name.into(),
            tensor: tensor.detach(),
            trainable: false,
        }
    }

    /// This parameter's name, the key it is stored under.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the optimizer should update it.
    pub fn is_trainable(&self) -> bool {
        self.trainable
    }

    /// Shape of the underlying tensor.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Element count.
    pub fn n_elems(&self) -> usize {
        self.tensor.n_elems()
    }

    /// The stable tensor ID gradients arrive under.
    pub fn id(&self) -> TensorId {
        self.tensor.id()
    }

    /// The raw, untracked tensor. Use this for a frozen operand.
    pub fn tensor(&self) -> &Tensor<B> {
        &self.tensor
    }

    /// Copy the parameter's values to the host.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        self.tensor.to_vec()
    }

    /// A tape-tracked view of this parameter, for use in a forward pass.
    ///
    /// Returns a clone carrying `tape`, or the plain untracked tensor when the
    /// parameter is frozen. Frozen operands are the ordinary LoRA case and
    /// `record_op` handles them by design (KL-003).
    ///
    /// # Called once per forward pass, on purpose
    ///
    /// `Tensor::clone` preserves the ID, and `with_grad` re-registers that ID
    /// with the tape. Since `Tape::clear()` drops all registrations between
    /// steps, the parameter has to re-register each step, and this is where that
    /// happens. The ID is stable across every call, so gradients always land in
    /// the same slot and optimizer state keyed on it stays valid.
    pub fn tracked(&self, tape: &Arc<Mutex<Tape>>) -> Tensor<B> {
        if self.trainable {
            self.tensor.clone().with_grad(tape.clone())
        } else {
            self.tensor.clone()
        }
    }

    /// Overwrite the parameter's values, keeping name, shape and ID.
    ///
    /// The optimizer's write path. Rejects a length mismatch rather than
    /// resizing: a wrong-length update means the optimizer state and the
    /// parameter have diverged, and silently reshaping would bury that.
    ///
    /// Callers must satisfy KL-006 first. See
    /// [`Tensor::replace_data`][crate::tensor::Tensor] and the guard in
    /// [`crate::optim::Optimizer::step`].
    pub fn set_data(&mut self, data: Vec<f32>) -> Result<()> {
        if !self.trainable {
            return Err(GlTrainError::InvalidOp(format!(
                "parameter '{}' is frozen and cannot be updated",
                self.name
            )));
        }
        self.tensor.replace_data(data)
    }
}

impl<B: Backend> std::fmt::Debug for TPParameter<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TPParameter")
            .field("name", &self.name)
            .field("shape", &self.tensor.shape())
            .field("trainable", &self.trainable)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// Values round-trip through Vec<f32> with no arithmetic, so any difference
    /// is a real bug rather than accumulated error.
    const TOL_EXACT: f32 = 0.0;

    fn param() -> TPParameter<GlProc> {
        let t = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        TPParameter::trainable("w", t)
    }

    #[test]
    fn trainable_parameter_reports_its_name_and_shape() {
        let p = param();
        assert_eq!(p.name(), "w");
        assert_eq!(p.shape(), &[2, 2]);
        assert!(p.is_trainable());
    }

    #[test]
    fn tracked_view_keeps_the_same_tensor_id_across_calls() {
        // This is what lets optimizer state survive tape.clear() between steps.
        let p = param();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let a = p.tracked(&tape);
        let b = p.tracked(&tape);
        assert_eq!(a.id(), b.id());
        assert_eq!(a.id(), p.id());
    }

    #[test]
    fn tracked_view_of_a_trainable_param_requires_grad() {
        let p = param();
        let tape = Arc::new(Mutex::new(Tape::new()));
        assert!(p.tracked(&tape).requires_grad());
    }

    #[test]
    fn tracked_view_of_a_frozen_param_does_not_require_grad() {
        let t = Tensor::<GlProc>::zeros(&[2, 2]).unwrap();
        let p = TPParameter::frozen("base", t);
        let tape = Arc::new(Mutex::new(Tape::new()));
        assert!(!p.tracked(&tape).requires_grad());
    }

    #[test]
    fn set_data_overwrites_values_and_keeps_the_id() {
        let mut p = param();
        let id_before = p.id();
        p.set_data(vec![9.0, 9.0, 9.0, 9.0]).unwrap();
        assert_eq!(p.id(), id_before, "the ID must survive an update");
        for v in p.to_vec().unwrap() {
            assert!((v - 9.0).abs() <= TOL_EXACT);
        }
    }

    #[test]
    fn set_data_rejects_a_length_mismatch() {
        let mut p = param();
        assert!(p.set_data(vec![1.0, 2.0]).is_err());
    }

    #[test]
    fn set_data_on_a_frozen_parameter_is_rejected() {
        let t = Tensor::<GlProc>::zeros(&[2, 2]).unwrap();
        let mut p = TPParameter::frozen("base", t);
        let err = p.set_data(vec![1.0; 4]);
        assert!(err.is_err(), "a frozen parameter must refuse an update");
    }
}
