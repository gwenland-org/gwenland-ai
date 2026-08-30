//! Stummañ Gwiskadur: DoRA. **STUB — parameters are real, forward is not.**
//!
//! From Liu et al. 2024 (arXiv:2402.09353):
//!
//! > `W = m · V/‖V‖_c`, where `‖·‖_c` is the vector-wise norm of a matrix across
//! > each column.
//!
//! > Fine-tuning: `W' = m · (W0 + BA)/‖W0 + BA‖_c`
//!
//! `m` is initialised to `‖W0‖_c`, and the paper detaches the norm from the
//! gradient graph ("treat `‖V+ΔV‖_c` as a constant"), which it measures at
//! ~24.4% gradient-memory saving on LLaMA with no accuracy cost.
//!
//! # Why this is not "LoRA with an extra vector"
//!
//! DoRA renormalizes the **combined** weight. It is not `base_out + delta_out`,
//! so it cannot be computed without the base weight's *values* -- which is why
//! [`crate::nn::adapter::Adapter::forward`] takes
//! `base_weight: &Tensor<B>` rather than a precomputed base output. That
//! signature exists for this adapter. Nothing about the trait needs to change
//! when this stub is filled in.
//!
//! # What remains (M3)
//!
//! 1. **A column-norm op.** `‖·‖_c` reduces `[d_in, d_out]` to `[1, d_out]`
//!    along the `d_in` axis. There is no reduction-along-an-axis op in the
//!    crate: `sum` and `mean` collapse to a scalar `[1]`. This is the real
//!    blocker, and it is a tensor-layer gap, not an adapter one.
//! 2. **A detach-inside-the-graph path.** The paper's memory trick needs the
//!    norm treated as a constant while its *input* still receives gradients.
//!    `Tensor::detach` detaches a whole tensor from the tape, which is close but
//!    not the same thing: it produces a new leaf, so the graph downstream of it
//!    stops. What is needed is a node whose backward returns `None` for its
//!    input while forward still reads it.
//! 3. **`div` on tensors.** `Backend::div` exists as of M2 (the optimizer needs
//!    it); `Tensor::div` does not, because nothing tracked has needed division
//!    yet. Adding it means writing its backward: `d(a/b) = da/b - a·db/b²`.
//!
//! Ordering note: (1) and (3) are shared with several later adapters, so they
//! belong to a tensor-ops wave, not to DoRA specifically.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for DoRA.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "dora",
    status: ENSkillStatus::Stub {
        reason: "needs a column-wise norm reduction, Tensor::div, and a \
                 detach-in-graph node; all three are tensor-layer gaps",
        milestone: "M3",
    },
    trainable_params: "r * (d_in + d_out) + d_out",
    mergeable: true,
    // The distinguishing property: the forward pass reads W0 itself.
    requires_base_values: true,
    // ||W0 + BA||_c requires forming W0 + BA.
    materializes_delta: true,
    shares_params_across_layers: false,
    source: "Liu et al. 2024, arXiv:2402.09353",
};

/// DoRA adapter: magnitude/direction decomposition over a LoRA update.
///
/// The parameters are allocated with their researched shapes so a later wave
/// inherits them rather than re-deriving them. Only the compute path is missing.
pub struct LRDora<B: Backend> {
    a: TPParameter<B>,
    b: TPParameter<B>,
    /// Magnitude, one scalar per **output** column. Shape `[1, d_out]`.
    ///
    /// Initialised to zero here rather than to `‖W0‖_c`, because the base weight
    /// is not available at construction. A real implementation must set it from
    /// `W0` before the first step; leaving it at zero would collapse the layer.
    /// This is recorded in the stub's error rather than silently half-done.
    magnitude: TPParameter<B>,
    d_in: usize,
    d_out: usize,
    r: usize,
}

impl<B: Backend> LRDora<B> {
    /// Allocate DoRA's parameters at their researched shapes.
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        if spec.r == 0 || spec.r > spec.d_in.min(spec.d_out) {
            return Err(GlTrainError::InvalidOp(format!(
                "DoRA rank {} must be in 1..=min(d_in, d_out) = {}",
                spec.r,
                spec.d_in.min(spec.d_out)
            )));
        }
        let std = 1.0 / spec.r as f32;
        Ok(Self {
            a: TPParameter::trainable(
                "lora_a",
                Tensor::randn(&[spec.d_in, spec.r], std, spec.seed)?,
            ),
            b: TPParameter::trainable("lora_b", Tensor::zeros(&[spec.r, spec.d_out])?),
            magnitude: TPParameter::trainable("magnitude", Tensor::zeros(&[1, spec.d_out])?),
            d_in: spec.d_in,
            d_out: spec.d_out,
            r: spec.r,
        })
    }

    /// The magnitude vector, shape `[1, d_out]`.
    pub fn magnitude(&self) -> &TPParameter<B> {
        &self.magnitude
    }

    /// Rank.
    pub fn rank(&self) -> usize {
        self.r
    }

    /// Adapted layer dimensions, `(d_in, d_out)`.
    pub fn dims(&self) -> (usize, usize) {
        (self.d_in, self.d_out)
    }
}

impl<B: Backend> Adapter<B> for LRDora<B> {
    fn forward(
        &self,
        _x: &Tensor<B>,
        _base_weight: &Tensor<B>,
        _tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        Err(GlTrainError::Unsupported {
            skill: "dora",
            reason: "column-wise norm reduction, Tensor::div and a \
                     detach-in-graph node are not implemented",
            milestone: "M3",
        })
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        vec![&self.a, &self.b, &self.magnitude]
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        vec![&mut self.a, &mut self.b, &mut self.magnitude]
    }

    fn merge_into(&self, _base_weight: &mut Tensor<B>) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "dora",
            reason: "merging needs the same column-norm op the forward pass needs",
            milestone: "M3",
        })
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRDora::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    fn dora() -> LRDora<GlProc> {
        LRDora::new(&VLAdapterSpec::new(4, 3, 2, 7)).unwrap()
    }

    #[test]
    fn constructs_with_the_researched_parameter_shapes() {
        let d = dora();
        assert_eq!(d.a.shape(), &[4, 2], "A is [d_in, r]");
        assert_eq!(d.b.shape(), &[2, 3], "B is [r, d_out]");
        assert_eq!(
            d.magnitude().shape(),
            &[1, 3],
            "magnitude is one scalar per output column"
        );
    }

    #[test]
    fn has_three_parameters_not_two() {
        assert_eq!(dora().parameters().len(), 3);
    }

    #[test]
    fn trainable_count_matches_the_capability_formula() {
        // r*(d_in + d_out) + d_out = 2*(4+3) + 3 = 17
        let d = dora();
        let n: usize = d.parameters().iter().map(|p| p.n_elems()).sum();
        assert_eq!(n, 17);
    }

    #[test]
    fn forward_returns_unsupported_naming_the_milestone() {
        let d = dora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        let w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        match d.forward(&x, &w, &tape) {
            Err(GlTrainError::Unsupported { skill, milestone, .. }) => {
                assert_eq!(skill, "dora");
                assert_eq!(milestone, "M3");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    /// The forbidden behaviour, asserted directly: a DoRA request must never
    /// quietly produce a LoRA result.
    #[test]
    fn forward_does_not_silently_fall_back_to_lora() {
        let d = dora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        let w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(
            d.forward(&x, &w, &tape).is_err(),
            "a stub must refuse, not compute something else"
        );
    }

    #[test]
    fn merge_returns_unsupported() {
        let d = dora();
        let mut w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(matches!(
            d.merge_into(&mut w),
            Err(GlTrainError::Unsupported { .. })
        ));
    }

    #[test]
    fn capability_says_it_needs_base_values() {
        assert!(dora().capability().requires_base_values);
    }

    #[test]
    fn an_invalid_rank_is_rejected_at_construction() {
        assert!(LRDora::<GlProc>::new(&VLAdapterSpec::new(4, 3, 9, 1)).is_err());
    }
}
