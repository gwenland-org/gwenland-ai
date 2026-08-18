//! Stummañ Gwiskadur: LoHa. **STUB — parameters are real, forward is not.**
//!
//! From Yeh et al. 2023 (arXiv:2309.14859), building on FedPara:
//!
//! > `ΔW = (B1 A1) ⊙ (B2 A2)`, with `B1,B2 ∈ R^{p×r}`, `A1,A2 ∈ R^{r×q}`, where
//! > `⊙` is the Hadamard product.
//!
//! > `h' = W0 h + b + γ[(B1 A1) ⊙ (B2 A2)] h`
//!
//! The point is rank: a Hadamard product of two rank-`r` matrices reaches rank
//! up to `r²`, against LoRA's `2r` for a comparable parameter count. The paper
//! notes `2r < r²` for `r > 2`.
//!
//! # The finding that changes a memory budget
//!
//! `(B1A1) ⊙ (B2A2)` **does not factor through `x`**. LoRA computes
//! `x @ A @ B` and never forms the `[d_in, d_out]` delta; there is no equivalent
//! regrouping for an elementwise product of two matrix products. The full delta
//! has to be materialized on every forward pass.
//!
//! So the usual reasoning -- "adapters are low-rank, therefore cheap" -- is
//! false here, and `materializes_delta: true` in the capability record exists to
//! say so before someone sizes a run on the LoRA assumption. LyCORIS mitigates
//! it with a "custom backward which will reconstruct B and A when actually
//! needed" rather than caching, which is a real design constraint on the
//! backward pass, not an implementation detail.
//!
//! # What remains (M3)
//!
//! 1. **Four-operand backward.** `d/dB1 = (dΔW ⊙ (B2A2)) @ A1^T` and so on for
//!    all four. Each gradient needs the *other* branch's product, so a naive
//!    implementation caches two `[d_in, d_out]` matrices per site. This is the
//!    part LyCORIS reworks; the crate's `BackwardFn` captures `Vec<f32>`
//!    snapshots, so the naive version would be memory-heavy but correct.
//! 2. **Nothing else.** The Hadamard product is `Tensor::mul`, which exists, and
//!    the matmuls exist. Unlike DoRA and LoCon, LoHa is **not blocked on a
//!    missing tensor op** -- only on writing the four-way backward.
//!
//! That makes LoHa the cheapest of the remaining adapters, which is why
//! `M2_RESEARCH.md` §10 puts it ahead of VeRA and QLoRA.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for LoHa.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "loha",
    status: ENSkillStatus::Stub {
        reason: "needs the four-operand Hadamard backward pass; every tensor op \
                 it requires already exists",
        milestone: "M3",
    },
    trainable_params: "2 * r * (d_in + d_out)",
    mergeable: true,
    requires_base_values: false,
    // The headline finding. See the module docs.
    materializes_delta: true,
    shares_params_across_layers: false,
    source: "Yeh et al. 2023, arXiv:2309.14859",
};

/// LoHa adapter: Hadamard product of two low-rank products.
///
/// Four trainable matrices, not two. Named `a1`/`b1`/`a2`/`b2` after the paper.
pub struct LRLoHa<B: Backend> {
    a1: TPParameter<B>,
    b1: TPParameter<B>,
    a2: TPParameter<B>,
    b2: TPParameter<B>,
    d_in: usize,
    d_out: usize,
    r: usize,
}

impl<B: Backend> LRLoHa<B> {
    /// Allocate LoHa's four matrices.
    ///
    /// Only `b1` and `b2` are zeroed. Zeroing all four would make every gradient
    /// zero as well, since each branch's gradient is multiplied by the other
    /// branch's product: the adapter would never leave the origin. One zeroed
    /// factor per branch is what keeps `ΔW = 0` at init while leaving the
    /// gradients alive.
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        if spec.r == 0 || spec.r > spec.d_in.min(spec.d_out) {
            return Err(GlTrainError::InvalidOp(format!(
                "LoHa rank {} must be in 1..=min(d_in, d_out) = {}",
                spec.r,
                spec.d_in.min(spec.d_out)
            )));
        }
        let std = 1.0 / spec.r as f32;
        let (d_in, d_out, r) = (spec.d_in, spec.d_out, spec.r);
        Ok(Self {
            a1: TPParameter::trainable("hada_a1", Tensor::randn(&[d_in, r], std, spec.seed)?),
            b1: TPParameter::trainable("hada_b1", Tensor::zeros(&[r, d_out])?),
            a2: TPParameter::trainable(
                "hada_a2",
                // A different seed offset, or both branches would be identical
                // and the Hadamard product would be an elementwise square.
                Tensor::randn(&[d_in, r], std, spec.seed.wrapping_add(0x9E37_79B9))?,
            ),
            b2: TPParameter::trainable("hada_b2", Tensor::zeros(&[r, d_out])?),
            d_in,
            d_out,
            r,
        })
    }

    /// Rank.
    pub fn rank(&self) -> usize {
        self.r
    }

    /// The maximum rank this parameterization can reach: `r²`.
    ///
    /// Against LoRA's `2r` at the same rank setting. This is the reason to pick
    /// LoHa over LoRA, so it is exposed rather than left in a comment.
    pub fn max_effective_rank(&self) -> usize {
        self.r * self.r
    }

    /// Adapted layer dimensions.
    pub fn dims(&self) -> (usize, usize) {
        (self.d_in, self.d_out)
    }
}

impl<B: Backend> Adapter<B> for LRLoHa<B> {
    fn forward(
        &self,
        _x: &Tensor<B>,
        _base_weight: &Tensor<B>,
        _tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        Err(GlTrainError::Unsupported {
            skill: "loha",
            reason: "the four-operand Hadamard backward pass is not written",
            milestone: "M3",
        })
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        vec![&self.a1, &self.b1, &self.a2, &self.b2]
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        vec![&mut self.a1, &mut self.b1, &mut self.a2, &mut self.b2]
    }

    fn merge_into(&self, _base_weight: &mut Tensor<B>) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "loha",
            reason: "merging needs the Hadamard delta the forward pass would build",
            milestone: "M3",
        })
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRLoHa::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    const TOL_EXACT: f32 = 0.0;

    fn loha() -> LRLoHa<GlProc> {
        LRLoHa::new(&VLAdapterSpec::new(4, 3, 2, 11)).unwrap()
    }

    #[test]
    fn has_four_matrices_not_two() {
        assert_eq!(loha().parameters().len(), 4);
    }

    #[test]
    fn matrix_shapes_match_the_paper() {
        let l = loha();
        assert_eq!(l.a1.shape(), &[4, 2]);
        assert_eq!(l.b1.shape(), &[2, 3]);
        assert_eq!(l.a2.shape(), &[4, 2]);
        assert_eq!(l.b2.shape(), &[2, 3]);
    }

    #[test]
    fn trainable_count_matches_the_capability_formula() {
        // 2 * r * (d_in + d_out) = 2 * 2 * 7 = 28
        let n: usize = loha().parameters().iter().map(|p| p.n_elems()).sum();
        assert_eq!(n, 28);
    }

    #[test]
    fn one_factor_per_branch_is_zero_and_the_other_is_not() {
        // Zeroing all four would kill every gradient; zeroing none would make
        // the adapter non-identity at init. Exactly one per branch is correct.
        let l = loha();
        assert!(l.a1.to_vec().unwrap().iter().any(|v| *v != 0.0));
        assert!(l.a2.to_vec().unwrap().iter().any(|v| *v != 0.0));
        for v in l.b1.to_vec().unwrap() {
            assert!(v.abs() <= TOL_EXACT);
        }
        for v in l.b2.to_vec().unwrap() {
            assert!(v.abs() <= TOL_EXACT);
        }
    }

    #[test]
    fn the_two_branches_are_initialized_differently() {
        // Identical branches would make the Hadamard product an elementwise
        // square, halving the expressiveness the method exists for.
        let l = loha();
        assert_ne!(l.a1.to_vec().unwrap(), l.a2.to_vec().unwrap());
    }

    #[test]
    fn max_effective_rank_is_r_squared() {
        assert_eq!(loha().max_effective_rank(), 4);
    }

    #[test]
    fn capability_flags_the_delta_materialization() {
        assert!(loha().capability().materializes_delta);
    }

    #[test]
    fn forward_returns_unsupported() {
        let l = loha();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        let w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(matches!(
            l.forward(&x, &w, &tape),
            Err(GlTrainError::Unsupported { skill: "loha", .. })
        ));
    }

    #[test]
    fn merge_returns_unsupported() {
        let mut w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(matches!(
            loha().merge_into(&mut w),
            Err(GlTrainError::Unsupported { .. })
        ));
    }
}
