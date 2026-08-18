//! Stummañ Gwiskadur: VeRA. **STUB — parameters are real, forward is not.**
//!
//! From Kopiczko et al. 2024 (arXiv:2310.11454):
//!
//! > `h = W0 x + Λ_b B Λ_d A x`
//!
//! `A` and `B` are "frozen, random, and shared across layers"; only the scaling
//! vectors are trained. Parameter count is
//! `|Θ| = L_tuned × (d_model + r)` against LoRA's
//! `2 × L_tuned × d_model × r`.
//!
//! # The vector lengths, which are easy to get backwards
//!
//! | vector | length | scales |
//! |---|---|---|
//! | `d` | **r** | the rank axis, between `A` and `B` |
//! | `b` | **d_out** | the output axis, after `B` |
//!
//! The parameter-count formula is the internal check: `d_model + r` per layer is
//! `d_out` for `b` plus `r` for `d`. Getting these equal-length would pass every
//! shape assertion whenever `r == d_out`, which is why the check is written down.
//!
//! Init, from the paper: Kaiming for `A` and `B`; `Λ_d` to a constant `d_init`
//! (e.g. `1e-1`); `Λ_b` to **zeros**, "which aligns with the initialization of
//! matrix B in LoRA" -- so the adapter starts as a no-op, same as LoRA.
//!
//! # Two structural consequences, both real
//!
//! 1. **Parameters are not owned per layer.** `A` and `B` are one pair for the
//!    whole model. A tree walk that returns each layer's parameters returns the
//!    shared pair once per layer, and an optimizer that received it repeatedly
//!    would apply the update once per occurrence. They are frozen here so it
//!    cannot bite in M2, but
//!    [`crate::nn::module::trainable_parameters`] dedupes by name for this
//!    reason and the capability record sets
//!    `shares_params_across_layers: true`.
//! 2. **The checkpoint stores a seed, not the matrices.** The paper: the frozen
//!    matrices "do not need to be stored in memory" since they "can be
//!    regenerated from a random number generator (RNG) seed". That is a
//!    *format* requirement, not an optimization: the checkpoint format has to be
//!    able to say "this tensor is generated, here is its seed" as well as "here
//!    are its bytes". `crate::checkpoint` reserves a field for it rather than
//!    discovering the need later.
//!
//! # What remains (M3)
//!
//! 1. **Kaiming init.** Uniform over `±sqrt(1/fan_in)`. The RNG has
//!    `next_f32`, so this is a few lines, but it is not written.
//! 2. **A shared-parameter owner.** Something above the layer has to own the one
//!    `A`/`B` pair and hand references down. That is a model-tree question, and
//!    the model tree is M3.
//! 3. **Diagonal scaling as an op.** `Λ_d A` is a row-scale, not a general
//!    matmul. `Tensor::mul` needs broadcasting to express it, which the crate
//!    does not have (`check_same_shape` requires exact equality).

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for VeRA.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "vera",
    status: ENSkillStatus::Stub {
        reason: "needs Kaiming init, broadcasting for diagonal scaling, and a \
                 cross-layer owner for the shared frozen pair",
        milestone: "M3",
    },
    // Per adapted layer. The shared A/B pair is counted once for the model, not
    // per layer, and is frozen anyway.
    trainable_params: "d_out + r",
    mergeable: true,
    requires_base_values: false,
    materializes_delta: false,
    // The property that distinguishes VeRA from every other adapter here.
    shares_params_across_layers: true,
    source: "Kopiczko et al. 2024, arXiv:2310.11454",
};

/// The default `d_init` from the paper.
pub const DEFAULT_D_INIT: f32 = 1e-1;

/// VeRA adapter: trainable scaling vectors over a frozen shared random pair.
pub struct LRVeRA<B: Backend> {
    /// Frozen, random, shared across layers in a real model. Shape `[d_in, r]`.
    a: TPParameter<B>,
    /// Frozen, random, shared across layers. Shape `[r, d_out]`.
    b: TPParameter<B>,
    /// Trainable. Length `r`, scales the rank axis. Shape `[1, r]`.
    lambda_d: TPParameter<B>,
    /// Trainable. Length `d_out`, scales the output axis. Shape `[1, d_out]`.
    lambda_b: TPParameter<B>,
    /// The seed the frozen pair was generated from, so a checkpoint can store it
    /// instead of the matrices.
    seed: u64,
    d_in: usize,
    d_out: usize,
    r: usize,
}

impl<B: Backend> LRVeRA<B> {
    /// Allocate VeRA's parameters.
    ///
    /// `A` and `B` are generated from `spec.seed` and marked **frozen**;
    /// `lambda_d` starts at `DEFAULT_D_INIT` and `lambda_b` at zero.
    ///
    /// The random pair uses this adapter's own seed rather than a model-wide one,
    /// because there is no model-wide owner yet. A real implementation shares one
    /// pair, and that is item 2 of "what remains".
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        if spec.r == 0 || spec.r > spec.d_in.min(spec.d_out) {
            return Err(GlTrainError::InvalidOp(format!(
                "VeRA rank {} must be in 1..=min(d_in, d_out) = {}",
                spec.r,
                spec.d_in.min(spec.d_out)
            )));
        }
        let (d_in, d_out, r) = (spec.d_in, spec.d_out, spec.r);
        // Placeholder for Kaiming: std = sqrt(1/fan_in) matches Kaiming's
        // variance even though the paper's uniform variant differs in shape.
        // Flagged in the stub reason rather than passed off as correct.
        let std_a = (1.0 / d_in as f32).sqrt();
        let std_b = (1.0 / r as f32).sqrt();
        Ok(Self {
            a: TPParameter::frozen("vera_a", Tensor::randn(&[d_in, r], std_a, spec.seed)?),
            b: TPParameter::frozen(
                "vera_b",
                Tensor::randn(&[r, d_out], std_b, spec.seed.wrapping_add(1))?,
            ),
            lambda_d: TPParameter::trainable(
                "vera_lambda_d",
                Tensor::from_vec(vec![DEFAULT_D_INIT; r], &[1, r])?,
            ),
            lambda_b: TPParameter::trainable("vera_lambda_b", Tensor::zeros(&[1, d_out])?),
            seed: spec.seed,
            d_in,
            d_out,
            r,
        })
    }

    /// The seed the frozen pair can be regenerated from.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// `Λ_d`, length `r`.
    pub fn lambda_d(&self) -> &TPParameter<B> {
        &self.lambda_d
    }

    /// `Λ_b`, length `d_out`.
    pub fn lambda_b(&self) -> &TPParameter<B> {
        &self.lambda_b
    }

    /// Trainable scalars for this layer: `d_out + r`.
    pub fn trainable_count(&self) -> usize {
        self.d_out + self.r
    }

    /// What LoRA would need at the same rank: `r * (d_in + d_out)`.
    ///
    /// Exposed so the parameter-efficiency claim is checkable rather than
    /// asserted in prose.
    pub fn lora_equivalent_count(&self) -> usize {
        self.r * (self.d_in + self.d_out)
    }
}

impl<B: Backend> Adapter<B> for LRVeRA<B> {
    fn forward(
        &self,
        _x: &Tensor<B>,
        _base_weight: &Tensor<B>,
        _tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        Err(GlTrainError::Unsupported {
            skill: "vera",
            reason: "diagonal scaling needs broadcasting, and the shared frozen \
                     pair needs a cross-layer owner",
            milestone: "M3",
        })
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        // The frozen pair is reported: a checkpoint validator needs to see it,
        // and `trainable_parameters` filters it out for the optimizer.
        vec![&self.a, &self.b, &self.lambda_d, &self.lambda_b]
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        vec![
            &mut self.a,
            &mut self.b,
            &mut self.lambda_d,
            &mut self.lambda_b,
        ]
    }

    fn merge_into(&self, _base_weight: &mut Tensor<B>) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "vera",
            reason: "merging needs the diagonal-scaled product the forward pass would build",
            milestone: "M3",
        })
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRVeRA::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;
    use crate::nn::module::Module;

    const TOL_EXACT: f32 = 0.0;
    /// d_init is stored verbatim, so no arithmetic tolerance is needed.
    const TOL_INIT: f32 = 1e-9;

    fn vera() -> LRVeRA<GlProc> {
        LRVeRA::new(&VLAdapterSpec::new(8, 4, 2, 5)).unwrap()
    }

    #[test]
    fn scaling_vectors_have_the_researched_lengths() {
        // d has length r, b has length d_out. The formula d_model + r is the
        // check; equal lengths would pass every assertion when r == d_out.
        let v = vera();
        assert_eq!(v.lambda_d().shape(), &[1, 2], "lambda_d has length r");
        assert_eq!(v.lambda_b().shape(), &[1, 4], "lambda_b has length d_out");
    }

    #[test]
    fn the_random_pair_is_frozen_and_the_vectors_are_trainable() {
        let v = vera();
        assert!(!v.a.is_trainable(), "A must be frozen");
        assert!(!v.b.is_trainable(), "B must be frozen");
        assert!(v.lambda_d().is_trainable());
        assert!(v.lambda_b().is_trainable());
    }

    #[test]
    fn lambda_b_starts_at_zero_so_the_adapter_is_initially_a_no_op() {
        for val in vera().lambda_b().to_vec().unwrap() {
            assert!(val.abs() <= TOL_EXACT);
        }
    }

    #[test]
    fn lambda_d_starts_at_the_papers_d_init() {
        for val in vera().lambda_d().to_vec().unwrap() {
            assert!((val - DEFAULT_D_INIT).abs() < TOL_INIT);
        }
    }

    #[test]
    fn trainable_count_is_far_below_the_lora_equivalent() {
        // The whole claim of the paper, made checkable.
        // VeRA: d_out + r = 4 + 2 = 6. LoRA: r*(d_in+d_out) = 2*12 = 24.
        let v = vera();
        assert_eq!(v.trainable_count(), 6);
        assert_eq!(v.lora_equivalent_count(), 24);
        assert!(v.trainable_count() < v.lora_equivalent_count());
    }

    #[test]
    fn the_frozen_pair_is_excluded_from_trainable_parameters() {
        struct Wrap(LRVeRA<GlProc>);
        impl Module<GlProc> for Wrap {
            fn forward(&self, x: &Tensor<GlProc>, _: &Arc<Mutex<Tape>>) -> Result<Tensor<GlProc>> {
                Ok(x.clone())
            }
            fn parameters(&self) -> Vec<&TPParameter<GlProc>> {
                Adapter::parameters(&self.0)
            }
            fn parameters_mut(&mut self) -> Vec<&mut TPParameter<GlProc>> {
                Adapter::parameters_mut(&mut self.0)
            }
        }
        let w = Wrap(vera());
        // Only the two scaling vectors: 2 + 4 = 6 scalars.
        assert_eq!(w.trainable_param_count(), 6);
    }

    #[test]
    fn the_seed_is_retained_so_a_checkpoint_can_store_it_instead_of_the_matrices() {
        assert_eq!(vera().seed(), 5);
    }

    #[test]
    fn capability_flags_cross_layer_sharing() {
        assert!(vera().capability().shares_params_across_layers);
    }

    #[test]
    fn forward_returns_unsupported() {
        let v = vera();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 8]).unwrap();
        let w = Tensor::<GlProc>::ones(&[8, 4]).unwrap();
        assert!(matches!(
            v.forward(&x, &w, &tape),
            Err(GlTrainError::Unsupported { skill: "vera", .. })
        ));
    }
}
