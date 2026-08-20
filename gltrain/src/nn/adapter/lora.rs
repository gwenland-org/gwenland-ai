//! Stummañ Gwiskadur: canonical LoRA. **FULL implementation.**
//!
//! From Hu et al. 2021 (arXiv:2106.09685) §4.1:
//!
//! > `h = W0 x + ΔW x = W0 x + B A x`, with `B ∈ R^{d×r}`, `A ∈ R^{r×k}`,
//! > `r << min(d,k)`.
//!
//! > "We use a random Gaussian initialization for A and zero for B, so ΔW = BA
//! > is zero at the beginning of training."
//!
//! > "We then scale ΔWx by α/r, where α is a constant in r."
//!
//! # Shape convention: this file is transposed relative to the paper
//!
//! The paper uses column vectors (`h = Wx`, `W: [d_out, d_in]`). Stummañ uses
//! row vectors (`y = x @ W`, `W: [d_in, d_out]`), for the reasons in
//! [`crate::nn::linear`]. So:
//!
//! | | paper | here |
//! |---|---|---|
//! | base | `W0: [d_out, d_in]` | `[d_in, d_out]` |
//! | A | `[r, d_in]` | `[d_in, r]` |
//! | B | `[d_out, r]` | `[r, d_out]` |
//! | delta | `B A` | `A @ B` |
//! | forward | `W0 x + (α/r) B A x` | `x @ W0 + (α/r) (x @ A @ B)` |
//!
//! The properties that matter are preserved: A is the random one, B is the zero
//! one, `A @ B` is `[d_in, d_out]` and is zero at step 0.
//!
//! # A note on which init the "canonical" one is
//!
//! The paper says Gaussian for A. PEFT's default is
//! `kaiming_uniform_(a=sqrt(5))`, with Gaussian available as
//! `init_lora_weights="gaussian"` and `std = 1/r`. Two references, two defaults.
//! This implementation follows **the paper**, with `std = 1/r` taken from PEFT's
//! Gaussian branch since the paper does not state a variance. Recorded rather
//! than silently chosen, because a checkpoint trained under one init is not
//! numerically comparable to one trained under the other.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{
    Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec,
};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for LoRA.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "lora",
    status: ENSkillStatus::Full,
    trainable_params: "r * (d_in + d_out)",
    mergeable: true,
    // Additive: the base weight is consumed by the base matmul only.
    requires_base_values: false,
    // The whole point of LoRA: `x @ A @ B` never forms the [d_in, d_out] delta.
    materializes_delta: false,
    shares_params_across_layers: false,
    source: "Hu et al. 2021, arXiv:2106.09685 §4.1",
};

/// Configuration for a LoRA adapter, as it appears in a checkpoint.
///
/// `VL` because it is a plain config bag with derived traits only.
#[derive(Debug, Clone, PartialEq)]
pub struct VLLoraConfig {
    /// Rank.
    pub r: usize,
    /// Alpha.
    pub alpha: f32,
    /// `alpha/sqrt(r)` instead of `alpha/r`.
    pub rslora: bool,
    /// Input dimension of the adapted layer.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
}

impl VLLoraConfig {
    /// The scaling factor applied to the delta.
    pub fn scale(&self) -> f32 {
        if self.rslora {
            self.alpha / (self.r as f32).sqrt()
        } else {
            self.alpha / self.r as f32
        }
    }
}

/// Canonical LoRA adapter.
pub struct LRLora<B: Backend> {
    a: TPParameter<B>,
    b: TPParameter<B>,
    config: VLLoraConfig,
}

impl<B: Backend> LRLora<B> {
    /// Build a LoRA adapter for a `[d_in, d_out]` layer.
    ///
    /// `A` is `N(0, (1/r)^2)`, `B` is zero, so the delta starts at exactly zero
    /// and the adapted layer initially reproduces the base layer bit for bit.
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        if spec.r == 0 {
            return Err(GlTrainError::InvalidOp(
                "LoRA rank must be at least 1; r = 0 has no parameters".into(),
            ));
        }
        if spec.r > spec.d_in.min(spec.d_out) {
            return Err(GlTrainError::InvalidOp(format!(
                "LoRA rank {} exceeds min(d_in, d_out) = {}; the decomposition \
                 would have more parameters than the weight it adapts",
                spec.r,
                spec.d_in.min(spec.d_out)
            )));
        }
        if spec.alpha == 0.0 {
            return Err(GlTrainError::InvalidOp(
                "LoRA alpha must be non-zero; alpha = 0 freezes the adapter at zero".into(),
            ));
        }

        let std = 1.0 / spec.r as f32;
        let a = Tensor::randn(&[spec.d_in, spec.r], std, spec.seed)?;
        let b = Tensor::zeros(&[spec.r, spec.d_out])?;

        Ok(Self {
            a: TPParameter::trainable("lora_a", a),
            b: TPParameter::trainable("lora_b", b),
            config: VLLoraConfig {
                r: spec.r,
                alpha: spec.alpha,
                rslora: spec.rslora,
                d_in: spec.d_in,
                d_out: spec.d_out,
            },
        })
    }

    /// Rebuild from stored tensors. Used when loading a checkpoint.
    ///
    /// Shapes are checked against `config` rather than trusted, because a
    /// transposed `A` has the same element count as a correct one whenever
    /// `d_in == r`, and would load silently.
    pub fn from_tensors(config: VLLoraConfig, a: Tensor<B>, b: Tensor<B>) -> Result<Self> {
        let want_a = vec![config.d_in, config.r];
        let want_b = vec![config.r, config.d_out];
        if a.shape() != want_a.as_slice() {
            return Err(GlTrainError::ShapeMismatch {
                expected: want_a,
                got: a.shape().to_vec(),
            });
        }
        if b.shape() != want_b.as_slice() {
            return Err(GlTrainError::ShapeMismatch {
                expected: want_b,
                got: b.shape().to_vec(),
            });
        }
        Ok(Self {
            a: TPParameter::trainable("lora_a", a),
            b: TPParameter::trainable("lora_b", b),
            config,
        })
    }

    /// The adapter's configuration.
    pub fn config(&self) -> &VLLoraConfig {
        &self.config
    }

    /// The down-projection `A`, shape `[d_in, r]`.
    pub fn a(&self) -> &TPParameter<B> {
        &self.a
    }

    /// The up-projection `B`, shape `[r, d_out]`.
    pub fn b(&self) -> &TPParameter<B> {
        &self.b
    }

    /// The scaling factor `alpha/r` (or `alpha/sqrt(r)`).
    pub fn scale(&self) -> f32 {
        self.config.scale()
    }

    /// The full delta `scale * (A @ B)`, shape `[d_in, d_out]`.
    ///
    /// Only used by [`Adapter::merge_into`]. The forward pass deliberately does
    /// **not** call this: routing `x` through `A` then `B` costs
    /// `batch*d_in*r + batch*r*d_out`, while forming the delta costs
    /// `d_in*r*d_out` and then another full matmul. For a rank-8 adapter on a
    /// 2048-wide layer that is the difference between ~33k and ~33M
    /// multiply-accumulates per token.
    pub fn delta(&self) -> Result<Tensor<B>> {
        self.a.tensor().matmul(self.b.tensor())?.mul_scalar(self.scale())
    }
}

impl<B: Backend> Adapter<B> for LRLora<B> {
    fn forward(
        &self,
        x: &Tensor<B>,
        base_weight: &Tensor<B>,
        tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        if base_weight.shape() != [self.config.d_in, self.config.d_out] {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![self.config.d_in, self.config.d_out],
                got: base_weight.shape().to_vec(),
            });
        }
        // The base weight arrives untracked, so this matmul contributes a
        // gradient to `x` but never to the base. That is KL-003's documented
        // "frozen operand" path, and it is the primary LoRA shape.
        let base_out = x.matmul(base_weight)?;

        // x @ A @ B, never forming A @ B.
        let down = x.matmul(&self.a.tracked(tape))?;
        let up = down.matmul(&self.b.tracked(tape))?;
        let scaled = up.mul_scalar(self.scale())?;

        base_out.add(&scaled)
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        vec![&self.a, &self.b]
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        vec![&mut self.a, &mut self.b]
    }

    fn merge_into(&self, base_weight: &mut Tensor<B>) -> Result<()> {
        if base_weight.shape() != [self.config.d_in, self.config.d_out] {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![self.config.d_in, self.config.d_out],
                got: base_weight.shape().to_vec(),
            });
        }
        let merged = base_weight.add(&self.delta()?)?;
        base_weight.replace_data(merged.to_vec()?)
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRLora::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// Matmul accumulates over the rank/K dimension; matches tensor.rs.
    const TOL_MATMUL: f32 = 1e-4;
    /// Exact zero is exact: B is zeros, so the delta is 0.0 bit for bit.
    const TOL_EXACT: f32 = 0.0;

    fn spec() -> VLAdapterSpec {
        VLAdapterSpec::new(4, 3, 2, 42)
    }

    fn lora() -> LRLora<GlProc> {
        LRLora::new(&spec()).unwrap()
    }

    #[test]
    fn a_is_random_and_b_is_zero_at_init() {
        let l = lora();
        assert!(
            l.a().to_vec().unwrap().iter().any(|v| *v != 0.0),
            "A must be randomly initialized"
        );
        for v in l.b().to_vec().unwrap() {
            assert!(v.abs() <= TOL_EXACT, "B must start at exactly zero");
        }
    }

    #[test]
    fn parameter_shapes_follow_the_row_vector_convention() {
        let l = lora();
        assert_eq!(l.a().shape(), &[4, 2], "A is [d_in, r]");
        assert_eq!(l.b().shape(), &[2, 3], "B is [r, d_out]");
    }

    #[test]
    fn trainable_parameter_count_matches_the_capability_formula() {
        // r * (d_in + d_out) = 2 * (4 + 3) = 14
        let l = lora();
        assert_eq!(l.a().n_elems() + l.b().n_elems(), 14);
    }

    #[test]
    fn delta_is_exactly_zero_at_init() {
        // The property the zero-init of B exists to guarantee: an untrained
        // adapter is a no-op, so attaching one cannot change a model's output.
        for v in lora().delta().unwrap().to_vec().unwrap() {
            assert!(v.abs() <= TOL_EXACT);
        }
    }

    #[test]
    fn forward_at_init_reproduces_the_base_layer_exactly() {
        let l = lora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let base = Tensor::<GlProc>::from_vec((1..=12).map(|v| v as f32).collect(), &[4, 3]).unwrap();
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();

        let want = x.matmul(&base).unwrap().to_vec().unwrap();
        let got = l.forward(&x, &base, &tape).unwrap().to_vec().unwrap();
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() < TOL_MATMUL, "got {got:?} want {want:?}");
        }
    }

    /// The numerical anchor: hand-computed, no reference implementation involved.
    ///
    /// d_in=2, d_out=2, r=1, alpha=2 so scale = 2/1 = 2.
    /// A = [[1],[2]]  B = [[3, 4]]   x = [[1, 1]]   W0 = [[1,0],[0,1]]
    ///   x @ W0  = [1, 1]
    ///   x @ A   = [1*1 + 1*2] = [3]
    ///   (x@A)@B = [3*3, 3*4] = [9, 12]
    ///   scaled  = [18, 24]
    ///   total   = [19, 25]
    #[test]
    fn forward_matches_a_hand_computed_example() {
        let config = VLLoraConfig {
            r: 1,
            alpha: 2.0,
            rslora: false,
            d_in: 2,
            d_out: 2,
        };
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
        let l = LRLora::from_tensors(config, a, b).unwrap();

        let tape = Arc::new(Mutex::new(Tape::new()));
        let w0 = Tensor::<GlProc>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 1.0], &[1, 2]).unwrap();

        let got = l.forward(&x, &w0, &tape).unwrap().to_vec().unwrap();
        for (g, w) in got.iter().zip([19.0, 25.0]) {
            assert!((g - w).abs() < TOL_MATMUL, "got {got:?}, want [19, 25]");
        }
    }

    /// Merging must give the same answer as the two-matmul forward path.
    /// This is the property that makes deployment sound: the same weights, one
    /// matmul instead of three.
    #[test]
    fn merged_weight_reproduces_the_forward_pass() {
        let config = VLLoraConfig {
            r: 1,
            alpha: 2.0,
            rslora: false,
            d_in: 2,
            d_out: 2,
        };
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0], &[2, 1]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![3.0, 4.0], &[1, 2]).unwrap();
        let l = LRLora::from_tensors(config, a, b).unwrap();

        let tape = Arc::new(Mutex::new(Tape::new()));
        let mut w = Tensor::<GlProc>::from_vec(vec![1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 1.0], &[1, 2]).unwrap();

        let unmerged = l.forward(&x, &w, &tape).unwrap().to_vec().unwrap();
        l.merge_into(&mut w).unwrap();
        let merged = x.matmul(&w).unwrap().to_vec().unwrap();

        for (u, m) in unmerged.iter().zip(&merged) {
            assert!((u - m).abs() < TOL_MATMUL, "unmerged {unmerged:?} merged {merged:?}");
        }
    }

    #[test]
    fn forward_records_gradients_for_both_a_and_b() {
        let l = lora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let base = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap().with_grad(tape.clone());

        let y = l.forward(&x, &base, &tape).unwrap();
        y.sum().unwrap();
        Tape::lock(&tape).backward().unwrap();

        let guard = Tape::lock(&tape);
        assert!(guard.grad(l.a().id()).is_some(), "A must receive a gradient");
        assert!(guard.grad(l.b().id()).is_some(), "B must receive a gradient");
    }

    #[test]
    fn the_frozen_base_weight_never_receives_a_gradient() {
        // The defining property of LoRA. If this fails, the base model is being
        // trained and the memory argument for LoRA is gone.
        let l = lora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let base = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap().with_grad(tape.clone());

        let y = l.forward(&x, &base, &tape).unwrap();
        y.sum().unwrap();
        Tape::lock(&tape).backward().unwrap();

        assert!(
            Tape::lock(&tape).grad(base.id()).is_none(),
            "the frozen base weight must not accumulate a gradient"
        );
    }

    #[test]
    fn rank_zero_is_rejected() {
        let s = VLAdapterSpec::new(4, 4, 0, 1);
        assert!(LRLora::<GlProc>::new(&s).is_err());
    }

    #[test]
    fn a_rank_larger_than_the_weight_is_rejected() {
        let s = VLAdapterSpec::new(4, 3, 8, 1);
        assert!(LRLora::<GlProc>::new(&s).is_err());
    }

    #[test]
    fn zero_alpha_is_rejected() {
        let s = VLAdapterSpec {
            alpha: 0.0,
            ..VLAdapterSpec::new(4, 4, 2, 1)
        };
        assert!(LRLora::<GlProc>::new(&s).is_err());
    }

    #[test]
    fn from_tensors_rejects_a_transposed_a() {
        // [r, d_in] instead of [d_in, r]. Same element count, wrong answer.
        let config = VLLoraConfig {
            r: 2,
            alpha: 2.0,
            rslora: false,
            d_in: 4,
            d_out: 3,
        };
        let a_wrong = Tensor::<GlProc>::zeros(&[2, 4]).unwrap();
        let b = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        assert!(LRLora::from_tensors(config, a_wrong, b).is_err());
    }

    #[test]
    fn forward_rejects_a_base_weight_of_the_wrong_shape() {
        let l = lora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let base = Tensor::<GlProc>::zeros(&[3, 4]).unwrap();
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        assert!(l.forward(&x, &base, &tape).is_err());
    }

    #[test]
    fn init_is_reproducible_from_the_seed() {
        let a1 = LRLora::<GlProc>::new(&spec()).unwrap().a().to_vec().unwrap();
        let a2 = LRLora::<GlProc>::new(&spec()).unwrap().a().to_vec().unwrap();
        assert_eq!(a1, a2);
    }

    #[test]
    fn a_different_seed_gives_a_different_a() {
        let s1 = VLAdapterSpec::new(4, 3, 2, 1);
        let s2 = VLAdapterSpec::new(4, 3, 2, 2);
        let a1 = LRLora::<GlProc>::new(&s1).unwrap().a().to_vec().unwrap();
        let a2 = LRLora::<GlProc>::new(&s2).unwrap().a().to_vec().unwrap();
        assert_ne!(a1, a2);
    }
}
