//! Stummañ Gwiskadur: QLoRA. **STUB — and deliberately the thinnest one.**
//!
//! From Dettmers et al. 2023 (arXiv:2305.14314):
//!
//! > `Y^BF16 = X^BF16 doubleDequant(c1^FP32, c2^k-bit, W^NF4) + X^BF16 L1^BF16 L2^BF16`
//!
//! # Read the adapter term: `+ X L1 L2` is plain LoRA
//!
//! Nothing about the adapter is new. QLoRA's contributions are all elsewhere:
//!
//! | contribution | where it belongs | why not here |
//! |---|---|---|
//! | NF4 4-bit quantization | a quantized `BaseWeightSource` | it is a storage format for `W0` |
//! | Double quantization | same | it quantizes the *quantization constants* |
//! | BF16 compute dtype | a backend precision policy | `Backend::Scalar` is f32-only today |
//! | Paged optimizers | an optimizer memory strategy | applies to plain AdamW too |
//! | `+ X L1 L2` | **`LRLora`, unchanged** | it is literally LoRA |
//!
//! So this type exists to hold the composition contract, not a parameterization.
//! It owns a real `LRLora` and adds a description of the base-weight source it
//! would need. `M2_RESEARCH.md` §7-C has the full argument.
//!
//! The practical payoff of modelling it this way: **QDoRA** (QLoRA + DoRA, which
//! is a real published combination) is expressible as
//! `NF4 base × LoRA adapter × MagnitudeDirection composition` without any new
//! type. A `QLoRA`-as-adapter-subtype design would need a `QDoRA` subtype too,
//! and then one per further combination.
//!
//! # NF4, recorded so a later wave does not re-derive it
//!
//! Quantile levels: `q_i = ½(Q_X(i/(2^k+1)) + Q_X((i+1)/(2^k+1)))`, with `Q_X`
//! the standard-normal quantile function. Blockwise absmax normalization.
//! Double quantization uses FP8 constants with a second-level block size of
//! **256**, taking the constant overhead from **0.5 to 0.127 bits/parameter**.
//! Weights are "dequantized from storage to BFloat16, then perform matrix
//! multiplication in 16-bit" -- storage precision and compute precision are
//! separate, which is the single most important thing to get right.
//!
//! # What remains (M3+, largest surface of any adapter here)
//!
//! 1. **An NF4 codec** (quantize + dequantize, blockwise) with a scalar
//!    reference first, per Inference-First rule 1.
//! 2. **A `BaseWeightSource` abstraction** so a layer can hold a quantized
//!    weight and dequantize per use. Today `Adapter::forward` takes a dense
//!    `&Tensor<B>`, which is exactly the signature that has to widen.
//! 3. **Non-f32 storage.** `Backend::Storage` is `Vec<f32>` in both backends,
//!    and NF4 is a byte format. This is the deepest change.
//!
//! Before starting any of it, read the quant/bandwidth-wall findings in
//! `gl-agent-skills/cpu-skills/rejected-optimizations.md`: a native Q4_K path in
//! glproc was built and **lost 33%**, because nibble unpacking is compute-bound
//! on this tier. QLoRA's memory win is real, but its throughput will not follow
//! from the bit width.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::lora::LRLora;
use crate::nn::adapter::{Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for QLoRA.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "qlora",
    status: ENSkillStatus::Stub {
        reason: "needs an NF4 codec, a quantized BaseWeightSource, and non-f32 \
                 Backend::Storage; the adapter half is already LRLora",
        milestone: "M4",
    },
    // Identical to LoRA, because the trainable half *is* LoRA.
    trainable_params: "r * (d_in + d_out)",
    // The adapter merges; the 4-bit base cannot be losslessly rewritten.
    mergeable: false,
    requires_base_values: false,
    materializes_delta: false,
    shares_params_across_layers: false,
    source: "Dettmers et al. 2023, arXiv:2305.14314",
};

/// How the frozen base weight is stored.
///
/// `EN` because a closed set of variants is the whole job. Only `Dense` is
/// implemented; the other two are what QLoRA needs, named so the composition is
/// visible in the type system rather than in prose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENBaseWeightFormat {
    /// Dense f32. What M2 uses everywhere.
    Dense,
    /// 4-bit NormalFloat, blockwise absmax.
    Nf4 {
        /// Elements per quantization block. QLoRA uses 64.
        block_size: usize,
    },
    /// NF4 with the quantization constants themselves quantized.
    Nf4DoubleQuant {
        /// First-level block size, 64 in the paper.
        block_size: usize,
        /// Second-level block size, 256 in the paper.
        constant_block_size: usize,
    },
}

impl ENBaseWeightFormat {
    /// QLoRA's configuration as published: NF4, 64/256 double quantization.
    pub fn qlora_default() -> Self {
        ENBaseWeightFormat::Nf4DoubleQuant {
            block_size: 64,
            constant_block_size: 256,
        }
    }

    /// Whether this crate can currently read this format.
    pub fn is_implemented(&self) -> bool {
        matches!(self, ENBaseWeightFormat::Dense)
    }
}

/// QLoRA: a LoRA adapter over a quantized frozen base.
///
/// Owns a real [`LRLora`], because that is genuinely all the adapter is.
pub struct LRQLora<B: Backend> {
    inner: LRLora<B>,
    base_format: ENBaseWeightFormat,
}

impl<B: Backend> LRQLora<B> {
    /// Build with QLoRA's published base format.
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        Ok(Self {
            inner: LRLora::new(spec)?,
            base_format: ENBaseWeightFormat::qlora_default(),
        })
    }

    /// The base-weight format this adapter expects.
    pub fn base_format(&self) -> ENBaseWeightFormat {
        self.base_format
    }

    /// The LoRA adapter underneath. It is complete and works.
    ///
    /// Exposed to make the central research finding checkable: the trainable
    /// half of QLoRA is not a variant of LoRA, it *is* LoRA.
    pub fn inner_lora(&self) -> &LRLora<B> {
        &self.inner
    }
}

impl<B: Backend> Adapter<B> for LRQLora<B> {
    fn forward(
        &self,
        _x: &Tensor<B>,
        _base_weight: &Tensor<B>,
        _tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        // Note what is *not* happening here. The inner LoRA could compute a
        // perfectly good answer against a dense base weight, and returning it
        // would look like success. It would also silently be plain LoRA with
        // none of QLoRA's memory behaviour, which is the whole reason someone
        // asked for QLoRA. So it refuses.
        Err(GlTrainError::Unsupported {
            skill: "qlora",
            reason: "the NF4 quantized base-weight path does not exist; running \
                     the inner LoRA against a dense base would be plain LoRA \
                     with none of QLoRA's memory behaviour",
            milestone: "M4",
        })
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        Adapter::parameters(&self.inner)
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        Adapter::parameters_mut(&mut self.inner)
    }

    fn merge_into(&self, _base_weight: &mut Tensor<B>) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "qlora",
            reason: "merging into a 4-bit base requires requantization, which is \
                     lossy; merge the adapter into a dequantized base instead",
            milestone: "M4",
        })
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRQLora::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    fn qlora() -> LRQLora<GlProc> {
        LRQLora::new(&VLAdapterSpec::new(4, 3, 2, 3)).unwrap()
    }

    #[test]
    fn the_trainable_half_is_exactly_lora() {
        // The central research finding, asserted rather than argued: QLoRA's
        // parameters are LoRA's parameters, same count and same shapes.
        let q = qlora();
        let l = LRLora::<GlProc>::new(&VLAdapterSpec::new(4, 3, 2, 3)).unwrap();
        let q_shapes: Vec<Vec<usize>> = Adapter::parameters(&q)
            .iter()
            .map(|p| p.shape().to_vec())
            .collect();
        let l_shapes: Vec<Vec<usize>> = Adapter::parameters(&l)
            .iter()
            .map(|p| p.shape().to_vec())
            .collect();
        assert_eq!(q_shapes, l_shapes);
    }

    #[test]
    fn the_inner_lora_is_a_working_adapter() {
        // Not a stub inside a stub. The inner LoRA computes.
        let q = qlora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        let w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(q.inner_lora().forward(&x, &w, &tape).is_ok());
    }

    #[test]
    fn qlora_itself_still_refuses_even_though_the_inner_lora_works() {
        // The important one. A wrong implementation would return the inner
        // LoRA's answer here and look correct in every test but this.
        let q = qlora();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 4]).unwrap();
        let w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(matches!(
            q.forward(&x, &w, &tape),
            Err(GlTrainError::Unsupported { skill: "qlora", .. })
        ));
    }

    #[test]
    fn the_default_base_format_is_nf4_with_double_quantization() {
        assert_eq!(
            qlora().base_format(),
            ENBaseWeightFormat::Nf4DoubleQuant {
                block_size: 64,
                constant_block_size: 256,
            }
        );
    }

    #[test]
    fn only_the_dense_base_format_is_implemented() {
        assert!(ENBaseWeightFormat::Dense.is_implemented());
        assert!(!ENBaseWeightFormat::qlora_default().is_implemented());
        assert!(!ENBaseWeightFormat::Nf4 { block_size: 64 }.is_implemented());
    }

    #[test]
    fn capability_says_it_is_not_mergeable() {
        // Unlike every other adapter here: a 4-bit base cannot be rewritten
        // losslessly.
        assert!(!qlora().capability().mergeable);
    }

    #[test]
    fn merge_returns_unsupported() {
        let mut w = Tensor::<GlProc>::ones(&[4, 3]).unwrap();
        assert!(matches!(
            qlora().merge_into(&mut w),
            Err(GlTrainError::Unsupported { .. })
        ));
    }
}
