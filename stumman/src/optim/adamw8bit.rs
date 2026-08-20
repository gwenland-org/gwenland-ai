//! Stummañ Gwellaer: 8-bit AdamW. **STUB, wrapping a real [`OPAdamW`].**
//!
//! Dettmers et al. 2022 (arXiv:2110.02861).
//!
//! # This is a codec, not a fourth update rule
//!
//! The update is AdamW's, unchanged. Only the *storage* of `m` and `v` differs.
//! That is why `M2_RESEARCH.md` §6 calls it an `OptimizerStateCodec` rather
//! than listing it beside Lion and Adafactor: those two change what is
//! computed, this one changes what is kept.
//!
//! So this type **wraps** a real [`OPAdamW`] rather than reimplementing the
//! math a second time. The quantize/dequantize belongs in `state_tensors` and
//! `load_state`, which is the only place the 8-bit representation is visible.
//! The registry still gives it its own id: composition is an implementation
//! detail, not something a caller needs to know.
//!
//! # Four things the codec has to get right, none of them optional
//!
//! - **Both `m` and `v` are quantized.** Unlike Lion, which drops `v` entirely.
//! - **Block-wise, 2048 elements per block, independent absmax per block.**
//!   Not one global scale: per-block scaling is what confines an outlier to its
//!   own block instead of crushing the resolution of every other value.
//! - **Dynamic tree quantization, not linear.** Optimizer states span roughly
//!   seven orders of magnitude. A `Vec<u8>` with a naive uniform scale is a
//!   different, worse codec, not a simplified version of this one.
//! - **The update runs in 32-bit.** Dequantize, run AdamW's ordinary update,
//!   requantize. Never arithmetic on the quantized bytes.
//!
//! # The stable-embedding exception
//!
//! Embedding-layer optimizer state stays 32-bit even when everything else is
//! 8-bit. A real implementation needs a way to exempt parameters by name
//! pattern; [`OPAdamW8bit::exempt`] is that hook, present now so the
//! requirement is not discovered only after training an embedding layer badly.

use crate::autograd::grad_store::VLGradStore;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use crate::nn::param::TPParameter;
use crate::optim::adamw::{OPAdamW, VLAdamWConfig};
use crate::optim::{
    ENOptimizerStateShape, Optimizer, VLNamedTensor, VLOptimizerCapability, VLOptimizerSpec,
    VLParamGroup,
};
use crate::tensor::backend::Backend;
use std::collections::BTreeSet;

/// Why this is a stub, in the error a caller sees.
const STUB_REASON: &str =
    "needs a dynamic-tree 8-bit codec with 2048-element blockwise absmax scaling; a linear u8 \
     scale is a different and worse codec, so shipping one would not be a simplified version of \
     this optimizer";
const STUB_MILESTONE: &str = "M3";

/// Elements per independently-scaled quantization block.
pub const BLOCK_SIZE: usize = 2048;

/// 8-bit AdamW's capability record.
pub static CAPABILITY: &VLOptimizerCapability = &VLOptimizerCapability {
    id: "adamw8bit",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    state_shape: ENOptimizerStateShape::Quantized {
        bits: 8,
        block: BLOCK_SIZE,
    },
    // Two buffers at one byte per element instead of four: 2 * 0.25 = 0.5x the
    // parameter's f32 size, against AdamW's 2.0x. Plus one f32 absmax per
    // 2048-element block, which rounds to nothing.
    memory_multiplier: 0.5,
    source: "Dettmers et al. 2022, arXiv:2110.02861",
};

/// AdamW with 8-bit optimizer state. Constructs and introspects; refuses to
/// compute.
pub struct OPAdamW8bit<B: Backend> {
    /// The real update rule. Composition, not reimplementation.
    inner: OPAdamW<B>,
    /// Parameter names whose state stays 32-bit. The stable-embedding rule.
    exempt: BTreeSet<String>,
}

impl<B: Backend> OPAdamW8bit<B> {
    /// An optimizer wrapping an [`OPAdamW`] with the given hyperparameters.
    pub fn new(config: VLAdamWConfig) -> Self {
        Self {
            inner: OPAdamW::new(config),
            exempt: BTreeSet::new(),
        }
    }

    /// The wrapped AdamW. Its update rule is this optimizer's update rule.
    pub fn inner(&self) -> &OPAdamW<B> {
        &self.inner
    }

    /// Keep this parameter's state in 32-bit.
    ///
    /// The stable-embedding exception. Quantizing an embedding layer's
    /// optimizer state degrades training in a way that shows up as a slightly
    /// worse model rather than as an error, so the exemption has to be
    /// expressible before anyone needs it.
    pub fn exempt(&mut self, param_name: impl Into<String>) {
        self.exempt.insert(param_name.into());
    }

    /// Whether this parameter's state would stay 32-bit.
    pub fn is_exempt(&self, param_name: &str) -> bool {
        self.exempt.contains(param_name)
    }

    /// How many blocks an `n`-element buffer is split into, each with its own
    /// absmax scale.
    pub fn block_count(n_elems: usize) -> usize {
        n_elems.div_ceil(BLOCK_SIZE)
    }
}

impl<B: Backend> Optimizer<B> for OPAdamW8bit<B> {
    fn step(&mut self, _params: &mut [&mut TPParameter<B>], _grads: &VLGradStore) -> Result<()> {
        // Deliberately NOT `self.inner.step(...)`. That would compute a correct
        // AdamW update with 32-bit state and report success, which is "asked
        // for 8-bit, got 32-bit" with no error and no memory saving: the one
        // property the optimizer exists for, silently absent.
        Err(GlTrainError::Unsupported {
            skill: "adamw8bit",
            reason: STUB_REASON,
            milestone: STUB_MILESTONE,
        })
    }

    fn groups(&self) -> &[VLParamGroup] {
        self.inner.groups()
    }

    fn add_group(&mut self, group: VLParamGroup) -> Result<()> {
        self.inner.add_group(group)
    }

    fn assign_group(&mut self, param_name: &str, group_name: &str) -> Result<()> {
        self.inner.assign_group(param_name, group_name)
    }

    fn state_tensors(&self, _params: &[&TPParameter<B>]) -> Result<Vec<VLNamedTensor>> {
        // This is where the codec actually lives, so it is the one method that
        // cannot delegate to `inner`: doing so would write 32-bit state under
        // an "adamw8bit" manifest, which is a file that lies about its format.
        Err(GlTrainError::Unsupported {
            skill: "adamw8bit",
            reason: STUB_REASON,
            milestone: STUB_MILESTONE,
        })
    }

    fn load_state(&mut self, _params: &[&TPParameter<B>], _named: &[VLNamedTensor]) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "adamw8bit",
            reason: STUB_REASON,
            milestone: STUB_MILESTONE,
        })
    }

    fn capability(&self) -> &'static VLOptimizerCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLOptimizerSpec) -> Result<Box<dyn Optimizer<B>>> {
    let d = VLAdamWConfig::default();
    Ok(Box::new(OPAdamW8bit::<B>::new(VLAdamWConfig {
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
    use crate::backend::GlProc;
    use crate::tensor::Tensor;

    fn param(name: &str, n: usize) -> TPParameter<GlProc> {
        TPParameter::trainable(name, Tensor::<GlProc>::zeros(&[n]).unwrap())
    }

    #[test]
    fn adamw8bit_step_returns_unsupported() {
        let mut opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig::default());
        let mut p = param("w", 4);
        let err = opt.step(&mut [&mut p], &VLGradStore::new());
        assert!(matches!(
            err,
            Err(GlTrainError::Unsupported {
                skill: "adamw8bit",
                ..
            })
        ));
    }

    /// The trap this stub exists to avoid: delegating to the wrapped AdamW
    /// would produce a correct-looking update with 32-bit state, reporting
    /// success while delivering none of the memory saving that is the entire
    /// reason to pick this optimizer.
    #[test]
    fn adamw8bit_does_not_silently_delegate_to_the_wrapped_adamw() {
        let mut opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig::default());
        let mut p = param("w", 1);
        let mut g = VLGradStore::new();
        g.accumulate(p.id(), vec![0.5], vec![1]).unwrap();

        let before = p.to_vec().unwrap()[0];
        assert!(opt.step(&mut [&mut p], &g).is_err());
        assert_eq!(
            p.to_vec().unwrap()[0],
            before,
            "a refused step must not move the weight"
        );
        assert!(
            opt.inner().moments(p.id()).is_none(),
            "the wrapped AdamW must not have run"
        );
    }

    /// Serializing 32-bit state under an "adamw8bit" id would produce a file
    /// that lies about its own format, so this refuses too.
    #[test]
    fn adamw8bit_refuses_to_serialize_state_it_cannot_yet_encode() {
        let opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig::default());
        let p = param("w", 4);
        assert!(opt.state_tensors(&[&p]).is_err());
    }

    /// The update rule is AdamW's, so the wrapped optimizer is real, and its
    /// hyperparameters are the ones that will be used when the codec lands.
    #[test]
    fn adamw8bit_wraps_a_real_adamw_rather_than_reimplementing_it() {
        let opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig {
            lr: 0.5,
            ..VLAdamWConfig::default()
        });
        assert_eq!(opt.inner().config().lr, 0.5);
        assert_eq!(opt.inner().capability().id, "adamw");
        assert_eq!(opt.capability().id, "adamw8bit");
    }

    /// Group handling passes straight through, so LoRA+ works here the day the
    /// codec lands without a second implementation of it.
    #[test]
    fn adamw8bit_forwards_parameter_groups_to_the_wrapped_adamw() {
        let mut opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig::default());
        opt.add_group(VLParamGroup::new("lora_b", 4.0)).unwrap();
        opt.assign_group("lora_b", "lora_b").unwrap();
        assert_eq!(opt.inner().effective_lr("lora_b"), 4.0 * 1e-3);
    }

    /// Blocks are 2048 elements with an independent scale each. One global
    /// scale would let a single outlier crush every other value's resolution.
    #[test]
    fn adamw8bit_blocks_are_2048_elements_with_a_partial_final_block() {
        assert_eq!(BLOCK_SIZE, 2048);
        assert_eq!(OPAdamW8bit::<GlProc>::block_count(2048), 1);
        assert_eq!(OPAdamW8bit::<GlProc>::block_count(2049), 2);
        assert_eq!(OPAdamW8bit::<GlProc>::block_count(1), 1);
        assert_eq!(OPAdamW8bit::<GlProc>::block_count(0), 0);
        assert_eq!(
            CAPABILITY.state_shape,
            ENOptimizerStateShape::Quantized {
                bits: 8,
                block: BLOCK_SIZE
            }
        );
    }

    /// Embedding-layer state stays 32-bit. Without a hook for this, the
    /// requirement gets discovered only after training an embedding badly.
    #[test]
    fn adamw8bit_can_exempt_a_parameter_from_quantization() {
        let mut opt = OPAdamW8bit::<GlProc>::new(VLAdamWConfig::default());
        assert!(!opt.is_exempt("tok_embeddings.weight"));
        opt.exempt("tok_embeddings.weight");
        assert!(opt.is_exempt("tok_embeddings.weight"));
        assert!(!opt.is_exempt("lora_a"));
    }

    /// Half of AdamW's 2.0x: two buffers at one byte per element instead of
    /// four.
    #[test]
    fn adamw8bit_claims_a_quarter_of_adamws_state_memory() {
        assert_eq!(CAPABILITY.memory_multiplier, 0.5);
        assert_eq!(crate::optim::adamw::CAPABILITY.memory_multiplier, 2.0);
    }
}
