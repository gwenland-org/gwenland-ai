//! Stummañ Gwiskadur: LoCon. **STUB — and blocked below this layer.**
//!
//! From Yeh et al. 2023 (arXiv:2309.14859) and the LyCORIS algorithm notes:
//! LoCon extends LoRA to convolutional layers. A `Conv2d` with kernel
//! `(out, in, kh, kw)` factors into
//!
//! ```text
//! Conv(in,  dim, ksize, stride, padding)   ->   Conv(dim, out, 1x1)
//! ```
//!
//! with `rank(ΔW) <= dim`; the optional Tucker form inserts a `1x1` on both
//! sides of a full-kernel middle convolution:
//!
//! ```text
//! Conv(in, dim, 1x1) -> Conv(dim, dim, khxkw, stride, padding) -> Conv(dim, out, 1x1)
//! ```
//!
//! # Why this stub is honest about being differently blocked
//!
//! Every other adapter here is gated on adapter-level or op-level work. LoCon is
//! gated on the **tensor layer**, three levels down, and none of it has anything
//! to do with adapters:
//!
//! 1. **There are no 4-D tensors.** `Tensor::matmul` calls
//!    `check_matmul_shapes`, which rejects any rank other than 2 outright
//!    ("matmul requires 2D tensors in Wave 1"). A conv kernel is rank 4.
//! 2. **There is no convolution.** Not in `Tensor`, not in `Backend`, not in
//!    glproc's kernels. Forward *and* backward would both be new.
//! 3. **There is no im2col, no stride, no padding, and no dilation** anywhere in
//!    the crate.
//!
//! So filling this in is a conv-support project that happens to end with an
//! adapter, and scoping it as "add LoCon" would misjudge it by an order of
//! magnitude. That is why `M2_RESEARCH.md` §10 puts it 12th of 13, ahead only of
//! incremental checkpointing.
//!
//! # And a scope question worth asking before any of that
//!
//! GwenLand is an LLM stack. LoCon exists because "the convolutional layers play
//! a key role in Stable Diffusion" -- it is a diffusion-model technique. There
//! are no convolutions in Qwen2/Qwen3/Llama, which are the architectures this
//! repo tests against. **Nothing in the current model coverage would use this
//! adapter even if it were finished.**
//!
//! Recording that is more useful than a stub that implies otherwise: the
//! decision to make is not "when do we implement LoCon" but "does GwenLand
//! target diffusion models at all". Until that is answered, this type is a
//! placeholder for a question, not for an implementation.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::adapter::{Adapter, ENSkillStatus, VLAdapterCapability, VLAdapterSpec};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// Capability record for LoCon.
pub static CAPABILITY: &VLAdapterCapability = &VLAdapterCapability {
    id: "locon",
    status: ENSkillStatus::Stub {
        reason: "blocked at the tensor layer: no 4-D tensors, no convolution, no \
                 im2col. Also a diffusion-model technique with no consumer in \
                 this repo's LLM architectures",
        milestone: "unscheduled",
    },
    trainable_params: "dim * (in * kh * kw + out)",
    mergeable: true,
    requires_base_values: false,
    materializes_delta: false,
    shares_params_across_layers: false,
    source: "Yeh et al. 2023, arXiv:2309.14859",
};

/// A 2-D convolution's shape, recorded so the researched layout is not lost.
///
/// `VL` because it is a plain descriptor with derived traits only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VLConv2dShape {
    /// Output channels.
    pub out_channels: usize,
    /// Input channels.
    pub in_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Stride, both axes.
    pub stride: usize,
    /// Zero padding, both axes.
    pub padding: usize,
}

impl VLConv2dShape {
    /// A `3x3` convolution with stride 1 and padding 1, the common case.
    pub fn k3(out_channels: usize, in_channels: usize) -> Self {
        Self {
            out_channels,
            in_channels,
            kernel_h: 3,
            kernel_w: 3,
            stride: 1,
            padding: 1,
        }
    }

    /// Element count of the dense kernel: `out * in * kh * kw`.
    pub fn n_elems(&self) -> usize {
        self.out_channels * self.in_channels * self.kernel_h * self.kernel_w
    }

    /// Kernel rank. Always 4, which is the blocking fact.
    pub fn rank(&self) -> usize {
        4
    }
}

/// LoCon adapter: LoRA for convolutional layers.
///
/// Unlike the other stubs, this one allocates **no parameters**. The down-conv
/// kernel is `[dim, in, kh, kw]`, a rank-4 shape that `Tensor` cannot represent
/// meaningfully today. Allocating a flattened stand-in would encode a layout
/// nobody has designed yet, and a later wave would inherit it as though it were
/// a decision. The researched shape lives in [`VLConv2dShape`] instead.
pub struct LRLoCon {
    conv: VLConv2dShape,
    dim: usize,
}

impl LRLoCon {
    /// Record the intended adaptation without allocating anything.
    ///
    /// Takes the generic [`VLAdapterSpec`] so the registry can build it like any
    /// other adapter. `d_in`/`d_out` are read as channel counts and a `3x3`
    /// kernel is assumed, since the spec has no field for kernel geometry.
    /// A real implementation needs a conv-aware spec; that is part of the
    /// unscheduled work.
    pub fn new(spec: &VLAdapterSpec) -> Result<Self> {
        if spec.r == 0 {
            return Err(GlTrainError::InvalidOp(
                "LoCon dim must be at least 1".into(),
            ));
        }
        Ok(Self {
            conv: VLConv2dShape::k3(spec.d_out, spec.d_in),
            dim: spec.r,
        })
    }

    /// The convolution being adapted.
    pub fn conv_shape(&self) -> VLConv2dShape {
        self.conv
    }

    /// The bottleneck width, LoCon's analogue of LoRA's rank.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Trainable scalars the finished implementation would hold.
    ///
    /// `dim * (in*kh*kw)` for the down-conv plus `dim * out` for the `1x1`
    /// up-conv. Computed so the capability formula is checkable.
    pub fn projected_param_count(&self) -> usize {
        let down = self.dim * self.conv.in_channels * self.conv.kernel_h * self.conv.kernel_w;
        let up = self.dim * self.conv.out_channels;
        down + up
    }
}

impl<B: Backend> Adapter<B> for LRLoCon {
    fn forward(
        &self,
        _x: &Tensor<B>,
        _base_weight: &Tensor<B>,
        _tape: &Arc<Mutex<Tape>>,
    ) -> Result<Tensor<B>> {
        Err(GlTrainError::Unsupported {
            skill: "locon",
            reason: "no convolution and no 4-D tensors exist; matmul rejects any \
                     rank but 2",
            milestone: "unscheduled",
        })
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        // Deliberately empty. See the type's doc comment.
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        Vec::new()
    }

    fn merge_into(&self, _base_weight: &mut Tensor<B>) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "locon",
            reason: "merging a conv adapter needs the convolution that does not exist",
            milestone: "unscheduled",
        })
    }

    fn capability(&self) -> &'static VLAdapterCapability {
        CAPABILITY
    }
}

/// Registry constructor.
pub fn build<B: Backend>(spec: &VLAdapterSpec) -> Result<Box<dyn Adapter<B>>> {
    Ok(Box::new(LRLoCon::new(spec)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    fn locon() -> LRLoCon {
        LRLoCon::new(&VLAdapterSpec::new(8, 16, 4, 1)).unwrap()
    }

    #[test]
    fn constructs_and_records_the_conv_shape() {
        let l = locon();
        let c = l.conv_shape();
        assert_eq!(c.in_channels, 8);
        assert_eq!(c.out_channels, 16);
        assert_eq!((c.kernel_h, c.kernel_w), (3, 3));
        assert_eq!(l.dim(), 4);
    }

    #[test]
    fn the_conv_kernel_is_rank_4_which_is_the_blocking_fact() {
        // Tensor::matmul rejects any rank but 2, so this number is the reason
        // LoCon cannot be implemented at the adapter layer.
        assert_eq!(locon().conv_shape().rank(), 4);
    }

    #[test]
    fn projected_param_count_matches_the_capability_formula() {
        // dim*(in*kh*kw) + dim*out = 4*(8*9) + 4*16 = 288 + 64 = 352
        assert_eq!(locon().projected_param_count(), 352);
    }

    #[test]
    fn it_allocates_no_parameters_rather_than_guessing_a_layout() {
        let l = locon();
        assert!(Adapter::<GlProc>::parameters(&l).is_empty());
    }

    #[test]
    fn forward_returns_unsupported_with_no_milestone_promised() {
        let l = locon();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::ones(&[1, 8]).unwrap();
        let w = Tensor::<GlProc>::ones(&[8, 16]).unwrap();
        match Adapter::<GlProc>::forward(&l, &x, &w, &tape) {
            Err(GlTrainError::Unsupported { skill, milestone, .. }) => {
                assert_eq!(skill, "locon");
                // Honest: nothing in this repo's model coverage would use it.
                assert_eq!(milestone, "unscheduled");
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }

    #[test]
    fn a_zero_dim_is_rejected() {
        assert!(LRLoCon::new(&VLAdapterSpec::new(8, 16, 0, 1)).is_err());
    }
}
