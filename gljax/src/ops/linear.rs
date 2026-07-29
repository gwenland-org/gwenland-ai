//! Linear projection against a HuggingFace-layout weight.
//!
//! # ⛔ `nn.Linear` stores `[out_features, in_features]`
//!
//! PyTorch's `nn.Linear` holds its weight transposed relative to the maths:
//! the forward pass is `x @ W.T`, and safetensors stores `W`. So a projection
//! from 896 to 4864 is stored as **`[4864, 896]`**, not `[896, 4864]`.
//!
//! gljax originally traced `[in, out]` and the checkpoint binder rejected 120
//! tensors on the first real load:
//!
//! ```text
//! model.layers.0.mlp.gate_proj.weight: trace wants [896, 4864],
//!   checkpoint has [4864, 896] (transposed — same element count, different layout)
//! ```
//!
//! ⭐ **The square projections hid it.** `q_proj` and `o_proj` are both
//! `[896, 896]`, so they matched on shape while being equally wrong — 48 more
//! tensors that no shape check could ever have flagged. Had the FFN happened
//! to be square too, this would have loaded cleanly and produced fluent
//! garbage. P4, and the reason the binder reports *every* disagreement rather
//! than the first.
//!
//! # Why not transpose on load
//!
//! Because `dot_general` can contract whichever axis it is told to. Choosing
//! `rhs_contracting = [1]` costs nothing and reads the weight exactly as
//! stored; materialising `W.T` for 24 layers would be pure waste.

use crate::stablehlo::ops::DotDimensionNumbers;
use crate::tensor::Tensor;

/// `y[..., out] = x[..., in] · W[out, in]`.
///
/// `w` is in HuggingFace layout — the axes are `[out_features, in_features]`,
/// which is what safetensors stores.
///
/// # Panics
/// If `w` is not rank 2, or its `in_features` axis disagrees with `x`'s last
/// dimension.
pub fn linear(x: &Tensor, w: &Tensor) -> Tensor {
    assert_eq!(
        w.rank(),
        2,
        "linear: weight must be rank-2 [out, in], got {:?}",
        w.shape().dims
    );
    let in_features = x.dim(x.rank() - 1);
    assert_eq!(
        w.dim(1),
        in_features,
        "linear: weight is [{}, {}] but x's last dimension is {in_features}. \
         HuggingFace stores nn.Linear as [out, in]; a [in, out] weight here \
         means the checkpoint convention was mis-read",
        w.dim(0),
        w.dim(1)
    );

    x.dot_general(
        w,
        &DotDimensionNumbers {
            lhs_batching: vec![],
            rhs_batching: vec![],
            lhs_contracting: vec![x.rank() - 1],
            // ⭐ Axis 1, not 0 — this single index is the whole fix.
            rhs_contracting: vec![1],
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;
    use crate::stablehlo::types::{DType, Shape};

    #[test]
    fn linear_contracts_the_in_features_axis() {
        let mut cx = TraceCx::new("main", "lin");
        let x = cx.input("x", Shape::new([1, 128, 896], DType::F32));
        // HF layout: gate_proj is [4864, 896].
        let w = cx.weight("gate_proj.weight", Shape::new([4864, 896], DType::F32));
        let y = linear(&x, &w);
        assert_eq!(y.shape().dims, vec![1, 128, 4864]);

        let mlir = cx.finish(&[&y]).mlir;
        assert!(mlir.contains("lhs_contracting_dimensions = [2],"), "{mlir}");
        assert!(
            mlir.contains("rhs_contracting_dimensions = [1]"),
            "must contract the weight's in_features axis:\n{mlir}"
        );
        assert!(
            mlir.contains("(tensor<1x128x896xf32>, tensor<4864x896xf32>) -> tensor<1x128x4864xf32>"),
            "{mlir}"
        );
    }

    /// ⛔ The narrowing projection is where the orientation is visible.
    /// Qwen2's `k_proj` is `[128, 896]`: 2 kv heads × 64.
    #[test]
    fn linear_handles_the_gqa_narrowing_projection() {
        let mut cx = TraceCx::new("main", "lin");
        let x = cx.input("x", Shape::new([1, 8, 896], DType::F32));
        let w = cx.weight("k_proj.weight", Shape::new([128, 896], DType::F32));
        let y = linear(&x, &w);
        assert_eq!(y.shape().dims, vec![1, 8, 128]);
    }

    /// A `[in, out]` weight must be refused, not silently contracted on the
    /// wrong axis.
    #[test]
    #[should_panic(expected = "HuggingFace stores nn.Linear as [out, in]")]
    fn a_transposed_weight_is_refused() {
        let mut cx = TraceCx::new("main", "lin");
        let x = cx.input("x", Shape::new([1, 8, 896], DType::F32));
        // The old, wrong orientation.
        let w = cx.weight("k_proj.weight", Shape::new([896, 128], DType::F32));
        let _ = linear(&x, &w);
    }

    /// ⚠️ A square projection is orientation-blind: both layouts have the same
    /// shape, so nothing here can tell them apart. This test exists to record
    /// that limitation, not to claim coverage.
    #[test]
    fn a_square_projection_cannot_be_checked_by_shape_alone() {
        let mut cx = TraceCx::new("main", "lin");
        let x = cx.input("x", Shape::new([1, 8, 896], DType::F32));
        let w = cx.weight("q_proj.weight", Shape::new([896, 896], DType::F32));
        let y = linear(&x, &w);
        assert_eq!(y.shape().dims, vec![1, 8, 896]);
        // Which is exactly why q_proj/o_proj did not appear among the 120
        // reported disagreements while being equally wrong.
    }
}
