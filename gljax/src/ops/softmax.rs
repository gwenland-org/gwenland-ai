//! Numerically stable softmax (ARTX03 §1).

use crate::ops::util::{broadcast_keepdim_to, reduce_max_keepdim, reduce_sum_keepdim};
use crate::precision;
use crate::tensor::Tensor;

/// Softmax along `dim`.
///
/// ```text
/// softmax(x)_i = exp(x_i - max(x)) / Σ_j exp(x_j - max(x))
/// ```
///
/// ⚠️ The subtract-max is not an optimisation. Attention logits reach into the
/// tens; `exp(100)` is `inf` in F32 and the whole row becomes NaN. There is no
/// "fast path" that skips it.
///
/// The reduces run at [`precision::PrecisionPolicy::softmax_reduce`] and the
/// result is returned at the input dtype, so a BF16 trace still sums in F32.
pub fn softmax(x: &Tensor, dim: usize) -> Tensor {
    assert!(
        dim < x.rank(),
        "softmax: dim {dim} out of range for rank {}",
        x.rank()
    );
    let orig_dtype = x.dtype();
    let acc_dtype = precision::current().softmax_reduce;
    let dims = x.shape().dims.clone();

    let x_acc = x.to_dtype(acc_dtype);

    let max_keep = reduce_max_keepdim(&x_acc, dim);
    let max_bc = broadcast_keepdim_to(&max_keep, dims.clone());
    let shifted = x_acc.sub(&max_bc);

    let exp = shifted.exp();

    let sum_keep = reduce_sum_keepdim(&exp, dim);
    let sum_bc = broadcast_keepdim_to(&sum_keep, dims);

    exp.div(&sum_bc).to_dtype(orig_dtype)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;
    use crate::stablehlo::types::{DType, Shape};
    use crate::PrecisionPolicy;

    #[test]
    fn softmax_subtracts_the_row_max_before_exponentiating() {
        let mut cx = TraceCx::new("main", "softmax");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let y = softmax(&x, 1);
        assert_eq!(y.shape().dims, vec![2, 4]);
        let mlir = cx.finish(&[&y]).mlir;

        // The order matters: maximum, then subtract, then exponential.
        let max_at = mlir.find(r#""stablehlo.maximum""#).expect("no reduce-max");
        let sub_at = mlir.find(r#""stablehlo.subtract""#).expect("no subtract");
        let exp_at = mlir
            .find(r#""stablehlo.exponential""#)
            .expect("no exponential");
        assert!(max_at < sub_at, "max must precede the subtract:\n{mlir}");
        assert!(sub_at < exp_at, "subtract must precede the exp:\n{mlir}");
    }

    #[test]
    fn softmax_reduce_init_is_negative_infinity() {
        // A finite init clamps the max from below and skews the whole row.
        let mut cx = TraceCx::new("main", "softmax");
        let x = cx.input("x", Shape::new([2, 4], DType::F32));
        let y = softmax(&x, 1);
        let mlir = cx.finish(&[&y]).mlir;
        assert!(mlir.contains("dense<0xFF800000>"), "{mlir}");
    }

    #[test]
    fn softmax_upcasts_for_the_reduce_and_returns_the_input_dtype() {
        let built = crate::with_policy(PrecisionPolicy::bf16(), || {
            let mut cx = TraceCx::new("main", "softmax");
            let x = cx.input("x", Shape::new([1, 14, 128, 128], DType::BF16));
            let y = softmax(&x, 3);
            assert_eq!(y.dtype(), DType::BF16, "must return at the input dtype");
            cx.finish(&[&y])
        });
        let mlir = &built.mlir;
        assert!(mlir.contains("-> tensor<1x14x128x128xf32>"), "{mlir}");
        assert!(mlir.contains("-> tensor<1x14x128x128xbf16>"), "{mlir}");
    }

    #[test]
    fn softmax_at_f32_policy_emits_no_conversions() {
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            let mut cx = TraceCx::new("main", "softmax");
            let x = cx.input("x", Shape::new([2, 4], DType::F32));
            let y = softmax(&x, 1);
            cx.finish(&[&y])
        });
        assert!(
            !built.mlir.contains("stablehlo.convert"),
            "an all-f32 softmax needs no upcast:\n{}",
            built.mlir
        );
    }
}
