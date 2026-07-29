//! RMSNorm (ARTX03 §2).

use crate::ops::util::{broadcast_keepdim_to, reduce_sum_keepdim, splat_like};
use crate::precision;
use crate::tensor::Tensor;

/// RMSNorm over the last dimension, scaled by `weight`.
///
/// ```text
/// RMSNorm(x, w) = x / sqrt(mean(x²) + ε) · w
/// ```
///
/// ⛔ **ε goes on the mean, inside the sqrt.** `x / sqrt(mean(x²) + ε)` is not
/// the same function as `x / (sqrt(mean(x²)) + ε)`, and it is not the same as
/// adding ε to `x²`. For ordinary activations the three agree to several
/// decimal places, which is exactly what makes this dangerous: the output stays
/// fluent and the error only shows up on small-norm rows.
///
/// This placement matches llama.cpp's `ggml_compute_forward_rms_norm_f32`
/// (`ops.cpp:3795`), which is the reference glproc was validated against.
///
/// * `x`: `[..., D]`
/// * `weight`: `[D]`
/// * output: `[..., D]` at `x`'s dtype
///
/// The sum of squares runs at [`precision::PrecisionPolicy::norm_reduce`]. That
/// matters more here than almost anywhere else: squaring a BF16 activation and
/// summing thousands of them in BF16 loses the low bits of every term.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    rms_norm_impl(x, weight, eps, false)
}

/// RMSNorm with Gemma's zero-centered weight convention (ARTX11 §4.2):
///
/// ```text
/// RMSNormZeroCentered(x, w) = x / sqrt(mean(x²) + ε) · (1 + w)
/// ```
///
/// ⛔ Everything [`rms_norm`]'s doc comment says about ε placement applies
/// unchanged here — this only changes the final scale from `w` to `1 + w`.
/// Gemma's checkpoints store norm weights centered at 0 (so an untrained/
/// identity norm is `weight = 0`, not `weight = 1`); applying them as a plain
/// `rms_norm` scale is shape-valid and silently wrong (P4) — every norm in
/// every layer would be scaled by a value near 0 instead of near 1.
pub fn rms_norm_zero_centered(x: &Tensor, weight: &Tensor, eps: f64) -> Tensor {
    rms_norm_impl(x, weight, eps, true)
}

fn rms_norm_impl(x: &Tensor, weight: &Tensor, eps: f64, zero_centered: bool) -> Tensor {
    let rank = x.rank();
    assert!(rank >= 1, "rms_norm: x must have at least rank 1");
    let d = x.dim(rank - 1);
    assert_eq!(
        weight.shape().dims.as_slice(),
        &[d],
        "rms_norm: weight must be [D] matching x's last dimension ({d})"
    );

    let orig_dtype = x.dtype();
    let acc = precision::current().norm_reduce;

    // 1. Upcast, then square.
    let x_acc = x.to_dtype(acc);
    let x2 = x_acc.mul(&x_acc);

    // 2. mean(x²) over the last axis, keeping it.
    let sum = reduce_sum_keepdim(&x2, rank - 1);
    let keep_dims = sum.shape().dims.clone();
    let inv_d = splat_like(&sum, 1.0 / d as f64, keep_dims.clone(), acc);
    let mean = sum.mul(&inv_d);

    // 3. ε on the mean — inside the sqrt, per the doc comment above.
    let eps_t = splat_like(&mean, eps, keep_dims, acc);
    let mean_eps = mean.add(&eps_t);

    // 4. 1/sqrt(...), broadcast back over D.
    let rrms = mean_eps.rsqrt();
    let rrms_bc = broadcast_keepdim_to(&rrms, x_acc.shape().dims.clone());
    let normed = x_acc.mul(&rrms_bc);

    // 5. Back to the activation dtype, then scale by the weight — `1 + w`
    // under the zero-centered convention, `w` otherwise.
    let normed_out = normed.to_dtype(orig_dtype);
    let w = weight.to_dtype(orig_dtype);
    let w = if zero_centered {
        let one = splat_like(&w, 1.0, w.shape().dims.clone(), orig_dtype);
        w.add(&one)
    } else {
        w
    };
    let w_bc = w.broadcast_to(vec![rank - 1], x.shape().dims.clone());
    normed_out.mul(&w_bc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;
    use crate::stablehlo::types::{DType, Shape};
    use crate::PrecisionPolicy;

    /// The scalar reference, written the way llama.cpp writes it.
    fn reference_rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
        let d = x.len() as f64;
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / d;
        let scale = 1.0 / (mean_sq + eps).sqrt();
        x.iter().zip(w).map(|(v, ww)| v * scale * ww).collect()
    }

    /// ⭐ Pins the ε placement numerically, on the input where the three
    /// plausible placements actually diverge.
    ///
    /// gljax cannot execute MLIR on this machine (no PJRT plugin), so this
    /// compares the *reference* against the two wrong forms rather than against
    /// gljax's output. What it proves is that the distinction is real and worth
    /// the doc comment; what it cannot prove is that the emitted graph
    /// implements the right one. That needs ARTX12 Part B and a plugin.
    #[test]
    fn rms_norm_epsilon_inside_sqrt_matches_llama_cpp() {
        // Relative tolerance for f64 scalar arithmetic: a handful of ulps.
        const TOL_REL: f64 = 1e-12;
        let eps = 1e-6;
        let w = [1.0, 1.0, 1.0, 1.0];

        // Ordinary activations: all three placements agree to ~5 decimals,
        // which is why this bug survives casual testing.
        let ordinary = [1.0, 2.0, 3.0, 4.0];
        let want = reference_rms_norm(&ordinary, &w, eps);
        let eps_outside: Vec<f64> = {
            let d = ordinary.len() as f64;
            let mean_sq: f64 = ordinary.iter().map(|v| v * v).sum::<f64>() / d;
            let scale = 1.0 / (mean_sq.sqrt() + eps);
            ordinary.iter().map(|v| v * scale).collect()
        };
        let max_rel = want
            .iter()
            .zip(&eps_outside)
            .map(|(a, b)| ((a - b) / a).abs())
            .fold(0.0f64, f64::max);
        assert!(
            max_rel < 1e-6,
            "on ordinary inputs the wrong placement should look almost right \
             (got {max_rel:e}) — that is the trap this test documents"
        );

        // Small-norm row: now they diverge by orders of magnitude.
        let small = [1e-4, 1e-4, 1e-4, 1e-4];
        let want_small = reference_rms_norm(&small, &w, eps);
        let wrong_small: Vec<f64> = {
            let d = small.len() as f64;
            let mean_sq: f64 = small.iter().map(|v| v * v).sum::<f64>() / d;
            let scale = 1.0 / (mean_sq.sqrt() + eps);
            small.iter().map(|v| v * scale).collect()
        };
        let rel = ((want_small[0] - wrong_small[0]) / want_small[0]).abs();
        assert!(
            rel > 1e-2,
            "on a small-norm row the placements must visibly disagree, got {rel:e}"
        );

        // And the reference itself is stable.
        let again = reference_rms_norm(&ordinary, &w, eps);
        for (a, b) in want.iter().zip(&again) {
            assert!((a - b).abs() <= a.abs() * TOL_REL);
        }
    }

    #[test]
    fn rms_norm_reduces_the_last_axis_and_keeps_the_shape() {
        let mut cx = TraceCx::new("main", "rms");
        let x = cx.input("x", Shape::new([1, 128, 896], DType::F32));
        let w = cx.weight("norm.weight", Shape::new([896], DType::F32));
        let y = rms_norm(&x, &w, 1e-6);
        assert_eq!(y.shape().dims, vec![1, 128, 896]);

        let mlir = cx.finish(&[&y]).mlir;
        assert!(mlir.contains("dimensions = array<i64: 2>"), "{mlir}");
        // 1/D as an exact float: 1/896.
        assert!(
            mlir.contains(&format!("dense<{:?}>", 1.0f64 / 896.0)),
            "{mlir}"
        );
        assert!(mlir.contains("dense<1.0e-6>"), "{mlir}");
        assert!(mlir.contains(r#""stablehlo.rsqrt""#), "{mlir}");
    }

    /// The ε add must land on the mean, between the `1/D` multiply and the
    /// `rsqrt` — not before the reduce and not after the rsqrt.
    #[test]
    fn epsilon_is_added_between_the_mean_and_the_rsqrt() {
        let mut cx = TraceCx::new("main", "rms");
        let x = cx.input("x", Shape::new([2, 8], DType::F32));
        let w = cx.weight("w", Shape::new([8], DType::F32));
        let y = rms_norm(&x, &w, 1e-6);
        let mlir = cx.finish(&[&y]).mlir;

        let reduce_at = mlir.find(r#""stablehlo.reduce""#).unwrap();
        let eps_at = mlir.find("dense<1.0e-6>").unwrap();
        let rsqrt_at = mlir.find(r#""stablehlo.rsqrt""#).unwrap();
        assert!(reduce_at < eps_at, "ε must not be added before the reduce:\n{mlir}");
        assert!(eps_at < rsqrt_at, "ε must be inside the sqrt:\n{mlir}");
    }

    #[test]
    fn rms_norm_upcasts_the_reduce_under_a_bf16_policy() {
        let built = crate::with_policy(PrecisionPolicy::bf16(), || {
            let mut cx = TraceCx::new("main", "rms");
            let x = cx.input("x", Shape::new([1, 8, 896], DType::BF16));
            let w = cx.weight("w", Shape::new([896], DType::BF16));
            let y = rms_norm(&x, &w, 1e-6);
            assert_eq!(y.dtype(), DType::BF16);
            cx.finish(&[&y])
        });
        assert!(
            built.mlir.contains("(tensor<1x8x896xbf16>) -> tensor<1x8x896xf32>"),
            "the sum of squares must run in f32:\n{}",
            built.mlir
        );
    }

    #[test]
    #[should_panic(expected = "weight must be [D]")]
    fn rms_norm_rejects_a_weight_that_does_not_match_the_last_axis() {
        let mut cx = TraceCx::new("main", "rms");
        let x = cx.input("x", Shape::new([2, 8], DType::F32));
        let w = cx.weight("w", Shape::new([16], DType::F32));
        let _ = rms_norm(&x, &w, 1e-6);
    }

    /// ⛔ The whole point of the zero-centered variant: a weight of all zeros
    /// (Gemma's "untrained"/identity initialization) must scale by 1, not by
    /// 0. Plain `rms_norm` on the same weight would zero every output.
    #[test]
    fn rms_norm_zero_centered_treats_a_zero_weight_as_the_identity_scale() {
        let mut cx = TraceCx::new("main", "rms");
        let x = cx.input("x", Shape::new([2, 8], DType::F32));
        let w = cx.weight("w", Shape::new([8], DType::F32));
        let y = rms_norm_zero_centered(&x, &w, 1e-6);
        let mlir = cx.finish(&[&y]).mlir;
        // The weight must be added to a constant dense<1.0> before the final
        // multiply — the literal signature of "(1 + weight)".
        assert!(mlir.contains("dense<1.0>"), "must add a literal 1 to the weight:\n{mlir}");
        assert!(
            mlir.contains(r#""stablehlo.add"(%v1, "#),
            "the weight (%v1) itself must be an add operand:\n{mlir}"
        );
    }

    #[test]
    fn rms_norm_zero_centered_emits_one_more_add_than_plain_rms_norm() {
        let mut plain_cx = TraceCx::new("main", "rms");
        let x1 = plain_cx.input("x", Shape::new([2, 8], DType::F32));
        let w1 = plain_cx.weight("w", Shape::new([8], DType::F32));
        let plain = rms_norm(&x1, &w1, 1e-6);
        let plain_mlir = plain_cx.finish(&[&plain]).mlir;

        let mut zc_cx = TraceCx::new("main", "rms");
        let x2 = zc_cx.input("x", Shape::new([2, 8], DType::F32));
        let w2 = zc_cx.weight("w", Shape::new([8], DType::F32));
        let zc = rms_norm_zero_centered(&x2, &w2, 1e-6);
        let zc_mlir = zc_cx.finish(&[&zc]).mlir;

        let plain_adds = plain_mlir.matches(r#""stablehlo.add""#).count();
        let zc_adds = zc_mlir.matches(r#""stablehlo.add""#).count();
        assert_eq!(zc_adds, plain_adds + 1, "plain:\n{plain_mlir}\nzero-centered:\n{zc_mlir}");
    }

    #[test]
    #[should_panic(expected = "weight must be [D]")]
    fn rms_norm_zero_centered_rejects_a_weight_that_does_not_match_the_last_axis() {
        let mut cx = TraceCx::new("main", "rms");
        let x = cx.input("x", Shape::new([2, 8], DType::F32));
        let w = cx.weight("w", Shape::new([16], DType::F32));
        let _ = rms_norm_zero_centered(&x, &w, 1e-6);
    }
}
