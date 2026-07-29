//! Grouped-query causal attention (ARTX03 §4).

use std::rc::Rc;

use crate::graph::value::SsaValue;
use crate::ops::softmax::softmax;
use crate::ops::util::{dense_const_f32, expect_rank4, splat_like};
use crate::stablehlo::ops::{emit_dynamic_slice, DotDimensionNumbers};
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;
use crate::GlError;

/// Builds the causal mask as a dense `[1, 1, S, S]` constant: `0` on and below
/// the diagonal, `−∞` above it.
///
/// ⚠️ This is O(S²) *text*. At S=512 it is ~2 MB of MLIR; at ARTX05's 2048
/// bucket it is 4.2 M elements and [`dense_const_f32`] refuses it outright. See
/// [`crate::stablehlo::ops::MAX_DENSE_CONSTANT_ELEMS`] — a mask at that scale
/// wants to be a runtime input, not a baked constant, and that decision belongs
/// to Wave A5 where the bucket grid is chosen.
pub fn causal_mask(like: &Tensor, seq_len: usize, dtype: DType) -> Result<Tensor, GlError> {
    let mut data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let shape = Shape::new([1, 1, seq_len, seq_len], DType::F32);
    let mask = dense_const_f32(like, &data, shape)?;
    Ok(mask.to_dtype(dtype))
}

/// Extracts row `pos` of a precomputed `[1, 1, W, W]` causal mask as the
/// `[1, 1, 1, W]` position mask a decode step needs.
///
/// ⭐ The static causal mask already encodes exactly the rule ARTX05 §3
/// describes computing from scratch via `iota` + `compare` + `select`:
/// `causal_mask[pos, j] = 0` if `j <= pos` else `-inf` — decode's "attend to
/// real history, mask the unwritten tail of the bucket" is just row `pos` of
/// the same matrix `causal_mask` already builds. Reusing it needs one
/// `dynamic_slice` on a mask that's already emitted and parse-verified,
/// instead of three new StableHLO op emitters that would each carry their own
/// syntax risk (the reduce-region-braces and empty-`array<i64>` bugs were
/// exactly this kind of first use).
pub fn causal_mask_row(mask: &Tensor, pos: &Tensor, zero: &Tensor) -> Tensor {
    assert_eq!(
        mask.rank(),
        4,
        "causal_mask_row: mask must be [1, 1, W, W], got rank {}",
        mask.rank()
    );
    assert_eq!(
        (mask.dim(0), mask.dim(1)),
        (1, 1),
        "causal_mask_row: mask must be [1, 1, W, W], got {:?}",
        mask.shape().dims
    );
    let w = mask.dim(3);
    assert_eq!(
        mask.dim(2),
        w,
        "causal_mask_row: mask must be square, got [{}, {}]",
        mask.dim(2),
        w
    );

    let idx = |t: &Tensor| (t.value().ssa(), t.shape().clone());
    let out_shape = Shape::new([1, 1, 1, w], mask.dtype());
    let name = {
        let mut b = mask.builder().borrow_mut();
        emit_dynamic_slice(
            b.emitter_mut(),
            mask.value().ssa(),
            &[idx(zero), idx(zero), idx(pos), idx(zero)],
            &[1, 1, 1, w],
            mask.shape(),
            &out_shape,
        )
    };
    Tensor::new(SsaValue::new(name, out_shape), Rc::clone(mask.builder()))
}

/// Scaled dot-product attention with GQA head expansion and an additive mask.
///
/// * `q`: `[B, n_heads, S_q, head_dim]`
/// * `k`, `v`: `[B, n_kv_heads, S_kv, head_dim]`
/// * `mask`: `[1, 1, S_q, S_kv]`, additive (`0` / `−∞`)
/// * output: `[B, n_heads, S_q, head_dim]`
///
/// ⛔ **KV head expansion is interleave-sensitive.** ARTX11 §7 records the
/// failure directly: grouping KV heads in blocks instead of interleaved gives
/// identical shapes and corrupted attention. Query head `h` must read KV head
/// `h / repeat` — which is what `reshape → broadcast → reshape` produces, in
/// that order, when the repeat axis is inserted *after* the KV-head axis.
pub fn gqa_attention(q: &Tensor, k: &Tensor, v: &Tensor, mask: &Tensor) -> Tensor {
    let head_dim = expect_rank4(q, "gqa_attention: q")[3];
    let default_scale = 1.0 / (head_dim as f64).sqrt();
    gqa_attention_with_scale(q, k, v, mask, default_scale)
}

/// [`gqa_attention`] with an explicit query scale, overriding the default
/// `1/sqrt(head_dim)` (ARTX11 §4.2's "custom query pre-attention scalar" —
/// Gemma scales by a value tied to the *un-grouped* head count rather than
/// `head_dim` itself).
pub fn gqa_attention_with_scale(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f64,
) -> Tensor {
    let [b, n_heads, s_q, head_dim] = expect_rank4(q, "gqa_attention: q");
    let [bk, n_kv_heads, s_kv, dk] = expect_rank4(k, "gqa_attention: k");
    let [bv, n_kv_v, s_v, dv] = expect_rank4(v, "gqa_attention: v");

    assert!(
        b == bk && b == bv,
        "gqa_attention: batch disagrees between q/k/v ({b}, {bk}, {bv})"
    );
    assert!(
        n_kv_heads == n_kv_v,
        "gqa_attention: k has {n_kv_heads} kv heads, v has {n_kv_v}"
    );
    assert!(
        s_kv == s_v,
        "gqa_attention: k has {s_kv} positions, v has {s_v}"
    );
    assert!(
        head_dim == dk && head_dim == dv,
        "gqa_attention: head_dim disagrees ({head_dim}, {dk}, {dv})"
    );
    assert!(
        n_kv_heads > 0 && n_heads.is_multiple_of(n_kv_heads),
        "gqa_attention: {n_heads} query heads is not a multiple of {n_kv_heads} kv heads"
    );
    assert_eq!(
        mask.shape().dims,
        vec![1, 1, s_q, s_kv],
        "gqa_attention: mask must be [1, 1, S_q, S_kv]"
    );

    let repeat = n_heads / n_kv_heads;
    let k_exp = expand_kv_heads(k, repeat);
    let v_exp = expand_kv_heads(v, repeat);

    // Scale Q before the product. Scaling Q rather than the scores is one
    // fewer S×S-sized elementwise pass.
    let scale_t = splat_like(q, scale, q.shape().dims.clone(), q.dtype());
    let q_scaled = q.mul(&scale_t);

    // QKᵀ: [B,H,S_q,D] · [B,H,S_kv,D] contracting D on both sides. Expressed
    // directly rather than as transpose-then-matmul — dot_general can contract
    // the last axis of both operands, so the transpose would be dead weight.
    let scores = q_scaled.dot_general(
        &k_exp,
        &DotDimensionNumbers {
            lhs_batching: vec![0, 1],
            rhs_batching: vec![0, 1],
            lhs_contracting: vec![3],
            rhs_contracting: vec![3],
        },
    );

    // Additive mask, broadcast over batch and heads.
    let mask_bc = mask
        .to_dtype(scores.dtype())
        .broadcast_to(vec![0, 1, 2, 3], vec![b, n_heads, s_q, s_kv]);
    let masked = scores.add(&mask_bc);

    let weights = softmax(&masked, 3);

    // Weights · V: [B,H,S_q,S_kv] · [B,H,S_kv,D] -> [B,H,S_q,D]
    weights.dot_general(
        &v_exp,
        &DotDimensionNumbers {
            lhs_batching: vec![0, 1],
            rhs_batching: vec![0, 1],
            lhs_contracting: vec![3],
            rhs_contracting: vec![2],
        },
    )
}

/// Per-head RMSNorm on Q and K before RoPE (ARTX11 §4.2's QK-norm).
///
/// * `q`: `[B, n_heads, S, head_dim]`, `q_norm_weight`: `[head_dim]`
/// * `k`: `[B, n_kv_heads, S, head_dim]`, `k_norm_weight`: `[head_dim]`
///
/// [`crate::ops::norm::rms_norm`] already reduces over *the last axis
/// regardless of rank* — `head_dim` is q/k's last axis whether the tensor is
/// rank 2 (a single head, in a unit test) or rank 4 (the real `[B,H,S,D]`
/// shape), so this needs no new reduction machinery, only the right weight
/// shape at the call site.
pub fn apply_qk_norm(
    q: &Tensor,
    k: &Tensor,
    q_norm_weight: &Tensor,
    k_norm_weight: &Tensor,
    eps: f64,
) -> (Tensor, Tensor) {
    let q_normed = crate::ops::norm::rms_norm(q, q_norm_weight, eps);
    let k_normed = crate::ops::norm::rms_norm(k, k_norm_weight, eps);
    (q_normed, k_normed)
}

/// `[B, n_kv, S, D]` → `[B, n_kv · repeat, S, D]`, each KV head repeated
/// `repeat` times **consecutively**.
///
/// The repeat axis is inserted directly after the KV-head axis, so flattening
/// puts head `h`'s copies at output positions `h·repeat ..< (h+1)·repeat`. That
/// is the layout query head `q` expects when it reads KV head `q / repeat`.
fn expand_kv_heads(kv: &Tensor, repeat: usize) -> Tensor {
    if repeat == 1 {
        return kv.clone_ref();
    }
    let [b, n_kv, s, d] = expect_rank4(kv, "expand_kv_heads");
    kv.reshape(vec![b, n_kv, 1, s, d])
        .broadcast_to(vec![0, 1, 2, 3, 4], vec![b, n_kv, repeat, s, d])
        .reshape(vec![b, n_kv * repeat, s, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    #[test]
    fn causal_mask_is_zero_on_and_below_the_diagonal() {
        let mut cx = TraceCx::new("main", "mask");
        let x = cx.input("x", Shape::new([1], DType::F32));
        let m = causal_mask(&x, 3, DType::F32).expect("mask");
        assert_eq!(m.shape().dims, vec![1, 1, 3, 3]);
        let mlir = cx.finish(&[&m]).mlir;

        // Row-major f32 little-endian: 0.0 is 00000000, -inf is 000080FF.
        let zero = "00000000";
        let ninf = "000080FF";
        let expected = format!(
            "{zero}{ninf}{ninf}{zero}{zero}{ninf}{zero}{zero}{zero}"
        );
        assert!(
            mlir.contains(&expected),
            "mask must be upper-triangular -inf:\n{mlir}"
        );
    }

    #[test]
    fn causal_mask_refuses_a_bucket_that_would_emit_tens_of_megabytes() {
        let mut cx = TraceCx::new("main", "mask");
        let x = cx.input("x", Shape::new([1], DType::F32));
        // 2048² = 4.2M elements.
        let err = causal_mask(&x, 2048, DType::F32).expect_err("must refuse");
        assert!(err.to_string().contains("runtime weight"), "{err}");
        let _ = cx;
    }

    /// ⛔ ARTX11 §7's silent-corruption case. Query head `h` must map to KV
    /// head `h / repeat`; block grouping (`h % n_kv`) has the same shape.
    #[test]
    fn kv_head_expansion_repeats_each_head_consecutively() {
        let mut cx = TraceCx::new("main", "gqa");
        // Qwen2-0.5B: 14 query heads, 2 KV heads, repeat 7.
        let k = cx.input("k", Shape::new([1, 2, 8, 64], DType::F32));
        let expanded = expand_kv_heads(&k, 7);
        assert_eq!(expanded.shape().dims, vec![1, 14, 8, 64]);

        let mlir = cx.finish(&[&expanded]).mlir;
        // The repeat axis goes at index 2 — after the kv-head axis, before S.
        assert!(
            mlir.contains("(tensor<1x2x8x64xf32>) -> tensor<1x2x1x8x64xf32>"),
            "the repeat axis must be inserted after the kv-head axis:\n{mlir}"
        );
        assert!(
            mlir.contains("(tensor<1x2x1x8x64xf32>) -> tensor<1x2x7x8x64xf32>"),
            "{mlir}"
        );
        assert!(
            mlir.contains("(tensor<1x2x7x8x64xf32>) -> tensor<1x14x8x64xf32>"),
            "{mlir}"
        );
    }

    #[test]
    fn mha_needs_no_expansion() {
        let mut cx = TraceCx::new("main", "mha");
        let k = cx.input("k", Shape::new([1, 4, 8, 64], DType::F32));
        let same = expand_kv_heads(&k, 1);
        assert_eq!(same.value(), k.value(), "repeat=1 must emit nothing");
        let mlir = cx.finish(&[&same]).mlir;
        assert!(!mlir.contains("stablehlo."), "{mlir}");
    }

    #[test]
    fn gqa_attention_produces_query_shaped_output() {
        let mut cx = TraceCx::new("main", "attn");
        let q = cx.input("q", Shape::new([1, 14, 16, 64], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 16, 64], DType::F32));
        let v = cx.input("v", Shape::new([1, 2, 16, 64], DType::F32));
        let mask = causal_mask(&q, 16, DType::F32).expect("mask");
        let out = gqa_attention(&q, &k, &v, &mask);
        assert_eq!(out.shape().dims, vec![1, 14, 16, 64]);

        let mlir = cx.finish(&[&out]).mlir;
        // Scores contract head_dim on both operands, producing [B,H,S,S].
        assert!(
            mlir.contains("(tensor<1x14x16x64xf32>, tensor<1x14x16x64xf32>) -> tensor<1x14x16x16xf32>"),
            "QK^T must contract head_dim:\n{mlir}"
        );
        // AV contracts S_kv against V's position axis.
        assert!(
            mlir.contains("(tensor<1x14x16x16xf32>, tensor<1x14x16x64xf32>) -> tensor<1x14x16x64xf32>"),
            "AV must contract the key axis:\n{mlir}"
        );
    }

    #[test]
    fn attention_scales_queries_by_one_over_sqrt_head_dim() {
        let mut cx = TraceCx::new("main", "attn");
        let q = cx.input("q", Shape::new([1, 2, 4, 64], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 4, 64], DType::F32));
        let v = cx.input("v", Shape::new([1, 2, 4, 64], DType::F32));
        let mask = causal_mask(&q, 4, DType::F32).expect("mask");
        let out = gqa_attention(&q, &k, &v, &mask);
        let mlir = cx.finish(&[&out]).mlir;
        // 1/sqrt(64) = 0.125, exactly representable.
        assert!(mlir.contains("dense<0.125>"), "{mlir}");
    }

    #[test]
    fn causal_mask_row_slices_out_one_row_at_a_runtime_position() {
        let mut cx = TraceCx::new("main", "mask_row");
        let x = cx.input("x", Shape::new([1], DType::F32));
        let mask = causal_mask(&x, 8, DType::F32).expect("mask");
        let pos = cx.input("pos", Shape::scalar(DType::I32));
        let zero = cx.input("zero", Shape::scalar(DType::I32));
        let row = causal_mask_row(&mask, &pos, &zero);
        assert_eq!(row.shape().dims, vec![1, 1, 1, 8]);

        let mlir = cx.finish(&[&row]).mlir;
        assert!(mlir.contains(r#""stablehlo.dynamic_slice""#), "{mlir}");
        assert!(mlir.contains("slice_sizes = array<i64: 1, 1, 1, 8>"), "{mlir}");
    }

    #[test]
    #[should_panic(expected = "must be square")]
    fn causal_mask_row_rejects_a_non_square_mask() {
        let mut cx = TraceCx::new("main", "mask_row");
        let mask = cx.input("mask", Shape::new([1, 1, 4, 8], DType::F32));
        let pos = cx.input("pos", Shape::scalar(DType::I32));
        let zero = cx.input("zero", Shape::scalar(DType::I32));
        let _ = causal_mask_row(&mask, &pos, &zero);
    }

    /// The whole point of `causal_mask_row`: its output must be a drop-in
    /// `[1, 1, S_q, S_kv]` mask for a decode-shaped `gqa_attention` call
    /// (`S_q = 1`), with no dynamic-shape gymnastics on the attention side.
    #[test]
    fn causal_mask_row_is_usable_as_a_decode_step_attention_mask() {
        let mut cx = TraceCx::new("main", "mask_row");
        let x = cx.input("x", Shape::new([1], DType::F32));
        let mask = causal_mask(&x, 8, DType::F32).expect("mask");
        let pos = cx.input("pos", Shape::scalar(DType::I32));
        let zero = cx.input("zero", Shape::scalar(DType::I32));
        let row = causal_mask_row(&mask, &pos, &zero);

        let q = cx.input("q", Shape::new([1, 2, 1, 4], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 8, 4], DType::F32));
        let v = cx.input("v", Shape::new([1, 2, 8, 4], DType::F32));
        let out = gqa_attention(&q, &k, &v, &row);
        assert_eq!(out.shape().dims, vec![1, 2, 1, 4]);
    }

    #[test]
    fn gqa_attention_with_scale_overrides_the_default_head_dim_scale() {
        let mut cx = TraceCx::new("main", "attn");
        let q = cx.input("q", Shape::new([1, 2, 4, 64], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 4, 64], DType::F32));
        let v = cx.input("v", Shape::new([1, 2, 4, 64], DType::F32));
        let mask = causal_mask(&q, 4, DType::F32).expect("mask");
        // Gemma-style: a custom scalar unrelated to 1/sqrt(head_dim) (0.125).
        let out = gqa_attention_with_scale(&q, &k, &v, &mask, 0.0625);
        let mlir = cx.finish(&[&out]).mlir;
        assert!(mlir.contains("dense<0.0625>"), "{mlir}");
        assert!(!mlir.contains("dense<0.125>"), "{mlir}");
    }

    #[test]
    fn gqa_attention_and_gqa_attention_with_scale_agree_at_the_default() {
        let mut default_cx = TraceCx::new("main", "attn");
        let q1 = default_cx.input("q", Shape::new([1, 2, 4, 64], DType::F32));
        let k1 = default_cx.input("k", Shape::new([1, 2, 4, 64], DType::F32));
        let v1 = default_cx.input("v", Shape::new([1, 2, 4, 64], DType::F32));
        let mask1 = causal_mask(&q1, 4, DType::F32).expect("mask");
        let out1 = gqa_attention(&q1, &k1, &v1, &mask1);
        let mlir1 = default_cx.finish(&[&out1]).mlir;

        let mut explicit_cx = TraceCx::new("main", "attn");
        let q2 = explicit_cx.input("q", Shape::new([1, 2, 4, 64], DType::F32));
        let k2 = explicit_cx.input("k", Shape::new([1, 2, 4, 64], DType::F32));
        let v2 = explicit_cx.input("v", Shape::new([1, 2, 4, 64], DType::F32));
        let mask2 = causal_mask(&q2, 4, DType::F32).expect("mask");
        let out2 = gqa_attention_with_scale(&q2, &k2, &v2, &mask2, 1.0 / (64f64).sqrt());
        let mlir2 = explicit_cx.finish(&[&out2]).mlir;

        assert_eq!(mlir1, mlir2, "gqa_attention must be gqa_attention_with_scale at the default");
    }

    #[test]
    fn apply_qk_norm_normalizes_over_head_dim_not_the_head_axis() {
        let mut cx = TraceCx::new("main", "qknorm");
        let q = cx.input("q", Shape::new([1, 14, 16, 64], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 16, 64], DType::F32));
        let qw = cx.weight("q_norm.weight", Shape::new([64], DType::F32));
        let kw = cx.weight("k_norm.weight", Shape::new([64], DType::F32));
        let (q_normed, k_normed) = apply_qk_norm(&q, &k, &qw, &kw, 1e-6);
        assert_eq!(q_normed.shape().dims, vec![1, 14, 16, 64]);
        assert_eq!(k_normed.shape().dims, vec![1, 2, 16, 64]);

        let mlir = cx.finish(&[&q_normed, &k_normed]).mlir;
        // One reduce per tensor (q, k), each over the last axis (index 3).
        // The reduce op's own `{dimensions = ...}` attribute is distinguished
        // from `broadcast_dimensions = ...` (which also contains axis 3, for
        // the weight broadcast) by matching the braced reduce-attribute form.
        assert_eq!(mlir.matches(r#""stablehlo.reduce""#).count(), 2, "{mlir}");
        assert_eq!(
            mlir.matches("{dimensions = array<i64: 3>}").count(),
            2,
            "QK-norm must reduce head_dim, not the head axis:\n{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "is not a multiple of")]
    fn gqa_rejects_a_head_count_that_does_not_divide() {
        let mut cx = TraceCx::new("main", "attn");
        let q = cx.input("q", Shape::new([1, 5, 4, 8], DType::F32));
        let k = cx.input("k", Shape::new([1, 2, 4, 8], DType::F32));
        let v = cx.input("v", Shape::new([1, 2, 4, 8], DType::F32));
        let mask = causal_mask(&q, 4, DType::F32).expect("mask");
        let _ = gqa_attention(&q, &k, &v, &mask);
    }
}
