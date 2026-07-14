//! Scaled dot-product attention with optional causal masking.

use crate::kernels::matmul_t;

/// Numerically stable in-place softmax over a slice.
///
/// Dispatches to the vectorized kernel (see [`crate::kernels::ops::softmax`]);
/// the scalar body that used to live here is now that module's ground truth.
pub fn softmax(x: &mut [f32]) {
    crate::kernels::softmax_inplace(x);
}

/// Scaled dot-product attention for a single head.
///
/// * `q`, `k`, `v`: `[seq_len, head_dim]` row-major
/// * returns `[seq_len, head_dim]`
///
/// Steps: `scores = Q @ K^T / sqrt(head_dim)`, optional causal mask (upper
/// triangle set to `-inf`), row-wise softmax, `output = scores @ V`.
pub fn scaled_dot_product_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_len: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();

    // scores = Q @ K^T — K is [seq_len, head_dim] so K^T access = matmul_t
    let mut scores = vec![0.0f32; seq_len * seq_len];
    matmul_t(q, k, &mut scores, seq_len, head_dim, seq_len);
    for s in scores.iter_mut() {
        *s *= scale;
    }

    if causal {
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
    }

    for row in scores.chunks_mut(seq_len) {
        softmax(row);
    }

    // output = scores @ V
    let mut out = vec![0.0f32; seq_len * head_dim];
    crate::kernels::matmul(&scores, v, &mut out, seq_len, seq_len, head_dim);
    out
}

/// Single-query attention against a KV cache — the decode-step fast path.
///
/// * `q`: `[head_dim]` — the current token's query
/// * `k_cache`, `v_cache`: `[cached_len * head_dim]` flat slices
/// * returns `[head_dim]`
///
/// Causality is implicit: the cache only contains past positions.
/// Convenience wrapper that allocates; the decode loop uses
/// [`attention_one_into`] with reused buffers instead.
pub fn attention_one(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
) -> Vec<f32> {
    let cached_len = k_cache.len() / head_dim.max(1);
    let mut scores = vec![0.0f32; cached_len];
    let mut out = vec![0.0f32; head_dim];
    attention_one_into(q, k_cache, v_cache, head_dim, &mut scores, &mut out);
    out
}

/// Allocation-free single-query attention for the decode loop.
///
/// * `scores` — scratch, `len == cached_len` (`k_cache.len() / head_dim`)
/// * `out` — result, `len == head_dim`, overwritten
pub fn attention_one_into(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    scores: &mut [f32],
    out: &mut [f32],
) {
    let cached_len = k_cache.len() / head_dim.max(1);
    debug_assert_eq!(scores.len(), cached_len);
    debug_assert_eq!(out.len(), head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let strategy = crate::simd_strategy::SimdStrategy::detect();

    for (t, s) in scores.iter_mut().enumerate() {
        let k_row = &k_cache[t * head_dim..(t + 1) * head_dim];
        // SAFETY: strategy comes from SimdStrategy::detect(), so the
        // required CPU features are present.
        let dot = match strategy {
            crate::simd_strategy::SimdStrategy::Avx512 => unsafe {
                crate::kernels::matmul::avx512::dot_f32(q, k_row)
            },
            crate::simd_strategy::SimdStrategy::Avx2 => unsafe {
                crate::kernels::matmul::avx2::dot_f32(q, k_row)
            },
            crate::simd_strategy::SimdStrategy::Scalar => {
                crate::kernels::matmul::scalar::dot_f32(q, k_row)
            }
        };
        *s = dot * scale;
    }
    softmax(scores);

    // out[d] = Σ_t scores[t] * v_cache[t][d]. This was a scalar loop while the
    // Q·K dots above it were already AVX2 — half of attention's arithmetic was
    // leaving the vector units idle. See `kernels::ops::attn_accum`.
    crate::kernels::attn_accum(scores, v_cache, out, head_dim);
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f32 = 1e-4;

    #[test]
    fn causal_attention_reference() {
        // seq_len=2, head_dim=1, q=k=[1,2], v=[10,20], scale=1.
        // scores = [[1,2],[2,4]]; causal row0 -> [1,-inf] -> [1, 0]
        // row1 softmax([2,4]) = [0.119203, 0.880797]
        // out = [10.0, 18.807971]
        let q = [1.0, 2.0];
        let k = [1.0, 2.0];
        let v = [10.0, 20.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 1, true);
        assert!((out[0] - 10.0).abs() <= TOL);
        assert!((out[1] - 18.807971).abs() <= TOL);
    }

    #[test]
    fn non_causal_attention_reference() {
        // Same inputs, no mask. row0 softmax([1,2]) = [0.268941, 0.731059]
        // out0 = 10*0.268941 + 20*0.731059 = 17.310586
        let q = [1.0, 2.0];
        let k = [1.0, 2.0];
        let v = [10.0, 20.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 2, 1, false);
        assert!((out[0] - 17.310586).abs() <= TOL);
        assert!((out[1] - 18.807971).abs() <= TOL);
    }

    #[test]
    fn uniform_keys_average_values() {
        // Identical keys -> uniform weights -> output = mean of V rows.
        let q = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // 3 identical queries
        let k = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]; // 3 identical keys
        let v = [1.0, 0.0, 2.0, 0.0, 3.0, 0.0];
        let out = scaled_dot_product_attention(&q, &k, &v, 3, 2, false);
        // Only checking the last row (fully unmasked in causal too).
        assert!((out[4] - 2.0).abs() <= TOL);
        assert!(out[5].abs() <= TOL);
    }

    #[test]
    fn attention_one_matches_full_attention_last_row() {
        // The incremental path must agree with the full causal computation.
        let q_all = [0.3, -0.1, 0.7, 0.2, -0.5, 0.9];
        let k_all = [0.1, 0.4, -0.2, 0.8, 0.5, -0.3];
        let v_all = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let head_dim = 2;
        let full = scaled_dot_product_attention(&q_all, &k_all, &v_all, 3, head_dim, true);
        let one = attention_one(&q_all[4..6], &k_all, &v_all, head_dim);
        assert!((one[0] - full[4]).abs() <= TOL);
        assert!((one[1] - full[5]).abs() <= TOL);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        softmax(&mut x);
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() <= 1e-6);
        assert!(x[3] > x[2] && x[2] > x[1] && x[1] > x[0]);
    }
}
