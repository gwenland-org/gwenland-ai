//! AVX2 attention value accumulation, 8 lanes per YMM.

use std::arch::x86_64::*;

/// `out[d] = Σ_t weights[t] * v_cache[t * head_dim + d]`, overwriting `out`.
///
/// Structure: tile the `head_dim` axis 32 wide (4 YMM registers), and for each
/// tile sweep the whole `t` axis with the accumulators held in registers. The
/// obvious alternative — loop `t` outermost and reload/store `out` on every row
/// — round-trips the accumulator through memory `weights.len()` times. Here it
/// is written exactly once per tile.
///
/// Four accumulators per tile also give the FMA chains independent dependency
/// paths, so a Sunny Cove core can keep more than one FMA in flight per cycle
/// instead of serializing on a single accumulator's latency.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(weights: &[f32], v_cache: &[f32], out: &mut [f32], head_dim: usize) {
    debug_assert_eq!(out.len(), head_dim);
    debug_assert!(v_cache.len() >= weights.len() * head_dim);

    let vp = v_cache.as_ptr();
    let op = out.as_mut_ptr();

    let mut d = 0usize;

    // Main path: 32 floats (4 YMM) of `head_dim` at a time. head_dim is 128 on
    // Qwen3 (and 64/80/128 across the llama family), so this covers all of it.
    while d + 32 <= head_dim {
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();

        for (t, &w) in weights.iter().enumerate() {
            let wv = _mm256_set1_ps(w);
            // SAFETY: t < n and d + 32 <= head_dim, so every load is inside
            // v_cache's n * head_dim floats (asserted above).
            let row = vp.add(t * head_dim + d);
            a0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row), a0);
            a1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row.add(8)), a1);
            a2 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row.add(16)), a2);
            a3 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(row.add(24)), a3);
        }

        // SAFETY: d + 32 <= head_dim == out.len().
        _mm256_storeu_ps(op.add(d), a0);
        _mm256_storeu_ps(op.add(d + 8), a1);
        _mm256_storeu_ps(op.add(d + 16), a2);
        _mm256_storeu_ps(op.add(d + 24), a3);
        d += 32;
    }

    // 8-wide remainder (head_dim not a multiple of 32).
    while d + 8 <= head_dim {
        let mut acc = _mm256_setzero_ps();
        for (t, &w) in weights.iter().enumerate() {
            // SAFETY: as above.
            let v = _mm256_loadu_ps(vp.add(t * head_dim + d));
            acc = _mm256_fmadd_ps(_mm256_set1_ps(w), v, acc);
        }
        // SAFETY: d + 8 <= head_dim == out.len().
        _mm256_storeu_ps(op.add(d), acc);
        d += 8;
    }

    // Scalar tail (head_dim % 8 != 0).
    for dd in d..head_dim {
        let mut acc = 0.0f32;
        for (t, &w) in weights.iter().enumerate() {
            acc += w * *vp.add(t * head_dim + dd);
        }
        *op.add(dd) = acc;
    }
}
