//! AVX2 in-place softmax: max, exp and normalize, 8 lanes at a time.

use std::arch::x86_64::*;

/// Horizontal max of 8 f32 lanes.
#[target_feature(enable = "avx2")]
unsafe fn hmax(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let m = _mm_max_ps(lo, hi);
    let m = _mm_max_ps(m, _mm_movehl_ps(m, m));
    let m = _mm_max_ss(m, _mm_shuffle_ps::<0b01>(m, m));
    _mm_cvtss_f32(m)
}

/// Horizontal sum of 8 f32 lanes.
#[target_feature(enable = "avx2")]
unsafe fn hsum(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}

/// Numerically stable in-place softmax, vectorized.
///
/// Three sweeps, all 8-wide with scalar tails:
///   1. max         — the shift that keeps `exp` from overflowing
///   2. exp + sum   — via the vector `fast_exp` (same polynomial as scalar)
///   3. normalize   — multiply by `1/sum`
///
/// Degenerate all-`-inf` rows spread uniformly, matching the scalar reference.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(x: &mut [f32]) {
    let n = x.len();

    // --- pass 1: max ---
    let mut max = f32::NEG_INFINITY;
    let mut i = 0;
    if n >= 8 {
        // SAFETY: i + 8 <= n bounds every load.
        let mut vmax = _mm256_loadu_ps(x.as_ptr());
        i = 8;
        while i + 8 <= n {
            vmax = _mm256_max_ps(vmax, _mm256_loadu_ps(x.as_ptr().add(i)));
            i += 8;
        }
        max = hmax(vmax);
    }
    for &v in &x[i..] {
        max = max.max(v);
    }

    if !max.is_finite() {
        // All -inf (fully masked row) — degenerate; spread uniformly. Same
        // behavior as the scalar reference.
        let n = n as f32;
        x.fill(1.0 / n);
        return;
    }

    // --- pass 2: exp + sum ---
    let vm = _mm256_set1_ps(max);
    let mut vsum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: i + 8 <= n bounds the load and the store.
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        let e = crate::kernels::ops::fast_exp::avx2::run_vec(_mm256_sub_ps(v, vm));
        _mm256_storeu_ps(x.as_mut_ptr().add(i), e);
        vsum = _mm256_add_ps(vsum, e);
        i += 8;
    }
    let mut sum = hsum(vsum);
    for v in &mut x[i..] {
        *v = crate::kernels::ops::fast_exp::avx2::run(*v - max);
        sum += *v;
    }

    // --- pass 3: normalize ---
    if sum > 0.0 {
        let inv = 1.0 / sum;
        let vinv = _mm256_set1_ps(inv);
        let mut i = 0;
        while i + 8 <= n {
            // SAFETY: as above.
            let v = _mm256_loadu_ps(x.as_ptr().add(i));
            _mm256_storeu_ps(x.as_mut_ptr().add(i), _mm256_mul_ps(v, vinv));
            i += 8;
        }
        for v in &mut x[i..] {
            *v *= inv;
        }
    }
}
