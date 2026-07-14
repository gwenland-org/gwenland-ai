//! In-place softmax over one score row.
//!
//! The last scalar holdout in attention. After the Q·K dots and the V
//! accumulation were vectorized, the softmax between them still called the
//! *scalar* `fast_exp` once per cached position — at ctx 252 on Qwen3-1.7B
//! that is 252 × 16 heads × 28 layers ≈ 113k scalar exp calls per decoded
//! token. Measured by phase-splitting the attention bucket (cold rotate over
//! all 28 layers, ctx 252): qk dots 41%, **softmax 17%**, v-accum 42%.
//!
//! The AVX2 path does max, exp (via the vector `fast_exp`) and the normalize
//! sweep 8 lanes at a time. Semantics match the scalar path exactly, including
//! the degenerate all-`-inf` row (a fully masked row spreads uniformly instead
//! of dividing by zero).

pub mod avx2;
pub mod scalar;

#[cfg(test)]
mod tests {
    /// Deterministic pseudo-random f32. No rand dep.
    fn prng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0
    }

    #[test]
    fn avx2_matches_scalar_across_lengths() {
        if !matches!(
            crate::simd_strategy::SimdStrategy::detect(),
            crate::simd_strategy::SimdStrategy::Avx2 | crate::simd_strategy::SimdStrategy::Avx512
        ) {
            eprintln!("skipping: no wide backend");
            return;
        }
        let mut seed = 0x50F7u64;
        // 1 = single element; 7 = pure tail; 8 = exactly one lane; 252 = the
        // real decode ctx; 255/256/257 straddle the lane boundary.
        for &n in &[1usize, 7, 8, 9, 64, 252, 255, 256, 257] {
            let src: Vec<f32> = (0..n).map(|_| prng(&mut seed)).collect();

            let mut want = src.clone();
            super::scalar::run(&mut want);
            let mut got = src.clone();
            // SAFETY: wide backend confirmed above.
            unsafe { super::avx2::run(&mut got) };

            let mut sum = 0f32;
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                // Same max-shift, same fast_exp polynomial (scalar and vector
                // share coefficients) — only summation order differs, plus the
                // scalar path's serial sum. Tight relative tolerance.
                let tol = w.abs().max(1e-6) * 1e-4;
                assert!((g - w).abs() < tol, "n={n} i={i}: got {g}, want {w}");
                sum += g;
            }
            assert!((sum - 1.0).abs() < 1e-4, "n={n}: probabilities sum to {sum}");
        }
    }

    /// A fully masked row (all -inf) must spread uniformly, not divide by zero
    /// — the scalar path's documented degenerate case, preserved bit-for-bit.
    #[test]
    fn all_neg_inf_spreads_uniformly() {
        if !matches!(
            crate::simd_strategy::SimdStrategy::detect(),
            crate::simd_strategy::SimdStrategy::Avx2 | crate::simd_strategy::SimdStrategy::Avx512
        ) {
            return;
        }
        for &n in &[3usize, 8, 20] {
            let mut x = vec![f32::NEG_INFINITY; n];
            // SAFETY: wide backend confirmed above.
            unsafe { super::avx2::run(&mut x) };
            for (i, &v) in x.iter().enumerate() {
                assert!(
                    (v - 1.0 / n as f32).abs() < 1e-6,
                    "n={n} i={i}: got {v}, want uniform {}",
                    1.0 / n as f32
                );
            }
        }
    }

    /// One dominant logit must take essentially all the mass — the shape a
    /// confident attention row actually has, and the case where an exp
    /// overflow bug would show up as NaN.
    #[test]
    fn dominant_score_takes_the_mass() {
        if !matches!(
            crate::simd_strategy::SimdStrategy::detect(),
            crate::simd_strategy::SimdStrategy::Avx2 | crate::simd_strategy::SimdStrategy::Avx512
        ) {
            return;
        }
        let mut x = vec![-2.0f32; 100];
        x[37] = 60.0; // large enough to overflow a shift-less exp
        // SAFETY: wide backend confirmed above.
        unsafe { super::avx2::run(&mut x) };
        assert!(x.iter().all(|v| v.is_finite()), "overflowed to NaN/inf");
        assert!(x[37] > 0.999, "dominant logit got {}", x[37]);
    }
}
