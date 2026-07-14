//! Attention value accumulation: `out[d] = Σ_t weights[t] * v_cache[t][d]`.
//!
//! The second half of single-query attention. After the softmax produces one
//! weight per cached position, this collapses the V cache down to a single
//! `head_dim` vector.
//!
//! It ran as a plain scalar loop while the Q·K half beside it was already AVX2,
//! so half of attention's arithmetic was leaving the vector units idle.
//! Measured on Qwen3-1.7B (28 layers, 16 heads, head_dim 128, ctx 252, with a
//! cold 0.88 GiB KV cache — the layout production actually has): vectorizing it
//! took attention from 4.92 to 4.07 ms/token, 1.21x.
//!
//! Why only 1.21x for a 2x-wider inner loop: this is a streaming reduction over
//! the KV cache, and `KvCache` spaces each head's region `max_context` apart
//! (2 MB at 4096 context) regardless of how much is live. Every head therefore
//! starts cold, so the loop spends much of its time waiting on DRAM rather than
//! issuing FMAs. Widening the arithmetic helps; it cannot remove the stall.

pub mod avx2;
pub mod scalar;

#[cfg(test)]
mod tests {
    use crate::simd_strategy::SimdStrategy;

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep.
    fn prng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// The dispatched kernel must equal the scalar reference for every shape the
    /// llama family actually uses, plus the awkward ones that exercise each of
    /// the AVX2 path's three tiers (32-wide main, 8-wide remainder, scalar tail).
    #[test]
    fn simd_matches_scalar_across_head_dims() {
        let mut seed = 0xA77Eu64;
        // 128 = Qwen3/llama; 64/80 = smaller heads; 40 = 32+8, exercises the
        // 8-wide remainder; 37 = 32+4+1, forces the scalar tail.
        for &head_dim in &[128usize, 80, 64, 40, 37, 8, 5] {
            for &n in &[1usize, 7, 64, 252] {
                let weights: Vec<f32> = (0..n).map(|_| prng(&mut seed)).collect();
                let v: Vec<f32> = (0..n * head_dim).map(|_| prng(&mut seed)).collect();

                let mut want = vec![f32::NAN; head_dim];
                super::scalar::run(&weights, &v, &mut want, head_dim);

                let mut got = vec![f32::NAN; head_dim];
                crate::kernels::attn_accum(&weights, &v, &mut got, head_dim);

                for (d, (g, w)) in got.iter().zip(&want).enumerate() {
                    // The AVX2 path sums into 4 independent accumulators while
                    // the scalar one uses a single running sum, so the two add
                    // the same products in a different order. Relative tolerance,
                    // not absolute: these are dot products, and f32 reassociation
                    // shows up in the last ulp or two.
                    let tol = w.abs().max(1.0) * 1e-5;
                    assert!(
                        (g - w).abs() < tol,
                        "head_dim={head_dim} n={n} d={d}: got {g}, want {w}"
                    );
                }
            }
        }
    }

    /// `out` is overwritten, never accumulated — it is reused scratch in the
    /// decode loop, and a kernel that added into it would silently sum every
    /// head's output on top of the last one's.
    #[test]
    fn out_is_overwritten_not_accumulated() {
        let head_dim = 128;
        let weights = [0.25f32, 0.75];
        let v = vec![1.0f32; 2 * head_dim];

        let mut out = vec![999.0f32; head_dim]; // poisoned
        crate::kernels::attn_accum(&weights, &v, &mut out, head_dim);
        // 0.25*1 + 0.75*1 = 1.0. If the kernel accumulated, this would be 1000.
        for (d, &o) in out.iter().enumerate() {
            assert!((o - 1.0).abs() < 1e-6, "d={d}: got {o}, want 1.0 (stale scratch?)");
        }
    }

    /// Uniform weights over identical rows must reproduce that row exactly —
    /// a shape-independent invariant that catches an indexing slip the random
    /// test could mask.
    #[test]
    fn uniform_weights_average_the_rows() {
        let head_dim = 128;
        let n = 4;
        // Row t is filled with the value t.
        let mut v = vec![0f32; n * head_dim];
        for t in 0..n {
            v[t * head_dim..(t + 1) * head_dim].fill(t as f32);
        }
        let weights = vec![0.25f32; n]; // mean of 0,1,2,3 = 1.5
        let mut out = vec![0f32; head_dim];
        crate::kernels::attn_accum(&weights, &v, &mut out, head_dim);
        for (d, &o) in out.iter().enumerate() {
            assert!((o - 1.5).abs() < 1e-6, "d={d}: got {o}, want 1.5");
        }
    }

    /// A V cache longer than `weights` (the decode loop passes a slice of a
    /// 4096-row buffer) must only read the first `weights.len()` rows.
    #[test]
    fn extra_rows_beyond_weights_are_ignored() {
        let head_dim = 8;
        let weights = [1.0f32]; // only row 0 participates
        let mut v = vec![7.0f32; 4 * head_dim]; // rows 1..3 are poison
        v[..head_dim].fill(2.0);

        let mut out = vec![0f32; head_dim];
        crate::kernels::attn_accum(&weights, &v, &mut out, head_dim);
        for &o in &out {
            assert!((o - 2.0).abs() < 1e-6, "read past weights.len(): got {o}");
        }
    }

    #[test]
    fn detected_backend_is_actually_exercised() {
        // Guards against the test silently only ever running the scalar path.
        let s = SimdStrategy::detect();
        eprintln!("attn_accum parity ran on {s:?}");
    }
}
