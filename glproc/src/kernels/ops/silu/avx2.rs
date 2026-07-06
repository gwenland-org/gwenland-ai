//! AVX2 fused SwiGLU gating, 8 lanes per iteration.

use std::arch::x86_64::*;

/// `gate[i] = silu(gate[i]) * up[i]`, in place, using the vector `fast_exp`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(gate: &mut [f32], up: &[f32]) {
    debug_assert_eq!(gate.len(), up.len());
    let n = gate.len();
    let one = _mm256_set1_ps(1.0);
    let neg = _mm256_set1_ps(-0.0); // sign-flip mask

    let mut i = 0;
    while i + 8 <= n {
        // SAFETY: i + 8 <= n bounds both loads and the store.
        let g = _mm256_loadu_ps(gate.as_ptr().add(i));
        let u = _mm256_loadu_ps(up.as_ptr().add(i));
        let e = crate::kernels::ops::fast_exp::avx2::run_vec(_mm256_xor_ps(g, neg));
        let s = _mm256_div_ps(_mm256_mul_ps(g, u), _mm256_add_ps(one, e));
        _mm256_storeu_ps(gate.as_mut_ptr().add(i), s);
        i += 8;
    }
    // Scalar tail (hidden_dim % 8 != 0 models).
    for j in i..n {
        let x = gate[j];
        gate[j] = x / (1.0 + crate::kernels::ops::fast_exp::avx2::run(-x)) * up[j];
    }
}
