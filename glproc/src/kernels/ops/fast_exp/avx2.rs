use std::arch::x86_64::*;

/// Fast e^x via range reduction: e^x = 2^n * 2^f with n integer and
/// f in [-0.5, 0.5]; 2^n is built by bit-twiddling the exponent field and
/// 2^f by a degree-5 polynomial (coefficients from the 2^f Taylor series).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(x: f32) -> f32 {
    let y = x * std::f32::consts::LOG2_E;

    // clamp y to [-126.0, 126.0] to avoid underflow/overflow of 2^n
    let y = y.clamp(-126.0, 126.0);

    let n = y.round();
    let f = y - n; // fractional part in [-0.5, 0.5]

    // 2^n
    let exp_bits = ((n as i32) + 127) << 23;
    let two_n = f32::from_bits(exp_bits as u32);

    // 2^f approx: c_k = ln(2)^k / k!
    let c1 = std::f32::consts::LN_2;
    let c2 = 0.2402265f32;
    let c3 = 0.0555041f32;
    let c4 = 0.0096181f32;
    let c5 = 0.0013325f32;

    let mut p = c5;
    p = p * f + c4;
    p = p * f + c3;
    p = p * f + c2;
    p = p * f + c1;
    p = p * f + 1.0;
    
    two_n * p
}

/// Vectorized fast e^x for 8 lanes; same algorithm as [`run`].
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run_vec(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(std::f32::consts::LOG2_E);
    let y = _mm256_mul_ps(x, log2e);
    
    let clamp_min = _mm256_set1_ps(-126.0);
    let clamp_max = _mm256_set1_ps(126.0);
    let y = _mm256_max_ps(clamp_min, _mm256_min_ps(clamp_max, y));
    
    let n = _mm256_round_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm256_sub_ps(y, n);
    
    let n_i32 = _mm256_cvtps_epi32(n);
    let exp_bits = _mm256_slli_epi32(_mm256_add_epi32(n_i32, _mm256_set1_epi32(127)), 23);
    let two_n = _mm256_castsi256_ps(exp_bits);
    
    let c5 = _mm256_set1_ps(0.0013325);
    let c4 = _mm256_set1_ps(0.0096181);
    let c3 = _mm256_set1_ps(0.0555041);
    let c2 = _mm256_set1_ps(0.2402265);
    let c1 = _mm256_set1_ps(std::f32::consts::LN_2);
    let c0 = _mm256_set1_ps(1.0);
    
    let mut p = c5;
    p = _mm256_fmadd_ps(p, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    
    _mm256_mul_ps(two_n, p)
}
