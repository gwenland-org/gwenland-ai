use std::arch::x86_64::*;

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run(x: f32) -> f32 {
    crate::kernels::ops::fast_exp::avx2::run(x)
}

#[target_feature(enable = "avx512f", enable = "avx512bw")]
pub unsafe fn run_vec(x: __m512) -> __m512 {
    let log2e = _mm512_set1_ps(std::f32::consts::LOG2_E);
    let y = _mm512_mul_ps(x, log2e);
    
    let clamp_min = _mm512_set1_ps(-126.0);
    let clamp_max = _mm512_set1_ps(126.0);
    let y = _mm512_max_ps(clamp_min, _mm512_min_ps(clamp_max, y));
    
    let n = _mm512_roundscale_ps(y, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    let f = _mm512_sub_ps(y, n);
    
    let n_i32 = _mm512_cvtps_epi32(n);
    let exp_bits = _mm512_slli_epi32(_mm512_add_epi32(n_i32, _mm512_set1_epi32(127)), 23);
    let two_n = _mm512_castsi512_ps(exp_bits);
    
    let c5 = _mm512_set1_ps(0.0013325);
    let c4 = _mm512_set1_ps(0.0096181);
    let c3 = _mm512_set1_ps(0.0555041);
    let c2 = _mm512_set1_ps(0.2402265);
    let c1 = _mm512_set1_ps(std::f32::consts::LN_2);
    let c0 = _mm512_set1_ps(1.0);
    
    let mut p = c5;
    p = _mm512_fmadd_ps(p, f, c4);
    p = _mm512_fmadd_ps(p, f, c3);
    p = _mm512_fmadd_ps(p, f, c2);
    p = _mm512_fmadd_ps(p, f, c1);
    p = _mm512_fmadd_ps(p, f, c0);
    
    _mm512_mul_ps(two_n, p)
}
