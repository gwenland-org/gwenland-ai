use std::arch::x86_64::*;

#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(x: &[f32], w: &[f32], eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    run_into(x, w, eps, &mut out);
    out
}

/// Allocation-free variant for the decode loop: writes into `out`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA, and
/// `out.len() == x.len() == w.len()`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run_into(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    debug_assert_eq!(out.len(), x.len());
    let n = x.len();
    let mut sum_sq = _mm256_setzero_ps();
    
    let mut i = 0;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(x.as_ptr().add(i));
        sum_sq = _mm256_fmadd_ps(v, v, sum_sq);
        i += 8;
    }
    
    let mut tmp = [0.0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), sum_sq);
    let mut mean_sq = tmp.iter().sum::<f32>();
    
    while i < n {
        let v = x[i];
        mean_sq += v * v;
        i += 1;
    }
    
    mean_sq /= n.max(1) as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    let inv_vec = _mm256_set1_ps(inv);
    
    let mut j = 0;
    while j + 8 <= n {
        let vx = _mm256_loadu_ps(x.as_ptr().add(j));
        let vw = _mm256_loadu_ps(w.as_ptr().add(j));
        let scaled = _mm256_mul_ps(vx, inv_vec);
        let res = _mm256_mul_ps(scaled, vw);
        _mm256_storeu_ps(out.as_mut_ptr().add(j), res);
        j += 8;
    }

    while j < n {
        out[j] = x[j] * inv * w[j];
        j += 1;
    }
}
