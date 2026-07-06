pub mod bridge;
pub mod dequant;
pub mod matmul;
pub mod ops;

use crate::simd_strategy::SimdStrategy;

// NOTE: `SimdStrategy::detect()` is cached behind a OnceLock — calling it in
// these dispatchers costs one atomic load, not a CPUID probe.

pub fn dequant_q4_0(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { dequant::q4_0::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q4_0::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q4_0::scalar::run(data),
    }
}

pub fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { dequant::q8_0::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q8_0::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q8_0::scalar::run(data),
    }
}

pub fn dequant_f16(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { dequant::f16::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::f16::avx2::run(data) },
        SimdStrategy::Scalar => dequant::f16::scalar::run(data),
    }
}

pub fn dequant_bf16(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { dequant::bf16::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::bf16::avx2::run(data) },
        SimdStrategy::Scalar => dequant::bf16::scalar::run(data),
    }
}

pub fn dequant_q4_k(data: &[u8]) -> Result<Vec<f32>, glcore::GlError> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { dequant::q4_k::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q4_k::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q4_k::scalar::run(data),
    }
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run(a, b, c, m, k, n) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run(a, b, c, m, k, n) },
        SimdStrategy::Scalar => matmul::scalar::run(a, b, c, m, k, n),
    }
}

pub fn matmul_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run_t(a, b, c, m, k, n) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run_t(a, b, c, m, k, n) },
        SimdStrategy::Scalar => matmul::scalar::run_t(a, b, c, m, k, n),
    }
}

pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run_matvec(w, x, y, out_dim, in_dim) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run_matvec(w, x, y, out_dim, in_dim) },
        SimdStrategy::Scalar => matmul::scalar::run_matvec(w, x, y, out_dim, in_dim),
    }
}

pub fn fast_exp(x: f32) -> f32 {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { ops::fast_exp::avx512::run(x) },
        SimdStrategy::Avx2 => unsafe { ops::fast_exp::avx2::run(x) },
        SimdStrategy::Scalar => ops::fast_exp::scalar::run(x),
    }
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { ops::rms_norm::avx2::run(x, weight, eps) }, // Fallback to AVX2 if no AVX-512 specific
        SimdStrategy::Avx2 => unsafe { ops::rms_norm::avx2::run(x, weight, eps) },
        SimdStrategy::Scalar => ops::rms_norm::scalar::run(x, weight, eps),
    }
}

/// Allocation-free RMSNorm for the decode loop. `out.len() == x.len()`.
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    match SimdStrategy::detect() {
        // AVX-512 falls back to AVX2 — no AVX-512-specific rms_norm yet.
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
            ops::rms_norm::avx2::run_into(x, weight, eps, out)
        },
        SimdStrategy::Scalar => ops::rms_norm::scalar::run_into(x, weight, eps, out),
    }
}

/// Dot product dispatcher (single-threaded; the runner's hot path calls the
/// backend-specific kernels directly through `threading::par_matvec`).
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 => unsafe { matmul::avx512::dot_f32(a, b) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::dot_f32(a, b) },
        SimdStrategy::Scalar => matmul::scalar::dot_f32(a, b),
    }
}
