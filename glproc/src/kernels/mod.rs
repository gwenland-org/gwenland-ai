pub mod bridge;
pub mod dequant;
pub mod gquant;
pub mod matmul;
pub mod ops;
pub mod qdot;

use crate::simd_strategy::SimdStrategy;

// NOTE: `SimdStrategy::detect()` is cached behind a OnceLock — calling it in
// these dispatchers costs one atomic load, not a CPUID probe.
//
// ── The SAFETY contract shared by every dispatcher below ──
// Each `unsafe` call in this file is a `#[target_feature]` kernel, and the
// `match` arm *is* the feature proof: `detect()` yields `Avx512` only where
// AVX-512F probed present and `Avx2` only where AVX2+FMA did (see
// `simd_strategy.rs`, which is policy as well as detection — it declines
// AVX-512 on some CPUs that report it). Reaching a kernel any other way, or
// calling one directly, voids this and is UB on the wrong CPU.
//
// Where an arm reads `Avx512 | Avx2 => …avx2::…`, the fallback is sound for a
// second reason worth stating: an AVX-512F CPU is a strict superset, so the
// 256-bit kernel's feature requirement is satisfied under both variants.
//
// Per-dispatcher comments below add only what is *not* covered by the above —
// the memory precondition specific to that kernel.

pub fn dequant_q4_0(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). `run`
        // walks `data` as whole 18-byte Q4_0 blocks with `chunks_exact`, sizing its
        // output from that same count, so a short or ragged buffer truncates instead
        // of reading past the end — no length precondition on the caller.
        SimdStrategy::Avx512 => unsafe { dequant::q4_0::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q4_0::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q4_0::scalar::run(data),
    }
}

pub fn dequant_q8_0(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). Same
        // truncating `chunks_exact` shape as `dequant_q4_0`, over 34-byte Q8_0 blocks.
        SimdStrategy::Avx512 => unsafe { dequant::q8_0::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q8_0::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q8_0::scalar::run(data),
    }
}

pub fn dequant_f16(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). f16 is
        // element-wise over 2-byte lanes with a scalar tail, so any `data` length is
        // in bounds.
        SimdStrategy::Avx512 => unsafe { dequant::f16::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::f16::avx2::run(data) },
        SimdStrategy::Scalar => dequant::f16::scalar::run(data),
    }
}

pub fn dequant_bf16(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). bf16 is
        // element-wise over 2-byte lanes with a scalar tail, so any `data` length is
        // in bounds.
        SimdStrategy::Avx512 => unsafe { dequant::bf16::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::bf16::avx2::run(data) },
        SimdStrategy::Scalar => dequant::bf16::scalar::run(data),
    }
}

pub fn dequant_q4_k(data: &[u8]) -> Result<Vec<f32>, glcore::GlError> {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). Q4_K
        // iterates whole `BLOCK_BYTES` superblocks with `chunks_exact`; a length that
        // is not a whole multiple is rejected as a `GlError` by the kernel rather than
        // partially decoded, so malformed tensor bytes cannot walk off the end.
        SimdStrategy::Avx512 => unsafe { dequant::q4_k::avx512::run(data) },
        SimdStrategy::Avx2 => unsafe { dequant::q4_k::avx2::run(data) },
        SimdStrategy::Scalar => dequant::q4_k::scalar::run(data),
    }
}

pub fn matmul(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). Every
        // `a`/`b`/`c` access inside `run` goes through a bounds-checked slice
        // (`a[i*k..]`, `b[p*n..]`, `c[i*n..]`) before any pointer arithmetic, so
        // dimensions that disagree with the buffers panic rather than corrupt memory.
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run(a, b, c, m, k, n) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run(a, b, c, m, k, n) },
        SimdStrategy::Scalar => matmul::scalar::run(a, b, c, m, k, n),
    }
}

pub fn matmul_t(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above).
        // `run_t` slices `a` and `c` per row (bounds-checked, panics on mismatch) and
        // hands `run_matvec` a row of length exactly `k` as its `in_dim`, so it
        // discharges that kernel's unchecked read precondition internally — unlike
        // [`matvec`] below, this entry point places no extra obligation on the caller.
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run_t(a, b, c, m, k, n) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run_t(a, b, c, m, k, n) },
        SimdStrategy::Scalar => matmul::scalar::run_t(a, b, c, m, k, n),
    }
}

pub fn matvec(w: &[f32], x: &[f32], y: &mut [f32], out_dim: usize, in_dim: usize) {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above).
        // The SIMD `run_matvec`s walk `x` by raw pointer to `in_dim`; each now
        // opens with a real `assert!(x.len() >= in_dim)`, so a short `x` is a
        // panic rather than the out-of-bounds read it used to be, and no
        // length obligation is left on this dispatcher's callers. (`w` is
        // sliced per output row and `y[o]` is indexed — already checked. The
        // scalar path indexes `x[p]` and was never affected.)
        SimdStrategy::Avx512 => unsafe { matmul::avx512::run_matvec(w, x, y, out_dim, in_dim) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::run_matvec(w, x, y, out_dim, in_dim) },
        SimdStrategy::Scalar => matmul::scalar::run_matvec(w, x, y, out_dim, in_dim),
    }
}

pub fn fast_exp(x: f32) -> f32 {
    match SimdStrategy::detect() {
        // SAFETY: dispatch arm proves the ISA (see the module contract above). Scalar
        // in, scalar out — the kernel touches no caller memory, so the ISA proof is
        // the whole obligation.
        SimdStrategy::Avx512 => unsafe { ops::fast_exp::avx512::run(x) },
        SimdStrategy::Avx2 => unsafe { ops::fast_exp::avx2::run(x) },
        SimdStrategy::Scalar => ops::fast_exp::scalar::run(x),
    }
}

pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    match SimdStrategy::detect() {
        // SAFETY: both SIMD arms call the *AVX2* kernel, which the module contract
        // covers for `Avx512` too (superset). `run` allocates its own output at
        // `x.len()` and delegates to `run_into`, which reads `x` and `weight` in
        // 8-wide steps up to `x.len()`, so **the caller must guarantee
        // `weight.len() >= x.len()`** — the only check is a `debug_assert`, absent
        // from release builds.
        SimdStrategy::Avx512 => unsafe { ops::rms_norm::avx2::run(x, weight, eps) }, // Fallback to AVX2 if no AVX-512 specific
        SimdStrategy::Avx2 => unsafe { ops::rms_norm::avx2::run(x, weight, eps) },
        SimdStrategy::Scalar => ops::rms_norm::scalar::run(x, weight, eps),
    }
}

/// Fused SwiGLU gating for the decode loop: `gate[i] = silu(gate[i]) * up[i]`.
pub fn silu_mul(gate: &mut [f32], up: &[f32]) {
    match SimdStrategy::detect() {
        // AVX-512 falls back to AVX2 — no AVX-512-specific silu yet.
        // SAFETY: the arm proves AVX2+FMA under both variants (module contract
        // above). The kernel strides `gate` and `up` together to `gate.len()`,
        // so **the caller must guarantee `up.len() >= gate.len()`** — enforced
        // only by a `debug_assert`, which release builds drop.
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { ops::silu::avx2::run(gate, up) },
        SimdStrategy::Scalar => ops::silu::scalar::run(gate, up),
    }
}

/// Numerically stable in-place softmax over one score row.
///
/// Was the last scalar holdout in attention: the Q·K dots and the V
/// accumulation on either side of it were already vectorized, while this
/// called the scalar `fast_exp` once per cached position — measured at 17% of
/// the whole attention bucket (phase-split, ctx 252, cold rotate).
pub fn softmax_inplace(x: &mut [f32]) {
    match SimdStrategy::detect() {
        // AVX-512 falls back to AVX2 — no AVX-512-specific softmax yet.
        // SAFETY: the arm proves AVX2+FMA under both variants (module contract
        // above). The kernel is in-place over a single slice and derives every
        // bound from `x.len()` itself, with a scalar tail for the remainder —
        // no length relationship for the caller to uphold.
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { ops::softmax::avx2::run(x) },
        SimdStrategy::Scalar => ops::softmax::scalar::run(x),
    }
}

/// Attention value accumulation: `out[d] = Σ_t weights[t] * v_cache[t][d]`.
///
/// The second half of single-query attention, collapsing the V cache to one
/// `head_dim` vector. Was a scalar loop while the Q·K half beside it ran AVX2 —
/// see [`ops::attn_accum`] for the measurement.
pub fn attn_accum(weights: &[f32], v_cache: &[f32], out: &mut [f32], head_dim: usize) {
    match SimdStrategy::detect() {
        // AVX-512 falls back to AVX2 — no AVX-512-specific kernel yet, and on
        // the parts where AVX-512 is selected the 256-bit path is not the
        // bottleneck here (this loop stalls on DRAM, not on issue width).
        // SAFETY: the arm proves AVX2+FMA under both variants (module contract
        // above). The kernel walks `v_cache` as `weights.len()` consecutive
        // `head_dim`-sized rows and writes `head_dim` floats to `out`, so
        // **the caller must guarantee `out.len() == head_dim` and
        // `v_cache.len() >= weights.len() * head_dim`** — both are
        // `debug_assert`s only, so release builds do not catch a violation.
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe {
            ops::attn_accum::avx2::run(weights, v_cache, out, head_dim)
        },
        SimdStrategy::Scalar => ops::attn_accum::scalar::run(weights, v_cache, out, head_dim),
    }
}

/// Allocation-free RMSNorm for the decode loop. `out.len() == x.len()`.
pub fn rms_norm_into(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    match SimdStrategy::detect() {
        // AVX-512 falls back to AVX2 — no AVX-512-specific rms_norm yet.
        // SAFETY: the arm proves AVX2+FMA under both variants (module contract
        // above). Same read pattern as [`rms_norm`], plus a write of `x.len()`
        // floats into `out`, so **the caller must guarantee
        // `out.len() == x.len()` and `weight.len() >= x.len()`** — both are
        // `debug_assert`s, dropped in release.
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
        // SAFETY: dispatch arm proves the ISA (see the module contract above). Both
        // kernels stride `a` and `b` in lockstep to `a.len()` through raw pointers, so
        // **the caller must guarantee `b.len() >= a.len()`** — the equality is only a
        // `debug_assert`, so a short `b` reads out of bounds in release.
        SimdStrategy::Avx512 => unsafe { matmul::avx512::dot_f32(a, b) },
        SimdStrategy::Avx2 => unsafe { matmul::avx2::dot_f32(a, b) },
        SimdStrategy::Scalar => matmul::scalar::dot_f32(a, b),
    }
}
