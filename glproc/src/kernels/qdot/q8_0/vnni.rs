//! Q8_0 × Q8 activation integer dot using AVX512-VNNI at 256-bit width.
//!
//! `vpdpbusd` fuses the `maddubs` + `madd` pair into a single instruction
//! (unsigned × signed bytes, accumulated straight into i32 lanes). The EVEX
//! 256-bit form (AVX512VL + AVX512VNNI, Tiger Lake+) executes at the same
//! frequency license as AVX2 — it is not a 512-bit datapath, so the X5
//! AVX-512 thermal concern does not apply. Same accumulation order as the
//! AVX2 kernel, so results are bit-identical.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// One Q8_0 row (`n_blocks * 34` bytes) · quantized activation.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA, F16C, AVX512VL and
/// AVX512VNNI, and that `act` was quantized for at least
/// `row.len() / 34 * 32` elements.
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "f16c",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let mut acc = [_mm256_setzero_ps(); 2];

    for (j, block) in row.chunks_exact(34).enumerate() {
        // Same stream prefetch as the AVX2 kernel (measured +2 tok/s).
        // SAFETY: prefetch is a hint; past-the-end addresses are harmless.
        _mm_prefetch::<_MM_HINT_T0>(block.as_ptr().add(544) as *const i8);
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]])) * act.scales[j];

        // SAFETY: block has 34 bytes (2 header + 32 quants); act.q holds at
        // least (j+1)*32 int8 values per the function contract.
        let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
        let a = _mm256_loadu_si256(act.q.as_ptr().add(j * 32) as *const __m256i);

        let w_abs = _mm256_sign_epi8(w, w); // |w|, unsigned operand
        let a_signed = _mm256_sign_epi8(a, w); // a * sign(w)
        // One instruction where AVX2 needs maddubs + madd.
        let p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_abs, a_signed);

        acc[j & 1] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc[j & 1]);
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
    tmp.iter().sum()
}

/// One Q8_0 row · `G` Q8 activations at once — the batched-prefill inner
/// kernel. The weight block load, sign prep and f16 scale conversion are
/// paid once and reused for all `G` activations (row_dot repeats them per
/// activation), and the `G` independent accumulator chains hide the
/// FMA/`vpdpbusd` latency a single-activation dot serializes on. `G` of 8
/// uses 8 accumulator registers plus ~4 temporaries — still inside the 16
/// ymm registers, so no spills.
///
/// # Safety
/// Same CPU-feature contract as [`row_dot`]; every activation must be
/// quantized for at least `row.len() / 34 * 32` elements.
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "f16c",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
pub unsafe fn row_dot_xn<const G: usize>(
    row: &[u8],
    acts: [&QuantizedActivation; G],
) -> [f32; G] {
    let mut acc = [_mm256_setzero_ps(); G];

    for (j, block) in row.chunks_exact(34).enumerate() {
        // SAFETY: prefetch is a hint; past-the-end addresses are harmless.
        _mm_prefetch::<_MM_HINT_T0>(block.as_ptr().add(544) as *const i8);
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]]));

        // SAFETY: block has 34 bytes (2 header + 32 quants); each act.q
        // holds at least (j+1)*32 int8 values per the function contract.
        let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
        let w_abs = _mm256_sign_epi8(w, w);

        for g in 0..G {
            let a = _mm256_loadu_si256(acts[g].q.as_ptr().add(j * 32) as *const __m256i);
            let a_signed = _mm256_sign_epi8(a, w);
            let p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_abs, a_signed);
            acc[g] = _mm256_fmadd_ps(
                _mm256_set1_ps(d * acts[g].scales[j]),
                _mm256_cvtepi32_ps(p32),
                acc[g],
            );
        }
    }

    let mut out = [0f32; G];
    let mut tmp = [0f32; 8];
    for g in 0..G {
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[g]);
        out[g] = tmp.iter().sum();
    }
    out
}
