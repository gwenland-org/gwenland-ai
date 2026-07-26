//! PROBE ONLY — R output rows × ONE shared activation, R independent
//! accumulator chains, VNNI-256 width.
//!
//! Exists to test a different axis than `vnni512.rs`: that kernel widened a
//! single row's dot product (256→512-bit, still one row at a time, still
//! zero instruction-level parallelism across calls) and came up neutral in
//! production — see `gl-agent-skills/cpu-skills/rejected-optimizations.md`
//! entry 3's updated note. `architecture/percival/CPU/ARTX02-IceLake.md`
//! Finding F05 says llama.cpp's *actual* IceLake fast path is not a wider
//! per-row dot at all — it's a 16-row-tiled GEMM with 16 independent
//! accumulator chains **across output rows**, hiding `vpdpbusd`'s latency by
//! having many rows' instructions in flight at once, sharing one activation
//! load.
//!
//! This is the transpose of the existing `vnni::row_dot_xn` (which tiles `G`
//! *activations* against one row, reusing the weight load): here, `R` rows
//! are tiled against one *activation*, reusing the activation load. Decode
//! only ever has one activation (single token), so `row_dot_q8`'s current
//! single-row dispatch has essentially zero ILP — this kernel targets
//! exactly that gap, unlike the prefill-focused `row_dot_q8_packed8` path.
//!
//! Deliberately kept at VNNI-256 (not 512): the variable under test here is
//! "more independent chains", not "wider instruction" — conflating both
//! would leave an ambiguous result if the probe wins. Combine with 512-bit
//! width only as a follow-up, once row-tiling alone is shown to help.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// `R` Q8_0 rows (each `n_blocks * 34` bytes, same `n_blocks` for all) ·
/// one shared quantized activation. `R` independent accumulator chains are
/// interleaved in the block loop so `vpdpbusd`'s latency is hidden by rows
/// in flight, not by an unrolled single-row loop.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA, F16C, AVX512VL and
/// AVX512VNNI (same contract as [`super::vnni::row_dot`]), and that `act`
/// was quantized for at least `rows[0].len() / 34 * 32` elements. All `R`
/// rows must have equal length.
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "f16c",
    enable = "avx512vl",
    enable = "avx512vnni"
)]
pub unsafe fn row_tile_dot<const R: usize>(
    rows: [&[u8]; R],
    act: &QuantizedActivation,
) -> [f32; R] {
    let n_blocks = rows[0].len() / 34;
    let mut acc = [_mm256_setzero_ps(); R];

    for j in 0..n_blocks {
        // SAFETY: act.q holds at least (j+1)*32 int8 values per the
        // function contract; the QuantizedActivation quantize() call that
        // built it guarantees this for j < n_blocks.
        let a = _mm256_loadu_si256(act.q.as_ptr().add(j * 32) as *const __m256i);

        // R independent chains: each row's load + sign-prep + dpbusd is
        // issued before the previous row's result is consumed, so the
        // out-of-order window can overlap R in-flight `vpdpbusd`s instead
        // of serializing on one row's dependency chain.
        for r in 0..R {
            let block = &rows[r][j * 34..j * 34 + 34];
            // SAFETY: prefetch is a hint; past-the-end addresses are
            // harmless.
            _mm_prefetch::<_MM_HINT_T0>(block.as_ptr().add(544) as *const i8);
            let d = f16_hw(u16::from_le_bytes([block[0], block[1]])) * act.scales[j];

            // SAFETY: block has 34 bytes (2 header + 32 quants).
            let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
            let w_abs = _mm256_sign_epi8(w, w);
            let a_signed = _mm256_sign_epi8(a, w);
            let p32 = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_abs, a_signed);

            acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc[r]);
        }
    }

    let mut out = [0f32; R];
    let mut tmp = [0f32; 8];
    for r in 0..R {
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[r]);
        out[r] = tmp.iter().sum();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: &mut u64) -> u8 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u8
    }

    fn prng_f32(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn half_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 {
            return sign;
        }
        sign | ((exp as u16) << 10) | mant
    }

    fn q8_row(n_blocks: usize, seed: &mut u64) -> Vec<u8> {
        let mut v = Vec::with_capacity(n_blocks * 34);
        for _ in 0..n_blocks {
            v.extend_from_slice(&half_bits(0.02).to_le_bytes());
            for _ in 0..32 {
                v.push(prng(seed));
            }
        }
        v
    }

    /// Parity against `scalar::row_dot`, called once per row — the format's
    /// ground truth, per `ArchGLLM_X5.md`'s scalar-counterpart rule.
    #[test]
    fn matches_scalar_ground_truth() {
        if !(std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("avx512vl")
            && std::arch::is_x86_feature_detected!("avx512vnni"))
        {
            eprintln!("SKIP: AVX-512VNNI not available on this CPU");
            return;
        }

        const R: usize = 8;
        let mut seed = 0xB16710Eu64;
        for n_blocks in [1usize, 4, 28] {
            let rows: Vec<Vec<u8>> = (0..R).map(|_| q8_row(n_blocks, &mut seed)).collect();
            let row_refs: [&[u8]; R] = std::array::from_fn(|r| rows[r].as_slice());

            let x: Vec<f32> = (0..n_blocks * 32).map(|_| prng_f32(&mut seed)).collect();
            let mut act = super::super::super::QuantizedActivation::with_capacity(n_blocks * 32);
            act.quantize(&x);

            let expected: [f32; R] =
                std::array::from_fn(|r| super::super::scalar::row_dot(&rows[r], &act));
            // SAFETY: feature-checked above.
            let actual = unsafe { row_tile_dot::<R>(row_refs, &act) };

            for r in 0..R {
                let tol = (expected[r].abs() * 1e-3).max(1e-3);
                assert!(
                    (actual[r] - expected[r]).abs() <= tol,
                    "n_blocks={n_blocks} r={r}: tile={} scalar={} (tol {tol})",
                    actual[r],
                    expected[r]
                );
            }
        }
    }
}
