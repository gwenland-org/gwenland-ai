//! PROBE ONLY — Q8_0 × Q8 activation integer dot using genuine 512-bit-wide
//! AVX-512VNNI (`vpdpbusd` on zmm, two blocks per instruction).
//!
//! Not wired into `qdot::row_dot_q8`'s dispatch and not called from any
//! production path. Exists solely so `benches/vnni512_probe.rs` can measure
//! whether 512-bit width beats the 256-bit EVEX VNNI kernel
//! ([`super::vnni`]) on this machine — see `gl-agent-skills/cpu-skills/
//! rejected-optimizations.md` entry 3, which closed "at least use
//! AVX-512VNNI-512" for thermal/downclock reasons without a kernel-level
//! measurement to back it. This file is that measurement, run under
//! JinXSuper's explicit override to revisit the entry.
//!
//! AVX-512 has no `vpsignb` (the `_mm256_sign_epi8` this format's sign-trick
//! normally uses), so the signed-activation step is reconstructed from
//! primitives that do exist at 512-bit width: `_mm512_movepi8_mask` extracts
//! each weight byte's sign bit into a `__mmask64`, and `_mm512_mask_blend_epi8`
//! selects `a` or `-a` per lane. Same arithmetic as the 256-bit kernel's
//! `_mm256_sign_epi8(a, w)`, different instructions to get there.
//!
//! Two Q8_0 blocks (34 bytes each: 2-byte f16 scale + 32 int8 quants) feed
//! one 512-bit `vpdpbusd`: block quants are not memory-contiguous (each has
//! its own 2-byte header), so the two 32-byte quant spans are loaded as two
//! 256-bit halves and combined with `_mm512_inserti64x4`, not one `loadu512`.
//! The result's low/high 256-bit halves are each block's independent partial
//! dot (8 i32 lanes), scaled by that block's own f16 scale before summing —
//! `vpdpbusd` cannot itself apply two different float scales mid-instruction.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// One Q8_0 row · quantized activation, processing block-pairs at 512-bit
/// width. Falls back to the 256-bit kernel for a trailing odd block.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA, F16C, AVX512F, AVX512BW
/// and AVX512VNNI, and that `act` was quantized for at least
/// `row.len() / 34 * 32` elements.
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "f16c",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vnni"
)]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let n_blocks = row.len() / 34;
    let paired = n_blocks - (n_blocks % 2);

    // Two 256-bit accumulators (block-j-parity halves of the 512-bit
    // result), summed at the end — avoids fighting cast intrinsics to
    // re-merge two differently-scaled halves back into one zmm accumulator
    // every iteration, at zero cost (still one `vpdpbusd` per block-pair).
    let mut acc_lo = _mm256_setzero_ps();
    let mut acc_hi = _mm256_setzero_ps();
    let mut j = 0;
    while j < paired {
        let b0 = &row[j * 34..j * 34 + 34];
        let b1 = &row[(j + 1) * 34..(j + 1) * 34 + 34];

        // SAFETY: prefetch is a hint; past-the-end addresses are harmless.
        _mm_prefetch::<_MM_HINT_T0>(b1.as_ptr().add(544) as *const i8);

        let d0 = f16_hw(u16::from_le_bytes([b0[0], b0[1]])) * act.scales[j];
        let d1 = f16_hw(u16::from_le_bytes([b1[0], b1[1]])) * act.scales[j + 1];

        // SAFETY: each block has 34 bytes (2 header + 32 quants); act.q
        // holds at least (j+2)*32 int8 values per the function contract.
        let w0 = _mm256_loadu_si256(b0.as_ptr().add(2) as *const __m256i);
        let w1 = _mm256_loadu_si256(b1.as_ptr().add(2) as *const __m256i);
        let w = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);

        let a0 = _mm256_loadu_si256(act.q.as_ptr().add(j * 32) as *const __m256i);
        let a1 = _mm256_loadu_si256(act.q.as_ptr().add((j + 1) * 32) as *const __m256i);
        let a = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1);

        // |w| (unsigned operand) and a*sign(w) (signed operand) — see
        // module doc for why this replaces `_mm256_sign_epi8` at 512-bit.
        let w_abs = _mm512_abs_epi8(w);
        let neg_mask = _mm512_movepi8_mask(w);
        let neg_a = _mm512_sub_epi8(_mm512_setzero_si512(), a);
        let a_signed = _mm512_mask_blend_epi8(neg_mask, a, neg_a);

        // One instruction covering both blocks' 32-wide dot products.
        let p32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_abs, a_signed);

        // Low 256 bits = block j's 8 partial i32 sums, high 256 = block j+1's.
        let p32_lo = _mm512_extracti64x4_epi64(p32, 0);
        let p32_hi = _mm512_extracti64x4_epi64(p32, 1);
        acc_lo = _mm256_fmadd_ps(_mm256_set1_ps(d0), _mm256_cvtepi32_ps(p32_lo), acc_lo);
        acc_hi = _mm256_fmadd_ps(_mm256_set1_ps(d1), _mm256_cvtepi32_ps(p32_hi), acc_hi);

        j += 2;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc_lo, acc_hi));
    let mut sum: f32 = tmp.iter().sum();

    // Odd trailing block (n_blocks is odd): finish with a plain scalar dot,
    // no need for a third SIMD width just for one block.
    if paired < n_blocks {
        let b = &row[paired * 34..paired * 34 + 34];
        let d = f16_hw(u16::from_le_bytes([b[0], b[1]])) * act.scales[paired];
        let mut idot = 0i32;
        for (i, &wq) in b[2..34].iter().enumerate() {
            idot += (wq as i8) as i32 * act.q[paired * 32 + i] as i32;
        }
        sum += d * idot as f32;
    }

    sum
}

/// One Q8_0 row · a packed panel of 8 activations (quants `[block][act][32]`,
/// scales `[block][act]`, per [`super::vnni::row_dot_packed8`]'s layout) —
/// the batched-prefill inner kernel, at 512-bit width. Processes block-pairs
/// exactly as [`row_dot`], but shares each block-pair's weight load/sign-prep
/// across all 8 activations instead of doing it once per single dot. This is
/// prefill's actual hot path (`ffn_gate_up`/`ffn_down`, ~55-71% of measured
/// FFN time per `benchmarks/full-bottleneck-e2e.json`), so it — not
/// [`row_dot`] alone — is what a production A/B needs to exercise to be
/// representative of where the measured gap actually is.
///
/// # Safety
/// Same CPU-feature contract as [`row_dot`]. `pq` must hold
/// `row.len() / 34 * 8 * 32` quant bytes and `ps` `row.len() / 34 * 8` scales.
/// `row.len() / 34` must be even (no odd-block fallback here — the packed8
/// panel is only ever built for whole-row batched calls with even block
/// counts on every shape this probe exercises; unlike `row_dot`, silently
/// dropping the last block would be a correctness bug worth a hard
/// assertion instead of a hidden scalar tail).
#[target_feature(
    enable = "avx2",
    enable = "fma",
    enable = "f16c",
    enable = "avx512f",
    enable = "avx512bw",
    enable = "avx512vnni"
)]
pub unsafe fn row_dot_packed8(row: &[u8], pq: &[u8], ps: &[f32]) -> [f32; 8] {
    let n_blocks = row.len() / 34;
    debug_assert_eq!(n_blocks % 2, 0, "row_dot_packed8 (vnni512) requires an even block count");

    let mut acc_lo = [_mm256_setzero_ps(); 8];
    let mut acc_hi = [_mm256_setzero_ps(); 8];

    let mut j = 0;
    while j < n_blocks {
        let b0 = &row[j * 34..j * 34 + 34];
        let b1 = &row[(j + 1) * 34..(j + 1) * 34 + 34];

        // SAFETY: prefetch is a hint; past-the-end addresses are harmless.
        _mm_prefetch::<_MM_HINT_T0>(b1.as_ptr().add(544) as *const i8);

        let d0 = f16_hw(u16::from_le_bytes([b0[0], b0[1]]));
        let d1 = f16_hw(u16::from_le_bytes([b1[0], b1[1]]));

        // SAFETY: each block has 34 bytes; pq/ps hold 8 interleaved lanes
        // per block per the function contract.
        let w0 = _mm256_loadu_si256(b0.as_ptr().add(2) as *const __m256i);
        let w1 = _mm256_loadu_si256(b1.as_ptr().add(2) as *const __m256i);
        let w = _mm512_inserti64x4(_mm512_castsi256_si512(w0), w1, 1);
        let w_abs = _mm512_abs_epi8(w);
        let neg_mask = _mm512_movepi8_mask(w);

        let qbase0 = j * 8 * 32;
        let qbase1 = (j + 1) * 8 * 32;
        let sbase0 = j * 8;
        let sbase1 = (j + 1) * 8;

        for g in 0..8 {
            let a0 = _mm256_loadu_si256(pq.as_ptr().add(qbase0 + g * 32) as *const __m256i);
            let a1 = _mm256_loadu_si256(pq.as_ptr().add(qbase1 + g * 32) as *const __m256i);
            let a = _mm512_inserti64x4(_mm512_castsi256_si512(a0), a1, 1);

            let neg_a = _mm512_sub_epi8(_mm512_setzero_si512(), a);
            let a_signed = _mm512_mask_blend_epi8(neg_mask, a, neg_a);

            let p32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w_abs, a_signed);
            let p32_lo = _mm512_extracti64x4_epi64(p32, 0);
            let p32_hi = _mm512_extracti64x4_epi64(p32, 1);

            let sg0 = d0 * *ps.get_unchecked(sbase0 + g);
            let sg1 = d1 * *ps.get_unchecked(sbase1 + g);
            acc_lo[g] = _mm256_fmadd_ps(_mm256_set1_ps(sg0), _mm256_cvtepi32_ps(p32_lo), acc_lo[g]);
            acc_hi[g] = _mm256_fmadd_ps(_mm256_set1_ps(sg1), _mm256_cvtepi32_ps(p32_hi), acc_hi[g]);
        }

        j += 2;
    }

    let mut out = [0f32; 8];
    let mut tmp = [0f32; 8];
    for g in 0..8 {
        _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc_lo[g], acc_hi[g]));
        out[g] = tmp.iter().sum();
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

    /// `n_blocks` Q8_0 blocks (34 bytes each) of pseudo-random data.
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

    /// Parity across even AND odd block counts (exercises the scalar tail),
    /// against `scalar::row_dot` — the format's ground truth per
    /// `ArchGLLM_X5.md`'s "every SIMD function has a scalar counterpart"
    /// rule. Skips instead of failing if this CPU lacks AVX-512VNNI, so CI
    /// on non-AVX-512 hardware stays green.
    #[test]
    fn matches_scalar_ground_truth() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni"))
        {
            eprintln!("SKIP: AVX-512VNNI not available on this CPU");
            return;
        }

        let mut seed = 0xC0FFEEu64;
        for n_blocks in [1usize, 2, 3, 4, 7, 16, 48, 280] {
            let row = q8_row(n_blocks, &mut seed);
            let x: Vec<f32> = (0..n_blocks * 32).map(|_| prng_f32(&mut seed)).collect();
            let mut act = super::super::super::QuantizedActivation::with_capacity(n_blocks * 32);
            act.quantize(&x);

            let expected = super::super::scalar::row_dot(&row, &act);
            // SAFETY: feature-checked above.
            let actual = unsafe { row_dot(&row, &act) };

            let tol = (expected.abs() * 1e-3).max(1e-3);
            assert!(
                (actual - expected).abs() <= tol,
                "n_blocks={n_blocks}: vnni512={actual} scalar={expected} (tol {tol})"
            );
        }
    }

    /// Same ground-truth check for [`row_dot_packed8`], built by summing
    /// `scalar::row_dot` per activation against a de-interleaved panel — the
    /// panel layout itself has no independent "ground truth" function, so
    /// this reconstructs one from the already-trusted single-row scalar path.
    #[test]
    fn packed8_matches_scalar_ground_truth() {
        if !(std::arch::is_x86_feature_detected!("avx512f")
            && std::arch::is_x86_feature_detected!("avx512bw")
            && std::arch::is_x86_feature_detected!("avx512vnni"))
        {
            eprintln!("SKIP: AVX-512VNNI not available on this CPU");
            return;
        }

        let mut seed = 0xFACADEu64;
        for n_blocks in [2usize, 4, 16, 48] {
            let row = q8_row(n_blocks, &mut seed);

            // Build 8 independent activations, quantize each, then pack them
            // into the [block][act][32] / [block][act] panel layout.
            let mut acts = Vec::with_capacity(8);
            for _ in 0..8 {
                let x: Vec<f32> = (0..n_blocks * 32).map(|_| prng_f32(&mut seed)).collect();
                let mut act = super::super::super::QuantizedActivation::with_capacity(n_blocks * 32);
                act.quantize(&x);
                acts.push(act);
            }

            let mut expected = [0f32; 8];
            for (g, act) in acts.iter().enumerate() {
                expected[g] = super::super::scalar::row_dot(&row, act);
            }

            let mut pq = vec![0u8; n_blocks * 8 * 32];
            let mut ps = vec![0f32; n_blocks * 8];
            for j in 0..n_blocks {
                for (g, act) in acts.iter().enumerate() {
                    pq[j * 8 * 32 + g * 32..j * 8 * 32 + g * 32 + 32]
                        .copy_from_slice(&act.q[j * 32..j * 32 + 32].iter().map(|&v| v as u8).collect::<Vec<_>>());
                    ps[j * 8 + g] = act.scales[j];
                }
            }

            // SAFETY: feature-checked above; n_blocks is even in this test.
            let actual = unsafe { row_dot_packed8(&row, &pq, &ps) };

            for g in 0..8 {
                let tol = (expected[g].abs() * 1e-3).max(1e-3);
                assert!(
                    (actual[g] - expected[g]).abs() <= tol,
                    "n_blocks={n_blocks} g={g}: vnni512={} scalar={} (tol {tol})",
                    actual[g],
                    expected[g]
                );
            }
        }
    }
}
