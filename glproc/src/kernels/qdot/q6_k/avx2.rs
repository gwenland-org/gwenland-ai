//! Q6_K × Q8 activation integer dot, AVX2. Validated against `scalar`.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// One Q6_K row (`n_blocks * 210` bytes) · quantized activation.
///
/// All unpacking stays in the byte domain, mirroring the GGML layout
/// directly: one 32-byte `ql` load feeds two weight groups (low/high
/// nibbles) and one `qh` load supplies the 2 high bits of all four groups.
/// The per-16 signed scales are applied inside `madd_epi16`, so each group
/// costs one `maddubs` + one `madd`; only the final per-group accumulate is
/// floating point. Byte-shifts use the 16-bit shift + mask idiom (the mask
/// discards the bits that bleed across byte boundaries).
///
/// Two accumulators, alternated per weight group, keep the 8 FMAs a block
/// issues off a single 4-cycle dependency chain.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA and F16C, and that `act`
/// was quantized for at least `row.len() / 210 * 256` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let m4 = _mm256_set1_epi8(0x0F);
    let m2 = _mm256_set1_epi8(0x03);

    let mut acc = [_mm256_setzero_ps(); 2];
    // Two scalar chains as well: 8 dependent FP subs per block would
    // otherwise serialize on the FP-add latency.
    let mut acc_off = [0f32; 2];

    for (jb, block) in row.chunks_exact(210).enumerate() {
        let d = f16_hw(u16::from_le_bytes([block[208], block[209]]));

        for half in 0..2 {
            // SAFETY: all loads stay inside the 210-byte block: ql spans
            // bytes 0..128, qh 128..192, scales 192..208.
            let ql = block.as_ptr().add(half * 64);
            let qh = block.as_ptr().add(128 + half * 32);
            let sc = &block[192 + half * 8..192 + half * 8 + 8];
            let base = jb * 256 + half * 128; // activation element index

            let ql_lo = _mm256_loadu_si256(ql as *const __m256i);
            let ql_hi = _mm256_loadu_si256(ql.add(32) as *const __m256i);
            let qh_v = _mm256_loadu_si256(qh as *const __m256i);

            // Unsigned 6-bit quants, one 32-byte vector per weight group.
            // The 2-bit masks are applied before the left shift, so nothing
            // bleeds across byte lanes inside the 16-bit shifts.
            let q = [
                _mm256_or_si256(
                    _mm256_and_si256(ql_lo, m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(qh_v, m2)),
                ),
                _mm256_or_si256(
                    _mm256_and_si256(ql_hi, m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<2>(qh_v), m2)),
                ),
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_lo), m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<4>(qh_v), m2)),
                ),
                _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi16::<4>(ql_hi), m4),
                    _mm256_slli_epi16::<4>(_mm256_and_si256(_mm256_srli_epi16::<6>(qh_v), m2)),
                ),
            ];

            for (g, &qg) in q.iter().enumerate() {
                let sc0 = sc[2 * g] as i8;
                let sc1 = sc[2 * g + 1] as i8;
                let d_a = act.scales[(base + g * 32) / 32];

                // SAFETY: act.q holds base + g*32 + 32 int8 values per the
                // function contract.
                let a = _mm256_loadu_si256(act.q.as_ptr().add(base + g * 32) as *const __m256i);
                // q unsigned (≤63) × a signed: |63·127·2| fits i16.
                let p16 = _mm256_maddubs_epi16(qg, a);
                // i16 lanes 0..7 cover weights 0..15 (scale sc0), lanes
                // 8..15 cover 16..31 (sc1); madd applies them exactly.
                let scv = _mm256_set_m128i(_mm_set1_epi16(sc1 as i16), _mm_set1_epi16(sc0 as i16));
                let p32 = _mm256_madd_epi16(p16, scv);

                acc[g & 1] =
                    _mm256_fmadd_ps(_mm256_set1_ps(d * d_a), _mm256_cvtepi32_ps(p32), acc[g & 1]);

                // The −32 offset per sub-block, settled with the per-16
                // activation sums: Σ d·d_a·sc·(q−32)·a needs −32·sc·Σa.
                let s = (base + g * 32) / 16;
                acc_off[g & 1] -= 32.0
                    * d
                    * d_a
                    * (sc0 as f32 * act.sums16[s] as f32 + sc1 as f32 * act.sums16[s + 1] as f32);
            }
        }
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
    tmp.iter().sum::<f32>() + acc_off[0] + acc_off[1]
}
