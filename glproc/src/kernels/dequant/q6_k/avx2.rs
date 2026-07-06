//! Q6_K dequantization, AVX2+FMA. Validated against `scalar::dequant_block`.

use std::arch::x86_64::*;

use glcore::format::gguf::f16_to_f32;

use super::scalar::BLOCK_BYTES;

/// Dequantize one 210-byte Q6_K super-block into 256 f32 weights.
///
/// Vectorization follows the GGML layout directly: within each 128-weight
/// half, 8 consecutive `ql`/`qh` bytes yield 4 groups of 8 weights that sit
/// 32 apart in the output (`l`, `l+32`, `l+64`, `l+96`).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA and `data.len() >= 210`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_block(data: &[u8], output: &mut [f32; 256]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([data[208], data[209]]));

    let low_mask = _mm256_set1_epi32(0x0F);
    let two_mask = _mm256_set1_epi32(0x03);
    let center = _mm256_set1_epi32(32);

    for half in 0..2 {
        let ql = data.as_ptr().add(half * 64);
        let qh = data.as_ptr().add(128 + half * 32);
        let sc = &data[192 + half * 8..192 + half * 8 + 8];
        let out = output.as_mut_ptr().add(half * 128);

        for l in (0..32).step_by(8) {
            // 8 weights per group; the sub-block scale is constant within
            // each 16-weight run, and l is a multiple of 8, so l/16 is
            // constant across the group.
            let is = l / 16;
            let d1 = _mm256_set1_ps(d * (sc[is] as i8 as f32));
            let d2 = _mm256_set1_ps(d * (sc[is + 2] as i8 as f32));
            let d3 = _mm256_set1_ps(d * (sc[is + 4] as i8 as f32));
            let d4 = _mm256_set1_ps(d * (sc[is + 6] as i8 as f32));

            // SAFETY: l + 8 <= 32, so ql reads stay within the 64-byte half
            // and qh reads within its 32-byte half.
            let ql_lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(l) as *const __m128i));
            let ql_hi = _mm256_cvtepu8_epi32(_mm_loadl_epi64(ql.add(l + 32) as *const __m128i));
            let qh_v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(qh.add(l) as *const __m128i));

            // q = (4-bit low part) | (2-bit high part << 4), then center -32.
            let q1 = _mm256_sub_epi32(
                _mm256_or_si256(
                    _mm256_and_si256(ql_lo, low_mask),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(qh_v, two_mask)),
                ),
                center,
            );
            let q2 = _mm256_sub_epi32(
                _mm256_or_si256(
                    _mm256_and_si256(ql_hi, low_mask),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<2>(qh_v), two_mask)),
                ),
                center,
            );
            let q3 = _mm256_sub_epi32(
                _mm256_or_si256(
                    _mm256_srli_epi32::<4>(ql_lo),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<4>(qh_v), two_mask)),
                ),
                center,
            );
            let q4 = _mm256_sub_epi32(
                _mm256_or_si256(
                    _mm256_srli_epi32::<4>(ql_hi),
                    _mm256_slli_epi32::<4>(_mm256_and_si256(_mm256_srli_epi32::<6>(qh_v), two_mask)),
                ),
                center,
            );

            _mm256_storeu_ps(out.add(l), _mm256_mul_ps(_mm256_cvtepi32_ps(q1), d1));
            _mm256_storeu_ps(out.add(l + 32), _mm256_mul_ps(_mm256_cvtepi32_ps(q2), d2));
            _mm256_storeu_ps(out.add(l + 64), _mm256_mul_ps(_mm256_cvtepi32_ps(q3), d3));
            _mm256_storeu_ps(out.add(l + 96), _mm256_mul_ps(_mm256_cvtepi32_ps(q4), d4));
        }
    }
}
