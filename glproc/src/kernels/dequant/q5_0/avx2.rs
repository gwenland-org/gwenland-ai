//! Q5_0 dequantization, AVX2+FMA. Validated against `scalar::dequant_block`.

use std::arch::x86_64::*;

use glcore::format::gguf::f16_to_f32;

use super::scalar::BLOCK_BYTES;

/// Dequantize up to 8 consecutive Q5_0 blocks (`n_blocks * 22` bytes) into
/// `output[..n_blocks * 32]`. Batching amortizes the call overhead: the
/// caller dots 256 weights at a time instead of 32.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA, `n_blocks <= 8`, and
/// `data.len() >= n_blocks * 22`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_blocks(data: &[u8], n_blocks: usize, output: &mut [f32; 256]) {
    debug_assert!(n_blocks <= 8);
    for b in 0..n_blocks {
        dequant_block(&data[b * BLOCK_BYTES..], &mut output[b * 32..b * 32 + 32]);
    }
}

/// Dequantize one 22-byte Q5_0 block into 32 f32 weights
/// (`output.len() >= 32`).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA and `data.len() >= 22`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_block(data: &[u8], output: &mut [f32]) {
    debug_assert!(output.len() >= 32);
    debug_assert!(data.len() >= BLOCK_BYTES);
    let d = _mm256_set1_ps(f16_to_f32(u16::from_le_bytes([data[0], data[1]])));
    let qh = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as i32;
    let qs = data.as_ptr().add(6);
    let out = output.as_mut_ptr();

    let low_mask = _mm_set1_epi8(0x0F);
    let qh_vec = _mm256_set1_epi32(qh);
    let one = _mm256_set1_epi32(1);
    let center = _mm256_set1_ps(-16.0);

    // 16 qs bytes: low nibbles are weights 0..16, high nibbles 16..32; bit i
    // of qh is weight i's 5th bit. Process 8 weights per iteration.
    let bytes = _mm_loadu_si128(qs as *const __m128i);
    let lo = _mm_and_si128(bytes, low_mask);
    let hi = _mm_and_si128(_mm_srli_epi16::<4>(bytes), low_mask);

    for g in 0..2 {
        // Weight lanes g*8..g*8+8 (low nibbles) and 16+g*8.. (high nibbles).
        let nib_lo = _mm256_cvtepu8_epi32(if g == 0 {
            lo
        } else {
            _mm_srli_si128::<8>(lo)
        });
        let nib_hi = _mm256_cvtepu8_epi32(if g == 0 {
            hi
        } else {
            _mm_srli_si128::<8>(hi)
        });

        // Per-lane variable shift extracts each weight's qh bit.
        let base_lo = g as i32 * 8;
        let shifts_lo = _mm256_setr_epi32(
            base_lo,
            base_lo + 1,
            base_lo + 2,
            base_lo + 3,
            base_lo + 4,
            base_lo + 5,
            base_lo + 6,
            base_lo + 7,
        );
        let shifts_hi = _mm256_add_epi32(shifts_lo, _mm256_set1_epi32(16));

        let bit_lo = _mm256_and_si256(_mm256_srlv_epi32(qh_vec, shifts_lo), one);
        let bit_hi = _mm256_and_si256(_mm256_srlv_epi32(qh_vec, shifts_hi), one);

        let q_lo = _mm256_or_si256(nib_lo, _mm256_slli_epi32::<4>(bit_lo));
        let q_hi = _mm256_or_si256(nib_hi, _mm256_slli_epi32::<4>(bit_hi));

        // w = d * (q - 16), the -16 folded into one FMA.
        let w_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q_lo), d, _mm256_mul_ps(d, center));
        let w_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(q_hi), d, _mm256_mul_ps(d, center));

        _mm256_storeu_ps(out.add(g * 8), w_lo);
        _mm256_storeu_ps(out.add(16 + g * 8), w_hi);
    }
}
