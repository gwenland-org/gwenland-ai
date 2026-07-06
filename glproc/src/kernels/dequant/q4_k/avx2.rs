//! Q4_K dequantization, AVX2+FMA. Validated against `scalar::dequant_block`.

use std::arch::x86_64::*;

use glcore::format::gguf::f16_to_f32;
use glcore::GlError;

use super::scalar::{scale_min, BLOCK_BYTES, BLOCK_NUMEL};

/// Dequantize one 144-byte Q4_K super-block into 256 f32 weights.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA
/// (`SimdStrategy::detect() == Avx2` or better) and `data.len() >= 144`.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dequant_block(data: &[u8], output: &mut [f32; 256]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([data[2], data[3]]));
    let scales = &data[4..16];
    let qs = data.as_ptr().add(16);
    let out = output.as_mut_ptr();

    let low_mask = _mm_set1_epi8(0x0F);

    for chunk in 0..4 {
        let (sc1, m1) = scale_min(2 * chunk, scales);
        let (sc2, m2) = scale_min(2 * chunk + 1, scales);
        let d1 = _mm256_set1_ps(d * sc1 as f32);
        // Negated min so the whole affine step is one FMA: d*q + (-min).
        let nmin1 = _mm256_set1_ps(-(dmin * m1 as f32));
        let d2 = _mm256_set1_ps(d * sc2 as f32);
        let nmin2 = _mm256_set1_ps(-(dmin * m2 as f32));

        let q = qs.add(chunk * 32);
        let dst = out.add(chunk * 64);

        // 32 qs bytes → 4 groups of 8; each group yields 8 low-nibble weights
        // (this sub-block) and 8 high-nibble weights (the next sub-block).
        for g in 0..4 {
            // SAFETY: q + g*8 + 8 <= qs + 128, inside the 144-byte block.
            let bytes = _mm_loadl_epi64(q.add(g * 8) as *const __m128i);
            let lo = _mm_and_si128(bytes, low_mask);
            let hi = _mm_and_si128(_mm_srli_epi16::<4>(bytes), low_mask);

            let lo_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(lo));
            let hi_f = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(hi));

            _mm256_storeu_ps(dst.add(g * 8), _mm256_fmadd_ps(d1, lo_f, nmin1));
            _mm256_storeu_ps(dst.add(32 + g * 8), _mm256_fmadd_ps(d2, hi_f, nmin2));
        }
    }
}

/// Dequantize a whole Q4_K tensor. Load-time path only (allocates).
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA.
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(GlError::Parse(format!(
            "Q4_K data length {} is not a multiple of {BLOCK_BYTES}",
            data.len()
        )));
    }
    let n_blocks = data.len() / BLOCK_BYTES;
    let mut out = vec![0f32; n_blocks * BLOCK_NUMEL];
    let mut buf = [0f32; BLOCK_NUMEL];
    for (i, block) in data.chunks_exact(BLOCK_BYTES).enumerate() {
        dequant_block(block, &mut buf);
        out[i * BLOCK_NUMEL..(i + 1) * BLOCK_NUMEL].copy_from_slice(&buf);
    }
    Ok(out)
}
