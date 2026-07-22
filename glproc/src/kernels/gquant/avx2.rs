//! AVX2 fast path for GQ4A dequant (Pridwen v5 §10.2). Must match the scalar
//! reference bit-for-bit — the encoding produces integer-valued codes in
//! `[0, 15]` and scales in `f32`, so the same float ops in the same order as
//! `super::scalar::run` give identical results (0 ULP).
//!
//! GQ4A packs nibbles **interleaved**, not split like GGML's Q4_0: byte `k`
//! holds output weights `2k` (low nibble) and `2k+1` (high nibble) — see
//! `super::scalar::run`'s `idx.is_multiple_of(2)` selector. This is the
//! opposite layout from `dequant::q4_0::avx2` (which packs elements
//! `[0..16)` in low nibbles and `[16..32)` in high nibbles), so the lane
//! shuffle differs: after widening 8 bytes to 8 low-nibble and 8
//! high-nibble `i32` lanes, the two vectors are interleaved back together
//! with `unpack{lo,hi}_ps` + `permute2f128_ps` before the store, instead of
//! being stored as two contiguous halves.

use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

use super::GQ4ABlock;

/// Interleave two 8-lane `f32` vectors `lo=[l0..l7]`, `hi=[h0..h7]` into
/// `(out_a, out_b) = ([l0,h0,l1,h1,l2,h2,l3,h3], [l4,h4,l5,h5,l6,h6,l7,h7])`.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2.
#[target_feature(enable = "avx2")]
unsafe fn interleave_ps(lo: __m256, hi: __m256) -> (__m256, __m256) {
    let unpacklo = _mm256_unpacklo_ps(lo, hi);
    let unpackhi = _mm256_unpackhi_ps(lo, hi);
    let out_a = _mm256_permute2f128_ps(unpacklo, unpackhi, 0x20);
    let out_b = _mm256_permute2f128_ps(unpacklo, unpackhi, 0x31);
    (out_a, out_b)
}

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (see
/// [`crate::simd_strategy::SimdStrategy::detect`]).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(block: &GQ4ABlock, out: &mut [f32; 256]) {
    let super_scale = f16_to_f32(block.super_scale);

    for blk in 0..GQ4ABlock::SUB_BLOCKS {
        let scale_i = super_scale * (block.scale_delta[blk] as f32 / 127.0);
        let scale_vec = _mm256_set1_ps(scale_i);
        let mask_0f = _mm256_set1_epi32(0x0f);
        let eight = _mm256_set1_epi32(8);

        let byte_base = blk * 16;
        let q_ptr = block.weights[byte_base..byte_base + 16].as_ptr();
        let out_base = blk * GQ4ABlock::SUB_BLOCK_WEIGHTS;
        let out_ptr = out.as_mut_ptr().add(out_base);

        // Two passes of 8 bytes (16 output weights each) cover the 16-byte
        // sub-block.
        for pass in 0..2 {
            let q8 = _mm_loadl_epi64(q_ptr.add(pass * 8) as *const __m128i);
            let widened = _mm256_cvtepu8_epi32(q8);

            let lo_codes = _mm256_and_si256(widened, mask_0f);
            let hi_codes = _mm256_and_si256(_mm256_srli_epi32(widened, 4), mask_0f);

            let lo_signed = _mm256_sub_epi32(lo_codes, eight);
            let hi_signed = _mm256_sub_epi32(hi_codes, eight);

            let lo_f = _mm256_mul_ps(_mm256_cvtepi32_ps(lo_signed), scale_vec);
            let hi_f = _mm256_mul_ps(_mm256_cvtepi32_ps(hi_signed), scale_vec);

            let (out_a, out_b) = interleave_ps(lo_f, hi_f);
            _mm256_storeu_ps(out_ptr.add(pass * 16), out_a);
            _mm256_storeu_ps(out_ptr.add(pass * 16 + 8), out_b);
        }
    }
}

/// Byte-stream form matching [`super::scalar::run_stream`]'s shape.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (see
/// [`crate::simd_strategy::SimdStrategy::detect`]).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run_stream(data: &[u8]) -> Vec<f32> {
    let n_blocks = data.len() / GQ4ABlock::BYTES;
    let mut out = vec![0.0f32; n_blocks * GQ4ABlock::WEIGHTS];
    for (bi, chunk) in data.chunks_exact(GQ4ABlock::BYTES).enumerate() {
        let block = GQ4ABlock::from_bytes(chunk).expect("chunks_exact guarantees exact BYTES length");
        let mut block_out = [0.0f32; 256];
        run(block, &mut block_out);
        out[bi * GQ4ABlock::WEIGHTS..(bi + 1) * GQ4ABlock::WEIGHTS].copy_from_slice(&block_out);
    }
    out
}
