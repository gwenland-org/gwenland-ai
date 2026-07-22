//! AVX2 fast path for GQ2A dequant (Pridwen v5 §10.2). Must match the scalar
//! reference bit-for-bit — the encoding produces integer-valued codes in
//! `[0, 3]` and scales in `f32`, so the same float ops in the same order as
//! `super::gq2a_scalar::run` give identical results (0 ULP).
//!
//! Unlike GQ4A (which packs nibbles **interleaved**, byte `k` = weights `2k`
//! and `2k+1`), GQ2A packs 4 codes per byte **sequentially**: byte `k` holds
//! weights `4k, 4k+1, 4k+2, 4k+3` in bits `[0:2), [2:4), [4:6), [6:8)` low to
//! high (Pridwen v5 §3.2's bit-packing addendum). This means no lane
//! interleave step is needed here — extracting 8 weights from 2 input bytes
//! already produces them in the correct output order once the 2-bit codes
//! are unpacked via a per-lane variable shift.

use std::arch::x86_64::*;
use glcore::format::gguf::f16_to_f32;

use super::GQ2ABlock;

/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (see
/// [`crate::simd_strategy::SimdStrategy::detect`]).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run(block: &GQ2ABlock, out: &mut [f32; 256]) {
    let super_scale = f16_to_f32(block.super_scale);
    let super_min = f16_to_f32(block.super_min);

    // Fixed per-lane shift amounts: lanes 0..4 extract a byte's 4 codes
    // (bits 0,2,4,6), lanes 4..8 repeat for the next byte.
    let shifts = _mm256_setr_epi32(0, 2, 4, 6, 0, 2, 4, 6);
    let mask_03 = _mm256_set1_epi32(0b11);

    for blk in 0..GQ2ABlock::SUB_BLOCKS {
        let scale_d = block.scale_delta_at(blk) as f32;
        let min_d = block.min_delta_at(blk) as f32;
        let scale_i = super_scale * (1.0 + scale_d / 7.0);
        let min_i = super_min + super_scale * (min_d / 7.0);
        let scale_vec = _mm256_set1_ps(scale_i);
        let min_vec = _mm256_set1_ps(min_i);
        // Deliberately a real division (_mm256_div_ps below), not a
        // precomputed-reciprocal multiply: `code * (1.0/3.0)` rounds twice
        // (once building the reciprocal, once multiplying) and is not
        // guaranteed bit-exact with the scalar reference's single `code /
        // 3.0` division. A real per-lane divide matches it operation-for-
        // operation.
        let three_vec = _mm256_set1_ps(3.0);

        let byte_base = blk * 4; // 4 bytes per 16-weight sub-block
        let out_base = blk * GQ2ABlock::SUB_BLOCK_WEIGHTS;
        let out_ptr = out.as_mut_ptr().add(out_base);

        // Two passes of 2 bytes (8 output weights each) cover the 4-byte
        // sub-block.
        for pass in 0..2 {
            let byte_a = block.weights[byte_base + pass * 2];
            let byte_b = block.weights[byte_base + pass * 2 + 1];
            // Stage [A,A,A,A,B,B,B,B] so cvtepu8_epi32 widens each repeated
            // byte into the lane that will extract one of its 4 codes.
            let staged: [u8; 8] = [byte_a, byte_a, byte_a, byte_a, byte_b, byte_b, byte_b, byte_b];
            let q8 = _mm_loadl_epi64(staged.as_ptr() as *const __m128i);
            let widened = _mm256_cvtepu8_epi32(q8);

            let shifted = _mm256_srlv_epi32(widened, shifts);
            let codes = _mm256_and_si256(shifted, mask_03);
            let codes_f = _mm256_cvtepi32_ps(codes);

            // weight_f32 = min_i + scale_i * (code / 3.0) — deliberately
            // NOT _mm256_fmadd_ps here: FMA fuses the multiply and add into
            // one rounding step, but the scalar reference does two separate
            // f32 operations (mul, then add), each with its own rounding.
            // Using FMA would silently diverge from scalar by up to 1 ULP
            // on some inputs — plain mul + add matches the scalar's
            // operation-by-operation rounding exactly, which is what the
            // bit-exact parity test (kernel_parity.rs, `to_bits()` equality)
            // requires.
            let scaled = _mm256_div_ps(codes_f, three_vec);
            let term = _mm256_mul_ps(scaled, scale_vec);
            let weights = _mm256_add_ps(min_vec, term);

            _mm256_storeu_ps(out_ptr.add(pass * 8), weights);
        }
    }
}

/// Byte-stream form matching [`super::gq2a_scalar::run_stream`]'s shape.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2 and FMA (see
/// [`crate::simd_strategy::SimdStrategy::detect`]).
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn run_stream(data: &[u8]) -> Vec<f32> {
    let n_blocks = data.len() / GQ2ABlock::BYTES;
    let mut out = vec![0.0f32; n_blocks * GQ2ABlock::WEIGHTS];
    for (bi, chunk) in data.chunks_exact(GQ2ABlock::BYTES).enumerate() {
        let block = GQ2ABlock::from_bytes(chunk).expect("chunks_exact guarantees exact BYTES length");
        let mut block_out = [0.0f32; 256];
        run(block, &mut block_out);
        out[bi * GQ2ABlock::WEIGHTS..(bi + 1) * GQ2ABlock::WEIGHTS].copy_from_slice(&block_out);
    }
    out
}
