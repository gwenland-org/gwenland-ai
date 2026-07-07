//! Q4_K scalar dequantization — the ground truth every SIMD path is
//! validated against.
//!
//! GGML Q4_K super-block layout (144 bytes → 256 weights):
//!
//! | bytes    | field                                             |
//! |----------|---------------------------------------------------|
//! | 0..2     | `d`    — super-scale (f16, LE)                    |
//! | 2..4     | `dmin` — super-min scale (f16, LE)                |
//! | 4..16    | `scales` — 8 sub-block (scale, min) pairs, 6-bit packed |
//! | 16..144  | `qs` — 128 bytes of 4-bit quants                  |
//!
//! Each of the 8 sub-blocks covers 32 weights: `w = d*sc * q - dmin*m`.
//! Nibble order is GGML's: within each 32-byte `qs` chunk, the low nibbles
//! are weights `0..32` of one sub-block and the high nibbles are weights
//! `0..32` of the *next* sub-block (not low/high interleaved per byte).

use glcore::format::gguf::f16_to_f32;
use glcore::GlError;

/// Weights per Q4_K super-block.
pub const BLOCK_NUMEL: usize = 256;
/// Bytes per Q4_K super-block.
pub const BLOCK_BYTES: usize = 144;

/// Unpack the 6-bit (scale, min) pair for sub-block `j` (0..8) from the
/// 12-byte packed `scales` field. Mirrors ggml's `get_scale_min_k4`.
#[inline(always)]
pub fn scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        // Sub-blocks 4..8 borrow their two high bits from bytes 0..8.
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize one 144-byte Q4_K super-block into 256 f32 weights.
/// Pure safe Rust — this is the parity reference for the SIMD kernels.
pub fn dequant_block(data: &[u8], output: &mut [f32; 256]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    let dmin = f16_to_f32(u16::from_le_bytes([data[2], data[3]]));
    let scales = &data[4..16];
    let qs = &data[16..144];

    let mut out = 0;
    // 4 chunks of 32 qs bytes; each chunk yields 2 sub-blocks of 32 weights.
    for chunk in 0..4 {
        let (sc1, m1) = scale_min(2 * chunk, scales);
        let (sc2, m2) = scale_min(2 * chunk + 1, scales);
        let d1 = d * sc1 as f32;
        let min1 = dmin * m1 as f32;
        let d2 = d * sc2 as f32;
        let min2 = dmin * m2 as f32;
        let q = &qs[chunk * 32..chunk * 32 + 32];
        for (l, &byte) in q.iter().enumerate() {
            // Low nibbles fill this sub-block, high nibbles fill the next one.
            output[out + l] = d1 * (byte & 0x0F) as f32 - min1;
            output[out + 32 + l] = d2 * (byte >> 4) as f32 - min2;
        }
        out += 64;
    }
}

/// Dequantize a whole Q4_K tensor (`data.len() % 144 == 0`) to a fresh Vec.
/// Load-time path only — never call in the decode loop (allocates).
pub fn run(data: &[u8]) -> Result<Vec<f32>, GlError> {
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

/// Repack Q4_K blocks to Q8_0 blocks (dequantize, then requantize).
///
/// Not bit-exact: Q4_K's per-32 affine (scale + min) values are requantized
/// onto Q8_0's per-32 symmetric grid, adding at most one int8 rounding step
/// (~0.4% relative) — well under Q4_K's own quantization noise, and the same
/// trade already accepted for the Q6_K repack. Buys the integer-dot Q8_0
/// kernels for both decode and the batched-prefill matmul; the f32 bridge
/// fallback re-dequantized every block once per batch row, which made
/// Q4_K layers ~15x slower than repacked ones in prefill.
pub fn repack_to_q8_0(data: &[u8]) -> Result<Vec<u8>, GlError> {
    Ok(crate::kernels::dequant::q8_0::scalar::quantize(&run(data)?))
}
