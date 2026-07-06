//! Q5_0 scalar dequantization — ground truth for the SIMD path.
//!
//! GGML Q5_0 block layout (22 bytes → 32 weights):
//!
//! | bytes  | field                                        |
//! |--------|----------------------------------------------|
//! | 0..2   | `d` — scale (f16, LE)                        |
//! | 2..6   | `qh` — 32 high bits, one per weight (u32 LE) |
//! | 6..22  | `qs` — 16 bytes of 4-bit quants              |
//!
//! Weight `i`'s 5-bit value is `nibble(i) | (qh_bit(i) << 4)`, centered:
//! `w = d * (q - 16)`. Low nibbles are weights 0..16, high nibbles 16..32.

use glcore::format::gguf::f16_to_f32;
use glcore::GlError;

/// Weights per Q5_0 block.
pub const BLOCK_NUMEL: usize = 32;
/// Bytes per Q5_0 block.
pub const BLOCK_BYTES: usize = 22;

/// Dequantize one 22-byte Q5_0 block into 32 f32 weights
/// (`output.len() >= 32`).
pub fn dequant_block(data: &[u8], output: &mut [f32]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    debug_assert!(output.len() >= 32);
    let d = f16_to_f32(u16::from_le_bytes([data[0], data[1]]));
    let qh = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);
    let qs = &data[6..22];

    for (i, &byte) in qs.iter().enumerate() {
        // Each byte packs 2 weights: low nibble = weight[i],
        // high nibble = weight[i + 16]; bit i of qh is weight i's 5th bit.
        let lo = ((byte & 0x0F) as u32 | ((qh >> i) & 1) << 4) as f32;
        let hi = ((byte >> 4) as u32 | ((qh >> (i + 16)) & 1) << 4) as f32;
        output[i] = d * (lo - 16.0);
        output[i + 16] = d * (hi - 16.0);
    }
}

/// Dequantize a whole Q5_0 tensor. Load-time path only (allocates).
pub fn run(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(GlError::Parse(format!(
            "Q5_0 data length {} is not a multiple of {BLOCK_BYTES}",
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
