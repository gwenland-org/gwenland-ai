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

/// Repack a whole Q5_0 tensor as Q8_0 blocks. Load-time path only.
///
/// A Q5_0 weight is `d · (q − 16)` with `q − 16` ∈ [−16, 15]. Scaling the
/// integer by 8 and the scale by ⅛ (an exponent decrement in f16) keeps
/// every value **bit-exact** in the Q8_0 range [−128, 120]. The trade:
/// +55% weight bytes for Q8_0's much cheaper inner loop — a net win
/// because the Q5_0 high-bit unpack makes its dot compute-bound while the
/// memory stream still has headroom (measured on the i3-1115G4).
pub fn repack_to_q8_0(data: &[u8]) -> Result<Vec<u8>, GlError> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(GlError::Parse(format!(
            "Q5_0 data length {} is not a multiple of {BLOCK_BYTES}",
            data.len()
        )));
    }
    let n_blocks = data.len() / BLOCK_BYTES;
    // Q8_0 block = 2-byte f16 scale + 32 int8 quants.
    let mut out = Vec::with_capacity(n_blocks * 34);
    let mut dq = [0f32; BLOCK_NUMEL];
    for block in data.chunks_exact(BLOCK_BYTES) {
        let d_bits = u16::from_le_bytes([block[0], block[1]]);
        let exp = (d_bits >> 10) & 0x1F;
        if exp > 3 && exp < 0x1F {
            // Normal scale with room to shift: d/8 = exponent − 3, exact.
            let d8_bits = (d_bits & 0x83FF) | ((exp - 3) << 10);
            out.extend_from_slice(&d8_bits.to_le_bytes());
            let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
            let qs = &block[6..22];
            let mut q8 = [0u8; BLOCK_NUMEL];
            for (i, &byte) in qs.iter().enumerate() {
                let lo = ((byte & 0x0F) as i32) | ((((qh >> i) & 1) as i32) << 4);
                let hi = ((byte >> 4) as i32) | ((((qh >> (i + 16)) & 1) as i32) << 4);
                q8[i] = (((lo - 16) * 8) as i8) as u8;
                q8[i + 16] = (((hi - 16) * 8) as i8) as u8;
            }
            out.extend_from_slice(&q8);
        } else {
            // Tiny/degenerate scale (would underflow the shift): requantize
            // generically. Adds ≤0.4% error on values that are ~0 anyway.
            dequant_block(block, &mut dq);
            out.extend_from_slice(&crate::kernels::dequant::q8_0::scalar::quantize(&dq));
        }
    }
    Ok(out)
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
