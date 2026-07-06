//! Q6_K scalar dequantization, faithful to GGML's `dequantize_row_q6_K`.
//!
//! GGML Q6_K super-block layout (210 bytes → 256 weights):
//!
//! | bytes     | field                                          |
//! |-----------|------------------------------------------------|
//! | 0..128    | `ql` — low 4 bits, 2 weights per byte          |
//! | 128..192  | `qh` — high 2 bits, 4 weights per byte         |
//! | 192..208  | `scales` — 16 signed i8 sub-block scales       |
//! | 208..210  | `d` — super-scale (f16, LE)                    |
//!
//! The element order is NOT linear: the block is two 128-weight halves, and
//! within a half, byte `l` of `ql` contributes weight `l` (low nibble) and
//! weight `l + 64` (high nibble), while `qh[l]` packs the 2 high bits of
//! weights `l`, `l+32`, `l+64` and `l+96`. `w = d * scale * (q6 - 32)`.
//!
//! NOTE: glcore's `dequant_q6_k` assumes a naive linear nibble order and
//! disagrees with this layout on real llama.cpp files — the loader routes
//! Q6_K through this kernel instead.

use glcore::format::gguf::f16_to_f32;
use glcore::GlError;

/// Weights per Q6_K super-block.
pub const BLOCK_NUMEL: usize = 256;
/// Bytes per Q6_K super-block.
pub const BLOCK_BYTES: usize = 210;

/// Dequantize one 210-byte Q6_K super-block into 256 f32 weights.
pub fn dequant_block(data: &[u8], output: &mut [f32; 256]) {
    debug_assert!(data.len() >= BLOCK_BYTES);
    let d = f16_to_f32(u16::from_le_bytes([data[208], data[209]]));

    // Two independent 128-weight halves.
    for half in 0..2 {
        let ql = &data[half * 64..half * 64 + 64];
        let qh = &data[128 + half * 32..128 + half * 32 + 32];
        let sc = &data[192 + half * 8..192 + half * 8 + 8];
        let out = &mut output[half * 128..half * 128 + 128];

        for l in 0..32 {
            let is = l / 16; // sub-block selector within the half
            // qh[l] packs the 2 high bits of 4 weights, 32 apart.
            let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32 - 32;
            let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
            let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
            let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;

            out[l] = d * (sc[is] as i8 as f32) * q1 as f32;
            out[l + 32] = d * (sc[is + 2] as i8 as f32) * q2 as f32;
            out[l + 64] = d * (sc[is + 4] as i8 as f32) * q3 as f32;
            out[l + 96] = d * (sc[is + 6] as i8 as f32) * q4 as f32;
        }
    }
}

/// Dequantize a whole Q6_K tensor. Load-time path only (allocates).
pub fn run(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if data.len() % BLOCK_BYTES != 0 {
        return Err(GlError::Parse(format!(
            "Q6_K data length {} is not a multiple of {BLOCK_BYTES}",
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
