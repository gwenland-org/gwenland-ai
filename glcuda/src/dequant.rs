//! Host-side dequantization for model load (cold path only).
//!
//! glcore's GGUF module handles F32/F16/BF16/Q4_0/Q8_0; the k-quant and
//! Q5_0 layouts below are glcuda's own copies of the glproc scalar ground
//! truth (ADR-001: engines duplicate small shared logic rather than link
//! each other). The math must stay byte-for-byte faithful to glproc's
//! `kernels::dequant::*::scalar` — the host parity tests in this module
//! enforce that against glproc directly.

use glcore::format::gguf::{f16_to_f32, GgufDType, GgufFile, GgufTensorInfo};
use glcore::GlError;

/// Weights per Q4_K/Q6_K super-block.
const K_BLOCK_NUMEL: usize = 256;
/// Bytes per Q4_K super-block.
const Q4_K_BLOCK_BYTES: usize = 144;
/// Bytes per Q5_0 block (32 weights).
const Q5_0_BLOCK_BYTES: usize = 22;
/// Bytes per Q6_K super-block.
const Q6_K_BLOCK_BYTES: usize = 210;
/// Bytes per Q8_0 block (32 weights): f16 scale + 2 bytes padding + 32 i8.
pub const Q8_0_BLOCK_BYTES: usize = 36;
/// Bytes per Q4_0 block (32 weights): f16 scale + 16 bytes of nibbles.
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Unpack the 6-bit (scale, min) pair for Q4_K sub-block `j` (0..8) from
/// the 12-byte packed scales field. Mirrors ggml's `get_scale_min_k4`.
#[inline(always)]
fn q4_k_scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Dequantize a whole Q4_K tensor. Layout: 144-byte super-blocks of 256
/// weights, 8 sub-blocks with 6-bit (scale, min) pairs; `w = d*sc*q - dmin*m`.
pub fn dequant_q4_k(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if !data.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(GlError::Parse(format!(
            "Q4_K data length {} is not a multiple of {Q4_K_BLOCK_BYTES}",
            data.len()
        )));
    }
    let mut out = vec![0f32; data.len() / Q4_K_BLOCK_BYTES * K_BLOCK_NUMEL];
    for (bi, block) in data.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];
        let o = &mut out[bi * K_BLOCK_NUMEL..(bi + 1) * K_BLOCK_NUMEL];
        // 4 chunks of 32 qs bytes; low nibbles fill one sub-block of 32,
        // high nibbles fill the next (GGML order, not per-byte interleave).
        for chunk in 0..4 {
            let (sc1, m1) = q4_k_scale_min(2 * chunk, scales);
            let (sc2, m2) = q4_k_scale_min(2 * chunk + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            let q = &qs[chunk * 32..chunk * 32 + 32];
            for (l, &byte) in q.iter().enumerate() {
                o[chunk * 64 + l] = d1 * (byte & 0x0F) as f32 - min1;
                o[chunk * 64 + 32 + l] = d2 * (byte >> 4) as f32 - min2;
            }
        }
    }
    Ok(out)
}

/// Dequantize a whole Q5_0 tensor. Layout: 22-byte blocks of 32 weights:
/// f16 scale, u32 of high bits, 16 nibble bytes; `w = d * (q - 16)`.
pub fn dequant_q5_0(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if !data.len().is_multiple_of(Q5_0_BLOCK_BYTES) {
        return Err(GlError::Parse(format!(
            "Q5_0 data length {} is not a multiple of {Q5_0_BLOCK_BYTES}",
            data.len()
        )));
    }
    let mut out = vec![0f32; data.len() / Q5_0_BLOCK_BYTES * 32];
    for (bi, block) in data.chunks_exact(Q5_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let o = &mut out[bi * 32..(bi + 1) * 32];
        for (i, &byte) in qs.iter().enumerate() {
            let lo = ((byte & 0x0F) as u32 | ((qh >> i) & 1) << 4) as f32;
            let hi = ((byte >> 4) as u32 | ((qh >> (i + 16)) & 1) << 4) as f32;
            o[i] = d * (lo - 16.0);
            o[i + 16] = d * (hi - 16.0);
        }
    }
    Ok(out)
}

/// Dequantize a whole Q6_K tensor, faithful to GGML's `dequantize_row_q6_K`
/// (glcore's Q6_K assumes a naive linear nibble order and is wrong on real
/// llama.cpp files — route through this instead).
pub fn dequant_q6_k(data: &[u8]) -> Result<Vec<f32>, GlError> {
    if !data.len().is_multiple_of(Q6_K_BLOCK_BYTES) {
        return Err(GlError::Parse(format!(
            "Q6_K data length {} is not a multiple of {Q6_K_BLOCK_BYTES}",
            data.len()
        )));
    }
    let mut out = vec![0f32; data.len() / Q6_K_BLOCK_BYTES * K_BLOCK_NUMEL];
    for (bi, block) in data.chunks_exact(Q6_K_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let o = &mut out[bi * K_BLOCK_NUMEL..(bi + 1) * K_BLOCK_NUMEL];
        // Two independent 128-weight halves; qh[l] packs the 2 high bits of
        // 4 weights spaced 32 apart.
        for half in 0..2 {
            let ql = &block[half * 64..half * 64 + 64];
            let qh = &block[128 + half * 32..128 + half * 32 + 32];
            let sc = &block[192 + half * 8..192 + half * 8 + 8];
            let oh = &mut o[half * 128..half * 128 + 128];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32 - 32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32 - 32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32 - 32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32 - 32;
                oh[l] = d * (sc[is] as i8 as f32) * q1 as f32;
                oh[l + 32] = d * (sc[is + 2] as i8 as f32) * q2 as f32;
                oh[l + 64] = d * (sc[is + 4] as i8 as f32) * q3 as f32;
                oh[l + 96] = d * (sc[is + 6] as i8 as f32) * q4 as f32;
            }
        }
    }
    Ok(out)
}

/// Dequantize any supported tensor dtype to f32, routing the formats
/// glcore does not (or does not correctly) handle through the local copies.
pub fn dequant_any(gguf: &GgufFile, info: &GgufTensorInfo) -> Result<Vec<f32>, GlError> {
    match info.dtype {
        GgufDType::Q4_K => dequant_q4_k(gguf.tensor_data(info)?),
        GgufDType::Q5_0 => dequant_q5_0(gguf.tensor_data(info)?),
        GgufDType::Q6_K => dequant_q6_k(gguf.tensor_data(info)?),
        _ => gguf.dequantize(info),
    }
}

/// Dequantize one row of a Q8_0 matrix (`dim % 32 == 0`) — the host-side
/// embedding lookup, mirroring `GlprocModel::embed_into`.
pub fn q8_0_row_into(blocks: &[u8], row: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), dim);
    let row_bytes = dim / 32 * Q8_0_BLOCK_BYTES;
    let r = &blocks[row * row_bytes..(row + 1) * row_bytes];
    for (j, block) in r.chunks_exact(Q8_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (i, &q) in block[4..36].iter().enumerate() {
            out[j * 32 + i] = d * (q as i8) as f32;
        }
    }
}

/// Dequantize one row of a Q4_0 matrix (`dim % 32 == 0`) — the host-side
/// embedding lookup.
pub fn q4_0_row_into(blocks: &[u8], row: usize, dim: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), dim);
    let row_bytes = dim / 32 * Q4_0_BLOCK_BYTES;
    let r = &blocks[row * row_bytes..(row + 1) * row_bytes];
    for (j, block) in r.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for (i, &byte) in block[2..18].iter().enumerate() {
            let l = byte & 0x0F;
            let h = byte >> 4;
            out[j * 32 + i] = d * ((l as i8) - 8) as f32;
            out[j * 32 + i + 16] = d * ((h as i8) - 8) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random bytes for synthetic quantized blocks.
    fn rand_bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                (state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u8
            })
            .collect()
    }

    /// Overwrite each block's f16 scale field with a small sane value —
    /// random bytes can encode inf/NaN scales, which are not comparable.
    fn set_scales(data: &mut [u8], block_bytes: usize, scale_off: usize) {
        for block in data.chunks_exact_mut(block_bytes) {
            block[scale_off..scale_off + 2].copy_from_slice(&0x2e66u16.to_le_bytes()); // ~0.1
        }
    }

    #[test]
    fn q4_k_matches_glproc_scalar() {
        let mut data = rand_bytes(144 * 3, 1);
        set_scales(&mut data, 144, 0);
        set_scales(&mut data, 144, 2); // dmin too
        let ours = dequant_q4_k(&data).unwrap();
        let theirs = glproc::kernels::dequant::q4_k::scalar::run(&data).unwrap();
        assert_eq!(ours, theirs, "Q4_K dequant must be bit-identical to glproc");
    }

    #[test]
    fn q5_0_matches_glproc_scalar() {
        let mut data = rand_bytes(22 * 5, 2);
        set_scales(&mut data, 22, 0);
        let ours = dequant_q5_0(&data).unwrap();
        let theirs = glproc::kernels::dequant::q5_0::scalar::run(&data).unwrap();
        assert_eq!(ours, theirs, "Q5_0 dequant must be bit-identical to glproc");
    }

    #[test]
    fn q6_k_matches_glproc_scalar() {
        let mut data = rand_bytes(210 * 3, 3);
        set_scales(&mut data, 210, 208);
        let ours = dequant_q6_k(&data).unwrap();
        let theirs = glproc::kernels::dequant::q6_k::scalar::run(&data).unwrap();
        assert_eq!(ours, theirs, "Q6_K dequant must be bit-identical to glproc");
    }

    #[test]
    fn q8_0_row_matches_full_dequant() {
        let (rows, dim) = (3usize, 64usize);
        let mut data = rand_bytes(rows * dim / 32 * 34, 4);
        set_scales(&mut data, 34, 0);
        let full = glproc::kernels::dequant::q8_0::scalar::run(&data);
        
        let mut padded = Vec::with_capacity((data.len() / 34) * 36);
        for block in data.chunks_exact(34) {
            padded.extend_from_slice(&block[0..2]);
            padded.extend_from_slice(&[0, 0]);
            padded.extend_from_slice(&block[2..34]);
        }
        
        for row in 0..rows {
            let mut out = vec![0f32; dim];
            q8_0_row_into(&padded, row, dim, &mut out);
            assert_eq!(out, full[row * dim..(row + 1) * dim].to_vec());
        }
    }

    #[test]
    fn misaligned_lengths_error() {
        assert!(dequant_q4_k(&[0u8; 143]).is_err());
        assert!(dequant_q5_0(&[0u8; 21]).is_err());
        assert!(dequant_q6_k(&[0u8; 209]).is_err());
    }
}
