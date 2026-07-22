//! G-Quant dequant kernels (Pridwen v5 §10.2). Phase 1: GQ4A. Phase 2: GQ2A.
//!
//! `glproc` has no dependency on `glictus-caliburni` (the crate boundary
//! runs the other way: `glictus-caliburni`'s optional `glproc-backend`
//! feature depends on `glproc`, never the reverse), so [`GQ4ABlock`]/
//! [`GQ2ABlock`] here are local, byte-layout-identical mirrors of
//! `glictus-caliburni::gquant::{GQ4ABlock, GQ2ABlock}` rather than imports —
//! both sides are `#[repr(C)]` with the same field order/sizes (Pridwen v5
//! §3.1/§3.2), so a byte buffer produced by one reads correctly through the
//! other. This mirrors how `glictus-caliburni::converter` already re-derives
//! GGUF dtype facts locally instead of depending on `glcore`'s enum for
//! anything beyond the `converter` feature's own tensor-loading needs.

pub mod scalar;
pub mod avx2;
pub mod gq2a_scalar;
pub mod gq2a_avx2;

use crate::simd_strategy::SimdStrategy;

/// One GQ4A superblock (Pridwen v5 §3.1): 256 weights as an `f16` (bits, no
/// native `f16` type) super-scale, 8 per-sub-block `i8` deltas, and 128
/// bytes of packed 4-bit codes. `#[repr(C)]` so `size_of` is exactly
/// [`GQ4ABlock::BYTES`] on every target — this layout IS the binary format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GQ4ABlock {
    pub super_scale: u16,
    pub scale_delta: [i8; 8],
    pub weights: [u8; 128],
}

impl GQ4ABlock {
    pub const WEIGHTS: usize = 256;
    pub const SUB_BLOCK_WEIGHTS: usize = 32;
    pub const SUB_BLOCKS: usize = 8;
    pub const BYTES: usize = 138;

    /// Reinterpret a 138-byte slice as a `GQ4ABlock` without copying.
    pub fn from_bytes(raw: &[u8]) -> Option<&GQ4ABlock> {
        if raw.len() != Self::BYTES {
            return None;
        }
        // SAFETY: `repr(C)`, plain-old-data fields only (u16/i8/u8), no
        // padding (2 + 8 + 128 = 138, every field's alignment is <= 2), and
        // length was just checked against `size_of::<GQ4ABlock>()`.
        Some(unsafe { &*(raw.as_ptr() as *const GQ4ABlock) })
    }
}

/// One GQ2A superblock (Pridwen v5 §3.2): 256 weights as `f16` super-scale +
/// super-min (bits, no native `f16` type), 16 per-sub-block packed `i4`
/// scale deltas, 16 per-sub-block packed `i4` min deltas, and 64 bytes of
/// packed 2-bit codes. `#[repr(C)]` so `size_of` is exactly
/// [`GQ2ABlock::BYTES`] on every target — this layout IS the binary format.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GQ2ABlock {
    pub super_scale: u16,
    pub super_min: u16,
    pub scale_delta: [u8; 8],
    pub min_delta: [u8; 8],
    pub weights: [u8; 64],
}

impl GQ2ABlock {
    pub const WEIGHTS: usize = 256;
    pub const SUB_BLOCK_WEIGHTS: usize = 16;
    pub const SUB_BLOCKS: usize = 16;
    pub const BYTES: usize = 84;

    /// Reinterpret an 84-byte slice as a `GQ2ABlock` without copying.
    pub fn from_bytes(raw: &[u8]) -> Option<&GQ2ABlock> {
        if raw.len() != Self::BYTES {
            return None;
        }
        // SAFETY: `repr(C)`, plain-old-data fields only (u16/u8), no padding
        // (2 + 2 + 8 + 8 + 64 = 84, every field's alignment is <= 2), and
        // length was just checked against `size_of::<GQ2ABlock>()`.
        Some(unsafe { &*(raw.as_ptr() as *const GQ2ABlock) })
    }

    /// Unpack sub-block `blk`'s raw i4 scale delta (two's-complement,
    /// [-8, 7]) — see Pridwen v5 §3.2's bit-packing addendum.
    pub fn scale_delta_at(&self, blk: usize) -> i8 {
        unpack_i4(&self.scale_delta, blk)
    }

    /// Unpack sub-block `blk`'s raw i4 min delta (two's-complement, [-8, 7]).
    pub fn min_delta_at(&self, blk: usize) -> i8 {
        unpack_i4(&self.min_delta, blk)
    }

    /// Unpack weight `idx`'s 2-bit code ([0, 3]).
    pub fn weight_at(&self, idx: usize) -> u8 {
        let byte = self.weights[idx / 4];
        (byte >> ((idx % 4) * 2)) & 0b11
    }
}

/// Unpack a signed 4-bit two's-complement nibble at logical index `idx`
/// from a byte array packing 2 nibbles per byte (low nibble = even index,
/// high nibble = odd index) — shared by `scale_delta`/`min_delta` unpacking.
fn unpack_i4(packed: &[u8], idx: usize) -> i8 {
    let byte = packed[idx / 2];
    let nibble = if idx.is_multiple_of(2) { byte & 0x0f } else { byte >> 4 };
    if nibble >= 8 {
        (nibble as i8) - 16
    } else {
        nibble as i8
    }
}

/// Scalar reference dequant — the correctness oracle every fast path is
/// checked against. Decodes one superblock (138 bytes) to 256 `f32`s.
pub fn dequant_gq4a(block: &GQ4ABlock, out: &mut [f32; 256]) {
    scalar::run(block, out);
}

/// Dequant a raw byte stream of N concatenated GQ4A superblocks (the shape
/// every other kernel in this module dispatches on — `&[u8] -> Vec<f32>`),
/// selecting the fastest available SIMD backend at runtime.
///
/// `data.len()` must be a multiple of [`GQ4ABlock::BYTES`]; a short trailing
/// remainder is dropped (mirrors `chunks_exact` used by every sibling
/// dequant kernel in this module — a tensor byte count that doesn't divide
/// evenly means the manifest and tensor index disagree upstream, which is a
/// corrupt-package problem, not this function's to paper over).
pub fn dequant_gq4a_stream(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { avx2::run_stream(data) },
        SimdStrategy::Scalar => scalar::run_stream(data),
    }
}

/// Scalar reference dequant for GQ2A — the correctness oracle every fast
/// path is checked against. Decodes one superblock (84 bytes) to 256 `f32`s.
pub fn dequant_gq2a(block: &GQ2ABlock, out: &mut [f32; 256]) {
    gq2a_scalar::run(block, out);
}

/// Dequant a raw byte stream of N concatenated GQ2A superblocks.
///
/// `data.len()` must be a multiple of [`GQ2ABlock::BYTES`]; a short trailing
/// remainder is dropped (same corrupt-package rationale as
/// [`dequant_gq4a_stream`]).
pub fn dequant_gq2a_stream(data: &[u8]) -> Vec<f32> {
    match SimdStrategy::detect() {
        SimdStrategy::Avx512 | SimdStrategy::Avx2 => unsafe { gq2a_avx2::run_stream(data) },
        SimdStrategy::Scalar => gq2a_scalar::run_stream(data),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gq4a_block_size_is_138_bytes() {
        assert_eq!(std::mem::size_of::<GQ4ABlock>(), GQ4ABlock::BYTES);
    }

    #[test]
    fn gq2a_block_size_is_84_bytes() {
        assert_eq!(std::mem::size_of::<GQ2ABlock>(), GQ2ABlock::BYTES);
    }

    #[test]
    fn gq2a_unpack_i4_sign_extends_correctly() {
        let block = GQ2ABlock {
            super_scale: 0,
            super_min: 0,
            scale_delta: [0x8Fu8, 0x70, 0, 0, 0, 0, 0, 0],
            min_delta: [0u8; 8],
            weights: [0u8; 64],
        };
        assert_eq!(block.scale_delta_at(0), -1); // low nibble 0xF
        assert_eq!(block.scale_delta_at(1), -8); // high nibble 0x8
        assert_eq!(block.scale_delta_at(2), 0);
        assert_eq!(block.scale_delta_at(3), 7);
    }

    #[test]
    fn gq2a_weight_at_unpacks_four_per_byte_low_to_high() {
        let block = GQ2ABlock {
            super_scale: 0,
            super_min: 0,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights: [0b11_10_01_00u8; 64],
        };
        assert_eq!(block.weight_at(0), 0b00);
        assert_eq!(block.weight_at(1), 0b01);
        assert_eq!(block.weight_at(2), 0b10);
        assert_eq!(block.weight_at(3), 0b11);
    }
}
