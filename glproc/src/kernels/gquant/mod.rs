//! G-Quant dequant kernels (Pridwen v5 §10.2). Phase 1 scope: GQ4A only.
//!
//! `glproc` has no dependency on `glictus-caliburni` (the crate boundary
//! runs the other way: `glictus-caliburni`'s optional `glproc-backend`
//! feature depends on `glproc`, never the reverse), so [`GQ4ABlock`] here is
//! a local, byte-layout-identical mirror of
//! `glictus-caliburni::gquant::GQ4ABlock` rather than an import — both sides
//! are `#[repr(C)]` with the same field order/sizes (Pridwen v5 §3.1), so a
//! byte buffer produced by one reads correctly through the other. This
//! mirrors how `glictus-caliburni::converter` already re-derives GGUF dtype
//! facts locally instead of depending on `glcore`'s enum for anything beyond
//! the `converter` feature's own tensor-loading needs.

pub mod scalar;
pub mod avx2;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gq4a_block_size_is_138_bytes() {
        assert_eq!(std::mem::size_of::<GQ4ABlock>(), GQ4ABlock::BYTES);
    }
}
