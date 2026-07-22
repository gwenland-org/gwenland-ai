//! G-Quant native quantization formats (Pridwen proposal v5).
//!
//! Phase 1 scope: GQ4A only (Architecture A, 4-bit foundation, §3.1). The
//! block struct here is the binary storage definition; encode direction
//! lives in [`encoder`] (behind the `converter` feature, since it needs
//! `glcore`'s F32 dequant path upstream of it — see Pridwen v5 §10.1 for why
//! encode is a format-crate concern, not a `glproc` kernel concern). Decode
//! direction (dequant kernels) lives in `glproc`, not here.

#[cfg(feature = "converter")]
pub mod encoder;

/// One GQ4A superblock: 256 weights packed as a `f16` super-scale, 8
/// per-sub-block `i8` scale deltas, and 128 bytes of packed 4-bit codes.
///
/// Reconstruction (Pridwen v5 §3.1):
/// ```text
/// scale_i    = super_scale * (scale_delta_i / 127.0)
/// weight_f32 = scale_i * (code - 8)              // code in [0,15] -> [-8,7]
/// ```
///
/// `#[repr(C)]` fixes field order/padding so [`GQ4ABlock::BYTES`] (138) is
/// the actual in-memory size on every target — this struct is read/written
/// as raw bytes in tensor layer files, so its layout is part of the binary
/// format, not an implementation detail.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GQ4ABlock {
    /// Superblock-wide scale, `f16` bits stored as `u16` (no native `f16`
    /// arithmetic type in std; encode/decode convert at the boundary).
    pub super_scale: u16,
    /// Per-sub-block signed delta against `super_scale`, one per 32-weight
    /// sub-block (8 sub-blocks × 32 weights = 256).
    pub scale_delta: [i8; 8],
    /// 256 weights packed two 4-bit codes per byte, little-endian within
    /// each byte (low nibble = even index, high nibble = odd index).
    pub weights: [u8; 128],
}

impl GQ4ABlock {
    /// Weights per superblock.
    pub const WEIGHTS: usize = 256;
    /// Weights per sub-block (one `scale_delta` entry covers this many).
    pub const SUB_BLOCK_WEIGHTS: usize = 32;
    /// Sub-blocks per superblock.
    pub const SUB_BLOCKS: usize = 8;
    /// On-disk size in bytes: 2 (super_scale) + 8 (scale_delta) + 128 (weights).
    pub const BYTES: usize = 138;

    /// Reinterpret a 138-byte slice as a `GQ4ABlock` without copying.
    ///
    /// `raw.len()` must be exactly [`Self::BYTES`] — the caller (layer I/O)
    /// already validated tensor byte counts against dtype/shape, so a
    /// mismatch here means the manifest and tensor index disagree, which is
    /// a corrupt package, not something to silently truncate.
    pub fn from_bytes(raw: &[u8]) -> Option<&GQ4ABlock> {
        if raw.len() != Self::BYTES {
            return None;
        }
        // SAFETY: GQ4ABlock is `repr(C)`, contains only `u16`/`i8`/`u8`
        // fields (no padding between them: 2 + 8 + 128 = 138 with no
        // alignment gaps, since every field's alignment is <= 2 and the
        // struct itself only needs 2-byte alignment), and `raw.len()` was
        // just checked to equal `size_of::<GQ4ABlock>()`. Byte slices have
        // no alignment guarantee stronger than 1, but `GQ4ABlock`'s max
        // field alignment is 2 (from `u16`); callers reading from a
        // 64-byte-aligned tensor data segment (§3.5) and indexing by whole
        // 138-byte blocks preserve 2-byte alignment for every block after
        // the first only when 138 is even, which it is.
        Some(unsafe { &*(raw.as_ptr() as *const GQ4ABlock) })
    }

    /// View this block as a raw 138-byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: mirrors `from_bytes` — `repr(C)`, no padding, plain-old-data
        // fields only.
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, Self::BYTES) }
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
    fn gq4a_dtype_code() {
        assert_eq!(crate::constants::dtype_codes::GQ4A, 0x0200);
    }

    #[test]
    fn gq4a_from_bytes_rejects_wrong_length() {
        assert!(GQ4ABlock::from_bytes(&[0u8; 137]).is_none());
        assert!(GQ4ABlock::from_bytes(&[0u8; 139]).is_none());
    }

    #[test]
    fn gq4a_from_bytes_and_as_bytes_round_trip() {
        let mut raw = [0u8; GQ4ABlock::BYTES];
        raw[0..2].copy_from_slice(&1u16.to_le_bytes());
        raw[2] = 5; // scale_delta[0]
        raw[10] = 0xAB; // first weights byte
        let block = GQ4ABlock::from_bytes(&raw).unwrap();
        assert_eq!(block.super_scale, 1);
        assert_eq!(block.scale_delta[0], 5);
        assert_eq!(block.weights[0], 0xAB);
        assert_eq!(block.as_bytes(), &raw[..]);
    }
}
