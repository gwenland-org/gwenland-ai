//! G-Quant native quantization formats (Pridwen proposal v5).
//!
//! Phase 1: GQ4A (Architecture A, 4-bit foundation, §3.1). Phase 2: GQ2A
//! (Architecture A, 2-bit, asymmetric superblock, §3.2). The block structs
//! here are the binary storage definitions; encode direction lives in
//! [`encoder`] (behind the `converter` feature, since it needs `glcore`'s
//! F32 dequant path upstream of it — see Pridwen v5 §10.1 for why encode is
//! a format-crate concern, not a `glproc` kernel concern). Decode direction
//! (dequant kernels) lives in `glproc`, not here.

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

/// One GQ2A superblock: 256 weights packed as `f16` super-scale + super-min,
/// 16 per-sub-block `i4`-packed scale deltas, 16 per-sub-block `i4`-packed
/// min deltas, and 64 bytes of packed 2-bit codes.
///
/// Reconstruction (Pridwen v5 §3.2 — `min_delta` is additive, not
/// multiplicative: `super_min` can legitimately be 0, which would collapse
/// a multiplicative delta to 0 for every block):
/// ```text
/// scale_i    = super_scale * (1.0 + scale_delta_i / 7.0)
/// min_i      = super_min + super_scale * (min_delta_i / 7.0)
/// weight_f32 = min_i + scale_i * (code / 3.0)         // code in [0,3]
/// ```
///
/// Bit-packing (§3.2 addendum): `scale_delta`/`min_delta` pack 2 raw i4
/// two's-complement nibbles per byte (byte k = delta[2k] low nibble |
/// delta[2k+1] high nibble) — the same convention [`GQ4ABlock`] uses for its
/// 4-bit weight codes. `weights` packs 4 u2 codes per byte, sequential
/// low-to-high (byte k = weight[4k] | weight[4k+1]<<2 | weight[4k+2]<<4 |
/// weight[4k+3]<<6), so one 16-weight sub-block's codes occupy exactly 4
/// contiguous bytes.
///
/// `#[repr(C)]` fixes field order/padding so [`GQ2ABlock::BYTES`] (84) is
/// the actual in-memory size on every target — this struct is read/written
/// as raw bytes in tensor layer files, so its layout is part of the binary
/// format, not an implementation detail.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GQ2ABlock {
    /// Superblock-wide scale, `f16` bits stored as `u16`.
    pub super_scale: u16,
    /// Superblock-wide min, `f16` bits stored as `u16` — the asymmetric
    /// offset that lets GQ2A's unsigned 2-bit codes represent any
    /// per-block range, not just zero-centered ones.
    pub super_min: u16,
    /// Per-sub-block signed scale delta against `super_scale`, packed 2 raw
    /// i4 two's-complement nibbles per byte, one entry per 16-weight
    /// sub-block (16 sub-blocks x 16 weights = 256).
    pub scale_delta: [u8; 8],
    /// Per-sub-block signed min delta against `super_min`, same packing as
    /// `scale_delta`.
    pub min_delta: [u8; 8],
    /// 256 weights packed four 2-bit codes per byte, sequential
    /// low-to-high within each byte.
    pub weights: [u8; 64],
}

impl GQ2ABlock {
    /// Weights per superblock.
    pub const WEIGHTS: usize = 256;
    /// Weights per sub-block (one scale_delta/min_delta entry covers this
    /// many).
    pub const SUB_BLOCK_WEIGHTS: usize = 16;
    /// Sub-blocks per superblock.
    pub const SUB_BLOCKS: usize = 16;
    /// On-disk size in bytes: 2 (super_scale) + 2 (super_min) + 8
    /// (scale_delta) + 8 (min_delta) + 64 (weights).
    pub const BYTES: usize = 84;

    /// Reinterpret an 84-byte slice as a `GQ2ABlock` without copying.
    ///
    /// `raw.len()` must be exactly [`Self::BYTES`] — the caller (layer I/O)
    /// already validated tensor byte counts against dtype/shape, so a
    /// mismatch here means the manifest and tensor index disagree, which is
    /// a corrupt package, not something to silently truncate.
    pub fn from_bytes(raw: &[u8]) -> Option<&GQ2ABlock> {
        if raw.len() != Self::BYTES {
            return None;
        }
        // SAFETY: GQ2ABlock is `repr(C)`, contains only `u16`/`u8` fields
        // (no padding between them: 2 + 2 + 8 + 8 + 64 = 84 with no
        // alignment gaps, since every field's alignment is <= 2 and the
        // struct itself only needs 2-byte alignment), and `raw.len()` was
        // just checked to equal `size_of::<GQ2ABlock>()`.
        Some(unsafe { &*(raw.as_ptr() as *const GQ2ABlock) })
    }

    /// View this block as a raw 84-byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: mirrors `from_bytes` — `repr(C)`, no padding, plain-old-data
        // fields only.
        unsafe { std::slice::from_raw_parts(self as *const Self as *const u8, Self::BYTES) }
    }

    /// Unpack sub-block `blk`'s raw i4 scale delta (two's-complement,
    /// [-8, 7]) from the packed byte pair.
    pub fn scale_delta_at(&self, blk: usize) -> i8 {
        unpack_i4(&self.scale_delta, blk)
    }

    /// Unpack sub-block `blk`'s raw i4 min delta (two's-complement,
    /// [-8, 7]) from the packed byte pair.
    pub fn min_delta_at(&self, blk: usize) -> i8 {
        unpack_i4(&self.min_delta, blk)
    }

    /// Unpack weight `idx`'s 2-bit code ([0, 3]) from the packed byte.
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
    // Sign-extend: a nibble >= 8 (bit 3 set) represents -16 + nibble.
    if nibble >= 8 {
        (nibble as i8) - 16
    } else {
        nibble as i8
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

    #[test]
    fn gq2a_block_size_is_84_bytes() {
        assert_eq!(std::mem::size_of::<GQ2ABlock>(), GQ2ABlock::BYTES);
    }

    #[test]
    fn gq2a_dtype_code() {
        assert_eq!(crate::constants::dtype_codes::GQ2A, 0x0201);
    }

    #[test]
    fn gq2a_from_bytes_rejects_wrong_length() {
        assert!(GQ2ABlock::from_bytes(&[0u8; 83]).is_none());
        assert!(GQ2ABlock::from_bytes(&[0u8; 85]).is_none());
    }

    #[test]
    fn gq2a_from_bytes_and_as_bytes_round_trip() {
        let mut raw = [0u8; GQ2ABlock::BYTES];
        raw[0..2].copy_from_slice(&1u16.to_le_bytes()); // super_scale
        raw[2..4].copy_from_slice(&2u16.to_le_bytes()); // super_min
        raw[4] = 0xAB; // scale_delta byte 0
        raw[12] = 0xCD; // min_delta byte 0
        raw[20] = 0xEF; // first weights byte
        let block = GQ2ABlock::from_bytes(&raw).unwrap();
        assert_eq!(block.super_scale, 1);
        assert_eq!(block.super_min, 2);
        assert_eq!(block.scale_delta[0], 0xAB);
        assert_eq!(block.min_delta[0], 0xCD);
        assert_eq!(block.weights[0], 0xEF);
        assert_eq!(block.as_bytes(), &raw[..]);
    }

    #[test]
    fn gq2a_unpack_i4_sign_extends_correctly() {
        // Byte 0x8F: low nibble 0xF (=15 -> -1), high nibble 0x8 (=8 -> -8).
        let mut block = GQ2ABlock {
            super_scale: 0,
            super_min: 0,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights: [0u8; 64],
        };
        block.scale_delta[0] = 0x8F;
        assert_eq!(block.scale_delta_at(0), -1); // low nibble 0xF
        assert_eq!(block.scale_delta_at(1), -8); // high nibble 0x8

        // Byte 0x70: low nibble 0x0 (=0), high nibble 0x7 (=7, max positive).
        block.scale_delta[1] = 0x70;
        assert_eq!(block.scale_delta_at(2), 0);
        assert_eq!(block.scale_delta_at(3), 7);
    }

    #[test]
    fn gq2a_weight_at_unpacks_four_per_byte_low_to_high() {
        let mut block = GQ2ABlock {
            super_scale: 0,
            super_min: 0,
            scale_delta: [0u8; 8],
            min_delta: [0u8; 8],
            weights: [0u8; 64],
        };
        // byte 0 = 0b11_10_01_00 -> weight[0]=0b00, weight[1]=0b01,
        // weight[2]=0b10, weight[3]=0b11 (low-to-high, sequential).
        block.weights[0] = 0b11_10_01_00;
        assert_eq!(block.weight_at(0), 0b00);
        assert_eq!(block.weight_at(1), 0b01);
        assert_eq!(block.weight_at(2), 0b10);
        assert_eq!(block.weight_at(3), 0b11);
    }
}
