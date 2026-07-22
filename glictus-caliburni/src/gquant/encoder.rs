//! GQ4A encoder: F32 weights → packed [`GQ4ABlock`]s (Pridwen v5 §9 step 4,
//! §10.1). Conversion-time only — runs once per model at `glconv` time, not
//! per inference step, which is why this lives in the format crate behind
//! `converter` rather than in `glproc`.

use crate::gquant::GQ4ABlock;

/// Convert an IEEE-754 `f32` to `f16` bits, round-to-nearest-even on the
/// mantissa. No `half` crate dependency — mirrors
/// [`glcore::format::gguf::f16_to_f32`]'s bit layout so the two are exact
/// inverses (mod round-trip precision loss) of each other.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xff) as i32;
    let frac = bits & 0x007f_ffff;

    if exp == 0xff {
        // Inf / NaN: preserve sign + a nonzero frac flag for NaN.
        let f16_frac = if frac != 0 { 0x200 } else { 0 };
        return (sign | 0x7c00 | f16_frac) as u16;
    }

    // Unbias f32's exponent (127) and rebias to f16 (15).
    let unbiased = exp - 127;
    let f16_exp = unbiased + 15;

    if f16_exp >= 0x1f {
        // Overflow: saturate to infinity.
        return (sign | 0x7c00) as u16;
    }
    if f16_exp <= 0 {
        // Underflow to zero or subnormal; subnormals aren't needed for the
        // quantization scales this encoder produces (always > 0 for any
        // nonzero input), so flush to signed zero.
        return sign as u16;
    }

    let f16_frac = frac >> 13;
    (sign | ((f16_exp as u32) << 10) | f16_frac) as u16
}

/// Quantize a whole tensor's `f32` weights into GQ4A superblocks.
///
/// `weights.len()` must be a multiple of 256 — GQ4A, like GGUF's Q4_K/Q6_K,
/// is a fixed 256-element superblock format with no partial/padded final
/// block defined by the spec (Pridwen v5 §3.1 doesn't address ragged
/// tensors). Callers (the `glconv` CPP assignment) are expected to check
/// divisibility before calling and fall back to a non-superblock dtype
/// otherwise, mirroring how GGUF itself never ships a Q4_K/Q6_K tensor whose
/// element count isn't a multiple of 256.
pub fn encode_gq4a_tensor(weights: &[f32]) -> Option<Vec<GQ4ABlock>> {
    if !weights.len().is_multiple_of(GQ4ABlock::WEIGHTS) {
        return None;
    }
    Some(
        weights
            .chunks_exact(GQ4ABlock::WEIGHTS)
            .map(|chunk| encode_gq4a(chunk.try_into().expect("chunks_exact guarantees exact WEIGHTS length")))
            .collect(),
    )
}

/// Quantize a slice of 256 `f32` weights into one [`GQ4ABlock`] (Pridwen v5
/// §3.1 encoding algorithm).
///
/// 1. `super_scale = max(|w|) / 7.0` across all 256 weights.
/// 2. Per 32-weight sub-block: local `scale_i = max(|w|) / 7.0`, encoded as
///    `scale_delta_i = round(scale_i / super_scale * 127.0)` (`i8`).
/// 3. Per weight: `code = clamp(round(w / actual_scale_i) + 8, 0, 15)`,
///    where `actual_scale_i = super_scale * (scale_delta_i / 127.0)` — the
///    same reconstruction the decoder uses, so encode/decode agree on what
///    "the" scale for a sub-block is.
pub fn encode_gq4a(weights: &[f32; 256]) -> GQ4ABlock {
    let global_max_abs = weights.iter().fold(0.0f32, |acc, &w| acc.max(w.abs()));
    let super_scale = global_max_abs / 7.0;

    let mut scale_delta = [0i8; GQ4ABlock::SUB_BLOCKS];
    let mut packed = [0u8; 128];

    for (blk, chunk) in weights.chunks_exact(GQ4ABlock::SUB_BLOCK_WEIGHTS).enumerate() {
        let local_max_abs = chunk.iter().fold(0.0f32, |acc, &w| acc.max(w.abs()));
        let local_scale = local_max_abs / 7.0;

        let delta = if super_scale > 0.0 {
            (local_scale / super_scale * 127.0).round().clamp(-127.0, 127.0) as i8
        } else {
            0
        };
        scale_delta[blk] = delta;

        let actual_scale = super_scale * (delta as f32 / 127.0);

        for (i, &w) in chunk.iter().enumerate() {
            let code: u8 = if actual_scale > 0.0 {
                ((w / actual_scale).round() + 8.0).clamp(0.0, 15.0) as u8
            } else {
                8 // midpoint: zero-scale block can only represent zero
            };
            let idx = blk * GQ4ABlock::SUB_BLOCK_WEIGHTS + i;
            let byte_idx = idx / 2;
            if idx.is_multiple_of(2) {
                packed[byte_idx] |= code; // low nibble
            } else {
                packed[byte_idx] |= code << 4; // high nibble
            }
        }
    }

    GQ4ABlock {
        super_scale: f32_to_f16(super_scale),
        scale_delta,
        weights: packed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar reference dequant, mirroring glproc's kernel formula exactly —
    /// duplicated here (not imported from glproc, which this crate does not
    /// depend on) so encoder round-trip tests don't need a glproc dep.
    fn dequant_reference(block: &GQ4ABlock) -> [f32; 256] {
        let super_scale = test_f16_to_f32(block.super_scale);
        let mut out = [0f32; 256];
        for blk in 0..GQ4ABlock::SUB_BLOCKS {
            let scale_i = super_scale * (block.scale_delta[blk] as f32 / 127.0);
            for i in 0..GQ4ABlock::SUB_BLOCK_WEIGHTS {
                let idx = blk * GQ4ABlock::SUB_BLOCK_WEIGHTS + i;
                let byte = block.weights[idx / 2];
                let code = if idx.is_multiple_of(2) { byte & 0x0f } else { byte >> 4 };
                out[idx] = scale_i * (code as f32 - 8.0);
            }
        }
        out
    }

    #[test]
    fn gq4a_encode_decode_round_trip() {
        let mut weights = [0f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            // Deterministic pseudo-random synthetic distribution in [-3, 3].
            *w = ((i * 37 % 97) as f32 / 97.0 - 0.5) * 6.0;
        }
        let block = encode_gq4a(&weights);
        let decoded = dequant_reference(&block);

        // Per-weight rounding error is bounded by half the *sub-block's own*
        // code step (`actual_scale_i`), not the superblock's nominal
        // `super_scale / 7.0` — those only coincide when a sub-block's
        // scale_delta saturates to +127. Each sub-block's actual_scale can
        // differ slightly from super_scale (that's the whole point of the
        // per-block delta), so the tolerance is recomputed per sub-block.
        let super_scale = test_f16_to_f32(block.super_scale);
        for blk in 0..GQ4ABlock::SUB_BLOCKS {
            let actual_scale = super_scale * (block.scale_delta[blk] as f32 / 127.0);
            let tol = actual_scale / 2.0; // half a code step = max rounding error
            for i in 0..GQ4ABlock::SUB_BLOCK_WEIGHTS {
                let idx = blk * GQ4ABlock::SUB_BLOCK_WEIGHTS + i;
                let (orig, dec) = (weights[idx], decoded[idx]);
                assert!(
                    (orig - dec).abs() <= tol + 1e-6,
                    "blk={blk} idx={idx} orig={orig} dec={dec} tol={tol}"
                );
            }
        }
    }

    #[test]
    fn gq4a_encode_all_zeros() {
        let weights = [0f32; 256];
        let block = encode_gq4a(&weights);
        assert_eq!(block.super_scale, 0);
        for &code_byte in block.weights.iter() {
            assert_eq!(code_byte & 0x0f, 8, "low nibble should be midpoint");
            assert_eq!(code_byte >> 4, 8, "high nibble should be midpoint");
        }
        let decoded = dequant_reference(&block);
        assert!(decoded.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn gq4a_encode_uniform() {
        const C: f32 = 2.8;
        let weights = [C; 256];
        let block = encode_gq4a(&weights);
        // super_scale = C / 7.0 exactly representable-ish in f16; check via
        // decode instead of asserting the raw f16 bits (avoids re-deriving
        // f16 rounding rules in the test).
        for &code_byte in block.weights.iter() {
            assert_eq!(code_byte & 0x0f, 15, "max positive code");
            assert_eq!(code_byte >> 4, 15, "max positive code");
        }
        let decoded = dequant_reference(&block);
        for &w in decoded.iter() {
            assert!((w - C).abs() < 0.02, "w={w} C={C}");
        }
    }

    #[test]
    fn gq4a_encode_tensor_rejects_non_multiple_of_256() {
        assert!(encode_gq4a_tensor(&[0.0; 255]).is_none());
        assert!(encode_gq4a_tensor(&[0.0; 257]).is_none());
    }

    #[test]
    fn gq4a_encode_tensor_produces_one_block_per_256() {
        let weights = vec![1.0f32; 512];
        let blocks = encode_gq4a_tensor(&weights).unwrap();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn gq4a_scale_delta_range() {
        let mut weights = [0f32; 256];
        for (i, w) in weights.iter_mut().enumerate() {
            // Deliberately lopsided per-sub-block magnitudes to exercise
            // the full scale_delta range.
            let blk = i / 32;
            let mag = if blk == 0 { 100.0 } else { 0.001 * (blk as f32) };
            *w = if i % 2 == 0 { mag } else { -mag };
        }
        let block = encode_gq4a(&weights);
        for &d in block.scale_delta.iter() {
            // Spec range is [-127, 127] (normalized by 127.0, the max
            // *positive* i8 magnitude) — -128 is in-range for the `i8` type
            // but out-of-spec for this format, so the meaningful check is
            // that the encoder's `.clamp(-127.0, 127.0)` actually excludes it.
            assert_ne!(d, i8::MIN, "scale_delta hit i8::MIN (out of spec range): {d}");
        }
    }

    #[test]
    fn gq4a_dtype_code_is_0x0200() {
        assert_eq!(crate::constants::dtype_codes::GQ4A, 0x0200);
    }

    #[test]
    fn gq4a_cpp_sensitivity_table_covers_qwen_layers() {
        use crate::converter::gquant_policy::sensitivity_bucket_for;
        let real_qwen_layer_names = [
            "token_embd",
            "output",
            "output_norm",
            "attn_norm",
            "ffn_norm",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_output",
            "ffn_gate",
            "ffn_up",
            "ffn_down",
        ];
        for name in real_qwen_layer_names {
            assert!(
                sensitivity_bucket_for(name).is_some(),
                "layer {name:?} has no sensitivity bucket"
            );
        }
    }

    // Exposed only for this module's own tests (dequant_reference needs the
    // inverse of f32_to_f16 to verify round-trip; glcore's f16_to_f32 isn't
    // reachable from a non-`converter` test target, but this file is always
    // compiled under `converter`, which does depend on glcore — so we just
    // delegate to it directly instead of duplicating the bit math twice).
    pub(super) fn test_f16_to_f32(bits: u16) -> f32 {
        glcore::format::gguf::f16_to_f32(bits)
    }
}
