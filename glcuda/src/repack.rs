//! Load-time weight repacking into the Structure-of-Arrays layouts the
//! GEMV/GEMM kernels read (cold path only).
//!
//! Two repacks live here:
//!
//! * Q4_K -> SoA (`q4_k_to_soa`) — the M2.1 Task A layout for
//!   `gl_gemv_q4_k_soa`. The 144-byte AoS super-block is split into three
//!   contiguous row-major streams:
//!
//!   - `qs`: 4-bit quants, repacked so one u32 holds 8 *consecutive* values
//!     as `byte j = v[8g+j] | v[8g+4+j] << 4` (j = 0..4 within value-group
//!     g). The kernel then splits a u32 load into two dp4a operands with two
//!     `and`s and a `shr` — `w & 0x0F0F0F0F` is values `8g..8g+4`,
//!     `(w >> 4) & 0x0F0F0F0F` is values `8g+4..8g+8`, each matching one
//!     aligned u32 of int8 activations. 128 B per super-block.
//!   - `scales`: per-sub-block f16 of the PRE-MULTIPLIED scale `d * sc`
//!     (8 per super-block). Pre-multiplying at load removes ggml's branchy
//!     6-bit scale unpack from the hot loop entirely; the f16 storage
//!     rounds to nearest-even, adding at most 2^-11 relative error with a
//!     symmetric (cancelling) sign — well below the 1e-2 Q4_K parity
//!     epsilon. 16 B per super-block.
//!   - `mins`: f16 of `dmin * m`, same shape as `scales`. 16 B.
//!
//!   Total: 160 B per 256 weights = 5.0 bpw, vs 4.5 native (the +11% buys
//!   the scale unpack out of the loop) and 8.5 for Q8_0 SoA (the ~2x decode
//!   headline). The per-32 sub-block granularity matches the activation
//!   quantizer's block size, so the integer dot decomposes exactly:
//!   `sum(w*x) = (d*sc)*xs*dot(q, xq) - (dmin*m)*xs*sum(xq)` per sub-block.
//!
//! * f32 -> Q8_0 SoA (`f32_to_q8_0_soa`) — the fallback for quantized GGUF
//!   dtypes with no native kernel (Q6_K, Q5_0, ... — Q4_K_M files carry
//!   Q6_K `ffn_down`/`attn_v`/`output` tensors). Mirrors glproc's documented
//!   repack policy: requantizing an already-lossy int format to Q8_0 adds
//!   ~2^-8 relative error (an order below the source format's own loss),
//!   while the dense-f32 alternative multiplies VRAM *and* per-token DRAM
//!   traffic by 4-6x — which is what would actually sink the 7B decode
//!   budget. Quantization math is byte-for-byte glproc's
//!   `dequant::q8_0::scalar::quantize` (ADR-001 duplication).

use glcore::format::gguf::f16_to_f32;
use glcore::GlError;

/// Weights per Q4_K super-block.
pub const Q4_K_NUMEL: usize = 256;
/// Bytes per AoS Q4_K super-block.
pub const Q4_K_BLOCK_BYTES: usize = 144;
/// Sub-blocks per super-block (32 weights each — the activation block size).
pub const Q4_K_SUB_BLOCKS: usize = 8;

/// f32 -> f16 bit pattern, ROUND-TO-NEAREST-EVEN. Used for the Q4_K
/// pre-multiplied scale/min pairs, where rounding mode is load-bearing:
/// truncation's error is one-sided (every stored scale slightly LOW), so
/// across the 8-16 sub-blocks of a dot product the errors accumulate
/// coherently instead of cancelling — the first T4 parity run failed by
/// exactly that margin (|diff| 1.024e-2 vs the 1e-2 epsilon). RNE is
/// symmetric and half the max error (2^-11 relative).
pub fn f32_to_f16_bits_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let man = bits & 0x7F_FFFF;
    if exp == 0xFF {
        // inf/NaN (quietened): propagate rather than round.
        return sign | 0x7C00 | (((man != 0) as u16) << 9);
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7C00; // overflow -> inf
    }
    if e <= 0 {
        return sign; // < 6.1e-5 flushes to zero (far below any real scale)
    }
    // Truncate 23 -> 10 mantissa bits, then round on the 13 dropped bits.
    // A mantissa carry deliberately overflows into the exponent field —
    // f16's encoding is monotonic, so +1 at the binade boundary is exact.
    let h = ((e as u32) << 10) | (man >> 13);
    let dropped = man & 0x1FFF;
    let round_up = dropped > 0x1000 || (dropped == 0x1000 && (h & 1) == 1);
    let h = h + round_up as u32;
    if h >= 0x7C00 {
        return sign | 0x7C00;
    }
    sign | h as u16
}

/// f32 -> f16 bit pattern, truncating the mantissa (glproc's converter,
/// ADR-001 duplication). Used for the Q8_0 requant scales, where staying
/// byte-for-byte identical to glproc's quantizer matters more than the
/// sub-quantization-noise rounding difference.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = ((bits >> 13) & 0x3FF) as u16;
    if exp <= 0 {
        sign // underflow -> signed zero
    } else if exp >= 31 {
        sign | 0x7C00 // overflow -> infinity
    } else {
        sign | ((exp as u16) << 10) | mantissa
    }
}

/// Unpack the 6-bit (scale, min) pair for sub-block `j` (0..8) from the
/// 12-byte packed scales field. Mirrors ggml's `get_scale_min_k4`.
#[inline(always)]
fn scale_min(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// The Q4_K SoA streams: `(qs, scales_f16, mins_f16)`.
pub type Q4KSoaStreams = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Repack raw AoS Q4_K super-blocks into the SoA triple
/// `(qs, scales_f16, mins_f16)` described in the module doc. Rows stay in
/// their original order (row-major in, row-major out): per super-block the
/// outputs are 128 B of qs, 8 f16 scales and 8 f16 mins.
pub fn q4_k_to_soa(data: &[u8]) -> Result<Q4KSoaStreams, GlError> {
    if !data.len().is_multiple_of(Q4_K_BLOCK_BYTES) {
        return Err(GlError::Parse(format!(
            "Q4_K data length {} is not a multiple of {Q4_K_BLOCK_BYTES}",
            data.len()
        )));
    }
    let n_blocks = data.len() / Q4_K_BLOCK_BYTES;
    let mut qs_out = Vec::with_capacity(n_blocks * 128);
    let mut sc_out = Vec::with_capacity(n_blocks * Q4_K_SUB_BLOCKS * 2);
    let mut mn_out = Vec::with_capacity(n_blocks * Q4_K_SUB_BLOCKS * 2);

    for block in data.chunks_exact(Q4_K_BLOCK_BYTES) {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        for j in 0..Q4_K_SUB_BLOCKS {
            let (sc, m) = scale_min(j, scales);
            // RNE, not the truncating converter — see f32_to_f16_bits_rne.
            sc_out.extend_from_slice(&f32_to_f16_bits_rne(d * sc as f32).to_le_bytes());
            mn_out.extend_from_slice(&f32_to_f16_bits_rne(dmin * m as f32).to_le_bytes());
        }

        // Linearize the ggml nibble order (per 32-byte chunk: low nibbles are
        // one sub-block, high nibbles the next) ...
        let mut v = [0u8; Q4_K_NUMEL];
        for chunk in 0..4 {
            for l in 0..32 {
                let byte = qs[chunk * 32 + l];
                v[chunk * 64 + l] = byte & 0x0F;
                v[chunk * 64 + 32 + l] = byte >> 4;
            }
        }
        // ... then pack 8 consecutive values per u32 in the kernel's
        // lo/hi-nibble split order (see module doc).
        for g in 0..32 {
            for j in 0..4 {
                qs_out.push(v[g * 8 + j] | (v[g * 8 + 4 + j] << 4));
            }
        }
    }
    Ok((qs_out, sc_out, mn_out))
}

/// Bytes per AoS Q4_0 block (f16 scale + 16 nibble bytes, 32 weights).
pub const Q4_0_BLOCK_BYTES: usize = 18;

/// Repack raw AoS Q4_0 blocks into the SoA pair `(qs, scales_f16)` for
/// `gl_gemv_q4_0_soa` (M2.2 Task C-2). The nibble stream uses the same
/// kernel order as Q4_K (`byte j = v[8g+j] | v[8g+4+j] << 4`, so a u32
/// load splits into two dp4a operands with two masks); the f16 block
/// scales are copied VERBATIM — Q4_0's `d` is already the final per-32
/// scale, so unlike Q4_K there is no pre-multiply and no rounding loss.
/// 4.5 bpw streamed, byte-exact dequant vs the AoS ground truth.
pub fn q4_0_to_soa(data: &[u8]) -> Result<(Vec<u8>, Vec<u8>), GlError> {
    if !data.len().is_multiple_of(Q4_0_BLOCK_BYTES) {
        return Err(GlError::Parse(format!(
            "Q4_0 data length {} is not a multiple of {Q4_0_BLOCK_BYTES}",
            data.len()
        )));
    }
    let n_blocks = data.len() / Q4_0_BLOCK_BYTES;
    let mut qs_out = Vec::with_capacity(n_blocks * 16);
    let mut sc_out = Vec::with_capacity(n_blocks * 2);
    for block in data.chunks_exact(Q4_0_BLOCK_BYTES) {
        sc_out.extend_from_slice(&block[0..2]); // f16 d, verbatim
        // GGML order: byte i holds value i (low nibble) and value i+16
        // (high nibble). Linearize, then repack in the kernel order.
        let mut v = [0u8; 32];
        for (i, &byte) in block[2..18].iter().enumerate() {
            v[i] = byte & 0x0F;
            v[i + 16] = byte >> 4;
        }
        for g in 0..4 {
            for j in 0..4 {
                qs_out.push(v[g * 8 + j] | (v[g * 8 + 4 + j] << 4));
            }
        }
    }
    Ok((qs_out, sc_out))
}

/// Quantize dense f32 weights (`len % 32 == 0`) straight into the Q8_0 SoA
/// pair `(qs, scales_f16)` the `gl_gemv_q8_0_soa` / `gl_gemm_q8_0_soa`
/// kernels read. Per 32-group: `d = max|v| / 127` (round-tripped through
/// f16 so `q` is computed against the exact scale the kernel reads back),
/// `q = round-half-away(v / d)` — glproc's quantizer, minus the AoS
/// interleave.
pub fn f32_to_q8_0_soa(values: &[f32]) -> (Vec<u8>, Vec<u8>) {
    debug_assert!(values.len().is_multiple_of(32));
    let n_blocks = values.len() / 32;
    let mut qs = Vec::with_capacity(n_blocks * 32);
    let mut scales = Vec::with_capacity(n_blocks * 2);
    for group in values.chunks_exact(32) {
        let amax = group.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let d_bits = f32_to_f16_bits(amax / 127.0);
        let d = f16_to_f32(d_bits);
        scales.extend_from_slice(&d_bits.to_le_bytes());
        let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
        for &v in group {
            let scaled = v * inv;
            qs.push(((scaled + 0.5f32.copysign(scaled)) as i32 as i8) as u8);
        }
    }
    (qs, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// Give every super-block finite, sane d / dmin (random bytes can encode
    /// inf/NaN f16 scales).
    fn set_scales(data: &mut [u8]) {
        for block in data.chunks_exact_mut(Q4_K_BLOCK_BYTES) {
            block[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // d ~0.1
            block[2..4].copy_from_slice(&0x2a66u16.to_le_bytes()); // dmin ~0.05
        }
    }

    /// Reconstruct weight `i` of super-block `bi` from the SoA arrays with
    /// exactly the kernel's math: `w = f16(d*sc) * q - f16(dmin*m)`.
    fn soa_weight(qs: &[u8], sc: &[u8], mn: &[u8], bi: usize, i: usize) -> f32 {
        let sub = i / 32;
        let s = f16_to_f32(u16::from_le_bytes([
            sc[(bi * 8 + sub) * 2],
            sc[(bi * 8 + sub) * 2 + 1],
        ]));
        let m = f16_to_f32(u16::from_le_bytes([
            mn[(bi * 8 + sub) * 2],
            mn[(bi * 8 + sub) * 2 + 1],
        ]));
        // Value i lives in group g = i/8; byte j = i%8 (mod 4) holds it in
        // the low (i%8 < 4) or high nibble.
        let g = i / 8;
        let r = i % 8;
        let byte = qs[bi * 128 + g * 4 + (r % 4)];
        let q = if r < 4 { byte & 0x0F } else { byte >> 4 };
        s * q as f32 - m
    }

    /// The SoA repack must reconstruct, weight for weight, the same value as
    /// the glproc scalar ground truth — up to the documented f16 rounding of
    /// the pre-multiplied (d*sc, dmin*m) pairs, which is bounded well below
    /// the Q4_K parity epsilon.
    #[test]
    fn q4_k_soa_reconstructs_glproc_dequant() {
        let mut data = rand_bytes(Q4_K_BLOCK_BYTES * 5, 11);
        set_scales(&mut data);
        let want = glproc::kernels::dequant::q4_k::scalar::run(&data).unwrap();
        let (qs, sc, mn) = q4_k_to_soa(&data).unwrap();
        assert_eq!(qs.len(), 128 * 5);
        assert_eq!(sc.len(), 16 * 5);
        assert_eq!(mn.len(), 16 * 5);
        for (bi, block) in data.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            for i in 0..Q4_K_NUMEL {
                let got = soa_weight(&qs, &sc, &mn, bi, i);
                let w = want[bi * Q4_K_NUMEL + i];
                // RNE premul: each pair is within 2^-11 relative, so the
                // elementwise bound is on the GROSS magnitudes
                // |d*sc*q| + |dmin*m| (their difference — the weight — can
                // be much smaller through cancellation).
                let (s6, m6) = scale_min(i / 32, &block[4..16]);
                let gross = d * s6 as f32 * 15.0 + dmin * m6 as f32;
                let tol = gross * 2f32.powi(-11) + 1e-6;
                assert!(
                    (got - w).abs() <= tol,
                    "block {bi} weight {i}: soa {got} vs glproc {w} (tol {tol})"
                );
            }
        }
    }

    /// Exactness check of the *packing* itself (no f16 premul involved): the
    /// nibble extracted from the SoA stream must equal the nibble ggml's
    /// order assigns to that linear index.
    #[test]
    fn q4_k_soa_nibble_order_is_exact() {
        let mut data = rand_bytes(Q4_K_BLOCK_BYTES * 2, 12);
        set_scales(&mut data);
        let (qs, _, _) = q4_k_to_soa(&data).unwrap();
        for (bi, block) in data.chunks_exact(Q4_K_BLOCK_BYTES).enumerate() {
            let raw = &block[16..144];
            for i in 0..Q4_K_NUMEL {
                // ggml linear order: chunk = i/64, within-chunk index i%64;
                // < 32 -> low nibble of byte, >= 32 -> high nibble.
                let (chunk, w) = (i / 64, i % 64);
                let expect = if w < 32 {
                    raw[chunk * 32 + w] & 0x0F
                } else {
                    raw[chunk * 32 + (w - 32)] >> 4
                };
                let g = i / 8;
                let r = i % 8;
                let byte = qs[bi * 128 + g * 4 + (r % 4)];
                let got = if r < 4 { byte & 0x0F } else { byte >> 4 };
                assert_eq!(got, expect, "block {bi} value {i} packed wrong");
            }
        }
    }

    #[test]
    fn q4_k_soa_rejects_ragged_input() {
        assert!(q4_k_to_soa(&[0u8; Q4_K_BLOCK_BYTES - 1]).is_err());
    }

    /// Q4_0 SoA repack must reconstruct, bit-exactly, the same weights as
    /// the glproc scalar dequant — the scales are verbatim f16 and the
    /// nibbles are lossless, so unlike Q4_K there is no tolerance at all.
    #[test]
    fn q4_0_soa_reconstructs_glproc_dequant_exactly() {
        let mut data = rand_bytes(Q4_0_BLOCK_BYTES * 7, 21);
        for block in data.chunks_exact_mut(Q4_0_BLOCK_BYTES) {
            block[0..2].copy_from_slice(&0x2e66u16.to_le_bytes()); // sane d
        }
        let want = glproc::kernels::dequant::q4_0::scalar::run(&data);
        let (qs, sc) = q4_0_to_soa(&data).unwrap();
        assert_eq!(qs.len(), 16 * 7);
        assert_eq!(sc.len(), 2 * 7);
        for (bi, _) in data.chunks_exact(Q4_0_BLOCK_BYTES).enumerate() {
            let d = f16_to_f32(u16::from_le_bytes([sc[bi * 2], sc[bi * 2 + 1]]));
            for i in 0..32 {
                // Kernel order: value i -> group i/8, byte (i%8)%4, lo/hi.
                let (g, r) = (i / 8, i % 8);
                let byte = qs[bi * 16 + g * 4 + (r % 4)];
                let q = if r < 4 { byte & 0x0F } else { byte >> 4 };
                let got = d * (q as i8 - 8) as f32;
                assert_eq!(got, want[bi * 32 + i], "block {bi} value {i}");
            }
        }
    }

    #[test]
    fn q4_0_soa_rejects_ragged_input() {
        assert!(q4_0_to_soa(&[0u8; Q4_0_BLOCK_BYTES - 1]).is_err());
    }

    /// The f32 -> Q8_0 SoA requant must be glproc's quantizer followed by
    /// the loader's AoS -> SoA split — byte for byte.
    #[test]
    fn f32_to_q8_0_soa_matches_glproc_quantize() {
        let values: Vec<f32> = (0..96).map(|i| ((i as f32) * 0.37).sin() * 0.2).collect();
        let aos = glproc::kernels::dequant::q8_0::scalar::quantize(&values);
        let mut want_qs = Vec::new();
        let mut want_sc = Vec::new();
        for block in aos.chunks_exact(34) {
            want_sc.extend_from_slice(&block[0..2]);
            want_qs.extend_from_slice(&block[2..34]);
        }
        let (qs, sc) = f32_to_q8_0_soa(&values);
        assert_eq!(qs, want_qs);
        assert_eq!(sc, want_sc);
    }

    /// RNE converter: exact values survive, ties round to even, the
    /// mantissa carry at a binade boundary lands on the next exponent.
    #[test]
    fn f16_bits_rne_rounds_to_nearest_even() {
        for v in [0.0f32, 1.0, -2.0, 0.5, 65504.0] {
            assert_eq!(f16_to_f32(f32_to_f16_bits_rne(v)), v);
        }
        // 1.0 + 2^-11 is exactly halfway between 1.0 and the next f16
        // (1.0 + 2^-10): ties-to-even keeps the even mantissa (1.0).
        assert_eq!(f16_to_f32(f32_to_f16_bits_rne(1.0 + 2f32.powi(-11))), 1.0);
        // Just above halfway rounds up.
        assert_eq!(
            f16_to_f32(f32_to_f16_bits_rne(1.0 + 2f32.powi(-11) + 2f32.powi(-16))),
            1.0 + 2f32.powi(-10)
        );
        // Largest f16 mantissa rounding up must carry into the exponent.
        assert_eq!(f16_to_f32(f32_to_f16_bits_rne(1.9999999f32)), 2.0);
        // Above f16 max -> inf; RNE error bound holds everywhere else.
        assert_eq!(f16_to_f32(f32_to_f16_bits_rne(1e9)), f32::INFINITY);
        for i in 0..1000 {
            let v = 0.003 + i as f32 * 0.37;
            let rt = f16_to_f32(f32_to_f16_bits_rne(v));
            assert!((rt - v).abs() <= v.abs() * 2f32.powi(-11), "{v} -> {rt}");
        }
    }

    #[test]
    fn f16_bits_round_trip() {
        for v in [0.0f32, 1.0, -2.0, 0.1, 1e-3, 65504.0] {
            let rt = f16_to_f32(f32_to_f16_bits(v));
            assert!((rt - v).abs() <= v.abs() * 1e-3, "{v} -> {rt}");
        }
        assert_eq!(f16_to_f32(f32_to_f16_bits(1e9)), f32::INFINITY);
    }
}
