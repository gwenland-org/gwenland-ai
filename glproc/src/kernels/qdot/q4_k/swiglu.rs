//! Fused Q4_K SwiGLU — the Wave-4 lever.
//!
//! Wave 3 proved native Q4_K's problem was not the dot kernel (per-MAC parity
//! with Q8_0 in isolation) but the **loss of SwiGLU fusion**: routing Q4_K
//! through `GateUp::Split` gave up `par_matvec_swiglu`, dropping gate_up from
//! 86% to 13% of the bandwidth ceiling. This module restores fusion for Q4_K.
//!
//! # Two candidates, measured — not assumed
//!
//! The Wave-4 spec proposed an **f32-domain** kernel: dequant each nibble to
//! f32, apply a polynomial SiLU per element, multiply. That is structurally the
//! same per-element unpack that made the *first* native Q4_K attempt lose 33%
//! (compute-bound, gap identical L2-warm). So both approaches are built here
//! and benched head to head against the Q8_0 baseline, and the numbers decide:
//!
//! - [`fused_swiglu_q8k`] — **integer domain**: two `row_dot_q8k` calls (the
//!   Wave-2 kernel that reached Q8_0 parity), fused into one dispatch with SiLU
//!   inline in registers on the two scalar results. Same shape as the Q8_0
//!   `par_matvec_swiglu` that hits 86%.
//! - [`fused_swiglu_f32`] — **f32 domain**, the literal spec: dequant→SiLU→mul.
//!
//! [`fused_swiglu_scalar`] is the ground truth both are validated against.
//!
//! # Interleaved layout
//!
//! `packed` holds `[gate_block_0 (144B)][up_block_0 (144B)][gate_1][up_1]…` per
//! output row — the same row-interleaving `fuse_gate_up` already does for Q8_0,
//! so gate and up stream as one contiguous DRAM read per thread. Gate and up
//! keep their own `d`/`dmin`/`scales`; only the layout is interleaved.

use glcore::format::gguf::f16_to_f32;

use crate::kernels::dequant::q4_k::scalar::{decode_scales, dequant_block, BLOCK_BYTES};
use crate::kernels::qdot::q8_k::Q8KActivation;

/// Bytes of one Q4_K weight row of `in_dim` weights.
#[inline]
pub fn row_bytes(in_dim: usize) -> usize {
    in_dim / 256 * BLOCK_BYTES
}

/// Scalar ground truth: `y = silu(gate·x) * (up·x)`.
///
/// Dequantizes gate and up to f32 (via the validated `dequant_block`), dots
/// each against the **dequantized** Q8_K activation, applies an *exact* SiLU
/// (`f32::exp`, not the approximation), multiplies. Every SIMD variant must
/// match this within tolerance.
pub fn fused_swiglu_scalar(gate_row: &[u8], up_row: &[u8], act: &Q8KActivation) -> f32 {
    let n_blocks = gate_row.len() / BLOCK_BYTES;
    let mut g = 0f32;
    let mut u = 0f32;
    let mut blk = [0f32; 256];

    for b in 0..n_blocks {
        let d8 = act.d[b];
        dequant_block(&gate_row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES], &mut blk);
        for (i, &w) in blk.iter().enumerate() {
            g += w * d8 * act.q[b * 256 + i] as f32;
        }
        dequant_block(&up_row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES], &mut blk);
        for (i, &w) in blk.iter().enumerate() {
            u += w * d8 * act.q[b * 256 + i] as f32;
        }
    }

    // Exact SiLU: g * sigmoid(g) = g / (1 + e^-g).
    g / (1.0 + (-g).exp()) * u
}

/// Candidate A — integer domain. Two `row_dot_q8k` (the proven Wave-2 kernel),
/// SiLU inline on the results. This is the Q8_0-fused shape, ported to Q4_K.
///
/// # Safety
/// Caller must ensure AVX2/FMA/F16C, and `act` quantized for the row width.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn fused_swiglu_q8k(gate_row: &[u8], up_row: &[u8], act: &Q8KActivation) -> f32 {
    let g = super::avx2::row_dot_q8k(gate_row, act);
    let u = super::avx2::row_dot_q8k(up_row, act);
    // `fast_exp` is the same vector-poly approximation the Q8_0 fused path uses,
    // so the two SwiGLU paths are numerically identical here.
    g / (1.0 + crate::kernels::fast_exp(-g)) * u
}

/// Candidate B — f32 domain, the literal Wave-4 spec. Dequant gate and up to
/// f32, dot against the dequantized activation, SiLU-mul. Built to be measured,
/// not because it is expected to win: it is the per-element unpack that lost
/// 33% before, now with a polynomial SiLU added on top.
///
/// # Safety
/// Caller must ensure AVX2/FMA/F16C.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn fused_swiglu_f32(gate_row: &[u8], up_row: &[u8], act: &Q8KActivation) -> f32 {
    use std::arch::x86_64::*;

    let n_blocks = gate_row.len() / BLOCK_BYTES;
    let mut gacc = _mm256_setzero_ps();
    let mut uacc = _mm256_setzero_ps();

    for b in 0..n_blocks {
        let d8 = _mm256_set1_ps(act.d[b]);
        dequant_dot_block(gate_row, b, act, d8, &mut gacc);
        dequant_dot_block(up_row, b, act, d8, &mut uacc);
    }

    let g = hsum256(gacc);
    let u = hsum256(uacc);
    // Polynomial SiLU per the spec — on the SCALAR result, so it costs nothing;
    // the expensive part the spec put per-element is the dequant above.
    let sig = crate::kernels::fast_exp(-g);
    g / (1.0 + sig) * u
}

/// One super-block of the f32-domain dot: dequant 256 weights to f32 and FMA
/// them against the dequantized activation, accumulating in `acc`.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn dequant_dot_block(
    row: &[u8],
    b: usize,
    act: &Q8KActivation,
    d8: std::arch::x86_64::__m256,
    acc: &mut std::arch::x86_64::__m256,
) {
    use std::arch::x86_64::*;

    let block = &row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES];
    let d = crate::kernels::qdot::f16_hw(u16::from_le_bytes([block[0], block[1]]));
    let dmin = crate::kernels::qdot::f16_hw(u16::from_le_bytes([block[2], block[3]]));
    let (sc, mn) = decode_scales(&block[4..16]);
    let qs = &block[16..144];
    let lo_mask = _mm256_set1_epi8(0x0F);

    for chunk in 0..4 {
        let j_lo = 2 * chunk;
        // SAFETY: chunk<4 keeps this inside the 128-byte qs region.
        let packed = _mm256_loadu_si256(qs.as_ptr().add(chunk * 32) as *const __m256i);
        let w_lo = _mm256_and_si256(packed, lo_mask);
        let w_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), lo_mask);

        // Dequant coefficients for the two sub-blocks.
        for (half, wbytes, j) in [(0usize, w_lo, j_lo), (1, w_hi, j_lo + 1)] {
            let scale = _mm256_set1_ps(d * sc[j] as f32);
            let min = _mm256_set1_ps(dmin * mn[j] as f32);
            let a_base = b * 256 + j * 32;
            // 32 weights = 4 lanes of 8.
            for k in 0..4 {
                // Extract 8 nibble bytes -> i32 -> f32.
                let bytes8 = extract8(wbytes, k);
                let wf = _mm256_cvtepi32_ps(bytes8);
                // w = d*sc*nibble - dmin*min
                let deq = _mm256_fmsub_ps(scale, wf, min);
                // activation (i8 -> f32) * d8
                let a8 = load8_i8_to_f32(act.q.as_ptr().add(a_base + k * 8));
                let ax = _mm256_mul_ps(a8, d8);
                *acc = _mm256_fmadd_ps(deq, ax, *acc);
                let _ = half;
            }
        }
    }
}

/// Extract lanes `[k*8 .. k*8+8]` of a 32-byte register as i32×8.
#[target_feature(enable = "avx2")]
unsafe fn extract8(v: std::arch::x86_64::__m256i, k: usize) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    let mut tmp = [0u8; 32];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut __m256i, v);
    let mut lanes = [0i32; 8];
    for (l, lane) in lanes.iter_mut().enumerate() {
        *lane = tmp[k * 8 + l] as i32;
    }
    _mm256_loadu_si256(lanes.as_ptr() as *const __m256i)
}

/// Load 8 i8 activations and widen to f32×8.
#[target_feature(enable = "avx2")]
unsafe fn load8_i8_to_f32(p: *const i8) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    let mut lanes = [0i32; 8];
    for (l, lane) in lanes.iter_mut().enumerate() {
        *lane = *p.add(l) as i32;
    }
    _mm256_cvtepi32_ps(_mm256_loadu_si256(lanes.as_ptr() as *const __m256i))
}

/// Horizontal sum of a f32×8 register.
#[target_feature(enable = "avx2")]
unsafe fn hsum256(v: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps::<1>(v);
    let s = _mm_add_ps(lo, hi);
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_shuffle_ps::<0b01>(s, s));
    _mm_cvtss_f32(s)
}

/// `d` and `dmin` of one block as f32 (for tests / callers).
#[inline]
pub fn block_scales(block: &[u8]) -> (f32, f32) {
    (
        f16_to_f32(u16::from_le_bytes([block[0], block[1]])),
        f16_to_f32(u16::from_le_bytes([block[2], block[3]])),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::qdot::q8_k::Q8KActivation;

    fn prng_byte(seed: &mut u64) -> u8 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (*seed >> 33) as u8
    }
    fn prng_f32(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }
    fn half_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 { return sign; }
        sign | ((exp as u16) << 10) | mant
    }
    fn q4k_row(nb: usize, seed: &mut u64) -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..nb {
            v.extend_from_slice(&half_bits(0.02).to_le_bytes());
            v.extend_from_slice(&half_bits(0.01).to_le_bytes());
            for _ in 0..12 { v.push(prng_byte(seed)); }
            for _ in 0..128 { v.push(prng_byte(seed)); }
        }
        v
    }

    fn wide() -> bool {
        matches!(
            crate::simd_strategy::SimdStrategy::detect(),
            crate::simd_strategy::SimdStrategy::Avx2 | crate::simd_strategy::SimdStrategy::Avx512
        )
    }

    /// Both SIMD candidates must match the scalar ground truth. The SwiGLU
    /// nonlinearity makes this stricter than a plain dot: an error in the gate
    /// dot passes through `silu`, so tolerance is on the final product.
    #[test]
    fn both_candidates_match_scalar() {
        if !wide() {
            eprintln!("skip: no wide backend");
            return;
        }
        eprintln!(
            "fused q4k swiglu parity on {:?}, vnni256={}",
            crate::simd_strategy::SimdStrategy::detect(),
            crate::kernels::qdot::has_vnni_256()
        );
        let mut seed = 0x5091u64;
        for &nb in &[1usize, 2, 6] {
            let in_dim = nb * 256;
            let gate = q4k_row(nb, &mut seed);
            let up = q4k_row(nb, &mut seed);
            let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
            let mut act = Q8KActivation::with_capacity(in_dim);
            act.quantize(&x);

            let want = fused_swiglu_scalar(&gate, &up, &act);
            // SAFETY: wide backend confirmed.
            let a = unsafe { fused_swiglu_q8k(&gate, &up, &act) };
            let b = unsafe { fused_swiglu_f32(&gate, &up, &act) };

            let tol = want.abs().max(1.0) * 1e-2;
            assert!((a - want).abs() < tol, "nb={nb} q8k {a} vs scalar {want}");
            assert!((b - want).abs() < tol, "nb={nb} f32 {b} vs scalar {want}");
            // The two SIMD candidates should also agree with each other closely
            // — same math, different domain.
            assert!(
                (a - b).abs() < want.abs().max(1.0) * 2e-2,
                "nb={nb} candidates disagree: q8k {a} vs f32 {b}"
            );
        }
    }
}
