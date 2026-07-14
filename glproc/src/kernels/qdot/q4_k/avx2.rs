//! Q4_K × Q8 activation integer dot, AVX2 (+ VNNI when available).
//!
//! See [`super::scalar`] for the affine derivation. This is the same algorithm
//! with the sub-block dots vectorized.
//!
//! # Nibble unpack
//!
//! Each 32-byte `qs` chunk holds two sub-blocks: low nibbles are sub-block
//! `2c`, high nibbles are sub-block `2c+1`. One 256-bit load gives both, and a
//! mask + shift splits them — no shuffle table needed, because GGML's layout
//! puts each sub-block's 32 weights in the same nibble position across the 32
//! bytes (it is NOT interleaved per weight).
//!
//! Q4_K quants are **unsigned 0..15**, which is exactly what `vpdpbusd` wants
//! for its unsigned operand — so unlike Q8_0 there is no sign trick to apply.
//! The activation is the signed operand.
//!
//! # Why the accumulator stays in a register
//!
//! The first version of this kernel horizontally summed each sub-block's i32
//! lanes down to a scalar, converted to f32, and multiplied by the scale — 8
//! horizontal sums per 256-weight super-block. A horizontal sum is three
//! shuffle+add stages, and doing it per sub-block put the kernel at **7.2 GB/s**
//! against the 23.0 GB/s the Q8_0 path sustains: it read 1.89x fewer bytes yet
//! ran 1.7x slower per MAC. Compute-bound, not bandwidth-bound — the byte
//! saving was spent on unpack overhead and then some.
//!
//! The fix is the shape Q8_0 already uses: FMA each sub-block's i32 lanes
//! straight into an **f32 YMM accumulator**, scaled by a broadcast of
//! `d·sc_j·s_j`, and horizontally sum **once per row** instead of 8x per block.
//! Same for the affine offset term, which accumulates as a plain scalar (it is
//! one f32 multiply-add per sub-block, not a vector op).

use std::arch::x86_64::*;

use crate::kernels::dequant::q4_k::scalar::{decode_scales, scale_min, BLOCK_BYTES};
use crate::kernels::qdot::q8_k::Q8KActivation;
use crate::kernels::qdot::{f16_hw, has_vnni_256, QuantizedActivation};

/// One Q4_K row · **Q8_K** activation, AVX2 (+VNNI), integer-domain sub-block
/// scaling.
///
/// See [`super::scalar::row_dot_q8k`] for the derivation. The vector trick
/// that distinguishes this from the per-32-scale kernel that lost 33%:
///
/// - **AVX2 path:** `maddubs` gives i16 pair-sums (≤ 3810, no overflow), and
///   `madd_epi16(p16, set1(sc))` **fuses the 6-bit sub-block scale into the
///   16→32 widening step** — the scale costs zero extra instructions.
///   Per-lane bound: 2·3810·63 ≈ 480K; 8 sub-blocks per super-block ≈ 3.84M
///   per lane; a 143k-wide row would be needed to overflow i32.
/// - **VNNI path:** `vpdpbusd` goes straight to i32, so the scale is a
///   `mullo_epi32` afterwards.
/// - Per super-block, the only float work is ONE `cvtepi32_ps` + ONE broadcast
///   FMA (for `d·d8`) plus one scalar FMA for the min term. The failed kernel
///   did 8 of each.
///
/// # Safety
/// Caller must ensure AVX2, FMA and F16C, and that `act` was quantized for at
/// least `row.len() / 144 * 256` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot_q8k(row: &[u8], act: &Q8KActivation) -> f32 {
    if has_vnni_256() {
        row_dot_q8k_inner::<true>(row, act)
    } else {
        row_dot_q8k_inner::<false>(row, act)
    }
}

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn row_dot_q8k_inner<const VNNI: bool>(row: &[u8], act: &Q8KActivation) -> f32 {
    let lo_mask = _mm256_set1_epi8(0x0F);
    // f32 accumulator lanes across the whole row; one hsum at the end.
    let mut accf = _mm256_setzero_ps();
    // Min-term accumulates as a scalar — it is 8 integer multiplies per
    // super-block on precomputed sums, nothing to vectorize.
    let mut offset = 0f32;

    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        // SAFETY: block is 144 bytes; f16_hw needs F16C (enabled above).
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_hw(u16::from_le_bytes([block[2], block[3]]));
        let (sc, mn) = decode_scales(&block[4..16]);
        let d8 = act.d[b];

        // Integer accumulator for THIS super-block (reset per block because
        // d·d8 differs per block). Two chains for ILP across the 4 chunks.
        let mut isum = [_mm256_setzero_si256(); 2];

        for chunk in 0..4 {
            let j_lo = 2 * chunk;
            // SAFETY: qs starts at byte 16; chunk < 4 keeps the load inside
            // the 144-byte block.
            let packed =
                _mm256_loadu_si256(block.as_ptr().add(16 + chunk * 32) as *const __m256i);
            let w_lo = _mm256_and_si256(packed, lo_mask);
            let w_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), lo_mask);

            // SAFETY: act quantized for n_blocks*256 elements (contract).
            let a_lo = _mm256_loadu_si256(
                act.q.as_ptr().add(b * 256 + j_lo * 32) as *const __m256i
            );
            let a_hi = _mm256_loadu_si256(
                act.q.as_ptr().add(b * 256 + (j_lo + 1) * 32) as *const __m256i
            );

            if VNNI {
                // dpbusd lands in i32; scale with a 32-bit multiply.
                let p_lo = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_lo, a_lo);
                let p_hi = _mm256_dpbusd_epi32(_mm256_setzero_si256(), w_hi, a_hi);
                isum[0] = _mm256_add_epi32(
                    isum[0],
                    _mm256_mullo_epi32(p_lo, _mm256_set1_epi32(sc[j_lo])),
                );
                isum[1] = _mm256_add_epi32(
                    isum[1],
                    _mm256_mullo_epi32(p_hi, _mm256_set1_epi32(sc[j_lo + 1])),
                );
            } else {
                // maddubs -> i16 pair sums (<= 3810), then madd BY THE SCALE:
                // the widening multiply-add does the scaling for free.
                let p16_lo = _mm256_maddubs_epi16(w_lo, a_lo);
                let p16_hi = _mm256_maddubs_epi16(w_hi, a_hi);
                isum[0] = _mm256_add_epi32(
                    isum[0],
                    _mm256_madd_epi16(p16_lo, _mm256_set1_epi16(sc[j_lo] as i16)),
                );
                isum[1] = _mm256_add_epi32(
                    isum[1],
                    _mm256_madd_epi16(p16_hi, _mm256_set1_epi16(sc[j_lo + 1] as i16)),
                );
            }
        }

        // ONE float scale for the whole super-block's positive term.
        let block_i = _mm256_add_epi32(isum[0], isum[1]);
        accf = _mm256_fmadd_ps(
            _mm256_set1_ps(d * d8),
            _mm256_cvtepi32_ps(block_i),
            accf,
        );

        // Min term from precomputed sums.
        let mut msum = 0i32;
        for j in 0..8 {
            msum += mn[j] * act.bsums[b * 8 + j];
        }
        offset += dmin * d8 * msum as f32;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), accf);
    tmp.iter().sum::<f32>() - offset
}

/// `Σ q_i · a_i` over 32 unsigned-4-bit weights × 32 signed int8 activations,
/// left in i32 lanes (NOT horizontally summed).
#[target_feature(enable = "avx2")]
unsafe fn dot32_lanes(w: __m256i, a: __m256i) -> __m256i {
    // maddubs: unsigned × signed -> i16 pair sums. Weights are 0..15 and
    // activations are int8, so |15 · 127 · 2| = 3810 — nowhere near i16
    // overflow, unlike Q8_0 where the bound has to be justified.
    let p16 = _mm256_maddubs_epi16(w, a);
    _mm256_madd_epi16(p16, _mm256_set1_epi16(1))
}

/// VNNI variant: one `vpdpbusd` replaces the maddubs+madd pair.
#[target_feature(enable = "avx2", enable = "avx512vl", enable = "avx512vnni")]
unsafe fn dot32_lanes_vnni(w: __m256i, a: __m256i) -> __m256i {
    _mm256_dpbusd_epi32(_mm256_setzero_si256(), w, a)
}

/// One Q4_K row (`n_blocks * 144` bytes) · quantized activation.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA and F16C, and that `act` was
/// quantized for at least `row.len() / 144 * 256` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    // Branch once per row, not per block. `has_vnni_256()` is a cached OnceLock
    // load, but hoisting it also lets the compiler specialize each loop body.
    if has_vnni_256() {
        row_dot_inner::<true>(row, act)
    } else {
        row_dot_inner::<false>(row, act)
    }
}

#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
unsafe fn row_dot_inner<const VNNI: bool>(row: &[u8], act: &QuantizedActivation) -> f32 {
    let lo_mask = _mm256_set1_epi8(0x0F);

    // Two f32 accumulators, alternated per chunk: a single accumulator would
    // serialize on the FMA's 4-cycle latency across 4 dependent chunks per
    // block. Two chains let consecutive chunks overlap in the out-of-order
    // window — the same reason the Q8_0 kernel alternates.
    let mut acc = [_mm256_setzero_ps(); 2];
    // The affine offset (`−dmin·m_j·s_j·Σa_j`) is one scalar FMA per sub-block.
    // Vectorizing it would cost more shuffling than it saves.
    let mut offset = 0f32;

    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        // SAFETY: block is 144 bytes; f16_hw requires F16C, enabled above.
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_hw(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];

        // Unpack all 8 (scale, min) pairs ONCE per super-block, and premultiply
        // them into their final f32 coefficients here.
        //
        // `scale_min` branches on `j < 4` and does ~10 scalar bit ops. Calling
        // it inside the chunk loop put 8 branchy scalar sequences on the
        // critical path of every 256 weights, while the vector side is only
        // ~4 loads and ~12 SIMD ops — the scalar bookkeeping, not the SIMD,
        // was the bottleneck. Hoisting it lets the branch resolve 8x per block
        // instead of interleaving with the FMA chain.
        let mut coef = [0f32; 8]; // d · sc_j
        let mut moff = [0f32; 8]; // dmin · m_j
        for j in 0..8 {
            let (sc, m) = scale_min(j, scales);
            coef[j] = d * sc as f32;
            moff[j] = dmin * m as f32;
        }

        for chunk in 0..4 {
            // SAFETY: qs starts at byte 16; chunk < 4, so this reads bytes
            // 16+chunk*32 .. 48+chunk*32 — inside the 144-byte block.
            let packed = _mm256_loadu_si256(block.as_ptr().add(16 + chunk * 32) as *const __m256i);
            // Low nibbles -> sub-block 2c; high nibbles -> sub-block 2c+1.
            let w_lo = _mm256_and_si256(packed, lo_mask);
            let w_hi = _mm256_and_si256(_mm256_srli_epi16::<4>(packed), lo_mask);

            let g_lo = b * 8 + 2 * chunk;
            let g_hi = g_lo + 1;
            // SAFETY: `act` is quantized for n_blocks*256 elements, so groups
            // g_lo and g_hi are in bounds (g_hi < n_blocks*8).
            let a_lo = _mm256_loadu_si256(act.q.as_ptr().add(g_lo * 32) as *const __m256i);
            let a_hi = _mm256_loadu_si256(act.q.as_ptr().add(g_hi * 32) as *const __m256i);

            let (p_lo, p_hi) = if VNNI {
                (dot32_lanes_vnni(w_lo, a_lo), dot32_lanes_vnni(w_hi, a_hi))
            } else {
                (dot32_lanes(w_lo, a_lo), dot32_lanes(w_hi, a_hi))
            };

            // Scale and accumulate IN THE VECTOR DOMAIN — no horizontal sum
            // here. The i32 lanes convert to f32 and FMA straight into `acc`.
            let s_lo = act.scales[g_lo];
            let s_hi = act.scales[g_hi];
            let j_lo = 2 * chunk;
            acc[0] = _mm256_fmadd_ps(
                _mm256_set1_ps(coef[j_lo] * s_lo),
                _mm256_cvtepi32_ps(p_lo),
                acc[0],
            );
            acc[1] = _mm256_fmadd_ps(
                _mm256_set1_ps(coef[j_lo + 1] * s_hi),
                _mm256_cvtepi32_ps(p_hi),
                acc[1],
            );

            // Affine offset: −dmin·m_j·s_j·Σ(a_j). `act.sums` is precomputed by
            // the quantizer, so this needs no vector work at all.
            offset += moff[j_lo] * s_lo * act.sums[g_lo] as f32
                + moff[j_lo + 1] * s_hi * act.sums[g_hi] as f32;
        }
    }

    // One horizontal sum for the whole row, not 8 per super-block.
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
    tmp.iter().sum::<f32>() - offset
}
