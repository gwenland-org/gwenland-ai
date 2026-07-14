//! Q4_K × Q8 activation integer dot — ground truth for the SIMD paths.
//!
//! # Why this kernel exists
//!
//! Q4_K was the **only** quantized format without an integer-dot kernel, so the
//! loader repacked it to Q8_0 at load time. That threw away the entire size
//! advantage: Q4_K is 144 bytes per 256 weights (0.5625 B/weight) against
//! Q8_0's 34 per 32 (1.0625 B/weight). On a Q4_K model the repack inflates
//! per-token DRAM traffic by **1.70x** (measured on Qwen2.5-1.5B-q4_k_m:
//! 1111 MB native vs 1889 MB repacked), and decode is bandwidth-bound.
//!
//! # The math
//!
//! Q4_K is **affine**, not purely scaled: each sub-block of 32 weights has its
//! own scale *and* its own min:
//!
//! ```text
//!   w_i = (d · sc_j) · q_i  −  (dmin · m_j)          for i in sub-block j
//! ```
//!
//! The activation is `x_i = s_j · a_i` (int8 `a`, one f32 scale per 32-group).
//! Substituting and splitting the sum:
//!
//! ```text
//!   Σ w_i·x_i = Σ_j [ d·sc_j·s_j · (Σ_i q_i·a_i) ]   ← integer dot
//!             − Σ_j [ dmin·m_j·s_j · (Σ_i a_i)   ]   ← offset correction
//!                                       └────────┘
//!                                   act.sums[j], already computed
//! ```
//!
//! Both terms stay in the integer domain. **No dequantization to f32 anywhere**
//! — which is the whole point: the weights never expand in RAM or in the read
//! path.
//!
//! This is the same shape as the existing Q5_0 kernel (`d · (q − 16)`, which
//! uses `act.sums` for its constant −16 offset). Q4_K differs only in that the
//! offset is **per sub-block** (`dmin · m_j`) rather than a constant, so the
//! correction is applied per sub-block instead of per block.
//!
//! # Alignment
//!
//! Q4_K's sub-block is 32 weights and `QuantizedActivation` groups are also 32,
//! so sub-block `j` of super-block `b` maps exactly onto activation group
//! `b*8 + j`. No regrouping, no partial groups.

use glcore::format::gguf::f16_to_f32;

use crate::kernels::dequant::q4_k::scalar::{decode_scales, scale_min, BLOCK_BYTES};
use crate::kernels::qdot::q8_k::Q8KActivation;
use crate::kernels::qdot::QuantizedActivation;

/// One Q4_K row · **Q8_K** activation — the super-block-aligned variant.
///
/// Q8_K has ONE f32 scale per 256 elements, aligned with Q4_K's super-block,
/// so the 8 sub-block scales become **integer multipliers** and the float math
/// collapses to two multiplies per super-block:
///
/// ```text
///   acc += d·d8 · (Σ_j sc_j · idot_j)  −  dmin·d8 · (Σ_j m_j · bsum_j)
/// ```
///
/// This is the structural difference from [`row_dot`] (the per-32-scale
/// variant that lost 33% end-to-end): there the per-group activation scales
/// forced 8 float-scaled accumulations per super-block; here both inner sums
/// stay integer until one final scale.
///
/// Overflow, checked: `idot_j <= 32·15·127 = 60 960`; `sc_j <= 63`;
/// `Σ_j sc_j·idot_j <= 8·63·60 960 ≈ 30.7M` — comfortably inside i32.
pub fn row_dot_q8k(row: &[u8], act: &Q8KActivation) -> f32 {
    let mut acc = 0f32;

    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let (sc, mn) = decode_scales(&block[4..16]);
        let qs = &block[16..144];
        let d8 = act.d[b];

        let mut isum = 0i32;
        for chunk in 0..4 {
            let j_lo = 2 * chunk;
            let q = &qs[chunk * 32..chunk * 32 + 32];
            let a_lo = &act.q[b * 256 + j_lo * 32..b * 256 + j_lo * 32 + 32];
            let a_hi = &act.q[b * 256 + (j_lo + 1) * 32..b * 256 + (j_lo + 1) * 32 + 32];

            let mut idot_lo = 0i32;
            let mut idot_hi = 0i32;
            for (i, &byte) in q.iter().enumerate() {
                idot_lo += (byte & 0x0F) as i32 * a_lo[i] as i32;
                idot_hi += (byte >> 4) as i32 * a_hi[i] as i32;
            }
            isum += sc[j_lo] * idot_lo + sc[j_lo + 1] * idot_hi;
        }

        // Min term: entirely from the precomputed activation sums.
        let mut msum = 0i32;
        for j in 0..8 {
            msum += mn[j] * act.bsums[b * 8 + j];
        }

        acc += d * d8 * isum as f32 - dmin * d8 * msum as f32;
    }
    acc
}

/// One Q4_K row (`n_blocks * 144` bytes) · quantized activation.
///
/// `act` must be quantized for at least `row.len() / 144 * 256` elements.
pub fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let mut acc = 0f32;

    for (b, block) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let qs = &block[16..144];

        // 4 chunks of 32 qs bytes; each chunk carries TWO sub-blocks of 32
        // weights — low nibbles are sub-block 2c, high nibbles are sub-block
        // 2c+1. This is GGML's order: NOT low/high interleaved per weight.
        for chunk in 0..4 {
            let (sc_lo, m_lo) = scale_min(2 * chunk, scales);
            let (sc_hi, m_hi) = scale_min(2 * chunk + 1, scales);
            let q = &qs[chunk * 32..chunk * 32 + 32];

            // Activation groups for these two sub-blocks.
            let g_lo = b * 8 + 2 * chunk;
            let g_hi = g_lo + 1;
            let a_lo = &act.q[g_lo * 32..g_lo * 32 + 32];
            let a_hi = &act.q[g_hi * 32..g_hi * 32 + 32];

            let mut idot_lo = 0i32;
            let mut idot_hi = 0i32;
            for (i, &byte) in q.iter().enumerate() {
                idot_lo += (byte & 0x0F) as i32 * a_lo[i] as i32;
                idot_hi += (byte >> 4) as i32 * a_hi[i] as i32;
            }

            // w·x = (d·sc)·s·Σ(q·a) − (dmin·m)·s·Σ(a)
            acc += d * sc_lo as f32 * act.scales[g_lo] * idot_lo as f32
                - dmin * m_lo as f32 * act.scales[g_lo] * act.sums[g_lo] as f32;
            acc += d * sc_hi as f32 * act.scales[g_hi] * idot_hi as f32
                - dmin * m_hi as f32 * act.scales[g_hi] * act.sums[g_hi] as f32;
        }
    }
    acc
}
