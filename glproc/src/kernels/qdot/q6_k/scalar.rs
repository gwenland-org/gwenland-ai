//! Q6_K × Q8 activation integer dot — ground truth for the AVX2 path.

use glcore::format::gguf::f16_to_f32;

use crate::kernels::qdot::QuantizedActivation;

/// One Q6_K row (`n_blocks * 210` bytes) · quantized activation.
///
/// Q6_K weights are `d * sc[sub] * (q - 32)` with a signed i8 scale per 16
/// weights, so per 16-weight sub-block:
/// `Σ w·x ≈ d · d_a · sc · (Σ q·a_q − 32·Σ a_q)` — the per-16 activation
/// sums come pre-computed from the quantizer. Weight groups follow GGML's
/// interleaved layout: within each 128-weight half, `ql` byte `l` feeds
/// weights `l` and `l+64`, `ql` byte `l+32` feeds `l+32` and `l+96`, and
/// `qh[l]` packs the 2 high bits of all four.
pub fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let mut acc = 0f32;
    for (jb, block) in row.chunks_exact(210).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));

        for half in 0..2 {
            let ql = &block[half * 64..half * 64 + 64];
            let qh = &block[128 + half * 32..128 + half * 32 + 32];
            let sc = &block[192 + half * 8..192 + half * 8 + 8];
            let base = jb * 256 + half * 128; // activation element index

            // Four 32-weight groups (g), each split into two 16-weight
            // sub-blocks (s) with scale sc[2g + s].
            let mut idot = [[0i32; 2]; 4];
            for l in 0..32 {
                let s = l / 16;
                let q1 = ((ql[l] & 0x0F) | ((qh[l] & 0x03) << 4)) as i32;
                let q2 = ((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 0x03) << 4)) as i32;
                let q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 0x03) << 4)) as i32;
                let q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 0x03) << 4)) as i32;
                idot[0][s] += q1 * act.q[base + l] as i32;
                idot[1][s] += q2 * act.q[base + 32 + l] as i32;
                idot[2][s] += q3 * act.q[base + 64 + l] as i32;
                idot[3][s] += q4 * act.q[base + 96 + l] as i32;
            }

            for g in 0..4 {
                let d_a = act.scales[(base + g * 32) / 32];
                let mut sub = 0f32;
                for s in 0..2 {
                    let scale = sc[2 * g + s] as i8 as f32;
                    let a_sum = act.sums16[(base + g * 32) / 16 + s];
                    sub += scale * (idot[g][s] - 32 * a_sum) as f32;
                }
                acc += d * d_a * sub;
            }
        }
    }
    acc
}
