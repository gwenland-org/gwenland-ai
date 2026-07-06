//! Q8_0 × Q8 activation integer dot — ground truth for the AVX2 path.

use glcore::format::gguf::f16_to_f32;

use crate::kernels::qdot::QuantizedActivation;

/// One Q8_0 row (`n_blocks * 34` bytes) · quantized activation.
/// Per block: `d_w * d_a * Σ w_q[i] * a_q[i]` — the inner sum is exact
/// integer arithmetic; only the two scales are floating point.
pub fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let mut acc = 0f32;
    for (j, block) in row.chunks_exact(34).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let aq = &act.q[j * 32..j * 32 + 32];
        let mut idot = 0i32;
        for (&w, &a) in block[2..34].iter().zip(aq) {
            idot += (w as i8) as i32 * a as i32;
        }
        acc += d * act.scales[j] * idot as f32;
    }
    acc
}
