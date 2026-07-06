//! Q5_0 × Q8 activation integer dot — ground truth for the AVX2 path.

use glcore::format::gguf::f16_to_f32;

use crate::kernels::qdot::QuantizedActivation;

/// One Q5_0 row (`n_blocks * 22` bytes) · quantized activation.
///
/// Q5_0 weights are `d * (q - 16)` with unsigned `q` in 0..32, so
/// `Σ w·x ≈ d · d_a · (Σ q·a_q − 16·Σ a_q)` — the `Σ a_q` term comes
/// pre-computed from the activation quantizer.
pub fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let mut acc = 0f32;
    for (j, block) in row.chunks_exact(22).enumerate() {
        let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        let aq = &act.q[j * 32..j * 32 + 32];

        let mut idot = 0i32;
        for (i, &byte) in qs.iter().enumerate() {
            // Low nibble = weight i, high nibble = weight i+16; bit i of qh
            // is weight i's 5th bit.
            let lo = ((byte & 0x0F) as u32 | ((qh >> i) & 1) << 4) as i32;
            let hi = ((byte >> 4) as u32 | ((qh >> (i + 16)) & 1) << 4) as i32;
            idot += lo * aq[i] as i32 + hi * aq[i + 16] as i32;
        }
        acc += d * act.scales[j] * (idot - 16 * act.sums[j]) as f32;
    }
    acc
}
