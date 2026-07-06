//! Q5_0 × Q8 activation integer dot, AVX2, with byte-domain unpacking.
//! Validated against `scalar`.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// One Q5_0 row (`n_blocks * 22` bytes) · quantized activation.
///
/// The whole block is unpacked in the byte domain: nibbles land in a 32-byte
/// register (low lane = weights 0..16, high lane = 16..32), and the 32 `qh`
/// bits are expanded to 0x00/0x10 bytes via the shuffle + `cmpeq` bit-test
/// idiom — no per-lane variable shifts, no i32 widening before the dot.
/// The unsigned 5-bit quants then feed `maddubs` directly; the −16 offset
/// is settled per block using the activation's precomputed group sum.
///
/// Two accumulators, alternated per block, keep consecutive blocks off a
/// single 4-cycle FMA dependency chain.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA and F16C, and that `act`
/// was quantized for at least `row.len() / 22 * 32` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let low_mask = _mm256_set1_epi8(0x0F);
    let ones = _mm256_set1_epi16(1);
    // Byte i of the unpacked vector needs bit (i % 8) of qh byte (i / 8).
    // shuffle control replicates each qh byte 8×; the bit mask then selects
    // one bit per byte and cmpeq turns "bit set" into 0xFF → & 0x10.
    #[rustfmt::skip]
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
        2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3,
    );
    #[rustfmt::skip]
    let bits = _mm256_setr_epi8(
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
    );
    let high_bit = _mm256_set1_epi8(0x10);

    let mut acc = [_mm256_setzero_ps(); 2];
    // Same two-chain trick for the scalar offset sum — FP adds are not
    // reassociable, so a single accumulator would serialize 28 subs per row.
    let mut acc_off = [0f32; 2];

    for (j, block) in row.chunks_exact(22).enumerate() {
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]])) * act.scales[j];
        let qh = i32::from_le_bytes([block[2], block[3], block[4], block[5]]);

        // SAFETY: block has 22 bytes (2 + 4 header, 16 nibbles); act.q holds
        // at least (j+1)*32 int8 values per the function contract.
        let nib = _mm_loadu_si128(block.as_ptr().add(6) as *const __m128i);
        // Low lane = low nibbles (weights 0..16), high lane = high nibbles.
        let q4 = _mm256_and_si256(
            _mm256_set_m128i(_mm_srli_epi16::<4>(nib), nib),
            low_mask,
        );
        // Expand the 32 qh bits to a 0x00/0x10 byte per weight.
        let qh_bytes = _mm256_shuffle_epi8(_mm256_set1_epi32(qh), shuf);
        let has_bit = _mm256_cmpeq_epi8(_mm256_and_si256(qh_bytes, bits), bits);
        let q = _mm256_or_si256(q4, _mm256_and_si256(has_bit, high_bit)); // 0..32, unsigned

        let a = _mm256_loadu_si256(act.q.as_ptr().add(j * 32) as *const __m256i);
        // q unsigned (≤31) × a signed: |31·127·2| fits i16 comfortably.
        let p16 = _mm256_maddubs_epi16(q, a);
        let p32 = _mm256_madd_epi16(p16, ones);

        acc[j & 1] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc[j & 1]);
        // The −16 offset: Σ d·d_a·(q−16)·a = (integer dot above) − 16·d·d_a·Σa.
        acc_off[j & 1] -= 16.0 * d * act.sums[j] as f32;
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
    tmp.iter().sum::<f32>() + acc_off[0] + acc_off[1]
}
