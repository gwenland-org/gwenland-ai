//! Q8_0 × Q8 activation integer dot, AVX2. Validated against `scalar`.

use std::arch::x86_64::*;

use crate::kernels::qdot::{f16_hw, QuantizedActivation};

/// One Q8_0 row (`n_blocks * 34` bytes) · quantized activation.
///
/// `maddubs` needs one unsigned operand, so the standard sign trick applies:
/// `|w| ⊗ (a * sign(w))` — 32 multiply-adds per instruction pair.
///
/// Two accumulators, alternated per block: a single accumulator serializes
/// on the 4-cycle FMA latency (28 dependent FMAs on a 896-wide row); two
/// chains let consecutive blocks overlap in the out-of-order window.
///
/// # Safety
/// Caller must ensure the CPU supports AVX2, FMA and F16C, and that `act`
/// was quantized for at least `row.len() / 34 * 32` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot(row: &[u8], act: &QuantizedActivation) -> f32 {
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); 2];

    for (j, block) in row.chunks_exact(34).enumerate() {
        // Pull the stream ~16 blocks ahead (544 B ≈ 8.5 cache lines, into
        // the next row — rows are contiguous). The hardware prefetcher
        // stalls at 4 KiB page boundaries; software prefetch bridges them.
        // Measured +2 tok/s on the i3-1115G4 (A/B'd both ways).
        // SAFETY: prefetch is a hint — an out-of-bounds address past the
        // tensor's end is allowed and simply does nothing.
        _mm_prefetch::<_MM_HINT_T0>(block.as_ptr().add(544) as *const i8);
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]])) * act.scales[j];

        // SAFETY: block has 34 bytes (2 header + 32 quants); act.q holds at
        // least (j+1)*32 int8 values per the function contract.
        let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
        let a = _mm256_loadu_si256(act.q.as_ptr().add(j * 32) as *const __m256i);

        let w_abs = _mm256_sign_epi8(w, w); // |w|, unsigned operand
        let a_signed = _mm256_sign_epi8(a, w); // a * sign(w)
        // i16 pair sums: |w|·a·sign(w) = w·a. Max |63·127·2| well in range.
        let p16 = _mm256_maddubs_epi16(w_abs, a_signed);
        let p32 = _mm256_madd_epi16(p16, ones);

        acc[j & 1] = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(p32), acc[j & 1]);
    }

    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
    tmp.iter().sum()
}

/// One Q8_0 row · `G` Q8 activations at once — the batched-prefill inner
/// kernel (see the VNNI variant for the rationale): the weight block load,
/// sign prep and f16 scale conversion are shared across all `G`
/// activations, and `G` accumulator chains keep the FMA pipe busy.
///
/// # Safety
/// Same CPU-feature contract as [`row_dot`]; every activation must be
/// quantized for at least `row.len() / 34 * 32` elements.
#[target_feature(enable = "avx2", enable = "fma", enable = "f16c")]
pub unsafe fn row_dot_xn<const G: usize>(
    row: &[u8],
    acts: [&QuantizedActivation; G],
) -> [f32; G] {
    let ones = _mm256_set1_epi16(1);
    let mut acc = [_mm256_setzero_ps(); G];

    for (j, block) in row.chunks_exact(34).enumerate() {
        // SAFETY: prefetch is a hint; past-the-end addresses are harmless.
        _mm_prefetch::<_MM_HINT_T0>(block.as_ptr().add(544) as *const i8);
        let d = f16_hw(u16::from_le_bytes([block[0], block[1]]));

        // SAFETY: block has 34 bytes (2 header + 32 quants); each act.q
        // holds at least (j+1)*32 int8 values per the function contract.
        let w = _mm256_loadu_si256(block.as_ptr().add(2) as *const __m256i);
        let w_abs = _mm256_sign_epi8(w, w);

        for g in 0..G {
            let a = _mm256_loadu_si256(acts[g].q.as_ptr().add(j * 32) as *const __m256i);
            let a_signed = _mm256_sign_epi8(a, w);
            let p16 = _mm256_maddubs_epi16(w_abs, a_signed);
            let p32 = _mm256_madd_epi16(p16, ones);
            acc[g] = _mm256_fmadd_ps(
                _mm256_set1_ps(d * acts[g].scales[j]),
                _mm256_cvtepi32_ps(p32),
                acc[g],
            );
        }
    }

    let mut out = [0f32; G];
    let mut tmp = [0f32; 8];
    for g in 0..G {
        _mm256_storeu_ps(tmp.as_mut_ptr(), acc[g]);
        out[g] = tmp.iter().sum();
    }
    out
}
