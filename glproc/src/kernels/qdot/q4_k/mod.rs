//! Q4_K integer-dot kernels.
//!
//! See [`scalar`] for the derivation. The short version: Q4_K is affine
//! (`w = d·sc·q − dmin·m`), so the dot splits into an integer term
//! (`vpdpbusd`-able) plus a per-sub-block offset correction that reuses the
//! activation sums the quantizer already computes.

pub mod avx2;
pub mod scalar;

#[cfg(test)]
mod tests {
    use crate::kernels::dequant::q4_k::scalar::{dequant_block, BLOCK_BYTES, BLOCK_NUMEL};
    use crate::kernels::qdot::QuantizedActivation;

    /// Deterministic pseudo-random bytes. No rand dep.
    fn prng(seed: &mut u64) -> u8 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*seed >> 33) as u8
    }

    fn prng_f32(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    /// A synthetic Q4_K row with plausible f16 scales — a random `d`/`dmin`
    /// bit pattern could be inf/NaN and would test nothing.
    fn q4_k_row(n_blocks: usize, seed: &mut u64) -> Vec<u8> {
        let mut row = Vec::with_capacity(n_blocks * BLOCK_BYTES);
        for _ in 0..n_blocks {
            // d ≈ 0.01, dmin ≈ 0.005 — the magnitudes real quantizers produce.
            let d = half_bits(0.01 + 0.02 * (prng(seed) as f32 / 255.0));
            let dmin = half_bits(0.005 + 0.01 * (prng(seed) as f32 / 255.0));
            row.extend_from_slice(&d.to_le_bytes());
            row.extend_from_slice(&dmin.to_le_bytes());
            // 12 bytes of packed 6-bit (scale, min) pairs.
            for _ in 0..12 {
                row.push(prng(seed));
            }
            // 128 bytes of 4-bit quants.
            for _ in 0..128 {
                row.push(prng(seed));
            }
        }
        row
    }

    /// f32 → f16 bits. Only needs to handle the small positive values above.
    fn half_bits(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
        let mant = ((bits >> 13) & 0x3FF) as u16;
        if exp <= 0 {
            return sign;
        }
        sign | ((exp as u16) << 10) | mant
    }

    /// The load-bearing test: the integer-dot kernel must agree with a fully
    /// independent reference — dequantize the row to f32, then dot in f32.
    ///
    /// This is NOT a self-comparison. `dequant_block` is the pre-existing
    /// ground truth used to validate every Q4_K dequant path; if the integer
    /// derivation in `scalar::row_dot` is wrong, this catches it.
    #[test]
    fn integer_dot_matches_dequantized_f32_dot() {
        let mut seed = 0x4B4Bu64;

        for &n_blocks in &[1usize, 2, 7] {
            let in_dim = n_blocks * BLOCK_NUMEL;
            let row = q4_k_row(n_blocks, &mut seed);

            // Reference: dequantize the whole row to f32.
            let mut w = vec![0f32; in_dim];
            for b in 0..n_blocks {
                let mut blk = [0f32; 256];
                dequant_block(&row[b * BLOCK_BYTES..(b + 1) * BLOCK_BYTES], &mut blk);
                w[b * BLOCK_NUMEL..(b + 1) * BLOCK_NUMEL].copy_from_slice(&blk);
            }

            // A real-ish activation, then quantize it the way the runner does.
            let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
            let mut act = QuantizedActivation::with_capacity(in_dim);
            act.quantize(&x);

            // f32 reference dot uses the DEQUANTIZED weights against the
            // QUANTIZED activation — so the only thing under test is the
            // weight-side integer derivation, not activation quantization
            // error (which both paths share).
            let want: f32 = (0..in_dim)
                .map(|i| w[i] * act.scales[i / 32] * act.q[i] as f32)
                .sum();

            let got = super::scalar::row_dot(&row, &act);

            // Both paths sum the same products; they differ only in float
            // association order and in doing the multiply in int vs f32.
            let tol = want.abs().max(1.0) * 1e-4;
            assert!(
                (got - want).abs() < tol,
                "n_blocks={n_blocks}: got {got}, want {want} (diff {})",
                (got - want).abs()
            );
        }
    }

    /// A zero activation must dot to exactly zero, offset term included. If the
    /// `dmin·m·Σa` correction were applied with the wrong sign or without the
    /// activation sum, this would drift off zero.
    #[test]
    fn zero_activation_dots_to_zero() {
        let mut seed = 0xBEEFu64;
        let row = q4_k_row(2, &mut seed);
        let in_dim = 2 * BLOCK_NUMEL;

        let mut act = QuantizedActivation::with_capacity(in_dim);
        act.quantize(&vec![0f32; in_dim]);

        let got = super::scalar::row_dot(&row, &act);
        assert!(got.abs() < 1e-6, "zero activation must dot to 0, got {got}");
    }

    /// The SIMD path must agree with the scalar ground truth on the machine's
    /// actual backend (AVX2 or AVX2+VNNI — both go through `row_dot`, which
    /// branches internally).
    #[test]
    fn simd_matches_scalar() {
        use crate::simd_strategy::SimdStrategy;
        if !matches!(SimdStrategy::detect(), SimdStrategy::Avx2 | SimdStrategy::Avx512) {
            eprintln!("skipping: no wide backend on this machine");
            return;
        }
        eprintln!(
            "q4_k simd parity on {:?}, vnni256={}",
            SimdStrategy::detect(),
            crate::kernels::qdot::has_vnni_256()
        );

        let mut seed = 0xD07u64;
        for &n_blocks in &[1usize, 2, 5, 19] {
            let in_dim = n_blocks * BLOCK_NUMEL;
            let row = q4_k_row(n_blocks, &mut seed);
            let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
            let mut act = QuantizedActivation::with_capacity(in_dim);
            act.quantize(&x);

            let want = super::scalar::row_dot(&row, &act);
            // SAFETY: gated on a wide backend above.
            let got = unsafe { super::avx2::row_dot(&row, &act) };

            // Integer terms are exact; only the f32 accumulation order differs.
            let tol = want.abs().max(1.0) * 1e-5;
            assert!(
                (got - want).abs() < tol,
                "n_blocks={n_blocks}: simd {got}, scalar {want}"
            );
        }
    }

    /// The offset term must actually be doing something. Build a row whose
    /// `dmin` is large, and confirm the result differs from what a
    /// scale-only (no-min) interpretation would give. Guards against silently
    /// dropping the affine term — which would still pass a "close enough"
    /// test on small dmin.
    #[test]
    fn offset_term_is_not_dropped() {
        let mut seed = 0xFACEu64;
        let in_dim = BLOCK_NUMEL;
        let mut row = q4_k_row(1, &mut seed);
        // Force a large dmin (0.5) so the offset term dominates.
        row[2..4].copy_from_slice(&half_bits(0.5).to_le_bytes());

        let x: Vec<f32> = (0..in_dim).map(|_| prng_f32(&mut seed)).collect();
        let mut act = QuantizedActivation::with_capacity(in_dim);
        act.quantize(&x);

        // Reference via full dequant (which definitely includes the min term).
        let mut blk = [0f32; 256];
        dequant_block(&row, &mut blk);
        let want: f32 = (0..in_dim)
            .map(|i| blk[i] * act.scales[i / 32] * act.q[i] as f32)
            .sum();

        let got = super::scalar::row_dot(&row, &act);
        let tol = want.abs().max(1.0) * 1e-4;
        assert!(
            (got - want).abs() < tol,
            "large-dmin row diverged: got {got}, want {want}"
        );
    }
}
