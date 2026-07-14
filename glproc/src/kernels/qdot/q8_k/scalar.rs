//! Scalar Q8_K quantizer — the ground truth.
//!
//! Runs once per matvec over `in_dim` elements and is amortized over every
//! weight row that consumes it (thousands), so a scalar implementation is not
//! on the hot path. The Wave-2 spec asked for AVX2 here; deferred until a
//! profile says the quantizer is measurable at all — the same discipline that
//! keeps every other cold path in this crate scalar.

use super::{Q8KActivation, BLOCK_NUMEL};

/// Quantize `x` into `act`: per 256-block, `d = max|x| / 127`,
/// `q_i = round(x_i / d)`, plus the per-32 sums the min-term needs.
pub fn quantize(act: &mut Q8KActivation, x: &[f32]) {
    debug_assert_eq!(x.len() % BLOCK_NUMEL, 0);
    debug_assert!(x.len() <= act.q.len());
    act.len = x.len();

    for (b, block) in x.chunks_exact(BLOCK_NUMEL).enumerate() {
        let amax = block.iter().fold(0f32, |m, &v| m.max(v.abs()));
        if amax == 0.0 {
            act.d[b] = 0.0;
            act.q[b * BLOCK_NUMEL..(b + 1) * BLOCK_NUMEL].fill(0);
            act.bsums[b * 8..(b + 1) * 8].fill(0);
            continue;
        }
        let d = amax / 127.0;
        let inv = 127.0 / amax;
        act.d[b] = d;

        for (j, sub) in block.chunks_exact(32).enumerate() {
            let mut sum = 0i32;
            for (i, &v) in sub.iter().enumerate() {
                let q = (v * inv).round() as i32;
                // round() of |v|<=amax scaled by 127/amax stays in [-127,127];
                // clamp anyway so a rounding edge can't wrap the i8.
                let q = q.clamp(-127, 127);
                act.q[b * BLOCK_NUMEL + j * 32 + i] = q as i8;
                sum += q;
            }
            act.bsums[b * 8 + j] = sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prng(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0
    }

    /// Quantize -> dequantize round trip. One int8 step of a max-127 scale is
    /// the theoretical error bound; assert against it, not a loose blanket.
    #[test]
    fn round_trip_error_within_one_quant_step() {
        let mut seed = 0x08AAu64;
        let n = 512; // two blocks
        let x: Vec<f32> = (0..n).map(|_| prng(&mut seed)).collect();
        let mut act = Q8KActivation::with_capacity(n);
        act.quantize(&x);

        for b in 0..n / BLOCK_NUMEL {
            let d = act.d[b];
            let amax = x[b * 256..(b + 1) * 256]
                .iter()
                .fold(0f32, |m, &v| m.max(v.abs()));
            // Half a quantization step, plus float slack.
            let bound = amax / 127.0 * 0.5 + 1e-6;
            for i in 0..BLOCK_NUMEL {
                let got = d * act.q[b * BLOCK_NUMEL + i] as f32;
                let want = x[b * BLOCK_NUMEL + i];
                assert!(
                    (got - want).abs() <= bound,
                    "b={b} i={i}: {got} vs {want} (bound {bound})"
                );
            }
        }
    }

    /// `bsums` must equal the actual per-32 sums of the stored quants — the
    /// min-term consumes them blind, so a drift here corrupts every dot.
    #[test]
    fn bsums_match_stored_quants() {
        let mut seed = 0xB5u64;
        let n = 768; // three blocks
        let x: Vec<f32> = (0..n).map(|_| prng(&mut seed)).collect();
        let mut act = Q8KActivation::with_capacity(n);
        act.quantize(&x);

        for b in 0..n / BLOCK_NUMEL {
            for j in 0..8 {
                let want: i32 = act.q[b * 256 + j * 32..b * 256 + j * 32 + 32]
                    .iter()
                    .map(|&q| q as i32)
                    .sum();
                assert_eq!(act.bsums[b * 8 + j], want, "b={b} sub={j}");
            }
        }
    }

    /// An all-zero block must quantize to exact zeros with zero scale, not NaN
    /// from a 0/0.
    #[test]
    fn zero_block_is_exact() {
        let mut act = Q8KActivation::with_capacity(256);
        act.quantize(&[0.0; 256]);
        assert_eq!(act.d[0], 0.0);
        assert!(act.q[..256].iter().all(|&q| q == 0));
        assert!(act.bsums[..8].iter().all(|&s| s == 0));
    }

    /// The extremes must hit ±127 exactly — an off-by-one in the scale
    /// direction shows up here as ±126 or a clamp artifact.
    #[test]
    fn extremes_reach_full_range() {
        let mut x = [0.5f32; 256];
        x[3] = 2.0;
        x[7] = -2.0;
        let mut act = Q8KActivation::with_capacity(256);
        act.quantize(&x);
        assert_eq!(act.q[3], 127);
        assert_eq!(act.q[7], -127);
    }
}
