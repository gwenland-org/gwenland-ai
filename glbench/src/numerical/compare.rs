//! Bit-profile divergence between two tensors.
//!
//! # The question this answers
//!
//! Two quantisations of the same weight both show residual KL against the f32
//! original. Is that the scheme doing its job, or a bug?
//!
//! # Two kinds of defect, two kinds of axis
//!
//! The distribution axes (exponent L1, bit fractions, mantissa entropy) see what
//! values a tensor holds. A quantisation scheme registers on them clearly (GQ4A
//! vs f32: exponent L1 0.40, entropy -4.29 bits), and so does a defect that moves
//! values across a scale boundary.
//!
//! They are blind to order. Measured 2026-08-20 by
//! `glbench/examples/bitprof_quant_divergence.rs`, with the real GQ4A encoder and
//! dequant kernel: a correct decode and a nibble-swapped one score exactly zero
//! on all of them, while their MAE is 14x the scheme's own residual. Swapping the
//! two 4-bit codes in a byte exchanges weights that share a scale, so the decoded
//! multiset is unchanged, and those statistics are permutation-invariant by
//! construction. That is the Q6_K bug class, which corrupted every layer.
//!
//! [`VLBitDivergence::order_rms`] closes that. It compares the two tensors'
//! position-signed sketches, which estimates the RMS element-wise difference
//! between them without keeping either tensor. Both axes are needed: a defect
//! that only reorders shows on this one alone.
//!

use crate::numerical::bitprof::VLBitProfile;

/// How two bit profiles differ. Deltas are `b - a` throughout.
#[derive(Debug, Clone)]
pub struct VLBitDivergence {
    /// Per bit position: `b.bit_set_fraction[i] - a.bit_set_fraction[i]`.
    /// Positive means more bits set in `b`. This is the axis a structured
    /// bug shows up on.
    pub bit_fraction_delta: [f64; 32],

    /// L1 distance between the two exponent distributions, count-invariant.
    ///
    /// Range [0, 2]: 0 for identical distributions, 2 for disjoint ones.
    pub exponent_l1: f64,

    /// `b.mantissa_entropy_bits - a.mantissa_entropy_bits`, in bits. `None`
    /// when either side declined its mantissa map (D-12), because a delta
    /// against a value that was never computed is not a small difference — it
    /// is no difference at all, and must not read as zero.
    pub mantissa_entropy_delta: Option<f64>,

    /// The largest absolute per-position delta, and where it is. Convenience
    /// for the CLI summary; a reader scanning 32 numbers wants the outlier
    /// named.
    pub max_bit_delta: f64,
    /// Bit position of [`Self::max_bit_delta`].
    pub max_bit_position: usize,

    /// Estimated RMS element-wise difference between the two tensors, from
    /// their order sketches. The axis that sees a reordering. `None` for tensors
    /// of different lengths, which have no element-wise difference.
    pub order_rms: Option<f64>,
}

/// Compare two bit profiles.
pub fn compare(a: &VLBitProfile, b: &VLBitProfile) -> VLBitDivergence {
    let mut bit_fraction_delta = [0.0f64; 32];
    let mut max_bit_delta = 0.0f64;
    let mut max_bit_position = 0usize;
    for (i, slot) in bit_fraction_delta.iter_mut().enumerate() {
        let delta = b.bit_set_fraction[i] - a.bit_set_fraction[i];
        *slot = delta;
        if delta.abs() > max_bit_delta.abs() {
            max_bit_delta = delta;
            max_bit_position = i;
        }
    }

    let mantissa_entropy_delta = match (a.mantissa_entropy_bits, b.mantissa_entropy_bits) {
        (Some(ea), Some(eb)) => Some(eb - ea),
        _ => None,
    };

    VLBitDivergence {
        bit_fraction_delta,
        exponent_l1: exponent_l1(a, b),
        mantissa_entropy_delta,
        max_bit_delta,
        max_bit_position,
        order_rms: order_rms(a, b),
    }
}

/// The sketches' distance per lane estimates the L2 distance between the tensors;
/// dividing by the element count's root makes that an RMS.
fn order_rms(a: &VLBitProfile, b: &VLBitProfile) -> Option<f64> {
    if a.count != b.count || a.count == 0 {
        return None;
    }
    let lanes = a.order_sketch.len() as f64;
    let squares: f64 =
        a.order_sketch.iter().zip(&b.order_sketch).map(|(x, y)| (x - y) * (x - y)).sum();
    Some((squares / lanes / a.count as f64).sqrt())
}

/// L1 distance between two exponent histograms, each normalised to a
/// distribution first.
///
/// Normalising **each** histogram by **its own** count is what makes this
/// count-invariant, which is the property the field is for: two tensors of
/// different sizes drawn from the same distribution must score 0.
///
/// Deviation from the Wave 2 brief, recorded because it changes a number: the
/// brief writes this as `sum(|a[i] - b[i]|) / max(a.count, b.count)`, which is
/// not count-invariant despite the accompanying note saying it is. Under that
/// formula a 100-element and a 200-element tensor of the identical constant
/// score 0.5 rather than 0 — it reports a difference between two identical
/// distributions purely because one is longer. Tensor sizes genuinely differ
/// across the comparisons this exists for, so the stated property was kept and
/// the formula corrected.
fn exponent_l1(a: &VLBitProfile, b: &VLBitProfile) -> f64 {
    if a.count == 0 || b.count == 0 {
        return 0.0;
    }
    let (na, nb) = (a.count as f64, b.count as f64);
    (0..256)
        .map(|i| {
            let pa = a.exponent_histogram[i] as f64 / na;
            let pb = b.exponent_histogram[i] as f64 / nb;
            (pa - pb).abs()
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerical::bitprof::profile;

    /// Same rationale as bitprof's: exact integer counts, one division.
    const TOL_BIT: f64 = 1e-10;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL_BIT
    }

    #[test]
    fn two_identical_profiles_diverge_by_nothing() {
        let a = profile(&[1.0_f32; 100]);
        let b = profile(&[1.0_f32; 100]);
        let d = compare(&a, &b);

        assert!(d.bit_fraction_delta.iter().all(|&x| x.abs() < TOL_BIT));
        assert!(d.exponent_l1 < TOL_BIT, "got {}", d.exponent_l1);
        assert!(close(d.max_bit_delta, 0.0));
        assert!(close(d.mantissa_entropy_delta.unwrap(), 0.0));
    }

    /// The property the corrected `exponent_l1` formula exists for: same
    /// distribution, different lengths, zero divergence.
    #[test]
    fn exponent_l1_is_invariant_to_tensor_length() {
        let a = profile(&[1.0_f32; 100]);
        let b = profile(&[1.0_f32; 200]);
        let d = compare(&a, &b);
        assert!(
            d.exponent_l1 < TOL_BIT,
            "identical distributions of different lengths must not diverge, got {}",
            d.exponent_l1
        );

        // And on a non-trivial distribution: b is a doubled to twice the length.
        let base: Vec<f32> = (1..=64).map(|i| i as f32 / 3.0).collect();
        let doubled: Vec<f32> = base.iter().chain(base.iter()).copied().collect();
        let d = compare(&profile(&base), &profile(&doubled));
        assert!(d.exponent_l1 < TOL_BIT, "got {}", d.exponent_l1);
    }

    #[test]
    fn fully_disjoint_exponent_distributions_score_the_maximum_of_two() {
        // 1.0 has exponent 127; 1024.0 has exponent 137. No overlap.
        let a = profile(&[1.0_f32; 50]);
        let b = profile(&[1024.0_f32; 50]);
        let d = compare(&a, &b);
        assert!(close(d.exponent_l1, 2.0), "got {}", d.exponent_l1);
    }

    #[test]
    fn a_half_overlapping_distribution_scores_one() {
        // a: all exponent 127. b: half 127, half 137.
        let a = profile(&[1.0_f32; 100]);
        let mut mixed = vec![1.0_f32; 50];
        mixed.extend(std::iter::repeat_n(1024.0_f32, 50));
        let b = profile(&mixed);
        let d = compare(&a, &b);
        // |1 - 0.5| + |0 - 0.5| = 1.0
        assert!(close(d.exponent_l1, 1.0), "got {}", d.exponent_l1);
    }

    #[test]
    fn the_sign_bit_delta_is_reported_with_its_position() {
        let a = profile(&[1.0_f32; 100]);
        let b = profile(&[-1.0_f32; 100]);
        let d = compare(&a, &b);
        // Only the sign bit moved, and it moved all the way.
        assert!(close(d.bit_fraction_delta[31], 1.0));
        assert_eq!(d.max_bit_position, 31);
        assert!(close(d.max_bit_delta, 1.0));
        // Exponents are identical, so that axis stays quiet — which is exactly
        // the "structured, not distributional" signature.
        assert!(d.exponent_l1 < TOL_BIT);
    }

    #[test]
    fn deltas_are_signed_and_oriented_b_minus_a() {
        let a = profile(&[-1.0_f32; 100]);
        let b = profile(&[1.0_f32; 100]);
        let d = compare(&a, &b);
        assert!(close(d.bit_fraction_delta[31], -1.0), "b has fewer sign bits set");
        assert!(close(d.max_bit_delta, -1.0));
    }

    /// A profile that declined its mantissa map has no entropy to subtract, and
    /// the delta must say so rather than reading as "no change".
    #[test]
    fn mantissa_entropy_delta_is_none_when_either_side_declined_the_map() {
        let small = profile(&[1.0_f32; 100]);
        let large: Vec<f32> = (0..200_000).map(|i| i as f32).collect();
        let large = profile(&large);
        assert!(large.mantissa_sparse_skipped);

        assert!(compare(&small, &large).mantissa_entropy_delta.is_none());
        assert!(compare(&large, &small).mantissa_entropy_delta.is_none());
        assert!(compare(&large, &large).mantissa_entropy_delta.is_none());
        // Both present → Some.
        assert!(compare(&small, &small).mantissa_entropy_delta.is_some());
    }

    /// Losing mantissa detail is what a quantisation scheme does, so the delta
    /// must be negative in that direction.
    #[test]
    fn quantising_away_mantissa_detail_shows_as_a_negative_entropy_delta() {
        // `fine` has many distinct mantissa patterns; `coarse` rounds them all
        // onto exact powers of two, leaving one.
        let fine: Vec<f32> = (1..=1024).map(|i| i as f32 / 1024.0 + 1.0).collect();
        let coarse: Vec<f32> = vec![1.0_f32; 1024];
        let d = compare(&profile(&fine), &profile(&coarse));

        let delta = d.mantissa_entropy_delta.expect("both sides are under the cap");
        assert!(delta < -1.0, "expected a clear entropy drop, got {delta}");
    }

    /// ⛔ The blind spot, pinned.
    ///
    /// Every statistic in a bit profile is computed from an unordered tally, so
    /// reordering the values cannot change any of them. This test exists to
    /// stop the module's own claim from drifting back to "detects nibble-order
    /// bugs" — see the module docs for the measurement that established it.
    ///
    /// If this test ever fails, something gained positional sensitivity and the
    /// docs above need rewriting in the *good* direction.
    #[test]
    fn a_permutation_is_invisible_to_the_distribution_axes_and_caught_by_order_rms() {
        let values: Vec<f32> = (1..=512).map(|i| i as f32 * 0.017).collect();

        // The same multiset, pairwise-swapped — the shape a nibble-order defect
        // produces inside one scale sub-block.
        let mut swapped = values.clone();
        for pair in swapped.chunks_exact_mut(2) {
            pair.swap(0, 1);
        }
        assert_ne!(values, swapped, "the permutation must actually reorder");

        let d = compare(&profile(&values), &profile(&swapped));
        assert!(d.exponent_l1 < TOL_BIT, "exponent axis saw a permutation: {}", d.exponent_l1);
        assert!(close(d.max_bit_delta, 0.0), "bit axis saw a permutation");
        assert!(
            close(d.mantissa_entropy_delta.unwrap(), 0.0),
            "entropy axis saw a permutation"
        );
        let swap_rms = d.order_rms.unwrap();
        assert!(swap_rms > 0.005, "the order axis must see the swap: {swap_rms}");

        // The true RMS difference is 0.017 for adjacent values; the sketch
        // estimates it within its stated error.
        assert!((swap_rms - 0.017).abs() < 0.017 * 0.75, "estimate off: {swap_rms}");

        // A reversal is the extreme case, equally invisible to the first three.
        let mut reversed = values.clone();
        reversed.reverse();
        let d = compare(&profile(&values), &profile(&reversed));
        assert!(d.exponent_l1 < TOL_BIT);
        assert!(close(d.max_bit_delta, 0.0));
        assert!(d.order_rms.unwrap() > swap_rms * 10.0, "a reversal moves values far further");

        // And the same tensor in the same order scores exactly zero.
        assert_eq!(compare(&profile(&values), &profile(&values)).order_rms, Some(0.0));
    }

    #[test]
    fn tensors_of_different_lengths_have_no_order_distance() {
        let d = compare(&profile(&[1.0, 2.0]), &profile(&[1.0, 2.0, 3.0]));
        assert_eq!(d.order_rms, None);
    }

    /// The other half of the same finding: changing the *values* does register,
    /// so the blind spot above is specific to ordering rather than general
    /// insensitivity.
    #[test]
    fn changing_the_distribution_does_register() {
        let values: Vec<f32> = (1..=512).map(|i| i as f32 * 0.017).collect();
        let scaled: Vec<f32> = values.iter().map(|v| v * 4.0).collect();

        let d = compare(&profile(&values), &profile(&scaled));
        assert!(
            d.exponent_l1 > 0.1,
            "a 4x scale shift must move the exponent distribution, got {}",
            d.exponent_l1
        );
    }

    #[test]
    fn an_empty_profile_on_either_side_gives_a_quiet_exponent_axis() {
        let empty = profile(&[]);
        let full = profile(&[1.0_f32; 10]);
        assert!(close(compare(&empty, &full).exponent_l1, 0.0));
        assert!(close(compare(&full, &empty).exponent_l1, 0.0));
    }
}
