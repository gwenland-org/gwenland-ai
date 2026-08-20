//! Bit-profile divergence between two tensors.
//!
//! # The question this answers
//!
//! Two quantisations of the same weight both show residual KL against the f32
//! original. Is that the scheme doing its job, or a bug?
//!
//! # ⛔ Measured limit: this cannot see a permutation
//!
//! The design (research §12 Case 2) expected a wrong-nibble-order defect — the
//! Q6_K class — to show up here as a *structured per-position anomaly*.
//! **It does not.** Measured 2026-08-20 by
//! `glbench/examples/bitprof_quant_divergence.rs`, using the real GQ4A encoder
//! and the real dequant kernel: a correct GQ4A decode and a nibble-swapped one
//! score **exactly zero** on every axis this module reports, while their MAE
//! differs by 14x the scheme's own residual.
//!
//! The reason is structural rather than a defect to fix. Swapping the two 4-bit
//! codes inside a byte exchanges weights 2i and 2i+1, which share a 32-weight
//! sub-block and therefore a scale. The decoded *multiset* is unchanged, and
//! every statistic in a [`VLBitProfile`] — histogram, per-position fraction,
//! entropy — is permutation-invariant by construction. None of them can see a
//! reordering. [`permutation_invariance_is_a_known_blind_spot`] pins this so it
//! cannot be quietly forgotten.
//!
//! ## What it does see
//!
//! - **A quantisation scheme** shifts the exponent distribution and drops
//!   mantissa entropy — it throws away low-order information on purpose,
//!   evenly, and that registers clearly (GQ4A vs f32: exponent L1 0.40,
//!   entropy −4.29 bits).
//! - **A defect that moves values across a scale boundary** is not a pure
//!   permutation and does register — the sub-block rotation control scored
//!   exponent L1 0.049 against a correct decode of the same scheme.
//!
//! So the honest scope is: **this detects defects that change the distribution
//! of values, and is blind to defects that only change their order.** A
//! permutation-class bug needs an element-wise positional check against an
//! oracle decode, which is [`crate::validation::parity`]'s job, not this
//! module's. Recorded rather than papered over, per
//! `bench-skills/measurement-discipline.md` rule 9.
//!
//! [`permutation_invariance_is_a_known_blind_spot`]: tests::permutation_invariance_is_a_known_blind_spot
//! [`VLBitProfile`]: crate::numerical::bitprof::VLBitProfile

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
    }
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
    fn permutation_invariance_is_a_known_blind_spot() {
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

        // A reversal is the extreme case and is equally invisible.
        let mut reversed = values.clone();
        reversed.reverse();
        let d = compare(&profile(&values), &profile(&reversed));
        assert!(d.exponent_l1 < TOL_BIT);
        assert!(close(d.max_bit_delta, 0.0));
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
