//! GLBitProf — what the bits of a tensor actually look like (D-11, D-12).
//!
//! One function, one input type:
//!
//! ```text
//! profile(&[f32]) -> VLBitProfile
//! ```
//!
//! Weights, gradients and optimizer state all cross the boundary as flat f32
//! (design F-04), so there is exactly one implementation and no generic. The
//! math is **ungated** — bit-profiling a gradient has nothing to do with
//! `.gllm` packages — while the *sources* are gated per source (D-11).
//!
//! # Three tiers, and why the middle one has a precondition
//!
//! **Tier 1 — exponent, dense.** 256 buckets, 2 KiB, always computed. This is
//! where dynamic range lives, and it is the tier that answers "did this
//! quantisation scheme shift the scale".
//!
//! **Tier 2 — mantissa, sparse, full 23-bit resolution.** A *dense*
//! full-resolution histogram would be 2²³ = 8,388,608 buckets — 67 MB at `u64`,
//! past any L3, turning a linear scan into a random-access cache-miss
//! generator. Bucketing to 12 bits was the first mitigation and was rejected
//! for losing the fine-grained signal a researcher wants. So the map is sparse,
//! and collected **only when it can be complete**.
//!
//! The precondition is the whole point. FP32 mantissa bytes sit near maximum
//! entropy (~7.97 of 8 bits/byte in the ZipNN/ENEC/DFloat11 literature), so
//! distinct keys accumulate at nearly one per element: by the birthday-bound
//! occupancy formula `m·(1 − e^(−n/m))` with m = 2²³, the cap of
//! [`MANTISSA_SPARSE_CAP`] keys is reached around n ≈ 132,107 elements.
//! Checking `values.len()` up front and declining is honest. Collecting
//! mid-run and truncating at the cap would not be: it yields the mantissa
//! patterns that happened to appear *first*, which is an order-biased sample,
//! not a distribution — and nothing in the output would say so.
//!
//! **Tier 3 — per bit position, all 32.** Always present, always exact, and
//! cheap. This is the tier that catches a *structured* anomaly: a wrong nibble
//! order shows up as specific positions moving, where a quantisation scheme
//! change shows up as a smooth shift in Tier 1.
//!
//! # Definitional choices, stated because they are choices
//!
//! - **`-0.0` has bit 31 set.** Counting it in [`VLBitProfile::sign_set_ratio`]
//!   would report a freshly zero-initialised LoRA `B` matrix as 100% negative.
//!   It is counted in `negative_zero_count`; `zero_count` is `+0.0` only.
//! - **NaN and Inf** are excluded from `exponent_min`/`exponent_max` and from
//!   the mantissa map, and counted in their own fields. They stay in
//!   `bit_set_fraction` and `exponent_histogram`, which are raw per-position
//!   and per-bucket counts and are documented as such.
//! - **Entropy is in bits** throughout, `0·log₂0` taken as 0.

use std::collections::HashMap;

/// Largest number of distinct mantissa patterns the sparse map will hold.
///
/// Also used as the element-count precondition: a tensor longer than this
/// cannot be profiled into a complete map (see the module docs), so it is
/// declined rather than sampled.
pub const MANTISSA_SPARSE_CAP: u32 = 131_072;

/// Mask for the 23 mantissa bits of an IEEE-754 binary32.
const MANTISSA_MASK: u32 = 0x007F_FFFF;

/// `+0.0` — the only bit pattern [`VLBitProfile::zero_count`] counts.
const POSITIVE_ZERO_BITS: u32 = 0x0000_0000;

/// `-0.0` — sign bit set, everything else clear.
const NEGATIVE_ZERO_BITS: u32 = 0x8000_0000;

/// The exponent byte of a NaN or an infinity.
const EXPONENT_ALL_SET: u8 = 0xFF;

/// Denominator for [`VLBitProfile::dynamic_range_used`]: the span of exponent
/// bytes a finite value can occupy, 1..=254 (0 is zero/subnormal, 255 is
/// NaN/Inf).
const FINITE_EXPONENT_SPAN: f64 = 254.0;

/// What the bits of one tensor look like.
#[derive(Debug, Clone)]
pub struct VLBitProfile {
    /// How many values were profiled.
    pub count: u64,

    /// Fraction of values with bit 31 set, **excluding** `-0.0`. See the module
    /// docs for why that exclusion exists.
    pub sign_set_ratio: f64,

    /// Tier 1: count per exponent byte (bits 30..=23, biased by 127). Raw —
    /// NaN and Inf land in bucket 255 rather than being dropped.
    pub exponent_histogram: Box<[u64; 256]>,
    /// Smallest exponent byte among finite values. 0 when there are none.
    pub exponent_min: u8,
    /// Largest exponent byte among finite values. 0 when there are none.
    pub exponent_max: u8,
    /// `(exponent_max - exponent_min) / 254`, derived. 0.0 when no value is
    /// finite.
    pub dynamic_range_used: f64,

    /// Tier 2: full-resolution mantissa pattern counts. `None` when the tensor
    /// is too long for the map to be complete — never a truncated map.
    pub mantissa_sparse: Option<HashMap<u32, u64>>,
    /// Always [`MANTISSA_SPARSE_CAP`], carried so a consumer reading the
    /// archive alone can see which threshold produced the decision.
    pub mantissa_sparse_cap: u32,
    /// True when Tier 2 was declined. Tiers 1 and 3 are computed regardless.
    pub mantissa_sparse_skipped: bool,
    /// Shannon entropy over `mantissa_sparse`, in bits. `None` exactly when
    /// `mantissa_sparse_skipped`.
    pub mantissa_entropy_bits: Option<f64>,

    /// Tier 3: fraction of values with bit *i* set, for every position. Raw,
    /// including NaN and Inf.
    pub bit_set_fraction: [f64; 32],
    /// Binary entropy of each bit position, in bits. Range [0, 1].
    pub bit_entropy: [f64; 32],

    /// Count of `+0.0`.
    pub zero_count: u64,
    /// Count of `-0.0`.
    pub negative_zero_count: u64,
    /// Count of subnormals (exponent 0, mantissa non-zero).
    pub subnormal_count: u64,
    /// Count of NaNs (exponent 255, mantissa non-zero).
    pub nan_count: u64,
    /// Count of infinities (exponent 255, mantissa zero).
    pub inf_count: u64,
}

/// Profile the bit patterns of `values`.
///
/// One linear pass. The Tier 2 decision is made from `values.len()` before the
/// pass starts, so the expensive map is never allocated for a tensor that
/// cannot fill it completely.
pub fn profile(values: &[f32]) -> VLBitProfile {
    let count = values.len() as u64;

    // D-12's precondition. Decided up front, from the length alone — see the
    // module docs for why a mid-run truncation would be a different and much
    // worse thing than declining.
    let skip_mantissa = values.len() > MANTISSA_SPARSE_CAP as usize;
    // Sized for the worst case up front. Distinct mantissa patterns are bounded
    // above by the element count, and below the cap that is at most
    // MANTISSA_SPARSE_CAP entries — a few MB. Starting small and growing costs
    // far more than the memory saves: measured on 65,536 near-uniform elements,
    // repeated rehashing made the per-element cost *higher* than a 16M-element
    // tensor that skips the map entirely.
    let mut mantissa_sparse: Option<HashMap<u32, u64>> = if skip_mantissa {
        None
    } else {
        Some(HashMap::with_capacity(values.len()))
    };

    let mut exponent_histogram = Box::new([0u64; 256]);
    let mut bit_counts = [0u64; 32];
    let mut sign_set = 0u64;
    let mut zero_count = 0u64;
    let mut negative_zero_count = 0u64;
    let mut subnormal_count = 0u64;
    let mut nan_count = 0u64;
    let mut inf_count = 0u64;
    let mut finite_exponent_min: Option<u8> = None;
    let mut finite_exponent_max: Option<u8> = None;

    for &value in values {
        let bits = value.to_bits();
        let exponent = ((bits >> 23) & 0xFF) as u8;
        let mantissa = bits & MANTISSA_MASK;

        exponent_histogram[exponent as usize] += 1;
        for (i, slot) in bit_counts.iter_mut().enumerate() {
            *slot += ((bits >> i) & 1) as u64;
        }

        // Classify. Order matters only in that the two zeroes are checked as
        // whole bit patterns before the subnormal test, which they would
        // otherwise satisfy.
        match bits {
            POSITIVE_ZERO_BITS => zero_count += 1,
            NEGATIVE_ZERO_BITS => negative_zero_count += 1,
            _ => {
                if bits >> 31 == 1 {
                    sign_set += 1;
                }
                if exponent == 0 {
                    subnormal_count += 1;
                } else if exponent == EXPONENT_ALL_SET {
                    if mantissa == 0 {
                        inf_count += 1;
                    } else {
                        nan_count += 1;
                    }
                }
            }
        }

        let is_finite = exponent != EXPONENT_ALL_SET;
        if is_finite {
            finite_exponent_min = Some(match finite_exponent_min {
                Some(m) => m.min(exponent),
                None => exponent,
            });
            finite_exponent_max = Some(match finite_exponent_max {
                Some(m) => m.max(exponent),
                None => exponent,
            });
            if let Some(map) = mantissa_sparse.as_mut() {
                *map.entry(mantissa).or_insert(0) += 1;
            }
        }
    }

    let denominator = count as f64;
    let mut bit_set_fraction = [0.0f64; 32];
    let mut bit_entropy = [0.0f64; 32];
    if count > 0 {
        for i in 0..32 {
            let p = bit_counts[i] as f64 / denominator;
            bit_set_fraction[i] = p;
            bit_entropy[i] = binary_entropy(p);
        }
    }

    let exponent_min = finite_exponent_min.unwrap_or(0);
    let exponent_max = finite_exponent_max.unwrap_or(0);
    let dynamic_range_used = if finite_exponent_min.is_some() {
        (exponent_max as f64 - exponent_min as f64) / FINITE_EXPONENT_SPAN
    } else {
        // Every value was NaN or Inf: no finite range was used at all.
        0.0
    };

    let mantissa_entropy_bits = mantissa_sparse.as_ref().map(sparse_entropy_bits);

    VLBitProfile {
        count,
        sign_set_ratio: if count > 0 { sign_set as f64 / denominator } else { 0.0 },
        exponent_histogram,
        exponent_min,
        exponent_max,
        dynamic_range_used,
        mantissa_sparse,
        mantissa_sparse_cap: MANTISSA_SPARSE_CAP,
        mantissa_sparse_skipped: skip_mantissa,
        mantissa_entropy_bits,
        bit_set_fraction,
        bit_entropy,
        zero_count,
        negative_zero_count,
        subnormal_count,
        nan_count,
        inf_count,
    }
}

/// Binary entropy of a Bernoulli(`p`), in bits. `0·log₂0` is taken as 0, which
/// is the limit and also the only value that keeps the result finite.
fn binary_entropy(p: f64) -> f64 {
    if p <= 0.0 || p >= 1.0 {
        return 0.0;
    }
    -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
}

/// Shannon entropy over a sparse count map, in bits.
///
/// An empty map is 0.0 rather than NaN: a tensor with no finite values has no
/// mantissa distribution, and reporting that as "not a number" would be a
/// worse answer than reporting it as "no information".
pub fn sparse_entropy_bits(map: &HashMap<u32, u64>) -> f64 {
    let total: u64 = map.values().sum();
    if total == 0 {
        return 0.0;
    }
    let total = total as f64;
    map.values()
        .map(|&c| {
            let p = c as f64 / total;
            if p <= 0.0 {
                0.0
            } else {
                -p * p.log2()
            }
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Bit-level quantities are computed from exact integer counts, so the only
    /// error is the final division and the log. 1e-10 is far above that and far
    /// below any difference these tests care about.
    const TOL_BIT: f64 = 1e-10;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < TOL_BIT
    }

    // -----------------------------------------------------------------------
    // Test 5 — GLBitProf known answers (design §11)
    // -----------------------------------------------------------------------

    /// The case §6.4's zero-handling rule exists for, and the one a freshly
    /// initialised LoRA `B` matrix hits.
    #[test]
    fn test5_positive_and_negative_zero_are_counted_apart() {
        // 0.0  = 0 00000000 00000000000000000000000 = 0x00000000
        // -0.0 = 1 00000000 00000000000000000000000 = 0x80000000
        let values = [0.0_f32, -0.0_f32];
        assert_eq!(values[0].to_bits(), 0x0000_0000);
        assert_eq!(values[1].to_bits(), 0x8000_0000);

        let p = profile(&values);
        assert_eq!(p.count, 2);
        assert_eq!(p.zero_count, 1);
        assert_eq!(p.negative_zero_count, 1);
        // -0.0 carries the sign bit but must not inflate the ratio.
        assert!(close(p.sign_set_ratio, 0.0), "got {}", p.sign_set_ratio);
        // Both have an all-zero exponent byte.
        assert_eq!(p.exponent_histogram[0], 2);
        assert_eq!(p.exponent_min, 0);
        assert_eq!(p.exponent_max, 0);
        // Neither is subnormal: mantissa is zero for both.
        assert_eq!(p.subnormal_count, 0);
        // The raw per-position count still sees the sign bit, by design.
        assert!(close(p.bit_set_fraction[31], 0.5), "got {}", p.bit_set_fraction[31]);
    }

    /// The LoRA `B` matrix case at scale: all `-0.0`.
    #[test]
    fn test5_a_tensor_of_only_negative_zero_reports_no_negative_sign_ratio() {
        let values = vec![-0.0_f32; 100];
        let p = profile(&values);
        assert!(close(p.sign_set_ratio, 0.0), "got {}", p.sign_set_ratio);
        assert_eq!(p.negative_zero_count, 100);
        assert_eq!(p.zero_count, 0);
        // Every value shares one bit pattern, so no position carries
        // information even though bit 31 is always set.
        assert!(p.bit_entropy.iter().all(|&e| close(e, 0.0)));
        assert!(close(p.bit_set_fraction[31], 1.0));
    }

    #[test]
    fn test5_an_all_zero_tensor_has_no_entropy_in_any_bit_position() {
        let p = profile(&[0.0_f32; 64]);
        assert_eq!(p.zero_count, 64);
        assert_eq!(p.negative_zero_count, 0);
        for (i, &e) in p.bit_entropy.iter().enumerate() {
            assert!(close(e, 0.0), "bit {i} entropy {e}");
        }
        for (i, &f) in p.bit_set_fraction.iter().enumerate() {
            assert!(close(f, 0.0), "bit {i} fraction {f}");
        }
        // One distinct mantissa pattern → zero entropy, not NaN.
        assert_eq!(p.mantissa_sparse.as_ref().unwrap().len(), 1);
        assert!(close(p.mantissa_entropy_bits.unwrap(), 0.0));
    }

    #[test]
    fn test5_an_all_ones_bit_pattern_is_counted_as_nan() {
        // 0xFFFFFFFF = 1 11111111 11111111111111111111111 — exponent all set,
        // mantissa non-zero, so NaN rather than Inf.
        let values = vec![f32::from_bits(0xFFFF_FFFF); 32];
        let p = profile(&values);
        assert_eq!(p.nan_count, 32);
        assert_eq!(p.inf_count, 0);
        // Excluded from the finite exponent range...
        assert_eq!(p.exponent_min, 0);
        assert_eq!(p.exponent_max, 0);
        assert!(close(p.dynamic_range_used, 0.0));
        // ...but present in the raw histogram and the raw bit counts.
        assert_eq!(p.exponent_histogram[255], 32);
        assert!(p.bit_set_fraction.iter().all(|&f| close(f, 1.0)));
        // No finite value, so the mantissa map is empty — and its entropy is
        // 0.0, not NaN.
        assert!(p.mantissa_sparse.as_ref().unwrap().is_empty());
        assert!(close(p.mantissa_entropy_bits.unwrap(), 0.0));
    }

    #[test]
    fn test5_the_smallest_subnormal_is_counted_as_subnormal_not_zero() {
        // 0x00000001: exponent 0, mantissa 1.
        let values = [f32::from_bits(0x0000_0001)];
        let p = profile(&values);
        assert_eq!(p.subnormal_count, 1);
        assert_eq!(p.zero_count, 0);
        assert_eq!(p.negative_zero_count, 0);
        assert_eq!(p.exponent_min, 0);
        assert_eq!(p.exponent_max, 0);
        assert!(close(p.bit_set_fraction[0], 1.0));
    }

    #[test]
    fn test5_both_infinities_are_counted_as_inf_and_excluded_from_the_range() {
        let values = [f32::INFINITY, f32::NEG_INFINITY];
        let p = profile(&values);
        assert_eq!(p.inf_count, 2);
        assert_eq!(p.nan_count, 0);
        // NEG_INFINITY carries the sign bit and is not a zero, so it counts.
        assert!(close(p.sign_set_ratio, 0.5), "got {}", p.sign_set_ratio);
        assert_eq!(p.exponent_histogram[255], 2);
        assert!(close(p.dynamic_range_used, 0.0));
    }

    #[test]
    fn test5_an_exact_power_of_two_has_the_expected_exponent_and_empty_mantissa() {
        // 1.0f32 = 0x3F800000 = 0 01111111 00000000000000000000000
        let values = [1.0_f32];
        assert_eq!(values[0].to_bits(), 0x3F80_0000);

        let p = profile(&values);
        assert_eq!(p.exponent_min, 127);
        assert_eq!(p.exponent_max, 127);
        assert_eq!(p.exponent_histogram[127], 1);
        // Exponent byte 0111_1111: bits 23..=29 set, bit 30 clear, sign clear.
        for i in 23..=29 {
            assert!(close(p.bit_set_fraction[i], 1.0), "bit {i} should be set");
        }
        assert!(close(p.bit_set_fraction[30], 0.0), "bit 30 must be clear");
        assert!(close(p.bit_set_fraction[31], 0.0), "sign must be clear");
        // Mantissa is exactly zero for a power of two.
        for i in 0..23 {
            assert!(close(p.bit_set_fraction[i], 0.0), "mantissa bit {i} should be clear");
        }
        let map = p.mantissa_sparse.as_ref().unwrap();
        assert_eq!(map.len(), 1);
        assert_eq!(map.get(&0), Some(&1));
    }

    #[test]
    fn test5_a_mixed_sign_tensor_reports_the_sign_ratio_over_all_values() {
        let values = [1.0_f32, -1.0, -2.0, 4.0];
        let p = profile(&values);
        assert!(close(p.sign_set_ratio, 0.5), "got {}", p.sign_set_ratio);
        assert_eq!(p.exponent_min, 127); // 1.0
        assert_eq!(p.exponent_max, 129); // 4.0
        assert!(close(p.dynamic_range_used, 2.0 / 254.0));
    }

    // -----------------------------------------------------------------------
    // D-12 precondition guard
    // -----------------------------------------------------------------------

    #[test]
    fn a_small_tensor_collects_a_complete_mantissa_map() {
        let small: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let p = profile(&small);
        assert!(p.mantissa_sparse.is_some());
        assert!(!p.mantissa_sparse_skipped);
        assert!(p.mantissa_entropy_bits.is_some());
        assert_eq!(p.mantissa_sparse_cap, MANTISSA_SPARSE_CAP);
    }

    #[test]
    fn a_large_tensor_declines_the_mantissa_map_but_keeps_tiers_one_and_three() {
        let large: Vec<f32> = (0..200_000).map(|i| i as f32).collect();
        let p = profile(&large);
        assert!(p.mantissa_sparse.is_none());
        assert!(p.mantissa_sparse_skipped);
        assert!(p.mantissa_entropy_bits.is_none());

        // Tier 1 and Tier 3 are unconditional.
        assert_eq!(p.exponent_histogram.iter().sum::<u64>(), 200_000);
        assert!(p.bit_entropy.iter().all(|&e| (0.0..=1.0).contains(&e)));
        assert!(p.exponent_max > p.exponent_min, "a ramp must span exponents");
    }

    /// The decision is the one most likely to be mis-implemented as an abort or
    /// as a truncated map, so both sides of the boundary are pinned — including
    /// the assertion that `CAP + 1` gives `None` rather than a `Some` holding
    /// exactly `CAP` entries.
    #[test]
    fn the_precondition_boundary_is_exact_at_cap_and_cap_plus_one() {
        let cap = MANTISSA_SPARSE_CAP as usize;

        let at_cap: Vec<f32> = (0..cap).map(|i| i as f32).collect();
        let p = profile(&at_cap);
        assert!(!p.mantissa_sparse_skipped, "exactly CAP elements must be profiled");
        assert!(p.mantissa_sparse.is_some());
        assert!(p.mantissa_entropy_bits.is_some());

        let over_cap: Vec<f32> = (0..cap + 1).map(|i| i as f32).collect();
        let p = profile(&over_cap);
        assert!(p.mantissa_sparse_skipped, "CAP + 1 elements must be declined");
        assert!(
            p.mantissa_sparse.is_none(),
            "must be None, not a Some truncated to CAP entries"
        );
        assert!(p.mantissa_entropy_bits.is_none());
    }

    // -----------------------------------------------------------------------
    // General properties
    // -----------------------------------------------------------------------

    #[test]
    fn an_empty_tensor_profiles_to_zeros_rather_than_panicking() {
        let p = profile(&[]);
        assert_eq!(p.count, 0);
        assert!(close(p.sign_set_ratio, 0.0));
        assert!(close(p.dynamic_range_used, 0.0));
        assert!(p.bit_set_fraction.iter().all(|&f| close(f, 0.0)));
        assert!(p.bit_entropy.iter().all(|&e| close(e, 0.0)));
        // An empty tensor is under the cap, so the map exists and is empty.
        assert!(!p.mantissa_sparse_skipped);
        assert!(close(p.mantissa_entropy_bits.unwrap(), 0.0));
    }

    #[test]
    fn bit_entropy_peaks_at_one_when_a_position_is_evenly_split() {
        // 1.0 and -1.0 differ in exactly the sign bit.
        let values = [1.0_f32, -1.0_f32];
        let p = profile(&values);
        assert!(close(p.bit_entropy[31], 1.0), "got {}", p.bit_entropy[31]);
        // Every other position is constant across the two values.
        for i in 0..31 {
            assert!(close(p.bit_entropy[i], 0.0), "bit {i} entropy {}", p.bit_entropy[i]);
        }
    }

    #[test]
    fn bit_entropy_stays_within_zero_and_one_on_arbitrary_data() {
        let values: Vec<f32> = (0..5000).map(|i| ((i * 2654435761u64 as usize) as f32).sin()).collect();
        let p = profile(&values);
        for (i, &e) in p.bit_entropy.iter().enumerate() {
            assert!((0.0..=1.0 + TOL_BIT).contains(&e), "bit {i} entropy out of range: {e}");
        }
    }

    #[test]
    fn the_exponent_histogram_totals_the_element_count() {
        let values: Vec<f32> = (1..500).map(|i| i as f32 / 7.0).collect();
        let p = profile(&values);
        assert_eq!(p.exponent_histogram.iter().sum::<u64>(), p.count);
    }

    #[test]
    fn the_mantissa_map_totals_the_finite_element_count() {
        let values = [1.0_f32, 2.0, f32::NAN, f32::INFINITY, 3.0];
        let p = profile(&values);
        let map = p.mantissa_sparse.as_ref().unwrap();
        let total: u64 = map.values().sum();
        assert_eq!(total, 3, "NaN and Inf must be excluded from the mantissa map");
        assert_eq!(p.nan_count, 1);
        assert_eq!(p.inf_count, 1);
    }

    /// A uniform distribution over k patterns has entropy log2(k) — the check
    /// that the entropy formula is the one it claims to be.
    #[test]
    fn sparse_entropy_of_a_uniform_map_is_log2_of_its_size() {
        for k in [2u32, 4, 8, 256] {
            let map: HashMap<u32, u64> = (0..k).map(|i| (i, 10u64)).collect();
            let h = sparse_entropy_bits(&map);
            assert!(close(h, (k as f64).log2()), "k={k} gave {h}");
        }
    }

    #[test]
    fn sparse_entropy_of_a_single_pattern_is_zero() {
        let map: HashMap<u32, u64> = [(42u32, 1000u64)].into_iter().collect();
        assert!(close(sparse_entropy_bits(&map), 0.0));
    }

    #[test]
    fn sparse_entropy_of_an_empty_map_is_zero_not_nan() {
        let map: HashMap<u32, u64> = HashMap::new();
        let h = sparse_entropy_bits(&map);
        assert!(h.is_finite(), "empty map gave {h}");
        assert!(close(h, 0.0));
    }
}
