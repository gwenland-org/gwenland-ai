//! Sequence-length bucketing (ARTX05, P3).
//!
//! Static shapes mean one compiled artifact per sequence length. Bucketing is
//! the compromise: round every prompt up to one of a small set of lengths and
//! pad, so the artifact count stays bounded.
//!
//! # ⚠️ The bucket grid is a capacity decision, not a tuning knob
//!
//! Overall-Architecture §P3 spells out how the key dimensions multiply:
//!
//! ```text
//! ARTX05   key = (seq_bucket, dtype, device)
//! ARTX07   + batch_size
//! ARTX11   + gamma, + arch_hash
//! ARTX14   + K
//! ```
//!
//! 5 buckets × 4 slot counts × 3 speculation depths × 2 architectures is **120
//! artifacts**, at ARTX16 §4.2's 20–30 minutes cold each. Adding a bucket is
//! never free, and the cost is multiplicative with every later feature.

/// The sprint's bucket grid.
///
/// ⛔ 2048 is present because ARTX05 specifies it, but a 2048 trace currently
/// **fails**: the causal mask is a dense constant of S² elements and 2048²
/// exceeds [`crate::stablehlo::ops::MAX_DENSE_CONSTANT_ELEMS`]. That is a real
/// unresolved item, not a rounding error — see
/// [`crate::ops::causal_mask`]. Buckets up to 1024 trace today.
pub const BUCKETS: [usize; 5] = [128, 256, 512, 1024, 2048];

/// The largest bucket that can currently be traced end to end.
pub const MAX_TRACEABLE_BUCKET: usize = 1024;

/// Rounds a prompt length up to the smallest bucket that fits.
///
/// Returns `None` when the prompt is longer than every bucket — which is a
/// refusal, not a clamp. Silently truncating a prompt to the largest bucket
/// would drop the user's tokens and still return fluent text (P5).
pub fn bucket_for(prompt_len: usize, buckets: &[usize]) -> Option<usize> {
    buckets.iter().copied().find(|&b| b >= prompt_len)
}

/// How many padding positions a prompt needs in its bucket.
pub fn padding_for(prompt_len: usize, bucket: usize) -> usize {
    bucket.saturating_sub(prompt_len)
}

/// Pads token ids to `bucket` with `pad_id`, at the **end**.
///
/// ⚠️ Right padding, and it only works because the mask is causal: position
/// `i` attends to `0..=i`, so real tokens never see the pad. Left padding would
/// shift every position and silently break RoPE, which reads absolute indices.
///
/// The logits for a prompt of length `n` are therefore at index `n - 1`, not at
/// the end of the bucket.
pub fn pad_to_bucket(ids: &[i32], bucket: usize, pad_id: i32) -> Vec<i32> {
    let mut out = Vec::with_capacity(bucket);
    out.extend_from_slice(ids);
    out.resize(bucket, pad_id);
    out
}

/// Index of the logits row that corresponds to the last real token.
pub fn last_real_position(prompt_len: usize) -> usize {
    prompt_len.saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prompt_rounds_up_to_the_smallest_bucket_that_fits() {
        assert_eq!(bucket_for(1, &BUCKETS), Some(128));
        assert_eq!(bucket_for(128, &BUCKETS), Some(128));
        assert_eq!(bucket_for(129, &BUCKETS), Some(256));
        assert_eq!(bucket_for(2048, &BUCKETS), Some(2048));
    }

    /// ⛔ Clamping instead of refusing would drop tokens off the end of the
    /// prompt and still produce fluent output.
    #[test]
    fn a_prompt_longer_than_every_bucket_is_refused_not_clamped() {
        assert_eq!(bucket_for(2049, &BUCKETS), None);
    }

    #[test]
    fn padding_fills_the_tail_and_leaves_the_prompt_untouched() {
        let ids = [7, 8, 9];
        let padded = pad_to_bucket(&ids, 8, 0);
        assert_eq!(padded, vec![7, 8, 9, 0, 0, 0, 0, 0]);
        assert_eq!(padding_for(3, 8), 5);
        // Right padding, so the last real token is still at index 2.
        assert_eq!(last_real_position(ids.len()), 2);
    }

    #[test]
    fn a_prompt_that_exactly_fills_its_bucket_gets_no_padding() {
        let ids = [1, 2, 3, 4];
        assert_eq!(pad_to_bucket(&ids, 4, 0), vec![1, 2, 3, 4]);
        assert_eq!(padding_for(4, 4), 0);
        assert_eq!(last_real_position(4), 3);
    }

    /// The sprint's working buckets all trace; 2048 is the one that does not.
    #[test]
    fn the_documented_traceable_range_matches_the_bucket_grid() {
        let traceable: Vec<usize> = BUCKETS
            .iter()
            .copied()
            .filter(|&b| b <= MAX_TRACEABLE_BUCKET)
            .collect();
        assert_eq!(traceable, vec![128, 256, 512, 1024]);
        assert!(BUCKETS.contains(&2048));
    }

    /// A grid where a later bucket is smaller than an earlier one would make
    /// `bucket_for` pick a bucket the prompt does not fit in.
    #[test]
    fn the_bucket_grid_is_sorted_ascending() {
        assert!(
            BUCKETS.windows(2).all(|w| w[0] < w[1]),
            "bucket_for takes the first fit, so the grid must be ascending"
        );
    }
}
