//! Repetition — how much of the output is recycled n-grams.
//!
//! Degenerate looping ("the the the", or a sentence repeating verbatim) is the
//! most common failure of a broken sampler, a corrupted weight, or a KV-cache
//! bug. It is also invisible in tok/s: a model can loop at full speed.
//!
//! Computed from token ids alone — no trace required, so this signal is always
//! available.

use std::collections::HashSet;

/// N-gram diversity of a generated sequence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RepetitionSignal {
    /// Distinct 1-grams / total 1-grams. 1.0 = every token unique.
    pub unique_1gram_ratio: f64,
    /// Distinct 2-grams / total 2-grams.
    pub unique_2gram_ratio: f64,
    /// Distinct 3-grams / total 3-grams. The most diagnostic of the three: a
    /// model can legitimately reuse single tokens ("the") and even bigrams
    /// ("of the"), but a low 3-gram ratio means whole phrases are recurring.
    pub unique_3gram_ratio: f64,
    /// Longest run of one token repeated back to back. A hard failure signal —
    /// healthy text almost never exceeds 2 or 3.
    pub max_token_run: usize,
    /// Tokens the ratios were computed over.
    pub tokens: usize,
}

impl RepetitionSignal {
    /// Compute over a generated sequence. `None` for an empty sequence — there
    /// is no diversity to report about nothing.
    pub fn compute(tokens: &[u32]) -> Option<RepetitionSignal> {
        if tokens.is_empty() {
            return None;
        }
        Some(RepetitionSignal {
            unique_1gram_ratio: unique_ngram_ratio(tokens, 1),
            unique_2gram_ratio: unique_ngram_ratio(tokens, 2),
            unique_3gram_ratio: unique_ngram_ratio(tokens, 3),
            max_token_run: max_run(tokens),
            tokens: tokens.len(),
        })
    }

    /// A crude "is this obviously looping" flag, for the renderer.
    ///
    /// Deliberately conservative: 3-gram diversity under 0.5 means half of all
    /// three-token windows are duplicates, which healthy prose does not do.
    /// This is a display hint, not a verdict — glbench reports, it does not
    /// judge.
    pub fn looks_degenerate(&self) -> bool {
        // Only meaningful once there is enough text for the ratio to settle;
        // a 4-token output can trivially score low by chance.
        self.tokens >= 16 && (self.unique_3gram_ratio < 0.5 || self.max_token_run >= 5)
    }
}

/// Distinct n-grams / total n-grams. Returns 1.0 when the sequence is shorter
/// than `n` (no window exists to repeat, so nothing is recycled).
fn unique_ngram_ratio(tokens: &[u32], n: usize) -> f64 {
    if n == 0 || tokens.len() < n {
        return 1.0;
    }
    let windows: Vec<&[u32]> = tokens.windows(n).collect();
    let total = windows.len();
    let distinct: HashSet<&[u32]> = windows.into_iter().collect();
    distinct.len() as f64 / total as f64
}

/// Longest back-to-back repetition of a single token.
fn max_run(tokens: &[u32]) -> usize {
    let mut best = 1usize;
    let mut cur = 1usize;
    for w in tokens.windows(2) {
        if w[0] == w[1] {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 1;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_unique_tokens_score_one() {
        let r = RepetitionSignal::compute(&[1, 2, 3, 4, 5]).unwrap();
        assert!((r.unique_1gram_ratio - 1.0).abs() < 1e-9);
        assert!((r.unique_3gram_ratio - 1.0).abs() < 1e-9);
        assert_eq!(r.max_token_run, 1);
        assert!(!r.looks_degenerate());
    }

    #[test]
    fn a_single_repeated_token_is_maximally_degenerate() {
        // The classic broken-decode output: "the the the the ...".
        let r = RepetitionSignal::compute(&[7; 32]).unwrap();
        assert!((r.unique_1gram_ratio - 1.0 / 32.0).abs() < 1e-9);
        assert_eq!(r.max_token_run, 32);
        assert!(r.looks_degenerate());
    }

    #[test]
    fn a_looping_phrase_is_caught_by_3gram_not_1gram() {
        // "a b c a b c a b c ..." — every TOKEN is common (1-gram ratio is
        // low), but the real tell is that only 3 distinct 3-grams exist across
        // many windows. This is the case a 1-gram-only metric would miss.
        let tokens: Vec<u32> = (0..30).map(|i| (i % 3) as u32).collect();
        let r = RepetitionSignal::compute(&tokens).unwrap();
        assert!(r.unique_3gram_ratio < 0.2, "got {}", r.unique_3gram_ratio);
        assert_eq!(r.max_token_run, 1, "no token repeats back-to-back here");
        assert!(r.looks_degenerate(), "phrase looping must be flagged");
    }

    #[test]
    fn short_output_is_not_flagged_on_thin_evidence() {
        // 4 tokens can score low by chance; refusing to call it degenerate is
        // the honest choice.
        let r = RepetitionSignal::compute(&[1, 1, 1, 1]).unwrap();
        assert!(!r.looks_degenerate(), "too few tokens to conclude anything");
    }

    #[test]
    fn sequence_shorter_than_n_has_no_repetition_to_report() {
        let r = RepetitionSignal::compute(&[1, 2]).unwrap();
        // No 3-gram window exists, so nothing was recycled.
        assert!((r.unique_3gram_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn empty_is_none() {
        assert!(RepetitionSignal::compute(&[]).is_none());
    }
}
