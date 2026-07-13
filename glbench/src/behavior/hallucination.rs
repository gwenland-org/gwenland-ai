//! Confidence/rank divergence — a **proxy**, and a weak one. Read this first.
//!
//! # This does not detect hallucination
//!
//! The name is inherited from the signal list, and it oversells what the number
//! can do. Detecting confabulation requires knowing whether a statement is
//! *true*, and a profiler has no ground truth. What is actually measured here is
//! narrower and honest:
//!
//! - **how confident** the model was at each step (`top_prob`, entropy), and
//! - **how far down its own ranking** the emitted token sat (`rank`).
//!
//! Those correlate with confabulation in the literature, but the correlation
//! breaks in both directions, and both failures are common:
//!
//! - **Confidently wrong**: the model states a false fact with rank 0 and
//!   `top_prob` 0.98. Divergence is *zero*. This metric sees nothing.
//! - **Uncertain and right**: a legitimately open-ended continuation ("the
//!   colour I like best is ...") is high-entropy and may sample a low-ranked
//!   token. Divergence is *high*. Nothing is wrong.
//!
//! So: use it to find **where the model was unsure**, which is a real and useful
//! thing to know. Do not use it to decide whether the model lied. If you need
//! that, you need labelled data and an evaluator, not a profiler.
//!
//! glbench reports numbers; it does not judge. This module's job is to give an
//! honest number and refuse to let its own name imply more.

use glcore::trace::TokenTrace;

use super::mean_std;

/// Confidence and rank behavior across a generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HallucinationSignal {
    /// Fraction of tokens where the model emitted its own top choice
    /// (`rank == 0`). With greedy sampling this is 1.0 by construction — the
    /// metric only carries information when sampling is stochastic.
    pub top_choice_rate: f64,
    /// Mean rank of the emitted token in the raw distribution. 0 = always the
    /// argmax.
    pub mean_rank: f64,
    /// Worst rank emitted. A large value means the sampler reached deep into
    /// the tail at least once.
    pub max_rank: u32,
    /// Mean gap between the top token's probability and the emitted token's.
    /// 0 = we always took what the model wanted; large = we routinely overrode
    /// a confident model.
    pub mean_confidence_gap: f64,
    /// Fraction of tokens that were BOTH low-confidence (`top_prob < 0.3`) AND
    /// off the model's top choice. This is the closest thing here to a real
    /// signal: the model had no strong preference, and we picked something it
    /// liked even less.
    pub uncertain_offpick_rate: f64,
    /// Tokens measured.
    pub tokens: usize,
}

/// Below this top-probability the model is treated as having no strong
/// preference. Chosen, not derived — a model spreading 70% of its mass outside
/// its own favourite is hedging by any reading, but the exact cut is a
/// convention and is documented here rather than buried as a literal.
const LOW_CONFIDENCE_TOP_PROB: f64 = 0.3;

impl HallucinationSignal {
    /// `None` when nothing was traced.
    pub fn compute(traces: &[TokenTrace]) -> Option<HallucinationSignal> {
        if traces.is_empty() {
            return None;
        }
        let n = traces.len() as f64;

        let top_choice = traces.iter().filter(|t| t.rank == 0).count() as f64;
        let ranks: Vec<f64> = traces.iter().map(|t| t.rank as f64).collect();
        let (mean_rank, _) = mean_std(&ranks)?;

        // Gap between the model's top probability and the probability it
        // assigned to what we actually emitted.
        let gaps: Vec<f64> = traces
            .iter()
            .map(|t| (t.top_prob as f64 - (t.logprob as f64).exp()).max(0.0))
            .collect();
        let (mean_confidence_gap, _) = mean_std(&gaps)?;

        let uncertain_offpick = traces
            .iter()
            .filter(|t| (t.top_prob as f64) < LOW_CONFIDENCE_TOP_PROB && t.rank > 0)
            .count() as f64;

        Some(HallucinationSignal {
            top_choice_rate: top_choice / n,
            mean_rank,
            max_rank: traces.iter().map(|t| t.rank).max().unwrap_or(0),
            mean_confidence_gap,
            uncertain_offpick_rate: uncertain_offpick / n,
            tokens: traces.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(rank: u32, top_prob: f32, chosen_p: f32) -> TokenTrace {
        TokenTrace {
            token_id: 1,
            logprob: chosen_p.ln(),
            rank,
            entropy: 1.0,
            top_prob,
            since_prev_ns: 0,
        }
    }

    #[test]
    fn greedy_decoding_is_all_top_choice_and_zero_gap() {
        // Every token is the argmax: rank 0, chosen prob == top prob.
        let ts = vec![trace(0, 0.9, 0.9); 6];
        let h = HallucinationSignal::compute(&ts).unwrap();
        assert!((h.top_choice_rate - 1.0).abs() < 1e-9);
        assert!(h.mean_rank < 1e-9);
        assert!(h.mean_confidence_gap < 1e-6);
        assert!(h.uncertain_offpick_rate < 1e-9);
    }

    #[test]
    fn overriding_a_confident_model_shows_a_large_gap() {
        // The model wanted a 0.95 token; we sampled a 0.01 one instead.
        let ts = vec![trace(7, 0.95, 0.01); 4];
        let h = HallucinationSignal::compute(&ts).unwrap();
        assert!((h.mean_confidence_gap - 0.94).abs() < 1e-3, "{}", h.mean_confidence_gap);
        assert_eq!(h.max_rank, 7);
        assert!(h.top_choice_rate < 1e-9);
        // NOT flagged as uncertain-offpick: the model was confident (0.95), we
        // just overrode it. That is a sampler decision, not model uncertainty —
        // conflating the two is exactly the error this metric must not make.
        assert!(h.uncertain_offpick_rate < 1e-9);
    }

    #[test]
    fn uncertain_offpick_needs_both_low_confidence_and_off_top() {
        // Model hedging (top_prob 0.1) AND we took a non-top token: the one
        // case that carries real signal.
        let hedging_offpick = vec![trace(3, 0.1, 0.05); 4];
        let h = HallucinationSignal::compute(&hedging_offpick).unwrap();
        assert!((h.uncertain_offpick_rate - 1.0).abs() < 1e-9);

        // Hedging, but we still took its top pick => not an off-pick.
        let hedging_toppick = vec![trace(0, 0.1, 0.1); 4];
        let h2 = HallucinationSignal::compute(&hedging_toppick).unwrap();
        assert!(h2.uncertain_offpick_rate < 1e-9);
    }

    #[test]
    fn confidence_gap_never_goes_negative() {
        // Float noise could make chosen_p exceed top_prob by an ulp; a negative
        // "gap" is meaningless and would drag the mean below zero.
        let ts = vec![trace(0, 0.5, 0.5000001); 3];
        let h = HallucinationSignal::compute(&ts).unwrap();
        assert!(h.mean_confidence_gap >= 0.0);
    }

    #[test]
    fn untraced_is_none() {
        assert!(HallucinationSignal::compute(&[]).is_none());
    }
}
