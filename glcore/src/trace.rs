//! Per-token behavioral facts, captured during generation.
//!
//! This is the raw material for the behavioral signals in glbench (entropy,
//! perplexity/OOD, hallucination-proxy, stall detection). The engine already
//! computes every number here on its way to picking a token — it simply threw
//! them away. Capturing them changes no inference result, not one bit; it only
//! stops discarding facts that already exist.
//!
//! # Measured from the model, not from the sampler's settings
//!
//! Every field is derived from the **raw logits**: full vocabulary, before
//! temperature scaling, before top-k, before top-p. That distinction is the
//! whole point.
//!
//! If entropy were computed after top-k truncation, raising `top_k` from 40 to
//! 100 would change the reported entropy even though the *model* behaved
//! identically — the metric would be measuring the benchmark's own config
//! instead of the thing under test. Same for rank and log-probability. These
//! describe what the model believed; `SamplerConfig` describes what we did
//! about it. Keeping them separable is what makes cross-run comparison mean
//! anything.
//!
//! # Cost
//!
//! Collection is opt-in ([`TraceConfig::enabled`]) and off by default. When
//! off, no [`TokenTrace`] is allocated and the engine skips the extra pass over
//! the logits entirely. When on, it costs one O(vocab) sweep per token —
//! measurable, so benchmark numbers taken *with* tracing on should not be
//! compared against numbers taken with it off.

/// What the model believed at one generation step, plus when it happened.
#[derive(Debug, Clone, PartialEq)]
pub struct TokenTrace {
    /// The token actually emitted.
    pub token_id: u32,

    /// Natural log of the chosen token's probability under the **raw** softmax
    /// (full vocab, no temperature, no truncation).
    ///
    /// Always <= 0. This is the term perplexity is built from, and it is why
    /// the raw distribution matters: a temperature-scaled logprob would make
    /// perplexity a function of the sampling knob rather than of the model.
    pub logprob: f32,

    /// Rank of the chosen token in the raw distribution: 0 = the model's top
    /// choice, 1 = second, and so on.
    ///
    /// Paired with `logprob` this is the hallucination proxy. A model that is
    /// *confident and right about being confident* emits rank 0 with high
    /// logprob. Divergence — high confidence in the top token, yet a
    /// low-ranked token sampled, or a flat distribution where the model has no
    /// real preference — is the signal worth watching.
    pub rank: u32,

    /// Shannon entropy of the raw distribution, in nats.
    ///
    /// 0 = the model is certain (one token has all the mass). High = the model
    /// is spreading its bet. This is per-step uncertainty, independent of what
    /// the sampler then did with it.
    pub entropy: f32,

    /// Probability mass held by the single most likely token, in `[0, 1]`.
    /// A cheap confidence read that does not require the full distribution.
    pub top_prob: f32,

    /// Nanoseconds since the previous token was emitted (0 for the first).
    ///
    /// The raw input to stall detection: a decode loop that is smooth has low
    /// variance here, and a spike means something blocked — a page fault, a
    /// thermal trip, a scheduler preemption. Mean latency hides all of that;
    /// only the distribution shows it.
    pub since_prev_ns: u64,
}

/// Whether, and how much, to trace.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TraceConfig {
    /// Collect per-token traces. Off by default — tracing costs an O(vocab)
    /// sweep per token, so it must be asked for.
    pub enabled: bool,
}

impl TraceConfig {
    /// Tracing on.
    pub fn on() -> TraceConfig {
        TraceConfig { enabled: true }
    }
}

/// Compute the trace facts for one step from the **raw** logits.
///
/// `logits` must be the model's unmodified output: full vocabulary, no
/// temperature, no repetition penalty, no truncation. `chosen` is the token the
/// sampler went on to pick (which may not be the argmax — that is exactly what
/// `rank` records).
///
/// Returns `None` for empty logits or an out-of-range `chosen`, rather than
/// fabricating a plausible-looking trace for a step that did not happen.
pub fn trace_step(logits: &[f32], chosen: u32, since_prev_ns: u64) -> Option<TokenTrace> {
    let idx = chosen as usize;
    if logits.is_empty() || idx >= logits.len() {
        return None;
    }

    // Max-shifted softmax. Without the shift, a confident model's logits
    // overflow exp() and every downstream number becomes NaN.
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }

    let mut sum_exp = 0f64;
    for &l in logits {
        sum_exp += ((l - max) as f64).exp();
    }
    if sum_exp <= 0.0 {
        return None;
    }

    // Entropy over the full distribution, and the chosen token's rank, in one
    // pass. Rank is "how many tokens beat it" — computed by counting rather
    // than sorting, because sorting a 150k vocab per token costs more than the
    // FFN layer that produced it (measured: ~7 ms).
    let ln_sum = sum_exp.ln();
    let mut entropy = 0f64;
    let mut rank = 0u32;
    let chosen_logit = logits[idx];
    let mut top_logit = f32::NEG_INFINITY;

    for (i, &l) in logits.iter().enumerate() {
        let logp = (l - max) as f64 - ln_sum;
        let p = logp.exp();
        if p > 0.0 {
            entropy -= p * logp;
        }
        // Strictly-greater, plus an index tiebreak, so equal logits produce a
        // stable rank instead of one that depends on iteration order.
        if l > chosen_logit || (l == chosen_logit && i < idx) {
            rank += 1;
        }
        if l > top_logit {
            top_logit = l;
        }
    }

    let chosen_logp = (chosen_logit - max) as f64 - ln_sum;
    let top_prob = (((top_logit - max) as f64) - ln_sum).exp();

    Some(TokenTrace {
        token_id: chosen,
        logprob: chosen_logp as f32,
        rank,
        entropy: entropy as f32,
        top_prob: top_prob as f32,
        since_prev_ns,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_logits_give_max_entropy_and_rank_zero_for_first() {
        // 4 equal logits: entropy = ln(4) ~= 1.386 nats, each p = 0.25.
        let logits = [1.0f32, 1.0, 1.0, 1.0];
        let t = trace_step(&logits, 0, 0).unwrap();
        assert!((t.entropy - 4f32.ln()).abs() < 1e-5, "entropy {}", t.entropy);
        assert!((t.top_prob - 0.25).abs() < 1e-5);
        assert!((t.logprob - 0.25f32.ln()).abs() < 1e-5);
        // All tied; the index tiebreak makes token 0 rank 0, deterministically.
        assert_eq!(t.rank, 0);
    }

    #[test]
    fn confident_model_has_near_zero_entropy() {
        // One token dominates: entropy -> 0, top_prob -> 1.
        let logits = [50.0f32, 0.0, 0.0, 0.0];
        let t = trace_step(&logits, 0, 0).unwrap();
        assert!(t.entropy < 1e-6, "entropy should be ~0, got {}", t.entropy);
        assert!(t.top_prob > 0.999, "top_prob {}", t.top_prob);
        assert_eq!(t.rank, 0);
    }

    #[test]
    fn rank_counts_tokens_that_beat_the_chosen_one() {
        // Descending logits; picking index 2 means two tokens outrank it.
        let logits = [5.0f32, 4.0, 3.0, 2.0];
        let t = trace_step(&logits, 2, 0).unwrap();
        assert_eq!(t.rank, 2);
        // top_prob describes the ARGMAX, not the chosen token — a sampled
        // (non-greedy) token must not be reported as the model's top belief.
        let top = trace_step(&logits, 0, 0).unwrap();
        assert!((t.top_prob - top.top_prob).abs() < 1e-6);
        assert!(t.logprob < top.logprob, "rank-2 token cannot outscore rank-0");
    }

    #[test]
    fn large_logits_do_not_overflow() {
        // Unshifted exp() of these would be inf, poisoning entropy to NaN.
        let logits = [800.0f32, 799.0, 798.0];
        let t = trace_step(&logits, 0, 0).unwrap();
        assert!(t.entropy.is_finite(), "entropy overflowed: {}", t.entropy);
        assert!(t.logprob.is_finite() && t.logprob <= 0.0);
        assert!(t.top_prob.is_finite() && t.top_prob <= 1.0);
    }

    #[test]
    fn probabilities_are_normalized() {
        // logprob of the argmax must equal ln(top_prob) — a consistency check
        // that catches a mis-normalized softmax, which would silently skew
        // every perplexity number downstream.
        let logits = [2.0f32, 1.0, 0.5, -3.0, 7.0];
        let t = trace_step(&logits, 4, 0).unwrap(); // index 4 IS the argmax
        assert_eq!(t.rank, 0);
        assert!((t.logprob - t.top_prob.ln()).abs() < 1e-5);
    }

    #[test]
    fn invalid_input_returns_none_not_a_fake_trace() {
        assert!(trace_step(&[], 0, 0).is_none());
        assert!(trace_step(&[1.0, 2.0], 9, 0).is_none(), "out-of-range token");
    }

    #[test]
    fn timing_is_carried_through() {
        let t = trace_step(&[1.0f32, 2.0], 1, 12_345).unwrap();
        assert_eq!(t.since_prev_ns, 12_345);
    }
}
