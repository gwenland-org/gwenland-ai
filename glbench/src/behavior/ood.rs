//! Out-of-distribution — perplexity, and how far it drifts from a baseline.
//!
//! Perplexity is `exp(-mean logprob)` over the emitted tokens: the effective
//! number of tokens the model was choosing between. Low = the output was
//! unsurprising to the model; high = it kept picking things it considered
//! unlikely.
//!
//! # What a perplexity spike does and does not tell you
//!
//! It says the model found *its own output* surprising. That happens when the
//! prompt is out of distribution, but it also happens when temperature is high
//! (the sampler is deliberately picking unlikely tokens), when the prompt is
//! simply hard, or when a weight is corrupted. It is a **screening** signal:
//! worth investigating, never a diagnosis on its own.
//!
//! Note the sampler's settings do influence this one legitimately — a high
//! temperature really does make the model emit tokens it rates as unlikely, and
//! that really is higher perplexity. The `logprob` itself is still read from the
//! raw distribution, so what we measure is "how unlikely was this token to the
//! model", not "how unlikely after we reshaped the distribution".

use glcore::trace::TokenTrace;

use super::quantile;

/// Perplexity of a generation, with an optional comparison to a baseline.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OodSignal {
    /// `exp(-mean logprob)`. Effective branching factor of the output.
    pub perplexity: f64,
    /// Mean log-probability of the emitted tokens, nats. Always <= 0.
    pub mean_logprob: f64,
    /// Worst (most negative) single-token logprob — the most surprising token.
    pub min_logprob: f64,
    /// 95th-percentile surprise: `-logprob` at p95. High means a long tail of
    /// tokens the model did not expect.
    pub p95_surprise: f64,
    /// Tokens measured.
    pub tokens: usize,
}

impl OodSignal {
    /// `None` when nothing was traced.
    pub fn compute(traces: &[TokenTrace]) -> Option<OodSignal> {
        if traces.is_empty() {
            return None;
        }
        let lps: Vec<f64> = traces.iter().map(|t| t.logprob as f64).collect();
        let n = lps.len() as f64;
        let mean_logprob = lps.iter().sum::<f64>() / n;

        // Surprise = -logprob, so the p95 is the 95th-percentile *worst* token.
        let surprise: Vec<f64> = lps.iter().map(|l| -l).collect();

        Some(OodSignal {
            perplexity: (-mean_logprob).exp(),
            mean_logprob,
            min_logprob: lps.iter().copied().fold(f64::INFINITY, f64::min),
            p95_surprise: quantile(&surprise, 0.95)?,
            tokens: traces.len(),
        })
    }

    /// Ratio of this run's perplexity to a baseline's. `> 1.0` means this run
    /// was more surprising to the model than the baseline was.
    ///
    /// Returns `None` if the baseline perplexity is not positive — dividing by
    /// it would manufacture an infinity that looks like a finding.
    pub fn spike_vs(&self, baseline: &OodSignal) -> Option<f64> {
        (baseline.perplexity > 0.0).then(|| self.perplexity / baseline.perplexity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(logprob: f32) -> TokenTrace {
        TokenTrace {
            token_id: 1,
            logprob,
            rank: 0,
            entropy: 1.0,
            top_prob: 0.5,
            since_prev_ns: 0,
        }
    }

    #[test]
    fn a_perfectly_confident_model_has_perplexity_one() {
        // logprob 0 => probability 1 => the model was never choosing between
        // anything. Perplexity 1 is the floor.
        let ts = vec![trace(0.0); 4];
        let o = OodSignal::compute(&ts).unwrap();
        assert!((o.perplexity - 1.0).abs() < 1e-9, "got {}", o.perplexity);
    }

    #[test]
    fn uniform_over_k_tokens_gives_perplexity_k() {
        // Every token drawn with p = 1/8 => perplexity should read 8, the
        // effective branching factor. This is the definition, and it is the
        // check that would catch a sign error or a missing exp().
        let lp = (1.0f64 / 8.0).ln() as f32;
        let ts = vec![trace(lp); 10];
        let o = OodSignal::compute(&ts).unwrap();
        assert!((o.perplexity - 8.0).abs() < 1e-4, "got {}", o.perplexity);
    }

    #[test]
    fn spike_ratio_compares_against_baseline() {
        let base = OodSignal::compute(&vec![trace((0.5f64).ln() as f32); 4]).unwrap(); // ppl 2
        let hot = OodSignal::compute(&vec![trace((0.125f64).ln() as f32); 4]).unwrap(); // ppl 8
        let ratio = hot.spike_vs(&base).unwrap();
        assert!((ratio - 4.0).abs() < 1e-4, "8/2 should be 4x, got {ratio}");
    }

    #[test]
    fn p95_surprise_tracks_the_worst_tokens() {
        // Nine easy tokens and one very surprising one: the mean barely moves,
        // but p95 surprise catches the outlier. That gap is the point of
        // reporting both.
        let mut ts = vec![trace(-0.1); 9];
        ts.push(trace(-10.0));
        let o = OodSignal::compute(&ts).unwrap();
        assert!(o.min_logprob <= -10.0 + 1e-6);
        assert!(o.p95_surprise > 5.0, "p95 surprise {} too low", o.p95_surprise);
    }

    #[test]
    fn zero_baseline_yields_none_not_infinity() {
        let mut base = OodSignal::compute(&[trace(0.0)]).unwrap();
        base.perplexity = 0.0;
        let run = OodSignal::compute(&[trace(-1.0)]).unwrap();
        assert!(run.spike_vs(&base).is_none(), "must not divide by zero");
    }

    #[test]
    fn untraced_is_none() {
        assert!(OodSignal::compute(&[]).is_none());
    }
}
