//! Entropy — how spread the model's per-step distribution was.
//!
//! Low entropy means the model was certain; high means it was hedging across
//! many tokens. Neither is "good": a factual completion *should* be low-entropy,
//! and creative text *should* be higher. What matters is the shape and the
//! change — a model that collapses to near-zero entropy mid-generation is
//! looping, and one that spikes has lost the thread.
//!
//! Measured in **nats**, from the raw distribution (full vocab, pre-temperature,
//! pre-truncation). See [`glcore::trace`] for why that matters: entropy taken
//! after top-k would move when `top_k` moved, measuring our config instead of
//! the model.

use glcore::trace::TokenTrace;

use super::{mean_std, quantile};

/// Distribution spread across a generation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntropySignal {
    /// Mean per-token entropy, nats.
    pub mean: f64,
    /// Standard deviation — how much the model's certainty varied step to step.
    pub std_dev: f64,
    /// Lowest per-token entropy seen. Near 0 = the model was locked in.
    pub min: f64,
    /// Highest per-token entropy seen.
    pub max: f64,
    /// Median.
    pub p50: f64,
    /// 95th percentile — the "how uncertain did it get at its worst" number.
    pub p95: f64,
    /// Mean probability of the model's top choice, `[0, 1]`. A second read on
    /// confidence that does not depend on vocabulary size the way entropy does
    /// (entropy's ceiling is ln(vocab), so it is not comparable across models
    /// with different vocabularies — this is).
    pub mean_top_prob: f64,
    /// Tokens measured.
    pub tokens: usize,
}

impl EntropySignal {
    /// `None` when nothing was traced — not measured is not the same as zero.
    pub fn compute(traces: &[TokenTrace]) -> Option<EntropySignal> {
        if traces.is_empty() {
            return None;
        }
        let e: Vec<f64> = traces.iter().map(|t| t.entropy as f64).collect();
        let (mean, std_dev) = mean_std(&e)?;
        let tops: Vec<f64> = traces.iter().map(|t| t.top_prob as f64).collect();
        let (mean_top_prob, _) = mean_std(&tops)?;

        Some(EntropySignal {
            mean,
            std_dev,
            min: e.iter().copied().fold(f64::INFINITY, f64::min),
            max: e.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            p50: quantile(&e, 0.5)?,
            p95: quantile(&e, 0.95)?,
            mean_top_prob,
            tokens: traces.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trace(entropy: f32, top_prob: f32) -> TokenTrace {
        TokenTrace {
            token_id: 1,
            logprob: -0.5,
            rank: 0,
            entropy,
            top_prob,
            since_prev_ns: 0,
        }
    }

    #[test]
    fn constant_entropy_has_zero_spread() {
        let ts = vec![trace(2.0, 0.4); 5];
        let e = EntropySignal::compute(&ts).unwrap();
        assert!((e.mean - 2.0).abs() < 1e-9);
        assert!(e.std_dev < 1e-9);
        assert!((e.min - e.max).abs() < 1e-9);
    }

    #[test]
    fn varying_entropy_reports_range_and_percentiles() {
        let ts: Vec<TokenTrace> = (0..=10)
            .map(|i| trace(i as f32 * 0.5, 0.5))
            .collect();
        let e = EntropySignal::compute(&ts).unwrap();
        assert!((e.min - 0.0).abs() < 1e-6);
        assert!((e.max - 5.0).abs() < 1e-6);
        assert!((e.p50 - 2.5).abs() < 1e-6);
        assert!(e.p95 >= e.p50, "p95 must not fall below the median");
        assert_eq!(e.tokens, 11);
    }

    #[test]
    fn a_collapsed_model_shows_near_zero_entropy_and_top_prob_one() {
        // The signature of a looping / degenerate decode.
        let ts = vec![trace(0.0001, 0.999); 8];
        let e = EntropySignal::compute(&ts).unwrap();
        assert!(e.mean < 0.01);
        assert!(e.mean_top_prob > 0.99);
    }

    #[test]
    fn untraced_is_none() {
        assert!(EntropySignal::compute(&[]).is_none());
    }
}
