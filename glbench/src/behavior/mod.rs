//! Behavioral signals — what the model *did*, measured in pure numbers.
//!
//! glbench observes; it does not optimize, and it does not judge. Every signal
//! here is computed from facts the engine already produced (token ids, raw
//! per-token distributions, stage timings) and reported as a number. None of it
//! feeds back into the engine, and none of it changes a single line of what the
//! engine runs.
//!
//! # The signals, and what each one can honestly claim
//!
//! | module | measures | needs |
//! |--------|----------|-------|
//! | [`repetition`] | n-gram reuse in the output | token ids |
//! | [`entropy`] | per-step uncertainty of the distribution | token trace |
//! | [`stall`] | inter-token latency spikes | token trace |
//! | [`ood`] | perplexity, and its gap vs a baseline | token trace |
//! | [`hallucination`] | confidence/rank divergence (a **proxy**) | token trace |
//! | [`drift`] | Δ ms/call between sessions | telemetry |
//! | [`performance`] | ms/call, share, layer variance | telemetry |
//! | [`toxicity`] | **not implemented** — see the module | — |
//!
//! # Two warnings that belong at the top, not the bottom
//!
//! **The trace-based signals require raw distributions.** They are computed
//! from the model's untouched logits — full vocabulary, before temperature,
//! before top-k/top-p, before the repetition penalty. If they were computed
//! after sampling had truncated the distribution, changing `top_k` would move
//! the numbers while the model behaved identically, and every cross-run
//! comparison would be measuring our own config. See [`glcore::trace`].
//!
//! **`hallucination` is a proxy, and its name oversells it.** It measures
//! confidence/rank divergence, which *correlates* with confabulation but does
//! not detect it: a model can be confidently wrong (low divergence, false
//! statement) or uncertain and right (high divergence, true statement).
//! Detecting hallucination requires ground truth, which a profiler does not
//! have. Read the number as "how sure was the model about what it picked",
//! never as "how much did it make up".

pub mod drift;
pub mod entropy;
pub mod hallucination;
pub mod ood;
pub mod performance;
pub mod repetition;
pub mod stall;
pub mod toxicity;

use glcore::trace::TokenTrace;

/// Every behavioral signal glbench could compute for one run.
///
/// Each field is `Option`: a signal whose inputs were not captured is `None`,
/// never a zero. A zeroed metric is a claim ("the model repeated nothing");
/// an absent one is an admission ("we did not look").
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BehaviorReport {
    /// N-gram reuse. Available whenever tokens were generated.
    pub repetition: Option<repetition::RepetitionSignal>,
    /// Distribution spread per step. Needs tracing.
    pub entropy: Option<entropy::EntropySignal>,
    /// Inter-token latency spikes. Needs tracing.
    pub stall: Option<stall::StallSignal>,
    /// Perplexity. Needs tracing.
    pub ood: Option<ood::OodSignal>,
    /// Confidence/rank divergence proxy. Needs tracing.
    pub hallucination: Option<hallucination::HallucinationSignal>,
}

impl BehaviorReport {
    /// Compute every signal the available data supports.
    ///
    /// `tokens` is always present; `traces` is empty unless the run asked for
    /// tracing, in which case the four distribution-based signals stay `None`
    /// rather than being estimated from what is missing.
    pub fn compute(tokens: &[u32], traces: &[TokenTrace]) -> BehaviorReport {
        BehaviorReport {
            repetition: repetition::RepetitionSignal::compute(tokens),
            entropy: entropy::EntropySignal::compute(traces),
            stall: stall::StallSignal::compute(traces),
            ood: ood::OodSignal::compute(traces),
            hallucination: hallucination::HallucinationSignal::compute(traces),
        }
    }

    /// True when nothing could be computed at all.
    pub fn is_empty(&self) -> bool {
        self.repetition.is_none()
            && self.entropy.is_none()
            && self.stall.is_none()
            && self.ood.is_none()
            && self.hallucination.is_none()
    }
}

/// Mean and standard deviation of a sample, or `None` if it is empty.
///
/// Shared by several signals. Population (not sample) standard deviation: we
/// have the whole run, not a draw from it.
pub(crate) fn mean_std(xs: &[f64]) -> Option<(f64, f64)> {
    if xs.is_empty() {
        return None;
    }
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    Some((mean, var.sqrt()))
}

/// The `p`-quantile (0.0–1.0) of an unsorted sample, by nearest rank.
pub(crate) fn quantile(xs: &[f64], p: f64) -> Option<f64> {
    if xs.is_empty() {
        return None;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let idx = ((p * (v.len() - 1) as f64).round() as usize).min(v.len() - 1);
    Some(v[idx])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_std_of_constant_sample_has_zero_spread() {
        let (m, s) = mean_std(&[5.0, 5.0, 5.0]).unwrap();
        assert!((m - 5.0).abs() < 1e-9);
        assert!(s < 1e-9);
    }

    #[test]
    fn mean_std_empty_is_none_not_zero() {
        assert!(mean_std(&[]).is_none());
    }

    #[test]
    fn quantile_picks_by_rank() {
        let xs = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&xs, 0.0), Some(1.0));
        assert_eq!(quantile(&xs, 1.0), Some(5.0));
        assert_eq!(quantile(&xs, 0.5), Some(3.0));
    }

    #[test]
    fn untraced_run_leaves_distribution_signals_none() {
        // Tokens but no traces: repetition is computable, the rest are NOT.
        // They must be None (not measured), never Some(0.0) (measured as zero).
        let r = BehaviorReport::compute(&[1, 2, 3, 1, 2, 3], &[]);
        assert!(r.repetition.is_some(), "repetition needs only token ids");
        assert!(r.entropy.is_none());
        assert!(r.stall.is_none());
        assert!(r.ood.is_none());
        assert!(r.hallucination.is_none());
    }
}
