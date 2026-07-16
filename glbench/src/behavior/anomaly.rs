//! Intra-session anomaly facts — drift and spikes *within one generation*.
//!
//! [`super::drift`] compares two sessions; [`super::stall`] reports the gap
//! distribution with position thrown away. This module keeps the position:
//! it buckets the token stream into quarters and asks two questions the
//! aggregate views cannot answer —
//!
//! - **Did the loop get slower as the sequence grew?** (`drift_frac`,
//!   `quarter_gap_ms`). A rising staircase across quarters is the KV-cache
//!   growth signature; a flat run with isolated spikes is not drift at all.
//! - **Did the model hit a patch of text it found surprising?**
//!   (`spike_ratio`, `spike_token`) — the worst sliding-window perplexity
//!   against the whole-run perplexity, with the token index where it peaked.
//!
//! Facts only: the numbers say *what* changed and *where*. Why it changed is
//! [`crate::analysis::hypothesis`]'s job, and cause attribution from timing
//! alone would over-claim (the stall module's warning applies here too).

use glcore::trace::TokenTrace;

use super::mean_std;

/// Perplexity window length, tokens. Short enough to localize a spike to a
/// region, long enough that one rare-but-correct token does not read as OOD.
const SPIKE_WINDOW: usize = 32;

/// Positional signals across one generation.
#[derive(Debug, Clone, PartialEq)]
pub struct AnomalySignal {
    /// Mean inter-token gap per quarter of the generation, ms. The shape:
    /// a monotonic rise is growth-driven drift, a single hot quarter is an
    /// incident, flat is healthy.
    pub quarter_gap_ms: [f64; 4],
    /// Relative gap change, last quarter vs first: `(q4 - q1) / q1`.
    /// +0.12 = the loop ended 12% slower per token than it started.
    pub drift_frac: f64,
    /// Worst [`SPIKE_WINDOW`]-token window perplexity divided by the whole-run
    /// perplexity. `None` when the run is shorter than one window — a spike
    /// needs a baseline around it to be a spike.
    pub spike_ratio: Option<f64>,
    /// Token index (0-based, into the generated stream) where the worst
    /// window starts. `None` alongside `spike_ratio`.
    pub spike_token: Option<usize>,
    /// Inter-token gaps measured.
    pub samples: usize,
}

/// Drift beyond this fraction is worth surfacing as a display hint.
const DRIFT_HINT_FRAC: f64 = 0.10;

impl AnomalySignal {
    /// `None` when fewer than 8 gaps were traced: with under two samples per
    /// quarter, a "quarter mean" would be a single noisy gap wearing a trend's
    /// clothing.
    pub fn compute(traces: &[TokenTrace]) -> Option<AnomalySignal> {
        // Index 0 carries no gap (no predecessor) — same exclusion as stall.
        if traces.len() < 9 {
            return None;
        }
        let gaps: Vec<f64> = traces[1..]
            .iter()
            .map(|t| t.since_prev_ns as f64 / 1e6)
            .collect();

        let q = gaps.len() / 4;
        let mut quarter_gap_ms = [0.0f64; 4];
        for (i, slot) in quarter_gap_ms.iter_mut().enumerate() {
            // Last quarter absorbs the remainder so no gap is dropped.
            let end = if i == 3 { gaps.len() } else { (i + 1) * q };
            let (mean, _) = mean_std(&gaps[i * q..end])?;
            *slot = mean;
        }
        let drift_frac = if quarter_gap_ms[0] > 0.0 {
            (quarter_gap_ms[3] - quarter_gap_ms[0]) / quarter_gap_ms[0]
        } else {
            0.0
        };

        let (spike_ratio, spike_token) = perplexity_spike(traces);

        Some(AnomalySignal {
            quarter_gap_ms,
            drift_frac,
            spike_ratio,
            spike_token,
            samples: gaps.len(),
        })
    }

    /// Whether the drift is large enough to mention. A display hint, not a
    /// verdict.
    pub fn has_drift(&self) -> bool {
        self.drift_frac.abs() > DRIFT_HINT_FRAC
    }

    /// True when the quarter means rise monotonically — the shape that points
    /// at sequence-length-driven cost (KV cache) rather than an incident.
    pub fn drift_is_monotonic(&self) -> bool {
        self.quarter_gap_ms.windows(2).all(|w| w[1] >= w[0])
    }
}

/// Worst window perplexity vs whole-run perplexity, and where it peaked.
fn perplexity_spike(traces: &[TokenTrace]) -> (Option<f64>, Option<usize>) {
    if traces.len() < SPIKE_WINDOW * 2 {
        // Need at least one window *and* enough outside it for the whole-run
        // figure to be a baseline rather than mostly the window itself.
        return (None, None);
    }
    let logprobs: Vec<f64> = traces.iter().map(|t| t.logprob as f64).collect();
    let overall = perplexity(&logprobs);
    if overall <= 0.0 {
        return (None, None);
    }
    let (mut worst, mut at) = (f64::NEG_INFINITY, 0);
    for start in 0..=(logprobs.len() - SPIKE_WINDOW) {
        let p = perplexity(&logprobs[start..start + SPIKE_WINDOW]);
        if p > worst {
            worst = p;
            at = start;
        }
    }
    (Some(worst / overall), Some(at))
}

/// `exp(-mean logprob)` — the standard definition, matching [`super::ood`].
fn perplexity(logprobs: &[f64]) -> f64 {
    let mean = logprobs.iter().sum::<f64>() / logprobs.len() as f64;
    (-mean).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(since_prev_ns: u64, logprob: f32) -> TokenTrace {
        TokenTrace {
            token_id: 1,
            logprob,
            rank: 0,
            entropy: 1.0,
            top_prob: 0.5,
            since_prev_ns,
        }
    }

    fn traces(gaps_ms: &[f64], logprob: f32) -> Vec<TokenTrace> {
        let mut ts = vec![t(0, logprob)];
        ts.extend(gaps_ms.iter().map(|g| t((g * 1e6) as u64, logprob)));
        ts
    }

    #[test]
    fn a_flat_run_has_no_drift() {
        let ts = traces(&[50.0; 40], -1.0);
        let a = AnomalySignal::compute(&ts).unwrap();
        assert!(a.drift_frac.abs() < 1e-9);
        assert!(!a.has_drift());
    }

    #[test]
    fn a_growing_kv_cache_shows_as_monotonic_drift() {
        // Gaps rising steadily 50 -> 60 ms across the run: +20% end-to-end.
        let gaps: Vec<f64> = (0..40).map(|i| 50.0 + i as f64 * 0.25).collect();
        let a = AnomalySignal::compute(&traces(&gaps, -1.0)).unwrap();
        assert!(a.has_drift(), "drift_frac {}", a.drift_frac);
        assert!(a.drift_is_monotonic());
        assert!(a.quarter_gap_ms[3] > a.quarter_gap_ms[0]);
    }

    #[test]
    fn one_slow_quarter_is_drift_but_not_monotonic() {
        // Healthy, then a hot third quarter, then healthy: an incident, not growth.
        let mut gaps = vec![50.0; 40];
        for g in gaps.iter_mut().skip(20).take(10) {
            *g = 90.0;
        }
        let a = AnomalySignal::compute(&traces(&gaps, -1.0)).unwrap();
        assert!(!a.drift_is_monotonic());
    }

    #[test]
    fn perplexity_spike_locates_the_surprising_region() {
        // 128 easy tokens (logprob -0.1) with a hard patch (-4.0) in the middle.
        let mut ts = traces(&[50.0; 127], -0.1);
        for tr in ts.iter_mut().skip(60).take(32) {
            tr.logprob = -4.0;
        }
        let a = AnomalySignal::compute(&ts).unwrap();
        let ratio = a.spike_ratio.expect("long run must yield a spike ratio");
        assert!(ratio > 3.0, "hard patch must spike well past baseline, got {ratio}");
        let at = a.spike_token.unwrap();
        assert!((55..=65).contains(&at), "spike located at {at}, expected ~60");
    }

    #[test]
    fn short_runs_yield_no_signal_rather_than_noise() {
        assert!(AnomalySignal::compute(&traces(&[50.0; 7], -1.0)).is_none());
        // Long enough for quarters but not for a spike window + baseline.
        let a = AnomalySignal::compute(&traces(&[50.0; 20], -1.0)).unwrap();
        assert!(a.spike_ratio.is_none());
        assert!(a.spike_token.is_none());
    }
}
