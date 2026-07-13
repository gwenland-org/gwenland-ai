//! Stall — inter-token latency spikes.
//!
//! Mean tok/s hides stalls completely. A run that emits 30 tokens in 30 ms and
//! then blocks for 900 ms reports the same average as one that emits steadily
//! every 31 ms — but only one of them is usable interactively, and only one of
//! them has a bug.
//!
//! Causes worth separating, all of which look identical in the mean:
//!
//! - **Page fault**: a weight page was evicted and had to come back from disk.
//!   `warm_and_lock_model` exists to prevent this; a stall here means it failed.
//! - **Thermal throttle**: sustained load, clock drops. Shows as a *widening*
//!   trend rather than isolated spikes.
//! - **Scheduler preemption**: another process took the core. Isolated, random.
//! - **Allocator**: a buffer grew mid-loop and faulted in fresh pages.
//!
//! This module reports the distribution and flags the outliers; it does not
//! attribute a cause, because the timing alone cannot distinguish them.

use glcore::trace::TokenTrace;

use super::{mean_std, quantile};

/// How steady the decode loop was.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StallSignal {
    /// Mean inter-token latency, ms.
    pub mean_ms: f64,
    /// Standard deviation, ms. The headline number: a healthy decode loop is
    /// nearly constant, so a large value is itself the finding.
    pub std_dev_ms: f64,
    /// Median — resistant to the spikes, so `p50` far below `mean` is itself
    /// evidence that a few outliers are dragging the average.
    pub p50_ms: f64,
    /// 99th percentile: the tail a user actually feels.
    pub p99_ms: f64,
    /// Slowest single inter-token gap, ms.
    pub max_ms: f64,
    /// Tokens whose gap exceeded [`STALL_FACTOR`]x the median. Isolated spikes,
    /// not a general slowdown.
    pub stall_count: usize,
    /// Coefficient of variation (std/mean) — a scale-free jitter measure, so a
    /// slow-but-steady engine and a fast-but-steady one are comparable.
    pub jitter: f64,
    /// Inter-token gaps measured (one fewer than tokens: the first has no
    /// predecessor).
    pub samples: usize,
}

/// A gap this many times the median counts as a stall. 3x is deliberately
/// generous — normal jitter on a loaded desktop reaches 1.5–2x, so this fires
/// on genuine blocking rather than on noise.
const STALL_FACTOR: f64 = 3.0;

impl StallSignal {
    /// `None` when fewer than two tokens were traced — an inter-token gap needs
    /// two tokens, and the first token's `since_prev_ns` is 0 by definition, so
    /// including it would fabricate a zero-latency sample.
    pub fn compute(traces: &[TokenTrace]) -> Option<StallSignal> {
        if traces.len() < 2 {
            return None;
        }
        // Skip index 0: its `since_prev_ns` is 0 (no predecessor), and counting
        // that as a 0 ms gap would halve the reported mean on a 2-token run.
        let gaps: Vec<f64> = traces[1..]
            .iter()
            .map(|t| t.since_prev_ns as f64 / 1e6)
            .collect();

        let (mean_ms, std_dev_ms) = mean_std(&gaps)?;
        let p50_ms = quantile(&gaps, 0.5)?;
        let threshold = p50_ms * STALL_FACTOR;

        Some(StallSignal {
            mean_ms,
            std_dev_ms,
            p50_ms,
            p99_ms: quantile(&gaps, 0.99)?,
            max_ms: gaps.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            stall_count: gaps.iter().filter(|&&g| g > threshold).count(),
            jitter: if mean_ms > 0.0 { std_dev_ms / mean_ms } else { 0.0 },
            samples: gaps.len(),
        })
    }

    /// Whether the loop had spikes worth investigating. A display hint, not a
    /// verdict.
    pub fn has_stalls(&self) -> bool {
        self.stall_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traces_from_gaps_ms(gaps: &[f64]) -> Vec<TokenTrace> {
        // The first trace has no predecessor (gap 0); the rest carry the gaps.
        let mut ts = vec![t(0)];
        ts.extend(gaps.iter().map(|g| t((g * 1e6) as u64)));
        ts
    }

    fn t(since_prev_ns: u64) -> TokenTrace {
        TokenTrace {
            token_id: 1,
            logprob: -1.0,
            rank: 0,
            entropy: 1.0,
            top_prob: 0.5,
            since_prev_ns,
        }
    }

    #[test]
    fn a_steady_loop_has_no_stalls_and_near_zero_jitter() {
        let ts = traces_from_gaps_ms(&[90.0; 10]);
        let s = StallSignal::compute(&ts).unwrap();
        assert!((s.mean_ms - 90.0).abs() < 1e-6);
        assert!(s.jitter < 1e-6);
        assert_eq!(s.stall_count, 0);
        assert!(!s.has_stalls());
    }

    #[test]
    fn one_long_block_is_caught_even_though_the_mean_barely_moves() {
        // Nine steady 90 ms tokens and one 900 ms block. Mean rises to 171 ms —
        // which alone looks like a merely-slow model. The median stays at 90 and
        // the stall count catches the real story.
        let mut gaps = vec![90.0; 9];
        gaps.push(900.0);
        let s = StallSignal::compute(&traces_from_gaps_ms(&gaps)).unwrap();
        assert!((s.p50_ms - 90.0).abs() < 1e-6, "median must resist the spike");
        assert_eq!(s.stall_count, 1, "the 900ms block must be flagged");
        assert!((s.max_ms - 900.0).abs() < 1e-6);
        assert!(s.has_stalls());
        // The whole point: mean is misleading, median + stall_count is not.
        assert!(s.mean_ms > 150.0, "mean is dragged up by the spike");
    }

    #[test]
    fn first_token_is_excluded_not_counted_as_a_zero_gap() {
        // Two 100ms gaps. If index 0's zero were included the mean would read
        // 66.7 ms instead of 100 ms — a 33% understatement.
        let ts = traces_from_gaps_ms(&[100.0, 100.0]);
        let s = StallSignal::compute(&ts).unwrap();
        assert_eq!(s.samples, 2);
        assert!((s.mean_ms - 100.0).abs() < 1e-6, "got {}", s.mean_ms);
    }

    #[test]
    fn single_token_has_no_gap_to_measure() {
        assert!(StallSignal::compute(&[t(0)]).is_none());
        assert!(StallSignal::compute(&[]).is_none());
    }
}
