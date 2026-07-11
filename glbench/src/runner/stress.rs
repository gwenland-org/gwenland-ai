//! Stress phase: sustained repeated requests for stability/thermal observation.
//!
//! A stress workload is just many measured iterations run back-to-back; the
//! interesting output is not the mean but the *drift* — does throughput sag as
//! the device heats or memory fragments? This module computes that drift signal
//! from the per-iteration series the runner already collected. It does not run
//! anything itself (the planner's measured loop already does the repetition).

use crate::core::metrics::MeasurementSet;

/// A stability verdict over a stress run's iteration series.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stability {
    /// Decode tok/s of the first quarter of iterations (early baseline).
    pub early_tps: f64,
    /// Decode tok/s of the last quarter of iterations (late, post-warmup drift).
    pub late_tps: f64,
}

impl Stability {
    /// Relative drift (late - early) / early; negative means throughput sagged.
    pub fn drift(&self) -> f64 {
        if self.early_tps == 0.0 {
            0.0
        } else {
            (self.late_tps - self.early_tps) / self.early_tps
        }
    }
}

/// Compute early-vs-late decode throughput over a measurement series. Needs at
/// least four iterations to split into quarters; returns `None` otherwise.
pub fn stability(m: &MeasurementSet) -> Option<Stability> {
    let samples = m.decode_tps_samples();
    let n = samples.len();
    if n < 4 {
        return None;
    }
    let q = n / 4;
    let early = mean(&samples[..q]);
    let late = mean(&samples[n - q..]);
    Some(Stability { early_tps: early, late_tps: late })
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::IterationMetrics;

    fn set(decode_ms_series: &[f64]) -> MeasurementSet {
        MeasurementSet {
            iterations: decode_ms_series
                .iter()
                .map(|&ms| IterationMetrics {
                    prompt_tokens: 10,
                    generated_tokens: 100,
                    prefill_ms: 5.0,
                    decode_ms: ms,
                    total_ms: ms + 5.0,
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn detects_sag() {
        // decode_ms rises over time -> tok/s falls -> negative drift.
        let m = set(&[1000.0, 1000.0, 1100.0, 1200.0, 1300.0, 1400.0, 1500.0, 1600.0]);
        let s = stability(&m).unwrap();
        assert!(s.drift() < 0.0);
    }

    #[test]
    fn too_few_iters_is_none() {
        assert!(stability(&set(&[1000.0, 1000.0])).is_none());
    }
}
