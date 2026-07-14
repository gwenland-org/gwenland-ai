//! Drift — how per-stage cost changed between two sessions.
//!
//! Two runs of the same model on the same machine should cost the same. When
//! they do not, the *shape* of the difference tells you why, and a single
//! tok/s delta cannot:
//!
//! - **Uniform slowdown across every stage** → the clock dropped. Thermal
//!   throttling, or a power-profile change. Nothing about the code changed.
//! - **One stage slower, others flat** → that stage regressed. A code change,
//!   a different kernel path, a weight that stopped fitting in cache.
//! - **Memory-heavy stages slower, compute-heavy stages flat** → the working
//!   set stopped fitting, or pages are being faulted back in.
//!
//! Distinguishing "the machine got hot" from "the code got slower" is the whole
//! reason this exists — on a 15 W laptop the first one is constant and will
//! masquerade as the second in any single-number comparison.
//!
//! This module reports the per-stage deltas and the uniformity of the change.
//! It does not name a cause; the timings alone cannot prove one.

use glcore::telemetry::PhaseProfile;

/// Per-stage change between a baseline and a candidate run.
#[derive(Debug, Clone, PartialEq)]
pub struct DriftSignal {
    /// Per-stage `(name, baseline ms/call, candidate ms/call, ratio)`.
    /// Ratio > 1.0 means the candidate is slower.
    pub stages: Vec<StageDrift>,
    /// Mean of the per-stage ratios.
    pub mean_ratio: f64,
    /// Spread of those ratios (coefficient of variation).
    ///
    /// **This is the diagnostic number.** Near 0 means every stage moved by the
    /// same factor — a clock change, not a code change. Large means the change
    /// hit some stages and not others, which is what a real regression looks
    /// like.
    pub ratio_spread: f64,
    /// Stage whose cost grew the most.
    pub worst_stage: Option<String>,
    /// Its ratio.
    pub worst_ratio: f64,
}

/// One stage's change between runs.
#[derive(Debug, Clone, PartialEq)]
pub struct StageDrift {
    pub name: String,
    pub baseline_ms_per_call: f64,
    pub candidate_ms_per_call: f64,
    /// candidate / baseline. > 1.0 = slower.
    pub ratio: f64,
}

/// Below this spread, the change is called uniform (i.e. a clock effect rather
/// than a per-stage regression). A convention, not a derivation — stated here
/// rather than buried in the code.
const UNIFORM_SPREAD_MAX: f64 = 0.15;

impl DriftSignal {
    /// Compare two runs of the same phase.
    ///
    /// Only stages present in **both** profiles are compared; a stage that
    /// exists in one run and not the other is not a slowdown, it is a different
    /// code path, and pretending otherwise would report an infinite ratio.
    pub fn compute(baseline: &PhaseProfile, candidate: &PhaseProfile) -> Option<DriftSignal> {
        let mut stages = Vec::new();

        for b in &baseline.stages {
            let Some(c) = candidate.stages.iter().find(|s| s.name == b.name) else {
                continue;
            };
            if b.calls == 0 || c.calls == 0 {
                continue;
            }
            let bp = b.total_ms / b.calls as f64;
            let cp = c.total_ms / c.calls as f64;
            if bp <= 0.0 {
                continue; // a zero baseline yields an infinite ratio, not a fact
            }
            stages.push(StageDrift {
                name: b.name.clone(),
                baseline_ms_per_call: bp,
                candidate_ms_per_call: cp,
                ratio: cp / bp,
            });
        }

        if stages.is_empty() {
            return None;
        }

        let ratios: Vec<f64> = stages.iter().map(|s| s.ratio).collect();
        let (mean_ratio, std) = super::mean_std(&ratios)?;
        let worst = stages
            .iter()
            .max_by(|a, b| a.ratio.total_cmp(&b.ratio))
            .cloned();

        Some(DriftSignal {
            mean_ratio,
            ratio_spread: if mean_ratio > 0.0 { std / mean_ratio } else { 0.0 },
            worst_ratio: worst.as_ref().map(|s| s.ratio).unwrap_or(1.0),
            worst_stage: worst.map(|s| s.name),
            stages,
        })
    }

    /// True when every stage moved by roughly the same factor — consistent with
    /// a clock change (thermal, power profile), not a code regression.
    ///
    /// A hint for the reader, not a diagnosis: only a >5% overall move is worth
    /// interpreting at all, and even then the timings cannot *prove* the cause.
    pub fn looks_like_clock_change(&self) -> bool {
        self.ratio_spread < UNIFORM_SPREAD_MAX && (self.mean_ratio - 1.0).abs() > 0.05
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glcore::telemetry::StageTiming;

    fn phase(stages: &[(&str, f64, u64)]) -> PhaseProfile {
        let stages: Vec<StageTiming> = stages
            .iter()
            .map(|&(n, ms, c)| StageTiming {
                name: n.into(),
                total_ms: ms,
                calls: c,
                bytes_read: None,
                macs: None,
            })
            .collect();
        let total_ms = stages.iter().map(|s| s.total_ms).sum();
        PhaseProfile { stages, total_ms }
    }

    #[test]
    fn a_uniform_slowdown_reads_as_a_clock_change() {
        // Every stage 20% slower: the machine throttled, the code did not
        // change. Spread is ~0, so this must NOT be reported as a regression.
        let base = phase(&[("ffn", 100.0, 10), ("attn", 50.0, 10), ("qkv", 25.0, 10)]);
        let cand = phase(&[("ffn", 120.0, 10), ("attn", 60.0, 10), ("qkv", 30.0, 10)]);
        let d = DriftSignal::compute(&base, &cand).unwrap();
        assert!((d.mean_ratio - 1.2).abs() < 1e-6);
        assert!(d.ratio_spread < 1e-6, "uniform change must have ~0 spread");
        assert!(d.looks_like_clock_change());
    }

    #[test]
    fn a_single_regressed_stage_does_not_look_like_a_clock_change() {
        // ffn doubled; everything else identical. This is a code regression,
        // and the spread is what distinguishes it from throttling.
        let base = phase(&[("ffn", 100.0, 10), ("attn", 50.0, 10), ("qkv", 25.0, 10)]);
        let cand = phase(&[("ffn", 200.0, 10), ("attn", 50.0, 10), ("qkv", 25.0, 10)]);
        let d = DriftSignal::compute(&base, &cand).unwrap();
        assert_eq!(d.worst_stage.as_deref(), Some("ffn"));
        assert!((d.worst_ratio - 2.0).abs() < 1e-6);
        assert!(d.ratio_spread > UNIFORM_SPREAD_MAX, "spread {}", d.ratio_spread);
        assert!(!d.looks_like_clock_change(), "must not blame the clock");
    }

    #[test]
    fn identical_runs_show_no_drift_and_no_clock_claim() {
        let p = phase(&[("ffn", 100.0, 10), ("attn", 50.0, 10)]);
        let d = DriftSignal::compute(&p, &p).unwrap();
        assert!((d.mean_ratio - 1.0).abs() < 1e-9);
        // Spread is 0, but the move is 0 too — claiming a "clock change" for an
        // identical run would be nonsense.
        assert!(!d.looks_like_clock_change());
    }

    #[test]
    fn stages_missing_from_one_run_are_skipped_not_treated_as_infinite() {
        // The candidate took a different code path and has no "attn" stage.
        // That is not an infinite speedup; it is not comparable at all.
        let base = phase(&[("ffn", 100.0, 10), ("attn", 50.0, 10)]);
        let cand = phase(&[("ffn", 110.0, 10)]);
        let d = DriftSignal::compute(&base, &cand).unwrap();
        assert_eq!(d.stages.len(), 1, "only the shared stage is comparable");
        assert!(d.mean_ratio.is_finite());
    }

    #[test]
    fn nothing_in_common_is_none() {
        let base = phase(&[("a", 1.0, 1)]);
        let cand = phase(&[("b", 1.0, 1)]);
        assert!(DriftSignal::compute(&base, &cand).is_none());
    }
}
