//! Performance — ms/call, share, and where the variance lives.
//!
//! Built entirely on [`glcore::telemetry`], which the engine already fills. This
//! module only ranks and summarizes; it computes no timing of its own.
//!
//! # ms/call vs share: two different questions
//!
//! `share` answers *where did the wall time go* — the optimization target.
//! `ms/call` answers *is this stage expensive, or merely frequent* — which
//! decides what kind of fix applies. They diverge sharply in practice: on
//! Qwen3-1.7B decode, `lm_head` costs 22.5 ms/call (15x any other stage) yet is
//! only 9.8% of total, because it runs once per token while the per-layer stages
//! run 28 times. Optimizing by `share` alone would send you to the FFN;
//! optimizing by `ms/call` alone would send you to the LM head. You need both.

use glcore::telemetry::PhaseProfile;

/// Summary of one phase's stage breakdown.
#[derive(Debug, Clone, PartialEq)]
pub struct PerformanceSignal {
    /// Stage with the largest share of wall time — the optimization target.
    pub hottest_stage: String,
    /// Its share of the phase, `[0, 1]`.
    pub hottest_share: f64,
    /// Stage with the highest per-call cost — expensive, though possibly rare.
    pub costliest_call: String,
    /// Its per-call cost, ms.
    pub costliest_ms_per_call: f64,
    /// Time not attributed to any named stage, as a fraction of the phase.
    /// A large value means the engine's instrumentation has a blind spot, and
    /// every share below it is correspondingly overstated.
    pub unattributed_share: f64,
    /// Coefficient of variation of per-call cost across stages — how unevenly
    /// the work is distributed. Near 0 means every stage costs about the same
    /// per call; high means one dominates.
    pub call_cost_spread: f64,
}

impl PerformanceSignal {
    /// `None` for a phase that did not run or reported no stages.
    pub fn compute(p: &PhaseProfile) -> Option<PerformanceSignal> {
        if p.stages.is_empty() || p.total_ms <= 0.0 {
            return None;
        }
        let hottest = p.hotspots().first().copied()?;

        // Per-call cost, skipping stages that never ran — dividing by zero
        // calls would produce an infinity that then wins "costliest".
        let per_call: Vec<(&str, f64)> = p
            .stages
            .iter()
            .filter(|s| s.calls > 0)
            .map(|s| (s.name.as_str(), s.total_ms / s.calls as f64))
            .collect();
        let costliest = per_call
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .copied()?;

        let costs: Vec<f64> = per_call.iter().map(|&(_, c)| c).collect();
        let (mean, std) = super::mean_std(&costs)?;

        Some(PerformanceSignal {
            hottest_stage: hottest.name.clone(),
            hottest_share: hottest.share_of(p.total_ms).unwrap_or(0.0),
            costliest_call: costliest.0.to_string(),
            costliest_ms_per_call: costliest.1,
            unattributed_share: p.unattributed_ms() / p.total_ms,
            call_cost_spread: if mean > 0.0 { std / mean } else { 0.0 },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glcore::telemetry::StageTiming;

    fn stage(name: &str, ms: f64, calls: u64) -> StageTiming {
        StageTiming { name: name.into(), total_ms: ms, calls, bytes_read: None, macs: None }
    }

    #[test]
    fn hottest_and_costliest_can_be_different_stages() {
        // The real Qwen3-1.7B decode shape: ffn dominates total time, but
        // lm_head costs far more per call because it runs once per token
        // instead of once per layer. Conflating these sends you to the wrong
        // optimization.
        let p = PhaseProfile {
            stages: vec![
                stage("ffn_gate_up", 3491.45, 1344), // 2.6 ms/call, 31.6%
                stage("lm_head", 1080.36, 48),       // 22.5 ms/call, 9.8%
                stage("attention", 804.0, 1344),     // 0.6 ms/call
            ],
            total_ms: 11043.7,
        };
        let s = PerformanceSignal::compute(&p).unwrap();
        assert_eq!(s.hottest_stage, "ffn_gate_up", "share leader");
        assert_eq!(s.costliest_call, "lm_head", "per-call leader");
        assert!((s.costliest_ms_per_call - 22.5).abs() < 0.1);
        assert!(s.hottest_share > 0.3 && s.hottest_share < 0.33);
    }

    #[test]
    fn unattributed_time_is_reported_as_a_share() {
        // Stages sum to 90 of 100 ms: 10% blind spot.
        let p = PhaseProfile {
            stages: vec![stage("a", 50.0, 10), stage("b", 40.0, 10)],
            total_ms: 100.0,
        };
        let s = PerformanceSignal::compute(&p).unwrap();
        assert!((s.unattributed_share - 0.1).abs() < 1e-9);
    }

    #[test]
    fn zero_call_stages_do_not_become_infinitely_costly() {
        // A stage with 0 calls would divide by zero and win "costliest" with
        // an infinity. It must be skipped instead.
        let p = PhaseProfile {
            stages: vec![stage("real", 100.0, 10), stage("never_ran", 0.0, 0)],
            total_ms: 100.0,
        };
        let s = PerformanceSignal::compute(&p).unwrap();
        assert_eq!(s.costliest_call, "real");
        assert!(s.costliest_ms_per_call.is_finite());
    }

    #[test]
    fn empty_phase_is_none() {
        let p = PhaseProfile { stages: vec![], total_ms: 0.0 };
        assert!(PerformanceSignal::compute(&p).is_none());
    }
}
