//! Root-cause hypotheses — cross-signal pattern matching, phrased honestly.
//!
//! Individual signals say *what* happened: drift says the loop slowed, stall
//! says it spiked, repetition says the text looped, the roofline says which
//! bucket sat where. This module reads them *together* and emits the
//! explanations the combined pattern is consistent with.
//!
//! Two rules keep this module on the right side of DESIGN.md:
//!
//! - **Hypotheses, not verdicts.** Every line says "consistent with", because
//!   timing patterns genuinely cannot prove causes (the stall module's warning).
//!   glbench has no thermal sensor or PMU access, so a throttle can be
//!   *suggested* by shape but never confirmed — and the text says so.
//! - **Observations, not actions.** "The pattern points at KV-cache growth" is
//!   allowed; "shrink your context" is not.

use crate::behavior::cot::EntropyFlag;
use crate::core::session::BenchmarkSession;

/// A window perplexity this many times the run baseline reads as OOD input.
const OOD_SPIKE_RATIO: f64 = 3.0;

/// Generate root-cause hypotheses from every signal the session captured.
/// Signals that were not measured contribute nothing — no hypothesis is ever
/// built on an absent number.
///
/// `roofline` is passed in rather than read off the session because this runs
/// *inside* [`super::summary::analyze`], before the report is attached.
pub fn hypotheses(
    session: &BenchmarkSession,
    roofline: Option<&super::roofline::RooflineReport>,
) -> Vec<String> {
    let mut out = Vec::new();
    let Some(b) = &session.behavior else { return out };

    // -- Intra-session latency drift: shape distinguishes growth from incident.
    if let Some(a) = &b.anomaly {
        if a.has_drift() && a.drift_frac > 0.0 {
            if a.drift_is_monotonic() {
                out.push(format!(
                    "Inter-token latency rose {:+.0}% from the first to the last quarter, \
                     monotonically — consistent with KV-cache growth (attention cost scales \
                     with sequence length), not with an external incident.",
                    a.drift_frac * 100.0
                ));
            } else {
                out.push(format!(
                    "Inter-token latency ended {:+.0}% slower but not monotonically — \
                     consistent with external interference (scheduler contention, or a \
                     clock change from thermal/power state; unconfirmable without sensors) \
                     rather than sequence-length growth.",
                    a.drift_frac * 100.0
                ));
            }
        }

        if let (Some(ratio), Some(at)) = (a.spike_ratio, a.spike_token) {
            if ratio > OOD_SPIKE_RATIO {
                out.push(format!(
                    "Perplexity in the window starting at token {at} ran {ratio:.1}x the \
                     run baseline — the model hit text it found far out of distribution there."
                ));
            }
        }
    }

    // -- Entropy collapse: the CoT-aware read, refined by repetition.
    if let Some(c) = &b.cot {
        match c.flag {
            EntropyFlag::LowEntropyAnomaly => {
                let looping =
                    b.repetition.as_ref().map(|r| r.looks_degenerate()).unwrap_or(false);
                if looping {
                    out.push(
                        "Entropy collapsed AND n-gram diversity collapsed on a model with \
                         no thinking mode — the decode is looping."
                            .to_string(),
                    );
                } else {
                    out.push(
                        "Entropy collapsed on a model with no thinking mode, but the text \
                         is not n-gram-repeating — consistent with a highly constrained \
                         prompt; worth inspecting the output."
                            .to_string(),
                    );
                }
            }
            EntropyFlag::LowEntropyCotExpected => {
                out.push(
                    "Entropy is very low, which is expected here: the model is \
                     thinking-capable and low-entropy reasoning stretches are by design."
                        .to_string(),
                );
            }
            EntropyFlag::Normal => {}
        }
    }

    // -- Stalls without drift: isolated blocking, not a trend.
    if let (Some(st), Some(a)) = (&b.stall, &b.anomaly) {
        if st.has_stalls() && !a.has_drift() {
            out.push(format!(
                "{} isolated latency spike(s) (worst {:.0} ms) on an otherwise flat run — \
                 consistent with scheduler preemption or page faults, not with a \
                 sustained slowdown.",
                st.stall_count, st.max_ms
            ));
        }
    }

    // -- Roofline: name the decode bucket that is NOT bandwidth-bound.
    if let Some(analysis_roofline) = roofline {
        for bucket in &analysis_roofline.decode {
            if bucket.verdict == super::roofline::BucketVerdict::NotBandwidthBound
                && bucket.share.map(|s| s > 0.10).unwrap_or(false)
            {
                out.push(format!(
                    "Decode bucket '{}' holds {:.0}% of the phase but reaches only {:.0}% \
                     of the bandwidth ceiling — it is stalled on something other than \
                     memory traffic, so reading fewer bytes there would not help.",
                    bucket.bucket.as_str(),
                    bucket.share.unwrap_or(0.0) * 100.0,
                    bucket.ceiling_frac.unwrap_or(0.0) * 100.0,
                ));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::behavior::anomaly::AnomalySignal;
    use crate::behavior::cot::CotAssessment;
    use crate::behavior::BehaviorReport;
    use crate::core::metrics::MeasurementSet;
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn session_with(behavior: BehaviorReport) -> BenchmarkSession {
        let mut s = BenchmarkSession::new(
            SessionMetadata::new("test"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata::default(),
            WorkloadSpec::default(),
            MeasurementSet::default(),
        );
        s.behavior = Some(behavior);
        s
    }

    fn anomaly(quarters: [f64; 4], spike: Option<(f64, usize)>) -> AnomalySignal {
        AnomalySignal {
            quarter_gap_ms: quarters,
            drift_frac: (quarters[3] - quarters[0]) / quarters[0],
            spike_ratio: spike.map(|(r, _)| r),
            spike_token: spike.map(|(_, t)| t),
            samples: 100,
        }
    }

    #[test]
    fn monotonic_drift_names_kv_cache_growth() {
        let b = BehaviorReport {
            anomaly: Some(anomaly([50.0, 53.0, 56.0, 60.0], None)),
            ..Default::default()
        };
        let h = hypotheses(&session_with(b), None);
        assert!(h.iter().any(|s| s.contains("KV-cache growth")), "{h:?}");
    }

    #[test]
    fn non_monotonic_drift_does_not_claim_kv_cache() {
        let b = BehaviorReport {
            anomaly: Some(anomaly([50.0, 70.0, 52.0, 58.0], None)),
            ..Default::default()
        };
        let h = hypotheses(&session_with(b), None);
        assert!(!h.iter().any(|s| s.contains("KV-cache growth")), "{h:?}");
        assert!(h.iter().any(|s| s.contains("external interference")), "{h:?}");
    }

    #[test]
    fn ood_spike_reports_the_token_position() {
        let b = BehaviorReport {
            anomaly: Some(anomaly([50.0; 4], Some((4.2, 137)))),
            ..Default::default()
        };
        let h = hypotheses(&session_with(b), None);
        assert!(h.iter().any(|s| s.contains("token 137") && s.contains("4.2x")), "{h:?}");
    }

    #[test]
    fn a_session_with_nothing_measured_hypothesizes_nothing() {
        assert!(hypotheses(&session_with(BehaviorReport::default()), None).is_empty());
        // And no behavior at all: also nothing (built on absence of data).
        let s = session_with(BehaviorReport::default());
        let mut bare = s;
        bare.behavior = None;
        assert!(hypotheses(&bare, None).is_empty());
    }

    #[test]
    fn cot_expected_low_entropy_is_explained_not_alarmed() {
        use crate::behavior::entropy::EntropySignal;
        let e = EntropySignal {
            mean: 0.16,
            std_dev: 0.1,
            min: 0.0,
            max: 0.5,
            p50: 0.1,
            p95: 0.4,
            mean_top_prob: 0.95,
            tokens: 200,
        };
        let b = BehaviorReport {
            entropy: Some(e),
            cot: Some(CotAssessment::assess(&e, true)),
            ..Default::default()
        };
        let h = hypotheses(&session_with(b), None);
        assert!(h.iter().any(|s| s.contains("expected")), "{h:?}");
        assert!(!h.iter().any(|s| s.contains("looping")), "{h:?}");
    }
}
