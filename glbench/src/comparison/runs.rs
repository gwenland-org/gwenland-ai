//! Run-to-run comparison: baseline vs candidate.
//!
//! The core comparison primitive — take two sessions and report the delta in
//! the headline metrics, plus a regression verdict. Everything else in
//! [`crate::comparison`] (engine, quantization, hardware comparisons) is this
//! same delta viewed along a particular axis.

use crate::comparison::regression::{regression_verdict, Regression};
use crate::comparison::statistics::Stats;
use crate::core::result::ENMeasurementMode;
use crate::core::schema::ToJson;
use crate::core::session::BenchmarkSession;
use crate::export::json::Json;

/// A single metric's before/after and relative change.
#[derive(Debug, Clone, Copy)]
pub struct Delta {
    /// Baseline value.
    pub baseline: f64,
    /// Candidate value.
    pub candidate: f64,
}

impl Delta {
    /// Relative change (candidate - baseline) / baseline; 0 if baseline is 0.
    pub fn relative(&self) -> f64 {
        if self.baseline == 0.0 {
            0.0
        } else {
            (self.candidate - self.baseline) / self.baseline
        }
    }

    /// Candidate / baseline ratio; 0 if baseline is 0.
    pub fn ratio(&self) -> f64 {
        if self.baseline == 0.0 {
            0.0
        } else {
            self.candidate / self.baseline
        }
    }

    fn to_json(self) -> Json {
        Json::obj([
            ("baseline", Json::n(self.baseline)),
            ("candidate", Json::n(self.candidate)),
            ("relative", Json::n(self.relative())),
            ("ratio", Json::n(self.ratio())),
        ])
    }
}

/// The result of comparing two sessions.
#[derive(Debug, Clone)]
pub struct ComparisonReport {
    /// Human labels for the two sides.
    pub baseline_label: String,
    pub candidate_label: String,
    /// Decode throughput delta (tokens/second).
    pub decode_tps: Delta,
    /// Prefill throughput delta (tokens/second).
    pub prefill_tps: Delta,
    /// Regression verdict on the headline (decode) metric.
    pub regression: Regression,
    /// Observations.
    pub notes: Vec<String>,
}

impl ToJson for ComparisonReport {
    fn to_json(&self) -> Json {
        Json::obj([
            ("baseline_label", Json::s(self.baseline_label.clone())),
            ("candidate_label", Json::s(self.candidate_label.clone())),
            ("decode_tps", self.decode_tps.to_json()),
            ("prefill_tps", self.prefill_tps.to_json()),
            ("regression", Json::s(self.regression.as_str())),
            (
                "notes",
                Json::Arr(self.notes.iter().map(|n| Json::s(n.clone())).collect()),
            ),
        ])
    }
}

/// Compare two full sessions. The candidate is judged against the baseline;
/// a `threshold` (e.g. 0.05 for 5%) sets how large a decode drop counts as a
/// regression.
pub fn compare(
    baseline: &BenchmarkSession,
    candidate: &BenchmarkSession,
    threshold: f64,
) -> ComparisonReport {
    let base_dec = Stats::from_samples(&baseline.measurements.decode_tps_samples()).mean;
    let cand_dec = Stats::from_samples(&candidate.measurements.decode_tps_samples()).mean;
    let base_pre = Stats::from_samples(&baseline.measurements.prefill_tps_samples()).mean;
    let cand_pre = Stats::from_samples(&candidate.measurements.prefill_tps_samples()).mean;

    let decode_tps = Delta { baseline: base_dec, candidate: cand_dec };
    let prefill_tps = Delta { baseline: base_pre, candidate: cand_pre };
    let regression = regression_verdict(decode_tps.relative(), threshold);

    let mut notes = Vec::new();
    if baseline.metadata.measurement_mode != ENMeasurementMode::Production
        || candidate.metadata.measurement_mode != ENMeasurementMode::Production
    {
        notes.push(format!(
            "NON-AUTHORITATIVE throughput comparison: measurement modes are baseline={} and candidate={}.",
            baseline.metadata.measurement_mode.as_str(),
            candidate.metadata.measurement_mode.as_str(),
        ));
    }
    let base_dispatch = baseline
        .metadata
        .dispatch
        .as_ref()
        .map(|dispatch| dispatch.config_fingerprint.as_str());
    let candidate_dispatch = candidate
        .metadata
        .dispatch
        .as_ref()
        .map(|dispatch| dispatch.config_fingerprint.as_str());
    if base_dispatch != candidate_dispatch {
        notes.push(format!(
            "Dispatch configuration differs: baseline={} candidate={}.",
            base_dispatch.unwrap_or("unknown"),
            candidate_dispatch.unwrap_or("unknown"),
        ));
    }
    notes.push(format!(
        "Decode {:+.1}% ({:.1} -> {:.1} tok/s).",
        decode_tps.relative() * 100.0,
        decode_tps.baseline,
        decode_tps.candidate,
    ));
    notes.push(format!(
        "Prefill {:+.1}% ({:.1} -> {:.1} tok/s).",
        prefill_tps.relative() * 100.0,
        prefill_tps.baseline,
        prefill_tps.candidate,
    ));

    ComparisonReport {
        baseline_label: baseline.metadata.label.clone(),
        candidate_label: candidate.metadata.label.clone(),
        decode_tps,
        prefill_tps,
        regression,
        notes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    fn session(label: &str) -> BenchmarkSession {
        let mut measurements = MeasurementSet::default();
        measurements.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 100,
            prefill_ms: 100.0,
            decode_ms: 1_000.0,
            total_ms: 1_100.0,
        });
        BenchmarkSession::new(
            SessionMetadata::new(label),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            measurements,
        )
    }

    #[test]
    fn instrumented_comparison_is_explicitly_non_authoritative() {
        let baseline = session("baseline");
        let mut candidate = session("candidate");
        candidate.metadata.measurement_mode = ENMeasurementMode::Instrumented;

        let report = compare(&baseline, &candidate, 0.05);
        assert!(report.notes[0].contains("NON-AUTHORITATIVE"));
        assert!(report.notes[0].contains("candidate=instrumented"));
    }
}
