//! Historical trend over a series of archived sessions.
//!
//! No database (storage rule) — the series is whatever ordered set of archive
//! files the user hands in. This computes the decode-throughput trajectory and
//! flags the largest regression between consecutive points, so a user watching
//! a metric over commits can see drift without a persistent store.

use crate::comparison::regression::{regression_verdict, Regression};
use crate::comparison::statistics::Stats;
use crate::core::result::ENMeasurementMode;
use crate::core::session::BenchmarkSession;

/// One point in a trend: a label and its decode throughput.
#[derive(Debug, Clone)]
pub struct TrendPoint {
    /// Session label (usually the archive filename or metadata label).
    pub label: String,
    /// Mean decode throughput, tokens/second.
    pub decode_tps: f64,
    /// Mean prefill throughput, tokens/second.
    pub prefill_tps: f64,
}

/// The computed trend across an ordered series of sessions.
#[derive(Debug, Clone)]
pub struct TrendReport {
    /// Points in input order.
    pub points: Vec<TrendPoint>,
    /// Verdict comparing the last point to the first.
    pub overall: Regression,
    /// The largest decode drop between consecutive points: the index of the
    /// later point and the relative change. `None` when nothing dropped.
    pub largest_drop: Option<(usize, f64)>,
    /// Why the series may not be one setup over time.
    pub notes: Vec<String>,
}

/// Build a trend from sessions given in chronological order, using `threshold`
/// for the first-vs-last regression verdict.
pub fn trend(sessions: &[BenchmarkSession], threshold: f64) -> TrendReport {
    let points: Vec<TrendPoint> = sessions
        .iter()
        .map(|s| TrendPoint {
            label: s.metadata.label.clone(),
            decode_tps: Stats::from_samples(&s.measurements.decode_tps_samples()).mean,
            prefill_tps: Stats::from_samples(&s.measurements.prefill_tps_samples()).mean,
        })
        .collect();

    let largest_drop = points
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| pair[0].decode_tps > 0.0)
        .map(|(i, pair)| (i + 1, (pair[1].decode_tps - pair[0].decode_tps) / pair[0].decode_tps))
        .filter(|&(_, change)| change < 0.0)
        .min_by(|a, b| a.1.total_cmp(&b.1));

    let mut notes = Vec::new();
    let distinct = |key: fn(&BenchmarkSession) -> String| {
        let mut values: Vec<String> = sessions.iter().map(key).collect();
        values.sort();
        values.dedup();
        values.len()
    };
    if distinct(|s| s.engine.name.clone()) > 1 {
        notes.push("Series mixes engines; a trend assumes one setup over time.".to_string());
    }
    if distinct(crate::comparison::hardware::device_label) > 1 {
        notes.push("Series mixes devices; a trend assumes one setup over time.".to_string());
    }
    if distinct(|s| s.workload.kind.as_str().to_string()) > 1 {
        notes.push("Series mixes workload kinds, which measure different things.".to_string());
    }
    let non_production = sessions
        .iter()
        .filter(|s| s.metadata.measurement_mode != ENMeasurementMode::Production)
        .count();
    if non_production > 0 {
        notes.push(format!(
            "{non_production} of {} sessions are not production measurements.",
            sessions.len()
        ));
    }

    let overall = match (points.first(), points.last()) {
        (Some(first), Some(last)) if first.decode_tps > 0.0 && points.len() > 1 => {
            let rel = (last.decode_tps - first.decode_tps) / first.decode_tps;
            regression_verdict(rel, threshold)
        }
        _ => Regression::Neutral,
    };

    TrendReport { points, overall, largest_drop, notes }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;

    /// A session whose decode runs at `tps` tokens/second.
    fn session(tps: f64) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 100,
            prefill_ms: 100.0,
            decode_ms: 100.0 / tps * 1000.0,
            total_ms: 0.0,
            ttft_ms: None,
        });
        let mut metadata = SessionMetadata::new(format!("run-{tps}"));
        metadata.measurement_mode = ENMeasurementMode::Production;
        BenchmarkSession::new(
            metadata,
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
            m,
        )
    }

    #[test]
    fn the_largest_drop_is_found_even_when_the_series_recovers() {
        let sessions = [session(100.0), session(110.0), session(55.0), session(100.0)];
        let report = trend(&sessions, 0.05);
        assert_eq!(report.overall, Regression::Neutral);
        let (at, change) = report.largest_drop.expect("the series dropped");
        assert_eq!(at, 2);
        assert!((change + 0.5).abs() < 1e-9, "{change}");
        assert!(report.notes.is_empty(), "{:?}", report.notes);
    }

    #[test]
    fn a_rising_series_has_no_drop() {
        let report = trend(&[session(100.0), session(120.0)], 0.05);
        assert_eq!(report.overall, Regression::Improved);
        assert_eq!(report.largest_drop, None);
    }

    #[test]
    fn a_series_that_is_not_one_setup_says_so() {
        let mut other = session(100.0);
        other.engine.name = "glcuda".into();
        other.metadata.measurement_mode = ENMeasurementMode::Instrumented;
        let report = trend(&[session(100.0), other], 0.05);
        assert!(report.notes.iter().any(|n| n.contains("mixes engines")), "{:?}", report.notes);
        assert!(report.notes.iter().any(|n| n.contains("1 of 2")), "{:?}", report.notes);
    }
}
