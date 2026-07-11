//! Historical trend over a series of archived sessions.
//!
//! No database (storage rule) — the series is whatever ordered set of archive
//! files the user hands in. This computes the decode-throughput trajectory and
//! flags the largest regression between consecutive points, so a user watching
//! a metric over commits can see drift without a persistent store.

use crate::comparison::regression::{regression_verdict, Regression};
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;

/// One point in a trend: a label and its decode throughput.
#[derive(Debug, Clone)]
pub struct TrendPoint {
    /// Session label (usually the archive filename or metadata label).
    pub label: String,
    /// Mean decode throughput, tokens/second.
    pub decode_tps: f64,
}

/// The computed trend across an ordered series of sessions.
#[derive(Debug, Clone)]
pub struct TrendReport {
    /// Points in input order.
    pub points: Vec<TrendPoint>,
    /// Verdict comparing the last point to the first.
    pub overall: Regression,
}

/// Build a trend from sessions given in chronological order, using `threshold`
/// for the first-vs-last regression verdict.
pub fn trend(sessions: &[BenchmarkSession], threshold: f64) -> TrendReport {
    let points: Vec<TrendPoint> = sessions
        .iter()
        .map(|s| TrendPoint {
            label: s.metadata.label.clone(),
            decode_tps: Stats::from_samples(&s.measurements.decode_tps_samples()).mean,
        })
        .collect();

    let overall = match (points.first(), points.last()) {
        (Some(first), Some(last)) if first.decode_tps > 0.0 && points.len() > 1 => {
            let rel = (last.decode_tps - first.decode_tps) / first.decode_tps;
            regression_verdict(rel, threshold)
        }
        _ => Regression::Neutral,
    };

    TrendReport { points, overall }
}
