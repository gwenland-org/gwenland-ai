//! The [`AnalysisReport`] — derived insight, kept strictly separate from the
//! raw [`crate::core::metrics::MeasurementSet`] it is computed from.
//!
//! This module defines the report *shape* and the top-level [`analyze`] entry
//! point that runs each analyzer. The analyzers themselves live in sibling
//! modules ([`super::health`], [`super::bottleneck`], [`super::ceiling`],
//! [`super::efficiency`], [`super::roofline`]). Every conclusion here is a
//! recommendation, never an action — glbench observes, it does not optimize.

use crate::analysis::bottleneck::Bottleneck;
use crate::comparison::statistics::Stats;
use crate::core::schema::ToJson;
use crate::core::session::BenchmarkSession;
use crate::export::json::Json;

/// Derived analysis over one session's measurements.
#[derive(Debug, Clone)]
pub struct AnalysisReport {
    /// Decode throughput statistics, tokens/second.
    pub decode_tps: Stats,
    /// Prefill throughput statistics, tokens/second.
    pub prefill_tps: Stats,
    /// Overall performance health, 0.0..=1.0 (see [`super::health`]).
    pub health: f64,
    /// The dominant limiting factor, as classified from the facts.
    pub bottleneck: Bottleneck,
    /// Achieved fraction of the relevant hardware ceiling, 0.0..=1.0, if a
    /// ceiling could be established.
    pub ceiling_efficiency: Option<f64>,
    /// Human-readable notes — the recommendations, phrased as observations.
    pub notes: Vec<String>,
}

impl ToJson for AnalysisReport {
    fn to_json(&self) -> Json {
        Json::obj([
            ("decode_tps", stats_json(&self.decode_tps)),
            ("prefill_tps", stats_json(&self.prefill_tps)),
            ("health", Json::n(self.health)),
            ("bottleneck", Json::s(self.bottleneck.as_str())),
            (
                "ceiling_efficiency",
                match self.ceiling_efficiency {
                    Some(e) => Json::n(e),
                    None => Json::Null,
                },
            ),
            (
                "notes",
                Json::Arr(self.notes.iter().map(|n| Json::s(n.clone())).collect()),
            ),
        ])
    }
}

/// Render a [`Stats`] as a JSON object.
pub fn stats_json(s: &Stats) -> Json {
    Json::obj([
        ("count", Json::n(s.count as f64)),
        ("mean", Json::n(s.mean)),
        ("median", Json::n(s.median)),
        ("min", Json::n(s.min)),
        ("max", Json::n(s.max)),
        ("std_dev", Json::n(s.std_dev)),
        ("p95", Json::n(s.p95)),
        ("p99", Json::n(s.p99)),
    ])
}

/// Run the full analysis pipeline over a session's measurements and
/// environment. Pure: reads the session, returns a report, mutates nothing.
pub fn analyze(session: &BenchmarkSession) -> AnalysisReport {
    let m = &session.measurements;
    let decode_tps = Stats::from_samples(&m.decode_tps_samples());
    let prefill_tps = Stats::from_samples(&m.prefill_tps_samples());

    let ceiling = super::ceiling::analyze(session, &decode_tps);
    let bottleneck = super::bottleneck::classify(session, &ceiling);
    let health = super::health::score(&decode_tps, &prefill_tps, ceiling.efficiency);

    let mut notes = Vec::new();
    notes.extend(ceiling.notes.clone());
    notes.push(bottleneck.recommendation().to_string());
    if decode_tps.coefficient_of_variation() > 0.10 && decode_tps.count > 1 {
        notes.push(format!(
            "Decode throughput is noisy across runs (CV {:.0}%); increase measure_iters for a stabler figure.",
            decode_tps.coefficient_of_variation() * 100.0
        ));
    }

    AnalysisReport {
        decode_tps,
        prefill_tps,
        health,
        bottleneck,
        ceiling_efficiency: ceiling.efficiency,
        notes,
    }
}
