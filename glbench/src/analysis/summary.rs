//! The [`AnalysisReport`] — derived insight, kept strictly separate from the
//! raw [`crate::core::metrics::MeasurementSet`] it is computed from.
//!
//! This module defines the report *shape* and the top-level [`analyze`] entry
//! point that runs each analyzer. The analyzers themselves live in sibling
//! modules ([`super::health`], [`super::bottleneck`], [`super::ceiling`],
//! [`super::efficiency`], [`super::roofline`]). Every conclusion here is a
//! recommendation, never an action — glbench observes, it does not optimize.

use crate::analysis::bottleneck::Bottleneck;
use crate::analysis::ceiling::CeilingBasis;
use crate::analysis::roofline::RooflineReport;
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
    /// Where the ceiling number came from — measured on this machine, a
    /// vendor's published spec, or undetermined. Decides how literally
    /// `ceiling_efficiency` should be read (see [`CeilingBasis`]'s docs).
    pub ceiling_basis: CeilingBasis,
    /// Per-bucket roofline (Attention / FFN / lm_head vs the bandwidth
    /// ceiling), when the engine reported stage telemetry.
    pub roofline: Option<RooflineReport>,
    /// Root-cause hypotheses from cross-signal pattern matching. Explicitly
    /// hypotheses — each says what the pattern is *consistent with*, never a
    /// confirmed cause (see [`super::hypothesis`]).
    pub hypotheses: Vec<String>,
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
            ("ceiling_basis", Json::s(self.ceiling_basis.as_str())),
            (
                "roofline",
                self.roofline.as_ref().map(roofline_json).unwrap_or(Json::Null),
            ),
            (
                "hypotheses",
                Json::Arr(self.hypotheses.iter().map(|n| Json::s(n.clone())).collect()),
            ),
            (
                "notes",
                Json::Arr(self.notes.iter().map(|n| Json::s(n.clone())).collect()),
            ),
        ])
    }
}

/// JSON projection of the per-bucket roofline. Raw counters (ms, GB/s) are
/// written alongside the verdicts so a reader who disagrees with the
/// thresholds can re-derive the classification.
fn roofline_json(r: &RooflineReport) -> Json {
    let buckets = |bs: &[crate::analysis::roofline::BucketRoofline]| {
        Json::Arr(
            bs.iter()
                .map(|b| {
                    Json::obj([
                        ("bucket", Json::s(b.bucket.as_str())),
                        ("total_ms", Json::n(b.total_ms)),
                        ("share", b.share.map(Json::Num).unwrap_or(Json::Null)),
                        ("gb_per_s", b.gb_per_s.map(Json::Num).unwrap_or(Json::Null)),
                        (
                            "ceiling_frac",
                            b.ceiling_frac.map(Json::Num).unwrap_or(Json::Null),
                        ),
                        (
                            "intensity_flop_per_byte",
                            b.intensity_flop_per_byte.map(Json::Num).unwrap_or(Json::Null),
                        ),
                        ("verdict", Json::s(b.verdict.as_str())),
                    ])
                })
                .collect(),
        )
    };
    Json::obj([
        (
            "ceiling_gbs",
            r.ceiling_gbs.map(Json::Num).unwrap_or(Json::Null),
        ),
        ("decode", buckets(&r.decode)),
        ("prefill", buckets(&r.prefill)),
    ])
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
        // null below 3 samples — a t-interval on 1-2 points is not a claim
        // worth making, never fabricated as a number that looks precise.
        ("ci95", match s.ci95 {
            Some(ci) => Json::n(ci),
            None => Json::Null,
        }),
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

    // Per-bucket roofline over the engine's stage telemetry. The ceiling is
    // the measured CPU read bandwidth when present (the honest number), else
    // the GPU's published spec; with neither, verdicts stay Unknown.
    let hw = &session.environment.hardware;
    let bandwidth_ceiling = hw.cpu.read_bandwidth_gbs.or(hw.gpu.peak_bandwidth_gbs);
    let roofline = session
        .telemetry
        .as_ref()
        .and_then(|t| RooflineReport::compute(t, bandwidth_ceiling));

    let hypotheses = super::hypothesis::hypotheses(session, roofline.as_ref());

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
        ceiling_basis: ceiling.basis,
        roofline,
        hypotheses,
        notes,
    }
}
