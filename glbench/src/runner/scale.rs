//! A scaling sweep: run the identical workload at several axis values (token
//! budgets) and classify how decode throughput scales across them.
//!
//! This is CLI orchestration only — it drives [`planner::run`] once per axis
//! value and hands the resulting sessions to [`crate::analysis::scaling`],
//! which already does the classification. No new measurement or analysis
//! logic lives here.

use glcore::GlError;

use crate::analysis::scaling::{classify, ScalePoint, Scaling};
use crate::comparison::statistics::Stats;
use crate::core::session::BenchmarkSession;
use crate::core::workload::WorkloadSpec;
use crate::runner::planner::{self, Progress};

/// One completed sweep point: the axis value, the full session that produced
/// it, and the decode-tps summary pulled out for convenience.
pub struct SweepPoint {
    pub axis: f64,
    pub session: BenchmarkSession,
}

/// The full sweep result: every point plus the scaling verdict.
pub struct SweepReport {
    pub points: Vec<SweepPoint>,
    pub scaling: Scaling,
}

/// Run `spec` once per value in `axis_values` (each substituted into
/// `max_new_tokens`), sequentially — same reasoning as `ab`: concurrent runs
/// would contend for memory bandwidth and corrupt every number in the sweep.
///
/// `axis_values` need not be sorted; the result is sorted ascending by axis
/// before classification, since [`classify`] assumes that order.
pub fn run_sweep(
    spec: &WorkloadSpec,
    axis_values: &[usize],
    progress: Progress<'_>,
) -> Result<SweepReport, GlError> {
    let mut points = Vec::with_capacity(axis_values.len());
    for (n, &tokens) in axis_values.iter().enumerate() {
        progress("scale", n, axis_values.len());
        let mut point_spec = spec.clone();
        point_spec.max_new_tokens = tokens;
        let session = planner::run(&point_spec, progress)?;
        points.push(SweepPoint { axis: tokens as f64, session });
    }
    points.sort_by(|a, b| a.axis.total_cmp(&b.axis));

    let scale_points: Vec<ScalePoint> = points
        .iter()
        .map(|p| ScalePoint {
            axis: p.axis,
            throughput: Stats::from_samples(&p.session.measurements.decode_tps_samples()).mean,
        })
        .collect();
    let scaling = classify(&scale_points);

    Ok(SweepReport { points, scaling })
}
