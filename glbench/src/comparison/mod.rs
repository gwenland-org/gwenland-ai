//! Comparison: a first-class subsystem for run-vs-run, engine-vs-engine,
//! quantization, hardware, statistics, regression, and trend. All are views of
//! the same [`runs::compare`] delta along a particular axis. glbench compares;
//! it never routes between engines.

pub mod accuracy;
pub mod engine;
pub mod hardware;
pub mod quantization;
pub mod regression;
pub mod runs;
pub mod statistics;
/// Training-configuration comparison (Wave 4). Gated with the training tree.
#[cfg(feature = "train-bench")]
pub mod training;
pub mod trend;

use crate::core::session::BenchmarkSession;
use runs::ComparisonReport;

/// Compare two sessions along whichever axis actually separates them.
///
/// A regression verdict only means something between two runs of the same
/// setup: glproc against glcuda is not a 99% regression, it is two engines. The
/// strongest difference names the comparison; a weaker one is noted, since a
/// delta across two axes at once cannot be attributed to either.
pub fn compare_along_axis(
    baseline: &BenchmarkSession,
    candidate: &BenchmarkSession,
    threshold: f64,
) -> ComparisonReport {
    let engine = baseline.engine.name != candidate.engine.name;
    let device = hardware::device_label(baseline) != hardware::device_label(candidate);
    let quant = baseline.engine.quantization != candidate.engine.quantization;

    let mut report = if engine {
        engine::compare_engines(baseline, candidate, threshold)
    } else if device {
        hardware::compare_hardware(baseline, candidate, threshold)
    } else if quant {
        quantization::compare_quantization(baseline, candidate, threshold)
    } else {
        runs::compare(baseline, candidate, threshold)
    };

    let also: Vec<&str> = [(engine, "engine"), (device, "hardware"), (quant, "quantization")]
        .into_iter()
        .filter(|&(differs, name)| differs && report.axis != Some(name))
        .map(|(_, name)| name)
        .collect();
    if !also.is_empty() {
        report.notes.push(format!(
            "Also differs in {}, so this delta cannot be attributed to one cause.",
            also.join(" and ")
        ));
    }
    if baseline.workload.kind != candidate.workload.kind {
        report.notes.push(format!(
            "Workload kinds differ ({} vs {}), so these runs measured different things.",
            baseline.workload.kind.as_str(),
            candidate.workload.kind.as_str()
        ));
    }
    report
}
