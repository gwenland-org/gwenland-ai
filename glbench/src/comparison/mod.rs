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

/// A mean SM clock this far apart is worth a sentence in a comparison.
const CLOCK_NOTE_THRESHOLD: f64 = 0.03;

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
    // Two engines on one machine report different devices, a GPU and a CPU, so
    // across engines the machine is what can differ: its CPU is on both sides.
    let device = if engine {
        baseline.environment.hardware.cpu.model != candidate.environment.hardware.cpu.model
    } else {
        hardware::device_label(baseline) != hardware::device_label(candidate)
    };
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
    let clock = |s: &BenchmarkSession| s.measurements.gpu.as_ref()?.sm_clock_mhz.map(|c| c.mean);
    if let (Some(a), Some(b)) = (clock(baseline), clock(candidate)) {
        if a > 0.0 && (b / a - 1.0).abs() > CLOCK_NOTE_THRESHOLD {
            report.notes.push(format!(
                "Mean SM clock differs ({a:.0} vs {b:.0} MHz): the card ran at a different speed."
            ));
        }
    }
    let storage = |s: &'_ BenchmarkSession| s.environment.hardware.storage.model_sha256.clone();
    if let (Some(a), Some(b)) = (storage(baseline), storage(candidate)) {
        if a != b && !quant {
            report.axis = report.axis.or(Some("model"));
            report.notes.push("The model files differ, by SHA-256.".to_string());
        }
    }
    if baseline.workload.kind != candidate.workload.kind {
        report.axis = report.axis.or(Some("workload"));
        report.notes.push(format!(
            "Workload kinds differ ({} vs {}), so these runs measured different things.",
            baseline.workload.kind.as_str(),
            candidate.workload.kind.as_str()
        ));
    }
    report
}
