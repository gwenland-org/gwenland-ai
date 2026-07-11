//! Quantization-vs-quantization comparison (e.g. Q8_0 vs Q4_K_M) — a run
//! comparison viewed along the quantization axis, where the interesting trade
//! is throughput against the model's byte footprint.

use crate::comparison::runs::{compare, ComparisonReport};
use crate::core::session::BenchmarkSession;

/// Compare two sessions expected to differ by quantization, annotating the
/// note with each side's quant label and model byte footprint.
pub fn compare_quantization(
    baseline: &BenchmarkSession,
    candidate: &BenchmarkSession,
    threshold: f64,
) -> ComparisonReport {
    let mut report = compare(baseline, candidate, threshold);
    let qa = baseline.engine.quantization.as_deref().unwrap_or("?");
    let qb = candidate.engine.quantization.as_deref().unwrap_or("?");
    report
        .notes
        .insert(0, format!("Quantization comparison: {qa} (baseline) vs {qb} (candidate)."));

    let ba = baseline.environment.hardware.storage.model_file_bytes;
    let bb = candidate.environment.hardware.storage.model_file_bytes;
    if let (Some(ba), Some(bb)) = (ba, bb) {
        report.notes.push(format!(
            "Model footprint: {:.2} GB -> {:.2} GB ({:+.0}%).",
            ba as f64 / 1e9,
            bb as f64 / 1e9,
            (bb as f64 - ba as f64) / ba as f64 * 100.0,
        ));
    }
    report
}
