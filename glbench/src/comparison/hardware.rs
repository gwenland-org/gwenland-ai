//! Hardware-vs-hardware comparison (e.g. same engine/model on a T4 vs a CPU) —
//! a run comparison viewed along the hardware axis.

use crate::comparison::runs::{compare, ComparisonReport};
use crate::core::session::BenchmarkSession;

/// Compare two sessions expected to differ by hardware, annotating with each
/// side's device label (GPU name if present, else CPU model).
pub fn compare_hardware(
    baseline: &BenchmarkSession,
    candidate: &BenchmarkSession,
    threshold: f64,
) -> ComparisonReport {
    let mut report = compare(baseline, candidate, threshold);
    let a = device_label(baseline);
    let b = device_label(candidate);
    report
        .notes
        .insert(0, format!("Hardware comparison: {a} (baseline) vs {b} (candidate)."));
    report
}

/// The best available device label for a session: GPU name, else CPU model,
/// else the OS/arch.
fn device_label(s: &BenchmarkSession) -> String {
    let hw = &s.environment.hardware;
    if let Some(name) = &hw.gpu.name {
        name.clone()
    } else if let Some(cpu) = &hw.cpu.model {
        cpu.clone()
    } else {
        format!("{}/{}", s.environment.runtime.os, s.environment.runtime.arch)
    }
}
