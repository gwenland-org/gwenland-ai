//! Engine-vs-engine comparison (e.g. glcuda vs glproc, or vs an external
//! baseline) — a run comparison viewed along the engine axis.
//!
//! It is the same [`super::runs::compare`] delta; this wrapper just asserts the
//! two sessions differ in engine and annotates the note accordingly. glbench
//! compares engines; it never *routes* between them.

use crate::comparison::runs::{compare, ComparisonReport};
use crate::core::session::BenchmarkSession;

/// Compare two sessions expected to differ by engine.
pub fn compare_engines(
    baseline: &BenchmarkSession,
    candidate: &BenchmarkSession,
    threshold: f64,
) -> ComparisonReport {
    let mut report = compare(baseline, candidate, threshold);
    let a = &baseline.engine.name;
    let b = &candidate.engine.name;
    report
        .notes
        .insert(0, format!("Engine comparison: {a} (baseline) vs {b} (candidate)."));
    if a == b {
        report
            .notes
            .push(format!("Note: both sessions used engine '{a}'."));
    }
    report
}
