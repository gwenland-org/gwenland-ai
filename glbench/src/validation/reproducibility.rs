//! Reproducibility validation: is the session self-describing enough that
//! someone else could re-run it and know what they're comparing against?
//!
//! This checks that the archived environment captured the facts that make a
//! result meaningful — the engine, the model, and the hardware it ran on — and
//! flags invalid benchmark conditions (e.g. an engine that reported itself
//! unavailable, or a missing model footprint that voids the ceiling analysis).

use crate::core::session::BenchmarkSession;
use crate::validation::integrity::{Severity, ValidationReport};

/// Check the session records enough to be reproduced and interpreted.
pub fn check(session: &BenchmarkSession, report: &mut ValidationReport) {
    if !session.engine.available {
        report.push(
            Severity::Error,
            "reproducibility",
            format!(
                "engine '{}' reported unavailable on this machine; measurements may be a fallback path",
                session.engine.name
            ),
        );
    }

    if session.workload.model_path.is_empty() {
        report.push(
            Severity::Warning,
            "reproducibility",
            "no model path recorded",
        );
    }

    let hw = &session.environment.hardware;
    if hw.storage.model_file_bytes.is_none() && session.measurements.model_bytes.is_none() {
        report.push(
            Severity::Info,
            "reproducibility",
            "model byte footprint unknown; bandwidth-ceiling efficiency cannot be computed",
        );
    }

    // A GPU backend that reported no device facts is a hole in the record.
    if session.engine.backend != "cpu" && !hw.gpu.is_present() {
        report.push(
            Severity::Warning,
            "reproducibility",
            format!(
                "backend '{}' is not CPU but no GPU device facts were captured",
                session.engine.backend
            ),
        );
    }
}
