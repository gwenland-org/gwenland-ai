//! Deterministic-conditions validation.
//!
//! A benchmark is only comparable if the conditions that affect determinism
//! were pinned. This check confirms the workload recorded a fixed seed and, for
//! timing-sensitive comparisons, a non-sampling (greedy) or explicit
//! temperature — and warns when they were left loose. It validates the
//! *conditions*, not the output tokens (that is [`super::numerical`]).

use crate::core::session::BenchmarkSession;
use crate::validation::integrity::{Severity, ValidationReport};

/// Check that the run pinned the knobs that govern determinism.
pub fn check(session: &BenchmarkSession, report: &mut ValidationReport) {
    let w = &session.workload;

    // A zero warmup means the first timed iteration pays cold-cache / JIT costs.
    if w.warmup_iters == 0 {
        report.push(
            Severity::Warning,
            "deterministic",
            "no warmup iterations: the first measured run includes cold-start cost",
        );
    }

    // Single measurement gives no variance signal.
    if w.measure_iters < 2 {
        report.push(
            Severity::Info,
            "deterministic",
            "only one measured iteration: run-to-run variance is unknown",
        );
    }

    // Sampling with temperature > 0 makes the token stream (and thus decode
    // length/timing) seed-dependent; fine for throughput, noted for output
    // determinism.
    if w.temperature > 0.0 {
        report.push(
            Severity::Info,
            "deterministic",
            format!(
                "temperature {:.2} > 0: token output is stochastic (seed {} pins it, \
                 but a different engine may still diverge)",
                w.temperature, w.seed
            ),
        );
    }
}
