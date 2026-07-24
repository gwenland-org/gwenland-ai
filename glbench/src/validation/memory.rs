//! KV-cache memory-risk validation: would this model's own weights plus its
//! *configured* KV cache actually fit in the RAM this machine had free before
//! the run started?
//!
//! [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions]
//! 1. What problem: the KV cache is sized from `max_seq_len` at load time, not
//!    from how much context a run actually uses — a context length picked for
//!    a bigger machine silently demands more RAM than a smaller one has. This
//!    is exactly the "KV cache is the memory trap" finding from the ARTX05
//!    runtime work (a naive full-context KV cache on an 8 GB machine exceeded
//!    it), just generalized into a check instead of a one-off finding.
//! 2. Who benefits: anyone deploying a model+context combination onto a
//!    smaller machine than it was developed on.
//! 3. Production/research use: capacity planning ("will this configuration
//!    fit") is standard practice before any inference deployment.
//! 4. How calculated: `telemetry.memory.model_bytes + kv_cache_bytes`,
//!    compared against `environment.hardware.memory.available_bytes` — RAM
//!    free on the machine *before* this process claimed any of it (probed at
//!    the start of the run, see `runner::planner`'s "before load" comment),
//!    which is the right baseline for "would this fit from a cold start".
//! 5. Reproducible: yes — both figures are already-measured facts (engine
//!    telemetry + OS memory probe), this only compares two numbers already in
//!    the session.
//! 6. Actionable: yes — a red flag here says exactly what to shrink
//!    (`max_seq_len`) rather than leaving an OOM as a surprise mid-run.
//! 7. Lightweight: a comparison of two numbers already collected elsewhere;
//!    no new measurement, no new engine hook.
//! 8. Philosophy: this lives in `validation`, not `measurement`, on purpose —
//!    "would this fit" is a judgment call over raw facts, exactly what
//!    validation (not measurement) is for.

use crate::core::session::BenchmarkSession;
use crate::validation::integrity::{Severity, ValidationReport};

/// Above this fraction of the machine's free RAM, flag a warning even though
/// it would technically still fit — a tight fit risks being pushed into swap
/// by anything else running on the machine.
const TIGHT_FIT_FRACTION: f64 = 0.8;

/// Check the model + KV cache footprint against available RAM.
pub fn check(session: &BenchmarkSession, report: &mut ValidationReport) {
    let Some(mem) = session.telemetry.as_ref().and_then(|t| t.memory.as_ref()) else {
        return; // Engine reported no memory telemetry — not measured, not flagged.
    };
    let Some(available) = session.environment.hardware.memory.available_bytes else {
        return; // No OS memory probe on this platform — same "not measured" rule.
    };

    let needed = mem.model_bytes + mem.kv_cache_bytes;
    if needed > available {
        report.push(
            Severity::Error,
            "memory",
            format!(
                "model + KV cache needs {:.2} GiB but only {:.2} GiB was free before load; \
                 this configuration would not fit — shrink max_seq_len or use a smaller model",
                gib(needed),
                gib(available),
            ),
        );
    } else if needed as f64 > available as f64 * TIGHT_FIT_FRACTION {
        report.push(
            Severity::Warning,
            "memory",
            format!(
                "model + KV cache uses {:.2} GiB of {:.2} GiB that was free ({:.0}%); a tight \
                 fit risks swapping if anything else runs on this machine",
                gib(needed),
                gib(available),
                needed as f64 / available as f64 * 100.0,
            ),
        );
    }
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;
    use glcore::telemetry::{EngineTelemetry, MemoryTelemetry};

    fn sample_with(model_bytes: u64, kv_cache_bytes: u64, available_bytes: Option<u64>) -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 10,
            generated_tokens: 10,
            prefill_ms: 5.0,
            decode_ms: 50.0,
            total_ms: 55.0,
        });
        let mut session = BenchmarkSession::new(
            SessionMetadata::new("test"),
            EnvironmentSnapshot::probe(""),
            EngineMetadata {
                name: "glproc".into(),
                backend: "cpu".into(),
                available: true,
                model_arch: None,
                quantization: None,
                thinking_capable: None,
            },
            WorkloadSpec::default(),
            m,
        );
        session.environment.hardware.memory.available_bytes = available_bytes;
        session.telemetry = Some(EngineTelemetry {
            prefill: None,
            decode: None,
            backend: None,
            memory: Some(MemoryTelemetry { model_bytes, kv_cache_bytes, scratch_bytes: 0 }),
            moe: None,
        });
        session
    }

    const GIB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn comfortable_margin_produces_no_finding() {
        let session = sample_with(1 * GIB, 1 * GIB, Some(8 * GIB));
        let mut report = ValidationReport::default();
        check(&session, &mut report);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn exceeding_available_ram_is_an_error() {
        let session = sample_with(4 * GIB, 5 * GIB, Some(8 * GIB));
        let mut report = ValidationReport::default();
        check(&session, &mut report);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].severity, Severity::Error);
        assert_eq!(report.findings[0].check, "memory");
        assert!(!report.passed());
    }

    #[test]
    fn tight_fit_under_the_ram_ceiling_is_a_warning() {
        // 7.5 GiB needed of 8 GiB free = 93.75%, over the 80% tight-fit line
        // but still under the hard ceiling.
        let session = sample_with(3 * GIB + 512 * 1024 * 1024, 4 * GIB, Some(8 * GIB));
        let mut report = ValidationReport::default();
        check(&session, &mut report);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].severity, Severity::Warning);
        // A warning must not fail the session.
        assert!(report.passed());
    }

    #[test]
    fn missing_telemetry_produces_no_finding() {
        let mut session = sample_with(1 * GIB, 1 * GIB, Some(8 * GIB));
        session.telemetry = None;
        let mut report = ValidationReport::default();
        check(&session, &mut report);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn missing_available_ram_produces_no_finding() {
        let session = sample_with(1 * GIB, 1 * GIB, None);
        let mut report = ValidationReport::default();
        check(&session, &mut report);
        assert!(report.findings.is_empty());
    }
}
