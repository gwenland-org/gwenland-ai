//! The [`ValidationReport`] and benchmark-integrity checks.
//!
//! Validation answers "should I trust these numbers?" — it does not judge
//! performance, only the *conditions* under which it was measured. A session
//! with one warmup-less iteration, a zero prefill time, or huge run-to-run
//! variance is flagged so a reader does not draw conclusions from noise.

use crate::comparison::statistics::Stats;
use crate::core::schema::ToJson;
use crate::core::session::BenchmarkSession;
use crate::export::json::Json;

/// Severity of a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// Informational — worth noting, does not invalidate the run.
    Info,
    /// The result is usable but should be read with caution.
    Warning,
    /// The result is not trustworthy as a benchmark.
    Error,
}

impl Severity {
    /// Stable identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Severity::Info => "info",
            Severity::Warning => "warning",
            Severity::Error => "error",
        }
    }
}

/// A single validation finding.
#[derive(Debug, Clone)]
pub struct Finding {
    /// How serious the finding is.
    pub severity: Severity,
    /// The check that produced it.
    pub check: String,
    /// What was found.
    pub message: String,
}

/// The full validation result for a session.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    /// All findings, most-severe first is not guaranteed — read `passed`.
    pub findings: Vec<Finding>,
}

impl ValidationReport {
    /// True if no `Error`-severity finding is present (warnings are allowed).
    pub fn passed(&self) -> bool {
        !self.findings.iter().any(|f| f.severity == Severity::Error)
    }

    /// Add a finding.
    pub fn push(&mut self, severity: Severity, check: &str, message: impl Into<String>) {
        self.findings.push(Finding {
            severity,
            check: check.to_string(),
            message: message.into(),
        });
    }
}

impl ToJson for ValidationReport {
    fn to_json(&self) -> Json {
        Json::obj([
            ("passed", Json::Bool(self.passed())),
            (
                "findings",
                Json::Arr(
                    self.findings
                        .iter()
                        .map(|f| {
                            Json::obj([
                                ("severity", Json::s(f.severity.as_str())),
                                ("check", Json::s(f.check.clone())),
                                ("message", Json::s(f.message.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

/// Run every validation check over a session and collect the findings.
pub fn validate(session: &BenchmarkSession) -> ValidationReport {
    let mut report = ValidationReport::default();
    check_integrity(session, &mut report);
    check_thermal(session, &mut report);
    check_workload_fit(session, &mut report);
    check_gpu_limits(session, &mut report);
    super::deterministic::check(session, &mut report);
    super::reproducibility::check(session, &mut report);
    super::memory::check(session, &mut report);
    report
}

/// Flag a GPU that was held back during the run. The numbers are real, but they
/// describe a limited card, and a comparison against an unlimited run measures
/// the limit.
fn check_gpu_limits(session: &BenchmarkSession, report: &mut ValidationReport) {
    let Some(gpu) = &session.measurements.gpu else {
        return;
    };
    for (reason, samples) in gpu.limiting_reasons() {
        report.push(
            Severity::Warning,
            "gpu",
            format!("GPU clocks limited by {reason} in {samples} of {} samples", gpu.samples),
        );
    }
}

/// Does the workload measure what its kind claims? A short prompt reports launch
/// overhead rather than prefill throughput, a short generation reports edge
/// effects rather than decode throughput, and a stress run that sagged needs
/// saying even though every number in it is real.
fn check_workload_fit(session: &BenchmarkSession, report: &mut ValidationReport) {
    use crate::core::workload::WorkloadKind;
    use crate::runner::{decode, prefill, stress};

    let kind = session.workload.kind;
    let Some(first) = session.measurements.iterations.first() else {
        return;
    };
    if kind != WorkloadKind::Decode && !prefill::is_representative(first.prompt_tokens as usize) {
        report.push(
            Severity::Warning,
            "workload",
            format!(
                "prompt is {} tokens, under the {} a prefill figure needs; this mostly measures launch overhead",
                first.prompt_tokens,
                prefill::MIN_REPRESENTATIVE_PROMPT
            ),
        );
    }
    if kind != WorkloadKind::Prefill && !decode::is_stable_budget(first.generated_tokens as usize) {
        report.push(
            Severity::Warning,
            "workload",
            format!(
                "generated {} tokens, under the {} a stable decode figure needs",
                first.generated_tokens,
                decode::MIN_DECODE_TOKENS
            ),
        );
    }
    if kind == WorkloadKind::Stress {
        if let Some(s) = stress::stability(&session.measurements) {
            if s.drift() < -stress::SAG_THRESHOLD {
                report.push(
                    Severity::Warning,
                    "stress",
                    format!(
                        "decode sagged {:.1}% between the first and last quarter of the run",
                        -s.drift() * 100.0
                    ),
                );
            }
        }
    }
}

/// Flag a CPU clock drop between the start and end of the session — glbench
/// has no thermal sensor, so a falling clock under sustained load is the one
/// indirect, honestly-measured throttle signal available (see
/// `environment::thermal`'s module docs). `Warning`, not `Error`: the numbers
/// are still real measurements, just possibly measuring a machine that
/// slowed down partway through.
fn check_thermal(session: &BenchmarkSession, report: &mut ValidationReport) {
    let t = &session.environment.hardware.thermal;
    if !t.throttled() {
        return;
    }
    let (start, end) = match (t.start_mhz, t.end_mhz) {
        (Some(s), Some(e)) => (s, e),
        _ => return, // throttled() already guards this, but avoid a panic on a future change
    };
    report.push(
        Severity::Warning,
        "thermal",
        format!(
            "thermal throttling detected: CPU clock dropped from {start:.0} MHz to {end:.0} MHz \
             ({:.0}% of start) during the run",
            end / start * 100.0
        ),
    );
}

/// Structural integrity: the session must actually contain measurements, and
/// its counters must be internally consistent.
fn check_integrity(session: &BenchmarkSession, report: &mut ValidationReport) {
    let m = &session.measurements;
    if m.is_empty() {
        report.push(Severity::Error, "integrity", "no measured iterations recorded");
        return;
    }
    if session.workload.measure_iters != m.len() {
        report.push(
            Severity::Warning,
            "integrity",
            format!(
                "requested {} measure iterations but recorded {}",
                session.workload.measure_iters,
                m.len()
            ),
        );
    }
    for (i, it) in m.iterations.iter().enumerate() {
        if it.generated_tokens > 0 && it.decode_ms <= 0.0 {
            report.push(
                Severity::Error,
                "integrity",
                format!("iteration {i} generated tokens with zero decode time"),
            );
        }
        if it.prompt_tokens > 0 && it.prefill_ms <= 0.0 {
            report.push(
                Severity::Warning,
                "integrity",
                format!("iteration {i} has prompt tokens but zero prefill time"),
            );
        }
    }

    // Noise check: high run-to-run variance undermines any single number.
    let dec = Stats::from_samples(&m.decode_tps_samples());
    if dec.count > 1 && dec.coefficient_of_variation() > 0.20 {
        report.push(
            Severity::Warning,
            "integrity",
            format!(
                "decode throughput varies {:.0}% run-to-run; results are noisy",
                dec.coefficient_of_variation() * 100.0
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::metrics::{IterationMetrics, MeasurementSet};
    use crate::core::result::SessionMetadata;
    use crate::core::workload::WorkloadSpec;
    use crate::engine::metadata::EngineMetadata;
    use crate::environment::hardware::EnvironmentSnapshot;
    use crate::environment::thermal::ThermalSnapshot;

    fn sample() -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms: 4000.0,
            total_ms: 4100.0,
            ttft_ms: None,
        });
        BenchmarkSession::new(
            SessionMetadata::new("test-run"),
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
        )
    }

    fn fit_findings(sess: &BenchmarkSession) -> Vec<String> {
        let mut report = ValidationReport::default();
        check_workload_fit(sess, &mut report);
        report.findings.into_iter().map(|f| f.message).collect()
    }

    #[test]
    fn each_kind_is_only_held_to_the_phase_it_measures() {
        use crate::core::workload::WorkloadKind;

        let mut sess = sample();
        sess.measurements.iterations[0].prompt_tokens = 20;
        sess.measurements.iterations[0].generated_tokens = 1;

        sess.workload.kind = WorkloadKind::EndToEnd;
        assert_eq!(fit_findings(&sess).len(), 2);

        sess.workload.kind = WorkloadKind::Prefill;
        let findings = fit_findings(&sess);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("prompt is 20 tokens"), "{findings:?}");

        sess.workload.kind = WorkloadKind::Decode;
        let findings = fit_findings(&sess);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("generated 1 tokens"), "{findings:?}");
    }

    #[test]
    fn a_sagging_stress_run_warns_and_a_steady_one_does_not() {
        use crate::core::workload::WorkloadKind;

        let iteration = |decode_ms: f64| IterationMetrics {
            prompt_tokens: 200,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms,
            total_ms: decode_ms + 100.0,
            ttft_ms: None,
        };
        let mut sess = sample();
        sess.workload.kind = WorkloadKind::Stress;
        sess.measurements.iterations = [1000.0, 1000.0, 1000.0, 1000.0].map(iteration).to_vec();
        assert!(fit_findings(&sess).is_empty());

        sess.measurements.iterations = [1000.0, 1000.0, 1000.0, 1200.0].map(iteration).to_vec();
        let findings = fit_findings(&sess);
        assert_eq!(findings.len(), 1, "{findings:?}");
        assert!(findings[0].contains("sagged"), "{findings:?}");
    }

    #[test]
    fn a_clock_drop_past_threshold_warns() {
        let mut sess = sample();
        sess.environment.hardware.thermal =
            ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2500.0), avg_mhz: None };
        let mut report = ValidationReport::default();
        check_thermal(&sess, &mut report);
        assert_eq!(report.findings.len(), 1);
        assert_eq!(report.findings[0].severity, Severity::Warning);
        assert_eq!(report.findings[0].check, "thermal");
        assert!(report.findings[0].message.contains("3000"), "{}", report.findings[0].message);
        assert!(report.findings[0].message.contains("2500"), "{}", report.findings[0].message);
        // A warning must not fail the session — the numbers are still real.
        assert!(report.passed());
    }

    #[test]
    fn no_readings_produces_no_thermal_finding() {
        let mut sess = sample();
        sess.environment.hardware.thermal = ThermalSnapshot::default();
        let mut report = ValidationReport::default();
        check_thermal(&sess, &mut report);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn a_small_drop_under_threshold_produces_no_finding() {
        let mut sess = sample();
        sess.environment.hardware.thermal =
            ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2900.0), avg_mhz: None };
        let mut report = ValidationReport::default();
        check_thermal(&sess, &mut report);
        assert!(report.findings.is_empty());
    }

    #[test]
    fn validate_includes_the_thermal_check() {
        let mut sess = sample();
        sess.environment.hardware.thermal =
            ThermalSnapshot { start_mhz: Some(3000.0), end_mhz: Some(2000.0), avg_mhz: None };
        let report = validate(&sess);
        assert!(report.findings.iter().any(|f| f.check == "thermal"));
    }
}
