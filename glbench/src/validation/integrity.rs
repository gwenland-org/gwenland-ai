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
    super::deterministic::check(session, &mut report);
    super::reproducibility::check(session, &mut report);
    report
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
