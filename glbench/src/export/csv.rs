//! CSV export — one row per measured iteration, for spreadsheets and plotting.
//!
//! Hand-rolled (no csv crate): fields are simple numbers and a label, quoted
//! only when they contain a comma or quote. Deterministic column order.

use crate::core::session::BenchmarkSession;

/// The CSV header line (no trailing newline).
///
/// `energy_j_per_token` repeats the same session-level figure on every row
/// (like `label` already does) rather than living in a column of its own —
/// CSV here is flat, one-row-per-iteration, with no separate session-summary
/// row, so a session-level fact has nowhere else to go. Empty (not `0`) when
/// energy was not measured — never fabricate a number that was not read.
pub const HEADER: &str = "label,iteration,prompt_tokens,generated_tokens,prefill_ms,decode_ms,total_ms,prefill_tps,decode_tps,energy_j_per_token";

/// Render a session's iterations as CSV (including the header).
///
/// The cold (first-ever) iteration, when present, is emitted as its own row
/// with `iteration` = `cold` rather than folded into the numbered warm rows
/// or omitted — it is a real measurement (see `MeasurementSet::cold`'s
/// docs), and a spreadsheet reader filtering on that literal can separate it
/// from the warm statistics exactly as the terminal/Markdown reports already
/// do visually.
pub fn render(session: &BenchmarkSession) -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    s.push('\n');
    let label = &session.metadata.label;
    let energy_field = match session.measurements.joules_per_token() {
        Some(jpt) => format!("{jpt:.4}"),
        None => String::new(),
    };
    if let Some(c) = &session.measurements.cold {
        push_row(&mut s, label, "cold", c, &energy_field);
    }
    for (i, it) in session.measurements.iterations.iter().enumerate() {
        push_row(&mut s, label, &i.to_string(), it, &energy_field);
    }
    s
}

fn push_row(
    s: &mut String,
    label: &str,
    iteration: &str,
    it: &crate::core::metrics::IterationMetrics,
    energy_field: &str,
) {
    s.push_str(&quote(label));
    s.push_str(&format!(
        ",{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4},{}\n",
        iteration,
        it.prompt_tokens,
        it.generated_tokens,
        it.prefill_ms,
        it.decode_ms,
        it.total_ms,
        it.prefill_tps(),
        it.decode_tps(),
        energy_field,
    ));
}

/// Quote a field if it contains a comma, quote, or newline (RFC-4180 style).
fn quote(field: &str) -> String {
    if field.contains([',', '"', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
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

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote("has,comma"), "\"has,comma\"");
        assert_eq!(quote("has\"quote"), "\"has\"\"quote\"");
    }

    fn sample() -> BenchmarkSession {
        let mut m = MeasurementSet::default();
        m.iterations.push(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 100.0,
            decode_ms: 4000.0,
            total_ms: 4100.0,
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

    #[test]
    fn cold_iteration_gets_its_own_labeled_row() {
        let mut sess = sample();
        sess.measurements.cold = Some(IterationMetrics {
            prompt_tokens: 100,
            generated_tokens: 128,
            prefill_ms: 500.0,
            decode_ms: 6000.0,
            total_ms: 6500.0,
        });
        let out = render(&sess);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3, "header + cold row + one warm row: {out}");
        assert!(lines[1].starts_with("test-run,cold,"), "{}", lines[1]);
        assert!(lines[2].starts_with("test-run,0,"), "{}", lines[2]);
    }

    #[test]
    fn no_cold_iteration_means_no_cold_row() {
        let out = render(&sample());
        assert!(!out.contains(",cold,"), "{out}");
    }

    #[test]
    fn energy_column_is_empty_not_zero_when_unmeasured() {
        let mut sess = sample();
        sess.measurements.energy_joules = None;
        let out = render(&sess);
        let data_line = out.lines().nth(1).unwrap();
        // Last field is energy_j_per_token; empty means "not measured".
        assert!(data_line.ends_with(','), "{data_line}");
    }

    #[test]
    fn energy_column_carries_the_real_figure_when_measured() {
        let mut sess = sample();
        sess.measurements.energy_joules = Some(25.6); // 128 tokens -> 0.2 J/token
        let out = render(&sess);
        let data_line = out.lines().nth(1).unwrap();
        assert!(data_line.ends_with(",0.2000"), "{data_line}");
    }

    #[test]
    fn header_declares_the_energy_column() {
        assert!(HEADER.ends_with("energy_j_per_token"));
    }
}
