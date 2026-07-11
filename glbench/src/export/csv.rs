//! CSV export — one row per measured iteration, for spreadsheets and plotting.
//!
//! Hand-rolled (no csv crate): fields are simple numbers and a label, quoted
//! only when they contain a comma or quote. Deterministic column order.

use crate::core::session::BenchmarkSession;

/// The CSV header line (no trailing newline).
pub const HEADER: &str =
    "label,iteration,prompt_tokens,generated_tokens,prefill_ms,decode_ms,total_ms,prefill_tps,decode_tps";

/// Render a session's iterations as CSV (including the header).
pub fn render(session: &BenchmarkSession) -> String {
    let mut s = String::new();
    s.push_str(HEADER);
    s.push('\n');
    let label = &session.metadata.label;
    for (i, it) in session.measurements.iterations.iter().enumerate() {
        s.push_str(&quote(label));
        s.push_str(&format!(
            ",{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            i,
            it.prompt_tokens,
            it.generated_tokens,
            it.prefill_ms,
            it.decode_ms,
            it.total_ms,
            it.prefill_tps(),
            it.decode_tps(),
        ));
    }
    s
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

    #[test]
    fn quotes_only_when_needed() {
        assert_eq!(quote("plain"), "plain");
        assert_eq!(quote("has,comma"), "\"has,comma\"");
        assert_eq!(quote("has\"quote"), "\"has\"\"quote\"");
    }
}
