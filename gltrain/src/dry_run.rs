// @INFO: Shared dry-run report primitives used by all commands that support --dry-run.
//        Each command builds a DryRunReport, then calls print_report() or to_json().
// @DANGER: All functions here must be read-only — no file writes, no spawns, no network calls.

use serde::Serialize;

// ── exit codes ────────────────────────────────────────────────────────────────

pub const EXIT_READY: i32 = 0;
pub const EXIT_VALIDATION_FAILED: i32 = 1;
pub const EXIT_INSUFFICIENT_RESOURCES: i32 = 2;

// ── report types ─────────────────────────────────────────────────────────────

/// A single line in the dry-run summary table.
#[derive(Debug, Clone, Serialize)]
pub struct DryRunLine {
    pub label: String,
    pub value: String,
    pub ok: bool,
}

impl DryRunLine {
    pub fn ok(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self { label: label.into(), value: value.into(), ok: true }
    }
    pub fn fail(label: impl Into<String>, value: impl Into<String>) -> Self {
        Self { label: label.into(), value: value.into(), ok: false }
    }
    pub fn info(label: impl Into<String>, value: impl Into<String>) -> Self {
        // info lines are always "ok" for JSON purposes; they carry no pass/fail semantics
        Self { label: label.into(), value: value.into(), ok: true }
    }
}

/// Full dry-run report for one command.
#[derive(Debug, Serialize)]
pub struct DryRunReport {
    pub command: String,
    pub valid: bool,
    pub lines: Vec<DryRunLine>,
    /// Optional structured fields for JSON consumers (e.g. vram_gb, disk_free_gb).
    #[serde(flatten)]
    pub extras: serde_json::Map<String, serde_json::Value>,
}

impl DryRunReport {
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            valid: true,
            lines: Vec::new(),
            extras: serde_json::Map::new(),
        }
    }

    pub fn push(&mut self, line: DryRunLine) {
        if !line.ok {
            self.valid = false;
        }
        self.lines.push(line);
    }

    pub fn set<V: Into<serde_json::Value>>(&mut self, key: &str, value: V) {
        self.extras.insert(key.to_string(), value.into());
    }

    pub fn exit_code(&self) -> i32 {
        if self.valid { EXIT_READY } else { EXIT_VALIDATION_FAILED }
    }
}

// ── output helpers ────────────────────────────────────────────────────────────

/// Print the report as a bordered TUI-style summary table.
pub fn print_report(report: &DryRunReport) {
    let title = format!(" gwen {} --dry-run ", report.command);
    let inner_width: usize = 39;
    let border = "─".repeat(inner_width + 2);

    println!("┌{}┐", border);
    println!("│ {:<inner_width$} │", title, inner_width = inner_width);
    println!("│{}│", " ".repeat(inner_width + 2));

    for line in &report.lines {
        let symbol = if line.ok { "✦" } else { "✗" };
        let row = format!("{} {:<16} {}", symbol, format!("{}:", line.label), line.value);
        println!("│  {:<inner_width$}│", row, inner_width = inner_width);
    }

    println!("│{}│", " ".repeat(inner_width + 2));

    let verdict = if report.valid {
        "Ready to run. Dry-run only."
    } else {
        "Cannot run. Fix issues above."
    };
    println!("│  {:<inner_width$}│", verdict, inner_width = inner_width);
    println!("└{}┘", border);
}

/// Emit a single JSON object on stdout (for --json or --non-interactive --json).
pub fn print_json(report: &DryRunReport) {
    let val = serde_json::to_value(report).unwrap_or(serde_json::Value::Null);
    println!("{}", val);
}
