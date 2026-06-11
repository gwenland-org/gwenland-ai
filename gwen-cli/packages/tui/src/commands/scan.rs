// @INFO: TUI command handler for `gwen scan`. Currently exposes one subcommand: `dataset`.
// Model scanning is deferred to JIN-172.
// @EDITABLE: Yes. Wire additional ScanCommands variants (e.g. Model) when JIN-172 lands.

use clap::{Args, Subcommand};
use gwenland_core::dataset::scan::{CheckKind, DatasetScanResult, ScanOptions, ScanSeverity};
use std::collections::HashSet;
use std::path::PathBuf;

// ── CLI args ──────────────────────────────────────────────────────────────────

// @INFO: Top-level args struct for `gwen scan`. Delegates to a subcommand.
#[derive(Args, Debug)]
#[command(
    about = "Safety scanner for models and datasets",
    long_about = "Run safety scans on JSONL datasets: toxicity detection, PII identification,\n\
                  prompt injection patterns, bias scoring, and category balance analysis.\n\n\
                  Exit code 0 = no error-severity issues. Exit code 1 = errors found.\n\
                  Warnings alone do not trigger exit code 1.\n\n\
                  Examples:\n  \
                    gwen scan dataset -i data.jsonl\n  \
                    gwen scan dataset -i data.jsonl --check safety,pii\n  \
                    gwen scan dataset -i data.jsonl --check safety,pii,injection,bias,balance\n  \
                    gwen scan dataset -i data.jsonl --report scan_report.json\n  \
                    gwen scan dataset -i data.jsonl --patterns extra_patterns.txt"
)]
pub struct ScanArgs {
    #[command(subcommand)]
    pub action: ScanCommands,
}

// @INFO: Available scan subcommands. Model scan is intentionally absent (JIN-172).
#[derive(Subcommand, Debug)]
pub enum ScanCommands {
    /// Scan a JSONL dataset for safety, PII, injection, bias, and balance issues
    Dataset(DatasetScanArgs),
}

// @INFO: Arguments for `gwen scan dataset`.
// @EDITABLE: Yes. Add flags here as new checks are implemented.
#[derive(Args, Debug)]
pub struct DatasetScanArgs {
    /// Input JSONL file to scan (e.g. -i data.jsonl)
    #[arg(short = 'i', long, value_name = "FILE")]
    pub input: PathBuf,

    /// Comma-separated checks to run (default: all five).
    /// Valid values: safety, pii, injection, bias, balance.
    /// Example: --check safety,pii,injection
    #[arg(long = "check", value_delimiter = ',', value_name = "CHECKS")]
    pub check: Vec<String>,

    /// Path to a plain-text file with extra prompt-injection patterns (one per line)
    #[arg(long = "patterns", value_name = "FILE")]
    pub patterns: Option<PathBuf>,

    /// Write the full JSON scan report to this file (e.g. --report scan.json)
    #[arg(long = "report", value_name = "FILE")]
    pub report: Option<PathBuf>,
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

// @INFO: Entry point called from main.rs. Dispatches to the appropriate scan subcommand.
pub async fn run_scan_cmd(args: ScanArgs) {
    match args.action {
        ScanCommands::Dataset(a) => run_dataset_scan(a),
    }
}

// ── Dataset scan runner ───────────────────────────────────────────────────────

// @INFO: Parses --check values, runs the core scan, renders terminal output, optionally writes
// the JSON report, and exits with code 1 if any error-severity issues were found.
// @EDITABLE: Yes. Update output formatting here if the display spec changes.
// @WARNING: Exit code logic: warnings alone → exit 0; any error → exit 1.
fn run_dataset_scan(args: DatasetScanArgs) {
    // Parse --check strings into CheckKind variants.
    let checks: Option<Vec<CheckKind>> = if args.check.is_empty() {
        None // run all
    } else {
        let mut kinds = Vec::new();
        for s in &args.check {
            match CheckKind::from_str(s) {
                Ok(k)  => kinds.push(k),
                Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
            }
        }
        Some(kinds)
    };

    let opts = ScanOptions {
        checks,
        extra_patterns_path: args.patterns,
    };

    let result = match gwenland_core::dataset::scan::run_scan(&args.input, &opts) {
        Ok(r)  => r,
        Err(e) => { eprintln!("error: {}", e); std::process::exit(1); }
    };

    // Print any load-time warnings above the separator.
    let use_color = atty::is(atty::Stream::Stdout);
    let (green, yellow, red, reset) = color_codes(use_color);

    for w in &result.load_warnings {
        println!("  {}{}{}", yellow, w, reset);
    }

    // Render the main output table.
    print_scan_output(&result, use_color, green, yellow, red, reset);

    // Optionally write JSON report.
    if let Some(report_path) = &args.report {
        if let Err(e) = write_report(&result, report_path) {
            eprintln!("warning: could not write report: {}", e);
        }
    }

    // Exit 1 if any error-severity issues exist.
    let has_errors = result.issues.iter().any(|i| i.severity == ScanSeverity::Error);
    if has_errors {
        std::process::exit(1);
    }
}

// ── Terminal renderer ─────────────────────────────────────────────────────────

// @INFO: Renders the human-readable scan summary to stdout.
// Max 10 affected line numbers shown inline per check; remainder is truncated.
// @EDITABLE: Yes. Adjust MAX_INLINE_LINES or column widths for display preference.
fn print_scan_output(
    result:    &DatasetScanResult,
    use_color: bool,
    green:     &str,
    yellow:    &str,
    red:       &str,
    reset:     &str,
) {
    const MAX_INLINE_LINES: usize = 10;
    let sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    println!("{}", sep);
    println!("  {:<20} {}", "Samples scanned", format_count(result.total_scanned));

    // Helper: collect unique affected line numbers for one CheckKind.
    let affected_lines = |kind: &CheckKind| -> Vec<usize> {
        let mut lines: Vec<usize> = result
            .issues
            .iter()
            .filter(|i| &i.check == kind)
            .map(|i| i.line)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        lines.sort_unstable();
        lines
    };

    for kind in &result.checks_run {
        match kind {
            CheckKind::Safety => {
                let lines = affected_lines(kind);
                let has_error = lines.iter().any(|ln| {
                    result.issues.iter().any(|i| {
                        &i.check == kind && i.line == *ln && i.severity == ScanSeverity::Error
                    })
                });
                if lines.is_empty() {
                    println!("  {:<20} {}✓{}", "Toxic content", green, reset);
                } else {
                    let (icon, color) = if has_error { ("✗", red) } else { ("⚠", yellow) };
                    let n     = lines.len();
                    let label = if n == 1 { "sample" } else { "samples" };
                    let shown = &lines[..n.min(MAX_INLINE_LINES)];
                    let lines_str = shown.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
                    let tail = if n > MAX_INLINE_LINES {
                        format!("  (... and {} more)", n - MAX_INLINE_LINES)
                    } else {
                        String::new()
                    };
                    println!(
                        "  {:<20} {}{}{}  {}{}{}  ({} {}: {}){}",
                        "Toxic content", color, n, reset,
                        color, icon, reset,
                        n, label, lines_str, tail,
                    );
                }
            }

            CheckKind::Pii => {
                let lines = affected_lines(kind);
                if lines.is_empty() {
                    println!("  {:<20} {}✓{}", "PII detected", green, reset);
                } else {
                    let n     = lines.len();
                    let label = if n == 1 { "sample" } else { "samples" };
                    let shown = &lines[..n.min(MAX_INLINE_LINES)];
                    let lines_str = shown.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
                    let tail = if n > MAX_INLINE_LINES {
                        format!("  (... and {} more)", n - MAX_INLINE_LINES)
                    } else {
                        String::new()
                    };
                    println!(
                        "  {:<20} {}{}{reset}  {}⚠{reset}  (lines: {}){}",
                        "PII detected", yellow, n, yellow, lines_str, tail,
                        reset = reset,
                    );
                }
            }

            CheckKind::Injection => {
                let lines = affected_lines(kind);
                if lines.is_empty() {
                    println!("  {:<20} {}✓{}", "Prompt injection", green, reset);
                } else {
                    let n     = lines.len();
                    let label = if n == 1 { "sample" } else { "samples" };
                    let shown = &lines[..n.min(MAX_INLINE_LINES)];
                    let lines_str = shown.iter().map(|l| l.to_string()).collect::<Vec<_>>().join(", ");
                    let tail = if n > MAX_INLINE_LINES {
                        format!("  (... and {} more)", n - MAX_INLINE_LINES)
                    } else {
                        String::new()
                    };
                    println!(
                        "  {:<20} {}{}{}  {}✗{}  ({} {}: {}){}",
                        "Prompt injection", red, n, reset, red, reset,
                        n, label, lines_str, tail,
                    );
                }
            }

            CheckKind::Bias => {
                let (icon, color) = if result.bias_warning {
                    ("⚠", yellow)
                } else {
                    ("✓", green)
                };
                println!(
                    "  {:<20} {:.2}        {}{}{}",
                    "Bias score", result.bias_score, color, icon, reset
                );
            }

            CheckKind::Balance => {
                if result.balance_warnings.is_empty() {
                    println!("  {:<20} {}✓{}", "Category balance", green, reset);
                } else {
                    let detail: Vec<String> = result
                        .balance_warnings
                        .iter()
                        .map(|(cat, pct)| format!("{}: {:.1}%", cat, pct))
                        .collect();
                    println!(
                        "  {:<20} {}⚠{}  ({})",
                        "Category balance", yellow, reset,
                        detail.join(", ")
                    );
                }
            }
        }
    }

    println!("{}", sep);

    // Footer.
    let error_count = result
        .issues
        .iter()
        .filter(|i| i.severity == ScanSeverity::Error)
        .map(|i| i.line)
        .collect::<HashSet<_>>()
        .len();

    let warn_count = result
        .issues
        .iter()
        .filter(|i| i.severity == ScanSeverity::Warning)
        .map(|i| i.line)
        .collect::<HashSet<_>>()
        .len();
    let bias_warn   = if result.bias_warning { 1 } else { 0 };
    let bal_warn    = result.balance_warnings.len();
    let total_warns = warn_count + bias_warn + bal_warn;

    if error_count == 0 && total_warns == 0 {
        println!("  {}✓ All checks passed. Dataset is clean.{}", green, reset);
    } else if error_count == 0 {
        println!(
            "  {}{} warning{} found.{} Review before training.",
            yellow, total_warns, if total_warns == 1 { "" } else { "s" }, reset
        );
    } else {
        println!(
            "  {}{} issue{} found.{} Review before training.",
            red, error_count, if error_count == 1 { "" } else { "s" }, reset
        );
    }
}

// ── JSON report writer ────────────────────────────────────────────────────────

// @INFO: Writes the full JSON report to the specified path.
// Issues array includes only per-row (safety/pii/injection) issues — bias and balance
// appear in the summary object.
// @EDITABLE: Yes. Extend the report schema here if consumers need additional fields.
fn write_report(result: &DatasetScanResult, path: &std::path::Path) -> Result<(), String> {
    let issues_json: Vec<serde_json::Value> = result
        .issues
        .iter()
        .map(|i| {
            serde_json::json!({
                "line":     i.line,
                "check":    i.check.as_str(),
                "severity": match i.severity {
                    ScanSeverity::Error   => "error",
                    ScanSeverity::Warning => "warning",
                },
                "detail":   i.detail,
            })
        })
        .collect();

    let checks_run: Vec<&str> = result.checks_run.iter().map(|c| c.as_str()).collect();

    // Build summary object from the pre-computed summary map.
    let mut summary_obj = serde_json::Map::new();
    for kind in &result.checks_run {
        if let Some(s) = result.summary.get(kind) {
            let mut entry = serde_json::json!({
                "status": s.status,
                "count":  s.count,
            });
            if let Some(score) = s.score {
                entry["score"] = serde_json::json!(score);
            }
            summary_obj.insert(kind.as_str().into(), entry);
        }
    }

    let report = serde_json::json!({
        "total_scanned": result.total_scanned,
        "checks_run":    checks_run,
        "issues":        issues_json,
        "summary":       serde_json::Value::Object(summary_obj),
    });

    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| format!("serialization error: {}", e))?;
    std::fs::write(path, json)
        .map_err(|e| format!("cannot write '{}': {}", path.display(), e))?;
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

// @INFO: Returns ANSI color codes for the four named colors, or empty strings when not a TTY.
fn color_codes(use_color: bool) -> (&'static str, &'static str, &'static str, &'static str) {
    if use_color {
        ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m")
    } else {
        ("", "", "", "")
    }
}

// @INFO: Formats a usize with thousands separators (e.g. 5000 → "5,000").
fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(ch);
    }
    out.chars().rev().collect()
}
