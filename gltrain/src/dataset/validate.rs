use crate::dataset::schema::{GwenDatasetRow, RawDatasetRow};
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::io::{BufRead, BufReader, Write};
use std::path::Path;
use std::sync::OnceLock;

// Compiled once, shared across threads.
fn re_think_open() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)<think>").unwrap())
}
fn re_think_close() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)</think>").unwrap())
}
fn re_thought_open() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)<thought>").unwrap())
}
fn re_thought_close() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)</thought>").unwrap())
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    Error,
    Warning,
}

#[derive(Debug, Clone, Serialize)]
pub struct Issue {
    pub line: usize,
    pub severity: Severity,
    pub code: String,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fixable: Option<bool>,
}

#[derive(Debug, Default)]
pub struct ValidationResult {
    pub total: usize,
    pub valid: usize,
    pub error_count: usize,
    pub warning_count: usize,
    pub issues: Vec<Issue>,
    pub ready: bool,
}

// Per-line parse result passed from reader thread to validator.
struct ParsedLine {
    line_no: usize,
    result: LineResult,
}

enum LineResult {
    Empty,
    JsonError(String),
    ConversionError(String),
    Row(GwenDatasetRow, String), // row + original JSON text for fix output
}

pub struct ValidateOptions {
    pub strict: bool,
    pub fix: bool,
    pub inplace: bool,
}

pub fn run_validation(
    input_path: &Path,
    opts: &ValidateOptions,
) -> Result<ValidationResult, String> {
    // Read all lines into memory for rayon parallelism while keeping line numbers.
    let file = std::fs::File::open(input_path)
        .map_err(|e| format!("cannot open file: {}", e))?;

    let reader = BufReader::new(file);
    let raw_lines: Vec<(usize, String)> = reader
        .lines()
        .enumerate()
        .map(|(i, line)| {
            let line = line.map_err(|e| format!("line {}: read error: {}", i + 1, e))?;
            Ok((i + 1, line))
        })
        .collect::<Result<Vec<_>, String>>()?;

    let total_non_empty = raw_lines.iter().filter(|(_, l)| !l.trim().is_empty()).count();

    // Parse + validate in parallel.
    let parsed: Vec<ParsedLine> = raw_lines
        .into_par_iter()
        .map(|(line_no, raw)| {
            if raw.trim().is_empty() {
                return ParsedLine { line_no, result: LineResult::Empty };
            }
            match serde_json::from_str::<RawDatasetRow>(&raw) {
                Err(e) => ParsedLine {
                    line_no,
                    result: LineResult::JsonError(e.to_string()),
                },
                Ok(raw_row) => match raw_row.into_gwen_row() {
                    Err(e) => ParsedLine {
                        line_no,
                        result: LineResult::ConversionError(e.to_string()),
                    },
                    Ok(row) => ParsedLine {
                        line_no,
                        result: LineResult::Row(row, raw),
                    },
                },
            }
        })
        .collect();

    // Validate sequentially to keep issue list in line-number order.
    let mut issues: Vec<Issue> = Vec::new();
    let mut fixed_rows: Vec<String> = Vec::new();
    let mut valid = 0usize;

    for p in &parsed {
        match &p.result {
            LineResult::Empty => continue,
            LineResult::JsonError(msg) => {
                issues.push(Issue {
                    line: p.line_no,
                    severity: Severity::Error,
                    code: "invalid_json".into(),
                    message: format!("invalid JSON: {}", msg),
                    fixable: None,
                });
            }
            LineResult::ConversionError(msg) => {
                issues.push(Issue {
                    line: p.line_no,
                    severity: Severity::Error,
                    code: "missing_output".into(),
                    message: msg.clone(),
                    fixable: None,
                });
            }
            LineResult::Row(row, original_json) => {
                let row_issues = validate_row(row, p.line_no, opts.strict);
                let has_error = row_issues.iter().any(|i| i.severity == Severity::Error);
                let fixable_only = !has_error
                    && row_issues.iter().all(|i| i.fixable == Some(true));

                if row_issues.is_empty() {
                    valid += 1;
                    if opts.fix {
                        fixed_rows.push(build_fixed_json(row));
                    }
                } else if opts.fix && !has_error && fixable_only {
                    // Apply fixes in-memory and include the row.
                    let fixed = apply_row_fixes(row);
                    fixed_rows.push(build_fixed_json(&fixed));
                    valid += 1;
                } else if opts.fix && !has_error {
                    // Partially fixable warnings — apply what we can, include row.
                    let fixed = apply_row_fixes(row);
                    fixed_rows.push(build_fixed_json(&fixed));
                    valid += 1;
                } else if opts.fix {
                    // Has unfixable errors — skip from fixed file.
                    let _ = original_json;
                }

                issues.extend(row_issues);
            }
        }
    }

    // Count errors/warnings after strict upgrade already applied inside validate_row.
    let error_count = issues.iter().filter(|i| i.severity == Severity::Error).count();
    let warning_count = issues.iter().filter(|i| i.severity == Severity::Warning).count();
    let ready = error_count == 0;

    if opts.fix && !fixed_rows.is_empty() {
        let out_path = if opts.inplace {
            input_path.to_path_buf()
        } else {
            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("dataset");
            let parent = input_path.parent().unwrap_or(Path::new("."));
            parent.join(format!("{}.fixed.jsonl", stem))
        };
        write_fixed_file(&out_path, &fixed_rows)
            .map_err(|e| format!("cannot write fixed file: {}", e))?;
    }

    Ok(ValidationResult {
        total: total_non_empty,
        valid,
        error_count,
        warning_count,
        issues,
        ready,
    })
}

fn validate_row(row: &GwenDatasetRow, line_no: usize, strict: bool) -> Vec<Issue> {
    let mut issues = Vec::new();

    let mut push = |severity: Severity, code: &str, message: String, fixable: Option<bool>| {
        let sev = if strict && severity == Severity::Warning {
            Severity::Error
        } else {
            severity
        };
        issues.push(Issue {
            line: line_no,
            severity: sev,
            code: code.into(),
            message,
            fixable,
        });
    };

    // Empty input/output
    if row.input.trim().is_empty() {
        push(Severity::Error, "empty_input", "empty input string".into(), None);
    }
    if row.output.trim().is_empty() {
        push(Severity::Error, "empty_output", "empty output string".into(), None);
    }

    // <think> checks on output
    let open_count = re_think_open().find_iter(&row.output).count();
    let close_count = re_think_close().find_iter(&row.output).count();

    if open_count > 1 {
        push(Severity::Error, "nested_think", "nested <think> tags detected".into(), None);
    } else if open_count != close_count {
        push(Severity::Error, "unclosed_think", "<think> tag is unclosed".into(), None);
    }

    // <think> in input field (warning)
    if re_think_open().is_match(&row.input) {
        push(
            Severity::Warning,
            "think_in_input",
            "<think> tag found in input field".into(),
            None,
        );
    }

    // <thought> tag (wrong tag, fixable warning)
    if re_thought_open().is_match(&row.output) || re_thought_close().is_match(&row.output) {
        push(
            Severity::Warning,
            "wrong_think_tag",
            "found <thought> tag — should be <think>".into(),
            Some(true),
        );
    }

    // Missing category (warning)
    if row.category.is_none() {
        push(Severity::Warning, "missing_category", "missing `category` field".into(), None);
    }

    // Very short output (warning)
    if !row.output.trim().is_empty() && row.output.trim().len() < 10 {
        push(
            Severity::Warning,
            "short_output",
            format!("output is very short ({} chars)", row.output.trim().len()),
            None,
        );
    }

    issues
}

fn apply_row_fixes(row: &GwenDatasetRow) -> GwenDatasetRow {
    let output = re_thought_open()
        .replace_all(&row.output, "<think>")
        .into_owned();
    let output = re_thought_close()
        .replace_all(&output, "</think>")
        .into_owned();

    GwenDatasetRow {
        input: row.input.trim().to_string(),
        output: output.trim().to_string(),
        category: row.category.clone(),
        source_format: row.source_format.clone(),
    }
}

fn build_fixed_json(row: &GwenDatasetRow) -> String {
    let mut map = serde_json::Map::new();
    map.insert("input".into(), serde_json::Value::String(row.input.clone()));
    map.insert("output".into(), serde_json::Value::String(row.output.clone()));
    if let Some(cat) = &row.category {
        map.insert("category".into(), serde_json::Value::String(cat.clone()));
    }
    serde_json::to_string(&serde_json::Value::Object(map)).unwrap_or_default()
}

fn write_fixed_file(path: &Path, rows: &[String]) -> std::io::Result<()> {
    let mut file = std::fs::File::create(path)?;
    for row in rows {
        file.write_all(row.as_bytes())?;
        file.write_all(b"\n")?;
    }
    Ok(())
}
