// @INFO: Dataset safety scanner for GwenLand.
// Implements five independent checks — safety, pii, injection, bias, balance —
// that can be run individually via --check or all at once (default).
// Per-row checks run in parallel via rayon; aggregate checks (bias, balance) run sequentially.
// @EDITABLE: Yes. Add new checks by implementing a new check_* fn and wiring it into run_scan.

use crate::dataset::schema::{load_rows_from_path, GwenDatasetRow, LoadedLine};
use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::Regex;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

// ── Toxic keyword list ────────────────────────────────────────────────────────

// @INFO: Curated list of ~50 single-token keywords that are unambiguous indicators of toxic content.
// Single-token only so the ±5 window context logic works at token boundaries.
// @EDITABLE: Yes. Add or remove keywords as the dataset domain requires. Keep as single words.
// @WARNING: Do not add overly broad words (e.g. "kill", "attack") — they generate too many false
//           positives. Use words that are toxic in virtually every training-data context.
const TOXIC_KEYWORDS: &[&str] = &[
    // Extreme violence
    "genocide",      "massacre",      "torture",       "mutilate",    "decapitate",
    "slaughter",     "exterminate",   "butcher",       "behead",      "disembowel",
    "eviscerate",    "crucify",       "lynching",      "immolate",    "dismember",
    "impale",        "flay",          "garrote",       "strangle",    "castrate",
    "napalm",        "beheading",     "vivisect",      "firebomb",
    // Dehumanising language
    "subhuman",      "vermin",        "parasites",     "infestation",
    "filth",         "scum",          "degenerate",    "cockroaches",
    // Hate ideology
    "nazi",          "fascist",       "supremacist",   "eugenics",
    "extermination", "annihilation",  "eradication",
    // Sexual violence
    "rape",          "molest",
    // Dangerous materials / synthesis
    "ricin",         "sarin",         "anthrax",       "polonium",
    "cyanide",       "strychnine",    "bioweapon",     "bioterror",
    // Extremist framing
    "jihadi",        "jihadist",
    // Exploitation
    "trafficking",
];

// @INFO: Words that, if found in the ±5-token window around a toxic keyword, indicate a
// benign technical or fictional context. Presence of these downgrades Error → Warning.
// @EDITABLE: Yes. Extend if false-positive rates are high in a specific domain.
const BENIGN_CONTEXT: &[&str] = &[
    "process", "task", "character", "game", "story",
    "function", "thread", "script", "job", "command",
];

// @INFO: Male-associated tokens used for gender-bias scoring.
// @EDITABLE: Yes. Add pronouns or occupational terms relevant to your dataset domain.
const MALE_INDICATORS: &[&str] = &[
    "he", "him", "his", "man", "men", "boy", "father", "husband",
];

// @INFO: Female-associated tokens used for gender-bias scoring (symmetric with MALE_INDICATORS).
// @EDITABLE: Yes.
const FEMALE_INDICATORS: &[&str] = &[
    "she", "her", "hers", "woman", "women", "girl", "mother", "wife",
];

// ── Statics ───────────────────────────────────────────────────────────────────

// @INFO: All regexes and the injection pattern set are compiled once at first use via lazy_static.
// @WARNING: Do not move these into per-call scope — repeated Regex::new() is expensive at scale.
lazy_static! {
    // PII patterns — all Warning severity, never Error.
    static ref EMAIL_RE: Regex = Regex::new(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ).expect("EMAIL_RE");

    static ref PHONE_RE: Regex = Regex::new(
        r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    ).expect("PHONE_RE");

    static ref SSN_RE: Regex = Regex::new(
        r"\b\d{3}-\d{2}-\d{4}\b"
    ).expect("SSN_RE");

    static ref CREDIT_CARD_RE: Regex = Regex::new(
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
    ).expect("CREDIT_CARD_RE");

    static ref IPV4_RE: Regex = Regex::new(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ).expect("IPV4_RE");

    static ref IPV6_RE: Regex = Regex::new(
        r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    ).expect("IPV6_RE");

    // @INFO: Built-in jailbreak/injection patterns. Case-insensitive substring match.
    // @EDITABLE: Yes. Add new jailbreak patterns as they are discovered.
    static ref INJECTION_PATTERNS: HashSet<&'static str> = {
        let mut s = HashSet::new();
        s.insert("ignore previous instructions");
        s.insert("you are now dan");
        s.insert("pretend you are");
        s.insert("disregard your training");
        s.insert("act as if you have no restrictions");
        s.insert("forget your previous instructions");
        s.insert("you have no limitations");
        s.insert("developer mode");
        s.insert("jailbreak");
        s.insert("ignore all prior instructions");
        s.insert("disregard all previous");
        s.insert("bypass your safety");
        s.insert("override your instructions");
        s.insert("you are now free");
        s
    };
}

// ── Public types ──────────────────────────────────────────────────────────────

// @INFO: Identifies which scan category produced an issue.
// @EDITABLE: No. Adding a new variant here requires wiring it into CheckKind::all(), from_str(),
//            run_scan(), and the TUI display function.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckKind {
    Safety,
    Pii,
    Injection,
    Bias,
    Balance,
}

// @INFO: Issue severity. Errors block training readiness; warnings are informational.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScanSeverity {
    Error,
    Warning,
}

// @INFO: A single flagged item from one check on one row.
#[derive(Debug, Clone, Serialize)]
pub struct ScanIssue {
    pub line: usize,
    pub check: CheckKind,
    pub severity: ScanSeverity,
    pub detail: String,
}

// @INFO: Per-check summary entry included in the JSON report.
#[derive(Debug, Clone, Serialize)]
pub struct CheckSummary {
    pub status: String,         // "pass" | "warn" | "fail"
    pub count: usize,           // flagged rows for this check
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f64>,     // only populated for bias
}

// @INFO: Top-level result returned by run_scan(). Consumed by both the JSON reporter and TUI renderer.
// Bias and balance are separate fields rather than issues in the vec — they are aggregate, not per-row.
#[derive(Debug)]
pub struct DatasetScanResult {
    pub total_scanned: usize,
    pub checks_run: Vec<CheckKind>,
    /// Per-row issues from safety / pii / injection checks.
    pub issues: Vec<ScanIssue>,
    /// Computed gender bias score (0.0 = balanced, 1.0 = fully skewed).
    pub bias_score: f64,
    /// True when bias_score > 0.3.
    pub bias_warning: bool,
    /// Category → count, sorted descending. Empty if no rows have a category field.
    pub category_distribution: Vec<(String, usize)>,
    /// Categories that exceed 40% of total samples: (category, percentage).
    pub balance_warnings: Vec<(String, f64)>,
    /// Lines that could not be parsed — not counted in total_scanned.
    pub load_warnings: Vec<String>,
    /// Pre-built per-check summary for the JSON report.
    pub summary: HashMap<CheckKind, CheckSummary>,
}

// @INFO: Options passed from the TUI layer into run_scan.
pub struct ScanOptions {
    /// None = run all 5 checks.
    pub checks: Option<Vec<CheckKind>>,
    /// Path to a newline-delimited file of extra injection patterns.
    pub extra_patterns_path: Option<std::path::PathBuf>,
}

// ── CheckKind helpers ─────────────────────────────────────────────────────────

impl CheckKind {
    // @INFO: Returns the canonical set of all checks in display order.
    pub fn all() -> Vec<Self> {
        vec![
            CheckKind::Safety,
            CheckKind::Pii,
            CheckKind::Injection,
            CheckKind::Bias,
            CheckKind::Balance,
        ]
    }

    // @INFO: Parses a user-supplied string (from --check flag) into a CheckKind.
    // @EDITABLE: Yes, if CheckKind gains new variants.
    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.trim().to_lowercase().as_str() {
            "safety"    => Ok(CheckKind::Safety),
            "pii"       => Ok(CheckKind::Pii),
            "injection" => Ok(CheckKind::Injection),
            "bias"      => Ok(CheckKind::Bias),
            "balance"   => Ok(CheckKind::Balance),
            other => Err(format!(
                "unknown check '{}'; valid: safety, pii, injection, bias, balance",
                other
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            CheckKind::Safety    => "safety",
            CheckKind::Pii       => "pii",
            CheckKind::Injection => "injection",
            CheckKind::Bias      => "bias",
            CheckKind::Balance   => "balance",
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

// @INFO: Main entry point for the scan subcommand. Loads rows, runs checks in parallel (per-row)
// and sequentially (aggregate), then assembles the DatasetScanResult.
// @EDITABLE: Yes. Wire new checks here after implementing their check_* functions.
// @WARNING: Bias and balance are aggregate — do NOT move them into the rayon par_iter block.
pub fn run_scan(path: &Path, opts: &ScanOptions) -> Result<DatasetScanResult, String> {
    let checks_to_run = opts.checks.clone().unwrap_or_else(CheckKind::all);

    let run_safety    = checks_to_run.contains(&CheckKind::Safety);
    let run_pii       = checks_to_run.contains(&CheckKind::Pii);
    let run_injection = checks_to_run.contains(&CheckKind::Injection);
    let run_bias      = checks_to_run.contains(&CheckKind::Bias);
    let run_balance   = checks_to_run.contains(&CheckKind::Balance);

    // Load extra injection patterns from file if supplied.
    let extra_patterns = match &opts.extra_patterns_path {
        Some(p) => load_extra_patterns(p)?,
        None    => HashSet::new(),
    };

    // Load all rows — format auto-detected from first parseable line.
    let loaded = load_rows_from_path(path)?;
    let mut load_warnings: Vec<String> = Vec::new();
    let mut rows: Vec<(usize, GwenDatasetRow)> = Vec::new();

    for entry in loaded {
        match entry {
            LoadedLine::Row { line_no, row } => rows.push((line_no, row)),
            LoadedLine::Skipped { line_no, reason } => {
                load_warnings.push(format!("⚠ Line {}: skipped ({})", line_no, reason));
            }
        }
    }

    let total_scanned = rows.len();

    // ── Phase 1: per-row checks (parallel) ───────────────────────────────────

    // @INFO: rayon splits the row slice across CPU cores. Each closure is Send because
    // extra_patterns is shared read-only behind a reference.
    let per_row_issues: Vec<Vec<ScanIssue>> = rows
        .par_iter()
        .map(|(line_no, row)| {
            // Combine input + output so checks scan both fields.
            let text = format!("{} {}", row.input, row.output);
            let mut issues: Vec<ScanIssue> = Vec::new();
            if run_safety    { issues.extend(check_safety(&text, *line_no)); }
            if run_pii       { issues.extend(check_pii(&text, *line_no)); }
            if run_injection { issues.extend(check_injection(&text, *line_no, &extra_patterns)); }
            issues
        })
        .collect();

    let mut issues: Vec<ScanIssue> = per_row_issues.into_iter().flatten().collect();
    // Sort by line number so the output is deterministic.
    issues.sort_by_key(|i| i.line);

    // ── Phase 2: aggregate checks (sequential) ───────────────────────────────

    let mut bias_score   = 0.0f64;
    let mut bias_warning = false;
    let mut category_counts: HashMap<String, usize> = HashMap::new();

    if run_bias || run_balance {
        let mut male_count   = 0usize;
        let mut female_count = 0usize;

        for (_, row) in &rows {
            if run_balance {
                if let Some(cat) = &row.category {
                    *category_counts.entry(cat.clone()).or_insert(0) += 1;
                }
            }

            if run_bias {
                let combined = format!("{} {}", row.input, row.output).to_lowercase();
                for token in combined.split_whitespace() {
                    let clean = token.trim_matches(|c: char| !c.is_alphabetic());
                    if MALE_INDICATORS.contains(&clean)   { male_count   += 1; }
                    if FEMALE_INDICATORS.contains(&clean) { female_count += 1; }
                }
            }
        }

        if run_bias {
            bias_score = (male_count as f64 - female_count as f64).abs()
                / (male_count as f64 + female_count as f64 + 1.0);
            bias_warning = bias_score > 0.3;
        }
    }

    let mut category_distribution: Vec<(String, usize)> =
        category_counts.iter().map(|(k, v)| (k.clone(), *v)).collect();
    category_distribution.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    // Categories that exceed 40% of total samples.
    let mut balance_warnings: Vec<(String, f64)> = Vec::new();
    if run_balance && total_scanned > 0 {
        for (cat, count) in &category_distribution {
            let pct = *count as f64 / total_scanned as f64 * 100.0;
            if pct > 40.0 {
                balance_warnings.push((cat.clone(), pct));
            }
        }
    }

    // ── Build summary ─────────────────────────────────────────────────────────

    let summary = build_summary(
        &checks_to_run,
        &issues,
        bias_score,
        bias_warning,
        &balance_warnings,
    );

    Ok(DatasetScanResult {
        total_scanned,
        checks_run: checks_to_run,
        issues,
        bias_score,
        bias_warning,
        category_distribution,
        balance_warnings,
        load_warnings,
        summary,
    })
}

// ── Per-row check functions ───────────────────────────────────────────────────

// @INFO: Safety check using token-level keyword matching with a ±5-token context window.
// Matching a BENIGN_CONTEXT word in the window downgrades severity from Error to Warning,
// allowing legitimate uses of toxic words (e.g. "kill the process") to pass without flagging.
// @EDITABLE: Yes. Adjust WINDOW_HALF to change context sensitivity.
// @WARNING: Deduplicates per keyword per line — one issue per unique keyword match.
fn check_safety(text: &str, line_no: usize) -> Vec<ScanIssue> {
    const WINDOW_HALF: usize = 5;

    let lower  = text.to_lowercase();
    let tokens: Vec<&str> = lower.split_whitespace().collect();
    let mut issues: Vec<ScanIssue> = Vec::new();
    let mut seen: HashSet<&str> = HashSet::new();

    for (idx, token) in tokens.iter().enumerate() {
        let clean = token.trim_matches(|c: char| !c.is_alphabetic());
        if clean.is_empty() || !TOXIC_KEYWORDS.contains(&clean) || seen.contains(clean) {
            continue;
        }
        seen.insert(clean);

        // Extract window tokens on both sides.
        let start  = idx.saturating_sub(WINDOW_HALF);
        let end    = (idx + WINDOW_HALF + 1).min(tokens.len());
        let window = &tokens[start..end];

        let has_benign = window.iter().any(|t| {
            let ct = t.trim_matches(|c: char| !c.is_alphabetic());
            BENIGN_CONTEXT.contains(&ct)
        });

        issues.push(ScanIssue {
            line:     line_no,
            check:    CheckKind::Safety,
            severity: if has_benign { ScanSeverity::Warning } else { ScanSeverity::Error },
            detail:   format!("toxic keyword: `{}`", clean),
        });
    }

    issues
}

// @INFO: PII check using pre-compiled regexes. All PII matches are Warning severity.
// Reports match type in the detail field so the user knows what kind of PII was found.
// @EDITABLE: Yes. Add new regex patterns by adding a new lazy_static Regex and a check_re! call.
fn check_pii(text: &str, line_no: usize) -> Vec<ScanIssue> {
    let mut issues: Vec<ScanIssue> = Vec::new();

    // @TODO: Add more PII types here as needed (passport numbers, NIDs, etc.)
    macro_rules! check_re {
        ($re:expr, $kind:literal) => {
            if $re.is_match(text) {
                issues.push(ScanIssue {
                    line:     line_no,
                    check:    CheckKind::Pii,
                    severity: ScanSeverity::Warning,
                    detail:   format!("{} detected", $kind),
                });
            }
        };
    }

    check_re!(EMAIL_RE,       "email address");
    check_re!(PHONE_RE,       "phone number");
    check_re!(SSN_RE,         "SSN (Social Security Number)");
    check_re!(CREDIT_CARD_RE, "credit card number");
    check_re!(IPV4_RE,        "IPv4 address");
    check_re!(IPV6_RE,        "IPv6 address");

    issues
}

// @INFO: Injection detection via case-insensitive substring match against a built-in
// HashSet of known jailbreak patterns plus any user-supplied extra patterns.
// Stops at the first match per row — one injection issue is enough to flag a line.
// @EDITABLE: Yes. Add patterns to INJECTION_PATTERNS lazy_static or via --patterns flag.
fn check_injection(
    text:           &str,
    line_no:        usize,
    extra_patterns: &HashSet<String>,
) -> Vec<ScanIssue> {
    let lower = text.to_lowercase();

    // Check built-in patterns first.
    for pattern in INJECTION_PATTERNS.iter() {
        if lower.contains(*pattern) {
            return vec![ScanIssue {
                line:     line_no,
                check:    CheckKind::Injection,
                severity: ScanSeverity::Error,
                detail:   format!("injection pattern: `{}`", pattern),
            }];
        }
    }

    // Then user-supplied patterns.
    for pattern in extra_patterns {
        if lower.contains(pattern.as_str()) {
            return vec![ScanIssue {
                line:     line_no,
                check:    CheckKind::Injection,
                severity: ScanSeverity::Error,
                detail:   format!("custom injection pattern: `{}`", pattern),
            }];
        }
    }

    vec![]
}

// ── Helper functions ──────────────────────────────────────────────────────────

// @INFO: Reads extra injection patterns from a plain-text file, one pattern per line.
// Lines are lowercased and empty lines are skipped.
// @EDITABLE: Yes. Extend if a different pattern file format is needed.
fn load_extra_patterns(path: &Path) -> Result<HashSet<String>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read patterns file '{}': {}", path.display(), e))?;
    let set = content
        .lines()
        .map(|l| l.trim().to_lowercase())
        .filter(|l| !l.is_empty())
        .collect();
    Ok(set)
}

// @INFO: Assembles the per-check summary map used by both the JSON reporter and terminal display.
// Bias and balance get special handling since they are aggregate rather than per-row.
// @EDITABLE: Yes. Adjust status thresholds here if scoring semantics change.
fn build_summary(
    checks:          &[CheckKind],
    issues:          &[ScanIssue],
    bias_score:      f64,
    bias_warning:    bool,
    balance_warnings: &[(String, f64)],
) -> HashMap<CheckKind, CheckSummary> {
    let mut map = HashMap::new();

    for kind in checks {
        let summary = match kind {
            CheckKind::Bias => CheckSummary {
                status: if bias_warning { "warn".into() } else { "pass".into() },
                count:  if bias_warning { 1 } else { 0 },
                score:  Some(bias_score),
            },
            CheckKind::Balance => CheckSummary {
                status: if balance_warnings.is_empty() { "pass".into() } else { "warn".into() },
                count:  balance_warnings.len(),
                score:  None,
            },
            _ => {
                let kind_issues: Vec<_> = issues.iter().filter(|i| &i.check == kind).collect();
                let has_error = kind_issues.iter().any(|i| i.severity == ScanSeverity::Error);
                let has_warn  = kind_issues.iter().any(|i| i.severity == ScanSeverity::Warning);
                // Deduplicate by line number for count.
                let unique_lines: HashSet<usize> =
                    kind_issues.iter().map(|i| i.line).collect();
                CheckSummary {
                    status: if has_error { "fail".into() } else if has_warn { "warn".into() } else { "pass".into() },
                    count:  unique_lines.len(),
                    score:  None,
                }
            }
        };
        map.insert(kind.clone(), summary);
    }

    map
}
