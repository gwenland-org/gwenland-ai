//! Candle-free dataset types shared by `dry_run` and `dataset`.
//!
//! Split from `dataset.rs` so that `dry_run` (which has no candle deps) can
//! import `Sample` and `load_jsonl` without pulling in the candle feature gate.

use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;

pub const DEFAULT_MAX_LEN: usize = 1024;

#[derive(Debug, Clone, Deserialize)]
pub struct Sample {
    pub input: String,
    pub output: String,
}

/// Parse a JSONL file into `Sample`s.
///
/// Malformed lines are logged to stderr and skipped. Returns `Err` if the
/// file is empty after filtering.
pub fn load_jsonl(path: &Path) -> Result<Vec<Sample>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("cannot open dataset '{}'", path.display()))?;
    let reader = BufReader::new(file);

    let mut samples = Vec::new();

    for (idx, line_res) in reader.lines().enumerate() {
        let line_no = idx + 1;

        let line = match line_res {
            Ok(l) => l,
            Err(e) => {
                eprintln!("[warn] dataset line {}: read error — {}", line_no, e);
                continue;
            }
        };

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        match serde_json::from_str::<Sample>(trimmed) {
            Ok(s) => samples.push(s),
            Err(e) => {
                eprintln!(
                    "[warn] dataset line {}: malformed JSON, skipping — {}",
                    line_no, e
                );
            }
        }
    }

    if samples.is_empty() {
        bail!(
            "dataset '{}' produced zero valid samples after parsing",
            path.display()
        );
    }

    Ok(samples)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_jsonl(lines: &[&str]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        for line in lines {
            writeln!(f, "{}", line).unwrap();
        }
        f
    }

    #[test]
    fn load_valid_jsonl() {
        let f = write_jsonl(&[
            r#"{"input":"hello","output":"world"}"#,
            r#"{"input":"foo","output":"bar"}"#,
        ]);
        let samples = load_jsonl(f.path()).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].input, "hello");
        assert_eq!(samples[1].output, "bar");
    }

    #[test]
    fn skips_malformed_lines() {
        let f = write_jsonl(&[
            r#"{"input":"good","output":"line"}"#,
            r#"not json at all"#,
            r#"{"input":"also","output":"good"}"#,
        ]);
        let samples = load_jsonl(f.path()).unwrap();
        assert_eq!(samples.len(), 2);
    }

    #[test]
    fn empty_dataset_returns_err() {
        let f = write_jsonl(&["not json", "   ", ""]);
        assert!(load_jsonl(f.path()).is_err());
    }
}
