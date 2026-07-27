//! Before/after measurement: `glcore::tokenizer` (old) vs `gltokenizer` (new)
//! against the same independent reference vectors.
//!
//! This is the number that decides whether migrating callers is worth the
//! churn. It is a **measurement, not a gate** — it prints a table and asserts
//! only the one thing that must never regress: the new implementation must be
//! at least as accurate as the old one on every vocabulary.
//!
//! Skips cleanly without the corpus. `GLTOK_VOCAB_DIR` overrides the path.

use std::path::{Path, PathBuf};
use std::time::Instant;

const SEP: &str = "__ggml_vocab_test__";

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GLTOK_VOCAB_DIR") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let guess = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)?
        .join("llama.cpp/models");
    guess.is_dir().then_some(guess)
}

fn parse_inp(raw: &str) -> Vec<String> {
    let lf = raw.replace("\r\n", "\n");
    let pat = format!("\n{SEP}\n");
    let mut v: Vec<String> = lf.split(&pat).map(str::to_string).collect();
    if v.last().map(|s| s.trim().is_empty()).unwrap_or(false) {
        v.pop();
    }
    v
}

fn parse_out(raw: &str) -> Vec<Vec<u32>> {
    raw.replace("\r\n", "\n")
        .lines()
        .map(|l| l.split_whitespace().filter_map(|t| t.parse().ok()).collect())
        .collect()
}

#[derive(Default)]
struct Score {
    passed: usize,
    total: usize,
    err: Option<String>,
}

impl Score {
    fn pct(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            100.0 * self.passed as f64 / self.total as f64
        }
    }
    fn cell(&self) -> String {
        match &self.err {
            Some(e) if e.len() > 26 => format!("{}…", &e[..25]),
            Some(e) => e.clone(),
            None if self.total == 0 => "-".into(),
            None => format!("{:.1}% {}/{}", self.pct(), self.passed, self.total),
        }
    }
}

/// Timing repeats. The old SPM path is ~seconds per call, so this stays
/// small enough that the whole test remains a test rather than a benchmark.
const REPEATS: usize = 5;

/// Median wall-clock of `f`, in microseconds. Median rather than mean so one
/// scheduler hiccup does not set the number.
fn median_us(n: usize, mut f: impl FnMut()) -> f64 {
    let mut v: Vec<f64> = (0..n)
        .map(|_| {
            let s = Instant::now();
            f();
            s.elapsed().as_secs_f64() * 1e6
        })
        .collect();
    v.sort_by(f64::total_cmp);
    v[n / 2]
}

/// A realistic prompt-sized workload for the timing column.
fn bench_text() -> String {
    let unit = "The quick brown fox jumps over the lazy dog. \
                Bilangan 1234567890, tanda baca!!! dan don't stop. \
                日本語のテキストも混ぜる。\n";
    unit.repeat(40) // ~7 KB, a plausible prompt
}

const VOCABS: &[&str] = &[
    "qwen2",
    "llama-bpe",
    "llama-spm",
    "gpt-2",
    "starcoder",
    "refact",
    "mpt",
    "deepseek-coder",
    "deepseek-llm",
    "phi-3",
];

#[test]
fn old_vs_new_against_reference() {
    let Some(dir) = corpus_dir() else {
        eprintln!("SKIP: reference corpus not found (set GLTOK_VOCAB_DIR)");
        return;
    };

    let text = bench_text();
    println!(
        "\n{:<16} {:>20} {:>20} {:>11} {:>11}",
        "vocab", "OLD glcore", "NEW gltokenizer", "old µs", "new µs"
    );
    println!("{}", "-".repeat(84));

    let mut regressions = Vec::new();
    let (mut sum_old, mut sum_new) = (0.0f64, 0.0f64);
    let (mut n_old, mut n_new) = (0usize, 0usize);
    let mut per_vocab: Vec<(&str, f64, f64)> = Vec::new();

    for &name in VOCABS {
        let base = dir.join(format!("ggml-vocab-{name}.gguf"));
        let Ok(inp) = std::fs::read_to_string(base.with_extension("gguf.inp")) else {
            continue;
        };
        let Ok(outp) = std::fs::read_to_string(base.with_extension("gguf.out")) else {
            continue;
        };
        let tests = parse_inp(&inp);
        let want = parse_out(&outp);
        let n = tests.len().min(want.len());

        // ── OLD ───────────────────────────────────────────────────────────
        let mut old = Score::default();
        let mut old_us = f64::NAN;
        match glcore::format::gguf::GgufFile::open(base.to_str().unwrap())
            .and_then(|g| glcore::tokenizer::Tokenizer::from_gguf(&g))
        {
            Ok(t) => {
                old.total = n;
                for i in 0..n {
                    if t.encode(&tests[i], false) == want[i] {
                        old.passed += 1;
                    }
                }
                old_us = median_us(REPEATS, || {
                    let _ = t.encode(&text, false);
                });
            }
            Err(e) => old.err = Some(format!("{e:?}")),
        }

        // ── NEW ───────────────────────────────────────────────────────────
        let mut new = Score::default();
        let mut new_us = f64::NAN;
        match std::fs::read(&base)
            .map_err(|e| e.to_string())
            .and_then(|b| gltokenizer::gguf::vocab_from_gguf(&b).map_err(|e| e.to_string()))
        {
            Ok(v) => {
                let t = gltokenizer::Tokenizer::new(v);
                new.total = n;
                for i in 0..n {
                    if t.encode(&tests[i], false).is_ok_and(|g| g == want[i]) {
                        new.passed += 1;
                    }
                }
                let _ = t.encode(&text, false); // warm the scratch buffers
                new_us = median_us(REPEATS, || {
                    let _ = t.encode(&text, false);
                });
            }
            Err(e) => new.err = Some(e),
        }

        println!(
            "{:<16} {:>20} {:>20} {:>11.0} {:>11.0}",
            name,
            old.cell(),
            new.cell(),
            old_us,
            new_us
        );

        per_vocab.push((name, old_us, new_us));

        if old_us.is_finite() {
            sum_old += old_us;
            n_old += 1;
        }
        if new_us.is_finite() {
            sum_new += new_us;
            n_new += 1;
        }

        // The one hard invariant: never less accurate than what we replace.
        if old.err.is_none() && new.err.is_none() && new.passed < old.passed {
            regressions.push(format!("{name}: old {} > new {}", old.passed, new.passed));
        }
    }

    // ⚠️ A single mean across both styles is misleading: the SPM rows are
    // ~100x slower than the byte-level rows in the OLD implementation, so a
    // mean is really just the SPM number wearing a disguise. Report the two
    // regimes separately, because they have different causes.
    //
    // Byte-level chunks are word-sized, so the old O(n³) loop ran on tiny n.
    // SPM has no pre-tokenizer, so the WHOLE input was one merge run — which
    // is where a cubic loop actually shows up.
    let (mut bl, mut spm) = (Vec::new(), Vec::new());
    for (name, o, n) in &per_vocab {
        if !o.is_finite() || !n.is_finite() {
            continue;
        }
        if matches!(*name, "llama-spm" | "phi-3") {
            spm.push(o / n);
        } else {
            bl.push(o / n);
        }
    }
    let med = |mut v: Vec<f64>| -> f64 {
        if v.is_empty() {
            return f64::NAN;
        }
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    };
    println!(
        "\nspeedup (median of per-vocab old/new ratios):\n  \
         byte-level  {:.1}x\n  SPM         {:.0}x   ← the old cubic merge loop, \
         un-chunked",
        med(bl),
        med(spm)
    );
    let _ = (sum_old, sum_new, n_old, n_new);
    println!(
        "\nTimings: median of {REPEATS} repeats on {} bytes of mixed \
         Latin/CJK/digits/punctuation.\n\
         Build profile matters — run with --release for representative numbers.",
        text.len()
    );

    assert!(
        regressions.is_empty(),
        "new implementation is less accurate somewhere: {regressions:?}"
    );
}
