//! Token-id parity against an independent reference.
//!
//! llama.cpp ships vocabulary-only GGUF files together with reference test
//! vectors (`*.gguf.inp` / `*.gguf.out`) whose expected ids were produced by
//! the HuggingFace tokenizers. Comparing against that **data** — not against
//! any implementation — is what turns "should be correct" into "matches the
//! reference".
//!
//! This is ARTX13 wave A13.0. It skips cleanly when the corpus is absent, so
//! `cargo test` still works on a machine without llama.cpp checked out.
//!
//! Point it elsewhere with `GLTOK_VOCAB_DIR=/path/to/llama.cpp/models`.

use std::path::{Path, PathBuf};

use glcore::tokenizer::{gguf, GllmTokenizer};

const SEP: &str = "__ggml_vocab_test__";

fn corpus_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("GLTOK_VOCAB_DIR") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    // Sibling checkout, the layout this repo uses.
    //
    // ⚠️ Searched by walking *up* rather than by a fixed depth. A hardcoded
    // `nth(3)` silently stopped resolving when this test moved one directory
    // up during the step-4 extraction — and because a missing corpus skips,
    // the test kept reporting `ok` while checking nothing at all. A skip that
    // is indistinguishable from a pass is worse than a failure.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .map(|a| a.join("llama.cpp/models"))
        .find(|p| p.is_dir())
}

/// Split an `.inp` file into its test strings.
///
/// ⚠️ The corpus is stored with LF endings, but a Windows checkout may have
/// rewritten them to CRLF. Several tests are pure-whitespace strings, so an
/// un-normalised read compares different data than the reference used.
fn parse_inp(raw: &str) -> Vec<String> {
    let lf = raw.replace("\r\n", "\n");
    let pat = format!("\n{SEP}\n");
    let mut out: Vec<String> = lf.split(&pat).map(str::to_string).collect();
    // The file ends with a trailing separator, producing one empty tail.
    if out.last().map(|s| s.trim().is_empty()).unwrap_or(false) {
        out.pop();
    }
    out
}

fn parse_out(raw: &str) -> Vec<Vec<u32>> {
    raw.replace("\r\n", "\n")
        .lines()
        .map(|l| l.split_whitespace().filter_map(|t| t.parse().ok()).collect())
        .collect()
}

struct Report {
    vocab: &'static str,
    total: usize,
    passed: usize,
    first_fail: Option<String>,
    load_error: Option<String>,
}

fn check(dir: &Path, name: &'static str) -> Report {
    let base = dir.join(format!("ggml-vocab-{name}.gguf"));
    let mut r = Report {
        vocab: name,
        total: 0,
        passed: 0,
        first_fail: None,
        load_error: None,
    };

    let bytes = match std::fs::read(&base) {
        Ok(b) => b,
        Err(e) => {
            r.load_error = Some(format!("read: {e}"));
            return r;
        }
    };
    let v = match gguf::vocab_from_gguf(&bytes) {
        Ok(v) => v,
        Err(e) => {
            r.load_error = Some(e.to_string());
            return r;
        }
    };
    let tok = GllmTokenizer::new(v);

    let inp = match std::fs::read_to_string(base.with_extension("gguf.inp")) {
        Ok(s) => s,
        Err(e) => {
            r.load_error = Some(format!("inp: {e}"));
            return r;
        }
    };
    let outp = match std::fs::read_to_string(base.with_extension("gguf.out")) {
        Ok(s) => s,
        Err(e) => {
            r.load_error = Some(format!("out: {e}"));
            return r;
        }
    };

    let tests = parse_inp(&inp);
    let want = parse_out(&outp);
    r.total = tests.len().min(want.len());

    for i in 0..r.total {
        // The reference vectors are raw-text tokenizations with no BOS.
        let got = match tok.encode(&tests[i], false) {
            Ok(g) => g,
            Err(e) => {
                if r.first_fail.is_none() {
                    r.first_fail = Some(format!("#{i} {:?} -> error {e}", tests[i]));
                }
                continue;
            }
        };
        if got == want[i] {
            r.passed += 1;
        } else if r.first_fail.is_none() {
            r.first_fail = Some(format!(
                "#{i} {:?}\n      want {:?}\n      got  {:?}",
                tests[i], want[i], got
            ));
        }
    }
    r
}

/// Every vocabulary in the corpus, so the report shows coverage honestly
/// rather than only the ones that happen to pass.
const ALL: &[&str] = &[
    "qwen2",
    "qwen35",
    "llama-bpe",
    "llama-spm",
    "gpt-2",
    "gpt-neox",
    "falcon",
    "starcoder",
    "refact",
    "mpt",
    "command-r",
    "deepseek-coder",
    "deepseek-llm",
    "phi-3",
    "gemma-4",
    "baichuan",
    "aquila",
];

#[test]
fn reference_token_id_parity() {
    let Some(dir) = corpus_dir() else {
        // ⛔ A skip and a pass look identical to `cargo test`, and this one
        // already hid a real breakage once. `GLTOK_REQUIRE_CORPUS=1` turns the
        // skip into a failure, which is what CI should set — locally the skip
        // stays, so the suite still runs without a llama.cpp checkout.
        let msg = "reference corpus not found; set GLTOK_VOCAB_DIR to a \
                   llama.cpp `models/` directory";
        assert!(
            std::env::var("GLTOK_REQUIRE_CORPUS").is_err(),
            "GLTOK_REQUIRE_CORPUS is set but {msg}"
        );
        eprintln!("SKIP parity: {msg}");
        return;
    };
    eprintln!("corpus: {}", dir.display());

    let reports: Vec<Report> = ALL.iter().map(|n| check(&dir, n)).collect();

    // A corpus directory that exists but yields nothing is the same silent
    // no-op as a missing one, one level down.
    assert!(
        reports.iter().any(|r| r.total > 0),
        "corpus at {} produced no test vectors for ANY vocabulary — \
         the directory resolved but its contents did not",
        dir.display()
    );

    eprintln!("\n{:<16} {:>8}  note", "vocab", "parity");
    eprintln!("{}", "-".repeat(72));
    for r in &reports {
        match (&r.load_error, r.total) {
            (Some(e), _) => eprintln!("{:<16} {:>8}  {e}", r.vocab, "-"),
            (None, 0) => eprintln!("{:<16} {:>8}  no vectors", r.vocab, "-"),
            _ => {
                let pct = 100.0 * r.passed as f64 / r.total as f64;
                eprintln!(
                    "{:<16} {:>7.1}%  {}/{}",
                    r.vocab, pct, r.passed, r.total
                );
            }
        }
    }
    eprintln!();
    for r in &reports {
        if let Some(f) = &r.first_fail {
            eprintln!("first mismatch [{}]:\n  {f}\n", r.vocab);
        }
    }

    // Vocabularies this crate claims to support must be exact. The claim is
    // deliberately narrow: an unsupported family should show up as a load
    // refusal in the table above, not as a silent partial pass.
    // Every family the crate claims to support. Listing all fourteen means a
    // regression in any of them fails the build, not just the headline ones.
    let must_be_exact = [
        "qwen2", "qwen35", "llama-bpe", "llama-spm", "gpt-2", "starcoder", "refact", "mpt",
        "command-r", "deepseek-coder", "deepseek-llm", "phi-3", "gemma-4", "falcon",
    ];
    // Families that must REFUSE rather than partially work (see gguf.rs).
    //
    // ⚠️ `gpt-neox` and `aquila` carry no `tokenizer.ggml.pre` key, so they
    // reach llama.cpp's `default` fallback arm. `BpeSplit::DEFAULT` expresses
    // that shape exactly, so this is no longer a capability gap — it is a
    // judgement call: neither ships reference vectors, so enabling them would
    // claim support this crate cannot measure.
    for r in &reports {
        if matches!(r.vocab, "gpt-neox" | "aquila") {
            assert!(
                r.load_error.is_some(),
                "{} must be refused, not silently supported",
                r.vocab
            );
        }
    }
    let mut failures = Vec::new();
    for r in &reports {
        if !must_be_exact.contains(&r.vocab) {
            continue;
        }
        if let Some(e) = &r.load_error {
            failures.push(format!("{}: load failed: {e}", r.vocab));
        } else if r.passed != r.total || r.total == 0 {
            failures.push(format!("{}: {}/{}", r.vocab, r.passed, r.total));
        }
    }
    assert!(failures.is_empty(), "parity failures: {failures:?}");
}
