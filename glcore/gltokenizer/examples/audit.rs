//! GGUF tokenizer support audit.
//!
//! Prints, for every reference vocabulary in a llama.cpp `models/` directory:
//! what the GGUF declares, whether this crate loads it, and its parity score.
//!
//! For a vocabulary this crate **refuses**, the audit additionally *probes*
//! every splitter shape it knows and reports the best score reached.
//!
//! ⚠️ A probe is a diagnostic, never a claim. It answers "how far off are we,
//! and along which axis" so an open item has a size instead of a shrug. A
//! probe reaching 46/46 is a reason to add a real mapping in `gguf.rs`; it is
//! not itself support, because nothing outside this file can reach it.
//!
//! Run: cargo run -p gltokenizer --example audit --release [-- <models_dir>]

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use gltokenizer::gguf::{self, Meta};
use gltokenizer::{BpeSplit, PreTok, Style, Tokenizer, Vocab, VocabParts};

const SEP: &str = "__ggml_vocab_test__";

/// Every `ggml-vocab-*.gguf` llama.cpp ships, so coverage is reported over the
/// whole corpus rather than over the subset that happens to work.
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
    "bert-bge",
    "nomic-bert-moe",
];

/// The splitter shapes to try when the real mapping refuses.
const PROBES: &[(&str, BpeSplit)] = &[
    ("GPT2", BpeSplit::GPT2),
    ("STARCODER", BpeSplit::STARCODER),
    ("LLAMA3", BpeSplit::LLAMA3),
    ("QWEN2", BpeSplit::QWEN2),
];

fn parse_inp(raw: &str) -> Vec<String> {
    let lf = raw.replace("\r\n", "\n");
    let pat = format!("\n{SEP}\n");
    let mut out: Vec<String> = lf.split(&pat).map(str::to_string).collect();
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

/// Score a tokenizer against the reference vectors; also return the first
/// mismatch, which is what actually points at a cause.
fn score(tok: &Tokenizer, tests: &[String], want: &[Vec<u32>]) -> (usize, usize, Option<String>) {
    let n = tests.len().min(want.len());
    let mut passed = 0;
    let mut first: Option<String> = None;
    for i in 0..n {
        match tok.encode(&tests[i], false) {
            Ok(got) if got == want[i] => passed += 1,
            Ok(got) => {
                if first.is_none() {
                    let at = (0..got.len().min(want[i].len()))
                        .find(|&k| got[k] != want[i][k])
                        .unwrap_or(got.len().min(want[i].len()));
                    first = Some(format!(
                        "#{i} {:?}\n      want {:?}\n      got  {:?}\n      diverges at index {at}",
                        tests[i], want[i], got
                    ));
                }
            }
            Err(e) => {
                if first.is_none() {
                    first = Some(format!("#{i} {:?} -> error {e}", tests[i]));
                }
            }
        }
    }
    (passed, n, first)
}

/// What the file itself declares, before any interpretation.
struct Declared {
    model: String,
    pre: String,
    tokens: usize,
    merges: usize,
    scores: usize,
    add_bos: Option<bool>,
}

fn declared(m: &HashMap<String, Meta>) -> Declared {
    let s = |k: &str| match m.get(k) {
        Some(Meta::Str(v)) => v.clone(),
        _ => "-".to_string(),
    };
    let n = |k: &str| match m.get(k) {
        Some(Meta::ArrStr(v)) => v.len(),
        Some(Meta::ArrF32(v)) => v.len(),
        _ => 0,
    };
    Declared {
        model: s("tokenizer.ggml.model"),
        pre: s("tokenizer.ggml.pre"),
        tokens: n("tokenizer.ggml.tokens"),
        merges: n("tokenizer.ggml.merges"),
        scores: n("tokenizer.ggml.scores"),
        add_bos: match m.get("tokenizer.ggml.add_bos_token") {
            Some(Meta::Bool(b)) => Some(*b),
            _ => None,
        },
    }
}

/// Build a vocabulary bypassing `gguf.rs`'s name mapping, so a refused family
/// can still be measured.
fn probe_vocab(m: &HashMap<String, Meta>, split: BpeSplit) -> Result<Vocab, String> {
    let tokens = match m.get("tokenizer.ggml.tokens") {
        Some(Meta::ArrStr(v)) => v.clone(),
        _ => return Err("no tokens".into()),
    };
    let merges: Vec<(String, String)> = match m.get("tokenizer.ggml.merges") {
        Some(Meta::ArrStr(v)) => v
            .iter()
            .filter_map(|s| s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .collect(),
        _ => Vec::new(),
    };
    let special_ids: Vec<u32> = match m.get("tokenizer.ggml.token_type") {
        Some(Meta::ArrI32(t)) => t
            .iter()
            .enumerate()
            .filter(|(_, &ty)| ty == 3 || ty == 4)
            .map(|(i, _)| i as u32)
            .collect(),
        _ => Vec::new(),
    };
    let eos = match m.get("tokenizer.ggml.eos_token_id") {
        Some(Meta::U32(v)) => *v,
        Some(Meta::I32(v)) if *v >= 0 => *v as u32,
        _ => 0,
    };
    VocabParts {
        id_to_token: tokens,
        scores: Vec::new(),
        merges,
        special_ids,
        style: Style::ByteLevel,
        pretok: PreTok::Bpe(split),
        add_dummy_prefix: false,
        ignore_merges: false,
        bos_id: None,
        eos_id: eos,
        unk_id: None,
        add_bos_default: false,
    }
    .into_vocab()
    .map_err(|e| e.to_string())
}

fn corpus_dir() -> Option<PathBuf> {
    if let Some(a) = std::env::args().nth(1) {
        let p = PathBuf::from(a);
        return p.is_dir().then_some(p);
    }
    if let Ok(p) = std::env::var("GLTOK_VOCAB_DIR") {
        let p = PathBuf::from(p);
        return p.is_dir().then_some(p);
    }
    let guess = Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(3)?
        .join("llama.cpp/models");
    guess.is_dir().then_some(guess)
}

/// Audit one real model file rather than the corpus.
///
/// A refusal here is a *deployment* problem, not a coverage statistic: it means
/// a model on this machine stopped loading. Worth being able to check directly,
/// because the corpus says nothing about which GGUFs anyone actually runs.
fn audit_one(path: &Path) {
    println!("model: {}", path.display());
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            println!("  unreadable: {e}");
            return;
        }
    };
    match gguf::read_metadata(&bytes) {
        Ok(m) => {
            let d = declared(&m);
            println!(
                "  declares: model={} pre={} tokens={} merges={} scores={} add_bos={}",
                d.model,
                d.pre,
                d.tokens,
                d.merges,
                d.scores,
                d.add_bos.map(|b| b.to_string()).unwrap_or("-".into())
            );
        }
        Err(e) => {
            println!("  metadata unreadable: {e}");
            return;
        }
    }
    match gguf::vocab_from_gguf(&bytes) {
        Ok(v) => {
            let tok = Tokenizer::new(v);
            // A fixed probe string, so two runs of this tool are comparable.
            const S: &str = "Hello, World! 123 don't\n\nstop 日本語 🎉";
            match tok.encode(S, false) {
                Ok(ids) => println!(
                    "  LOADS — vocab_size={} eos={} add_bos_default={}\n  encode({S:?}) = {ids:?}",
                    tok.vocab_size(),
                    tok.eos_id(),
                    tok.add_bos_default()
                ),
                Err(e) => println!("  loads but cannot encode: {e}"),
            }
        }
        Err(e) => println!("  ⛔ REFUSED: {e}"),
    }
}

fn main() {
    // A file argument audits that model; a directory (or nothing) audits the
    // reference corpus.
    if let Some(a) = std::env::args().nth(1) {
        let p = PathBuf::from(&a);
        if p.is_file() {
            audit_one(&p);
            return;
        }
    }
    let Some(dir) = corpus_dir() else {
        eprintln!("no corpus: pass a llama.cpp models/ dir, a .gguf file, or set GLTOK_VOCAB_DIR");
        std::process::exit(1);
    };
    println!("corpus: {}\n", dir.display());

    println!(
        "{:<16} {:<8} {:<14} {:>8} {:>8} {:>7}  status",
        "vocab", "model", "pre", "tokens", "merges", "parity"
    );
    println!("{}", "-".repeat(96));

    let mut deferred: Vec<(String, String)> = Vec::new();

    for &name in ALL {
        let base = dir.join(format!("ggml-vocab-{name}.gguf"));
        let Ok(bytes) = std::fs::read(&base) else {
            println!("{name:<16} {:<8} {:<14} {:>8} {:>8} {:>7}  file absent", "-", "-", "-", "-", "-");
            continue;
        };
        let Ok(meta) = gguf::read_metadata(&bytes) else {
            println!("{name:<16} {:<8} {:<14} {:>8} {:>8} {:>7}  unreadable", "-", "-", "-", "-", "-");
            continue;
        };
        let d = declared(&meta);

        let vectors = match (
            std::fs::read_to_string(base.with_extension("gguf.inp")),
            std::fs::read_to_string(base.with_extension("gguf.out")),
        ) {
            (Ok(i), Ok(o)) => Some((parse_inp(&i), parse_out(&o))),
            _ => None,
        };

        let row = |parity: &str, status: &str| {
            println!(
                "{name:<16} {:<8} {:<14} {:>8} {:>8} {parity:>7}  {status}",
                d.model, d.pre, d.tokens, d.merges
            );
        };

        match gguf::vocab_from_gguf(&bytes) {
            Ok(v) => {
                let tok = Tokenizer::new(v);
                match &vectors {
                    Some((tests, want)) => {
                        let (p, n, first) = score(&tok, tests, want);
                        row(
                            &format!("{p}/{n}"),
                            if p == n { "SUPPORTED" } else { "LOADS, MISMATCHES" },
                        );
                        if let Some(f) = first {
                            deferred.push((name.to_string(), f));
                        }
                    }
                    None => row("-", "loads; NO REFERENCE VECTORS — unverifiable"),
                }
            }
            Err(e) => {
                // Refused. Probe every shape so the gap has a measured size.
                let mut best = String::from("-");
                let mut note = String::new();
                if let Some((tests, want)) = &vectors {
                    let mut results: Vec<(usize, usize, &str)> = Vec::new();
                    for (label, split) in PROBES {
                        if let Ok(v) = probe_vocab(&meta, *split) {
                            let (p, n, first) = score(&Tokenizer::new(v), tests, want);
                            results.push((p, n, label));
                            if p == n {
                                deferred.push((
                                    format!("{name} [probe {label}]"),
                                    format!(
                                        "reaches {p}/{n} — a real mapping is all that is missing"
                                    ),
                                ));
                            } else if let Some(f) = first {
                                deferred.push((format!("{name} [probe {label}]"), f));
                            }
                        }
                    }
                    // Report every probe, not just the winner: two shapes
                    // scoring the same is itself a finding (it means the
                    // vectors do not exercise the axis that separates them).
                    let all: Vec<String> = results
                        .iter()
                        .map(|(p, n, l)| format!("{l} {p}/{n}"))
                        .collect();
                    results.sort_by_key(|r| std::cmp::Reverse(r.0));
                    if let Some((p, n, _)) = results.first() {
                        best = format!("{p}/{n}");
                        note = format!(" [probes: {}]", all.join(", "));
                    }
                }
                row(&best, &format!("REFUSED: {e}{note}"));
            }
        }
        // Extra declarations worth seeing next to the verdict.
        if d.scores > 0 || d.add_bos.is_some() {
            println!(
                "{:<16} {:>72}",
                "",
                format!(
                    "scores={} add_bos={}",
                    d.scores,
                    d.add_bos.map(|b| b.to_string()).unwrap_or("-".into())
                )
            );
        }
    }

    if !deferred.is_empty() {
        println!("\n\nfirst divergence per open item\n{}", "=".repeat(96));
        for (who, what) in &deferred {
            println!("\n[{who}]\n  {what}");
        }
    }
}
