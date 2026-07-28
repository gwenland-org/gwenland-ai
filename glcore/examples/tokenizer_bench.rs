//! Throughput of the pre-tokenizer and of full encoding.
//!
//! # ⛔ The corpus decides whether this tool lies
//!
//! An earlier version of this file built its input by repeating one sentence
//! until it reached the target size. That is tolerable for the pre-tokenizer,
//! whose cost per byte does not depend on what came before — and
//! **catastrophic** for measuring the pre-token cache, which would score a
//! ~100 % hit rate on text made of one repeated sentence and report a speedup
//! no real workload can reproduce.
//!
//! So the default corpus is **real files from this repository**: prose from the
//! audit note and Rust source from the tokenizer itself. Pass a path to use
//! your own. Neither substitutes for measuring your actual workload — the hit
//! rate *is* the result, and it is a property of your text, not of this code.
//!
//! # ⛔ And the machine decides whether you can trust the number
//!
//! On the i3-1115G4 this repo is developed on, best-of-5 gave a **5× spread**
//! across identical runs. Best-of-40 brings the same case to within a few
//! percent. Noise only ever *adds* time, so the minimum over many passes is the
//! closest thing to the true cost this hardware can report — but if the
//! `spread` column is not near 1.0, the number beside it means nothing.
//!
//! ⚠️ Build profile is part of the measurement: `glcore` carries
//! `[profile.release.package.glcore] opt-level = 3`. Measured under the
//! workspace default `opt-level = "z"` the same scanner runs **3.4× slower**.
//!
//! Linearity is a structural property of the scanner (single forward pass, no
//! backtracking), asserted by `pretok::tests::linear_on_pathological_input`
//! rather than inferred from these timings.
//!
//! Run: cargo run -p glcore --example tokenizer_bench --release -- <model.gguf> [corpus]

use std::time::Instant;

use glcore::tokenizer::{BpeSplit, GllmTokenizer, PreTok};

/// Files that exist in every checkout, chosen for *shape*: English prose with
/// tables and punctuation, plus dense Rust with identifiers and symbols. Real
/// long-tailed word distributions, which is the whole point.
const CORPUS_FILES: &[&str] = &[
    "notes/gltokenizer-gguf-support-audit.md",
    "glcore/src/tokenizer/mod.rs",
    "glcore/src/tokenizer/pretok.rs",
    "glcore/src/tokenizer/vocab.rs",
    "glcore/src/tokenizer/gguf.rs",
];

fn load_corpus() -> String {
    if let Some(p) = std::env::args().nth(2) {
        return std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p}: {e}"));
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate has a parent directory");
    let mut s = String::new();
    for f in CORPUS_FILES {
        if let Ok(t) = std::fs::read_to_string(root.join(f)) {
            s.push_str(&t);
            s.push('\n');
        }
    }
    assert!(
        s.len() > 32 * 1024,
        "corpus is only {} bytes — run from the repo, or pass a file",
        s.len()
    );
    s
}

const REPS: usize = 40;

/// Best-of-N, reporting the spread so the reader can see whether to believe it.
fn bench(label: &str, bytes: usize, mut f: impl FnMut() -> usize) {
    let units = f();
    let (mut best, mut worst) = (f64::MAX, 0.0f64);
    for _ in 0..REPS {
        let t = Instant::now();
        std::hint::black_box(f());
        let el = t.elapsed().as_secs_f64();
        best = best.min(el);
        worst = worst.max(el);
    }
    println!(
        "  {label:<26} {:>8.2} ns/byte  {:>8.2} MB/s  spread {:>4.1}x  ({units} units)",
        best * 1e9 / bytes as f64,
        bytes as f64 / best / 1e6,
        worst / best,
    );
}

fn main() {
    let text = load_corpus();
    println!(
        "corpus: {} KiB of real repo prose + Rust source\n",
        text.len() / 1024
    );

    println!("pre-tokenizer only (no vocabulary, no merges)");
    for (name, sp) in [
        ("gpt-2", BpeSplit::GPT2),
        ("starcoder", BpeSplit::STARCODER),
        ("llama-bpe / cl100k", BpeSplit::LLAMA3),
        ("qwen2", BpeSplit::QWEN2),
        ("qwen35 (marks in words)", BpeSplit::QWEN35),
        ("falcon (3-stage pipeline)", BpeSplit::FALCON),
    ] {
        let p = PreTok::Bpe(sp);
        bench(name, text.len(), || {
            let mut n = 0usize;
            p.split(&text, |_| n += 1);
            n
        });
    }

    let Some(path) = std::env::args().nth(1) else {
        println!("\n(pass a .gguf to also measure full encoding)");
        return;
    };
    let tok = match GllmTokenizer::from_gguf_path(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };

    println!("\nfull encode — {path}");
    // ⭐ In-process A/B. Two separate binaries would differ in build layout and
    // in thermal state; the same process, seconds apart, differs in neither.
    GllmTokenizer::set_pretoken_cache(false);
    GllmTokenizer::reset_pretoken_cache();
    bench("cache OFF", text.len(), || {
        tok.encode(&text, false).map(|v| v.len()).unwrap_or(0)
    });

    // ⛔ COLD is the number to quote. `bench` runs the same input 40 times, so
    // a cache left warm is being asked "how fast is re-encoding a document you
    // have already seen" — which no server does. Clearing before every pass
    // measures the realistic case: a fresh document against a fresh cache,
    // where the only hits are *within* that document.
    GllmTokenizer::set_pretoken_cache(true);
    bench("cache ON, cold each pass", text.len(), || {
        GllmTokenizer::reset_pretoken_cache();
        GllmTokenizer::set_pretoken_cache(true);
        tok.encode(&text, false).map(|v| v.len()).unwrap_or(0)
    });
    let (h, m) = GllmTokenizer::pretoken_cache_stats();
    println!(
        "  {:<26} hit rate {:.1}%  — {m} distinct pre-tokens in {} total",
        "",
        100.0 * h as f64 / (h + m).max(1) as f64,
        h + m
    );

    // WARM is the upper bound: a long-lived process whose traffic repeats.
    // Real serving sits between the two, nearer cold for varied prompts.
    GllmTokenizer::reset_pretoken_cache();
    GllmTokenizer::set_pretoken_cache(true);
    let _ = tok.encode(&text, false);
    bench("cache ON, warm (upper bound)", text.len(), || {
        tok.encode(&text, false).map(|v| v.len()).unwrap_or(0)
    });

    // The claim that matters more than any timing above.
    GllmTokenizer::set_pretoken_cache(false);
    let cold = tok.encode(&text, false).expect("encode");
    GllmTokenizer::set_pretoken_cache(true);
    GllmTokenizer::reset_pretoken_cache();
    let warm = tok.encode(&text, false).expect("encode");
    assert_eq!(cold, warm, "the cache changed the ids — this is a bug");
    println!("\ncache produces byte-identical ids: {} tokens", warm.len());
}
