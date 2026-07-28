//! Throughput of the pre-tokenizer and of full encoding.
//!
//! Exists to answer one question: **did making the character classes exact
//! cost throughput?** The scanner was already regex-free and single-pass; what
//! `unicode_tables` changed is that `\p{L}` went from `char::is_alphabetic`
//! (one inlined property lookup) to an ASCII bitmap plus a binary search. That
//! is strictly more work per non-ASCII character, so the cost is worth a
//! number rather than an assumption.
//!
//! ⛔ **Read the noise floor before believing a number from this.** Measured
//! on the i3-1115G4 this repo is developed on, three back-to-back repeats of
//! the identical binary gave 73.1, 33.8 and 174.9 ns/byte for the same qwen2
//! case — a **5× spread**. Anything smaller than that is not a result here.
//! Reporting best-of-N below reduces but does not remove it. Machines with
//! stable clocks will do better; this one will not.
//!
//! What *is* trustworthy from this tool is the unit count, which is
//! deterministic, and the ratio to full encoding — pre-tokenization is a small
//! fraction of it, so even a real regression here moves end-to-end encoding
//! little.
//!
//! Linearity is a structural property of the scanner (single forward pass, no
//! backtracking), asserted by `pretok::tests::linear_on_pathological_input`
//! rather than inferred from these timings, which cache effects dominate long
//! before any algorithmic term would show.
//!
//! Run: cargo run -p gltokenizer --example tokbench --release -- <model.gguf>

use std::time::Instant;

use gltokenizer::{BpeSplit, PreTok, Tokenizer};

/// Mixed-script text: ASCII takes the bitmap path, the rest binary-searches.
const SEED: &str = "The quick brown fox jumps over 13 lazy dogs, don't you think? \
     日本語のテキストも含めて、混合スクリプトで測定する。 Ünïcödé àccênts, emoji 🎉🚀, \
     numbers 1234567890, punctuation …—''\"\" and code: fn main() { let x = 1 + 2; }\n\n";

fn corpus(target_bytes: usize) -> String {
    let mut s = String::with_capacity(target_bytes + SEED.len());
    while s.len() < target_bytes {
        s.push_str(SEED);
    }
    s
}

/// Best-of-N: report the fastest pass, and the spread alongside it so a reader
/// can see immediately whether the fastest number means anything.
fn bench(label: &str, text: &str, reps: usize, mut f: impl FnMut(&str) -> usize) {
    // One untimed pass so the first run's page faults are not in the sample.
    let units = f(text);
    let mut best = f64::MAX;
    let mut worst: f64 = 0.0;
    for _ in 0..reps.max(5) {
        let t = Instant::now();
        std::hint::black_box(f(text));
        let el = t.elapsed().as_secs_f64();
        best = best.min(el);
        worst = worst.max(el);
    }
    println!(
        "  {label:<28} {:>9.2} MB/s  {:>7.2} ns/byte  spread {:>4.1}x  ({units} units)",
        text.len() as f64 / best / 1e6,
        best * 1e9 / text.len() as f64,
        worst / best,
    );
}

fn main() {
    let sizes = [64 * 1024usize, 256 * 1024, 1024 * 1024];

    println!("pre-tokenizer only (no vocabulary, no merges)\n");
    for &n in &sizes {
        let text = corpus(n);
        println!("{} KiB:", text.len() / 1024);
        // Enough repetitions at every size that the timer resolution is not
        // the thing being measured.
        let reps = (8 * 1024 * 1024 / text.len()).max(3);
        for (name, sp) in [
            ("qwen2", BpeSplit::QWEN2),
            ("qwen35 (marks in words)", BpeSplit::QWEN35),
            ("llama-bpe / cl100k", BpeSplit::LLAMA3),
            ("gpt-2", BpeSplit::GPT2),
            ("falcon (3-stage pipeline)", BpeSplit::FALCON),
        ] {
            let p = PreTok::Bpe(sp);
            bench(name, &text, reps, |t| {
                let mut n = 0usize;
                p.split(t, |_| n += 1);
                n
            });
        }
        println!();
    }

    let Some(path) = std::env::args().nth(1) else {
        println!("(pass a .gguf to also measure full encoding)");
        return;
    };
    let tok = match Tokenizer::from_gguf_path(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path}: {e}");
            return;
        }
    };
    println!("full encode — {path}\n");
    for &n in &sizes {
        let text = corpus(n);
        let reps = (2 * 1024 * 1024 / text.len()).max(3);
        bench(
            &format!("{} KiB", text.len() / 1024),
            &text,
            reps,
            |t| tok.encode(t, false).map(|v| v.len()).unwrap_or(0),
        );
    }
}
