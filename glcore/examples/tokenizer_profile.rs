//! Where does encoding actually spend its time?
//!
//! Written because reasoning about it was wrong twice. A hasher swap looked
//! like the obvious win and measured neutral; the arithmetic that followed
//! said a *warm* cache — where every pre-token is a hash hit and nothing is
//! merged — still costs ~198 ns per ~4-byte pre-token, which no hash lookup
//! explains. So this decomposes the path instead of arguing about it.
//!
//! Everything here goes through the public API. Each stage is isolated by
//! building a `Vocab` that *cannot* do the later ones:
//!
//! | Stage | Isolated by |
//! |---|---|
//! | special-token scan | a vocabulary with the specials removed |
//! | pre-tokenizer split | calling `PreTok::split` directly |
//! | byte remap + cache + merge | the remainder |
//!
//! Run: cargo run -p glcore --example tokenizer_profile --release -- <model.gguf> [corpus]

use std::time::Instant;

use glcore::tokenizer::gguf::{self, Meta};
use glcore::tokenizer::{GllmTokenizer, Style, Vocab, VocabParts};

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
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let mut s = String::new();
    for f in CORPUS_FILES {
        if let Ok(t) = std::fs::read_to_string(root.join(f)) {
            s.push_str(&t);
            s.push('\n');
        }
    }
    s
}

const REPS: usize = 25;

fn best(mut f: impl FnMut() -> usize) -> (f64, usize) {
    let units = f();
    let mut b = f64::MAX;
    for _ in 0..REPS {
        let t = Instant::now();
        std::hint::black_box(f());
        b = b.min(t.elapsed().as_secs_f64());
    }
    (b, units)
}

/// Rebuild the vocabulary from the same GGUF metadata, optionally dropping the
/// special-token list. Everything else is identical, so the difference between
/// the two is exactly what the special scan costs.
fn vocab_from(meta: &std::collections::HashMap<String, Meta>, with_specials: bool) -> Vocab {
    let tokens = match meta.get("tokenizer.ggml.tokens") {
        Some(Meta::ArrStr(v)) => v.clone(),
        _ => panic!("no tokens"),
    };
    let merges: Vec<(String, String)> = match meta.get("tokenizer.ggml.merges") {
        Some(Meta::ArrStr(v)) => v
            .iter()
            .filter_map(|s| s.split_once(' ').map(|(a, b)| (a.to_string(), b.to_string())))
            .collect(),
        _ => Vec::new(),
    };
    let special_ids: Vec<u32> = if with_specials {
        match meta.get("tokenizer.ggml.token_type") {
            Some(Meta::ArrI32(t)) => t
                .iter()
                .enumerate()
                .filter(|(_, &ty)| ty == 3 || ty == 4)
                .map(|(i, _)| i as u32)
                .collect(),
            _ => Vec::new(),
        }
    } else {
        Vec::new()
    };
    let pre = match meta.get("tokenizer.ggml.pre") {
        Some(Meta::Str(s)) => s.clone(),
        _ => "gpt-2".into(),
    };
    VocabParts {
        id_to_token: tokens,
        scores: Vec::new(),
        merges,
        special_ids,
        style: Style::ByteLevel,
        pretok: gguf::pretok_from_name(&pre).expect("supported pre-tokenizer"),
        add_dummy_prefix: false,
        ignore_merges: false,
        bos_id: None,
        eos_id: 0,
        unk_id: None,
        add_bos_default: false,
    }
    .into_vocab()
    .expect("vocab")
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: tokenizer_profile <model.gguf> [corpus]");
        std::process::exit(1);
    };
    let text = load_corpus();
    let bytes = text.len();
    let meta = gguf::read_metadata(&std::fs::read(&path).expect("read gguf")).expect("gguf");

    let with = GllmTokenizer::new(vocab_from(&meta, true));
    let without = GllmTokenizer::new(vocab_from(&meta, false));
    let n_specials = match meta.get("tokenizer.ggml.token_type") {
        Some(Meta::ArrI32(t)) => t.iter().filter(|&&ty| ty == 3 || ty == 4).count(),
        _ => 0,
    };

    println!("corpus {} KiB · {n_specials} special tokens in this vocabulary\n", bytes / 1024);

    let pt = with.vocab().pretok();
    let (t_split, n_pretok) = best(|| {
        let mut n = 0usize;
        pt.split(&text, |_| n += 1);
        n
    });

    // Warm both caches, then time. Warm means every pre-token is a cache hit,
    // so what is left is the special scan, the split, the byte remap and the
    // lookup — no merging at all.
    GllmTokenizer::set_pretoken_cache(true);
    GllmTokenizer::reset_pretoken_cache();
    let _ = with.encode(&text, false);
    let (t_with, ntok) = best(|| with.encode(&text, false).map(|v| v.len()).unwrap_or(0));

    GllmTokenizer::reset_pretoken_cache();
    let _ = without.encode(&text, false);
    let (t_without, _) = best(|| without.encode(&text, false).map(|v| v.len()).unwrap_or(0));

    let ns = |t: f64| t * 1e9 / bytes as f64;
    let per_pre = |t: f64| t * 1e9 / n_pretok as f64;

    println!("{:<34} {:>9} {:>12} {:>10}", "stage (warm cache)", "ns/byte", "ns/pre-token", "share");
    println!("{}", "-".repeat(70));
    let scan = t_with - t_without;
    for (label, t) in [
        ("special-token scan", scan),
        ("pre-tokenizer split", t_split),
        ("byte remap + cache hit + push", t_without - t_split),
        ("TOTAL warm encode", t_with),
    ] {
        println!(
            "{label:<34} {:>9.2} {:>12.1} {:>9.1}%",
            ns(t),
            per_pre(t),
            100.0 * t / t_with
        );
    }
    println!("\n{n_pretok} pre-tokens, {ntok} tokens, {:.1} bytes/pre-token", bytes as f64 / n_pretok as f64);
    // ⚠️ Kept as a reminder of what the shape of this cost used to be. The
    // naive `find_special` ran one full substring search per special token,
    // so this line described real work; it now describes work that no longer
    // happens, which is the point.
    println!(
        "the naive scan would do {n_specials} substring searches over {} KiB =          {:.1} MiB per encode; one pass with a first-byte skip table replaced it",
        bytes / 1024,
        (n_specials * bytes) as f64 / (1024.0 * 1024.0)
    );
}
