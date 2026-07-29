//! KV cache throughput benchmark — the number Gate A5 never measured.
//!
//! `tests/wave_a5_kv_cache.rs` proved `CachedSession` is *correct*
//! (token-for-token identical to the recomputation oracle, on a real PJRT
//! plugin — CI run `30447306245`). It never measured *speed*. `gate_a5.rs`
//! measures speed, but only for `Session` — the uncached path — because it
//! predates `CachedSession` entirely.
//!
//! This is the missing half: the same prompt, the same lengths, run through
//! `CachedSession::generate` (prefill once, then O(1)-per-token decode), with
//! tok/s compared against `Session`'s already-measured numbers from that same
//! CI run.
//!
//! ```bash
//! PJRT_PLUGIN_CPU=~/pjrt/libpjrt_cpu.so \
//! QWEN2_HF_DIR=~/qwen2-0.5b \
//! cargo run -p gljax --release --example bench_kv_cache
//!
//! KV_BENCH_LENGTHS=8,32,128 cargo run -p gljax --release --example bench_kv_cache
//! ```
//!
//! # ⚠️ The speedup column is not a live A/B
//!
//! Re-running `Session` here to get a same-run baseline would re-pay its
//! O(n·S) cost — the 512-token/bucket-1024 row alone measured **3205s** in
//! run `30447306245`. Rather than spend that again, `BASELINE_TOK_S` below is
//! that run's own numbers, copied verbatim. This is a real risk: if the
//! runner's throughput drifts between runs, the ratio is wrong even though
//! both individual numbers are right. gljax's own history says this
//! particular workload has been reproducible across runs on this runner class
//! (the Gate A5 sweep in run `30447306245` reproduced its predecessor's
//! figures exactly), which is why this trade is taken here — but a wrong
//! speedup number from stale drift is exactly the kind of thing that looks
//! authoritative and is not, so treat it as an estimate, not a certificate.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use gljax::pjrt::{cpu_plugin_path, PjrtPlugin};
use gljax::runtime::bucket::{bucket_for, BUCKETS};
use gljax::runtime::{CachedSession, HfCheckpoint};
use gljax::GlError;

const PROMPT: &str = "The capital of France is";

/// Unlike Gate A5, decode here is meant to be O(1) per token, so there is no
/// reason to hold back the long lengths by default.
const DEFAULT_LENGTHS: &[usize] = &[8, 16, 32, 64, 128, 256, 512];

const ON_TOPIC: &[&str] = &["paris", "france", "french"];

/// `Session` (uncached) tok/s, measured in CI run `30447306245` — see this
/// file's module docs for why these are copied rather than re-measured.
const BASELINE_TOK_S: &[(usize, f64)] = &[
    (8, 1.18),
    (16, 1.28),
    (32, 1.28),
    (64, 1.28),
    (128, 0.68),
    (256, 0.34),
    (512, 0.16),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger_init();

    let Some(plugin_path) = cpu_plugin_path() else {
        eprintln!("SKIP bench_kv_cache: no PJRT plugin configured (set PJRT_PLUGIN_CPU)");
        return Ok(());
    };
    let Ok(model_dir) = std::env::var("QWEN2_HF_DIR") else {
        eprintln!("SKIP bench_kv_cache: QWEN2_HF_DIR not set");
        return Ok(());
    };
    let model_dir = PathBuf::from(model_dir);

    let lengths = lengths_from_env()?;

    println!("gljax KV cache throughput benchmark");
    println!("plugin : {}", plugin_path.display());
    println!("model  : {}", model_dir.display());
    println!("prompt : {PROMPT:?}");
    println!("lengths: {lengths:?}");

    let plugin = Rc::new(PjrtPlugin::load(&plugin_path)?);
    let (major, minor) = plugin.api_version();
    println!("PJRT API {major}.{minor}\n");

    let checkpoint = HfCheckpoint::open(&model_dir)?;
    let cfg = &checkpoint.config;
    println!(
        "config : {} layers, hidden {}, {} q-heads / {} kv-heads, vocab {}, rope_base {}",
        cfg.n_layers, cfg.hidden, cfg.n_heads, cfg.n_kv_heads, cfg.vocab, cfg.rope_base
    );

    let prompt_ids = checkpoint.encode(PROMPT)?;
    println!("prompt encodes to {} tokens: {prompt_ids:?}\n", prompt_ids.len());
    let prompt_len = prompt_ids.len();

    // Same grouping Gate A5 uses: one compiled window (prefill bucket + cache
    // capacity, see trace_prefill_with_cache's docs) serves every length that
    // fits in it, so a 24-layer model is not compiled twice for nothing.
    let mut plan: Vec<(usize, Vec<usize>)> = Vec::new();
    for &n in &lengths {
        let window = bucket_for(prompt_len + n, &BUCKETS).ok_or_else(|| {
            GlError::Engine(format!(
                "{prompt_len} prompt + {n} new tokens exceeds every bucket {BUCKETS:?}"
            ))
        })?;
        match plan.iter_mut().find(|(w, _)| *w == window) {
            Some((_, ns)) => ns.push(n),
            None => plan.push((window, vec![n])),
        }
    }
    println!(
        "{} window(s) to compile (prefill + decode each): {:?}\n",
        plan.len(),
        plan.iter().map(|(w, _)| *w).collect::<Vec<_>>()
    );

    let mut rows: Vec<Row> = Vec::new();
    let mut all_passed = true;

    for (window, ns) in &plan {
        println!("── window {window} ──────────────────────────────────────");
        let t0 = Instant::now();
        let mut session = CachedSession::from_hf_dir(Rc::clone(&plugin), &model_dir, *window)?;
        println!(
            "compile (prefill + decode) + weight upload: {:.1}s",
            t0.elapsed().as_secs_f64()
        );

        for &n in ns {
            let t0 = Instant::now();
            let output = session.generate_text(PROMPT, n)?;
            let elapsed = t0.elapsed().as_secs_f64();
            let tok_s = n as f64 / elapsed;

            let verdict = check_coherent(&output, n);
            if verdict.is_err() {
                all_passed = false;
            }
            let baseline = BASELINE_TOK_S.iter().find(|(bn, _)| *bn == n).map(|(_, t)| *t);
            rows.push(Row {
                tokens: n,
                window: *window,
                wall_s: elapsed,
                tok_s,
                baseline_tok_s: baseline,
                output,
                verdict,
            });
            let r = rows.last().expect("just pushed");
            let speedup = r
                .baseline_tok_s
                .map(|b| format!("{:.0}x", r.tok_s / b))
                .unwrap_or_else(|| "?".to_string());
            println!(
                "  {:>4} tok  {:>8.2}s  {:>7.2} tok/s  ({speedup} vs Session)  {}",
                r.tokens,
                r.wall_s,
                r.tok_s,
                match &r.verdict {
                    Ok(()) => "✅".to_string(),
                    Err(why) => format!("❌ {why}"),
                }
            );
            println!("       {:?}", truncate(&r.output, 90));
        }
        println!();
    }

    print_table(&rows);

    if all_passed && !rows.is_empty() {
        println!(
            "\nKV cache throughput bench: PASSED ✅ — {} length(s) coherent",
            rows.len()
        );
        Ok(())
    } else if rows.is_empty() {
        eprintln!("\nKV cache throughput bench: INCONCLUSIVE — nothing ran");
        std::process::exit(1);
    } else {
        eprintln!("\nKV cache throughput bench: FAILED ❌ — see the table above");
        std::process::exit(1);
    }
}

struct Row {
    tokens: usize,
    window: usize,
    wall_s: f64,
    tok_s: f64,
    baseline_tok_s: Option<f64>,
    output: String,
    verdict: Result<(), String>,
}

fn print_table(rows: &[Row]) {
    let rule = "─".repeat(96);
    println!("{rule}");
    println!(
        "{:>6} {:>7} {:>9} {:>10} {:>11} {:>8}  {:<8} first 40 chars",
        "tokens", "window", "wall_s", "tok/s", "Session tok/s", "speedup", "verdict"
    );
    println!("{rule}");
    for r in rows {
        let (baseline_str, speedup_str) = match r.baseline_tok_s {
            Some(b) => (format!("{b:.2}"), format!("{:.0}x", r.tok_s / b)),
            None => ("?".to_string(), "?".to_string()),
        };
        println!(
            "{:>6} {:>7} {:>9.2} {:>10.2} {:>11} {:>8}  {:<8} {:?}",
            r.tokens,
            r.window,
            r.wall_s,
            r.tok_s,
            baseline_str,
            speedup_str,
            if r.verdict.is_ok() { "PASS" } else { "FAIL" },
            truncate(&r.output, 40)
        );
    }
    println!("{rule}");
    println!(
        "Session tok/s column is CI run 30447306245's measured baseline (not re-measured this \
         run) — see this file's module docs."
    );
}

/// Same content check `gate_a5.rs` uses — a numerics bug here looks
/// identical to one there: fluent, on-topic-or-not text, valid UTF-8.
fn check_coherent(output: &str, max_new_tokens: usize) -> Result<(), String> {
    if output.is_empty() {
        return Err("empty output".into());
    }
    if output.contains('\u{FFFD}') {
        return Err("contains U+FFFD — the decoder is producing invalid bytes".into());
    }
    if !output.chars().any(|c| c.is_alphanumeric()) {
        return Err("no alphanumeric characters at all".into());
    }
    if let Some(unit) = repeated_tail(output) {
        return Err(format!("repetition loop on {unit:?}"));
    }
    if max_new_tokens >= 8 {
        let lower = output.to_lowercase();
        if !ON_TOPIC.iter().any(|w| lower.contains(w)) {
            return Err(format!(
                "never mentions any of {ON_TOPIC:?} — the model is fluent but not \
                 answering the prompt, which is what a numerics bug looks like"
            ));
        }
    }
    Ok(())
}

/// The shortest unit (1-20 chars) that the tail repeats at least 4 times.
fn repeated_tail(s: &str) -> Option<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() < 16 {
        return None;
    }
    for unit in 1..=20usize.min(chars.len() / 4) {
        let tail = &chars[chars.len() - unit * 4..];
        let first = &tail[..unit];
        if tail.chunks(unit).all(|c| c == first) && first.iter().any(|c| !c.is_whitespace()) {
            return Some(first.iter().collect());
        }
    }
    None
}

/// Char-boundary-safe truncation — model output routinely has multi-byte
/// characters, and `&s[..n]` panics mid-character.
fn truncate(s: &str, n: usize) -> String {
    let cleaned: String = s.chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    if cleaned.chars().count() <= n {
        return cleaned;
    }
    cleaned.chars().take(n).collect::<String>() + "…"
}

fn lengths_from_env() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    match std::env::var("KV_BENCH_LENGTHS") {
        Err(_) => Ok(DEFAULT_LENGTHS.to_vec()),
        Ok(raw) => {
            let mut out = Vec::new();
            for part in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                out.push(
                    part.parse::<usize>()
                        .map_err(|_| format!("KV_BENCH_LENGTHS: {part:?} is not a number"))?,
                );
            }
            if out.is_empty() {
                return Err("KV_BENCH_LENGTHS is empty".into());
            }
            Ok(out)
        }
    }
}

/// Minimal `RUST_LOG` handling — gljax has no `env_logger` dependency.
fn env_logger_init() {
    struct Stderr(log::LevelFilter);
    impl log::Log for Stderr {
        fn enabled(&self, m: &log::Metadata) -> bool {
            m.level() <= self.0
        }
        fn log(&self, r: &log::Record) {
            if self.enabled(r.metadata()) {
                eprintln!("[{}] {}", r.level(), r.args());
            }
        }
        fn flush(&self) {}
    }
    let level = match std::env::var("RUST_LOG").as_deref() {
        Ok("trace") => log::LevelFilter::Trace,
        Ok("debug") => log::LevelFilter::Debug,
        Ok("warn") => log::LevelFilter::Warn,
        Ok("error") => log::LevelFilter::Error,
        Ok("off") => log::LevelFilter::Off,
        _ => log::LevelFilter::Info,
    };
    let logger: &'static Stderr = Box::leak(Box::new(Stderr(level)));
    let _ = log::set_logger(logger).map(|()| log::set_max_level(level));
}
