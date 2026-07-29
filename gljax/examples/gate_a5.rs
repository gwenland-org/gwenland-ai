//! Gate A5 — coherence benchmark across output lengths.
//!
//! Runs gljax on one fixed prompt at several output lengths and checks the
//! output is real language, not garbage bytes, a repetition loop, or a
//! truncated fragment.
//!
//! ```bash
//! PJRT_PLUGIN_CPU=~/pjrt/libpjrt_cpu.so \
//! QWEN2_HF_DIR=~/qwen2-0.5b \
//! cargo run -p gljax --release --example gate_a5
//!
//! GATE_A5_LENGTHS=8,16,32 cargo run -p gljax --release --example gate_a5
//! ```
//!
//! # ⚠️ Why the default stops at 128
//!
//! `Session::generate` has **no KV cache** (ARTX03 §4's decision, kept): every
//! decode step re-runs the whole padded sequence. So generating `n` tokens
//! costs `n` full forward passes at the bucket width, and the bucket grows
//! with `n` — the cost is superlinear, not linear.
//!
//! Qwen2-0.5B is 494 M MACs per position, so one forward is:
//!
//! | bucket | GFLOP / forward |
//! |---|---|
//! | 128 | 128 |
//! | 256 | 259 |
//! | 512 | 528 |
//! | 1024 | 1102 |
//!
//! On a 4-vCPU runner at an optimistic 60 GFLOP/s, 512 tokens (bucket 1024)
//! is **2.6 hours**; at a realistic 30 GFLOP/s it is 5.2 hours, and the whole
//! 7-length sweep is 6.9 hours against a **6-hour** GitHub Actions job limit.
//!
//! The lengths ≤ 128 are minutes. Those are the default; 256 and 512 are
//! opt-in via `GATE_A5_LENGTHS`. This is the brief's own Risk 4 fallback, and
//! the fix is a KV cache, not a bigger runner.

use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use gljax::pjrt::{cpu_plugin_path, PjrtPlugin};
use gljax::runtime::bucket::{bucket_for, BUCKETS};
use gljax::runtime::{HfCheckpoint, Session};
use gljax::GlError;

const PROMPT: &str = "The capital of France is";

/// CI-safe by default. Override with `GATE_A5_LENGTHS=8,16,...`.
const DEFAULT_LENGTHS: &[usize] = &[8, 16, 32, 64, 128];

/// Words that make a completion of this prompt recognizably on-topic.
const ON_TOPIC: &[&str] = &["paris", "france", "french"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger_init();

    let Some(plugin_path) = cpu_plugin_path() else {
        eprintln!("SKIP gate_a5: no PJRT plugin configured (set PJRT_PLUGIN_CPU)");
        return Ok(());
    };
    let Ok(model_dir) = std::env::var("QWEN2_HF_DIR") else {
        eprintln!("SKIP gate_a5: QWEN2_HF_DIR not set");
        return Ok(());
    };
    let model_dir = PathBuf::from(model_dir);

    let lengths = lengths_from_env()?;

    println!("gljax Gate A5 — coherence benchmark");
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

    // Group by bucket: 8/16/32/64 all share bucket 128, and an XLA compile of
    // a 24-layer model is not something to repeat four times for nothing.
    let mut plan: Vec<(usize, Vec<usize>)> = Vec::new();
    for &n in &lengths {
        let bucket = bucket_for(prompt_len + n, &BUCKETS).ok_or_else(|| {
            GlError::Engine(format!(
                "{prompt_len} prompt + {n} new tokens exceeds every bucket {BUCKETS:?}"
            ))
        })?;
        match plan.iter_mut().find(|(b, _)| *b == bucket) {
            Some((_, ns)) => ns.push(n),
            None => plan.push((bucket, vec![n])),
        }
    }
    println!(
        "{} bucket(s) to compile: {:?}\n",
        plan.len(),
        plan.iter().map(|(b, _)| *b).collect::<Vec<_>>()
    );

    let mut rows: Vec<Row> = Vec::new();
    let mut all_passed = true;

    for (bucket, ns) in &plan {
        println!("── bucket {bucket} ──────────────────────────────────────");
        let t0 = Instant::now();
        let session = Session::from_hf_dir(Rc::clone(&plugin), &model_dir, *bucket, None)?;
        println!("compile + weight upload: {:.1}s", t0.elapsed().as_secs_f64());

        for &n in ns {
            let t0 = Instant::now();
            let output = session.generate_text(PROMPT, n)?;
            let elapsed = t0.elapsed().as_secs_f64();

            let verdict = check_coherent(&output, n);
            if verdict.is_err() {
                all_passed = false;
            }
            rows.push(Row {
                tokens: n,
                bucket: *bucket,
                wall_s: elapsed,
                tok_s: n as f64 / elapsed,
                output,
                verdict,
            });
            let r = rows.last().expect("just pushed");
            println!(
                "  {:>4} tok  {:>8.1}s  {:>6.2} tok/s  {}",
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
        println!("\nGate A5: PASSED ✅ — {} length(s) coherent", rows.len());
        Ok(())
    } else if rows.is_empty() {
        eprintln!("\nGate A5: INCONCLUSIVE — nothing ran");
        std::process::exit(1);
    } else {
        eprintln!("\nGate A5: FAILED ❌ — see the table above");
        std::process::exit(1);
    }
}

struct Row {
    tokens: usize,
    bucket: usize,
    wall_s: f64,
    tok_s: f64,
    output: String,
    verdict: Result<(), String>,
}

fn print_table(rows: &[Row]) {
    let rule = "─".repeat(78);
    println!("{rule}");
    println!(
        "{:>6} {:>7} {:>9} {:>9}  {:<8} first 40 chars",
        "tokens", "bucket", "wall_s", "tok/s", "verdict"
    );
    println!("{rule}");
    for r in rows {
        println!(
            "{:>6} {:>7} {:>9.1} {:>9.2}  {:<8} {:?}",
            r.tokens,
            r.bucket,
            r.wall_s,
            r.tok_s,
            if r.verdict.is_ok() { "PASS" } else { "FAIL" },
            truncate(&r.output, 40)
        );
    }
    println!("{rule}");
}

/// Is this real language, and is it about the prompt?
///
/// ⚠️ Deliberately checks *content*, not just bytes. Every failure mode this
/// gate exists to catch — wrong RoPE base, transposed weight, off-by-one in
/// the sampled row — produces output that is still valid UTF-8 and still
/// looks like words. "Non-empty and decodes" would pass all of them.
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

    // A repetition loop is the classic signature of a broken position encoding:
    // fluent tokens, no progression.
    if let Some(unit) = repeated_tail(output) {
        return Err(format!("repetition loop on {unit:?}"));
    }

    // Content. Only meaningful once there are enough tokens to say anything.
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

/// The shortest unit (1–20 chars) that the tail repeats at least 4 times.
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

/// Char-boundary-safe truncation. `&s[..n]` panics mid-character, and this
/// runs on model output where multi-byte characters are routine.
fn truncate(s: &str, n: usize) -> String {
    let cleaned: String = s.chars().map(|c| if c == '\n' { '⏎' } else { c }).collect();
    if cleaned.chars().count() <= n {
        return cleaned;
    }
    cleaned.chars().take(n).collect::<String>() + "…"
}

fn lengths_from_env() -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    match std::env::var("GATE_A5_LENGTHS") {
        Err(_) => Ok(DEFAULT_LENGTHS.to_vec()),
        Ok(raw) => {
            let mut out = Vec::new();
            for part in raw.split(',').map(str::trim).filter(|p| !p.is_empty()) {
                out.push(part.parse::<usize>().map_err(|_| {
                    format!("GATE_A5_LENGTHS: {part:?} is not a number")
                })?);
            }
            if out.is_empty() {
                return Err("GATE_A5_LENGTHS is empty".into());
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
