// ─────────────────────────────────────────────────────────────────────────────
// GwenLand — Truth Benchmark
// Criterion 0.5 bench covering every measurable hot-path in gwen-core.
//
// HOW TO RUN:
//   Copy this file to:
//     gwen-cli/packages/core/benches/core_bench.rs
//   Then:
//     cd gwen-cli
//     cargo bench -p gwenland-core 2>&1 | tee bench_results.txt
//
// HTML report: gwen-cli/target/criterion/report/index.html
// ─────────────────────────────────────────────────────────────────────────────

use criterion::{
    black_box, criterion_group, criterion_main,
    BenchmarkId, Criterion, Throughput,
};

use gwenland_core::engine::tokenizer::{
    detect_budget_from_model, estimate_context, estimate_tokens, ModelBudget,
};
use gwenland_core::engine::windowing::{
    extract_query_terms, extract_relevant_windows, WindowConfig,
};
use gwenland_core::diagnostics::estimator::{estimate_time, estimate_vram, Device};
use gwenland_core::platform::scanner::FileEntry;

// ─── 1. estimate_tokens ──────────────────────────────────────────────────────
// Expected: O(1) — single integer division, no allocation.
// Target:   < 2 ns at all input sizes.

fn bench_estimate_tokens(c: &mut Criterion) {
    let inputs: &[(&str, &str)] = &[
        ("11chars",  "hello world"),
        ("1k_chars", &"a".repeat(1_000)),
        ("100k_chars", &"a".repeat(100_000)),
    ];

    let mut group = c.benchmark_group("estimate_tokens");
    for (label, input) in inputs {
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("len", label),
            input,
            |b, i| b.iter(|| estimate_tokens(black_box(i))),
        );
    }
    group.finish();
}

// ─── 2. detect_budget_from_model ─────────────────────────────────────────────
// Expected: O(n patterns) — string contains() scan over model name.
// Target:   < 600 ns worst-case (unknown model, falls through all arms).

fn bench_detect_budget(c: &mut Criterion) {
    let cases: &[(&str, &str)] = &[
        ("small_7b",      "llama-3-8b-instruct"),
        ("medium_13b",    "codellama-13b-instruct"),
        ("large_mistral", "mistral-large-instruct-2407"),
        ("unknown",       "some-totally-unknown-model-xyz-v2"),
    ];

    let mut group = c.benchmark_group("detect_budget_from_model");
    for (label, model) in cases {
        group.bench_with_input(
            BenchmarkId::new("model", label),
            model,
            |b, m| b.iter(|| detect_budget_from_model(black_box(m))),
        );
    }
    group.finish();
}

// ─── 3. estimate_context ─────────────────────────────────────────────────────
// Expected: O(n files) — iterates over FileEntry slice, reads disk for real
//           paths (these are nonexistent so it falls back to size_bytes / 4).
// Target:   < 3 ms for 50 files on consumer hardware.

fn bench_estimate_context(c: &mut Criterion) {
    let make_files = |n: usize| -> Vec<FileEntry> {
        (0..n).map(|i| FileEntry {
            path:       format!("nonexistent_{i}.rs"),
            size_bytes: 4_000, // 4 KB each → 1000 tokens each
            extension:  Some("rs".to_string()),
            is_binary:  false,
        }).collect()
    };

    let prompt = "Refactor this codebase to use async/await throughout";

    let mut group = c.benchmark_group("estimate_context");
    for n in [10, 50, 100] {
        let files = make_files(n);
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new("files", n),
            &files,
            |b, f| b.iter(|| estimate_context(black_box(f), black_box(prompt), ModelBudget::Large)),
        );
    }
    group.finish();
}

// ─── 4. extract_query_terms ──────────────────────────────────────────────────
// Expected: O(n words) — split + stopword filter + dedup.
// Target:   < 5 µs for a realistic natural-language query.

fn bench_extract_query_terms(c: &mut Criterion) {
    let queries: &[(&str, &str)] = &[
        ("short",  "authenticate error"),
        ("medium", "why is the authentication function returning a 403 error in production?"),
        ("long",   "how do i refactor the async token refresh logic to use the new context window \
                    manager and make sure it integrates with the existing error handling pipeline?"),
    ];

    let mut group = c.benchmark_group("extract_query_terms");
    for (label, query) in queries {
        group.bench_with_input(
            BenchmarkId::new("query", label),
            query,
            |b, q| b.iter(|| extract_query_terms(black_box(q))),
        );
    }
    group.finish();
}

// ─── 5. extract_relevant_windows ─────────────────────────────────────────────
// Expected: O(lines × terms) — TF scoring pass + boundary expansion + merge.
// Target:   < 500 µs for a 500-line file with a 5-term query.
// This is the most expensive hot-path per file in the context engine.

fn bench_extract_relevant_windows(c: &mut Criterion) {
    let enabled = WindowConfig {
        enabled:      true,
        token_budget: 4096,
        window_size:  20,
        max_windows:  5,
    };

    // Realistic Rust source file simulation
    let make_file = |lines: usize| -> String {
        let snippets = [
            "fn authenticate(user: &str, token: &str) -> Result<(), AuthError> {",
            "    let hash = compute_hash(token);",
            "    if hash != expected_hash(user) { return Err(AuthError::Invalid); }",
            "    Ok(())",
            "}",
            "pub struct TokenManager { budget: usize, tokens: Vec<String> }",
            "impl TokenManager {",
            "    pub fn new(budget: usize) -> Self { Self { budget, tokens: vec![] } }",
            "    pub fn push(&mut self, t: String) { self.tokens.push(t); }",
            "}",
            "// helper: validates input before processing",
            "fn validate_input(s: &str) -> bool { !s.is_empty() && s.len() < 1024 }",
            "fn process_request(req: Request) -> Response { Response::ok() }",
            "async fn fetch_model(name: &str) -> Result<Model, Error> {",
            "    let path = resolve_path(name)?;",
            "    Model::load(&path).await",
            "}",
        ];
        (0..lines)
            .map(|i| snippets[i % snippets.len()].to_string())
            .collect::<Vec<_>>()
            .join("\n")
    };

    let query = "authenticate token error";

    let mut group = c.benchmark_group("extract_relevant_windows");
    for lines in [100, 500, 1000] {
        let content = make_file(lines);
        group.throughput(Throughput::Bytes(content.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("lines", lines),
            &content,
            |b, fc| b.iter(|| extract_relevant_windows(black_box(fc), black_box(query), black_box(&enabled))),
        );
    }
    group.finish();
}

// ─── 6. estimate_vram ────────────────────────────────────────────────────────
// Expected: O(1) — pure arithmetic, no allocations.
// Target:   < 50 ns.

fn bench_estimate_vram(c: &mut Criterion) {
    let cases: &[(&str, usize, usize, usize, usize, usize, usize)] = &[
        // (label, total_params, lora_params, batch, seq, d_model, layers)
        ("7b_lora_r8",  7_000_000_000, 2_097_152,  1, 1024, 4096, 32),
        ("13b_lora_r16",13_000_000_000, 8_388_608, 1, 1024, 5120, 40),
        ("70b_lora_r8", 70_000_000_000, 4_194_304, 1, 2048, 8192, 80),
    ];

    let mut group = c.benchmark_group("estimate_vram");
    for (label, total, lora, batch, seq, d, layers) in cases {
        group.bench_with_input(
            BenchmarkId::new("model", label),
            &(*total, *lora, *batch, *seq, *d, *layers),
            |b, &(t, l, bs, sq, dm, nl)| {
                b.iter(|| estimate_vram(
                    black_box(t), black_box(l),
                    black_box(bs), black_box(sq),
                    black_box(dm), black_box(nl),
                ))
            },
        );
    }
    group.finish();
}

// ─── 7. estimate_time ────────────────────────────────────────────────────────
// Expected: O(1) — floating-point division.
// Target:   < 20 ns.

fn bench_estimate_time(c: &mut Criterion) {
    let cases: &[(&str, usize, usize, f32)] = &[
        // (label, total_tokens, epochs, device_tflops)
        ("1b_tokens_cpu",   1_000_000_000, 1, Device::Cpu.tflops()),
        ("1b_tokens_t4",    1_000_000_000, 3, Device::T4.tflops()),
        ("1b_tokens_a100",  1_000_000_000, 3, Device::A100.tflops()),
        ("1b_tokens_4090",  1_000_000_000, 3, Device::Rtx4090.tflops()),
    ];

    let mut group = c.benchmark_group("estimate_time");
    for (label, tokens, epochs, tflops) in cases {
        group.bench_with_input(
            BenchmarkId::new("device", label),
            &(*tokens, *epochs, *tflops),
            |b, &(t, e, tf)| {
                b.iter(|| estimate_time(black_box(t), black_box(e), black_box(tf)))
            },
        );
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────

criterion_group!(
    benches,
    bench_estimate_tokens,
    bench_detect_budget,
    bench_estimate_context,
    bench_extract_query_terms,
    bench_extract_relevant_windows,
    bench_estimate_vram,
    bench_estimate_time,
);
criterion_main!(benches);
