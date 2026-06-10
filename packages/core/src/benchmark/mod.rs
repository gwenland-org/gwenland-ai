/// Benchmark module — micro-benchmarks for GwenLand runtime characteristics.
///
/// Why this module exists:
/// GwenLand claims specific performance characteristics (≤ 2.6 ms cold start,
/// sub-50 MB baseline, competitive token throughput). This module provides a
/// self-contained, reproducible way to verify those claims against the local
/// environment without requiring an external benchmarking harness.
///
/// Why not criterion?
/// Criterion is a statistically rigorous benchmarking framework designed for
/// micro-optimisation work — it runs thousands of iterations, warms up the CPU
/// cache, and applies outlier detection. For GwenLand's current stage we only
/// need to verify that the binary meets its published targets (cold start,
/// memory floor). Running criterion in CI would add ~30 s compile time and
/// require `[[bench]]` harness wiring. We use `std::time::Instant` directly for
/// now; criterion is planned once the core optimisation loop begins.
pub mod cold_start;
pub mod convert_bench;
pub mod inference;
pub mod layer_load_bench;
pub mod memory;
pub mod report;

pub use layer_load_bench::{LayerLoadResult, LayerLoadSample};

use std::time::Instant;

// ── Result types ──────────────────────────────────────────────────────────────

/// Aggregated results for the cold-start benchmark.
///
/// Reports min/max/mean/median over N iterations. Median is reported alongside
/// mean because cold-start times have a right-skewed distribution: the first run
/// after a cache miss can be 2–5× slower than subsequent runs due to OS page
/// faults on the binary. Median is more representative of steady-state start
/// time; mean exposes the tail latency that CI might see on a cold container.
#[derive(Debug, Clone)]
pub struct ColdStartResult {
    /// Fastest observed start time across all iterations.
    pub min_ms: f64,
    /// Slowest observed start time across all iterations.
    pub max_ms: f64,
    /// Arithmetic mean of all iterations.
    pub mean_ms: f64,
    /// Middle value (sorted), robust to outlier spikes.
    pub median_ms: f64,
    /// Number of iterations run.
    pub iterations: usize,
}

/// Token generation throughput result from native inference.
///
/// Uses the 4-chars/token heuristic (same as eval/metrics.rs and dry_run.rs)
/// because the native proxy stream format does not return a discrete token
/// count in the non-streaming response at the eval-relevant prompt lengths.
#[derive(Debug, Clone, serde::Serialize)]
pub struct InferenceResult {
    /// Tokens per second (response_chars / 4 / elapsed_secs).
    pub tokens_per_sec: f64,
    /// Estimated token count of the generated response.
    pub total_tokens: usize,
    /// Wall-clock seconds from request dispatch to response receipt.
    pub elapsed_secs: f64,
    /// Which inference backend produced this result. Defaults to "proxy".
    pub backend: String,
    /// Path basename of the model file used, if known.
    pub model_file: Option<String>,
}

/// Timing results for the in-process dequantisation micro-benchmark.
///
/// "ns/element" is chosen as the unit (not ms/tensor) because the interesting
/// comparison is between Standard and Euler mode per-element cost — the tensor
/// size cancels out and the ratio reveals the cosine overhead vs simple multiply.
#[derive(Debug, Clone)]
pub struct ConvertBenchResult {
    /// Mean nanoseconds per element for standard Q8_0 linear dequant.
    pub standard_ns_per_elem: f64,
    /// Mean nanoseconds per element for Euler cosine-projection dequant.
    pub euler_ns_per_elem: f64,
    /// Standard deviation across iterations for the standard mode (ns/elem).
    pub standard_stddev: f64,
    /// Standard deviation across iterations for the Euler mode (ns/elem).
    pub euler_stddev: f64,
}

/// Current process resident memory in megabytes.
#[derive(Debug, Clone)]
pub struct MemoryResult {
    /// Resident set size at the moment of sampling, in megabytes.
    /// Returns 0.0 when the platform measurement is unavailable.
    pub baseline_mb: f64,
}

/// Top-level aggregate of all four benchmark suites.
///
/// Fields are `Option` so individual suites can be skipped (e.g. when the
/// native proxy is not running, `inference` is None) without failing the whole run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Cold-start timing (always populated — uses a child process spawn).
    pub cold_start: Option<ColdStartResult>,
    /// Token generation throughput (None if native proxy is not running).
    pub inference: Option<InferenceResult>,
    /// In-process dequantisation speed (always populated).
    pub convert: Option<ConvertBenchResult>,
    /// Process resident memory at benchmark start (always populated).
    pub memory: Option<MemoryResult>,
    /// Total wall-clock seconds for the entire benchmark run.
    pub total_elapsed_secs: f64,
    /// Layer-load timing (None unless explicitly invoked via run_layer_load_bench).
    pub layer_load: Option<LayerLoadResult>,
}

// ── Report format ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum OutputFormat {
    Json,
    Text,
}

/// Render a `BenchmarkResult` as a String in the given format.
pub fn format_benchmark_report(result: &BenchmarkResult, fmt: OutputFormat) -> String {
    report::format_benchmark_report(result, fmt)
}

// ── Filter flags passed from the TUI layer ────────────────────────────────────

/// Which subset of benchmarks to run.
///
/// The TUI layer converts CLI flags (`--cold-start`, `--inference`, etc.) into
/// this struct and passes it to `run_benchmarks`. Keeping the filter here
/// (rather than in gwen-tui) means a hypothetical API caller can use the same
/// filtering logic without duplicating the flag-to-filter mapping.
#[derive(Debug, Clone, Copy, Default)]
pub struct BenchmarkFilter {
    /// Run cold-start benchmark.
    pub cold_start: bool,
    /// Run token generation benchmark.
    pub inference: bool,
    /// Run convert pipeline benchmark.
    pub convert: bool,
    /// All suites + detailed per-operation breakdown (currently same as all).
    pub full: bool,
}

impl BenchmarkFilter {
    /// Returns true when no specific filter is set — meaning "run all".
    pub fn is_all(&self) -> bool {
        !self.cold_start && !self.inference && !self.convert && !self.full
    }

    /// Should the cold-start suite run?
    pub fn run_cold_start(&self) -> bool {
        self.is_all() || self.cold_start || self.full
    }

    /// Should the inference suite run?
    pub fn run_inference(&self) -> bool {
        self.is_all() || self.inference || self.full
    }

    /// Should the convert suite run?
    pub fn run_convert(&self) -> bool {
        self.is_all() || self.convert || self.full
    }

    /// Memory baseline is always collected — it is near-zero cost and provides
    /// essential context for interpreting all other results.
    pub fn run_memory(&self) -> bool {
        true
    }
}

// ── Progress callback ─────────────────────────────────────────────────────────

/// Called when a benchmark suite begins or completes so the TUI layer can
/// print `[1/4] Cold Start ... ✓` lines in real time.
///
/// `suite_index` — 1-based index of the suite (1 = cold-start, 4 = memory).
/// `suite_name`  — human-readable name for the progress line.
/// `done`        — false when the suite is starting, true when it finishes.
pub type ProgressCallback = Box<dyn Fn(usize, usize, &str, bool)>;

// ── Orchestrator ──────────────────────────────────────────────────────────────

/// Run the requested benchmark suites and return aggregated results.
///
/// `filter`   — which suites to run (see `BenchmarkFilter`).
/// `progress` — optional progress callback for live output in the TUI layer.
///
/// All suites that encounter non-fatal errors (e.g. native proxy not running) return
/// `None` for that field rather than propagating an error — the caller prints a
/// skip warning and continues.
pub fn run_benchmarks(
    filter: BenchmarkFilter,
    progress: Option<&ProgressCallback>,
    model_path: Option<&std::path::Path>,
) -> BenchmarkResult {
    let wall_start = Instant::now();

    // Determine how many suites are enabled so the progress `[i/total]` label
    // is accurate even when only a subset of suites is selected.
    let mut suite_count: usize = 0;
    if filter.run_cold_start() { suite_count += 1; }
    if filter.run_inference()  { suite_count += 1; }
    if filter.run_convert()    { suite_count += 1; }
    // Memory is always run; include it in the count unconditionally.
    suite_count += 1;

    let mut suite_idx = 0usize;

    // ── Cold Start ────────────────────────────────────────────────────────────
    let cold_start_result = if filter.run_cold_start() {
        suite_idx += 1;
        emit_progress(&progress, suite_idx, suite_count, "Cold Start", false);
        let r = cold_start::run_cold_start();
        emit_progress(&progress, suite_idx, suite_count, "Cold Start", true);
        Some(r)
    } else {
        None
    };

    // ── Inference ─────────────────────────────────────────────────────────────
    let inference_result = if filter.run_inference() {
        suite_idx += 1;
        emit_progress(&progress, suite_idx, suite_count, "Token Generation", false);
        let r = {
            #[cfg(feature = "mistralrs-backend")]
            let result = if let Some(mp) = model_path {
                let r = inference::run_mistralrs_bench(mp);
                if r.is_some() { r } else { inference::run_inference_bench() }
            } else {
                inference::run_inference_bench()
            };
            #[cfg(not(feature = "mistralrs-backend"))]
            let result = inference::run_inference_bench();
            result
        };
        emit_progress(&progress, suite_idx, suite_count, "Token Generation", true);
        r
    } else {
        None
    };

    // ── Convert Pipeline ──────────────────────────────────────────────────────
    let convert_result = if filter.run_convert() {
        suite_idx += 1;
        emit_progress(&progress, suite_idx, suite_count, "Convert Pipeline", false);
        let r = convert_bench::run_convert_bench();
        emit_progress(&progress, suite_idx, suite_count, "Convert Pipeline", true);
        Some(r)
    } else {
        None
    };

    // ── Memory Baseline ───────────────────────────────────────────────────────
    {
        suite_idx += 1;
        emit_progress(&progress, suite_idx, suite_count, "Memory Baseline", false);
    }
    let memory_result = Some(memory::sample_memory());
    {
        emit_progress(&progress, suite_idx, suite_count, "Memory Baseline", true);
    }

    BenchmarkResult {
        cold_start: cold_start_result,
        inference: inference_result,
        convert: convert_result,
        memory: memory_result,
        total_elapsed_secs: wall_start.elapsed().as_secs_f64(),
        layer_load: None,
    }
}

// ── Helper ────────────────────────────────────────────────────────────────────

fn emit_progress(
    cb: &Option<&ProgressCallback>,
    idx: usize,
    total: usize,
    name: &str,
    done: bool,
) {
    if let Some(f) = cb {
        f(idx, total, name, done);
    }
}
