/// `gwen benchmark` — CLI args, live progress output, and final report.
///
/// All measurement logic lives in gwen-core. This file owns only:
///   - CLI arg definitions (clap)
///   - Live `[i/N] Suite ... ✓` progress printing
///   - Final report rendering (the spec-prescribed format)
///   - Proxy-skip warning (when the native inference proxy is not running)
///   - Error handling and exit codes
///
/// This keeps the crate boundary clean: gwen-tui never touches timing,
/// process spawning, or memory sampling directly.
use std::path::PathBuf;

use clap::Args;
use gwenland_core::benchmark::{
    layer_load_bench::run_layer_load_bench, run_benchmarks, BenchmarkFilter, BenchmarkResult,
    ProgressCallback,
};
use gwenland_core::storage::config::GwenConfig;

// ── CLI args ───────────────────────────────────────────────────────────────────

/// Gwen Orange ANSI escape code — used for section headers in the report.
const GWEN_ORANGE: &str = "\x1b[38;2;255;140;66m";
const RESET: &str = "\x1b[0m";

#[derive(Args, Debug)]
#[command(
    about = "Benchmark GwenLand runtime characteristics",
    long_about = "Measure cold-start latency, token generation speed, convert\n\
                  pipeline throughput, and process memory baseline.\n\n\
                  Flags select which suites to run; omitting all flags runs\n\
                  every suite.\n\n\
                  Examples:\n  \
                    gwen benchmark\n  \
                    gwen benchmark --cold-start\n  \
                    gwen benchmark --inference\n  \
                    gwen benchmark --convert\n  \
                    gwen benchmark --full\n  \
                    gwen benchmark --layer-load model.gguf\n  \
                    gwen benchmark --model model.gguf"
)]
pub struct BenchmarkArgs {
    /// Measure cold-start latency only (spawns gwenland --version 10×)
    #[arg(long)]
    pub cold_start: bool,

    /// Measure token generation speed via native inference proxy only
    #[arg(long)]
    pub inference: bool,

    /// Measure GGUF dequantisation pipeline speed only (in-process, no file needed)
    #[arg(long)]
    pub convert: bool,

    /// Run all suites with detailed per-operation breakdown
    #[arg(long)]
    pub full: bool,

    /// Number of layers to sample in the layer-load benchmark (0 = all layers)
    /// (default: from config benchmark.layer_load, or omitted)
    #[arg(long, value_name = "N")]
    pub layer_load: Option<u32>,

    /// GGUF model file for mistral.rs inference benchmark
    /// (default: from config benchmark.model or inference.model)
    #[arg(long, value_name = "GGUF_PATH")]
    pub model: Option<PathBuf>,

    /// Quantization format to report (informational; default: from config or "Q8_0")
    #[arg(long, value_name = "FORMAT", help = "Quantization format (default: from config or Q8_0)")]
    pub quantization: Option<String>,

    /// Directory to write benchmark result JSON files
    /// (default: from config benchmark.output_dir or ./benchmark_results)
    #[arg(long, value_name = "DIR", help = "Output directory for results (default: from config or ./benchmark_results)")]
    pub output: Option<PathBuf>,
}

// ── Config resolver ───────────────────────────────────────────────────────────

/// Fully-resolved benchmark arguments after applying the priority chain:
/// CLI flag > config.toml [benchmark] section > hardcoded default.
struct ResolvedBenchmarkArgs {
    model: Option<PathBuf>,
    layer_load: Option<u32>,
    quantization: String,
    output: Option<PathBuf>,
}

impl ResolvedBenchmarkArgs {
    fn resolve(args: BenchmarkArgs, config: &GwenConfig) -> Self {
        let model = args.model
            .or_else(|| config.benchmark.model.clone());

        let layer_load = args.layer_load
            .or_else(|| config.benchmark.layer_load);

        let quantization = args.quantization
            .or_else(|| config.benchmark.quantization.clone())
            .unwrap_or_else(|| "Q8_0".to_string());

        let output = args.output
            .or_else(|| config.benchmark.output_dir.clone());

        Self { model, layer_load, quantization, output }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Main dispatch function for `gwen benchmark`.
///
/// Runs synchronously — all benchmark suites are blocking operations (child
/// process spawn, blocking HTTP, in-process timing). No tokio async needed.
pub fn run_benchmark_cmd(args: BenchmarkArgs) {
    let filter = BenchmarkFilter {
        cold_start: args.cold_start,
        inference: args.inference,
        convert: args.convert,
        full: args.full,
    };

    let config = GwenConfig::load();
    let resolved = ResolvedBenchmarkArgs::resolve(args, &config);

    println!("\u{23F1} Running benchmarks...");

    let progress_cb: ProgressCallback = Box::new(|idx, total, name, done| {
        if !done {
            print!("[{}/{}] {:<22} ...", idx, total, name);
            use std::io::Write;
            let _ = std::io::stdout().flush();
        } else {
            println!(" \u{2713}");
        }
    });

    let model_path = resolved.model.as_deref();
    let mut result = run_benchmarks(filter, Some(&progress_cb), model_path);

    // Run layer-load bench when a sample count is set.
    // Requires a model path; emit a clear error if missing.
    // sample_layers == 0 means "all layers" (None); any positive value is a real count.
    if let Some(sample_layers) = resolved.layer_load {
        match &resolved.model {
            None => eprintln!("⚠ --layer-load requires a model path (pass --model <GGUF_PATH> or set benchmark.model in config)"),
            Some(ll_path) => {
                let sample_count = if sample_layers > 0 { Some(sample_layers as usize) } else { None };
                print!("[+] {:<22} ...", "Layer Load");
                use std::io::Write;
                let _ = std::io::stdout().flush();
                match run_layer_load_bench(ll_path, sample_count) {
                    Ok(ll_result) => {
                        result.layer_load = Some(ll_result);
                        println!(" \u{2713}");
                    }
                    Err(e) => {
                        println!(" \x1b[31m✗ {}\x1b[0m", e);
                    }
                }
            }
        }
    }

    println!();
    print_report(&result, resolved.quantization.as_str());

    // Write JSON output if an output directory is configured.
    if let Some(ref out_dir) = resolved.output {
        use gwenland_core::benchmark::report::write_benchmark_file;
        if let Err(e) = std::fs::create_dir_all(out_dir) {
            eprintln!("⚠ Could not create output dir {}: {}", out_dir.display(), e);
        } else {
            let ts = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0);
            let filename = format!("benchmark_{}.json", ts);
            let out_path = out_dir.join(&filename);
            match write_benchmark_file(&result, &out_path) {
                Ok(()) => println!("✦ Results saved to {}", out_path.display()),
                Err(e) => eprintln!("⚠ Could not write results: {}", e),
            }
        }
    }
}

// ── Report printer ────────────────────────────────────────────────────────────

/// Print the final benchmark report in the spec-prescribed format.
///
/// Uses Gwen Orange for section headers per the spec. Each section is
/// printed only when the corresponding result is present (i.e. the suite ran).
fn print_report(result: &BenchmarkResult, quantization: &str) {
    let sep = "\u{2501}".repeat(40); // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    println!("\u{1F4CA} GwenLand Benchmark Results  [quant: {}]", quantization);
    println!("{}", sep);

    // ── Cold Start ────────────────────────────────────────────────────────────
    if let Some(cs) = &result.cold_start {
        println!("{GWEN_ORANGE}Cold Start{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  {:<14} {:.1}ms", "Min", cs.min_ms);
        println!("  {:<14} {:.1}ms", "Max", cs.max_ms);
        println!("  {:<14} {:.1}ms", "Mean", cs.mean_ms);
        println!("  {:<14} {:.1}ms", "Median", cs.median_ms);
        // Print a target-check hint: GwenLand claims ≤ 10 ms cold start.
        // Showing whether the measured value meets the claim helps the user
        // understand whether their environment is healthy without reading docs.
        if cs.median_ms > 10.0 {
            println!("  \x1b[33m⚠ median > 10ms target (check disk speed / AV scanner)\x1b[0m");
        }
    }

    // ── Token Generation ──────────────────────────────────────────────────────
    if let Some(inf) = &result.inference {
        println!("{GWEN_ORANGE}Token Generation{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  {:<14} {:.0} tok/s", "Speed", inf.tokens_per_sec);
        println!("  {:<14} {}", "Tokens", inf.total_tokens);
        println!("  {:<14} {:.2}s", "Elapsed", inf.elapsed_secs);
    } else if should_show_inference_skip(&result) {
        // If the filter included inference (or was "all"), show the skip warning.
        println!("{GWEN_ORANGE}Token Generation{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  \x1b[33m\u{26A0} Native proxy not running \u{2014} skipping inference benchmark (start with `gwen serve`)\x1b[0m");
    }

    // ── Convert Pipeline ──────────────────────────────────────────────────────
    if let Some(cv) = &result.convert {
        println!("{GWEN_ORANGE}Convert Pipeline{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  {:<14} {:.1} ns/element", "Standard", cv.standard_ns_per_elem);
        println!("  {:<14} {:.1} ns/element", "Euler", cv.euler_ns_per_elem);
        // Report the overhead ratio so the user can decide whether Euler mode
        // is acceptable for their throughput requirements.
        if cv.standard_ns_per_elem > 0.0 {
            let ratio = cv.euler_ns_per_elem / cv.standard_ns_per_elem;
            println!("  {:<14} {:.1}×  Euler overhead vs Standard", "Overhead", ratio);
        }
    }

    // ── Layer Load ────────────────────────────────────────────────────────────
    if let Some(ll) = &result.layer_load {
        println!("{GWEN_ORANGE}Layer Load{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  {:<14} {}", "Layers", ll.num_layers);
        println!("  {:<14} {:.1} MB", "File Size",
            ll.file_size_bytes as f64 / (1024.0 * 1024.0));
        println!("  {:<14} {} µs", "Min Load", ll.min_load_us);
        println!("  {:<14} {} µs", "Max Load", ll.max_load_us);
        println!("  {:<14} {:.1} µs", "Mean Load", ll.mean_load_us);
        println!("  {:<14} {:.1} MB", "Peak RSS", ll.peak_rss_mb);
        println!("  {:<14} {:.1} MB", "Full Est.", ll.full_load_estimate_mb);
        // Per-layer table (only if a manageable number of samples)
        if ll.samples.len() <= 32 {
            println!("  {:>6}  {:>10}  {:>10}  {:>10}", "Layer", "Load µs", "Unload µs", "RSS ΔMB");
            for s in &ll.samples {
                println!("  {:>6}  {:>10}  {:>10}  {:>10.2}",
                    s.layer_idx, s.load_us, s.unload_us, s.rss_delta_mb);
            }
        }
    }

    // ── Memory Baseline ───────────────────────────────────────────────────────
    if let Some(mem) = &result.memory {
        println!("{GWEN_ORANGE}Memory Baseline{RESET}",
            GWEN_ORANGE = GWEN_ORANGE, RESET = RESET);
        println!("  {:<14} {:.1} MB", "Process", mem.baseline_mb);
        if mem.baseline_mb == 0.0 {
            println!("  \x1b[33m⚠ memory measurement unavailable on this platform\x1b[0m");
        }
    }

    println!("{}", sep);

    // Format total elapsed: same logic as convert command summary.
    let time_str = if result.total_elapsed_secs >= 1.0 {
        format!("{:.1}s", result.total_elapsed_secs)
    } else {
        format!("{:.0}ms", result.total_elapsed_secs * 1000.0)
    };

    println!("\u{2726} Benchmark complete in {}", time_str);
}

/// Returns true when the inference suite was requested (or all suites were
/// requested) so we know to show the "proxy not running" warning.
/// We can't store the filter in BenchmarkResult so we infer it from what's present:
/// if cold_start or convert ran (Some), inference must have been requested too
/// unless the filter was selective. The most correct heuristic: if inference
/// result is None but cold_start or convert ran, the proxy was probably skipped.
fn should_show_inference_skip(result: &BenchmarkResult) -> bool {
    result.cold_start.is_some() || result.convert.is_some() || result.memory.is_some()
}
