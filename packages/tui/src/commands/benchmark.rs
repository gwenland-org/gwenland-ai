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
use clap::Args;
use gwenland_core::benchmark::{
    run_benchmarks, BenchmarkFilter, BenchmarkResult, ProgressCallback,
};

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
                    gwen benchmark --full"
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

    println!("\u{23F1} Running benchmarks...");

    // The progress callback is called by gwen-core once per suite start/finish.
    // We print `[i/N] Suite Name    ...` on start and append ` ✓` on finish.
    // Using a mutable tracking approach: we print the start line without newline,
    // then overwrite with the done line. However, since gwen-core calls the
    // callback with separate start/done signals, we use a two-line approach
    // (start: print line; done: print checkmark on next line at same indent).
    //
    // We allocate the callback on the heap (Box<dyn Fn>) as required by the
    // ProgressCallback type alias in gwen-core.
    let progress_cb: ProgressCallback = Box::new(|idx, total, name, done| {
        if !done {
            // Print the "starting" line — no newline so we can overwrite with ✓.
            // We use a fixed-width name field so the ✓ column is always aligned.
            print!("[{}/{}] {:<22} ...", idx, total, name);
            // Flush stdout so the partial line appears before the suite runs.
            // Without flush, buffered I/O would hold the line until the done call.
            use std::io::Write;
            let _ = std::io::stdout().flush();
        } else {
            println!(" \u{2713}");
        }
    });

    let result = run_benchmarks(filter, Some(&progress_cb), None);

    println!();
    print_report(&result);
}

// ── Report printer ────────────────────────────────────────────────────────────

/// Print the final benchmark report in the spec-prescribed format.
///
/// Uses Gwen Orange for section headers per the spec. Each section is
/// printed only when the corresponding result is present (i.e. the suite ran).
fn print_report(result: &BenchmarkResult) {
    let sep = "\u{2501}".repeat(40); // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    println!("\u{1F4CA} GwenLand Benchmark Results");
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
