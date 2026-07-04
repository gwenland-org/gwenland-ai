/// GwenLand GGQR Benchmark binary.
///
/// Usage:
///   bench_ggqr <gguf_path> [--mode standard|euler] [--iterations N] [--expected-sum <value>]
///
/// Measures load + dequant throughput over N iterations and optionally checks
/// the regression sum (sum of all dequantised f32 values, accumulated as f64,
/// rounded to i64) against a known-good expected value.
use std::fs;
use std::path::Path;
use std::time::Instant;

use gwenland_core::convert::dequant::DequantMode;
use gwenland_core::engine::gguf_loader;

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    gguf_path: String,
    mode: DequantMode,
    iterations: usize,
    expected_sum: Option<i64>,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();

    if raw.len() < 2 {
        return Err(format!(
            "Usage: {} <gguf_path> [--mode standard|euler] [--iterations N] [--expected-sum <value>]",
            raw[0]
        ));
    }

    let gguf_path = raw[1].clone();
    let mut mode = DequantMode::Standard;
    let mut iterations: usize = 3;
    let mut expected_sum: Option<i64> = None;

    let mut i = 2usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--mode" => {
                i += 1;
                if i >= raw.len() {
                    return Err("--mode requires a value (standard|euler)".into());
                }
                mode = match raw[i].as_str() {
                    "standard" => DequantMode::Standard,
                    "euler"    => DequantMode::Euler,
                    other      => return Err(format!("unknown mode '{}'; expected standard|euler", other)),
                };
            }
            "--iterations" => {
                i += 1;
                if i >= raw.len() {
                    return Err("--iterations requires a numeric value".into());
                }
                iterations = raw[i].parse::<usize>()
                    .map_err(|_| format!("'{}' is not a valid iteration count", raw[i]))?;
                if iterations == 0 {
                    return Err("--iterations must be >= 1".into());
                }
            }
            "--expected-sum" => {
                i += 1;
                if i >= raw.len() {
                    return Err("--expected-sum requires a numeric value".into());
                }
                expected_sum = Some(
                    raw[i].parse::<i64>()
                        .map_err(|_| format!("'{}' is not a valid i64 sum", raw[i]))?
                );
            }
            other => {
                return Err(format!("unknown argument '{}'", other));
            }
        }
        i += 1;
    }

    Ok(Args { gguf_path, mode, iterations, expected_sum })
}

// ── Formatting helpers ────────────────────────────────────────────────────────

/// Format a byte count as a human-readable GiB string (3 decimal places).
fn format_gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
}

/// Format throughput in GiB/s.
fn throughput_gibs(bytes: u64, elapsed_ms: f64) -> f64 {
    let elapsed_s = elapsed_ms / 1000.0;
    if elapsed_s == 0.0 {
        return 0.0;
    }
    (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / elapsed_s
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args = match parse_args() {
        Ok(a)  => a,
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(2);
        }
    };

    let path = Path::new(&args.gguf_path);

    // Get file size for throughput calculation.
    let file_size_bytes: u64 = match fs::metadata(path) {
        Ok(m)  => m.len(),
        Err(e) => {
            eprintln!("Error: cannot stat '{}': {e}", args.gguf_path);
            std::process::exit(2);
        }
    };

    let mode_str = match args.mode {
        DequantMode::Standard => "Standard",
        DequantMode::Euler    => "Euler",
    };

    // ── Header ────────────────────────────────────────────────────────────────
    println!("GwenLand GGQR Benchmark");
    println!("========================");
    println!("File:       {} ({})", path.file_name().unwrap_or_default().to_string_lossy(), format_gib(file_size_bytes));
    println!("Mode:       {mode_str}");
    println!("Iterations: {}", args.iterations);
    println!();

    // ── Benchmark loop ────────────────────────────────────────────────────────
    let mut elapsed_ms_vec: Vec<f64> = Vec::with_capacity(args.iterations);
    let mut last_sum: f64 = 0.0;  // regression sum from the final iteration

    for run in 1..=args.iterations {
        let t0 = Instant::now();

        let weights = match gguf_loader::load_and_dequant(path, args.mode) {
            Ok(w)  => w,
            Err(e) => {
                eprintln!("Error on run {run}: {e}");
                std::process::exit(1);
            }
        };

        let elapsed = t0.elapsed();
        let ms = elapsed.as_secs_f64() * 1000.0;
        elapsed_ms_vec.push(ms);

        // Accumulate regression sum as f64 to avoid f32 precision loss.
        let sum: f64 = weights.values()
            .flat_map(|v| v.iter())
            .map(|&x| x as f64)
            .sum();
        last_sum = sum;

        let gb_s = throughput_gibs(file_size_bytes, ms);
        println!("Run {run}: {ms:.0}ms  |  {gb_s:.2} GiB/s");
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let avg_ms = elapsed_ms_vec.iter().sum::<f64>() / elapsed_ms_vec.len() as f64;
    let avg_gibs = throughput_gibs(file_size_bytes, avg_ms);
    const TARGET_GIBS: f64 = 9.0;

    println!();
    println!("Average:    {avg_ms:.1}ms  |  {avg_gibs:.2} GiB/s");
    let pass_fail = if avg_gibs >= TARGET_GIBS { "✓ PASS" } else { "✗ BELOW TARGET" };
    println!("Target:     >{TARGET_GIBS:.1} GiB/s  {pass_fail}");

    // ── Regression check ──────────────────────────────────────────────────────
    println!();
    let rounded_sum = last_sum.round() as i64;
    match args.expected_sum {
        Some(expected) => {
            if rounded_sum == expected {
                println!("Regression: sum={rounded_sum}  ✓ MATCH (expected: {expected})");
            } else {
                println!("Regression: sum={rounded_sum}  ✗ MISMATCH (expected: {expected})");
                std::process::exit(1);
            }
        }
        None => {
            println!("Regression: sum={rounded_sum}  (no --expected-sum provided, skipping check)");
        }
    }
}
