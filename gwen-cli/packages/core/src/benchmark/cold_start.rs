/// Cold-start benchmark — measures time from process spawn to first stdout byte.
///
/// Why spawn a child process rather than measuring something in-process?
/// A cold-start benchmark must pay the OS loader cost: dynamic linker resolution,
/// page faults on the text segment, and runtime initialisation (tokio, clap parse).
/// None of those costs are visible inside a running process. Spawning
/// `gwenland --version` as a child process is the only way to measure the full
/// path that a user experiences.
///
/// Why `--version` specifically?
/// It is the cheapest possible exit path: clap intercepts `--version` before
/// the tokio runtime starts (see main.rs line 133), so the measurement captures
/// the loader + clap parse time without the cost of building the tokio threadpool.
/// This isolates the "time to first useful output" from "time to run a command".
///
/// Why 10 iterations?
/// - 1 iteration would be dominated by cold page-faults (first-run bias).
/// - 100 iterations would make the benchmark take ~1 second on slow disks.
/// - 10 is the practical sweet-spot: enough to produce a stable median while
///   completing in under 150 ms on typical hardware.
use std::io::Read;
use std::time::Instant;

use super::ColdStartResult;

/// Number of times to spawn the process and measure first-byte latency.
/// 10 iterations gives a stable median without making the benchmark perceptible
/// to the user on modern NVMe storage (< 150 ms total).
const ITERATIONS: usize = 10;

/// Run the cold-start benchmark: spawn `gwenland --version` N times, record
/// the wall-clock time from spawn to first stdout byte for each iteration.
///
/// Returns aggregated statistics. Never returns an error: if the binary cannot
/// be found, all iteration times will be 0.0 (the error path records 0 ms so
/// the report clearly shows something is wrong without aborting the benchmark).
pub fn run_cold_start() -> ColdStartResult {
    let binary = resolve_binary_path();
    let mut samples = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let elapsed = measure_first_byte(&binary);
        samples.push(elapsed);
    }

    aggregate(samples)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Find the path to the running binary so we can re-spawn it.
///
/// `std::env::current_exe()` is the most reliable cross-platform way to get
/// the path of the current process's binary. We prefer it over a hard-coded
/// "gwenland" string because:
///   1. The binary may be invoked as "gwen" via the shell alias and the PATH
///      lookup would resolve differently.
///   2. In CI the binary lives in `target/debug/` or `target/release/` not
///      on PATH.
///   3. On Windows the exe extension must be present.
fn resolve_binary_path() -> std::path::PathBuf {
    std::env::current_exe().unwrap_or_else(|_| {
        // Fallback to "gwen" on PATH if current_exe fails (e.g. proc filesystem
        // missing on some container setups). This is unlikely in practice.
        std::path::PathBuf::from("gwen")
    })
}

/// Spawn `binary --version`, start the clock at spawn time, and return the
/// wall-clock milliseconds until the first byte arrives on stdout.
///
/// Why measure to first byte rather than process exit?
/// First byte reflects the user-perceived latency — the terminal renders as
/// soon as the first character arrives. Process exit adds a non-deterministic
/// OS cleanup tail that inflates the measurement without reflecting UX.
///
/// Returns 0.0 on any spawn or read error so the outer loop can still produce
/// a result rather than panicking mid-benchmark.
fn measure_first_byte(binary: &std::path::Path) -> f64 {
    use std::process::{Command, Stdio};

    let mut child = match Command::new(binary)
        .arg("--version")
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(_) => return 0.0,
    };

    // Clock starts the moment spawn() returns. The OS has scheduled the child
    // at this point; the loader hasn't necessarily run yet, but the scheduling
    // latency is part of the user-perceived cold-start.
    let t0 = Instant::now();

    let mut stdout = match child.stdout.take() {
        Some(s) => s,
        None => {
            let _ = child.wait();
            return 0.0;
        }
    };

    // Read exactly 1 byte — we don't care about the content, only the arrival time.
    let mut buf = [0u8; 1];
    let elapsed_ms = match stdout.read_exact(&mut buf) {
        Ok(_) => t0.elapsed().as_secs_f64() * 1000.0,
        Err(_) => 0.0,
    };

    // Reap the child to avoid zombie processes on Linux. We don't wait for
    // completion — the child will have exited by the time we reap it.
    let _ = child.wait();

    elapsed_ms
}

/// Compute min, max, mean, and median from a Vec of f64 samples.
///
/// Median uses the sort-and-pick-middle approach rather than an online algorithm
/// because N=10 makes allocation cost negligible, and the sort is needed anyway
/// for min/max verification.
fn aggregate(mut samples: Vec<f64>) -> ColdStartResult {
    // Guard against an empty sample set (can happen if ITERATIONS == 0, which
    // would be a programming error, but we handle it gracefully).
    if samples.is_empty() {
        return ColdStartResult {
            min_ms: 0.0,
            max_ms: 0.0,
            mean_ms: 0.0,
            median_ms: 0.0,
            iterations: 0,
        };
    }

    let n = samples.len();
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min_ms = samples[0];
    let max_ms = samples[n - 1];
    let mean_ms = samples.iter().sum::<f64>() / n as f64;

    // For even N: average the two middle values (unbiased median estimator).
    // For odd N: take the middle element.
    let median_ms = if n % 2 == 0 {
        (samples[n / 2 - 1] + samples[n / 2]) / 2.0
    } else {
        samples[n / 2]
    };

    ColdStartResult {
        min_ms,
        max_ms,
        mean_ms,
        median_ms,
        iterations: n,
    }
}
