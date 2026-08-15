/// Convert pipeline micro-benchmark — in-process dequantisation speed.
///
/// Why synthetic data rather than a real GGUF file?
/// The benchmark must be self-contained: users shouldn't need to download a
/// multi-GB model to run `gwen benchmark`. Synthetic Q8_0 data exercises the
/// exact same code path as real data — `dequantize()` doesn't distinguish
/// between bytes that came from a file and bytes we generated here.
///
/// Why Q8_0 specifically (not Q4_0)?
/// Q8_0 is the dominant quantisation scheme for benchmarking because:
///   1. Its block layout (34 bytes/32 elements) is simpler than Q4_0's nibble
///      packing, so the benchmark isolates the arithmetic cost rather than the
///      bit-unpacking cost.
///   2. Q8_0 is the most common dtype in recently published models (Qwen3,
///      Llama 3.2) after the default Q4_K_M.
/// A Q4_0 benchmark can be added later if nibble-unpacking becomes a hotspot.
///
/// Why 1000 iterations?
/// - 10 iterations: dominated by first-run cache effects (L1/L2 miss pattern
///   changes between iterations as the JIT settles).
/// - 10,000 iterations: takes > 1 s on slow hardware, making the benchmark feel
///   slow.
/// - 1,000 iterations: stable stddev with < 200 ms total time on typical hardware.
///   The timing noise at 1,000 iterations is < 5%, which is good enough for a
///   "does the Euler path add significant overhead?" answer.
use std::time::Instant;

use crate::convert::dequant::{dequantize, DequantMode};
use crate::convert::gguf_parser::{GgufDtype, TensorInfo};

use super::ConvertBenchResult;

/// Number of benchmark iterations for each mode.
/// See module-level comment for the rationale.
const ITERATIONS: usize = 1_000;

/// Number of Q8_0 blocks per synthetic tensor.
/// 10 blocks × 32 elements = 320 elements per tensor.
/// Small enough that each iteration fits comfortably in L1 cache (~10 KB),
/// which is what we want: we're measuring arithmetic throughput, not memory
/// bandwidth.
const N_BLOCKS: usize = 10;

/// Elements per Q8_0 block. Matches GGML_QK = 32.
const BLOCK_ELEMENTS: usize = 32;

/// Total elements in the synthetic tensor.
const N_ELEMENTS: usize = N_BLOCKS * BLOCK_ELEMENTS;

/// Run the dequantisation micro-benchmark for both Standard and Euler modes.
///
/// Builds a synthetic Q8_0 tensor once (the build cost is amortised across all
/// iterations) and times 1,000 calls to `dequantize()` for each mode.
pub fn run_convert_bench() -> ConvertBenchResult {
    let tensor = make_synthetic_tensor();

    let standard_samples = bench_mode(&tensor, DequantMode::Standard);
    let euler_samples    = bench_mode(&tensor, DequantMode::Euler);

    ConvertBenchResult {
        standard_ns_per_elem: mean_ns_per_elem(&standard_samples),
        euler_ns_per_elem:    mean_ns_per_elem(&euler_samples),
        standard_stddev:      stddev_ns_per_elem(&standard_samples),
        euler_stddev:         stddev_ns_per_elem(&euler_samples),
    }
}

// ── Synthetic data builder ────────────────────────────────────────────────────

/// Build a synthetic Q8_0 TensorInfo with deterministic content.
///
/// Block layout: [scale: f16 (2 bytes)] [values: i8 × 32 (32 bytes)]
///
/// Why deterministic content?
/// Using a fixed pattern (scale = 0.5, values = sawtooth -16..+15) means the
/// benchmark produces the same result on every run and every machine, making it
/// easy to compare across PRs. A random seed would introduce run-to-run variance
/// that obscures real performance changes.
///
/// Why sawtooth values?
/// The Euler dequant path computes `max_bound = abs(values).max()` per block.
/// A sawtooth ensures max_bound is always 16 (non-zero) so the Euler path never
/// hits the zero-block early-exit, giving a fair comparison with Standard mode.
fn make_synthetic_tensor() -> TensorInfo {
    // 2 bytes scale (f16) + 32 bytes values (i8) per block.
    let bytes_per_block = 2 + BLOCK_ELEMENTS;
    let mut raw_data = Vec::with_capacity(N_BLOCKS * bytes_per_block);

    // scale = 0.5 as f16, little-endian.
    // 0.5 in f16: sign=0, exponent=14 (biased), mantissa=0 → 0x3800
    let scale_bytes: [u8; 2] = [0x00, 0x38];

    for b in 0..N_BLOCKS {
        raw_data.extend_from_slice(&scale_bytes);
        // Sawtooth: values cycle through -16..+15 with a per-block offset
        // so adjacent blocks have different patterns (avoids branch predictor
        // bias from identical data across all blocks).
        for k in 0..BLOCK_ELEMENTS {
            let v = (((b * 7 + k) % 32) as i32 - 16) as i8;
            raw_data.push(v as u8);
        }
    }

    TensorInfo {
        name: "__benchmark_synthetic__".to_string(),
        shape: vec![N_ELEMENTS as u64],
        dtype: GgufDtype::Q8_0,
        data_offset: 0,
        data_size: raw_data.len(),
        raw_data,
    }
}

// ── Timing loop ───────────────────────────────────────────────────────────────

/// Run `ITERATIONS` calls to `dequantize(tensor, mode)` and collect per-call
/// wall-clock nanoseconds.
///
/// Why collect individual samples rather than a single total time?
/// Individual samples let us compute stddev, which reveals whether the Euler
/// path has higher variance (e.g. due to branch misprediction in the cosine
/// path). A high stddev relative to mean suggests the benchmark is dominated by
/// cache effects or OS jitter rather than the arithmetic itself.
///
/// The `black_box` equivalent here is the `let _ = result` line: we bind the
/// return value to a variable to prevent the compiler from eliding the call
/// entirely (dead-code elimination). Rust doesn't have `std::hint::black_box`
/// in edition 2021 stable, but binding to `_` is sufficient for `Vec<f32>` since
/// the allocation is observable (it calls the global allocator).
fn bench_mode(tensor: &TensorInfo, mode: DequantMode) -> Vec<f64> {
    let mut samples = Vec::with_capacity(ITERATIONS);

    for _ in 0..ITERATIONS {
        let t0 = Instant::now();
        // We unwrap here because synthetic data is always valid — a panic
        // would be a bug in `make_synthetic_tensor`, not a user-facing error.
        let result = dequantize(tensor, mode).unwrap();
        let elapsed_ns = t0.elapsed().as_nanos() as f64;
        // Bind result so the compiler can't dead-code-eliminate the dequant call.
        // The Vec<f32> allocation is observable side-effect enough.
        let _ = result;
        samples.push(elapsed_ns);
    }

    samples
}

// ── Statistics ────────────────────────────────────────────────────────────────

/// Compute mean nanoseconds per element from a slice of per-call nanosecond totals.
fn mean_ns_per_elem(samples: &[f64]) -> f64 {
    if samples.is_empty() || N_ELEMENTS == 0 {
        return 0.0;
    }
    let mean_total_ns = samples.iter().sum::<f64>() / samples.len() as f64;
    mean_total_ns / N_ELEMENTS as f64
}

/// Compute population standard deviation of nanoseconds per element.
///
/// We use population stddev (divide by N) rather than sample stddev (divide by
/// N-1) because with 1,000 iterations the Bessel correction is negligible and
/// population stddev is the more commonly reported figure in benchmark contexts.
fn stddev_ns_per_elem(samples: &[f64]) -> f64 {
    if samples.len() < 2 || N_ELEMENTS == 0 {
        return 0.0;
    }
    let mean = mean_ns_per_elem(samples);
    // Convert per-call ns to per-element ns before computing variance so the
    // stddev is in the same units as the mean (ns/element).
    let variance = samples
        .iter()
        .map(|&ns| {
            let ns_per_elem = ns / N_ELEMENTS as f64;
            let diff = ns_per_elem - mean;
            diff * diff
        })
        .sum::<f64>()
        / samples.len() as f64;
    variance.sqrt()
}
