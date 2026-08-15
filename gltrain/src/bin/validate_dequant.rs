/// validate_dequant — end-to-end dequantisation validator.
///
/// Parses a real GGUF file, dequantises the requested tensors, and compares
/// each element against a reference CSV produced by a known-good tool
/// (e.g. llama.cpp or gguf-py).
///
/// Usage:
///   validate_dequant <gguf_path> <reference_csv> [--euler] [--tensor <name>]
///
/// reference_csv format (no header required, header line is auto-detected):
///   tensor_name,element_index,expected_f32
///
/// Tolerances:
///   Standard mode: |got - expected| < 1e-3
///   Euler mode:    |got - expected| < 0.1
///
/// Exit codes:
///   0 — all reference rows passed
///   1 — one or more rows failed, or a parse/IO error occurred
use std::collections::HashMap;
use std::path::Path;
use std::process;

use gwenland_core::convert::{
    dequant::{self, DequantMode},
    gguf_parser,
};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!(
            "Usage: validate_dequant <gguf_path> <reference_csv> [--euler] [--tensor <name>]"
        );
        process::exit(1);
    }

    let gguf_path      = &args[1];
    let reference_path = &args[2];

    let mode = if args.contains(&"--euler".to_string()) {
        DequantMode::Euler
    } else {
        DequantMode::Standard
    };

    let filter_tensor: Option<String> = args
        .windows(2)
        .find(|w| w[0] == "--tensor")
        .map(|w| w[1].clone());

    let tolerance = match mode {
        DequantMode::Standard => 1e-3_f32,
        DequantMode::Euler    => 0.1_f32,
    };

    // ── Parse GGUF ────────────────────────────────────────────────────────────
    eprintln!("[validate_dequant] Parsing GGUF: {gguf_path}");
    let gguf = gguf_parser::parse(Path::new(gguf_path)).unwrap_or_else(|e| {
        eprintln!("Error: failed to parse GGUF file: {e}");
        process::exit(1);
    });
    eprintln!(
        "[validate_dequant] Loaded {} tensor(s) from GGUF v{}",
        gguf.tensors.len(),
        gguf.version
    );

    // ── Load reference CSV ────────────────────────────────────────────────────
    // Format: tensor_name,element_index,expected_f32
    // Header line (if present) is detected by checking whether the second
    // field parses as a u64 — if not, the line is skipped.
    eprintln!("[validate_dequant] Loading reference CSV: {reference_path}");
    let csv_raw = std::fs::read_to_string(reference_path).unwrap_or_else(|e| {
        eprintln!("Error: cannot read reference CSV: {e}");
        process::exit(1);
    });

    struct RefRow {
        tensor_name: String,
        element_idx: usize,
        expected:    f32,
    }

    let mut ref_rows: Vec<RefRow> = Vec::new();
    for (line_no, line) in csv_raw.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() { continue; }

        let parts: Vec<&str> = line.splitn(3, ',').collect();
        if parts.len() < 3 {
            eprintln!("Warning: line {}: expected 3 comma-separated fields, skipping", line_no + 1);
            continue;
        }

        // Auto-detect header: if element_index field is not a number, skip.
        let element_idx = match parts[1].trim().parse::<usize>() {
            Ok(v)  => v,
            Err(_) => {
                // Likely a header row — skip silently on line 1, warn otherwise.
                if line_no != 0 {
                    eprintln!(
                        "Warning: line {}: element_index '{}' is not a number, skipping",
                        line_no + 1,
                        parts[1].trim()
                    );
                }
                continue;
            }
        };

        let expected = match parts[2].trim().parse::<f32>() {
            Ok(v)  => v,
            Err(_) => {
                eprintln!(
                    "Warning: line {}: expected_f32 '{}' is not a float, skipping",
                    line_no + 1,
                    parts[2].trim()
                );
                continue;
            }
        };

        let tensor_name = parts[0].trim().to_string();

        // Apply --tensor filter if set.
        if let Some(ref filter) = filter_tensor {
            if &tensor_name != filter {
                continue;
            }
        }

        ref_rows.push(RefRow { tensor_name, element_idx, expected });
    }

    if ref_rows.is_empty() {
        eprintln!("Error: no valid reference rows found in CSV (after filter).");
        process::exit(1);
    }
    eprintln!("[validate_dequant] {} reference row(s) to validate", ref_rows.len());

    // ── Dequantise tensors (cached per name) ──────────────────────────────────
    // Build a lookup map from tensor name → TensorInfo for O(1) access.
    let tensor_map: HashMap<&str, &gguf_parser::TensorInfo> = gguf
        .tensors
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();

    // Cache: tensor_name → dequantised Vec<f32>.
    // Each tensor is dequantised at most once regardless of how many reference
    // rows target it.
    let mut cache: HashMap<String, Vec<f32>> = HashMap::new();

    // ── Validate each reference row ───────────────────────────────────────────
    let mut pass_count  = 0usize;
    let mut fail_count  = 0usize;
    let mut max_err     = 0.0_f32;
    let mut sum_err     = 0.0_f32;

    for row in &ref_rows {
        // Dequantise tensor on first access; reuse cached result thereafter.
        let weights = match cache.get(&row.tensor_name) {
            Some(w) => w,
            None => {
                let tensor_info = match tensor_map.get(row.tensor_name.as_str()) {
                    Some(t) => *t,
                    None => {
                        eprintln!(
                            "[FAIL] tensor '{}' not found in GGUF file",
                            row.tensor_name
                        );
                        fail_count += 1;
                        continue;
                    }
                };

                match dequant::dequantize(tensor_info, mode) {
                    Ok(w) => {
                        cache.insert(row.tensor_name.clone(), w);
                        cache.get(&row.tensor_name).unwrap()
                    }
                    Err(e) => {
                        eprintln!(
                            "[FAIL] tensor '{}': dequantisation error: {e}",
                            row.tensor_name
                        );
                        fail_count += 1;
                        continue;
                    }
                }
            }
        };

        // Bounds check.
        if row.element_idx >= weights.len() {
            eprintln!(
                "[FAIL] {}[{}]: index out of bounds (tensor has {} elements)",
                row.tensor_name,
                row.element_idx,
                weights.len()
            );
            fail_count += 1;
            continue;
        }

        let got = weights[row.element_idx];
        let err = (got - row.expected).abs();

        sum_err += err;
        if err > max_err { max_err = err; }

        if err < tolerance {
            println!(
                "[PASS] {}[{}] got={:.6} expected={:.6} err={:.6}",
                row.tensor_name, row.element_idx, got, row.expected, err
            );
            pass_count += 1;
        } else {
            println!(
                "[FAIL] {}[{}] got={:.6} expected={:.6} err={:.6}",
                row.tensor_name, row.element_idx, got, row.expected, err
            );
            fail_count += 1;
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    let total    = pass_count + fail_count;
    let mean_err = if total > 0 { sum_err / total as f32 } else { 0.0 };

    println!(
        "Summary: {}/{} passed | max_err={:.6} | mean_err={:.6}",
        pass_count, total, max_err, mean_err
    );

    if fail_count > 0 {
        process::exit(1);
    }
}
