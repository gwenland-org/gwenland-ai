use std::path::Path;

use anyhow::Result;

use super::{BenchmarkResult, OutputFormat};

pub fn format_benchmark_report(result: &BenchmarkResult, fmt: OutputFormat) -> String {
    match fmt {
        OutputFormat::Json => format_json(result),
        OutputFormat::Text => format_text(result),
    }
}

/// Write benchmark result as JSON to `path` and print text summary to stdout.
pub fn write_benchmark_file(result: &BenchmarkResult, path: &Path) -> Result<()> {
    let json = format_benchmark_report(result, OutputFormat::Json);
    std::fs::write(path, &json)?;
    println!("{}", format_benchmark_report(result, OutputFormat::Text));
    Ok(())
}

// ── JSON branch ───────────────────────────────────────────────────────────────

fn format_json(result: &BenchmarkResult) -> String {
    let timestamp = chrono::Utc::now().to_rfc3339();

    let cold_start = result.cold_start.as_ref().map(|r| {
        serde_json::json!({
            "min_ms":    r.min_ms,
            "max_ms":    r.max_ms,
            "mean_ms":   r.mean_ms,
            "median_ms": r.median_ms,
            "iterations": r.iterations,
        })
    });

    let inference = result.inference.as_ref().map(|r| {
        serde_json::json!({
            "tokens_per_sec": r.tokens_per_sec,
            "total_tokens":   r.total_tokens,
            "elapsed_secs":   r.elapsed_secs,
            "backend":        r.backend,
            "model_file":     r.model_file,
        })
    });

    let convert = result.convert.as_ref().map(|r| {
        serde_json::json!({
            "standard_ns_per_elem": r.standard_ns_per_elem,
            "euler_ns_per_elem":    r.euler_ns_per_elem,
            "standard_stddev":      r.standard_stddev,
            "euler_stddev":         r.euler_stddev,
        })
    });

    let memory = result.memory.as_ref().map(|r| {
        serde_json::json!({ "baseline_mb": r.baseline_mb })
    });

    let layer_load = result.layer_load.as_ref().map(|r| {
        let samples: Vec<_> = r.samples.iter().map(|s| {
            serde_json::json!({
                "layer_idx":    s.layer_idx,
                "load_us":      s.load_us,
                "unload_us":    s.unload_us,
                "rss_delta_mb": s.rss_delta_mb,
                "byte_total":   s.byte_total,
                "slice_count":  s.slice_count,
            })
        }).collect();
        serde_json::json!({
            "samples":               samples,
            "file_size_bytes":       r.file_size_bytes,
            "num_layers":            r.num_layers,
            "min_load_us":           r.min_load_us,
            "max_load_us":           r.max_load_us,
            "mean_load_us":          r.mean_load_us,
            "peak_rss_mb":           r.peak_rss_mb,
            "full_load_estimate_mb": r.full_load_estimate_mb,
        })
    });

    let obj = serde_json::json!({
        "schema_version":    "2",
        "timestamp":         timestamp,
        "total_elapsed_secs": result.total_elapsed_secs,
        "cold_start":        cold_start,
        "inference":         inference,
        "convert":           convert,
        "memory":            memory,
        "layer_load":        layer_load,
    });

    serde_json::to_string_pretty(&obj).unwrap_or_default()
}

// ── Text branch ───────────────────────────────────────────────────────────────

fn format_text(result: &BenchmarkResult) -> String {
    let date = chrono::Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let mut out = String::new();

    out.push_str(&format!("GwenLand Benchmark — {}\n", date));
    out.push_str("════════════════════════════════\n");

    // Cold start
    match &result.cold_start {
        Some(r) => {
            let ok = if r.mean_ms < 15.0 { "✓" } else { "✗" };
            out.push_str(&format!("Cold Start:    {:.1} ms  {} (< 15 ms)\n", r.mean_ms, ok));
        }
        None => out.push_str("Cold Start:      (not measured)\n"),
    }

    // Inference
    match &result.inference {
        Some(r) => {
            let model = r.model_file.as_deref().unwrap_or("unknown");
            out.push_str(&format!(
                "Inference:     {:.1} tok/s  [{}, {}]\n",
                r.tokens_per_sec, r.backend, model
            ));
        }
        None => out.push_str("Inference:       (not measured)\n"),
    }

    // Layer load
    match &result.layer_load {
        Some(r) => {
            out.push_str(&format!(
                "Layer Load:    mean {:.0} µs/layer  |  peak RSS {:.0} MB  |  est. full {:.0} MB\n",
                r.mean_load_us, r.peak_rss_mb, r.full_load_estimate_mb
            ));
        }
        None => out.push_str("Layer Load:      (not measured)\n"),
    }

    // Memory
    match &result.memory {
        Some(r) => out.push_str(&format!("Memory Floor:  {:.1} MB RSS\n", r.baseline_mb)),
        None => out.push_str("Memory Floor:    (not measured)\n"),
    }

    out.push_str("────────────────────────────────\n");
    out.push_str(&format!("Total:         {:.2} s\n", result.total_elapsed_secs));

    out
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::{BenchmarkResult, InferenceResult, OutputFormat};

    fn empty_result() -> BenchmarkResult {
        BenchmarkResult {
            cold_start:         None,
            inference:          None,
            convert:            None,
            memory:             None,
            layer_load:         None,
            total_elapsed_secs: 0.0,
        }
    }

    #[test]
    fn test_format_json_is_valid_json() {
        let json_str = format_benchmark_report(&empty_result(), OutputFormat::Json);
        let parsed = serde_json::from_str::<serde_json::Value>(&json_str);
        assert!(parsed.is_ok(), "output must be valid JSON: {}", json_str);
    }

    #[test]
    fn test_format_json_schema_version() {
        let json_str = format_benchmark_report(&empty_result(), OutputFormat::Json);
        let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed["schema_version"], "2");
    }

    #[test]
    fn test_format_text_contains_header() {
        let text = format_benchmark_report(&empty_result(), OutputFormat::Text);
        assert!(text.contains("GwenLand Benchmark"), "missing header in:\n{}", text);
    }

    #[test]
    fn test_format_text_layer_load_none() {
        let text = format_benchmark_report(&empty_result(), OutputFormat::Text);
        assert!(text.contains("(not measured)"), "expected '(not measured)' in:\n{}", text);
    }

    #[test]
    fn test_format_text_inference_backend() {
        let mut result = empty_result();
        result.inference = Some(InferenceResult {
            tokens_per_sec: 42.0,
            total_tokens:   64,
            elapsed_secs:   1.5,
            backend:        "mistralrs".to_string(),
            model_file:     Some("qwen3.gguf".to_string()),
        });
        let text = format_benchmark_report(&result, OutputFormat::Text);
        assert!(text.contains("mistralrs"), "expected 'mistralrs' in:\n{}", text);
    }
}
