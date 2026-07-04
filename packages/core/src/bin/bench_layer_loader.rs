/// GwenLand Layer Loader Benchmark.
///
/// Usage:
///   bench_layer_loader <gguf_path> [--layer N] [--iterations M] [--compare-full] [--format text|json]
///
/// Measures per-layer load/unload latency and RSS peak, then optionally
/// projects what a hypothetical full-model load would cost in RSS.
///
/// Metrics (per layer):
///   load_us      — wall-clock µs to map + touch layer N into RSS
///   unload_us    — wall-clock µs to drop LoadedLayer (triggers MADV_DONTNEED on Unix)
///   rss_delta_mb — RSS delta between pre-load and post-load snapshots (MB)
///
/// When --compare-full is set, the report includes:
///   full_load_estimate_mb — single_layer_rss_peak × num_layers
///   peak_rss_mb           — actual peak RSS observed across all sampled layers
use std::path::PathBuf;
use std::time::Instant;

use gwenland_core::train::LayerLoader;

// ── Argument parsing ──────────────────────────────────────────────────────────

#[derive(Debug)]
enum OutputFormat { Text, Json }

#[derive(Debug)]
struct Args {
    gguf_path:    PathBuf,
    layer:        Option<usize>,
    iterations:   usize,
    compare_full: bool,
    format:       OutputFormat,
}

fn parse_args() -> Result<Args, String> {
    let raw: Vec<String> = std::env::args().collect();
    if raw.len() < 2 {
        return Err(format!(
            "Usage: {} <gguf_path> [--layer N] [--iterations M] [--compare-full] [--format text|json]",
            raw[0]
        ));
    }

    let gguf_path = PathBuf::from(&raw[1]);
    let mut layer:        Option<usize> = None;
    let mut iterations:   usize        = 3;
    let mut compare_full: bool         = false;
    let mut format                     = OutputFormat::Text;

    let mut i = 2usize;
    while i < raw.len() {
        match raw[i].as_str() {
            "--layer" => {
                i += 1;
                if i >= raw.len() { return Err("--layer requires a value".into()); }
                layer = Some(
                    raw[i].parse::<usize>()
                        .map_err(|_| format!("'{}' is not a valid layer index", raw[i]))?
                );
            }
            "--iterations" => {
                i += 1;
                if i >= raw.len() { return Err("--iterations requires a value".into()); }
                let n = raw[i].parse::<usize>()
                    .map_err(|_| format!("'{}' is not a valid iteration count", raw[i]))?;
                if n == 0 { return Err("--iterations must be >= 1".into()); }
                iterations = n;
            }
            "--compare-full" => { compare_full = true; }
            "--format" => {
                i += 1;
                if i >= raw.len() { return Err("--format requires text|json".into()); }
                format = match raw[i].as_str() {
                    "text" => OutputFormat::Text,
                    "json" => OutputFormat::Json,
                    other  => return Err(format!("unknown format '{}'; expected text|json", other)),
                };
            }
            other => return Err(format!("unknown argument '{}'", other)),
        }
        i += 1;
    }

    Ok(Args { gguf_path, layer, iterations, compare_full, format })
}

// ── RSS sampling ──────────────────────────────────────────────────────────────

fn rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }
    #[cfg(not(target_os = "linux"))]
    {
        use sysinfo::{Pid, System};
        let mut sys = System::new();
        sys.refresh_processes();
        let pid = Pid::from(std::process::id() as usize);
        sys.process(pid)
            .map(|p| p.memory() as f64 / (1024.0 * 1024.0))
            .unwrap_or(0.0)
    }
}

// ── Per-layer measurement ─────────────────────────────────────────────────────

struct LayerSample {
    layer_idx:    usize,
    load_us:      u64,
    unload_us:    u64,
    rss_delta_mb: f64,
    slice_count:  usize,
    byte_total:   u64,
}

fn measure_layer(loader: &LayerLoader, layer_idx: usize, iterations: usize) -> LayerSample {
    let mut total_load_us:   u64 = 0;
    let mut total_unload_us: u64 = 0;
    let mut total_delta_mb:  f64 = 0.0;
    let mut slice_count:     usize = 0;
    let mut byte_total:      u64   = 0;

    for iter in 0..iterations {
        let rss_before = rss_mb();

        let t0 = Instant::now();
        let loaded = loader.load_layer(layer_idx)
            .unwrap_or_else(|e| { eprintln!("load_layer({layer_idx}) failed: {e}"); std::process::exit(1); });
        total_load_us += t0.elapsed().as_micros() as u64;

        let rss_after = rss_mb();
        total_delta_mb += (rss_after - rss_before).max(0.0);

        // Capture metadata from first iteration only.
        if iter == 0 {
            slice_count = loaded.slices.len();
            byte_total  = loaded.slices.iter().map(|(_, b)| b.len() as u64).sum();
        }

        // Touch a byte to force the OS to actually page in the mapping.
        let _ = loaded.slices.first().and_then(|(_, b)| b.first()).copied();

        let t1 = Instant::now();
        drop(loaded);
        total_unload_us += t1.elapsed().as_micros() as u64;
    }

    LayerSample {
        layer_idx,
        load_us:      total_load_us   / iterations as u64,
        unload_us:    total_unload_us / iterations as u64,
        rss_delta_mb: total_delta_mb  / iterations as f64,
        slice_count,
        byte_total,
    }
}

// ── Output helpers ────────────────────────────────────────────────────────────

fn print_text(samples: &[LayerSample], num_layers: usize, compare_full: bool) {
    println!("GwenLand Layer Loader Benchmark");
    println!("================================");
    println!("Total layers: {num_layers}");
    println!();
    println!("{:<6}  {:>10}  {:>12}  {:>14}  {:>8}  {:>10}",
        "Layer", "Load (µs)", "Unload (µs)", "RSS delta (MB)", "Slices", "Bytes");
    println!("{:-<6}  {:->10}  {:->12}  {:->14}  {:->8}  {:->10}",
        "", "", "", "", "", "");

    let mut peak_rss_mb = 0.0f64;
    for s in samples {
        peak_rss_mb = peak_rss_mb.max(s.rss_delta_mb);
        println!("{:<6}  {:>10}  {:>12}  {:>14.2}  {:>8}  {:>10}",
            s.layer_idx, s.load_us, s.unload_us, s.rss_delta_mb, s.slice_count, s.byte_total);
    }

    if compare_full && !samples.is_empty() {
        let avg_rss = samples.iter().map(|s| s.rss_delta_mb).sum::<f64>() / samples.len() as f64;
        let estimate_mb = avg_rss * num_layers as f64;
        println!();
        println!("Full-load estimate (avg_rss × num_layers): {estimate_mb:.2} MB");
        println!("Peak RSS observed across sampled layers:    {peak_rss_mb:.2} MB");
    }
}

fn print_json(samples: &[LayerSample], num_layers: usize, compare_full: bool) {
    let mut peak_rss_mb = 0.0f64;
    let mut layers_json = String::new();
    for (i, s) in samples.iter().enumerate() {
        peak_rss_mb = peak_rss_mb.max(s.rss_delta_mb);
        if i > 0 { layers_json.push(','); }
        layers_json.push_str(&format!(
            r#"{{"layer_idx":{idx},"load_us":{load},"unload_us":{unload},"rss_delta_mb":{delta:.4},"slice_count":{sc},"byte_total":{bt}}}"#,
            idx   = s.layer_idx,
            load  = s.load_us,
            unload= s.unload_us,
            delta = s.rss_delta_mb,
            sc    = s.slice_count,
            bt    = s.byte_total,
        ));
    }

    let full_load_str = if compare_full && !samples.is_empty() {
        let avg_rss = samples.iter().map(|s| s.rss_delta_mb).sum::<f64>() / samples.len() as f64;
        let estimate_mb = avg_rss * num_layers as f64;
        format!(r#","full_load_estimate_mb":{estimate_mb:.4},"peak_rss_mb":{peak_rss_mb:.4}"#)
    } else {
        String::new()
    };

    println!(
        r#"{{"benchmark":"bench_layer_loader","num_layers":{num_layers},"layers":[{layers_json}]{full_load_str}}}"#
    );
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    let args = match parse_args() {
        Ok(a)  => a,
        Err(e) => { eprintln!("Error: {e}"); std::process::exit(2); }
    };

    let loader = match LayerLoader::open(&args.gguf_path) {
        Ok(l)  => l,
        Err(e) => { eprintln!("Error opening '{}': {e}", args.gguf_path.display()); std::process::exit(1); }
    };

    let num_layers = loader.num_layers();
    if num_layers == 0 {
        eprintln!("Error: no model.layers.* tensors found in '{}'", args.gguf_path.display());
        std::process::exit(1);
    }

    // Determine which layers to sample.
    let layer_indices: Vec<usize> = match args.layer {
        Some(n) => {
            if n >= num_layers {
                eprintln!("Error: --layer {n} out of range (file has {num_layers} layers, indices 0..{})", num_layers - 1);
                std::process::exit(1);
            }
            vec![n]
        }
        None => (0..num_layers).collect(),
    };

    let samples: Vec<LayerSample> = layer_indices
        .iter()
        .map(|&idx| measure_layer(&loader, idx, args.iterations))
        .collect();

    match args.format {
        OutputFormat::Text => print_text(&samples, num_layers, args.compare_full),
        OutputFormat::Json => print_json(&samples, num_layers, args.compare_full),
    }
}

// ── Smoke tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn raw_args(args: &[&str]) -> Vec<String> {
        std::iter::once("bench_layer_loader")
            .chain(args.iter().copied())
            .map(String::from)
            .collect()
    }

    fn parse(args: &[&str]) -> Result<Args, String> {
        let raw = raw_args(args);
        // Replicate parse_args but from a Vec instead of std::env::args.
        if raw.len() < 2 {
            return Err("Usage: bench_layer_loader <gguf_path> ...".into());
        }
        let gguf_path = PathBuf::from(&raw[1]);
        let mut layer:        Option<usize> = None;
        let mut iterations:   usize        = 3;
        let mut compare_full: bool         = false;
        let mut format                     = OutputFormat::Text;
        let mut i = 2usize;
        while i < raw.len() {
            match raw[i].as_str() {
                "--layer" => {
                    i += 1;
                    layer = Some(raw[i].parse::<usize>().map_err(|_| "bad layer".to_string())?);
                }
                "--iterations" => {
                    i += 1;
                    let n = raw[i].parse::<usize>().map_err(|_| "bad iterations".to_string())?;
                    if n == 0 { return Err("--iterations must be >= 1".into()); }
                    iterations = n;
                }
                "--compare-full" => { compare_full = true; }
                "--format" => {
                    i += 1;
                    format = match raw[i].as_str() {
                        "text" => OutputFormat::Text,
                        "json" => OutputFormat::Json,
                        other  => return Err(format!("unknown format '{}'", other)),
                    };
                }
                other => return Err(format!("unknown argument '{}'", other)),
            }
            i += 1;
        }
        Ok(Args { gguf_path, layer, iterations, compare_full, format })
    }

    #[test]
    fn parse_minimal() {
        let a = parse(&["model.gguf"]).unwrap();
        assert_eq!(a.gguf_path, PathBuf::from("model.gguf"));
        assert!(a.layer.is_none());
        assert_eq!(a.iterations, 3);
        assert!(!a.compare_full);
        assert!(matches!(a.format, OutputFormat::Text));
    }

    #[test]
    fn parse_all_flags() {
        let a = parse(&["model.gguf", "--layer", "2", "--iterations", "5",
                         "--compare-full", "--format", "json"]).unwrap();
        assert_eq!(a.layer, Some(2));
        assert_eq!(a.iterations, 5);
        assert!(a.compare_full);
        assert!(matches!(a.format, OutputFormat::Json));
    }

    #[test]
    fn parse_zero_iterations_rejected() {
        assert!(parse(&["model.gguf", "--iterations", "0"]).is_err());
    }

    #[test]
    fn parse_unknown_flag_rejected() {
        assert!(parse(&["model.gguf", "--bogus"]).is_err());
    }

    #[test]
    fn parse_unknown_format_rejected() {
        assert!(parse(&["model.gguf", "--format", "xml"]).is_err());
    }
}
