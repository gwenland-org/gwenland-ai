use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::train::LayerLoader;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLoadSample {
    pub layer_idx:    usize,
    pub load_us:      u64,
    pub unload_us:    u64,
    pub rss_delta_mb: f64,
    pub byte_total:   u64,
    pub slice_count:  usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerLoadResult {
    pub samples:               Vec<LayerLoadSample>,
    pub file_size_bytes:       u64,
    pub num_layers:            usize,
    pub min_load_us:           u64,
    pub max_load_us:           u64,
    pub mean_load_us:          f64,
    pub peak_rss_mb:           f64,
    pub full_load_estimate_mb: f64,
}

pub fn run_layer_load_bench(
    path: &Path,
    sample_layers: Option<usize>,
) -> Result<LayerLoadResult> {
    let loader = LayerLoader::open(path)?;
    let num_layers = loader.num_layers();
    let file_size_bytes = std::fs::metadata(path)?.len();

    let indices: Vec<usize> = match sample_layers {
        Some(k) if k < num_layers => (0..k).map(|i| i * num_layers / k).collect(),
        _ => (0..num_layers).collect(),
    };

    let mut samples = Vec::with_capacity(indices.len());
    let mut peak_rss_mb: f64 = 0.0;

    for layer_idx in &indices {
        let rss_before = sample_rss_mb();

        let t_load = std::time::Instant::now();
        let loaded = loader.load_layer(*layer_idx)?;
        let load_us = t_load.elapsed().as_micros() as u64;

        let rss_after = sample_rss_mb();
        let rss_delta_mb = (rss_after - rss_before).max(0.0);
        peak_rss_mb = peak_rss_mb.max(rss_after);

        let byte_total: u64 = loaded.slices.iter().map(|(_, b)| b.len() as u64).sum();
        let slice_count = loaded.slices.len();

        let t_unload = std::time::Instant::now();
        loaded.unload();
        let unload_us = t_unload.elapsed().as_micros() as u64;

        samples.push(LayerLoadSample {
            layer_idx: *layer_idx,
            load_us,
            unload_us,
            rss_delta_mb,
            byte_total,
            slice_count,
        });
    }

    let (min_load_us, max_load_us, mean_load_us) = if samples.is_empty() {
        (0, 0, 0.0)
    } else {
        let min = samples.iter().map(|s| s.load_us).min().unwrap();
        let max = samples.iter().map(|s| s.load_us).max().unwrap();
        let mean = samples.iter().map(|s| s.load_us as f64).sum::<f64>() / samples.len() as f64;
        (min, max, mean)
    };

    let full_load_estimate_mb = peak_rss_mb * num_layers as f64;

    Ok(LayerLoadResult {
        samples,
        file_size_bytes,
        num_layers,
        min_load_us,
        max_load_us,
        mean_load_us,
        peak_rss_mb,
        full_load_estimate_mb,
    })
}

fn sample_rss_mb() -> f64 {
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
        if let Some(proc) = sys.process(pid) {
            return proc.memory() as f64 / (1024.0 * 1024.0);
        }
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_layer_load_bench_invalid_path() {
        let result = run_layer_load_bench(Path::new("nonexistent_file.gguf"), None);
        assert!(result.is_err());
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[test]
    fn test_run_layer_load_bench_sample_layers_subset() {
        use crate::train::layer_loader::write_minimal_gguf_pub;

        let data = vec![0u8; 64];
        let tmp = write_minimal_gguf_pub(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.1.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.2.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let result = run_layer_load_bench(tmp.path(), Some(2)).unwrap();
        assert_eq!(result.samples.len(), 2);
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[test]
    fn test_run_layer_load_bench_all_layers() {
        use crate::train::layer_loader::write_minimal_gguf_pub;

        let data = vec![0u8; 64];
        let tmp = write_minimal_gguf_pub(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.1.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.2.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let result = run_layer_load_bench(tmp.path(), None).unwrap();
        assert_eq!(result.samples.len(), 3);
    }

    #[cfg(any(test, feature = "test-utils"))]
    #[test]
    fn test_full_load_estimate_formula() {
        use crate::train::layer_loader::write_minimal_gguf_pub;

        let data = vec![0u8; 64];
        let tmp = write_minimal_gguf_pub(&[
            ("model.layers.0.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.1.self_attn.q_proj.weight", data.as_slice()),
            ("model.layers.2.self_attn.q_proj.weight", data.as_slice()),
        ]);
        let result = run_layer_load_bench(tmp.path(), None).unwrap();
        let expected = result.peak_rss_mb * result.num_layers as f64;
        assert!((result.full_load_estimate_mb - expected).abs() < f64::EPSILON);
    }
}
