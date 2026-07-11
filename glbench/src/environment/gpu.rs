//! GPU facts.
//!
//! glbench does not link a GPU SDK (dependency rule). It learns GPU facts the
//! same way the engines do: by asking the engine adapter, which owns the
//! backend. This module is the *shape* of what an engine reports; the adapter
//! fills it in from the engine's own device probe (e.g. glcuda's `Cuda::probe`).
//! On a CPU-only run this stays empty.

/// Observed accelerator facts, as reported by the active engine's backend.
#[derive(Debug, Clone, Default)]
pub struct GpuInfo {
    /// Device name, e.g. `"Tesla T4"`.
    pub name: Option<String>,
    /// Backend kind: `"cuda"`, `"vulkan"`, `"metal"`.
    pub backend: Option<String>,
    /// Compute capability / arch string, e.g. `"sm_75"`.
    pub compute: Option<String>,
    /// Total device memory in bytes.
    pub total_memory_bytes: Option<u64>,
    /// Theoretical peak memory bandwidth in GB/s, if known for the device.
    /// This is a *hardware capability*, used later as a ceiling — not a
    /// measured value.
    pub peak_bandwidth_gbs: Option<f64>,
    /// Theoretical peak INT8/FP16 compute in TOPS/TFLOPS, if known.
    pub peak_compute_tops: Option<f64>,
}

impl GpuInfo {
    /// True if any GPU fact was reported (i.e. a GPU engine is active).
    pub fn is_present(&self) -> bool {
        self.name.is_some()
    }
}
