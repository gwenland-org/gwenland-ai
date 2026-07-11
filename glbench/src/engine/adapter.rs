//! The engine adapter — glbench's single boundary to the engines.
//!
//! glbench does not implement inference (DESIGN.md, ENGINE EXECUTION MODEL). It
//! runs everything through glcore's [`Runtime`], which owns tokenization and
//! holds one `Box<dyn GlEngine>`. This adapter's job is only: pick the engine
//! named in the workload, load the model, run a request, and translate the
//! engine's [`InferOutput`] into glbench's raw [`IterationMetrics`]. No timing
//! policy, no analysis — it hands back facts.

use glcore::engine_trait::{GlEngine, InferInput, InferOutput};
use glcore::runtime::Runtime;
use glcore::GlError;

use crate::core::metrics::IterationMetrics;
use crate::core::workload::WorkloadSpec;
use crate::engine::metadata::EngineMetadata;
use crate::environment::gpu::GpuInfo;

/// The set of engine names glbench can construct. Kept explicit rather than a
/// registry so an unavailable backend is a clear error, not a silent fallback.
pub const KNOWN_ENGINES: &[&str] = &["glproc", "glcuda"];

/// A loaded engine ready to run workloads, plus the facts it reported about
/// itself and its device.
pub struct EngineAdapter {
    runtime: Runtime,
    metadata: EngineMetadata,
    gpu: GpuInfo,
}

impl EngineAdapter {
    /// Construct the engine named in `spec`, initialize it, and load the model.
    ///
    /// The engine is created here (the only place glbench names concrete engine
    /// types); everything after goes through the [`Runtime`] trait object.
    pub fn load(spec: &WorkloadSpec) -> Result<EngineAdapter, GlError> {
        let (engine, gpu) = build_engine(&spec.engine)?;
        let spec_meta = engine.capabilities();
        let metadata = EngineMetadata {
            name: spec_meta.name.to_string(),
            backend: spec_meta.backend.to_string(),
            available: spec_meta.available,
            model_arch: None,
            quantization: None,
        };

        let mut runtime = Runtime::new(engine)?;
        runtime.load(&spec.model_path)?;

        Ok(EngineAdapter { runtime, metadata, gpu })
    }

    /// The engine facts (name/backend/availability). Model arch/quant are filled
    /// by the runner from the GGUF header where available.
    pub fn metadata(&self) -> &EngineMetadata {
        &self.metadata
    }

    /// GPU facts reported by the engine's backend probe (empty for CPU engines).
    pub fn gpu(&self) -> &GpuInfo {
        &self.gpu
    }

    /// Run one inference request for `spec`'s prompt and token budget, returning
    /// the raw per-iteration facts. This is a thin pass-through to the engine —
    /// the engine already separates prefill from decode timing in its output.
    pub fn run_once(&self, spec: &WorkloadSpec) -> Result<IterationMetrics, GlError> {
        let out = self.run_request(spec)?;
        Ok(IterationMetrics {
            prompt_tokens: out.prompt_tokens as u64,
            generated_tokens: out.tokens_generated as u64,
            prefill_ms: out.prefill_ms,
            decode_ms: out.generation_ms,
            total_ms: out.elapsed_ms as f64,
        })
    }

    /// Run a request and return the generated token ids — used by numerical
    /// validation against the glproc oracle.
    pub fn run_tokens(&self, spec: &WorkloadSpec) -> Result<Vec<u32>, GlError> {
        Ok(self.run_request(spec)?.token_ids)
    }

    fn run_request(&self, spec: &WorkloadSpec) -> Result<InferOutput, GlError> {
        let config = InferInput {
            token_ids: Vec::new(), // Runtime fills these from the prompt
            max_new_tokens: spec.max_new_tokens,
            temperature: spec.temperature,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
        };
        // A no-op sink: glbench measures, it does not consume the text stream.
        self.runtime.stream(&spec.prompt, config, |_text| {})
    }
}

/// Build the named engine as a trait object, plus any GPU facts it exposes.
///
/// This is deliberately the *only* function that names concrete engine types.
/// Adding a backend (glvulkan, glmetal) means one arm here — nothing else in
/// glbench changes.
fn build_engine(name: &str) -> Result<(Box<dyn GlEngine>, GpuInfo), GlError> {
    match name {
        "glproc" => Ok((Box::new(glproc::GlprocEngine::new()), GpuInfo::default())),
        "glcuda" => {
            // The CUDA engine self-probes at init(); reflect its device facts
            // into a GpuInfo (with a published-ceiling lookup) so the analysis
            // layer has a bandwidth ceiling to compare against.
            let gpu = probe_cuda_gpu();
            Ok((Box::new(glcuda::GlcudaEngine::new()), gpu))
        }
        other => Err(GlError::Engine(format!(
            "unknown engine '{other}' (known: {})",
            KNOWN_ENGINES.join(", ")
        ))),
    }
}

/// Probe the CUDA device for its name/compute/memory and attach a published
/// bandwidth ceiling if the device is in the capability table. Returns an empty
/// [`GpuInfo`] if no CUDA device is present.
fn probe_cuda_gpu() -> GpuInfo {
    use crate::engine::capability;

    match glcuda::driver::Cuda::probe() {
        Ok(cuda) => {
            let i = &cuda.info;
            let ceiling = capability::lookup(&i.name);
            GpuInfo {
                name: Some(i.name.clone()),
                backend: Some("cuda".to_string()),
                compute: Some(format!("sm_{}{}", i.sm_major, i.sm_minor)),
                total_memory_bytes: Some(i.total_mem as u64),
                peak_bandwidth_gbs: ceiling.map(|c| c.peak_bandwidth_gbs),
                peak_compute_tops: ceiling.map(|c| c.peak_int8_tops),
            }
        }
        Err(_) => GpuInfo::default(),
    }
}
