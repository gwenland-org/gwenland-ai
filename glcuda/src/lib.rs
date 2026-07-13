//! # glcuda
//!
//! NVIDIA CUDA backend for GwenLand AI — the M2 SIMT engine specified by
//! `architecture/ArchGLML_X2.md`.
//!
//! Layer map (M2 milestone graph):
//! * [`ffi`] — CUDA Driver API, dynamically loaded (no link-time CUDA
//!   dependency; machines without the driver just report unavailable)
//! * [`driver`] — safe wrappers: device probe, primary context, PTX module
//!   JIT, memcpy, kernel launch
//! * [`buffer`] — the backend buffer: one `cuMemAlloc` at load, bump
//!   sub-allocation, zero allocation on the hot path (ADR-005)
//! * [`kernels`] — hand-authored PTX SIMT kernel suite + typed launchers
//! * [`dequant`] — host-side dequantization for model load (glproc-faithful
//!   Q4_K/Q5_0/Q6_K copies; ADR-001 duplication)
//! * [`loader`] / [`model`] — GGUF → host staging → single-allocation VRAM
//!   upload with the full footprint computed up front
//! * [`kv_cache`] — device KV cache, glproc's layout in VRAM (f32)
//! * [`runner`] — the static layer-graph walk, one stream, one sync/token
//! * [`sampler`] — engine-owned sampler (ADR-001 duplication of glproc's)

pub mod buffer;
pub mod cache;
pub mod dequant;
pub mod driver;
pub mod ffi;
pub mod kernels;
pub mod kv_cache;
pub mod loader;
pub mod model;
pub mod repack;
pub mod runner;
pub mod sampler;

use std::sync::Mutex;
use std::time::Instant;

use glcore::engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
use glcore::format::gguf::GgufFile;
use glcore::tokenizer::Tokenizer;
use glcore::GlError;

use driver::Cuda;
use kernels::KernelSet;
use model::GpuModel;
use sampler::{Sampler, SamplerConfig};

/// Engine-level knobs (mirror of glproc's config surface).
#[derive(Debug, Clone, Default)]
pub struct GlcudaConfig {
    /// Fixed RNG seed for reproducible sampling; `None` = time-seeded.
    pub seed: Option<u64>,
}

/// Optional per-token callback threaded through [`GlcudaEngine::run`].
type TokenSink<'a> = Option<&'a mut dyn FnMut(u32, &str)>;

/// The CUDA engine: device handle + JIT'd kernels after [`GlEngine::init`],
/// VRAM-resident model after [`GlEngine::load_model`].
#[derive(Default)]
pub struct GlcudaEngine {
    // Field order is drop order: everything holding CUDA resources (the model's
    // buffer + captured graph, and the kernel module) must drop BEFORE `cuda`
    // releases the context, or their destructors run against a dead context and
    // segfault. So `cuda` is declared LAST. (`shutdown()` also frees explicitly
    // in this order; this protects the implicit-Drop path.)
    /// Mutex because `GlEngine::infer` takes `&self` while a forward pass
    /// mutates the KV cursor and host staging buffers. One inference at a
    /// time per engine — the single-GPU invariant, enforced.
    model: Option<Mutex<GpuModel>>,
    kernels: Option<KernelSet>,
    tokenizer: Option<Tokenizer>,
    config: GlcudaConfig,
    /// Owns the CUDA context; released on drop. Declared last so it drops after
    /// everything above that holds CUDA resources.
    cuda: Option<Cuda>,
}

impl GlcudaEngine {
    /// Create an uninitialized engine. Cheap; hardware is touched in
    /// [`GlEngine::init`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an engine with an explicit configuration.
    pub fn with_config(config: GlcudaConfig) -> Self {
        GlcudaEngine { config, ..Self::default() }
    }

    /// The probed device, once initialized.
    pub fn cuda(&self) -> Option<&Cuda> {
        self.cuda.as_ref()
    }

    /// The loaded kernel suite, once initialized.
    pub fn kernels(&self) -> Option<&KernelSet> {
        self.kernels.as_ref()
    }

    /// Encode `text` to token ids with the loaded model's tokenizer, adding
    /// the BOS token. Available after [`GlEngine::load_model`]; the runtime
    /// normally tokenizes upstream, but front-ends and examples that hold
    /// only the engine need a way in.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, GlError> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| GlError::Engine("no tokenizer loaded — call load_model() first".into()))?;
        Ok(tok.encode(text, true))
    }

    /// Encode a chat turn using the model's chat template (ChatML for the
    /// Qwen/Llama-instruct families), falling back to plain [`Self::encode`]
    /// when the tokenizer defines no template.
    pub fn encode_chat(&self, user: &str) -> Result<Vec<u32>, GlError> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| GlError::Engine("no tokenizer loaded — call load_model() first".into()))?;
        Ok(tok.encode_chat(user).unwrap_or_else(|| tok.encode(user, true)))
    }

    fn sampler_for(&self, input: &InferInput) -> Sampler {
        Sampler::new(SamplerConfig {
            temperature: input.temperature,
            top_k: input.top_k,
            top_p: input.top_p,
            repeat_penalty: input.repeat_penalty,
            seed: self.config.seed,
        })
    }

    /// Run generation, invoking `on_token` per token when provided.
    fn run(&self, input: &InferInput, mut on_token: TokenSink<'_>) -> Result<InferOutput, GlError> {
        let cuda = self
            .cuda
            .as_ref()
            .ok_or_else(|| GlError::Engine("glcuda not initialized".into()))?;
        let kernels = self
            .kernels
            .as_ref()
            .ok_or_else(|| GlError::Engine("glcuda kernels not loaded".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| GlError::Engine("no model loaded".into()))?;
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| GlError::Engine("no tokenizer loaded".into()))?;

        let mut model = model
            .lock()
            .map_err(|_| GlError::Engine("model lock poisoned by an earlier panic".into()))?;
        let mut sampler = self.sampler_for(input);
        let started = Instant::now();
        let mut text = String::new();

        let (token_ids, timing) = model.generate(
            cuda,
            kernels,
            &input.token_ids,
            input.max_new_tokens,
            &mut sampler,
            |id| tokenizer.is_stop_token(id),
            |id| {
                let piece = tokenizer.decode_token_text(id);
                if let Some(cb) = on_token.as_deref_mut() {
                    cb(id, &piece);
                }
                text.push_str(&piece);
            },
        )?;

        let tokens_generated = token_ids.len();
        Ok(InferOutput {
            token_ids,
            text,
            tokens_generated,
            elapsed_ms: started.elapsed().as_millis() as u64,
            prompt_tokens: timing.prompt_tokens,
            prefill_ms: timing.prefill.as_secs_f64() * 1e3,
            generation_ms: timing.decode.as_secs_f64() * 1e3,
            // No per-token tracing on the CUDA path yet: logits live in device
            // memory, so capturing the raw distribution per token means a
            // device-to-host copy of the full vocabulary every step. Empty
            // means NOT captured — glbench reports no behavior signals for
            // this backend rather than fabricating them.
            traces: Vec::new(),
        })
    }
}

impl GlEngine for GlcudaEngine {
    fn init(&mut self) -> Result<(), GlError> {
        let cuda = Cuda::probe()?;
        let kernels = KernelSet::load(&cuda)?;
        let i = &cuda.info;
        // One startup line, like glproc's [simd] line: name the hardware
        // path so a silent mis-selection is visible.
        eprintln!(
            "[glcuda] device: {} | sm_{}{} | {} SMs | {:.1} GiB VRAM | driver {}.{}",
            i.name,
            i.sm_major,
            i.sm_minor,
            i.sm_count,
            i.total_mem as f64 / (1u64 << 30) as f64,
            i.driver_version / 1000,
            (i.driver_version % 1000) / 10,
        );
        self.cuda = Some(cuda);
        self.kernels = Some(kernels);
        Ok(())
    }

    fn load_model(&mut self, path: &str) -> Result<(), GlError> {
        let cuda = self
            .cuda
            .as_ref()
            .ok_or_else(|| GlError::Engine("glcuda not initialized — call init() first".into()))?;
        if path.to_ascii_lowercase().ends_with(".safetensors") {
            return Err(GlError::Engine(
                "glcuda loads GGUF only — safetensors needs a config.json sidecar".into(),
            ));
        }
        let gguf = GgufFile::open(path)?;
        let t_parse = Instant::now();
        self.tokenizer = Some(Tokenizer::from_gguf(&gguf)?);
        let parse_s = t_parse.elapsed().as_secs_f64();

        let t_stage = Instant::now();
        // Staging repacks Q8_0 -> SoA in parallel (see loader). The disk cache
        // is OPT-IN via GLCUDA_CACHE=1: it only pays off on fast local disk —
        // on slow/virtualized disk (e.g. Colab) writing the ~7.5 GB cache costs
        // far more than the repack it saves, so it is off by default.
        let host = if std::env::var_os("GLCUDA_CACHE").is_some() {
            cache::load_host_cached(path, || loader::load_host(&gguf))?
        } else {
            loader::load_host(&gguf)?
        };
        let stage_s = t_stage.elapsed().as_secs_f64();

        let t_upload = Instant::now();
        let gpu = GpuModel::upload(cuda, host)?;
        cuda.synchronize()?;
        eprintln!(
            "[glcuda] load: tokenizer {parse_s:.2}s | stage {stage_s:.2}s | \
             upload {:.2}s | {} MiB VRAM reserved",
            t_upload.elapsed().as_secs_f64(),
            gpu.total_vram_bytes >> 20,
        );
        self.model = Some(Mutex::new(gpu));
        Ok(())
    }

    fn infer(&self, input: InferInput) -> Result<InferOutput, GlError> {
        self.run(&input, None)
    }

    fn stream(
        &self,
        input: InferInput,
        on_token: &(dyn Fn(u32, &str) + Send),
    ) -> Result<InferOutput, GlError> {
        let mut forward = |id: u32, piece: &str| on_token(id, piece);
        self.run(&input, Some(&mut forward))
    }

    fn shutdown(&mut self) {
        // Free the model's VRAM explicitly while the context is still live,
        // then drop kernels (module) before the context they were JIT'd in.
        if let (Some(model), Some(cuda)) = (self.model.take(), self.cuda.as_ref()) {
            if let Ok(m) = model.into_inner() {
                let _ = m.free(cuda);
            }
        }
        self.tokenizer = None;
        self.kernels = None;
        self.cuda = None;
    }

    fn capabilities(&self) -> EngineSpec {
        EngineSpec {
            name: "glcuda",
            backend: "cuda",
            available: driver::cuda_available(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capabilities_match_hardware_reality() {
        let mut e = GlcudaEngine::new();
        let available = e.capabilities().available;
        if available {
            e.init().expect("driver reported available, init must succeed");
            assert!(e.cuda().is_some());
            assert!(e.kernels().is_some());
            e.shutdown();
            assert!(e.cuda().is_none());
        } else {
            assert!(e.init().is_err(), "init must fail cleanly without a CUDA device");
        }
    }

    #[test]
    fn model_calls_before_init_error_cleanly() {
        let mut e = GlcudaEngine::new();
        assert!(e.load_model("x.gguf").is_err());
        assert!(e.infer(InferInput::default()).is_err());
    }
}
