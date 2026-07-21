//! The engine adapter — glbench's single boundary to the engines.
//!
//! glbench does not implement inference (DESIGN.md, ENGINE EXECUTION MODEL). For
//! glproc/glcuda it runs everything through glcore's [`Runtime`], which owns
//! tokenization and holds one `Box<dyn GlEngine>`. This adapter's job is only:
//! pick the engine named in the workload, load the model, run a request, and
//! translate the engine's [`InferOutput`] into glbench's raw [`IterationMetrics`].
//! No timing policy, no analysis — it hands back facts.
//!
//! ## The GLLM exception
//!
//! `Runtime::load` always builds a tokenizer (GGUF metadata or a sibling
//! `tokenizer.json`) and always rejects a path that isn't `.gguf`/
//! `.safetensors`. A GLLM package is a *directory* with no embedded
//! tokenizer at all (ARTX1 OQ3's `GLLMTokenizer.gllm` unit is decided but
//! not yet emitted by the converter), so it cannot go through `Runtime`.
//! [`EngineAdapter`] therefore holds either a `Runtime` (every engine that
//! has a real tokenizer) or a bare `Box<dyn GlEngine>` plus synthesized
//! token ids (GLLM only) — see [`Backend`].

use glcore::engine_trait::{GlEngine, InferInput, InferOutput};
use glcore::runtime::Runtime;
use glcore::GlError;

use crate::core::metrics::IterationMetrics;
use crate::core::workload::WorkloadSpec;
use crate::engine::metadata::EngineMetadata;
use crate::environment::gpu::GpuInfo;

/// The set of engine names glbench can construct. Kept explicit rather than a
/// registry so an unavailable backend is a clear error, not a silent fallback.
pub const KNOWN_ENGINES: &[&str] = &["glproc", "glcuda", "gllm"];

/// How a loaded engine is actually driven — the two shapes described in the
/// module docs.
///
/// `RawTokens` is only ever constructed behind `gllm-bench` (see
/// [`EngineAdapter::load_gllm`]) — without that feature there is no
/// tokenizer-less engine to name, so the variant is legitimately unused
/// rather than a bug; `#[allow(dead_code)]` says so instead of complicating
/// every match arm below with a second `cfg`.
#[cfg_attr(not(feature = "gllm-bench"), allow(dead_code))]
enum Backend {
    /// A real tokenizer exists; prompts are encoded normally. Boxed: `Runtime`
    /// is far larger than `RawTokens` (which is already a `Box<dyn GlEngine>`
    /// plus a small `Option<usize>`), and an unboxed variant would size every
    /// `Backend` — including every `RawTokens` — to the bigger one.
    Tokenized(Box<Runtime>),
    /// No embedded tokenizer (GLLM's `.gllm` package format carries none —
    /// see the module docs). When `spec.tokenizer_path` names a source
    /// `.gguf` file, `tokenizer` holds a real [`GllmTokenizer`] built from
    /// its metadata and `--prompt`'s TEXT is actually encoded. When absent
    /// (the pre-ARTX-OQ3 default), token ids are synthesized deterministically
    /// from the workload's seed and the model's real vocabulary size — see
    /// [`synthetic_token_ids`] — never a stand-in for a real prompt, since
    /// that path benchmarks throughput and correctness-of-computation, not
    /// prompt-conditioned generation quality.
    ///
    /// `tokenizer` is boxed for the same reason `Tokenized` is: `GllmTokenizer`
    /// carries its full vocab/merge tables by value, which would otherwise
    /// make it (not `Tokenized(Box<Runtime>)`) the variant that sizes the
    /// whole enum for every `Backend`, including ones that never load GLLM.
    RawTokens {
        engine: Box<dyn GlEngine>,
        vocab_size: Option<usize>,
        #[cfg(feature = "gllm-bench")]
        tokenizer: Box<Option<glictus_caliburni::GllmTokenizer>>,
    },
}

/// A loaded engine ready to run workloads, plus the facts it reported about
/// itself and its device.
pub struct EngineAdapter {
    backend: Backend,
    metadata: EngineMetadata,
    gpu: GpuInfo,
}

impl EngineAdapter {
    /// Construct the engine named in `spec`, initialize it, and load the model.
    ///
    /// The engine is created here (the only place glbench names concrete engine
    /// types); everything after goes through [`Backend`].
    pub fn load(spec: &WorkloadSpec) -> Result<EngineAdapter, GlError> {
        if spec.engine == "gllm" {
            return Self::load_gllm(spec);
        }

        let (engine, gpu) = build_engine(&spec.engine)?;
        let spec_meta = engine.capabilities();
        let metadata = EngineMetadata {
            name: spec_meta.name.to_string(),
            backend: spec_meta.backend.to_string(),
            available: spec_meta.available,
            model_arch: None,
            quantization: None,
            thinking_capable: None,
        };

        let mut runtime = Runtime::new(engine)?;
        runtime.load(&spec.model_path)?;

        Ok(EngineAdapter { backend: Backend::Tokenized(Box::new(runtime)), metadata, gpu })
    }

    /// The GLLM path: no `Runtime`, no embedded tokenizer — see the module
    /// docs. `spec.model_path` is a GLLM package root (a directory, or the
    /// `gllm.json` inside one). `spec.tokenizer_path`, when given, names the
    /// source `.gguf` file to build a real [`glictus_caliburni::GllmTokenizer`]
    /// from — the `.gllm` package itself carries no tokenizer metadata
    /// (ARTX1 OQ3; `glictus_caliburni::converter` drops it on conversion).
    #[cfg(feature = "gllm-bench")]
    fn load_gllm(spec: &WorkloadSpec) -> Result<EngineAdapter, GlError> {
        let mut engine = glictus_caliburni::runtime::GllmEngine::new();
        engine.init()?;
        engine.load_model(&spec.model_path)?;
        let spec_meta = engine.capabilities();
        let vocab_size = engine.vocab_size();

        let tokenizer = match &spec.tokenizer_path {
            Some(path) => Some(
                glictus_caliburni::GllmTokenizer::from_gguf(std::path::Path::new(path))
                    .map_err(|e| GlError::Parse(format!("loading tokenizer from {path}: {e}")))?,
            ),
            None => None,
        };

        let metadata = EngineMetadata {
            name: spec_meta.name.to_string(),
            backend: spec_meta.backend.to_string(),
            available: spec_meta.available,
            model_arch: Some("gllm".to_string()),
            quantization: None,
            thinking_capable: None,
        };
        Ok(EngineAdapter {
            backend: Backend::RawTokens { engine: Box::new(engine), vocab_size, tokenizer: Box::new(tokenizer) },
            metadata,
            gpu: GpuInfo::default(),
        })
    }

    #[cfg(not(feature = "gllm-bench"))]
    fn load_gllm(_spec: &WorkloadSpec) -> Result<EngineAdapter, GlError> {
        Err(GlError::Engine(
            "engine 'gllm' needs glbench built with --features gllm-bench".into(),
        ))
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

    /// Per-stage telemetry from the engine's most recent run, or `None` if it
    /// collects none (profiling off, or a backend that reports nothing).
    ///
    /// A pull, not a callback: glbench asks the engine for facts it already
    /// keeps. The engine never learns that a benchmark harness is what asked —
    /// which is what keeps `glbench observes, it does not control` true in code
    /// rather than only in DESIGN.md.
    pub fn telemetry(&self) -> Option<glcore::telemetry::EngineTelemetry> {
        match &self.backend {
            Backend::Tokenized(rt) => rt.telemetry(),
            Backend::RawTokens { engine, .. } => engine.telemetry(),
        }
    }

    /// Run one inference request for `spec`'s prompt and token budget, returning
    /// the raw per-iteration facts. This is a thin pass-through to the engine —
    /// the engine already separates prefill from decode timing in its output.
    pub fn run_once(&self, spec: &WorkloadSpec) -> Result<IterationMetrics, GlError> {
        // Tracing OFF: this is the timed path, and tracing costs an O(vocab)
        // sweep per token. Taxing the measurement with the instrument is
        // exactly the mistake this split exists to avoid.
        let out = self.run_request(spec, false)?;
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
        Ok(self.run_request(spec, false)?.token_ids)
    }

    /// Run once **with per-token tracing on**, for the behavioral signals.
    ///
    /// This is a separate run, deliberately not one of the measured iterations:
    /// tracing costs an O(vocab) sweep plus a logits copy per token, so folding
    /// it into the timed loop would tax the very throughput being reported. The
    /// behavior of the model does not change — the same tokens come out — but
    /// the clock does, so the two must not share a run.
    ///
    /// Returns the generated tokens and their traces.
    pub fn run_traced(
        &self,
        spec: &WorkloadSpec,
    ) -> Result<(Vec<u32>, Vec<glcore::trace::TokenTrace>), GlError> {
        let out = self.run_request(spec, true)?;
        Ok((out.token_ids, out.traces))
    }

    fn run_request(&self, spec: &WorkloadSpec, trace: bool) -> Result<InferOutput, GlError> {
        let config = InferInput {
            token_ids: Vec::new(), // filled below, per backend
            max_new_tokens: spec.max_new_tokens,
            temperature: spec.temperature,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            trace: if trace {
                glcore::trace::TraceConfig::on()
            } else {
                glcore::trace::TraceConfig::default()
            },
        };
        match &self.backend {
            Backend::Tokenized(rt) => {
                // A no-op sink: glbench measures, it does not consume the text stream.
                rt.stream(&spec.prompt, config, |_text| {})
            }
            #[cfg(feature = "gllm-bench")]
            Backend::RawTokens { engine, vocab_size, tokenizer } => {
                let mut config = config;
                config.token_ids = match &**tokenizer {
                    // No add_eos: this is a prefill prompt the engine then
                    // continues decoding past, not a complete turn — forcing
                    // an end-of-text token into the prompt would be measuring
                    // a different (and stranger) workload than --prompt asked
                    // for. No add_bos either: Qwen2/Qwen3 never default it on
                    // (GllmTokenizer::add_bos_default() is false for them),
                    // and glbench has no reason to override the model's own
                    // convention just because it's benchmarking.
                    Some(tok) => tok.encode(
                        &spec.prompt,
                        &glictus_caliburni::TokenizerConfig { add_bos: false, add_eos: false },
                    ),
                    None => synthetic_token_ids(spec, *vocab_size),
                };
                engine.stream(config, &|_id, _text| {})
            }
            #[cfg(not(feature = "gllm-bench"))]
            Backend::RawTokens { engine, vocab_size } => {
                let mut config = config;
                config.token_ids = synthetic_token_ids(spec, *vocab_size);
                engine.stream(config, &|_id, _text| {})
            }
        }
    }
}

/// Synthesize a deterministic prompt of token ids for engines with no
/// tokenizer (GLLM). Length is derived from `spec.prompt`'s word count when
/// a prompt was given (so `--prompt` still controls *how much* prefill work
/// happens, even though its *text* cannot be encoded), defaulting to 32
/// tokens otherwise. Ids are generated from `spec.seed` with a tiny
/// xorshift so a fixed seed reproduces the identical synthetic prompt run
/// to run — never random, since an unreproducible input would make every
/// downstream comparison meaningless.
fn synthetic_token_ids(spec: &WorkloadSpec, vocab_size_hint: Option<usize>) -> Vec<u32> {
    let len = if spec.prompt.is_empty() {
        32
    } else {
        spec.prompt.split_whitespace().count().max(1)
    };
    // A vocabulary size this small is common even for tiny reference models;
    // GllmEngine rejects any id that turns out to be out of range for the
    // model actually loaded, so a hint this conservative only under-fills
    // the id space, it never produces an invalid token.
    let vocab = vocab_size_hint.unwrap_or(32_000).max(1) as u64;
    let mut state = spec.seed.max(1);
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state % vocab) as u32
        })
        .collect()
}

/// Build the named engine as a trait object, plus any GPU facts it exposes.
///
/// This is deliberately the *only* function that names concrete engine types
/// for the `Runtime`-backed path (`gllm` is handled separately in
/// [`EngineAdapter::load_gllm`], since it does not go through `Runtime`).
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

#[cfg(test)]
mod tests {
    use super::*;

    fn spec_with(prompt: &str, seed: u64) -> WorkloadSpec {
        WorkloadSpec { prompt: prompt.to_string(), seed, ..WorkloadSpec::default() }
    }

    #[test]
    fn synthetic_ids_are_deterministic_for_a_fixed_seed() {
        let a = synthetic_token_ids(&spec_with("hello world", 7), Some(1000));
        let b = synthetic_token_ids(&spec_with("hello world", 7), Some(1000));
        assert_eq!(a, b);
    }

    #[test]
    fn synthetic_ids_change_with_the_seed() {
        let a = synthetic_token_ids(&spec_with("hello world", 1), Some(1000));
        let b = synthetic_token_ids(&spec_with("hello world", 2), Some(1000));
        assert_ne!(a, b);
    }

    #[test]
    fn synthetic_id_count_follows_the_prompt_word_count() {
        let ids = synthetic_token_ids(&spec_with("one two three four", 1), Some(1000));
        assert_eq!(ids.len(), 4);
    }

    #[test]
    fn an_empty_prompt_still_yields_a_default_length() {
        let ids = synthetic_token_ids(&spec_with("", 1), Some(1000));
        assert_eq!(ids.len(), 32);
    }

    #[test]
    fn synthetic_ids_stay_within_the_vocabulary_hint() {
        let ids = synthetic_token_ids(&spec_with("a b c d e f g h i j", 99), Some(50));
        assert!(ids.iter().all(|&id| (id as usize) < 50), "{ids:?}");
    }

    #[test]
    fn known_engines_lists_gllm() {
        assert!(KNOWN_ENGINES.contains(&"gllm"));
    }
}
