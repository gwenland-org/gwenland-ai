//! `GllmEngine` — a token-level [`GlEngine`] over [`GllmRuntime`] +
//! [`GlprocBackend`] (ARTX11 / glbench GLLM integration).
//!
//! [`GllmRuntime`] is layer-only: it forwards one activation through every
//! layer and knows nothing about tokens (see `runtime/mod.rs`'s module
//! docs). Turning that into "tokens in, tokens out" needs three things
//! `GllmRuntime` deliberately does not own — embedding lookup, the final
//! norm + LM head, and a sampling loop — so this module supplies exactly
//! those three, and delegates everything else (every layer's attention and
//! FFN) to the already-verified `GlprocBackend` path from ARTX10 Wave 1.
//!
//! ## Why this exists
//!
//! glbench's only boundary to an engine is [`glcore::engine_trait::GlEngine`]
//! (`glcore::runtime::Runtime` wraps exactly one `Box<dyn GlEngine>` and
//! feeds it token ids). Nothing before this implemented that trait for
//! GLLM, so a GLLM package was unbenchmarkable. This is the one adapter that
//! makes it benchmarkable — it does not introduce a second inference path
//! for glictus-caliburni to maintain: the layer math is entirely
//! `GlprocBackend`, unchanged.
//!
//! ## What is real and what is not
//!
//! Real: every layer's attention/FFN math (`GlprocBackend`, ARTX10 Wave 1,
//! already verified end-to-end), the embedding lookup and LM head (plain
//! `glproc::kernels::matvec` against the shared component's real tensors),
//! and sampling (`glproc::sampler::Sampler`, the same sampler `GlprocEngine`
//! uses).
//!
//! Not real: **tokenization**. GLLM packages do not yet embed a tokenizer
//! (ARTX1 OQ3 decided the design — a `GLLMTokenizer.gllm` unit — but the
//! converter does not emit it yet, see `converter.rs`'s
//! `tokenizer metadata present in GGUF but NOT packaged` note). So
//! [`GlEngine::load_model`] takes a package *directory*, not a `.gguf`/
//! `.safetensors` file, and [`GlEngine::infer`]/[`stream`](GlEngine::stream)
//! consume `InferInput::token_ids` directly — there is no prompt-text path.
//! A caller that wants a specific prompt must already have token ids for
//! this model's vocabulary (glbench's `gllm` engine path synthesizes them,
//! since there is no tokenizer to encode real text with yet).

use std::sync::Arc;

use glcore::engine_trait::{EngineSpec, GlEngine, InferInput, InferOutput};
use glcore::GlError;

use crate::manifest::ModelMetadata;
use crate::runtime::backend::ExecutionBackend;
use crate::runtime::glproc_backend::GlprocBackend;
use crate::runtime::mmap::LayerMapping;
use crate::runtime::runtime::GllmRuntime;
use crate::runtime::types::{ActivationBuffer, RuntimeConfig};

/// Canonical shared-tensor names the real ARTX07 converter writes (see
/// `shared.rs`'s `SHARED_*` constants and `converter.rs`'s `map_tensor_name`).
/// Duplicated as `&str` rather than imported because those constants live in
/// a module (`shared.rs`) that only describes the *header*, not tensor
/// bytes — the actual read path here goes through `LayerMapping`, same as
/// every layer file.
const TOKEN_EMBEDDINGS: &str = "token_embeddings";
const OUTPUT_NORM: &str = "output_norm.weight";
const OUTPUT_HEAD: &str = "output_head.weight";

/// RMSNorm epsilon — mirrors [`super::glproc_backend`]'s constant of the
/// same value; ARTX03's `ModelMetadata` carries no epsilon field.
const RMS_EPS: f32 = 1e-5;

/// A loaded GLLM package driven as a [`GlEngine`].
///
/// Holds the embedding table and LM head **decoded to f32 in RAM** — the
/// same simplification `GlprocBackend`'s per-layer tensors accept (Wave 1
/// scope: F32 only, no quantized shared tensors yet). For the small
/// reference models this targets that is a few hundred KB to a few MB, not
/// the multi-GB weight class quantization exists to avoid.
pub struct GllmEngine {
    package_root: Option<std::path::PathBuf>,
    model: Option<LoadedModel>,
}

struct LoadedModel {
    metadata: ModelMetadata,
    embed: Vec<f32>,    // [vocab_size, dim]
    output_norm: Vec<f32>, // [dim]
    lm_head: Vec<f32>,  // [vocab_size, dim] (tied to `embed` when the package has no separate head)
    dim: usize,
    vocab_size: usize,
}

impl Default for GllmEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GllmEngine {
    /// Construct an unloaded engine.
    pub fn new() -> Self {
        GllmEngine { package_root: None, model: None }
    }

    fn model(&self) -> Result<&LoadedModel, GlError> {
        self.model.as_ref().ok_or_else(|| GlError::Engine("no model loaded".into()))
    }

    /// The loaded model's vocabulary size, if a model is loaded.
    ///
    /// Exposed so a caller synthesizing token ids (glbench's `gllm` engine
    /// path has no tokenizer to draw ids from) can stay in the real
    /// vocabulary's range instead of guessing one.
    pub fn vocab_size(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.vocab_size)
    }

    /// Read the shared component's embedding table, final norm, and LM head
    /// (falling back to tied embeddings when no separate head tensor
    /// exists) — the only tensors a `GllmRuntime` forward pass does not
    /// already handle.
    fn load_shared(root: &std::path::Path) -> Result<LoadedModel, GlError> {
        let package = crate::package::GllmPackage::open(root)
            .map_err(|e| GlError::Parse(format!("opening GLLM package: {e}")))?;
        let metadata = package.manifest().metadata.clone();
        let dim = metadata.embedding_length as usize;
        let vocab_size = metadata.vocab_size as usize;

        let shared_path = &package.layout.shared_path;
        let mapping = LayerMapping::open(shared_path, None, false)
            .map_err(|e| GlError::Parse(format!("mapping shared component: {e}")))?;

        let decode = |name: &str| -> Result<Vec<f32>, GlError> {
            let entry = package.manifest().shared.tensor(name).ok_or_else(|| {
                GlError::Parse(format!("shared component is missing tensor {name:?}"))
            })?;
            let bytes = mapping.tensor_bytes(name).ok_or_else(|| {
                GlError::Parse(format!("shared component has no data for tensor {name:?}"))
            })?;
            let dtype_str = serde_json::to_value(entry.dtype)
                .ok()
                .and_then(|v| v.as_str().map(str::to_string))
                .unwrap_or_default();
            glcore::format::decode_tensor(name, &entry.shape, &dtype_str, bytes)
                .map(|t| t.data)
                .map_err(|e| GlError::Parse(format!("tensor {name:?}: {e}")))
        };

        let embed = decode(TOKEN_EMBEDDINGS)?;
        let output_norm = decode(OUTPUT_NORM)?;
        // Tied embeddings (common on small models): no separate LM head
        // tensor means the embedding table doubles as the output projection.
        let lm_head = match decode(OUTPUT_HEAD) {
            Ok(w) => w,
            Err(_) => embed.clone(),
        };

        if embed.len() != vocab_size * dim {
            return Err(GlError::Parse(format!(
                "token_embeddings has {} elements, expected vocab_size*dim = {}",
                embed.len(),
                vocab_size * dim
            )));
        }

        Ok(LoadedModel { metadata, embed, output_norm, lm_head, dim, vocab_size })
    }

    /// One token's embedding row, `[dim]`.
    fn embed_token(model: &LoadedModel, token: u32) -> Result<Vec<f32>, GlError> {
        let row = token as usize;
        if row >= model.vocab_size {
            return Err(GlError::Engine(format!(
                "token id {token} out of vocabulary range ({})",
                model.vocab_size
            )));
        }
        Ok(model.embed[row * model.dim..(row + 1) * model.dim].to_vec())
    }

    /// Final norm + LM head over a hidden state, `[dim] -> [vocab_size]` logits.
    fn logits_for(model: &LoadedModel, hidden: &[f32]) -> Vec<f32> {
        let mut normed = vec![0.0f32; model.dim];
        glproc::kernels::rms_norm_into(hidden, &model.output_norm, RMS_EPS, &mut normed);
        let mut logits = vec![0.0f32; model.vocab_size];
        glproc::kernels::matvec(&model.lm_head, &normed, &mut logits, model.vocab_size, model.dim);
        logits
    }

    /// Run one full request: prefill every prompt token, then decode up to
    /// `max_new_tokens`, sampling with `config`'s hyperparameters.
    ///
    /// `on_token` is called once per generated token (prompt tokens are
    /// never replayed to it — same contract as [`GlEngine::stream`]).
    fn run_request(
        &self,
        config: &InferInput,
        mut on_token: impl FnMut(u32),
    ) -> Result<InferOutput, GlError> {
        let model = self.model()?;
        let root = self.package_root.as_ref().ok_or_else(|| GlError::Engine("no model loaded".into()))?;

        let backend: Arc<dyn ExecutionBackend> = Arc::new(
            GlprocBackend::new(&model.metadata)
                .map_err(|e| GlError::Engine(format!("building glproc backend: {e}")))?,
        );
        let mut runtime = GllmRuntime::open(root, RuntimeConfig::default(), backend)
            .map_err(|e| GlError::Engine(format!("opening GLLM runtime: {e}")))?;

        let mut sampler = glproc::sampler::Sampler::new(glproc::sampler::SamplerConfig {
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            repeat_penalty: config.repeat_penalty,
            seed: Some(42), // fixed: benchmark determinism, the whole reason glbench pins temperature/seed
        });

        let started = std::time::Instant::now();
        let mut generated: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
        let mut traces = Vec::new();

        // --- prefill: every prompt token forwards through the stack, only
        // the last one's logits matter (same shape as glproc::runner). ---
        let prefill_start = std::time::Instant::now();
        let mut hidden = vec![0.0f32; model.dim];
        for &token in &config.token_ids {
            let embedded = Self::embed_token(model, token)?;
            let mut activation = ActivationBuffer { data: embedded, shape: vec![model.dim] };
            runtime
                .run(&mut activation)
                .map_err(|e| GlError::Engine(format!("forward pass: {e}")))?;
            hidden = activation.data;
        }
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;

        // --- decode: sample from the last hidden state, then feed the
        // sampled token's embedding back in, one position at a time. ---
        let decode_start = std::time::Instant::now();
        let mut last_token_at: Option<std::time::Instant> = None;
        for _ in 0..config.max_new_tokens {
            // Raw logits (unmodified: no temperature, no penalty) are what
            // `trace_step` needs — snapshotted before the in-place penalty
            // rewrites them, mirroring glproc::runner::Runner::generate.
            let raw_logits = Self::logits_for(model, &hidden);
            let mut logits = raw_logits.clone();
            glproc::sampler::apply_repetition_penalty(&mut logits, &generated, config.repeat_penalty);
            let next = sampler.sample(&logits);

            if config.trace.enabled {
                let now = std::time::Instant::now();
                let since = last_token_at.map(|p| now.duration_since(p).as_nanos() as u64).unwrap_or(0);
                last_token_at = Some(now);
                if let Some(tr) = glcore::trace::trace_step(&raw_logits, next, since) {
                    traces.push(tr);
                }
            }

            on_token(next);
            generated.push(next);

            let embedded = Self::embed_token(model, next)?;
            let mut activation = ActivationBuffer { data: embedded, shape: vec![model.dim] };
            runtime
                .run(&mut activation)
                .map_err(|e| GlError::Engine(format!("forward pass: {e}")))?;
            hidden = activation.data;
        }
        let decode_ms = decode_start.elapsed().as_secs_f64() * 1e3;

        Ok(InferOutput {
            token_ids: generated.clone(),
            text: String::new(), // no tokenizer — see module docs
            tokens_generated: generated.len(),
            elapsed_ms: started.elapsed().as_millis() as u64,
            prompt_tokens: config.token_ids.len(),
            prefill_ms,
            generation_ms: decode_ms,
            traces,
        })
    }
}

impl GlEngine for GllmEngine {
    fn init(&mut self) -> Result<(), GlError> {
        Ok(())
    }

    /// `path` is a GLLM package **directory** (or the `gllm.json` inside
    /// one) — not a `.gguf`/`.safetensors` file. This deliberately does not
    /// match [`glcore::runtime::Runtime::load`]'s extension dispatch; a
    /// caller wiring this engine into that `Runtime` would need a
    /// package-aware bypass (glbench's `gllm` engine path does this
    /// directly rather than through `Runtime`, since there is no tokenizer
    /// for `Runtime::load` to build anyway).
    fn load_model(&mut self, path: &str) -> Result<(), GlError> {
        let root = std::path::Path::new(path);
        let root = if root.is_file() {
            root.parent().ok_or_else(|| GlError::Parse(format!("{path}: no parent directory")))?
        } else {
            root
        };
        self.model = Some(Self::load_shared(root)?);
        self.package_root = Some(root.to_path_buf());
        Ok(())
    }

    fn infer(&self, input: InferInput) -> Result<InferOutput, GlError> {
        self.run_request(&input, |_| {})
    }

    fn stream(
        &self,
        input: InferInput,
        on_token: &(dyn Fn(u32, &str) + Send),
    ) -> Result<InferOutput, GlError> {
        // No tokenizer means no text to decode a token into — the callback
        // still fires (per-token timing/callers may depend on cadence), with
        // an empty string rather than fabricating text.
        self.run_request(&input, |id| on_token(id, ""))
    }

    fn shutdown(&mut self) {
        self.model = None;
        self.package_root = None;
    }

    fn capabilities(&self) -> EngineSpec {
        EngineSpec { name: "gllm", backend: "cpu", available: true }
    }
}
