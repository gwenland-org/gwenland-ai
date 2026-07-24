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
            // `glcore::format::decode_tensor` only understands F32/F16/BF16
            // (glcore has no glproc dependency — GQ4A/GQ2A dequant kernels
            // live in glproc, and glproc already depends on glcore, so the
            // reverse dependency would be circular). CPP's Stage 1 table
            // quantizes token_embd/output to GQ4A (or escapes to it under
            // GQ2A_CPP) unconditionally, so a real converted package's
            // shared tensors need this dequant handled here, at the one
            // place in the crate that already legitimately depends on
            // glproc (ADR: architecture/Pridwen-P2-ADR-glproc-dequant.md).
            match entry.dtype {
                crate::manifest::DType::GQ4A => Ok(glproc::kernels::gquant::dequant_gq4a_stream(bytes)),
                crate::manifest::DType::GQ2A => Ok(glproc::kernels::gquant::dequant_gq2a_stream(bytes)),
                _ => {
                    let dtype_str = serde_json::to_value(entry.dtype)
                        .ok()
                        .and_then(|v| v.as_str().map(str::to_string))
                        .unwrap_or_default();
                    glcore::format::decode_tensor(name, &entry.shape, &dtype_str, bytes)
                        .map(|t| t.data)
                        .map_err(|e| GlError::Parse(format!("tensor {name:?}: {e}")))
                }
            }
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

    /// Teacher-forced raw logits for every position in `token_ids` (glbench's
    /// `ppl` subcommand, Wave 2).
    ///
    /// Unlike [`Self::infer`]/[`run_request`](Self::run_request), which
    /// autoregressively samples and feeds back *its own* output, this feeds
    /// exactly the given sequence — the model never sees anything but the
    /// ground-truth tokens. Position `i`'s logits are that position's
    /// prediction for the token at `i + 1` (standard next-token teacher
    /// forcing), so the returned vector has the same length as `token_ids`
    /// and its last element predicts one token past the end of the sequence.
    ///
    /// Raw logits only: full vocabulary, no temperature, no repetition
    /// penalty, no truncation — the same "raw" contract
    /// [`glcore::trace::trace_step`] documents, because a credible perplexity
    /// number is a property of the model, not of a sampling config that
    /// never runs here.
    pub fn score_sequence(&self, token_ids: &[u32]) -> Result<Vec<Vec<f32>>, GlError> {
        let model = self.model()?;
        let root = self.package_root.as_ref().ok_or_else(|| GlError::Engine("no model loaded".into()))?;

        let backend: Arc<dyn ExecutionBackend> = Arc::new(
            GlprocBackend::new(&model.metadata)
                .map_err(|e| GlError::Engine(format!("building glproc backend: {e}")))?,
        );
        let mut runtime = GllmRuntime::open(root, RuntimeConfig::default(), backend)
            .map_err(|e| GlError::Engine(format!("opening GLLM runtime: {e}")))?;

        let mut all_logits = Vec::with_capacity(token_ids.len());
        for &token in token_ids {
            let embedded = Self::embed_token(model, token)?;
            let mut activation = ActivationBuffer { data: embedded, shape: vec![model.dim] };
            runtime
                .run(&mut activation)
                .map_err(|e| GlError::Engine(format!("forward pass: {e}")))?;
            all_logits.push(Self::logits_for(model, &activation.data));
        }
        Ok(all_logits)
    }

    /// Final norm + LM head over a hidden state, `[dim] -> [vocab_size]` logits.
    fn logits_for(model: &LoadedModel, hidden: &[f32]) -> Vec<f32> {
        let mut normed = vec![0.0f32; model.dim];
        glproc::kernels::rms_norm_into(
            hidden,
            &model.output_norm,
            model.metadata.effective_rms_eps() as f32,
            &mut normed,
        );
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

        // Union of manifest-embedded and caller-supplied stop ids — neither
        // source alone is safe to treat as complete (real GGUF metadata is
        // routinely *less* complete than a Tokenizer's vocab-scanned resolve;
        // see glcore::stopping's module docs for the measured Qwen2.5-0.5B
        // case that motivated this). `config.stopping` also carries
        // `ignore_eos` when a caller (glbench's synthetic-prompt throughput
        // path) explicitly wants the full token budget regardless.
        let stopping = glcore::stopping::StoppingCriteria::new(model.metadata.eos_token_ids.iter().copied())
            .merge(&config.stopping);

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

            // Stop token first, same placement and reasoning as
            // glproc::runner::Runner::generate: it is not emitted, so
            // tracing it would leave `traces` one longer than `generated`
            // and silently misalign every per-token metric downstream.
            if stopping.is_stop(next) {
                break;
            }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::sha256_file;
    use crate::constants::SHARED_FILENAME;
    use tempfile::TempDir;

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|f| f.to_le_bytes()).collect()
    }

    /// A shared-only package (zero transformer layers): embedding table,
    /// tied-off LM head, and a final norm gain — exactly the tensors
    /// [`GllmEngine::load_shared`] reads, nothing [`GlprocBackend`] needs.
    /// `GllmRuntime::run` over zero layers passes its input straight through
    /// unchanged, so this isolates [`GllmEngine::score_sequence`]'s own
    /// per-position bookkeeping from the (separately, already) verified
    /// per-layer math in `glproc_backend.rs`'s
    /// `a_real_model_generates_a_token_through_gllm_end_to_end`.
    fn write_shared_only_package(dir: &std::path::Path, vocab: usize, dim: usize) {
        let embed: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.13).sin()).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.29).cos()).collect();
        let final_norm = vec![1.0f32; dim];
        let shared_path = dir.join(SHARED_FILENAME);
        let shared_entries = crate::layer_io::write_unit_file(
            &shared_path,
            &[
                (TOKEN_EMBEDDINGS, &[vocab as u64, dim as u64], crate::manifest::DType::F32, &f32_bytes(&embed)),
                (OUTPUT_HEAD, &[vocab as u64, dim as u64], crate::manifest::DType::F32, &f32_bytes(&lm_head)),
                (OUTPUT_NORM, &[dim as u64], crate::manifest::DType::F32, &f32_bytes(&final_norm)),
            ],
        )
        .unwrap();

        let manifest_json = serde_json::json!({
            "gllm_version": "1.0.0",
            "model_id": "org.gwenland.score-sequence-fixture",
            "architecture": "transformer",
            "metadata": {
                "vocab_size": vocab, "context_length": 64, "embedding_length": dim,
                "num_layers": 0, "num_heads": 2, "head_count_kv": 2,
            },
            "shared": {
                "file": SHARED_FILENAME,
                "checksum": format!("sha256:{}", sha256_file(&shared_path).unwrap()),
                "tensors": shared_entries,
            },
            "layers": [],
            "extensions": [],
        });
        std::fs::write(dir.join("gllm.json"), manifest_json.to_string()).unwrap();
    }

    /// Same shape as [`write_shared_only_package`], with manifest-level
    /// `eos_token_ids` set — for the stopping-criteria tests, which need a
    /// package whose *manifest* (not just the caller's `InferInput`)
    /// declares stop ids.
    fn write_package_with_manifest_eos(dir: &std::path::Path, vocab: usize, dim: usize, eos_ids: &[u32]) {
        let embed: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.13).sin()).collect();
        let lm_head: Vec<f32> = (0..vocab * dim).map(|i| (i as f32 * 0.29).cos()).collect();
        let final_norm = vec![1.0f32; dim];
        let shared_path = dir.join(SHARED_FILENAME);
        let shared_entries = crate::layer_io::write_unit_file(
            &shared_path,
            &[
                (TOKEN_EMBEDDINGS, &[vocab as u64, dim as u64], crate::manifest::DType::F32, &f32_bytes(&embed)),
                (OUTPUT_HEAD, &[vocab as u64, dim as u64], crate::manifest::DType::F32, &f32_bytes(&lm_head)),
                (OUTPUT_NORM, &[dim as u64], crate::manifest::DType::F32, &f32_bytes(&final_norm)),
            ],
        )
        .unwrap();

        let manifest_json = serde_json::json!({
            "gllm_version": "1.0.0",
            "model_id": "org.gwenland.stopping-fixture",
            "architecture": "transformer",
            "metadata": {
                "vocab_size": vocab, "context_length": 64, "embedding_length": dim,
                "num_layers": 0, "num_heads": 2, "head_count_kv": 2,
                "eos_token_ids": eos_ids,
            },
            "shared": {
                "file": SHARED_FILENAME,
                "checksum": format!("sha256:{}", sha256_file(&shared_path).unwrap()),
                "tensors": shared_entries,
            },
            "layers": [],
            "extensions": [],
        });
        std::fs::write(dir.join("gllm.json"), manifest_json.to_string()).unwrap();
    }

    /// Greedy (deterministic), no repetition penalty confound, small budget
    /// — the shape every stopping test below builds its `InferInput` from.
    fn greedy_input(token_ids: Vec<u32>, max_new_tokens: usize) -> InferInput {
        InferInput {
            token_ids,
            max_new_tokens,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            trace: Default::default(),
            stopping: Default::default(),
        }
    }

    #[test]
    fn score_sequence_returns_one_logits_vector_per_input_token() {
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_shared_only_package(dir.path(), VOCAB, DIM);

        let mut engine = GllmEngine::new();
        engine.load_model(dir.path().to_str().unwrap()).unwrap();

        let token_ids = vec![0u32, 2, 5, 1];
        let all_logits = engine.score_sequence(&token_ids).unwrap();

        assert_eq!(all_logits.len(), token_ids.len(), "one logits row per input position");
        for row in &all_logits {
            assert_eq!(row.len(), VOCAB, "each row is a full-vocab logits vector");
            assert!(row.iter().all(|x| x.is_finite()), "{row:?}");
        }
    }

    #[test]
    fn score_sequence_is_teacher_forced_not_sampling_dependent() {
        // Feeding the same sequence twice must give byte-identical logits —
        // there is no RNG/sampling anywhere in this path (unlike `infer`,
        // which seeds a sampler). This is the property PPL depends on: two
        // runs of the same benchmark must agree bit-for-bit.
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_shared_only_package(dir.path(), VOCAB, DIM);

        let mut engine = GllmEngine::new();
        engine.load_model(dir.path().to_str().unwrap()).unwrap();

        let token_ids = vec![3u32, 1, 4];
        let a = engine.score_sequence(&token_ids).unwrap();
        let b = engine.score_sequence(&token_ids).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn score_sequence_rejects_an_out_of_vocab_token() {
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_shared_only_package(dir.path(), VOCAB, DIM);

        let mut engine = GllmEngine::new();
        engine.load_model(dir.path().to_str().unwrap()).unwrap();

        let err = engine.score_sequence(&[VOCAB as u32]).unwrap_err();
        assert!(matches!(err, GlError::Engine(_)), "got {err:?}");
    }

    #[test]
    fn score_sequence_errors_without_a_loaded_model() {
        let engine = GllmEngine::new();
        let err = engine.score_sequence(&[0]).unwrap_err();
        assert!(matches!(err, GlError::Engine(_)), "got {err:?}");
    }

    #[test]
    fn generate_without_stopping_criteria_runs_to_max_tokens() {
        // Regression guard for the pre-fix behavior: no manifest EOS ids,
        // no caller-supplied stopping — the decode loop must still run the
        // full budget, exactly as it always has.
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir.path(), VOCAB, DIM, &[]);

        let mut engine = GllmEngine::new();
        engine.load_model(dir.path().to_str().unwrap()).unwrap();

        let out = engine.infer(greedy_input(vec![0], 5)).unwrap();
        assert_eq!(out.token_ids.len(), 5);
    }

    #[test]
    fn generate_halts_at_eos_token_and_does_not_emit_it() {
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir.path(), VOCAB, DIM, &[]);

        let mut engine = GllmEngine::new();
        engine.load_model(dir.path().to_str().unwrap()).unwrap();

        // Observe the natural greedy trajectory first (deterministic:
        // temperature 0, no repetition penalty to confound it).
        let baseline = engine.infer(greedy_input(vec![0], 5)).unwrap();
        assert_eq!(baseline.token_ids.len(), 5, "sanity: nothing stops it yet");
        let target = baseline.token_ids[2];

        // Now ask the engine (via InferInput, not the manifest) to stop on
        // that exact token.
        let mut input = greedy_input(vec![0], 5);
        input.stopping = glcore::stopping::StoppingCriteria::new([target]);
        let out = engine.infer(input).unwrap();

        assert!(out.token_ids.len() < 5, "must stop before the full budget: {:?}", out.token_ids);
        assert!(!out.token_ids.contains(&target), "stop token must not be emitted: {:?}", out.token_ids);
        assert_eq!(
            out.token_ids,
            baseline.token_ids[..out.token_ids.len()],
            "tokens before the stop point must match the unstopped trajectory exactly"
        );
    }

    #[test]
    fn generate_unions_manifest_and_infer_input_stop_ids() {
        // The manifest declares a LATE-occurring id; InferInput declares a
        // *different*, EARLIER-occurring one the manifest doesn't know
        // about. If the engine only consulted the manifest (ignoring
        // InferInput whenever the manifest is non-empty — the literal
        // "manifest overrides" design this test guards against), it would
        // run past the caller's earlier id and stop only at the manifest's
        // later one. A correct union stops at the earlier point.
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir.path(), VOCAB, DIM, &[]);
        let mut probe = GllmEngine::new();
        probe.load_model(dir.path().to_str().unwrap()).unwrap();
        let baseline = probe.infer(greedy_input(vec![0], 5)).unwrap();
        assert_eq!(baseline.token_ids.len(), 5);

        let early_tok = baseline.token_ids[0];
        let late_idx = (1..baseline.token_ids.len())
            .find(|&i| baseline.token_ids[i] != early_tok)
            .expect("need a later, distinct token for this to prove anything");
        let late_tok = baseline.token_ids[late_idx];

        // Manifest only knows the LATE token.
        let dir2 = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir2.path(), VOCAB, DIM, &[late_tok]);
        let mut engine = GllmEngine::new();
        engine.load_model(dir2.path().to_str().unwrap()).unwrap();

        // Sanity: manifest alone stops at `late_idx`.
        let out_manifest_only = engine.infer(greedy_input(vec![0], 5)).unwrap();
        assert_eq!(out_manifest_only.token_ids, baseline.token_ids[..late_idx]);

        // InferInput supplies the EARLY token — one the manifest never
        // declared. If the union is real, this must stop at position 0
        // (before `late_idx`), not run past it to the manifest's own id.
        let mut input = greedy_input(vec![0], 5);
        input.stopping = glcore::stopping::StoppingCriteria::new([early_tok]);
        let out_union = engine.infer(input).unwrap();
        assert_eq!(
            out_union.token_ids.len(),
            0,
            "the caller's earlier id must be honored even though the manifest also has a (later, different) id: {:?}",
            out_union.token_ids
        );
    }

    #[test]
    fn qwen2_dual_eos_both_ids_stop_generation() {
        // Qwen2.5's real dual EOS: <|im_end|> (151645) and <|endoftext|>
        // (151643) must independently halt generation when either is
        // sampled — mirrors glcore::tokenizer's
        // qwen_style_stop_markers_resolve_from_vocab, at the engine level.
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir.path(), VOCAB, DIM, &[]);
        let mut probe = GllmEngine::new();
        probe.load_model(dir.path().to_str().unwrap()).unwrap();
        let baseline = probe.infer(greedy_input(vec![0], 5)).unwrap();
        assert_eq!(baseline.token_ids.len(), 5);

        // Use whatever the natural trajectory's first two *distinct* tokens
        // are as stand-ins for the two real Qwen ids — the fixture's vocab
        // is 6, not 151936, but the mechanism under test (multiple ids, any
        // one of which halts) is identical.
        let first = baseline.token_ids[0];
        let second_distinct =
            baseline.token_ids.iter().copied().find(|&t| t != first).expect("need two distinct tokens");

        for stop_id in [first, second_distinct] {
            let dir2 = TempDir::new().unwrap();
            write_package_with_manifest_eos(dir2.path(), VOCAB, DIM, &[first, second_distinct]);
            let mut engine = GllmEngine::new();
            engine.load_model(dir2.path().to_str().unwrap()).unwrap();
            let out = engine.infer(greedy_input(vec![0], 5)).unwrap();
            assert!(!out.token_ids.contains(&stop_id), "{stop_id} must not be emitted: {:?}", out.token_ids);
        }
    }

    #[test]
    fn ignoring_eos_runs_full_budget_even_with_manifest_eos_ids() {
        // glbench's own motivation for `ignore_eos`: a manifest that DOES
        // declare stop ids must still be fully suppressible by the caller.
        const VOCAB: usize = 6;
        const DIM: usize = 8;
        let dir = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir.path(), VOCAB, DIM, &[]);
        let mut probe = GllmEngine::new();
        probe.load_model(dir.path().to_str().unwrap()).unwrap();
        let baseline = probe.infer(greedy_input(vec![0], 5)).unwrap();
        let target = baseline.token_ids[2];

        let dir2 = TempDir::new().unwrap();
        write_package_with_manifest_eos(dir2.path(), VOCAB, DIM, &[target]);
        let mut engine = GllmEngine::new();
        engine.load_model(dir2.path().to_str().unwrap()).unwrap();

        // Without ignore_eos: stops early, as the tests above already prove.
        let stopped = engine.infer(greedy_input(vec![0], 5)).unwrap();
        assert!(stopped.token_ids.len() < 5);

        // With ignore_eos: runs the full budget despite the manifest's id.
        let mut input = greedy_input(vec![0], 5);
        input.stopping = glcore::stopping::StoppingCriteria::default().ignoring_eos();
        let out = engine.infer(input).unwrap();
        assert_eq!(out.token_ids.len(), 5, "ignore_eos must suppress the manifest's stop id too");
    }
}
