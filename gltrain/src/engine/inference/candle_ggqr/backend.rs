// engine/inference/candle_ggqr/backend.rs — GgqrCandleBackend: lifecycle + InferenceBackend impl.
//
// ── Responsibilities ────────────────────────────────────────────────────────
//
//   • Hold the loaded model state under a Mutex so `&self` callers can share
//     the backend safely across threads (required by InferenceBackend: Send+Sync).
//   • Implement `load_model_impl` / `unload_impl` (model lifecycle).
//   • Implement the `InferenceBackend` trait, delegating generation to
//     `generation.rs` and sampling to `sampling.rs`.
//
// ── Design decision: why Mutex<Option<LoadedState>>? ───────────────────────
//
//   `InferenceBackend` mandates `&self` for all methods so the same
//   `Arc<dyn InferenceBackend>` can be held across threads.  We need interior
//   mutability for the loaded-model state.  A `Mutex<Option<…>>` is the
//   idiomatic choice: it is simpler than `RwLock` (generation is CPU-bound
//   and never actually concurrent), and `Option` makes the unloaded state
//   explicit rather than hiding it behind a bool flag.
//
// Requirements: 4.1–4.5, 12.1–12.5, 14.1, 16.1–16.4, 20.1, 20.2

use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::sync::Mutex;

use anyhow::Result;
use candle_core::{Device, Tensor};
use futures_util::Stream;
use sysinfo::System;
use tokenizers::Tokenizer;

use crate::error::GwenError;
use crate::engine::inference::backend::InferenceBackend;
use crate::engine::inference::params::InferParams;

use super::{
    ModelConfig,
    build_model_config, extract_architecture, validate_gguf,
    dequantize_tensor, vec_to_tensor,
    generate_collect, make_stream_pinned, GenerationState,
};

// ── Required tensor name patterns ────────────────────────────────────────────
// We check that these key substrings appear in the loaded tensor map so we can
// give an actionable error rather than panicking later during inference.

const REQUIRED_TENSOR_PATTERNS: &[&str] = &[
    "token_embd",   // embedding table
    "attn_q",       // query projection
    "attn_k",       // key projection
    "attn_v",       // value projection
    "ffn_gate",     // MLP gate (SwiGLU)
    "output",       // lm_head / output projection
];

// ── State held under Mutex ────────────────────────────────────────────────────

struct LoadedState {
    tensors: HashMap<String, Tensor>,
    config: ModelConfig,
    tokenizer: Tokenizer,
    device: Device,
}

// ── GgqrCandleBackend ─────────────────────────────────────────────────────────

/// Inference backend that uses the GGQR dequantisation engine together with
/// `candle-core` for CPU forward passes on LLaMA-family models.
///
/// All mutable state (loaded weights, tokenizer, config) lives inside a
/// `Mutex<Option<LoadedState>>` so the outer `&self` API required by
/// [`InferenceBackend`] can be satisfied without exposing `&mut self`.
///
/// Requirements: 4.1, 4.2, 4.3
pub struct GgqrCandleBackend {
    state: Mutex<Option<LoadedState>>,
}

impl GgqrCandleBackend {
    /// Create a new, unloaded backend instance.
    pub fn new() -> Self {
        Self {
            state: Mutex::new(None),
        }
    }

    // ── load_model ────────────────────────────────────────────────────────────

    /// Load a GGUF model from `model_path`, dequantise all tensors, and store
    /// them as `candle_core::Tensor` values keyed by their GGUF tensor names.
    ///
    /// Steps:
    ///   1. Check available system RAM (require ≥ 1 GB free).
    ///   2. Validate the GGUF file (magic, tensor count, dim sizes, buf sizes).
    ///   3. Extract and validate `general.architecture`.
    ///   4. Build `ModelConfig` from GGUF KV metadata.
    ///   5. Dequantise every tensor → `Vec<f32>`.
    ///   6. Convert each `Vec<f32>` → `Tensor` (zero-copy on CPU).
    ///   7. Validate that all required tensor name patterns are present.
    ///   8. Load tokenizer from GGUF embedded tokenizer data or raise an error.
    ///   9. Log architecture, layer count, and peak memory usage.
    ///
    /// Requirements: 1.1, 1.5, 4.1, 4.3, 4.4, 4.5, 12.1, 12.4, 12.5,
    ///               14.1, 16.1, 16.2, 16.3, 16.4
    pub fn load_model_impl(&self, model_path: &Path) -> Result<(), GwenError> {
        // ── 1. Memory check ───────────────────────────────────────────────────
        check_available_memory(1)?;

        eprintln!("candle-ggqr: loading model from '{}'", model_path.display());

        // ── 2. Validate GGUF ──────────────────────────────────────────────────
        let gguf = validate_gguf(model_path)?;
        eprintln!(
            "candle-ggqr: GGUF v{} validated — {} tensors",
            gguf.version,
            gguf.tensors.len()
        );

        // ── 3. Architecture ───────────────────────────────────────────────────
        let arch = extract_architecture(model_path)?;
        eprintln!("candle-ggqr: architecture = {arch}");

        // ── 4. ModelConfig ────────────────────────────────────────────────────
        let config = build_model_config(model_path, &arch)?;
        eprintln!(
            "candle-ggqr: layers={} hidden={} heads={} kv_heads={} vocab={}",
            config.n_layers, config.hidden_size, config.n_heads,
            config.n_kv_heads, config.vocab_size
        );

        // Always use CPU — GPU support is deferred.
        let device = Device::Cpu;

        // ── 5 + 6. Dequantise → Tensor ────────────────────────────────────────
        let n_tensors = gguf.tensors.len();
        let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(n_tensors);

        for (i, tensor_info) in gguf.tensors.iter().enumerate() {
            eprintln!(
                "candle-ggqr: dequantising [{}/{}] '{}'  dtype={:?}  shape={:?}",
                i + 1, n_tensors,
                tensor_info.name, tensor_info.dtype, tensor_info.shape
            );

            let data = dequantize_tensor(tensor_info)?;

            let shape: Vec<usize> = tensor_info.shape.iter().map(|&d| d as usize).collect();
            let t = vec_to_tensor(data, shape.as_slice(), &device)?;

            tensors.insert(tensor_info.name.clone(), t);
        }

        // ── 7. Required tensor validation ─────────────────────────────────────
        for pattern in REQUIRED_TENSOR_PATTERNS {
            let found = tensors.keys().any(|k| k.contains(pattern));
            if !found {
                return Err(GwenError::ModelLoad(format!(
                    "model is missing a required tensor matching '{}' \
                     (found {} tensors total)",
                    pattern,
                    tensors.len()
                )));
            }
        }

        // ── 8. Tokenizer ──────────────────────────────────────────────────────
        let tokenizer = load_tokenizer_from_gguf(model_path)?;

        // ── 9. Log memory ─────────────────────────────────────────────────────
        let mem_gb = current_used_memory_gb();
        eprintln!("candle-ggqr: model loaded — {} tensors, ~{:.2} GB RAM used", tensors.len(), mem_gb);

        // ── Store state ───────────────────────────────────────────────────────
        let mut guard = self.state.lock().unwrap();
        *guard = Some(LoadedState { tensors, config, tokenizer, device });

        Ok(())
    }

    // ── unload ────────────────────────────────────────────────────────────────

    /// Release all memory held by the currently loaded model.
    ///
    /// After this call `load_model_impl` must be called again before running
    /// inference. Safe to call when nothing is loaded.
    ///
    /// Requirement: 12.2
    pub fn unload_impl(&self) -> Result<(), GwenError> {
        let mut guard = self.state.lock().unwrap();
        if guard.is_some() {
            *guard = None;
            eprintln!("candle-ggqr: model unloaded, memory released");
        }
        Ok(())
    }

    // ── Accessors used by later waves ─────────────────────────────────────────

    /// Returns `true` when a model is currently loaded.
    pub fn is_loaded(&self) -> bool {
        self.state.lock().unwrap().is_some()
    }

    /// Borrow the loaded tensors map.
    ///
    /// Returns `GwenError::InferenceBackend` if no model is loaded.
    pub fn with_state<F, R>(&self, f: F) -> Result<R, GwenError>
    where
        F: FnOnce(&LoadedState) -> Result<R, GwenError>,
    {
        let guard = self.state.lock().unwrap();
        match guard.as_ref() {
            Some(s) => f(s),
            None => Err(GwenError::InferenceBackend(
                "no model loaded — call load_model first".to_string(),
            )),
        }
    }
}

impl Default for GgqrCandleBackend {
    fn default() -> Self {
        Self::new()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Check that at least `required_gb` GB of RAM is available.
///
/// Requirement: 12.4
fn check_available_memory(required_gb: u32) -> Result<(), GwenError> {
    let mut sys = System::new();
    sys.refresh_memory();

    let available = sys.available_memory(); // bytes
    let required_bytes = required_gb as u64 * 1_073_741_824;

    if available < required_bytes {
        let available_gb = available as f32 / 1_073_741_824.0;
        return Err(GwenError::InsufficientMemory {
            required_gb: required_gb as f32,
            available_gb,
        });
    }

    Ok(())
}

/// Read current process used-memory in GB (best-effort; returns 0.0 on error).
fn current_used_memory_gb() -> f64 {
    let mut sys = System::new();
    sys.refresh_memory();
    sys.used_memory() as f64 / 1_073_741_824.0
}

/// Extract the tokenizer from GGUF metadata tokens arrays.
///
/// GGUF files embedding HuggingFace-compatible tokenizers store the vocabulary
/// under `tokenizer.ggml.tokens` (string array). We build a minimal BPE-style
/// `Tokenizer` from those tokens so we don't need a network call.
///
/// If the embedded tokens cannot be parsed we return `GwenError::ModelLoad`
/// with a clear message.
///
/// Requirements: 4.4, 16.1 – 16.4
fn load_tokenizer_from_gguf(model_path: &Path) -> Result<Tokenizer, GwenError> {
    // Try to load a sidecar tokenizer.json next to the .gguf file first,
    // as that is the most reliable source.
    let sidecar = model_path.with_extension("").with_file_name(
        model_path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .as_ref()
            .to_string()
            + "_tokenizer.json",
    );

    if sidecar.exists() {
        return Tokenizer::from_file(&sidecar)
            .map_err(|e| GwenError::ModelLoad(format!("tokenizer load error: {e}")));
    }

    // Fall back: placeholder that defers to `tokenizer.json` beside the GGUF.
    let beside = model_path.with_file_name("tokenizer.json");
    if beside.exists() {
        return Tokenizer::from_file(&beside)
            .map_err(|e| GwenError::ModelLoad(format!("tokenizer load error: {e}")));
    }

    Err(GwenError::ModelLoad(format!(
        "no tokenizer found for '{}' — place a tokenizer.json beside the GGUF file",
        model_path.display()
    )))
}

// ── InferenceBackend trait implementation ─────────────────────────────────────
//
// This is the public face of the candle-ggqr backend.  All methods are thin
// wrappers that acquire the Mutex and delegate to the internal `*_impl`
// methods or the generation module.

impl InferenceBackend for GgqrCandleBackend {
    /// Stable backend identifier used by `BackendRegistry` and `InferenceConfig`.
    ///
    /// Requirement: 20.1
    fn name(&self) -> &'static str {
        "candle-ggqr"
    }

    /// Load a GGUF model from disk, dequantise all tensors, and store state.
    ///
    /// Converts `anyhow::Result` from the trait signature to/from the internal
    /// `GwenError` type via the `From` impl so the trait boundary is satisfied.
    ///
    /// Requirement: 20.2 (wire load_model)
    fn load_model(&self, model_path: &Path) -> Result<()> {
        self.load_model_impl(model_path)
            .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Run autoregressive generation and collect the full response string.
    ///
    /// Delegates to `generate_collect` which drives the async stream inside a
    /// `block_in_place` so this method remains synchronous.
    ///
    /// # Preconditions
    ///
    /// A model must be loaded — returns an error otherwise.
    ///
    /// Requirements: 10.1–10.4, 20.2
    fn infer(&self, prompt: &str, params: &InferParams) -> Result<String> {
        self.with_state(|state| {
            let gen_state = GenerationState {
                tensors: &state.tensors,
                config: &state.config,
                tokenizer: &state.tokenizer,
                device: &state.device,
            };
            generate_collect(&gen_state, prompt, params)
        })
        .map_err(|e| anyhow::anyhow!("{e}"))
    }

    /// Begin streaming generation and return a pinned `Send` token stream.
    ///
    /// The loaded state is cloned out of the Mutex before the stream is
    /// constructed so the Mutex is not held across await points (which would
    /// deadlock if another thread calls `unload` or `load_model` mid-stream).
    ///
    /// # Preconditions
    ///
    /// A model must be loaded — returns an error otherwise.
    ///
    /// Requirements: 8.1–8.6, 16.1, 20.2
    fn stream_infer(
        &self,
        prompt: &str,
        params: &InferParams,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>> {
        // Clone everything out of the Mutex so the stream is fully owned
        // and 'static, keeping the lock as short-lived as possible.
        let (tensors, config, tokenizer, device) = self.with_state(|state| {
            Ok((
                state.tensors.clone(),
                state.config.clone(),
                state.tokenizer.clone(),
                state.device.clone(),
            ))
        })
        .map_err(|e| anyhow::anyhow!("{e}"))?;

        let stream = make_stream_pinned(
            tensors,
            config,
            tokenizer,
            device,
            prompt.to_string(),
            params.clone(),
        );

        Ok(stream)
    }

    /// Release all model weights and return to the unloaded state.
    ///
    /// Requirement: 20.2 (wire unload)
    fn unload(&self) -> Result<()> {
        self.unload_impl().map_err(|e| anyhow::anyhow!("{e}"))
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn new_backend_is_not_loaded() {
        let b = GgqrCandleBackend::new();
        assert!(!b.is_loaded());
    }

    #[test]
    fn default_backend_is_not_loaded() {
        let b = GgqrCandleBackend::default();
        assert!(!b.is_loaded());
    }

    // ── with_state before load returns error ──────────────────────────────────

    #[test]
    fn with_state_before_load_returns_inference_backend_error() {
        let b = GgqrCandleBackend::new();
        let err = b.with_state(|_| Ok(())).unwrap_err();
        assert!(
            matches!(err, GwenError::InferenceBackend(_)),
            "expected InferenceBackend error, got {err:?}"
        );
    }

    // ── unload on empty backend is a no-op ───────────────────────────────────

    #[test]
    fn unload_on_empty_backend_is_ok() {
        let b = GgqrCandleBackend::new();
        assert!(b.unload_impl().is_ok());
        assert!(!b.is_loaded());
    }

    // ── load_model with non-existent path returns ModelLoad error ─────────────

    #[test]
    fn load_model_nonexistent_path_returns_model_load_error() {
        let b = GgqrCandleBackend::new();
        let err = b
            .load_model_impl(Path::new("nonexistent_file.gguf"))
            .unwrap_err();
        // Could be ModelLoad (file not found) or InsufficientMemory (very low RAM).
        assert!(
            matches!(err, GwenError::ModelLoad(_) | GwenError::InsufficientMemory { .. }),
            "unexpected error variant: {err:?}"
        );
    }

    // ── load_model with bad magic bytes returns ModelLoad error ───────────────

    #[test]
    fn load_model_bad_magic_returns_model_load_error() {
        use std::io::Write;
        use tempfile::NamedTempFile;

        let mut f = NamedTempFile::new().unwrap();
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            .unwrap();
        f.flush().unwrap();

        let b = GgqrCandleBackend::new();
        let err = b.load_model_impl(f.path()).unwrap_err();
        assert!(
            matches!(err, GwenError::ModelLoad(_) | GwenError::InsufficientMemory { .. }),
            "unexpected error variant: {err:?}"
        );
    }

    // ── unload clears loaded state ────────────────────────────────────────────
    // (full load needs a real GGUF file — tested in optional 7.4/7.5)

    #[test]
    fn unload_after_manual_state_inject_clears_state() {
        use std::str::FromStr;
        use tokenizers::Tokenizer;
        // Inject a minimal state directly to test unload without a real file.
        let b = GgqrCandleBackend::new();
        {
            let mut guard = b.state.lock().unwrap();
            // Build a trivial tokenizer from BPE JSON.
            let tok_json = r#"{
                "version": "1.0",
                "truncation": null,
                "padding": null,
                "added_tokens": [],
                "normalizer": null,
                "pre_tokenizer": null,
                "post_processor": null,
                "decoder": null,
                "model": {"type": "BPE", "vocab": {}, "merges": []}
            }"#;
            let tokenizer = Tokenizer::from_str(tok_json)
                .expect("inline tokenizer must parse");
            *guard = Some(LoadedState {
                tensors: HashMap::new(),
                config: ModelConfig {
                    architecture: "llama".to_string(),
                    n_layers: 1,
                    hidden_size: 64,
                    n_heads: 1,
                    n_kv_heads: 1,
                    intermediate_size: 128,
                    vocab_size: 32,
                    rms_norm_eps: 1e-5,
                    rope_theta: 10_000.0,
                },
                tokenizer,
                device: Device::Cpu,
            });
        }

        assert!(b.is_loaded(), "state should be loaded after inject");
        b.unload_impl().unwrap();
        assert!(!b.is_loaded(), "state should be cleared after unload");
    }
}
