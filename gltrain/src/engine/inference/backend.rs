// engine/inference/backend.rs — InferenceBackend trait definition.
//
// Defines the unified interface that all inference engine implementations must
// satisfy. Backends are registered at runtime via `BackendRegistry` and
// selected through configuration, allowing new engines to be added without
// modifying client code.
//
// Requirement: 1.1 – 1.7

use std::path::Path;
use std::pin::Pin;

use anyhow::Result;
use futures_util::Stream;

use super::params::InferParams;

/// Unified interface for all inference backends.
///
/// Every backend (candle, mistral.rs, …) must implement this trait so that
/// client code can drive inference without knowing which engine is active.
///
/// # Thread safety
///
/// The trait bounds `Send + Sync` allow an `Arc<dyn InferenceBackend>` to be
/// shared across threads in an async runtime. Implementations that need
/// interior mutability (e.g. to cache a loaded model) must use
/// `Arc<Mutex<…>>` internally — the public API is always `&self`.
///
/// # Lifecycle
///
/// 1. Call [`load_model`] once to deserialise weights and warm up the engine.
/// 2. Call [`infer`] or [`stream_infer`] for each generation request.
/// 3. Call [`unload`] to free GPU/CPU memory when the backend is no longer
///    needed.
///
/// [`load_model`]: InferenceBackend::load_model
/// [`infer`]: InferenceBackend::infer
/// [`stream_infer`]: InferenceBackend::stream_infer
/// [`unload`]: InferenceBackend::unload
pub trait InferenceBackend: Send + Sync {
    /// Load a model from `model_path` and prepare it for inference.
    ///
    /// # Preconditions
    ///
    /// - `model_path` must point to a readable GGUF file.
    /// - If a model is already loaded it **must** be freed before the new one
    ///   is initialised (Requirement 11.1).
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - The path does not exist or is not readable.
    /// - The file format or model architecture is not supported.
    /// - Allocating device memory fails.
    fn load_model(&self, model_path: &Path) -> Result<()>;

    /// Run a single, blocking inference pass and return the full generated
    /// text.
    ///
    /// # Preconditions
    ///
    /// - A model must have been successfully loaded via [`load_model`].
    /// - `params` should be validated with [`InferParams::validate`] before
    ///   being passed here.
    ///
    /// # Returns
    ///
    /// The complete generated string, equivalent to concatenating every token
    /// that [`stream_infer`] would yield for the same inputs
    /// (Requirement 7.4).
    ///
    /// # Errors
    ///
    /// Returns `Err` when generation fails for any reason (e.g. model not
    /// loaded, OOM, tokeniser error).
    ///
    /// [`load_model`]: InferenceBackend::load_model
    /// [`stream_infer`]: InferenceBackend::stream_infer
    fn infer(&self, prompt: &str, params: &InferParams) -> Result<String>;

    /// Begin a streaming inference pass and return a token stream.
    ///
    /// Each `String` item in the stream is one decoded token fragment.  All
    /// items are guaranteed to be valid UTF-8 (Requirement 6.2).  The stream
    /// terminates naturally when:
    /// - The `max_tokens` limit in `params` is reached (Requirement 6.5), or
    /// - A stop sequence is encountered (Requirement 6.6), or
    /// - Generation completes normally.
    ///
    /// If generation fails mid-stream the stream **must** terminate; callers
    /// should treat a short stream as a possible error (Requirement 6.4).
    ///
    /// # Preconditions
    ///
    /// - A model must have been successfully loaded via [`load_model`].
    /// - `params` should be validated before being passed here.
    ///
    /// # Returns
    ///
    /// A heap-allocated, pinned, `Send` stream of token strings.  Using
    /// `Pin<Box<dyn Stream<…>>>` avoids naming the concrete future type while
    /// keeping the stream moveable across await points.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the stream cannot be created (e.g. model not loaded,
    /// tokeniser error on the prompt itself).
    ///
    /// [`load_model`]: InferenceBackend::load_model
    fn stream_infer(
        &self,
        prompt: &str,
        params: &InferParams,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>>;

    /// Release all GPU/CPU memory allocated for the currently loaded model.
    ///
    /// After this call the backend is back in an unloaded state; [`infer`] and
    /// [`stream_infer`] must not be called again until [`load_model`] succeeds.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the underlying engine reports an error while tearing
    /// down the model pipeline.  Implementations should make a best-effort
    /// attempt to free resources even when returning an error.
    ///
    /// [`infer`]: InferenceBackend::infer
    /// [`stream_infer`]: InferenceBackend::stream_infer
    /// [`load_model`]: InferenceBackend::load_model
    fn unload(&self) -> Result<()>;

    /// Return a stable, human-readable identifier for this backend.
    ///
    /// The returned string is used as the lookup key in [`BackendRegistry`]
    /// and as the value for the `backend` field in `InferenceConfig`.
    ///
    /// # Examples
    ///
    /// - `"candle"`
    /// - `"mistralrs"`
    ///
    /// [`BackendRegistry`]: crate::engine::inference::registry::BackendRegistry
    fn name(&self) -> &'static str;
}
