//! The `GlEngine` trait — the contract every GL compute backend implements.

use crate::error::GlError;

/// Input to the engine for a single inference request.
#[derive(Debug, Clone)]
pub struct InferInput {
    /// Prompt tokens, already encoded by a tokenizer.
    pub token_ids: Vec<u32>,
    /// Maximum number of tokens to generate.
    pub max_new_tokens: usize,
    /// Sampling temperature. `1.0` = no change, `0.0` = greedy.
    pub temperature: f32,
    /// Top-k cutoff. `0` = disabled.
    pub top_k: usize,
    /// Top-p (nucleus) cutoff. `1.0` = disabled.
    pub top_p: f32,
}

impl Default for InferInput {
    fn default() -> Self {
        InferInput {
            token_ids: Vec::new(),
            max_new_tokens: 256,
            temperature: 0.8,
            top_k: 40,
            top_p: 0.95,
        }
    }
}

/// Output from a single inference request.
#[derive(Debug, Clone)]
pub struct InferOutput {
    /// Generated token ids (not including the prompt).
    pub token_ids: Vec<u32>,
    /// Generated tokens decoded to text.
    pub text: String,
    /// Number of tokens generated.
    pub tokens_generated: usize,
    /// Wall-clock generation time in milliseconds.
    pub elapsed_ms: u64,
}

/// Static metadata about an engine.
#[derive(Debug, Clone)]
pub struct EngineSpec {
    /// Human-readable engine name, e.g. `"glproc"`.
    pub name: &'static str,
    /// Backend kind: `"cpu"`, `"cuda"`, `"vulkan"`, or `"metal"`.
    pub backend: &'static str,
    /// Whether this engine can actually run on the current machine.
    pub available: bool,
}

/// The core engine trait all GL engines must implement.
///
/// Note: the streaming callback is a `&dyn Fn` (rather than `impl Fn`) so the
/// trait stays object-safe — the shared [`crate::runtime::Runtime`] holds
/// engines as `Box<dyn GlEngine>`.
pub trait GlEngine: Send + Sync {
    /// Initialize the engine (allocate resources, detect hardware).
    fn init(&mut self) -> Result<(), GlError>;

    /// Load a model from a file path (GGUF or safetensors).
    fn load_model(&mut self, path: &str) -> Result<(), GlError>;

    /// Run synchronous inference.
    fn infer(&self, input: InferInput) -> Result<InferOutput, GlError>;

    /// Stream tokens via callback — the default implementation wraps
    /// [`GlEngine::infer`] and replays tokens after the fact.
    fn stream(
        &self,
        input: InferInput,
        on_token: &(dyn Fn(u32, &str) + Send),
    ) -> Result<(), GlError> {
        let out = self.infer(input)?;
        for (id, ch) in out.token_ids.iter().zip(out.text.chars()) {
            on_token(*id, &ch.to_string());
        }
        Ok(())
    }

    /// Graceful shutdown (free GPU memory, close handles).
    fn shutdown(&mut self);

    /// Return static metadata about this engine.
    fn capabilities(&self) -> EngineSpec;
}
