//! Shared runtime engine manager, used by `glcli` and `gltui`.
//!
//! The [`Runtime`] owns exactly one engine and routes requests to it. It
//! never performs compute itself — its job is tokenization at the boundary
//! and engine lifecycle management.

use std::path::Path;

use crate::engine_trait::{GlEngine, InferInput};
use crate::error::GlError;
use crate::format::gguf::GgufFile;
use crate::tokenizer::Tokenizer;

/// Owns the active engine and the tokenizer for the loaded model.
pub struct Runtime {
    engine: Box<dyn GlEngine>,
    tokenizer: Option<Tokenizer>,
    /// Force raw completion encoding even for chat models.
    raw_prompt: bool,
}

impl Runtime {
    /// Initialize the runtime with a specific engine.
    pub fn new(mut engine: Box<dyn GlEngine>) -> Result<Self, GlError> {
        engine.init()?;
        Ok(Runtime {
            engine,
            tokenizer: None,
            raw_prompt: false,
        })
    }

    /// Skip the chat template and encode prompts as raw completions
    /// (base-model behavior) even when the vocab has chat markers.
    pub fn set_raw_prompt(&mut self, raw: bool) {
        self.raw_prompt = raw;
    }

    /// Load a model file — GGUF or safetensors, chosen by extension.
    ///
    /// For GGUF the tokenizer is read from the file's metadata. For
    /// safetensors a `tokenizer.json` is expected next to the weights.
    pub fn load(&mut self, model_path: &str) -> Result<(), GlError> {
        let ext = Path::new(model_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        match ext.as_str() {
            "gguf" => {
                let gguf = GgufFile::open(model_path)?;
                self.tokenizer = Some(Tokenizer::from_gguf(&gguf)?);
            }
            "safetensors" => {
                let sibling = Path::new(model_path)
                    .parent()
                    .map(|d| d.join("tokenizer.json"))
                    .filter(|p| p.exists())
                    .ok_or_else(|| {
                        GlError::Parse(
                            "safetensors model needs a tokenizer.json next to it".into(),
                        )
                    })?;
                let path_str = sibling.to_string_lossy();
                self.tokenizer = Some(Tokenizer::from_file(&path_str)?);
            }
            other => {
                return Err(GlError::Parse(format!(
                    "unknown model extension '.{other}' (expected .gguf or .safetensors)"
                )))
            }
        }
        self.engine.load_model(model_path)
    }

    /// Borrow the tokenizer for the currently loaded model.
    pub fn tokenizer(&self) -> Option<&Tokenizer> {
        self.tokenizer.as_ref()
    }

    /// Run inference on a text prompt; returns the generated text.
    pub fn infer(&self, prompt: &str, mut config: InferInput) -> Result<String, GlError> {
        config.token_ids = self.encode_prompt(prompt)?;
        Ok(self.engine.infer(config)?.text)
    }

    /// Stream generated text via callback, one token at a time. Returns
    /// the finished request's stats (token counts, prefill/decode timing).
    pub fn stream(
        &self,
        prompt: &str,
        mut config: InferInput,
        on_token: impl Fn(&str) + Send,
    ) -> Result<crate::engine_trait::InferOutput, GlError> {
        config.token_ids = self.encode_prompt(prompt)?;
        self.engine
            .stream(config, &move |_id, text| on_token(text))
    }

    /// Shut the engine down gracefully, consuming the runtime.
    pub fn shutdown(mut self) {
        self.engine.shutdown();
    }

    fn encode_prompt(&self, prompt: &str) -> Result<Vec<u32>, GlError> {
        let tk = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| GlError::Engine("no model loaded".into()))?;
        if !self.raw_prompt {
            // Chat models answer (and emit their stop token) only when the
            // prompt is wrapped in their chat template; raw text makes them
            // ramble as text completion until max_tokens.
            if let Some(ids) = tk.encode_chat(prompt) {
                return Ok(ids);
            }
        }
        Ok(tk.encode(prompt, tk.add_bos_default()))
    }
}
