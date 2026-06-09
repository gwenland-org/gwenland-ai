// engine/inference/config.rs — Configuration for inference backend selection.
//
// Defines `InferenceConfig`, which controls which backend to use ("candle",
// "mistralrs", or "auto"), where to find models, and default generation
// parameters. Integrates with `GwenConfig` through serde default.
//
// Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 13.1, 13.6

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::params::InferParams;

/// Configuration for the inference subsystem.
///
/// This struct controls which backend is active, where models are stored, and
/// the default generation parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    /// Backend name: "candle", "mistralrs", or "auto".
    ///
    /// The "auto" option attempts to load backends in the order:
    /// ["mistralrs", "candle"], picking the first available.
    pub backend: String,

    /// Model identifier or filename (e.g. "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf").
    ///
    /// Paths starting with "/" or "./" are treated as absolute or relative and
    /// used as-is. Other values are joined to `model_path`.
    pub model: String,

    /// Base directory for models.
    ///
    /// When `model` is not an absolute/relative path, this directory is used
    /// as the prefix.
    pub model_path: PathBuf,

    /// Default generation parameters (temperature, top-p, max tokens, etc.).
    pub params: InferParams,

    /// Optional tokenizer override (Hugging Face model ID).
    ///
    /// When set, this tokenizer is loaded instead of using the model's
    /// embedded tokenizer.
    pub tokenizer_id: Option<String>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        let model_path = dirs::config_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("gwen")
            .join("models");

        Self {
            backend: "candle".to_string(),
            model: String::new(),
            model_path,
            params: InferParams::default(),
            tokenizer_id: None,
        }
    }
}

impl InferenceConfig {
    /// Validate the configuration fields.
    ///
    /// # Errors
    ///
    /// Returns `Err` when:
    /// - `backend` is not one of {"candle", "mistralrs", "auto"}
    ///   (Requirement 8.6, 16.1)
    /// - `model` is empty (Requirement 8.3)
    /// - `model_path` does not exist (Requirement 8.7, 15.3, 15.4)
    /// - `params` fail validation (delegates to `InferParams::validate`)
    pub fn validate(&self) -> Result<()> {
        // Requirement 8.6, 16.1, 21.2 — backend whitelist.
        // "candle" is the legacy alias for "candle-ggqr" and remains valid so
        // that existing config files don't break (selector.rs maps it at runtime).
        if !["candle", "candle-ggqr", "mistralrs", "auto"].contains(&self.backend.as_str()) {
            bail!(
                "backend must be one of 'candle', 'candle-ggqr', 'mistralrs', or 'auto'; got '{}'",
                self.backend
            );
        }

        // Requirement 8.3 — non-empty model
        if self.model.is_empty() {
            bail!("model field must not be empty");
        }

        // Requirement 8.7, 15.3, 15.4 — model_path existence
        if !self.model_path.exists() {
            bail!(
                "model_path does not exist: {}",
                self.model_path.display()
            );
        }

        // Delegate to InferParams validation
        self.params.validate().context("params validation failed")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default values ────────────────────────────────────────────────────────

    #[test]
    fn default_backend_is_candle() {
        assert_eq!(InferenceConfig::default().backend, "candle");
    }

    #[test]
    fn default_model_is_empty() {
        assert!(InferenceConfig::default().model.is_empty());
    }

    #[test]
    fn default_params_is_inferparams_default() {
        let cfg = InferenceConfig::default();
        let expected = InferParams::default();
        assert_eq!(cfg.params.max_tokens, expected.max_tokens);
        assert!((cfg.params.temperature - expected.temperature).abs() < f32::EPSILON);
    }

    #[test]
    fn default_tokenizer_id_is_none() {
        assert!(InferenceConfig::default().tokenizer_id.is_none());
    }

    // ── Validation: backend whitelist ─────────────────────────────────────────

    #[test]
    fn validate_accepts_candle() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir(); // guaranteed to exist
        cfg.backend = "candle".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_candle_ggqr() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.backend = "candle-ggqr".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_mistralrs() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.backend = "mistralrs".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_accepts_auto() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.backend = "auto".to_string();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn validate_rejects_unknown_backend() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.backend = "llama.cpp".to_string();
        assert!(cfg.validate().is_err());
    }

    // ── Validation: non-empty model ───────────────────────────────────────────

    #[test]
    fn validate_rejects_empty_model() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "".to_string();
        cfg.model_path = std::env::temp_dir();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_non_empty_model() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        assert!(cfg.validate().is_ok());
    }

    // ── Validation: model_path existence ──────────────────────────────────────

    #[test]
    fn validate_rejects_nonexistent_model_path() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = PathBuf::from("/nonexistent/path/that/should/never/exist");
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_existing_model_path() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        assert!(cfg.validate().is_ok());
    }

    // ── Validation: params delegation ─────────────────────────────────────────

    #[test]
    fn validate_rejects_invalid_params() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.params.temperature = 0.0; // invalid
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_accepts_valid_params() {
        let mut cfg = InferenceConfig::default();
        cfg.model = "test.gguf".to_string();
        cfg.model_path = std::env::temp_dir();
        cfg.params.temperature = 0.7;
        assert!(cfg.validate().is_ok());
    }
}
