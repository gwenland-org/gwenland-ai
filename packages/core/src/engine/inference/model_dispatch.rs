// engine/inference/model_dispatch.rs — GGUF architecture → candle model mapping.
//
// Reads the `general.architecture` metadata key from a GGUF file header and
// maps it to the `ModelKind` enum so the runner can pick the correct
// candle-transformers model implementation.
//
// Supported architectures mirror what candle-transformers ships:
//   llama    → LLaMA 3 / LLaMA 2
//   mistral  → Mistral 7B variants
//   qwen2    → Qwen2 / Qwen3
//   phi3     → Phi-3 mini/medium
//
// If the architecture key is absent or unrecognised the caller receives an
// explicit error rather than a silent wrong-model crash.

use anyhow::{bail, Context, Result};
use std::path::Path;

/// Supported model architectures for native inference.
#[derive(Debug, Clone, PartialEq)]
pub enum ModelKind {
    LLaMA3,
    Mistral,
    Qwen,
    Phi3,
}

impl ModelKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelKind::LLaMA3  => "LLaMA3",
            ModelKind::Mistral => "Mistral",
            ModelKind::Qwen    => "Qwen",
            ModelKind::Phi3    => "Phi3",
        }
    }
}

/// Detect the model kind by reading the `general.architecture` key from a GGUF file.
///
/// Why parse manually instead of using a GGUF crate?
/// candle-transformers exposes `gguf_file::Content` for exactly this purpose.
/// Using it keeps the dependency count minimal.
pub fn detect_from_gguf(path: &Path) -> Result<ModelKind> {
    // Use candle_core's built-in GGUF reader.
    use candle_core::quantized::gguf_file;
    use std::fs::File;
    use std::io::BufReader;

    let f = File::open(path)
        .with_context(|| format!("cannot open GGUF file: {}", path.display()))?;
    let mut reader = BufReader::new(f);

    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("failed to parse GGUF header: {}", path.display()))?;

    // candle_core::quantized::gguf_file::Value has a to_string() method that
    // returns Result<&str>. We convert &str to String for the owned arch variable.
    let arch: String = content
        .metadata
        .get("general.architecture")
        .and_then(|v| v.to_string().ok().map(|s| s.to_string()))
        .unwrap_or_default();

    map_architecture(&arch)
}

/// Map an architecture string from GGUF metadata to a `ModelKind`.
///
/// Matching is case-insensitive and substring-based so "qwen2" and "qwen3"
/// both resolve to `Qwen`, matching candle-transformers' unified Qwen model.
pub fn map_architecture(arch: &str) -> Result<ModelKind> {
    let lower = arch.to_lowercase();
    if lower.contains("llama") {
        return Ok(ModelKind::LLaMA3);
    }
    if lower.contains("mistral") {
        return Ok(ModelKind::Mistral);
    }
    if lower.contains("qwen") {
        return Ok(ModelKind::Qwen);
    }
    if lower.contains("phi") {
        return Ok(ModelKind::Phi3);
    }

    bail!(
        "Unsupported architecture '{}'. Supported: llama, mistral, qwen, phi.",
        arch
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_dispatch_qwen() {
        assert_eq!(map_architecture("qwen2").unwrap(), ModelKind::Qwen);
        assert_eq!(map_architecture("qwen3").unwrap(), ModelKind::Qwen);
        assert_eq!(map_architecture("Qwen2").unwrap(), ModelKind::Qwen);
    }

    #[test]
    fn test_model_dispatch_llama() {
        assert_eq!(map_architecture("llama").unwrap(), ModelKind::LLaMA3);
        assert_eq!(map_architecture("LLaMA3").unwrap(), ModelKind::LLaMA3);
        assert_eq!(map_architecture("llama2").unwrap(), ModelKind::LLaMA3);
    }

    #[test]
    fn test_model_dispatch_mistral() {
        assert_eq!(map_architecture("mistral").unwrap(), ModelKind::Mistral);
    }

    #[test]
    fn test_model_dispatch_phi3() {
        assert_eq!(map_architecture("phi3").unwrap(), ModelKind::Phi3);
        assert_eq!(map_architecture("phi-3").unwrap(), ModelKind::Phi3);
    }

    #[test]
    fn test_model_dispatch_unknown_fails() {
        assert!(map_architecture("unknown_arch_xyz").is_err());
    }
}
