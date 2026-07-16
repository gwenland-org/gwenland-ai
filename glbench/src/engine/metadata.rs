//! Static metadata about the engine a session ran through.

use crate::core::schema::ToJson;
use crate::export::json::Json;

/// Engine facts captured for a session, mirroring glcore's `EngineSpec` plus
/// what the model file declared.
#[derive(Debug, Clone, Default)]
pub struct EngineMetadata {
    /// Engine name, e.g. `"glcuda"`.
    pub name: String,
    /// Backend kind: `"cpu"`, `"cuda"`, `"vulkan"`, `"metal"`.
    pub backend: String,
    /// Whether the engine was actually available on this machine.
    pub available: bool,
    /// Model architecture as declared by the file, if known (e.g. `"qwen2"`).
    pub model_arch: Option<String>,
    /// Quantization label of the model, if known (e.g. `"Q4_K_M"`).
    pub quantization: Option<String>,
    /// Whether the model is thinking-capable (CoT), as resolved for this run:
    /// GGUF auto-detection, overridden by the workload's `cot_mode` when set.
    /// `None` when the model file could not be probed and no override was given.
    pub thinking_capable: Option<bool>,
}

impl ToJson for EngineMetadata {
    fn to_json(&self) -> Json {
        Json::obj([
            ("name", Json::s(self.name.clone())),
            ("backend", Json::s(self.backend.clone())),
            ("available", Json::Bool(self.available)),
            (
                "model_arch",
                match &self.model_arch {
                    Some(s) => Json::s(s.clone()),
                    None => Json::Null,
                },
            ),
            (
                "quantization",
                match &self.quantization {
                    Some(s) => Json::s(s.clone()),
                    None => Json::Null,
                },
            ),
            (
                "thinking_capable",
                match self.thinking_capable {
                    Some(b) => Json::Bool(b),
                    None => Json::Null,
                },
            ),
        ])
    }
}
