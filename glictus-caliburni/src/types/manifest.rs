use serde::{Deserialize, Serialize};
use crate::types::tensor::TensorEntry;

/// Model architecture variants supported by GLLM
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Architecture {
    Transformer,
    Mamba,
    Moe,
    Hybrid,
    #[serde(other)]
    Unknown,
}

/// Core model metadata (dari manifest)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub vocab_size: u32,
    pub context_length: u32,
    pub embedding_length: u32,
    pub num_layers: u32,
    pub num_heads: u32,
    pub head_count_kv: Option<u32>,
    pub rope_dims: Option<u32>,
    pub rope_freq_base: Option<f64>,
    #[serde(flatten)]
    pub custom: std::collections::HashMap<String, serde_json::Value>,
}

/// Shared component entry dalam manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SharedComponent {
    pub file: String,
    pub checksum: String,
    pub tensors: Vec<TensorEntry>,
}

/// Layer entry dalam manifest
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerEntry {
    pub index: u32,
    pub file: String,
    pub checksum: String,
    #[serde(rename = "type")]
    pub layer_type: String,   // URI: "gllm:transformer/standard@v1"
    pub tensors: Vec<TensorEntry>,
    pub device: Option<String>, // "cpu", "cuda:0", "rank:0/cuda:0"
}

/// Projector entry (optional, multimodal)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectorEntry {
    pub file: String,
    pub checksum: String,
    #[serde(rename = "type")]
    pub projector_type: String,
}

/// Root manifest struct — gllm.json
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GllmManifest {
    pub gllm_version: String,
    pub model_id: String,
    pub architecture: Architecture,
    pub parameters: u64,
    pub quantization: Option<String>,
    pub metadata: ModelMetadata,
    pub shared: SharedComponent,
    pub layers: Vec<LayerEntry>,
    pub projector: Option<ProjectorEntry>,
    pub extensions: Vec<String>,
}

impl GllmManifest {
    pub fn num_layers(&self) -> usize { self.layers.len() }

    pub fn layer_file(&self, index: usize) -> Option<&LayerEntry> {
        self.layers.get(index)
    }

    pub fn all_extension_uris(&self) -> &[String] {
        &self.extensions
    }

    /// Total estimated size in bytes from tensor metadata
    pub fn estimated_memory_bytes(&self) -> u64 {
        let shared: u64 = self.shared.tensors.iter().map(|t| t.size).sum();
        let layers: u64 = self.layers.iter()
            .flat_map(|l| l.tensors.iter())
            .map(|t| t.size)
            .sum();
        shared + layers
    }
}
