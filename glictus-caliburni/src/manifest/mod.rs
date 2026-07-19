//! `gllm.json` manifest — parsing, data types, and semantic validation
//! (ARTX03).
//!
//! The manifest is the single source of truth for a GLLM package: after
//! parsing it, a runtime can construct the execution graph, plan memory,
//! and verify integrity without reading any tensor data.

pub mod metadata;
pub mod types;
pub mod validator;

pub use metadata::{
    FormatVersion, ModelMetadata, RUNTIME_FORMAT_VERSION, RopeScaling, VersionCompatibility,
};
pub use types::{DType, ExtensionUri, TensorEntry, known_extensions};
pub use validator::{ManifestValidator, ValidationResult};

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::GllmError;

/// Canonical layer filename for an index: zero-padded to 3 digits
/// minimum, never truncated (`layer_000.gllm`, `layer_1000.gllm`).
pub fn format_layer_filename(index: u32) -> String {
    format!("layer_{index:03}.gllm")
}

/// Extract the bare hex digest from a `"sha256:<64-hex>"` manifest
/// checksum string.
pub(crate) fn parse_prefixed_checksum(s: &str) -> Result<String, GllmError> {
    let hex = s.strip_prefix("sha256:").ok_or_else(|| {
        GllmError::IntegrityError(format!("missing sha256: prefix in {s:?}"))
    })?;
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(GllmError::IntegrityError(format!(
            "checksum {hex:?} is not a 64-char hex SHA-256"
        )));
    }
    Ok(hex.to_ascii_lowercase())
}

/// Descriptor for `shared.gllm` in the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SharedManifest {
    /// Relative filename, normally `"shared.gllm"`.
    pub file: String,
    /// SHA-256 checksum, prefixed: `"sha256:<hex>"`.
    pub checksum: String,
    /// Tensor metadata list (in file order).
    #[serde(default)]
    pub tensors: Vec<TensorEntry>,
}

impl SharedManifest {
    /// Extract the bare hex checksum (strip the `sha256:` prefix).
    pub fn checksum_hex(&self) -> Result<String, GllmError> {
        parse_prefixed_checksum(&self.checksum)
    }

    /// Validate: file non-empty, checksum well-formed, tensor entries valid.
    pub fn validate(&self) -> Result<(), GllmError> {
        if self.file.is_empty() {
            return Err(GllmError::ManifestValidationError(
                "shared.file is empty".into(),
            ));
        }
        self.checksum_hex()?;
        for t in &self.tensors {
            t.validate()?;
        }
        Ok(())
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }
}

/// Device placement hint for a layer, as written in manifests.
///
/// Deliberately coarse (per ARTX03): the runtime's real device model is
/// `types::execution::Device`; this is only the manifest's serialized
/// hint, with unknown strings preserved as [`DevicePlacement::Unknown`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DevicePlacement {
    /// CPU execution (default).
    Cpu,
    /// First CUDA device.
    #[serde(rename = "cuda:0")]
    Cuda0,
    /// Second CUDA device.
    #[serde(rename = "cuda:1")]
    Cuda1,
    /// Apple Metal.
    Metal,
    /// Vulkan.
    Vulkan,
    /// Unknown/custom device string (forward compat).
    #[serde(other)]
    Unknown,
}

/// Descriptor for a single `layer_NNN.gllm` file in the manifest.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerManifest {
    /// Layer index (0-based); must match the filename.
    pub index: u32,
    /// Relative filename (e.g. `"layer_000.gllm"`).
    pub file: String,
    /// SHA-256 checksum, prefixed: `"sha256:<hex>"`.
    pub checksum: String,
    /// Extension URI identifying this layer's type; must be registered in
    /// the manifest's `extensions` list.
    #[serde(rename = "type")]
    pub layer_type: ExtensionUri,
    /// Tensor metadata list (in file order).
    #[serde(default)]
    pub tensors: Vec<TensorEntry>,
    /// Optional device placement hint (default: CPU).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device: Option<DevicePlacement>,
}

impl LayerManifest {
    /// Extract the bare hex checksum.
    pub fn checksum_hex(&self) -> Result<String, GllmError> {
        parse_prefixed_checksum(&self.checksum)
    }

    /// Expected filename for this layer's index.
    pub fn expected_filename(&self) -> String {
        format_layer_filename(self.index)
    }

    /// Validate: filename matches index, checksum well-formed, layer type
    /// URI well-formed, tensor entries valid.
    pub fn validate(&self) -> Result<(), GllmError> {
        if self.file != self.expected_filename() {
            return Err(GllmError::ManifestValidationError(format!(
                "layer {}: file {:?} does not match expected {:?}",
                self.index,
                self.file,
                self.expected_filename()
            )));
        }
        self.checksum_hex()?;
        ExtensionUri::parse(&self.layer_type.0)?;
        for t in &self.tensors {
            t.validate()?;
        }
        Ok(())
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Effective device (CPU when unspecified).
    pub fn effective_device(&self) -> DevicePlacement {
        self.device.clone().unwrap_or(DevicePlacement::Cpu)
    }
}

/// Optional `projector.gllm` descriptor (multimodal models).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProjectorManifest {
    /// Relative filename, normally `"projector.gllm"`.
    pub file: String,
    /// SHA-256 checksum, prefixed: `"sha256:<hex>"`.
    pub checksum: String,
    /// Extension URI identifying the projector type.
    #[serde(rename = "type")]
    pub projector_type: ExtensionUri,
    /// Tensor metadata list (in file order).
    #[serde(default)]
    pub tensors: Vec<TensorEntry>,
}

impl ProjectorManifest {
    /// Extract the bare hex checksum.
    pub fn checksum_hex(&self) -> Result<String, GllmError> {
        parse_prefixed_checksum(&self.checksum)
    }

    /// Validate: checksum + type URI well-formed, tensor entries valid.
    pub fn validate(&self) -> Result<(), GllmError> {
        self.checksum_hex()?;
        ExtensionUri::parse(&self.projector_type.0)?;
        for t in &self.tensors {
            t.validate()?;
        }
        Ok(())
    }
}

/// Free-form key-value metadata for tooling/provenance.
/// Not interpreted by the runtime — preserved as-is.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct CustomMetadata(pub HashMap<String, serde_json::Value>);

impl CustomMetadata {
    /// Look up a metadata value by key.
    pub fn get(&self, key: &str) -> Option<&serde_json::Value> {
        self.0.get(key)
    }

    /// True when no custom metadata is present.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

/// The complete deserialized `gllm.json` manifest — the single source of
/// truth for a GLLM package.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GllmManifest {
    /// GLLM format version, e.g. `"1.0.0"`.
    pub gllm_version: FormatVersion,
    /// Unique model identifier, e.g. `"org.gwenland.qwen3-1.7b-q4k"`.
    pub model_id: String,
    /// Architecture name, e.g. `"transformer"`, `"mamba"`, `"moe"`.
    pub architecture: String,
    /// Approximate total parameter count.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<u64>,
    /// Primary quantization scheme name (informational).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    /// Model hyperparameters.
    pub metadata: ModelMetadata,
    /// Shared component descriptor.
    pub shared: SharedManifest,
    /// Layer descriptors, one per layer file.
    pub layers: Vec<LayerManifest>,
    /// Optional projector descriptor.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<ProjectorManifest>,
    /// All extension URIs used in this package.
    #[serde(default)]
    pub extensions: Vec<ExtensionUri>,
    /// Optional user-defined metadata.
    #[serde(default, skip_serializing_if = "CustomMetadata::is_empty")]
    pub custom_metadata: CustomMetadata,
}

impl GllmManifest {
    /// Parse a manifest from a JSON string.
    ///
    /// Inherent method rather than `FromStr` for parity with the spec'd
    /// API (`GllmManifest::from_str`).
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(json: &str) -> Result<Self, GllmError> {
        serde_json::from_str(json)
            .map_err(|e| GllmError::ManifestParseError(format!("gllm.json: {e}")))
    }

    /// Parse a manifest from a file path.
    pub fn from_file(path: &Path) -> Result<Self, GllmError> {
        let text = std::fs::read_to_string(path)?;
        serde_json::from_str(&text)
            .map_err(|e| GllmError::ManifestParseError(format!("{}: {e}", path.display())))
    }

    /// Serialize to pretty-printed JSON.
    pub fn to_json_pretty(&self) -> Result<String, GllmError> {
        serde_json::to_string_pretty(self)
            .map_err(|e| GllmError::ManifestParseError(format!("serialize: {e}")))
    }

    /// Number of layer entries.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Layer entry by position (positions equal indices in a valid
    /// manifest — rule V07).
    pub fn layer_file(&self, index: usize) -> Option<&LayerManifest> {
        self.layers.get(index)
    }

    /// Total estimated size in bytes from tensor metadata.
    pub fn estimated_memory_bytes(&self) -> u64 {
        let shared: u64 = self.shared.tensors.iter().map(|t| t.size).sum();
        let layers: u64 = self
            .layers
            .iter()
            .flat_map(|l| l.tensors.iter())
            .map(|t| t.size)
            .sum();
        shared + layers
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::fixtures;

    const HEX64: &str = "0000000000000000000000000000000000000000000000000000000000000000";

    #[test]
    fn test_shared_manifest_checksum_hex_ok() {
        let s = SharedManifest {
            file: "shared.gllm".into(),
            checksum: format!("sha256:{HEX64}"),
            tensors: vec![],
        };
        assert_eq!(s.checksum_hex().unwrap(), HEX64);
    }

    #[test]
    fn test_shared_manifest_checksum_hex_missing_prefix() {
        let s = SharedManifest {
            file: "shared.gllm".into(),
            checksum: HEX64.into(),
            tensors: vec![],
        };
        let err = s.checksum_hex().unwrap_err();
        assert!(matches!(err, GllmError::IntegrityError(_)), "got {err:?}");
    }

    #[test]
    fn test_shared_manifest_tensor_lookup() {
        let m = GllmManifest::from_str(&fixtures::minimal_manifest_json(1)).unwrap();
        assert!(m.shared.tensor("token_embeddings").is_some());
        assert!(m.shared.tensor("nope").is_none());
    }

    #[test]
    fn test_layer_manifest_expected_filename() {
        assert_eq!(format_layer_filename(0), "layer_000.gllm");
        assert_eq!(format_layer_filename(10), "layer_010.gllm");
        assert_eq!(format_layer_filename(100), "layer_100.gllm");
        assert_eq!(format_layer_filename(1000), "layer_1000.gllm");
    }

    #[test]
    fn test_layer_manifest_effective_device_fallback() {
        let m = GllmManifest::from_str(&fixtures::minimal_manifest_json(1)).unwrap();
        assert_eq!(m.layers[0].effective_device(), DevicePlacement::Cpu);
    }

    #[test]
    fn test_gllm_manifest_from_str_minimal() {
        let m = GllmManifest::from_str(&fixtures::minimal_manifest_json(2)).unwrap();
        assert_eq!(m.model_id, "org.gwenland.test-model");
        assert_eq!(m.num_layers(), 2);
        assert_eq!(m.metadata.num_layers, 2);
        assert!(m.projector.is_none());
    }

    #[test]
    fn test_gllm_manifest_from_str_full() {
        let m = GllmManifest::from_str(&fixtures::full_manifest_json()).unwrap();
        assert_eq!(m.num_layers(), 2);
        assert!(m.projector.is_some());
        assert!(m.metadata.is_gqa());
        assert_eq!(m.extensions.len(), 2);
        assert_eq!(
            m.layers[1].effective_device(),
            DevicePlacement::Cuda0,
            "layer 1 carries an explicit cuda:0 hint"
        );
        assert!(m.quantization.is_some());
    }

    #[test]
    fn test_gllm_manifest_roundtrip() {
        let m = GllmManifest::from_str(&fixtures::full_manifest_json()).unwrap();
        let json = m.to_json_pretty().unwrap();
        let back = GllmManifest::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn test_gllm_manifest_from_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("gllm.json");
        std::fs::write(&path, fixtures::minimal_manifest_json(1)).unwrap();
        let m = GllmManifest::from_file(&path).unwrap();
        assert_eq!(m.num_layers(), 1);
    }

    #[test]
    fn test_gllm_manifest_missing_required_field() {
        // Remove metadata.vocab_size from otherwise-valid JSON.
        let json = fixtures::minimal_manifest_json(1).replace("\"vocab_size\":", "\"NOT_vocab\":");
        let err = GllmManifest::from_str(&json).unwrap_err();
        assert!(
            matches!(err, GllmError::ManifestParseError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_custom_metadata_preserved() {
        let m = GllmManifest::from_str(&fixtures::full_manifest_json()).unwrap();
        assert_eq!(
            m.custom_metadata.get("converted_by").and_then(|v| v.as_str()),
            Some("glictus-caliburni-test")
        );
        // Survives a serialize/parse cycle untouched.
        let back = GllmManifest::from_str(&m.to_json_pretty().unwrap()).unwrap();
        assert_eq!(m.custom_metadata, back.custom_metadata);
    }
}
