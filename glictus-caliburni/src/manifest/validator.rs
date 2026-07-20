//! Semantic manifest validation (ARTX03 Wave 4).
//!
//! JSON parsing only checks syntax; this validator checks that the
//! manifest is internally consistent and executable — rules V01–V17 from
//! the ARTX03 spec. All findings are collected (never short-circuits,
//! never panics) so a broken manifest reports its full damage at once.

use crate::error::GllmError;
use crate::manifest::metadata::{FormatVersion, RUNTIME_FORMAT_VERSION, VersionCompatibility};
use crate::manifest::{ExtensionUri, GllmManifest};

/// Sanity cap for the `parameters` field (rule V17): 2 trillion.
const PARAMETERS_SANITY_CAP: u64 = 2_000_000_000_000;

/// Result of full manifest validation.
#[derive(Debug, Default)]
pub struct ValidationResult {
    /// Fatal errors — the package cannot be loaded.
    pub errors: Vec<String>,
    /// Non-fatal warnings — loadable, but may behave unexpectedly.
    pub warnings: Vec<String>,
}

impl ValidationResult {
    /// True when no fatal errors were found.
    pub fn is_ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// True when any warnings were collected.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

/// Semantic validator for a parsed [`GllmManifest`].
pub struct ManifestValidator<'a> {
    manifest: &'a GllmManifest,
}

impl<'a> ManifestValidator<'a> {
    /// Create a validator borrowing the manifest.
    pub fn new(manifest: &'a GllmManifest) -> Self {
        Self { manifest }
    }

    /// Run every validation rule, collecting all errors and warnings.
    pub fn validate(&self) -> ValidationResult {
        let mut result = ValidationResult::default();
        let m = self.manifest;

        // V01/V02: format version compatibility.
        let runtime = FormatVersion(RUNTIME_FORMAT_VERSION.to_string());
        if m.gllm_version.parts().is_none() {
            result.errors.push(format!(
                "gllm_version {:?} is not MAJOR.MINOR.PATCH",
                m.gllm_version.0
            ));
        } else {
            match m.gllm_version.is_compatible_with(&runtime) {
                VersionCompatibility::Compatible => {}
                VersionCompatibility::MinorMismatch { manifest, runtime } => {
                    result.warnings.push(format!(
                        "V02: manifest version {manifest} vs runtime {runtime} (minor mismatch)"
                    ));
                }
                VersionCompatibility::Incompatible { manifest, runtime } => {
                    result.errors.push(
                        GllmError::VersionMismatch {
                            manifest_version: manifest,
                            supported: runtime,
                        }
                        .to_string(),
                    );
                }
            }
        }

        // V03/V04: identity fields.
        if m.model_id.is_empty() {
            result.errors.push("V03: model_id is empty".into());
        }
        if m.architecture.is_empty() {
            result.errors.push("V04: architecture is empty".into());
        }

        // V05: metadata sanity (also covers MoE consistency).
        if let Err(e) = m.metadata.validate() {
            result.errors.push(format!("V05: {e}"));
        }

        // V16: KV heads cannot exceed query heads.
        if m.metadata.head_count_kv > m.metadata.num_heads {
            result.errors.push(format!(
                "V16: head_count_kv={} > num_heads={}",
                m.metadata.head_count_kv, m.metadata.num_heads
            ));
        }

        // V06: layer count agreement.
        if m.metadata.num_layers as usize != m.layers.len() {
            result.errors.push(format!(
                "V06: num_layers={} but {} layer entries in manifest",
                m.metadata.num_layers,
                m.layers.len()
            ));
        }

        // V07: indices 0-based, sequential, no gaps.
        for (i, layer) in m.layers.iter().enumerate() {
            if layer.index != i as u32 {
                result.errors.push(
                    GllmError::LayerIndexGap {
                        expected: i as u32,
                        found: layer.index,
                    }
                    .to_string(),
                );
                break; // one gap poisons every later position; report once
            }
        }

        // V08: filename must match index.
        for layer in &m.layers {
            if layer.file != layer.expected_filename() {
                result.errors.push(format!(
                    "V08: layer {} file {:?}, expected {:?}",
                    layer.index,
                    layer.file,
                    layer.expected_filename()
                ));
            }
        }

        // V09/V14: per-layer checksum format and tensor entries.
        for layer in &m.layers {
            if let Err(e) = layer.checksum_hex() {
                result
                    .errors
                    .push(format!("V09: layer {}: {e}", layer.index));
            }
            for t in &layer.tensors {
                if let Err(e) = t.validate() {
                    result
                        .errors
                        .push(format!("V14: layer {}: {e}", layer.index));
                }
            }
            if ExtensionUri::parse(&layer.layer_type.0).is_err() {
                result.errors.push(format!(
                    "V11: layer {} type {:?} is not a valid extension URI",
                    layer.index, layer.layer_type.0
                ));
            }
        }

        // V10/V13: shared checksum format and tensor entries.
        if let Err(e) = m.shared.checksum_hex() {
            result.errors.push(format!("V10: shared: {e}"));
        }
        for t in &m.shared.tensors {
            if let Err(e) = t.validate() {
                result.errors.push(format!("V13: shared: {e}"));
            }
        }

        // V11: every layer type must be registered in `extensions`.
        for layer in &m.layers {
            if !m.extensions.contains(&layer.layer_type) {
                result.errors.push(format!(
                    "V11: layer {} uses extension {} not registered in extensions list",
                    layer.index, layer.layer_type
                ));
            }
        }

        // V12 [warning]: projector type should be registered too.
        if let Some(projector) = &m.projector {
            if !m.extensions.contains(&projector.projector_type) {
                result.warnings.push(format!(
                    "V12: projector type {} not registered in extensions list",
                    projector.projector_type
                ));
            }
            if let Err(e) = projector.checksum_hex() {
                result.errors.push(format!("V10: projector: {e}"));
            }
        }

        // V15 [warning]: MoE metadata without any MoE-looking layer.
        if m.metadata.is_moe() {
            let any_moe_layer = m
                .layers
                .iter()
                .any(|l| l.layer_type.0.contains("moe"));
            if !any_moe_layer {
                result.warnings.push(
                    "V15: metadata declares MoE (expert_count set) but no layer type contains \"moe\""
                        .into(),
                );
            }
        }

        // V17 [warning]: parameter count plausibility.
        if let Some(p) = m.parameters
            && (p == 0 || p >= PARAMETERS_SANITY_CAP)
        {
            result.warnings.push(format!(
                "V17: parameters={p} is implausible (expected 0 < p < {PARAMETERS_SANITY_CAP})"
            ));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::fixtures;

    fn manifest(json: &str) -> GllmManifest {
        GllmManifest::from_str(json).unwrap()
    }

    fn validate(json: &str) -> ValidationResult {
        ManifestValidator::new(&manifest(json)).validate()
    }

    #[test]
    fn test_validator_valid_manifest() {
        let r = validate(&fixtures::minimal_manifest_json(3));
        assert!(r.is_ok(), "errors: {:?}", r.errors);
        assert!(!r.has_warnings(), "warnings: {:?}", r.warnings);

        let r = validate(&fixtures::full_manifest_json());
        assert!(r.is_ok(), "errors: {:?}", r.errors);
    }

    #[test]
    fn test_validator_version_incompatible() {
        let json = fixtures::minimal_manifest_json(1).replace("\"1.0.0\"", "\"2.0.0\"");
        let r = validate(&json);
        assert!(!r.is_ok());
        assert!(
            r.errors.iter().any(|e| e.contains("version mismatch")
                || e.contains("Format version")),
            "errors: {:?}",
            r.errors
        );
    }

    #[test]
    fn test_validator_version_minor_mismatch() {
        let json = fixtures::minimal_manifest_json(1).replace("\"1.0.0\"", "\"1.1.0\"");
        let r = validate(&json);
        assert!(r.is_ok(), "errors: {:?}", r.errors);
        assert!(r.has_warnings());
        assert!(r.warnings.iter().any(|w| w.starts_with("V02")));
    }

    #[test]
    fn test_validator_empty_model_id() {
        let json =
            fixtures::minimal_manifest_json(1).replace("org.gwenland.test-model", "");
        let r = validate(&json);
        assert!(r.errors.iter().any(|e| e.starts_with("V03")), "{:?}", r.errors);
    }

    #[test]
    fn test_validator_num_layers_mismatch() {
        // metadata says 3 layers, entries say 2.
        let json = fixtures::minimal_manifest_json(2).replace(
            "\"num_layers\":2",
            "\"num_layers\":3",
        );
        let r = validate(&json);
        assert!(r.errors.iter().any(|e| e.starts_with("V06")), "{:?}", r.errors);
    }

    #[test]
    fn test_validator_layer_index_gap() {
        // Indices [0, 2]: layer 1 missing.
        let json = fixtures::minimal_manifest_json(2)
            .replace("\"index\":1", "\"index\":2");
        let r = validate(&json);
        assert!(
            r.errors.iter().any(|e| e.contains("Layer index gap")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn test_validator_layer_filename_mismatch() {
        let json = fixtures::minimal_manifest_json(2)
            .replace("GLLMTensorLayer-0001.gllm", "GLLMTensorLayer-0042.gllm");
        let r = validate(&json);
        assert!(r.errors.iter().any(|e| e.starts_with("V08")), "{:?}", r.errors);
    }

    #[test]
    fn test_validator_bad_layer_checksum() {
        let json = fixtures::minimal_manifest_json(1)
            .replace(&format!("sha256:{}", fixtures::DUMMY_SHA256), "md5:abcd");
        let r = validate(&json);
        assert!(
            r.errors.iter().any(|e| e.starts_with("V09") || e.starts_with("V10")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn test_validator_extension_not_registered() {
        let json = fixtures::minimal_manifest_json(1).replace(
            "\"extensions\":[\"gllm:transformer/standard@v1\"]",
            "\"extensions\":[]",
        );
        let r = validate(&json);
        assert!(
            r.errors
                .iter()
                .any(|e| e.starts_with("V11") && e.contains("not registered")),
            "{:?}",
            r.errors
        );
    }

    #[test]
    fn test_validator_gqa_valid() {
        // full fixture: 64 heads / 8 KV heads.
        let r = validate(&fixtures::full_manifest_json());
        assert!(r.is_ok(), "errors: {:?}", r.errors);
    }

    #[test]
    fn test_validator_kv_heads_exceed_heads() {
        let json = fixtures::minimal_manifest_json(1)
            .replace("\"head_count_kv\":8", "\"head_count_kv\":64");
        let r = validate(&json);
        assert!(r.errors.iter().any(|e| e.starts_with("V16")), "{:?}", r.errors);
    }

    #[test]
    fn test_validator_moe_metadata_no_moe_layer() {
        let json = fixtures::minimal_manifest_json(1).replace(
            "\"num_heads\":8",
            "\"num_heads\":8,\"expert_count\":8,\"expert_used_count\":2",
        );
        let r = validate(&json);
        assert!(r.is_ok(), "errors: {:?}", r.errors);
        assert!(r.warnings.iter().any(|w| w.starts_with("V15")), "{:?}", r.warnings);
    }

    #[test]
    fn test_validator_parameters_implausible() {
        let json = fixtures::minimal_manifest_json(1).replace(
            "\"architecture\":\"transformer\"",
            "\"architecture\":\"transformer\",\"parameters\":9000000000000",
        );
        let r = validate(&json);
        assert!(r.warnings.iter().any(|w| w.starts_with("V17")), "{:?}", r.warnings);
    }

    #[test]
    fn test_validator_collect_all_errors() {
        // Three independent problems: empty model_id, empty architecture,
        // KV heads > heads.
        let json = fixtures::minimal_manifest_json(1)
            .replace("org.gwenland.test-model", "")
            .replace("\"architecture\":\"transformer\"", "\"architecture\":\"\"")
            .replace("\"head_count_kv\":8", "\"head_count_kv\":64");
        let r = validate(&json);
        assert!(r.errors.len() >= 3, "expected >=3 errors, got {:?}", r.errors);
    }
}
