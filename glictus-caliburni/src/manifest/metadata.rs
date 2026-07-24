//! Model metadata & format versioning (ARTX03 Wave 2).

use serde::{Deserialize, Serialize};

use crate::error::GllmError;

/// The GLLM format version this runtime implements.
pub const RUNTIME_FORMAT_VERSION: &str = "1.0.0";

/// Default RoPE frequency base when the manifest omits it.
const DEFAULT_ROPE_FREQ_BASE: f64 = 10_000.0;

/// Default RMSNorm epsilon when the manifest omits it — matches
/// `glproc::loader`'s own fallback for GGUF sources that don't declare
/// `{arch}.attention.layer_norm_rms_epsilon`.
const DEFAULT_RMS_EPS: f64 = 1e-5;

/// GLLM format version (semantic versioning), stored as its raw string.
///
/// Serde is `transparent`; like [`super::types::ExtensionUri`],
/// deserialization does not validate — [`Self::parse`] is the validating
/// constructor and the manifest validator re-checks.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct FormatVersion(pub String);

impl FormatVersion {
    /// Parse and validate a `"MAJOR.MINOR.PATCH"` version string.
    pub fn parse(s: &str) -> Result<Self, GllmError> {
        match Self(s.to_string()).parts() {
            Some(_) => Ok(Self(s.to_string())),
            None => Err(GllmError::ManifestParseError(format!(
                "invalid format version {s:?}: expected MAJOR.MINOR.PATCH"
            ))),
        }
    }

    /// The `(major, minor, patch)` triple; `None` if the string is not a
    /// valid three-part numeric version.
    pub fn parts(&self) -> Option<(u32, u32, u32)> {
        let mut it = self.0.split('.');
        let major = it.next()?.parse().ok()?;
        let minor = it.next()?.parse().ok()?;
        let patch = it.next()?.parse().ok()?;
        if it.next().is_some() {
            return None;
        }
        Some((major, minor, patch))
    }

    /// Major component (0 when the string is malformed — use
    /// [`Self::parts`] when malformed must be detected).
    pub fn major(&self) -> u32 {
        self.parts().map(|(m, _, _)| m).unwrap_or(0)
    }

    /// Minor component (0 when malformed).
    pub fn minor(&self) -> u32 {
        self.parts().map(|(_, m, _)| m).unwrap_or(0)
    }

    /// Patch component (0 when malformed).
    pub fn patch(&self) -> u32 {
        self.parts().map(|(_, _, p)| p).unwrap_or(0)
    }

    /// Compatibility of a manifest carrying this version with a runtime
    /// implementing `runtime_version`. Same major = loadable (minor
    /// mismatch is a warning); different major = refuse.
    pub fn is_compatible_with(&self, runtime_version: &FormatVersion) -> VersionCompatibility {
        if self.major() != runtime_version.major() {
            VersionCompatibility::Incompatible {
                manifest: self.0.clone(),
                runtime: runtime_version.0.clone(),
            }
        } else if self.minor() != runtime_version.minor() {
            VersionCompatibility::MinorMismatch {
                manifest: self.0.clone(),
                runtime: runtime_version.0.clone(),
            }
        } else {
            VersionCompatibility::Compatible
        }
    }
}

/// Outcome of a manifest-vs-runtime format version check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VersionCompatibility {
    /// Fully compatible.
    Compatible,
    /// Same major, different minor — proceed with warning.
    MinorMismatch { manifest: String, runtime: String },
    /// Different major — refuse to load.
    Incompatible { manifest: String, runtime: String },
}

/// RoPE scaling configuration (optional).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RopeScaling {
    /// Scaling type, e.g. `"linear"`, `"yarn"`, `"dynamic"`.
    #[serde(rename = "type")]
    pub scaling_type: String,
    /// Scaling factor.
    pub factor: f64,
}

/// Required and optional model hyperparameters.
///
/// Embedded in the manifest — the runtime must be able to construct the
/// execution graph from this alone, without reading tensor data.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelMetadata {
    // --- Required fields ---
    /// Vocabulary size (e.g. 32000, 151936).
    pub vocab_size: u64,
    /// Maximum context length in tokens.
    pub context_length: u64,
    /// Hidden dimension (embedding length).
    pub embedding_length: u64,
    /// Total number of transformer layers.
    pub num_layers: u32,
    /// Number of attention query heads.
    pub num_heads: u32,
    /// Number of KV heads (for GQA; equals `num_heads` if MHA).
    pub head_count_kv: u32,

    // --- Optional fields ---
    /// RoPE dimensionality (often = embedding_length / num_heads).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_dims: Option<u32>,
    /// RoPE frequency base (default 10000.0 when absent).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_freq_base: Option<f64>,
    /// RoPE scaling config (long-context variants).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rope_scaling: Option<RopeScaling>,
    /// Total number of experts (MoE models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_count: Option<u32>,
    /// Active experts per token (MoE models).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expert_used_count: Option<u32>,
    /// Sliding window attention size (e.g. Mistral).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sliding_window: Option<u32>,
    /// Whether attention layers use bias.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attention_bias: Option<bool>,
    /// RMSNorm epsilon (default 1e-5 when absent — see
    /// [`Self::effective_rms_eps`]). Sourced from GGUF's
    /// `{arch}.attention.layer_norm_rms_epsilon`; many real models (e.g.
    /// Qwen2.5) declare `1e-6`, not the default, so this must be read from
    /// the source model rather than assumed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rms_eps: Option<f64>,
    /// Token ids that end generation, extracted from the source GGUF at
    /// conversion time (`tokenizer.ggml.eos_token_id` and/or
    /// `tokenizer.ggml.eos_token_ids` — see `converter::extract_eos_token_ids`).
    /// Empty when absent from the source or on packages converted before
    /// this field existed — never treated as an error by itself; a `.gllm`
    /// package with no EOS ids simply cannot self-stop and falls back to
    /// whatever `InferInput::stopping` a caller supplies (see
    /// `glcore::stopping` and `GllmEngine::run_request`'s manifest-first,
    /// `InferInput`-fallback priority).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub eos_token_ids: Vec<u32>,
}

impl ModelMetadata {
    /// Validate that all required fields are sane (non-zero where zero is
    /// meaningless, MoE fields internally consistent).
    ///
    /// `num_layers` MAY be 0 (shared-only package); layer-count agreement
    /// with the manifest's layer list is the validator's job (rule V06).
    pub fn validate(&self) -> Result<(), GllmError> {
        let nonzero: [(&str, u64); 5] = [
            ("vocab_size", self.vocab_size),
            ("context_length", self.context_length),
            ("embedding_length", self.embedding_length),
            ("num_heads", self.num_heads as u64),
            ("head_count_kv", self.head_count_kv as u64),
        ];
        for (field, value) in nonzero {
            if value == 0 {
                return Err(GllmError::ManifestValidationError(format!(
                    "metadata.{field} must be non-zero"
                )));
            }
        }
        if let (Some(count), Some(used)) = (self.expert_count, self.expert_used_count)
            && used > count
        {
            return Err(GllmError::ManifestValidationError(format!(
                "metadata.expert_used_count ({used}) > expert_count ({count})"
            )));
        }
        Ok(())
    }

    /// Head dimension = embedding_length / num_heads; `None` when the
    /// division is not exact (or num_heads is 0).
    pub fn head_dim(&self) -> Option<u64> {
        let heads = self.num_heads as u64;
        if heads == 0 || !self.embedding_length.is_multiple_of(heads) {
            return None;
        }
        Some(self.embedding_length / heads)
    }

    /// Returns true if this is a MoE model.
    pub fn is_moe(&self) -> bool {
        self.expert_count.is_some()
    }

    /// Returns true if this uses Grouped Query Attention.
    pub fn is_gqa(&self) -> bool {
        self.head_count_kv != self.num_heads
    }

    /// Effective RoPE frequency base (10000.0 when absent).
    pub fn effective_rope_freq_base(&self) -> f64 {
        self.rope_freq_base.unwrap_or(DEFAULT_ROPE_FREQ_BASE)
    }

    /// Effective RMSNorm epsilon (1e-5 when absent).
    pub fn effective_rms_eps(&self) -> f64 {
        self.rms_eps.unwrap_or(DEFAULT_RMS_EPS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runtime() -> FormatVersion {
        FormatVersion(RUNTIME_FORMAT_VERSION.to_string())
    }

    fn metadata() -> ModelMetadata {
        ModelMetadata {
            vocab_size: 151_936,
            context_length: 32_768,
            embedding_length: 4096,
            num_layers: 32,
            num_heads: 32,
            head_count_kv: 8,
            rope_dims: Some(128),
            rope_freq_base: None,
            rope_scaling: None,
            expert_count: None,
            expert_used_count: None,
            sliding_window: None,
            attention_bias: None,
            rms_eps: None,
            eos_token_ids: Vec::new(),
        }
    }

    #[test]
    fn test_format_version_parse_valid() {
        let v = FormatVersion::parse("1.0.0").unwrap();
        assert_eq!((v.major(), v.minor(), v.patch()), (1, 0, 0));
        let v = FormatVersion::parse("2.13.7").unwrap();
        assert_eq!((v.major(), v.minor(), v.patch()), (2, 13, 7));
    }

    #[test]
    fn test_format_version_parse_invalid() {
        for bad in ["1.0", "abc", "", "1.0.0.0", "1.x.0"] {
            assert!(FormatVersion::parse(bad).is_err(), "{bad:?} should fail");
        }
    }

    #[test]
    fn test_version_compatible_same_major() {
        let v = FormatVersion::parse("1.0.3").unwrap();
        assert_eq!(v.is_compatible_with(&runtime()), VersionCompatibility::Compatible);
    }

    #[test]
    fn test_version_minor_mismatch() {
        let v = FormatVersion::parse("1.1.0").unwrap();
        assert!(matches!(
            v.is_compatible_with(&runtime()),
            VersionCompatibility::MinorMismatch { .. }
        ));
    }

    #[test]
    fn test_version_incompatible_major() {
        let v = FormatVersion::parse("2.0.0").unwrap();
        assert!(matches!(
            v.is_compatible_with(&runtime()),
            VersionCompatibility::Incompatible { .. }
        ));
    }

    #[test]
    fn test_metadata_validate_ok() {
        metadata().validate().unwrap();
    }

    #[test]
    fn test_metadata_validate_zero_vocab() {
        let mut m = metadata();
        m.vocab_size = 0;
        let err = m.validate().unwrap_err();
        assert!(
            matches!(err, GllmError::ManifestValidationError(_)),
            "got {err:?}"
        );
    }

    #[test]
    fn test_metadata_validate_moe_inconsistent() {
        let mut m = metadata();
        m.expert_count = Some(8);
        m.expert_used_count = Some(16);
        assert!(m.validate().is_err());
    }

    #[test]
    fn test_metadata_head_dim_exact() {
        assert_eq!(metadata().head_dim(), Some(128));
    }

    #[test]
    fn test_metadata_head_dim_inexact() {
        let mut m = metadata();
        m.num_heads = 30;
        assert_eq!(m.head_dim(), None);
    }

    #[test]
    fn test_metadata_is_moe() {
        assert!(!metadata().is_moe());
        let mut m = metadata();
        m.expert_count = Some(8);
        assert!(m.is_moe());
    }

    #[test]
    fn test_metadata_is_gqa() {
        assert!(metadata().is_gqa()); // 32 heads vs 8 KV heads
        let mut m = metadata();
        m.head_count_kv = 32;
        assert!(!m.is_gqa());
    }

    #[test]
    fn test_metadata_effective_rope_freq_base_default() {
        assert_eq!(metadata().effective_rope_freq_base(), 10_000.0);
        let mut m = metadata();
        m.rope_freq_base = Some(1_000_000.0);
        assert_eq!(m.effective_rope_freq_base(), 1_000_000.0);
    }

    #[test]
    fn model_metadata_eos_token_ids_roundtrips_serde() {
        // Absent case: the field must not appear in the JSON at all
        // (backward compatibility with packages converted before it
        // existed — `skip_serializing_if` is the contract under test).
        let absent = metadata();
        assert!(absent.eos_token_ids.is_empty());
        let json = serde_json::to_string(&absent).unwrap();
        assert!(!json.contains("eos_token_ids"), "{json}");
        let back: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back, absent);

        // Present case: Qwen2.5's real dual EOS ids.
        let mut present = metadata();
        present.eos_token_ids = vec![151_645, 151_643];
        let json = serde_json::to_string(&present).unwrap();
        assert!(json.contains("eos_token_ids"), "{json}");
        let back: ModelMetadata = serde_json::from_str(&json).unwrap();
        assert_eq!(back, present);
    }
}
