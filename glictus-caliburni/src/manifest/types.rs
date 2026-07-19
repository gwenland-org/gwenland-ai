//! Core manifest data types (ARTX03 Wave 1): `DType`, `TensorEntry`,
//! `ExtensionUri`.
//!
//! These supersede the ARTX01 placeholders in `types/tensor.rs` and
//! `types/extension.rs`, which now re-export from here.

use serde::{Deserialize, Serialize};

use crate::constants::dtype_codes;
use crate::error::{GllmError, GllmResult};

/// Tensor data type codes supported in GLLM v1.
///
/// Serde names follow the manifest convention (`"F32"`, `"BF16"`,
/// `"Q4_K_M"`, …); unknown strings deserialize to [`DType::Unknown`]
/// instead of failing, so newer packages still parse (forward compat).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DType {
    /// 32-bit float
    F32,
    /// 16-bit float (IEEE 754)
    F16,
    /// Brain float 16
    Bf16,
    /// 8-bit float, E4M3 (`"F8"` accepted as an alias in manifests)
    #[serde(alias = "F8")]
    Fp8E4m3,
    /// 8-bit float, E5M2
    Fp8E5m2,
    /// 4-bit quantization (basic)
    Q4_0,
    /// 4-bit quantization with per-block min
    Q4_1,
    /// 4-bit K-quant
    Q4K,
    /// 4-bit K-quant, medium
    #[serde(rename = "Q4_K_M")]
    Q4Km,
    /// 4-bit K-quant, small (`"Q4KS"` accepted as an alias)
    #[serde(rename = "Q4_K_S", alias = "Q4KS")]
    Q4Ks,
    /// 5-bit quantization, block 32 (llama.cpp fallback rows)
    Q5_0,
    /// 6-bit K-quant, super-block 256 (typical GGUF output heads)
    #[serde(rename = "Q6_K")]
    Q6K,
    /// 8-bit block quantization
    Q8_0,
    /// 8-bit K-quant
    Q8K,
    /// 32-bit integer
    I32,
    /// Unknown dtype (forward compat — newer packages keep parsing)
    #[serde(other)]
    Unknown,
}

/// Binary code emitted for [`DType::Unknown`]; never valid in
/// [`DType::from_code`], so an Unknown can round-trip only as an error.
const DTYPE_CODE_UNKNOWN: u16 = 0xFFFF;

impl DType {
    /// Decode a binary dtype code (from a layer file's tensor index).
    pub fn from_code(code: u16) -> GllmResult<Self> {
        match code {
            dtype_codes::FP32 => Ok(Self::F32),
            dtype_codes::FP16 => Ok(Self::F16),
            dtype_codes::BF16 => Ok(Self::Bf16),
            dtype_codes::FP8_E4M3 => Ok(Self::Fp8E4m3),
            dtype_codes::FP8_E5M2 => Ok(Self::Fp8E5m2),
            dtype_codes::Q4_0 => Ok(Self::Q4_0),
            dtype_codes::Q4_1 => Ok(Self::Q4_1),
            dtype_codes::Q4_K => Ok(Self::Q4K),
            dtype_codes::Q4_K_M => Ok(Self::Q4Km),
            dtype_codes::Q4_K_S => Ok(Self::Q4Ks),
            dtype_codes::Q5_0 => Ok(Self::Q5_0),
            dtype_codes::Q6_K => Ok(Self::Q6K),
            dtype_codes::Q8_0 => Ok(Self::Q8_0),
            dtype_codes::Q8_K => Ok(Self::Q8K),
            dtype_codes::I32 => Ok(Self::I32),
            code => Err(GllmError::UnsupportedDtype(code)),
        }
    }

    /// Encode to the binary dtype code.
    pub fn to_code(self) -> u16 {
        match self {
            Self::F32 => dtype_codes::FP32,
            Self::F16 => dtype_codes::FP16,
            Self::Bf16 => dtype_codes::BF16,
            Self::Fp8E4m3 => dtype_codes::FP8_E4M3,
            Self::Fp8E5m2 => dtype_codes::FP8_E5M2,
            Self::Q4_0 => dtype_codes::Q4_0,
            Self::Q4_1 => dtype_codes::Q4_1,
            Self::Q4K => dtype_codes::Q4_K,
            Self::Q4Km => dtype_codes::Q4_K_M,
            Self::Q4Ks => dtype_codes::Q4_K_S,
            Self::Q5_0 => dtype_codes::Q5_0,
            Self::Q6K => dtype_codes::Q6_K,
            Self::Q8_0 => dtype_codes::Q8_0,
            Self::Q8K => dtype_codes::Q8_K,
            Self::I32 => dtype_codes::I32,
            Self::Unknown => DTYPE_CODE_UNKNOWN,
        }
    }

    /// Exact bytes-per-element for non-block dtypes.
    /// `None` for block quantizations (size depends on block structure)
    /// and for [`DType::Unknown`].
    pub fn bytes_per_element(&self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::Fp8E4m3 | Self::Fp8E5m2 => Some(1),
            _ => None,
        }
    }

    /// Approximate bytes-per-element including block quantizations —
    /// for memory *estimates* only, never for exact offsets.
    pub fn approx_bytes_per_element(&self) -> f64 {
        match self {
            Self::F32 | Self::I32 => 4.0,
            Self::F16 | Self::Bf16 => 2.0,
            Self::Fp8E4m3 | Self::Fp8E5m2 => 1.0,
            Self::Q8_0 | Self::Q8K => 1.0,
            Self::Q5_0 => 22.0 / 32.0,  // GGML block: 22 bytes / 32 elements
            Self::Q6K => 210.0 / 256.0, // GGML super-block: 210 bytes / 256 elements
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::Q4Km | Self::Q4Ks => 0.5,
            // Pessimistic guess for estimation; exact size must come from
            // the tensor entry's `size` field.
            Self::Unknown => 4.0,
        }
    }

    /// Returns true if this is a quantized (block) type.
    pub fn is_quantized(&self) -> bool {
        matches!(
            self,
            Self::Q4_0 | Self::Q4_1 | Self::Q4K | Self::Q4Km | Self::Q4Ks
                | Self::Q5_0 | Self::Q6K | Self::Q8_0 | Self::Q8K
        )
    }
}

/// Metadata for a single tensor within an execution unit.
/// Does NOT contain tensor data — only location and shape info.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorEntry {
    /// Tensor name (e.g. `"attn_q.weight"`, `"token_embeddings"`).
    pub name: String,
    /// Shape as a list of dimensions, e.g. `[8192, 8192]`.
    pub shape: Vec<u64>,
    /// Data type.
    pub dtype: DType,
    /// Byte offset from the start of the tensor data region in the file
    /// (after the 16-byte GLLM header + tensor index).
    pub offset: u64,
    /// Size in bytes.
    pub size: u64,
}

impl TensorEntry {
    /// Number of elements (product of shape dimensions).
    pub fn num_elements(&self) -> u64 {
        self.shape.iter().product()
    }

    /// Rank (number of dimensions).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Validate that the entry is structurally sane: non-empty name and
    /// shape, no zero dimensions, non-zero size.
    pub fn validate(&self) -> Result<(), GllmError> {
        if self.name.is_empty() {
            return Err(GllmError::TensorEntryInvalid("empty tensor name".into()));
        }
        if self.shape.is_empty() {
            return Err(GllmError::TensorEntryInvalid(format!(
                "{}: empty shape",
                self.name
            )));
        }
        if self.shape.contains(&0) {
            return Err(GllmError::TensorEntryInvalid(format!(
                "{}: zero dimension in shape {:?}",
                self.name, self.shape
            )));
        }
        if self.size == 0 {
            return Err(GllmError::TensorEntryInvalid(format!(
                "{}: size is 0",
                self.name
            )));
        }
        Ok(())
    }
}

/// A GLLM extension URI, stored as its raw string form.
///
/// Format: `<namespace>:<category>/<name>@<version>` — e.g.
/// `gllm:transformer/standard@v1`, `myorg:custom/fused@v2`.
///
/// Serde is `transparent`, so manifests carry plain strings; note that
/// deserialization does NOT validate the format — the manifest validator
/// (ARTX03 Wave 4) re-checks every URI it encounters. [`Self::parse`] is
/// the validating constructor.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ExtensionUri(pub String);

impl ExtensionUri {
    /// Parse and validate an extension URI string.
    pub fn parse(s: &str) -> Result<Self, GllmError> {
        let invalid = || {
            GllmError::InvalidExtensionUri(format!(
                "{s:?}: expected <namespace>:<category>/<name>@<version>"
            ))
        };
        let (namespace, rest) = s.split_once(':').ok_or_else(invalid)?;
        let (cat_name, version) = rest.split_once('@').ok_or_else(invalid)?;
        let (category, name) = cat_name.split_once('/').ok_or_else(invalid)?;
        if namespace.is_empty() || category.is_empty() || name.is_empty() || version.is_empty()
        {
            return Err(invalid());
        }
        Ok(Self(s.to_string()))
    }

    /// Namespace part (e.g. `"gllm"`, `"myvendor"`).
    pub fn namespace(&self) -> &str {
        self.0.split(':').next().unwrap_or("")
    }

    fn after_namespace(&self) -> &str {
        self.0.split_once(':').map(|(_, rest)| rest).unwrap_or("")
    }

    /// Category part (e.g. `"transformer"`, `"mamba"`).
    pub fn category(&self) -> &str {
        self.after_namespace().split('/').next().unwrap_or("")
    }

    /// Name part (e.g. `"standard"`, `"moe"`).
    pub fn name(&self) -> &str {
        self.after_namespace()
            .split_once('/')
            .map(|(_, rest)| rest)
            .unwrap_or("")
            .split('@')
            .next()
            .unwrap_or("")
    }

    /// Version part (e.g. `"v1"`).
    pub fn version(&self) -> &str {
        self.0.split_once('@').map(|(_, v)| v).unwrap_or("")
    }

    /// Returns true if this is a first-party GLLM URI (`gllm:` namespace).
    pub fn is_official(&self) -> bool {
        self.namespace() == "gllm"
    }
}

impl std::fmt::Display for ExtensionUri {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Known built-in layer type URIs.
pub mod known_extensions {
    pub const TRANSFORMER_STANDARD: &str = "gllm:transformer/standard@v1";
    pub const TRANSFORMER_MOE: &str = "gllm:transformer/moe@v1";
    pub const TRANSFORMER_MLA: &str = "gllm:transformer/mla@v1";
    pub const MAMBA_STANDARD: &str = "gllm:mamba/standard@v1";
    pub const PROJECTOR_LINEAR: &str = "gllm:projector/linear@v1";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dtype_serde_f32() {
        assert_eq!(serde_json::to_string(&DType::F32).unwrap(), "\"F32\"");
        let back: DType = serde_json::from_str("\"F32\"").unwrap();
        assert_eq!(back, DType::F32);
    }

    #[test]
    fn test_dtype_serde_q4km() {
        assert_eq!(serde_json::to_string(&DType::Q4Km).unwrap(), "\"Q4_K_M\"");
        let back: DType = serde_json::from_str("\"Q4_K_M\"").unwrap();
        assert_eq!(back, DType::Q4Km);
    }

    #[test]
    fn test_dtype_unknown_forward_compat() {
        let back: DType = serde_json::from_str("\"FUTURE_Q2_XXS\"").unwrap();
        assert_eq!(back, DType::Unknown);
    }

    #[test]
    fn test_dtype_is_quantized() {
        assert!(DType::Q4_0.is_quantized());
        assert!(DType::Q8_0.is_quantized());
        assert!(DType::Q4K.is_quantized());
        assert!(!DType::F32.is_quantized());
        assert!(!DType::F16.is_quantized());
        assert!(!DType::Unknown.is_quantized());
    }

    #[test]
    fn test_dtype_bytes_per_element_none_for_blocks() {
        assert_eq!(DType::F32.bytes_per_element(), Some(4));
        assert_eq!(DType::F16.bytes_per_element(), Some(2));
        assert_eq!(DType::Q4_0.bytes_per_element(), None);
        assert_eq!(DType::Q8_0.bytes_per_element(), None);
        assert_eq!(DType::Unknown.bytes_per_element(), None);
    }

    fn entry(shape: Vec<u64>, size: u64) -> TensorEntry {
        TensorEntry {
            name: "attn_q.weight".into(),
            shape,
            dtype: DType::F32,
            offset: 0,
            size,
        }
    }

    #[test]
    fn test_tensor_entry_num_elements() {
        assert_eq!(entry(vec![8192, 8192], 1).num_elements(), 67_108_864);
        assert_eq!(entry(vec![8192, 8192], 1).rank(), 2);
    }

    #[test]
    fn test_tensor_entry_validate_ok() {
        entry(vec![4096, 128], 4096 * 128 * 4).validate().unwrap();
    }

    #[test]
    fn test_tensor_entry_validate_zero_size() {
        let err = entry(vec![4096], 0).validate().unwrap_err();
        assert!(matches!(err, GllmError::TensorEntryInvalid(_)), "got {err:?}");
    }

    #[test]
    fn test_tensor_entry_validate_empty_shape() {
        let err = entry(vec![], 16).validate().unwrap_err();
        assert!(matches!(err, GllmError::TensorEntryInvalid(_)), "got {err:?}");
    }

    #[test]
    fn test_tensor_entry_validate_zero_dim() {
        let err = entry(vec![4096, 0], 16).validate().unwrap_err();
        assert!(matches!(err, GllmError::TensorEntryInvalid(_)), "got {err:?}");
    }

    #[test]
    fn test_extension_uri_parse_valid() {
        let uri = ExtensionUri::parse("gllm:transformer/standard@v1").unwrap();
        assert_eq!(uri.namespace(), "gllm");
        assert_eq!(uri.category(), "transformer");
        assert_eq!(uri.name(), "standard");
        assert_eq!(uri.version(), "v1");
    }

    #[test]
    fn test_extension_uri_parse_invalid() {
        for bad in ["", "gllm", "gllm:transformer", "gllm:transformer/standard",
                    ":transformer/standard@v1", "gllm:/standard@v1",
                    "gllm:transformer/@v1", "gllm:transformer/standard@"] {
            let err = ExtensionUri::parse(bad).unwrap_err();
            assert!(
                matches!(err, GllmError::InvalidExtensionUri(_)),
                "{bad:?} gave {err:?}"
            );
        }
    }

    #[test]
    fn test_extension_uri_is_official() {
        assert!(ExtensionUri::parse("gllm:transformer/moe@v1").unwrap().is_official());
        assert!(!ExtensionUri::parse("myvendor:custom/fused@v2").unwrap().is_official());
    }

    #[test]
    fn test_extension_uri_display() {
        let s = "gllm:adapter/lora@v1";
        let uri = ExtensionUri::parse(s).unwrap();
        assert_eq!(uri.to_string(), s);
        // Serde transparent: serializes as a bare string.
        assert_eq!(serde_json::to_string(&uri).unwrap(), format!("{s:?}"));
    }
}
