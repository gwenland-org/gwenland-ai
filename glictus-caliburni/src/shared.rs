//! Shared components loader (ARTX02 Wave 4).
//!
//! `GLLMShared.gllm` holds the tensors every layer depends on: token
//! embeddings, final norm, output head, and the RoPE tables. ARTX02 scope
//! is structural validation only — header + optional checksum — actual
//! tensor loading is ARTX04+.

use std::path::{Path, PathBuf};

use crate::checksum::sha256_file;
use crate::error::GllmError;
use crate::execution_unit::{ExecutionUnit, ExecutionUnitHeader};

/// Known shared tensor names in GLLM v1.
pub const SHARED_TOKEN_EMBEDDINGS: &str = "token_embeddings";
/// Final RMSNorm weight applied before the output head.
pub const SHARED_OUTPUT_NORM: &str = "output_norm.weight";
/// Output head (LM head) projection weight.
pub const SHARED_OUTPUT_HEAD: &str = "output_head.weight";
/// Precomputed RoPE cosine table.
pub const SHARED_ROPE_COS: &str = "rope_cos";
/// Precomputed RoPE sine table.
pub const SHARED_ROPE_SIN: &str = "rope_sin";

/// Metadata about a validated `GLLMShared.gllm` file.
///
/// Does NOT load tensor data.
#[derive(Debug, Clone)]
pub struct SharedComponents {
    /// Path the shared file was opened from.
    pub path: PathBuf,
    /// Validated file header.
    pub header: ExecutionUnitHeader,
    /// File size in bytes.
    pub file_size: u64,
    /// Whether this shared file has been checksum-verified.
    pub checksum_verified: bool,
}

impl SharedComponents {
    /// Open `GLLMShared.gllm`, validate its header, and — when
    /// `expected_checksum` is given — verify the whole file's SHA-256
    /// against it before trusting the contents.
    pub fn open(path: &Path, expected_checksum: Option<&str>) -> Result<Self, GllmError> {
        let unit = ExecutionUnit::open(path)?;

        let checksum_verified = match expected_checksum {
            Some(expected) => {
                let expected = expected.to_ascii_lowercase();
                let actual = sha256_file(path)?;
                if actual != expected {
                    return Err(GllmError::ChecksumMismatch {
                        file: path.display().to_string(),
                        expected,
                        actual,
                    });
                }
                true
            }
            None => false,
        };

        Ok(Self {
            path: unit.path,
            header: unit.header,
            file_size: unit.file_size,
            checksum_verified,
        })
    }

    /// Returns true if the file has been checksum-verified.
    pub fn is_verified(&self) -> bool {
        self.checksum_verified
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checksum::sha256_bytes;
    use crate::test_helpers::make_test_gllm_file;

    #[test]
    fn test_shared_open_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        make_test_gllm_file(&path);

        let shared = SharedComponents::open(&path, None).unwrap();
        assert_eq!(shared.header, ExecutionUnitHeader::new_v1());
        assert!(!shared.is_verified());
    }

    #[test]
    fn test_shared_open_with_checksum() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        let contents = make_test_gllm_file(&path);

        let shared =
            SharedComponents::open(&path, Some(&sha256_bytes(&contents))).unwrap();
        assert!(shared.is_verified());
        assert_eq!(shared.file_size, contents.len() as u64);
    }

    #[test]
    fn test_shared_open_wrong_checksum() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        make_test_gllm_file(&path);

        let wrong = sha256_bytes(b"not the file");
        let err = SharedComponents::open(&path, Some(&wrong)).unwrap_err();
        assert!(
            matches!(err, GllmError::ChecksumMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn test_shared_open_invalid_header() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        let mut bad = make_test_gllm_file(&path);
        bad[0..4].copy_from_slice(b"XXXX");
        std::fs::write(&path, &bad).unwrap();

        let err = SharedComponents::open(&path, None).unwrap_err();
        assert!(matches!(err, GllmError::InvalidMagic(_)), "got {err:?}");
    }
}
