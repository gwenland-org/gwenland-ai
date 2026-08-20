//! SHA-256 integrity verification (ARTX02 Wave 3).
//!
//! Fail Fast, Fail Loud: every execution unit carries a SHA-256 checksum
//! (from `checksums.sha256` or the manifest) that is verified before the
//! file is trusted for mmap.
//!
//! # Where the hash actually lives
//!
//! The two primitives below are thin wrappers over [`glcore::hash`]; the
//! algorithm itself is not implemented here. `glbench` needs the same
//! primitive for archive content digests, unconditionally and with zero
//! external dependencies, and a second copy of SHA-256 in the workspace is
//! precisely the failure mode
//! `architecture/gl-stack-audit-2026-07/ARTX2-Quant.md` catalogues.
//!
//! [`ChecksumVerifier`] stays here: it is `.gllm` package policy (parsing
//! `checksums.sha256`, deciding what a mismatch means), not a hash.

use std::path::Path;

use crate::error::GllmError;

/// Length of a SHA-256 digest as lowercase hex.
const SHA256_HEX_LEN: usize = 64;

/// Compute SHA-256 of a file (streamed) and return a lowercase hex string.
///
/// Delegates to [`glcore::hash::sha256_file`]; the [`GllmError`] wrapper is
/// what every caller in this crate already handles.
pub fn sha256_file(path: &Path) -> Result<String, GllmError> {
    Ok(glcore::hash::sha256_file(path)?)
}

/// Compute SHA-256 of a byte slice and return a lowercase hex string.
///
/// Delegates to [`glcore::hash::sha256_bytes`].
pub fn sha256_bytes(data: &[u8]) -> String {
    glcore::hash::sha256_bytes(data)
}

/// Per-file checksum entry (from `checksums.sha256` or the manifest).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChecksumEntry {
    /// Filename relative to the package root, e.g. `GLLMTensorLayer-0000.gllm`.
    pub filename: String,
    /// Expected digest: lowercase hex, 64 chars.
    pub expected_sha256: String,
}

/// Verifier that checks execution units against expected checksums.
#[derive(Debug, Clone)]
pub struct ChecksumVerifier {
    /// All known checksum entries.
    pub entries: Vec<ChecksumEntry>,
}

impl ChecksumVerifier {
    /// Parse a `checksums.sha256` file in standard `sha256sum` output
    /// format: `<64-hex>  <filename>` per line (a leading `*` on the
    /// filename — binary-mode marker — is stripped). Blank lines are
    /// ignored; anything else is an [`GllmError::IntegrityError`].
    pub fn from_file(path: &Path) -> Result<Self, GllmError> {
        let text = std::fs::read_to_string(path)?;
        let mut entries = Vec::new();
        for (lineno, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Some((hash, name)) = line.split_once(char::is_whitespace) else {
                return Err(GllmError::IntegrityError(format!(
                    "{}:{}: expected '<sha256>  <filename>', got {line:?}",
                    path.display(),
                    lineno + 1
                )));
            };
            let hash = hash.to_ascii_lowercase();
            if hash.len() != SHA256_HEX_LEN || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(GllmError::IntegrityError(format!(
                    "{}:{}: {hash:?} is not a 64-char hex SHA-256",
                    path.display(),
                    lineno + 1
                )));
            }
            let name = name.trim_start().trim_start_matches('*');
            if name.is_empty() {
                return Err(GllmError::IntegrityError(format!(
                    "{}:{}: missing filename after checksum",
                    path.display(),
                    lineno + 1
                )));
            }
            entries.push(ChecksumEntry {
                filename: name.to_string(),
                expected_sha256: hash,
            });
        }
        Ok(Self { entries })
    }

    /// Build a verifier from a list of entries directly (e.g. from the
    /// manifest).
    pub fn from_entries(entries: Vec<ChecksumEntry>) -> Self {
        Self { entries }
    }

    /// Look up the expected checksum for a filename.
    pub fn expected_for(&self, filename: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|e| e.filename == filename)
            .map(|e| e.expected_sha256.as_str())
    }

    /// Verify a single file against its expected checksum.
    ///
    /// Errors with [`GllmError::IntegrityError`] when no entry exists for
    /// `filename`, and [`GllmError::ChecksumMismatch`] when digests differ.
    pub fn verify_file(&self, filename: &str, path: &Path) -> Result<(), GllmError> {
        let Some(expected) = self.expected_for(filename) else {
            return Err(GllmError::IntegrityError(format!(
                "no checksum entry for {filename}"
            )));
        };
        let actual = sha256_file(path)?;
        if actual != expected {
            return Err(GllmError::ChecksumMismatch {
                file: filename.to_string(),
                expected: expected.to_string(),
                actual,
            });
        }
        Ok(())
    }

    /// Verify every entry against the files under `base_dir`.
    ///
    /// Does NOT short-circuit — checks all files and returns every
    /// `(filename, error)` pair that failed, so a corrupted package reports
    /// the full damage in one pass.
    pub fn verify_all(&self, base_dir: &Path) -> Vec<(String, GllmError)> {
        let mut failures = Vec::new();
        for entry in &self.entries {
            let path = base_dir.join(&entry.filename);
            if let Err(err) = self.verify_file(&entry.filename, &path) {
                failures.push((entry.filename.clone(), err));
            }
        }
        failures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// SHA-256 of b"hello" — standard published test vector.
    const SHA256_HELLO: &str =
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824";

    #[test]
    fn test_sha256_bytes_known_vector() {
        assert_eq!(sha256_bytes(b"hello"), SHA256_HELLO);
    }

    #[test]
    fn test_sha256_file_matches_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("data.bin");
        let contents = b"some gllm payload bytes";
        fs::write(&path, contents).unwrap();
        assert_eq!(sha256_file(&path).unwrap(), sha256_bytes(contents));
    }

    #[test]
    fn test_parse_checksums_file_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("checksums.sha256");
        fs::write(
            &path,
            format!(
                "{SHA256_HELLO}  GLLMShared.gllm\n\
                 {SHA256_HELLO}  GLLMTensorLayer-0000.gllm\n\
                 {SHA256_HELLO}  *GLLMTensorLayer-0001.gllm\n"
            ),
        )
        .unwrap();

        let verifier = ChecksumVerifier::from_file(&path).unwrap();
        assert_eq!(verifier.entries.len(), 3);
        assert_eq!(verifier.entries[0].filename, "GLLMShared.gllm");
        // Binary-mode marker '*' is stripped.
        assert_eq!(verifier.entries[2].filename, "GLLMTensorLayer-0001.gllm");
        assert_eq!(verifier.expected_for("GLLMTensorLayer-0000.gllm"), Some(SHA256_HELLO));
    }

    #[test]
    fn test_parse_checksums_file_empty() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("checksums.sha256");
        fs::write(&path, "").unwrap();

        let verifier = ChecksumVerifier::from_file(&path).unwrap();
        assert!(verifier.entries.is_empty());
    }

    #[test]
    fn test_parse_checksums_file_malformed_hash() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("checksums.sha256");
        fs::write(&path, "nothex  GLLMShared.gllm\n").unwrap();

        let err = ChecksumVerifier::from_file(&path).unwrap_err();
        assert!(matches!(err, GllmError::IntegrityError(_)), "got {err:?}");
    }

    #[test]
    fn test_verify_file_match() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        fs::write(&path, b"hello").unwrap();

        let verifier = ChecksumVerifier::from_entries(vec![ChecksumEntry {
            filename: "GLLMShared.gllm".into(),
            expected_sha256: SHA256_HELLO.into(),
        }]);
        verifier.verify_file("GLLMShared.gllm", &path).unwrap();
    }

    #[test]
    fn test_verify_file_mismatch() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("GLLMShared.gllm");
        fs::write(&path, b"tampered").unwrap();

        let verifier = ChecksumVerifier::from_entries(vec![ChecksumEntry {
            filename: "GLLMShared.gllm".into(),
            expected_sha256: SHA256_HELLO.into(),
        }]);
        let err = verifier.verify_file("GLLMShared.gllm", &path).unwrap_err();
        assert!(
            matches!(err, GllmError::ChecksumMismatch { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn test_verify_all_partial_failure() {
        let tmp = tempfile::tempdir().unwrap();
        let mut entries = Vec::new();
        for (name, contents) in [
            ("a.gllm", b"aaa".as_slice()),
            ("b.gllm", b"bbb"),
            ("c.gllm", b"ccc"),
        ] {
            fs::write(tmp.path().join(name), contents).unwrap();
            entries.push(ChecksumEntry {
                filename: name.into(),
                expected_sha256: sha256_bytes(contents),
            });
        }
        // Corrupt b.gllm after recording its checksum.
        fs::write(tmp.path().join("b.gllm"), b"CORRUPT").unwrap();

        let failures = ChecksumVerifier::from_entries(entries).verify_all(tmp.path());
        // Exactly one failure, and c.gllm was still checked after it.
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, "b.gllm");
        assert!(
            matches!(failures[0].1, GllmError::ChecksumMismatch { .. }),
            "got {:?}",
            failures[0].1
        );
    }
}
