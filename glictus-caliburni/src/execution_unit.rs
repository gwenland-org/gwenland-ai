//! Execution unit file header (ARTX02, revised by the ARTX04 hybrid
//! decision — see `notes/gllm-layerheader-vs-executionunitheader.md`).
//!
//! Every `.gllm` file starts with a fixed 16-byte header:
//!
//! ```text
//! Offset 0  | 4 bytes | Magic: b"GLLM"
//! Offset 4  | 2 bytes | Version (u16 little-endian, currently 1 —
//!           |         | byte-identical to ARTX04's major/minor u8 pair
//!           |         | for v1.0)
//! Offset 6  | 2 bytes | Flags (u16 little-endian; see types::layer::flags)
//! Offset 8  | 4 bytes | Tensor count (u32 little-endian, from ARTX04)
//! Offset 12 | 4 bytes | Reserved, must be all zeros
//! ```
//!
//! The tensor index (ARTX04) begins at offset 16, not ARTX04's offset 12
//! — the declared deviation of the hybrid layout.

use std::io::Read;
use std::path::{Path, PathBuf};

use crate::error::GllmError;

/// Magic bytes at the start of every GLLM execution unit file.
/// Same value as [`crate::constants::GLLM_MAGIC`] (`0x474C4C4D` big-endian),
/// expressed as bytes for direct comparison against file contents.
pub const GLLM_MAGIC: &[u8; 4] = b"GLLM";

/// Total size of the fixed file header in bytes.
pub const GLLM_HEADER_SIZE: usize = 16;

/// The only execution-unit format version this crate can read/write.
pub const GLLM_CURRENT_VERSION: u16 = 1;

/// Parsed 16-byte header of a GLLM execution unit file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionUnitHeader {
    /// Format version (little-endian u16 on disk).
    pub version: u16,
    /// Flag bits (endianness/compression bitmask, see
    /// [`crate::types::layer::flags`]). Writers emit 0; readers carry the
    /// value through without interpreting it.
    pub flags: u16,
    /// Number of tensors in this unit's tensor index (ARTX04). 0 is valid
    /// while the index itself is unimplemented (pre-ARTX04 files).
    pub tensor_count: u32,
}

impl ExecutionUnitHeader {
    /// Parse a header from the first [`GLLM_HEADER_SIZE`] bytes of a file.
    ///
    /// Errors: [`GllmError::InvalidHeader`] when fewer than 16 bytes or the
    /// reserved tail is non-zero, [`GllmError::InvalidMagic`] on wrong magic,
    /// [`GllmError::UnsupportedVersion`] on any version other than
    /// [`GLLM_CURRENT_VERSION`].
    pub fn parse(bytes: &[u8]) -> Result<Self, GllmError> {
        if bytes.len() < GLLM_HEADER_SIZE {
            return Err(GllmError::InvalidHeader(format!(
                "need {GLLM_HEADER_SIZE} bytes, got {}",
                bytes.len()
            )));
        }
        if &bytes[0..4] != GLLM_MAGIC {
            let got = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
            return Err(GllmError::InvalidMagic(got));
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != GLLM_CURRENT_VERSION {
            return Err(GllmError::UnsupportedVersion { version });
        }
        let flags = u16::from_le_bytes([bytes[6], bytes[7]]);
        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        if bytes[12..16].iter().any(|&b| b != 0) {
            return Err(GllmError::InvalidHeader(
                "reserved bytes 12..16 must be zero".into(),
            ));
        }
        Ok(Self {
            version,
            flags,
            tensor_count,
        })
    }

    /// Serialize the header to its 16-byte on-disk form.
    pub fn to_bytes(&self) -> [u8; GLLM_HEADER_SIZE] {
        let mut out = [0u8; GLLM_HEADER_SIZE];
        out[0..4].copy_from_slice(GLLM_MAGIC);
        out[4..6].copy_from_slice(&self.version.to_le_bytes());
        out[6..8].copy_from_slice(&self.flags.to_le_bytes());
        out[8..12].copy_from_slice(&self.tensor_count.to_le_bytes());
        out
    }

    /// Create a default v1 header (flags = 0, no tensors indexed).
    pub fn new_v1() -> Self {
        Self {
            version: GLLM_CURRENT_VERSION,
            flags: 0,
            tensor_count: 0,
        }
    }

    /// Create a v1 header declaring `tensor_count` indexed tensors.
    pub fn new_v1_with_tensors(tensor_count: u32) -> Self {
        Self {
            version: GLLM_CURRENT_VERSION,
            flags: 0,
            tensor_count,
        }
    }
}

/// Lightweight handle to an execution unit file.
///
/// Only validates the header — tensor data is never read here (that is
/// ARTX04+ scope).
#[derive(Debug)]
pub struct ExecutionUnit {
    /// Path the unit was opened from.
    pub path: PathBuf,
    /// Validated file header.
    pub header: ExecutionUnitHeader,
    /// File size in bytes (for pre-validation before mmap).
    pub file_size: u64,
}

impl ExecutionUnit {
    /// Open an execution unit file, read + validate the header only.
    pub fn open(path: &Path) -> Result<Self, GllmError> {
        let mut file = std::fs::File::open(path)?;
        let file_size = file.metadata()?.len();

        let mut buf = [0u8; GLLM_HEADER_SIZE];
        let mut filled = 0;
        while filled < GLLM_HEADER_SIZE {
            let n = file.read(&mut buf[filled..])?;
            if n == 0 {
                return Err(GllmError::InvalidHeader(format!(
                    "{}: file is {filled} bytes, header needs {GLLM_HEADER_SIZE}",
                    path.display()
                )));
            }
            filled += n;
        }

        let header = ExecutionUnitHeader::parse(&buf)?;
        Ok(Self {
            path: path.to_path_buf(),
            header,
            file_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_test_gllm_file;

    #[test]
    fn test_header_parse_valid() {
        let mut bytes = [0u8; GLLM_HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"GLLM");
        bytes[4] = 1; // version 1 LE
        let header = ExecutionUnitHeader::parse(&bytes).unwrap();
        assert_eq!(header.version, 1);
        assert_eq!(header.flags, 0);
        assert_eq!(header.tensor_count, 0);
    }

    #[test]
    fn test_header_tensor_count_roundtrip() {
        // ARTX04 hybrid: tensor_count lives at bytes 8..12.
        let header = ExecutionUnitHeader::new_v1_with_tensors(11);
        let bytes = header.to_bytes();
        assert_eq!(&bytes[8..12], &11u32.to_le_bytes());
        assert_eq!(ExecutionUnitHeader::parse(&bytes).unwrap(), header);
    }

    #[test]
    fn test_header_parse_bad_magic() {
        let mut bytes = [0u8; GLLM_HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"GGUF");
        bytes[4] = 1;
        let err = ExecutionUnitHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, GllmError::InvalidMagic(_)), "got {err:?}");
    }

    #[test]
    fn test_header_parse_wrong_version() {
        let mut bytes = [0u8; GLLM_HEADER_SIZE];
        bytes[0..4].copy_from_slice(b"GLLM");
        bytes[4] = 99;
        let err = ExecutionUnitHeader::parse(&bytes).unwrap_err();
        assert!(
            matches!(err, GllmError::UnsupportedVersion { version: 99 }),
            "got {err:?}"
        );
    }

    #[test]
    fn test_header_parse_nonzero_reserved() {
        let mut bytes = ExecutionUnitHeader::new_v1().to_bytes();
        bytes[12] = 0xFF;
        let err = ExecutionUnitHeader::parse(&bytes).unwrap_err();
        assert!(matches!(err, GllmError::InvalidHeader(_)), "got {err:?}");
    }

    #[test]
    fn test_header_to_bytes_roundtrip() {
        let header = ExecutionUnitHeader {
            version: 1,
            flags: 0,
            tensor_count: 24,
        };
        let parsed = ExecutionUnitHeader::parse(&header.to_bytes()).unwrap();
        assert_eq!(parsed, header);
    }

    #[test]
    fn test_execution_unit_open_valid() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("unit.gllm");
        make_test_gllm_file(&path);

        let unit = ExecutionUnit::open(&path).unwrap();
        assert_eq!(unit.header, ExecutionUnitHeader::new_v1());
        assert!(unit.file_size > GLLM_HEADER_SIZE as u64);
        assert_eq!(unit.path, path);
    }

    #[test]
    fn test_execution_unit_open_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let err = ExecutionUnit::open(&tmp.path().join("nope.gllm")).unwrap_err();
        assert!(matches!(err, GllmError::Io(_)), "got {err:?}");
    }

    #[test]
    fn test_execution_unit_open_too_short() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("short.gllm");
        std::fs::write(&path, b"GLLM").unwrap();

        let err = ExecutionUnit::open(&path).unwrap_err();
        assert!(matches!(err, GllmError::InvalidHeader(_)), "got {err:?}");
    }
}
