use crate::constants::{GLLM_MAGIC, GLLM_VERSION_MAJOR, GLLM_VERSION_MINOR};
use crate::error::{GllmError, GllmResult};
use crate::types::tensor::TensorEntry;

/// Binary header untuk setiap layer file
///
/// Layout (dari ARTX spec):
/// | Offset 0 | 4 bytes | Magic: 0x474C4C4D ("GLLM")
/// | Offset 4 | 1 byte  | Version major
/// | Offset 5 | 1 byte  | Version minor
/// | Offset 6 | 2 bytes | Flags
/// | Offset 8 | 4 bytes | Tensor count
#[derive(Debug, Clone)]
pub struct LayerHeader {
    pub magic: u32,
    pub version_major: u8,
    pub version_minor: u8,
    pub flags: u16,
    pub tensor_count: u32,
}

impl LayerHeader {
    pub fn new(tensor_count: u32) -> Self {
        Self {
            magic: GLLM_MAGIC,
            version_major: GLLM_VERSION_MAJOR,
            version_minor: GLLM_VERSION_MINOR,
            flags: 0,
            tensor_count,
        }
    }

    pub fn validate(&self) -> GllmResult<()> {
        if self.magic != GLLM_MAGIC {
            return Err(GllmError::InvalidMagic(self.magic));
        }
        if self.version_major != GLLM_VERSION_MAJOR {
            return Err(GllmError::UnsupportedVersion {
                major: self.version_major,
                minor: self.version_minor,
            });
        }
        Ok(())
    }

    pub fn size_bytes() -> usize { 12 } // 4+1+1+2+4
}

/// Flags bitmask
pub mod flags {
    pub const LITTLE_ENDIAN: u16 = 0x0001;
    pub const COMPRESSED: u16 = 0x0002;
}

/// In-memory representation of a parsed layer file
#[derive(Debug, Clone)]
pub struct LayerFile {
    pub header: LayerHeader,
    pub tensor_index: Vec<TensorEntry>,
    /// Offset dalam file dimana tensor data mulai
    pub data_offset: u64,
}

impl LayerFile {
    pub fn tensor(&self, name: &str) -> Option<&TensorEntry> {
        self.tensor_index.iter().find(|t| t.name == name)
    }

    pub fn tensor_count(&self) -> usize {
        self.tensor_index.len()
    }

    pub fn total_data_size(&self) -> u64 {
        self.tensor_index.iter().map(|t| t.size).sum()
    }
}
