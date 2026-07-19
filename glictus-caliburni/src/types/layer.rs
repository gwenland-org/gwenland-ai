use crate::execution_unit::ExecutionUnitHeader;
use crate::types::tensor::TensorEntry;

// ARTX01's 12-byte `LayerHeader` was retired by the ARTX04 hybrid
// decision: every execution unit (shared/layer/projector) shares the
// 16-byte `ExecutionUnitHeader`, which carries ARTX04's tensor_count at
// bytes 8..12. See notes/gllm-layerheader-vs-executionunitheader.md.

/// Flags bitmask for the header's flags field (ARTX04: "endianness,
/// compression").
pub mod flags {
    pub const LITTLE_ENDIAN: u16 = 0x0001;
    pub const COMPRESSED: u16 = 0x0002;
}

/// In-memory representation of a parsed layer file
#[derive(Debug, Clone)]
pub struct LayerFile {
    pub header: ExecutionUnitHeader,
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
