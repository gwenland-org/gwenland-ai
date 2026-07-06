//! Safetensors file parser, written from scratch.
//!
//! Format: an 8-byte little-endian header length `N`, followed by `N` bytes
//! of JSON metadata mapping tensor names to dtype/shape/offsets, followed by
//! the raw tensor data section. The file is memory-mapped for zero-copy
//! tensor access.

use std::collections::HashMap;
use std::fs::File;

use byteorder::{ByteOrder, LittleEndian};
use serde::Deserialize;

use crate::error::GlError;
use crate::format::gguf::f16_to_f32;

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone, Deserialize)]
pub struct SafetensorsMeta {
    /// Element type as written in the file: `"F32"`, `"F16"`, `"BF16"`, ...
    pub dtype: String,
    /// Dimension sizes, outermost first (row-major).
    pub shape: Vec<usize>,
    /// `[start, end)` byte range relative to the data section.
    pub data_offsets: [usize; 2],
}

/// A parsed, memory-mapped safetensors file.
pub struct SafetensorsFile {
    /// Tensor name → metadata.
    pub tensors: HashMap<String, SafetensorsMeta>,
    /// Absolute byte offset where the tensor data section begins.
    pub data_offset: usize,
    mmap: memmap2::Mmap,
}

impl SafetensorsFile {
    /// Open and mmap a safetensors file, parsing the JSON header.
    pub fn open(path: &str) -> Result<Self, GlError> {
        let file = File::open(path)?;
        // SAFETY: read-only mapping that lives as long as `SafetensorsFile`;
        // all slices handed out borrow from `self`. Concurrent truncation of
        // the file by another process could fault — accepted mmap trade-off.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        if mmap.len() < 8 {
            return Err(GlError::Parse(
                "safetensors: file too small for header".into(),
            ));
        }
        let header_len = usize::try_from(LittleEndian::read_u64(&mmap[0..8]))
            .map_err(|_| GlError::Parse("safetensors: header size overflows usize".into()))?;
        let data_offset = header_len
            .checked_add(8)
            .filter(|&e| e <= mmap.len())
            .ok_or_else(|| {
                GlError::Parse("safetensors: header extends past end of file".into())
            })?;

        let header_json: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&mmap[8..data_offset])
                .map_err(|e| GlError::Parse(format!("safetensors: invalid header JSON: {e}")))?;

        let mut tensors = HashMap::new();
        for (name, value) in header_json {
            if name == "__metadata__" {
                continue; // free-form string map, not a tensor
            }
            let meta: SafetensorsMeta = serde_json::from_value(value).map_err(|e| {
                GlError::Parse(format!("safetensors: bad entry for tensor '{name}': {e}"))
            })?;
            if meta.data_offsets[1] < meta.data_offsets[0]
                || data_offset + meta.data_offsets[1] > mmap.len()
            {
                return Err(GlError::Parse(format!(
                    "safetensors: tensor '{name}' data range out of bounds"
                )));
            }
            tensors.insert(name, meta);
        }

        Ok(SafetensorsFile {
            tensors,
            data_offset,
            mmap,
        })
    }

    /// All tensor names in the file (unordered).
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(String::as_str).collect()
    }

    /// Zero-copy raw bytes of a tensor.
    pub fn tensor_data(&self, name: &str) -> Result<&[u8], GlError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| GlError::Parse(format!("safetensors: no tensor named '{name}'")))?;
        let start = self.data_offset + meta.data_offsets[0];
        let end = self.data_offset + meta.data_offsets[1];
        Ok(&self.mmap[start..end]) // ranges validated in open()
    }

    /// Decode a tensor to `f32`. Supports `F32`, `F16` and `BF16`.
    pub fn to_f32(&self, name: &str) -> Result<Vec<f32>, GlError> {
        let meta = self
            .tensors
            .get(name)
            .ok_or_else(|| GlError::Parse(format!("safetensors: no tensor named '{name}'")))?;
        let dtype = meta.dtype.clone();
        let raw = self.tensor_data(name)?;
        match dtype.as_str() {
            "F32" => Ok(raw.chunks_exact(4).map(LittleEndian::read_f32).collect()),
            "F16" => Ok(raw
                .chunks_exact(2)
                .map(|b| f16_to_f32(LittleEndian::read_u16(b)))
                .collect()),
            "BF16" => Ok(raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((LittleEndian::read_u16(b) as u32) << 16))
                .collect()),
            other => Err(GlError::UnsupportedDtype(other.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    fn build_test_file() -> Vec<u8> {
        let header = r#"{"w":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]},"__metadata__":{"format":"pt"}}"#;
        let mut buf = Vec::new();
        buf.extend_from_slice(&(header.len() as u64).to_le_bytes());
        buf.extend_from_slice(header.as_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0] {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    #[test]
    fn parse_and_read() {
        let bytes = build_test_file();
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&bytes).unwrap();
        tmp.flush().unwrap();

        let st = SafetensorsFile::open(tmp.path().to_str().unwrap()).unwrap();
        assert_eq!(st.tensor_names(), vec!["w"]);
        assert_eq!(st.tensors["w"].shape, vec![2, 2]);
        assert_eq!(st.to_f32("w").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
        assert!(st.to_f32("missing").is_err());
    }

    #[test]
    fn rejects_truncated() {
        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&[1, 2, 3]).unwrap();
        tmp.flush().unwrap();
        assert!(SafetensorsFile::open(tmp.path().to_str().unwrap()).is_err());
    }
}
