//! Memory-mapped layer files.
//!
//! ARTX05 §"Memory Mapping Strategy": the runtime maps a layer, executes it,
//! and unmaps it. The OS page cache does the actual I/O; the runtime only
//! hints at the access pattern.
//!
//! A mapping is validated at open time — magic, version, and tensor index are
//! parsed before any caller sees the bytes, so a corrupt file fails loud here
//! rather than as garbage activations later.

use std::fs::File;
use std::path::{Path, PathBuf};

use crate::error::{GllmError, GllmResult};
use crate::types::layer::LayerFile;

/// A read-only mapping of one execution-unit file (a layer, or `GLLMShared.gllm`).
///
/// The mapping is released when this value drops.
#[derive(Debug)]
pub struct LayerMapping {
    /// Index this file holds; `None` for shared components.
    pub layer_index: Option<u32>,
    /// Path the mapping was created from.
    pub path: PathBuf,
    /// Parsed header + tensor index. Data bytes are *not* read at open.
    pub layer: LayerFile,
    /// Size of the mapped file in bytes.
    pub file_size: u64,
    /// Whether this mapping was created ahead of need by the prefetcher.
    pub was_prefetched: bool,
    mmap: memmap2::Mmap,
}

impl LayerMapping {
    /// Map and validate an execution-unit file.
    ///
    /// Parses the header and tensor index eagerly (cheap — a few hundred
    /// bytes) but touches no tensor data, so cost is independent of layer
    /// size. Returns [`GllmError::InvalidMagic`] / [`GllmError::InvalidHeader`]
    /// on a malformed file.
    pub fn open(path: &Path, layer_index: Option<u32>, prefetched: bool) -> GllmResult<Self> {
        let file = File::open(path).map_err(|source| GllmError::MapFailed {
            path: path.display().to_string(),
            source,
        })?;
        let file_size = file
            .metadata()
            .map_err(|source| GllmError::MapFailed {
                path: path.display().to_string(),
                source,
            })?
            .len();

        // SAFETY: mapped read-only and never mutated through this handle. The
        // mapping outlives every slice handed out, because `as_bytes` borrows
        // from `self`. A concurrent truncation by another process could fault
        // on access — the same accepted risk glcore takes for GGUF loading.
        let mmap = unsafe { memmap2::Mmap::map(&file) }.map_err(|source| {
            GllmError::MapFailed {
                path: path.display().to_string(),
                source,
            }
        })?;

        // Layers are consumed front-to-back; ask the kernel to read ahead
        // rather than fault in 4 KB pages on demand. Best-effort, and a no-op
        // on Windows, which has no madvise equivalent here.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Sequential);

        // Validate before returning: a bad file must not reach execution.
        let layer = LayerFile::parse(&mmap)?;

        Ok(LayerMapping {
            layer_index,
            path: path.to_path_buf(),
            layer,
            file_size,
            was_prefetched: prefetched,
            mmap,
        })
    }

    /// The whole mapped file, header included.
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Bytes of one tensor, located via the tensor index.
    ///
    /// Returns `None` if the tensor is absent, or if its recorded range falls
    /// outside the file — a truncated layer yields `None`, never a panic or a
    /// read past the mapping.
    pub fn tensor_bytes(&self, name: &str) -> Option<&[u8]> {
        let (offset, size) = self.layer.absolute_range(name)?;
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(usize::try_from(size).ok()?)?;
        self.mmap.get(start..end)
    }

    /// Number of tensors in this unit.
    pub fn tensor_count(&self) -> usize {
        self.layer.tensor_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::write_test_layer;
    use tempfile::TempDir;

    #[test]
    fn open_maps_and_parses_a_valid_layer() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("GLLMTensorLayer-0000.gllm");
        write_test_layer(&path, &[("attn_q.weight", 64), ("attn_k.weight", 32)]);

        let m = LayerMapping::open(&path, Some(0), false).unwrap();
        assert_eq!(m.layer_index, Some(0));
        assert_eq!(m.tensor_count(), 2);
        assert!(!m.was_prefetched);
        assert_eq!(m.file_size, m.as_bytes().len() as u64);
    }

    #[test]
    fn open_reads_back_exact_tensor_bytes() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("GLLMTensorLayer-0000.gllm");
        write_test_layer(&path, &[("a", 16), ("b", 48)]);

        let m = LayerMapping::open(&path, Some(0), false).unwrap();
        assert_eq!(m.tensor_bytes("a").unwrap().len(), 16);
        assert_eq!(m.tensor_bytes("b").unwrap().len(), 48);
        assert!(m.tensor_bytes("missing").is_none());
    }

    #[test]
    fn open_rejects_a_file_with_bad_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("GLLMTensorLayer-0000.gllm");
        write_test_layer(&path, &[("a", 8)]);

        // Corrupt the magic in place.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes[0] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();

        let err = LayerMapping::open(&path, Some(0), false).unwrap_err();
        assert!(
            matches!(err, GllmError::InvalidMagic(_)),
            "expected InvalidMagic, got {err:?}"
        );
    }

    #[test]
    fn open_missing_file_reports_the_path() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("nope.gllm");

        let err = LayerMapping::open(&path, Some(7), false).unwrap_err();
        match err {
            GllmError::MapFailed { path: p, .. } => assert!(p.contains("nope.gllm")),
            other => panic!("expected MapFailed, got {other:?}"),
        }
    }

    #[test]
    fn prefetched_flag_is_carried_through() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("GLLMTensorLayer-0003.gllm");
        write_test_layer(&path, &[("a", 8)]);

        let m = LayerMapping::open(&path, Some(3), true).unwrap();
        assert!(m.was_prefetched);
        assert_eq!(m.layer_index, Some(3));
    }

    #[test]
    fn dropping_a_mapping_releases_the_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("GLLMTensorLayer-0000.gllm");
        write_test_layer(&path, &[("a", 8)]);

        let m = LayerMapping::open(&path, Some(0), false).unwrap();
        drop(m);

        // On Windows a still-mapped file cannot be removed; this succeeding is
        // the observable proof that Drop unmapped it.
        std::fs::remove_file(&path).expect("file must be unmapped after drop");
    }
}
