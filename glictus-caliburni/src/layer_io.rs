//! Layer/unit file binary I/O (ARTX04 Waves 2–3): tensor index codec,
//! full-file reader, and a writer for converters/fixtures.
//!
//! File layout (hybrid header decision, see
//! `notes/gllm-layerheader-vs-executionunitheader.md`):
//!
//! ```text
//! [16-byte ExecutionUnitHeader][tensor index][pad][tensor data region]
//! ```
//!
//! Tensor index entry (ARTX04):
//!
//! ```text
//! name_len  u16 LE | name UTF-8 | shape_len u8 | shape u32 LE each
//! dtype u16 LE | offset u64 LE | size u64 LE
//! ```
//!
//! Two declared deviations from the ARTX04 spec text:
//! - the index starts at offset 16, not 12 (hybrid header);
//! - `offset` is relative to the start of the tensor data region (matching
//!   the manifest's `TensorEntry.offset` semantics from ARTX03), not an
//!   absolute file offset. The data region and every tensor within it are
//!   aligned to [`TENSOR_ALIGNMENT`] bytes.

use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::constants::TENSOR_ALIGNMENT;
use crate::error::GllmError;
use crate::execution_unit::{ExecutionUnitHeader, GLLM_HEADER_SIZE};
use crate::manifest::{DType, LayerManifest, TensorEntry};
use crate::types::layer::LayerFile;

/// Round `x` up to the next multiple of `align`.
fn align_up(x: u64, align: u64) -> u64 {
    x.div_ceil(align) * align
}

fn read_array<const N: usize, R: Read>(r: &mut R) -> Result<[u8; N], GllmError> {
    let mut buf = [0u8; N];
    r.read_exact(&mut buf)?;
    Ok(buf)
}

/// Encoded size in bytes of one index entry for this tensor.
pub fn entry_encoded_size(entry: &TensorEntry) -> usize {
    2 + entry.name.len() + 1 + 4 * entry.shape.len() + 2 + 8 + 8
}

/// Read one tensor index entry. Binary u32 dims are widened to the
/// manifest's u64 representation.
pub fn read_entry<R: Read>(r: &mut R) -> Result<TensorEntry, GllmError> {
    let name_len = u16::from_le_bytes(read_array::<2, _>(r)?) as usize;
    let mut name_buf = vec![0u8; name_len];
    r.read_exact(&mut name_buf)?;
    let name = String::from_utf8(name_buf).map_err(|_| {
        GllmError::TensorEntryInvalid("tensor name is not valid UTF-8".into())
    })?;

    let shape_len = read_array::<1, _>(r)?[0] as usize;
    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        shape.push(u32::from_le_bytes(read_array::<4, _>(r)?) as u64);
    }

    let dtype = DType::from_code(u16::from_le_bytes(read_array::<2, _>(r)?))?;
    let offset = u64::from_le_bytes(read_array::<8, _>(r)?);
    let size = u64::from_le_bytes(read_array::<8, _>(r)?);

    Ok(TensorEntry {
        name,
        shape,
        dtype,
        offset,
        size,
    })
}

/// Write one tensor index entry. Rejects entries the binary format cannot
/// represent: >u16 name, >255 dims, dims above u32, or [`DType::Unknown`].
pub fn write_entry<W: Write>(w: &mut W, entry: &TensorEntry) -> Result<(), GllmError> {
    if entry.name.len() > u16::MAX as usize {
        return Err(GllmError::TensorEntryInvalid(format!(
            "tensor name is {} bytes, max {}",
            entry.name.len(),
            u16::MAX
        )));
    }
    if entry.shape.len() > u8::MAX as usize {
        return Err(GllmError::TensorEntryInvalid(format!(
            "{}: {} dims, max {}",
            entry.name,
            entry.shape.len(),
            u8::MAX
        )));
    }
    if entry.dtype == DType::Unknown {
        return Err(GllmError::TensorEntryInvalid(format!(
            "{}: Unknown dtype cannot be encoded",
            entry.name
        )));
    }

    w.write_all(&(entry.name.len() as u16).to_le_bytes())?;
    w.write_all(entry.name.as_bytes())?;
    w.write_all(&[entry.shape.len() as u8])?;
    for &dim in &entry.shape {
        let dim32 = u32::try_from(dim).map_err(|_| {
            GllmError::TensorEntryInvalid(format!(
                "{}: dimension {dim} exceeds u32 (binary shape is u32 per ARTX04)",
                entry.name
            ))
        })?;
        w.write_all(&dim32.to_le_bytes())?;
    }
    w.write_all(&entry.dtype.to_code().to_le_bytes())?;
    w.write_all(&entry.offset.to_le_bytes())?;
    w.write_all(&entry.size.to_le_bytes())?;
    Ok(())
}

impl LayerFile {
    /// Read and validate a unit file's header + full tensor index.
    /// Tensor data is located, bounds-checked, but never read.
    pub fn read(path: &Path) -> Result<Self, GllmError> {
        let file_size = std::fs::metadata(path)?.len();
        let reader = BufReader::new(std::fs::File::open(path)?);
        Self::read_from(reader, file_size, &path.display().to_string())
    }

    /// Parse a unit file's header + tensor index from bytes already in memory
    /// (typically a mapping). Tensor data is located and bounds-checked
    /// against `bytes.len()`, but never read.
    ///
    /// This is the mmap path: it reads only the header and index, so its cost
    /// does not scale with layer size.
    pub fn parse(bytes: &[u8]) -> Result<Self, GllmError> {
        Self::read_from(bytes, bytes.len() as u64, "<mapping>")
    }

    /// Shared implementation behind [`read`](Self::read) and
    /// [`parse`](Self::parse). `unit_size` is the total size of the unit, used
    /// to bounds-check every tensor's region; `source` names it in errors.
    fn read_from<R: Read>(
        mut reader: R,
        unit_size: u64,
        source: &str,
    ) -> Result<Self, GllmError> {
        let header = ExecutionUnitHeader::parse(&read_array::<GLLM_HEADER_SIZE, _>(
            &mut reader,
        )?)?;

        let mut tensor_index = Vec::new();
        let mut index_size = 0usize;
        for _ in 0..header.tensor_count {
            let entry = read_entry(&mut reader)?;
            index_size += entry_encoded_size(&entry);
            tensor_index.push(entry);
        }
        let data_offset = align_up(
            (GLLM_HEADER_SIZE + index_size) as u64,
            TENSOR_ALIGNMENT as u64,
        );

        for entry in &tensor_index {
            entry.validate()?;
            let end = data_offset
                .checked_add(entry.offset)
                .and_then(|s| s.checked_add(entry.size));
            match end {
                Some(end) if end <= unit_size => {}
                _ => {
                    return Err(GllmError::IntegrityError(format!(
                        "{}: tensor {} (region offset {}, size {}) exceeds file size {}",
                        source, entry.name, entry.offset, entry.size, unit_size
                    )));
                }
            }
        }

        Ok(Self {
            header,
            tensor_index,
            data_offset,
        })
    }

    /// Absolute `(file_offset, size)` byte range for a tensor, ready for
    /// seek/mmap. `None` when the name is not in the index.
    pub fn absolute_range(&self, name: &str) -> Option<(u64, u64)> {
        self.tensor(name)
            .map(|t| (self.data_offset + t.offset, t.size))
    }
}

/// Write a complete unit file: header + index + 64-byte-aligned data
/// region, with per-tensor alignment. Offsets are computed here; returns
/// the index entries as written (region-relative offsets).
///
/// This is the fixture/converter path — the runtime never writes.
pub fn write_unit_file(
    path: &Path,
    tensors: &[(&str, &[u64], DType, &[u8])],
) -> Result<Vec<TensorEntry>, GllmError> {
    // Pass 1: entries with sequential, per-tensor-aligned region offsets.
    let mut entries = Vec::with_capacity(tensors.len());
    let mut cursor = 0u64;
    for (name, shape, dtype, data) in tensors {
        cursor = align_up(cursor, TENSOR_ALIGNMENT as u64);
        entries.push(TensorEntry {
            name: (*name).to_string(),
            shape: shape.to_vec(),
            dtype: *dtype,
            offset: cursor,
            size: data.len() as u64,
        });
        cursor += data.len() as u64;
    }

    let index_size: usize = entries.iter().map(entry_encoded_size).sum();
    let data_offset = align_up(
        (GLLM_HEADER_SIZE + index_size) as u64,
        TENSOR_ALIGNMENT as u64,
    );

    let tensor_count = u32::try_from(entries.len()).map_err(|_| {
        GllmError::TensorEntryInvalid(format!("{} tensors exceed u32", entries.len()))
    })?;
    let mut out = Vec::with_capacity(data_offset as usize + cursor as usize);
    out.extend_from_slice(&ExecutionUnitHeader::new_v1_with_tensors(tensor_count).to_bytes());
    for entry in &entries {
        write_entry(&mut out, entry)?;
    }
    out.resize(data_offset as usize, 0);
    for (entry, (_, _, _, data)) in entries.iter().zip(tensors) {
        out.resize((data_offset + entry.offset) as usize, 0);
        out.extend_from_slice(data);
    }
    std::fs::write(path, out)?;
    Ok(entries)
}

/// Cross-check a parsed layer file's binary index against its manifest
/// entry (ARTX04 Wave 4). The package checksum already guarantees the
/// bytes; this guarantees the *metadata* agrees. Returns human-readable
/// mismatch descriptions — empty means fully consistent.
pub fn cross_check_manifest(layer: &LayerFile, manifest: &LayerManifest) -> Vec<String> {
    let mut mismatches = Vec::new();
    if layer.tensor_index.len() != manifest.tensors.len() {
        mismatches.push(format!(
            "tensor count: binary index has {}, manifest lists {}",
            layer.tensor_index.len(),
            manifest.tensors.len()
        ));
    }
    for expected in &manifest.tensors {
        match layer.tensor(&expected.name) {
            None => mismatches.push(format!("tensor {} missing from binary index", expected.name)),
            Some(actual) if actual != expected => {
                mismatches.push(format!(
                    "tensor {}: binary {:?}/{:?} @{}+{} vs manifest {:?}/{:?} @{}+{}",
                    expected.name,
                    actual.shape,
                    actual.dtype,
                    actual.offset,
                    actual.size,
                    expected.shape,
                    expected.dtype,
                    expected.offset,
                    expected.size
                ));
            }
            Some(_) => {}
        }
    }
    for actual in &layer.tensor_index {
        if manifest.tensor(&actual.name).is_none() {
            mismatches.push(format!("tensor {} not listed in manifest", actual.name));
        }
    }
    mismatches
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::make_test_gllm_file;
    use std::io::Cursor;

    fn sample_entry() -> TensorEntry {
        TensorEntry {
            name: "attn_q.weight".into(),
            shape: vec![896, 896],
            dtype: DType::Q4Km,
            offset: 128,
            size: 451_584,
        }
    }

    #[test]
    fn entry_roundtrips_through_binary_encoding() {
        let entry = sample_entry();
        let mut buf = Vec::new();
        write_entry(&mut buf, &entry).unwrap();
        assert_eq!(buf.len(), entry_encoded_size(&entry));
        let back = read_entry(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(back, entry);
    }

    #[test]
    fn entry_write_rejects_unknown_dtype() {
        let mut entry = sample_entry();
        entry.dtype = DType::Unknown;
        let err = write_entry(&mut Vec::new(), &entry).unwrap_err();
        assert!(matches!(err, GllmError::TensorEntryInvalid(_)), "got {err:?}");
    }

    #[test]
    fn entry_write_rejects_dim_above_u32() {
        let mut entry = sample_entry();
        entry.shape = vec![u64::from(u32::MAX) + 1];
        let err = write_entry(&mut Vec::new(), &entry).unwrap_err();
        assert!(matches!(err, GllmError::TensorEntryInvalid(_)), "got {err:?}");
    }

    #[test]
    fn entry_read_truncated_is_io_error() {
        let mut buf = Vec::new();
        write_entry(&mut buf, &sample_entry()).unwrap();
        buf.truncate(buf.len() - 4);
        let err = read_entry(&mut Cursor::new(&buf)).unwrap_err();
        assert!(matches!(err, GllmError::Io(_)), "got {err:?}");
    }

    #[test]
    fn unit_file_write_read_roundtrip_with_aligned_data() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("layer_000.gllm");
        let a = vec![1u8; 100];
        let b = vec![2u8; 64];
        let written = write_unit_file(
            &path,
            &[
                ("input_norm.weight", &[64], DType::F32, &a),
                ("attn_q.weight", &[8, 8], DType::Q8_0, &b),
            ],
        )
        .unwrap();

        let layer = LayerFile::read(&path).unwrap();
        assert_eq!(layer.header.tensor_count, 2);
        assert_eq!(layer.tensor_index, written);
        assert_eq!(layer.data_offset % TENSOR_ALIGNMENT as u64, 0);
        // Second tensor starts at the next 64-byte boundary after 100.
        assert_eq!(layer.tensor_index[1].offset, 128);

        // Absolute ranges must point at the exact bytes written.
        let bytes = std::fs::read(&path).unwrap();
        let (off, size) = layer.absolute_range("attn_q.weight").unwrap();
        assert_eq!(&bytes[off as usize..(off + size) as usize], b.as_slice());
        let (off, size) = layer.absolute_range("input_norm.weight").unwrap();
        assert_eq!(&bytes[off as usize..(off + size) as usize], a.as_slice());
    }

    #[test]
    fn unit_file_read_empty_index_ok() {
        // Pre-ARTX04 fixture files carry tensor_count = 0.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("shared.gllm");
        make_test_gllm_file(&path);

        let layer = LayerFile::read(&path).unwrap();
        assert!(layer.tensor_index.is_empty());
        assert!(layer.absolute_range("anything").is_none());
    }

    #[test]
    fn unit_file_read_rejects_tensor_beyond_eof() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("layer_000.gllm");
        let data = vec![7u8; 64];
        write_unit_file(&path, &[("t", &[16], DType::F32, &data)]).unwrap();
        // Truncate into the data region.
        let bytes = std::fs::read(&path).unwrap();
        std::fs::write(&path, &bytes[..bytes.len() - 32]).unwrap();

        let err = LayerFile::read(&path).unwrap_err();
        assert!(matches!(err, GllmError::IntegrityError(_)), "got {err:?}");
    }

    fn layer_manifest_with(tensors: Vec<TensorEntry>) -> LayerManifest {
        LayerManifest {
            index: 0,
            file: "layer_000.gllm".into(),
            checksum: format!("sha256:{}", "0".repeat(64)),
            layer_type: crate::manifest::ExtensionUri(
                "gllm:transformer/standard@v1".into(),
            ),
            tensors,
            device: None,
        }
    }

    #[test]
    fn cross_check_passes_when_binary_matches_manifest() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("layer_000.gllm");
        let data = vec![0u8; 256];
        let written = write_unit_file(&path, &[("attn_q.weight", &[8, 8], DType::F32, &data)])
            .unwrap();
        let layer = LayerFile::read(&path).unwrap();

        let manifest = layer_manifest_with(written);
        assert!(cross_check_manifest(&layer, &manifest).is_empty());
    }

    #[test]
    fn cross_check_reports_shape_and_missing_mismatches() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("layer_000.gllm");
        let data = vec![0u8; 256];
        let mut written = write_unit_file(
            &path,
            &[("attn_q.weight", &[8, 8], DType::F32, &data)],
        )
        .unwrap();
        let layer = LayerFile::read(&path).unwrap();

        // Manifest claims a different shape AND an extra tensor.
        written[0].shape = vec![16, 4];
        written.push(TensorEntry {
            name: "ffn_up.weight".into(),
            shape: vec![4],
            dtype: DType::F32,
            offset: 0,
            size: 16,
        });
        let manifest = layer_manifest_with(written);

        let mismatches = cross_check_manifest(&layer, &manifest);
        assert_eq!(mismatches.len(), 3, "{mismatches:?}"); // count + shape + missing
    }
}
