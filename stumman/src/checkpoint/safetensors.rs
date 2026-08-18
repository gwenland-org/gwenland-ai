//! Stummañ Pik: the safetensors **writer**.
//!
//! `glcore/src/format/safetensors.rs` has a from-scratch, mmap-backed reader.
//! There is no writer anywhere in the repo, and this is it.
//!
//! # Format, as the spec states it
//!
//! ```text
//! [0..8)          u64 little-endian: header length N
//! [8..8+N)        N bytes of UTF-8 JSON
//! [8+N..)         raw tensor bytes, no holes
//! ```
//!
//! The header maps a tensor name to `{"dtype", "shape", "data_offsets"}`, where
//! `data_offsets` is `[begin, end)` **relative to the start of the data
//! section**, not to the start of the file. `__metadata__` is reserved for a
//! flat string-to-string map and is not a tensor.
//!
//! # Why this is written in-tree rather than pulled from a crate
//!
//! Because `glcore`'s reader can then serve as an independent round-trip
//! oracle: write with this, read back with `SafetensorsFile::open`, compare.
//! A third-party writer paired with that same third-party reader would only
//! prove the library is self-consistent. `safetensors_writer_round_trips_*`
//! below is the check nothing else in the repo can perform.

use crate::error::{GlTrainError, Result};
use crate::optim::VLNamedTensor;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use super::json::Json;

/// Bytes per element. Everything this crate writes is `F32`.
const F32_BYTES: usize = 4;

/// The dtype string written for every tensor.
pub const DTYPE_F32: &str = "F32";

/// How a tensor's values are obtained on load.
///
/// `EN` because a closed set of variants is this type's whole job.
///
/// # Why `Generated` exists before anything produces it
///
/// VeRA (M3) shares one frozen random `A`/`B` pair across every adapted layer,
/// and its paper is explicit that those matrices "do not need to be stored in
/// memory" and "can be regenerated from a random number generator (RNG) seed".
/// A schema that only knows how to say "here are the bytes" cannot express
/// that, and widening it later is a breaking format change. The variant costs
/// nothing today and is the reason `stumman`'s RNG is documented as
/// bit-stable.
///
/// Only [`ENTensorEntry::Stored`] is produced until VeRA lands.
#[derive(Debug, Clone, PartialEq)]
pub enum ENTensorEntry {
    /// Bytes are in the file, at `[begin, end)` of the data section.
    Stored {
        /// `[begin, end)`, relative to the start of the data section.
        data_offsets: [usize; 2],
    },
    /// Values are regenerated from a seed rather than stored.
    Generated {
        /// Seed for [`crate::rng::Xorshift64Star`].
        seed: u64,
        /// Distribution name, e.g. `"normal"`.
        distribution: String,
    },
}

/// Write `tensors` and a free-form string metadata map to a safetensors file.
///
/// Rejects duplicate names: two entries under one key would make the file's
/// meaning depend on which the reader's map kept.
pub fn write(
    path: &Path,
    tensors: &[VLNamedTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<()> {
    let bytes = to_bytes(tensors, metadata)?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    std::fs::write(path, bytes)?;
    Ok(())
}

/// Serialize to an in-memory buffer. Split out so tests can exercise the
/// encoding without touching the filesystem.
pub fn to_bytes(
    tensors: &[VLNamedTensor],
    metadata: &BTreeMap<String, String>,
) -> Result<Vec<u8>> {
    let mut header = BTreeMap::new();
    let mut cursor = 0usize;
    let mut seen = BTreeSet::new();

    for t in tensors {
        if t.name == "__metadata__" {
            return Err(GlTrainError::Checkpoint(
                "'__metadata__' is reserved by the safetensors format and cannot name a tensor"
                    .into(),
            ));
        }
        if !seen.insert(t.name.clone()) {
            return Err(GlTrainError::Checkpoint(format!(
                "duplicate tensor name '{}' in one file",
                t.name
            )));
        }
        // Shape and data must agree before anything is written. A file whose
        // header promises a shape its bytes cannot fill is unreadable, and the
        // reader would only notice as a length surprise much later.
        let expected: usize = t.shape.iter().product();
        if t.shape.is_empty() || expected != t.data.len() {
            return Err(GlTrainError::Checkpoint(format!(
                "tensor '{}' has shape {:?} ({} elements) but {} values",
                t.name,
                t.shape,
                if t.shape.is_empty() { 0 } else { expected },
                t.data.len()
            )));
        }

        let len = t.data.len() * F32_BYTES;
        header.insert(
            t.name.clone(),
            Json::obj([
                ("dtype", Json::s(DTYPE_F32)),
                ("shape", Json::usizes(t.shape.iter().copied())),
                ("data_offsets", Json::usizes([cursor, cursor + len])),
            ]),
        );
        cursor += len;
    }

    if !metadata.is_empty() {
        header.insert(
            "__metadata__".to_string(),
            Json::Obj(
                metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), Json::s(v.clone())))
                    .collect(),
            ),
        );
    }

    let mut header_json = Json::Obj(header).to_compact();
    // The spec does not require it, but every reference writer pads the header
    // with spaces so the data section starts on an 8-byte boundary. Costs at
    // most 7 bytes and keeps the file loadable by readers that assume it.
    // Trailing whitespace after a JSON document is insignificant.
    while !(8 + header_json.len()).is_multiple_of(8) {
        header_json.push(' ');
    }

    let mut out = Vec::with_capacity(8 + header_json.len() + cursor);
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(header_json.as_bytes());
    for t in tensors {
        for v in &t.data {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    Ok(out)
}

/// Read a safetensors file back into [`VLNamedTensor`]s.
///
/// Delegates the parse to `glcore`'s reader rather than re-implementing it:
/// this crate owns the writer, and having a second parser here would let the
/// two drift apart with no test able to notice.
pub fn read(path: &Path) -> Result<Vec<VLNamedTensor>> {
    let path_str = path.to_str().ok_or_else(|| {
        GlTrainError::Checkpoint(format!("path is not valid UTF-8: {}", path.display()))
    })?;
    let file = glcore::format::safetensors::SafetensorsFile::open(path_str)?;

    // The reader hands back a HashMap, so sort by name for a deterministic
    // order. Callers look tensors up by name, but a stable order makes test
    // failures readable.
    let mut names: Vec<String> = file.tensors.keys().cloned().collect();
    names.sort();

    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let meta = &file.tensors[&name];
        let data = file.to_f32(&name)?;
        let expected: usize = meta.shape.iter().product();
        if data.len() != expected {
            return Err(GlTrainError::Checkpoint(format!(
                "tensor '{name}' declares shape {:?} ({expected} elements) but holds {}",
                meta.shape,
                data.len()
            )));
        }
        out.push(VLNamedTensor::new(name, data, meta.shape.clone()));
    }
    Ok(out)
}

/// Read the `__metadata__` map, if present.
pub fn read_metadata(path: &Path) -> Result<BTreeMap<String, String>> {
    let bytes = std::fs::read(path)?;
    if bytes.len() < 8 {
        return Err(GlTrainError::Checkpoint(
            "safetensors: file too small for a header".into(),
        ));
    }
    let n = u64::from_le_bytes(bytes[0..8].try_into().expect("8 bytes")) as usize;
    let end = 8usize.checked_add(n).filter(|e| *e <= bytes.len()).ok_or_else(|| {
        GlTrainError::Checkpoint("safetensors: header extends past end of file".into())
    })?;
    let text = std::str::from_utf8(&bytes[8..end])
        .map_err(|e| GlTrainError::Checkpoint(format!("safetensors: header is not UTF-8: {e}")))?;
    let doc = super::json::parse(text)
        .map_err(|e| GlTrainError::Checkpoint(format!("safetensors: bad header JSON: {e}")))?;

    let mut out = BTreeMap::new();
    if let Some(m) = doc.get("__metadata__").and_then(Json::as_obj) {
        for (k, v) in m {
            if let Some(s) = v.as_str() {
                out.insert(k.clone(), s.to_string());
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values round-trip through f32 bytes with no arithmetic, so any
    /// difference at all is a real encoding bug.
    const TOL_EXACT: f32 = 0.0;

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "stumman_st_{tag}_{}_{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// The primary correctness check for the writer, and one nothing else in
    /// the repo can perform: `glcore`'s reader is an independent parser, so
    /// agreement between the two is evidence rather than self-consistency.
    #[test]
    fn safetensors_writer_round_trips_a_single_tensor_through_the_glcore_reader() {
        let dir = tmp_dir("single");
        let path = dir.join("one.safetensors");
        let t = VLNamedTensor::new("w", vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        write(&path, std::slice::from_ref(&t), &BTreeMap::new()).unwrap();

        let file =
            glcore::format::safetensors::SafetensorsFile::open(path.to_str().unwrap()).unwrap();
        assert_eq!(file.tensor_names(), vec!["w"]);
        assert_eq!(file.tensors["w"].shape, vec![2, 2]);
        assert_eq!(file.tensors["w"].dtype, "F32");
        let back = file.to_f32("w").unwrap();
        for (i, (g, w)) in back.iter().zip(&t.data).enumerate() {
            assert!((g - w).abs() <= TOL_EXACT, "element {i}: {g} != {w}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Offsets are cumulative and relative to the data section. Getting that
    /// wrong shows up only when a file holds more than one tensor.
    #[test]
    fn safetensors_writer_round_trips_multiple_tensors() {
        let dir = tmp_dir("multi");
        let path = dir.join("many.safetensors");
        let tensors = vec![
            VLNamedTensor::new("lora_a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]),
            VLNamedTensor::new("lora_b", vec![7.0, 8.0], vec![2, 1]),
            VLNamedTensor::new("bias", vec![9.0], vec![1]),
        ];
        write(&path, &tensors, &BTreeMap::new()).unwrap();

        let back = read(&path).unwrap();
        assert_eq!(back.len(), 3);
        for want in &tensors {
            let got = back
                .iter()
                .find(|t| t.name == want.name)
                .unwrap_or_else(|| panic!("'{}' missing after round trip", want.name));
            assert_eq!(got.shape, want.shape, "shape for '{}'", want.name);
            assert_eq!(got.data, want.data, "data for '{}'", want.name);
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn safetensors_writer_includes_the_metadata_map() {
        let dir = tmp_dir("meta");
        let path = dir.join("m.safetensors");
        let mut meta = BTreeMap::new();
        meta.insert("adapter_type".to_string(), "lora".to_string());
        meta.insert("step".to_string(), "500".to_string());
        write(
            &path,
            &[VLNamedTensor::new("w", vec![1.0], vec![1])],
            &meta,
        )
        .unwrap();

        assert_eq!(read_metadata(&path).unwrap(), meta);
        // And the metadata entry must not be mistaken for a tensor.
        let back = read(&path).unwrap();
        assert_eq!(back.len(), 1);
        assert_eq!(back[0].name, "w");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Every reference writer aligns the data section to 8 bytes. Costs at
    /// most 7 bytes and keeps the file loadable by readers that assume it.
    #[test]
    fn the_data_section_starts_on_an_eight_byte_boundary() {
        for n in 1..12usize {
            let name = "x".repeat(n);
            let bytes = to_bytes(
                &[VLNamedTensor::new(name.clone(), vec![1.0], vec![1])],
                &BTreeMap::new(),
            )
            .unwrap();
            let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
            assert_eq!(
                (8 + header_len) % 8,
                0,
                "name length {n} left the data section unaligned"
            );
        }
    }

    /// A file whose header promises a shape its bytes cannot fill is
    /// unreadable, and the reader would only see it much later as a length
    /// surprise.
    #[test]
    fn writing_a_tensor_whose_shape_disagrees_with_its_data_is_refused() {
        let bad = VLNamedTensor::new("w", vec![1.0, 2.0, 3.0], vec![2, 2]);
        assert!(to_bytes(&[bad], &BTreeMap::new()).is_err());
        let empty_shape = VLNamedTensor::new("w", vec![1.0], vec![]);
        assert!(to_bytes(&[empty_shape], &BTreeMap::new()).is_err());
    }

    /// Two entries under one key would make the file's meaning depend on which
    /// one the reader's map happened to keep.
    #[test]
    fn writing_two_tensors_with_the_same_name_is_refused() {
        let ts = vec![
            VLNamedTensor::new("w", vec![1.0], vec![1]),
            VLNamedTensor::new("w", vec![2.0], vec![1]),
        ];
        assert!(to_bytes(&ts, &BTreeMap::new()).is_err());
    }

    /// `__metadata__` is the format's own key. A tensor by that name would be
    /// silently dropped by any conforming reader.
    #[test]
    fn a_tensor_named_metadata_is_refused() {
        let ts = vec![VLNamedTensor::new("__metadata__", vec![1.0], vec![1])];
        assert!(to_bytes(&ts, &BTreeMap::new()).is_err());
    }

    /// Only `Stored` is produced on M2, but the schema can already express
    /// VeRA's "regenerate from this seed" without a breaking change.
    #[test]
    fn the_tensor_entry_schema_can_express_generated_as_well_as_stored() {
        let stored = ENTensorEntry::Stored {
            data_offsets: [0, 16],
        };
        let generated = ENTensorEntry::Generated {
            seed: 42,
            distribution: "normal".to_string(),
        };
        assert_ne!(stored, generated);
        assert!(matches!(stored, ENTensorEntry::Stored { .. }));
        assert!(matches!(
            generated,
            ENTensorEntry::Generated { seed: 42, .. }
        ));
    }

    #[test]
    fn an_empty_tensor_list_still_produces_a_readable_file() {
        let dir = tmp_dir("empty");
        let path = dir.join("none.safetensors");
        write(&path, &[], &BTreeMap::new()).unwrap();
        assert!(read(&path).unwrap().is_empty());
        std::fs::remove_dir_all(&dir).ok();
    }
}
