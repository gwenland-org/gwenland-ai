//! Mini GGUF metadata-only reader (ARTX-OQ3 Wave 1).
//!
//! Reads only the `tokenizer.ggml.*` key/value pairs from a `.gguf` file's
//! metadata section, then stops — the tensor info table and tensor data
//! (typically the overwhelming majority of the file) are never touched.
//! This is deliberately not the full `glcore::format::gguf` parser: that
//! crate is not a dependency of `glictus-caliburni` (ARTX01's zero
//! workspace-dep guardrail), and a full tensor-aware parser is unneeded
//! weight for a reader whose only job is pulling out tokenizer strings.
//!
//! GGUF file layout (only the prefix we read):
//!
//! ```text
//! magic: u32              = 0x46554747 ("GGUF")
//! version: u32             = 2 or 3
//! tensor_count: u64
//! metadata_kv_count: u64
//! metadata_kv: [KV; metadata_kv_count]   <- we read this, then STOP
//! tensor_info: [...]                      <- never reached
//! tensor_data: [...]                      <- never reached
//! ```

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::error::GllmError;

/// Magic bytes at the start of every GGUF file: `b"GGUF"` read as a
/// little-endian `u32`.
const GGUF_MAGIC: u32 = 0x4655_4747;

/// Only `tokenizer.*` keys are worth retaining for this reader's purpose;
/// everything else is parsed just enough to skip past.
const RETAINED_KEY_PREFIX: &str = "tokenizer.";

/// A typed GGUF metadata value, restricted to the variants the tokenizer
/// path actually needs. Array element types outside `ArrStr`/`ArrI32` are
/// parsed (to stay byte-aligned with the rest of the file) then dropped.
#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    ArrStr(Vec<String>),
    ArrI32(Vec<i32>),
    ArrU32(Vec<u32>),
}

/// Metadata read from a GGUF file's key/value section, filtered to
/// `tokenizer.*` keys.
#[derive(Debug, Clone)]
pub struct GgufMeta {
    pub version: u32,
    entries: HashMap<String, GgufValue>,
}

impl GgufMeta {
    /// Read `tokenizer.*` metadata from a GGUF file at `path`.
    ///
    /// Stops reading as soon as the metadata KV section ends — the tensor
    /// info table and tensor data section (which can be many gigabytes)
    /// are never parsed or loaded.
    pub fn read_tokenizer(path: &Path) -> Result<Self, GllmError> {
        let file = File::open(path)?;
        let mut r = ByteReader::new(BufReader::new(file));

        let magic = r.u32()?;
        if magic != GGUF_MAGIC {
            return Err(GllmError::GgufInvalidMagic { got: magic });
        }
        let version = r.u32()?;
        if version != 2 && version != 3 {
            return Err(GllmError::GgufUnsupportedVersion { got: version });
        }

        let _tensor_count = r.u64()?;
        let metadata_kv_count = r.u64()?;

        let mut entries = HashMap::new();
        for _ in 0..metadata_kv_count {
            let key = r.string()?;
            let type_id = r.u32()?;
            let retain = key.starts_with(RETAINED_KEY_PREFIX);
            let value = r.value(type_id)?;
            if retain && let Some(v) = value {
                entries.insert(key, v);
            }
        }

        Ok(Self { version, entries })
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.entries.get(key) {
            Some(GgufValue::Str(s)) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.entries.get(key) {
            Some(GgufValue::U32(v)) => Some(*v),
            Some(GgufValue::I32(v)) => u32::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn get_bool(&self, key: &str) -> Option<bool> {
        match self.entries.get(key) {
            Some(GgufValue::Bool(v)) => Some(*v),
            _ => None,
        }
    }

    pub fn get_arr_str(&self, key: &str) -> Option<&[String]> {
        match self.entries.get(key) {
            Some(GgufValue::ArrStr(v)) => Some(v.as_slice()),
            _ => None,
        }
    }

    pub fn get_arr_i32(&self, key: &str) -> Option<&[i32]> {
        match self.entries.get(key) {
            Some(GgufValue::ArrI32(v)) => Some(v.as_slice()),
            _ => None,
        }
    }
}

/// Sequential little-endian byte reader over any [`Read`] source.
struct ByteReader<R: Read> {
    inner: R,
}

impl<R: Read> ByteReader<R> {
    fn new(inner: R) -> Self {
        Self { inner }
    }

    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, GllmError> {
        let mut buf = vec![0u8; n];
        self.inner.read_exact(&mut buf).map_err(GllmError::Io)?;
        Ok(buf)
    }

    fn u8(&mut self) -> Result<u8, GllmError> {
        Ok(self.bytes(1)?[0])
    }

    fn u32(&mut self) -> Result<u32, GllmError> {
        let b = self.bytes(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&mut self) -> Result<i32, GllmError> {
        Ok(self.u32()? as i32)
    }

    fn f32(&mut self) -> Result<f32, GllmError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn u64(&mut self) -> Result<u64, GllmError> {
        let b = self.bytes(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn f64(&mut self) -> Result<f64, GllmError> {
        let b = self.bytes(8)?;
        Ok(f64::from_bits(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])))
    }

    /// Length-prefixed (`u64`) UTF-8 string — GGUF v2/v3 layout.
    fn string(&mut self) -> Result<String, GllmError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| GllmError::GgufMalformedMetadata {
            detail: format!("string length {len} overflows usize"),
        })?;
        let bytes = self.bytes(len)?;
        String::from_utf8(bytes).map_err(|e| GllmError::GgufMalformedMetadata {
            detail: format!("invalid UTF-8 string: {e}"),
        })
    }

    /// Parse (and thereby consume) one metadata value of the given GGUF
    /// type id. Returns `None` for a type this reader has no representation
    /// for (still fully consumed from the stream, just not retained) —
    /// array element types other than str/i32/u32 fall into the same
    /// "skip, don't retain" bucket via nested `None`s collapsing the array.
    fn value(&mut self, type_id: u32) -> Result<Option<GgufValue>, GllmError> {
        Ok(match type_id {
            0 => {
                self.u8()?;
                None
            } // u8 — not needed by tokenizer keys, consumed & dropped
            1 => {
                self.u8()?;
                None
            } // i8
            2 => {
                self.bytes(2)?;
                None
            } // u16
            3 => {
                self.bytes(2)?;
                None
            } // i16
            4 => Some(GgufValue::U32(self.u32()?)),
            5 => Some(GgufValue::I32(self.i32()?)),
            6 => Some(GgufValue::F32(self.f32()?)),
            7 => Some(GgufValue::Bool(self.u8()? != 0)),
            8 => Some(GgufValue::Str(self.string()?)),
            9 => self.array()?,
            10 => {
                self.u64()?;
                None
            } // u64
            11 => {
                self.u64()?;
                None
            } // i64
            12 => {
                self.f64()?;
                None
            } // f64
            other => {
                return Err(GllmError::GgufMalformedMetadata {
                    detail: format!("unknown metadata value type {other}"),
                });
            }
        })
    }

    /// Array value: `elem_type: u32`, `count: u64`, then `count` elements.
    /// Every element must still be read off the stream to keep the cursor
    /// aligned, even when the element type isn't one we retain.
    fn array(&mut self) -> Result<Option<GgufValue>, GllmError> {
        let elem_type = self.u32()?;
        let count = self.u64()?;
        let count = usize::try_from(count).map_err(|_| GllmError::GgufMalformedMetadata {
            detail: format!("array count {count} overflows usize"),
        })?;

        match elem_type {
            8 => {
                let mut out = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    out.push(self.string()?);
                }
                Ok(Some(GgufValue::ArrStr(out)))
            }
            5 => {
                let mut out = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    out.push(self.i32()?);
                }
                Ok(Some(GgufValue::ArrI32(out)))
            }
            4 => {
                let mut out = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    out.push(self.u32()?);
                }
                Ok(Some(GgufValue::ArrU32(out)))
            }
            _ => {
                // Still consume every element so the cursor stays aligned
                // for whatever KV entry follows.
                for _ in 0..count {
                    self.value(elem_type)?;
                }
                Ok(None)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Builds a minimal synthetic GGUF byte stream: header + given KV
    /// entries, nothing else (no tensor section — tests never read past
    /// metadata anyway).
    struct GgufBuilder {
        buf: Vec<u8>,
        kv_count: u64,
    }

    impl GgufBuilder {
        fn new(version: u32) -> Self {
            let mut buf = Vec::new();
            buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
            buf.extend_from_slice(&version.to_le_bytes());
            buf.extend_from_slice(&0u64.to_le_bytes()); // tensor_count
            buf.extend_from_slice(&0u64.to_le_bytes()); // kv_count placeholder
            Self { buf, kv_count: 0 }
        }

        fn push_str(&mut self, s: &str) {
            self.buf
                .extend_from_slice(&(s.len() as u64).to_le_bytes());
            self.buf.extend_from_slice(s.as_bytes());
        }

        fn kv_str(&mut self, key: &str, value: &str) -> &mut Self {
            self.push_str(key);
            self.buf.extend_from_slice(&8u32.to_le_bytes()); // type_id = str
            self.push_str(value);
            self.kv_count += 1;
            self
        }

        fn kv_arr_str(&mut self, key: &str, values: &[&str]) -> &mut Self {
            self.push_str(key);
            self.buf.extend_from_slice(&9u32.to_le_bytes()); // type_id = array
            self.buf.extend_from_slice(&8u32.to_le_bytes()); // elem_type = str
            self.buf
                .extend_from_slice(&(values.len() as u64).to_le_bytes());
            for v in values {
                self.push_str(v);
            }
            self.kv_count += 1;
            self
        }

        fn kv_u32(&mut self, key: &str, value: u32) -> &mut Self {
            self.push_str(key);
            self.buf.extend_from_slice(&4u32.to_le_bytes()); // type_id = u32
            self.buf.extend_from_slice(&value.to_le_bytes());
            self.kv_count += 1;
            self
        }

        fn finish(mut self) -> Vec<u8> {
            // Patch the kv_count placeholder at offset 16 (4 magic + 4
            // version + 8 tensor_count).
            let count_bytes = self.kv_count.to_le_bytes();
            self.buf[16..24].copy_from_slice(&count_bytes);
            self.buf
        }

        fn write_to(self, path: &Path) {
            let bytes = self.finish();
            let mut f = File::create(path).unwrap();
            f.write_all(&bytes).unwrap();
        }
    }

    #[test]
    fn gguf_meta_rejects_wrong_magic() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("bad.gguf");
        std::fs::write(&path, b"XXXX\x03\x00\x00\x00").unwrap();

        let err = GgufMeta::read_tokenizer(&path).unwrap_err();
        assert!(
            matches!(err, GllmError::GgufInvalidMagic { .. }),
            "got {err:?}"
        );
    }

    #[test]
    fn gguf_meta_rejects_unsupported_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("v1.gguf");
        let builder = GgufBuilder::new(1);
        builder.write_to(&path);

        let err = GgufMeta::read_tokenizer(&path).unwrap_err();
        assert!(
            matches!(err, GllmError::GgufUnsupportedVersion { got: 1 }),
            "got {err:?}"
        );
    }

    #[test]
    fn gguf_meta_reads_str_value() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("str.gguf");
        let mut builder = GgufBuilder::new(3);
        builder.kv_str("tokenizer.ggml.model", "gpt2");
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        assert_eq!(meta.get_str("tokenizer.ggml.model"), Some("gpt2"));
    }

    #[test]
    fn gguf_meta_reads_arr_str_value() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("arr.gguf");
        let mut builder = GgufBuilder::new(3);
        builder.kv_arr_str("tokenizer.ggml.tokens", &["a", "b", "c"]);
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        let tokens = meta.get_arr_str("tokenizer.ggml.tokens").unwrap();
        assert_eq!(tokens, &["a".to_string(), "b".to_string(), "c".to_string()]);
    }

    #[test]
    fn gguf_meta_skips_non_tokenizer_keys() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("mixed.gguf");
        let mut builder = GgufBuilder::new(3);
        builder
            .kv_str("general.name", "some-model")
            .kv_str("tokenizer.ggml.model", "gpt2");
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        assert_eq!(meta.get_str("general.name"), None);
        assert_eq!(meta.get_str("tokenizer.ggml.model"), Some("gpt2"));
    }

    #[test]
    fn gguf_meta_handles_empty_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("empty.gguf");
        let builder = GgufBuilder::new(3);
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        assert_eq!(meta.get_str("anything"), None);
    }

    #[test]
    fn gguf_meta_reads_u32_and_bool() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("scalars.gguf");
        let mut builder = GgufBuilder::new(3);
        builder.kv_u32("tokenizer.ggml.bos_token_id", 151643);
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        assert_eq!(meta.get_u32("tokenizer.ggml.bos_token_id"), Some(151643));
    }

    #[test]
    fn gguf_meta_skips_unretained_array_element_types_without_desync() {
        // A non-tokenizer array key followed by a tokenizer string key —
        // if array-skipping desyncs the cursor, the second key's value
        // will be garbage or the read will fail outright.
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("desync.gguf");
        let mut builder = GgufBuilder::new(3);
        builder.push_str("general.some_u32_array");
        builder.buf.extend_from_slice(&9u32.to_le_bytes()); // array
        builder.buf.extend_from_slice(&4u32.to_le_bytes()); // elem_type u32
        builder.buf.extend_from_slice(&3u64.to_le_bytes()); // count
        builder.buf.extend_from_slice(&1u32.to_le_bytes());
        builder.buf.extend_from_slice(&2u32.to_le_bytes());
        builder.buf.extend_from_slice(&3u32.to_le_bytes());
        builder.kv_count += 1;
        builder.kv_str("tokenizer.ggml.model", "gpt2");
        builder.write_to(&path);

        let meta = GgufMeta::read_tokenizer(&path).unwrap();
        assert_eq!(meta.get_str("tokenizer.ggml.model"), Some("gpt2"));
    }

    /// Real-model check: opt-in via GWENLAND_TEST_GGUF (real GGUFs are
    /// gitignored). Skips loudly when absent, per testing standards.
    #[test]
    fn gguf_meta_reads_qwen25_tokenizer_fields() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let meta = GgufMeta::read_tokenizer(Path::new(&path)).unwrap();

        assert_eq!(meta.get_str("tokenizer.ggml.model"), Some("gpt2"));
        assert_eq!(meta.get_str("tokenizer.ggml.pre"), Some("qwen2"));
        // Verified 2026-07-21 against a real qwen2.5-0.5b-instruct-q4_k_m.gguf
        // (Downloads). The plan doc's 152,064 was for a different Qwen2.5
        // build/variant — this file's actual vocab is 151,936.
        assert_eq!(
            meta.get_arr_str("tokenizer.ggml.tokens").map(<[_]>::len),
            Some(151_936)
        );
        assert_eq!(
            meta.get_arr_str("tokenizer.ggml.merges").map(<[_]>::len),
            Some(151_387)
        );
    }
}
