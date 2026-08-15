// engine/inference/candle_ggqr/gguf.rs — GGUF validation, architecture extraction,
// and ModelConfig construction for the candle-backend.
//
// Reads GGUF KV metadata without touching the existing dequant parser, which
// only skips KV entries. The functions here open the same file and decode only
// the header + KV section; tensor data is left to the existing `convert::gguf_parser`.
//
// Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 22.1, 22.2, 22.3, 22.4

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use crate::convert::gguf_parser::{GgufFile, TensorInfo, parse, read_gguf_string, read_u32_le, read_u64_le};
use crate::error::GwenError;
use super::ModelConfig;

// ── Limits (Requirements: 22.1 – 22.4) ───────────────────────────────────────

const MAX_TENSOR_COUNT: usize = 10_000;
const MAX_DIM_SIZE: u64 = 100_000;

/// Supported architectures — lower-case, as stored in `general.architecture`.
const SUPPORTED_ARCHITECTURES: &[&str] = &["llama", "qwen2", "phi3"];

// ── Public API ────────────────────────────────────────────────────────────────

/// Validate a GGUF file at `path` and return the parsed `GgufFile`.
///
/// Checks (Requirements 1.1, 1.6, 22.1 – 22.4):
/// - Magic bytes are `GGUF`.
/// - Tensor count ≤ 10 000.
/// - No tensor dimension exceeds 100 000.
/// - Each tensor's buffer size matches the byte count implied by its dtype + shape.
///
/// # Errors
///
/// Returns `GwenError::ModelLoad` with a descriptive message on any violation.
pub fn validate_gguf(path: &Path) -> Result<GgufFile, GwenError> {
    let gguf = parse(path)
        .map_err(|e| GwenError::ModelLoad(e))?;

    // Requirement 22.1 — tensor count
    if gguf.tensors.len() > MAX_TENSOR_COUNT {
        return Err(GwenError::ModelLoad(format!(
            "GGUF file '{}' contains {} tensors, which exceeds the limit of {}",
            path.display(),
            gguf.tensors.len(),
            MAX_TENSOR_COUNT,
        )));
    }

    for tensor in &gguf.tensors {
        // Requirement 22.2 — dimension sizes
        for &dim in &tensor.shape {
            if dim > MAX_DIM_SIZE {
                return Err(GwenError::ModelLoad(format!(
                    "tensor '{}' has a dimension of {} which exceeds the limit of {}",
                    tensor.name, dim, MAX_DIM_SIZE,
                )));
            }
        }

        // Requirement 22.4 — buffer size matches expected byte count
        let expected = expected_byte_count(tensor);
        if tensor.raw_data.len() != expected {
            return Err(GwenError::ModelLoad(format!(
                "tensor '{}' buffer size mismatch: expected {} bytes, got {}",
                tensor.name,
                expected,
                tensor.raw_data.len(),
            )));
        }
    }

    Ok(gguf)
}

/// Extract and validate `general.architecture` from a GGUF file's KV metadata.
///
/// Reads the KV section of `path` directly (without re-parsing tensor data) and
/// looks for the `"general.architecture"` key. Returns the architecture string if
/// it is one of `["llama", "qwen2", "phi3"]`.
///
/// # Errors
///
/// - `GwenError::ModelLoad` if the file cannot be opened or parsed.
/// - `GwenError::ModelLoad` if `general.architecture` is missing.
/// - `GwenError::ArchitectureNotSupported` for any other architecture value.
///
/// Requirements: 1.2, 1.3, 1.4
pub fn extract_architecture(path: &Path) -> Result<String, GwenError> {
    let kv = read_kv_metadata(path)?;

    let arch = kv
        .get("general.architecture")
        .and_then(|v| v.as_string())
        .ok_or_else(|| GwenError::ModelLoad(
            format!("'{}': missing 'general.architecture' key in GGUF metadata", path.display())
        ))?;

    if !SUPPORTED_ARCHITECTURES.contains(&arch.as_str()) {
        return Err(GwenError::ArchitectureNotSupported(format!(
            "'{}' (supported: {})",
            arch,
            SUPPORTED_ARCHITECTURES.join(", "),
        )));
    }

    Ok(arch)
}

/// Build a `ModelConfig` by reading the KV metadata section of `path`.
///
/// Uses `arch` (already extracted and validated by [`extract_architecture`]) to
/// form the correct GGUF key prefix (e.g. `"llama.block_count"`).
///
/// Missing required keys return `GwenError::ModelLoad`. Optional keys fall back
/// to sensible defaults (e.g. `rope_theta` defaults to 10 000.0).
///
/// Requirements: 1.2, 1.5
pub fn build_model_config(path: &Path, arch: &str) -> Result<ModelConfig, GwenError> {
    let kv = read_kv_metadata(path)?;

    let get_u32 = |key: &str| -> Result<u32, GwenError> {
        kv.get(key)
            .and_then(|v| v.as_u32())
            .ok_or_else(|| GwenError::ModelLoad(format!(
                "'{}': missing or non-integer GGUF metadata key '{}'",
                path.display(), key,
            )))
    };

    let get_f32_or = |key: &str, default: f32| -> f32 {
        kv.get(key)
            .and_then(|v| v.as_f32())
            .unwrap_or(default)
    };

    let n_layers   = get_u32(&format!("{arch}.block_count"))?;
    let hidden_size = get_u32(&format!("{arch}.embedding_length"))?;
    let n_heads    = get_u32(&format!("{arch}.attention.head_count"))?;
    let n_kv_heads = get_u32(&format!("{arch}.attention.head_count_kv"))?;
    let intermediate_size = get_u32(&format!("{arch}.feed_forward_length"))?;

    // vocab_size: try arch-prefixed key first, fall back to tokenizer count.
    let vocab_size = kv
        .get(&format!("{arch}.vocab_size"))
        .and_then(|v| v.as_u32())
        .or_else(|| kv.get("tokenizer.ggml.token_type_count").and_then(|v| v.as_u32()))
        .ok_or_else(|| GwenError::ModelLoad(format!(
            "'{}': cannot determine vocab_size (tried '{arch}.vocab_size' and 'tokenizer.ggml.token_type_count')",
            path.display(),
        )))?;

    let rms_norm_eps = get_f32_or(
        &format!("{arch}.attention.layer_norm_rms_epsilon"),
        1e-5,
    );
    let rope_theta = get_f32_or(&format!("{arch}.rope.freq_base"), 10_000.0);

    Ok(ModelConfig {
        architecture: arch.to_string(),
        n_layers,
        hidden_size,
        n_heads,
        n_kv_heads,
        intermediate_size,
        vocab_size,
        rms_norm_eps,
        rope_theta,
    })
}

// ── KV metadata reader ────────────────────────────────────────────────────────

/// A single GGUF KV value, decoded from the binary stream.
#[derive(Debug, Clone)]
pub enum KvValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(String),
    U64(u64),
    I64(i64),
    F64(f64),
    Array(Vec<KvValue>),
}

impl KvValue {
    fn as_string(&self) -> Option<String> {
        if let KvValue::Str(s) = self { Some(s.clone()) } else { None }
    }

    fn as_u32(&self) -> Option<u32> {
        match self {
            KvValue::U8(v)  => Some(*v as u32),
            KvValue::U16(v) => Some(*v as u32),
            KvValue::U32(v) => Some(*v),
            KvValue::I32(v) if *v >= 0 => Some(*v as u32),
            KvValue::U64(v) if *v <= u32::MAX as u64 => Some(*v as u32),
            _ => None,
        }
    }

    fn as_f32(&self) -> Option<f32> {
        match self {
            KvValue::F32(v) => Some(*v),
            KvValue::F64(v) => Some(*v as f32),
            _ => None,
        }
    }
}

/// Read the GGUF header + KV section from `path` and return every key-value pair.
///
/// This re-opens the file independently of the tensor-loading parser so we can
/// decode values rather than skip them.
fn read_kv_metadata(path: &Path) -> Result<HashMap<String, KvValue>, GwenError> {
    let file = File::open(path)
        .map_err(|e| GwenError::ModelLoad(format!("cannot open '{}': {}", path.display(), e)))?;
    let mut r = BufReader::new(file);

    // Magic
    let magic = read_u32_le(&mut r)
        .map_err(|e| GwenError::ModelLoad(e))?;
    if magic != 0x4647_4755 {
        return Err(GwenError::ModelLoad(format!(
            "'{}' is not a GGUF file (magic 0x{:08X})", path.display(), magic
        )));
    }

    // Version
    let _version = read_u32_le(&mut r)
        .map_err(|e| GwenError::ModelLoad(e))?;

    // Tensor count + KV count
    let _tensor_count = read_u64_le(&mut r)
        .map_err(|e| GwenError::ModelLoad(e))?;
    let kv_count = read_u64_le(&mut r)
        .map_err(|e| GwenError::ModelLoad(e))? as usize;

    let mut map = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)
            .map_err(|e| GwenError::ModelLoad(e))?;
        let vtype = read_u32_le(&mut r)
            .map_err(|e| GwenError::ModelLoad(e))?;
        let value = read_kv_value(&mut r, vtype)
            .map_err(|e| GwenError::ModelLoad(e))?;
        map.insert(key, value);
    }

    Ok(map)
}

fn read_kv_value<R: Read>(r: &mut R, vtype: u32) -> Result<KvValue, String> {
    match vtype {
        0  => Ok(KvValue::U8(read_u8(r)?)),
        1  => Ok(KvValue::I8(read_u8(r)? as i8)),
        2  => Ok(KvValue::U16(read_u16_le(r)?)),
        3  => Ok(KvValue::I16(read_i16_le(r)?)),
        4  => Ok(KvValue::U32(read_u32_le(r)?)),
        5  => Ok(KvValue::I32(read_i32_le(r)?)),
        6  => Ok(KvValue::F32(read_f32_le(r)?)),
        7  => Ok(KvValue::Bool(read_u8(r)? != 0)),
        8  => Ok(KvValue::Str(read_gguf_string(r)?)),
        9  => {
            let elem_type = read_u32_le(r)?;
            let count = read_u64_le(r)? as usize;
            let mut items = Vec::with_capacity(count.min(1024));
            for _ in 0..count {
                items.push(read_kv_value(r, elem_type)?);
            }
            Ok(KvValue::Array(items))
        }
        10 => Ok(KvValue::U64(read_u64_le(r)?)),
        11 => Ok(KvValue::I64(read_i64_le(r)?)),
        12 => Ok(KvValue::F64(read_f64_le(r)?)),
        other => Err(format!("unknown KV value type {} in GGUF metadata", other)),
    }
}

// ── Buffer size oracle ────────────────────────────────────────────────────────

/// Recompute the expected byte count for `tensor` from its shape and dtype,
/// mirroring the logic in `convert::gguf_parser::raw_size_bytes`.
///
/// Used by `validate_gguf` for Requirement 22.4.
fn expected_byte_count(tensor: &TensorInfo) -> usize {
    use crate::convert::gguf_parser::GgufDtype::*;

    let n: usize = tensor.shape.iter().map(|&d| d as usize).product();
    const QK: usize = 32;
    const SB: usize = 256;

    match tensor.dtype {
        F32  => n * 4,
        F16  => n * 2,
        Q8_0 => { let b = n.div_ceil(QK); b * (2 + QK) }
        Q4_0 => { let b = n.div_ceil(QK); b * (2 + QK / 2) }
        Q2_K => { let b = n.div_ceil(SB); b * 84  }
        Q3_K => { let b = n.div_ceil(SB); b * 110 }
        Q4_K => { let b = n.div_ceil(SB); b * 144 }
        Q5_K => { let b = n.div_ceil(SB); b * 176 }
        Q6_K => { let b = n.div_ceil(SB); b * 210 }
    }
}

// ── Local primitive readers ───────────────────────────────────────────────────
// We duplicate these small helpers rather than making the private functions in
// gguf_parser.rs public, keeping the parser's API surface minimal.

fn read_u8<R: Read>(r: &mut R) -> Result<u8, String> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading u8: {e}"))?;
    Ok(b[0])
}

fn read_u16_le<R: Read>(r: &mut R) -> Result<u16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading u16: {e}"))?;
    Ok(u16::from_le_bytes(b))
}

fn read_i16_le<R: Read>(r: &mut R) -> Result<i16, String> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading i16: {e}"))?;
    Ok(i16::from_le_bytes(b))
}

fn read_i32_le<R: Read>(r: &mut R) -> Result<i32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading i32: {e}"))?;
    Ok(i32::from_le_bytes(b))
}

fn read_f32_le<R: Read>(r: &mut R) -> Result<f32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading f32: {e}"))?;
    Ok(f32::from_le_bytes(b))
}

fn read_i64_le<R: Read>(r: &mut R) -> Result<i64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading i64: {e}"))?;
    Ok(i64::from_le_bytes(b))
}

fn read_f64_le<R: Read>(r: &mut R) -> Result<f64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| format!("I/O error reading f64: {e}"))?;
    Ok(f64::from_le_bytes(b))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Build a minimal valid GGUF byte buffer with KV metadata.
    //
    // Layout (all little-endian):
    //   magic:        u32  = 0x46474755
    //   version:      u32  = 3
    //   tensor_count: u64  = 0
    //   kv_count:     u64  = N
    //   [KV entries]
    fn make_gguf(kv: &[(&str, KvValue)]) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // header
        buf.extend_from_slice(&0x4647_4755u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&0u64.to_le_bytes());                      // tensor_count
        buf.extend_from_slice(&(kv.len() as u64).to_le_bytes());         // kv_count

        for (key, value) in kv {
            write_gguf_string(&mut buf, key);
            write_kv_value(&mut buf, value);
        }

        buf
    }

    fn write_gguf_string(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn write_kv_value(buf: &mut Vec<u8>, v: &KvValue) {
        match v {
            KvValue::U32(n) => {
                buf.extend_from_slice(&4u32.to_le_bytes());  // type tag
                buf.extend_from_slice(&n.to_le_bytes());
            }
            KvValue::F32(f) => {
                buf.extend_from_slice(&6u32.to_le_bytes());
                buf.extend_from_slice(&f.to_le_bytes());
            }
            KvValue::Str(s) => {
                buf.extend_from_slice(&8u32.to_le_bytes());
                write_gguf_string(buf, s);
            }
            _ => unimplemented!("test helper only covers U32/F32/Str"),
        }
    }

    fn write_to_tempfile(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        f
    }

    // ── extract_architecture ─────────────────────────────────────────────────

    #[test]
    fn extract_architecture_llama_ok() {
        let data = make_gguf(&[
            ("general.architecture", KvValue::Str("llama".to_string())),
        ]);
        let f = write_to_tempfile(&data);
        let arch = extract_architecture(f.path()).unwrap();
        assert_eq!(arch, "llama");
    }

    #[test]
    fn extract_architecture_qwen2_ok() {
        let data = make_gguf(&[
            ("general.architecture", KvValue::Str("qwen2".to_string())),
        ]);
        let f = write_to_tempfile(&data);
        assert_eq!(extract_architecture(f.path()).unwrap(), "qwen2");
    }

    #[test]
    fn extract_architecture_phi3_ok() {
        let data = make_gguf(&[
            ("general.architecture", KvValue::Str("phi3".to_string())),
        ]);
        let f = write_to_tempfile(&data);
        assert_eq!(extract_architecture(f.path()).unwrap(), "phi3");
    }

    #[test]
    fn extract_architecture_unsupported_returns_err() {
        let data = make_gguf(&[
            ("general.architecture", KvValue::Str("gpt2".to_string())),
        ]);
        let f = write_to_tempfile(&data);
        let err = extract_architecture(f.path()).unwrap_err();
        assert!(matches!(err, GwenError::ArchitectureNotSupported(_)));
    }

    #[test]
    fn extract_architecture_missing_key_returns_model_load_err() {
        let data = make_gguf(&[]);
        let f = write_to_tempfile(&data);
        let err = extract_architecture(f.path()).unwrap_err();
        assert!(matches!(err, GwenError::ModelLoad(_)));
    }

    // ── build_model_config ───────────────────────────────────────────────────

    #[test]
    fn build_model_config_llama_all_keys_present() {
        let data = make_gguf(&[
            ("llama.block_count",                         KvValue::U32(32)),
            ("llama.embedding_length",                    KvValue::U32(4096)),
            ("llama.attention.head_count",                KvValue::U32(32)),
            ("llama.attention.head_count_kv",             KvValue::U32(8)),
            ("llama.feed_forward_length",                 KvValue::U32(11008)),
            ("llama.vocab_size",                          KvValue::U32(32000)),
            ("llama.attention.layer_norm_rms_epsilon",    KvValue::F32(1e-5)),
            ("llama.rope.freq_base",                      KvValue::F32(10000.0)),
        ]);
        let f = write_to_tempfile(&data);
        let cfg = build_model_config(f.path(), "llama").unwrap();
        assert_eq!(cfg.n_layers, 32);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.n_heads, 32);
        assert_eq!(cfg.n_kv_heads, 8);
        assert_eq!(cfg.intermediate_size, 11008);
        assert_eq!(cfg.vocab_size, 32000);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-8);
        assert!((cfg.rope_theta - 10000.0).abs() < 1.0);
    }

    #[test]
    fn build_model_config_rope_theta_defaults_to_10000() {
        let data = make_gguf(&[
            ("llama.block_count",             KvValue::U32(1)),
            ("llama.embedding_length",        KvValue::U32(64)),
            ("llama.attention.head_count",    KvValue::U32(1)),
            ("llama.attention.head_count_kv", KvValue::U32(1)),
            ("llama.feed_forward_length",     KvValue::U32(128)),
            ("llama.vocab_size",              KvValue::U32(100)),
        ]);
        let f = write_to_tempfile(&data);
        let cfg = build_model_config(f.path(), "llama").unwrap();
        assert!((cfg.rope_theta - 10_000.0).abs() < 1.0);
    }

    #[test]
    fn build_model_config_vocab_size_falls_back_to_tokenizer_key() {
        let data = make_gguf(&[
            ("llama.block_count",                      KvValue::U32(1)),
            ("llama.embedding_length",                 KvValue::U32(64)),
            ("llama.attention.head_count",             KvValue::U32(1)),
            ("llama.attention.head_count_kv",          KvValue::U32(1)),
            ("llama.feed_forward_length",              KvValue::U32(128)),
            ("tokenizer.ggml.token_type_count",        KvValue::U32(9999)),
        ]);
        let f = write_to_tempfile(&data);
        let cfg = build_model_config(f.path(), "llama").unwrap();
        assert_eq!(cfg.vocab_size, 9999);
    }

    #[test]
    fn build_model_config_missing_required_key_returns_err() {
        // Omit block_count — must fail.
        let data = make_gguf(&[
            ("llama.embedding_length",        KvValue::U32(64)),
            ("llama.attention.head_count",    KvValue::U32(1)),
            ("llama.attention.head_count_kv", KvValue::U32(1)),
            ("llama.feed_forward_length",     KvValue::U32(128)),
            ("llama.vocab_size",              KvValue::U32(100)),
        ]);
        let f = write_to_tempfile(&data);
        let err = build_model_config(f.path(), "llama").unwrap_err();
        assert!(matches!(err, GwenError::ModelLoad(_)));
    }

    // ── validate_gguf (via the existing parser; tensor_count=0 so limits don't trip) ──

    #[test]
    fn validate_gguf_bad_magic_returns_err() {
        // Write a file with wrong magic bytes.
        let mut data = vec![0u8; 32];
        data[0..4].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        let f = write_to_tempfile(&data);
        let err = validate_gguf(f.path()).unwrap_err();
        assert!(matches!(err, GwenError::ModelLoad(_)));
    }
}
