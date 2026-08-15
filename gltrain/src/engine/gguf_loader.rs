use std::collections::HashMap;
use std::path::Path;

use crate::convert::dequant::{self, DequantMode};
use crate::convert::gguf_parser::{self, GgufFile};

use super::loader::{LoadMode, MmapLoader};

/// Load a GGUF file from disk using mmap and parse it into a [`GgufFile`].
///
/// This is the single entry point for all GGUF loading in GwenLand.
///
/// # Pipeline
///
/// ```text
/// disk → MmapLoader (mmap + MADV_SEQUENTIAL hint, auto LoadMode)
///      → magic validation → gguf_parser::parse() → GgufFile
/// ```
///
/// The mmap step validates the magic header, auto-detects [`LoadMode`] based
/// on available RAM vs file size, and advises the OS accordingly. Structured
/// header/tensor parsing is delegated to [`gguf_parser::parse`].
pub fn load_gguf(path: &Path) -> Result<GgufFile, String> {
    let loader = MmapLoader::open(path)?;
    eprintln!("[gwenland] load mode: {:?}", loader.mode);
    gguf_parser::parse(path)
}

/// Load a GGUF file with an explicit [`LoadMode`], bypassing auto-detection.
///
/// Useful for benchmarking (force Eager/Lazy) or when the caller has already
/// determined available RAM.
pub fn load_gguf_with_mode(path: &Path, mode: LoadMode) -> Result<GgufFile, String> {
    let loader = MmapLoader::open_with_mode(path, mode)?;
    eprintln!("[gwenland] load mode: {:?} (explicit)", loader.mode);
    gguf_parser::parse(path)
}

/// Dequantise all tensors in a loaded [`GgufFile`].
///
/// Returns a map of `tensor_name → Vec<f32>` weights.
///
/// Tensors whose dequant returns an error containing `"unsupported"` are
/// skipped with a warning printed to stderr; all other errors propagate as
/// `Err`.
pub fn dequant_all(
    model: &GgufFile,
    mode: DequantMode,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let mut out = HashMap::new();
    for tensor in &model.tensors {
        match dequant::dequantize(tensor, mode) {
            Ok(weights) => {
                out.insert(tensor.name.clone(), weights);
            }
            Err(e) if e.contains("unsupported") => {
                eprintln!("[warn] skipping tensor '{}': {}", tensor.name, e);
            }
            Err(e) => return Err(e),
        }
    }
    Ok(out)
}

/// Load and dequantise a GGUF file in one call.
///
/// Convenience wrapper around [`load_gguf`] + [`dequant_all`].
///
/// # Regression baseline
///
/// Qwen3-1.7B Q8_0, Standard mode (verified 2026-06-07, GGQR-CF-mmap v1.0):
///   `sum = 340_913_024`
///
/// To verify: `bench_ggqr <path> --expected-sum 340913024`
pub fn load_and_dequant(
    path: &Path,
    mode: DequantMode,
) -> Result<HashMap<String, Vec<f32>>, String> {
    let model = load_gguf(path)?;
    dequant_all(&model, mode)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use crate::convert::gguf_parser::{GgufDtype, TensorInfo};

    #[test]
    fn test_load_gguf_nonexistent_path() {
        let result = load_gguf(Path::new("nonexistent_file_that_does_not_exist.gguf"));
        assert!(result.is_err(), "expected Err for nonexistent path");
    }

    #[test]
    fn test_load_gguf_invalid_gguf() {
        let mut f = NamedTempFile::new().expect("tempfile");
        f.write_all(b"RANDOM_GARBAGE_BYTES_NOT_A_GGUF_FILE").expect("write");
        f.flush().expect("flush");

        let result = load_gguf(f.path());
        assert!(result.is_err(), "expected Err for invalid GGUF bytes");
        let msg = result.err().unwrap();
        assert!(!msg.is_empty(), "error message should be non-empty");
    }

    /// Helper: build a minimal GgufFile with the given tensors.
    fn make_model(tensors: Vec<TensorInfo>) -> GgufFile {
        GgufFile { version: 3, tensors, data_base: 0 }
    }

    /// Helper: build a TensorInfo with pre-filled F32 raw data for N elements.
    fn f32_tensor(name: &str, n: usize) -> TensorInfo {
        let mut raw = Vec::with_capacity(n * 4);
        for i in 0..n {
            raw.extend_from_slice(&(i as f32).to_le_bytes());
        }
        TensorInfo {
            name: name.to_string(),
            shape: vec![n as u64],
            dtype: GgufDtype::F32,
            data_offset: 0,
            data_size: raw.len(),
            raw_data: raw,
        }
    }

    #[test]
    fn test_dequant_all_empty_model() {
        let model = make_model(vec![]);
        let result = dequant_all(&model, DequantMode::Standard);
        assert!(result.is_ok(), "expected Ok for empty model");
        assert!(result.unwrap().is_empty(), "expected empty HashMap");
    }

    #[test]
    fn test_dequant_all_unsupported_dtype_skipped() {
        let zero = TensorInfo {
            name: "zero_tensor".to_string(),
            shape: vec![0u64],
            dtype: GgufDtype::F32,
            data_offset: 0,
            data_size: 0,
            raw_data: vec![],
        };
        let good = f32_tensor("good_tensor", 4);
        let model = make_model(vec![zero, good]);
        let result = dequant_all(&model, DequantMode::Standard);
        assert!(result.is_ok(), "expected Ok: {:?}", result.err());
        let map = result.unwrap();
        assert!(map.contains_key("good_tensor"), "good tensor should be in output");
        assert_eq!(map["good_tensor"].len(), 4);
        assert!(map.contains_key("zero_tensor"), "zero tensor should be in output");
        assert!(map["zero_tensor"].is_empty());
    }
}
