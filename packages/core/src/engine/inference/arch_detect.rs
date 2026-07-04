// engine/inference/arch_detect.rs — GGUF architecture detection.
//
// Reads only the GGUF KV metadata section to extract `general.architecture`.
// Does NOT parse tensor data — this is fast and only reads the header.
//
// Requirements: 3.2, 5.1–5.8, 16.2, 16.3

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use anyhow::{bail, Result};

// Re-use low-level primitive readers from gguf_parser (they're pub)
use crate::convert::gguf_parser::{read_gguf_string, read_u32_le, read_u64_le};

/// GGUF magic bytes in little-endian u32 form.
const GGUF_MAGIC: u32 = 0x4647_4755; // b"GGUF"

/// Maximum tensor count accepted before rejecting as potentially malicious.
const MAX_TENSOR_COUNT: u64 = 10_000;

/// Detect the model architecture by reading the GGUF header's KV metadata.
///
/// Extracts the `general.architecture` string and maps it to the identifier
/// used by mistral.rs.
///
/// # Errors
///
/// Returns `Err` when:
/// - The file does not exist or cannot be read
/// - The file does not start with GGUF magic bytes (Req 16.2)
/// - The tensor count exceeds 10,000 (security limit, Req 16.3)
/// - The `general.architecture` key is absent
/// - The architecture is not in the supported set (Req 5.8)
pub fn detect_architecture(path: &Path) -> Result<&'static str> {
    let file = File::open(path)
        .map_err(|e| anyhow::anyhow!("cannot open '{}': {}", path.display(), e))?;
    let mut r = BufReader::new(file);

    // Validate magic bytes (Requirement 16.2)
    let magic = read_u32_le(&mut r)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF magic: {}", e))?;
    if magic != GGUF_MAGIC {
        bail!("'{}' is not a GGUF file (invalid magic bytes)", path.display());
    }

    // Read version (u32) — must be 1, 2, or 3
    let _version = read_u32_le(&mut r)
        .map_err(|e| anyhow::anyhow!("failed to read GGUF version: {}", e))?;

    // Read tensor_count — security limit (Requirement 16.3)
    let tensor_count = read_u64_le(&mut r)
        .map_err(|e| anyhow::anyhow!("failed to read tensor count: {}", e))?;
    if tensor_count > MAX_TENSOR_COUNT {
        bail!(
            "GGUF file claims {} tensors which exceeds the safety limit of {}",
            tensor_count, MAX_TENSOR_COUNT
        );
    }

    // Read kv_count
    let kv_count = read_u64_le(&mut r)
        .map_err(|e| anyhow::anyhow!("failed to read KV count: {}", e))?;

    // Scan KV entries for `general.architecture`
    let mut arch: Option<String> = None;
    for _ in 0..kv_count {
        let key = read_gguf_string(&mut r)
            .map_err(|e| anyhow::anyhow!("failed to read KV key: {}", e))?;
        let value_type = read_u32_le(&mut r)
            .map_err(|e| anyhow::anyhow!("failed to read KV value type: {}", e))?;

        if key == "general.architecture" && value_type == 8 {
            // Value type 8 = STRING
            let val = read_gguf_string(&mut r)
                .map_err(|e| anyhow::anyhow!("failed to read architecture string: {}", e))?;
            arch = Some(val);
            break; // Found what we need, stop reading
        } else {
            // Skip this value
            skip_kv_value(&mut r, value_type)?;
        }
    }

    let arch_str = arch.ok_or_else(|| {
        anyhow::anyhow!("'general.architecture' key not found in GGUF metadata")
    })?;

    // Map to mistral.rs architecture identifiers (Requirements 5.3-5.7)
    match arch_str.as_str() {
        "llama"           => Ok("llama"),
        "qwen2" | "qwen"  => Ok("qwen2"),
        "phi3"            => Ok("phi3"),
        "mistral"         => Ok("mistral"),
        "gemma"           => Ok("gemma"),
        other => bail!(
            "unsupported model architecture for mistralrs: '{}' \
             (supported: llama, qwen/qwen2, phi3, mistral, gemma)",
            other
        ),
    }
}

/// Skip a KV value of the given type without interpreting it.
fn skip_kv_value<R: Read>(r: &mut R, vtype: u32) -> Result<()> {
    match vtype {
        0 | 1 | 7 => { read_byte(r)?; }          // UINT8/INT8/BOOL
        2 | 3     => { read_exact(r, 2)?; }       // UINT16/INT16
        4 | 5 | 6 => { read_exact(r, 4)?; }       // UINT32/INT32/FLOAT32
        8         => { read_gguf_string(r).map_err(|e| anyhow::anyhow!("{}", e))?; } // STRING
        9         => {                             // ARRAY
            let elem_type = read_u32_le(r).map_err(|e| anyhow::anyhow!("{}", e))?;
            let count = read_u64_le(r).map_err(|e| anyhow::anyhow!("{}", e))?;
            for _ in 0..count {
                skip_kv_value(r, elem_type)?;
            }
        }
        10 | 11 | 12 => { read_exact(r, 8)?; }   // UINT64/INT64/FLOAT64
        other => bail!("unknown KV value type {} in GGUF metadata", other),
    }
    Ok(())
}

fn read_byte<R: Read>(r: &mut R) -> Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b).map_err(|e| anyhow::anyhow!("read error: {}", e))?;
    Ok(b[0])
}

fn read_exact<R: Read>(r: &mut R, n: usize) -> Result<()> {
    let mut buf = vec![0u8; n];
    r.read_exact(&mut buf).map_err(|e| anyhow::anyhow!("read error: {}", e))?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    // ── GGUF binary builder ───────────────────────────────────────────────────

    /// Write a minimal valid GGUF file with a single `general.architecture`
    /// KV entry set to `arch_value`.
    ///
    /// GGUF layout:
    ///   magic (u32 LE) | version (u32 LE) | tensor_count (u64 LE) |
    ///   kv_count (u64 LE) | KV entries…
    ///
    /// For the magic, GGUF_MAGIC = 0x4647_4755 stored as little-endian bytes
    /// = [0x55, 0x47, 0x47, 0x46].
    fn write_gguf_with_arch(arch_value: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        write_valid_gguf_header(&mut f, 0, 1);
        write_arch_kv_entry(&mut f, arch_value);
        f.flush().unwrap();
        f
    }

    /// Write a GGUF header with configurable tensor_count and kv_count.
    fn write_valid_gguf_header(f: &mut impl Write, tensor_count: u64, kv_count: u64) {
        // magic: 0x4647_4755 as little-endian bytes
        f.write_all(&[0x55u8, 0x47, 0x47, 0x46]).unwrap();
        // version 3 as u32 LE
        f.write_all(&3u32.to_le_bytes()).unwrap();
        // tensor_count as u64 LE
        f.write_all(&tensor_count.to_le_bytes()).unwrap();
        // kv_count as u64 LE
        f.write_all(&kv_count.to_le_bytes()).unwrap();
    }

    /// Write the `general.architecture` KV entry (key + value_type=8 + value).
    fn write_arch_kv_entry(f: &mut impl Write, arch_value: &str) {
        write_gguf_string_bytes(f, "general.architecture");
        // value_type = 8 (STRING) as u32 LE
        f.write_all(&8u32.to_le_bytes()).unwrap();
        write_gguf_string_bytes(f, arch_value);
    }

    /// Write a GGUF-format string: u64 length prefix + UTF-8 bytes.
    fn write_gguf_string_bytes(f: &mut impl Write, s: &str) {
        f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
        f.write_all(s.as_bytes()).unwrap();
    }

    // ── Architecture mapping tests ────────────────────────────────────────────

    #[test]
    fn test_detect_llama() {
        let f = write_gguf_with_arch("llama");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "llama");
    }

    #[test]
    fn test_detect_qwen() {
        let f = write_gguf_with_arch("qwen");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "qwen2");
    }

    #[test]
    fn test_detect_qwen2() {
        let f = write_gguf_with_arch("qwen2");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "qwen2");
    }

    #[test]
    fn test_detect_phi3() {
        let f = write_gguf_with_arch("phi3");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "phi3");
    }

    #[test]
    fn test_detect_mistral() {
        let f = write_gguf_with_arch("mistral");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "mistral");
    }

    #[test]
    fn test_detect_gemma() {
        let f = write_gguf_with_arch("gemma");
        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "gemma");
    }

    // ── Error path tests ──────────────────────────────────────────────────────

    #[test]
    fn test_unsupported_architecture_returns_err() {
        let f = write_gguf_with_arch("rwkv");
        let err = detect_architecture(f.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("rwkv"),
            "error message should contain the unsupported arch name, got: {msg}"
        );
    }

    #[test]
    fn test_invalid_magic_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        // Write garbage magic bytes
        f.write_all(&[0xDE, 0xAD, 0xBE, 0xEF]).unwrap();
        // version, tensor_count, kv_count
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.flush().unwrap();

        let err = detect_architecture(f.path()).unwrap_err();
        assert!(
            err.to_string().contains("not a GGUF file") || err.to_string().contains("invalid magic"),
            "expected magic validation error, got: {}",
            err
        );
    }

    #[test]
    fn test_tensor_count_exceeds_limit_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        write_valid_gguf_header(&mut f, 10_001, 0);
        f.flush().unwrap();

        let err = detect_architecture(f.path()).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("10001") || msg.contains("safety limit") || msg.contains("exceeds"),
            "expected tensor count safety error, got: {msg}"
        );
    }

    #[test]
    fn test_tensor_count_at_limit_is_ok_if_arch_present() {
        // Exactly 10,000 tensors should be accepted (limit is > 10_000)
        let mut f = NamedTempFile::new().unwrap();
        write_valid_gguf_header(&mut f, 10_000, 1);
        write_arch_kv_entry(&mut f, "llama");
        f.flush().unwrap();

        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "llama");
    }

    #[test]
    fn test_missing_architecture_key_returns_err() {
        let mut f = NamedTempFile::new().unwrap();
        // kv_count = 1, but write a different key
        write_valid_gguf_header(&mut f, 0, 1);
        // Write a KV entry with key "general.name" (not "general.architecture")
        write_gguf_string_bytes(&mut f, "general.name");
        // value_type = 8 (STRING)
        f.write_all(&8u32.to_le_bytes()).unwrap();
        write_gguf_string_bytes(&mut f, "my-model");
        f.flush().unwrap();

        let err = detect_architecture(f.path()).unwrap_err();
        assert!(
            err.to_string().contains("general.architecture"),
            "error should mention the missing key, got: {}",
            err
        );
    }

    #[test]
    fn test_arch_key_not_first_entry_still_found() {
        // Put another KV entry before general.architecture to test the skip logic
        let mut f = NamedTempFile::new().unwrap();
        // kv_count = 2
        write_valid_gguf_header(&mut f, 0, 2);

        // First entry: "general.name" = STRING "my-model"
        write_gguf_string_bytes(&mut f, "general.name");
        f.write_all(&8u32.to_le_bytes()).unwrap(); // value_type STRING
        write_gguf_string_bytes(&mut f, "my-model");

        // Second entry: "general.architecture" = STRING "gemma"
        write_arch_kv_entry(&mut f, "gemma");
        f.flush().unwrap();

        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "gemma");
    }

    #[test]
    fn test_nonexistent_file_returns_err() {
        let path = std::path::Path::new("/nonexistent/path/model.gguf");
        let err = detect_architecture(path).unwrap_err();
        assert!(
            err.to_string().contains("cannot open"),
            "expected file-open error, got: {}",
            err
        );
    }

    #[test]
    fn test_skip_numeric_kv_types_before_arch() {
        // Verify the skip logic handles UINT32, FLOAT32, UINT64 types correctly
        let mut f = NamedTempFile::new().unwrap();
        write_valid_gguf_header(&mut f, 0, 4);

        // UINT32 entry
        write_gguf_string_bytes(&mut f, "some.uint32");
        f.write_all(&4u32.to_le_bytes()).unwrap(); // value_type UINT32
        f.write_all(&42u32.to_le_bytes()).unwrap();

        // FLOAT32 entry
        write_gguf_string_bytes(&mut f, "some.float");
        f.write_all(&6u32.to_le_bytes()).unwrap(); // value_type FLOAT32
        f.write_all(&1.0f32.to_le_bytes()).unwrap();

        // UINT64 entry
        write_gguf_string_bytes(&mut f, "some.uint64");
        f.write_all(&10u32.to_le_bytes()).unwrap(); // value_type UINT64
        f.write_all(&100u64.to_le_bytes()).unwrap();

        // Finally, the architecture entry
        write_arch_kv_entry(&mut f, "mistral");
        f.flush().unwrap();

        let result = detect_architecture(f.path()).unwrap();
        assert_eq!(result, "mistral");
    }
}
