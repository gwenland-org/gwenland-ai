// engine/inference/candle_ggqr/dequant.rs — GGQR dequantisation integration.
//
// Wraps `convert::dequant::dequantize` with `DequantMode::Standard` and maps
// the raw `String` error to `GwenError::Dequantization` so callers get a typed
// error with the tensor name attached.
//
// F32 and F16 tensors are pass-through: `dequantize` handles them already.
// Q2_K … Q6_K are delegated to the GGQR experimental paths; callers should
// validate the GGUF file before calling here.
//
// Requirements: 2.1, 2.2, 2.3, 2.4, 2.5

use crate::convert::dequant::{DequantMode, dequantize};
use crate::convert::gguf_parser::TensorInfo;
use crate::error::GwenError;

// ── Public API ────────────────────────────────────────────────────────────────

/// Dequantise `tensor` to a flat `Vec<f32>` using linear (Standard) mode.
///
/// Supports all dtypes handled by the GGQR engine:
/// - `F32` / `F16` — direct copy / upcast, no quantisation applied.
/// - `Q4_0` / `Q8_0` — symmetric block dequantisation (stable).
/// - `Q2_K` / `Q3_K` / `Q4_K` / `Q5_K` / `Q6_K` — K-quant superblocks
///   (experimental; verified against GGML reference bit-patterns).
///
/// # Errors
///
/// Returns `GwenError::Dequantization` with the tensor name and the underlying
/// error message on failure. (Requirement 2.5)
pub fn dequantize_tensor(tensor: &TensorInfo) -> Result<Vec<f32>, GwenError> {
    dequantize(tensor, DequantMode::Standard).map_err(|e| GwenError::Dequantization {
        tensor_name: tensor.name.clone(),
        error: e,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::convert::gguf_parser::{GgufDtype, TensorInfo};

    // Build a minimal TensorInfo with the given dtype and raw_data.
    fn make_tensor(name: &str, dtype: GgufDtype, raw_data: Vec<u8>, shape: Vec<u64>) -> TensorInfo {
        TensorInfo {
            name: name.to_string(),
            shape,
            dtype,
            data_offset: 0,
            data_size: raw_data.len(),
            raw_data,
        }
    }

    // ── F32 pass-through ──────────────────────────────────────────────────────

    #[test]
    fn f32_tensor_returns_data_unchanged() {
        // Two f32 values: 1.0 and -2.5.
        let v1: f32 = 1.0;
        let v2: f32 = -2.5;
        let mut raw = Vec::new();
        raw.extend_from_slice(&v1.to_le_bytes());
        raw.extend_from_slice(&v2.to_le_bytes());

        let tensor = make_tensor("test.weight", GgufDtype::F32, raw, vec![2]);
        let result = dequantize_tensor(&tensor).unwrap();

        assert_eq!(result.len(), 2);
        assert!((result[0] - 1.0).abs() < f32::EPSILON, "expected 1.0, got {}", result[0]);
        assert!((result[1] - (-2.5)).abs() < f32::EPSILON, "expected -2.5, got {}", result[1]);
    }

    // ── Q4_K dequantisation produces a Vec<f32> ───────────────────────────────

    #[test]
    fn q4_k_tensor_dequantizes_to_vec_f32() {
        // One Q4_K superblock = 256 elements, 144 bytes.
        // Fill with zeros — all zero quantised values dequantise to 0.0.
        let raw = vec![0u8; 144];
        let tensor = make_tensor("attn.weight", GgufDtype::Q4_K, raw, vec![256]);
        let result = dequantize_tensor(&tensor).unwrap();

        assert_eq!(result.len(), 256, "expected 256 f32 values from one Q4_K superblock");
        // All-zero superblock: d=0, dmin=0, all q=0 → all weights are 0.0.
        assert!(
            result.iter().all(|&v| v == 0.0),
            "all-zero Q4_K block should produce all-zero f32 weights"
        );
    }

    // ── Unsupported / corrupt data returns GwenError::Dequantization ──────────

    #[test]
    fn truncated_q4_k_buffer_returns_dequantization_error() {
        // A Q4_K superblock needs 144 bytes but we give only 10.
        let raw = vec![0u8; 10];
        let tensor = make_tensor("corrupt.weight", GgufDtype::Q4_K, raw, vec![256]);
        let err = dequantize_tensor(&tensor).unwrap_err();

        match err {
            GwenError::Dequantization { tensor_name, .. } => {
                assert_eq!(tensor_name, "corrupt.weight");
            }
            other => panic!("expected Dequantization error, got {:?}", other),
        }
    }

    // ── Error carries the tensor name ─────────────────────────────────────────

    #[test]
    fn dequantization_error_includes_tensor_name() {
        // Give a Q8_0 tensor a buffer that is too small to parse even one block.
        let raw = vec![0u8; 1]; // Q8_0 block = 34 bytes, we give 1.
        let tensor = make_tensor("my.special.tensor", GgufDtype::Q8_0, raw, vec![32]);
        let err = dequantize_tensor(&tensor).unwrap_err();

        let name = match &err {
            GwenError::Dequantization { tensor_name, .. } => tensor_name.clone(),
            other => panic!("wrong error variant: {:?}", other),
        };
        assert_eq!(name, "my.special.tensor");
    }
}
