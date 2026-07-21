//! GLLM tensor decoding — turns a layer file's raw tensor bytes into
//! [`Tensor`](crate::tensor::Tensor)s the engines can compute on.
//!
//! This module is deliberately byte-in, `Tensor`-out: it knows nothing about
//! `.gllm` package layout, manifests, or mmap — that lives in
//! `glictus-caliburni`, which hands this module the raw bytes for one named
//! tensor (already located via its own tensor index) plus the dtype and
//! shape the manifest recorded. Mirrors [`super::gguf`]'s
//! `tensor_data`/`dequantize` split, so a GLLM package and a GGUF file are
//! decoded through the same shape of API.
//!
//! ARTX10 Wave 1 ships `F32` only, since every consumer (glproc's per-op
//! kernels) computes in `f32` and the GLLM converter (ARTX7) has so far only
//! been exercised producing dense `F32`/`Q8_0` layers. Quantized dtypes
//! return [`GlError::UnsupportedDtype`] rather than silently truncating —
//! the same choice `gguf::dequantize` makes for `Q4_K`/`Q5_0`.

use byteorder::{ByteOrder, LittleEndian};

use crate::error::GlError;
use crate::tensor::Tensor;

/// GLLM manifest dtype codes this decoder understands.
///
/// A small, local mirror of `glictus-caliburni::manifest::DType` — this
/// crate does not depend on glictus-caliburni, so the caller passes the
/// dtype as a plain string (the manifest's own serde representation, e.g.
/// `"F32"`) and this module matches on it rather than sharing a type.
fn bytes_per_element(dtype: &str) -> Option<usize> {
    match dtype {
        "F32" => Some(4),
        "F16" | "BF16" => Some(2),
        _ => None,
    }
}

/// Decode one tensor's raw little-endian bytes into an owned `f32` [`Tensor`].
///
/// `dtype` is the manifest's dtype string (e.g. `"F32"`, `"Q4_K_M"`); `shape`
/// is the manifest's recorded dimensions. `raw` must be exactly
/// `numel * bytes_per_element(dtype)` bytes — a mismatch means the manifest
/// and the tensor index disagree, which is a corrupt package, not something
/// this decoder should paper over.
///
/// Only `F32`, `F16`, and `BF16` are supported (Wave 1 scope — see module
/// docs); anything else is [`GlError::UnsupportedDtype`].
pub fn decode_tensor(name: &str, shape: &[u64], dtype: &str, raw: &[u8]) -> Result<Tensor, GlError> {
    let numel: u64 = shape.iter().product();
    let elem_bytes = bytes_per_element(dtype).ok_or_else(|| {
        GlError::UnsupportedDtype(format!(
            "{dtype} (tensor {name}) — GLLM Wave 1 CPU path computes in f32 only"
        ))
    })?;
    let expected = numel as usize * elem_bytes;
    if raw.len() != expected {
        return Err(GlError::Parse(format!(
            "tensor {name}: expected {expected} bytes for shape {shape:?} as {dtype}, got {}",
            raw.len()
        )));
    }

    let data: Vec<f32> = match dtype {
        "F32" => raw.chunks_exact(4).map(LittleEndian::read_f32).collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|b| crate::format::gguf::f16_to_f32(LittleEndian::read_u16(b)))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|b| f32::from_bits((LittleEndian::read_u16(b) as u32) << 16))
            .collect(),
        _ => unreachable!("bytes_per_element already rejected every other dtype"),
    };

    let shape_usize: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    Ok(Tensor::new(data, shape_usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn decodes_f32_tensor() {
        let raw = f32_bytes(&[1.0, -2.5, 3.25, 0.0]);
        let t = decode_tensor("attn_q.weight", &[2, 2], "F32", &raw).unwrap();
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data, vec![1.0, -2.5, 3.25, 0.0]);
        assert_eq!(t.dtype, crate::tensor::DType::F32);
    }

    #[test]
    fn decodes_f16_tensor() {
        // 1.0 in f16 is 0x3C00.
        let raw = 0x3C00u16.to_le_bytes();
        let t = decode_tensor("x", &[1], "F16", &raw).unwrap();
        assert_eq!(t.data, vec![1.0]);
    }

    #[test]
    fn rejects_quantized_dtype() {
        let err = decode_tensor("ffn_down.weight", &[8, 8], "Q4_K_M", &[0u8; 32]).unwrap_err();
        assert!(matches!(err, GlError::UnsupportedDtype(_)), "got {err:?}");
    }

    #[test]
    fn rejects_a_byte_count_that_disagrees_with_shape() {
        let raw = f32_bytes(&[1.0, 2.0]); // 8 bytes, shape wants 4 (16 bytes)
        let err = decode_tensor("x", &[2, 2], "F32", &raw).unwrap_err();
        assert!(matches!(err, GlError::Parse(_)), "got {err:?}");
    }

    #[test]
    fn zero_shape_dim_yields_zero_elements() {
        // Degenerate but not this decoder's job to reject (TensorEntry::validate
        // already rules out zero dims at the manifest layer).
        let t = decode_tensor("x", &[0], "F32", &[]).unwrap();
        assert!(t.data.is_empty());
    }
}
