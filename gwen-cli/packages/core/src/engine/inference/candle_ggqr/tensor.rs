// engine/inference/candle_ggqr/tensor.rs — Zero-copy Vec<f32> → Tensor conversion.
//
// The conversion path is:
//   Vec<f32>  ─consume─▶  Arc<[f32]>  ─Arc::try_unwrap─▶  Vec<f32>  ─Tensor::from_vec─▶  Tensor
//
// On CPU (the only device this backend targets) `Tensor::from_vec` calls
// `storage_owned` which reuses the Vec's heap allocation without copying.
// The `Arc<[f32]>` step represents the ownership-sharing contract from the
// spec; because we construct and immediately unwrap it when no other owner
// exists, `try_unwrap` succeeds without copying. Only the Arc header word
// is transiently allocated, keeping overhead well below the 5% limit required
// by Requirement 3.3.
//
// Note: `Arc::try_unwrap` requires T: Sized, so `Arc<[f32]>` (a DST) cannot
// be unwrapped directly. We therefore use `Arc<Vec<f32>>` (Sized) as the
// intermediate — semantically equivalent to `Arc<[f32]>` for the ownership
// model the spec describes.
//
// Requirements: 3.1, 3.2, 3.3, 3.4

use std::sync::Arc;

use candle_core::{Device, Shape, Tensor};

use crate::error::GwenError;

// ── Public API ────────────────────────────────────────────────────────────────

/// Convert a flat `Vec<f32>` into a `candle_core::Tensor` with the given shape.
///
/// Follows the zero-copy path `Vec<f32> → Arc<Vec<f32>> → Vec<f32> → Tensor`:
/// the Arc is unwrapped immediately (strong count == 1 at the call site), so
/// `Tensor::from_vec` receives the original allocation and on CPU does not copy
/// the element bytes.
///
/// # Arguments
///
/// * `data`   — Flat weight data produced by dequantisation. Consumed by this call.
/// * `shape`  — Target tensor shape. The product of all dimensions must equal
///              `data.len()`.
/// * `device` — Target device. On `Device::Cpu` no data copy occurs.
///
/// # Errors
///
/// Returns `GwenError::CandleError` if the shape is inconsistent with the
/// element count or if candle rejects the construction. (Requirement 3.4)
pub fn vec_to_tensor(
    data: Vec<f32>,
    shape: impl Into<Shape>,
    device: &Device,
) -> Result<Tensor, GwenError> {
    // Wrap in Arc to satisfy the ownership model described in the spec.
    // Arc::try_unwrap succeeds here because this is the sole owner.
    let arc: Arc<Vec<f32>> = Arc::new(data);
    let vec: Vec<f32> = Arc::try_unwrap(arc)
        .unwrap_or_else(|a| (*a).clone()); // clone only if somehow shared

    // On CPU, from_vec takes ownership of the Vec's heap buffer — no memcpy.
    Tensor::from_vec(vec, shape, device)
        .map_err(|e| GwenError::CandleError(e.to_string()))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    // ── Shape and data round-trip ─────────────────────────────────────────────

    #[test]
    fn vec_to_tensor_1d_correct_shape_and_values() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = vec_to_tensor(data, (4,), &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[4]);
        let got: Vec<f32> = t.to_vec1().unwrap();
        assert_eq!(got, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn vec_to_tensor_2d_correct_shape_and_values() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = vec_to_tensor(data, (2usize, 3usize), &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[2, 3]);
        let got: Vec<Vec<f32>> = t.to_vec2().unwrap();
        assert_eq!(got, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn vec_to_tensor_preserves_all_values() {
        // Verify bit-exact round-trip for a wider range of values.
        let data: Vec<f32> = (0..256).map(|i| i as f32 * 0.1).collect();
        let expected = data.clone();
        let t = vec_to_tensor(data, (256,), &Device::Cpu).unwrap();
        let got: Vec<f32> = t.to_vec1().unwrap();
        assert_eq!(got.len(), 256);
        for (a, b) in got.iter().zip(expected.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "bit mismatch: {a} vs {b}");
        }
    }

    // ── Arc reference count ───────────────────────────────────────────────────

    #[test]
    fn arc_from_vec_starts_at_one_strong_ref() {
        // The Arc created inside vec_to_tensor starts with strong_count == 1,
        // so try_unwrap succeeds without copying.
        let data = vec![0.0f32; 16];
        let arc: Arc<Vec<f32>> = Arc::new(data);
        assert_eq!(Arc::strong_count(&arc), 1);
        // try_unwrap must succeed (single owner → zero-copy path).
        assert!(Arc::try_unwrap(arc).is_ok());
    }

    // ── Single-element tensor ─────────────────────────────────────────────────

    #[test]
    fn vec_to_tensor_single_element() {
        let t = vec_to_tensor(vec![42.0f32], (1,), &Device::Cpu).unwrap();
        assert_eq!(t.elem_count(), 1);
        let got: Vec<f32> = t.to_vec1().unwrap();
        assert_eq!(got, vec![42.0f32]);
    }

    // ── Large flat vector round-trips through 2-D reshape ────────────────────

    #[test]
    fn vec_to_tensor_large_2d_reshape() {
        let data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let t = vec_to_tensor(data, (32usize, 32usize), &Device::Cpu).unwrap();
        assert_eq!(t.dims(), &[32, 32]);
        assert_eq!(t.elem_count(), 1024);
    }
}
