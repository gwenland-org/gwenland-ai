//! Core tensor representation shared by all engines.
//!
//! Phase 1 keeps this deliberately simple: owned, heap-allocated, row-major
//! `f32` storage. Quantized dtypes are tracked so parsers can describe what
//! is stored on disk, but compute always happens in `f32`.

use crate::error::GlError;

/// Element type of a tensor as stored on disk or in memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // Q4_0/Q4_K/Q8_0 are the canonical GGML names
pub enum DType {
    /// 32-bit IEEE float (the only compute dtype in Phase 1).
    F32,
    /// 16-bit IEEE float.
    F16,
    /// bfloat16.
    BF16,
    /// GGML 4-bit quantization, block size 32.
    Q4_0,
    /// GGML 4-bit K-quantization, super-block size 256.
    Q4_K,
    /// GGML 8-bit quantization, block size 32.
    Q8_0,
}

/// Core tensor representation — owned, heap-allocated, `f32` only for Phase 1.
///
/// Data is row-major: the last dimension is contiguous.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Flat row-major element storage.
    pub data: Vec<f32>,
    /// Dimension sizes, outermost first.
    pub shape: Vec<usize>,
    /// Logical dtype. Always `F32` for in-memory compute tensors.
    pub dtype: DType,
}

impl Tensor {
    /// Create a tensor from existing data and a shape.
    ///
    /// The caller must ensure `data.len() == shape.iter().product()`.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        Tensor {
            data,
            shape,
            dtype: DType::F32,
        }
    }

    /// Create a zero-filled tensor of the given shape.
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel = shape.iter().product();
        Tensor {
            data: vec![0.0; numel],
            shape,
            dtype: DType::F32,
        }
    }

    /// Total number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Return a copy of this tensor with a new shape.
    ///
    /// Fails with [`GlError::ShapeMismatch`] if the element counts differ.
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, GlError> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(GlError::ShapeMismatch {
                expected: self.shape.clone(),
                got: new_shape,
            });
        }
        Ok(Tensor {
            data: self.data.clone(),
            shape: new_shape,
            dtype: self.dtype,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_and_numel() {
        let t = Tensor::zeros(vec![2, 3, 4]);
        assert_eq!(t.numel(), 24);
        assert_eq!(t.data.len(), 24);
        assert!(t.data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn reshape_ok() {
        let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let r = t.reshape(vec![3, 2]).unwrap();
        assert_eq!(r.shape, vec![3, 2]);
        assert_eq!(r.data, t.data);
    }

    #[test]
    fn reshape_mismatch() {
        let t = Tensor::zeros(vec![2, 3]);
        assert!(matches!(
            t.reshape(vec![4, 2]),
            Err(GlError::ShapeMismatch { .. })
        ));
    }
}
