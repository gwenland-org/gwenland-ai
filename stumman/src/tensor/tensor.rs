//! Stummañ Kevrin — Tensor abstraction.
//!
//! `Tensor<B>` is the core data structure for Stummañ.
//! It wraps backend storage with shape metadata.
//! Wave 1: no autograd, no gradient tracking. Pure data.
//! Wave 2 will add: tape reference, requires_grad, grad field.

use crate::error::{GlTrainError, Result};
use crate::tensor::backend::Backend;
use std::sync::Arc;

/// A multi-dimensional tensor backed by compute backend B.
///
/// Shape semantics:
/// - 1D: `[N]`          — vector
/// - 2D: `[M, N]`       — matrix (row-major)
/// - 3D: `[B, M, N]`    — batched matrix
///
/// Storage is reference-counted (Arc) so tensors can be cloned cheaply.
/// Cloning a Tensor shares the underlying storage (copy-on-write NOT implemented
/// in Wave 1 — mutating ops always allocate new storage).
#[derive(Clone)]
pub struct Tensor<B: Backend> {
    /// Shape of the tensor (e.g., [4, 8] for a 4×8 matrix)
    pub(crate) shape: Vec<usize>,

    /// Backend-specific storage (reference-counted for cheap clone)
    pub(crate) storage: Arc<B::Storage>,

    /// Phantom marker for backend type
    _backend: std::marker::PhantomData<B>,
}

impl<B: Backend> Tensor<B> {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Create a tensor filled with zeros of the given shape.
    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        let storage = B::zeros(n)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Create a tensor filled with ones of the given shape.
    pub fn ones(shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        let storage = B::ones(n)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Create a tensor from a host `Vec<f32>` with the given shape.
    ///
    /// Returns [`GlTrainError::ShapeMismatch`] if `data.len()` disagrees with
    /// the product of `shape`.
    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        let n = shape_to_n_elems(shape)?;
        if data.len() != n {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![n],
                got: vec![data.len()],
            });
        }
        let storage = B::from_vec(data)?;
        Ok(Self::from_storage(storage, shape.to_vec()))
    }

    /// Internal constructor from raw storage + shape.
    pub(crate) fn from_storage(storage: B::Storage, shape: Vec<usize>) -> Self {
        Self {
            shape,
            storage: Arc::new(storage),
            _backend: std::marker::PhantomData,
        }
    }

    // ── Shape accessors ──────────────────────────────────────────────────

    /// Return the tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    pub fn n_elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    // ── Data access ──────────────────────────────────────────────────────

    /// Copy tensor data to a host `Vec<f32>`.
    pub fn to_vec(&self) -> Result<Vec<f32>> {
        B::to_vec(&self.storage)
    }

    /// Return a single scalar value (only valid for 0-d or 1-elem tensors).
    pub fn item(&self) -> Result<f32> {
        let v = self.to_vec()?;
        if v.len() != 1 {
            return Err(GlTrainError::InvalidOp(format!(
                "item() called on tensor with {} elements (expected 1)",
                v.len()
            )));
        }
        Ok(v[0])
    }

    // ── Ops (dispatch to backend) ─────────────────────────────────────────

    /// Matrix multiplication: self @ other
    /// self: [M, K], other: [K, N] → result: [M, N]
    pub fn matmul(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_matmul_shapes(&self.shape, &other.shape)?;
        let m = self.shape[0];
        let n = other.shape[other.shape.len() - 1];
        let storage = B::matmul(&self.storage, &other.storage, &self.shape, &other.shape)?;
        Ok(Tensor::from_storage(storage, vec![m, n]))
    }

    /// Transpose a 2D tensor: [M, N] → [N, M]
    pub fn transpose(&self) -> Result<Tensor<B>> {
        if self.shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "transpose() requires a 2D tensor, got shape {:?}",
                self.shape
            )));
        }
        let storage = B::transpose(&self.storage, &self.shape)?;
        Ok(Tensor::from_storage(
            storage,
            vec![self.shape[1], self.shape[0]],
        ))
    }

    /// Element-wise addition: self + other (shapes must match exactly)
    pub fn add(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::add(&self.storage, &other.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Element-wise subtraction: self - other
    pub fn sub(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::sub(&self.storage, &other.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Element-wise multiplication: self * other
    pub fn mul(&self, other: &Tensor<B>) -> Result<Tensor<B>> {
        check_same_shape(&self.shape, &other.shape)?;
        let n = self.n_elems();
        let storage = B::mul(&self.storage, &other.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Scale all elements by scalar: self * scalar
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::mul_scalar(&self.storage, scalar, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// ReLU activation: max(0, x)
    pub fn relu(&self) -> Result<Tensor<B>> {
        let n = self.n_elems();
        let storage = B::relu(&self.storage, n)?;
        Ok(Tensor::from_storage(storage, self.shape.clone()))
    }

    /// Sum all elements to a scalar.
    pub fn sum(&self) -> Result<f32> {
        B::sum(&self.storage)
    }

    /// Mean of all elements.
    pub fn mean(&self) -> Result<f32> {
        B::mean(&self.storage, self.n_elems())
    }
}

// ── Shape validation helpers ─────────────────────────────────────────────────

fn shape_to_n_elems(shape: &[usize]) -> Result<usize> {
    if shape.is_empty() {
        return Err(GlTrainError::InvalidOp("shape cannot be empty".into()));
    }
    Ok(shape.iter().product())
}

fn check_same_shape(a: &[usize], b: &[usize]) -> Result<()> {
    if a != b {
        return Err(GlTrainError::ShapeMismatch {
            expected: a.to_vec(),
            got: b.to_vec(),
        });
    }
    Ok(())
}

fn check_matmul_shapes(a: &[usize], b: &[usize]) -> Result<()> {
    // Support 2D only in Wave 1: [M, K] @ [K, N]
    if a.len() != 2 || b.len() != 2 {
        return Err(GlTrainError::InvalidOp(format!(
            "matmul requires 2D tensors in Wave 1, got shapes {a:?} and {b:?}"
        )));
    }
    if a[1] != b[0] {
        return Err(GlTrainError::ShapeMismatch {
            expected: vec![a[0], a[1], b[1]],
            got: vec![a[0], b[0], b[1]],
        });
    }
    Ok(())
}

// ── Debug ────────────────────────────────────────────────────────────────────

impl<B: Backend> std::fmt::Debug for Tensor<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor(shape={:?})", self.shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// Tolerance for f32 elementwise ops (rounding only, no accumulation).
    const TOL_ELEM: f32 = 1e-6;

    /// Tolerance for matmul (accumulation over K dimension can drift).
    const TOL_MATMUL: f32 = 1e-4;

    /// Relative tolerance for matmul at realistic K, as a fraction of the
    /// largest output magnitude. At K=64 an absolute bound no longer means
    /// anything — the error grows with the accumulation, so the bound must too.
    const TOL_MATMUL_REL: f32 = 1e-5;

    // ── Allocation ───────────────────────────────────────────────────────

    #[test]
    fn zeros_produces_correct_shape_and_values() {
        let t = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.n_elems(), 6);
        let v = t.to_vec().unwrap();
        assert!(
            v.iter().all(|&x| x == 0.0),
            "expected all zeros, got {v:?}"
        );
    }

    #[test]
    fn from_vec_roundtrip_preserves_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let t = Tensor::<GlProc>::from_vec(data.clone(), &[2, 2]).unwrap();
        assert_eq!(t.shape(), &[2, 2]);
        let out = t.to_vec().unwrap();
        for (a, b) in out.iter().zip(&data) {
            assert!((a - b).abs() < TOL_ELEM, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn from_vec_rejects_size_mismatch() {
        let result = Tensor::<GlProc>::from_vec(vec![1.0, 2.0], &[2, 2]);
        assert!(result.is_err(), "expected Err for size mismatch");
    }

    // ── Matmul ───────────────────────────────────────────────────────────

    #[test]
    fn matmul_2x2_by_2x2_produces_correct_result() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.to_vec().unwrap();
        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < TOL_MATMUL,
                "matmul mismatch: got {got}, expected {exp}"
            );
        }
    }

    /// A LoRA-shaped matmul: [16, 64] @ [64, 32] is the shape of a rank-ish
    /// projection on a batch of 16, and unlike the 2×2 case above it actually
    /// reaches glproc's AVX2 path — M, K and N are all whole multiples of the
    /// 8-float lane, so the SIMD body runs with no scalar tail to hide behind.
    ///
    /// Checked against an f64 accumulation of the same product computed here,
    /// so the reference does not share any code with the kernel under test.
    #[test]
    fn matmul_16x64_by_64x32_matches_f64_reference() {
        let (m, k, n) = (16usize, 64usize, 32usize);
        // Deterministic, mixed-sign, O(1) magnitude — no RNG, so a failure is
        // always reproducible.
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
            .collect();

        let got = Tensor::<GlProc>::from_vec(a.clone(), &[m, k])
            .unwrap()
            .matmul(&Tensor::<GlProc>::from_vec(b.clone(), &[k, n]).unwrap())
            .unwrap();
        assert_eq!(got.shape(), &[m, n]);
        let got = got.to_vec().unwrap();
        assert_eq!(got.len(), m * n);

        let mut expected = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for l in 0..k {
                    acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
                }
                expected[i * n + j] = acc;
            }
        }

        // K=64 accumulation makes an absolute bound meaningless, so the
        // tolerance scales with the largest output magnitude.
        let scale = expected.iter().fold(1.0f64, |acc, v| acc.max(v.abs()));
        for (idx, (got, exp)) in got.iter().zip(&expected).enumerate() {
            let err = (f64::from(*got) - exp).abs();
            assert!(
                err <= f64::from(TOL_MATMUL_REL) * scale,
                "matmul mismatch at {idx}: got {got}, expected {exp}, err {err:.3e}"
            );
        }
    }

    #[test]
    fn matmul_shape_mismatch_returns_error() {
        // [2, 3] @ [2, 2] is invalid (K mismatch: 3 ≠ 2)
        let a = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<GlProc>::zeros(&[2, 2]).unwrap();
        assert!(a.matmul(&b).is_err(), "expected Err for K mismatch");
    }

    // ── Transpose ────────────────────────────────────────────────────────

    #[test]
    fn transpose_2x3_produces_3x2() {
        // [[1, 2, 3], [4, 5, 6]] → [[1, 4], [2, 5], [3, 6]]
        let t =
            Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tr = t.transpose().unwrap();
        assert_eq!(tr.shape(), &[3, 2]);
        let v = tr.to_vec().unwrap();
        let expected = [1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < TOL_ELEM,
                "transpose mismatch: got {got}, expected {exp}"
            );
        }
    }

    // ── Elementwise ──────────────────────────────────────────────────────

    #[test]
    fn add_two_tensors_element_wise() {
        let a = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0], &[3]).unwrap();
        let b = Tensor::<GlProc>::from_vec(vec![4.0, 5.0, 6.0], &[3]).unwrap();
        let c = a.add(&b).unwrap();
        let v = c.to_vec().unwrap();
        let expected = [5.0f32, 7.0, 9.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    #[test]
    fn relu_zeroes_negative_values() {
        let t = Tensor::<GlProc>::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]).unwrap();
        let r = t.relu().unwrap();
        let v = r.to_vec().unwrap();
        let expected = [0.0f32, 0.0, 0.0, 1.0, 2.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    #[test]
    fn mul_scalar_scales_all_elements() {
        let t = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]).unwrap();
        let scaled = t.mul_scalar(2.5).unwrap();
        let v = scaled.to_vec().unwrap();
        let expected = [2.5f32, 5.0, 7.5, 10.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!((got - exp).abs() < TOL_ELEM, "{got} vs {exp}");
        }
    }

    // ── Reduction ────────────────────────────────────────────────────────

    #[test]
    fn sum_of_ones_tensor_equals_n_elems() {
        let t = Tensor::<GlProc>::ones(&[3, 4]).unwrap();
        let s = t.sum().unwrap();
        assert!((s - 12.0).abs() < TOL_ELEM, "sum={s}, expected 12.0");
    }
}
