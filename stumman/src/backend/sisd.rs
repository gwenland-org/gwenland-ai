//! Stummañ Karg — SISD backend (pure scalar reference).
//!
//! Storage: `Vec<f32>` (heap-allocated, row-major) — same layout as
//! [`crate::backend::GlProc`], so results are directly comparable.
//!
//! SISD = Single Instruction, Single Data: every op here is a plain scalar
//! loop with no SIMD, no threading, and no dispatch. This backend exists to be
//! *boring and obviously correct*, not fast.
//!
//! ## Why this duplicates GlProc instead of sharing code with it
//!
//! This is the numerical oracle the Wave 4 gradient check compares against. An
//! oracle that shares its implementation with the code under test cannot catch
//! a bug in that shared code — two mistakes that cancel look like agreement.
//! So the arithmetic below is written out independently on purpose. The
//! duplication is the feature; do not refactor these two backends onto a common
//! helper.
//!
//! ## Precision
//!
//! Accumulation is f32, matching GlProc — this is a *SIMD-vs-scalar* reference,
//! not a *higher-precision* reference. If Wave 4's finite-difference check turns
//! out to need more headroom than f32 accumulation leaves, the fix is a separate
//! f64-accumulating oracle, not a change here (that would stop this backend from
//! being a faithful scalar mirror).

use crate::error::{GlTrainError, Result};
use crate::tensor::backend::Backend;

/// Pure-scalar CPU backend. Reference implementation, no SIMD.
#[derive(Clone, Debug, Default)]
pub struct SisdBackend;

/// Reject a storage buffer that disagrees with the element count the caller
/// promised, so a mismatch is a named error rather than a truncated result or a
/// panic from inside a loop.
fn check_len(storage: &[f32], n_elems: usize, op: &str, operand: &str) -> Result<()> {
    if storage.len() != n_elems {
        return Err(GlTrainError::Backend(format!(
            "sisd {op}: {operand} storage has {} elements, expected {n_elems}",
            storage.len()
        )));
    }
    Ok(())
}

impl Backend for SisdBackend {
    type Storage = Vec<f32>;

    fn zeros(n_elems: usize) -> Result<Self::Storage> {
        Ok(vec![0.0f32; n_elems])
    }

    fn ones(n_elems: usize) -> Result<Self::Storage> {
        Ok(vec![1.0f32; n_elems])
    }

    fn from_vec(data: Vec<f32>) -> Result<Self::Storage> {
        Ok(data)
    }

    fn to_vec(storage: &Self::Storage) -> Result<Vec<f32>> {
        Ok(storage.clone())
    }

    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Result<Self::Storage> {
        if a_shape.len() != 2 || b_shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "sisd matmul requires 2D shapes, got {a_shape:?} and {b_shape:?}"
            )));
        }
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        if b_shape[0] != k {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![k, n],
                got: b_shape.to_vec(),
            });
        }
        check_len(a, m * k, "matmul", "lhs")?;
        check_len(b, k * n, "matmul", "rhs")?;

        // Textbook i-j-l triple loop, one multiply-add at a time. Deliberately
        // not tiled, not vectorised, not reordered for cache.
        let mut c = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        Ok(c)
    }

    fn transpose(a: &Self::Storage, shape: &[usize]) -> Result<Self::Storage> {
        if shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "sisd transpose requires a 2D shape, got {shape:?}"
            )));
        }
        let m = shape[0];
        let n = shape[1];
        check_len(a, m * n, "transpose", "input")?;

        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                out[j * m + i] = a[i * n + j];
            }
        }
        Ok(out)
    }

    fn add(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "add", "lhs")?;
        check_len(b, n_elems, "add", "rhs")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = a[i] + b[i];
        }
        Ok(out)
    }

    fn sub(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sub", "lhs")?;
        check_len(b, n_elems, "sub", "rhs")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = a[i] - b[i];
        }
        Ok(out)
    }

    fn mul(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "mul", "lhs")?;
        check_len(b, n_elems, "mul", "rhs")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = a[i] * b[i];
        }
        Ok(out)
    }

    fn mul_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "mul_scalar", "input")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = a[i] * scalar;
        }
        Ok(out)
    }

    fn relu(x: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(x, n_elems, "relu", "input")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = if x[i] > 0.0 { x[i] } else { 0.0 };
        }
        Ok(out)
    }

    fn sum(a: &Self::Storage) -> Result<f32> {
        let mut acc = 0.0f32;
        for &v in a.iter() {
            acc += v;
        }
        Ok(acc)
    }

    fn mean(a: &Self::Storage, n_elems: usize) -> Result<f32> {
        if n_elems == 0 {
            return Err(GlTrainError::InvalidOp("sisd mean of empty tensor".into()));
        }
        check_len(a, n_elems, "mean", "input")?;
        let mut acc = 0.0f32;
        for &v in a.iter() {
            acc += v;
        }
        Ok(acc / n_elems as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;
    use crate::tensor::Tensor;

    /// Tolerance for f32 elementwise ops (rounding only, no accumulation).
    const TOL_ELEM: f32 = 1e-6;

    /// Relative tolerance for matmul, as a fraction of the largest output
    /// magnitude. Accumulation over K makes an absolute bound meaningless once
    /// K grows, so the bound scales with the result.
    const TOL_MATMUL_REL: f32 = 1e-5;

    #[test]
    fn zeros_produces_correct_shape_and_values() {
        let t = Tensor::<SisdBackend>::zeros(&[2, 3]).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.n_elems(), 6);
        let v = t.to_vec().unwrap();
        assert!(v.iter().all(|&x| x == 0.0), "expected all zeros, got {v:?}");
    }

    #[test]
    fn matmul_2x2_by_2x2_produces_correct_result() {
        // A = [[1, 2], [3, 4]]
        // B = [[5, 6], [7, 8]]
        // C = A @ B = [[19, 22], [43, 50]]
        let a = Tensor::<SisdBackend>::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::<SisdBackend>::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let v = c.to_vec().unwrap();
        let expected = [19.0f32, 22.0, 43.0, 50.0];
        for (got, &exp) in v.iter().zip(&expected) {
            assert!(
                (got - exp).abs() < TOL_ELEM,
                "sisd matmul mismatch: got {got}, expected {exp}"
            );
        }
    }

    /// The reason this backend exists: GlProc's SIMD-dispatched kernel and this
    /// scalar reference must agree. Shape [16,64] @ [64,32] straddles the AVX2
    /// 8-float lane in both M and N with no ragged tail, and K=64 gives the
    /// accumulator enough depth for a reordered SIMD sum to diverge if the
    /// wiring were wrong.
    #[test]
    fn glproc_matches_sisd_reference_on_wide_shape() {
        let (m, k, n) = (16usize, 64usize, 32usize);
        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
            .collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i * 53 % 97) as f32 - 48.0) / 48.0)
            .collect();

        let simd = Tensor::<GlProc>::from_vec(a.clone(), &[m, k])
            .unwrap()
            .matmul(&Tensor::<GlProc>::from_vec(b.clone(), &[k, n]).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();
        let scalar = Tensor::<SisdBackend>::from_vec(a, &[m, k])
            .unwrap()
            .matmul(&Tensor::<SisdBackend>::from_vec(b, &[k, n]).unwrap())
            .unwrap()
            .to_vec()
            .unwrap();

        assert_eq!(simd.len(), scalar.len(), "backends disagree on output size");
        let scale = scalar.iter().fold(1.0f32, |acc, v| acc.max(v.abs()));
        for (idx, (got, exp)) in simd.iter().zip(&scalar).enumerate() {
            assert!(
                (got - exp).abs() <= TOL_MATMUL_REL * scale,
                "glproc/sisd divergence at {idx}: simd={got}, scalar={exp}"
            );
        }
    }
}
