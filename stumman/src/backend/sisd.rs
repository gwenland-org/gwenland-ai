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

    fn div(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "div", "lhs")?;
        check_len(b, n_elems, "div", "rhs")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            if b[i] == 0.0 {
                return Err(GlTrainError::Backend(format!(
                    "sisd div: divisor is zero at index {i}"
                )));
            }
            out[i] = a[i] / b[i];
        }
        Ok(out)
    }

    fn sqrt(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sqrt", "input")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            if a[i] < 0.0 {
                return Err(GlTrainError::Backend(format!(
                    "sisd sqrt: negative input {} at index {i}",
                    a[i]
                )));
            }
            out[i] = a[i].sqrt();
        }
        Ok(out)
    }

    fn add_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "add_scalar", "input")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = a[i] + scalar;
        }
        Ok(out)
    }

    fn neg(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "neg", "input")?;
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = -a[i];
        }
        Ok(out)
    }

    fn sign(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sign", "input")?;
        // Deliberately not `f32::signum`, which returns +1.0 for 0.0.
        let mut out = vec![0.0f32; n_elems];
        for i in 0..n_elems {
            out[i] = if a[i] > 0.0 {
                1.0
            } else if a[i] < 0.0 {
                -1.0
            } else {
                0.0
            };
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

    // ── M2 Wave 1: the optimizer's arithmetic ops ────────────────────────
    //
    // `div`, `sqrt` and `add_scalar` were added to `Backend` for OPAdamW
    // (M2_RESEARCH.md R5). They shipped without tests; these are them. Each
    // runs against BOTH backends, because a divergence between the SIMD path
    // and the scalar oracle is exactly what this file exists to catch.

    #[test]
    fn div_matches_elementwise_division_on_both_backends() {
        let a = vec![1.0f32, -6.0, 7.5, 0.0];
        let b = vec![2.0f32, 3.0, -2.5, 4.0];
        let expected = [0.5f32, -2.0, -3.0, 0.0];

        let sisd = SisdBackend::div(&a, &b, 4).unwrap();
        let glp = GlProc::div(&a, &b, 4).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            assert!(
                (sisd[i] - exp).abs() < TOL_ELEM,
                "sisd div[{i}] = {}, expected {exp}",
                sisd[i]
            );
            assert!(
                (glp[i] - exp).abs() < TOL_ELEM,
                "glproc div[{i}] = {}, expected {exp}",
                glp[i]
            );
        }
    }

    /// A zero divisor is a configuration bug (AdamW's denominator is
    /// `sqrt(v) + eps`, zero only if eps was zero). Both backends must name it
    /// rather than hand back an infinity that becomes a NaN weight later.
    #[test]
    fn div_rejects_a_zero_divisor_on_both_backends() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 0.0];
        assert!(SisdBackend::div(&a, &b, 2).is_err(), "sisd accepted 1/0");
        assert!(GlProc::div(&a, &b, 2).is_err(), "glproc accepted 1/0");
    }

    #[test]
    fn sqrt_matches_elementwise_square_root_on_both_backends() {
        let a = vec![0.0f32, 1.0, 4.0, 2.25];
        let expected = [0.0f32, 1.0, 2.0, 1.5];

        let sisd = SisdBackend::sqrt(&a, 4).unwrap();
        let glp = GlProc::sqrt(&a, 4).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            assert!((sisd[i] - exp).abs() < TOL_ELEM, "sisd sqrt[{i}]");
            assert!((glp[i] - exp).abs() < TOL_ELEM, "glproc sqrt[{i}]");
        }
    }

    /// A negative second moment means the optimizer state is already corrupt.
    /// Returning NaN would let that corruption travel silently.
    #[test]
    fn sqrt_rejects_a_negative_input_on_both_backends() {
        let a = vec![1.0f32, -1e-9];
        assert!(SisdBackend::sqrt(&a, 2).is_err(), "sisd accepted sqrt(-x)");
        assert!(GlProc::sqrt(&a, 2).is_err(), "glproc accepted sqrt(-x)");
        // And the failure must be an error, never a NaN that passes as a value.
        let got = GlProc::sqrt(&a, 2);
        assert!(matches!(got, Err(GlTrainError::Backend(_))));
    }

    #[test]
    fn add_scalar_offsets_every_element_on_both_backends() {
        let a = vec![1.0f32, -2.0, 0.0];
        let expected = [1.5f32, -1.5, 0.5];

        let sisd = SisdBackend::add_scalar(&a, 0.5, 3).unwrap();
        let glp = GlProc::add_scalar(&a, 0.5, 3).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            assert!((sisd[i] - exp).abs() < TOL_ELEM, "sisd add_scalar[{i}]");
            assert!((glp[i] - exp).abs() < TOL_ELEM, "glproc add_scalar[{i}]");
        }
    }

    #[test]
    fn neg_flips_the_sign_of_every_element_on_both_backends() {
        let a = vec![1.0f32, -2.5, 0.0];
        let expected = [-1.0f32, 2.5, 0.0];

        let sisd = SisdBackend::neg(&a, 3).unwrap();
        let glp = GlProc::neg(&a, 3).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            assert!((sisd[i] - exp).abs() < TOL_ELEM, "sisd neg[{i}]");
            assert!((glp[i] - exp).abs() < TOL_ELEM, "glproc neg[{i}]");
        }
    }

    /// Zero must map to zero, not to +1.0. `f32::signum` returns +1.0 for 0.0
    /// and -1.0 for -0.0, which would give a Lion parameter with no momentum a
    /// full-size step in a direction decided by a sign bit.
    #[test]
    fn sign_maps_zero_to_zero_not_to_one() {
        let a = vec![3.0f32, -3.0, 0.0, -0.0];
        let expected = [1.0f32, -1.0, 0.0, 0.0];

        let sisd = SisdBackend::sign(&a, 4).unwrap();
        let glp = GlProc::sign(&a, 4).unwrap();
        for (i, &exp) in expected.iter().enumerate() {
            assert!((sisd[i] - exp).abs() < TOL_ELEM, "sisd sign[{i}] = {}", sisd[i]);
            assert!((glp[i] - exp).abs() < TOL_ELEM, "glproc sign[{i}] = {}", glp[i]);
        }
    }

    /// Every op on the trait rejects a storage buffer that disagrees with the
    /// element count the caller promised, rather than truncating.
    #[test]
    fn the_new_ops_reject_a_length_mismatch() {
        let a = vec![1.0f32, 2.0];
        let b = vec![1.0f32, 2.0];
        assert!(SisdBackend::div(&a, &b, 3).is_err());
        assert!(SisdBackend::sqrt(&a, 3).is_err());
        assert!(SisdBackend::add_scalar(&a, 1.0, 3).is_err());
        assert!(SisdBackend::neg(&a, 3).is_err());
        assert!(SisdBackend::sign(&a, 3).is_err());
    }
}
