//! Stummañ Karg — GlProc backend (CPU / AVX2).
//!
//! Storage: `Vec<f32>` (heap-allocated, row-major).
//! Dispatch: `matmul` calls glproc's SIMD-dispatched f32 kernel
//! ([`glproc::kernels::matmul()`], AVX-512 → AVX2 → scalar by runtime CPU
//! probe); elementwise ops are scalar iterator loops.
//!
//! No unsafe code in this file — unsafe lives inside glproc's own kernels.

use crate::error::{GlTrainError, Result};
use crate::tensor::backend::Backend;

/// CPU backend using glproc AVX2 kernels.
#[derive(Clone, Debug, Default)]
pub struct GlProc;

/// Reject a storage buffer that disagrees with the element count the caller
/// promised. Without this the ops below would silently truncate (iterator
/// adaptors) or panic deep inside a kernel; both are worse than an error that
/// names the operand.
fn check_len(storage: &[f32], n_elems: usize, op: &str, operand: &str) -> Result<()> {
    if storage.len() != n_elems {
        return Err(GlTrainError::Backend(format!(
            "{op}: {operand} storage has {} elements, expected {n_elems}",
            storage.len()
        )));
    }
    Ok(())
}

impl Backend for GlProc {
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
                "matmul requires 2D shapes, got {a_shape:?} and {b_shape:?}"
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

        // glproc's dispatcher picks AVX-512 / AVX2 / scalar from a cached CPUID
        // probe and writes C = A @ B row-major, [M,K] @ [K,N] -> [M,N] — the
        // exact contract of this trait method, so no shim is needed.
        let mut c = vec![0.0f32; m * n];
        glproc::kernels::matmul(a, b, &mut c, m, k, n);
        Ok(c)
    }

    fn transpose(a: &Self::Storage, shape: &[usize]) -> Result<Self::Storage> {
        if shape.len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "transpose requires a 2D shape, got {shape:?}"
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
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x + y).collect())
    }

    fn sub(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sub", "lhs")?;
        check_len(b, n_elems, "sub", "rhs")?;
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x - y).collect())
    }

    fn mul(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "mul", "lhs")?;
        check_len(b, n_elems, "mul", "rhs")?;
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
    }

    fn mul_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "mul_scalar", "input")?;
        Ok(a.iter().map(|x| x * scalar).collect())
    }

    fn div(a: &Self::Storage, b: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "div", "lhs")?;
        check_len(b, n_elems, "div", "rhs")?;
        // A zero divisor is a caller bug, not a number. Returning inf here would
        // surface three optimizer steps later as a NaN weight with no trace back
        // to this call.
        if let Some(idx) = b.iter().position(|y| *y == 0.0) {
            return Err(GlTrainError::Backend(format!(
                "div: divisor is zero at index {idx}"
            )));
        }
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x / y).collect())
    }

    fn sqrt(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sqrt", "input")?;
        if let Some(idx) = a.iter().position(|x| *x < 0.0) {
            return Err(GlTrainError::Backend(format!(
                "sqrt: negative input {} at index {idx}",
                a[idx]
            )));
        }
        Ok(a.iter().map(|x| x.sqrt()).collect())
    }

    fn add_scalar(a: &Self::Storage, scalar: f32, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "add_scalar", "input")?;
        Ok(a.iter().map(|x| x + scalar).collect())
    }

    fn neg(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "neg", "input")?;
        Ok(a.iter().map(|x| -x).collect())
    }

    fn sign(a: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(a, n_elems, "sign", "input")?;
        // Not `f32::signum`: it maps 0.0 to +1.0 and -0.0 to -1.0. Lion would
        // then move a parameter with zero momentum by a full step.
        Ok(a
            .iter()
            .map(|x| {
                if *x > 0.0 {
                    1.0
                } else if *x < 0.0 {
                    -1.0
                } else {
                    0.0
                }
            })
            .collect())
    }

    fn relu(x: &Self::Storage, n_elems: usize) -> Result<Self::Storage> {
        check_len(x, n_elems, "relu", "input")?;
        Ok(x.iter().map(|v| v.max(0.0)).collect())
    }

    fn sum(a: &Self::Storage) -> Result<f32> {
        Ok(a.iter().sum())
    }

    fn mean(a: &Self::Storage, n_elems: usize) -> Result<f32> {
        if n_elems == 0 {
            return Err(GlTrainError::InvalidOp("mean of empty tensor".into()));
        }
        check_len(a, n_elems, "mean", "input")?;
        Ok(a.iter().sum::<f32>() / n_elems as f32)
    }
}
