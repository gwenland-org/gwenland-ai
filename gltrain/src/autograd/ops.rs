//! Stummañ Oberour: pure math helpers for the backward pass.
//!
//! These operate directly on `&[f32]` with no `Backend` involved. Backward
//! closures are backend-agnostic by design (see
//! [`crate::autograd::node::BackwardFn`]), so they cannot dispatch through
//! `B::matmul`. That costs speed and buys the property that `Tape` never
//! becomes generic.
//!
//! Wave 4 can revisit this: the closures could capture a backend-specific
//! function pointer at record time and keep the tape agnostic anyway.

use crate::error::{GlTrainError, Result};

/// Naive f32 matmul: `a` is `[M,K]`, `b` is `[K,N]`, result is `[M,N]`,
/// row-major throughout.
///
/// i-l-j loop order so the inner loop walks `b` and `c` contiguously.
pub fn matmul_f32(
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
) -> Result<Vec<f32>> {
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(GlTrainError::InvalidOp(format!(
            "matmul_f32 requires 2D shapes, got {a_shape:?} and {b_shape:?}"
        )));
    }
    let (m, k, n) = (a_shape[0], a_shape[1], b_shape[1]);
    if b_shape[0] != k {
        return Err(GlTrainError::ShapeMismatch {
            expected: vec![k, n],
            got: b_shape.to_vec(),
        });
    }
    if a.len() != m * k {
        return Err(GlTrainError::Backend(format!(
            "matmul_f32: lhs has {} elements, expected {}",
            a.len(),
            m * k
        )));
    }
    if b.len() != k * n {
        return Err(GlTrainError::Backend(format!(
            "matmul_f32: rhs has {} elements, expected {}",
            b.len(),
            k * n
        )));
    }

    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for l in 0..k {
            let a_il = a[i * k + l];
            let b_row = &b[l * n..(l + 1) * n];
            let c_row = &mut c[i * n..(i + 1) * n];
            for (c_ij, &b_lj) in c_row.iter_mut().zip(b_row) {
                *c_ij += a_il * b_lj;
            }
        }
    }
    Ok(c)
}

/// Transpose a 2D matrix: `shape` is `[M,N]`, result is `[N,M]`.
///
/// The shape must be rank 2 and must match `a`'s length. An earlier version
/// returned the input untouched on a mismatch, which would have handed a
/// backward pass a silently untransposed buffer of the right size. Checked and
/// returned as an error instead of asserted, because a panic inside a backward
/// closure kills a training run outright.
pub fn transpose_2d(a: &[f32], shape: &[usize]) -> Result<Vec<f32>> {
    if shape.len() != 2 {
        return Err(GlTrainError::InvalidOp(format!(
            "transpose_2d requires a 2D shape, got {shape:?}"
        )));
    }
    let (m, n) = (shape[0], shape[1]);
    if a.len() != m * n {
        return Err(GlTrainError::ShapeMismatch {
            expected: vec![m, n],
            got: vec![a.len()],
        });
    }

    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[j * m + i] = a[i * n + j];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact small integers here, so anything above rounding is a real bug.
    const TOL: f32 = 1e-6;

    #[test]
    fn matmul_f32_matches_hand_computed_2x2() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let c = matmul_f32(
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            &[5.0, 6.0, 7.0, 8.0],
            &[2, 2],
        )
        .unwrap();
        for (got, want) in c.iter().zip(&[19.0f32, 22.0, 43.0, 50.0]) {
            assert!((got - want).abs() < TOL, "got {got}, want {want}");
        }
    }

    /// Non-square catches an index-order bug that a square case hides.
    #[test]
    fn matmul_f32_handles_non_square() {
        // [2,3] @ [3,2] -> [2,2]
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let c = matmul_f32(&a, &[2, 3], &b, &[3, 2]).unwrap();
        assert_eq!(c.len(), 4);
        // row0 = [1,2,3] . cols of [[1,2],[3,4],[5,6]] = [22, 28]
        // row1 = [4,5,6] . same                        = [49, 64]
        for (got, want) in c.iter().zip(&[22.0f32, 28.0, 49.0, 64.0]) {
            assert!((got - want).abs() < TOL, "got {got}, want {want}");
        }
    }

    #[test]
    fn matmul_f32_rejects_inner_dim_mismatch() {
        let r = matmul_f32(&[1.0, 2.0], &[1, 2], &[1.0, 2.0, 3.0], &[3, 1]);
        assert!(r.is_err(), "K mismatch must be rejected");
    }

    #[test]
    fn transpose_2d_swaps_rows_and_cols() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let t = transpose_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        for (got, want) in t.iter().zip(&[1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]) {
            assert!((got - want).abs() < TOL, "got {got}, want {want}");
        }
    }

    #[test]
    fn transpose_2d_twice_is_identity() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let once = transpose_2d(&a, &[2, 3]).unwrap();
        let twice = transpose_2d(&once, &[3, 2]).unwrap();
        for (got, want) in twice.iter().zip(&a) {
            assert!((got - want).abs() < TOL);
        }
    }

    /// The case the old no-op version would have swallowed: a shape whose
    /// element count does not match the buffer.
    #[test]
    fn transpose_2d_rejects_length_mismatch() {
        let r = transpose_2d(&[1.0, 2.0, 3.0], &[2, 2]);
        assert!(r.is_err(), "4-element shape over a 3-element buffer must fail");
    }

    #[test]
    fn transpose_2d_rejects_non_2d_shape() {
        assert!(transpose_2d(&[1.0, 2.0], &[2]).is_err());
        assert!(transpose_2d(&[1.0, 2.0], &[1, 2, 1]).is_err());
    }
}
