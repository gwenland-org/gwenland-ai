//! Small shared helpers for the ops layer.
//!
//! These exist because ARTX03's sketches repeat the same three-line
//! `borrow_mut / emit / drop` dance at every call site, which is both noisy and
//! the kind of thing where one forgotten `drop` becomes a `RefCell` panic at
//! trace time.

use std::rc::Rc;

use crate::graph::value::SsaValue;
use crate::stablehlo::ops;
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;
use crate::GlError;

/// Emits a rank-0 constant into the same trace as `like`.
pub(crate) fn scalar_const(like: &Tensor, value: f64, dtype: DType) -> Tensor {
    let v = like.builder().borrow_mut().constant_scalar(value, dtype);
    Tensor::new(v, Rc::clone(like.builder()))
}

/// Emits a dense `f32` constant into the same trace as `like`.
pub(crate) fn dense_const_f32(
    like: &Tensor,
    data: &[f32],
    shape: Shape,
) -> Result<Tensor, GlError> {
    let name = {
        let mut b = like.builder().borrow_mut();
        ops::emit_constant_dense_f32(b.emitter_mut(), data, &shape)?
    };
    Ok(Tensor::new(
        SsaValue::new(name, shape),
        Rc::clone(like.builder()),
    ))
}

/// A scalar broadcast to `dims` — the shape every elementwise op needs, since
/// StableHLO elementwise ops do not broadcast on their own.
pub(crate) fn splat_like(like: &Tensor, value: f64, dims: Vec<usize>, dtype: DType) -> Tensor {
    let s = scalar_const(like, value, dtype);
    s.broadcast_to(vec![], dims)
}

/// Sum over `dim`, keeping it as a size-1 axis.
///
/// `stablehlo.reduce` always drops the reduced axes, so "keepdim" is a reduce
/// followed by a reshape. Keeping the axis is what makes the subsequent
/// broadcast-back unambiguous.
pub(crate) fn reduce_sum_keepdim(x: &Tensor, dim: usize) -> Tensor {
    let init = scalar_const(x, 0.0, x.dtype());
    let reduced = {
        let v = x
            .builder()
            .borrow_mut()
            .reduce_add(x.value(), init.value(), &[dim]);
        Tensor::new(v, Rc::clone(x.builder()))
    };
    reinsert_axis(&reduced, x, dim)
}

/// Maximum over `dim`, keeping it as a size-1 axis.
///
/// The init is −∞: any finite init would clamp the result from below, which for
/// softmax's subtract-max step means a silently wrong distribution rather than
/// an error.
pub(crate) fn reduce_max_keepdim(x: &Tensor, dim: usize) -> Tensor {
    let init = scalar_const(x, f64::NEG_INFINITY, x.dtype());
    let reduced = {
        let v = x
            .builder()
            .borrow_mut()
            .reduce_max(x.value(), init.value(), &[dim]);
        Tensor::new(v, Rc::clone(x.builder()))
    };
    reinsert_axis(&reduced, x, dim)
}

/// Reshapes `reduced` back to `original`'s rank with `dim` restored at size 1.
fn reinsert_axis(reduced: &Tensor, original: &Tensor, dim: usize) -> Tensor {
    let mut dims = original.shape().dims.clone();
    dims[dim] = 1;
    reduced.reshape(dims)
}

/// Broadcasts a keepdim tensor back to `target`'s shape.
///
/// The mapping is the identity — every axis of a keepdim tensor lines up with
/// the same axis of the target, and the size-1 one expands. Deriving the
/// mapping from "which dims are not 1" (as ARTX03 §1 does) silently drops any
/// axis that is genuinely of size 1 in the target, e.g. batch size 1.
pub(crate) fn broadcast_keepdim_to(src: &Tensor, target_dims: Vec<usize>) -> Tensor {
    let identity: Vec<usize> = (0..src.rank()).collect();
    src.broadcast_to(identity, target_dims)
}

/// Unpacks a rank-4 shape, naming the op that expected it.
pub(crate) fn expect_rank4(x: &Tensor, what: &str) -> [usize; 4] {
    match x.shape().dims.as_slice() {
        &[b, h, s, d] => [b, h, s, d],
        other => panic!("{what}: expected rank-4 [B, H, S, D], got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    #[test]
    fn keepdim_reduce_preserves_rank() {
        let mut cx = TraceCx::new("main", "t");
        let x = cx.input("x", Shape::new([2, 3, 4], DType::F32));
        let s = reduce_sum_keepdim(&x, 2);
        assert_eq!(s.shape().dims, vec![2, 3, 1]);
        let m = reduce_max_keepdim(&x, 1);
        assert_eq!(m.shape().dims, vec![2, 1, 4]);
    }

    /// ⛔ ARTX03 §1's `broadcast_like` derives the mapping from which source
    /// dims are not 1. With batch size 1 — the only batch size gljax v1 ever
    /// uses — that drops axis 0 from the mapping and the broadcast is wrong.
    #[test]
    fn broadcast_back_works_when_a_real_axis_also_has_size_one() {
        let mut cx = TraceCx::new("main", "t");
        let x = cx.input("x", Shape::new([1, 16, 512, 512], DType::F32));
        let m = reduce_max_keepdim(&x, 3);
        assert_eq!(m.shape().dims, vec![1, 16, 512, 1]);
        let back = broadcast_keepdim_to(&m, vec![1, 16, 512, 512]);
        assert_eq!(back.shape().dims, vec![1, 16, 512, 512]);
        let built = cx.finish(&[&back]);
        assert!(
            built
                .mlir
                .contains("broadcast_dimensions = array<i64: 0, 1, 2, 3>"),
            "the mapping must be the identity, not just the non-1 axes:\n{}",
            built.mlir
        );
    }
}
