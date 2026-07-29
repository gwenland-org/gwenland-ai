//! Token embedding lookup (ARTX03 §6).

use std::rc::Rc;

use crate::graph::value::SsaValue;
use crate::stablehlo::ops::{emit_gather, GatherDimensionNumbers};
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;

/// Looks up `[B, S]` token ids in a `[vocab, D]` table, giving `[B, S, D]`.
///
/// ⛔ The six gather dimension-number fields decide *which rows come back*, and
/// a wrong one returns a correctly-shaped tensor of the wrong embeddings. What
/// each one means here:
///
/// | field | value | why |
/// |---|---|---|
/// | `offset_dims` | `[1]` | output axis 1 carries the D-wide slice |
/// | `collapsed_slice_dims` | `[0]` | the vocab axis is size-1 per slice and dropped |
/// | `start_index_map` | `[0]` | each index addresses table axis 0 |
/// | `index_vector_dim` | `1` | indices are `[N, 1]`; axis 1 holds the 1-element vector |
/// | `slice_sizes` | `[1, D]` | one row, all D columns |
///
/// Indices are flattened to `[B·S, 1]` first, which keeps the dimension numbers
/// rank-independent — the alternative is a different `index_vector_dim` for
/// every input rank.
pub fn gather_embed(table: &Tensor, indices: &Tensor) -> Tensor {
    let (b, s) = match indices.shape().dims.as_slice() {
        &[b, s] => (b, s),
        other => panic!("gather_embed: indices must be rank-2 [B, S], got {other:?}"),
    };
    let (vocab, d) = match table.shape().dims.as_slice() {
        &[v, d] => (v, d),
        other => panic!("gather_embed: table must be rank-2 [vocab, D], got {other:?}"),
    };
    assert!(
        matches!(indices.dtype(), DType::I32 | DType::I64),
        "gather_embed: token ids must be an integer dtype, got {:?}",
        indices.dtype()
    );
    assert!(vocab > 0, "gather_embed: empty vocabulary");

    let flat = indices.reshape(vec![b * s, 1]);
    let out_shape = Shape::new([b * s, d], table.dtype());

    let dnums = GatherDimensionNumbers {
        offset_dims: vec![1],
        collapsed_slice_dims: vec![0],
        start_index_map: vec![0],
        index_vector_dim: 1,
        ..Default::default()
    };

    let name = {
        let mut builder = table.builder().borrow_mut();
        emit_gather(
            builder.emitter_mut(),
            table.value().ssa(),
            flat.value().ssa(),
            &dnums,
            &[1, d],
            table.shape(),
            flat.shape(),
            &out_shape,
        )
    };
    let gathered = Tensor::new(
        SsaValue::new(name, out_shape),
        Rc::clone(table.builder()),
    );
    gathered.reshape(vec![b, s, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::TraceCx;

    #[test]
    fn embedding_lookup_produces_batch_seq_hidden() {
        let mut cx = TraceCx::new("main", "embed");
        let table = cx.weight(
            "model.embed_tokens.weight",
            Shape::new([151936, 896], DType::F32),
        );
        let ids = cx.input("input_ids", Shape::new([1, 128], DType::I32));
        let out = gather_embed(&table, &ids);
        assert_eq!(out.shape().dims, vec![1, 128, 896]);

        let mlir = cx.finish(&[&out]).mlir;
        assert!(mlir.contains("offset_dims = [1],"), "{mlir}");
        assert!(mlir.contains("collapsed_slice_dims = [0],"), "{mlir}");
        assert!(mlir.contains("start_index_map = [0],"), "{mlir}");
        assert!(mlir.contains("index_vector_dim = 1"), "{mlir}");
        assert!(mlir.contains("slice_sizes = array<i64: 1, 896>,"), "{mlir}");
        assert!(
            mlir.contains("(tensor<151936x896xf32>, tensor<128x1xi32>) -> tensor<128x896xf32>"),
            "{mlir}"
        );
    }

    #[test]
    #[should_panic(expected = "token ids must be an integer dtype")]
    fn embedding_lookup_rejects_float_token_ids() {
        let mut cx = TraceCx::new("main", "embed");
        let table = cx.weight("t", Shape::new([16, 4], DType::F32));
        let ids = cx.input("ids", Shape::new([1, 2], DType::F32));
        let _ = gather_embed(&table, &ids);
    }
}
