//! T0 — the pure-Rust f64 reference (ARTX12 §2.1).
//!
//! ⛔ **Do not optimize this file.** Its only virtues are that it is obviously
//! correct by inspection and has no dependencies. The moment this uses FMA,
//! blocking, or parallel reduction it starts sharing failure modes with the
//! thing it exists to validate. A reference that is slow is fine — it only
//! ever runs on the small shapes a unit test needs.
//!
//! This mirrors the StableHLO spec's own definition of `dot_general` (output
//! dims = batch dims, then lhs free dims, then rhs free dims) directly, not
//! gljax's `FuncBuilder`/`ops` implementation of it — an oracle that shares
//! code with the thing it checks is not an oracle.

use crate::stablehlo::ops::DotDimensionNumbers;

/// A dense row-major f64 tensor. Exists only for this module — nothing
/// outside `oracle` needs a runtime tensor type, since gljax otherwise never
/// materializes numbers (see `tensor/mod.rs`'s "handle, not a buffer" docs).
#[derive(Debug, Clone, PartialEq)]
pub struct TensorF64 {
    pub dims: Vec<usize>,
    data: Vec<f64>,
}

impl TensorF64 {
    pub fn zeros(dims: &[usize]) -> Self {
        let n: usize = dims.iter().product();
        TensorF64 { dims: dims.to_vec(), data: vec![0.0; n] }
    }

    pub fn from_data(dims: &[usize], data: Vec<f64>) -> Self {
        assert_eq!(
            data.len(),
            dims.iter().product::<usize>(),
            "TensorF64::from_data: {} elements does not match dims {dims:?}",
            data.len()
        );
        TensorF64 { dims: dims.to_vec(), data }
    }

    pub fn at(&self, idx: &[usize]) -> f64 {
        self.data[flat_index(&self.dims, idx)]
    }

    pub fn set(&mut self, idx: &[usize], value: f64) {
        let i = flat_index(&self.dims, idx);
        self.data[i] = value;
    }

    pub fn data(&self) -> &[f64] {
        &self.data
    }
}

/// Row-major flat offset: incrementally `flat = flat * dim + index`, which is
/// the standard "most-significant axis first" stride computation without
/// materializing strides.
fn flat_index(dims: &[usize], idx: &[usize]) -> usize {
    assert_eq!(idx.len(), dims.len(), "index rank {} != tensor rank {}", idx.len(), dims.len());
    let mut flat = 0usize;
    for (&i, &d) in idx.iter().zip(dims) {
        assert!(i < d, "index {i} out of bounds for dim {d}");
        flat = flat * d + i;
    }
    flat
}

/// Every multi-index over `dims`, in row-major order. Materialized (not
/// lazy) — acceptable because T0 only ever runs on small shapes (ARTX12
/// §2.1's own stated constraint).
fn index_space(dims: &[usize]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new()];
    for &d in dims {
        let mut next = Vec::with_capacity(out.len() * d.max(1));
        for idx in &out {
            for i in 0..d {
                let mut v = idx.clone();
                v.push(i);
                next.push(v);
            }
        }
        out = next;
    }
    out
}

/// Where batching/contracting axes place a multi-index's components inside a
/// full-rank index: `batch_idx` at the `batching` positions, `contract_idx`
/// at the `contracting` positions, `free_idx` at whatever positions are left
/// (in ascending order — the same rule `infer_dot_general_shape` uses for
/// which axes are "free").
fn scatter_index(
    rank: usize,
    batching: &[usize],
    contracting: &[usize],
    batch_idx: &[usize],
    contract_idx: &[usize],
    free_idx: &[usize],
) -> Vec<usize> {
    let mut idx = vec![0usize; rank];
    for (i, &p) in batching.iter().enumerate() {
        idx[p] = batch_idx[i];
    }
    for (i, &p) in contracting.iter().enumerate() {
        idx[p] = contract_idx[i];
    }
    let free_positions: Vec<usize> =
        (0..rank).filter(|p| !batching.contains(p) && !contracting.contains(p)).collect();
    for (i, &p) in free_positions.iter().enumerate() {
        idx[p] = free_idx[i];
    }
    idx
}

/// Reference `dot_general` in f64: the naive triple(-plus-batch) loop,
/// sequential accumulation, no reassociation. Mirrors the StableHLO spec
/// directly — output dims = batch dims, then lhs free dims, then rhs free
/// dims — the same convention `infer_dot_general_shape`
/// (`graph/builder.rs`) implements, checked independently here.
pub fn dot_general_f64(lhs: &TensorF64, rhs: &TensorF64, dnums: &DotDimensionNumbers) -> TensorF64 {
    assert_eq!(dnums.lhs_batching.len(), dnums.rhs_batching.len());
    assert_eq!(dnums.lhs_contracting.len(), dnums.rhs_contracting.len());
    for (&li, &ri) in dnums.lhs_batching.iter().zip(&dnums.rhs_batching) {
        assert_eq!(lhs.dims[li], rhs.dims[ri], "batch dim size mismatch");
    }
    for (&li, &ri) in dnums.lhs_contracting.iter().zip(&dnums.rhs_contracting) {
        assert_eq!(lhs.dims[li], rhs.dims[ri], "contracting dim size mismatch");
    }

    let batch_dims: Vec<usize> = dnums.lhs_batching.iter().map(|&i| lhs.dims[i]).collect();
    let contract_dims: Vec<usize> = dnums.lhs_contracting.iter().map(|&i| lhs.dims[i]).collect();
    let lhs_free_dims: Vec<usize> = (0..lhs.dims.len())
        .filter(|p| !dnums.lhs_batching.contains(p) && !dnums.lhs_contracting.contains(p))
        .map(|p| lhs.dims[p])
        .collect();
    let rhs_free_dims: Vec<usize> = (0..rhs.dims.len())
        .filter(|p| !dnums.rhs_batching.contains(p) && !dnums.rhs_contracting.contains(p))
        .map(|p| rhs.dims[p])
        .collect();

    let out_dims: Vec<usize> =
        batch_dims.iter().chain(&lhs_free_dims).chain(&rhs_free_dims).copied().collect();
    let mut out = TensorF64::zeros(&out_dims);

    for b in index_space(&batch_dims) {
        for m in index_space(&lhs_free_dims) {
            for n in index_space(&rhs_free_dims) {
                let mut acc = 0.0f64;
                for k in index_space(&contract_dims) {
                    let lhs_idx =
                        scatter_index(lhs.dims.len(), &dnums.lhs_batching, &dnums.lhs_contracting, &b, &k, &m);
                    let rhs_idx =
                        scatter_index(rhs.dims.len(), &dnums.rhs_batching, &dnums.rhs_contracting, &b, &k, &n);
                    acc += lhs.at(&lhs_idx) * rhs.at(&rhs_idx);
                }
                let out_idx: Vec<usize> = b.iter().chain(&m).chain(&n).copied().collect();
                out.set(&out_idx, acc);
            }
        }
    }
    out
}

/// Reference RMSNorm in f64: `x / sqrt(mean(x^2) + eps) * weight`. ε inside
/// the sqrt — see `ops::norm::rms_norm`'s doc comment for why the placement
/// is load-bearing, not stylistic.
pub fn rms_norm_f64(x: &[f64], weight: &[f64], eps: f64) -> Vec<f64> {
    assert_eq!(x.len(), weight.len());
    let d = x.len() as f64;
    let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / d;
    let scale = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(weight).map(|(v, w)| v * scale * w).collect()
}

/// Reference softmax in f64: subtract the row max before exponentiating (the
/// numerically-stable form `ops::softmax::softmax` also implements).
pub fn softmax_f64(x: &[f64]) -> Vec<f64> {
    let max = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dnums(lb: &[usize], rb: &[usize], lc: &[usize], rc: &[usize]) -> DotDimensionNumbers {
        DotDimensionNumbers {
            lhs_batching: lb.to_vec(),
            rhs_batching: rb.to_vec(),
            lhs_contracting: lc.to_vec(),
            rhs_contracting: rc.to_vec(),
        }
    }

    #[test]
    fn dot_general_f64_matches_ordinary_2x2_matmul() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let lhs = TensorF64::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = TensorF64::from_data(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let out = dot_general_f64(&lhs, &rhs, &dnums(&[], &[], &[1], &[0]));
        assert_eq!(out.dims, vec![2, 2]);
        assert_eq!(out.data(), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn dot_general_f64_batches_leading_axes() {
        // Two independent 1x1 @ 1x1 "batches": batch 0 -> 2*3=6, batch 1 -> 4*5=20.
        let lhs = TensorF64::from_data(&[2, 1, 1], vec![2.0, 4.0]);
        let rhs = TensorF64::from_data(&[2, 1, 1], vec![3.0, 5.0]);
        let out = dot_general_f64(&lhs, &rhs, &dnums(&[0], &[0], &[2], &[1]));
        assert_eq!(out.dims, vec![2, 1, 1]);
        assert_eq!(out.data(), &[6.0, 20.0]);
    }

    /// ⭐ The whole reason T0 exists as a *second* implementation: contracting
    /// the wrong axis of a non-square operand is shape-invalid here (it would
    /// panic on a dim mismatch), but on a square operand it is shape-**valid**
    /// and numerically wrong — `DotDimensionNumbers`'s own doc comment in
    /// `stablehlo/ops.rs` names exactly this as the first structural bug this
    /// document exists to catch. This test proves the two contractions give
    /// different numbers on a case an assert-based shape check cannot flag.
    #[test]
    fn transposed_contracting_dims_on_a_square_operand_silently_changes_the_answer() {
        let lhs = TensorF64::from_data(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let rhs = TensorF64::from_data(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]);
        let correct = dot_general_f64(&lhs, &rhs, &dnums(&[], &[], &[1], &[0]));
        // Contract lhs's axis 0 instead of axis 1 — same shapes throughout.
        let transposed = dot_general_f64(&lhs, &rhs, &dnums(&[], &[], &[0], &[0]));
        assert_eq!(correct.dims, transposed.dims, "both must be shape-valid");
        assert_ne!(correct.data(), transposed.data(), "but the numbers must differ");
    }

    #[test]
    fn dot_general_f64_matches_gqa_attention_scores_shape() {
        // [1,2,4,3] batched over (B,H) contracting head_dim=3, mirroring
        // ops::attention::gqa_attention's QK^T dimension numbers.
        let q = TensorF64::from_data(
            &[1, 1, 2, 3],
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        );
        let k = TensorF64::from_data(
            &[1, 1, 2, 3],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        );
        let scores = dot_general_f64(&q, &k, &dnums(&[0, 1], &[0, 1], &[3], &[3]));
        assert_eq!(scores.dims, vec![1, 1, 2, 2]);
        // row 0 (q=[1,0,0]) . k rows [1,0,0] and [0,0,1] -> [1, 0]
        // row 1 (q=[0,1,0]) . k rows [1,0,0] and [0,0,1] -> [0, 0]
        assert_eq!(scores.data(), &[1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn rms_norm_f64_matches_hand_computed_values() {
        // x=[3,4], mean(x^2)=(9+16)/2=12.5, scale=1/sqrt(12.5+0)=1/3.5355...
        let out = rms_norm_f64(&[3.0, 4.0], &[1.0, 1.0], 0.0);
        let expected_scale = 1.0 / 12.5f64.sqrt();
        assert!((out[0] - 3.0 * expected_scale).abs() < 1e-12);
        assert!((out[1] - 4.0 * expected_scale).abs() < 1e-12);
    }

    #[test]
    fn rms_norm_f64_zero_weight_zeroes_the_output_unlike_the_zero_centered_variant() {
        // Contrast with ops::norm::rms_norm_zero_centered, which treats a
        // zero weight as the *identity* scale — this plain reference must not.
        let out = rms_norm_f64(&[1.0, 2.0, 3.0], &[0.0, 0.0, 0.0], 1e-6);
        assert!(out.iter().all(|&v| v == 0.0), "{out:?}");
    }

    #[test]
    fn softmax_f64_sums_to_one_and_matches_uniform_on_equal_inputs() {
        let out = softmax_f64(&[1.0, 1.0, 1.0, 1.0]);
        for v in &out {
            assert!((v - 0.25).abs() < 1e-12);
        }
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn softmax_f64_is_shift_invariant() {
        let a = softmax_f64(&[1.0, 2.0, 3.0]);
        let b = softmax_f64(&[1001.0, 1002.0, 1003.0]);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).abs() < 1e-9, "{a:?} vs {b:?}");
        }
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn tensor_f64_rejects_an_out_of_bounds_index() {
        let t = TensorF64::zeros(&[2, 2]);
        let _ = t.at(&[2, 0]);
    }
}
