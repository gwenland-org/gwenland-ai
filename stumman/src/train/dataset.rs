//! Stummañ Deskiñ: the in-memory dataset M2's convergence test trains on.
//!
//! Deliberately tiny and deliberately synthetic. M2's exit criterion is "the
//! loop closes and the loss goes down", which needs a task whose answer is
//! known in advance and reachable at the rank being tested. Reading a real
//! corpus would add a tokenizer, a collator, and an I/O path to a wave whose
//! job is to prove the optimizer, the tape and the checkpoint fit together.

use crate::error::{GlTrainError, Result};
use crate::rng::Xorshift64Star;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;

/// A fixed list of `(input, target)` pairs, held on the host.
///
/// `VL` because it is a plain data bag: no backend, no tape, no behaviour
/// beyond indexing. Tensors are built on demand in [`VLMicroDataset::sample`],
/// so one dataset serves any backend.
#[derive(Debug, Clone, PartialEq)]
pub struct VLMicroDataset {
    samples: Vec<(Vec<f32>, Vec<f32>)>,
    d_in: usize,
    d_out: usize,
}

impl VLMicroDataset {
    /// An empty dataset expecting `[1, d_in]` inputs and `[1, d_out]` targets.
    pub fn new(d_in: usize, d_out: usize) -> Self {
        Self {
            samples: Vec::new(),
            d_in,
            d_out,
        }
    }

    /// Add one pair. Rejects a length that disagrees with the declared
    /// dimensions rather than storing it and failing at the first forward pass.
    pub fn push(&mut self, input: Vec<f32>, target: Vec<f32>) -> Result<()> {
        if input.len() != self.d_in {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![1, self.d_in],
                got: vec![1, input.len()],
            });
        }
        if target.len() != self.d_out {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![1, self.d_out],
                got: vec![1, target.len()],
            });
        }
        self.samples.push((input, target));
        Ok(())
    }

    /// How many pairs.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether there is nothing to train on.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Input dimension.
    pub fn d_in(&self) -> usize {
        self.d_in
    }

    /// Output dimension.
    pub fn d_out(&self) -> usize {
        self.d_out
    }

    /// Build sample `idx` as a `[1, d_in]` input and a `[1, d_out]` target.
    ///
    /// Batch is 1 because `Tensor::matmul` is 2-D only and `ABLinear`'s bias
    /// path requires it. A wider batch is a real matmul here and would work
    /// for the no-bias case, but nothing in M2 needs one.
    pub fn sample<B: Backend>(&self, idx: usize) -> Result<(Tensor<B>, Tensor<B>)> {
        let (x, y) = self.samples.get(idx).ok_or_else(|| {
            GlTrainError::InvalidOp(format!(
                "sample {idx} is out of range for a {}-sample dataset",
                self.samples.len()
            ))
        })?;
        Ok((
            Tensor::<B>::from_vec(x.clone(), &[1, self.d_in])?,
            Tensor::<B>::from_vec(y.clone(), &[1, self.d_out])?,
        ))
    }

    /// A regression task with a known answer: `x ~ N(0,1)`, `target = x @ W`.
    ///
    /// Returns the dataset and the `[d_in, d_out]` matrix it was generated
    /// from, row-major. The caller needs `W` to build a base weight that is
    /// deliberately *not* it, so the adapter has a real residual to learn.
    ///
    /// Deterministic in `seed`, so a failing convergence run reproduces
    /// exactly. That is a requirement rather than a nicety: a loss curve that
    /// cannot be reproduced cannot be debugged.
    pub fn synthetic_regression(
        n: usize,
        d_in: usize,
        d_out: usize,
        seed: u64,
    ) -> Result<(Self, Vec<f32>)> {
        if n == 0 || d_in == 0 || d_out == 0 {
            return Err(GlTrainError::InvalidOp(
                "synthetic_regression needs n, d_in and d_out all at least 1".into(),
            ));
        }
        let mut rng = Xorshift64Star::new(seed);
        let w: Vec<f32> = rng.normal_vec(d_in * d_out, 1.0);

        let mut ds = Self::new(d_in, d_out);
        for _ in 0..n {
            let x = rng.normal_vec(d_in, 1.0);
            // target = x @ W, with W row-major [d_in, d_out].
            let mut y = vec![0.0f32; d_out];
            for (i, xi) in x.iter().enumerate() {
                for (j, yj) in y.iter_mut().enumerate() {
                    *yj += xi * w[i * d_out + j];
                }
            }
            ds.push(x, y)?;
        }
        Ok((ds, w))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// One matmul over d_in = 4, so only f32 rounding separates the generated
    /// target from a recomputation of it.
    const TOL_MATMUL: f32 = 1e-4;

    #[test]
    fn synthetic_regression_produces_the_requested_shape_and_count() {
        let (ds, w) = VLMicroDataset::synthetic_regression(10, 4, 3, 42).unwrap();
        assert_eq!(ds.len(), 10);
        assert_eq!(ds.d_in(), 4);
        assert_eq!(ds.d_out(), 3);
        assert_eq!(w.len(), 12);
        let (x, y) = ds.sample::<GlProc>(0).unwrap();
        assert_eq!(x.shape(), &[1, 4]);
        assert_eq!(y.shape(), &[1, 3]);
    }

    /// The targets must actually be `x @ W`, or the task has no reachable
    /// answer and a failing convergence test would be blaming the optimizer
    /// for a broken dataset.
    #[test]
    fn synthetic_targets_equal_the_input_times_the_generating_matrix() {
        let (ds, w) = VLMicroDataset::synthetic_regression(5, 4, 3, 7).unwrap();
        let w_t = Tensor::<GlProc>::from_vec(w, &[4, 3]).unwrap();
        for i in 0..ds.len() {
            let (x, y) = ds.sample::<GlProc>(i).unwrap();
            let recomputed = x.matmul(&w_t).unwrap().to_vec().unwrap();
            for (j, (got, want)) in y.to_vec().unwrap().iter().zip(&recomputed).enumerate() {
                assert!(
                    (got - want).abs() < TOL_MATMUL,
                    "sample {i} element {j}: {got} != {want}"
                );
            }
        }
    }

    /// A loss curve that cannot be reproduced cannot be debugged.
    #[test]
    fn synthetic_regression_is_deterministic_in_its_seed() {
        let (a, wa) = VLMicroDataset::synthetic_regression(6, 4, 4, 99).unwrap();
        let (b, wb) = VLMicroDataset::synthetic_regression(6, 4, 4, 99).unwrap();
        assert_eq!(a, b);
        assert_eq!(wa, wb);

        let (c, wc) = VLMicroDataset::synthetic_regression(6, 4, 4, 100).unwrap();
        assert_ne!(wa, wc, "a different seed must give a different problem");
        assert_ne!(a, c);
    }

    #[test]
    fn push_rejects_a_row_of_the_wrong_length() {
        let mut ds = VLMicroDataset::new(3, 2);
        assert!(ds.push(vec![1.0, 2.0], vec![1.0, 2.0]).is_err());
        assert!(ds.push(vec![1.0, 2.0, 3.0], vec![1.0]).is_err());
        assert!(ds.push(vec![1.0, 2.0, 3.0], vec![1.0, 2.0]).is_ok());
        assert_eq!(ds.len(), 1);
    }

    #[test]
    fn sampling_out_of_range_is_an_error_rather_than_a_panic() {
        let ds = VLMicroDataset::new(2, 2);
        assert!(ds.is_empty());
        assert!(ds.sample::<GlProc>(0).is_err());
    }

    #[test]
    fn synthetic_regression_rejects_a_degenerate_request() {
        assert!(VLMicroDataset::synthetic_regression(0, 4, 4, 1).is_err());
        assert!(VLMicroDataset::synthetic_regression(4, 0, 4, 1).is_err());
        assert!(VLMicroDataset::synthetic_regression(4, 4, 0, 1).is_err());
    }
}
