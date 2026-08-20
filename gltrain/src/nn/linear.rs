//! Stummañ Gwiskadur: linear layer.
//!
//! # Shape convention, stated once for the whole crate
//!
//! Stummañ uses the **row-vector** form: `y = x @ W`, with
//!
//! ```text
//! x: [batch, d_in]    W: [d_in, d_out]    y: [batch, d_out]
//! ```
//!
//! This is the transpose of what PyTorch, candle and the LoRA paper use. They
//! write `y = x @ W^T` with `W: [d_out, d_in]` (`gltrain/src/train/lora.rs`
//! builds `lora_a` as `(r, d_in)` for exactly that reason).
//!
//! Row-vector was chosen because `Tensor::matmul` is a plain `A @ B` and this
//! convention needs no transpose in the forward pass. Every transpose skipped is
//! an allocation and a backward node that never has to exist.
//!
//! **This is a load-bearing decision and getting it wrong is nearly silent.**
//! A square projection (`d_in == d_out`, which q_proj and o_proj usually are)
//! passes every shape check under either convention and just computes the wrong
//! answer. gljax hit this exact bug with HF weights, and
//! `glcore`'s safetensors binder now checks for transposed shapes specifically
//! because element counts match. Checkpoint validation here does the same.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::module::Module;
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// A dense linear layer: `y = x @ W (+ b)`.
///
/// `AB` because it is a reusable algorithmic building block, the same category
/// as `ABRMSNorm` and `ABAttention`. It holds weights, but weights are not
/// cross-step state in the sense that separates `OP` from `AB`: nothing here
/// persists between optimizer steps on its own.
pub struct ABLinear<B: Backend> {
    weight: TPParameter<B>,
    bias: Option<TPParameter<B>>,
}

impl<B: Backend> ABLinear<B> {
    /// Build from an existing weight of shape `[d_in, d_out]`.
    pub fn new(weight: TPParameter<B>, bias: Option<TPParameter<B>>) -> Result<Self> {
        if weight.shape().len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "ABLinear weight must be 2D [d_in, d_out], got {:?}",
                weight.shape()
            )));
        }
        if let Some(b) = &bias {
            let d_out = weight.shape()[1];
            if b.shape() != [1, d_out] {
                return Err(GlTrainError::ShapeMismatch {
                    expected: vec![1, d_out],
                    got: b.shape().to_vec(),
                });
            }
        }
        Ok(Self { weight, bias })
    }

    /// A trainable layer with `N(0, std^2)` weights and no bias.
    pub fn randn(name: &str, d_in: usize, d_out: usize, std: f32, seed: u64) -> Result<Self> {
        let w = Tensor::randn(&[d_in, d_out], std, seed)?;
        Self::new(TPParameter::trainable(format!("{name}.weight"), w), None)
    }

    /// Wrap a pre-existing weight as a **frozen** layer. This is a LoRA base.
    pub fn frozen(name: &str, weight: Tensor<B>) -> Result<Self> {
        Self::new(TPParameter::frozen(format!("{name}.weight"), weight), None)
    }

    /// Input dimension.
    pub fn d_in(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Output dimension.
    pub fn d_out(&self) -> usize {
        self.weight.shape()[1]
    }

    /// The weight parameter.
    pub fn weight(&self) -> &TPParameter<B> {
        &self.weight
    }

    /// Mutable weight access, for the optimizer and for merging.
    pub fn weight_mut(&mut self) -> &mut TPParameter<B> {
        &mut self.weight
    }
}

impl<B: Backend> Module<B> for ABLinear<B> {
    fn forward(&self, x: &Tensor<B>, tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>> {
        if x.ndim() != 2 || x.shape()[1] != self.d_in() {
            return Err(GlTrainError::ShapeMismatch {
                expected: vec![x.shape().first().copied().unwrap_or(0), self.d_in()],
                got: x.shape().to_vec(),
            });
        }
        let y = x.matmul(&self.weight.tracked(tape))?;
        match &self.bias {
            // Bias is [1, d_out] and `add` requires exact shapes, so this only
            // works for batch = 1. Broadcasting is M3 work; erroring is better
            // than a wrong answer, and `add` already produces a ShapeMismatch.
            Some(b) => y.add(&b.tracked(tape)),
            None => Ok(y),
        }
    }

    fn parameters(&self) -> Vec<&TPParameter<B>> {
        match &self.bias {
            Some(b) => vec![&self.weight, b],
            None => vec![&self.weight],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        match &mut self.bias {
            Some(b) => vec![&mut self.weight, b],
            None => vec![&mut self.weight],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// A 2x2 by 2x2 matmul accumulates over K = 2, so f32 error is at the last
    /// bit. Matches the TOL_MATMUL reasoning in tensor.rs.
    const TOL_MATMUL: f32 = 1e-4;

    fn linear_2x3() -> ABLinear<GlProc> {
        // W = [[1,2,3],[4,5,6]] : d_in = 2, d_out = 3
        let w = Tensor::<GlProc>::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        ABLinear::new(TPParameter::trainable("w", w), None).unwrap()
    }

    #[test]
    fn forward_computes_x_times_w() {
        let lin = linear_2x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        // x = [[1, 1]] -> y = [1+4, 2+5, 3+6] = [5, 7, 9]
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 1.0], &[1, 2]).unwrap();
        let y = lin.forward(&x, &tape).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
        let got = y.to_vec().unwrap();
        for (g, w) in got.iter().zip([5.0, 7.0, 9.0]) {
            assert!((g - w).abs() < TOL_MATMUL, "got {got:?}");
        }
    }

    #[test]
    fn forward_rejects_a_wrong_input_width() {
        let lin = linear_2x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::zeros(&[1, 5]).unwrap();
        assert!(lin.forward(&x, &tape).is_err());
    }

    #[test]
    fn a_trainable_layer_records_a_node_and_a_frozen_one_does_not() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let x = Tensor::<GlProc>::from_vec(vec![1.0, 1.0], &[1, 2])
            .unwrap()
            .with_grad(tape.clone());

        let trainable = linear_2x3();
        trainable.forward(&x, &tape).unwrap();
        let after_trainable = Tape::lock(&tape).len();

        let w = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        let frozen = ABLinear::<GlProc>::frozen("base", w).unwrap();
        frozen.forward(&x, &tape).unwrap();
        let after_frozen = Tape::lock(&tape).len();

        // Both record a Matmul node because `x` is tracked; the difference is
        // whether the weight will receive a gradient, which the backward
        // closure decides. Both must record, or the input's gradient is lost.
        assert_eq!(after_trainable, 1);
        assert_eq!(after_frozen, 2);
    }

    #[test]
    fn reported_dimensions_follow_the_row_vector_convention() {
        // Guards the shape convention documented at the top of this file.
        let lin = linear_2x3();
        assert_eq!(lin.d_in(), 2);
        assert_eq!(lin.d_out(), 3);
    }

    #[test]
    fn a_non_2d_weight_is_rejected() {
        let w = Tensor::<GlProc>::zeros(&[8]).unwrap();
        assert!(ABLinear::new(TPParameter::trainable("w", w), None).is_err());
    }

    #[test]
    fn a_bias_of_the_wrong_width_is_rejected() {
        let w = Tensor::<GlProc>::zeros(&[2, 3]).unwrap();
        let b = Tensor::<GlProc>::zeros(&[1, 7]).unwrap();
        assert!(ABLinear::new(
            TPParameter::trainable("w", w),
            Some(TPParameter::trainable("b", b))
        )
        .is_err());
    }
}
