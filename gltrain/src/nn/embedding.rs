//! Stummañ Gwiskadur: embedding table.
//!
//! The first layer of any language model, and the one place in the forward
//! pass where the input is not a tensor of floats.
//!
//! # Why this does not implement [`Module`]
//!
//! [`Module::forward`] is `Tensor<B> -> Tensor<B>`. An embedding is
//! `&[u32] -> Tensor<B>`: its input is token ids, which are indices, not
//! values. Squeezing them through an f32 tensor would mean rounding on the way
//! in and would make an out-of-range id look like an ordinary number rather
//! than the vocabulary mismatch it is.
//!
//! So [`ABEmbedding::forward_ids`] is an inherent method and the parameter
//! accessors are inherent too. The cost is that
//! [`crate::nn::trainable_parameters`] does not reach it and a caller collects
//! its parameters explicitly. That is the honest trade: the alternative is a
//! `Module` impl whose `forward` always fails, which is a lie told to the type
//! system.
//!
//! # Shape convention
//!
//! The table is `[vocab_size, d_model]`, row `t` being token `t`'s vector.
//! This matches every checkpoint format in circulation, and it is also the
//! layout the gather wants: one row is contiguous.
//!
//! Output is `[n_tokens, d_model]`, which feeds [`crate::nn::ABLinear`]
//! directly under this crate's row-vector convention. A caller holding a
//! `[batch, seq]` batch flattens it to `batch * seq` ids first — there is no
//! 3-D storage in this crate, the same restriction `matmul` carries.

use crate::autograd::tape::Tape;
use crate::error::{GlTrainError, Result};
use crate::nn::param::TPParameter;
use crate::tensor::backend::Backend;
use crate::tensor::Tensor;
use std::sync::{Arc, Mutex};

/// A token-embedding table: `y[i] = weight[ids[i]]`.
///
/// `AB` because it is a reusable algorithmic building block, the same category
/// as [`crate::nn::ABLinear`].
pub struct ABEmbedding<B: Backend> {
    weight: TPParameter<B>,
}

impl<B: Backend> ABEmbedding<B> {
    /// Build from an existing table of shape `[vocab_size, d_model]`.
    pub fn new(weight: TPParameter<B>) -> Result<Self> {
        if weight.shape().len() != 2 {
            return Err(GlTrainError::InvalidOp(format!(
                "ABEmbedding weight must be 2D [vocab_size, d_model], got {:?}",
                weight.shape()
            )));
        }
        if weight.shape()[0] == 0 || weight.shape()[1] == 0 {
            return Err(GlTrainError::InvalidOp(format!(
                "ABEmbedding weight must have a non-zero vocab and width, got {:?}",
                weight.shape()
            )));
        }
        Ok(Self { weight })
    }

    /// A trainable table with `N(0, std^2)` rows.
    pub fn randn(
        name: &str,
        vocab_size: usize,
        d_model: usize,
        std: f32,
        seed: u64,
    ) -> Result<Self> {
        let w = Tensor::randn(&[vocab_size, d_model], std, seed)?;
        Self::new(TPParameter::trainable(format!("{name}.weight"), w))
    }

    /// Wrap a pre-existing table as **frozen**.
    pub fn frozen(name: &str, weight: Tensor<B>) -> Result<Self> {
        Self::new(TPParameter::frozen(format!("{name}.weight"), weight))
    }

    /// Number of rows in the table.
    pub fn vocab_size(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Width of one embedding vector.
    pub fn d_model(&self) -> usize {
        self.weight.shape()[1]
    }

    /// The table parameter.
    pub fn weight(&self) -> &TPParameter<B> {
        &self.weight
    }

    /// Mutable table access, for the optimizer.
    pub fn weight_mut(&mut self) -> &mut TPParameter<B> {
        &mut self.weight
    }

    /// This layer's parameters, in a stable order.
    pub fn parameters(&self) -> Vec<&TPParameter<B>> {
        vec![&self.weight]
    }

    /// Mutable parameter access, for the optimizer's update.
    pub fn parameters_mut(&mut self) -> Vec<&mut TPParameter<B>> {
        vec![&mut self.weight]
    }

    /// Gather one row per token id, producing `[ids.len(), d_model]`.
    ///
    /// Records an "Embedding" node when the table is trainable. The gather
    /// itself, including its scatter-add backward pass, is
    /// [`Tensor::embedding`]; this method only supplies the tracked table.
    pub fn forward_ids(&self, ids: &[u32], tape: &Arc<Mutex<Tape>>) -> Result<Tensor<B>> {
        if ids.is_empty() {
            return Err(GlTrainError::InvalidOp(
                "ABEmbedding::forward_ids needs at least one token id".into(),
            ));
        }
        // The gather, its shape checks and its scatter-add backward all live on
        // `Tensor`, where every other tape-recording op in this crate lives.
        // `tracked` is what re-registers the table with the tape each step.
        self.weight.tracked(tape).embedding(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::GlProc;

    /// A table whose row `t` is `[t, t, t]`, so a gather is trivially checkable.
    fn table_4x3() -> ABEmbedding<GlProc> {
        let data: Vec<f32> = (0..4).flat_map(|t| [t as f32; 3]).collect();
        let w = Tensor::<GlProc>::from_vec(data, &[4, 3]).unwrap();
        ABEmbedding::new(TPParameter::trainable("embed", w)).unwrap()
    }

    #[test]
    fn forward_gathers_the_requested_rows_in_order() {
        let e = table_4x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let y = e.forward_ids(&[2, 0, 3], &tape).unwrap();
        assert_eq!(y.shape(), &[3, 3]);
        assert_eq!(
            y.to_vec().unwrap(),
            vec![2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 3.0, 3.0, 3.0]
        );
    }

    #[test]
    fn reported_dimensions_match_the_table() {
        let e = table_4x3();
        assert_eq!(e.vocab_size(), 4);
        assert_eq!(e.d_model(), 3);
    }

    #[test]
    fn an_id_past_the_vocabulary_is_named_rather_than_read_out_of_bounds() {
        let e = table_4x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let err = e.forward_ids(&[0, 9], &tape).unwrap_err();
        assert!(
            err.to_string().contains("outside a vocabulary of 4"),
            "got {err}"
        );
    }

    #[test]
    fn an_empty_id_slice_is_refused() {
        let e = table_4x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        assert!(e.forward_ids(&[], &tape).is_err());
    }

    #[test]
    fn a_non_2d_table_is_rejected() {
        let w = Tensor::<GlProc>::zeros(&[8]).unwrap();
        assert!(ABEmbedding::new(TPParameter::trainable("e", w)).is_err());
    }

    #[test]
    fn a_trainable_table_records_a_node_and_a_frozen_one_does_not() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        table_4x3().forward_ids(&[1], &tape).unwrap();
        assert_eq!(Tape::lock(&tape).len(), 1);

        let w = Tensor::<GlProc>::zeros(&[4, 3]).unwrap();
        let frozen = ABEmbedding::<GlProc>::frozen("base", w).unwrap();
        frozen.forward_ids(&[1], &tape).unwrap();
        // Nothing else in this graph is tracked, so the frozen gather records
        // nothing at all.
        assert_eq!(Tape::lock(&tape).len(), 1);
    }

    // ── Backward ─────────────────────────────────────────────────────────

    /// The gradient must land on exactly the rows that were gathered.
    #[test]
    fn backward_scatters_the_gradient_onto_the_selected_rows() {
        let e = table_4x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let y = e.forward_ids(&[1, 3], &tape).unwrap();
        // `sum` gives every output element a gradient of 1.
        y.sum().unwrap();
        Tape::lock(&tape).backward().unwrap();

        let (grad, shape) = Tape::lock(&tape).grad(e.weight().id()).cloned().unwrap();
        assert_eq!(shape, vec![4, 3]);
        // Rows 1 and 3 gathered once each; rows 0 and 2 were never touched.
        assert_eq!(
            grad,
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
        );
    }

    /// The property an `=` instead of a `+=` would break: a token used twice
    /// accumulates twice. Assigning would keep only the last occurrence and
    /// silently underweight the most frequent tokens.
    #[test]
    fn a_repeated_token_accumulates_once_per_occurrence() {
        let e = table_4x3();
        let tape = Arc::new(Mutex::new(Tape::new()));
        let y = e.forward_ids(&[2, 2, 2], &tape).unwrap();
        y.sum().unwrap();
        Tape::lock(&tape).backward().unwrap();

        let (grad, _) = Tape::lock(&tape).grad(e.weight().id()).cloned().unwrap();
        assert_eq!(&grad[6..9], &[3.0, 3.0, 3.0], "three occurrences, not one");
        assert!(grad[..6].iter().all(|&g| g == 0.0));
        assert!(grad[9..].iter().all(|&g| g == 0.0));
    }

    #[test]
    fn a_frozen_table_receives_no_gradient() {
        let tape = Arc::new(Mutex::new(Tape::new()));
        let w = Tensor::<GlProc>::from_vec((0..12).map(|v| v as f32).collect(), &[4, 3]).unwrap();
        let frozen = ABEmbedding::<GlProc>::frozen("base", w).unwrap();
        let y = frozen.forward_ids(&[1, 2], &tape).unwrap();
        assert!(y.sum().is_ok());
        // Nothing was tracked, so there is no node and no gradient to read.
        assert!(Tape::lock(&tape).is_empty());
        assert!(Tape::lock(&tape).grad(frozen.weight().id()).is_none());
    }
}
