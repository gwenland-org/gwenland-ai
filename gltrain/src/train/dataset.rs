//! Candle-dependent dataset helpers: tokenisation and batching.
//!
//! The candle-free parts (`Sample`, `load_jsonl`, `DEFAULT_MAX_LEN`) live in
//! `samples.rs` so they can be used by `dry_run` without the candle feature.
//! This module re-exports them for callers that already depend on candle.

pub use crate::train::samples::{load_jsonl, Sample, DEFAULT_MAX_LEN};

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

// ── tokenize ─────────────────────────────────────────────────────────────────

/// Encode each sample as `"<input>\n<output>"`, truncate to `max_len` token
/// IDs, and convert to a 1-D `Tensor` of `u32` on `device`.
pub fn tokenize(
    samples: &[Sample],
    tokenizer: &Tokenizer,
    max_len: usize,
    device: &Device,
) -> Result<Vec<Tensor>> {
    samples
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let text = format!("{}\n{}", s.input, s.output);
            let encoding = tokenizer
                .encode(text.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenizer failed on sample {}: {}", i, e))?;

            let ids = encoding.get_ids();
            let truncated: &[u32] = if ids.len() > max_len {
                &ids[..max_len]
            } else {
                ids
            };

            Tensor::from_slice(truncated, truncated.len(), device)
                .with_context(|| format!("failed to build tensor for sample {}", i))
        })
        .collect()
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Average token count across all tensors (truncated to `usize`).
/// Returns 0 for an empty slice.
pub fn avg_token_length(tensors: &[Tensor]) -> usize {
    if tensors.is_empty() {
        return 0;
    }
    let total: usize = tensors.iter().map(|t| t.elem_count()).sum();
    total / tensors.len()
}

/// Partition `tensors` into chunks of `batch_size`. The last chunk may be
/// smaller if `tensors.len()` is not evenly divisible.
pub fn batch(tensors: Vec<Tensor>, batch_size: usize) -> Vec<Vec<Tensor>> {
    let batch_size = batch_size.max(1);
    tensors
        .into_iter()
        .collect::<Vec<_>>()
        .chunks(batch_size)
        .map(|c| c.to_vec())
        .collect()
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_splits_correctly() {
        let device = candle_core::Device::Cpu;
        let tensors: Vec<Tensor> = (0u32..7)
            .map(|i| Tensor::from_slice(&[i], 1, &device).unwrap())
            .collect();
        let batches = batch(tensors, 3);
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 1);
    }

    #[test]
    fn avg_token_length_empty() {
        assert_eq!(avg_token_length(&[]), 0);
    }

    #[test]
    fn avg_token_length_uniform() {
        let device = candle_core::Device::Cpu;
        let tensors: Vec<Tensor> = vec![
            Tensor::from_slice(&[0u32, 1, 2, 3], 4, &device).unwrap(),
            Tensor::from_slice(&[0u32, 1, 2, 3], 4, &device).unwrap(),
        ];
        assert_eq!(avg_token_length(&tensors), 4);
    }
}
