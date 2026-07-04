//! Autograd-compatible transformer tensor operations shared by training code.
//!
//! These helpers contain no model loading, tensor-name lookup, KV cache, or
//! inference-specific error handling. All operations stay inside Candle's
//! differentiable graph.

use candle_core::{D, DType, Result, Tensor};
use candle_nn::ops::softmax;

/// Root-mean-square normalization over the last dimension.
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let mean_sq = x.sqr()?.mean_keepdim(D::Minus1)?;
    let rms = (mean_sq + eps as f64)?.sqrt()?;
    x.broadcast_div(&rms)?.broadcast_mul(weight)
}

/// Build RoPE cosine and sine tables for absolute positions
/// `position_offset..position_offset + seq_len`.
pub fn rope_tables(
    seq_len: usize,
    head_dim: usize,
    rope_theta: f32,
    position_offset: usize,
    dtype: DType,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    if head_dim == 0 || head_dim % 2 != 0 {
        candle_core::bail!("RoPE head_dim must be positive and even, got {head_dim}");
    }
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        candle_core::bail!("RoPE theta must be finite and positive, got {rope_theta}");
    }

    let inv_freq: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let positions: Vec<f32> = (position_offset..position_offset + seq_len)
        .map(|p| p as f32)
        .collect();

    let inv_freq = Tensor::from_vec(inv_freq, (1, head_dim / 2), device)?.to_dtype(dtype)?;
    let positions = Tensor::from_vec(positions, (seq_len, 1), device)?.to_dtype(dtype)?;
    let freqs = positions.matmul(&inv_freq)?;
    Ok((freqs.cos()?, freqs.sin()?))
}

/// Apply non-interleaved LLaMA/Qwen RoPE to `[batch, heads, seq, head_dim]`.
pub fn apply_rope(x: &Tensor, rope_theta: f32, position_offset: usize) -> Result<Tensor> {
    let (_, _, seq_len, head_dim) = x.dims4()?;
    let (cos, sin) = rope_tables(
        seq_len,
        head_dim,
        rope_theta,
        position_offset,
        x.dtype(),
        x.device(),
    )?;
    candle_nn::rotary_emb::rope_slow(x, &cos, &sin)
}

/// Expand grouped K/V heads from `[batch, kv_heads, seq, head_dim]` to
/// `[batch, query_heads, seq, head_dim]`.
pub fn repeat_kv(x: &Tensor, query_heads: usize) -> Result<Tensor> {
    let (batch, kv_heads, seq_len, head_dim) = x.dims4()?;
    if kv_heads == 0 || query_heads == 0 || query_heads % kv_heads != 0 {
        candle_core::bail!(
            "query_heads ({query_heads}) must be divisible by kv_heads ({kv_heads})"
        );
    }
    if query_heads == kv_heads {
        return Ok(x.clone());
    }

    let groups = query_heads / kv_heads;
    x.unsqueeze(2)?
        .broadcast_as((batch, kv_heads, groups, seq_len, head_dim))?
        .contiguous()?
        .reshape((batch, query_heads, seq_len, head_dim))
}

/// Add a standard autoregressive mask to attention scores shaped
/// `[batch, heads, seq, seq]`.
pub fn apply_causal_mask(scores: &Tensor) -> Result<Tensor> {
    let (_, _, query_len, key_len) = scores.dims4()?;
    if query_len != key_len {
        candle_core::bail!(
            "full-sequence causal attention requires query_len == key_len, got {query_len} and {key_len}"
        );
    }

    let mut values = vec![0.0f32; query_len * key_len];
    for query in 0..query_len {
        for key in query + 1..key_len {
            values[query * key_len + key] = f32::NEG_INFINITY;
        }
    }
    let mask = Tensor::from_vec(values, (1, 1, query_len, key_len), scores.device())?
        .to_dtype(scores.dtype())?;
    scores.broadcast_add(&mask)
}

/// Causal scaled dot-product attention.
///
/// Q/K/V use `[batch, heads, seq, head_dim]`. K/V must already be expanded to
/// the query-head count.
pub fn causal_scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
) -> Result<Tensor> {
    let (_, query_heads, _, head_dim) = query.dims4()?;
    let (_, key_heads, _, _) = key.dims4()?;
    let (_, value_heads, _, _) = value.dims4()?;
    if query_heads != key_heads || query_heads != value_heads {
        candle_core::bail!(
            "attention head mismatch: query={query_heads}, key={key_heads}, value={value_heads}"
        );
    }

    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores = query.matmul(&key.transpose(2, 3)?)?.affine(scale, 0.0)?;
    let scores = apply_causal_mask(&scores)?;
    let weights = softmax(&scores, D::Minus1)?;
    weights.matmul(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Var};

    fn flat(t: &Tensor) -> Vec<f32> {
        t.flatten_all().unwrap().to_vec1().unwrap()
    }

    #[test]
    fn rope_preserves_shape_changes_later_positions_and_backprops() {
        let device = Device::Cpu;
        let input = Tensor::from_vec(
            vec![1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
            (1, 1, 2, 4),
            &device,
        )
        .unwrap();
        let input = Var::from_tensor(&input).unwrap();
        let output = apply_rope(input.as_tensor(), 10_000.0, 0).unwrap();

        assert_eq!(output.dims(), &[1, 1, 2, 4]);
        let values = flat(&output);
        assert_eq!(&values[..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_ne!(&values[4..], &[1.0, 2.0, 3.0, 4.0]);

        let grads = output.sum_all().unwrap().backward().unwrap();
        let grad = grads.get(input.as_tensor()).expect("input gradient");
        assert!(flat(grad).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn repeat_kv_repeats_each_kv_head_as_a_group() {
        let device = Device::Cpu;
        let kv = Tensor::from_vec(vec![1.0f32, 2.0], (1, 2, 1, 1), &device).unwrap();
        let repeated = repeat_kv(&kv, 4).unwrap();
        assert_eq!(repeated.dims(), &[1, 4, 1, 1]);
        assert_eq!(flat(&repeated), vec![1.0, 1.0, 2.0, 2.0]);
    }

    #[test]
    fn causal_attention_never_reads_future_values() {
        let device = Device::Cpu;
        let q = Tensor::zeros((1, 1, 3, 2), DType::F32, &device).unwrap();
        let k = Tensor::zeros((1, 1, 3, 2), DType::F32, &device).unwrap();
        let v = Tensor::from_vec(
            vec![1.0f32, 1.0, 3.0, 3.0, 100.0, 100.0],
            (1, 1, 3, 2),
            &device,
        )
        .unwrap();

        let output = causal_scaled_dot_product_attention(&q, &k, &v).unwrap();
        let values = flat(&output);
        assert!((values[0] - 1.0).abs() < 1e-6);
        assert!((values[2] - 2.0).abs() < 1e-6);
        assert!((values[4] - (104.0 / 3.0)).abs() < 1e-5);
    }
}
