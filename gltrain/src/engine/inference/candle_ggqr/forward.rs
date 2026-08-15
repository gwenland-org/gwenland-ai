// engine/inference/candle_ggqr/forward.rs — RMSNorm, attention, MLP, forward pass.
//
// All functions operate on pre-dequantised `candle_core::Tensor` values stored
// in the `LoadedState::tensors` HashMap. The HashMap keys follow the standard
// GGUF naming convention, e.g.:
//   "token_embd.weight"
//   "blk.{layer}.attn_norm.weight"
//   "blk.{layer}.attn_q.weight"
//   "blk.{layer}.attn_k.weight"
//   "blk.{layer}.attn_v.weight"
//   "blk.{layer}.attn_output.weight"
//   "blk.{layer}.ffn_norm.weight"
//   "blk.{layer}.ffn_gate.weight"
//   "blk.{layer}.ffn_up.weight"
//   "blk.{layer}.ffn_down.weight"
//   "output_norm.weight"
//   "output.weight"
//
// No KV cache — each forward call re-computes full attention over the current
// context (deferred to GWEN-215 as noted in the spec).
//
// Requirements: 5.1, 5.2, 5.3, 5.4, 5.6

use std::collections::HashMap;

use candle_core::Tensor;
use candle_nn::ops::{self, softmax};

use crate::error::GwenError;
use super::ModelConfig;

// ── Helper: tensor lookup ─────────────────────────────────────────────────────

fn get(tensors: &HashMap<String, Tensor>, key: &str) -> Result<Tensor, GwenError> {
    tensors.get(key).cloned().ok_or_else(|| GwenError::InferenceError {
        layer: key.to_string(),
        operation: "weight_lookup".to_string(),
        error: format!("tensor '{}' not found in loaded model", key),
    })
}

// ── 9.1 RMSNorm ──────────────────────────────────────────────────────────────

/// Root-mean-square layer normalisation.
///
/// Formula: `y = x / sqrt(mean(x²) + eps) * weight`
///
/// Requirement: 5.4
pub fn rms_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor, GwenError> {
    let layer = "rms_norm";
    let op = |e: candle_core::Error| GwenError::InferenceError {
        layer: layer.to_string(),
        operation: "rms_norm".to_string(),
        error: e.to_string(),
    };

    // mean(x²) over the last dimension, keep dims for broadcasting.
    let x_sq = x.sqr().map_err(op)?;
    let mean_sq = x_sq.mean_keepdim(candle_core::D::Minus1).map_err(op)?;

    // rms = sqrt(mean(x²) + eps)
    let rms = (mean_sq + eps as f64).map_err(op)?.sqrt().map_err(op)?;

    // normalise and scale
    let x_norm = x.broadcast_div(&rms).map_err(op)?;
    x_norm.broadcast_mul(weight).map_err(op)
}

// ── 10.1 Multi-head attention (no KV cache) ───────────────────────────────────

/// Scaled dot-product multi-head attention for a single transformer layer.
///
/// Layout (all tensors are 2-D weight matrices as stored in the GGUF file):
///   W_q: [hidden, hidden]   (may be [n_heads * head_dim, hidden])
///   W_k: [kv_dim, hidden]   (kv_dim = n_kv_heads * head_dim)
///   W_v: [kv_dim, hidden]
///   W_o: [hidden, hidden]
///
/// Requirement: 5.2, 5.3
pub fn attention(
    x: &Tensor,
    tensors: &HashMap<String, Tensor>,
    layer: usize,
    config: &ModelConfig,
) -> Result<Tensor, GwenError> {
    let ctx = |op: &'static str| move |e: candle_core::Error| GwenError::InferenceError {
        layer: format!("blk.{layer}"),
        operation: op.to_string(),
        error: e.to_string(),
    };

    let w_q = get(tensors, &format!("blk.{layer}.attn_q.weight"))?;
    let w_k = get(tensors, &format!("blk.{layer}.attn_k.weight"))?;
    let w_v = get(tensors, &format!("blk.{layer}.attn_v.weight"))?;
    let w_o = get(tensors, &format!("blk.{layer}.attn_output.weight"))?;

    // x: [seq_len, hidden]
    // Linear projections — weights are stored transposed in GGUF (output, input).
    let q = x.matmul(&w_q.t().map_err(ctx("q_proj_t"))?).map_err(ctx("q_proj"))?;
    let k = x.matmul(&w_k.t().map_err(ctx("k_proj_t"))?).map_err(ctx("k_proj"))?;
    let v = x.matmul(&w_v.t().map_err(ctx("v_proj_t"))?).map_err(ctx("v_proj"))?;

    let n_heads = config.n_heads as usize;
    let n_kv_heads = config.n_kv_heads as usize;
    let head_dim = config.hidden_size as usize / n_heads;
    let seq_len = x.dim(0).map_err(ctx("seq_len"))?;

    // Reshape to [seq_len, n_heads, head_dim] then transpose to [n_heads, seq_len, head_dim].
    let q = q.reshape((seq_len, n_heads, head_dim))
        .map_err(ctx("q_reshape"))?
        .transpose(0, 1)
        .map_err(ctx("q_transpose"))?;

    let k = k.reshape((seq_len, n_kv_heads, head_dim))
        .map_err(ctx("k_reshape"))?
        .transpose(0, 1)
        .map_err(ctx("k_transpose"))?;

    let v = v.reshape((seq_len, n_kv_heads, head_dim))
        .map_err(ctx("v_reshape"))?
        .transpose(0, 1)
        .map_err(ctx("v_transpose"))?;

    // GQA: repeat K/V heads when n_kv_heads < n_heads.
    let (k, v) = if n_kv_heads < n_heads {
        let repeat = n_heads / n_kv_heads;
        let k = k.repeat((1, repeat, 1)).map_err(ctx("k_repeat"))?
            .reshape((n_heads, seq_len, head_dim)).map_err(ctx("k_gqa_reshape"))?;
        let v = v.repeat((1, repeat, 1)).map_err(ctx("v_repeat"))?
            .reshape((n_heads, seq_len, head_dim)).map_err(ctx("v_gqa_reshape"))?;
        (k, v)
    } else {
        (k, v)
    };

    // scores = Q @ K^T / sqrt(head_dim)   → [n_heads, seq_len, seq_len]
    let scale = (head_dim as f64).sqrt();
    let scores = q.matmul(&k.transpose(1, 2).map_err(ctx("k_t"))?)
        .map_err(ctx("scores"))?
        .affine(1.0 / scale, 0.0)
        .map_err(ctx("scale"))?;

    // Softmax over last dim.
    let weights = softmax(&scores, candle_core::D::Minus1).map_err(ctx("softmax"))?;

    // context = weights @ V   → [n_heads, seq_len, head_dim]
    let context = weights.matmul(&v).map_err(ctx("context"))?;

    // Transpose back to [seq_len, n_heads, head_dim] and reshape to [seq_len, hidden].
    let context = context
        .transpose(0, 1).map_err(ctx("context_t"))?
        .contiguous().map_err(ctx("contiguous"))?
        .reshape((seq_len, config.hidden_size as usize)).map_err(ctx("merge_heads"))?;

    // Output projection.
    context.matmul(&w_o.t().map_err(ctx("o_proj_t"))?).map_err(ctx("o_proj"))
}

// ── 11.1 MLP with SwiGLU ─────────────────────────────────────────────────────

/// Feed-forward MLP block using SwiGLU activation.
///
/// Formula: `out = (silu(gate) * up) @ W_down`
///
/// Requirement: 5.3
pub fn mlp(
    x: &Tensor,
    tensors: &HashMap<String, Tensor>,
    layer: usize,
) -> Result<Tensor, GwenError> {
    let ctx = |op: &'static str| move |e: candle_core::Error| GwenError::InferenceError {
        layer: format!("blk.{layer}.mlp"),
        operation: op.to_string(),
        error: e.to_string(),
    };

    let w_gate = get(tensors, &format!("blk.{layer}.ffn_gate.weight"))?;
    let w_up   = get(tensors, &format!("blk.{layer}.ffn_up.weight"))?;
    let w_down = get(tensors, &format!("blk.{layer}.ffn_down.weight"))?;

    let gate = x.matmul(&w_gate.t().map_err(ctx("gate_t"))?).map_err(ctx("gate"))?;
    let up   = x.matmul(&w_up.t().map_err(ctx("up_t"))?).map_err(ctx("up"))?;

    // SwiGLU: silu(gate) * up
    let activated = ops::silu(&gate).map_err(ctx("silu"))?.mul(&up).map_err(ctx("swiglu"))?;

    activated.matmul(&w_down.t().map_err(ctx("down_t"))?).map_err(ctx("down"))
}

// ── 12.1 Forward pass ────────────────────────────────────────────────────────

/// Full LLaMA-family forward pass: embeddings → N transformer layers → logits.
///
/// `input_ids` is a 1-D tensor of token IDs `[seq_len]`.
/// Returns a 2-D logits tensor `[seq_len, vocab_size]` (no KV cache).
///
/// Requirements: 5.1, 5.2, 5.3, 5.4, 5.6
pub fn forward(
    input_ids: &Tensor,
    tensors: &HashMap<String, Tensor>,
    config: &ModelConfig,
) -> Result<Tensor, GwenError> {
    let ctx = |layer: &str, op: &'static str| {
        let l = layer.to_string();
        move |e: candle_core::Error| GwenError::InferenceError {
            layer: l.clone(),
            operation: op.to_string(),
            error: e.to_string(),
        }
    };

    // Embedding lookup: [seq_len] → [seq_len, hidden_size]
    let embd_w = get(tensors, "token_embd.weight")?;
    let mut hidden = embd_w
        .embedding(input_ids)
        .map_err(ctx("embedding", "embedding_lookup"))?;

    // Transformer layers.
    for i in 0..config.n_layers as usize {
        let layer_tag = format!("blk.{i}");

        // Pre-attention RMSNorm.
        let attn_norm_w = get(tensors, &format!("{layer_tag}.attn_norm.weight"))?;
        let normed = rms_norm(&hidden, &attn_norm_w, config.rms_norm_eps)?;

        // Attention + residual.
        let attn_out = attention(&normed, tensors, i, config)?;
        hidden = hidden.add(&attn_out).map_err(ctx(&layer_tag, "attn_residual"))?;

        // Pre-MLP RMSNorm.
        let ffn_norm_w = get(tensors, &format!("{layer_tag}.ffn_norm.weight"))?;
        let normed = rms_norm(&hidden, &ffn_norm_w, config.rms_norm_eps)?;

        // MLP + residual.
        let mlp_out = mlp(&normed, tensors, i)?;
        hidden = hidden.add(&mlp_out).map_err(ctx(&layer_tag, "mlp_residual"))?;
    }

    // Final RMSNorm.
    let output_norm_w = get(tensors, "output_norm.weight")?;
    let hidden = rms_norm(&hidden, &output_norm_w, config.rms_norm_eps)?;

    // lm_head projection: [seq_len, hidden] → [seq_len, vocab_size]
    let lm_head = get(tensors, "output.weight")?;
    hidden
        .matmul(&lm_head.t().map_err(ctx("lm_head", "t"))?)
        .map_err(ctx("lm_head", "matmul"))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    fn cpu() -> Device { Device::Cpu }

    fn randn(shape: &[usize], device: &Device) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.01) - (n as f32 * 0.005)).collect();
        Tensor::from_vec(data, shape, device).unwrap()
    }

    fn ones(shape: &[usize], device: &Device) -> Tensor {
        Tensor::ones(shape, candle_core::DType::F32, device).unwrap()
    }

    fn zeros(shape: &[usize], device: &Device) -> Tensor {
        Tensor::zeros(shape, candle_core::DType::F32, device).unwrap()
    }

    // ── 9.2 RMSNorm tests ────────────────────────────────────────────────────

    #[test]
    fn rms_norm_output_shape_matches_input() {
        let x = randn(&[4, 8], &cpu());
        let w = ones(&[8], &cpu());
        let out = rms_norm(&x, &w, 1e-5).unwrap();
        assert_eq!(out.dims(), x.dims());
    }

    #[test]
    fn rms_norm_all_zeros_input_with_eps_does_not_produce_nan() {
        // All-zero input → mean(x²)=0 → rms=sqrt(eps) → finite output.
        let x = zeros(&[2, 4], &cpu());
        let w = ones(&[4], &cpu());
        let out = rms_norm(&x, &w, 1e-5).unwrap();
        let vals: Vec<f32> = out.to_vec2::<f32>().unwrap().into_iter().flatten().collect();
        assert!(vals.iter().all(|v| v.is_finite()), "expected all finite, got {vals:?}");
    }

    #[test]
    fn rms_norm_unit_weight_normalises_variance() {
        // With unit weight the output RMS of each row should be ≈ 1.
        let seq = 1usize;
        let hidden = 64usize;
        // Use values spread enough to give non-trivial RMS.
        let data: Vec<f32> = (0..hidden).map(|i| i as f32 - 32.0).collect();
        let x = Tensor::from_vec(data, (seq, hidden), &cpu()).unwrap();
        let w = ones(&[hidden], &cpu());
        let out = rms_norm(&x, &w, 1e-5).unwrap();
        let vals: Vec<f32> = out.flatten_all().unwrap().to_vec1().unwrap();
        let rms: f32 = (vals.iter().map(|v| v * v).sum::<f32>() / vals.len() as f32).sqrt();
        assert!((rms - 1.0).abs() < 0.01, "expected RMS ≈ 1.0, got {rms}");
    }

    // ── 11.2 MLP / SwiGLU tests ──────────────────────────────────────────────

    #[test]
    fn mlp_output_shape_matches_hidden_size() {
        let seq = 3usize;
        let hidden = 8usize;
        let intermediate = 16usize;

        let x = randn(&[seq, hidden], &cpu());

        let mut tensors = HashMap::new();
        // Weight shapes in GGUF: [out, in]  (matmul with W.T)
        tensors.insert("blk.0.ffn_gate.weight".to_string(), randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_up.weight".to_string(),   randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_down.weight".to_string(), randn(&[hidden, intermediate], &cpu()));

        let out = mlp(&x, &tensors, 0).unwrap();
        assert_eq!(out.dims(), &[seq, hidden]);
    }

    #[test]
    fn mlp_missing_weight_returns_inference_error() {
        let x = randn(&[1, 8], &cpu());
        let tensors = HashMap::new(); // empty
        let err = mlp(&x, &tensors, 0).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }

    // ── 10.2 Attention tests ─────────────────────────────────────────────────

    #[test]
    fn attention_output_shape_matches_input() {
        let seq = 2usize;
        let hidden = 8usize;
        let n_heads = 2usize;
        let head_dim = hidden / n_heads;

        let cfg = ModelConfig {
            architecture: "llama".to_string(),
            n_layers: 1,
            hidden_size: hidden as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_heads as u32, // MHA (no GQA)
            intermediate_size: 16,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };

        let x = randn(&[seq, hidden], &cpu());
        let mut tensors = HashMap::new();
        // [out_dim, in_dim] layout
        tensors.insert("blk.0.attn_q.weight".to_string(),      randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_k.weight".to_string(),      randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_v.weight".to_string(),      randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_output.weight".to_string(), randn(&[hidden, hidden], &cpu()));

        let out = attention(&x, &tensors, 0, &cfg).unwrap();
        assert_eq!(out.dims(), &[seq, hidden]);
    }

    #[test]
    fn attention_missing_weight_returns_inference_error() {
        let cfg = ModelConfig {
            architecture: "llama".to_string(),
            n_layers: 1,
            hidden_size: 8,
            n_heads: 2,
            n_kv_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let x = randn(&[1, 8], &cpu());
        let tensors = HashMap::new();
        let err = attention(&x, &tensors, 0, &cfg).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }

    // ── 12.2 Forward pass tests ───────────────────────────────────────────────

    #[test]
    fn forward_pass_produces_correct_logit_shape() {
        // Tiny 1-layer model: hidden=8, heads=2, intermediate=16, vocab=32.
        let hidden = 8usize;
        let n_heads = 2usize;
        let intermediate = 16usize;
        let vocab = 32usize;
        let seq = 1usize;

        let cfg = ModelConfig {
            architecture: "llama".to_string(),
            n_layers: 1,
            hidden_size: hidden as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_heads as u32,
            intermediate_size: intermediate as u32,
            vocab_size: vocab as u32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(),    randn(&[vocab, hidden], &cpu()));
        tensors.insert("blk.0.attn_norm.weight".to_string(), ones(&[hidden], &cpu()));
        tensors.insert("blk.0.attn_q.weight".to_string(),    randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_k.weight".to_string(),    randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_v.weight".to_string(),    randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_output.weight".to_string(), randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.ffn_norm.weight".to_string(),  ones(&[hidden], &cpu()));
        tensors.insert("blk.0.ffn_gate.weight".to_string(),  randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_up.weight".to_string(),    randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_down.weight".to_string(),  randn(&[hidden, intermediate], &cpu()));
        tensors.insert("output_norm.weight".to_string(),     ones(&[hidden], &cpu()));
        tensors.insert("output.weight".to_string(),          randn(&[vocab, hidden], &cpu()));

        let input_ids = Tensor::from_vec(vec![0u32], seq, &cpu()).unwrap();
        let logits = forward(&input_ids, &tensors, &cfg).unwrap();
        assert_eq!(logits.dims(), &[seq, vocab], "logits shape mismatch");
    }

    #[test]
    fn forward_pass_does_not_panic_with_valid_input() {
        // Identical to above — confirms no panic on the happy path.
        let hidden = 8usize;
        let n_heads = 2usize;
        let intermediate = 16usize;
        let vocab = 32usize;

        let cfg = ModelConfig {
            architecture: "llama".to_string(),
            n_layers: 1,
            hidden_size: hidden as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_heads as u32,
            intermediate_size: intermediate as u32,
            vocab_size: vocab as u32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };

        let mut tensors = HashMap::new();
        tensors.insert("token_embd.weight".to_string(),       randn(&[vocab, hidden], &cpu()));
        tensors.insert("blk.0.attn_norm.weight".to_string(),  ones(&[hidden], &cpu()));
        tensors.insert("blk.0.attn_q.weight".to_string(),     randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_k.weight".to_string(),     randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_v.weight".to_string(),     randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.attn_output.weight".to_string(),randn(&[hidden, hidden], &cpu()));
        tensors.insert("blk.0.ffn_norm.weight".to_string(),   ones(&[hidden], &cpu()));
        tensors.insert("blk.0.ffn_gate.weight".to_string(),   randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_up.weight".to_string(),     randn(&[intermediate, hidden], &cpu()));
        tensors.insert("blk.0.ffn_down.weight".to_string(),   randn(&[hidden, intermediate], &cpu()));
        tensors.insert("output_norm.weight".to_string(),      ones(&[hidden], &cpu()));
        tensors.insert("output.weight".to_string(),           randn(&[vocab, hidden], &cpu()));

        let input_ids = Tensor::from_vec(vec![7u32, 3u32], 2usize, &cpu()).unwrap();
        let result = forward(&input_ids, &tensors, &cfg);
        assert!(result.is_ok(), "forward pass returned error: {:?}", result.err());
    }

    #[test]
    fn forward_pass_missing_tensor_returns_inference_error() {
        let cfg = ModelConfig {
            architecture: "llama".to_string(),
            n_layers: 1,
            hidden_size: 8,
            n_heads: 2,
            n_kv_heads: 2,
            intermediate_size: 16,
            vocab_size: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
        };
        let tensors = HashMap::new(); // empty — will fail on embedding lookup
        let input_ids = Tensor::from_vec(vec![0u32], 1usize, &cpu()).unwrap();
        let err = forward(&input_ids, &tensors, &cfg).unwrap_err();
        assert!(matches!(err, GwenError::InferenceError { .. }));
    }
}
