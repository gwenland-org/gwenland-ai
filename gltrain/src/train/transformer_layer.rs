//! Training-side transformer layer pieces.
//!
//! The functions in this module operate on one dequantized base layer at a
//! time. Base weights stay frozen while the seven projection-specific LoRA
//! adapters remain in Candle's autograd graph.

use anyhow::{Context, Result, bail};
use candle_core::Tensor;
use candle_nn::ops::silu;

use crate::engine::transformer_ops::{
    apply_rope, causal_scaled_dot_product_attention, repeat_kv, rms_norm,
};

/// Architecture values required by a single attention block.
#[derive(Debug, Clone, Copy)]
pub struct AttentionConfig {
    pub hidden_size: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub rope_theta: f32,
    pub rms_norm_eps: f32,
}

impl AttentionConfig {
    pub fn head_dim(self) -> Result<usize> {
        if self.n_heads == 0 || self.n_kv_heads == 0 {
            bail!("attention head counts must be positive");
        }
        if self.hidden_size % self.n_heads != 0 {
            bail!(
                "hidden_size {} is not divisible by n_heads {}",
                self.hidden_size,
                self.n_heads
            );
        }
        if self.n_heads % self.n_kv_heads != 0 {
            bail!(
                "n_heads {} is not divisible by n_kv_heads {}",
                self.n_heads,
                self.n_kv_heads
            );
        }
        Ok(self.hidden_size / self.n_heads)
    }
}

/// Architecture values required by a complete transformer layer.
#[derive(Debug, Clone, Copy)]
pub struct TransformerLayerConfig {
    pub attention: AttentionConfig,
    pub intermediate_size: usize,
}

impl TransformerLayerConfig {
    pub fn validate(self) -> Result<()> {
        self.attention.head_dim()?;
        if self.intermediate_size == 0 {
            bail!("intermediate_size must be positive");
        }
        Ok(())
    }
}

/// Frozen weights for the pre-normalized attention sub-block.
pub struct AttentionWeights<'a> {
    pub attn_norm: &'a Tensor,
    pub q_proj: &'a Tensor,
    pub k_proj: &'a Tensor,
    pub v_proj: &'a Tensor,
    pub o_proj: &'a Tensor,
    /// Optional Qwen3 per-head query RMSNorm weight.
    pub q_norm: Option<&'a Tensor>,
    /// Optional Qwen3 per-head key RMSNorm weight.
    pub k_norm: Option<&'a Tensor>,
}

/// One LoRA adapter in native linear dimensions.
pub struct ProjectionLora<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    pub scale: f32,
}

/// Projection-matched LoRA adapters for attention.
#[derive(Default)]
pub struct AttentionLoras<'a> {
    pub q_proj: Option<ProjectionLora<'a>>,
    pub k_proj: Option<ProjectionLora<'a>>,
    pub v_proj: Option<ProjectionLora<'a>>,
    pub o_proj: Option<ProjectionLora<'a>>,
}

/// Frozen weights for the pre-normalized SwiGLU sub-block.
pub struct MlpWeights<'a> {
    pub ffn_norm: &'a Tensor,
    pub gate_proj: &'a Tensor,
    pub up_proj: &'a Tensor,
    pub down_proj: &'a Tensor,
}

/// Projection-matched LoRA adapters for the SwiGLU sub-block.
#[derive(Default)]
pub struct MlpLoras<'a> {
    pub gate_proj: Option<ProjectionLora<'a>>,
    pub up_proj: Option<ProjectionLora<'a>>,
    pub down_proj: Option<ProjectionLora<'a>>,
}

/// Frozen weights for one complete transformer layer.
pub struct TransformerLayerWeights<'a> {
    pub attention: AttentionWeights<'a>,
    pub mlp: MlpWeights<'a>,
}

/// All seven projection-specific LoRA adapters for one layer.
#[derive(Default)]
pub struct TransformerLayerLoras<'a> {
    pub attention: AttentionLoras<'a>,
    pub mlp: MlpLoras<'a>,
}

/// Linear projection for `[batch, seq, in]` with a frozen `[out, in]` base
/// weight and an optional LoRA delta.
pub fn linear_with_lora(
    x: &Tensor,
    weight: &Tensor,
    lora: Option<&ProjectionLora<'_>>,
) -> Result<Tensor> {
    let (batch, seq_len, input_dim) = x.dims3().context("linear input must be 3-D")?;
    let (output_dim, weight_input_dim) = weight.dims2().context("linear weight must be 2-D")?;
    if input_dim != weight_input_dim {
        bail!("linear input dim {input_dim} does not match weight input dim {weight_input_dim}");
    }

    let flat = x
        .reshape((batch * seq_len, input_dim))
        .context("flatten linear input")?;
    let mut output = flat
        .matmul(&weight.t().context("transpose base weight")?)
        .context("base projection")?;

    if let Some(lora) = lora {
        let (rank, a_input_dim) = lora.a.dims2().context("LoRA A must be 2-D")?;
        let (b_output_dim, b_rank) = lora.b.dims2().context("LoRA B must be 2-D")?;
        if a_input_dim != input_dim || b_rank != rank || b_output_dim != output_dim {
            bail!(
                "LoRA shape mismatch: input={input_dim}, output={output_dim}, A={:?}, B={:?}",
                lora.a.dims(),
                lora.b.dims()
            );
        }
        let delta = flat
            .matmul(&lora.a.t().context("transpose LoRA A")?)
            .context("LoRA A projection")?
            .matmul(&lora.b.t().context("transpose LoRA B")?)
            .context("LoRA B projection")?
            .affine(lora.scale as f64, 0.0)
            .context("scale LoRA delta")?;
        output = (output + delta).context("add LoRA delta")?;
    }

    output
        .reshape((batch, seq_len, output_dim))
        .context("restore linear output shape")
}

/// Attention sub-block:
///
/// RMSNorm -> Q/K/V (+ matched LoRA) -> optional Q/K norm -> RoPE -> GQA ->
/// causal attention -> output projection (+ LoRA) -> residual.
pub fn attention_forward(
    input: &Tensor,
    weights: &AttentionWeights<'_>,
    loras: &AttentionLoras<'_>,
    config: AttentionConfig,
    position_offset: usize,
) -> Result<Tensor> {
    let (batch, seq_len, hidden_size) = input
        .dims3()
        .context("attention input must be [batch, seq, hidden]")?;
    if hidden_size != config.hidden_size {
        bail!(
            "attention input hidden size {hidden_size} does not match config {}",
            config.hidden_size
        );
    }
    let head_dim = config.head_dim()?;

    let normed =
        rms_norm(input, weights.attn_norm, config.rms_norm_eps).context("pre-attention RMSNorm")?;
    let query = linear_with_lora(&normed, weights.q_proj, loras.q_proj.as_ref())
        .context("query projection")?;
    let key = linear_with_lora(&normed, weights.k_proj, loras.k_proj.as_ref())
        .context("key projection")?;
    let value = linear_with_lora(&normed, weights.v_proj, loras.v_proj.as_ref())
        .context("value projection")?;

    let query = query
        .reshape((batch, seq_len, config.n_heads, head_dim))
        .context("reshape query heads")?
        .transpose(1, 2)
        .context("transpose query heads")?;
    let key = key
        .reshape((batch, seq_len, config.n_kv_heads, head_dim))
        .context("reshape key heads")?
        .transpose(1, 2)
        .context("transpose key heads")?;
    let value = value
        .reshape((batch, seq_len, config.n_kv_heads, head_dim))
        .context("reshape value heads")?
        .transpose(1, 2)
        .context("transpose value heads")?;

    let query = match weights.q_norm {
        Some(weight) => {
            rms_norm(&query, weight, config.rms_norm_eps).context("query head RMSNorm")?
        }
        None => query,
    };
    let key = match weights.k_norm {
        Some(weight) => rms_norm(&key, weight, config.rms_norm_eps).context("key head RMSNorm")?,
        None => key,
    };

    let query = apply_rope(&query, config.rope_theta, position_offset).context("query RoPE")?;
    let key = apply_rope(&key, config.rope_theta, position_offset).context("key RoPE")?;
    let key = repeat_kv(&key, config.n_heads).context("expand GQA keys")?;
    let value = repeat_kv(&value, config.n_heads).context("expand GQA values")?;

    let context = causal_scaled_dot_product_attention(&query, &key, &value)
        .context("causal scaled dot-product attention")?;
    let context = context
        .transpose(1, 2)
        .context("transpose attention context")?
        .contiguous()
        .context("contiguous attention context")?
        .reshape((batch, seq_len, config.hidden_size))
        .context("merge attention heads")?;
    let projected = linear_with_lora(&context, weights.o_proj, loras.o_proj.as_ref())
        .context("attention output projection")?;

    input.add(&projected).context("attention residual")
}

/// MLP sub-block:
///
/// RMSNorm -> gate/up projections (+ matched LoRA) -> SwiGLU -> down projection
/// (+ LoRA) -> residual.
pub fn mlp_forward(
    input: &Tensor,
    weights: &MlpWeights<'_>,
    loras: &MlpLoras<'_>,
    config: TransformerLayerConfig,
) -> Result<Tensor> {
    config.validate()?;
    let (_, _, hidden_size) = input
        .dims3()
        .context("MLP input must be [batch, seq, hidden]")?;
    if hidden_size != config.attention.hidden_size {
        bail!(
            "MLP input hidden size {hidden_size} does not match config {}",
            config.attention.hidden_size
        );
    }

    let normed = rms_norm(
        input,
        weights.ffn_norm,
        config.attention.rms_norm_eps,
    )
    .context("pre-MLP RMSNorm")?;
    let gate = linear_with_lora(&normed, weights.gate_proj, loras.gate_proj.as_ref())
        .context("MLP gate projection")?;
    let up = linear_with_lora(&normed, weights.up_proj, loras.up_proj.as_ref())
        .context("MLP up projection")?;
    if gate.dim(2)? != config.intermediate_size || up.dim(2)? != config.intermediate_size {
        bail!(
            "MLP intermediate size mismatch: expected {}, gate={}, up={}",
            config.intermediate_size,
            gate.dim(2)?,
            up.dim(2)?
        );
    }

    let activated = silu(&gate)
        .context("MLP SiLU")?
        .mul(&up)
        .context("MLP SwiGLU multiply")?;
    let projected = linear_with_lora(
        &activated,
        weights.down_proj,
        loras.down_proj.as_ref(),
    )
    .context("MLP down projection")?;
    input.add(&projected).context("MLP residual")
}

/// Complete pre-normalized transformer layer.
pub fn transformer_layer_forward(
    input: &Tensor,
    weights: &TransformerLayerWeights<'_>,
    loras: &TransformerLayerLoras<'_>,
    config: TransformerLayerConfig,
    position_offset: usize,
) -> Result<Tensor> {
    config.validate()?;
    let attention = attention_forward(
        input,
        &weights.attention,
        &loras.attention,
        config.attention,
        position_offset,
    )?;
    mlp_forward(&attention, &weights.mlp, &loras.mlp, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Var};

    fn tensor(data: Vec<f32>, shape: impl Into<candle_core::Shape>) -> Tensor {
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn deterministic(shape: &[usize], scale: f32) -> Tensor {
        let count: usize = shape.iter().product();
        let data = (0..count)
            .map(|i| ((i % 17) as f32 - 8.0) * scale)
            .collect();
        Tensor::from_vec(data, shape, &Device::Cpu).unwrap()
    }

    fn all_finite(t: &Tensor) -> bool {
        t.flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .iter()
            .all(|v| v.is_finite())
    }

    #[test]
    fn linear_with_lora_adds_known_projection_delta() {
        let x = tensor(vec![1.0, 2.0], (1, 1, 2));
        let base = Tensor::zeros((2, 2), DType::F32, &Device::Cpu).unwrap();
        let a = tensor(vec![1.0, 1.0], (1, 2));
        let b = tensor(vec![2.0, 3.0], (2, 1));
        let lora = ProjectionLora {
            a: &a,
            b: &b,
            scale: 0.5,
        };

        let output = linear_with_lora(&x, &base, Some(&lora)).unwrap();
        assert_eq!(output.to_vec3::<f32>().unwrap()[0][0], vec![3.0, 4.5]);
    }

    #[test]
    fn attention_block_gqa_shape_values_and_gradients_are_finite() {
        let config = AttentionConfig {
            hidden_size: 8,
            n_heads: 4,
            n_kv_heads: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        };
        let input = deterministic(&[2, 3, 8], 0.02);
        let input = Var::from_tensor(&input).unwrap();
        let attn_norm = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let q_proj = deterministic(&[8, 8], 0.01);
        let k_proj = deterministic(&[4, 8], 0.01);
        let v_proj = deterministic(&[4, 8], 0.01);
        let o_proj = deterministic(&[8, 8], 0.01);
        let q_norm = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let k_norm = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();

        let q_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let q_b = Var::from_tensor(&deterministic(&[8, 2], 0.01)).unwrap();
        let o_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let o_b = Var::from_tensor(&deterministic(&[8, 2], 0.01)).unwrap();
        let weights = AttentionWeights {
            attn_norm: &attn_norm,
            q_proj: &q_proj,
            k_proj: &k_proj,
            v_proj: &v_proj,
            o_proj: &o_proj,
            q_norm: Some(&q_norm),
            k_norm: Some(&k_norm),
        };
        let loras = AttentionLoras {
            q_proj: Some(ProjectionLora {
                a: q_a.as_tensor(),
                b: q_b.as_tensor(),
                scale: 0.5,
            }),
            o_proj: Some(ProjectionLora {
                a: o_a.as_tensor(),
                b: o_b.as_tensor(),
                scale: 0.5,
            }),
            ..AttentionLoras::default()
        };

        let output = attention_forward(input.as_tensor(), &weights, &loras, config, 0).unwrap();
        assert_eq!(output.dims(), &[2, 3, 8]);
        assert!(all_finite(&output));

        let grads = output
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .backward()
            .unwrap();
        for (name, variable) in [
            ("input", input.as_tensor()),
            ("q_a", q_a.as_tensor()),
            ("q_b", q_b.as_tensor()),
            ("o_a", o_a.as_tensor()),
            ("o_b", o_b.as_tensor()),
        ] {
            let grad = grads
                .get(variable)
                .unwrap_or_else(|| panic!("missing attention gradient for {name}"));
            assert!(all_finite(grad));
        }
    }

    #[test]
    fn invalid_gqa_configuration_is_rejected() {
        let config = AttentionConfig {
            hidden_size: 8,
            n_heads: 4,
            n_kv_heads: 3,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        };
        assert!(config.head_dim().is_err());
    }

    #[test]
    fn full_layer_shape_values_and_all_lora_gradients_are_finite() {
        let attention = AttentionConfig {
            hidden_size: 8,
            n_heads: 4,
            n_kv_heads: 2,
            rope_theta: 10_000.0,
            rms_norm_eps: 1e-5,
        };
        let config = TransformerLayerConfig {
            attention,
            intermediate_size: 12,
        };
        let input = Var::from_tensor(&deterministic(&[2, 3, 8], 0.02)).unwrap();
        let attn_norm = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let q_proj = deterministic(&[8, 8], 0.01);
        let k_proj = deterministic(&[4, 8], 0.01);
        let v_proj = deterministic(&[4, 8], 0.01);
        let o_proj = deterministic(&[8, 8], 0.01);
        let q_norm = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let k_norm = Tensor::ones(2, DType::F32, &Device::Cpu).unwrap();
        let ffn_norm = Tensor::ones(8, DType::F32, &Device::Cpu).unwrap();
        let gate_proj = deterministic(&[12, 8], 0.01);
        let up_proj = deterministic(&[12, 8], 0.008);
        let down_proj = deterministic(&[8, 12], 0.006);
        let weights = TransformerLayerWeights {
            attention: AttentionWeights {
                attn_norm: &attn_norm,
                q_proj: &q_proj,
                k_proj: &k_proj,
                v_proj: &v_proj,
                o_proj: &o_proj,
                q_norm: Some(&q_norm),
                k_norm: Some(&k_norm),
            },
            mlp: MlpWeights {
                ffn_norm: &ffn_norm,
                gate_proj: &gate_proj,
                up_proj: &up_proj,
                down_proj: &down_proj,
            },
        };

        let q_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let q_b = Var::from_tensor(&deterministic(&[8, 2], 0.01)).unwrap();
        let k_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let k_b = Var::from_tensor(&deterministic(&[4, 2], 0.01)).unwrap();
        let v_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let v_b = Var::from_tensor(&deterministic(&[4, 2], 0.01)).unwrap();
        let o_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let o_b = Var::from_tensor(&deterministic(&[8, 2], 0.01)).unwrap();
        let gate_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let gate_b = Var::from_tensor(&deterministic(&[12, 2], 0.01)).unwrap();
        let up_a = Var::from_tensor(&deterministic(&[2, 8], 0.01)).unwrap();
        let up_b = Var::from_tensor(&deterministic(&[12, 2], 0.01)).unwrap();
        let down_a = Var::from_tensor(&deterministic(&[2, 12], 0.01)).unwrap();
        let down_b = Var::from_tensor(&deterministic(&[8, 2], 0.01)).unwrap();
        let lora = |a, b| ProjectionLora { a, b, scale: 0.5 };
        let loras = TransformerLayerLoras {
            attention: AttentionLoras {
                q_proj: Some(lora(q_a.as_tensor(), q_b.as_tensor())),
                k_proj: Some(lora(k_a.as_tensor(), k_b.as_tensor())),
                v_proj: Some(lora(v_a.as_tensor(), v_b.as_tensor())),
                o_proj: Some(lora(o_a.as_tensor(), o_b.as_tensor())),
            },
            mlp: MlpLoras {
                gate_proj: Some(lora(gate_a.as_tensor(), gate_b.as_tensor())),
                up_proj: Some(lora(up_a.as_tensor(), up_b.as_tensor())),
                down_proj: Some(lora(down_a.as_tensor(), down_b.as_tensor())),
            },
        };

        let output =
            transformer_layer_forward(input.as_tensor(), &weights, &loras, config, 0).unwrap();
        assert_eq!(output.dims(), &[2, 3, 8]);
        assert!(all_finite(&output));
        assert_ne!(
            output.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            input
                .as_tensor()
                .flatten_all()
                .unwrap()
                .to_vec1::<f32>()
                .unwrap()
        );

        let grads = output.sqr().unwrap().mean_all().unwrap().backward().unwrap();
        for (name, variable) in [
            ("input", input.as_tensor()),
            ("q_a", q_a.as_tensor()),
            ("q_b", q_b.as_tensor()),
            ("k_a", k_a.as_tensor()),
            ("k_b", k_b.as_tensor()),
            ("v_a", v_a.as_tensor()),
            ("v_b", v_b.as_tensor()),
            ("o_a", o_a.as_tensor()),
            ("o_b", o_b.as_tensor()),
            ("gate_a", gate_a.as_tensor()),
            ("gate_b", gate_b.as_tensor()),
            ("up_a", up_a.as_tensor()),
            ("up_b", up_b.as_tensor()),
            ("down_a", down_a.as_tensor()),
            ("down_b", down_b.as_tensor()),
        ] {
            let grad = grads
                .get(variable)
                .unwrap_or_else(|| panic!("missing full-layer gradient for {name}"));
            assert!(all_finite(grad), "non-finite gradient for {name}");
        }
    }
}
