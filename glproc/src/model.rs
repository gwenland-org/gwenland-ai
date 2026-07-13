//! Model configuration and weight registry for the CPU engine.

/// How rotary position embeddings pair up dimensions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Original llama style: rotate adjacent pairs `(2i, 2i+1)`.
    Norm,
    /// GPT-NeoX style (qwen2, phi, gemma, ...): rotate `(i, i + dim/2)`.
    Neox,
}

/// Hyperparameters of a loaded transformer, read from GGUF metadata.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// Architecture string from `general.architecture` (e.g. `"llama"`).
    pub arch: String,
    /// Embedding width (`{arch}.embedding_length`).
    pub dim: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (< `n_heads` under GQA).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN inner width (`{arch}.feed_forward_length`).
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length the model was trained for.
    pub max_seq: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency (`{arch}.rope.freq_base`, default 10000).
    pub rope_freq_base: f32,
    /// RoPE dimension pairing convention.
    pub rope_style: RopeStyle,
}

use crate::kernels::bridge::QuantFormat;

/// One weight matrix, either dequantized or kept in its quantized on-disk
/// form for the Bridge-ing matvec path.
///
/// Quantized matrices stay as raw GGML blocks: the bridge dequantizes one
/// block at a time into an L1-resident stack buffer during the matvec, so
/// the f32 form never exists in RAM. That keeps the per-token working set
/// at the quantized size (e.g. 7× smaller for Q4_K) — decode is
/// memory-bandwidth-bound, so a smaller working set is proportionally
/// faster.
#[derive(Clone)]
pub enum WeightMatrix {
    /// Dense f32, row-major `[out_features, in_features]`.
    F32(Vec<f32>),
    /// Raw quantized blocks, rows contiguous; `in_features` is a multiple
    /// of the format's block size.
    Quant(QuantFormat, Vec<u8>),
}

impl WeightMatrix {
    /// Borrow the f32 payload, if dense.
    pub fn as_f32(&self) -> Option<&[f32]> {
        match self {
            WeightMatrix::F32(w) => Some(w),
            WeightMatrix::Quant(..) => None,
        }
    }
}

/// The SwiGLU gate and up projections, `[hidden_dim, dim]` each.
///
/// When both share a bridge-supported quantized format the loader
/// interleaves them row-wise — `[gate row 0][up row 0][gate row 1]…` — so
/// the fused SwiGLU matvec streams ONE contiguous region per thread.
/// Two separate matrices would give each thread two DRAM streams megabytes
/// apart; single-channel DDR4 pays for that in page locality.
pub enum GateUp {
    /// Row-interleaved quantized pair; row `o` occupies
    /// `[o * 2 * row_bytes, (o + 1) * 2 * row_bytes)` (gate half, up half).
    FusedQuant(QuantFormat, Vec<u8>),
    /// Separate matrices (f32 fallback or mismatched formats).
    Split(WeightMatrix, WeightMatrix),
}

/// The Q, K and V projections. When all three share a bridge-supported
/// quantized format the loader stacks them into one matrix —
/// `[q rows][k rows][v rows]`, `[(q_dim + 2*kv_dim), dim]` — so a single
/// pool dispatch computes the whole projection into one output buffer.
/// The K/V matrices are tiny under GQA (128 rows here), so as separate
/// matvecs their dispatch overhead is proportionally large.
pub enum QkvWeights {
    /// Stacked rows, quantized.
    FusedQuant(QuantFormat, Vec<u8>),
    /// Separate q, k, v matrices (f32 fallback or mismatched formats).
    Split(WeightMatrix, WeightMatrix, WeightMatrix),
}

/// A block's feed-forward network: one dense SwiGLU, or a routed mixture.
///
/// Per layer, not per model: Qwen3-MoE may interleave dense and MoE blocks,
/// and the loader decides from the presence of `ffn_gate_exps` tensors.
pub enum FfnLayer {
    /// Dense SwiGLU — every token through the same weights.
    Dense {
        /// Gate + up projections, `[hidden_dim, dim]` each (row-interleaved
        /// when quantized — see [`GateUp`]).
        gate_up: GateUp,
        /// Down projection, `[dim, hidden_dim]`.
        w_down: WeightMatrix,
    },
    /// Routed mixture of experts. Boxed: an `MoELayer` owns `num_experts`
    /// weight sets, so inlining it would bloat every dense layer's
    /// `LayerWeights` by the size of the largest MoE variant.
    MoE(Box<crate::moe::MoELayer>),
}

/// Weights of a single transformer block.
///
/// Matrices use GGUF layout `[out_features, in_features]`, row-major.
/// Norm gains and biases are always small and stay f32.
pub struct LayerWeights {
    /// Pre-attention RMSNorm gain, `[dim]`.
    pub attn_norm: Vec<f32>,
    /// Q, K and V projections (stacked when quantized — see [`QkvWeights`]).
    pub qkv: QkvWeights,
    /// Attention output projection, `[dim, n_heads * head_dim]`.
    pub wo: WeightMatrix,
    /// Optional query bias (qwen2-style models).
    pub bq: Option<Vec<f32>>,
    /// Optional key bias.
    pub bk: Option<Vec<f32>>,
    /// Optional value bias.
    pub bv: Option<Vec<f32>>,
    /// Optional per-head query RMSNorm gain, `[head_dim]` (qwen3-style).
    pub q_norm: Option<Vec<f32>>,
    /// Optional per-head key RMSNorm gain, `[head_dim]` (qwen3-style).
    pub k_norm: Option<Vec<f32>>,
    /// Pre-FFN RMSNorm gain, `[dim]`.
    pub ffn_norm: Vec<f32>,
    /// The feed-forward network: dense SwiGLU or routed experts.
    pub ffn: FfnLayer,
}

/// A fully loaded model: config plus all weights.
pub struct GlprocModel {
    /// Hyperparameters.
    pub config: ModelConfig,
    /// Token embedding table, `[vocab_size, dim]`. Kept quantized (Q8_0)
    /// when possible — lookups dequantize one row on demand, which costs
    /// well under a microsecond and saves the ~4x f32 blow-up in RAM
    /// (~500 MB on 150k-vocab models) plus its dequantization at load.
    pub token_embd: WeightMatrix,
    /// All transformer blocks, in order.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain, `[dim]`.
    pub output_norm: Vec<f32>,
    /// LM head, `[vocab_size, dim]`. Tied to `token_embd` when the file
    /// has no separate `output.weight`.
    pub output: WeightMatrix,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_with_embd(token_embd: WeightMatrix, vocab: usize, dim: usize) -> GlprocModel {
        GlprocModel {
            config: ModelConfig {
                arch: "llama".into(),
                dim,
                n_layers: 0,
                n_heads: 1,
                n_kv_heads: 1,
                head_dim: dim,
                hidden_dim: dim,
                vocab_size: vocab,
                max_seq: 8,
                rms_eps: 1e-5,
                rope_freq_base: 10_000.0,
                rope_style: RopeStyle::Norm,
            },
            token_embd,
            layers: Vec::new(),
            output_norm: vec![1.0; dim],
            output: WeightMatrix::F32(vec![0.0; vocab * dim]),
        }
    }

    #[test]
    fn quantized_embedding_row_matches_f32() {
        let (vocab, dim) = (5usize, 64usize);
        let table: Vec<f32> = (0..vocab * dim).map(|i| (i % 23) as f32 * 0.5 - 5.0).collect();
        // Quantize the table to Q8_0 and compare row lookups.
        let q = crate::kernels::dequant::q8_0::scalar::quantize(&table);
        let mf = model_with_embd(WeightMatrix::F32(table.clone()), vocab, dim);
        let mq = model_with_embd(WeightMatrix::Quant(QuantFormat::Q8_0, q), vocab, dim);
        let mut a = vec![0f32; dim];
        let mut b = vec![0f32; dim];
        for t in 0..vocab as u32 {
            mf.embed_into(t, &mut a).unwrap();
            mq.embed_into(t, &mut b).unwrap();
            for (x, y) in a.iter().zip(&b) {
                // One int8 quantization step of error at most: values span
                // [-5, 6], so scale ≤ 6/127 and error ≤ half a step.
                assert!((x - y).abs() <= 6.0 / 127.0 * 0.5 + 1e-6, "{x} vs {y}");
            }
        }
        // Out-of-range ids error on both representations.
        assert!(mf.embed_into(vocab as u32, &mut a).is_err());
        assert!(mq.embed_into(9999, &mut b).is_err());
    }
}

impl GlprocModel {
    /// Copy `token`'s embedding row into `out` (`[dim]`), dequantizing on
    /// the fly when the table is stored quantized.
    pub fn embed_into(&self, token: u32, out: &mut [f32]) -> Result<(), glcore::GlError> {
        let dim = self.config.dim;
        debug_assert_eq!(out.len(), dim);
        let row = token as usize;
        if row >= self.config.vocab_size {
            return Err(glcore::GlError::Engine(format!(
                "token id {token} out of embedding range"
            )));
        }
        match &self.token_embd {
            WeightMatrix::F32(v) => out.copy_from_slice(&v[row * dim..(row + 1) * dim]),
            WeightMatrix::Quant(QuantFormat::Q8_0, b) => {
                // Q8_0 row: dim/32 blocks of [f16 scale][32 x i8].
                let row_bytes = dim / 32 * 34;
                let r = &b[row * row_bytes..(row + 1) * row_bytes];
                for (j, block) in r.chunks_exact(34).enumerate() {
                    let d = glcore::format::gguf::f16_to_f32(u16::from_le_bytes([
                        block[0], block[1],
                    ]));
                    for (i, &q) in block[2..34].iter().enumerate() {
                        out[j * 32 + i] = d * (q as i8) as f32;
                    }
                }
            }
            // The loader only stores the table as F32 or Q8_0.
            WeightMatrix::Quant(fmt, _) => {
                return Err(glcore::GlError::Engine(format!(
                    "unsupported embedding format {fmt:?}"
                )))
            }
        }
        Ok(())
    }
}
