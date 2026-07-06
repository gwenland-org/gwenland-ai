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
    /// SwiGLU gate + up projections, `[hidden_dim, dim]` each (interleaved
    /// when quantized — see [`GateUp`]).
    pub gate_up: GateUp,
    /// Down projection, `[dim, hidden_dim]`.
    pub w_down: WeightMatrix,
}

/// A fully loaded model: config plus all weights.
pub struct GlprocModel {
    /// Hyperparameters.
    pub config: ModelConfig,
    /// Token embedding table, `[vocab_size, dim]`, always f32 (row lookup).
    pub token_embd: Vec<f32>,
    /// All transformer blocks, in order.
    pub layers: Vec<LayerWeights>,
    /// Final RMSNorm gain, `[dim]`.
    pub output_norm: Vec<f32>,
    /// LM head, `[vocab_size, dim]`. Tied to `token_embd` when the file
    /// has no separate `output.weight`.
    pub output: WeightMatrix,
}
