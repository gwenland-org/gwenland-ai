// engine/inference/candle_ggqr/mod.rs — GgqrCandleBackend: GGQR dequant + Candle forward pass.
//
// Feature-gated behind `candle-backend`. All items in this module are only
// compiled when that feature is enabled.
//
// Requirements: 15.1 – 15.5 (feature gate), 1.2/1.3/1.5 (ModelConfig)

#![cfg(feature = "candle-backend")]

use serde::{Deserialize, Serialize};

pub mod gguf;
pub use gguf::{validate_gguf, extract_architecture, build_model_config};

pub mod dequant;
pub use dequant::dequantize_tensor;

pub mod tensor;
pub use tensor::vec_to_tensor;

pub mod backend;
pub use backend::GgqrCandleBackend;

pub mod forward;
pub use forward::{rms_norm, attention, mlp, forward};

pub mod sampling;
pub use sampling::{greedy_sample, top_p_sample, sample_token};

pub mod generation;
pub use generation::{GenerationState, generate_stream, generate_collect, make_stream_pinned};

// ── ModelConfig ───────────────────────────────────────────────────────────────

/// Architecture metadata extracted from a GGUF file's key-value store.
///
/// Every field maps directly to a standard GGUF metadata key
/// (e.g. `llama.block_count`, `llama.embedding_length`).
/// Unsupported architectures are rejected before this struct is constructed.
///
/// Requirements: 1.2, 1.3, 1.5
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// GGUF `general.architecture` value — e.g. `"llama"`, `"qwen2"`, `"phi3"`.
    pub architecture: String,

    /// Number of transformer layers (`<arch>.block_count`).
    pub n_layers: u32,

    /// Hidden embedding dimension (`<arch>.embedding_length`).
    pub hidden_size: u32,

    /// Number of query attention heads (`<arch>.attention.head_count`).
    pub n_heads: u32,

    /// Number of key/value attention heads (`<arch>.attention.head_count_kv`).
    /// Equals `n_heads` for MHA; smaller for GQA/MQA.
    pub n_kv_heads: u32,

    /// Feed-forward intermediate size (`<arch>.feed_forward_length`).
    pub intermediate_size: u32,

    /// Vocabulary size (`tokenizer.ggml.token_type_count` or `<arch>.vocab_size`).
    pub vocab_size: u32,

    /// RMS normalisation epsilon (`<arch>.attention.layer_norm_rms_epsilon`).
    pub rms_norm_eps: f32,

    /// RoPE base frequency (`<arch>.rope.freq_base`). Defaults to 10 000.0.
    pub rope_theta: f32,
}
