//! Loads a GGUF file into a [`GlprocModel`].
//!
//! Q4_K weight matrices are kept in their raw quantized form and dequantized
//! block-by-block inside the Bridge-ing matvec at decode time; everything
//! else is dequantized to f32 up front.

use glcore::format::gguf::{GgufDType, GgufFile, GgufValue};
use glcore::GlError;

use crate::kernels;
use crate::kernels::bridge::QuantFormat;
use crate::model::{GlprocModel, LayerWeights, ModelConfig, RopeStyle, WeightMatrix};

/// Read `{arch}.{suffix}` from metadata as u64.
fn meta_u64(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<u64> {
    gguf.get_meta(&format!("{arch}.{suffix}"))
        .and_then(GgufValue::as_u64)
}

/// Read `{arch}.{suffix}` from metadata as f32.
fn meta_f32(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<f32> {
    gguf.get_meta(&format!("{arch}.{suffix}"))
        .and_then(GgufValue::as_f32)
}

/// Dequantize any supported dtype to f32, routing the formats glcore does
/// not (or does not correctly) handle through glproc's own kernels: Q4_K
/// and Q5_0 have no glcore path, and glcore's Q6_K assumes a naive linear
/// nibble order that disagrees with real llama.cpp files.
fn dequant_any(gguf: &GgufFile, info: &glcore::format::gguf::GgufTensorInfo) -> Result<Vec<f32>, GlError> {
    match info.dtype {
        GgufDType::Q4_K => kernels::dequant_q4_k(gguf.tensor_data(info)?),
        GgufDType::Q5_0 => kernels::dequant::q5_0::scalar::run(gguf.tensor_data(info)?),
        GgufDType::Q6_K => kernels::dequant::q6_k::scalar::run(gguf.tensor_data(info)?),
        _ => gguf.dequantize(info),
    }
}

/// Dequantize a required tensor by name to f32.
fn tensor(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    dequant_any(gguf, info)
}

/// Dequantize an optional tensor (e.g. attention biases).
fn tensor_opt(gguf: &GgufFile, name: &str) -> Result<Option<Vec<f32>>, GlError> {
    match gguf.find_tensor(name) {
        Some(info) => Ok(Some(dequant_any(gguf, info)?)),
        None => Ok(None),
    }
}

/// Load a required weight matrix. Bridge-supported quantized formats stay
/// quantized (raw GGML blocks) for the bridge matvec; other dtypes are
/// dequantized to f32.
fn weight(gguf: &GgufFile, name: &str) -> Result<WeightMatrix, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    let fmt = match info.dtype {
        GgufDType::Q4_K => Some(QuantFormat::Q4K),
        GgufDType::Q5_0 => Some(QuantFormat::Q5_0),
        GgufDType::Q6_K => Some(QuantFormat::Q6K),
        GgufDType::Q8_0 => Some(QuantFormat::Q8_0),
        _ => None,
    };
    match fmt {
        // dimensions[0] is the contiguous axis = in_features; GGML packs
        // quantization blocks along it, so it is always a whole number of
        // blocks — guard anyway and fall back to f32 if not.
        Some(fmt) if info.dimensions[0] as usize % fmt.block_numel() == 0 => {
            Ok(WeightMatrix::Quant(fmt, gguf.tensor_data(info)?.to_vec()))
        }
        _ => Ok(WeightMatrix::F32(dequant_any(gguf, info)?)),
    }
}

/// Parse config and dequantize every weight of a GGUF transformer.
///
/// Supports llama-family architectures (llama, mistral, qwen2, tinyllama...)
/// that use the standard `blk.N.*` tensor naming.
pub fn load_gguf(gguf: &GgufFile) -> Result<GlprocModel, GlError> {
    let arch = gguf
        .get_meta("general.architecture")
        .and_then(GgufValue::as_str)
        .ok_or_else(|| GlError::Parse("GGUF: missing general.architecture".into()))?
        .to_string();

    let dim = meta_u64(gguf, &arch, "embedding_length")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.embedding_length")))?
        as usize;
    let n_layers = meta_u64(gguf, &arch, "block_count")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.block_count")))?
        as usize;
    let n_heads = meta_u64(gguf, &arch, "attention.head_count")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.attention.head_count")))?
        as usize;
    if dim == 0 || n_layers == 0 || n_heads == 0 {
        return Err(GlError::Parse(
            "GGUF: model dimensions must be non-zero".into(),
        ));
    }
    let n_kv_heads =
        meta_u64(gguf, &arch, "attention.head_count_kv").unwrap_or(n_heads as u64) as usize;
    let head_dim =
        meta_u64(gguf, &arch, "attention.key_length").unwrap_or((dim / n_heads) as u64) as usize;
    let hidden_dim = meta_u64(gguf, &arch, "feed_forward_length")
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing {arch}.feed_forward_length")))?
        as usize;
    let max_seq = meta_u64(gguf, &arch, "context_length").unwrap_or(2048) as usize;
    let rms_eps = meta_f32(gguf, &arch, "attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
    let rope_freq_base = meta_f32(gguf, &arch, "rope.freq_base").unwrap_or(10_000.0);
    // Original-llama models rotate adjacent dim pairs; most newer archs
    // (qwen2, phi3, gemma, stablelm) use the NeoX half-split convention.
    let rope_style = match arch.as_str() {
        "llama" | "llama2" | "minicpm" => RopeStyle::Norm,
        _ => RopeStyle::Neox,
    };

    let token_embd = tensor(gguf, "token_embd.weight")?;
    let vocab_size = token_embd.len() / dim;

    let mut layers = Vec::with_capacity(n_layers);
    for i in 0..n_layers {
        layers.push(LayerWeights {
            attn_norm: tensor(gguf, &format!("blk.{i}.attn_norm.weight"))?,
            wq: weight(gguf, &format!("blk.{i}.attn_q.weight"))?,
            wk: weight(gguf, &format!("blk.{i}.attn_k.weight"))?,
            wv: weight(gguf, &format!("blk.{i}.attn_v.weight"))?,
            wo: weight(gguf, &format!("blk.{i}.attn_output.weight"))?,
            bq: tensor_opt(gguf, &format!("blk.{i}.attn_q.bias"))?,
            bk: tensor_opt(gguf, &format!("blk.{i}.attn_k.bias"))?,
            bv: tensor_opt(gguf, &format!("blk.{i}.attn_v.bias"))?,
            q_norm: tensor_opt(gguf, &format!("blk.{i}.attn_q_norm.weight"))?,
            k_norm: tensor_opt(gguf, &format!("blk.{i}.attn_k_norm.weight"))?,
            ffn_norm: tensor(gguf, &format!("blk.{i}.ffn_norm.weight"))?,
            w_gate: weight(gguf, &format!("blk.{i}.ffn_gate.weight"))?,
            w_up: weight(gguf, &format!("blk.{i}.ffn_up.weight"))?,
            w_down: weight(gguf, &format!("blk.{i}.ffn_down.weight"))?,
        });
    }

    let output_norm = tensor(gguf, "output_norm.weight")?;
    // Tied embeddings: fall back to the embedding table as LM head.
    let output = match gguf.find_tensor("output.weight") {
        Some(_) => weight(gguf, "output.weight")?,
        None => WeightMatrix::F32(token_embd.clone()),
    };

    Ok(GlprocModel {
        config: ModelConfig {
            arch,
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            head_dim,
            hidden_dim,
            vocab_size,
            max_seq,
            rms_eps,
            rope_freq_base,
            rope_style,
        },
        token_embd,
        layers,
        output_norm,
        output,
    })
}
