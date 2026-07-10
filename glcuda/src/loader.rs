//! GGUF → [`HostModel`]: config parsing and weight staging.
//!
//! The config parse mirrors glproc's loader key-for-key (same metadata
//! keys, same defaults, same RoPE-style table) so both engines read one
//! file identically. Weight policy (M2.1 Task A): Q8_0 and Q4_K matmul
//! tensors stay quantized in their native SoA layouts (gl_gemv_q8_0_soa /
//! gl_gemv_q4_k_soa read them directly — quantized VRAM *is* the decode
//! bandwidth budget); other quantized dtypes (Q6_K, Q5_0) are requantized
//! to Q8_0 SoA (see `repack`); f16/f32 tensors upload dense f32.

use glcore::format::gguf::{GgufDType, GgufFile, GgufValue};
use glcore::GlError;

use crate::dequant::dequant_any;
use crate::model::{GpuModelConfig, HostLayer, HostMat, HostModel, HostWeight, RopeStyle};
use crate::repack::{f32_to_q8_0_soa, q4_0_to_soa, q4_k_to_soa, q6_k_to_soa};

/// Read `{arch}.{suffix}` from metadata as u64.
fn meta_u64(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<u64> {
    gguf.get_meta(&format!("{arch}.{suffix}")).and_then(GgufValue::as_u64)
}

/// Read `{arch}.{suffix}` from metadata as f32.
fn meta_f32(gguf: &GgufFile, arch: &str, suffix: &str) -> Option<f32> {
    gguf.get_meta(&format!("{arch}.{suffix}")).and_then(GgufValue::as_f32)
}

/// Dequantize a required tensor by name to f32.
fn tensor(gguf: &GgufFile, name: &str) -> Result<Vec<f32>, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    dequant_any(gguf, info)
}

/// Dequantize an optional tensor (attention biases, qwen3 head norms).
fn tensor_opt(gguf: &GgufFile, name: &str) -> Result<Option<Vec<f32>>, GlError> {
    match gguf.find_tensor(name) {
        Some(info) => Ok(Some(dequant_any(gguf, info)?)),
        None => Ok(None),
    }
}

/// Load a required weight matrix: Q8_0 stays quantized when its rows are
/// whole blocks; everything else becomes dense f32.
fn weight(gguf: &GgufFile, name: &str) -> Result<HostMat, GlError> {
    let info = gguf
        .find_tensor(name)
        .ok_or_else(|| GlError::Parse(format!("GGUF: missing tensor '{name}'")))?;
    // dimensions[0] is the contiguous axis = in_features (GGUF order).
    let in_dim = info.dimensions.first().copied().unwrap_or(0) as usize;
    let out_dim = info.dimensions.get(1).copied().unwrap_or(1) as usize;
    if in_dim == 0 || in_dim * out_dim != info.numel() {
        return Err(GlError::Parse(format!(
            "GGUF: tensor '{name}' has unsupported shape {:?}",
            info.dimensions
        )));
    }
    let w = match info.dtype {
        GgufDType::Q8_0 if in_dim.is_multiple_of(32) => {
            let data = gguf.tensor_data(info)?;
            // Structure-of-Arrays: split the 34-byte blocks into a contiguous
            // int8 qs stream + a contiguous f16 scale stream (both row-major),
            // so the GEMV reads qs as one coalesced transaction with no padding.
            let n_blocks = data.len() / 34;
            let mut qs = Vec::with_capacity(n_blocks * 32);
            let mut scales = Vec::with_capacity(n_blocks * 2);
            for block in data.chunks_exact(34) {
                scales.extend_from_slice(&block[0..2]);  // f16 scale
                qs.extend_from_slice(&block[2..34]);     // 32 quantized weights
            }
            HostWeight::Q8_0Soa { qs, scales }
        }
        // Native Q4_K path (M2.1 Task A): repack the 144-byte super-blocks
        // into the SoA triple gl_gemv_q4_k_soa streams at 5.0 bpw. The
        // in_dim % 256 guard is belt-and-braces — ggml cannot emit a Q4_K
        // tensor with a ragged row (QK_K divisibility is a format invariant).
        GgufDType::Q4_K if in_dim.is_multiple_of(256) => {
            let (qs, scales, mins) = q4_k_to_soa(gguf.tensor_data(info)?)?;
            HostWeight::Q4KSoa { qs, scales, mins }
        }
        // Native Q4_0 path (M2.2 Task C-2): SoA nibbles + verbatim f16
        // scales for gl_gemv_q4_0_soa. The kernel has a block tail, so
        // in % 32 (the format's own invariant) is the only requirement.
        GgufDType::Q4_0 if in_dim.is_multiple_of(32) => {
            let (qs, scales) = q4_0_to_soa(gguf.tensor_data(info)?)?;
            HostWeight::Q4_0Soa { qs, scales }
        }
        // Native Q6_K path (M2.2 Task C-1): four SoA streams at the exact
        // native 6.5625 bpw — replaces the M2.1 requant-to-Q8_0 detour that
        // streamed these tensors (half of Q4_K_M's ffn_down/attn_v, plus
        // output.weight) at 8.5 bpw. Zero added quantization error: every
        // stream is verbatim or losslessly relocated.
        GgufDType::Q6_K if in_dim.is_multiple_of(256) => {
            let (ql, qh, scales, d) = q6_k_to_soa(gguf.tensor_data(info)?)?;
            HostWeight::Q6KSoa { ql, qh, scales, d }
        }
        // Quantized dtypes with no native kernel (Q5_0, plus ragged-row
        // k-quants, which the format itself cannot normally produce):
        // requantize to Q8_0 SoA instead of dense f32. Same policy glproc
        // documents for its repack: Q8_0 adds ~2^-8 relative error on top
        // of an already-lossy format, where dense f32 would multiply the
        // tensor's VRAM and per-token DRAM traffic by 4-6x.
        GgufDType::Q5_0 | GgufDType::Q6_K | GgufDType::Q4_K
            if in_dim.is_multiple_of(32) =>
        {
            let (qs, scales) = f32_to_q8_0_soa(&dequant_any(gguf, info)?);
            HostWeight::Q8_0Soa { qs, scales }
        }
        _ => HostWeight::F32(dequant_any(gguf, info)?),
    };
    Ok(HostMat { w, out_dim, in_dim })
}

/// Fuse the FFN gate and up projections into one `[2*hidden, dim]` matrix
/// (gate rows then up rows) so the decode FFN is one GEMV instead of two —
/// the input is streamed once and one launch replaces two. Gate and up in a
/// GGUF always share a dtype, so the fast path stacks them directly; the
/// defensive fallback re-loads both as dense f32 (always stackable).
fn fuse_gate_up(
    gate: HostMat,
    up: HostMat,
    gguf: &GgufFile,
    layer: usize,
) -> Result<HostMat, GlError> {
    if gate.stackable(&up) {
        return Ok(gate.stack_rows(up));
    }
    // Mismatched representations (shouldn't happen for a real gate/up pair):
    // reload both as dense f32 and stack.
    let g = dequant_any(
        gguf,
        gguf.find_tensor(&format!("blk.{layer}.ffn_gate.weight"))
            .ok_or_else(|| GlError::Parse(format!("GGUF: missing blk.{layer}.ffn_gate.weight")))?,
    )?;
    let u = dequant_any(
        gguf,
        gguf.find_tensor(&format!("blk.{layer}.ffn_up.weight"))
            .ok_or_else(|| GlError::Parse(format!("GGUF: missing blk.{layer}.ffn_up.weight")))?,
    )?;
    let gm = HostMat { w: HostWeight::F32(g), out_dim: gate.out_dim, in_dim: gate.in_dim };
    let um = HostMat { w: HostWeight::F32(u), out_dim: up.out_dim, in_dim: up.in_dim };
    Ok(gm.stack_rows(um))
}

/// Parse config and stage every weight of a GGUF transformer for upload.
/// Supports the same llama-family architectures as glproc (standard
/// `blk.N.*` tensor naming).
/// Stage one transformer block's weights (the parallel unit of `load_host`).
fn build_layer(gguf: &GgufFile, i: usize) -> Result<HostLayer, GlError> {
    Ok(HostLayer {
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
        w_gate_up: fuse_gate_up(
            weight(gguf, &format!("blk.{i}.ffn_gate.weight"))?,
            weight(gguf, &format!("blk.{i}.ffn_up.weight"))?,
            gguf,
            i,
        )?,
        w_down: weight(gguf, &format!("blk.{i}.ffn_down.weight"))?,
    })
}

pub fn load_host(gguf: &GgufFile) -> Result<HostModel, GlError> {
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
        return Err(GlError::Parse("GGUF: model dimensions must be non-zero".into()));
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
    // Same style table as glproc: original-llama rotates adjacent pairs,
    // newer archs use the NeoX half-split convention.
    let rope_style = match arch.as_str() {
        "llama" | "llama2" | "minicpm" => RopeStyle::Norm,
        _ => RopeStyle::Neox,
    };

    // Embedding table: stays host-side (row lookup per token). Q8_0 is kept
    // quantized (row dequant costs well under a microsecond); other quants
    // go f32.
    let embd_info = gguf
        .find_tensor("token_embd.weight")
        .ok_or_else(|| GlError::Parse("GGUF: missing tensor 'token_embd.weight'".into()))?;
    let vocab_size = embd_info.dimensions.get(1).copied().unwrap_or(0) as usize;
    if vocab_size == 0 {
        return Err(GlError::Parse("GGUF: token_embd.weight has no vocab dimension".into()));
    }
    let token_embd = match embd_info.dtype {
        GgufDType::Q8_0 if dim.is_multiple_of(32) => {
            let data = gguf.tensor_data(embd_info)?;
            let mut padded = Vec::with_capacity((data.len() / 34) * 36);
            for block in data.chunks_exact(34) {
                padded.extend_from_slice(&block[0..2]);
                padded.extend_from_slice(&[0, 0]);
                padded.extend_from_slice(&block[2..34]);
            }
            HostWeight::Q8_0(padded)
        }
        GgufDType::Q4_0 if dim.is_multiple_of(32) => {
            HostWeight::Q4_0(gguf.tensor_data(embd_info)?.to_vec())
        }
        // Q4_K embedding (Q4_K_M files): keep the raw super-blocks and
        // dequantize one row per token (q4_k_row_into) — 4.5 bpw host RAM
        // instead of a dense-f32 table (~2 GB for a 152k x 3584 vocab).
        GgufDType::Q4_K if dim.is_multiple_of(256) => {
            HostWeight::Q4K(gguf.tensor_data(embd_info)?.to_vec())
        }
        // Q6_K embedding: raw super-blocks + per-row dequant, like Q4_K.
        GgufDType::Q6_K if dim.is_multiple_of(256) => {
            HostWeight::Q6K(gguf.tensor_data(embd_info)?.to_vec())
        }
        _ => HostWeight::F32(dequant_any(gguf, embd_info)?),
    };

    // Repacking the per-layer weights (Q8_0 -> SoA) is the bulk of staging and
    // is embarrassingly parallel across layers — each reads a disjoint slice of
    // the (Sync) mmap. Fan out over the available cores; a single core keeps the
    // old sequential behavior.
    let n_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(n_layers.max(1));
    eprintln!("[glcuda] stage: repacking {n_layers} layers across {n_threads} threads");
    let mut built: Vec<Option<HostLayer>> = (0..n_layers).map(|_| None).collect();
    std::thread::scope(|s| -> Result<(), GlError> {
        let handles: Vec<_> = (0..n_threads)
            .map(|t| {
                s.spawn(move || -> Result<Vec<(usize, HostLayer)>, GlError> {
                    let mut out = Vec::new();
                    let mut i = t;
                    while i < n_layers {
                        out.push((i, build_layer(gguf, i)?));
                        i += n_threads;
                    }
                    Ok(out)
                })
            })
            .collect();
        for h in handles {
            for (i, layer) in h.join().expect("layer-staging thread panicked")? {
                built[i] = Some(layer);
            }
        }
        Ok(())
    })?;
    let layers: Vec<HostLayer> = built.into_iter().map(|o| o.expect("every layer built")).collect();

    let output_norm = tensor(gguf, "output_norm.weight")?;
    // Tied embeddings: reuse the embedding tensor as LM head — staged
    // through `weight` (not a clone of `token_embd`) so the matmul gets the
    // SoA repack the embedding table's AoS row-lookup layout doesn't have.
    let output = match gguf.find_tensor("output.weight") {
        Some(_) => weight(gguf, "output.weight")?,
        None => weight(gguf, "token_embd.weight")?,
    };

    Ok(HostModel {
        config: GpuModelConfig {
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
