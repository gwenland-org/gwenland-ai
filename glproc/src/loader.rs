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

/// Page size used for the warm-touch stride. 4 KiB on every x86 target we
/// ship on; touching one byte per page faults the whole page in.
const PAGE_BYTES: usize = 4096;

#[cfg(windows)]
mod winmem {
    use std::ffi::c_void;
    #[link(name = "kernel32")]
    extern "system" {
        pub fn VirtualLock(addr: *const c_void, size: usize) -> i32;
        pub fn GetCurrentProcess() -> *mut c_void;
        pub fn SetProcessWorkingSetSize(process: *mut c_void, min: usize, max: usize) -> i32;
    }
}

/// The memory region backing one weight matrix, whichever representation
/// it is stored in.
fn weight_region(w: &WeightMatrix) -> (*const u8, usize) {
    match w {
        WeightMatrix::F32(v) => (v.as_ptr() as *const u8, std::mem::size_of_val(v.as_slice())),
        WeightMatrix::Quant(_, b) => (b.as_ptr(), b.len()),
    }
}

/// Warm the model's weights into RAM and pin them there. Call once, after
/// load and before the first decode.
///
/// A cold or evicted page costs a fault mid-matvec — at this machine's disk
/// speed that is milliseconds per page, which destroys decode latency. So:
/// touch every page up front (fault them in while nothing is latency
/// sensitive), then pin them (`VirtualLock` / `mlock`) so an 8 GB box under
/// memory pressure cannot swap the weights back out between requests.
///
/// Deviation from the X5 sketch: glproc copies tensors out of the GGUF mmap
/// into owned heap buffers at load, so the decode working set is those
/// buffers, not the mmap — warm-and-lock targets each weight buffer.
///
/// Best effort: pinning can fail (working-set quota, `ulimit -l`); the
/// prefetch touch still helps, so a failure only warns.
pub fn warm_and_lock_model(model: &GlprocModel) {
    let mut regions: Vec<(*const u8, usize)> = Vec::new();
    let mut push_f32 = |regions: &mut Vec<_>, v: &[f32]| {
        regions.push((v.as_ptr() as *const u8, std::mem::size_of_val(v)));
    };
    push_f32(&mut regions, &model.token_embd);
    push_f32(&mut regions, &model.output_norm);
    regions.push(weight_region(&model.output));
    for l in &model.layers {
        push_f32(&mut regions, &l.attn_norm);
        push_f32(&mut regions, &l.ffn_norm);
        for opt in [&l.bq, &l.bk, &l.bv, &l.q_norm, &l.k_norm] {
            if let Some(v) = opt {
                push_f32(&mut regions, v);
            }
        }
        for w in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
            regions.push(weight_region(w));
        }
    }
    regions.retain(|&(_, size)| size > 0);
    let total: usize = regions.iter().map(|&(_, size)| size).sum();

    // Step 1: touch one byte per page. Faults every page in now, so the
    // decode loop never takes one — and it still helps if pinning fails.
    for &(ptr, size) in &regions {
        let mut i = 0;
        while i < size {
            // SAFETY: ptr..ptr+size is a live owned buffer of the model.
            unsafe { std::ptr::read_volatile(ptr.add(i)) };
            i += PAGE_BYTES;
        }
    }

    // Step 2: pin the pages so they cannot be evicted.
    let mut failed = 0usize;
    #[cfg(windows)]
    {
        // VirtualLock is capped by the process working-set maximum (a few
        // MB by default), so raise it to model size + slack first.
        const SLACK: usize = 256 << 20;
        // SAFETY: plain kernel32 calls on the current process handle.
        unsafe {
            winmem::SetProcessWorkingSetSize(
                winmem::GetCurrentProcess(),
                total + SLACK,
                total + SLACK * 2,
            );
        }
        for &(ptr, size) in &regions {
            // SAFETY: region is a live owned buffer, valid for `size` bytes.
            if unsafe { winmem::VirtualLock(ptr as *const _, size) } == 0 {
                failed += 1;
            }
        }
    }
    #[cfg(unix)]
    {
        for &(ptr, size) in &regions {
            // SAFETY: region is a live owned buffer, valid for `size` bytes.
            if unsafe { libc::mlock(ptr as *const libc::c_void, size) } != 0 {
                failed += 1;
            }
        }
    }
    if failed > 0 {
        eprintln!(
            "warning: pinning failed for {failed}/{} weight buffers ({} MB total) — \
             pages are prefetched but may be swapped out under memory pressure \
             (unix: try `ulimit -l unlimited`)",
            regions.len(),
            total >> 20,
        );
    }
}

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
        // Q5_0 is repacked to Q8_0 at load: the conversion is bit-exact
        // (see `repack_to_q8_0`) and Q8_0's inner loop is much cheaper than
        // Q5_0's high-bit unpack, which measured compute-bound.
        Some(QuantFormat::Q5_0) if info.dimensions[0] as usize % 32 == 0 => {
            Ok(WeightMatrix::Quant(
                QuantFormat::Q8_0,
                kernels::dequant::q5_0::scalar::repack_to_q8_0(gguf.tensor_data(info)?)?,
            ))
        }
        // Q6_K likewise repacks to Q8_0 (requantized; error well under the
        // format's own quantization noise — see `repack_to_q8_0`).
        Some(QuantFormat::Q6K) if info.dimensions[0] as usize % 256 == 0 => {
            Ok(WeightMatrix::Quant(
                QuantFormat::Q8_0,
                kernels::dequant::q6_k::scalar::repack_to_q8_0(gguf.tensor_data(info)?)?,
            ))
        }
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
