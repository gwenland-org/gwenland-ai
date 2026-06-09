/// LoRA-to-GGUF merge pipeline: key mapping, quantization, and weight merging.
///
/// @INFO This module is the merge side of GWEN-213. It maps candle LoRA
/// variable names to their GGUF counterparts, dequantizes base weights,
/// applies the LoRA delta, and requantizes to Q8_0.
use std::collections::HashMap;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write as IoWrite};
use std::path::Path;

use candle_core::{Device, Error as CandleError, Tensor};
use sysinfo::System;

use crate::convert::gguf_parser::{self, GgufDtype};
use crate::error::GwenError;
use crate::train::lora_bridge::LoraAdapter;

// ── KeyMapper ─────────────────────────────────────────────────────────────────

/// Bijective mapping between candle LoRA key names and GGUF weight key names.
///
/// @INFO The two naming schemes are structurally different: candle uses
/// `lora_layer_{N}_{proj}_proj` while GGUF uses
/// `model.layers.{N}.{module}.{proj}_proj.weight`. This struct provides
/// both directions so the merger can look up adapters by GGUF key and
/// convert back for diagnostic messages.
pub struct KeyMapper;

impl KeyMapper {
    /// Convert a candle LoRA key to its GGUF base-weight key.
    ///
    /// Input format:  `lora_layer_{N}_{proj}_proj`
    /// Output format: `model.layers.{N}.self_attn.{proj}_proj.weight` for q/k/v/o
    ///                `model.layers.{N}.mlp.{proj}_proj.weight`        for gate/up/down
    ///
    /// Returns a descriptive Err for keys that do not match the expected pattern.
    /// @DANGER The module assignment (self_attn vs mlp) is inferred from the
    /// projection name; any unrecognised proj is silently routed to `mlp`.
    pub fn candle_to_gguf(key: &str) -> candle_core::Result<String> {
        let (layer_idx, proj) = parse_candle_key(key)
            .ok_or_else(|| CandleError::Msg(format!("malformed candle LoRA key: '{key}'")))?;
        let module = attn_or_mlp(proj);
        Ok(format!(
            "model.layers.{layer_idx}.{module}.{proj}_proj.weight"
        ))
    }

    /// Convert a GGUF base-weight key back to its candle LoRA key.
    ///
    /// Input format:  `model.layers.{N}.{self_attn|mlp}.{proj}_proj.weight`
    /// Output format: `lora_layer_{N}_{proj}_proj`
    ///
    /// Supports both `self_attn` and `mlp` modules.
    /// Returns a descriptive Err for keys that do not match the expected pattern.
    pub fn gguf_to_candle(key: &str) -> candle_core::Result<String> {
        let (layer_idx, proj) = parse_gguf_key(key)
            .ok_or_else(|| CandleError::Msg(format!("malformed GGUF weight key: '{key}'")))?;
        Ok(format!("lora_layer_{layer_idx}_{proj}_proj"))
    }
}

// ── internal key helpers ──────────────────────────────────────────────────────

/// Parse `lora_layer_{N}_{proj}_proj` → (layer_idx, proj).
///
/// @INFO The suffix `_proj` is stripped so only the projection name (e.g.
/// "q", "gate") is returned. Returns None for any key that doesn't match.
fn parse_candle_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("lora_layer_")?;
    let underscore = rest.find('_')?;
    let layer_idx: usize = rest[..underscore].parse().ok()?;
    let after_idx = &rest[underscore + 1..];
    let proj = after_idx.strip_suffix("_proj")?;
    Some((layer_idx, proj))
}

/// Parse `model.layers.{N}.{module}.{proj}_proj.weight` → (layer_idx, proj).
///
/// @INFO The module segment (self_attn / mlp) is consumed but not returned;
/// it is fully determined by the projection name via `attn_or_mlp`.
fn parse_gguf_key(key: &str) -> Option<(usize, &str)> {
    let rest = key.strip_prefix("model.layers.")?;
    let dot = rest.find('.')?;
    let layer_idx: usize = rest[..dot].parse().ok()?;
    let rest = &rest[dot + 1..];
    let dot2 = rest.find('.')?;
    let rest = &rest[dot2 + 1..];
    let proj_weight = rest.strip_suffix(".weight")?;
    let proj = proj_weight.strip_suffix("_proj")?;
    Some((layer_idx, proj))
}

/// Map a projection name to its GGUF module segment.
///
/// @INFO q/k/v/o are attention projections; gate/up/down are MLP projections.
/// Any unrecognised name is routed to mlp as a safe fallback.
fn attn_or_mlp(proj: &str) -> &'static str {
    match proj {
        "q" | "k" | "v" | "o" => "self_attn",
        _ => "mlp",
    }
}

// ── Q8_0 quantization ─────────────────────────────────────────────────────────

/// Number of f32 elements per Q8_0 block (GGML constant).
const Q8_0_BLOCK: usize = 32;

/// Quantize a slice of f32 weights to GGML Q8_0 format.
///
/// Each 32-element block is encoded as:
///   2 bytes: block scale as IEEE 754 f16, little-endian
///  32 bytes: i8 quantized values, clamped to [-127, 127]
///
/// @INFO scale = max_abs / 127.0. All-zero blocks use scale = 1.0 to avoid
/// division by zero, which means all quantized values will be 0.
/// @EDITABLE The clamp bound [-127, 127] leaves the value -128 unused,
/// matching the original GGML Q8_0 spec.
/// @TODO Add support for Q4_0 and Q4_K if finer memory control is needed.
pub fn quantize_q8_0(weights: &[f32]) -> candle_core::Result<Vec<u8>> {
    if weights.len() % Q8_0_BLOCK != 0 {
        return Err(CandleError::Msg(format!(
            "quantize_q8_0: weight length {} is not a multiple of {}",
            weights.len(),
            Q8_0_BLOCK
        )));
    }

    let n_blocks = weights.len() / Q8_0_BLOCK;
    let mut out = vec![0u8; n_blocks * (2 + Q8_0_BLOCK)];
    let mut out_pos = 0;

    for block in weights.chunks_exact(Q8_0_BLOCK) {
        let max_abs = block.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs == 0.0 { 1.0f32 } else { max_abs / 127.0 };

        let scale_f16_bits = f32_to_f16_bits(scale);
        out[out_pos] = (scale_f16_bits & 0xFF) as u8;
        out[out_pos + 1] = ((scale_f16_bits >> 8) & 0xFF) as u8;
        out_pos += 2;

        for &w in block {
            let q = (w / scale).round().clamp(-127.0, 127.0) as i8;
            out[out_pos] = q as u8;
            out_pos += 1;
        }
    }

    Ok(out)
}

/// Dequantize a Q8_0-encoded byte slice back to f32 weights.
///
/// Each block of (2 + 32) bytes is decoded as:
///   2 bytes → f16 scale (little-endian)
///  32 bytes → i8 values; w[i] = scale × q[i]
///
/// @INFO The f16 conversion is done via manual bit manipulation to avoid
/// pulling in the `half` crate as a direct dependency.
pub fn dequantize_q8_0(bytes: &[u8]) -> candle_core::Result<Vec<f32>> {
    let block_bytes = 2 + Q8_0_BLOCK;
    if bytes.len() % block_bytes != 0 {
        return Err(CandleError::Msg(format!(
            "dequantize_q8_0: byte length {} is not a multiple of {}",
            bytes.len(),
            block_bytes
        )));
    }

    let n_blocks = bytes.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * Q8_0_BLOCK);

    for block in bytes.chunks_exact(block_bytes) {
        let scale_bits = (block[0] as u16) | ((block[1] as u16) << 8);
        let scale = f16_bits_to_f32(scale_bits);
        for &byte in &block[2..] {
            let q = byte as i8;
            out.push(scale * (q as f32));
        }
    }

    Ok(out)
}

// ── f16 bit-level helpers ─────────────────────────────────────────────────────

/// Convert an f32 value to its IEEE 754 f16 bit pattern.
///
/// @INFO Round-to-nearest-even is not implemented; truncation of mantissa bits
/// is acceptable for scale values which are always positive and moderate-sized.
/// @DANGER Subnormal f32 inputs underflow to zero in f16. Inputs > 65504
/// saturate to f16 infinity.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3FF;

    if exp <= 0 {
        sign
    } else if exp >= 31 {
        sign | 0x7C00
    } else {
        sign | ((exp as u16) << 10) | (mantissa as u16)
    }
}

/// Convert an IEEE 754 f16 bit pattern to f32.
///
/// @INFO Handles zero, subnormals, normals, infinity, and NaN correctly.
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mantissa = (bits & 0x3FF) as u32;

    let (exp32, mant32) = if exp == 0 {
        (0u32, mantissa << 13)
    } else if exp == 31 {
        (0xFF, mantissa << 13)
    } else {
        ((exp + 127 - 15) as u32, mantissa << 13)
    };

    f32::from_bits(sign | (exp32 << 23) | mant32)
}

// ── LoraMerger ────────────────────────────────────────────────────────────────

/// Maximum permitted adapter file size (10 GB).
const MAX_ADAPTER_BYTES: u64 = 10 * 1024 * 1024 * 1024;

/// Merges a candle-trained LoRA adapter SafeTensors file into a GGUF base model.
///
/// The merge strategy is dequant → add delta → requant (Q8_0), applied
/// layer-by-layer in a streaming fashion so peak RAM stays within `memory_budget`.
///
/// @INFO Sized for 8 GB RAM machines (GwenLand hardware target). On machines
/// with more RAM, increase the budget via `with_memory_budget()` to allow
/// larger in-flight tensors and potentially faster throughput.
pub struct LoraMerger {
    /// Maximum bytes of RAM the merge loop may consume at any instant.
    ///
    /// @EDITABLE Default is 2 GB. Override with `with_memory_budget()` for
    /// machines with more RAM or for test isolation with synthetic tensors.
    pub memory_budget: usize,
}

impl LoraMerger {
    /// Construct a merger with the default 2 GB memory budget.
    ///
    /// @INFO The 2 GB default leaves headroom on 8 GB machines for the OS,
    /// the inference runtime, and tokenizer state running concurrently.
    pub fn new() -> Self {
        Self {
            memory_budget: 2 * 1024 * 1024 * 1024,
        }
    }

    /// Construct a merger with a custom memory budget in bytes.
    ///
    /// @EDITABLE Use a small value (e.g. 64 MB) in tests to verify that
    /// `MemoryBudgetExceeded` is surfaced correctly for large tensors.
    pub fn with_memory_budget(budget: usize) -> Self {
        Self {
            memory_budget: budget,
        }
    }

    /// Merge a LoRA adapter SafeTensors file into a GGUF base model.
    ///
    /// Pipeline:
    ///   1. Validate all three paths.
    ///   2. Load adapter tensors from the SafeTensors file.
    ///   3. Parse the base GGUF via `gguf_parser::parse()`.
    ///   4. For each base tensor: if a matching adapter exists, dequant → add
    ///      delta → requant; otherwise copy raw bytes verbatim.
    ///   5. Write output: base file header + processed tensor data.
    ///
    /// @INFO Q8_0 → merge → Q8_0 preserves block count and byte size exactly,
    /// so the GGUF header (tensor offsets and sizes) does not need updating.
    /// @DANGER Only Q8_0-quantized tensors can be merged; other formats return
    /// `UnsupportedQuantization`. F32 base models are not yet supported.
    /// @TODO Wave 5: extend to Q4_K and other formats via dequant→merge→requant.
    pub fn merge_into_gguf(
        &self,
        base_path: &Path,
        adapter_path: &Path,
        output_path: &Path,
    ) -> std::result::Result<(), GwenError> {
        // ── Step 0: Path validation (from Wave 3 stub) ────────────────────────
        if !base_path.exists() {
            return Err(GwenError::CandleError(format!(
                "base model path does not exist: {}",
                base_path.display()
            )));
        }
        std::fs::File::open(base_path).map_err(|e| {
            GwenError::CandleError(format!(
                "base model path is not readable ({}): {}",
                base_path.display(),
                e
            ))
        })?;

        if !adapter_path.exists() {
            return Err(GwenError::CandleError(format!(
                "adapter path does not exist: {}",
                adapter_path.display()
            )));
        }
        let adapter_meta = std::fs::metadata(adapter_path).map_err(|e| {
            GwenError::CandleError(format!(
                "cannot stat adapter path ({}): {}",
                adapter_path.display(),
                e
            ))
        })?;
        if adapter_meta.len() > MAX_ADAPTER_BYTES {
            return Err(GwenError::CandleError(format!(
                "adapter file exceeds 10 GB limit: {} bytes ({})",
                adapter_meta.len(),
                adapter_path.display()
            )));
        }

        let output_parent = output_path.parent().ok_or_else(|| {
            GwenError::CandleError(format!(
                "output path has no parent directory: {}",
                output_path.display()
            ))
        })?;
        if !output_parent.exists() {
            return Err(GwenError::CandleError(format!(
                "output parent directory does not exist: {}",
                output_parent.display()
            )));
        }

        // ── Step 1: Load adapter tensors from SafeTensors ─────────────────────
        let adapters = load_adapter_safetensors(adapter_path)?;

        // ── Step 2: Parse base GGUF ───────────────────────────────────────────
        // gguf_parser::parse() loads all raw_data eagerly — no mmap needed here.
        // For very large files this is memory-intensive, but it gives us clean
        // sequential I/O which is faster than random seeks on spinning disks.
        let gguf = gguf_parser::parse(base_path)
            .map_err(|e| GwenError::CandleError(format!("GGUF parse failed: {e}")))?;

        // Verify magic is present by checking the version parsed successfully.
        // gguf_parser::parse() already validates magic and version internally.

        // ── Step 3: Streaming merge loop ─────────────────────────────────────
        let mut sys = System::new_all();

        // Collect processed tensor bytes in order.
        let mut processed_tensors: Vec<(String, Vec<u8>)> =
            Vec::with_capacity(gguf.tensors.len());
        let mut merged_count: usize = 0;

        for tensor in &gguf.tensors {
            // Memory budget check — refresh sysinfo before each tensor.
            sys.refresh_memory();
            let available_bytes = sys.available_memory() as usize;
            let tensor_size = tensor.data_size;

            if available_bytes < tensor_size {
                return Err(GwenError::MemoryBudgetExceeded {
                    required: tensor_size,
                    available: available_bytes,
                });
            }

            // Try to find a matching LoRA adapter for this GGUF tensor.
            let candle_key = KeyMapper::gguf_to_candle(&tensor.name).ok();
            let maybe_adapter = candle_key.as_deref().and_then(|k| adapters.get(k));

            let out_bytes = if let Some(adapter) = maybe_adapter {
                // ── Task 5.4: dequant → merge → requant ───────────────────────

                // Only Q8_0 tensors can be merged.
                if tensor.dtype != GgufDtype::Q8_0 {
                    return Err(GwenError::UnsupportedQuantization {
                        format: format!("{:?}", tensor.dtype),
                    });
                }

                // Dequantize base weights to f32.
                let base_f32 = dequantize_q8_0(&tensor.raw_data)
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;

                // Compute adapter delta Δ = (alpha/rank) × B @ A.
                // @INFO alpha=1.0 at merge time; scaling is baked into export.
                let delta_tensor = adapter
                    .compute_delta()
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;

                // Flatten delta to Vec<f32>.
                let delta_f32 = delta_tensor
                    .flatten_all()
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;

                // Shape check: delta and base weights must have the same element count.
                if delta_f32.len() != base_f32.len() {
                    let base_shape: Vec<usize> =
                        tensor.shape.iter().map(|&d| d as usize).collect();
                    let adapter_shape = adapter.lora_b.dims().to_vec();
                    return Err(GwenError::ShapeMismatch {
                        adapter: adapter_shape,
                        base: base_shape,
                    });
                }

                // W_merged = W_base + Δ element-wise.
                let mut merged: Vec<f32> = base_f32
                    .iter()
                    .zip(delta_f32.iter())
                    .map(|(&b, &d)| b + d)
                    .collect();

                // Validate finiteness — NaN or Inf in merged weights corrupts inference.
                for (idx, &v) in merged.iter().enumerate() {
                    if !v.is_finite() {
                        return Err(GwenError::InvalidMergedWeights {
                            layer_name: tensor.name.clone(),
                            index: idx,
                        });
                    }
                }

                // Requantize merged f32 back to Q8_0.
                // If merged element count is not a multiple of 32, pad with zeros
                // to the next block boundary (should never happen with valid models).
                let remainder = merged.len() % Q8_0_BLOCK;
                if remainder != 0 {
                    merged.extend(std::iter::repeat(0.0f32).take(Q8_0_BLOCK - remainder));
                }

                let out = quantize_q8_0(&merged)
                    .map_err(|e| GwenError::CandleError(e.to_string()))?;

                merged_count += 1;
                eprintln!(
                    "[gwen-merge] merged layer: {} ({} elements)",
                    tensor.name,
                    merged.len()
                );

                out
            } else {
                // No adapter for this tensor — copy raw bytes verbatim.
                tensor.raw_data.clone()
            };

            processed_tensors.push((tensor.name.clone(), out_bytes));
        }

        // ── Step 4: Write output GGUF ─────────────────────────────────────────
        // Strategy: copy the entire base file verbatim, then seek back and patch
        // each tensor's data region with the processed bytes. This preserves all
        // KV metadata, alignment padding, and header offsets exactly.
        // @INFO This works because Q8_0→merge→Q8_0 is byte-size-preserving.
        write_output_gguf(base_path, output_path, &gguf, &processed_tensors)?;

        eprintln!(
            "[gwen-merge] complete: {}/{} tensors merged, output: {}",
            merged_count,
            gguf.tensors.len(),
            output_path.display()
        );

        Ok(())
    }
}

impl Default for LoraMerger {
    /// Default impl delegates to `new()` so `LoraMerger::default()` works.
    fn default() -> Self {
        Self::new()
    }
}

// ── adapter SafeTensors loader ────────────────────────────────────────────────

/// Parse a SafeTensors adapter file written by `LoraExporter::export_safetensors`.
///
/// Returns a map from candle layer name (e.g. `lora_layer_0_q_proj`) to a
/// fully-constructed `LoraAdapter` on CPU.
///
/// @INFO The adapter file uses the format we write in lora_bridge.rs:
///   - 8-byte LE u64: header JSON length
///   - JSON header: `{"{layer}.lora_a": {dtype, shape, data_offsets}, ...}`
///   - Contiguous F32 data blobs
/// @DANGER Tensor offsets in the JSON are relative to the start of the data
/// blob (i.e., immediately after the header bytes), not to the file start.
fn load_adapter_safetensors(
    path: &Path,
) -> std::result::Result<HashMap<String, LoraAdapter>, GwenError> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| GwenError::CandleError(format!("cannot open adapter file: {e}")))?;

    // Read 8-byte LE header length.
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| GwenError::CandleError(format!("adapter header read failed: {e}")))?;
    let header_len = u64::from_le_bytes(len_buf) as usize;

    // Read and parse JSON header.
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|e| GwenError::CandleError(format!("adapter header body read failed: {e}")))?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| GwenError::CandleError(format!("adapter header JSON parse failed: {e}")))?;

    // Read all data bytes after the header.
    let mut data_blob: Vec<u8> = Vec::new();
    file.read_to_end(&mut data_blob)
        .map_err(|e| GwenError::CandleError(format!("adapter data read failed: {e}")))?;

    // Parse tensor entries: group by layer name, pair lora_a with lora_b.
    let obj = header
        .as_object()
        .ok_or_else(|| GwenError::CandleError("adapter header is not a JSON object".to_string()))?;

    // Collect tensors: key = full name (e.g. "lora_layer_0_q_proj.lora_a")
    let mut lora_a_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
    let mut lora_b_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();

    for (tensor_key, meta) in obj {
        let shape: Vec<usize> = meta["shape"]
            .as_array()
            .ok_or_else(|| GwenError::CandleError(format!("{tensor_key}: missing shape")))?
            .iter()
            .map(|v| v.as_u64().unwrap_or(0) as usize)
            .collect();

        let offsets = meta["data_offsets"]
            .as_array()
            .ok_or_else(|| GwenError::CandleError(format!("{tensor_key}: missing data_offsets")))?;
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;

        if end > data_blob.len() || start > end {
            return Err(GwenError::CandleError(format!(
                "{tensor_key}: data_offsets [{start},{end}] out of bounds (data len={})",
                data_blob.len()
            )));
        }

        // Decode F32 little-endian bytes.
        let byte_slice = &data_blob[start..end];
        if byte_slice.len() % 4 != 0 {
            return Err(GwenError::CandleError(format!(
                "{tensor_key}: data byte length {} not f32-aligned",
                byte_slice.len()
            )));
        }
        let floats: Vec<f32> = byte_slice
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Split on ".lora_a" / ".lora_b" suffix.
        if let Some(layer_name) = tensor_key.strip_suffix(".lora_a") {
            lora_a_map.insert(layer_name.to_string(), (shape, floats));
        } else if let Some(layer_name) = tensor_key.strip_suffix(".lora_b") {
            lora_b_map.insert(layer_name.to_string(), (shape, floats));
        }
        // Ignore unrecognised keys (e.g. metadata entries).
    }

    // Pair lora_a with lora_b and construct LoraAdapter objects.
    let dev = &Device::Cpu;
    let mut result: HashMap<String, LoraAdapter> = HashMap::new();

    for (layer_name, (a_shape, a_data)) in lora_a_map {
        let (b_shape, b_data) = lora_b_map.remove(&layer_name).ok_or_else(|| {
            // Extract layer index from name for error variant.
            let layer_idx = parse_candle_key(&layer_name)
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            GwenError::MissingLoraPair { layer_idx }
        })?;

        let rank = a_shape.first().copied().unwrap_or(1);

        let lora_a = Tensor::from_vec(a_data, a_shape, dev)
            .map_err(|e| GwenError::CandleError(e.to_string()))?;
        let lora_b = Tensor::from_vec(b_data, b_shape, dev)
            .map_err(|e| GwenError::CandleError(e.to_string()))?;

        // Convert candle layer name to GGUF key for the lookup map.
        // The map is keyed by candle name so KeyMapper::gguf_to_candle() in the
        // merge loop can find the right adapter.
        result.insert(
            layer_name.clone(),
            LoraAdapter {
                layer_name: layer_name.clone(),
                lora_a,
                lora_b,
                rank,
                // @INFO alpha=1.0 at merge time: the effective scale (alpha/rank)
                // was already baked into the exported adapter weights by the
                // training loop. Using alpha=1.0 and rank=rank gives scale=1/rank,
                // but since we set rank from lora_a.shape()[0] and alpha = rank,
                // effective scale = alpha/rank = 1.0, which is what we want.
                alpha: rank as f32,
            },
        );
    }

    Ok(result)
}

// ── output GGUF writer ────────────────────────────────────────────────────────

/// Write the output GGUF by copying the base file then patching tensor data.
///
/// Since Q8_0 → merge → Q8_0 preserves block count and byte size exactly,
/// the GGUF header (tensor offsets and sizes) remains valid after patching.
///
/// @INFO The base file is copied verbatim first so all KV metadata, version
/// fields, alignment padding, and tensor index are preserved without re-parsing.
/// @DANGER The patch assumes `processed_tensors` is in the same order as
/// `gguf.tensors`. Reordering would corrupt the output silently.
fn write_output_gguf(
    base_path: &Path,
    output_path: &Path,
    gguf: &gguf_parser::GgufFile,
    processed_tensors: &[(String, Vec<u8>)],
) -> std::result::Result<(), GwenError> {
    // Read the entire base file into memory for copying.
    // @INFO For multi-GB production models this uses significant RAM; Wave 5
    // should switch to a streaming copy + patch approach.
    let base_bytes = std::fs::read(base_path)
        .map_err(|e| GwenError::CandleError(format!("failed to read base GGUF: {e}")))?;

    // Find the data segment start: it is the offset immediately after the last
    // byte of the last tensor's raw_data in the base file. We compute it by
    // finding the minimum data_offset from the tensor index (which is the
    // relative offset within the data segment) and adding the absolute file
    // position of the data segment start.
    //
    // The data segment start is not stored explicitly in GgufFile, but we can
    // recover it: for the first tensor, abs_offset = data_segment_start +
    // data_offset. Since data_offset for the first tensor is typically 0, the
    // data_segment_start = abs position of first tensor in the file.
    //
    // We locate it by finding where the first tensor's raw_data appears in
    // base_bytes, using its data_offset relative to others.
    //
    // Simpler: we know data_offset is relative to the data segment. We find the
    // data segment start by scanning forward from the magic bytes for the known
    // raw_data pattern. But the cleanest approach: reconstruct it from the
    // gguf_parser's internal alignment logic.
    //
    // Practical approach: compute data_segment_start as
    //   file_size - sum(all data_size) - alignment_slack
    // But alignment_slack is unknown without re-parsing.
    //
    // Most reliable: re-open the base and parse just enough to find the data
    // block start position, then use gguf tensor data_offset values (which are
    // relative) to locate each tensor's absolute position.
    let data_segment_start = find_data_segment_start(&base_bytes)
        .ok_or_else(|| GwenError::CandleError("cannot locate GGUF data segment".to_string()))?;

    // Clone base bytes into output buffer.
    let mut out_bytes = base_bytes;

    // Patch each tensor's region in the output buffer.
    for ((tensor_name, new_data), tensor_info) in
        processed_tensors.iter().zip(gguf.tensors.iter())
    {
        debug_assert_eq!(
            tensor_name, &tensor_info.name,
            "tensor order mismatch: {} vs {}",
            tensor_name, tensor_info.name
        );

        let abs_start =
            (data_segment_start + tensor_info.data_offset as usize).min(out_bytes.len());
        let abs_end = abs_start + tensor_info.data_size;

        if abs_end > out_bytes.len() {
            return Err(GwenError::CandleError(format!(
                "tensor '{}' data region [{abs_start},{abs_end}) exceeds file size {}",
                tensor_name,
                out_bytes.len()
            )));
        }

        // Verify byte sizes match before patching.
        if new_data.len() != tensor_info.data_size {
            return Err(GwenError::CandleError(format!(
                "tensor '{}': new data {} bytes != original {} bytes (size mismatch after merge)",
                tensor_name,
                new_data.len(),
                tensor_info.data_size
            )));
        }

        out_bytes[abs_start..abs_end].copy_from_slice(new_data);
    }

    // Write the patched buffer to the output file.
    let out_file = std::fs::File::create(output_path)
        .map_err(|e| GwenError::CandleError(format!("cannot create output file: {e}")))?;
    let mut writer = BufWriter::new(out_file);
    writer
        .write_all(&out_bytes)
        .map_err(|e| GwenError::CandleError(format!("output write failed: {e}")))?;
    writer
        .flush()
        .map_err(|e| GwenError::CandleError(format!("output flush failed: {e}")))?;

    Ok(())
}

/// Locate the data segment start offset within the raw GGUF bytes.
///
/// @INFO The GGUF data segment starts at a 32-byte-aligned boundary after
/// the header (magic, version, tensor count, kv count, KV entries, tensor
/// info entries). We find it by scanning for the alignment point.
///
/// The reliable way: re-parse the header with a cursor to find the position
/// right after tensor info, then round up to 32-byte alignment.
///
/// @DANGER Returns None only if the bytes are not a valid GGUF file at all;
/// all valid GGUF files have a data segment.
fn find_data_segment_start(bytes: &[u8]) -> Option<usize> {
    use std::io::Cursor;

    let mut cursor = Cursor::new(bytes);

    // Skip magic (4 bytes) + version (4 bytes).
    cursor.seek(SeekFrom::Start(8)).ok()?;

    // Read tensor_count and kv_count (8 bytes each).
    let tensor_count = read_u64_le_cursor(&mut cursor)?;
    let kv_count = read_u64_le_cursor(&mut cursor)?;

    // Skip KV entries.
    for _ in 0..kv_count {
        skip_kv_cursor(&mut cursor)?;
    }

    // Skip tensor info entries (name + n_dims + dims + dtype + data_offset).
    for _ in 0..tensor_count {
        skip_tensor_info_cursor(&mut cursor)?;
    }

    // The data segment starts at the next 32-byte boundary.
    let pos = cursor.position() as usize;
    let alignment = 32usize;
    let remainder = pos % alignment;
    if remainder == 0 {
        Some(pos)
    } else {
        Some(pos + (alignment - remainder))
    }
}

// ── cursor-based GGUF navigation helpers ─────────────────────────────────────

/// Read a little-endian u64 from a Cursor<&[u8]>.
fn read_u64_le_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<u64> {
    let mut buf = [0u8; 8];
    c.read_exact(&mut buf).ok()?;
    Some(u64::from_le_bytes(buf))
}

/// Read a little-endian u32 from a Cursor<&[u8]>.
fn read_u32_le_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<u32> {
    let mut buf = [0u8; 4];
    c.read_exact(&mut buf).ok()?;
    Some(u32::from_le_bytes(buf))
}

/// Skip one byte from a Cursor<&[u8]>.
fn read_u8_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<u8> {
    let mut buf = [0u8; 1];
    c.read_exact(&mut buf).ok()?;
    Some(buf[0])
}

/// Skip a GGUF string (u64 length prefix + length bytes).
fn skip_gguf_string_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<()> {
    let len = read_u64_le_cursor(c)? as usize;
    c.seek(SeekFrom::Current(len as i64)).ok()?;
    Some(())
}

/// Skip one GGUF KV entry (key string + value type tag + value).
fn skip_kv_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<()> {
    skip_gguf_string_cursor(c)?; // key
    let vtype = read_u32_le_cursor(c)?;
    skip_kv_value_cursor(c, vtype)?;
    Some(())
}

/// Skip a GGUF KV value by type tag.
fn skip_kv_value_cursor(c: &mut std::io::Cursor<&[u8]>, vtype: u32) -> Option<()> {
    match vtype {
        0 | 1 | 7 => { read_u8_cursor(c)?; }    // UINT8 / INT8 / BOOL
        2 | 3 => { c.seek(SeekFrom::Current(2)).ok()?; }   // UINT16 / INT16
        4 | 5 | 6 => { c.seek(SeekFrom::Current(4)).ok()?; } // UINT32 / INT32 / F32
        8 => { skip_gguf_string_cursor(c)?; }   // STRING
        9 => {
            let elem_type = read_u32_le_cursor(c)?;
            let count = read_u64_le_cursor(c)? as usize;
            for _ in 0..count {
                skip_kv_value_cursor(c, elem_type)?;
            }
        }
        10 | 11 | 12 => { c.seek(SeekFrom::Current(8)).ok()?; } // UINT64 / INT64 / F64
        _ => return None, // unknown type — bail
    }
    Some(())
}

/// Skip one tensor info entry in a GGUF header cursor.
///
/// Layout: name (gguf string) + n_dims (u32) + dims (u64 × n_dims)
///         + dtype (u32) + data_offset (u64)
fn skip_tensor_info_cursor(c: &mut std::io::Cursor<&[u8]>) -> Option<()> {
    skip_gguf_string_cursor(c)?; // name
    let n_dims = read_u32_le_cursor(c)? as usize;
    c.seek(SeekFrom::Current((n_dims as i64) * 8)).ok()?; // dims (u64 each)
    c.seek(SeekFrom::Current(4)).ok()?; // dtype (u32)
    c.seek(SeekFrom::Current(8)).ok()?; // data_offset (u64)
    Some(())
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Task 3.1 / 3.2: key mapping round-trips ───────────────────────────────

    /// Round-trip candle→gguf→candle for all attention projection types.
    #[test]
    fn key_mapping_roundtrip_attn_projections() {
        for proj in ["q", "k", "v", "o"] {
            for layer in [0usize, 1, 10, 99] {
                let candle_key = format!("lora_layer_{layer}_{proj}_proj");
                let gguf_key = KeyMapper::candle_to_gguf(&candle_key).unwrap();
                let back = KeyMapper::gguf_to_candle(&gguf_key).unwrap();
                assert_eq!(back, candle_key, "roundtrip failed for {candle_key}");
            }
        }
    }

    /// Round-trip candle→gguf→candle for all MLP projection types.
    #[test]
    fn key_mapping_roundtrip_mlp_projections() {
        for proj in ["gate", "up", "down"] {
            let candle_key = format!("lora_layer_5_{proj}_proj");
            let gguf_key = KeyMapper::candle_to_gguf(&candle_key).unwrap();
            assert!(gguf_key.contains(".mlp."), "expected mlp module for {proj}");
            let back = KeyMapper::gguf_to_candle(&gguf_key).unwrap();
            assert_eq!(back, candle_key);
        }
    }

    /// candle_to_gguf must return Err for structurally invalid keys.
    #[test]
    fn key_mapping_rejects_malformed_candle_key() {
        assert!(KeyMapper::candle_to_gguf("bad_key").is_err());
        assert!(KeyMapper::candle_to_gguf("lora_layer_notanumber_q_proj").is_err());
    }

    /// gguf_to_candle must return Err for structurally invalid keys.
    #[test]
    fn key_mapping_rejects_malformed_gguf_key() {
        assert!(KeyMapper::gguf_to_candle("model.layers.foo.self_attn.q_proj.weight").is_err());
        assert!(KeyMapper::gguf_to_candle("totally_wrong").is_err());
    }

    // ── Task 3.3: full bijectivity property test ──────────────────────────────

    /// Verify the mapping is bijective for all 7 projection types × layers 0–10.
    #[test]
    fn test_key_mapping_bijectivity() {
        let projections = ["q", "k", "v", "o", "gate", "up", "down"];

        for layer in 0..=10usize {
            for &proj in &projections {
                let candle_key = format!("lora_layer_{layer}_{proj}_proj");
                let gguf_key = KeyMapper::candle_to_gguf(&candle_key)
                    .unwrap_or_else(|e| panic!("candle_to_gguf({candle_key}) failed: {e}"));
                let back_to_candle = KeyMapper::gguf_to_candle(&gguf_key)
                    .unwrap_or_else(|e| panic!("gguf_to_candle({gguf_key}) failed: {e}"));
                assert_eq!(back_to_candle, candle_key);

                let back_to_gguf = KeyMapper::candle_to_gguf(&back_to_candle)
                    .unwrap_or_else(|e| panic!("candle_to_gguf({back_to_candle}) failed: {e}"));
                assert_eq!(back_to_gguf, gguf_key);
            }
        }
    }

    // ── Task 4.1 tests: Q8_0 quantization ────────────────────────────────────

    /// All-zero input must dequantize back to all zeros.
    #[test]
    fn q8_0_roundtrip_all_zeros() {
        let weights = vec![0.0f32; 64];
        let encoded = quantize_q8_0(&weights).unwrap();
        let decoded = dequantize_q8_0(&encoded).unwrap();
        assert_eq!(decoded.len(), weights.len());
        for v in decoded {
            assert_eq!(v, 0.0f32);
        }
    }

    /// Round-trip error must be bounded by scale/127 for a linear ramp.
    #[test]
    fn q8_0_roundtrip_error_bound() {
        let weights: Vec<f32> = (0..32).map(|i| (i as f32) * 0.01 - 0.15).collect();
        let encoded = quantize_q8_0(&weights).unwrap();
        let decoded = dequantize_q8_0(&encoded).unwrap();

        let max_abs = weights.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
        let scale = max_abs / 127.0;

        for (orig, dec) in weights.iter().zip(decoded.iter()) {
            let err = (orig - dec).abs();
            assert!(err <= scale + 1e-5, "error {err} exceeds scale={scale}");
        }
    }

    /// Non-multiple-of-32 input lengths must be rejected.
    #[test]
    fn q8_0_rejects_non_multiple_of_32() {
        assert!(quantize_q8_0(&[0.0f32; 31]).is_err());
        assert!(quantize_q8_0(&[0.0f32; 33]).is_err());
    }

    /// Constant input (all 1.0) must round-trip within tight error bounds.
    #[test]
    fn q8_0_large_values_clamped() {
        let weights = vec![1.0f32; 32];
        let encoded = quantize_q8_0(&weights).unwrap();
        let decoded = dequantize_q8_0(&encoded).unwrap();
        for v in &decoded {
            assert!((v - 1.0).abs() < 0.01, "clamped roundtrip err");
        }
    }

    // ── Task 4.3: quantization round-trip error bound property test ───────────

    fn assert_q8_0_error_bound(label: &str, weights: &[f32]) {
        let encoded = quantize_q8_0(weights)
            .unwrap_or_else(|e| panic!("{label}: quantize failed: {e}"));
        let decoded = dequantize_q8_0(&encoded)
            .unwrap_or_else(|e| panic!("{label}: dequantize failed: {e}"));

        assert_eq!(decoded.len(), weights.len(), "{label}: length mismatch");

        for (block_idx, (orig_block, dec_block)) in weights
            .chunks_exact(Q8_0_BLOCK)
            .zip(decoded.chunks_exact(Q8_0_BLOCK))
            .enumerate()
        {
            let max_abs = orig_block.iter().map(|w| w.abs()).fold(0.0f32, f32::max);
            let scale_f32 = if max_abs == 0.0 { 1.0f32 } else { max_abs / 127.0 };
            let tolerance = scale_f32 / 2.0 + 1e-6;

            for (i, (&orig, &dec)) in orig_block.iter().zip(dec_block.iter()).enumerate() {
                let err = (orig - dec).abs();
                assert!(
                    err <= tolerance,
                    "{label}: block {block_idx} elem {i}: err={err:.2e} > tol={tolerance:.2e}"
                );
            }
        }
    }

    /// Verify round-trip error bounds across diverse input patterns.
    #[test]
    fn test_quantization_roundtrip_error_bounds() {
        assert_q8_0_error_bound("all-zeros", &vec![0.0f32; 64]);
        assert_q8_0_error_bound("all-ones", &vec![1.0f32; 64]);
        let alternating: Vec<f32> = (0..64).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert_q8_0_error_bound("alternating", &alternating);
        let mixed: Vec<f32> = [
            0.1f32, 0.5, -0.3, 0.8, -0.7, 0.2, -0.9, 0.4,
            0.6, -0.1, 0.3, -0.5, 0.9, -0.4, 0.7, -0.2,
            -0.8, 0.15, -0.35, 0.65, 0.45, -0.25, 0.85, -0.55,
            0.05, -0.75, 0.95, -0.15, 0.55, -0.45, 0.25, -0.65,
            0.01, 0.05, -0.03, 0.08, -0.07, 0.02, -0.09, 0.04,
            0.06, -0.01, 0.03, -0.05, 0.09, -0.04, 0.07, -0.02,
            -0.08, 0.015, -0.035, 0.065, 0.045, -0.025, 0.085, -0.055,
            0.005, -0.075, 0.095, -0.015, 0.055, -0.045, 0.025, -0.065,
        ].to_vec();
        assert_q8_0_error_bound("mixed-seed", &mixed);
        assert!(quantize_q8_0(&[0.0f32; 31]).is_err());
        assert!(quantize_q8_0(&[0.0f32; 33]).is_err());
        assert!(quantize_q8_0(&[0.0f32; 1]).is_err());
    }

    // ── Task 5.1 / Wave 3: LoraMerger struct ─────────────────────────────────

    /// Default constructor must produce a 2 GB memory budget.
    #[test]
    fn lora_merger_default_budget() {
        assert_eq!(LoraMerger::new().memory_budget, 2 * 1024 * 1024 * 1024);
    }

    /// with_memory_budget must store exactly the value provided.
    #[test]
    fn lora_merger_custom_budget() {
        let budget = 64 * 1024 * 1024;
        assert_eq!(LoraMerger::with_memory_budget(budget).memory_budget, budget);
    }

    /// Default impl must agree with new().
    #[test]
    fn lora_merger_default_trait() {
        assert_eq!(LoraMerger::default().memory_budget, 2 * 1024 * 1024 * 1024);
    }

    /// merge_into_gguf must reject a non-existent base_path.
    #[test]
    fn merge_into_gguf_rejects_missing_base() {
        let merger = LoraMerger::new();
        let tmp = tempfile::TempDir::new().unwrap();
        let result = merger.merge_into_gguf(
            Path::new("/nonexistent/base.gguf"),
            Path::new("/nonexistent/adapter.safetensors"),
            &tmp.path().join("out.gguf"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("base model path does not exist"));
    }

    /// merge_into_gguf must reject a non-existent adapter_path.
    #[test]
    fn merge_into_gguf_rejects_missing_adapter() {
        let merger = LoraMerger::new();
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path().join("base.gguf");
        std::fs::write(&base, b"fake-gguf").unwrap();
        let result = merger.merge_into_gguf(
            &base,
            Path::new("/nonexistent/adapter.safetensors"),
            &tmp.path().join("out.gguf"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("adapter path does not exist"));
    }

    /// merge_into_gguf must reject an output_path whose parent does not exist.
    #[test]
    fn merge_into_gguf_rejects_missing_output_parent() {
        let merger = LoraMerger::new();
        let tmp = tempfile::TempDir::new().unwrap();
        let base = tmp.path().join("base.gguf");
        std::fs::write(&base, b"fake-gguf").unwrap();
        let adapter = tmp.path().join("adapter.safetensors");
        std::fs::write(&adapter, b"fake-adapter").unwrap();
        let result = merger.merge_into_gguf(
            &base,
            &adapter,
            Path::new("/nonexistent/subdir/out.gguf"),
        );
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("output parent directory does not exist"));
    }

    // ── Wave 4 helpers: synthetic GGUF builder ────────────────────────────────

    /// Build a minimal single-tensor GGUF file in memory with one Q8_0 tensor.
    ///
    /// @INFO The format is: magic(4) + version(4) + tensor_count(8) +
    /// kv_count(8) + [tensor_info]+ + align_pad + [tensor_data]+
    /// With kv_count=0 and a single tensor.
    fn build_minimal_gguf(tensor_name: &str, weights_q8_0: &[u8], n_elements: usize) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Magic: gguf_parser reads a u32 LE and compares to 0x4647_4755.
        // So file bytes must be [0x55, 0x47, 0x47, 0x46] = "UGGF".
        buf.extend_from_slice(&0x4647_4755u32.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        // tensor_count = 1
        buf.extend_from_slice(&1u64.to_le_bytes());
        // kv_count = 0
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Tensor info entry:
        //   name (gguf string: u64 len + bytes)
        let name_bytes = tensor_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u64).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        //   n_dims = 1
        buf.extend_from_slice(&1u32.to_le_bytes());
        //   dims[0] = n_elements (u64)
        buf.extend_from_slice(&(n_elements as u64).to_le_bytes());
        //   dtype = 8 (Q8_0)
        buf.extend_from_slice(&8u32.to_le_bytes());
        //   data_offset = 0 (relative to data segment start)
        buf.extend_from_slice(&0u64.to_le_bytes());

        // Align to 32 bytes.
        let pos = buf.len();
        let remainder = pos % 32;
        if remainder != 0 {
            buf.extend(std::iter::repeat(0u8).take(32 - remainder));
        }

        // Tensor data.
        buf.extend_from_slice(weights_q8_0);
        buf
    }

    /// Build a SafeTensors adapter file for a single lora_a/lora_b pair.
    ///
    /// Uses the same wire format as `LoraExporter::export_safetensors`.
    fn build_adapter_safetensors(
        layer_name: &str,
        lora_a: &[f32],
        a_shape: &[usize],
        lora_b: &[f32],
        b_shape: &[usize],
    ) -> Vec<u8> {
        // Build data blob.
        let mut data: Vec<u8> = Vec::new();
        let a_start = data.len();
        for &v in lora_a {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let a_end = data.len();
        let b_start = data.len();
        for &v in lora_b {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let b_end = data.len();

        // Build JSON header.
        let header = serde_json::json!({
            format!("{layer_name}.lora_a"): {
                "dtype": "F32",
                "shape": a_shape,
                "data_offsets": [a_start, a_end]
            },
            format!("{layer_name}.lora_b"): {
                "dtype": "F32",
                "shape": b_shape,
                "data_offsets": [b_start, b_end]
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut out: Vec<u8> = Vec::new();
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(&header_bytes);
        out.extend_from_slice(&data);
        out
    }

    // ── Wave 4 core tests ─────────────────────────────────────────────────────

    /// Merging an all-zero adapter must produce an output identical to the base.
    ///
    /// @INFO A zero adapter means lora_a = 0, lora_b = 0 → Δ = 0 → merged = base.
    /// After requantization the values may differ by at most the Q8_0 rounding
    /// error, so we check element-wise error rather than byte equality.
    #[test]
    fn test_merge_identity() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Build 64 Q8_0 weights (2 blocks of 32) with known values.
        let base_weights: Vec<f32> = (0..64).map(|i| (i as f32) * 0.01).collect();
        let base_q8 = quantize_q8_0(&base_weights).unwrap();
        let n_elements = 64usize;

        // GGUF tensor name for layer 0, q projection.
        let gguf_name = "model.layers.0.self_attn.q_proj.weight";

        // Corresponding candle LoRA key.
        let candle_name = "lora_layer_0_q_proj";

        // Build synthetic GGUF.
        let gguf_bytes = build_minimal_gguf(gguf_name, &base_q8, n_elements);
        let base_path = tmp.path().join("base.gguf");
        std::fs::write(&base_path, &gguf_bytes).unwrap();

        // Build zero adapter: rank=4, d_in=16, d_out=4 → Δ = 0.
        // lora_a shape (rank=4, d_in=16), lora_b shape (d_out=4, rank=4)
        // For 64-element weight the shape is (1D: 64 elements); adapter
        // compute_delta gives (d_out=4, d_in=16)=64 elements total.
        let rank = 4usize;
        let d_in = 16usize;
        let d_out = 4usize;
        let lora_a_data = vec![0.0f32; rank * d_in];
        let lora_b_data = vec![0.0f32; d_out * rank];

        let adapter_bytes = build_adapter_safetensors(
            candle_name,
            &lora_a_data, &[rank, d_in],
            &lora_b_data, &[d_out, rank],
        );
        let adapter_path = tmp.path().join("adapter.safetensors");
        std::fs::write(&adapter_path, &adapter_bytes).unwrap();

        let output_path = tmp.path().join("out.gguf");
        let merger = LoraMerger::new();
        merger.merge_into_gguf(&base_path, &adapter_path, &output_path).unwrap();

        // Read merged output and locate the tensor data.
        let out_bytes = std::fs::read(&output_path).unwrap();
        // Find data segment start in output.
        let data_start = find_data_segment_start(&out_bytes).unwrap();
        let merged_q8 = &out_bytes[data_start..data_start + base_q8.len()];
        let merged_f32 = dequantize_q8_0(merged_q8).unwrap();
        let base_f32_rt = dequantize_q8_0(&base_q8).unwrap();

        // After round-tripping through Q8_0 twice, error should be small.
        for (i, (&m, &b)) in merged_f32.iter().zip(base_f32_rt.iter()).enumerate() {
            let err = (m - b).abs();
            assert!(err < 0.05, "identity merge mismatch at {i}: merged={m} base={b} err={err}");
        }
    }

    /// A shape mismatch between adapter delta and base weights must return ShapeMismatch.
    #[test]
    fn test_merge_shape_mismatch() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Base GGUF: 64 Q8_0 elements.
        let base_weights = vec![0.1f32; 64];
        let base_q8 = quantize_q8_0(&base_weights).unwrap();
        let gguf_bytes = build_minimal_gguf(
            "model.layers.0.self_attn.q_proj.weight",
            &base_q8, 64,
        );
        let base_path = tmp.path().join("base.gguf");
        std::fs::write(&base_path, &gguf_bytes).unwrap();

        // Adapter with wrong shape: rank=4, d_in=8, d_out=4 → delta has 32 elements,
        // but base has 64 → ShapeMismatch.
        let rank = 4usize;
        let lora_a = vec![0.1f32; rank * 8];   // (4, 8)
        let lora_b = vec![0.1f32; 4 * rank];    // (4, 4) → delta (4,8) = 32 elements ≠ 64
        let adapter_bytes = build_adapter_safetensors(
            "lora_layer_0_q_proj",
            &lora_a, &[rank, 8],
            &lora_b, &[4, rank],
        );
        let adapter_path = tmp.path().join("adapter.safetensors");
        std::fs::write(&adapter_path, &adapter_bytes).unwrap();

        let result = LoraMerger::new().merge_into_gguf(
            &base_path,
            &adapter_path,
            &tmp.path().join("out.gguf"),
        );
        assert!(
            matches!(result, Err(GwenError::ShapeMismatch { .. })),
            "expected ShapeMismatch, got: {:?}",
            result.err()
        );
    }

    /// Injecting NaN into the adapter delta must produce InvalidMergedWeights.
    #[test]
    fn test_merge_nan_detection() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Base GGUF: 64 Q8_0 elements.
        let base_weights = vec![0.1f32; 64];
        let base_q8 = quantize_q8_0(&base_weights).unwrap();
        let gguf_bytes = build_minimal_gguf(
            "model.layers.0.self_attn.q_proj.weight",
            &base_q8, 64,
        );
        let base_path = tmp.path().join("base.gguf");
        std::fs::write(&base_path, &gguf_bytes).unwrap();

        // Adapter: rank=4, d_in=16, d_out=4 → delta (4×16)=64 elements.
        // Inject NaN into lora_b to produce NaN in the delta.
        let rank = 4usize;
        let d_in = 16usize;
        let d_out = 4usize;
        let lora_a = vec![1.0f32; rank * d_in];
        let mut lora_b = vec![1.0f32; d_out * rank];
        lora_b[0] = f32::NAN; // inject NaN → delta contains NaN

        let adapter_bytes = build_adapter_safetensors(
            "lora_layer_0_q_proj",
            &lora_a, &[rank, d_in],
            &lora_b, &[d_out, rank],
        );
        let adapter_path = tmp.path().join("adapter.safetensors");
        std::fs::write(&adapter_path, &adapter_bytes).unwrap();

        let result = LoraMerger::new().merge_into_gguf(
            &base_path,
            &adapter_path,
            &tmp.path().join("out.gguf"),
        );
        assert!(
            matches!(result, Err(GwenError::InvalidMergedWeights { .. })),
            "expected InvalidMergedWeights, got: {:?}",
            result.err()
        );
    }
}
