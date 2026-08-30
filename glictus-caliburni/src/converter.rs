//! GGUF → GLLM converter (ARTX07-lite). Feature-gated behind
//! `converter` so the core format library stays zero-workspace-dep.
//!
//! Pipeline (per ARTX7): parse GGUF → group tensors into shared/layers →
//! write unit files via [`crate::layer_io::write_unit_file`] → generate
//! `gllm.json` + `checksums.sha256` → re-open with [`GllmPackage`] and
//! cross-check every layer as the final validation gate.
//!
//! Declared deviations from the ARTX7 spec text (lite scope):
//! - RoPE tables are NOT materialized into `GLLMShared.gllm` — they are
//!   derivable (Minimalism principle); the runtime computes them.
//! - Unmapped non-`blk.*` tensors go to `GLLMShared.gllm` under their
//!   original names (with a warning), not to `GLLMProj.gllm`.
//! - Tokenizer is NOT packaged (the spec's own open question #3); a
//!   warning is emitted. Runtimes must source the tokenizer elsewhere
//!   until that question is settled.
//! - GGUF dims are fastest-moving-first; manifest/binary shapes are
//!   written row-major (reversed).

pub mod gquant_policy;

use std::collections::BTreeMap;
use std::path::Path;

use glcore::format::gguf::{GgufDType, GgufFile};

use crate::checksum::sha256_file;
use crate::constants::{CHECKSUMS_FILENAME, MANIFEST_FILENAME, SHARED_FILENAME};
use crate::error::GllmError;
use crate::gquant::{GQ2ABlock, GQ4ABlock};
use crate::gquant::encoder::{encode_gq2a_tensor, encode_gq4a_tensor, f32_to_f16};
use crate::layer_io::write_unit_file;
use crate::manifest::{
    CustomMetadata, DType, ExtensionUri, FormatVersion, GllmManifest, LayerManifest,
    ModelMetadata, RUNTIME_FORMAT_VERSION, SharedManifest, TensorEntry, format_layer_filename,
    known_extensions,
};
use crate::package::GllmPackage;

/// G-Quant target format for `--quant` (Pridwen v5 §2). GQ1A is a later
/// phase and has no encoder yet.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantTarget {
    /// No G-Quant conversion — tensors keep their original GGUF dtype
    /// (existing ARTX7-lite behavior).
    #[default]
    None,
    /// GQ4A, Architecture A 4-bit foundation (Pridwen v5 §3.1).
    Gq4a,
    /// GQ2A, Architecture A 2-bit asymmetric superblock (Pridwen v5 §3.2) —
    /// heterogeneous with GQ4A as an escape hatch for EXTREME/HIGH
    /// sensitivity tensors under CPP (see `gquant_policy::assign_gq2a_cpp`).
    Gq2a,
    /// Diagnostic only. Every tensor is dequantized to real F32 bytes,
    /// regardless of its original GGUF dtype — no GQ4A/GQ2A encoder is
    /// touched. This is NOT `None` (which keeps each tensor's original GGUF
    /// dtype, including Q4_K/Q5_0/Q6_K/etc — dtypes the GLLM runtime cannot
    /// read at all today). `F32` exists to isolate whether a garbage-output
    /// bug is in G-Quant dequantization or somewhere deeper (attention,
    /// RoPE, embedding lookup): a package with zero quantized tensors is the
    /// cleanest possible control group. Packages are uncompressed and huge
    /// (~4x a Q4_K_M source) — never a shipping format.
    F32,
}

/// Assignment policy for `--policy` (Pridwen v5 §4). Phase 1 scope: CPP
/// Stage 1 (hardcoded sensitivity table) only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantPolicy {
    /// Combined Precision Policy, Stage 1 (Pridwen v5 §5, §7).
    #[default]
    Cpp,
}

/// Options for a conversion run.
#[derive(Debug, Default)]
pub struct ConvertOptions {
    /// Override for the manifest `model_id` (default: GGUF `general.name`,
    /// falling back to the input file stem).
    pub model_id: Option<String>,
    /// `--quant` target (default: no G-Quant conversion).
    pub quant: QuantTarget,
    /// `--policy` assignment strategy (only consulted when `quant` is set).
    pub policy: QuantPolicy,
}

/// Summary of a successful conversion.
#[derive(Debug)]
pub struct ConvertReport {
    /// Final `model_id` written to the manifest.
    pub model_id: String,
    /// Number of layer files written.
    pub num_layers: u32,
    /// Tensors placed in `GLLMShared.gllm`.
    pub shared_tensors: usize,
    /// Non-fatal notes (unmapped tensors, tokenizer skip, …).
    pub warnings: Vec<String>,
    /// EOS token ids extracted from the source GGUF (see
    /// `extract_eos_token_ids`), same value written to
    /// `ModelMetadata::eos_token_ids`. Surfaced here so the CLI can report
    /// what it found without re-deriving it.
    pub eos_token_ids: Vec<u32>,
}

fn convert_err(msg: impl Into<String>) -> GllmError {
    GllmError::ValidationError(format!("converter: {}", msg.into()))
}

/// Map a GGUF/GGML dtype onto the GLLM dtype table. Loud error on types
/// the format cannot represent — never a silent fallback.
fn map_dtype(dtype: GgufDType, tensor: &str) -> Result<DType, GllmError> {
    match dtype {
        GgufDType::F32 => Ok(DType::F32),
        GgufDType::F16 => Ok(DType::F16),
        GgufDType::BF16 => Ok(DType::Bf16),
        GgufDType::Q4_0 => Ok(DType::Q4_0),
        GgufDType::Q5_0 => Ok(DType::Q5_0),
        GgufDType::Q8_0 => Ok(DType::Q8_0),
        GgufDType::Q4_K => Ok(DType::Q4K),
        GgufDType::Q6_K => Ok(DType::Q6K),
        GgufDType::Unknown(id) => Err(convert_err(format!(
            "tensor {tensor}: unsupported ggml type id {id} (E004)"
        ))),
    }
}

/// GGUF stores dims fastest-moving first; GLLM shapes are row-major.
fn row_major_shape(gguf_dims: &[u64]) -> Vec<u64> {
    gguf_dims.iter().rev().copied().collect()
}

/// Whether this GGUF source dtype can be decoded to F32 for G-Quant
/// (GQ4A/GQ2A) re-encoding, considering both `glcore::GgufFile::dequantize`
/// and the glproc fallback added by the Pridwen Phase 2 ADR
/// (architecture/Pridwen-P2-ADR-glproc-dequant.md). Mirrors
/// [`dequantize_for_gquant`]'s own match arms rather than calling it
/// speculatively, so the assignment step can decide *before* committing to
/// a quantized dtype (and thus before any warning/error bookkeeping gets
/// tangled with the write pass).
fn gguf_dtype_is_dequantizable(dtype: GgufDType) -> bool {
    matches!(
        dtype,
        GgufDType::F32 | GgufDType::F16 | GgufDType::BF16
            | GgufDType::Q4_0 | GgufDType::Q8_0 | GgufDType::Q6_K
            | GgufDType::Q4_K | GgufDType::Q5_0
    )
}

/// Dequantize a tensor to F32 for G-Quant (GQ4A or GQ2A) re-encoding, using
/// `glcore`'s path for everything it supports and falling back to
/// `glproc`'s scalar dequant kernels for `Q4_K`/`Q5_0` (which
/// `glcore::GgufFile::dequantize` deliberately rejects — see its own doc
/// comment: "dequant lives in glproc") and for `Q6_K` (which `glcore` DOES
/// accept but gets wrong: `glcore::format::gguf::dequant_q6_k` assumes a
/// naive linear nibble order, while `glproc::kernels::dequant::q6_k::scalar`
/// implements GGML's real two-half interleaved layout — see that module's
/// own doc comment, which flags the disagreement explicitly. Confirmed via
/// `diff_dump.rs` on a real Q4_K_M model: every layer's Q6_K-sourced
/// `ffn_down.weight` was silently corrupted by the wrong nibble order,
/// which is what produced garbage `.gllm` output while
/// `glproc::runner::Runner` (which never calls `glcore`'s Q6_K path) stayed
/// coherent on the identical GGUF. This is the one place `converter`
/// crosses into `glproc`; every other tensor still goes through `glcore`
/// unchanged (Pridwen Phase 2 ADR). Shared by both GQ4A and GQ2A encoding —
/// the dequant-to-F32 step is identical regardless of which superblock
/// format the F32 buffer then gets re-encoded into.
fn dequantize_for_gquant(gguf: &GgufFile, info: &glcore::format::gguf::GgufTensorInfo) -> Result<Vec<f32>, GllmError> {
    match info.dtype {
        GgufDType::Q4_K | GgufDType::Q5_0 | GgufDType::Q6_K => {
            let raw = gguf
                .tensor_data(info)
                .map_err(|e| convert_err(format!("tensor {}: {e}", info.name)))?;
            let dequant = match info.dtype {
                GgufDType::Q4_K => glproc::kernels::dequant::q4_k::scalar::run(raw),
                GgufDType::Q5_0 => glproc::kernels::dequant::q5_0::scalar::run(raw),
                GgufDType::Q6_K => glproc::kernels::dequant::q6_k::scalar::run(raw),
                _ => unreachable!(),
            };
            dequant.map_err(|e| {
                convert_err(format!("tensor {}: glproc dequant failed: {e}", info.name))
            })
        }
        _ => gguf
            .dequantize(info)
            .map_err(|e| convert_err(format!("tensor {}: dequant for G-Quant encode failed: {e}", info.name))),
    }
}

/// Where a GGUF tensor lands in the GLLM package.
#[derive(Debug, PartialEq, Eq)]
enum Dest {
    /// `GLLMShared.gllm`, under the given GLLM tensor name.
    Shared(String),
    /// `GLLMTensorLayer-NNNN.gllm`, under the given (stripped) tensor name.
    Layer(u32, String),
    /// Non-standard tensor routed to shared under its original name.
    SharedUnmapped(String),
}

fn map_tensor_name(name: &str) -> Result<Dest, GllmError> {
    match name {
        "token_embd.weight" => Ok(Dest::Shared("token_embeddings".into())),
        "output_norm.weight" => Ok(Dest::Shared("output_norm.weight".into())),
        "output.weight" => Ok(Dest::Shared("output_head.weight".into())),
        _ => {
            if let Some(rest) = name.strip_prefix("blk.") {
                let (idx, tensor) = rest.split_once('.').ok_or_else(|| {
                    convert_err(format!("malformed layer tensor name {name:?}"))
                })?;
                let idx: u32 = idx.parse().map_err(|_| {
                    convert_err(format!("non-numeric layer index in {name:?}"))
                })?;
                Ok(Dest::Layer(idx, tensor.to_string()))
            } else {
                Ok(Dest::SharedUnmapped(name.to_string()))
            }
        }
    }
}

/// Fetch `{arch}.{key}` from GGUF metadata as u64.
fn arch_meta_u64(gguf: &GgufFile, arch: &str, key: &str) -> Option<u64> {
    gguf.get_meta(&format!("{arch}.{key}")).and_then(|v| v.as_u64())
}

/// Extract EOS token ids from GGUF tokenizer metadata.
///
/// `tokenizer.ggml.eos_token_ids` (an array — the less common, but more
/// complete key some newer conversions write for multi-EOS models) *overrides*
/// `tokenizer.ggml.eos_token_id` (the single, standard key) when present,
/// rather than merging the two — an array author who wrote down a specific
/// set meant that set, not "this plus whatever the singular key also says."
/// Deduplicated and sorted for a stable, comparable result.
///
/// Empty when neither key is present — not an error. Some models
/// legitimately omit EOS metadata; a `.gllm` package converted from one
/// simply has no manifest-level stop ids and falls back to whatever
/// `InferInput::stopping` a caller supplies (see `glcore::stopping`).
fn extract_eos_token_ids(gguf: &GgufFile) -> Vec<u32> {
    let mut ids: Vec<u32> = gguf
        .get_meta("tokenizer.ggml.eos_token_ids")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_u64()).map(|v| v as u32).collect())
        .unwrap_or_default();

    if ids.is_empty() {
        if let Some(id) = gguf.get_meta("tokenizer.ggml.eos_token_id").and_then(|v| v.as_u64()) {
            ids.push(id as u32);
        }
    }

    ids.sort_unstable();
    ids.dedup();
    ids
}

/// Convert a GGUF file into a GLLM package directory.
///
/// Writes `gllm.json`, `GLLMShared.gllm`, `GLLMTensorLayer-NNNN.gllm`, and
/// `checksums.sha256` into `out_dir` (created if absent), then re-opens
/// the result with [`GllmPackage::open`] and cross-checks every layer —
/// the converter validates its own output rather than trusting the write
/// path (ARTX7).
pub fn convert(input: &Path, out_dir: &Path, opts: &ConvertOptions) -> Result<ConvertReport, GllmError> {
    let input_str = input
        .to_str()
        .ok_or_else(|| convert_err("input path is not valid UTF-8"))?;
    let gguf = GgufFile::open(input_str).map_err(|e| convert_err(format!("{input_str}: {e}")))?;
    let mut warnings = Vec::new();

    // --- Metadata mapping (ARTX7 §Metadata Extraction) ---
    let arch = gguf
        .get_meta("general.architecture")
        .and_then(|v| v.as_str())
        .ok_or_else(|| convert_err("missing general.architecture (E002)"))?
        .to_string();
    let num_layers = arch_meta_u64(&gguf, &arch, "block_count")
        .ok_or_else(|| convert_err(format!("missing {arch}.block_count (E002)")))? as u32;
    let embedding_length = arch_meta_u64(&gguf, &arch, "embedding_length")
        .ok_or_else(|| convert_err(format!("missing {arch}.embedding_length (E002)")))?;
    let context_length = arch_meta_u64(&gguf, &arch, "context_length")
        .ok_or_else(|| convert_err(format!("missing {arch}.context_length (E002)")))?;
    let num_heads = arch_meta_u64(&gguf, &arch, "attention.head_count")
        .ok_or_else(|| convert_err(format!("missing {arch}.attention.head_count (E002)")))?
        as u32;
    let head_count_kv =
        arch_meta_u64(&gguf, &arch, "attention.head_count_kv").map_or(num_heads, |v| v as u32);

    let vocab_size = arch_meta_u64(&gguf, &arch, "vocab_size")
        .or_else(|| {
            gguf.get_meta("tokenizer.ggml.tokens")
                .and_then(|v| v.as_array())
                .map(|a| a.len() as u64)
        })
        .or_else(|| {
            // Last resort: token_embd GGUF dims are [D, V] fastest-first.
            gguf.find_tensor("token_embd.weight")
                .and_then(|t| t.dimensions.last().copied())
        })
        .ok_or_else(|| convert_err("cannot determine vocab_size (E002)"))?;

    // Not a warning when found — that's the success path, and the CLI
    // reports it as its own info line from `ConvertReport::eos_token_ids`
    // (see `glconv.rs`) rather than mixing it into `warnings`, which is
    // reserved for non-fatal *problems*.
    let eos_token_ids = extract_eos_token_ids(&gguf);
    if eos_token_ids.is_empty() {
        warnings.push(
            "no tokenizer.ggml.eos_token_id(s) found in source GGUF; this package cannot \
             self-stop at end-of-sequence and will fall back to whatever InferInput::stopping \
             a caller supplies"
                .to_string(),
        );
    }

    let metadata = ModelMetadata {
        vocab_size,
        context_length,
        embedding_length,
        num_layers,
        num_heads,
        head_count_kv,
        rope_dims: arch_meta_u64(&gguf, &arch, "rope.dimension_count").map(|v| v as u32),
        rope_freq_base: gguf
            .get_meta(&format!("{arch}.rope.freq_base"))
            .and_then(|v| v.as_f32())
            .map(f64::from),
        rope_scaling: None,
        expert_count: arch_meta_u64(&gguf, &arch, "expert_count").map(|v| v as u32),
        expert_used_count: arch_meta_u64(&gguf, &arch, "expert_used_count").map(|v| v as u32),
        sliding_window: arch_meta_u64(&gguf, &arch, "attention.sliding_window").map(|v| v as u32),
        attention_bias: None,
        rms_eps: gguf
            .get_meta(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .and_then(|v| v.as_f32())
            .map(f64::from),
        eos_token_ids: eos_token_ids.clone(),
    };

    let model_id = opts
        .model_id
        .clone()
        .or_else(|| {
            gguf.get_meta("general.name")
                .and_then(|v| v.as_str())
                .map(str::to_string)
        })
        .or_else(|| input.file_stem().and_then(|s| s.to_str()).map(str::to_string))
        .ok_or_else(|| convert_err("cannot determine model_id; pass --model-id"))?;

    // --- Tensor grouping (ARTX7 §Layer Extraction), preserving GGUF file order ---
    struct Planned {
        gllm_name: String,
        shape: Vec<u64>,
        dtype: DType,
        gguf_index: usize,
    }
    let mut shared_plan: Vec<Planned> = Vec::new();
    let mut layer_plan: BTreeMap<u32, Vec<Planned>> = BTreeMap::new();
    for (i, info) in gguf.tensors.iter().enumerate() {
        let mut dtype = map_dtype(info.dtype, &info.name)?;
        let shape = row_major_shape(&info.dimensions);

        // Diagnostic F32 passthrough (see `QuantTarget::F32`'s doc comment):
        // every tensor becomes real F32, unconditionally — no sensitivity
        // table, no superblock eligibility check (F32 has no block size),
        // no CPP policy involved at all. Checked before the G-Quant branch
        // below so a `--quant F32` run never consults `gquant_policy`.
        if opts.quant == QuantTarget::F32 {
            dtype = DType::F32;
        } else if opts.quant != QuantTarget::None {
            // G-Quant assignment (Pridwen v5 §9 step 3): CPP Stage 1
            // hardcoded sensitivity table overrides the GGUF-native dtype.
            // Both GQ4A and GQ2A are 256-weight-superblock formats, so the
            // same eligibility constraints apply to whichever one the
            // policy assigns: only tensors whose element count divides
            // evenly into 256 are eligible (same constraint GGUF's own
            // Q4_K/Q6_K already impose), and the source dtype must have an
            // available dequant path. Anything that fails either check
            // keeps its original dtype and gets a warning rather than a
            // padded/truncated block the spec doesn't define (see
            // notes/pridwen-p1-notes.md ragged-dim deviation).
            debug_assert_eq!(opts.policy, QuantPolicy::Cpp, "only CPP is implemented so far");
            let assigned = match opts.quant {
                QuantTarget::Gq4a => gquant_policy::assign_gq4a_cpp(&info.name),
                QuantTarget::Gq2a => gquant_policy::assign_gq2a_cpp(&info.name),
                QuantTarget::None | QuantTarget::F32 => {
                    unreachable!("guarded by the outer if/else-if")
                }
            };
            if let Some(assigned) = assigned {
                if assigned == DType::GQ4A || assigned == DType::GQ2A {
                    let superblock_weights = match assigned {
                        DType::GQ4A => GQ4ABlock::WEIGHTS,
                        DType::GQ2A => GQ2ABlock::WEIGHTS,
                        _ => unreachable!("guarded above"),
                    };
                    if !info.numel().is_multiple_of(superblock_weights) {
                        warnings.push(format!(
                            "tensor {:?}: CPP assigned {:?} but numel {} is not a multiple of {}; \
                             keeping original dtype {:?}",
                            info.name, assigned, info.numel(), superblock_weights, dtype
                        ));
                    } else if !gguf_dtype_is_dequantizable(info.dtype) {
                        warnings.push(format!(
                            "tensor {:?}: CPP assigned {:?} but source dtype {:?} has no dequant \
                             path (glcore or glproc); keeping original dtype {:?}",
                            info.name, assigned, info.dtype, dtype
                        ));
                    } else {
                        dtype = assigned;
                    }
                } else {
                    dtype = assigned;
                }
            }
        }

        let planned = |gllm_name: String| Planned {
            gllm_name,
            shape: shape.clone(),
            dtype,
            gguf_index: i,
        };
        match map_tensor_name(&info.name)? {
            Dest::Shared(name) => shared_plan.push(planned(name)),
            Dest::SharedUnmapped(name) => {
                warnings.push(format!(
                    "tensor {name:?} has no standard GLLM mapping; stored in GLLMShared.gllm as-is"
                ));
                shared_plan.push(planned(name));
            }
            Dest::Layer(idx, name) => {
                if idx >= num_layers {
                    return Err(convert_err(format!(
                        "tensor {} claims layer {idx} but {arch}.block_count is {num_layers}",
                        info.name
                    )));
                }
                layer_plan.entry(idx).or_default().push(planned(name));
            }
        }
    }
    for idx in 0..num_layers {
        if !layer_plan.contains_key(&idx) {
            return Err(convert_err(format!(
                "no tensors found for layer {idx} (expected {num_layers} layers)"
            )));
        }
    }
    if gguf.get_meta("tokenizer.ggml.tokens").is_some() {
        warnings.push(
            "tokenizer metadata present in GGUF but NOT packaged — GLLM tokenizer \
             packaging is an open spec question (ARTX1 OQ3)"
                .into(),
        );
    }

    // --- Write unit files ---
    std::fs::create_dir_all(out_dir)?;
    let write_group = |plans: &[Planned], path: &Path| -> Result<Vec<TensorEntry>, GllmError> {
        // Owned buffer per tensor: G-Quant-assigned tensors are re-encoded
        // from a freshly dequantized F32 buffer (not a borrow of the
        // mmap'd GGUF bytes), so every entry needs to own its bytes
        // uniformly rather than mixing borrowed-vs-owned across the same Vec.
        let mut datas: Vec<Vec<u8>> = Vec::with_capacity(plans.len());
        for p in plans {
            let info = &gguf.tensors[p.gguf_index];
            if p.dtype == DType::GQ4A {
                let f32_weights = dequantize_for_gquant(&gguf, info)?;
                let blocks = encode_gq4a_tensor(&f32_weights).ok_or_else(|| {
                    convert_err(format!(
                        "tensor {}: numel {} not a multiple of 256 (assignment step should have caught this)",
                        info.name, f32_weights.len()
                    ))
                })?;
                let mut bytes = Vec::with_capacity(blocks.len() * GQ4ABlock::BYTES);
                for block in &blocks {
                    bytes.extend_from_slice(&block.super_scale.to_le_bytes());
                    for d in block.scale_delta {
                        bytes.push(d as u8);
                    }
                    bytes.extend_from_slice(&block.weights);
                }
                datas.push(bytes);
            } else if p.dtype == DType::GQ2A {
                let f32_weights = dequantize_for_gquant(&gguf, info)?;
                let blocks = encode_gq2a_tensor(&f32_weights).ok_or_else(|| {
                    convert_err(format!(
                        "tensor {}: numel {} not a multiple of 256 (assignment step should have caught this)",
                        info.name, f32_weights.len()
                    ))
                })?;
                let mut bytes = Vec::with_capacity(blocks.len() * GQ2ABlock::BYTES);
                for block in &blocks {
                    bytes.extend_from_slice(&block.super_scale.to_le_bytes());
                    bytes.extend_from_slice(&block.super_min.to_le_bytes());
                    bytes.extend_from_slice(&block.scale_delta);
                    bytes.extend_from_slice(&block.min_delta);
                    bytes.extend_from_slice(&block.weights);
                }
                datas.push(bytes);
            } else if p.dtype == DType::F32 && info.dtype != GgufDType::F32 {
                // `--quant F32` diagnostic passthrough (or any future CPP
                // "F32 always" row whose source isn't already F32): actually
                // dequantize, don't just relabel — the same corruption class
                // the F16-norm branch below already exists to prevent. Reuses
                // `dequantize_for_gquant` (Q4_K/Q5_0 via glproc, everything
                // else via glcore) even though no G-Quant re-encoding follows;
                // it is simply "decode this tensor to f32", which is exactly
                // what this branch needs.
                let f32_weights = dequantize_for_gquant(&gguf, info)?;
                let bytes: Vec<u8> = f32_weights.iter().flat_map(|f| f.to_le_bytes()).collect();
                datas.push(bytes);
            } else if p.dtype == DType::F16 && info.dtype == GgufDType::F32 {
                // CPP's "F16 always" norm-tensor assignment (attn_norm/
                // ffn_norm) relabels the manifest dtype without the source
                // GGUF ever having been F16 — this model's norms are F32.
                // Actually narrow the bytes here; previously this branch
                // fell through to the plain copy below, which wrote raw F32
                // bytes under a manifest entry claiming F16 (half the bytes
                // the shape/dtype pair promised), corrupting every reader
                // that trusts the manifest's dtype to size its read.
                let f32_bytes = gguf
                    .tensor_data(info)
                    .map_err(|e| convert_err(format!("tensor {}: {e}", info.name)))?;
                let f16_bytes: Vec<u8> = f32_bytes
                    .chunks_exact(4)
                    .flat_map(|b| f32_to_f16(f32::from_le_bytes([b[0], b[1], b[2], b[3]])).to_le_bytes())
                    .collect();
                datas.push(f16_bytes);
            } else {
                let bytes = gguf
                    .tensor_data(info)
                    .map_err(|e| convert_err(format!("tensor {}: {e}", info.name)))?;
                datas.push(bytes.to_vec());
            }
        }
        let spec: Vec<(&str, &[u64], DType, &[u8])> = plans
            .iter()
            .zip(&datas)
            .map(|(p, d)| (p.gllm_name.as_str(), p.shape.as_slice(), p.dtype, d.as_slice()))
            .collect();
        write_unit_file(path, &spec)
    };

    let shared_entries = write_group(&shared_plan, &out_dir.join(SHARED_FILENAME))?;
    let mut layers: Vec<LayerManifest> = Vec::with_capacity(num_layers as usize);
    let standard_uri = ExtensionUri(known_extensions::TRANSFORMER_STANDARD.to_string());
    for (idx, plans) in &layer_plan {
        let file = format_layer_filename(*idx);
        let entries = write_group(plans, &out_dir.join(&file))?;
        let checksum = format!("sha256:{}", sha256_file(&out_dir.join(&file))?);
        layers.push(LayerManifest {
            index: *idx,
            file,
            checksum,
            layer_type: standard_uri.clone(),
            tensors: entries,
            device: None,
        });
    }

    // --- Manifest + checksums file ---
    let quantization = {
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for p in layer_plan.values().flatten().filter(|p| p.dtype.is_quantized()) {
            let name = serde_json::to_string(&p.dtype)
                .map(|s| s.trim_matches('"').to_string())
                .unwrap_or_default();
            *counts.entry(name).or_default() += 1;
        }
        counts.into_iter().max_by_key(|(_, n)| *n).map(|(name, _)| name)
    };
    let manifest = GllmManifest {
        gllm_version: FormatVersion(RUNTIME_FORMAT_VERSION.to_string()),
        model_id: model_id.clone(),
        architecture: arch,
        parameters: None,
        quantization,
        metadata,
        shared: SharedManifest {
            file: SHARED_FILENAME.to_string(),
            checksum: format!("sha256:{}", sha256_file(&out_dir.join(SHARED_FILENAME))?),
            tensors: shared_entries,
        },
        layers,
        projector: None,
        extensions: vec![standard_uri],
        custom_metadata: CustomMetadata::default(),
    };
    std::fs::write(out_dir.join(MANIFEST_FILENAME), manifest.to_json_pretty()?)?;

    let mut checksum_lines = String::new();
    checksum_lines.push_str(&format!(
        "{}  {}\n",
        manifest.shared.checksum_hex()?,
        SHARED_FILENAME
    ));
    for layer in &manifest.layers {
        checksum_lines.push_str(&format!("{}  {}\n", layer.checksum_hex()?, layer.file));
    }
    std::fs::write(out_dir.join(CHECKSUMS_FILENAME), checksum_lines)?;

    // --- Final gate: re-open + cross-check own output (ARTX7 §Validation) ---
    let pkg = GllmPackage::open(out_dir)?;
    for idx in 0..num_layers {
        let mismatches = pkg.cross_check_layer(idx)?;
        if !mismatches.is_empty() {
            return Err(GllmError::IntegrityError(format!(
                "converter self-check failed for layer {idx}: {mismatches:?}"
            )));
        }
    }
    let failures = pkg.verify_integrity();
    if !failures.is_empty() {
        return Err(GllmError::IntegrityError(format!(
            "converter self-check: checksum failures {failures:?}"
        )));
    }

    Ok(ConvertReport {
        model_id,
        num_layers,
        shared_tensors: shared_plan.len(),
        warnings,
        eos_token_ids,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write as _;

    // --- Minimal synthetic GGUF builder (v3, little-endian) ---

    fn gstr(buf: &mut Vec<u8>, s: &str) {
        buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
        buf.extend_from_slice(s.as_bytes());
    }

    fn meta_str(buf: &mut Vec<u8>, key: &str, val: &str) {
        gstr(buf, key);
        buf.extend_from_slice(&8u32.to_le_bytes());
        gstr(buf, val);
    }

    fn meta_u32(buf: &mut Vec<u8>, key: &str, val: u32) {
        gstr(buf, key);
        buf.extend_from_slice(&4u32.to_le_bytes());
        buf.extend_from_slice(&val.to_le_bytes());
    }

    /// F32 tensor: (name, gguf_dims fastest-first). Data = index-valued f32s.
    fn synth_gguf(tensors: &[(&str, &[u64])], extra_meta: &[(&str, u32)]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&glcore::format::gguf::GGUF_MAGIC.to_le_bytes());
        buf.extend_from_slice(&3u32.to_le_bytes());
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        let base_meta = 7 + extra_meta.len() as u64;
        buf.extend_from_slice(&base_meta.to_le_bytes());

        meta_str(&mut buf, "general.architecture", "llama");
        meta_str(&mut buf, "general.name", "synth-model");
        meta_u32(&mut buf, "llama.block_count", 2);
        meta_u32(&mut buf, "llama.context_length", 128);
        meta_u32(&mut buf, "llama.embedding_length", 8);
        meta_u32(&mut buf, "llama.attention.head_count", 2);
        meta_u32(&mut buf, "llama.vocab_size", 16);
        for (k, v) in extra_meta {
            meta_u32(&mut buf, k, *v);
        }

        // Tensor index: sequential offsets, 32-byte aligned.
        let mut offset = 0u64;
        for (name, dims) in tensors {
            gstr(&mut buf, name);
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in *dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            buf.extend_from_slice(&0u32.to_le_bytes()); // F32
            buf.extend_from_slice(&offset.to_le_bytes());
            let numel: u64 = dims.iter().product();
            offset = (offset + numel * 4).div_ceil(32) * 32;
        }

        let padded = buf.len().div_ceil(32) * 32;
        buf.resize(padded, 0);
        let mut cursor = 0u64;
        for (_, dims) in tensors {
            let numel: u64 = dims.iter().product();
            for i in 0..numel {
                buf.extend_from_slice(&(i as f32).to_le_bytes());
            }
            let end = (cursor + numel * 4).div_ceil(32) * 32;
            buf.extend(std::iter::repeat_n(0u8, (end - cursor - numel * 4) as usize));
            cursor = end;
        }
        buf
    }

    fn standard_tensors() -> Vec<(&'static str, &'static [u64])> {
        vec![
            ("token_embd.weight", &[8, 16][..]), // GGUF order [D, V]
            ("output_norm.weight", &[8][..]),
            ("blk.0.attn_q.weight", &[8, 8][..]),
            ("blk.0.ffn_up.weight", &[8, 4][..]),
            ("blk.1.attn_q.weight", &[8, 8][..]),
            ("blk.1.ffn_up.weight", &[8, 4][..]),
        ]
    }

    fn write_gguf(dir: &Path, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.join("model.gguf");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(bytes).unwrap();
        path
    }

    #[test]
    fn convert_synthetic_gguf_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&standard_tensors(), &[]));
        let out = tmp.path().join("out");

        let report = convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();
        assert_eq!(report.model_id, "synth-model");
        assert_eq!(report.num_layers, 2);
        assert_eq!(report.shared_tensors, 2);

        // The package opens, validates, and cross-checks clean.
        let pkg = GllmPackage::open(&out).unwrap();
        assert_eq!(pkg.layer_count(), 2);
        assert!(pkg.has_checksum_file());
        assert!(pkg.shared.is_verified());
        assert!(pkg.verify_integrity().is_empty());

        // Shape is row-major: GGUF [D=8, V=16] -> [16, 8].
        let te = pkg.manifest().shared.tensor("token_embeddings").unwrap();
        assert_eq!(te.shape, vec![16, 8]);

        // Layer tensor names have the blk.N. prefix stripped.
        let l0 = pkg.layer_manifest(0).unwrap();
        assert!(l0.tensor("attn_q.weight").is_some());
        assert!(l0.tensor("ffn_up.weight").is_some());

        // head_count_kv falls back to num_heads when absent.
        assert_eq!(pkg.manifest().metadata.head_count_kv, 2);
    }

    #[test]
    fn glconv_extracts_eos_from_gguf_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(
            tmp.path(),
            &synth_gguf(&standard_tensors(), &[("tokenizer.ggml.eos_token_id", 7)]),
        );
        let out = tmp.path().join("out");

        let report = convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();
        assert_eq!(report.eos_token_ids, vec![7]);
        assert!(
            !report.warnings.iter().any(|w| w.contains("no tokenizer.ggml.eos_token_id")),
            "{:?}",
            report.warnings
        );

        // Round-trips through the written manifest, not just the in-memory report.
        let pkg = GllmPackage::open(&out).unwrap();
        assert_eq!(pkg.manifest().metadata.eos_token_ids, vec![7]);
    }

    #[test]
    fn glconv_warns_when_source_has_no_eos_metadata() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&standard_tensors(), &[]));
        let out = tmp.path().join("out");

        let report = convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();
        assert!(report.eos_token_ids.is_empty());
        assert!(
            report.warnings.iter().any(|w| w.contains("no tokenizer.ggml.eos_token_id")),
            "{:?}",
            report.warnings
        );
    }

    #[test]
    fn convert_preserves_tensor_bytes_exactly() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&standard_tensors(), &[]));
        let out = tmp.path().join("out");
        convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();

        // blk.0.attn_q.weight data = f32 values 0..64 by construction.
        let layer = crate::types::layer::LayerFile::read(&out.join("GLLMTensorLayer-0000.gllm")).unwrap();
        let (off, size) = layer.absolute_range("attn_q.weight").unwrap();
        let bytes = std::fs::read(out.join("GLLMTensorLayer-0000.gllm")).unwrap();
        let data = &bytes[off as usize..(off + size) as usize];
        let vals: Vec<f32> = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        assert_eq!(vals.len(), 64);
        assert_eq!(vals[0], 0.0);
        assert_eq!(vals[63], 63.0);
    }

    #[test]
    fn convert_gqa_metadata_mapped() {
        let tmp = tempfile::tempdir().unwrap();
        let bytes = synth_gguf(&standard_tensors(), &[("llama.attention.head_count_kv", 1)]);
        let gguf_path = write_gguf(tmp.path(), &bytes);
        let out = tmp.path().join("out");
        convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert_eq!(pkg.manifest().metadata.head_count_kv, 1);
        assert!(pkg.manifest().metadata.is_gqa());
    }

    #[test]
    fn convert_unmapped_tensor_goes_to_shared_with_warning() {
        let tmp = tempfile::tempdir().unwrap();
        let mut tensors = standard_tensors();
        tensors.push(("rope_freqs.weight", &[4][..]));
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&tensors, &[]));
        let out = tmp.path().join("out");

        let report = convert(&gguf_path, &out, &ConvertOptions::default()).unwrap();
        assert!(report.warnings.iter().any(|w| w.contains("rope_freqs.weight")));
        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.manifest().shared.tensor("rope_freqs.weight").is_some());
    }

    #[test]
    fn convert_missing_layer_tensors_fails_loud() {
        let tmp = tempfile::tempdir().unwrap();
        // block_count = 2 but only layer 0 tensors exist.
        let tensors: Vec<(&str, &[u64])> = vec![
            ("token_embd.weight", &[8, 16][..]),
            ("output_norm.weight", &[8][..]),
            ("blk.0.attn_q.weight", &[8, 8][..]),
        ];
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&tensors, &[]));
        let err = convert(&gguf_path, &tmp.path().join("out"), &ConvertOptions::default())
            .unwrap_err();
        assert!(err.to_string().contains("no tensors found for layer 1"), "{err}");
    }

    #[test]
    fn map_tensor_name_rules() {
        assert_eq!(
            map_tensor_name("token_embd.weight").unwrap(),
            Dest::Shared("token_embeddings".into())
        );
        assert_eq!(
            map_tensor_name("blk.12.attn_v.weight").unwrap(),
            Dest::Layer(12, "attn_v.weight".into())
        );
        assert_eq!(
            map_tensor_name("mystery.weight").unwrap(),
            Dest::SharedUnmapped("mystery.weight".into())
        );
        assert!(map_tensor_name("blk.x.attn_q.weight").is_err());
    }

    /// Real-model conversion: opt-in via GWENLAND_TEST_GGUF (real GGUFs are
    /// gitignored). Skips loudly when absent, per testing standards.
    #[test]
    fn convert_real_gguf_if_available() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out");
        let report = convert(Path::new(&path), &out, &ConvertOptions::default()).unwrap();
        eprintln!(
            "converted {} ({} layers, {} shared tensors, {} warnings)",
            report.model_id,
            report.num_layers,
            report.shared_tensors,
            report.warnings.len()
        );
        let pkg = GllmPackage::open(&out).unwrap();
        assert_eq!(pkg.layer_count() as u32, report.num_layers);
        assert!(pkg.verify_integrity().is_empty());
    }

    /// GQ4A-eligible fixture: `attn_q` at 256 elements (one GQ4A superblock
    /// exactly) so the CPP assignment's divisible-by-256 path is actually
    /// exercised end-to-end, not just skipped with a warning.
    fn gq4a_tensors() -> Vec<(&'static str, &'static [u64])> {
        vec![
            ("token_embd.weight", &[8, 16][..]),
            ("output_norm.weight", &[8][..]),
            ("blk.0.attn_norm.weight", &[8][..]),
            ("blk.0.attn_q.weight", &[16, 16][..]), // 256 elements
            ("blk.0.ffn_up.weight", &[8, 4][..]),   // 32 elements: NOT eligible
        ]
    }

    #[test]
    fn convert_gq4a_cpp_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&gq4a_tensors(), &[("llama.block_count", 1)]));
        let out = tmp.path().join("out");

        let opts = ConvertOptions { quant: QuantTarget::Gq4a, policy: QuantPolicy::Cpp, ..Default::default() };
        let report = convert(&gguf_path, &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        // attn_q (256 elements, sensitivity HIGH -> GQ4A escape) is assigned GQ4A.
        let l0 = pkg.layer_manifest(0).unwrap();
        let attn_q = l0.tensor("attn_q.weight").unwrap();
        assert_eq!(attn_q.dtype, DType::GQ4A);
        assert_eq!(attn_q.size, GQ4ABlock::BYTES as u64, "one 256-elem tensor = one superblock");

        // ffn_up (32 elements, not divisible by 256) keeps its original
        // dtype despite being CPP-eligible by sensitivity bucket, with a
        // warning explaining why.
        let ffn_up = l0.tensor("ffn_up.weight").unwrap();
        assert_eq!(ffn_up.dtype, DType::F32);
        assert!(report.warnings.iter().any(|w| w.contains("ffn_up") && w.contains("not a multiple of 256")));

        // attn_norm (HIGH bucket, "F16 always" in the GQ4A_CPP column) is
        // F16, not GQ4A, even though quant=Gq4a is active.
        let attn_norm = l0.tensor("attn_norm.weight").unwrap();
        assert_eq!(attn_norm.dtype, DType::F16);
        // Regression: the manifest dtype must match the actual bytes on
        // disk. This tensor's source GGUF dtype is F32 (synth_gguf always
        // writes F32) — the CPP relabel to F16 must carry a real conversion,
        // not just overwrite the dtype tag on untouched F32 bytes (which
        // silently doubled every reader's expected byte count).
        assert_eq!(attn_norm.size, 8 * 2, "8 elements at 2 bytes/elem for real F16, not F32's 4");

        // output_norm (Extreme, "F32 always") stays F32.
        assert_eq!(pkg.manifest().shared.tensor("output_norm.weight").unwrap().dtype, DType::F32);
    }

    /// GQ2A-eligible fixture set: mixes a HIGH-sensitivity tensor (escapes
    /// to GQ4A under assign_gq2a_cpp) and a MEDIUM-HIGH one (assigned GQ2A)
    /// alongside the norm rows, so the test exercises real heterogeneity —
    /// unlike GQ4A_CPP, which is degenerate (Pridwen v5 §5's note).
    fn gq2a_tensors() -> Vec<(&'static str, &'static [u64])> {
        vec![
            ("token_embd.weight", &[8, 16][..]),
            ("output_norm.weight", &[8][..]),
            ("blk.0.attn_norm.weight", &[8][..]),
            ("blk.0.attn_q.weight", &[16, 16][..]),      // 256 elements, HIGH -> GQ4A escape
            ("blk.0.attn_v.weight", &[16, 16][..]),      // 256 elements, MEDIUM-HIGH -> GQ2A
            ("blk.0.ffn_down.weight", &[8, 4][..]),      // 32 elements: NOT eligible (MEDIUM-LOW)
        ]
    }

    #[test]
    fn convert_gq2a_cpp_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&gq2a_tensors(), &[("llama.block_count", 1)]));
        let out = tmp.path().join("out");

        let opts = ConvertOptions { quant: QuantTarget::Gq2a, policy: QuantPolicy::Cpp, ..Default::default() };
        let report = convert(&gguf_path, &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        let l0 = pkg.layer_manifest(0).unwrap();

        // attn_q (HIGH sensitivity) escapes to GQ4A even under --quant GQ2A —
        // this is the heterogeneous assignment GQ4A_CPP never exercises.
        let attn_q = l0.tensor("attn_q.weight").unwrap();
        assert_eq!(attn_q.dtype, DType::GQ4A);
        assert_eq!(attn_q.size, GQ4ABlock::BYTES as u64);

        // attn_v (MEDIUM-HIGH sensitivity) is assigned GQ2A.
        let attn_v = l0.tensor("attn_v.weight").unwrap();
        assert_eq!(attn_v.dtype, DType::GQ2A);
        assert_eq!(attn_v.size, GQ2ABlock::BYTES as u64, "one 256-elem tensor = one superblock");

        // ffn_down (32 elements, not divisible by 256) keeps its original
        // dtype despite being CPP-eligible by sensitivity bucket.
        let ffn_down = l0.tensor("ffn_down.weight").unwrap();
        assert_eq!(ffn_down.dtype, DType::F32);
        assert!(report.warnings.iter().any(|w| w.contains("ffn_down") && w.contains("not a multiple of 256")));

        // Norm rows are identical to GQ4A_CPP's "always" assignment.
        let attn_norm = l0.tensor("attn_norm.weight").unwrap();
        assert_eq!(attn_norm.dtype, DType::F16);
        assert_eq!(pkg.manifest().shared.tensor("output_norm.weight").unwrap().dtype, DType::F32);
    }

    /// `--quant F32` diagnostic passthrough: every tensor gets F32,
    /// unconditionally — no sensitivity table consulted, no GQ4A/GQ2A
    /// tensor anywhere in the manifest, regardless of the tensor's role.
    #[test]
    fn convert_quant_f32_assigns_f32_to_every_tensor() {
        let tmp = tempfile::tempdir().unwrap();
        // Reuse the GQ2A fixture's tensor set (deliberately mixes every
        // sensitivity bucket) so this proves F32 mode ignores the sensitivity
        // table entirely, not just that the two "always" rows happen to
        // already be F32/F16 by coincidence.
        let gguf_path = write_gguf(tmp.path(), &synth_gguf(&gq2a_tensors(), &[("llama.block_count", 1)]));
        let out = tmp.path().join("out");

        let opts = ConvertOptions { quant: QuantTarget::F32, ..Default::default() };
        let report = convert(&gguf_path, &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        let l0 = pkg.layer_manifest(0).unwrap();
        for name in ["attn_norm.weight", "attn_q.weight", "attn_v.weight", "ffn_down.weight"] {
            assert_eq!(l0.tensor(name).unwrap().dtype, DType::F32, "{name} must be F32 under --quant F32");
        }
        assert_eq!(pkg.manifest().shared.tensor("output_norm.weight").unwrap().dtype, DType::F32);
        assert_eq!(pkg.manifest().shared.tensor("token_embeddings").unwrap().dtype, DType::F32);

        // No superblock-eligibility warnings: F32 has no block size, so
        // nothing should ever fall back with a "not a multiple of 256" note.
        assert!(
            report.warnings.iter().all(|w| !w.contains("not a multiple of")),
            "F32 mode must never hit the superblock-eligibility path: {:?}",
            report.warnings
        );
        assert!(pkg.manifest().quantization.is_none(), "an all-F32 package has no quantization scheme");
    }

    /// F32 diagnostic vs a real Q4_K_M source: opt-in via GWENLAND_TEST_GGUF,
    /// same bar as the GQ4A/GQ2A baselines. This is the one test that
    /// actually exercises the new dequantize-on-mismatch branch in
    /// `write_group` (real Q4_K/Q5_0/Q6_K source tensors, not the synthetic
    /// fixture's all-F32 data) — proof the diagnostic control group is
    /// itself uncorrupted before anyone draws a conclusion from its E2E
    /// output (see notes/project_gllm_e2e_garbage_output.md).
    #[test]
    fn quant_f32_diagnostic_dequantizes_every_real_tensor() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out_f32");
        let opts = ConvertOptions { quant: QuantTarget::F32, ..Default::default() };
        let report = convert(Path::new(&path), &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        let mut non_f32 = Vec::new();
        for t in &pkg.manifest().shared.tensors {
            if t.dtype != DType::F32 {
                non_f32.push(t.name.clone());
            }
        }
        for layer in &pkg.manifest().layers {
            for t in &layer.tensors {
                if t.dtype != DType::F32 {
                    non_f32.push(format!("layer {}: {}", layer.index, t.name));
                }
            }
        }
        assert!(non_f32.is_empty(), "every tensor must be F32 under --quant F32, found: {non_f32:?}");

        let package_bytes: u64 = walk_dir_size(&out);
        let source_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "F32 diagnostic: {} ({} layers, {} shared tensors, {} warnings)\n\
             source GGUF: {source_bytes} bytes, GLLM+F32 package: {package_bytes} bytes \
             (expected: uncompressed, several times larger than the Q4_K_M source)",
            report.model_id, report.num_layers, report.shared_tensors, report.warnings.len()
        );
    }

    /// Regression test for the Q6_K silent-corruption bug (found via
    /// `diff_dump.rs` while investigating `.gllm` E2E garbage output, see
    /// notes/issues/gllm-e2e-garbage-output.md): `glcore::GgufFile::dequantize`
    /// DOES accept Q6_K (unlike Q4_K/Q5_0, which it rejects outright) but gets
    /// the nibble layout wrong — `dequantize_for_gquant` must route Q6_K
    /// through `glproc::kernels::dequant::q6_k::scalar::run` (the
    /// GGML-faithful implementation), not fall through to `glcore`. Every
    /// real Q4_K_M GGUF has Q6_K-sourced tensors (`ffn_down.weight` in this
    /// model's case, every layer), so this is opt-in via GWENLAND_TEST_GGUF
    /// rather than a synthetic block — the bug only manifested on real
    /// tensor bytes, not the hand-built single-block fixtures `glproc`'s own
    /// kernel tests use.
    #[test]
    fn dequantize_for_gquant_routes_q6_k_through_glproc_not_glcore() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let gguf = GgufFile::open(&path).unwrap();
        let info = gguf
            .tensors
            .iter()
            .find(|t| t.dtype == GgufDType::Q6_K)
            .expect("a real Q4_K_M model must have at least one Q6_K tensor to test against");

        let routed = dequantize_for_gquant(&gguf, info).unwrap();
        let raw = gguf.tensor_data(info).unwrap();
        let ground_truth = glproc::kernels::dequant::q6_k::scalar::run(raw).unwrap();
        assert_eq!(
            routed, ground_truth,
            "tensor {:?}: dequantize_for_gquant must match glproc's GGML-faithful Q6_K dequant",
            info.name
        );
    }

    /// GQ4A vs Q4_K_M baseline: opt-in via GWENLAND_TEST_GGUF (Pridwen v5
    /// §14 Phase 1's `gq4a_ppl_vs_q4km_baseline`, as scoped in
    /// notes/pridwen-p1-notes.md — glbench cannot load `.gllm` packages or
    /// decode any quantized GLLM dtype today (pre-existing gap, not part of
    /// this phase), so this test's bar is the spec's own fallback: "passes
    /// if conversion completes without error; PPL delta is recorded to
    /// notes, not gated." Records package size only (a reachable proxy for
    /// the "Size" row in Pridwen v5 §13's Expected Results table).
    #[test]
    fn gq4a_ppl_vs_q4km_baseline() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out_gq4a");
        let opts = ConvertOptions { quant: QuantTarget::Gq4a, policy: QuantPolicy::Cpp, ..Default::default() };
        let report = convert(Path::new(&path), &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        let package_bytes: u64 = walk_dir_size(&out);
        let source_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "GQ4A_CPP baseline: {} ({} layers, {} shared tensors, {} warnings)\n\
             source GGUF: {source_bytes} bytes, GLLM+GQ4A package: {package_bytes} bytes\n\
             NOTE: PPL/decode comparison vs Q4_K_M requires glbench .gllm support, \
             which does not exist yet (see notes/pridwen-p1-notes.md) — size is the \
             only number this test can measure.",
            report.model_id, report.num_layers, report.shared_tensors, report.warnings.len()
        );
    }

    /// GQ2A vs Q4_K_M baseline: opt-in via GWENLAND_TEST_GGUF, same bar as
    /// `gq4a_ppl_vs_q4km_baseline` (glbench .gllm support doesn't exist yet).
    /// Additionally tallies dtype counts across the manifest — this is the
    /// automated version of the manual GQ4A-coverage measurement recorded in
    /// notes/pridwen-p1-notes.md's Phase 2 FINDING entries (25/291 before
    /// the glproc dequant fix, 170/291 after) — for GQ2A_CPP, both GQ4A and
    /// GQ2A counts are expected to be nonzero given the sensitivity table's
    /// heterogeneous assignment.
    #[test]
    fn gq2a_ppl_vs_q4km_baseline() {
        let Ok(path) = std::env::var("GWENLAND_TEST_GGUF") else {
            eprintln!("SKIP: GWENLAND_TEST_GGUF not set (no real GGUF available)");
            return;
        };
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out_gq2a");
        let opts = ConvertOptions { quant: QuantTarget::Gq2a, policy: QuantPolicy::Cpp, ..Default::default() };
        let report = convert(Path::new(&path), &out, &opts).unwrap();

        let pkg = GllmPackage::open(&out).unwrap();
        assert!(pkg.verify_integrity().is_empty());

        let mut gq4a_count = 0usize;
        let mut gq2a_count = 0usize;
        let mut other_count = 0usize;
        let count_dtype = |dtype: DType, gq4a: &mut usize, gq2a: &mut usize, other: &mut usize| match dtype {
            DType::GQ4A => *gq4a += 1,
            DType::GQ2A => *gq2a += 1,
            _ => *other += 1,
        };
        for t in &pkg.manifest().shared.tensors {
            count_dtype(t.dtype, &mut gq4a_count, &mut gq2a_count, &mut other_count);
        }
        for layer in &pkg.manifest().layers {
            for t in &layer.tensors {
                count_dtype(t.dtype, &mut gq4a_count, &mut gq2a_count, &mut other_count);
            }
        }

        let package_bytes: u64 = walk_dir_size(&out);
        let source_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
        eprintln!(
            "GQ2A_CPP baseline: {} ({} layers, {} shared tensors, {} warnings)\n\
             dtype tally: GQ4A={gq4a_count} GQ2A={gq2a_count} other={other_count}\n\
             source GGUF: {source_bytes} bytes, GLLM+GQ2A package: {package_bytes} bytes\n\
             NOTE: PPL/decode comparison vs Q4_K_M requires glbench .gllm support, \
             which does not exist yet (see notes/pridwen-p1-notes.md) — size and dtype \
             tally are the only numbers this test can measure.",
            report.model_id, report.num_layers, report.shared_tensors, report.warnings.len()
        );

        // The whole point of GQ2A_CPP: real heterogeneity across a real
        // model, not just the synthetic fixture's handful of tensors.
        assert!(gq4a_count > 0, "expected at least one GQ4A-escaped tensor");
        assert!(gq2a_count > 0, "expected at least one GQ2A tensor");
    }

    fn walk_dir_size(dir: &Path) -> u64 {
        let mut total = 0u64;
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    total += if meta.is_dir() { walk_dir_size(&entry.path()) } else { meta.len() };
                }
            }
        }
        total
    }
}
