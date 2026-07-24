//! `glproc` execution backend (ARTX10 Wave 1).
//!
//! Adapts [`ExecutionBackend`] to glproc's per-op CPU kernels — RMSNorm,
//! matvec, RoPE-adjacent attention, SwiGLU — so a `gllm:transformer/standard@v1`
//! layer actually computes instead of the [`NullBackend`](super::backend::NullBackend)
//! passthrough.
//!
//! ## Scope (Wave 1)
//!
//! Dense `F32` tensors only, one query token at a time (decode shape, not
//! batched prefill). Tensor bytes are decoded through
//! [`glcore::format::gllm::decode_tensor`] — the same byte-in/`Tensor`-out
//! boundary `gguf.rs` uses for GGUF — so no tensor math lives in this crate;
//! every FLOP happens inside glproc's `kernels` module. Quantized layers
//! (`Q4_K`, `Q8_0`, ...) are out of Wave 1's scope and return
//! [`GllmError::ExecutionFailed`] rather than silently mis-computing —
//! follow-up work, not claimed here.
//!
//! Threading is deliberately absent: glproc's `par_*` helpers exist to
//! amortize a persistent `ThreadPool` across a whole generation run, which
//! this per-layer, per-call boundary does not have. The scalar/SIMD
//! (non-`par_*`) kernels this module calls are already vectorized per
//! `glproc::simd_strategy::SimdStrategy::detect()` — only the *thread*
//! fan-out is missing, not the SIMD.

use crate::error::{GllmError, GllmResult};
use crate::manifest::{ModelMetadata, LayerManifest, TensorEntry};
use crate::runtime::kv_cache::KvCacheSlot;
use crate::runtime::mmap::LayerMapping;
use crate::runtime::types::ActivationBuffer;
use crate::runtime::backend::{ExecutionBackend, KV_ELEMENT_SIZE_F32};

/// Tensor names every `gllm:transformer/standard@v1` layer carries (mirrors
/// `TransformerStandardPlugin::required_tensors` in `plugin.rs` — duplicated
/// rather than shared because the plugin registry validates *presence*, at
/// manifest-parse time, while this list drives *access*, at execute time, and
/// the two must not become the same call site with different failure modes).
const ATTN_NORM: &str = "attn_norm.weight";
const ATTN_Q: &str = "attn_q.weight";
const ATTN_K: &str = "attn_k.weight";
const ATTN_V: &str = "attn_v.weight";
const ATTN_OUTPUT: &str = "attn_output.weight";
const FFN_NORM: &str = "ffn_norm.weight";
const FFN_GATE: &str = "ffn_gate.weight";
const FFN_UP: &str = "ffn_up.weight";
const FFN_DOWN: &str = "ffn_down.weight";

/// Attention Q/K/V bias — present in Qwen2 (and thus real Qwen2.5 `.gllm`
/// packages, confirmed against a real converted manifest), absent in most
/// other architectures (including Qwen3, per `plugin.rs`'s own doc comment).
/// Naming matches every other tensor constant above: `.weight`/`.bias`
/// suffix on the bare role name, exactly what the ARTX07 converter writes —
/// verified directly against a real package manifest rather than assumed
/// (`grep '"name": "attn_[qkv].bias"' gllm.json`), since a wrong guess here
/// (e.g. an underscore instead of a dot) would silently never match and
/// leave the bias unapplied again, just for a different reason.
const ATTN_Q_BIAS: &str = "attn_q.bias";
const ATTN_K_BIAS: &str = "attn_k.bias";
const ATTN_V_BIAS: &str = "attn_v.bias";

/// Fixed hyperparameters a [`GlprocBackend`] needs that
/// [`ExecutionBackend::execute_layer`] is not handed per call — attention
/// shape and RoPE configuration are model-wide, not layer-wide, so they are
/// captured once at construction from the manifest's [`ModelMetadata`]
/// instead of being re-derived (or, worse, guessed) on every layer.
#[derive(Debug, Clone, Copy)]
struct AttnShape {
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    rope_freq_base: f32,
    /// RMSNorm epsilon (`ModelMetadata::effective_rms_eps`, sourced from the
    /// GGUF's `{arch}.attention.layer_norm_rms_epsilon` at conversion time).
    /// Previously hardcoded `1e-5` here regardless of the source model —
    /// real models disagree (Qwen2.5 ships `1e-6`), and a wrong epsilon
    /// silently distorts every RMSNorm call's output magnitude rather than
    /// erroring, so it must come from the model, not an assumed constant.
    rms_eps: f32,
}

/// CPU execution backend that forwards to glproc's per-op kernels.
///
/// One instance per loaded model — [`AttnShape`] is fixed for the model's
/// lifetime, so `execute_layer` never has to reconsult the manifest.
#[derive(Debug)]
pub struct GlprocBackend {
    shape: AttnShape,
}

impl GlprocBackend {
    /// Build a backend from the model's hyperparameters.
    ///
    /// Fails with [`GllmError::MissingMetadata`] when `head_dim` is not
    /// derivable (`embedding_length` not divisible by `num_heads`) — the same
    /// check [`crate::runtime::kv_cache::KvCacheConfig::from_metadata`] makes,
    /// since both need the same quantity for the same reason.
    pub fn new(metadata: &ModelMetadata) -> GllmResult<Self> {
        let head_dim = metadata.head_dim().ok_or_else(|| {
            GllmError::MissingMetadata(format!(
                "head_dim: embedding_length {} is not divisible by num_heads {}",
                metadata.embedding_length, metadata.num_heads
            ))
        })?;
        Ok(GlprocBackend {
            shape: AttnShape {
                n_heads: metadata.num_heads as usize,
                n_kv_heads: metadata.head_count_kv as usize,
                head_dim: head_dim as usize,
                rope_freq_base: metadata.effective_rope_freq_base() as f32,
                rms_eps: metadata.effective_rms_eps() as f32,
            },
        })
    }
}

/// Look up a required tensor's bytes and decode them to `f32`, in one step —
/// every call site needs both the manifest entry (for shape/dtype) and the
/// mapped bytes (for data), and forgetting either half is the same bug either
/// way, so they are fetched together.
fn required_tensor(
    layer_index: u32,
    mapping: &LayerMapping,
    manifest: &LayerManifest,
    name: &str,
) -> GllmResult<Vec<f32>> {
    let entry: &TensorEntry = manifest.tensor(name).ok_or_else(|| GllmError::ExecutionFailed {
        layer: layer_index,
        reason: format!("layer {layer_index} manifest is missing tensor {name:?}"),
    })?;
    let bytes = mapping.tensor_bytes(name).ok_or_else(|| GllmError::ExecutionFailed {
        layer: layer_index,
        reason: format!("layer {layer_index} file has no data for tensor {name:?}"),
    })?;
    // `glcore::format::decode_tensor` only understands F32/F16/BF16 (glcore
    // has no glproc dependency — GQ4A/GQ2A dequant kernels live in glproc,
    // and glproc already depends on glcore, so the reverse dependency would
    // be circular). CPP assigns most per-layer weight tensors GQ4A/GQ2A, so
    // this — the one place in the crate that already legitimately depends
    // on glproc (ADR: architecture/Pridwen-P2-ADR-glproc-dequant.md) — is
    // where that dequant has to happen, mirroring `GllmEngine::load_shared`.
    match entry.dtype {
        crate::manifest::DType::GQ4A => return Ok(glproc::kernels::gquant::dequant_gq4a_stream(bytes)),
        crate::manifest::DType::GQ2A => return Ok(glproc::kernels::gquant::dequant_gq2a_stream(bytes)),
        _ => {}
    }
    let dtype_str = serde_json::to_value(entry.dtype)
        .ok()
        .and_then(|v| v.as_str().map(str::to_string))
        .unwrap_or_default();
    glcore::format::decode_tensor(name, &entry.shape, &dtype_str, bytes)
        .map(|t| t.data)
        .map_err(|e| GllmError::ExecutionFailed {
            layer: layer_index,
            reason: format!("tensor {name:?}: {e}"),
        })
}

/// Like [`required_tensor`], but returns `None` when the tensor is absent
/// from the manifest instead of erroring — for tensors some architectures
/// carry and others do not (Qwen2's `attn_q/k/v.bias`, absent on Qwen3 and
/// most other families). A tensor that IS present but fails to decode still
/// propagates its error: "optional" means optional *presence*, never
/// "ignore a real decode failure."
fn optional_tensor(
    layer_index: u32,
    mapping: &LayerMapping,
    manifest: &LayerManifest,
    name: &str,
) -> GllmResult<Option<Vec<f32>>> {
    if manifest.tensor(name).is_none() {
        return Ok(None);
    }
    required_tensor(layer_index, mapping, manifest, name).map(Some)
}

/// Apply rotary position embeddings in place, NeoX style (Qwen2/Qwen3/phi/
/// gemma — the family every reference model here belongs to). Mirrors
/// `glproc::runner::rope`, which is private to that module; duplicated
/// rather than exposed because it is 8 lines of arithmetic, not a kernel —
/// the actual dot products and softmax it feeds still come from glproc.
fn rope_neox(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (sin, cos) = theta.sin_cos();
        let x0 = x[i];
        let x1 = x[i + half];
        x[i] = x0 * cos - x1 * sin;
        x[i + half] = x0 * sin + x1 * cos;
    }
}

impl ExecutionBackend for GlprocBackend {
    fn name(&self) -> &'static str {
        "glproc-cpu"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn kv_element_size(&self) -> usize {
        KV_ELEMENT_SIZE_F32
    }

    fn execute_layer(
        &self,
        layer: &LayerMapping,
        manifest: &LayerManifest,
        kv_cache: &mut KvCacheSlot,
        input: &ActivationBuffer,
        output: &mut ActivationBuffer,
    ) -> GllmResult<()> {
        let idx = manifest.index;
        let s = self.shape;
        let dim = s.n_heads * s.head_dim; // embedding_length, derived so a
        // mismatched input buffer is caught below rather than assumed.
        let kv_dim = s.n_kv_heads * s.head_dim;
        let heads_per_kv = s.n_heads / s.n_kv_heads.max(1);

        if input.data.len() != dim {
            return Err(GllmError::ExecutionFailed {
                layer: idx,
                reason: format!(
                    "input has {} elements, model dim is {dim} ({}x{})",
                    input.data.len(),
                    s.n_heads,
                    s.head_dim
                ),
            });
        }
        let pos = kv_cache.current_seq_len as usize;

        let attn_norm_w = required_tensor(idx, layer, manifest, ATTN_NORM)?;
        let wq = required_tensor(idx, layer, manifest, ATTN_Q)?;
        let wk = required_tensor(idx, layer, manifest, ATTN_K)?;
        let wv = required_tensor(idx, layer, manifest, ATTN_V)?;
        let bq = optional_tensor(idx, layer, manifest, ATTN_Q_BIAS)?;
        let bk = optional_tensor(idx, layer, manifest, ATTN_K_BIAS)?;
        let bv = optional_tensor(idx, layer, manifest, ATTN_V_BIAS)?;
        let wo = required_tensor(idx, layer, manifest, ATTN_OUTPUT)?;
        let ffn_norm_w = required_tensor(idx, layer, manifest, FFN_NORM)?;
        let w_gate = required_tensor(idx, layer, manifest, FFN_GATE)?;
        let w_up = required_tensor(idx, layer, manifest, FFN_UP)?;
        let w_down = required_tensor(idx, layer, manifest, FFN_DOWN)?;
        let hidden_dim = w_gate.len() / dim.max(1);

        // --- attention block ---
        let mut xn = vec![0.0f32; dim];
        glproc::kernels::rms_norm_into(&input.data, &attn_norm_w, s.rms_eps, &mut xn);

        let mut q = vec![0.0f32; dim];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        glproc::kernels::matvec(&wq, &xn, &mut q, dim, dim);
        glproc::kernels::matvec(&wk, &xn, &mut k, kv_dim, dim);
        glproc::kernels::matvec(&wv, &xn, &mut v, kv_dim, dim);

        // Attention bias, added right after the projection and BEFORE RoPE —
        // matches glproc::runner::Runner::step's order exactly (runner.rs
        // ~847-861). This was the confirmed root cause of the E2E garbage
        // output bug (notes/pridwen-p1-notes.md): Qwen2/2.5 carries these
        // three tensors in every layer, they were present in the manifest
        // and passed validation, but this function never fetched or applied
        // them — every Q/K/V vector in every layer was silently missing an
        // additive term the model was trained with. `None` (tensor absent,
        // e.g. Qwen3) means skip, not zero-and-continue-with-a-warning: a
        // model that never had bias should behave exactly as before this fix.
        //
        // TODO: q_norm/k_norm (Qwen3's per-head RMSNorm on Q/K, used INSTEAD
        // of attention bias) is still not implemented here. Not needed for
        // Qwen2/2.5, but the same class of silent omission would corrupt a
        // Qwen3 package (e.g. the gllm-qwen3-17b package already on disk)
        // the same way this bug corrupted Qwen2.5. Tracked: GWEN-XXX.
        if let Some(b) = &bq {
            for (qi, bi) in q.iter_mut().zip(b) {
                *qi += bi;
            }
        }
        if let Some(b) = &bk {
            for (ki, bi) in k.iter_mut().zip(b) {
                *ki += bi;
            }
        }
        if let Some(b) = &bv {
            for (vi, bi) in v.iter_mut().zip(b) {
                *vi += bi;
            }
        }

        for h in 0..s.n_heads {
            rope_neox(&mut q[h * s.head_dim..(h + 1) * s.head_dim], pos, s.head_dim, s.rope_freq_base);
        }
        for h in 0..s.n_kv_heads {
            rope_neox(&mut k[h * s.head_dim..(h + 1) * s.head_dim], pos, s.head_dim, s.rope_freq_base);
        }

        // Append this token's K/V into the external slot (backend-owned
        // format, f32, `[num_kv_heads][max_seq_len][head_dim]` — see
        // `KvCacheSlot` docs). Writing before reading is what makes the
        // current position's own K/V visible to its own attention pass,
        // matching glproc::runner's write-then-read order.
        let max_seq_len = kv_cache.max_seq_len() as usize;
        for h in 0..s.n_kv_heads {
            let cell = (h * max_seq_len + pos) * s.head_dim;
            let byte_off = cell * 4;
            let k_head = &k[h * s.head_dim..(h + 1) * s.head_dim];
            let v_head = &v[h * s.head_dim..(h + 1) * s.head_dim];
            write_f32_row(&mut kv_cache.key, byte_off, k_head);
            write_f32_row(&mut kv_cache.value, byte_off, v_head);
        }
        kv_cache.advance(1)?;
        let cached_len = kv_cache.current_seq_len as usize;

        let mut attn_out = vec![0.0f32; dim];
        let mut scores = vec![0.0f32; cached_len];
        for h in 0..s.n_heads {
            let kv_head = h / heads_per_kv.max(1);
            let k_cache = read_f32_rows(&kv_cache.key, kv_head, max_seq_len, s.head_dim, cached_len);
            let v_cache = read_f32_rows(&kv_cache.value, kv_head, max_seq_len, s.head_dim, cached_len);
            glproc::attention::attention_one_into(
                &q[h * s.head_dim..(h + 1) * s.head_dim],
                &k_cache,
                &v_cache,
                s.head_dim,
                &mut scores,
                &mut attn_out[h * s.head_dim..(h + 1) * s.head_dim],
            );
        }

        let mut proj = vec![0.0f32; dim];
        glproc::kernels::matvec(&wo, &attn_out, &mut proj, dim, dim);
        let mut x: Vec<f32> = input.data.clone();
        for (xi, pi) in x.iter_mut().zip(&proj) {
            *xi += pi;
        }

        // --- feed-forward block (dense SwiGLU) ---
        glproc::kernels::rms_norm_into(&x, &ffn_norm_w, s.rms_eps, &mut xn);
        let mut gate = vec![0.0f32; hidden_dim];
        let mut up = vec![0.0f32; hidden_dim];
        glproc::kernels::matvec(&w_gate, &xn, &mut gate, hidden_dim, dim);
        glproc::kernels::matvec(&w_up, &xn, &mut up, hidden_dim, dim);
        glproc::kernels::silu_mul(&mut gate, &up);

        let mut down = vec![0.0f32; dim];
        glproc::kernels::matvec(&w_down, &gate, &mut down, dim, hidden_dim);
        for (xi, di) in x.iter_mut().zip(&down) {
            *xi += di;
        }

        output.data.clear();
        output.data.extend_from_slice(&x);
        output.shape.clone_from(&input.shape);
        Ok(())
    }
}

/// Write one head's `[head_dim]` f32 row into a KV slot's raw byte buffer at
/// a `[head][position][head_dim]`-layout cell.
fn write_f32_row(buf: &mut [u8], byte_offset: usize, row: &[f32]) {
    for (i, &v) in row.iter().enumerate() {
        let off = byte_offset + i * 4;
        buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
}

/// Read `cached_len` positions of one KV head back out as a flat
/// `[cached_len * head_dim]` f32 vec — the shape
/// [`glproc::attention::attention_one_into`] expects.
fn read_f32_rows(buf: &[u8], head: usize, max_seq_len: usize, head_dim: usize, cached_len: usize) -> Vec<f32> {
    let row_start = head * max_seq_len * head_dim * 4;
    let mut out = Vec::with_capacity(cached_len * head_dim);
    for pos in 0..cached_len {
        let off = row_start + pos * head_dim * 4;
        for d in 0..head_dim {
            let b = &buf[off + d * 4..off + d * 4 + 4];
            out.push(f32::from_le_bytes([b[0], b[1], b[2], b[3]]));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::ExtensionUri;
    use crate::runtime::kv_cache::KvCacheConfig;
    use tempfile::TempDir;

    /// A tiny dense transformer block: dim=8, 2 heads, 2 KV heads (MHA, no
    /// GQA — keeps the fixture small), head_dim=4, hidden_dim=16.
    fn fixture_metadata() -> ModelMetadata {
        ModelMetadata {
            vocab_size: 32,
            context_length: 64,
            embedding_length: 8,
            num_layers: 1,
            num_heads: 2,
            head_count_kv: 2,
            rope_dims: Some(4),
            rope_freq_base: Some(10_000.0),
            rope_scaling: None,
            expert_count: None,
            expert_used_count: None,
            sliding_window: None,
            attention_bias: None,
            rms_eps: None,
            eos_token_ids: Vec::new(),
        }
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// The 9 f32 tensors `TransformerStandardPlugin` requires, sized for
    /// dim=8, hidden_dim=16 (`fixture_metadata()`'s shape).
    ///
    /// Deterministic, non-trivial values (not all equal, not all zero) so a
    /// shape bug (e.g. a transposed matvec) is very likely to surface as a
    /// NaN or a wrong-length panic rather than silently matching by luck.
    /// `seed` perturbs every value so two layers in a stack are never
    /// bit-identical — a multi-layer test that reused one layer's weights
    /// could not tell "ran once" from "ran per layer".
    fn fixture_layer_tensors(seed: f32) -> Vec<(&'static str, Vec<u64>, Vec<f32>)> {
        const DIM: usize = 8;
        const HIDDEN: usize = 16;
        let vec_of = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * scale + seed).sin()).collect()
        };
        vec![
            (ATTN_NORM, vec![DIM as u64], vec![1.0; DIM]),
            (ATTN_Q, vec![DIM as u64, DIM as u64], vec_of(DIM * DIM, 0.1)),
            (ATTN_K, vec![DIM as u64, DIM as u64], vec_of(DIM * DIM, 0.2)),
            (ATTN_V, vec![DIM as u64, DIM as u64], vec_of(DIM * DIM, 0.3)),
            (ATTN_OUTPUT, vec![DIM as u64, DIM as u64], vec_of(DIM * DIM, 0.4)),
            (FFN_NORM, vec![DIM as u64], vec![1.0; DIM]),
            (FFN_GATE, vec![HIDDEN as u64, DIM as u64], vec_of(HIDDEN * DIM, 0.05)),
            (FFN_UP, vec![HIDDEN as u64, DIM as u64], vec_of(HIDDEN * DIM, 0.06)),
            (FFN_DOWN, vec![DIM as u64, HIDDEN as u64], vec_of(DIM * HIDDEN, 0.07)),
        ]
    }

    /// Writes a real layer file with [`fixture_layer_tensors`] and the
    /// matching [`LayerManifest`], via [`crate::layer_io::write_unit_file`]
    /// — the same writer the converter uses, so the fixture cannot drift
    /// from the real on-disk format.
    fn write_fixture_layer(dir: &std::path::Path, index: u32, seed: f32) -> (LayerMapping, LayerManifest) {
        write_layer_from_tensors(dir, index, fixture_layer_tensors(seed))
    }

    /// As [`fixture_layer_tensors`], plus `attn_q/k/v.bias` — the tensor set
    /// a real Qwen2-family layer carries (see `plugin.rs`'s
    /// `standard_transformer_accepts_a_real_qwen2_layer`), for exercising the
    /// bias-add path the plain fixture never touches.
    fn fixture_layer_tensors_with_bias(seed: f32) -> Vec<(&'static str, Vec<u64>, Vec<f32>)> {
        const DIM: usize = 8;
        let vec_of = |n: usize, scale: f32| -> Vec<f32> {
            (0..n).map(|i| (i as f32 * scale + seed).sin()).collect()
        };
        let mut tensors = fixture_layer_tensors(seed);
        tensors.push((ATTN_Q_BIAS, vec![DIM as u64], vec_of(DIM, 0.21)));
        tensors.push((ATTN_K_BIAS, vec![DIM as u64], vec_of(DIM, 0.22)));
        tensors.push((ATTN_V_BIAS, vec![DIM as u64], vec_of(DIM, 0.23)));
        tensors
    }

    fn write_fixture_layer_with_bias(dir: &std::path::Path, index: u32, seed: f32) -> (LayerMapping, LayerManifest) {
        write_layer_from_tensors(dir, index, fixture_layer_tensors_with_bias(seed))
    }

    /// Shared write path for both [`write_fixture_layer`] and
    /// [`write_fixture_layer_with_bias`] — writing tensor bytes and building
    /// the matching manifest is identical either way; only which tensors are
    /// in the list differs.
    fn write_layer_from_tensors(
        dir: &std::path::Path,
        index: u32,
        tensors: Vec<(&'static str, Vec<u64>, Vec<f32>)>,
    ) -> (LayerMapping, LayerManifest) {
        let byte_data: Vec<Vec<u8>> = tensors.iter().map(|(_, _, v)| f32_bytes(v)).collect();
        let specs: Vec<(&str, &[u64], crate::manifest::DType, &[u8])> = tensors
            .iter()
            .zip(&byte_data)
            .map(|((name, shape, _), data)| (*name, shape.as_slice(), crate::manifest::DType::F32, data.as_slice()))
            .collect();

        let path = dir.join(crate::manifest::format_layer_filename(index));
        let entries = crate::layer_io::write_unit_file(&path, &specs).unwrap();
        let mapping = LayerMapping::open(&path, Some(index), false).unwrap();

        let manifest = LayerManifest {
            index,
            file: crate::manifest::format_layer_filename(index),
            checksum: format!("sha256:{}", "0".repeat(64)),
            layer_type: ExtensionUri::parse("gllm:transformer/standard@v1").unwrap(),
            tensors: entries,
            device: None,
        };

        (mapping, manifest)
    }

    fn fixture_kv_slot(metadata: &ModelMetadata, max_seq_len: u32) -> KvCacheSlot {
        let config = KvCacheConfig {
            num_layers: 1,
            num_kv_heads: metadata.head_count_kv,
            head_dim: metadata.head_dim().unwrap(),
            max_seq_len,
            element_size: 4,
        };
        KvCacheSlot::new(0, &config).unwrap()
    }

    #[test]
    fn backend_reports_glproc_cpu_and_f32_kv() {
        let backend = GlprocBackend::new(&fixture_metadata()).unwrap();
        assert_eq!(backend.name(), "glproc-cpu");
        assert!(backend.is_available());
        assert_eq!(backend.kv_element_size(), 4);
    }

    #[test]
    fn new_rejects_indivisible_head_dim() {
        let mut meta = fixture_metadata();
        meta.num_heads = 3; // 8 / 3 is not exact
        let err = GlprocBackend::new(&meta).unwrap_err();
        assert!(matches!(err, GllmError::MissingMetadata(_)), "got {err:?}");
    }

    #[test]
    fn execute_layer_produces_a_finite_same_shape_output() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        let backend = GlprocBackend::new(&metadata).unwrap();
        let mut kv = fixture_kv_slot(&metadata, 16);

        let input = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.37).cos()).collect(),
            shape: vec![8],
        };
        let mut output = ActivationBuffer::zeros(vec![8]);

        backend
            .execute_layer(&mapping, &manifest, &mut kv, &input, &mut output)
            .unwrap();

        assert_eq!(output.shape, input.shape);
        assert_eq!(output.data.len(), 8);
        assert!(output.data.iter().all(|x| x.is_finite()), "{:?}", output.data);
        assert_ne!(output.data, input.data, "a real layer must transform the input");
        assert_eq!(kv.current_seq_len, 1, "one token advanced the cache");
    }

    #[test]
    fn execute_layer_is_deterministic() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        let backend = GlprocBackend::new(&metadata).unwrap();

        let input = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.37).cos()).collect(),
            shape: vec![8],
        };

        let mut kv_a = fixture_kv_slot(&metadata, 16);
        let mut out_a = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping, &manifest, &mut kv_a, &input, &mut out_a).unwrap();

        let mut kv_b = fixture_kv_slot(&metadata, 16);
        let mut out_b = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping, &manifest, &mut kv_b, &input, &mut out_b).unwrap();

        assert_eq!(out_a.data, out_b.data, "same input must give the same output");
    }

    #[test]
    fn execute_layer_writes_kv_cache_at_the_current_position() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        let backend = GlprocBackend::new(&metadata).unwrap();
        let mut kv = fixture_kv_slot(&metadata, 16);

        let input = ActivationBuffer::zeros(vec![8]);
        let mut output = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping, &manifest, &mut kv, &input, &mut output).unwrap();
        assert_eq!(kv.current_seq_len, 1);

        // Second token: a fresh position must not clobber the first, and the
        // slot must accept it without erroring.
        backend.execute_layer(&mapping, &manifest, &mut kv, &input, &mut output).unwrap();
        assert_eq!(kv.current_seq_len, 2);
    }

    /// Regression test for the confirmed E2E garbage-output root cause
    /// (notes/pridwen-p1-notes.md): a layer carrying real `attn_q/k/v.bias`
    /// tensors (a real Qwen2/2.5 layer's tensor set) must produce a
    /// different output than the identical weights with no bias — proving
    /// the bias is actually fetched and added, not merely present in the
    /// manifest and silently ignored the way it was before this fix.
    #[test]
    fn execute_layer_applies_attention_bias_when_present() {
        let dir_no_bias = TempDir::new().unwrap();
        let dir_with_bias = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let backend = GlprocBackend::new(&metadata).unwrap();

        let input = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.37).cos()).collect(),
            shape: vec![8],
        };

        let (mapping_a, manifest_a) = write_fixture_layer(dir_no_bias.path(), 0, 0.0);
        let mut kv_a = fixture_kv_slot(&metadata, 16);
        let mut out_a = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping_a, &manifest_a, &mut kv_a, &input, &mut out_a).unwrap();

        let (mapping_b, manifest_b) = write_fixture_layer_with_bias(dir_with_bias.path(), 0, 0.0);
        let mut kv_b = fixture_kv_slot(&metadata, 16);
        let mut out_b = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping_b, &manifest_b, &mut kv_b, &input, &mut out_b).unwrap();

        assert_ne!(
            out_a.data, out_b.data,
            "identical weights but with real attn_q/k/v.bias present must give a different output"
        );
        assert!(out_b.data.iter().all(|x| x.is_finite()), "{:?}", out_b.data);
    }

    /// The other half of the bias contract: an architecture that never had
    /// attention bias (Qwen3, or the plain synthetic fixture used by every
    /// other test in this file) must behave exactly as it did before this
    /// fix — no error, and numerically identical to explicitly adding a
    /// zero bias vector. This is the "optional means optional" half of
    /// `optional_tensor`'s contract: absence must not just "not crash," it
    /// must be indistinguishable from a real but all-zero bias.
    #[test]
    fn execute_layer_missing_bias_tensors_do_not_error_and_match_explicit_zero_bias() {
        let dir_absent = TempDir::new().unwrap();
        let dir_zero = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let backend = GlprocBackend::new(&metadata).unwrap();

        let input = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.37).cos()).collect(),
            shape: vec![8],
        };

        // No bias tensors in the manifest at all (the standard fixture).
        let (mapping_absent, manifest_absent) = write_fixture_layer(dir_absent.path(), 0, 0.0);
        let mut kv_absent = fixture_kv_slot(&metadata, 16);
        let mut out_absent = ActivationBuffer::zeros(vec![8]);
        backend
            .execute_layer(&mapping_absent, &manifest_absent, &mut kv_absent, &input, &mut out_absent)
            .expect("missing bias tensors must not be an error — most architectures have none");

        // Bias tensors present, but every value is zero.
        const DIM: u64 = 8;
        let mut tensors = fixture_layer_tensors(0.0);
        tensors.push((ATTN_Q_BIAS, vec![DIM], vec![0.0; DIM as usize]));
        tensors.push((ATTN_K_BIAS, vec![DIM], vec![0.0; DIM as usize]));
        tensors.push((ATTN_V_BIAS, vec![DIM], vec![0.0; DIM as usize]));
        let (mapping_zero, manifest_zero) = write_layer_from_tensors(dir_zero.path(), 0, tensors);
        let mut kv_zero = fixture_kv_slot(&metadata, 16);
        let mut out_zero = ActivationBuffer::zeros(vec![8]);
        backend
            .execute_layer(&mapping_zero, &manifest_zero, &mut kv_zero, &input, &mut out_zero)
            .unwrap();

        assert_eq!(
            out_absent.data, out_zero.data,
            "an absent bias tensor must behave identically to an explicit all-zero one"
        );
    }

    /// DIAGNOSTIC (not yet a fix-validating regression test — see the gate
    /// report): does a token's output actually change depending on whether a
    /// prior token's KV context is present? If KV-cached attention is wired
    /// correctly, processing token2 right after token1 (same slot, pos 0 then
    /// pos 1) must give a different result than processing token2 alone on a
    /// fresh slot (pos 0, nothing to attend to but itself).
    #[test]
    fn execute_layer_uses_prior_kv_context_not_just_current_token() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        let backend = GlprocBackend::new(&metadata).unwrap();

        let token1 = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.11).sin()).collect(),
            shape: vec![8],
        };
        let token2 = ActivationBuffer {
            data: (0..8).map(|i| (i as f32 * 0.53).cos()).collect(),
            shape: vec![8],
        };

        // Sequential: token1 then token2, same KV slot (pos 0, then pos 1).
        let mut kv_seq = fixture_kv_slot(&metadata, 16);
        let mut out1 = ActivationBuffer::zeros(vec![8]);
        backend.execute_layer(&mapping, &manifest, &mut kv_seq, &token1, &mut out1).unwrap();
        let mut out2_with_context = ActivationBuffer::zeros(vec![8]);
        backend
            .execute_layer(&mapping, &manifest, &mut kv_seq, &token2, &mut out2_with_context)
            .unwrap();

        // Isolated: token2 alone on a FRESH KV slot (pos 0, no token1 present).
        let mut kv_fresh = fixture_kv_slot(&metadata, 16);
        let mut out2_isolated = ActivationBuffer::zeros(vec![8]);
        backend
            .execute_layer(&mapping, &manifest, &mut kv_fresh, &token2, &mut out2_isolated)
            .unwrap();

        assert_ne!(
            out2_with_context.data, out2_isolated.data,
            "token2's output must depend on whether token1's KV context is present"
        );
    }

    #[test]
    fn execute_layer_rejects_a_missing_tensor() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, mut manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        manifest.tensors.retain(|t| t.name != ATTN_Q);

        let backend = GlprocBackend::new(&metadata).unwrap();
        let mut kv = fixture_kv_slot(&metadata, 16);
        let input = ActivationBuffer::zeros(vec![8]);
        let mut output = ActivationBuffer::zeros(vec![8]);

        let err = backend
            .execute_layer(&mapping, &manifest, &mut kv, &input, &mut output)
            .unwrap_err();
        assert!(matches!(err, GllmError::ExecutionFailed { layer: 0, .. }), "got {err:?}");
    }

    #[test]
    fn execute_layer_rejects_a_dim_mismatched_input() {
        let dir = TempDir::new().unwrap();
        let metadata = fixture_metadata();
        let (mapping, manifest) = write_fixture_layer(dir.path(), 0, 0.0);
        let backend = GlprocBackend::new(&metadata).unwrap();
        let mut kv = fixture_kv_slot(&metadata, 16);

        let input = ActivationBuffer::zeros(vec![5]); // model dim is 8
        let mut output = ActivationBuffer::zeros(vec![5]);

        let err = backend
            .execute_layer(&mapping, &manifest, &mut kv, &input, &mut output)
            .unwrap_err();
        assert!(matches!(err, GllmError::ExecutionFailed { layer: 0, .. }), "got {err:?}");
    }

    /// ARTX10 Wave 1's actual acceptance test: build a complete on-disk GLLM
    /// package (2 real transformer-block layers, a token embedding table,
    /// and an LM head, all `gllm:transformer/standard@v1` / f32), drive every
    /// layer through [`GllmRuntime`](crate::runtime::GllmRuntime) with a real
    /// [`GlprocBackend`], then reduce the final hidden state to a token id —
    /// the same embed -> layers -> norm -> lm_head -> argmax shape
    /// `glproc::runner::Runner` uses, just assembled from a GLLM package
    /// instead of a GGUF file. Proves a model can generate a token through
    /// the GLLM format end to end, not just that one layer computes.
    #[test]
    fn a_real_model_generates_a_token_through_gllm_end_to_end() {
        use crate::checksum::sha256_file;
        use crate::constants::SHARED_FILENAME;
        use crate::runtime::backend::ExecutionBackend;
        use crate::runtime::runtime::GllmRuntime;
        use crate::runtime::types::RuntimeConfig;
        use std::sync::Arc;

        const DIM: usize = 8;
        const VOCAB: usize = 6;
        let metadata = fixture_metadata();
        let dir = TempDir::new().unwrap();

        // Two distinct layers (seeds 0.0 and 1.0), so this actually exercises
        // "run every layer in the stack," not "run one layer twice."
        write_fixture_layer(dir.path(), 0, 0.0);
        write_fixture_layer(dir.path(), 1, 1.0);

        // Shared component: token embedding table `[VOCAB, DIM]`, plus an LM
        // head `[VOCAB, DIM]` and a final norm gain — everything the runtime
        // itself does not compute, so the test does, the same way
        // `glproc::runner::Runner` treats embedding/lm_head as outside the
        // per-layer loop.
        let embed: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32 * 0.13).sin()).collect();
        let lm_head: Vec<f32> = (0..VOCAB * DIM).map(|i| (i as f32 * 0.29).cos()).collect();
        let final_norm = vec![1.0f32; DIM];
        let shared_path = dir.path().join(SHARED_FILENAME);
        let embed_bytes = f32_bytes(&embed);
        let lm_head_bytes = f32_bytes(&lm_head);
        let norm_bytes = f32_bytes(&final_norm);
        let shared_entries = crate::layer_io::write_unit_file(
            &shared_path,
            &[
                ("token_embd.weight", &[VOCAB as u64, DIM as u64], crate::manifest::DType::F32, &embed_bytes),
                ("output.weight", &[VOCAB as u64, DIM as u64], crate::manifest::DType::F32, &lm_head_bytes),
                ("output_norm.weight", &[DIM as u64], crate::manifest::DType::F32, &norm_bytes),
            ],
        )
        .unwrap();

        // Offsets/sizes come from re-parsing what write_unit_file wrote for
        // each layer (tensor layout — alignment, ordering — is that
        // function's job, not this test's to reimplement).
        let layer_manifests: Vec<serde_json::Value> = (0..2u32)
            .map(|i| {
                let path = dir.path().join(crate::manifest::format_layer_filename(i));
                let mapping = LayerMapping::open(&path, Some(i), false).unwrap();
                let tensors: Vec<serde_json::Value> = fixture_layer_tensors(i as f32)
                    .iter()
                    .map(|(name, _, _)| {
                        let (offset, size) = mapping.layer.absolute_range(name).unwrap();
                        serde_json::json!({
                            "name": name, "shape": mapping.layer.tensor(name).unwrap().shape,
                            "dtype": "F32", "offset": offset, "size": size,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "index": i,
                    "file": crate::manifest::format_layer_filename(i),
                    "checksum": format!("sha256:{}", sha256_file(&path).unwrap()),
                    "type": "gllm:transformer/standard@v1",
                    "tensors": tensors,
                })
            })
            .collect();

        let manifest_json = serde_json::json!({
            "gllm_version": "1.0.0",
            "model_id": "org.gwenland.artx10-wave1-fixture",
            "architecture": "transformer",
            "metadata": {
                "vocab_size": VOCAB, "context_length": 64, "embedding_length": DIM,
                "num_layers": 2, "num_heads": metadata.num_heads,
                "head_count_kv": metadata.head_count_kv,
                "rope_dims": metadata.rope_dims, "rope_freq_base": metadata.rope_freq_base,
            },
            "shared": {
                "file": SHARED_FILENAME,
                "checksum": format!("sha256:{}", sha256_file(&shared_path).unwrap()),
                "tensors": shared_entries,
            },
            "layers": layer_manifests,
            "extensions": ["gllm:transformer/standard@v1"],
        });
        std::fs::write(dir.path().join("gllm.json"), manifest_json.to_string()).unwrap();

        // --- drive every layer through the real runtime + real backend ---
        let backend: Arc<dyn ExecutionBackend> = Arc::new(GlprocBackend::new(&metadata).unwrap());
        let mut runtime = GllmRuntime::open(
            dir.path(),
            RuntimeConfig::with_max_seq_len(16),
            backend,
        )
        .unwrap();

        // Token 2's embedding row is the model's input — "run the model on
        // one real token," the smallest meaningful unit of "generate."
        let token_id = 2usize;
        let mut hidden = ActivationBuffer {
            data: embed[token_id * DIM..(token_id + 1) * DIM].to_vec(),
            shape: vec![DIM],
        };
        let stats = runtime.run(&mut hidden).unwrap();
        assert_eq!(stats.layers_executed, 2, "both layers must have run");
        assert!(hidden.data.iter().all(|x| x.is_finite()), "{:?}", hidden.data);

        // --- reduce to a token: norm -> lm_head -> argmax ---
        let mut normed = vec![0.0f32; DIM];
        glproc::kernels::rms_norm_into(&hidden.data, &final_norm, 1e-5, &mut normed);
        let mut logits = vec![0.0f32; VOCAB];
        glproc::kernels::matvec(&lm_head, &normed, &mut logits, VOCAB, DIM);

        assert!(logits.iter().all(|x| x.is_finite()), "{logits:?}");
        let (best, _) = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(best < VOCAB, "argmax must select a real vocabulary id");
    }
}
