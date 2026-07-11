//! Model types for the CUDA engine: host-side staging (`HostModel`) and the
//! VRAM-resident `GpuModel`, plus the upload step between them.
//!
//! The split exists for testability and for the ArchGLML_X2 §12 memory
//! plan: the loader produces a `HostModel` (pure host data, no GPU), and
//! `GpuModel::upload` performs the *entire* VRAM footprint calculation and
//! the single backend-buffer allocation before any byte is copied. If the
//! model does not fit, failure happens there — never mid-generation.

use glcore::GlError;

use crate::buffer::{BackendBuffer, DevSlice, ALIGN};
use crate::dequant::q8_0_row_into;
use crate::driver::Cuda;
use crate::kernels;
use crate::kv_cache::KvCacheDev;

/// KV cache sequence capacity cap — same rationale and value as glproc:
/// pre-allocating a 32k-token cache costs GBs; cap at 4096.
pub const MAX_KV_CONTEXT: usize = 4096;

/// How rotary position embeddings pair up dimensions (mirror of glproc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RopeStyle {
    /// Original llama style: rotate adjacent pairs `(2i, 2i+1)`.
    Norm,
    /// GPT-NeoX style (qwen2, phi, gemma, ...): rotate `(i, i + dim/2)`.
    Neox,
}

/// Hyperparameters of a loaded transformer — field-for-field the same
/// contract as glproc's `ModelConfig`, so the two engines read one GGUF
/// identically.
#[derive(Debug, Clone)]
pub struct GpuModelConfig {
    /// Architecture string from `general.architecture`.
    pub arch: String,
    /// Embedding width.
    pub dim: usize,
    /// Number of transformer blocks.
    pub n_layers: usize,
    /// Number of query heads.
    pub n_heads: usize,
    /// Number of key/value heads (< `n_heads` under GQA).
    pub n_kv_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN inner width.
    pub hidden_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum context length the model was trained for.
    pub max_seq: usize,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// RoPE base frequency.
    pub rope_freq_base: f32,
    /// RoPE dimension pairing convention.
    pub rope_style: RopeStyle,
}

/// One weight tensor in host staging form. Q8_0 tensors keep their raw
/// GGML blocks (uploaded quantized — 4x less VRAM and DRAM traffic, the
/// gl_gemv_q8_0 kernel dequantizes in registers); everything else is f32.
#[derive(Clone)]
pub enum HostWeight {
    /// Dense f32, row-major `[out_features, in_features]`.
    F32(Vec<f32>),
    /// Raw padded Q8_0 blocks (36 B), rows contiguous. Used for the host-side
    /// embedding table (`q8_0_row_into`); matmul weights use `Q8_0Soa`.
    Q8_0(Vec<u8>),
    /// Structure-of-Arrays Q8_0 for matmul weights: contiguous int8 `qs`
    /// `[out, in]` + contiguous f16 block `scales` `[out, in/32]`. Enables the
    /// coalesced `gl_gemv_q8_0_soa` kernel (no interleaved-scale/padding BW loss).
    Q8_0Soa { qs: Vec<u8>, scales: Vec<u8> },
    /// Raw GGML Q4_0 blocks, rows contiguous, `in_features % 32 == 0`.
    /// Embedding-table only since M2.2 (host row dequant); matmul weights
    /// use `Q4_0Soa`. The legacy AoS `gl_gemv_q4_0` kernel still accepts it.
    Q4_0(Vec<u8>),
    /// Structure-of-Arrays Q4_0 for matmul weights (M2.2 Task C-2):
    /// contiguous packed nibbles `qs` `[out, in/2]` + VERBATIM f16 block
    /// scales `[out, in/32]` (no pre-multiply — `d` is the final scale).
    /// 4.5 bpw streamed; the layout `gl_gemv_q4_0_soa` reads.
    Q4_0Soa { qs: Vec<u8>, scales: Vec<u8> },
    /// Raw GGML Q4_K super-blocks (144 B / 256 weights), rows contiguous.
    /// Embedding-table only (host-side row dequant, `q4_k_row_into`);
    /// matmul weights use `Q4KSoa`.
    Q4K(Vec<u8>),
    /// Raw GGML Q6_K super-blocks (210 B / 256 weights), rows contiguous.
    /// Embedding-table only (host row dequant, `q6_k_row_into`); matmul
    /// weights use `Q6KSoa`.
    Q6K(Vec<u8>),
    /// Structure-of-Arrays Q6_K for matmul weights (M2.2 Task C-1): four
    /// contiguous streams — packed low nibbles `ql` `[out, in/2]`, 2-bit
    /// highs `qh` widened into the identical nibble layout `[out, in/2]`
    /// (the bytes-for-ALU trade that lifted the kernel off its 155-183
    /// GB/s compute stall — see `repack::q6_k_to_soa`), verbatim i8
    /// sub-block `scales` `[out, in/16]`, verbatim f16 super-block `d`
    /// `[out, in/256]`. 7.0625 bpw; the layout `gl_gemv_q6_k_soa` reads.
    Q6KSoa { ql: Vec<u8>, qh: Vec<u8>, scales: Vec<u8>, d: Vec<u8> },
    /// Structure-of-Arrays Q4_K for matmul weights (M2.1 Task A): contiguous
    /// packed nibbles `qs` `[out, in/2]` + f16 PRE-MULTIPLIED sub-block
    /// `scales` (`d*sc`) and `mins` (`dmin*m`), each `[out, in/32]` f16.
    /// 5.0 bpw streamed per decode token vs 8.5 for Q8_0 SoA — the layout
    /// `gl_gemv_q4_k_soa` reads (see `repack::q4_k_to_soa`).
    Q4KSoa { qs: Vec<u8>, scales: Vec<u8>, mins: Vec<u8> },
}

impl HostWeight {
    /// VRAM this weight reserves, per backend-buffer region: every separately
    /// uploaded stream starts on an ALIGN boundary, so multi-stream (SoA)
    /// representations must round each stream up individually — summing raw
    /// lengths under-counts by up to `(n_streams - 1) * ALIGN` per matrix.
    pub fn vram_reserved(&self) -> u64 {
        let a = |n: usize| (n as u64).div_ceil(ALIGN) * ALIGN;
        match self {
            HostWeight::F32(v) => a(v.len() * 4),
            HostWeight::Q8_0(b) => a(b.len()),
            HostWeight::Q8_0Soa { qs, scales } => a(qs.len()) + a(scales.len()),
            HostWeight::Q4_0(b) => a(b.len()),
            HostWeight::Q4_0Soa { qs, scales } => a(qs.len()) + a(scales.len()),
            HostWeight::Q4K(b) => a(b.len()),
            HostWeight::Q4KSoa { qs, scales, mins } => a(qs.len()) + a(scales.len()) + a(mins.len()),
            HostWeight::Q6K(b) => a(b.len()),
            HostWeight::Q6KSoa { ql, qh, scales, d } => {
                a(ql.len()) + a(qh.len()) + a(scales.len()) + a(d.len())
            }
        }
    }
}

/// A host-staged weight matrix with its dimensions.
pub struct HostMat {
    /// The payload.
    pub w: HostWeight,
    /// Output features (rows).
    pub out_dim: usize,
    /// Input features (columns; the contiguous axis).
    pub in_dim: usize,
}

impl HostMat {
    /// Vertically stack `self` on top of `other` into one matrix whose rows
    /// are `self`'s rows followed by `other`'s. Both must share `in_dim` and
    /// weight representation. Used to fuse the FFN gate and up projections
    /// into a single `[2*hidden, dim]` weight so one GEMV streams the input
    /// once and one launch replaces two — the FFN is ~60% of a decode token.
    pub(crate) fn stack_rows(self, other: HostMat) -> HostMat {
        debug_assert_eq!(self.in_dim, other.in_dim);
        let w = match (self.w, other.w) {
            (HostWeight::F32(mut a), HostWeight::F32(b)) => {
                a.extend_from_slice(&b);
                HostWeight::F32(a)
            }
            (HostWeight::Q8_0(mut a), HostWeight::Q8_0(b)) => {
                // Rows are whole Q8_0 blocks; concatenation stacks them.
                a.extend_from_slice(&b);
                HostWeight::Q8_0(a)
            }
            (
                HostWeight::Q8_0Soa { qs: mut aq, scales: mut asc },
                HostWeight::Q8_0Soa { qs: bq, scales: bsc },
            ) => {
                // qs and scales are both row-major [out, ..]; concatenating
                // each stacks the rows (gate rows then up rows).
                aq.extend_from_slice(&bq);
                asc.extend_from_slice(&bsc);
                HostWeight::Q8_0Soa { qs: aq, scales: asc }
            }
            (HostWeight::Q4_0(mut a), HostWeight::Q4_0(b)) => {
                // Rows are whole Q4_0 blocks; concatenation stacks them.
                a.extend_from_slice(&b);
                HostWeight::Q4_0(a)
            }
            (
                HostWeight::Q4_0Soa { qs: mut aq, scales: mut asc },
                HostWeight::Q4_0Soa { qs: bq, scales: bsc },
            ) => {
                // Both streams row-major; concatenation stacks the rows.
                aq.extend_from_slice(&bq);
                asc.extend_from_slice(&bsc);
                HostWeight::Q4_0Soa { qs: aq, scales: asc }
            }
            (
                HostWeight::Q4KSoa { qs: mut aq, scales: mut asc, mins: mut amn },
                HostWeight::Q4KSoa { qs: bq, scales: bsc, mins: bmn },
            ) => {
                // All three streams are row-major [out, ..]; concatenating
                // each stacks the rows (gate rows then up rows).
                aq.extend_from_slice(&bq);
                asc.extend_from_slice(&bsc);
                amn.extend_from_slice(&bmn);
                HostWeight::Q4KSoa { qs: aq, scales: asc, mins: amn }
            }
            (
                HostWeight::Q6KSoa { ql: mut al, qh: mut ah, scales: mut asc, d: mut ad },
                HostWeight::Q6KSoa { ql: bl, qh: bh, scales: bsc, d: bd },
            ) => {
                // All four streams row-major; concatenation stacks the rows.
                al.extend_from_slice(&bl);
                ah.extend_from_slice(&bh);
                asc.extend_from_slice(&bsc);
                ad.extend_from_slice(&bd);
                HostWeight::Q6KSoa { ql: al, qh: ah, scales: asc, d: ad }
            }
            // Mixed representations shouldn't happen for a matched gate/up
            // pair, but if they do, the caller must keep them separate.
            _ => panic!("stack_rows: mismatched weight representations"),
        };
        HostMat { w, out_dim: self.out_dim + other.out_dim, in_dim: self.in_dim }
    }

    /// True when `self` and `other` can be [`stack_rows`]-fused (same input
    /// width and same representation).
    pub(crate) fn stackable(&self, other: &HostMat) -> bool {
        self.in_dim == other.in_dim
            && matches!(
                (&self.w, &other.w),
                (HostWeight::F32(_), HostWeight::F32(_))
                    | (HostWeight::Q8_0(_), HostWeight::Q8_0(_))
                    | (HostWeight::Q8_0Soa { .. }, HostWeight::Q8_0Soa { .. })
                    | (HostWeight::Q4_0(_), HostWeight::Q4_0(_))
                    | (HostWeight::Q4_0Soa { .. }, HostWeight::Q4_0Soa { .. })
                    | (HostWeight::Q4KSoa { .. }, HostWeight::Q4KSoa { .. })
                    | (HostWeight::Q6KSoa { .. }, HostWeight::Q6KSoa { .. })
            )
    }
}

/// Host-staged weights of one transformer block (glproc's `LayerWeights`
/// without the CPU-specific fusion — the GPU engine dispatches per matrix).
pub struct HostLayer {
    /// Pre-attention RMSNorm gain, `[dim]`.
    pub attn_norm: Vec<f32>,
    /// Query projection, `[q_dim, dim]`.
    pub wq: HostMat,
    /// Key projection, `[kv_dim, dim]`.
    pub wk: HostMat,
    /// Value projection, `[kv_dim, dim]`.
    pub wv: HostMat,
    /// Attention output projection, `[dim, q_dim]`.
    pub wo: HostMat,
    /// Optional query bias (qwen2-style).
    pub bq: Option<Vec<f32>>,
    /// Optional key bias.
    pub bk: Option<Vec<f32>>,
    /// Optional value bias.
    pub bv: Option<Vec<f32>>,
    /// Optional per-head query RMSNorm gain, `[head_dim]` (qwen3-style).
    pub q_norm: Option<Vec<f32>>,
    /// Optional per-head key RMSNorm gain, `[head_dim]`.
    pub k_norm: Option<Vec<f32>>,
    /// Pre-FFN RMSNorm gain, `[dim]`.
    pub ffn_norm: Vec<f32>,
    /// Fused SwiGLU gate+up projection, `[2*hidden_dim, dim]` — gate rows
    /// `[0, hidden)` then up rows `[hidden, 2*hidden)`. One GEMV computes
    /// both, streaming the input once (FFN is ~60% of a decode token).
    pub w_gate_up: HostMat,
    /// Down projection, `[dim, hidden_dim]`.
    pub w_down: HostMat,
}

/// A fully staged model, ready for [`GpuModel::upload`].
pub struct HostModel {
    /// Hyperparameters.
    pub config: GpuModelConfig,
    /// Token embedding table, `[vocab_size, dim]`. Stays on the host —
    /// one row is dequantized and uploaded per token (a few KB HtoD).
    pub token_embd: HostWeight,
    /// All transformer blocks, in order.
    pub layers: Vec<HostLayer>,
    /// Final RMSNorm gain, `[dim]`.
    pub output_norm: Vec<f32>,
    /// LM head, `[vocab_size, dim]`; the embedding table when tied.
    pub output: HostMat,
}

impl HostModel {
    /// Copy `token`'s embedding row into `out` (`[dim]`), mirroring
    /// `GlprocModel::embed_into`.
    pub fn embed_into(&self, token: u32, out: &mut [f32]) -> Result<(), GlError> {
        let dim = self.config.dim;
        debug_assert_eq!(out.len(), dim);
        let row = token as usize;
        if row >= self.config.vocab_size {
            return Err(GlError::Engine(format!("token id {token} out of embedding range")));
        }
        match &self.token_embd {
            HostWeight::F32(v) => out.copy_from_slice(&v[row * dim..(row + 1) * dim]),
            HostWeight::Q8_0(b) => q8_0_row_into(b, row, dim, out),
            HostWeight::Q8_0Soa { .. }
            | HostWeight::Q4_0Soa { .. }
            | HostWeight::Q4KSoa { .. }
            | HostWeight::Q6KSoa { .. } => {
                unreachable!("embedding table is AoS, never SoA")
            }
            HostWeight::Q4_0(b) => crate::dequant::q4_0_row_into(b, row, dim, out),
            HostWeight::Q4K(b) => crate::dequant::q4_k_row_into(b, row, dim, out),
            HostWeight::Q6K(b) => crate::dequant::q6_k_row_into(b, row, dim, out),
        }
        Ok(())
    }
}

/// One weight tensor resident in VRAM.
pub enum GpuWeight {
    /// Dense f32.
    F32(DevSlice),
    /// Q8_0 blocks (AoS, padded 36 B).
    Q8_0(DevSlice),
    /// SoA Q8_0: contiguous int8 `qs` + separate f16 `scales`.
    Q8_0Soa { qs: DevSlice, scales: DevSlice },
    /// Q4_0 blocks (AoS legacy — no loader path produces this since M2.2).
    Q4_0(DevSlice),
    /// SoA Q4_0: packed nibbles + verbatim f16 block scales.
    Q4_0Soa { qs: DevSlice, scales: DevSlice },
    /// SoA Q4_K: packed nibbles + pre-multiplied f16 sub-block scales/mins.
    Q4KSoa { qs: DevSlice, scales: DevSlice, mins: DevSlice },
    /// SoA Q6_K: low nibbles + 2-bit highs + i8 sub-block scales + f16 d.
    Q6KSoa { ql: DevSlice, qh: DevSlice, scales: DevSlice, d: DevSlice },
}

/// A VRAM weight matrix with launch dimensions.
pub struct GpuMat {
    /// The payload.
    pub w: GpuWeight,
    /// Output features.
    pub out_dim: u32,
    /// Input features.
    pub in_dim: u32,
}

/// VRAM weights of one transformer block.
pub struct GpuLayer {
    pub(crate) attn_norm: DevSlice,
    pub(crate) wq: GpuMat,
    pub(crate) wk: GpuMat,
    pub(crate) wv: GpuMat,
    pub(crate) wo: GpuMat,
    pub(crate) bq: Option<DevSlice>,
    pub(crate) bk: Option<DevSlice>,
    pub(crate) bv: Option<DevSlice>,
    pub(crate) q_norm: Option<DevSlice>,
    pub(crate) k_norm: Option<DevSlice>,
    pub(crate) ffn_norm: DevSlice,
    /// Fused gate+up, `[2*hidden, dim]` (gate rows then up rows).
    pub(crate) w_gate_up: GpuMat,
    pub(crate) w_down: GpuMat,
}

/// Pre-allocated per-forward-pass device buffers (the GPU mirror of
/// glproc's `Workspace`) plus the host staging the hot path needs. All
/// allocated once at upload — the decode loop performs zero allocations.
pub(crate) struct Workspace {
    /// Residual stream, `[dim]`.
    pub x: DevSlice,
    /// RMSNorm output, `[dim]`.
    pub xn: DevSlice,
    /// Q/K/V vectors in one buffer, `[q_dim + 2*kv_dim]`.
    pub qkv: DevSlice,
    /// Attention output, `[q_dim]`.
    pub attn_out: DevSlice,
    /// Projection back to the residual, `[dim]`.
    pub proj: DevSlice,
    /// Fused SwiGLU gate+up output, `[2*hidden_dim]` — the one GEMV over the
    /// fused gate+up weight writes gate into `[0, hidden)` and up into
    /// `[hidden, 2*hidden)`; `silu_mul` then folds them into the first half.
    pub gate_up: DevSlice,
    /// Output logits, `[vocab_size]`.
    pub logits: DevSlice,
    /// Host-precomputed RoPE cos table for every position,
    /// `[kv_capacity * head_dim/2]` (host owns transcendental precision).
    pub rope_cos: DevSlice,
    /// RoPE sin table, same shape.
    pub rope_sin: DevSlice,
    /// Per-token parameters read by the kernels from device memory:
    /// `[0] = pos`, `[1] = cached_len` (both u32). Updated by one tiny HtoD
    /// before each graph replay so the captured graph stays valid across
    /// tokens without re-capture (M2.2).
    pub token_params: DevSlice,
    /// Host copy of the logits, filled once per sampled token.
    pub logits_host: Vec<f32>,
    /// Host staging for the embedding row uploaded per token.
    pub embed_host: Vec<f32>,
    /// Q8_0 decoupled quantizer output (INT8 weights), `[hidden_dim]` max.
    pub q8_qs: DevSlice,
    /// Q8_0 decoupled quantizer output (FP32 block scales), `[hidden_dim / 32]` max.
    pub q8_scales: DevSlice,
    /// Batched prefill scratch, all `[PREFILL_BATCH, width]` row-major. Used only
    /// by `prefill_batched`, which processes up to `PREFILL_BATCH` prompt tokens
    /// per pass so the weight GEMMs stream weights once per tile instead of once
    /// per token. `pf_qs`/`pf_scales` are sized to the widest matmul input
    /// (`hidden_dim`) and reused for every quantize in the batched pass.
    pub pf_x: DevSlice,
    pub pf_xn: DevSlice,
    pub pf_q: DevSlice,
    pub pf_k: DevSlice,
    pub pf_v: DevSlice,
    pub pf_attn: DevSlice,
    pub pf_proj: DevSlice,
    pub pf_gate: DevSlice,
    pub pf_up: DevSlice,
    pub pf_qs: DevSlice,
    pub pf_scales: DevSlice,
}

/// Prompt tokens processed per batched-prefill pass. Bounds the batched
/// workspace VRAM; longer prompts are processed in consecutive chunks.
pub(crate) const PREFILL_BATCH: usize = 32;

/// A model resident in VRAM: weights, KV cache and workspace, all carved
/// from one backend buffer.
pub struct GpuModel {
    /// Hyperparameters.
    pub config: GpuModelConfig,
    /// Embedding table (host side — see [`HostModel::token_embd`]).
    pub(crate) token_embd: HostWeight,
    pub(crate) layers: Vec<GpuLayer>,
    pub(crate) output_norm: DevSlice,
    pub(crate) output: GpuMat,
    pub(crate) kv: KvCacheDev,
    pub(crate) ws: Workspace,
    /// The captured per-token decode graph (M2.2), built lazily on the first
    /// decode step and replayed thereafter. `None` until captured; the
    /// prefill path never uses it.
    pub(crate) graph: Option<crate::driver::GraphExec>,
    buffer: BackendBuffer,
    /// Total VRAM reserved, for the load-time report.
    pub total_vram_bytes: u64,
}

fn align_up(bytes: u64) -> u64 {
    bytes.div_ceil(ALIGN) * ALIGN
}

/// VRAM the whole model needs: every region `upload` will allocate, with
/// each start rounded to the backend-buffer alignment. An upper bound that
/// is exact up to per-region alignment padding.
fn vram_total(host: &HostModel, kv_capacity: usize) -> u64 {
    let c = &host.config;
    let q_dim = c.n_heads * c.head_dim;
    let kv_dim = c.n_kv_heads * c.head_dim;
    let f32s = |n: usize| align_up((n * 4) as u64);
    // vram_reserved already rounds every upload stream to ALIGN individually.
    let mat = |m: &HostMat| m.w.vram_reserved();

    let mut total = 0u64;
    for l in &host.layers {
        total += f32s(l.attn_norm.len()) + f32s(l.ffn_norm.len());
        for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate_up, &l.w_down] {
            total += mat(m);
        }
        for v in [&l.bq, &l.bk, &l.bv, &l.q_norm, &l.k_norm].into_iter().flatten() {
            total += f32s(v.len());
        }
    }
    total += f32s(host.output_norm.len()) + mat(&host.output);
    total += f32s(KvCacheDev::numel(c.n_layers, c.n_kv_heads, c.head_dim, kv_capacity));
    // Workspace.
    total += f32s(c.dim) * 3; // x, xn, proj
    total += f32s(q_dim + 2 * kv_dim); // qkv
    total += f32s(q_dim); // attn_out
    total += f32s(2 * c.hidden_dim); // fused gate+up
    total += f32s(c.vocab_size); // logits
    total += f32s(kv_capacity * (c.head_dim / 2)) * 2; // rope tables
    total += align_up(2 * 4); // token_params [pos, cached_len]
    total += align_up(c.hidden_dim as u64); // q8_qs
    total += f32s(c.hidden_dim / 32); // q8_scales
    // Batched prefill scratch (PREFILL_BATCH rows each).
    let b = PREFILL_BATCH;
    total += f32s(b * c.dim) * 3; // pf_x, pf_xn, pf_proj
    total += f32s(b * q_dim) * 2; // pf_q, pf_attn
    total += f32s(b * kv_dim) * 2; // pf_k, pf_v
    total += f32s(b * c.hidden_dim) * 2; // pf_gate, pf_up
    total += align_up((b * c.hidden_dim) as u64); // pf_qs
    total += f32s(b * c.hidden_dim / 32); // pf_scales
    total
}

/// Upload one f32 slice into the backend buffer.
fn up_f32(cuda: &Cuda, buf: &mut BackendBuffer, v: &[f32]) -> Result<DevSlice, GlError> {
    let s = buf.alloc_f32(v.len())?;
    cuda.htod_f32(s.dptr, v)?;
    Ok(s)
}

fn up_f32_opt(
    cuda: &Cuda,
    buf: &mut BackendBuffer,
    v: &Option<Vec<f32>>,
) -> Result<Option<DevSlice>, GlError> {
    v.as_ref().map(|v| up_f32(cuda, buf, v)).transpose()
}

/// Upload one weight matrix, preserving its representation.
fn up_mat(cuda: &Cuda, buf: &mut BackendBuffer, m: &HostMat) -> Result<GpuMat, GlError> {
    let w = match &m.w {
        HostWeight::F32(v) => GpuWeight::F32(up_f32(cuda, buf, v)?),
        HostWeight::Q8_0(b) => {
            let s = buf.alloc(b.len() as u64)?;
            cuda.htod(s.dptr, b)?;
            GpuWeight::Q8_0(s)
        }
        HostWeight::Q8_0Soa { qs, scales } => {
            let dq = buf.alloc(qs.len() as u64)?;
            cuda.htod(dq.dptr, qs)?;
            let ds = buf.alloc(scales.len() as u64)?;
            cuda.htod(ds.dptr, scales)?;
            GpuWeight::Q8_0Soa { qs: dq, scales: ds }
        }
        HostWeight::Q4_0(b) => {
            let s = buf.alloc(b.len() as u64)?;
            cuda.htod(s.dptr, b)?;
            GpuWeight::Q4_0(s)
        }
        HostWeight::Q4_0Soa { qs, scales } => {
            let dq = buf.alloc(qs.len() as u64)?;
            cuda.htod(dq.dptr, qs)?;
            let ds = buf.alloc(scales.len() as u64)?;
            cuda.htod(ds.dptr, scales)?;
            GpuWeight::Q4_0Soa { qs: dq, scales: ds }
        }
        HostWeight::Q4K(_) | HostWeight::Q6K(_) => {
            unreachable!("raw k-quants are embedding-only; matmul weights are repacked to SoA")
        }
        HostWeight::Q6KSoa { ql, qh, scales, d } => {
            let dl = buf.alloc(ql.len() as u64)?;
            cuda.htod(dl.dptr, ql)?;
            let dh = buf.alloc(qh.len() as u64)?;
            cuda.htod(dh.dptr, qh)?;
            let ds = buf.alloc(scales.len() as u64)?;
            cuda.htod(ds.dptr, scales)?;
            let dd = buf.alloc(d.len() as u64)?;
            cuda.htod(dd.dptr, d)?;
            GpuWeight::Q6KSoa { ql: dl, qh: dh, scales: ds, d: dd }
        }
        HostWeight::Q4KSoa { qs, scales, mins } => {
            let dq = buf.alloc(qs.len() as u64)?;
            cuda.htod(dq.dptr, qs)?;
            let ds = buf.alloc(scales.len() as u64)?;
            cuda.htod(ds.dptr, scales)?;
            let dm = buf.alloc(mins.len() as u64)?;
            cuda.htod(dm.dptr, mins)?;
            GpuWeight::Q4KSoa { qs: dq, scales: ds, mins: dm }
        }
    };
    Ok(GpuMat { w, out_dim: m.out_dim as u32, in_dim: m.in_dim as u32 })
}

impl GpuModel {
    /// Compute the full VRAM footprint, allocate the backend buffer once,
    /// and copy every weight in. Fails before copying anything when the
    /// footprint exceeds free VRAM (ADR-005: early, explicit failure).
    pub fn upload(cuda: &Cuda, host: HostModel) -> Result<GpuModel, GlError> {
        let c = host.config.clone();
        let kv_capacity = c.max_seq.clamp(1, MAX_KV_CONTEXT);
        let total = vram_total(&host, kv_capacity);

        let (free, _) = cuda.mem_get_info()?;
        if total > free as u64 {
            return Err(GlError::Engine(format!(
                "model needs {} MiB VRAM but only {} MiB is free — no partial \
                 load, no silent OOM mid-generation",
                total >> 20,
                free >> 20,
            )));
        }
        let mut buf = BackendBuffer::new(cuda, total)?;

        let mut layers = Vec::with_capacity(host.layers.len());
        for l in &host.layers {
            layers.push(GpuLayer {
                attn_norm: up_f32(cuda, &mut buf, &l.attn_norm)?,
                wq: up_mat(cuda, &mut buf, &l.wq)?,
                wk: up_mat(cuda, &mut buf, &l.wk)?,
                wv: up_mat(cuda, &mut buf, &l.wv)?,
                wo: up_mat(cuda, &mut buf, &l.wo)?,
                bq: up_f32_opt(cuda, &mut buf, &l.bq)?,
                bk: up_f32_opt(cuda, &mut buf, &l.bk)?,
                bv: up_f32_opt(cuda, &mut buf, &l.bv)?,
                q_norm: up_f32_opt(cuda, &mut buf, &l.q_norm)?,
                k_norm: up_f32_opt(cuda, &mut buf, &l.k_norm)?,
                ffn_norm: up_f32(cuda, &mut buf, &l.ffn_norm)?,
                w_gate_up: up_mat(cuda, &mut buf, &l.w_gate_up)?,
                w_down: up_mat(cuda, &mut buf, &l.w_down)?,
            });
        }
        let output_norm = up_f32(cuda, &mut buf, &host.output_norm)?;
        let output = up_mat(cuda, &mut buf, &host.output)?;

        let kv_slice =
            buf.alloc_f32(KvCacheDev::numel(c.n_layers, c.n_kv_heads, c.head_dim, kv_capacity))?;
        let kv = KvCacheDev::new(kv_slice, c.n_layers, c.n_kv_heads, c.head_dim, kv_capacity);

        // RoPE tables for every position, computed on the host with exactly
        // glproc's frequency formula (the 1e-7 RoPE ε forbids device
        // sin.approx).
        let half = c.head_dim / 2;
        let mut cos_all = Vec::with_capacity(kv_capacity * half);
        let mut sin_all = Vec::with_capacity(kv_capacity * half);
        for pos in 0..kv_capacity {
            let (cos, sin) = kernels::rope_tables(pos, c.head_dim, c.rope_freq_base);
            cos_all.extend_from_slice(&cos);
            sin_all.extend_from_slice(&sin);
        }

        let q_dim = c.n_heads * c.head_dim;
        let kv_dim = c.n_kv_heads * c.head_dim;
        let ws = Workspace {
            x: buf.alloc_f32(c.dim)?,
            xn: buf.alloc_f32(c.dim)?,
            qkv: buf.alloc_f32(q_dim + 2 * kv_dim)?,
            attn_out: buf.alloc_f32(q_dim)?,
            proj: buf.alloc_f32(c.dim)?,
            gate_up: buf.alloc_f32(2 * c.hidden_dim)?,
            logits: buf.alloc_f32(c.vocab_size)?,
            rope_cos: up_f32(cuda, &mut buf, &cos_all)?,
            rope_sin: up_f32(cuda, &mut buf, &sin_all)?,
            token_params: buf.alloc(2 * 4)?, // [pos, cached_len] as u32
            logits_host: vec![0.0; c.vocab_size],
            embed_host: vec![0.0; c.dim],
            q8_qs: buf.alloc(c.hidden_dim as u64)?,
            q8_scales: buf.alloc_f32(c.hidden_dim / 32)?,
            pf_x: buf.alloc_f32(PREFILL_BATCH * c.dim)?,
            pf_xn: buf.alloc_f32(PREFILL_BATCH * c.dim)?,
            pf_q: buf.alloc_f32(PREFILL_BATCH * q_dim)?,
            pf_k: buf.alloc_f32(PREFILL_BATCH * kv_dim)?,
            pf_v: buf.alloc_f32(PREFILL_BATCH * kv_dim)?,
            pf_attn: buf.alloc_f32(PREFILL_BATCH * q_dim)?,
            pf_proj: buf.alloc_f32(PREFILL_BATCH * c.dim)?,
            pf_gate: buf.alloc_f32(PREFILL_BATCH * c.hidden_dim)?,
            pf_up: buf.alloc_f32(PREFILL_BATCH * c.hidden_dim)?,
            pf_qs: buf.alloc((PREFILL_BATCH * c.hidden_dim) as u64)?,
            pf_scales: buf.alloc_f32(PREFILL_BATCH * c.hidden_dim / 32)?,
        };

        Ok(GpuModel {
            config: c,
            token_embd: host.token_embd,
            layers,
            output_norm,
            output,
            kv,
            ws,
            graph: None,
            buffer: buf,
            total_vram_bytes: total,
        })
    }

    /// Release the model's VRAM (the whole backend buffer).
    pub fn free(self, cuda: &Cuda) -> Result<(), GlError> {
        self.buffer.free(cuda)
    }
}
