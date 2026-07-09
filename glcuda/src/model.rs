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
use crate::dequant::{q8_0_row_into, Q8_0_BLOCK_BYTES};
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
    /// Raw GGML Q8_0 blocks, rows contiguous, `in_features % 32 == 0`.
    Q8_0(Vec<u8>),
}

impl HostWeight {
    /// Bytes this weight occupies in VRAM.
    pub fn vram_bytes(&self) -> u64 {
        match self {
            HostWeight::F32(v) => (v.len() * 4) as u64,
            HostWeight::Q8_0(b) => b.len() as u64,
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
    /// SwiGLU gate projection, `[hidden_dim, dim]`.
    pub w_gate: HostMat,
    /// SwiGLU up projection, `[hidden_dim, dim]`.
    pub w_up: HostMat,
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
        }
        Ok(())
    }
}

/// One weight tensor resident in VRAM.
pub enum GpuWeight {
    /// Dense f32.
    F32(DevSlice),
    /// Raw GGML Q8_0 blocks (gl_gemv_q8_0 consumes them directly).
    Q8_0(DevSlice),
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
    pub(crate) w_gate: GpuMat,
    pub(crate) w_up: GpuMat,
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
    /// SwiGLU gate, `[hidden_dim]`.
    pub gate: DevSlice,
    /// SwiGLU up, `[hidden_dim]`.
    pub up: DevSlice,
    /// Output logits, `[vocab_size]`.
    pub logits: DevSlice,
    /// Host-precomputed RoPE cos table for every position,
    /// `[kv_capacity * head_dim/2]` (host owns transcendental precision).
    pub rope_cos: DevSlice,
    /// RoPE sin table, same shape.
    pub rope_sin: DevSlice,
    /// Host copy of the logits, filled once per sampled token.
    pub logits_host: Vec<f32>,
    /// Host staging for the embedding row uploaded per token.
    pub embed_host: Vec<f32>,
}

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
    let mat = |m: &HostMat| align_up(m.w.vram_bytes());

    let mut total = 0u64;
    for l in &host.layers {
        total += f32s(l.attn_norm.len()) + f32s(l.ffn_norm.len());
        for m in [&l.wq, &l.wk, &l.wv, &l.wo, &l.w_gate, &l.w_up, &l.w_down] {
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
    total += f32s(c.hidden_dim) * 2; // gate, up
    total += f32s(c.vocab_size); // logits
    total += f32s(kv_capacity * (c.head_dim / 2)) * 2; // rope tables
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
            debug_assert_eq!(b.len(), m.out_dim * m.in_dim / 32 * Q8_0_BLOCK_BYTES);
            let s = buf.alloc(b.len() as u64)?;
            cuda.htod(s.dptr, b)?;
            GpuWeight::Q8_0(s)
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
                w_gate: up_mat(cuda, &mut buf, &l.w_gate)?,
                w_up: up_mat(cuda, &mut buf, &l.w_up)?,
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
            gate: buf.alloc_f32(c.hidden_dim)?,
            up: buf.alloc_f32(c.hidden_dim)?,
            logits: buf.alloc_f32(c.vocab_size)?,
            rope_cos: up_f32(cuda, &mut buf, &cos_all)?,
            rope_sin: up_f32(cuda, &mut buf, &sin_all)?,
            logits_host: vec![0.0; c.vocab_size],
            embed_host: vec![0.0; c.dim],
        };

        Ok(GpuModel {
            config: c,
            token_embd: host.token_embd,
            layers,
            output_norm,
            output,
            kv,
            ws,
            buffer: buf,
            total_vram_bytes: total,
        })
    }

    /// Release the model's VRAM (the whole backend buffer).
    pub fn free(self, cuda: &Cuda) -> Result<(), GlError> {
        self.buffer.free(cuda)
    }
}
