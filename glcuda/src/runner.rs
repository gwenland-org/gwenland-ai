//! The transformer forward pass on the GPU: the static layer graph of
//! ArchGLML_X2 §13, walked once per token.
//!
//! Scheduling model (M2): every kernel is submitted asynchronously to the
//! default stream in graph order — the stream ordering *is* the dependency
//! edge set, so the host never waits mid-layer. The only synchronization
//! point per token is the logits download before sampling. Host work per
//! token: one embedding-row upload, one logits download, sampling.
//!
//! Hot-path rules (mirroring glproc's runner):
//! * zero allocation per token — every device buffer was carved from the
//!   backend buffer at upload, host buffers live in the workspace
//! * cursor-based KV cache, one advance per token
//! * prefill = the same step per prompt token, logits only for the last
//!   (batched GEMM prefill is an M2.1 concern; correctness first)

use std::time::Instant;

use glcore::GlError;

use crate::driver::Cuda;
use crate::ffi::CUdeviceptr;
use crate::kernels::KernelSet;
use crate::model::{GpuMat, GpuModel, GpuWeight, RopeStyle, PREFILL_BATCH};
use crate::sampler::{apply_repetition_penalty, Sampler};

/// How many recent tokens the repetition penalty looks back over — same
/// window as glproc (and llama.cpp's `repeat_last_n` default).
const REPEAT_WINDOW: usize = 64;

/// Wall-clock timing for one [`GpuModel::generate`] call, split at the
/// prefill/decode boundary (mirror of glproc's `GenTiming`).
#[derive(Debug, Clone, Copy, Default)]
pub struct GenTiming {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Time to process the prompt.
    pub prefill: std::time::Duration,
    /// Time in the decode loop.
    pub decode: std::time::Duration,
}

fn gemv_w(
    cuda: &Cuda,
    k: &KernelSet,
    ws: &crate::model::Workspace,
    m: &GpuMat,
    x: CUdeviceptr,
    y: CUdeviceptr,
) -> Result<(), GlError> {
    match &m.w {
        GpuWeight::F32(s) => k.gemv(cuda, s.dptr, x, y, m.out_dim, m.in_dim),
        GpuWeight::Q8_0(s) => {
            k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
            k.gemv_q8_0(cuda, s.dptr, ws.q8_qs.dptr, ws.q8_scales.dptr, y, m.out_dim, m.in_dim)
        }
        GpuWeight::Q8_0Soa { qs, scales } => {
            k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
            k.gemv_q8_0_soa(
                cuda,
                qs.dptr,
                scales.dptr,
                ws.q8_qs.dptr,
                ws.q8_scales.dptr,
                y,
                m.out_dim,
                m.in_dim,
            )
        }
        GpuWeight::Q4_0(s) => k.gemv_q4_0(cuda, s.dptr, x, y, m.out_dim, m.in_dim),
        GpuWeight::Q4_0Soa { qs, scales } => {
            k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
            k.gemv_q4_0_soa(
                cuda,
                qs.dptr,
                scales.dptr,
                ws.q8_qs.dptr,
                ws.q8_scales.dptr,
                y,
                m.out_dim,
                m.in_dim,
            )
        }
        GpuWeight::Q4KSoa { qs, scales, mins } => {
            k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
            k.gemv_q4_k_soa(
                cuda,
                qs.dptr,
                scales.dptr,
                mins.dptr,
                ws.q8_qs.dptr,
                ws.q8_scales.dptr,
                y,
                m.out_dim,
                m.in_dim,
            )
        }
        GpuWeight::Q6KSoa { ql, qh, scales, d } => {
            k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
            k.gemv_q6_k_soa(
                cuda,
                ql.dptr,
                qh.dptr,
                scales.dptr,
                d.dptr,
                ws.q8_qs.dptr,
                ws.q8_scales.dptr,
                y,
                m.out_dim,
                m.in_dim,
            )
        }
    }
}

/// Device address `elems` f32 past `base`.
#[inline(always)]
fn at(base: CUdeviceptr, elems: usize) -> CUdeviceptr {
    base + (elems * 4) as u64
}

/// Batched matmul of `rows` output rows starting at `row0` of `m`, for `n`
/// tokens: `y[n, rows] = x[n, in] @ m[row0..row0+rows, :]^T`. Q8_0-SoA weights
/// use the batched GEMM (weight streamed once per token tile); f32 falls back
/// to a per-token GEMV. `x_qs`/`x_scales` are the int8-quantized `x_f32`
/// (produced once by the caller). `row0 > 0` is only used for the gate/up split
/// and only occurs with Q8_0-SoA or f32 weights.
#[allow(clippy::too_many_arguments)]
fn gemm_rows(
    cuda: &Cuda,
    k: &KernelSet,
    m: &GpuMat,
    row0: u32,
    rows: u32,
    x_f32: CUdeviceptr,
    x_qs: CUdeviceptr,
    x_scales: CUdeviceptr,
    y: CUdeviceptr,
    n: u32,
) -> Result<(), GlError> {
    let inb = m.in_dim; // in elements
    match &m.w {
        GpuWeight::Q8_0Soa { qs, scales } => {
            let wqs = qs.dptr + (row0 * inb) as u64; // int8, 1 B/elem
            let wsc = scales.dptr + (row0 * (inb / 32) * 2) as u64; // f16, 2 B/block
            // Runtime kernel selection (M2.1 Task B): the tensor-core GEMM
            // on sm_75+, the dp4a GEMM as the sm_70 fallback. Same weight
            // bytes either way; the MMA path needs whole 8-row output tiles
            // (every real model dim satisfies this — the guard is for odd
            // test shapes). The prefill scratch is PREFILL_BATCH rows, so
            // the MMA's read-padding to 8 token rows is always in bounds.
            if k.has_mma() && rows.is_multiple_of(8) {
                k.gemm_mma_q8(cuda, wqs, wsc, x_qs, x_scales, y, rows, inb, n)
            } else {
                k.gemm_q8_0_soa(cuda, wqs, wsc, x_qs, x_scales, y, rows, inb, n)
            }
        }
        GpuWeight::F32(s) => {
            let w = s.dptr + (row0 * inb) as u64 * 4;
            for t in 0..n {
                let xt = x_f32 + (t * inb) as u64 * 4;
                let yt = y + (t * rows) as u64 * 4;
                k.gemv(cuda, w, xt, yt, rows, inb)?;
            }
            Ok(())
        }
        GpuWeight::Q4_0(s) => {
            debug_assert_eq!(row0, 0, "Q4_0 batched matmul does not use row offsets");
            for t in 0..n {
                let xt = x_f32 + (t * inb) as u64 * 4;
                let yt = y + (t * rows) as u64 * 4;
                k.gemv_q4_0(cuda, s.dptr, xt, yt, rows, inb)?;
            }
            Ok(())
        }
        // Q4_0 SoA prefill: per-token GEMV over the pre-quantized rows,
        // same fallback shape as Q4_K below.
        GpuWeight::Q4_0Soa { qs, scales } => {
            let wqs = qs.dptr + (row0 * (inb / 2)) as u64; // nibbles, 0.5 B/elem
            let wsc = scales.dptr + (row0 * (inb / 32) * 2) as u64; // f16/block
            for t in 0..n {
                let xq = x_qs + (t * inb) as u64;
                let xs = x_scales + (t * (inb / 32)) as u64 * 4;
                let yt = y + (t * rows) as u64 * 4;
                k.gemv_q4_0_soa(cuda, wqs, wsc, xq, xs, yt, rows, inb)?;
            }
            Ok(())
        }
        // Q6_K SoA prefill: per-token GEMV fallback, same shape as Q4_K.
        GpuWeight::Q6KSoa { ql, qh, scales, d } => {
            let wql = ql.dptr + (row0 * (inb / 2)) as u64; // low nibbles
            let wqh = qh.dptr + (row0 * (inb / 2)) as u64; // 2-bit highs (widened)
            let wsc = scales.dptr + (row0 * (inb / 16)) as u64; // i8/sub-block
            let wd = d.dptr + (row0 * (inb / 256) * 2) as u64; // f16/super-block
            for t in 0..n {
                let xq = x_qs + (t * inb) as u64;
                let xs = x_scales + (t * (inb / 32)) as u64 * 4;
                let yt = y + (t * rows) as u64 * 4;
                k.gemv_q6_k_soa(cuda, wql, wqh, wsc, wd, xq, xs, yt, rows, inb)?;
            }
            Ok(())
        }
        // Q4_K SoA prefill: per-token GEMV over the already-quantized rows of
        // x_qs/x_scales. Streams the weight once per token (no 4-token tile
        // yet) — Task A ships the decode kernel; a batched Q4_K GEMM is the
        // Task B / M2.1 follow-up if Q4_K prefill throughput matters.
        GpuWeight::Q4KSoa { qs, scales, mins } => {
            let wqs = qs.dptr + (row0 * (inb / 2)) as u64; // nibbles, 0.5 B/elem
            let wsub = scales.dptr + (row0 * (inb / 32) * 2) as u64; // f16/sub-block
            let wmin = mins.dptr + (row0 * (inb / 32) * 2) as u64;
            for t in 0..n {
                let xq = x_qs + (t * inb) as u64; // int8, 1 B/elem
                let xs = x_scales + (t * (inb / 32)) as u64 * 4; // f32/block
                let yt = y + (t * rows) as u64 * 4;
                k.gemv_q4_k_soa(cuda, wqs, wsub, wmin, xq, xs, yt, rows, inb)?;
            }
            Ok(())
        }
        GpuWeight::Q8_0(_) => Err(GlError::Engine(
            "batched prefill does not support AoS Q8_0 matmul weights".into(),
        )),
    }
}

impl GpuModel {
    /// Upload `token`'s embedding into the residual stream and write the
    /// per-token params (`pos`, `cached_len`) into device memory — the only
    /// host→device work each token, done *before* the kernel sequence (or
    /// its graph replay) reads them.
    fn set_token_inputs(&mut self, cuda: &Cuda, token: u32, pos: usize) -> Result<(), GlError> {
        let mut embed = std::mem::take(&mut self.ws.embed_host);
        let r = self.embed_row(token, &mut embed);
        self.ws.embed_host = embed;
        r?;
        cuda.htod_f32(self.ws.x.dptr, &self.ws.embed_host)?;
        // token_params = [pos, cached_len] (cached_len = pos + 1).
        let params = [pos as u32, (pos + 1) as u32];
        // SAFETY: reinterpret the 2 u32s as bytes for the HtoD.
        let bytes = unsafe {
            std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), std::mem::size_of_val(&params))
        };
        cuda.htod(self.ws.token_params.dptr, bytes)
    }

    /// Issue the per-token forward-pass kernel sequence. Reads `pos` /
    /// `cached_len` from `token_params` in device memory (set by
    /// [`Self::set_token_inputs`]), so the exact same sequence is valid for
    /// every token — which is what lets it be captured once into a graph and
    /// replayed (M2.2). Does no host↔device transfer and does not touch the
    /// KV cursor; the caller advances it.
    fn record_forward(&self, cuda: &Cuda, k: &KernelSet, want_logits: bool) -> Result<(), GlError> {
        let c = &self.config;
        let dim = c.dim as u32;
        let head_dim = c.head_dim;
        let q_dim = c.n_heads * head_dim;
        let kv_dim = c.n_kv_heads * head_dim;
        let heads_per_kv = (c.n_heads / c.n_kv_heads.max(1)).max(1) as u32;
        let neox = c.rope_style == RopeStyle::Neox;
        let head_stride = self.kv.head_stride() as u32;
        let pos_ptr = self.ws.token_params.dptr; // &token_params[0] == pos
        let clen_ptr = self.ws.token_params.dptr + 4; // &token_params[1] == cached_len

        let ws = &self.ws;
        let (x, xn) = (ws.x.dptr, ws.xn.dptr);
        let q_ptr = ws.qkv.dptr;
        let k_ptr = at(ws.qkv.dptr, q_dim);
        let v_ptr = at(ws.qkv.dptr, q_dim + kv_dim);

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            k.rms_norm(cuda, x, layer.attn_norm.dptr, xn, dim, c.rms_eps)?;
            gemv_w(cuda, k, ws, &layer.wq, xn, q_ptr)?;
            gemv_w(cuda, k, ws, &layer.wk, xn, k_ptr)?;
            gemv_w(cuda, k, ws, &layer.wv, xn, v_ptr)?;

            if let Some(b) = &layer.bq {
                k.add(cuda, q_ptr, b.dptr, q_dim as u32)?;
            }
            if let Some(b) = &layer.bk {
                k.add(cuda, k_ptr, b.dptr, kv_dim as u32)?;
            }
            if let Some(b) = &layer.bv {
                k.add(cuda, v_ptr, b.dptr, kv_dim as u32)?;
            }

            // qwen3-style per-head RMSNorm on Q/K, before RoPE.
            if let Some(qn) = &layer.q_norm {
                for h in 0..c.n_heads {
                    let seg = at(q_ptr, h * head_dim);
                    k.rms_norm(cuda, seg, qn.dptr, seg, head_dim as u32, c.rms_eps)?;
                }
            }
            if let Some(kn) = &layer.k_norm {
                for h in 0..c.n_kv_heads {
                    let seg = at(k_ptr, h * head_dim);
                    k.rms_norm(cuda, seg, kn.dptr, seg, head_dim as u32, c.rms_eps)?;
                }
            }

            // RoPE reads `pos` from device memory (token-invariant args).
            k.rope(cuda, q_ptr, ws.rope_cos.dptr, ws.rope_sin.dptr, c.n_heads as u32, head_dim as u32, neox, pos_ptr)?;
            k.rope(cuda, k_ptr, ws.rope_cos.dptr, ws.rope_sin.dptr, c.n_kv_heads as u32, head_dim as u32, neox, pos_ptr)?;

            // KV write is a single kernel per K/V per layer (computes the
            // destination from device `pos`) — replaces the per-head memcpy
            // and is graph-static. read_k/read_v(l, 0) give this layer's
            // cache base (independent of the cursor).
            k.kv_write(cuda, self.kv.read_k(l, 0), k_ptr, pos_ptr, head_dim as u32, c.n_kv_heads as u32, head_stride)?;
            k.kv_write(cuda, self.kv.read_v(l, 0), v_ptr, pos_ptr, head_dim as u32, c.n_kv_heads as u32, head_stride)?;

            // Fused decode attention over ALL heads (cached_len from device).
            let scale = 1.0 / (head_dim as f32).sqrt();
            k.attn_decode(
                cuda,
                q_ptr,
                self.kv.read_k(l, 0),
                self.kv.read_v(l, 0),
                ws.attn_out.dptr,
                c.n_heads as u32,
                head_dim as u32,
                clen_ptr,
                heads_per_kv,
                head_stride,
                scale,
            )?;

            gemv_w(cuda, k, ws, &layer.wo, ws.attn_out.dptr, ws.proj.dptr)?;
            k.add(cuda, x, ws.proj.dptr, dim)?;

            // --- SwiGLU feed-forward block ---
            // One GEMV over the fused gate+up weight streams `xn` once and
            // writes gate into [0, hidden) and up into [hidden, 2*hidden).
            k.rms_norm(cuda, x, layer.ffn_norm.dptr, xn, dim, c.rms_eps)?;
            let gate = ws.gate_up.dptr;
            let up = at(ws.gate_up.dptr, c.hidden_dim);
            gemv_w(cuda, k, ws, &layer.w_gate_up, xn, gate)?;
            k.silu_mul(cuda, gate, up, c.hidden_dim as u32)?;
            gemv_w(cuda, k, ws, &layer.w_down, gate, ws.proj.dptr)?;
            k.add(cuda, x, ws.proj.dptr, dim)?;
        }

        if want_logits {
            k.rms_norm(cuda, x, self.output_norm.dptr, xn, dim, c.rms_eps)?;
            gemv_w(cuda, k, ws, &self.output, xn, ws.logits.dptr)?;
        }
        Ok(())
    }

    /// Batched prefill: run the whole prompt through the model, processing up to
    /// `PREFILL_BATCH` tokens per pass so the weight matmuls become batched GEMMs
    /// (each weight streamed once per 4-token tile instead of once per token).
    /// The weight-free per-token work (RoPE, KV write, attention) stays a loop
    /// over the batch, using the same kernels as the sequential path — identical
    /// semantics. Leaves the last prompt token's logits in `ws.logits` and
    /// advances the KV cursor to `prompt.len()`.
    pub fn prefill_batched(&mut self, cuda: &Cuda, k: &KernelSet, prompt: &[u32]) -> Result<(), GlError> {
        let c = &self.config;
        let dim = c.dim;
        let head_dim = c.head_dim;
        let q_dim = c.n_heads * head_dim;
        let kv_dim = c.n_kv_heads * head_dim;
        let hidden = c.hidden_dim;
        let n_heads = c.n_heads;
        let n_kv_heads = c.n_kv_heads;
        let heads_per_kv = (n_heads / n_kv_heads.max(1)).max(1) as u32;
        let neox = c.rope_style == RopeStyle::Neox;
        let rms_eps = c.rms_eps;
        let head_stride = self.kv.head_stride() as u32;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // Workspace device pointers (Copy) — capturing them ends the &self.ws
        // borrow so the embedding loop can mutate ws.embed_host.
        let ws = &self.ws;
        let (pf_x, pf_xn) = (ws.pf_x.dptr, ws.pf_xn.dptr);
        let (pf_q, pf_k, pf_v) = (ws.pf_q.dptr, ws.pf_k.dptr, ws.pf_v.dptr);
        let (pf_attn, pf_proj) = (ws.pf_attn.dptr, ws.pf_proj.dptr);
        let (pf_gate, pf_up) = (ws.pf_gate.dptr, ws.pf_up.dptr);
        let (pf_qs, pf_scales) = (ws.pf_qs.dptr, ws.pf_scales.dptr);
        let tp = ws.token_params.dptr;
        let (pos_ptr, clen_ptr) = (tp, tp + 4);
        let (rope_cos, rope_sin) = (ws.rope_cos.dptr, ws.rope_sin.dptr);
        let single_xn = ws.xn.dptr;
        let logits = ws.logits.dptr;
        let fq = |base: CUdeviceptr, elems: usize| base + (elems as u64) * 4;

        let p = prompt.len();
        if p > self.kv.max_context {
            return Err(GlError::Engine(format!(
                "prompt length {p} exceeds context window {}",
                self.kv.max_context
            )));
        }

        let mut base = 0usize;
        while base < p {
            let n = (p - base).min(PREFILL_BATCH);

            // Embed this chunk's tokens into pf_x rows.
            for i in 0..n {
                let mut embed = std::mem::take(&mut self.ws.embed_host);
                let r = self.embed_row(prompt[base + i], &mut embed);
                self.ws.embed_host = embed;
                r?;
                cuda.htod_f32(fq(pf_x, i * dim), &self.ws.embed_host)?;
            }

            for l in 0..self.layers.len() {
                let layer = &self.layers[l];

                // --- attention block ---
                for i in 0..n {
                    k.rms_norm(cuda, fq(pf_x, i * dim), layer.attn_norm.dptr, fq(pf_xn, i * dim), dim as u32, rms_eps)?;
                }
                k.quantize_q8(cuda, pf_xn, pf_qs, pf_scales, (n * dim) as u32)?;
                gemm_rows(cuda, k, &layer.wq, 0, q_dim as u32, pf_xn, pf_qs, pf_scales, pf_q, n as u32)?;
                gemm_rows(cuda, k, &layer.wk, 0, kv_dim as u32, pf_xn, pf_qs, pf_scales, pf_k, n as u32)?;
                gemm_rows(cuda, k, &layer.wv, 0, kv_dim as u32, pf_xn, pf_qs, pf_scales, pf_v, n as u32)?;

                for i in 0..n {
                    let pos = (base + i) as u32;
                    let params = [pos, pos + 1];
                    let bytes = unsafe {
                        std::slice::from_raw_parts(params.as_ptr().cast::<u8>(), 8)
                    };
                    cuda.htod(tp, bytes)?;
                    let (qi, ki, vi) = (fq(pf_q, i * q_dim), fq(pf_k, i * kv_dim), fq(pf_v, i * kv_dim));

                    if let Some(b) = &layer.bq {
                        k.add(cuda, qi, b.dptr, q_dim as u32)?;
                    }
                    if let Some(b) = &layer.bk {
                        k.add(cuda, ki, b.dptr, kv_dim as u32)?;
                    }
                    if let Some(b) = &layer.bv {
                        k.add(cuda, vi, b.dptr, kv_dim as u32)?;
                    }
                    if let Some(qn) = &layer.q_norm {
                        for h in 0..n_heads {
                            let seg = fq(qi, h * head_dim);
                            k.rms_norm(cuda, seg, qn.dptr, seg, head_dim as u32, rms_eps)?;
                        }
                    }
                    if let Some(kn) = &layer.k_norm {
                        for h in 0..n_kv_heads {
                            let seg = fq(ki, h * head_dim);
                            k.rms_norm(cuda, seg, kn.dptr, seg, head_dim as u32, rms_eps)?;
                        }
                    }
                    k.rope(cuda, qi, rope_cos, rope_sin, n_heads as u32, head_dim as u32, neox, pos_ptr)?;
                    k.rope(cuda, ki, rope_cos, rope_sin, n_kv_heads as u32, head_dim as u32, neox, pos_ptr)?;
                    k.kv_write(cuda, self.kv.read_k(l, 0), ki, pos_ptr, head_dim as u32, n_kv_heads as u32, head_stride)?;
                    k.kv_write(cuda, self.kv.read_v(l, 0), vi, pos_ptr, head_dim as u32, n_kv_heads as u32, head_stride)?;
                    k.attn_decode(
                        cuda, qi, self.kv.read_k(l, 0), self.kv.read_v(l, 0), fq(pf_attn, i * q_dim),
                        n_heads as u32, head_dim as u32, clen_ptr, heads_per_kv, head_stride, scale,
                    )?;
                    self.kv.advance();
                }

                k.quantize_q8(cuda, pf_attn, pf_qs, pf_scales, (n * q_dim) as u32)?;
                gemm_rows(cuda, k, &layer.wo, 0, dim as u32, pf_attn, pf_qs, pf_scales, pf_proj, n as u32)?;
                k.add(cuda, pf_x, pf_proj, (n * dim) as u32)?;

                // --- SwiGLU feed-forward block ---
                for i in 0..n {
                    k.rms_norm(cuda, fq(pf_x, i * dim), layer.ffn_norm.dptr, fq(pf_xn, i * dim), dim as u32, rms_eps)?;
                }
                k.quantize_q8(cuda, pf_xn, pf_qs, pf_scales, (n * dim) as u32)?;
                // gate = fused rows [0, hidden); up = fused rows [hidden, 2*hidden).
                gemm_rows(cuda, k, &layer.w_gate_up, 0, hidden as u32, pf_xn, pf_qs, pf_scales, pf_gate, n as u32)?;
                gemm_rows(cuda, k, &layer.w_gate_up, hidden as u32, hidden as u32, pf_xn, pf_qs, pf_scales, pf_up, n as u32)?;
                k.silu_mul(cuda, pf_gate, pf_up, (n * hidden) as u32)?;
                k.quantize_q8(cuda, pf_gate, pf_qs, pf_scales, (n * hidden) as u32)?;
                gemm_rows(cuda, k, &layer.w_down, 0, dim as u32, pf_gate, pf_qs, pf_scales, pf_proj, n as u32)?;
                k.add(cuda, pf_x, pf_proj, (n * dim) as u32)?;
            }

            // Logits only for the final prompt token (last row of the last chunk).
            if base + n == p {
                let last = fq(pf_x, (n - 1) * dim);
                k.rms_norm(cuda, last, self.output_norm.dptr, single_xn, dim as u32, rms_eps)?;
                gemv_w(cuda, k, &self.ws, &self.output, single_xn, logits)?;
            }
            base += n;
        }
        Ok(())
    }

    /// Run one forward pass for `token` at position `pos` (direct execution,
    /// no graph — the prefill path). Advances the KV cursor.
    pub fn step(
        &mut self,
        cuda: &Cuda,
        k: &KernelSet,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<(), GlError> {
        if self.kv.is_full() {
            return Err(GlError::Engine(format!(
                "KV cache full ({} tokens) — context limit reached",
                self.kv.max_context
            )));
        }
        debug_assert_eq!(pos, self.kv.current_pos());
        self.set_token_inputs(cuda, token, pos)?;
        self.record_forward(cuda, k, want_logits)?;
        self.kv.advance();
        Ok(())
    }

    /// Decode one token via the captured graph (M2.2): update the device
    /// token inputs, replay the whole per-token kernel sequence in a single
    /// graph launch, advance the cursor. The graph is captured on first use.
    /// Always computes logits (decode needs them every token).
    pub fn decode_step(
        &mut self,
        cuda: &Cuda,
        k: &KernelSet,
        token: u32,
        pos: usize,
    ) -> Result<(), GlError> {
        if self.kv.is_full() {
            return Err(GlError::Engine(format!(
                "KV cache full ({} tokens) — context limit reached",
                self.kv.max_context
            )));
        }
        debug_assert_eq!(pos, self.kv.current_pos());
        self.set_token_inputs(cuda, token, pos)?;

        if self.graph.is_none() {
            // Capture the sequence once. record_forward reads pos/cached_len
            // from device memory, so the captured graph is valid for every
            // subsequent token.
            //
            // SAFETY of the borrow dance: capture() takes a closure that
            // only issues launches; we borrow &self inside it via a raw
            // pointer because the closure cannot also hold &mut self. The
            // launches touch only device memory owned by self and mutate no
            // Rust state.
            let this: *const GpuModel = self;
            let graph = cuda.capture(|| {
                // SAFETY: `this` outlives the capture call; record_forward
                // takes &self and does not alias the &mut borrow (no Rust
                // field is written).
                unsafe { (*this).record_forward(cuda, k, true) }
            })?;
            self.graph = Some(graph);
        }
        // Replay.
        let graph = self.graph.as_ref().expect("graph captured above");
        cuda.graph_launch(graph)?;
        self.kv.advance();
        Ok(())
    }

    /// Embedding row lookup into a caller buffer (host side).
    fn embed_row(&self, token: u32, out: &mut [f32]) -> Result<(), GlError> {
        let dim = self.config.dim;
        let row = token as usize;
        if row >= self.config.vocab_size {
            return Err(GlError::Engine(format!("token id {token} out of embedding range")));
        }
        match &self.token_embd {
            crate::model::HostWeight::F32(v) => {
                out.copy_from_slice(&v[row * dim..(row + 1) * dim])
            }
            crate::model::HostWeight::Q8_0(b) => crate::dequant::q8_0_row_into(b, row, dim, out),
            crate::model::HostWeight::Q8_0Soa { .. }
            | crate::model::HostWeight::Q4_0Soa { .. }
            | crate::model::HostWeight::Q4KSoa { .. }
            | crate::model::HostWeight::Q6KSoa { .. } => {
                unreachable!("embedding table is AoS, never SoA")
            }
            crate::model::HostWeight::Q4_0(b) => crate::dequant::q4_0_row_into(b, row, dim, out),
            crate::model::HostWeight::Q4K(b) => crate::dequant::q4_k_row_into(b, row, dim, out),
            crate::model::HostWeight::Q6K(b) => crate::dequant::q6_k_row_into(b, row, dim, out),
        }
        Ok(())
    }

    /// Synchronize the stream and download the logits of the most recent
    /// `step(.., want_logits = true)`.
    pub fn logits_host(&mut self, cuda: &Cuda) -> Result<&mut [f32], GlError> {
        cuda.synchronize()?;
        let mut host = std::mem::take(&mut self.ws.logits_host);
        let r = cuda.dtoh_f32(&mut host, self.ws.logits.dptr);
        self.ws.logits_host = host;
        r?;
        Ok(&mut self.ws.logits_host)
    }

    /// Generate up to `max_new_tokens` continuation tokens for `prompt` —
    /// the same contract, stop semantics and timing split as glproc's
    /// `Runner::generate` (including the pos-guarded decode loop shape,
    /// hence the counter-loop allow).
    #[allow(clippy::too_many_arguments, clippy::explicit_counter_loop)]
    pub fn generate(
        &mut self,
        cuda: &Cuda,
        k: &KernelSet,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut Sampler,
        is_stop: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, GenTiming), GlError> {
        if prompt.is_empty() {
            return Err(GlError::Engine("empty prompt".into()));
        }
        self.kv.reset();
        let max_seq = self.config.max_seq.min(self.kv.max_context);
        if prompt.len() > max_seq {
            return Err(GlError::Engine(format!(
                "prompt length {} exceeds context window {max_seq}",
                prompt.len()
            )));
        }

        // Prefill: process the whole prompt in batched passes so the weight
        // matmuls are batched GEMMs (weights streamed once per tile, not once
        // per token). Logits land for the last prompt token only.
        let prefill_start = Instant::now();
        self.prefill_batched(cuda, k, prompt)?;
        cuda.synchronize()?; // honest prefill timing: submission != done
        let prefill = prefill_start.elapsed();

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut recent: std::collections::VecDeque<u32> =
            std::collections::VecDeque::with_capacity(REPEAT_WINDOW);
        let mut pos = prompt.len();
        // Opt-in split timing: GLCUDA_PROFILE_DECODE=1 attributes each token's
        // wall time to GPU (the decode graph + a trailing sync so all kernel
        // work is captured regardless of whether graph_launch blocks) vs HOST
        // (logits DtoH + repetition penalty + CPU sample over the full vocab,
        // during which the GPU is idle). A large host share means GPU-side
        // kernel work is NOT the decode bottleneck.
        let profile = std::env::var_os("GLCUDA_PROFILE_DECODE").is_some();
        let (mut t_gpu, mut t_host) = (std::time::Duration::ZERO, std::time::Duration::ZERO);
        for _ in 0..max_new_tokens {
            if pos >= max_seq {
                break;
            }
            // HOST: the logits consumed here were produced by the previous
            // token's graph (or prefill), which we already synced below, so
            // logits_host's internal sync is a no-op and this is pure CPU.
            let h = Instant::now();
            let penalty = sampler.repeat_penalty();
            let next = {
                let logits = self.logits_host(cuda)?;
                apply_repetition_penalty(logits, recent.make_contiguous(), penalty);
                sampler.sample(logits)
            };
            if profile {
                t_host += h.elapsed();
            }
            if is_stop(next) {
                break;
            }
            on_token(next);
            generated.push(next);
            if recent.len() == REPEAT_WINDOW {
                recent.pop_front();
            }
            recent.push_back(next);
            // GPU: launch the decode graph and (in profile mode) sync so the
            // full kernel time lands in t_gpu even if graph_launch is async.
            let g = Instant::now();
            self.decode_step(cuda, k, next, pos)?;
            if profile {
                cuda.synchronize()?;
                t_gpu += g.elapsed();
            }
            pos += 1;
        }
        if profile {
            let n = generated.len().max(1) as f64;
            eprintln!(
                "[decode split] {} tokens | GPU {:.2} ms/tok | HOST {:.2} ms/tok | host share {:.0}%",
                generated.len(),
                t_gpu.as_secs_f64() * 1e3 / n,
                t_host.as_secs_f64() * 1e3 / n,
                100.0 * t_host.as_secs_f64() / (t_gpu.as_secs_f64() + t_host.as_secs_f64()).max(1e-9),
            );
        }
        Ok((
            generated,
            GenTiming {
                prompt_tokens: prompt.len(),
                prefill,
                decode: decode_start.elapsed(),
            },
        ))
    }
}
