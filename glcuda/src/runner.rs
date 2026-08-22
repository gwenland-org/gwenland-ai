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

/// Does this weight's GEMV read the int8 activation scratch
/// (`ws.q8_qs`/`ws.q8_scales`), or the f32 activation directly?
///
/// The match is exhaustive on purpose: a new [`GpuWeight`] variant must
/// answer this question, because getting it wrong means either a wasted
/// quantize or a GEMV reading a stale scratch buffer — and only the second
/// one is visible in the output.
fn consumes_q8_act(w: &GpuWeight) -> bool {
    match w {
        // Read `x` (f32) straight: no quantized activation involved.
        GpuWeight::F32(_) | GpuWeight::Q4_0(_) => false,
        GpuWeight::Q8_0(_)
        | GpuWeight::Q8_0Soa { .. }
        | GpuWeight::Q4_0Soa { .. }
        | GpuWeight::Q4KSoa { .. }
        | GpuWeight::Q6KSoa { .. } => true,
    }
}

/// Quantize `x` if this weight needs it, then run the GEMV.
///
/// For a weight whose input is shared with other GEMVs, hoist the quantize
/// to the caller and use [`gemv_w_pre`] instead — see the q/k/v trio in
/// [`GpuModel::record_forward`].
fn gemv_w(
    cuda: &Cuda,
    k: &KernelSet,
    ws: &crate::model::Workspace,
    m: &GpuMat,
    x: CUdeviceptr,
    y: CUdeviceptr,
) -> Result<(), GlError> {
    if consumes_q8_act(&m.w) {
        k.quantize_q8(cuda, x, ws.q8_qs.dptr, ws.q8_scales.dptr, m.in_dim)?;
    }
    gemv_w_pre(cuda, k, ws, m, x, y)
}

/// [`gemv_w`] without the quantize step.
///
/// Precondition when [`consumes_q8_act`] is true for `m`: the caller has
/// already quantized this GEMV's input into `ws.q8_qs`/`ws.q8_scales` for
/// exactly `m.in_dim` elements. Weights that read `x` directly ignore the
/// scratch and are safe to call either way.
fn gemv_w_pre(
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
            k.gemv_q8_0(cuda, s.dptr, ws.q8_qs.dptr, ws.q8_scales.dptr, y, m.out_dim, m.in_dim)
        }
        GpuWeight::Q8_0Soa { qs, scales } => {
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

/// Does the 256-row GEMM read the weights fewer times than the 64-row one
/// for `n` token rows?
///
/// This is the whole of r256's advantage, so it is the whole of the decision.
/// Weight fragments are read once per slab, so the counts are `ceil(n/64)` and
/// `ceil(n/256)`; a tie means r256 offers nothing and loses on shared memory
/// (9216 bytes against 2304), which costs blocks per SM.
///
/// Deliberately no occupancy term. Any coefficient for that would be invented
/// rather than measured; if a production A/B shows r256 losing somewhere above
/// 64, that measurement earns the term.
fn r256_pays(n: u32) -> bool {
    n.div_ceil(256) < n.div_ceil(64)
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
                // ---- Which MMA GEMM, and in what slab size ----
                //
                // r256 exists to do exactly one thing: read each weight
                // fragment once per 256 token rows instead of once per 64. So
                // the choice is made on that quantity, computed from n at the
                // call site rather than from a tuned constant:
                //
                //     reads_64 = ceil(n/64)    reads_256 = ceil(n/256)
                //
                // Take r256 only when it strictly reduces that. At n <= 64
                // they tie -- both read the weights once -- and a tie goes to
                // the 64-row kernel, which uses 2304 bytes of shared memory
                // against r256's 9216 and therefore fits more blocks per SM.
                // The familiar "threshold of 64" falls out of the arithmetic
                // instead of being typed in.
                //
                // What this deliberately does NOT model is occupancy. Any
                // coefficient for that would be a guess wearing arithmetic's
                // clothes; if the production A/B shows r256 losing in some
                // band above 64, that measurement is what earns a cost term.
                //
                // r256 is opt-in (GLCUDA_R256). It is measured correct on a T4
                // -- parity green, max_abs_diff 0.00e0 against gemm_mma_q8 at
                // real shapes -- and measured 31% faster at 512-row chunks,
                // after a staging-register bug that made it compute the wrong
                // answer for months. The last wire-in attempt crashed with
                // CUDA_ERROR_MISALIGNED_ADDRESS, which that same bug explains
                // (a clobbered row index becomes a global address) but does
                // not prove fixed until a production run says so.
                //
                // Note the interaction with GLCUDA_MULTI_STREAM_PREFILL: at
                // n = 220 r256 issues a single slab, so there is nothing left
                // for the stream pool to overlap. The two switches are
                // alternatives, not additions.
                let use_r256 = k.r256_enabled() && r256_pays(n);
                let slab_rows = if use_r256 { 256 } else { 64 };

                // The sub-slabs below are independent: same weights, disjoint
                // activation rows, disjoint output. They are sequential only
                // because they share a stream. With
                // GLCUDA_MULTI_STREAM_PREFILL set they are issued across a
                // stream pool instead, putting one sub-slab's worth of blocks
                // in flight per stream.
                //
                // This matters most where the grid is smallest. The grid is
                // ceil_div(out_dim, 64) and nothing splits K or tokens, so on
                // a 40-SM T4 `down` gets 14 blocks -- and a measured profile
                // put down+o at 66% of prefill while gate+up, moving twice the
                // weight bytes, took 10%.
                //
                // Sync is a host round-trip per stream, not an event: blunt,
                // but it answers whether the idea is worth the machinery
                // before the machinery gets built.
                let pool = cuda.prefill_streams();
                let mut t0 = 0u32;
                let mut slab = 0usize;
                while t0 < n {
                    let nn = (n - t0).min(slab_rows);
                    let issue = || {
                        let gemm = if use_r256 {
                            KernelSet::gemm_mma_q8_r256
                        } else {
                            KernelSet::gemm_mma_q8
                        };
                        gemm(
                            k,
                            cuda,
                            wqs,
                            wsc,
                            x_qs + (t0 * inb) as u64,
                            x_scales + (t0 * (inb / 32)) as u64 * 4,
                            y + (t0 * rows) as u64 * 4,
                            rows,
                            inb,
                            nn,
                        )
                    };
                    match pool {
                        Some(p) => cuda.on_stream(p, slab, issue)?,
                        None => issue()?,
                    }
                    t0 += nn;
                    slab += 1;
                }
                // Only when work was actually spread: a single sub-slab on one
                // pool stream still has to be waited for, so the guard is on
                // whether a pool was used at all, not on the slab count.
                if let Some(p) = pool {
                    cuda.sync_pool(p)?;
                }
                Ok(())
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

            // q/k/v read the SAME normalized activation `xn` over the same
            // in_dim, so the int8 copy is made once and all three GEMVs share
            // it. Letting each `gemv_w` quantize for itself wrote the same
            // bytes into the same scratch three times per layer.
            //
            // Prefill has always done it this way (see `prefill_batched`:
            // one `quantize_q8`, then three `gemm_rows`); decode simply never
            // followed. Same math either way — the dropped launches were
            // recomputing a value that was already there.
            debug_assert!(
                layer.wq.in_dim == dim && layer.wk.in_dim == dim && layer.wv.in_dim == dim,
                "q/k/v must share in_dim with the shared quantize below"
            );
            if consumes_q8_act(&layer.wq.w)
                || consumes_q8_act(&layer.wk.w)
                || consumes_q8_act(&layer.wv.w)
            {
                k.quantize_q8(cuda, xn, ws.q8_qs.dptr, ws.q8_scales.dptr, dim)?;
            }
            gemv_w_pre(cuda, k, ws, &layer.wq, xn, q_ptr)?;
            gemv_w_pre(cuda, k, ws, &layer.wk, xn, k_ptr)?;
            gemv_w_pre(cuda, k, ws, &layer.wv, xn, v_ptr)?;

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
            //
            // Q is [n_heads, head_dim] contiguous and K is [n_kv_heads,
            // head_dim] contiguous, which is exactly `rms_norm_rows`'s input
            // shape — one block per head instead of one LAUNCH per head. The
            // norm weight is shared by every head, and the kernel does not
            // offset `w` by the row, so it broadcasts as required.
            //
            // Bit-exact with the loop it replaces: `gl_rms_norm_rows_f32` is
            // instruction-for-instruction `gl_rms_norm_f32` plus the row-base
            // computation, so the reduction order and rounding are unchanged.
            //
            // On Qwen3-1.7B this drops 24 single-block launches per layer
            // (672 per token, over half of all decode launches), each of
            // which occupied 1 SM of 40.
            if let Some(qn) = &layer.q_norm {
                k.rms_norm_rows(
                    cuda, q_ptr, qn.dptr, q_ptr, head_dim as u32, c.rms_eps, c.n_heads as u32,
                )?;
            }
            if let Some(kn) = &layer.k_norm {
                k.rms_norm_rows(
                    cuda, k_ptr, kn.dptr, k_ptr, head_dim as u32, c.rms_eps, c.n_kv_heads as u32,
                )?;
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

    /// Batched prefill: run the whole prompt through the model with up to
    /// `PREFILL_BATCH` (512) tokens RESIDENT per pass — the layer-first
    /// execution graph (Acceleratio Stellarum Phase A): prompts up to 512
    /// tokens traverse the layer loop once with every row available to each
    /// weight's GEMM. Causality is per-row inside the attention kernel
    /// (`cached_len = pos_seq[t] + 1`), so the schedule change does not touch
    /// the math. Until the Phase B GEMM contract lands, `gemm_rows` still
    /// issues the tensor-core GEMM in 64-row sub-slabs, so weight traffic is
    /// unchanged in Phase A by design. Leaves the last prompt token's logits
    /// in `ws.logits` and advances the KV cursor to `prompt.len()`.
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
        let pos_seq = ws.pos_seq.dptr;
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

        // Opt-in per-phase GPU timing (GLCUDA_PROFILE_PREFILL=1). Each phase
        // syncs and accumulates wall time into a bucket, so the split is
        // exact at the cost of serializing the pipeline — diagnostic only,
        // never on in production. Buckets: qkv (norm+quant+Q/K/V GEMMs),
        // attn (bias/qk-norm/rope/kv-write/attention core), ffn (norm+quant+
        // gate/up/down GEMMs+silu+residual).
        let prof = std::env::var_os("GLCUDA_PROFILE_PREFILL").is_some();
        // t_attn/t_ffn are recomputed from their sub-buckets below; only t_qkv
        // is still accumulated directly in the loop.
        let (mut t_qkv, t_attn, t_ffn) =
            (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
        // Fine-grained FFN sub-buckets (only meaningful with the profiler on):
        // gate+up GEMMs, down GEMM, and the elementwise glue (quant/silu/
        // norm/add) — to localize the 51-67% FFN cost the coarse split shows.
        let (mut t_gu, mut t_dn, mut t_elt) =
            (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
        // Fine-grained attn sub-buckets: profiling showed attn is ~40% of
        // prefill and, unlike FFN, contains no big GEMM — so localize it into
        // norm (bias+qk-norm+rope), kv-write, and the attention core. t_attn is
        // their sum, so these inner phase! calls replace the outer wrapper (no
        // double-counting). "norm" groups the pre-core elementwise/rope glue;
        // "core" is attn_decode_rows, the decode-shaped kernel run over prefill
        // rows and the prime suspect for the disproportionate attn cost.
        let (mut t_an, mut t_kv, mut t_ac) =
            (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
        macro_rules! phase {
            ($bucket:expr, $body:block) => {{
                if prof {
                    cuda.synchronize()?;
                    let _t = Instant::now();
                    $body
                    cuda.synchronize()?;
                    $bucket += _t.elapsed();
                } else {
                    $body
                }
            }};
        }

        let mut base = 0usize;
        while base < p {
            let n = (p - base).min(PREFILL_BATCH);

            // Embed this chunk's tokens host-side, then ONE HtoD for the
            // whole chunk (Phase A: the old per-token copies were n small
            // synchronous transfers). The staging vec lives in the workspace
            // so prefill stays allocation-free across chunks.
            {
                let mut staging = std::mem::take(&mut self.ws.pf_embed_host);
                let mut embed_err = Ok(());
                for i in 0..n {
                    let row = &mut staging[i * dim..(i + 1) * dim];
                    if let Err(e) = self.embed_row(prompt[base + i], row) {
                        embed_err = Err(e);
                        break;
                    }
                }
                // Return the vec before bailing so the workspace keeps it.
                let upload = cuda.htod_f32(pf_x, &staging[..n * dim]);
                self.ws.pf_embed_host = staging;
                embed_err?;
                upload?;
            }

            // Positions are consecutive integers: element i of the pos_seq
            // identity array (offset to the chunk base) IS row i's position,
            // and cached_len = pos+1 is the next element. One device array,
            // uploaded once at load — no HtoD anywhere in this loop (the old
            // per-token token_params copy was ~896 synchronous, pipeline-
            // draining copies per chunk and the top prefill cost).
            let pos_base = pos_seq + (base * 4) as u64;

            for l in 0..self.layers.len() {
                let layer = &self.layers[l];

                // --- attention block (M2.3: every per-token op is ONE
                // batched launch over the chunk's rows) ---
                phase!(t_qkv, {
                    k.rms_norm_rows(cuda, pf_x, layer.attn_norm.dptr, pf_xn, dim as u32, rms_eps, n as u32)?;
                    k.quantize_q8(cuda, pf_xn, pf_qs, pf_scales, (n * dim) as u32)?;
                    gemm_rows(cuda, k, &layer.wq, 0, q_dim as u32, pf_xn, pf_qs, pf_scales, pf_q, n as u32)?;
                    gemm_rows(cuda, k, &layer.wk, 0, kv_dim as u32, pf_xn, pf_qs, pf_scales, pf_k, n as u32)?;
                    gemm_rows(cuda, k, &layer.wv, 0, kv_dim as u32, pf_xn, pf_qs, pf_scales, pf_v, n as u32)?;
                });

                // attn, split into norm (bias+qk-norm+rope) / kv-write / core.
                // t_attn is their sum, reported below — no outer phase! wrapper,
                // so the inner syncs don't double-count.
                phase!(t_an, {
                    if let Some(b) = &layer.bq {
                        k.add_bias_rows(cuda, pf_q, b.dptr, q_dim as u32, (n * q_dim) as u32)?;
                    }
                    if let Some(b) = &layer.bk {
                        k.add_bias_rows(cuda, pf_k, b.dptr, kv_dim as u32, (n * kv_dim) as u32)?;
                    }
                    if let Some(b) = &layer.bv {
                        k.add_bias_rows(cuda, pf_v, b.dptr, kv_dim as u32, (n * kv_dim) as u32)?;
                    }
                    // Per-head q/k norms: a [n, heads*head_dim] block is exactly
                    // n*heads contiguous rows of head_dim.
                    if let Some(qn) = &layer.q_norm {
                        k.rms_norm_rows(cuda, pf_q, qn.dptr, pf_q, head_dim as u32, rms_eps, (n * n_heads) as u32)?;
                    }
                    if let Some(kn) = &layer.k_norm {
                        k.rms_norm_rows(cuda, pf_k, kn.dptr, pf_k, head_dim as u32, rms_eps, (n * n_kv_heads) as u32)?;
                    }
                    k.rope_rows(cuda, pf_q, rope_cos, rope_sin, n_heads as u32, head_dim as u32, neox, pos_base, n as u32)?;
                    k.rope_rows(cuda, pf_k, rope_cos, rope_sin, n_kv_heads as u32, head_dim as u32, neox, pos_base, n as u32)?;
                });
                phase!(t_kv, {
                    k.kv_write_rows(cuda, self.kv.read_k(l, 0), pf_k, pos_base, head_dim as u32, n_kv_heads as u32, head_stride, n as u32)?;
                    k.kv_write_rows(cuda, self.kv.read_v(l, 0), pf_v, pos_base, head_dim as u32, n_kv_heads as u32, head_stride, n as u32)?;
                });
                phase!(t_ac, {
                    // Causal by construction: row t reads cached_len = pos+1
                    // rows, so later rows (already written above) are never seen.
                    k.attn_decode_rows(
                        cuda, pf_q, self.kv.read_k(l, 0), self.kv.read_v(l, 0), pf_attn,
                        n_heads as u32, head_dim as u32, pos_base, heads_per_kv, head_stride, scale,
                        n as u32,
                    )?;
                });

                // FFN, split into GEMM sub-buckets (t_gu / t_dn) vs the
                // elementwise glue (t_elt). t_ffn is their sum, reported
                // below — no outer phase! wrapper, so the inner syncs don't
                // double-count. wo (o-proj) GEMM is grouped into t_dn.
                phase!(t_elt, {
                    k.quantize_q8(cuda, pf_attn, pf_qs, pf_scales, (n * q_dim) as u32)?;
                });
                phase!(t_dn, {
                    gemm_rows(cuda, k, &layer.wo, 0, dim as u32, pf_attn, pf_qs, pf_scales, pf_proj, n as u32)?;
                });
                phase!(t_elt, {
                    k.add(cuda, pf_x, pf_proj, (n * dim) as u32)?;
                    k.rms_norm_rows(cuda, pf_x, layer.ffn_norm.dptr, pf_xn, dim as u32, rms_eps, n as u32)?;
                    k.quantize_q8(cuda, pf_xn, pf_qs, pf_scales, (n * dim) as u32)?;
                });
                phase!(t_gu, {
                    gemm_rows(cuda, k, &layer.w_gate_up, 0, hidden as u32, pf_xn, pf_qs, pf_scales, pf_gate, n as u32)?;
                    gemm_rows(cuda, k, &layer.w_gate_up, hidden as u32, hidden as u32, pf_xn, pf_qs, pf_scales, pf_up, n as u32)?;
                });
                phase!(t_elt, {
                    k.silu_mul(cuda, pf_gate, pf_up, (n * hidden) as u32)?;
                    k.quantize_q8(cuda, pf_gate, pf_qs, pf_scales, (n * hidden) as u32)?;
                });
                phase!(t_dn, {
                    gemm_rows(cuda, k, &layer.w_down, 0, dim as u32, pf_gate, pf_qs, pf_scales, pf_proj, n as u32)?;
                });
                phase!(t_elt, {
                    k.add(cuda, pf_x, pf_proj, (n * dim) as u32)?;
                });
            }

            // Commit the chunk: ONE advance per token (the cursor contract).
            // The old code advanced inside the layer loop — n * n_layers per
            // chunk — which overcounted current_pos 28x and would falsely
            // report "KV cache full" on any prompt longer than
            // max_context / n_layers (146 tokens on the 7B).
            for _ in 0..n {
                self.kv.advance();
            }

            // Logits only for the final prompt token (last row of the last chunk).
            if base + n == p {
                let last = fq(pf_x, (n - 1) * dim);
                k.rms_norm(cuda, last, self.output_norm.dptr, single_xn, dim as u32, rms_eps)?;
                gemv_w(cuda, k, &self.ws, &self.output, single_xn, logits)?;
            }
            base += n;
        }
        if prof {
            let _ = t_attn; // superseded by the t_an/t_kv/t_ac sub-buckets
            let _ = t_ffn; // superseded by the t_gu/t_dn/t_elt sub-buckets
            let t_attn = t_an + t_kv + t_ac;
            let t_ffn = t_gu + t_dn + t_elt;
            let tot = (t_qkv + t_attn + t_ffn).as_secs_f64().max(1e-9);
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3;
            let pc = |d: std::time::Duration| 100.0 * d.as_secs_f64() / tot;
            eprintln!(
                "[prefill split] {p} tok | qkv {:.0}ms ({:.0}%) | attn {:.0}ms ({:.0}%) | ffn {:.0}ms ({:.0}%)",
                ms(t_qkv), pc(t_qkv), ms(t_attn), pc(t_attn), ms(t_ffn), pc(t_ffn),
            );
            eprintln!(
                "[attn detail]  norm+rope {:.0}ms ({:.0}%) | kv-write {:.0}ms ({:.0}%) | attn core {:.0}ms ({:.0}%)",
                ms(t_an), pc(t_an), ms(t_kv), pc(t_kv), ms(t_ac), pc(t_ac),
            );
            eprintln!(
                "[ffn detail]   gate+up GEMM {:.0}ms ({:.0}%) | down+o GEMM {:.0}ms ({:.0}%) | elementwise {:.0}ms ({:.0}%)",
                ms(t_gu), pc(t_gu), ms(t_dn), pc(t_dn), ms(t_elt), pc(t_elt),
            );
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

        // Degraded mode: a driver without the CUDA Graph API (pre-CUDA 10)
        // runs the identical kernel sequence launch-by-launch. Same math,
        // same order, same buffers — only the per-launch host overhead the
        // graph exists to remove comes back. This path is also the
        // correctness reference the graph is validated against, so it must
        // stay wired even though every supported device today takes the
        // branch below.
        if !cuda.graphs_available() {
            self.record_forward(cuda, k, true)?;
            self.kv.advance();
            return Ok(());
        }

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

#[cfg(test)]
mod tests {
    use super::{consumes_q8_act, r256_pays};
    use crate::buffer::DevSlice;
    use crate::model::GpuWeight;

    /// A stand-in device region. `consumes_q8_act` matches on the variant and
    /// never dereferences, so no GPU and no real allocation are involved.
    fn slice() -> DevSlice {
        DevSlice { dptr: 0, bytes: 0 }
    }

    /// Hoisting the q/k/v quantize out of `gemv_w` made this predicate the
    /// thing that decides whether a GEMV gets a fresh int8 activation. A
    /// variant wrongly answering `false` reads whatever the previous layer
    /// left in the scratch -- wrong numbers, no crash, no failing launch.
    /// So each variant is pinned to the buffer its kernel actually reads.
    #[test]
    fn q8_scratch_consumers_are_exactly_the_quantized_gemvs() {
        // Take the f32 activation `x` directly: quantizing for these would be
        // pure waste, and skipping it is always safe.
        assert!(!consumes_q8_act(&GpuWeight::F32(slice())));
        assert!(!consumes_q8_act(&GpuWeight::Q4_0(slice())));

        // Read ws.q8_qs / ws.q8_scales: these REQUIRE a caller-side quantize.
        assert!(consumes_q8_act(&GpuWeight::Q8_0(slice())));
        assert!(consumes_q8_act(&GpuWeight::Q8_0Soa { qs: slice(), scales: slice() }));
        assert!(consumes_q8_act(&GpuWeight::Q4_0Soa { qs: slice(), scales: slice() }));
        assert!(consumes_q8_act(&GpuWeight::Q4KSoa {
            qs: slice(),
            scales: slice(),
            mins: slice(),
        }));
        assert!(consumes_q8_act(&GpuWeight::Q6KSoa {
            ql: slice(),
            qh: slice(),
            scales: slice(),
            d: slice(),
        }));
    }

    /// The hoist is only valid because ONE quantize serves all three GEMVs,
    /// which holds only while any of them needing it means the shared copy
    /// gets made. Mixed-precision q/k/v is not a shape the loader produces
    /// today, but the guard costs nothing and the failure would be silent.
    #[test]
    fn a_single_quantized_projection_still_triggers_the_shared_quantize() {
        let f32_w = GpuWeight::F32(slice());
        let q8_w = GpuWeight::Q8_0Soa { qs: slice(), scales: slice() };
        assert!(
            consumes_q8_act(&f32_w) || consumes_q8_act(&q8_w),
            "one quantized projection among three must still make the copy"
        );
        assert!(
            !(consumes_q8_act(&GpuWeight::F32(slice()))
                || consumes_q8_act(&GpuWeight::Q4_0(slice()))),
            "an all-f32 trio must skip the quantize entirely"
        );
    }

    /// The rule is arithmetic, so it is checked as arithmetic -- no GPU, and
    /// no chance of the "threshold of 64" drifting away from the reason for it.
    #[test]
    fn r256_only_pays_when_it_saves_a_weight_read() {
        // Ties: both kernels read the weights exactly once. r256 adds 4x the
        // shared memory for nothing.
        for n in [1u32, 8, 63, 64] {
            assert!(!r256_pays(n), "n={n}: one slab either way, r256 buys nothing");
        }
        // 65..=256 is one r256 slab against two, three or four 64-row slabs.
        for n in [65u32, 128, 192, 220, 256] {
            assert!(r256_pays(n), "n={n}: r256 reads the weights once, 64-row several times");
        }
        // Above 256 the ratio narrows but never inverts.
        for n in [257u32, 384, 512] {
            assert!(r256_pays(n), "n={n}");
        }
    }

    /// The exact figures the rule turns on, so a change to either divisor
    /// fails here rather than silently altering dispatch.
    #[test]
    fn weight_read_counts_are_what_the_rule_compares() {
        let reads = |n: u32, slab: u32| n.div_ceil(slab);
        assert_eq!((reads(220, 64), reads(220, 256)), (4, 1)); // the prefill case
        assert_eq!((reads(64, 64), reads(64, 256)), (1, 1));   // the tie
        assert_eq!((reads(512, 64), reads(512, 256)), (8, 2)); // a full chunk
    }

    #[test]
    fn zero_rows_is_a_tie_not_a_panic() {
        assert!(!r256_pays(0));
    }
}
