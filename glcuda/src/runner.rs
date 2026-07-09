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
use crate::model::{GpuMat, GpuModel, GpuWeight, RopeStyle};
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

/// GEMV over either weight representation.
fn gemv_w(
    cuda: &Cuda,
    k: &KernelSet,
    m: &GpuMat,
    x: CUdeviceptr,
    y: CUdeviceptr,
) -> Result<(), GlError> {
    match &m.w {
        GpuWeight::F32(s) => k.gemv(cuda, s.dptr, x, y, m.out_dim, m.in_dim),
        GpuWeight::Q8_0(s) => k.gemv_q8_0(cuda, s.dptr, x, y, m.out_dim, m.in_dim),
    }
}

/// Device address `elems` f32 past `base`.
#[inline(always)]
fn at(base: CUdeviceptr, elems: usize) -> CUdeviceptr {
    base + (elems * 4) as u64
}

impl GpuModel {
    /// Run one forward pass for `token` at position `pos`, leaving the
    /// logits in device memory (fetch via [`GpuModel::logits_host`]).
    /// Advances the KV cursor — call with strictly increasing `pos`.
    pub fn step(
        &mut self,
        cuda: &Cuda,
        k: &KernelSet,
        token: u32,
        pos: usize,
        want_logits: bool,
    ) -> Result<(), GlError> {
        let c = self.config.clone();
        let dim = c.dim as u32;
        let head_dim = c.head_dim;
        let q_dim = c.n_heads * head_dim;
        let kv_dim = c.n_kv_heads * head_dim;
        let heads_per_kv = c.n_heads / c.n_kv_heads.max(1);
        let half = head_dim / 2;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let neox = c.rope_style == RopeStyle::Neox;

        if self.kv.is_full() {
            return Err(GlError::Engine(format!(
                "KV cache full ({} tokens) — context limit reached",
                self.kv.max_context
            )));
        }
        debug_assert_eq!(pos, self.kv.current_pos());

        // Embedding on the host, one small HtoD (the residual stream x).
        {
            let mut embed = std::mem::take(&mut self.ws.embed_host);
            let r = self.embed_row(token, &mut embed);
            self.ws.embed_host = embed;
            r?;
        }
        cuda.htod_f32(self.ws.x.dptr, &self.ws.embed_host)?;

        let ws = &self.ws;
        let (x, xn) = (ws.x.dptr, ws.xn.dptr);
        let q_ptr = ws.qkv.dptr;
        let k_ptr = at(ws.qkv.dptr, q_dim);
        let v_ptr = at(ws.qkv.dptr, q_dim + kv_dim);
        let cos_pos = at(ws.rope_cos.dptr, pos * half);
        let sin_pos = at(ws.rope_sin.dptr, pos * half);

        let kv = &mut self.kv;
        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention block ---
            k.rms_norm(cuda, x, layer.attn_norm.dptr, xn, dim, c.rms_eps)?;
            gemv_w(cuda, k, &layer.wq, xn, q_ptr)?;
            gemv_w(cuda, k, &layer.wk, xn, k_ptr)?;
            gemv_w(cuda, k, &layer.wv, xn, v_ptr)?;

            if let Some(b) = &layer.bq {
                k.add(cuda, q_ptr, b.dptr, q_dim as u32)?;
            }
            if let Some(b) = &layer.bk {
                k.add(cuda, k_ptr, b.dptr, kv_dim as u32)?;
            }
            if let Some(b) = &layer.bv {
                k.add(cuda, v_ptr, b.dptr, kv_dim as u32)?;
            }

            // qwen3-style per-head RMSNorm on Q/K, before RoPE. In place is
            // safe: the kernel's reduction completes (barrier) before the
            // element-wise write, and each thread rewrites only its own
            // elements.
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

            // RoPE over all heads in one launch each (tables are position-
            // indexed device memory, uploaded once at model load).
            k.rope(cuda, q_ptr, cos_pos, sin_pos, c.n_heads as u32, head_dim as u32, neox)?;
            k.rope(cuda, k_ptr, cos_pos, sin_pos, c.n_kv_heads as u32, head_dim as u32, neox)?;

            for h in 0..c.n_kv_heads {
                kv.write_k(cuda, l, h, at(k_ptr, h * head_dim))?;
                kv.write_v(cuda, l, h, at(v_ptr, h * head_dim))?;
            }

            // Fused decode attention over ALL heads in one launch (M2.1):
            // one block per head does Q·K, scaled softmax and the weighted-V
            // sum in shared memory. Replaces the per-head triple of
            // gemv+softmax+gemv_t (3*n_heads launches -> 1), which was ~2/3
            // of the per-token launch count and the decode bottleneck.
            let cached_len = (kv.current_pos() + 1) as u32;
            k.attn_decode(
                cuda,
                q_ptr,
                kv.read_k(l, 0),
                kv.read_v(l, 0),
                ws.attn_out.dptr,
                c.n_heads as u32,
                head_dim as u32,
                cached_len,
                heads_per_kv.max(1) as u32,
                kv.head_stride() as u32,
                scale,
            )?;

            gemv_w(cuda, k, &layer.wo, ws.attn_out.dptr, ws.proj.dptr)?;
            k.add(cuda, x, ws.proj.dptr, dim)?;

            // --- SwiGLU feed-forward block ---
            k.rms_norm(cuda, x, layer.ffn_norm.dptr, xn, dim, c.rms_eps)?;
            gemv_w(cuda, k, &layer.w_gate, xn, ws.gate.dptr)?;
            gemv_w(cuda, k, &layer.w_up, xn, ws.up.dptr)?;
            k.silu_mul(cuda, ws.gate.dptr, ws.up.dptr, c.hidden_dim as u32)?;
            gemv_w(cuda, k, &layer.w_down, ws.gate.dptr, ws.proj.dptr)?;
            k.add(cuda, x, ws.proj.dptr, dim)?;
        }

        // All layers committed this token's K/V — advance the cursor once.
        kv.advance();

        if want_logits {
            k.rms_norm(cuda, x, self.output_norm.dptr, xn, dim, c.rms_eps)?;
            gemv_w(cuda, k, &self.output, xn, ws.logits.dptr)?;
        }
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

        // Prefill: one step per prompt token, logits only for the last —
        // the LM head is the single biggest GEMV, skip it where unneeded.
        let prefill_start = Instant::now();
        for (i, &tok) in prompt.iter().enumerate() {
            self.step(cuda, k, tok, i, i + 1 == prompt.len())?;
        }
        cuda.synchronize()?; // honest prefill timing: submission != done
        let prefill = prefill_start.elapsed();

        let decode_start = Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut recent: std::collections::VecDeque<u32> =
            std::collections::VecDeque::with_capacity(REPEAT_WINDOW);
        let mut pos = prompt.len();
        for _ in 0..max_new_tokens {
            if pos >= max_seq {
                break;
            }
            let penalty = sampler.repeat_penalty();
            let next = {
                let logits = self.logits_host(cuda)?;
                apply_repetition_penalty(logits, recent.make_contiguous(), penalty);
                sampler.sample(logits)
            };
            if is_stop(next) {
                break;
            }
            on_token(next);
            generated.push(next);
            if recent.len() == REPEAT_WINDOW {
                recent.pop_front();
            }
            recent.push_back(next);
            self.step(cuda, k, next, pos, true)?;
            pos += 1;
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
