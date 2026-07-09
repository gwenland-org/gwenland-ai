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
        GpuWeight::Q4_0(s) => k.gemv_q4_0(cuda, s.dptr, x, y, m.out_dim, m.in_dim),
    }
}

/// Device address `elems` f32 past `base`.
#[inline(always)]
fn at(base: CUdeviceptr, elems: usize) -> CUdeviceptr {
    base + (elems * 4) as u64
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
            crate::model::HostWeight::Q4_0(b) => crate::dequant::q4_0_row_into(b, row, dim, out),
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
