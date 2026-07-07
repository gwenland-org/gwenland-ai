//! Autoregressive generation loop: the transformer forward pass, one token
//! at a time, with KV caching.
//!
//! M1.5 hot-path rules enforced here:
//! * zero heap allocation per decoded token — every buffer lives in the
//!   pre-allocated [`Workspace`] and the cursor-based [`KvCache`]
//! * no `dyn` dispatch — the SIMD backend is a `match` on a cached enum
//! * matvecs run on the persistent [`ThreadPool`], rows interleaved
//! * Q4_K weights stay quantized and go through the Bridge-ing pipeline

use glcore::GlError;

use crate::attention::attention_one_into;
use crate::kernels;
use crate::kernels::bridge::bridge_matvec_quant;
use crate::kernels::qdot::{self, QuantizedActivation};
use crate::kv_cache::KvCache;
use crate::model::{GateUp, GlprocModel, QkvWeights, RopeStyle, WeightMatrix};
use crate::sampler::Sampler;
use crate::simd_strategy::SimdStrategy;
use crate::threading::{
    par_matvec, par_matvec_qdot, par_matvec_quant, par_matvec_swiglu, ThreadPool,
};

/// KV cache sequence capacity cap. Qwen-class models advertise 32k context;
/// pre-allocating that costs GBs, so the cache is sized to
/// `min(model max_seq, this)`. ~200 MB for Qwen2.5-0.5B dims.
const MAX_KV_CONTEXT: usize = 4096;

/// Decode threads. 4 matches the i3-1115G4 (4 logical cores); capped by the
/// machine's actual core count so small VMs don't oversubscribe.
/// `GLPROC_THREADS` overrides — for benchmarking and as the X5 thermal
/// knob (reduce thread count if the CPU runs hot).
const N_THREADS: usize = 4;

/// Thread count after the optional `GLPROC_THREADS` override.
fn n_threads() -> usize {
    std::env::var("GLPROC_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(N_THREADS)
        .min(num_cpus::get())
        .max(1)
}

/// Below this many multiply-accumulates a matvec runs on the calling thread —
/// waking workers costs more than the work itself.
const PAR_MIN_WORK: usize = 1 << 16;

/// Apply rotary position embeddings in place to one head's vector.
fn rope(x: &mut [f32], pos: usize, head_dim: usize, freq_base: f32, style: RopeStyle) {
    let half = head_dim / 2;
    for i in 0..half {
        let freq = 1.0 / freq_base.powf(2.0 * i as f32 / head_dim as f32);
        let theta = pos as f32 * freq;
        let (sin, cos) = theta.sin_cos();
        let (a, b) = match style {
            RopeStyle::Norm => (2 * i, 2 * i + 1),
            RopeStyle::Neox => (i, i + half),
        };
        let x0 = x[a];
        let x1 = x[b];
        x[a] = x0 * cos - x1 * sin;
        x[b] = x0 * sin + x1 * cos;
    }
}

/// True when `w` is consumed through the integer-dot path, i.e. the caller
/// must quantize the activation into the workspace first.
fn needs_q8(w: &WeightMatrix) -> bool {
    matches!(w, WeightMatrix::Quant(fmt, _) if qdot::supports(*fmt))
}

/// Matvec over either weight representation, threaded when the work is big
/// enough to amortize waking the pool. Quantized formats with an integer
/// kernel use `act` — the caller must have quantized the current `x` into
/// it (once per distinct vector, even when several matrices consume it);
/// the rest go through the f32 bridge.
fn matvec_w(
    pool: &ThreadPool,
    strategy: SimdStrategy,
    w: &WeightMatrix,
    x: &[f32],
    act: &QuantizedActivation,
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
) {
    let parallel = out_dim * in_dim >= PAR_MIN_WORK && pool.n_threads() > 1;
    match w {
        WeightMatrix::F32(data) => {
            if parallel {
                par_matvec(pool, data, x, y, out_dim, in_dim, strategy);
            } else {
                kernels::matvec(data, x, y, out_dim, in_dim);
            }
        }
        WeightMatrix::Quant(fmt, blocks) => {
            if qdot::supports(*fmt) {
                debug_assert_eq!(act.len, in_dim, "caller must quantize x into act first");
                if parallel {
                    par_matvec_qdot(pool, *fmt, blocks, act, y, out_dim, in_dim, strategy);
                } else {
                    let row_bytes = in_dim / fmt.block_numel() * fmt.block_bytes();
                    for (o, out) in y.iter_mut().enumerate() {
                        let row = &blocks[o * row_bytes..(o + 1) * row_bytes];
                        *out = qdot::row_dot_q8(*fmt, row, act, strategy);
                    }
                }
            } else if parallel {
                par_matvec_quant(pool, *fmt, blocks, x, y, out_dim, in_dim, strategy);
            } else {
                bridge_matvec_quant(*fmt, blocks, x, y, out_dim, in_dim, strategy);
            }
        }
    }
}

/// Every buffer the forward pass writes to, allocated once at Runner
/// construction and reused for each token (Rule 6: zero alloc in decode loop).
struct Workspace {
    /// Residual stream, `[dim]`.
    x: Vec<f32>,
    /// RMSNorm output, `[dim]`.
    xn: Vec<f32>,
    /// Q, K and V vectors in one buffer, `[q_dim + 2 * kv_dim]` — the
    /// fused QKV matvec writes all three in a single dispatch.
    qkv: Vec<f32>,
    /// Per-head RMSNorm scratch (qwen3 q/k norm), `[head_dim]`.
    head: Vec<f32>,
    /// Attention output, `[n_heads * head_dim]`.
    attn_out: Vec<f32>,
    /// Attention/FFN projection back to the residual, `[dim]`.
    proj: Vec<f32>,
    /// SwiGLU gate, `[hidden_dim]`.
    gate: Vec<f32>,
    /// SwiGLU up, `[hidden_dim]`.
    up: Vec<f32>,
    /// Attention score scratch, `[kv capacity]`.
    scores: Vec<f32>,
    /// Output logits, `[vocab_size]`.
    logits: Vec<f32>,
    /// Q8-quantized activation for the integer-domain matvec path, sized
    /// for the widest input this model feeds a quantized matrix.
    act: QuantizedActivation,
}

/// Per-phase wall-time accumulators for the decode loop, enabled by setting
/// `GLPROC_PROFILE=1`. A measurement tool for finding the fat — when
/// disabled (`None`) the decode loop takes zero timestamps.
#[derive(Default)]
struct Prof {
    /// Attention-norm + Q/K/V matvecs + biases + head norms + RoPE + cache.
    qkv: std::time::Duration,
    /// Per-head single-query attention against the KV cache.
    attn: std::time::Duration,
    /// Output projection + FFN norm + gate/up/SiLU/down + residuals.
    ffn: std::time::Duration,
    /// Final norm + LM-head matvec over the full vocabulary.
    lm_head: std::time::Duration,
    /// Token sampling (measured in `generate`).
    sampler: std::time::Duration,
    tokens: u32,
}

impl Prof {
    fn report(&self) {
        let toks = self.tokens.max(1);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / toks as f64;
        eprintln!(
            "[profile] per token over {} tokens: qkv {:.2}ms | attn {:.2}ms | \
             ffn {:.2}ms | lm_head {:.2}ms | sampler {:.2}ms",
            self.tokens,
            ms(self.qkv),
            ms(self.attn),
            ms(self.ffn),
            ms(self.lm_head),
            ms(self.sampler),
        );
    }
}

/// Drives token-by-token inference over a loaded model.
pub struct Runner<'m> {
    model: &'m GlprocModel,
    cache: KvCache,
    pool: ThreadPool,
    strategy: SimdStrategy,
    ws: Workspace,
    /// `Some` only when `GLPROC_PROFILE` is set in the environment.
    prof: Option<Box<Prof>>,
}

impl<'m> Runner<'m> {
    /// Create a runner: fresh cursor-based KV cache, persistent thread pool,
    /// pre-allocated workspace. No further allocation happens per token.
    pub fn new(model: &'m GlprocModel) -> Self {
        let c = &model.config;
        let kv_capacity = c.max_seq.min(MAX_KV_CONTEXT).max(1);
        let q_dim = c.n_heads * c.head_dim;
        let kv_dim = c.n_kv_heads * c.head_dim;
        Runner {
            model,
            cache: KvCache::new(c.n_layers, c.n_kv_heads, c.head_dim, kv_capacity),
            pool: ThreadPool::new(n_threads()),
            strategy: SimdStrategy::detect(),
            ws: Workspace {
                x: vec![0.0; c.dim],
                xn: vec![0.0; c.dim],
                qkv: vec![0.0; q_dim + 2 * kv_dim],
                head: vec![0.0; c.head_dim],
                attn_out: vec![0.0; q_dim],
                proj: vec![0.0; c.dim],
                gate: vec![0.0; c.hidden_dim],
                up: vec![0.0; c.hidden_dim],
                scores: vec![0.0; kv_capacity],
                logits: vec![0.0; c.vocab_size],
                act: QuantizedActivation::with_capacity(
                    c.dim.max(c.hidden_dim).max(q_dim),
                ),
            },
            prof: std::env::var("GLPROC_PROFILE")
                .ok()
                .filter(|v| !v.is_empty() && v != "0")
                .map(|_| Box::new(Prof::default())),
        }
    }

    /// Run one forward pass for `token` at position `pos`, leaving the
    /// logits in the workspace (borrow them via [`Runner::logits`]).
    /// Advances the KV cursor — call with strictly increasing `pos`.
    pub fn forward_into(&mut self, token: u32, pos: usize) -> Result<(), GlError> {
        self.step(token, pos, true)
    }

    /// Forward pass with an optional LM head. Prefill only needs the KV
    /// cache side effects — the head is the single biggest matvec (full
    /// vocabulary), so skipping it for all but the last prompt token saves
    /// its full cost per prefill position.
    fn step(&mut self, token: u32, pos: usize, want_logits: bool) -> Result<(), GlError> {
        let c = &self.model.config;
        let dim = c.dim;
        let head_dim = c.head_dim;
        let q_dim = c.n_heads * head_dim;
        let kv_dim = c.n_kv_heads * head_dim;
        let heads_per_kv = c.n_heads / c.n_kv_heads.max(1);

        if self.cache.is_full() {
            return Err(GlError::Engine(format!(
                "KV cache full ({} tokens) — context limit reached",
                self.cache.max_context
            )));
        }
        debug_assert_eq!(pos, self.cache.current_pos());

        let start = (token as usize)
            .checked_mul(dim)
            .filter(|&s| s + dim <= self.model.token_embd.len())
            .ok_or_else(|| {
                GlError::Engine(format!("token id {token} out of embedding range"))
            })?;
        let ws = &mut self.ws;
        ws.x.copy_from_slice(&self.model.token_embd[start..start + dim]);

        // One timestamp per phase boundary, only when profiling.
        let mut t = self.prof.as_ref().map(|_| std::time::Instant::now());
        let mut lap = |p: &mut Option<Box<Prof>>, pick: fn(&mut Prof) -> &mut std::time::Duration| {
            if let Some(p) = p.as_deref_mut() {
                let now = std::time::Instant::now();
                *pick(p) += now - t.unwrap();
                t = Some(now);
            }
        };

        for (l, layer) in self.model.layers.iter().enumerate() {
            // --- attention block ---
            kernels::rms_norm_into(&ws.x, &layer.attn_norm, c.rms_eps, &mut ws.xn);

            match &layer.qkv {
                // Fused QKV: all three projections in one dispatch over one
                // contiguous weight stream, written into one buffer.
                QkvWeights::FusedQuant(fmt, packed) => {
                    ws.act.quantize(&ws.xn);
                    par_matvec_qdot(
                        &self.pool,
                        *fmt,
                        packed,
                        &ws.act,
                        &mut ws.qkv,
                        q_dim + 2 * kv_dim,
                        dim,
                        self.strategy,
                    );
                }
                QkvWeights::Split(wq, wk, wv) => {
                    // One quantization feeds all three projections.
                    if needs_q8(wq) || needs_q8(wk) || needs_q8(wv) {
                        ws.act.quantize(&ws.xn);
                    }
                    let (q, rest) = ws.qkv.split_at_mut(q_dim);
                    let (k, v) = rest.split_at_mut(kv_dim);
                    matvec_w(&self.pool, self.strategy, wq, &ws.xn, &ws.act, q, q_dim, dim);
                    matvec_w(&self.pool, self.strategy, wk, &ws.xn, &ws.act, k, kv_dim, dim);
                    matvec_w(&self.pool, self.strategy, wv, &ws.xn, &ws.act, v, kv_dim, dim);
                }
            }
            let (q, rest) = ws.qkv.split_at_mut(q_dim);
            let (k, v) = rest.split_at_mut(kv_dim);

            if let Some(b) = &layer.bq {
                for (qi, bi) in q.iter_mut().zip(b) {
                    *qi += bi;
                }
            }
            if let Some(b) = &layer.bk {
                for (ki, bi) in k.iter_mut().zip(b) {
                    *ki += bi;
                }
            }
            if let Some(b) = &layer.bv {
                for (vi, bi) in v.iter_mut().zip(b) {
                    *vi += bi;
                }
            }

            // qwen3-style per-head RMSNorm on Q/K, applied before RoPE.
            if let Some(qn) = &layer.q_norm {
                for h in 0..c.n_heads {
                    let seg = &q[h * head_dim..(h + 1) * head_dim];
                    kernels::rms_norm_into(seg, qn, c.rms_eps, &mut ws.head);
                    q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&ws.head);
                }
            }
            if let Some(kn) = &layer.k_norm {
                for h in 0..c.n_kv_heads {
                    let seg = &k[h * head_dim..(h + 1) * head_dim];
                    kernels::rms_norm_into(seg, kn, c.rms_eps, &mut ws.head);
                    k[h * head_dim..(h + 1) * head_dim].copy_from_slice(&ws.head);
                }
            }

            for h in 0..c.n_heads {
                rope(
                    &mut q[h * head_dim..(h + 1) * head_dim],
                    pos,
                    head_dim,
                    c.rope_freq_base,
                    c.rope_style,
                );
            }
            for h in 0..c.n_kv_heads {
                rope(
                    &mut k[h * head_dim..(h + 1) * head_dim],
                    pos,
                    head_dim,
                    c.rope_freq_base,
                    c.rope_style,
                );
                self.cache.write_k(l, h, &k[h * head_dim..(h + 1) * head_dim]);
                self.cache.write_v(l, h, &v[h * head_dim..(h + 1) * head_dim]);
            }
            lap(&mut self.prof, |p| &mut p.qkv);

            let cached_len = self.cache.current_pos() + 1;
            for h in 0..c.n_heads {
                let kv_head = h / heads_per_kv.max(1);
                attention_one_into(
                    &q[h * head_dim..(h + 1) * head_dim],
                    self.cache.read_k(l, kv_head),
                    self.cache.read_v(l, kv_head),
                    head_dim,
                    &mut ws.scores[..cached_len],
                    &mut ws.attn_out[h * head_dim..(h + 1) * head_dim],
                );
            }
            lap(&mut self.prof, |p| &mut p.attn);

            if needs_q8(&layer.wo) {
                ws.act.quantize(&ws.attn_out);
            }
            matvec_w(
                &self.pool,
                self.strategy,
                &layer.wo,
                &ws.attn_out,
                &ws.act,
                &mut ws.proj,
                dim,
                q_dim,
            );
            for (xi, pi) in ws.x.iter_mut().zip(&ws.proj) {
                *xi += pi;
            }

            // --- SwiGLU feed-forward block ---
            kernels::rms_norm_into(&ws.x, &layer.ffn_norm, c.rms_eps, &mut ws.xn);
            match &layer.gate_up {
                // Fused SwiGLU over row-interleaved weights: one contiguous
                // stream per thread, one dispatch, no intermediate vectors.
                GateUp::FusedQuant(fmt, packed) => {
                    ws.act.quantize(&ws.xn);
                    par_matvec_swiglu(
                        &self.pool,
                        *fmt,
                        packed,
                        &ws.act,
                        &mut ws.gate,
                        c.hidden_dim,
                        dim,
                        self.strategy,
                    );
                }
                GateUp::Split(w_gate, w_up) => {
                    // One quantization feeds both gate and up.
                    if needs_q8(w_gate) || needs_q8(w_up) {
                        ws.act.quantize(&ws.xn);
                    }
                    matvec_w(
                        &self.pool,
                        self.strategy,
                        w_gate,
                        &ws.xn,
                        &ws.act,
                        &mut ws.gate,
                        c.hidden_dim,
                        dim,
                    );
                    matvec_w(
                        &self.pool,
                        self.strategy,
                        w_up,
                        &ws.xn,
                        &ws.act,
                        &mut ws.up,
                        c.hidden_dim,
                        dim,
                    );
                    kernels::silu_mul(&mut ws.gate, &ws.up);
                }
            }
            if needs_q8(&layer.w_down) {
                ws.act.quantize(&ws.gate);
            }
            matvec_w(
                &self.pool,
                self.strategy,
                &layer.w_down,
                &ws.gate,
                &ws.act,
                &mut ws.proj,
                dim,
                c.hidden_dim,
            );
            for (xi, di) in ws.x.iter_mut().zip(&ws.proj) {
                *xi += di;
            }
            lap(&mut self.prof, |p| &mut p.ffn);
        }

        // All layers committed this token's K/V — advance the cursor once.
        self.cache.advance();

        if want_logits {
            kernels::rms_norm_into(&ws.x, &self.model.output_norm, c.rms_eps, &mut ws.xn);
            if needs_q8(&self.model.output) {
                ws.act.quantize(&ws.xn);
            }
            matvec_w(
                &self.pool,
                self.strategy,
                &self.model.output,
                &ws.xn,
                &ws.act,
                &mut ws.logits,
                c.vocab_size,
                dim,
            );
            lap(&mut self.prof, |p| &mut p.lm_head);
        }
        if let Some(p) = self.prof.as_deref_mut() {
            p.tokens += 1;
        }
        Ok(())
    }

    /// The logits produced by the most recent forward pass.
    pub fn logits(&self) -> &[f32] {
        &self.ws.logits
    }

    /// Convenience wrapper: one forward pass, logits returned as a fresh
    /// `Vec`. Allocates — use [`Runner::forward_into`] + [`Runner::logits`]
    /// in the decode loop.
    pub fn forward(&mut self, token: u32, pos: usize) -> Result<Vec<f32>, GlError> {
        self.forward_into(token, pos)?;
        Ok(self.ws.logits.clone())
    }

    /// Generate up to `max_new_tokens` continuation tokens for `prompt`.
    ///
    /// `on_token` fires once per generated token. Generation stops early
    /// when `is_stop` returns true for a sampled token (the stop token is
    /// not emitted), at the model's context limit, or at the KV cache
    /// capacity. Callers wire `is_stop` to the tokenizer's stop set so all
    /// of a model's EOS variants (`<|im_end|>`, `<|endoftext|>`, ...) halt
    /// generation, not just the single metadata EOS.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut Sampler,
        is_stop: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<Vec<u32>, GlError> {
        if prompt.is_empty() {
            return Err(GlError::Engine("empty prompt".into()));
        }
        self.cache.reset();
        let max_seq = self.model.config.max_seq.min(self.cache.max_context);

        // Prefill: run every prompt token; only the last one needs logits,
        // so earlier positions skip the (huge) LM-head matvec entirely.
        for (pos, &tok) in prompt.iter().enumerate() {
            if pos >= max_seq {
                return Err(GlError::Engine(format!(
                    "prompt length {} exceeds context window {max_seq}",
                    prompt.len()
                )));
            }
            self.step(tok, pos, pos + 1 == prompt.len())?;
        }

        let mut generated = Vec::with_capacity(max_new_tokens);
        let mut pos = prompt.len();
        for _ in 0..max_new_tokens {
            if pos >= max_seq {
                break;
            }
            let t = self.prof.as_ref().map(|_| std::time::Instant::now());
            let next = sampler.sample(&self.ws.logits);
            if let (Some(p), Some(t)) = (self.prof.as_deref_mut(), t) {
                p.sampler += t.elapsed();
            }
            if is_stop(next) {
                break;
            }
            on_token(next);
            generated.push(next);
            self.forward_into(next, pos)?;
            pos += 1;
        }
        if let Some(p) = &self.prof {
            p.report();
        }
        Ok(generated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{GlprocModel, LayerWeights, ModelConfig, RopeStyle, WeightMatrix};
    use crate::sampler::{Sampler, SamplerConfig};

    /// Deterministic pseudo-random weights in [-0.1, 0.1].
    fn weights(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed | 1;
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                ((state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as f32
                    / (1u64 << 24) as f32
                    - 0.5)
                    * 0.2
            })
            .collect()
    }

    fn w(n: usize, seed: u64) -> WeightMatrix {
        WeightMatrix::F32(weights(n, seed))
    }

    /// Tiny 2-layer model: dim=8, 2 heads, 1 kv head (GQA), vocab=16.
    fn tiny_model() -> GlprocModel {
        let dim = 8;
        let n_heads = 2;
        let n_kv_heads = 1;
        let head_dim = 4;
        let hidden = 16;
        let vocab = 16;
        let layers = (0..2)
            .map(|i| LayerWeights {
                attn_norm: vec![1.0; dim],
                qkv: crate::model::QkvWeights::Split(
                    w(n_heads * head_dim * dim, 11 + i),
                    w(n_kv_heads * head_dim * dim, 22 + i),
                    w(n_kv_heads * head_dim * dim, 33 + i),
                ),
                wo: w(dim * n_heads * head_dim, 44 + i),
                bq: None,
                bk: None,
                bv: None,
                q_norm: None,
                k_norm: None,
                ffn_norm: vec![1.0; dim],
                gate_up: crate::model::GateUp::Split(
                    w(hidden * dim, 55 + i),
                    w(hidden * dim, 66 + i),
                ),
                w_down: w(dim * hidden, 77 + i),
            })
            .collect();
        GlprocModel {
            config: ModelConfig {
                arch: "llama".into(),
                dim,
                n_layers: 2,
                n_heads,
                n_kv_heads,
                head_dim,
                hidden_dim: hidden,
                vocab_size: vocab,
                max_seq: 64,
                rms_eps: 1e-5,
                rope_freq_base: 10_000.0,
                rope_style: RopeStyle::Norm,
            },
            token_embd: weights(vocab * dim, 99),
            layers,
            output_norm: vec![1.0; dim],
            output: w(vocab * dim, 99), // tied
        }
    }

    #[test]
    fn forward_produces_finite_logits() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        let logits = runner.forward(3, 0).unwrap();
        assert_eq!(logits.len(), 16);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn forward_is_deterministic() {
        let model = tiny_model();
        let a = Runner::new(&model).forward(5, 0).unwrap();
        let b = Runner::new(&model).forward(5, 0).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn generate_respects_max_tokens_and_streams() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0, // greedy
            top_k: 0,
            top_p: 1.0,
            seed: Some(1),
        });
        let mut streamed = Vec::new();
        let out = runner
            .generate(&[1, 2, 3], 5, &mut sampler, |_| false, |t| streamed.push(t))
            .unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(streamed, out);
        assert!(out.iter().all(|&t| (t as usize) < 16));
    }

    #[test]
    fn generate_halts_on_stop_token_without_emitting_it() {
        let model = tiny_model();
        let mut greedy = || {
            Sampler::new(SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                seed: Some(1),
            })
        };
        // Greedy decode with no stop set → learn the first sampled token.
        let mut runner = Runner::new(&model);
        let free = runner
            .generate(&[1, 2, 3], 5, &mut greedy(), |_| false, |_| {})
            .unwrap();
        let first = free[0];
        // Same decode, but the first token is now a stop token → nothing
        // is emitted or returned.
        let mut streamed = Vec::new();
        let stopped = runner
            .generate(&[1, 2, 3], 5, &mut greedy(), |t| t == first, |t| {
                streamed.push(t)
            })
            .unwrap();
        assert!(stopped.is_empty());
        assert!(streamed.is_empty());
    }

    #[test]
    fn generate_rejects_empty_prompt() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        let mut sampler = Sampler::new(SamplerConfig::default());
        assert!(runner
            .generate(&[], 5, &mut sampler, |_| false, |_| {})
            .is_err());
    }

    #[test]
    fn invalid_token_errors_not_panics() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        assert!(runner.forward(9999, 0).is_err());
    }

    #[test]
    fn generate_twice_reuses_cache_deterministically() {
        // The cursor-based cache must reset cleanly between conversations.
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        let mut s1 = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(1),
        });
        let a = runner
            .generate(&[1, 2, 3], 5, &mut s1, |_| false, |_| {})
            .unwrap();
        let mut s2 = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: Some(1),
        });
        let b = runner
            .generate(&[1, 2, 3], 5, &mut s2, |_| false, |_| {})
            .unwrap();
        assert_eq!(a, b);
    }
}
