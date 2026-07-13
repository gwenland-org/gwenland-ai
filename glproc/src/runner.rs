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
use crate::model::{FfnLayer, GateUp, GlprocModel, QkvWeights, RopeStyle, WeightMatrix};
use crate::sampler::Sampler;
use crate::simd_strategy::SimdStrategy;
use crate::threading::{
    par_matmul, par_matmul_qdot, par_matmul_swiglu, par_matvec, par_matvec_qdot,
    par_matvec_quant, par_matvec_swiglu, ThreadPool,
};

/// KV cache sequence capacity cap. Qwen-class models advertise 32k context;
/// pre-allocating that costs GBs, so the cache is sized to
/// `min(model max_seq, this)`. ~200 MB for Qwen2.5-0.5B dims.
const MAX_KV_CONTEXT: usize = 4096;

/// Decode threads. 4 matches the i3-1115G4 (4 logical cores); capped by the
/// machine's actual core count so small VMs don't oversubscribe.
/// `GLPROC_THREADS` overrides — for benchmarking and as the X5 thermal
/// knob (reduce thread count if the CPU runs hot).
///
/// Deliberately *logical* threads, not physical cores. Sizing this pool from
/// `topology::physical_core_count()` (2 on the i3-1115G4) was measured on
/// Qwen3-1.7B Q8_0 decode and lost: 8.5 vs 11.0 tok/s at steady state, a 23%
/// regression, reproduced across three alternating runs. Decode is
/// bandwidth-heavy but not issue-saturated — the per-block f16 scale
/// conversions, integer dot chains and scalar tails leave gaps that an SMT
/// sibling fills, keeping more loads in flight than 2 threads can. Running at
/// ~69% of the DRAM read ceiling is the evidence *for* SMT here, not against:
/// a saturated loop would sit near the ceiling with nothing left to overlap.
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

/// The decode pool's thread count, for telemetry. Same value the Runner uses.
pub fn thread_count() -> usize {
    n_threads()
}

/// Bytes the KV cache will allocate for this model — K and V, every layer,
/// every KV head, at the capped context length, f32.
///
/// Exposed for telemetry: this is the one part of the memory footprint that
/// scales with context, so a combined "peak RSS" figure cannot substitute for
/// it. Mirrors the sizing in [`Runner::new`]; a divergence here would misreport
/// rather than misbehave, so the constant is shared rather than duplicated.
pub fn kv_cache_bytes(c: &crate::model::ModelConfig) -> usize {
    let capacity = c.max_seq.min(MAX_KV_CONTEXT).max(1);
    // 2 = K and V.
    2 * c.n_layers * c.n_kv_heads * c.head_dim * capacity * std::mem::size_of::<f32>()
}

/// Below this many multiply-accumulates a matvec runs on the calling thread —
/// waking workers costs more than the work itself.
const PAR_MIN_WORK: usize = 1 << 16;

/// Prompt tokens processed per batched-prefill chunk. Batching lets every
/// weight row stream from DRAM once per chunk instead of once per token —
/// prefill flips from bandwidth-bound to compute-bound. 32 keeps the chunk's
/// activation set (32 × hidden_dim Q8 rows) inside L2.
const PREFILL_CHUNK: usize = 32;

/// How many of the most recent generated tokens the repetition penalty
/// looks back over. Matches llama.cpp's `repeat_last_n` default.
const REPEAT_WINDOW: usize = 64;

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

/// Batched matmul over either weight representation. The batched analog of
/// [`matvec_w`]: quantized formats with an integer kernel consume `acts`
/// (one pre-quantized activation per batch row); f32 weights read rows of
/// `xb` directly; bridge-only formats fall back to per-row bridge dots.
/// Output cell `(b, col_off + o)` lands at `yb[b * y_stride + col_off + o]`.
#[allow(clippy::too_many_arguments)]
fn matmul_w(
    pool: &ThreadPool,
    strategy: SimdStrategy,
    w: &WeightMatrix,
    xb: &[f32],
    x_stride: usize,
    acts: &[QuantizedActivation],
    yb: &mut [f32],
    y_stride: usize,
    col_off: usize,
    out_dim: usize,
    in_dim: usize,
    batch: usize,
) {
    match w {
        WeightMatrix::F32(data) => {
            par_matmul(
                pool, data, xb, x_stride, yb, y_stride, col_off, out_dim, in_dim, batch,
                strategy,
            );
        }
        WeightMatrix::Quant(fmt, blocks) => {
            if qdot::supports(*fmt) {
                debug_assert_eq!(acts.len(), batch, "caller must quantize xb rows first");
                par_matmul_qdot(
                    pool, *fmt, blocks, acts, yb, y_stride, col_off, out_dim, in_dim,
                    strategy,
                );
            } else {
                // No integer kernel (Q4_K): correctness fallback through the
                // f32 bridge, one batch row at a time.
                for b in 0..batch {
                    let x = &xb[b * x_stride..b * x_stride + in_dim];
                    let y = &mut yb[b * y_stride + col_off..b * y_stride + col_off + out_dim];
                    bridge_matvec_quant(*fmt, blocks, x, y, out_dim, in_dim, strategy);
                }
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

/// Batched-prefill buffers: the per-token [`Workspace`] fields with a
/// leading batch dimension of [`PREFILL_CHUNK`] rows, allocated once at
/// Runner construction (~1.5 MB for Qwen2.5-0.5B dims). All row-major
/// `[batch][width]`.
struct BatchWorkspace {
    /// Residual streams, `[chunk][dim]`.
    xb: Vec<f32>,
    /// RMSNorm outputs, `[chunk][dim]`.
    xnb: Vec<f32>,
    /// Q/K/V projections, `[chunk][q_dim + 2 * kv_dim]`.
    qkvb: Vec<f32>,
    /// Attention outputs, `[chunk][n_heads * head_dim]`.
    attn_outb: Vec<f32>,
    /// Projections back to the residual, `[chunk][dim]`.
    projb: Vec<f32>,
    /// SwiGLU gate (or fused gate·silu·up) outputs, `[chunk][hidden_dim]`.
    gateb: Vec<f32>,
    /// SwiGLU up outputs for the split-weights path, `[chunk][hidden_dim]`.
    upb: Vec<f32>,
    /// Attention score scratch, one row per chunk position — positions run
    /// attention in parallel, so they cannot share the decode scratch.
    /// `[chunk][kv capacity]`.
    scoresb: Vec<f32>,
    /// One Q8 activation per batch row for the integer-dot matmuls.
    acts: Vec<QuantizedActivation>,
}

/// Raw pointer into a batch buffer, handed to the pool for disjoint
/// per-position writes during chunk attention.
struct PtrShare(*mut f32);

// SAFETY: threads write disjoint per-position regions (each position `b`
// is owned by exactly one thread) and nobody reads until the pool's
// barrier in `run` has passed.
unsafe impl Send for PtrShare {}
unsafe impl Sync for PtrShare {}

/// Per-phase wall-time accumulators for the decode loop, enabled by setting
/// `GLPROC_PROFILE=1`. A measurement tool for finding the fat — when
/// disabled (`None`) the decode loop takes zero timestamps.
#[derive(Default)]
struct Prof {
    /// Attention-norm + Q/K/V matvecs + biases + head norms + RoPE + cache.
    qkv: std::time::Duration,
    /// Per-head single-query attention against the KV cache.
    attn: std::time::Duration,
    /// Attention output projection (+ its activation quantize) + residual.
    wo: std::time::Duration,
    /// FFN norm + activation quantize + fused gate/up/SiLU matvec.
    gateup: std::time::Duration,
    /// Gate-vector quantize + down matvec + residual.
    down: std::time::Duration,
    /// Final norm + LM-head matvec over the full vocabulary.
    lm_head: std::time::Duration,
    /// Token sampling (measured in `generate`).
    sampler: std::time::Duration,
    tokens: u32,

    /// Prefill (step_chunk) buckets, per prompt token.
    /// Embed + norms + activation quantizes (serial on the caller).
    p_serial: std::time::Duration,
    /// QKV batched matmul.
    p_qkv: std::time::Duration,
    /// Per-position bias/head-norm/RoPE/cache writes (serial).
    p_fixup: std::time::Duration,
    /// Chunk attention (parallel over positions).
    p_attn: std::time::Duration,
    /// Attention output projection matmul + residual.
    p_wo: std::time::Duration,
    /// Fused gate/up/SiLU batched matmul.
    p_gateup: std::time::Duration,
    /// Gate-vector quantize ahead of the down projection.
    p_downq: std::time::Duration,
    /// Down projection matmul + residual.
    p_down: std::time::Duration,
    p_tokens: u32,

    /// Tokens routed to each expert, summed over every MoE layer and token.
    /// Empty on a dense model — which is itself the signal that no routing
    /// happened, so it is left empty rather than zero-filled.
    expert_load: Vec<u64>,
    /// Shape of the MoE layers, if any: (num_experts, top_k, moe_layer_count).
    moe_shape: Option<(usize, usize, usize)>,
}

impl Prof {
    fn report(&self) {
        if self.p_tokens > 0 {
            let toks = self.p_tokens;
            let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / toks as f64;
            eprintln!(
                "[profile] prefill per token over {toks} tokens: serial {:.2}ms | \
                 qkv {:.2}ms | fixup {:.2}ms | attn {:.2}ms | wo {:.2}ms | \
                 gateup {:.2}ms | downq {:.2}ms | down {:.2}ms",
                ms(self.p_serial),
                ms(self.p_qkv),
                ms(self.p_fixup),
                ms(self.p_attn),
                ms(self.p_wo),
                ms(self.p_gateup),
                ms(self.p_downq),
                ms(self.p_down),
            );
        }
        let toks = self.tokens.max(1);
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1e3 / toks as f64;
        eprintln!(
            "[profile] per token over {} tokens: qkv {:.2}ms | attn {:.2}ms | \
             wo {:.2}ms | gateup {:.2}ms | down {:.2}ms | lm_head {:.2}ms | \
             sampler {:.2}ms",
            self.tokens,
            ms(self.qkv),
            ms(self.attn),
            ms(self.wo),
            ms(self.gateup),
            ms(self.down),
            ms(self.lm_head),
            ms(self.sampler),
        );
    }

    /// Project the raw counters into glcore's shared telemetry vocabulary.
    ///
    /// Totals, not per-token averages: glbench divides by whatever denominator
    /// it is reporting against, and an already-averaged number cannot be
    /// re-aggregated across iterations.
    ///
    /// `calls` counts actual *invocations*, which is not the token count: the
    /// per-layer stages (qkv, attention, ffn, ...) run once per layer per
    /// token, while `lm_head` and `sampler` run once per token. Reporting the
    /// token count for everything would understate the per-layer stages by a
    /// factor of `n_layers` and make `ms/call` meaningless.
    fn to_telemetry(&self, n_layers: usize) -> glcore::telemetry::EngineTelemetry {
        use glcore::telemetry::{EngineTelemetry, MoeTelemetry, PhaseProfile, StageTiming};

        let stage = |name: &str, d: std::time::Duration, calls: u64| StageTiming {
            name: name.to_string(),
            total_ms: d.as_secs_f64() * 1e3,
            calls,
        };

        let decode = (self.tokens > 0).then(|| {
            let toks = self.tokens as u64;
            let per_layer = toks * n_layers as u64;
            let stages = vec![
                stage("qkv", self.qkv, per_layer),
                stage("attention", self.attn, per_layer),
                stage("attn_out", self.wo, per_layer),
                stage("ffn_gate_up", self.gateup, per_layer),
                stage("ffn_down", self.down, per_layer),
                // Once per token, not per layer — the LM head runs after the
                // whole stack, and sampling after that.
                stage("lm_head", self.lm_head, toks),
                stage("sampler", self.sampler, toks),
            ];
            // The phase total is the sum of what we measured. glproc times
            // every stage of the decode loop, so there is no separate
            // wall-clock to compare against here; `unattributed_ms()` will
            // read 0, which is honest — the blind spot really is zero.
            let total_ms = stages.iter().map(|s| s.total_ms).sum();
            PhaseProfile { stages, total_ms }
        });

        let prefill = (self.p_tokens > 0).then(|| {
            // Every prefill bucket is inside the per-layer loop, so each ran
            // once per layer per chunk. Chunks, not tokens: prefill batches
            // tokens (PREFILL_CHUNK at a time), so a "call" processes a whole
            // batch. Round up — a partial last chunk is still a call.
            let chunks = (self.p_tokens as u64).div_ceil(PREFILL_CHUNK as u64);
            let per_layer = chunks * n_layers as u64;
            let stages = vec![
                stage("serial", self.p_serial, per_layer),
                stage("qkv", self.p_qkv, per_layer),
                stage("fixup", self.p_fixup, per_layer),
                stage("attention", self.p_attn, per_layer),
                stage("attn_out", self.p_wo, per_layer),
                stage("ffn_gate_up", self.p_gateup, per_layer),
                stage("ffn_downq", self.p_downq, per_layer),
                stage("ffn_down", self.p_down, per_layer),
            ];
            let total_ms = stages.iter().map(|s| s.total_ms).sum();
            PhaseProfile { stages, total_ms }
        });

        let moe = self
            .moe_shape
            .map(|(num_experts, num_experts_per_tok, moe_layers)| MoeTelemetry {
                num_experts,
                num_experts_per_tok,
                expert_load: self.expert_load.clone(),
                moe_layers,
            });

        EngineTelemetry {
            prefill,
            decode,
            backend: None, // filled by the engine, which knows the strategy
            memory: None,  // filled by the engine, which knows the model size
            moe,
        }
    }
}

/// Accumulate one MoE layer's routing into the profile.
///
/// No-op when profiling is off (`prof == None`), which is the common case —
/// so a routed model pays nothing for telemetry it did not ask for. When on,
/// the cost is one add per selected expert (top_k per token), which is
/// negligible next to the expert FFN it just ran.
fn record_moe(
    prof: &mut Option<Box<Prof>>,
    config: &crate::moe::MoEConfig,
    routing: &crate::moe::RoutingResult,
) {
    let Some(p) = prof.as_mut() else {
        return;
    };
    // First MoE layer seen defines the shape and sizes the accumulator. The
    // layer *count* is not derivable here (this fires once per layer per
    // token, so counting visits would multiply by token count) — the engine
    // fills it from the model, which actually knows.
    if p.moe_shape.is_none() {
        p.moe_shape = Some((config.num_experts, config.num_experts_per_tok, 0));
        p.expert_load = vec![0u64; config.num_experts];
    }
    for (e, &count) in routing.expert_load.iter().enumerate() {
        p.expert_load[e] += count as u64;
    }
}

/// Wall-clock timing for one [`Runner::generate`] call, split at the
/// prefill/decode boundary. Prefill (prompt processing) and generation
/// (the decode loop) have very different tok/s — one blended number hides
/// the real decode speed.
#[derive(Debug, Clone, Copy, Default)]
pub struct GenTiming {
    /// Number of prompt tokens processed during prefill.
    pub prompt_tokens: usize,
    /// Time to process the prompt (forward passes, mostly without LM head).
    pub prefill: std::time::Duration,
    /// Time in the decode loop: sampling, stop check, forward per token.
    pub decode: std::time::Duration,
}

/// Drives token-by-token inference over a loaded model.
pub struct Runner<'m> {
    model: &'m GlprocModel,
    cache: KvCache,
    pool: ThreadPool,
    strategy: SimdStrategy,
    ws: Workspace,
    bws: BatchWorkspace,
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
            bws: BatchWorkspace {
                xb: vec![0.0; PREFILL_CHUNK * c.dim],
                xnb: vec![0.0; PREFILL_CHUNK * c.dim],
                qkvb: vec![0.0; PREFILL_CHUNK * (q_dim + 2 * kv_dim)],
                attn_outb: vec![0.0; PREFILL_CHUNK * q_dim],
                projb: vec![0.0; PREFILL_CHUNK * c.dim],
                gateb: vec![0.0; PREFILL_CHUNK * c.hidden_dim],
                upb: vec![0.0; PREFILL_CHUNK * c.hidden_dim],
                scoresb: vec![0.0; PREFILL_CHUNK * kv_capacity],
                acts: (0..PREFILL_CHUNK)
                    .map(|_| {
                        QuantizedActivation::with_capacity(c.dim.max(c.hidden_dim).max(q_dim))
                    })
                    .collect(),
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

        let ws = &mut self.ws;
        self.model.embed_into(token, &mut ws.x)?;

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
            lap(&mut self.prof, |p| &mut p.wo);

            // --- feed-forward block (dense SwiGLU or routed experts) ---
            kernels::rms_norm_into(&ws.x, &layer.ffn_norm, c.rms_eps, &mut ws.xn);
            match &layer.ffn {
                FfnLayer::MoE(moe) => {
                    // Routes this token to top-k experts and writes the
                    // weighted combination into `proj`. Same contract as the
                    // dense path below: overwrite, caller adds the residual.
                    let routing = moe.forward(&ws.xn, &mut ws.proj, 1, &self.pool, self.strategy);
                    record_moe(&mut self.prof, &moe.config, &routing);
                    lap(&mut self.prof, |p| &mut p.gateup);
                }
                FfnLayer::Dense { gate_up, w_down } => {
                    match gate_up {
                        // Fused SwiGLU over row-interleaved weights: one
                        // contiguous stream per thread, one dispatch, no
                        // intermediate vectors.
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
                    lap(&mut self.prof, |p| &mut p.gateup);
                    if needs_q8(w_down) {
                        ws.act.quantize(&ws.gate);
                    }
                    matvec_w(
                        &self.pool,
                        self.strategy,
                        w_down,
                        &ws.gate,
                        &ws.act,
                        &mut ws.proj,
                        dim,
                        c.hidden_dim,
                    );
                }
            }
            for (xi, di) in ws.x.iter_mut().zip(&ws.proj) {
                *xi += di;
            }
            lap(&mut self.prof, |p| &mut p.down);
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

    /// Batched prefill step: run a chunk of `tokens` (≤ [`PREFILL_CHUNK`])
    /// through all layers together, writing K/V at explicit positions
    /// `start_pos..start_pos + tokens.len()`. The batched matmuls stream
    /// each weight row from DRAM once per *chunk* instead of once per
    /// token — the weight-row reuse that makes prefill compute-bound.
    /// Causality holds because a chunk's K/V rows are all written before
    /// any of its positions run attention, and position `p` only attends
    /// to rows `0..=p`. Advances the KV cursor by the chunk length; leaves
    /// the last token's logits in the workspace when `want_logits`.
    fn step_chunk(
        &mut self,
        tokens: &[u32],
        start_pos: usize,
        want_logits: bool,
    ) -> Result<(), GlError> {
        let c = &self.model.config;
        let dim = c.dim;
        let head_dim = c.head_dim;
        let q_dim = c.n_heads * head_dim;
        let kv_dim = c.n_kv_heads * head_dim;
        let q2kv = q_dim + 2 * kv_dim;
        let heads_per_kv = c.n_heads / c.n_kv_heads.max(1);
        let bsz = tokens.len();
        debug_assert!(bsz >= 1 && bsz <= PREFILL_CHUNK);
        debug_assert_eq!(start_pos, self.cache.current_pos());

        let bws = &mut self.bws;
        // One timestamp per phase boundary, only when profiling.
        let mut t = self.prof.as_ref().map(|_| std::time::Instant::now());
        let mut lap = |p: &mut Option<Box<Prof>>, pick: fn(&mut Prof) -> &mut std::time::Duration| {
            if let Some(p) = p.as_deref_mut() {
                let now = std::time::Instant::now();
                *pick(p) += now - t.unwrap();
                t = Some(now);
            }
        };
        for (b, &token) in tokens.iter().enumerate() {
            self.model
                .embed_into(token, &mut bws.xb[b * dim..(b + 1) * dim])?;
        }

        for (l, layer) in self.model.layers.iter().enumerate() {
            // --- attention block ---
            for b in 0..bsz {
                kernels::rms_norm_into(
                    &bws.xb[b * dim..(b + 1) * dim],
                    &layer.attn_norm,
                    c.rms_eps,
                    &mut bws.xnb[b * dim..(b + 1) * dim],
                );
            }
            let qkv_quant = match &layer.qkv {
                QkvWeights::FusedQuant(..) => true,
                QkvWeights::Split(wq, wk, wv) => needs_q8(wq) || needs_q8(wk) || needs_q8(wv),
            };
            if qkv_quant {
                for b in 0..bsz {
                    bws.acts[b].quantize(&bws.xnb[b * dim..(b + 1) * dim]);
                }
            }
            lap(&mut self.prof, |p| &mut p.p_serial);
            match &layer.qkv {
                QkvWeights::FusedQuant(fmt, packed) => {
                    par_matmul_qdot(
                        &self.pool,
                        *fmt,
                        packed,
                        &bws.acts[..bsz],
                        &mut bws.qkvb,
                        q2kv,
                        0,
                        q2kv,
                        dim,
                        self.strategy,
                    );
                }
                QkvWeights::Split(wq, wk, wv) => {
                    for (w, off, rows) in [
                        (wq, 0, q_dim),
                        (wk, q_dim, kv_dim),
                        (wv, q_dim + kv_dim, kv_dim),
                    ] {
                        matmul_w(
                            &self.pool,
                            self.strategy,
                            w,
                            &bws.xnb,
                            dim,
                            &bws.acts[..bsz],
                            &mut bws.qkvb,
                            q2kv,
                            off,
                            rows,
                            dim,
                            bsz,
                        );
                    }
                }
            }
            lap(&mut self.prof, |p| &mut p.p_qkv);

            // Per-position fixups: biases, head norms, RoPE, cache writes.
            // Every position's K/V must land in the cache before any
            // position of this chunk runs attention.
            for b in 0..bsz {
                let pos = start_pos + b;
                let row = &mut bws.qkvb[b * q2kv..(b + 1) * q2kv];
                let (q, rest) = row.split_at_mut(q_dim);
                let (k, v) = rest.split_at_mut(kv_dim);

                if let Some(bias) = &layer.bq {
                    for (qi, bi) in q.iter_mut().zip(bias) {
                        *qi += bi;
                    }
                }
                if let Some(bias) = &layer.bk {
                    for (ki, bi) in k.iter_mut().zip(bias) {
                        *ki += bi;
                    }
                }
                if let Some(bias) = &layer.bv {
                    for (vi, bi) in v.iter_mut().zip(bias) {
                        *vi += bi;
                    }
                }

                if let Some(qn) = &layer.q_norm {
                    for h in 0..c.n_heads {
                        let seg = &q[h * head_dim..(h + 1) * head_dim];
                        kernels::rms_norm_into(seg, qn, c.rms_eps, &mut self.ws.head);
                        q[h * head_dim..(h + 1) * head_dim].copy_from_slice(&self.ws.head);
                    }
                }
                if let Some(kn) = &layer.k_norm {
                    for h in 0..c.n_kv_heads {
                        let seg = &k[h * head_dim..(h + 1) * head_dim];
                        kernels::rms_norm_into(seg, kn, c.rms_eps, &mut self.ws.head);
                        k[h * head_dim..(h + 1) * head_dim].copy_from_slice(&self.ws.head);
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
                    self.cache
                        .write_k_at(l, h, pos, &k[h * head_dim..(h + 1) * head_dim]);
                    self.cache
                        .write_v_at(l, h, pos, &v[h * head_dim..(h + 1) * head_dim]);
                }
            }
            lap(&mut self.prof, |p| &mut p.p_fixup);

            // Causal attention, positions split across the pool. Attention
            // cost grows with cached length, so for long prompts this loop
            // dominates the chunk — run it on all threads. Every position
            // gets its own scores row; output rows are disjoint per position.
            {
                let n = self.pool.n_threads();
                let bchunk = bsz.div_ceil(n);
                let qkvb = &bws.qkvb;
                let cache = &self.cache;
                let n_heads = c.n_heads;
                let scores_stride = self.cache.max_context;
                let out_ptr = PtrShare(bws.attn_outb.as_mut_ptr());
                let scores_ptr = PtrShare(bws.scoresb.as_mut_ptr());
                self.pool.run(&|tid| {
                    let out_ptr = &out_ptr;
                    let scores_ptr = &scores_ptr;
                    let lo = (tid * bchunk).min(bsz);
                    let hi = (lo + bchunk).min(bsz);
                    for b in lo..hi {
                        let cached_len = start_pos + b + 1;
                        let q = &qkvb[b * q2kv..b * q2kv + q_dim];
                        // SAFETY: position b belongs to exactly one thread;
                        // its scores row and output row are disjoint from
                        // every other position's, and both are in bounds
                        // (b < bsz ≤ PREFILL_CHUNK, cached_len ≤ capacity).
                        let scores = unsafe {
                            std::slice::from_raw_parts_mut(
                                scores_ptr.0.add(b * scores_stride),
                                cached_len,
                            )
                        };
                        for h in 0..n_heads {
                            let kv_head = h / heads_per_kv.max(1);
                            // SAFETY: as above — (b, h) output segments are
                            // disjoint across threads.
                            let head_out = unsafe {
                                std::slice::from_raw_parts_mut(
                                    out_ptr.0.add(b * q_dim + h * head_dim),
                                    head_dim,
                                )
                            };
                            attention_one_into(
                                &q[h * head_dim..(h + 1) * head_dim],
                                cache.read_k_to(l, kv_head, cached_len),
                                cache.read_v_to(l, kv_head, cached_len),
                                head_dim,
                                scores,
                                head_out,
                            );
                        }
                    }
                });
            }

            lap(&mut self.prof, |p| &mut p.p_attn);

            if needs_q8(&layer.wo) {
                for b in 0..bsz {
                    bws.acts[b].quantize(&bws.attn_outb[b * q_dim..(b + 1) * q_dim]);
                }
            }
            matmul_w(
                &self.pool,
                self.strategy,
                &layer.wo,
                &bws.attn_outb,
                q_dim,
                &bws.acts[..bsz],
                &mut bws.projb,
                dim,
                0,
                dim,
                q_dim,
                bsz,
            );
            for b in 0..bsz {
                for (xi, pi) in bws.xb[b * dim..(b + 1) * dim]
                    .iter_mut()
                    .zip(&bws.projb[b * dim..(b + 1) * dim])
                {
                    *xi += pi;
                }
            }
            lap(&mut self.prof, |p| &mut p.p_wo);

            // --- SwiGLU feed-forward block ---
            for b in 0..bsz {
                kernels::rms_norm_into(
                    &bws.xb[b * dim..(b + 1) * dim],
                    &layer.ffn_norm,
                    c.rms_eps,
                    &mut bws.xnb[b * dim..(b + 1) * dim],
                );
            }
            match &layer.ffn {
                FfnLayer::MoE(moe) => {
                    // Batched routing: tokens are grouped by expert inside
                    // `forward`, so each active expert streams its weights
                    // once for all of its tokens, and inactive experts are
                    // never touched.
                    lap(&mut self.prof, |p| &mut p.p_serial);
                    let routing = moe.forward(
                        &bws.xnb[..bsz * dim],
                        &mut bws.projb[..bsz * dim],
                        bsz,
                        &self.pool,
                        self.strategy,
                    );
                    record_moe(&mut self.prof, &moe.config, &routing);
                    lap(&mut self.prof, |p| &mut p.p_gateup);
                }
                FfnLayer::Dense { gate_up, w_down } => {
                    match gate_up {
                        GateUp::FusedQuant(fmt, packed) => {
                            for b in 0..bsz {
                                bws.acts[b].quantize(&bws.xnb[b * dim..(b + 1) * dim]);
                            }
                            lap(&mut self.prof, |p| &mut p.p_serial);
                            par_matmul_swiglu(
                                &self.pool,
                                *fmt,
                                packed,
                                &bws.acts[..bsz],
                                &mut bws.gateb,
                                c.hidden_dim,
                                c.hidden_dim,
                                dim,
                                self.strategy,
                            );
                        }
                        GateUp::Split(w_gate, w_up) => {
                            if needs_q8(w_gate) || needs_q8(w_up) {
                                for b in 0..bsz {
                                    bws.acts[b].quantize(&bws.xnb[b * dim..(b + 1) * dim]);
                                }
                            }
                            for (w, y) in [(w_gate, &mut bws.gateb), (w_up, &mut bws.upb)] {
                                matmul_w(
                                    &self.pool,
                                    self.strategy,
                                    w,
                                    &bws.xnb,
                                    dim,
                                    &bws.acts[..bsz],
                                    y,
                                    c.hidden_dim,
                                    0,
                                    c.hidden_dim,
                                    dim,
                                    bsz,
                                );
                            }
                            for b in 0..bsz {
                                let lo = b * c.hidden_dim;
                                let hi = lo + c.hidden_dim;
                                kernels::silu_mul(&mut bws.gateb[lo..hi], &bws.upb[lo..hi]);
                            }
                        }
                    }
                    lap(&mut self.prof, |p| &mut p.p_gateup);
                    if needs_q8(w_down) {
                        for b in 0..bsz {
                            bws.acts[b]
                                .quantize(&bws.gateb[b * c.hidden_dim..(b + 1) * c.hidden_dim]);
                        }
                    }
                    lap(&mut self.prof, |p| &mut p.p_downq);
                    matmul_w(
                        &self.pool,
                        self.strategy,
                        w_down,
                        &bws.gateb,
                        c.hidden_dim,
                        &bws.acts[..bsz],
                        &mut bws.projb,
                        dim,
                        0,
                        dim,
                        c.hidden_dim,
                        bsz,
                    );
                }
            }
            for b in 0..bsz {
                for (xi, di) in bws.xb[b * dim..(b + 1) * dim]
                    .iter_mut()
                    .zip(&bws.projb[b * dim..(b + 1) * dim])
                {
                    *xi += di;
                }
            }
            lap(&mut self.prof, |p| &mut p.p_down);
        }
        if let Some(p) = self.prof.as_deref_mut() {
            p.p_tokens += bsz as u32;
        }

        // Every layer committed K/V for all chunk positions.
        self.cache.advance_by(bsz);

        if want_logits {
            let last = bsz - 1;
            let ws = &mut self.ws;
            kernels::rms_norm_into(
                &bws.xb[last * dim..(last + 1) * dim],
                &self.model.output_norm,
                c.rms_eps,
                &mut ws.xn,
            );
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
    ///
    /// Per-stage telemetry for the runs so far, or `None` when profiling is
    /// off (`GLPROC_PROFILE` unset).
    ///
    /// `None` means *not measured*, never *zero* — a consumer that renders a
    /// missing profile as "0 ms in attention" would be inventing a fact.
    ///
    /// Backend/memory/layer-count fields are left empty here and filled by
    /// [`crate::engine::GlprocEngine`], which is the layer that knows the SIMD
    /// strategy, the model size, and how many layers are routed.
    pub fn telemetry(&self) -> Option<glcore::telemetry::EngineTelemetry> {
        let n_layers = self.model.config.n_layers;
        self.prof.as_ref().map(|p| p.to_telemetry(n_layers))
    }

    /// Returns the generated tokens plus a [`GenTiming`] separating prefill
    /// from decode wall time.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        max_new_tokens: usize,
        sampler: &mut Sampler,
        is_stop: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<(Vec<u32>, GenTiming), GlError> {
        if prompt.is_empty() {
            return Err(GlError::Engine("empty prompt".into()));
        }
        self.cache.reset();
        let max_seq = self.model.config.max_seq.min(self.cache.max_context);

        // Prefill in batched chunks: each weight row streams from DRAM once
        // per chunk instead of once per token. Only the last chunk's last
        // token computes logits, so earlier positions still skip the (huge)
        // LM-head matvec entirely.
        if prompt.len() > max_seq {
            return Err(GlError::Engine(format!(
                "prompt length {} exceeds context window {max_seq}",
                prompt.len()
            )));
        }
        let prefill_start = std::time::Instant::now();
        let mut consumed = 0;
        for chunk in prompt.chunks(PREFILL_CHUNK) {
            let is_last = consumed + chunk.len() == prompt.len();
            self.step_chunk(chunk, consumed, is_last)?;
            consumed += chunk.len();
        }
        let prefill = prefill_start.elapsed();

        let decode_start = std::time::Instant::now();
        let mut generated = Vec::with_capacity(max_new_tokens);
        // Sliding window of recent tokens for the repetition penalty.
        // Allocated once per generate call, outside the per-token loop.
        let mut recent: std::collections::VecDeque<u32> =
            std::collections::VecDeque::with_capacity(REPEAT_WINDOW);
        let mut pos = prompt.len();
        for _ in 0..max_new_tokens {
            if pos >= max_seq {
                break;
            }
            let t = self.prof.as_ref().map(|_| std::time::Instant::now());
            crate::sampler::apply_repetition_penalty(
                &mut self.ws.logits,
                recent.make_contiguous(),
                sampler.repeat_penalty(),
            );
            let next = sampler.sample(&self.ws.logits);
            if let (Some(p), Some(t)) = (self.prof.as_deref_mut(), t) {
                p.sampler += t.elapsed();
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
            self.forward_into(next, pos)?;
            pos += 1;
        }
        if let Some(p) = &self.prof {
            p.report();
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
                ffn: crate::model::FfnLayer::Dense {
                    gate_up: crate::model::GateUp::Split(
                        w(hidden * dim, 55 + i),
                        w(hidden * dim, 66 + i),
                    ),
                    w_down: w(dim * hidden, 77 + i),
                },
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
            token_embd: WeightMatrix::F32(weights(vocab * dim, 99)),
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

    /// Batched prefill must produce the same logits (and KV state) as the
    /// sequential per-token path — checked here by comparing final logits
    /// and one greedy continuation after prefill.
    fn assert_chunked_prefill_matches_sequential(prompt: &[u32]) {
        let model = tiny_model();

        // Sequential ground truth: one step per token.
        let mut seq = Runner::new(&model);
        for (pos, &tok) in prompt.iter().enumerate() {
            seq.step(tok, pos, pos + 1 == prompt.len()).unwrap();
        }
        let want: Vec<f32> = seq.logits().to_vec();

        // Chunked path, exactly as generate() drives it.
        let mut bat = Runner::new(&model);
        let mut consumed = 0;
        for chunk in prompt.chunks(PREFILL_CHUNK) {
            let is_last = consumed + chunk.len() == prompt.len();
            bat.step_chunk(chunk, consumed, is_last).unwrap();
            consumed += chunk.len();
        }
        let got = bat.logits();

        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-4,
                "logit {i}: chunked {g} vs sequential {w} (prompt len {})",
                prompt.len()
            );
        }

        // KV parity: the greedy next token and one more step must agree.
        let next = Sampler::greedy(got);
        assert_eq!(next, Sampler::greedy(&want));
        seq.forward_into(next, prompt.len()).unwrap();
        bat.forward_into(next, prompt.len()).unwrap();
        for (g, w) in bat.logits().iter().zip(seq.logits()) {
            assert!((g - w).abs() < 1e-4, "post-prefill decode diverged");
        }
    }

    #[test]
    fn chunked_prefill_matches_sequential_single_chunk() {
        assert_chunked_prefill_matches_sequential(&[1, 2, 3, 4, 5]);
    }

    #[test]
    fn chunked_prefill_matches_sequential_multi_chunk() {
        // 40 tokens: one full 32-chunk plus a ragged 8-token tail, crossing
        // the chunk boundary (max_seq of the tiny model is 64).
        let prompt: Vec<u32> = (0..40).map(|i| (i % 16) as u32).collect();
        assert_chunked_prefill_matches_sequential(&prompt);
    }

    #[test]
    fn chunked_prefill_rejects_bad_token() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        assert!(runner.step_chunk(&[1, 9999], 0, true).is_err());
    }

    #[test]
    fn generate_respects_max_tokens_and_streams() {
        let model = tiny_model();
        let mut runner = Runner::new(&model);
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0, // greedy
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(1),
        });
        let mut streamed = Vec::new();
        let (out, timing) = runner
            .generate(&[1, 2, 3], 5, &mut sampler, |_| false, |t| streamed.push(t))
            .unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(streamed, out);
        assert!(out.iter().all(|&t| (t as usize) < 16));
        assert_eq!(timing.prompt_tokens, 3);
        assert!(timing.prefill > std::time::Duration::ZERO);
        assert!(timing.decode > std::time::Duration::ZERO);
    }

    #[test]
    fn generate_halts_on_stop_token_without_emitting_it() {
        let model = tiny_model();
        let mut greedy = || {
            Sampler::new(SamplerConfig {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                repeat_penalty: 1.0,
                seed: Some(1),
            })
        };
        // Greedy decode with no stop set → learn the first sampled token.
        let mut runner = Runner::new(&model);
        let (free, _) = runner
            .generate(&[1, 2, 3], 5, &mut greedy(), |_| false, |_| {})
            .unwrap();
        let first = free[0];
        // Same decode, but the first token is now a stop token → nothing
        // is emitted or returned.
        let mut streamed = Vec::new();
        let (stopped, _) = runner
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
            repeat_penalty: 1.0,
            seed: Some(1),
        });
        let (a, _) = runner
            .generate(&[1, 2, 3], 5, &mut s1, |_| false, |_| {})
            .unwrap();
        let mut s2 = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            seed: Some(1),
        });
        let (b, _) = runner
            .generate(&[1, 2, 3], 5, &mut s2, |_| false, |_| {})
            .unwrap();
        assert_eq!(a, b);
    }
}
