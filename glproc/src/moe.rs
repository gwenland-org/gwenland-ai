//! Mixture-of-Experts feed-forward block (Qwen3-MoE and friends).
//!
//! A dense FFN runs one SwiGLU projection per token. An MoE FFN keeps
//! `num_experts` independent SwiGLU projections and routes each token through
//! only `top_k` of them, so the *activated* parameter count per token is
//! `top_k / num_experts` of the total. Qwen3-30B-A3B is 128 experts, top-8:
//! 30B stored, ~3B activated (the "A3B").
//!
//! Layout reuse, not reinvention: each expert's gate+up pair is stored in the
//! same row-interleaved [`GateUp::FusedQuant`] form the dense path uses, so an
//! expert's FFN is exactly [`par_matvec_swiglu`] / [`par_matmul_swiglu`] over
//! that expert's slice. Nothing new is added to the kernel layer.
//!
//! ## Threading
//!
//! Experts run *sequentially*, each using the whole pool internally, rather
//! than one expert per worker. Two reasons. The pool is not reentrant —
//! `ThreadPool::run` blocks until its workers finish, so calling a `par_*`
//! kernel from inside a `run` closure would deadlock. And expert-per-worker
//! would be the wrong split anyway: at decode (one token, `top_k` experts) the
//! per-expert matvec is already big enough to saturate two cores, and row-
//! chunking it keeps each thread on one sequential DRAM stream — the property
//! `threading.rs` documents as worth ~35% end-to-end.
//!
//! ## Decode vs prefill
//!
//! Decode (`n_tokens == 1`) touches `top_k` experts and takes the matvec path.
//! Prefill batches tokens, and different tokens pick different experts, so
//! tokens are *grouped by expert*: each expert runs one batched matmul over
//! just the tokens routed to it. An expert holding no tokens is skipped
//! entirely — never touched, never streamed from DRAM. That skip is the whole
//! performance argument for MoE, so it is structural here, not an
//! optimization.

use crate::kernels::qdot::QuantizedActivation;
use crate::kernels::{self, qdot};
use crate::model::{GateUp, WeightMatrix};
use crate::simd_strategy::SimdStrategy;
use crate::threading::{
    par_matmul_qdot, par_matmul_swiglu, par_matvec_qdot, par_matvec_swiglu, ThreadPool,
};

/// Largest `top_k` we store inline. Qwen3 uses 8; Mixtral-style models use 2.
/// Routing picks per token are held in a fixed-size array to keep the hot loop
/// allocation-free — a model wanting more than this falls back to nothing, so
/// the loader must reject it rather than silently truncate.
pub const MAX_TOP_K: usize = 8;

/// MoE hyperparameters, read from GGUF metadata — never hardcoded.
///
/// GGUF keys (llama.cpp convention):
/// `{arch}.expert_count` → [`num_experts`](Self::num_experts),
/// `{arch}.expert_used_count` → [`num_experts_per_tok`](Self::num_experts_per_tok),
/// `{arch}.expert_feed_forward_length` → [`expert_ffn_size`](Self::expert_ffn_size).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoEConfig {
    /// Total experts held in the layer (128 for Qwen3-30B-A3B).
    pub num_experts: usize,
    /// Experts each token is routed through — "top_k" (8 for Qwen3).
    pub num_experts_per_tok: usize,
    /// Inner SwiGLU width of *one* expert. Much smaller than a dense model's
    /// `hidden_dim` (768 for Qwen3-30B-A3B, vs 2048 hidden).
    pub expert_ffn_size: usize,
    /// Model embedding width; an expert maps `hidden_size → hidden_size`.
    pub hidden_size: usize,
    /// Rescale the `top_k` selected probabilities to sum to 1. Qwen3 sets
    /// this; without it the residual is scaled by however much mass the
    /// unselected experts held.
    pub norm_topk_prob: bool,
}

impl MoEConfig {
    /// Reject configs the kernels cannot honor, so a bad GGUF fails at load
    /// instead of silently producing wrong logits.
    pub fn validate(&self) -> Result<(), glcore::GlError> {
        let bad = |m: String| Err(glcore::GlError::Engine(m));
        if self.num_experts == 0 {
            return bad("MoE: expert_count is 0".into());
        }
        if self.num_experts_per_tok == 0 {
            return bad("MoE: expert_used_count is 0".into());
        }
        if self.num_experts_per_tok > self.num_experts {
            return bad(format!(
                "MoE: expert_used_count {} exceeds expert_count {}",
                self.num_experts_per_tok, self.num_experts
            ));
        }
        if self.num_experts_per_tok > MAX_TOP_K {
            return bad(format!(
                "MoE: expert_used_count {} exceeds supported max {MAX_TOP_K}",
                self.num_experts_per_tok
            ));
        }
        if self.expert_ffn_size == 0 || self.hidden_size == 0 {
            return bad("MoE: zero expert_ffn_size or hidden_size".into());
        }
        Ok(())
    }
}

/// One expert's weights, in the dense path's layouts.
pub struct ExpertWeights {
    /// SwiGLU gate + up, `[expert_ffn_size, hidden_size]` each, row-interleaved
    /// when quantized (see [`GateUp`]).
    pub gate_up: GateUp,
    /// Down projection, `[hidden_size, expert_ffn_size]`.
    pub w_down: WeightMatrix,
}

/// An MoE feed-forward block: a router plus `num_experts` experts.
pub struct MoELayer {
    pub config: MoEConfig,
    /// Router ("gate_inp") projection, `[num_experts, hidden_size]` row-major
    /// — GGUF's `[out, in]` convention, same as every other weight here.
    /// Tiny (128 x 2048), so it stays f32 and is dotted on the caller thread.
    pub router: Vec<f32>,
    /// Experts, indexed by expert id.
    pub experts: Vec<ExpertWeights>,
}

/// Which experts each token picked, and with what weight.
pub struct RoutingResult {
    /// `[n_tokens][top_k]` expert ids. Only the first `top_k` entries of each
    /// row are meaningful.
    pub expert_ids: Vec<[u32; MAX_TOP_K]>,
    /// `[n_tokens][top_k]` combine weights, post-softmax and post-renorm.
    pub weights: Vec<[f32; MAX_TOP_K]>,
    /// `[num_experts]` — tokens routed to each expert. Load-balance telemetry;
    /// also tells the dispatcher which experts to skip (count 0).
    pub expert_load: Vec<usize>,
}

/// Softmax over `scores` in place, then pick the `top_k` largest.
///
/// Softmax *before* selection (not after) is what Qwen3 does: the weights are
/// probabilities over all `num_experts`, and `norm_topk_prob` then rescales the
/// chosen `top_k` back to sum 1. Selecting first and softmaxing only the
/// winners gives different numbers.
fn softmax_top_k(
    scores: &mut [f32],
    top_k: usize,
    norm: bool,
) -> ([u32; MAX_TOP_K], [f32; MAX_TOP_K]) {
    // Softmax, max-shifted so exp() cannot overflow on a confident router.
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        // fast_exp, not f32::exp — this is a hot path (once per token per layer).
        let e = kernels::fast_exp(*s - max);
        *s = e;
        sum += e;
    }
    let inv = 1.0 / sum;
    for s in scores.iter_mut() {
        *s *= inv;
    }

    // Partial selection sort: top_k <= 8 and num_experts <= 128, so k passes
    // over the score vector beats sorting all 128.
    let mut ids = [0u32; MAX_TOP_K];
    let mut wts = [0f32; MAX_TOP_K];
    let mut taken = [false; 256];
    debug_assert!(scores.len() <= taken.len(), "num_experts > 256 unsupported");
    for slot in 0..top_k {
        let mut best = usize::MAX;
        let mut best_v = f32::NEG_INFINITY;
        for (e, &v) in scores.iter().enumerate() {
            // `>` not `>=`: ties resolve to the lowest expert id, which makes
            // routing deterministic across runs (the tests rely on this, and
            // so does reproducible generation).
            if !taken[e] && v > best_v {
                best_v = v;
                best = e;
            }
        }
        taken[best] = true;
        ids[slot] = best as u32;
        wts[slot] = best_v;
    }

    if norm {
        let s: f32 = wts[..top_k].iter().sum();
        // A uniform router over 128 experts gives each ~0.0078; the sum of 8
        // is ~0.06, far from zero. Guard anyway — a degenerate all-(-inf) row
        // would otherwise produce NaN weights and poison the residual.
        if s > 0.0 {
            let inv = 1.0 / s;
            for w in wts[..top_k].iter_mut() {
                *w *= inv;
            }
        }
    }
    (ids, wts)
}

impl MoELayer {
    /// Route every token and return the selection. `input` is
    /// `[n_tokens, hidden_size]`, row-major.
    pub fn compute_routing(&self, input: &[f32], n_tokens: usize) -> RoutingResult {
        let c = &self.config;
        debug_assert_eq!(input.len(), n_tokens * c.hidden_size);

        let mut expert_ids = Vec::with_capacity(n_tokens);
        let mut weights = Vec::with_capacity(n_tokens);
        let mut expert_load = vec![0usize; c.num_experts];
        let mut scores = vec![0f32; c.num_experts];

        for t in 0..n_tokens {
            let x = &input[t * c.hidden_size..(t + 1) * c.hidden_size];
            // Router matvec on the caller thread: [128 x 2048] is ~1 MFLOP,
            // far below PAR_MIN_WORK, so waking the pool would cost more than
            // the dot itself.
            kernels::matvec(&self.router, x, &mut scores, c.num_experts, c.hidden_size);
            let (ids, wts) = softmax_top_k(&mut scores, c.num_experts_per_tok, c.norm_topk_prob);
            for &e in &ids[..c.num_experts_per_tok] {
                expert_load[e as usize] += 1;
            }
            expert_ids.push(ids);
            weights.push(wts);
        }

        RoutingResult {
            expert_ids,
            weights,
            expert_load,
        }
    }

    /// MoE feed-forward: `output[t] = sum_k w[t,k] * expert[id[t,k]](input[t])`.
    ///
    /// `output` is *overwritten*, not accumulated — the caller adds the
    /// residual, matching the dense FFN's contract.
    pub fn forward(
        &self,
        input: &[f32],
        output: &mut [f32],
        n_tokens: usize,
        pool: &ThreadPool,
        strategy: SimdStrategy,
    ) -> RoutingResult {
        let c = &self.config;
        debug_assert_eq!(input.len(), n_tokens * c.hidden_size);
        debug_assert_eq!(output.len(), n_tokens * c.hidden_size);

        let routing = self.compute_routing(input, n_tokens);
        output.fill(0.0);

        // Invert the routing: for each expert, the tokens that chose it and
        // the weight each gave it. Experts with no tokens never appear, so
        // their weights are never streamed from DRAM.
        let mut per_expert: Vec<Vec<(usize, f32)>> = vec![Vec::new(); c.num_experts];
        for t in 0..n_tokens {
            for k in 0..c.num_experts_per_tok {
                let e = routing.expert_ids[t][k] as usize;
                per_expert[e].push((t, routing.weights[t][k]));
            }
        }

        let mut scratch = Scratch::new(c.hidden_size, c.expert_ffn_size);
        for (e, toks) in per_expert.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            self.run_expert(e, toks, input, output, pool, strategy, &mut scratch);
        }
        routing
    }

    /// Run expert `e` over its assigned tokens and scatter-accumulate the
    /// weighted result into `output`.
    #[allow(clippy::too_many_arguments)]
    fn run_expert(
        &self,
        e: usize,
        toks: &[(usize, f32)],
        input: &[f32],
        output: &mut [f32],
        pool: &ThreadPool,
        strategy: SimdStrategy,
        s: &mut Scratch,
    ) {
        let c = &self.config;
        let (h, f) = (c.hidden_size, c.expert_ffn_size);
        let expert = &self.experts[e];
        let n = toks.len();

        // Gather this expert's tokens into a contiguous batch. The gather is
        // what lets one matmul serve many tokens; without it each token would
        // re-stream the expert's weights from DRAM.
        s.gathered.resize(n * h, 0.0);
        for (i, &(t, _)) in toks.iter().enumerate() {
            s.gathered[i * h..(i + 1) * h].copy_from_slice(&input[t * h..(t + 1) * h]);
        }
        s.gate.resize(n * f, 0.0);
        s.down.resize(n * h, 0.0);

        // Destructure the scratch once: the kernels need `&gathered` and
        // `&mut gate` live at the same time, which a `&mut Scratch` method
        // call cannot express. Field-splitting the borrow is what makes the
        // whole body safe without a single raw pointer.
        let Scratch {
            gathered,
            gate,
            up,
            down,
            acts,
            act_cap,
        } = s;
        let act_cap = *act_cap;

        // --- gate+up (SwiGLU) ---
        match &expert.gate_up {
            GateUp::FusedQuant(fmt, packed) => {
                quantize_into(acts, gathered, n, h, act_cap);
                if n == 1 {
                    par_matvec_swiglu(pool, *fmt, packed, &acts[0], gate, f, h, strategy);
                } else {
                    par_matmul_swiglu(
                        pool,
                        *fmt,
                        packed,
                        &acts[..n],
                        gate,
                        f,
                        f,
                        h,
                        strategy,
                    );
                }
            }
            // f32 / mismatched-format fallback: gate and up as two matmuls,
            // then the SiLU-multiply sweep. Same shape the dense path takes.
            GateUp::Split(w_gate, w_up) => {
                up.resize(n * f, 0.0);
                let needs_q8 = |w: &WeightMatrix| {
                    matches!(w, WeightMatrix::Quant(fmt, _) if qdot::supports(*fmt))
                };
                if needs_q8(w_gate) || needs_q8(w_up) {
                    quantize_into(acts, gathered, n, h, act_cap);
                }
                dense_matmul(pool, strategy, w_gate, gathered, h, acts, gate, f, h, n);
                dense_matmul(pool, strategy, w_up, gathered, h, acts, up, f, h, n);
                for b in 0..n {
                    kernels::silu_mul(&mut gate[b * f..(b + 1) * f], &up[b * f..(b + 1) * f]);
                }
            }
        }

        // --- down projection ---
        match &expert.w_down {
            WeightMatrix::Quant(fmt, blocks) if qdot::supports(*fmt) => {
                quantize_into(acts, gate, n, f, act_cap);
                if n == 1 {
                    par_matvec_qdot(pool, *fmt, blocks, &acts[0], down, h, f, strategy);
                } else {
                    par_matmul_qdot(
                        pool,
                        *fmt,
                        blocks,
                        &acts[..n],
                        down,
                        h,
                        0,
                        h,
                        f,
                        strategy,
                    );
                }
            }
            w => {
                dense_matmul(pool, strategy, w, gate, f, acts, down, h, f, n);
            }
        }

        // --- weighted scatter-accumulate ---
        // `+=`, because a token routed to top_k experts lands here top_k times.
        for (i, &(t, w)) in toks.iter().enumerate() {
            let dst = &mut output[t * h..(t + 1) * h];
            let src = &down[i * h..(i + 1) * h];
            for (o, v) in dst.iter_mut().zip(src) {
                *o += w * v;
            }
        }
    }
}

/// Quantize `n` rows of width `in_dim` from `src` into `acts`, growing `acts`
/// as needed. Free function, not a method, so the caller can hold `&src` and
/// `&mut acts` as separate borrows of the same `Scratch`.
///
/// `act_cap` sizes newly pushed buffers. It must be the *largest* width any
/// caller will quantize into this `acts` — the gate step uses `hidden_size`
/// and the down step uses `expert_ffn_size`, and `QuantizedActivation::quantize`
/// does not grow its buffers (it only `debug_assert`s the fit), so a buffer
/// sized to the smaller of the two would write out of bounds in release.
fn quantize_into(
    acts: &mut Vec<QuantizedActivation>,
    src: &[f32],
    n: usize,
    in_dim: usize,
    act_cap: usize,
) {
    debug_assert!(in_dim <= act_cap, "act_cap must cover the widest quantize");
    while acts.len() < n {
        acts.push(QuantizedActivation::with_capacity(act_cap));
    }
    for b in 0..n {
        acts[b].quantize(&src[b * in_dim..(b + 1) * in_dim]);
    }
}

/// Per-layer reusable buffers. Sized to the *largest* expert batch seen so
/// far and never shrunk — a fresh Vec per expert would fault in new pages on
/// every one of the 128, which `threading.rs` records as a source of
/// multi-millisecond stalls.
struct Scratch {
    gathered: Vec<f32>,
    gate: Vec<f32>,
    up: Vec<f32>,
    down: Vec<f32>,
    acts: Vec<QuantizedActivation>,
    /// Width to allocate each `QuantizedActivation` at — see [`quantize_into`].
    act_cap: usize,
}

impl Scratch {
    fn new(hidden: usize, ffn: usize) -> Self {
        Scratch {
            gathered: Vec::new(),
            gate: Vec::new(),
            up: Vec::new(),
            down: Vec::new(),
            acts: Vec::new(),
            // The gate step quantizes `hidden`-wide rows, the down step
            // `ffn`-wide ones, and both share `acts`.
            act_cap: hidden.max(ffn),
        }
    }
}

/// Matmul over either weight representation, batch-aware. Mirrors the runner's
/// `matmul_w` but is local here so MoE does not depend on runner internals.
///
/// `acts` is only read on the integer-dot path; the f32 and bridge paths take
/// `x` directly, and may be handed an empty slice.
#[allow(clippy::too_many_arguments)]
fn dense_matmul(
    pool: &ThreadPool,
    strategy: SimdStrategy,
    w: &WeightMatrix,
    x: &[f32],
    x_stride: usize,
    acts: &[QuantizedActivation],
    y: &mut [f32],
    out_dim: usize,
    in_dim: usize,
    batch: usize,
) {
    match w {
        WeightMatrix::F32(data) => {
            crate::threading::par_matmul(
                pool, data, x, x_stride, y, out_dim, 0, out_dim, in_dim, batch, strategy,
            );
        }
        WeightMatrix::Quant(fmt, blocks) if qdot::supports(*fmt) => {
            debug_assert!(
                acts.len() >= batch,
                "integer-dot path needs {batch} quantized activations, got {}",
                acts.len()
            );
            par_matmul_qdot(
                pool,
                *fmt,
                blocks,
                &acts[..batch],
                y,
                out_dim,
                0,
                out_dim,
                in_dim,
                strategy,
            );
        }
        // Bridge formats (no integer dot): dequantize-and-dot per row, per token.
        WeightMatrix::Quant(fmt, blocks) => {
            for b in 0..batch {
                crate::threading::par_matvec_quant(
                    pool,
                    *fmt,
                    blocks,
                    &x[b * x_stride..b * x_stride + in_dim],
                    &mut y[b * out_dim..(b + 1) * out_dim],
                    out_dim,
                    in_dim,
                    strategy,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference MoE: the definition, written as directly as possible. Every
    /// performance trick in `forward` (gather, batch-by-expert, skip-empty,
    /// fused SwiGLU) must reproduce this exactly.
    ///
    /// Note the SiLU below uses `f32::exp`, not `kernels::fast_exp`, and must
    /// keep doing so: the point of a reference is to be independently right.
    /// Sharing `fast_exp` with the code under test would make the pair agree
    /// by construction and stop testing the approximation at all.
    fn reference_forward(layer: &MoELayer, input: &[f32], n_tokens: usize) -> Vec<f32> {
        let c = &layer.config;
        let (h, f) = (c.hidden_size, c.expert_ffn_size);
        let mut out = vec![0f32; n_tokens * h];

        for t in 0..n_tokens {
            let x = &input[t * h..(t + 1) * h];
            let mut scores = vec![0f32; c.num_experts];
            for e in 0..c.num_experts {
                scores[e] = (0..h).map(|i| layer.router[e * h + i] * x[i]).sum();
            }
            let (ids, wts) = softmax_top_k(&mut scores, c.num_experts_per_tok, c.norm_topk_prob);

            for k in 0..c.num_experts_per_tok {
                let e = ids[k] as usize;
                let (gate_w, up_w) = split_gate_up(&layer.experts[e].gate_up, f, h);
                let down_w = layer.experts[e].w_down.as_f32().unwrap();

                let mut act = vec![0f32; f];
                for o in 0..f {
                    let g: f32 = (0..h).map(|i| gate_w[o * h + i] * x[i]).sum();
                    let u: f32 = (0..h).map(|i| up_w[o * h + i] * x[i]).sum();
                    act[o] = g / (1.0 + (-g).exp()) * u;
                }
                for o in 0..h {
                    let d: f32 = (0..f).map(|i| down_w[o * f + i] * act[i]).sum();
                    out[t * h + o] += wts[k] * d;
                }
            }
        }
        out
    }

    fn split_gate_up(gu: &GateUp, f: usize, h: usize) -> (Vec<f32>, Vec<f32>) {
        match gu {
            GateUp::Split(g, u) => (g.as_f32().unwrap().to_vec(), u.as_f32().unwrap().to_vec()),
            GateUp::FusedQuant(..) => unreachable!("f32 test fixture"),
        }
    }

    /// Deterministic pseudo-random f32 in [-1, 1). No rand dep.
    fn prng(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    fn make_layer(num_experts: usize, top_k: usize, h: usize, f: usize, norm: bool) -> MoELayer {
        let mut seed = 0x5EEDu64;
        let router = (0..num_experts * h).map(|_| prng(&mut seed)).collect();
        let experts = (0..num_experts)
            .map(|_| ExpertWeights {
                gate_up: GateUp::Split(
                    WeightMatrix::F32((0..f * h).map(|_| prng(&mut seed)).collect()),
                    WeightMatrix::F32((0..f * h).map(|_| prng(&mut seed)).collect()),
                ),
                w_down: WeightMatrix::F32((0..h * f).map(|_| prng(&mut seed)).collect()),
            })
            .collect();
        MoELayer {
            config: MoEConfig {
                num_experts,
                num_experts_per_tok: top_k,
                expert_ffn_size: f,
                hidden_size: h,
                norm_topk_prob: norm,
            },
            router,
            experts,
        }
    }

    fn make_input(n_tokens: usize, h: usize) -> Vec<f32> {
        let mut seed = 0xC0FFEEu64;
        (0..n_tokens * h).map(|_| prng(&mut seed)).collect()
    }

    #[test]
    fn routing_selects_correct_top_k() {
        let layer = make_layer(16, 4, 32, 8, true);
        let input = make_input(5, 32);
        let r = layer.compute_routing(&input, 5);

        for t in 0..5 {
            let ids = &r.expert_ids[t][..4];
            // Exactly top_k distinct experts, all in range.
            let mut sorted = ids.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), 4, "token {t}: duplicate expert selected");
            assert!(ids.iter().all(|&e| (e as usize) < 16));

            // Selected weights are descending — the partial sort's contract.
            let w = &r.weights[t][..4];
            for k in 1..4 {
                assert!(w[k] <= w[k - 1], "token {t}: weights not descending");
            }
        }
    }

    #[test]
    fn routing_picks_the_actual_largest_scores() {
        // Independently recompute the router scores and confirm the chosen
        // experts really are the top-k — a partial sort that silently picked
        // the wrong ones would still pass a "distinct and descending" check.
        let (ne, k, h) = (16usize, 4usize, 32usize);
        let layer = make_layer(ne, k, h, 8, false);
        let input = make_input(3, h);
        let r = layer.compute_routing(&input, 3);

        for t in 0..3 {
            let x = &input[t * h..(t + 1) * h];
            let mut scored: Vec<(usize, f32)> = (0..ne)
                .map(|e| (e, (0..h).map(|i| layer.router[e * h + i] * x[i]).sum()))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            let want: Vec<u32> = scored[..k].iter().map(|&(e, _)| e as u32).collect();
            assert_eq!(&r.expert_ids[t][..k], &want[..], "token {t}");
        }
    }

    #[test]
    fn routing_weights_sum_to_one() {
        let layer = make_layer(32, 8, 32, 8, /* norm_topk_prob */ true);
        let input = make_input(6, 32);
        let r = layer.compute_routing(&input, 6);
        for t in 0..6 {
            let s: f32 = r.weights[t][..8].iter().sum();
            assert!((s - 1.0).abs() < 1e-5, "token {t}: weights sum to {s}, want 1");
        }
    }

    #[test]
    fn routing_weights_unnormalized_sum_below_one() {
        // Without renorm the top_k mass is only part of the full softmax, so
        // it must sum to strictly less than 1 (the other experts hold the rest).
        let layer = make_layer(32, 8, 32, 8, /* norm_topk_prob */ false);
        let input = make_input(4, 32);
        let r = layer.compute_routing(&input, 4);
        for t in 0..4 {
            let s: f32 = r.weights[t][..8].iter().sum();
            assert!(s > 0.0 && s < 1.0, "token {t}: unnormalized sum {s}");
        }
    }

    #[test]
    fn expert_load_populated() {
        let (n_tokens, k) = (12usize, 4usize);
        let layer = make_layer(16, k, 32, 8, true);
        let input = make_input(n_tokens, 32);
        let r = layer.compute_routing(&input, n_tokens);

        assert_eq!(r.expert_load.len(), 16);
        // Every token contributes exactly top_k assignments.
        let total: usize = r.expert_load.iter().sum();
        assert_eq!(total, n_tokens * k);
        assert!(r.expert_load.iter().any(|&c| c > 0));
    }

    #[test]
    fn forward_output_shape_correct() {
        let layer = make_layer(8, 2, 16, 8, true);
        let input = make_input(3, 16);
        let mut out = vec![f32::NAN; 3 * 16];
        let pool = ThreadPool::new(2);
        layer.forward(&input, &mut out, 3, &pool, SimdStrategy::Scalar);
        assert_eq!(out.len(), 3 * 16);
        assert!(out.iter().all(|v| v.is_finite()), "output has NaN/inf");
    }

    /// The load-bearing test: the whole gather / batch-by-expert / skip-empty /
    /// scatter-accumulate machine must equal the naive definition.
    #[test]
    fn forward_matches_naive_reference_decode() {
        let layer = make_layer(16, 4, 32, 24, true);
        let input = make_input(1, 32); // n_tokens = 1 -> matvec path
        let pool = ThreadPool::new(2);
        let mut got = vec![0f32; 32];
        layer.forward(&input, &mut got, 1, &pool, SimdStrategy::Scalar);
        let want = reference_forward(&layer, &input, 1);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // Relative tolerance: `forward` and the reference sum the same
            // products in different orders (batched/threaded vs sequential),
            // so f32 rounding diverges in the last ulp or two. Outputs here
            // reach the hundreds, where a fixed 1e-4 absolute bound is below
            // the representable spacing. Matches `threading.rs`'s convention.
            let tol = w.abs().max(1.0) * 1e-5;
            assert!((g - w).abs() < tol, "i={i}: got {g}, want {w}");
        }
    }

    #[test]
    fn forward_matches_naive_reference_prefill_batch() {
        // n_tokens > 1 -> matmul path, tokens grouped by expert, and with 16
        // experts x top-4 over 9 tokens some experts get 0 tokens (skipped)
        // and some get several (batched). Both branches exercised.
        let layer = make_layer(16, 4, 32, 24, true);
        let n = 9;
        let input = make_input(n, 32);
        let pool = ThreadPool::new(2);
        let mut got = vec![0f32; n * 32];
        let r = layer.forward(&input, &mut got, n, &pool, SimdStrategy::Scalar);
        let want = reference_forward(&layer, &input, n);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // Relative tolerance: `forward` and the reference sum the same
            // products in different orders (batched/threaded vs sequential),
            // so f32 rounding diverges in the last ulp or two. Outputs here
            // reach the hundreds, where a fixed 1e-4 absolute bound is below
            // the representable spacing. Matches `threading.rs`'s convention.
            let tol = w.abs().max(1.0) * 1e-5;
            assert!((g - w).abs() < tol, "i={i}: got {g}, want {w}");
        }
        // Confirm the skip path really was taken.
        assert!(
            r.expert_load.iter().any(|&c| c == 0),
            "test is not exercising the empty-expert skip"
        );
    }

    #[test]
    fn forward_matches_reference_without_renorm() {
        let layer = make_layer(8, 2, 16, 12, /* norm_topk_prob */ false);
        let input = make_input(4, 16);
        let pool = ThreadPool::new(2);
        let mut got = vec![0f32; 4 * 16];
        layer.forward(&input, &mut got, 4, &pool, SimdStrategy::Scalar);
        let want = reference_forward(&layer, &input, 4);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // Relative tolerance: `forward` and the reference sum the same
            // products in different orders (batched/threaded vs sequential),
            // so f32 rounding diverges in the last ulp or two. Outputs here
            // reach the hundreds, where a fixed 1e-4 absolute bound is below
            // the representable spacing. Matches `threading.rs`'s convention.
            let tol = w.abs().max(1.0) * 1e-5;
            assert!((g - w).abs() < tol, "i={i}: got {g}, want {w}");
        }
    }

    /// Qwen3-30B-A3B's real routing shape. Guards against a top_k=2 (Mixtral)
    /// assumption creeping in, and against MAX_TOP_K overflow at k=8.
    #[test]
    fn qwen3_shape_128_experts_top_8() {
        let layer = make_layer(128, 8, 32, 16, true);
        assert!(layer.config.validate().is_ok());
        let input = make_input(2, 32);
        let pool = ThreadPool::new(2);
        let mut got = vec![0f32; 2 * 32];
        let r = layer.forward(&input, &mut got, 2, &pool, SimdStrategy::Scalar);

        assert_eq!(r.expert_load.iter().sum::<usize>(), 2 * 8);
        let want = reference_forward(&layer, &input, 2);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // Relative tolerance: `forward` and the reference sum the same
            // products in different orders (batched/threaded vs sequential),
            // so f32 rounding diverges in the last ulp or two. Outputs here
            // reach the hundreds, where a fixed 1e-4 absolute bound is below
            // the representable spacing. Matches `threading.rs`'s convention.
            let tol = w.abs().max(1.0) * 1e-5;
            assert!((g - w).abs() < tol, "i={i}: got {g}, want {w}");
        }
    }

    #[test]
    fn config_validate_rejects_bad_shapes() {
        let base = MoEConfig {
            num_experts: 128,
            num_experts_per_tok: 8,
            expert_ffn_size: 768,
            hidden_size: 2048,
            norm_topk_prob: true,
        };
        assert!(base.validate().is_ok());

        let mut c = base.clone();
        c.num_experts_per_tok = 0;
        assert!(c.validate().is_err(), "top_k=0 must be rejected");

        let mut c = base.clone();
        c.num_experts_per_tok = 200; // > num_experts and > MAX_TOP_K
        assert!(c.validate().is_err(), "top_k > num_experts must be rejected");

        let mut c = base.clone();
        c.num_experts_per_tok = MAX_TOP_K + 1;
        assert!(c.validate().is_err(), "top_k > MAX_TOP_K must be rejected");

        let mut c = base;
        c.num_experts = 0;
        assert!(c.validate().is_err(), "0 experts must be rejected");
    }
}
