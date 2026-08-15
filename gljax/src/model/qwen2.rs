//! Qwen2 transformer, traced end to end.
//!
//! This is the first place the ops in [`crate::ops`] are composed into
//! something that could actually run a model, and it is what Waves A4/A5 build
//! a `Session` around.
//!
//! # ⛔ Qwen2 has QKV biases
//!
//! ARTX03 never mentions them: its `gqa_attention` takes q/k/v and nothing
//! else, and its end-to-end trace declares four attention weights. Qwen2 adds a
//! bias to each of the query, key and value projections — glproc loads exactly
//! these three (`glproc/src/loader.rs:638`, `attn_q.bias` / `attn_k.bias` /
//! `attn_v.bias`, optional and "qwen2-style" per `model.rs:129`).
//!
//! Dropping them costs nothing structurally — same shapes, same op count, a
//! model that still emits fluent text — and is wrong at every layer. P4 again.
//! [`Qwen2Config::attn_bias`] controls it and defaults to `true` for Qwen2.

use crate::graph::{BuiltFunc, TraceCx};
use crate::ops::rope::QWEN2_ROPE_BASE;
use crate::ops::util::scalar_const;
use crate::ops::{
    causal_mask, causal_mask_row, emit_rope_tables, gather_embed, gqa_attention, kv_cache, linear,
    rms_norm, rope_neox, rope_neox_at, swiglu_ffn,
};
use crate::stablehlo::ops::DotDimensionNumbers;
use crate::stablehlo::types::{DType, Shape};
use crate::tensor::Tensor;
use crate::{precision, GlError};

/// Architecture hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen2Config {
    pub hidden: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub ffn: usize,
    pub vocab: usize,
    pub rms_eps: f64,
    pub rope_base: f32,
    pub max_seq_len: usize,
    /// Qwen2 ties the LM head to the embedding table.
    pub tie_word_embeddings: bool,
    /// Qwen2 biases the q/k/v projections. See the module docs.
    pub attn_bias: bool,
}

impl Qwen2Config {
    /// Qwen2-0.5B — the sprint's target model.
    ///
    /// ⚠️ `head_dim` is 64 and `n_kv_heads` is 2, giving a GQA repeat of 7.
    /// ARTX03 §4 states Qwen2-0.5B is `n_heads=16, n_kv_heads=8, head_dim=64`
    /// and calls it MHA; the sprint brief states `n_heads=14, n_kv_heads=2`.
    /// The brief matches the published config (hidden 896 = 14 × 64), so that
    /// is what is used here — and it means the GQA expansion path is on the
    /// critical path, not a dormant branch.
    pub fn qwen2_0_5b() -> Self {
        Qwen2Config {
            hidden: 896,
            n_layers: 24,
            n_heads: 14,
            n_kv_heads: 2,
            head_dim: 64,
            ffn: 4864,
            vocab: 151_936,
            rms_eps: 1e-6,
            // ⛔ 1e6, not 1e4 — see QWEN2_ROPE_BASE.
            rope_base: QWEN2_ROPE_BASE,
            // config.json says 131072. gljax's largest traceable bucket is
            // 1024 (the causal mask is a dense O(S²) constant), so the real
            // ceiling is the bucket grid, not this.
            max_seq_len: 131_072,
            tie_word_embeddings: true,
            attn_bias: true,
        }
    }

    /// A deliberately tiny model for tests — same structure, shapes that fit in
    /// a assertion message.
    pub fn tiny() -> Self {
        Qwen2Config {
            hidden: 32,
            n_layers: 2,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 8,
            ffn: 64,
            vocab: 128,
            rms_eps: 1e-6,
            rope_base: crate::ops::DEFAULT_ROPE_BASE,
            max_seq_len: 64,
            tie_word_embeddings: true,
            attn_bias: true,
        }
    }

    /// How many query heads share each KV head.
    pub fn gqa_repeat(&self) -> usize {
        self.n_heads / self.n_kv_heads
    }

    fn validate(&self) -> Result<(), GlError> {
        if self.n_kv_heads == 0 || !self.n_heads.is_multiple_of(self.n_kv_heads) {
            return Err(GlError::Engine(format!(
                "qwen2: {} query heads is not a multiple of {} kv heads",
                self.n_heads, self.n_kv_heads
            )));
        }
        if self.n_heads * self.head_dim != self.hidden {
            return Err(GlError::Engine(format!(
                "qwen2: n_heads × head_dim = {} does not match hidden = {}",
                self.n_heads * self.head_dim,
                self.hidden
            )));
        }
        if !self.head_dim.is_multiple_of(2) {
            return Err(GlError::Engine(format!(
                "qwen2: head_dim {} must be even for RoPE",
                self.head_dim
            )));
        }
        Ok(())
    }
}

/// Traces a full forward pass: token ids in, logits out.
///
/// Static shapes throughout (P3): `seq_len` is baked into the compiled
/// artifact, so each bucket is a separate compilation. `seq_offset` is the
/// absolute position of the first token, which is what a decode step past a
/// prefill needs — though without a KV cache (Wave A5) the only sensible value
/// here is 0.
///
/// The emitted signature is `(input_ids, …weights) -> logits`, with weight
/// names in checkpoint form.
pub fn trace_forward(cfg: &Qwen2Config, seq_len: usize, seq_offset: usize) -> Result<BuiltFunc, GlError> {
    cfg.validate()?;
    if seq_offset + seq_len > cfg.max_seq_len {
        return Err(GlError::Engine(format!(
            "qwen2: positions {seq_offset}..{} exceed max_seq_len {}",
            seq_offset + seq_len,
            cfg.max_seq_len
        )));
    }

    let policy = precision::current();
    let act = policy.activation;
    let wt = policy.weight;

    let mut cx = TraceCx::new("main", "qwen2");
    let ids = cx.input("input_ids", Shape::new([1, seq_len], DType::I32));

    let (embed, logits) = {
        let embed = cx.scope("model", |cx| {
            cx.weight("embed_tokens.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        });
        let mut x = gather_embed(&embed, &ids).to_dtype(act);

        // The RoPE table and the causal mask are shared by every layer — one
        // constant each, not one per layer.
        let table_rows = seq_offset + seq_len;
        let (cos, sin) = emit_rope_tables(&x, table_rows, cfg.head_dim, cfg.rope_base)?;
        let mask = causal_mask(&x, seq_len, act)?;

        for layer in 0..cfg.n_layers {
            let (out, _cache_tap) = cx.scope("model", |cx| {
                cx.scope(format!("layers.{layer}"), |cx| {
                    trace_layer(cx, cfg, &x, &cos, &sin, &mask, seq_len, seq_offset, false)
                })
            });
            x = out;
        }

        let final_norm_w = cx.scope("model", |cx| {
            cx.scope("norm", |cx| cx.weight("weight", Shape::new([cfg.hidden], wt)))
        });
        let x = rms_norm(&x, &final_norm_w, cfg.rms_eps);

        // Weight tying: the LM head is the embedding table, contracted on its
        // hidden axis. ARTX01 §7.8 — the same SSA value in both places, so XLA
        // stores the weights once.
        let head = if cfg.tie_word_embeddings {
            embed.clone_ref()
        } else {
            cx.weight("lm_head.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        };
        let logits = x.dot_general(
            &head,
            &DotDimensionNumbers {
                lhs_batching: vec![],
                rhs_batching: vec![],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            },
        );
        (embed, logits)
    };
    drop(embed);

    Ok(cx.finish(&[&logits]))
}

/// Traces a prefill pass that fills a KV cache alongside the ordinary
/// full-sequence forward pass (ARTX05 §1/§4's "prefill function").
///
/// The attention computation is identical to [`trace_forward`]'s — full
/// self-attention over the prompt, the static causal mask, no cache reads —
/// because prefill only ever *writes* the cache; nothing yet in it is read
/// until a decode step runs. `trace_layer`'s `emit_cache_tap` hook taps the
/// post-RoPE K and un-rotated V so the write costs no recomputation.
///
/// `cache_window` is the compiled cache's capacity — a decode bucket
/// (128/256/.../1024) — and must be at least `seq_len`. The same value must be
/// passed to `trace_decode` for the two programs to agree on the cache's
/// layout.
///
/// The emitted signature is
/// `(input_ids, …weights, k_cache, v_cache) -> (logits, k_cache', v_cache')`,
/// with `k_cache'`/`v_cache'` donated (ARTX05 §6) to their matching inputs —
/// see [`crate::graph::builder::FuncBuilder::alias_output`].
pub fn trace_prefill_with_cache(
    cfg: &Qwen2Config,
    seq_len: usize,
    cache_window: usize,
) -> Result<BuiltFunc, GlError> {
    cfg.validate()?;
    if seq_len > cache_window {
        return Err(GlError::Engine(format!(
            "qwen2: a prefill of {seq_len} tokens exceeds the cache window of {cache_window} — \
             the window must cover the whole prompt"
        )));
    }
    if seq_len > cfg.max_seq_len {
        return Err(GlError::Engine(format!(
            "qwen2: {seq_len} positions exceed max_seq_len {}",
            cfg.max_seq_len
        )));
    }

    let policy = precision::current();
    let act = policy.activation;
    let wt = policy.weight;

    let mut cx = TraceCx::new("main", "qwen2_prefill_cache");
    let ids = cx.input("input_ids", Shape::new([1, seq_len], DType::I32));

    let cache_shape = kv_cache::cache_shape(cfg.n_layers, cache_window, cfg.n_kv_heads, cfg.head_dim, act);
    let k_cache_param = cx.kv_cache("k_cache", cache_shape.clone());
    let v_cache_param = cx.kv_cache("v_cache", cache_shape);

    let (embed, logits, k_cache_final, v_cache_final) = {
        let embed = cx.scope("model", |cx| {
            cx.weight("embed_tokens.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        });
        let mut x = gather_embed(&embed, &ids).to_dtype(act);

        let (cos, sin) = emit_rope_tables(&x, seq_len, cfg.head_dim, cfg.rope_base)?;
        let mask = causal_mask(&x, seq_len, act)?;

        // Shared compile-time-constant indices: prefill always starts at
        // position 0 and every non-position start index (kv-head axis,
        // head_dim axis) is 0 too, so one constant covers all of them.
        let zero = scalar_const(&x, 0.0, DType::I32);

        let mut k_cache = k_cache_param.clone_ref();
        let mut v_cache = v_cache_param.clone_ref();

        for layer in 0..cfg.n_layers {
            // The layer axis is a runtime-typed `dynamic_update_slice` index,
            // but its value is known at trace time — one unrolled iteration
            // per layer, so this is a constant, not a fresh input.
            let layer_const = scalar_const(&x, layer as f64, DType::I32);

            let (out, tap) = cx.scope("model", |cx| {
                cx.scope(format!("layers.{layer}"), |cx| {
                    trace_layer(cx, cfg, &x, &cos, &sin, &mask, seq_len, 0, true)
                })
            });
            x = out;

            let (k_ready, v_ready) = tap.expect("emit_cache_tap = true guarantees Some");
            // Defensive: a no-op today (every PrecisionPolicy sets weight ==
            // activation), but dynamic_update_slice requires the cache and
            // the update to share a dtype, so this is the contract, not a
            // guess that happens to hold.
            let k_ready = k_ready.to_dtype(act);
            let v_ready = v_ready.to_dtype(act);

            k_cache = kv_cache::write_range(&k_cache, &k_ready, &layer_const, &zero, &zero);
            v_cache = kv_cache::write_range(&v_cache, &v_ready, &layer_const, &zero, &zero);
        }

        let final_norm_w = cx.scope("model", |cx| {
            cx.scope("norm", |cx| cx.weight("weight", Shape::new([cfg.hidden], wt)))
        });
        let x = rms_norm(&x, &final_norm_w, cfg.rms_eps);

        let head = if cfg.tie_word_embeddings {
            embed.clone_ref()
        } else {
            cx.weight("lm_head.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        };
        let logits = x.dot_general(
            &head,
            &DotDimensionNumbers {
                lhs_batching: vec![],
                rhs_batching: vec![],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            },
        );
        (embed, logits, k_cache, v_cache)
    };
    drop(embed);

    // Outputs are [logits, k_cache', v_cache'] — donate each cache input to
    // its matching output index.
    cx.alias_output(&k_cache_param, 1);
    cx.alias_output(&v_cache_param, 2);

    Ok(cx.finish(&[&logits, &k_cache_final, &v_cache_final]))
}

/// Traces a single decode step: one new token in, next-token logits and the
/// updated KV cache out (ARTX05 §4's "decode function").
///
/// `window` is the compiled cache's capacity, matching whatever
/// `trace_prefill_with_cache` filled. `pos` — the token's **absolute**
/// position — is the one genuine per-step runtime input; everything else
/// that would look like it varies per layer (which layer's cache slot) is a
/// trace-time constant instead, because the model is unrolled one call per
/// layer rather than looped on-device (same shape as `trace_forward`'s own
/// unrolled layer loop).
///
/// The emitted signature is
/// `(input_id, pos, …weights, k_cache, v_cache) -> (logits, k_cache', v_cache')`,
/// with the caches donated exactly as in `trace_prefill_with_cache`.
///
/// Reads the **full** `window`-sized cache every step and masks the unwritten
/// tail with `causal_mask_row` (ARTX05 §2's design decision) rather than a
/// dynamic-shape read — the wasted attention compute over padding positions is
/// the trade that keeps every shape static (P3).
///
/// # ⚠️ Unverified against the recomputation oracle
///
/// This traces, type-checks and parses (see `tools/verify_mlir.py`), but
/// nothing here has run against a real PJRT plugin. ARTX05's headline risk
/// holds: a cache that reads one position off produces fluent, wrong text with
/// no error anywhere. Do not trust this path's *output* until it has been
/// checked token-for-token against `trace_forward`'s recomputation path on the
/// same prompt — see `runtime::cached_session`'s gated parity test.
pub fn trace_decode(cfg: &Qwen2Config, window: usize) -> Result<BuiltFunc, GlError> {
    cfg.validate()?;

    let policy = precision::current();
    let act = policy.activation;
    let wt = policy.weight;

    let mut cx = TraceCx::new("main", "qwen2_decode");
    let ids = cx.input("input_id", Shape::new([1, 1], DType::I32));
    let pos = cx.input("pos", Shape::scalar(DType::I32));

    let cache_shape = kv_cache::cache_shape(cfg.n_layers, window, cfg.n_kv_heads, cfg.head_dim, act);
    let k_cache_param = cx.kv_cache("k_cache", cache_shape.clone());
    let v_cache_param = cx.kv_cache("v_cache", cache_shape);

    let (embed, logits, k_cache_final, v_cache_final) = {
        let embed = cx.scope("model", |cx| {
            cx.weight("embed_tokens.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        });
        let mut x = gather_embed(&embed, &ids).to_dtype(act);

        // The table covers every position the cache can hold; the mask is the
        // same dense [window, window] constant `trace_prefill_with_cache`
        // would build for a full-window prompt — decode just reads one row of
        // it, at a runtime offset, instead of the whole thing.
        let (cos, sin) = emit_rope_tables(&x, window, cfg.head_dim, cfg.rope_base)?;
        let full_mask = causal_mask(&x, window, act)?;
        let zero = scalar_const(&x, 0.0, DType::I32);
        let mask_row = causal_mask_row(&full_mask, &pos, &zero);

        let mut k_cache = k_cache_param.clone_ref();
        let mut v_cache = v_cache_param.clone_ref();

        for layer in 0..cfg.n_layers {
            let layer_const = scalar_const(&x, layer as f64, DType::I32);
            let (out, k_next, v_next) = cx.scope("model", |cx| {
                cx.scope(format!("layers.{layer}"), |cx| {
                    trace_layer_decode(
                        cx, cfg, &x, &cos, &sin, &mask_row, &pos, &zero, &layer_const, &k_cache,
                        &v_cache, window,
                    )
                })
            });
            x = out;
            k_cache = k_next;
            v_cache = v_next;
        }

        let final_norm_w = cx.scope("model", |cx| {
            cx.scope("norm", |cx| cx.weight("weight", Shape::new([cfg.hidden], wt)))
        });
        let x = rms_norm(&x, &final_norm_w, cfg.rms_eps);

        let head = if cfg.tie_word_embeddings {
            embed.clone_ref()
        } else {
            cx.weight("lm_head.weight", Shape::new([cfg.vocab, cfg.hidden], wt))
        };
        let logits = x.dot_general(
            &head,
            &DotDimensionNumbers {
                lhs_batching: vec![],
                rhs_batching: vec![],
                lhs_contracting: vec![2],
                rhs_contracting: vec![1],
            },
        );
        (embed, logits, k_cache, v_cache)
    };
    drop(embed);

    cx.alias_output(&k_cache_param, 1);
    cx.alias_output(&v_cache_param, 2);

    Ok(cx.finish(&[&logits, &k_cache_final, &v_cache_final]))
}

/// Traces one transformer layer.
///
/// `emit_cache_tap` is ARTX05's hook for [`trace_prefill_with_cache`]: when
/// true, the second return value is `Some((k, v))` — this layer's post-RoPE K
/// and un-rotated V, already transposed to the cache's `[1, S, n_kv, head_dim]`
/// layout (position axis before heads, vs. attention's own `[1, n_kv, S,
/// head_dim]`) — ready to hand to `kv_cache::write_range`. [`trace_forward`]
/// passes `false` and discards it: computing that transpose costs two ops per
/// layer that nothing downstream reads, so it is conditional rather than
/// always-on waste in the recomputation oracle's hot path.
#[allow(clippy::too_many_arguments)]
fn trace_layer(
    cx: &mut TraceCx,
    cfg: &Qwen2Config,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mask: &Tensor,
    seq_len: usize,
    seq_offset: usize,
    emit_cache_tap: bool,
) -> (Tensor, Option<(Tensor, Tensor)>) {
    let policy = precision::current();
    let wt = policy.weight;
    let (h, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let kv_width = nkv * hd;

    let ln1 = cx.scope("input_layernorm", |cx| {
        cx.weight("weight", Shape::new([h], wt))
    });
    let normed = rms_norm(x, &ln1, cfg.rms_eps);

    let (attn_out, cache_tap) = cx.scope("self_attn", |cx| {
        // ⛔ HuggingFace layout: `nn.Linear` stores `[out_features, in_features]`.
        // ⚠️ q_proj and o_proj are both [896, 896] on Qwen2-0.5B, so they are
        // orientation-blind — a wrong layout here is invisible to every shape
        // check. Only k/v/gate/up/down are narrow enough to catch it, which is
        // exactly how the real checkpoint reported 120 disagreements while
        // these two were equally wrong. See ops::linear.
        let q_w = cx.weight("q_proj.weight", Shape::new([nh * hd, h], wt));
        let k_w = cx.weight("k_proj.weight", Shape::new([kv_width, h], wt));
        let v_w = cx.weight("v_proj.weight", Shape::new([kv_width, h], wt));
        let o_w = cx.weight("o_proj.weight", Shape::new([h, nh * hd], wt));

        let mut q = linear(&normed, &q_w);
        let mut k = linear(&normed, &k_w);
        let mut v = linear(&normed, &v_w);

        if cfg.attn_bias {
            let q_b = cx.weight("q_proj.bias", Shape::new([nh * hd], wt));
            let k_b = cx.weight("k_proj.bias", Shape::new([kv_width], wt));
            let v_b = cx.weight("v_proj.bias", Shape::new([kv_width], wt));
            q = add_bias(&q, &q_b);
            k = add_bias(&k, &k_b);
            v = add_bias(&v, &v_b);
        }

        // [1, S, n·hd] -> [1, n, S, hd]
        let split = |t: &Tensor, heads: usize| {
            t.reshape(vec![1, seq_len, heads, hd])
                .transpose(vec![0, 2, 1, 3])
        };
        let q = rope_neox(&split(&q, nh), cos, sin, seq_offset);
        let k = rope_neox(&split(&k, nkv), cos, sin, seq_offset);
        let v = split(&v, nkv);

        // ⚠️ RoPE is applied to Q and K only. Rotating V is a classic silent
        // error: shapes match, output stays fluent.
        let attn = gqa_attention(&q, &k, &v, mask);

        let merged = attn
            .transpose(vec![0, 2, 1, 3])
            .reshape(vec![1, seq_len, nh * hd]);

        // [1, n_kv, S, hd] -> [1, S, n_kv, hd], matching kv_cache's layout.
        let cache_tap = emit_cache_tap
            .then(|| (k.transpose(vec![0, 2, 1, 3]), v.transpose(vec![0, 2, 1, 3])));

        (linear(&merged, &o_w), cache_tap)
    });

    let h1 = x + &attn_out;

    let ln2 = cx.scope("post_attention_layernorm", |cx| {
        cx.weight("weight", Shape::new([h], wt))
    });
    let normed2 = rms_norm(&h1, &ln2, cfg.rms_eps);

    let mlp_out = cx.scope("mlp", |cx| {
        // HuggingFace layout again: [out, in].
        let gate = cx.weight("gate_proj.weight", Shape::new([cfg.ffn, h], wt));
        let up = cx.weight("up_proj.weight", Shape::new([cfg.ffn, h], wt));
        let down = cx.weight("down_proj.weight", Shape::new([h, cfg.ffn], wt));
        swiglu_ffn(&normed2, &gate, &up, &down)
    });

    (&h1 + &mlp_out, cache_tap)
}

/// Traces one transformer layer for a decode step: `x` is `[1, 1, hidden]`,
/// RoPE reads the table at a runtime `pos`, and attention reads/writes this
/// layer's slot of a running KV cache instead of self-attending over a
/// freshly-computed K/V window.
///
/// Kept as a separate function from `trace_layer` rather than a third
/// branch on it — ARTX05 itself makes the same call for `gqa_attention_decode`
/// vs `gqa_attention` — because the attention body genuinely differs (cache
/// read/write, a sliced mask row, a dynamic RoPE offset), not just a flag's
/// worth of behavior.
///
/// Returns `(hidden_out, k_cache_updated, v_cache_updated)` — unlike
/// `trace_layer`'s optional tap, decode always touches the cache, so there is
/// no `Option` here.
#[allow(clippy::too_many_arguments)]
fn trace_layer_decode(
    cx: &mut TraceCx,
    cfg: &Qwen2Config,
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mask_row: &Tensor,
    pos: &Tensor,
    zero: &Tensor,
    layer: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    window: usize,
) -> (Tensor, Tensor, Tensor) {
    let policy = precision::current();
    let wt = policy.weight;
    let act = policy.activation;
    let (h, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let kv_width = nkv * hd;

    let ln1 = cx.scope("input_layernorm", |cx| {
        cx.weight("weight", Shape::new([h], wt))
    });
    let normed = rms_norm(x, &ln1, cfg.rms_eps);

    let (attn_out, k_cache_next, v_cache_next) = cx.scope("self_attn", |cx| {
        let q_w = cx.weight("q_proj.weight", Shape::new([nh * hd, h], wt));
        let k_w = cx.weight("k_proj.weight", Shape::new([kv_width, h], wt));
        let v_w = cx.weight("v_proj.weight", Shape::new([kv_width, h], wt));
        let o_w = cx.weight("o_proj.weight", Shape::new([h, nh * hd], wt));

        let mut q = linear(&normed, &q_w);
        let mut k = linear(&normed, &k_w);
        let mut v = linear(&normed, &v_w);

        if cfg.attn_bias {
            let q_b = cx.weight("q_proj.bias", Shape::new([nh * hd], wt));
            let k_b = cx.weight("k_proj.bias", Shape::new([kv_width], wt));
            let v_b = cx.weight("v_proj.bias", Shape::new([kv_width], wt));
            q = add_bias(&q, &q_b);
            k = add_bias(&k, &k_b);
            v = add_bias(&v, &v_b);
        }

        // [1, 1, n·hd] -> [1, n, 1, hd] — one token, so S is always 1 here.
        let split = |t: &Tensor, heads: usize| {
            t.reshape(vec![1, 1, heads, hd]).transpose(vec![0, 2, 1, 3])
        };
        let q = rope_neox_at(&split(&q, nh), cos, sin, pos, zero);
        let k_new = rope_neox_at(&split(&k, nkv), cos, sin, pos, zero);
        let v_new = split(&v, nkv);

        // Write this step's K/V into the cache, [1, n_kv, 1, hd] -> [1, 1,
        // n_kv, hd] to match the cache's position-before-heads layout.
        let k_new_cache = k_new.to_dtype(act).transpose(vec![0, 2, 1, 3]);
        let v_new_cache = v_new.to_dtype(act).transpose(vec![0, 2, 1, 3]);
        let k_cache_next = kv_cache::write_at(k_cache, &k_new_cache, layer, pos, zero);
        let v_cache_next = kv_cache::write_at(v_cache, &v_new_cache, layer, pos, zero);

        // Read the full window back (including the position just written) and
        // restore attention's [1, n_kv, S, hd] layout.
        let k_window = kv_cache::read_window(&k_cache_next, layer, zero, zero, window)
            .transpose(vec![0, 2, 1, 3]);
        let v_window = kv_cache::read_window(&v_cache_next, layer, zero, zero, window)
            .transpose(vec![0, 2, 1, 3]);

        // ⚠️ Same silent-error shape as trace_layer: RoPE only ever touches Q
        // and the newly-written K, never V.
        let attn = gqa_attention(&q, &k_window, &v_window, mask_row);

        let merged = attn.transpose(vec![0, 2, 1, 3]).reshape(vec![1, 1, nh * hd]);
        (linear(&merged, &o_w), k_cache_next, v_cache_next)
    });

    let h1 = x + &attn_out;

    let ln2 = cx.scope("post_attention_layernorm", |cx| {
        cx.weight("weight", Shape::new([h], wt))
    });
    let normed2 = rms_norm(&h1, &ln2, cfg.rms_eps);

    let mlp_out = cx.scope("mlp", |cx| {
        let gate = cx.weight("gate_proj.weight", Shape::new([cfg.ffn, h], wt));
        let up = cx.weight("up_proj.weight", Shape::new([cfg.ffn, h], wt));
        let down = cx.weight("down_proj.weight", Shape::new([h, cfg.ffn], wt));
        swiglu_ffn(&normed2, &gate, &up, &down)
    });

    (&h1 + &mlp_out, k_cache_next, v_cache_next)
}

/// Adds a `[N]` bias to a `[..., N]` activation.
fn add_bias(x: &Tensor, bias: &Tensor) -> Tensor {
    let last = x.rank() - 1;
    let b = bias
        .to_dtype(x.dtype())
        .broadcast_to(vec![last], x.shape().dims.clone());
    x.add(&b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrecisionPolicy;

    #[test]
    fn qwen2_0_5b_config_is_internally_consistent() {
        let cfg = Qwen2Config::qwen2_0_5b();
        cfg.validate().expect("published config must validate");
        assert_eq!(cfg.n_heads * cfg.head_dim, cfg.hidden);
        assert_eq!(cfg.gqa_repeat(), 7, "14 query heads over 2 kv heads");
    }

    #[test]
    fn config_refuses_head_counts_that_do_not_divide() {
        let mut cfg = Qwen2Config::tiny();
        cfg.n_kv_heads = 3;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_refuses_a_hidden_size_that_disagrees_with_the_heads() {
        let mut cfg = Qwen2Config::tiny();
        cfg.hidden = 33;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn forward_returns_logits_over_the_vocabulary() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 8, 0))
            .expect("trace");
        assert_eq!(built.signature.outputs[0].dims, vec![1, 8, cfg.vocab]);
        assert_eq!(built.signature.inputs.len(), 1);
        assert_eq!(built.signature.inputs[0].name, "input_ids");
    }

    #[test]
    fn weight_names_match_the_qwen2_checkpoint_layout() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
            .expect("trace");
        let names: Vec<&str> = built
            .signature
            .weights
            .iter()
            .map(|w| w.name.as_str())
            .collect();

        assert_eq!(names[0], "model.embed_tokens.weight");
        for key in [
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.0.self_attn.q_proj.bias",
            "model.layers.0.self_attn.k_proj.bias",
            "model.layers.0.self_attn.v_proj.bias",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.1.self_attn.q_proj.weight",
            "model.norm.weight",
        ] {
            assert!(names.contains(&key), "missing {key} in {names:?}");
        }
        // Tied embeddings: no separate lm_head.
        assert!(!names.iter().any(|n| n.starts_with("lm_head")));
    }

    #[test]
    fn attention_biases_can_be_switched_off_for_non_qwen_models() {
        let mut cfg = Qwen2Config::tiny();
        cfg.attn_bias = false;
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
            .expect("trace");
        assert!(
            !built
                .signature
                .weights
                .iter()
                .any(|w| w.name.ends_with("q_proj.bias")),
            "attn_bias = false must not declare bias weights"
        );
    }

    /// Weight tying must reuse the *same* SSA parameter, not declare a second
    /// one — otherwise the checkpoint loader uploads 272 MB twice.
    #[test]
    fn tied_embeddings_declare_the_table_once() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
            .expect("trace");
        let embed_count = built
            .signature
            .weights
            .iter()
            .filter(|w| w.name == "model.embed_tokens.weight")
            .count();
        assert_eq!(embed_count, 1);
    }

    #[test]
    fn untied_models_declare_a_separate_lm_head() {
        let mut cfg = Qwen2Config::tiny();
        cfg.tie_word_embeddings = false;
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
            .expect("trace");
        assert!(built
            .signature
            .weights
            .iter()
            .any(|w| w.name == "lm_head.weight"));
    }

    #[test]
    fn each_layer_emits_the_expected_op_counts() {
        let mut cfg = Qwen2Config::tiny();
        cfg.n_layers = 1;
        let built = crate::with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0))
            .expect("trace");
        let mlir = &built.mlir;

        // Per layer: q, k, v, o, gate, up, down (7 projections) plus the two
        // inside attention (QKᵀ and ·V) = 9. Plus the lm_head = 10.
        assert_eq!(mlir.matches("stablehlo.dot_general").count(), 10, "{mlir}");
        // One embedding gather.
        assert_eq!(mlir.matches(r#""stablehlo.gather""#).count(), 1, "{mlir}");
        // RoPE runs on Q and K, never on V — 2 negates, 2 concatenates.
        assert_eq!(mlir.matches(r#""stablehlo.negate""#).count(), 2, "{mlir}");
        assert_eq!(mlir.matches(r#""stablehlo.concatenate""#).count(), 2, "{mlir}");
        // 3 RMSNorms (2 in the layer + the final one), each with one rsqrt.
        assert_eq!(mlir.matches(r#""stablehlo.rsqrt""#).count(), 3, "{mlir}");
        // One softmax → one logistic only from SiLU.
        assert_eq!(mlir.matches(r#""stablehlo.logistic""#).count(), 1, "{mlir}");
    }

    #[test]
    fn positions_past_max_seq_len_are_refused() {
        let cfg = Qwen2Config::tiny();
        let err = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_forward(&cfg, 8, cfg.max_seq_len)
        })
        .expect_err("must refuse");
        assert!(err.to_string().contains("max_seq_len"), "{err}");
    }

    // ── trace_prefill_with_cache (ARTX05) ────────────────────────────────────

    #[test]
    fn prefill_with_cache_declares_two_kv_caches_and_three_outputs() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 4, 8)
        })
        .expect("trace");

        let kv = &built.signature.kv_caches;
        assert_eq!(kv.len(), 2, "k_cache and v_cache");
        assert_eq!(kv[0].name, "k_cache");
        assert_eq!(kv[1].name, "v_cache");
        let expected_dims = vec![cfg.n_layers, 8, cfg.n_kv_heads, cfg.head_dim];
        assert_eq!(kv[0].shape.dims, expected_dims);
        assert_eq!(kv[1].shape.dims, expected_dims);

        assert_eq!(built.signature.outputs.len(), 3, "logits + two updated caches");
        assert_eq!(built.signature.outputs[0].dims, vec![1, 4, cfg.vocab]);
        assert_eq!(built.signature.outputs[1].dims, expected_dims);
        assert_eq!(built.signature.outputs[2].dims, expected_dims);
    }

    #[test]
    fn prefill_with_cache_donates_both_caches_to_their_matching_outputs() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 4, 8)
        })
        .expect("trace");

        assert_eq!(built.signature.kv_caches[0].alias_output, Some(1));
        assert_eq!(built.signature.kv_caches[1].alias_output, Some(2));
        assert!(
            built.mlir.contains("{tf.aliasing_output = 1 : i32}"),
            "{}",
            built.mlir
        );
        assert!(
            built.mlir.contains("{tf.aliasing_output = 2 : i32}"),
            "{}",
            built.mlir
        );
    }

    #[test]
    fn prefill_with_cache_writes_the_whole_prompt_in_one_update_per_layer_per_tensor() {
        let mut cfg = Qwen2Config::tiny();
        cfg.n_layers = 2;
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 4, 8)
        })
        .expect("trace");
        // One dynamic_update_slice per layer per tensor (K and V): 2 layers x 2 = 4.
        assert_eq!(
            built
                .mlir
                .matches(r#""stablehlo.dynamic_update_slice""#)
                .count(),
            4,
            "{}",
            built.mlir
        );
        // No dynamic_slice — prefill never reads the cache back.
        assert_eq!(
            built.mlir.matches(r#""stablehlo.dynamic_slice""#).count(),
            0,
            "{}",
            built.mlir
        );
    }

    #[test]
    fn prefill_with_cache_refuses_a_prompt_wider_than_the_cache_window() {
        let cfg = Qwen2Config::tiny();
        let err = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 9, 8)
        })
        .expect_err("must refuse");
        assert!(err.to_string().contains("cache window"), "{err}");
    }

    /// A prompt shorter than the window is the normal case — most prompts do
    /// not exactly fill their bucket.
    #[test]
    fn prefill_with_cache_accepts_a_prompt_shorter_than_the_cache_window() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 3, 8)
        })
        .expect("a prompt shorter than the window must trace");
        assert_eq!(built.signature.outputs[0].dims, vec![1, 3, cfg.vocab]);
    }

    // ── trace_decode (ARTX05) ─────────────────────────────────────────────────

    #[test]
    fn decode_declares_a_single_token_input_a_runtime_pos_and_two_kv_caches() {
        let cfg = Qwen2Config::tiny();
        let built =
            crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 8)).expect("trace");

        assert_eq!(built.signature.inputs.len(), 2, "input_id and pos");
        assert_eq!(built.signature.inputs[0].name, "input_id");
        assert_eq!(built.signature.inputs[0].shape.dims, vec![1, 1]);
        assert_eq!(built.signature.inputs[1].name, "pos");
        assert_eq!(built.signature.inputs[1].shape.dims, Vec::<usize>::new());

        let kv = &built.signature.kv_caches;
        assert_eq!(kv.len(), 2);
        assert_eq!(kv[0].name, "k_cache");
        assert_eq!(kv[1].name, "v_cache");
        let expected_dims = vec![cfg.n_layers, 8, cfg.n_kv_heads, cfg.head_dim];
        assert_eq!(kv[0].shape.dims, expected_dims);
        assert_eq!(kv[1].shape.dims, expected_dims);
    }

    #[test]
    fn decode_outputs_single_token_logits_plus_the_updated_caches() {
        let cfg = Qwen2Config::tiny();
        let built =
            crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 8)).expect("trace");

        assert_eq!(built.signature.outputs.len(), 3);
        assert_eq!(built.signature.outputs[0].dims, vec![1, 1, cfg.vocab]);
        let expected_dims = vec![cfg.n_layers, 8, cfg.n_kv_heads, cfg.head_dim];
        assert_eq!(built.signature.outputs[1].dims, expected_dims);
        assert_eq!(built.signature.outputs[2].dims, expected_dims);
    }

    #[test]
    fn decode_donates_both_caches_to_their_matching_outputs() {
        let cfg = Qwen2Config::tiny();
        let built =
            crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 8)).expect("trace");

        assert_eq!(built.signature.kv_caches[0].alias_output, Some(1));
        assert_eq!(built.signature.kv_caches[1].alias_output, Some(2));
        assert!(
            built.mlir.contains("{tf.aliasing_output = 1 : i32}"),
            "{}",
            built.mlir
        );
        assert!(
            built.mlir.contains("{tf.aliasing_output = 2 : i32}"),
            "{}",
            built.mlir
        );
    }

    /// Pins the exact op count per layer: two RoPE calls (q, k) at 2
    /// `dynamic_slice`s each (cos + sin table reads) = 4, two cache reads
    /// (`read_window` for k and v) = 2, for 6 `dynamic_slice`s per layer, plus
    /// one more (not per layer) for the position-mask row. Writes are two
    /// `dynamic_update_slice`s per layer (k and v).
    #[test]
    fn decode_reads_and_writes_the_cache_the_expected_number_of_times() {
        let mut cfg = Qwen2Config::tiny();
        cfg.n_layers = 1;
        let built =
            crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 8)).expect("trace");
        let mlir = &built.mlir;

        assert_eq!(
            mlir.matches(r#""stablehlo.dynamic_update_slice""#).count(),
            2,
            "{mlir}"
        );
        assert_eq!(
            mlir.matches(r#""stablehlo.dynamic_slice""#).count(),
            7,
            "{mlir}"
        );
    }

    #[test]
    fn decode_refuses_a_window_too_large_for_the_dense_mask() {
        let cfg = Qwen2Config::tiny();
        let err = crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 2048))
            .expect_err("must refuse");
        assert!(err.to_string().contains("runtime weight"), "{err}");
    }
}
