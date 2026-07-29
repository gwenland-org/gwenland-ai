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
use crate::ops::{
    causal_mask, emit_rope_tables, gather_embed, gqa_attention, linear, rms_norm, rope_neox,
    swiglu_ffn,
};
use crate::ops::rope::QWEN2_ROPE_BASE;
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
            x = cx.scope("model", |cx| {
                cx.scope(format!("layers.{layer}"), |cx| {
                    trace_layer(cx, cfg, &x, &cos, &sin, &mask, seq_len, seq_offset)
                })
            });
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
) -> Tensor {
    let policy = precision::current();
    let wt = policy.weight;
    let (h, hd, nh, nkv) = (cfg.hidden, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads);
    let kv_width = nkv * hd;

    let ln1 = cx.scope("input_layernorm", |cx| {
        cx.weight("weight", Shape::new([h], wt))
    });
    let normed = rms_norm(x, &ln1, cfg.rms_eps);

    let attn_out = cx.scope("self_attn", |cx| {
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
        linear(&merged, &o_w)
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

    &h1 + &mlp_out
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
}
