//! ARTX12 Part A — model zoo: trace (not execute) a representative block for
//! each [`Architecture`] in the zoo.
//!
//! "Trace, not execute" is the doc's own stated bar for this check — no PJRT
//! plugin needed, so this runs everywhere `cargo test` does. It exists to
//! catch structural bugs (a shape that doesn't compose, an op that panics on
//! a config the Qwen2-only path never exercises) *before* a real second
//! architecture's checkpoint is ever loaded — this environment has no PJRT
//! plugin and no Gemma checkpoint to go further than that (see
//! `gljax::arch`'s module docs for the full honesty note on that gap).

use gljax::arch::{Architecture, EmbeddingKind, FfnKind, NormKind};
use gljax::graph::TraceCx;
use gljax::model::qwen2::Qwen2Config;
use gljax::ops::{
    apply_qk_norm, causal_mask, gather_embed, gather_embed_scaled, geglu_ffn, gqa_attention_with_scale,
    linear, rms_norm, rms_norm_zero_centered, swiglu_ffn,
};
use gljax::stablehlo::types::{DType, Shape};
use gljax::tensor::Tensor;

fn zoo() -> Vec<(&'static str, Architecture)> {
    let qwen_cfg = Qwen2Config::tiny();
    vec![
        ("qwen2-tiny", Architecture::from_qwen2(&qwen_cfg)),
        // GQA, like the real Qwen2-0.5B/Gemma 3 shapes.
        ("gemma3-shaped-gqa", Architecture::gemma3_shaped(4, 2, 8)),
        // MHA degenerate case (n_kv_heads == n_heads) at the same descriptor.
        ("gemma3-shaped-mha", Architecture::gemma3_shaped(4, 4, 8)),
    ]
}

fn norm(x: &Tensor, weight: &Tensor, eps: f64, kind: NormKind) -> Tensor {
    match kind {
        NormKind::RmsNorm { .. } => rms_norm(x, weight, eps),
        NormKind::RmsNormZeroCentered { .. } => rms_norm_zero_centered(x, weight, eps),
    }
}

/// Traces embedding -> one transformer block, dispatching on every field
/// `arch` carries. Not a full model (no RoPE, no `lm_head`) — the point is
/// composability of the descriptor's choices, which is exactly what ARTX12
/// Part A checks and nothing more.
///
/// ⚠️ `arch.attention.layer_pattern` is read nowhere here — per
/// `gljax::arch`'s documented scope decision, `LayerPattern` is data ready to
/// drive a sliding-window mask once tracing is rewired to consume it; this
/// zoo test always uses the full causal mask, which is correct for
/// `LayerPattern::Uniform` and an intentional approximation for
/// `LocalGlobal` (attends over more history than the real architecture
/// would, never less — so this cannot hide a masking bug that lets a layer
/// see the future, only one that makes a local layer needlessly non-local).
fn trace_one_block(arch: &Architecture) -> String {
    let attn = &arch.attention;
    let hidden = attn.n_heads * attn.head_dim;
    let kv_width = attn.n_kv_heads * attn.head_dim;
    let ffn_hidden = hidden * 2;
    let seq_len = 4usize;
    let vocab = 16usize;
    let eps = arch.norm.eps() as f64;

    let mut cx = TraceCx::new("main", "zoo_block");

    let table = cx.weight("embed_tokens.weight", Shape::new([vocab, hidden], DType::F32));
    let ids = cx.input("input_ids", Shape::new([1, seq_len], DType::I32));
    let embedded = match arch.embedding {
        EmbeddingKind::Plain => gather_embed(&table, &ids),
        EmbeddingKind::ScaledBySqrtHidden => gather_embed_scaled(&table, &ids, (hidden as f64).sqrt()),
    };

    let ln1 = cx.weight("input_layernorm.weight", Shape::new([hidden], DType::F32));
    let normed = norm(&embedded, &ln1, eps, arch.norm);

    let q_w = cx.weight("q_proj.weight", Shape::new([hidden, hidden], DType::F32));
    let k_w = cx.weight("k_proj.weight", Shape::new([kv_width, hidden], DType::F32));
    let v_w = cx.weight("v_proj.weight", Shape::new([kv_width, hidden], DType::F32));
    let o_w = cx.weight("o_proj.weight", Shape::new([hidden, hidden], DType::F32));

    let q = linear(&normed, &q_w);
    let k = linear(&normed, &k_w);
    let v = linear(&normed, &v_w);

    let split = |t: &Tensor, heads: usize| {
        t.reshape(vec![1, seq_len, heads, attn.head_dim]).transpose(vec![0, 2, 1, 3])
    };
    let mut q = split(&q, attn.n_heads);
    let mut k = split(&k, attn.n_kv_heads);
    let v = split(&v, attn.n_kv_heads);

    if attn.qk_norm {
        let q_norm_w = cx.weight("q_norm.weight", Shape::new([attn.head_dim], DType::F32));
        let k_norm_w = cx.weight("k_norm.weight", Shape::new([attn.head_dim], DType::F32));
        let (qn, kn) = apply_qk_norm(&q, &k, &q_norm_w, &k_norm_w, eps);
        q = qn;
        k = kn;
    }

    let mask = causal_mask(&q, seq_len, DType::F32).expect("mask");
    let attn_out = gqa_attention_with_scale(&q, &k, &v, &mask, attn.effective_query_scale());
    let merged = attn_out.transpose(vec![0, 2, 1, 3]).reshape(vec![1, seq_len, hidden]);
    let attn_proj = linear(&merged, &o_w);

    let h1 = &embedded + &attn_proj;
    let h1 = if attn.post_norms {
        let w = cx.weight("post_attention_layernorm.weight", Shape::new([hidden], DType::F32));
        norm(&h1, &w, eps, arch.norm)
    } else {
        h1
    };

    let ln2 = cx.weight("pre_ffn_layernorm.weight", Shape::new([hidden], DType::F32));
    let normed2 = norm(&h1, &ln2, eps, arch.norm);

    let gate_w = cx.weight("gate_proj.weight", Shape::new([ffn_hidden, hidden], DType::F32));
    let up_w = cx.weight("up_proj.weight", Shape::new([ffn_hidden, hidden], DType::F32));
    let down_w = cx.weight("down_proj.weight", Shape::new([hidden, ffn_hidden], DType::F32));
    let ffn_out = match arch.ffn {
        FfnKind::SwiGlu => swiglu_ffn(&normed2, &gate_w, &up_w, &down_w),
        FfnKind::GeGlu { tanh_approx } => {
            geglu_ffn(&normed2, &gate_w, &up_w, &down_w, tanh_approx).expect("geglu_ffn")
        }
    };

    let h2 = &h1 + &ffn_out;
    let h2 = if attn.post_norms {
        let w = cx.weight("post_ffn_layernorm.weight", Shape::new([hidden], DType::F32));
        norm(&h2, &w, eps, arch.norm)
    } else {
        h2
    };

    cx.finish(&[&h2]).mlir
}

#[test]
fn model_zoo_all_trace_one_block_without_panicking() {
    for (name, arch) in zoo() {
        eprintln!("tracing {name}...");
        let mlir = trace_one_block(&arch);
        assert!(mlir.contains("stablehlo.dot_general"), "{name}: no matmuls emitted:\n{mlir}");
        // Balanced braces is a cheap, real syntax sanity check independent of
        // any single op's correctness.
        assert_eq!(
            mlir.matches('{').count(),
            mlir.matches('}').count(),
            "{name}: unbalanced braces:\n{mlir}"
        );
    }
}

#[test]
fn each_zoo_entry_uses_the_activation_its_ffn_kind_selects() {
    for (name, arch) in zoo() {
        let mlir = trace_one_block(&arch);
        match arch.ffn {
            FfnKind::SwiGlu => assert!(
                mlir.contains(r#""stablehlo.logistic""#),
                "{name}: SwiGlu must use logistic (SiLU):\n{mlir}"
            ),
            FfnKind::GeGlu { .. } => assert!(
                mlir.contains(r#""stablehlo.tanh""#),
                "{name}: GeGlu must use tanh (gelu_pytorch_tanh):\n{mlir}"
            ),
        }
    }
}

/// Parameter *names* never reach the emitted MLIR text at all — `cx.weight`
/// tracks them separately for checkpoint binding (ARTX04), and the
/// `func.func @main(...)` signature spells every parameter as a bare `%vN`.
/// Discovered by this file's own first draft, which asserted on
/// `mlir.contains("q_norm.weight")` and failed for the right reason: the
/// substring was never there to find. Structural comparisons below key on
/// parameter *count* and op *count* instead — things that actually vary in
/// the text.
fn param_count(mlir: &str) -> usize {
    let sig_line = mlir.lines().find(|l| l.contains("func.func @main")).expect("signature line");
    sig_line.matches(": tensor<").count()
}

#[test]
fn gqa_repeat_greater_than_one_emits_more_broadcasts_than_mha() {
    // Same architecture family (gemma3_shaped) throughout, varying only
    // n_kv_heads -- isolates the KV-expansion variable instead of comparing
    // two zoo entries that differ in norm/qk_norm/ffn too.
    let mha = Architecture::gemma3_shaped(4, 4, 8);
    let mut gqa = mha.clone();
    gqa.attention.n_kv_heads = 2;
    assert_eq!(gqa.attention.gqa_repeat(), 2);
    assert_eq!(mha.attention.gqa_repeat(), 1);

    let mha_mlir = trace_one_block(&mha);
    let gqa_mlir = trace_one_block(&gqa);
    let mha_broadcasts = mha_mlir.matches(r#""stablehlo.broadcast_in_dim""#).count();
    let gqa_broadcasts = gqa_mlir.matches(r#""stablehlo.broadcast_in_dim""#).count();
    assert!(
        gqa_broadcasts > mha_broadcasts,
        "GQA (repeat=2) must emit more broadcasts than MHA (repeat=1) at an \
         otherwise identical architecture: {gqa_broadcasts} vs {mha_broadcasts}"
    );
}

#[test]
fn qk_norm_adds_exactly_two_weight_parameters() {
    let mut without = Architecture::gemma3_shaped(4, 2, 8);
    without.attention.qk_norm = false;
    let mut with = without.clone();
    with.attention.qk_norm = true;

    let params_without = param_count(&trace_one_block(&without));
    let params_with = param_count(&trace_one_block(&with));
    assert_eq!(
        params_with,
        params_without + 2,
        "qk_norm must add exactly q_norm.weight + k_norm.weight as parameters"
    );
}
