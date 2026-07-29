//! Wave A3 — the ops layer and the Qwen2 model.
//!
//! # What these tests can and cannot prove
//!
//! Host-only. gljax cannot execute MLIR on this machine (no PJRT plugin for
//! Windows — see `gljax/README.md`), so **nothing here checks a number that
//! came out of the graph**. What is checked:
//!
//! * the emitted module has the ops, in the order, with the shapes intended;
//! * the *scalar references* for the two numerics traps (RMSNorm's ε placement,
//!   NeoX's pairing) behave as documented, taken from llama.cpp and glproc;
//! * the module parses — `gljax/tools/verify_mlir.py` runs it through jaxlib's
//!   MLIR parser, which already caught two bugs these assertions did not.
//!
//! Proving the graph computes the reference is ARTX12 Part B, and needs a
//! plugin. Until then a green run here means "structurally right", not
//! "correct".

use gljax::model::{trace_forward, Qwen2Config};
use gljax::ops::{
    causal_mask, emit_rope_tables, gather_embed, gqa_attention, rms_norm, rope_neox, softmax,
    swiglu_ffn, DEFAULT_ROPE_BASE,
};
use gljax::stablehlo::types::{DType, Shape};
use gljax::{with_policy, PrecisionPolicy, TraceCx};

// ---------------------------------------------------------------------------
// The Wave A3 gate: a full Qwen2 block traces end to end
// ---------------------------------------------------------------------------

#[test]
fn qwen2_block_traces_end_to_end() {
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.n_layers = 1;
    let built =
        with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 128, 0)).expect("trace");
    let mlir = &built.mlir;

    // Every op class the block is built from.
    assert!(mlir.contains(r#""stablehlo.gather""#), "embedding lookup");
    assert!(mlir.contains(r#""stablehlo.dot_general""#), "matmuls");
    assert!(mlir.contains(r#""stablehlo.reduce""#), "rms_norm / softmax");
    assert!(
        mlir.contains(r#""stablehlo.broadcast_in_dim""#),
        "GQA expand + norm broadcast"
    );
    assert!(
        mlir.contains(r#""stablehlo.constant""#),
        "causal mask + rope tables"
    );
    assert!(mlir.contains(r#""stablehlo.rsqrt""#), "rms_norm");
    assert!(mlir.contains(r#""stablehlo.logistic""#), "swiglu");
    assert!(mlir.contains(r#""stablehlo.exponential""#), "softmax");

    // BF16 policy: weights in bf16, reduces upcast to f32.
    assert!(mlir.contains("bf16"), "{mlir:.400}");
    assert!(mlir.contains(r#""stablehlo.convert""#), "precision boundaries");

    // Logits over the full vocabulary.
    assert_eq!(built.signature.outputs[0].dims, vec![1, 128, cfg.vocab]);
    assert_eq!(built.signature.inputs[0].name, "input_ids");
}

/// The signature is the checkpoint contract. If a key is wrong the loader
/// either fails loudly or — worse — binds nothing and leaves a zero tensor.
#[test]
fn qwen2_signature_lists_every_weight_a_layer_needs() {
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.n_layers = 1;
    let built = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 16, 0)).expect("trace");

    let names: Vec<&str> = built
        .signature
        .weights
        .iter()
        .map(|w| w.name.as_str())
        .collect();

    let expected = [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.self_attn.q_proj.bias",
        "model.layers.0.self_attn.k_proj.bias",
        "model.layers.0.self_attn.v_proj.bias",
        "model.layers.0.post_attention_layernorm.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.norm.weight",
    ];
    for key in expected {
        assert!(names.contains(&key), "missing {key}\nhave: {names:#?}");
    }
    assert_eq!(
        names.len(),
        expected.len(),
        "unexpected extra weights: {names:#?}"
    );

    // Shapes the checkpoint must match.
    let by_name = |n: &str| {
        built
            .signature
            .weights
            .iter()
            .find(|w| w.name == n)
            .unwrap_or_else(|| panic!("no {n}"))
    };
    assert_eq!(by_name("model.embed_tokens.weight").shape.dims, vec![151936, 896]);
    assert_eq!(
        by_name("model.layers.0.self_attn.q_proj.weight").shape.dims,
        vec![896, 896]
    );
    // 2 kv heads × 64 = 128, not 896 — the GQA narrowing.
    assert_eq!(
        by_name("model.layers.0.self_attn.k_proj.weight").shape.dims,
        vec![896, 128]
    );
    assert_eq!(
        by_name("model.layers.0.self_attn.k_proj.bias").shape.dims,
        vec![128]
    );
    assert_eq!(
        by_name("model.layers.0.mlp.down_proj.weight").shape.dims,
        vec![4864, 896]
    );
}

// ---------------------------------------------------------------------------
// Numerics traps, against their references
// ---------------------------------------------------------------------------

/// ⭐ ε goes on the mean, inside the sqrt — llama.cpp `ops.cpp:3795`.
///
/// The unit test in `ops/norm.rs` pins the scalar behaviour. This one pins the
/// *emitted order*: reduce → ×(1/D) → +ε → rsqrt.
#[test]
fn rms_norm_epsilon_is_inside_the_sqrt_in_the_emitted_graph() {
    let mut cx = TraceCx::new("main", "rms");
    let x = cx.input("x", Shape::new([1, 4, 8], DType::F32));
    let w = cx.weight("w", Shape::new([8], DType::F32));
    let y = rms_norm(&x, &w, 1e-6);
    let mlir = cx.finish(&[&y]).mlir;

    let pos = |needle: &str| {
        mlir.find(needle)
            .unwrap_or_else(|| panic!("missing {needle} in:\n{mlir}"))
    };
    let reduce = pos(r#""stablehlo.reduce""#);
    let inv_d = pos(&format!("dense<{:?}>", 1.0f64 / 8.0));
    let eps = pos("dense<1.0e-6>");
    let rsqrt = pos(r#""stablehlo.rsqrt""#);

    assert!(reduce < inv_d, "sum before the 1/D scale:\n{mlir}");
    assert!(inv_d < eps, "ε lands on the mean, not on the sum:\n{mlir}");
    assert!(eps < rsqrt, "ε is inside the sqrt:\n{mlir}");
}

/// ⭐ NeoX pairs `(i, i + D/2)`. Settled against `glproc/src/runner.rs:161`,
/// which was validated end-to-end on Qwen2.5-0.5B — not against the series
/// docs, which contradict each other on this point.
#[test]
fn rope_neox_uses_the_half_split_not_adjacent_pairs() {
    const TOL_ROPE: f32 = 1e-5;
    let head_dim = 8;
    let half = head_dim / 2;
    let (cos, sin) = gljax::ops::rope_tables(4, head_dim, DEFAULT_ROPE_BASE);

    // The table must repeat each angle across the half boundary; the adjacent
    // convention would repeat it across `2i`/`2i+1` instead.
    for pos in 0..4 {
        let row = pos * head_dim;
        for i in 0..half {
            assert_eq!(cos[row + i], cos[row + i + half], "pos {pos} lane {i}");
            assert_eq!(sin[row + i], sin[row + i + half], "pos {pos} lane {i}");
        }
    }

    // glproc's rotation, applied at position 3.
    let row = 3 * head_dim;
    let (cos_row, sin_row) = (&cos[row..row + head_dim], &sin[row..row + head_dim]);
    let x: Vec<f32> = (0..head_dim).map(|i| 1.0 + i as f32).collect();
    let mut expected = x.clone();
    for i in 0..half {
        let (x0, x1) = (expected[i], expected[i + half]);
        expected[i] = x0 * cos_row[i] - x1 * sin_row[i];
        expected[i + half] = x0 * sin_row[i] + x1 * cos_row[i];
    }

    // gljax emits `x·cos + rotate_half(x)·sin`; evaluate that form here.
    let rot: Vec<f32> = (0..head_dim)
        .map(|i| if i < half { -x[i + half] } else { x[i - half] })
        .collect();
    let got: Vec<f32> = (0..head_dim)
        .map(|i| x[i] * cos_row[i] + rot[i] * sin_row[i])
        .collect();

    for i in 0..head_dim {
        assert!(
            (expected[i] - got[i]).abs() <= TOL_ROPE,
            "lane {i}: glproc {} vs gljax {}",
            expected[i],
            got[i]
        );
    }
}

/// The emitted RoPE must slice contiguous halves, never stride 2.
#[test]
fn rope_emits_contiguous_half_slices() {
    let mut cx = TraceCx::new("main", "rope");
    let q = cx.input("q", Shape::new([1, 14, 8, 64], DType::F32));
    let (cos, sin) = emit_rope_tables(&q, 64, 64, DEFAULT_ROPE_BASE).expect("tables");
    let out = rope_neox(&q, &cos, &sin, 0);
    let mlir = cx.finish(&[&out]).mlir;

    assert!(
        mlir.contains("start_indices = array<i64: 0, 0, 0, 32>"),
        "the second half must start at head_dim/2:\n{mlir}"
    );
    assert!(
        !mlir.contains("strides = array<i64: 1, 1, 1, 2>"),
        "stride-2 slicing is the *adjacent* convention:\n{mlir}"
    );
}

/// Softmax must subtract the row max before exponentiating, and the reduce
/// init must be −∞.
#[test]
fn softmax_is_the_numerically_stable_form() {
    let mut cx = TraceCx::new("main", "softmax");
    let x = cx.input("x", Shape::new([1, 14, 16, 16], DType::F32));
    let y = softmax(&x, 3);
    let mlir = cx.finish(&[&y]).mlir;

    let max = mlir.find(r#""stablehlo.maximum""#).expect("reduce-max");
    let sub = mlir.find(r#""stablehlo.subtract""#).expect("subtract");
    let exp = mlir.find(r#""stablehlo.exponential""#).expect("exp");
    let div = mlir.find(r#""stablehlo.divide""#).expect("divide");
    assert!(max < sub && sub < exp && exp < div, "{mlir}");
    assert!(mlir.contains("dense<0xFF800000>"), "-inf init:\n{mlir}");
}

// ---------------------------------------------------------------------------
// Individual ops
// ---------------------------------------------------------------------------

#[test]
fn gqa_expands_kv_heads_and_returns_query_shaped_output() {
    let mut cx = TraceCx::new("main", "attn");
    // Qwen2-0.5B: 14 query heads over 2 kv heads.
    let q = cx.input("q", Shape::new([1, 14, 16, 64], DType::F32));
    let k = cx.input("k", Shape::new([1, 2, 16, 64], DType::F32));
    let v = cx.input("v", Shape::new([1, 2, 16, 64], DType::F32));
    let mask = causal_mask(&q, 16, DType::F32).expect("mask");
    let out = gqa_attention(&q, &k, &v, &mask);
    assert_eq!(out.shape().dims, vec![1, 14, 16, 64]);

    let mlir = cx.finish(&[&out]).mlir;
    // The repeat axis is inserted after the kv-head axis, so head h's copies
    // land consecutively and query head q reads kv head q/7.
    assert!(
        mlir.contains("(tensor<1x2x1x16x64xf32>) -> tensor<1x2x7x16x64xf32>"),
        "{mlir}"
    );
    assert!(mlir.contains("dense<0.125>"), "1/sqrt(64) scale:\n{mlir}");
}

#[test]
fn embedding_lookup_and_swiglu_compose_at_qwen2_shapes() {
    let mut cx = TraceCx::new("main", "block");
    let table = cx.weight("model.embed_tokens.weight", Shape::new([151936, 896], DType::F32));
    let ids = cx.input("input_ids", Shape::new([1, 32], DType::I32));
    let x = gather_embed(&table, &ids);
    assert_eq!(x.shape().dims, vec![1, 32, 896]);

    let gate = cx.weight("g", Shape::new([896, 4864], DType::F32));
    let up = cx.weight("u", Shape::new([896, 4864], DType::F32));
    let down = cx.weight("d", Shape::new([4864, 896], DType::F32));
    let y = swiglu_ffn(&x, &gate, &up, &down);
    assert_eq!(y.shape().dims, vec![1, 32, 896]);

    let mlir = cx.finish(&[&y]).mlir;
    assert!(mlir.contains("slice_sizes = array<i64: 1, 896>,"), "{mlir}");
}

#[test]
#[should_panic(expected = "MoE FFN is not implemented")]
fn moe_refuses_rather_than_emitting_an_untested_graph() {
    let mut cx = TraceCx::new("main", "moe");
    let x = cx.input("x", Shape::new([1, 4], DType::F32));
    let r = cx.weight("r", Shape::new([4, 8], DType::F32));
    let _ = gljax::ops::moe::moe_ffn(&x, &r, &[]);
}

// ---------------------------------------------------------------------------
// Constant-size guardrail
// ---------------------------------------------------------------------------

/// ⚠️ The causal mask is O(S²) of MLIR *text*. ARTX03 calls a 512-wide mask
/// "1 MB, acceptable"; ARTX05's 2048 bucket is 34 MB, per bucket. Refusing
/// beats emitting it and wondering why compilation hangs.
#[test]
fn a_2048_bucket_causal_mask_is_refused_with_a_way_forward() {
    let mut cx = TraceCx::new("main", "mask");
    let x = cx.input("x", Shape::new([1], DType::F32));
    let err = causal_mask(&x, 2048, DType::F32).expect_err("must refuse");
    let msg = err.to_string();
    assert!(msg.contains("runtime weight"), "{msg}");

    // And the full trace surfaces that refusal rather than panicking.
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.n_layers = 1;
    let err = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 2048, 0))
        .expect_err("a 2048 bucket cannot be traced with a constant mask yet");
    assert!(err.to_string().contains("runtime weight"), "{err}");
}

/// Buckets up to 1024 do fit, so the sprint's working shapes are unaffected.
#[test]
fn buckets_up_to_1024_trace_within_the_constant_budget() {
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.n_layers = 1;
    for bucket in [128, 256, 512, 1024] {
        let built = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, bucket, 0))
            .unwrap_or_else(|e| panic!("bucket {bucket} failed: {e}"));
        assert_eq!(built.signature.outputs[0].dims, vec![1, bucket, cfg.vocab]);
    }
}

// ---------------------------------------------------------------------------
// Structure
// ---------------------------------------------------------------------------

#[test]
fn the_full_model_emits_unique_ssa_names_and_balanced_braces() {
    let cfg = Qwen2Config::tiny();
    let built = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 8, 0)).expect("trace");
    let mlir = &built.mlir;

    assert_eq!(mlir.matches('{').count(), mlir.matches('}').count(), "braces");

    let mut assigned: Vec<&str> = mlir
        .lines()
        .filter_map(|l| l.trim().split(" = ").next().filter(|n| n.starts_with("%v")))
        .collect();
    let before = assigned.len();
    assigned.sort_unstable();
    assigned.dedup();
    assert_eq!(before, assigned.len(), "an SSA name was assigned twice");
    assert!(before > 100, "a 2-layer model should emit many ops, got {before}");
}
