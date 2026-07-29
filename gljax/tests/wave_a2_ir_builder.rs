//! Wave A2 — the IR builder: `FuncBuilder`, `TraceCx`, `Tensor`.
//!
//! Host-only. No PJRT plugin is involved and none is needed: this wave's
//! deliverable is *text*, and every assertion here is about what the emitter
//! produced.
//!
//! ⚠️ Structural assertions are not a verifier. "Contains `dot_general`" says
//! the op was emitted, not that the module parses or that the numbers are
//! right. The MLIR verifier lives inside the PJRT plugin (ARTX01 §2.4), and
//! numerics arrive with ARTX12 Part B. What these tests *can* catch is the
//! class that has already bitten this series twice: dimension numbers that are
//! structurally plausible and semantically transposed.

use gljax::stablehlo::ops::DotDimensionNumbers;
use gljax::{DType, PrecisionPolicy, Shape, Tensor, TraceCx};

// ---------------------------------------------------------------------------
// The three Wave A2 gate tests
// ---------------------------------------------------------------------------

#[test]
fn func_builder_emits_single_matmul() {
    let mut cx = TraceCx::new("main", "single_matmul");
    let x = cx.input("x", Shape::new([4, 8], DType::F32));
    let w = cx.weight("w", Shape::new([8, 16], DType::F32));
    let y = x.matmul(&w);
    assert_eq!(y.shape().dims, vec![4, 16]);

    let built = cx.finish(&[&y]);
    let mlir = &built.mlir;

    assert!(mlir.contains(r#""stablehlo.dot_general""#), "{mlir}");
    // A [4,8] @ [8,16] matmul contracts lhs axis 1 against rhs axis 0 and
    // batches nothing. All four lists are asserted, because three correct ones
    // and a wrong fourth still produces a correctly-shaped result.
    assert!(mlir.contains("lhs_batching_dimensions = [],"), "{mlir}");
    assert!(mlir.contains("rhs_batching_dimensions = [],"), "{mlir}");
    assert!(mlir.contains("lhs_contracting_dimensions = [1],"), "{mlir}");
    assert!(mlir.contains("rhs_contracting_dimensions = [0]"), "{mlir}");
    assert!(
        mlir.contains("(tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>"),
        "{mlir}"
    );
}

#[test]
fn precision_bf16_wraps_weight_in_convert() {
    let mut cx = TraceCx::new("main", "mixed_precision");
    let x = cx.input("x", Shape::new([4, 8], DType::F32));
    let w = cx.weight("w", Shape::new([8, 16], DType::BF16));
    let y = x.matmul(&w);

    let built = cx.finish(&[&y]);
    let mlir = &built.mlir;

    assert!(mlir.contains(r#""stablehlo.convert""#), "{mlir}");
    assert!(mlir.contains("bf16"), "{mlir}");
    // The convert widens the weight to meet the activation. Narrowing the
    // activation instead would make the shape check pass while quietly
    // dropping 16 bits of mantissa — P5.
    assert!(
        mlir.contains("(tensor<8x16xbf16>) -> tensor<8x16xf32>"),
        "the bf16 weight must be widened to f32, not the other way round:\n{mlir}"
    );
    assert_eq!(y.dtype(), DType::F32);
}

#[test]
fn scope_naming_is_hierarchical() {
    let mut cx = TraceCx::new("main", "scoped");
    let (x, w) = {
        let x = cx.input("hidden_states", Shape::new([1, 4, 8], DType::F32));
        let w = cx.scope("model", |cx| {
            cx.scope("layers.0", |cx| {
                cx.scope("self_attn", |cx| {
                    cx.weight("q_proj.weight", Shape::new([8, 8], DType::F32))
                })
            })
        });
        (x, w)
    };
    let y = x.matmul(&w);
    let built = cx.finish(&[&y]);

    assert!(built.mlir.contains("module @scoped"), "{}", built.mlir);

    // ⭐ The point of the scope stack: the traced name IS the safetensors key.
    assert_eq!(
        built.signature.weights[0].name,
        "model.layers.0.self_attn.q_proj.weight"
    );
    // Inputs stay flat — they are the caller's ABI, not checkpoint keys.
    assert_eq!(built.signature.inputs[0].name, "hidden_states");
}

// ---------------------------------------------------------------------------
// Composition — does the layer actually stack up?
// ---------------------------------------------------------------------------

/// A SwiGLU MLP block traced end to end, with the weight names and shapes a
/// real Qwen2 checkpoint uses.
///
/// This is the largest thing Wave A2 can trace: RMSNorm, RoPE and GQA
/// attention need ops that arrive in Wave A3. It still exercises the whole
/// stack — scopes, weight naming, matmul batching against rank-2 weights,
/// SiLU, the gated multiply, and the residual add.
#[test]
fn swiglu_mlp_block_traces_with_checkpoint_exact_weight_names() {
    // Qwen2-0.5B: hidden 896, ffn 4864.
    const B: usize = 1;
    const S: usize = 128;
    const D: usize = 896;
    const FFN: usize = 4864;

    let built = gljax::with_policy(PrecisionPolicy::bf16(), || {
        let mut cx = TraceCx::new("main", "qwen2_mlp");
        let x = cx.input("hidden_states", Shape::new([B, S, D], DType::BF16));
        let residual = x.clone_ref();

        let out = cx.scope("model", |cx| {
            cx.scope("layers.0", |cx| {
                cx.scope("mlp", |cx| {
                    let gate_w = cx.weight("gate_proj.weight", Shape::new([D, FFN], DType::BF16));
                    let up_w = cx.weight("up_proj.weight", Shape::new([D, FFN], DType::BF16));
                    let down_w = cx.weight("down_proj.weight", Shape::new([FFN, D], DType::BF16));

                    let gated = &x.matmul(&gate_w).silu() * &x.matmul(&up_w);
                    gated.matmul(&down_w)
                })
            })
        });

        let y = &residual + &out;
        cx.finish(&[&y])
    });

    // Shapes survive the round trip.
    assert_eq!(built.signature.outputs[0].dims, vec![B, S, D]);

    // Weight keys, in declaration order, exactly as safetensors spells them.
    let names: Vec<&str> = built
        .signature
        .weights
        .iter()
        .map(|w| w.name.as_str())
        .collect();
    assert_eq!(
        names,
        vec![
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
        ]
    );

    let mlir = &built.mlir;
    // Three projections.
    assert_eq!(mlir.matches("stablehlo.dot_general").count(), 3, "{mlir}");
    // SiLU = logistic + multiply; the gated product is a second multiply.
    assert_eq!(mlir.matches(r#""stablehlo.logistic""#).count(), 1, "{mlir}");
    assert_eq!(mlir.matches(r#""stablehlo.multiply""#).count(), 2, "{mlir}");
    // The residual.
    assert_eq!(mlir.matches(r#""stablehlo.add""#).count(), 1, "{mlir}");
    // Uniform BF16 throughout means no reconciliation converts were needed.
    assert!(
        !mlir.contains("stablehlo.convert"),
        "an all-bf16 trace should need no conversions:\n{mlir}"
    );
    // ⛔ [B,S,D] @ [D,FFN] must not batch — ARTX02 §7's matmul would have
    // emitted `lhs_batching = [0]` here and silently contracted the wrong axis.
    assert!(!mlir.contains("lhs_batching_dimensions = [0],"), "{mlir}");
}

/// The same trace under the F64 oracle policy produces an F64 program from
/// unchanged model code (ARTX01 §3.4). If precision were baked into the ops,
/// the oracle would be a second implementation — and therefore a second thing
/// that can be wrong.
#[test]
fn the_same_trace_re_emits_at_f64_under_the_oracle_policy() {
    fn trace(dtype: DType) -> String {
        let mut cx = TraceCx::new("main", "oracle");
        let x = cx.input("x", Shape::new([2, 4], dtype));
        let w = cx.weight("w", Shape::new([4, 4], dtype));
        let y = x.matmul(&w).silu();
        cx.finish(&[&y]).mlir
    }

    let bf16 = gljax::with_policy(PrecisionPolicy::bf16(), || trace(PrecisionPolicy::bf16().activation));
    let f64_ = gljax::with_policy(PrecisionPolicy::f64_oracle(), || {
        trace(PrecisionPolicy::f64_oracle().activation)
    });

    assert!(bf16.contains("bf16") && !bf16.contains("f64"), "{bf16}");
    assert!(f64_.contains("f64") && !f64_.contains("bf16"), "{f64_}");
    // Same op sequence, different types — that is the whole claim.
    assert_eq!(
        bf16.matches("stablehlo.").count(),
        f64_.matches("stablehlo.").count()
    );
}

/// Attention scores are `[B,H,S,D] @ [B,H,D,S]`. This is the shape ARTX11 §7
/// records as a silent-corruption source when the head grouping is wrong, so
/// the batching must cover both leading axes and nothing else.
#[test]
fn batched_attention_score_matmul_batches_both_leading_axes() {
    let mut cx = TraceCx::new("main", "attn_scores");
    let q = cx.input("q", Shape::new([1, 14, 128, 64], DType::F32));
    let k_t = cx.input("k_t", Shape::new([1, 14, 64, 128], DType::F32));
    let scores = q.matmul(&k_t);

    assert_eq!(scores.shape().dims, vec![1, 14, 128, 128]);
    let built = cx.finish(&[&scores]);
    assert!(
        built.mlir.contains("lhs_batching_dimensions = [0, 1],"),
        "{}",
        built.mlir
    );
    assert!(
        built.mlir.contains("lhs_contracting_dimensions = [3],"),
        "{}",
        built.mlir
    );
    assert!(
        built.mlir.contains("rhs_contracting_dimensions = [2]"),
        "{}",
        built.mlir
    );
}

/// An explicit `dot_general` still works when `matmul`'s convention is not what
/// is wanted — ARTX03's GQA path will need this.
#[test]
fn explicit_dot_dimension_numbers_are_honoured_verbatim() {
    let mut cx = TraceCx::new("main", "explicit");
    let a = cx.input("a", Shape::new([2, 3, 4], DType::F32));
    let b = cx.input("b", Shape::new([2, 4, 5], DType::F32));
    let dnums = DotDimensionNumbers {
        lhs_batching: vec![0],
        rhs_batching: vec![0],
        lhs_contracting: vec![2],
        rhs_contracting: vec![1],
    };
    let c = a.dot_general(&b, &dnums);
    assert_eq!(c.shape().dims, vec![2, 3, 5]);
    let built = cx.finish(&[&c]);
    assert!(
        built.mlir.contains("lhs_batching_dimensions = [0],"),
        "{}",
        built.mlir
    );
}

// ---------------------------------------------------------------------------
// Structure of the emitted module
// ---------------------------------------------------------------------------

#[test]
fn emitted_module_is_ascii_brace_balanced_and_newline_terminated() {
    let mut cx = TraceCx::new("main", "structure");
    let x = cx.input("x", Shape::new([2, 6], DType::F32));
    let zero = Tensor::reduce_sum(&x, &[1]);
    let built = cx.finish(&[&zero]);
    let mlir = &built.mlir;

    assert!(mlir.is_ascii(), "{mlir}");
    assert!(!mlir.contains('\0'), "{mlir}");
    assert!(mlir.ends_with("  }\n}\n"), "{mlir}");
    assert_eq!(
        mlir.matches('{').count(),
        mlir.matches('}').count(),
        "unbalanced braces — the reduce region is the usual culprit:\n{mlir}"
    );
}

#[test]
fn every_ssa_name_is_assigned_exactly_once() {
    // SSA means static *single* assignment; a repeated `%vN =` is a name
    // collision that MLIR rejects, and the reduce region's block arguments are
    // where one would most easily creep in.
    let mut cx = TraceCx::new("main", "ssa");
    let x = cx.input("x", Shape::new([2, 6], DType::F32));
    let s = x.reduce_sum(&[1]);
    let y = s.exp();
    let built = cx.finish(&[&y]);

    let mut assigned: Vec<&str> = built
        .mlir
        .lines()
        .filter_map(|l| l.trim().split(" = ").next().filter(|n| n.starts_with("%v")))
        .collect();
    let before = assigned.len();
    assigned.sort_unstable();
    assigned.dedup();
    assert_eq!(
        before,
        assigned.len(),
        "an SSA name was assigned twice:\n{}",
        built.mlir
    );
}
