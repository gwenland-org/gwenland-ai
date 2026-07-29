//! Dumps the StableHLO gljax emits, for inspection or external verification.
//!
//! ```bash
//! cargo run -p gljax --example dump_mlir              # the MLP block, bf16
//! cargo run -p gljax --example dump_mlir -- mlp f64   # same trace, oracle precision
//! cargo run -p gljax --example dump_mlir -- ops       # every Wave A2 op in one module
//! ```
//!
//! ARTX01 §2.3's debugging advice is to print the module and paste it into
//! `stablehlo-opt --verify-diagnostics`. On a machine with no PJRT plugin —
//! see `gljax/README.md` — an external parser is the *only* way to find out
//! whether what gljax emits actually parses, so this exists to make that one
//! command away. `gljax/tools/verify_mlir.py` pipes it into jaxlib's MLIR
//! parser, which is the closest thing to a verifier available offline.
//!
//! The module text goes to stdout and the signature to stderr, so
//! `... --example dump_mlir > module.mlir` yields a file a verifier can eat.

use gljax::stablehlo::types::DType;
use gljax::{with_policy, BuiltFunc, PrecisionPolicy, Shape, Tensor, TraceCx};

// Qwen2-0.5B.
const B: usize = 1;
const S: usize = 128;
const D: usize = 896;
const FFN: usize = 4864;

fn main() {
    let mut args = std::env::args().skip(1);
    let which = args.next().unwrap_or_else(|| "mlp".to_owned());
    let policy = match args.next().unwrap_or_else(|| "bf16".to_owned()).as_str() {
        "bf16" => PrecisionPolicy::bf16(),
        "f32" => PrecisionPolicy::f32(),
        "f64" => PrecisionPolicy::f64_oracle(),
        other => {
            eprintln!("unknown policy {other:?} — expected bf16, f32 or f64");
            std::process::exit(2);
        }
    };

    let built = match which.as_str() {
        "mlp" => with_policy(policy, || trace_mlp(policy)),
        "ops" => with_policy(policy, || trace_every_op(policy)),
        // A full Qwen2 forward pass. `tiny` keeps the dense RoPE/mask constants
        // small enough to read; `block` is one real Qwen2-0.5B layer at S=128.
        "tiny" => with_policy(policy, || {
            gljax::model::trace_forward(&gljax::model::Qwen2Config::tiny(), 8, 0)
                .expect("tiny trace")
        }),
        "block" => with_policy(policy, || {
            let mut cfg = gljax::model::Qwen2Config::qwen2_0_5b();
            cfg.n_layers = 1;
            gljax::model::trace_forward(&cfg, 128, 0).expect("qwen2 block trace")
        }),
        // The ARTX05 KV-cache primitives: a write at a runtime position and a
        // static-window read. Not wired into the model — see ops/kv_cache.rs.
        "kv" => with_policy(policy, || trace_kv_cache(policy)),
        other => {
            eprintln!("unknown module {other:?} — expected mlp, ops, tiny, block or kv");
            std::process::exit(2);
        }
    };

    println!("{}", built.mlir);

    eprintln!("--- signature ---");
    for p in &built.signature.param_order {
        eprintln!("  {:?} {:<40} {}", p.kind, p.name, p.shape.mlir_type());
    }
    for s in &built.signature.outputs {
        eprintln!("  Output {:<39} {}", "", s.mlir_type());
    }
}

/// A Qwen2 SwiGLU MLP block with checkpoint-exact weight names.
fn trace_mlp(policy: PrecisionPolicy) -> BuiltFunc {
    let (act, wt) = (policy.activation, policy.weight);
    let mut cx = TraceCx::new("main", "qwen2_mlp");
    let x = cx.input("hidden_states", Shape::new([B, S, D], act));
    let residual = x.clone_ref();

    let out = cx.scope("model", |cx| {
        cx.scope("layers.0", |cx| {
            cx.scope("mlp", |cx| {
                let gate_w = cx.weight("gate_proj.weight", Shape::new([D, FFN], wt));
                let up_w = cx.weight("up_proj.weight", Shape::new([D, FFN], wt));
                let down_w = cx.weight("down_proj.weight", Shape::new([FFN, D], wt));
                let gated = &x.matmul(&gate_w).silu() * &x.matmul(&up_w);
                gated.matmul(&down_w)
            })
        })
    });

    let y = &residual + &out;
    cx.finish(&[&y])
}

/// A KV cache write followed by a read, exercising `dynamic_update_slice` and
/// `dynamic_slice` with runtime start indices.
fn trace_kv_cache(policy: PrecisionPolicy) -> BuiltFunc {
    use gljax::ops::kv_cache;
    let dt = policy.activation;
    let mut cx = TraceCx::new("main", "kv_cache");

    let cache = cx.input("k_cache", kv_cache::cache_shape(2, 16, 2, 8, dt));
    let update = cx.input("k_new", Shape::new([1, 1, 2, 8], dt));
    let layer = cx.input("layer", Shape::scalar(DType::I32));
    let pos = cx.input("pos", Shape::scalar(DType::I32));
    let zero = cx.input("zero", Shape::scalar(DType::I32));

    let written = kv_cache::write_at(&cache, &update, &layer, &pos, &zero);
    let window = kv_cache::read_window(&written, &layer, &zero, &zero, 16);
    cx.finish(&[&window])
}

/// One module touching every op emitted in Wave A2.
///
/// The point is coverage for an external verifier, not a meaningful
/// computation — `reduce` in particular has a region, which is where the
/// syntax is easiest to get wrong and where a structural test says least.
fn trace_every_op(policy: PrecisionPolicy) -> BuiltFunc {
    let act = policy.activation;
    let mut cx = TraceCx::new("main", "every_op");

    let x = cx.input("x", Shape::new([2, 3, 4], act));
    let y = cx.input("y", Shape::new([2, 3, 4], act));
    let w = cx.weight("w", Shape::new([4, 5], policy.weight));
    let init = cx.input("init", Shape::scalar(act));

    // Elementwise, binary and unary.
    let a = &(&x + &y) * &x;
    let b = &a - &y;
    let c = b.div(&x).max(&y).min(&x);
    let d = c.neg().abs().sqrt().rsqrt().exp().log().tanh().silu();

    // Shape ops.
    let flat = d.reshape(vec![24]);
    let back = flat.reshape(vec![2, 3, 4]);
    let t = back.transpose(vec![0, 2, 1]);
    let sliced = t.slice(vec![0, 0, 0], vec![2, 4, 2], vec![1, 2, 1]);
    let joined = Tensor::concat(&[&sliced, &sliced], 2);
    let bcast = init.broadcast_to(vec![], vec![2, 2, 4]);
    let mixed = &joined + &bcast;

    // Reduce (a region), then a matmul against a rank-2 weight.
    let summed = mixed.reduce_sum(&[1]);
    let maxed = mixed.reduce_max_with_init(&init, &[1]);
    let combined = &summed + &maxed;
    let projected = combined.matmul(&w);

    // A precision boundary, so `convert` appears too.
    let converted = projected.to_dtype(if act == DType::F64 {
        DType::F32
    } else {
        DType::F64
    });

    cx.finish(&[&converted])
}
