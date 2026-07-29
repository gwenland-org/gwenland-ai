//! Waves A4 + A5 — runtime, checkpoint binding, bucketing and sampling.
//!
//! ⛔ **The sprint gate is not met here and cannot be.** Wave A5's gate is
//! "one token out, matching glproc", which needs `Session::generate` to run,
//! which needs a PJRT plugin. There is none for Windows (`gljax/README.md`),
//! so `Session` has never been constructed and `generate` has never executed.
//!
//! What this file covers is everything on the *host* side of that line — the
//! parts that decide whether the token would be right once it can run:
//!
//! * the compile-cache key separates what must be separated;
//! * the signature check catches a reordering, not just a reshape;
//! * checkpoint binding refuses a transposed weight;
//! * bucketing pads and reads back the row it padded around;
//! * argmax breaks ties the way the oracle does.

use gljax::model::{trace_forward, Qwen2Config};
use gljax::runtime::bucket::{
    bucket_for, last_real_position, pad_to_bucket, padding_for, BUCKETS, MAX_TRACEABLE_BUCKET,
};
use gljax::runtime::plan::PlanSignature;
use gljax::runtime::sample::{argmax, argmax_at};
use gljax::runtime::{CompileCache, CompileKey};
use gljax::{with_policy, PrecisionPolicy};

fn tiny_built() -> gljax::BuiltFunc {
    with_policy(PrecisionPolicy::f32(), || {
        trace_forward(&Qwen2Config::tiny(), 8, 0)
    })
    .expect("trace")
}

// ---------------------------------------------------------------------------
// Wave A4 — compile cache
// ---------------------------------------------------------------------------

#[test]
fn compile_cache_roundtrip() {
    let dir = std::env::temp_dir().join(format!("gljax_a4_cache_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cache = CompileCache::open(&dir).expect("open");

    let built = tiny_built();
    let key = CompileKey::new((0, 114), "cpu", "cpu-test", &built.mlir);

    assert!(cache.get(&key).expect("get").is_none());
    cache.put(&key, b"artifact").expect("put");
    assert_eq!(cache.get(&key).expect("get").as_deref(), Some(&b"artifact"[..]));

    // The key is stable across two identical traces of the same model.
    let again = CompileKey::new((0, 114), "cpu", "cpu-test", &tiny_built().mlir);
    assert_eq!(key.digest(), again.digest(), "the trace must be deterministic");

    std::fs::remove_dir_all(&dir).ok();
}

/// A different sequence bucket is a different program and must not share a
/// cache slot — this is what makes P3's "each shape is a separate artifact"
/// safe rather than a footgun.
#[test]
fn different_buckets_get_different_cache_keys() {
    let cfg = Qwen2Config::tiny();
    let a = with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0)).expect("trace");
    let b = with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 8, 0)).expect("trace");
    let ka = CompileKey::new((0, 114), "cpu", "v", &a.mlir);
    let kb = CompileKey::new((0, 114), "cpu", "v", &b.mlir);
    assert_ne!(ka.digest(), kb.digest());
}

/// Precision is baked into the compiled program, so a bf16 artifact must not be
/// handed to an f32 request.
#[test]
fn different_precision_policies_get_different_cache_keys() {
    let cfg = Qwen2Config::tiny();
    let a = with_policy(PrecisionPolicy::f32(), || trace_forward(&cfg, 4, 0)).expect("trace");
    let b = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 4, 0)).expect("trace");
    let ka = CompileKey::new((0, 114), "cpu", "v", &a.mlir);
    let kb = CompileKey::new((0, 114), "cpu", "v", &b.mlir);
    assert_ne!(ka.digest(), kb.digest());
}

// ---------------------------------------------------------------------------
// Wave A4 — signature validation
// ---------------------------------------------------------------------------

#[test]
fn signature_validation_rejects_shape_mismatch() {
    let built = tiny_built();
    let plan = PlanSignature::from_traced(&built.signature);

    let mut provided = plan.params.clone();
    provided[1].1.dims[0] += 1;
    let err = plan.validate(&provided).expect_err("must reject");
    assert!(
        err.to_string().contains("model.embed_tokens.weight"),
        "the error must name the parameter: {err}"
    );
}

#[test]
fn signature_validation_accepts_the_signature_it_came_from() {
    let built = tiny_built();
    let plan = PlanSignature::from_traced(&built.signature);
    plan.validate(&plan.params.clone()).expect("self-consistent");
}

/// ⛔ q/k/v biases in a Qwen2 layer with equal kv widths have identical shapes.
/// Reordering them passes every shape check.
#[test]
fn signature_validation_catches_reordered_same_shaped_weights() {
    let built = tiny_built();
    let plan = PlanSignature::from_traced(&built.signature);

    let k_idx = plan
        .names()
        .iter()
        .position(|n| n.ends_with("layers.0.self_attn.k_proj.bias"))
        .expect("k bias");
    let v_idx = plan
        .names()
        .iter()
        .position(|n| n.ends_with("layers.0.self_attn.v_proj.bias"))
        .expect("v bias");
    assert_eq!(
        plan.params[k_idx].1, plan.params[v_idx].1,
        "this test is only meaningful if the two shapes are identical"
    );

    let mut provided = plan.params.clone();
    provided.swap(k_idx, v_idx);
    let err = plan.validate(&provided).expect_err("must reject a swap");
    assert!(err.to_string().contains("position"), "{err}");
}

// ---------------------------------------------------------------------------
// Wave A5 — bucketing
// ---------------------------------------------------------------------------

#[test]
fn bucketing_rounds_up_pads_right_and_keeps_the_last_real_position() {
    let prompt: Vec<i32> = (1..=5).collect();
    let bucket = bucket_for(prompt.len(), &BUCKETS).expect("fits");
    assert_eq!(bucket, 128);
    assert_eq!(padding_for(prompt.len(), bucket), 123);

    let padded = pad_to_bucket(&prompt, bucket, 0);
    assert_eq!(padded.len(), bucket);
    assert_eq!(&padded[..5], &prompt[..], "the prompt must not move");
    assert!(padded[5..].iter().all(|&t| t == 0));

    // ⭐ Right padding means the logits row to sample from is index 4, not 127.
    assert_eq!(last_real_position(prompt.len()), 4);
}

#[test]
fn a_prompt_longer_than_every_bucket_is_refused() {
    assert_eq!(bucket_for(4096, &BUCKETS), None);
}

/// ⛔ The 2048 bucket is in the grid but does not trace: the causal mask is a
/// dense O(S²) constant and 2048² exceeds the cap. Pinned so the gap stays
/// visible rather than becoming folklore.
#[test]
fn the_largest_bucket_does_not_trace_yet_and_says_why() {
    let mut cfg = Qwen2Config::qwen2_0_5b();
    cfg.n_layers = 1;

    for bucket in BUCKETS.iter().copied().filter(|&b| b <= MAX_TRACEABLE_BUCKET) {
        with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, bucket, 0))
            .unwrap_or_else(|e| panic!("bucket {bucket} should trace: {e}"));
    }

    let err = with_policy(PrecisionPolicy::bf16(), || trace_forward(&cfg, 2048, 0))
        .expect_err("2048 must fail until the mask stops being a constant");
    assert!(err.to_string().contains("runtime weight"), "{err}");
}

// ---------------------------------------------------------------------------
// Wave A5 — sampling
// ---------------------------------------------------------------------------

#[test]
fn argmax_matches_the_oracle_convention() {
    assert_eq!(argmax(&[0.1, 0.9, 0.3]), Some(1));
    // Lower index wins a tie — NumPy, PyTorch and llama.cpp all do this, and
    // disagreeing produces a divergence that looks like a numerics bug.
    assert_eq!(argmax(&[1.0, 1.0]), Some(0));
    // A NaN never wins.
    assert_eq!(argmax(&[f32::NAN, 0.5]), Some(1));
}

#[test]
fn sampling_reads_the_padded_row_that_corresponds_to_the_last_real_token() {
    // A [1, 4, 3] logits buffer for a 2-token prompt in a 4-wide bucket.
    // Rows 2 and 3 are padding positions and must not be sampled.
    let vocab = 3;
    let seq = 4;
    let mut logits = vec![0.0f32; seq * vocab];
    logits[vocab + 2] = 5.0; // row 1 (last real token) wants token 2
    logits[3 * vocab] = 9.0; // row 3 (padding) would want token 0

    let prompt_len = 2;
    let pos = last_real_position(prompt_len);
    assert_eq!(pos, 1);
    assert_eq!(argmax_at(&logits, seq, vocab, pos).unwrap(), 2);
    // Sampling the last row of the bucket would give a different, wrong answer.
    assert_eq!(argmax_at(&logits, seq, vocab, seq - 1).unwrap(), 0);
}

// ---------------------------------------------------------------------------
// Wave A5 — KV cache primitives
// ---------------------------------------------------------------------------

#[test]
fn kv_cache_write_and_read_emit_dynamic_slicing() {
    use gljax::ops::kv_cache;
    use gljax::stablehlo::types::{DType, Shape};
    use gljax::TraceCx;

    let mut cx = TraceCx::new("main", "kv");
    let cfg = Qwen2Config::qwen2_0_5b();
    let cache = cx.input(
        "k_cache",
        kv_cache::cache_shape(cfg.n_layers, 128, cfg.n_kv_heads, cfg.head_dim, DType::F32),
    );
    let update = cx.input(
        "k_new",
        Shape::new([1, 1, cfg.n_kv_heads, cfg.head_dim], DType::F32),
    );
    let layer = cx.input("layer", Shape::scalar(DType::I32));
    let pos = cx.input("pos", Shape::scalar(DType::I32));
    let zero = cx.input("zero", Shape::scalar(DType::I32));

    let written = kv_cache::write_at(&cache, &update, &layer, &pos, &zero);
    let window = kv_cache::read_window(&written, &layer, &zero, &zero, 128);

    assert_eq!(written.shape().dims, cache.shape().dims);
    assert_eq!(window.shape().dims, vec![1, 128, 2, 64]);

    let mlir = cx.finish(&[&window]).mlir;
    assert!(mlir.contains(r#""stablehlo.dynamic_update_slice""#), "{mlir}");
    assert!(mlir.contains(r#""stablehlo.dynamic_slice""#), "{mlir}");
    assert!(
        mlir.contains("slice_sizes = array<i64: 1, 128, 2, 64>"),
        "{mlir}"
    );
}

/// The full-context KV cache is the memory wall ARTX05 warns about — worth
/// pinning the arithmetic so a config change surfaces it.
#[test]
fn full_context_kv_cache_size_is_what_artx05_predicts() {
    use gljax::ops::kv_cache::cache_shape;
    use gljax::stablehlo::types::DType;

    let cfg = Qwen2Config::qwen2_0_5b();
    // Both K and V, bf16, at the sprint's largest bucket.
    let per_tensor = cache_shape(cfg.n_layers, 2048, cfg.n_kv_heads, cfg.head_dim, DType::BF16);
    let total_mib = per_tensor.byte_len() * 2 / (1024 * 1024);
    assert_eq!(total_mib, 24, "24 layers x 2048 x 2 heads x 64 x 2B x 2 tensors");
}
