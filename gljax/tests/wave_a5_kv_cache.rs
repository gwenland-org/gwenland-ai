//! Wave A5 — KV cache parity against the recomputation oracle.
//!
//! ARTX05's headline risk: a cache that reads one position off produces
//! fluent, wrong text with no error anywhere. Gate A5's coherence check
//! (`examples/gate_a5.rs`) would very likely still pass on exactly that bug —
//! it checks the output is real, on-topic language, not that it matches a
//! known-good run. The only check that actually catches an off-by-one cache
//! read is comparing `CachedSession::generate` token-for-token against
//! `Session::generate` — the unchanged, already-CI-gated recomputation path —
//! on the same prompt, same weights, same sampling.
//!
//! Needs a real PJRT plugin and a real Qwen2-0.5B checkpoint, so it SKIPs
//! without both — see `gljax/README.md`. `.github/workflows/gljax-pjrt.yml`'s
//! `gate-a5` job sets both `PJRT_PLUGIN_CPU` and `QWEN2_HF_DIR`, so this is
//! the one place in CI it actually runs.
//!
//! ⚠️ Run with `--release`: an XLA client compiles fine in debug and is far
//! too slow to generate with, same as Gate A5.

use std::path::PathBuf;
use std::rc::Rc;

use gljax::model::{trace_decode, trace_forward, trace_prefill_with_cache};
use gljax::pjrt::{cpu_plugin_path, PjrtPlugin};
use gljax::runtime::{CachedSession, HfCheckpoint, Session};
use gljax::{with_policy, PrecisionPolicy};

const PROMPT: &str = "The capital of France is";
const NEW_TOKENS: usize = 12;
/// Covers `PROMPT`'s ~6 tokens plus `NEW_TOKENS` comfortably — same bucket
/// Gate A5 already exercises at this token count.
const WINDOW: usize = 128;

#[test]
fn cached_decode_matches_the_recomputation_oracle_token_for_token() {
    let test_name = "cached_decode_matches_the_recomputation_oracle_token_for_token";

    let Some(plugin_path) = cpu_plugin_path() else {
        eprintln!("SKIP {test_name}: no PJRT plugin configured (set PJRT_PLUGIN_CPU)");
        return;
    };
    let Ok(model_dir) = std::env::var("QWEN2_HF_DIR") else {
        eprintln!("SKIP {test_name}: QWEN2_HF_DIR not set");
        return;
    };
    let model_dir = PathBuf::from(model_dir);

    let checkpoint = HfCheckpoint::open(&model_dir).expect("open the HF checkpoint");
    let config = checkpoint.config.clone();
    let prompt_ids: Vec<i32> = checkpoint
        .encode(PROMPT)
        .expect("encode the prompt")
        .into_iter()
        .map(|id| id as i32)
        .collect();
    assert!(
        prompt_ids.len() + NEW_TOKENS <= WINDOW,
        "prompt is {} tokens; WINDOW={WINDOW} must cover prompt + {NEW_TOKENS} new tokens",
        prompt_ids.len()
    );
    let eos = Some(checkpoint.eos_id as i32);
    let pad = checkpoint.eos_id as i32;

    // One plugin, two clients — Session and CachedSession each get their own
    // PJRT_Client, but there is no need to dlopen the shared library twice.
    let plugin = Rc::new(PjrtPlugin::load(&plugin_path).expect("load the PJRT plugin"));

    println!("── oracle: Session::generate (full recomputation) ──");
    let built_recompute = with_policy(PrecisionPolicy::f32(), || trace_forward(&config, WINDOW, 0))
        .expect("trace_forward");
    let oracle = Session::open(
        Rc::clone(&plugin),
        &built_recompute,
        config.clone(),
        WINDOW,
        &checkpoint.weights,
        None,
    )
    .expect("open the recomputation-path session");
    let oracle_tokens = oracle
        .generate(&prompt_ids, NEW_TOKENS, eos, pad)
        .expect("oracle generate");
    println!("oracle   : {oracle_tokens:?}");

    println!("── cached: CachedSession::generate (prefill + KV-cached decode) ──");
    let built_prefill =
        with_policy(PrecisionPolicy::f32(), || trace_prefill_with_cache(&config, WINDOW, WINDOW))
            .expect("trace_prefill_with_cache");
    let built_decode =
        with_policy(PrecisionPolicy::f32(), || trace_decode(&config, WINDOW)).expect("trace_decode");
    let mut cached = CachedSession::open(
        Rc::clone(&plugin),
        &built_prefill,
        &built_decode,
        config,
        WINDOW,
        &checkpoint.weights,
    )
    .expect("open the cached session");
    let cached_tokens = cached
        .generate(&prompt_ids, NEW_TOKENS, eos, pad)
        .expect("cached generate");
    println!("cached   : {cached_tokens:?}");

    assert_eq!(
        cached_tokens, oracle_tokens,
        "KV-cached decode diverged from the recomputation oracle — this is exactly ARTX05's \
         headline risk: a cache reading one position off produces fluent, wrong text with no \
         error anywhere. The oracle path is unchanged from the one Gate A5 already validated, \
         so a divergence here means the cache wiring, not the model, is wrong."
    );
    println!("KV cache parity: PASSED — {} tokens match the oracle exactly", oracle_tokens.len());
}
