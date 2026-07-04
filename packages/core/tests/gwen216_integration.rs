/// GWEN-216 integration and regression tests.
///
/// These are Rust integration tests (live in `tests/`) so they exercise only
/// the library's public API.  The no-full-load invariant test uses the
/// `LIVE_LAYER_COUNT` atomic counter that is injected into `layer_loader.rs`
/// under `#[cfg(test)]`.
#[allow(unused_imports)]
use gwenland_core::train::{LayerLoader, LayeredTrainingLoop};
#[allow(unused_imports)]
use gwenland_core::train::layer_loader::LIVE_LAYER_COUNT;

// write_minimal_gguf_pub is exposed via feature = "test-utils" in layer_loader.rs.
use gwenland_core::train::layer_loader::write_transformer_gguf_pub;

use candle_core::{Device, Tensor};
use candle_nn::VarMap;
use gwenland_core::train::config::{LoraConfig, NewTrainConfig};
use tempfile::TempDir;

// ── Shared helpers ────────────────────────────────────────────────────────────

fn two_layer_gguf() -> tempfile::NamedTempFile {
    write_transformer_gguf_pub(2)
}

fn make_varmap() -> VarMap {
    VarMap::new()
}

fn make_config(output: std::path::PathBuf) -> NewTrainConfig {
    NewTrainConfig {
        output_path: output,
        lora: LoraConfig { r: 2, alpha: 4.0, dropout: 0.0, target_modules: vec![] },
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        lr: 1e-4,
        ..NewTrainConfig::default()
    }
}

/// Two tokens → one batch entry of shape (1, 1) matching d_in=1.
fn make_batch(n: usize) -> Vec<Vec<Tensor>> {
    let ids: Vec<u32> = (1..=(n as u32)).collect();
    let t = Tensor::from_vec(ids, (n,), &Device::Cpu).unwrap();
    vec![vec![t]]
}

// ── Integration test 1: end-to-end run with finite loss ───────────────────────

#[test]
fn integration_layered_training_loop_loss_is_finite() {
    let f   = two_layer_gguf();
    let td  = TempDir::new().unwrap();
    let cfg = make_config(td.path().to_path_buf());

    let mut ltl = LayeredTrainingLoop::new(
        cfg, f.path(), make_batch(2), make_varmap(), None, 0,
    ).expect("LayeredTrainingLoop::new failed");

    let result = ltl.run().expect("LayeredTrainingLoop::run failed");
    assert!(
        result.final_loss.is_finite(),
        "expected finite loss, got {}", result.final_loss
    );
    assert!(result.total_steps >= 1, "expected at least one optimizer step");
}

// ── Integration test 2: no-full-load invariant ────────────────────────────────
//
// Uses the `LIVE_LAYER_COUNT` atomic counter injected under `#[cfg(test)]`
// into `LayerLoader::load_layer` (+1) and `LoadedLayer::Drop` (-1).
//
// The test loads every layer sequentially, asserting the counter is exactly 1
// while the layer is held and 0 immediately after drop.  If any path ever
// materialises two layers simultaneously the assertion fires.
//
// The object-lifetime counter is platform-independent. Unix additionally
// issues MADV_DONTNEED on drop; Windows leaves physical-page reclamation to
// the kernel, which is outside this test's scope.
#[test]
fn invariant_never_more_than_one_layer_in_ram() {
    use std::sync::atomic::Ordering;

    let f = write_transformer_gguf_pub(3);

    let loader = LayerLoader::open(f.path()).expect("open");
    assert_eq!(loader.num_layers(), 3);

    // Reset counter — other tests in the same process may have incremented it.
    LIVE_LAYER_COUNT.store(0, Ordering::SeqCst);

    for n in 0..loader.num_layers() {
        assert_eq!(
            LIVE_LAYER_COUNT.load(Ordering::SeqCst), 0,
            "counter must be 0 before loading layer {n}"
        );

        let layer = loader.load_layer(n).expect("load_layer");

        assert_eq!(
            LIVE_LAYER_COUNT.load(Ordering::SeqCst), 1,
            "counter must be exactly 1 while layer {n} is live"
        );

        // Explicitly drop — triggers MADV_DONTNEED + counter decrement.
        drop(layer);

        assert_eq!(
            LIVE_LAYER_COUNT.load(Ordering::SeqCst), 0,
            "counter must be 0 after dropping layer {n}"
        );
    }
}

// ── Module wiring: public re-exports are reachable ────────────────────────────

#[test]
fn public_types_are_reachable() {
    // This test will fail to *compile* (not just fail at runtime) if any of the
    // required public types are missing from `gwenland_core::train`.
    fn _assert_reachable(
        _: gwenland_core::train::LayerSlice,
        _: gwenland_core::train::LayerIndex,
        _: gwenland_core::train::LayerLoader,
        _: gwenland_core::train::LayeredTrainingLoop,
    ) {}
    // No need to call _assert_reachable at runtime — the type-check is enough.
    let _ = std::mem::size_of::<gwenland_core::train::LayerSlice>();
    let _ = std::mem::size_of::<gwenland_core::train::LayerIndex>();
    let _ = std::mem::size_of::<gwenland_core::train::LayerLoader>();
    let _ = std::mem::size_of::<gwenland_core::train::LayeredTrainingLoop>();
}
