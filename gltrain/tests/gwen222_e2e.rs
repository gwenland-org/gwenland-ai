/// GWEN-222 end-to-end tests: checkpoint resume + export-adapter shape validation.
///
/// These are integration tests (separate crate) so they exercise only the public
/// API. The GGUF fixture helper is gated behind `feature = "test-utils"`; this
/// target declares `required-features = ["test-utils"]` in Cargo.toml.

use candle_core::{DType, Device, Tensor, Var};
use candle_nn::VarMap;
use tempfile::TempDir;

use gwenland_core::error::GwenError;
use gwenland_core::train::checkpoint_resumer;
use gwenland_core::train::config::{LoraConfig, NewTrainConfig, ResumeMode};
use gwenland_core::train::layer_loader::write_transformer_gguf_pub;
use gwenland_core::train::lora_cli::export_adapter;
use gwenland_core::train::LayeredTrainingLoop;

// ── helpers ─────────────────────────────────────────────────────────────────

fn make_config(output: std::path::PathBuf) -> NewTrainConfig {
    NewTrainConfig {
        output_path: output,
        lora: LoraConfig {
            r: 2,
            alpha: 4.0,
            dropout: 0.0,
            target_modules: vec![],
        },
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        lr: 1e-4,
        ..NewTrainConfig::default()
    }
}

/// One batch of `n` token IDs `[1, 2, …, n]`.
fn make_batch(n: usize) -> Vec<Vec<Tensor>> {
    let ids: Vec<u32> = (1..=(n as u32)).collect();
    let t = Tensor::from_vec(ids, (n,), &Device::Cpu).unwrap();
    vec![vec![t]]
}

/// Insert a flat-layout LoRA pair `lora_{a|b}_layer_{N}_{proj}_proj` into a VarMap.
/// `lora_a` is `[rank, d_in]`, `lora_b` is `[d_out, rank]`.
fn insert_flat_pair(vm: &VarMap, layer: usize, proj: &str, rank: usize, d_in: usize, d_out: usize) {
    let dev = &Device::Cpu;
    let mut data = vm.data().lock().unwrap();
    let a = Tensor::ones((rank, d_in), DType::F32, dev).unwrap();
    let b = Tensor::ones((d_out, rank), DType::F32, dev).unwrap();
    data.insert(
        format!("lora_a_layer_{layer}_{proj}_proj"),
        Var::from_tensor(&a).unwrap(),
    );
    data.insert(
        format!("lora_b_layer_{layer}_{proj}_proj"),
        Var::from_tensor(&b).unwrap(),
    );
}

// ── Task 4.1: resume round-trip ─────────────────────────────────────────────

/// Full save → discover → load → resume round-trip via the public API.
///
/// We seed the first loop with `initial_step = 499`; after one optimizer step it
/// reaches step 500, which is `% 500 == 0`, so the loop writes a real
/// `checkpoint_000500.safetensors`. We then auto-discover it, load it into a
/// fresh loop seeded at step 500, and assert the second run counts only its own
/// steps.
#[test]
fn test_e2e_resume() {
    let gguf = write_transformer_gguf_pub(1);
    let dir = TempDir::new().unwrap();
    let cfg = make_config(dir.path().to_path_buf());

    // First loop: seeded at 499 so a single step trips the 500-step save.
    let mut loop_a =
        LayeredTrainingLoop::new(cfg.clone(), gguf.path(), make_batch(2), VarMap::new(), None, 499)
            .expect("construct loop A");
    let res_a = loop_a.run().expect("run loop A");
    assert_eq!(res_a.total_steps, 1, "loop A ran exactly one step");

    let ckpt = dir.path().join("checkpoint_000500.safetensors");
    assert!(ckpt.exists(), "a checkpoint must be saved at the resumed 500-step interval");

    // Auto-discovery resolves the just-written checkpoint and its step.
    let (resolved, step) =
        checkpoint_resumer::resolve_checkpoint(&ResumeMode::Auto, dir.path()).expect("resolve");
    let resolved = resolved.expect("auto-discovery finds the checkpoint");
    assert_eq!(step, 500, "parsed step from checkpoint filename");
    assert_eq!(resolved, ckpt);

    // Second loop: resume from step 500, load the adapter weights, run once more.
    let mut loop_b =
        LayeredTrainingLoop::new(cfg, gguf.path(), make_batch(2), VarMap::new(), None, step)
            .expect("construct loop B");
    loop_b.load_checkpoint(&resolved).expect("load checkpoint into loop B");
    let res_b = loop_b.run().expect("run loop B");

    // total_steps reflects ONLY the second run, not the cumulative 501.
    assert_eq!(
        res_b.total_steps, 1,
        "resumed run reports current-run steps only, got {}",
        res_b.total_steps
    );
}

// ── Task 4.1b: auto-discovery with no checkpoints (fresh start) ──────────────

/// `--resume` (Auto) on a directory with no checkpoints must NOT error — it
/// returns `(None, 0)` and a fresh `initial_step = 0` loop runs cleanly.
#[test]
fn test_e2e_resume_auto_no_checkpoints() {
    let gguf = write_transformer_gguf_pub(1);
    let dir = TempDir::new().unwrap();

    let result = checkpoint_resumer::resolve_checkpoint(&ResumeMode::Auto, dir.path())
        .expect("auto-resume on empty dir is Ok, not Err");
    assert_eq!(result, (None, 0), "empty dir → fresh start");

    let cfg = make_config(dir.path().to_path_buf());
    let mut loop_fresh =
        LayeredTrainingLoop::new(cfg, gguf.path(), make_batch(2), VarMap::new(), None, 0)
            .expect("fresh loop constructs cleanly");
    let res = loop_fresh.run().expect("fresh run does not panic");
    assert!(res.final_loss.is_finite(), "fresh run produces finite loss");
}

// ── Task 4.2: export-adapter --base-gguf shape validation ────────────────────

/// A checkpoint whose adapter dims match the base GGUF exports successfully;
/// a mismatched one returns `ShapeMismatch` and leaves no output file on disk.
#[test]
fn test_e2e_export_shape_validation() {
    // Base GGUF: blk.0.attn_q.weight is [4, 4] → q_proj d_in = d_out = 4.
    let gguf = write_transformer_gguf_pub(1);
    let dir = TempDir::new().unwrap();

    // ── matching adapter ──
    let good_ckpt = dir.path().join("good_checkpoint.safetensors");
    {
        let vm = VarMap::new();
        insert_flat_pair(&vm, 0, "q", 2, 4, 4); // lora_a (2,4), lora_b (4,2)
        vm.save(&good_ckpt).unwrap();
    }
    let good_out = dir.path().join("good_adapter.safetensors");
    let n = export_adapter(&good_ckpt, &good_out, false, Some(gguf.path()))
        .expect("matching adapter must export");
    assert_eq!(n, 1, "one adapter pair exported");
    assert!(good_out.exists(), "matching export writes the output file");

    // ── mismatched adapter (d_in = 128 ≠ 4) ──
    let bad_ckpt = dir.path().join("bad_checkpoint.safetensors");
    {
        let vm = VarMap::new();
        insert_flat_pair(&vm, 0, "q", 2, 128, 4); // lora_a (2,128) → d_in 128
        vm.save(&bad_ckpt).unwrap();
    }
    let bad_out = dir.path().join("bad_adapter.safetensors");
    let err = export_adapter(&bad_ckpt, &bad_out, false, Some(gguf.path()))
        .expect_err("mismatched adapter must fail validation");
    assert!(
        matches!(err, GwenError::ShapeMismatch { .. }),
        "expected ShapeMismatch, got {err:?}"
    );
    assert!(
        !bad_out.exists(),
        "no output file may be written when validation fails"
    );
}

// ── Task 4.3 (Property 4): checkpoint load populates VarMap (round-trip) ──────

/// Save a VarMap of known tensors, load it back into a fresh VarMap that already
/// owns identically-named (zero-valued) Vars, and assert the values are restored.
#[test]
fn prop_checkpoint_varmap_roundtrip() {
    for (rank, d_in, d_out) in [(2usize, 4usize, 4usize), (4, 8, 16), (1, 3, 5)] {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("checkpoint_000001.safetensors");
        let dev = &Device::Cpu;

        // Source VarMap: lora_a filled with 0.5, lora_b filled with -0.25.
        let src = VarMap::new();
        {
            let mut data = src.data().lock().unwrap();
            let a = (Tensor::ones((rank, d_in), DType::F32, dev).unwrap() * 0.5).unwrap();
            let b = (Tensor::ones((d_out, rank), DType::F32, dev).unwrap() * -0.25).unwrap();
            data.insert("l0.lora_a".into(), Var::from_tensor(&a).unwrap());
            data.insert("l0.lora_b".into(), Var::from_tensor(&b).unwrap());
        }
        src.save(&path).unwrap();

        // Target VarMap: same names, zero-valued — load must overwrite them.
        let mut dst = VarMap::new();
        {
            let mut data = dst.data().lock().unwrap();
            let za = Tensor::zeros((rank, d_in), DType::F32, dev).unwrap();
            let zb = Tensor::zeros((d_out, rank), DType::F32, dev).unwrap();
            data.insert("l0.lora_a".into(), Var::from_tensor(&za).unwrap());
            data.insert("l0.lora_b".into(), Var::from_tensor(&zb).unwrap());
        }
        checkpoint_resumer::load_checkpoint_into_varmap(&mut dst, &path).expect("load");

        let data = dst.data().lock().unwrap();
        let a_val = data["l0.lora_a"].as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b_val = data["l0.lora_b"].as_tensor().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(
            a_val.iter().all(|&v| (v - 0.5).abs() < 1e-6),
            "lora_a restored to 0.5 (rank={rank}, d_in={d_in})"
        );
        assert!(
            b_val.iter().all(|&v| (v + 0.25).abs() < 1e-6),
            "lora_b restored to -0.25 (d_out={d_out}, rank={rank})"
        );
    }
}
