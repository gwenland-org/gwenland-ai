//! GWEN-219 native dry-run validation against a real local GGUF.
//!
//! This is the empirical "no regression on dry-run" check for the multi-tensor
//! layered loop (default path) and the experimental `--gdtqp` rank-allocation
//! path. It exercises the exact code path the CLI dry-run uses
//! (`run_native_local` with `max_steps = Some(1)`), so it is the same one step
//! that `gwen train --config … --dry-run` runs.
//!
//! It is **env-gated** so CI (which has no model file) skips it: set
//! `GWEN_DRYRUN_GGUF` to a local `.gguf` to run it, e.g.
//!
//! ```text
//! GWEN_DRYRUN_GGUF="C:/Users/me/Downloads/Qwen3-1.7B-Q8_0.gguf" \
//!   cargo test -p gwenland-core --test gwen219_dryrun -- --nocapture
//! ```
//!
//! The tokenizer is fetched from HF Hub by repo inferred from the GGUF name
//! (`Qwen3-1.7B-Q8_0.gguf` → `Qwen/Qwen3-1.7B`); set `HF_HUB_OFFLINE=1` to use
//! only the local HF cache.

use std::path::PathBuf;

use gwenland_core::train::config::{LoraConfig, NewTrainConfig};
use gwenland_core::train::native_runner::run_native_local;

/// All seven transformer projections — the full GWEN-219 coverage set.
fn all_projections() -> Vec<String> {
    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn dryrun_config(gdtqp: bool, dataset: PathBuf, output: PathBuf) -> NewTrainConfig {
    NewTrainConfig {
        dataset_path: dataset,
        output_path: output,
        epochs: 1,
        batch_size: 1,
        grad_accum: 1,
        lr: 1e-4,
        dry_run: true,
        max_steps: Some(1), // one optimiser step — mirrors `--dry-run`
        lora: LoraConfig {
            r: 8,
            alpha: 16.0,
            dropout: 0.05,
            target_modules: all_projections(),
        },
        gdtqp,
        ..NewTrainConfig::default()
    }
}

#[test]
fn gwen219_native_dryrun_default_and_gdtqp() {
    let Ok(gguf) = std::env::var("GWEN_DRYRUN_GGUF") else {
        eprintln!("[gwen219_dryrun] SKIP — set GWEN_DRYRUN_GGUF to a local .gguf to run");
        return;
    };
    let gguf = PathBuf::from(gguf);
    assert!(gguf.exists(), "GWEN_DRYRUN_GGUF does not exist: {}", gguf.display());

    let dataset = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/sample_100.jsonl");
    let output = std::env::temp_dir().join("gwen219_dryrun_out");
    std::fs::create_dir_all(&output).unwrap();

    // ── Default path: all projections, uniform rank ──────────────────────────
    eprintln!("[gwen219_dryrun] === DEFAULT PATH ===");
    let cfg = dryrun_config(false, dataset.clone(), output.clone());
    let r = run_native_local(&cfg, &gguf, None, None)
        .expect("default-path dry-run failed");
    assert!(r.final_loss.is_finite(), "default loss not finite: {}", r.final_loss);
    assert!(r.total_steps >= 1, "expected ≥1 optimiser step");

    // ── EXPERIMENTAL --gdtqp path: S(ρ)-allocated per-projection rank ─────────
    eprintln!("[gwen219_dryrun] === EXPERIMENTAL --gdtqp PATH ===");
    let cfg = dryrun_config(true, dataset, output);
    let r = run_native_local(&cfg, &gguf, None, None)
        .expect("gdtqp dry-run failed");
    assert!(r.final_loss.is_finite(), "gdtqp loss not finite: {}", r.final_loss);
    assert!(r.total_steps >= 1, "expected ≥1 optimiser step (gdtqp)");
}
