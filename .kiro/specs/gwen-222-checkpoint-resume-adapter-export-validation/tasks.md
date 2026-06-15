# Implementation Plan: GWEN-222 — Checkpoint Resume + Adapter Export Validation

## Overview

Two additive capabilities delivered in four gated waves:
1. **Wave 1 (Audit)** — read-only inventory of every touch-point; writes `audit.md`, zero source changes.
2. **Wave 2 (Checkpoint Resume)** — `ResumeMode` enum, `checkpoint_resumer.rs`, `--resume` flag wired end-to-end into `LayeredTrainingLoop::new(initial_step)`.
3. **Wave 3 (Export Adapter Shape Validation)** — `--base-gguf` flag, `AdapterShapeValidator`, extended `GwenError::ShapeMismatch`, `pub(crate)` promotion of `lora_merger` helpers.
4. **Wave 4 (E2E Validation)** — integration tests exercising the full round-trips of both features.

Each wave ends with a `cargo build --workspace` clean-build gate + `cargo test --workspace` pass before the next wave may begin.

---

## Tasks

### Wave 1 — Audit

- [ ] 1. Write `audit.md` — read-only source survey, no code changes
  - [ ] 1.1 Survey all touch-point files and record exact line numbers for each change site
    - Read `packages/core/src/train/config.rs`: note where `NewTrainConfig` struct body ends and where `impl Default` closes — that is where `resume_checkpoint: ResumeMode` and its default will be inserted.
    - Read `packages/core/src/train/layered_training_loop.rs`: record the exact line of `LayeredTrainingLoop::new()` signature and the line where `optimizer_steps` is initialised to `0` inside `run()`.
    - Read `packages/core/src/train/native_runner.rs`: record line of `let varmap = VarMap::new()` and the `LayeredTrainingLoop::new(...)` call site that will gain `initial_step`.
    - Read `packages/core/src/train/lora_cli.rs`: record the `export_adapter` function signature line and the guard before `exporter.export_safetensors` — the shape-validation hook goes between extraction and write.
    - Read `packages/core/src/train/lora_merger.rs`: record lines of `parse_gguf_key` (line ~89) and `shape_to_2d` (lives in `layered_training_loop.rs`, line ~956 — note it is *not* in `lora_merger.rs` yet; confirm the correct source file).
    - Read `packages/core/src/error.rs`: record the current `ShapeMismatch` variant — it has `adapter: Vec<usize>` and `base: Vec<usize>` but is **missing** `adapter_key: String`; note all existing call sites that construct it.
    - Read `packages/tui/src/commands/train.rs`: record `ExportAdapterArgs` struct end-line, `TrainArgs` struct end-line, and both call sites of `lora_cli::export_adapter` (`run_export_adapter` and the `--auto-merge` path).
    - Write findings to `.kiro/specs/gwen-222-checkpoint-resume-adapter-export-validation/audit.md` with a table: `file | symbol | current line | change required`.
    - _Requirements: 1.1, 2.1, 3.1, 5.1, 6.1, 7.1, 8.1_

  - [ ] 1.2 Confirm `shape_to_2d` location and decide promotion strategy
    - Verify `shape_to_2d` is defined in `layered_training_loop.rs` (private `fn`) — not in `lora_merger.rs`.
    - Decide: either move it to a shared internal helper module or duplicate the four-line body in `lora_cli.rs` under a local private function. Record the decision in `audit.md`.
    - Confirm `parse_gguf_key` in `lora_merger.rs` (line ~89) is currently `fn` (private); record that it must become `pub(crate)` for `lora_cli.rs` to use it.
    - _Requirements: 8.2, 8.3, 8.4_

  - [ ] 1.3 Wave 1 gate — `cargo build --workspace` must be clean before proceeding
    - Ensure `cargo build --workspace` completes without error (no source changes made in Wave 1).
    - _Requirements: all_

---

### Wave 2 — Checkpoint Resume

- [ ] 2. Add `ResumeMode` enum and `resume_checkpoint` field to `config.rs`
  - [ ] 2.1 Implement `ResumeMode` enum and extend `NewTrainConfig`
    - In `packages/core/src/train/config.rs`, add `ResumeMode` enum with three variants: `None` (default via `#[default]`), `Auto`, `Explicit(PathBuf)`.
    - Derive `Debug`, `Clone`, `Default` on `ResumeMode`.
    - Add `pub resume_checkpoint: ResumeMode` field to `NewTrainConfig`.
    - Add `resume_checkpoint: ResumeMode::None` to `impl Default for NewTrainConfig`.
    - _Requirements: 1.2, 1.3, 1.4_

  - [ ] 2.2 Create `checkpoint_resumer.rs` — new module with three public functions
    - Create `packages/core/src/train/checkpoint_resumer.rs`.
    - Implement `pub fn resolve_checkpoint(mode: &ResumeMode, output_path: &Path) -> Result<(Option<PathBuf>, usize)>`:
      - `ResumeMode::None` → return `Ok((None, 0))` immediately.
      - `ResumeMode::Auto` → read `output_path`, collect entries matching `checkpoint_*.safetensors`, sort lexicographically, take last. If none found: `eprintln!("[resume] warning: no checkpoint_*.safetensors found in '{}'; starting from step 0", output_path.display())`, return `Ok((None, 0))`. If found: log the selected path to stderr, return `Ok((Some(path), parse_step_from_filename(&path)))`.
      - `ResumeMode::Explicit(p)` → if `p.exists()`, return `Ok((Some(p.clone()), parse_step_from_filename(p)))`; otherwise `bail!("checkpoint path does not exist: {}", p.display())`.
    - Implement `pub fn parse_step_from_filename(path: &Path) -> usize`:
      - Strip directory, strip `.safetensors`, strip `checkpoint_` prefix, parse as `usize`. On any failure: `eprintln!("[resume] warning: could not parse step from '{}'; step counter will restart from 0", path.display())`, return `0`.
    - Implement `pub fn load_checkpoint_into_varmap(varmap: &mut VarMap, path: &Path) -> Result<()>`:
      - Call `varmap.load(path).map_err(|e| anyhow::anyhow!("VarMap::load failed for '{}': {}", path.display(), e))`.
    - Register the module in `packages/core/src/train/mod.rs`.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4_

  - [ ]* 2.3 Write unit tests for `checkpoint_resumer.rs`
    - In a `#[cfg(test)]` block at the bottom of `checkpoint_resumer.rs`:
    - **Property 3: Auto-discovery selects lexicographically greatest checkpoint** — `test_resolve_auto_picks_lex_max`: create temp dir, write three zero-byte files (`checkpoint_000000.safetensors`, `checkpoint_000500.safetensors`, `checkpoint_001000.safetensors`), call `resolve_checkpoint(Auto, dir)`, assert returned path is `checkpoint_001000.safetensors` and step is 1000.
    - **Property 5: Step counter parsed correctly from any valid checkpoint filename** — `test_parse_step_roundtrip`: for each `N` in `[0, 500, 1000, 999999]`, construct `format!("checkpoint_{N:06}.safetensors")`, assert `parse_step_from_filename(path) == N`.
    - `test_resolve_auto_empty_dir`: empty temp dir, assert `Ok((None, 0))`.
    - `test_parse_step_nonstandard`: `arbitrary_name.safetensors` → returns `0`, no panic.
    - **Property 2: Non-existent explicit checkpoint path fails before training** — `test_explicit_path_missing`: nonexistent path under `ResumeMode::Explicit` → `Err`.
    - _Requirements: 2.1, 3.3, 3.4_

  - [ ] 2.4 Add `--resume` flag to `TrainArgs` in `train.rs` and map to `ResumeMode`
    - In `packages/tui/src/commands/train.rs`, add to `TrainArgs`:
      ```rust
      #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "AUTO",
            help = "Resume from the last checkpoint (auto-discover) or an explicit checkpoint \
                    path. Restores LoRA adapter weights only — AdamW optimizer state is NOT \
                    restored; a momentum warm-up period will occur after resuming.")]
      pub resume: Option<String>,
      ```
    - In `run_native_path()` (and in `build_train_config_from_args()` for the config-YAML path), map `args.resume` to `ResumeMode`:
      ```rust
      let resume_mode = match &args.resume {
          None                          => ResumeMode::None,
          Some(s) if s == "AUTO"        => ResumeMode::Auto,
          Some(p)                       => ResumeMode::Explicit(PathBuf::from(p)),
      };
      config.resume_checkpoint = resume_mode;
      ```
    - Add `use gwenland_core::train::config::ResumeMode;` to imports.
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 4.3_

  - [ ] 2.5 Wire checkpoint loading into `native_runner.rs` before `LayeredTrainingLoop::new`
    - In `packages/core/src/train/native_runner.rs`, in `run_native_local()`:
      - After `let varmap = VarMap::new();`, add:
        ```rust
        use crate::train::checkpoint_resumer;
        let (ckpt_path, initial_step) =
            checkpoint_resumer::resolve_checkpoint(&config.resume_checkpoint, &config.output_path)
            .context("checkpoint resume discovery failed")?;
        if let Some(ref path) = ckpt_path {
            checkpoint_resumer::load_checkpoint_into_varmap(&mut varmap, path)
                .context("failed to load checkpoint into VarMap")?;
        }
        ```
      - Update `LayeredTrainingLoop::new(...)` call to pass `initial_step` (see task 2.6).
    - _Requirements: 2.3, 3.1, 3.2_

  - [ ] 2.6 Extend `LayeredTrainingLoop::new()` to accept `initial_step: usize`
    - In `packages/core/src/train/layered_training_loop.rs`:
      - Add `global_step: usize` field to `LayeredTrainingLoop` struct.
      - Add `initial_step: usize` parameter to `LayeredTrainingLoop::new()`.
      - Initialize `global_step: initial_step` in the `Ok(Self { ... })` constructor.
      - In `run()`, change `let mut optimizer_steps: usize = 0;` to `let mut optimizer_steps: usize = self.global_step;`.
      - The checkpoint save logic `if optimizer_steps % 500 == 0` already fires correctly relative to the restored step because `optimizer_steps` now starts at `initial_step`.
      - Keep `TrainResult::total_steps` as the local `steps_this_run` counter (initialize a separate `let mut steps_this_run: usize = 0;`, increment it each optimizer step, and return it as `total_steps`) so `total_steps` reflects only the current run's steps, not the cumulative count.
      - Update the single call site in `native_runner.rs` to pass `initial_step`.
    - _Requirements: 3.3, 3.5, 3.6, 4.1, 4.2_

  - [ ]* 2.7 Write unit tests for step counter and checkpoint interval behaviour
    - **Property 6: Checkpoint interval is relative to resumed step** — `test_initial_step_offsets_checkpoint_interval`: construct a `LayeredTrainingLoop` (or a minimal mock) with `initial_step=400`, assert first checkpoint save fires at step 500 (100 additional steps), not at step 0 or 400.
    - **Property 7: total_steps reflects only current-run steps** — `test_total_steps_is_current_run_only`: resume from step 1000, run 50 steps, assert `TrainResult::total_steps == 50`.
    - **Property 8: Checkpoint files contain only LoRA adapter weights** — `test_checkpoint_keys_lora_only`: after calling `save_checkpoint` in a test, open the `.safetensors` file and verify all keys start with `lora_` and none contain `adam`/`moment`/`exp_avg`.
    - _Requirements: 3.5, 3.6, 4.1_

  - [ ] 2.8 Wave 2 gate — `cargo build --workspace` clean + `cargo test --workspace` pass
    - Run `cargo build --workspace` — must complete with no errors.
    - Run `cargo test --workspace` — all tests including new `checkpoint_resumer` tests must pass.
    - _Requirements: all Wave 2_

---

### Wave 3 — Export Adapter Shape Validation

- [ ] 3. Extend `GwenError::ShapeMismatch` and promote `lora_merger` helpers
  - [ ] 3.1 Add `adapter_key: String` field to `GwenError::ShapeMismatch` in `error.rs`
    - In `packages/core/src/error.rs`, change:
      ```rust
      #[error("shape mismatch: adapter {adapter:?} vs base {base:?}")]
      ShapeMismatch { adapter: Vec<usize>, base: Vec<usize> },
      ```
      to:
      ```rust
      #[error("shape mismatch for adapter '{adapter_key}': adapter {adapter:?} vs base {base:?}")]
      ShapeMismatch { adapter_key: String, adapter: Vec<usize>, base: Vec<usize> },
      ```
    - Fix all existing call sites that construct `GwenError::ShapeMismatch` (in `lora_merger.rs` — grep for `ShapeMismatch {` and add `adapter_key: tensor.name.clone()` or equivalent to each).
    - Confirm `cargo build --workspace` compiles after fixing call sites.
    - _Requirements: 6.1, 6.2_

  - [ ] 3.2 Promote `parse_gguf_key` to `pub(crate)` in `lora_merger.rs`
    - In `packages/core/src/train/lora_merger.rs`, change `fn parse_gguf_key` (line ~89) to `pub(crate) fn parse_gguf_key`.
    - Add a private `fn shape_to_2d` in `lora_cli.rs` (or re-export the one in `layered_training_loop.rs` as `pub(crate)`) so `AdapterShapeValidator` can call it without duplicating the GGUF dimension-ordering logic.
    - Confirm no existing tests break.
    - _Requirements: 8.2, 8.3, 8.4_

  - [ ] 3.3 Add `AdapterShapeValidator` and update `export_adapter` signature in `lora_cli.rs`
    - In `packages/core/src/train/lora_cli.rs`, add `base_gguf_path: Option<&Path>` as the fourth parameter to `export_adapter`.
    - Implement `struct AdapterShapeValidator` (private to the module) with:
      ```rust
      fn validate(adapters: &[LoraAdapter], base_gguf_path: &Path) -> std::result::Result<(), GwenError>
      ```
      Logic:
      1. Call `crate::convert::gguf_parser::parse(base_gguf_path)` — propagate error.
      2. Build `HashMap<String, (usize, usize)>` keyed by candle layer name using `parse_gguf_key` + `shape_to_2d` for each tensor.
      3. For each adapter: look up `adapter.layer_name` in the map; check `lora_a.dims()[1] == d_in` and `lora_b.dims()[0] == d_out`; on mismatch return `Err(GwenError::ShapeMismatch { adapter_key: adapter.layer_name.clone(), adapter: ..., base: ... })`.
    - In `export_adapter`, after `let adapters = exporter.extract_adapters(...)`, add:
      ```rust
      if let Some(base_path) = base_gguf_path {
          AdapterShapeValidator::validate(&adapters, base_path)?;
      }
      ```
      This guard must appear **before** any `File::create` / `export_safetensors` call.
    - Update the two existing call sites of `export_adapter` in `train.rs` (tui) to pass `None` as the fourth argument — preserves backward compatibility.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 8.1, 8.2, 8.3_

  - [ ]* 3.4 Write unit tests for `AdapterShapeValidator` in `lora_cli.rs`
    - In `#[cfg(test)]` block:
    - **Property 9 (pass branch): Shape validation passes iff adapter dims match base GGUF** — `test_shape_validation_pass`: build mock adapters with shapes matching mock GGUF tensors; assert `AdapterShapeValidator::validate` returns `Ok(())`.
    - **Property 9 (fail branch): `d_in` mismatch** — `test_shape_validation_mismatch_d_in`: `lora_a` second dim wrong → `Err(GwenError::ShapeMismatch { ... })`.
    - **Property 9 (fail branch): `d_out` mismatch** — `test_shape_validation_mismatch_d_out`: `lora_b` first dim wrong → `Err(GwenError::ShapeMismatch { ... })`.
    - **Property 10: No output file written when any error occurs** — `test_no_partial_file_on_mismatch`: call `export_adapter(ckpt, output, false, Some(bad_base_gguf))` with a shape-mismatch base; assert `output_path` does not exist after the call.
    - **Property 10 (backward compat)** — `test_backward_compat_no_base_gguf`: call with `base_gguf_path=None`; assert validation is skipped, the adapter file is written, AND a warning is printed to stderr containing "shape validation skipped".
    - **Property 10 (dry-run)** — `test_dry_run_with_base_gguf_no_write`: `dry_run=true` + valid `base_gguf_path`; assert no output file created.
    - **Property 11: Shape mismatch error message contains all diagnostic fields** — `test_shape_mismatch_error_format`: construct `GwenError::ShapeMismatch { adapter_key: "lora_layer_0_q_proj".into(), adapter: vec![4096, 8], base: vec![4096, 4096] }`, format with `{e}`, assert the formatted string contains `"lora_layer_0_q_proj"`, `"[4096, 8]"`, and `"[4096, 4096]"`.
    - _Requirements: 5.2, 5.3, 5.4, 5.5, 6.1_

  - [ ] 3.5 Add `--base-gguf` flag to `ExportAdapterArgs` and update dispatch in `train.rs`
    - In `packages/tui/src/commands/train.rs`, add to `ExportAdapterArgs`:
      ```rust
      #[arg(long, value_name = "PATH",
            help = "Base GGUF file for pre-export shape validation (optional). \
                    When provided, lora_a/lora_b dimensions are checked against \
                    the base model tensors before any output is written.")]
      pub base_gguf: Option<PathBuf>,
      ```
    - In `run_export_adapter(args: ExportAdapterArgs)`:
      - Add early path-existence check: if `args.base_gguf.is_some()` and the path does not exist, `bail!("--base-gguf path does not exist: {}", path.display())`.
      - WHEN `args.base_gguf` is `None`, print to stderr before calling `export_adapter`:
        ```
        [export-adapter] warning: --base-gguf not provided; shape validation skipped — exported adapter has not been verified against a base model
        ```
      - Update `lora_cli::export_adapter` call to pass `args.base_gguf.as_deref()` as the fourth argument.
      - On `Err(GwenError::ShapeMismatch { adapter_key, adapter, base })`, print:
        ```
        [export-adapter] shape mismatch: adapter '{adapter_key}' has dims {adapter:?}, expected {base:?} from base GGUF
        ```
        then `std::process::exit(1)`.
    - The `--auto-merge` path in `run_native_path()` passes `None` (shape validation not needed there).
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 6.1, 6.2, 6.3, 6.4_

  - [ ]* 3.6 Write unit test for `--base-gguf` path validation in `train.rs`
    - `test_export_adapter_base_gguf_missing_path`: call `run_export_adapter` with a nonexistent `base_gguf` path; assert `Err` is returned before any checkpoint is loaded.
    - _Requirements: 7.2_

  - [ ] 3.7 Wave 3 gate — `cargo build --workspace` clean + `cargo test --workspace` pass
    - Run `cargo build --workspace` — must complete with no errors.
    - Run `cargo test --workspace` — all tests including new `lora_cli` and `train.rs` tests must pass.
    - _Requirements: all Wave 3_

---

### Wave 4 — E2E Validation

- [ ] 4. Integration tests for checkpoint resume and shape validation round-trips
  - [ ] 4.1 Write E2E integration test for `--resume` round-trip
    - In `packages/core/src/train/` (or `packages/core/tests/`), write `test_e2e_resume`:
      - Build a minimal synthetic GGUF (single-layer, tiny vocab) using the existing `build_minimal_gguf` helper in `lora_merger.rs` tests or a similar fixture.
      - Run one training micro-step (1 optimizer step) via `LayeredTrainingLoop` to produce `checkpoint_000001.safetensors`.
      - Construct a new `LayeredTrainingLoop` with the checkpoint loaded via `checkpoint_resumer::resolve_checkpoint(Explicit(path), ...)` + `load_checkpoint_into_varmap`.
      - Assert the loop's initial `optimizer_steps` counter equals `1` (the restored step).
      - Assert `TrainResult::total_steps` from the second run equals the number of steps taken in that second run only (not 1 + new steps).
    - _Requirements: 2.1, 3.1, 3.3, 3.5, 3.6_

  - [ ] 4.1b Write integration test for `--resume` auto-discovery with no checkpoints (fresh-start edge case)
    - Write `test_e2e_resume_auto_no_checkpoints`:
      - Set up an empty `output_path` directory (no `checkpoint_*.safetensors` files present).
      - Call `checkpoint_resumer::resolve_checkpoint(ResumeMode::Auto, &output_path)`.
      - Assert the result is `Ok((None, 0))` — not an error.
      - Assert that a warning message was emitted to stderr (capture stderr or use a test logger).
      - Assert that a subsequent `LayeredTrainingLoop::new(..., initial_step=0)` starts cleanly from step 0 (no panic, no crash).
    - This covers the "first run" scenario where user passes `--resume` on a fresh training directory.
    - _Requirements: 2.2, 2.3_

  - [ ] 4.2 Write E2E integration test for `export-adapter --base-gguf` shape validation
    - Write `test_e2e_export_shape_validation`:
      - Build a synthetic GGUF with known per-projection dims (e.g. q_proj: `d_in=64, d_out=64`).
      - Build a mock SafeTensors checkpoint with matching `lora_a (rank=4, d_in=64)` and `lora_b (d_out=64, rank=4)` adapters.
      - Call `export_adapter(ckpt, output, false, Some(base_gguf))` — assert `Ok(n)` and `output` file exists.
      - Build a second checkpoint with mismatched dims (e.g. `d_in=128`).
      - Call `export_adapter(ckpt2, output2, false, Some(base_gguf))` — assert `Err(GwenError::ShapeMismatch { ... })` and `output2` does **not** exist on disk.
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.6_

  - [ ]* 4.3 Write property tests for `ResumeMode` CLI round-trip and `shape_to_2d`
    - Using `quickcheck` (already in `[dev-dependencies]`):
    - **Property 1: Explicit resume path round-trips through ResumeMode** — `prop_resume_mode_explicit_roundtrip`: for any non-empty `String` `p`, map through the `--resume p` sentinel logic and assert result is `ResumeMode::Explicit(PathBuf::from(p))`.
    - **Property 4: Checkpoint load populates VarMap (round-trip)** — `prop_checkpoint_varmap_roundtrip`: for small random `(rank, d_in, d_out)` dims, save a SafeTensors file, call `load_checkpoint_into_varmap`, assert tensor values match originals within f32 precision.
    - **Property 12: `shape_to_2d` reverses GGUF dimension ordering** — `prop_shape_to_2d_reversal`: for any `(d_in, d_out)` pair of positive `u64`, assert `shape_to_2d(&[d_in, d_out]) == Ok((d_out as usize, d_in as usize))`.
    - _Requirements: 1.3, 3.1, 8.3_

  - [ ] 4.4 Wave 4 gate — `cargo build --workspace` clean + `cargo test --workspace` pass
    - Run `cargo build --workspace` — must complete with no errors.
    - Run `cargo test --workspace` — all tests including new E2E and property tests must pass.
    - _Requirements: all_

---

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP; the core behavior is fully implemented by the non-optional tasks.
- Each wave ends with an explicit gate task (build + test); do **not** proceed to the next wave until the gate passes.
- `shape_to_2d` decision from task 1.2 (move vs duplicate) must be locked before starting Wave 3; the Wave 3 tasks assume `pub(crate)` promotion but both strategies are valid.
- The `--resume AUTO` sentinel relies on clap's `default_missing_value`; test with both `--resume` (no argument) and `--resume AUTO` explicitly.
- `lora_cli::export_adapter` is called in three places in `train.rs`: `run_export_adapter`, the `--auto-merge` path in `run_native_path`, and a reference in comments. Only the first two are call sites; both must be updated in task 3.5 but only `run_export_adapter` gains `--base-gguf` plumbing.
- Property tests use `quickcheck` which is already in `packages/core` dev-dependencies; no new deps required.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.4"] },
    { "id": 3, "tasks": ["2.3", "2.5"] },
    { "id": 4, "tasks": ["2.6"] },
    { "id": 5, "tasks": ["2.7", "2.8"] },
    { "id": 6, "tasks": ["3.1", "3.2"] },
    { "id": 7, "tasks": ["3.3"] },
    { "id": 8, "tasks": ["3.4", "3.5"] },
    { "id": 9, "tasks": ["3.6", "3.7"] },
    { "id": 10, "tasks": ["4.1", "4.2"] },
    { "id": 11, "tasks": ["4.3"] },
    { "id": 12, "tasks": ["4.4"] }
  ]
}
```
