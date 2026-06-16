# Implementation Plan: GWEN-223 — Checkpoint Resume: Serialize AdamW Optimizer State

## Overview

Persist the full AdamW optimizer state (`m1`, `m2`, `step_t`) to a companion
`checkpoint_{step:06}_adamw.safetensors` file written alongside every weight checkpoint.
On `--resume`, the state is injected back into a parallel `MomentStore` inside
`LayeredTrainingLoop`. Missing or corrupt state files degrade gracefully to GWEN-222
behaviour; save failures never abort the training run.

Implementation follows the 5-wave structure from the design: audit, in-memory bookkeeping,
save path, load/resume path, and validation.

---

## Tasks

- [x] 1. Wave 1 — Audit Candle internals and register new module

  - [x] 1.1 Verify Candle AdamW and safetensors API surface
    - Read `candle_nn/src/optim.rs` in the Cargo registry cache to confirm `AdamW` exposes
      no `m1`/`m2` accessor (only `.step()` is public), validating Option A.
    - Verify `candle_core::safetensors::save` accepts `HashMap<String, Tensor>` entries with
      `DType::U64`. If `U64` is unsupported, note that `step` must be stored as `F32` and
      document the cast in a comment.
    - Confirm `Tensor::id()` is a stable `u64` monotonic counter available for VarMap key
      resolution via pointer-identity scan.
    - Write findings in a comment block at the top of the new `adamw_state.rs` module
      (created in task 2.1).
    - _Requirements: 1.1, 1.3, 3.1_

  - [x] 1.2 Register `adamw_state` module in `train/mod.rs`
    - Add `pub(crate) mod adamw_state;` to `packages/core/src/train/mod.rs`, so the new
      module is visible to `layered_training_loop.rs` and `native_runner.rs`.
    - _Requirements: 7.1_

- [x] 2. Wave 2 — In-memory moment bookkeeping

  - [x] 2.1 Create `adamw_state.rs` with key-translation helpers and `MomentStore` type
    - Create `packages/core/src/train/adamw_state.rs`.
    - Declare `pub(crate) type MomentStore = HashMap<String, (Tensor, Tensor)>;`.
    - Implement `pub(crate) fn varmap_key_to_adamw_prefix(varmap_key: &str) -> Option<String>`:
      maps `l{n}.{proj}.lora_{a|b}` → `layer_{n}.{proj}.lora_{a|b}` and
      `l{n}.lora_{a|b}` → `layer_{n}.lora_{a|b}`; returns `None` for unrecognised patterns.
    - Implement `pub(crate) fn adamw_prefix_to_varmap_key(prefix: &str) -> Option<String>`:
      the inverse transformation; must satisfy the round-trip property.
    - Implement `pub(crate) fn varmap_key_for(var: &Var, data: &HashMap<String, Var>) -> Option<String>`:
      scans the map for `Tensor::id()` equality; returns the matching key string.
    - _Requirements: 4.1, 4.2, 4.3, 4.7_

  - [x]* 2.2 Write unit tests for key-translation functions
    - `test_varmap_key_to_adamw_prefix` — table of `(input, expected_output)` pairs
      covering multi-projection pattern, fallback pattern, and invalid inputs returning `None`.
    - `test_adamw_prefix_to_varmap_key` — symmetrical table for the reverse function.
    - `test_varmap_key_for_resolves` — insert a `Var` into a `VarMap`-style `HashMap`,
      call `varmap_key_for`, assert the correct key string is returned.
    - _Requirements: 4.1, 4.2, 4.3_

  - [x]* 2.3 Write property test for AdamW key round-trip (Property 7)
    - **Property 7: AdamW state key round-trip**
    - **Validates: Requirements 4.7**
    - Use `quickcheck` to generate valid VarMap key strings and assert
      `adamw_prefix_to_varmap_key(varmap_key_to_adamw_prefix(k)?) == Some(k)`.

  - [x] 2.4 Add `moment_store` and `step_t` fields to `LayeredTrainingLoop`
    - Add `moment_store: HashMap<String, (Tensor, Tensor)>` and `step_t: usize` fields
      to the `LayeredTrainingLoop` struct in `layered_training_loop.rs`.
    - In `new()`, initialize `moment_store = HashMap::new()` and `step_t = initial_step`.
    - Add `use std::collections::HashMap;` import if not already present.
    - _Requirements: 3.1, 3.2, 7.1_

  - [x] 2.5 Implement `update_moments` method on `LayeredTrainingLoop`
    - Implement `fn update_moments(&mut self, grads: &GradStore, vars: &[Var], varmap_data: &HashMap<String, Var>) -> Result<()>` as a private method.
    - For each `var` in `vars`: resolve the VarMap key via `varmap_key_for`; skip with a
      warning if unresolvable (requirement 3.7).
    - Retrieve gradient from `grads`; skip the key if gradient shape ≠ stored moment shape,
      emitting a warning (requirement 3.8).
    - On first encounter, initialize `(m1_prev, m2_prev)` to zero tensors of the same shape
      and dtype as the gradient (requirement 3.6).
    - Apply the update:
      - `m1' = 0.9 · m1_prev + 0.1 · g`
      - `m2' = 0.999 · m2_prev + 0.001 · (g ⊙ g)`
    - Increment `self.step_t` by 1 after processing all vars (requirement 3.2).
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8_

  - [x] 2.6 Call `update_moments` in `run()` after each optimizer step
    - In `LayeredTrainingLoop::run()`, immediately after `self.adamw.step(&grads)`, call
      `self.update_moments(&grads, &trainable_vars, &varmap_data)?` where `varmap_data` is
      obtained by locking `self.varmap.data()`.
    - Acquire the VarMap data lock once per optimizer step; release before proceeding.
    - _Requirements: 3.1, 3.2, 3.3, 3.4_

  - [x]* 2.7 Write unit tests for `update_moments`
    - `test_update_moments_initial_zero` — after 1 step with a zero gradient, moments equal zero.
    - `test_update_moments_formula` — with known gradient `g` and zero initial moments,
      assert `m1 ≈ 0.1·g` and `m2 ≈ 0.001·g²` element-wise (tolerance 1e-6).
    - `test_update_moments_step_t_increments` — after `N` calls, `step_t == initial_step + N`.
    - `test_moment_shape_unchanged` — stored moment shapes equal the gradient shape.
    - _Requirements: 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 2.8 Write property test for moment update formula (Property 10)
    - **Property 10: moment update formula correctness**
    - **Validates: Requirements 3.3, 3.4**
    - Use `quickcheck` to generate arbitrary `(m1_0, m2_0, g)` float tensors and assert
      that after one `update_moments` call both moments match the closed-form formula
      element-wise within 1e-6 tolerance.

  - [ ]* 2.9 Write property test for m1/m2 shape invariant (Property 1)
    - **Property 1: m1/m2 shape invariant**
    - **Validates: Requirements 3.5**
    - Use `quickcheck` to generate arbitrary tensor shapes and gradient sequences and assert
      that `moment_store[key].0.shape() == var.shape()` after any number of `update_moments`
      calls.

  - [ ]* 2.10 Write property test for `step_t` increment invariant (implied by Property 2 pre-condition)
    - **Property 2 (counter half): step_t after K steps equals initial_step + K**
    - **Validates: Requirements 3.2**
    - Use `quickcheck` to generate arbitrary initial steps and step counts; assert
      `step_t == initial_step + K` after `K` `update_moments` calls.

  - [ ]* 2.11 Write property test for moment_store entry count invariant (Property 6)
    - **Property 6: moment_store entry count invariant**
    - **Validates: Requirements 5.1, 5.2**
    - For non-fallback mode with N layers and P projections, assert `moment_store.len() == N × P × 2`
      after ≥1 optimizer steps.
    - For fallback mode with N layers, assert `moment_store.len() == N × 2`.

- [x] 3. Checkpoint: Wave 2 complete — all in-memory tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 4. Wave 3 — Save AdamW state to safetensors

  - [x] 4.1 Implement `adamw_state_path` in `adamw_state.rs`
    - Implement `pub(crate) fn adamw_state_path(weight_ckpt_path: &Path) -> PathBuf`.
    - Extract the file stem (without `.safetensors`), append `_adamw.safetensors`, keep
      the same parent directory.
    - Handle the edge case where the filename has no `.safetensors` extension: append
      `_adamw.safetensors` to the stem regardless (requirement 4.5).
    - _Requirements: 4.4, 4.5, 4.6_

  - [x] 4.2 Implement `save_adamw_state` in `adamw_state.rs`
    - Implement `pub(crate) fn save_adamw_state(store: &MomentStore, step_t: usize, output_path: &Path, step: usize) -> Result<()>`.
    - Build the flat tensor map: for each `(varmap_key, (m1, m2))` in `store`, call
      `varmap_key_to_adamw_prefix`, then insert `"{prefix}.m1"` and `"{prefix}.m2"`.
      Skip unrecognised keys silently.
    - Create a `U64` tensor of shape `[1]` for `step_t` under key `"step"`.
      If `U64` is unsupported by `candle_core::safetensors::save` (per Wave 1 audit),
      use `F32` and document the cast.
    - Call `candle_core::safetensors::save(&tensors, &path)?` and emit
      `[checkpoint] AdamW state saved → {path}` on success.
    - Propagate errors to the caller (the method-level wrapper in task 4.3 is what swallows them).
    - _Requirements: 1.1, 1.2, 1.3, 1.4_

  - [x] 4.3 Promote `save_checkpoint` to `save_checkpoint_and_adamw_state` method
    - In `layered_training_loop.rs`, convert the module-level free function `save_checkpoint`
      into a method `fn save_checkpoint_and_adamw_state(&self, step: usize)` on
      `LayeredTrainingLoop`.
    - Step 1: save LoRA weights (GWEN-222 path via `self.varmap.save()`); emit warning on
      failure and return early (do not propagate).
    - Step 2: call `crate::train::adamw_state::save_adamw_state(&self.moment_store, self.step_t, &self.config.output_path, step)`;
      on `Err`, emit `[resume] WARNING: failed to save AdamW state for checkpoint {step}: {e}`
      and continue — do NOT propagate (requirement 1.5).
    - Update the `% 500` call site in `run()` to call `self.save_checkpoint_and_adamw_state(optimizer_steps);`
      (remove the `?` — both saves are best-effort).
    - _Requirements: 1.1, 1.5, 7.3_

  - [x]* 4.4 Write unit tests for `save_adamw_state` and `adamw_state_path`
    - `test_save_adamw_state_creates_file` — after calling `save_adamw_state` with a temp
      dir, the `_adamw.safetensors` file exists.
    - `test_save_adamw_state_filename_pattern` — filename matches `checkpoint_{step:06}_adamw.safetensors`.
    - `test_save_adamw_state_contains_step_key` — the written file contains a `"step"` key.
    - `test_save_adamw_state_key_count` — for N entries in `moment_store`, file contains
      exactly `2N + 1` tensor keys (requirement 1.2).
    - `test_save_adamw_state_empty_store` — empty `MomentStore` writes only `"step"`, no error
      (requirement 1.4).
    - `test_adamw_state_path_derivation` — verify `_adamw.safetensors` suffix for standard and
      non-standard checkpoint names (requirements 4.4, 4.5, 4.6).
    - _Requirements: 1.1, 1.2, 1.3, 1.4, 4.4, 4.5, 4.6_

  - [ ]* 4.5 Write property test for `adamw_state_path` purity (Property 8)
    - **Property 8: adamw_state_path is a pure function of weight_ckpt_path**
    - **Validates: Requirements 4.4, 4.5, 4.6**
    - Use `quickcheck` to generate arbitrary path strings and assert that the result
      has the same parent, starts with the input stem, and ends with `_adamw.safetensors`.

- [x] 5. Checkpoint: Wave 3 complete — save path tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Wave 4 — Load AdamW state on resume

  - [x] 6.1 Implement `load_adamw_state` in `adamw_state.rs`
    - Implement `pub(crate) fn load_adamw_state(weight_ckpt_path: &Path) -> Result<Option<(MomentStore, usize)>>`.
    - If `adamw_state_path(weight_ckpt_path)` does not exist, return `Ok(None)` (requirement 2.3).
    - Load tensors with `candle_core::safetensors::load(&adamw_path, &Device::Cpu)`;
      propagate errors to the method-level wrapper (which logs and falls back).
    - Extract `step_t` from the `"step"` key via `.to_vec1::<u64>()`; return `Err` if missing
      (requirement 2.2).
    - Reconstruct `MomentStore` by pairing `*.m1` / `*.m2` keys via `adamw_prefix_to_varmap_key`;
      skip unpaired `m1` entries silently (partial-write scenario).
    - Apply shape-validation pass: for each loaded `(m1, m2)` pair, if `m1.shape()` or
      `m2.shape()` does not match the corresponding VarMap Var's shape, drop the entry and
      emit a per-key warning (requirement 2.5, 2.6).
    - Return `Ok(Some((store, step_t as usize)))` on success.
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.1, 6.2, 6.3_

  - [x] 6.2 Add `load_adamw_state` method to `LayeredTrainingLoop`
    - Add `pub fn load_adamw_state(&mut self, weight_ckpt_path: &Path)` (infallible, no `?`) to
      `LayeredTrainingLoop` in `layered_training_loop.rs`.
    - On `Ok(Some((store, step_t)))`: set `self.moment_store = store` and `self.step_t = step_t`;
      emit `[resume] AdamW state restored: {N} moment pairs, step_t={step_t}`.
    - On `Ok(None)`: emit the "not found" warning, leave `moment_store` empty and `step_t`
      at `initial_step` (GWEN-222 fallback, requirement 2.3).
    - On `Err(e)`: emit `[resume] WARNING: failed to load AdamW state: {e}. Resuming with fresh optimizer.`;
      call `self.moment_store.clear()`; leave `step_t` at `initial_step` (requirement 2.4).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 7.2, 7.4_

  - [x] 6.3 Call `load_adamw_state` in `native_runner.rs`
    - In `run_native_local`, after the existing `training_loop.load_checkpoint(path)?` call,
      add `training_loop.load_adamw_state(path);` (no `?` — infallible by design).
    - _Requirements: 2.1, 2.2, 7.1_

  - [x]* 6.4 Write unit tests for `load_adamw_state`
    - `test_load_adamw_state_missing_file` — returns `Ok(None)`, no panic (requirement 6.3).
    - `test_load_adamw_state_roundtrip` — save then load; assert all `(m1, m2)` tensors are
      element-wise equal within 1e-7 and shapes are identical (requirement 6.1).
    - `test_load_adamw_state_step_roundtrip` — saved step `S` is restored as exactly `S`
      (requirement 6.2).
    - `test_load_adamw_state_corrupt_file` — write garbage bytes; `load_adamw_state` returns `Err`.
    - `test_load_adamw_state_missing_step_key` — valid safetensors with no `"step"` key returns `Err`.
    - `test_load_adamw_state_shape_mismatch_dropped` — mismatched-shape entries are dropped with
      a warning; non-mismatched entries are still restored (requirements 2.5, 2.6).
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 6.1, 6.2, 6.3_

  - [ ]* 6.5 Write property test for missing file graceful fallback (Property 3)
    - **Property 3: missing `_adamw` file does not abort training**
    - **Validates: Requirements 2.3, 7.2, 7.4**
    - Use `quickcheck` to generate arbitrary weight checkpoint paths where the weight file
      exists but the `_adamw` file does not, and assert `load_adamw_state` returns `Ok(None)`.

  - [ ]* 6.6 Write property test for save-failure non-fatal (Property 4)
    - **Property 4: save failure does not abort training**
    - **Validates: Requirements 1.4, 1.5, 7.3**
    - Simulate a write failure (read-only temp directory) and assert that
      `save_checkpoint_and_adamw_state` does not panic and the weight checkpoint exists.

  - [ ]* 6.7 Write property test for step round-trip (Property 2)
    - **Property 2: step_t after restore equals saved step**
    - **Validates: Requirements 2.2, 6.2**
    - Use `quickcheck` to generate arbitrary step values `S` (where `S % 500 == 0`), save,
      load into a fresh loop, assert `step_t == S`.

  - [ ]* 6.8 Write property test for round-trip fidelity (Property 5)
    - **Property 5: round-trip fidelity**
    - **Validates: Requirements 6.1**
    - Use `quickcheck` to generate `MomentStore` instances with arbitrary F32 tensors in
      `(-1e6, 1e6)` and assert that every element after save+load is within 1e-7 of the original.

  - [ ]* 6.9 Write property test for step tensor parseability (Property 9)
    - **Property 9: step tensor is parseable from any valid _adamw file**
    - **Validates: Requirements 1.3, 6.2**
    - Use `quickcheck` with arbitrary `usize` step values; assert the written file's `"step"` key
      round-trips through `to_vec1::<u64>()[0]` back to the original value.

  - [ ]* 6.10 Write integration test for end-to-end resume
    - `test_e2e_true_resume` — run N steps on a micro-GGUF fixture, force checkpoint,
      construct a fresh `LayeredTrainingLoop`, call `load_checkpoint` + `load_adamw_state`,
      assert `moment_store` is non-empty and `step_t == N`.
    - _Requirements: 2.1, 2.2, 5.1_

- [x] 7. Checkpoint: Wave 4 complete — load and integration tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Wave 5 — Validation tests

  - [x] 8.1 Implement `test_moment_values_match_adamw_internal`
    - Run the training loop for K steps capturing the pre-clip gradient at each step.
    - Independently apply the AdamW moment formula with the same β1/β2 and captured gradients.
    - Assert element-wise max error between `moment_store` values and the reference < 1e-5.
    - _Requirements: 3.3, 3.4_

  - [ ]* 8.2 Implement `test_loss_curve_no_step_back`
    - Resume mid-run on a micro-GGUF fixture; assert that loss at step N+1 after resume ≤
      loss at step N before pause + small ε (no catastrophic spike).
    - _Requirements: 2.1, 2.2, 6.1_

  - [ ]* 8.3 Implement `test_no_regression_fresh_run`
    - Run a fresh (no-resume) training run with GWEN-223 code and assert identical loss
      values as a baseline run without `moment_store` active (`moment_store` is empty;
      behaviour must be identical to GWEN-222).
    - _Requirements: 7.1_

- [x] 9. Final checkpoint — Ensure all tests pass
  - Run `cargo test -p gwen-core -- train` and confirm the full test suite is green.
  - Ensure all tests pass, ask the user if questions arise.

---

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP.
- Each task references specific requirements for traceability.
- The design is purely additive: no public API surfaces change shape, and the `_adamw` file is
  always optional from the reader's perspective.
- Waves 1–4 correspond directly to code changes; Wave 5 is validation only.
- `quickcheck` and `tempfile` are already in `[dev-dependencies]` — no new crates required.
- The `U64` dtype question (Wave 1 audit) may change the `step` encoding to `F32`; the
  implementation must reflect whichever the audit confirms.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2"] },
    { "id": 1, "tasks": ["2.1"] },
    { "id": 2, "tasks": ["2.2", "2.3", "2.4"] },
    { "id": 3, "tasks": ["2.5", "2.6"] },
    { "id": 4, "tasks": ["2.7", "2.8", "2.9", "2.10", "2.11", "4.1"] },
    { "id": 5, "tasks": ["4.2"] },
    { "id": 6, "tasks": ["4.3"] },
    { "id": 7, "tasks": ["4.4", "4.5", "6.1"] },
    { "id": 8, "tasks": ["6.2"] },
    { "id": 9, "tasks": ["6.3"] },
    { "id": 10, "tasks": ["6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10"] },
    { "id": 11, "tasks": ["8.1"] },
    { "id": 12, "tasks": ["8.2", "8.3"] }
  ]
}
```
