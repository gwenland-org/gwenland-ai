# Implementation Plan: GWEN-216 — Selective Layer Loading for LoRA Training

## Overview

Five waves of implementation, each building on the previous. Wave 1 establishes the index data structures, Wave 2 adds the zero-copy mmap slicer, Wave 3 wires everything into the training loop, Wave 4 adds the benchmark binary, and Wave 5 verifies integration and regressions. Each task is scoped to a single file or coherent unit, executable by Claude Code.

## Tasks

### Wave 1 — Foundation: LayerSlice + LayerIndex

- [ ] 1.1 Create `packages/core/src/train/layer_loader.rs` with `LayerSlice` struct

  **File**: `packages/core/src/train/layer_loader.rs`

  Define the `LayerSlice` struct:
  ```rust
  #[derive(Debug, Clone)]
  pub struct LayerSlice {
      pub layer_idx:   usize,
      pub tensor_name: String,
      pub byte_offset: u64,
      pub byte_len:    usize,
  }
  ```
  Add a doc comment explaining its role (lightweight tensor descriptor, no heap beyond Vec construction).

  **Acceptance**: File compiles; `LayerSlice` is `pub`, `Clone`, `Debug`. No other items yet.

---

- [ ] 1.2 Add `LayerIndex` struct and `LayerIndex::scan` to `layer_loader.rs`

  **File**: `packages/core/src/train/layer_loader.rs`

  Add:
  ```rust
  pub struct LayerIndex {
      slices: Vec<LayerSlice>,
      pub num_layers: usize,
  }

  impl LayerIndex {
      pub fn scan(file: &GgufFile) -> Self { ... }
      pub fn layer_slices(&self, n: usize) -> &[LayerSlice] { ... }
  }
  ```

  `scan` algorithm: iterate `file.tensors`; for each tensor whose `name` starts with `"model.layers."`, parse the layer index `N` from the next segment (chars up to the next `.`); build a `LayerSlice` with `byte_offset = tensor_info.data_offset`, `byte_len = tensor_info.data_size`; sort `slices` by `(layer_idx, tensor_name)`; compute `num_layers = slices.iter().map(|s| s.layer_idx).max().map(|m| m + 1).unwrap_or(0)`.

  `layer_slices(n)` returns the contiguous subslice of `self.slices` for layer `n`.

  **Required imports**: `use crate::convert::gguf_parser::GgufFile;`

  **Acceptance**: `scan` correctly filters `model.layers.*` tensors; `layer_slices(n)` returns the right subset; non-layer tensors excluded.

---

- [ ] 1.3 Write unit tests for `LayerIndex::scan` in `layer_loader.rs`

  **File**: `packages/core/src/train/layer_loader.rs` (inside `#[cfg(test)] mod tests`)

  Tests to write:
  - `test_scan_empty`: empty `GgufFile` → `num_layers == 0`
  - `test_scan_filters_non_layer_tensors`: `"token_embd.weight"`, `"output_norm.weight"` excluded
  - `test_scan_extracts_layer_count`: `"model.layers.0.*"`, `"model.layers.1.*"` → `num_layers == 2`
  - `test_scan_sorted_order`: output slices sorted by `(layer_idx, tensor_name)`
  - `test_layer_slices_out_of_range`: `layer_slices(99)` on a 2-layer index → empty slice, no panic

  **Acceptance**: All 5 tests pass with `cargo test -p gwenland-core layer_loader`.

---

- [ ] 1.4 Write property-based tests for `LayerIndex` using `quickcheck`

  **File**: `packages/core/src/train/layer_loader.rs` (`#[cfg(test)] mod tests`)

  ```rust
  use quickcheck_macros::quickcheck;
  ```

  **Property 1** — `scan_then_layer_slices_covers_all_matched_tensors`: for any vec of `(u8, String)` pairs used as `(layer_idx, name_suffix)`, the total count of returned slices equals the count of tensors with `"model.layers.*"` names.

  **Property 2** — `scan_num_layers_equals_max_plus_one`: `index.num_layers == max(layer_idx) + 1` for any non-empty input.

  **Property 3** — `layer_slices_sorted`: `layer_slices(n)` is sorted by `tensor_name` for any valid `n`.

  **Acceptance**: `cargo test -p gwenland-core layer_loader` passes including quickcheck tests.

  **PBT**: This is a property-based test task.

---

- [ ] 1.5 Register `layer_loader` module in `packages/core/src/train/mod.rs`

  **File**: `packages/core/src/train/mod.rs`

  Add the line:
  ```rust
  pub mod layer_loader;
  ```

  **Acceptance**: `cargo build -p gwenland-core` compiles without errors.

---

### Wave 2 — Core Loading: LayerLoader + LoadedLayer

- [ ] 2.1 Add `LoadedLayer` struct to `layer_loader.rs`

  **File**: `packages/core/src/train/layer_loader.rs`

  Add:
  ```rust
  pub struct LoadedLayer<'mmap> {
      pub slices:            Vec<(&'mmap str, &'mmap [u8])>,
      pub(crate) mmap_range: std::ops::Range<usize>,
      pub(crate) mmap_data:  &'mmap memmap2::Mmap,
  }

  impl<'mmap> LoadedLayer<'mmap> {
      pub fn unload(self) { drop(self); }
  }

  impl<'mmap> Drop for LoadedLayer<'mmap> {
      fn drop(&mut self) {
          #[cfg(unix)]
          { let _ = self.mmap_data.advise_range(
              memmap2::Advice::DontNeed,
              self.mmap_range.start,
              self.mmap_range.len(),
          ); }
      }
  }
  ```

  **Acceptance**: Compiles on all platforms; `#[cfg(unix)]` gate present; `unload()` compiles.

---

- [ ] 2.2 Add `pub fn mmap(&self) -> &memmap2::Mmap` accessor to `MmapLoader` in `loader.rs`

  **File**: `packages/core/src/engine/loader.rs`

  Add to `impl MmapLoader`:
  ```rust
  /// Return a reference to the underlying `Mmap` for callers that need
  /// to issue madvise calls on specific byte ranges (e.g. `LoadedLayer`).
  pub fn mmap(&self) -> &memmap2::Mmap {
      &self.data
  }
  ```

  **Acceptance**: `cargo build -p gwenland-core` passes; existing `loader.rs` tests still pass.

---

- [ ] 2.3 Add `LayerLoader` struct and `open` / `load_layer` to `layer_loader.rs`

  **File**: `packages/core/src/train/layer_loader.rs`

  Add:
  ```rust
  pub struct LayerLoader {
      mmap:  MmapLoader,
      index: LayerIndex,
  }

  impl LayerLoader {
      pub fn open(path: &Path) -> anyhow::Result<Self>;
      pub fn num_layers(&self) -> usize;
      pub fn load_layer<'a>(&'a self, n: usize) -> anyhow::Result<LoadedLayer<'a>>;
  }
  ```

  `open` uses `MmapLoader::open_with_mode(path, LoadMode::Lazy)` and `gguf_parser::parse(path)` to build the index. `load_layer(n)` returns `Err` if `n >= num_layers()`, otherwise slices `mmap.as_bytes()` at the offsets from `index.layer_slices(n)` and constructs `LoadedLayer`.

  **Acceptance**: `load_layer(n)` returns `Ok(LoadedLayer)` with correct slice count; out-of-range returns `Err`.

---

- [ ] 2.4 Write unit tests for `LayerLoader` and `LoadedLayer`

  **File**: `packages/core/src/train/layer_loader.rs` (`#[cfg(test)]`)

  Add helper `fn write_minimal_gguf(tensors: &[(&str, &[u8])]) -> NamedTempFile` that writes a minimal valid GGUF v3 binary.

  Tests:
  - `test_layer_loader_open_invalid_path` — `Err` on nonexistent file
  - `test_layer_loader_open_invalid_magic` — `Err` on garbage bytes
  - `test_layer_loader_load_layer_oor` — valid open, `load_layer(999)` returns `Err`
  - `test_layer_loader_load_layer_ok` — minimal GGUF with one `model.layers.0.*` tensor, verify `slices.len() == 1` and byte content is correct

  **Acceptance**: All tests pass with `cargo test -p gwenland-core layer_loader`.

---

- [ ] 2.5 Write property-based tests for `LoadedLayer` byte range coverage

  **File**: `packages/core/src/train/layer_loader.rs` (`#[cfg(test)]`)

  **Property 4** — `loaded_layer_mmap_range_covers_all_slices`: for any valid minimal GGUF with N layer tensors of known offsets, `loaded.mmap_range.start == min(byte_offset)` and `loaded.mmap_range.end == max(byte_offset + byte_len)`.

  **Property 5** — `load_then_unload_does_not_panic`: for any valid minimal GGUF, `load_layer(n)` followed by `.unload()` does not panic; a subsequent `load_layer(n)` is valid.

  Use the `write_minimal_gguf` helper. Generate varying tensor counts and byte lengths.

  **Acceptance**: `cargo test -p gwenland-core layer_loader` passes all quickcheck runs.

  **PBT**: This is a property-based test task.

---

### Wave 3 — LayeredTrainingLoop

- [ ] 3.1 Extract `step_accumulated` as a `pub(crate)` free function in `training_loop.rs`

  **File**: `packages/core/src/train/training_loop.rs`

  Extract the current `TrainingLoop::step_accumulated` method body into:
  ```rust
  pub(crate) fn step_accumulated(
      adamw:  &mut AdamW,
      stores: &[candle_core::backprop::GradStore],
  ) -> anyhow::Result<()>
  ```

  Update the existing `TrainingLoop::step_accumulated` to delegate to this free function. No behaviour change.

  **Acceptance**: Existing `TrainingLoop` behaviour unchanged; free function callable from sibling modules; `cargo test` passes.

---

- [ ] 3.2 Create `packages/core/src/train/layered_training_loop.rs` with struct + `new`

  **File**: `packages/core/src/train/layered_training_loop.rs`

  Define `LayeredTrainingLoop` struct with fields: `config: NewTrainConfig`, `layer_loader: LayerLoader`, `batches: Vec<Vec<Tensor>>`, `varmap: VarMap`, `adamw: AdamW`, `tx: Option<Sender<String>>`.

  Implement `new(config, gguf_path, batches, varmap, tx) -> Result<Self>`:
  - `LayerLoader::open(gguf_path)?`
  - Seed `AdamW` from `varmap.all_vars()` with same `ParamsAdamW` defaults as `TrainingLoop::new()`
  - Return `Err` if `varmap.all_vars().is_empty()` or `layer_loader.num_layers() == 0`

  **Acceptance**: `new()` compiles; `cargo build -p gwenland-core` passes.

---

- [ ] 3.3 Implement `LayeredTrainingLoop::run`

  **File**: `packages/core/src/train/layered_training_loop.rs`

  Implement `pub fn run(&mut self) -> anyhow::Result<TrainResult>` following `design.md § Algorithm 3`:
  - Outer: epoch loop; middle: batch loop; inner: layer loop
  - `global_batch += 1` per `(layer, batch)` pair
  - `load_layer(n)` → dequantise each tensor slice (Q8_0→f32) → build candle Tensors → construct `LoraLayer`
  - Forward → cross-entropy → scaled backward → `grad_stores.push(grads)`
  - At accumulation boundary: call `step_accumulated(&mut self.adamw, &grad_stores)?`, clear
  - Every 500 `optimizer_steps`: `varmap.save(checkpoint_path)?`
  - Emit same JSON progress events as `TrainingLoop::run()`
  - `loaded.unload()` at end of inner iteration; drop all layer-scoped values
  - Return `TrainResult`

  No new `unsafe` blocks.

  **Acceptance**: Compiles; logic matches pseudocode in design.md.

---

- [ ] 3.4 Write unit tests for `LayeredTrainingLoop`

  **File**: `packages/core/src/train/layered_training_loop.rs` (`#[cfg(test)]`)

  Tests:
  - `test_new_rejects_empty_varmap`
  - `test_new_rejects_zero_layers`
  - `test_run_single_epoch_produces_result`: 2-layer synthetic GGUF, 1 batch, 1 epoch → `result.total_steps >= 1` and `result.final_loss.is_finite()`
  - `test_run_emits_done_json`: verify `{"event":"done",...}` JSON emitted on stdout

  **Acceptance**: All 4 tests pass.

---

- [ ] 3.5 Write property-based tests for `LayeredTrainingLoop` gradient accumulation

  **File**: `packages/core/src/train/layered_training_loop.rs` (`#[cfg(test)]`)

  **Property 6** — `total_steps_matches_formula`: for any `(num_layers: u8 in [1..8], num_batches: u8 in [1..4], epochs: u8 in [1..3], grad_accum: u8 in [1..8])`, `result.total_steps == ceil(num_layers × num_batches × epochs / grad_accum)`.

  **Property 7** — `final_loss_is_finite`: for any `(num_layers: u8 in [1..4], grad_accum: u8 in [1..4])`, `result.final_loss.is_finite() == true`.

  Use `quickcheck::QuickCheck::new().tests(20)` to limit wall time.

  **Acceptance**: `cargo test -p gwenland-core layered_training_loop` passes all quickcheck runs.

  **PBT**: This is a property-based test task.

---

- [ ] 3.6 Register `layered_training_loop` in `packages/core/src/train/mod.rs`

  **File**: `packages/core/src/train/mod.rs`

  Add:
  ```rust
  pub mod layered_training_loop;
  pub use layer_loader::{LayerIndex, LayerLoader, LayerSlice, LoadedLayer};
  pub use layered_training_loop::LayeredTrainingLoop;
  ```

  **Acceptance**: `cargo build -p gwenland-core` compiles; no breaking changes to existing public API.

---

### Wave 4 — Benchmark Binary

- [ ] 4.1 Create `packages/core/src/bin/bench_layer_loader.rs`

  **File**: `packages/core/src/bin/bench_layer_loader.rs`

  CLI: `bench_layer_loader <gguf_path> [--layer N] [--iterations M] [--compare-full] [--format text|json]`

  Implementation:
  1. Parse args (manual style matching `bench_ggqr.rs` — no `clap`)
  2. `LayerLoader::open(path)?`
  3. For each target layer: read RSS via `sysinfo`, time `load_layer(n)`, report raw bytes / RSS delta / load ms; time `unload()`, report RSS after unload
  4. `--compare-full`: run `gguf_loader::load_and_dequant` baseline, report RSS delta and time
  5. Summary: min/max/avg per-layer RSS delta
  6. `--format json`: one JSON object per line; `--format text`: human-readable table

  **Acceptance**: `cargo build --bin bench_layer_loader -p gwenland-core` succeeds; running on a small test file produces well-formed output.

---

- [ ] 4.2 Add `[[bin]]` entry to `packages/core/Cargo.toml`

  **File**: `packages/core/Cargo.toml`

  Add after existing `[[bin]]` entries:
  ```toml
  [[bin]]
  name = "bench_layer_loader"
  path = "src/bin/bench_layer_loader.rs"
  ```

  **Acceptance**: `cargo build --bin bench_layer_loader -p gwenland-core` resolves the binary target without errors.

---

- [ ] 4.3 Write smoke tests for `bench_layer_loader` argument parsing

  **File**: `packages/core/src/bin/bench_layer_loader.rs` (inline `#[cfg(test)]`)

  Tests:
  - `test_parse_args_default`: no flags → `layer = None`, `iterations = 1`, `compare_full = false`, `format = Text`
  - `test_parse_args_layer_flag`: `--layer 5` → `layer = Some(5)`
  - `test_parse_args_json_format`: `--format json` → `format = Json`
  - `test_parse_args_invalid_layer`: `--layer abc` → returns `Err`

  **Acceptance**: `cargo test --bin bench_layer_loader -p gwenland-core` passes.

---

### Wave 5 — Integration & Regression

- [ ] 5.1 Verify GWEN-213 compatibility: VarMap key format unchanged

  **File**: `packages/core/src/train/layered_training_loop.rs` (`#[cfg(test)]`)

  Integration test: create `VarMap`, insert `lora_a`/`lora_b` Vars via `VarBuilder::get_with_hints`, export via `LoraExporter`, assert exported SafeTensors contains keys `"lora_a"` and `"lora_b"`. Confirms `LayeredTrainingLoop` does not change the VarMap schema from GWEN-213.

  **Acceptance**: Test passes; no changes to `lora_bridge.rs` or `lora_merger.rs` required.

---

- [ ] 5.2 Run full existing test suite and confirm zero regressions

  **Action**: Run `cargo test -p gwenland-core` and verify all pre-existing tests still pass. The 13 LoRA training tests and all loader/parser tests must pass.

  **Acceptance**: `cargo test -p gwenland-core` exit code 0; test count ≥ pre-GWEN-216 baseline.

---

- [ ] 5.3 Write no-full-load invariant integration test

  **File**: `packages/core/src/train/layered_training_loop.rs` (`#[cfg(test)]`)

  Test: `test_no_more_than_one_layer_live_at_a_time` (Unix only, `#[cfg(unix)]`).

  Use a shared `AtomicUsize` counter (test-only, behind `#[cfg(test)]`) incremented in `LoadedLayer` construction and decremented in `Drop`, tracking total live mmap bytes. Assert the counter never exceeds `2 × largest_single_layer_byte_count` at any point during a 3-epoch synthetic training run.

  **Acceptance**: Test passes on Linux/macOS; test is `#[cfg(unix)]` gated.

---

- [ ] 5.4 Final module wiring and public re-exports in `packages/core/src/train/mod.rs`

  **File**: `packages/core/src/train/mod.rs`

  Confirm final module list includes all existing entries plus:
  ```rust
  pub mod layer_loader;
  pub mod layered_training_loop;
  pub use layer_loader::{LayerIndex, LayerLoader, LayerSlice, LoadedLayer};
  pub use layered_training_loop::LayeredTrainingLoop;
  ```

  **Acceptance**: `cargo build -p gwenland-core` clean build; `cargo test -p gwenland-core` passes.

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 1,
      "name": "Foundation: LayerSlice + LayerIndex",
      "tasks": ["1.1", "1.2", "1.3", "1.4", "1.5"]
    },
    {
      "wave": 2,
      "name": "Core Loading: LayerLoader + LoadedLayer",
      "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5"],
      "dependsOn": ["1.2", "1.5"]
    },
    {
      "wave": 3,
      "name": "LayeredTrainingLoop",
      "tasks": ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6"],
      "dependsOn": ["2.3"]
    },
    {
      "wave": 4,
      "name": "Benchmark Binary",
      "tasks": ["4.1", "4.2", "4.3"],
      "dependsOn": ["2.3"]
    },
    {
      "wave": 5,
      "name": "Integration and Regression",
      "tasks": ["5.1", "5.2", "5.3", "5.4"],
      "dependsOn": ["3.3", "4.1"]
    }
  ],
  "taskDependencies": {
    "1.2": ["1.1"],
    "1.3": ["1.2"],
    "1.4": ["1.2"],
    "1.5": ["1.2"],
    "2.1": ["1.5"],
    "2.2": ["1.5"],
    "2.3": ["1.2", "2.1", "2.2"],
    "2.4": ["2.3"],
    "2.5": ["2.3"],
    "3.1": ["1.5"],
    "3.2": ["2.3"],
    "3.3": ["3.1", "3.2"],
    "3.4": ["3.3"],
    "3.5": ["3.3"],
    "3.6": ["3.2"],
    "4.2": [],
    "4.1": ["2.3", "4.2"],
    "4.3": ["4.1"],
    "5.1": ["3.2"],
    "5.2": ["3.6", "4.3"],
    "5.3": ["3.3"],
    "5.4": ["5.2"]
  }
}
```

## Notes

- **No new unsafe**: All `unsafe` is confined to the existing `MmapOptions::new().map(&file)` block in `loader.rs`. Task 2.2 adds a safe accessor, not a new unsafe block.
- **No new dependencies**: `memmap2`, `anyhow`, `candle-core`, `candle-nn`, `sysinfo`, `quickcheck`, `quickcheck_macros` are all already in `Cargo.toml`.
- **Windows compatibility**: Every `MADV_DONTNEED` call is inside `#[cfg(unix)]`. Tasks 2.1 and 5.3 have explicit notes about this.
- **PBT tasks**: Tasks 1.4, 2.5, and 3.5 are property-based test tasks using the `quickcheck` crate already in `[dev-dependencies]`.
- **Synthetic GGUF helper**: Task 2.4 introduces `write_minimal_gguf()` — a test helper that should be reused across Wave 2–3 tests to avoid duplication.
- **Qwen3-1.7B tensor naming**: Layer tensors follow the pattern `model.layers.{N}.{component}.{proj}.weight` (e.g. `model.layers.0.self_attn.q_proj.weight`). The `LayerIndex::scan` parser splits on `.` after the `model.layers.` prefix to extract `N`.
