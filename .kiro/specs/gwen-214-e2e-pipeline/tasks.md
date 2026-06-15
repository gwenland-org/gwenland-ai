# Implementation Plan: GWEN-214 — E2E Pipeline Validation & Benchmark Update

## Overview

Five waves. Wave 1 adds the layer-load benchmark module. Wave 2 extends the existing
`BenchmarkResult` with the new fields and adds the report formatter. Wave 3 wires
mistral.rs into the inference benchmark path. Wave 4 builds the four E2E integration
tests. Wave 5 verifies zero regressions and validates all assertions against real output.

Each task targets a single file or a coherent unit. All tasks are executable by Claude Code.
Zero new Cargo dependencies. All tasks are Windows-compatible.

---

## Tasks

### Wave 1 — Layer-Load Benchmark Module

- [ ] 1.1 Create `packages/core/src/benchmark/layer_load_bench.rs`

  **File**: `packages/core/src/benchmark/layer_load_bench.rs`

  Define the two public structs and the benchmark entry point:

  ```rust
  use serde::{Serialize, Deserialize};

  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct LayerLoadSample {
      pub layer_idx:    usize,
      pub load_us:      u64,
      pub unload_us:    u64,
      pub rss_delta_mb: f64,
      pub byte_total:   u64,
      pub slice_count:  usize,
  }

  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub struct LayerLoadResult {
      pub samples:               Vec<LayerLoadSample>,
      pub file_size_bytes:       u64,
      pub num_layers:            usize,
      pub min_load_us:           u64,
      pub max_load_us:           u64,
      pub mean_load_us:          f64,
      pub peak_rss_mb:           f64,
      pub full_load_estimate_mb: f64,
  }

  pub fn run_layer_load_bench(
      path: &std::path::Path,
      sample_layers: Option<usize>,
  ) -> anyhow::Result<LayerLoadResult>
  ```

  Implementation of `run_layer_load_bench`:
  - `LayerLoader::open(path)?` using `crate::train::LayerLoader`
  - Build `indices` vector: if `sample_layers = Some(k)` and `k < num_layers`,
    use `(0..k).map(|i| i * num_layers / k)`; else use `0..num_layers`
  - Per-layer loop (see `design.md § Algorithm 5`):
    - Sample RSS before via `sysinfo::System::refresh_processes_specifics`
    - `Instant::now()` → `loader.load_layer(n)?` → record `load_us`
    - Sample RSS after, compute `rss_delta_mb`
    - `Instant::now()` → `loaded.unload()` → record `unload_us`
    - Collect `byte_total` and `slice_count` from `loaded.slices` BEFORE unload
  - Aggregate: min/max/mean `load_us`, `peak_rss_mb`, `full_load_estimate_mb`
  - Return `LayerLoadResult`

  **Required imports**:
  ```rust
  use crate::train::LayerLoader;
  use sysinfo::{ProcessRefreshKind, RefreshKind, System};
  ```

  **Acceptance**: `cargo build -p gwenland-core` compiles without errors.

---

- [ ] 1.2 Write unit tests for `run_layer_load_bench` in `layer_load_bench.rs`

  **File**: `packages/core/src/benchmark/layer_load_bench.rs` (`#[cfg(test)]`)

  Tests:
  - `test_run_layer_load_bench_invalid_path` — returns `Err` for nonexistent file
  - `test_run_layer_load_bench_sample_layers_subset` — using `write_minimal_gguf_pub`
    from `crate::train::layer_loader`, create a 3-layer GGUF, call
    `run_layer_load_bench(path, Some(2))` → `result.samples.len() == 2`
  - `test_run_layer_load_bench_all_layers` — 3-layer GGUF, `sample_layers = None`
    → `result.samples.len() == 3`
  - `test_full_load_estimate_formula` — `result.full_load_estimate_mb ==
    result.peak_rss_mb * result.num_layers as f64`

  **Feature flag required**: `required-features = ["test-utils"]` since `write_minimal_gguf_pub`
  is gated on that feature.

  **Acceptance**: `cargo test -p gwenland-core --features test-utils layer_load_bench` passes.

---

- [ ] 1.3 Register `layer_load_bench` in `packages/core/src/benchmark/mod.rs`

  **File**: `packages/core/src/benchmark/mod.rs`

  Add:
  ```rust
  pub mod layer_load_bench;
  pub use layer_load_bench::{LayerLoadResult, LayerLoadSample};
  ```

  Also add the new field to `BenchmarkResult`:
  ```rust
  pub struct BenchmarkResult {
      pub cold_start:         Option<ColdStartResult>,
      pub inference:          Option<InferenceResult>,
      pub convert:            Option<ConvertBenchResult>,
      pub memory:             Option<MemoryResult>,
      pub total_elapsed_secs: f64,
      // NEW
      pub layer_load:         Option<LayerLoadResult>,
  }
  ```

  Update the `BenchmarkResult` construction in `run_benchmarks` to include
  `layer_load: None` (existing callers are unaffected; the new field is populated
  only when `run_layer_load_bench` is explicitly invoked).

  **Acceptance**: `cargo build -p gwenland-core` passes; no existing tests broken.

---

### Wave 2 — Benchmark Report Formatter

- [ ] 2.1 Extend `InferenceResult` with `backend` and `model_file` fields

  **File**: `packages/core/src/benchmark/mod.rs`

  Update `InferenceResult`:
  ```rust
  #[derive(Debug, Clone, serde::Serialize)]
  pub struct InferenceResult {
      pub tokens_per_sec: f64,
      pub total_tokens:   usize,
      pub elapsed_secs:   f64,
      // NEW — defaults to "proxy" for backward compatibility
      pub backend:        String,
      // NEW — None when using the proxy path (model name unknown)
      pub model_file:     Option<String>,
  }
  ```

  Update the existing proxy-based construction in `inference::run_inference_bench()` to
  set `backend: "proxy".to_string(), model_file: None` on the returned `InferenceResult`.

  **Acceptance**: `cargo build -p gwenland-core` compiles; no test changes required.

---

- [ ] 2.2 Add `OutputFormat` enum and `format_benchmark_report` to `benchmark/mod.rs`

  **File**: `packages/core/src/benchmark/mod.rs`

  Add:
  ```rust
  #[derive(Debug, Clone, Copy)]
  pub enum OutputFormat {
      Json,
      Text,
  }

  /// Render a BenchmarkResult as a String in the given format.
  pub fn format_benchmark_report(result: &BenchmarkResult, fmt: OutputFormat) -> String
  ```

  Delegate to `report::format_benchmark_report` (task 2.3).

  **Acceptance**: Compiles; no implementation needed yet (can be a stub returning `String::new()`).

---

- [ ] 2.3 Create `packages/core/src/benchmark/report.rs`

  **File**: `packages/core/src/benchmark/report.rs`

  Implement `pub fn format_benchmark_report(result: &BenchmarkResult, fmt: OutputFormat) -> String`:

  **JSON branch** — use `serde_json::json!` macro to build the object with schema defined in
  `design.md § Data Models > BenchmarkFileOutput`. Include:
  - `schema_version: "2"`
  - `timestamp`: `chrono::Utc::now().to_rfc3339()`
  - All `Option` fields serialised as their inner value when `Some`, omitted when `None`
  - `layer_load.samples`: full array (or empty array if `None`)
  - Final serialisation via `serde_json::to_string_pretty` for readability

  **Text branch** — produce the table format from `design.md § Component 5`:
  ```
  GwenLand Benchmark — {date}
  ════════════════════════════════
  Binary Size:   {mb:.2} MB  {✓|✗} (< 15 MB)
  Cold Start:    {ms:.1} ms  {✓|✗} (< 15 ms)
  Inference:     {tok/s:.1} tok/s  [{backend}, {model}]
  Layer Load:    mean {µs} µs/layer  |  peak RSS {mb:.0} MB  |  est. full {est_mb:.0} MB
  Memory Floor:  {mb:.1} MB RSS
  Total:         {secs:.2} s
  ```
  Use `────` separators. For any `None` field print `  (not measured)`.

  Add `pub mod report;` to `benchmark/mod.rs`.

  **Acceptance**: `cargo build -p gwenland-core` compiles.

---

- [ ] 2.4 Write tests for `format_benchmark_report`

  **File**: `packages/core/src/benchmark/report.rs` (`#[cfg(test)]`)

  Tests:
  - `test_format_json_is_valid_json` — build a `BenchmarkResult` with all `None` fields,
    call `format_benchmark_report(..., Json)`, parse via `serde_json::from_str::<serde_json::Value>`.
    Assert `Ok`.
  - `test_format_json_schema_version` — parsed JSON object contains `"schema_version": "2"`.
  - `test_format_text_contains_header` — text output contains `"GwenLand Benchmark"`.
  - `test_format_text_layer_load_none` — when `layer_load = None`, text contains
    `"(not measured)"`.
  - `test_format_text_inference_backend` — when `inference.backend = "mistralrs"`, text
    contains `"mistralrs"`.

  **Acceptance**: `cargo test -p gwenland-core report` passes (5 tests).

---

- [ ] 2.5 Create `packages/core/src/benchmark/report.rs` — `write_benchmark_file`

  **File**: `packages/core/src/benchmark/report.rs`

  Add:
  ```rust
  /// Write benchmark result as JSON to `path` and print text summary to stdout.
  pub fn write_benchmark_file(
      result: &BenchmarkResult,
      path: &std::path::Path,
  ) -> anyhow::Result<()>
  ```

  Implementation:
  - `format_benchmark_report(result, OutputFormat::Json)` → write to `path` via `std::fs::write`
  - `format_benchmark_report(result, OutputFormat::Text)` → `println!`
  - Return `Ok(())`

  **Acceptance**: Compiles; used by Wave 4 integration tests.

---

### Wave 3 — Mistral.rs Inference Benchmark Path

- [ ] 3.1 Add `run_mistralrs_bench` to `packages/core/src/benchmark/inference.rs`

  **File**: `packages/core/src/benchmark/inference.rs`

  Add a new function, feature-gated:
  ```rust
  #[cfg(feature = "mistralrs-backend")]
  pub fn run_mistralrs_bench(model_path: &std::path::Path) -> Option<InferenceResult>
  ```

  Implementation:
  - `MistralRsBackend::new()` → `load_model(model_path)` — if error, return `None` with
    `eprintln!` (not panic)
  - `InferParams { max_tokens: 128, temperature: 0.0, ..Default::default() }`
  - `Instant::now()` → `backend.infer(BENCHMARK_PROMPT, params)` → record `elapsed_secs`
  - Estimate `total_tokens = result_text.len() / 4` (same 4-chars/token heuristic)
  - `tokens_per_sec = total_tokens as f64 / elapsed_secs`
  - Return `Some(InferenceResult { tokens_per_sec, total_tokens, elapsed_secs, backend: "mistralrs".to_string(), model_file: Some(basename) })`
  - `backend.unload()` after measurement

  `BENCHMARK_PROMPT` is the same constant already in the file.

  **Acceptance**: `cargo build -p gwenland-core --features mistralrs-backend` compiles; feature-absent build also compiles.

---

- [ ] 3.2 Write smoke test for `run_mistralrs_bench` (no-model path)

  **File**: `packages/core/src/benchmark/inference.rs` (`#[cfg(test)]`)

  Test:
  - `test_run_mistralrs_bench_missing_model` (feature-gated `#[cfg(feature = "mistralrs-backend")]`):
    call `run_mistralrs_bench(Path::new("nonexistent.gguf"))`, assert result is `None`.
    (The backend returns an error on bad path; the function returns None.)

  **Acceptance**: `cargo test -p gwenland-core --features mistralrs-backend inference` passes.

---

- [ ] 3.3 Update `BenchmarkFilter` and `run_benchmarks` to call `run_mistralrs_bench`

  **File**: `packages/core/src/benchmark/mod.rs`

  In `run_benchmarks`, update the inference suite block to attempt `run_mistralrs_bench`
  when a `model_path: Option<&Path>` parameter is provided and the feature is active:

  ```rust
  pub fn run_benchmarks(
      filter: BenchmarkFilter,
      progress: Option<&ProgressCallback>,
      // NEW optional parameter — None preserves existing proxy-only behaviour
      model_path: Option<&std::path::Path>,
  ) -> BenchmarkResult
  ```

  Inference suite logic (updated):
  ```rust
  let inference_result = if filter.run_inference() {
      // Try in-process mistralrs first (feature-gated), then fall back to proxy
      #[cfg(feature = "mistralrs-backend")]
      if let Some(mp) = model_path {
          let r = inference::run_mistralrs_bench(mp);
          if r.is_some() { r } else { inference::run_inference_bench() }
      } else {
          inference::run_inference_bench()
      }
      #[cfg(not(feature = "mistralrs-backend"))]
      inference::run_inference_bench()
  } else {
      None
  };
  ```

  Update all existing call sites of `run_benchmarks` (search for `run_benchmarks(` in the
  codebase) to pass `model_path: None` as the third argument — preserving existing behaviour.

  **Acceptance**: `cargo build -p gwenland-core` compiles; `cargo build --features mistralrs-backend` compiles; existing callers compile with `None`.

---

### Wave 4 — E2E Integration Tests

- [ ] 4.1 Create `packages/core/tests/gwen214_e2e.rs` with helpers

  **File**: `packages/core/tests/gwen214_e2e.rs`

  Create the test file with:
  1. Env-var helper and `require_model!` macro (see `design.md § Component 1`)
  2. `assert_binary_size(exe_path: &Path, max_bytes: u64)` helper
  3. `assert_cold_start_ms(binary: &Path, max_ms: f64)` helper (warm-up + 5 samples)
  4. `assert_no_oom(baseline_mb: f64, max_delta_mb: f64)` helper using
     `gwenland_core::benchmark::memory::sample_memory()`

  No test functions yet — just the helpers and imports. File must compile cleanly.

  **Required imports**:
  ```rust
  use gwenland_core::benchmark::memory;
  ```

  **Acceptance**: `cargo build -p gwenland-core --features test-utils` produces no compile errors
  for the test file.

---

- [ ] 4.2 Add `test_binary_size_under_15mb` to `gwen214_e2e.rs`

  **File**: `packages/core/tests/gwen214_e2e.rs`

  ```rust
  #[test]
  fn test_binary_size_under_15mb() {
      let exe = find_release_binary();
      match exe {
          None => eprintln!("release binary not found — skipping binary size check"),
          Some(p) => assert_binary_size(&p, 15 * 1024 * 1024),
      }
  }
  ```

  `find_release_binary` implementation:
  - Start from `std::env::current_exe().unwrap()`
  - Walk up ancestors until a component named `"target"` is found
  - Append `release/gwenland` (Linux/macOS) or `release/gwenland.exe` (Windows)
  - Return `Some(path)` if it exists, else `None`

  **Acceptance**: Test runs; passes when binary is ≤ 15 MB; skips when binary not found.

---

- [ ] 4.3 Add `test_cold_start_under_15ms` to `gwen214_e2e.rs`

  **File**: `packages/core/tests/gwen214_e2e.rs`

  ```rust
  #[test]
  fn test_cold_start_under_15ms() {
      let binary = match find_release_binary() {
          Some(p) => p,
          None => {
              eprintln!("release binary not found — skipping cold-start test");
              return;
          }
      };
      assert_cold_start_ms(&binary, 15.0);
  }
  ```

  Uses the `assert_cold_start_ms` helper from task 4.1.
  Measures `gwen --help` (not `--version`) as per the GWEN-214 requirement.

  **Acceptance**: Test passes when cold-start ≤ 15 ms; skips when binary unavailable.

---

- [ ] 4.4 Add `test_e2e_chat_inference` to `gwen214_e2e.rs`

  **File**: `packages/core/tests/gwen214_e2e.rs`

  Feature-gated `#[cfg(feature = "mistralrs-backend")]`:
  ```rust
  #[cfg(feature = "mistralrs-backend")]
  #[test]
  fn test_e2e_chat_inference() {
      let model_path = require_model!();
      // 1. Baseline RSS
      let baseline_mb = memory::sample_memory().baseline_mb;
      // 2. Load model via MistralRsBackend
      // 3. Infer with max_tokens=64, temperature=0.0
      // 4. Assert tokens_generated > 0
      // 5. Assert no OOM (< 3000 MB RSS delta)
      // 6. Unload
  }
  ```

  Use `gwenland_core::engine::inference::{MistralRsBackend, InferParams, InferenceBackend}`.
  The test skips (returns early) when `GWEN_TEST_MODEL_PATH` is not set.

  **Acceptance**: Test compiles with `--features mistralrs-backend`; skips without env var;
  passes when env var points to valid Qwen3-1.7B Q8_0 GGUF.

---

- [ ] 4.5 Add `test_e2e_lora_training` to `gwen214_e2e.rs`

  **File**: `packages/core/tests/gwen214_e2e.rs`

  ```rust
  #[test]
  #[cfg(feature = "test-utils")]
  fn test_e2e_lora_training() {
      let model_path = require_model!();
      // 1. Build NewTrainConfig: 1 epoch, grad_accum=1, rank=4, alpha=8.0
      // 2. Build VarMap + synthetic 1-batch of 4 token IDs
      // 3. LayeredTrainingLoop::new(config, model_path, batches, varmap, None)
      // 4. .run() → TrainResult
      // 5. Assert result.final_loss.is_finite()
      // 6. Assert result.total_steps >= 1
      // 7. LoraExporter::export_safetensors(varmap, tempdir/adapter.safetensors)
      // 8. Assert adapter file exists and size > 0
      // 9. Assert RSS < 6000 MB
  }
  ```

  Use:
  - `gwenland_core::train::{LayeredTrainingLoop, config::NewTrainConfig}`
  - `gwenland_core::train::lora_bridge::LoraExporter`
  - `candle_core::{Device, Tensor}`
  - `candle_nn::VarMap`
  - `tempfile::tempdir`

  **Acceptance**: Compiles; skips without env var; passes when model is available and OOM-safe.

---

- [ ] 4.6 Add `[[test]]` entry to `packages/core/Cargo.toml`

  **File**: `packages/core/Cargo.toml`

  Add after existing `[[test]]` entry:
  ```toml
  [[test]]
  name = "gwen214_e2e"
  path = "tests/gwen214_e2e.rs"
  required-features = ["test-utils"]
  ```

  **Acceptance**: `cargo test --test gwen214_e2e --features test-utils` resolves the test target.

---

### Wave 5 — Integration, Regression, and Benchmark File Output

- [ ] 5.1 Add `test_benchmark_json_round_trip` to `gwen214_e2e.rs`

  **File**: `packages/core/tests/gwen214_e2e.rs`

  ```rust
  #[test]
  fn test_benchmark_json_round_trip() {
      use gwenland_core::benchmark::{BenchmarkResult, OutputFormat, format_benchmark_report};
      
      let result = BenchmarkResult {
          cold_start: None, inference: None, convert: None,
          memory: None, layer_load: None, total_elapsed_secs: 0.0,
      };
      let json_str = format_benchmark_report(&result, OutputFormat::Json);
      let parsed: serde_json::Value = serde_json::from_str(&json_str)
          .expect("benchmark JSON output must be valid JSON");
      assert_eq!(parsed["schema_version"], "2");
  }
  ```

  **Acceptance**: Test passes without any env var or model file.

---

- [ ] 5.2 Run full test suite and confirm zero new failures

  **Action**: Run `cargo test -p gwenland-core --lib` and verify:
  - All pre-GWEN-214 tests still pass (228 tests + whatever was passing before)
  - No new failures introduced
  - The 6 pre-existing failures from `lora_merger` and `inference::selector` remain as
    the only failures (unchanged from GWEN-216 baseline)

  **Acceptance**: `cargo test -p gwenland-core --lib` exit code 0 with same or higher pass count.

---

- [ ] 5.3 Run integration test suite with `test-utils` feature

  **Action**:
  ```powershell
  cargo test --test gwen214_e2e --features test-utils
  ```

  Verify:
  - `test_binary_size_under_15mb` — passes (binary 11.11 MB) or skips (release not built)
  - `test_cold_start_under_15ms` — passes or skips
  - `test_benchmark_json_round_trip` — passes (no model needed)
  - `test_e2e_chat_inference` — skips (GWEN_TEST_MODEL_PATH not set in CI default)
  - `test_e2e_lora_training` — skips (GWEN_TEST_MODEL_PATH not set in CI default)

  **Acceptance**: `cargo test --test gwen214_e2e --features test-utils` exits 0.

---

- [ ] 5.4 Run release build and verify binary size

  **Action**:
  ```powershell
  cargo build --release -p gwenland-core
  ```

  Then in the test output or manually:
  ```powershell
  (Get-Item "target/release/gwenland.exe").Length / 1MB
  ```

  **Acceptance**: Binary ≤ 15 MB. Current baseline is 11.11 MB.

---

- [ ] 5.5 Verify `bench_layer_loader.exe` still compiles and reports correctly

  **Action**:
  ```powershell
  cargo build --release --bin bench_layer_loader -p gwenland-core
  ```

  Verify:
  - Binary builds without errors
  - `bench_layer_loader.exe --help` or `bench_layer_loader.exe --format json` prints valid output
    (can use a dummy/test GGUF file)

  **Acceptance**: Binary builds; existing 5 smoke tests still pass:
  `cargo test --bin bench_layer_loader -p gwenland-core`

---

- [ ] 5.6 Confirm `BenchmarkFilter::run_benchmarks` backward compatibility

  **File**: Search codebase for all callers of `run_benchmarks(` and confirm each was updated
  in task 3.3 to pass `model_path: None` as the third argument.

  Callers to check (likely in `diagnostics/benchmark.rs` and any TUI layer):
  - `diagnostics/benchmark.rs` (if it calls `run_benchmarks`)
  - `main.rs` or any CLI handler

  **Acceptance**: `cargo build -p gwenland-core` and any dependent packages compile cleanly.

---

## Task Dependency Graph

```json
{
  "waves": [
    {
      "wave": 1,
      "name": "Layer-Load Benchmark Module",
      "tasks": ["1.1", "1.2", "1.3"]
    },
    {
      "wave": 2,
      "name": "Benchmark Report Formatter",
      "tasks": ["2.1", "2.2", "2.3", "2.4", "2.5"],
      "dependsOn": ["1.3"]
    },
    {
      "wave": 3,
      "name": "Mistral.rs Inference Benchmark Path",
      "tasks": ["3.1", "3.2", "3.3"],
      "dependsOn": ["2.1"]
    },
    {
      "wave": 4,
      "name": "E2E Integration Tests",
      "tasks": ["4.1", "4.2", "4.3", "4.4", "4.5", "4.6"],
      "dependsOn": ["2.5", "3.3"]
    },
    {
      "wave": 5,
      "name": "Integration, Regression, and Verification",
      "tasks": ["5.1", "5.2", "5.3", "5.4", "5.5", "5.6"],
      "dependsOn": ["4.6"]
    }
  ],
  "taskDependencies": {
    "1.2": ["1.1"],
    "1.3": ["1.1"],
    "2.1": ["1.3"],
    "2.2": ["2.1"],
    "2.3": ["2.2"],
    "2.4": ["2.3"],
    "2.5": ["2.3"],
    "3.1": ["2.1"],
    "3.2": ["3.1"],
    "3.3": ["3.1", "2.2"],
    "4.1": ["2.5", "3.3"],
    "4.2": ["4.1"],
    "4.3": ["4.1"],
    "4.4": ["4.1", "3.1"],
    "4.5": ["4.1"],
    "4.6": ["4.4", "4.5"],
    "5.1": ["4.6"],
    "5.2": ["4.6"],
    "5.3": ["4.6"],
    "5.4": ["5.2"],
    "5.5": ["5.4"],
    "5.6": ["3.3"]
  }
}
```

---

## Notes

### Zero New Dependencies
All types used are already in `Cargo.toml`:
- `sysinfo = "0.30"` — RAM sampling in `layer_load_bench.rs`
- `serde + serde_json = "1"` — JSON output in `report.rs`
- `chrono = "0.4"` — timestamp in benchmark JSON
- `anyhow = "1"` — error propagation throughout
- `tempfile = "3"` — temporary directories in E2E tests
- `candle-core = "0.9"`, `candle-nn = "0.9"` — already unconditional deps

### Windows Compatibility
- All `MADV_DONTNEED` calls are `#[cfg(unix)]` (unchanged from GWEN-216)
- `sysinfo` works on Windows for RSS sampling
- `current_exe()` returns `.exe` path on Windows automatically
- `find_release_binary` appends `.exe` on Windows via `cfg!(windows)`

### Feature-Gated Tests
- `test_e2e_chat_inference` is `#[cfg(feature = "mistralrs-backend")]`
- `test_e2e_lora_training` is `#[cfg(feature = "test-utils")]`
- The `[[test]]` entry requires `test-utils` so the file compiles without it being
  the active feature — when `test-utils` is absent, cargo simply skips the entire test target

### Skip vs Fail Pattern
Tests that depend on `GWEN_TEST_MODEL_PATH` skip (return early with `eprintln!`) rather than
fail when the env var is absent. This ensures CI without the model file still passes. CI with
the model file will exercise the full E2E assertions.

### Existing Benchmark Files
The existing benchmark output files in `benchmark/Gwen-Benchmark-2026-06-03_22-00.md` follow
a text format. GWEN-214 adds JSON alongside the existing text format — both are valid outputs
of `gwen benchmark`. The `schema_version: "2"` field distinguishes v2 (with `layer_load`)
from v1 files.

### PBT Tasks
Tasks 1.2 and 2.4 include deterministic unit tests. There are no new quickcheck property tests
in GWEN-214 (the layer-load inputs are path-dependent; the formatter inputs are
serde-deterministic). The existing PBT tasks from GWEN-216 (layer_loader Properties 1–7)
remain green and cover the underlying layer-loading correctness.

### Pre-existing Failures (unchanged)
The 6 pre-existing test failures from GWEN-216 baseline are not touched by GWEN-214:
- `engine::inference::selector::{empty_stop_sequences_ok, relative_gguf_ok, tilde_expand}`
- `train::lora_merger::{test_merge_identity, test_merge_nan_detection, test_merge_shape_mismatch}`

GWEN-214 must not introduce any additional failures.
