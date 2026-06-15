# Requirements Document

## Introduction

GWEN-216 enables LoRA fine-tuning of Qwen3-1.7B Q8_0 (1.92 GiB) on 8 GB RAM hardware with no discrete GPU, by loading only the active transformer layer into RAM, training it, and unloading it before moving to the next. Peak RAM per layer target is ~50–80 MB Q8_0 data (total system peak including dequant buffers: ~420 MB) versus the 1.92 GiB full-model baseline.

The feature builds directly on GWEN-209/210 (`MmapLoader`, OOM-safe mmap), the GGUF parser (`GgufFile`/`TensorInfo`), and GWEN-213 (candle LoRA training loop, `LoraLayer`, `VarMap`). No new Cargo dependencies are introduced.

## Requirements

### Requirement 1: LayerSlice and LayerIndex

**User Story:** As a developer implementing GWEN-216, I need a data structure that maps transformer layer numbers to their tensor byte ranges in the GGUF file, so that I can slice the mmap without reading the full file.

#### Acceptance Criteria

1.1 The system shall provide a `LayerSlice` struct with fields: `layer_idx: usize`, `tensor_name: String`, `byte_offset: u64`, `byte_len: usize`.

1.2 The system shall provide a `LayerIndex::scan(file: &GgufFile) -> LayerIndex` function that scans all `TensorInfo` entries in the `GgufFile` and includes only those whose names match the pattern `model.layers.{N}.*`, where `N` is a non-negative integer.

1.3 `LayerIndex::scan` shall exclude embedding tensors, final norm tensors, and LM-head tensors — any tensor whose name does not start with `model.layers.`.

1.4 The resulting `LayerIndex.slices` shall be sorted by `(layer_idx ASC, tensor_name ASC)` to guarantee deterministic iteration order.

1.5 `LayerIndex.num_layers` shall equal `max(layer_idx) + 1` across all matched tensors, or 0 if no layer tensors are found.

1.6 For every `LayerSlice s` in the index, `s.byte_offset + s.byte_len` shall not exceed the GGUF file's total size. Slices violating this constraint shall be rejected with an error during scan.

1.7 The initial `Vec<LayerSlice>` construction is the only heap allocation in `LayerIndex::scan`. No per-tensor heap allocation occurs during the scan loop body.

---

### Requirement 2: LayerLoader

**User Story:** As the training loop, I need a loader that opens a GGUF file in lazy mode and returns zero-copy byte slices for individual transformer layers, so that only the active layer occupies physical RAM at any time.

#### Acceptance Criteria

2.1 The system shall provide `LayerLoader::open(path: &Path) -> Result<Self>` that opens the GGUF file using `MmapLoader::open_with_mode(path, LoadMode::Lazy)` — enforcing lazy mode unconditionally so the full model is never prefetched into RAM.

2.2 `LayerLoader::num_layers() -> usize` shall return `self.index.num_layers`.

2.3 `LayerLoader::load_layer<'a>(&'a self, n: usize) -> Result<LoadedLayer<'a>>` shall return byte slices that are subslices of `self.mmap.as_bytes()` — no heap copy of tensor data.

2.4 `load_layer` shall return `Err` if `n >= self.num_layers()` with the message `"layer {n} out of range (model has {num_layers} layers)"`.

2.5 The `LoadedLayer.mmap_range` produced by `load_layer(n)` shall span exactly the union of all tensor byte ranges for layer `n`: `start = min(byte_offset)`, `end = max(byte_offset + byte_len)`.

2.6 `LayerLoader` shall not introduce any new `unsafe` blocks. It shall reuse `MmapLoader::open_with_mode` (which contains the existing `unsafe { MmapOptions::new().map(&file) }` block in `loader.rs`).

2.7 `LayerLoader` shall compile and run on Windows. All madvise-specific code shall be gated with `#[cfg(unix)]`.

---

### Requirement 3: LoadedLayer

**User Story:** As the training loop, I need a loaded layer handle that automatically reclaims physical memory pages when I am done with a layer, so that RAM usage stays bounded throughout training.

#### Acceptance Criteria

3.1 `LoadedLayer` shall expose `slices: Vec<(&'mmap str, &'mmap [u8])>` — pairs of `(tensor_name, raw_q8_0_bytes)` pointing directly into the mmap.

3.2 The `'mmap` lifetime shall tie all byte slices in `LoadedLayer` to the originating `LayerLoader`, preventing use-after-free at compile time.

3.3 The `Drop` implementation for `LoadedLayer` shall call `mmap_data.advise_range(Advice::DontNeed, mmap_range.start, mmap_range.len())` on Unix targets. The return value shall be discarded (`let _ = ...`) — the hint is advisory.

3.4 On non-Unix targets (including Windows), `LoadedLayer::drop` shall be a no-op, mirroring the existing `apply_madvise` stub pattern in `loader.rs`.

3.5 `LoadedLayer` shall provide `pub fn unload(self)` that is semantically identical to `drop(self)`, provided for call-site clarity.

---

### Requirement 4: LayeredTrainingLoop

**User Story:** As the GwenLand CLI, I need a LoRA training loop that trains Qwen3-1.7B on 8 GB RAM by sequentially loading, training, and unloading each transformer layer, so that I can fine-tune large models without OOM errors.

#### Acceptance Criteria

4.1 At no point during `LayeredTrainingLoop::run()` shall the full model's Q8_0 data be resident in physical RAM simultaneously. Only the active layer's byte range and one in-flight dequant buffer shall be live at any given time.

4.2 `run()` shall iterate layers in order 0..num_layers for each (epoch, batch) pair: load layer N → forward → backward → accumulate gradients → unload layer N → proceed to layer N+1.

4.3 The gradient accumulation semantics of `LayeredTrainingLoop::run()` shall be identical to `TrainingLoop::run()`: the same `step_accumulated` helper is called unchanged, with the same `grad_accum` cadence.

4.4 `varmap` (containing `lora_a` and `lora_b` Vars) shall persist for the entire training run across all layer passes. It shall not be re-created per layer.

4.5 A `LoraLayer` instance shall be constructed for each layer pass using the dequantised base weights from the current `LoadedLayer`. The `LoraLayer` shall be dropped at the end of the layer pass (before `LoadedLayer::unload()`).

4.6 For each layer pass, lora_a/lora_b buffers shall use a stack-allocated `[f32; 512]` array when `rank × d_model ≤ 512`, and fall back to `Vec<f32>` otherwise.

4.7 `LayeredTrainingLoop::run()` shall emit the same JSON progress events to stdout as `TrainingLoop::run()`: `{"event":"step","epoch":N,"step":S,"loss":L,"elapsed_secs":T}` after every `(layer, batch)` pair, and `{"event":"done",...}` on completion.

4.8 A safetensors checkpoint shall be written every 500 optimiser steps using `varmap.save(path)`, identical to `TrainingLoop::save_checkpoint`.

4.9 `LayeredTrainingLoop` shall accept `tx: Option<Sender<String>>` and forward all progress JSON to the sender when `Some`, matching `TrainingLoop`'s TUI pipe semantics.

4.10 `run()` shall return `TrainResult { final_loss, total_steps, elapsed }` with the same semantics as `TrainingLoop::run()`.

---

### Requirement 5: Compatibility with Existing Types

**User Story:** As a GwenLand maintainer, I need GWEN-216 to integrate with existing types without breaking any of the 178+ passing tests or changing the GWEN-213 LoRA adapter key schema.

#### Acceptance Criteria

5.1 `LayerIndex::scan` shall accept `&GgufFile` directly and use `TensorInfo.data_offset`, `TensorInfo.data_size`, and `TensorInfo.name` as documented in `gguf_parser.rs`.

5.2 `LayeredTrainingLoop` shall construct `LoraLayer` instances using the same `LoraLayer::new(d_in, d_out, base_weight, config, vb)` signature defined in `lora.rs`. No changes to `LoraLayer` are required.

5.3 `LayeredTrainingLoop` shall initialise `AdamW` from `varmap.all_vars()` with the same `ParamsAdamW` defaults as `TrainingLoop::new()`.

5.4 The `VarMap` keys for lora_a and lora_b shall remain compatible with the format established in GWEN-213 (`lora_a`, `lora_b` via `VarBuilder::get_with_hints`).

5.5 The `step_accumulated(&[GradStore])` helper shall be extracted into a shared `pub(crate)` free function so both `TrainingLoop` and `LayeredTrainingLoop` can call it without code duplication.

---

### Requirement 6: Hard Constraints

**User Story:** As a GwenLand contributor, I need all hard constraints from the ticket (no full model in RAM, no new unsafe, no new deps, Windows compat, no regressions) to be formally specified, so that they are enforced by CI and code review.

#### Acceptance Criteria

6.1 Training shall never require the full Qwen3-1.7B model (1.92 GiB) to be resident in physical RAM simultaneously.

6.2 No `unsafe` blocks shall be introduced beyond the existing `unsafe { MmapOptions::new().map(&file) }` in `loader.rs`.

6.3 No new entries shall be added to `[dependencies]` or `[dev-dependencies]` in `packages/core/Cargo.toml`.

6.4 All new code shall compile on Windows. `#[cfg(unix)]` guards shall be used for any Unix-specific madvise calls.

6.5 The 178+ existing tests (including 13 LoRA training tests) shall continue to pass after GWEN-216 is merged.

---

### Requirement 7: Benchmark Binary

**User Story:** As a GwenLand developer, I need a benchmark binary that measures per-layer RAM peak versus the full-load baseline, so that I can verify the ~50–80 MB per-layer target and demonstrate the RAM savings to stakeholders.

#### Acceptance Criteria

7.1 A new binary `bench_layer_loader` shall be added at `packages/core/src/bin/bench_layer_loader.rs` and registered in `Cargo.toml` under `[[bin]]`.

7.2 The benchmark shall measure and report peak RSS (Resident Set Size) before and after `load_layer(N)` and after `LoadedLayer::unload()`.

7.3 The benchmark shall measure and report wall-clock time to execute `load_layer(N)` and throughput in MiB/s.

7.4 The benchmark shall support a `--compare-full` flag that runs `gguf_loader::load_and_dequant` baseline for comparison.

7.5 The benchmark shall support a `--layer N` flag to benchmark a specific layer, and iterate all layers by default.

7.6 The benchmark shall support `--format text` (default, human-readable) and `--format json` (machine-readable) output.

---

### Requirement 8: Module Wiring

**User Story:** As a developer using the `gwenland-core` crate, I need the new modules and types to be accessible via the standard `use gwenland_core::train::*` path, so that integration is straightforward.

#### Acceptance Criteria

8.1 `packages/core/src/train/mod.rs` shall be updated to include `pub mod layer_loader;` and `pub mod layered_training_loop;`.

8.2 `packages/core/Cargo.toml` shall include a `[[bin]]` entry for `bench_layer_loader` pointing to `src/bin/bench_layer_loader.rs`.

## Glossary

- **LayerSlice**: Lightweight descriptor for a single tensor belonging to one transformer layer; contains byte offset and length within the GGUF mmap.
- **LayerIndex**: Pre-built map from layer index N to its set of `LayerSlice` descriptors; constructed once at startup from `GgufFile.tensors`.
- **LayerLoader**: Holds the open `MmapLoader` and `LayerIndex`; serves zero-copy `LoadedLayer` views.
- **LoadedLayer**: A set of lifetime-tied `&[u8]` slices into the mmap for one transformer layer; issues `MADV_DONTNEED` on drop.
- **LayeredTrainingLoop**: Layer-sequential LoRA training loop; wraps the same `AdamW`/`VarMap` infrastructure as `TrainingLoop` but loads one layer at a time.
- **MADV_DONTNEED**: Linux/macOS madvise hint requesting the OS reclaim physical pages for a virtual address range; advisory (OS may ignore).
- **Q8_0**: GGUF 8-bit quantisation format — blocks of 32 elements, each with a 1×f16 scale value. Raw bytes → f32 via `dequant::dequantize`.
- **LORA_STACK_THRESHOLD**: Compile-time constant (512 elements) controlling when lora_a/lora_b buffers are stack- vs heap-allocated.
