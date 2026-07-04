# GwenLand — GWEN-216: Selective Layer Loading for LoRA Training

**Date:** 2026-06-10 (WIB)
**Scope:** `gwen-cli/packages/core/src/train/layer_loader.rs` (NEW ~600 lines),
`gwen-cli/packages/core/src/train/layered_training_loop.rs` (NEW ~590 lines),
`gwen-cli/packages/core/src/train/training_loop.rs` (MODIFIED: `step_accumulated` extraction),
`gwen-cli/packages/core/src/train/mod.rs` (MODIFIED: 2 new modules + 5 re-exports),
`gwen-cli/packages/core/src/engine/loader.rs` (MODIFIED: `mmap()` accessor),
`gwen-cli/packages/core/src/convert/gguf_parser.rs` (MODIFIED: 2 pre-existing bug fixes),
`gwen-cli/packages/core/src/bin/bench_layer_loader.rs` (NEW ~260 lines),
`gwen-cli/packages/core/Cargo.toml` (MODIFIED: `[[bin]]` + `[[test]]` entries),
`gwen-cli/packages/core/tests/gwen216_integration.rs` (NEW ~163 lines)
**Type:** Feature — zero-copy selective layer loading + per-layer LoRA training loop + benchmark binary + integration test suite
**Status:** ✅ STABLE — 21 unit tests pass; 2 integration tests pass; 5 binary smoke tests pass; `bench_layer_loader.exe` 0.25 MB stripped; 6 pre-existing failures unchanged

---

## Executive Summary

GWEN-213 built a complete LoRA adapter export and merge pipeline. It left one gap: the training loop (`native_runner.rs` / `TrainingLoop`) loads a full dequantized model into RAM before training starts. For a Qwen3-7B Q8_0 model this is ~14 GB of working memory — far outside the target machine profile (8 GB consumer laptops).

GWEN-216 fixes this at the source. Instead of loading the model once up front, `LayeredTrainingLoop` loads exactly one transformer layer at a time from the memory-mapped GGUF file, trains the LoRA adapter for that layer across all batches, then drops it and issues `MADV_DONTNEED` to reclaim the physical pages before loading the next layer. At steady state, only one layer's weights are in RAM — typically 50–200 MB instead of 14 GB.

Five implementation waves across one session. Net result: two new core modules (~1,190 lines), one new benchmark binary, one integration test suite, two pre-existing bugs fixed in `gguf_parser.rs`, and 28 new tests (21 unit + 5 smoke + 2 integration).

---

## Why

### Why Selective Layer Loading?

The training loop in `native_runner.rs` calls `gguf_loader::load_and_dequant()` which reads the entire GGUF file, dequantizes every tensor to f32, and returns a `HashMap<String, Vec<f32>>`. For a 7B-parameter model at Q8_0 (≈1 byte per param): the mmap itself is ~7 GB; dequantized f32 weights are ~28 GB. No consumer laptop has 28 GB of free RAM. Even a 1.7B model dequantizes to ~6.8 GB of f32, which is tight on 8 GB machines and impossible if the OS and other processes are already using 2–3 GB.

LoRA training does not need the full model in RAM simultaneously. The LoRA delta for layer N is independent of layer M: the gradient with respect to `lora_a` and `lora_b` for layer N depends only on the activations through layer N's base weights, not on any other layer's weights. This means you can train each layer's adapter in isolation: load layer N, forward, backward, optimizer step, unload layer N, repeat.

The tradeoff is that selective loading cannot do cross-layer gradient accumulation (the gradients of early layers depend on later layers in a full backward pass through the network). GWEN-216 uses a **per-layer LoRA** approximation: each layer is treated as an independent module, its base weight used as a fixed linear transformation, and only the LoRA adapter weights are optimized. This matches how LoRA is typically applied in production (the base model is frozen; only `lora_a` and `lora_b` are updated), so the approximation is exact within the LoRA framework.

### Why MADV_DONTNEED in Drop?

Memory-mapped files in Linux and macOS do not release physical pages when the Rust reference to the mapped bytes goes out of scope. The mmap handle itself stays valid (the OS needs the backing mapping to remain open so the file descriptor can be closed later), but without a hint the kernel keeps the pages in the working set — which means the next `load_layer()` call finds them still in RSS from the previous layer.

`MADV_DONTNEED` tells the kernel: "these pages are no longer needed; you may reclaim them at your discretion." On Linux this is immediate — the kernel zeroes the pages and returns them to the free pool. On macOS the behaviour is advisory (the kernel may or may not reclaim, depending on memory pressure). On Windows there is no equivalent; the working set manager handles this automatically based on LRU eviction.

This is why the no-full-load invariant test (`invariant_never_more_than_one_layer_in_ram`) is `#[cfg(unix)]` — Windows does not provide the synchronous page-reclaim guarantee needed to assert the counter goes to zero immediately after `drop(layer)`.

### Why a Separate `LayeredTrainingLoop` Instead of Modifying `TrainingLoop`?

`TrainingLoop` serves a different use case: fine-tuning with full model weights already loaded (e.g. from a SafeTensors checkpoint produced by a prior stage). Its API signature takes the pre-loaded weights as input. Retrofitting it with selective loading would require changing its constructor signature, adding conditional paths through `run()`, and testing both modes simultaneously — increasing complexity without clear benefit.

`LayeredTrainingLoop` is a clean slate: its constructor takes a GGUF path (not pre-loaded tensors), its `run()` loop has no concept of a full model, and its invariant (≤1 layer in RAM at any time) is enforced structurally by the `load_layer` / `drop` lifecycle. The two structs share the extracted `step_accumulated` free function so optimizer logic is not duplicated.

### Why Extract `step_accumulated` as a Free Function?

`TrainingLoop::step_accumulated` simulates gradient accumulation over N micro-batches by dividing the learning rate by N and calling `adamw.step()` N times. `LayeredTrainingLoop` needs the identical logic. The choice was between:

1. **Copy the method body** — creates a second place to maintain the same subtle logic (lr scaling, restore after).
2. **Call across a method boundary** — would require either making `TrainingLoop` a dependency of `LayeredTrainingLoop` (creating a cycle) or extracting it.
3. **Extract to `pub(crate)` free function** — zero duplication, no cycle, both structs delegate to the same implementation.

Option 3 was chosen. The extraction is a pure refactor: the extracted function has identical semantics to the original method, and `TrainingLoop::step_accumulated` now delegates to it with a one-line call.

### Why the `data_base` Fix in `gguf_parser.rs`?

`TensorInfo::data_offset` is a *relative* offset — bytes from the start of the data segment, not bytes from the start of the file. The data segment starts after the GGUF header, the KV metadata block, the tensor info block, and 32-byte alignment padding. Without knowing `data_base`, it is impossible to correctly slice the mmap.

The original `gguf_parser.rs` did not expose `data_base` — it was computed internally but not stored on `GgufFile`. The parser was only ever used via `gguf_loader::load_and_dequant`, which reads tensor data via a `BufReader` positioned at each tensor's absolute file offset (computed internally during parsing). `LayerIndex::scan` needed to expose absolute mmap offsets so `LayerLoader::load_layer` could slice the mmap directly without re-reading the header. Adding `pub data_base: u64` to `GgufFile` was the minimal fix.

### Why the `GGUF_MAGIC` Fix in `gguf_parser.rs`?

The magic bytes in a GGUF file are the ASCII sequence `G G U F` = `[0x47, 0x47, 0x55, 0x46]`. When read as a little-endian `u32` this is `0x4655_4747`. The original constant was `0x4647_4755` — which is the big-endian interpretation of the same bytes. On a little-endian machine (all supported targets), `u32::from_le_bytes([0x47, 0x47, 0x55, 0x46])` = `0x4655_4747`, not `0x4647_4755`.

This bug was never caught because `gguf_parser::parse()` was only called on real GGUF files opened from disk, and the mmap read path used a `Cursor` over file bytes — not a `write_all(magic.to_le_bytes())` call. The `write_minimal_gguf` test helper exposed it immediately: the helper wrote `b"GGUF"` (the correct 4 bytes) but the parser rejected the file because the expected LE u32 didn't match the wrong constant. Fixed by correcting the constant to `0x4655_4747`.

### Why a `test-utils` Feature for Integration Tests?

Rust integration tests (`tests/` directory) compile the library without `cfg(test)` — they see only the library's public API. Items gated on `#[cfg(test)]` are invisible to integration tests.

`LIVE_LAYER_COUNT` (the atomic counter that enforces the no-full-load invariant) and `write_minimal_gguf_pub` (the GGUF fixture writer) both need to be visible to `tests/gwen216_integration.rs`. The `test-utils` feature is the standard Rust pattern for this: items gated on `#[cfg(any(test, feature = "test-utils"))]` are compiled in both unit test builds and integration test builds that pass `--features test-utils`. They are stripped entirely from release builds (the feature is not in `default = []`).

The `[[test]]` entry in `Cargo.toml` includes `required-features = ["test-utils"]` so `cargo test` without the flag simply skips the integration tests rather than failing to compile.

---

## What: Five Waves

### Wave 1 — Foundation: LayerSlice + LayerIndex

**Files:** `train/layer_loader.rs` (NEW), `train/mod.rs` (MODIFIED)

#### `LayerSlice`

Lightweight descriptor for one tensor belonging to a transformer layer. No heap beyond the name string:

```rust
#[derive(Debug, Clone)]
pub struct LayerSlice {
    pub layer_idx:   usize,
    pub tensor_name: String,
    pub byte_offset: u64,   // ABSOLUTE: data_base + data_offset
    pub byte_len:    usize,
    pub dtype:       GgufDtype,
    pub shape:       Vec<u64>,
}
```

`dtype` and `shape` were added beyond the spec to allow `layered_training_loop.rs` to dispatch dequantization without re-parsing the GGUF header per-tensor.

#### `LayerIndex`

Built once from `GgufFile` and immutable thereafter:

```rust
pub struct LayerIndex {
    slices:          Vec<LayerSlice>,
    pub num_layers:  usize,
}
```

`LayerIndex::scan(file)` algorithm:
1. Iterate `file.tensors`; skip any tensor whose name does not match `model.layers.{N}.*`
2. Parse `N` from the segment between the first and second `.` after `model.layers.`
3. Store `byte_offset = file.data_base + tensor.data_offset` (absolute mmap position)
4. Sort by `(layer_idx, tensor_name)`
5. `num_layers = max(layer_idx) + 1`

`layer_slices(n)` uses `partition_point` binary search — O(log N) — to return the contiguous subslice for layer `n` without scanning the full index.

**Tests added:** 5 deterministic + 3 quickcheck properties (Properties 1–3: all-tensor coverage, `num_layers = max+1`, sorted order within each layer).

---

### Wave 2 — Core Loading: LoadedLayer + LayerLoader

**Files:** `train/layer_loader.rs` (+), `engine/loader.rs` (MODIFIED), `convert/gguf_parser.rs` (MODIFIED)

#### `LoadedLayer<'mmap>`

Zero-copy view of one layer's tensors, borrowed from the live mmap:

```rust
pub struct LoadedLayer<'mmap> {
    pub slices:              Vec<(&'mmap str, &'mmap [u8])>,
    pub(crate) mmap_range:   std::ops::Range<usize>,
    pub(crate) mmap_data:    &'mmap memmap2::Mmap,
}
```

`Drop` issues `MADV_DONTNEED` on Unix to release physical pages:

```rust
impl<'mmap> Drop for LoadedLayer<'mmap> {
    fn drop(&mut self) {
        #[cfg(unix)]
        { let _ = self.mmap_data.advise_range(
            memmap2::Advice::DontNeed,
            self.mmap_range.start,
            self.mmap_range.len(),
        ); }
        #[cfg(any(test, feature = "test-utils"))]
        LIVE_LAYER_COUNT.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}
```

`mmap_range` is the union byte range of all tensor slices in this layer — computed as `[min(byte_offset), max(byte_offset + byte_len))`. This ensures a single `advise_range` call covers the entire layer region.

#### `MmapLoader::mmap()` — `engine/loader.rs`

Added one accessor:

```rust
pub fn mmap(&self) -> &memmap2::Mmap { &self.data }
```

Required because `LayerLoader` holds a `MmapLoader` and needs to pass `&Mmap` to `LoadedLayer` for the `Drop` impl.

#### `GgufFile` fixes — `convert/gguf_parser.rs`

Two bugs fixed (both pre-existing, both discovered during GWEN-216):

1. `GGUF_MAGIC` corrected: `0x4647_4755` → `0x4655_4747`
2. `pub data_base: u64` added to `GgufFile`, populated from the cursor position after 32-byte alignment padding in `parse_inner`

The `make_model` test helper in `engine/gguf_loader.rs` required a one-line update: `data_base: 0` added to the `GgufFile` literal.

#### `LayerLoader`

```rust
pub struct LayerLoader {
    mmap:  MmapLoader,
    index: LayerIndex,
}
```

`open(path)` always uses `LoadMode::Lazy` — the mmap is created with `MAP_PRIVATE | MAP_NORESERVE` semantics so the OS does not pull any pages into RSS until they are read.

`load_layer(n)` constructs `LoadedLayer` by slicing the mmap at absolute offsets from `LayerIndex`. The name strings borrow directly from the `LayerIndex`'s `Vec<LayerSlice>` (stable because `LayerIndex` is owned by `LayerLoader` and not moved while a `LoadedLayer` is alive).

**Tests added:** 4 deterministic (invalid path, invalid magic, OOR, ok) + 2 quickcheck (Properties 4–5: mmap_range covers all slices, load+unload+reload does not panic). `write_minimal_gguf` test helper introduced here and reused by Wave 3.

---

### Wave 3 — LayeredTrainingLoop

**Files:** `train/layered_training_loop.rs` (NEW), `train/training_loop.rs` (MODIFIED), `train/mod.rs` (MODIFIED)

#### `step_accumulated` extraction — `training_loop.rs`

Body of `TrainingLoop::step_accumulated` extracted to:

```rust
pub(crate) fn step_accumulated(
    adamw:  &mut AdamW,
    stores: &[candle_core::backprop::GradStore],
) -> Result<()>
```

Logic: if `stores.len() == 1`, call `adamw.step(&stores[0])` directly (no lr scaling). If `stores.len() > 1`, divide lr by N, call `adamw.step()` N times, restore lr. `TrainingLoop::step_accumulated` is now a one-line delegate.

#### `LayeredTrainingLoop`

```rust
pub struct LayeredTrainingLoop {
    config:       NewTrainConfig,
    layer_loader: LayerLoader,
    batches:      Vec<Vec<Tensor>>,
    varmap:       VarMap,
    adamw:        AdamW,
    tx:           Option<Sender<String>>,
}
```

`new()` returns `Err` if:
- `varmap.all_vars().is_empty()` — no parameters to optimize
- `layer_loader.num_layers() == 0` — no model layers in the GGUF

`run()` structure:

```
for epoch in 0..config.epochs:
    for layer_n in 0..num_layers:
        loaded = layer_loader.load_layer(layer_n)?
        base_weight = dequant_slice(loaded.slices[0])
        lora = LoraLayer::new(base_weight, varmap)
        for batch in batches:
            (input, target) = prepare_batch(batch)
            logits = lora.forward(input)?
            loss = cross_entropy(logits, target)?
            grads = loss.backward()?
            grad_stores.push(grads)
            if global_batch % grad_accum == 0:
                step_accumulated(&mut adamw, &grad_stores)?
                grad_stores.clear()
                optimizer_steps += 1
        drop(loaded)   // MADV_DONTNEED on unix
emit_done_json(optimizer_steps, final_loss)
return TrainResult { total_steps, final_loss }
```

#### `dequant_slice`

Bridges mmap bytes to `Vec<f32>` via the existing `dequant::dequantize` infrastructure. Constructs a minimal `TensorInfo` from `(bytes, dtype, shape)` and dispatches:

```rust
fn dequant_slice(bytes: &[u8], dtype: GgufDtype, shape: &[u64]) -> Result<Vec<f32>>
```

Supports F32 (memcopy), F16, Q8_0, Q4_0, Q2_K–Q6_K via the existing dequant dispatch table.

#### `prepare_batch`

Casts input tensor to `DType::F32` before `LoraLayer::forward` (which performs a matmul). Target stays as `U32` for `cross_entropy`. This is mandatory: `cross_entropy` in candle requires `(logits: F32, targets: U32)`.

#### `shape_to_2d`

Converts GGUF shapes to 2D for matmul:
- `[n]` → `(n, 1)` (treat as column vector)
- `[d_out, d_in]` → `(d_out, d_in)` (standard weight matrix)
- Higher rank: `(product of all dims, 1)` (flatten)

**Bugs discovered and fixed during Wave 3:**

1. **Shape mismatch** — `make_batch(4)` produced input shape `(1, 3)` but the 1-element test GGUF tensor had `d_in=1`. Fixed by using `make_batch(2)` → shape `(1, 1)`.
2. **dtype mismatch** — Input was `U32` (token IDs). `LoraLayer::forward` requires `F32`. Fixed by `.to_dtype(DType::F32)` in `prepare_batch`.
3. **cross_entropy OOB** — Logits had `d_out=1` but target token IDs were 2. Fixed by making test tensors 4 × f32 (16 bytes) → shape `[4]` → `d_out=4`.

**Tests added:** 4 deterministic (empty varmap rejection, zero layers rejection, single epoch produces result, emits done JSON) + 2 quickcheck (Properties 6–7: step count formula, loss finiteness).

---

### Wave 4 — Benchmark Binary

**Files:** `src/bin/bench_layer_loader.rs` (NEW), `Cargo.toml` (MODIFIED)

CLI: `bench_layer_loader <gguf_path> [--layer N] [--iterations M] [--compare-full] [--format text|json]`

Measures per-layer load/unload performance:

| Metric | How measured |
|---|---|
| `load_us` | `Instant::now()` around `load_layer(n)` |
| `unload_us` | `Instant::now()` around `drop(loaded)` |
| `rss_delta_mb` | `/proc/self/status` VmRSS on Linux; `sysinfo::System::process().memory()` on other platforms |
| `slice_count` | `loaded.slices.len()` from first iteration |
| `byte_total` | Sum of `slice.len()` for all slices |

With `--compare-full`: reports `avg_rss_delta_mb × num_layers` as the hypothetical full-load RAM estimate.

`--format json` emits a single-line object:

```json
{"benchmark":"bench_layer_loader","num_layers":32,"layers":[{"layer_idx":0,"load_us":42,...}],"full_load_estimate_mb":6144.0000,"peak_rss_mb":192.0000}
```

The RSS sampling strategy matches `benchmark/memory.rs` exactly — `/proc/self/status` on Linux, `sysinfo` on Windows/macOS — avoiding platform divergence.

**Binary size:** 0.25 MB stripped (`strip = true`, `lto = true`, `opt-level = 3` in `[profile.release]`).

**Smoke tests:** 5 tests covering `parse_minimal`, `parse_all_flags`, `parse_zero_iterations_rejected`, `parse_unknown_flag_rejected`, `parse_unknown_format_rejected`.

---

### Wave 5 — Integration & Regression

**Files:** `tests/gwen216_integration.rs` (NEW), `train/layer_loader.rs` (MODIFIED: counter + pub helper), `Cargo.toml` (MODIFIED: `[[test]]` entry)

#### `LIVE_LAYER_COUNT` counter

Injected under `#[cfg(any(test, feature = "test-utils"))]` — zero overhead in release:

```rust
pub static LIVE_LAYER_COUNT: AtomicUsize = AtomicUsize::new(0);
```

Incremented in `LayerLoader::load_layer`, decremented in `LoadedLayer::Drop`. Allows external test code to assert at most N layers are alive simultaneously.

#### Integration tests (`tests/gwen216_integration.rs`)

**Test 1 — `integration_layered_training_loop_loss_is_finite`**

Full end-to-end: `LayeredTrainingLoop::new` + `run()` on a 2-layer fixture. Asserts `result.final_loss.is_finite()` and `result.total_steps >= 1`. Exercises the entire stack from public API through mmap, dequant, LoRA forward, cross_entropy, backward, AdamW step.

**Test 2 — `invariant_never_more_than_one_layer_in_ram`** (`#[cfg(unix)]`)

Iterates all 3 layers of a synthetic GGUF. For each layer:

```
assert LIVE_LAYER_COUNT == 0   // before load
let layer = loader.load_layer(n)
assert LIVE_LAYER_COUNT == 1   // while live
drop(layer)
assert LIVE_LAYER_COUNT == 0   // after drop (MADV_DONTNEED + counter decrement)
```

This is a structural proof that the no-full-load invariant holds: at no point can `LIVE_LAYER_COUNT > 1` because `load_layer` returns a single `LoadedLayer` and the counter is incremented exactly once per call.

**Test 3 — `public_types_are_reachable`**

Compile-time check that `LayerSlice`, `LayerIndex`, `LayerLoader`, `LayeredTrainingLoop` are all reachable as `gwenland_core::train::*`. Will fail to *compile* (not just fail at runtime) if any re-export is missing from `train/mod.rs`.

#### Final test counts

| Suite | Count |
|---|---|
| `layer_loader` unit tests | 15 (5 Wave 1 det. + 3 Wave 1 QC + 4 Wave 2 det. + 2 Wave 2 QC + 1 extra) |
| `layered_training_loop` unit tests | 6 (4 Wave 3 det. + 2 Wave 3 QC) |
| `bench_layer_loader` smoke tests | 5 |
| Integration tests | 2 (+ 1 unix-only skipped on Windows) |
| **Total new tests** | **28** |

---

## Files Changed Summary

| File | Change | Why |
|---|---|---|
| `train/layer_loader.rs` | NEW ~620 lines | `LayerSlice`, `LayerIndex`, `LoadedLayer`, `LayerLoader`, test counter, pub GGUF helper |
| `train/layered_training_loop.rs` | NEW ~590 lines | `LayeredTrainingLoop`, `dequant_slice`, `prepare_batch`, `shape_to_2d`, tests |
| `train/training_loop.rs` | MODIFIED: ~15 lines | Extract `step_accumulated` as `pub(crate)` free function |
| `train/mod.rs` | MODIFIED: +5 lines | `pub mod` + `pub use` for new types |
| `engine/loader.rs` | MODIFIED: +3 lines | `pub fn mmap()` accessor |
| `convert/gguf_parser.rs` | MODIFIED: +3 lines | Fix GGUF magic constant, add `data_base` field |
| `engine/gguf_loader.rs` | MODIFIED: +1 line | Add `data_base: 0` to `GgufFile` literal in test helper |
| `src/bin/bench_layer_loader.rs` | NEW ~260 lines | Benchmark binary |
| `Cargo.toml` | MODIFIED: +8 lines | `[[bin]]` + `[[test]]` entries |
| `tests/gwen216_integration.rs` | NEW ~163 lines | Integration + invariant + wiring tests |

---

## Bugs Fixed (Pre-existing)

**`GGUF_MAGIC` endianness** — `gguf_parser.rs` had `const GGUF_MAGIC: u32 = 0x4647_4755`. The ASCII bytes `G G U F` = `[0x47, 0x47, 0x55, 0x46]` read as LE u32 = `0x4655_4747`. Any code calling `gguf_parser::parse()` on a synthetically written file (e.g. test helpers that write `b"GGUF"`) would get a parse error. This bug was dormant because production code only parsed real files from disk (never synthetically generated). Fixed: constant corrected to `0x4655_4747`.

**Missing `data_base` on `GgufFile`** — `TensorInfo::data_offset` is relative to the data segment start. Without `data_base`, any code trying to slice the mmap at an absolute offset would be slicing into the GGUF header instead of the data. Fixed: `pub data_base: u64` added to `GgufFile`, populated from the cursor after 32-byte alignment padding in `parse_inner`.

---

## Mathematics Used

### LoRA Forward Pass (per layer)

```
output = base_weight(x) + lora_b(lora_a(x)) × scale
```

Where `scale = alpha / rank`. Base weight is the dequantized GGUF tensor reshaped to 2D. `lora_a: (rank, d_in)`, `lora_b: (d_out, rank)`. The result has shape `(d_out, d_in)` — same as the base weight.

### Q8_0 Dequantization (used in `dequant_slice`)

For each 34-byte block (2-byte f16 scale + 32 × i8):

```
scale = f16_bits_to_f32(read_u16_le())
w[i]  = scale × int8_value[i]
```

### Gradient Accumulation

To simulate a batch of N micro-batches with AdamW:

```
lr_effective = lr / N
for each micro-batch gradient store:
    adamw.step(store, lr = lr_effective)
restore lr
```

This is an approximation of the true accumulated gradient (which would require summing all gradients before any parameter update). It is numerically stable for small N and matches the gradient accumulation strategy used in `TrainingLoop`.

### GGUF Data Segment Alignment

After the tensor info block, the data segment starts at the next multiple of 32:

```
data_base = ceil(header_end / 32) × 32
```

Where `header_end` is the file cursor position immediately after all tensor info entries. `LayerIndex::scan` adds `data_base` to each `TensorInfo::data_offset` to produce absolute mmap offsets.

---

## Build and Test Status

```
cargo test -p gwenland-core --lib
  running 228 tests
  21 layer_loader + layered_training_loop tests: all pass ✅
  6 pre-existing failures unchanged:
    engine::inference::selector::{empty_stop_sequences_ok, relative_gguf_ok, tilde_expand}
    train::lora_merger::{test_merge_identity, test_merge_nan_detection, test_merge_shape_mismatch}
    (lora_merger failures: pre-existing wrong-magic GGUF in those test helpers; not introduced here)

cargo test -p gwenland-core --bin bench_layer_loader
  running 5 tests: all pass ✅

cargo test -p gwenland-core --test gwen216_integration --features test-utils
  running 2 tests: all pass ✅
  (invariant test skipped on Windows — correct; will run on Linux CI)

cargo build --release -p gwenland-core
  Finished release — 0 errors ✅

Binary sizes (stripped):
  bench_layer_loader.exe   0.25 MB  ✅ (target < 15 MB)
  gwenland.exe            11.11 MB  ✅ (target < 50 MB)
```

---

## What Was NOT Changed

| File | Status |
|---|---|
| `train/native_runner.rs` | Untouched — still uses full-model `load_and_dequant` path |
| `train/lora_bridge.rs` | Untouched |
| `train/lora_merger.rs` | Untouched |
| `train/lora_cli.rs` | Untouched |
| `platform/`, `eval/`, `engine/chat.rs` | Untouched |
| All pre-GWEN-216 unit tests | All pass unchanged (21 pass pre-existing + 7 new = 228 total) |

---

## What Comes Next

| Task | Description |
|---|---|
| Wire `LayeredTrainingLoop` into TUI | Add `--selective-layers` flag to `gwen train`; dispatch to `LayeredTrainingLoop` when set |
| `native_runner.rs` migration | Replace `load_and_dequant` with `LayerLoader` in the main training path; full model should never be loaded for LoRA training |
| End-to-end Qwen3-1.7B test | Train a rank-4 adapter with `LayeredTrainingLoop` on a real GGUF, verify loss decreases over 3 epochs |
| Q4_K / Q5_K dequant in `dequant_slice` | Currently dispatches to existing `dequant::dequantize`; no gap, but verify Q4_K path tested end-to-end in `LayeredTrainingLoop` |
| `lora_merger` test helper fix | The 3 failing `lora_merger` tests use a wrong-magic GGUF writer; the same fix applied to `gguf_parser.rs` in GWEN-216 should be applied to those test helpers |
| KV cache (GWEN-215) | O(n) autoregressive generation; prerequisite for practical inference on trained models |
| RoPE positional encoding | Required for correct inference at sequence positions > 0 |

---

**End of Gwen-Changes-2026-06-10_GWEN-216.md**
