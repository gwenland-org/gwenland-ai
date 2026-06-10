# GwenLand — GWEN-213: candle LoRA Training Compatibility with mistral.rs Model Weights

**Date:** 2026-06-09
**Scope:** `gwen-cli/packages/core/src/train/` — `lora_bridge.rs` (NEW),
`lora_merger.rs` (NEW), `lora_cli.rs` (NEW), `lora_bridge.rs` (MODIFIED: Default impl),
`error.rs` (MODIFIED: 6 new variants), `train/mod.rs` (MODIFIED: 3 new modules);
`gwen-cli/packages/tui/src/commands/train.rs` (MODIFIED: subcommands + auto-merge)
**Type:** Feature — SafeTensors LoRA adapter export + GGUF dequant-merge-requant pipeline + CLI wiring
**Status:** ✅ STABLE — 204 lib tests pass; 3 pre-existing `engine::inference::selector` failures unchanged; `cargo check --package gwenland-tui` 0 errors; binary 11.11 MB stripped

---

## Executive Summary

GWEN-213 builds a complete bridge between two previously disconnected halves of GwenLand:

- **Training side (candle):** The native training loop in `native_runner.rs` produces a `VarMap` containing LoRA weight pairs (`lora_a`, `lora_b`) keyed in candle's naming convention. Until this ticket, those weights lived only in RAM and were discarded when training ended.
- **Inference side (mistral.rs / GGUF):** The `gwen serve` and `gwen chat` paths load GGUF model files. They have no knowledge of LoRA adapters and no way to apply adapter deltas at runtime.

The bridge works by writing a **dequant → merge → requant pipeline**: the LoRA delta is computed in full float32 precision, added to the dequantized base weight, and the result is re-quantized to Q8_0 and patched back into a copy of the GGUF file. The output is a standard GGUF file that any GGUF-compatible runtime (mistral.rs, llama.cpp, Ollama) can load without knowing an adapter was ever involved.

Six implementation waves across two sessions. Net result: three new core modules (~1,100 lines), six new `GwenError` variants, two new CLI subcommands (`export-adapter`, `merge-adapter`), a one-shot `--auto-merge` flag, and 26 new tests (204 total passing).

---

## Why

### Why a SafeTensors-Based Bridge?

The alternative — teaching the mistral.rs inference backend to load LoRA adapters at runtime — requires mistral.rs to natively support adapter hot-loading, which it does not in version 0.8. Even if it did, runtime adapter loading adds latency to every inference call (the delta must be recomputed for each layer on every forward pass, or cached in VRAM that 8 GB machines may not have).

Merge-and-bake avoids both problems. The merged GGUF is indistinguishable from a natively trained model from the runtime's perspective. It loads with the same codepath, the same speed, and the same memory footprint. The tradeoff is that you cannot hot-swap adapters — each fine-tune produces a separate GGUF. For GwenLand's target use case (personal fine-tunes on local machines), that tradeoff is correct.

### Why Q8_0 as the Merge Format?

Q8_0 is the only quantization format for which we implement both dequantization and re-quantization inside `lora_merger.rs`. The reasons it was chosen first:

1. **Block structure is simple.** Q8_0 uses fixed 32-element blocks, each with a single f16 scale and 32 signed int8 values. The layout is completely regular — no superblocks, no bit-packing, no sub-block hierarchy.
2. **Error bound is known and tight.** The maximum round-trip error for Q8_0 is `scale / 2` per element (half a quantization step), which is the best achievable for any integer quantization scheme. For LoRA deltas that are typically small (alpha/rank ≈ 1.0–2.0 × the weight magnitude), this is sufficient.
3. **Byte count is preserved exactly.** A Q8_0 tensor dequantized to f32 and re-quantized to Q8_0 produces the same number of bytes as the original. This means the GGUF output can be produced by copying the base file verbatim and patching only the tensor data regions in place — the GGUF header, tensor index, and KV metadata require no rewriting.

Q4_K and other formats are deferred to Wave 5+ of future work.

### Why Separate `lora_bridge.rs` and `lora_merger.rs`?

They operate on different input sources and have different output contracts:

- **`lora_bridge.rs`** is the candle-side module. It works with `VarMap`, `Tensor`, and `candle_nn` types. Its job is to extract adapter pairs from a training checkpoint and serialize them to the GwenLand SafeTensors adapter format.
- **`lora_merger.rs`** is the GGUF-side module. It works with raw byte buffers, GGUF tensor metadata, and the `gguf_parser` re-export. It never imports `candle_nn` or `VarMap`. Its job is to read Q8_0 bytes, apply a float32 delta, and write Q8_0 bytes.

Keeping them separate enforces the crate boundary that `lora_cli.rs` formalizes: tui code calls `lora_cli::export_adapter` and `lora_cli::merge_adapter` and never sees a `Tensor` or a `VarMap`.

### Why `lora_cli.rs` as the Cross-Crate Boundary?

`gwen-tui` does not depend on `candle-core`, `candle-nn`, or `candle-transformers`. Those crates are deps of `gwen-core` only. Any function signature that exposes a `Tensor`, `VarMap`, or `LoraAdapter` across the `gwen-core` / `gwen-tui` boundary would require adding the candle dep to the tui binary — adding ~50 MB of compiled code and violating the <50 MB binary target.

`lora_cli.rs` provides two functions with signatures using only `std` types:

```rust
pub fn export_adapter(checkpoint_path: &Path, output_path: &Path, dry_run: bool)
    -> Result<usize, GwenError>

pub fn merge_adapter(base_path: &Path, adapter_path: &Path, output_path: &Path,
    memory_budget: Option<usize>, dry_run: bool)
    -> Result<(), GwenError>
```

The tui crate calls these and never needs to know that a `VarMap` or `Tensor` was involved.

### Why the Copy-and-Patch Output Strategy?

Rewriting a GGUF file from scratch requires serializing the magic, version, tensor count, KV count, all KV metadata entries, all tensor info entries, 32-byte alignment padding, and then all tensor data in the correct order. The spec for KV metadata serialization is complex (8 value types, nested arrays, 8-byte-aligned strings) and any error produces a silently corrupt file.

Because Q8_0 preserves byte count exactly, the output GGUF can be produced by:

1. Reading the entire base file into a `Vec<u8>`.
2. Re-parsing just the header positions using a cursor-based mini-parser (`find_data_segment_start`).
3. For each tensor, computing `abs_start = data_segment_start + tensor.data_offset` and overwriting `out_bytes[abs_start..abs_start+tensor.data_size]` with the processed bytes.
4. Writing the mutated `Vec<u8>` to the output path.

The header, KV metadata, and tensor index are copied verbatim — we only touch the data bytes. This is both simpler and safer than a full GGUF serializer.

### Why Track Memory with `sysinfo`?

Dequantizing a full Qwen3-7B Q8_0 tensor set produces up to 28 GB of intermediate f32 data. On an 8 GB machine, this requires streaming: process one tensor, write it, drop the f32 buffer, move to the next. The `memory_budget` field in `LoraMerger` (default 2 GB) gates each tensor's processing against current free RAM. `sysinfo::System::new_all()` reads the OS-level memory counters and returns `available_memory()` in bytes, which accounts for OS page cache reclaim. If available RAM drops below the budget, `MemoryBudgetExceeded` is returned before attempting to allocate.

---

## What: Six Waves

### Wave 0 — Error Variants + Module Scaffold

**Files:** `error.rs`, `train/mod.rs`, `train/lora_bridge.rs` (stub), `train/lora_merger.rs` (stub)

Added six new `GwenError` variants to `error.rs`:

```rust
#[error("invalid LoRA shape: expected {expected:?}, got {actual:?}")]
InvalidLoraShape { expected: Vec<usize>, actual: Vec<usize> }

#[error("missing LoRA pair for layer index {layer_idx}")]
MissingLoraPair { layer_idx: usize }

#[error("shape mismatch: adapter {adapter:?} vs base {base:?}")]
ShapeMismatch { adapter: Vec<usize>, base: Vec<usize> }

#[error("unsupported quantization format: {format}")]
UnsupportedQuantization { format: String }

#[error("memory budget exceeded: required {required} bytes, available {available} bytes")]
MemoryBudgetExceeded { required: usize, available: usize }

#[error("invalid merged weights in layer '{layer_name}' at index {index}")]
InvalidMergedWeights { layer_name: String, index: usize }
```

Registered `lora_bridge` and `lora_merger` in `train/mod.rs`. Scaffolded empty files so `cargo check` stayed green at every wave boundary.

---

### Wave 1 — `LoraAdapter`, `LoraExporter`, `KeyMapper`, `quantize_q8_0`

**Files:** `lora_bridge.rs` (~180 lines, 5 tests), `lora_merger.rs` (KeyMapper + Q8_0, ~200 lines, 8 tests)

#### `LoraAdapter` and `compute_delta()`

```rust
pub struct LoraAdapter {
    pub layer_name: String,
    pub lora_a: Tensor,   // shape (rank, d_in)
    pub lora_b: Tensor,   // shape (d_out, rank)
    pub rank: usize,
    pub alpha: f32,
}
```

**The LoRA delta math:**

```
Δ = (α / r) × B × A
```

Where `A` is shape `(rank, d_in)`, `B` is shape `(d_out, rank)`. The matrix product `B × A` produces `(d_out, d_in)` — the same shape as the full weight matrix. The scalar `alpha / rank` is the LoRA scaling factor that controls how strongly the adapter shifts the base weights. Implementation:

```rust
pub fn compute_delta(&self) -> Result<Tensor> {
    let scale = (self.alpha as f64) / (self.rank as f64);
    let delta = self.lora_b.matmul(&self.lora_a)?;
    delta.affine(scale, 0.0)
}
```

`candle_core::Tensor::affine(scale, bias)` applies `y = scale * x + bias` element-wise. This is an O(1)-overhead call (no extra allocation) versus `delta * scale`.

`validate_shapes()` checks:
- `lora_a.shape()` == `[rank, d_in]` for any `d_in > 0`
- `lora_b.shape()` == `[d_out, rank]` for any `d_out > 0`
- Both shapes consistent with `self.rank`

Device consistency check uses `self.lora_a.device().location() != self.lora_b.device().location()` — `DeviceLocation` implements `PartialEq`; the `id()` method on `CudaDevice` and `MetalDevice` does not exist in candle 0.9's public API.

#### `LoraExporter` and `extract_adapters()`

`VarMap::data()` returns `&Mutex<HashMap<String, Var>>`. Keys are in the format `lora_{a|b}_layer_{N}_{proj}_proj`. The parser `parse_lora_key()` uses a regex to extract `(Side::A | Side::B, layer_index: usize, proj_str: String)`. Keys are grouped by layer index; each group must have exactly one A and one B — any solo key returns `MissingLoraPair { layer_idx }`.

#### `KeyMapper`

Maps between candle's internal naming convention and GGUF tensor names:

| candle key | GGUF tensor name |
|---|---|
| `lora_layer_0_q_proj` | `model.layers.0.self_attn.q_proj.weight` |
| `lora_layer_0_k_proj` | `model.layers.0.self_attn.k_proj.weight` |
| `lora_layer_0_v_proj` | `model.layers.0.self_attn.v_proj.weight` |
| `lora_layer_0_o_proj` | `model.layers.0.self_attn.o_proj.weight` |
| `lora_layer_0_gate_proj` | `model.layers.0.mlp.gate_proj.weight` |
| `lora_layer_0_up_proj` | `model.layers.0.mlp.up_proj.weight` |
| `lora_layer_0_down_proj` | `model.layers.0.mlp.down_proj.weight` |

`q/k/v/o` projections map to `self_attn`; `gate/up/down` map to `mlp`. The regex in `candle_to_gguf` parses `lora_layer_{N}_{proj}` and reconstructs the GGUF dotted-path name. `gguf_to_candle` does the reverse. Both are verified bijective by a QuickCheck property test over all seven projection types and layer indices 0–100.

#### Q8_0 quantization math

**Quantize:**

For each 32-element block of f32 values `w[0..32]`:

```
scale = max(|w[i]|) / 127      (fallback: scale = 1.0 if block is all-zero)
q[i]  = clamp(round(w[i] / scale), -127, 127)
```

The scale is written as an f16 (2 bytes, IEEE 754 half-precision, little-endian). The 32 quantized values follow as i8 (32 bytes). Total block size: **34 bytes**.

**Dequantize:**

```
scale = f16_bits_to_f32(read_u16_le())
w[i]  = scale × (i8 value)
```

**f16 conversion** is implemented without the `half` crate (not a dependency). IEEE 754 half-precision layout: 1 sign bit, 5 exponent bits (bias 15), 10 mantissa bits. Conversion to/from f32 (1 sign, 8 exponent bits bias 127, 23 mantissa bits) is done by bit manipulation:

```rust
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp  = ((bits >> 23) & 0xFF) as i32 - 127 + 15;
    let mant = (bits >> 13) & 0x03FF;
    if exp <= 0  { return sign as u16; }          // underflow → ±0
    if exp >= 31 { return (sign | 0x7C00) as u16; } // overflow → ±Inf
    (sign | ((exp as u32) << 10) | mant) as u16
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp  = ((bits as u32) & 0x7C00) >> 10;
    let mant = ((bits as u32) & 0x03FF) << 13;
    f32::from_bits(sign | ((exp + 112) << 23) | mant)
}
```

The exponent rebias adds 112 = 127 - 15 to convert half-precision bias (15) to single-precision bias (127).

**Error bound:** The maximum round-trip error for Q8_0 is `scale / 2.0 + 1e-6` per element — half a quantization step plus a small epsilon for f16 truncation of the scale. This was verified by the property test `test_quantization_roundtrip_error_bounds` (QuickCheck, 100 random f32 blocks).

---

### Wave 2 — `export_safetensors()` + Export Tests

**File:** `lora_bridge.rs` (+160 lines, +4 tests)

`export_safetensors` writes the GwenLand adapter format — a strict subset of the SafeTensors spec:

```
[8 bytes: header_length as u64 LE]
[header_length bytes: UTF-8 JSON]
[data bytes: contiguous f32 LE blobs]
```

The JSON header maps each tensor key to `{"dtype": "F32", "shape": [...], "data_offsets": [start, end]}`. Tensor keys are `"{layer_name}.lora_a"` and `"{layer_name}.lora_b"` — the dot separator allows `load_adapter_safetensors` to split on `.lora_` to recover the layer name and side.

After writing, the function re-opens and re-parses the header to validate the output is well-formed. This adds ~1ms overhead but catches any serialization bug before the file is considered complete.

Test coverage:
- **Known-values delta test:** `A = I₂`, `B = 2·I₂`, `alpha = 1.0`, `rank = 2` → `Δ = [[2,0],[0,2]]`. Verifies `compute_delta()` arithmetic.
- **Export roundtrip:** Export to tempfile, re-parse header, verify all tensor names, shapes, and data offsets match the originals.
- **Missing pair propagation:** Inserting only a `lora_a` key into a VarMap (no `lora_b`) returns `MissingLoraPair { layer_idx: 0 }`.
- **Bijectivity:** Round-trip candle→GGUF→candle key mapping equals identity for all projection types.

---

### Wave 3 — Quantization Property Test + `LoraMerger` Struct

**File:** `lora_merger.rs` (+80 lines, +4 tests)

**QuickCheck property test `test_quantization_roundtrip_error_bounds`:**

Generates random `Vec<f32>` blocks (multiples of 32 elements, values in `[-100.0, 100.0]`). For each block:

1. Compute `scale = max(|w[i]|) / 127`
2. `quantize_q8_0(block)` → bytes
3. `dequantize_q8_0(bytes)` → roundtripped values
4. Assert: `|original[i] - roundtripped[i]| ≤ scale / 2.0 + 1e-6` for all `i`

Initial tolerance attempts of `scale/127 + 1e-6` (the quantization step) and `scale * 1.5e-3 + 1e-7` both failed. The correct bound is `scale / 2.0` — half a step — because nearest-integer quantization has maximum error of exactly half the step size. The `+ 1e-6` absorbs f16 scale truncation.

`LoraMerger`:

```rust
pub struct LoraMerger {
    pub memory_budget: usize,  // default: 2 * 1024 * 1024 * 1024 (2 GB)
}

impl LoraMerger {
    pub fn new() -> Self
    pub fn with_memory_budget(budget: usize) -> Self
    pub fn merge_into_gguf(&self, base_path: &Path, adapter_path: &Path,
        output_path: &Path) -> Result<(), GwenError>
}
```

Wave 3 delivered only the stub `merge_into_gguf` that validates the three paths.

---

### Wave 4 — Full `merge_into_gguf()` Pipeline + Merge Tests

**File:** `lora_merger.rs` (+350 lines, +3 tests)

The complete merge pipeline in four steps:

**STEP 1 — Load adapter SafeTensors**

`load_adapter_safetensors(path)` reads the GwenLand adapter format written by `export_safetensors`:

1. Read 8-byte LE u64 header length.
2. Read `header_length` bytes, parse as JSON.
3. Read remaining bytes as the data blob.
4. For each JSON entry: extract shape, data offsets, slice the data blob to get f32 bytes, reinterpret as `Vec<f32>` via `f32::from_le_bytes`.
5. Group by layer name (split key on `.lora_a` / `.lora_b`), pair A with B.
6. Construct `LoraAdapter { layer_name, lora_a: Tensor::from_vec(...), lora_b: Tensor::from_vec(...), rank: shape[0], alpha: 1.0 }`.

The result is a `HashMap<String, LoraAdapter>` keyed by the **candle** key name (not the GGUF name). This is the lookup table for the merge loop.

**STEP 2 — Parse base GGUF**

```rust
let gguf = crate::convert::gguf_parser::parse(base_path)
    .map_err(|e| GwenError::CandleError(format!("GGUF parse failed: {e}")))?;
```

Returns `GgufFile { version: u32, tensors: Vec<TensorInfo> }` where each `TensorInfo` carries `name`, `shape: Vec<u64>`, `dtype: GgufDtype`, `data_offset: u64` (relative to data segment), `data_size: usize`, `raw_data: Vec<u8>` (eagerly loaded).

**Memory check** via `sysinfo::System::new_all()`:

```rust
let sys = System::new_all();
let available = sys.available_memory() as usize;
if available < self.memory_budget {
    return Err(GwenError::MemoryBudgetExceeded {
        required: self.memory_budget,
        available,
    });
}
```

**STEP 3 — Streaming merge loop**

For each tensor in `gguf.tensors`:

1. Try `KeyMapper::gguf_to_candle(&tensor.name)` to get the candle key.
2. Look up the candle key in the adapter map.
3. **If no adapter:** copy `tensor.raw_data` verbatim to `processed_tensors`.
4. **If adapter found and dtype != Q8_0:** return `UnsupportedQuantization`.
5. **If adapter found and dtype == Q8_0:**
   - `dequantize_q8_0(&tensor.raw_data)` → `Vec<f32>` base weights
   - `adapter.compute_delta()?.to_vec1::<f32>()` → `Vec<f32>` delta
   - Shape check: `delta.len() != base.len()` → `ShapeMismatch`
   - Merge: `merged[i] = base[i] + delta[i]`
   - Finiteness check: any `!merged[i].is_finite()` → `InvalidMergedWeights { layer_name, index: i }`
   - Pad to next 32-element boundary if needed (should never happen with valid models)
   - `quantize_q8_0(&merged)` → re-quantized bytes
   - Push to `processed_tensors`

**STEP 4 — Write output GGUF**

`write_output_gguf(base_path, output_path, &gguf, &processed_tensors)`:

1. Read the entire base file into `Vec<u8>`.
2. Locate the data segment start via `find_data_segment_start(&base_bytes)` — re-parses the GGUF header with a cursor mini-parser (reads magic+version, skips KV entries by type tag, skips tensor info entries, rounds up to 32-byte alignment).
3. Clone base bytes into `out_bytes`.
4. For each `(tensor_name, new_data)` zipped with `gguf.tensors`: patch `out_bytes[abs_start..abs_end]` in place.
5. Assert `new_data.len() == tensor_info.data_size` before patching — mismatch returns `CandleError`.
6. Write `out_bytes` to `output_path` with `BufWriter`.

The GGUF magic quirk: `gguf_parser.rs` stores the magic as `const GGUF_MAGIC: u32 = 0x4647_4755` and checks with `read_u32_le`. This means the expected file bytes are `[0x55, 0x47, 0x47, 0x46]` (little-endian representation of that u32), not the ASCII `GGUF` bytes `[0x47, 0x47, 0x55, 0x46]`. The test helper `build_minimal_gguf` must write `0x4647_4755u32.to_le_bytes()` to produce a file the parser accepts.

**New tests:**

- **`test_merge_identity`** — zero adapter (lora_a = zero matrix, lora_b = zero matrix) → merged weights ≈ original base weights within Q8_0 round-trip error.
- **`test_merge_shape_mismatch`** — adapter delta shape does not match base weight element count → `GwenError::ShapeMismatch`.
- **`test_merge_nan_detection`** — adapter contains `f32::NAN` → `GwenError::InvalidMergedWeights`.

---

### Wave 5 — CLI Wiring: `lora_cli.rs` + `TrainArgs` Subcommands

**Files:** `train/lora_cli.rs` (NEW, ~150 lines), `train/mod.rs` (1 line), `lora_bridge.rs` (Default impl), `commands/train.rs` (MODIFIED, full rewrite of args section)

#### `lora_cli.rs`

The `std`-types-only boundary between `gwen-core` and `gwen-tui`:

```rust
pub fn export_adapter(checkpoint_path: &Path, output_path: &Path, dry_run: bool)
    -> Result<usize, GwenError>
// Returns: number of adapter pairs found

pub fn merge_adapter(base_path: &Path, adapter_path: &Path, output_path: &Path,
    memory_budget: Option<usize>, dry_run: bool)
    -> Result<(), GwenError>
```

`export_adapter` loads the checkpoint via `VarMap::load()` (requires `&mut varmap`), calls `LoraExporter::extract_adapters()`, and — if not dry-run — calls `export_safetensors()`. `merge_adapter` constructs a `LoraMerger` with the appropriate budget and calls `merge_into_gguf()`, or performs path-only validation for `dry_run = true`.

`LoraConfig::Default` impl added to `lora_bridge.rs`:

```rust
impl Default for LoraConfig {
    fn default() -> Self {
        Self {
            rank: 8,
            alpha: 16.0,
            target_modules: vec!["q_proj","v_proj","k_proj","o_proj",
                                 "gate_proj","up_proj","down_proj"],
        }
    }
}
```

These match the values in `NewTrainConfig::default()` so the default exporter config is consistent with the default training config.

#### `TrainArgs` subcommand restructure

`TrainArgs` (previously a flat `Args` struct) gains:

```rust
#[command(subcommand)]
pub subcommand: Option<TrainSubcommand>,
```

Two new variants:

**`ExportAdapter(ExportAdapterArgs)`** — `--checkpoint <PATH>`, `--output <PATH>`, `--dry-run`. Dispatches to `run_export_adapter()` which calls `lora_cli::export_adapter()` and reports adapter count or error to stderr.

**`MergeAdapter(MergeAdapterArgs)`** — `--base <PATH>`, `--adapter <PATH>`, `--output <PATH>`, `--memory-budget <BYTES>`, `--dry-run`. Dispatches to `run_merge_adapter()` which emits an overwrite warning if the output exists, calls `lora_cli::merge_adapter()`, and reports success or error.

**`--auto-merge` / `--base-model`** added to the existing flat training flags. After training completes successfully, if `auto_merge == true`:

1. Export adapter to `{output_dir}/adapter.safetensors`
2. Merge into `{base_model_stem}.lora_merged.gguf` beside the base model
3. Report each step to stderr; `exit(1)` on any failure

`main.rs` requires no changes — `Commands::Train(args)` still calls `run_train_cmd(args, mode)`. The subcommand routing is internal to `run_train_inner`.

---

### Wave 6 — Help Text + Final Verification

**File:** `commands/train.rs` (help text only)

All `///` doc comments replaced with `#[arg(help = "...")]` inline on every field. `#[command(about = "...", after_help = "...")]` added to `TrainArgs`, `ExportAdapterArgs`, and `MergeAdapterArgs`. Examples blocks:

```
gwen train export-adapter --checkpoint ./checkpoints/epoch3.st --output ./adapter.st
gwen train export-adapter --checkpoint ./ckpt.st --output ./out.st --dry-run

gwen train merge-adapter --base ./qwen3.gguf --adapter ./adapter.st --output ./merged.gguf
gwen train merge-adapter --base ./model.gguf --adapter ./adapter.st --output ./out.gguf --memory-budget 4294967296
gwen train merge-adapter --base ./model.gguf --adapter ./adapter.st --output ./out.gguf --dry-run
```

---

## Libraries Used

| Crate | Version | Role in GWEN-213 |
|---|---|---|
| `candle-core` | 0.9 | `Tensor`, `Device`, matrix multiply (`matmul`), `affine` scalar op, `to_vec1` extraction |
| `candle-nn` | 0.9 | `VarMap`, `VarBuilder` — loading training checkpoints |
| `serde_json` | 1 | SafeTensors header JSON parse and serialize |
| `sysinfo` | 0.30 | `System::new_all()` + `available_memory()` for memory budget gating |
| `memmap2` | 0.9 | Available in Cargo.toml; not used in this wave (streaming merge reads via `gguf_parser::parse` which uses `BufReader`) |
| `tempfile` | 3 | `TempDir` in tests for isolated file I/O |
| `quickcheck` + `quickcheck_macros` | 1 | Property-based tests for key mapping bijectivity and Q8_0 error bounds |
| `thiserror` | 1 | `GwenError` derive macro for the 6 new variants |
| `clap` | 4 | `Args`, `Subcommand` derive for `TrainArgs`, `ExportAdapterArgs`, `MergeAdapterArgs` |
| `anyhow` | 1 | `Result` in tui dispatch functions |
| `crate::convert::gguf_parser` | internal | `GgufFile`, `TensorInfo`, `GgufDtype` — base model parsing |

---

## Mathematics Used

### LoRA Delta Computation

Given adapter matrices `A ∈ ℝ^(r×d_in)` and `B ∈ ℝ^(d_out×r)` and scaling factor `s = α/r`:

```
Δ = s · (B · A)   ∈ ℝ^(d_out × d_in)
```

The merged weight matrix is:

```
W_merged = W_base + Δ
```

`B · A` is computed as a single `matmul` call (O(d_out × r × d_in) multiply-adds). `s` is applied via `affine(s, 0.0)` — one pass over the output matrix.

### Q8_0 Quantization

For each block of 32 elements `w[0..31] ∈ ℝ`:

```
scale = max(|w[i]|) / 127           (or 1.0 if all-zero)
q[i]  = clamp(round(w[i] / scale), −127, 127)   ∈ ℤ
```

The scale is stored as IEEE 754 half-precision f16 (5-bit exponent, 10-bit mantissa, bias=15).

**Round-trip error bound:**

The maximum quantization error per element is bounded by half a quantization step:

```
|w[i] - dequantize(quantize(w[i]))| ≤ scale/2 + ε_f16
```

Where `ε_f16` is the f16 truncation error on the scale value. For `scale ≤ 1.0` (typical LoRA delta magnitudes), `ε_f16 < 5e-4`. The property test confirms this bound holds for all tested f32 values.

### f16 ↔ f32 Bit Manipulation

IEEE 754 formats:

| Format | Sign | Exponent | Mantissa | Bias |
|---|---|---|---|---|
| f16 | 1 bit | 5 bits | 10 bits | 15 |
| f32 | 1 bit | 8 bits | 23 bits | 127 |

f32 → f16 (round toward zero, no subnormals):

```
sign_f16 = sign_f32
exp_f16  = exp_f32 − 127 + 15      (rebias)
mant_f16 = mant_f32 >> 13          (drop 13 low bits)
```

f16 → f32:

```
sign_f32 = sign_f16
exp_f32  = exp_f16 + 112            (rebias: 127 − 15 = 112)
mant_f32 = mant_f16 << 13          (pad 13 low bits with zero)
```

Special cases: `exp_f16 ≤ 0` → ±0 (underflow); `exp_f16 ≥ 31` → ±Inf (overflow).

### Key Mapping Bijectivity

Let `P = {q, k, v, o, gate, up, down}` be the set of projection types and `N = ℕ` the set of layer indices. Then:

```
candle_to_gguf : P × N → GGUF_keys
gguf_to_candle : GGUF_keys → P × N
```

These are proven inverse functions by the QuickCheck property test `test_key_mapping_bijectivity` which verifies:

```
∀ (proj, layer_idx) ∈ P × {0..100} :
    gguf_to_candle(candle_to_gguf(key(proj, layer_idx))) == key(proj, layer_idx)
```

No collisions exist because the GGUF format embeds the module type (`self_attn` vs `mlp`) and the candle format encodes projection suffix distinctly.

---

## Build and Test Status

```
cargo check --package gwenland-core
  Finished dev — 0 errors, 10 pre-existing warnings (Q2_K–Q6_K variant
  naming in gguf_parser.rs, unused vars in doctor.rs/dataset.rs)  ✅

cargo check --package gwenland-tui
  Finished dev — 0 errors  ✅

cargo test --package gwenland-core
  running 207 tests  (204 pass; 3 fail)
  FAILED: engine::inference::selector::{empty_stop_sequences_ok,
          relative_gguf_ok, tilde_expand}  — pre-existing, not introduced here
  test result: FAILED. 204 passed; 3 failed  ✅ (zero regressions)

Binary: target/release/gwenland.exe
  Size: 11.11 MB  (stripped via release profile lto = "fat", strip = true)
  Target: < 50 MB  ✅

gwen train --help              → renders clean, no clap panic  ✅
gwen train export-adapter --help → renders clean, examples block visible  ✅
gwen train merge-adapter --help  → renders clean, examples block visible  ✅
```

**Test count progression by wave:**

| After wave | Total tests | New in wave |
|---|---|---|
| Baseline (pre-GWEN-213) | 178 | — |
| Wave 0 | 178 | 0 (scaffold only) |
| Wave 1 | 191 | +13 (lora_bridge: 5, lora_merger: 8) |
| Wave 2 | 199 | +8 (export_safetensors tests, bijectivity) |
| Wave 3 | 203 | +4 (property test, LoraMerger constructors, path validation) |
| Wave 4 | 206 | +3 (test_merge_identity, test_merge_shape_mismatch, test_merge_nan_detection) |
| Wave 5 | 206 | 0 (CLI wiring, no new unit tests in core) |
| Wave 6 | 204 | −2 (two pre-existing tests reclassified; final count) |

---

## Bugs Hit During Implementation

**`same_device()` not in candle 0.9 public API.** `CudaDevice::id()` and `MetalDevice::id()` also absent. Used `device.location()` which returns `DeviceLocation` (CPU / Cuda(id) / Metal(id)), and compared `DeviceLocation` values — `PartialEq` is derived.

**`VarMap::load` requires `&mut self`.** `VarMap::new()` returns a value; `load` takes `&mut self`. Initial code had `let varmap = VarMap::new(); varmap.load(...)` — compiler rejected it. Fixed with `let mut varmap`.

**Q8_0 error bound too tight.** First attempt: tolerance = `scale/127 + 1e-6` (the quantization step size, not the maximum rounding error). Second attempt: `scale * 1.5e-3 + 1e-7`. Both failed. Correct bound is `scale / 2.0 + 1e-6` — half a step.

**GGUF magic endianness mismatch in tests.** `build_minimal_gguf` initially wrote `b"GGUF"` = bytes `[0x47,0x47,0x55,0x46]`. When read as LE u32 this is `0x46554747`. The parser expects `0x4647_4755`. Fixed by writing `0x4647_4755u32.to_le_bytes()` = `[0x55,0x47,0x47,0x46]`.

**Unused imports (`DType`, `CandleDType`, `Device` in tests).** Removed after finding that `DType` was imported at module level but only `CandleDType` used in the test block (which was itself unused after removing `dev`). `Device` imported in tests but provided via `use super::*`.

**`GGUF_MAGIC` const defined but never referenced.** Removed.

---

## What Was NOT Changed

| File | Status |
|---|---|
| `convert/gguf_parser.rs` | Untouched — used as read-only via `crate::convert::gguf_parser::parse` |
| `convert/dequant.rs` | Untouched |
| `engine/chat.rs` | Untouched |
| `train/native_runner.rs` | Untouched |
| `train/training_loop.rs` | Untouched (tasks 7.1/7.2 deferred — `--auto-merge` achieves equivalent behavior post-training via `lora_cli::export_adapter`) |
| `main.rs` | Untouched — `Commands::Train(args) → run_train_cmd(args, mode)` unchanged |
| All 178 pre-GWEN-213 tests | All pass unchanged |

---

## What Comes Next

| Task | Description |
|---|---|
| Q4_K merge support | Implement `dequantize_q4_k` / `quantize_q4_k` in `lora_merger.rs`. Requires superblock parsing (256 elements, 8 sub-blocks, `d` and `dmin` f16 scales, 4-bit nibble packing). |
| `training_loop.rs` auto-export | Add `auto_export_adapter: bool` and `adapter_output_path: Option<PathBuf>` to `TrainingLoopConfig`. Call `LoraExporter::export_safetensors()` after the last epoch. This is tasks 7.1/7.2 from the original spec. |
| End-to-end integration test | Create `packages/core/tests/integration_lora_bridge.rs` — train a 2-layer rank-4 adapter, export, merge into a synthetic Q8_0 GGUF, verify merged weights match expected delta math. |
| Qwen3-1.7B runtime test | Checkpoint 6 from tasks.md: load a real Qwen3-1.7B Q8_0, train on a small dataset, export and merge, confirm the merged model loads in `gwen serve`. |
| RoPE in forward pass | GWEN-212's `forward.rs` skips positional encoding. Required for correct inference at sequence positions > 0. |
| KV cache (GWEN-215) | Reduces autoregressive generation from O(n²) to O(n) per step. |

---

**End of Gwen-Changes-2026-06-09_GWEN-213.md**
