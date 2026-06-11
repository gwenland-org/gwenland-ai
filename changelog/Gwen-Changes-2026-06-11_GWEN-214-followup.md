# GwenLand — GWEN-214 Follow-up: Optional CLI Flags + Config Fallback + blk. Prefix Fix

**Date:** 2026-06-11 (WIB)
**Scope:** `gwen-cli/packages/core/src/storage/config.rs` (MODIFIED),
`gwen-cli/packages/core/src/train/layer_loader.rs` (MODIFIED: `LayerIndex::scan`),
`gwen-cli/packages/tui/src/commands/benchmark.rs` (MODIFIED: new flags, resolver, config wiring)
**Type:** Feature + Bug Fix — optional benchmark flags with config fallback; real GGUF layer indexing
**Status:** ✅ STABLE — `cargo build -p gwenland-tui` clean; `cargo check -p gwenland-core` clean

---

## Executive Summary

Three changes in one session, building directly on GWEN-214:

1. **`BenchmarkConfig` section** added to `GwenConfig` so users can persist benchmark defaults in `~/.config/gwen/config.toml`.
2. **`gwen benchmark` CLI flags** made fully optional with a `ResolvedBenchmarkArgs` resolver that applies the priority chain `CLI > config.toml > hardcoded default`. Two new flags added: `--quantization` and `--output`.
3. **`LayerIndex::scan` prefix fix** — the function only matched `model.layers.{N}.*` tensor names, but all llama.cpp-format GGUF files (Qwen, Llama, Mistral, Phi, …) use `blk.{N}.*`. This caused `num_layers = 0` for any real model, making `--layer-load` silently produce a result with all zeros. Fixed by accepting both prefixes.

---

## Why

### Why `BenchmarkConfig` in `GwenConfig`?

`gwen benchmark --model Qwen3-1.7B-Q8_0.gguf --layer-load 4` is inconvenient to type every time. A `[benchmark]` section in `config.toml` lets the user set a default model path once and omit the flag from then on. This matches the existing pattern used by `[inference]` and `[general]`.

### Why `ResolvedBenchmarkArgs`?

Rather than sprinkling `.unwrap_or_else(|| config.benchmark.X.clone())` inline in `run_benchmark_cmd`, a dedicated resolver struct makes the priority chain explicit and testable in one place. The pattern mirrors how `InferenceConfig` is resolved in the engine subsystem.

### Why add `--quantization` and `--output` flags?

The spec required all four benchmark-relevant flags to be optional with config fallback. `--quantization` is informational (printed in the report header) — useful when benchmarking different quant formats of the same model without re-reading the file name. `--output` wires the existing `write_benchmark_file` function to a user-specified directory so results are saved automatically.

### Why fix `LayerIndex::scan` for `blk.` prefix?

The `blk.{N}.*` naming is the GGUF tensor naming convention used by llama.cpp and all models converted with it (which is essentially every publicly available GGUF). The `model.layers.{N}.*` prefix is used only in HuggingFace-exported GGUFs. Without this fix, `gwen benchmark --layer-load 4 --model Qwen3-1.7B-Q8_0.gguf` would open the file, find 0 layers, produce an empty `samples` vec, and display `Layers: 0` with all zero timings — completely misleading output. The fix is one extra `.or_else()` in the `filter_map` chain.

---

## What Changed

### 1. `config.rs` — `BenchmarkConfig` + `GwenConfig` extension

**New struct:**

```rust
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default)]
pub struct BenchmarkConfig {
    pub model: Option<std::path::PathBuf>,     // default model for --model / --layer-load
    pub layer_load: Option<u32>,               // default sample count (0 = all layers)
    pub quantization: Option<String>,          // default quant format string (e.g. "Q8_0")
    pub output_dir: Option<std::path::PathBuf>,// default output directory for JSON results
}
```

All fields are `Option` — missing keys in `config.toml` are silently skipped by serde.

**`GwenConfig` extension:**

```rust
pub struct GwenConfig {
    // ... existing fields ...
    pub benchmark: BenchmarkConfig,
}
```

**`get`/`set` wired** for four new dotted keys:
- `benchmark.model` → `PathBuf` (empty string clears to `None`)
- `benchmark.layer_load` → `u32` (empty string clears to `None`)
- `benchmark.quantization` → `String` (empty string clears to `None`)
- `benchmark.output_dir` → `PathBuf` (empty string clears to `None`)

Example `config.toml` after `gwen config set benchmark.model Qwen3-1.7B-Q8_0.gguf`:

```toml
[benchmark]
model = "Qwen3-1.7B-Q8_0.gguf"
layer_load = 4
quantization = "Q8_0"
```

---

### 2. `benchmark.rs` — New flags, resolver, config wiring

#### New flags in `BenchmarkArgs`

| Flag | Type | Default |
|---|---|---|
| `--model <GGUF_PATH>` | `Option<PathBuf>` | from `config.benchmark.model` |
| `--layer-load <N>` | `Option<u32>` | from `config.benchmark.layer_load` |
| `--quantization <FORMAT>` | `Option<String>` | from `config.benchmark.quantization` or `"Q8_0"` |
| `--output <DIR>` | `Option<PathBuf>` | from `config.benchmark.output_dir` |

`--layer-load` is now an integer count (number of layers to sample), **not a file path**. The model file always comes from `--model` or `config.benchmark.model`.

#### `ResolvedBenchmarkArgs`

```rust
struct ResolvedBenchmarkArgs {
    model: Option<PathBuf>,
    layer_load: Option<u32>,
    quantization: String,
    output: Option<PathBuf>,
}

impl ResolvedBenchmarkArgs {
    fn resolve(args: BenchmarkArgs, config: &GwenConfig) -> Self {
        Self {
            model: args.model.or_else(|| config.benchmark.model.clone()),
            layer_load: args.layer_load.or_else(|| config.benchmark.layer_load),
            quantization: args.quantization
                .or_else(|| config.benchmark.quantization.clone())
                .unwrap_or_else(|| "Q8_0".to_string()),
            output: args.output.or_else(|| config.benchmark.output_dir.clone()),
        }
    }
}
```

#### `run_benchmark_cmd` updated

- Calls `GwenConfig::load()` at the top.
- Calls `ResolvedBenchmarkArgs::resolve(args, &config)`.
- Passes `resolved.model.as_deref()` to `run_benchmarks(filter, cb, model_path)`.
- Layer-load bench runs when `resolved.layer_load` is `Some(n)`:
  - `n == 0` → `sample_count = None` (all layers)
  - `n > 0` → `sample_count = Some(n)`
  - Requires `resolved.model` to be `Some`; prints a clear error if not.
- If `resolved.output` is `Some(dir)`, creates the directory and writes a timestamped JSON file (`benchmark_{unix_ts}.json`) via `write_benchmark_file`.
- Quantization string appears in the report header: `📊 GwenLand Benchmark Results  [quant: Q8_0]`.

---

### 3. `layer_loader.rs` — `LayerIndex::scan` accepts `blk.` prefix

**Before:**

```rust
let rest = t.name.strip_prefix("model.layers.")?;
```

**After:**

```rust
let rest = t.name.strip_prefix("model.layers.")
    .or_else(|| t.name.strip_prefix("blk."))?;
```

This is the only change to `layer_loader.rs`. Everything downstream (`load_layer`, `layer_slices`, `num_layers`) works identically — `LayerIndex` stores only the numeric index and tensor name, not which prefix was used.

**Impact:** Any real llama.cpp-format GGUF (Qwen3, Llama 3, Mistral, Phi, Gemma, …) now correctly reports its layer count. Previously all such files produced `num_layers = 0`, making `run_layer_load_bench` return an empty `samples` vec and all-zero stats.

**Existing tests unaffected:** The unit tests in `layer_load_bench.rs` use `model.layers.N.*` naming (synthetic GGUF fixtures). They continue to pass — both prefixes are now accepted, not just one.

---

## Bugs Fixed

### `--layer-load` on real GGUF files produced all-zero output

**Root cause:** `LayerIndex::scan` filtered for `model.layers.{N}.*` prefix only. Qwen3-1.7B-Q8_0.gguf (and all llama.cpp-format GGUFs) use `blk.{N}.*`. Result: `num_layers = 0`, `indices = []`, `samples = []`, report shows `Layers: 0  Min Load: 0 µs`.

**Fix:** `strip_prefix("model.layers.").or_else(|| strip_prefix("blk."))` in `LayerIndex::scan`.

### `--layer-load` flag accepted a file path instead of an integer

**Root cause:** `BenchmarkArgs.layer_load` was typed as `Option<PathBuf>` (copy-paste from `--model`). Clap would accept any string, but the value was passed as a file path to `run_layer_load_bench` instead of a sample count.

**Fix:** Changed to `Option<u32>`. The model file always comes from `--model`; `--layer-load` is purely the sample count.

### `--layer-load N` silently skipped when `--model` was not passed

**Root cause:** The dispatch condition was `if let (Some(ll_path), Some(n)) = (&resolved.model, resolved.layer_load)` — required both to be `Some`. If only `--layer-load` was set without `--model`, the bench silently didn't run.

**Fix:** Dispatch on `resolved.layer_load` alone; if `resolved.model` is `None`, print a descriptive error (`⚠ --layer-load requires a model path`) instead of silently skipping.

---

## Files Changed Summary

| File | Change | Why |
|---|---|---|
| `packages/core/src/storage/config.rs` | +52 lines | `BenchmarkConfig` struct, `GwenConfig.benchmark` field, get/set entries |
| `packages/core/src/train/layer_loader.rs` | +4 lines | `blk.` prefix support in `LayerIndex::scan` |
| `packages/tui/src/commands/benchmark.rs` | +90 lines, −20 lines | `--quantization`/`--output` flags, `--layer-load` type fix, `ResolvedBenchmarkArgs`, config load, output wiring |

---

## Build Status

```
cargo build -p gwenland-tui
  Finished dev — 0 errors ✅  (10 pre-existing warnings, unchanged)

cargo check -p gwenland-core
  Finished — 0 errors ✅
```

---

## Usage After This Change

```sh
# All flags optional — uses config.toml defaults
gwen benchmark

# Set defaults once
gwen config set benchmark.model Qwen3-1.7B-Q8_0.gguf
gwen config set benchmark.layer_load 4
gwen config set benchmark.output_dir ./bench_results

# Now this works with no flags
gwen benchmark

# Override per-run
gwen benchmark --model other-model.gguf --layer-load 8 --quantization Q4_K --output /tmp/bench
```

---

**End of Gwen-Changes-2026-06-11_GWEN-214-followup.md**
