# Design Document — GWEN-222: Checkpoint Resume + Adapter Export Validation

## Overview

GWEN-222 adds two orthogonal capabilities to the native LoRA training pipeline:

1. **Checkpoint resume** — restart `gwen train` from the last persisted
   `checkpoint_{step:06}.safetensors` without losing progress. LoRA adapter
   weights are restored into the `VarMap`; AdamW moment state is not persisted
   (by deliberate design — the optimizer warm-up period is acceptable).

2. **Adapter export shape validation** — a pre-write GGUF shape-check inside
   `export_adapter()` that guards against silently corrupt output files when
   an adapter was trained against a different base model.

Both are additive, backward-compatible changes. All new public types are in
`packages/core`; `packages/tui` surfaces them through two new CLI flags only.

---

## Architecture

### Component Map

```
packages/tui/src/commands/train.rs
  TrainArgs                    ← +--resume [path] flag
  ExportAdapterArgs            ← +--base-gguf <path> flag
  run_export_adapter()         ← updated dispatch to new export_adapter sig

packages/core/src/train/
  config.rs
    ResumeMode                 ← NEW enum (None | Auto | Explicit(PathBuf))
    NewTrainConfig             ← +resume_checkpoint: ResumeMode field

  checkpoint_resumer.rs        ← NEW module
    resolve_checkpoint()       ← auto-discovery + validation logic
    parse_step_from_filename() ← "checkpoint_000500.safetensors" → 500
    load_checkpoint_into_varmap() ← VarMap::load wrapper

  lora_cli.rs
    export_adapter()           ← +base_gguf_path: Option<&Path> param
    AdapterShapeValidator      ← NEW inline helper struct

  native_runner.rs
    run_native_local()         ← +checkpoint load before LayeredTrainingLoop::new

  layered_training_loop.rs
    LayeredTrainingLoop::new() ← +initial_step: usize param
    save_checkpoint() interval ← already correct; step counter fed from new param

  error.rs
    GwenError::ShapeMismatch   ← extend with adapter_key field for better messages
```

### Data-Flow: Checkpoint Resume

```
gwen train --resume [path] -m model.gguf -d data.jsonl
  │
  ▼ train.rs: TrainArgs::resume → ResumeMode
  │
  ▼ runner.rs / native_runner.rs: run_native_local()
      │
      ├─ checkpoint_resumer::resolve_checkpoint(mode, output_path)
      │     ├─ Auto  → glob output_path/checkpoint_*.safetensors, pick lex-max
      │     ├─ Explicit(p) → validate p exists
      │     └─ None  → return (None, 0)
      │
      ├─ checkpoint_resumer::load_checkpoint_into_varmap(&varmap, path)
      │     └─ varmap.load(path)  ← populates lora_a/lora_b tensors
      │
      ├─ step = checkpoint_resumer::parse_step_from_filename(path)
      │
      └─ LayeredTrainingLoop::new(config, gguf_path, batches, varmap, tx, initial_step=step)
           └─ self.global_step = initial_step
              self.next_ckpt_step = initial_step + 500
```

### Data-Flow: Adapter Shape Validation

```
gwen train export-adapter --checkpoint ckpt.st --output out.st --base-gguf base.gguf
  │
  ▼ train.rs: ExportAdapterArgs → run_export_adapter()
  │
  ▼ lora_cli::export_adapter(checkpoint, output, dry_run, base_gguf_path=Some(base))
      │
      ├─ VarMap::load(checkpoint)
      ├─ LoraExporter::extract_adapters(&varmap)   ← existing
      │
      ├─ [if base_gguf_path.is_some()]
      │     AdapterShapeValidator::validate(adapters, base_gguf_path)
      │       ├─ gguf_parser::parse(base_gguf_path)
      │       ├─ for each tensor: parse_gguf_key() → candle_key → shape_to_2d()
      │       └─ for each adapter:
      │             lora_a.shape()[1] == d_in  (adapter d_in vs GGUF d_in)
      │             lora_b.shape()[0] == d_out (adapter d_out vs GGUF d_out)
      │             → on mismatch: return Err(GwenError::ShapeMismatch{...})
      │
      └─ [only if validation passes or base_gguf_path.is_none()]
           LoraExporter::export_safetensors(&varmap, output)
```

The key invariant: `export_safetensors` is never reached if `validate` returns
an error. No `File::create` is called for `output_path` until validation
completes, so no partial file is ever written.

---

## Components and Interfaces

### `ResumeMode` (new enum in `config.rs`)

```rust
#[derive(Debug, Clone, Default)]
pub enum ResumeMode {
    #[default]
    None,
    Auto,
    Explicit(PathBuf),
}
```

`NewTrainConfig` gains:

```rust
pub struct NewTrainConfig {
    // ... existing fields ...
    pub resume_checkpoint: ResumeMode,
}

impl Default for NewTrainConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            resume_checkpoint: ResumeMode::None,
        }
    }
}
```

### `checkpoint_resumer.rs` (new module)

Three pure public functions, no struct state:

```rust
/// Resolve the checkpoint path and step count from a ResumeMode.
/// Returns (Some(path), step) or (None, 0) for ResumeMode::None.
pub fn resolve_checkpoint(
    mode: &ResumeMode,
    output_path: &Path,
) -> Result<(Option<PathBuf>, usize)>

/// Parse the six-digit step field from a checkpoint filename.
/// Returns 0 and logs a warning if the filename doesn't match the pattern.
pub fn parse_step_from_filename(path: &Path) -> usize

/// Load a SafeTensors checkpoint into an existing VarMap.
/// Calls varmap.load(path). Errors are propagated unmodified.
pub fn load_checkpoint_into_varmap(varmap: &mut VarMap, path: &Path) -> Result<()>
```

`resolve_checkpoint` logic:
- `ResumeMode::Explicit(p)`: if `p` exists, return `(Some(p), parse_step_from_filename(p))`;
  otherwise `bail!` with a descriptive message.
- `ResumeMode::Auto`: read `output_path`, filter entries matching
  `checkpoint_*.safetensors`, sort lexicographically, take last. If none
  found: log warning to stderr, return `(None, 0)`.
- `ResumeMode::None`: return `(None, 0)` immediately.

`parse_step_from_filename` logic:
- Strip directory, strip `.safetensors` extension, strip `checkpoint_` prefix.
- Parse remaining 6-digit decimal string as `usize`.
- If any step fails: log `[resume] warning: could not parse step from '{}'...` to
  stderr and return 0.

### `LayeredTrainingLoop` — `initial_step` parameter

`LayeredTrainingLoop::new()` gains an `initial_step: usize` parameter. The
existing `global_step` counter is initialized to `initial_step`. The checkpoint
interval logic (`global_step % 500 == 0`) already counts from zero; since it
now counts from `initial_step`, the first checkpoint fires at the next multiple
of 500 after `initial_step`.

`TrainResult::total_steps` is already computed as steps taken in the current
run (it's a local counter starting at 0 each run), so no change is needed there.

### `export_adapter()` — updated signature

```rust
pub fn export_adapter(
    checkpoint_path: &Path,
    output_path: &Path,
    dry_run: bool,
    base_gguf_path: Option<&Path>,  // NEW
) -> std::result::Result<usize, GwenError>
```

The `AdapterShapeValidator` is a small inline helper (not a separate module):

```rust
struct AdapterShapeValidator;

impl AdapterShapeValidator {
    fn validate(
        adapters: &[LoraAdapter],
        base_gguf_path: &Path,
    ) -> std::result::Result<(), GwenError> {
        // 1. gguf_parser::parse(base_gguf_path)
        // 2. Build a HashMap<candle_key, (d_out, d_in)> from the GGUF tensors
        //    using parse_gguf_key() + shape_to_2d()
        // 3. For each adapter: look up by layer_name in the map
        //    - check lora_a.dims()[1] == d_in
        //    - check lora_b.dims()[0] == d_out
        //    - on mismatch: return Err(GwenError::ShapeMismatch{ adapter_key, adapter, base })
        Ok(())
    }
}
```

The `GwenError::ShapeMismatch` variant is extended with an `adapter_key` field
so the error message names the projection:

```rust
#[error("shape mismatch for adapter '{adapter_key}': adapter {adapter:?} vs base {base:?}")]
ShapeMismatch {
    adapter_key: String,
    adapter: Vec<usize>,
    base: Vec<usize>,
}
```

Callers in `lora_merger.rs` that already construct `ShapeMismatch` must be
updated to supply `adapter_key: tensor.name.clone()`.

### `ExportAdapterArgs` — `--base-gguf` flag

```rust
pub struct ExportAdapterArgs {
    // ... existing fields ...
    #[arg(long, value_name = "PATH",
          help = "Base GGUF file for pre-export shape validation (optional). \
                  When provided, lora_a/lora_b dimensions are checked against \
                  the base model tensors before any output is written.")]
    pub base_gguf: Option<PathBuf>,
}
```

### `TrainArgs` — `--resume` flag

```rust
#[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "AUTO",
      help = "Resume from the last checkpoint (auto-discover) or an explicit \
              checkpoint file path. Restores LoRA adapter weights only — \
              AdamW optimizer state is NOT restored; a momentum warm-up \
              period will occur after resuming.")]
pub resume: Option<String>,
```

Mapping in `run_native_local()`:

```rust
let resume_mode = match &args.resume {
    None                              => ResumeMode::None,
    Some(s) if s == "AUTO"            => ResumeMode::Auto,
    Some(p)                           => ResumeMode::Explicit(PathBuf::from(p)),
};
```

The sentinel `"AUTO"` is the `default_missing_value` for clap's `num_args=0..=1`
pattern, which lets `--resume` (no value) and `--resume ./ckpt.st` both parse
cleanly.

---

## Data Models

### `ResumeMode`

```rust
pub enum ResumeMode {
    None,                    // default — start from step 0
    Auto,                    // discover lex-max checkpoint_*.safetensors in output_path
    Explicit(PathBuf),       // explicit file path provided by user
}
```

### `NewTrainConfig` (extended)

| Field | Type | Description |
|---|---|---|
| `resume_checkpoint` | `ResumeMode` | How to find the checkpoint at startup. Default: `None`. |

All other existing fields are unchanged.

### `GwenError::ShapeMismatch` (extended)

```rust
ShapeMismatch {
    adapter_key: String,    // e.g. "lora_layer_0_q_proj"
    adapter: Vec<usize>,    // actual adapter tensor dims, e.g. [d_out, rank] for lora_b
    base: Vec<usize>,       // expected base GGUF dims, e.g. [d_out, d_in]
}
```

The `adapter_key` field is new in GWEN-222. Existing callers in `lora_merger.rs`
supply `adapter_key: tensor.name.clone()`.

### Checkpoint file naming convention

```
{output_path}/checkpoint_{step:06}.safetensors
```

Examples:
- `./gwen-output/checkpoint_000000.safetensors`  (step 0 dry-run)
- `./gwen-output/checkpoint_000500.safetensors`  (first real checkpoint)
- `./gwen-output/checkpoint_001000.safetensors`  (second checkpoint)

Lexicographic sort of these names is equivalent to numeric sort for steps up to
999,999 because of the zero-padded six-digit format.

---

## Error Handling

| Scenario | Error | CLI behaviour |
|---|---|---|
| `--resume <path>` not found | `bail!` in `resolve_checkpoint` | stderr + exit 1 before training |
| Auto-discovery finds no files | warning log, continue from step 0 | no error |
| `VarMap::load()` fails | propagated `anyhow::Error` | stderr + exit 1 |
| Step field unparseable | warning log, step 0 used | no error |
| `--base-gguf` path not found | early `Err` in `run_export_adapter` | stderr + exit 1 |
| `--base-gguf` omitted | `None` passed to `export_adapter` | stderr WARNING: "shape validation skipped — pass --base-gguf to verify adapter dims before export" |
| Shape mismatch (`d_in` or `d_out`) | `GwenError::ShapeMismatch` | stderr with key + shapes, exit 1, no file written |
| Any other export error | propagated error | stderr + exit 1, no file written |

The "no partial file" guarantee comes from structuring `export_adapter` so that
`std::fs::File::create(output_path)` inside `export_safetensors` is never
reached when `validate` returns an error. This is enforced by the call order:
validate → then call `export_safetensors`.

---

## Implementation Sequence (Wave Structure)

### Wave 1 — Audit
- Write `audit.md` in the spec directory noting exact line numbers for each
  touch point confirmed in the source.

### Wave 2 — Checkpoint Resume
1. `config.rs`: add `ResumeMode` enum + `NewTrainConfig::resume_checkpoint`.
2. `checkpoint_resumer.rs`: new module with the three functions above.
3. `train.rs` (tui): add `--resume` to `TrainArgs`; map to `ResumeMode`; pass
   through `NewTrainConfig`.
4. `native_runner.rs`: call `resolve_checkpoint` + `load_checkpoint_into_varmap`
   before `LayeredTrainingLoop::new`.
5. `layered_training_loop.rs`: accept `initial_step: usize`; initialize
   `global_step` from it.
6. Tests: `#[cfg(test)]` in `checkpoint_resumer.rs` covering auto-discovery
   lexicographic selection and step parsing.

### Wave 3 — Export Adapter Shape Validation
1. `error.rs`: add `adapter_key: String` field to `ShapeMismatch`; fix
   existing call site in `lora_merger.rs`.
2. `lora_cli.rs`: add `base_gguf_path: Option<&Path>` parameter to
   `export_adapter`; implement `AdapterShapeValidator::validate` reusing
   `parse_gguf_key` / `shape_to_2d` from `lora_merger.rs`.
3. `train.rs` (tui): add `--base-gguf` to `ExportAdapterArgs`; update
   dispatch; improve error message format.
4. Tests: shape-match and shape-mismatch cases; no-partial-file check.

### Wave 4 — E2E Validation
- Integration test exercising the full `gwen train --resume` round-trip with a
  real micro-GGUF file.
- Integration test exercising `export-adapter --base-gguf` with a synthetic
  GGUF that has intentional dimension mismatches.

---

## GGUF Shape Reuse Details

`lora_merger.rs` already exposes (as module-private helpers) `parse_gguf_key`
and `shape_to_2d`. For Wave 3 these are promoted to `pub(crate)` so
`lora_cli.rs` can use them directly without duplication. `gguf_parser::parse`
is already `pub` in `crate::convert::gguf_parser`.

The validation loop:

```rust
let gguf = gguf_parser::parse(base_gguf_path)?;
let mut shape_map: HashMap<String, (usize, usize)> = HashMap::new();
for tensor in &gguf.tensors {
    if let Some((layer_idx, proj)) = parse_gguf_key(&tensor.name) {
        let (d_out, d_in) = shape_to_2d(&tensor.shape)?;
        let candle_key = format!("lora_layer_{}_{}_proj", layer_idx, proj);
        shape_map.insert(candle_key, (d_out, d_in));
    }
}

for adapter in adapters {
    if let Some(&(d_out, d_in)) = shape_map.get(&adapter.layer_name) {
        let a_d_in  = adapter.lora_a.dims().get(1).copied().unwrap_or(0);
        let b_d_out = adapter.lora_b.dims().get(0).copied().unwrap_or(0);
        if a_d_in != d_in || b_d_out != d_out {
            return Err(GwenError::ShapeMismatch {
                adapter_key: adapter.layer_name.clone(),
                adapter:     adapter.lora_b.dims().to_vec(),  // (d_out, rank)
                base:        vec![d_out, d_in],
            });
        }
    }
    // If no GGUF entry found for this adapter key, skip (new projection not in base).
}
```

---

## Testing Strategy

### Unit Tests (in `checkpoint_resumer.rs`)

- `test_resolve_auto_picks_lex_max`: create temp dir with three checkpoint files
  at different step counts, call `resolve_checkpoint(Auto, dir)`, assert the
  last step is returned.
- `test_resolve_auto_empty_dir`: empty temp dir, assert `Ok((None, 0))`.
- `test_parse_step_roundtrip`: for steps 0, 500, 1000, 999999, format and parse.
- `test_parse_step_nonstandard`: arbitrary filename → step 0, no panic.
- `test_explicit_path_missing`: nonexistent path → `Err`.

### Unit Tests (in `lora_cli.rs`)

- `test_shape_validation_pass`: adapter dims match mock GGUF shapes → `Ok`.
- `test_shape_validation_mismatch_d_in`: lora_a d_in wrong → `ShapeMismatch`.
- `test_shape_validation_mismatch_d_out`: lora_b d_out wrong → `ShapeMismatch`.
- `test_no_partial_file_on_mismatch`: verify `output_path` absent after mismatch.
- `test_backward_compat_no_base_gguf`: `base_gguf_path=None` skips validation.
- `test_dry_run_with_base_gguf_no_write`: `dry_run=true` + valid base GGUF → no file.

### Property Tests (via `quickcheck` in `packages/core`)

See Correctness Properties section below. Each property maps to a `quickcheck`
`#[quickcheck]` or `proptest` property. The `quickcheck` crate is already in
`[dev-dependencies]`.

### Integration Tests (Wave 4)

- `test_e2e_resume`: run one micro-training step, save checkpoint, resume from
  it, verify step counter starts at the saved step.
- `test_e2e_export_shape_validation`: write a synthetic GGUF with known dims,
  export an adapter with matching dims, then export one with mismatched dims and
  confirm the error.

---

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all
valid executions of a system — essentially, a formal statement about what the
system should do. Properties serve as the bridge between human-readable
specifications and machine-verifiable correctness guarantees.*

### Property 1: Explicit resume path round-trips through ResumeMode

*For any* non-empty path string `p`, when `--resume p` is parsed by
`TrainArgs`, the resulting `NewTrainConfig::resume_checkpoint` SHALL equal
`ResumeMode::Explicit(PathBuf::from(p))`.

**Validates: Requirements 1.3**

---

### Property 2: Non-existent explicit checkpoint path fails before training

*For any* path string that does not correspond to an existing file on disk,
calling `resolve_checkpoint(ResumeMode::Explicit(p), output_path)` SHALL
return an `Err` result, and no `VarMap::load` call or training-loop
construction SHALL have been made.

**Validates: Requirements 1.5**

---

### Property 3: Auto-discovery selects lexicographically greatest checkpoint

*For any* non-empty collection of filenames that each match the pattern
`checkpoint_*.safetensors`, the auto-discovery function SHALL return the
filename that is lexicographically greatest in that collection.

**Validates: Requirements 2.1**

---

### Property 4: Checkpoint load populates VarMap (round-trip)

*For any* set of LoRA tensor pairs `(lora_a, lora_b)` with valid shapes
`(rank, d_in)` and `(d_out, rank)`, saving them to a SafeTensors file and then
calling `load_checkpoint_into_varmap` SHALL produce a `VarMap` whose data
contains tensor values equal to the originals.

**Validates: Requirements 3.1**

---

### Property 5: Step counter parsed correctly from any valid checkpoint filename

*For any* step count `N` in `[0, 999_999]`, formatting it as
`checkpoint_{N:06}.safetensors` and passing that filename to
`parse_step_from_filename` SHALL return `N`.

**Validates: Requirements 3.3**

---

### Property 6: Checkpoint interval is relative to resumed step

*For any* resume step `R` and any incremental step count `S`, the first
checkpoint emitted after resume SHALL be at the step where
`(R + S) % 500 == 0` (i.e., the next 500-step boundary at or after `R`), and
the global step counter in the training loop SHALL equal `R + S` when that
checkpoint is saved.

**Validates: Requirements 3.5**

---

### Property 7: total_steps reflects only current-run steps

*For any* resume step `R` and any number of additional training steps `S`,
`TrainResult::total_steps` SHALL equal `S`, not `R + S`.

**Validates: Requirements 3.6**

---

### Property 8: Checkpoint files contain only LoRA adapter weights

*For any* checkpoint file produced by `save_checkpoint`, parsing its
SafeTensors header SHALL yield a key set containing only names matching
`lora_*` (adapter weights). No key matching optimizer-state patterns
(`adam`, `moment`, `exp_avg`, `exp_avg_sq`) SHALL be present.

**Validates: Requirements 4.1**

---

### Property 9: Shape validation passes iff adapter dims match base GGUF

*For any* adapter pair with shapes `lora_a: (rank, d_in)` and
`lora_b: (d_out, rank)` and any base GGUF tensor with outer dimension `D_out`
and inner dimension `D_in`:
- When `d_in == D_in` AND `d_out == D_out`, `AdapterShapeValidator::validate`
  SHALL return `Ok(())`.
- When `d_in != D_in` OR `d_out != D_out`, `AdapterShapeValidator::validate`
  SHALL return `Err(GwenError::ShapeMismatch { ... })` AND `export_adapter`
  SHALL NOT create or write to `output_path`.

**Validates: Requirements 5.2, 5.3, 5.4**

---

### Property 10: No output file written when any error occurs

*For any* call to `export_adapter` that returns an `Err` result (from any
cause — missing checkpoint, GGUF parse error, shape mismatch, extraction
failure), the file at `output_path` SHALL NOT exist after the call if it did
not exist before the call.

**Validates: Requirements 5.4, 6.4**

---

### Property 11: Shape mismatch error message contains all diagnostic fields

*For any* `GwenError::ShapeMismatch { adapter_key, adapter, base }`, the
formatted error message emitted by the `export-adapter` CLI handler SHALL
contain the `adapter_key` string, the expected `base` shape, and the actual
`adapter` shape.

**Validates: Requirements 6.1**

---

### Property 12: shape_to_2d reverses GGUF dimension ordering

*For any* pair `(d_in, d_out)` of positive integers, `shape_to_2d(&[d_in as u64, d_out as u64])`
SHALL return `Ok((d_out as usize, d_in as usize))`.

**Validates: Requirements 8.3**
