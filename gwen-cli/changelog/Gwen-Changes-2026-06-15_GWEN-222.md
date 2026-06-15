# GwenLand — GWEN-222: Checkpoint Resume + Adapter Export Shape Validation

**Date:** 2026-06-15 (WIB)
**Scope:** `train/config.rs`, new `train/checkpoint_resumer.rs`, `train/layered_training_loop.rs`, `train/native_runner.rs`, `train/runner.rs`, `train/mod.rs`, `error.rs`, `train/lora_merger.rs`, `train/lora_cli.rs`, `tui/commands/train.rs`, new `tests/gwen222_e2e.rs`, `packages/core/Cargo.toml`
**Type:** Two additive capabilities — (A) resume native LoRA training from a saved checkpoint via `--resume`; (B) optional pre-export shape validation of adapters against a base GGUF via `export-adapter --base-gguf`. Delivered in four gated waves (audit → resume → validation → E2E).
**Status:** ✅ Implementation + tests complete. All new unit/integration/property tests green. **One pre-existing latent bug in `export_adapter` was found by the Wave-4 E2E test and fixed** (see §4). Four pre-existing, unrelated test failures remain (see "Known pre-existing issues").

---

## Executive Summary

Two independent features:

1. **Checkpoint resume.** `gwen train --resume [PATH]` restores LoRA adapter
   weights from a prior checkpoint and continues training. Bare `--resume`
   auto-discovers the lexicographically-greatest `checkpoint_*.safetensors` in the
   output dir; `--resume <path>` resumes from an explicit file. The optimiser step
   counter is restored so the periodic checkpoint interval stays on the global
   step axis across resumes. **AdamW optimiser state is intentionally NOT
   persisted** — adapter weights only — so a brief momentum warm-up occurs after
   a resume. This is documented in the `--resume` help text.

2. **Export adapter shape validation.** `gwen train export-adapter --base-gguf
   <model.gguf>` checks every extracted adapter's `(d_in, d_out)` against the
   matching projection tensor in the base model **before any file is written**. A
   mismatch returns `GwenError::ShapeMismatch` (now carrying the offending
   `adapter_key`) and leaves no partial output on disk. Without `--base-gguf`,
   validation is skipped (backward-compatible) and a stderr warning is printed.

The plan was followed wave-by-wave, but the read-only audit (Wave 1) and the
E2E tests (Wave 4) surfaced several places where the spec's assumptions did not
match the codebase. Every such deviation is documented in §3 below, in the
interest of full transparency for reviewers.

---

## What Changed

### Wave 1 — Audit (no source changes)

`.kiro/specs/gwen-222-.../audit.md` records every touch-point with exact line
numbers and the three plan/codebase mismatches resolved before coding (§3, D1–D3).
Build gate: `cargo build --workspace` clean.

### Wave 2 — Checkpoint resume

- **`train/config.rs`** — new `ResumeMode` enum (`None` `#[default]` / `Auto` /
  `Explicit(PathBuf)`); new `resume_checkpoint: ResumeMode` field on
  `NewTrainConfig` (+ its `Default`).
- **`train/checkpoint_resumer.rs`** (new) — three public functions:
  - `resolve_checkpoint(&ResumeMode, &Path) -> Result<(Option<PathBuf>, usize)>` —
    `None`→`(None,0)`; `Auto`→newest checkpoint (lexicographic max), or a warning
    + `(None,0)` on an empty/missing dir; `Explicit`→the path or `bail!`.
  - `parse_step_from_filename(&Path) -> usize` — tolerant parse of
    `checkpoint_{NNNNNN}.safetensors`; non-standard names warn + return 0.
  - `load_checkpoint_into_varmap(&mut VarMap, &Path)` — wraps `VarMap::load`.
- **`train/layered_training_loop.rs`** — new `global_step` field; `new()` gains an
  `initial_step: usize` param; new `pub fn load_checkpoint(&mut self, &Path)`. In
  `run()`, `optimizer_steps` is now seeded from `global_step` (keeps the `% 500`
  checkpoint interval + filenames on the global axis), while a separate
  `steps_this_run` counter is what `TrainResult::total_steps` reports.
- **`train/native_runner.rs`** — `run_native_local` resolves the checkpoint, passes
  `initial_step` to `new()`, then calls `load_checkpoint` **after** construction
  (see §3, D4).
- **`train/runner.rs`** — `run_train_with_opts` and `train_config_to_native` gain a
  `resume: ResumeMode` parameter (mirroring the existing `gdtqp` threading); the
  `run_train` convenience wrapper passes `ResumeMode::None`.
- **`tui/commands/train.rs`** — `--resume` flag (clap
  `num_args=0..=1, default_missing_value="AUTO"`); `resume_mode_from_args` helper;
  wired into both `run_train_with_opts` call sites.

### Wave 3 — Export adapter shape validation

- **`error.rs`** — `GwenError::ShapeMismatch` gains `adapter_key: String`; the
  `#[error]` format now names the adapter. The single existing construction site
  in `lora_merger.rs` was updated (`adapter_key: tensor.name.clone()`).
- **`train/lora_merger.rs`** — `parse_gguf_key` promoted to `pub(crate)`.
- **`train/layered_training_loop.rs`** — `shape_to_2d` promoted to `pub(crate)`
  (decision locked in Wave 1: promote, do not duplicate — single source of truth
  for GGUF dimension-reversal).
- **`train/lora_cli.rs`** — `export_adapter` gains a 4th param
  `base_gguf_path: Option<&Path>`; new private `AdapterShapeValidator` reads the
  base GGUF descriptors (via `parse_header`, see §3 D5), builds
  `candle_layer_name → (d_out, d_in)`, and checks each adapter's
  `lora_a.dims()[1]`/`lora_b.dims()[0]`. The validation hook runs **before** the
  dry-run return and **before** any write.
- **`tui/commands/train.rs`** — `--base-gguf` flag on `ExportAdapterArgs`;
  `run_export_adapter` fails fast on a non-existent path, warns when validation is
  skipped, and prints a diagnostic + `exit(1)` on `ShapeMismatch`. The
  `--auto-merge` export call passes `None` (validation not needed there).

### Wave 4 — E2E + property tests

- **`tests/gwen222_e2e.rs`** (new, `required-features = ["test-utils"]`) —
  `test_e2e_resume` (real save→discover→load→resume round-trip),
  `test_e2e_resume_auto_no_checkpoints` (fresh-start edge case),
  `test_e2e_export_shape_validation` (match exports; mismatch errors + writes no
  file), `prop_checkpoint_varmap_roundtrip`.
- `prop_shape_to_2d_reversal` (quickcheck) added as a **unit** test inside
  `layered_training_loop.rs` because `shape_to_2d` is `pub(crate)` and not
  reachable from an integration crate.

---

## 3. Deviations from the spec (full disclosure)

The tasks plan was largely accurate, but the following points differed from the
codebase and were resolved deliberately. None expand the feature surface; they
correct *where*/*how* the spec's steps land.

- **D1 — resume wiring path.** The plan mapped `--resume` in
  `train.rs::run_native_path()`. That function calls `native_runner::run_native`
  (the **old HF `TrainingLoop`**, synthetic base weight), **not**
  `LayeredTrainingLoop`. The actual local-GGUF training path is
  `runner.rs::run_train_with_opts → train_config_to_native → run_native_local`.
  Resume is threaded there instead, mirroring the existing `gdtqp: bool` param.
- **D2 — wrong config struct.** The plan also set `resume_checkpoint` in
  `build_train_config_from_args()`, which returns `TrainConfig` (the YAML struct)
  — it has no such field. The field lives on `NewTrainConfig`, populated in
  `train_config_to_native`.
- **D3 — extra compile-forced edit.** `NewTrainConfig` is also built as a full
  struct literal in `train_config_to_native`; the new field had to be added there
  too (the plan only mentioned `impl Default`).
- **D4 — checkpoint load ordering.** The plan loaded the checkpoint into the
  VarMap **before** `LayeredTrainingLoop::new()`. candle's `VarMap::load` only
  refreshes Vars *already present* in the map, and the adapter Vars are created
  **inside** `new()`. Loading into an empty map is a no-op, so resume now resolves
  → constructs → **then** loads via the new `load_checkpoint` method.
- **D5 — `parse_header` not `parse`.** The validator uses
  `gguf_parser::parse_header` (tensor descriptors only) instead of `parse` (which
  eagerly loads every tensor payload — multiple GB for a real base model) since
  only names + shapes are needed.
- **D6 — validator key construction.** The plan said "key by candle layer name";
  in practice `parse_gguf_key` returns `(layer_idx, proj)`, and
  `format!("lora_layer_{idx}_{proj}_proj")` exactly matches
  `LoraAdapter.layer_name` produced by `extract_adapters` — verified against that
  code. Unmatched adapters are skipped (the merge step skips them too), not
  treated as errors.
- **D7 — skip-warning location.** The "shape validation skipped" warning lives
  only in `tui::run_export_adapter`, not in core `export_adapter`. Emitting it in
  core would spam the `--auto-merge` path (which always passes `None`
  deliberately). The corresponding optional unit test asserts the file-written /
  validation-skipped behaviour rather than the warning string.

---

## 4. ⚠️ Pre-existing bug found and fixed: `export_adapter` extracted zero adapters

**This is the most important item for reviewers.** The Wave-4 E2E test
`test_e2e_export_shape_validation` initially failed at `assert_eq!(n, 1)` — a real
exported checkpoint yielded **0** adapter pairs.

Root cause: `lora_cli::export_adapter` did

```rust
let mut varmap = VarMap::new();   // empty
varmap.load(checkpoint_path)?;    // <-- no-op
let adapters = exporter.extract_adapters(&varmap)?;  // -> 0 adapters
```

candle-nn's `VarMap::load` (verified in `candle-nn-0.10.2/src/var_map.rs:42`) only
refreshes Vars **already present** in the map: *"values for variables that are
currently not in the map are not kept."* Loading into a fresh, empty VarMap does
nothing, so `extract_adapters` saw no tensors. The function returned `Ok(0)` and
wrote an empty adapter file — meaning `export-adapter` (and the `--auto-merge`
export step) had been silently producing **empty adapters**, and the new shape
validation would have been **inert** (validating an empty list).

This bug **predates GWEN-222** — the empty-VarMap+load pattern was already in
`export_adapter`; Wave 3 only added the `base_gguf_path` parameter and the
validator on top of it. It had not been caught because no test exercised
`export_adapter` end-to-end with a real saved checkpoint (the existing
`lora_bridge` tests populate a VarMap directly and never round-trip through
`VarMap::load`).

**Fix** (`train/lora_cli.rs`): read every tensor explicitly and insert each as a
`Var` so `extract_adapters` sees the real adapter pairs:

```rust
let tensors = candle_core::safetensors::load(checkpoint_path, &Device::Cpu)?;
let varmap = VarMap::new();
{
    let mut data = varmap.data().lock().unwrap();
    for (name, tensor) in tensors {
        data.insert(name, Var::from_tensor(&tensor)?);
    }
}
```

Note the **Wave-2 resume path was already correct** and is unaffected: there the
checkpoint is loaded *after* `LayeredTrainingLoop::new()` has created the adapter
Vars, so `VarMap::load` finds existing Vars to refresh (`test_e2e_resume`
confirms this). The two paths differ precisely on whether the Vars exist before
the load — which is exactly the candle semantic at issue.

---

## Validation

`cargo build --workspace` → clean (pre-existing warnings only).

`cargo test -p gwenland-core --features test-utils --no-fail-fast`:

| Target | Result |
|--------|--------|
| core lib unit tests | **265 passed**, 3 failed (pre-existing selector — see below) |
| `tests/gwen216_integration` | 3 passed |
| `tests/gwen219_dryrun` | 1 passed |
| `tests/gwen220_wave4` | 1 passed |
| `tests/gwen222_e2e` | **4 passed** |
| `bench_layer_loader` unit | 5 passed |
| tui (`gwenland` bin) | **2 passed** (`test_resume_mode_mapping`, `test_export_adapter_base_gguf_missing_path`) |
| doctests | 1 failed (pre-existing dequant — see below) |

New tests added by GWEN-222 (all green): 6 in `checkpoint_resumer`, 2 in
`layered_training_loop` (`test_total_steps_is_current_run_only`,
`test_checkpoint_keys_lora_only`) + `prop_shape_to_2d_reversal`, 5 in `lora_cli`
(validator pass / d_in / d_out mismatch / unmatched-skip / error-format), 2 in
tui `train`, and 4 in `tests/gwen222_e2e`.

---

## Known pre-existing issues (NOT introduced here)

Verified pre-existing — `cargo build --workspace` is clean and these reproduce
without any GWEN-222 change:

- **`engine::inference::selector::tests::{tilde_expand, relative_gguf_ok,
  empty_stop_sequences_ok}`** fail under a default `cargo test`: they assert
  `select_backend(..)` is `Ok`, but `default = []` compiles in no inference
  backend, so `resolve_backend("auto")` returns an error. They pass with
  `--features candle-backend`. `selector.rs` is untouched by GWEN-222; confirmed
  pre-existing by stashing all changes and re-running against the base commit.
  Same item flagged in the GWEN-219 and GWEN-220 entries. A one-line
  `#[cfg(feature = "candle-backend")]`-style guard (or accepting
  `BackendNotAvailable`) on each would fix them, as a sibling test already does.
- **Doctest `convert::dequant::dequant_q6_k_standard (dequant.rs:800)`** fails:
  the `///` doc comment contains an indented pseudocode block that rustdoc parses
  as a Rust code fence and tries to compile (`unknown start of token: \u{2014}`).
  `dequant.rs` has **0 git diff vs HEAD** — committed, pre-existing, surfaced only
  because the Wave-4 run used `--no-fail-fast` (earlier runs aborted at the unit
  failures before reaching doctests). Fix is a one-liner: fence the block as
  ` ```text `.

Neither is in GWEN-222's scope; both are left for a separate cleanup so this
change stays focused. Per-project policy they can be made green on request.

---

**End of Gwen-Changes-2026-06-15_GWEN-222.md**
