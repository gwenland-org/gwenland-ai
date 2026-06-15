# GWEN-222 — Wave 1 Audit (read-only source survey)

**Date:** 2026-06-15
**Branch:** feature/gwen-217-loss-fix
**Scope:** Inventory every touch-point for (A) checkpoint resume and (B) export-adapter
shape validation. **No source changed in Wave 1.** Line numbers are as-of this commit and
will shift once edits begin.

---

## 0. Executive summary — deltas vs. the tasks.md plan

The plan is broadly accurate, but the audit surfaced **three corrections** that must be honored
in Wave 2/3. None block the work; they redirect *where* edits land.

| # | Plan assumption | Reality | Correct action |
|---|-----------------|---------|----------------|
| **D1** | Task 2.4: map `--resume` → `ResumeMode` in `train.rs::run_native_path()` | `run_native_path` builds a `NewTrainConfig` but calls `native_runner::run_native` — the **old HF `TrainingLoop`** (synthetic base weight), **not** `LayeredTrainingLoop`. It is *not* the checkpoint-resume target. | Thread resume through **`runner.rs::run_train_with_opts` → `train_config_to_native`**, mirroring the existing `gdtqp: bool` parameter. |
| **D2** | Task 2.4: also set `config.resume_checkpoint` in `build_train_config_from_args()` | `build_train_config_from_args` (train.rs:560) returns **`TrainConfig`** (the YAML struct), which has no `resume_checkpoint` field. | Set `resume_checkpoint` only on **`NewTrainConfig`**, inside `train_config_to_native` (runner.rs:578). |
| **D3** | Task 2.1: only `impl Default` needs the new field | `NewTrainConfig` is **also** built as a full struct literal in `train_config_to_native` (runner.rs:583). Adding a field there is **compile-forced**. | Add `resume_checkpoint` to *both* `impl Default` (config.rs) **and** the `train_config_to_native` literal (runner.rs). |

**Net effect on the wiring chain (checkpoint resume):**

```
tui TrainArgs.--resume (Option<String>)
  └─ map to ResumeMode in train.rs
       └─ run_train_with_opts(config, …, resume: ResumeMode)        [runner.rs:83 — NEW param]
            └─ train_config_to_native(cfg, dry_run, gdtqp, resume)  [runner.rs:578 — NEW param]
                 └─ NewTrainConfig { …, resume_checkpoint: resume } [runner.rs:583 — NEW field]
                      └─ run_native_local(&native_cfg, …)            [native_runner.rs:44]
                           ├─ resolve_checkpoint(&config.resume_checkpoint, &output)  [NEW]
                           ├─ load_checkpoint_into_varmap(&mut varmap, path)          [NEW]
                           └─ LayeredTrainingLoop::new(…, initial_step)               [native_runner.rs:91]
```

`run_native_path` / `run_native` (the HF download path) stay **untouched** — out of scope for resume.

---

## 1. Task 1.2 decisions (locked)

- **`shape_to_2d` location:** CONFIRMED in `layered_training_loop.rs:956` as a **private `fn`**.
  It is **not** in `lora_merger.rs`. Signature: `fn shape_to_2d(shape: &[u64]) -> Result<(usize, usize)>`,
  returns `(d_out, d_in)` (reverses GGUF `[d_in, d_out]` ordering). Matches Property 12 in task 4.3.
- **Promotion strategy (move vs. duplicate):** **PROMOTE to `pub(crate)` in place** (do not duplicate).
  Rationale: GGUF dimension-reversal is a subtle correctness point; a single source of truth prevents
  drift between the trainer and the validator. `lora_cli.rs` will call
  `crate::train::layered_training_loop::shape_to_2d(...)`.
- **`parse_gguf_key`:** CONFIRMED `lora_merger.rs:89`, private `fn parse_gguf_key(key: &str) -> Option<(usize, &str)>`
  → returns `(layer_idx, proj)` only (no shape). Must become **`pub(crate)`**.

---

## 2. Wave 2 — Checkpoint Resume — touch-point table

| file | symbol | current line | change required |
|------|--------|--------------|-----------------|
| `core/src/train/config.rs` | `ResumeMode` enum | (new) | Add enum `None`(`#[default]`)/`Auto`/`Explicit(PathBuf)`, derive `Debug,Clone,Default`. Place near top (after `LoraConfig`, ~line 28). |
| `core/src/train/config.rs` | `NewTrainConfig` struct body | ends L69 (`gdtqp: bool,`), close L70 | Insert `pub resume_checkpoint: ResumeMode,` after L69. |
| `core/src/train/config.rs` | `impl Default for NewTrainConfig` | L72–90; last field `gdtqp: false,` L87 | Insert `resume_checkpoint: ResumeMode::None,` after L87. |
| `core/src/train/config.rs` | `From<Cli>` | L106–122 | **No change** — uses `default()` + assignment; `Cli` stub has no resume field. |
| `core/src/train/mod.rs` | module list | L6–24 | Add `pub mod checkpoint_resumer;`. |
| `core/src/train/checkpoint_resumer.rs` | whole module | (new file) | `resolve_checkpoint`, `parse_step_from_filename`, `load_checkpoint_into_varmap` per task 2.2. |
| `core/src/train/runner.rs` | `run_train_with_opts` sig | L83–90 (`gdtqp: bool` L89) | **(D1)** Add `resume: ResumeMode` param after `gdtqp`. |
| `core/src/train/runner.rs` | `train_config_to_native` sig | L578–582 (`gdtqp: bool` L581) | **(D2)** Add `resume: ResumeMode` param. |
| `core/src/train/runner.rs` | `train_config_to_native` literal | L583–606 (`gdtqp,` L605) | **(D3)** Add `resume_checkpoint: resume,` — compile-forced. |
| `core/src/train/runner.rs` | `train_config_to_native` call (dry-run) | L131 | Pass resume. Decision: dry-run is a 1-step memory probe — recommend passing `ResumeMode::None` here so the probe never loads a checkpoint. |
| `core/src/train/runner.rs` | `train_config_to_native` call (real) | L143 | Pass the real resume mode. |
| `core/src/train/native_runner.rs` | `let varmap = VarMap::new();` | L87 (`run_native_local`) | Change to `let mut varmap`; after it, `resolve_checkpoint(&config.resume_checkpoint, &config.output_path)` + conditional `load_checkpoint_into_varmap(&mut varmap, path)`. |
| `core/src/train/native_runner.rs` | `LayeredTrainingLoop::new(...)` call | L91–93 | Add `initial_step` argument (the `usize` from `resolve_checkpoint`). **Only call site.** |
| `core/src/train/native_runner.rs` | `run_native` / its `VarMap::new()` | L133 / L193 | **No change** — HF path, old `TrainingLoop`, not a resume target. |
| `core/src/train/layered_training_loop.rs` | `struct LayeredTrainingLoop` | L106–129 | Add `global_step: usize` field. |
| `core/src/train/layered_training_loop.rs` | `fn new(...)` sig | L144–150 | Add `initial_step: usize` param. |
| `core/src/train/layered_training_loop.rs` | `Ok(Self { ... })` ctor | L307–321 | Add `global_step: initial_step,`. **NB:** the *other* `Ok(Self {` at L751 is a different type — do not touch. |
| `core/src/train/layered_training_loop.rs` | `let mut optimizer_steps: usize = 0;` | L348 | Change `0` → `self.global_step;`. Add separate `let mut steps_this_run: usize = 0;`. |
| `core/src/train/layered_training_loop.rs` | optimizer step increment | L383 (`optimizer_steps += 1;`) | Also `steps_this_run += 1;`. |
| `core/src/train/layered_training_loop.rs` | checkpoint save guard | L388–389 (`% 500`) | No change — already relative to `global_step` once `optimizer_steps` starts there. ✅ Property 6. |
| `core/src/train/layered_training_loop.rs` | `TrainResult { total_steps: optimizer_steps }` | L448 | Change to `total_steps: steps_this_run`. ⚠ The `done` JSON at L438 also emits `optimizer_steps` (cumulative) — decide whether to keep cumulative in JSON or switch to `steps_this_run` for consistency. |
| `tui/src/commands/train.rs` | imports | L24 | Add `ResumeMode` to the `gwenland_core::train::config` import. |
| `tui/src/commands/train.rs` | `TrainArgs` struct | L39–102 (close L102) | Add `--resume` flag (`Option<String>`, `num_args=0..=1`, `default_missing_value="AUTO"`) before L102. |
| `tui/src/commands/train.rs` | `run_train_with_opts` call (local-GGUF flags) | L210–213 | Add resume arg (map `args.resume`). |
| `tui/src/commands/train.rs` | `run_train_with_opts` call (`--config` path) | L371 | Add resume arg (map `args.resume`). |
| `tui/src/commands/train.rs` | `run_native_path` / `build_train_config_from_args` | L391 / L560 | **No change for resume** — see D1/D2. |

### Resume-flag → ResumeMode mapping (shared helper)
Map once in `train.rs` (or a small `fn`), reused at both `run_train_with_opts` call sites:
```rust
let resume_mode = match &args.resume {
    None                   => ResumeMode::None,
    Some(s) if s == "AUTO" => ResumeMode::Auto,
    Some(p)                => ResumeMode::Explicit(PathBuf::from(p)),
};
```

---

## 3. Wave 3 — Export Adapter Shape Validation — touch-point table

| file | symbol | current line | change required |
|------|--------|--------------|-----------------|
| `core/src/error.rs` | `ShapeMismatch` variant + `#[error]` | L59–60 | Add `adapter_key: String` field; update format string to `"shape mismatch for adapter '{adapter_key}': adapter {adapter:?} vs base {base:?}"`. |
| `core/src/train/lora_merger.rs` | `ShapeMismatch { … }` construction | L440–443 | **Only construction site.** Add `adapter_key: tensor.name.clone(),` (`tensor.name` confirmed in scope — used at L407). |
| `core/src/train/lora_merger.rs` | `matches!(…, ShapeMismatch { .. })` (test) | L1420 | **No change** — uses `{ .. }` pattern. |
| `core/src/train/lora_merger.rs` | `fn parse_gguf_key` | L89 | `fn` → `pub(crate) fn`. |
| `core/src/train/layered_training_loop.rs` | `fn shape_to_2d` | L956 | `fn` → `pub(crate) fn` (locked decision §1). |
| `core/src/train/lora_cli.rs` | `export_adapter` sig | L32–36 | Add 4th param `base_gguf_path: Option<&Path>`. |
| `core/src/train/lora_cli.rs` | validation hook | after `extract_adapters` L59–61, before `dry_run` return L65 | Insert `if let Some(base)=base_gguf_path { AdapterShapeValidator::validate(&adapters, base)?; }`. Placing it **before** the L65 dry-run return guarantees no write on mismatch in any mode (satisfies Property 10 + dry-run test). |
| `core/src/train/lora_cli.rs` | `AdapterShapeValidator` | (new) | Private struct + `fn validate(&[LoraAdapter], &Path) -> Result<(), GwenError>`. Needs imports: `std::collections::HashMap`, `crate::convert::gguf_parser`, `parse_gguf_key`, `shape_to_2d`, and the `LoraAdapter` type (from `lora_bridge`). |
| `tui/src/commands/train.rs` | `ExportAdapterArgs` struct | L122–134 (close L134) | Add `--base-gguf` flag (`Option<PathBuf>`) before L134. |
| `tui/src/commands/train.rs` | `run_export_adapter` | L251 | Add `--base-gguf` existence pre-check; print "shape validation skipped" warning when `None`; on `ShapeMismatch` print diagnostic + `exit(1)`. |
| `tui/src/commands/train.rs` | `export_adapter` call (export path) | L262 | Add 4th arg `args.base_gguf.as_deref()`. |
| `tui/src/commands/train.rs` | `export_adapter` call (`--auto-merge`) | L483 | Add 4th arg `None`. |

### gguf_parser API (for `AdapterShapeValidator`)
- `convert::gguf_parser::parse(path: &Path) -> Result<GgufFile, String>` (L163).
- `GgufFile` (L101) holds tensors; `TensorInfo` (L80) has `pub name: String` (L82) + `pub shape: Vec<u64>` (L84).
- Validator builds `HashMap<candle_key, (d_out, d_in)>` via `parse_gguf_key(&t.name)` + `shape_to_2d(&t.shape)`,
  then checks `adapter.lora_a.dims()[1] == d_in` and `adapter.lora_b.dims()[0] == d_out`.
- ⚠ Open detail for Wave 3: the backward-compat "shape validation skipped" warning is specified in
  **two** places (task 3.4 expects it from `lora_cli`; task 3.5 emits it from `train.rs`). Pick one
  emission point to avoid a double warning — recommend `train.rs` (task 3.5) as the user-facing layer,
  and have the `lora_cli` test assert on the stderr it controls.

---

## 4. Existing-behavior facts worth recording

- **Checkpoint filename format** already matches the resume parser: `save_checkpoint` writes
  `format!("checkpoint_{:06}.safetensors", step)` (layered_training_loop.rs:968) → 6-digit zero-pad,
  so `parse_step_from_filename` must strip `checkpoint_` + `.safetensors` and parse, tolerating non-pad widths.
- **`save_checkpoint` writes the entire VarMap** (`varmap.save`, L976). Property 8 (keys are LoRA-only)
  depends on the VarMap containing *only* `lora_*` vars + frozen model tensors. The frozen
  `model_embedding`/`output_norm`/`lm_head` are loaded as plain `Tensor`s (not `Var`s) and are **not**
  in the VarMap, so checkpoints should already be adapter-only — confirm in the Wave 2 test, do not assume.
- **AdamW optimizer state is not persisted** anywhere — resume restores adapter weights only, matching the
  `--resume` help text's momentum-warmup caveat. No optimizer-state code exists to extend.
- **gdtqp precedent** is the exact template for threading `resume` through `runner.rs` (separate typed param,
  not folded into `TrainConfig`).

---

## 5. Wave 1 gate

- Task 1.3: `cargo build --workspace` must be clean (no source changed in Wave 1). Run and record result below.

> Build gate result: ✅ **PASS** (2026-06-15). `cargo build --workspace` finished the `dev`
> profile in ~2m36s with **0 errors**; only pre-existing warnings (unused `label` in scan.rs,
> unused `train_result` in train.rs:421, dead-code in ui.rs/chat_pane.rs). No source changed in Wave 1.
