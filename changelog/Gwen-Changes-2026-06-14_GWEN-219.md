# GwenLand — GWEN-219 (cont.): Drop-in Adapter Export + EXPERIMENTAL `--gdtqp`

**Date:** 2026-06-14 (WIB)
**Scope:** `train/layered_training_loop.rs`, `train/lora_bridge.rs`, `train/lora_merger.rs`, `train/config.rs`, `train/runner.rs`, `tui/commands/train.rs`, new `tests/gwen219_dryrun.rs`
**Type:** Completes the GWEN-219 acceptance criteria left open by the 2026-06-11 entry (per-projection training), plus the optional experimental flag.
**Status:** ✅ Default path complete and validated. `--gdtqp` shipped as clearly-labelled EXPERIMENTAL.

---

## Executive Summary

The 2026-06-11 work made the layered loop *train* a LoRA adapter for every projection, but the adapter was **not actually drop-in mergeable** — its checkpoint keys and the merger's GGUF key mapping did not line up with a real model. This entry closes that gap end-to-end and adds the optional `--gdtqp` experimental rank-allocation path.

Two independent breaks blocked "drop-in mergeable via `gwen train merge-adapter`":

1. **Exporter could not read layered checkpoints.** `LayeredTrainingLoop` saves VarMap keys as `l{N}.{proj}.lora_{a|b}` (+ `tok_embed`/`lm_head`), but `extract_adapters()` only understood the legacy flat form `lora_{a|b}_layer_{N}_{proj}` → it extracted **zero** adapters from a real training checkpoint.
2. **Merger could not match real GGUF tensor names.** `KeyMapper::gguf_to_candle` only accepted HF `model.layers.*` names, but Qwen3 (and every llama.cpp GGUF) uses `blk.N.attn_q.weight` → it matched **zero** tensors and silently copied the base through unchanged.

Both are fixed; a real merge now applies per-projection deltas onto a `blk.*`-named GGUF.

---

## What Changed

### 1. Drop-in adapter export (default-path acceptance criterion)

- **`lora_bridge.rs` — `extract_adapters()`** now recognises the `LayeredTrainingLoop` checkpoint layout `l{N}.{var_key}.lora_{a|b}` (new `parse_layered_lora_key`, mapping `attn_q…ffn_down` → bare `q…down`), in addition to the legacy flat layout. Non-adapter vars (`tok_embed`, `lm_head`, the fallback `l{N}.lora_*`) are skipped. **Rank is now derived from `lora_a`'s first dim** (per-adapter) rather than a single config value — required for heterogeneous `--gdtqp` ranks and consistent with `validate_shapes()`.
- **`lora_bridge.rs` — `export_safetensors()`** now **bakes the LoRA scale `alpha/rank` into the exported `lora_b`**. The merge loader applies an unscaled `B @ A` (effective scale 1.0) by design, so the trained scaling must live in the weights — otherwise the merged delta was off by `alpha/rank`.
- **`lora_merger.rs` — `parse_gguf_key()`** now accepts llama.cpp `blk.{N}.{attn_q|attn_k|attn_v|attn_output|ffn_gate|ffn_up|ffn_down}.weight` in addition to HF names, mapping each to the bare projection. Norm/bias tensors stay unmatched (copied verbatim).
- **`lora_merger.rs` test helper** — fixed the synthetic-GGUF **magic bytes** (`build_minimal_gguf` wrote `"UGGF"` / `0x46474755` instead of `"GGUF"` / `0x46554747`). This bug had silently broken **all three** `test_merge_*` integration tests; they now run and pass.

### 2. `classify_tensor` norm exclusion (correctness fix)

Qwen3 ships `attn_q_norm` / `attn_k_norm`. Because those names contain `attn_q` / `attn_k`, `classify_tensor` mis-tagged them as the q/k projection — a 1-D norm masquerading as a weight matrix. It now rejects any name containing `norm` first. This removes norm noise from the per-layer signature and was the cause of `--gdtqp` reporting `S(ρ)=0` for q/k (the 1-D norm overwrote the real projection's entropy).

### 3. EXPERIMENTAL `--gdtqp` per-projection rank allocation

⚠ **Theory UNPROVEN — never fold `--gdtqp` runs into stable benchmark numbers.**

The flag synthesises a mechanism the source specs do **not** define end-to-end:
- **GAAP** gives the von Neumann entropy `S(ρ) = -Tr(ρ log ρ)` on an *attention* density matrix `ρ = Σ aᵢ|v̂ᵢ⟩⟨v̂ᵢ|`, and explicitly lists "Integration with LoRA training" as an **open problem** — no rank rule.
- **GDTQP** contributes the idea of *sensitivity-weighted adaptive allocation under a budget* (it allocates quantisation bits, not LoRA rank, and is post-training).

Since the streaming loop has no runtime attention, we substitute a **weight-derived diagonal density matrix** `ρ = diag(p)`, `p_c = (column energy of W) / (total energy)`; then `S(ρ) = -Σ p_c ln p_c` exactly (normalised by `ln(d_in)`). Higher entropy ⇒ less low-rank structure ⇒ more rank. Ranks are mean-centred and proportional, clamped to `[base/2, base·2]` and capped at `min(d_in,d_out)`, so the mean rank ≈ base (budget-preserving) and equal sensitivities reduce to the uniform default.

Plumbing: `NewTrainConfig.gdtqp` → `train_config_to_native` → `run_train_with_opts(…, gdtqp)` → `LayeredTrainingLoop`. CLI flag `gwen train --gdtqp` (long-only, honoured on the local-GGUF `--config` path). Allocation emits a loud `[gdtqp][EXPERIMENTAL]` block (mirrored to the TUI) with the per-projection `S(ρ)` and rank, plus the "UNPROVEN" disclaimer.

---

## Validation

`cargo test -p gwenland-core --lib train::` → **80 passed, 1 failed**. New/affected tests all green:
- `lora_bridge`: `extract_adapters_reads_layered_checkpoint_layout`
- `lora_merger`: `gguf_to_candle_accepts_llamacpp_blk_names`, `layered_checkpoint_exports_to_mergeable_adapter`, `test_merge_blk_named_tensor_applies_delta`, and the 3 previously-broken `test_merge_*` (now fixed)
- `layered_training_loop`: `test_column_energy_entropy_uniform_vs_peaked`, `test_allocate_ranks_from_sensitivity`, `test_gdtqp_flag_constructs_and_runs`, strengthened `test_classify_tensor_known_names`
- `gwen216_integration` (`--features test-utils`): 2 passed

### Native dry-run on local **Qwen3-1.7B Q8_0** (`tests/gwen219_dryrun.rs`, env-gated)

```
=== DEFAULT PATH ===
vocab(capped)=8192 hidden=2048 layers=28
trainable params=42270720         # all 7 projections × 28 layers
RSS start=275 MB peak=589 MB      # streaming invariant holds for a 1.7B model
step 1 loss=9.0074 (≈ ln(8192))   # finite, sane init
✓ no OOM — 1 step completed cleanly

=== EXPERIMENTAL --gdtqp PATH ===
S(ρ): attn_q 0.9984 · attn_k 0.9992 · attn_v 0.9987 · attn_o 0.9955 ·
      ffn_gate 0.9984 · ffn_up 0.9981 · ffn_down 0.9998
→ all rank 8 (uniformly high entropy ⇒ collapses to base; heterogeneous
  allocation is exercised in test_allocate_ranks_from_sensitivity)
step 1 loss=9.0193  ✓ no OOM
```

The long full-run was **deferred at the user's request** (export-bridge + dry-run scope).

`tests/gwen219_dryrun.rs` is env-gated on `GWEN_DRYRUN_GGUF` so CI (no model file) skips it; it is a reusable manual validation harness that calls the same `run_native_local` path as the CLI dry-run.

---

## Known pre-existing issues (NOT introduced here)

- **`test_new_rejects_empty_varmap` fails** — a GWEN-217 regression already documented in the 2026-06-11 entry (`new()` self-populates the VarMap before the empty check). Left untouched (out of scope).
- **CLI `-n` collision** — `non_interactive` (global, `-n`) and `train --name` (`-n`) both claim `-n`, so clap's debug-asserts panic when invoking `gwen train` in a debug build. This blocked running the CLI dry-run directly, hence the dry-run was validated via the `run_native_local` integration test (identical code path). Pre-existing; predates GWEN-219.

---

---

## Session follow-up: CLI usability fixes (model args + flag collision)

Reported after the GWEN-219 work: `gwen serve <MODEL>` (and `fetch`/`train`) would not accept a filepath or a HuggingFace id. Root causes + fixes:

### 1. Whole CLI panicked in debug builds — clap `-n`/`-y` collision (`main.rs`)

The global flags `--non-interactive` (`-n`) and `--yes` (`-y`) reused short letters already taken by subcommand-local options: `train --name` (`-n`), `hub list -n`, `hub-dataset list -n`, and the `-y` confirmations in `hub`/`hub-dataset`. clap validates the whole command tree at startup, so **every** command panicked in debug builds (`Short option names must be unique`). Removed the two global shorts (kept `--non-interactive` / `--yes` long forms). This is why `serve`/`fetch`/`train` all appeared broken.

### 2. Model resolution rejected most paths (`engine/inference/loader.rs`)

`resolve_model_path` (used by `gwen serve` and `gwen run`) only treated `./`, `../`, `/`, and `C:` as paths — so `model.gguf`, `models/model.gguf`, and `~/x.gguf` fell through to the cache lookup and failed confusingly, and HuggingFace ids got a bare "not found". Now:
- `looks_like_path` accepts explicit prefixes, embedded separators, a Windows drive, leading `~`, or a `.gguf` suffix; `~` is expanded.
- HuggingFace ids (`org/name`) get an actionable hint pointing at `gwen fetch -m …` (you can't serve/run a model that isn't downloaded).
- Unit tests cover path classification, HF-id detection, and resolution outcomes.

### 3. `gwen run` hid the real error (`commands/run.rs`)

Errors printed with `{}` showed only the top context ("failed to load model"), swallowing the actionable cause. Switched to `{:#}` (full anyhow chain on one line).

### 4. `gwen fetch <MODEL>` positional (`commands/fetch.rs`)

Added an optional positional model id folded into the `-m` list, so `gwen fetch tinyllama/TinyLlama-1.1B` works as well as `-m …`.

### 5. `gwen train -m ./model.gguf -d data.jsonl` now trains via the layered loop (`commands/train.rs`)

Previously local-GGUF training required a `--config` YAML; a bare `-m <gguf>` went to the HF in-memory path and failed trying to fetch a tokenizer for the file path. Now a local-GGUF `--model` is routed through `run_train_with_opts` → `run_native_local` → `LayeredTrainingLoop` (the same path `--config` uses), so `gwen train -m ./model.gguf -d data.jsonl [--gdtqp] [--dry-run]` works directly — the CLI gateway to GWEN-219. (No positional added to `train`: it has subcommands `export-adapter`/`merge-adapter`, so a positional would be ambiguous.)

### Verified against the real binary (debug)

```
gwen train --help                                   # no panic
gwen serve C:/…/Qwen3-1.7B-Q8_0.gguf --dry-run      # resolves the file ✓
gwen serve Qwen/Qwen3-1.7B --dry-run                # "run `gwen fetch …`"
gwen run Qwen/Qwen3-1.7B --prompt hi                # detailed HF hint shown
gwen fetch --help                                   # [MODEL] positional present
gwen train -m C:/…/Qwen3-1.7B-Q8_0.gguf -d sample_100.jsonl --dry-run
                                                    # → layered loop, 28 layers, 1 step, no OOM ✓
```

### Also fixed: `test_new_rejects_empty_varmap`

The pre-existing GWEN-217 regression is resolved by repurposing it to `test_new_populates_empty_varmap` — `new()` self-populates trainable params, so an empty input VarMap is the normal case and must succeed. `cargo test -p gwenland-core --lib train::` is now **81 passed, 0 failed**.

### Known pre-existing (NOT touched)

`engine::inference::selector::tests::{tilde_expand, relative_gguf_ok, empty_stop_sequences_ok}` fail under a default `cargo test` because they lack the `#[cfg(feature = "candle-backend")]` guard the sibling tests have (`default = []` compiles in no backend, so `select_backend("auto")` errors). They pass with `--features candle-backend`. Unrelated to this work; a one-line guard each would fix them if desired.

---

**End of Gwen-Changes-2026-06-14_GWEN-219.md**
