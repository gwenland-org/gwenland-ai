# Changelog

The notable changes, newest first. The blow-by-blow per-session notes live in [`gwen-cli/changelog/`](gwen-cli/changelog/).

## Unreleased

### 2026-06-20 — GUI packaging, a serve fix, and some CI housekeeping

- Brought the GUI back to life: its frontend build tooling (`package.json`, Vite, TypeScript, Tailwind) was missing, so the Tauri window had nothing to load. Added it, and fixed a bundle that pointed at a deleted icon.
- The desktop installer now ships the `gwen` CLI alongside the app as a Tauri sidecar, so the GUI's "start the server" button actually has a binary to run.
- Fixed `gwen serve` rejecting the model: it took the model as a positional argument, but the app's own hints told you to pass `--model`, which didn't exist. It now accepts both, and falls back to the last model you served if you don't name one.
- Fixed the real reason `gwen serve` looked like it hung for ten minutes — it was fetching the tokenizer from HuggingFace on every chat message, keyed on the local model name. It now reads the tokenizer from next to the model (or a real repo, with a timeout) and caches it.
- Fixed a macOS build break in the layer loader: `MADV_DONTNEED` is now scoped to Linux, where it actually exists and does something.
- Added a CI pipeline and a `CONTRIBUTING.md`. The pipeline is parked for now (GitLab's shared runners want a credit card).

### 2026-06-16 — GWEN-224: storage moved to `~/.gwenland/`, plus crash reports

This is a breaking change with no automatic migration.

- Config, models, and the registry moved out of `~/.config/gwen/` into a single folder in your home directory: `~/.gwenland/{config,models,crash-logs}/`. The old location is left untouched — nothing is deleted or copied. If you're upgrading from a pre-1.0 build, re-run `gwen fetch <model>` to repopulate.
- Any panic in `gwen` (CLI, TUI, or GUI) now writes a readable crash report to `~/.gwenland/crash-logs/`, with the version, which surface was running, the command line, OS details, and the panic message and location. Backtraces show up when `RUST_BACKTRACE=1`.
- Lower-level faults that don't go through Rust's panic machinery — segfaults and friends, including ones from native inference code — get caught by a best-effort signal handler (or the unhandled-exception filter on Windows) and written to the same place.
- `gwen doctor` now checks that the new folders exist and are writable.

### 2026-06-11 — GWEN-214 follow-up

- `gwen benchmark` takes optional CLI flags now, falling back to config and then defaults.
- Added a `BenchmarkConfig` section to the config file.
- `LayerIndex::scan` handles the `blk.{N}.*` tensor naming used by llama.cpp, Qwen, Mistral, and Llama GGUFs.
- Fixed `--layer-load N` producing all-zero output on real models.

### 2026-06-10 — GWEN-214: end-to-end pipeline checks and benchmark updates

- A layer-load benchmark that samples RSS per layer.
- A benchmark report formatter, in both JSON and plain text.
- A feature-gated mistral.rs inference benchmark path.
- An end-to-end integration suite covering binary size, cold start, LoRA training, and JSON round-trips.

### 2026-06-10 — GWEN-216: selective layer loading

- `LayeredTrainingLoop`, which mmaps and loads layers lazily.
- A `LIVE_LAYER_COUNT` counter to keep training inside its RAM budget.
- The `LayerLoader`, `LayerIndex`, and `LoadedLayer` types.

### 2026-06-09 — GWEN-213: the LoRA adapter pipeline

- A Candle LoRA to GGUF dequant-merge-requant pipeline.
- `LoraExporter`, `LoraMerger`, and `LoraConfig`.
- The GGQR-Candle zero-copy inference backend.

### Before that

See [`gwen-cli/changelog/`](gwen-cli/changelog/) for the full session-by-session history.
