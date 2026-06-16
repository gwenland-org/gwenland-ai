# Changelog

All notable changes to GwenLand are documented here.
Full per-session changelogs live in [`gwen-cli/changelog/`](gwen-cli/changelog/).

---

## [Unreleased]

### 2026-06-16 — GWEN-224: Storage restructure to `~/.gwenland/` + crash reports

**Breaking change — no auto-migration.**

- All config, model, and registry storage moved from `~/.config/gwen/` to a
  single home-dotfile root: `~/.gwenland/{config,models,crash-logs}/`
  - `~/.gwenland/config/config.json` (was `~/.config/gwen/config.json`)
  - `~/.gwenland/models/` (was `~/.config/gwen/models/`)
  - `~/.gwenland/crash-logs/` (new)
- The old `~/.config/gwen/` directory is left untouched — nothing is deleted
  or copied. **Pre-1.0 users upgrading must re-run `gwen fetch <model>`** to
  re-populate models under the new path.
- Panics anywhere in `gwen` (CLI/TUI/GUI) now write a human-readable crash
  report to `~/.gwenland/crash-logs/crash-<timestamp>.txt`, including
  version, surface (CLI/TUI/GUI/Serve), command line, OS details, and the
  panic message + location. Backtraces are included when `RUST_BACKTRACE=1`.
- OS-level faults (SIGSEGV/SIGABRT/SIGILL/SIGBUS on Unix, unhandled
  structured exceptions on Windows) are now also captured to the same
  crash-log directory via a best-effort signal handler / unhandled-exception
  filter — useful for native inference crashes (candle/mistral.rs) that
  don't go through Rust's panic machinery.
- `gwen doctor` now reports existence + writability of
  `~/.gwenland/{root,config,models,crash-logs}/`.

### 2026-06-11 — GWEN-214 Follow-up
- Optional CLI flags for `gwen benchmark` with config fallback (CLI > config.toml > default)
- `BenchmarkConfig` section added to `~/.config/gwen/config.toml`
- `LayerIndex::scan` now handles `blk.{N}.*` tensor prefix (llama.cpp / Qwen / Mistral / Llama format)
- Fixed `--layer-load N` producing all-zero output on real GGUF models

### 2026-06-10 — GWEN-214: E2E Pipeline Validation & Benchmark Update
- Layer-load benchmark module (`run_layer_load_bench`) with per-layer RSS sampling
- Benchmark report formatter: JSON (`schema_version: "2"`) and human-readable text
- Feature-gated mistral.rs inference benchmark path
- E2E integration test suite (binary size, cold-start, LoRA training, JSON round-trip)

### 2026-06-10 — GWEN-216: Selective Layer Loading
- `LayeredTrainingLoop` with mmap-based lazy layer loading
- `LIVE_LAYER_COUNT` atomic counter for RSS-bounded training
- `LayerLoader`, `LayerIndex`, `LoadedLayer` types

### 2026-06-09 — GWEN-213: LoRA Adapter Pipeline
- Candle LoRA → GGUF dequant-merge-requant pipeline
- `LoraExporter`, `LoraMerger`, `LoraConfig`
- GGQR-Candle zero-copy inference backend

### Earlier
See [`gwen-cli/changelog/`](gwen-cli/changelog/) for full session-by-session history.
