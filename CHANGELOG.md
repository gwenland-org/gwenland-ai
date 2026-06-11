# Changelog

All notable changes to GwenLand are documented here.
Full per-session changelogs live in [`gwen-cli/changelog/`](gwen-cli/changelog/).

---

## [Unreleased]

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
