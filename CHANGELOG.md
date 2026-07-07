# Changelog

The notable changes, newest first. The blow-by-blow per-session notes live in [`changelog/`](changelog/).

## Unreleased

### 2026-07-07 — M1.5 correctness fixes, M1.6 batched prefill, M1.7 fast load: the GL engine reaches llama.cpp parity

The full architecture and benchmark story is in [`ArchGLML.md`](ArchGLML.md); the session notes are in [`changelog/Gwen-Changes-2026-07-07_16-30.md`](changelog/Gwen-Changes-2026-07-07_16-30.md).

Correctness (post-audit):

- Generation now stops at *any* of a model's stop tokens (`<|im_end|>`, `<|endoftext|>`, `</s>`, ...), not just the single metadata EOS id. The stop set is resolved from the vocab at load; stop tokens are never emitted.
- Added a repetition penalty over a 64-token sliding window (default 1.1, `gwen run --repeat-penalty`, 1.0 disables). Small models no longer loop — which also ended the artificially inflated tok/s that looping produced by keeping the same weights hot in L3.
- Chat models get their ChatML prompt template applied automatically (`<|im_start|>`/`<|im_end|>` emitted as special token ids); `--raw` opts out for base models. "What is 1+1" now answers "1+1 equals 2." and stops cleanly instead of rambling to `max_tokens`.
- Benchmark output separates prefill from generation: `[benchmark] prefill: N tokens @ X tok/s | generation: M tokens @ Y tok/s`. The old blended number understated decode speed and hid the looping artifact.

Performance (Qwen2.5-0.5B Q4_K_M on the i3-1115G4 dev box, quiet machine):

- Batched prefill: prompts run through the transformer in 32-token chunks so every weight row streams from DRAM once per chunk instead of once per token, with grouped row-dot kernels (8 activations share each weight block's load, sign prep, and f16 scale conversion) and chunk attention parallelized across the pool.
- Q4_K weights now repack to Q8_0 at load like Q5_0/Q6_K. This fixed the single biggest hidden cost: half the layers' `ffn_down` tensors are Q4_K in this file and were falling into the f32-bridge fallback, running ~15× slower than their repacked neighbors in prefill.
- The token embedding table stays quantized; lookups dequantize one row on demand. Saves ~500 MB of RAM on 150k-vocab models (933 MB on the 1.5B) and the table's dequantization at load. Tied-head models reuse the quantized table as the LM head.
- Model load is parallel across cores and reports a breakdown (`[load] tokenizer 0.08s | weights 0.72s | pin 0.07s`).
- Net effect: prefill 35 → **128–132 tok/s** (llama.cpp: 124.5), generation 20-ish honest → **33.5–35.2 tok/s** (llama.cpp: 39.0), load 2.5s → **0.9s**, peak RAM ~1.7 GB → **1.19 GB**. The 1.5B went from 5.3 to 12.1 tok/s generation.

Diagnostics:

- `[simd]` startup line names the SIMD strategy and each hot weight class's kernel path, so a scalar fallback can't hide.
- `GLPROC_PROFILE=1` now also prints a per-phase prefill profile alongside the decode profile.
- Benchmark hygiene, learned the hard way: Windows Defender rescanning the binary and model after every build silently collapsed benchmarks by 2–4×; exclude the workspace and model folder, and check CPU load is below ~15% before trusting any number.

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
