# Changelog

The notable changes, newest first. The blow-by-blow per-session notes live in [`changelog/`](changelog/).

## Unreleased

glcuda — M2.3 Stage 1, prefill de-serialization (post head-to-head):

- First llama.cpp head-to-head (CUDA build, same T4/models): **decode at
  parity** — 0.95× (Q8_0), 1.03× (Q4_K_M, glcuda ahead), 0.97× (Q4_0) —
  but prefill 17–30× behind (45–78 vs 1340–1470 tok/s).
- Stage 1a: token positions are consecutive integers, so a `pos_seq`
  identity array uploaded once at load replaces prefill's per-token
  `token_params` HtoD (~896 synchronous, pipeline-draining copies per
  32-token chunk). Zero PTX changes — the kernels already read pos by
  pointer. Also fixes a latent cursor bug (advance ran per layer,
  overcounting 28×; prompts >146 tokens on 7B would falsely hit "KV
  cache full").
- Stage 1b: five batched-over-tokens kernel variants (`gl_rms_norm_rows`,
  `gl_add_bias_rows`, `gl_rope_rows`, `gl_kv_write_rows`,
  `gl_attn_decode_rows` — causal via `cached_len = pos_seq[t]+1`) collapse
  prefill's serial per-token loop (~7000 launches/chunk) into ~15 launches
  per layer. Single-token originals untouched (the decode graph is
  captured against them). New parity test pins batched attention's
  causality row-by-row.
- Remaining prefill gap (Stage 2): Q4_K/Q4_0/Q6_K still fall back to
  per-token GEMV — batched tile GEMMs for the quant formats are next.

glcuda — M2.2 Q6_K kernel tuning (post-T4):

- First T4 run showed `gl_gemv_q6_k_soa` correct (parity green) but
  compute-stalled at 155–183 GB/s vs the Q4_K kernel's 242 — the in-kernel
  32-op per-byte 2-bit spread starved the memory pipeline, so Q4_K_M decode
  stayed flat (~36.5) instead of gaining.
- Fix: qh is repacked into the identical u32-per-8-values nibble layout as
  ql (each 2-bit field in a nibble slot), so the kernel rebuilds q6 with a
  single and/shl/or per int8×4 half and both quant loads are coalesced
  u32s. This widens qh 64→128 B/super-block (6.5625 → 7.0625 bpw, +0.5) —
  a deliberate bytes-for-ALU trade: the kernel had bandwidth headroom and
  no compute headroom, and 7.06 still beats the 8.5 requant path it
  replaced. Notebook 14c now builds `llama-quantize` and converts the Q8_0
  file to *pure* Q4_0 (`--pure --allow-requantize`) since public "Q4_0"
  GGUFs are mixed quants (Q4_1 ffn_down = the Unknown(3) load error).

glcuda — M2.2 Task C-1, native Q6_K SoA decode path:

- New PTX kernel `gl_gemv_q6_k_soa`: four SoA streams (packed low nibbles,
  2-bit highs, verbatim i8 sub-block scales, verbatim f16 super-block `d`)
  at the exact native 6.5625 bpw. q6 values are assembled in registers
  (`ql | qh<<4`, masked shifts into int8x4 lanes) and the −32 centering
  folds into the integer domain: `d·sc·xs·(dot(q6,xq) − 32·Σxq)` per
  16-value sub-block. Zero added quantization error — every stream is
  verbatim or losslessly relocated.
- Replaces the M2.1 Q6_K→Q8_0 requant detour (8.5 bpw) for the ~1.6 GB of
  Q6_K tensors a Q4_K_M 7B streams per token: expected ~+3 tok/s decode
  (38.8 → ~42). Q6_K embedding tables stay quantized host-side
  (`q6_k_row_into`). Disk cache bumped (`GLCACHE5`).
- Repack linearizes ggml's half/quarter interleave (its 16 i8 scales turn
  out to already be in linear sub-block order — verbatim copy); host test
  pins the reconstruction bit-exact vs glproc. Note: the real GGML
  `block_q6_K` is 210 B (qh is 64 B, not 32) — the task brief's 178 B
  sketch was corrected against `dequant.rs`/glproc.
- Parity `gemv_q6_k_soa` ε 2e-3 (5× under the task bound); bench `[q6k]`
  section at 7B shapes.

glcuda — M2.2 Task C-2, native Q4_0 SoA decode path:

- New PTX kernel `gl_gemv_q4_0_soa`: the Q4_K kernel's structure minus the
  mins stream, with the −8 centering folded into the integer domain
  (`d·xs·(dot(q,xq) − 8·Σxq)`, both dp4a chains, one f32 mul + one fma per
  32-value block). Verbatim f16 block scales — no pre-multiply, zero
  rounding loss. Guarded tail iteration keeps the requirement at
  `in % 32 == 0`, so dim-896-class Q4_0 models work.
- Loader: Q4_0 matmul tensors repack to SoA (4.5 bpw streamed); the AoS
  path and legacy `gl_gemv_q4_0` kernel remain for the embedding table
  only. Disk cache bumped (`GLCACHE4`).
- Parity: two shapes covering the grouped path and the tail (ε 1e-3 — the
  error structure is Q8_0 SoA's, stricter than the task's 1e-2 bound);
  bench gains a `[q4_0]` section at 7B shapes.
- Expected on a Q4_0 7B file: ~3.9 GB/token stream → ≥50 tok/s decode DoD.

glcuda — M2.1 Task B, INT8 tensor-core prefill GEMM:

- New hand-authored `gl_gemm_mma_q8` in a separate `.target sm_75` PTX
  module (`glcuda_sm75.ptx`), loaded only on sm_75+ devices; the sm_70 dp4a
  `gl_gemm_q8_0_soa` stays as the runtime fallback (and `GLCUDA_NO_MMA=1`
  forces it, as the A/B benchmark switch).
- Shape correction vs the task brief: integer `mma.m16n8k16` is sm_80+ per
  the PTX ISA — Turing's INT8 tensor-core shape is `m8n8k16`, which is what
  the kernel uses (one warp per 8×8 output tile, 8-token tiles vs the dp4a
  GEMM's 4, so weight streams halve on top of the tensor-core dot).
- No new weight layout: the row-major Q8_0 SoA qs stream *is* the col-major
  B fragment `mma.row.col` expects (W row-major = Bᵀ col-major), so both
  GEMMs share one weight image.
- Dequant epilogue fused per 32-K block in registers (Q8_0 scales are
  per-32 but one MMA covers K=16, so int32 accumulation is only
  scale-uniform across two chained MMAs); FP32 accumulation across blocks.
- Parity: `gemm_mma_q8` vs the dequantized scalar reference (ε 1e-3,
  ragged token count exercising the guarded writes); bench gains an MMA
  section reporting kernel time/TOPS next to the dp4a GEMM on identical
  data.

glcuda — M2.1 Task A, native Q4_K decode path:

- New hand-authored PTX kernel `gl_gemv_q4_k_soa`: one warp per output row,
  one loop iteration per 256-weight super-block (128 coalesced qs bytes),
  dp4a integer dots against the int8-quantized activation. Streams 5.0 bpw
  per decode token vs 8.5 for Q8_0 SoA — the lever past the 7B Q8_0
  bandwidth ceiling (29.7 tok/s ≈ 88% of achievable on a T4).
- New `repack` module: Q4_K super-blocks → SoA streams (nibbles repacked
  for u32/dp4a consumption; sub-block scales and mins pre-multiplied to f16
  `d*sc` / `dmin*m`, buying ggml's 6-bit scale unpack out of the hot loop),
  plus an f32 → Q8_0-SoA requantizer.
- Loader policy for Q4_K_M files: Q4_K matmul tensors go native SoA; the
  Q6_K/Q5_0 tensors those files carry are requantized to Q8_0 SoA (glproc's
  repack policy) instead of dense f32; Q4_K embedding tables stay 4.5 bpw on
  the host with per-row dequant; tied LM heads are staged through the same
  repack as any matmul weight. Staged-model disk cache format bumped
  (`GLCACHE3`) so pre-Q4_K caches restage.
- Parity: `gemv_q4_k_soa` vs glproc scalar Q4_K dequant + matvec (ε 1e-2);
  host-side repack round-trip tests; the existing GPU parity suite is
  untouched.

## 0.1.48-alpha — 2026-07-08

First tagged alpha release. Bundles the M1.5–M1.7 GL engine wave with a full
TUI visual overhaul, and cleans up release plumbing. All workspace crates are
versioned `0.1.48`; published as a GitHub **pre-release**.

TUI (`gltui`) — OpenCode-inspired visual redesign:

- Retired the ASCII "haunted mansion" welcome logo. The welcome card now shows
  the **GwenLand** wordmark in the ANSI Shadow figlet font (the Claude Code CLI
  banner style) in the accent orange, with a graceful plain-text fallback on
  narrow terminals.
- The card also reports live **device info** — CPU (brand + core count), RAM
  (available / total), arch/OS, and GPU — probed once at startup via
  `gwenland-core`'s hardware profiler so it never runs on the render path.
- Reworked the palette to a single warm accent (`#b56936`) on a near-black
  surface, a floating rounded command palette (search line + right-aligned
  shortcuts + orange selection), a rounded input box that lights up when
  focused, and a tab-style status bar. No logic, command, or keybinding changes.

Packaging & licensing:

- Consolidated the two overlapping license files into a single `LICENSE`
  (MIT + Commons Clause v1.0). All crate `license-file` fields and the README
  now point at it; fixed a stale `license-file` path in `gwenland-core`.
- Every workspace crate bumped to `0.1.48`.

CI/CD:

- Release build (`build.yml`) now builds the real package: `--package glcli`
  (the `gwen` binary) instead of the nonexistent `gwenland-tui`, with binary
  paths corrected from `gwenland` to `gwen`. The serve+chat e2e workflow builds
  `glcli`/`gltui` and tests against `gltui`.
- Tags matching `-alpha`/`-beta`/`-rc` publish as GitHub pre-releases; this
  release ships under `v0.1.48-alpha`.

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
