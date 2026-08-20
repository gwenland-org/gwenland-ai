# glbench — Architecture

This is the **structural** map: crate layout, entry points, dependency
direction. For *why* it's shaped this way (responsibility boundaries, design
decisions), see [`DESIGN.md`](DESIGN.md). For *how to use it*, see
[`README.md`](README.md). Overlap between the three is intentional at the
edges (README's own "Architecture" section gives the short version of the
module table below) — this document is the one that stays accurate against
`src/` as the crate grows, so update it, not the other two, when a module is
added or moved.

## Where glbench sits in the workspace

```
                      ┌─────────────────────────────────────────┐
                      │              glbench (this crate)         │
                      │  observes — never optimizes, never       │
                      │  modifies a model file or a kernel        │
                      └───────────────┬─────────────────────────┘
                                       │ depends on (one-directional)
              ┌────────────┬──────────┼──────────┬─────────────────────┐
              ▼            ▼          ▼          ▼                     ▼
          glcore       glproc     glcuda   glictus-caliburni      (nothing
       (Runtime,     (CPU engine, (CUDA    (.gllm / GLLM runtime,  else —
     GlEngine trait,  the numeric engine)   optional, feature      zero
      tokenizer)      oracle)               `gllm-bench`)          external
                                                                    deps)
```

`glcore`/`glproc`/`glcuda` are unconditional dependencies (workspace-path,
add no external surface). `glictus-caliburni` is the one *optional*
dependency, gated behind the `gllm-bench` Cargo feature — pulling it in
unconditionally would make every glbench build carry `glproc-backend`
support for a format most builds never touch. No engine crate, and no other
part of the workspace, depends back on glbench — the arrow only ever points
one way.

## Crate-internal module map (current, `src/`)

```
glbench/
├── src/
│   ├── main.rs          — CLI entry point (hand-rolled arg parser, no clap —
│   │                       see "Dependencies: zero" in DESIGN.md §9)
│   ├── lib.rs            — re-exports, top-level module doc
│   ├── core/             — BenchmarkSession data model (see DESIGN.md §4)
│   ├── environment/       — machine probe (CPU/GPU/memory/storage/runtime)
│   ├── engine/            — the only boundary to glcore/glproc/glcuda/gllm
│   ├── runner/            — orchestrates one run: warmup → iterations → phases
│   ├── measurement/        — raw facts only (latency, tok/s, bytes, peak RSS,
│   │                         process CPU-time / utilization)
│   ├── behavior/           — per-token signals (repetition, entropy, stall,
│   │                         ood, hallucination-proxy, anomaly, cot, drift)
│   ├── analysis/           — facts → insight (bottleneck, ceiling, health)
│   ├── comparison/         — run/engine/quant/hardware deltas, regression,
│   │                         trend, accuracy-vs-performance
│   ├── validation/         — integrity, determinism, numerical parity (oracle),
│   │                         KV-cache memory-risk
│   ├── numerical/          — GLBitProf: bit-level tensor observation. The
│   │                         math is UNGATED (`&[f32] -> VLBitProfile`); its
│   │                         sources are gated per source (v3 D-11)
│   ├── training/           — training observation, all behind `train-bench`
│   │                         so a default build never compiles gltrain (D-02)
│   ├── export/             — hand-rolled JSON / Markdown / CSV writers
│   ├── render/             — terminal text + tables + ASCII flame graph +
│   │                         ASCII loss curve (ungated, plots (step, loss))
│   ├── storage/            — user-managed archive files (no database)
│   ├── quant_info.rs       — `.gllm` manifest tally (see "Two .gllm readers" below)
│   ├── ppl.rs              — WikiText-2 perplexity via the `.gllm` runtime
│   │                          (`#[cfg(feature = "gllm-bench")]`)
│   ├── kl_divergence.rs    — per-position KL-divergence, `.gllm` vs the
│   │                          glproc oracle, teacher-forced over the same
│   │                          WikiText-2 sample as `ppl` (also `gllm-bench`)
│   └── tensor_stats.rs     — decodes every tensor, flags NaN/Inf/
│                              zero-variance; `--full` adds a per-tensor
│                              distribution, `--norm-only` scopes to RMSNorm
│                              gamma weights (also `gllm-bench`)
└── tests/
    ├── ppl_tests.rs
    └── quant_info_tests.rs
```

`quant_info.rs`, `ppl.rs`, and `kl_divergence.rs` are flat top-level modules
(not folders) — smaller, single-file additions from later work (Wave
1/Wave 2 of the GLLM benchmarking effort, plus the KL-divergence follow-up)
that didn't yet warrant a directory. `ppl` and `kl_divergence` are gated
behind `gllm-bench`; `quant_info` is not (it doesn't touch
`glictus-caliburni` at all — see below), so it's always compiled.

## Entry point → subcommand → module

`main.rs` dispatches on `args.first()` as a plain string match (no `clap`,
consistent with the zero-dependency rule — `glcli`, by contrast, does use
`clap`, because interface crates get more latitude, see the workspace
`CONTRIBUTING.md`):

| subcommand | handler | primary modules touched |
|---|---|---|
| `run` | `cmd_run` | `runner` → `engine` → `core` → `render`/`storage` |
| `ab` | `cmd_ab` | `runner` (N sequential runs) → `comparison` |
| `compare` | `cmd_compare` | `storage` (load 2 archives) → `comparison` |
| `validate` | `cmd_validate` | `runner` (candidate + oracle) → `validation` |
| `scale` | `cmd_scale` | `runner` (sweep) → `analysis::scaling` |
| `thread-scale` | `cmd_thread_scale` | `runner::thread_scale` (sweep over `GLPROC_THREADS`) → `analysis::scaling`, glproc-only |
| `inspect` | `cmd_inspect` | `storage` → `render` (no new measurement) |
| `export` | `cmd_export` | `storage` → `export` (no new measurement) |
| `accuracy-vs-perf` | `cmd_accuracy_vs_perf` | `storage` + raw JSON parse (2 archives) → `comparison::accuracy` (no new measurement) |
| `quant-info` | `cmd_quant_info` | `quant_info` only — no `engine`, no inference |
| `ppl` | `cmd_ppl` | `ppl` → `glictus-caliburni::GllmEngine::score_sequence` directly (bypasses `engine`/`Runtime` — see below) |
| `kl-div` | `cmd_kl_div` | `kl_divergence` → `glproc::runner::Runner` (oracle) + `GllmEngine::score_sequence` (candidate), both directly — same bypass as `ppl`, on both sides |
| `tensor-stats` | `cmd_tensor_stats` | `tensor_stats` → `glictus-caliburni::package`/`glproc::kernels::gquant` directly — same "no tokenizer needed" bypass shape, since this never runs inference at all |

Every subcommand except `quant-info`, `ppl`, `kl-div`, `tensor-stats`, and
`accuracy-vs-perf` goes through the shared `runner` → `engine::adapter` seam
described in DESIGN.md §2. Those five are the exceptions, for different
reasons:

- **`quant-info`** never runs inference at all — it's a static file reader,
  so there's no `engine` boundary to cross.
- **`tensor-stats`** also never runs inference — it decodes tensor bytes to
  f32 directly (the same dispatch `GllmEngine::load_shared` uses), same
  "static reader, no `engine` boundary" shape as `quant-info`.
- **`accuracy-vs-perf`** never runs inference either — it reads two already-written
  archives off disk and joins them; no `engine` boundary, no new measurement.
- **`ppl`** *does* run inference, but a `.gllm` package has no tokenizer
  (ARTX1 OQ3), so it cannot go through `glcore::Runtime` (which owns
  tokenization) the way every other engine does. It talks to
  `GllmEngine::score_sequence` directly, feeding it token ids rather than
  prompt text — see `engine/adapter.rs`'s module docs and the `gllm` engine
  path in `README.md`'s "Benchmarking GLLM" section for the same shape of
  workaround at the `run`/`ab`/`scale` layer.
- **`kl-div`** has the same tokenizer problem as `ppl` on the candidate
  side, plus a second reason on the oracle side: it needs full raw logit
  vectors at every position, and `GlEngine`/`InferOutput` do not expose
  those (only sampled token ids, or the already-reduced-to-scalars
  `glcore::trace::TokenTrace`) — so it drives `glproc::runner::Runner`
  directly via `forward_into` + `logits()`, the same pair of calls
  `glictus-caliburni/examples/diff_dump.rs` uses for the identical reason.

## Two independent `.gllm` readers, on purpose — with a caveat

`quant_info.rs`'s own doc comment states it directly: *"glbench does not
import glictus-caliburni, so this is *a* reader of the `.gllm` manifest
shape, not *the* reader — it knows only the fields it needs... and treats
everything else as opaque."* This is a deliberate, narrow, hand-rolled JSON
parser reading `gllm.json`'s dtype tallies only — kept separate from
`glictus-caliburni::manifest::GllmManifest` (the real, full parser) so that
`quant-info` (unlike `ppl`, `--engine gllm`) never needs the `gllm-bench`
feature or its `glictus-caliburni` dependency at all.

The caveat, now that it's been named explicitly elsewhere in this repo: this
is structurally the same shape as the "two independent implementations of
the same format, agreeing only by construction" pattern documented in
[`architecture/gl-stack-audit-2026-07/ARTX2-Quant.md`](../architecture/gl-stack-audit-2026-07/ARTX2-Quant.md)
— there, two independent *bit-level* dequant implementations silently
diverged (the Q6_K bug). The risk here is much smaller (`quant_info.rs`
parses JSON structure and a dtype string tally, not packed binary weights —
far less surface for a silent semantic disagreement), but if `gllm.json`'s
schema changes and only `glictus-caliburni::manifest` gets updated,
`quant-info`'s independent reader could silently under- or over-count
without erroring. Worth a schema-drift regression test if `quant-info`
becomes load-bearing for anything beyond its current diagnostic role — not
done as part of this document, flagged for whoever picks it up.

## Feature gates, and what each one costs

| Feature | Pulls in | Enables |
|---|---|---|
| *(default)* | `glcore`, `glproc`, `glcuda` | everything except the two below |
| `gllm-bench` | `glictus-caliburni` (`glproc-backend`, `converter`) | `ppl`, `kl-div`, `tensor-stats`, `--bit-scope weights` |
| `train-bench` | `gltrain` | `glbench train`, `glbench unified`, `--bit-scope gradients\|optimizer` |

Both are optional for the same reason: a default `cargo build --workspace`
must not compile them. Verified rather than assumed — building the workspace
with no features compiles `gltrain` **zero times**.

`train-bench` needs one thing the design did not anticipate: `gltrain` declares
its own `[workspace]`, so it must also appear in the **root** `Cargo.toml`
`exclude` list or cargo refuses the path dependency with "multiple workspace
roots found in the same workspace". Both isolation properties still hold after
that change — the default build ignores gltrain, and `cd gltrain && cargo test`
still works standalone.

## The v3 envelope (schema v2)

`SCHEMA_VERSION` is 2. A v3 build reads a v1 archive by defaulting
`session_mode` to `inference_only`, `availability` to an empty map, and
`integrity` to absent; a v1 build correctly refuses a v2 archive through the
existing check in `storage::archive::read`. The v1 fixture under
`tests/fixtures/v1_archive.json` was derived from the pre-v3 writer's own
source and its key set verified against it, so the compatibility claim is not
the v3 writer grading its own homework.

Three things the envelope adds:

- **`availability`** — a sparse map from dotted field path to why that field
  has no value. A `null` with no entry is a defect, and
  `validation::availability` fails the session on one (D-10). `archive::write`
  is the finalisation point where that check runs, so a malformed archive is
  never written; `--null-semantics lenient` downgrades it to a warning.
- **`integrity`** — a `sha256-128` content digest over the archive's own
  canonical JSON. It detects accidental modification and is **not a
  signature**: anyone who can edit the file can recompute it, and
  `storage::digest`'s own docs say so.
- **`inference` / `training`** — each carrying an explicit role, so a
  `unified` session's before-and-after runs are distinguishable from the data
  rather than from their position in the tree.

## Data flow

See [`DESIGN.md` §3](DESIGN.md#3-data-flow) for the full `WorkloadSpec →
BenchmarkSession → {analysis, validation} → {export, render, storage}`
pipeline diagram — not repeated here to avoid the two documents drifting
out of sync with each other.
