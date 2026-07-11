# GwenLand - 2026-07-12: glbench (Mensura Veritatis) — Benchmark Framework, Phase 1–4 Foundation

**Date:** 2026-07-12 (WIB / SEAST)
**Scope:**
- New crate: `glbench/` (workspace member) — 40 source files across 11 module folders
- Workspace: `Cargo.toml` (added `glbench` to `members`), `Cargo.lock`
- Docs: `glbench/{README.md,DESIGN.md,ROADMAP.md}`
**Type:** New subsystem — a standalone benchmark execution and performance-analysis framework.
**Status:** Implemented; full workspace builds; 40 crate tests green; `cargo clippy -p glbench` clean. Not yet run against a live model (no local GGUF this session).
**Hardware:** Dev box (i3-1115G4, Windows 11). Release binary: **736 KB**.

---

## Executive Summary

glbench is a new first-class `gl*` crate: a benchmark framework whose one job is
to **measure the truth about engine performance** and never to optimize it.

```
Execute → Measure → Analyze → Compare → Validate → Report
```

The whole thing hangs off a single pure-data `BenchmarkSession` (the source of
truth) and runs inference **only** through the existing `GlEngine` contract in
glcore — it never duplicates inference logic, touches a kernel, or modifies a
model. It builds with **zero external dependencies**: standard library plus the
existing workspace crates (`glcore`, `glproc`, `glcuda`). The JSON parser/writer,
CSV writer, Markdown renderer, table layout, and CLI arg parser are all
hand-rolled, which is why the release binary is 736 KB with `glcore`/`glproc`/
`glcuda` statically linked.

Delivered as one crate with internal module folders (not sub-crates), matching
the design brief's architecture tree exactly. Phases 1–3 landed together as the
foundation; Phase 4 (rendering) has a working text + table baseline.

---

## Architecture

`core::session::BenchmarkSession` is a **data model with no business logic** —
every subsystem reads or fills one of its fields:

```
BenchmarkSession
├── SessionMetadata      — label, timestamp, tool + schema version
├── EnvironmentSnapshot  — CPU / GPU / memory / storage / runtime
├── EngineMetadata       — engine name, backend, model arch, quantization
├── WorkloadSpec         — what was run
├── MeasurementSet       — raw facts only (latency, tok/s, bytes)
├── AnalysisReport       — derived insight (health, bottleneck, ceiling)
├── ComparisonReport     — run/engine/quant/hardware delta + regression
└── ValidationReport     — is the benchmark trustworthy?
```

| module        | responsibility                                                     |
|---------------|--------------------------------------------------------------------|
| `core`        | the data model (session, metrics, workload, schema)                |
| `environment` | probe the machine — std + OS files only (`/proc` on Linux)         |
| `engine`      | the **only** boundary to the engines; runs via glcore's `Runtime`  |
| `runner`      | orchestrate a run: load → warmup → measured iterations → phases     |
| `measurement` | store raw facts; convert counts+durations to rates                 |
| `analysis`    | facts → insight, always as recommendations, never actions          |
| `comparison`  | run/engine/quant/hardware deltas, regression, trend, statistics    |
| `validation`  | integrity, determinism, reproducibility, numerical parity vs glproc |
| `export`      | hand-rolled JSON / Markdown / CSV                                  |
| `render`      | terminal text + fixed-width tables                                 |
| `storage`     | user-managed archive files (no database)                          |

---

## What Changed

### 1. The engine boundary (`engine::adapter`)

glbench does not implement inference. `EngineAdapter` drives glcore's `Runtime`
(which owns tokenization and holds one `Box<dyn GlEngine>`) and reads the
engine's `InferOutput` — which already separates prefill from decode timing —
translating it into raw `IterationMetrics` at a single auditable seam
(`measurement::raw::from_infer_output`). `build_engine` is the **only** function
in the crate that names concrete engine types (`GlprocEngine`, `GlcudaEngine`);
adding glvulkan/glmetal later is one match arm.

The glcuda path reflects the device's `Cuda::probe()` facts (name, `sm_XX`,
total VRAM) into the GPU snapshot and looks up a published bandwidth ceiling from
a small capability table (T4, A100, V100, L4, 3090, 4090). A device not in the
table yields no ceiling and the analysis says so rather than inventing one.

### 2. Facts vs conclusions — the cardinal line

`MeasurementSet` stores numbers only: `memory_bandwidth = 240.0` lives in
measurement; `bottleneck = MemoryBound` lives in analysis, derived from it and
always separable. This is what lets glbench claim to measure truth — the facts
are auditable, the interpretation is re-derivable.

### 3. Analysis, honest about what it can't know

- **Ceiling**: for memory-bound decode, `peak_bandwidth / model_bytes` is the
  tok/s ceiling; observed / ceiling is the efficiency. No bandwidth figure → no
  ceiling, stated explicitly.
- **Bottleneck**: an explicitly-labelled heuristic over ceiling efficiency
  (≥85% memory-bound, ≥40% compute-bound, below that launch-overhead); with no
  ceiling it returns `Undetermined` rather than over-claiming. Each verdict
  carries a `recommendation()` phrased as an observation — never an action.
- **Health**: blends ceiling efficiency with run-to-run stability (coefficient
  of variation); with no ceiling, reports stability alone and notes it's partial.
- Plus `efficiency`, `roofline` (arithmetic-intensity ridge point), and
  `scaling` (linear / sub-linear / saturating over a sweep).

### 4. Comparison + regression

`runs::compare` is the core delta (baseline vs candidate, decode/prefill,
relative + ratio, regression verdict at a threshold). engine/quantization/
hardware comparisons are the same delta viewed along one axis. `statistics`
gives mean/median/min/max/std/p95/p99 with interpolated percentiles;
`regression` classifies improved/neutral/regressed; `trend` walks an ordered set
of archives with no persistent store.

### 5. Validation + the glproc oracle

integrity (structural sanity, counter consistency, variance), deterministic
(seed/warmup/temperature pinned?), reproducibility (archive self-describing
enough to re-run?), and numerical — a candidate engine's greedy token stream vs
glproc's, reported as the longest matching prefix. An `Error`-severity finding
fails validation; warnings are allowed.

### 6. Storage, export, CLI

Archives are single JSON files (no database, no cloud, no unlimited history),
each stamped with `glbench_version` + `schema_version`; the reader refuses a
schema newer than the build understands. Exporters: hand-rolled JSON (pretty,
round-trips), Markdown (house-style tables + prose), CSV (one row per iteration,
RFC-4180 quoting). CLI (`run` / `compare` / `inspect` / `export`) with a
hand-rolled arg parser — no clap.

---

## Design Decisions (surfaced, not assumed)

- **Repo path.** The brief said `gwenland-ai/`, but the actual layout has the
  engine crates at the repo root (`glcore/`, `glcuda/`, …). Placed `glbench/`
  there alongside them and added it to the workspace `members`.
- **No serde, despite glcore using it.** The "zero external serialization
  crates" rule is explicit; `export::json` is a small hand-rolled value model.
  It also decouples the data model from serde derives on foreign types.
- **One crate, internal modules** — not `glbench-core`/`glbench-analysis`/…
  Splitting into crates is deferred until real architecture pressure requires it.

Open flag for the user: the adapter links `glcuda` (which `dlopen`s the CUDA
driver). It self-probes and reports `available: false` on a CPU-only box, but a
`cuda` feature-gate can drop the link entirely on non-CUDA builds if wanted.

---

## Test State

- 40 crate tests green (JSON round-trip incl. unicode/escapes/non-finite,
  statistics + percentile interpolation, ceiling/health/roofline/scaling math,
  regression verdicts, timeline residual, stress drift, CSV quoting, archive
  disk round-trip, numerical prefix match, capability lookup).
- `cargo clippy -p glbench` clean; full workspace builds (`cargo build`).
- Release binary 736 KB (workspace `opt-level="z"` + fat LTO + strip; zero-dep
  graph).

Full architecture + usage: [`../glbench/README.md`](../glbench/README.md),
[`../glbench/DESIGN.md`](../glbench/DESIGN.md), [`../glbench/ROADMAP.md`](../glbench/ROADMAP.md).
