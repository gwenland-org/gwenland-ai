# glbench — Design

Codename **Mensura Veritatis** ("measure of truth"). This document records the
responsibility boundaries, the data flow, and the design decisions that keep
glbench honest.

> **⚠️ This is the v1/v2 design.** The v3 design — schema v2, null semantics,
> content digest, GLBitProf, training observation — lives at
> **`architecture/glbench-v3/DESIGN.md`**, with decisions numbered D-01…D-20 and
> grounding findings F-01…F-11. If you are looking for a section number like
> §6.5 or a decision like D-12, that is the file you want; nothing in *this*
> document uses that numbering.
>
> Everything below still holds. v3 extends the envelope and adds modules; it
> changes none of the boundaries here. §9's zero-dependency rule in particular
> survived v3 intact — the whole of v3 added **zero** external crates and
> removed seven.

---

## 1. Responsibility boundary

The single most important design fact: **glbench observes performance; it does
not optimize it.** Everything else follows from that line.

| Owner            | Owns                                                          |
|------------------|--------------------------------------------------------------|
| **GwenLand AI**  | inference execution, training, model loading, runtime lifecycle, GGUF→GLLM pipeline |
| **Engine crates** (glproc, glcuda, glvulkan, glmetal) | tensor execution, kernels, hardware acceleration |
| **glbench**      | benchmark execution, metric collection, analysis, comparison, validation, report generation |

glbench **must not**: optimize/modify/autotune kernels, modify model files,
change quantization, touch hardware config, create a scheduler, manage multi-GPU,
route between engines, replace the runtime, manage cloud infra, or keep an
unlimited history database.

glbench **may** provide optimization *recommendations* — phrased as
observations. Good: *"Performance is memory-bandwidth bound."* Bad:
*"Automatically rewrite the CUDA kernel."* The `analysis::bottleneck` module is
where this line is enforced in code: it emits a classification plus a
`recommendation()` string, and nothing actionable.

## 2. Engine execution model

glbench does not implement inference. It runs everything through glcore's
`Runtime`, which owns tokenization and holds one `Box<dyn GlEngine>`:

```
glbench
   │
   ▼
engine::adapter  ──►  glcore::Runtime  ──►  Box<dyn GlEngine>  ──►  hardware
```

`engine::adapter::build_engine` is the **only** function in the crate that names
concrete engine types (`GlprocEngine`, `GlcudaEngine`). Adding a backend is one
match arm there; nothing else changes. The adapter never duplicates inference
logic — it reads the engine's `InferOutput` (which already separates prefill from
decode timing) and translates it into glbench's raw metrics at a single
auditable seam (`measurement::raw::from_infer_output`).

## 3. Data flow

```
WorkloadSpec ─► runner::planner
                    │  probe environment
                    │  load engine + model (adapter)
                    │  warmup (untimed)
                    │  measured iterations ─► MeasurementSet (raw facts)
                    ▼
              BenchmarkSession ──► analysis::analyze   ─► AnalysisReport
                    │           └► validation::validate ─► ValidationReport
                    ▼
       export::{json,markdown,csv} / render::text / storage::archive
```

Every stage reads or fills a field of the session. No stage holds hidden state;
re-running analysis over the same measurements is deterministic.

## 4. The single source of truth

`core::session::BenchmarkSession` is a **data model with no business logic**.
This is deliberate:

- Renderers and exporters consume the session; none builds its own data model.
- Analysis / comparison / validation are free functions *over* a session, not
  methods that mutate hidden state.
- The session's JSON projection *is* the archive format, so what you see
  rendered and what you store are the same facts.

## 5. Measurement stores facts, not conclusions

The cardinal rule of the `measurement` / `core::metrics` layer:

```
correct:   memory_bandwidth = 240.0        (a fact)
wrong:     bottleneck       = MemoryBound  (a conclusion)
```

`MeasurementSet` holds latency, token counts, phase durations, model bytes, and
optional device counters — numbers only. The interpretation
(`Bottleneck::MemoryBound`) lives in `analysis`, computed *from* those numbers
and always separable from them. This separation is what lets glbench claim to
measure truth: the facts are auditable, and the interpretation can be re-derived
or disputed without re-running the benchmark.

## 6. Analysis is honest about what it can't know

- **Ceiling analysis** needs a hardware bandwidth figure. glbench does not link a
  GPU SDK (dependency rule), so it uses a small published-spec table
  (`engine::capability`) keyed by device name. A device not in the table yields
  no ceiling, and the analysis *says so* rather than inventing one.
- **Bottleneck classification** is an explicitly-labelled heuristic over ceiling
  efficiency; with no ceiling it declines to over-claim (`Undetermined`).
- **Health** blends ceiling efficiency with run-to-run stability; with no
  ceiling it reports stability alone and notes the number is partial.

This mirrors the project's hard-won lesson: measure before concluding, and never
state a theory as a fact.

## 7. Validation & the glproc oracle

`glproc` is the ground-truth oracle (pure-Rust scalar engine). Validation splits
into:

- **integrity** — the session structurally makes sense (has iterations, counters
  are consistent, variance isn't wild).
- **deterministic** — the knobs governing determinism (seed, warmup, temperature)
  were pinned; loose ones are flagged.
- **reproducibility** — the archive records enough (engine, model, device) to be
  re-run and interpreted.
- **numerical** — a candidate engine's greedy token stream matches glproc's,
  reported as the longest matching prefix. glbench does not run inference here —
  the caller supplies both token streams (it runs engines through the adapter).

A run with an `Error`-severity finding fails validation; warnings are allowed.

## 8. Storage: files, not a database

STORAGE RULE: no database, no cloud sync, no unlimited history. A benchmark
archive is a single JSON file the user manages:

```
benchmarks/
├── qwen-glcuda-001.json
├── qwen-glproc-001.json
```

Every archive stamps `glbench_version` and `schema_version`. `storage::archive`
refuses to read a file whose schema is newer than the running build. Historical
trend (`comparison::trend`) is computed over whatever ordered set of files the
user hands in — there is no persistent store.

## 8b. Integrity and null semantics (v3)

Two properties were added to the archive in v3, both in service of §5's rule
that measurement stores facts:

- **Every `null` says why it is null.** An `availability` map keyed by dotted
  field path records the reason — measured, estimated, derived, unsupported,
  unavailable, not applicable, not observed, does not exist. A `null` with no
  entry fails the session *at write time*, which is what keeps the map from
  decaying into optional prose.
- **Every archive carries a content digest.** `sha256-128` over the archive's
  own canonical JSON, verified by default on read. It detects accidental
  modification. It is **not** a signature — anyone who can edit the file can
  recompute it — and the archive states that in its own `integrity.note` field
  rather than leaving a reader to assume otherwise.

Both are elaborations of the same idea §6 already states: be honest about what
you do not know, in the file itself and not only in the docs.

## 9. Dependencies: zero, on purpose

glbench adds **no** crates.io dependencies. The JSON reader/writer
(`export::json`), CSV writer, Markdown renderer, table layout, argument parser,
SHA-256 (via `glcore::hash`), and ASCII loss curve are all hand-rolled against
the standard library.

This survived v3 unchanged, and was measured rather than assumed: the workspace
went from 132 to **125** unique crates across the five waves — seven removed
(`sha2` and its transitive tree), none added. This keeps the data
model decoupled from any serialization framework and guarantees glbench builds
and runs offline. The workspace crates it depends on (`glcore`, `glproc`,
`glcuda`) are the engines it measures, not external dependencies.

## 10. One crate, internal modules

glbench is a single crate with internal module folders (`core/`, `analysis/`,
…), **not** a set of sub-crates (`glbench-core`, `glbench-analysis`). Splitting
into crates is allowed only when real architecture pressure requires it. Until
then, module boundaries carry the structure and compile times stay low.
