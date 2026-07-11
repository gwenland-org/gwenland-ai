# glbench — Mensura Veritatis

A standalone benchmark execution and performance-analysis framework for
GwenLand AI. glbench measures the **truth** about engine performance.

```
Execute → Measure → Analyze → Compare → Validate → Report
```

**glbench is not an optimizer.** It observes performance; engine developers
optimize it. glbench never touches a kernel, a model file, or a hardware
setting — it runs inference through the existing `GlEngine` contract and reports
what the hardware did.

---

## Purpose

Answer, with auditable numbers, questions like:

- How fast does this engine decode / prefill this model on this machine?
- What fraction of the hardware's bandwidth ceiling are we actually using?
- Is decode memory-bound, compute-bound, or launch-overhead-bound?
- Did this change regress throughput versus the last archived run?
- Does the accelerated engine still match the glproc oracle token-for-token?

glbench produces a single `BenchmarkSession` — the source of truth — and renders
it to the terminal, JSON, Markdown, or CSV.

## Install / build

glbench is a workspace member. Build the CLI:

```sh
cargo build --release -p glbench
```

The binary is `glbench`. It has **zero external dependencies** — only the Rust
standard library and existing GwenLand workspace crates (`glcore`, `glproc`,
`glcuda`). It works fully offline; the only network access anywhere in the stack
is model fetching, which is GwenLand AI's job, not glbench's.

## Usage

Run a benchmark and print a report:

```sh
glbench run --engine glcuda --model qwen2.5-7b-q8_0.gguf
```

Run with an explicit workload and archive the session:

```sh
glbench run --engine glproc --model model.gguf \
    --prompt "Explain entropy." --tokens 128 \
    --warmup 1 --iters 5 --kind decode \
    --out benchmarks/qwen-glproc-001.json
```

Compare two archived runs (regression check at a 5% threshold by default):

```sh
glbench compare benchmarks/qwen-glcuda-001.json benchmarks/qwen-glproc-001.json
```

Re-render an archive, or convert it:

```sh
glbench inspect benchmarks/qwen-glcuda-001.json
glbench export  benchmarks/qwen-glcuda-001.json --format md  --out report.md
glbench export  benchmarks/qwen-glcuda-001.json --format csv --out runs.csv
```

### `run` flags

| flag            | default | meaning                                          |
|-----------------|---------|--------------------------------------------------|
| `--engine`      | glproc  | engine to run through (`glproc`, `glcuda`)       |
| `--model`       | —       | path to a `.gguf` / `.safetensors` (**required**)|
| `--prompt`      | builtin | prompt text (default is long enough for prefill) |
| `--tokens`      | 128     | tokens to generate in the measured decode phase  |
| `--warmup`      | 1       | untimed warmup iterations                        |
| `--iters`       | 3       | timed measured iterations (feeds the statistics) |
| `--temperature` | 0.0     | sampling temperature (0 = greedy, deterministic) |
| `--seed`        | 42      | RNG seed for deterministic sampling              |
| `--kind`        | end_to_end | `prefill`, `decode`, `end_to_end`, `stress`   |
| `--out`         | —       | archive the session as JSON                      |

## Architecture

The `BenchmarkSession` is a **pure data model** — every subsystem reads or fills
one of its fields, and every renderer consumes it.

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

Module map (one crate, internal module folders — no sub-crates):

| module         | responsibility                                             |
|----------------|------------------------------------------------------------|
| `core`         | the data model (session, metrics, workload, schema)        |
| `environment`  | probe the machine (std + OS files only)                    |
| `engine`       | the **only** boundary to the engines; runs via `Runtime`   |
| `runner`       | orchestrate a run: warmup → measured iterations → phases   |
| `measurement`  | store raw facts, convert counts+durations to rates         |
| `analysis`     | facts → insight, always as recommendations, never actions  |
| `comparison`   | run/engine/quant/hardware deltas, regression, trend, stats |
| `validation`   | integrity, determinism, numerical parity vs glproc oracle  |
| `export`       | hand-rolled JSON / Markdown / CSV                          |
| `render`       | terminal text + tables                                     |
| `storage`      | user-managed archive files (no database)                   |

See [`DESIGN.md`](DESIGN.md) for the responsibility boundaries and data flow,
and [`ROADMAP.md`](ROADMAP.md) for planned features and non-goals.

## The one rule

glbench observes. It may say *"performance is memory-bandwidth bound"* or
*"kernel launch overhead is significant."* It will never *"automatically rewrite
the CUDA kernel."* Optimization is the engine developer's job; measuring the
truth is glbench's.
