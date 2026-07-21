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

A/B two (or more) models under one identical workload, in one command — each
candidate is diffed against the first. Sequential on purpose: parallel decodes
would contend for the memory bus and corrupt every number:

```sh
glbench ab --engine glproc --model qwen2.5-0.5b-q8_0.gguf --model qwen2.5-0.5b-q4_k_m.gguf
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
| `--cot`         | auto    | thinking-model override (`on`/`off`); unset lets the GGUF header decide |
| `--out`         | —       | archive the session as JSON                      |

### What a report contains (v2)

Beyond throughput statistics, a report carries — each section only when its
inputs were actually measured, absent otherwise:

- **cold vs warm** — the first-ever iteration is timed separately (page-in /
  cache-fill cost), never mixed into the warm statistics.
- **roofline** — engine stage telemetry bucketed into attention / ffn /
  lm_head, each classified against the *measured* bandwidth ceiling
  (bandwidth-bound / not-bandwidth-bound / indeterminate).
- **behavior** — entropy (with a CoT-aware flag: low entropy on a
  thinking-capable model is expected, on a plain model it is an anomaly),
  repetition, perplexity, confidence, stall, and intra-session drift
  (per-quarter inter-token latency plus the worst OOD perplexity window).
- **hypotheses** — cross-signal root-cause patterns, phrased as what the data
  is *consistent with*, never as verdicts.
- **energy** — Joules/token via Linux RAPL (powercap sysfs) when readable;
  never estimated from TDP. Not available on Windows/macOS (RAPL needs a
  kernel driver there, and glbench will not pretend otherwise).

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

## Benchmarking GLLM

`--engine gllm` runs a `.gllm` package through `glictus-caliburni`'s
`GllmRuntime` + `GlprocBackend` (ARTX10). It needs the `gllm-bench` feature:

```sh
cargo build --release -p glbench --features gllm-bench
glbench run --engine gllm --model path/to/package/ --kind decode --tokens 128
```

Two things are different from every other engine, and both are load-bearing,
not bugs:

- **`--model` is a directory** (the package root — `gllm.json` plus its
  `GLLMShared.gllm` and `GLLMTensorLayer-*.gllm` files), not a single file.
- **`--prompt` cannot be encoded.** GLLM packages carry no tokenizer yet
  (ARTX1 OQ3's `GLLMTokenizer.gllm` unit is decided but not emitted by the
  converter). The `gllm` engine synthesizes deterministic token ids from
  `--seed` and the prompt's word count instead of real text — this measures
  real throughput and real per-layer computation, not prompt-conditioned
  generation quality. Fine for `run`/`ab`/`scale`; do not read behavior
  signals (entropy, perplexity, repetition) as if they described the model's
  reaction to your prompt — they describe its reaction to a synthetic one.
- **F32 tensors only** (`GlprocBackend`'s ARTX10 Wave 1 scope). A package
  converted from a quantized GGUF (Q4_K_M, Q8_0, ...) fails loudly with
  `Unsupported dtype` rather than computing wrong numbers — this is an
  honest limitation, not a bug to work around.

## Windows: Defender exclusions required

Windows Defender rescans large files on rebuild and on first `mmap` — this is
not a rare edge case on this project's own reference machine (i3-1115G4 /
Windows 11). Measured pollution: **2–4× on affected runs**, worse than most
real regressions glbench exists to catch. An un-excluded benchmark is not
noisy, it is *wrong*.

Add exclusions before benchmarking:

```powershell
Add-MpPreference -ExclusionPath "C:\path\to\gwenland-ai\target"
Add-MpPreference -ExclusionPath "C:\path\to\models"
Get-MpPreference | Select-Object -ExpandProperty ExclusionPath
```

Verify the exclusions are actually active before every session (the `Get-MpPreference`
line above). Never archive a result taken without them — a session file has no
field for "Defender was scanning during this run," so the number would look
identical to a clean one while being 2–4× off.

## The one rule

glbench observes. It may say *"performance is memory-bandwidth bound"* or
*"kernel launch overhead is significant."* It will never *"automatically rewrite
the CUDA kernel."* Optimization is the engine developer's job; measuring the
truth is glbench's.
