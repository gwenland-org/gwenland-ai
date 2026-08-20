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
`glcuda`, and optionally `glictus-caliburni` behind `--features gllm-bench`,
see [Benchmarking GLLM](#benchmarking-gllm)).

That's glbench's own charter rule, and it's stricter than GwenLand's
project-wide baseline. The project-wide rule is **Zero ML Dependency** — no
torch/candle/ort/ggml, anywhere — which still leaves room for things like an
OS mmap wrapper or a CLI parser elsewhere in the tree. glbench goes further
on purpose: it adds no crates.io dependency at all, ML or otherwise, so its
JSON/CSV/Markdown export stays hand-rolled and decoupled from any
serialization framework (see [`DESIGN.md`](DESIGN.md) §9). It works fully
offline; the only network access anywhere in the stack is model fetching,
which is GwenLand AI's job, not glbench's.

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

Check numerical parity against an oracle engine (default `glproc`) — runs
both under forced greedy decoding on the identical prompt and reports the
matching token-id prefix length; exits non-zero on divergence, so it composes
as a CI gate:

```sh
glbench validate --engine glcuda --model model.gguf --against glproc
```

Sweep decode throughput across a set of token budgets and classify how it
scales (linear / sub-linear / saturating), sequentially — same
bandwidth-contention reasoning as `ab`:

```sh
glbench scale --engine glproc --model model.gguf --sweep 32,64,128,256,512
```

Sweep decode throughput across `glproc`'s own thread-pool sizes (via
`GLPROC_THREADS`) and report speedup/efficiency relative to the lowest
thread count — glproc-only, since other engines have no equivalent knob:

```sh
glbench thread-scale --engine glproc --model model.gguf --sweep 1,2,4,8
```

Decode every tensor in a `.gllm` package, flagging NaN/Inf/zero-variance
(`gllm-bench` feature); `--full` adds a per-tensor mean/std/min/max
distribution, `--norm-only` restricts the scan to RMSNorm gamma weights:

```sh
glbench tensor-stats --model path/to/package/ --full --norm-only
```

Join a `run` archive's throughput with a `kl-div`/`ppl` archive's
numerical-accuracy figures into one side-by-side view — no new measurement,
both archives must already exist:

```sh
glbench accuracy-vs-perf run.json kl-div.json
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
| `--engine`      | glproc  | engine to run through (`glproc`, `glcuda`, `gllm`) |
| `--model`       | —       | path to a `.gguf` / `.safetensors`, or (for `gllm`) a package directory (**required**) |
| `--prompt`      | builtin | prompt text (default is long enough for prefill) |
| `--tokens`      | 128     | tokens to generate in the measured decode phase  |
| `--cold-iters`  | 5       | timed cold-start iterations, run before warmup, each individually recorded (see [Interpreting Results](#interpreting-results)) |
| `--warmup`      | 1       | untimed warmup iterations                        |
| `--iters`       | 3       | timed measured iterations (feeds the statistics) |
| `--temperature` | 0.0     | sampling temperature (0 = greedy, deterministic) |
| `--seed`        | 42      | RNG seed for deterministic sampling              |
| `--kind`        | end_to_end | `prefill`, `decode`, `end_to_end`, `stress`   |
| `--cot`         | auto    | thinking-model override (`on`/`off`); unset lets the GGUF header decide |
| `--verify-against` | unset | oracle engine to auto cross-check the first 50 tokens against, folded into the validation report; skipped (not just trivial) when equal to `--engine` |
| `--out`         | —       | archive the session as JSON                      |

`validate` additionally takes `--against <oracle>` (default `glproc`); `scale`
additionally takes `--sweep N,N,N,...` (at least two token budgets) in place
of `--tokens`. Both otherwise accept the same `--engine`/`--model`/`--prompt`
flags as `run`.

### What a report contains (v2)

Beyond throughput statistics, a report carries — each section only when its
inputs were actually measured, absent otherwise:

- **cold vs warm** — a dedicated cold-start phase (`--cold-iters`, default 5)
  runs before warmup, every iteration individually timed and reported as
  median + range, never mixed into the warm statistics (see
  [Interpreting Results](#interpreting-results)).
- **roofline** — engine stage telemetry bucketed into attention / ffn /
  lm_head, each classified against the *measured* bandwidth ceiling
  (bandwidth-bound / not-bandwidth-bound / indeterminate).
- **behavior** — every signal is computed from facts the engine already
  produced (token ids, raw per-token distributions, stage timings) and is a
  number, never a verdict:
  - `repetition` — n-gram reuse in the output (needs only token ids).
  - `entropy` — per-step distribution uncertainty, with a CoT-aware flag
    (low entropy on a thinking-capable model is expected; on a plain model
    it is an anomaly).
  - `stall` — inter-token latency spikes.
  - `ood` — perplexity, and its gap vs. a baseline.
  - `hallucination` — confidence/rank divergence, a **proxy** for
    confabulation, not a detector: a model can be confidently wrong (low
    divergence, false statement) or uncertain and right (high divergence,
    true statement). Read it as "how sure was the model," never as "how
    much did it make up."
  - `anomaly` — intra-session drift: per-quarter inter-token latency plus
    the worst OOD perplexity window, by position.
  - `performance` / `drift` — ms/call, share, layer variance, and Δ ms/call
    between sessions, from engine telemetry rather than traces.
  - **`toxicity` is deliberately not implemented** (see
    `src/behavior/toxicity.rs`): every metric considered measures affinity
    to a flagged word list, not toxicity, and is wrong in both directions
    (a model discussing "carcinoma" or "exploit" in a legitimate medical/
    security context scores high for saying nothing wrong; implicit bias
    or confident misinformation in ordinary vocabulary scores zero). A
    profiler number carries authority it would not have earned here.
- **hypotheses** — cross-signal root-cause patterns, phrased as what the data
  is *consistent with*, never as verdicts.
- **energy** — Joules/token via Linux RAPL (powercap sysfs) when readable;
  never estimated from TDP. Not available on Windows/macOS (RAPL needs a
  kernel driver there, and glbench will not pretend otherwise).
- **parity** (`validate` only) — matching-prefix token length against the
  oracle engine under forced-greedy decoding, plus a pass/fail exit code.
- **scaling** (`scale` only) — `linear` / `sub-linear` / `saturating` /
  `insufficient data`, classified from the sweep's decode-tps-vs-tokens
  slope.

## Training observation (v3, `--features train-bench`)

```bash
cargo build --release -p glbench --features train-bench

# Observe a LoRA fine-tune on stumman and archive it.
glbench train --d-in 64 --d-out 64 --rank 4 --samples 32 --epochs 8               --target-loss 0.5 --step-sample 8               --bit-scope gradients,optimizer --out train.json

# Same, with inference roles labelled either side of the run.
glbench unified --d-in 64 --d-out 64 --rank 4 --samples 32 --epochs 8

# Per-step rows for a spreadsheet.
glbench export train.json --format training-csv
```

**glbench does not drive training.** It builds a `Trainer`, installs an
observer, and calls stumman's own `Trainer::train`. It never calls `train_step`
in a loop of its own, never touches optimizer state, and never writes a
parameter — the same measuring/doing boundary the inference side keeps.

**There is no `--model` or `--dataset`, deliberately.** stumman M2 generates its
frozen base weight from `--seed` and builds its dataset in memory from
`--samples`/`--dataset-seed`; neither flag has a subject, so both are rejected
with an error that explains why rather than reporting an unknown flag. The shape
and seed flags fully determine the run, which makes it reproducible in a way a
path would not be.

**`--target-loss` has no default.** Time-to-target needs someone to say what
good means for their model and their data. Without it, `steps_to_target` is
archived as absent (`not_applicable`) rather than guessed; a target that *was*
given and never reached is `not_observed`, which is a different claim.

**`--step-sample N` matters more than it looks.** Observer overhead grows with
layer width — measured at +4.3% (64×64), +11.5% (256×256), +14.2% (512×512) —
and bit profiles follow the same `N`. On a 192-step run with both training bit
scopes on, `--step-sample 16` took the archive from 1.21 MB to 83.5 KB.

## Interpreting Results

A report is a lot of numbers. Here is what the ones people misread most
often actually mean.

### `ceiling_efficiency` — what fraction of what, exactly?

`ceiling_efficiency` is `observed decode tok/s ÷ theoretical ceiling tok/s`,
where the ceiling is `peak bandwidth ÷ model weight bytes` (decode is
memory-bandwidth-bound: every token streams the full weight set once). It
always carries a `ceiling_basis`, and the basis changes what the number is
worth:

- **`measured`** — the bandwidth is this machine's own sustained sequential
  read throughput, probed at session start (or, on GPU, an on-device
  measurement if the engine ever reports one). This is the honest ceiling:
  it already reflects this exact machine's RAM, channel count, and thermal
  state at the time of the run. An efficiency of 85% here really means "85%
  of what this box can actually do."
- **`estimated_from_table`** — the bandwidth came from a GPU vendor's
  published spec sheet (`engine::capability`'s static table), because no
  measurement was available. Real devices commonly sustain only 60-85% of
  their advertised peak even on a fully optimized kernel — so an efficiency
  computed against this basis systematically *understates* how well-utilized
  the device actually is. Read a 70% figure here as "at least 70%, possibly
  much closer to 100% of what this device can really sustain," not as a
  precise fraction.
- **`undetermined`** — no ceiling exists at all (unrecognized GPU, no CPU
  bandwidth probe result). `ceiling_efficiency` is `null`; there is nothing
  to interpret.

### `stall_count` / `jitter` — the behavior signals for a rough decode loop

`stall_count` counts inter-token gaps that exceeded 3× the *median* gap —
deliberately the median, not the mean, because one 900ms block among ninety
30ms tokens drags the mean up without moving the median, and `stall_count`
is exactly the number that catches what the mean hides. `jitter` is the
coefficient of variation (`std_dev / mean`) of those same gaps — a
scale-free "how rough was it" figure, so a slow-but-steady engine and a
fast-but-steady one are comparable on this axis even though their raw
latencies are not. Neither number attributes a *cause* (page fault, thermal
throttle, scheduler preemption, and a growing allocator all look identical
in the timing alone) — they flag that something happened, not what.

### Why cold-start gets its own phase, with a range, not a mean

The very first requests against a freshly loaded model pay costs the warm
statistics deliberately exclude: page-ins, cold caches, (on GPU) PTX JIT and
graph capture. `--cold-iters` (default 5) runs that many iterations before
warmup even begins, and reports them as **median + min/max**, never a single
number. The reason is the same as `stall_count`'s: "this model always pays
~500ms to page in" and "it usually pays 90ms but once paid 900ms because the
OS scheduled something else" are two very different facts, and a mean cannot
tell them apart — only the range can. If the cold-start range is wide, that
is itself the finding, not noise to average away.

### `±95% CI` — how much to trust the mean

Past 3 measured iterations, every throughput table carries a 95% confidence
interval for the mean (`mean ± ci95`, a Student's-t interval using the
*sample* standard deviation — the small-`n`-correct one, not the population
form the rest of the table uses). Below 3 iterations it reads `n/a`, on
purpose: a confidence interval on 1-2 points has 0-1 degrees of freedom and
is not a number worth printing. A wide interval relative to the mean is the
same signal as high `jitter` from a different angle — it means "run more
iterations before treating this mean as a fact," not "something is broken."

### The `--verify-against` cross-check

`--verify-against <oracle>` (e.g. `--verify-against glproc`) loads a second
engine and compares the first 50 generated tokens against it under forced
greedy decoding, folding the result into the validation report as a `parity`
finding. It is opt-in, not automatic, because it roughly doubles the cost of
the run (a second full engine load and inference pass) — turn it on when
you actually need to know "does this accelerated backend still agree with
the reference implementation," not on every routine throughput check. It is
silently skipped, not run, when the oracle equals `--engine`: comparing an
engine's output to itself always matches and would report a check that
verified nothing.

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
| `engine`       | the **only** boundary to the engines; runs via `Runtime` (`glproc`/`glcuda`) or directly against a `Box<dyn GlEngine>` (`gllm` — see [Benchmarking GLLM](#benchmarking-gllm)) |
| `runner`       | orchestrate a run: warmup → measured iterations → phases   |
| `measurement`  | store raw facts, convert counts+durations to rates         |
| `behavior`     | per-token signals from traces: repetition, entropy, stall, ood, hallucination (proxy), anomaly, cot, drift, performance |
| `analysis`     | facts → insight, always as recommendations, never actions  |
| `comparison`   | run/engine/quant/hardware deltas, regression, trend, stats |
| `validation`   | integrity, determinism, numerical parity vs glproc oracle  |
| `export`       | hand-rolled JSON / Markdown / CSV                          |
| `render`       | terminal text + tables                                     |
| `storage`      | user-managed archive files (no database)                   |
| `quant_info`   | static `.gllm` manifest dtype tally, no inference (own hand-rolled JSON reader, not glictus-caliburni's) |
| `ppl`          | WikiText-2 perplexity via the `.gllm` runtime (behind `gllm-bench`) |
| `kl_divergence` | per-position KL-divergence between a `.gllm` package and the `glproc` oracle, teacher-forced (behind `gllm-bench`) |
| `tensor_stats` | decodes every tensor in a `.gllm` package and flags NaN/Inf/zero-variance (behind `gllm-bench`) |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for the crate layout, entry-point →
module map, and dependency direction; [`DESIGN.md`](DESIGN.md) for the
responsibility boundaries and data flow; [`ROADMAP.md`](ROADMAP.md) for
planned features and non-goals.

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
