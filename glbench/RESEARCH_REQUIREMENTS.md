# Mensura Veritatis — Feature Expansion & Research Requirements

## Primary Objective

Before implementing any new benchmark, profiler, analyzer, metric, diagnostic, report, or visualization, conduct thorough research.

Implementation must never begin from assumptions.

It must begin from evidence.

---

## Research Requirements (Mandatory)

Before proposing or implementing a feature, research the current state of AI inference benchmarking and performance engineering.

Investigate:

- Modern AI inference benchmarks
- LLM observability tools
- Runtime profilers
- Performance engineering practices
- AI infrastructure monitoring
- CUDA / Vulkan / Metal profiling techniques
- CPU optimization methodologies
- Memory bandwidth analysis
- Numerical validation methods
- Existing industry tools and their limitations
- Academic literature relevant to inference benchmarking
- Industry needs and engineering workflows as of 2026

Every proposed feature must answer:

1. What engineering problem does this solve?
2. Who benefits from this metric?
3. Is this used in production or research?
4. How is it calculated?
5. Can it be reproduced?
6. Does it provide actionable insight?
7. Is the implementation lightweight?
8. Does it align with Mensura Veritatis philosophy?

If the answer is unclear, the feature should not be implemented.

**Before implementing, also check whether the feature (or something close to it)
already exists** — cross-reference `README.md`'s feature list and
`ARCHITECTURE.md`'s module map first. Building a second, independent
implementation of something glbench already has is exactly the failure mode
`architecture/mensura-veritatis-v3/ARTX2-Quant.md` documents in the engines
themselves (two implementations of the same thing, silently disagreeing) —
the same discipline applies here.

---

## Design Philosophy

Mensura Veritatis is **not** an optimizer.

Mensura Veritatis is **not** an auto-tuner.

Mensura Veritatis is **not** a model editor.

Mensura Veritatis is a **read-only observability framework**.

Its responsibility is to:

- Measure
- Observe
- Analyze
- Validate
- Diagnose
- Report

Nothing more.

Nothing less.

(This restates `DESIGN.md` §1's responsibility boundary — see that document
for the enforcement mechanism, `analysis::bottleneck`'s recommendation-only
contract.)

---

## Read-Only Rule

Mensura Veritatis must never modify:

- Model weights
- Runtime configuration
- Engine behavior
- Kernel implementation
- Quantization
- Memory allocation strategy
- Execution order

The benchmark observes reality.

It never changes reality.

---

## Candidate Features

Status column tracks whether the candidate has been vetted against the 8
questions above and, if accepted, implemented. Update it as work lands —
this list is a backlog, not a changelog; once a feature ships, its detail
belongs in `README.md`, not here.

### Core Benchmark

| Feature | Status |
|---|---|
| Model loading benchmark | done — `--cold-iters` |
| Cold vs warm startup | done — see README "Why cold-start gets its own phase" |
| Prefill throughput | done — `--kind prefill` |
| Decode throughput | done — `--kind decode` |
| End-to-end latency | done — `--kind end_to_end` |
| Token latency (P50 / P95 / P99) | done — `behavior::StallSignal` already had p50/p99 (inter-token gaps, the same "token latency" data under a different name); added p95 (2026-07-24) to complete the conventional trio. Caught mid-research: this was nearly re-implemented as a whole new module before the required existing-feature check found it. |
| Peak memory usage | done — `MeasurementSet::peak_memory_bytes` existed in the schema (JSON export, round-trip tests) since before this pass but nothing ever populated it; now filled from the OS-tracked RSS high-water-mark (`/proc/self/status` VmHWM on Linux, `wmic process ... get PeakWorkingSetSize` on Windows) and rendered in text/markdown output for the first time. Found and fixed a real bug during vetting: `PeakWorkingSetSize` is in **kilobytes**, unlike `WorkingSetSize`'s bytes — an undocumented-enough WMI inconsistency that the first implementation got wrong by 1024x, caught only by cross-checking against `Get-Process`'s .NET-documented-bytes fields on a real PID |
| Average memory usage | deliberately not built as a separate time-averaged figure — would need a background sampling thread (real complexity + overhead) for a number peak/before/after already mostly substitutes for on this crate's typical load-once-then-run workload shape; see `measurement::memory`'s module doc comment for the full reasoning |
| CPU utilization | done — `measurement::cpu::process_cpu_time`, bracketed around the measured phase the same way as the energy meter (before/after, no sampling thread), divided by wall-clock and logical-core count into a percentage. Linux reads `/proc/self/stat` utime+stime under the standard but locally-unverified `USER_HZ = 100` assumption (flagged honestly in the module doc, unlike every other OS fact in this crate which is independently confirmed). Windows reads `wmic ... KernelModeTime,UserModeTime`; **empirically verified** (not just trusted from docs, after the `PeakWorkingSetSize` KB surprise above) by comparing a controlled busy-loop's wmic reading against `Get-Process`'s independently-computed `TotalProcessorTime.TotalSeconds` for the same PID — converged on 10,000,000 units/sec, confirming the documented 100ns unit was correct this time. Real E2E run on Qwen2.5-0.5B/glproc: 38.0% across 4 logical cores, a plausible reading given this model's known single/low-thread decode path. |
| GPU utilization | partial — glcuda self-reports at init; no independent probe |
| Power consumption (where available) | done — RAPL, Linux-only, honestly scoped |
| Export (JSON / CSV / Markdown) | done |

### Runtime Profiler

| Feature | Status |
|---|---|
| Layer execution breakdown | done — roofline bucketing (attention/ffn/lm_head) |
| Kernel execution breakdown | partial — bucket-level, not per-kernel |
| Operator benchmark | done — found already built, not previously reflected here: `glcore::telemetry::PhaseProfile`'s per-stage table (7 stages in practice: qkv/attention/attn_out/ffn_gate_up/ffn_down/lm_head/sampler, each with ms/share/ms-per-call/GB/s/GMAC/s), rendered in `render::text::telemetry` and `export::markdown`. Bucket-level, same granularity caveat as "kernel execution breakdown" above — not per-individual-kernel. **Real bug found and fixed while verifying this row (2026-07-24):** `glproc::runner::Runner` only collects this data when the `GLPROC_PROFILE` env var is set, and glbench never set it — so this entire section, and the roofline section built on top of it, had been silently absent from every `glbench run` since telemetry was first built, unless a caller happened to already know a glproc-internal env var name. Fixed in `engine::adapter::EngineAdapter::load` (`enable_engine_profiling`, sets it once per process, respects an explicit override). Confirmed live: a real `glbench run` now prints the full backend/timeline/memory section and roofline classification that never appeared before. |
| Timeline analysis | done — same telemetry table; the code has called this "timeline" since it was built (`"{label} timeline ({total_ms} ms total)"`). Honestly scoped as an **aggregate breakdown** (total ms / share / calls per stage), not a per-call chronological trace — a true Gantt-style timeline would need a timestamp per individual kernel launch, which is a heavier instrument this crate has consistently avoided (matches the no-sampling-thread philosophy in `measurement::memory`/`measurement::cpu`). Also unblocked by the `GLPROC_PROFILE` fix above. |
| ASCII flame graph | done (2026-07-24) — `render::flamegraph`, a proportional-width ASCII bar chart built from the *existing* stage-share telemetry (no new measurement, pure re-render of `PhaseProfile.hotspots()`). Wired into `render::text` and `export::markdown` (as a fenced code block). Verified live on Qwen2.5-0.5B/glproc: decode dominated by `ffn_gate_up` (36.1%) down to `attention` (0.6%), bars proportional and correctly ordered. Real archive: `benchmarks/qwen2.5-0.5b-glproc-004.json` (`inspect` it, or `run` again with the same flags, to see the flame graph — it's a `run`-time-only section, not preserved by the JSON archive round-trip, see `core::session::BenchmarkSession::from_json`'s doc comment). |
| Memory bandwidth utilization | done — `ceiling_efficiency` |
| Cache statistics | **rejected** (2026-07-24) — vetted and declined. L1/L2/L3 *miss-rate* counters (the actually useful profiling signal) require `perf_event_open` (Linux, needs `CAP_PERFMON`/non-default `perf_event_paranoid`) or ETW/Intel-PCM-style kernel drivers (Windows, needs admin) — both violate the zero-dependency, no-elevated-privileges bar every other probe in this crate holds to. Confirmed empirically: `wmic path Win32_CacheMemory` and `Win32_PerfRawData_PerfOS_Processor` expose only **static cache size** (L2 96 KB / L3 6 MB on this machine), never hit/miss counts — not the same thing as "cache statistics." Not implemented. |
| Occupancy estimation | **rejected** (2026-07-24) — vetted and declined. CUDA SM-occupancy (warps/SM, registers/thread) has no meaning outside a CUDA device, and this development machine has no CUDA-capable GPU (confirmed via `Win32_VideoController`: Intel UHD integrated graphics only) — fails question 5 ("can it be reproduced?") outright, since the feature could not even be exercised, let alone verified, here. Deferred until real GPU test hardware is available for glcuda. |
| Thread scaling | done (2026-07-24) — `glbench thread-scale`, glproc-only (uses the engine's existing `GLPROC_THREADS` env override, reloading the engine once per sweep point — no new engine API needed). Honestly scoped as engine-limited, same pattern as RAPL being Linux-only. Real sweep on Qwen2.5-0.5B (1/2/4 threads on this 4-logical-core machine): sub-linear, 100% -> 81-83% -> 44-46% scaling efficiency across two separate runs — consistent with the already-documented finding that this model's 2 KV heads limit how much threading helps (see the attention-fix memory note). `thread-scale` has no `--out`/JSON archive (same as the existing `scale` command it mirrors); real terminal output saved at `benchmarks/qwen2.5-0.5b-glproc-thread-scale.txt`. |
| NUMA awareness | **rejected** (2026-07-24) — vetted and declined. Confirmed via `wmic computersystem`: this machine reports `NumberOfProcessors=1` (single socket, 4 logical / 2 physical cores) — there is no second NUMA node to observe a cross-node penalty against, so the feature's entire premise is unverifiable on the only available hardware. Also no zero-dependency OS path: Windows NUMA topology needs the `GetNumaHighestNodeNumber` kernel32 API, not reachable via a `wmic` subprocess. Not implemented. |
| Arithmetic intensity | done — feeds roofline bottleneck classification |
| Compute-bound vs memory-bound analysis | done — `analysis::bottleneck` |

### Model Analysis

| Feature | Status |
|---|---|
| Tensor statistics | done — `glbench tensor-stats` (`gllm-bench`): decodes every real tensor, flags NaN/Inf/zero-variance. Real run on Qwen2.5-0.5B GQ4A_CPP: 291/291 scanned, clean. Deliberately does NOT flag by magnitude — no principled threshold without a calibration baseline, see the module's own doc comment |
| Weight distribution | done (2026-07-24) — `tensor-stats --full` surfaces the mean/std/min/max already computed internally per tensor (previously only fed the 3 flagged conditions, now exported in JSON alongside them). Real run on Qwen2.5-0.5B GQ4A_CPP: 291/291 tensors, full distribution written to JSON, e.g. `output_head.weight` mean 5.06e-5, std 0.0153, min -0.201, max 0.166. Real archive: `benchmarks/qwen2.5-0.5b-gq4a-tensor-stats-full.json`. |
| Quantization statistics | partial — see `quant-info`, no error/PPL attribution per format (see `architecture/mensura-veritatis-v3/ARTX4-Benchmark.md`) |
| Outlier detection | partial — see `tensor-stats`; NaN/Inf/zero-variance only, magnitude-based outliers explicitly deferred (needs a baseline, see above) |
| Hidden-state norm analysis | **rejected / deferred** (2026-07-24) — vetted and declined for this pass. Needs new per-layer activation-capture instrumentation inside glproc's forward pass; nothing like this exists today — `glcore::trace::TokenTrace` (the only existing trace hook) captures facts derived from the *final* logits only (token id, logprob, rank, entropy, top-prob), never intermediate layer activations. `diff_dump.rs` (glictus-caliburni) does something similar ad hoc, but as a one-off debug dump, not a reusable engine API. Comparable in scope to the `score_sequence` engine surgery KL-divergence needed, but deeper (every layer, not just the head) — flagged as a real future engine-instrumentation project, not a lightweight glbench-only task (fails question 7). |
| RMSNorm analysis | reframed and done (2026-07-24) as the **static** variant — `tensor-stats --norm-only` filters to `*norm.weight` (gamma) tensors specifically and reports their own distribution summary, reusing the existing tensor-decode infra (no new engine hook). The **runtime/activation** variant (post-normalization activation statistics during a forward pass) carries the identical instrumentation blocker as hidden-state norm analysis above and is deferred for the same reason. Real run on Qwen2.5-0.5B: 49/49 norm tensors (24 layers × `attn_norm`+`ffn_norm`, plus `output_norm`), clean, gamma magnitude visibly growing with depth (`attn_norm` mean ~0.03 at layer 0 vs ~2.4 at layer 23) — a real, plausible pattern, not a red flag. Real archive: `benchmarks/qwen2.5-0.5b-gq4a-tensor-stats-norm.json`. |
| Attention statistics | **rejected / deferred** (2026-07-24) — vetted and declined for this pass. Needs post-softmax attention-weight capture per head per layer — an even heavier hook than hidden-state capture (O(seq²) per head, transient per position), with no existing scaffolding to build from. Same recommendation as hidden-state norm analysis: a real engine-instrumentation project, not in scope here. |
| MoE routing analysis | done — found already fully built end-to-end, not previously reflected here: `glcore::telemetry::MoeTelemetry` (num_experts, top_k, moe_layers, expert_load counters, load balance, routing entropy), populated live in `glproc::runner::record_moe` during generation, rendered in `render::text::telemetry`. Not re-verified E2E this session — none of the locally available GGUF files (Qwen2.5-0.5B, Qwen2.5-1.5B, Qwen3-1.7B) are MoE architectures — but `[[project_glproc_moe_status]]`-equivalent prior verification exists at Qwen3 MoE scale (128 experts, top-8). |
| KV-cache analysis | done (2026-07-24) — the raw size (`kv_cache_bytes`, statically computed from model config) was already rendered; added the genuinely missing piece, a **memory-risk validation finding**: `kv_cache_bytes + model_bytes` compared against the machine's available RAM, warning when the model's own footprint plus its configured KV cache would not fit — directly matching the documented "KV cache is the memory trap" finding from the ARTX05 runtime work, where an unclamped context length can silently demand more RAM than the machine has. Real run on Qwen2.5-0.5B: model 0.62 GiB + KV cache 0.09 GiB against 2.4 GiB free — comfortably clear, correctly produced no finding; the error/warning thresholds are unit-tested directly (real hardware was never going to hit an OOM condition on demand). Real archive: `benchmarks/qwen2.5-0.5b-glproc-004.json`'s `validation` block. |

### Numerical Validation

| Feature | Status |
|---|---|
| Numerical parity | done — `validate`, token-id-prefix matching only |
| KL-divergence vs oracle logits | done — `glbench kl-div`, GLLM-only (`gllm-bench`) so far; real run on Qwen2.5-0.5B GQ4A_CPP: mean 0.999 nats, max 7.52 nats at position 0 — a real, nonzero residual divergence, confirming the qualitative "slightly derailed" E2E text observed after the Q6_K fix. Real archive: `benchmarks/qwen2.5-0.5b-gq4a-kl-div.json` |
| Determinism verification | done — `validation::deterministic` |
| Regression detection | done — `compare --threshold` |
| Stability analysis | done — CI95 on throughput means |
| Repeated benchmark statistics | done |
| Accuracy vs performance comparison | done (2026-07-24) — `glbench accuracy-vs-perf <run.json> <accuracy.json>`, a pure report-level combinator: no new measurement, joins an existing `run` session's throughput with an existing `kl-div`/`ppl` session's numerical-accuracy figures into one side-by-side view (auto-detects which shape the second file is). Exists precisely because the two kinds of session were previously only ever viewable separately. Real join on Qwen2.5-0.5B, both recognized accuracy shapes verified live (not just unit-tested): decode/prefill tok/s alongside KL-divergence mean/max nats, and separately alongside perplexity over evaluated tokens — both runs correctly flagged the same real model-path mismatch when the two archives were deliberately given differently-named model paths, rather than silently joining unrelated runs. `accuracy-vs-perf` has no `--out` (a pure terminal report); real output saved at `benchmarks/qwen2.5-0.5b-accuracy-vs-perf-kldiv.txt` (joining `benchmarks/qwen2.5-0.5b-glproc-004.json` against `benchmarks/qwen2.5-0.5b-gq4a-kl-div.json`) and `benchmarks/qwen2.5-0.5b-accuracy-vs-perf-ppl.txt` (same run archive against `benchmarks/qwen2.5-0.5b-gq4a-ppl.json`, perplexity 52.924 over 768 evaluated tokens). |

### Diagnostics

Every diagnosis must include:

- Supporting evidence
- Confidence score
- Root cause analysis
- Performance impact estimation
- Suggested optimization target

Recommendations must never be speculative.

Every recommendation must originate from measurable evidence.

**Open question, not yet resolved**: a "confidence score" that isn't itself
derived from something measurable (sample size, CI width, agreement across
repeated runs) would violate `DESIGN.md` §5 ("measurement stores facts, not
conclusions") — any diagnostics feature adding a confidence score must define
its derivation before it's accepted, not leave it as a heuristic label.

### Reporting

| Feature | Status |
|---|---|
| Markdown report | done |
| JSON report | done |
| CSV report | done |
| Human-readable summary | done |
| Machine-readable output | done |
| CI/CD compatible artifacts | done — `validate`'s non-zero exit on divergence |

---

## Feature Acceptance Criteria

A feature may only be accepted if:

- It solves a real engineering problem.
- It is useful in production or research.
- It produces reproducible measurements.
- It maintains low runtime overhead.
- It does not mutate the benchmark target.
- It follows the read-only principle.
- It remains consistent with the philosophy of Mensura Veritatis.
- It does not duplicate an existing module without a stated reason (see the
  cross-reference note under Research Requirements above).

Otherwise, reject the feature.

---

## Guiding Principle

> **You can't reach the truth without the real numbers.**

Measure first.

Understand second.

Optimize elsewhere.
