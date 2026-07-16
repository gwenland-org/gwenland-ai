# glbench — Roadmap

Phased plan and explicit non-goals. Priority: **build a clean foundation
first, do not over-engineer.**

---

## Status

| Phase | Scope                                                        | State |
|-------|--------------------------------------------------------------|-------|
| 1     | Crate skeleton, `BenchmarkSession`, workload + measurement schema | ✅ done |
| 2     | Benchmark runner, engine adapter, exporters (JSON/MD/CSV), storage | ✅ done |
| 3     | Analysis, comparison, validation subsystems                  | ✅ done |
| 4     | Advanced rendering                                           | 🚧 baseline in place |
| v2    | Mensura Veritatis v2: CoT-aware quality, per-bucket roofline, intra-session anomaly + hypotheses, cold/warm split, `ab` command, RAPL energy | ✅ done |

Phase 1–3 landed together as the first foundation; Phase 4 has a working text +
table renderer, with the richer output below still open.

## v2 — "LLM Performance Doctor" (landed)

The five v2 modules, mapped onto the existing single-crate layout rather than
the PRD's two-crate sketch (DESIGN.md §10: split only under real pressure):

- **Quality RCA** — `engine::model_probe` reads the GGUF header (arch, quant,
  thinking capability); `behavior::cot` flags low entropy as
  `LOW_ENTROPY_COT_EXPECTED` vs `LOW_ENTROPY_ANOMALY`. Override: `--cot on|off`.
- **Roofline** — `analysis::roofline::RooflineReport` buckets stage telemetry
  into attention/ffn/lm_head and classifies each against the measured ceiling.
- **Anomaly** — `behavior::anomaly` (per-quarter drift, OOD perplexity window)
  plus `analysis::hypothesis` (cross-signal root-cause patterns, phrased as
  "consistent with", never verdicts).
- **Hardware** — `environment::power` measures Joules/token via Linux RAPL
  (powercap sysfs, std-only). Cache counters / PCIe / Windows RAPL are out:
  not honestly reachable without kernel drivers or external deps.
- **Throughput** — cold first-run captured separately from warm stats;
  `glbench ab` benchmarks N models under one identical workload sequentially
  (parallel would contend for bandwidth and corrupt both numbers).

## Phase 4 and beyond — planned

- **Richer rendering.** Sparkline/bar throughput visualizations in the terminal;
  a self-contained HTML report export (still zero-dependency, inlined).
- **Scaling sweeps as a first-class command.** `glbench scale` over a set of
  prompt lengths / token budgets, driving `analysis::scaling` and rendering the
  curve. The analysis primitive exists; the CLI orchestration does not yet.
- **Roofline plot.** `analysis::roofline` computes arithmetic intensity and the
  ridge point; a textual/HTML roofline chart would make the memory-vs-compute
  verdict visual.
- **Numerical parity command.** Wire `validation::numerical` into a
  `glbench validate --against glproc` flow that runs both engines through the
  adapter and reports the matching-prefix length. The comparison primitive is
  done; the two-engine driver is the remaining piece.
- **Per-phase timeline capture.** `measurement::timeline` models prefill /
  decode / overhead; surfacing a per-token decode timeline would need an engine
  hook that streams per-token timestamps (an engine-side change, coordinated —
  not a glbench-only feature).
- **Engine coverage.** `glvulkan` / `glmetal` adapters — each is one match arm
  in `engine::adapter::build_engine` once those engines implement `GlEngine`.
- **Device capability table growth.** Extend `engine::capability` as the project
  validates on more hardware. Kept small and honest: a device absent from the
  table simply yields no ceiling.

## Non-goals (will not build)

These are out of scope by design, not by omission:

- **Kernel optimization / autotuning / rewriting.** glbench observes; engines
  optimize. This is the defining boundary.
- **Model modification** — no changing quantization, no editing weights, no
  GGUF→GLLM conversion (that is GwenLand AI's pipeline).
- **Hardware configuration** — no clock/power/affinity tuning.
- **Scheduling / multi-GPU management / engine routing.** glbench measures one
  engine at a time; choosing or orchestrating engines belongs to the runtime.
- **Runtime replacement.** glbench runs *through* glcore's `Runtime`; it does not
  reimplement inference.
- **A performance database or cloud sync.** Archives are user-managed files.
  Trend analysis reads whatever files it is given; there is no persistent store
  and no history service.
- **External dependencies.** No crates.io additions, no Python, no ML/CUDA/Vulkan
  SDKs, no cloud SDKs. glbench stays offline-capable and hand-rolls its
  serialization.
- **Duplicating inference logic.** Ever. The engine adapter is the only path to
  compute.
