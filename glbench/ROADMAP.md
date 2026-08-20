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
| v3    | Mensura Veritatis v3: schema v2 envelope, null semantics, content digest, join, GLBitProf, training observation | ✅ done (5 waves) |

Phase 1–3 landed together as the first foundation; Phase 4 has a working text +
table renderer, with the richer output below still open.

## v3 — "Mensura Veritatis v3" (landed, 5 waves)

Design: `architecture/glbench-v3/DESIGN.md` (not this directory's `DESIGN.md`,
which is v1/v2 and has no D-/F- numbering).

| Wave | Scope | Gate result |
|---|---|---|
| 1 | Schema v2 envelope, null semantics (D-09/D-10), `sha256-128` content digest, `glbench join` | 1169 → 1270 tests; **7 external crates removed, 0 added** |
| 2 | GLBitProf math, weights scope, bit-profile divergence | 1270 → 1306; cost measured; **negative result recorded** (below) |
| 3 | stumman observer hook (`StepObserver`, phase timing, `VLGradStore::iter`) | stumman 327 tests; overhead measured at 3 layer widths |
| 4 | `glbench train` / `unified`, convergence, attribution, memory, adapter, gradient+optimizer bit scopes | `--features train-bench` 334 → 417 |
| 5 | Loss curve, training flame graph, Markdown/CSV training sections, docs | this table |

### What v3 measured, and what it found

Every number below was produced by a command in this repo, not estimated.

- **GLBitProf cost** (`examples/bench_bitprof.rs`, i3-1115G4, release, median
  of 10): Tier 1+3 — the exponent histogram and 32 per-position bit counts that
  *every* tensor pays — is **~27 ns/element**, flat from 1M to 16M elements.
  Tier 2, the sparse mantissa map, is paid only under the cap and multiplies
  that by **4.59×** to ~123 ns/element. Reported as two numbers because their
  average (~75) is a figure no tensor actually incurs.
- **D-12's birthday-bound premise, measured rather than cited**: 131,072
  near-uniform elements give **128,132** distinct mantissa patterns against the
  `m·(1 − e^(−n/m))` prediction of 130,053 — **ratio 0.985**.
- **⛔ Bit-profile divergence cannot detect a permutation.** The design expected
  a wrong-nibble-order defect (the Q6_K class) to show as a structured
  per-position anomaly. Measured with the real GQ4A encoder and dequant kernel
  (`examples/bitprof_quant_divergence.rs`): a correct decode and a
  nibble-swapped one score **exactly zero** on every axis, while their MAE
  differs by 14× the scheme's own residual. Every statistic in a `VLBitProfile`
  is permutation-invariant by construction. A sub-block rotation control, which
  is *not* a pure permutation, does register (exponent L1 0.049). Pinned by
  `numerical::compare::tests::permutation_invariance_is_a_known_blind_spot`.
- **Observer overhead grows with layer width** (`stumman/M2_5_OBSERVABILITY.md`,
  2 repeats): +4.3% at 64×64, +11.5% at 256×256, **+14.2% at 512×512**. The
  "fixed cost" hypothesis is refuted by the sweep, which is why D-19's
  `--step-sample N` is load-bearing rather than a nicety.
- **Sampling works on the archive too**: a 192-step run with
  `--bit-scope gradients,optimizer` went from **1.21 MB / 2,688 profiles** to
  **83.5 KB / 168 profiles** at `--step-sample 16`. Before Wave 4 fixed it,
  sampling thinned the steps but not the profiles at all.

### Deviations from the v3 design, all declared

- **F-06 was factually wrong.** `glictus-caliburni::checksum` used the `sha2`
  crate, not a hand-rolled std-only SHA-256, and `glcore` was optional there,
  not unconditional. The genuinely hand-rolled implementation was in
  `gljax/src/runtime/digest.rs`. Resolved by writing a std-only SHA-256 in
  `glcore::hash` and delegating both — workspace implementations 2 → 1.
- **D-02 does not work as written.** `stumman = { path = ..., optional = true }`
  fails with "multiple workspace roots found in the same workspace"; the root
  `Cargo.toml` must also `exclude` it.
- **§7.1's observer ordering was over-specified** and is corrected in the design
  document: the observer runs *after* `optimizer.step()`, because
  `optimizer_ns` cannot be reported by a callback that runs before the
  optimizer does. KL-006 never depended on the position.
- **`glbench train` has no `--model` or `--dataset`.** stumman M2 generates its
  frozen base weight from a seed and builds its dataset in memory; neither flag
  has a subject, and both are rejected with an error that says so.

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
- ~~**Scaling sweeps as a first-class command.**~~ ✅ done —
  `runner::scale::run_sweep` drives `planner::run` once per `--sweep` token
  budget (sequentially, same bandwidth-contention reasoning as `ab`), then
  feeds the per-point decode-tps means to `analysis::scaling::classify`.
  `glbench scale --engine <name> --model <path> --sweep N,N,N,...`.
- **Roofline plot.** `analysis::roofline` computes arithmetic intensity and the
  ridge point; a textual/HTML roofline chart would make the memory-vs-compute
  verdict visual.
- ~~**Numerical parity command.**~~ ✅ done — `validation::parity` drives
  both engines through `EngineAdapter` (forcing greedy decoding on both,
  since parity is only meaningful at temperature 0) and reports the
  matching-prefix length via `glbench validate --engine <name> --model <path>
  --against <oracle>` (default oracle `glproc`). Exit code is non-zero on
  divergence, so it composes as a CI gate.
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
