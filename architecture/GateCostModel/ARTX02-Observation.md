# ARTX02 — Observation Registry
## GateCostModel · GwenLand AI
## Last updated: 2026-07-25

---

## Purpose

This document records measurable facts about the pipelines described in
ARTX01, each with its evidence source, measurement method, unit, and
determinism characteristics. It is the evidentiary base for every claim
made in ARTX05, ARTX08, and ARTX09. It records facts only — interpretation
of what these facts mean for any architectural decision is ARTX05's job,
not this document's.

---

## Observation Table

| ID | Fact | Unit | Source | Deterministic | Variance Source |
|----|------|------|--------|---------------|-----------------|
| OBS-01 | Measured multi-threaded sequential read bandwidth ≈ 29 GB/s | GB/s | `glbench/src/environment/bandwidth.rs:38-42,115` | NO | thermal state (session-to-session, 26.6–31.6 GB/s observed) |
| OBS-02 | Native Q4_K FFN kernels measured lower MACs/s than Q8_0; gap persisted with working set L2-resident | MAC/s (GMAC/s) | `gl-agent-skills/cpu-skills/quantization.md:18-19`; GMAC/s figures (1.5–2.0 vs 3.3) from `glcore/src/telemetry.rs:70-71` | YES | fixed kernel, fixed input size |
| OBS-03 | Native Q4_K ≈ -33% end-to-end throughput vs Q8_0 repack, same hardware/workload | tok/s delta (%) | `gl-agent-skills/cpu-skills/quantization.md:18` | NO | scheduling noise (small, per source) |
| OBS-04 | FFN computation ≈ 52% of glproc decode wall-clock on Reference Tier | % of wall-clock | `gl-agent-skills/cpu-skills/memory-bandwidth.md:74` | NO | token content, KV cache size |
| OBS-05 | Package energy readable only via Linux RAPL (`powercap` sysfs); unconditionally unavailable on Windows | N/A (binary availability) | `glbench/src/environment/power.rs:13-18,48-73` | YES | — |
| OBS-06 | Numerical deviation measured as discrete longest-matching-leading-token-prefix count vs oracle engine; valid only under fixed-seed, non-sampling (greedy) decoding | token count (integer) | `glbench/src/validation/numerical.rs:10-17,38-48`; `glbench/src/validation/deterministic.rs:13-48` | YES | fixed seed, greedy |
| OBS-07 | glproc decode, Qwen2.5-0.5B Q4_K_M, Reference Tier = 36.7–39.1 tok/s | tok/s | Veritas Secunda session, 2026-07-25, `glbench run` (archived: `benchmarks/clean-baseline-n4.json`, `benchmarks/clean-baseline-n4-repro.json`) | NO | thermal state, scheduling noise (~6% spread across 2 reproductions) |
| OBS-08 | glproc prefill, Qwen2.5-0.5B Q4_K_M, Reference Tier = 128.5–135.5 tok/s | tok/s | Veritas Secunda session, 2026-07-25, `glbench run` (same archives as OBS-07) | NO | same as OBS-07 |
| OBS-09 | N_THREADS=4 (logical core count) outperforms both measured alternatives: N=2 (physical core count) = -23% decode; N=3 = -5.9% decode / -12.7% prefill | tok/s delta (%) | N=2: `glproc/src/runner.rs:36-44` (doc comment, prior measurement). N=3: Veritas Secunda session, 2026-07-25, `glbench compare` (archived: `benchmarks/clean-baseline-n4-repro.json` vs `benchmarks/clean-n3.json`) | NO | scheduling noise; N=3 result required discarding an earlier same-session reading contaminated by an unrelated runaway background process (see note below table) |
| OBS-10 | glproc decode loop (`Runner::step`) performs zero heap allocations per token in steady state; scratch buffers allocated once in `Runner::new` | allocations/token (integer = 0) | `glproc/src/runner.rs:742-766` (source read); cross-referenced against ARTX01-Verified.md | YES | — |

**Note on OBS-09:** the Veritas Secunda session first measured N=3 vs N=4 using
`glbench thread-scale`, which showed N=3 *winning* (28.6–28.9 vs 21.2–21.4
tok/s decode). That reading was later found to be contaminated by two
long-running orphaned `find` processes from an earlier, unrelated file
search in the same session, consuming CPU on the reference tier's 4 logical
cores throughout the measurement window. After killing the orphaned
processes and reproducing the N=4 baseline twice (OBS-07, OBS-08), a clean
re-measurement of N=3 via `glbench run` + `glbench compare` reversed the
result: N=3 regresses vs N=4. The contaminated `thread-scale` reading is
not recorded as an observation — only the clean, reproduced result is.
This is stated here as a fact about how OBS-09 was obtained, not as an
interpretation of what it means for thread-pool design (that is ARTX05's
job, if applicable).

---

## Observations — Detail

### OBS-01 — Memory Bandwidth
**Fact:** Measured multi-threaded sequential DRAM read bandwidth on the
Reference Tier is approximately 29 GB/s, with the source code's own
comment recording a 26.6–31.6 GB/s spread (±19%) across measurement
sessions on the i3-1115G4.
**Unit:** GB/s.
**Source:** `glbench/src/environment/bandwidth.rs:38-42` (variance figure,
in-source comment), `:71-119` (`measure_read_gbs`, the measurement
method), `:115` (unit derivation: `BUF_BYTES as f64 / el / 1e9`).
**Deterministic:** NO.
**Variance source:** Thermal state. The source comment attributes the
spread to the CPU's thermal condition ("a thermally noisy laptop needs
several tries to land one clean one") and the measurement method takes
the best of 12 passes specifically to counter this, rather than the mean.

### OBS-02 — Native Q4_K vs Q8_0 MACs/s
**Fact:** Native Q4_K feed-forward kernels were measured to execute at
lower multiply-accumulate throughput than the Q8_0 kernels currently
shipped. The gap between the two persisted when the working set was made
to fit inside the fastest cache level (L2), which is the evidence that
the loss is compute-bound (nibble-unpack cost), not a memory-bandwidth
effect.
**Unit:** MAC/s, reported as GMAC/s (multiply-accumulate operations per
second, billions).
**Source:** `gl-agent-skills/cpu-skills/quantization.md:18-19` (the L2-
resident test and its outcome). The specific GMAC/s figures cited
elsewhere in this project (native Q4_K: 1.5–2.0 GMAC/s vs Q8_0: 3.3
GMAC/s) are recorded in `glcore/src/telemetry.rs:70-71` (a source-code
doc comment), not in `quantization.md` itself — cited separately here so
the number is attributed to where it actually appears.
**Deterministic:** YES.
**Variance source:** None stated — fixed kernel, fixed input size per the
source's own framing.

### OBS-03 — Native Q4_K End-to-End Throughput
**Fact:** Native Q4_K measured approximately 33% slower end-to-end
(tokens/second) than the Q8_0 load-time repack path, on the same
hardware and the same workload.
**Unit:** tok/s delta, expressed as a percentage.
**Source:** `gl-agent-skills/cpu-skills/quantization.md:18` ("measured
33% slower").
**Deterministic:** NO.
**Variance source:** Scheduling noise, described in the source as small
relative to the effect size.

### OBS-04 — FFN Share of Decode Wall-Clock
**Fact:** Feed-forward network computation accounts for approximately
52% of glproc decode wall-clock time on the Reference Tier.
**Unit:** Percentage of wall-clock time.
**Source:** `gl-agent-skills/cpu-skills/memory-bandwidth.md:74` ("CPU
decode is FFN-bound (~52%)").
**Deterministic:** NO.
**Variance source:** Token content and KV cache size — the source notes
this figure describes "the measured profile of the target engine," which
varies with what is actually generated and how much context is cached.

### OBS-05 — Energy Measurement Availability
**Fact:** Package energy consumption is readable only through the Linux
`powercap` sysfs interface (Intel RAPL / AMD's compatible
implementation). On the Reference Tier's host operating system (Windows),
this interface does not exist, and the energy meter unconditionally
returns `None` rather than zero or an estimated value.
**Unit:** N/A — this is a binary availability fact, not a quantity.
**Source:** `glbench/src/environment/power.rs:13-18` (doc comment
stating the Windows/macOS unavailability explicitly and why: RAPL on
Windows requires a signed kernel driver for MSR access), `:48-73`
(`EnergyMeter::start`, which returns `None` when
`/sys/class/powercap` cannot be read), `:105-111` (test enforcing this
contract).
**Deterministic:** YES — the unavailability itself does not vary; it is
a fixed property of the OS/interface combination.

### OBS-06 — Numerical Deviation Measurement
**Fact:** Numerical deviation between a candidate engine and the
designated oracle engine (glproc) is currently measured as a discrete
count: the length of the longest leading run of exactly-matching token
ids between the two generated streams. This comparison is only
meaningful under fixed-seed, non-sampling (greedy, temperature 0)
decoding — under sampling, two correct engines can legitimately diverge
without either being wrong.
**Unit:** Token count (integer): `matching_prefix` out of `compared`
tokens.
**Source:** `glbench/src/validation/numerical.rs:10-17` (`NumericalCheck`
struct), `:38-48` (`compare_tokens`, the comparison method — a
break-on-first-mismatch prefix scan, not a continuous distance metric);
`glbench/src/validation/deterministic.rs:13-48` (`check`, which validates
that the run's conditions — warmup, iteration count, temperature — were
suitable for a determinism-sensitive comparison, though it does not
itself measure the deviation).
**Deterministic:** YES, under the stated conditions (fixed seed, greedy
decoding).

### OBS-07 — Decode Baseline (Reference Tier, Clean)
**Fact:** glproc decode throughput on Qwen2.5-0.5B Q4_K_M, measured on
the Reference Tier with background CPU contention confirmed absent
(verified via process inspection before each measurement), was 36.7 tok/s
(one session) and 39.1 tok/s (a second, independent session), each the
mean of 5 measured iterations after 1 warmup iteration.
**Unit:** tok/s.
**Source:** Veritas Secunda benchmarking session, 2026-07-25, produced
via `glbench run --engine glproc --kind decode`. Archived sessions:
`benchmarks/clean-baseline-n4.json`, `benchmarks/clean-baseline-n4-repro.json`.
**Deterministic:** NO.
**Variance source:** Thermal state and OS scheduling noise. The two
reproductions differ by approximately 6% (36.7 vs 39.1 tok/s) despite
identical workload, engine, and thread count.

### OBS-08 — Prefill Baseline (Reference Tier, Clean)
**Fact:** glproc prefill throughput on Qwen2.5-0.5B Q4_K_M, same clean
conditions as OBS-07, was 135.5 tok/s (first session) and 128.5 tok/s
(second session).
**Unit:** tok/s.
**Source:** Same sessions and archives as OBS-07.
**Deterministic:** NO.
**Variance source:** Same as OBS-07.

### OBS-09 — Thread Count Comparison
**Fact:** Two thread-count alternatives to the shipped default
(`N_THREADS = 4`, logical core count) have been measured on the
Reference Tier and both regress decode throughput:
- N=2 (physical core count): -23% decode (8.5 vs 11.0 tok/s, on
  Qwen3-1.7B Q8_0 — a different model/quant than OBS-07/08's Qwen2.5-0.5B
  Q4_K_M).
- N=3: -5.9% decode, -12.7% prefill (36.7 → 34.5 tok/s decode, 135.5 →
  118.3 tok/s prefill, on Qwen2.5-0.5B Q4_K_M — the same workload as
  OBS-07/08).
**Unit:** tok/s delta, expressed as a percentage relative to N=4.
**Source:** N=2 figure: `glproc/src/runner.rs:36-44`, a doc comment
recording a prior measurement (model/quant as stated above; exact date
not given in the source). N=3 figure: Veritas Secunda session, 2026-07-25,
via `glbench compare` (archived: `benchmarks/clean-baseline-n4-repro.json`
as baseline vs `benchmarks/clean-n3.json` as candidate; verdict:
"regressed").
**Deterministic:** NO.
**Variance source:** Scheduling noise. See the table note above this
section for how the N=3 figure was obtained — a first attempt at this
measurement, using a different glbench subcommand
(`thread-scale`), returned the opposite result and was traced to
background process contamination unrelated to thread count; it is not
recorded as an observation.

### OBS-10 — Decode Loop Is Zero-Allocation
**Fact:** glproc's per-token decode function (`Runner::step`) performs no
heap allocation in steady state. All scratch buffers it writes to
(`Workspace` and `BatchWorkspace` fields) are allocated exactly once,
inside `Runner::new`, via `vec![...]` calls sized from the model's
config.
**Unit:** Allocations per token (integer count = 0).
**Source:** `glproc/src/runner.rs:742-766` (the `vec![...]` allocation
sites inside `Runner::new`); cross-referenced against the prior
`ARTX01-Verified.md` audit of this same code, which reached the same
conclusion independently by reading the decode hot path
(`runner.rs:827-1125`) for allocation calls and finding none outside a
diagnostic-only, opt-in code path.
**Deterministic:** YES — this is a structural property of the code, not
a measurement subject to run-to-run variance.

---

## UNKNOWN Items (pending ARTX11)

### UNKNOWN-01 — glcuda Prefill Attention/FFN Split
Prior project records reference a figure for the split between attention
cost and feed-forward cost within glcuda's prefill phase. This figure was
**not** re-verified against current glcuda source during this research
effort — no glcuda source file was read as part of this document's Phase
1, and no current benchmark session establishing this split was located
or run. Recording it here from memory would violate this document's own
sourcing requirement (every observation must cite a currently-read
source). It belongs in ARTX11 as an item requiring re-measurement (or a
fresh source read) before any cost model may rely on it.

### UNKNOWN-02 — Continuous L2 Numerical Error Per Layer
GATE's fifth metric (m₅, per prior GateCostModel/GATE documents) requires
a continuous, per-layer L2 error estimate to compare numerical fidelity
between execution plans. The current implementation in
`glbench/src/validation/numerical.rs` (see OBS-06) provides only a
discrete leading-token-match count against a single designated oracle
engine — this is a fundamentally different kind of measurement (a
pass/fail-style prefix count, not a continuous distance) and cannot
substitute for a continuous per-layer error metric without new
instrumentation. This gap — between what GATE's metric definition needs
and what glbench currently measures — is recorded here as a fact about
present capability, not as a judgment on whether closing it is worth
doing (that determination belongs to ARTX05/ARTX11).

---

## References

- ARTX01-Reality.md (what these observations are measurements of)
- ARTX05-Gap.md (where these observations are interpreted)
- ARTX09-Variables.md (where these observations become named quantities)
