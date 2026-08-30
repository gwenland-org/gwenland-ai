# glbench Usage — Running a Benchmark That Means Something

> **Domain:** bench-skills
> **Applies to:** [`glbench/`](../../glbench/) (binary: `glbench`); measuring any engine
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] Windows? **Defender exclusions verified first** — [windows-defender-gotcha.md](windows-defender-gotcha.md). No exclusion, no benchmark.
- [ ] I know glbench's one rule: **it observes, it never optimizes.** It runs through the `GlEngine` contract and reports; changing engines is my job, not glbench's.
- [ ] I know which question I'm asking (throughput? regression? A/B? parity?) — each has a dedicated subcommand; ad-hoc timing scripts are not welcome.

## Context

glbench ("Mensura Veritatis") is the project's only accepted source of
performance truth: it produces a `BenchmarkSession` (environment snapshot,
workload spec, raw measurements, analysis) that can be archived as JSON,
re-rendered, exported, and diffed. Numbers that don't come from a session are
anecdotes — and this project has been burned by anecdotes (probes off by
0.07×–2.40×).

## Rules

1. **Build release, bench the binary:** `cargo build --release -p glbench`.
   Never benchmark a debug build; never benchmark under `cargo run`'s
   compile-check noise.
2. **The standard commands:**
   ```sh
   # measure one engine/model:
   glbench run --engine glproc --model m.gguf --tokens 128 \
       --warmup 1 --iters 5 --kind decode --out benchmarks/run-001.json

   # A/B candidates under ONE identical workload (sequential on purpose):
   glbench ab --engine glproc --model a-q8_0.gguf --model a-q4_k_m.gguf

   # regression check vs an archived baseline (5 % threshold default):
   glbench compare benchmarks/baseline.json benchmarks/candidate.json

   # re-render / convert an archive:
   glbench inspect benchmarks/run-001.json
   glbench export  benchmarks/run-001.json --format md --out report.md
   ```
3. **Determinism unless you're studying sampling:** default
   `--temperature 0.0` (greedy) + fixed `--seed` keeps token streams
   comparable across runs; a changed token stream changes the workload.
4. **`--kind` matches the question:** `decode`, `prefill`, `end_to_end`, or
   `stress` — a decode conclusion drawn from an `end_to_end` run is mixing
   phases with different physics.
5. **Archive decision-grade runs** (`--out benchmarks/….json`). A number
   quoted in a PR/gate without an archived session behind it is
   unverifiable — sessions are the citations.
6. **A/B is sequential BY DESIGN.** Never "speed it up" with parallel
   candidate runs — parallel decodes contend for the memory bus and corrupt
   every number on a bandwidth-bound workload.
7. **CoT-capable models:** `--cot` defaults to the GGUF header's word;
   override (`on`/`off`) only when studying the flag itself — it changes
   behavioral-signal interpretation (entropy expectations).
8. **glbench stays dependency-free and observation-only.** Feature PRs that
   make it fetch models, tweak hardware, or auto-apply "fixes" violate its
   charter (see [`glbench/README.md`](../../glbench/README.md)).

## ✅ Correct Pattern

```sh
# "Did my kernel change help?" — the whole ritual:
glbench run --engine glproc --model q.gguf --kind decode --iters 5 \
    --out benchmarks/base.json          # BEFORE the change, archived
# ...apply change, rebuild release...
glbench run --engine glproc --model q.gguf --kind decode --iters 5 \
    --out benchmarks/cand.json          # same box, same session-ish conditions
glbench compare benchmarks/base.json benchmarks/cand.json
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ let t = Instant::now(); …; println!("{:?}", t.elapsed());  // ad-hoc probe
❌ Comparing today's run against a number remembered from last week's
   different machine/quant/ctx — compare archived sessions or nothing.
❌ Benchmarking with --engine unset and not checking which engine the
   session actually recorded (fallback may have landed you on glproc).
```

## GwenLand-Specific Notes

- glbench is also the **profiler**: it pulls per-stage engine telemetry
  (attention/ffn/lm_head buckets) and behavioral signals from raw logits —
  interpreting those is [rca-interpretation.md](rca-interpretation.md)'s
  topic.
- Energy (J/token) comes from Linux RAPL only, when readable — it is never
  estimated from TDP, and it is absent on Windows/macOS by design. Don't
  fake it downstream.

## Related Skills

- [measurement-discipline.md](measurement-discipline.md)
- [windows-defender-gotcha.md](windows-defender-gotcha.md)
- [rca-interpretation.md](rca-interpretation.md)
