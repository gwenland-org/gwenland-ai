# Measurement Discipline

> **Domain:** bench-skills
> **Applies to:** every performance claim made anywhere in the project
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] **Baseline first:** I have an archived session of the *unchanged* code on this machine before I measure my change.
- [ ] Windows → Defender exclusions verified ([windows-defender-gotcha.md](windows-defender-gotcha.md)).
- [ ] I know the cardinal history: probes vs production in this repo have disagreed by **0.07×–2.40× for the same change**. Production numbers or it didn't happen.

## Context

This project's benchmark traps have each produced *confidently wrong*
answers: KV-layout probe artifacts, guessed context sizes, thermal drift,
Defender rescans, cold/warm mixing. The discipline below is the accumulated
antidote. It is deliberately boring; the exciting part is supposed to be the
result, not the methodology.

## Rules

1. **Measure in production, not probes.** The number that decides a change
   is the real decode/prefill path via `glbench` on the reference machine.
   Microbenches and standalone kernel probes may guide *where* to look —
   they never justify merging or rejecting a change by themselves.
2. **Baseline first, A/B in one session-window:** measure base and candidate
   on the same box, same power state, back-to-back — `glbench ab` exists
   precisely for the multi-candidate case (and runs candidates sequentially
   on purpose; never parallelize a bandwidth-bound comparison).
3. **Warmup, then measure; cold reported separately.** glbench times the
   first-ever iteration apart from warm statistics — keep that separation in
   anything you build or quote. For full-session runs, `--warmup ≥ 1` and
   `--iters ≥ 5` for decision-grade claims; for microbench-style kernel
   timings, ≥ 10 warmup and ≥ 100 measured iterations.
4. **Report distributions, not just means: P50/P90/P99.** Per-token
   latencies supply the sample mass in a decode run. A mean-only claim
   hides stalls and drift — the report's percentile and drift sections
   exist; quote them.
5. **Control the environment and say so:** AC power (not battery), no
   background builds/indexers, thermal state noted if the box was hot. On a
   15 W laptop CPU, thermals *are* a variable — an unexplained ±10 % between
   sessions is weather until reproduced cold.
6. **Fix the workload:** same model file, same quant, same context/token
   counts, greedy sampling + fixed seed. "Roughly the same prompt" is a
   different workload — archived sessions pin this for you.
7. **State the machine.** Numbers travel with (CPU/GPU, RAM config, OS,
   commit hash). The i3-1115G4 tier is the CPU reference; a win on another
   box is a fact about that box, not about the project's target.
8. **Repeat before you believe:** any surprising result (good or bad)
   reproduces on a fresh run before it's reported. One session = anecdote;
   two matching sessions = a number.
9. **Negative results get archived too** — the rejected-optimizations list
   is built from properly measured losses; an unrecorded failed experiment
   will be re-run by the next optimist.

## ✅ Correct Pattern

```text
Claim in a PR/gate report:
"Decode +7.3 % (P50 41.2 → 44.2 tok/s, P99 stable), Qwen2.5-0.5B Q8_0,
 i3-1115G4, Linux, AC, commit abc1234, warm iters=5, cold unchanged.
 Sessions: benchmarks/base-014.json vs benchmarks/cand-015.json (compare
 output attached). Reproduced twice."
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "The kernel probe shows 2.1× — merging."          (probes lie: 0.07×–2.40×)
❌ "Mean improved 4 %" with no percentiles, no session archive, no machine.
❌ Comparing today's laptop-on-battery run to last week's AC baseline.
❌ Believing a single spectacular regression/win without a reproduction.
❌ Quietly discarding the run that disagreed with the other two.
```

## GwenLand-Specific Notes

- The roofline is the sanity check bracketing every claim: a decode result
  implying > 100 % of the measured 29 GB/s ceiling is wrong somewhere —
  find the error before publishing (usually workload or unit math).
- Perf waves present these numbers **at the wave gate**
  ([`../before-coding/wave-confirmation-gates.md`](../before-coding/wave-confirmation-gates.md));
  "it should be faster" is not a gate report.

## Related Skills

- [glbench-usage.md](glbench-usage.md)
- [windows-defender-gotcha.md](windows-defender-gotcha.md)
- [../cpu-skills/rejected-optimizations.md](../cpu-skills/rejected-optimizations.md)
