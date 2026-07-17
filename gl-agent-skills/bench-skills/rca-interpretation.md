# Interpreting glbench Analysis Output (RCA)

> **Domain:** bench-skills
> **Applies to:** the `AnalysisReport`/hypotheses/roofline/behavior sections of a `BenchmarkSession`
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the epistemic contract: glbench **hypotheses are "consistent with", never verdicts.** They rank what to investigate; they don't convict.
- [ ] I will not turn an absent metric into a zero: sections appear **only when their inputs were measured** — absence means "not measured".
- [ ] I know the two axes that get conflated: **ms/call vs share %** — a bucket can dominate share % while being fast per call (called often), and vice versa.

## Context

glbench v2 doesn't just print tok/s: it buckets engine telemetry into a
roofline, extracts behavioral signals from raw logits, and emits cross-signal
root-cause *hypotheses*. That output is designed to be the start of a
diagnosis, and it is easy to misread in exactly the ways this project has
been burned before — over-trusting a share %, mistaking config effects for
kernel bugs, or believing one anomalous run.

## Rules

1. **Hypotheses are leads, not findings.** The report phrases them as "the
   data is consistent with X" — repeating one as "glbench says X is the
   bug" is a misquote. Confirm with a targeted experiment before acting.
2. **Roofline classification is per-bucket and ceiling-relative:** each
   bucket (attention / ffn / lm_head) is classified against the **measured**
   bandwidth ceiling of *that machine* — `bandwidth-bound`,
   `not-bandwidth-bound`, or `indeterminate`. `indeterminate` is a real
   answer; don't round it to whichever class your plan prefers.
3. **ms/call ≠ share %.** Diagnose with both: high share + low ms/call →
   call count/structure problem (fusion territory); high ms/call → kernel
   or layout problem. Optimizing the highest-share bucket's *kernel* when
   the issue is call structure wastes a week.
4. **Share % is not comparable across models.** Different configs shift the
   bucket mix (the 0.5B vs 7B lesson) — compare absolute rates (GMAC/s,
   GB/s effective) across models, share % only within one model.
5. **Cold ≠ warm, ever.** The first iteration (page-in, cache-fill) is
   reported separately; mixing it into warm statistics — or diagnosing a
   "regression" from a cold number — is invalid. Cold-vs-warm *deltas* are
   themselves diagnostic (mmap behavior, LFB effects on cold strided KV).
6. **Behavioral signals are context-dependent:** low entropy on a
   CoT-capable model is expected; on a plain model it's an anomaly — the
   report already applies that CoT-awareness, so read the flag before
   reacting. Repetition/perplexity/drift signals point at *model or
   config* problems as often as engine bugs.
7. **One anomalous session proves nothing.** Reproduce it (same archived
   workload) before opening the investigation; thermal state and background
   load produce one-off ghosts (see
   [measurement-discipline.md](measurement-discipline.md)).
8. **Validation failures outrank everything:** if the session's
   `ValidationReport` flags parity/determinism problems, the performance
   numbers in that session are void — fix trust first.

## ✅ Correct Pattern

```text
Report: ffn share 52 %, bandwidth-bound; hypothesis: "consistent with
memory-bound FFN streaming".
Response: matches the known CPU profile → no bug. Optimization must reduce
FFN bytes (quant/layout), not "speed up the FFN kernel" — and the Q4_K
verdict already bounds what byte-reduction is allowed to attempt.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "glbench proved the sampler is broken" — it emitted a hypothesis ranked
   by signal overlap. Design the confirming experiment.

❌ "attention is only 12 % share, ignore it" on a model where it's 12 % of
   a huge total — check GMAC/s and ms/call before dismissing.

❌ Treating a missing energy section as "0 J/token" in a report table.
```

## GwenLand-Specific Notes

- The famous internal example: a scary attention anomaly was eventually
  explained by **single-core LFB limits on cold strided KV** — a hardware
  interaction, not a kernel bug. The hypothesis engine can only point;
  the explanation took targeted experiments. Budget for that step.
- Toxicity-style content metrics are **deliberately unimplemented** in the
  behavior section — don't "complete" them; that's a charter decision, not
  a gap.

## Related Skills

- [glbench-usage.md](glbench-usage.md)
- [measurement-discipline.md](measurement-discipline.md)
- [../cpu-skills/memory-bandwidth.md](../cpu-skills/memory-bandwidth.md)
