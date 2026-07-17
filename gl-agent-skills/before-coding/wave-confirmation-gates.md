# Wave Confirmation Gates

> **Domain:** before-coding
> **Applies to:** any multi-step task executed as waves/phases/sprints
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] The task is split into explicitly numbered waves with a defined deliverable per wave.
- [ ] I know what "done" looks like for the CURRENT wave only.
- [ ] I understand I will STOP at the end of this wave — no auto-proceed.

## Context

GwenLand work is executed wave-by-wave with a human gate between waves. The
gates exist because measured reality regularly kills plans that looked perfect
on paper (entire optimization tracks have been closed by one benchmark run).
Auto-proceeding into Wave N+1 on top of an unreviewed Wave N compounds mistakes
and wastes effort in a direction JinXSuper may already know is dead.

## Rules

1. **STOP after every wave.** Completing Wave N never implies permission to
   start Wave N+1.
2. **The confirmation report at each gate must contain:**
   - the wave number and what it was supposed to deliver;
   - the list of files created/modified/deleted;
   - key results (test output, measured numbers, deviations from the plan);
   - open questions or discovered risks;
   - the explicit question: **"Siap lanjut ke Wave N+1?"**
3. **Never auto-proceed**, even if the next wave seems trivial, even if you are
   confident, even if the plan document lists all waves up front. The plan
   authorizes the *shape* of the work, not its uninterrupted execution.
4. **Any test failure = immediate STOP.** Report the failure verbatim, with
   your best hypothesis, and wait for instruction. Do not silently fix-forward
   into the next wave.
5. **Deviations must be declared at the gate**, not buried. If reality forced a
   different file name, a different approach, or a skipped item — say so
   explicitly in the gate report.
6. If the user's instructions and a plan document conflict mid-wave, finish
   nothing: stop at the conflict and ask.

## ✅ Correct Pattern

```text
Wave 2 complete — rust-skills/ (5 files).

Files created:
- gl-agent-skills/rust-skills/error-handling.md
- ... (4 more)

Key results:
- All examples type-check conceptually; two marked "conceptual pseudocode".
- Deviation: repo has no GateError type yet — documented the target pattern
  and flagged it, instead of inventing an import that doesn't compile.

Siap lanjut ke Wave 3 (cuda-skills/)?
```

## ❌ Anti-Pattern (Never Do This)

```text
"Wave 2 done. Since Wave 3 is straightforward I went ahead and finished
Waves 3, 4, and 5 as well."   ← gate violation, even if the output is good.

"Tests failed in Wave 2 but they look unrelated, continuing to Wave 3."
                              ← STOP was mandatory at the first failure.
```

## GwenLand-Specific Notes

- Gates are also where **measured numbers** get reviewed. Performance waves
  must present benchmark output at the gate (see
  [`../bench-skills/measurement-discipline.md`](../bench-skills/measurement-discipline.md)) —
  "it should be faster" is not a gate report.
- History lesson: probe benchmarks in this repo have produced answers from
  0.07× to 2.40× for the *same* change. A gate report claiming a win must say
  where the number was measured (production path vs probe).

## Related Skills

- [check-existing-tests.md](check-existing-tests.md)
- [branch-strategy.md](branch-strategy.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
