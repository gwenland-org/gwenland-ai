# Inference First — The Philosophy

> **Domain:** architecture-skills
> **Applies to:** every design decision in the repository
> **Last updated:** 2026-08-17

## BEFORE YOU START

- [ ] I can state the priority order: **correct inference anywhere → measured performance on the target tier → everything else.**
- [ ] My change doesn't trade correctness, portability, or the zero-ML-dependency build for speed.
- [ ] If my change is a speedup, its claim is (or will be) a production measurement, not a projection.

## Context

GwenLand's tagline is *"finding its limit, not the speed."* The project's
value is not being the fastest engine — llama.cpp exists — it is being a
**fully understood** engine: every component written from scratch, every
performance number measured and explained, every limit known and documented.
"Inference First" operationalizes that: a model must run *correctly* on
whatever hardware is present (CPU floor included) before any backend chases
its hardware's ceiling.

## Rules

1. **Correctness precedes performance, structurally:** every op gets a
   scalar/reference implementation first; optimized paths are validated
   against it within explicit tolerances. An optimization PR that arrives
   without its reference comparison is incomplete.
2. **Anywhere precedes fast:** a feature that only works with a GPU/toolkit/
   OS-specific dependency at *build* time is rejected — runtime detection
   with a fallback is the pattern
   ([fallback-chain.md](fallback-chain.md),
   [`../cuda-skills/dynamic-loading.md`](../cuda-skills/dynamic-loading.md)).
3. **The 8 GB machine is a first-class citizen**, not a degraded mode. If a
   change makes the reference i3 worse to make a big machine better, it
   needs explicit sign-off, not a silent trade.
4. **Measured limits are deliverables.** "We are at 88 % of T4 bandwidth"
   and "native Q4_K loses 33 % on this tier" are project *outputs* — record
   them (changelog, architecture docs, these skills) even when the result
   is negative. Especially when it is negative.
5. **No speculative complexity:** no abstraction for engines that don't
   exist yet, no config knobs "for later", no cross-backend frameworks.
   Each engine earns its complexity from its own hardware's measurements.
6. **From-scratch is a constraint, not a preference:** no external ML
   dependencies, no C bindings, no CMake. Pulling in a crate that "just
   does the tokenizer" breaks the point of the project.
7. **When speed and understanding conflict, understanding wins:** an opaque
   5 % win (unexplainable, unreproducible, or probe-only) does not merge; an
   explained 0 % result (a closed dead end) does — as documentation.

## ✅ Correct Pattern

```text
New op (e.g. a new activation):
1. Scalar reference in the engine + unit tests vs known values.
2. Wire into the model graph; verify end-to-end coherence on CPU.
3. Only then: SIMD/PTX versions, each parity-tested vs the reference.
4. glbench before/after on the production path; record the numbers.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Skipped the scalar version — the AVX2 one is obviously right."
❌ "Requires the CUDA toolkit at build time, but it's 8 % faster."
❌ "It regressed the i3 slightly but the Xeon box loves it." (silent trade)
❌ "The experiment failed, so I deleted the branch and moved on."
   (the number was the deliverable — write it down: changelog + rejected list)
```

## GwenLand-Specific Notes

- This philosophy is why the repo keeps *negative* results so prominently
  (rejected-optimizations, closed dead ends): the project's moat is a
  correct mental model of the hardware, and that model is built from
  falsified hypotheses as much as from wins.
- "Pre-1.0" status is honest scoping, not apology: correctness claims are
  bounded by the parity suites and the model families actually tested
  (Llama/Qwen2/Qwen3, Q8_0/Q4_K) — don't advertise beyond them.

## Related Skills

- [fallback-chain.md](fallback-chain.md)
- [../before-coding/read-architecture-first.md](../before-coding/read-architecture-first.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
