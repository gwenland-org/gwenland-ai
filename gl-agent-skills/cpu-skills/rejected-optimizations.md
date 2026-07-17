# ⛔ Rejected Optimizations — DO NOT REVISIT

> **Domain:** cpu-skills
> **Applies to:** `glproc` (and the judgment patterns generalize to every engine)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I checked my plan against every entry below **by mechanism, not by name** — a rejected idea usually returns wearing different words.
- [ ] If my proposal matches an entry: I stop. Revisiting requires **explicit permission from JinXSuper** plus new evidence (different hardware tier or a mechanism provably absent from the original test).
- [ ] I also grepped `changelog/` for prior attempts at my idea.

## Context

Every entry here was implemented (or seriously probed), measured on the
reference machine (i3-1115G4, 8 GB DDR4-2667), and rejected with numbers.
This list is the project's scar tissue. It exists because these ideas are
*attractive* — each one follows from a reasonable mental model that this
specific hardware then falsifies. An agent that re-proposes them isn't being
creative; it's re-running a completed experiment at the user's expense.

## The List

1. **L2 tiling for decode** — REJECTED: no benefit. Decode streams each
   weight byte once per token; there is no reuse for tiling to exploit, and
   bandwidth was already the binding constraint. (Tiling remains legitimate
   for *prefill* GEMMs, which have reuse.)
2. **Interleaved row layout** — REJECTED: **-35 % regression**, reverted
   immediately. Broke the linear streaming pattern the hardware prefetcher
   was feeding on.
3. **AVX-512F** — REJECTED: thermal/downclock risk on Tiger Lake's 15 W
   envelope; 512-bit execution costs more in frequency than it gains in
   width. The engine *detects* AVX-512 and *declines* it — that behavior is
   intentional. (Includes "at least use AVX-512VNNI-512" — declined for the
   same reason; 256-bit VNNI is the ceiling.)
4. **PREFETCH_CHUNK = 64 software prefetching** — REJECTED: tested, no
   improvement. The hardware prefetcher already saturates linear weight
   streams; software prefetch just burned issue slots.
5. **Raw-mmap lazy layer paging** — REJECTED: architecturally incompatible
   with the Q8_0 repack path (repacked weights live in anonymous memory, not
   the file mapping; there is nothing to lazily page). Also reintroduces
   cold-fault jitter into decode.
6. **"Fix B" topology threading** — REJECTED: **-23 % decode**, reverted.
   Plausible-looking thread/topology assignment that fought the real
   bottleneck (see the LFB note in [threading-model.md](threading-model.md)).
7. **Native Q4_K decode compute (integer-dot and fused SwiGLU variants)** —
   REJECTED: **-33 % vs the Q8_0 path**, gap identical with the working set
   in L2 ⇒ compute-bound nibble unpack, not bandwidth. Closed as a
   hardware-tier dead end; the shipped answer is load-time repack → Q8_0.
   Full post-mortem context in [quantization.md](quantization.md).

## How rejections happen (the ✅ pattern)

```text
Idea → roofline sanity check → implementation behind a flag →
glbench A/B on the PRODUCTION path, reference machine, warm+cold reported →
number is negative or flat → REVERT fully → changelog entry with the number →
entry added here. The branch dies; the knowledge doesn't.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "The interleaving idea failed, but MY layout interleaves at tile
   granularity instead of row granularity" — same mechanism (breaking the
   linear stream), same verdict. Mechanism-match, not name-match.

❌ Quietly deleting an entry here because "hardware moved on" — additions
   need a measurement; removals need JinXSuper.

❌ Citing a probe/microbench as grounds to overturn an entry — probes in
   this repo have disagreed with production by 0.07×–2.40×.
```

## GwenLand-Specific Notes

- This list is **per hardware tier**. A future AVX-512 desktop tier or an
  ARM NEON port re-opens questions *for that tier only* — as new, explicitly
  scoped experiments with their own measurements, never by editing this
  list's verdicts for the i3 tier.
- The meta-rule behind every entry: **glproc's hot kernels are already at or
  near the practical optimum for this machine.** The burden of proof is on
  the optimization, and the only accepted proof is a production measurement.

## Related Skills

- [quantization.md](quantization.md)
- [threading-model.md](threading-model.md)
- [memory-bandwidth.md](memory-bandwidth.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
