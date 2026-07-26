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
   **Re-measured 2026-07-26 under JinXSuper's explicit override** (a
   VNNI-512 `q8_0` qdot kernel, `row_dot` + `row_dot_packed8`, parity-tested
   and kept behind `GLPROC_VNNI512=1`, default off — see `glproc/src/
   kernels/qdot/q8_0/vnni512.rs` and `benches/vnni512_probe.rs`). Isolated
   kernel probe: **+20-26% GMAC/s over VNNI-256**, real and repeatable.
   Production `glbench` A/B (decode+prefill, 2 repeats, thermal-checked):
   **decode flat (+0.3%, +1.1%), prefill flat-to-negative (-1.5%, -17.7%,
   within this hardware's known ~20% session noise) — verdict `neutral`
   both times.** No throttling observed in any of the 4 runs (2995 MHz
   constant start/avg/end) — **the original thermal/downclock mechanism was
   never actually triggered here**, so the verdict's true cause is not
   downclock but that qdot is not the pipeline's actual bottleneck at
   production scale (same shape as entry 7's fusion lesson — a
   faster-in-isolation kernel doesn't move end-to-end tok/s when something
   else dominates wall clock). **Conclusion unchanged (still REJECTED for
   production use), but now grounded in a real A/B instead of policy alone
   — the width genuinely doesn't pay off, for a different reason than
   originally assumed.** Do not re-open this without a new mechanism
   (e.g. a redesigned dispatch path that removes whatever overhead is
   currently swamping the kernel-level gain).
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
8. **Row-tiled qdot (R=8 output rows batched against one shared activation,
   8 independent accumulator chains — `glproc/src/kernels/qdot/q8_0/
   row_tile.rs`, `GLPROC_ROW_TILE=1`)** — REJECTED for production default,
   2026-07-26. Isolated probe (`benches/row_tile_probe.rs`): **2x GMAC/s**
   over the sequential dispatch — the strongest isolated result of any entry
   here. Production `glbench` A/B (2 repeats, thermal-checked, no
   throttling): decode flat (+0.9%, +0.7%), prefill mixed sign (+6.2%,
   -2.1%) — verdict `neutral` both times. Stage-level breakdown explains the
   gap: `ffn_down` (moderate-size matmul) genuinely gained **+8.9% GMAC/s**,
   but `lm_head` (151936×896, already `bandwidth-bound` per its own roofline
   verdict) **lost 9.4%** — compute-side ILP cannot help a stage that's
   already bandwidth-bound, and `lm_head`'s ~23-25% share of decode time
   cancels `ffn_down`'s gain in the aggregate. **This is the third time this
   exact shape has appeared** (entry 7's fusion lesson, entry 3's VNNI-512
   re-measurement, now this): an isolated-kernel win evaporates in
   production because the real workload is a *mix* of compute-bound and
   bandwidth-bound stages, and a technique tuned for one regime doesn't
   transfer to the other — a single global dispatch flag can't help; it
   needs to be selective per-stage. Kernel kept (parity-tested), default
   off. If revisited, the open question is not "does row-tiling work" (yes,
   confirmed on `ffn_down`) but "can the flag be scoped to only the
   compute-bound stages" — that is the actual next experiment, not another
   whole-pipeline A/B.

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
