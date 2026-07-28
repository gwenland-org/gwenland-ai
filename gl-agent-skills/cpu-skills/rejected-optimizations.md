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

---

## Tokenizer entries (`glcore::tokenizer`, added 2026-07-28)

These follow the same rule as the list above: matched **by mechanism, not by
name**.

T1. **FxHash / any faster hasher for the tokenizer's lookup tables** —
    REJECTED: **neutral**, and it adds a hash-flooding surface.

    The reasoning that leads here is sound and still wrong. `merge_ranks.get(piece)`
    runs once *per candidate pair on every merge*, keys are 1–20 bytes, and
    SipHash's per-call setup dominates at that size — so a fast hasher looks
    like an obvious win. Implemented (FxHash with the `rotate_left(20)`
    finalizer, wired into `token_to_id`, `merge_ranks`, both id sets and the
    pre-token cache) and A/B'd against SipHash on a **frozen** corpus,
    three repeats each:

    | | SipHash | FxHash |
    |---|---|---|
    | cold cache | 90.68 / 91.12 / 92.35 ns/byte | 90.31 / 90.60 / 89.94 |
    | cache OFF (merge-heavy) | 173.8 / 153.7 / 154.0 | 153.8 / 154.5 / 155.8 |

    ⛔ **The first measurement said +8–10 % and was wrong.** The bench's default
    corpus is *files from this repository*, which the same session had been
    editing — 113 KiB became 120 KiB of different text between the two runs.
    Freezing the corpus made the difference vanish. Any tokenizer A/B must pin
    its input.

    What it teaches: **hashing is not the bottleneck.** At ~154 ns/byte with
    the cache off and ~30 000 pre-tokens over 123 KiB, that is ~600 ns per
    ~4-byte pre-token — far more than its handful of map lookups can account
    for. The next attempt should *profile* rather than reason from where the
    calls are.

    Two by-products worth keeping even though the change was reverted:
    * `Vocab::specials_by_len` was sorted by length only, **stably**, over a
      `HashSet` iteration — so equal-length special tokens kept hash order and
      `find_special` picked between them arbitrarily. A latent bug that any
      hasher change would have exposed as moved token ids. Now sorted by
      `(Reverse(len), text)`. **Fixed and kept.**
    * FxHash's low bits are its weakest (multiplication propagates entropy
      upward) and `hashbrown` indexes buckets with exactly those. If this is
      ever revisited, the finalizer matters: measured 814 of 1024 buckets
      without a final rotate against an ideal of ~885.

T2. **Splitting one input across threads** — REJECTED: **not correct**, not
    merely unprofitable. See
    `pretok::tests::splitting_the_input_changes_the_segmentation`. `\s+(?!\S)`
    keeps a whitespace run whole when it reaches end-of-input and surrenders
    its last character when it does not, so *any* cut re-segments the seam.
    The intuitive safe point — immediately after a newline — is a
    counterexample under the GPT-2 shape: `"a \nb"` segments as
    `"a" · " " · "\n" · "b"` whole and `"a" · " \n" · "b"` when cut.
    **Inter-request parallelism is unaffected and already works**:
    `GllmTokenizer` is `Sync` and its scratch is thread-local.

T3. **SWAR and dual-cursor ILP in the pre-tokenizer** — NOT REJECTED, but
    **deprioritised, and dual-cursor matches entry 5 above by mechanism.**
    Gigatoken reports 380 → 1049 MiB/s from these. Pre-tokenization is ~3 % of
    encoding here (4.9 of ~154 ns/byte), so the ceiling on both together is
    ~2 % end-to-end. Dual-cursor ILP is the *same shape* as the row-tile GEMM
    lead this repo has already rejected twice for winning in a probe and going
    neutral in production.

T4. **Streaming pre-tokens through the `split` callback instead of collecting
    them into a `Vec`** — REJECTED: **8 % slower on the miss path.**

    Collecting costs one 16-byte entry per pre-token — ~475 KiB written per
    120 KiB encoded, for a list read once in order and dropped — so removing it
    looks strictly free. Measured on a frozen corpus: cold cache 48–51 → 52–55
    ns/byte, cache off 110–120 → 122–126. Warm improved ~5 %, cold regressed,
    and **cold is the number that gets quoted**. Presumably `merger.run` stops
    inlining through two closure layers.

    General shape: *removing an allocation is not automatically a win when the
    allocation is amortised and the replacement changes inlining.*
