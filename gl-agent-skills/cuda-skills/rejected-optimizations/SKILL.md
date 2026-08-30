# ⛔ Rejected Optimizations (CUDA) — DO NOT REVISIT

> **Domain:** cuda-skills
> **Applies to:** `glcuda` on NVIDIA Turing (T4, sm_75) — the reference GPU
> **Last updated:** 2026-08-22

## BEFORE YOU START

- [ ] I checked my plan against every entry below **by mechanism, not by name** — a rejected idea usually returns wearing different words.
- [ ] If my proposal matches an entry: I stop. Revisiting requires **explicit permission from JinXSuper** plus new evidence (different hardware, or a mechanism provably absent from the original test).
- [ ] I read `architecture/glcuda-research/` — `ceiling-sprint-summary.md` §7 and `bottleneck-audit-2026-08-22.md` §8 carry the longer reasoning behind several entries here.

## Context

This is the GPU counterpart to `cpu-skills/rejected-optimizations.md`, which
is **CPU-only** and does not constrain `glcuda` — that file's entries (L2
tiling for decode, interleaved rows, AVX-512F) were measured on an i3 and say
nothing about a T4. Mixing them up has already cost one audit a wrong premise.

Reference hardware: **Tesla T4** — sm_75, 40 SMs, 16 GB GDDR6, ~320 GB/s,
8.1 TFLOPS FP32, 130 TOPS INT8, 64 KB shared memory per SM, no `cp.async`.

⚠️ This machine class drifts **~8% between sessions**. Every A/B here was run
**interleaved** within one session. Sequential A/B on this box has produced
impossible results before and is not acceptable evidence.

## The List

1. **`GLCUDA_FORCE_Q8` as the default weight format** — REJECTED as a default,
   **kept as an opt-in flag**. Measured 2026-08-22 on one T4, interleaved,
   3 repeats per arm, correctness-gated (`parity` + `graph_replay` + `forward`
   green on hardware before any timing was read):

   | model | prefill | warm decode | VRAM |
   |---|---|---|---|
   | Qwen2.5-0.5B | **+113.6%** (no overlap) | −2.2% (**overlap**) | +3.6% |
   | Qwen2.5-3B | **+693.2%** (no overlap) | **−19.7%** (no overlap) | **+39.0%** |

   The prefill win is real and large — it is the `in_dim % 256 == 0` GEMV trap
   (`down` never reaches the tensor-core GEMM). But requantizing k-quants
   (4.5–6.5625 bpw) to Q8_0 (8.5 bpw) widens every weight, and **decode is
   bandwidth-bound**: it reads every weight byte per token. The penalty is
   proportional and grows with model size — VRAM +39% against decode −19.7%
   at 3B, i.e. decode slowed almost exactly as much as the weights grew.

   **Do not re-propose flipping the default on the strength of the prefill
   number.** The trade is workload-dependent (long-prompt/short-output wins,
   chat loses), which is what a flag is for.

   ⭐ **The trade should not exist.** `force_q8` fixes a *kernel dispatch*
   problem by changing the *storage format*. The correct fix is a GEMM that
   consumes k-quants directly, which buys the prefill win without the decode
   cost — see `bottleneck-audit-2026-08-22.md` §7/A1.

2. **Multi-stream prefill (`GLCUDA_MULTI_STREAM_PREFILL`)** — REJECTED:
   **−0.6%**, neutral. The stream pool works, but the GEMM grid is already too
   small to fill the machine (`ceil_div(out_dim, 64)` gives `down` 14 blocks on
   40 SMs) *and* the k-quant tensors never take the GEMM path at all, so there
   was nothing to overlap. Code kept behind the flag, off by default.

3. **cuda-oxide port of the MMA GEMM** — REJECTED, track closed. **7.7× slower**
   (2759 µs vs 358 µs). Root-caused from the generated PTX: a runtime `m_tiles`
   loop bound blocks accumulator register promotion, spilling
   `[[f32;2];32]` to `.local` and adding `ld.local`/`st.local` on every MMA
   iteration. Occupancy was *not* the issue (50 reg vs 44). Note the inversion
   this implies: the hand kernel's full unroll is what **avoids** the spill.

4. **Removing the `ntok <= 64` contract from `gl_gemm_mma_q8`** — REJECTED in
   the ceiling sprint's Phase 2. The contract was never the blocker; the useful
   part was obtained without touching the 8-m-tile accumulator structure the
   limit derives from.

5. **"Hold the INT32 accumulator to the end of the k-loop"** (deferred epilogue)
   — REJECTED as **numerically invalid**, not merely unprofitable. Q8_0 carries
   a separate f16 weight scale *and* f32 activation scale per 32-element K
   block:

   ```
   d = Σ_kb ( s32_acc[kb] × wsc[kb] × xsc[kb] )
   ```

   Blocks have different quanta, so their integer accumulators cannot be summed.
   The kernel already defers as far as the format permits — the two chained
   MMAs inside one 32-K block *do* accumulate in s32 — and 1 `cvt` + 1 `mul` +
   1 `fma` per output is already minimal for this formulation.

   ⚠️ The **diagnosis** behind this proposal was correct and is NOT rejected:
   the FP32 scaling epilogue costs ~1.5× the tensor-core work it serves
   (3.0 vs 2.0 cycles per warp per m-tile per k-block), so roughly 60% of issue
   time goes to CUDA cores. The lever is **scale granularity** — a per-row
   activation scale would factor `xsc` out of the k-loop — which is an accuracy
   decision requiring re-derived parity tolerances, not an assembly edit.

6. **Predicating `mma.sync` instead of branching** (to "fix warp divergence")
   — REJECTED: there is no divergence to fix. The predicates in both MMA
   kernels are **warp-uniform** (`p11` from warp id, `p1..p7` from `ntok`, a
   kernel parameter), so all 32 lanes branch together: one instruction, no
   serialization. Worse, `mma.sync.aligned` is a warp-level collective
   requiring all 32 threads — per-thread predication is undefined behaviour if
   the predicate is ever non-uniform. The branch is also *cheaper*: it skips
   every remaining m-tile at once, where a predicated-off `mma.sync` still
   consumes an issue slot.

## Not rejected — deferred, with the reason still standing

These are live ideas held back for a stated cause. Check the cause before
picking one up; if it no longer holds, the idea is available again.

- **Double-buffering / software prefetch in `gl_gemm_mma_q8`** — deferred:
  *"latency hiding is the wrong tool while achieved occupancy is 25% for lack
  of blocks."* Fix the grid first (see below). Note r256 already has the
  B-fragment prefetch; the default 8-tile kernel does not.
- **Shared-memory bank conflict on the A fragment** — real and derived, but
  **2-way**, not catastrophic: the address is `(lane/4)*32 + (lane%4)*4`, giving
  16 banks × 2 lanes. Same deferral reason as above. (A claim of *32-way* here
  is a misreading of `groupID` as `lane`.)
- **K-splitting / persistent accumulation** — deferred: needs f32 atomics or a
  second reduction pass plus device scratch, which violates glcuda's
  **zero-`cuMemAlloc`-after-init** contract (`memory-management.md` rules 1-2).
  Requires an `architecture/` spec update and sign-off, not a patch.
- **Finer `out_dim` tiling / 2D grid** — the ceiling sprint measured **3.28×**
  and **3.56×** in a diagnostic harness, but the patch is deliberately **not**
  in the monorepo: the merge gate is glbench `prefill_tps` on a real model,
  because this project has twice recorded a ~2× isolated win that went neutral
  in production. This is the highest-value open GPU item.

## The pattern this list keeps recording

Three separate optimizations have now measured **~2-3× in isolation and
neutral-or-negative in production** (VNNI-512 and row-tile GEMM on CPU;
multi-stream prefill on GPU). A fourth, `force_q8`, measured **+693% prefill
and −19.7% decode** — a genuine win on one axis paid for on another.

The lesson is not "optimizations don't work". It is that **an isolated
speedup is a hypothesis about the whole system, and this project has been
wrong about that hypothesis more often than right.** Production A/B,
interleaved, in one session, with a correctness gate in front of it.
