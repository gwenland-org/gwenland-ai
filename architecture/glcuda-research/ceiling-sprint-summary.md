# glcuda INT8 GEMM Ceiling Sprint — Research Summary

> **Date:** 2026-08-04
> **Hardware:** NVIDIA T4 (sm_75), Kaggle — 40 SMs, 65 TOPS INT8, 320 GB/s HBM2
> **Kernel:** `gl_gemm_mma_q8` ([glcuda/src/kernels/glcuda_sm75.ptx](../../glcuda/src/kernels/glcuda_sm75.ptx))
> **Phases:** [phase 1](ceiling-sprint-phase1.md) · [phase 2](ceiling-sprint-phase2.md) · Gate 3 below
> **Monorepo status:** PTX **unchanged**. All work is in the research copy
> ([notebooks/glcuda_cuda_oxide_kernel_research.ipynb](../../notebooks/glcuda_cuda_oxide_kernel_research.ipynb)).

## 1. Baseline

Pre-sprint numbers, from the notebook's Cell 9 before any change:

```
case out=64  in=128  ntok=8     31.999us   0.004 TOPS   (0.0% of 65)
case out=256 in=896  ntok=64    62.717us   0.468 TOPS   (0.7% of 65)
case out=512 in=4096 ntok=256   SKIP -- harness never implemented the chunk loop
```

Methodology: one **cold**, un-warmed launch with `synchronize()` inside the
timed region.

## 2. Root causes found (Phase 1)

**RC1 — the baseline measured launch overhead, not the kernel. [HIGH]**
Cases 1 and 2 differ by **224x in work** but only **1.9x in measured time**.
That is impossible for a kernel-bound measurement. Case 1 performs 65,536
MACs and still cost ~32 us; that was the overhead floor, directly observed.
Every pre-sprint TOPS figure inherited this.

**RC2 — the grid ignored `ntok` entirely. [HIGH]**
`grid = (ceil_div(out_dim, 64), 1, 1)` ([mod.rs:584](../../glcuda/src/kernels/mod.rs#L584))
depends only on `out_dim`. Case 3 launched **8 blocks on a 40-SM GPU**; the
`ntok` axis was walked *serially on the host* instead.

**RC3 — host chunking made the only compute-bound shape memory-bound. [HIGH]**
`runner.rs:166-180` chunks `ntok` at 64 rows per launch, and each chunk
re-reads the entire weight matrix. Case 3 arithmetic intensity:
**273 ops/byte unchunked -> 101 ops/byte as actually launched** (ridge = 203),
i.e. chunking pushed it from compute-bound to memory-bound.

**RC4 — `gl_gemm_mma_q8_r256` is numerically broken. [MEDIUM]** See §8.

**RC5 — cases 1 and 2 cannot demonstrate tensor-core value.** Intensity 11.0
and 78.6 ops/byte, both below the 203 ridge: memory-bound by construction.

**Correction to the sprint premise.** The brief framed case 3 as "blocked by
the `ntok <= 64` contract". It is not: production chunks, `PREFILL_BATCH` is
512, and case 3 was skipped only by the *notebook harness*. Candidate A
("remove the contract") was rejected on that basis.

## 3. Change implemented

**2D grid over `(out_dim, ntok)` — Change 1B, the only kernel change.**

`grid = (ceil(out_dim/64), ceil(ntok/64), 1)`. At entry the kernel reads
`%ctaid.y`, rebases its three token-indexed pointers by `t0 = ctaid.y * 64`,
and clamps `ntok` to its own row count. That reproduces exactly what the host
chunk loop did per chunk, so **every instruction below the insertion point is
unchanged** — MMA math, `m8n8k16` fragment layout, per-32-K scale epilogue,
register accumulators.

Mechanism, both parts measured:
1. **SM utilization.** Case 3 goes from 4 sequential launches x 8 blocks to
   **one launch x 32 blocks: 8/40 -> 32/40 SMs.**
2. **L2 weight reuse.** The 4 token-chunks now run *concurrently* over one
   2.2 MB weight matrix, which fits the T4's 4 MB L2, so RC3's redundant
   weight reads become L2 hits rather than DRAM re-reads.

Padding contract verified unchanged: `t0` is a multiple of 64 (hence of 8), so
`t0 + round8(nn) <= round8(ntok)`. The existing "x rows allocated to
round8(ntok)" guarantee is exactly sufficient.

**Change 1A (methodology)** was applied first: 1 warmup launch, correctness
check, then 50 measured iterations with a single sync after the loop —
matching `bench.rs`'s existing `[mma gate]` pattern.

## 4. Final results (Gate 3)

```
Hardware: T4, Kaggle
Methodology: 1 warmup + 50-iter warm mean, single sync after loop

case out=64 in=128 ntok=8:
  1D chunked: 8.58us   0.015 TOPS (0.0%)  [PASS]
  2D grid:    8.59us   0.015 TOPS (0.0%)  [PASS]  delta -0.1% (expected neutral, grid.y=1)

case out=256 in=896 ntok=64:
  1D chunked: 67.77us  0.433 TOPS (0.7%)  [PASS]
  2D grid:    67.64us  0.434 TOPS (0.7%)  [PASS]  delta +0.2% (expected neutral, grid.y=1)

case out=512 in=4096 ntok=256:
  1D chunked: 1177.08us  0.912 TOPS (1.4%)  8/40 SM   [PASS]
  2D grid:     358.71us  2.993 TOPS (4.6%)  32/40 SM  [PASS]  delta +228%

case out=256 in=896 ntok=200 (bonus, ntok NOT a multiple of 64):
  1D chunked: 136.94us  0.670 TOPS (1.0%)  4/40 SM   [PASS]
  2D grid:     38.48us  2.384 TOPS (3.7%)  16/40 SM  [PASS]  delta +256%

gl_gemm_mma_q8_r256: correctness FAIL all cases (independent numerical bug,
                     not the alignment crash)
cuda-oxide m32 case 3: 2759us vs hand 358us = 7.7x slower -- research track closed
```

Derived figures:

| case | speedup | TOPS/SM 1D | TOPS/SM 2D | scaling efficiency |
|---|---|---|---|---|
| 3 (512/4096/256) | 3.28x | 0.1140 | 0.0935 | **82%** |
| 4 (256/896/200) | 3.56x | 0.1675 | 0.1490 | **89%** |

4x the SMs returned 3.28x the throughput — 82% scaling efficiency, an 18%
per-SM loss consistent with more blocks contending for L2/DRAM. Case 4 scales
better (89%) on 4x fewer SMs, because its 1D baseline was the more starved of
the two (4 blocks, 4 launches).

Notes on the numbers:

- **The two neutral cases are the predicted outcome, not a null result.**
  `ntok <= 64` gives `grid.y = 1`, so `ctaid.y` is 0 and the patch is a
  provable no-op. Phase 2 stated this in advance; measured -0.1% / +0.2% is
  noise around zero, and confirms the patch adds no overhead.
- **Case 4 (`ntok=200`) is the tail-block test.** 200 is deliberately not a
  multiple of 64, so `grid.y=4` has a ragged final block (8 rows). It PASSES,
  validating the tail predication — the exact class of bug that the
  dim-896 regression test exists to catch.
- **Phase 2's estimate held.** It predicted "2-4x on case 3, capped by 80% SM
  utilization and L2 behavior". Measured 3.28x, inside the range. Worth
  recording given this project's history of probes landing 0.07x-2.40x off
  reality.
- **Do not compare §1 against §4 as a speedup.** They are different sessions
  *and* different methodologies. Cold-vs-warm is a measurement correction, not
  a performance change; this machine class also drifts ~24% between sessions.
  The 1D-vs-2D deltas above are same-session, same-methodology, and are the
  only valid comparison here.

## 5. Ceiling gap analysis

Case 3, 2D grid, 2.993 TOPS achieved. The denominator matters:

| denominator | value | achieved |
|---|---|---|
| T4 peak INT8 | 65 TOPS | **4.6%** |
| SM-capped peak at 32/40 SMs | 52 TOPS | **5.8%** |
| Phase 1 "attainable" for the **1D** config | 13 TOPS | **23.0%** |
| memory roofline if chunked traffic persisted | 32.3 TOPS | 9.3% |

The "23% of attainable" figure is real but needs its label: it measures
against the ceiling that applied *before* the change (13 TOPS = 20% SM cap on
a 1D grid). The change **raised the attainable ceiling itself**, 13 -> ~52
TOPS, by unlocking 4x the SMs. Against the new ceiling, efficiency is 5.8%
(vs 7.0% before) — absolute throughput rose 3.28x while per-SM efficiency fell
18%. Both statements are true; quoting only the first would overstate the
result.

65 TOPS remains unreachable for these shapes regardless: cases 1, 2 and 4 sit
below the 203 ops/byte ridge and are memory-bound by construction.

## 6. Remaining gap

**Correction to the Phase 3 brief's framing:** it proposed recording the
remaining gap as "shared memory staging overhead, warp scheduling — not
structural anymore". The arithmetic does not support "not structural".

At 100% theoretical occupancy the kernel hosts 4 blocks/SM (binding limit:
1024 threads/SM / 256 threads per block; registers allow 5, smem allows 28).
The machine therefore has **40 x 4 = 160 concurrent block slots**. Case 3's
2D grid launches **32 blocks = 20% of block capacity**, so each of the 32
busy SMs runs exactly **one** block = 8 of its 32 warp slots = **25% achieved
occupancy**, and 8 SMs still get nothing.

So the top remaining item is the *same* root cause as RC2, one level down and
much less severe: the grid is still ~5x too small to fill the machine. With
one block per SM there are no other warps to cover staging and `bar.sync`
stalls, which is also why the per-SM efficiency fell 18%.

Ranked remaining work:
1. **[HIGH] Grid still 5x too small.** ~160 blocks needed, 32 launched.
   Requires finer `out_dim` tiling (Phase 2's Change 2, gated and not
   attempted) or K-splitting. This is the only item with a clear mechanism
   for a large further win.
2. **[MEDIUM] Staging + barrier overhead per block.** `in=4096` means 128
   K-blocks, each with a 2 KB + 256 B staging round and 2 `bar.sync`. At 25%
   occupancy these stalls are exposed rather than hidden. Partly a symptom
   of item 1.
3. **[LOW] Warm launch floor ~8.3 us.** Two-point fit across cases 1 and 2
   (assumes perfect work-proportionality, so treat as an estimate, not a
   measurement): case 1 is ~97% floor, case 2 ~12%, case 3 ~2%. Irrelevant
   at the shapes that matter.

## 7. Rejected approaches

**cuda-oxide port — REJECTED, track closed.**
Case 3: 2759 us vs the hand kernel's 358.71 us = **7.7x slower**. Root cause
identified from the generated PTX: `.local .align 4 .b8 __local_depot0[264]`
— because `m_tiles` is a runtime loop bound, LLVM cannot promote the
`[[f32;2];32]` accumulator to registers and spills it to local memory, adding
`ld.local`/`st.local` traffic on every MMA iteration. The hand kernel's
accumulators are named PTX registers throughout. Occupancy was *not* the
issue (50 reg vs 44, both 100% theoretical). A `const M_TILES` monomorphized
variant would likely recover much of this, but the track is closed per the
sprint brief.

**K-slicing / persistent accumulation — DEFERRED.**
Needs f32 atomics or a second reduction pass plus device scratch. New device
scratch violates glcuda's **zero `cuMemAlloc` after init** contract
(`memory-management.md` rules 1-2), which requires an `architecture/` spec
update and sign-off. Not justified while 20% of block slots are in use — the
free parallelism in item 1 above comes first.

**Double buffering / async prefetch — DEFERRED.**
`cp.async` is sm_80+; T4 needs a manual double-buffer (2x smem = 4608 B, which
does fit) plus extra barrier structure. Latency hiding is the wrong tool while
achieved occupancy is 25% for lack of blocks.

**Candidate A, "remove the `ntok <= 64` contract" — REJECTED in Phase 2.**
The contract was never the blocker (see §2 correction). Change 1B subsumed
the useful part without touching the 8-m-tile accumulator structure the limit
derives from.

## 8. Side findings

**`gl_gemm_mma_q8_r256` has an independent numerical bug.**
FAILs correctness on **all** cases with synthetic, always-**aligned** buffers:
max_abs_diff 15.74 / 54.375 / 136.05 against a 5.0e-2 tolerance, growing with
problem size. This is distinct from its known `CUDA_ERROR_MISALIGNED_ADDRESS`
crash ([runner.rs:156-165](../../glcuda/src/runner.rs#L156-L165)) — it
launches cleanly and computes wrong numbers. Per that comment r256's parity
test had **never run on real hardware**; this is the first time it has. The
kernel is not in production. **Action: report to the glcuda project; it is
independent of this sprint.** It was deliberately left unpatched — optimizing
a kernel that computes wrong numbers produces meaningless timings.

**Shared-memory staging bug found and fixed in the cuda-oxide port.**
The staging destination index omitted the pass offset (`row_in_pass` instead
of `row`), so with `stage_rows_total=256` all 4 staging passes wrote to the
same 2 KB of smem while `SM_A[2048..8191]` was never written. Invisible at
`m_tiles=8` (one pass) and at small `ntok` (later passes' write-gate
suppressed the collision); it only surfaced at `ntok=256`, corrupting output
completely (max_abs_diff 141.61).

**Non-deterministic numerics in the oxide kernel at large shapes.**
`m_tiles=32` case 3 flipped FAIL (1.4161e2) -> PASS (4.6913e-2) across two
runs on identical input. Mechanism: the staging bug above had **no `bar.sync`
between passes**, so warps racing at different passes produced genuinely
non-deterministic corruption. Consistent with the observed flip. Recorded per
the sprint's non-determinism rule; the favorable run was not taken.

**cudarc/cuda-oxide ABI mismatch (harness-level).**
cudarc's `.arg(&CudaSlice<T>)` pushes only a device pointer, but cuda-oxide
compiles Rust `&[T]` to a `(ptr, len)` fat pointer. Pointer-only arguments
desynced every subsequent parameter and segfaulted. Fixed by pushing explicit
`(ptr, len)` pairs via `DevicePtr`/`DevicePtrMut`. The hand kernels are
unaffected — their PTX takes one raw `.u64` per buffer, matching cudarc's
assumption.

## 9. Status and next step

- Gate 3 **closed**: +228% (case 3) and +256% (case 4), all cases PASS at
  `max_abs_diff < 5.0e-2`, including the ragged `ntok=200` tail case.
- **The monorepo PTX is unchanged and this result does not yet justify
  changing it.** Per Phase 2 the merge gate is **glbench `prefill_tps` on a
  real model via `--engine cuda`**, not this diagnostic harness. This repo has
  twice recorded a ~2x isolated win that went neutral in production
  (VNNI-512, row-tile GEMM, both in `rejected-optimizations.md`).
- **Blocked on:** a GGUF on the Kaggle box to run the glbench gate. Model not
  yet chosen.
- Note for that run: glbench cannot attribute time to this kernel — `glcuda`
  does not implement `GlEngine::telemetry()`, so the bucket roofline is
  unavailable on CUDA and only end-to-end `prefill_tps` is measurable
  (Phase 2 §0).
