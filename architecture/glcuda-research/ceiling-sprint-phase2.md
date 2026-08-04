# glcuda INT8 GEMM Ceiling Sprint — Phase 2 Plan

> **Date:** 2026-08-04
> **Depends on:** [ceiling-sprint-phase1.md](ceiling-sprint-phase1.md) (Gate 1 confirmed)
> **Status:** Plan only. No source file modified in Phase 2.

## 0. Instrument decision — glbench vs bench.rs

Requested: re-baseline with `glbench` rather than `bench.rs`. I verified what
glbench can actually do against `glcuda` before planning around it. The
instinct is right — and it is also **not sufficient on its own**. Both are
needed, for different jobs.

### What glbench CAN do here (verified)

- Run against the CUDA engine (`--engine cuda`) — glbench depends on `glcuda`
  ([`glbench/Cargo.toml`](../../glbench/Cargo.toml)), which self-probes and
  reports availability.
- Report **end-to-end `prefill_tps`** through the real production path
  ([`core/metrics.rs:30`](../../glbench/src/core/metrics.rs#L30)).
- `glbench ab` for a real A/B; T4 is in the bandwidth capability table
  ([`engine/capability.rs:20`](../../glbench/src/engine/capability.rs#L20)).

### What glbench CANNOT do here (verified, and it changes the plan)

1. **No per-kernel or per-stage attribution on GPU.** `GlEngine::telemetry()`
   defaults to `None` ([`engine_trait.rs:108`](../../glcore/src/engine_trait.rs#L108))
   and **`glcuda` never implements it** — `grep -rn "telemetry" glcuda/src/*.rs`
   returns zero matches. glbench's profiling enabler sets `GLPROC_PROFILE`
   and is explicitly glproc-only
   ([`engine/adapter.rs:230`](../../glbench/src/engine/adapter.rs#L230)).
   Therefore `RooflineReport::compute()` returns `None` on CUDA: **no bucket
   roofline, no `intensity_flop_per_byte`, no ceiling verdicts, no way to
   separate `gl_gemm_mma_q8` from attention or norms.**
2. **`scale --sweep` is the wrong axis.** It sweeps `max_new_tokens`
   ([`runner/scale.rs:45`](../../glbench/src/runner/scale.rs#L45)) — the
   *decode* length. It does not vary prompt length, so it never moves the
   prefill GEMM's `ntok`. Prefill batch size is driven by `--prompt` text.
3. **Cannot express the sprint's headline metric.** glbench's ceiling
   machinery is **bandwidth (GB/s)**, and on GPU it is `EstimatedFromTable` —
   a vendor spec the code itself warns runs 60-85% in practice. There is no
   INT8-TOPS ceiling anywhere in glbench, so "% of 65 TOPS" cannot come from
   it.
4. **Cannot measure the three sprint shapes at all.** They are synthetic
   (`out=64/in=128/ntok=8`), not model shapes; there is no kernel-level entry
   point.

### Decision: both instruments, different authority

| | `bench.rs` (fixed) | `glbench` |
|---|---|---|
| **Role** | diagnostic | **decision / merge gate** |
| Isolates the GEMM | yes | no (end-to-end only) |
| Sprint's 3 synthetic shapes | yes | no |
| "% of 65 TOPS" | yes | no (bandwidth only) |
| Real production path | no | **yes** |
| Authoritative for merge | **no** | **yes** |

This split is what the project's own rules already demand:
`kernel-design.md` rule 8 ("measure on production decode/prefill, not
standalone kernel probes") and the sprint's own hard rule that only the
production A/B justifies merging. **glbench is promoted to the merge gate;
`bench.rs` is demoted to diagnosis.** That is a stronger position than the
brief's original "bench.rs is authoritative", and it matches the request.

**Precedent that makes this non-negotiable:** this repo has twice recorded an
optimization that was **~2x faster in isolation and neutral in production**
(VNNI-512, row-tile GEMM — both in `rejected-optimizations.md`). A `bench.rs`
win is *not* evidence for merging. Only the glbench prefill number is.

**Prerequisite:** glbench needs a real GGUF on the Kaggle box (`*.gguf` is
gitignored). Step 0b below covers fetching one.

---

## 1. Re-baseline protocol (Step 0 — blocks everything else)

### 0a. Fix `bench.rs` methodology and add the sprint shapes

The existing `[mma gate]` block already uses the correct pattern — warmup
launch, `iters = 50`, one `synchronize()` at the end, divide by iters
([`bench.rs:642-648`](../../glcuda/examples/bench.rs#L642-L648)). It just
does not cover the sprint's shapes. Add a shape table
(`64/128/8`, `256/896/64`, `512/4096/256`) driven through that same
amortized pattern, reporting us + TOPS + max_abs_diff per shape.

`512/4096/256` must be driven **exactly as production does it** — through the
`runner.rs` 64-row chunk loop — so the baseline includes the chunking cost
that Phase 1 identified. Anything else measures a kernel production never
runs.

### 0b. Establish the glbench production baseline

```
glbench run --engine cuda --model <model>.gguf --prompt <long prompt> --tokens 1
```

A long prompt with `--tokens 1` maximizes the prefill share and minimizes
decode contamination — the closest glbench can get to a prefill-only
measurement given constraint #1 above. Record `prefill_tps`. Repeat >= 2x;
per `feedback_glproc_benchmark_traps`, this machine class drifts ~24% between
sessions, so **only same-session numbers may be compared.**

**Exit criterion for Step 0:** a trustworthy before-number from *both*
instruments, in the same session. Without this, no Gate 3 delta is
falsifiable.

---

## 2. Proposed changes

### Change 1 (PRIMARY) — 2D grid over (out_dim, ntok)

- **Problem solved:** Phase 1 RC#2 (grid ignores `ntok`; 8 of 40 SMs on case
  3) and partially RC#3 (chunking traffic).
- **Mechanism:** `grid.y = ceil_div(ntok, 64)`; the kernel derives its token
  row offset from `%ctaid.y` instead of the host offsetting pointers per
  chunk. This converts the **4 sequential launches** the runner does today
  into **1 launch with 4x the blocks**. Case 3 goes 8 blocks -> **32 blocks
  (80% of 40 SMs)**. Second-order benefit: the 4 token-chunks now run
  *concurrently* over the same 2.2 MB weight matrix, which **fits in the T4's
  4 MB L2** — the redundant weight reads that Phase 1 measured (273 -> 101
  ops/byte) become L2 hits instead of DRAM re-reads.
- **Risk:** MEDIUM-LOW. The MMA math, `m8n8k16` fragment layout, scale
  epilogue, and accumulator structure are all **untouched** — only index
  derivation and launch geometry change. Real risks: (a) the tail block when
  `ntok % 64 != 0` needs correct predication; (b) the "x rows allocated to
  round8(ntok)" contract must still hold per-block; (c) `runner.rs`'s chunk
  loop must be removed in the same commit or work will be done twice.
- **Estimated gain (roofline-bounded, upper bound — must be measured):**
  case 3 only. SM utilization 20% -> 80% is a 4x parallelism increase; the
  memory floor at full BW is 12.3 us (unchunked) vs a compute floor of
  16.5 us, so with L2 absorbing the repeats the shape becomes compute-bound
  again and the bound at 80% SMs is **~20.6 us**. Against a current
  4-launch baseline this suggests **2-4x**, capped by how much of the weight
  re-read L2 actually absorbs. **Cases 1 and 2 get `grid.y = 1` and are
  expected to be exactly neutral** — that is a correct outcome, not a
  failure, and must not be read as one.

### Change 2 (CONDITIONAL) — finer out_dim tiling for small shapes

- **Problem solved:** RC#2 for case 2, which Change 1 does not help
  (`ntok=64` -> `grid.y=1`).
- **Mechanism:** fewer output rows per block (e.g. 4 warps/128 threads = 32
  rows) raises case 2 from 4 to 8 blocks. Splitting `out_dim` duplicates
  **activation** reads (64.5 KB for case 2), not weight reads (243.7 KB) —
  the cheaper of the two axes to duplicate, which is why this is preferred
  over splitting `ntok` for this shape.
- **Risk:** MEDIUM. Block size is load-bearing for the staging code (256
  threads stage 64 rows x 32 B); changing warps/block means rewriting
  staging and re-deriving the smem layout. This is exactly the kind of edit
  that produced the cuda-oxide staging bug found in the notebook.
- **Gate:** **only attempt if Step 0's corrected numbers show case 2 is
  genuinely kernel-bound.** Phase 1's arithmetic says case 2 is a 29 MOP
  problem = 0.45 us of math at peak; it is very likely to remain
  launch-latency-dominated no matter what the kernel does, in which case
  this change is unwinnable by construction and should be dropped rather
  than attempted.

---

## 3. Rejected candidates (with reasons)

**A. Remove the `ntok <= 64` contract — REJECTED as stated.**
Phase 1 established the contract is not what blocks case 3: `runner.rs:166`
already chunks, and `PREFILL_BATCH` is 512. Case 3 was skipped in my
*notebook harness*, not by the kernel. Change 1 subsumes the useful part of
this (making the chunks concurrent) without touching the 8-m-tile
accumulator structure that the 64-row limit actually comes from.

**C. Shared-memory K-slicing / persistent accumulation — DEFERRED.**
Splitting K across blocks needs either f32 atomics or a second reduction
pass plus scratch. New device scratch collides with glcuda's **zero
`cuMemAlloc` after init** contract (`memory-management.md` rules 1-2), which
per that skill requires an `architecture/` spec update and sign-off. Not
justified while 32 of 40 SMs are still idle — fix the free parallelism
first.

**D. Double buffering / async prefetch — DEFERRED.**
`cp.async` is sm_80+; T4 would need a manual double-buffer (2x smem: 4608 B,
which does fit) plus extra `bar.sync` structure. Latency hiding is pointless
while the grid leaves most of the GPU idle — this only becomes worth
measuring after Change 1.

**r256's numerical bug — REPORT, DO NOT FIX HERE.**
Phase 1 RC#4: r256 failed parity on all 3 cases on aligned synthetic buffers
(errors 15.74 / 54.375 / 136.05, growing with size), which is distinct from
its known misaligned-address crash. It is not in production and not on this
sprint's path. It gets written up in the Phase 4 summary and carried to the
glcuda project separately.

---

## 4. Implementation order (Phase 3)

1. **Step 0a** — fix/extend `bench.rs` shapes (diagnostic baseline).
2. **Step 0b** — glbench production baseline, >= 2 repeats, same session.
3. **Change 1** — in one commit: PTX kernel index derivation + `mod.rs`
   launch geometry + remove `runner.rs`'s chunk loop.
4. **Parity** — `cargo test -p glcuda --test parity -- --test-threads=1`.
   Must add a **ragged `ntok`** case (e.g. 200, not a multiple of 64) to
   exercise the `grid.y` tail block, alongside the existing dim-896 shape.
   Per `testing-standards.md` rule 8, this test must fail on the pre-change
   kernel to be worth anything.
5. **Diagnostic** — `bench.rs` A/B, all shapes still PASS at
   `max_abs_diff < 5.0e-2`.
6. **Decision** — glbench `prefill_tps` A/B on the real model. This is the
   merge gate; a `bench.rs`-only win does not qualify.
7. **Change 2** — only if step 5's numbers justify it (see its gate).

### PTX rules in force (from `ptx-writing.md`)

Pure ASCII + LF; unique `%r_`-prefixed register names per kernel section, no
duplicate `.reg` in one body; kernel header comment updated with the **new
launch geometry** (this change alters it, so the header is part of the
diff); `mma.sync` stays in `glcuda_sm75.ptx`; `m8n8k16` A row-major s8 /
B col-major s8 / s32 accumulate contract unchanged.

---

## 5. What success looks like

- **Case 3:** first-ever production-path number for this shape, PASS at
  tolerance, with a measured speedup over the 4-launch baseline.
- **Cases 1 and 2:** unchanged within noise. Explicitly expected.
- **glbench `prefill_tps`:** improved on a real model, same session, >= 2
  repeats. If this is neutral while `bench.rs` improved, the change **does
  not merge** and goes to "Rejected Approaches" with its numbers — the
  VNNI-512 / row-tile outcome, which has happened twice here already.

**Gate 2: awaiting confirmation before Phase 3.**
