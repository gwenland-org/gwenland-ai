# glcuda INT8 GEMM Ceiling Sprint — Phase 1 Diagnosis

> **Date:** 2026-08-04
> **Kernel:** `gl_gemm_mma_q8` ([glcuda/src/kernels/glcuda_sm75.ptx](../../glcuda/src/kernels/glcuda_sm75.ptx))
> **Hardware:** Kaggle T4 (sm_75, 40 SMs, 65 TOPS INT8, 320 GB/s HBM2)
> **Status:** Research only. No source file was modified in Phase 1.

## TL;DR — the sprint premise needs correcting before any code is written

Two findings, both high confidence, change what this sprint should be:

1. **The Cell 9 baseline numbers do not measure kernel throughput.** They
   measure launch overhead. Case 1 and case 2 differ by **224x in work** but
   only **1.9x in measured time**. That is arithmetically impossible for a
   kernel-bound measurement and is conclusive on its own — no overhead
   subtraction or re-run required to establish it.
2. **65 TOPS is not a reachable ceiling for any of these three shapes.**
   Two of the three are memory-bound by arithmetic intensity, and all three
   are grid-capped to 2.5% / 10% / 20% of the GPU's SMs. The "<1% of ceiling"
   framing compares against a number that is unattainable *by construction*
   at these problem sizes — not against a number the kernel is failing to
   reach.

The real optimization target exists and is worth pursuing, but it is
smaller than 125x and it lives almost entirely in **case 3**, which is the
only compute-bound shape. Details below.

---

## 1A — Launch Geometry Audit

Source: [`glcuda/src/kernels/mod.rs:584`](../../glcuda/src/kernels/mod.rs#L584) (production launch)

```rust
// 256 threads = 8 warps = 8 output tiles of 8 rows per block.
cuda.launch(*f, (ceil_div(out_dim, 64), 1, 1), (256, 1, 1), 0, &mut params)
```

**Grid/block per case.** Grid depends only on `out_dim` — `ntok` and `in_dim`
do not affect it at all:

| case | out_dim | grid (blocks) | block | warps/block | SMs used | **% of 40 SMs** |
|---|---|---|---|---|---|---|
| 1 | 64 | 1 | 256 | 8 | 1 | **2.5%** |
| 2 | 256 | 4 | 256 | 8 | 4 | **10%** |
| 3 | 512 | 8 | 256 | 8 | 8 | **20%** |

**Warps per SM at launch:** 8 warps (one block) per SM. T4 supports 32
resident warps/SM, so a single block occupies 25% of one SM's warp slots.
With one block per SM there is nothing to fill the other 75%.

**How many SMs for case 2:** **4 of 40.** 36 SMs are idle for the entire
kernel.

**Theoretical occupancy (44 reg/thread, 2304 B smem/block, 256 thr/block):**
- register limit: 65536 / (44 x 256) = 5.8 -> 5 blocks/SM
- smem limit: 65536 / 2304 = 28 blocks/SM
- thread limit: 1024 / 256 = 4 blocks/SM  <- binding
- => **4 blocks/SM, 1024 threads/SM = 100% theoretical occupancy**

This confirms the `ptxas -v` derived occupancy from the notebook's Cell 8.
**Occupancy is not the problem** — the kernel could host 4 blocks per SM, but
the grid only ever supplies 1, 4, or 8 blocks *total across the whole GPU*.
Per-SM occupancy is 100% on paper and ~25% in practice purely because the
grid is too small to fill even one block-slot per SM, let alone four.

**Is `ntok <= 64` a correctness constraint or an artificial cap?**
**Structural, but NOT a blocker** — and the sprint brief's framing of this is
incorrect. The kernel has 8 m-tiles x 8 rows = 64 rows of accumulator
registers; that is a genuine structural limit of *this* kernel body. But
production does **not** refuse `ntok > 64` — it chunks
([`glcuda/src/runner.rs:166-180`](../../glcuda/src/runner.rs#L166-L180)):

```rust
let mut t0 = 0u32;
while t0 < n {
    let nn = (n - t0).min(64);
    k.gemm_mma_q8(cuda, wqs, wsc, /* x offset by t0 */ ..., nn)?;
    t0 += nn;
}
```

`PREFILL_BATCH` is **512** ([`model.rs:431`](../../glcuda/src/model.rs#L431)),
not 64. So case 3 (`ntok=256`) is fully supported in production as 4 chunked
launches. It was SKIPPED in Cell 9 only because *my notebook harness* did not
implement the chunk loop that `runner.rs` already has. That is a harness gap,
not a kernel contract gap.

This matters for Phase 2: **"remove the ntok<=64 contract" (candidate A) is
not needed to unlock case 3.** The real question is whether chunking costs
throughput — see 1B, where it does, measurably and for a specific reason.

---

## 1B — Arithmetic Intensity

**Unit convention (correcting the brief).** The brief gives
"MACs = out x in x ntok x 2" and "ridge = 65 TOPS / 320 GB/s = ~203
MACs/byte". Those two are inconsistent: `out x in x ntok x 2` is **ops**
(2 ops per MAC), and 65e12/320e9 = 203 is **ops/byte**. In MAC terms the
ridge is 101.6 MACs/byte. Both are fine as long as they match. **This
document uses ops/byte throughout, ridge = 203 ops/byte**, which matches
both the brief's ridge figure and `bench.rs`'s existing
`ops = 2.0 * (ntok * out_dim * in_dim)` convention.

Byte accounting includes the full Q8_0 SoA layout: int8 `qs` + f16 weight
scales (2 B per 32-elem block) + int8 activations + f32 activation scales
(4 B per block) + f32 output.

| case | ops | bytes | **intensity** | vs ridge 203 | verdict |
|---|---|---|---|---|---|
| 1 (64/128/8) | 131,072 | 11,904 | **11.0** | 18x below | **MEMORY-BOUND** |
| 2 (256/896/64) | 29,360,128 | 373,760 | **78.6** | 2.6x below | **MEMORY-BOUND** |
| 3 (512/4096/256) unchunked | 1,073,741,824 | 3,932,160 | **273** | 1.3x above | **COMPUTE-BOUND** |
| 3 as actually launched (4x64 chunks) | 1,073,741,824 | 10,616,832 | **101** | 2.0x below | **MEMORY-BOUND** |

### The chunking penalty is real and quantified

Case 3 is the **only** shape with genuine compute intensity — and chunking
at 64 rows destroys it. Each of the 4 chunks re-reads the **entire weight
matrix** from DRAM (2.1 MB qs + 131 KB scales), so weight traffic is 4x what
the math requires: 3.93 MB -> 10.6 MB. That drags intensity from 273 ops/byte
(above the ridge, compute-bound) to 101 ops/byte (below it, memory-bound).

This is precisely the mechanism Acceleratio Stellarum Phase B / `r256` was
designed to address, and it is now measured rather than asserted.

### Attainable ceilings — the number that actually matters

"% of 65 TOPS" is the wrong denominator. The attainable ceiling for each case
is bounded by **both** the memory roofline **and** the fraction of SMs the
grid actually occupies:

| case | roofline max | SM-fraction max | **attainable ceiling** | as % of 65 TOPS |
|---|---|---|---|---|
| 1 | 3.52 TOPS (5.4%) | 2.5% -> 1.63 TOPS | **~1.6 TOPS** | **2.5%** |
| 2 | 25.1 TOPS (38.7%) | 10% -> 6.5 TOPS | **~6.5 TOPS** | **10%** |
| 3 (chunked) | 32.3 TOPS (49.7%) | 20% -> 13 TOPS | **~13 TOPS** | **20%** |

So for case 2, the measured "0.8% of ceiling" should be read as **0.8 / 10 =
~8% of what is physically attainable at that shape and grid** — and even that
is inflated by the overhead problem in 1C-1. The honest gap is real but it is
roughly one order of magnitude, not 125x.

---

## 1C — Root Cause Hypotheses (ranked)

**1. [HIGH] The baseline measurement is invalid — it measures launch overhead, not the kernel.**

The Cell 9 harness (my notebook `runner/src/main.rs`) times **one un-warmed
launch** with a wall clock wrapped around `launch()` + `synchronize()`:

```rust
let t0 = Instant::now();
let mut builder = stream.launch_builder(&f_mma8);
/* ...8x .arg()... */
unsafe { builder.launch(cfg8) }.unwrap();
stream.synchronize().unwrap();
let elapsed = t0.elapsed();     // includes builder setup + launch + full sync
```

The authoritative harness, [`bench.rs:642-648`](../../glcuda/examples/bench.rs#L642-L648),
does the opposite and is correct:

```rust
let iters = 50;
let t = Instant::now();
for _ in 0..iters { k.gemm_mma_q8(...)?; }   // 50 launches, pipelined
cuda.synchronize()?;                          // ONE sync at the end
let mma = t.elapsed().as_secs_f64() / iters as f64;
```

...and it runs a correctness launch first, which warms the module.

**The proof that overhead dominates, requiring no re-run:** case 1 does
65,536 MACs; case 2 does 14,680,064 MACs — **224x more work**. Measured time
went 32.0us -> 62.7us (run 1) and 31.1us -> 59.4us (run 2): **1.9x**. A
kernel-bound measurement scales with work. This one does not, so a large
fixed per-launch cost is the dominant term in every Cell 9 figure. Case 1 —
which performs essentially no work — costing ~31us *is* that floor, directly
observed.

Every TOPS and "% of ceiling" number in the sprint brief inherits this, and
project rule `measurement-discipline.md` (probes have ranged 0.07x-2.40x vs
reality) says exactly this class of number must not be trusted.

**2. [HIGH] Grid underutilization — the grid ignores `ntok` entirely.**

`grid = ceil_div(out_dim, 64)` means work is parallelized *only* across
output rows. Case 2 launches **4 blocks onto a 40-SM GPU**; 36 SMs idle.
The `ntok` dimension — 64 rows of independent token work — is handled
*serially inside each block's m-tile loop* instead of being spread across
SMs. This is a genuine structural limitation and, unlike #1, it survives a
correct re-measurement. It is the top *kernel-side* cause.

**3. [HIGH] Chunking at 64 rows makes the only compute-bound case memory-bound.**

Quantified in 1B: case 3's intensity drops 273 -> 101 ops/byte because each
64-row chunk re-streams all 2.2 MB of weights. Fixing this is what `r256`
attempts. Note it is *not* a "remove the contract" problem — the contract is
already worked around by chunking; the cost is the redundant DRAM traffic.

**4. [MEDIUM] `gl_gemm_mma_q8_r256` is numerically broken, so the obvious fix to #3 is currently unavailable.**

Cell 9 shows r256 FAILing parity on **all three** cases, with error growing
with problem size (15.74 / 54.375 / 136.05, tolerance 5e-2). This harness
uses synthetic, always-aligned buffers, so this is **not** the known
`CUDA_ERROR_MISALIGNED_ADDRESS` crash documented at
[`runner.rs:156-165`](../../glcuda/src/runner.rs#L156-L165) — it launched
successfully and computed wrong numbers. Per that same comment, r256's
parity test has **never been run on real hardware** before now. This is a
new, independent finding: r256 appears to have a numerical bug *in addition
to* the alignment crash that got it reverted from production.

**5. [MEDIUM] Two of three benchmark shapes cannot demonstrate tensor-core value at all.**

Cases 1 and 2 sit 18x and 2.6x below the ridge point. Tensor cores raise the
compute ceiling; they cannot move a memory-bound kernel. Optimizing against
these two shapes will produce misleading conclusions no matter how carefully
it is measured. The benchmark set itself needs a compute-bound shape that is
run *unchunked* to be a meaningful ceiling test.

**6. [LOW] Register/occupancy tuning.**

44 registers, 2304 B smem, 100% theoretical occupancy, 0 spill (`ptxas -v`).
There is nothing here to win. Explicitly listed to close it off — per
`kernel-design.md` rule 3, chasing occupancy on a kernel that is already at
100% theoretical and is grid-starved would be cargo cult.

### Non-determinism flagged (per project rule)

The brief reports oxide `m_tiles=32` flipping FAIL (`1.4161e2`) -> PASS
(`4.6913e-2`) between runs. I have a mechanism for this, found while fixing
the cuda-oxide port: the shared-memory staging destination index omitted the
pass offset, so with `stage_rows_total=256` all 4 staging passes wrote to the
same 2 KB of smem with **no `bar.sync` between passes** — different warps
racing at different passes gives genuinely non-deterministic corruption. That
is consistent with a FAIL/PASS flip on identical input. It is fixed in the
notebook's kernel source, but it is a cuda-oxide-path bug and per the brief
that path is closed, so it does not affect this sprint beyond explaining the
flip.

---

## What I recommend for Phase 2 (preview, not a commitment)

Given the above, the highest-value work is **not** "close a 125x gap":

- **First, re-baseline using `bench.rs` methodology** (warmed, 50-iter
  amortized) so Phase 3 has a trustworthy before-number. Without this, any
  delta claimed in Gate 3 is unfalsifiable. This is cheap and blocks
  everything else.
- **Then attack #2 (grid underutilization)** — parallelize across `ntok` as
  well as `out_dim` so the grid can fill 40 SMs. This is the single largest
  *real* kernel-side win and it helps every shape.
- **Then #3 (chunking traffic)**, which is where case 3's compute-bound
  headroom actually is.
- **#4 (r256 numerical bug)** should be reported to the glcuda project
  regardless of whether this sprint fixes it.

Candidate A from the brief ("remove ntok<=64 contract") should be **dropped
as stated** — the contract is not what blocks case 3.

**Gate 1: awaiting confirmation before Phase 2.**
