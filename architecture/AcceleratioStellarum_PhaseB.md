# Acceleratio Stellarum — Phase B: Weight-Reuse GEMM (design spec)

Decision: **measure 16 vs 32 m-tiles empirically on the T4.** The register
model proves both are feasible; only hardware settles whether the halved
occupancy at 32 tiles costs achieved bandwidth on a kernel we have measured
to be weight-BW-bound.

## Evidence base (all T4-measured, 2026-07-12)

- `[gemm-reuse]` time/token falls ~45% per ntok doubling (gate_up
  205.7→121.4→56.2→36.9 us/tok at n8/16/32/64; down 95.9→51.6→29.0→18.9).
  Steeply-falling time/token = **weight-BW-bound → reuse pays.**
- JIT `ptxas -v`: `gl_gemm_mma_q8` = **44 registers, 2304 B smem, 0 spills.**
- T4 sm_75: 65536 registers/SM, 64 KB smem/SM, 32 warps/SM max.

## Register / occupancy model (measured slope: ~1 reg per accumulator)

| m-tiles | rows/read | accum | ~regs | blocks/SM | occupancy | re-streams (512 chunk) |
|--------:|----------:|------:|------:|----------:|----------:|-----------------------:|
| 8 (now) | 64        | 16    | 44    | 5 (reg) / cap 4 | ~100% | 8 |
| **16**  | 128       | 32    | ~60   | 4         | ~100%     | 4 |
| 24      | 192       | 48    | ~76   | 3         | ~75%      | 3 (⌈512/192⌉) |
| **32**  | 256       | 64    | ~92   | 2         | ~50%      | 2 |

smem scales with rows: `sm_a` = rows×32 B, `sm_xs` = rows×4 B. At 256 rows
= 8192 + 1024 = 9.2 KB → 7 blocks/SM by smem, never the limit.

## The tension the A/B resolves

- Reuse win is **sub-linear** in m-tiles: 8→16 halves re-streams (8→4),
  16→32 only halves again (4→2). First doubling captures more absolute
  traffic.
- Occupancy loss is **super-linear** near the top: 8→16 is ~free (still
  ≥100% warp fill), 16→32 drops to 50%. On a BW-bound kernel, occupancy is
  the latency-hiding budget — halving it may *lower* achieved GB/s and eat
  the reuse gain. This second-order effect is unmodelable; hence the A/B.

## Implementation plan

1. **One kernel, 32 m-tiles** (`gl_gemm_mma_q8_r256`): mechanical extension
   of the existing 8-tile unroll — 24 more compute blocks (identical
   12-instr pattern, accumulators %f{10+2m}/%f{11+2m}, guard %p{m}), 24 more
   write-back blocks, smem `sm_a[8192]` + `sm_xs[1024]`. Body GENERATED
   (build script or python emitter), not hand-typed, so the repetition is
   provably correct. Contract: ntok ≤ 256, x rows allocated to round8(ntok).
2. **16-tile behavior comes free** from the same kernel via the ntok guards:
   call it with ntok≤128 and only 16 m-tiles activate. So the A/B needs ONE
   new kernel, not two — 128-row and 256-row are two call shapes of it.
3. **A/B harness** (`bench [gemm-phaseb]`): process a fixed 512-row chunk of
   gate_up and down three ways — 8 calls×64 (baseline), 4 calls×128, 2
   calls×256 — report total us and achieved GB/s each. Winner = lowest total
   time for the full chunk. This directly pits reuse against occupancy at
   the real work size.
4. **Wire the winner** into `gemm_rows` sub-slab loop (replace the hardcoded
   64 with the measured-best tile), gated on `has_mma()` and the ntok cap.

## Validation

- Parity: new kernel vs CPU reference at ntok=5/64/128/256 (ragged +
  full), EPS_MATMUL — same as the existing MMA parity test.
- `batched_prefill_matches_glproc` (530-tok) must stay green.
- glbench prefill archive before/after; `[prefill split]` FFN bucket must
  drop; decode untouched.

## Predicted outcome (analytical — to be validated)

If occupancy holds up: 4-call (128-row) roughly halves the FFN weight
traffic vs the 8-call baseline; 2-call (256-row) up to 4× but at 50%
occupancy. The A/B decides which net-wins. Prefill 348 →, conservatively,
450-600 tok/s if the 128-row point wins clean; higher only if 256 does not
lose bandwidth to occupancy. No 900-1400 claim until the archive says so.
