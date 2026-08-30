# sm_75 INT8 MMA memory-pipeline fixes

## Problem

Both Turing INT8 MMA GEMMs emitted two scalar output stores for every adjacent
column pair and read activation fragments through a shared-memory layout with
two-way bank conflicts. The 8-tile kernel also loaded each B fragment directly
before its first `mma.sync`, leaving global-memory latency exposed.

## Root cause

The output epilogue did not express the guaranteed 8-byte lane-pair alignment.
The activation rows used a 32-byte pitch, mapping the second 16 lanes onto the
same 16 shared-memory banks as the first 16. The existing B-fragment software
pipeline had only been added to the r256 kernel.

## Fix

- Replaced all 40 scalar output-store pairs with `st.global.v2.f32`.
- Changed the activation shared-memory pitch to 48 bytes. For the MMA lane map,
  the eight row groups start at word banks 0, 12, 24, 4, 16, 28, 8, and 20;
  their four-word fragments cover all 32 banks exactly once. Unlike a 36-byte
  pitch, 48 bytes also keeps every `st.shared.u64` staging destination aligned.
- Backported the r256 B-fragment and scale software pipeline to the 8-tile
  kernel without moving either `bar.sync`.
- Kept all five global-u64-to-shared-u64 activation staging paths intact.

The resulting per-block shared-memory use is 3,328 bytes for
`gl_gemm_mma_q8` and 13,312 bytes for `gl_gemm_mma_q8_r256`, both within the
T4's 64 KiB shared-memory budget.
