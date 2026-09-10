# Gwen Changes - 2026-09-01 - T4 Ceiling Wave 24

## Added

- Added `notebooks/glcuda_t4_wave24_mma2_gemm.ipynb`, a self-contained Tesla
  T4 reconstruction and production gate for independent K16 MMA accumulator
  chains in the retained B-stage GEMM.
- Added the frozen Wave 24 design gate and complete production result under
  `architecture/glcuda-research/`.
- Added structural, compiler, spill/stack, dual-launch residency, direct
  bit-exact, full CUDA parity, dispatch, token-oracle, decode and tail gates.

## Validation

- The local reconstructed stack passes the Wave 24 direct example check and
  66/66 `glcuda` library tests.
- The final notebook reproduces the 39,703-byte Wave 24 patch at SHA-256
  `b161faf1e0bc88688c320e423ef1b3f4ade1ebdf1850895910cb195e6d0e8ee8`.
- On Tesla T4, the candidate assembles with 50 registers, 9,728 B static
  shared memory, zero stack and zero spills. Driver residency remains four
  blocks/SM at 256 threads and two blocks/SM at 512 threads.
- Kaggle version 1 passes 66/66 library tests, 33/33 CUDA parity tests,
  bit-exact f32 output at all three registered shapes and every 50/50 token
  oracle.
- Production dispatch observes retained `mma4-fused` attention in both arms
  and `bstage-mma2` only in the candidate arm.

## Decision

Wave 24 is **REJECTED**. The candidate measures 10,473.7 tok/s versus
10,542.5 tok/s for retained K32, a -0.65% median delta. Paired deltas are
-2.292%, +0.854%, -0.013% and -1.562%; only one of four is positive. Median
session-max latency regresses 1.45%, while decode and correctness gates pass.

The retained K32 B-stage GEMM stays the production default. No second run is
required and no candidate product code is promoted; the notebook preserves
the rejected implementation and evidence for reproducibility.
