# Gwen Changes - 2026-09-01 - T4 Ceiling Wave 25

## Added

- Added `notebooks/glcuda_t4_wave25_v16_stage.ipynb`, a self-contained Tesla
  T4 reconstruction and production gate for 16-byte cooperative A/B staging
  in the retained K32 B-stage GEMM.
- Added the frozen Wave 25 design gate and complete production result under
  `architecture/glcuda-research/`.
- Added structural, ptxas, spill/stack, dual-launch residency, exact-dimension
  direct, full CUDA parity, dispatch, token-oracle, decode and tail gates.

## Validation

- The repaired local cumulative stack passes 66/66 `glcuda` library tests;
  the candidate PTX remained byte identical across the helper-only repair.
- A static audit rejected notebook v1 before upload because its stabilization
  call scheduled two iterations. Executed v2 uses exactly 0 cold, 0 warmup and
  1 measured stabilization invocation per arm while retaining the same
  38,930-byte Wave 25 patch at SHA-256
  `6e956d09c37230c873720d867561940f5d47419f6f816296518fb465eddcb1de`.
- Kaggle version 1 passes 66/66 library tests, 33/33 CUDA parity tests,
  bit-exact f32 output at all three registered shapes and every 50/50 token
  oracle.
- On Tesla T4, V16 assembles with 50 registers, 9,728 B static shared memory,
  zero stack and zero spills. Driver residency remains four blocks/SM at 256
  threads and two blocks/SM at 512 threads.
- Both direct production shapes pass the 0.95x screen: gate/up measures
  -1.48%, while down measures +1.05% versus retained.
- All eight production sessions use the frozen 244-token, 5 cold, 5 warmup
  and 10 measured protocol with Wave 20 attention retained in both arms.

## Decision

Wave 25 is **WEAK / NOT RETAINED**. V16 measures 10,386.1 tok/s versus
10,271.5 tok/s for retained K32, a +1.116% ratio of session-P50 medians.
Paired deltas are -0.925%, +1.265%, +1.479% and -0.794%; only two of four are
positive, and their median is +0.235%.

Correctness, decode and tail gates pass, but the candidate misses both the +5%
retention bar and the all-positive requirement. Retained K32 stays the
production default, no second run is required, and no candidate product code
is promoted.
