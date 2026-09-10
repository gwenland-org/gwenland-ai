# Gwen Changes - 2026-08-31 - T4 Ceiling Wave 23

## Added

- Added `notebooks/glcuda_t4_wave23_k128_gemm.ipynb`, a self-contained Tesla
  T4 reconstruction and production gate for a four-block K128 B-stage GEMM
  superstage.
- Added the frozen Wave 23 design gate and complete production result under
  `architecture/glcuda-research/`.
- Added structural, compiler, spill/stack, dual-launch residency, direct
  bit-exact, full CUDA parity, dispatch, token-oracle, decode and tail gates.

## Validation

- The local reconstructed stack passes the Wave 23 direct example check and
  65/65 `glcuda` library tests.
- The final notebook reproduces the 33,265-byte Wave 23 patch at SHA-256
  `90927231489e073259796131f3d04971f5281a2596fc843a116ad6c51fca2ad6`.
- On Tesla T4, the candidate assembles with 64 registers, 38,912 B static
  shared memory, zero stack and zero spills. The driver reports one active
  block per SM at both 256 and 512 threads.
- Kaggle version 2 passes 65/65 library tests, 33/33 CUDA parity tests,
  bit-exact f32 output at all three registered shapes and every 50/50 token
  oracle.
- Production dispatch observes retained `mma4-fused` attention in both arms
  and `bstage-k128` only in the candidate arm.

## Decision

Wave 23 is **REJECTED**. K128 measures 8,681.1 tok/s versus 10,097.7 tok/s for
the retained K32 path, a -14.03% median delta. All four paired deltas are
negative (-15.49%, -14.06%, -13.44%, -13.59%), and median session-max latency
regresses 15.18%. The direct `ffn_down` kernel is 33.11% slower.

The retained K32 B-stage GEMM stays the production default. No second run is
required and no candidate product code is promoted; the notebook preserves
the rejected implementation and evidence for reproducibility.
