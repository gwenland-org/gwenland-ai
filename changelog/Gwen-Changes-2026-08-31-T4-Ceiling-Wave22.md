# Gwen Changes - 2026-08-31 - T4 Ceiling Wave 22

## Added

- Added `notebooks/glcuda_t4_wave22_fused_swiglu.ipynb`, a self-contained
  Tesla T4 reconstruction that repairs Wave 21's illegal special-register
  shift without changing the fused-SwiGLU computation or launch policy.
- Added the frozen Wave 22 design gate and the complete production result in
  `architecture/glcuda-research/`.
- Added a structural regression assertion for the corrected `%ctaid.x` move,
  plus unchanged compiler, occupancy, exact-parity, CUDA-parity, dispatch,
  token-oracle, decode and tail gates.

## Validation

- The local reconstructed stack passes the Wave 21 direct example check and
  64/64 `glcuda` library tests.
- The final notebook reproduces the 4,223-byte repair patch at SHA-256
  `07c60277043e66874ddfdc36a0fcfa4057bdb6f8c7a73372064bd359411d6239`.
- On Tesla T4, the repaired candidate assembles with 57 registers, 16,384 B
  static shared memory, zero stack, zero spills and two active blocks per SM.
- Kaggle version 2 passes 64/64 library tests, 33/33 CUDA parity tests, exact
  Q8 byte/scale parity at all three registered shapes and every 50/50 token
  oracle.
- The production dispatch audit observes retained `mma4-fused` attention in
  both arms and `fused-swiglu-q8` only in the candidate arm.

## Decision

Wave 22 is **WEAK**, not retained. Fused SwiGLU measures 10,389.0 tok/s versus
10,310.6 tok/s for the retained path, a +0.76% median delta. Paired deltas are
+0.58%, -0.08%, +1.61% and +1.37%; the result misses both the +5% threshold
and the every-pair-positive requirement. The retained SwiGLU path stays the
production default, so no second confirmation run is required.
