# Gwen Changes - 2026-08-31 - T4 Ceiling Wave 21

## Added

- Added `notebooks/glcuda_t4_wave21_fused_swiglu.ipynb`, a self-contained
  reconstruction and gated Tesla T4 experiment for paired gate/up MMA,
  SwiGLU and direct Q8_0 output.
- Added an experimental, opt-in fused-SwiGLU PTX image inside the notebook's
  embedded patch, with retained fallbacks for unsupported capability, layout,
  type and shape combinations.
- Added explicit runtime dispatch logging, a driver occupancy query and direct
  exact-byte Q8/scale parity cases for full, ragged and production shapes.
- Added the compiler-gate result in
  `architecture/glcuda-research/wave21-fused-swiglu-production-result.md`.

## Validation

- The local reconstructed stack passes the Wave 21 example check and 64/64
  `glcuda` library tests.
- The final 48,604-byte embedded patch reproduces SHA-256
  `7f01ed0e154cd76312b27af2a436499ab5559fe07c1eaba713717126c8c84d5b`.
- Kaggle version 1 exposed and rejected an over-broad notebook structural
  predicate; version 2 used a scoped predicate with the byte-identical kernel
  patch.
- On Tesla T4, the retained sm_75 module assembles with zero stack and spills;
  `gl_gemm_mma_q8_bstage` uses 49 registers and 9,728 B static shared memory.
- The Wave 21 module fails `ptxas` at line 83 because `shl` cannot consume
  `%ctaid.x` directly. No device parity or timing claim is made.

## Decision

Wave 21 is a **compile-gate reject**. The compiler failure terminates the wave
before occupancy, parity or production A/B, so Wave 20's 10,242.4 tok/s remains
the latest valid production number. Correcting the special-register use is a
new candidate for a later wave, not a fix-forward inside this result.
