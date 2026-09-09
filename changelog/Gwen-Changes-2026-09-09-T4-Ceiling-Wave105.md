# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 105

- Added an isolated N16/M32 next-K32 register-prefetch candidate for the exact
  FFN-down production shape (`896 x 4864 x 244`) behind
  `GLCUDA_GEMM_N16_M32_PREFETCH`.
- Preserved the retained M32 launch geometry, shared-memory image, arithmetic,
  epilogue, and exact shape guard; the candidate changes only load issue time.
- Added a reproducible PTX generator and a counterbalanced, bit-exact direct
  gate with ptxas, spill, occupancy, and two-run speed checks.
- Passed 66/66 local `glcuda` library tests and the release example build.
- Archived Kaggle version 1, which correctly stopped at the hardware gate
  because auto-allocation supplied P100/sm_60 instead of T4/sm_75.
- Classified the run as infrastructure-only. No candidate or production
  performance conclusion was drawn.
