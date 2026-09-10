# Gwen Changes - 2026-09-09 - T4 Ceiling Wave 91

- Added an isolated SM75 N16 register-prefetch kernel for the pinned FFN
  gate/up projection.
- Added opt-in loading and exact shape-gated dispatch through
  `GLCUDA_GEMM_N16_PREFETCH=1`; production defaults remain unchanged.
- Preserved the retained Q8 layout, MMA count and order, f32 epilogue, shared
  image, launch geometry, and synchronization count.
- Added a structural PTX/dispatch contract test; the glcuda library suite now
  passes 65/65 tests.
- Deferred all resource, device parity, direct speed, and production claims to
  the hard T4 gate.
