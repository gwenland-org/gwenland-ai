# Gwen Changes — 2026-08-25 — T4 Ceiling Wave 9

## Added

- Added the opt-in `GLCUDA_L2_RASTER` production dispatch for the retained
  sm_75 grid64 INT8 MMA kernel.
- Added a uniform PTX coordinate swizzle that groups token-slab CTAs by output
  weight tile without changing MMA arithmetic, barriers, or shared memory.
- Added exact hardware bit-parity coverage between the original and grouped
  launch rasters at the 244-token production geometry.
- Added a `gate_up`/`down` diagnostic A/B marker to the CUDA benchmark.
- Added `notebooks/glcuda_t4_ceiling_wave9.ipynb`, a direct-fetch Kaggle
  notebook comparing clean retained Wave 4 against Wave 9.
- Added the Wave 9 research and retention contract in
  `architecture/glcuda-research/ceiling-sprint-wave9.md`.

## Validation

- Current development tree: 46 `glcuda` library tests pass.
- Current development tree: 26 parity test entries pass locally; CUDA tests
  self-skip because the workstation has no CUDA device.
- Clean Wave 4 + Wave 9 tree: 42 library tests and 22 parity entries pass
  locally with the same hardware limitation.
- Notebook JSON parses, every code cell compiles, the embedded patch round-trips
  byte-for-byte, and the clean Wave 4 tree accepts it with `git apply --check`.
- Embedded Wave 9 patch SHA-256:
  `d569f0967b6cb4c11479eaeb22e20ebfb6fcfa1421ab6c35c51a2160401a5df3`.

## Decision

Wave 9 is not retained yet. The Kaggle T4 production gate remains authoritative.

