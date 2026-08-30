# Gwen Changes - 2026-08-25 - T4 Ceiling Wave 10

## Added

- Added `notebooks/glcuda_t4_ceiling_wave10.ipynb`, a four-arm factorial Kaggle
  sprint comparing retained Wave 4, rowCTA, L2 raster, and their combination.
- Added a clean combined patch relative to Wave 4 while keeping the independent
  Wave 8 and Wave 9 arms for attribution.
- Added a four-repeat Latin-square schedule so every arm occupies every run
  position once.
- Added pre/post session T4 P-state, temperature, power, clock, utilization, and
  memory snapshots.
- Added per-arm exact oracle, spill, occupancy, latency, and decode gates plus
  main-effect and interaction reporting.
- Added the Wave 10 experiment contract in
  `architecture/glcuda-research/ceiling-sprint-wave10.md`.

## Validation

- Current development tree baseline: 46 `glcuda` library tests pass.
- Current development tree baseline: 26 parity test entries pass locally.
- Clean combined Wave 4 + Wave 8 + Wave 9 stack: `cargo check`, 46 library
  tests, and 26 parity entries pass locally.
- Notebook JSON, all eight code cells, Git object syntax, embedded patch hashes,
  and whitespace checks pass.
- Updated the notebook to `wave10-rowcta-raster-factorial-v2`: its Wave 8
  structural gate now checks the real `self.q8_rowcta` dispatch in
  `kernels/mod.rs`, instead of looking for a nonexistent runner helper.
- Combined patch SHA-256:
  `37d7df45dd48eb02cf10d8e3a94e8f210d679eaea4e4134d8eb63a8bb5e45ce3`.
- Notebook SHA-256:
  `541a60641f16775b3e63b7b7b89159716cac3fdf21aba6f4f90be7718c4c0480`.

## Decision

Wave 10 is not retained yet. The four-arm Kaggle T4 production gate remains
authoritative.
