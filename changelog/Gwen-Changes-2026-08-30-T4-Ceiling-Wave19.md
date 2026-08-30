# Gwen Changes - 2026-08-30 - T4 Ceiling Wave 19

## Added

- Added `notebooks/glcuda_t4_wave19_mma_qk.ipynb`, an isolated sm_75
  compensated-f16 MMA QK feasibility harness.
- Added host numeric screening for three- and four-product f16 decomposition.
- Added same-contract qk4 and MMA4 causal-score diagnostics with host
  softmax/AV grading against the f32 attention oracle.
- Added decision-grade paired timing: 10 warmups, 100 measured launches and
  five forward/reverse repeats.
- Added `ptxas -v` register, shared-memory and spill reporting plus CUDA
  resident-block queries.
- Added the reproduced decision in
  `architecture/glcuda-research/wave19-mma-qk-feasibility-result.md`.

## Validation

- Kaggle versions 2 and 3 completed on a Tesla T4; version 1 was excluded
  because the harness had not bootstrapped `cargo` into the Kaggle PATH.
- Both decision runs passed 61/61 `glcuda` library tests.
- The four-product host model passed the retained `1e-5` attention tolerance at
  `2.466e-6`; the three-product control failed at `7.460e-5`.
- Device MMA4 attention matched the f32 oracle at `2.682e-7` max absolute error
  in both runs.
- `ptxas` reports 43 registers, 4 KiB shared memory and zero spills for MMA4;
  both diagnostics permit 16 active blocks per SM.
- Paired-median QK speedup reproduced at `2.3237x` and `2.3100x`; all ten
  paired repeats exceeded the registered `1.5x` gate.
- Notebook JSON, its Python cell, the embedded patch digest and both Rust
  diagnostics were validated locally.

## Decision

Wave 19 is a **feasibility pass**, not a production retention. Four-product
compensated-f16 MMA is accurate and fast enough to justify a fused attention
implementation. Three-product compensation and plain f16 remain rejected. No
product kernel or dispatch default changes in this wave.
