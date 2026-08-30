# Gwen Changes - 2026-08-31 - T4 Ceiling Wave 20

## Added

- Added `notebooks/glcuda_t4_wave20_production.ipynb`, a self-contained Tesla
  T4 reconstruction, parity, resource and production A/B harness.
- Added an opt-in fused sm_75 attention candidate that computes four-product
  compensated-f16 MMA QK, softmax and AV without a global score matrix.
- Added direct device parity coverage for the 244-token production shape and
  ragged, chunked and short-tail shapes with padded Q strides.
- Added separate `ptxas -v` audits for the base and sm_75 modules plus CUDA
  active-block occupancy queries.
- Added the reproduced production decision in
  `architecture/glcuda-research/wave20-mma-attention-production-result.md`.

## Validation

- Kaggle versions 2 and 3 ran identical notebook bytes on Tesla T4; version 1
  was screening-only because its baseline `ptxas` parser inspected the sm_75
  module instead of the base qk4 module.
- Both decision runs passed 62/62 library tests and 33/33 device parity tests.
- Every one of 16 production sessions passed exact 50/50 token-oracle
  validation; all eight paired repeats improved prefill throughput.
- qk4 uses 63 registers, 1,012 B dynamic shared memory, zero spills and permits
  eight active blocks/SM. Fused MMA4 uses 48 registers, 15,616 B dynamic shared
  memory, zero spills and permits three active blocks/SM.
- Production throughput reproduced at +12.84% and +14.50%. Across both runs,
  the midpoint moves from 9,010.2 to 10,242.4 tok/s (+13.67%), with paired
  deltas spanning +11.01% to +14.80%.
- Median prefill latency moves from 27.083 to 23.826 ms (-12.02%); tail and
  decode gates pass.

## Decision

Wave 20 is a **production retention pass**. The fused compensated-MMA attention
patch is licensed for integration and a default flip, with qk4 retained as the
capacity fallback. The current research branch records the qualified patch in
the notebook because the measured historical performance stack is not present
as product source at this branch head.
