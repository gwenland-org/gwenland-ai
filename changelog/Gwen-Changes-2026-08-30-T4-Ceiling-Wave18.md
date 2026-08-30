# Gwen Changes - 2026-08-30 - T4 Ceiling Wave 18

## Changed

- Updated `notebooks/glcuda_t4_attn_split.ipynb` to decision-grade timing with
  10 warmups and 100 measured launches per arm.
- Made the stop-0 validity check a paired, counterbalanced
  `probe -> real -> real -> probe` A/B inside every repeat.
- Reused dead `%r47` and `%p7` in the qk4 diagnostic probe instead of adding
  probe-only named registers across the kernel.
- Added `ptxas` register, spill, shared-memory, resident-block, and occupancy
  reporting for the production kernels and both probe images.
- Added the reproduced Wave 18 decision in
  `architecture/glcuda-research/wave18-attention-split-result.md`.

## Validation

- Kaggle versions 8 and 9 ran the identical notebook SHA-256
  `20972feb0af2494b3c56d0fb85bc513f12087bed690b16cddb9613f279cf887` on a
  Tesla T4.
- Both runs passed 32/32 CUDA parity tests and 61/61 glcuda library tests.
- qk4's paired stop-0 gate differed from the real kernel by -0.6% and +0.1%; the
  rows control differed by -0.1% and -0.2%.
- The real and probe qk4 images each use 63 registers, zero spills, and the same
  eight-blocks-per-SM tier. The prior occupancy explanation is falsified.
- Notebook JSON, every Python code cell, and the fully rewritten embedded Rust
  source parse locally.

## Decision

Wave 18 confirms the shipped qk4 split at an average **65.7% QK, 3.8% softmax,
and 30.5% AV**. The old unpaired qk4 splits remain rejected; their contradictory
validity results came from an order-biased gate. QK is about 18.1% of the whole
measured prefill and is the attention-side optimization target.
