# Gwen Changes - 2026-09-27 - T4 Ceiling Wave 136

- Added an opt-in fused QKV decode grid for Q8 SoA weights behind
  `GLCUDA_DECODE_QKV_FUSED=1` and the retained Wave 132 decode-Q8 contract.
- Preserved one warp per output row and the retained GEMV accumulation order;
  unsupported formats and shapes keep the three-launch fallback.
- Passed 76 host tests, 37 serial CUDA parity tests, four CUDA Graph replay
  tests, a zero-spill compiler gate, and 50/50 oracle checks in four sessions.
- Improved Tesla T4 production decode P50 from 216.61 to 223.63 tok/s (+3.24%)
  while retaining 15,744.51 prefill tok/s.
- Archived the counterbalanced sessions and self-contained reproduction
  notebook under the Wave 136 benchmark and research paths.
