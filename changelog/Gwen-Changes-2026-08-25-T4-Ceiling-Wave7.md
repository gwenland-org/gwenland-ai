# Gwen Changes - 2026-08-25 - T4 Ceiling Wave 7

Implemented an opt-in W8A8 row-scaled path for Turing after the Wave 6 r128
kernel lost 13.77% in production.

- Added `HostWeight::W8PcSoa` / `GpuWeight::W8PcSoa` and a cold-path f32 to
  signed-INT8 per-output-row repack.
- Added distinct `.w8pc.glcache` serialization so scale contracts cannot mix.
- Added a 256-thread, one-CTA-per-token row activation quantizer.
- Added a row-scaled dp4a GEMV for decode and fallback.
- Added `gl_gemm_mma_w8pc` for sm_75. It keeps s32 fragments through full K
  and performs one scale epilogue while preserving the retained grid64 launch,
  48-byte shared-memory pitch, B prefetch, barriers, and MMA instruction shape.
- Routed Q/K/V shared quantization, attention output, FFN gate/up, FFN down,
  and LM head through the consuming weight's explicit scale contract.
- Added cache, traffic-accounting, structural, repack, GEMV, and real-shape MMA
  parity coverage.

The path is disabled unless `GLCUDA_W8PC=1`. T4 ptxas, hardware correctness,
model validation, and production retention remain decision gates in
`notebooks/glcuda_t4_ceiling_wave7.ipynb`.
