# Gwen Changes - 2026-09-01 - T4 Ceiling Wave 26

## Added

- Added `architecture/glcuda-research/wave26-external-gemm-research.md`, an
  external, primary-source review of the remaining Tesla T4 INT8 GEMM design
  space after Waves 1-25.
- Added a post-Wave-20 Amdahl model showing that a broad FFN kernel needs about
  1.12x direct speedup to have a credible path through the +5% production
  retention gate.
- Added a ranked candidate ledger covering wider per-warp N tiles, integer
  WMMA shapes, SM75 matrix loads, CTA N32, split-K, and Stream-K.

## Research decision

- Froze Wave 27 as an M64/K32 warp-N16 experiment using the retained explicit
  `mma.sync.aligned.m8n8k16` primitive.
- The candidate will reuse one activation fragment across two adjacent N8
  output fragments while preserving the K32-major B image, Q8_0 scale order,
  f32 accumulation order, 9,728-byte shared image, and production grid.
- Set hard pre-production gates of at most 72 registers, zero stack/spills,
  exact output, and at least 1.12x direct speedup on both production FFN shapes.
- Kept warp-N32 plus `ldmatrix`, wider WMMA spelling, and Stream-K as separate
  research candidates. None may be folded into Wave 27 as fix-forward.

## Scope

Wave 26 is documentation-only. It changes no PTX, Rust dispatch, notebook,
benchmark output, or production default. No generated benchmark or staging
directory was created.
