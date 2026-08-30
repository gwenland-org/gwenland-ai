# T4 Ceiling Wave 3 — 2-D token-grid prefill

## Problem

The Wave 2 T4 artifact reached 5,251.5 prefill tok/s, still 2.86x below the
15k target. The 8-m-tile MMA kernel processed prompts above 64 tokens through
a serial host launch loop, so token slabs could not occupy the GPU together.

## Change

- Added `%ctaid.y` token-slab rebasing to `gl_gemm_mma_q8` without changing
  its MMA/dequant body or barriers.
- Changed its launch geometry to `(ceil(out/64), ceil(ntok/64), 1)`.
- Added opt-in production dispatch with `GLCUDA_GRID2D=1`; r256 remains
  available as the Wave 2 comparison arm.
- Added GPU parity cases at 65/200/256 tokens and `256x896x200`.
- Corrected stage byte accounting so r256 uses its real 256-row logical slab.
- Added a direct-fetch Kaggle notebook with pinned model SHA, PTXAS spill and
  register gates, hardware correctness, two paired production sessions,
  telemetry, optional NCU, and an automatic retain/reject verdict.

## Local validation

- `cargo test -p glcuda --lib --locked`: **41 passed**.
- `cargo test -p glcuda --test parity --locked -- --test-threads=1`: **21
  passed**, with CUDA work explicitly skipped because this host has no CUDA
  driver/device; this is compilation evidence only.
- PTX is pure ASCII with LF endings and `git diff --check` is clean.
- Notebook JSON parses, all eight Python code cells pass `ast.parse`, the
  embedded patch SHA verifies, and reverse `git apply --check` succeeds.
- Fixed the notebook PTXAS resource parser to match complete function names;
  a regression fixture covers the `gl_gemm_mma_q8` / `_r256` prefix collision.
  Replaying the parser over the Wave 2 logs now reports base/candidate resources
  independently instead of assigning the r256 counters to both kernels.
- Local `ptxas` is unavailable. Assembly/resource and real parity are hard
  gates in the T4 notebook, not claimed locally.

## Retention status

**Pending T4 Wave 3 artifact.** Keep the candidate opt-in until production
glbench improves by at least 5% in both paired sessions with correctness and
resource gates green.
