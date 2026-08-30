# T4 Ceiling Sprint — Wave 9

**Status:** ready for T4 confirmation  
**Baseline:** retained Wave 4 (`BASE + Wave 3 + Wave 4`)  
**Target:** 15,000+ prefill tok/s on Tesla T4

## Why this wave

Wave 8 preserved exact Q8_0 bytes and improved its isolated quantizer by about
51%, but production moved only 3.31%. It remains HOLD. Wave 9 therefore starts
again from retained Wave 4 and changes one different variable: CTA ordering in
the existing 64-row INT8 Tensor Core GEMM.

OpenAI Triton's matrix-multiplication tutorial explicitly groups program IDs so
tiles that reuse data are launched near one another, improving L2 hit rate. AMD
Composable Kernel likewise uses an explicit block-to-C-tile map instead of
treating tile order as incidental. Stream-K shows that GEMM work distribution
can materially affect utilization, although split-K/Stream-K is deliberately
outside this wave because it adds partial-result coordination and changes much
more than scheduling.

Primary references:

- <https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html>
- <https://github.com/triton-lang/triton/blob/main/python/tutorials/03-matrix-multiplication.py>
- <https://github.com/ROCm/composable_kernel/blob/develop/include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp>
- <https://arxiv.org/abs/2301.03598>

## Hypothesis

For the production 244-token prompt, grid64 has four token slabs. The retained
launch linearizes output tiles first, then advances to the next slab. Wave 9
swaps the two grid axes and maps them back inside PTX:

```text
Wave 4 grid: x=output tile, y=token slab
Wave 9 grid: x=token slab,  y=output tile
```

CUDA linear block IDs are x-major, so Wave 9 places the four CTAs that consume
one weight tile next to each other in the launch raster. This is a scheduling
hint, not a formal CTA execution-order guarantee, and the production A/B is the
only retention evidence.

The change is opt-in with `GLCUDA_L2_RASTER=1`. The existing PTX entry receives
one uniform flag and selects coordinates before the original body. It does not
change:

- `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` count or operands;
- Q8_0 scale/dequantization arithmetic;
- shared-memory declarations or addresses;
- `bar.sync` count or placement;
- output stores or numerical order within a CTA.

## Clean experiment

The Kaggle notebook constructs two trees from revision
`3bce8dd7b8aaa2765855ab927c611b54981f9241`:

1. baseline: archived Wave 3 plus retained Wave 4;
2. candidate: the same stack plus the Wave 9-only patch.

Rejected/HOLD Waves 5, 6, 7, and 8 are not applied. The embedded Wave 9 patch
SHA-256 is `d569f0967b6cb4c11479eaeb22e20ebfb6fcfa1421ab6c35c51a2160401a5df3`.

## Retention gate

Wave 9 may be retained only when all of these hold on a real T4:

- ptxas succeeds for both PTX modules with zero spills;
- the modified grid64 kernel remains in the at-least-24-resident-warp tier;
- r256 and the main PTX module stay structurally/resource identical;
- direct grid64 versus L2-raster output is bit-identical at 244 tokens;
- the complete parity suite executes on hardware rather than self-skipping;
- both production sessions improve prefill P50 and mean by at least 5%;
- P95 prefill latency and decode do not regress by more than 5%;
- both sessions retain exact greedy next-token parity against `glproc`.

The two-shape GEMM timing and telemetry stage shares are diagnostic only.

