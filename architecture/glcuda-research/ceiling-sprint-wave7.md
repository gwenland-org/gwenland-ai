# T4 Ceiling Sprint - Wave 7

Status: implementation complete; T4 production decision pending.

## Why Wave 6 was rejected

Wave 6 widened the token tile to 128 rows. It was correct, but production fell
from 6592.9 to 5684.8 prefill tok/s (-13.77%) and P95 latency rose 15.8%.
The larger tile increased normalized A+B traffic by 23.8% and reduced available
warps per CTA. It is not a candidate for another launch-geometry iteration.

## Single lever

Wave 7 changes only the INT8 scale contract for matmul weights and activations:

- one symmetric signed-INT8 scale per output row of `W`;
- one symmetric signed-INT8 scale per token row of `X`;
- one exact s32 accumulator per Tensor Core output lane across the complete K
  dimension;
- one `f32(acc) * w_scale * x_scale` epilogue after the K loop.

The retained Q8_0 path applies a scale/FMA after every K32 block. For the
Qwen2.5-0.5B production matrices, the Wave 7 contract reduces those scale folds
from about 2.73 billion to 74 million per prompt (36.8x). The row quantizer also
changes the nominal prompt work from about 1.38 million warp CTAs to about
23 thousand row CTAs (59x fewer). Tensor Core shape, token slab, shared-memory
pitch, barriers, and retained dynamic-shared attention do not change.

The longest production dot is safe in s32:
`4864 * 127 * 127 = 78,451,456`, only 3.65% of `INT32_MAX`.

## Runtime contract

`GLCUDA_W8PC=1` is an experimental, opt-in gate. The loader replaces each
matmul tensor with a contiguous signed-INT8 stream and an f32 row-scale stream;
it never keeps a duplicate dense image. Cache files use the distinct
`.w8pc.glcache` suffix. Decode uses a matching row-scaled dp4a GEMV, while
prefill on sm_75 uses `gl_gemm_mma_w8pc`.

## Decision gate

The direct-fetch Kaggle notebook compares retained Wave 4 against Wave 7 on the
pinned Qwen2.5-0.5B Q4_K_M model. Retain only when all of these hold:

1. both PTX modules assemble for sm_75 with zero spills;
2. hardware parity, forward, and graph-replay tests run without CUDA skips;
3. model validation stays green;
4. each of two order-reversed production pairs improves both P50 and mean by
   at least 5%;
5. P95 prefill latency and decode P50 regress by no more than 5%;
6. static resource evidence preserves at least 24 resident warps per T4 SM.

Microbenchmarks diagnose the mechanism but cannot retain the candidate.
