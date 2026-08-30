# glcuda T4 Ceiling Sprint - Wave 6

**Date:** 2026-08-25  
**Target:** NVIDIA T4 (`sm_75`), Qwen2.5-0.5B-Instruct Q4_K_M, 15,000 prefill tok/s  
**Status:** candidate implemented; T4 production gate pending  
**Notebook:** [`notebooks/glcuda_t4_ceiling_wave6.ipynb`](../../notebooks/glcuda_t4_ceiling_wave6.ipynb)

## Why Wave 5 stopped

Wave 5 raised the small-kernel footprint from 56 registers / 3,328 bytes to
63 registers / 6,656 bytes, but delivered only **+0.49%** production P50 and
failed both throughput gates. The A/xscale software pipeline is rejected.
Wave 6 reconstructs retained Wave 4 and changes one different variable.

The latest paired run measured 6,861.3 tok/s, or 35.56 ms for the 244-token
prompt. Reaching 15,000 tok/s requires 16.27 ms and another 2.19x end-to-end
gain. GEMM remains 57.9% of prompt time, so this experiment stays on the
Tensor Core path.

## Research basis

Turing allows at most 32 resident warps and 16 resident blocks per SM, with
64K 32-bit registers and 64 KiB shared memory. That makes a 128-thread CTA a
useful middle point: it can enlarge the output tile while retaining enough
independent CTAs and warps when register use is controlled. NVIDIA's CUTLASS
GEMM model similarly treats a larger warp/threadblock tile as a reuse and ILP
tradeoff, not an unconditional win. Its Turing INT8 example uses the same
`m8n8k16` Tensor Core generation and much larger hierarchical tiles than this
project's original 64-row path.

Primary references:

- [NVIDIA Turing Tuning Guide](https://docs.nvidia.com/cuda/turing-tuning-guide/index.html)
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Turing TensorOp INT8 example](https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## One causal lever: 128-row B reuse

Wave 6 adds `gl_gemm_mma_q8_r128`, a dedicated middle kernel:

```text
output tile                 128 rows x 32 columns
CTA                         128 threads / 4 warps
MMA instructions/K-block   32
A/xscale shared memory      6,144 + 512 = 6,656 bytes
grid                        ceil(out/32) x ceil(ntok/128)
```

One loaded B fragment now serves 128 output rows rather than 64. At the
244-token production prompt, each output slab therefore launches two token
tiles instead of four. Unlike the r256 kernel, the r128 grid still exposes 56
CTAs for the representative `out=896` layer, enough to cover the T4's 40 SMs;
r256 exposes only 14. The dispatch is opt-in with `GLCUDA_R128=1`, takes
precedence over the retained 64-row grid only when it reduces the number of
token tiles, and keeps grid64 on the `ntok <= 64` tie.

The kernel is derived from the hardware-correct r256 schedule, trimmed to 16
M tiles. It preserves the `m8n8k16` operands, B-fragment prefetch, Q8_0 scale
multiply/FMA, 48-byte bank-conflict-free A pitch, 64-bit A staging, vector
stores, guards, and barrier placement. It does not use `cp.async`, which is
not available on `sm_75`. The retained grid64 and r256 kernel bodies and the
attention PTX must compare byte-for-byte equal between arms.

## Falsifiable resource hypothesis

Removing half of r256's accumulator set should put r128 near 78 registers.
The notebook accepts no more than 85 registers, zero spills, and exactly
6,656 bytes of shared memory. At 85 registers and 128 threads, the 64K Turing
register file permits six CTAs, or 24 resident warps per SM; shared memory is
not the limiter. This is only a static projection. PTXAS and optional NCU
evidence must confirm it, and production timing remains authoritative.

## Retention gate

The direct-fetch notebook reconstructs Wave 4 and Wave 6 from SHA-verified
patches, fetches the pinned GGUF directly, and then requires:

1. exact structural preservation of grid64, r256, and main attention PTX;
2. PTXAS success, zero spills, r128 registers <= 85, exactly 6,656 bytes of
   shared memory, and no resource change in retained kernels;
3. CUDA lib/parity/forward/graph correctness with no silent hardware skip;
4. two reversed-order production sessions, each improving P50 and mean by at
   least 5%; and
5. no more than 5% regression in P95 prompt latency or decode P50.

The r128 diagnostic is explanatory only. Ship the candidate only if the
production gate says `RETAIN`.
