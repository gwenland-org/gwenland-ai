# Deep research — implementing the next FFN/GEMM optimization correctly

**Audience:** GwenLand/glcuda kernel maintainers  
**Date:** 2026-09-06  
**Scope:** Tesla T4 (SM75), Q8_0 prefill, Qwen2.5-0.5B, current Wave58 evidence. The decision is how to pursue a real path from ~11.5k to 15k tok/s without weakening correctness gates.

## Executive answer

The correct implementation is a dedicated SM75 INT8 Tensor-Core FFN GEMM with a hierarchical CTA→warp→MMA tile, two-stage software pipeline, and a fused-but-bit-controlled epilogue. Keep the existing Q8 packing and `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` arithmetic contract, but prototype the kernel behind a separate dispatch flag. Tune only one dimension at a time: CTA/warp tile, stage count, then epilogue fusion. Every candidate must pass PTXAS resource limits, driver occupancy, direct f32 output parity, full CUDA parity, and the 50/50 token oracle before production timing is considered.

The T4 has 64K 32-bit registers/SM, 64KB shared memory/SM, 32 resident warps, and 16 resident blocks/SM. These limits make a wider tile useful only if register and shared-memory growth preserves enough active blocks. The current Wave58 profile puts 66.0% of time in FFN gate/up, down, and elementwise stages, so attention work cannot bridge the 30% gap to 15k by itself.

## Implementation recipe

1. Preserve the current Q8_0 B-stage byte layout and dequantization order. Use the explicit SM75 `m8n8k16` signed-int8 MMA primitive; CUTLASS's Turing example confirms this opcode shape and s8×s8→s32 accumulation.
2. Use a CTA tile sized for the actual FFN shapes, then map it to warp tiles and `m8n8k16` instruction tiles. Measure `ffn_gate_up` (M=244,N=9728,K=896) and `ffn_down` (M=244,N=896,K=4864) independently; do not optimize only the aggregate.
3. Implement a two-stage shared-memory pipeline: while stage 0 feeds MMA, stage 1 loads the next A/B tile. CUTLASS documents this exact overlap pattern for Turing and warns that accumulator registers are usually at least half of a thread's register budget.
4. Keep the epilogue explicit initially: convert s32 accumulators, apply scale and f32 accumulation in the existing order, and store with the existing guards. Only after the GEMM body wins should SwiGLU/linear glue fusion be reintroduced as a separate experiment.
5. Record PTXAS registers, stack, spills, static/dynamic shared memory, and `cuOccupancyMaxActiveBlocksPerMultiprocessor` output for each shape. Reject any spill, unexpected barrier, or residency drop below the retained tier.
6. Verify numerical behavior in layers: direct GEMM f32 bit comparison; 33/33 CUDA parity; teacher-forced logits over 50 steps; then the immutable 50/50 token oracle. A top-token match alone is not an exact-parity claim.

## What not to do

Do not resurrect r256 (known numerical bug), K128 or independent-MMA chains (previous production rejects), or change quantization while claiming Q8-vs-Q8. Do not use Turing `cp.async`/Hopper TMA assumptions. Do not relax the oracle after a failed run; Wave58's 48–49/50 teacher-forced agreement is evidence of drift, not acceptance.

## Evidence and gap matrix

| Claim | Evidence | Confidence / gap |
|---|---|---|
| T4 resource limits are 64K registers, 64KB shared, 32 warps, 16 blocks | NVIDIA Turing Tuning Guide | High; exact per-device launch still needs recording |
| Hierarchical tiles and two-stage software pipelining are the right GEMM structure | NVIDIA CUTLASS Efficient GEMM guide and Turing example | High; tile dimensions remain workload-specific |
| SM75 INT8 MMA is `m8n8k16`, s8×s8→s32 | NVIDIA CUTLASS Turing example; PTX ISA | High |
| Occupancy must be measured with driver API, not inferred | NVIDIA CUDA Driver API occupancy docs | High |
| FFN is next lever in this repo | Wave58 `events.json`: FFN 66.0% of 20.26ms | High for current prompt; generalization to other contexts unmeasured |
| 15k tok/s is reachable | No primary evidence yet | Unknown; requires new kernel and T4 A/B runs |

## Source ledger

- **Turing Tuning Guide**, NVIDIA, current CUDA 13.3 documentation. https://docs.nvidia.com/cuda/turing-tuning-guide/ — T4 occupancy/resource and integer arithmetic guidance.
- **Efficient GEMM in CUDA**, NVIDIA CUTLASS documentation. https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html — hierarchical tiling, software pipelining, accumulator/occupancy trade-offs, epilogue separation.
- **Turing TensorOp GEMM example**, NVIDIA CUTLASS. https://github.com/NVIDIA/cutlass/blob/main/examples/08_turing_tensorop_gemm/turing_tensorop_gemm.cu — SM75 INT8 types, `GemmShape<8,8,16>`, two-stage pipeline example.
- **Parallel Thread Execution ISA 6.5**, NVIDIA. https://docs.nvidia.com/cuda/archive/10.2/pdf/ptx_isa_6.5.pdf — `mma.sync.aligned.m8n8k16` signed-int8 form.
- **CUDA Driver Occupancy API**, NVIDIA. https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__OCCUPANCY.html — runtime occupancy functions.
- **Wave58 logits diagnostic**, repository evidence: `benchmarks/glcuda-t4-wave58-evidence.zip` and `architecture/glcuda-research/wave58-logits-diagnostic.md`.

Research stopped after the primary NVIDIA sources converged on the same design constraints and the remaining unknown (a new FFN tile's measured speed/correctness) can only be resolved by implementing and running the next controlled T4 experiment.
