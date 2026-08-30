# How PTX Should Be Used in an ML Inference Stack

Date: 2026-08-25  
Target: NVIDIA T4 (`sm_75`), Qwen2.5-0.5B, quantized prefill

## Conclusion

PTX is the narrow, architecture-specific implementation layer of an ML system. It is useful when the kernel needs an instruction or data layout that a standard library cannot express efficiently, such as Turing INT8 `mma.sync` combined with Q8_0 block-scale dequantization. It is not the layer that should decide the whole GEMM schedule, model graph, or every shape with one fixed kernel.

The practical hierarchy is:

1. The model/runtime chooses operations, fusion boundaries, tensor formats, and batching.
2. A shape dispatcher chooses a tested kernel variant or library algorithm.
3. The GEMM kernel implements CTA, warp, instruction, and memory tiling.
4. PTX expresses the architecture-specific inner loop and memory operations.
5. `ptxas` or the driver converts PTX to native GPU instructions; SASS and hardware counters are the final evidence.

NVIDIA describes PTX as a low-level **virtual** ISA that is translated to a target GPU ISA. It explicitly supports hand-written performance kernels, but PTX is still input to an optimizing backend assembler. Therefore source-level instruction counting alone is not a reliable performance verdict.

## Default decision rule

Use the highest layer that can express the required operation without losing the required performance:

| Requirement | Preferred implementation |
|---|---|
| Standard dense GEMM with supported types/scales | cuBLASLt, with heuristic selection cached per shape |
| Custom layout, quantization, or fused epilogue | CUTLASS/CUDA kernel family |
| Exact Turing MMA fragment mapping or an operation the compiler cannot reliably express | Small hand-written PTX specialization |
| Model-wide launch and memory reduction | Runtime/graph fusion, not PTX |

For GwenLand, cuBLASLt is an important ceiling reference but is not automatically a drop-in replacement for Q8_0. Q8_0 has a floating scale for every 32-value block, while the current kernel performs scale-aware accumulation inside the K loop. A fair library comparison needs a compatible representation or a custom CUTLASS epilogue/mainloop; otherwise it benchmarks a different mathematical operation.

## What an efficient GEMM implementation must own

CUTLASS' reference design maps GEMM hierarchically across threadblocks, warps, and MMA instructions, with explicit reuse in shared memory and registers. It also changes tile shapes for different M/N/K regimes. A complete high-performance implementation therefore needs more than the correct `mma.sync` opcode:

- a 2D CTA decomposition over token rows and output columns;
- enough independent CTAs to occupy all 40 T4 SMs;
- shape-specific CTA/warp tiles rather than one universal tile;
- conflict-free shared-memory layouts and coalesced global transactions;
- software pipelining on Turing, because `cp.async` is unavailable on `sm_75`;
- efficient, fused epilogues;
- split-K or smaller output tiles when M/N expose too little parallel work;
- measured algorithm selection per important model shape.

The current hand-written kernel gets the instruction-level primitive right, but it is still a fixed schedule:

- one block covers 64 output rows;
- the token dimension is slabbed sequentially by the host;
- there is no production 2D token/output grid yet;
- the tile family is only the 64-token and 256-token variants;
- Q8_0 scaling adds FP32 conversion/multiply/FMA work at each 32-wide K block;
- each projection is launched independently, limiting activation reuse.

That is why polishing PTX instructions can improve the microkernel while leaving most of the T4 idle at the model level.

## The 15,000 prefill-token/s target

The target is not excluded by the advertised arithmetic roofline. NVIDIA specifies 130 INT8 TOPS for T4, equivalent to about 65 trillion integer multiply-accumulates per second when a multiply and add are counted as two operations. At approximately 0.5 billion linear-weight MACs per token:

```text
15,000 token/s * 0.5e9 MAC/token = 7.5e12 MAC/s
7.5 / 65 = 11.5% of advertised INT8 tensor-core MAC throughput
```

This is only a compute lower bound. It excludes Q8_0 scale work, activation quantization, attention, normalization, memory traffic, kernel launches, incomplete tiles, and serialization. It proves that 15k is mathematically plausible, not that the current execution graph can attain it.

The measured forced-Q8 prefill result is about 3.8k token/s, so reaching 15k requires about 3.95x end-to-end. The isolated research 2D-grid result was as high as 3.28x. Even if that full speedup applied unrealistically to the entire model, the projection would be only about 12.5k token/s:

```text
3,800 * 3.28 = 12,464 token/s
```

Thus the 2D grid is necessary evidence, but it cannot be the only lever. The next measurement must determine the post-force-Q8 share of GEMM time and apply Amdahl's law to measured stage timings.

## Recommended T4 kernel architecture

### 1. Keep PTX, but make it a kernel family

Retain the `mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32` PTX primitive. Surround it with variants selected by `(M, N, K, alignment, quant format)`:

- CTA-N 64 / 32 / possibly 16 output-row variants;
- CTA-M 64 / 128 / 256 token variants;
- 128- and 256-thread variants where register pressure permits;
- 2D grid variants for long prefill;
- an optional split-K variant only for shapes that still launch too few CTAs.

Variant selection should be benchmarked once on the actual T4 and cached, analogous to cuBLASLt's algorithm heuristic. A source-code guess should not become the permanent dispatch rule.

### 2. Fix grid-level utilization before deeper instruction tuning

For the observed large shape, the old launch exposed only 32 CTAs on a 40-SM GPU. CUTLASS explicitly identifies too few threadblocks as a failure mode and recommends additional M/N tiling or K partitioning. The production priority is therefore:

1. land and validate the 2D token/output grid;
2. reduce the output tile so representative shapes expose substantially more than 40 CTAs;
3. measure registers, active warps, eligible warps, tensor-pipe utilization, DRAM throughput, shared conflicts, and local-memory spills;
4. only then tune prefetch distance, instruction order, and additional buffering.

### 3. Optimize quantization at the operation boundary

The Q8_0 scale is semantically part of the matrix product, so it should be designed with the GEMM mainloop or epilogue, not treated as incidental instructions. Candidate experiments are:

- keep exact per-32 scales but overlap scale loads/conversions with MMA;
- load/reuse activation tiles across fused projections;
- fuse activation quantization into the producer operation;
- evaluate a prefill-only activation scale granularity only as a new numerical format with a full parity gate;
- investigate direct native K-quant GEMM so `GLCUDA_FORCE_Q8` is not a global policy.

Changing scale granularity is not a PTX-only optimization: it changes the numerical contract and requires accuracy validation.

### 4. Fuse at the ML graph level

PTX cannot recover launches and memory round trips created above it. High-value model-level candidates are:

- combine Q/K/V projections or at least share one staged activation tile;
- combine gate/up projections and reuse the same activation load;
- fuse SiLU/multiply with quantization for the down projection;
- fuse compatible residual, normalization, and quantization boundaries;
- use CUDA Graphs where the launch sequence is stable.

These must be measured separately from the microkernel A/B because graph fusion changes the amount of work and memory traffic.

## Measurement contract

Each kernel candidate must pass the following gates on a real T4:

1. `ptxas -arch=sm_75` assembly and resource report;
2. numerical parity across tail and boundary shapes;
3. interleaved A/B microbenchmarks with identical clocks and inputs;
4. production prefill P50/P90/P99 over repeated runs;
5. stage telemetry without folding telemetry overhead into the timing result;
6. Nsight Compute inspection of the emitted kernel, including Speed of Light, tensor utilization, occupancy, stalls, shared conflicts, global-load efficiency, and spills;
7. an end-to-end acceptance gate against 15,000 token/s, not just a faster isolated GEMM.

The Nsight roofline should be treated as a classification tool: it distinguishes compute, memory-feed, and latency/under-utilization limits. Advertised TOPS is not itself the attainable roofline of a scaled Q8_0 kernel.

## Proposed experiment order

1. Complete the current Wave 2 baseline/candidate notebook with forced-Q8 on both arms and collect post-force-Q8 stage telemetry.
2. Productionize the 2D token grid if correctness and end-to-end results pass.
3. Sweep CTA-N/thread-count variants for the dominant 0.5B shapes and cache the winner per shape class.
4. Add a cuBLASLt or CUTLASS-compatible ceiling oracle for mathematically comparable INT8 GEMMs.
5. Prototype fused gate+up activation reuse; then QKV reuse.
6. Decide between direct K-quant GEMM and selective dual-format weights using measured VRAM/decode tradeoffs.

Do not combine steps 2-6 into one patch: each changes a different ceiling and must preserve causal attribution.

## Primary sources

- NVIDIA, [PTX ISA — Introduction](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- NVIDIA, [CUDA Compiler Driver — GPU compilation and virtual/real architectures](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
- NVIDIA, [CUTLASS — Efficient GEMM in CUDA](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- NVIDIA, [cuBLAS/cuBLASLt documentation](https://docs.nvidia.com/cuda/cublas/)
- NVIDIA, [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)
- NVIDIA, [Nsight Compute Profiling Guide — Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts)
- NVIDIA, [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)
- NVIDIA, [T4 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/tesla-t4/)
