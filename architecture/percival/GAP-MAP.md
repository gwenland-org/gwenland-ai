# GAP-MAP — Architectural Gaps Surfaced by Percival

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`

A "gap" is an architectural observation that has consequences for GwenLand —
either something llama.cpp does that GwenLand must consciously decide to
adopt/adapt/reject, or something llama.cpp lacks that GwenLand must build.

Gaps are added here **only after** an ARTX document has been completed with
supporting source evidence. Gaps without an ARTX reference are forbidden.

## Gap entries

| Gap ID | ARTX    | Category               | One-liner                                                                 | Priority | Status   |
| ------ | ------- | ---------------------- | ------------------------------------------------------------------------- | -------- | -------- |
| G01    | ARTX01  | BACKEND_DESIGN         | CPU backend is purely synchronous (no events, no async transfers)         | Medium   | Open     |
| G02    | ARTX01  | EXECUTION_GRAPH        | Per-node SPMD barrier; no per-node parallelism                            | High     | Open     |
| G03    | ARTX01  | EXECUTION_GRAPH        | Op fusion limited to one pattern; detected at execution time              | High     | Open     |
| G04    | ARTX01  | SIMD_STRATEGY          | type_traits_cpu is static const — no runtime per-dtype kernel swap        | Medium   | Open     |
| G05    | ARTX01  | CORRECTNESS_SHORTCUT   | Matmul output is ULP-non-deterministic for nth>1 (chunk stealing + SIMD reassociation) | Low | Open |
| G06    | ARTX01  | LAYOUT_SUBOPTIMAL      | Per-thread cpumask is bool[MAX_THREADS], not a bitmap                     | Low      | Open     |
| G07    | ARTX01  | CORRECTNESS_SHORTCUT   | GELU/QuickGELU precomputed as f16 LUT (11-bit precision)                  | Low      | Open     |
| G08    | ARTX01  | BACKEND_DESIGN         | Multi-binary ISA dispatch vs. function-pointer dispatch — design decision | Low      | Open     |
| G09    | ARTX01  | THREADING_MISMATCH     | NUMA falls back to one-chunk-per-thread (no locality-aware stealing)      | Medium   | Open     |
| G10    | ARTX02  | SIMD_STRATEGY          | No native 512-bit AVX-512 vecdot in arch/x86/quants.c (only 256-bit VNNI) | High     | Open     |
| G11    | ARTX03  | SIMD_STRATEGY          | No Zen-specific 256-bit batched GEMM (512-bit splits into 2 uops on Zen4) | High     | Open     |
| G12    | ARTX03  | BACKEND_DESIGN         | `zen4` build variant misnamed — requires AVX512_BF16, loads only on Zen5  | Medium   | Open     |
| G13    | ARTX04  | SIMD_STRATEGY          | No baseline-NEON batched GEMV/GEMM (baseline ARM runs scalar prefill)     | High     | Open     |
| G14    | ARTX05  | SIMD_STRATEGY          | SVE VL switch has `assert(false)` for non-{128,256,512} (correctness landmine) | High | Open |
| G15    | ARTX06  | QUANTIZATION           | IQ3/IQ2 from_float commented out — inference-only formats no round-trip  | Medium   | Open     |
| G16    | ARTX07  | EXECUTION_GRAPH        | Central-counter barrier estimated ~3-4µs at 16 threads; scales linearly   | Medium   | Open     |
| G17    | ARTX07  | MISSING_FEATURE        | `GGML_NUMA_STRATEGY_MIRROR` enum exists but switch has no case (dead)     | Medium   | Open     |
| G18    | ARTX08  | EXECUTION_GRAPH        | CUDA backend has ~12 fusion patterns but detected at execution time       | High     | Open     |
| G19    | ARTX08  | MISSING_FEATURE        | Cross-backend `event_wait` is `#if 0`'d out — opaque why disabled        | Medium   | Open     |
| G20    | ARTX10  | GPU_KERNEL             | MMQ has no K-iteration pipeline (4 __syncthreads per iter, no cp.async)   | High     | Open     |
| G21    | ARTX10  | GPU_KERNEL             | No Hopper wgmma/TMA in MMQ — uses Ampere mma.sync even on Hopper         | High     | Open     |
| G22    | ARTX11  | MISSING_FEATURE        | No FP8 KV cache path on CUDA                                              | High     | Open     |
| G23    | ARTX11  | GPU_KERNEL             | No Hopper-native FA (wgmma/TMA) — uses mma.sync on Hopper                 | High     | Open     |
| G24    | ARTX12  | GPU_KERNEL             | IQ1_M absent from MMQ dispatch but not excluded from should_use_mmq (latent ABORT) | High | Open |
| G25    | ARTX15  | BACKEND_DESIGN         | Metal `MTLCreateSystemDefaultDevice` ignores multi-GPU Mac Pro            | Medium   | Open     |
| G26    | ARTX16  | GPU_KERNEL             | `mul_mm` legacy path is barrier-bound (no double-buffering)               | Medium   | Open     |
| G27    | ARTX17  | MISSING_FEATURE        | No FP8 / K-quant / IQ-quant KV paths in Metal FA                          | High     | Open     |
| G28    | ARTX18  | BACKEND_DESIGN         | No `VkPipelineCache` with on-disk persistence — cold-start pipeline compile | High  | Open     |
| G29    | ARTX20  | SIMD_STRATEGY          | `mul_mm_cm2.comp` hardcodes `subgroup_size=32` for shmem sizing (silent correctness risk) | High | Open |
| G30    | ARTX20  | GPU_KERNEL             | Softmax/argmax/sum_rows lack subgroup ops (pure shmem)                    | Medium   | Open     |
| G31    | ARTX21  | LAYOUT_SUBOPTIMAL      | CPU `TENSOR_ALIGNMENT = 32` insufficient for AVX-512 (64) / AMX (128)     | High     | Open     |
| G32    | ARTX21  | EXECUTION_GRAPH        | `tensor_copy_async` falls back to sync when either side lacks async       | High     | Open     |
| G33    | ARTX22  | EXECUTION_GRAPH        | Sequential split execution — independent splits cannot run concurrently   | High     | Open     |
| G34    | ARTX22  | EXECUTION_GRAPH        | Per-split `graph_optimize` cannot fuse across split boundaries            | High     | Open     |
| G35    | ARTX23  | BACKEND_DESIGN         | `api_version` 1→2 history unclear — needs git archaeology                 | Low      | Open     |
| G36    | ARTX24  | MISSING_FEATURE        | Cross-backend FA precision flag (`prec`) is a single bit — no F8/F6/F4 enum | Medium | Open     |
| G37    | ARTX24  | MISSING_FEATURE        | MLA `V_is_K_view` detection is CUDA/Metal-only; absent on Vulkan          | Medium   | Open     |

## Priority legend

* **Critical** — blocks correctness or determinism in GwenLand.
* **High**    — blocks a stated GwenLand target backend or kernel class.
* **Medium**  — meaningful efficiency / maintainability impact.
* **Low**     — minor; monitor only.

## Category legend

Mirrors the Percival finding categories:

`CORRECTNESS_SHORTCUT`, `LAYOUT_SUBOPTIMAL`, `MEMORY_PATTERN`,
`THREADING_MISMATCH`, `SIMD_STRATEGY`, `GPU_KERNEL`, `EXECUTION_GRAPH`,
`QUANTIZATION`, `BACKEND_DESIGN`, `ADOPT`, `MISSING_FEATURE`, `OTHER`.
