# ARTX10 — CUDA GEMM and MMQ (Matrix-Matrix Quantized) Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (graph execution, kernel selection)

---

## 1. Executive Summary

The CUDA GEMM path is the **prefill and batched-decode hot path**. For every
generated token beyond the GEMV regime (ARTX09), each linear layer issues an
`M×K × K×N = M×N` matrix–matrix product with `N > 8`. Two kernel families cover
this regime:

1. **MMQ** (`mmq.cu` 371 + `mmq.cuh` 1571 + `mmq-vec-dot.cuh` 1251 +
   `mmq-load-tiles.cuh` 1679 + six `mmq-config-*.cuh` files) — quantized GEMM
   that operates directly on Q4_0/Q4_K/etc. weights *without* dequantizing
   them to F16 first. The activation is pre-quantized to a custom
   `block_q8_1_mmq` layout (128 int8s + 4×`half2` of scale/sum per 128-element
   block). A single template `mul_mat_q<type, J, fallback>` covers every quant
   format (22) and every supported GPU family (Pascal, Ampere, Blackwell, CDNA,
   RDNA2, RDNA4). On Blackwell, a dedicated native-FP4 path (`block_fp4_mmq` +
   `mma_block_scaled_fp4`) uses the new
   `mma.sync.aligned.kind::mxf4.block_scale` PTX and bypasses Q8_1 dequant
   entirely.
2. **MMF** (`mmf.cu` 191 + `mmf.cuh` 909 + `mma.cuh` 1456) — F32/F16/BF16 GEMM
   via hand-written `mma.sync` PTX and `ldmatrix`. Covers the boundary `ne11 ∈
   {1..16}` on Tensor-Core hardware. cuBLAS handles larger batches via
   `ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:1406-1660`).

Routing is performed inside `ggml_cuda_mul_mat` (`ggml-cuda.cu:1812-1852`)
with strict precedence MMVF → MMF → MMVQ → MMQ → cuBLAS (ARTX09 §5.6). The
MMQ-vs-cuBLAS threshold is governed by `ggml_cuda_should_use_mmq`
(`mmq.cu:256-371`), with per-arch, per-type, per-batch decision branches.

For GwenLand, the decisions worth **ADOPT**ing are: (a) the per-(type, J, arch)
config-table architecture; (b) the dual `dp4a`/`mma` vec_dot codepath design;
(c) **stream-K** decomposition with fixup kernel; (d) the Blackwell native
FP4 block-scaled MMA path; (e) `J`-template enumeration; (f) MoE MUL_MAT_ID
support via `mm_ids_helper` + `expert_bounds` + dedup-bcast
quantize-once-scatter. The decisions worth **REJECT**ing are: (a) the absence
of `cp.async` pipelining in MMQ; (b) the unused Hopper `wgmma` / TMA
infrastructure (only PDL is used). The decisions worth **MONITOR**ing are: (a)
the hand-maintained per-arch `should_use_mmq` threshold tables; (b) the TF32
cuBLAS default for F32 GEMM.

---

## 2. Purpose

Provide the CUDA kernels that service `MUL_MAT` and `MUL_MAT_ID` when the
activation matrix has a "large" column count (`ne11 > MMVQ_MAX_BATCH_SIZE =
8`) — i.e., the prefill and batched-decode case. Specifically:

* `M×K × K×N` quantized GEMM for every supported quant format (22 in total)
  without ever materializing a dequantized F16 weight matrix.
* `M×K × K×N` F32/F16/BF16 GEMM via Tensor Cores for the `ne11 ∈ {1..16}`
  small-batch boundary that MMVF/MMVQ do not cover.
* cuBLAS fallback for `ne11 > 16` F32/F16/BF16 GEMM with compute-type
  selection and output-dtype auto-selection.
* Auto-select the best kernel based on shape, dtype, compute capability,
  expert count, and per-arch hand-tuned thresholds.

It is **not** responsible for: GEMV (`ne11 ≤ 8`, ARTX09), attention matmuls
(`fattn*.cu`), fusion decisions (ARTX08), activation quantization policy
(delegated to `quantize.cu`), or graph-level scheduling.

---

## 3. Source Files

| File                                          | Lines  | Role                                                                              |
| --------------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/mmq.cu`                   | 371    | MMQ dispatch shell: `ggml_cuda_mul_mat_q_switch_type` (23-arm switch), `ggml_cuda_mul_mat_q` entry (asserts, padding-clear, activation pre-quantization routing, MoE helper launch), `ggml_cuda_should_use_mmq` predicate. |
| `ggml/src/ggml-cuda/mmq.cuh`                  | 1571   | MMQ launcher templates: `ggml_cuda_mmq_config` struct, `CASE` macro, `mmq_get_nbytes_shared`, `mul_mat_q_process_tile`, `mul_mat_q` kernel, `mul_mat_q_stream_k_fixup` kernel, `launch_mul_mat_q`, `mul_mat_q_switch_J` (16-way), `mul_mat_q_case`. |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh`          | 1251   | Per-quant tile-vecdot implementations: `_dp4a` (CUDA-core) and `_mma` (Tensor-Core) variants for every quant format. Includes Blackwell `ggml_cuda_mmq_vec_dot_fp4_fp4_mma`. |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh`       | 1679   | Per-quant tile loaders: `ggml_cuda_mmq_load_tiles_<type>` for every quant format. Two layouts per loader: MMA (sram_stride) and dp4a (`MMQ_DP4A_TXS_*`). Includes Blackwell `ggml_cuda_mmq_load_tiles_mxfp4_fp4` and `..._nvfp4_nvfp4`. |
| `ggml/src/ggml-cuda/mmq-config-ampere.cuh`    | 366    | Ampere/Volta/Ada config table: 22 types × 16 J values × 2 fallback = ~700 CASE lines. All entries: `nthreads=256, occupancy=1, I=128, stream_k=true`. |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | 37     | Blackwell-specific entries for MXFP4 and NVFP4 only (32 lines), then defers to Ampere config for all other types. Uses `MMQ_ITER_K_FP4=512` and `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4`. |
| `ggml/src/ggml-cuda/mmq-config-pascal.cuh`    | 261    | Pascal config: `nthreads=256, occupancy=2, I=64, stream_k=false`. Smaller I (64 vs 128) and no stream-K. |
| `ggml/src/ggml-cuda/mmq-config-cdna.cuh`      | 177    | CDNA (MI100/MI210/MI300) config: `nthreads=512, occupancy=1, I=128, stream_k=true`. J range is smaller (16/32/48/64 only). |
| `ggml/src/ggml-cuda/mmq-config-rdna2.cuh`     | 261    | RDNA2 config: `nthreads=256, occupancy=2, I=128, stream_k=false`. J up to 64. |
| `ggml/src/ggml-cuda/mmq-config-rdna4.cuh`     | 282    | RDNA4 config: `nthreads=256, occupancy=2, I=128, stream_k=false`. J up to 128. |
| `ggml/src/ggml-cuda/mmf.cu`                   | 191    | MMF dispatch shell `ggml_cuda_mul_mat_f`, `mmf_get_rows_per_block`, `ggml_cuda_should_use_mmf` predicate. |
| `ggml/src/ggml-cuda/mmf.cuh`                  | 909    | MMF Tensor-Core kernel `mul_mat_f`, `mul_mat_f_ids`, `mul_mat_f_switch_*` dispatchers, template-instantiation macros. |
| `ggml/src/ggml-cuda/mma.cuh`                  | 1456   | `mma.sync` PTX wrappers: `tile<I,J,T,dl>` template with `get_i`/`get_j`, `load_ldmatrix`, `load_generic`, `mma` overloads for `(s8,s8,s32)`, `(f16,f16,f32)`, `(bf16,bf16,f32)`, `(f32,f32,f32)`, `(f16,f16,f16)`, plus Blackwell `mma_block_scaled_fp4`. |
| `ggml/src/ggml-cuda/vecdotq.cuh`              | 1322   | Per-block `vec_dot_*_q8_1` device functions, `VDR_*_MMQ`/`VDR_*_MMVQ` constants, `vec_dot_q4_0_q8_1_impl` etc., shared by MMVQ (ARTX09) and MMQ tile-vecdot. |
| `ggml/src/ggml-cuda/common.cuh`               | 1661   | `ggml_cuda_dp4a`, `warp_reduce_sum`, CC capability constants (Pascal/Volta/Turing/Ampere/Hopper/Blackwell/CDNA1-4/RDNA1-4), `cp_async_available`, `blackwell_mma_available`, `MATRIX_ROW_PADDING=512`, `ggml_cuda_type_traits<ggml_type>`. (Audited in ARTX08; only GEMM-relevant helpers summarised here.) |
| `ggml/src/ggml-cuda/cp-async.cuh`             | 58     | `cp_async_cg_16<preload>` (16-byte async copy with L2::256B/128B/64B hints), `cp_async_wait_all`. Used by flash-attn (ARTX11) — **not** by MMQ/MMF. |
| `ggml/src/ggml-cuda/ggml-cuda.cu`             | 5425   | `ggml_cuda_mul_mat` router (1812), `ggml_cuda_mul_mat_cublas_impl` (1406), `ggml_cuda_mul_mat_cublas` (1619). Audited in ARTX08; only the matmul-routing and cuBLAS sections summarised here. |

> Note: the audit prompt's reference to "MMQ using Tensor Cores" is partially
> correct — MMQ has both a `dp4a` (CUDA-core) and an `mma` (Tensor-Core)
> codepath, selected at compile time via `use_mma_data_layout()`. The `mma`
> path uses `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` (and `m16n8k32` on
> Ampere+) for int8 Tensor Cores, not the FP16/BF16/FP4 Tensor Cores that
> MMF uses.

---

## 4. Architecture Overview

```
                ┌────────────────────────────────────────────────┐
                │  ggml-cuda.cu : ggml_cuda_mul_mat              │
                │  (cuda matmul router — ARTX08 §5.4)            │
                └────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
   should_use_mmq        should_use_mmf           else (ne11 > 16
   (all quants)          (F32/F16/BF16)           or non-MMA CC)
        │                     │                          │
        ▼                     ▼                          ▼
   mmq.cu                mmf.cu + mmf.cuh           ggml-cuda.cu
   ggml_cuda_mul_mat_q   ggml_cuda_mul_mat_f        ggml_cuda_mul_mat_cublas
   ├─ padding-clear      (mma.sync + ldmatrix)      (cublasSgemm / GemmEx /
   ├─ quantize_mmq_*                                GemmStridedBatchedEx /
   │  _cuda (F32→Q8_1_mmq                           GemmBatchedEx)
   │   or F32→block_fp4_mmq)
   ├─ mm_ids_helper (if MUL_MAT_ID)
   └─ mul_mat_q_case<type>
        │
        ▼
   mul_mat_q_switch_J<type, fallback>
        (16-way switch on J_best: 8..128 step 8)
        │
        ▼
   launch_mul_mat_q<type, J, fallback>
   ├─ ggml_cuda_mmq_get_config → (nthreads, I, K_vram, stream_k, sram_layout)
   ├─ mmq_get_nbytes_shared → CUDA_SET_SHARED_MEMORY_LIMIT
   ├─ if !stream_k: dim3(nty, ntx, ntzw)
   ├─ if  stream_k: dim3(nsm, 1, 1) + fixup kernel
   └─ mul_mat_q<type, J, fallback><<<...>>>
        │
        ▼
   mul_mat_q_process_tile<type, J, fallback, fixup>
   ├─ load_tiles(x, tile_x, ...)  // ggml_cuda_mmq_load_tiles_<type>
   ├─ memcpy y → tile_y (all threads, sync)
   ├─ vec_dot(tile_x, tile_y, sum, 0)        // ggml_cuda_mmq_vec_dot_*_<dp4a|mma>
   ├─ memcpy y' → tile_y (second half of MMQ_ITER_K)
   ├─ vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K)
   └─ write_back(sum, ids_dst, dst, y_scale, ...)
```

Key design points:

* **Template-on-`type` and template-on-`J`.** The compiler produces a
  separate kernel binary for every `(type, J ∈ {8,16,24,...,128}, fallback)
  combination. The runtime `switch` over `J_best` (`mmq.cuh:1470`) selects
  the right instantiation. This is the same pattern as ARTX09-R1 but on the
  *output-column-tile* axis instead of the *output-column-count* axis.
* **Config-driven tiling.** `ggml_cuda_mmq_config` (`mmq.cuh:164-203`) is
  a `constexpr` struct carrying `(nthreads, occupancy, I, J, sram_layout,
  K_vram, stream_k, fallback)`. Six per-arch config files (Pascal, Ampere,
  Blackwell, CDNA, RDNA2, RDNA4) populate the table via the `CASE` macro.
  The host and device both see the same table (`__host__ __device__`
  qualifier on `ggml_cuda_mmq_get_config_*`), so the kernel can read its
  tiling constants at compile time.
* **Dual vec_dot codepaths.** Each quant format has two tile-vecdot
  implementations: `*_dp4a` (uses `ggml_cuda_dp4a` on CUDA cores, no
  Tensor Cores) and `*_mma` (uses `mma.sync` + `ldmatrix` on Tensor Cores).
  The `use_mma_data_layout()` predicate (`mmq.cuh:188-201`) selects the
  codepath at compile time based on `TURING_MMA_AVAILABLE` /
  `AMD_MFMA_AVAILABLE` / `AMD_WMMA_AVAILABLE`. Pascal and RDNA2 use dp4a;
  everything Turing+ or MFMA/WMMA uses mma.
* **MMQ is asynchronous-stream only — no cp.async.** Despite Ampere+
  having `cp.async`, MMQ does synchronous global→shared loads followed by
  `__syncthreads()` (`mmq.cuh:887, 891, 903, 907`). The kernel has zero
  K-iteration pipeline depth. cp.async is implemented in `cp-async.cuh` but
  only used by flash-attention.
* **Blackwell FP4 native path.** When `blackwell_mma_available(cc)` and
  `src0->type ∈ {MXFP4, NVFP4}`, MMQ uses a separate `block_fp4_mmq`
  layout (`mmq.cuh:51-54`) and `mma_block_scaled_fp4<type>` PTX
  (`mma.cuh:1126-1154`). The activation is quantized to `block_fp4_mmq`
  (not Q8_1) via `quantize_mmq_fp4_cuda`. This is the only path that
  bypasses the Q8_1 dequantization+dp4a scheme entirely.
* **cuBLAS as the bulk GEMM workhorse.** For F32/F16/BF16 weights with
  `ne11 > 16` (or when MMQ/MMF predicates return false), `ggml_cuda_mul_mat`
  falls through to `ggml_cuda_mul_mat_cublas`. cuBLAS handles compute-type
  selection (F32 / F16 / BF16, with TF32 tensor-op math enabled globally
  in `common.cuh:1494`), output-dtype auto-selection (F32 vs F16/BF16 with
  explicit dequantize), and four dispatch variants (Sgemm, GemmEx,
  GemmStridedBatchedEx, GemmBatchedEx) based on batch shape and contiguity.

---

## 5. Execution Flow

### 5.1 MMQ entry and dispatch

`ggml_cuda_mul_mat_q` (`mmq.cu:82`) asserts `src1->type == F32`, `dst->type ==
F32`, and innermost-dim contiguity for all three tensors. As in MMVQ (ARTX09
§5.1), if `src0` is a temporary compute buffer with `size_alloc > size_data`,
it issues `cudaMemsetAsync` to zero the tail (`mmq.cu:107-114`). It computes
`ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)` (512). For Blackwell FP4,
it allocates a separate `src1_scale` buffer of `ne13*ne12*ne11` floats
(NVFP4 only). It then calls either `quantize_mmq_q8_1_cuda` (default) or
`quantize_mmq_fp4_cuda` (Blackwell FP4) to convert F32 src1 → the MMQ
activation layout. Finally it calls `ggml_cuda_mul_mat_q_switch_type`
(`mmq.cu:8`), a 23-arm `switch` over quant type.

### 5.2 Inside `mul_mat_q` (the GEMM kernel)

`mul_mat_q<type, J, fallback>` (`mmq.cuh:920-1205`). Skip-block: if the
config returns `GGML_TYPE_COUNT`, the kernel calls `NO_DEVICE_CODE` and
returns (`mmq.cuh:932-935`), letting the compiler emit a no-op kernel for
unsupported combinations.

Block layout: `dim3 block_dims(warp_size, nwarps, 1)` where `nwarps =
nthreads / warp_size`. For Ampere (256/32) and CDNA (512/64), `nwarps = 8`;
for Pascal (256/32) `nwarps = 8`. So `nwarps = 8` is universal; only the
warp_size and nthreads differ. Grid: `dim3(nty, ntx, ntzw)` where `nty =
ceil(nrows_x / I)`, `ntx = ceil(ncols_max / J)`, `ntzw = nchannels_y *
nsamples_y`.

Per-block shared memory (`mmq.cuh:857-859`): `ids_dst_shared[J]` +
`tile_y[J * MMQ_TILE_Y_K]` (padded) + `tile_x[I * sram_stride]`.

The K-dim loop lives inside `mul_mat_q_process_tile` (`mmq.cuh:842-915`).
For each `kb0` in `[kb0_start, kb0_stop)` stepping by `blocks_per_iter =
ITER_K / qk`:

1. `load_tiles(x, tile_x, ...)` — every thread reads one quant block from
   global `x`, unpacks to int8, and writes to `tile_x` in shared memory.
2. Copy `J * MMQ_TILE_Y_K` ints from global `y` to `tile_y` in shared
   memory (one int per thread; no unpacking needed because `y` is already in
   `block_q8_1_mmq` layout).
3. `__syncthreads()`.
4. `vec_dot(tile_x, tile_y, sum, 0)` — every thread accumulates into its
   `sum[...]` registers. The MMA path uses `load_ldmatrix` to fetch tile
   fragments from shared memory into Tensor-Core registers, then issues
   `mma.sync` PTX.
5. `__syncthreads()`.
6. Copy the *second* half of `MMQ_ITER_K` worth of `y` data into `tile_y`.
   (Two passes per `kb0` because `MMQ_ITER_K = 256` but `MMQ_TILE_NE_K = 32`.)
7. `__syncthreads()` → `vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K)` →
   `__syncthreads()`.

After the K loop, `write_back(sum, ids_dst, dst, y_scale, ...)` writes the
`J*I` output elements to `dst` (or to `tmp_fixup` if `fixup == true`, i.e.,
this is the last partial tile of a stream-K decomposition).

### 5.3 Stream-K decomposition

When `config.stream_k == true` (Ampere/Blackwell/CDNA configs), the launch
heuristic in `launch_mul_mat_q` (`mmq.cuh:1395-1441`) chooses between:

* **Tiled-K** with `gridDim = ntiles_dst`: used when
  `tiles_efficiency_percent >= 90` on NVIDIA (i.e., the tile grid already
  covers the SM count evenly). No fixup kernel needed.
* **Stream-K** with `gridDim = nsm` (number of SMs): used otherwise. Each
  block is assigned a contiguous range `[kbc, kbc_stop)` of K-block
  iterations across the *entire* (i,j,tile) space, not a fixed (i,j) tile.
  The last block of a tile writes to `tmp_fixup[blockIdx.x * J*I]` instead
  of `dst`, then a second kernel `mul_mat_q_stream_k_fixup<type, J,
  fallback>` (`mmq.cuh:1207-1343`) iterates over previous blocks, sums
  partial sums from the fixup buffer, and atomic-adds the final result to
  `dst`.

This is the standard stream-K scheme of DeepSeek's
[arXiv:2301.03598](https://arxiv.org/abs/2301.03598). The fixup kernel uses
`dst[idx] += sum`, which is safe because each `dst[idx]` is written by
exactly one fixup invocation.

### 5.4 MMF entry, dispatch, and kernel

`ggml_cuda_mul_mat_f` (`mmf.cu:13`) asserts contiguity, then calls
`mul_mat_f_switch_rows_per_block<T>` (`mmf.cuh:836`) which selects
`MMF_ROWS_PER_BLOCK = 32` (NVIDIA/RDNA) or `MMF_ROWS_PER_BLOCK_CDNA = 64`
(CDNA). Then `mul_mat_f_switch_cols_per_block<T, rows_per_block>`
(`mmf.cuh:733`) selects `cols_per_block ∈ {1..16}` based on
`ncols_dst_total` (or 16 if MoE with `ncols_dst > 16`). Then
`mul_mat_f_cuda<T, rows_per_block, cols_per_block>` (`mmf.cuh:619`)
auto-tunes `nwarps_best` (1..8) by minimizing `niter = ceil(ncols_x /
(nwarps * warp_size * 2))`. The `nwarps_best`-way switch (`mmf.cuh:668-728`)
launches the final template instance.

The kernel `mul_mat_f<T, rows_per_block, cols_per_block, nwarps, has_ids>`
(`mmf.cuh:48-294`) uses `tile<16, 8, T>` (Turing+), `tile<32, 4, T>`
(Volta), `tile<16, 16, T, DATA_LAYOUT_I_MAJOR>` (AMD WMMA/MFMA). Each block
computes a `rows_per_block × cols_per_block` output tile. The K-dim loop
tiles across `col ∈ [0, ncols)` in strides of `warp_size`. Each iteration:
load `tile_A` from `x` via shared-memory `tile_xy` then `load_ldmatrix`;
load `tile_B` from `y` (cast to T via `ggml_cuda_cast` for F16/BF16); issue
`mma(C, A, B)`. After the K loop, the per-thread C fragments are written
to `buf_iw` shared memory, transposed, and summed across warps for the
final `rows_per_block` outputs per column.

### 5.5 cuBLAS path

`ggml_cuda_mul_mat_cublas` (`ggml-cuda.cu:1619-1660`) selects `compute_type
∈ {F32, F16, BF16}` based on `src0->type` (with F16 preferred for quantized
src0 if `fast_fp16_hardware_available`), the `op_params[0] == GGML_PREC_F32`
user override, and the `GGML_CUDA_CUBLAS_COMPUTE_TYPE` env var. It then
dispatches to `ggml_cuda_mul_mat_cublas_impl<compute_type>` (`:1406-1617`),
which: (1) converts `src0` and `src1` to `compute_type` if needed (via
`traits::convert` / `traits::convert_nc`); (2) picks `cu_data_type`
(output dtype): F32 if `prefer_f32_output` (true for F16 compute on
Volta/RDNA4/CDNA, true for BF16 compute on non-RDNA3 non-CDNA), else
F16/BF16; (3) allocates a `dst_temp` buffer if output is not F32; the
cuBLAS call writes F16/BF16 there, then `to_fp32_cuda` converts back to F32;
(4) calls one of four cuBLAS variants based on shape: `cublasSgemm` (single
F32), `cublasGemmEx` (single non-F32), `cublasGemmStridedBatchedEx`
(batched strided), `cublasGemmBatchedEx` (batched general pointer array,
uses `k_compute_batched_ptrs` kernel to populate the pointer arrays).

The `CUBLAS_GEMM_DEFAULT_TENSOR_OP` algo flag is used everywhere, allowing
cuBLAS to use Tensor Cores. cuBLAS handle is initialized with
`CUBLAS_TF32_TENSOR_OP_MATH` (`common.cuh:1494`) — TF32 is on by default
for F32 GEMM.

---

## 6. Data Layout

### 6.1 Weight tensor (src0)

Required: `nb00 == ggml_type_size(src0->type)` (contiguous in innermost
dim). MMF additionally requires `src0_ne[0] % (warp_size * (4/ts)) == 0`
and `src0_nb[i] % (2*ts) == 0` for `i ≥ 1` (`mmf.cu:140-153`) because
the kernel casts `src0` to `half2*` / `nv_bfloat162*` / `float2*`. MMQ
has no such requirement on `nb[i]` — it indexes per-block via
`stride_row_x = ne00 / ggml_blck_size(type)`.

For MMQ, `nrows_x % 128 == 0` triggers `fallback = false` (`mmq.cuh:1528`),
which lets the load_tiles skip the `if (fallback) i = min(i, i_max)` bounds
check. If `nrows_x % 128 != 0`, `fallback = true` adds the bounds check
to every load.

### 6.2 Activation tensor (src1)

`nb10 == ggml_type_size(F32) == 4`. MMQ pre-quantizes src1 to one of two
MMQ-specific layouts:

* **`block_q8_1_mmq`** (`mmq.cuh:27-46`): 128 int8 values + a 16-byte
  scale/sum union (4×`half2`). The 128 values are a "transposed" view of
  four 32-element `block_q8_1` blocks, where the 4 blocks are interleaved
  so they can be copied to shared memory as one contiguous 144-byte unit.
  The 16-byte union is one of `d4[4]` (4 f16 scales, one per 32 values),
  `ds4[4]` (4 `(scale, sum)` pairs), or `d2s6[8]` (a Q2_K-specific layout)
  depending on the weight quant format (`mmq_get_q8_1_ds_layout`).
* **`block_fp4_mmq`** (`mmq.cuh:51-54`): 4×`uint32_t` scales + 128 int8
  nibble-packed FP4 values. Used only on Blackwell when `src0->type ∈
  {MXFP4, NVFP4}`. The activation is quantized to FP4 (matching the
  weight format) instead of Q8_1, so the Tensor Core can do
  FP4×FP4→F32 directly via `mma_block_scaled_fp4`.

Padding: `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)` (512). The
extra elements are zero-filled by the quantize kernel, so `vec_dot` calls
that read past the data end pick up zeros (multiplied by the activation
scale, contributing zero to the sum).

### 6.3 Output tensor (dst)

`nb0 == 4` (F32 contiguous in innermost). For `MUL_MAT`, `stride_col_dst
= dst->nb[1] / 4`. For `MUL_MAT_ID`, `stride_col_dst = dst->nb[2] / 4`
(expert axis = `ne1`, token axis = `ne2`); the layout swap is done in
`ggml_cuda_mul_mat_q` (`mmq.cu:244-252`).

### 6.4 MoE routing: `ids_dst` and `expert_bounds`

For `MUL_MAT_ID` (`ids != nullptr`), the `mm_ids_helper` kernel
(`mmid.cu`, audited in ARTX09 §5.7) sorts and compacts the expert routing.
The output is two arrays:

* `expert_bounds[ne02 + 1]`: range of compacted indices per expert. Block
  `zt` reads `col_low = expert_bounds[zt]` and
  `col_high = expert_bounds[zt + 1]`; it skips entirely if
  `jt * J >= col_high - col_low`.
* `ids_dst[ne_get_rows]`: maps compacted column index → original
  `(token, channel)` for output writes. Block writes
  `dst[ids_dst[col_low + jt*J + j] * stride_col_dst + i]`.

When `dedup_bcast = (ne11 == 1 && n_expert_used > 1)` (gate/up
activations broadcast across experts), `mm_ids_helper` writes the *inverse
map* `ids_src1` (token slot → compacted row). The activation quantization
then uses `quantize_scatter_mmq_*_cuda` which quantizes each token once
and scatters to all its expert slots, avoiding `n_expert_used`× duplicate
quantization work.

### 6.5 Per-quant block layout

Block sizes (`qk`), `qr` (elements per int), and `qi` (32-bit ints per
block) are in `ggml-common.h` and surfaced via `ggml_cuda_type_traits<type>`
(`common.cuh:964-1125`). The `vdr` (vec-dot ratio) is from `vecdotq.cuh:VDR_*_MMQ`
and equals the number of `qi`-chunks one thread processes per `vec_dot` call.
MMQ `vdr` values are typically 2× larger than MMVQ `vdr` values (e.g.,
`VDR_Q4_0_Q8_1_MMQ = 4` vs `VDR_Q4_0_Q8_1_MMVQ = 2`,
`vecdotq.cuh:109-113`) because the MMA path uses 16×8×16 Tensor Cores that
consume 16 K-elements per warp per instruction.

---

## 7. Memory Layout

### 7.1 Per-op scratch (`ctx.pool()`)

MMQ allocates one or two scratch buffers per call:

* `src1_q8_1`: `ne13*ne12 * ne11*ne10_padded * y_block_size/y_values_per_block
  + J_max * sizeof(block_q8_1_mmq)` bytes (`mmq.cu:133-134`). For a 4096-dim
  FFN with batch 64, this is `1 * 1 * 64 * 4096 * 144 / 128 = 294912` bytes
  (288 KiB) plus up to `128 * 144 = 18432` bytes of `J_max` padding.
* `src1_scale` (NVFP4 only): `ne13*ne12*ne11 * sizeof(float)` bytes.
* For MoE: `ids_src1`, `ids_dst`, `expert_bounds` arrays (small, `O(ne02
  + ne_get_rows)` ints).
* For stream-K: `tmp_fixup` of `block_nums_stream_k.x * J*I * sizeof(float)`
  bytes — only if `ntiles_dst % block_nums_stream_k.x != 0`.

All allocations use `ggml_cuda_pool_alloc<T>` (RAII, returns to pool at
scope exit). Total per-call scratch for a 4096-dim FFN with batch 64 is
~300 KiB.

### 7.2 MMQ shared memory

`mmq_get_nbytes_shared` (`mmq.cuh:1354-1359`):
```
nbs_ids = J * sizeof(int)                                              // up to 128*4 = 512 B
nbs_x   = I * sram_stride * 4        (MMA)                             // 128 * 76 * 4 = 38912 B (Q8_0, Ampere)
          OR (txs.qs + txs.dm + txs.sc) * 4  (dp4a)                    // varies, e.g., 128*65 + 128 + 0 = 8448 B
nbs_y   = J * sizeof(block_q8_1_mmq)                                  // up to 128*144 = 18432 B
nbs_y_padded = GGML_PAD(nbs_y, nthreads * sizeof(int))                // 256*4 = 1024 B alignment
total = nbs_ids + nbs_x + nbs_y_padded
```

For Ampere Q8_0 with `J=128, I=128`: `512 + 38912 + 18432 = 57856` bytes.
`CUDA_SET_SHARED_MEMORY_LIMIT` is called before launch
(`mmq.cuh:1375-1376`) to ensure the kernel can use up to this much. The
maximum supported shared memory per block on Ampere is 164 KiB (configurable
via `cudaFuncSetAttribute`); on Blackwell it is 228 KiB. The config tables
are sized to stay within `smpbo` (shared memory per block opt-in), checked
in `mul_mat_q_switch_J` (`mmq.cuh:1458`).

### 7.3 MMF shared memory

`extern __shared__ char data_mmv[]` of size `max(nbytes_shared_iter,
nbytes_shared_combine) + nbytes_slotmap` (`mmf.cuh:657-662`):

* `nbytes_shared_iter = nwarps_best * tile_A::I * (warp_size + padding) *
  4` — staging buffer for `tile_xy` used by `load_ldmatrix`.
* `nbytes_shared_combine = GGML_PAD(cols_per_block, tile_B::I) *
  (nwarps_best * rows_per_block + padding) * 4` — output reduction
  buffer.
* `nbytes_slotmap = GGML_PAD(cols_per_block, 16) * sizeof(int)` — only
  for `has_ids`.

For a typical Ampere F16 case with `nwarps=4, rows_per_block=32,
cols_per_block=8`: `4 * 16 * 36 * 4 = 9216` B for iter, `8 * (128 + 4) *
4 = 4224` B for combine → max = 9216 B.

### 7.4 cuBLAS workspace

cuBLAS manages its own workspace internally. The `ggml_cuda_mul_mat_cublas_impl`
allocates `src0_alloc`, `src1_alloc`, and `dst_temp` from `ctx.pool()` only
when dtype conversion is needed; otherwise it passes the original tensor
pointers directly to cuBLAS.

### 7.5 L2 persistence

Not used by MMQ or MMF. No `cudaStreamSetAttribute` with
`cudaStreamAttributeAccessPolicyWindow` calls anywhere in the MMQ/MMF
paths. The weight matrix (typically resident for the entire inference
session) would benefit from L2 persistence across decode steps, but this
is left to the hardware L2 replacement policy.

---

## 8. Parallelism Strategy

### 8.1 MMQ: tile-blocked GEMM with optional stream-K

Each thread block owns an `I × J` output tile. Within a block, `nwarps`
warps cooperatively process the K dimension. The K-loop steps by
`blocks_per_iter = ITER_K / qk` (e.g., `256 / 32 = 8` blocks for Q4_0).
Each iteration loads `I` weight-block-rows × `blocks_per_iter` blocks
into `tile_x`, plus `J` × `blocks_per_iter` Q8_1 blocks into `tile_y`,
then issues two `vec_dot` calls (one per `MMQ_TILE_NE_K = 32` half of
the `MMQ_ITER_K = 256` K-elements).

The `vec_dot` call dispatches to either:

* `*_dp4a`: each warp covers `J / nwarps` output columns × `I / warp_size`
  output rows. Each thread reads its slice of `tile_x` and `tile_y`, calls
  `ggml_cuda_dp4a` `vdr` times, multiplies by the scale, and adds to its
  `sum[...]` register.
* `*_mma`: each warp owns `rows_per_warp = (J >= 48 && J % 16 == 0) ? 32
  : 16` output rows (NVIDIA) or 16 (AMD). The warp uses `load_ldmatrix`
  to fetch a `tile_A = 16×8` fragment from `tile_x` and a `tile_B = 8×8`
  fragment from `tile_y`, then `mma.sync.aligned.m16n8k16.s32.s8.s8.s32`
  (or `m16n8k32` on Ampere+) to accumulate into a `tile_C = 16×8` int32
  fragment. The int32 fragment is then multiplied by the per-block
  scales (`dA * dB`) and accumulated into the float `sum[...]`.

For stream-K (`config.stream_k == true`), the grid is `nsm` blocks
instead of `ntiles_dst` blocks. Each block is assigned a *contiguous
range of K-iterations across the whole (i, j, tile) space*, not a fixed
`(i, j)` tile. The mapping uses `fast_div_modulo` and `fastmodulo`
helpers for integer division by `blocks_per_ne00`, `ntx`,
`nchannels_y`, `nsamples_y` — all compiled as `uint3` with
`init_fastdiv_values` for the fast-division trick (Wang et al.
"Engineering fast integer division"). The last partial tile of each
block writes to `tmp_fixup` instead of `dst`; a second
`mul_mat_q_stream_k_fixup` kernel reads back the partial sums,
atomic-adds them to `dst`, and writes the final result.

### 8.2 MMF: tile-blocked GEMM with auto-tuned nwarps

Each block owns a `rows_per_block × cols_per_block` output tile (32×N
or 64×N). The K-dim loop tiles across `col ∈ [0, ncols)` in strides of
`warp_size`. `nwarps_best` is auto-tuned (1..8) to minimize
`niter = ceil(ncols_x / (nwarps * warp_size * 2))`. No stream-K; pure
tiled GEMM.

### 8.3 cuBLAS: opaque

cuBLAS picks its own tiling internally. The only control the caller has
is `CUBLAS_GEMM_DEFAULT_TENSOR_OP` (allow Tensor Cores),
`CUBLAS_TF32_TENSOR_OP_MATH` (enable TF32 for F32), and the
compute/data type combination. The `cublasSetStream` call binds the
cuBLAS handle to the backend's stream so async execution works.

### 8.4 Multi-GPU

Not directly in MMQ/MMF. Multi-GPU tensor parallelism is handled at a
higher level by the comm-context AllReduce path (ARTX08 §5.6). The MMQ
kernel itself is single-GPU. The `ggml_cuda_op_mul_mat` template
(`ggml-cuda.cu:1329`, audited in ARTX08) is the legacy multi-GPU
split-buffer entry; it is no longer called from `ggml_cuda_mul_mat`
(ARTX08-F04).

---

## 9. GPU Strategy

### 9.1 Per-arch config tables

The single most important GPU-strategy decision in MMQ is the
config-table architecture. `ggml_cuda_mmq_config` (`mmq.cuh:164-203`)
carries:

| Field | Meaning | Range across archs |
| ----- | ------- | ------------------ |
| `nthreads` | Threads per block | 256 (NVIDIA/RDNA), 512 (CDNA) |
| `occupancy` | Target blocks per SM | 1 (Ampere/Blackwell/CDNA), 2 (Pascal/RDNA2/RDNA4) |
| `I` | Output-row tile width | 64 (Pascal), 128 (everything else) |
| `J` | Output-column tile width | 8..128 (step 8) — picked at runtime |
| `sram_layout` | Shared-memory layout enum | Q8_0, Q8_1, Q2_K, Q3_K, Q6_K, FP4, NVFP4 |
| `K_vram` | VRAM tile length in K dim | 256 (MMQ_ITER_K) or 512 (MMQ_ITER_K_FP4 for Blackwell FP4) |
| `stream_k` | Use stream-K decomposition | true (Ampere/Blackwell/CDNA), false (Pascal/RDNA2/RDNA4) |

The config is selected via `ggml_cuda_mmq_get_config(type, J, fallback,
cc)` (`mmq.cuh:225-242`) which dispatches to one of six per-arch
functions based on `cc`. Each per-arch function is a sequence of `CASE`
macros that match `(type, J, fallback)` and return a
`ggml_cuda_mmq_config`. If no entry matches, the function returns a
sentinel `ggml_cuda_mmq_config(GGML_TYPE_COUNT, ...)` which the kernel
checks for and calls `NO_DEVICE_CODE` (`mmq.cuh:932-935`).

The same `ggml_cuda_mmq_get_config` is also defined as a `constexpr
__device__` function (`mmq.cuh:244-263`) so the kernel body can read
its config at compile time. The host and device versions must agree —
this is enforced by the host launching the template instance compiled
for the matching arch.

### 9.2 Two vec_dot codepaths: dp4a and mma

Each quant format has two tile-vecdot implementations in
`mmq-vec-dot.cuh`:

* `ggml_cuda_mmq_vec_dot_*_dp4a`: uses `vec_dot_*_q8_1_impl` from
  `vecdotq.cuh` (which calls `ggml_cuda_dp4a`), accumulates into a
  per-thread `float sum[...]`. Each thread independently computes one
  `(i, j)` output element by reading its slice of `tile_x` and `tile_y`
  from shared memory.
* `ggml_cuda_mmq_vec_dot_*_mma`: uses `mma.sync` PTX via the `mma`
  wrappers in `mma.cuh`. Each warp collectively computes a `16×8` (or
  `16×16` on AMD) output tile by issuing `mma.sync.aligned.m16n8k16` (or
  `m16n8k32` on Ampere+). The int32 accumulator is multiplied by
  per-block scales and added to the float `sum[...]`.

The `use_mma_data_layout()` predicate (`mmq.cuh:188-201`) selects the
codepath based on `TURING_MMA_AVAILABLE` (NVIDIA Turing+) /
`AMD_MFMA_AVAILABLE` (CDNA) / `AMD_WMMA_AVAILABLE` (RDNA3+). Pascal and
RDNA2 use dp4a; everything else uses mma. The dp4a path uses a different
shared-memory layout (`MMQ_DP4A_TXS_*` macros in `mmq.cuh:362-371`) —
the mma path uses a single `sram_stride`-padded layout compatible with
`load_ldmatrix`.

### 9.3 Tensor Core instruction variants

`mma.cuh` exposes these `mma.sync` PTX wrappers, each used by MMQ/MMF:

| PTX instruction | Used by | Purpose |
| --------------- | ------- | ------- |
| `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` | MMQ mma (Turing+) | int8 Tensor Core, 16×8 output |
| `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` | MMQ mma (Ampere+) | int8 Tensor Core, 2× K width |
| `mma.sync.aligned.m8n8k16.s32.s8.s8.s32` | MMQ mma (Turing fallback) | 2-4 calls to emulate m16n8k16/32 |
| `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` | MMF F16 GEMM | FP16 Tensor Core with F32 acc |
| `mma.sync.aligned.m16n8k8.f32.f16.f16.f32` | MMF F16 GEMM (Turing) | Half-width FP16 Tensor Core |
| `mma.sync.aligned.m16n8k16.f32.bf16.bf16.f32` | MMF BF16 GEMM (Ampere+) | BF16 Tensor Core |
| `mma.sync.aligned.m16n8k16.f32.f16.f16.f32` (overload) | MMF F16 acc | FP16 Tensor Core with F16 acc |
| `mma.sync.aligned.m16n8k8.row.row.f16.f16.f16.f16` | MMF F16 acc (Volta) | Volta FP16 Tensor Core |
| `mma.sync.aligned.m16n8k8.f32.tf32.tf32.f32` | MMF F32 GEMM (CDNA3) | TF32 Tensor Core (via MFMA) |
| `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.f32.e2m1.e2m1.f32.ue8m0` | MMQ MXFP4 (Blackwell) | Native block-scaled MXFP4 |
| `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.f32.e2m1.e2m1.f32.ue4m3` | MMQ NVFP4 (Blackwell) | Native block-scaled NVFP4 |

No Hopper `wgmma.mma_async` instructions are used. No Blackwell
`tc_gen5.mma` instructions are used. The MMQ/MMF kernels are pure
`mma.sync` (synchronous, warp-scoped).

### 9.4 Blackwell FP4 native path

When `blackwell_mma_available(cc)` (CC ≥ 1200, < 1300) and `src0->type
∈ {MXFP4, NVFP4}`, MMQ takes a separate dispatch path
(`mmq.cuh:663-680`):

1. Activation is quantized to `block_fp4_mmq` (FP4 format matching the
   weight) via `quantize_mmq_fp4_cuda` (not `quantize_mmq_q8_1_cuda`).
2. Tile loader is `ggml_cuda_mmq_load_tiles_mxfp4_fp4` or
   `..._nvfp4_nvfp4` (`mmq-load-tiles.cuh:1542, 1640`). These bypass the
   int8 unpacking — they copy the raw FP4 nibbles directly into shared
   memory.
3. Tile vecdot is `ggml_cuda_mmq_vec_dot_fp4_fp4_mma`
   (`mmq-vec-dot.cuh:1186-1250`). It uses
   `mma_block_scaled_fp4<type>(C, A, B, scaleA, scaleB)` which issues
   the `kind::mxf4.block_scale.scale_vec::2X.m16n8k64` (MXFP4) or
   `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (NVFP4) PTX.
4. The block scale registers are loaded per-quad (2 threads per quad
   supply the scale register) per the
   [PTX warp-level block-scaling spec](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling).
5. NVFP4 path also supports a separate `y_scale` per-output-column
   factor, applied in `write_back` (`mmq.cuh:439-444,
   492-497`).

This is the single most important recent addition to MMQ — it gives
Blackwell ~2× throughput on FP4 weights vs the Q8_1 dequant + int8 mma
path used on Ampere.

### 9.5 PDL and launch bounds

MMQ does **not** call `ggml_cuda_pdl_sync` / `ggml_cuda_pdl_lc` in the
kernel body (contrast MMVQ in ARTX09 §9.4). The `__launch_bounds__`
annotation uses the per-config occupancy:
`__launch_bounds__(ggml_cuda_mmq_get_nthreads(type, J, fallback),
ggml_cuda_mmq_get_occupancy(type, J, fallback))` (`mmq.cuh:921`). For
Ampere this is `__launch_bounds__(256, 1)` (1 block per SM); for Pascal
it is `__launch_bounds__(256, 2)` (2 blocks per SM). The fixup kernel
uses `__launch_bounds__(nthreads/2, 1)` (`mmq.cuh:1208`).

### 9.6 cuBLAS TF32 default

The cuBLAS handle is initialized with `CUBLAS_TF32_TENSOR_OP_MATH`
(`common.cuh:1494`). This means F32 GEMM via cuBLAS uses TF32 Tensor
Cores (8-bit exponent, 10-bit mantissa) by default. The user can opt out
via `dst->op_params[0] = GGML_PREC_F32` which forces `compute_type = F32`
*without* disabling TF32 (TF32 is a property of the cuBLAS handle, not
the per-call compute type). To fully disable TF32, the user must set
`GGML_CUDA_CUBLAS_COMPUTE_TYPE=F32` env var *and* the cuBLAS handle
math mode would need to be reset (llama.cpp does not expose this).

---

## 10. Quantization Strategy

### 10.1 Activation pre-quantization to `block_q8_1_mmq`

The MMQ entry pre-quantizes `src1` (F32) to a custom MMQ layout via
`quantize_mmq_q8_1_cuda` (`quantize.cu`, declared in
`quantize.cuh:24-27`). The `block_q8_1_mmq` struct (`mmq.cuh:27-46`)
holds 128 int8 values + a 16-byte scale/sum union. The 128 values are
organized as four 32-element `block_q8_1` blocks, but interleaved so
they can be copied to shared memory as one contiguous 144-byte unit
("The y float data is first grouped as blocks of 128 values. These
blocks are then treated as individual data values and transposed."
`mmq.cuh:28-31`).

The scale/sum union layout depends on the weight quant format
(`mmq_get_q8_1_ds_layout`, `mmq.cuh:60-100`):

| Layout | Used by weight types | Format |
| ------ | -------------------- | ------ |
| `MMQ_Q8_1_DS_LAYOUT_D4` | Q1_0, Q5_0, Q8_0, MXFP4, NVFP4, Q3_K, Q6_K, IQ2_XXS, IQ3_*, IQ4_* | 4× f16 scales (one per 32 values) |
| `MMQ_Q8_1_DS_LAYOUT_DS4` | Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S | 4× (f16 scale, f16 abs-sum) pairs |
| `MMQ_Q8_1_DS_LAYOUT_D2S6` | Q2_K | 2× f16 scales + 6× f16 partial sums (Q2_K-specific) |

The abs-sum `s` is used by Q4_0/Q5_0/Q8_0 vec_dot to subtract the
implicit `-8` bias of symmetric Q4_0/Q5_0/Q8_0 encoding
(`vecdotq.cuh:115-134`, audited in ARTX09 §10.3).

### 10.2 Inline dequantize via `vec_dot_*_q8_1_impl`

The dp4a MMQ path uses the same `vec_dot_*_q8_1_impl` device functions
as MMVQ (`vecdotq.cuh:115+`). Each call reads one quant block from
`tile_x` (already in shared memory) and one `block_q8_1` chunk from
`tile_y`, performs `vdr` `dp4a` reductions, multiplies by the per-block
scales, and returns a single `float`. The dequantized values live only
in registers for the duration of the `dp4a` reduction.

The mma MMQ path does *not* use `vec_dot_*_q8_1_impl`. Instead it loads
the raw int8 quantized values into Tensor Core fragments via
`load_ldmatrix`, issues `mma.sync` to get an int32 accumulator, then
multiplies by the per-block scales. The "dequantize" step happens as a
scalar F32 multiply after the int32 Tensor Core reduction — fundamentally
different from the dp4a path which dequantizes inside the per-thread
loop.

### 10.3 NVFP4 / MXFP4 special cases

On non-Blackwell hardware, NVFP4 and MXFP4 use the Q8_1 dequant + dp4a/mma
path. The weight's 4-bit indices are expanded to 8-bit via
`get_int_from_table_16` (`vecdotq.cuh:34-95`) using the `kvalues_mxfp4`
lookup table. NVFP4 additionally supports per-block ue4m3 scales
(`ggml_cuda_ue4m3_to_fp32`, `mmq-load-tiles.cuh:1628, 1634`).

On Blackwell, the native `mma_block_scaled_fp4` path (§9.4) is used
instead, bypassing the table lookup entirely.

### 10.4 Supported quant formats

MMQ dispatch (`mmq.cu:9-79`) covers the same 22 quant formats as MMVQ
(ARTX09 §10.5): Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS,
IQ4_NL, MXFP4, NVFP4. Each gets its own template instantiation per `J
∈ {8,16,...,128}` (16 values) and `fallback ∈ {true, false}` (2 values),
so the compiled binary contains `22 * 16 * 2 = 704` MMQ kernel
instantiations (plus `22 * 16 * 2 = 704` stream-K fixup instantiations,
plus the Blackwell-specific MXFP4/NVFP4 native instantiations).

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions with correctness
consequences.

### 11.1 Floating-point reassociation

* **Per-thread `vec_dot` accumulation**. Each `vec_dot_*_q8_1_impl` call
  sums `vdr` `dp4a` results in an `int sumi` accumulator (`vecdotq.cuh:118,
  245, 260, 290`). The integer sum is exact; reassociation happens only in
  the float post-multiply (`return d4 * (sumi * ds8f.x - ...)`).
* **Per-thread `sum[...]` accumulation across K-loop**. Each thread
  accumulates into `float sum[J*I / (nwarps*warp_size)]` (`mmq.cuh:871`).
  The K-loop adds to `sum[...]` in iteration order; reassociation is
  deterministic for fixed `(type, J, fallback, K, CC)`.
* **MMA accumulator reordering**. The mma path issues `mma.sync` per
  K-fragment, then `sum[...] += C.x[l] * dA * dB` per fragment. The
  reduction order across K-fragments is fixed by the K-loop iteration
  order, but the *intra-fragment* reduction is determined by the Tensor
  Core's internal accumulator tree, which is deterministic per CC but
  differs from a sequential left-to-right sum at the ULP level.

### 11.2 Stream-K fixup atomic-add

The `mul_mat_q_stream_k_fixup` kernel reads partial sums from
`tmp_fixup` and atomic-adds them to `dst` (`mmq.cuh:1309`:
`dst[j*stride_col_dst + i] += sum[j0/nwarps]`). This is safe because:

* Each `dst[idx]` is written by exactly one fixup invocation (the fixup
  kernel iterates over previous blocks for the *same* `(i, j)` output).
* The fixup kernel runs *after* the main kernel completes (sequential
  stream launch, no overlap).
* The atomic-add is `atomicAdd` semantics (relaxed memory order is
  sufficient since the fixup kernel is the only writer to `dst` at this
  point).

However, the *reduction order* across the fixup's previous-blocks loop
is `bidx0-1, bidx0-2, ...` (descending `bidx`), which is fixed but
differs from a left-to-right sequential sum at the ULP level. Combined
with the per-block reassociation, the F32 result of a stream-K MMQ
matmul can vary at the ULP level across runs with different
`gridDim.x` (i.e., different SM counts).

### 11.3 Quantization rounding

Same as ARTX09 §11.2: `quantize_mmq_q8_1_cuda` rounds each F32 value to
int8 and stores the per-block scale `d = max_abs / 127`. The resulting
Q4×Q8 matmul is the deliberate accuracy/speed tradeoff of all
llama.cpp quantized paths.

### 11.4 Padding zeroing

Same as ARTX09 §11.3: if `src0` is a temporary compute buffer with
`size_alloc > size_data`, `ggml_cuda_mul_mat_q` issues
`cudaMemsetAsync` on the tail bytes (`mmq.cu:107-114`). Without this,
`vec_dot` calls that read past the data end could pick up non-zero
garbage in the padding, which would be multiplied by the activation
scale and corrupt the result. The bad-padding-clear check
(`ggml-cuda.cu:1823-1824`) routes to cuBLAS if `src0` is a view of
another tensor (clearing would overwrite valid data).

### 11.5 Determinism

* **Tiled-K MMQ** (no stream-K): deterministic for fixed `(type, J,
  fallback, K, CC)`. Each output tile is computed by exactly one block
  in a fixed K-iteration order.
* **Stream-K MMQ**: deterministic for fixed `(type, J, fallback, K, CC,
  nsm)`. The `nsm`-dependent block decomposition means the result can
  vary at the ULP level across GPUs with different SM counts.
* **MMF**: deterministic for fixed `(T, rows_per_block, cols_per_block,
  nwarps, K, CC)`. Pure tiled GEMM, no atomics.
* **cuBLAS**: opaque. cuBLAS may pick different kernels for different
  shapes/CCs; the result can vary at the ULP level.
* **Atomic accumulation on `dst`**: only in stream-K fixup. Otherwise
  output tiles are written by exactly one block.

### 11.6 Architecture-specific assumptions

* `use_mma_data_layout()` is true on Turing+ / CDNA / RDNA3+. The dp4a
  path is used on Pascal and RDNA2. Same input → different reduction
  tree → different ULPs.
* Blackwell FP4 native path produces different ULPs than the Q8_1 path
  because the FP4 Tensor Core does its own internal accumulation
  (e2m1×e2m1→f32 with block scaling), not the int8→int32→f32 path.
* `MATRIX_ROW_PADDING = 512` (`common.cuh:176`). Activations are padded
  to a multiple of 512 elements. The padding contributes zero to the
  dot product (zeros × scale = 0), but the padding bytes themselves
  must be zero-filled by the quantize kernel.
* `__launch_bounds__(nthreads, occupancy)` with `occupancy = 1` on
  Ampere/Blackwell/CDNA means only one block resident per SM. If the
  register pressure is too high, the kernel may spill to local memory;
  this is not detected at compile time.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization | Where | Notes |
| ------------ | ----- | ----- |
| Per-(type,J,arch) config table | `mmq.cuh:164-242`, `mmq-config-*.cuh` | Compile-time-constant tiling policy per arch; same struct visible to host and device. |
| Template-on-`J` enumeration | `mmq.cuh:1443-1524` | 16-way switch over J ∈ {8..128} step 8; compiler unrolls J-axis. |
| Template-on-`type` enumeration | `mmq.cu:8-80` | 22-way switch over quant type; each gets its own load_tiles + vec_dot + write_back. |
| Dual dp4a/mma codepaths | `mmq-vec-dot.cuh:*_dp4a`, `*_mma` | Same template compiles to either CUDA-core or Tensor-Core instructions based on arch. |
| Stream-K decomposition | `mmq.cuh:918, 1030-1205` | Load-balanced tiling for shapes where tiled-K leaves SMs idle; fixup kernel for partial tiles. |
| `mma.sync.aligned.m16n8k32` (Ampere+) | `mma.cuh:946` | 2× K width per instruction vs Turing's m16n8k16. |
| Blackwell `mma_block_scaled_fp4` | `mma.cuh:1126-1154` | Native FP4×FP4→F32 with hardware block scaling; bypasses Q8_1 dequant entirely. |
| `block_q8_1_mmq` transposed layout | `mmq.cuh:27-46` | 128 int8s + 16 B scale/sum, padded to avoid bank conflicts, copied as one contiguous block. |
| Bank-conflict-aware sram_stride | `mmq.cuh:131-158` | `sram_stride % 8 == 4` static_asserted for every layout — XOR-padding for 8-bank shared memory. |
| `load_ldmatrix` for Tensor Core fragments | `mma.cuh:load_ldmatrix` | Single instruction loads 16×8 fragment from shared memory into warp registers; required for `mma.sync`. |
| `fast_div_modulo` / `fastmodulo` | `mmq.cuh:1034+` | Wang-style fast integer division for stream-K index arithmetic. |
| `__launch_bounds__(nthreads, occupancy)` | `mmq.cuh:921` | Per-arch occupancy target (1 or 2 blocks/SM). |
| `CUDA_SET_SHARED_MEMORY_LIMIT` | `mmq.cuh:1375-1376` | Opt-in to >48 KB shared memory per block on Ampere+. |
| MoE `mm_ids_helper` + `expert_bounds` | `mmq.cu:197-200, mmid.cu` | On-device expert routing sort/compact; no host synchronization. |
| MoE `dedup_bcast` quantize-once-scatter | `mmq.cu:190, 222-235` | When `ne11==1 && n_expert_used>1`, quantize each token once and scatter to all expert slots. |
| `use_native_fp4` dispatch | `mmq.cu:128-130` | Blackwell FP4 path bypasses Q8_1, uses `block_fp4_mmq` + native FP4 mma. |
| cuBLAS compute-type auto-selection | `ggml-cuda.cu:1619-1660` | F16 for quantized weights on fast-fp16 hardware; F32 if user requests; env var override. |
| cuBLAS output-dtype auto-selection | `ggml-cuda.cu:1507-1528` | F32 output for F16 compute on Volta/RDNA4/CDNA; F16 output otherwise with explicit dequantize. |
| cuBLAS four dispatch variants | `ggml-cuda.cu:1540-1610` | Sgemm / GemmEx / GemmStridedBatchedEx / GemmBatchedEx based on shape and contiguity. |
| `J_best` minimization | `mmq.cuh:1452-1468` | Picks smallest J that minimizes output-column tile count, capped by `smpbo` shared-memory limit. |
| `nwarps_best` minimization (MMF) | `mmf.cuh:646-655` | Picks smallest nwarps that minimizes K-iteration count, capped by `max_block_size`. |

### 12.2 Optimizations *not* present (worth noting)

* **No `cp.async` pipelining in MMQ.** The kernel does synchronous
  global→shared loads followed by `__syncthreads()` between every
  K-iteration (`mmq.cuh:887, 891, 903, 907`). The Ampere+ `cp.async`
  hardware (and the `cp-async.cuh` helper) is available but unused.
  Flash-attention uses it (ARTX11); MMQ does not. A double-buffered
  `cp.async` pipeline would let the next K-tile load overlap with the
  current `vec_dot` compute.
* **No Hopper `wgmma.mma_async`.** Hopper's warp-group MMA
  (`wgmma.mma_async.sync.aligned.m64n*k16`) gives 4× throughput per
  warp group vs Ampere's `mma.sync`. MMQ uses only Ampere-style
  `mma.sync.m16n8k*`. No `cuTensorMap`/TMA either.
* **No `cp.async.bulk` (TMA) on Hopper/Blackwell.** The bulk-copy
  hardware is unused; MMQ still uses per-thread `memcpy` from global to
  shared.
* **No persistent L2 hints** (`cudaStreamSetAttribute` with
  `cudaStreamAttributeAccessPolicyWindow`). The weight matrix would
  benefit from L2 persistence across decode steps.
* **No kernel fusion** in MMQ/MMF themselves. FFNGLU fusion (gate +
  bias + GLU) exists in MMVQ/MMVF (ARTX09) but not in MMQ/MMF — MMQ is
  only called for `ne11 > 8`, and the fusion pattern is currently
  `ncols_dst == 1`-only.
* **No F16-accumulate paths in MMQ.** All MMQ accumulation is F32
  (`float sum[...]` in dp4a, `tile<16, 8, float>` in mma). Contrast
  MMVF which has `type_acc = half` for F16 weights (ARTX09-W7).
* **No split-K parallelism** in MMF. MMF is pure tiled GEMM; no
  atomic-accumulate split-K scheme for very long K. (MMQ has stream-K
  which serves a similar purpose but is not the same algorithm.)

---

## 13. Architectural Strengths

1. **Per-(type, J, arch) config-table architecture** is the single best
   design decision in MMQ. The `ggml_cuda_mmq_config` struct carries the
   full tiling policy as compile-time constants; the host and device see
   the same struct; the per-arch files are pure data (no logic). Adding
   a new GPU family means adding one config file and one dispatch branch
   in `ggml_cuda_mmq_get_config` — no kernel code changes. This is the
   CUDA analogue of ARTX01-F03 (CPU type-traits table).

2. **Dual dp4a/mma codepaths behind a single template.** The
   `use_mma_data_layout()` predicate lets the *same* `mul_mat_q<type, J,
   fallback>` template compile down to either CUDA-core `dp4a` or
   Tensor-Core `mma.sync` instructions. The compiler eliminates the
   unused codepath via `if constexpr` / `#ifdef`. This is the cleanest
   way to support both pre-Turing (no int8 Tensor Cores) and Turing+
   (int8 Tensor Cores) GPUs from one source.

3. **Stream-K decomposition with fixup kernel.** The stream-K algorithm
   (`mmq.cuh:918, 1030-1205`) load-balances shapes where tiled-K leaves
   SMs idle (e.g., `ntiles_dst < nsm`). The launch heuristic
   (`mmq.cuh:1407-1410`) automatically picks tiled-K when tile
   efficiency ≥ 90% and stream-K otherwise. The fixup kernel handles
   partial-tile atomic-add without complicating the main kernel.

4. **Blackwell native FP4 block-scaled MMA.** The
   `mma_block_scaled_fp4<type>` wrapper (`mma.cuh:1126-1154`) exposes
   the `kind::mxf4.block_scale.scale_vec::2X` and
   `kind::mxf4nvf4.block_scale.scale_vec::4X` PTX instructions. The
   `block_fp4_mmq` layout + `quantize_mmq_fp4_cuda` + native FP4 vec_dot
   form a complete FP4 path that bypasses Q8_1 dequantization. This is
   forward-looking hardware support done right — the rest of MMQ is
   unchanged, only the load_tiles + vec_dot + config entries differ.

5. **`block_q8_1_mmq` transposed-and-padded layout.** The 128-int8 +
   16-byte scale layout lets the activation be copied to shared memory
   as one contiguous 144-byte block, with the 16-byte padding doubling
   as scale storage. The `sram_stride % 8 == 4` static_assert
   (`mmq.cuh:152-158`) enforces XOR-padding to avoid 8-bank shared
   memory conflicts.

6. **`J_best` runtime selection over a compile-time-enumerated `J`
   axis.** The 16-way `switch (J_best)` (`mmq.cuh:1470-1523`) is verbose
   but lets the compiler unroll the J-axis in the kernel while still
   picking the best J at runtime based on the actual `ncols_dst`. The
   `smpbo` check (`mmq.cuh:1458`) ensures the chosen J's shared-memory
   footprint fits on the device.

7. **MoE MUL_MAT_ID with dedup-bcast quantize-once-scatter.** When
   gate/up activations are broadcast across experts (`ne11 == 1 &&
   n_expert_used > 1`), MMQ uses `quantize_scatter_mmq_*_cuda` which
   quantizes each token *once* and scatters to all its expert slots
   (`mmq.cu:222-235`). This avoids `n_expert_used`× duplicate
   quantization work, which can be 8× for typical MoE configs.

8. **MMF hand-written `mma.sync` instead of cuBLAS for small batches.**
   For `ne11 ∈ {1..16}`, cuBLAS's per-call overhead (kernel selection,
   workspace allocation) dominates. MMF's hand-written kernel
   (`mmf.cuh:48-294`) launches in <1 µs and uses the exact same
   `mma.sync` instructions cuBLAS would, but with a tighter,
   shape-specialized tile. The 16-way `cols_per_block` enumeration
   (`mmf.cuh:748-832`) gives the compiler full visibility into the
   output-column count.

9. **cuBLAS compute-type and output-dtype auto-selection.** The
   `prefer_f32_output` heuristic (`ggml-cuda.cu:1507-1512`) picks F32
   output when the hardware's F16→F32 conversion is fast (Volta, RDNA4,
   CDNA) and F16 output otherwise, with an explicit `to_fp32_cuda`
   dequantize step. This avoids the F16-accumulator overflow risk
   (ARTX09-W7) on hardware where F32 output is cheap, while preserving
   F16-output throughput on hardware where it isn't.

---

## 14. Architectural Weaknesses

### W1 — No `cp.async` pipelining in MMQ

**Evidence**: `mmq.cuh:887, 891, 903, 907` — four `__syncthreads()` per
K-iteration. `cp-async.cuh:22-46` defines `cp_async_cg_16<preload>` with
L2::256B/128B/64B hints; `common.cuh:356` defines `cp_async_available(cc)`.
Grep for `cp.async` in `mmq.cuh` / `mmq-vec-dot.cuh` / `mmq-load-tiles.cuh`
returns zero matches.

**Impact**: MMQ has zero K-iteration pipeline depth. Each iteration
serializes: global load → `__syncthreads` → vec_dot → `__syncthreads` →
global load → `__syncthreads` → vec_dot → `__syncthreads`. A
double-buffered `cp.async` pipeline would overlap the next iteration's
global load with the current iteration's vec_dot compute, hiding global
memory latency. Flash-attention already uses this pattern
(`fattn-mma-f16.cuh:371-475`); MMQ does not.

**Why it's hard to fix**: The dp4a path reads `tile_x` and `tile_y`
inside the `vec_dot` call via per-thread shared-memory indices, so a
naive double-buffer would require duplicating the entire tile_x and
tile_y allocation in shared memory. The mma path uses `load_ldmatrix`
which expects a specific layout; a double-buffer would require
`cp.async`-compatible staging buffers.

### W2 — No Hopper `wgmma.mma_async` or TMA

**Evidence**: Grep for `wgmma`, `cuTensorMap`, `cp.async.bulk`,
`tc_gen5` in `mma.cuh` / `mmq.cuh` / `mmf.cuh` returns zero matches.
`GGML_CUDA_CC_HOPPER = 900` is defined (`common.cuh:55`) but only PDL
(`ggml_cuda_pdl_sync`, ARTX08 §9.4) is used on Hopper.

**Impact**: Hopper's `wgmma.mma_async.sync.aligned.m64n*k16` gives 4×
throughput per warp group vs Ampere's `mma.sync.m16n8k*`. Hopper's TMA
(`cp.async.bulk.tensor`) gives asynchronous bulk shared-memory loads
without occupying registers. MMQ on Hopper uses the Ampere codepath
(`mma.sync.m16n8k16` + `m16n8k32`), leaving 4× Tensor Core throughput on
the table. Blackwell's `tc_gen5.mma` is similarly unused (only
`mma_block_scaled_fp4` for FP4 is used).

### W3 — Hand-maintained per-arch `should_use_mmq` thresholds

**Evidence**: `mmq.cu:256-371` — a 115-line decision tree with comments
like "As of ROCM 7.0 rocblas/tensile performs very poorly on CDNA3"
(`:316`), "For some quantization types MMQ can have lower peak TOPS than
hipBLAS so it's only faster for sufficiently small batch sizes" (`:343`),
and per-type thresholds like `case GGML_TYPE_Q2_K: return ne11 <= 128`
(`:347`).

**Impact**: Maintenance burden grows with `(arch count) × (quant
count) × (MoE config)`. A single wrong threshold can regress performance
by 10× with no compile-time warning. The CDNA3 comment cites a
hipblaslt bug ("currently suffering from a crash on this architecture"
`:317`) with a `TODO: Revisit when hipblaslt is fixed` — when the bug is
fixed upstream, the threshold will silently keep forcing MMQ.

### W4 — Per-arch config tables are 700+ lines of copy-paste

**Evidence**: `mmq-config-ampere.cuh` is 366 lines, mostly
`CASE(GGML_TYPE_Q4_0, 256, 1, 128, 8, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,
MMQ_ITER_K, true, true);` repeated for 22 types × 16 J values × 2
fallback. The six config files total 1384 lines of near-duplicate `CASE`
macros.

**Impact**: When a new quant format is added, all six config files must
be updated with the right `(sram_layout, K_vram, stream_k)` tuple per
arch. A single wrong entry produces a `GGML_TYPE_COUNT` sentinel
(`mmq.cuh:365`) which the kernel silently skips via `NO_DEVICE_CODE`
(`mmq.cuh:932-935`) — the kernel runs but produces no output, and the
user sees a corrupted result with no error. A code generator (Python
script emitting the CASE lines from a single YAML/JSON source) would be
more maintainable.

### W5 — `mul_mat_q_process_tile` has two y-loads per K-iteration

**Evidence**: `mmq.cuh:875-908` — the K-loop body loads y, syncs, vecdots,
syncs, loads y again (the second half of `MMQ_ITER_K = 256`), syncs,
vecdots, syncs. The comment is implicit: `MMQ_ITER_K = 256` but
`MMQ_TILE_NE_K = 32`, so each iteration covers `2 * 32 = 64` K-elements
but the y-load is split into two 32-element chunks.

**Impact**: Double the `__syncthreads` count vs a single 64-element
y-load. The split exists because `tile_y` is sized `J * MMQ_TILE_Y_K =
J * 36` ints, and loading 2×`MMQ_TILE_Y_K` at once would exceed the
tile_y allocation. A larger tile_y (and larger shared-memory budget)
would let the kernel load the full `MMQ_ITER_K` worth of y in one pass.

### W6 — `mul_mat_q_stream_k_fixup` reads partial sums serially

**Evidence**: `mmq.cuh:1246-1274` — the fixup kernel iterates `bidx =
bidx0 - 1; bidx--;` in a `while(true)` loop, reading
`tmp_last_tile[bidx*(J*I) + j*I + i]` and accumulating into `sum[...]`.

**Impact**: For large `gridDim.x` and unlucky tile decomposition, the
fixup kernel can do `O(gridDim.x)` serial reads per output element. The
fixup kernel uses half the threads of the main kernel (`block_dims.y/2`,
`mmq.cuh:1423`), so its throughput is half. For shapes where stream-K is
selected but the fixup is large, the fixup can dominate total runtime.

### W7 — `mmf_get_max_block_size` and `mmf_get_padding` are per-vendor hardcodes

**Evidence**: `mmf.cuh:12-26` — `if (GGML_CUDA_CC_IS_CDNA(cc)) return
512; else return 256;` for max block size, `if (GGML_CUDA_CC_IS_CDNA(cc))
return 2; else return 4;` for padding. The device-side
`mmf_get_padding()` is a `#if defined(AMD_MFMA_AVAILABLE) return 2; #else
return 4; #endif` compile-time constant.

**Impact**: When a new GPU family is added (e.g., Rubin), these helpers
must be updated, or the new family silently gets the "else" branch's
values. There is no compile-time check that the chosen padding/max_block_size
is optimal for the new family.

### W8 — `should_use_mmf` rejects CDNA1/CDNA2 F16/BF16

**Evidence**: `mmf.cu:171-175` — `else if (GGML_CUDA_CC_IS_CDNA2(cc) &&
(type == GGML_TYPE_F16 || type == GGML_TYPE_BF16)) return false;` and
the same for CDNA1. The comment is `//TODO: truse CDNA2 as CDNA1, tune
the perf when CDNA2 is available.`

**Impact**: CDNA1/CDNA2 (MI100/MI210) F16/BF16 GEMM falls through to
cuBLAS (hipBLAS) for `ne11 > 16`. hipBLAS may be slower than a tuned
MMF on these archs. The TODO suggests the team knows this is suboptimal
but hasn't tuned it yet.

### W9 — `ggml_cuda_mul_mat_cublas_impl` allocates `dst_temp` for F16 output even when `dst` is F32

**Evidence**: `ggml-cuda.cu:1520-1527` — when `!prefer_f32_output` and
`compute_type != F32`, `dst_ptr = (char *) dst_temp.alloc(ne_dst)` and
`nbd2 /= sizeof(float) / sizeof(cuda_t)`. The cuBLAS call writes F16/BF16
to `dst_temp`, then `to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, ...)`
converts back to F32.

**Impact**: For F16/BF16 GEMM with F32 output (the common case for
non-Volta/RDNA4/CDNA), this costs one extra `ne_dst * sizeof(cuda_t)`
allocation and one extra `ne_dst * sizeof(float)` dequantize kernel
launch. On shapes where cuBLAS is already faster than MMF, the
allocation + dequantize overhead may be negligible, but on shapes near
the MMF/cuBLAS boundary it can tip the balance.

### W10 — TF32 cuBLAS default is implicit

**Evidence**: `common.cuh:1494` — `cublasSetMathMode(cublas_handles[device],
CUBLAS_TF32_TENSOR_OP_MATH)`. No user-facing flag to disable.

**Impact**: F32 GEMM via cuBLAS uses TF32 Tensor Cores (10-bit mantissa)
by default. The user can set `dst->op_params[0] = GGML_PREC_F32` to force
`compute_type = F32`, but this does *not* disable TF32 — TF32 is a
property of the cuBLAS handle's math mode, not the per-call compute type.
To fully disable TF32, the user must either patch the source or use a
non-cuBLAS path. This is a precision reduction that is not clearly
documented.

### W11 — `load_ldmatrix` requires 16-byte shared-memory alignment

**Evidence**: `mma.cuh:17` — "all pointers for load_ldmatrix must be to
shared memory and aligned to 16 bytes."

**Impact**: The `tile_x` and `tile_y` shared-memory layouts must be
designed to keep every `load_ldmatrix` source 16-byte aligned. The
`sram_stride % 8 == 4` static_assert (`mmq.cuh:152-158`) is partly
about this — the `+4` padding ensures the next row starts on a 16-byte
boundary (since each int is 4 bytes, `8 ints + 4 ints = 12 ints = 48
bytes` is not 16-byte aligned, but the layout is more subtle). A
misalignment would silently produce wrong Tensor Core results.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda` | **ADOPT** | Per-(type, J, arch) config-table architecture | Compile-time-constant tiling policy per arch; same struct visible to host and device; adding a GPU family = adding one config file. |
| `glcuda` | **ADOPT** | Dual dp4a/mma codepaths behind `use_mma_data_layout()` | Same template compiles to CUDA-core or Tensor-Core instructions; compiler eliminates the unused path. |
| `glcuda` | **ADOPT** | Stream-K decomposition with fixup kernel | Load-balanced tiling for shapes where tiled-K leaves SMs idle; auto-switches based on tile efficiency. |
| `glcuda` | **ADOPT** | Blackwell native FP4 `mma_block_scaled_fp4` path | Forward-looking hardware support; bypasses Q8_1 dequant; rest of MMQ unchanged. |
| `glcuda` | **ADOPT** | `block_q8_1_mmq` transposed-and-padded layout | Single contiguous copy to shared memory; padding doubles as scale storage; bank-conflict-free. |
| `glcuda` | **ADOPT** | Template-on-`J` enumeration with runtime `J_best` | Compiler unrolls J-axis; runtime picks smallest J that minimizes tile count, capped by `smpbo`. |
| `glcuda` | **ADOPT** | MoE `dedup_bcast` quantize-once-scatter | When `ne11==1 && n_expert_used>1`, quantize each token once and scatter; avoids `n_expert_used`× duplicate work. |
| `glcuda` | **ADOPT** | MMF hand-written `mma.sync` for small-batch F16/BF16/F32 GEMM | Avoids cuBLAS per-call overhead for `ne11 ≤ 16`; shape-specialized via `cols_per_block` enumeration. |
| `glcuda` | **ADAPT** | `should_use_mmq` per-arch thresholds | Keep the per-arch structure but generate the thresholds from a single source of truth; add per-arch test coverage to catch stale thresholds. |
| `glcuda` | **ADAPT** | cuBLAS compute-type / output-dtype auto-selection | Keep the heuristic but make TF32 explicit (opt-in or opt-out flag), not implicit via `CUBLAS_TF32_TENSOR_OP_MATH` default. |
| `glcuda` | **REJECT** | Absence of `cp.async` pipelining in MMQ | Add a double-buffered `cp.async` pipeline (Ampere+) to overlap global loads with vec_dot compute. Flash-attention already does this. |
| `glcuda` | **REJECT** | Absence of Hopper `wgmma` / TMA | Use Hopper's `wgmma.mma_async` (4× Tensor Core throughput per warp group) and TMA (`cp.async.bulk.tensor`) for bulk shared-memory loads. |
| `glcuda` | **MONITOR** | Hand-maintained per-arch config `CASE` tables | Watch for stale entries when new quants/archs are added; consider a Python code generator. |
| `glcuda` | **MONITOR** | TF32 cuBLAS default | Watch for precision regressions; consider exposing a user flag to disable TF32 without source patches. |
| `glcuda` | **MONITOR** | `should_use_mmf` CDNA1/CDNA2 F16/BF16 rejection | Watch for hipBLAS regressions; revisit the TODO when CDNA2 tuning is done. |
| `glcuda` | **DEFER** | Stream-K fixup kernel | Adopt once the main MMQ kernel is in place; the fixup is a follow-on optimization for shapes where tiled-K is inefficient. |
| `GATE` | **ADOPT** | `J_best` as a graph-plan-time constant | The graph planner knows the decode batch size; it can pin `J` at plan time so the right template instance is launched without runtime dispatch. |
| `GATE` | **ADOPT** | `should_use_mmq` decision at plan time | The planner knows `type, cc, ne11, n_experts` at plan time; no need to re-evaluate per call. |

---

## 16. Recommendations

### R1 — ADOPT per-(type, J, arch) config-table architecture
**Priority:** Critical **Difficulty:** L **Dependencies:** none
GwenLand's `glcuda` MMQ kernel should be driven by a
`gl_mmq_config[type][J][arch]` table carrying `(nthreads, occupancy, I,
sram_layout, K_vram, stream_k, fallback)` as compile-time constants. The
same struct should be visible to host and device (`__host__ __device__`
qualifier). One config file per GPU family. Same ABI as
`ggml_cuda_mmq_config` (`mmq.cuh:164-203`).

### R2 — ADOPT dual dp4a/mma codepath design
**Priority:** Critical **Difficulty:** XL **Dependencies:** R1
GwenLand's MMQ should have two tile-vecdot implementations per quant
format: `*_dp4a` (CUDA-core, for pre-Turing/RDNA2) and `*_mma`
(Tensor-Core, for Turing+/CDNA/RDNA3+). The `use_mma_data_layout()`
predicate selects the codepath at compile time. Both codepaths share
the same driver skeleton (`mul_mat_q_process_tile` equivalent).

### R3 — ADOPT stream-K decomposition with fixup kernel
**Priority:** High **Difficulty:** L **Dependencies:** R1, R2
Implement the stream-K algorithm (arXiv:2301.03598) for shapes where
tiled-K leaves SMs idle. Auto-switch between tiled-K and stream-K based
on a 90% tile-efficiency threshold. Use a separate fixup kernel for
partial-tile atomic-add to `dst`.

### R4 — ADOPT Blackwell native FP4 path
**Priority:** High **Difficulty:** L **Dependencies:** R1, R2
Implement `mma_block_scaled_fp4<type>` via the
`mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64` (MXFP4)
and `kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64` (NVFP4) PTX
instructions. Use a separate `block_fp4_mmq` activation layout and
`quantize_mmq_fp4_cuda` quantizer. Bypass Q8_1 dequant entirely on
Blackwell.

### R5 — REJECT absence of `cp.async` pipelining; add double-buffered pipeline
**Priority:** High **Difficulty:** XL **Dependencies:** R1, R2
Add a double-buffered `cp.async` pipeline (Ampere+) to overlap the next
K-tile's global load with the current K-tile's vec_dot compute. Use the
`cp.async.cg.shared.global.L2::256B` hint for best L2 behavior. This is
the single highest-impact optimization missing from MMQ. Flash-attention
already uses this pattern (`fattn-mma-f16.cuh:371-475`); MMQ should too.

### R6 — REJECT absence of Hopper `wgmma` / TMA; add Hopper codepath
**Priority:** High **Difficulty:** XL **Dependencies:** R1, R2
Use Hopper's `wgmma.mma_async.sync.aligned.m64n*k16` (4× Tensor Core
throughput per warp group) and TMA (`cp.async.bulk.tensor` via
`cuTensorMapEncodeTiled`) for bulk shared-memory loads. This requires a
new `ggml_cuda_mmq_config` entry with `nthreads = 128` (one warp group)
and a new tile-vecdot implementation using `wgmma` instead of `mma.sync`.

### R7 — ADOPT `block_q8_1_mmq` transposed-and-padded layout
**Priority:** High **Difficulty:** M **Dependencies:** R1
Replicate the 128-int8 + 16-byte scale/sum layout with the three
`mmq_q8_1_ds_layout` variants (D4 / DS4 / D2S6). The
`sram_stride % 8 == 4` static_assert enforces XOR-padding for 8-bank
shared memory.

### R8 — ADOPT MoE MUL_MAT_ID with dedup-bcast
**Priority:** Medium **Difficulty:** L **Dependencies:** R1, R2
Implement the `mm_ids_helper` on-device expert routing sort/compact, the
`expert_bounds` range array, and the `dedup_bcast` quantize-once-scatter
optimization for `ne11==1 && n_expert_used>1`.

### R9 — ADAPT `should_use_mmq` per-arch thresholds to a single source of truth
**Priority:** Medium **Difficulty:** M **Dependencies:** R1
Generate the per-arch `(mmq_thresholds[type][arch])` thresholds from a
single YAML/JSON source. Add per-arch test coverage to catch stale
thresholds. Document the reasoning behind each threshold (e.g., the
CDNA3 hipblaslt-crash TODO at `mmq.cu:316-318`).

### R10 — ADOPT MMF hand-written `mma.sync` for small-batch GEMM
**Priority:** Medium **Difficulty:** L **Dependencies:** R1
For `ne11 ∈ {1..16}` F32/F16/BF16 GEMM, use hand-written `mma.sync` PTX
(`mma.cuh` equivalent) instead of cuBLAS. Template on
`cols_per_block ∈ {1..16}` for compiler unrolling. Auto-tune `nwarps`
based on K-iteration count.

### R11 — ADAPT cuBLAS compute-type / output-dtype selection; make TF32 explicit
**Priority:** Medium **Difficulty:** S **Dependencies:** none
Keep the auto-selection heuristic but make TF32 an explicit opt-in flag
(per-tensor or per-backend), not an implicit global default via
`CUBLAS_TF32_TENSOR_OP_MATH`. Document the precision implications.

### R12 — MONITOR hand-maintained per-arch config `CASE` tables
**Priority:** Low **Difficulty:** M **Dependencies:** R1
Watch for stale entries when new quants/archs are added. Consider
replacing the 1384 lines of `CASE` macros across six config files with a
Python code generator that emits the CASE lines from a single
declarative source.

---

## 17. Findings

### Finding ARTX10-F01

```
Finding ID:           ARTX10-F01
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MMQ config-table architecture
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             ggml_cuda_mmq_config (struct), ggml_cuda_mmq_get_config
Lines:                164-242
Summary:              MMQ expresses its full tiling policy as a per-(type, J,
                      arch) constexpr config struct, shared between host and device.
Observation:          The ggml_cuda_mmq_config struct carries (nthreads,
                      occupancy, I, J, sram_layout, K_vram, stream_k, fallback)
                      as compile-time constants. Six per-arch config files
                      (Pascal, Ampere, Blackwell, CDNA, RDNA2, RDNA4) populate
                      the table via the CASE macro. The host function
                      ggml_cuda_mmq_get_config(type, J, fallback, cc) dispatches
                      to the right per-arch function based on cc; the device
                      function ggml_cuda_mmq_get_config(type, J, fallback) is a
                      constexpr that selects at compile time via #ifdef. Both
                      must agree — enforced by the host launching the template
                      instance compiled for the matching arch.
Evidence:             mmq.cuh:164-203 (struct), 205-223 (CASE macro + includes),
                      225-242 (host dispatch), 244-263 (device dispatch).
Architectural Impact: Adding a new GPU family = adding one config file and one
                      dispatch branch. No kernel code changes. Adding a new
                      quant = adding entries to all six config files. The
                      config is the single source of truth for tiling policy.
Correctness Impact:   None. The config is purely a performance hint; the kernel
                      produces correct results for any valid config.
Optimization Type:    tiling / blocking (compile-time-constant tile sizes).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda's MMQ should be driven by an equivalent
                      gl_mmq_config[type][J][arch] table.
Priority:             Critical
Difficulty:           L
Dependencies:         none
Confidence:           High
```

### Finding ARTX10-F02

```
Finding ID:           ARTX10-F02
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            MMQ tile-vecdot dual codepaths
Source File:          ggml/src/ggml-cuda/mmq.cuh, ggml/src/ggml-cuda/mmq-vec-dot.cuh
Function:             use_mma_data_layout, ggml_cuda_mmq_vec_dot_*_dp4a, ggml_cuda_mmq_vec_dot_*_mma
Lines:                mmq.cuh:188-201; mmq-vec-dot.cuh:10-280 (Q4_0/Q4_1/Q8_0 examples)
Summary:              Each quant format has two tile-vecdot implementations
                      (dp4a on CUDA cores, mma on Tensor Cores); a compile-time
                      predicate selects the codepath.
Observation:          The use_mma_data_layout() predicate returns true on
                      Turing+ / CDNA / RDNA3+ (based on TURING_MMA_AVAILABLE,
                      AMD_MFMA_AVAILABLE, AMD_WMMA_AVAILABLE). When true, the
                      kernel uses load_ldmatrix + mma.sync PTX via the mma
                      wrappers in mma.cuh. When false, the kernel uses
                      vec_dot_*_q8_1_impl + ggml_cuda_dp4a on CUDA cores. The
                      dp4a path uses a different shared-memory layout
                      (MMQ_DP4A_TXS_* macros) than the mma path (single
                      sram_stride-padded layout). The two paths share the same
                      driver skeleton (mul_mat_q_process_tile).
Evidence:             mmq.cuh:188-201 (predicate), 521-817 (util_funcs switch
                      with both _dp4a and _mma entries); mmq-vec-dot.cuh:10-58
                      (Q4_0 dp4a), 142-280 (Q8_0 mma).
Architectural Impact: Same template compiles to either CUDA-core or Tensor-Core
                      instructions. Compiler eliminates the unused path via
                      if constexpr / #ifdef. Lets MMQ support both pre-Turing
                      (no int8 Tensor Cores) and Turing+ from one source.
Correctness Impact:   None. Both paths produce equivalent results at the ULP
                      level (different reduction trees, same arithmetic).
Optimization Type:    SIMD (Tensor Core mma.sync vs CUDA-core dp4a).
GwenLand Target:      glcuda
Recommendation:       ADOPT. glcuda's MMQ should have dual dp4a/mma codepaths
                      behind a use_mma_data_layout() predicate.
Priority:             Critical
Difficulty:           XL
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX10-F03

```
Finding ID:           ARTX10-F03
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMQ stream-K decomposition
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             mul_mat_q, mul_mat_q_stream_k_fixup, launch_mul_mat_q
Lines:                918-1205 (kernel), 1207-1343 (fixup), 1395-1441 (launch)
Summary:              MMQ implements stream-K work decomposition (arXiv:2301.03598)
                      with a separate fixup kernel for partial-tile atomic-add.
Observation:          When config.stream_k == true (Ampere/Blackwell/CDNA), the
                      launch heuristic picks between tiled-K (gridDim =
                      ntiles_dst, no fixup) and stream-K (gridDim = nsm, with
                      fixup) based on a 90% tile-efficiency threshold. In
                      stream-K mode, each block is assigned a contiguous range
                      [kbc, kbc_stop) of K-block iterations across the entire
                      (i, j, tile) space. The last partial tile of each block
                      writes to tmp_fixup[blockIdx.x * J*I] instead of dst. A
                      second kernel mul_mat_q_stream_k_fixup iterates over
                      previous blocks, sums partial sums from the fixup buffer,
                      and atomic-adds the final result to dst.
Evidence:             mmq.cuh:918 (comment citing arXiv:2301.03598), 1030-1127
                      (stream-K block body), 1129-1205 (last-tile fixup write),
                      1207-1343 (fixup kernel), 1407-1414 (launch heuristic),
                      1418-1440 (fixup allocation and launch).
Architectural Impact: Load-balanced tiling for shapes where tiled-K leaves SMs
                      idle (e.g., ntiles_dst < nsm). The fixup kernel handles
                      partial-tile atomic-add without complicating the main
                      kernel. Auto-switching based on tile efficiency avoids
                      stream-K overhead when tiled-K is already efficient.
Correctness Impact:   The atomic-add to dst is safe because each dst[idx] is
                      written by exactly one fixup invocation. The reduction
                      order is fixed (descending bidx) but differs from a
                      sequential sum at the ULP level. Stream-K results can
                      vary across GPUs with different SM counts.
Optimization Type:    persistent threads / asynchronous execution (stream-K
                      block decomposition).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Implement stream-K with fixup for shapes where
                      tiled-K is inefficient. Auto-switch based on tile
                      efficiency.
Priority:             High
Difficulty:           L
Dependencies:         ARTX10-F01, ARTX10-F02
Confidence:           High
```

### Finding ARTX10-F04

```
Finding ID:           ARTX10-F04
Category:             ADOPT
Engine:               CUDA
Component:            Blackwell native FP4 block-scaled MMA
Source File:          ggml/src/ggml-cuda/mma.cuh, ggml/src/ggml-cuda/mmq.cuh, ggml/src/ggml-cuda/mmq-vec-dot.cuh, ggml/src/ggml-cuda/mmq-load-tiles.cuh, ggml/src/ggml-cuda/mmq-config-blackwell.cuh
Function:             mma_block_scaled_fp4, ggml_cuda_mmq_vec_dot_fp4_fp4_mma, ggml_cuda_mmq_load_tiles_mxfp4_fp4, ggml_cuda_mmq_load_tiles_nvfp4_nvfp4
Lines:                mma.cuh:1126-1154; mmq.cuh:51-58 (block_fp4_mmq), 663-680 (util_funcs dispatch); mmq-vec-dot.cuh:1186-1250; mmq-load-tiles.cuh:1542-1582, 1640-1679; mmq-config-blackwell.cuh:1-37
Summary:              Blackwell MMQ uses native mma.sync.aligned.kind::mxf4.block_scale
                      PTX for MXFP4/NVFP4, bypassing Q8_1 dequantization entirely.
Observation:          When blackwell_mma_available(cc) (CC >= 1200, < 1300) and
                      src0->type ∈ {MXFP4, NVFP4}, MMQ uses a separate dispatch
                      path. The activation is quantized to block_fp4_mmq (FP4
                      format matching the weight) via quantize_mmq_fp4_cuda. The
                      tile loader copies raw FP4 nibbles into shared memory
                      without int8 unpacking. The tile vecdot issues
                      mma_block_scaled_fp4<type> which uses the
                      kind::mxf4.block_scale.scale_vec::2X.m16n8k64 (MXFP4) or
                      kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 (NVFP4)
                      PTX. Block scale registers are loaded per-quad (2 threads
                      per quad supply the scale register) per the PTX warp-level
                      block-scaling spec. NVFP4 additionally supports a separate
                      y_scale per-output-column factor.
Evidence:             mma.cuh:1126-1154 (mma_block_scaled_fp4 with both PTX
                      variants); mmq.cuh:48-58 (block_fp4_mmq struct +
                      static_asserts), 128-130 (use_native_fp4 dispatch in
                      ggml_cuda_mul_mat_q), 663-680 (util_funcs Blackwell
                      branch); mmq-vec-dot.cuh:1186-1250 (fp4_fp4_mma);
                      mmq-load-tiles.cuh:1542-1582 (mxfp4_fp4 loader), 1640-1679
                      (nvfp4_nvfp4 loader); mmq-config-blackwell.cuh:1-37 (32
                      CASE entries for MXFP4/NVFP4 with MMQ_ITER_K_FP4=512 and
                      GGML_CUDA_MMQ_SRAM_LAYOUT_FP4).
Architectural Impact: ~2× throughput on FP4 weights vs the Q8_1 dequant + int8
                      mma path used on Ampere. The rest of MMQ is unchanged —
                      only the load_tiles + vec_dot + config entries differ.
                      This is the cleanest forward-looking hardware support
                      pattern in MMQ.
Correctness Impact:   The FP4 Tensor Core does its own internal accumulation
                      (e2m1×e2m1→f32 with block scaling), which produces
                      different ULPs than the int8→int32→f32 path. The
                      y_scale factor for NVFP4 is applied in write_back, not
                      inside the mma instruction.
Optimization Type:    SIMD (native FP4 Tensor Core instruction).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Implement mma_block_scaled_fp4<type> via the
                      kind::mxf4.block_scale.scale_vec::2X and
                      kind::mxf4nvf4.block_scale.scale_vec::4X PTX. Use a
                      separate block_fp4_mmq activation layout and
                      quantize_mmq_fp4_cuda quantizer. Bypass Q8_1 dequant
                      entirely on Blackwell.
Priority:             High
Difficulty:           L
Dependencies:         ARTX10-F01, ARTX10-F02
Confidence:           High
```

### Finding ARTX10-F05

```
Finding ID:           ARTX10-F05
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            MMQ K-iteration pipeline
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             mul_mat_q_process_tile
Lines:                875-908
Summary:              MMQ does not use cp.async; each K-iteration serializes
                      global load → sync → vec_dot → sync → load → sync → vec_dot → sync.
Observation:          The K-loop body in mul_mat_q_process_tile issues four
                      __syncthreads() per iteration: after the first y-load,
                      after the first vec_dot, after the second y-load, and
                      after the second vec_dot. The x-load (load_tiles) and
                      y-load are synchronous global→shared copies. The cp.async
                      hardware (Ampere+) and the cp-async.cuh helper are
                      available but unused — grep for "cp.async" in mmq.cuh,
                      mmq-vec-dot.cuh, mmq-load-tiles.cuh returns zero matches.
                      Flash-attention uses cp.async (fattn-mma-f16.cuh:371-475);
                      MMQ does not.
Evidence:             mmq.cuh:887 (__syncthreads after first y-load), 891 (after
                      first vec_dot), 903 (after second y-load), 907 (after
                      second vec_dot); cp-async.cuh:22-46 (cp_async_cg_16 with
                      L2::256B/128B/64B hints, unused by MMQ); common.cuh:356-358
                      (cp_async_available, returns true on Ampere+).
Architectural Impact: MMQ has zero K-iteration pipeline depth. Global memory
                      latency is exposed on every iteration. A double-buffered
                      cp.async pipeline would overlap the next iteration's
                      global load with the current iteration's vec_dot compute,
                      hiding most of the global memory latency. This is likely
                      the single highest-impact optimization missing from MMQ.
Correctness Impact:   None. Synchronous loads are correct by definition. Adding
                      cp.async would require careful double-buffering of tile_x
                      and tile_y to avoid race conditions.
Optimization Type:    asynchronous execution / software prefetch (cp.async
                      double-buffered pipeline — currently absent).
GwenLand Target:      glcuda
Recommendation:       REJECT this absence. Add a double-buffered cp.async
                      pipeline (Ampere+) to overlap global loads with vec_dot
                      compute. Use cp.async.cg.shared.global.L2::256B for best
                      L2 behavior. Flash-attention already does this; MMQ
                      should too.
Priority:             High
Difficulty:           XL
Dependencies:         ARTX10-F01, ARTX10-F02
Confidence:           High
```

### Finding ARTX10-F06

```
Finding ID:           ARTX10-F06
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            MMQ shared-memory tile layout
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             block_q8_1_mmq (struct), ggml_cuda_mmq_get_sram_stride
Lines:                27-58, 131-158
Summary:              MMQ's block_q8_1_mmq packs 128 int8s + 16 bytes of
                      scale/padding per block; sram_stride is static_asserted
                      to be % 8 == 4 for bank-conflict avoidance.
Observation:          The block_q8_1_mmq struct holds 128 int8 quantized values
                      + a 16-byte union (d4[4] / ds4[4] / d2s6[8]) that serves
                      as both scale storage and shared-memory padding. The 128
                      values are organized as four 32-element block_q8_1 blocks,
                      interleaved so they can be copied to shared memory as one
                      contiguous 144-byte unit. The sram_stride (K-dim stride
                      per row in shared memory) is computed per sram_layout
                      (Q8_0, Q8_1, Q2_K, Q3_K, Q6_K, FP4, NVFP4) and is
                      static_asserted to be % 8 == 4 for every layout. The +4
                      padding (vs a multiple of 8) provides XOR-padding to
                      avoid 8-bank shared memory conflicts when accessing
                      consecutive rows.
Evidence:             mmq.cuh:27-46 (block_q8_1_mmq with comment "The y float
                      data is first grouped as blocks of 128 values. These
                      blocks are then treated as individual data values and
                      transposed. To avoid shared memory bank conflicts each
                      block is padded with 16 bytes. This padding is also used
                      to store block scales/partial sums."), 51-58 (block_fp4_mmq
                      with static_assert sizeof == block_q8_1_mmq), 131-158
                      (ggml_cuda_mmq_get_sram_stride with % 8 == 4
                      static_asserts for all 7 layouts).
Architectural Impact: Single contiguous copy from global to shared memory per
                      block (no per-element scatter). Bank-conflict-free
                      shared-memory access pattern. The 16-byte padding doubles
                      as scale storage, so it's not wasted space.
Correctness Impact:   None. The layout is purely a performance optimization;
                      the arithmetic is the same regardless of layout.
Optimization Type:    tiling / blocking / vectorization (bank-conflict-free
                      shared-memory layout).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Replicate the 128-int8 + 16-byte scale/sum layout
                      with the three mmq_q8_1_ds_layout variants. Enforce
                      sram_stride % 8 == 4 via static_assert.
Priority:             High
Difficulty:           M
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX10-F07

```
Finding ID:           ARTX10-F07
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MMQ dispatch predicate
Source File:          ggml/src/ggml-cuda/mmq.cu
Function:             ggml_cuda_should_use_mmq
Lines:                256-371
Summary:              should_use_mmq is a 115-line per-arch, per-type, per-batch
                      decision tree with hand-tuned thresholds and TODO comments.
Observation:          The function dispatches based on (type, cc, ne11,
                      n_experts). For Turing+ NVIDIA, MMQ always wins. For
                      pre-Turing NVIDIA, MMQ wins only if no FP16 MMA hardware
                      or ne11 < MMQ_DP4A_MAX_BATCH_SIZE (64). For CDNA3, MMQ
                      always wins due to "rocblas/tensile performs very poorly
                      on CDNA3 and hipblaslt [...] is currently suffering from
                      a crash" (with a TODO to revisit). For RDNA3, MMQ wins
                      for n_experts >= 64 always, otherwise per-type thresholds
                      apply (Q2_K ≤ 128, Q6_K ≤ 256, IQ2_XS/S ≤ 128, etc.).
                      For RDNA4, MMQ always wins. For Vega (gfx900), MMQ only
                      for MoE. The function reads environment variables
                      GGML_CUDA_FORCE_CUBLAS and GGML_CUDA_FORCE_MMQ to override.
Evidence:             mmq.cu:256-371 (full function), 316-318 (CDNA3 TODO
                      comment), 343-355 (RDNA3 per-type thresholds), 366-368
                      (Vega MoE-only).
Architectural Impact: The dispatch is opaque to the user — there's no way to
                      know which kernel was selected without instrumenting the
                      code. A single wrong threshold can regress performance by
                      10× with no compile-time warning. The CDNA3 threshold
                      will silently keep forcing MMQ even after the hipblaslt
                      bug is fixed.
Correctness Impact:   None. The predicate only selects between correct kernels.
Optimization Type:    None (this is a kernel-selection policy, not an
                      in-kernel optimization).
GwenLand Target:      glcuda, GATE
Recommendation:       ADAPT. Keep the per-arch structure but generate the
                      thresholds from a single source of truth (YAML/JSON).
                      Add per-arch test coverage to catch stale thresholds.
                      Document the reasoning behind each threshold.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX10-F08

```
Finding ID:           ARTX10-F08
Category:             ADOPT
Engine:               CUDA
Component:            MMQ J-template enumeration
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             mul_mat_q_switch_J, launch_mul_mat_q
Lines:                1443-1524 (switch_J), 1361-1441 (launch)
Summary:              MMQ enumerates J ∈ {8,16,24,...,128} at compile time and
                      picks J_best at runtime by minimizing output-column tile count.
Observation:          mul_mat_q_switch_J<type, fallback> iterates J from 8 to
                      128 (step 8), computes the config for each J, skips J
                      values whose shared-memory footprint exceeds smpbo, and
                      picks the smallest J that minimizes ntiles_x = ceil(ncols_max
                      / J). A 16-way switch (mmq.cuh:1470-1523) then dispatches
                      to launch_mul_mat_q<type, J, fallback>. The compiler
                      produces a separate kernel binary for each J, with the
                      J-axis fully unrolled inside the kernel body.
Evidence:             mmq.cuh:1443-1468 (J_best loop with smpbo check), 1470-1523
                      (16-way switch), 1361-1441 (launch_mul_mat_q template).
Architectural Impact: The compiler unrolls the J-axis (output-column tile
                      width) in the kernel body, holding per-J accumulators in
                      registers. The runtime picks the best J based on actual
                      ncols_dst, capped by the device's shared-memory limit.
                      Same pattern as ARTX09-R1 (ncols_dst enumeration) but on
                      the J-axis.
Correctness Impact:   None. The J-axis is purely a tiling parameter.
Optimization Type:    tiling / blocking (compile-time-enumerated J with runtime
                      selection).
GwenLand Target:      glcuda, GATE
Recommendation:       ADOPT. Enumerate J ∈ {8..128} step 8 at compile time;
                      pick J_best at runtime by minimizing tile count, capped
                      by smpbo. The 16-way switch is verbose but trivially
                      generated.
Priority:             High
Difficulty:           M
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX10-F09

```
Finding ID:           ARTX10-F09
Category:             ADOPT
Engine:               CUDA
Component:            MMQ MoE MUL_MAT_ID support
Source File:          ggml/src/ggml-cuda/mmq.cu
Function:             ggml_cuda_mul_mat_q (MoE branch)
Lines:                176-253
Summary:              MMQ supports MUL_MAT_ID via mm_ids_helper + expert_bounds +
                      optional dedup_bcast quantize-once-scatter.
Observation:          When ids != nullptr, MMQ launches mm_ids_helper to sort
                      and compact the expert routing on-device (no host sync).
                      The output is ids_dst (compacted column → original token
                      for output writes) and expert_bounds (per-expert range of
                      compacted indices). Each block reads its expert's col_low
                      and col_high from expert_bounds and skips entirely if
                      jt*J >= col_diff. When dedup_bcast = (ne11 == 1 &&
                      n_expert_used > 1) (gate/up activations broadcast across
                      experts), mm_ids_helper writes the inverse map ids_src1
                      (token slot → compacted row), and the activation
                      quantization uses quantize_scatter_mmq_*_cuda which
                      quantizes each token once and scatters to all its expert
                      slots, avoiding n_expert_used× duplicate quantization work.
Evidence:             mmq.cu:176-186 (MoE asserts + alloc), 190 (dedup_bcast
                      predicate), 197-200 (mm_ids_helper launch), 222-235
                      (quantize_scatter branch), 244-253 (MoE args construction).
Architectural Impact: On-device expert routing avoids host synchronization
                      (compatible with CUDA graph capture, ARTX08 §5.5). The
                      dedup_bcast optimization saves n_expert_used× quantization
                      work for the common MoE FFN pattern where gate and up
                      activations are broadcast across experts.
Correctness Impact:   None. The expert routing is a deterministic sort; the
                      output writes use ids_dst to map compacted columns back to
                      original (token, channel) positions.
Optimization Type:    kernel fusion (expert routing + activation quantization +
                      matmul in one op).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Implement mm_ids_helper + expert_bounds +
                      dedup_bcast quantize-once-scatter for MoE MUL_MAT_ID.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX10-F01, ARTX10-F02
Confidence:           High
```

### Finding ARTX10-F10

```
Finding ID:           ARTX10-F10
Category:             ADOPT
Engine:               CUDA
Component:            MMF hand-written Tensor-Core GEMM
Source File:          ggml/src/ggml-cuda/mmf.cuh, ggml/src/ggml-cuda/mma.cuh
Function:             mul_mat_f, mma (overloads)
Lines:                mmf.cuh:48-294 (kernel), 619-731 (auto-tune + dispatch); mma.cuh:1156-1218 (f16/bf16 mma)
Summary:              MMF is a hand-written mma.sync + ldmatrix kernel for
                      small-batch F32/F16/BF16 GEMM, avoiding cuBLAS per-call overhead.
Observation:          mul_mat_f<T, rows_per_block, cols_per_block, nwarps, has_ids>
                      uses tile<16, 8, T> (Turing+), tile<32, 4, T> (Volta), or
                      tile<16, 16, T, DATA_LAYOUT_I_MAJOR> (AMD WMMA/MFMA). Each
                      block computes a rows_per_block × cols_per_block output
                      tile. The K-dim loop tiles across col ∈ [0, ncols) in
                      strides of warp_size. Each iteration: load tile_A from x
                      via shared-memory tile_xy then load_ldmatrix; load tile_B
                      from y (cast to T via ggml_cuda_cast for F16/BF16); issue
                      mma(C, A, B). After the K loop, the per-thread C fragments
                      are written to buf_iw shared memory, transposed, and
                      summed across warps. Auto-tunes nwarps_best (1..8) by
                      minimizing niter. Template-enumerates cols_per_block ∈
                      {1..16} for compiler unrolling.
Evidence:             mmf.cuh:48-84 (tile type selection per arch), 86-95
                      (ntA/ntB), 127-225 (K-loop with ldmatrix + mma), 227-284
                      (output reduction via buf_iw), 619-655 (nwarps auto-tune),
                      668-728 (nwarps switch), 733-833 (cols_per_block switch);
                      mma.cuh:1156-1174 (m16n8k16.f32.f16.f16.f32 Ampere),
                      1167-1173 (Turing m8n8k8 fallback), 1183-1193 (bf16 mma).
Architectural Impact: For ne11 ∈ {1..16}, cuBLAS's per-call overhead (kernel
                      selection, workspace allocation) dominates. MMF's
                      hand-written kernel launches in <1 µs and uses the exact
                      same mma.sync instructions cuBLAS would, but with a
                      tighter, shape-specialized tile. The 16-way cols_per_block
                      enumeration gives the compiler full visibility into the
                      output-column count.
Correctness Impact:   None. mma.sync is deterministic per CC. The output
                      reduction via buf_iw uses fixed iteration order.
Optimization Type:    SIMD (Tensor Core mma.sync) / tiling (rows_per_block ×
                      cols_per_block × warp_size K-tile).
GwenLand Target:      glcuda
Recommendation:       ADOPT. For ne11 ∈ {1..16} F32/F16/BF16 GEMM, use
                      hand-written mma.sync + ldmatrix instead of cuBLAS.
                      Template on cols_per_block ∈ {1..16}. Auto-tune nwarps.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX10-F11

```
Finding ID:           ARTX10-F11
Category:             ADOPT
Engine:               CUDA
Component:            cuBLAS compute-type / output-dtype selection
Source File:          ggml/src/ggml-cuda/ggml-cuda.cu
Function:             ggml_cuda_mul_mat_cublas, ggml_cuda_mul_mat_cublas_impl
Lines:                1406-1660
Summary:              cuBLAS fallback auto-selects compute_type (F32/F16/BF16),
                      output dtype (F32 vs F16/BF16), and one of four dispatch
                      variants based on shape and contiguity.
Observation:          ggml_cuda_mul_mat_cublas picks compute_type based on
                      src0->type (F16 preferred for quantized src0 if
                      fast_fp16_hardware_available), the op_params[0] ==
                      GGML_PREC_F32 user override, and the
                      GGML_CUDA_CUBLAS_COMPUTE_TYPE env var.
                      ggml_cuda_mul_mat_cublas_impl picks cu_data_type (output
                      dtype): F32 if prefer_f32_output (true for F16 compute on
                      Volta/RDNA4/CDNA, true for BF16 compute on non-RDNA3
                      non-CDNA), else F16/BF16. If output is not F32, allocates
                      dst_temp and converts back via to_fp32_cuda. Dispatches
                      one of four cuBLAS variants: cublasSgemm (single F32
                      matrix), cublasGemmEx (single non-F32 matrix),
                      cublasGemmStridedBatchedEx (batched with strided
                      broadcast), cublasGemmBatchedEx (batched with general
                      pointer array, uses k_compute_batched_ptrs kernel to
                      populate the pointer arrays).
Evidence:             ggml-cuda.cu:1619-1660 (compute_type selection with env
                      var override), 1406-1467 (src0/src1 conversion), 1499-1528
                      (output dtype selection + dst_temp allocation), 1537-1610
                      (four cuBLAS dispatch variants), 1612-1616 (F16→F32
                      dequantize via to_fp32_cuda).
Architectural Impact: cuBLAS handles all large-batch GEMM. The auto-selection
                      avoids F16-accumulator overflow on hardware where F32
                      output is cheap, while preserving F16-output throughput
                      where it isn't. The four dispatch variants cover the
                      full shape matrix (single/batched × strided/general ×
                      F32/non-F32).
Correctness Impact:   F16/BF16 output with explicit dequantize to F32 is
                      bit-equivalent to direct F32 output for the same cuBLAS
                      kernel. TF32 (via CUBLAS_TF32_TENSOR_OP_MATH on the
                      handle) reduces F32 precision to 10-bit mantissa —
                      implicit, not user-visible (see ARTX10-W10).
Optimization Type:    None (this is kernel selection, not in-kernel
                      optimization).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Keep the auto-selection heuristic but make TF32
                      explicit (opt-in or opt-out flag), not implicit via
                      CUBLAS_TF32_TENSOR_OP_MATH default.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX10-F12

```
Finding ID:           ARTX10-F12
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            Hopper / Blackwell Tensor Core instructions
Source File:          ggml/src/ggml-cuda/mma.cuh, ggml/src/ggml-cuda/mmq.cuh, ggml/src/ggml-cuda/mmf.cuh
Function:             (whole file)
Lines:                mma.cuh:1-1456; mmq.cuh:1-1571; mmf.cuh:1-909
Summary:              MMQ/MMF use only Ampere-style mma.sync; no Hopper wgmma,
                      TMA, or Blackwell tc_gen5.mma (except for FP4 block-scaled).
Observation:          Grep for "wgmma", "cuTensorMap", "cp.async.bulk",
                      "tc_gen5" in mma.cuh, mmq.cuh, mmf.cuh returns zero
                      matches. GGML_CUDA_CC_HOPPER = 900 is defined (common.cuh:55)
                      but only PDL (ggml_cuda_pdl_sync, ARTX08 §9.4) is used on
                      Hopper. The MMQ kernel uses mma.sync.aligned.m16n8k16
                      (Turing) and mma.sync.aligned.m16n8k32 (Ampere+) — no
                      wgmma.mma_async.sync.aligned.m64n*k16 (Hopper, 4× throughput
                      per warp group). The MMF kernel similarly uses mma.sync
                      only. On Blackwell, only mma_block_scaled_fp4 (FP4) uses
                      the new instruction; F16/BF16/F32 GEMM still uses Ampere
                      mma.sync. No TMA (cp.async.bulk.tensor via
                      cuTensorMapEncodeTiled) for bulk shared-memory loads.
Evidence:             Grep results: 0 matches for wgmma|cuTensorMap|cp.async.bulk|tc_gen5
                      in mma.cuh, mmq.cuh, mmf.cuh. mma.cuh:920-1220 (all mma.sync
                      variants, no wgmma). common.cuh:55 (GGML_CUDA_CC_HOPPER defined
                      but unused except for PDL gating).
Architectural Impact: Hopper's wgmma gives 4× Tensor Core throughput per warp
                      group vs Ampere's mma.sync. Hopper's TMA gives asynchronous
                      bulk shared-memory loads without occupying registers.
                      Blackwell's tc_gen5.mma gives further throughput gains.
                      MMQ on Hopper uses the Ampere codepath, leaving 4× Tensor
                      Core throughput on the table. The Blackwell FP4 path
                      (ARTX10-F04) is the only forward-looking instruction
                      usage; F16/BF16/F32 GEMM on Blackwell uses Ampere
                      instructions.
Correctness Impact:   None. mma.sync is correct on Hopper/Blackwell; it's just
                      suboptimal.
Optimization Type:    None (this is the absence of an optimization).
GwenLand Target:      glcuda
Recommendation:       REJECT this absence. Add a Hopper codepath using
                      wgmma.mma_async (4× Tensor Core throughput) and TMA
                      (cp.async.bulk.tensor via cuTensorMapEncodeTiled) for bulk
                      shared-memory loads. Add a Blackwell tc_gen5.mma codepath
                      for F16/BF16/F32 GEMM (not just FP4).
Priority:             High
Difficulty:           XL
Dependencies:         ARTX10-F01, ARTX10-F02, ARTX10-F05
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the stream-K fixup kernel's serial read loop
  (`mmq.cuh:1246-1274`) is a measurable bottleneck for shapes where
  stream-K is selected and `gridDim.x` is large. Requires runtime
  profiling of fixup-vs-main-kernel time ratio. Static analysis shows
  the loop is `O(gridDim.x)` per output element, but the constant factor
  is small (one shared-memory read per iteration).
* **U2**. Whether the absence of `cp.async` pipelining (ARTX10-F05) is
  intentional (e.g., because the dp4a path's per-thread shared-memory
  indexing makes double-buffering hard) or simply unimplemented.
  Requires git archaeology on the MMQ kernel history. Flash-attention's
  `cp.async` usage shows the pattern is known to the team.
* **U3**. Whether the per-arch config tables (ARTX10-F01, W4) are
  auto-generated or hand-maintained. The `template-instances/generate_cu_files.py`
  script exists for MMF/MMVQ/fattn template instantiation, but no
  equivalent is visible for `mmq-config-*.cuh`. Requires inspecting the
  build system.
* **U4**. Whether the CDNA3 `should_use_mmq` always-true override
  (`mmq.cu:319-321`, with TODO) is still needed at the audited commit.
  The comment cites a hipblaslt crash "currently suffering from a crash
  on this architecture" — requires checking if the upstream hipblaslt
  bug has been fixed since the comment was written.
* **U5**. Whether the `mul_mat_q_process_tile` two-y-loads-per-iteration
  pattern (W5) is fundamental to the tile_y sizing or could be
  eliminated with a larger shared-memory budget. Requires measuring the
  shared-memory utilization at the current `J` values and comparing to
  `smpbo`.
* **U6**. Whether the cuBLAS TF32 default (W10) causes measurable
  precision regressions for F32 GEMM in actual llama.cpp workloads.
  The 10-bit mantissa (vs F32's 23-bit) is a 13-bit precision loss;
  for transformer inference this is usually fine, but for training or
  numerical-research workloads it could matter.
* **U7**. Whether the MMF `should_use_mmf` CDNA1/CDNA2 F16/BF16 rejection
  (W8, `mmf.cu:171-175` with TODO) is still suboptimal. Requires
  benchmarking MMF vs hipBLAS on MI100/MI210 for F16/BF16 GEMM with
  `ne11 ∈ {1..16}`.
* **U8**. Whether the Blackwell FP4 native path (ARTX10-F04) produces
  bit-identical results to the Q8_1 path for the same input. The FP4
  Tensor Core does its own internal accumulation; the Q8_1 path uses
  int8→int32→f32. Same input → different reduction tree → different
  ULPs. Requires executing both paths on the same input on Blackwell
  hardware.
* **U9**. Whether the stream-K fixup kernel's `atomicAdd` to `dst`
  (`mmq.cuh:1309`) is a performance bottleneck for shapes with many
  partial tiles. `atomicAdd` on F32 is relatively fast on Ampere+ (hardware
  FP32 atomic), but for large grids the contention could matter.
  Requires profiling.
* **U10**. Whether the `__launch_bounds__(nthreads, 1)` (Ampere/Blackwell/
  CDNA) actually achieves the target occupancy, or whether register
  pressure spills to local memory. Requires `cuobjdump -res-usage` on
  the compiled MMQ kernels per (type, J, arch) combination. Static
  analysis cannot determine register usage.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-cuda/mmq.cu`                         | `ggml_cuda_mul_mat_q_switch_type`              | 8–80          |
| R02       | `ggml/src/ggml-cuda/mmq.cu`                         | `ggml_cuda_mul_mat_q`                          | 82–254        |
| R03       | `ggml/src/ggml-cuda/mmq.cu`                         | `ggml_cuda_should_use_mmq`                     | 256–371       |
| R04       | `ggml/src/ggml-cuda/mmq.cuh`                        | `block_q8_1_mmq` (struct)                      | 27–46         |
| R05       | `ggml/src/ggml-cuda/mmq.cuh`                        | `block_fp4_mmq` (struct)                       | 51–54         |
| R06       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mmq_get_q8_1_ds_layout`                       | 60–100        |
| R07       | `ggml/src/ggml-cuda/mmq.cuh`                        | `ggml_cuda_mmq_get_sram_stride`               | 131–158       |
| R08       | `ggml/src/ggml-cuda/mmq.cuh`                        | `ggml_cuda_mmq_config` (struct)                | 164–203       |
| R09       | `ggml/src/ggml-cuda/mmq.cuh`                        | `CASE` macro + per-arch config includes        | 205–223       |
| R10       | `ggml/src/ggml-cuda/mmq.cuh`                        | `ggml_cuda_mmq_get_config` (host)              | 225–242       |
| R11       | `ggml/src/ggml-cuda/mmq.cuh`                        | `ggml_cuda_mmq_get_config` (device constexpr)  | 244–263       |
| R12       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mmq_get_dp4a_tile_x_sizes`                   | 362–398       |
| R13       | `ggml/src/ggml-cuda/mmq.cuh`                        | `ggml_cuda_mmq_get_util_funcs`                 | 521–817       |
| R14       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mul_mat_q_process_tile`                       | 842–915       |
| R15       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mul_mat_q` (kernel)                           | 920–1205      |
| R16       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mul_mat_q_stream_k_fixup`                     | 1207–1343     |
| R17       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mmq_args` (struct), `mmq_get_nbytes_shared`  | 1345–1359     |
| R18       | `ggml/src/ggml-cuda/mmq.cuh`                        | `launch_mul_mat_q`                             | 1361–1441     |
| R19       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mul_mat_q_switch_J`                           | 1443–1524     |
| R20       | `ggml/src/ggml-cuda/mmq.cuh`                        | `mul_mat_q_case`                               | 1526–1535     |
| R21       | `ggml/src/ggml-cuda/mmq-config-ampere.cuh`         | `ggml_cuda_mmq_get_config_ampere`              | 1–366         |
| R22       | `ggml/src/ggml-cuda/mmq-config-blackwell.cuh`      | `ggml_cuda_mmq_get_config_blackwell`           | 1–37          |
| R23       | `ggml/src/ggml-cuda/mmq-config-pascal.cuh`         | `ggml_cuda_mmq_get_config_pascal`              | 1–261         |
| R24       | `ggml/src/ggml-cuda/mmq-config-cdna.cuh`           | `ggml_cuda_mmq_get_config_cdna`                | 1–177         |
| R25       | `ggml/src/ggml-cuda/mmq-config-rdna2.cuh`          | `ggml_cuda_mmq_get_config_rdna2`               | 1–261         |
| R26       | `ggml/src/ggml-cuda/mmq-config-rdna4.cuh`          | `ggml_cuda_mmq_get_config_rdna4`               | 1–282         |
| R27       | `ggml/src/ggml-cuda/mmq-vec-dot.cuh`                | `ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a`         | 10–58         |
| R28       | `ggml/src/ggml-cuda/mmq-vec-dot.cuh`                | `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`          | 142–280       |
| R29       | `ggml/src/ggml-cuda/mmq-vec-dot.cuh`                | `ggml_cuda_mmq_vec_dot_fp4_fp4_mma`            | 1186–1250     |
| R30       | `ggml/src/ggml-cuda/mmq-load-tiles.cuh`             | `ggml_cuda_mmq_load_tiles_q1_0`                | 7–96          |
| R31       | `ggml/src/ggml-cuda/mmq-load-tiles.cuh`             | `ggml_cuda_mmq_load_tiles_mxfp4_fp4`           | 1542–1582     |
| R32       | `ggml/src/ggml-cuda/mmq-load-tiles.cuh`             | `ggml_cuda_mmq_load_tiles_nvfp4_nvfp4`         | 1640–1679     |
| R33       | `ggml/src/ggml-cuda/mmf.cu`                         | `ggml_cuda_mul_mat_f`                          | 13–131        |
| R34       | `ggml/src/ggml-cuda/mmf.cu`                         | `ggml_cuda_should_use_mmf`                     | 133–191       |
| R35       | `ggml/src/ggml-cuda/mmf.cuh`                        | `mul_mat_f` (kernel)                           | 48–294        |
| R36       | `ggml/src/ggml-cuda/mmf.cuh`                        | `mul_mat_f_ids` (kernel)                       | 299–570       |
| R37       | `ggml/src/ggml-cuda/mmf.cuh`                        | `mul_mat_f_cuda` (auto-tune + dispatch)        | 619–731       |
| R38       | `ggml/src/ggml-cuda/mma.cuh`                        | `tile<I,J,T,dl>` template                      | 98–620        |
| R39       | `ggml/src/ggml-cuda/mma.cuh`                        | `mma` (s8×s8→s32 overloads)                    | 920–960       |
| R40       | `ggml/src/ggml-cuda/mma.cuh`                        | `mma` (f16×f16→f32 / bf16×bf16→f32)            | 1156–1193     |
| R41       | `ggml/src/ggml-cuda/mma.cuh`                        | `mma_block_scaled_fp4`                         | 1126–1154     |
| R42       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q4_0_q8_1_impl`, `VDR_*_MMQ`          | 109–134       |
| R43       | `ggml/src/ggml-cuda/common.cuh`                     | `ggml_cuda_dp4a`                               | 703–741       |
| R44       | `ggml/src/ggml-cuda/common.cuh`                     | `cp_async_available`, `blackwell_mma_available`| 356–363       |
| R45       | `ggml/src/ggml-cuda/common.cuh`                     | `MATRIX_ROW_PADDING = 512`                     | 176           |
| R46       | `ggml/src/ggml-cuda/common.cuh`                     | `cublas_handle` (TF32 default)                 | 1490–1497     |
| R47       | `ggml/src/ggml-cuda/cp-async.cuh`                   | `cp_async_cg_16`, `cp_async_wait_all`          | 22–57         |
| R48       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_mul_mat` (router)                   | 1812–1852     |
| R49       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_mul_mat_cublas`                     | 1619–1660     |
| R50       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_mul_mat_cublas_impl`                | 1406–1617     |
| R51       | `ggml/src/ggml-cuda/quantize.cuh`                   | `quantize_mmq_q8_1_cuda`, `quantize_mmq_fp4_cuda` declarations | 24–43 |
| R52       | `ggml/src/ggml-cuda/mmid.cu`                        | `ggml_cuda_launch_mm_ids_helper`               | (ARTX09 §5.7) |
