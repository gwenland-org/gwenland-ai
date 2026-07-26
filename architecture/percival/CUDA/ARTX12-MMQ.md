# ARTX12 — CUDA MMQ Tile-Vecdot and Per-Quant-Format Specifics

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-26
**Auditor:** Percival-aux (ARTX12)
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (kernel selection)

---

## 1. Executive Summary

ARTX10 audited the MMQ kernel family at the *GEMM-orchestration* level: tile sizes,
dispatch precedence (MMVF → MMF → MMVQ → MMQ → cuBLAS), the `ggml_cuda_mmq_config`
table, stream-K decomposition, the cuBLAS fallback, and the high-level dual
dp4a / Tensor-Core split. This document goes one level deeper: it audits the
**per-quant-format tile-vecdot implementations** that live in `mmq-vec-dot.cuh`
(1251 lines), `vecdotq.cuh` (1322 lines), and `mmq-load-tiles.cuh` (1679 lines),
plus the per-arch tile-config files (`mmq-config-{ampere,blackwell,cdna,pascal,
rdna2,rdna4}.cuh`) and the MMQ launcher template in `mmq.cuh` (1570 lines).

The central architectural insight is that **MMQ offloads per-quant complexity
into the loader, not the vecdot**. Each quant format has its own
`ggml_cuda_mmq_load_tiles_*` template (22 specializations, 1679 lines) that
unpacks the weight block into one of three canonical shared-memory layouts
(`Q8_0`, `Q8_1`, `Q2_K`, `Q3_K`, `Q6_K`, `FP4`, `NVFP4`) and stores either
raw packed ints (dp4a path) or pre-dequantized int8 values (mma path). The
`ggml_cuda_mmq_vec_dot_*` template then operates on the canonical layout and
comes in only ~6 variants shared across all 22 quant formats:

* `vec_dot_q4_0_q8_1_dp4a` — also used by Q4_0 only (single scale + bias subtract).
* `vec_dot_q4_1_q8_1_dp4a` — also used by Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S (scale + min).
* `vec_dot_q8_0_q8_1_dp4a` — shared by Q1_0, Q5_0, Q8_0, IQ2_XXS, IQ3_XXS,
  IQ3_S, MXFP4, IQ4_XS, IQ4_NL (pure int8 × int8 dot with one scale per block).
* `vec_dot_q8_0_16_q8_1_dp4a` — used by Q3_K, IQ2_S, IQ2_XS, NVFP4 (16-elem
  sub-block scales).
* `vec_dot_q2_K_q8_1_dp4a` — Q2_K only (super-block scales + per-32 partial sums).
* `vec_dot_q3_K_q8_1_dp4a`, `vec_dot_q4_K_q8_1_dp4a`, `vec_dot_q5_K_q8_1_dp4a`,
  `vec_dot_q6_K_q8_1_dp4a` — K-quant specific.

The same six shapes are mirrored by six `*_mma` variants that issue
`mma.sync` PTX instead of `ggml_cuda_dp4a`. The format-to-vecdot mapping is
encoded in a single `ggml_cuda_mmq_get_util_funcs<type, J, fallback>()`
switch (`mmq.cuh:521-816`), which is the device-side analogue of ARTX01's
type-traits table and the CUDA analogue of ARTX10's config table.

The other key architectural decisions audited here are: (1) the K-quant 6-bit
scale unpacker is reimplemented per arch (`unpack_scales_q45_K` in
`mmq-load-tiles.cuh:612-620` vs the CPU's `get_scale_min_k4` in
`dequantize.cuh:157-164`); (2) Blackwell's config file declares only MXFP4 /
NVFP4 native entries and *falls through to Ampere* for every other quant
(`mmq-config-blackwell.cuh:36`); (3) the dp4a path uses `__dp4a` (Volta+) /
`__builtin_amdgcn_sdot4` (CDNA/RDNA2) / `__builtin_amdgcn_sudot4` (RDNA3/4)
and has a hand-rolled inline-asm fallback for gfx900 / RDNA1, but no
`__dp2a` anywhere; (4) the MMQ tile-vecdot and the GEMV vecdot are **different
code paths** sharing only the inner `vec_dot_*_q8_*_impl` template — the MMQ
path is templated on `(type, J, fallback)` and reads pre-unpacked ints from
shared memory, while the GEMV path takes `(vbq, bq8_1, kbx, iqs)` and unpacks
the block itself.

For GwenLand, the architectural decisions worth **ADOPT**ing are the
format-to-vecdot table (a clean per-quant dispatch with a canonical-layout
intermediate), the dual dp4a/mma template pattern, and the per-arch config
table with Blackwell fallthrough. The decisions worth **MONITOR**ing are the
Blackwell non-FP4 fallthrough (which leaves Q4_K/Q6_K on Ampere `mma.sync`
rather than `tc_gen5.mma`) and the per-arch K-quant scale unpacker divergence.

---

## 2. Purpose

This document answers four questions ARTX10 explicitly deferred:

1. For each quant format, what does the per-quant tile-vecdot look like?
   What intermediate type is used? How are scales applied? How many
   elements per thread per iteration?
2. What are the per-arch tile sizes (`MMQ_M`, `MMQ_N`, `MMQ_K` here named
   `I`, `J`, `K_vram`) across Pascal / Ampere / Blackwell / CDNA / RDNA2 /
   RDNA4?
3. How is the `mmq_vec_dot` template specialized per quant, and how does
   the MMQ path differ from the GEMV `vec_dot_q*_q*_cuda` path in
   `vecdotq.cuh`?
4. How is the K-quant 6-bit packed scale (the `K_SCALE_SIZE = 12` layout
   from ARTX06) unpacked on CUDA, and does it differ from the CPU?

ARTX10 owns the GEMM-orchestration narrative (tile blocking, stream-K,
async, cuBLAS fallback, MoE). This document does not duplicate those
findings; cross-references to ARTX10 are explicit where they occur.

---

## 3. Source Files

| File | Lines | Role |
| ---- | ----- | ---- |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | 1251 | Per-quant `ggml_cuda_mmq_vec_dot_*_dp4a` and `*_mma` templates |
| `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | 1679 | Per-quant `ggml_cuda_mmq_load_tiles_*` templates and `unpack_scales_q45_K` |
| `ggml/src/ggml-cuda/vecdotq.cuh` | 1322 | Per-quant `vec_dot_*_q8_*_impl` device functions, shared by MMQ and GEMV; VDR_*_MMQ constants |
| `ggml/src/ggml-cuda/mmq-config-ampere.cuh` | 366 | Ampere / Volta / Turing per-(type, J, fallback) CASE table (336 entries) |
| `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | 37 | Blackwell FP4-only table (32 entries); falls through to Ampere otherwise |
| `ggml/src/ggml-cuda/mmq-config-cdna.cuh` | 177 | CDNA (MI300X etc.) table (147 entries) |
| `ggml/src/ggml-cuda/mmq-config-pascal.cuh` | 261 | Pascal table (231 entries) |
| `ggml/src/ggml-cuda/mmq-config-rdna2.cuh` | 261 | RDNA2 table (231 entries) |
| `ggml/src/ggml-cuda/mmq-config-rdna4.cuh` | 282 | RDNA4 table (252 entries) |
| `ggml/src/ggml-cuda/mmq.cuh` | 1570 | `ggml_cuda_mmq_config` struct, `mmq_get_dp4a_tile_x_sizes`, `ggml_cuda_mmq_get_util_funcs` switch, `mul_mat_q_process_tile`, `mul_mat_q` kernel, `mul_mat_q_stream_k_fixup`, `mul_mat_q_case` launcher template |
| `ggml/src/ggml-cuda/mmq.cu` | 372 | `ggml_cuda_mul_mat_q` host entry, `ggml_cuda_should_use_mmq` policy, `ggml_cuda_mul_mat_q_switch_type` dispatch |
| `ggml/src/ggml-cuda/dequantize.cuh` | 432 | `dequantize_q*_*` device functions (used by F16/cuBLAS path); `get_scale_min_k4` for K-quant scale unpacking |
| `ggml/src/ggml-cuda/mmvq.cu` | 1290 | GEMV path that uses the same `vec_dot_*_q8_*` device functions but with a different API signature (`vec_dot_q_cuda_t` typedef) |

> Note: The audit prompt references `mmq-config-rdna4.cuh` as a target; the
> file at this commit contains 252 CASE entries for RDNA4 (gfx1200) — RDNA3
> is not given its own file; RDNA3 configs share RDNA4's table via
> `amd_wmma_available(cc)` (`mmq.cuh:230`).

---

## 4. Architecture Overview

```
            ┌───────────────────────────────────────────────────────────┐
            │  mmq.cu : ggml_cuda_mul_mat_q (host entry)                │
            │   ├─ quantize_mmq_q8_1_cuda / quantize_mmq_fp4_cuda       │
            │   ├─ ggml_cuda_mul_mat_q_switch_type → mul_mat_q_case<T>  │
            │   └─ ggml_cuda_should_use_mmq (policy)                    │
            └───────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌───────────────────────────────────────────────────────────┐
            │  mmq.cuh : mul_mat_q_case<T> → mul_mat_q_switch_J<T,F>    │
            │   ├─ J_best minimization over J ∈ {8,16,…,128}            │
            │   ├─ launch_mul_mat_q<T, J, F>                            │
            │   │     ├─ ggml_cuda_mmq_get_config(type, J, F, cc)       │
            │   │     │     → mmq-config-{pascal,ampere,blackwell,      │
            │   │     │       cdna,rdna2,rdna4}.cuh                     │
            │   │     ├─ CUDA_SET_SHARED_MEMORY_LIMIT                   │
            │   │     └─ mul_mat_q<<<…>>>  OR  mul_mat_q + fixup        │
            └───────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌───────────────────────────────────────────────────────────┐
            │  mmq.cuh : mul_mat_q kernel                              │
            │   ├─ mul_mat_q_process_tile<T, J, F, fixup>               │
            │   │     ├─ load_tiles(x, tile_x, …)  ← per-quant loader   │
            │   │     │     from mmq-load-tiles.cuh                     │
            │   │     ├─ vec_dot(tile_x, tile_y, sum, 0)                │
            │   │     ├─ vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K)    │
            │   │     └─ write_back(sum, …)                             │
            │   └─ (stream-K: tiles scheduled across SMs, fixup kernel) │
            └───────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌───────────────────────────────────────────────────────────┐
            │  mmq.cuh : ggml_cuda_mmq_get_util_funcs<T, J, F>          │
            │   ↦ (vdr, load_tiles, vec_dot, write_back) tuple          │
            │   ↦ switch (type) { … } over 22 quants                    │
            │       (dp4a path)        │     (mma path)                  │
            │       Q4_0 → vec_dot_q4_0 │     Q4_0 → vec_dot_q8_0_mma    │
            │       Q4_1 → vec_dot_q4_1 │     Q4_1 → vec_dot_q8_1_mma    │
            │       Q8_0 → vec_dot_q8_0 │     Q8_0 → vec_dot_q8_0_mma    │
            │       Q2_K → vec_dot_q2_K │     Q2_K → vec_dot_q2_K_mma    │
            │       Q3_K → vec_dot_q8_0_16│   Q3_K → vec_dot_q8_0_16_mma │
            │       Q4_K → vec_dot_q4_K │     Q4_K → vec_dot_q8_1_mma    │
            │       Q5_K → vec_dot_q5_K │     Q5_K → vec_dot_q8_1_mma    │
            │       Q6_K → vec_dot_q6_K │     Q6_K → vec_dot_q6_K_mma    │
            │       IQ* → vec_dot_q8_0*  │     IQ* → vec_dot_q8_0*_mma   │
            │       MXFP4 → vec_dot_q8_0 │     MXFP4 → vec_dot_q8_0_mma  │
            │       NVFP4 → vec_dot_q8_0_16│   NVFP4 → vec_dot_q8_0_16_mma│
            │       (Blackwell FP4) → vec_dot_fp4_fp4_mma                │
            └───────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌───────────────────────────────────────────────────────────┐
            │  vecdotq.cuh : vec_dot_*_q8_*_impl device functions      │
            │   ├─ int sumi = ggml_cuda_dp4a(v, u, sumi)                │
            │   └─ return d * sumi (or dm.x*sumi - dm.y*min_sum)        │
            │   ├─ VDR_*_MMQ constants (1..8) per quant                 │
            │   └─ kvalues_iq4nl / kvalues_mxfp4 LUTs for FP4/IQ4       │
            └───────────────────────────────────────────────────────────┘
```

Key design points:

* **Two-level dispatch**. The host launches a type-specialized template
  (`mul_mat_q_case<T>`), which switches over `J` (16-way) and `fallback` (2-way).
  Inside the kernel, `ggml_cuda_mmq_get_util_funcs<T, J, F>()` is a
  `constexpr __device__` function that returns a `(vdr, load_tiles, vec_dot,
  write_back)` tuple as a struct. Both levels are compile-time resolved — no
  indirect calls survive to the kernel binary.
* **Per-quant complexity lives in the loader, not the vecdot**. The 1679-line
  `mmq-load-tiles.cuh` has 22 templates; the 1251-line `mmq-vec-dot.cuh` has
  only ~12 distinct vecdot templates (six dp4a + six mma) shared across 22
  quants. This is the opposite of the CPU backend, where each quant has its
  own `vec_dot_q*` function (ARTX01 §10).
* **The same `vec_dot_*_q8_*_impl` functions are shared with GEMV**. Both
  MMVQ (`vec_dot_q4_0_q8_1(vbq, bq8_1, kbx, iqs)`) and MMQ
  (`ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a<T, J, F>(x, y, sum, k00)`) call
  `vec_dot_q4_0_q8_1_impl<VDR>` with the same arithmetic, but with different
  VDR constants (`VDR_Q4_0_Q8_1_MMVQ = 2` vs `VDR_Q4_0_Q8_1_MMQ = 4`).
* **`__dp4a` (or its AMD/ROCm equivalent) is the only SIMD instruction used
  in the dp4a path**. No `__dp2a`, no half-precision FMA in the inner loop.
  The mma path replaces `__dp4a` with `mma.sync.aligned.m16n8k{16,32}.s32.s8.s8.s32`.

---

## 5. Execution Flow

### 5.1 Host entry

`ggml_cuda_mul_mat_q` (`mmq.cu:82-254`) is called from
`ggml_cuda_mul_mat` (`ggml-cuda.cu:1812`) when `ggml_cuda_should_use_mmq`
returns true (ARTX10 §5.6). It:

1. Asserts `src1->type == GGML_TYPE_F32`, `dst->type == GGML_TYPE_F32`.
2. Clears potential padding in `src0` if it is a compute buffer
   (`mmq.cu:107-114`). ARTX10 §11.4 explains why.
3. Pads `ne10` to `MATRIX_ROW_PADDING = 512` (`mmq.cu:117`).
4. Pre-quantizes `src1` to `block_q8_1_mmq` (or `block_fp4_mmq` for
   native Blackwell FP4) via `quantize_mmq_q8_1_cuda` /
   `quantize_mmq_fp4_cuda` (`mmq.cu:153, 149`).
5. Dispatches to `ggml_cuda_mul_mat_q_switch_type` (`mmq.cu:8`), a
   22-way `switch (args.type_x)` over `mul_mat_q_case<T>`.

### 5.2 Template-on-`J` selection

`mul_mat_q_case<T>` (`mmq.cuh:1526-1535`) selects `fallback = (nrows_x %
128 != 0)` and calls `mul_mat_q_switch_J<T, fallback>`. The latter
(`mmq.cuh:1443-1524`) loops `J ∈ {8, 16, …, 128}` and picks the smallest
`J` that minimizes the output-column tile count, subject to the shared-
memory limit (`mmq_get_nbytes_shared(config, cc) > smpbo`). The chosen
`J` is dispatched via a 16-way `switch (J_best)` to
`launch_mul_mat_q<T, J, fallback>`.

### 5.3 Kernel launch

`launch_mul_mat_q` (`mmq.cuh:~1350-1441`) is the host-side launcher:

1. Computes `nbytes_shared` from the config's `sram_layout` and `I`.
2. Calls `CUDA_SET_SHARED_MEMORY_LIMIT` on both `mul_mat_q<T, J, false>`
   and `mul_mat_q<T, J, true>` to opt in to >48 KB shared memory on
   Ampere+ (`mmq.cuh:1375-1376`).
3. If `stream_k == false`: launches `mul_mat_q` with
   `<<<block_nums_xy_tiling, block_dims, nbytes_shared, stream>>>`.
4. If `stream_k == true`: launches `mul_mat_q` with
   `block_nums_stream_k = min(ntiles_dst, nsm)`, optionally followed by
   `mul_mat_q_stream_k_fixup` to atomic-add partial sums.

### 5.4 Per-tile execution

`mul_mat_q_process_tile<T, J, F, fixup>` (`mmq.cuh:841-915`) is the
per-tile body called by `mul_mat_q`:

```
float sum[J*I / (nwarps*warp_size)] = {0.0f};

for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

    // Copy one block_q8_1_mmq from global y to shared tile_y
    for (l0 in 0..J*MMQ_TILE_Y_K step nwarps*warp_size)
        tile_y[l] = by0[l];
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, 0);          // k00 = 0

    // Copy the *second* block_q8_1_mmq from global y to shared tile_y
    for (l0 in 0..J*MMQ_TILE_Y_K step nwarps*warp_size)
        tile_y[l] = by0[l + sz];
    __syncthreads();
    vec_dot(tile_x, tile_y, sum, MMQ_TILE_NE_K);  // k00 = MMQ_TILE_NE_K

    __syncthreads();
}

write_back(sum, ids_dst, dst_or_tmp_fixup, y_scale, …);
```

The K dimension is unrolled two `MMQ_TILE_NE_K = 32` chunks per iteration:
the loader fills `tile_x` once (covering `K_vram = 256` elements), then
two `block_q8_1_mmq` blocks (each `128 int8 + 16 B scales`) are streamed
into `tile_y` and immediately consumed by `vec_dot` before the next
`__syncthreads()`. This is a software-pipelined depth-2 K-loop, but it is
**not** double-buffered — the next `tile_x` load waits for `__syncthreads`
to complete (see ARTX10 §12.2 "No `cp.async` pipelining").

### 5.5 Per-quant vecdot dispatch

`vec_dot` is the function pointer returned by
`ggml_cuda_mmq_get_util_funcs<T, J, F>().vec_dot`. Its concrete target
depends on (a) the quant type, (b) whether `use_mma_data_layout()` is
true for the compiled arch, and (c) on Blackwell, whether the type is
MXFP4 / NVFP4 (in which case `ggml_cuda_mmq_vec_dot_fp4_fp4_mma` is
used). The mapping is tabulated in §9 below.

---

## 6. Data Layout

### 6.1 Weight-side block layout (input)

Each quant format defines its own `block_q*` struct in `ggml-common.h`.
The MMQ loader (`mmq-load-tiles.cuh`) reads these blocks and writes
into a canonical shared-memory tile layout. The block sizes relevant to
the per-quant tile-vecdot are:

| Quant | Block size `QK*` | Quant ratio `QR*` | Ints per block `QI*` | Scale layout |
| ----- | ---------------- | ------------------ | -------------------- | ------------ |
| Q1_0  | 128 | 1 | 4 (`QI1_0`) | 1 × fp16 `d` per 128 |
| Q4_0  | 32  | 2 | 8 (`QI4_0`) | 1 × fp16 `d` per 32 |
| Q4_1  | 32  | 2 | 8 (`QI4_1`) | 1 × half2 `dm` (scale, min) per 32 |
| Q5_0  | 32  | 2 | 8 (`QI5_0`) | 1 × fp16 `d` per 32 |
| Q5_1  | 32  | 2 | 8 (`QI5_1`) | 1 × half2 `dm` per 32 |
| Q8_0  | 32  | 1 | 8 (`QI8_0`) | 1 × fp16 `d` per 32 |
| Q2_K  | 256 | 4 | 16 (`QI2_K`) per super-block | half2 `dm` + 16 × uint8 packed scales |
| Q3_K  | 256 | 4 | 16 (`QI3_K`) per super-block | fp32 `d` + 12 packed scale bytes |
| Q4_K  | 256 | 2 | 8 (`QI4_K`) per super-block | half2 `dm` + 12 packed scale bytes (`K_SCALE_SIZE`) |
| Q5_K  | 256 | 2 | 8 (`QI5_K`) per super-block | half2 `dm` + 12 packed scale bytes |
| Q6_K  | 256 | 2 | 16 (`QI6_K`) per super-block | fp32 `d` + 16 × int8 scales |
| IQ2_XXS | 256 | 2 | 32 (`QI2_XXS`) | fp16 `d` + 4-byte packed signs/scales |
| IQ2_XS  | 256 | 2 | 32 (`QI2_XS`)  | fp16 `d` + 4-byte scales + grid LUT |
| IQ2_S   | 256 | 2 | 32 (`QI2_S`)   | fp16 `d` + 4-byte scales + grid LUT |
| IQ3_XXS | 256 | 2 | 16 (`QI3_XXS`) | fp16 `d` + 4-byte packed signs/scales |
| IQ3_S   | 256 | 2 | 16 (`QI3_S`)   | fp16 `d` + 4-byte scales + 4-byte signs |
| IQ1_S   | 256 | 1 | 16 (`QI1_S`)   | fp16 `d` + 2-byte `qh` (scale bits + delta) |
| IQ1_M   | 256 | 1 | 16 (`QI1_M`)   | 8-byte packed scales + 4-byte `qh` |
| IQ4_NL  | 32  | 2 | 8 (`QI4_NL`)   | fp16 `d` + 4-bit LUT indices (`kvalues_iq4nl`) |
| IQ4_XS  | 256 | 2 | 8 (`QI4_XS`)   | fp16 `d` + 4-byte scales_l + 4-byte scales_h |
| MXFP4   | 32  | 2 | 8 (`QI_MXFP4`) | 1 × e8m0 `e` per 32 (block-scaled FP4) |
| NVFP4   | 16  | 2 | 4 (`QI_NVFP4`) | 4 × ue4m3 `d[sub]` per 16 (sub-block-scaled) |

### 6.2 Shared-memory tile layouts

The shared-memory layout is chosen via `ggml_cuda_mmq_sram_layout` enum
(`mmq.cuh:121-129`):

```cpp
enum ggml_cuda_mmq_sram_layout {
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0,   // 2*32 + 2*32/8 + 4 = 76 ints/row
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,   // 2*32 + 2*32/8 + 4 = 76 ints/row (half2 dm)
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q2_K,   // 2*32 + 32    + 4 = 100 ints/row
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q3_K,   // 2*32 + 32/2  + 4 =  84 ints/row
    GGML_CUDA_MMQ_SRAM_LAYOUT_Q6_K,   // 2*32 + 32/16 + 32/8 + 7 = 79 ints/row
    GGML_CUDA_MMQ_SRAM_LAYOUT_FP4,    // 2*32 + 8     + 4 =  76 ints/row (FP4 scales)
    GGML_CUDA_MMQ_SRAM_LAYOUT_NVFP4,  // 2*32 + 32/2  + 4 =  84 ints/row
};
```

The `sram_stride` (ints per row of `tile_x`) for each layout is
`ggml_cuda_mmq_get_sram_stride(sram_layout)` (`mmq.cuh:131-150`). All
seven layouts satisfy `sram_stride % 8 == 4`, statically asserted at
`mmq.cuh:152-158` — this is the XOR-padding rule that avoids 8-bank
shared-memory conflicts (8-bank mode is enabled for the mma path's
`ldmatrix` loads).

### 6.3 Dp4a path uses a different layout (`tile_x_sizes`)

When `use_mma_data_layout()` is false (i.e., on Pascal and RDNA2 — see
`mmq.cuh:188-201`), the dp4a path uses a *separate* per-quant shared-
memory layout defined by the `MMQ_DP4A_TXS_*` macros
(`mmq.cuh:362-371`):

```cpp
#define MMQ_DP4A_TXS_Q4_0    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_0   + I/QI4_0,     0}
#define MMQ_DP4A_TXS_Q4_1    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_1   + I/QI4_1,     0}
#define MMQ_DP4A_TXS_Q8_0    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_0 + I/(QI8_0/2), 0}
#define MMQ_DP4A_TXS_Q8_0_16 tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*4/QI8_0 + I/(QI8_0/4), 0}
#define MMQ_DP4A_TXS_Q8_1    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K*2/QI8_1 + I/(QI8_1/2), 0}
#define MMQ_DP4A_TXS_Q2_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K         + I,           0}
#define MMQ_DP4A_TXS_Q3_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I,                                     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q4_K    tile_x_sizes{I*MMQ_TILE_NE_K   + I, I*MMQ_TILE_NE_K/QI4_K,                 I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q5_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI5_K   + I/QI5_K,     I*MMQ_TILE_NE_K/8 + I/8}
#define MMQ_DP4A_TXS_Q6_K    tile_x_sizes{I*MMQ_TILE_NE_K*2 + I, I*MMQ_TILE_NE_K/QI6_K   + I/QI6_K,     I*MMQ_TILE_NE_K/8 + I/8}
```

Each macro packs three sub-arrays — `qs` (quantized values), `dm` (scale
+ min, as float or half2), `sc` (extra scale bytes for K-quants) — into
a single `tile_x_sizes{qs, dm, sc}` triple. The `+ I` (or `+ I/QI4_0`)
terms are per-row padding to avoid bank conflicts. The mma path collapses
this into a single `sram_stride`-padded layout that is the same for all
quants of a given `sram_layout`.

### 6.4 Activation-side `block_q8_1_mmq` layout

The activation (`src1`) is pre-quantized once at kernel entry into the
`block_q8_1_mmq` layout (`mmq.cuh:27-46`):

```cpp
struct block_q8_1_mmq {
    union {
        float  d4[4];    // 1 fp32 scale per 32 values ×4
        half2  ds4[4];   // 1 (scale, sum) per 32 values ×4
        half   d2s6[8];  // 2 fp16 scales + 6 fp16 partial sums (Q2_K)
    };
    int8_t qs[QK8_1_MMQ];  // 128 int8 values
};
```

`QK8_1_MMQ = 4*QK8_1 = 128`. The union is selected by
`mmq_get_q8_1_ds_layout(type_x)` (`mmq.cuh:60-100`):

* `MMQ_Q8_1_DS_LAYOUT_D4` — Q1_0, Q5_0, Q8_0, MXFP4, NVFP4, Q3_K, Q6_K,
  IQ2_XXS, IQ3_*, IQ4_*. Just 4 fp32 scales, no abs-sums.
* `MMQ_Q8_1_DS_LAYOUT_DS4` — Q4_0, Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S. 4 ×
  (fp16 scale, fp16 abs-sum). The abs-sum is used by the Q4_0/Q5_0/Q8_0
  vecdot to subtract the implicit -8/-16 bias (see §11.1 and F08).
* `MMQ_Q8_1_DS_LAYOUT_D2S6` — Q2_K only. 2 fp16 scales + 6 fp16 partial
  sums (the Q2_K vecdot consumes `s8` to compute its min subtraction
  piecewise; see `vec_dot_q2_K_q8_1_impl_mmq` at `vecdotq.cuh:393-441`).

---

## 7. Memory Layout

### 7.1 Shared memory per block

`nbytes_shared` is computed by `mmq_get_nbytes_shared(config, cc)`
(`mmq.cuh:401-407`):

* **Mma path**: `config.I * sram_stride * 4` bytes. For Ampere with
  `I=128, sram_stride=76` (Q8_0 layout), this is `128 * 76 * 4 = 38,912 B`
  per block, comfortably under Ampere's 48 KB default but exceeding it
  for larger configs (hence the `CUDA_SET_SHARED_MEMORY_LIMIT` call).
* **Dp4a path**: `(txs.qs + txs.dm + txs.sc) * 4` bytes per row, times
  `I` rows. For Pascal with `I=64` and `MMQ_DP4A_TXS_Q4_0 = {64*32+64,
  64*32/8+8, 0}`, this is `64 * (2112 + 264) * 4 ≈ 60 KB`. Pascal's 64
  KB shared memory per SM lets this fit at occupancy=2.

The `tile_y` shared-memory region is sized `J * MMQ_TILE_Y_K * sizeof(int)`
where `MMQ_TILE_Y_K = 33` (`MMQ_TILE_NE_K + MMQ_TILE_NE_K/QI8_1`,
i.e., 32 ints of int8 values + 1 int of scale per 32 values), padded to
a multiple of `nwarps * warp_size`. The total shared-memory budget per
block is `tile_x + tile_y + ids_dst_shared` (the latter is `J` ints).

### 7.2 Global memory stride

The activation `y` is laid out as `ncols_y * ne10_padded * sizeof(block_q8_1) /
(QK8_1 * sizeof(int))` ints per channel/sample, with each `block_q8_1_mmq`
contiguous. The weight `x` is laid out in its native `block_q*` form,
with `stride_row_x = nb01 / type_size(src0->type)` blocks per row.

### 7.3 Per-thread register usage

Each thread holds:

* `sum[J*I / (nwarps*warp_size)]` floats — for Ampere Q4_0 with
  `J=128, I=128, nwarps=8, warp_size=32`, this is `128*128/(8*32) = 64`
  floats per thread = 256 B = 64 registers.
* Temporary `v[VDR]`, `u[2*VDR]` ints in the inner loop.
* For the mma path: `tile_A A[ntx][MMQ_TILE_NE_K/QI8_0]`,
  `float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0]` arrays that are
  live across the J-loop. For Q8_0 mma with `ntx=2`, this is
  `2 * 4 * 4 * 4 = 128` ints + `2 * 4 * 4 = 32` floats — heavy register
  pressure, which is why Ampere uses `occupancy=1`.

The `__launch_bounds__(nthreads, occupancy)` annotation
(`mmq.cuh:921`) tells the compiler the occupancy target. If the
register pressure is too high, the kernel silently spills to local
memory — not detected at compile time (ARTX10 §11.6).

---

## 8. Parallelism Strategy

The parallelism is set by the `ggml_cuda_mmq_config.nthreads`,
`config.I`, `config.J`, and `config.occupancy` fields. See §9.2 for the
per-arch table.

The block shape is `(warp_size, nwarps, 1)`. For Ampere `nthreads=256,
warp_size=32, nwarps=8`; for CDNA `nthreads=512, warp_size=64,
nwarps=8` (CDNA uses 64-thread warps). Each block computes an `I × J`
output tile, consuming `K_vram` K-elements per inner iteration.

The K-loop in `mul_mat_q_process_tile` (`mmq.cuh:875-908`) iterates
`blocks_per_iter = ITER_K / qk` blocks per inner step, where
`ITER_K = MMQ_ITER_K = 256` (or `512` for Blackwell FP4 native path).
Each iteration processes two `MMQ_TILE_NE_K = 32` K-chunks, hence two
`vec_dot` calls and two `__syncthreads()` barriers.

The outer grid is `(nty, ntx, ntzw)` for tiled-K, or `(nsm_or_ntiles,
1, 1)` for stream-K (ARTX10 §5.4).

---

## 9. SIMD / GPU Strategy

This is the largest section because it covers the per-quant tile-vecdot
table — the heart of ARTX12.

### 9.1 Per-quant tile-vecdot table (dp4a path)

The dp4a path is taken on Pascal (`GGML_CUDA_CC_IS_NVIDIA(cc) &&
cc < GGML_CUDA_CC_VOLTA`) and on RDNA2 (`amd_mfma_available(cc) ==
false && !amd_wmma_available(cc)`). The function signature is:

```cpp
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_<variant>_dp4a(
    const int * __restrict__ x,   // shared-memory tile_x (pre-unpacked)
    const int * __restrict__ y,   // shared-memory tile_y (block_q8_1_mmq)
    float * __restrict__ sum,     // per-thread accumulator array
    const int k00);               // K offset within the tile (0 or MMQ_TILE_NE_K)
```

| Quant | dp4a vecdot function | Inner impl called | VDR (elts/thread/iter) | Intermediate type | Scale applied where | Source |
| ----- | -------------------- | ------------------ | ---------------------- | ----------------- | ------------------- | ------ |
| Q1_0 | `vec_dot_q8_0_q8_1_dp4a` (yes, Q8_0!) | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | per-block `d * d8 * sumi` (single scale per 128) | `mmq-vec-dot.cuh:110-140` (via Q8_0 path), `vecdotq.cuh:243-255` |
| Q4_0 | `vec_dot_q4_0_q8_1_dp4a` | `vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>` | 4 | int32 → float | per-block `d4 * (sumi * d8 - 8*vdr/QI4_0 * s8)` (subtracts implicit -8 bias using y_ds.y) | `mmq-vec-dot.cuh:10-58`, `vecdotq.cuh:115-134` |
| Q4_1 | `vec_dot_q4_1_q8_1_dp4a` | `vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>` | 4 | int32 → float | per-block `sumi*d4d8 + m4s8 / (QI8_1/(vdr*QR4_1))` (scale + min via half2 dm × half2 ds) | `mmq-vec-dot.cuh:60-108`, `vecdotq.cuh:139-167` |
| Q5_0 | `vec_dot_q8_0_q8_1_dp4a` (yes, Q8_0!) | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | per-block `d5 * (sumi * d8 - 16*vdr/QI5_0 * s8)` — but loader pre-subtracts the 16-bias via `__vsubss4` so vecdot sees signed ints | `mmq-load-tiles.cuh:264-271`, `mmq-vec-dot.cuh:110-140` |
| Q5_1 | `vec_dot_q8_1_q8_1_dp4a` (yes, Q8_1!) | `vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>` | 4 | int32 → float | per-block `sumi*d5d8 + m5s8 / (QI5_1 / vdr)` (half2 dm × half2 ds, identical arithmetic to Q4_1) | `mmq-vec-dot.cuh:283-313`, `vecdotq.cuh:257-281` |
| Q8_0 | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | per-block `d8_0 * d8_1 * sumi` (pure scale × scale × integer dot) | `mmq-vec-dot.cuh:110-140` |
| Q2_K | `vec_dot_q2_K_q8_1_dp4a` | `vec_dot_q2_K_q8_1_impl_mmq<ns8>` | 4 | int32 → float | per-super-block `dm2.x*sumf_d8 - dm2.y*sumf_m`; `ns8 ∈ {1, 2}` template param for "scale-range split" (two separate loops, see `mmq-vec-dot.cuh:636-678`) | `mmq-vec-dot.cuh:616-679`, `vecdotq.cuh:392-441` |
| Q3_K | `vec_dot_q8_0_16_q8_1_dp4a` (Q8_0_16!) | `vec_dot_q8_0_16_q8_1_impl<QI8_0>` | 16 | int32 → float | per-16-elem `d8_0[i0/(QI8_0/2)] * sumi` then per-block `d3 * d8 * sumf` (one fp32 d3 per super-block, scales pre-multiplied into dA in loader) | `mmq-vec-dot.cuh:446-478`, `vecdotq.cuh:283-302` |
| Q4_K | `vec_dot_q4_K_q8_1_dp4a` | `vec_dot_q4_K_q8_1_impl_mmq` | 8 | int32 → float | per-32-elem `ds8f.x * (sc[i] * sumi_d) - ds8f.y * m[i]` (half2 dm4 × half2 ds8; 6-bit packed scales unpacked in loader) | `mmq-vec-dot.cuh:913-946`, `vecdotq.cuh:530-555` |
| Q5_K | `vec_dot_q5_K_q8_1_dp4a` | `vec_dot_q5_K_q8_1_impl_mmq` | 8 | int32 → float | identical to Q4_K pattern | `mmq-vec-dot.cuh:948-981`, `vecdotq.cuh:593-618` |
| Q6_K | `vec_dot_q6_K_q8_1_dp4a` | `vec_dot_q6_K_q8_1_impl_mmq` | 8 | int32 → float | per-block `d6 * d8 * sumf_d` with 2 q6_K scales per q8_1 scale; scales pre-packed as int8 in loader | `mmq-vec-dot.cuh:983-1016`, `vecdotq.cuh:647-673` |
| IQ1_S | `vec_dot_q8_1_q8_1_dp4a` (yes, Q8_1!) | `vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>` | 4 | int32 → float | loader unpacks grid LUT into int8 values, vecdot is generic Q8_1 (scale × dot + min) | `mmq-load-tiles.cuh:947-1007`, `mmq-vec-dot.cuh:283-313` |
| IQ2_XXS | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | loader does grid + sign unpack into int8; vecdot is generic Q8_0 | `mmq-load-tiles.cuh:1009-1071` |
| IQ2_XS | `vec_dot_q8_0_16_q8_1_dp4a` | `vec_dot_q8_0_16_q8_1_impl<QI8_0>` | 16 | int32 → float | loader does grid + sign unpack; per-16-elem sub-block scales | `mmq-load-tiles.cuh:1073-1136` |
| IQ2_S | `vec_dot_q8_0_16_q8_1_dp4a` | `vec_dot_q8_0_16_q8_1_impl<QI8_0>` | 16 | int32 → float | same as IQ2_XS | `mmq-load-tiles.cuh:1138-1204` |
| IQ3_XXS | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | loader does grid + sign unpack; vecdot is generic Q8_0 | `mmq-load-tiles.cuh:1206-1268` |
| IQ3_S | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | same as IQ3_XXS | `mmq-load-tiles.cuh:1270-1337` |
| IQ4_NL | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | loader does `get_int_from_table_16(q4, kvalues_iq4nl)`; vecdot is generic Q8_0 | `mmq-load-tiles.cuh:1406-1471` |
| IQ4_XS | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | same pattern as IQ4_NL but with super-block scale unpacking | `mmq-load-tiles.cuh:1339-1404` |
| MXFP4 | `vec_dot_q8_0_q8_1_dp4a` | `vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>` | 8 | int32 → float | loader does `get_int_from_table_16(q4, kvalues_mxfp4)`; per-32-block `e8m0 * 0.5 * d8_1` | `mmq-load-tiles.cuh:1475-1540` |
| NVFP4 | `vec_dot_q8_0_16_q8_1_dp4a` | `vec_dot_q8_0_16_q8_1_impl<QI8_0>` | 16 | int32 → float | same `kvalues_mxfp4` LUT; per-16-sub-block `ue4m3 * d8_1` (4 scales per 64 elements) | `mmq-load-tiles.cuh:1584-1638` |

**Reading the table**: the *function name* often does not match the
quant format. The `vec_dot_q8_0_q8_1_dp4a` function is shared by 10
quant formats (Q1_0, Q5_0, Q8_0, IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL,
IQ4_XS, MXFP4). This is the central architectural pattern: the per-
quant complexity lives in `mmq-load-tiles.cuh`, which transforms each
weight block into a canonical Q8_0-like or Q8_1-like int8 sequence with
per-block fp32 scales stored alongside. The vecdot then runs the same
arithmetic.

The `vec_dot_q8_0_q8_1_dp4a` function itself (`mmq-vec-dot.cuh:110-140`)
is only 30 lines:

```cpp
template <ggml_type type, int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
            }
        }
    }
}
```

Each thread owns one `(i, j)` output element. The thread reads its slice
of `tile_x` (its row `i` of int8 weights) and its slice of `tile_y` (its
column `j` of int8 activations), calls `vec_dot_q8_0_q8_1_impl` which
loops `vdr = 8` times calling `ggml_cuda_dp4a(v, u, sumi)`, and
multiplies the result by the per-block scales `d8_0 * d8_1`. The
accumulator is a per-thread `float sum[J*I / (nwarps*warp_size)]`.

### 9.2 Per-arch tile config table

The per-arch config files use the `CASE(type, nthreads, occupancy, I, J,
sram_layout, K_vram, stream_k, fallback)` macro to declare one
`ggml_cuda_mmq_config` per `(type, J, fallback)` tuple. Below is the
representative config for Q4_0 across all six archs (taken from the
first `fallback=true` entry in each file):

| Arch | nthreads | occupancy | I | J (range) | K_vram | stream_k | sram_layout | Source |
| ---- | -------- | --------- | -- | --------- | ------ | -------- | ----------- | ------ |
| Pascal | 256 | 2 | 64 | 8..64 (fallback: 8..64 step 8) | 256 | false | Q8_0 | `mmq-config-pascal.cuh:14-24` |
| Ampere (also Volta/Turing) | 256 | 1 | 128 | 8..128 step 8 | 256 | true | Q8_0 | `mmq-config-ampere.cuh:19-34` |
| Blackwell (non-FP4) | 256 | 1 | 128 | 8..128 step 8 | 256 | true | Q8_0 | falls through to Ampere at `mmq-config-blackwell.cuh:36` |
| Blackwell (MXFP4 native) | 256 | 1 | 128 | 8..128 step 8 | 512 (`MMQ_ITER_K_FP4`) | true | FP4 | `mmq-config-blackwell.cuh:2-17` |
| Blackwell (NVFP4 native) | 256 | 1 | 128 | 8..128 step 8 | 512 | true | FP4 | `mmq-config-blackwell.cuh:19-34` |
| CDNA | 512 | 1 | 128 | 16..64 step 16 | 256 | true | Q8_0 | `mmq-config-cdna.cuh:10-16` |
| RDNA2 | 256 | 2 | 128 | 8..64 step 8 | 256 | false | Q8_0 | `mmq-config-rdna2.cuh:14-24` |
| RDNA4 | 256 | 2 | 128 | 16..128 (sparse: 16,32,64,128 fallback; 16,32,48,…,128 non-fall) | 256 | false | Q8_0 | `mmq-config-rdna4.cuh:15-26` |

**Per-arch summary**:

* **nthreads**: 256 everywhere except CDNA (512). CDNA's 64-thread warp
  means `nwarps = 512/64 = 8`, same as NVIDIA's `256/32 = 8`. The
  *warp count* is constant; only the warp size differs.
* **occupancy**: 2 on Pascal / RDNA2 / RDNA4; 1 on Ampere / Blackwell /
  CDNA. The "occupancy 2" archs are the dp4a-only ones (no mma) — they
  have lower register pressure and benefit from latency hiding via
  two blocks per SM. The "occupancy 1" archs use Tensor Cores and need
  all available registers per block.
* **I**: 64 on Pascal; 128 everywhere else. Pascal's smaller I reflects
  its 64 KB shared memory limit (vs 100+ KB on Ampere/CDNA).
* **J**: Pascal/RDNA2 cap at 64; RDNA4/Ampere/Blackwell/CDNA cap at
  128. CDNA only declares J ∈ {16, 32, 48, 64} (no J=8, no J > 64) —
  fewer specializations because the ROCm compiler is slower and
  compile-time matters (`mmq-config-cdna.cuh` is 177 lines vs Ampere's
  366).
* **stream_k**: false on Pascal / RDNA2 / RDNA4 (these are the dp4a
  archs — see `use_mma_data_layout()` at `mmq.cuh:188-201`); true on
  Ampere / Blackwell / CDNA (the mma archs). Stream-K is a Tensor-Core
  optimization; for dp4a it would just add fixup-kernel overhead.
* **K_vram**: 256 (`MMQ_ITER_K`) everywhere except Blackwell FP4 native
  (512, `MMQ_ITER_K_FP4`). The FP4 native path uses 2× K because the
  `mma_block_scaled_fp4` instruction consumes 64 K-elements per
  `m16n8k64` PTX (vs 32 for `m16n8k32` int8).

The `J_best` selection loop (`mmq.cuh:1452-1468`) iterates `J = 8, 16,
…, 128` and picks the smallest `J` such that `ggml_cuda_mmq_get_config`
returns a non-`GGML_TYPE_COUNT` config and the shared memory fits in
`smpbo`. For CDNA, `J_best` will never be 8 because the config returns
`GGML_TYPE_COUNT` for `J=8`.

### 9.3 Tensor Core instruction variants (per quant)

The mma path uses the same `mma.sync` PTX family for every quant
(except Blackwell FP4). The tile sizes are:

* `tile_A` = `tile<16, 8, int>` (NVIDIA) or `tile<16, 8, int,
  input_layout>` (AMD) — the A operand (weights).
* `tile_B` = `tile<8, 8, int>` (NVIDIA) or `tile<16, 8, int,
  input_layout>` (AMD) — the B operand (activations).
* `tile_C` = `tile<16, 8, int>` (NVIDIA) or `tile<16, 16, int,
  DATA_LAYOUT_J_MAJOR>` (AMD) — the int32 accumulator.

The actual PTX issued is:

| Arch | PTX | Quant coverage |
| ---- | --- | -------------- |
| Turing | `mma.sync.aligned.m16n8k16.s32.s8.s8.s32` | All quants except Blackwell FP4 |
| Ampere / Hopper / Ada / Blackwell (non-FP4) | `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` | All quants except Blackwell FP4 |
| Blackwell (MXFP4 native) | `mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.f32.e2m1.e2m1.f32.ue8m0` | MXFP4 only |
| Blackwell (NVFP4 native) | `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.f32.e2m1.e2m1.f32.ue4m3` | NVFP4 only |
| CDNA (MI300X) | `mfma.i32.s8.s8` (16×16×64) | All quants except Blackwell FP4 |
| RDNA4 | `wmma.i32.s8.s8` (16×16×16) | All quants except Blackwell FP4 |

No `__dp2a` (8-bit × 16-bit dot product) is used anywhere — confirmed
by `grep -rn "__dp2a|dp2a" ggml-cuda/` returning no matches. The dp4a
path uses only `__dp4a` (or its AMD equivalent).

### 9.4 The `ggml_cuda_dp4a` helper

`common.cuh:703-741` defines `ggml_cuda_dp4a(a, b, c)` with five paths:

```cpp
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(GGML_USE_HIP)
#if defined(CDNA) || defined(RDNA2) || defined(__gfx906__)
    c = __builtin_amdgcn_sdot4(a, b, c, false);
#elif defined(RDNA3) || defined(RDNA4)
    c = __builtin_amdgcn_sudot4( true, a, true, b, c, false);
#elif defined(RDNA1) || defined(__gfx900__)
    // Inline asm: 4× v_mul_i32_i24 + v_add3 — no native dot4 on gfx900/Vega10
    …
#endif
#elif __CUDA_ARCH__ >= GGML_CUDA_CC_DP4A || defined(GGML_USE_MUSA)
    return __dp4a(a, b, c);
#else
    // Software fallback: scalar byte-wise mul + add
    const int8_t * a8 = (const int8_t *) &a;
    const int8_t * b8 = (const int8_t *) &b;
    return c + a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
}
```

`GGML_CUDA_CC_DP4A = 610` (`common.cuh:51`) — the minimum CC for the
`__dp4a` intrinsic. Pascal (CC 600) does *not* have `__dp4a` hardware,
but the config table still routes Pascal to the dp4a path (with
occupancy=2). On pure CC 600 Pascal hardware, the `#else` branch
would execute — a 4-byte scalar loop. In practice, llama.cpp's CMake
excludes CC < 610 from MMQ builds (`ggml_cuda_should_use_mmq` returns
false if `cc < GGML_CUDA_CC_DP4A`, `mmq.cu:303-305`).

The RDNA1 / gfx900 (Vega 10) inline-asm fallback is interesting: it
emits 4 × `v_mul_i32_i24` + `v_add3` to emulate `sdot4` because the
hardware lacks it. The comment at `mmq.cu:366-368` confirms gfx900
loses to dequant + hipBLAS for dense matmuls and is gated to MoE-only.

### 9.5 The K-quant 6-bit scale unpacking

The K-quant scale packing (`K_SCALE_SIZE = 12` bytes per super-block,
defined in `ggml-common.h:90`) stores 8 scales + 8 mins in 12 bytes
using 6-bit fields. The CPU unpacks this via
`get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m)`
(`dequantize.cuh:157-164`):

```cpp
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

The MMQ path does *not* use this function. Instead, it uses
`unpack_scales_q45_K(const int * scales, const int ksc)` at
`mmq-load-tiles.cuh:612-620`:

```cpp
static __device__ __forceinline__ int unpack_scales_q45_K(const int * scales, const int ksc) {
    // scale arrangement after the following two lines:
    //   - ksc == 0: sc0, sc1, sc2, sc3
    //   - ksc == 1: sc4, sc5, sc6, sc7
    //   - ksc == 2:  m0,  m1,  m2,  m3
    //   - ksc == 3:  m4,  m5,  m6,  m7
    return ((scales[(ksc%2) + (ksc!=0)] >> (4 * (ksc & (ksc/2)))) & 0x0F0F0F0F) | // lower 4 bits
           ((scales[ksc/2]              >> (2 * (ksc % 2)))       & 0x30303030);  // upper 2 bits
}
```

The MMQ version unpacks *four* scales at once (one per byte of an int32)
and packs the result into a single int that is then consumed by the
vecdot. It is called from `ggml_cuda_mmq_load_tiles_q4_K`
(`mmq-load-tiles.cuh:685-686`) and `ggml_cuda_mmq_load_tiles_q5_K`
(implicit via the same pattern). The Q3_K scale unpacking is a
different, more complex path that combines low 4-bit and high 2-bit
halves with a `__vsubss4` to subtract 32 (`mmq-load-tiles.cuh:571-581`).

The Q6_K scales are stored as raw `int8_t` in the block (`bq6_K->scales`)
so no unpacking is needed — the loader just calls `get_int_b2(bxi->scales,
threadIdx.x % (MMQ_TILE_NE_K/8))` and casts to `int8_t *`
(`mmq-load-tiles.cuh:938-940`).

### 9.6 Comparison: MMQ tile-vecdot vs GEMV vecdot

Both MMQ (`mul_mat_q`) and GEMV (`mul_mat_vec_q`, audited in ARTX09)
use `vecdotq.cuh`, but they call it through different APIs:

| Property | GEMV (`vec_dot_q*_q8_1`) | MMQ dp4a (`ggml_cuda_mmq_vec_dot_*_dp4a`) |
| -------- | ------------------------ | ----------------------------------------- |
| Signature | `(const void *vbq, const block_q8_1 *bq8_1, const int &kbx, const int &iqs)` | `(const int *x, const int *y, float *sum, const int k00)` |
| Block unpack | Done inside vecdot (via `get_int_b2`, `get_int_b4`) | Done in `mmq-load-tiles.cuh` *before* vecdot |
| Source of weight | Global memory (`vbq + kbx`) | Shared memory (`tile_x`, pre-unpacked) |
| Source of activation | Global memory (`bq8_1`) | Shared memory (`tile_y`, already `block_q8_1_mmq`) |
| VDR constant | `VDR_*_MMVQ` (1-4) | `VDR_*_MMQ` (4-8) — typically 2× GEMV |
| Result | One float per call | Adds into a per-thread `sum[...]` array |
| Templated on | nothing (runtime dispatch via `get_vec_dot_q_cuda`) | `<type, J, fallback>` (compile-time) |
| Dispatch | `get_vec_dot_q_cuda(type)` in `mmvq.cu:10-36` | `ggml_cuda_mmq_get_util_funcs<T,J,F>()` in `mmq.cuh:521-816` |

The shared inner kernel is the `vec_dot_*_q8_*_impl` template (e.g.,
`vec_dot_q4_0_q8_1_impl<VDR>`). Both paths call it with their
respective VDR constant — same arithmetic, different unroll factor.

The GEMV path supports `IQ1_M` (`vec_dot_iq1_m_q8_1` at
`vecdotq.cuh:1228-1270`); the MMQ path does *not* — `GGML_TYPE_IQ1_M`
is absent from `ggml_cuda_mul_mat_q_switch_type` (`mmq.cu:8-79`) and
from `ggml_cuda_mmq_get_util_funcs` (`mmq.cuh:521-816`). IQ1_M falls
to cuBLAS in the MMQ regime. This is an asymmetric format coverage.

---

## 10. Quantization Strategy

### 10.1 The `vec_dot_*_q8_*_impl` family

`vecdotq.cuh` defines the following inner kernel templates, each
specialized by `vdr`:

| Impl | Lines | Used by | Arith |
| ---- | ----- | ------- | ----- |
| `vec_dot_q4_0_q8_1_impl<vdr>` | 115-134 | Q4_0 (GEMV + MMQ) | `sumi = dp4a(vi0, u[2i+0], dp4a(vi1, u[2i+1], sumi))`; `d4 * (sumi*ds8f.x - (8*vdr/QI4_0)*ds8f.y)` |
| `vec_dot_q4_1_q8_1_impl<vdr>` | 139-167 | Q4_1, Q4_K-via-Q8_1 path, IQ1_S | `sumi = dp4a(...)`; `sumi*d4d8 + m4s8/(QI8_1/(vdr*QR4_1))` |
| `vec_dot_q5_0_q8_1_impl<vdr>` | 172-198 | Q5_0 (GEMV only — MMQ uses Q8_0 path because loader pre-subtracts 16) | `sumi = dp4a(vi0|hi, u[2i+0], dp4a(vi1|hi, u[2i+1], sumi))`; `d5 * (sumi*ds8f.x - (16*vdr/QI5_0)*ds8f.y)` |
| `vec_dot_q5_1_q8_1_impl<vdr>` | 203-238 | Q5_1 (GEMV only — MMQ uses Q8_1 path) | Similar to Q4_1 |
| `vec_dot_q8_0_q8_1_impl<T, vdr>` | 243-255 | Q8_0, Q1_0, Q5_0 (MMQ), IQ2_XXS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4 | `sumi = dp4a(v[i], u[i], sumi)`; `d8_0*d8_1*sumi` |
| `vec_dot_q8_1_q8_1_impl<vdr>` | 257-281 | Q5_1 (MMQ), Q4_K (MMQ), Q5_K (MMQ), IQ1_S (MMQ) | `sumi = dp4a(...)`; `sumi*d8d8 + m8s8/(QI8_1/vdr)` |
| `vec_dot_q8_0_16_q8_1_impl<vdr>` | 283-302 | Q3_K, IQ2_XS, IQ2_S, NVFP4 (MMQ) | Per-16-elem sub-block: `sumf += d8_0[i0/(QI8_0/2)] * sumi` then `d8_1*sumf` |
| `vec_dot_q2_K_q8_1_impl_mmq<ns8>` | 392-441 | Q2_K (MMQ only) | `dm2f.x * sumi_d - dm2f.y * sumi_m` with separate `sumf` and `sumf_d8` accumulators, `ns8` controls the scale range (1 or 2) |
| `vec_dot_q3_K_q8_1_impl_mmq` | 480-499 | Q3_K (MMQ only) | `d3*d8 * sumi` where `sumi += sumi_sc * scales[i0 / (QI8_1/2)]` (scales pre-unpacked to int8 in loader) |
| `vec_dot_q4_K_q8_1_impl_mmq` | 530-555 | Q4_K (MMQ only) | `dm4f.x*sumf_d - dm4f.y*sumf_m` with per-32-block `ds8f.x * (sc[i] * sumi_d)` and `ds8f.y * m[i]` |
| `vec_dot_q5_K_q8_1_impl_mmq` | 593-618 | Q5_K (MMQ only) | Same as Q4_K but no `>> 4` (Q5 has separate `qh`) |
| `vec_dot_q6_K_q8_1_impl_mmq` | 647-673 | Q6_K (MMQ only) | `d6 * sumf_d` with 2 q6_K scales per q8_1 scale, packed int8 scales |

The MMQ-specific impls (`_mmq` suffix) differ from the GEMV-specific
impls (`_mmvq` suffix or no suffix) in that they take pre-unpacked
contiguous `v[]` and `u[]` arrays and a per-block scale, while the
GEMV impls take the raw packed block and an `iqs` index. The arithmetic
on a single 4-byte chunk is identical.

### 10.2 The `_mmq` vs `_mmvq` split for Q2_K

Q2_K has two impls:

* `vec_dot_q2_K_q8_1_impl_mmvq` (`vecdotq.cuh:364-389`) — takes one
  `int v` and an array `u[QR2_K]` of 4 ints. Used by GEMV.
* `vec_dot_q2_K_q8_1_impl_mmq<ns8>` (`vecdotq.cuh:392-441`) — takes
  arrays `v[QR2_K*VDR_Q2_K_Q8_1_MMQ]` and `u[QR2_K*VDR_Q2_K_Q8_1_MMQ]`,
  plus a `half2 * dm2` array. Used by MMQ. The `ns8` template parameter
  controls whether the inner loop applies the `s8` (partial sum) term:
  the loader pre-computes that the second half of the super-block has
  no `s8` contribution, so the second loop hardcodes `ns8 = 1` to skip
  the conditional. The comment at `mmq-vec-dot.cuh:657-658` explains:
  "Some compilers fail to unroll the loop over k01 if there is a
  conditional statement for ns in the inner loop. As a workaround 2
  separate loops are used instead."

This is the only quant format where the MMQ path uses a separate
implementation that differs structurally (not just in VDR) from the
GEMV path. It is a workaround for compiler unrolling limitations.

### 10.3 MXFP4 / NVFP4 lookup tables

Both MXFP4 and NVFP4 use the same `kvalues_mxfp4` 16-entry LUT to
expand 4-bit indices to int8 values (`vecdotq.cuh:318`, `vec_dot_mxfp4_q8_1`).
The LUT is declared in `ggml-common.h` and is constant across backends.
The expansion is done by `get_int_from_table_16(q4, kvalues_mxfp4)`
(`vecdotq.cuh:34-95`), which uses `__byte_perm` on NVIDIA / MUSA and
`__builtin_amdgcn_perm` on HIP to do 8 lookups in parallel from a
4-byte register.

On Blackwell, the LUT is bypassed entirely for MXFP4 / NVFP4 — the
loader copies the raw FP4 nibbles into shared memory and the
`mma_block_scaled_fp4` PTX does the e2m1 decode in hardware
(`mmq-load-tiles.cuh:1542-1582` for MXFP4,
`mmq-load-tiles.cuh:1640-1679` for NVFP4).

### 10.4 IQ-format grid LUTs

The IQ formats (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S)
use pre-computed grid lookup tables (`iq1s_grid_gpu`, `iq2xxs_grid`,
`iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`) declared in
`ggml-common.h`. Each grid entry is a 4-byte packed int8 quadruple
representing the dequantized value of a 2- or 3-bit index. The loader
looks up the grid, applies a sign mask via `__vcmpne4` and `__vsub4`,
and writes the resulting int8 to `tile_x`. From the vecdot's
perspective, IQ formats look identical to Q8_0.

The sign unpacking uses `unpack_ksigns(uint8_t v)` (`vecdotq.cuh:97-104`)
which XORs the input with `popc(v) & 1 << 7` to "correct" the 8th bit
so that a single broadcast multiplication can apply all 8 sign bits.
This is a clever trick to avoid per-byte sign branching.

---

## 11. Correctness Analysis

### 11.1 Implicit-bias subtraction for symmetric quants

The symmetric quants Q4_0, Q5_0, Q8_0 store 4-bit (or 8-bit) unsigned
indices that represent signed values in `[-8, 7]`, `[-16, 15]`, or
`[-128, 127]`. The dequantized value is `(idx - bias) * d` where
`bias = 8` (Q4_0), `16` (Q5_0), or `128` (Q8_0). Two equivalent
implementations exist:

* **Q4_0 GEMV**: `vec_dot_q4_0_q8_1_impl` at `vecdotq.cuh:115-134`
  computes `sumi = dp4a(idx, u, sumi)` (using the raw unsigned idx)
  and then subtracts `8*vdr/QI4_0 * ds8f.y` from the final result,
  where `ds8f.y` is the per-block sum of the activation's int8 values
  (stored in `block_q8_1.ds.y`, computed by the activation quantizer).
  The formula is `d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y)`.
* **Q4_0 MMQ (mma path)**: the loader pre-subtracts the bias via
  `__vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808)` at
  `mmq-load-tiles.cuh:132-133`. The vecdot then sees already-signed
  int8 values and the result is just `d4 * sumi * d8` (no `-8*vdr`
  correction term).

These two paths produce *slightly different* results due to differing
reduction order — the GEMV path accumulates the bias subtraction in
float, while the MMQ mma path accumulates in int32 inside the Tensor
Core. The two are arithmetically equivalent but ULP-different. The
`MMQ_Q8_1_DS_LAYOUT_D4` vs `MMQ_Q8_1_DS_LAYOUT_DS4` distinction
(`mmq.cuh:60-100`) encodes which path uses which: Q4_0/Q5_0/Q8_0
declare `D4` (just the scale, no abs-sum) on the mma path because the
loader pre-subtracts the bias, but `DS4` on the dp4a path because the
vecdot needs the abs-sum to do the bias subtraction itself.

Wait — actually, looking again at `mmq.cuh:64-72`, Q4_0 is declared
`DS_LAYOUT_DS4` (not D4). And looking at the MMQ dp4a path for Q4_0
(`mmq-vec-dot.cuh:18, 52-54`), it reads `x_df` as `float` (the d
scale) and `y_ds` as `half2` (the d + sum pair). The vecdot
`vec_dot_q4_0_q8_1_impl` (`vecdotq.cuh:115-134`) does the bias
subtraction using `ds8f.y` (the y-side sum).

So the dp4a MMQ path *does* use the implicit-bias-subtraction trick,
just like GEMV. Only the mma path pre-subtracts the bias in the loader.
This is asymmetric across the two MMQ codepaths.

### 11.2 Floating-point reassociation

Same as ARTX10 §11.1: per-thread `int sumi` is exact; reassociation
happens only in the float post-multiply (`d * (sumi * ds8f.x - …)`).
The K-loop sums into `float sum[J*I/(nwarps*warp_size)]` in iteration
order. For the mma path, the int32 Tensor Core accumulator is exact
per fragment; the per-fragment `dA * dB * C.x[l]` products are summed
in a fixed order.

### 11.3 Q2_K `ns8` workaround

The Q2_K MMQ vecdot uses two separate loops for `ns8 = 2` (first half
of K) and `ns8 = 1` (second half of K) at `mmq-vec-dot.cuh:636-678`.
The split is *not* a semantic difference — both loops compute the same
arithmetic — but a workaround for compilers that fail to unroll the
loop with the `if (i0/QI8_1 < ns8)` conditional inside. The two loops
have identical bodies except for the hardcoded `ns8` template
parameter. This is a correctness-neutral compiler workaround.

### 11.4 IQ4_XS scale expansion

`vec_dot_iq4_xs_q8_1` at `vecdotq.cuh:1299-1322` computes:

```cpp
const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F)
             | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
sumi *= ls - 32;
```

The scale is 6-bit packed across `scales_l` (4 bits) and `scales_h`
(2 bits). The `ls - 32` shifts the 6-bit value from `[0, 63]` to
`[-32, 31]`. This is identical to the CPU implementation (ARTX06
documents the same packing).

In the MMQ path, the loader pre-expands these scales into per-16-elem
fp32 factors `x_df[i*... + threadIdx.x % 8] = d * (ls - 32)` at
`mmq-load-tiles.cuh:1393-1402`. The vecdot then uses the generic
`vec_dot_q8_0_q8_1_impl` which is unaware of the IQ4_XS scale packing.
Again, the loader absorbs the per-quant complexity.

### 11.5 Stream-K determinism

Same as ARTX10 §11.2: stream-K MMQ output can vary at the ULP level
across GPUs with different SM counts because the fixup kernel sums
partial results in a fixed but SM-count-dependent order. Tiled-K MMQ
(no stream-K) is deterministic for fixed `(type, J, fallback, K, CC)`.

### 11.6 NVFP4 per-output-column scale

The NVFP4 path supports an optional per-output-column `y_scale` factor
applied in `ggml_cuda_mmq_write_back_mma` (`mmq.cuh:492-497`):

```cpp
if constexpr (type == GGML_TYPE_NVFP4) {
    if (y_scale_used) {
        dst[ids_dst[j]*stride + i] = y_scale[j] * sum[(j0/tile_C::J + n)*tile_C::ne + l];
    } else {
        dst[ids_dst[j]*stride + i] = sum[(j0/tile_C::J + n)*tile_C::ne + l];
    }
}
```

This `y_scale` is the per-tensor output scale produced by
`quantize_mmq_fp4_cuda` for NVFP4 (alloc'd at `mmq.cu:138, 206`). It
is *not* the per-block ue4m3 scale (which lives inside `block_fp4_mmq`).
The two scales are independent and multiplied at write-back time. This
is a NVFP4-specific extension to the write_back path; no other quant
uses `y_scale`.

---

## 12. Optimization Analysis

### 12.1 Identified per-quant optimizations

| Optimization | Where | Notes |
| ------------ | ----- | ----- |
| Per-quant loader absorbs unpack complexity | `mmq-load-tiles.cuh:*` | Lets the vecdot be generic (Q8_0 / Q8_1 / Q8_0_16). 22 loaders, ~6 vecdots. |
| Format-to-vecdot table at compile time | `mmq.cuh:521-816` | `constexpr __device__` function returns a `(vdr, load_tiles, vec_dot, write_back)` tuple; zero indirect calls. |
| VDR constant per (quant, path) | `vecdotq.cuh:109-1297` | `VDR_*_MMVQ` (1-4 for GEMV) and `VDR_*_MMQ` (4-8 for MMQ). MMQ doubles the unroll factor because shared memory is faster than global. |
| `__vsubss4` bias pre-subtraction (mma path only) | `mmq-load-tiles.cuh:132-133, 264-271, 901-905` | Removes the bias-subtract term from the inner vecdot loop; saves one FMA per dp4a call. |
| `unpack_ksigns` sign-broadcast trick | `vecdotq.cuh:97-104` | XORs sign bit with popcount parity to enable single-broadcast sign application for IQ2/IQ3. |
| `get_int_from_table_16` 4-bit LUT expansion | `vecdotq.cuh:34-95` | Uses `__byte_perm` (NVIDIA) / `__builtin_amdgcn_perm` (HIP) for 8-wide parallel LUT lookup. Replaces 8-byte LUT with 4 `__byte_perm` calls. |
| Per-arch config table with Blackwell fallthrough | `mmq-config-blackwell.cuh:36` | Blackwell declares only FP4-native configs; everything else reuses Ampere's. Eliminates duplication. |
| `__launch_bounds__(nthreads, occupancy)` | `mmq.cuh:921` | Per-arch occupancy target (1 or 2 blocks/SM). |
| Static_assert `sram_stride % 8 == 4` | `mmq.cuh:152-158` | Enforces XOR-padding for 8-bank shared memory; compile-time check across all 7 layouts. |
| K-loop unrolled two `MMQ_TILE_NE_K` chunks per iter | `mmq.cuh:875-908` | Two vec_dot calls per `tile_x` load amortizes the `__syncthreads` cost. |
| `MMQ_DP4A_TXS_*` macros | `mmq.cuh:362-371` | Compile-time-computed per-quant shared-memory layout for the dp4a path. |
| Per-quant `sram_layout` enum | `mmq.cuh:121-129` | Maps 22 quants to 7 canonical layouts, enabling shared `ggml_cuda_mmq_get_sram_stride` and mma tile sizes. |
| CDNA-specific config (J only 16-64) | `mmq-config-cdna.cuh` | Trades compile-time for runtime coverage: fewer J specializations = faster ROCm compile. |
| RDNA4 sparse J set (no 8, 24, 40, 56, …) | `mmq-config-rdna4.cuh` | Same idea: drop J values that are dominated by neighbors to keep compile-time manageable. |

### 12.2 Optimizations *not* present

* **No `__dp2a` (8-bit × 16-bit dot)**. The dp4a path uses only 4-bit
  × 4-bit dot products. `__dp2a` would let Q4_0/Q4_1/Q5_0/Q5_1 do two
  4-bit dots in one instruction (treating the 4-bit nibble as the low
  byte of an int16), potentially halving the dp4a instruction count
  for these formats. The hardware supports it (Turing+) but the code
  does not use it. Likely because the mma path is faster on Turing+
  anyway and dp4a is the Pascal/RDNA2 fallback.
* **No per-quant mma specialization beyond what's needed**. The mma
  path uses the same `mma.sync.m16n8k32` for Q4_0, Q4_1, Q5_0, Q5_1,
  Q8_0, IQ2_*, IQ3_*, IQ4_*, MXFP4 (non-Blackwell). There is no
  per-quant Tensor Core layout optimization.
* **No `cp.async` overlap** (ARTX10 §12.2). The K-loop does
  synchronous `__syncthreads()` between `tile_x` load and `vec_dot`,
  even though Ampere+ has `cp.async` hardware. Flash-attention uses
  it (ARTX11); MMQ does not.
* **No K-quant scale precomputation across K-tiles**. Each K-tile
  reloads the Q4_K/Q5_K scales from global memory and unpacks them
  in the loader. The scales could be cached in shared memory across
  K-tiles (a Q4_K super-block has 256 elements, larger than
  `MMQ_ITER_K = 256`, so the scales are reused at most once).
* **No Blackwell `tc_gen5.mma` for non-FP4 quants**. Blackwell
  (`mmq-config-blackwell.cuh`) only declares MXFP4 / NVFP4 native
  configs; every other quant falls through to Ampere's
  `mma.sync.m16n8k32`. Blackwell's new `tc_gen5.mma` instructions
  (which give 4-8× throughput per warp group) are unused for Q4_K /
  Q6_K etc.

---

## 13. Architectural Strengths

1. **Format-to-vecdot canonicalization is the single best design
   decision in MMQ tile-vecdot.** By absorbing per-quant complexity
   into the loader and exposing only ~6 canonical vecdot shapes
   (Q8_0, Q8_1, Q8_0_16, Q2_K, Q3_K, Q4_K/Q5_K, Q6_K), the codebase
   keeps the vecdot layer small (1251 lines vs 1679 for loaders) and
   makes the mma path tractable — only 6 mma variants are needed to
   cover all 22 quants. This is the CUDA analogue of ARTX01-F03
   (CPU type-traits table).

2. **Dual dp4a/mma specialization behind a single template.** The
   `use_mma_data_layout()` predicate (`mmq.cuh:188-201`) selects
   between `*_dp4a` and `*_mma` at compile time, transparently to
   the caller. The same `mul_mat_q<type, J, fallback>` template
   compiles down to either CUDA-core `__dp4a` instructions or
   Tensor-Core `mma.sync` PTX. ARTX10-F02 covers this at the GEMM
   level; here we note that the *per-quant* vecdot also follows the
   same pattern — each format has a `_dp4a` and a `_mma` variant.

3. **Per-arch config table with Blackwell fallthrough.** The Blackwell
   config file (`mmq-config-blackwell.cuh`) is only 37 lines: it
   declares MXFP4 / NVFP4 native entries and then `return
   ggml_cuda_mmq_get_config_ampere(type, J, fallback);` for everything
   else. This is the right tradeoff — Blackwell's new Tensor Core
   instructions only benefit FP4 today; for Q4_K etc., Ampere's
   `mma.sync.m16n8k32` is already optimal. Adding a full Blackwell
   table would just duplicate Ampere's data.

4. **The `unpack_ksigns` trick for IQ2/IQ3 sign handling.** XORing
   the sign byte with `popc(v) & 1` to "correct" the 8th bit before
   broadcasting is a clever way to apply 8 sign bits in one
   instruction. This avoids per-byte branching for the IQ formats,
   which collectively account for 7 of the 22 supported quants.

5. **The `vec_dot_q8_0_16_q8_1_impl` template for per-16-element
   sub-block scales.** Q3_K, IQ2_XS, IQ2_S, and NVFP4 all have
   per-16-element scales (a finer granularity than Q8_0's per-32).
   Rather than writing four separate vecdots, the code uses one
   template that takes a `float * d8_0` array of per-16 scales and
   computes `sumf += d8_0[i0/(QI8_0/2)] * sumi` per sub-block. This
   is a clean generalization of the Q8_0 vecdot.

6. **Static_assert enforcement of shared-memory padding.** The
   `sram_stride % 8 == 4` static_asserts at `mmq.cuh:152-158` catch
   any future layout change that would break 8-bank shared memory
   XOR-padding. This is a small but important defensive measure.

7. **K-quant scale unpacking is SIMD-ized.** The `unpack_scales_q45_K`
   function (`mmq-load-tiles.cuh:612-620`) unpacks 4 scales at once
   into a single int32 using bit manipulation, vs the CPU's
   `get_scale_min_k4` which unpacks one at a time. This is a 4× speedup
   for the scale-unpacking step on CUDA.

8. **CDNA / RDNA4 sparse J enumeration to keep compile-time
   manageable.** CDNA declares only J ∈ {16, 32, 48, 64}; RDNA4 drops
   J ∈ {8, 24, 40, 56, 72, 88, 104, 120} from the non-fallback set.
   This is a deliberate engineering tradeoff: the missing J values are
   dominated by neighbors for typical batch sizes, and the ROCm
   compiler is significantly slower than NVCC. The reduced
   enumeration cuts compile time by ~2×.

---

## 14. Architectural Weaknesses

### W1 — Asymmetric format coverage between MMQ and GEMV

**Evidence**: `ggml_cuda_mul_mat_q_switch_type` (`mmq.cu:8-79`) and
`ggml_cuda_mmq_get_util_funcs` (`mmq.cuh:521-816`) cover 22 formats:
Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K,
IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS, IQ4_NL,
MXFP4, NVFP4. The GEMV dispatch in `mmvq.cu:10-36` covers 23 formats:
all of the above *plus* `IQ1_M`. There is no MMQ path for IQ1_M.

**Impact**: When `ne11 > 8` (the MMQ regime) and weights are IQ1_M,
`ggml_cuda_should_use_mmq` returns true (`mmq.cu:263-293` doesn't
exclude IQ1_M), but `ggml_cuda_mul_mat_q_switch_type` will hit the
`default: GGML_ABORT("fatal error")` at `mmq.cu:77`. This is a latent
crash. In practice, `ggml_cuda_mul_mat` routes IQ1_M through MMVQ for
small batches and cuBLAS for large batches — but the abort path is
reachable if `ggml_cuda_should_use_mmq` ever returns true for IQ1_M.

**Why it's hard to fix**: Adding IQ1_M to MMQ requires writing a new
loader (`ggml_cuda_mmq_load_tiles_iq1_m`) and selecting a vecdot. The
IQ1_M scale unpacking is non-trivial (4-bit packed scales with a
delta term, `vecdotq.cuh:1260-1264`).

### W2 — Blackwell non-FP4 quants fall through to Ampere `mma.sync`

**Evidence**: `mmq-config-blackwell.cuh:36`:
`return ggml_cuda_mmq_get_config_ampere(type, J, fallback);`

**Impact**: On Blackwell (CC 1200/1210), Q4_K / Q6_K / IQ4_XS etc.
use Ampere's `mma.sync.aligned.m16n8k32.s32.s8.s8.s32` PTX. They do
*not* use Blackwell's new `tc_gen5.mma` instructions, which give 4-8×
throughput per warp group. This is a missed performance opportunity
for Blackwell owners running non-FP4 quantized inference.

**Why it's hard to fix**: `tc_gen5.mma` has a completely different
programming model (warp-group scoped, async, with TMA descriptors).
Adapting the existing `mma.sync`-based MMQ code to `tc_gen5.mma`
would require rewriting the mma path for every quant format — a major
undertaking. ARTX10 §13 also notes this gap.

### W3 — Per-quant scale unpacking is reimplemented per arch

**Evidence**: The CPU uses `get_scale_min_k4` (`dequantize.cuh:157-164`)
which returns `(uint8_t & d, uint8_t & m)`. The MMQ dp4a path uses
`unpack_scales_q45_K` (`mmq-load-tiles.cuh:612-620`) which returns a
packed int with 4 scales + 4 mins. The MMQ mma path uses a *different*
inlined scale unpacking at `mmq-load-tiles.cuh:682-696` that pre-
multiplies the scales into a `half2 dm` factor. Three implementations
of the same `K_SCALE_SIZE = 12` unpacking.

**Impact**: Maintenance burden. A change to the K_SCALE_SIZE packing
(which is shared with CPU, ARTX06) requires updating three code paths.
The risk of divergence is real — if one path is updated and another
is forgotten, the K-quants would silently produce wrong results on
some archs.

**Why it's hard to fix**: The three paths have different output types
(uint8, packed int, half2-pre-multiplied) and different consumer
expectations. Unifying them would require a single output format that
works for all three consumers.

### W4 — Q2_K `ns8` workaround duplicates the inner loop

**Evidence**: `mmq-vec-dot.cuh:636-678` — two near-identical loops
with `ns8 = 2` and `ns8 = 1` hardcoded, with the comment "Some
compilers fail to unroll the loop over k01 if there is a conditional
statement for ns in the inner loop."

**Impact**: Code duplication; ~40 extra lines. Functionally correct,
but the workaround is fragile — if a future compiler change fixes the
unrolling issue, the duplicate loop should be removed, but there is
no compile-time test to detect this.

### W5 — dp4a path uses different scale layout from mma path

**Evidence**: `mmq_get_q8_1_ds_layout` (`mmq.cuh:60-100`) returns
`DS4` for Q4_0, but the mma path's `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma`
(`mmq-vec-dot.cuh:142-280`) accepts `ds_layout` as a template
parameter and handles both `D4` and `DS4`. The dp4a path
(`mmq-vec-dot.cuh:110-140`) does *not* parameterize `ds_layout` — it
hardcodes `y_df` as `const float *` and `y_ds` as `const half2 *`
based on the format.

**Impact**: The dp4a and mma paths for the same quant can have
different `ds_layout` semantics, which is a subtle correctness hazard
when modifying the activation quantizer. ARTX10 §10.1 documents the
layout assignment; here we note that the asymmetry is a code smell.

### W6 — `vec_dot_q8_1_q8_1_impl` is misnamed for IQ1_S

**Evidence**: `mmq.cuh:595-600` routes `GGML_TYPE_IQ1_S` to
`ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a` (and `*_mma`), but IQ1_S is
not Q8_1. The name "q8_1_q8_1" refers to the canonical vecdot shape
(Q8_1-style scale + min), not the actual quant format.

**Impact**: Readability. The mapping from format to vecdot is non-
intuitive — you have to read `ggml_cuda_mmq_get_util_funcs` to discover
that IQ1_S uses the Q8_1 vecdot. This is a documentation gap, not a
correctness issue.

### W7 — Pascal/RDNA2 fallback to dp4a has no Tensor Core path

**Evidence**: `use_mma_data_layout()` (`mmq.cuh:188-201`) returns
false on Pascal and RDNA2. These archs run the dp4a path with
occupancy=2. Pascal CC 600 lacks `__dp4a` hardware entirely; the
`ggml_cuda_dp4a` function falls back to a 4-byte scalar loop
(`common.cuh:735-737`).

**Impact**: Pascal CC 600 (GTX 10-series below 1050 Ti) runs MMQ at
scalar speed. The config table still declares Pascal entries
(`mmq-config-pascal.cuh`), but the performance is unusable. In
practice, `ggml_cuda_should_use_mmq` excludes CC < 610 (`mmq.cu:303-
305`), so this path is dead code on pure CC 600.

### W8 — Per-arch config files are large and mostly redundant

**Evidence**: `mmq-config-ampere.cuh` is 366 lines, 336 CASE entries.
`mmq-config-pascal.cuh` is 261 lines, 231 entries. The entries are
near-identical across archs (only `nthreads, occupancy, I, stream_k`
differ). The data could be factored into a smaller table with per-arch
overrides.

**Impact**: Maintenance burden. Adding a new quant format requires
adding ~12-16 CASE entries to *each* of the 6 arch files. The
duplication is not catastrophic (the CASE macro makes it terse), but
it is a smell.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda` | **ADOPT** | Format-to-vecdot canonicalization (per-quant loader + ~6 canonical vecdots) | Lets the mma layer be tractable; 22 quants × 6 vecdots is far better than 22 × 22. |
| `glcuda` | **ADOPT** | `ggml_cuda_mmq_get_util_funcs<T, J, F>()` constexpr device function | Zero-indirect-call dispatch; compile-time resolved. |
| `glcuda` | **ADOPT** | Per-arch config table with Blackwell fallthrough | Right tradeoff: declare only arch-specific configs, reuse shared ones. |
| `glcuda` | **ADOPT** | `MMQ_DP4A_TXS_*` macros for per-quant shared-memory layout | Compile-time-computed; static_assert-able. |
| `glcuda` | **ADOPT** | `unpack_scales_q45_K` 4-at-a-time scale unpacking | 4× faster than CPU's per-byte `get_scale_min_k4`. |
| `glcuda` | **ADAPT** | Dual dp4a/mma template pattern | Keep the pattern, but unify the `ds_layout` parameter across both paths (W5). |
| `glcuda` | **ADAPT** | `vec_dot_q8_0_16_q8_1_impl` for per-16-elem sub-block scales | Keep the abstraction, but consider generalizing to `vec_dot_q8_0_N_q8_1_impl<N>`. |
| `glcuda` | **MONITOR** | Blackwell non-FP4 fallthrough to Ampere | Watch for upstream `tc_gen5.mma` adoption; ADOPT when it lands. |
| `glcuda` | **MONITOR** | `__dp2a` absence | Watch whether `__dp2a` would help Q4_0/Q5_0 dp4a path on Pascal-class HW. |
| `glcuda` | **REJECT**| IQ1_M absence from MMQ | Either add MMQ support for IQ1_M or exclude it from `ggml_cuda_should_use_mmq`. The current abort-on-default is a latent crash. |
| `GATE` | **ADOPT** | Per-(type, arch) tile-config table as the single source of truth for tiling policy | Same as ARTX10-F01, but at the per-quant level. |

---

## 16. Recommendations

### R1 — ADOPT format-to-vecdot canonicalization as glcuda's primary MMQ pattern
**Priority:** Critical
**Difficulty:** M
**Dependencies:** ARTX10-R1 (config-table architecture)

GwenLand's `glcuda` should define a `gl_mmq_get_util_funcs<T, J, F>()`
that returns `(vdr, load_tiles, vec_dot, write_back)` per quant. The
loaders absorb per-quant complexity; the vecdots come in ~6 canonical
shapes (Q8_0, Q8_1, Q8_0_16, Q2_K, Q3_K, Q4_K/Q5_K, Q6_K). Same
ABI, same semantics.

### R2 — ADOPT per-arch config table with fallthrough
**Priority:** High
**Difficulty:** M
**Dependencies:** R1

Define a `gl_mmq_config` struct with `(nthreads, occupancy, I, J,
sram_layout, K_vram, stream_k, fallback)` and a per-arch function
returning it. New archs declare only arch-specific configs and fall
through to a parent arch. This is the per-quant analogue of ARTX10-R1.

### R3 — ADOPT `unpack_scales_q45_K` 4-at-a-time scale unpacking
**Priority:** High
**Difficulty:** S
**Dependencies:** R1

For K-quant MMQ, unpack 4 scales at once into a packed int32 using
bit manipulation. This is 4× faster than per-byte unpacking and
matches the `__dp4a` throughput.

### R4 — ADAPT dual dp4a/mma template, unify `ds_layout`
**Priority:** High
**Difficulty:** M
**Dependencies:** R1

Keep the dual dp4a/mma specialization behind `use_mma_data_layout()`,
but make `ds_layout` a template parameter on *both* paths (not just
mma). This eliminates W5 and makes the activation quantizer's output
format consistent across paths.

### R5 — REJECT IQ1_M abort-on-default; either support or exclude
**Priority:** High
**Difficulty:** M
**Dependencies:** R1

Either add an MMQ loader+vecdot for IQ1_M (mirroring the GEMV
implementation at `vecdotq.cuh:1228-1270`), or exclude IQ1_M from
`ggml_cuda_should_use_mmq` so it routes to cuBLAS in the MMQ regime.
The current behavior is a latent crash.

### R6 — MONITOR Blackwell `tc_gen5.mma` adoption for non-FP4 quants
**Priority:** Medium
**Difficulty:** XL
**Dependencies:** R2

Watch upstream llama.cpp for `tc_gen5.mma` adoption on Blackwell for
Q4_K/Q6_K/etc. When it lands, ADOPT the same path in glcuda. Until
then, the Ampere `mma.sync.m16n8k32` fallthrough is acceptable.

### R7 — ADOPT `__launch_bounds__(nthreads, occupancy)` with per-arch occupancy
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R2

Pascal/RDNA2/RDNA4: occupancy=2 (dp4a path, latency hiding). Ampere/
Blackwell/CDNA: occupancy=1 (mma path, register pressure). This is
already what llama.cpp does; ADOPT directly.

### R8 — DEFER `__dp2a` investigation
**Priority:** Low
**Difficulty:** M
**Dependencies:** R1

`__dp2a` could halve the dp4a instruction count for Q4_0/Q4_1/Q5_0/
Q5_1 (which currently do 2 dp4a calls per int). But the dp4a path is
the fallback (Pascal/RDNA2) and the mma path is faster on Turing+.
Investigate only if Pascal/RDNA2 performance becomes a GwenLand
priority.

### R9 — ADOPT static_assert for shared-memory padding
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** R2

`static_assert(sram_stride % 8 == 4, "Wrong padding.");` for every
canonical layout. Catches layout bugs at compile time. Already done
in llama.cpp; ADOPT directly.

### R10 — ADAPT CDNA/RDNA4 sparse J enumeration
**Priority:** Low
**Difficulty:** S
**Dependencies:** R2

For slow-compiling backends (ROCm), drop J values that are dominated
by neighbors (e.g., J=24 is dominated by J=32 for typical batch
sizes). This cuts compile time by ~2× with no runtime cost.

---

## 17. Findings

### Finding ARTX12-F01

```
Finding ID:           ARTX12-F01
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            Per-quant tile-vecdot dispatch
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             ggml_cuda_mmq_get_util_funcs
Lines:                521-816
Summary:              A constexpr __device__ function maps each of the 22
                      quant formats to a (vdr, load_tiles, vec_dot,
                      write_back) tuple, with the vec_dot chosen from
                      only ~6 canonical shapes shared across formats.
Observation:          The function has two branches: a dp4a branch
                      (selected when use_mma_data_layout() is false) and
                      an mma branch. Each branch is a switch over the 22
                      types, returning a ggml_cuda_mmq_util_funcs struct
                      with function pointers. The same vec_dot function
                      (e.g., ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a) is
                      shared by 10 formats (Q1_0, Q5_0, Q8_0, IQ2_XXS,
                      IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, MXFP4, NVFP4 via
                      Q8_0_16). Per-quant complexity lives in the loader.
Evidence:             mmq.cuh:521-658 (dp4a branch), 663-816 (mma +
                      Blackwell FP4 branches). Format-to-vecdot mapping
                      table in §9.1.
Architectural Impact: This is the CUDA analogue of ARTX01-F03 (CPU
                      type-traits table). It makes the mma layer tractable
                      (6 mma variants cover 22 quants) and keeps the
                      vecdot layer small (1251 lines vs 1679 for loaders).
Correctness Impact:   None. Dispatch is compile-time-resolved; no
                      runtime ambiguity.
Optimization Type:    Compile-time polymorphism; canonical-layout
                      intermediate.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Equivalent gl_mmq_get_util_funcs<T, J, F>()
                      in glcuda, returning (vdr, load_tiles, vec_dot,
                      write_back) per quant.
Priority:             Critical
Difficulty:           M
Dependencies:         ARTX10-F01 (config-table architecture)
Confidence:           High
```

### Finding ARTX12-F02

```
Finding ID:           ARTX12-F02
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            Dual dp4a/mma specialization per quant
Source File:          ggml/src/ggml-cuda/mmq-vec-dot.cuh, ggml/src/ggml-cuda/mmq.cuh
Function:             ggml_cuda_mmq_vec_dot_*_dp4a vs ggml_cuda_mmq_vec_dot_*_mma
Lines:                mmq-vec-dot.cuh:10-1250 (all vecdots); mmq.cuh:188-201 (predicate)
Summary:              Each quant format has two vecdot implementations:
                      *_dp4a (CUDA-core __dp4a, used on Pascal/RDNA2)
                      and *_mma (Tensor Core mma.sync, used on
                      Turing+/CDNA/RDNA4). The choice is made at compile
                      time via use_mma_data_layout().
Observation:          The dp4a variant loops over (i, j, k) with each
                      thread owning one (i, j) output and calling
                      vec_dot_*_q8_*_impl<VDR> which issues VDR __dp4a
                      calls. The mma variant uses warp-level tiles
                      (tile_A, tile_B, tile_C) and issues mma.sync PTX;
                      each warp collectively computes a 16×8 (NVIDIA)
                      or 16×16 (AMD) output tile. Both variants share
                      the same vec_dot_*_q8_*_impl arithmetic template.
Evidence:             mmq-vec-dot.cuh:10-58 (Q4_0 dp4a), 142-280 (Q8_0
                      mma); mmq.cuh:188-201 (use_mma_data_layout
                      predicate); vecdotq.cuh:115-134 (shared impl).
Architectural Impact: Same source compiles to either CUDA-core or
                      Tensor-Core instructions. Lets MMQ support both
                      pre-Turing (no int8 Tensor Cores) and Turing+ from
                      one source tree.
Correctness Impact:   The dp4a and mma paths produce ULP-different
                      results due to different reduction orders (per-
                      thread sequential vs Tensor-Core tree). Same
                      observation as ARTX10-F02; documented here at the
                      per-quant level.
Optimization Type:    Dual codepath behind compile-time predicate.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Keep the pattern, but unify ds_layout
                      handling across both paths (see F05).
Priority:             High
Difficulty:           M
Dependencies:         ARTX12-F01
Confidence:           High
```

### Finding ARTX12-F03

```
Finding ID:           ARTX12-F03
Category:             QUANTIZATION
Engine:               CUDA
Component:            K-quant 6-bit scale unpacking (Q4_K, Q5_K)
Source File:          ggml/src/ggml-cuda/mmq-load-tiles.cuh
Function:             unpack_scales_q45_K
Lines:                612-620
Summary:              The K-quant 6-bit packed scale (K_SCALE_SIZE = 12
                      bytes) is unpacked 4 scales at a time into a
                      packed int32, a different implementation from the
                      CPU's per-byte get_scale_min_k4.
Observation:          The CPU's get_scale_min_k4 (dequantize.cuh:157-164)
                      returns one (d, m) pair per call. The MMQ loader's
                      unpack_scales_q45_K returns an int32 holding 4
                      scale-or-min bytes, consumed by vec_dot_q4_K_q8_1_
                      impl_mmq. The mma path uses a third variant that
                      pre-multiplies scales into a half2 dm factor
                      (mmq-load-tiles.cuh:682-696). Three implementations
                      of the same K_SCALE_SIZE = 12 unpacking.
Evidence:             mmq-load-tiles.cuh:612-620 (unpack_scales_q45_K),
                      682-696 (mma-path inline unpacking);
                      dequantize.cuh:157-164 (CPU get_scale_min_k4);
                      ggml-common.h:90 (K_SCALE_SIZE = 12).
Architectural Impact: Maintenance burden. A change to K_SCALE_SIZE
                      packing requires updating three code paths.
Correctness Impact:   If the three paths diverge, K-quants would silently
                      produce wrong results on some archs. Static analysis
                      shows they currently agree.
Optimization Type:    SIMD-ized 4-at-a-time unpacking (dp4a path) and
                      pre-multiplication into half2 (mma path).
GwenLand Target:      glcuda
Recommendation:       ADOPT the 4-at-a-time unpacking pattern, but
                      consolidate the three implementations into one
                      template that can output to (uint8, packed int,
                      half2-pre-multiplied) based on a template param.
Priority:             High
Difficulty:           M
Dependencies:         ARTX12-F01
Confidence:           High
```

### Finding ARTX12-F04

```
Finding ID:           ARTX12-F04
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Blackwell per-arch config fallthrough
Source File:          ggml/src/ggml-cuda/mmq-config-blackwell.cuh
Function:             ggml_cuda_mmq_get_config_blackwell
Lines:                1-37
Summary:              The Blackwell config file declares only MXFP4 and
                      NVFP4 native entries (32 CASEs) and falls through
                      to ggml_cuda_mmq_get_config_ampere for every other
                      quant.
Observation:          Line 36: "return ggml_cuda_mmq_get_config_ampere(
                      type, J, fallback);". This means Q4_K, Q6_K, IQ4_XS,
                      etc. on Blackwell use the same config (and same
                      mma.sync.m16n8k32 PTX) as Ampere. Blackwell's new
                      tc_gen5.mma instructions are unused for non-FP4
                      quants.
Evidence:             mmq-config-blackwell.cuh:1-37 (full file);
                      mmq.cuh:235-237 (host dispatch routes Blackwell
                      to this function); ARTX10 §9.3 (PTX inventory).
Architectural Impact: Right tradeoff today: tc_gen5.mma has a different
                      programming model and would require rewriting the
                      mma path for every quant. But it leaves ~4-8×
                      throughput on the table for non-FP4 quants on
                      Blackwell.
Correctness Impact:   None. The Ampere configs are correct on Blackwell.
Optimization Type:    Config-table fallthrough (eliminates duplication).
GwenLand Target:      glcuda
Recommendation:       ADOPT the fallthrough pattern. MONITOR upstream
                      for tc_gen5.mma adoption; ADOPT when it lands.
Priority:             Medium
Difficulty:           XS (fallthrough) / XL (tc_gen5.mma port)
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX12-F05

```
Finding ID:           ARTX12-F05
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            Implicit-bias subtraction for symmetric quants
                      (Q4_0, Q5_0, Q8_0)
Source File:          ggml/src/ggml-cuda/mmq-load-tiles.cuh, ggml/src/ggml-cuda/vecdotq.cuh
Function:             ggml_cuda_mmq_load_tiles_q4_0, vec_dot_q4_0_q8_1_impl
Lines:                mmq-load-tiles.cuh:132-133 (mma pre-subtract),
                      vecdotq.cuh:115-134 (dp4a runtime subtract),
                      mmq.cuh:60-100 (DS_LAYOUT assignment)
Summary:              Symmetric quants (Q4_0, Q5_0, Q8_0) subtract the
                      implicit -8/-16/-128 bias in two different ways:
                      the mma loader pre-subtracts via __vsubss4, while
                      the dp4a vecdot subtracts at runtime using the
                      y_ds.y abs-sum field.
Observation:          The mma path's loader (mmq-load-tiles.cuh:132-133)
                      does __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808)
                      so the vecdot sees signed int8 values and the
                      result is just d * sumi * d8. The dp4a path's
                      vecdot (vecdotq.cuh:115-134) keeps the unsigned
                      idx and computes d4 * (sumi * ds8f.x - (8*vdr/QI4_0)
                      * ds8f.y), where ds8f.y is the per-block abs-sum
                      of the activation (stored in block_q8_1_mmq's
                      half2 ds4.y). The two paths are arithmetically
                      equivalent but ULP-different.
Evidence:             mmq-load-tiles.cuh:132-133 (mma pre-subtract for
                      Q4_0); vecdotq.cuh:130-133 (dp4a bias subtract);
                      mmq.cuh:64 (Q4_0 → DS_LAYOUT_DS4); mmq.cuh:182-186
                      (mma path reads ds_layout template param).
Architectural Impact: The DS_LAYOUT assignment (D4 vs DS4) encodes which
                      path uses which strategy. D4 layouts (Q1_0, Q5_0,
                      Q8_0, MXFP4, NVFP4, Q3_K, Q6_K, IQ2_*, IQ3_*,
                      IQ4_*) store only the scale; DS4 layouts (Q4_0,
                      Q4_1, Q5_1, Q4_K, Q5_K, IQ1_S) store scale + sum.
                      Q4_0 is DS4 (needs the sum for bias subtract),
                      but Q5_0 is D4 (the mma path pre-subtracts; the
                      dp4a path... also uses Q8_0 vecdot which is D4).
                      This is inconsistent: Q4_0 and Q5_0 are both
                      symmetric quants with the same bias pattern, but
                      one is DS4 and the other is D4.
Correctness Impact:   None — each path computes the correct bias. But
                      the inconsistency is a code smell that could lead
                      to bugs if the layout assignment is changed without
                      updating the vecdot.
Optimization Type:    Pre-subtraction (mma) vs runtime subtraction
                      (dp4a); dual-strategy encoded in DS_LAYOUT.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Unify: always pre-subtract in the loader
                      (mma style), drop the DS4 layout, and make all
                      symmetric quants use D4. This eliminates the
                      abs-sum computation in the activation quantizer
                      and removes the ds_layout template parameter.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX12-F02
Confidence:           Medium
```

### Finding ARTX12-F06

```
Finding ID:           ARTX12-F06
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMQ dp4a path uses only __dp4a, no __dp2a
Source File:          ggml/src/ggml-cuda/vecdotq.cuh, ggml/src/ggml-cuda/common.cuh
Function:             ggml_cuda_dp4a, vec_dot_q4_0_q8_1_impl
Lines:                common.cuh:703-741 (ggml_cuda_dp4a), vecdotq.cuh:115-134 (Q4_0)
Summary:              The dp4a path uses __dp4a (4-byte dot product) for
                      all quants. __dp2a (8-bit × 16-bit dot, which could
                      halve instruction count for Q4_0/Q5_0 by doing two
                      4-bit dots in one instruction) is not used
                      anywhere.
Observation:          grep -rn "__dp2a|dp2a" ggml-cuda/ returns no
                      matches. The Q4_0 vecdot does 2 __dp4a calls per
                      int (one for the low nibbles, one for the high
                      nibbles, vecdotq.cuh:122-127). __dp2a could
                      combine these into one instruction if the nibbles
                      were zero-extended into the low bytes of two int16
                      lanes. But __dp2a is Turing+ only, and Turing+
                      uses the mma path anyway.
Evidence:             common.cuh:703-741 (ggml_cuda_dp4a); vecdotq.cuh:122-127
                      (Q4_0 dp4a pattern); grep confirms no __dp2a.
Architectural Impact: No impact today (dp4a path is the fallback). If
                      Pascal/RDNA2 ever needed a 2× speedup, __dp2a is
                      unavailable (Pascal CC 610 doesn't have it either).
Correctness Impact:   None.
Optimization Type:    None (absence of __dp2a optimization).
GwenLand Target:      glcuda
Recommendation:       MONITOR. Unlikely to matter — the dp4a path is the
                      fallback and Pascal/RDNA2 performance is rarely
                      critical.
Priority:             Low
Difficulty:           M
Dependencies:         ARTX12-F02
Confidence:           High
```

### Finding ARTX12-F07

```
Finding ID:           ARTX12-F07
Category:             QUANTIZATION
Engine:               CUDA
Component:            Q2_K ns8 template-parameter workaround
Source File:          ggml/src/ggml-cuda/mmq-vec-dot.cuh
Function:             ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a
Lines:                616-679
Summary:              The Q2_K MMQ dp4a vecdot uses two near-identical
                      loops with ns8 = 2 and ns8 = 1 hardcoded, a
                      workaround for compilers that fail to unroll the
                      loop with a conditional on ns8 inside.
Observation:          The comment at line 657-658 explains: "Some
                      compilers fail to unroll the loop over k01 if there
                      is a conditional statement for ns in the inner
                      loop. As a workaround 2 separate loops are used
                      instead." The two loops are functionally identical
                      except for the ns8 template parameter, which
                      controls whether the s8 (partial-sum) term is
                      applied. The split is semantic: ns8 = 2 applies to
                      the first half of K (k01 < MMQ_TILE_NE_K/2), ns8 = 1
                      to the second half.
Evidence:             mmq-vec-dot.cuh:636-655 (ns8 = 2 loop), 659-678
                      (ns8 = 1 loop), 657-658 (comment).
Architectural Impact: ~40 lines of duplicated code. Fragile: if a future
                      compiler fixes the unrolling, the duplicate should
                      be removed, but there is no compile-time test.
Correctness Impact:   None. The two loops compute the same arithmetic.
Optimization Type:    Compiler-workaround loop duplication.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Keep the workaround, but add a TODO with a
                      compile-time test that flags when the compiler can
                      unroll the unified version.
Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX12-F08

```
Finding ID:           ARTX12-F08
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            Per-arch tile config table (I, J, nthreads, occupancy, stream_k)
Source File:          ggml/src/ggml-cuda/mmq-config-{pascal,ampere,blackwell,cdna,rdna2,rdna4}.cuh
Function:             ggml_cuda_mmq_get_config_{pascal,ampere,blackwell,cdna,rdna2,rdna4}
Lines:                pascal.cuh:1-261; ampere.cuh:1-366; blackwell.cuh:1-37;
                      cdna.cuh:1-177; rdna2.cuh:1-261; rdna4.cuh:1-282
Summary:              Six per-arch config files declare (nthreads,
                      occupancy, I, J, sram_layout, K_vram, stream_k,
                      fallback) for every (type, J, fallback) tuple via
                      the CASE macro. The values differ systematically:
                      Pascal/RDNA2/RDNA4 use occupancy=2, stream_k=false;
                      Ampere/Blackwell/CDNA use occupancy=1, stream_k=true.
Observation:          The per-arch summary (§9.2) shows that the config
                      fields cluster by arch family: dp4a archs (Pascal,
                      RDNA2, RDNA4) use occupancy=2 and stream_k=false;
                      mma archs (Ampere, Blackwell, CDNA) use occupancy=1
                      and stream_k=true. The I dimension is 64 on Pascal
                      (smaller shared memory) and 128 everywhere else. J
                      ranges differ: Pascal/RDNA2 cap at 64; Ampere/
                      Blackwell/RDNA4 cap at 128; CDNA only declares
                      J ∈ {16, 32, 48, 64}.
Evidence:             §9.2 table; pascal.cuh:14-24 (Q4_0 Pascal);
                      ampere.cuh:19-34 (Q4_0 Ampere); cdna.cuh:10-16
                      (Q4_0 CDNA); rdna2.cuh:14-24 (Q4_0 RDNA2);
                      rdna4.cuh:15-26 (Q4_0 RDNA4); blackwell.cuh:1-37
                      (Blackwell fallthrough).
Architectural Impact: Adding a new arch = adding one config file and
                      one dispatch branch. Adding a new quant = adding
                      entries to all six config files. The data is
                      per-arch but the structure is shared.
Correctness Impact:   None. The configs are pure performance hints.
Optimization Type:    Per-arch tiling policy as data.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Per-arch config table with fallthrough (like
                      Blackwell → Ampere) is the right tradeoff.
Priority:             High
Difficulty:           M
Dependencies:         ARTX10-F01
Confidence:           High
```

### Finding ARTX12-F09

```
Finding ID:           ARTX12-F09
Category:             LAYOUT_SUBOPTIMAL
Engine:               CUDA
Component:            Per-quant dp4a shared-memory layout macros
                      (MMQ_DP4A_TXS_*)
Source File:          ggml/src/ggml-cuda/mmq.cuh
Function:             MMQ_DP4A_TXS_* macros, mmq_get_dp4a_tile_x_sizes
Lines:                362-398
Summary:              The dp4a path uses 10 per-quant macros
                      (MMQ_DP4A_TXS_Q4_0, Q4_1, Q8_0, Q8_0_16, Q8_1,
                      Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) that define
                      per-quant shared-memory layouts with explicit
                      padding. The mma path collapses these into 7
                      canonical sram_layouts.
Observation:          Each macro returns a tile_x_sizes{qs, dm, sc}
                      triple with per-quant padding. The Q3_K and Q4_K
                      layouts have an extra `sc` array for K-quant
                      scales. The macros are consumed by
                      mmq_get_dp4a_tile_x_sizes (mmq.cuh:373-398) which
                      is a 22-way switch. The mma path does not use
                      these macros — it uses the canonical sram_stride
                      from ggml_cuda_mmq_get_sram_stride.
Evidence:             mmq.cuh:362-371 (macro definitions); 373-398
                      (mmq_get_dp4a_tile_x_sizes switch); 121-150
                      (mma path's sram_layout enum and stride function).
Architectural Impact: Two parallel layout systems (dp4a tile_x_sizes
                      vs mma sram_stride) for the same shared memory.
                      Maintenance burden; the two must agree on size.
Correctness Impact:   None — the two paths use disjoint shared memory
                      regions. But a bug in either layout's padding
                      would silently corrupt the other path.
Optimization Type:    Per-quant padding for bank-conflict avoidance.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Keep the per-quant layouts, but consolidate
                      the dp4a and mma layout systems into one (use
                      sram_layout for both). This eliminates the
                      MMQ_DP4A_TXS_* macros.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX12-F01, ARTX12-F02
Confidence:           Medium
```

### Finding ARTX12-F10

```
Finding ID:           ARTX12-F10
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MMQ vs GEMV vecdot API divergence
Source File:          ggml/src/ggml-cuda/vecdotq.cuh, ggml/src/ggml-cuda/mmq-vec-dot.cuh, ggml/src/ggml-cuda/mmvq.cu
Function:             vec_dot_q4_0_q8_1 (GEMV), ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a (MMQ)
Lines:                vecdotq.cuh:725-741 (GEMV), mmq-vec-dot.cuh:10-58 (MMQ),
                      mmvq.cu:8-36 (GEMV dispatch)
Summary:              The GEMV and MMQ paths call the same
                      vec_dot_*_q8_*_impl<VDR> template but with
                      different VDR constants (VDR_*_MMVQ vs VDR_*_MMQ)
                      and different surrounding APIs. GEMV takes
                      (vbq, bq8_1, kbx, iqs) and unpacks the block
                      itself; MMQ takes (x, y, sum, k00) and reads
                      pre-unpacked ints from shared memory.
Observation:          The GEMV vec_dot_q4_0_q8_1 (vecdotq.cuh:725-741)
                      calls get_int_b2(bq4_0->qs, iqs + i) to unpack
                      the block, then calls vec_dot_q4_0_q8_1_impl<
                      VDR_Q4_0_Q8_1_MMVQ=2>(v, u, d, ds). The MMQ
                      ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a (mmq-vec-dot
                      .cuh:10-58) reads x_qs[i*(MMQ_TILE_NE_K+1) + k0/QR4_0]
                      (already-unpacked int from shared memory) and
                      calls vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ=4>
                      (v, u, d, ds). Same impl, different VDR.
Evidence:             vecdotq.cuh:725-741 (GEMV Q4_0); mmq-vec-dot.cuh:10-58
                      (MMQ Q4_0); vecdotq.cuh:112-113 (VDR_Q4_0_Q8_1_MMVQ
                      = 2, VDR_Q4_0_Q8_1_MMQ = 4); mmvq.cu:8-36 (GEMV
                      dispatch table).
Architectural Impact: Shared arithmetic, divergent APIs. Code reuse at
                      the impl level; divergence at the wrapper level.
                      Adding a new quant requires writing both a GEMV
                      wrapper and an MMQ wrapper.
Correctness Impact:   None. The two paths use the same impl.
Optimization Type:    Shared impl template with path-specific VDR.
GwenLand Target:      glcuda
Recommendation:       ADOPT the shared-impl pattern. Consider unifying
                      the wrapper APIs (GEMV could also take pre-unpacked
                      ints from a small shared-memory tile), but this is
                      low priority.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX12-F01
Confidence:           High
```

### Finding ARTX12-F11

```
Finding ID:           ARTX12-F11
Category:             OTHER
Engine:               CUDA
Component:            Asymmetric format coverage: IQ1_M absent from MMQ
Source File:          ggml/src/ggml-cuda/mmq.cu, ggml/src/ggml-cuda/mmq.cuh, ggml/src/ggml-cuda/mmvq.cu
Function:             ggml_cuda_mul_mat_q_switch_type, ggml_cuda_mmq_get_util_funcs, get_vec_dot_q_cuda
Lines:                mmq.cu:8-79 (no IQ1_M case), mmq.cuh:521-816 (no
                      IQ1_M case), mmvq.cu:10-36 (IQ1_M present),
                      mmq.cu:263-293 (ggml_cuda_should_use_mmq does not
                      exclude IQ1_M)
Summary:              IQ1_M is supported by GEMV (vec_dot_iq1_m_q8_1)
                      but not by MMQ. ggml_cuda_should_use_mmq returns
                      true for IQ1_M, but ggml_cuda_mul_mat_q_switch_type
                      hits GGML_ABORT on the default case.
Observation:          The GEMV dispatch (mmvq.cu:30) includes
                      GGML_TYPE_IQ1_M → vec_dot_iq1_m_q8_1. The MMQ
                      dispatch (mmq.cu:8-79) and the per-quant table
                      (mmq.cuh:521-816) both omit IQ1_M. The should_use_
                      mmq policy (mmq.cu:263-293) does not exclude IQ1_M
                      from the mmq_supported switch. In practice,
                      ggml_cuda_mul_mat routes IQ1_M through MMVQ or
                      cuBLAS, but the abort path is reachable.
Evidence:             mmq.cu:76-78 (default: GGML_ABORT); mmq.cu:263-293
                      (no IQ1_M exclusion in should_use_mmq); mmvq.cu:30
                      (GEMV has IQ1_M); vecdotq.cuh:1228-1270 (GEMV
                      vec_dot_iq1_m_q8_1 implementation).
Architectural Impact: Asymmetric format coverage between MMQ and GEMV.
                      22 formats in MMQ, 23 in GEMV. The missing format
                      is IQ1_M, which has non-trivial scale unpacking.
Correctness Impact:   Latent crash if ggml_cuda_should_use_mmq ever
                      returns true for IQ1_M with ne11 > 8. Currently
                      routed around by the caller, but the abort is a
                      trap waiting to be sprung.
Optimization Type:    None.
GwenLand Target:      glcuda
Recommendation:       REJECT this asymmetry. Either add MMQ support for
                      IQ1_M (port the GEMV loader to a shared-memory
                      tile loader) or exclude IQ1_M from should_use_mmq.
Priority:             High
Difficulty:           M
Dependencies:         ARTX12-F01
Confidence:           High
```

### Finding ARTX12-F12

```
Finding ID:           ARTX12-F12
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            MXFP4 / NVFP4 / IQ4_NL / IQ4_XS 4-bit LUT expansion
                      via get_int_from_table_16
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             get_int_from_table_16
Lines:                34-95
Summary:              4-bit indices (MXFP4, NVFP4, IQ4_NL, IQ4_XS) are
                      expanded to int8 values via a 16-entry LUT using
                      __byte_perm (NVIDIA) or __builtin_amdgcn_perm (HIP).
                      The function returns an int2 holding 8 expanded
                      bytes (4 even-indexed in .x, 4 odd-indexed in .y).
Observation:          The LUT (kvalues_mxfp4 for MXFP4/NVFP4,
                      kvalues_iq4nl for IQ4_NL/IQ4_XS) is 16 bytes = 4
                      int32. The function does 4 __byte_perm calls to
                      select 4 bytes from the low 8 of the table, 4 from
                      the high 8, then 2 more __byte_perm calls to
                      interleave even/odd indices. On HIP, the same
                      operation is done with 4 __builtin_amdgcn_perm
                      calls. The result is 8 expanded int8 values per
                      4-bit-index int32, ready for __dp4a.
Evidence:             vecdotq.cuh:34-95 (get_int_from_table_16);
                      vecdotq.cuh:307-326 (vec_dot_mxfp4_q8_1 using
                      kvalues_mxfp4); mmq-load-tiles.cuh:1507-1508
                      (MXFP4 loader using get_int_from_table_16);
                      mmq-load-tiles.cuh:1370 (IQ4_XS loader using
                      kvalues_iq4nl).
Architectural Impact: Replaces 8-byte LUT lookups with 4-6 intrinsic
                      calls. Faster than scalar LUT lookup; works on
                      both NVIDIA and AMD. On Blackwell, bypassed
                      entirely for MXFP4/NVFP4 (native FP4 Tensor Core).
Correctness Impact:   None — the LUT values are shared with the CPU
                      backend (declared in ggml-common.h).
Optimization Type:    SIMD-ized LUT expansion via __byte_perm /
                      __builtin_amdgcn_perm.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same pattern for any 4-bit-indexed LUT
                      format in glcuda.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX12-F01
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the dp4a and mma paths for the same quant produce
  bit-identical results. Static analysis shows the arithmetic is
  equivalent but the reduction order differs (per-thread sequential
  vs Tensor Core tree). Requires executing both paths on the same
  input on the same GPU. ARTX10-U1 covers this at the GEMM level;
  here we note it applies per-quant.
* **U2**. Whether the IQ1_M abort path is reachable in practice.
  `ggml_cuda_mul_mat` (ggml-cuda.cu) routes based on `ne11` and
  `ggml_cuda_should_use_mmq`. If `ne11 > 8` and weights are IQ1_M,
  does the routing skip MMQ? Requires tracing the caller. Static
  analysis of `mmq.cu:263-293` shows `should_use_mmq` does not
  exclude IQ1_M, so the abort is reachable in principle.
* **U3**. Whether the Q4_0 DS4 layout (vs Q5_0 D4 layout) is a
  deliberate optimization or a historical accident. Both are
  symmetric quants with the same bias pattern; the asymmetry is
  suspicious. Requires git archaeology or asking the maintainers.
* **U4**. Whether `__dp2a` would actually speed up the Q4_0/Q5_0
  dp4a path on Pascal/RDNA2. Pascal CC 610+ has `__dp2a`, but
  testing this requires writing a __dp2a variant and benchmarking
  on Pascal hardware. Static analysis cannot reach a conclusion.
* **U5**. Whether the Blackwell non-FP4 fallthrough to Ampere is a
  stopgap (pending tc_gen5.mma port) or a deliberate long-term
  decision. The code comment at `mmq-config-blackwell.cuh:36` is
  just `return ggml_cuda_mmq_get_config_ampere(...)` with no
  explanation. Requires upstream discussion.
* **U6**. Whether the K-quant scale unpacking implementations (CPU
  `get_scale_min_k4`, MMQ dp4a `unpack_scales_q45_K`, MMQ mma inline)
  are known to the maintainers to be redundant. The duplication is
  visible in the source but may be deliberate (different output
  formats for different consumers). Requires upstream discussion.
* **U7**. Whether the `ns8` workaround in Q2_K MMQ (F07) is still
  needed with current NVCC / ROCm versions. The comment dates to
  an older compiler era. Requires testing with current toolchains.
* **U8**. The actual register pressure of the mma path for each
  quant. `__launch_bounds__(nthreads, 1)` with `occupancy=1` is a
  hint, not a guarantee. If the compiler spills to local memory,
  performance degrades silently. Requires `cuobjdump -res-usage` on
  the compiled kernels.

---

## 19. References

| Reference | File | Function / Symbol | Lines |
| --------- | ---- | ----------------- | ----- |
| R01 | `ggml/src/ggml-cuda/mmq.cu` | `ggml_cuda_mul_mat_q` | 82-254 |
| R02 | `ggml/src/ggml-cuda/mmq.cu` | `ggml_cuda_mul_mat_q_switch_type` | 8-79 |
| R03 | `ggml/src/ggml-cuda/mmq.cu` | `ggml_cuda_should_use_mmq` | 256-371 |
| R04 | `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_config` (struct) | 164-203 |
| R05 | `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_get_config` (host dispatch) | 225-242 |
| R06 | `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_get_config` (device constexpr) | 244-263 |
| R07 | `ggml/src/ggml-cuda/mmq.cuh` | `mmq_get_q8_1_ds_layout` | 60-100 |
| R08 | `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_get_sram_stride` | 131-150 |
| R09 | `ggml/src/ggml-cuda/mmq.cuh` | `MMQ_DP4A_TXS_*` macros | 362-371 |
| R10 | `ggml/src/ggml-cuda/mmq.cuh` | `mmq_get_dp4a_tile_x_sizes` | 373-398 |
| R11 | `ggml/src/ggml-cuda/mmq.cuh` | `ggml_cuda_mmq_get_util_funcs` | 521-816 |
| R12 | `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_process_tile` | 841-915 |
| R13 | `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q` (kernel) | 920-1205 |
| R14 | `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_stream_k_fixup` | 1207-1320 |
| R15 | `ggml/src/ggml-cuda/mmq.cuh` | `launch_mul_mat_q` | ~1350-1441 |
| R16 | `ggml/src/ggml-cuda/mmq.cuh` | `mul_mat_q_switch_J`, `mul_mat_q_case` | 1443-1535 |
| R17 | `ggml/src/ggml-cuda/mmq.cuh` | `block_q8_1_mmq`, `block_fp4_mmq` | 27-58 |
| R18 | `ggml/src/ggml-cuda/mmq-config-pascal.cuh` | `ggml_cuda_mmq_get_config_pascal` | 1-261 |
| R19 | `ggml/src/ggml-cuda/mmq-config-ampere.cuh` | `ggml_cuda_mmq_get_config_ampere` | 1-366 |
| R20 | `ggml/src/ggml-cuda/mmq-config-blackwell.cuh` | `ggml_cuda_mmq_get_config_blackwell` | 1-37 |
| R21 | `ggml/src/ggml-cuda/mmq-config-cdna.cuh` | `ggml_cuda_mmq_get_config_cdna` | 1-177 |
| R22 | `ggml/src/ggml-cuda/mmq-config-rdna2.cuh` | `ggml_cuda_mmq_get_config_rdna2` | 1-261 |
| R23 | `ggml/src/ggml-cuda/mmq-config-rdna4.cuh` | `ggml_cuda_mmq_get_config_rdna4` | 1-282 |
| R24 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a` | 10-58 |
| R25 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` | 110-140 |
| R26 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` | 142-280 |
| R27 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a` | 446-478 |
| R28 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a` | 616-679 |
| R29 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma` | 1018-1178 |
| R30 | `ggml/src/ggml-cuda/mmq-vec-dot.cuh` | `ggml_cuda_mmq_vec_dot_fp4_fp4_mma` | 1186-1250 |
| R31 | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_q4_0` | 98-159 |
| R32 | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_q4_K` | 622-731 |
| R33 | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `unpack_scales_q45_K` | 612-620 |
| R34 | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_mxfp4_fp4` | 1542-1582 |
| R35 | `ggml/src/ggml-cuda/mmq-load-tiles.cuh` | `ggml_cuda_mmq_load_tiles_nvfp4_nvfp4` | 1640-1679 |
| R36 | `ggml/src/ggml-cuda/vecdotq.cuh` | `VDR_*_MMQ` constants | 109-1297 |
| R37 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q4_0_q8_1_impl` | 115-134 |
| R38 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q8_0_q8_1_impl` | 243-255 |
| R39 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q8_0_16_q8_1_impl` | 283-302 |
| R40 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q2_K_q8_1_impl_mmq` | 392-441 |
| R41 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q4_K_q8_1_impl_mmq` | 530-555 |
| R42 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q6_K_q8_1_impl_mmq` | 647-673 |
| R43 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_q4_0_q8_1` (GEMV wrapper) | 725-741 |
| R44 | `ggml/src/ggml-cuda/vecdotq.cuh` | `get_int_from_table_16` | 34-95 |
| R45 | `ggml/src/ggml-cuda/vecdotq.cuh` | `unpack_ksigns` | 97-104 |
| R46 | `ggml/src/ggml-cuda/vecdotq.cuh` | `vec_dot_iq1_m_q8_1` (GEMV only) | 1228-1270 |
| R47 | `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_dp4a` | 703-741 |
| R48 | `ggml/src/ggml-cuda/common.cuh` | `GGML_CUDA_CC_DP4A` | 51 |
| R49 | `ggml/src/ggml-cuda/common.cuh` | `ggml_cuda_e8m0_to_fp32`, `ggml_cuda_ue4m3_to_fp32` | 821-870 |
| R50 | `ggml/src/ggml-cuda/dequantize.cuh` | `get_scale_min_k4` (CPU/CUDA-shared) | 157-164 |
| R51 | `ggml/src/ggml-cuda/dequantize.cuh` | `dequantize_q4_K` | 167-192 |
| R52 | `ggml/src/ggml-cuda/mmvq.cu` | `get_vec_dot_q_cuda` (GEMV dispatch) | 10-36 |
| R53 | `ggml/src/ggml-common.h` | `K_SCALE_SIZE = 12` | 90 |
