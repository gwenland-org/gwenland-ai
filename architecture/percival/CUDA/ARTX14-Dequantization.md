# ARTX14 — CUDA Dequantization and Type Conversion

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-26
**Auditor:** Percival-aux (ARTX13+14)
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (kernel selection)

---

## 1. Executive Summary

The CUDA dequantization / type-conversion path is the **materialization
layer** between llama.cpp's packed quant formats and the F32/F16/BF16
arithmetic that the rest of the GPU backend consumes. It is invoked
whenever a quantized tensor must be presented as a non-quantized tensor
to a downstream op that has no fused-vecdot equivalent — for example,
attention key/value cache ingest, `GET_ROWS` on quantized embedding
tables, or cuBLAS fallback for matmuls with quantized weights. Three
files implement it:

1. **`dequantize.cuh`** (432 lines) — per-format `dequantize_*` device
   functions. Each takes a `(const void * vx, int64_t ib, [int tid],
   dst_t * yy / float2 & v)` triple and produces either a `float2`
   (simple quants, 2 elements per call) or a `QK_K`-element `dst_t`
   slice (K-quants and IQ formats, 256 elements per call). Templated
   on `dst_t` so the same kernel writes F32, F16, or BF16 directly —
   no F32 intermediate.
2. **`convert.cu`** (693 lines) — the host-side dispatch layer. Six
   `ggml_get_to_*_cuda(ggml_type)` functions return a function pointer
   to the right launcher for each `(source dtype, destination dtype,
   contiguous-or-not)` tuple. Hosts the generic `dequantize_block`
   template, the fused `dequantize_block_q8_0_f16` kernel, the per-quant
   `dequantize_block_q*` specialisations, and the generic
   `convert_unary<src_t, dst_t>` kernel for F32/F16/BF16 conversions.
3. **`quantize.cu`** (698 lines) — the F32→quant counterpart. Hosts
   `quantize_q8_1` (single-thread-per-block warp-reduce scale+sum),
   `quantize_mmq_q8_1` (float4 + shfl_xor reduce, three output layouts),
   and the Blackwell-only `quantize_mmq_nvfp4` (5-candidate ue4m3 scale
   search) and `quantize_mmq_mxfp4` (per-32-element e8m0 scale).

The architectural decisions worth **ADOPT**ing are: (a) the `dst_t`
template parameter on every dequantize function — one source, three
output dtypes, no F32 round-trip; (b) the `dequantize_kernel_t` typedef
(`common.cuh:947`) that lets `dequantize_block`, `k_get_rows`, and the
`cpy.cu` copy-fuser all share one device-function-pointer contract;
(c) the fused `dequantize_block_q8_0_f16` kernel (the only quant→F16
fast path; uses shared-memory staging + `__hmul2` broadcast); (d) the
`ggml_cuda_cast<dst_t>` SFINAE-style helper that handles every
src/dst type pair in one template. The decisions worth **ADAPT**ing
are: the six near-duplicate `ggml_get_to_*_cuda` switch tables (130+
lines of copy-paste that could be one table indexed by `(src_type,
dst_type)`); the K-quant 6-bit scale unpacker (canonical in
`dequantize.cuh:157-164` as `get_scale_min_k4`, but re-implemented
twice more in `vecdotq.cuh` and `mmq-load-tiles.cuh` — see ARTX13-F06);
the per-quant block thread-count policy (`dequantize_row_*_cuda`
launchers use 32 threads for some formats and 64 for others, baked into
each launcher). The decisions worth **REJECT**ing are: the absence of
any fused dequantize+RoPE kernel (RoPE is always applied in a separate
op) and the absence of explicit `__constant__` qualification on the
lookup tables (they live as `static const __device__` in
`ggml-common.h:493`).

This audit covers **only** the dequantize / convert device functions and
their host launchers. The fused vecdot path that bypasses materialization
(`vecdotq.cuh`, ARTX13) and the MMQ tile-loader path that pre-dequantizes
into shared memory (`mmq-load-tiles.cuh`, ARTX12) are cross-referenced
but not duplicated.

---

## 2. Purpose

Provide the CUDA kernels that materialize a quantized tensor to a
non-quantized dtype, and the kernels that convert between non-quantized
dtypes (F32↔F16↔BF16). Specifically:

* `dequantize_row_<type>_cuda(vx, y, k, stream)` — contiguous 1D
  materialization (used by cuBLAS fallback, the unary op path, and the
  public `ggml_cuda_op_dequantize` entry).
* `dequantize_block_cuda<qk, qr, kernel, dst_t>(vx, y, ne00..ne03,
  s01..s03, stream)` — strided 4D materialization (used by `cpy.cu`
  and `getrows.cu` when the source tensor is not contiguous).
* `convert_unary_cuda<src_t, dst_t>(vx, y, ne00..ne03, s01..s03,
  stream)` — strided 4D conversion between non-quantized dtypes.
* `ggml_get_to_{fp16,fp32,bf16,fp16_nc,bf16_nc,fp32_nc}_cuda(type)` —
  the six host-side dispatch tables that return the right launcher
  function pointer for a given source dtype.
* `quantize_row_q8_1_cuda(x, vy, ne0, ne1, ne2, ne3, stream)` —
  F32→Q8_1 activation quantization (consumed by MMVQ, ARTX09).
* `quantize_mmq_q8_1_cuda(...)` and `quantize_mmq_fp4_cuda(...)` —
  F32→Q8_1_MMQ / F32→block_fp4_mmq for the MMQ path (consumed by MMQ,
  ARTX10/12).

It is **not** responsible for: the fused vecdot path that avoids
materialization (ARTX13), the MMQ tile-loader's in-shared-memory
dequantization (ARTX12), the cuBLAS fallback matmul itself, the
`GET_ROWS` host dispatch, or graph-level fusion decisions (ARTX08).

---

## 3. Source Files

| File                                  | Lines | Role                                                                              |
| ------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/dequantize.cuh`   | 432   | **Primary.** Per-format `dequantize_*` device functions. Q1_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 use the `float2 & v` signature; K-quants and IQ formats use the `dst_t * yy, int tid` signature. Includes `get_scale_min_k4` (K-quant 6-bit scale unpacker). |
| `ggml/src/ggml-cuda/convert.cu`       | 693   | **Primary.** `dequantize_block` template, `dequantize_block_q8_0_f16` fused kernel, per-quant `dequantize_block_q*` specialisations, `convert_unary` template, six `ggml_get_to_*_cuda` dispatch tables, all `dequantize_row_*_cuda` host launchers. |
| `ggml/src/ggml-cuda/convert.cuh`      | 67    | `to_*_cuda_t` typedefs, `ggml_cuda_cast<dst_t>` SFINAE helper, `CUDA_DEQUANTIZE_BLOCK_SIZE = 256` constant. |
| `ggml/src/ggml-cuda/quantize.cu`      | 698   | **Secondary.** `quantize_q8_1` (warp-reduce scale+sum), `quantize_mmq_q8_1` (float4 + shfl_xor), `quantize_mmq_nvfp4` (5-candidate ue4m3 search, Blackwell-only), `quantize_mmq_mxfp4` (per-32 e8m0 scale), `quantize_scatter_mmq_*` (MoE dedup). |
| `ggml/src/ggml-cuda/quantize.cuh`     | (small) | `quantize_row_q8_1_cuda` / `quantize_mmq_q8_1_cuda` / `quantize_mmq_fp4_cuda` / `quantize_scatter_mmq_*_cuda` prototypes. |
| `ggml/src/ggml-cuda/common.cuh`       | 1661  | `dequantize_kernel_t` typedef (`:947`), `dequantize_kq_t<dst_t>` typedef (`:950`), `warp_reduce_max/sum`, `fast_div_modulo`, `fastdiv`. Audited in ARTX08. |
| `ggml/src/ggml-cuda/getrows.cu`       | 487   | Consumer. `k_get_rows` template (`:5-41`) and `k_get_rows_kq` template (`:43-70`) wrap the `dequantize_kernel_t` and `dequantize_kq_t` function pointers respectively. |
| `ggml/src/ggml-cuda/cpy.cu`           | (small) | Consumer. `cpy.cu:112` instantiates `dequantize_kernel_t dequant` inside the `cpy` kernel for fused dequantize+copy. |
| `ggml/src/ggml-common.h`              | 1912  | `GGML_TABLE_BEGIN` macro (`:493`): `static const __device__` qualifier for `kvalues_iq4nl`, `kvalues_mxfp4`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`, `ksigns_iq2xs`, `kmask_iq2xs`. |

> Note: the audit prompt references `convert_f16_f32` and `convert_i8_f32`
> by name. Neither exists literally. F16→F32 conversion is done via
> `convert_unary<half, float>` (instantiated in
> `ggml_get_to_fp32_nc_cuda(GGML_TYPE_F16)` at `convert.cu:675`); int8→F32
> is done via `dequantize_q8_0` (instantiated in
> `ggml_get_to_fp32_cuda(GGML_TYPE_Q8_0)` at `convert.cu:583-584` via
> `dequantize_block_cont_cuda<QK8_0, QR8_0, dequantize_q8_0>`).

---

## 4. Architecture Overview

```
                ┌────────────────────────────────────────────────────────┐
                │  Host dispatch (convert.cu)                            │
                │  ggml_get_to_fp16_cuda(type)  → to_fp16_cuda_t          │
                │  ggml_get_to_fp32_cuda(type)  → to_fp32_cuda_t          │
                │  ggml_get_to_bf16_cuda(type)  → to_bf16_cuda_t          │
                │  ggml_get_to_*_nc_cuda(type)  → to_*_nc_cuda_t (strided)│
                │  (six 22-arm switches, near-duplicate)                 │
                └────────────────────────────────────────────────────────┘
                                       │  function pointer
                                       ▼
                ┌────────────────────────────────────────────────────────┐
                │  Per-quant launcher (convert.cu)                       │
                │  ├─ dequantize_row_<type>_cuda(vx, y, k, stream)       │
                │  │   (1D contiguous, 22 specialisations)               │
                │  ├─ dequantize_block_cuda<qk,qr,kernel,dst_t>(...)     │
                │  │   (4D strided, generic template)                    │
                │  ├─ dequantize_block_q8_0_f16_cuda(vx, y, k, stream)   │
                │  │   (fused Q8_0→F16 fast path, shared-mem staged)     │
                │  └─ convert_unary_cuda<src_t,dst_t>(vx, y, ...)        │
                │      (F32/F16/BF16 ↔ F32/F16/BF16)                     │
                └────────────────────────────────────────────────────────┘
                                       │  kernel launch
                                       ▼
                ┌────────────────────────────────────────────────────────┐
                │  Per-quant device function (dequantize.cuh)            │
                │  ├─ dequantize_q4_0(vx, ib, iqs, float2 & v)           │
                │  │   (simple quants, 2 elements / call)                │
                │  ├─ dequantize_q4_K(vx, ib, dst_t * yy, int tid)       │
                │  │   (K-quants, 256 elements / call, 32 or 64 threads) │
                │  ├─ dequantize_iq2_xxs(vx, ibs, dst_t * yy, int tid)   │
                │  │   (IQ formats, 256 elements / call, 32 threads)     │
                │  └─ dequantize_mxfp4 / dequantize_nvfp4                │
                │      (4-bit FP, 32 elements / call, 32 threads)        │
                └────────────────────────────────────────────────────────┘
                                       │  output write via
                                       │  ggml_cuda_cast<dst_t>(...)
                                       ▼
                                dst_t * y  (F32, F16, or BF16)
```

Key design points:

* **Two device-function signatures.** The simple quants (Q1_0, Q4_0,
  Q4_1, Q5_0, Q5_1, Q8_0) use `void(const void *, int64_t, int,
  float2 &)` (typedef `dequantize_kernel_t` at `common.cuh:947`).
  K-quants and IQ formats use `void(const void *, int64_t, dst_t *,
  int tid)` (typedef `dequantize_kq_t<dst_t>` at `common.cuh:950`).
  The split reflects the block structure: simple quants produce 2
  elements per call; K-quants and IQ formats produce a whole 256-element
  super-block per call (one block of 32 or 64 threads cooperatively).
* **`dst_t` template parameter everywhere.** Every K-quant and IQ
  dequantize function is templated on `dst_t`, and every output write
  goes through `ggml_cuda_cast<dst_t>(float_value)`. This means the
  same device function produces F32, F16, or BF16 output with no F32
  intermediate write — a direct F16 write is half the bandwidth of an
  F32 write followed by a separate F32→F16 conversion.
* **Six dispatch tables.** `ggml_get_to_fp16_cuda`, `_fp32_cuda`,
  `_bf16_cuda`, `_fp16_nc_cuda`, `_bf16_nc_cuda`, `_fp32_nc_cuda`.
  Each is a 22-arm `switch` over `ggml_type` returning a function
  pointer. The `_nc` variants take 4D stride parameters; the
  non-`_nc` variants take a 1D `k` length. The six switches have
  near-identical bodies — only the function-pointer types differ.
* **No shared dispatch table.** Despite the six switches having the
  same 22 cases, there is no `struct { to_fp16, to_fp32, to_bf16,
  to_fp16_nc, to_bf16_nc, to_fp32_nc; } traits[GGML_TYPE_COUNT]`
  table. Each switch is independent.

---

## 5. Execution Flow

### 5.1 Top-level entry: which dispatch table?

The CUDA op dispatch (`ggml-cuda.cu`, audited in ARTX08) calls one of
the six `ggml_get_to_*_cuda(type)` functions based on the destination
dtype of the op (F32, F16, or BF16) and whether the source tensor is
contiguous. The returned function pointer is then called with the
appropriate arguments. The dispatch is **per-op, per-tensor**: there is
no caching of the function pointer across calls.

### 5.2 Inside `dequantize_block` (the generic template)

`dequantize_block<qk, qr, dequantize_kernel, dst_t>` (`convert.cu:8-41`):

1. `i00 = 2 * (blockDim.x * blockIdx.x + threadIdx.x)` — each thread
   owns 2 elements along the innermost dim. `blockDim.x =
   CUDA_DEQUANTIZE_BLOCK_SIZE = 256`, so each block covers 512 elements
   along the innermost dim.
2. Loop `i01` over `blockIdx.y + gridDim.y * k` (rows).
3. Loop `i0203` over `blockIdx.z + gridDim.z * k` (flattened i02, i03;
   unflattened via `fast_div_modulo`).
4. `ib = ibx0 + i00/qk` (block index); `iqs = (i00%qk)/qr` (quant
   index within block).
5. `float2 v; dequantize_kernel(vx, ib, iqs, v);` — call the per-quant
   device function.
6. `y[iy0 + 0] = ggml_cuda_cast<dst_t>(v.x); y[iy0 + y_offset] =
   ggml_cuda_cast<dst_t>(v.y);` — write 2 elements. `y_offset = qr ==
   1 ? 1 : qk/2`.

Block size 256 threads, grid `(ceil(ne00 / 512), min(ne01, 65535),
min(ne02*ne03, 65535))` (`convert.cu:252`).

### 5.3 Inside `dequantize_block_q8_0_f16` (the fused fast path)

`dequantize_block_q8_0_f16<need_check>` (`convert.cu:43-82`):

1. Block size: `WARP_SIZE = 32` threads.
2. Each block processes `CUDA_Q8_0_NE_ALIGN = 2048` elements (= 64
   Q8_0 blocks of 32 elements each).
3. **Shared memory staging**: `__shared__ int vals[nint]` where
   `nint = CUDA_Q8_0_NE_ALIGN/sizeof(int) + WARP_SIZE = 544` ints =
   2176 bytes per block. The `+WARP_SIZE` (32 ints = 128 bytes) is
   padding to avoid bank conflicts during the second loop.
4. **Stage 1** (lines 54-62): cooperative load. Each thread loads
   `nint / WARP_SIZE = 17` ints from `vx` into `vals[]`. The
   `need_check` template parameter guards bounds checking for the
   last block (whose `i0 + 2048` may exceed `k`).
5. `__syncthreads()`.
6. **Stage 2** (lines 66-77): cooperative dequantize + write. Each
   thread reads `2*threadIdx.x`-th half-scale from `vals[]` (the `d`
   field of `block_q8_0`), reads the corresponding `char2 qs` pair,
   and writes `y2[iy/2 + threadIdx.x] = __hmul2(make_half2(qs.x, qs.y),
   __half2half2(d))` — one `__hmul2` per thread per iteration, 32
   elements per warp per iteration, 2048 elements per block.

This is the **only fused quant→F16 kernel** in the file. Other quants
(Q4_0, Q4_K, IQ2_*, etc.) dequantize to F32 or F16 via the generic
`dequantize_block` template, which writes one element at a time (no
`__hmul2` broadcast).

### 5.4 Inside `dequantize_block_q4_0` (the Q4_0 specialisation)

`dequantize_block_q4_0<dst_t>` (`convert.cu:85-110`):

1. Block size: 32 threads (assumed, per comment `// assume 32 threads`).
2. Each grid block processes 8 Q4_0 blocks (= 256 elements). `ib = 8*i +
   ir` where `i = blockIdx.x`, `ir = tid % 8`. So 8 threads cooperatively
   dequantize one Q4_0 block.
3. Within a Q4_0 block: `il = tid / 8`, `q = x->qs + 4*il`. Each thread
   writes 4 lower nibbles + 4 upper nibbles = 8 elements per thread, 32
   elements per block.
4. The output write is `y[l+0] = d * (q[l] & 0xF) + dm; y[l+16] = d *
   (q[l] >> 4) + dm;` where `dm = -8*d`. Same arithmetic as
   `dequantize_q4_0` in `dequantize.cuh:26-38` but with a different
   thread-to-element mapping.

This specialisation exists because the generic `dequantize_block`
template's 2-element-per-thread mapping is suboptimal for Q4_0: Q4_0
blocks are 32 elements / 18 bytes, and the generic template's `iqs`
stride of 1 (since `qr = 2` for Q4_0, `iqs = i00/2`) means each thread
spans half a block. The specialised version packs 8 blocks per grid
block, with 8 threads per block, achieving better register reuse.

### 5.5 Inside the K-quant and IQ launchers

`dequantize_row_q4_K_cuda<dst_t>` (`convert.cu:300-303`):

```cpp
const int nb = k / QK_K;
dequantize_block_q4_K<<<nb, 32, 0, stream>>>(vx, y);
```

`dequantize_block_q4_K<dst_t>` (`convert.cu:156-160`) is a one-line
wrapper: `dequantize_q4_K(vx, i, yy + i*QK_K, threadIdx.x)`. So the
grid is `(nb, 1, 1)`, block is `(32, 1, 1)`, and each block calls the
device function from `dequantize.cuh:166-192`. Similar pattern for
Q2_K, Q3_K, Q5_K, Q6_K, IQ2_*, IQ3_*, IQ1_*, IQ4_* — except:

* Q2_K, Q3_K, Q5_K, Q6_K use **64 threads** per block
  (`convert.cu:276, 282, 308, 314`).
* Q4_K uses **32 threads** (`convert.cu:302`) — matches the
  `// assume 32 threads` comment in `dequantize_q4_K`.
* IQ2_*, IQ3_*, IQ1_*, IQ4_* use **32 threads** (`convert.cu:320, 326,
  332, 338, 344, 350, 356, 362, 368, 374`).

The thread-count policy is dictated by what the device function
assumes (per the `// assume N threads` comments in `dequantize.cuh`).
There is no central table; each launcher hardcodes its own count.

### 5.6 Inside `convert_unary` (non-quantized conversions)

`convert_unary<src_t, dst_t>` (`convert.cu:416-440`):

1. Same 4D strided structure as `dequantize_block`.
2. Each thread owns 1 element (`i00 = blockDim.x*blockIdx.x +
   threadIdx.x`), not 2.
3. `y[iy] = ggml_cuda_cast<dst_t>(x[ix])` — single-element cast via
   the SFINAE helper.

Block size 256, grid `(ceil(ne00 / 256), min(ne01, 65535),
min(ne02*ne03, 65535))` (`convert.cu:448`). Used by the `_nc` dispatch
tables for F32↔F16↔BF16 conversions.

### 5.7 Inside `quantize_q8_1` (F32→Q8_1 activation quantization)

`quantize_q8_1` (`quantize.cu:54-101`):

1. Block size: `CUDA_QUANTIZE_BLOCK_SIZE` (defined in `quantize.cuh`).
   `__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)`.
2. Each thread owns 1 element. `i0 = blockDim.x*blockIdx.x +
   threadIdx.x`. The block strides by `blockDim.x` per thread.
3. `xi = i0 < ne00 ? x[…s03 + s02 + s01 + i00] : 0.0f` (bounds-checked
   load).
4. `amax = warp_reduce_max<QK8_1>(fabsf(xi))` and `sum =
   warp_reduce_sum<QK8_1>(sum)` — `QK8_1 = 32`, so this requires
   `QK8_1 == WARP_SIZE`. The warp reduce functions
   (`common.cuh:455-462`) reduce across `QK8_1` lanes.
5. `d = amax / 127.0f; q = roundf(xi / d)`.
6. `y[ib].qs[iqs] = q` — single int8 write.
7. **Only thread `iqs == 0` writes the scale+sum**: `y[ib].ds =
   make_half2(d, sum)`. The other 31 threads return early.

The `QK8_1 == WARP_SIZE` assumption is hardcoded: the warp-reduce
template is parameterised on `QK8_1`, not `WARP_SIZE`. On AMD (where
`WARP_SIZE = 64`), this kernel would need to be reworked.

### 5.8 Inside `quantize_mmq_q8_1` (float4 + shfl_xor reduce)

`quantize_mmq_q8_1<ds_layout, scatter>` (`quantize.cu:457-556`):

1. Each thread loads `float4` (4 elements) per iteration.
2. `amax = fmaxf(…)` across the 4 lanes, then `__shfl_xor_sync` reduce
   across `vals_per_scale/8` threads (= 8 for D4/DS4, 16 for D2S6).
3. `sum = xi.x + xi.y + xi.z + xi.w`, then `__shfl_xor_sync` reduce
   across `vals_per_sum/8` threads (= 8 for D4, 4 for DS4, 2 for D2S6).
4. `q = roundf(xi * d_inv)` where `d_inv = 127.0f / amax`.
5. Write 4 int8s as a single `char4` (32-bit write).
6. Per-block scale+sum written by one thread per `vals_per_scale`/
   `vals_per_sum` group, in one of three layouts:
   * `D4`: 4 floats (one `d` per 32 elements, no sum).
   * `DS4`: 4 `half2` (one `(d, s)` per 32 elements).
   * `D2S6`: 2 floats + 6 floats (one `d` per 64 elements, one `s` per
     16 elements).

Layout selection: `mmq_get_q8_1_ds_layout(type_src0)` (in
`quantize.cuh`) returns the right enum based on the weight format.

---

## 6. Data Layout

### 6.1 Source tensor (quantized)

The dequantize device functions read from `const void * vx`, which is
the raw quantized tensor data. Block layouts are the same as in ARTX13
§6.1 (Q4_0: 18 bytes/block, Q4_K: 144 bytes/super-block, etc.). The
device functions cast `vx` to the right `block_*` type and index by
`ib` (block index) or `ibs` (super-block index).

### 6.2 Destination tensor

The destination is `dst_t * y` where `dst_t ∈ {float, half,
nv_bfloat16}`. The K-quant and IQ device functions write a full
`QK_K = 256`-element slice per call, indexed as `yy + 128*n + 32*j + l`
(see `dequantize_q2_K:108-124`). The simple-quant device functions
write 2 elements per call via the `float2 & v` out-parameter, indexed
by the caller (`dequantize_block` template, `convert.cu:36-38`).

### 6.3 Strided vs contiguous

The non-`_nc` dispatch tables (`ggml_get_to_*_cuda`) call
`dequantize_row_*_cuda(vx, y, k, stream)` which is 1D contiguous. The
`_nc` dispatch tables (`ggml_get_to_*_nc_cuda`) call
`dequantize_block_cuda<qk, qr, kernel, dst_t>(vx, y, ne00, ne01, ne02,
ne03, s01, s02, s03, stream)` which is 4D strided. The strided path
uses `fast_div_modulo` to unflatten `i0203 = i02 + i03 * ne02` back to
`(i02, i03)` (avoids the integer divide in the inner loop).

### 6.4 Q8_1 output layouts (quantize.cu)

Three layouts selected by `mmq_get_q8_1_ds_layout(type_src0)`:

* `MMQ_Q8_1_DS_LAYOUT_D4` — 4 × float `d` per 128 elements (no sum).
  Used by Q8_0 weights (the weight's own `d` is enough; activation
  sum is not needed for symmetric Q8_0).
* `MMQ_Q8_1_DS_LAYOUT_DS4` — 4 × half2 `(d, s)` per 128 elements. Used
  by Q4_0, Q5_0 (symmetric quants that need the `s` for the bias fold
  — see ARTX13-F03).
* `MMQ_Q8_1_DS_LAYOUT_D2S6` — 2 × float `d` + 6 × float `s` per 128
  elements. Used by Q4_K, Q5_K, Q2_K (asymmetric quants that need
  finer-grained sums for the min fold — see ARTX13-F04).

---

## 7. Memory Layout

### 7.1 Shared memory

Only two kernels in this audit use shared memory:

* `dequantize_block_q8_0_f16` (`convert.cu:43-82`): `__shared__ int
  vals[544]` = 2176 bytes per block. Used for staging the input Q8_0
  block so that the `__hmul2` epilogue can read the scale `d` from
  shared memory and broadcast it across the warp.
* `quantize_mmq_nvfp4` (`quantize.cu:128-332`): `__shared__ float
  warp_amax[CUDA_QUANTIZE_BLOCK_SIZE_MMQ / WARP_SIZE]` for cross-warp
  amax reduction (lines 169-194).

No other dequantize or quantize kernel uses shared memory. The K-quant
and IQ device functions are pure-register; the simple-quant device
functions are pure-register.

### 7.2 Lookup tables in read-only data memory

The IQ2/IQ3/IQ1/IQ4 lookup tables (`iq2xxs_grid`, `iq2xs_grid`,
`iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`,
`ksigns_iq2xs`, `kmask_iq2xs`, `kvalues_iq4nl`, `kvalues_mxfp4`) live
in `ggml-common.h:493` as `static const __device__ type name[size]`.
They are NOT explicitly `__constant__`-qualified; on NVIDIA they back
by read-only data memory (the L1 read-only cache, accessed via
`__ldg`-style loads). The compiler is free to use `__ldg` for these
reads since the `__device__ const` qualifier implies read-only.

The largest table is `iq2s_grid` at 1024 × 8 = 8 KB. The 64 KB
constant memory is reserved for kernel parameters and would not fit
these tables anyway. The L1 read-only cache (48 KB per SM on Ampere)
is the correct backing store.

### 7.3 Output write coalescing

For `dequantize_block` (generic template), each thread writes 2
consecutive `dst_t` elements. For `dst_t = float`, that's an 8-byte
write per thread, 256 threads per block = 2048 bytes per block, fully
coalesced. For `dst_t = half`, that's a 4-byte write per thread,
also fully coalesced.

For `dequantize_block_q4_0` (Q4_0 specialisation), each thread writes
8 elements at stride `4*il` and `4*il + 16` — not fully coalesced
(strided pattern). The 8-thread-per-block mapping means 8 threads
write 32 consecutive elements, then the next 8 threads write the next
32, etc. — coalesced within each 32-element group.

For K-quant and IQ device functions, the output write pattern is
dictated by the per-format thread mapping (e.g., `dequantize_q2_K`
writes `y[l], y[l+32], y[l+64], y[l+96]` per thread — stride-32
across threads, which is coalesced for `dst_t = float` and 2-byte-
coalesced for `dst_t = half`).

---

## 8. Parallelism Strategy

### 8.1 Generic template: 2 elements per thread, 256 threads per block

`dequantize_block` (the generic template) uses 256 threads per block,
each thread owning 2 elements. The grid is `(ceil(ne00 / 512),
min(ne01, 65535), min(ne02*ne03, 65535))`. The 65535 cap on Y and Z
is the CUDA grid-dimension limit (pre-CC3.0); for larger tensors the
loop in the kernel strides by `gridDim.y` and `gridDim.z` to cover
all rows.

### 8.2 K-quant and IQ: 32 or 64 threads per super-block

The K-quant and IQ launchers use one block per super-block (one block
per `QK_K = 256` elements). Block size is 32 or 64 threads depending
on the format (see §5.5). The grid is `(nb, 1, 1)` where `nb = k /
QK_K`.

For multi-row tensors, the launchers are called per-row (i.e., the host
dispatch loops over rows). This is suboptimal for short rows — each row
gets a separate kernel launch. The strided `_nc` variants of the
dispatch tables use the generic `dequantize_block` template instead,
which handles 4D strided inputs in a single launch.

### 8.3 Fused Q8_0→F16: 32 threads per block, 2048 elements per block

`dequantize_block_q8_0_f16` uses 32 threads per block, each block
covering 2048 elements. The grid is `(ceil(k / 2048), 1, 1)`. The
2-stage structure (stage 1: load to shared mem; stage 2: dequantize +
write) lets the second stage use `__hmul2` for the scale broadcast,
which requires the scale to be in registers (loaded from shared mem).

### 8.4 Quantize kernels: warp-cooperative reduction

`quantize_q8_1` uses one warp per `QK8_1 = 32` elements. Each thread
in the warp owns one element, computes `amax` and `sum` via
`warp_reduce_max/sum<QK8_1>`, and writes its own int8. Only thread 0
writes the scale+sum.

`quantize_mmq_q8_1` uses one block per `(token, k_block)` pair. Each
thread loads `float4` (4 elements), reduces across the warp via
`__shfl_xor_sync`, and writes 4 int8s + (if it's the first thread in
its group) the scale+sum. The block size is
`CUDA_QUANTIZE_BLOCK_SIZE_MMQ` (defined in `quantize.cuh`).

### 8.5 No inter-block communication

None of the kernels in this audit use cooperative groups, atomics, or
cross-block reduction. Each block is independent. This is by design —
the per-block work is large enough (32-2048 elements) that launch
overhead is amortised.

---

## 9. GPU Strategy

### 9.1 No `dp4a`, no Tensor Cores

Unlike `vecdotq.cuh` (ARTX13), the dequantize kernels do no arithmetic
reduction. They are pure elementwise conversions (read packed bytes,
multiply by scale, write `dst_t`). The only "SIMD" primitives used are
`__hmul2` (in `dequantize_block_q8_0_f16`) and `__half22float2` (in
many K-quant / IQ device functions for scale unpacking).

### 9.2 `__launch_bounds__` policy

* `quantize_q8_1`: `__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)`
  (1 block per SM; trades occupancy for register headroom — same
  pattern as MMVQ, ARTX09 §9.4).
* `dequantize_block` template: no explicit `__launch_bounds__`.
* `dequantize_block_q8_0_f16`: no explicit `__launch_bounds__`. Block
  size is 32 threads, so occupancy is naturally high.
* `quantize_mmq_q8_1`, `quantize_mmq_nvfp4`, `quantize_mmq_mxfp4`: no
  explicit `__launch_bounds__`.

### 9.3 Per-arch gating

* `dequantize_block_q8_0_f16` is gated by `#if __CUDA_ARCH__ >=
  GGML_CUDA_CC_PASCAL` (`convert.cu:45`). Pre-Pascal NVIDIA falls
  through to `NO_DEVICE_CODE` (compile-time error if instantiated).
* `quantize_mmq_nvfp4` is gated by `#if defined(BLACKWELL_MMA_AVAILABLE)`
  (`quantize.cu:132`). Non-Blackwell falls through to `NO_DEVICE_CODE`.
* `quantize_mmq_mxfp4` has no arch gate but uses `__nv_fp4x4_e2m1` if
  `CUDART_VERSION >= 12080`, otherwise falls back to a manual LUT
  conversion (`quantize.cu:411-423`).

### 9.4 PDL integration

* `quantize_q8_1` calls `ggml_cuda_pdl_sync()` and
  `ggml_cuda_pdl_lc()` (`quantize.cu:58, 83`). The `_lc()` is called
  early (line 58, before the main work); the `_sync()` is called
  mid-kernel (line 83, after the load). This is the standard PDL
  pattern from ARTX08 §9.4.
* `quantize_mmq_q8_1` calls `ggml_cuda_pdl_sync()` only
  (`quantize.cu:473`).
* `quantize_mmq_mxfp4` calls `ggml_cuda_pdl_sync()` (`quantize.cu:371`).
* No dequantize kernel in `convert.cu` calls PDL. The dequantize kernels
  are short-running and not in the critical path of overlapping kernels.

---

## 10. Quantization Strategy

### 10.1 Scale handling — three patterns (mirrors ARTX13 §10.1)

| Pattern | Quants | Where scale applied |
| ------- | ------ | ------------------- |
| Symmetric, post-multiply, bias-subtract | Q4_0, Q5_0, Q8_0, Q1_0 | `v.x = (q - 8.0f) * d;` (or `(2*bit - 1) * d` for Q1_0) |
| Asymmetric, post-multiply, min-add | Q4_1, Q5_1 | `v.x = (q * dm.x) + dm.y;` where `dm = (d, m)` |
| Per-32-element scale + min | Q4_K, Q5_K | `y[l] = d1 * (q & 0xF) - m1;` |
| Per-32-element scale, no min | Q2_K, Q3_K, Q6_K | `y[l] = d * sc * (q - bias);` |
| Per-sub-block grid + scale | IQ2_*, IQ3_*, IQ1_* | `y[j] = d * grid[j] * (signs & mask ? -1 : 1);` |
| Per-block FP scale | MXFP4, NVFP4 | `y[j] = d * kvalues_mxfp4[q] * 0.5f;` |

### 10.2 `dst_t` template: F32, F16, or BF16 directly

Every K-quant and IQ dequantize function is templated on `dst_t`. Every
output write goes through `ggml_cuda_cast<dst_t>(float_value)` (defined
in `convert.cuh:34-66`). The cast is a SFINAE-style `if constexpr`
chain:

* `dst_t == src_t`: identity.
* `dst_t == nv_bfloat16`: `__float2bfloat16(float(x))`.
* `src_t == nv_bfloat16`: `__bfloat162float(x)`.
* `src_t == float2, dst_t == half2`: `__float22half2_rn(x)`.
* `src_t == nv_bfloat162, dst_t == float2`: arch-dependent
  (`__bfloat1622float2` on Ampere+, else scalar).
* `dst_t == int32_t`: `int32_t(x)`.
* Fallback: `float(x)`.

This means the same device function produces F32, F16, or BF16 output
with no F32 intermediate write. The F16 path is half the bandwidth of
the F32 path, with a single `__float2half` instruction per element.

### 10.3 `get_scale_min_k4` — the canonical K-quant scale unpacker

`dequantize.cuh:157-164`:

```cpp
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q,
                                                uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
```

This is the canonical K-quant 6-bit scale unpacker — the third of
three implementations (see ARTX13-F06 for the other two in
`vecdotq.cuh` and `mmq-load-tiles.cuh`). It is shared by
`dequantize_q4_K` and `dequantize_q5_K`. The Q3_K and Q2_K device
functions have their own inline unpackers (`dequantize.cuh:141-144`
for Q3_K, `dequantize.cuh:120-123` for Q2_K).

### 10.4 NVFP4 scale quantization (5-candidate search)

`quantize_mmq_nvfp4` (`quantize.cu:128-332`) is the only quantizer in
this audit that does a per-sub-block mini-grid-search for the FP8
scale:

1. Compute `amax_sub` across 16 elements of the sub-block.
2. Convert `amax_sub / 6.0f` to a first ue4m3 candidate via
   `ggml_cuda_fp32_to_ue4m3`.
3. Try 5 candidate codes: `{first, first-1, first+1, first-2, first+2}`
   (clamped to `[0, 0x7e]`).
4. For each candidate, compute `nvfp4_native_scale_error` (CUDA 12.8+
   uses native `__nv_fp4x4_e2m1` intrinsics; otherwise falls back to
   a manual LUT-based error).
5. Pick the candidate with the lowest SSE.

This is a per-sub-block optimisation that the CPU quantizer
(`ggml-quants.c`) also does. The result is bit-identical to the CPU
path (modulo the FP rounding mode).

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions with correctness
consequences.

### 11.1 Floating-point reassociation

* **Per-element scale multiply.** Each output element is computed as
  `d * (q - bias)` or `d * q + m` in a single FMA. No cross-element
  reduction, so no reassociation within a dequantize call.
* **K-quant multi-scale reduction.** `dequantize_q4_K` computes
  `d1 * (q & 0xF) - m1` and `d2 * (q >> 4) - m2` separately, then
  writes both. The two are independent — no reassociation.
* **`ggml_cuda_cast` rounding.** The F32→F16 cast uses
  `__float2half` (round-to-nearest-even). The F32→BF16 cast uses
  `__float2bfloat16` (round-to-nearest-even). Both are deterministic.

### 11.2 Quantization rounding

* **`quantize_q8_1` rounding.** `q = roundf(xi / d)` where `d =
  amax / 127.0f`. Round-to-nearest-even via `roundf`. The result is
  deterministic per warp (the warp-reduce for `amax` is
  butterfly-deterministic).
* **`quantize_mmq_q8_1` rounding.** Same `roundf(xi * d_inv)` pattern.
  The `d_inv = 127.0f / amax` is computed per-group; adjacent groups
  may have different `d_inv` values, so the rounding is per-group
  independent.
* **`quantize_mmq_nvfp4` rounding.** The 5-candidate scale search
  produces a scale that minimises SSE. The actual FP4 conversion uses
  `__nv_fp4x4_e2m1` (CUDA 12.8+) or `ggml_cuda_float_to_fp4_e2m1`
  (LUT fallback). Both round to nearest.

### 11.3 Bounds checking

* **`dequantize_block` generic template.** Each thread checks `i00 >=
  ne00` and returns early. The grid is sized for `ceil(ne00 / 512)`,
  so the last block may have threads that exit early.
* **`dequantize_block_q8_0_f16`.** The `need_check` template parameter
  enables bounds checking in the last block (when `k %
  CUDA_Q8_0_NE_ALIGN != 0`). The host launcher (`convert.cu:262-271`)
  picks `need_check = true` if `k % 2048 != 0`, else `need_check =
  false` for the fast path.
* **`dequantize_block_q4_0` and `_q4_1`.** Each thread checks `ib >=
  nb32` and returns early (`convert.cu:94-96, 122-124`).
* **K-quant and IQ launchers.** No bounds checking. The grid is sized
  for `nb = k / QK_K`, so `k` must be a multiple of `QK_K`. The
  launcher asserts this implicitly (no explicit assert; just the
  truncating divide).

### 11.4 Architecture-specific assumptions

* `dequantize_block_q8_0_f16` requires Pascal+ (`convert.cu:45`). The
  `__hmul2` intrinsic is available pre-Pascal but the kernel structure
  relies on `__half2half2` which is Pascal+.
* `quantize_q8_1` assumes `QK8_1 == WARP_SIZE`. On AMD (WARP_SIZE=64),
  this kernel would need rework. The current code is NVIDIA-only (the
  `warp_reduce_max<QK8_1>` template would produce incorrect results
  on AMD because it reduces across only 32 of the 64 lanes).
* `quantize_mmq_nvfp4` requires Blackwell (`BLACKWELL_MMA_AVAILABLE`).
  Pre-Blackwell falls through to `NO_DEVICE_CODE` (compile-time error
  if instantiated).

### 11.5 No atomic accumulation

None of the kernels in this audit use atomics. Each output element is
written by exactly one thread; no race conditions.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                  | Where                                | Notes                                                                                  |
| --------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| `dst_t` template parameter                    | every K-quant and IQ dequantize      | Same source produces F32, F16, or BF16; no F32 intermediate write.                     |
| `ggml_cuda_cast<dst_t>` SFINAE helper         | `convert.cuh:34-66`                  | One template handles every src/dst type pair via `if constexpr`.                       |
| Fused Q8_0→F16 kernel with shared-mem staging | `convert.cu:43-82`                   | Only fused quant→F16 path; `__hmul2` broadcast for scale; 2-stage cooperative load.    |
| Q4_0/Q4_1 specialised block kernels           | `convert.cu:85-137`                  | 8 blocks per grid block, 8 threads per block; better register reuse than generic template. |
| `fast_div_modulo` for strided 4D dispatch     | `common.cuh:940-945`, used in `convert.cu:21, 431` | Avoids integer divide in inner loop; uses Barrett reduction via `fastdiv`. |
| `__shfl_xor_sync` for warp-reduce in quantize  | `quantize.cu:501, 511, 393`          | Butterfly reduce for amax and sum; same pattern as MMVQ (ARTX09).                      |
| `float4` loads in `quantize_mmq_q8_1`         | `quantize.cu:485, 492`               | 4 elements per thread per load; coalesced 16-byte transactions.                        |
| `__launch_bounds__(…, 1)` for `quantize_q8_1` | `quantize.cu:53`                     | 1 block per SM; trades occupancy for register headroom.                                |
| `char4` packed write for 4 int8s              | `quantize.cu:537-538`                | Single 32-bit write for 4 elements; better bandwidth than 4 byte writes.               |
| 5-candidate ue4m3 scale search for NVFP4      | `quantize.cu:233-279`                | Mini-grid-search; picks lowest-SSE scale per sub-block.                                 |
| `ggml_cuda_pdl_sync()` / `_lc()` in quantize  | `quantize.cu:58, 83, 371, 473`       | Hopper+ PDL overlap with downstream kernel.                                            |
| Per-thread `iqs > 0` early-exit in `quantize_q8_1` | `quantize.cu:96-98`              | Only thread 0 writes the scale+sum; other 31 threads exit early after writing their int8. |
| `dst_t` template instantiation for F16 path   | every K-quant / IQ dequantize        | Direct F16 write saves 2× bandwidth vs F32 write + separate F32→F16 conversion.        |

### 12.2 Optimizations *not* present

* **No fused dequantize + RoPE kernel.** RoPE is always applied
  post-dequantization in a separate kernel. The K-quants and IQ formats
  that are commonly used for KV cache (Q4_K, Q8_0, etc.) could benefit
  from a fused dequantize+RoPE kernel that avoids materializing the
  dequantized tensor.
* **No `__constant__` qualification on lookup tables.** The tables
  (`kvalues_iq4nl`, `iq2xxs_grid`, etc.) are `static const __device__`,
  not `__constant__`. The compiler may or may not place them in
  constant memory; explicit `__constant__` would force it. For 16-byte
  tables like `kvalues_iq4nl`, constant memory is faster than L1
  read-only cache.
* **No `cp.async` prefetching.** The dequantize kernels read weight
  blocks directly via `vx[ib]`. Async copy from global to shared is
  done in the MMQ loader (ARTX12), not here. For `dequantize_block_q8_0_f16`,
  the shared-memory staging uses synchronous loads (`vals[ix] = x0[ix]`).
* **No persistent kernel.** Each dequantize call is a separate kernel
  launch. For graphs with many small dequantize ops (e.g., MoE with
  per-expert weights), a persistent kernel that processes multiple
  tensors in one launch would reduce launch overhead.
* **No batched dequantize.** Each `dequantize_row_*_cuda` call handles
  one row. The strided `_nc` variants handle 4D inputs but still one
  tensor at a time.
* **No Q4_0→F16 / Q4_K→F16 fused fast paths.** Only Q8_0 has a fused
  quant→F16 kernel. Q4_0, Q4_K, etc. go through the generic
  `dequantize_block` template, which writes one element at a time.

---

## 13. Architectural Strengths

1. **`dst_t` template is the single best design decision.** Every
   K-quant and IQ dequantize function is templated on the output dtype,
   so the same source produces F32, F16, or BF16 with no F32
   intermediate. Direct F16 write is half the bandwidth of F32 write +
   separate conversion — a 2× bandwidth win for F16-output paths
   (KV cache, F16 matmul fallback).

2. **`dequantize_kernel_t` typedef is a clean contract.** The simple
   quants (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) share a single
   function signature `void(const void *, int64_t, int, float2 &)`,
   which is consumed by `dequantize_block`, `k_get_rows`, and the
   `cpy.cu` copy-fuser. Adding a new simple quant = adding one
   function + one switch case.

3. **`dequantize_block_q8_0_f16` is a model fused kernel.** 2-stage
   cooperative load (stage 1: load to shared mem; stage 2: dequantize
   + write with `__hmul2`), template-parameterised bounds check
   (`need_check`), 32-thread block for natural warp alignment. The
   only fused quant→F16 path in the file.

4. **`ggml_cuda_cast<dst_t>` is a clean SFINAE helper.** One template
   handles every (src, dst) type pair via `if constexpr`. No
   macro-based dispatch, no virtual functions, no runtime branching.

5. **Q4_0 / Q4_1 specialised block kernels.** The generic
   `dequantize_block` template's 2-element-per-thread mapping is
   suboptimal for Q4_0 (where each block is only 32 elements). The
   specialised `dequantize_block_q4_0` packs 8 blocks per grid block
   with 8 threads per block — better register reuse, fewer blocks
   launched.

6. **NVFP4 scale quantization does a 5-candidate search.** The
   per-sub-block ue4m3 scale is the single biggest determinant of
   NVFP4 accuracy. The 5-candidate search (offsets `{-2, -1, 0, +1,
   +2}` from the first guess) picks the lowest-SSE scale, matching
   the CPU quantizer's behaviour.

7. **`fast_div_modulo` for strided 4D dispatch.** The strided `_nc`
   variants use Barrett reduction (`fastdiv`) to avoid integer divide
   in the inner loop. The `init_fastdiv_values(ne02)` precomputes
   the magic numbers once per launch.

---

## 14. Architectural Weaknesses

### W1 — Six near-duplicate dispatch tables

**Evidence**: `convert.cu:458-693` — `ggml_get_to_bf16_cuda`,
`_fp16_cuda`, `_fp32_cuda`, `_fp16_nc_cuda`, `_bf16_nc_cuda`,
`_fp32_nc_cuda`. Each is a 22-arm `switch` over `ggml_type` returning
a function pointer. The first three (non-`_nc`) are 50 lines each; the
last three (`_nc`) are 25 lines each. Total ~250 lines of near-duplicate
code.

**Impact**: Adding a new quant format = adding 6 cases (one per
table). Forgetting one is a silent bug (the format falls through to
`nullptr`, which is a link error or runtime crash depending on the
caller). A single `struct { to_fp16, to_fp32, to_bf16, to_fp16_nc,
to_bf16_nc, to_fp32_nc; } traits[GGML_TYPE_COUNT]` table would be one
case per format.

### W2 — K-quant 6-bit scale unpacker duplicated three times

**Evidence**: `dequantize.cuh:157-164` (`get_scale_min_k4`, the
canonical version), `vecdotq.cuh:890-899` (MMVQ Q4_K inline), and
`mmq-load-tiles.cuh:612-620` (`unpack_scales_q45_K`, MMQ loader). All
three decode the same 6-bit packed layout. See ARTX13-F06 for full
analysis.

**Impact**: Triple-maintained bit manipulation. Bug fixes must be
applied in three places. None is canonical — the layouts are
duplicated knowledge.

### W3 — Per-quant block thread-count policy is per-launcher

**Evidence**: `convert.cu:276, 282, 302, 308, 314, 320, 326, 332, 338,
344, 350, 356, 362, 368, 374`. Each `dequantize_row_*_cuda` hardcodes
its own block size (32 or 64). The policy is dictated by what the
device function assumes (per `// assume N threads` comments in
`dequantize.cuh`).

**Impact**: No central table mapping format → block size. A new format
= a new launcher with a new hardcoded count. If the device function's
assumption changes, the launcher must be updated separately.

### W4 — No fused dequantize + RoPE kernel

**Evidence**: No `dequantize_rope` or `rope_dequant` function exists
anywhere in the CUDA backend (grep returns no matches). RoPE is
applied in `rope.cu` as a separate kernel that reads F32/F16 input
and writes F32/F16 output.

**Impact**: For quantized KV cache (Q4_K, Q8_0, etc.), the dequantize
→ RoPE → write-back path materializes the full F32 tensor in memory,
then reads it back for RoPE. A fused kernel would halve the memory
traffic. Especially costly for long-context inference where the KV
cache is large.

### W5 — Lookup tables not explicitly `__constant__`

**Evidence**: `ggml-common.h:493` — `#define GGML_TABLE_BEGIN(type,
name, size) static const __device__ type name[size] = {`. The tables
are `__device__ const`, not `__constant__`. The compiler may place
them in L1 read-only cache (via `__ldg`) or in global memory; the
placement is not guaranteed.

**Impact**: For small tables (`kvalues_iq4nl` = 16 bytes,
`kvalues_mxfp4` = 16 bytes), `__constant__` would be faster (constant
cache is broadcast-capable, 1 cycle per warp). For large tables
(`iq2s_grid` = 8 KB), `__constant__` would not fit (64 KB limit
shared with kernel parameters). A per-table policy would be optimal.

### W6 — `quantize_q8_1` assumes `QK8_1 == WARP_SIZE`

**Evidence**: `quantize.cu:88-89` — `amax = warp_reduce_max<QK8_1>(amax);
sum = warp_reduce_sum<QK8_1>(sum);`. The warp-reduce template is
parameterised on `QK8_1 = 32`, not `WARP_SIZE`. On AMD (where
`WARP_SIZE = 64`), this would reduce across only 32 of 64 lanes,
producing incorrect results.

**Impact**: Hardcoded NVIDIA assumption. The kernel is currently
NVIDIA-only (the build system gates it), but the assumption is
implicit, not documented.

### W7 — No Q4_0→F16 / Q4_K→F16 fused fast paths

**Evidence**: `convert.cu:43-82` — only `dequantize_block_q8_0_f16`
exists. Q4_0, Q4_K, IQ2_*, etc. go through the generic
`dequantize_block` template, which writes one element at a time (no
`__hmul2` broadcast). For Q4_0 specifically, the hand-specialised
`dequantize_block_q4_0` (lines 85-110) writes 8 elements per thread
but still uses scalar `ggml_cuda_cast<dst_t>(v.x)` writes.

**Impact**: 2-4× bandwidth waste for F16-output paths on Q4_0/Q4_K
vs a hypothetical fused kernel. The Q8_0 fused kernel exists because
Q8_0 is the simplest case (no bit unpacking, just int8 × scale); the
other formats would need more complex fused kernels.

### W8 — `dequantize_row_*_cuda` is per-row, not batched

**Evidence**: `convert.cu:273-375` — every `dequantize_row_*_cuda` is
`dequantize_block_q*<<<nb, 32, 0, stream>>>(vx, y)`, where `nb = k /
QK_K`. The host dispatch loops over rows for multi-row tensors.

**Impact**: For a `[ne01, ne00]` tensor with `ne01 = 4096` rows, that's
4096 separate kernel launches. Each launch has ~5µs overhead on
NVIDIA, so ~20 ms of pure launch overhead. The strided `_nc` variants
avoid this by handling 4D inputs in a single launch, but the non-`_nc`
path is per-row.

### W9 — `convert_unary` is 1 element per thread, not vectorized

**Evidence**: `convert.cu:421` — `i00 = blockDim.x*blockIdx.x +
threadIdx.x`. Each thread owns 1 element. For `src_t = float, dst_t =
half`, that's a 4-byte read + 2-byte write per thread, unvectorized.

**Impact**: 2-4× slower than a `float4`-vectorised version. The
`quantize_mmq_q8_1` kernel uses `float4` loads (line 485); the
`convert_unary` kernel does not.

### W10 — `__half22float2` used for scale unpacking in many K-quant paths

**Evidence**: `dequantize.cuh:118-119, 178-179, 205-206` — every K-quant
dequantize function unpacks the `half2 dm` to `float2` via
`__half22float2`. This is a 2-instruction sequence (`__low2half` +
`__high2half` + 2 `__half2float`).

**Impact**: Negligible per-call, but adds up over many calls. An
alternative would be to keep the scale as `half2` and use `__hmul2`
for the multiply, converting to `float` only at the final write. The
trade-off is F16 vs F32 precision for the intermediate scale product.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda` | **ADOPT** | `dst_t` template parameter on every dequantize function | Direct F16/BF16 output, no F32 intermediate. 2× bandwidth win for F16 paths. |
| `glcuda` | **ADOPT** | `dequantize_kernel_t` typedef as the simple-quant contract | One signature shared by `dequantize_block`, `k_get_rows`, `cpy.cu` copy-fuser. |
| `glcuda` | **ADOPT** | `ggml_cuda_cast<dst_t>` SFINAE helper | One template handles every (src, dst) type pair via `if constexpr`. |
| `glcuda` | **ADOPT** | Fused Q8_0→F16 kernel with shared-mem staging | Model for any future fused quant→F16 path. |
| `glcuda` | **ADOPT** | `fast_div_modulo` for strided 4D dispatch | Barrett reduction avoids integer divide in inner loop. |
| `glcuda` | **ADAPT** | Six dispatch tables → one traits table | Replace 6 switches with one `traits[GGML_TYPE_COUNT]` struct. |
| `glcuda` | **ADAPT** | K-quant scale unpacker → single canonical function | Consolidate `get_scale_min_k4` + 2 re-implementations (ARTX13-F06). |
| `glcuda` | **ADAPT** | Per-quant block thread-count policy → central table | Replace per-launcher hardcoded counts with a `traits[GGML_TYPE_COUNT].block_size` field. |
| `glcuda` | **REJECT** | Per-row `dequantize_row_*_cuda` launches | Use the strided `_nc` variant for multi-row tensors to avoid launch overhead. |
| `glcuda` | **REJECT** | No fused dequantize+RoPE kernel | Add a fused kernel for quantized KV cache; halves memory traffic. |
| `glcuda` | **MONITOR** | Lookup table placement (`__device__` vs `__constant__`) | Profile per-table; small tables to `__constant__`, large to `__device__` with `__ldg`. |
| `glcuda` | **MONITOR** | `quantize_q8_1` `QK8_1 == WARP_SIZE` assumption | Document explicitly; rework for AMD if glcuda targets RDNA. |
| `glcuda` | **DEFER** | NVFP4 5-candidate scale search | Defer until glcuda has basic quantize working; the search is correctness-relevant but not perf-critical. |

---

## 16. Recommendations

### R1 — ADOPT `dst_t` template parameter on every dequantize function
**Priority:** Critical **Difficulty:** M **Dependencies:** none
GwenLand's `glcuda` should template every K-quant and IQ dequantize
function on `dst_t`. Every output write goes through `gl_cuda_cast<dst_t>`.
The same source produces F32, F16, or BF16 with no F32 intermediate.

### R2 — ADOPT `dequantize_kernel_t` typedef as the simple-quant contract
**Priority:** Critical **Difficulty:** S **Dependencies:** R1
Define `typedef void (*gl_dequantize_kernel_t)(const void *, int64_t,
int, float2 &);` for the simple quants (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1,
Q8_0). Use it as the function-pointer type for the generic
`dequantize_block` template, `k_get_rows`, and the copy-fuser.

### R3 — ADOPT fused Q8_0→F16 kernel with shared-mem staging
**Priority:** High **Difficulty:** M **Dependencies:** R1, R2
For Q8_0 → F16, implement a 2-stage cooperative kernel: stage 1 loads
the input block to shared memory; stage 2 dequantizes + writes with
`__hmul2` for the scale broadcast. Template-parameterise the bounds
check (`need_check`). Block size = `WARP_SIZE` (32 threads).

### R4 — ADAPT: consolidate six dispatch tables into one traits table
**Priority:** High **Difficulty:** M **Dependencies:** R1, R2
Replace the six `ggml_get_to_*_cuda` switches with one
`struct gl_cuda_quant_traits { to_fp16, to_fp32, to_bf16, to_fp16_nc,
to_bf16_nc, to_fp32_nc; } traits[GGML_TYPE_COUNT]` table. One entry
per quant format.

### R5 — ADAPT: unify K-quant scale unpacker
**Priority:** High **Difficulty:** M **Dependencies:** R1, ARTX13-F06
Make `get_scale_min_k4` (or a successor) the single canonical K-quant
6-bit scale unpacker. All three call sites (dequantize, vecdot MMVQ,
vecdot MMQ loader) call this one function. See ARTX13-F06 for the
three current implementations.

### R6 — REJECT per-row `dequantize_row_*_cuda` launches for multi-row tensors
**Priority:** High **Difficulty:** M **Dependencies:** R1
For multi-row tensors, use the strided `_nc` variant in a single
launch. Reserve the per-row `dequantize_row_*_cuda` for true 1D
contiguous inputs.

### R7 — REJECT absence of fused dequantize+RoPE; add it
**Priority:** High **Difficulty:** L **Dependencies:** R1, GATE design
Add a fused `dequantize_rope_q*_cuda` kernel for the quantized KV
cache path. The kernel reads quantized K/V, dequantizes to F32 in
registers, applies RoPE in registers, writes F32/F16 to the KV cache.
Halves memory traffic vs dequantize+RoPE as separate kernels.

### R8 — ADOPT `fast_div_modulo` for strided 4D dispatch
**Priority:** Medium **Difficulty:** S **Dependencies:** R1
Use Barrett reduction (`fastdiv`) for the `i0203 → (i02, i03)`
unflatten in strided 4D kernels. Avoids integer divide in the inner
loop.

### R9 — MONITOR lookup table placement
**Priority:** Low **Difficulty:** S **Dependencies:** R1
For small tables (`kvalues_iq4nl`, `kvalues_mxfp4`, `kmask_iq2xs`:
≤ 128 bytes), qualify as `__constant__` for broadcast-capable constant
cache. For large tables (`iq2s_grid`, `iq3s_grid`: multi-KB), keep as
`__device__ const` with `__ldg`-style access. Profile per-table to
confirm.

### R10 — ADOPT NVFP4 5-candidate scale search (when implementing NVFP4)
**Priority:** Medium **Difficulty:** L **Dependencies:** R1
When implementing NVFP4 quantization, replicate the 5-candidate ue4m3
scale search (`test_offsets[5] = {0, -1, 1, -2, 2}`). This matches the
CPU quantizer's behaviour and is correctness-relevant.

---

## 17. Findings

### Finding ARTX14-F01

```
Finding ID:           ARTX14-F01
Category:             GPU_KERNEL
Engine:               CUDA
Component:            dequantize_block generic template
Source File:          ggml/src/ggml-cuda/convert.cu
Function:             dequantize_block
Lines:                8-41
Summary:              Generic 2-element-per-thread dequantize kernel
                      parameterised by (qk, qr, dequantize_kernel, dst_t);
                      256 threads per block, 4D strided grid.
Observation:          The template takes 4 compile-time parameters: qk
                      (block size), qr (element-per-int ratio),
                      dequantize_kernel (function pointer to the per-quant
                      device function), and dst_t (output dtype). Each
                      thread owns 2 elements (i00 = 2*(blockDim.x*blockIdx.x
                      + threadIdx.x)). The 4D grid (ne00, ne01, ne02, ne03)
                      is unflattened via fast_div_modulo. The kernel calls
                      dequantize_kernel(vx, ib, iqs, v) to get a float2,
                      then writes y[iy0 + {0, y_offset}] = ggml_cuda_cast<
                      dst_t>(v.{x,y}). Block size CUDA_DEQUANTIZE_BLOCK_SIZE
                      = 256; grid (ceil(ne00/512), min(ne01, 65535),
                      min(ne02*ne03, 65535)).
Evidence:             convert.cu:8-41 (template body); 246-255 (host
                      launcher dequantize_block_cuda); 257-260 (1D wrapper
                      dequantize_block_cont_cuda); convert.cuh:4
                      (CUDA_DEQUANTIZE_BLOCK_SIZE = 256).
Architectural Impact: This is the workhorse template for the simple
                      quants (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0) in both
                      contiguous and strided modes. The dst_t template
                      parameter is the key: same source produces F32, F16,
                      or BF16.
Correctness Impact:   None. The template is a pure wrapper around the per-
                      quant device function; correctness is determined by
                      the device function.
Optimization Type:    Tiling (256 threads / 512 elements per block) /
                      vectorization (2 elements per thread).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same template structure in glcuda, with the
                      dst_t parameter for direct F16/BF16 output.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX14-F02

```
Finding ID:           ARTX14-F02
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Six near-duplicate dispatch tables
Source File:          ggml/src/ggml-cuda/convert.cu
Function:             ggml_get_to_bf16_cuda, _fp16_cuda, _fp32_cuda, _fp16_nc_cuda, _bf16_nc_cuda, _fp32_nc_cuda
Lines:                458-693
Summary:              Six 22-arm switch tables map ggml_type → kernel
                      function pointer; ~250 lines of near-duplicate code.
Observation:          The six functions are structurally identical: each
                      is a switch over ggml_type returning a function
                      pointer to a per-quant launcher. The first three
                      (non-_nc) handle 1D contiguous inputs; the last
                      three (_nc) handle 4D strided inputs. The 22 cases
                      per switch are the same across all six tables; only
                      the function-pointer types differ (to_fp16_cuda_t vs
                      to_fp32_cuda_t vs to_bf16_cuda_t vs the _nc variants).
                      Adding a new quant format requires adding 6 cases.
Evidence:             convert.cu:458-511 (ggml_get_to_bf16_cuda); 513-569
                      (_fp16_cuda); 571-624 (_fp32_cuda); 626-647
                      (_fp16_nc_cuda); 649-670 (_bf16_nc_cuda); 672-693
                      (_fp32_nc_cuda).
Architectural Impact: Adding a quant = 6 switch cases. Forgetting one is
                      a silent bug (falls through to nullptr → runtime
                      crash). A single traits[GGML_TYPE_COUNT] table would
                      be one case per format.
Correctness Impact:   None for current code (all six tables are correct).
                      Risk of future divergence if one table is patched
                      and others aren't.
Optimization Type:    None (architectural concern).
GwenLand Target:      glcuda
Recommendation:       ADAPT. Replace with a single struct { to_fp16,
                      to_fp32, to_bf16, to_fp16_nc, to_bf16_nc,
                      to_fp32_nc; } traits[GGML_TYPE_COUNT] table.
Priority:             High
Difficulty:           M
Dependencies:         ARTX14-F01
Confidence:           High
```

### Finding ARTX14-F03

```
Finding ID:           ARTX14-F03
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Fused Q8_0→F16 kernel with shared-mem staging
Source File:          ggml/src/ggml-cuda/convert.cu
Function:             dequantize_block_q8_0_f16
Lines:                43-82
Summary:              Only fused quant→F16 kernel in the file; 2-stage
                      cooperative load (stage 1: load to shared mem, stage
                      2: dequantize + __hmul2 broadcast), template-
                      parameterised bounds check.
Observation:          Block size WARP_SIZE = 32 threads. Each block
                      processes CUDA_Q8_0_NE_ALIGN = 2048 elements (64 Q8_0
                      blocks of 32 elements). Shared memory: __shared__ int
                      vals[544] = 2176 bytes (2048/4 + 32 padding to avoid
                      bank conflicts). Stage 1: cooperative load, each
                      thread loads 17 ints. Stage 2: cooperative dequantize,
                      each thread reads the scale d from vals[], the char2
                      qs pair, and writes y2[…] = __hmul2(make_half2(qs.x,
                      qs.y), __half2half2(d)). The need_check template
                      parameter enables bounds checking for the last block
                      (when k % 2048 != 0). Gated by #if __CUDA_ARCH__ >=
                      GGML_CUDA_CC_PASCAL.
Evidence:             convert.cu:43-82 (kernel body); 262-271 (host
                      launcher dequantize_block_q8_0_f16_cuda); 513-528
                      (instantiation in ggml_get_to_fp16_cuda).
Architectural Impact: 2× bandwidth win vs the generic dequantize_block
                      template for Q8_0 → F16. The __hmul2 broadcast
                      pattern is the model for any future fused quant→F16
                      kernel.
Correctness Impact:   None. The arithmetic is mathematically equivalent
                      to the generic template; just faster.
Optimization Type:    Kernel fusion (dequantize + F16 conversion) /
                      vectorization (__hmul2).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same 2-stage pattern for Q8_0 → F16 in glcuda.
                      Consider extending to Q4_0 → F16, Q4_K → F16.
Priority:             High
Difficulty:           M
Dependencies:         ARTX14-F01
Confidence:           High
```

### Finding ARTX14-F04

```
Finding ID:           ARTX14-F04
Category:             QUANTIZATION
Engine:               CUDA
Component:            K-quant 6-bit scale unpacker (canonical version)
Source File:          ggml/src/ggml-cuda/dequantize.cuh
Function:             get_scale_min_k4
Lines:                157-164
Summary:              The canonical K-quant 6-bit scale unpacker, shared
                      by dequantize_q4_K and dequantize_q5_K; duplicated
                      in two other places (vecdotq.cuh, mmq-load-tiles.cuh).
Observation:          get_scale_min_k4(j, q, &d, &m) decodes the 6-bit
                      packed scale layout: for j < 4, d = q[j] & 63 and m =
                      q[j+4] & 63; for j >= 4, d = (q[j+4] & 0xF) | ((q[j-4]
                      >> 6) << 4) and m = (q[j+4] >> 4) | ((q[j-0] >> 6) <<
                      4). The 6-bit scale and 6-bit min are packed across
                      two bytes (low 6 bits in one byte, high 2 bits in
                      another). This function is the canonical version,
                      used by dequantize_q4_K (line 184) and dequantize_q5_K
                      (line 212). Two other implementations exist:
                      vecdotq.cuh:890-899 (inline in MMVQ Q4_K vecdot) and
                      mmq-load-tiles.cuh:612-620 (unpack_scales_q45_K in the
                      MMQ loader). See ARTX13-F06 for the cross-file
                      analysis.
Evidence:             dequantize.cuh:157-164 (definition); 184, 186, 212,
                      214 (call sites in dequantize_q4_K and dequantize_q5_K).
Architectural Impact: Triple-maintained bit manipulation. Bug fixes must
                      be applied in three places. glcuda should have ONE
                      canonical unpacker.
Correctness Impact:   None for current code (all three are correct). Risk
                      of future divergence.
Optimization Type:    None (architectural concern).
GwenLand Target:      glcuda
Recommendation:       ADAPT. Make this the single canonical unpacker; have
                      the other two call sites (vecdot MMVQ, MMQ loader)
                      call it. See ARTX13-F06 for the consolidation plan.
Priority:             High
Difficulty:           M
Dependencies:         ARTX13-F06
Confidence:           High
```

### Finding ARTX14-F05

```
Finding ID:           ARTX14-F05
Category:             GPU_KERNEL
Engine:               CUDA
Component:            dst_t template parameter on every dequantize function
Source File:          ggml/src/ggml-cuda/dequantize.cuh, ggml/src/ggml-cuda/convert.cuh
Function:             every dequantize_* template, ggml_cuda_cast
Lines:                dequantize.cuh:107-432 (per-format); convert.cuh:34-66 (cast helper)
Summary:              Every K-quant and IQ dequantize function is templated
                      on dst_t; every output write goes through
                      ggml_cuda_cast<dst_t>(float), enabling direct F16/BF16
                      output with no F32 intermediate.
Observation:          The dst_t template parameter appears in every K-quant
                      and IQ dequantize function: dequantize_q2_K<dst_t>,
                      dequantize_q4_K<dst_t>, dequantize_iq2_xxs<dst_t>,
                      dequantize_mxfp4<dst_t>, etc. Every output write is
                      y[j] = ggml_cuda_cast<dst_t>(d * grid[j] * …). The
                      ggml_cuda_cast<dst_t> helper (convert.cuh:34-66) is a
                      SFINAE-style if constexpr chain that handles every
                      (src_t, dst_t) pair: identity, float→half, float→bf16,
                      half→float, bf16→float, float2→half2, etc. The simple
                      quants (Q4_0, Q4_1, etc.) use float2 & v output instead
                      of dst_t * yy; the caller (dequantize_block template)
                      does the ggml_cuda_cast on v.x and v.y separately.
Evidence:             dequantize.cuh:107-432 (every template); convert.cuh:34-66
                      (ggml_cuda_cast definition); convert.cu:37-38 (caller
                      cast in dequantize_block).
Architectural Impact: Direct F16 output is 2× bandwidth win vs F32 write +
                      separate F32→F16 conversion. The same source produces
                      F32, F16, or BF16 with no code duplication.
Correctness Impact:   None. The cast is round-to-nearest-even for both F16
                      and BF16; deterministic.
Optimization Type:    Vectorization (direct F16 write) / kernel fusion
                      (dequantize + dtype convert in one call).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same dst_t template + SFINAE cast helper in
                      glcuda.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX14-F06

```
Finding ID:           ARTX14-F06
Category:             GPU_KERNEL
Engine:               CUDA
Component:            quantize_q8_1 (warp-cooperative scale + sum reduction)
Source File:          ggml/src/ggml-cuda/quantize.cu
Function:             quantize_q8_1
Lines:                53-101
Summary:              F32→Q8_1 activation quantizer: one thread per element,
                      warp_reduce_max/sum for per-block scale d=amax/127 and
                      sum s=sum(xi), single thread writes (d, s) half2.
Observation:          __launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1) — 1
                      block per SM. Each thread owns one element (i0 =
                      blockDim.x*blockIdx.x + threadIdx.x). The kernel
                      computes amax = warp_reduce_max<QK8_1>(fabsf(xi)) and
                      sum = warp_reduce_sum<QK8_1>(sum). The warp_reduce_*
                      templates (common.cuh:455-462) reduce across QK8_1
                      lanes — this assumes QK8_1 == WARP_SIZE (32 on NVIDIA).
                      The scale d = amax / 127.0f, int8 q = roundf(xi / d).
                      Each thread writes its q to y[ib].qs[iqs]; only thread
                      iqs == 0 writes y[ib].ds = make_half2(d, sum) (others
                      return early via if (iqs > 0) return;). PDL: calls
                      ggml_cuda_pdl_lc() at line 58 (early trigger) and
                      ggml_cuda_pdl_sync() at line 83 (mid-kernel).
Evidence:             quantize.cu:53-101 (kernel body); 558-573 (host
                      launcher quantize_row_q8_1_cuda).
Architectural Impact: This is the F32→Q8_1 path consumed by MMVQ (ARTX09)
                      for activation quantization. The warp-reduce pattern
                      is the standard CUDA-core reduction idiom; the
                      __launch_bounds__(…, 1) trades occupancy for register
                      headroom.
Correctness Impact:   None. The warp-reduce is butterfly-deterministic.
                      Rounding is roundf (round-to-nearest-even).
Optimization Type:    SIMD (warp_reduce_max/sum via __shfl_xor_sync) /
                      kernel fusion (scale + sum + int8 in one kernel) /
                      PDL (pdl_sync / pdl_lc).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same pattern in glcuda, but document the
                      QK8_1 == WARP_SIZE assumption explicitly. For AMD
                      (WARP_SIZE=64), rework to use 2 sub-warps of 32.
Priority:             High
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX14-F07

```
Finding ID:           ARTX14-F07
Category:             GPU_KERNEL
Engine:               CUDA
Component:            quantize_mmq_q8_1 (float4 + shfl_xor, three output layouts)
Source File:          ggml/src/ggml-cuda/quantize.cu
Function:             quantize_mmq_q8_1
Lines:                457-556
Summary:              F32→Q8_1_MMQ activation quantizer: float4 loads, shfl_xor
                      reduce for amax/sum, outputs in one of three layouts
                      (D4, DS4, D2S6) selected by weight format.
Observation:          Template parameters: ds_layout (enum), scatter (bool
                      for MoE dedup). Each thread loads float4 (4 elements),
                      computes amax via fmaxf across the 4 lanes + __shfl_xor_sync
                      reduce across vals_per_scale/8 threads (= 8 for D4/DS4,
                      16 for D2S6). Similarly sum via __shfl_xor_sync reduce
                      across vals_per_sum/8 threads. d_inv = 127.0f / amax,
                      q = roundf(xi * d_inv), d = 1.0f / d_inv. Writes 4 int8s
                      as a single char4 (32-bit write). Per-block scale+sum
                      written by one thread per vals_per_scale/vals_per_sum
                      group, in layout-specific field (y[ib].ds4[iqs/32] for
                      DS4, y[ib].d4[iqs/32] for D4, y[ib].d2s6[…] for D2S6).
                      The scatter=true path writes the same block to multiple
                      expert rows via ids[] (MoE dedup).
Evidence:             quantize.cu:457-556 (kernel body); 575-603 (host
                      launcher quantize_mmq_q8_1_cuda); 606-633 (MoE scatter
                      launcher).
Architectural Impact: This is the F32→Q8_1_MMQ path consumed by MMQ
                      (ARTX10/12). The float4 + shfl_xor pattern is 4× the
                      per-thread throughput of quantize_q8_1. The three
                      output layouts let the MMQ loader consume the
                      activation in the format most natural for the weight
                      (D4 for Q8_0 weights, DS4 for Q4_0/Q5_0, D2S6 for
                      Q4_K/Q5_K/Q2_K).
Correctness Impact:   None. The warp-reduce is butterfly-deterministic.
                      Rounding is roundf.
Optimization Type:    SIMD (float4 + shfl_xor reduce) / vectorization
                      (char4 packed write) / kernel fusion (MoE scatter).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same float4 + shfl_xor pattern, with the
                      three-layout output structure mirroring the weight
                      format.
Priority:             High
Difficulty:           M
Dependencies:         ARTX14-F06
Confidence:           High
```

### Finding ARTX14-F08

```
Finding ID:           ARTX14-F08
Category:             QUANTIZATION
Engine:               CUDA
Component:            NVFP4 5-candidate ue4m3 scale search
Source File:          ggml/src/ggml-cuda/quantize.cu
Function:             quantize_mmq_nvfp4
Lines:                128-332
Summary:              NVFP4 activation quantizer tries 5 candidate ue4m3
                      scales per sub-block ({0, -1, +1, -2, +2} offsets
                      from first guess) and picks the lowest-SSE scale.
Observation:          For each 16-element sub-block: compute amax_sub,
                      convert amax_sub / 6.0f to a first ue4m3 candidate
                      via ggml_cuda_fp32_to_ue4m3. Try 5 candidate codes
                      (test_offsets[5] = {0, -1, 1, -2, 2}, clamped to [0,
                      0x7e]). For each candidate, compute
                      nvfp4_native_scale_error (CUDA 12.8+ uses native
                      __nv_fp4x4_e2m1 intrinsics; otherwise falls back to
                      manual LUT-based error via kvalues_fp4). Pick the
                      candidate with the lowest sum-of-squared-errors.
                      Then quantize the 16 elements with the chosen scale.
                      The use_aligned_float8 template parameter selects
                      between float8 (256-bit) loads and scalar float loads.
                      Gated by #if defined(BLACKWELL_MMA_AVAILABLE).
Evidence:             quantize.cu:128-332 (kernel body); 233-279 (5-candidate
                      search loop); 665-697 (host launcher
                      quantize_mmq_fp4_cuda).
Architectural Impact: This is the only quantizer in the file that does a
                      per-sub-block mini-grid-search. The 5-candidate search
                      matches the CPU quantizer's behaviour
                      (ggml-quants.c) and is correctness-relevant — the
                      ue4m3 scale is the single biggest determinant of
                      NVFP4 accuracy.
Correctness Impact:   None. The search is deterministic (same inputs →
                      same candidate chosen). The chosen scale may differ
                      from the CPU quantizer's choice by ±1 ue4m3 code
                      due to FP rounding, but the SSE comparison is
                      deterministic.
Optimization Type:    SIMD (warp_reduce_max for amax) / kernel fusion
                      (scale search + FP4 quantize in one kernel) /
                      vectorization (float8 loads when aligned).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same 5-candidate search pattern when
                      implementing NVFP4 quantization. Use
                      __nv_fp4x4_e2m1 intrinsics on CUDA 12.8+; fall back
                      to LUT otherwise.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX14-F07
Confidence:           High
```

### Finding ARTX14-F09

```
Finding ID:           ARTX14-F09
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            No fused dequantize + RoPE kernel
Source File:          ggml/src/ggml-cuda/ (entire backend)
Function:             N/A
Lines:                N/A
Summary:              No dequantize_rope or rope_dequant kernel exists;
                      RoPE is always applied post-dequantization in a
                      separate kernel, materializing the full F32/F16
                      tensor in memory.
Observation:          Grep for dequantize[Rr]o[Pp][Ee] or rope.*dequant
                      in the CUDA backend returns no matches. The RoPE
                      kernel (rope.cu) reads F32/F16 input, applies the
                      rotation, writes F32/F16 output. For quantized KV
                      cache (Q4_K, Q8_0, IQ4_XS, etc.), the path is:
                      (1) dequantize Q4_K → F32 via dequantize_row_q4_K_cuda;
                      (2) apply RoPE to F32 via rope.cu; (3) write F32 back
                      to KV cache. Steps 1 and 3 each touch the full
                      tensor; step 2 touches it again. A fused
                      dequantize_rope_q4_K_cuda kernel would: (a) read Q4_K;
                      (b) dequantize to F32 in registers; (c) apply RoPE in
                      registers; (d) write F32/F16 to KV cache. One memory
                      pass instead of three.
Evidence:             No file matches (grep returns empty); rope.cu exists
                      as a separate kernel; dequantize_row_*_cuda in
                      convert.cu:273-375 are separate from rope.cu.
Architectural Impact: For long-context inference where the KV cache is
                      large (e.g., 32K context × 4096 dim × 32 layers ×
                      2 (K+V) = 16 GB F32), the dequantize + RoPE path
                      is 3× the memory traffic of a fused kernel. A
                      fused kernel would halve to two-thirds the memory
                      traffic.
Correctness Impact:   None. The separate-kernel path is correct; just
                      slow.
Optimization Type:    None (missing optimization).
GwenLand Target:      glcuda, GATE
Recommendation:       REJECT the absence; add a fused dequantize_rope
                      kernel for the quantized KV cache path. Priority
                      depends on whether glcuda targets long-context
                      inference.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX14-F01, GATE design (for fusion detection)
Confidence:           Medium
```

### Finding ARTX14-F10

```
Finding ID:           ARTX14-F10
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            Lookup tables not explicitly __constant__-qualified
Source File:          ggml/src/ggml-common.h
Function:             GGML_TABLE_BEGIN macro
Lines:                490-494
Summary:              All lookup tables (kvalues_iq4nl, kvalues_mxfp4,
                      iq2xxs_grid, etc.) are declared static const
                      __device__, not __constant__; placement is compiler-
                      dependent.
Observation:          The macro GGML_TABLE_BEGIN for CUDA/HIP/MUSA expands
                      to `static const __device__ type name[size] = {` (line
                      493). This places the table in device read-only data
                      memory, accessed via the L1 read-only cache (via
                      __ldg-style loads on NVIDIA). The tables are NOT
                      explicitly __constant__-qualified. The 64 KB constant
                      memory is reserved for kernel parameters and would
                      not fit the larger tables anyway (iq2s_grid = 8 KB,
                      iq3s_grid = 2 KB), but small tables (kvalues_iq4nl =
                      16 bytes, kvalues_mxfp4 = 16 bytes, kmask_iq2xs = 8
                      bytes, ksigns_iq2xs = 128 bytes) would fit and would
                      benefit from the broadcast-capable constant cache (1
                      cycle per warp when all lanes read the same address).
                      Currently, all tables use the L1 read-only cache
                      regardless of size.
Evidence:             ggml-common.h:490-494 (GGML_TABLE_BEGIN macro for
                      CUDA); 509-1650 (table definitions).
Architectural Impact: For small tables, __constant__ qualification would
                      be faster (constant cache vs L1 read-only). For
                      large tables, __device__ const is correct (constant
                      memory is too small). A per-table policy would be
                      optimal.
Correctness Impact:   None. The placement affects performance, not
                      correctness. The compiler is free to use __ldg for
                      __device__ const reads, which is correct.
Optimization Type:    None (suboptimal memory placement).
GwenLand Target:      glcuda
Recommendation:       MONITOR. Profile per-table; small tables (≤ 128
                      bytes) to __constant__, large tables to __device__
                      const with explicit __ldg.
Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether the `dequantize_block_q8_0_f16` kernel's 2-stage
  shared-memory staging is actually faster than a direct read +
  `__hmul2` per-thread approach. The shared-memory staging adds a
  `__syncthreads()` and a 2176-byte shared-memory allocation; the
  direct approach would issue 2 global loads per thread (one for the
  scale, one for the qs pair) but avoid the shared-memory round-trip.
  Requires profiling.
* **U2**. Whether the six dispatch tables (`ggml_get_to_*_cuda`) are
  actually called per-op or cached by the caller. If called per-op,
  the 22-arm switch is ~5 instructions per call (negligible). If
  cached, the switch is irrelevant. Requires call-site inspection of
  `ggml-cuda.cu`.
* **U3**. Whether the `quantize_q8_1` `QK8_1 == WARP_SIZE` assumption
  is enforced by the build system on AMD, or whether the kernel is
  silently broken on RDNA. The warp_reduce_max<QK8_1> template would
  reduce across only 32 of 64 lanes on AMD, producing wrong amax. The
  build system gating is not in scope of this audit.
* **U4**. Whether the NVFP4 5-candidate scale search produces bit-
  identical results to the CPU quantizer (`ggml-quants.c`). The CPU
  also does a 5-candidate search but with different FP rounding
  (possibly `fesetround(FE_TONEAREST)` vs CUDA's default round-to-
  nearest-even). Requires differential testing.
* **U5**. Whether the K-quant `dequantize_row_*_cuda` per-row launch
  pattern is a measurable bottleneck for typical llama.cpp workloads.
  For a 32-layer model with 7 matmuls per layer and 4096 rows each,
  that's 32 × 7 × 4096 = 917K kernel launches per inference step.
  At ~5µs per launch, that's ~4.6 s of pure launch overhead. Requires
  profiling.
* **U6**. Whether `__constant__` qualification on the small lookup
  tables (kvalues_iq4nl, kvalues_mxfp4, kmask_iq2xs, ksigns_iq2xs)
  would actually be faster on Ampere/Hopper. The constant cache is
  broadcast-capable (1 cycle when all lanes read the same address),
  but the L1 read-only cache may already broadcast for `__ldg` reads
  of `__device__ const` data. Requires PTX/SASS inspection.
* **U7**. Whether `convert_unary` (1 element per thread) is measurably
  slower than a `float4`-vectorised version for F32→F16 conversion.
  The bandwidth difference is 4× (4-byte read + 2-byte write per
  thread vs 16-byte read + 8-byte write per thread). Requires
  profiling.
* **U8**. Whether the IQ2/IQ3 grid-lookup patterns in `dequantize.cuh`
  (lines 263, 281, 297, 315, 316, 335, 336, 356, 379) generate
  efficient PTX. The `iq2xxs_grid + aux8[il]` pattern is a pointer
  offset, not an array index; the compiler may or may not emit `__ldg`.
  Requires PTX inspection.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines                |
| --------- | --------------------------------------------------- | ---------------------------------------------- | -------------------- |
| R01       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `dequantize_q1_0` … `dequantize_q8_0`          | 4-100                |
| R02       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `dequantize_q2_K` … `dequantize_q6_K`          | 107-246              |
| R03       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `get_scale_min_k4` (canonical K-quant unpacker) | 157-164              |
| R04       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `dequantize_iq2_xxs` … `dequantize_iq4_xs`     | 253-416              |
| R05       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `dequantize_mxfp4`                             | 418-432              |
| R06       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block` (generic template)          | 8-41                 |
| R07       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_q8_0_f16` (fused fast path)  | 43-82                |
| R08       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_q4_0`, `_q4_1`               | 85-137               |
| R09       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_q2_K` … `dequantize_block_mxfp4` | 142-244          |
| R10       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_cuda`, `_cont_cuda`          | 246-260              |
| R11       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_q8_0_f16_cuda`               | 262-271              |
| R12       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_row_*_cuda` (22 launchers)         | 273-375              |
| R13       | `ggml/src/ggml-cuda/convert.cu`                     | `dequantize_block_nvfp4`                       | 377-404              |
| R14       | `ggml/src/ggml-cuda/convert.cu`                     | `convert_unary`, `_cuda`, `_cont_cuda`         | 416-456              |
| R15       | `ggml/src/ggml-cuda/convert.cu`                     | `ggml_get_to_bf16_cuda`, `_fp16_cuda`, `_fp32_cuda` | 458-624         |
| R16       | `ggml/src/ggml-cuda/convert.cu`                     | `ggml_get_to_*_nc_cuda` (3 strided dispatchers) | 626-693             |
| R17       | `ggml/src/ggml-cuda/convert.cuh`                    | `CUDA_DEQUANTIZE_BLOCK_SIZE`, `to_*_cuda_t` typedefs | 1-32            |
| R18       | `ggml/src/ggml-cuda/convert.cuh`                    | `ggml_cuda_cast<dst_t>` SFINAE helper          | 34-66                |
| R19       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_q8_1`                                | 53-101               |
| R20       | `ggml/src/ggml-cuda/quantize.cu`                    | `compute_e8m0_scale`                           | 103-124              |
| R21       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_mmq_nvfp4` (5-candidate scale search) | 128-332             |
| R22       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_mmq_mxfp4`                           | 338-454              |
| R23       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_mmq_q8_1` (3 layouts)                | 457-556              |
| R24       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_row_q8_1_cuda`, `_mmq_q8_1_cuda`, `_mmq_fp4_cuda`, `_scatter_*_cuda` | 558-697 |
| R25       | `ggml/src/ggml-cuda/common.cuh`                     | `dequantize_kernel_t`, `dequantize_kq_t<dst_t>` typedefs | 947-950       |
| R26       | `ggml/src/ggml-cuda/common.cuh`                     | `fast_div_modulo`, `fastdiv`, `init_fastdiv_values` | 940-945        |
| R27       | `ggml/src/ggml-cuda/common.cuh`                     | `warp_reduce_max`, `warp_reduce_sum`           | 455-462              |
| R28       | `ggml/src/ggml-cuda/getrows.cu`                     | `k_get_rows` (consumer of dequantize_kernel_t) | 5-41                 |
| R29       | `ggml/src/ggml-cuda/getrows.cu`                     | `k_get_rows_kq` (consumer of dequantize_kq_t)  | 43-70                |
| R30       | `ggml/src/ggml-cuda/cpy.cu`                         | `cpy` kernel (consumer of dequantize_kernel_t for fused copy) | 112 |
| R31       | `ggml/src/ggml-common.h`                            | `GGML_TABLE_BEGIN` macro (CUDA/HIP/MUSA branch) | 490-494              |
| R32       | `ggml/src/ggml-common.h`                            | `kvalues_iq4nl`, `kvalues_mxfp4`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`, `ksigns_iq2xs`, `kmask_iq2xs` | 509-1650 |
| R33       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | (cross-reference ARTX13-F06: sister K-quant scale unpacker) | 890-899, 935-944 |
| R34       | `ggml/src/ggml-cuda/mmq-load-tiles.cuh`             | `unpack_scales_q45_K` (third K-quant scale unpacker) | 612-620         |
