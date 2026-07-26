# ARTX09 — CUDA GEMV Kernels (MMVQ, MMVF, small-batch MMF)

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (graph execution, fusion)

---

## 1. Executive Summary

The CUDA GEMV path in llama.cpp is the **autoregressive-decode hot path**.
For every generated token, each linear layer issues an `M×K × K×1 = M×1`
matrix–vector product; for MoE models, the same op is batched over the
expert axis as `M×K × K×B` with `B ≤ 8`. Three kernel families cover the
GEMV regime:

1. **MMVQ** (`mmvq.cu`, 1289 lines) — quantized weight GEMV. Single template
   `mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>` covers every
   quant format (Q4_0 … IQ4_XS, MXFP4, NVFP4). Each warp owns one or two
   weight rows; the K dimension is reduced with `vec_dot_q_cuda` calls and
   a warp-shuffle + shared-memory reduction.
2. **MMVF** (`mmvf.cu`, 870 lines) — F32/F16/BF16 weight GEMV. Single
   template `mul_mat_vec_f<T, type_acc, ncols_dst, block_size, …>`
   computes one output row per block, with auto-tuned block size
   (32–256 threads) and a choice of F32- or F16-accumulation. **No Tensor
   Cores** are used in MMVF — pure CUDA-core FMA / `half2` arithmetic.
3. **MMF** (`mmf.cu` + `mmf.cuh`, ~1100 lines) — small-batch F32/F16/BF16
   GEMM using `mma` / `ldmatrix` Tensor-Core instructions. Covers the
   boundary between GEMV and true GEMM (`ne11 ≤ 16`).

Routing between the three is performed inside `ggml_cuda_mul_mat`
(`ggml-cuda.cu:1812`) using `ggml_cuda_should_use_mmvf` / `_mmf` / `_mmvq`
/ `_mmq` predicates. The GEMV/GEMM threshold is `MMVQ_MAX_BATCH_SIZE =
MMVF_MAX_BATCH_SIZE = 8` (`mmvq.cuh:3`, `mmvf.cuh:3`). Above 8 the path
falls through to MMQ (quantized GEMM) or cuBLAS.

For GwenLand, the decisions worth **ADOPT**ing are: (a) the
template-on-`ncols_dst` enumeration that lets the compiler unroll the
output-column loop; (b) the per-arch / per-type tuning tables for
`nwarps`, `rows_per_block`, and `mmid_max_batch`; (c) the inline
`vec_dot_q_cuda` device-function-pointer pattern that fuses dequantization
into the dot product without staging to shared memory; (d) the
auto-tuned block-size search in `launch_mul_mat_vec_f_cuda`. The decisions
worth **REJECT**ing are: (a) the `ggml_cuda_should_use_mmvf` policy that
defers almost entirely to MMF on Ampere+ (the MMVF kernel is left
under-utilised); (b) the mandatory Q8_1 activation re-quantization on
every MMVQ call (no caching across decode steps); (c) the hand-maintained
per-arch batch-size tables that grow with every new GPU generation.

---

## 2. Purpose

Provide the CUDA kernels that service `MUL_MAT` and `MUL_MAT_ID` when the
activation matrix has a small column count (`ne11 ≤ 8`) — i.e., the
single-token or few-token autoregressive-decode case.

Specifically:

* Implement `M×K × K×1` (true GEMV) for every supported quant format and
  for F32/F16/BF16 weights, with shape-specialised template instances for
  `ne11 ∈ {1..8}`.
* Implement `M×K × K×B` (batched GEMV) up to `B = 8`, including the
  expert-routed `MUL_MAT_ID` path used by MoE FFN layers.
* Fuse adjacent FFNGLU-style operations (bias + SiLU/GELU + gate
  matmul) into the GEMV kernel prologue/epilogue when `ncols_dst == 1`.
* Auto-select the best kernel based on shape, dtype, compute capability,
  and (for `MUL_MAT_ID`) expert count.
* Expose standard entry points (`ggml_cuda_mul_mat_vec_q`,
  `ggml_cuda_mul_mat_vec_f`) to the CUDA op dispatch
  (`ggml-cuda.cu:ggml_cuda_mul_mat`).

It is **not** responsible for: graph-level fusion decisions (handled by
`ggml_cuda_try_fuse` in `ggml-cuda.cu`, audited in ARTX08), activation
quantization policy (delegated to `quantize.cu:quantize_row_q8_1_cuda`),
or cuBLAS fallback (handled in `ggml-cuda.cu:ggml_cuda_mul_mat_cublas`).

---

## 3. Source Files

| File                                          | Lines  | Role                                                                              |
| --------------------------------------------- | ------ | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/mmvq.cu`                  | 1289   | MMVQ template kernel `mul_mat_vec_q`, MoE-dedicated `mul_mat_vec_q_moe`, per-arch tuning tables, type/ncols/fusion dispatch chain, `ggml_cuda_mul_mat_vec_q` entry, `ggml_cuda_op_mul_mat_vec_q` legacy op wrapper. |
| `ggml/src/ggml-cuda/mmvf.cu`                  | 870    | MMVF template kernel `mul_mat_vec_f`, auto-tuned `launch_mul_mat_vec_f_cuda`, type/ncols/fusion dispatch chain, `ggml_cuda_mul_mat_vec_f` entry, `ggml_cuda_should_use_mmvf` predicate. |
| `ggml/src/ggml-cuda/mmf.cu`                   | 192    | MMF dispatch shell `ggml_cuda_mul_mat_f`, `ggml_cuda_should_use_mmf` predicate. Kernel template lives in `mmf.cuh`. |
| `ggml/src/ggml-cuda/mmf.cuh`                  | 909    | MMF Tensor-Core kernel `mul_mat_f`, `mul_mat_f_ids`, `mmf_get_*` helpers, template-instantiation macros. |
| `ggml/src/ggml-cuda/mmid.cu`                  | 170    | `mm_ids_helper` kernel: on-device sort/compaction of `MUL_MAT_ID` expert routing, used by MMF when `ncols_dst > 16`. |
| `ggml/src/ggml-cuda/mmvq.cuh`                 | 19     | `MMVQ_MAX_BATCH_SIZE = 8`, public prototypes.                                    |
| `ggml/src/ggml-cuda/mmvf.cuh`                 | ~20    | `MMVF_MAX_BATCH_SIZE = 8`, public prototypes.                                    |
| `ggml/src/ggml-cuda/dequantize.cuh`           | 433    | Per-format `dequantize_*` device functions. Used by MMQ and convert kernels, **not** by MMVQ (which uses `vec_dot_q_cuda` instead). |
| `ggml/src/ggml-cuda/vecdotq.cuh`              | 1323   | `vec_dot_q*_*_q8_1` device functions, `VDR_*_MMVQ` constants, `dp4a`-based quant dot product. Called from inside MMVQ. |
| `ggml/src/ggml-cuda/common.cuh`               | 1661   | `warp_reduce_sum`, `ggml_cuda_dp4a`, `ggml_cuda_mad`, `ggml_cuda_kernel_launch` (PDL-aware), `ggml_cuda_type_traits<ggml_type>`. (Audited in ARTX08; only GEMV-relevant helpers summarised here.) |

> Note: the audit prompt's reference to a separate "MMVQ vs `mul_mat_vec_q`
> template" is accurate — there is exactly one template `mul_mat_vec_q`
> in `mmvq.cu:480`, and the file is internally referred to as "MMVQ"
> (`#define MMVQ_MAX_BATCH_SIZE`). The dispatch chain
> `mul_mat_vec_q_switch_type → _switch_ncols_dst → _switch_fusion →
> mul_mat_vec_q` enumerates both the dtype and the `ncols_dst` axis.

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
   ggml_cuda_should_    ggml_cuda_should_         ggml_cuda_should_
   use_mmvf(...)        use_mmf(...)              use_mmvq(...)
   (F16/BF16/F32)       (F16/BF16/F32)            (all quants)
        │                     │                          │
        ▼                     ▼                          ▼
   mmvf.cu              mmf.cu + mmf.cuh           mmvq.cu
   ggml_cuda_mul_       ggml_cuda_mul_mat_f        ggml_cuda_mul_mat_vec_q
   mat_vec_f            (Tensor Cores via mma)     │
   (CUDA cores only)                               ├─ quantize_row_q8_1_cuda
                                                  │   (F32 src1 → Q8_1)
                                                  ▼
                                              mul_mat_vec_q_switch_type
                                                  │
                                                  ▼
                                              mul_mat_vec_q_switch_ncols_dst
                                                  │
                                                  ▼
                                              mul_mat_vec_q_switch_fusion
                                                  │
                                                  ▼
                                              mul_mat_vec_q<type, N, has_fusion>
                                                  │
                                                  ▼
                                              vec_dot_q_cuda (per-block, per-thread)
                                                  │
                                                  ▼
                                              warp_reduce_sum + shared-mem cross-warp
```

Key design points:

* **Three-kernel split.** MMVF (CUDA cores) and MMF (Tensor Cores) overlap
  in dtype coverage (F32/F16/BF16) but differ in whether they use `mma`.
  MMVQ is the only path for quantized weights in the GEMV regime; MMQ
  (audited in ARTX10) takes over when `ne11 > 8`.
* **Two template dimensions.** All three kernels are templated on `T`
  (dtype) **and** on `ncols_dst` (compile-time-constant output column
  count, 1..8 for MMVQ/MMVF, 1..16 for MMF). The `ncols_dst` enumeration
  lets the compiler unroll the inner output-column loop and stage
  per-column accumulators in registers.
* **Inline dequantize, no staging.** MMVQ never writes dequantized
  weights to shared memory. Each `vec_dot_q_cuda(vx, &y[...], kbx, kqs)`
  call dequantizes one quant block on the fly and accumulates into a
  per-thread F32 register (`tmp[ncols_dst][rows_per_cuda_block]`).
  Shared memory is used only for the cross-warp reduction of partial sums.
* **Activation re-quantization is mandatory in MMVQ.** The F32 `src1` is
  converted to Q8_1 (one scale + one sum per 32-element block) via
  `quantize_row_q8_1_cuda` before the kernel launch
  (`mmvq.cu:1223-1229`). The quantized activation lives in a per-op pool
  allocation; there is no caching across decode steps.
* **Fusion is `ncols_dst == 1`-only.** Both MMVQ and MMVF accept an
  optional `ggml_cuda_mm_fusion_args_device` struct carrying gate / bias
  / scale pointers. When `has_fusion` is true and `ncols_dst == 1`, the
  kernel computes the gate matmul in parallel with the main matmul, then
  applies SiLU / GELU / SwiGLU-OAI in the epilogue. Multi-column fusion
  is asserted out (`mmvq.cu:806`, `mmvf.cu:403`).

---

## 5. Execution Flow

### 5.1 MMVQ entry and dispatch

`ggml_cuda_mul_mat_vec_q` (`mmvq.cu:1145`) asserts `src1->type == F32`,
`dst->type == F32`, and innermost-dim contiguity for all three tensors.
If `src0` is a temporary compute buffer with padding, it issues
`cudaMemsetAsync` to zero the tail (`:1212-1220`) so that
`vec_dot_q_cuda` reads past the data end do not pick up NaN scales. It
then computes `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)` (512),
allocates a `block_q8_1` scratch from `ctx.pool()`, and launches
`quantize_row_q8_1_cuda` (`:1228`) to convert F32 src1 → Q8_1 (writing
both per-block scale `d` and per-block abs-sum `s`). For `MUL_MAT_ID`,
the `ne2` axis becomes `ncols_dst` (tokens) and `ne1` becomes
`nchannels_dst` (experts). Finally it calls
`mul_mat_vec_q_switch_type(...)` (`:1253`).

The dispatch chain is three layers: `_switch_type` (`:998`, 23-arm
`switch` over quant type) → `_switch_ncols_dst` (`:837`, 8-arm `switch`
over output column count, plus a `has_ids && ncols_dst > 1` branch that
routes to the MoE-dedicated kernel — §5.4) → `_switch_fusion` (`:783`,
2-way on `has_fusion && ncols_dst == 1`). The `case 1` arm of
`_switch_ncols_dst` also evaluates the `should_use_small_k` lambda
(`:861-901`) which bumps `rows_per_cuda_block` from 1 to `nwarps` when K
is short. Each leaf calls `ggml_cuda_kernel_launch` (the PDL-aware
launcher from `common.cuh:1641`).

### 5.2 Inside `mul_mat_vec_q` (the GEMV kernel)

`mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>` (`:478-699`)

Block layout: `dim3 block_dims(warp_size, nwarps, 1)`. For NVIDIA
Generic, `nwarps ∈ {1, 2, 4}` (1-4 → 4 warps, 5-8 → 2 warps, >8 → 1
warp); RDNA4 widens to 8 warps for simple-vec_dot types at `ncols_dst ==
1`. Grid: `(ceil(nrows_x / rows_per_cuda_block), nchannels_dst,
nsamples_or_ntokens)`. Per-thread state is `float
 tmp[ncols_dst][rows_per_cuda_block]` plus a matching `tmp_gate` when
`has_fusion`.

The K-dim loop (`:592-612`) strides by `blocks_per_iter = vdr * nwarps
* warp_size / qi` (`:505`). Each iteration, every thread issues one
`vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x
+ kbx, kqs)` call per (j, i) pair; this call internally performs `vdr`
`dp4a` reductions plus an F32 scale multiply (`vecdotq.cuh:115-134`).
For Q4_0 with `vdr=2, nwarps=4, warp_size=32, qi=2`, the warp group
 covers 128 quant blocks (= 4096 elements) per iteration.

After the K loop, warps > 0 dump their partial sums to
`tmp_shared[nwarps-1][ncols_dst][rows_per_cuda_block][warp_size]`
(`:614`), `__syncthreads`, then exit. Warp 0 reads back and reduces,
then calls `warp_reduce_sum<warp_size>(tmp[j][i])` (`:652`) which is a
butterfly `__shfl_xor_sync` reduction (`common.cuh:455-462`). One
designated thread per output lane writes `dst[j*stride_col_dst + i]`.

### 5.3 MoE-dedicated MMVQ kernel

`mul_mat_vec_q_moe<type, c_rows_per_block>` (`:705-769`). Grid:
`(ceil(nrows_x / c_rows_per_block), nchannels_dst)`, block
`(warp_size, ncols_dst)`. Each warp (`threadIdx.y`) processes one token
independently — no shared-memory cross-warp reduction, just a single
`warp_reduce_sum`. Launch is hard-coded to `rows_per_block = 2`
(`:824`). Selected when `has_ids && ncols_dst > 1` (multi-token
`MUL_MAT_ID`).

### 5.4 MMVF entry, dispatch, and kernel

`ggml_cuda_mul_mat_vec_f` (`mmvf.cu:629`) asserts shape/stride
contiguity, reads `cc` once, and selects `prec = fast_fp16_available(cc)
? ggml_prec(dst->op_params[0]) : GGML_PREC_F32` (`:650`) — the user can
force F32 accumulation by setting `op_params[0] = GGML_PREC_F32`. It
then calls `mul_mat_vec_f_cuda<T>(...)` (`:604`), which dispatches on
`prec`: if `T == half && prec == DEFAULT`, instantiate with `type_acc =
half` (F16-acc); otherwise `type_acc = float` (F32-acc).

The dispatch chain is two layers: `_switch_ncols_dst` (`:507`, 8-arm
`switch`, plus a `has_ids && ncols_dst > 1` branch to
`launch_mul_mat_vec_f_cuda<T, type_acc, 1, /*is_multi_token_id=*/true>`)
→ `launch_mul_mat_vec_f_cuda` (`:412-505`) which auto-tunes block size
and enters a 9-arm `switch (block_size_best)` over `32, 64, 96, 128,
160, 192, 224, 256`. The auto-tune picks the smallest block size that
minimises `niter = ceil(ncols / (2*block_size))`, capped at 256
(128 on pre-RDNA1 AMD).

The kernel `mul_mat_vec_f<T, type_acc, ncols_dst, block_size,
has_fusion, is_multi_token_id>` (`:7-379`) uses 1D `block_size` threads,
grid `(nrows, nchannels_dst, nsamples_or_ntokens)`. Each block computes
one output row × `ncols_dst` output columns. The K-dim loop tiles across
`col2 ∈ [0, ncols/2)` in strides of `block_size`; each thread loads
`float2` / `half2` / `nv_bfloat162` from `x` and `y` and FMAs into
`float sumf[ncols_dst]` (or `half2 sumh2[ncols_dst]` for F16-acc). The
`ggml_cuda_mad` helper (`common.cuh:743-770`) maps to `v_dot2_f32_f16`
on AMD RDNA2+/CDNA. Reduction: `warp_reduce_sum` per warp, then a
shared-memory cross-warp stage (size `warp_size*sizeof(float)`) only
when `block_size > warp_size`. Output: thread `tid < ncols_dst` writes
`dst[tid*stride_col_dst + row]`, with the fused GLU epilogue applied
when `has_fusion`.

### 5.5 MMF (Tensor-Core GEMM) — boundary case

When `ne11` is in the 1..16 range and the weights are F32/F16/BF16 with
MMA hardware available, `ggml_cuda_mul_mat` selects MMF instead of MMVF.
The MMF kernel `mul_mat_f<T, rows_per_block, cols_per_block, nwarps,
has_ids>` (`mmf.cuh:48-294`) uses `tile<16, 8, T>` / `tile<16, 8,
float>` types and `mma` / `load_ldmatrix` PTX intrinsics to compute
`rows_per_block × cols_per_block` outputs per block via Tensor Cores.
This is audited at skim level here; the Tensor-Core MMA design is shared
with the MMQ path and is covered in detail in ARTX10.

### 5.6 Dispatch in `ggml_cuda_mul_mat`

`ggml-cuda.cu:1812-1852`. The order is **strictly**: (1)
`should_use_mmvf` → `mul_mat_vec_f`; (2) `should_use_mmf` → `mul_mat_f`;
(3) `should_use_mmvq` → `mul_mat_vec_q`; (4) `should_use_mmq` →
`mul_mat_q` (ARTX10); (5) else → `mul_mat_cublas`. Because MMVF is
tried first, on Ampere+ NVIDIA with F16/BF16 weights the `should_use_mmvf`
threshold is `src0_small && ne11 == 1` (`mmvf.cu:823-824`) — i.e., MMVF
only wins for true GEMV on Ampere+. For `ne11 ∈ {2..16}` on Ampere+,
MMF wins. On Pascal/Turing without MMA, MMVF takes the wider `ne11 <=
8` slice. For quantized weights, MMVQ is the only GEMV option.

### 5.7 `MUL_MAT_ID` GEMV path

`ggml_cuda_mul_mat_id` (`ggml-cuda.cu:1854-1893`). For `ne2 <= 8` and
quantized weights, the per-arch `get_mmvq_mmid_max_batch(src0->type, cc)`
threshold gates MMVQ (`:1871-1875`). For AMD non-quantized, MMVF is used
directly (`:1877-1880`). Above the per-arch threshold, MMQ or MMF takes
over; if neither qualifies, the host-sorting fallback path runs
(synchronises the stream — incompatible with CUDA graph capture, see
ARTX08 §5.5).

---

## 6. Data Layout

### 6.1 Weight tensor (src0)

Required: `nb00 == ggml_type_size(src0->type)` (contiguous in
innermost dim). MMVF additionally requires `nb[i] % (2*type_size) == 0`
for `i ≥ 1` (`mmvf.cu:797-801`) because the kernel casts `x` to
`half2*` / `float2*`. MMVQ has no such requirement on `nb[i]` — it
indexes per-block via `stride_row_x = ne00 / ggml_blck_size(type)`.

### 6.2 Activation tensor (src1)

`nb10 == ggml_type_size(F32) == 4`. The MMVQ entry pre-quantizes src1 to
Q8_1 with `ne10_padded = GGML_PAD(ne10, MATRIX_ROW_PADDING)` (512).
Padding is zero-filled by the `quantize_q8_1` kernel
(`quantize.cu:558-573`), which writes both `d` (per-32-element scale)
and `s` (per-32-element abs-sum) into each `block_q8_1`. The padded
length must be a multiple of `QK8_1 = 32`.

### 6.3 Output tensor (dst)

`nb0 == 4` (F32 contiguous in innermost). `stride_col_dst` is either
`dst->nb[1] / 4` (for `MUL_MAT`) or `dst->nb[2] / 4` (for `MUL_MAT_ID`,
where `ne1` is the expert axis and `ne2` is the token axis). The
`MUL_MAT_ID` layout swap is done in `ggml_cuda_mul_mat_vec_q`
(`:1242-1250`).

### 6.4 Per-quant block layout

Block sizes and `qk`/`qr`/`qi` constants are in `ggml-common.h` (not in
scope). MMVQ uses `ggml_cuda_type_traits<type>::qk` (elements per block)
and `::qi` (32-bit ints per block) read at compile time from
`common.cuh:964-1125`. The `vdr` (vec-dot ratio) is from
`vecdotq.cuh:VDR_*_MMVQ` and equals the number of `qi`-chunks one
thread processes per `vec_dot_q_cuda` call.

---

## 7. Memory Layout

### 7.1 Per-op scratch (`ctx.pool()`)

MMVQ allocates `ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1`
bytes per call (`mmvq.cu:1223`). For a 4096-dim FFN with batch 1, this
is `1 * 1 * 1 * 4096 * 34 / 32 = 4352` bytes per matmul — small but
allocated and freed on every call. The `ggml_cuda_pool_alloc<char>`
RAII wrapper returns the buffer to the pool at kernel-launch completion.

MMVF does **not** allocate any scratch — it reads `src1` directly as
`float*`.

### 7.2 MMVQ shared memory

`mul_mat_vec_q` declares `__shared__ float tmp_shared[nwarps-1 > 0 ?
nwarps-1 : 1][ncols_dst][rows_per_cuda_block][warp_size]` (`mmvq.cu:614`).
For the typical `nwarps=4, ncols_dst=1, rows_per_cuda_block=1,
warp_size=32` case: `3 * 1 * 1 * 32 * 4 = 384` bytes. When `has_fusion`
is true, an equal-sized `tmp_shared_gate` is added. The kernel never
uses `extern __shared__` — the size is compile-time constant per
template instance.

### 7.3 MMVF shared memory

`extern __shared__ char data_mmv[]` of size
`warp_size * sizeof(float) + (has_fusion ? warp_size * sizeof(float) : 0)`
(`mmvf.cu:449`). This is **only used when `block_size > warp_size`** —
for the common `block_size == warp_size` (32-thread) case, no shared
memory is touched. The `nbytes_shared` is passed to
`ggml_cuda_kernel_launch_params` (`:797, 808`).

### 7.4 MMF shared memory

MMF uses `extern __shared__ char data_mmv[]` for two purposes:
(a) `slot_map[cols_per_block]` for the `has_ids` gather, and (b)
`tile_xy` staging for `load_ldmatrix`. Size is computed per template
instance. CDNA gets a larger max (`mmf_get_max_block_size` returns 512
vs 256, `mmf.cuh:12-18`).

### 7.5 Pinned host memory

Not used directly by the GEMV kernels. The `MUL_MAT_ID` host-sorting
fallback (`ggml-cuda.cu:1911-1945`) uses `cudaMemcpyAsync` D2H + sync,
but that path is outside the GEMV kernels themselves.

---

## 8. Parallelism Strategy

### 8.1 MMVQ: warp-per-(row, column) tiling

Each thread block owns `rows_per_cuda_block` (= 1 or 2) weight rows ×
`ncols_dst` output columns. Within a block, `nwarps` warps cooperatively
reduce the K dimension. Each warp owns `blocks_per_iter = vdr * nwarps *
warp_size / qi` quant blocks per iteration; the warp's 32 threads each
process `vdr` of the `qi`-chunk. After the K loop, warps reduce via
shared memory + warp shuffle. The `small_k` variant bumps
`rows_per_cuda_block` to `nwarps` when K is short (`mmvq.cu:861-901`),
disabled for IQ2/IQ3/Q2_K/Q3_K on archs where vec_dot cost dominates.

### 8.2 MMVF: thread-block-per-row with auto-tuned width

Each block owns one output row × `ncols_dst` output columns. K is
parallelised across all `block_size` threads. Auto-tuning picks
`block_size ∈ {32, 64, 96, 128, 160, 192, 224, 256}` to minimise
`niter = (ncols + 2*bs - 1) / (2*bs)`. The heuristic picks the smallest
size that achieves the minimum `niter` (prefers 32 over 64 when both
give `niter = 1`).

### 8.3 MMVQ MoE kernel: warp-per-token

`mul_mat_vec_q_moe` assigns one warp per (token, expert) pair via
`threadIdx.y` (`mmvq.cu:726`). Each warp independently reduces its K
dimension with no cross-warp communication. Block Y is `ncols_dst`
(tokens); with `ncols_dst ≤ 8`, at most 8 warps per block.

### 8.4 Grid / block sizing helpers

`calc_nwarps(type, ncols_dst, table_id)` (`mmvq.cu:352-456`):
per-arch/per-type/per-`ncols_dst` heuristic over six tables.
`calc_rows_per_block(ncols_dst, table_id, small_k, nwarps)` (`:458-476`):
1 or 2 for the main path; `nwarps` when `small_k`.
`calc_launch_params<type>(...)` (`:771-781`): returns
`(block_nums, block_dims)`.

### 8.5 Multi-GPU

The legacy `ggml_cuda_op_mul_mat_vec_q` (`:1260-1289`) exists for the
removed multi-GPU split-buffer mechanism (ARTX08-F04). No longer called
from `ggml_cuda_mul_mat`; retained for compatibility with the
`ggml_cuda_op_mul_mat` template in `ggml-cuda.cu:1329`.

---

## 9. GPU Strategy

### 9.1 Per-arch tuning tables

MMVQ has the most extensive per-arch tuning in the CUDA backend. Six
`mmvq_parameter_table_id` values (`mmvq.cu:64-71`) select different
`(nwarps, rows_per_block, mmid_max_batch)` policies. The host-side
`get_device_table_id(int cc)` (`:89-106`) reads `cc`; the device-side
`get_device_table_id()` (`:73-87`) uses `#if defined(RDNA4)` /
`__CUDA_ARCH__` macros so the table is a compile-time constant inside
the kernel. Both must agree; this is enforced by the host launching the
template instance compiled for the matching arch.

### 9.2 VDR (vec-dot ratio) and `dp4a`

`vecdotq.cuh` defines `VDR_<TYPE>_Q8_1_MMVQ` constants (e.g.,
`VDR_Q4_0_Q8_1_MMVQ = 2`, `VDR_NVFP4_Q8_1_MMVQ = 4`): the number of
`qi`-chunks (32-bit ints) one thread processes per `vec_dot_q_cuda`
call. The MMQ constants (`VDR_*_MMQ`) are typically 2× larger. All
integer quants reduce via `ggml_cuda_dp4a(a, b, c)` (`common.cuh:703`),
which maps to the `dp4a` PTX instruction: 4-way 8-bit dot product with
32-bit accumulate. 4× throughput vs scalar 8-bit MADD on supporting
hardware (all NVIDIA since Kepler, all AMD GCN+).

### 9.3 F16/BF16 GEMV uses `half2` arithmetic, not `mma`

MMVF's `mul_mat_vec_f` kernel (`mmvf.cu:127-303`) explicitly avoids
Tensor Cores. Three template branches:

* `T = float`: `float2` loads + FMA into `float sumf[]`.
* `T = half`: either F32-acc (convert `half2 → float2` per load, FMA in
  F32) or F16-acc (accumulate in `half2 sumh2[]`, reduce at end). F16-acc
  is gated by `prec == GGML_PREC_DEFAULT` and `FP16_AVAILABLE`.
* `T = nv_bfloat16`: similar, but HIP path uses raw `int` loads + manual
  extraction (`:235-269`) because `hip_bfloat162` lacks a native
  `v_dot2_f32_f16` intrinsic.

The AMD path uses `v_dot2_f32_f16` for `T = half` when
`V_DOT2_F32_F16_AVAILABLE` is defined (`common.cuh:752-754`), giving 2×
FMA throughput on RDNA2+/CDNA.

### 9.4 PDL and launch bounds

Both kernels call `ggml_cuda_pdl_sync()` near the prologue (`mmvq.cu:513`,
`mmvf.cu:28`) and `ggml_cuda_pdl_lc()` near the epilogue (`mmvq.cu:757`,
`mmvf.cu:305`). On Hopper+ these expand to `cudaGridDependencySynchronize()`
and `cudaTriggerProgrammaticLaunchCompletion()` respectively (see ARTX08
§9.4 for the launcher-side plumbing). MMVQ uses
`__launch_bounds__(calc_nwarps(...) * warp_size, 1)` (`mmvq.cu:479`) —
1 block per SM resident, trading occupancy for register pressure relief
on the IQ2/IQ3 paths. MMVF has no explicit `__launch_bounds__`.

---

## 10. Quantization Strategy

### 10.1 Inline `vec_dot_q_cuda` device-function pointer

`get_vec_dot_q_cuda(ggml_type type)` (`mmvq.cu:10-36`) is a
`constexpr __device__` function that returns a function pointer to the
right `vec_dot_*_q8_1` from `vecdotq.cuh`. The pointer is stored in a
`constexpr` local (`:500`) and the call is devirtualised by `nvcc` at
`-O3`. Each `vec_dot_*` function reads one quant block from `vx` and
one Q8_1 block from `y`, performs `vdr` `dp4a` reductions, multiplies
by the per-block scale(s), and returns a single `float`.

### 10.2 No dequantize-to-shared-memory

The `dequantize.cuh` functions (e.g., `dequantize_q4_0`, `dequantize_q4_K`)
are **not** used by MMVQ. They are used by the convert kernels
(`convert.cu`) and by the legacy MMQ path. MMVQ instead fuses the
dequantize + dot product into a single `vec_dot_q_cuda` call, with the
dequantized values living only in registers for the duration of the
`dp4a` reduction. This is the single most important optimisation in the
GEMV path: it avoids a shared-memory round-trip per quant block.

### 10.3 Activation quantization (F32 → Q8_1)

The MMVQ entry pre-quantizes `src1` to Q8_1 via
`quantize_row_q8_1_cuda` (`mmvq.cu:1228`). The Q8_1 block layout stores
both the per-32-element scale `d` (half) and the per-32-element
absolute-value sum `s` (half) in `block_q8_1.ds` (a `half2`). The `s`
field is used by the Q4_0/Q5_0/Q8_0 vec_dot implementations to subtract
the implicit `-8` bias (for symmetric quants) — see
`vec_dot_q4_0_q8_1_impl` (`vecdotq.cuh:115-134`):
`return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y)`.

### 10.4 NVFP4 / MXFP4 special case

`vec_dot_nvfp4_q8_1` (`vecdotq.cuh:331`) and `vec_dot_mxfp4_q8_1` (`:307`)
use the `kvalues_mxfp4` / `kvalues_iq4nl` lookup tables
(`get_int_from_table_16`, `vecdotq.cuh:34-95`) to expand 4-bit indices
into `int8` values, then `dp4a`. NVFP4 additionally supports the
`x_scale` / `gate_scale` fusion args (`mmvq.cu:541-546, 662-670`) — the
only quant type with scale fusion.

### 10.5 Supported quant formats

MMVQ dispatch (`mmvq.cu:1006-1143`) covers: Q1_0, Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS, IQ1_S, IQ1_M, IQ4_NL, IQ4_XS, IQ3_S. Twenty-two quant
formats in total. Each gets its own template instantiation per
`ncols_dst` (1..8) and `has_fusion` (true/false), so the compiled
binary contains `22 * 8 * 2 = 352` MMVQ kernel instantiations (plus
`small_k` variants for `ncols_dst == 1`).

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions with correctness
consequences.

### 11.1 Floating-point reassociation

* **Per-thread `vdr` accumulation**. Each `vec_dot_q_cuda` call sums
  `vdr` `dp4a` results in an `int sumi` accumulator (`vecdotq.cuh:118, 245,
  260, 290`). The integer sum is exact; reassociation happens only in the
  float post-multiply.
* **Cross-warp reduction in MMVQ**. `tmp_shared[l][j][i][threadIdx.x]`
  is summed across `l ∈ [0, nwarps-1)` (`mmvq.cu:644-645`) in ascending
  `l` order. Deterministic per warp configuration; differs from a
  left-to-right sequential sum at the ULP level.
* **Warp shuffle in MMVF**. `warp_reduce_sum<warp_size>` uses
  `__shfl_xor_sync` butterfly reduction (`common.cuh:455-462`), which
  reassociates differently from `__shfl_down_sync`.
* **F16 accumulation in MMVF**. When `type_acc = half`, the kernel
  accumulates into `half2 sumh2[ncols_dst]` (`mmvf.cu:193-220`). F16 has
  10 bits of mantissa; for K > 1024 the accumulated sum can overflow or
  lose precision. Opt out via `dst->op_params[0] = GGML_PREC_F32`.

### 11.2 Quantization rounding

* **Q8_1 activation quantization**. `quantize_q8_1` rounds each F32
  value to int8 and stores the per-block scale `d = max_abs / 127`. The
  resulting Q4×Q8 matmul is the deliberate accuracy/speed tradeoff of
  all llama.cpp quantized paths (matches ARTX01 §11.3).
* **Per-block scale broadcast**. The scale `d4` (weight block) is
  multiplied by `ds8f.x` (activation block scale) in F32 after the int8
  dot product. The subtraction `(8*vdr/QI4_0) * ds8f.y` uses the
  activation block's abs-sum `s` to remove the implicit `-8` bias of the
  symmetric Q4_0 encoding. Bit-exact across runs.

### 11.3 Padding zeroing

If `src0` is a temporary compute buffer with `size_alloc > size_data`,
`ggml_cuda_mul_mat_vec_q` issues `cudaMemsetAsync` on the tail bytes
(`mmvq.cu:1212-1220`). Without this, `vec_dot_q_cuda` calls that read
past the data end could pick up non-zero garbage in the padding, which
would be multiplied by the activation scale and corrupt the result.
Correctness requirement, not an optimisation.

### 11.4 Determinism

* **MMVQ / MMVF / MoE kernel** all use deterministic reduction orders
  for fixed `(type, ncols_dst, nwarps, K, CC)`. No dynamic chunk
  stealing (contrast ARTX01 §11.4).
* **Atomic accumulation**: none. Output writes are to disjoint
  addresses; no atomics on `dst`.
* **Conclusion**: GEMV output is bit-reproducible across runs for fixed
  `(kernel, shape, dtype, CC)`. Variation across CC is expected
  (different `nwarps`, `block_size`, `vdr`).

### 11.5 Architecture-specific assumptions

* `V_DOT2_F32_F16_AVAILABLE` is only defined for RDNA2+/CDNA AMD
  (`common.cuh:752-754`). On NVIDIA, the same `ggml_cuda_mad(float&,
  half2, half2)` falls back to `__hmul2` + scalar add. Same input →
  different reduction tree → different ULPs.
* `warp_size` is 32 (NVIDIA) or 64 (AMD). MMVQ's
  `block_dims(warp_size, nwarps, 1)` produces a 128-thread block on
  NVIDIA but 256-thread on AMD; `blocks_per_iter` scales accordingly.
* MMVF's auto-tune caps `max_block_size = 256` on NVIDIA, 128 on
  pre-RDNA1 AMD (`mmvf.cu:436-438`).

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                          | Where                                | Notes                                                                                  |
| ------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| Template-on-`ncols_dst` enumeration   | `mmvq.cu:913-996`, `mmvf.cu:541-601` | 8 (MMVQ) / 8 (MMVF) compile-time-constant output-column counts; lets compiler unroll inner loop and hold per-column accumulators in registers. |
| Inline dequantize fusion              | `mmvq.cu:602`, `vecdotq.cuh:115-…`   | Dequantize happens inside `vec_dot_q_cuda`, no shared-memory staging. 1 register-resident F32 per (j, i) accumulator. |
| `dp4a` 4-way int8 dot                 | `vecdotq.cuh:126, 151, …`            | 4× throughput vs scalar 8-bit MADD on supporting hardware.                             |
| `half2` FMA / `v_dot2_f32_f16`        | `mmvf.cu:207`, `common.cuh:758`      | 2× FMA throughput on AMD RDNA2+; equivalent on NVIDIA via `__hmul2` + scalar add.      |
| Per-arch tuning tables                | `mmvq.cu:64-244, 352-456`            | 6 arch tables × per-type `nwarps`/`rows_per_block`/`mmid_max_batch` selection.         |
| Auto-tuned block size                 | `mmvf.cu:433-445`                    | Picks smallest `block_size ∈ {32,64,…,256}` that minimises K iterations.               |
| `should_use_small_k` heuristic        | `mmvq.cu:861-901`                    | Switches to `rows_per_block = nwarps` when K is short to keep warps busy.              |
| Warp-shuffle butterfly reduction      | `common.cuh:455-462`                 | `__shfl_xor_sync` reduction; uses `__reduce_add_sync` on Ampere+ for `int`.            |
| `constexpr` device-function pointer   | `mmvq.cu:10-36, 500`                 | `get_vec_dot_q_cuda(type)` is `constexpr __device__`; the call is devirtualised.       |
| PDL sync / launch-completion          | `mmvq.cu:513, 757`, `mmvf.cu:28, 305`| Hopper+ overlap with downstream kernel.                                                |
| `__launch_bounds__(…, 1)` for MMVQ    | `mmvq.cu:479`                        | 1 block per SM; trades occupancy for register headroom on IQ paths.                    |
| Padding zeroing                       | `mmvq.cu:1212-1220`                  | Required for correctness (so padding reads don't pick up NaN scales).                  |
| MoE-dedicated kernel (no smem)        | `mmvq.cu:705-769`                    | One warp per token; pure warp-reduce, no shared-memory cross-warp reduction.            |
| NVFP4 scale fusion                    | `mmvq.cu:541-546, 662-670`           | Only quant type with `x_scale` / `gate_scale` fusion; saves a separate scale-mul kernel.|
| FFNGLU fusion (gate + bias + GLU)     | `mmvq.cu:533-583, 660-687`; `mmvf.cu:63-95, 347-372` | Saves a separate gate matmul + bias + GLU kernel; only for `ncols_dst == 1`. |

### 12.2 Optimizations *not* present (worth noting)

* **No activation quantization caching.** `quantize_row_q8_1_cuda`
  re-runs on every MMVQ call; no caching across decode steps or matmuls
  sharing the same `src1`.
* **No persistent L2 hints** (`cudaStreamSetAttribute` with
  `cudaStreamAttributeAccessPolicyWindow`). The weight matrix would
  benefit from L2 persistence across decode steps.
* **No async copy (`cp.async`)** in MMVQ / MMVF. MMF uses `ldmatrix` but
  not the Ampere `cp.async` pipeline; MMQ does (ARTX10).
* **No Tensor Cores in MMVF.** MMVF uses CUDA cores only; MMF is
  preferred whenever MMA hardware is available, leaving MMVF as the
  fallback for pre-Volta NVIDIA and non-MMA AMD.
* **No `int4` / `int8` MMA** in MMVQ. Tensor Cores are reserved for MMQ
  (ARTX10) and MMF; MMVQ relies on `dp4a`. Hopper's `mma.m8n8k32` int4
  instruction is unused in the GEMV path.
* **No split-K parallelism.** Both kernels reduce K within a single
  block; no atomic-accumulate split-K scheme for very long K.
* **No cooperative kernels** (`cooperative_groups`). The MoE MMVQ
  kernel could potentially benefit from grid-group reduction.

---

## 13. Architectural Strengths

1. **Template-on-`ncols_dst` enumeration** is the single best design
   decision in the GEMV path. By making the output-column count a
   compile-time constant, the compiler unrolls the inner loop, holds
   per-column accumulators in registers, and eliminates the per-column
   branch.

2. **Inline `vec_dot_q_cuda` fusion** combines dequantize + dot + scale
   into one device function called directly inside the K loop,
   eliminating the shared-memory round-trip. This is why MMVQ stays
   register-bound on IQ2/IQ3 paths despite their complex dequantize
   arithmetic.

3. **Per-arch tuning tables with host/device agreement.** The
   `mmvq_parameter_table_id` enum and dual host/device
   `get_device_table_id` (`mmvq.cu:73-106`) make the tuning policy a
   compile-time constant inside the kernel while letting the host launch
   the correct template instance.

4. **MoE-dedicated kernel** (`mul_mat_vec_q_moe`) lets the MoE case skip
   the shared-memory cross-warp reduction entirely. One warp per (token,
   expert) is the natural granularity for `MUL_MAT_ID` with `ne2 ≤ 8`.

5. **Auto-tuned block size in MMVF** (`mmvf.cu:433-445`) picks the
   smallest block size that achieves the minimum K-iteration count. A
   cheap, deterministic, no-state auto-tuner requiring no benchmarking.

6. **FFNGLU fusion prologue / epilogue** integrates bias + gate matmul +
   SiLU/GELU/SwiGLU-OAI directly into the GEMV kernel, gated on
   `ncols_dst == 1` (single-token decode) where the fusion saves the
   most bandwidth.

7. **PDL integration** via `ggml_cuda_pdl_sync()` / `_lc()` calls hooks
   into the Hopper+ Programmatic Dependent Launch pipeline without
   conditional compilation in the kernel body — macros expand to no-ops
   on non-Hopper hardware.

---

## 14. Architectural Weaknesses

### W1 — MMVF is under-utilised on Ampere+ NVIDIA

**Evidence**: `ggml_cuda_should_use_mmvf` (`mmvf.cu:823-824`) returns
`src0_small && ne11 == 1` for F16/BF16 on Ampere+ NVIDIA. MMVF only
wins the dispatch for true GEMV; for `ne11 ∈ {2..16}`, MMF wins. The
MMVF template instantiations for `ncols_dst ∈ {2..8}` on Ampere+ are
never launched.

**Impact**: ~384 maintained-but-unused kernel variants on Ampere+.
Either extend MMVF to win a wider range, or remove the `ncols_dst ≥ 2`
MMVF paths on Ampere+.

### W2 — Mandatory activation re-quantization on every MMVQ call

**Evidence**: `mmvq.cu:1223-1229` always allocates a Q8_1 scratch and
calls `quantize_row_q8_1_cuda`. No caching across decode steps or across
matmuls sharing the same `src1`.

**Impact**: For a 32-layer model with 7 matmuls per layer and batch 1,
that's 224 quantization launches per token, each reading 16 KiB F32 and
writing 4 KiB Q8_1. Estimated 10-20% of decode-step time.

### W3 — Hand-maintained per-arch batch-size tables

**Evidence**: `get_mmvq_mmid_max_batch_*` (`mmvq.cu:112-244`) — six
functions, each with a 15-25-arm `switch` over `ggml_type`. The PR
reference at `:110` (PR #20905) shows these are empirically tuned.

**Impact**: Maintenance burden grows with `(arch count) × (quant
count)`. A single wrong entry can regress performance by 10× with no
compile-time warning.

### W4 — `__launch_bounds__(…, 1)` limits MMVQ occupancy to 1 block per SM

**Evidence**: `mmvq.cu:479` sets `__launch_bounds__(nwarps * warp_size,
1)` uniformly for all types.

**Impact**: Trades occupancy for register headroom — the right call for
IQ2/IQ3 (register-heavy) but overly conservative for Q4_0/Q8_0
(register-light). A per-`type` launch-bound policy would be more
appropriate.

### W5 — `should_use_small_k` heuristic has hard-coded type blacklists

**Evidence**: `mmvq.cu:872-898` defines three arrays (`iq_slow_turing`,
`iq_slow_other`, `slow_pascal`) that list specific quants where
`small_k` should be disabled per arch, with no first-principles
justification.

**Impact**: When a new quant is added, the lists must be updated, or
the new quant silently gets the wrong `small_k` policy. No compile-time
check catches a missing entry.

### W6 — No Q8_1 deduplication for MoE

**Evidence**: `ggml_cuda_mul_mat_vec_q` quantizes the full
`ne13*ne12 * ne11*ne10_padded` activation tensor. The MoE-dedicated
`mul_mat_vec_q_moe` reads the same Q8_1 buffer for all experts —
correct, but the quantization is done per-token-per-channel, not
deduplicated across experts.

**Impact**: For `n_experts_used = 8` and `n_tokens = 1`, the
quantization does 8× the work it needs to. The MoE-dedicated
`quantize_scatter_mmq_q8_1_cuda` (`quantize.cu:606-633`) exists for the
MMQ path but is not used by MMVQ.

### W7 — MMVF F16-accumulate path can lose precision silently

**Evidence**: `mmvf.cu:614-622` selects `type_acc = half` when `T ==
half && prec == GGML_PREC_DEFAULT`. F16 accumulation is therefore the
default for F16 weights.

**Impact**: For K > 1024, F16 accumulation can overflow (max F16 ≈
65504) or lose precision (10-bit mantissa). The user must explicitly
set `op_params[0] = GGML_PREC_F32` to opt out. No warning is logged.

### W8 — `mul_mat_vec_q_switch_type` is a 23-arm switch with copy-pasted bodies

**Evidence**: `mmvq.cu:1006-1143`. Each arm calls
`mul_mat_vec_q_switch_ncols_dst<type>(...)` with identical arguments;
only the template parameter differs.

**Impact**: 130 lines of near-duplicate code. A code generator (Python
script emitting the switch arms from a single type list) would be more
maintainable.

### W9 — No fallback for misaligned MMVF inputs

**Evidence**: `ggml_cuda_should_use_mmvf` (`mmvf.cu:786-801`) returns
`false` if any `nb[i] % (2*ts) != 0`. No unaligned MMVF variant exists.

**Impact**: Tensors with odd-row strides silently fall through to MMF
or cuBLAS, which may be slower than a hypothetical unaligned MMVF.
Coverage gap, not a correctness issue.

### W10 — MMVQ shared-memory reduction is hard-coded to `nwarps ≤ 8`

**Evidence**: `mmvq.cu:614` declares `tmp_shared[nwarps-1][…]` with
`nwarps` constexpr from `calc_nwarps`, which returns at most 8 (RDNA4
path, `:400`).

**Impact**: Hard upper bound on `nwarps` baked into the kernel
structure. A future arch needing `nwarps = 16` would silently overflow
shared memory.

### W11 — Warp-reduce uses `__shfl_xor_sync`, not `__shfl_down_sync`

**Evidence**: `common.cuh:455-462` — `warp_reduce_sum<float>` uses
butterfly `__shfl_xor_sync` rather than tree `__shfl_down_sync`.

**Impact**: Functionally equivalent for a full-warp reduction. The
butterfly form is more general (supports sub-warp reductions via the
`width` parameter) but marginally slower on some hardware. Negligible
in practice.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda` | **ADOPT** | Template-on-`ncols_dst` enumeration | Lets the compiler unroll the output-column loop and hold per-column accumulators in registers. The 8-way switch is verbose but trivially generated. |
| `glcuda` | **ADOPT** | Inline `vec_dot_q_cuda` device-function-pointer pattern | Fuses dequantize + dot + scale into one call; no shared-memory staging. The `constexpr` device function returns a function pointer that is devirtualised by `nvcc`. |
| `glcuda` | **ADOPT** | MMVF auto-tuned block-size search | Cheap, deterministic, no-state auto-tuner. Picks smallest block size that minimises K-iteration count. |
| `glcuda` | **ADOPT** | MoE-dedicated GEMV kernel | One warp per (token, expert); no shared-memory cross-warp reduction. Natural granularity for `MUL_MAT_ID` with `ne2 ≤ 8`. |
| `glcuda` | **ADAPT** | Per-arch tuning tables | Keep the per-arch structure but generate the tables from a single source of truth (e.g., a Python script that emits both host and device tables). Avoid the dual host/device `get_device_table_id` duplication. |
| `glcuda` | **ADAPT** | FFNGLU fusion (bias + gate + GLU) | Keep the fusion but extend to `ncols_dst > 1` (multi-token decode). The current `ncols_dst == 1`-only restriction is artificial. |
| `glcuda` | **REJECT** | Mandatory Q8_1 re-quantization per call | Cache the Q8_1 form across decode steps when the same `src1` is consumed by multiple matmuls (KV-cache, MoE experts). |
| `glcuda` | **REJECT** | `__launch_bounds__(…, 1)` for all MMVQ types | Use a per-`type` launch-bound policy: `1` for IQ2/IQ3 (register-heavy), `2` or `4` for Q4_0/Q8_0 (register-light). |
| `glcuda` | **REJECT** | MMVF `should_use_mmvf` policy on Ampere+ | Either extend MMVF to win a wider range, or remove the `ncols_dst ≥ 2` MMVF paths on Ampere+ (they are dead code). |
| `glcuda` | **MONITOR** | F16-accumulate default in MMVF | Watch for precision regressions on large-K matmuls. Consider making F32-acc the default with F16-acc as opt-in. |
| `glcuda` | **MONITOR** | `should_use_small_k` heuristic blacklists | Watch for regressions when new quants are added; consider replacing the hard-coded type lists with a runtime cost model. |
| `glcuda` | **DEFER** | PDL integration | Adopt PDL plumbing from ARTX08; the kernel-side `pdl_sync` / `pdl_lc` calls are trivial once the launcher is in place. |
| `GATE` | **ADOPT** | `ncols_dst` as a graph-plan-time constant | The graph planner knows the decode batch size; it can pin `ncols_dst` at plan time so the right template instance is launched without runtime dispatch. |
| `GATE` | **ADAPT** | FFNGLU fusion detection | Move fusion detection from execution time to plan time (same recommendation as ARTX01-F08 / ARTX08). Extend to `MUL_MAT_ID + ADD + GLU` for MoE FFN. |

---

## 16. Recommendations

### R1 — ADOPT template-on-`ncols_dst` enumeration
**Priority:** Critical **Difficulty:** M **Dependencies:** none
GwenLand's `glcuda` GEMV kernel should be templated on `ncols_dst ∈
{1..8}` with a runtime `switch` selecting the instantiation (same as
`mul_mat_vec_q_switch_ncols_dst`). The compiler does the unrolling; the
switch is the only boilerplate.

### R2 — ADOPT inline `vec_dot` device-function-pointer pattern
**Priority:** Critical **Difficulty:** M **Dependencies:** R1
Fuse dequantize + dot + scale into a single `vec_dot_q_cuda` device
function called directly inside the K loop. The function pointer should
be `constexpr`-resolved at compile time. Eliminates the shared-memory
dequantize staging that a naive design would require.

### R3 — ADOPT auto-tuned block size for F16/BF16/F32 GEMV
**Priority:** High **Difficulty:** S **Dependencies:** R1
Replicate `launch_mul_mat_vec_f_cuda`'s block-size search
(`mmvf.cu:433-445`): pick smallest `block_size ∈ {32, …, 256}` that
minimises K iterations.

### R4 — ADOPT MoE-dedicated GEMV kernel
**Priority:** High **Difficulty:** M **Dependencies:** R1, R2
For `MUL_MAT_ID` with `ne2 ≤ 8`, launch a separate kernel with one warp
per (token, expert). No shared-memory cross-warp reduction (mirrors
`mul_mat_vec_q_moe`).

### R5 — REJECT mandatory Q8_1 re-quantization; add caching
**Priority:** High **Difficulty:** L **Dependencies:** R2, GATE design
Cache the Q8_1 form of `src1` across decode steps when the same tensor
is consumed by multiple matmuls. Cache key: `(src1 data ptr, src1 ne[0],
src1 type, src0 type)`. Invalidate on graph re-plan or data-ptr change.
Expected to save ~10-20% of decode-step time on 7B+ models.

### R6 — ADAPT per-arch tuning tables to a single source of truth
**Priority:** Medium **Difficulty:** M **Dependencies:** R1, R2
Generate the per-arch `(nwarps, rows_per_block, mmid_max_batch)` tables
from a Python script that emits both host and device tables. Avoid the
dual host/device duplication in `mmvq.cu:73-106`.

### R7 — ADAPT FFNGLU fusion to multi-token decode
**Priority:** Medium **Difficulty:** L **Dependencies:** R1, R2
Extend the `has_fusion` path to `ncols_dst > 1`. The current
`ncols_dst == 1`-only restriction is artificial: the gate / bias / GLU
arithmetic is per-output-column and parallelises naturally.

### R8 — REJECT `__launch_bounds__(…, 1)` for all MMVQ types
**Priority:** Medium **Difficulty:** S **Dependencies:** R1
Use a per-`type` launch-bound policy: `min_blocks_per_sm = 1` for
IQ2/IQ3 (register-heavy), `2` or `4` for Q4_0/Q8_0 (register-light).
Requires the launch-bound to be a template parameter or a `constexpr`
derived from `type`.

### R9 — MONITOR MMVF `should_use_mmvf` policy
**Priority:** Low **Difficulty:** M **Dependencies:** R3
The current policy under-utilises MMVF on Ampere+. Either extend MMVF
to win a wider range (e.g., `ne11 ≤ 4`), or remove the `ncols_dst ≥ 2`
MMVF paths on Ampere+ and let MMF own that range. Decide based on
benchmarking.

### R10 — ADOPT PDL kernel-side hooks
**Priority:** Medium **Difficulty:** XS **Dependencies:** ARTX08 PDL launcher
Call `ggml_cuda_pdl_sync()` near the kernel prologue and
`ggml_cuda_pdl_lc()` near the epilogue, mirroring `mmvq.cu:513, 757`
and `mmvf.cu:28, 305`. On non-Hopper hardware these expand to no-ops.

---

## 17. Findings

### Finding ARTX09-F01

```
Finding ID:           ARTX09-F01
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMVQ template kernel (mul_mat_vec_q)
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             mul_mat_vec_q
Lines:                478-699
Summary:              One-warp-per-(row,column) GEMV kernel with template
                      parameters for type, ncols_dst, has_fusion, small_k.
Observation:          The kernel is launched with block_dims(warp_size,
                      nwarps, 1) where nwarps is per-arch/per-type/per-
                      ncols_dst tuned. Each block owns rows_per_cuda_block
                      (=1 or 2) weight rows × ncols_dst output columns.
                      The K dimension is tiled in strides of
                      blocks_per_iter = vdr * nwarps * warp_size / qi
                      quant blocks. Per-thread accumulators
                      float tmp[ncols_dst][rows_per_cuda_block] live in
                      registers. Cross-warp reduction uses shared memory
                      (tmp_shared[nwarps-1][…]) plus warp_reduce_sum
                      butterfly shuffle.
Evidence:             mmvq.cu:478-699 (kernel body); :505 (blocks_per_iter);
                      :586-612 (K loop); :614 (shared mem decl);
                      :644-657 (cross-warp + warp reduction).
Architectural Impact: This is the dominant kernel for autoregressive
                      decode with quantized weights. The
                      template-on-ncols_dst enumeration lets the compiler
                      unroll the output-column loop and hold per-column
                      accumulators in registers, which is the single
                      biggest perf win for small ncols_dst.
Correctness Impact:   Reduction order is deterministic per (type,
                      ncols_dst, nwarps, K) configuration. Bit-reproducible
                      across runs.
Optimization Type:    SIMD / tiling / blocking / kernel fusion (inline
                      dequantize via vec_dot_q_cuda).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same template structure in glcuda, with
                      per-arch tuning tables adapted per R6.
Priority:             Critical
Difficulty:           L
Dependencies:         none
Confidence:           High
```

### Finding ARTX09-F02

```
Finding ID:           ARTX09-F02
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            MMVQ inline dequantize fusion
Source File:          ggml/src/ggml-cuda/mmvq.cu, ggml/src/ggml-cuda/vecdotq.cuh
Function:             mul_mat_vec_q (K loop), vec_dot_q4_0_q8_1_impl
Lines:                mmvq.cu:602-609; vecdotq.cuh:115-134
Summary:              Dequantize + dot product + scale multiply are fused
                      into a single vec_dot_q_cuda device function call;
                      no shared-memory staging of dequantized weights.
Observation:          Inside the K loop, the kernel calls
                      vec_dot_q_cuda(vx, &y[...], kbx, kqs) which internally
                      (a) reads one quant block from vx, (b) reads one
                      Q8_1 block from y, (c) does vdr dp4a reductions into
                      int sumi, (d) multiplies by per-block scales in F32.
                      The dequantized weight values exist only in
                      registers for the duration of the dp4a reduction and
                      are never written to shared memory. This is in
                      contrast to the dequantize.cuh functions
                      (dequantize_q4_0, dequantize_q4_K, ...) which write
                      to a dst_t* yy buffer and are used by the convert
                      kernels and legacy MMQ path, not by MMVQ.
Evidence:             mmvq.cu:602-609 (vec_dot_q_cuda call site);
                      vecdotq.cuh:115-134 (vec_dot_q4_0_q8_1_impl with
                      dp4a reductions and scale multiply);
                      dequantize.cuh:26-38 (dequantize_q4_0 writes to
                      float2& v, not used by MMVQ).
Architectural Impact: Eliminates a shared-memory round-trip per quant
                      block. The kernel stays register-bound, which is
                      critical for the IQ2/IQ3 paths where dequantize
                      involves lookup-table fetches and would overflow
                      shared memory if staged.
Correctness Impact:   None. The arithmetic is mathematically equivalent
                      to a dequantize-then-dot sequence.
Optimization Type:    Kernel fusion / SIMD (dp4a).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same pattern in glcuda: a constexpr device
                      function pointer resolved at compile time, called
                      directly inside the K loop.
Priority:             Critical
Difficulty:           M
Dependencies:         ARTX09-F01
Confidence:           High
```

### Finding ARTX09-F03

```
Finding ID:           ARTX09-F03
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMVQ per-arch tuning tables
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             calc_nwarps, calc_rows_per_block, get_mmvq_mmid_max_batch_*
Lines:                64-244, 352-476
Summary:              Six per-arch tables (GENERIC, TURING, GCN, RDNA2,
                      RDNA3_0, RDNA4) select nwarps, rows_per_block, and
                      mmid_max_batch as a function of (type, ncols_dst).
Observation:          The tables are implemented as 15-25-arm switch
                      statements, one per arch. Each switch returns an
                      int (nwarps or rows_per_block or max_batch). The
                      host-side get_device_table_id(int cc) and the
                      device-side get_device_table_id() must agree; both
                      are maintained by hand. The PR reference at :110
                      (PR #20905) shows the batch-size tables are
                      empirically tuned per (arch, quant).
Evidence:             mmvq.cu:64-71 (enum); :73-87 (device selector);
                      :89-106 (host selector); :112-244 (per-arch batch
                      functions); :352-456 (calc_nwarps); :458-476
                      (calc_rows_per_block).
Architectural Impact: Per-arch tuning is essential for the IQ2/IQ3 paths
                      where nwarps=8 regresses due to register pressure
                      on some archs but helps on others. The dual
                      host/device tables are a maintenance burden but
                      enable compile-time-constant nwarps inside the
                      kernel.
Correctness Impact:   None.
Optimization Type:    Per-arch heuristic tuning.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Keep the per-arch structure but generate
                      the tables from a single source of truth to avoid
                      the dual host/device duplication.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX09-F01
Confidence:           High
```

### Finding ARTX09-F04

```
Finding ID:           ARTX09-F04
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MMVQ dispatch chain
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             mul_mat_vec_q_switch_type, _switch_ncols_dst, _switch_fusion
Lines:                783-1143
Summary:              Three-level dispatch chain (switch_type → switch_ncols_dst
                      → switch_fusion) instantiates 22 types × 8 ncols_dst × 2
                      fusion × 2 small_k = 704 kernel variants.
Observation:          Each switch arm is near-identical copy-paste calling
                      the next layer with a different template parameter.
                      The chain exists because nvcc cannot template-
                      instantiate on a runtime value; the switch converts
                      the runtime type/ncols_dst/fusion flags into
                      compile-time template arguments. The compiled binary
                      contains all 704 variants; the linker discards
                      unused ones per .so.
Evidence:             mmvq.cu:998-1143 (switch_type, 23 arms);
                      :837-997 (switch_ncols_dst, 8 arms + MoE branch);
                      :783-813 (switch_fusion, 2-way on has_fusion &&
                      ncols_dst == 1).
Architectural Impact: The chain is verbose (350+ lines of boilerplate)
                      but trivially generated. Compilation time is the
                      main cost — 704 template instantiations per .cu
                      file. The template-instances/ directory separates
                      them into per-type .cu files to parallelise
                      compilation.
Correctness Impact:   None.
Optimization Type:    Template instantiation / compile-time dispatch.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same structure in glcuda. Consider a code
                      generator (Python script) to emit the switch arms
                      from a single type list.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX09-F01
Confidence:           High
```

### Finding ARTX09-F05

```
Finding ID:           ARTX09-F05
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMVF F16/BF16/F32 GEMV kernel
Source File:          ggml/src/ggml-cuda/mmvf.cu
Function:             mul_mat_vec_f, launch_mul_mat_vec_f_cuda
Lines:                7-379, 412-505
Summary:              CUDA-core-only GEMV for F32/F16/BF16 weights; one
                      block per output row, auto-tuned block_size, optional
                      F16 accumulation, no Tensor Cores.
Observation:          The kernel template parameters are (T, type_acc,
                      ncols_dst, block_size, has_fusion, is_multi_token_id).
                      Block layout is 1D block_size threads; grid is
                      (nrows, nchannels_dst, nsamples_or_ntokens). The K
                      dimension is parallelised by for (col2 = tid; col2 <
                      ncols2; col2 += block_size). Loads are float2/half2/
                      nv_bfloat162; reduction is warp_reduce_sum +
                      optional shared-memory cross-warp stage when
                      block_size > warp_size. Auto-tune picks the smallest
                      block_size in {32,64,...,256} that minimises
                      niter = ceil(ncols / (2*block_size)).
Evidence:             mmvf.cu:7-379 (kernel); :127-303 (per-type branches
                      for float/half/nv_bfloat16); :305-339 (reduction);
                      :412-505 (auto-tune + launch).
Architectural Impact: MMVF is the GEMV path for non-quantized weights on
                      pre-Volta NVIDIA and non-MMA AMD. On Ampere+ NVIDIA
                      it is largely bypassed by MMF (see F10). The
                      auto-tune is a clean, no-state heuristic that could
                      be reused elsewhere.
Correctness Impact:   F16-accumulate path can lose precision for K > 1024
                      (10-bit mantissa, max ~65504). User must opt out
                      via op_params[0] = GGML_PREC_F32.
Optimization Type:    Auto-tuned block size / half2 FMA / warp shuffle
                      reduction.
GwenLand Target:      glcuda
Recommendation:       ADOPT the auto-tune; MONITOR the F16-acc default;
                      consider making F32-acc the default for F16 weights
                      with K > 1024.
Priority:             High
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX09-F06

```
Finding ID:           ARTX09-F06
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            MMVF precision selection
Source File:          ggml/src/ggml-cuda/mmvf.cu
Function:             mul_mat_vec_f_cuda, mul_mat_vec_f (F16-acc branch)
Lines:                604-627, 191-220
Summary:              F16 weights default to F16-accumulation; F32 or BF16
                      always accumulate in F32; user can force F32 via
                      op_params[0] = GGML_PREC_F32.
Observation:          mul_mat_vec_f_cuda checks if T==half && prec==
                      GGML_PREC_DEFAULT, and if so instantiates the
                      half-accumulate variant. The half-acc branch
                      accumulates into half2 sumh2[ncols_dst] and reduces
                      to float only at the end. For K > 1024 this can
                      overflow (max F16 ≈ 65504) or lose precision.
Evidence:             mmvf.cu:614-622 (precision selection); :191-220
                      (half2 accumulator + final __low2float + __high2float
                      reduction).
Architectural Impact: The F16-acc path is ~2x faster than F32-acc on
                      pre-Volta NVIDIA (no F32 FMA for half inputs), but
                      the speedup is marginal on Ampere+ where MMF (Tensor
                      Cores) wins anyway.
Correctness Impact:   F16-acc produces different ULPs than F32-acc for
                      the same input. For K > 1024 the F16-acc sum can
                      silently overflow to ±Inf.
Optimization Type:    Reduced-precision accumulation.
GwenLand Target:      glcuda
Recommendation:       MONITOR. Make F32-acc the default for F16 weights
                      with K > 1024; keep F16-acc as opt-in for known-
                      small-K matmuls.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX09-F05
Confidence:           High
```

### Finding ARTX09-F07

```
Finding ID:           ARTX09-F07
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            MMVF shared memory usage
Source File:          ggml/src/ggml-cuda/mmvf.cu
Function:             mul_mat_vec_f (reduction stage)
Lines:                99-116, 305-339
Summary:              MMVF uses shared memory only when block_size >
                      warp_size; the buffer is a single warp_size*float
                      array for cross-warp partial-sum staging.
Observation:          The kernel declares extern __shared__ char
                      data_mmv[] of size warp_size*sizeof(float) +
                      (has_fusion ? warp_size*sizeof(float) : 0). When
                      block_size == warp_size (the common case for
                      ncols_dst == 1 with K ≤ 64), the shared memory is
                      allocated but never touched — the reduction is
                      pure warp_reduce_sum. When block_size > warp_size
                      (large K), per-warp partials are written to
                      buf_iw[tid/warp_size], synced, then warp 0 reads
                      them back and reduces again.
Evidence:             mmvf.cu:99-116 (smem decl + init); :316-338
                      (conditional cross-warp stage); :449 (nbytes_shared
                      computation).
Architectural Impact: Minimal shared memory footprint means high
                      occupancy. The auto-tune can pick large block_size
                      without smem pressure.
Correctness Impact:   None.
Optimization Type:    Conditional shared-memory staging.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same conditional staging pattern in glcuda.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX09-F05
Confidence:           High
```

### Finding ARTX09-F08

```
Finding ID:           ARTX09-F08
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMF Tensor-Core small-batch GEMM
Source File:          ggml/src/ggml-cuda/mmf.cuh, ggml/src/ggml-cuda/mmf.cu
Function:             mul_mat_f, ggml_cuda_should_use_mmf
Lines:                mmf.cuh:48-294; mmf.cu:133-191
Summary:              MMF uses mma / load_ldmatrix Tensor-Core instructions
                      for F32/F16/BF16 GEMM with ne11 <= 16; covers the
                      boundary between GEMV and true GEMM.
Observation:          The kernel uses tile<16, 8, T> / tile<16, 8, float>
                      types (or 32x4 on Volta) and mma() / load_ldmatrix()
                      intrinsics. rows_per_block is 32 (NVIDIA) or 64
                      (CDNA). The should_use_mmf predicate returns false
                      for quantized types, false for ne11 > 16 (non-MUL_MAT_ID)
                      or ne11 > 128/512 (MUL_MAT_ID depending on nrows),
                      and false on CDNA2/CDNA1 for F16/BF16 (deferred to
                      MMQ). Otherwise returns true if ampere_mma_available
                      or amd_mfma_available.
Evidence:             mmf.cuh:48-294 (kernel with mma intrinsics);
                      mmf.cu:133-191 (should_use_mmf predicate).
Architectural Impact: MMF is the F16/BF16 GEMV path on Ampere+ NVIDIA
                      for ne11 in {2..16}. It is significantly faster
                      than MMVF on those shapes because Tensor Cores
                      deliver 4-8x the FMA throughput of CUDA cores for
                      F16/BF16.
Correctness Impact:   Tensor-Core mma accumulates in F32 by default;
                      no precision regression vs MMVF F32-acc.
Optimization Type:    Tensor-Core mma / ldmatrix.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same Tensor-Core path for small-batch
                      F16/BF16 GEMM. The full MMF design (tile shapes,
                      ldmatrix staging, ids gather) is audited at skim
                      level here; ARTX10 covers the closely-related MMQ
                      Tensor-Core path in detail.
Priority:             High
Difficulty:           XL
Dependencies:         ARTX09-F05
Confidence:           High
```

### Finding ARTX09-F09

```
Finding ID:           ARTX09-F09
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMVQ MoE-dedicated kernel (mul_mat_vec_q_moe)
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             mul_mat_vec_q_moe, mul_mat_vec_q_moe_launch
Lines:                705-835
Summary:              Separate GEMV kernel for multi-token MUL_MAT_ID:
                      one warp per (token, expert), no shared-memory
                      cross-warp reduction.
Observation:          Grid is (ceil(nrows_x / c_rows_per_block),
                      nchannels_dst); block is (warp_size, ncols_dst).
                      Each warp (indexed by threadIdx.y) processes one
                      token independently. The K loop is identical to
                      the main kernel but uses a single tmp[c_rows_per_block]
                      accumulator (no [ncols_dst] dim — each warp owns
                      exactly one column). The reduction is pure
                      warp_reduce_sum; no __syncthreads, no shared memory.
                      rows_per_block is hard-coded to 2 ("best perf based
                      on tuning").
Evidence:             mmvq.cu:705-769 (kernel); :816-835 (launch helper
                      with rows_per_block = 2); :903-910 (call site from
                      switch_ncols_dst when has_ids && ncols_dst > 1).
Architectural Impact: Eliminates the shared-memory cross-warp reduction
                      for the MoE case, which is the natural granularity
                      (one warp per token). Reduces smem pressure to zero.
Correctness Impact:   None.
Optimization Type:    Specialised kernel for MoE routing shape.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same dedicated MoE kernel in glcuda for
                      MUL_MAT_ID with ne2 <= 8.
Priority:             High
Difficulty:           M
Dependencies:         ARTX09-F01
Confidence:           High
```

### Finding ARTX09-F10

```
Finding ID:           ARTX09-F10
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MMVF dispatch policy
Source File:          ggml/src/ggml-cuda/mmvf.cu
Function:             ggml_cuda_should_use_mmvf
Lines:                786-869
Summary:              MMVF's should_use predicate is extremely strict on
                      Ampere+ NVIDIA (ne11 == 1 only for F16/BF16), making
                      MMVF largely dead code on modern NVIDIA GPUs.
Observation:          For F16 on Ampere+ NVIDIA with src0_small, the
                      predicate returns ne11 == 1 (line 824). For BF16 on
                      Ampere+ NVIDIA with src0_small, also ne11 == 1
                      (line 850). For F32 on Ampere+ NVIDIA, returns
                      ne11 <= 3 (line 807). This means MMVF only wins the
                      dispatch for true GEMV (F16/BF16) or very thin
                      batches (F32). For ne11 in {2..16} on Ampere+, MMF
                      (Tensor Cores) wins. The MMVF kernel instantiations
                      for ncols_dst in {2..8} on Ampere+ are therefore
                      never launched.
Evidence:             mmvf.cu:803-819 (F32 branch); :820-845 (F16 branch
                      with src0_small && ne11 == 1 on Ampere); :846-865
                      (BF16 branch, same pattern).
Architectural Impact: The MMVF code path is maintained (8 ncols_dst
                      instantiations × 8 block_size instantiations × 3
                      dtypes × 2 acc types = 384 kernel variants) but
                      mostly unused on Ampere+. Maintenance cost without
                      benefit.
Correctness Impact:   None.
Optimization Type:    None (this is a dispatch policy).
GwenLand Target:      glcuda
Recommendation:       REJECT the current policy. Either (a) extend MMVF
                      to win ne11 in {2..4} on Ampere+ (would require
                      reg-matmul reformulation to compete with mma), or
                      (b) remove the ncols_dst >= 2 MMVF paths on Ampere+
                      and let MMF own that range exclusively.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX09-F05, ARTX09-F08
Confidence:           High
```

### Finding ARTX09-F11

```
Finding ID:           ARTX09-F11
Category:             MEMORY_PATTERN
Engine:               CUDA
Component:            MMVQ activation re-quantization
Source File:          ggml/src/ggml-cuda/mmvq.cu, ggml/src/ggml-cuda/quantize.cu
Function:             ggml_cuda_mul_mat_vec_q (entry), quantize_row_q8_1_cuda
Lines:                mmvq.cu:1222-1229; quantize.cu:558-573
Summary:              F32 src1 is re-quantized to Q8_1 on every MMVQ call;
                      no caching across decode steps or across matmuls
                      sharing the same src1.
Observation:          The entry allocates a per-op scratch buffer of size
                      ne13*ne12 * ne11*ne10_padded * sizeof(block_q8_1)/QK8_1
                      and launches quantize_row_q8_1_cuda to fill it. The
                      scratch is returned to ctx.pool() at function exit.
                      For a 32-layer model with 7 matmuls per layer and
                      batch 1, this is 224 quantization launches per
                      token, each reading 16 KiB F32 and writing 4 KiB
                      Q8_1. The same src1 (e.g., the hidden state between
                      attention and FFN) is quantized multiple times.
Evidence:             mmvq.cu:1222-1229 (alloc + quantize call); :1253
                      (switch_type call with src1_q8_1.get()); quantize.cu:
                      558-573 (quantize_row_q8_1_cuda implementation).
Architectural Impact: Significant bandwidth overhead in the decode hot
                      path. Estimated 10-20% of decode-step time for 7B+
                      models. The Q8_1 form is deterministic given (src1,
                      src0_type) and could be cached.
Correctness Impact:   None. The re-quantization is correct; just redundant.
Optimization Type:    None (this is the absence of caching).
GwenLand Target:      glcuda, GATE
Recommendation:       REJECT the no-cache design. Cache the Q8_1 form
                      keyed by (src1 data ptr, src1 ne[0], src1 type,
                      src0 type). Invalidate on graph re-plan or when
                      src1 data ptr changes.
Priority:             High
Difficulty:           L
Dependencies:         GATE design (cache invalidation hooks)
Confidence:           High
```

### Finding ARTX09-F12

```
Finding ID:           ARTX09-F12
Category:             GPU_KERNEL
Engine:               CUDA
Component:            MMVQ / MMVF small_k heuristic and __launch_bounds__
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             should_use_small_k (lambda), mul_mat_vec_q launch
Lines:                479, 861-901
Summary:              should_use_small_k uses hard-coded type blacklists
                      per arch; __launch_bounds__(…, 1) is applied
                      uniformly to all types regardless of register
                      pressure.
Observation:          The should_use_small_k lambda (mmvq.cu:861-901)
                      defines three arrays (iq_slow_turing, iq_slow_other,
                      slow_pascal) that list specific quants where
                      small_k should be disabled. The lists are based on
                      tuning runs and have no first-principles
                      justification. Separately, __launch_bounds__
                      (calc_nwarps(...) * warp_size, 1) at line 479
                      forces exactly 1 block per SM resident for all
                      types, including register-light Q4_0/Q8_0 that
                      could benefit from higher occupancy.
Evidence:             mmvq.cu:479 (__launch_bounds__ with min_blocks=1);
                      :872-898 (hard-coded type blacklist arrays); :889
                      (is_nvidia_turing_plus check); :894-897
                      (is_nvidia_pascal_older check).
Architectural Impact: The blacklists must be manually updated when new
                      quants are added, with no compile-time check. The
                      uniform launch-bound leaves register-light types
                      under-occupied.
Correctness Impact:   None.
Optimization Type:    Heuristic-based tuning.
GwenLand Target:      glcuda
Recommendation:       ADAPT. Replace the type blacklists with a runtime
                      cost model (e.g., "disable small_k if
                      vec_dot_register_count(type) > threshold"). Use
                      per-type launch_bounds: min_blocks=1 for IQ2/IQ3,
                      min_blocks=2 or 4 for Q4_0/Q8_0.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX09-F01, ARTX09-F03
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether MMVF's `ncols_dst >= 2` paths on Ampere+ are ever
  launched in practice. `should_use_mmvf` (`mmvf.cu:823-824`) requires
  `ne11 == 1` for F16/BF16 on Ampere+, which maps to `ncols_dst == 1`.
  Static analysis suggests the `ncols_dst >= 2` MMVF paths are dead code
  on Ampere+; requires runtime tracing to confirm.
* **U2**. The actual Q8_1 quantization overhead as a fraction of
  decode-step time. The 10-20% estimate in F11 is bandwidth-counting
  (224 launches × 20 KiB); requires profiling.
* **U3**. Whether the F16-accumulate path in MMVF overflows for typical
  LLM K dimensions (4096-12288). Max F16 ≈ 65504; expected accumulator
  magnitude for typical activations is ~400, but outliers may overflow.
  Requires stochastic testing.
* **U4**. Whether the per-arch `mmvq_parameter_table_id` tables are
  optimal for Blackwell. The tables include RDNA4 entries but Blackwell
  uses the GENERIC table (no dedicated `MMVQ_PARAMETERS_BLACKWELL`).
  Requires benchmarking on Blackwell hardware.
* **U5**. Whether the `should_use_small_k` heuristic (`mmvq.cu:861-901`)
  could be replaced by a single first-principles rule based on
  register-count. The hard-coded type blacklists suggest exceptions that
  may be arch-specific register-pressure effects.
* **U6**. Whether `__shfl_xor_sync` butterfly reduction
  (`common.cuh:455-462`) is measurably slower than `__shfl_down_sync`
  tree reduction on any target hardware. Requires micro-benchmarking.
* **U7**. Whether `mul_mat_vec_q_moe`'s hard-coded `rows_per_block = 2`
  (`mmvq.cu:824`) is optimal for all (arch, type) combinations. The
  comment cites tuning data that is not in the source. Requires per-arch
  benchmarking.
* **U8**. Whether the MMVF `buf_iw` shared-memory staging (used when
  `block_size > warp_size`) is faster than a pure shuffle-tree reduction
  across the full block (would require cooperative-groups tile shuffles).
  Requires benchmarking.

---

## 19. References

| Reference | File                                                | Function / Symbol                                | Lines         |
| --------- | --------------------------------------------------- | ------------------------------------------------ | ------------- |
| R01       | `ggml/src/ggml-cuda/mmvq.cu`                        | `get_vec_dot_q_cuda`                             | 10-36         |
| R02       | `ggml/src/ggml-cuda/mmvq.cu`                        | `mmvq_parameter_table_id` enum + `get_device_table_id` | 64-106   |
| R03       | `ggml/src/ggml-cuda/mmvq.cu`                        | `get_mmvq_mmid_max_batch_*` (6 arch tables)      | 112-244       |
| R04       | `ggml/src/ggml-cuda/mmvq.cu`                        | `ggml_cuda_should_use_mmvq`                      | 280-328       |
| R05       | `ggml/src/ggml-cuda/mmvq.cu`                        | `calc_nwarps`, `calc_rows_per_block`             | 352-476       |
| R06       | `ggml/src/ggml-cuda/mmvq.cu`                        | `mul_mat_vec_q` (kernel template)                | 478-699       |
| R07       | `ggml/src/ggml-cuda/mmvq.cu`                        | `mul_mat_vec_q_moe` (MoE kernel) + launch        | 705-835       |
| R08       | `ggml/src/ggml-cuda/mmvq.cu`                        | `mul_mat_vec_q_switch_*` (dispatch chain)        | 783-1143      |
| R09       | `ggml/src/ggml-cuda/mmvq.cu`                        | `should_use_small_k` lambda                      | 861-901       |
| R10       | `ggml/src/ggml-cuda/mmvq.cu`                        | `ggml_cuda_mul_mat_vec_q` (entry)                | 1145-1258     |
| R11       | `ggml/src/ggml-cuda/mmvf.cu`                        | `mul_mat_vec_f` (kernel template)                | 7-379         |
| R12       | `ggml/src/ggml-cuda/mmvf.cu`                        | `launch_mul_mat_vec_f_cuda` (auto-tune)          | 412-505       |
| R13       | `ggml/src/ggml-cuda/mmvf.cu`                        | `mul_mat_vec_f_cuda` (precision selection)       | 604-627       |
| R14       | `ggml/src/ggml-cuda/mmvf.cu`                        | `ggml_cuda_mul_mat_vec_f` (entry)                | 629-723       |
| R15       | `ggml/src/ggml-cuda/mmvf.cu`                        | `ggml_cuda_should_use_mmvf`                      | 786-869       |
| R16       | `ggml/src/ggml-cuda/mmf.cu`                         | `ggml_cuda_mul_mat_f`, `ggml_cuda_should_use_mmf`| 13-191        |
| R17       | `ggml/src/ggml-cuda/mmf.cuh`                        | `mul_mat_f` (Tensor-Core kernel)                 | 48-294        |
| R18       | `ggml/src/ggml-cuda/mmf.cuh`                        | `mul_mat_f_ids`, `MMF_ROWS_PER_BLOCK`            | 297-550, 9-10 |
| R19       | `ggml/src/ggml-cuda/mmid.cu`                        | `mm_ids_helper` + launch helper                  | 26-169        |
| R20       | `ggml/src/ggml-cuda/mmvq.cuh`, `mmvf.cuh`           | `MMVQ_MAX_BATCH_SIZE = 8`, `MMVF_MAX_BATCH_SIZE = 8` | 3        |
| R21       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q*_q8_1` family, `VDR_*_MMVQ`           | 109-362, 802-985 |
| R22       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `dequantize_q4_0` etc. (NOT used by MMVQ)        | 26-433        |
| R23       | `ggml/src/ggml-cuda/common.cuh`                     | `warp_reduce_sum`, `ggml_cuda_dp4a`, `ggml_cuda_mad`, `ggml_cuda_kernel_launch` | 443-462, 703, 743-770, 1641-1660 |
| R24       | `ggml/src/ggml-cuda/quantize.cu`                    | `quantize_row_q8_1_cuda`                         | 558-573       |
| R25       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_mul_mat` (matmul router)              | 1812-1852     |
| R26       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_mul_mat_id`                           | 1854-1992     |
| R27       | `ggml/src/ggml-cuda/ggml-cuda.cu`                   | `ggml_cuda_should_fuse_mul_mat_vec_{f,q}`        | 1756-1810     |
