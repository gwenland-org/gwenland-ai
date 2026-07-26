# ARTX16 — Metal Threadgroup Memory, Simdgroup Matrix, and Tiled Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX16)
**Target GwenLand module:** `glmetal` (kernel-side), `GATE` (tile-size policy)

---

## 1. Executive Summary

`ggml-metal.metal` is the 11 218-line kernel source of the Metal backend.
Where ARTX15 audited the *host-side* machinery (device discovery, command
buffers, pipeline cache, op-fusion engine, residency sets), this document
audits the *kernel-side*: what happens inside each `kernel void` after
the encoder has dispatched it. The dominant topics are:

1. **`[[threadgroup]]` memory allocation patterns.** The kernel source
   uses four distinct patterns: (a) no shmem at all (rope, bin_fuse,
   cpy, get_rows, opt_step, mul_mv_ext, mul_mv_qK_f32); (b) a tiny
   per-simdgroup reduction scratch (32·sizeof(float) bytes, used by
   softmax, norm, rms_norm, l2_norm, group_norm, sum_rows, mul_mv for
   Q8_0/F32/F16/BF16); (c) a 6 KiB or 8 KiB tile pair (`sa` + `sb`) for
   `mul_mm`; (d) a multi-region KV/Q/O/scratch layout (≤ 32 KiB) for
   flash attention. No header struct is used; the buffer is a flat
   `threadgroup char *` and every kernel reinterprets it locally.

2. **`simdgroup_multiply_accumulate`.** The legacy `kernel_mul_mm`
   instantiates 4 `simdgroup_*8x8` A-tiles + 2 B-tiles per inner
   iteration, accumulating into 8 C-tiles. Per-iteration
   `simdgroup_barrier(mem_flags::mem_none)` calls are compiler hints,
   not actual syncs. The same API drives the QK^T and PV matmuls in
   flash attention with 1 or 2 tile-pairs per iteration.

3. **Tile sizes.** `mul_mm` uses `NR0=64, NR1=32, NK=32` (legacy) or
   `NRA=64, NRB=128` (Metal4 tensor path). `mul_mv` uses per-dtype
   `N_R0_*`/`N_SG_*` constants from `ggml-metal-impl.h:24-88`, all
   compile-time. Flash attention uses `nqptg=8, ncpsg=64` (matrix
   kernel) or `nqptg=1, ncpsg=32` (vec kernel), with `nsg ∈ {4, 8}`
   chosen by `ne00 >= 512 ? 8 : 4`. No tile size is autotuned.

4. **Quantized weight unpacking.** `dequantize_*` are `device`-space
   template functions that take a `block_q*` and produce a 4×4 tile in
   a `thread` register. They are called per-thread (not per-simdgroup)
   inside `mul_mm` to populate `sa`, and per-thread inside `mul_mv`
   inside `block_q_n_dot_y`. K-quants (Q4_K etc.) bypass shmem entirely
   — each simdgroup unpacks into thread-private `float4 acc1/acc2`.

5. **Address spaces.** Every kernel takes a `constant
   ggml_metal_kargs_<op> & args` first parameter (read-only, lives in
   the device's constant cache). Tensor data is `device const char *`
   / `device char *`. Threadgroup is `threadgroup char * shmem`. The
   `thread` address space is used implicitly for register variables
   (`float sumf[NR0]`, `device const block_q4_0 * ax[NR0]`, etc.).

For GwenLand, the architectural decisions worth **ADOPT**ing are: the
`helper_mv_reduce_and_write` two-level reduction template, the
`simdgroup_multiply_accumulate` outer-product loop in `mul_mm`, the
flash-attention shmem layout (Q/O/scratch/K-scratch/V-scratch/mask in
one flat buffer), and the per-dtype `N_R0_*`/`N_SG_*` constant table.
The decisions worth **REJECT**ing are the hardcoded `4096`-byte split
inside `kernel_mul_mm` (works only because `S0` is always `half`), the
dead `nr0 ∈ {1,3,4}` cases in `kernel_mul_mv_t_t_disp`, and the
commented-out autotuner hooks in flash attention that leave `nsg` as a
hardcoded `4-or-8-by-ne00` heuristic.

---

## 2. Purpose

Provide the kernel-side implementation of every Metal compute kernel
that the backend dispatches. Specifically:

* implement the matmul family (`mul_mv`, `mul_mv_ext`, `mul_mm`,
  `mul_mm_id`) for every supported (src0, src1) dtype pair,
* implement the attention family (`flash_attn_ext`,
  `flash_attn_ext_vec`, `flash_attn_ext_pad`, `flash_attn_ext_blk`),
* implement elementwise and reduction ops (unary, bin, rope, norm,
  rms_norm, l2_norm, group_norm, soft_max, sum_rows),
* implement conversion ops (cpy, get_rows, set_rows, concat),
* implement optimizer ops (opt_step_adamw, opt_step_sgd),
* expose all kernels via `[[host_name("kernel_*")]]` linkage so that
  the host-side pipeline cache can fetch them by string name.

It is **not** responsible for: kernel selection (handled in
`ggml-metal-ops.cpp`, ARTX15), pipeline compilation (handled in
`ggml-metal-device.m`, ARTX15), or graph scheduling (handled in
`ggml-metal-context.m`, ARTX15). This file is pure kernel code; the
only host-visible symbols are the `kernel_*` entry points.

---

## 3. Source Files

| File                                       | Lines  | Role                                                                |
| ----------------------------------------- | ------ | ------------------------------------------------------------------ |
| `ggml/src/ggml-metal/ggml-metal.metal`    | 11 218 | All Metal kernels; ~80 `kernel void` functions, plus `device`/template helpers |
| `ggml/src/ggml-metal/ggml-metal-impl.h`   | 1 222  | Per-dtype tile constants (`N_R0_*`, `N_SG_*`), function-constant offsets (`FC_*`), per-op kernel-args structs |
| `ggml/src/ggml-metal/ggml-metal-common.h` | 52     | Public C interface to `ggml_mem_ranges` overlap tracker (host-side; referenced here only for completeness) |

> **Note on the constants header.** `ggml-metal-impl.h` is included by
> both the host-side `.cpp/.m` files and the `.metal` source. Every
> per-dtype `N_R0_*` and `N_SG_*` is a `#define`, so the same constant
> flows from the host's pipeline-selection code into the kernel's
> template parameter. This is the Metal analog of the CPU type-traits
> table (ARTX01-F03), but expressed as a flat sequence of preprocessor
> macros instead of an indexed array.

---

## 4. Architecture Overview

```
       ┌────────────────────────────────────────────────────────────┐
       │  ggml-metal.metal                                         │
       │                                                            │
       │  ┌─ dequantize_*  (device template functions)             │
       │  │   dequantize_f32, _f16, _bf16                          │
       │  │   dequantize_q1_0, _q2_0, _q4_0, _q4_1, _q5_0, _q5_1   │
       │  │   dequantize_q8_0, _mxfp4                              │
       │  │   dequantize_q2_K .. _q6_K                             │
       │  │   dequantize_iq1_s, _iq1_m, _iq2_*, _iq3_*, _iq4_*     │
       │  │   (also _t4 variants producing float4 instead of 4x4)  │
       │  │                                                        │
       │  ├─ quantize_*    (device functions, one block per call)  │
       │  │   quantize_q1_0 .. quantize_q8_0, quantize_iq4_nl      │
       │  │                                                        │
       │  ├─ block_q_n_dot_y  (inline per-quant dot-product helper)│
       │  │   overloads for block_q1_0, block_q2_0                 │
       │  │                                                        │
       │  ├─ helper_mv_reduce_and_write<NR0>  (two-level reduction)│
       │  │                                                        │
       │  ├─ kernel_mul_mv_*  (GEMV family, per-dtype)             │
       │  │   kernel_mul_mv_q1_0_f32, _q2_0, _q4_0, _q4_1, ...     │
       │  │   kernel_mul_mv_q8_0_f32 (uses shmem reduction)        │
       │  │   kernel_mul_mv_q2_K_f32 .. _q6_K (no shmem)           │
       │  │   kernel_mul_mv_iq2_xxs_f32 .. _iq4_xs (no shmem)      │
       │  │   kernel_mul_mv_t_t (F32/F16/BF16; uses shmem reduction)│
       │  │   kernel_mul_mv_t_t_4 (vectorized float4 variant)      │
       │  │   kernel_mul_mv_t_t_short (single-simdgroup, no shmem) │
       │  │   kernel_mul_mv_ext_q4_f32_disp (register-only)        │
       │  │   kernel_mul_mv_ext_q4x4_f32_disp (register-only)      │
       │  │                                                        │
       │  ├─ kernel_mul_mm  (template; 2 paths: legacy / tensor)   │
       │  │   legacy: simdgroup_multiply_accumulate, sa+sb in shmem│
       │  │   tensor: mpp::tensor_ops::matmul2d, sa only in shmem  │
       │  │                                                        │
       │  ├─ kernel_mul_mm_id_map0 + kernel_mul_mm_id              │
       │  │   (MoE expert-routing pre-pass + per-expert matmul)    │
       │  │                                                        │
       │  ├─ kernel_flash_attn_ext (template; Q+O+ss+sk+sv+sm2)    │
       │  │   kernel_flash_attn_ext_vec (single-query variant)     │
       │  │   kernel_flash_attn_ext_pad, _blk (mask pre-processing)│
       │  │   kernel_flash_attn_ext_vec_reduce (cross-tg reduce)   │
       │  │                                                        │
       │  ├─ kernel_norm_fuse_impl, kernel_rms_norm_fuse_impl      │
       │  │   kernel_l2_norm_impl, kernel_group_norm_f32           │
       │  │   (all use 32-float shmem reduction when ntg > 32)     │
       │  │                                                        │
       │  ├─ kernel_soft_max, kernel_soft_max_4                    │
       │  │   (two-pass max+sum, 32-float shmem if ntg > 32)       │
       │  │                                                        │
       │  ├─ kernel_rope_norm, _neox, _multi, _vision              │
       │  │   (no shmem, per-thread cos/sin)                       │
       │  │                                                        │
       │  ├─ kernel_bin_fuse_impl (ADD/MUL/SUB/DIV, N-way fused)   │
       │  │   (no shmem, FC_F ∈ {1..8} via function constant)      │
       │  │                                                        │
       │  ├─ kernel_cpy_t_t, kernel_cpy_f32_q, kernel_cpy_q_f32    │
       │  ├─ kernel_get_rows_q, kernel_get_rows_f                  │
       │  ├─ kernel_set_rows_q32, kernel_set_rows_f                │
       │  ├─ kernel_concat, kernel_repeat, kernel_add_id           │
       │  ├─ kernel_diag_f32, kernel_memset                        │
       │  ├─ kernel_argsort_f32_i32, kernel_argsort_merge_f32_i32  │
       │  ├─ kernel_argmax_f32, kernel_count_equal                 │
       │  ├─ kernel_pool_2d_max_f32, _avg, _1d_max, _1d_avg        │
       │  ├─ kernel_ssm_conv_f32_f32, kernel_ssm_scan_f32          │
       │  ├─ kernel_rwkv_wkv6_f32, kernel_rwkv_wkv7_f32            │
       │  ├─ kernel_gated_delta_net_impl                           │
       │  ├─ kernel_solve_tri_f32                                  │
       │  ├─ kernel_snake                                          │
       │  ├─ kernel_opt_step_adamw_f32, kernel_opt_step_sgd_f32    │
       │  └─ kernel_im2col, kernel_conv_2d, kernel_conv_2d_dw_tiled│
       │      kernel_conv_transpose_1d, _2d, kernel_conv_3d        │
       │      kernel_col2im_1d, kernel_upscale_*, kernel_pad_*     │
       │      kernel_roll_f32, kernel_arange_f32                   │
       │      kernel_timestep_embedding_f32, kernel_tri            │
       │      kernel_cumsum_blk, kernel_cumsum_add                 │
       └────────────────────────────────────────────────────────────┘
```

Key design points:

* **Template-driven code generation.** 90% of `kernel_mul_mv_*` and
  `kernel_mul_mm_*` is shared via C++ templates; per-dtype
  specializations are emitted via
  `template [[host_name("kernel_mul_mv_q4_0_f32")]] kernel ...`
  instantiation lines. One template, ~25 dtype instantiations.
* **Function constants, not function arguments.** Per-shape values
  (`r2`, `r3`, `nsg`, `has_mask`, `bc_inp`, `bc_out`, ...) are baked
  into the compiled pipeline via `MTLFunctionConstantValues`
  (ARTX15-F08). The kernel reads them as `constant short FC_mul_mv_nsg
  [[function_constant(FC_MUL_MV + 0)]]`. The compiler eliminates dead
  branches per specialization.
* **`constant` address space for args.** Every kernel's first parameter
  is `constant ggml_metal_kargs_<op> & args`. This struct is the
  host-prepared arguments block, copied via
  `ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0)`. It lives
  in the device's constant cache, broadcast to every thread.
* **Two parallel matmul implementations.** `kernel_mul_mm` exists in
  two forms selected by `#ifdef GGML_METAL_HAS_TENSOR`: (a) legacy
  `simdgroup_multiply_accumulate` path with `sa`+`sb` in threadgroup
  memory; (b) `mpp::tensor_ops::matmul2d` path with `sa` only in
  threadgroup memory (B is read directly from device). The tensor path
  is gated on `MTLGPUFamilyMetal4_GGML` and disabled by chip name for
  pre-M5 hardware (ARTX15-F10).

---

## 5. Execution Flow

### 5.1 Top-level kernel entry

Every kernel is a free function `kernel void kernel_<name>(...)`. The
host-side `ggml_metal_op_encode_impl` (ARTX15 §5.3) selects a pipeline
by string name, sets the `constant` slot 0 to the `ggml_metal_kargs_*`
struct, sets buffer slots 1..N to the src/dst MTLBuffers, sets the
threadgroup memory length via
`ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0)`, and
dispatches `(threadgroups_per_grid, threads_per_threadgroup)`.

### 5.2 `kernel_mul_mv_q4_0_f32` (representative GEMV)

`ggml-metal.metal:3778-3788`:

1. The kernel is a one-line forwarder to
   `mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, ...>` (line 3531).
2. Each thread computes `NR0 = N_R0_Q4_0 = 4` dot products across a
   shared K dimension. Each simdgroup processes a disjoint set of `NR0`
   rows (so `NSG = N_SG_Q4_0 = 2` simdgroups per threadgroup handle
   `2*4 = 8` rows total).
3. The activation vector `y` is read directly from device memory
   (`device const float * y`). No shmem staging of `y`. Each thread
   caches 16 elements of `y` in `thread float yl[16]` registers.
4. The weight rows are addressed via `device const block_q4_0 * ax[NR0]`
   — an array of NR0 pointers, one per row, kept in registers.
5. The inner loop iterates `ib` over `nb = ne00/QK4_0` blocks. Each
   iteration calls `block_q_n_dot_y(ax[row] + ib, sumy, yl, il)`, which
   is an inline `static` function that does the per-bit unpacking.
6. After the K loop, `helper_mv_reduce_and_write<NR0>(dst_f32, sumf, ...)`
   performs the cross-simdgroup reduction (Section 5.5) and writes the
   result.

### 5.3 `kernel_mul_mm` (representative GEMM, legacy path)

`ggml-metal.metal:10095-10303`:

1. The kernel is templated as
   `<S0, S0_4x4, S0_8x8, S1, S1_2x4, S1_8x8, block_q, nl, dequantize_func, T0, T0_4x4, T1, T1_2x4>`.
   `S0` is the threadgroup-storage type for A (always `half` or
   `bfloat`); `S1` is the threadgroup-storage type for B (always `half`
   or `bfloat`); `block_q` is the on-device quantized type for A.
2. Threadgroup memory is split: `sa = (threadgroup S0 *)(shmem)` and
   `sb = (threadgroup S1 *)(shmem + 4096)` (line 10106). The split
   point is a hardcoded `4096` byte offset.
3. Threadgroup size is `32 * 4 * 1 = 128` threads (4 simdgroups; see
   `N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y = 4`).
4. Output tile is `NR0 = 64` rows × `NR1 = 32` cols = 2048 elements per
   threadgroup. Each simdgroup writes a `32 × 8` sub-tile.
5. The K loop iterates `loop_k` from 0 to `args.ne00` in steps of `NK = 32`:
   a. **Dequantize A**: each thread loads 16 elements of A from device,
      dequantizes into a `S0_4x4` register, then scatters into `sa` at
      the position determined by `(sx, sy, lx, ly)` mapping.
   b. **Load B**: each thread loads a `S1_2x4` (8 elements) from device
      `y` and stores to `sb`.
   c. `threadgroup_barrier(mem_flags::mem_threadgroup)` — full
      threadgroup sync.
   d. **Outer-product accumulation**: `NK/8 = 4` iterations of:
      * `simdgroup_barrier(mem_flags::mem_none)` — compiler hint, no
        actual sync.
      * `simdgroup_load(ma[i], lsma + 64*i, 8, 0, false)` for `i ∈ 0..3`
        — 4 A-tiles.
      * `simdgroup_barrier(mem_flags::mem_none)`.
      * `simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false)` for `i ∈ 0..1`
        — 2 B-tiles.
      * `simdgroup_barrier(mem_flags::mem_none)`.
      * `simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i])`
        for `i ∈ 0..7` — 8 C-tiles updated.
      * Advance `lsma += 8*64; lsmb += 4*64`.
6. **Store**: if no bounds check needed, write `mc[i]` directly to
   device via `simdgroup_store`. Otherwise, stage through `shmem` (now
   reused as `temp_str`), barrier, then scalar copy with bounds check.

### 5.4 `kernel_mul_mv_ext_q4_f32_disp` (representative ext GEMV)

`ggml-metal.metal:4127-4148`:

1. No threadgroup memory at all. The kernel signature lacks the
   `threadgroup char * shmem` parameter.
2. The threadgroup is `32 * NSG * 1 = 64` threads (NSG=2 for the ext
   path).
3. The thread layout is a 2D grid inside the simdgroup: `nxpsg` threads
   horizontally (one per chunk of the K dimension) and `nypsg = 32/nxpsg`
   threads vertically (one per src0 row). With `nxpsg ∈ {16, 8, 4}`,
   `nypsg ∈ {2, 4, 8}`.
4. Each thread dequantizes `chpt = 4` chunks of 4 elements each (16
   elements total) into `thread float4 lx[chpt]` registers, then
   computes `sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg])` for each
   `ir1 ∈ 0..r1ptg-1`.
5. The intra-row reduction uses `simd_shuffle_down` (lines 3988-4001) —
   no shmem, no barriers. Each row's reduction is independent because
   each row is handled by a disjoint set of `nxpsg` threads inside the
   simdgroup.

### 5.5 `helper_mv_reduce_and_write<NR0>` (cross-simdgroup reduction)

`ggml-metal.metal:3483-3523`:

1. `simd_sum(sumf[row])` — intra-simdgroup horizontal sum.
2. `threadgroup_barrier(mem_flags::mem_threadgroup)` — wait for all
   simdgroups to finish their `simd_sum`.
3. `if (tiisg == 0) shmem_f32[row][sgitg] = sumf[row]` — lane 0 of each
   simdgroup writes its partial to a unique slot.
4. `threadgroup_barrier(mem_flags::mem_threadgroup)` — wait for all
   writes.
5. `tot = simd_sum(shmem_f32[row][tiisg])` — every thread now reads a
   different slot and does a second `simd_sum`.
6. Lane 0 of simdgroup 0 writes the final result to `dst_f32`.

The shmem footprint is `NR0 * 32 * sizeof(float)` bytes (one float per
thread per row). For `NR0 = 8` (Q1_0/Q2_0): 1024 bytes. For `NR0 = 2`
(Q8_0): 256 bytes. This is a fixed allocation regardless of `NSG`.

### 5.6 `kernel_soft_max` (representative reduction kernel)

`ggml-metal.metal:1950-2053`:

1. Two-pass: (a) parallel max with simd_max + optional shmem reduction;
   (b) parallel sum-exp with simd_sum + optional shmem reduction.
2. The shmem reduction is only triggered when `tptg.x > N_SIMDWIDTH`
   (i.e., when there are more than 32 threads = more than one simdgroup
   in the threadgroup). For single-simdgroup dispatches, shmem is
   untouched.
3. ALiBi bias is computed inline via `pow(base, exp)` per thread — no
   precomputed table.

---

## 6. Data Layout

### 6.1 Tensor descriptor in kernels

Kernels receive tensor metadata via the `constant
ggml_metal_kargs_<op> &` struct, not via `ggml_tensor *`. The struct
mirrors a subset of `ggml_tensor` fields: `ne00..ne03`, `nb00..nb03`
for src0, `ne10..ne13`, `nb10..nb13` for src1, `ne0..ne3`, `nb0..nb3`
for dst. Element counts are `int32_t` (to reduce register pressure);
strides are `uint64_t`. Broadcast factors `r2 = ne12/ne02` and
`r3 = ne13/ne03` are pre-computed on the host and passed as `int16_t`
function constants (`FC_mul_mv_r2`, `FC_mul_mv_r3`).

### 6.2 Per-thread register layout (`mul_mv` GEMV)

Each thread holds:

* `device const block_q * ax[NR0]` — array of NR0 device pointers to
  weight rows (8 pointers max for Q1_0/Q2_0; 2 for Q8_0).
* `float sumf[NR0]` — NR0 accumulators, one per row.
* `float yl[16]` (or `float yl[32]` for Q2_K) — activation cache.

For Q8_0, the inner loop accesses `ax[row][ib].qs[il*NQ + i]` directly
from device memory — no unpacking needed (Q8_0 is already int8). For
Q4_0, the per-block unpacking is inlined into `block_q_n_dot_y`.

### 6.3 Threadgroup tile layout (`mul_mm` legacy)

The threadgroup memory is a flat `char *` of size `smem = 4096 + 2048 =
6144` bytes (or `8192` if `bc_out`). The kernel reinterprets it as:

| Region | Type   | Size (bytes)                  | Purpose                              |
| ------ | ------ | ----------------------------- | ------------------------------------ |
| `sa`   | `S0`   | `4 * 64 * 8 * sizeof(S0) = 4096` (half) | A tile, 4 sub-tiles of 8×8 each |
| `sb`   | `S1`   | `2 * 64 * 8 * sizeof(S1) = 2048` (half) | B tile, 2 sub-tiles of 8×8 each |
| `temp_str` (only if `bc_out`) | `float` | `64 * 32 * 4 = 8192` | Staging for bounds-checked store |

The `sa` region is reused as `temp_str` after the matmul completes
(line 10275) — the input tiles are no longer needed once the result is
in `mc[]`. This is a clever memory-saving trick but creates a hidden
dependency between the matmul stage and the store stage.

### 6.4 Threadgroup tile layout (`mul_mm` tensor path)

The `GGML_METAL_HAS_TENSOR` path allocates only `smem_a = NRA *
N_MM_NK_TOTAL * sizeof(ggml_fp16_t) = 64 * 32 * 2 = 4096` bytes for
`sa`. B is read directly from device memory via the `tensor` API
(`auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), ...)`). The
output is staged through `cT` (a cooperative tensor) and stored
directly to device via `cT.store(tD.slice(ra, rb))`, which handles
bounds checking internally.

### 6.5 Threadgroup tile layout (`flash_attn_ext`)

`ggml-metal.metal:6402-6416` defines a 6-region layout in a single
`threadgroup half * shmem_f16` buffer:

| Region | Symbol | Offset (half-elements)             | Size                       |
| ------ | ------ | ---------------------------------- | -------------------------- |
| Q      | `sq`/`sq4` | `0`                            | `Q * DK`                   |
| O      | `so`/`so4` | `Q*DK`                         | `Q * PV` (PV = pad(DV,64)) |
| S scratch | `ss`/`ss2` | `Q*T` (T = DK + 2*PV)       | `Q * 2*SH` (SH = 2*C)      |
| K scratch | `sk`/`sk4x4` | `sgitg*(4*16*KV) + Q*T + Q*TS` | `4*16*KV` per simdgroup |
| V scratch | `sv`/`sv4x4` | same as sk (overlapped)      | `4*16*KV` per simdgroup    |
| Mask   | `sm2`  | `Q*T + 2*C`                        | `C` half2 elements         |

`sk` and `sv` share the same offset — they are loaded at different
times in the KV loop, so the overlap is safe. The host sizes shmem via
`FATTN_SMEM(nsg)` (`ggml-metal-ops.cpp:2817`):
`nqptg*(ne00 + 2*pad(ne20,64) + 2*(2*ncpsg)) + is_q*(16*32*nsg)`
half-elements, padded to 16 bytes. Typical sizes for DK=DV=128,
nqptg=8, ncpsg=64, nsg=4: 8*(128 + 256 + 256) + 16*32*4 = 5120 + 2048
= 7168 bytes (padded to 7168).

### 6.6 Quantized weight block layout

Source: `ggml-common.h` block definitions (referenced via
`#include "ggml-common.h"` at `ggml-metal.metal:6`). Each block is a
fixed-size struct: `block_q4_0` = 18 bytes (1 half `d` + 16 bytes
`qs`); `block_q8_0` = 34 bytes; `block_q4_K` = 144 bytes; etc. Blocks
are contiguous along the row. The `nl` template parameter is the number
of 4×4 sub-tiles per block: `nl = QK4_0/16 = 2` for Q4_0; `nl = QK_K/16
= 16` for K-quants.

---

## 7. Memory Layout

### 7.1 Address spaces

| Space      | Use                                                       | Example                                    |
| ---------- | --------------------------------------------------------- | ------------------------------------------ |
| `device`   | Tensor data (src0, src1, dst)                             | `device const char * src0`                 |
| `constant` | Per-op args struct (read-only, broadcast)                 | `constant ggml_metal_kargs_mul_mv & args`  |
| `threadgroup` | Per-threadgroup scratch (matmul tiles, reductions)    | `threadgroup char * shmem [[threadgroup(0)]]` |
| `thread`   | Per-thread registers (accumulators, pointers, yl[])      | `float sumf[NR0] = {0.f}`                  |

The `constant` space is the device's constant cache — small (typically
64 KiB), broadcast to every thread at fixed latency. The `device` space
is the global GPU memory; reads go through the L2/L1 cache hierarchy.
The `threadgroup` space is the on-chip shared memory of the simdgroup
cluster (typically 32 KiB per threadgroup on Apple Silicon, up to 64
KiB on Metal3+ devices).

### 7.2 Threadgroup memory budget

`ggml-metal-device.m:851` records `dev->props.max_theadgroup_memory_size
= dev->mtl_device.maxThreadgroupMemoryLength`. The kernel-side smem
allocations are bounded by:

| Kernel family         | smem size (bytes)            | Source                              |
| --------------------- | ---------------------------- | ----------------------------------- |
| `mul_mv` (F32/F16/BF16) | `32 * 4 * nr0 = 256` (nr0=2) | `ggml-metal-device.cpp:799`        |
| `mul_mv` (Q8_0)       | `32 * 4 * 2 = 256`           | `ggml-metal-device.cpp:837`        |
| `mul_mv` (MXFP4)      | `32 * 4 = 128`               | `ggml-metal-device.cpp:843`        |
| `mul_mv` (Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, K-quants, IQ) | 0 | (nullptr passed)         |
| `mul_mm` (legacy, no bc_out) | `4096 + 2048 = 6144`   | `ggml-metal-device.cpp:758`        |
| `mul_mm` (legacy, bc_out)    | `8192`                 | `ggml-metal-device.cpp:758`        |
| `mul_mm` (tensor)            | `4096` (smem_a only)   | `ggml-metal-device.cpp:752-753`    |
| `flash_attn_ext`      | `FATTN_SMEM(nsg)`, typically 4–16 KiB | `ggml-metal-ops.cpp:2817` |
| `soft_max`, `norm`, `rms_norm`, `l2_norm`, `group_norm`, `sum_rows` | `32 * sizeof(float) = 128` | `ggml-metal-device.cpp:386` |
| `conv_2d`             | `KW * KH * sizeof(float)`    | `ggml-metal-ops.cpp:4135`           |
| `argsort_merge`       | `nth * sizeof(int32_t)`      | `ggml-metal-ops.cpp:4489, 4595`     |

All allocations are well within the 32 KiB Metal limit. The largest is
flash attention at ~16 KiB for nsg=8.

### 7.3 Vectorized loads

The kernels use Metal vector types for coalesced device loads:

* `float4` for F32 paths (`kernel_mul_mv_t_t_4_impl`, line 4424).
* `half4` for F16 paths (via `T04 = half4` template parameter).
* `bfloat4` for BF16 paths (gated by `GGML_METAL_HAS_BF16`).
* `float4x4` / `half4x4` / `bfloat4x4` for 4×4 tile loads (used in
  `dequantize_*` and `kernel_mul_mv_ext_q4x4_f32_impl`).
* `half2x4` / `bfloat2x4` for `S1_2x4` (2-row × 4-col B tile in
  `mul_mm`).
* `int4` / `uint4` not used; quantized data is loaded as `uint8_t` and
  unpacked bit-by-bit.

### 7.4 `constexpr constant` tables

Two small constant tables live at file scope
(`ggml-metal.metal:47-53`):

* `kvalues_iq4nl_f[16]` — the 16 dequantization levels for IQ4_NL.
* `kvalues_mxfp4_f[16]` — the 16 dequantization levels for MXFP4.

These are `constexpr constant static`, so they live in the device's
constant memory and are shared across all threads. They are *not*
threadgroup memory.

---

## 8. Parallelism Strategy

### 8.1 Two-level parallelism inside a kernel

Every tiled kernel uses two levels of parallelism:

1. **Across simdgroups** (`sgitg = simdgroup_index_in_threadgroup`).
   Each simdgroup (32 threads) handles a disjoint sub-tile of the
   output. For `mul_mm`: 4 simdgroups × `32 × 8` sub-tiles = `64 × 32`
   output tile. For `flash_attn_ext`: `nsg ∈ {4, 8}` simdgroups
   cooperate on a single (Q, C) tile via cross-simdgroup shmem
   reduction.
2. **Across threads within a simdgroup** (`tiisg =
   thread_index_in_simdgroup`). The 32 threads cooperate via implicit
   warp-wide ops (`simd_sum`, `simd_min`, `simd_shuffle_down`) — no
   explicit sync needed because all 32 threads execute in lockstep.

### 8.2 Per-kernel threadgroup size

| Kernel            | Threads per TG           | Simdgroups per TG    |
| ----------------- | ------------------------ | -------------------- |
| `mul_mv` (Q1_0/Q2_0/Q4_0/Q4_1/Q5_0/Q5_1) | 32 × N_SG_* × 1 | 2 (N_SG=2)         |
| `mul_mv` (Q8_0)   | 32 × 4 × 1              | 4                    |
| `mul_mv` (Q4_K, Q5_K, IQ*) | 32 × 2 × 1       | 2                    |
| `mul_mv` (F32/F16/BF16)  | 32 × nsg × 1     | nsg = min(4, (ne00+127)/128) |
| `mul_mv_ext`      | 32 × 2 × 1              | 2                    |
| `mul_mm` (legacy) | 32 × 4 × 1              | 4 (2×2 grid)         |
| `mul_mm` (tensor) | 32 × 4 × 1              | 4                    |
| `flash_attn_ext`  | 32 × nsg × 1            | 4 or 8 (nsg)         |
| `soft_max`        | variable (32..512)      | 1..16                |
| `rope`            | variable (typically 256)| 8                    |
| `bin_fuse`        | variable (typically 256)| 8                    |

All threadgroup sizes are 1-dimensional in the Z dimension (the grid
has multiple Z slices for batch). The X dimension carries the
simdgroup count; the Y dimension is always 1.

### 8.3 Grid (threadgroups-per-grid) sizing

| Kernel            | Grid X                                          | Grid Y                | Grid Z              |
| ----------------- | ----------------------------------------------- | --------------------- | ------------------- |
| `mul_mv` (Q4_0..) | `ceil(ne01 / (N_R0_Q4_0 * N_SG_Q4_0))`         | `ne11`                | `ne12 * ne13`       |
| `mul_mv` (Q8_0)   | `ceil(ne01 / (N_R0_Q8_0 * N_SG_Q8_0))`         | `ne11`                | `ne12 * ne13`       |
| `mul_mm`          | `ceil(ne1 / NR1)`                               | `ceil(ne0 / NR0)`     | `ne12 * ne13`       |
| `flash_attn_ext`  | `ceil(ne01 / nqptg)`                            | `ne02`                | `ne03`              |
| `soft_max`        | `ne01`                                          | `ne02`                | `ne03`              |
| `rope`            | `ne01`                                          | `ne02`                | `ne03`              |
| `bin_fuse`        | `ceil(ne01*ne00 / ntg.x)` (or split per row)    | —                     | —                   |

The grid X dimension carries the row split; Y the column (or batch);
Z the batch (or expert, for `mul_mm_id`).

### 8.4 Cross-simdgroup synchronization

Three patterns:

1. **`threadgroup_barrier(mem_flags::mem_threadgroup)`** — full
   threadgroup sync, used after tile loads and before tile use.
2. **`simdgroup_barrier(mem_flags::mem_none)`** — compiler hint, no
   actual sync. Used in `mul_mm` between `simdgroup_load` calls to hint
   that the loads are independent and can be reordered.
3. **`simd_sum` / `simd_min` / `simd_max` / `simd_shuffle_down`** —
   implicit warp ops, no sync needed (32 threads execute in lockstep).

The `mem_flags::mem_none` argument to `simdgroup_barrier` is a Metal
peculiarity: it tells the compiler "I want a simdgroup-level ordering
point, but I don't need to flush any memory". In practice this is a
no-op; the call exists as a documentation marker that the kernel
author wanted the compiler to schedule the surrounding loads in a
particular order.

---

## 9. SIMD / GPU Strategy

### 9.1 `simdgroup_matrix<T, 8>` and the `simdgroup_*` API

The Metal Shading Language exposes an 8×8 matrix type that lives in
the simdgroup's register file. Each thread of the 32-thread simdgroup
holds 2 elements of the matrix (8×8 = 64 elements / 32 threads = 2
elements per thread). The API is:

* `simdgroup_matrix<T, 8>` — the matrix type.
* `make_filled_simdgroup_matrix<T, 8>(value)` — zero-initialize.
* `simdgroup_load(m, ptr, src_stride, transpose)` — load an 8×8 tile
  from `threadgroup` or `device` memory into a `simdgroup_matrix`.
* `simdgroup_store(m, ptr, dst_stride, transpose)` — store an 8×8 tile
  back to memory.
* `simdgroup_multiply(c, a, b)` — `c = a * b` (8×8 × 8×8 → 8×8).
* `simdgroup_multiply_accumulate(c, a, b, c)` — `c += a * b`.
* `simdgroup_barrier(mem_flags)` — ordering hint.

The kernels use `simdgroup_half8x8`, `simdgroup_float8x8`, and
`simdgroup_bfloat8x8` (the latter gated by `GGML_METAL_HAS_BF16`).

### 9.2 Outer-product accumulation in `mul_mm`

The `mul_mm` legacy path uses 4 A-tiles + 2 B-tiles + 8 C-tiles per
inner iteration:

```
ma[4]   — 4 A-tiles, each 8×8, covering 32 K-elements × 8 rows
mb[2]   — 2 B-tiles, each 8×8, covering 32 K-elements × 8 cols
mc[8]   — 8 C-tiles, each 8×8, covering 64 rows × 32 cols
```

The outer-product loop computes `mc[i] += mb[i/4] * ma[i%4]` for `i ∈
0..7`. This is a 4×2 = 8-tile outer product, producing a 64×32 output
tile from 4×2 = 8 tile multiplications. Per `loop_k` iteration, the
kernel does `NK/8 = 4` such outer products, for a total of 32 tile
multiplications per `loop_k` step.

### 9.3 Flash attention matmuls

`kernel_flash_attn_ext_impl` uses two simdgroup matmuls:

* **QK^T**: `simdgroup_multiply_accumulate(mqk, mq, mk, mqk)` (line
  6623) — `mq` is 8×DK, `mk` is 8×C (transposed), `mqk` is 8×C.
* **PV**: `simdgroup_multiply_accumulate(lo[ii], vs, mv[ii], lo[ii])`
  (line 6800) — `vs` is 8×C, `mv[ii]` is 8×DV, `lo[ii]` is 8×DV.

The QK^T matmul reads K from shmem (`sk`); the PV matmul reads V from
shmem (`sv`). Both K and V are staged through the same shmem region
(`sk`/`sv` share offset, loaded at different times in the KV loop).

### 9.4 `mul_mv_ext` register-only path

The ext path avoids shmem entirely by using `simd_shuffle_down` for the
intra-row reduction (lines 3988-4001). Each thread accumulates a
partial `sumf[ir1]` for `r1ptg` rows; the reduction shuffles the
partial down the simdgroup until lane 0 has the final sum. This works
because each row's threads form a contiguous subgroup of `nxpsg`
threads inside the 32-thread simdgroup.

### 9.5 Capability gating

* `simdgroup_matrix` requires `MTLGPUFamilyApple7` (ARTX15 §9.1).
* `simdgroup_reduction` (simd_sum, simd_min, simd_max) requires the
  same.
* `bfloat` types require `MTLGPUFamilyMetal3_GGML` or
  `MTLGPUFamilyApple6` (ARTX15 §9.4).
* `mpp::tensor_ops::matmul2d` requires `MTLGPUFamilyMetal4_GGML` and
  `GGML_METAL_HAS_TENSOR` (ARTX15 §9.3).

Kernels gate these via `#if defined(GGML_METAL_HAS_BF16)` and
`#ifdef GGML_METAL_HAS_TENSOR` preprocessor guards. The host-side
`supports_op` rejects ops the device cannot run.

---

## 10. Quantization Strategy

### 10.1 `dequantize_*` device functions

Each quant format has a `device` template function:

```cpp
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg);
```

The function takes a block pointer, a sub-tile index `il ∈ 0..nl-1`,
and a `thread` reference to a 4×4 matrix. It unpacks 16 elements from
the block into the matrix. The `il` parameter selects which 16-element
sub-tile of the block to unpack — for Q4_0 (QK=32, nl=2), `il=0` gives
elements 0..15 and `il=1` gives elements 16..31.

There is also a `_t4` variant that produces a `float4` instead of a
4×4 matrix, used by `kernel_mul_mv_ext_q4_f32_impl` and
`kernel_cpy_q_f32`.

### 10.2 Per-thread vs per-simdgroup unpacking

In `kernel_mul_mm` (line 10176-10196), each thread dequantizes one
4×4 sub-tile into a `S0_4x4` register, then scatters 16 elements into
`sa` at a position determined by `(sx, sy, lx, ly)`. The scatter
pattern is non-trivial:

```cpp
const short sx = 2*il0 + i/8;
const short sy = (tiitg/NL0)/8;
const short lx = (tiitg/NL0)%8;
const short ly = i%8;
const short ib = 8*sx + sy;
*(sa + 64*ib + 8*ly + lx) = temp_a[i/4][i%4];
```

This layout ensures that when a simdgroup later does `simdgroup_load(ma,
lsma + 64*i, 8, 0, false)`, it reads an 8×8 tile in the layout expected
by the simdgroup matrix API. The mapping is hand-tuned to avoid bank
conflicts.

In `kernel_mul_mv_q4_0_f32`, the unpacking is even more local: each
thread calls `block_q_n_dot_y(ax[row] + ib, ...)` which inlines the
unpacked bits directly into the dot product accumulator. No shmem
involvement at all.

### 10.3 `block_q_n_dot_y` per-quant overloads

`ggml-metal.metal:3317-3481` defines overloads for `block_q1_0`,
`block_q2_0`, and (implicitly) Q4_0/Q4_1/Q5_0/Q5_1 via the
`dequantize_*` + `dot` pattern. The Q1_0 and Q2_0 versions are
bit-twiddling specialists:

```cpp
inline float block_q_n_dot_y(device const block_q1_0 * qb, float sumy, thread float * yl, int il) {
    // 16 bit tests, accumulate yl[i] where bit is set
    return qb_curr->d * (2.0f * acc - sumy);
}
```

These avoid the 4×4 matrix intermediate and go directly from bits to
dot product. Faster than the `dequantize_q4_0` + `dot` path.

### 10.4 K-quant path

K-quants (`block_q4_K`, etc.) have a more complex block structure: 6-bit
scales, 4-bit quantized values, and a per-block `d` / `dmin` pair. The
`kernel_mul_mv_q4_K_f32_impl` (line 8420) unpacks these directly in
the inner loop, using `uint16_t sc16[4]` and `thread const uint8_t *
sc8 = (thread const uint8_t *)sc16` to access the 6-bit scales via
bit-masking. No `dequantize_q4_K` device function is called; the
unpacking is fully inlined.

### 10.5 `quantize_*` device functions

`ggml-metal.metal:240-469` defines `quantize_q1_0` through
`quantize_iq4_nl`. Each is a `device` function that takes a `device
const float *` (input) and a `device block_q &` (output). They are
annotated `#pragma METAL fp math_mode(safe)` — this disables fast-math
so that `round()` and division behave correctly. Each function
processes one block (e.g., 32 elements for Q4_0).

These are called from `kernel_cpy_f32_q` (one block per thread) and
`kernel_set_rows_q32` (one block per thread). There is no
cooperative-quantize kernel that uses shmem to share the amax/min
reduction across threads — each thread does the full block solo.

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

* **Simdgroup matmul.** `simdgroup_multiply_accumulate` accumulates in
  the simdgroup's register file. The reduction order is fixed by the
  Metal implementation but reassociates vs. a strict left-to-right
  scalar sum at the ULP level.
* **`helper_mv_reduce_and_write`.** The two-level reduction (intra-sg
  `simd_sum` then cross-sg `simd_sum` via shmem) reassociates vs. a
  single-tree reduction. The result is deterministic for a fixed
  `(NSG, NR0)` but differs from a scalar reference at ULP level.
* **`mul_mv_ext`.** The `simd_shuffle_down` reduction (lines 3988-4001)
  is a tree reduction: `sum += sum_shuffled(16); sum += sum_shuffled(8);
  ...`. Reassociates vs. a scalar sum.

### 11.2 Approximate math

* **`pow` in `kernel_rope_*`** (line 4623, 4676, 4759). Each thread
  computes `theta = theta_base * pow(args.freq_base, inv_ndims*i0)`.
  This is a transcendental per element; precision is whatever the
  Metal `pow` provides (typically faithful single-precision).
* **`exp` in `kernel_soft_max`** (line 2016). Per-element `exp`; same
  precision note.
* **`sqrt` in `kernel_norm_fuse_impl`** (line 3084). Per-row; standard
  precision.

### 11.3 Precision reduction

* **`mul_mm` threadgroup A storage.** Even when src0 is F32, the
  threadgroup `sa` stores `S0 = half` (per the template instantiation
  `kernel_mul_mm_f32_f32` at line 10684, which uses `S0 = half`).
  This is a deliberate precision trade-off: halve the shmem footprint,
  accept f16 intermediate precision. The accumulator `mc[8]` is
  `simdgroup_float8x8` (f32), so the matmul itself accumulates in
  f32 — only the A tile storage is f16.
* **`mul_mm` threadgroup B storage.** Same: `S1 = half` even for F32
  src1.
* **Flash attention.** K and V are stored in shmem as `half`
  (`threadgroup half * shmem_f16`). QK^T accumulates in f32
  (`simdgroup_float8x8 mqk`); PV accumulates in f32. Standard
  FlashAttention precision trade-off.

### 11.4 Non-deterministic reductions

* **Cross-simdgroup shmem reduction.** The `if (tiisg == 0)
  shmem_f32[row][sgitg] = sumf[row]` pattern in
  `helper_mv_reduce_and_write` is deterministic (each simdgroup writes
  a unique slot). The second `simd_sum(shmem_f32[row][tiisg])` reads
  the slots in a fixed order. Deterministic for fixed `(NSG, NR0)`.
* **Concurrent dispatch within a command buffer.** ARTX15 §11.4: when
  `use_concurrency` is true, Metal may overlap dispatches that write
  to disjoint buffers. Each dispatch is internally deterministic; the
  cross-dispatch order does not affect results because the overlap
  tracker prevents conflicts (ARTX15-F10).

### 11.5 Atomic accumulation

None in the matmul or attention paths. Output tiles are written by
exactly one threadgroup each. The only `atomic_int` in the kernel
source is `device atomic_int * dst` in `kernel_count_equal`
(line 11167), used to accumulate a global count across threadgroups.

### 11.6 Architecture-specific assumptions

* `N_SIMDWIDTH = 32` (`ggml-metal.metal:28`). Hardcoded; assumes
  Apple Silicon simdgroup size of 32. True on every Apple GPU; would
  break on a hypothetical 64-wide simdgroup device.
* `SZ_SIMDGROUP = 16` (`ggml-metal-impl.h:8`). Used in `mul_mm` tile
  math: `NRA = 16 * 2 * 2 = 64`. This is the simdgroup matrix
  dimension (8×8 matrices per simdgroup, but the simdgroup itself is
  32 threads). The name `SZ_SIMDGROUP` is misleading — it is not the
  simdgroup width but the matrix block factor.
* `MAXHALF` (used in `kernel_flash_attn_ext_blk`, line 6275). The
  maximum half-precision value (65504). Used as a sentinel for
  "no valid mask". Assumed to be defined in `<metal_stdlib>`.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                            | Where                                       | Notes                                                       |
| --------------------------------------- | ------------------------------------------- | ----------------------------------------------------------- |
| Simdgroup matrix outer-product tile     | `ggml-metal.metal:10238-10259`              | 4 A-tiles + 2 B-tiles + 8 C-tiles per inner iteration       |
| `simdgroup_barrier(mem_none)` hint      | `ggml-metal.metal:10239, 10245, 10251`      | Compiler hint; no actual sync                               |
| Register-only `mul_mv_ext`              | `ggml-metal.metal:3916-4015`                | No shmem; `simd_shuffle_down` reduction                     |
| `block_q_n_dot_y` bit-twiddling         | `ggml-metal.metal:3318-3481`                | Direct bit→accumulator for Q1_0/Q2_0; no 4×4 intermediate   |
| Per-thread `ax[NR0]` pointer array      | `ggml-metal.metal:3563-3568`                | Register-resident row pointers; no shmem indirection        |
| `helper_mv_reduce_and_write<NR0>`       | `ggml-metal.metal:3483-3523`                | Templated reduction; reuses shmem layout across dtypes      |
| `mul_mm` shmem reuse for output staging | `ggml-metal.metal:10275`                    | `temp_str` overlaps `sa` after matmul completes             |
| `#pragma METAL fp math_mode(safe)` on quantize | `ggml-metal.metal:279, 309, 338, ...` | Disables fast-math for `round()` correctness                |
| `constexpr constant static` LUTs        | `ggml-metal.metal:47-53`                    | IQ4_NL and MXFP4 dequant tables in device constant memory   |
| `FOR_UNROLL(x)` macro                   | `ggml-metal.metal:26`                       | `_Pragma("clang loop unroll(full)")` — full unroll hint     |
| `FOR_UNROLL` on row loops               | `ggml-metal.metal:3564, 3588, 3598, ...`    | Unrolls NR0=4 (or 2, 8) row loops                           |
| Vectorized `float4` loads in `mul_mv_t_t_4` | `ggml-metal.metal:4424`                 | 4 elements per load; reduces instruction count              |
| `mul_mv_t_t_short` for `ne00 < 32`      | `ggml-metal.metal:4485-4521`                | Single-simdgroup, no shmem, no reduction — minimal overhead |
| `kernel_soft_max_4` vectorized variant  | `ggml-metal.metal:2056-2161`                | `float4` loads; 4x fewer iterations                         |
| Function-constant branch elimination    | `ggml-metal.metal:3541, 6307-6321, ...`     | `FC_*` constants eliminate dead branches at compile time    |
| Template-per-dtype specialization       | `ggml-metal.metal:10684-10735`              | One `kernel_mul_mm` template, ~50 dtype instantiations      |
| `_t4` dequantize variants               | `ggml-metal.metal:96-99, 156-170, ...`      | `float4` output for ext GEMV path                           |
| `kernel_cpy_f32_q` quantize-on-copy     | `ggml-metal.metal:7882-7914`                | Fuses copy + quantize into one kernel dispatch              |

### 12.2 Optimizations *not* present

* **No autotuner.** All tile sizes (`NR0`, `NR1`, `NK`, `N_R0_*`,
  `N_SG_*`, `nqptg`, `ncpsg`, `nsg`) are compile-time constants. The
  only runtime choice is `nsg ∈ {4, 8}` for flash attention, decided
  by `ne00 >= 512`. No per-shape or per-device benchmarking.
* **No double-buffering.** The `mul_mm` K loop has a single
  `threadgroup_barrier` between the load phase and the compute phase.
  There is no overlap of `loop_k+1` loads with `loop_k` compute. A
  double-buffered version would allocate 2× shmem (12 KiB → 24 KiB)
  and remove the barrier from the critical path.
* **No software prefetch.** All loads are demand loads. No
  `threadgroup_async_copy` or similar prefetch hint (Metal does not
  expose one anyway).
* **No persistent threads.** Each threadgroup computes exactly one
  output tile and exits. No threadgroup persists across multiple
  tiles, so the L2 cache cannot be reused across tiles.
* **No `kernel_rope` fusion with downstream MUL/MV.** Each `rope` is
  a separate dispatch. The plan-time optimizer (ARTX15-F11) reorders
  ops for concurrency but does not fuse `rope` with its consumer.
* **No cooperative quantize.** Each `quantize_*` call processes one
  block solo; the amax/min scan is single-threaded. A cooperative
  version would split the QK=32 elements across 4 threads (8 each),
  reduce via `simd_max`, then quantize. Negligible gain for QK=32;
  could matter for K-quants (QK_K=256).

---

## 13. Architectural Strengths

1. **`helper_mv_reduce_and_write<NR0>` is a clean reduction template.**
   Every `mul_mv` kernel that needs cross-simdgroup reduction calls the
   same templated helper, parameterized only on `NR0`. The shmem
   layout (`NR0 * 32 * sizeof(float)`) is computed once and reused.
   Adding a new dtype means writing the dot-product kernel; the
   reduction is free.

2. **`mul_mv_ext` register-only path eliminates shmem entirely.** For
   small-batch mat-mv (the LLM decode case), the ext path uses
   `simd_shuffle_down` for intra-simdgroup reduction and no shmem.
   This is the optimal design for the decode workload — no shmem
   allocation, no barriers, no L2 traffic for tile staging.

3. **`block_q_n_dot_y` bit-twiddling for Q1_0/Q2_0.** These overloads
   bypass the `dequantize_*` 4×4 intermediate and go directly from
   packed bits to dot-product accumulator. The pattern
   `acc += select(0.0f, yl[i], bool(b0 & 0x01))` maps to a single
   `select` instruction per bit. This is the fastest possible Q1_0/Q2_0
   GEMV on Apple Silicon.

4. **`mul_mm` shmem reuse for output staging.** When `bc_out` is true
   (output tile doesn't fit the matrix bounds), the kernel stages the
   output through shmem (`temp_str` overlaps the now-dead `sa` region)
   before scalar-copying to device with bounds check. This avoids a
   separate shmem allocation for the staging buffer.

5. **Per-dtype `N_R0_*` / `N_SG_*` constant table.** Each quant gets
   its own tuned `(NR0, NSG)` pair in `ggml-metal-impl.h:24-88`. Q8_0
   (cheap per-element unpack) gets `NR0=2, NSG=4` (more rows per
   threadgroup, fewer rows per simdgroup); Q4_K (expensive per-block
   unpack) gets `NR0=2, NSG=2` (less work per simdgroup). The table
   is small, explicit, and easy to retune.

6. **`#pragma METAL fp math_mode(safe)` on quantize.** Quantization
   requires correct `round()` and correct division — fast-math would
   break it. The pragma is a clean way to opt out of fast-math per
   function without disabling it globally.

7. **Template-driven code generation.** One `kernel_mul_mm` template
   serves ~50 dtype instantiations. Adding a new quant means writing
   one `dequantize_*` function and one
   `template [[host_name(...)]] kernel mul_mm_t kernel_mul_mm<...>`
   line. No new kernel code.

8. **`simdgroup_barrier(mem_flags::mem_none)` as a compiler hint.**
   The kernel author uses these to mark scheduling boundaries without
   paying for an actual sync. Subtle but effective — the Metal
   compiler treats them as ordering hints for the surrounding loads.

---

## 14. Architectural Weaknesses

### W1 — Hardcoded `4096`-byte split in `kernel_mul_mm`

**Evidence**: `ggml-metal.metal:10106` `threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);`

The split point between `sa` and `sb` is a hardcoded `4096` byte
offset. This works because every template instantiation uses
`S0 = half` (or `bfloat`), so `sa` occupies exactly
`4 * 64 * 8 * sizeof(half) = 4096` bytes. If a future instantiation
used `S0 = float`, `sa` would need 8192 bytes and overflow into `sb`.

**Impact**: Fragile. The `4096` constant is correct by coincidence
(sizeof(half) == 2), not by construction. A future `kernel_mul_mm`
variant for a hypothetical f32-storage dtype would silently corrupt
`sb`.

### W2 — Dead `nr0 ∈ {1, 3, 4}` cases in `kernel_mul_mv_t_t_disp`

**Evidence**: `ggml-metal.metal:4330-4335`. The dispatcher's `switch
(args.nr0)` has only `case 2:` enabled; cases 1, 3, 4 are commented
out. The host (`ggml-metal-device.cpp:797`) always sets `nr0 = 2`.

**Impact**: Dead code with no documentation. If a maintainer uncomments
the cases and forgets to also set the host-side `nr0` correctly, the
dispatcher would silently fall through (no `default`). Either remove
the dead cases or document why `nr0 = 2` is the only tuned value.

### W3 — `mul_mm` legacy path has no double-buffering

**Evidence**: `ggml-metal.metal:10156-10260`. The K loop body is:
load A, load B, `threadgroup_barrier`, compute, `threadgroup_barrier`
(at the start of the next iteration's load). There is no overlap of
`loop_k+1` loads with `loop_k` compute.

**Impact**: The barrier is on the critical path. A double-buffered
version (allocate 2× shmem, ping-pong between `sa[0]` and `sa[1]`)
would hide the load latency behind the compute. Cost: 12 KiB shmem
instead of 6 KiB (still well within the 32 KiB budget). The has_tensor
path uses `mpp::tensor_ops::matmul2d` which may internally
double-buffer, but the legacy path does not.

### W4 — `kernel_rope_*` uses `pow` per element with no LUT

**Evidence**: `ggml-metal.metal:4623` `const float theta = theta_base * pow(args.freq_base, inv_ndims*i0);`.

Each thread computes a `pow` (transcendental) per pair of elements.
The CPU backend (ARTX01-F07) precomputes GELU in a 128 KiB f16 LUT; no
equivalent exists for rope's `cos`/`sin`. The `rope_yarn` helper
computes the correction factors inline.

**Impact**: `pow` is ~10–20 cycles on Apple Silicon. For a 4096-dim
rope, that's 2048 `pow` calls per row. A 2 KiB LUT (256 entries of
`cos`/`sin` pairs) would replace the `pow` with a table lookup and
linear interpolation. Not implemented.

### W5 — `mul_mv_t_t_short` for `ne00 < 32` is single-threaded per row

**Evidence**: `ggml-metal.metal:4493` `const int r0 = tgpig.x*32 + tiisg;`.
Each thread computes one full row of the output as a scalar dot
product. No simdgroup cooperation, no shmem.

**Impact**: For `ne00 < 32`, the row is short enough that the
scalar dot product is faster than the simdgroup setup overhead. But
the threshold (`ne00 < 32`) is hardcoded; for `ne00 ∈ [32, 64]` the
regular `mul_mv_t_t` path uses 2 simdgroups, which may be overkill.
No intermediate "small but simdgroup" path.

### W6 — `mul_mm` legacy path `sa` is `half` even when src0 is F32

**Evidence**: `ggml-metal.metal:10684`
`kernel_mul_mm_f32_f32` instantiates with `S0 = half, S0_4x4 =
half4x4, S0_8x8 = simdgroup_half8x8`. The F32 weight is downcast to
`half` on the way into shmem.

**Impact**: Precision loss in the A tile. The matmul accumulates in
f32 (`mc[8]` is `simdgroup_float8x8`), but the A operand of each
`simdgroup_multiply_accumulate` is f16. For weights that span a wide
dynamic range (e.g., post-softmax attention scores), this can cause
measurable error. The `has_tensor` path reads A directly from device
as f32 and avoids this.

### W7 — Flash attention `nsg` heuristic is hardcoded `4-or-8-by-ne00`

**Evidence**: `ggml-metal-ops.cpp:2835` `int32_t nsg = ne00 >= 512 ? 8 : 4;`.
The commented-out autotuner code above (lines 2819-2834) shows the
intent: compute `nsgmax` from `max_theadgroup_memory_size`, then pick
the largest `nsg` that fits. But the autotuner is disabled and the
heuristic is just `ne00 >= 512`.

**Impact**: Suboptimal for head sizes near 256 or 384. Also ignores
`ne11` (KV cache length), which affects how many KV iterations each
simdgroup processes.

### W8 — `helper_mv_reduce_and_write` always allocates shmem even when `NSG == 1`

**Evidence**: `ggml-metal.metal:3483-3523`. The helper writes to
`shmem_f32[row][sgitg]` unconditionally. If `NSG == 1`, only one
simdgroup is active, the shmem write and second barrier are no-ops in
effect (only one slot is written, only one slot is read). But the
barrier still executes.

**Impact**: One unnecessary `threadgroup_barrier` per call when
`NSG == 1`. The Q1_0 path (`kernel_mul_mv_q1_0_f32_impl`, line 3619)
already has a fast-path that skips `helper_mv_reduce_and_write` when
`NSG == 1` — it uses `simd_sum` directly. Other kernels (Q8_0,
`mul_mv_t_t`) always call the helper.

### W9 — `mul_mv` K-quants bypass shmem but still pay the `N_R0_*` template tax

**Evidence**: `ggml-metal.metal:8408-8417`. `kernel_mul_mv_q4_K_f32`
passes `nullptr` for `shmem`. The `kernel_mul_mv_q4_K_f32_impl`
template still takes a `threadgroup char * shmem` parameter (line 8426)
but never uses it.

**Impact**: Dead parameter. The dispatcher in
`ggml-metal-device.cpp:855-858` sets `smem = 0` for K-quants, so no
shmem is allocated. But the kernel signature still implies it might
use shmem, which is misleading.

### W10 — `simdgroup_barrier(mem_flags::mem_none)` is undocumented

**Evidence**: `ggml-metal.metal:10239, 10245, 10251, 6616, 6621, ...`.
The call appears between `simdgroup_load` batches with no comment
explaining why. A reader might think it's a real sync; in fact it's a
compiler hint.

**Impact**: Readability and maintainability. A GwenLand engineer
reading this code would not know whether removing the call is safe.
A `// compiler hint: no actual sync` comment would help.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glmetal`       | **ADOPT** | `helper_mv_reduce_and_write<NR0>` template | Clean reusable reduction; same shmem layout across dtypes |
| `glmetal`       | **ADOPT** | `simdgroup_multiply_accumulate` outer-product loop (4 ma + 2 mb + 8 mc) | Canonical Metal SMEM-based GEMM tile |
| `glmetal`       | **ADOPT** | `block_q_n_dot_y` bit-twiddling for Q1_0/Q2_0 | Fastest possible GEMV for these quants |
| `glmetal`       | **ADOPT** | Per-dtype `N_R0_*` / `N_SG_*` constant table | Small, explicit, easy to retune |
| `glmetal`       | **ADOPT** | `mul_mv_ext` register-only path with `simd_shuffle_down` | Optimal for decode (small-batch mat-mv) |
| `glmetal`       | **ADOPT** | `mul_mm` shmem reuse for `bc_out` staging | Saves an allocation; clever but safe |
| `glmetal`       | **ADOPT** | `#pragma METAL fp math_mode(safe)` on quantize | Per-function fast-math opt-out |
| `glmetal`       | **ADOPT** | Flash attention 6-region shmem layout (Q/O/SS/K/V/mask in one buffer) | Compact; overlapping K/V scratch is correct |
| `glmetal`       | **ADAPT** | `mul_mm` legacy `sa`/`sb` split | Replace hardcoded `4096` with `sizeof(S0) * NRA * N_MM_NK_TOTAL` |
| `glmetal`       | **ADAPT** | `mul_mv_t_t_disp` dispatcher | Remove dead `nr0` cases or enable them with autotuning |
| `glmetal`       | **ADAPT** | Flash attention `nsg` heuristic | Re-enable the autotuner that picks `nsg` from `max_theadgroup_memory_size` |
| `glmetal`       | **ADAPT** | `helper_mv_reduce_and_write` | Add an `NSG == 1` fast path that skips shmem and the second barrier |
| `glmetal`       | **REJECT**| Hardcoded `4096`-byte split in `kernel_mul_mm` | Compute split from `sizeof(S0)`; never hardcode byte offsets |
| `glmetal`       | **REJECT**| `mul_mm` legacy f16 storage of A when src0 is F32 | Use `has_tensor` path or accept f32 storage |
| `glmetal`       | **MONITOR**| `mpp::tensor_ops::matmul2d` (Metal4) | Currently disabled for pre-M5; revisit when M5 benchmarks are in |
| `glmetal`       | **DEFER** | Double-buffering in `mul_mm` legacy path | Worth 2× shmem for hidden load latency; implement when profiling shows barrier-bound |
| `glmetal`       | **DEFER** | `kernel_rope` cos/sin LUT | Only relevant if rope becomes a hot path; currently dominated by matmul |
| `GATE`          | **ADOPT** | Per-dtype `(NR0, NSG)` table as a host-side policy | Same idea as CPU type-traits (ARTX01-F03); lets GATE pick tile sizes per dtype |
| `GATE`          | **ADAPT** | `nsg` autotuner for flash attention | Move from kernel-side heuristic to GATE-side per-shape policy |

---

## 16. Recommendations

### R1 — ADOPT `helper_mv_reduce_and_write<NR0>` template
**Priority:** High | **Difficulty:** S | **Dependencies:** none
GwenLand's `glmetal` should define an equivalent `template <short NR0>
void gl_metal_mv_reduce_and_write(device float * dst, float sumf[NR0],
const int r0, const int ne01, ushort tiisg, ushort sgitg, threadgroup
char * shmem)`. Same two-barrier pattern. Reuse for every GEMV dtype.

### R2 — ADOPT `simdgroup_multiply_accumulate` outer-product tile
**Priority:** Critical | **Difficulty:** M | **Dependencies:** none
GwenLand's `glmetal` `kernel_mul_mm` legacy path should replicate the
4-ma + 2-mb + 8-mc outer-product loop. Use
`simdgroup_barrier(mem_flags::mem_none)` between `simdgroup_load`
batches as a compiler hint. Document that these are no-op hints, not
real syncs.

### R3 — ADOPT per-dtype `N_R0_*` / `N_SG_*` constant table
**Priority:** High | **Difficulty:** S | **Dependencies:** R1
Define a `gl_metal_dtype_tile` table indexed by `ggml_type` with
fields `nr0`, `nsg`, `smem`. Same values as `ggml-metal-impl.h:24-88`.
Make the values tunable per-device (overrides for Apple7 vs Apple8 vs
Metal3 vs Metal4).

### R4 — ADAPT `mul_mm` `sa`/`sb` split: compute from `sizeof(S0)`
**Priority:** High | **Difficulty:** XS | **Dependencies:** R2
Replace `shmem + 4096` with `shmem + sizeof(S0) * NRA * N_MM_NK_TOTAL`
(or equivalently `shmem + sizeof(S0) * 4 * 64 * 8`). This makes the
split safe for any `S0` type.

### R5 — REJECT f16 storage of A in `mul_mm` when src0 is F32
**Priority:** Medium | **Difficulty:** M | **Dependencies:** R2
For `kernel_mul_mm_f32_f32`, use `S0 = float` (and `S0_8x8 =
simdgroup_float8x8`). Cost: 8 KiB shmem for `sa` instead of 4 KiB.
Still within the 32 KiB budget. Gain: no precision loss in A tile.

### R6 — ADAPT flash attention `nsg` autotuner
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R3
Re-enable the commented-out code at `ggml-metal-ops.cpp:2819-2834`.
Compute `nsgmax` from `max_theadgroup_memory_size` and the
`FATTN_SMEM(nsg)` formula. Pick the largest `nsg ≤ nsgmax` that also
satisfies `nsg ≤ ne11/ncpsg` (don't allocate more simdgroups than
there are KV blocks).

### R7 — ADAPT `helper_mv_reduce_and_write` with `NSG == 1` fast path
**Priority:** Medium | **Difficulty:** XS | **Dependencies:** R1
Add `if (FC_mul_mv_nsg == 1) { simd_sum; if (tiisg == 0) dst[r0+row] =
sumf[row]; return; }` at the top. Skips both barriers and the shmem
write/read. Matches the existing fast path in
`kernel_mul_mv_q1_0_f32_impl`.

### R8 — ADOPT `block_q_n_dot_y` bit-twiddling for Q1_0/Q2_0
**Priority:** Medium | **Difficulty:** S | **Dependencies:** R1
For Q1_0 and Q2_0 GEMV, skip the 4×4 dequantize intermediate and go
directly from packed bits to dot-product accumulator using `select`.
This is the fastest path on Apple Silicon for these quants.

### R9 — ADOPT `mul_mv_ext` register-only path
**Priority:** High | **Difficulty:** M | **Dependencies:** R1
For small-batch mat-mv (the LLM decode case), use a kernel that takes
no threadgroup memory, uses `simd_shuffle_down` for intra-simdgroup
reduction, and assigns each thread a 2D (x, y) coordinate inside the
simdgroup. This is the optimal decode path.

### R10 — DEFER double-buffering in `mul_mm` legacy path
**Priority:** Low | **Difficulty:** M | **Dependencies:** R2
Allocate 2× shmem (12 KiB) and ping-pong between two `sa`/`sb` pairs.
Removes the barrier from the critical path. Only worth doing if
profiling shows the legacy path is barrier-bound; the has_tensor path
likely does not benefit.

### R11 — MONITOR `mpp::tensor_ops::matmul2d` (Metal4 tensor path)
**Priority:** Medium | **Difficulty:** L | **Dependencies:** none
The has_tensor path uses Apple's `matmul2d` primitive, which may
internally double-buffer and use AMX-equivalent hardware. Currently
disabled for pre-M5 chips (ARTX15-F10). Revisit when M5/A19 hardware
is available and benchmarks show a clear win.

### R12 — ADOPT `constexpr constant static` LUT pattern
**Priority:** Low | **Difficulty:** XS | **Dependencies:** none
For IQ4_NL and MXFP4 dequant tables, use `constexpr constant static
float table[16] = {...}` at file scope. These live in device constant
memory, shared across all threads. Same pattern as CPU's
`ggml_table_*` (ARTX01 §7.4) but without the runtime init.

---

## 17. Findings

### Finding ARTX16-F01

```
Finding ID:           ARTX16-F01
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            kernel_mul_mm (legacy path)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mm
Lines:                10105-10106
Summary:              Threadgroup memory split between sa and sb uses a
                      hardcoded 4096-byte offset, correct only because
                      S0 is always half.
Observation:          The kernel declares `threadgroup S0 * sa =
                      (threadgroup S0 *)(shmem);` and `threadgroup S1
                      * sb = (threadgroup S1 *)(shmem + 4096);`. The
                      4096 is a byte offset. The sa region needs
                      4*64*8*sizeof(S0) bytes; for S0=half this is
                      exactly 4096 bytes, so the split is correct. For
                      S0=float it would be 8192 bytes and overflow sb.
                      Every template instantiation (lines 10684-10735)
                      uses S0=half or S0=bfloat, so the bug is latent.
Evidence:             ggml-metal.metal:10106 (split), 10684-10735
                      (instantiations all use S0=half/bfloat).
Architectural Impact: The split is fragile — it depends on a
                      coincidence between the hardcoded constant and
                      sizeof(S0). Adding a new dtype that wants f32
                      threadgroup storage would silently corrupt sb.
Correctness Impact:   None today. Latent correctness risk if a future
                      dtype uses S0=float.
Optimization Type:    None (layout choice).
GwenLand Target:      glmetal
Recommendation:       ADAPT. Compute the split as
                      `sizeof(S0) * NRA * N_MM_NK_TOTAL` instead of
                      `4096`. Same value for S0=half; safe for any S0.
Priority:             Medium
Difficulty:           XS
Dependencies:         R2
Confidence:           High
```

### Finding ARTX16-F02

```
Finding ID:           ARTX16-F02
Category:             SIMD_STRATEGY
Engine:               Metal
Component:            kernel_mul_mm (legacy path)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mm
Lines:                10147-10259
Summary:              Outer-product accumulation uses 4 simdgroup A-tiles
                      + 2 B-tiles + 8 C-tiles per inner iteration, with
                      simdgroup_barrier(mem_none) hints between loads.
Observation:          The K loop body loads 4 A-tiles (ma[0..3]) and 2
                      B-tiles (mb[0..1]) from shmem, then computes 8
                      outer products `mc[i] += mb[i/4] * ma[i%4]` for
                      i in 0..7. Between the load batches,
                      `simdgroup_barrier(mem_flags::mem_none)` is
                      called as a compiler hint (no actual sync). This
                      produces a 64x32 output tile per threadgroup
                      using 4 simdgroups (2x2 grid).
Evidence:             ggml-metal.metal:10147 (ma[4], mb[2], mc[8]),
                      10238-10259 (K loop with simdgroup_barrier hints),
                      10254 (simdgroup_multiply_accumulate).
Architectural Impact: This is the canonical Metal simdgroup-matrix
                      GEMM tile. The 4x2 outer product covers 32
                      K-elements per inner iteration; NK/8=4 iterations
                      per loop_k step.
Correctness Impact:   None. Simdgroup matmul is deterministic.
Optimization Type:    SIMD / tiling / blocking.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the 4-ma + 2-mb + 8-mc pattern
                      in glmetal's kernel_mul_mm legacy path. Document
                      that simdgroup_barrier(mem_none) is a hint.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX16-F03

```
Finding ID:           ARTX16-F03
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            kernel_mul_mv family (GEMV)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             mul_vec_q_n_f32_impl, kernel_mul_mv_q4_K_f32_impl, kernel_mul_mv_q8_0_f32_impl
Lines:                3531-3617, 8029-8119, 3826-3898
Summary:              GEMV kernels register-pack weight rows
                      (device const block_q * ax[NR0]) and avoid
                      threadgroup memory for tile staging; only the
                      cross-simdgroup reduction uses shmem.
Observation:          For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and F32/F16/BF16,
                      each thread holds NR0 device pointers to weight
                      rows in registers (`device const block_q * ax[NR0]`)
                      and NR0 accumulators (`float sumf[NR0]`). The
                      activation y is read directly from device memory
                      and cached in `thread float yl[16]` per iteration.
                      Shmem is used only by helper_mv_reduce_and_write
                      for the cross-simdgroup reduction (NR0*32 floats).
                      For Q4_K, Q5_K, Q2_K, Q3_K, Q6_K, IQ2_*, IQ3_*,
                      IQ1_*, IQ4_* the kernel passes nullptr for shmem
                      entirely; single-simdgroup, simd_sum only.
Evidence:             ggml-metal.metal:3563-3568 (ax[NR0] pointer array),
                      3578 (yl[16] register cache), 8131 (nullptr shmem
                      for Q2_K), 3897-3898 (helper_mv_reduce_and_write
                      call for Q8_0).
Architectural Impact: Avoiding shmem for tile staging keeps the
                      threadgroup memory footprint tiny (<=1 KiB) and
                      avoids barriers in the K loop. The cost is
                      re-reading y from device memory inside each
                      thread (no broadcast via shmem).
Correctness Impact:   None.
Optimization Type:    Register blocking / SIMD.
GwenLand Target:      glmetal
Recommendation:       ADOPT. The register-pack pattern is optimal for
                      GEMV; replicate in glmetal's mul_mv family.
Priority:             High
Difficulty:           S
Dependencies:         R1
Confidence:           High
```

### Finding ARTX16-F04

```
Finding ID:           ARTX16-F04
Category:             SIMD_STRATEGY
Engine:               Metal
Component:            Per-dtype tile constants
Source File:          ggml/src/ggml-metal/ggml-metal-impl.h, ggml/src/ggml-metal/ggml-metal-device.cpp
Function:             N_R0_*, N_SG_* constants; ggml_metal_library_get_pipeline_mul_mv
Lines:                impl.h:24-88, device.cpp:803-955
Summary:              Per-dtype (NR0, NSG) constants are compile-time
                      #defines; each quant gets a hand-tuned pair. Q8_0
                      is the only one with NSG=4; all others use NSG=2.
Observation:          The constants table assigns:
                      Q1_0: NR0=8, NSG=2; Q2_0: NR0=8, NSG=2;
                      Q4_0: NR0=4, NSG=2; Q4_1: NR0=4, NSG=2;
                      Q5_0: NR0=4, NSG=2; Q5_1: NR0=4, NSG=2;
                      Q8_0: NR0=2, NSG=4; MXFP4: NR0=2, NSG=2;
                      Q2_K..Q6_K: NR0=1..4, NSG=2;
                      IQ1/IQ2/IQ3/IQ4: NR0=2..4, NSG=2.
                      Q8_0 has cheap per-element unpack (just int8*scale),
                      so it uses more simdgroups (4) and fewer rows per
                      simdgroup (2). Q4_K has expensive per-block unpack,
                      so it uses fewer simdgroups (2) and 1-2 rows per
                      simdgroup.
Evidence:             ggml-metal-impl.h:24-88 (definitions),
                      ggml-metal-device.cpp:803-955 (host-side lookup).
Architectural Impact: The table is small, explicit, and easy to retune.
                      No autotuner; values are hardcoded per-dtype.
Correctness Impact:   None.
Optimization Type:    Per-dtype tile tuning.
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Replicate the table in glmetal; consider
                      making it per-device (overrides for Apple7 vs
                      Apple8 vs Metal3 vs Metal4).
Priority:             High
Difficulty:           S
Dependencies:         R3
Confidence:           High
```

### Finding ARTX16-F05

```
Finding ID:           ARTX16-F05
Category:             THREADING_MISMATCH
Engine:               Metal
Component:            helper_mv_reduce_and_write
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             helper_mv_reduce_and_write
Lines:                3483-3523
Summary:              Two-barrier cross-simdgroup reduction always
                      executes both barriers even when NSG==1.
Observation:          The reduction template unconditionally:
                      (1) simd_sum, (2) threadgroup_barrier, (3) write
                      shmem_f32[row][sgitg], (4) threadgroup_barrier,
                      (5) read shmem_f32[row][tiisg], (6) simd_sum.
                      When NSG==1, only one simdgroup is active; the
                      shmem write and second barrier are unnecessary
                      (only slot 0 is written, only slot 0 is read).
                      The Q1_0 path (kernel_mul_mv_q1_0_f32_impl, line
                      3619) already has a fast-path that bypasses the
                      helper and uses simd_sum directly. Other kernels
                      (Q8_0, mul_mv_t_t) always call the helper.
Evidence:             ggml-metal.metal:3506, 3514 (two barriers),
                      3610-3616 (Q1_0 fast-path that bypasses helper).
Architectural Impact: One unnecessary barrier per call when NSG==1.
                      For Q8_0 (NSG=4) the helper is needed; for
                      K-quants (NSG=2, but shmem is nullptr) the
                      helper is not called.
Correctness Impact:   None. The extra barrier is a no-op in effect.
Optimization Type:    None (barrier overhead).
GwenLand Target:      glmetal
Recommendation:       ADAPT. Add an `if (FC_mul_mv_nsg == 1)` fast-path
                      at the top of helper_mv_reduce_and_write that
                      skips the shmem dance.
Priority:             Medium
Difficulty:           XS
Dependencies:         R1, R7
Confidence:           High
```

### Finding ARTX16-F06

```
Finding ID:           ARTX16-F06
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            kernel_mul_mv_ext_q4_f32_impl / _q4x4_f32_impl
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mv_ext_q4_f32_impl, kernel_mul_mv_ext_q4x4_f32_impl
Lines:                3916-4015, 4019-4122
Summary:              The ext (small-batch mat-mv) kernels take no
                      threadgroup memory; reduction uses simd_shuffle_down.
Observation:          The kernel signatures omit the `threadgroup char *
                      shmem` parameter. Inside, each thread dequantizes
                      `chpt=4` chunks (or 1 chunk for the 4x4 variant)
                      into `thread float4 lx[chpt]` registers, computes
                      `sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg])` for
                      each ir1 in 0..r1ptg-1, then reduces via
                      `simd_shuffle_down(sumf[ir1], 16/8/4/2/1)` until
                      lane 0 has the final sum. The thread layout is
                      2D: `tx = tiisg%nxpsg` (K-dim chunk), `ty =
                      tiisg/nxpsg` (row). Each row's reduction is
                      independent because each row is handled by a
                      disjoint set of nxpsg threads inside the simdgroup.
Evidence:             ggml-metal.metal:3916-3923 (kernel signature, no
                      shmem), 3988-4001 (simd_shuffle_down reduction),
                      3932-3933 (2D thread layout).
Architectural Impact: Optimal for decode (small-batch mat-mv): no shmem
                      allocation, no barriers, no L2 traffic for tile
                      staging. The kernel is purely register + simd ops.
Correctness Impact:   None.
Optimization Type:    SIMD / register-only.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the ext path for glmetal's
                      decode GEMV. Use simd_shuffle_down for the
                      intra-row reduction.
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX16-F07

```
Finding ID:           ARTX16-F07
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_mul_mm (legacy path)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mm
Lines:                10156-10260
Summary:              K loop pipeline is: dequantize A → load B →
                      threadgroup_barrier → 4 iterations of (load ma,
                      load mb, simdgroup_multiply_accumulate) → next K.
                      No double-buffering.
Observation:          Each K iteration:
                      (1) Each thread dequantizes 16 A-elements into
                      sa (with FOR_UNROLL on the 16-element scatter).
                      (2) Each thread loads a S1_2x4 B-tile into sb.
                      (3) threadgroup_barrier(mem_threadgroup) — full
                      sync.
                      (4) NK/8=4 inner iterations of:
                          simdgroup_barrier(mem_none) [hint];
                          simdgroup_load(ma[0..3]);
                          simdgroup_barrier(mem_none);
                          simdgroup_load(mb[0..1]);
                          simdgroup_barrier(mem_none);
                          simdgroup_multiply_accumulate(mc[i], ...).
                      (5) Advance lsma, lsmb pointers.
                      At the top of the next K iteration, the load
                      phase implicitly waits for the compute to finish
                      (because sa is overwritten).
Evidence:             ggml-metal.metal:10156-10259 (full K loop),
                      10179 (threadgroup_barrier after load),
                      10232 (threadgroup_barrier before compute),
                      10239/10245/10251 (simdgroup_barrier hints).
Architectural Impact: The barrier is on the critical path. A double-
                      buffered version (2x shmem, ping-pong sa[0]/sa[1])
                      would hide load latency behind compute.
Correctness Impact:   None.
Optimization Type:    Tiling / blocking (no double-buffering).
GwenLand Target:      glmetal
Recommendation:       ADOPT the basic pipeline; DEFER double-buffering
                      until profiling shows it's barrier-bound.
Priority:             High
Difficulty:           M
Dependencies:         R2
Confidence:           High
```

### Finding ARTX16-F08

```
Finding ID:           ARTX16-F08
Category:             SIMD_STRATEGY
Engine:               Metal
Component:            kernel_soft_max, kernel_norm_fuse_impl, kernel_rms_norm_fuse_impl, kernel_l2_norm_impl, kernel_group_norm_f32, kernel_sum_rows_impl
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_soft_max (representative)
Lines:                1950-2053
Summary:              Reduction kernels use a two-level pattern: simd_sum
                      then conditional shmem reduction (only when
                      tptg.x > N_SIMDWIDTH).
Observation:          The kernel first does simd_max/simd_sum on the
                      per-thread partial. If tptg.x > 32 (more than one
                      simdgroup), it then:
                      (1) lane 0 of each simdgroup writes its partial
                      to shmem[sgitg];
                      (2) threadgroup_barrier;
                      (3) every thread reads shmem[tiisg] and does a
                      second simd_max/simd_sum.
                      The shmem allocation is 32*sizeof(float) = 128
                      bytes (one slot per simdgroup lane, max 32
                      simdgroups). When tptg.x <= 32 (single simdgroup),
                      shmem is untouched.
                      Same pattern in kernel_norm_fuse_impl (3014),
                      kernel_rms_norm_fuse_impl (3112),
                      kernel_l2_norm_impl (3184),
                      kernel_group_norm_f32 (3236),
                      kernel_sum_rows_impl (1705).
Evidence:             ggml-metal.metal:1996-2011 (soft_max conditional
                      shmem reduction), 3026-3058 (norm_fuse_impl),
                      3124-3154 (rms_norm_fuse_impl).
Architectural Impact: The conditional shmem path avoids the barrier
                      cost for single-simdgroup dispatches. The shmem
                      footprint is fixed at 128 bytes regardless of
                      tptg.x.
Correctness Impact:   None.
Optimization Type:    Two-level reduction with conditional shmem.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the pattern in glmetal's
                      reduction kernels. Make the shmem allocation
                      `min(32, max_simdgroups_per_threadgroup) *
                      sizeof(float)`.
Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX16-F09

```
Finding ID:           ARTX16-F09
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_rope_norm, kernel_rope_neox, kernel_rope_multi, kernel_rope_vision
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_rope_norm (representative)
Lines:                4595-4645
Summary:              Rope kernels use no threadgroup memory; each thread
                      processes 2 elements per tptg.x stride and
                      computes pow() per element.
Observation:          The kernel signature has no `threadgroup` parameter.
                      Each thread iterates `for (int i0 = 2*tiitg; i0 <
                      args.ne0; i0 += 2*tptg.x)` and computes:
                      `const float theta = theta_base * pow(args.freq_base,
                      inv_ndims*i0);` then `rope_yarn(theta/freq_factor,
                      ...)` to get cos/sin, then applies the 2x2 rotation
                      `[x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]`.
                      The `pow` is a transcendental per element pair.
                      No fusion with downstream MUL or MV.
Evidence:             ggml-metal.metal:4595-4603 (kernel signature, no
                      shmem), 4619-4636 (per-thread loop with pow),
                      4623 (pow call).
Architectural Impact: Pow per element is ~10-20 cycles. For a 4096-dim
                      rope, that's 2048 pow calls per row. A 2 KiB LUT
                      would replace pow with a table lookup. Not
                      implemented.
Correctness Impact:   None. Pow is faithful single-precision.
Optimization Type:    None (per-thread scalar).
GwenLand Target:      glmetal
Recommendation:       DEFER a cos/sin LUT until profiling shows rope is
                      hot. ADOPT the no-shmem, per-thread-2-elements
                      layout.
Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX16-F10

```
Finding ID:           ARTX16-F10
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_bin_fuse_impl
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_bin_fuse_impl
Lines:                1265-1415
Summary:              N-way fused ADD/MUL/SUB/DIV kernel uses no
                      threadgroup memory; FC_F (1..8) chains fully
                      unrolled per thread.
Observation:          The kernel is templated `<typename T0, typename
                      T1, typename T>` and parameterized by function
                      constants FC_OP (0=add, 1=sub, 2=mul, 3=div),
                      FC_F (1..8, the chain length), FC_RB (row
                      broadcast), FC_CB (circular broadcast). The
                      inner loop `FOR_UNROLL (short j = 0; j < FC_F;
                      ++j) res += src1_ptr[j][i10];` is fully unrolled
                      at compile time. No shmem, no barriers. Each
                      thread processes one output element per iteration.
                      Vectorized variants (float4) exist via template
                      instantiation.
Evidence:             ggml-metal.metal:1265-1272 (kernel signature),
                      1305-1332 (unrolled N-way chain), 1417-1420
                      (instantiations: float and float4).
Architectural Impact: Encodes the ARTX15-F14 bin x N fusion at the
                      kernel level. Up to 8 source tensors fused into
                      one dispatch. No reduction needed (elementwise).
Correctness Impact:   None.
Optimization Type:    Kernel fusion + full unroll.
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Replicate the FC_F-chained unrolled pattern
                      in glmetal's bin kernel. Make FC_F a function
                      constant so the compiler eliminates dead chains.
Priority:             High
Difficulty:           S
Dependencies:         R1 (function-constant specialization, ARTX15-F08)
Confidence:           High
```

### Finding ARTX16-F11

```
Finding ID:           ARTX16-F11
Category:             MEMORY_PATTERN
Engine:               Metal
Component:            kernel_flash_attn_ext_impl
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_flash_attn_ext_impl
Lines:                6353-6905
Summary:              Flash attention uses a 6-region shmem layout
                      (Q, O, scratch, K-scratch, V-scratch, mask) in
                      one flat threadgroup half* buffer; K and V
                      scratches overlap.
Observation:          The kernel declares 6 threadgroup pointers into
                      the same shmem_f16 buffer:
                      sq  at offset 0             (size Q*DK half)
                      so  at offset Q*DK          (size Q*PV half)
                      ss  at offset Q*T           (size Q*2*SH half, T=DK+2*PV)
                      sk  at offset sgitg*4*16*KV + Q*T + Q*TS (size 4*16*KV per simdgroup)
                      sv  at offset sgitg*4*16*KV + Q*T + Q*TS (same as sk - overlapped)
                      sm2 at offset Q*T + 2*C     (size C half2)
                      sk and sv share the same offset because they are
                      loaded at different times in the KV loop (sk in
                      the QK^T phase, sv in the PV phase). The buffer
                      is typed `threadgroup half *` but the ss region
                      is accessed as `threadgroup float *` (4-byte
                      loads from a 2-byte-typed buffer - legal because
                      the underlying storage is just bytes).
Evidence:             ggml-metal.metal:6402-6416 (6-region layout),
                      6409-6413 (sk/sv share offset), 6407 (ss cast to
                      s_t*=float*).
Architectural Impact: Compact shmem layout fits in <=16 KiB for typical
                      head sizes (DK=DV=128, nsg=4 -> ~7 KiB). The
                      overlap of sk/sv saves 4*16*KV*sizeof(half) = 2
                      KiB per simdgroup. The mixed half/float access
                      works because Metal does not enforce type
                      alignment on threadgroup memory.
Correctness Impact:   None. The overlap is safe because sk and sv are
                      never live simultaneously.
Optimization Type:    Memory layout / overlapping scratch buffers.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the 6-region layout. Document
                      the sk/sv overlap with a comment. Use the
                      FATTN_SMEM formula for the host-side smem sizing.
Priority:             High
Difficulty:           L
Dependencies:         R3
Confidence:           High
```

### Finding ARTX16-F12

```
Finding ID:           ARTX16-F12
Category:             QUANTIZATION
Engine:               Metal
Component:            quantize_* device functions
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             quantize_q4_0, quantize_q4_1, quantize_q5_0, quantize_q5_1, quantize_q8_0, quantize_q1_0, quantize_q2_0, quantize_iq4_nl
Lines:                240-469
Summary:              Quantize functions are scalar device functions,
                      one block per call, with #pragma METAL fp
                      math_mode(safe) to disable fast-math.
Observation:          Each quantize function takes `device const float
                      * src` and `device block_q & dst` and processes
                      exactly one quantization block (QK4_0=32 elements
                      for q4_0, QK_K=256 for k-quants, etc.). The
                      function does a serial amax/min scan, then a
                      serial quantize loop. No threadgroup cooperation,
                      no shmem. The `#pragma METAL fp math_mode(safe)`
                      annotation at the top of each function disables
                      fast-math so that `round()` and division are
                      IEEE-compliant.
                      Called from kernel_cpy_f32_q (one block per
                      thread) and kernel_set_rows_q32 (one block per
                      thread).
Evidence:             ggml-metal.metal:278-306 (quantize_q4_0),
                      279 (math_mode(safe) pragma), 7907-7913 (called
                      once per thread in kernel_cpy_f32_q).
Architectural Impact: Simple and correct. No cooperative quantize
                      kernel that splits a block across threads. For
                      QK=32 this is fine; for K-quants (QK_K=256) a
                      cooperative version might help, but the amax/min
                      scan is cheap relative to the quantize loop.
Correctness Impact:   math_mode(safe) is essential — fast-math would
                      break round() and division. Without it, the
                      quantized output would be subtly wrong.
Optimization Type:    None (scalar device function).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the scalar per-block pattern
                      with math_mode(safe). For K-quants, consider a
                      cooperative version if profiling shows quantize
                      is hot (unlikely during inference).
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX16-F13

```
Finding ID:           ARTX16-F13
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_mul_mm (two paths)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mm
Lines:                9964-10086 (tensor path), 10088-10303 (legacy path)
Summary:              kernel_mul_mm has two implementations selected by
                      GGML_METAL_HAS_TENSOR: legacy simdgroup path
                      (4 simdgroups, 64x32 tile, sa+sb in shmem) and
                      Metal4 tensor path (mpp::tensor_ops::matmul2d,
                      128x64 tile, sa only in shmem).
Observation:          The tensor path uses Apple's MetalPerformancePrimitives
                      `mpp::tensor_ops::matmul2d` primitive with
                      `execution_simdgroups<N_MM_SIMD_GROUP_X *
                      N_MM_SIMD_GROUP_Y>`. The A tile is staged in
                      shmem (4096 bytes, smem_a = NRA*N_MM_NK_TOTAL*
                      sizeof(half)); B is read directly from device
                      via `auto tB = tensor(ptrB, ...)`. The output
                      is staged via `cT` (a cooperative tensor) and
                      stored via `cT.store(tD.slice(ra, rb))` which
                      handles bounds checking internally.
                      The legacy path uses the manual 4-ma/2-mb/8-mc
                      simdgroup loop (Finding F02) with both sa and
                      sb in shmem (6144 or 8192 bytes).
                      The has_tensor flag is gated on
                      MTLGPUFamilyMetal4_GGML (ARTX15 §9.3) and
                      disabled by chip name for pre-M5 hardware
                      (ARTX15-F10).
Evidence:             ggml-metal.metal:9964-10086 (tensor path with
                      mpp::tensor_ops::matmul2d), 10088-10303 (legacy
                      path), 9985-10015 (tensor tA/tB setup),
                      10018-10022 (matmul2d descriptor).
Architectural Impact: The tensor path is the future — it offloads the
                      matmul to Apple's primitive, which may use
                      AMX-equivalent hardware. The legacy path is the
                      present fallback for pre-Metal4 devices. Both
                      share the same `kernel_mul_mm` template
                      signature.
Correctness Impact:   None. Both paths produce the same result (modulo
                      ULP-level reassociation differences).
Optimization Type:    Hardware-specific matmul primitive.
GwenLand Target:      glmetal
Recommendation:       MONITOR. Keep both paths in glmetal. Prefer the
                      tensor path when available; fall back to legacy
                      for pre-Metal4. Revisit when M5/A19 benchmarks
                      show a clear win.
Priority:             Medium
Difficulty:           L
Dependencies:         R2, R11
Confidence:           High
```

### Finding ARTX16-F14

```
Finding ID:           ARTX16-F14
Category:             LAYOUT_SUBOPTIMAL
Engine:               Metal
Component:            kernel_mul_mv_t_t_disp, kernel_mul_mv_t_t_4_disp
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_mul_mv_t_t_disp, kernel_mul_mv_t_t_4_disp
Lines:                4320-4336, 4444-4460
Summary:              The dispatcher's switch on args.nr0 has only
                      `case 2` enabled; cases 1, 3, 4 are commented out.
                      The host always sets nr0=2, so the dead cases
                      never trigger.
Observation:          Both dispatchers (`kernel_mul_mv_t_t_disp` for the
                      scalar variant and `kernel_mul_mv_t_t_4_disp` for
                      the float4 variant) have:
                        switch (args.nr0) {
                          //case 1: ... impl<1>(...); break;
                            case 2: ... impl<2>(...); break;
                          //case 3: ... impl<3>(...); break;
                          //case 4: ... impl<4>(...); break;
                        }
                      No default case. The host
                      (ggml-metal-device.cpp:797) always sets `nr0 = 2`
                      for F32/F16/BF16. If the host ever set nr0 to 1,
                      3, or 4, the dispatcher would silently fall
                      through and produce no output (the dst tensor
                      would be uninitialized).
                      The commented-out cases suggest the maintainers
                      tried nr0=1,3,4 and found nr0=2 to be the winner
                      — but there is no comment documenting this.
Evidence:             ggml-metal.metal:4330-4335 (t_t_disp),
                      4454-4459 (t_t_4_disp), ggml-metal-device.cpp:797
                      (host always sets nr0=2).
Architectural Impact: Dead code with a silent-failure trap. A
                      maintainer who uncomments the cases without
                      fixing the host would get correct output; a
                      maintainer who changes the host to send nr0=3
                      without uncommenting would get silent zero
                      output.
Correctness Impact:   None today (host and kernel are in sync).
                      Latent silent-failure risk if host and kernel
                      drift.
Optimization Type:    None (dead code).
GwenLand Target:      glmetal
Recommendation:       ADAPT. Either remove the dead cases (and document
                      that nr0=2 is the only supported value) or enable
                      them with an autotuner that picks nr0 per shape.
                      Add a `default: GGML_ASSERT(false)` to catch
                      host/kernel drift.
Priority:             Medium
Difficulty:           XS
Dependencies:         R3
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the `mul_mm` legacy path is barrier-bound or
  compute-bound on current Apple Silicon. The K loop has one full
  `threadgroup_barrier` per `loop_k` step plus 3 `simdgroup_barrier
  (mem_none)` hints per inner iteration. If barrier-bound, R10
  (double-buffering) would help. Requires profiling with Metal
  System Trace.

* **U2**. Whether the `has_tensor` path's
  `mpp::tensor_ops::matmul2d` internally double-buffers or uses
  AMX-equivalent hardware. Static analysis cannot determine this.
  Requires M5/A19 hardware (currently unavailable) or Apple internal
  documentation.

* **U3**. Whether the `simdgroup_barrier(mem_flags::mem_none)` hints
  actually affect codegen. The Metal compiler may ignore them. If
  ignored, removing them is safe; if honored, removing them may
  increase register pressure. Requires `metal -dM -E` disassembly
  comparison.

* **U4**. The real-world precision impact of `mul_mm`'s f16 A-tile
  storage when src0 is F32. For typical LLM weights (post-quantization
  scale factors in [-1, 1]), f16 storage is lossless. For attention
  scores (which can be large), f16 storage may cause measurable error.
  Requires differential testing against the has_tensor path (which
  uses f32 A-tile storage).

* **U5**. Whether the Q4_K and Q5_K GEMV kernels' single-simdgroup
  design (NSG=2 but no inter-sg reduction, each sg handles disjoint
  rows) is optimal. An alternative design with NSG=1 and the same
  total rows-per-threadgroup would use fewer simdgroups but each
  would do more work. The current design uses 2 simdgroups × 1-2 rows
  = 2-4 rows per threadgroup. Requires per-shape benchmarking.

* **U6**. Whether the `kernel_mul_mv_t_t_short` threshold (`ne00 < 32`)
  is optimal. The regular `mul_mv_t_t` path requires `ne00 >= 32` for
  its 32-element blocking. The short path uses scalar dot product with
  no simdgroup cooperation. For `ne00 ∈ [32, 64]` the regular path
  uses 2 simdgroups (nsg = min(4, (ne00+127)/128) = 1), which may be
  overkill for very short rows. Requires microbenchmarking.

* **U7**. Whether the flash attention `nsg` heuristic (`ne00 >= 512 ?
  8 : 4`) is optimal for any head size other than 128 and 512. The
  commented-out autotuner code suggests the maintainers intended a
  more nuanced choice. Requires per-(DK, DV, ne11) benchmarking.

* **U8**. The actual shmem footprint of `kernel_flash_attn_ext` for
  the largest supported head size (DK=DV=576, nsg=8). The FATTN_SMEM
  formula gives `8*(576 + 2*pad(576,64) + 2*128) + 16*32*8 = 8*(576
  + 1152 + 256) + 4096 = 15872 + 4096 = 19968` bytes, padded to 16
  bytes = 19968 bytes. Within the 32 KiB limit but tight. Requires
  verification on the actual device.

* **U9**. Whether the `kernel_mul_mm_id_map0` kernel (MoE expert
  routing pre-pass) is correct when `ne20` (n_expert_used) > 16. The
  shmem allocation is `args.ne21 * ne20 * sizeof(uint16_t)` bytes
  (line 10326: `sids = shmem + tpitg*ne20`). For ne20=16 and
  ne21=512, that's 16 KiB — within budget. For ne20=64 and ne21=512,
  that's 64 KiB — exceeds the 32 KiB limit. The largest instantiated
  ne20 is 22 (line 10371), so 22*512*2 = 22528 bytes — within budget.
  But there is no runtime check that the allocation fits.

---

## 19. References

| Reference | File                                                | Function / Symbol                                | Lines                |
| --------- | --------------------------------------------------- | ------------------------------------------------ | -------------------- |
| R01       | `ggml/src/ggml-metal/ggml-metal.metal`              | `dequantize_f32`, `_f16`, `_bf16`, `_q*`         | 91-1037              |
| R02       | `ggml/src/ggml-metal/ggml-metal.metal`              | `quantize_q4_0` .. `quantize_iq4_nl`             | 240-469              |
| R03       | `ggml/src/ggml-metal/ggml-metal.metal`              | `block_q_n_dot_y` (Q1_0, Q2_0 overloads)        | 3317-3481            |
| R04       | `ggml/src/ggml-metal/ggml-metal.metal`              | `helper_mv_reduce_and_write<NR0>`                | 3483-3523            |
| R05       | `ggml/src/ggml-metal/ggml-metal.metal`              | `mul_vec_q_n_f32_impl`                           | 3531-3617            |
| R06       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_q1_0_f32` .. `kernel_mul_mv_q8_0_f32` | 3687-3911       |
| R07       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_ext_q4_f32_impl`                  | 3916-4015            |
| R08       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_ext_q4x4_f32_impl`                | 4019-4122            |
| R09       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_t_t_impl`, `_disp`                | 4240-4336            |
| R10       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_t_t_4_impl`, `_disp`              | 4361-4460            |
| R11       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_t_t_short`                        | 4485-4538            |
| R12       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_rope_norm`, `_neox`, `_multi`, `_vision` | 4595-4877            |
| R13       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_soft_max`, `kernel_soft_max_4`           | 1950-2161            |
| R14       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_norm_fuse_impl`, `kernel_rms_norm_fuse_impl`, `kernel_l2_norm_impl`, `kernel_group_norm_f32` | 3014-3315 |
| R15       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_bin_fuse_impl`                           | 1265-1415            |
| R16       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_impl`                     | 6323-6905            |
| R17       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext`, instantiations          | 6991-7786            |
| R18       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_vec`, `_reduce`           | 7218-7829            |
| R19       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_cpy_t_t`, `kernel_cpy_f32_q`, `kernel_cpy_q_f32` | 7830-7981     |
| R20       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_concat`                                  | 7982-8015            |
| R21       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_q2_K_f32_impl` .. `kernel_mul_mv_iq4_xs_f32_impl` | 8029-9733 |
| R22       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_get_rows_q`, `kernel_get_rows_f`         | 9748-9814            |
| R23       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_set_rows_q32`, `kernel_set_rows_f`       | 9841-9934            |
| R24       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mm` (tensor path)                    | 9964-10086           |
| R25       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mm` (legacy path)                    | 10088-10303          |
| R26       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mm` instantiations (f32, f16, bf16, quants) | 10682-10735   |
| R27       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mm_id_map0`, `kernel_mul_mm_id`      | 10308-10676          |
| R28       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_mul_mv_id`                               | 10849-10949          |
| R29       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_opt_step_adamw_f32`, `kernel_opt_step_sgd_f32` | 11100-11146   |
| R30       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_count_equal`, `kernel_memset`            | 11149-11163, 11149-11158 |
| R31       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `SZ_SIMDGROUP`, `N_MM_*` matmul tile constants  | 8-15                 |
| R32       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `N_R0_*`, `N_SG_*` per-dtype GEMV constants     | 24-88                |
| R33       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `FC_*` function-constant offsets                 | 91-106               |
| R34       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `ggml_metal_kargs_*` structs                     | 158-1222             |
| R35       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_mul_mm` (smem sizing) | 704-764        |
| R36       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_mul_mv` (per-dtype smem) | 766-955      |
| R37       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_flash_attn_ext` | 1395-1458            |
| R38       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `maxThreadgroupMemoryLength` query               | 851                  |
| R39       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `FATTN_SMEM(nsg)` macro                          | 2817, 2957           |
| R40       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `nsg = ne00 >= 512 ? 8 : 4` heuristic            | 2835                 |
| R41       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_encoder_set_threadgroup_memory_size` call sites | 1001, 1075, 1110, 1383, 1554, 1727, 2216, 2260, 2367, 2415, 2465, 2887, 3036, 3050, 3344, 3395, 3531, 4136, 4457, 4522, 4858 |
| R42       | `ggml/src/ggml-metal/ggml-metal-common.h`           | `ggml_mem_ranges` overlap tracker (host-side, referenced for completeness) | 14-48 |
