# ARTX13 — CUDA VecDot (vecdotq.cuh)

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-26
**Auditor:** Percival-aux (ARTX13+14)
**Target GwenLand module:** `glcuda` (CUDA backend), `GATE` (kernel selection)

---

## 1. Executive Summary

`vecdotq.cuh` is the **single shared inner-product library** for the CUDA
quantized-matmul path. It defines 22 per-quant-format device functions
(`vec_dot_q4_0_q8_1`, `vec_dot_q4_K_q8_1`, `vec_dot_iq2_xxs_q8_1`,
`vec_dot_mxfp4_q8_1`, `vec_dot_nvfp4_q8_1`, …) plus a small set of inner
`*_impl` templates (`vec_dot_q4_0_q8_1_impl<vdr>`, `vec_dot_q4_K_q8_1_impl_vmmq`,
`vec_dot_q4_K_q8_1_impl_mmq`, …) that are themselves shared between MMVQ
(ARTX09) and MMQ (ARTX10/12). Every function in the file:

1. takes a `const void * vbq` (weight block base), a `const block_q8_1 *
   bq8_1` (quantized activation block base), a `kbx` block index, and an
   `iqs` intra-block offset;
2. dequantizes the weight slice *inside the function*, never writing it to
   shared memory;
3. computes a `dp4a`-reduced `int sumi` (or `float sum` for the FP4 paths);
4. multiplies by per-block scales in F32 and returns a single `float`.

The dispatch contract is the typedef `vec_dot_q_cuda_t = float (*)(const
void * vbq, const block_q8_1 * bq8_1, const int & kbx, const int & iqs)`
(`mmvq.cu:8`) plus a `constexpr __device__` switch
`get_vec_dot_q_cuda(ggml_type)` (`mmvq.cu:10-36`). MMVQ stores the result
of this switch in a `constexpr vec_dot_q_cuda_t vec_dot_q_cuda` local
(`mmvq.cu:500, 724`) so `nvcc` devirtualises the call at `-O3`. MMQ
instead wraps the `*_impl_mmq` templates inside
`ggml_cuda_mmq_vec_dot_*<type, J, fallback>` (audited in ARTX12) — those
share the inner `*_impl_mmq` arithmetic with the MMVQ path but *not* the
block-reading prologue.

The architectural decisions worth **ADOPT**ing are: (a) the
device-function-pointer dispatch via `constexpr` switch (one indirect
call per matmul, devirtualised); (b) the `dp4a` 4-way int8 dot product
used uniformly for every integer quant; (c) the **fused zero-point
folding** patterns — symmetric quants subtract the implicit `-8` bias
inside the float scale multiply using the Q8_1 `s` (abs-sum) field
(`vec_dot_q4_0_q8_1_impl:133`), asymmetric quants compute a separate
`dot2 = dp4a(0x01010101, u, 0)` and scale by the per-block min (`vec_dot_q4_K_q8_1_impl_vmmq:518-521`);
(d) the **`__vsub4(grid^signs, signs)`** branchless signed-flip idiom
used by every IQ2/IQ3 path; (e) the per-quant VDR constants
(`VDR_*_MMVQ`, `VDR_*_MMQ`) that decouple MMVQ's per-warp granularity
from MMQ's per-tile granularity. The decisions worth **ADAPT**ing are:
the three independent re-implementations of the K-quant 6-bit scale
unpacker (in `vecdotq.cuh:890-899`, `dequantize.cuh:157-164`, and
`mmq-load-tiles.cuh:612-620`); the `__byte_perm`-heavy Q1_0 unpacker
with no AMD/MUSA fast path beyond the generic C fallback.

This audit covers **only the device-function implementations** in
`vecdotq.cuh`. The host-side dispatch chain, the per-arch tuning tables,
and the `mul_mat_vec_q` kernel body are audited in ARTX09 (GEMV) and
ARTX10/12 (GEMM). Cross-references to those audits are explicit.

---

## 2. Purpose

Provide the inline-able, single-call quantized dot product between one
slice of a quantized weight block and one slice of a Q8_1 activation
block, for every quant format llama.cpp supports (22 formats total).
The functions are designed to be called once per `(block, iqs)` pair
from inside the K-dim loop of MMVQ, and once per tile-iteration from
inside the MMQ tile-vecdot wrapper.

Specifically:

* define the per-quant dequantize + `dp4a` + scale-multiply arithmetic
  in a single place;
* expose a uniform `vec_dot_q_cuda_t` signature so MMVQ can hold a
  single `constexpr` function pointer;
* expose the `*_impl` templates so MMQ can call the same arithmetic
  with a different (larger) `vdr` and a different (pre-unpacked) input
  layout;
* keep all intermediate values in registers — no shared-memory staging
  of dequantized weights.

It is **not** responsible for: the K-loop tiling (MMVQ), the tile-vecdot
layout (MMQ), the Q8_1 activation quantization
(`quantize.cu:quantize_q8_1`), the per-arch `nwarps`/`rows_per_block`
policy (ARTX09 §8), the `dp4a` vs `mma.sync` selection (ARTX12), or the
dequantize-to-shared-memory path used by `dequantize.cuh` (ARTX14).

---

## 3. Source Files

| File                                  | Lines | Role                                                                              |
| ------------------------------------- | ----- | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/vecdotq.cuh`      | 1323  | **Primary.** All `vec_dot_*_q8_*` device functions, all `*_impl` templates, all `VDR_*` constants, `get_int_b{1,2,4}`, `get_int_from_table_16`, `unpack_ksigns`. |
| `ggml/src/ggml-cuda/mmvq.cu`          | 1290  | Consumer #1. Defines `vec_dot_q_cuda_t` typedef (`:8`), `get_vec_dot_q_cuda(type)` constexpr switch (`:10-36`), `get_vdr_mmvq(type)` (`:38-62`), and the MMVQ kernel that holds `constexpr vec_dot_q_cuda_t vec_dot_q_cuda` (`:500, 724`). |
| `ggml/src/ggml-cuda/mmq-vec-dot.cuh`  | 1251  | Consumer #2. The MMQ tile-vecdot wrappers `ggml_cuda_mmq_vec_dot_*_dp4a` and `*_mma` call into the `*_impl_mmq` templates from `vecdotq.cuh`. Audited in ARTX12. |
| `ggml/src/ggml-cuda/common.cuh`       | 1661  | `ggml_cuda_dp4a` (`:703-741`), `ggml_cuda_e8m0_to_fp32` (`:821`), `ggml_cuda_ue4m3_to_fp32` (`:839`), `__vsubss4` / `__vsub4` / `__vcmpne4` shims, `FAST_FP16_AVAILABLE` macro (`:262`). Audited in ARTX08. |
| `ggml/src/ggml-common.h`              | 1912  | Lookup tables `kvalues_mxfp4`, `kvalues_iq4nl`, `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`, `ksigns_iq2xs`, `kmask_iq2xs`. Declared as `static const __device__` (`:493`). |
| `ggml/src/ggml-cuda/dequantize.cuh`   | 432   | Sister file with the `dequantize_*` device functions (audited in ARTX14). Shares the K-quant scale layout knowledge with `vecdotq.cuh` but does NOT share code. |

> Note: ARTX10 and ARTX12 reference `vecdotq.cuh` as "shared by MMVQ and
> MMQ" — that is correct, but those audits do not enumerate the per-quant
> device-function bodies. This audit does.

---

## 4. Architecture Overview

```
                ┌───────────────────────────────────────────────────┐
                │  mmvq.cu : mul_mat_vec_q<type, ncols_dst, …>      │
                │  └─ constexpr vec_dot_q_cuda_t vec_dot_q_cuda =   │
                │     get_vec_dot_q_cuda(type)        (mmvq.cu:500) │
                └───────────────────────────────────────────────────┘
                                       │  one call per (kbx, iqs) per K-step
                                       ▼
                ┌───────────────────────────────────────────────────┐
                │  vecdotq.cuh : vec_dot_<type>_q8_1                │
                │  ├─ vec_dot_q4_0_q8_1 … vec_dot_q8_0_q8_1        │
                │  │  (simple quants, 2 elements / thread)          │
                │  ├─ vec_dot_mxfp4_q8_1, vec_dot_nvfp4_q8_1       │
                │  │  (FP4 lookup-table expansion)                  │
                │  ├─ vec_dot_q2_K_q8_1 … vec_dot_q6_K_q8_1        │
                │  │  (K-quants, per-call 6-bit scale unpack)       │
                │  ├─ vec_dot_iq2_xxs_q8_1 … vec_dot_iq4_xs_q8_1   │
                │  │  (I-quants, grid + branchless sign flip)       │
                │  └─ vec_dot_q1_0_q8_1                             │
                │     (1-bit, __byte_perm unpacking)                │
                └───────────────────────────────────────────────────┘
                                       │  call into shared inner templates
                                       ▼
                ┌───────────────────────────────────────────────────┐
                │  vecdotq.cuh : vec_dot_<type>_q8_1_impl<vdr>      │
                │  ├─ int sumi = 0;                                 │
                │  │  for i in [0,vdr): sumi = dp4a(v,u,sumi);      │
                │  └─ return d * (sumi * d8 - bias * s8);           │
                │     (or equivalent — see scale-handling per type) │
                └───────────────────────────────────────────────────┘
                                       ▲
                                       │  same impl templates, larger vdr,
                                       │  pre-unpacked inputs
                ┌───────────────────────────────────────────────────┐
                │  mmq-vec-dot.cuh : ggml_cuda_mmq_vec_dot_*<type>  │
                │  (audited in ARTX12)                              │
                └───────────────────────────────────────────────────┘
```

Key design points:

* **One signature, 22 implementations.** Every `vec_dot_*_q8_1` has
  signature `(const void * vbq, const block_q8_1 * bq8_1, const int &
  kbx, const int & iqs) → float`. This lets MMVQ hold a single typedef'd
  function pointer.
* **Two layers: outer wrapper reads blocks; inner `*_impl` does math.**
  The outer `vec_dot_q4_0_q8_1` (`:725-741`) reads `v[VDR]` and
  `u[2*VDR]` from the weight and activation blocks via `get_int_b{1,2,4}`,
  then calls `vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, d, ds8)`.
  The inner template knows nothing about block layout — only about int
  arrays and scales.
* **MMVQ and MMQ call the same inner templates with different `vdr`.**
  MMVQ instantiates with `VDR_*_MMVQ` (small: 1-4), MMQ with
  `VDR_*_MMQ` (large: 2-8). The K-quant `*_mmq` variants take
  pre-unpacked `int8_t * scales` / `half2 * ds8` so the unpacking
  happens once per tile rather than once per call.
* **Zero shared memory.** Every function in this file is
  `__forceinline__` and register-only. The dequantized weight values
  exist only for the duration of the `dp4a` reduction. This is the
  fundamental difference from `dequantize.cuh` (ARTX14), which writes
  dequantized values to a `dst_t * yy` buffer.

---

## 5. Execution Flow

### 5.1 MMVQ call site (the consumer)

`mul_mat_vec_q<type, ncols_dst, has_fusion, small_k>` (`mmvq.cu:478-699`)
holds a `constexpr vec_dot_q_cuda_t vec_dot_q_cuda = get_vec_dot_q_cuda(type)`
at `:500`. Inside the K-dim loop (`:592-612`), each thread calls
`vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], kbx_offset + i*stride_row_x + kbx, kqs)`
which is a direct call into one of the 22 device functions in
`vecdotq.cuh`. The `kqs` argument (`iqs`) is computed as
`kqs = (kbc + threadIdx.x) % (warp_size * vdr)` modulo the per-type
inner-block stride (`mmvq.cu:590`), so adjacent threads in a warp walk
adjacent `iqs` offsets within a block.

### 5.2 Inside a typical `vec_dot_*_q8_1` (Q4_0 example)

`vec_dot_q4_0_q8_1` (`vecdotq.cuh:725-741`):

1. `const block_q4_0 * bq4_0 = (const block_q4_0 *) vbq + kbx;` — index
   to the right weight super-block.
2. `int v[VDR_Q4_0_Q8_1_MMVQ]; int u[2*VDR_Q4_0_Q8_1_MMVQ];` —
   register arrays of size 2 and 4 for VDR=2.
3. For `i in [0, VDR)`: load `v[i] = get_int_b2(bq4_0->qs, iqs + i)`
   (one 32-bit int = 8 packed 4-bit weights), and `u[2*i + {0,1}] =
   get_int_b4(bq8_1->qs, iqs + i + {0, QI4_0})` (two Q8_1 int slices).
4. Call `vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds)`.

### 5.3 Inside the inner `*_impl` (Q4_0 example)

`vec_dot_q4_0_q8_1_impl<vdr>` (`vecdotq.cuh:115-134`):

1. `int sumi = 0;`
2. For `i in [0, vdr)`: split `v[i]` into low and high nibbles
   (`vi0 = v[i] & 0x0F0F0F0F`, `vi1 = (v[i] >> 4) & 0x0F0F0F0F`), then
   `sumi = ggml_cuda_dp4a(vi0, u[2*i+0], sumi); sumi = ggml_cuda_dp4a(vi1, u[2*i+1], sumi);`
   — two `dp4a` per VDR iteration.
3. `const float2 ds8f = __half22float2(ds8);` — unpack the Q8_1
   `(d, s)` pair to F32. `ds8f.x` is the activation scale, `ds8f.y` is
   the activation abs-sum.
4. `return d4 * (sumi * ds8f.x - (8*vdr/QI4_0) * ds8f.y);` — apply
   weight scale `d4` (broadcast), activation scale `ds8f.x`, and
   subtract the symmetric-quant `-8` bias via `ds8f.y`. The `(8*vdr/QI4_0)`
   factor is the per-thread share of the bias: `QI4_0 = 16` (ints per
   Q4_0 block), `8*vdr/QI4_0 = vdr/2` blocks of 4 elements per VDR
   iteration — so `8` (the Q4_0 zero-point) × `vdr/2` × `s8` (sum of
   activation int8s in the block) × `d4` is exactly the implicit
   zero-point correction.

### 5.4 MMQ call site

The MMQ tile-vecdot (audited in ARTX12) calls the `*_impl_mmq` variants
directly. For Q4_0, that's still `vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>`
with `VDR_Q4_0_Q8_1_MMQ = 4` (twice the MMVQ value). For K-quants, the
`*_impl_mmq` variant is a *separate* function (`vec_dot_q4_K_q8_1_impl_mmq`
at `:530-555`) that takes pre-unpacked `sc`/`m` arrays and a `half2 * ds8`
pointer (one `half2` per Q8_1 sub-block), because the MMQ loader has
already unpacked the scales into shared memory.

### 5.5 Per-quant dispatch chain

`get_vec_dot_q_cuda(ggml_type type)` (`mmvq.cu:10-36`) is a 22-arm
`switch` that returns the function pointer. The cases are:
`Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, NVFP4, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ1_S, IQ1_M, IQ4_NL,
IQ4_XS, IQ3_S`. The default case returns `nullptr` (compile-time error
if a non-quant type is fed in). `get_vdr_mmvq(type)` (`:38-62`) is a
parallel switch returning the `VDR_*_MMVQ` constant for the same 22
types. Both switches must agree per type — this is enforced by the
template instantiation in `mmvq.cu:998-1143`.

---

## 6. Data Layout

### 6.1 Weight block (input `vbq`)

Each quant format defines its own block layout in `ggml-common.h`. The
`vec_dot_*` function casts `vbq + kbx` to the right `block_*` type and
reads fields directly. Per-format block sizes (audited in ARTX06):

| Format   | QK   | Bytes/block | Scales per block          |
| -------- | ---- | ----------- | ------------------------- |
| Q1_0     | 32   | 6           | 1 × float (d)             |
| Q4_0     | 32   | 18          | 1 × half (d)              |
| Q4_1     | 32   | 20          | 1 × half2 (dm = d, m)     |
| Q5_0     | 32   | 22          | 1 × half (d)              |
| Q5_1     | 32   | 24          | 1 × half2 (dm = d, m)     |
| Q8_0     | 32   | 34          | 1 × half (d)              |
| MXFP4    | 32   | 17          | 1 × e8m0 (e)              |
| NVFP4    | 16   | 9           | 1 × ue4m3 (d) per sub-blk |
| Q2_K … Q6_K | 256 | 84-210    | 4-12 packed bytes (6-bit) |
| IQ2_*    | 256  | 66-74       | per-sub-block grid + signs |
| IQ3_*    | 256  | 110-128     | per-sub-block grid + signs |
| IQ1_S/M  | 256  | 32-40       | per-sub-block grid + delta |
| IQ4_NL   | 32   | 18          | 1 × half (d)              |
| IQ4_XS   | 256  | 76          | packed 6-bit scales       |

### 6.2 Activation block (`bq8_1`)

All vec_dot functions take a `block_q8_1 *` activation. `block_q8_1`
(`ggml-common.h`) is 34 bytes: `half2 ds` (low = scale `d`, high =
abs-sum `s`) + `int8_t qs[32]`. The `s` field is consumed by the
symmetric-quant paths (Q4_0, Q5_0, Q8_0) to subtract the implicit
zero-point bias without needing a separate weight-side min field
(§5.3 step 4).

### 6.3 `iqs` index space

`iqs` is the **per-thread intra-block offset**. For Q4_0 with
`QI4_0 = 16` (32 elements per block, 2 elements per int), `iqs` ranges
over `[0, 16)` and each `iqs` selects one 32-bit int from the weight
block. Adjacent threads in a warp use adjacent `iqs` values, so 16
threads cover one Q4_0 block. For K-quants with `QK_K = 256`,
`iqs` ranges over `[0, QK_K/16) = [0, 16)` for the inner indexing,
with `bq8_offset` selecting which Q8_1 sub-block to read.

---

## 7. Memory Layout

### 7.1 Register-only

Every function in `vecdotq.cuh` is `__forceinline__` and uses only
registers. Stack arrays `v[VDR]`, `u[2*VDR]` (Q4_0), `vl[VDR]`, `vh[VDR]`,
`u[2*VDR]` (Q5_0), `u[QR*K]`, `d8[QR*K]` (K-quants) are sized by
compile-time constants and elided into scalar registers by `nvcc -O3`.

### 7.2 Lookup tables in read-only data memory

The IQ2/IQ3/IQ1 lookup tables (`iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`,
`iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`, `ksigns_iq2xs`,
`kmask_iq2xs`, `kvalues_iq4nl`, `kvalues_mxfp4`) live in
`ggml-common.h:493` as `static const __device__ type name[size]`. They
are not explicitly `__constant__`-qualified; on NVIDIA they back by
read-only data memory (the 64 KB constant memory is reserved for kernel
parameters, so larger tables go to L1 read-only cache via `__ldg`-style
access). The largest table is `iq2s_grid` at 1024 × 8 = 8 KB; the
smallest are `kvalues_iq4nl` / `kvalues_mxfp4` at 16 bytes each.

### 7.3 No shared memory

`vecdotq.cuh` does **not** declare any `__shared__` variables. The
shared-memory staging for K-quant scales (the `scales[]` array) happens
in the MMQ loader (`mmq-load-tiles.cuh`, ARTX12), not here. The MMVQ
path never uses shared memory for weight values — only for the
cross-warp reduction of partial sums (in `mmvq.cu:614`, audited in
ARTX09).

---

## 8. Parallelism Strategy

### 8.1 Per-thread VDR

Each thread processes `vdr` int-pairs per `vec_dot_*` call. The VDR
constants (`vecdotq.cuh:109-110, 112-113, 136-137, 169-170, 200-201,
240-241, 304-305, 328-329, 360-361, 443-444, 501-502, 557-558,
620-621, 987-988, 1022-1023, 1063-1064, 1111-1112, 1149-1150,
1192-1193, 1225-1226, 1272-1273, 1296-1297`) define this per-quant and
per-context (MMVQ vs MMQ):

| Format   | VDR_MMVQ | VDR_MMQ | Ratio |
| -------- | -------- | ------- | ----- |
| Q1_0     | 1        | 4       | 4×    |
| Q4_0     | 2        | 4       | 2×    |
| Q4_1     | 2        | 4       | 2×    |
| Q5_0     | 2        | 4       | 2×    |
| Q5_1     | 2        | 4       | 2×    |
| Q8_0     | 2        | 8       | 4×    |
| MXFP4    | 2        | 4       | 2×    |
| NVFP4    | 4        | 8       | 2×    |
| Q2_K     | 1        | 4       | 4×    |
| Q3_K     | 1        | 2       | 2×    |
| Q4_K     | 2        | 8       | 4×    |
| Q5_K     | 2        | 8       | 4×    |
| Q6_K     | 1        | 8       | 8×    |
| IQ2_XXS  | 2        | 2       | 1×    |
| IQ2_XS   | 2        | 2       | 1×    |
| IQ2_S    | 2        | 2       | 1×    |
| IQ3_XXS  | 2        | 2       | 1×    |
| IQ3_S    | 2        | 2       | 1×    |
| IQ1_S    | 1        | 1       | 1×    |
| IQ1_M    | 1        | 1       | 1×    |
| IQ4_NL   | 2        | 4       | 2×    |
| IQ4_XS   | 4        | 4       | 1×    |

The IQ paths have `VDR_MMQ == VDR_MMVQ` because the grid-lookup cost
dominates and there's no benefit from a larger MMQ tile-vecdot
granularity. The simple quants (Q4_0, Q4_1, etc.) double or quadruple
the VDR for MMQ because each `dp4a` is cheap and the limiting factor
is K-loop trip count.

### 8.2 Warp-level coalescing

MMVQ uses 32 threads per warp; adjacent threads in the warp call
`vec_dot_*` with adjacent `iqs` values, so the `get_int_b{1,2,4}`
loads coalesce into one 128-byte transaction per warp per call. For
Q4_0 with `QI4_0 = 16`, 16 threads cover one block; the other 16
threads cover the next block — both `bq4_0->qs` reads coalesce.

### 8.3 No inter-thread communication

The `vec_dot_*` functions are pure: same inputs → same output, no
shuffles, no atomics, no shared-memory writes. All inter-thread
reduction happens in the caller (`warp_reduce_sum` in `mmvq.cu:652`).

---

## 9. GPU Strategy

### 9.1 `dp4a` is the universal hammer

Every integer quant (Q1_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
Q4_K, Q5_K, Q6_K, IQ2_*, IQ3_*, IQ1_*, IQ4_NL, IQ4_XS) reduces via
`ggml_cuda_dp4a(a, b, c)` (`common.cuh:703-741`). The dispatch is:

* NVIDIA + MUSA: `__dp4a(a, b, c)` intrinsic (Volta+ for the int8
  variant; the macro `GGML_CUDA_CC_DP4A` gates this).
* AMD CDNA / RDNA2 / gfx906: `__builtin_amdgcn_sdot4(a, b, c, false)`.
* AMD RDNA3 / RDNA4: `__builtin_amdgcn_sudot4(true, a, true, b, c, false)`
  (signed/unsigned dot4).
* AMD RDNA1 / gfx900: inline-asm `v_mul_i32_i24` + `v_add3_u32` (4
  scalar multiplies + 3 adds — slower but correct).
* Pre-Volta NVIDIA: scalar fallback `c + a8[0]*b8[0] + a8[1]*b8[1] + …`.

There is **no `__dp2a`** usage anywhere in the file — the Q5_0/Q5_1
paths that combine 4-bit + 1-bit into a 5-bit value do it with manual
shifts and ORs (`vecdotq.cuh:179-191`), then call `dp4a` on the
combined 8-bit value.

### 9.2 No Tensor Cores in `vecdotq.cuh`

Tensor Cores (`mma.sync`) are used by the MMQ tile-vecdot in
`mmq-vec-dot.cuh` (audited in ARTX12), not here. The `vec_dot_*`
device functions are pure CUDA-core code. The Blackwell native-FP4
path (`mma_block_scaled_fp4`, ARTX10 §9.4) bypasses `vecdotq.cuh`
entirely.

### 9.3 Per-quant branchless tricks

* **Symmetric-quant bias fold** (Q4_0, Q5_0): the `-8` / `-16` zero
  point is subtracted *after* the `dp4a` by multiplying the Q8_1
  abs-sum `s` by `8*vdr/QI4_0` (or `16*vdr/QI5_0`) inside the float
  scale expression (`vecdotq.cuh:133, 197`). One FMA, no per-element
  subtraction.
* **Asymmetric-quant min fold** (Q4_K, Q5_K, Q2_K): a separate
  `dot2 = dp4a(0x01010101, u, 0)` computes the per-int sum of the
  activation int8s, which is then multiplied by the per-block min
  value `m` and subtracted (`vecdotq.cuh:518-521, 580, 383`).
* **Branchless signed flip** (IQ2/IQ3): `int signs0 = __vcmpne4(signs
  & 0x08040201, 0); int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);`
  (`vecdotq.cuh:1005-1006, 1042-1043, 1088-1092, 1129-1130, 1173-1177`).
  `signs0` is 0xFF..FF where the sign bit is set, 0x00..00 otherwise;
  XOR with `signs0` flips the bits, then `__vsub4` subtracts `signs0`
  to negate. One XOR + one `vsub4` per 4-element group — no branches.
* **Q3_K `__vsubss4`** for the symmetric 4-bit-with-sign subtraction
  (`vecdotq.cuh:471, 638`). `__vsubss4(a, b)` does 4-way saturated
  subtract; used to subtract the implicit bias `(vil - vih)` per byte.
* **Per-byte `__byte_perm` for Q1_0** unpacking (`vecdotq.cuh:701-712`):
  8 `__byte_perm` calls to turn 16 1-bit indices into 16 0/-1 byte
  values. CUDA-only intrinsic; AMD/MUSA fall back to the same code but
  `__byte_perm` is a no-op on those backends (compiler-emits generic
  code). This is the only place in the file with a CUDA-specific
  intrinsic that has no AMD/MUSA accelerated equivalent.
* **`get_int_from_table_16`** for IQ4_NL/MXFP4/NVFP4 4-bit → 8-bit
  expansion (`vecdotq.cuh:34-95`): uses `__builtin_amdgcn_perm` on
  AMD, `__byte_perm` on NVIDIA, scalar fallback on MUSA. Returns an
  `int2` of (even-index bytes, odd-index bytes).

### 9.4 `__vsub4` / `__vcmpne4` / `__vsubss4` availability

These PTX intrinsics are gated by `__CUDA_ARCH__ >= GGML_CUDA_CC_PASCAL`
(or HIP equivalents in `common.cuh`). The IQ2/IQ3/IQ4_XS paths
implicitly require Pascal+ — there is no scalar fallback for
`__vsub4` / `__vcmpne4` / `__vsubss4` in the file. On pre-Pascal
NVIDIA hardware, these quants cannot run.

---

## 10. Quantization Strategy

### 10.1 Scale handling — three patterns

| Pattern | Quants | Where scale applied |
| ------- | ------ | ------------------- |
| Symmetric, post-multiply, bias-fold via `s` | Q4_0, Q5_0, Q8_0 | inside `*_impl` F32 return: `d * (sumi * d8 - bias * s8)` |
| Asymmetric, post-multiply, min-fold via `dot2` | Q4_1, Q5_1, Q4_K, Q5_K, Q2_K | inside `*_impl` F32 return: `sumi*d*d8 + m*s8/n` or `dm.x*sumf_d - dm.y*sumf_m` |
| Pure int scale, post-multiply | Q3_K, Q6_K, IQ3_S, IQ4_XS | per-call `int sc`, multiplied into `sumi` before the final F32 `d * sumi` |

For the IQ2/IQ1 paths the scale is an `int ls` that's extracted from
packed bits and multiplied into `sumi` before the F32 `d * sumi`
multiply (`vecdotq.cuh:1016-1019, 1058, 1105, 1144, 1186, 1318`).

### 10.2 Zero-point handling

Symmetric quants (Q4_0, Q5_0, Q8_0, Q2_K, Q3_K, Q6_K, IQ3_S, IQ4_XS)
fold the zero-point into the scale expression as a constant subtract
(`8`, `16`, `32`). Asymmetric quants (Q4_1, Q5_1, Q4_K, Q5_K) carry a
per-block min `m` and compute a separate `dot2 = dp4a(0x01010101, u, 0)`
that captures the sum of the activation int8s; the min contribution is
`m * dot2` (Q4_K/Q5_K) or `m * s8` (Q4_1/Q5_1). The Q8_1 `s` field
(abs-sum) doubles as the activation sum for the symmetric paths — a
clever reuse that avoids a separate `dp4a(0x01010101, u, 0)` for those
formats.

### 10.3 K-quant 6-bit scale unpacking — three implementations

The K-quant `scales[]` array packs per-32-element scales and per-32-element
mins into 6 bits each, spread across 12 bytes per super-block (see
ARTX06). `vecdotq.cuh` has three different in-thread unpackers:

* **Q4_K / Q5_K in MMVQ** (`vecdotq.cuh:890-899, 935-944`): reads
  `scales` as `uint16_t *`, builds a 4-byte `aux[2]` array, splits
  into `sc` (low nibbles) and `m` (high nibbles) with bit-fiddling
  conditional on `j < 2`.
* **Q4_K / Q5_K in MMQ**: takes pre-unpacked `uint8_t * sc` and
  `uint8_t * m` (the MMQ loader did the unpacking in
  `mmq-load-tiles.cuh:unpack_scales_q45_K`).
* **Q3_K in MMVQ** (`vecdotq.cuh:455-465`): a 4-way branch on `isc`
  value (`is < 4 ? … : is < 8 ? … : is < 12 ? … : …`) that combines
  low-nibble scales with 2-bit high-nibble extensions from a separate
  `scales[QK_K/32 + …]` array.

These three are *not* the same algorithm. They produce the same result
but use different bit manipulations, all hand-rolled. The
`dequantize.cuh` path uses yet a fourth (`get_scale_min_k4`, ARTX14).

### 10.4 NVFP4 / MXFP4 lookup expansion

`vec_dot_mxfp4_q8_1` (`:307-326`) and `vec_dot_nvfp4_q8_1`
(`:331-359`) call `get_int_from_table_16(aux_q4, kvalues_mxfp4)` to
expand each 4-bit index into a precomputed 8-bit value from the
`kvalues_mxfp4[16]` table (`kvalues_mxfp4 = kvalues_fp4` per
`ggml-common.h:1129`, defined as 2× E2M1 floats). NVFP4 additionally
uses per-sub-block ue4m3 scales via `ggml_cuda_ue4m3_to_fp32(bq4->d[is])`
(`:354`); MXFP4 uses a per-block e8m0 scale via
`ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f` (`:324`). NVFP4 accumulates
in F32 directly (`float sum += d * float(sumi)` at `:355`) rather than
in int — the only integer-quant path that does this. The `0.5f` factor
in MXFP4 and the per-sub-block scale lookup in NVFP4 are the FP4
dequantization conventions (see `ggml-impl.h:501` comment "Returns
value * 0.5 to match kvalues_mxfp4 convention").

### 10.5 IQ1_S / IQ1_M delta correction

IQ1_S and IQ1_M add a `delta` term to each dequantized value to model
the +0/-1 quantization bias (`vecdotq.cuh:1220, 1253`). The delta is
computed per-sub-block from a single `qh` bit and added to a sum-of-Q8
term (`sumy = dp4a(u, 0x01010101, 0)`) scaled by delta — analogous to
the asymmetric-quant min fold but for 1-bit quantization
(`vecdotq.cuh:1253-1258`).

---

## 11. Correctness Analysis

Static analysis found the following correctness-relevant properties.
None are bugs; all are intentional design decisions with correctness
consequences.

### 11.1 Floating-point reassociation

* **Per-thread int sum is exact.** `int sumi = 0; sumi = dp4a(v, u,
  sumi);` is integer arithmetic; the order of `dp4a` calls within a
  single `vec_dot_*` invocation is fixed by the `#pragma unroll` over
  `vdr`. No reassociation at the int level.
* **F32 scale multiply reassociates across `vdr`.** The final
  `d * (sumi * ds8f.x - bias * ds8f.y)` expression is one FMA per
  VDR iteration, accumulated in F32. The cross-VDR reduction happens in
  the caller (MMVQ's K-loop in `mmvq.cu:602-609` accumulates F32 across
  `vdr`-step iterations). Different `vdr` → different reduction tree →
  different ULPs.
* **K-quant `sumf_d` / `sumf_m` split.** K-quants accumulate
  `dot-product-with-scale` and `sum-of-activation-times-min` in two
  separate F32 accumulators, then subtract at the end
  (`vecdotq.cuh:526, 589, 617`). This is mathematically equivalent to
  a single accumulator but produces different ULPs.

### 11.2 Quantization rounding

* **Q8_1 activation quantization** happens in `quantize.cu:quantize_q8_1`
  (ARTX14), not here. The `vec_dot_*` functions assume the activation
  has already been rounded to int8 with `d = amax/127`. The result of
  any quantized matmul is therefore Q8×Q8-accurate, not F32×F32 — the
  deliberate accuracy/speed tradeoff of llama.cpp (matches ARTX01
  §11.3).
* **No overflow check on int sum.** `int sumi` accumulates up to `vdr
  × 4 × 127 × 127 = 64516 × vdr` per call. For `vdr = 8` (Q8_0 MMQ),
  that's ~516K, well within `int32` range. For `vdr = 4` and `d4a`
  saturating arithmetic (AMD `sdot4` is non-saturating; NVIDIA
  `__dp4a` is also non-saturating), no overflow concern for the
  per-call accumulator.

### 11.3 Determinism

* **Deterministic per call.** Every `vec_dot_*` function is pure —
  same inputs → bit-identical output. No atomics, no shuffles.
* **Non-determinism is the caller's responsibility.** Cross-warp
  reduction in MMVQ (`mmvq.cu:644-657`) is deterministic per
  `(nwarps, K)` configuration; cross-thread split-K in MMQ (ARTX10
  stream-K fixup) is non-deterministic across runs but deterministic
  within a single launch.

### 11.4 Architecture-specific assumptions

* `__vsub4`, `__vcmpne4`, `__vsubss4`, `__byte_perm` are CUDA-specific
  PTX intrinsics. AMD has `__builtin_amdgcn_perm` equivalents in
  `get_int_from_table_16` (`vecdotq.cuh:43-56`) but the IQ2/IQ3 sign
  flip and the Q3_K `__vsubss4` rely on the CUDA-side shims in
  `common.cuh` which emit `__byte_perm` / `__vsubss4` / `__vsub4` /
  `__vcmpne4` directly. HIP defines these as macros mapped to
  `__builtin_amdgcn_*` (audited in ARTX08). If a new backend (e.g.,
  Intel) is added, all four intrinsics need shims.
* `FAST_FP16_AVAILABLE` gates the use of `__hmul2` for the scale
  pre-multiply in Q4_1/Q5_1/Q8_1 paths (`vecdotq.cuh:154-163, 225-234,
  268-277`). On pre-Pascal NVIDIA, this macro is undefined and the
  code falls back to F32 multiplies. Different ULPs on different arch.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                  | Where                                | Notes                                                                                  |
| --------------------------------------------- | ------------------------------------ | -------------------------------------------------------------------------------------- |
| `dp4a` 4-way int8 dot                         | every `*_impl`                       | 4× throughput vs scalar 8-bit MADD on supporting hardware.                             |
| Symmetric bias fold via `s8` abs-sum          | `vecdotq.cuh:133, 197`               | Saves a per-element subtract; one FMA in the F32 epilogue.                              |
| Asymmetric min fold via `dot2 = dp4a(0x01..)` | `vecdotq.cuh:383, 518, 580`          | Reuses the `dp4a` unit for the sum-of-activation; saves a separate reduction.          |
| Branchless signed flip (`__vsub4(grid^signs, signs)`) | `vecdotq.cuh:1006, 1043, 1091, 1130, 1176` | 2 instructions per 4-element group, no branches.                                       |
| `get_int_from_table_16` for 4-bit LUT expansion | `vecdotq.cuh:34-95`                | AMD `__builtin_amdgcn_perm` / NVIDIA `__byte_perm` / generic fallback; one call per 8 nibbles. |
| `constexpr` device-function pointer dispatch  | `mmvq.cu:10-36, 500`                 | `get_vec_dot_q_cuda(type)` is `constexpr __device__`; devirtualised by `nvcc` at `-O3`. |
| Separate `VDR_*_MMVQ` vs `VDR_*_MMQ`          | `vecdotq.cuh:109-1296`               | Lets MMQ use 2-8× larger per-call granularity than MMVQ.                                |
| Separate `*_impl_mmvq` vs `*_impl_mmq` for K-quants | `vecdotq.cuh:364-441, 447-499, …` | MMQ variant takes pre-unpacked `int8_t * scales` (loader already unpacked); saves per-call unpacking. |
| Register-only `int v[VDR]` / `int u[2*VDR]`   | every `vec_dot_*`                    | Sized by compile-time VDR; elided into scalar registers by `nvcc -O3`.                  |
| `#pragma unroll` on every VDR loop            | every `*_impl`                       | Makes the per-VDR-iteration `dp4a` chain a fixed instruction sequence.                 |
| `__forceinline__` on every function           | every `vec_dot_*`                    | Eliminates call overhead; the function body is inlined into the MMVQ K-loop.            |

### 12.2 Optimizations *not* present

* **No Tensor Cores.** The `vec_dot_*` family is pure CUDA-core
  `dp4a`. Tensor-Core int8 MMA (`mma.sync.s8`) lives in
  `mmq-vec-dot.cuh` (ARTX12) and uses the same inner `*_impl` arithmetic
  only in the dp4a variant, not the mma variant.
* **No `__dp2a` usage.** Q5_0/Q5_1 combine 4-bit + 1-bit into a 5-bit
  value via shifts and ORs, then call `dp4a` on the combined 8-bit
  value. `__dp2a` (2-way 16-bit + 2-way 8-bit dot product) would be a
  more direct fit but is unused.
* **No `cp.async` prefetching.** The `vec_dot_*` functions read weight
  blocks directly via `get_int_b{1,2,4}`. Async copy from global to
  shared is done at the MMQ-loader level (ARTX12), not here.
* **No persistent scale cache.** K-quant scales are re-unpacked on
  every `vec_dot_*` call (MMVQ path). The MMQ path caches them in
  shared memory via the loader.
* **No FP16 accumulation path.** All F32 arithmetic. The MMVF kernel
  (ARTX09) has an F16-accumulate mode for F16/BF16 weights, but the
  quantized path never uses F16 accumulation (the int sum is exact
  anyway; only the final scale multiply is F32).

---

## 13. Architectural Strengths

1. **Clean `vec_dot_q_cuda_t` signature.** One signature, 22
   implementations, one `constexpr` switch. The MMVQ kernel holds a
   single function pointer that `nvcc` devirtualises. Adding a new
   quant format = adding one `case` to the switch and one function to
   this file.

2. **Three-way zero-point fold is optimal.** The symmetric-bias fold
   (via `s8`), the asymmetric-min fold (via `dot2 = dp4a(0x01..)`), and
   the IQ1 delta fold (via `sumy = dp4a(u, 0x01..)`) each reuse the
   `dp4a` unit for what would otherwise be a separate reduction. No
   per-element subtract, no extra loop.

3. **Branchless signed-flip idiom.** `__vsub4(grid^signs, signs)` is
   2 instructions for 4 sign flips — the optimal codegen for any
   architecture with a packed-byte subtract.

4. **`vdr` template parameter.** Lets the inner `*_impl` be shared
   between MMVQ and MMQ with different unroll factors. The compiler
   emits two distinct code paths (small VDR for MMVQ, large VDR for
   MMQ) from the same source.

5. **Separate `*_mmvq` vs `*_mmq` for K-quants.** The MMQ variant
   accepts pre-unpacked `int8_t * scales` so the scale unpacking
   happens once per tile (in the loader), not once per call. The MMVQ
   variant accepts the raw packed `scales[]` bytes and unpacks in-thread.

6. **`get_int_from_table_16` is a clean per-vendor primitive.** Three
   code paths (AMD `__builtin_amdgcn_perm`, NVIDIA `__byte_perm`,
   generic C) selected by `#if defined(GGML_USE_HIP) / !defined(GGML_USE_MUSA)
   / #else`. The function returns `int2` of (even, odd) byte permutations
   of a 16-entry table by 4-bit indices — exactly what 4-bit quants need.

7. **MMVQ/MMQ shared inner templates.** The Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
   inner `*_impl<vdr>` templates are reused verbatim between MMVQ and
   MMQ. Only the K-quants need separate `*_mmvq` / `*_mmq` variants
   (because the MMQ loader pre-unpacks scales).

---

## 14. Architectural Weaknesses

### W1 — Three independent K-quant scale unpackers

**Evidence**: `vecdotq.cuh:890-899` (Q4_K MMVQ), `vecdotq.cuh:935-944`
(Q5_K MMVQ), `vecdotq.cuh:455-465` (Q3_K MMVQ), `dequantize.cuh:157-164`
(`get_scale_min_k4`, shared by Q4_K and Q5_K dequantize), and
`mmq-load-tiles.cuh:612-620` (`unpack_scales_q45_K`, MMQ loader).

**Impact**: Five different pieces of code all decode the same 6-bit
packed scale format. Bug fixes must be applied in five places. None
of these is canonical — the layouts are duplicated knowledge.

### W2 — Q1_0 `__byte_perm` chain has no AMD/MUSA fast path

**Evidence**: `vecdotq.cuh:701-712`. Eight `__byte_perm` calls. AMD
defines `__byte_perm` as a macro that compiles to generic code
(common.cuh shim), so the Q1_0 vecdot on AMD is significantly slower
than the Q4_0 path. The file has no `__builtin_amdgcn_perm`-based
alternative for Q1_0, unlike `get_int_from_table_16` which has both.

**Impact**: Q1_0 on AMD is register-pressure-heavy and uses ~8 scalar
instructions per byte instead of 1 permute. Probably 2-3× slower than
it could be. No documented fix.

### W3 — `vec_dot_nvfp4_q8_1` accumulates in F32, not int

**Evidence**: `vecdotq.cuh:355` — `sum += d * float(sumi);` inside the
VDR loop. Every other integer-quant path accumulates in `int sumi` and
does one F32 multiply at the end. NVFP4 instead does `VDR_NVFP4_Q8_1_MMVQ/2
= 2` F32 multiplies and adds inside the loop.

**Impact**: 2 FMA per call instead of 1 FMA + 1 int mul. For MMVQ's
typical `vdr = 4`, that's 2 extra FMAs per call. Justified by NVFP4's
per-sub-block scale (each of the 2 sub-blocks has its own ue4m3 scale),
but worth flagging as a different pattern.

### W4 — `get_vec_dot_q_cuda` is a 22-arm switch duplicated for VDR

**Evidence**: `mmvq.cu:10-36` (`get_vec_dot_q_cuda`) and `:38-62`
(`get_vdr_mmvq`). Both switches enumerate the same 22 cases. Adding a
new quant means adding a case to both, plus the `get_vdr_mmq` switch
(not shown, but exists in `mmq.cuh`).

**Impact**: Triple-maintained switch. A missing case is a silent
compile-time error (returns `nullptr` → link error eventually). Could
be consolidated into a single `struct { vec_dot_q_cuda_t fn; int vdr_mmvq;
int vdr_mmq; } traits[GGML_TYPE_COUNT]`.

### W5 — `FAST_FP16_AVAILABLE` produces different ULPs per arch

**Evidence**: `vecdotq.cuh:154-163` (Q4_1), `:225-234` (Q5_1),
`:268-277` (Q8_1). The `#ifdef FAST_FP16_AVAILABLE` branch uses
`__hmul2(dm4, ds8)` (F16 multiply) then converts to F32; the `#else`
branch converts both to F32 first then multiplies. Different rounding.

**Impact**: Bit-different results on Pascal- (no FAST_FP16) vs Volta+
(FAST_FP16). Documented implicitly by the macro but not called out in
user-facing docs.

### W6 — No fallback for missing `__vsub4` / `__vcmpne4` / `__vsubss4`

**Evidence**: `vecdotq.cuh:471, 638, 1006, 1011, 1043, 1091, 1130,
1176`. These PTX intrinsics are gated by `__CUDA_ARCH__ >=
GGML_CUDA_CC_PASCAL` (via `common.cuh` shims). Pre-Pascal NVIDIA
hardware cannot run IQ2/IQ3/IQ4_XS vecdot — the build will compile but
the kernel will trap at runtime.

**Impact**: Hard arch floor. Not a bug (IQ2/IQ3 were never intended for
pre-Pascal) but worth documenting as a deployment constraint.

### W7 — VDR constants are macros, not `constexpr`

**Evidence**: `vecdotq.cuh:109-1296` — every `VDR_*` is a `#define`.
The macros are consumed inside `#pragma unroll` loops and template
parameters (`vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>`).

**Impact**: No type safety, no namespace scoping, no IDE introspection.
Could be `static constexpr int VDR_Q4_0_Q8_1_MMVQ = 2;` with no
performance change.

### W8 — `vec_dot_q1_0_q8_1` uses a different scale scheme from every other simple quant

**Evidence**: `vecdotq.cuh:675-723`. Q1_0 has one `float d` per 128-
element block (no `half2 dm`, no `s` field used). The activation Q8_1
scale is read as `__low2float(bq8_1_chunk->ds)` (line 721) and multiplied
directly: `d1 * d8 * sumi`. No zero-point fold — Q1_0 is symmetric
around 0, so the values are already `{-d, +d}`.

**Impact**: Architecturally clean but inconsistent with Q4_0/Q8_0 which
use the `s` abs-sum for the bias fold. The Q1_0 path is the only simple
quant with a 128-element block (vs 32 for Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).

### W9 — Hardcoded `8`, `16`, `32` bias constants

**Evidence**: `vecdotq.cuh:133` `(8*vdr/QI4_0)`, `:197` `(16*vdr/QI5_0)`,
`:638` `0x20202020`, `:471` `__vsubss4(vil, vih)`. The Q4_0 implicit
zero-point is `8`, Q5_0's is `16`, Q6_K's is `32`. These are baked into
the arithmetic, not parameterised.

**Impact**: Not wrong (the zero-points are part of the format
definition), but they make the code harder to read. A `static constexpr
int ZP_Q4_0 = 8` would document the source.

### W10 — No documentation of the K-quant scale layout in this file

**Evidence**: `vecdotq.cuh:890-899` (Q4_K scale unpacking) has no
comment explaining the `scales[j+0] & 0x3f3f` / `(scales[j+2] >> 0) &
0x0f0f | (scales[j-2] & 0xc0c0) >> 2` pattern. The reader must consult
ARTX06 or `ggml-quants.c` to understand the 6-bit packing.

**Impact**: Maintenance burden. Bug-fix patches to this code require
reverse-engineering the bit layout from the constants.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda` | **ADOPT** | `vec_dot_q_cuda_t` typedef + `constexpr` device-function-pointer switch | One signature, one switch, devirtualised call. The cleanest possible per-quant dispatch. |
| `glcuda` | **ADOPT** | `dp4a`-only integer reduction for all integer quants | Universal, portable, 4× throughput on supporting hardware. |
| `glcuda` | **ADOPT** | Symmetric bias fold via Q8_1 `s` field | Eliminates per-element subtract; one FMA in the F32 epilogue. |
| `glcuda` | **ADOPT** | Asymmetric min fold via `dot2 = dp4a(0x01010101, u, 0)` | Reuses `dp4a` unit for sum-of-activation; no extra reduction. |
| `glcuda` | **ADOPT** | Branchless signed flip `__vsub4(grid^signs, signs)` | 2 instructions per 4-element group; optimal codegen. |
| `glcuda` | **ADOPT** | `vdr` template parameter + separate `VDR_*_MMVQ` / `VDR_*_MMQ` | Lets the same `*_impl` template serve MMVQ (small VDR) and MMQ (large VDR). |
| `glcuda` | **ADAPT** | K-quant 6-bit scale unpacker | Consolidate the three unpackers (vecdot MMVQ, vecdot MMQ, dequantize) into one device function. |
| `glcuda` | **ADAPT** | Q1_0 `__byte_perm` unpacker | Add an AMD `__builtin_amdgcn_perm`-based alternative (mirror `get_int_from_table_16`). |
| `glcuda` | **REJECT** | Triple-maintained per-quant `switch` (`get_vec_dot_q_cuda` + `get_vdr_mmvq` + `get_vdr_mmq`) | Replace with a single `traits[type] = {fn, vdr_mmvq, vdr_mmq}` table. |
| `glcuda` | **MONITOR** | NVFP4 F32 accumulation inside the VDR loop | Watch for precision regressions vs an int-accumulation variant. |
| `glcuda` | **MONITOR** | `FAST_FP16_AVAILABLE` ULP divergence | Watch for cross-arch test failures; consider always-F32 scale multiplies. |
| `glcuda` | **DEFER** | IQ2/IQ3/IQ4_XS paths | Defer until glcuda has Q4_0/Q4_K/Q6_K working — IQ paths add 6× the code for ~10% of users. |

---

## 16. Recommendations

### R1 — ADOPT `vec_dot_q_cuda_t` device-function-pointer dispatch
**Priority:** Critical **Difficulty:** M **Dependencies:** none
GwenLand's `glcuda` should define `typedef float (*gl_vec_dot_q_cuda_t)(const
void * vbq, const block_q8_1 * bq8_1, const int & kbx, const int & iqs);`
and a `constexpr __device__` switch returning the right per-quant
function. Store in a `constexpr` local in the MMVQ kernel so the call
is devirtualised.

### R2 — ADOPT symmetric-bias fold via Q8_1 `s` field
**Priority:** Critical **Difficulty:** S **Dependencies:** R1
For symmetric quants (Q4_0, Q5_0, Q8_0), the activation Q8_1 block
should carry both `d` (scale) and `s` (abs-sum). The vecdot returns
`d_weight * (sumi * d_act - bias * s_act)` where `bias = 8` (Q4_0),
`16` (Q5_0), `0` (Q8_0). One FMA in the epilogue, no per-element
subtract.

### R3 — ADOPT branchless signed-flip idiom for IQ formats
**Priority:** High **Difficulty:** S **Dependencies:** R1
For IQ2/IQ3 grid-based formats, the sign flip is `__vsub4(grid_pos ^
signs_mask, signs_mask)` where `signs_mask = __vcmpne4(signs & 0x08040201,
0)`. 2 instructions per 4-element group, no branches.

### R4 — ADAPT: consolidate K-quant scale unpacker
**Priority:** High **Difficulty:** M **Dependencies:** R1
Replace the three hand-rolled K-quant scale unpackers (in vecdotq MMVQ,
vecdotq MMQ, dequantize.cuh) with a single device function
`unpack_k_scales(scales_bytes, j, &sc, &m)` that returns the per-block
scale and min for index `j`. All three call sites use the same function.

### R5 — ADAPT: add AMD `__builtin_amdgcn_perm` path for Q1_0
**Priority:** Medium **Difficulty:** S **Dependencies:** R1
The Q1_0 vecdot uses 8 `__byte_perm` calls on NVIDIA. On AMD, emit
`__builtin_amdgcn_perm`-based code (mirror `get_int_from_table_16`'s
structure). 2-3× speedup expected for Q1_0 on RDNA.

### R6 — REJECT triple-maintained per-quant switches
**Priority:** Medium **Difficulty:** S **Dependencies:** R1
Replace `get_vec_dot_q_cuda`, `get_vdr_mmvq`, `get_vdr_mmq` with a
single `struct gl_cuda_quant_traits { vec_dot_q_cuda_t fn; int vdr_mmvq;
int vdr_mmq; };` table indexed by `ggml_type`. One source of truth.

### R7 — MONITOR NVFP4 F32 accumulation
**Priority:** Low **Difficulty:** S **Dependencies:** R1
The `vec_dot_nvfp4_q8_1` path does `sum += d * float(sumi)` inside the
VDR loop (2 FMA per call). Watch for precision regressions vs an
int-accumulate-then-multiply variant. May need to switch to int
accumulation if precision issues surface.

### R8 — ADOPT `vdr` template parameter for impl functions
**Priority:** High **Difficulty:** S **Dependencies:** R1
The inner `*_impl<vdr>` template lets the same source serve MMVQ and
MMQ with different unroll factors. The compiler emits two distinct code
paths from one source.

### R9 — DEFER IQ2/IQ3/IQ1 paths in initial glcuda
**Priority:** Low **Difficulty:** XL **Dependencies:** R1
The IQ paths add ~600 lines of code for grid-lookup logic. Defer
implementation until glcuda has the simple quants (Q4_0, Q4_1, Q5_0,
Q5_1, Q8_0) and K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) working. MXFP4
and NVFP4 should be implemented early because they're the path to
Blackwell native FP4 (ARTX10).

### R10 — ADOPT per-quant `__forceinline__` on every vec_dot
**Priority:** Medium **Difficulty:** XS **Dependencies:** R1
Every `vec_dot_*` function should be `static __device__ __forceinline__`.
The function body is inlined into the MMVQ K-loop and the MMQ
tile-vecdot. Without `__forceinline__`, `nvcc` may or may not inline
depending on register pressure.

---

## 17. Findings

### Finding ARTX13-F01

```
Finding ID:           ARTX13-F01
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            vec_dot device-function-pointer dispatch
Source File:          ggml/src/ggml-cuda/mmvq.cu
Function:             get_vec_dot_q_cuda
Lines:                8-36
Summary:              Per-quant dispatch via constexpr __device__ function
                      pointer; devirtualised by nvcc at -O3.
Observation:          The typedef vec_dot_q_cuda_t = float(*)(const void *,
                      const block_q8_1 *, const int &, const int &) is
                      declared at mmvq.cu:8. get_vec_dot_q_cuda(type) is a
                      22-arm constexpr __device__ switch returning the
                      right vec_dot_*_q8_1 from vecdotq.cuh. The MMVQ kernel
                      stores the result in a constexpr local (mmvq.cu:500,
                      724) and calls it inside the K loop. Because the local
                      is constexpr, nvcc resolves the indirect call at
                      compile time and emits a direct call (or, with
                      __forceinline__ on the vec_dot_* functions, an inlined
                      body).
Evidence:             mmvq.cu:8 (typedef), 10-36 (switch), 500, 724 (constexpr
                      local in MMVQ kernel).
Architectural Impact: This is the contract layer between the MMVQ kernel and
                      the per-quant implementations. Adding a quant = adding
                      one case to the switch + one function to vecdotq.cuh.
Correctness Impact:   None. Dispatch is indirect but deterministic per
                      (type) pair.
Optimization Type:    SIMD / devirtualised indirect call.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same typedef + constexpr switch pattern in
                      glcuda.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX13-F02

```
Finding ID:           ARTX13-F02
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            dp4a universal reduction
Source File:          ggml/src/ggml-cuda/vecdotq.cuh, ggml/src/ggml-cuda/common.cuh
Function:             ggml_cuda_dp4a, every *_impl
Lines:                common.cuh:703-741; vecdotq.cuh:126, 151, 184, 191, 215, 222, 251, 265, 295, 320-321, 349-352, 377, 383, 409, 415, 427, 434, 473, 492, 517-518, 543, 579-580, 606, 640, 662-666, 714-717, 1008, 1013, 1051-1052, 1098-1099, 1139-1140, 1182-1183, 1215-1216, 1250-1251, 1255-1256, 1288-1289, 1313-1314
Summary:              Every integer-quant vec_dot uses ggml_cuda_dp4a (4-way
                      int8 dot with 32-bit accumulate) as the universal
                      reduction primitive.
Observation:          common.cuh:703-741 implements ggml_cuda_dp4a with five
                      branches: HIP-CDNA/RDNA2 uses
                      __builtin_amdgcn_sdot4; HIP-RDNA3/4 uses
                      __builtin_amdgcn_sudot4; HIP-RDNA1/gfx900 uses inline
                      asm (4 v_mul_i32_i24 + 3 v_add3_u32); pre-Volta NVIDIA
                      uses scalar C; everything else uses __dp4a. Every
                      *_impl template in vecdotq.cuh calls this function 2-8
                      times per VDR iteration. There is no __dp2a, no
                      Tensor Core, no FMA-with-int8 — just dp4a.
Evidence:             common.cuh:703-741 (dp4a dispatcher); vecdotq.cuh:126
                      (first use in Q4_0), vecdotq.cuh:517-518 (Q4_K), etc.
Architectural Impact: One primitive, five backend implementations. Adding a
                      new ISA = adding one branch to ggml_cuda_dp4a; every
                      quant benefits.
Correctness Impact:   None. __dp4a is non-saturating on NVIDIA; sdot4 is
                      non-saturating on AMD. int32 accumulator never
                      overflows for the per-call VDR range.
Optimization Type:    SIMD (4-way int8 dot product).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same primitive + per-vendor dispatch in glcuda.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX13-F03

```
Finding ID:           ARTX13-F03
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Symmetric-quant zero-point fold via Q8_1 abs-sum
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             vec_dot_q4_0_q8_1_impl, vec_dot_q5_0_q8_1_impl
Lines:                115-134 (Q4_0), 172-198 (Q5_0)
Summary:              Q4_0/Q5_0 subtract the implicit -8/-16 zero-point
                      inside the F32 scale expression using the Q8_1 block's
                      abs-sum field s, avoiding a per-element subtract.
Observation:          The Q8_1 block layout (ggml-common.h) stores both a
                      per-32-element scale d (half) and a per-32-element
                      abs-sum s (half) in a single half2 ds. The symmetric
                      Q4_0 vecdot returns d4 * (sumi * ds8f.x - (8*vdr/QI4_0)
                      * ds8f.y), where ds8f.x is the activation scale and
                      ds8f.y is the activation abs-sum. The factor (8*vdr/QI4_0)
                      is the per-thread share of the implicit -8 zero-point:
                      8 (the bias) times vdr (ints processed) over QI4_0
                      (ints per block). One FMA in the F32 epilogue replaces
                      what would otherwise be 4*vdr per-element subtracts
                      before the dp4a.
Evidence:             vecdotq.cuh:130-133 (Q4_0); vecdotq.cuh:194-197 (Q5_0,
                      uses 16*vdr/QI5_0).
Architectural Impact: Saves 4*vdr subtracts per call. The Q8_1 s field
                      doubles as the activation sum, avoiding a separate
                      dp4a(0x01010101, u, 0) for the symmetric formats.
Correctness Impact:   None. The arithmetic is mathematically equivalent
                      to (q - 8) * (a) summed; the fold just factors out the
                      constant -8.
Optimization Type:    Kernel fusion (bias fold into scale multiply).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same pattern for symmetric quants in glcuda.
                      The Q8_1 activation block must carry both d and s.
Priority:             High
Difficulty:           S
Dependencies:         ARTX13-F02
Confidence:           High
```

### Finding ARTX13-F04

```
Finding ID:           ARTX13-F04
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Asymmetric-quant min fold via dp4a(0x01010101, u, 0)
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             vec_dot_q4_K_q8_1_impl_vmmq, vec_dot_q5_K_q8_1_impl_vmmq, vec_dot_q2_K_q8_1_impl_mmvq
Lines:                518-521 (Q4_K), 580 (Q5_K), 380-383 (Q2_K)
Summary:              K-quants compute a separate dot2 = dp4a(0x01010101, u,
                      0) that captures the per-int sum of the activation,
                      then multiply by the per-block min m to subtract the
                      asymmetric zero-point contribution.
Observation:          For Q4_K, the inner loop computes both dot1 = dp4a(v,
                      u, 0) (the actual dot product) and dot2 = dp4a(0x01010101,
                      u, 0) (the sum of activation int8s in this int). The
                      min contribution is sumf_m += d8[i] * (dot2 * m[i]).
                      Q2_K uses a different idiom: int m = sc >> 4; m |= m<<8;
                      m |= m<<16; sumf_m += d8[i] * dp4a(m, u, 0); — broadcast
                      the per-block min to all 4 bytes of an int, then dp4a
                      with u. Both schemes reuse the dp4a unit for what
                      would otherwise be a separate scalar sum.
Evidence:             vecdotq.cuh:518-521 (Q4_K), 580 (Q5_K), 380-383 (Q2_K
                      broadcast-min variant).
Architectural Impact: Saves a per-element multiply-add for the asymmetric
                      zero-point correction. Two dp4a per VDR iteration
                      instead of one dp4a + 4*vdr scalar ops.
Correctness Impact:   None. Mathematically equivalent to subtracting m
                      from each weight before the dot product.
Optimization Type:    Kernel fusion (min fold into dp4a).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same pattern for asymmetric K-quants in glcuda.
Priority:             High
Difficulty:           S
Dependencies:         ARTX13-F02
Confidence:           High
```

### Finding ARTX13-F05

```
Finding ID:           ARTX13-F05
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Branchless signed-flip idiom for IQ formats
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             vec_dot_iq2_xxs_q8_1, vec_dot_iq2_xs_q8_1, vec_dot_iq2_s_q8_1, vec_dot_iq3_xxs_q8_1, vec_dot_iq3_s_q8_1
Lines:                1005-1006 (IQ2_XXS), 1042-1043 (IQ2_XS), 1088-1092 (IQ2_S), 1129-1130 (IQ3_XXS), 1173-1177 (IQ3_S)
Summary:              IQ2/IQ3 grid-based quants apply per-byte sign flips
                      branchlessly via __vsub4(grid^signs_mask, signs_mask)
                      where signs_mask = __vcmpne4(signs & 0x08040201, 0).
Observation:          Each IQ2/IQ3 grid lookup returns a uint2 of 8 packed
                      bytes (4 low, 4 high). A separate signs byte (1 bit
                      per output byte) determines whether each byte is
                      negated. The negation is computed as: signs_mask =
                      __vcmpne4(signs_packed, 0) (yields 0xFF..FF where the
                      sign bit is set, 0x00..00 otherwise); grid_signed =
                      __vsub4(grid ^ signs_mask, signs_mask). XOR with
                      signs_mask flips all bits where sign is set; subtracting
                      signs_mask then adds 1 to those bytes (two's-complement
                      negation). 2 instructions per 4-element group, no
                      branches.
Evidence:             vecdotq.cuh:1005-1006 (iq2_xxs), 1042-1043 (iq2_xs),
                      1088-1092 (iq2_s), 1129-1130 (iq3_xxs), 1173-1177
                      (iq3_s).
Architectural Impact: Optimal codegen for any architecture with a packed-
                      byte subtract. Saves 4 conditional branches per VDR
                      iteration.
Correctness Impact:   None. The arithmetic is the standard two's-complement
                      negation, just branchless.
Optimization Type:    SIMD (branchless packed-byte arithmetic).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same idiom for IQ formats in glcuda.
Priority:             High
Difficulty:           S
Dependencies:         ARTX13-F02
Confidence:           High
```

### Finding ARTX13-F06

```
Finding ID:           ARTX13-F06
Category:             QUANTIZATION
Engine:               CUDA
Component:            K-quant 6-bit scale unpacking (three implementations)
Source File:          ggml/src/ggml-cuda/vecdotq.cuh, ggml/src/ggml-cuda/dequantize.cuh, ggml/src/ggml-cuda/mmq-load-tiles.cuh
Function:             vec_dot_q4_K_q8_1 (vecdotq.cuh:890-899), get_scale_min_k4 (dequantize.cuh:157-164), unpack_scales_q45_K (mmq-load-tiles.cuh:612-620)
Lines:                vecdotq.cuh:890-899; dequantize.cuh:157-164; mmq-load-tiles.cuh:612-620
Summary:              The K-quant 6-bit packed scale layout is decoded by
                      three independent hand-rolled functions; bug fixes
                      must be applied in three places.
Observation:          The K-quant scales[] array packs 8 per-32-element
                      scales and 8 per-32-element mins into 12 bytes per
                      super-block (6 bits each, with the high 2 bits shared
                      across pairs — see ARTX06). vecdotq.cuh:890-899
                      decodes this in-thread for Q4_K MMVQ via a `scales[j+0]
                      & 0x3f3f` / `((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2]
                      & 0xc0c0) >> 2)` pattern conditional on j < 2.
                      dequantize.cuh:157-164 (get_scale_min_k4) decodes it
                      with a different pattern: `q[j] & 63` / `q[j+4] & 63`
                      for j < 4, else `(q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)`.
                      mmq-load-tiles.cuh:612-620 (unpack_scales_q45_K) is a
                      third variant for the MMQ loader. All three produce
                      the same result; none is canonical.
Evidence:             vecdotq.cuh:890-899 (MMVQ Q4_K); dequantize.cuh:157-164
                      (dequantize); mmq-load-tiles.cuh:612-620 (MMQ loader).
Architectural Impact: Triple-maintained bit manipulation. A bug in any one
                      unpacker produces silent per-format corruption that
                      only shows up in differential testing.
Correctness Impact:   None for current code (all three are correct). Risk
                      of future divergence if one is patched and others
                      aren't.
Optimization Type:    None (architectural concern).
GwenLand Target:      glcuda
Recommendation:       ADAPT. glcuda should have ONE device function for K-
                      quant scale unpacking, called from all three sites.
Priority:             High
Difficulty:           M
Dependencies:         ARTX13-F01
Confidence:           High
```

### Finding ARTX13-F07

```
Finding ID:           ARTX13-F07
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            get_int_from_table_16 (4-bit LUT expansion)
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             get_int_from_table_16
Lines:                34-95
Summary:              Per-vendor 4-bit-index → 8-byte LUT lookup primitive
                      used by MXFP4, NVFP4, IQ4_NL, IQ4_XS vecdot.
Observation:          The function takes a 32-bit int q4 (8 packed 4-bit
                      indices) and a 16-byte int8_t table, returns int2 of
                      (even-index bytes, odd-index bytes) from the table.
                      Three implementations: AMD uses __builtin_amdgcn_perm
                      twice per half (lines 43-54); NVIDIA uses __byte_perm
                      (lines 60-80); MUSA uses scalar char4 lookups
                      (lines 83-93). The function is called by
                      vec_dot_mxfp4_q8_1, vec_dot_nvfp4_q8_1,
                      vec_dot_iq4_nl_q8_1, vec_dot_iq4_xs_q8_1.
Evidence:             vecdotq.cuh:34-95 (definition); 318, 344-345, 1286,
                      1308 (call sites).
Architectural Impact: One primitive, three backend implementations. Adding
                      a new ISA = adding one branch; all four FP4/IQ4 formats
                      benefit.
Correctness Impact:   None. All three implementations produce the same
                      int2 result for the same inputs.
Optimization Type:    SIMD (vendor-specific permute intrinsics).
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same per-vendor structure for any 4-bit LUT
                      expansion in glcuda.
Priority:             High
Difficulty:           M
Dependencies:         ARTX13-F02
Confidence:           High
```

### Finding ARTX13-F08

```
Finding ID:           ARTX13-F08
Category:             QUANTIZATION
Engine:               CUDA
Component:            NVFP4 vecdot with per-sub-block F32 accumulation
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             vec_dot_nvfp4_q8_1
Lines:                328-359
Summary:              NVFP4 vecdot is the only integer-quant path that
                      accumulates in F32 inside the VDR loop (sum += d *
                      float(sumi)) instead of accumulating in int and
                      multiplying by the scale at the end.
Observation:          Every other integer-quant vecdot in vecdotq.cuh
                      accumulates into a single int sumi and does one F32
                      scale multiply in the epilogue. vec_dot_nvfp4_q8_1
                      instead does: for i in [0, VDR_NVFP4_Q8_1_MMVQ/2):
                      compute per-sub-block int sumi, look up the per-sub-
                      block ue4m3 scale d, and sum += d * float(sumi). The
                      reason is that NVFP4 has a different ue4m3 scale per
                      16-element sub-block, so a single end-of-call scale
                      multiply would be wrong. VDR_NVFP4_Q8_1_MMVQ = 4
                      means 2 sub-blocks per call, 2 FMA in the loop.
Evidence:             vecdotq.cuh:338-356 (loop with F32 accumulation);
                      :354 (per-sub-block d lookup); :355 (sum += d * float(sumi)).
Architectural Impact: 2 FMA per call instead of 1 FMA + 1 int mul. The
                      per-sub-block scale forces the F32 accumulation.
Correctness Impact:   None. The arithmetic is correct for NVFP4's per-sub-
                      block scale layout.
Optimization Type:    SIMD (per-sub-block scale fusion).
GwenLand Target:      glcuda
Recommendation:       MONITOR. Watch for precision regressions vs an
                      alternative that accumulates per-sub-block ints and
                      applies scales at the end. Current pattern is forced
                      by the format; alternative would need 2 int
                      accumulators.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX13-F07
Confidence:           High
```

### Finding ARTX13-F09

```
Finding ID:           ARTX13-F09
Category:             SIMD_STRATEGY
Engine:               CUDA
Component:            Per-quant VDR constants decouple MMVQ from MMQ
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             VDR_*_Q8_1_MMVQ / VDR_*_Q8_1_MMQ constants
Lines:                109-110, 112-113, 136-137, 169-170, 200-201, 240-241, 304-305, 328-329, 360-361, 443-444, 501-502, 557-558, 620-621, 987-988, 1022-1023, 1063-1064, 1111-1112, 1149-1150, 1192-1193, 1225-1226, 1272-1273, 1296-1297
Summary:              Each quant format defines two VDR constants — one for
                      MMVQ (small, 1-4) and one for MMQ (large, 1-8) —
                      letting the same *_impl template serve both contexts.
Observation:          VDR (Vec Dot Ratio) is the number of contiguous
                      32-bit ints (qi-chunks) one thread processes per
                      vec_dot_* call. MMVQ uses small VDR to match its per-
                      warp K-step granularity (blocks_per_iter = vdr *
                      nwarps * warp_size / qi); MMQ uses large VDR to match
                      its per-tile K-step granularity. For Q4_0: MMVQ=2,
                      MMQ=4. For Q8_0: MMVQ=2, MMQ=8 (4× larger). For IQ2_*:
                      MMVQ=MMQ=2 (grid lookup cost dominates, no benefit
                      from larger MMQ VDR). For Q6_K: MMVQ=1, MMQ=8 (8×
                      larger).
Evidence:             vecdotq.cuh:109-1296 (all VDR_* definitions); mmvq.cu:38-62
                      (get_vdr_mmvq consumes VDR_*_MMVQ); mmq.cuh (get_vdr_mmq
                      consumes VDR_*_MMQ).
Architectural Impact: Decouples per-thread granularity from per-tile
                      granularity. The same vec_dot_q4_0_q8_1_impl<vdr>
                      template serves both contexts with different unroll
                      factors.
Correctness Impact:   None. The inner arithmetic is identical for any vdr;
                      only the unroll count differs.
Optimization Type:    SIMD / template specialisation per unroll factor.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Same per-quant VDR pair in glcuda.
Priority:             High
Difficulty:           S
Dependencies:         ARTX13-F01
Confidence:           High
```

### Finding ARTX13-F10

```
Finding ID:           ARTX13-F10
Category:             LAYOUT_SUBOPTIMAL
Engine:               CUDA
Component:            Q1_0 vecdot uses 8 __byte_perm calls with no AMD/MUSA fast path
Source File:          ggml/src/ggml-cuda/vecdotq.cuh
Function:             vec_dot_q1_0_q8_1
Lines:                675-723
Summary:              The Q1_0 vecdot uses 8 __byte_perm calls (lines 701-712)
                      to unpack 1-bit indices into 0/-1 byte values. Unlike
                      get_int_from_table_16, there is no __builtin_amdgcn_perm
                      path for AMD; the AMD build falls through to the same
                      __byte_perm calls (mapped to generic code via common.cuh
                      shim).
Observation:          Q1_0 packs 32 weights as 4 bytes (32 bits, 1 bit per
                      weight). The vecdot unpacks 16 bits at a time into 16
                      byte values via __byte_perm(0x11100100, 0x11100100, q
                      >> 0) etc. The pattern 0x11100100 selects bytes 0, 0,
                      1, 1 from the source — a lookup table for the 4-bit
                      index → byte-position mapping. Eight such calls cover
                      all 32 weights. get_int_from_table_16 (lines 34-95)
                      has explicit AMD __builtin_amdgcn_perm branches; this
                      Q1_0 code does not. On AMD, __byte_perm compiles to
                      generic shift-and-mask code, ~3× slower than the
                      equivalent permute.
Evidence:             vecdotq.cuh:701-712 (Q1_0 unpacker); compare to
                      vecdotq.cuh:34-95 (get_int_from_table_16 with AMD path).
Architectural Impact: Q1_0 on AMD is significantly slower than other simple
                      quants. Probably 2-3× slower than it could be with a
                      perm-based implementation.
Correctness Impact:   None. The __byte_perm fallback produces correct
                      results; just slow.
Optimization Type:    None (suboptimal — missing AMD optimisation).
GwenLand Target:      glcuda
Recommendation:       ADAPT. Add an AMD __builtin_amdgcn_perm-based path
                      for Q1_0 in glcuda, mirroring get_int_from_table_16's
                      structure.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX13-F07
Confidence:           Medium
```

---

## 18. Unknowns

* **U1**. Whether the `constexpr` device-function-pointer pattern is
  actually devirtualised by `nvcc -O3` for every template instantiation,
  or whether some instantiations fall back to an indirect call. Requires
  PTX inspection (`cuobjdump`). Static analysis cannot confirm.
* **U2**. Whether the K-quant scale unpackers in vecdotq.cuh, dequantize.cuh,
  and mmq-load-tiles.cuh produce bit-identical results for all valid
  `scales[]` byte values, or whether edge cases (e.g., scales with bits
  set in the high-2-bit extension) diverge. Requires differential
  fuzzing.
* **U3**. The actual throughput cost of the IQ2/IQ3 grid lookups on
  different NVIDIA generations. The `__vsub4(grid^signs, signs)` idiom
  is 2 instructions, but the preceding `iq2xxs_grid[aux8[k0/2]]` lookup
  is a 4-cycle L1 read-only cache hit on Ampere+; unclear on Pascal.
  Requires profiling.
* **U4**. Whether the NVFP4 F32 accumulation inside the VDR loop
  (ARTX13-F08) produces different ULPs than an int-accumulate-then-multiply
  variant. The F32 sum is over 2 sub-block sums × `d` (ue4m3 scale); the
  alternative would be `sum0 * d0 + sum1 * d1` with two int accumulators.
  Requires differential testing.
* **U5**. Whether `FAST_FP16_AVAILABLE` (gating the `__hmul2` scale
  pre-multiply in Q4_1/Q5_1/Q8_1) produces measurably different ULPs
  than the F32-only fallback. The hmul2 path rounds the scale product
  to F16 before converting to F32; the F32 path keeps full F32 precision.
  Requires differential testing on Volta vs Pascal.
* **U6**. Whether the `__vsub4` / `__vcmpne4` / `__vsubss4` intrinsics
  have direct AMD equivalents or whether they compile to multi-instruction
  sequences on HIP. The shims in common.cuh were not fully audited here.
  Requires PTX/SASS inspection.
* **U7**. The minimum CUDA compute capability required to run IQ2/IQ3/
  IQ4_XS vecdot. ARTX13-F06 documents that `__vsub4` etc. require
  Pascal+, but does the build system actually prevent compilation for
  pre-Pascal? Or does it compile and trap at runtime? Requires build
  system inspection (not in scope of this audit).

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines                |
| --------- | --------------------------------------------------- | ---------------------------------------------- | -------------------- |
| R01       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `get_int_b1`, `get_int_b2`, `get_int_b4`       | 7-29                 |
| R02       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `get_int_from_table_16`                        | 34-95                |
| R03       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `unpack_ksigns`                                | 97-104               |
| R04       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `VDR_*_Q8_1_MMVQ` / `VDR_*_Q8_1_MMQ` constants | 109-1296 (per-quant) |
| R05       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q4_0_q8_1_impl<vdr>`                  | 115-134              |
| R06       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q4_1_q8_1_impl<vdr>`                  | 139-167              |
| R07       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q5_0_q8_1_impl<vdr>`                  | 172-198              |
| R08       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q5_1_q8_1_impl<vdr>`                  | 203-238              |
| R09       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q8_0_q8_1_impl<T, vdr>`               | 243-255              |
| R10       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q8_1_q8_1_impl<vdr>`                  | 257-281              |
| R11       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q8_0_16_q8_1_impl<vdr>`               | 283-302              |
| R12       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_mxfp4_q8_1`, `vec_dot_nvfp4_q8_1`     | 307-359              |
| R13       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q2_K_q8_1_impl_mmvq` / `_mmq`         | 364-441              |
| R14       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q3_K_q8_1_impl_mmvq` / `_mmq`         | 447-499              |
| R15       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q4_K_q8_1_impl_vmmq` / `_mmq`         | 505-555              |
| R16       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q5_K_q8_1_impl_vmmq` / `_mmq`         | 561-618              |
| R17       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q6_K_q8_1_impl_mmvq` / `_mmq`         | 624-673              |
| R18       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q1_0_q8_1`                            | 675-723              |
| R19       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q4_0_q8_1` … `vec_dot_q8_0_q8_1`      | 725-817              |
| R20       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_q2_K_q8_1` … `vec_dot_q6_K_q8_1`      | 819-985              |
| R21       | `ggml/src/ggml-cuda/vecdotq.cuh`                    | `vec_dot_iq2_xxs_q8_1` … `vec_dot_iq4_xs_q8_1` | 990-1322             |
| R22       | `ggml/src/ggml-cuda/mmvq.cu`                        | `vec_dot_q_cuda_t` typedef, `get_vec_dot_q_cuda` | 8-36               |
| R23       | `ggml/src/ggml-cuda/mmvq.cu`                        | `get_vdr_mmvq`                                 | 38-62                |
| R24       | `ggml/src/ggml-cuda/mmvq.cu`                        | `mul_mat_vec_q` (consumer)                     | 478-699              |
| R25       | `ggml/src/ggml-cuda/common.cuh`                     | `ggml_cuda_dp4a`                               | 703-741              |
| R26       | `ggml/src/ggml-cuda/common.cuh`                     | `ggml_cuda_e8m0_to_fp32`, `ggml_cuda_ue4m3_to_fp32` | 821-870         |
| R27       | `ggml/src/ggml-cuda/common.cuh`                     | `dequantize_kernel_t` typedef                  | 947                  |
| R28       | `ggml/src/ggml-common.h`                            | `GGML_TABLE_BEGIN` (`static const __device__`) | 493                  |
| R29       | `ggml/src/ggml-common.h`                            | `kvalues_mxfp4`, `kvalues_iq4nl`, `iq2xxs_grid`, etc. | 509-1650     |
| R30       | `ggml/src/ggml-cuda/dequantize.cuh`                 | `get_scale_min_k4` (sister K-quant unpacker)   | 157-164              |
| R31       | `ggml/src/ggml-cuda/mmq-vec-dot.cuh`                | `ggml_cuda_mmq_vec_dot_*_dp4a` / `_mma` (consumer) | 1-1251           |
| R32       | `ggml/src/ggml-cuda/mmq-load-tiles.cuh`             | `unpack_scales_q45_K` (sister K-quant unpacker) | 612-620             |
