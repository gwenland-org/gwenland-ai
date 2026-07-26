# ARTX02 — x86 AVX-512 / IceLake Quantized Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (AVX-512 kernel layer), `GATE` (kernel-selection contract)

---

## 1. Executive Summary

The x86 quantized kernels in llama.cpp live in two files: `arch/x86/quants.c`
(the per-block vecdot kernels invoked by the type-traits table) and
`arch/x86/repack.cpp` (the 8×8 batched GEMV/GEMM kernels that consume
"repacked" `block_q4_0x8` / `block_q8_0x4` weight layouts). A third file,
`llamafile/sgemm.cpp`, supplies a `tinyBLAS`-based FP32/FP16/BF16/Q8_0/Q4_0
GEMM that is compiled in alongside the IceLake kernels.

The single most important observation is that **`quants.c` contains no
native 512-bit AVX-512 kernels**. Every `ggml_vec_dot_*` function uses the
`#if defined(__AVX2__)` path with 256-bit `__m256`/`__m256i` accumulators
even when `__AVX512F__` is defined. The *only* AVX-512-specific instruction
used inside `quants.c` is `_mm256_dpbusd_epi32` (AVX512_VNNI + AVX512_VL),
and it is invoked at 256-bit width through the helper
`mul_sum_us8_pairs_float` (`quants.c:105`). In other words, the IceLake
build reaps the VNNI throughput benefit on the int8 dot product, but it
does **not** double the SIMD width by going to 512 bits.

True 512-bit AVX-512 execution exists only in:

1. `repack.cpp` — the `#if defined(__AVX512BW__) && defined(__AVX512DQ__)`
   block (`repack.cpp:663` and `repack.cpp:2077`) instantiates full
   `__m512`/`__m512i` accumulators (16 registers) for the 8×8 batched
   Q4_0/Q4_K/IQ4_NL/MXFP4/Q2_K GEMV and GEMM kernels. These are the
   actual IceLake fast path for prompt-processing shapes.
2. `llamafile/sgemm.cpp` — `tinyBLAS<16, __m512, __m512, …>` for F32/F16
   GEMM, `tinyBLAS<32, __m512, __m512bh, …>` for BF16 GEMM with
   `_mm512_dpbf16_ps` native BF16 dot product.

For GwenLand the decisions worth **ADOPT**ing are: the AVX512_VNNI 256-bit
helper pattern (`mul_sum_us8_pairs_float`), the multi-binary dispatch
scoring scheme (`ggml_backend_cpu_x86_score`), and the 8×8 batched GEMM
template in `repack.cpp` (when AVX512_BW+DQ are present). The decisions
worth **REJECT**ing are the absence of native 512-bit vecdot kernels and
the absence of AVX-512 FP16 dot products. The decisions worth **MONITOR**ing
are AVX-VNNI_INT8 (`_mm256_dpbssd_epi32`) which is wired in but effectively
bypassed by the sign-trick used for signed int8 inputs.

---

## 2. Purpose

Provide the AVX-512 / IceLake quantized-kernel layer for `glproc`. This
layer is responsible for:

* `vec_dot` kernels invoked through `type_traits_cpu[type].vec_dot` for
  every supported quant format (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K,
  Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M,
  IQ4_NL, IQ4_XS, MXFP4, NVFP4, TQ1_0, TQ2_0, Q1_0).
* 8×8 batched GEMV/GEMM fast paths for the four most common quants
  (Q4_0, Q4_K, IQ4_NL, MXFP4) and Q2_K, via "repacked" weight layouts.
* `llamafile_sgemm` — a tinyBLAS GEMM used for prompt-processing when
  A/B types are F32/F16/BF16/Q8_0/Q4_0/IQ4_NL.

It is **not** responsible for: graph scheduling (ARTX01), AMX tile-based
matmul (the `amx/` subdirectory, a separate ARTX), ARM kernels (ARTX04),
or elementwise ops (ARTX06).

---

## 3. Source Files

| File                                          | Lines | Role                                                                         |
| --------------------------------------------- | ----- | --------------------------------------------------------------------------- |
| `ggml/src/ggml-cpu/arch/x86/quants.c`         | 4108  | Per-block `vec_dot` kernels for every quant type; AVX2/AVX/SSSE3/scalar ladders |
| `ggml/src/ggml-cpu/arch/x86/repack.cpp`       | 6407  | 8×8 batched GEMV/GEMM for Q4_0/Q4_K/IQ4_NL/MXFP4/Q2_K with `__m512` AVX-512 BW+DQ paths |
| `ggml/src/ggml-cpu/llamafile/sgemm.cpp`       | 4058  | tinyBLAS GEMM templates: F32, F16, BF16, Q8_0, Q4_0, IQ4_NL (AVX, AVX2, AVX-512, BF16) |
| `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`    | 327   | CPUID + multi-binary score function (audited in ARTX01, referenced here)    |
| `ggml/src/ggml-cpu/simd-mappings.h`           | 1318  | `GGML_CPU_FP16_TO_FP32` / `GGML_CPU_E8M0_TO_FP32_HALF` / `GGML_CPU_UE4M3_TO_FP32` macros — x86 uses F16C intrinsics or LUTs |

> AMX tile-based code lives in `ggml/src/ggml-cpu/amx/{amx.cpp,mmq.cpp,
> mmq.h,common.h,amx.h}`. It is conditionally compiled when
> `__AMX_INT8__ && __AVX512VNNI__` are defined and registers itself as a
> `tensor_traits` extra buffer type (`amx/amx.cpp:19-42`). AMX repacks
> weights via `ggml_backend_amx_convert_weight` (`amx/amx.cpp:67-77`).
> The full AMX audit is a separate document; this audit only notes the
> boundary.

---

## 4. Architecture Overview

```
              ┌──────────────────────────────────────────────────────────┐
              │   type_traits_cpu[type].vec_dot  (ggml-cpu.c:214)        │
              │   type_traits_cpu[type].from_float                      │
              │   type_traits_cpu[type].nrows  (always 1 on x86)        │
              └──────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼─────────────────────┐
                ▼                                          ▼
   ┌────────────────────────┐                ┌─────────────────────────────┐
   │ arch/x86/quants.c      │                │ arch/x86/repack.cpp         │
   │ ───────────────        │                │ ──────────────              │
   │ ggml_vec_dot_q4_0_q8_0 │                │ ggml_gemv_q4_0_8x8_q8_0    │
   │ ggml_vec_dot_q4_K_q8_K │                │ ggml_gemm_q4_0_8x8_q8_0    │
   │ ggml_vec_dot_q6_K_q8_K │                │ ggml_gemv_q4_K_8x8_q8_K    │
   │ ggml_vec_dot_iq4_xs…   │                │ ggml_gemm_q4_K_8x8_q8_K    │
   │ …21 total vec_dot…     │                │ …Q4_0/Q4_K/IQ4_NL/MXFP4/Q2_K│
   │                        │                │                             │
   │ Path: AVX2 (256-bit)   │                │ Path: AVX-512 BW+DQ (512-bit)│
   │ Helper:                │                │ 16× __m512 accumulators     │
   │  mul_sum_us8_pairs_    │                │ __mmask blends              │
   │  float  (uses VNNI at  │                │                             │
   │  256-bit if AVX512_VNNI│                │ Repacked weight layouts:    │
   │  +VL, AVX-VNNI on ADL, │                │  block_q4_0x8, block_q8_0x4 │
   │  AVX-VNNI_INT8 on GNR) │                │  block_iq4_nlx8, …          │
   └────────────────────────┘                └─────────────────────────────┘
                ▲
                │ optional fast path for prompt processing
                │
   ┌────────────────────────────────────────────────────────────────────┐
   │ llamafile/sgemm.cpp  — tinyBLAS Q0_AVX + tinyBLAS<16, __m512>      │
   │  F32, F16 (cvtph→ps), BF16 (cvtph→ps or _mm512_dpbf16_ps)         │
   │  Q8_0, Q4_0, IQ4_NL (via tinyBLAS_Q0_AVX template)                │
   │  Selected when nth > 1 && n >= 2 && Ctype == F32                   │
   └────────────────────────────────────────────────────────────────────┘

   ┌────────────────────────────────────────────────────────────────────┐
   │ amx/amx.cpp (out of scope)                                         │
   │  Registers a tensor_traits that overrides compute_forward(MUL_MAT) │
   │  when __AMX_INT8__ && __AVX512VNNI__ are defined.                  │
   │  Repacks weights with ggml_backend_amx_convert_weight.             │
   └────────────────────────────────────────────────────────────────────┘
```

Key design points:

* **Two parallel code paths.** The vecdot kernels in `quants.c` use 256-bit
  vectors; the batched GEMM kernels in `repack.cpp` use 512-bit vectors
  when AVX512_BW+DQ are defined. They are *not* the same kernel in two
  widths — they are different algorithms selected by the type-traits table
  and the optional `gemm`/`gemv` overrides respectively.
* **No runtime CPU detection inside any kernel.** Every ISA branch is
  resolved at *compile time* via `#if defined(__AVX512*)`. The runtime
  selection happens once, in `ggml_backend_cpu_x86_score`
  (`cpu-feats.cpp:263`), to pick which compiled `.so` is loaded. This is
  the multi-binary dispatch model (ARTX01-F12).
* **The "IceLake" build.** Per `cpu-feats.cpp:297-316`, the AVX-512
  variant requires `GGML_AVX512` (F+CD+VL+DQ+BW) and optionally adds
  `GGML_AVX512_VBMI`, `GGML_AVX512_BF16`, `GGML_AVX512_VNNI`. An IceLake
  client part (e.g. i7-1065G7) has F+DQ+BW+CD+VL+VBMI+VNNI but no BF16;
  Sapphire Rapids / Granite Rapids adds BF16, FP16, and AMX. The score
  function differentiates them.
* **AVX-512 FP16 is detected but never used.** `cpu-feats.cpp:81` defines
  `AVX512_FP16()` and the score function reads it, but **no kernel in
  `quants.c`, `repack.cpp`, or `llamafile/sgemm.cpp` uses an AVX-512 FP16
  intrinsic**. The `_mm512_*ph*_ps` family does not appear anywhere in
  the audited files.

---

## 5. Execution Flow

### 5.1 Per-block vecdot (the `quants.c` path)

1. `ggml_compute_forward_mul_mat_one_chunk` (`ggml-cpu.c:1164`) walks
   tiles `(iir0, iir1)` and for each weight row calls
   `vec_dot(qk, &s, sizeof(float), src0_row, nb01, src1_row, nb11, nrows)`.
2. `vec_dot` is a function pointer from `type_traits_cpu[type].vec_dot`
   (`ggml-cpu.c:1181-1182`). On x86 the linker resolves it to one of the
   `ggml_vec_dot_*` functions in `arch/x86/quants.c`.
3. Inside the kernel: `assert(nrc == 1)`. The `#if defined(__AVX2__)`
   branch runs (lines 718-741 for Q4_0, 1325-1342 for Q8_0, 2057-2120
   for Q4_K, etc.). The helper `mul_sum_i8_pairs_float` is called once
   per 32-byte block; it dispatches at compile time to either:
   * `_mm256_dpbssd_epi32` if `__AVXVNNIINT8__` (line 123-126),
   * `_mm256_maddubs_epi16` + `sum_i16_pairs_float` otherwise
     (line 128-133), where the inner `mul_sum_us8_pairs_float` itself
     dispatches to `_mm256_dpbusd_epi32` if AVX512_VNNI+VL, or
     `_mm256_dpbusd_avx_epi32` if AVX-VNNI, or the maddubs ladder.
4. The kernel writes a single `float *s` result.

### 5.2 Batched GEMM (the `repack.cpp` path)

1. The `extra_buffer_type` mechanism (ARTX01-F04) detects that src0 was
   allocated with the "repack" buffer type and calls
   `tensor_traits::compute_forward` instead of the default
   `ggml_compute_forward_mul_mat`. (The full plumbing lives in
   `repack.h` / `repack.cpp` registration; not audited here.)
2. `ggml_gemm_q4_0_8x8_q8_0` (`repack.cpp:2026`) is invoked. It dispatches
   to `gemm_q4_b32_8x8_q8_0_lut_avx<block_q4_0x8>` (line 521+).
3. Inside, if `__AVX512BW__ && __AVX512DQ__` are defined, the 512-bit
   block runs (line 663-1096). It uses 16 `__m512` accumulators
   (`__m512 acc_rows[16]`, line 687-690), one per output row of a 16-row
   tile. The 4-bit nibbles are expanded to 8-bit via
   `_mm512_shuffle_epi8` against a `signextendlut`, then `_mm512_maddubs_epi16`
   + `_mm512_madd_epi16` form the i32 dot products. Reduction is via
   `_mm512_fmadd_ps` against the per-block d-scale.
4. If only AVX2 is available, the 256-bit block (line 1098+) runs with
   16 `__m256` accumulators instead — same algorithm, half the lanes.

### 5.3 `llamafile_sgemm` (the FP16/BF16/Q8 fast path)

1. `ggml_compute_forward_mul_mat` calls `llamafile_sgemm(...)` early in
   the matmul path (per ARTX01 §5.5).
2. `llamafile_sgemm` (`sgemm.cpp:3699`) inspects `(Atype, Btype, Ctype)`
   and dispatches to a `tinyBLAS<N, VecA, VecB, …>` template:
   * F32×F32 → `tinyBLAS<16, __m512, __m512, …>` on AVX-512F (line 3727).
   * BF16×BF16 → `tinyBLAS<32, __m512, __m512bh, …>` on AVX-512 BF16
     (line 3790) using `_mm512_dpbf16_ps`. Falls back to
     `tinyBLAS<16, __m512, __m512, …>` (FP32 emulation) on AVX-512F
     without BF16.
   * F16×F16 → `tinyBLAS<16, __m512, __m512, …>` always; F16 is converted
     to F32 via `_mm512_cvtph_ps` then operated in F32. No native F16 dot.
   * Q8_0/Q4_0/IQ4_NL → `tinyBLAS_Q0_AVX<>` (line 3939, 3976) with
     `VECTOR_REGISTERS == 32` on AVX-512, enabling 4×4 tile shapes.

---

## 6. Data Layout

### 6.1 Quantized weight blocks

`quants.c` consumes the standard ggml block layouts (`block_q4_0`,
`block_q8_0`, `block_q4_K`, `block_q6_K`, `block_iq4_xs`, …) defined in
`ggml-common.h`. Block sizes: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 = 32 elements;
Q2_K/Q3_K/Q4_K/Q5_K/Q6_K = 256 elements (`QK_K`); IQ4_XS = 256;
IQ4_NL = 32. Each block carries one or more fp16 scales (and `m`/`s`
zero-point fields for the `_1` variants).

### 6.2 "Repacked" 8×8 batched layouts (repack.cpp only)

`repack.cpp` consumes `block_q4_0x8`, `block_q8_0x4`, `block_iq4_nlx8`,
`block_mxfp4x8`, `block_q4_Kx8`, `block_q8_Kx4`, `block_q2_Kx8` — these
are *interleaved* weight layouts where 8 (or 4) original blocks are
shuffled together to allow a single vector load to deliver 8 different
weight rows' nibbles into one `__m512` lane. The conversion happens at
weight-load time (out of scope; see `ggml_backend_amx_convert_weight` in
the AMX directory and the `ggml_quantize_mat_q8_0_4x8` /
`ggml_quantize_mat_q8_K_4x8` quantizers in `repack.cpp:178, 290`).

### 6.3 Activation conversion (`wdata`)

When `src1->type != vec_dot_type`, the matmul path converts src1 into
`params->wdata` once per matmul (ARTX01 §6.2). For most quants this
means F32 → Q8_0 (or Q8_K) via `quantize_row_q8_0` (`quants.c:302`,
AVX2 path) or `quantize_row_q8_K` (`quants.c:505`, **scalar placeholder**,
see F12).

---

## 7. Memory Layout

### 7.1 Per-block layout in `quants.c`

Inside each kernel, the input blocks are streamed sequentially:

```
x = vx;  // block_q4_0 []  (size = nb * sizeof(block_q4_0))
y = vy;  // block_q8_0 []  (size = nb * sizeof(block_q8_0))
for (int ib = 0; ib < nb; ++ib) {
    _mm256_loadu_si256(x[ib].qs);  // 32-byte aligned-not load
    _mm256_loadu_si256(y[ib].qs);
}
```

No software prefetching is inserted in the AVX2 vecdot kernels (the SSSE3
Q4_0 path at `quants.c:782, 800` does use `_mm_prefetch`, but the AVX2
path drops it). The code relies on the hardware prefetcher to spot the
sequential 18-byte (Q4_0) and 34-byte (Q8_0) strides.

### 7.2 8×8 batched layout in `repack.cpp`

Each iteration of the inner loop loads `4 × 32 bytes = 128 bytes` from
each of 4 weight blocks (`b_ptr_0[b].qs`, `b_ptr_0[b].qs + 32`, …) and
`4 × 32 bytes = 128 bytes` from each of 4 activation blocks. The 16
`__m512` accumulators (`acc_rows[16]`) live entirely in registers across
the `for (b = 0; b < nb; ++b)` loop. Store happens once per (y, x) tile,
not per b.

### 7.3 Constant tables

The following tables are referenced from `quants.c` (defined in
`ggml-common.h` or `quants.c` itself):

* `kvalues_iq4nl[16]` — 4-bit → int8 LUT for IQ4_NL, IQ4_XS
  (`quants.c:3939, 4019`).
* `kvalues_fp4[16]` — 4-bit → int8 LUT for MXFP4 / NVFP4
  (`quants.c:937, 967, 1021, 1067`).
* `keven_signs_q2xs[1024]` — 1024-byte sign table for IQ2_XXS/XS
  (`quants.c:2624`).
* `get_scale_shuffle_q3k`, `get_scale_shuffle_k4`, `get_scale_shuffle`
  — `__m256i`/`__m128i` shuffle masks for Q3_K/Q4_K/Q6_K scale broadcast
  (`quants.c:518-552`). Stored as static `uint8_t[]` and loaded with
  `_mm256_loadu_si256` / `_mm_loadu_si128`.

### 7.4 FP16 / E8M0 / UE4M3 LUTs

`simd-mappings.h:117-125` declares three LUTs in `ggml-cpu.c`:
`ggml_table_f32_f16[1<<16]` (256 KB, used only when `__F16C__` is
undefined), `ggml_table_f32_e8m0_half[1<<8]` (1 KB, used by MXFP4
kernels at `quants.c:957-958, 992`), and `ggml_table_f32_ue4m3[1<<8]`
(1 KB, used by NVFP4 kernel at `quants.c:1052-1055`). On F16C-capable
x86 (always, since AVX implies F16C), the FP16 conversion goes through
`_cvtsh_ss` (`simd-mappings.h:61`), not the LUT.

---

## 8. Parallelism Strategy

The kernels themselves are single-threaded. Threading is layered above,
in `ggml_compute_forward_mul_mat` (ARTX01 §5.5, §8.4): dynamic chunk
stealing on `current_chunk` atomic for the per-block vecdot path, and a
similar split for the batched GEMM path. No SIMD-level parallelism
decision is made by the kernels; each kernel call computes exactly one
output element (per-block vecdot) or a fixed 16-row × 8-col tile
(`repack.cpp`).

The `nrc == 1` assertion in every `quants.c` vecdot (e.g.
`quants.c:706, 863, 1325`) means **x86 kernels consume exactly one
weight row × one activation row per call**. ARM I8MM kernels can set
`nrows = 2` and consume two rows in parallel from the same activation
block (ARTX01 §11.1). x86 has no equivalent.

---

## 9. SIMD / GPU Strategy

This is the meatiest section. The x86 quantized-kernel SIMD strategy is
fragmented across three files with different ISA targets and different
vector widths.

### 9.1 SIMD feature matrix (per file, per feature)

| Feature                | `quants.c`                          | `repack.cpp`                       | `llamafile/sgemm.cpp`              |
| ---------------------- | ----------------------------------- | ---------------------------------- | ---------------------------------- |
| SSE / SSSE3            | Ladder fallback for Q4_0/Q4_1/Q5_0  | n/a (file compiled only on AVX2+)  | `__m128` hsum, fallback load       |
| AVX (256-bit, no FMA)  | Ladder fallback (line 742, 875)     | n/a                                | `tinyBLAS<8, __m256, …>`           |
| AVX2 + FMA + F16C      | **Main path** for every vecdot      | 256-bit `__m256` GEMV/GEMM         | `tinyBLAS<8, __m256, …>`           |
| AVX-VNNI (no AVX-512)  | `_mm256_dpbusd_avx_epi32` (line 110)| n/a                                | `_mm256_dpbusd_avx_epi32` (line 1759) |
| AVX-512 F + DQ + BW + VL | No 512-bit use                    | **Full 512-bit GEMV/GEMM** (line 663) | `tinyBLAS<16, __m512, …>` (line 3727) |
| AVX-512 VNNI (256-bit VL) | `_mm256_dpbusd_epi32` (line 108) | `_mm512_dpbusd_epi32` (line 125)   | `_mm256_dpbusd_epi32` (line 1757)  |
| AVX-512 BF16           | Not used                            | Not used                           | `_mm512_dpbf16_ps` (line 154, 3790) |
| AVX-512 FP16           | **Not used**                        | **Not used**                       | **Not used**                       |
| AVX-VNNI_INT8 (Granite Rapids) | `_mm256_dpbssd_epi32` (line 125) | Not used                  | Not used                           |
| AMX (tile)             | n/a (separate file)                 | n/a (separate file)                | n/a (separate file)                |

### 9.2 The vecdot helper: `mul_sum_us8_pairs_float` (the only AVX-512 in quants.c)

```c
// quants.c:105-119
static inline __m256 mul_sum_us8_pairs_float(const __m256i ax, const __m256i sy) {
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#elif defined(__AVXVNNI__)
    const __m256i zero = _mm256_setzero_si256();
    const __m256i summed_pairs = _mm256_dpbusd_avx_epi32(zero, ax, sy);
    return _mm256_cvtepi32_ps(summed_pairs);
#else
    const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    return sum_i16_pairs_float(dot);
#endif
}
```

This is the *only* AVX-512-specific code in `quants.c`. It runs at 256-bit
width even when AVX-512 F is available, because the surrounding vecdot
kernels all use `__m256` accumulators. The benefit is that the inner
2-instruction sequence (`maddubs_epi16` → `madd_epi16` via
`sum_i16_pairs_float`) collapses to one instruction (`dpbusd_epi32`) on
IceLake and later. The signed-input wrapper `mul_sum_i8_pairs_float`
(line 122) optionally uses `_mm256_dpbssd_epi32` (AVX-VNNI_INT8, Granite
Rapids), but otherwise it falls back to the abs+sign trick to reuse
`mul_sum_us8_pairs_float`.

### 9.3 Tile blocking and accumulator count

| Kernel                                 | File:Line          | Accumulator count (AVX2 path)        | Tile shape             |
| -------------------------------------- | ------------------ | ------------------------------------ | ---------------------- |
| `ggml_vec_dot_q4_0_q8_0`               | `quants.c:718`     | 1 × `__m256`                         | 32 elements/block      |
| `ggml_vec_dot_q4_1_q8_1`               | `quants.c:875`     | 1 × `__m256`                         | 32 elements/block      |
| `ggml_vec_dot_q8_0_q8_0`               | `quants.c:1325`    | 1 × `__m256`                         | 32 elements/block      |
| `ggml_vec_dot_q5_0_q8_0`               | `quants.c:1159`    | 1 × `__m256`                         | 32 elements/block      |
| `ggml_vec_dot_q4_K_q8_K`               | `quants.c:2057`    | 1 × `__m256` + 1 × `__m128` (mins)   | 64 elements/inner iter |
| `ggml_vec_dot_q6_K_q8_K`               | `quants.c:2439`    | 1 × `__m256` + 4 parallel p16        | 128 elements/inner iter|
| `ggml_vec_dot_q2_K_q8_K`               | `quants.c:1586`    | 1 × `__m256`                         | 128 elements/inner iter|
| `ggml_vec_dot_q3_K_q8_K`               | `quants.c:1782`    | 1 × `__m256`                         | 128 elements/inner iter|
| `ggml_vec_dot_iq4_nl_q8_0`             | `quants.c:3937`    | 2 × `__m256` (`accum1`, `accum2`)    | 2 blocks/iter          |
| `ggml_vec_dot_iq4_xs_q8_K`             | `quants.c:4017`    | 1 × `__m256`                         | 2 blocks/iter          |
| `ggml_vec_dot_mxfp4_q8_0`              | `quants.c:935`     | 2 × `__m256`                         | 2 blocks/iter          |
| `ggml_vec_dot_nvfp4_q8_0`              | `quants.c:1019`    | 1 × `__m256`                         | 1 block/iter           |
| `gemm_q4_b32_8x8_q8_0_lut_avx` (512)   | `repack.cpp:663`   | **16 × `__m512`**                    | 16 rows × 8 cols       |
| `gemm_q4_b32_8x8_q8_0_lut_avx` (256)   | `repack.cpp:1098`  | 16 × `__m256`                        | 16 rows × 8 cols       |
| `tinyBLAS_Q0_AVX` (AVX-512)            | `sgemm.cpp:1351`   | 4×4 tile (32 regs) / 4×2 (16 regs)   | shape-adaptive         |
| `tinyBLAS<16, __m512>` F32 GEMM        | `sgemm.cpp:3727`   | `VECTOR_REGISTERS == 32` enables 4×4 | shape-adaptive         |

The vecdot kernels use a single accumulator (with the exception of
IQ4_NL/MXFP4 which use two for 2-block-per-iter unrolling). This is a
*dependence-chain* pattern: each `fmadd_ps` depends on the previous
iteration's accumulator. The Q4_K/Q6_K kernels partially break this by
computing 4 partial `p16_*` products in parallel before summing into
the single accumulator, but the final accumulator update is still serial.

The batched GEMM kernels use **16 independent accumulators**, breaking
the dependence chain completely. This is why the batched path can sustain
much higher throughput on IceLake than the per-block vecdot path.

### 9.4 Mask register usage (`__mmask16`)

`__mmask16` / `__mmask32` / `__mmask64` appear in the audited files only
in `repack.cpp`:

* `_mm512_mask_blend_epi32(0xCCCC, …, …)` (`repack.cpp:862-865`) — used
  to combine two `__m512i` dot-product halves into a single row-major
  output, selecting 4 lanes from each. This is the only mask-register
  use in the IceLake GEMM kernels.
* `_mm512_movepi8_mask` (`repack.cpp:139`) in the helper
  `mul_sum_i8_pairs_acc_int32x16` — used to compute the sign mask of
  int8 lanes for the abs+sign trick at 512-bit width.

`quants.c` uses no mask registers. `llamafile/sgemm.cpp` uses no mask
registers either. This is a missed opportunity: per-lane masking would
allow handling the `n % 32 != 0` tail without falling back to scalar,
but the current code instead relies on either the assertion
`assert(n % qk == 0)` (K-quants) or a scalar tail loop (Q4_0 line 840,
IQ4_NL line 3992).

### 9.5 VNNI / AVX-VNNI / AVX-VNNI_INT8 usage

| Instruction                | ISA                       | Where used                                    | Effect                                            |
| -------------------------- | ------------------------- | --------------------------------------------- | ------------------------------------------------- |
| `_mm256_dpbusd_epi32`      | AVX-512 VNNI + VL         | `quants.c:108`, `sgemm.cpp:1757`              | u8×i8→i32 dot product at 256-bit width            |
| `_mm256_dpbusd_avx_epi32`  | AVX-VNNI (Alder Lake)     | `quants.c:112`, `sgemm.cpp:1759`              | same, for non-AVX-512 CPUs                        |
| `_mm256_dpbssd_epi32`      | AVX-VNNI_INT8 (Granite Rapids) | `quants.c:125`                           | i8×i8→i32 dot product (signed)                    |
| `_mm512_dpbusd_epi32`      | AVX-512 VNNI              | `repack.cpp:125`                              | 512-bit u8×i8→i32 (only in helper, not in main loop) |
| `_mm512_dpbf16_ps`         | AVX-512 BF16              | `sgemm.cpp:154, 3790`                         | BF16×BF16→F32 dot product                         |

The signed-int8 path (`_mm256_dpbssd_epi32`) is wired into
`mul_sum_i8_pairs_float` (`quants.c:122-134`) under
`#if __AVXVNNIINT8__`. However, every vecdot kernel that calls it first
either (a) expands nibbles to bytes that are already in `[-8, 7]` (Q4_0
at `quants.c:730-731`) — so the inputs are *signed* but small — or
(b) calls `mul_add_epi8` directly (Q4_K line 2101, IQ4_XS line 4038),
which uses the unsigned `maddubs` + sign-extend pattern. The net effect
is that the `dpbssd` path is reachable only on Granite Rapids, and even
there it competes with the `dpbusd`+sign-extend path which is what every
other CPU uses. **The signed-VNNI path is plumbed but its performance
benefit over the established unsigned-VNNI+sign-extend pattern is
unverified** (see Unknowns U1).

### 9.6 BF16 usage (AVX-512 BF16)

Native BF16 dot product (`_mm512_dpbf16_ps`) is used **only in
`llamafile/sgemm.cpp`** (line 154, instantiated at line 3790). It is
selected when `Atype == GGML_TYPE_BF16 && Btype == GGML_TYPE_BF16 &&
__AVX512BF16__` is defined. The `tinyBLAS<32, __m512, __m512bh, …>`
template uses 32-wide BF16 lanes (vs. 16-wide FP32 lanes), doubling
throughput.

`quants.c` and `repack.cpp` do **not** use BF16 at all. Quantized
kernels accumulate in FP32 always.

### 9.7 FP16 usage (AVX-512 FP16)

**No AVX-512 FP16 intrinsics are used anywhere in the audited files.**
The `_mm512_*ph*_ps` family does not appear. FP16 is always converted to
FP32 first via:

* `_mm256_cvtph_ps` (F16C, used in `repack.cpp:30, 64, 86` helpers)
* `_mm512_cvtph_ps` (used in `sgemm.cpp:364`)
* `GGML_CPU_FP16_TO_FP32` macro (used pervasively in `quants.c` for
  per-block scale broadcast — 100+ call sites)

`cpu-feats.cpp:81` *does* detect `AVX512_FP16()` and the score function
counts it (`cpu-feats.cpp` does not have a separate `#ifdef` for FP16,
so the IceLake-variant `.so` may be compiled with `-mavx512fp16` and the
CPU will be detected as FP16-capable, but no kernel uses it).

This is a **MISSING_FEATURE**: on Granite Rapids / Sierra Forest / Arrow
Lake, native FP16 dot product would double throughput for F16 weights
relative to the current convert-then-multiply pattern, but the code
prefers the convert-then-multiply path everywhere.

### 9.8 AMX (high-level only — full audit is separate)

The `amx/` directory is conditionally compiled when
`__AMX_INT8__ && __AVX512VNNI__` (`amx/amx.cpp:19`). It registers a
`tensor_traits` (`amx/amx.cpp:23-36`) that overrides
`compute_forward` for `GGML_OP_MUL_MAT`. Weights allocated through the
AMX buffer type are repacked via `ggml_backend_amx_convert_weight`
(`amx/amx.cpp:67-77`) into the AMX-friendly 16-row × 64-byte tile layout.
At compute time, `ggml_backend_amx_mul_mat` runs the tile-based matmul
using `_tile_*` intrinsics (in `amx/mmq.cpp`, not audited here).

AMX is *not* invoked from `quants.c` or `repack.cpp`. The selection
happens at buffer-allocation time: if the model loaded into an AMX buffer,
the AMX `tensor_traits::compute_forward` wins over the default dispatch.
The IceLake path (no AMX) does not see this.

---

## 10. Quantization Strategy

`quants.c` provides SIMD `from_float` (quantize) kernels for two
activation formats only:

* `quantize_row_q8_0` (`quants.c:302-398`) — AVX2 path at line 309-391,
  scalar fallback at line 393-397. Uses `_mm256_max_ps` for max-abs,
  `_mm256_round_ps` for round-to-nearest, `_mm256_cvtps_epi32` for
  float→int, then `_mm256_packs_epi32` + `_mm256_packs_epi16` +
  `_mm256_permutevar8x32_epi32` to pack int32 → int8 in the correct
  lane order. The permute constant `_mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7)`
  (`quants.c:364`) fixes the well-known AVX2 `packs` lane-shuffle.
* `quantize_row_q8_1` (`quants.c:400-501`) — same structure as Q8_0 plus
  an extra `hsum_i32_8` to compute the block sum `s` field
  (`quants.c:452`).
* `quantize_row_q8_K` (`quants.c:505-507`) — **placeholder**. Always
  calls `quantize_row_q8_K_ref`. There is no SIMD optimization for Q8_K
  activation quantization on x86. See Finding F12.

All other `from_float` functions for the K-quants and I-quants are not
defined in this file; they live in the generic `quants.c` (parent
directory). The I-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M) have
`from_float = NULL` per ARTX01 — they are inference-only.

The `vec_dot` strategy across all quants is consistent:

1. Load one weight block (16-32 bytes via `_mm256_loadu_si256` or
   `_mm_loadu_si128`).
2. Expand the packed nibbles/bits to bytes via shift+mask+shuffle
   (`bytes_from_nibbles_32`, `bytes_from_bits_32`, `_mm_shuffle_epi8` LUT).
3. Load the corresponding Q8 activation block (32 bytes via
   `_mm256_loadu_si256`).
4. Compute the int32 dot product via `mul_sum_i8_pairs_float` /
   `mul_add_epi8` + `_mm256_madd_epi16` /
   `_mm256_maddubs_epi16` + `_mm256_madd_epi16` (with optional
   AVX-VNNI / AVX-512 VNNI acceleration).
5. Multiply by the per-block fp16 scale (broadcast to `__m256` via
   `_mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * …)`).
6. FMA into the single `__m256 acc`.
7. After all blocks: `hsum_float_8(acc)` and store.

K-quants add a scale-broadcast shuffle (`get_scale_shuffle_k4`,
`get_scale_shuffle_q3k`) to apply 8 different per-block-32 scales across
the 8 lanes of a `__m256i`. Q4_K additionally computes a `dmin * summs`
correction term in a separate `__m128 acc_m` accumulator (lines 2062,
2082-2084).

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

Every vecdot kernel in `quants.c` accumulates into a single `__m256 acc`
across blocks, then horizontally reduces via `hsum_float_8` at the end.
This reassociates the sum: the result differs from a strict left-to-right
scalar sum at the ULP level. The reduction order is deterministic for a
fixed `n` and a fixed compile-time ISA selection, but combined with
dynamic chunk stealing (ARTX01-F06) it is non-deterministic across runs
with `nth > 1`.

The 8×8 batched kernels in `repack.cpp` reduce 16 `__m512` accumulators
at the end of the tile, then horizontally reduce each `__m512` via
implicit `_mm512_reduce_add_ps` (or `_mm512_storeu_ps` + scalar sum).
Same reassociation consequence.

### 11.2 Approximate math

* **`quad_fp16_delta_float`** (`quants.c:265-269`): for the AVX (non-AVX2)
  Q4_0/Q4_1/Q8_0/IQ4_NL vecdot kernels, the FP16 scale is converted to
  FP32 *as a scalar* via `GGML_CPU_FP16_TO_FP32(x0)`, multiplied as a
  scalar, then broadcast with `_mm_set1_ps`. This is not approximate
  (the F16C conversion is exact), but it is a serial dependency that
  halves throughput on the AVX path. On the AVX2+ path (line 725) the
  same conversion is done once per block and broadcast.
* **No transcendental approximations** in any quantized kernel. All
  activations of that kind live in `ops.cpp` (ARTX06).
* **E8M0 / UE4M3 LUTs** (`simd-mappings.h:129, 136`): MXFP4 / NVFP4
  scales are looked up from 256-entry LUTs. The LUTs are populated to
  be bit-exact equivalents of the format's specification; no
  approximation beyond the format itself.

### 11.3 Precision reduction

* All K-quant and I-quant kernels convert the F32 activation to Q8_0 or
  Q8_K once up-front (ARTX01 §6.2). This is a lossy conversion before
  the dot product. It is the whole point of quantized inference.
* The dot product itself is computed in int32 (u8×i8 → i16 → i32), then
  converted to FP32 (`_mm256_cvtepi32_ps`) for the scale multiply.
  No precision loss beyond the format.
* The 8×8 batched GEMM kernels in `repack.cpp` compute the entire
  16-row × 8-col tile in int32, then convert to FP32 once at the end
  for the scale multiply. Same precision profile as the vecdot path.
* `llamafile_sgemm` BF16 path uses `_mm512_dpbf16_ps` which accumulates
  in FP32 (per Intel spec). No precision reduction beyond BF16 inputs.

### 11.4 Non-deterministic reductions

Same as ARTX01 §11.4: matmul output is deterministic bit-for-bit only
when `nth = 1`. With `nth > 1`, dynamic chunk stealing + per-chunk
reassociation produces ULP-level variation. The kernels themselves
are deterministic; the non-determinism is in the chunk scheduler.

### 11.5 Atomic accumulation

None in any audited kernel. Output tiles are written by exactly one
thread each (chunk stealing assigns disjoint chunks). The 8×8 batched
kernels write 16×8 = 128 floats per tile, all to a contiguous block
owned by one thread.

### 11.6 Architecture-specific assumptions

* **`assert(nrc == 1)`** in every `quants.c` vecdot. x86 kernels do not
  support multi-row consumption; this is enforced at runtime. ARM I8MM
  kernels set `nrows = 2` and accept `nrc == 2`. The two paths produce
  *slightly different* results on the same input (ARTX01 §11.6).
* **`assert(n % qk == 0)`** in every kernel. Block-aligned lengths only.
  The Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/NVFP4 kernels handle a scalar tail
  loop for the `n % qk != 0` case (e.g. `quants.c:840-854`); the
  K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_*, IQ3_*, IQ1_*, IQ4_XS)
  call the generic `_generic` reference function instead
  (`quants.c:2204-2213, 2615-2620`).
* **AVX2 `packs` lane-shuffle**. The `_mm256_permutevar8x32_epi32`
  with constant `{0, 4, 1, 5, 2, 6, 3, 7}` at `quants.c:364, 463` is
  required because AVX2 `packs_epi32`/`packs_epi16` operate
  independently on the two 128-bit lanes, scrambling the natural order.
  This is a well-known AVX2 quirk. The code does not handle the case
  where the input is not a multiple of 32 elements at the AVX2 path
  (handled by the scalar tail).
* **`MM256_SET_M128I` macro** (`quants.c:26`) is a gcc-7 compatibility
  shim for `_mm256_set_m128i`. Assumes the compiler does not provide
  the intrinsic natively.
* **No OS-XSAVE check**. `cpu-feats.cpp:264` comment: "FIXME: this does
  not check for OS support". If the OS has not enabled AVX-512 context
  saving (`XCR0 |= 0xE0`), the CPUID bits may report AVX-512 support
  but executing AVX-512 instructions will fault. The score function
  trusts CPUID. (Cross-reference ARTX01-F12; this is a known upstream
  issue.)

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                | Where                                       | Notes                                                                                          |
| ------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| Multi-binary ISA dispatch                   | `cpu-feats.cpp:263-323`                     | Compile N `.so` per ISA target; runtime score picks best. Cross-ref ARTX01-F12.                |
| AVX-512 VNNI 256-bit dpbusd in vecdot helper| `quants.c:106-118`                          | Single-instruction u8×i8→i32 dot product at 256-bit width.                                     |
| AVX-VNNI fallback (Alder Lake)              | `quants.c:110-113`                          | `_mm256_dpbusd_avx_epi32` for non-AVX-512 VNNI CPUs.                                           |
| AVX-VNNI_INT8 signed dot                    | `quants.c:123-126`                          | `_mm256_dpbssd_epi32` for Granite Rapids. Effectively bypassed; see F04.                       |
| FP16→FP32 via F16C intrinsic                | `simd-mappings.h:61` (`_cvtsh_ss`)          | Replaces 256 KB LUT on F16C-capable x86.                                                        |
| E8M0 / UE4M3 LUTs                           | `simd-mappings.h:129, 136`                  | 1 KB each, faster than bit manipulation for MXFP4 / NVFP4 scales.                              |
| 8×8 batched GEMV/GEMM with 16 × `__m512`    | `repack.cpp:663-1096`                       | True 512-bit AVX-512 path. Breaks dependency chain with 16 independent accumulators.           |
| Mask-blend accumulator combination          | `repack.cpp:862-865`                        | `_mm512_mask_blend_epi32(0xCCCC, …)` to merge two `__m512i` halves in one instruction.         |
| Sign-extend LUT for signed nibbles          | `repack.cpp:567, 933`                       | Single `_mm512_shuffle_epi8` against `signextendlut` expands 4-bit → 8-bit signed.             |
| Scale-shuffle constants                     | `quants.c:518-552`                          | Pre-loaded `__m256i` / `__m128i` shuffle masks for per-block-32 scale broadcast in K-quants.   |
| `keven_signs_q2xs[1024]`                    | `quants.c:2624-2657`                        | Pre-computed 1024-byte sign table for IQ2_XXS/XS; indexed by 7-bit sign index.                 |
| 2-block-per-iter unrolling                  | `quants.c:944, 3945, 4029`                  | IQ4_NL / MXFP4 / IQ4_XS process 2 blocks per loop iter with 2 accumulators.                    |
| 4-way parallel p16 in Q6_K                  | `quants.c:2483-2500`                        | 4 partial `maddubs` + `madd` computed in parallel before summing into single acc.              |
| `tinyBLAS` shape-adaptive tile selection    | `sgemm.cpp:1376-1459`                       | `mnpack` switch on `MIN(m,4)<<4 | MIN(n,4)` to pick best (mc, nc) tile shape.                  |
| `VECTOR_REGISTERS == 32` wider tiles        | `sgemm.cpp:66-70`                           | AVX-512 / ARM NEON get 32 registers → 4×4 tiles; AVX2 gets 16 → 4×2 tiles.                     |
| BF16 native dot product                     | `sgemm.cpp:154, 3790`                       | `_mm512_dpbf16_ps` doubles BF16 GEMM throughput on AVX-512 BF16.                               |
| FMA via `_mm256_fmadd_ps`                   | `quants.c:738, 1339, 2113, 2505, …`         | FMA3 fused multiply-add for the scale × int32 → acc step.                                      |
| Aligned-not loads (`_mm256_loadu_si256`)    | throughout                                  | All loads are unaligned; allows blocks to be packed without padding.                           |
| Disposable threadpool                       | (cross-ref ARTX01)                          | Already audited.                                                                               |

### 12.2 Optimizations *not* present (worth noting)

* **No native 512-bit AVX-512 vecdot kernels in `quants.c`.** Every
  vecdot runs at 256-bit width even on IceLake. The 16× throughput
  argument for 512-bit accumulators (16 lanes vs. 8) is left on the
  table for the per-block path. See F01.
* **No AVX-512 FP16 dot products.** `_mm512_*ph*_ps` family unused.
  See F08.
* **No AVX-512 BF16 in `quants.c` or `repack.cpp`.** BF16 native dot
  only in `llamafile/sgemm.cpp`. See F07.
* **No software prefetching** in AVX2 vecdot kernels. The SSSE3 Q4_0
  path (`quants.c:782, 800`) uses `_mm_prefetch(..., _MM_HINT_T0)` but
  the AVX2 path (line 718+) does not — likely a regression when the
  AVX2 path was written, or because the hardware prefetcher is
  sufficient. Not measurable statically.
* **No persistent threads** in the kernel layer. Threads are managed
  above (ARTX01).
* **No kernel fusion.** Quantized vecdot is a leaf operation; no
  upstream fusion with bias-add or activation. (Cross-ref ARTX01-F08.)
* **No AMX in `quants.c` / `repack.cpp`.** AMX is a separate plugin
  (`amx/`).

---

## 13. Architectural Strengths

1. **AVX-512 VNNI helper is a clean compile-time dispatch.** The
   `mul_sum_us8_pairs_float` helper (`quants.c:105-119`) is a 3-way
   `#if` ladder that picks the best instruction for the build (AVX-512
   VNNI, AVX-VNNI, or scalar maddubs). Every vecdot kernel uses this
   helper, so adding a new instruction (e.g. AVX10.2 XMM-state dpbusd)
   is one helper edit, not 20 kernel edits.

2. **Multi-binary dispatch is OS-portable and low-overhead.** The score
   function (`cpu-feats.cpp:263-323`) is a 50-line function that returns
   0 or a power-of-two-weighted integer. No ifunc, no PLT indirection,
   no runtime patching. The selected `.so` is the best match and every
   call inside is direct. Cross-ref ARTX01-F12.

3. **8×8 batched GEMM is a separate, well-tiled algorithm.** The
   `repack.cpp` 512-bit kernel is not just "the vecdot unrolled 16×" —
   it is a different algorithm with 16 independent accumulators, mask
   blends, and a 16-row × 8-col tile that fits in registers. This is
   the right design: the vecdot path is for GEMV (narrow src1), the
   batched path is for prompt processing (wide src1).

4. **`tinyBLAS` shape-adaptive tile selection.** The `mnpack` switch
   (`sgemm.cpp:1376-1459`) picks (mc, nc) tile shapes from (4,4) down
   to (1,1) based on the remaining (m, n). This avoids the
   "tile-boundary penalty" that fixed-shape GEMM kernels suffer at the
   edges. Reasonably clean code.

5. **AVX-VNNI fallback for non-AVX-512 CPUs.** Alder Lake and Zen 4
   have AVX-VNNI but not AVX-512. The same `_mm256_dpbusd_avx_epi32`
   instruction gives them the VNNI speedup at 256-bit width without
   needing a separate kernel.

6. **BF16 native dot in `llamafile_sgemm`.** The
   `tinyBLAS<32, __m512, __m512bh, …>` template (line 3790) is the
   only place in the audited files where AVX-512 BF16 is actually
   exploited. The 32-wide BF16 lanes double throughput vs. the 16-wide
   FP32 emulation path.

7. **Block-aligned assertion + generic fallback for K-quants.** Every
   K-quant vecdot has `assert(n % QK_K == 0)` and falls back to
   `ggml_vec_dot_*_generic` if the assert path is compiled out. This
   keeps the SIMD kernel simple (no tail handling) without sacrificing
   correctness for odd-shaped inputs.

---

## 14. Architectural Weaknesses

### W1 — No native 512-bit AVX-512 vecdot kernels in `quants.c`

**Evidence:** `quants.c:718, 859, 1325, 2038, 2426, 3920, 4004` — every
vecdot's `#if defined(__AVX2__)` branch is the top of the ladder; there
is no `#elif defined(__AVX512F__)` branch with `__m512` accumulators.
The only AVX-512 code in `quants.c` is the helper at line 106-118, and
it operates on `__m256i`.

**Impact:** IceLake and later CPUs run the per-block vecdot at 256-bit
width, leaving half the SIMD capacity unused. For GEMV (batch=1
inference), this is the hot path and the gap is real. The 8×8 batched
path in `repack.cpp` partially compensates for prompt processing, but
GEMV shapes still go through `quants.c`.

**Why it's hard to fix:** Writing a 512-bit vecdot for each of the 21
quant formats is a large engineering effort. The 8×8 batched path is
already 512-bit and is the preferred path when shapes allow; the
per-block vecdot is the fallback for shapes the batched path cannot
handle (e.g., n not a multiple of 8). A 512-bit vecdot would close
this gap but requires careful accumulator-count tuning to avoid
register spilling.

### W2 — AVX-VNNI_INT8 (`_mm256_dpbssd_epi32`) is wired but effectively bypassed

**Evidence:** `quants.c:123-126` declares the `__AVXVNNIINT8__` path
inside `mul_sum_i8_pairs_float`. But every caller of this helper
either (a) supplies nibble-expanded bytes that were already converted
to signed via `_mm256_sub_epi8(qx, off)` (`quants.c:731`) and then
calls `mul_sum_i8_pairs_float` which itself does abs+sign on top of
the now-signed input — defeating the signedness — or (b) bypasses it
entirely via `mul_add_epi8` (`quants.c:2101, 4038`) which uses
`_mm256_maddubs_epi16` (unsigned×signed).

**Impact:** Granite Rapids's signed-int8 VNNI instruction is plumbed
but its actual benefit is unclear. The unsigned-VNNI + sign-extend
pattern that the rest of the code uses is well-tested and may be
equally fast. **The signed-VNNI path may be dead code in practice.**

**Why it's hard to fix:** Requires benchmarking on Granite Rapids
hardware to determine whether `dpbssd` is faster than `dpbusd` + abs.
Static analysis cannot resolve this.

### W3 — No AVX-512 FP16 dot product anywhere

**Evidence:** Grep for `_mm512_.*ph.*_ps` across `quants.c`,
`repack.cpp`, `sgemm.cpp`, `simd-mappings.h` returns zero matches.
`cpu-feats.cpp:81` detects `AVX512_FP16()` but no kernel uses it. F16
weights are always converted to F32 via `_mm256_cvtph_ps` /
`_mm512_cvtph_ps` and operated in F32.

**Impact:** On Granite Rapids / Arrow Lake / Sierra Forest, native FP16
FMA (`_mm512_fcmadd_pch`) would double throughput for F16-input GEMM
vs. the convert-then-FMA pattern. For LLM inference with F16 weights,
this is a measurable loss.

**Why it's hard to fix:** Requires a new `tinyBLAS<32, __m512h, __m512h, …>`
template and an AVX-512 FP16 codepath. The codebase does not currently
distinguish FP16-capable AVX-512 from BF16-capable AVX-512 at the
kernel-selection level — both are folded into the same `__AVX512F__`
build macro.

### W4 — No AVX-512 BF16 in `quants.c` or `repack.cpp`

**Evidence:** `_mm512_dpbf16_ps` appears only in `llamafile/sgemm.cpp`
(line 154, 3790). `quants.c` and `repack.cpp` have no BF16 usage.
Quantized dot products accumulate in FP32 always.

**Impact:** Limited. Quantized weights are int4/int8; BF16 doesn't
apply. But the *activation* conversion path could theoretically use
BF16 for some quants (e.g. Q8_0 with a BF16 scale), saving 16 bits
per scale and doubling scale-load throughput. This is not done.

**Why it's hard to fix:** The activation is already Q8_0 (int8); BF16
doesn't help. The benefit would only materialize if a future quant
format stored BF16 scales, which is not currently the case.

### W5 — `quantize_row_q8_K` is a placeholder

**Evidence:** `quants.c:505-507`:
```c
// placeholder implementation for Apple targets
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}
```
No `#if defined(__AVX2__)` ladder. Always calls the generic reference.
The comment says "for Apple targets" but the function is unconditional
on x86.

**Impact:** Every matmul that targets a K-quant (Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ2_*, IQ3_*, IQ4_XS, IQ1_*) pays scalar performance for
the F32 → Q8_K activation conversion. This happens once per matmul
(ARTX01 §5.5 step 2), but for prompt processing with a large `ne10`,
the conversion is non-trivial. **The activation conversion is the
only step in the K-quant matmul pipeline that has no SIMD
optimization on x86.**

**Why it's hard to fix:** Q8_K has 256-element blocks with 32
sub-block scales; writing a SIMD quantizer that produces both the
packed int8 values and the per-sub-block sums (`bsums[QK_K/16]`) is
non-trivial but straightforward. Likely a missed optimization rather
than a deliberate design choice.

### W6 — `nrc == 1` enforced on x86; no multi-row vecdot

**Evidence:** `quants.c:706, 863, 1313, 2039, 2426, 3921, 4005` —
every vecdot asserts `nrc == 1`. The corresponding `nrows` in the
type-traits table is `1` for every x86 quant (vs. `2` for ARM I8MM;
ARTX01 §10).

**Impact:** For matmuls where `ne01` (number of weight rows) is large
and `ne11` (number of activation rows) is small (typical GEMV), the
per-block vecdot is called once per weight row. ARM I8MM can consume
2 weight rows per call from a single activation block, halving the
load on the activation cache. x86 has no equivalent — it reloads the
activation block for every weight row.

**Why it's hard to fix:** The 8×8 batched GEMM path in `repack.cpp`
already provides multi-row consumption (16 rows × 8 cols per call) and
is the preferred path when shapes allow. The per-block vecdot path is
the fallback for shapes the batched path cannot handle. A multi-row
vecdot would close this gap, but it would duplicate work already done
by the batched path.

### W7 — AVX2 vecdot kernels drop the SSSE3 software prefetch

**Evidence:** `quants.c:782, 800` (`_mm_prefetch(&x[ib] + sizeof(block_q4_0), _MM_HINT_T0)`)
in the SSSE3 Q4_0 path. The AVX2 Q4_0 path (`quants.c:718-741`) has
no prefetch. Same pattern for Q4_1, Q8_0.

**Impact:** Unknown statically. The hardware prefetcher may be
sufficient for the sequential block strides, or it may not. The SSSE3
path was written first and kept its prefetches; the AVX2 path was
written later and dropped them, possibly by oversight.

**Why it's hard to fix:** Requires runtime profiling to determine
whether prefetching helps. Static analysis cannot resolve.

### W8 — `quad_fp16_delta_float` does FP16→FP32 conversion as scalar

**Evidence:** `quants.c:265-269`:
```c
return _mm256_set_m128(_mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                       _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
```
Four scalar `_cvtsh_ss` calls + two scalar multiplies + two
broadcasts. This is on the AVX (non-AVX2) path of Q4_0/Q4_1/Q8_0/
IQ4_NL vecdot. The AVX2 path (`quants.c:725`) does the conversion
once per block via `_mm256_set1_ps`.

**Impact:** On AVX-only hardware (Sandy Bridge / Ivy Bridge, 2011-2012),
the scalar conversion creates a 4-instruction serial dependency per
block. On modern hardware (Haswell and later, which all have AVX2),
the AVX2 path is taken and this code is dead. So the impact is
limited to very old hardware.

**Why it's hard to fix:** The AVX path exists for compatibility with
pre-Haswell hardware. Modernizing it requires either dropping AVX
support or maintaining a third codepath.

### W9 — `cpu-feats.cpp` does not check OS XSAVE support

**Evidence:** `cpu-feats.cpp:264` comment: "FIXME: this does not check
for OS support". The score function trusts CPUID, which can report
AVX-512 support even when the OS has not enabled AVX-512 context
saving.

**Impact:** On systems where the OS does not set `XCR0[2:4] |= 0b111`,
executing AVX-512 instructions will fault with `#UD`. The selected
`.so` will crash. In practice, all modern Linux/macOS/Windows kernels
enable XSAVE for AVX-512, but the FIXME acknowledges the gap.

**Why it's hard to fix:** Requires a `xgetbv` instruction to check
`XCR0`. Trivial to add but requires careful handling for old OSes.

### W10 — No mask-register tail handling

**Evidence:** Every K-quant vecdot has `assert(n % QK_K == 0)` (e.g.
`quants.c:2039`). The Q4_0/Q4_1/Q8_0/IQ4_NL/MXFP4/NVFP4 vecdots have
a scalar tail loop (e.g. `quants.c:840-854`). No kernel uses
`__mmask16` masked load/multiply/store to handle the tail in SIMD.

**Impact:** For shapes where `n` is not a multiple of the block size,
the tail falls back to scalar (Q4_0 family) or to the generic
reference (K-quants). On IceLake with AVX-512, masked loads
(`_mm256_maskz_loadu_epi32`) would let the kernel handle the tail in
one or two masked vector ops.

**Why it's hard to fix:** The K-quants' 256-element block size means
the tail is rare in practice (most model dimensions are multiples of
256). The cost of adding masked-tail handling may not be worth the
complexity. The Q4_0 family's 32-element block size means the tail
is more common, but the scalar tail is short (at most 31 iterations).

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc` | **ADOPT** | `mul_sum_us8_pairs_float` 3-way VNNI dispatch helper | Clean compile-time ladder; one edit adds a new instruction. Cross-ref F02. |
| `glproc` | **ADOPT** | AVX-VNNI fallback for non-AVX-512 CPUs | Alder Lake / Zen 4 get VNNI speedup at 256-bit width. F03. |
| `glproc` | **ADOPT** | 8×8 batched GEMM with 16 × `__m512` accumulators (repack.cpp pattern) | Right algorithm for prompt-processing shapes; 16 independent accumulators break dep chain. F05. |
| `glproc` | **ADOPT** | `_mm512_mask_blend_epi32` for accumulator combination | One-instruction merge of two `__m512i` halves. F06. |
| `glproc` | **ADOPT** | Multi-binary dispatch via `ggml_backend_cpu_x86_score` | Low-overhead, OS-portable, no ifunc. Cross-ref ARTX01-F12. F09. |
| `glproc` | **ADOPT** | `tinyBLAS` shape-adaptive `mnpack` tile selection | Picks best (mc, nc) from remaining (m, n); avoids tile-boundary penalty. |
| `glproc` | **ADOPT** | BF16 native dot via `_mm512_dpbf16_ps` in tinyBLAS | Only place BF16 is actually exploited; doubles BF16 GEMM throughput. |
| `glproc` | **ADAPT** | Per-block vecdot 256-bit AVX2 path | Keep the structure, but add a 512-bit `__m512` variant for IceLake+ when nth=1 (GEMV). F01. |
| `glproc` | **REJECT** | The absence of native 512-bit AVX-512 vecdot kernels | GwenLand should provide 512-bit vecdot for at least Q4_0, Q4_K, Q6_K, IQ4_XS on IceLake+. F01. |
| `glproc` | **REJECT** | The absence of AVX-512 FP16 dot product | GwenLand should use `_mm512_fcmadd_pch` for F16 GEMM on Granite Rapids+. F08. |
| `glproc` | **MONITOR** | AVX-VNNI_INT8 (`_mm256_dpbssd_epi32`) signed dot | Plumbed but effectively bypassed; benchmark on Granite Rapids before adopting. F04. |
| `glproc` | **MONITOR** | `quantize_row_q8_K` placeholder | Should be SIMD-optimized; monitor whether upstream does it first. F12. |
| `glproc` | **MONITOR** | `quad_fp16_delta_float` scalar-conversion pattern | AVX-only; relevant only if GwenLand supports pre-Haswell x86. F11. |
| `glproc` | **DEFER** | AMX tile-based matmul | Separate ARTX; out of scope for IceLake. |
| `GATE` | **ADOPT** | Type-traits `nrows` parameter | Already adopted per ARTX01; lets multi-row vecdot be advertised per-quant. F10. |
| `GATE` | **ADAPT** | `llamafile_sgemm` integration | Keep the integration but make selection a plan-time decision (cross-ref ARTX01-R5). |

---

## 16. Recommendations

### R1 — ADOPT the `mul_sum_us8_pairs_float` 3-way VNNI dispatch helper
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `glproc` should define an equivalent `gl_mul_sum_us8_pairs_float` helper with the same `#if` ladder: AVX-512 VNNI+VL → AVX-VNNI → scalar maddubs. Every int8 vecdot kernel should route through this helper. (F02, F03.)

### R2 — REJECT the absence of native 512-bit AVX-512 vecdot kernels; provide `__m512` variants
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
For at least the four hottest quants (Q4_0, Q4_K, Q6_K, IQ4_XS), GwenLand should provide a 512-bit vecdot variant selected when `__AVX512F__` is defined. Use 8 independent `__m512` accumulators (vs. the single `__m256` accumulator in the AVX2 path) to break the dependence chain. Expected speedup: 1.5–2× on IceLake for GEMV shapes. (F01.)

### R3 — ADOPT the 8×8 batched GEMM template from `repack.cpp`
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
GwenLand should adopt the 16-`__m512`-accumulator tile-blocked GEMM pattern from `repack.cpp:663-1096`. This is the actual IceLake fast path for prompt-processing shapes. The "repacked" weight layouts (`block_q4_0x8`, `block_q8_0x4`, etc.) are a necessary companion; the repacking happens once at weight-load time. (F05, F06.)

### R4 — ADOPT AVX-512 BF16 native dot for BF16 GEMM
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
GwenLand should use `_mm512_dpbf16_ps` for BF16×BF16 GEMM when `__AVX512BF16__` is defined, falling back to convert-then-FMA otherwise. This is already done in `llamafile/sgemm.cpp:154, 3790`; adopt the same template pattern. (F07.)

### R5 — ADOPT AVX-512 FP16 native dot for F16 GEMM
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R4
GwenLand should use `_mm512_fcmadd_pch` for F16×F16 GEMM when `__AVX512FP16__` is defined. This requires a new `tinyBLAS<32, __m512h, __m512h, …>` template. The current code converts F16 to F32 and uses `_mm512_fmadd_ps`, which is correct but halves throughput on FP16-capable hardware. (F08.)

### R6 — MONITOR AVX-VNNI_INT8 signed dot before adopting
**Priority:** Low
**Difficulty:** S
**Dependencies:** R1
The `__AVXVNNIINT8__` path in `mul_sum_i8_pairs_float` (`quants.c:123-126`) is plumbed but effectively bypassed by the abs+sign trick. Before adopting it in GwenLand, benchmark on Granite Rapids to determine whether `_mm256_dpbssd_epi32` outperforms `_mm256_dpbusd_epi32` + abs+sign. (F04.)

### R7 — ADOPT multi-binary ISA dispatch via score function
**Priority:** High
**Difficulty:** M
**Dependencies:** none
GwenLand should adopt the `ggml_backend_cpu_x86_score` pattern: compile N `.so` variants per ISA target, score each at load time, pick the best. Add the OS XSAVE check that the upstream FIXME omits. (F09, cross-ref ARTX01-F12.)

### R8 — ADOPT mask-register tail handling for Q4_0 family
**Priority:** Low
**Difficulty:** S
**Dependencies:** R2
For 512-bit vecdot kernels, use `_mm512_maskz_loadu_epi32` / `_mm512_mask_storeu_ps` to handle the `n % 32 != 0` tail in SIMD. Eliminates the scalar tail loop. (W10.)

### R9 — ADOPT shape-adaptive `mnpack` tile selection from tinyBLAS
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R3
GwenLand's GEMM kernel should pick (mc, nc) tile shape from the remaining (m, n) at each step, like `tinyBLAS_Q0_AVX::mnpack` (`sgemm.cpp:1376-1459`). Avoids tile-boundary penalty at the edges. The `VECTOR_REGISTERS == 32` switch (32 regs for AVX-512 / ARM NEON, 16 for AVX2) is a clean way to express the wider tile.

### R10 — ADOPT `nrows = 2` (or more) for x86 multi-row vecdot
**Priority:** Low
**Difficulty:** M
**Dependencies:** R2
GwenLand should consider setting `nrows = 2` (or 4) for Q4_0/Q4_K/Q6_K on x86, consuming 2 weight rows per call from a single activation block. The 8×8 batched path already does this (16 rows per call); a 2-row vecdot would be the GEMV-friendly middle ground. (F10, W6.)

### R11 — ADAPT `quantize_row_q8_K` to be SIMD-optimized
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
The placeholder at `quants.c:505-507` should be replaced with an AVX2/AVX-512 SIMD implementation. The quantizer must produce both the packed int8 values and the per-sub-block `bsums[QK_K/16]` array; this is the non-trivial part. (F12, W5.)

---

## 17. Findings

### Finding ARTX02-F01

```
Finding ID:           ARTX02-F01
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Per-block vecdot kernels (all quant types)
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             ggml_vec_dot_q4_0_q8_0 (and 20 sibling vecdot functions)
Lines:                718-841 (Q4_0 representative); 859-915 (Q4_1); 1308-1374 (Q8_0);
                      1574-1650 (Q2_K); 1766-1885 (Q3_K); 2038-2120 (Q4_K); 2426-2508 (Q6_K);
                      3920-4001 (IQ4_NL); 4004-4107 (IQ4_XS)

Summary:              No native 512-bit AVX-512 vecdot kernels exist in quants.c; every
                      vecdot uses 256-bit __m256/__m256i accumulators via #if defined(__AVX2__)
                      even when __AVX512F__ is defined.

Observation:          The vecdot preprocessor ladder for every quant type is:
                      #if defined(__AVX2__) → 256-bit path
                      #elif defined(__AVX__) → 128-bit path
                      #elif defined(__SSSE3__) → 64-bit path
                      #else → generic ref
                      There is no #elif defined(__AVX512F__) branch with __m512/__m512i
                      accumulators. The only AVX-512-specific code in quants.c is the
                      helper mul_sum_us8_pairs_float (line 105-119), and it operates at
                      __m256i width (256-bit) even when AVX-512 F is available. The
                      AVX-512 VNNI instruction _mm256_dpbusd_epi32 is used (line 108) but
                      only at 256-bit width.

                      For GEMV (batch=1 inference), the per-block vecdot is the hot path
                      and 8 lanes vs. 16 lanes is a 2× SIMD width gap. The 8×8 batched
                      GEMM in repack.cpp (F05) partially compensates for prompt processing,
                      but GEMV shapes still go through this 256-bit path.

Evidence:             quants.c:718 (#if defined(__AVX2__) — top of ladder, no AVX-512 branch);
                      quants.c:105-119 (only AVX-512 helper, 256-bit width);
                      quants.c:123-126 (AVX-VNNI_INT8 helper, also 256-bit).

Architectural Impact: IceLake and later CPUs leave half the SIMD capacity unused on the
                      per-block vecdot path. For GEMV-heavy workloads (token generation),
                      this is the dominant cost.

Correctness Impact:   None. 256-bit and 512-bit kernels would produce different ULP-level
                      results due to different reduction orders, but both are correct.

Optimization Type:    SIMD vectorization (proposed: widen to 512-bit).

GwenLand Target:      glproc

Recommendation:       REJECT the absence; provide __m512 variants for at least Q4_0, Q4_K,
                      Q6_K, IQ4_XS on IceLake+. Use 8 independent __m512 accumulators to
                      break the dependence chain. Expected speedup: 1.5–2× on GEMV.

Priority:             High
Difficulty:           L
Dependencies:         none
Confidence:           High
```

### Finding ARTX02-F02

```
Finding ID:           ARTX02-F02
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            AVX-512 VNNI 256-bit helper
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             mul_sum_us8_pairs_float
Lines:                105-119

Summary:              The only AVX-512-specific code in quants.c is a 3-way compile-time
                      dispatch helper that selects _mm256_dpbusd_epi32 (AVX-512 VNNI+VL),
                      _mm256_dpbusd_avx_epi32 (AVX-VNNI), or _mm256_maddubs_epi16 + madd
                      (scalar AVX2), all at 256-bit width.

Observation:          The helper is invoked by mul_sum_i8_pairs_float (line 122-134) which
                      is in turn called by every Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q1_0 vecdot.
                      The dispatch is compile-time, resolved by #if defined(__AVX512VNNI__)
                      && defined(__AVX512VL__). On IceLake (which has both), the inner
                      int8 dot product becomes one instruction (_mm256_dpbusd_epi32)
                      instead of two (_mm256_maddubs_epi16 + _mm256_madd_epi16).

                      This is a clean design: one helper edit adds a new instruction
                      (e.g. AVX10.2 XMM-state dpbusd) without touching the 20+ vecdot
                      kernels that consume it. The 256-bit width is consistent with the
                      vecdot kernels' __m256 accumulators (F01).

Evidence:             quants.c:105-119 (helper definition);
                      quants.c:735 (Q4_0 callsite via mul_sum_i8_pairs_float);
                      quants.c:1336 (Q8_0 callsite).

Architectural Impact: AVX-512 VNNI throughput is harvested at 256-bit width on the per-block
                      vecdot path. The 512-bit width is harvested only in repack.cpp (F05)
                      and llamafile/sgemm.cpp.

Correctness Impact:   None. dpbusd is bit-exact equivalent to maddubs+madd+horizontal-add.

Optimization Type:    SIMD (instruction fusion: 2 ops → 1 op).

GwenLand Target:      glproc

Recommendation:       ADOPT. Define an equivalent gl_mul_sum_us8_pairs_float helper with
                      the same 3-way ladder.

Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX02-F03

```
Finding ID:           ARTX02-F03
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            AVX-VNNI fallback (Alder Lake / Zen 4)
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             mul_sum_us8_pairs_float
Lines:                110-113

Summary:              The VNNI helper has an AVX-VNNI fallback for CPUs that have VNNI
                      at 256-bit width but no AVX-512 (Alder Lake P-cores, Zen 4).

Observation:          When __AVXVNNI__ is defined but __AVX512VNNI__ + __AVX512VL__ are
                      not, the helper uses _mm256_dpbusd_avx_epi32 (note the _avx suffix)
                      which is the AVX-VNNI encoding of the same instruction. This gives
                      Alder Lake and Zen 4 the same 1-instruction int8 dot product that
                      IceLake gets via AVX-512 VNNI.

                      This fallback is the only AVX feature used by quants.c that is not
                      part of AVX2. It enables meaningful speedup on a class of CPUs that
                      do not have AVX-512 at all.

Evidence:             quants.c:110-113 (_mm256_dpbusd_avx_epi32 path);
                      cpu-feats.cpp:293-296 (GGML_AVX_VNNI score contribution).

Architectural Impact: Non-AVX-512 CPUs with AVX-VNNI get the VNNI speedup. Without this
                      fallback, they would fall to the 2-instruction maddubs+madd path.

Correctness Impact:   None.

Optimization Type:    SIMD (instruction fusion via AVX-VNNI).

GwenLand Target:      glproc

Recommendation:       ADOPT. Include the same AVX-VNNI fallback in GwenLand's helper.

Priority:             High
Difficulty:           S
Dependencies:         ARTX02-F02
Confidence:           High
```

### Finding ARTX02-F04

```
Finding ID:           ARTX02-F04
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            AVX-VNNI_INT8 signed dot (Granite Rapids)
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             mul_sum_i8_pairs_float
Lines:                122-134

Summary:              The AVX-VNNI_INT8 path (_mm256_dpbssd_epi32, signed int8 dot) is
                      plumbed under #if __AVXVNNIINT8__ but is effectively bypassed in
                      practice because every caller either supplies already-signed inputs
                      (defeating the abs+sign trick the helper uses to route to the
                      unsigned VNNI path) or bypasses this helper entirely via
                      mul_add_epi8.

Observation:          mul_sum_i8_pairs_float has two branches:
                        #if __AVXVNNIINT8__: _mm256_dpbssd_epi32 (signed)
                        #else: abs+sign trick, then mul_sum_us8_pairs_float (unsigned)
                      The signed path is reachable only on Granite Rapids (the only
                      shipping CPU with AVX-VNNI_INT8 as of 2026-07).

                      Every caller of mul_sum_i8_pairs_float supplies either:
                        (a) nibble-expanded bytes already in [-8, 7] (Q4_0 quants.c:731
                            subtracts _mm256_set1_epi8(8)), making the input signed. The
                            signed VNNI path takes these directly, the unsigned path
                            applies abs+sign on top — the abs is a no-op for already-
                            signed inputs but the sign vector is still computed.
                        (b) loadu_si256 of int8 weights (Q8_0 quants.c:1333-1334), which
                            are signed.

                      The performance benefit of dpbssd (1 instruction) vs. dpbusd + abs
                      + sign (3 instructions) is unclear. The dpbusd + abs + sign pattern
                      is what every non-Granite-Rapids CPU uses and is well-tested. The
                      signed-VNNI path may be faster (1 op vs. 3) or may not (the abs
                      and sign operations may be hidden behind the dpbusd latency).

Evidence:             quants.c:122-134 (helper definition);
                      quants.c:735 (Q4_0 callsite — signed input);
                      quants.c:1336 (Q8_0 callsite — signed input);
                      quants.c:2101 (Q4_K uses mul_add_epi8 directly, bypassing).

Architectural Impact: The signed-VNNI path is plumbed but its actual benefit is
                      unverified. It may be dead code in practice if the unsigned path
                      is equally fast on Granite Rapids.

Correctness Impact:   None. Both paths produce identical int32 dot products.

Optimization Type:    SIMD (signed int8 dot product).

GwenLand Target:      glproc

Recommendation:       MONITOR. Benchmark on Granite Rapids before adopting. If dpbssd is
                      faster, adopt the signed path; otherwise drop it to reduce code
                      complexity.

Priority:             Low
Difficulty:           S
Dependencies:         ARTX02-F02
Confidence:           Medium
```

### Finding ARTX02-F05

```
Finding ID:           ARTX02-F05
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            8×8 batched GEMV/GEMM (IceLake fast path)
Source File:          ggml/src/ggml-cpu/arch/x86/repack.cpp
Function:             gemm_q4_b32_8x8_q8_0_lut_avx (template instantiation for Q4_0x8);
                      gemv_q4_b32_8x8_q8_0_lut_avx; same template for Q4_K, IQ4_NL, MXFP4, Q2_K
Lines:                518-1096 (template); 1448-1713 (Q4_0/Q4_K/IQ4_NL/MXFP4/Q2_K GEMV
                      entry points); 2026-3526 (GEMM entry points)

Summary:              The actual IceLake fast path for prompt-processing is in repack.cpp,
                      not quants.c. It uses 16 × __m512 accumulators (16 independent
                      dependency chains), full 512-bit loads, and _mm512_shuffle_epi8
                      LUT-based nibble expansion.

Observation:          When __AVX512BW__ && __AVX512DQ__ are defined, the GEMM kernel
                      (repack.cpp:663-1096) uses:
                        - 16 × __m512 acc_rows[16] (line 687-690) — 16 independent
                          accumulators, one per output row of a 16-row × 8-col tile.
                        - _mm512_shuffle_epi8 against signextendlutexpanded (line 721-731)
                          to expand 4-bit nibbles to signed 8-bit values, one instruction
                          per 16-byte half-block.
                        - mul_sum_i8_pairs_acc_int32x16 (line 134-144) which uses
                          _mm512_abs_epi8 + _mm512_mask_sub_epi8 + _mm512_dpbusd_epi32
                          (when AVX-512 VNNI is available) or _mm512_maddubs_epi16 +
                          _mm512_madd_epi16 (otherwise).
                        - _mm512_fmadd_ps for the final scale × int32 → fp32 step
                          (line 872-875).
                        - _mm512_mask_blend_epi32(0xCCCC, …, …) (line 862-865) to merge
                          two __m512i dot-product halves into row-major output (F06).

                      The 256-bit fallback (line 1098+) uses 16 × __m256 accumulators —
                      same algorithm, half the lanes, half the throughput.

                      This kernel is the actual IceLake fast path. quants.c's 256-bit
                      vecdot (F01) is the fallback for GEMV shapes that the 8×8 batched
                      path cannot handle (n < 8, or weights not repacked into
                      block_q4_0x8 format).

Evidence:             repack.cpp:663 (#if defined(__AVX512BW__) && defined(__AVX512DQ__));
                      repack.cpp:687-690 (16 __m512 accumulators);
                      repack.cpp:721-731 (_mm512_shuffle_epi8 nibble expansion);
                      repack.cpp:862-865 (_mm512_mask_blend_epi32 accumulator merge);
                      repack.cpp:872-875 (_mm512_fmadd_ps scale step);
                      repack.cpp:2026-2042 (ggml_gemm_q4_0_8x8_q8_0 entry point).

Architectural Impact: This is the path that justifies the IceLake build. Without it,
                      IceLake would only get the VNNI 256-bit speedup (F02) over AVX2,
                      which is roughly 1.3×, not the 2× SIMD-width speedup. With it,
                      prompt-processing shapes (large ne11, ne01) get full 512-bit
                      throughput.

Correctness Impact:   None. Algorithm is bit-exact equivalent to the per-block vecdot
                      path; only the reduction order differs (16-row tile vs. 1-row).

Optimization Type:    SIMD (512-bit vectorization, 16-independent-accumulator dependency
                      breaking, mask-blend accumulator combination, LUT-based nibble
                      expansion).

GwenLand Target:      glproc

Recommendation:       ADOPT. This is the right algorithm for prompt-processing on IceLake+.
                      GwenLand should replicate the 16-__m512-accumulator tile-blocked GEMM
                      pattern, including the repacked weight layouts (block_q4_0x8,
                      block_q8_0x4, etc.) that make it possible.

Priority:             High
Difficulty:           L
Dependencies:         none
Confidence:           High
```

### Finding ARTX02-F06

```
Finding ID:           ARTX02-F06
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Mask register usage in 8×8 GEMM
Source File:          ggml/src/ggml-cpu/arch/x86/repack.cpp
Function:             gemm_q4_b32_8x8_q8_0_lut_avx (512-bit path)
Lines:                862-865; 139-140 (helper)

Summary:              Mask registers (__mmask16/32/64) appear in the audited files only
                      in repack.cpp, used for two purposes: (a) _mm512_mask_blend_epi32
                      to merge two __m512i dot-product halves into row-major output, and
                      (b) _mm512_movepi8_mask + _mm512_mask_sub_epi8 in the signed-int8
                      helper to compute the abs+sign pattern at 512-bit width.

Observation:          The mask-blend at line 862-865 combines two __m512i dot-product
                      halves (iacc_mat_00 and iacc_mat_01 shuffled) into a single
                      row-major __m512i. The mask 0xCCCC selects 4 lanes from each half:
                        iacc_row_0 = blend(0xCCCC, iacc_mat_00, shuffle(iacc_mat_01, 78))
                      This is one instruction instead of two shuffles + two inserts.

                      quants.c uses no mask registers. sgemm.cpp uses no mask registers.
                      The opportunity to use masked loads/stores for tail handling
                      (W10) is not taken.

Evidence:             repack.cpp:862-865 (_mm512_mask_blend_epi32);
                      repack.cpp:139-140 (_mm512_movepi8_mask + _mm512_mask_sub_epi8 in
                      mul_sum_i8_pairs_acc_int32x16 helper).

Architectural Impact: Mask registers are used in exactly one place in the IceLake path.
                      They are not used for tail handling, masked load/store, or
                      conditional accumulation in any vecdot or GEMM kernel.

Correctness Impact:   None.

Optimization Type:    SIMD (mask-blend accumulator combination).

GwenLand Target:      glproc

Recommendation:       ADOPT the mask-blend pattern. Consider also using masked loads
                      (_mm512_maskz_loadu_epi32) for tail handling in 512-bit vecdot
                      kernels (R2, R8).

Priority:             Medium
Difficulty:           S
Dependencies:         ARTX02-F05
Confidence:           High
```

### Finding ARTX02-F07

```
Finding ID:           ARTX02-F07
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            AVX-512 BF16 dot product
Source File:          ggml/src/ggml-cpu/llamafile/sgemm.cpp
Function:             tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, float>
                      (madd specialization at line 151-160; instantiation at line 3790)
Lines:                151-160; 372-385 (load helpers); 3788-3795 (instantiation)

Summary:              AVX-512 BF16 native dot product (_mm512_dpbf16_ps) is used only in
                      llamafile/sgemm.cpp. It is not used in quants.c or repack.cpp.

Observation:          The madd template specialization at line 151-160:
                        template<> inline __m512 madd(__m512bh a, __m512bh b, __m512 c) {
                            return _mm512_dpbf16_ps(c, a, b);
                        }
                      is instantiated by tinyBLAS<32, __m512, __m512bh, …> at line 3790
                      when Atype == GGML_TYPE_BF16 && Btype == GGML_TYPE_BF16 &&
                      __AVX512BF16__ is defined. The 32-wide BF16 lanes (vs. 16-wide FP32)
                      double throughput on Sapphire Rapids and later.

                      The fallback at line 3796 (AVX-512F without BF16) converts BF16 to
                      FP32 via _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(...), 16))
                      and uses tinyBLAS<16, __m512, __m512, …> with _mm512_fmadd_ps.

                      quants.c does not use BF16 at all (quantized weights are int4/int8).
                      repack.cpp does not use BF16 (its 8×8 GEMM is for int4 quants only).

Evidence:             sgemm.cpp:151-160 (madd specialization);
                      sgemm.cpp:372-385 (load helpers, including _mm512_cvtne2ps_pbh);
                      sgemm.cpp:3788-3795 (BF16 instantiation);
                      sgemm.cpp:3796-3803 (FP32 fallback).

Architectural Impact: BF16 GEMM throughput on Sapphire Rapids+ is doubled via native
                      dot product. The benefit is limited to GGML_TYPE_BF16 weights,
                      which are common for transformer inference (most LLMs store weights
                      in BF16 when quantization is not used).

Correctness Impact:   None. _mm512_dpbf16_ps accumulates in FP32 per Intel spec; the
                      result is bit-exact equivalent to convert-then-FMADD.

Optimization Type:    SIMD (native BF16 dot product).

GwenLand Target:      glproc

Recommendation:       ADOPT. Use _mm512_dpbf16_ps for BF16×BF16 GEMM when
                      __AVX512BF16__ is defined, with FP32-emulation fallback otherwise.

Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX02-F08

```
Finding ID:           ARTX02-F08
Category:             MISSING_FEATURE
Engine:               CPU
Component:            AVX-512 FP16 dot product
Source File:          (no usage; audited files: quants.c, repack.cpp, sgemm.cpp, simd-mappings.h)
Function:             N/A
Lines:                N/A

Summary:              No AVX-512 FP16 intrinsics are used anywhere in the audited files.
                      The _mm512_*ph*_ps family does not appear. FP16 is always converted
                      to FP32 via _mm512_cvtph_ps / _mm256_cvtph_ps / _cvtsh_ss and
                      operated in FP32.

Observation:          cpu-feats.cpp:81 defines AVX512_FP16() and the score function
                      reads it, but no kernel uses native FP16 dot product. The current
                      F16 GEMM path (sgemm.cpp:3852-3859) uses
                      tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, float>
                      which loads via _mm512_cvtph_ps (line 364) and computes via
                      _mm512_fmadd_ps (line 148). This is the convert-then-FMA pattern,
                      which uses only 16 FP32 lanes per cycle.

                      Native AVX-512 FP16 (via _mm512_fcmadd_pch) would use 32 FP16
                      lanes per cycle, doubling throughput on Granite Rapids / Arrow
                      Lake / Sierra Forest (the only shipping CPUs with AVX-512 FP16
                      as of 2026-07).

                      The opportunity is meaningful for LLM inference with F16 weights
                      (common for models exported from PyTorch with F16 storage).

Evidence:             cpu-feats.cpp:81 (AVX512_FP16 detection);
                      sgemm.cpp:363-365 (F16 → F32 conversion via _mm512_cvtph_ps);
                      sgemm.cpp:148-149 (F32 fmadd, not FP16);
                      (grep for "_mm512_.*ph.*_ps" across audited files: 0 matches).

Architectural Impact: F16 GEMM runs at half the achievable throughput on FP16-capable
                      AVX-512 hardware. For LLM inference with F16 weights, this is a
                      measurable loss.

Correctness Impact:   None. Native FP16 dot product accumulates in FP32 per Intel spec;
                      result is bit-exact equivalent to convert-then-FMADD.

Optimization Type:    SIMD (native FP16 dot product — proposed, not present).

GwenLand Target:      glproc

Recommendation:       REJECT the absence. Add a tinyBLAS<32, __m512h, __m512h, …> template
                      using _mm512_fcmadd_pch when __AVX512FP16__ is defined. Select it
                      for GGML_TYPE_F16 GEMM on FP16-capable AVX-512 hardware.

Priority:             Medium
Difficulty:           M
Dependencies:         ARTX02-F07
Confidence:           High
```

### Finding ARTX02-F09

```
Finding ID:           ARTX02-F09
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Multi-binary ISA dispatch (IceLake score)
Source File:          ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
Function:             ggml_backend_cpu_x86_score
Lines:                263-323

Summary:              IceLake (and AVX-512 in general) is selected via a multi-binary
                      dispatch: each .so variant is compiled with specific GGML_AVX*
                      macros; the runtime score returns 0 if the CPU lacks the features,
                      higher for more advanced features. The loader picks the best match.

Observation:          The score function is a 50-line function that checks each compile-
                      time GGML_* macro against the corresponding CPUID bit. If the CPU
                      lacks a feature required by the .so variant, the function returns 0
                      and the loader skips that variant. Otherwise it adds a power-of-two
                      weight (1<<n) for each feature present.

                      For the IceLake variant, the relevant macros are:
                        GGML_FMA, GGML_F16C, GGML_SSE42, GGML_BMI2, GGML_AVX, GGML_AVX2,
                        GGML_AVX_VNNI (optional, for Alder Lake),
                        GGML_AVX512 (requires F+CD+VL+DQ+BW),
                        GGML_AVX512_VBMI (optional),
                        GGML_AVX512_BF16 (optional, Sapphire Rapids+),
                        GGML_AVX512_VNNI (optional, IceLake+),
                        GGML_AMX_INT8 (optional, Sapphire Rapids+; separate plugin).

                      An IceLake client part (i7-1065G7) scores: FMA+F16C+SSE42+BMI2+
                      AVX+AVX2+AVX512+AVX512_VBMI+AVX512_VNNI = 1+2+4+8+16+32+128+256+
                      1024 = 1471. A Sapphire Rapids server part adds AVX512_BF16 (512)
                      and AMX_INT8 (2048) for ~4031.

                      The FIXME at line 264 ("does not check for OS support") is a
                      known gap: if the OS has not enabled XSAVE for AVX-512, the CPUID
                      bits may report support but executing AVX-512 instructions will
                      fault. In practice all modern OSes enable this.

Evidence:             cpu-feats.cpp:263-323 (score function);
                      cpu-feats.cpp:264 (FIXME comment);
                      cpu-feats.cpp:297-316 (AVX-512 macro checks).

Architectural Impact: Multi-binary dispatch is the cleanest way to handle x86 ISA
                      fragmentation: no ifunc, no PLT indirection, no runtime patching.
                      The selected .so has direct calls throughout. The cost is build/
                      distribution complexity (N .so files per architecture).

Correctness Impact:   None. The selected .so is the best match; correctness is enforced
                      by the score returning 0 for unsupported features.

Optimization Type:    Multi-binary ISA dispatch.

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the score function in GwenLand, with the OS XSAVE
                      check that the upstream FIXME omits. Cross-ref ARTX01-F12.

Priority:             High
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX02-F10

```
Finding ID:           ARTX02-F10
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Per-block vecdot nrc parameter
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             every ggml_vec_dot_* function
Lines:                706 (Q4_0); 863 (Q4_1); 1313 (Q8_0); 2039 (Q4_K); 2426 (Q6_K);
                      3921 (IQ4_NL); 4005 (IQ4_XS); etc.

Summary:              Every x86 vecdot asserts nrc == 1, meaning the kernel consumes
                      exactly one weight row × one activation row per call. x86 has no
                      equivalent of ARM I8MM's nrows = 2 multi-row vecdot.

Observation:          The nrc parameter (number of row columns) is asserted to 1 in
                      every x86 vecdot. The corresponding nrows in type_traits_cpu is 1
                      for every x86 quant (ARTX01 §10). On ARM I8MM builds, the same
                      quants set nrows = 2 and the kernel consumes 2 weight rows from a
                      single activation block, halving the activation cache pressure.

                      For matmuls where ne01 (number of weight rows) is large and ne11
                      (number of activation rows) is small (typical GEMV), the per-block
                      vecdot is called once per weight row. Each call reloads the
                      activation block from cache. A multi-row vecdot would amortize
                      this load across 2 (or more) weight rows.

                      The 8×8 batched GEMM in repack.cpp (F05) already does this — 16
                      weight rows per call. But it requires the batched shape
                      (ne11 >= 8) and repacked weights. For GEMV (ne11 = 1), the batched
                      path does not apply and the per-block vecdot's nrc == 1 is the
                      only option.

Evidence:             quants.c:706 (assert(nrc == 1) in Q4_0);
                      quants.c:863, 1313, 2039, 2426, 3921, 4005 (same in all vecdots);
                      ggml-cpu.c:243, 253, 275, 314, 330 (nrows = 2 for ARM I8MM only).

Architectural Impact: GEMV inference (token generation) pays full activation cache miss
                      cost per weight row. A 2-row vecdot would halve this. The benefit
                      is bounded by activation cache pressure, which depends on the
                      model dimension and L2 size.

Correctness Impact:   None. Multi-row vecdot produces the same per-row results, just
                      computed in parallel.

Optimization Type:    SIMD (multi-row consumption — proposed, not present on x86).

GwenLand Target:      glproc, GATE

Recommendation:       MONITOR. Consider setting nrows = 2 (or 4) for Q4_0/Q4_K/Q6_K on
                      x86 when nth = 1 (GEMV). The 8×8 batched path already covers the
                      nth > 1 (prompt processing) case. The per-block multi-row vecdot
                      would be the GEMV-friendly middle ground.

Priority:             Low
Difficulty:           M
Dependencies:         ARTX02-F01
Confidence:           Medium
```

### Finding ARTX02-F11

```
Finding ID:           ARTX02-F11
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            FP16 scale broadcast in AVX (non-AVX2) vecdot path
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             quad_fp16_delta_float
Lines:                265-269

Summary:              The AVX (non-AVX2) vecdot path for Q4_0/Q4_1/Q8_0/IQ4_NL computes
                      the per-block FP16 scale product as a scalar (via GGML_CPU_FP16_TO_FP32)
                      and broadcasts via _mm_set1_ps. This creates a serial dependency
                      per block.

Observation:          quad_fp16_delta_float (line 265-269):
                        return _mm256_set_m128(
                            _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x1) * GGML_CPU_FP16_TO_FP32(y1)),
                            _mm_set1_ps(GGML_CPU_FP16_TO_FP32(x0) * GGML_CPU_FP16_TO_FP32(y0)));
                      Four scalar _cvtsh_ss calls + two scalar multiplies + two
                      _mm_set1_ps broadcasts. Each call to _cvtsh_ss is a 1-cycle
                      instruction on F16C-capable hardware but the four are serial.

                      The AVX2 path (line 725) does the same conversion once per block
                      via _mm256_set1_ps(GGML_CPU_FP16_TO_FP32(x[ib].d) * GGML_CPU_FP16_TO_FP32(y[ib].d))
                      — only two _cvtsh_ss calls per block, not four, because it processes
                      one block at a time, not two.

                      This pattern is on the AVX (non-AVX2) path only. The AVX2 path is
                      taken on Haswell (2013) and later. The AVX path is taken on Sandy
                      Bridge / Ivy Bridge (2011-2012). Modern hardware does not hit this
                      code path.

Evidence:             quants.c:265-269 (quad_fp16_delta_float definition);
                      quants.c:765 (Q4_0 AVX path callsite);
                      quants.c:1357 (Q8_0 AVX path callsite);
                      quants.c:3985 (IQ4_NL AVX path callsite).

Architectural Impact: Limited to pre-Haswell hardware. On modern hardware the AVX2 path
                      is taken and this code is dead.

Correctness Impact:   None. F16C conversion is bit-exact; the scalar multiply is the
                      same as a vector multiply.

Optimization Type:    None (suboptimal pattern in legacy code path).

GwenLand Target:      glproc

Recommendation:       MONITOR. If GwenLand supports pre-Haswell x86, modernize this
                      pattern to use _mm256_cvtph_ps for vectorized FP16→FP32 conversion.
                      Otherwise, drop the AVX path entirely and rely on AVX2.

Priority:             Low
Difficulty:           S
Dependencies:         none
Confidence:           Medium
```

### Finding ARTX02-F12

```
Finding ID:           ARTX02-F12
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            Q8_K activation quantization
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c
Function:             quantize_row_q8_K
Lines:                505-507

Summary:              quantize_row_q8_K is a placeholder that always calls the scalar
                      reference (quantize_row_q8_K_ref). There is no SIMD optimization
                      for the F32 → Q8_K activation quantization step on x86, despite
                      SIMD paths existing for Q8_0 (line 302) and Q8_1 (line 400).

Observation:          The function is 3 lines:
                        void quantize_row_q8_K(const float * x, void * y, int64_t k) {
                            quantize_row_q8_K_ref(x, y, k);
                        }
                      The comment says "placeholder implementation for Apple targets"
                      but the function is unconditional on x86. There is no #if defined
                      (__AVX2__) ladder.

                      Q8_K is the activation format for every K-quant matmul (Q2_K, Q3_K,
                      Q4_K, Q5_K, Q6_K, IQ2_*, IQ3_*, IQ4_XS, IQ1_*). The activation
                      conversion happens once per matmul (ARTX01 §5.5 step 2) and writes
                      to params->wdata. For prompt processing with large ne10 (sequence
                      length × embedding dim), this conversion is non-trivial.

                      The Q8_K block format has 256-element blocks with 32 sub-block
                      scales (bsums[QK_K/16]). A SIMD quantizer would need to produce
                      both the packed int8 values and the per-sub-block sums, which is
                      non-trivial but straightforward with AVX2 (the Q8_0 quantizer at
                      line 302-398 already shows the pattern).

Evidence:             quants.c:505-507 (placeholder);
                      quants.c:302-398 (Q8_0 SIMD quantizer for comparison);
                      quants.c:400-501 (Q8_1 SIMD quantizer for comparison).

Architectural Impact: Every K-quant matmul pays scalar performance for the activation
                      conversion step. The dot-product step itself is SIMD-optimized;
                      the conversion is not. For prompt processing, the conversion can
                      be a measurable fraction of total matmul time.

Correctness Impact:   None. The reference quantizer is correct; it is just slow.

Optimization Type:    None (absence of SIMD optimization).

GwenLand Target:      glproc

Recommendation:       MONITOR. Should be SIMD-optimized. GwenLand should provide an
                      AVX2/AVX-512 implementation that produces both the packed int8
                      values and the per-sub-block bsums array. The Q8_0 quantizer
                      pattern (quants.c:302-398) is a starting point; the additional
                      work is the bsums computation.

Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether `_mm256_dpbssd_epi32` (AVX-VNNI_INT8, signed) is faster
  than `_mm256_dpbusd_epi32` + abs + sign (unsigned + abs) on Granite
  Rapids. The signed path is plumbed (`quants.c:123-126`) but
  effectively bypassed (F04). Requires runtime benchmarking on Granite
  Rapids hardware. Static analysis cannot resolve.

* **U2**. Whether the 8×8 batched GEMM in `repack.cpp` is actually
  selected for typical LLM inference shapes. The selection happens via
  the `extra_buffer_type` mechanism (ARTX01-F04), which depends on
  weights being allocated with the repack buffer type. Whether the
  default ggml allocator does this, or whether the user must opt in,
  is not visible in the audited files. Requires inspecting the
  repack buffer-type registration (not in scope).

* **U3**. Whether the per-block vecdot path (F01) or the 8×8 batched
  path (F05) dominates for typical batch=1 inference (token generation).
  The 8×8 batched path requires `nc >= 8` (8 activation rows); for
  batch=1, this fails and the per-block vecdot is used. But the
  threshold depends on the repack buffer type's `supports_op` logic,
  which is not in the audited files.

* **U4**. Whether software prefetching would help the AVX2 vecdot
  kernels. The SSSE3 Q4_0 path has prefetches (`quants.c:782, 800`);
  the AVX2 path dropped them. Whether this was a measured decision or
  an oversight is not documented. Requires runtime profiling.

* **U5**. The actual speedup of native AVX-512 FP16 dot product
  (`_mm512_fcmadd_pch`) over convert-then-FMA for F16 GEMM on Granite
  Rapids. Intel's documentation claims 2× throughput; whether this
  materializes for LLM-shaped GEMM is unverified. Requires runtime
  benchmarking on FP16-capable AVX-512 hardware.

* **U6**. Whether the 16-`__m512`-accumulator tile in `repack.cpp:687`
  causes register spilling on IceLake (32 AVX-512 registers available).
  16 accumulators + 4 input registers + scale registers + LUT registers
  may exceed 32. Static analysis counts ~24-28 registers used; whether
  the compiler spills is not determinable without inspecting the
  compiler's register allocation output.

* **U7**. Whether the OS XSAVE check (W9) matters in practice. All
  modern Linux kernels (≥4.x), macOS, and Windows enable XSAVE for
  AVX-512 when the CPU supports it. The FIXME at `cpu-feats.cpp:264`
  may be obsolete. Requires surveying the actual deployment matrix.

* **U8**. Whether `quantize_row_q8_K` (F12) is a meaningful bottleneck
  in practice. The conversion happens once per matmul; for large
  `ne10` (prompt processing) it may be a few percent of total time;
  for small `ne10` (token generation) it is negligible. Requires
  runtime profiling.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines              |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------------ |
| R01       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `mul_sum_us8_pairs_float` (VNNI helper)        | 105-119            |
| R02       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `mul_sum_i8_pairs_float` (signed VNNI helper)  | 122-134            |
| R03       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `mul_add_epi8` (AVX2/AVX-512F)                 | 69-73              |
| R04       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `packNibbles` (AVX-512F branch)                | 136-155            |
| R05       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `quad_fp16_delta_float`                        | 265-269            |
| R06       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `get_scale_shuffle_q3k` / `_k4` / (128-bit)    | 518-552            |
| R07       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `quantize_row_q8_0`                            | 302-398            |
| R08       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `quantize_row_q8_1`                            | 400-501            |
| R09       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `quantize_row_q8_K` (placeholder)              | 505-507            |
| R10       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q4_0_q8_0`                       | 701-857            |
| R11       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q4_1_q8_1`                       | 859-916            |
| R12       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_mxfp4_q8_0`                      | 918-1002           |
| R13       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_nvfp4_q8_0`                      | 1004-1140          |
| R14       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q5_0_q8_0` / `q5_1_q8_1`         | 1142-1305          |
| R15       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q8_0_q8_0`                       | 1308-1374          |
| R16       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_tq1_0_q8_K` / `tq2_0_q8_K`       | 1376-1572          |
| R17       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q2_K_q8_K`                       | 1574-1764          |
| R18       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q3_K_q8_K`                       | 1766-2036          |
| R19       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q4_K_q8_K`                       | 2038-2214          |
| R20       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q5_K_q8_K`                       | 2216-2424          |
| R21       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_q6_K_q8_K`                       | 2426-2621          |
| R22       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `keven_signs_q2xs[1024]`                       | 2624-2657          |
| R23       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq2_xxs_q8_K`                    | 2660-2776          |
| R24       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq2_xs_q8_K`                     | 2778-3073          |
| R25       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq2_s_q8_K`                      | 3075-3258          |
| R26       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq3_xxs_q8_K` / `iq3_s_q8_K`     | 3260-3592          |
| R27       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq1_s_q8_K` / `iq1_m_q8_K`       | 3594-3918          |
| R28       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq4_nl_q8_0`                     | 3920-4002          |
| R29       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `ggml_vec_dot_iq4_xs_q8_K`                     | 4004-4108          |
| R30       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `mul_sum_us8_pairs_acc_int32x16` (AVX-512)     | 123-132            |
| R31       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `mul_sum_i8_pairs_acc_int32x16` (AVX-512 VNNI) | 134-144            |
| R32       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `gemv_q4_b32_8x8_q8_0_lut_avx` (template)      | 522-639            |
| R33       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `gemm_q4_b32_8x8_q8_0_lut_avx` (template)      | 641-1096           |
| R34       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | 16 × `__m512 acc_rows` block (AVX-512 BW+DQ)   | 663-1096           |
| R35       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `_mm512_mask_blend_epi32` accumulator merge    | 862-865            |
| R36       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `ggml_gemv_q4_0_8x8_q8_0` (entry)              | 1448-1462          |
| R37       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `ggml_gemm_q4_0_8x8_q8_0` (entry)              | 2026-2040          |
| R38       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `ggml_gemm_q4_K_8x8_q8_K` (entry)              | 2042-2057          |
| R39       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `VECTOR_REGISTERS` (32 on AVX-512 / ARM NEON)  | 66-70              |
| R40       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `madd<__m512bh, __m512bh, __m512>` (BF16 dot)  | 151-160            |
| R41       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `load<ggml_fp16_t>` → `__m512` via `_mm512_cvtph_ps` | 363-365       |
| R42       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `load<ggml_bf16_t>` → `__m512bh` (native BF16) | 372-385            |
| R43       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `tinyBLAS_Q0_AVX::mnpack` (shape-adaptive tile)| 1376-1459          |
| R44       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `tinyBLAS_Q0_AVX::updot` (VNNI dispatch)       | 1754-1764          |
| R45       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `llamafile_sgemm` (entry)                      | 3699-4058          |
| R46       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | F32 GEMM instantiation                         | 3726-3731          |
| R47       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | BF16 GEMM instantiation (native + fallback)    | 3788-3811          |
| R48       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | F16 GEMM instantiation                         | 3852-3859          |
| R49       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | Q8_0/Q4_0 GEMM instantiation                   | 3938-3982          |
| R50       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `ggml_backend_cpu_x86_score`                   | 263-323            |
| R51       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `AVX512_FP16()` / `AVX512_BF16()` / `AVX512_VNNI()` detection | 79-83 |
| R52       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `AMX_TILE()` / `AMX_INT8()` / `AMX_BF16()` / `AMX_FP16()` detection | 85-88 |
| R53       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `GGML_CPU_FP16_TO_FP32` (F16C path)            | 56-63              |
| R54       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `GGML_CPU_FP16_TO_FP32` (256 KB LUT fallback)  | 145-152            |
| R55       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `GGML_CPU_E8M0_TO_FP32_HALF` / `_UE4M3_TO_FP32`| 127-139            |
| R56       | `ggml/src/ggml-cpu/amx/amx.cpp`                     | AMX `tensor_traits` registration (out of scope)| 19-42              |
| R57       | `ggml/src/ggml-cpu/amx/amx.cpp`                     | `ggml_backend_amx_convert_weight` (out of scope)| 67-77             |
