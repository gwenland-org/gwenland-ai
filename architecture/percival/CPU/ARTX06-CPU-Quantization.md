# ARTX06 — CPU Quantization Formats

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (quant format registry), `glcuda` / `glmetal` / `glvulkan` (shared block-layout ABI), `GATE` (activation-conversion planning)

---

## 1. Executive Summary

ARTX06 audits the *quant format taxonomy itself*: the block layouts, scale
encodings, zero-point representations, packing schemes, and the
`from_float` / `vec_dot` / `dequantize_row` interface contract that every
llama.cpp backend agrees on. It is the *format-level* audit. Per-ISA SIMD
kernel details live in ARTX02 (IceLake / AVX-512), ARTX03 (AMD Zen),
ARTX04 (ARM NEON baseline), and ARTX05 (AArch64 I8MM/SVE/SME); those audits
describe *how* a given format is computed on a given ISA. This audit
describes *what the formats are*.

The audited commit ships **30 distinct numeric types** in `enum ggml_type`
(`ggml/include/ggml.h:389-433`): 26 quantized weight formats, BF16, F16,
F32, and four integer/extra types (I8/I16/I32/I64/F64). The 26 quants
divide into six families:

1. **Simple block quants** (Q1_0, Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
   — one fp16 scale per 32/64/128 elements, optionally one fp16 min.
2. **K-quants** (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) — super-blocks of 256
   elements (`QK_K = 256`) with 6-bit (or 4-bit) packed sub-block scales.
3. **I-quants** (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M,
   IQ4_NL, IQ4_XS) — importance-sampled, lookup-table-driven, mostly
   inference-only.
4. **Ternary quants** (TQ1_0, TQ2_0) — {-1,0,+1} weights for BitNet b1.58
   and TriLMs.
5. **Microscaling quants** (MXFP4, NVFP4) — OCP MX-compliant 4-bit E2M1
   weights with per-block (E8M0) or per-sub-block (UE4M3) scales.
6. **Activation-only** (Q8_0, Q8_1, Q8_K) — used as `vec_dot_type` targets
   for the weight formats above, never as a stored weight type in
   inference (except in LoRA / low-rank adapter scenarios).

For GwenLand, the architectural decisions worth **ADOPT**ing are: (1) the
`static_assert`-enforced block-layout structs in `ggml-common.h` (a stable
on-disk ABI shared across CPU/CUDA/Metal/Vulkan); (2) the
`vec_dot_type`-indirection contract that lets each weight format specify
its expected activation format; (3) the 12-byte `K_SCALE_SIZE` 6-bit
packing scheme used by Q4_K/Q5_K. The decisions worth **REJECT**ing or
**ADAPT**ing are: the `from_float = NULL` constraint on IQ2_XXS/XS/IQ1_S/
IQ1_M (an inference-only design choice that complicates runtime weight
conversion) and the IQ1_M nibble-spread scale (correct but a maintenance
hazard).

This audit catalogues 12 findings (F01–F12) at the format level. Per-ISA
kernel findings live in ARTX02–ARTX05 and are explicitly not duplicated
here.

---

## 2. Purpose

Provide a format-level / layout-level / contract-level reference for every
quantized numeric type in llama.cpp, such that:

* a GwenLand engineer implementing `glproc`, `glcuda`, `glmetal`, or
  `glvulkan` can implement every format from this document alone — without
  reopening the upstream source tree;
* the format registry in each GwenLand backend shares the same block
  layouts, scale encodings, and zero-point encodings, so a model quantized
  by `glproc` loads unmodified into `glcuda`;
* the `vec_dot_type` indirection is preserved (each weight format declares
  which activation format it expects, and the matmul path cooperatively
  converts src1 to that format once up-front).

It is **not** responsible for: per-ISA kernel selection (ARTX02–05),
graph-level fusion (ARTX01), or backend scheduling (ARTX22).

---

## 3. Source Files

| File                                          | Lines  | Role                                                                                  |
| --------------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `ggml/src/ggml-common.h`                      | 1911   | Block layout structs (`block_q4_0`, `block_q4_K`, …), `QK_*` constants, LUT grids     |
| `ggml/src/ggml-cpu/quants.c`                  | 1339   | Generic `quantize_row_*` and `ggml_vec_dot_*_generic` skeletons (the reference path)  |
| `ggml/src/ggml-cpu/quants.h`                  | 106    | Public CPU quant API: every `quantize_row_*` / `ggml_vec_dot_*` declaration            |
| `ggml/src/ggml-quants.c`                      | 5667   | Reference quantizers / dequantizers / `_ref` impls; `ggml_quantize_init` infrastructure |
| `ggml/src/ggml.c`                             | 8024   | `type_traits[GGML_TYPE_COUNT]` (base traits), `ggml_quantize_init`, `ggml_quantize_chunk` |
| `ggml/src/ggml-cpu/ggml-cpu.c`                | 3896   | `type_traits_cpu[GGML_TYPE_COUNT]` (CPU-specific: `from_float`, `vec_dot`, `vec_dot_type`, `nrows`) |
| `ggml/include/ggml.h`                         | 2931   | `enum ggml_type`, `struct ggml_type_traits`, `ggml_from_float_t` typedef               |
| `ggml/include/ggml-cpu.h`                     | 153    | `struct ggml_type_traits_cpu`, `ggml_vec_dot_t` typedef                                |
| `ggml/src/ggml-impl.h`                        | 784    | `ggml_ue4m3_to_fp32`, `ggml_e8m0_to_fp32_half` scale converters                        |
| `ggml/src/ggml-cpu/arch-fallback.h`           | ~300   | `#define X_generic X` aliases that route the public API to the generic skeleton when no per-ISA kernel exists |

> Note: the per-ISA `arch/<isa>/quants.c` files (x86, arm, riscv, loongarch,
> powerpc, s390, wasm) override the generic `vec_dot` and `from_float`
> symbols at link time via the `arch-fallback.h` macro layer. ARTX02–05
> audit those overrides. ARTX06 audits the generic skeleton and the
> block-layout structs that *all* ISAs share.

---

## 4. Architecture Overview

```
            ┌──────────────────────────────────────────────────────────────┐
            │  ggml/include/ggml.h                                         │
            │  ├─ enum ggml_type (43 values)                               │
            │  └─ struct ggml_type_traits { blck_size, type_size,          │
            │                                is_quantized, to_float,       │
            │                                from_float_ref }              │
            └──────────────────────────────────────────────────────────────┘
                                  │ (base ABI)
                                  ▼
            ┌──────────────────────────────────────────────────────────────┐
            │  ggml/src/ggml-common.h                                      │
            │  ├─ QK_* constants (QK4_0=32, QK_K=256, …)                   │
            │  ├─ block_* structs (block_q4_0, block_q4_K, block_iq2_xxs…) │
            │  ├─ static_assert(sizeof(block_X) == expected)               │
            │  └─ LUT grids (iq2xxs_grid[256], iq1s_grid[2048], …)         │
            └──────────────────────────────────────────────────────────────┘
                                  │ (shared on-disk layout)
                                  ▼
            ┌──────────────────────────────────────────────────────────────┐
            │  ggml/include/ggml-cpu.h                                     │
            │  └─ struct ggml_type_traits_cpu {                            │
            │         from_float, vec_dot, vec_dot_type, nrows             │
            │     }                                                        │
            └──────────────────────────────────────────────────────────────┘
                                  │ (CPU dispatch contract)
                                  ▼
            ┌──────────────────────────────────────────────────────────────┐
            │  ggml/src/ggml-cpu/ggml-cpu.c                                │
            │  └─ static const type_traits_cpu[GGML_TYPE_COUNT]            │
            │     [GGML_TYPE_Q4_0] = { quantize_row_q4_0,                  │
            │                         ggml_vec_dot_q4_0_q8_0,              │
            │                         GGML_TYPE_Q8_0, nrows=1/2 }          │
            └──────────────────────────────────────────────────────────────┘
                                  │ (function pointers)
                ┌─────────────────┼─────────────────┐
                ▼                 ▼                 ▼
   ┌────────────────────┐ ┌────────────────────┐ ┌────────────────────┐
   │ quants.c (generic) │ │ arch/<isa>/quants.c│ │ ggml-quants.c      │
   │ _generic suffix    │ │ (link-time         │ │ _ref suffix        │
   │ fallback path      │ │  overrides)        │ │ reference path     │
   └────────────────────┘ └────────────────────┘ └────────────────────┘
```

Key design points:

* **Two-tier traits table.** The base `type_traits[]` in `ggml.c:631`
  carries the storage-level facts (`blck_size`, `type_size`,
  `is_quantized`, `to_float`, `from_float_ref`). The CPU-only
  `type_traits_cpu[]` in `ggml-cpu.c:214` carries the compute-level facts
  (`from_float` for the runtime path, `vec_dot`, `vec_dot_type`, `nrows`).
  GPU backends each have their own equivalent of `type_traits_cpu` but
  share `type_traits`. This split is the structural reason that block
  layouts are common across backends but kernels are not.

* **Link-time kernel selection.** Every per-ISA `arch/<isa>/quants.c`
  defines the *same* symbols (`ggml_vec_dot_q4_0_q8_0`, etc.) as the
  generic `quants.c`. The right .so is selected by the backend registry
  via the per-ISA `cpu_score` function (ARTX01-F12). The
  `arch-fallback.h` macro layer (`#define ggml_vec_dot_tq1_0_q8_K_generic
  ggml_vec_dot_tq1_0_q8_K`) ensures the generic name resolves to whatever
  the link unit provided.

* **No runtime kernel swap.** `type_traits_cpu` is `static const`
  (`ggml-cpu.c:214`). A tuned kernel cannot be installed at runtime
  without going through the `extra_buffer_type` plugin mechanism (ARTX01-
  F04, ARTX01-F11). This is the same constraint as ARTX01; ARTX06
  inherits it for the quant layer.

---

## 5. Execution Flow

The quant format system has no execution flow of its own; it is invoked
by the matmul path. ARTX01 §5.5 documents the matmul hot path. From the
quant-format perspective, the relevant slice is:

### 5.1 Activation conversion (`wdata` materialization)

In `ggml_compute_forward_mul_mat` (`ggml-cpu.c:1272-1355`):

1. Look up `vec_dot_type = type_traits_cpu[src0->type].vec_dot_type`.
2. If `src1->type != vec_dot_type`, threads cooperatively convert src1
   from F32 (or its native type) to `vec_dot_type`, writing into
   `params->wdata`. Partition: `ne10_block_start = (ith * ne10/bs) /
   nth`, where `bs = ggml_blck_size(vec_dot_type)` (`ggml-cpu.c:1347`).
3. **Barrier**.
4. Inside each chunk, `vec_dot` is called with the pre-converted src1.

### 5.2 Per-block `vec_dot` dispatch

The `vec_dot` function pointer is `type_traits_cpu[src0->type].vec_dot`
(`ggml-cpu.c:1181`). Its signature is
`void (*)(int n, float *s, size_t bs, const void *vx, size_t bx, const
void *vy, size_t by, int nrc)` (`ggml/include/ggml-cpu.h:114-115`). The
`nrc` parameter selects 1-row or 2-row mode (ARM I8MM uses `nrc=2`,
others use `nrc=1`).

### 5.3 Reference vs runtime path

* `from_float` is the *runtime* quantizer (used by the matmul path for
  activations).
* `from_float_ref` is the *reference* quantizer (used by `to_float` ↔
  `from_float_ref` round-trip tests and by `ggml_quantize_chunk` for
  offline quantization of weights).

The CPU's `from_float` and the base layer's `from_float_ref` may or may
not be the same function. For simple formats they frequently are
(`quantize_row_q4_0` in `quants.c:33` just calls `quantize_row_q4_0_ref`).
For IQ formats that require importance sampling, `from_float_ref` is
often `NULL` (see F03).

---

## 6. Data Layout

### 6.1 Block layout structs

Every quantized weight type has a fixed-size block struct defined in
`ggml-common.h`. The struct contains, in order:

1. **Super-block scale(s)** — one or two `ggml_half` (fp16) values, or
   for NVFP4 an array of UE4M3 bytes, or for MXFP4 a single E8M0 byte.
2. **Sub-block scales / mins** — packed bytes (4-bit, 6-bit, or 8-bit),
   sized by `K_SCALE_SIZE = 12` for K-quants.
3. **Packed weights** — `qs[]` array of bytes holding the per-element
   quantized values, possibly with a separate `qh[]` for high bits (Q5_0,
   Q5_1, Q5_K, Q3_K, Q6_K, IQ2_S, IQ3_S, IQ1_S, IQ1_M).

`static_assert(sizeof(block_X) == expected)` at every struct definition
enforces the on-disk ABI. Example: `static_assert(sizeof(block_q4_K) ==
2*sizeof(ggml_half) + K_SCALE_SIZE + QK_K/2, ...)` (`ggml-common.h:338`).

### 6.2 Activation layout

* **Q8_0 activations** (`block_q8_0`, `ggml-common.h:251-256`):
  `ggml_half d; int8_t qs[32];` — 34 bytes per 32 elements.
* **Q8_1 activations** (`block_q8_1`, `ggml-common.h:258-269`):
  `ggml_half d, s; int8_t qs[32];` — 36 bytes per 32 elements. `s = d *
  sum(qs[i])`, precomputed at quantization time (F11).
* **Q8_K activations** (`block_q8_K`, `ggml-common.h:371-376`):
  `float d; int8_t qs[256]; int16_t bsums[16];` — 292 bytes per 256
  elements. `bsums[j] = sum(qs[j*16..(j+1)*16])`, precomputed (F12).

The activation format chosen depends on the weight format's
`vec_dot_type`. Q4_0 expects Q8_0; Q4_K expects Q8_K; Q4_1 expects Q8_1.
Why the split? Because the dot product algebra differs (F02, F11, F12).

### 6.3 Interleaved / repacked layouts

The block structs above describe the **on-disk** layout. Several CPU
backends (ARM DOTPROD/I8MM, SpacemiT, x86 repack.cpp) re-pack weights
into **interleaved** layouts at load time, optimized for their SIMD
kernel. The repacked buffer is held by an `extra_buffer_type`
(ARTX01-F04). The on-disk layout is never overwritten; repacking is a
runtime cache. ARTX06 audited the on-disk layout only; repack internals
belong to ARTX02/ARTX04/ARTX05.

---

## 7. Memory Layout

### 7.1 Precomputed LUT grids

`ggml-common.h` defines the following large static tables, shared across
backends:

| Table              | Type        | Size    | Used by                          | Location (line) |
| ------------------ | ----------- | ------- | -------------------------------- | --------------- |
| `iq2xxs_grid`      | `uint64_t`  | 256     | IQ2_XXS                          | 560             |
| `iq2xs_grid`       | `uint64_t`  | 512     | IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S    | 627             |
| `iq2s_grid`        | `uint64_t`  | 1024    | IQ2_S                            | 758             |
| `iq3xxs_grid`      | `uint32_t`  | 256     | IQ3_XXS                          | 1017            |
| `iq3s_grid`        | `uint32_t`  | 512     | IQ3_S                            | 1052            |
| `iq1s_grid`        | `uint64_t`  | 2048    | IQ1_S, IQ1_M                     | 1135            |
| `kmask_iq2xs`      | `uint8_t`   | 8       | all IQ vecdots (sign bit mask)   | 509             |
| `ksigns_iq2xs`     | `uint8_t`   | 128     | all IQ vecdots (7-bit → 8 signs) | 513             |
| `ksigns64`         | `uint64_t`  | 128     | (alternate sign mask, x86 AVX-512)| 524            |
| `kvalues_iq4nl`    | `int8_t`    | 16      | IQ4_NL, IQ4_XS                   | 1120            |
| `kvalues_mxfp4`    | `int8_t`    | 16      | MXFP4, NVFP4 (E2M1 ×2)           | 1126            |

Total ~50 KB of static data, all `static const`. Populated at compile
time, never mutated.

### 7.2 Runtime-allocated neighbor tables

`ggml_quantize_init` (for IQ2_XXS/XS/S, IQ1_S/M, IQ3_XXS/S) builds
additional runtime tables (`kgrid_q2xs`, `kmap_q2xs`, `kneighbors_q2xs`)
at `ggml-quants.c:3114-3166`. These are needed only by the *quantizer*;
the *dequantizer* and *vecdot* paths use only the static grids above.
Total runtime cost: ~250 KB per IQ format family, allocated once via
`iq2xs_init_impl` / `iq3xs_init_impl` and freed in
`ggml_quantize_free` (`ggml.c:7891-7903`).

### 7.3 E8M0 / UE4M3 conversion tables

`ggml-cpu.c:3848, 3853` precompute two 256-entry FP32 LUTs:
`ggml_table_f32_e8m0_half[1<<8]` (1 KB) and `ggml_table_f32_ue4m3[1<<8]`
(1 KB). The reference converters `ggml_e8m0_to_fp32_half` and
`ggml_ue4m3_to_fp32` in `ggml-impl.h:477, 502` are pure scalar functions;
the LUTs are an optimization that some ISAs use.

---

## 8. Parallelism Strategy

The quant format layer is parallelism-agnostic. `quantize_row_*` and
`vec_dot_*` operate on a single row at a time and are called by the
matmul path inside the per-thread / per-chunk loop. Parallelism over
rows is owned by `ggml_compute_forward_mul_mat` (ARTX01 §5.5, §8.4).

The one exception is `iq2xs_init_impl` (`ggml-quants.c:3108-3260`):
the neighbor-search loop is `#pragma omp parallel for schedule(dynamic,
64)` when `GGML_USE_OPENMP` is defined. This is the only OpenMP
parallelism in the quant layer.

---

## 9. SIMD / GPU Strategy

ARTX06 audited the *generic* `quants.c` skeleton. By design it contains
**no SIMD**. The generic `ggml_vec_dot_q4_0_q8_0_generic` at
`quants.c:225-259` is a textbook 4-bit-nibble dot product with two
scalar int accumulators (`sumi0`, `sumi1`) combined into one float at
the end. It exists as a reference and as a fallback for ISAs that lack
a hand-tuned kernel.

Per-ISA SIMD for the same formats lives in:

* `arch/x86/quants.c` — AVX2 / AVX-512 / AVX-VNNI variants (ARTX02)
* `arch/arm/quants.c` — NEON / DOTPROD / I8MM / SVE variants (ARTX04,
  ARTX05)
* `arch/riscv/quants.c` — RVV variants (vector-length-dispatched via
  `switch(__riscv_vlen)` at `ggml/src/ggml-cpu/arch/riscv/quants.c:6281`)
* `arch/loongarch/quants.c`, `arch/powerpc/quants.c`,
  `arch/s390/quants.c`, `arch/wasm/quants.c` — analogous per-ISA variants

The dispatch is **link-time**, not runtime. The `arch-fallback.h` macro
layer aliases the generic name to whichever variant the link unit
supplied. There is no function-pointer swap based on CPUID at runtime
inside the quant layer; that decision was made earlier, when the
backend registry selected which .so to load (ARTX01-F12).

---

## 10. Quantization Strategy

This is the central section. It documents every quant format in the
audited commit.

### 10.1 Comprehensive format table

`bpw` = bits per weight (including scale overhead) = `type_size * 8 /
blck_size`. All numbers verified against the `static_assert`s in
`ggml-common.h`.

| Type        | Blck | Bytes/blk | bpw    | Scale format                          | Min/zero-pt                  | vec_dot_type | `from_float` runtime? | Source (struct)       |
| ----------- | ---- | --------- | ------ | ------------------------------------- | ---------------------------- | ------------ | --------------------- | --------------------- |
| `Q1_0`      | 128  | 18        | 1.125  | fp16 `d`                              | implicit (sign bit)          | Q8_0         | yes (`quantize_row_q1_0`) | `ggml-common.h:181`   |
| `Q2_0`      | 64   | 18        | 2.25   | fp16 `d`                              | implicit (4 levels: -1..+2)  | Q8_0         | yes                   | `ggml-common.h:188`   |
| `Q4_0`      | 32   | 18        | 4.5    | fp16 `d`                              | implicit (nibble-8)          | Q8_0         | yes                   | `ggml-common.h:195`   |
| `Q4_1`      | 32   | 20        | 5.0    | fp16 `d`                              | fp16 `m` (min)               | Q8_1         | yes                   | `ggml-common.h:202`   |
| `Q5_0`      | 32   | 22        | 5.5    | fp16 `d`                              | implicit (nibble+qh-16)      | Q8_0         | yes                   | `ggml-common.h:230`   |
| `Q5_1`      | 32   | 24        | 6.0    | fp16 `d`                              | fp16 `m`                     | Q8_1         | yes                   | `ggml-common.h:238`   |
| `Q8_0`      | 32   | 34        | 8.5    | fp16 `d`                              | n/a (int8 quants)            | Q8_0         | yes                   | `ggml-common.h:252`   |
| `Q8_1`      | 32   | 36        | 9.0    | fp16 `d` + fp16 `s=d*Σqs`             | n/a (s folds min)            | Q8_1         | yes                   | `ggml-common.h:259`   |
| `MXFP4`     | 32   | 17        | 4.25   | E8M0 (1 byte, per block)              | n/a                          | Q8_0         | yes                   | `ggml-common.h:215`   |
| `NVFP4`     | 64   | 36        | 4.5    | UE4M3 (4 bytes, per 16-elem sub-blk)  | n/a                          | Q8_0         | yes                   | `ggml-common.h:223`   |
| `Q2_K`      | 256  | 84        | 2.625  | fp16 `d` + 4-bit `scales[16]`         | fp16 `dmin` + 4-bit `mins[16]` | Q8_K       | yes                   | `ggml-common.h:298`   |
| `Q3_K`      | 256  | 110       | 3.4375 | fp16 `d` + 6-bit `scales[16]`         | implicit (hmask)             | Q8_K         | yes                   | `ggml-common.h:315`   |
| `Q4_K`      | 256  | 144       | 4.5    | fp16 `d` + 6-bit `scales[8]` (in 12 B) | fp16 `dmin` + 6-bit `mins[8]` | Q8_K       | yes                   | `ggml-common.h:327`   |
| `Q5_K`      | 256  | 176       | 5.5    | fp16 `d` + 6-bit `scales[8]`          | fp16 `dmin` + 6-bit `mins[8]` | Q8_K       | yes                   | `ggml-common.h:344`   |
| `Q6_K`      | 256  | 210       | 6.5625 | fp16 `d` + 8-bit `scales[16]`         | n/a (no min, signed)         | Q8_K         | yes                   | `ggml-common.h:362`   |
| `Q8_K`      | 256  | 292       | 9.125  | fp32 `d` + precomputed `bsums[16]`    | n/a                          | (intermediate) | yes                 | `ggml-common.h:371`   |
| `TQ1_0`     | 256  | 54        | 1.6875 | fp16 `d`                              | n/a (ternary)                | Q8_K         | yes                   | `ggml-common.h:276`   |
| `TQ2_0`     | 256  | 66        | 2.0625 | fp16 `d`                              | n/a (ternary)                | Q8_K         | yes                   | `ggml-common.h:284`   |
| `IQ2_XXS`   | 256  | 66        | 2.0625 | fp16 `d` + 4-bit LS per 32-elem (in qs) | n/a (signs in ksigns LUT)  | Q8_K         | **NULL**              | `ggml-common.h:381`   |
| `IQ2_XS`    | 256  | 74        | 2.3125 | fp16 `d` + 4-bit `scales[8]`          | n/a                          | Q8_K         | **NULL**              | `ggml-common.h:388`   |
| `IQ2_S`     | 256  | 82        | 2.5625 | fp16 `d` + 4-bit `scales[8]`          | n/a                          | Q8_K         | **NULL (commented)**  | `ggml-common.h:396`   |
| `IQ3_XXS`   | 256  | 98        | 3.0625 | fp16 `d` + 4-bit LS per 32-elem       | n/a (signs in ksigns LUT)    | Q8_K         | **NULL (commented)**  | `ggml-common.h:407`   |
| `IQ3_S`     | 256  | 110       | 3.4375 | fp16 `d` + 4-bit `scales[4]`          | n/a                          | Q8_K         | **NULL (commented)**  | `ggml-common.h:415`   |
| `IQ1_S`     | 256  | 50        | 1.5625 | fp16 `d` + 3-bit LS per 32-elem       | n/a (delta-encoded)          | Q8_K         | **NULL**              | `ggml-common.h:425`   |
| `IQ1_M`     | 256  | 56        | 1.75   | fp16 (packed in scales) + 3-bit LS    | n/a (delta-encoded)          | Q8_K         | **NULL**              | `ggml-common.h:433`   |
| `IQ4_NL`    | 32   | 18        | 4.5    | fp16 `d`                              | n/a (non-linear LUT)         | Q8_0         | yes                   | `ggml-common.h:448`   |
| `IQ4_XS`    | 256  | 136       | 4.25   | fp16 `d` + 6-bit `scales[8]`          | n/a (non-linear LUT)         | Q8_K         | yes (via `quantize_iq4_xs`) | `ggml-common.h:454` |
| `BF16`      | 1    | 2         | 16     | n/a                                   | n/a                          | BF16         | n/a (not quantized)   | (ggml.h:420)          |
| `F16`       | 1    | 2         | 16     | n/a                                   | n/a                          | F16          | n/a (not quantized)   | (ggml.h:391)          |
| `F32`       | 1    | 4         | 32     | n/a                                   | n/a                          | F32          | n/a (not quantized)   | (ggml.h:390)          |

### 10.2 Simple block quants (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / Q8_1)

These are the oldest formats, defined in `ggml-common.h:180-269`. Block
size 32 (Q1_0: 128, Q2_0: 64). Each block has one fp16 scale (`d`); the
`_1` variants additionally have one fp16 min (`m`).

* **Q4_0** — 4-bit nibbles, value = `nibble - 8` (signed), scale `d`.
  Dot product: `Σ (nibble-8) * q8 * d_x * d_y` (`quants.c:225-259`).
* **Q4_1** — 4-bit nibbles, value = `nibble` (unsigned, 0..15), scale
  `d`, min `m`. Dot product: `d_x * d_y * Σ nibble*q8 + m_x * s_y`,
  where `s_y = d_y * Σ q8` is precomputed in Q8_1 (`quants.c:262-296`).
* **Q5_0 / Q5_1** — like Q4_0/Q4_1 but with an extra `qh[4]` byte array
  providing the 5th bit per element. The 5th bit is unpacked via
  `((qh & (1u << j)) >> j) << 4` (`quants.c:391-395`).
* **Q8_0 / Q8_1** — 8-bit values. Used as weight formats only in
  low-rank scenarios; primarily used as `vec_dot_type` activations
  (§10.7).
* **Q1_0 / Q2_0** — minimal bit-width block quants with 128- and 64-
  element blocks respectively. See F13 / §10.3.

### 10.3 Q1_0 / Q2_0 — many-to-one activation ratio

Q1_0 packs 128 weights per block (one bit each + fp16 scale), so its
block size is 4× the Q8_0 activation block size (32). The generic
vecdot at `quants.c:127-175` walks `y[i*4 + k]` for `k = 0..3` — i.e.
one Q1_0 block consumes four Q8_0 activation blocks. The same pattern
applies to Q2_0 with a factor of 2 (`quants.c:177-223`).

### 10.4 K-quant super-block structure

All five K-quants (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) use `QK_K = 256` as the
super-block size (`ggml-common.h:89`). Within each 256-element
super-block:

* Q2_K / Q3_K / Q6_K: **16 sub-blocks of 16 elements each**.
* Q4_K / Q5_K: **8 sub-blocks of 32 elements each**.

The super-block has:

* one fp16 `d` (super-block scale for the per-sub-block scales)
* for Q2_K/Q4_K/Q5_K: one fp16 `dmin` (super-block scale for the per-
  sub-block mins)
* for Q3_K/Q6_K: no `dmin` (Q3_K has implicit zero via `hmask`; Q6_K is
  signed and has no min)
* a packed `scales` array holding per-sub-block scales (and mins where
  applicable)

**6-bit scale packing (Q4_K / Q5_K).** `K_SCALE_SIZE = 12` bytes
(`ggml-common.h:90`). These 12 bytes hold 8 sub-block scales + 8 sub-
block mins, each 6 bits → 8×6 + 8×6 = 96 bits = 12 bytes ✓. The 6-bit
values are not contiguous; they are interleaved across four 32-bit words
in the unpacking code at `quants.c:736-741` (kmask1=0x3f3f3f3f,
kmask2=0x0f0f0f0f, kmask3=0x03030303). The bit-shuffle is a fixed
property of the on-disk format; any implementation must replicate it.

**4-bit scale packing (Q2_K).** `scales[QK_K/16] = scales[16]` bytes
(`ggml-common.h:299`). Each byte holds one 4-bit scale (low nibble) and
one 4-bit min (high nibble) for the corresponding sub-block. Unpacked
at `quants.c:583-587`: `sc[j] >> 4` for the min, `sc[j] & 0xF` for the
scale.

**6-bit scale packing (Q3_K).** `scales[12]` bytes hold 16 6-bit
scales (16×6 = 96 bits = 12 bytes ✓). The unpacking at
`quants.c:675-680` uses `kmask1 = 0x03030303` and `kmask2 = 0x0f0f0f0f`
to extract the 6-bit values from a 4×32-bit shuffle. Q3_K has no mins;
the zero offset is implicit in the `hmask` bit (see F05).

**8-bit scale (Q6_K).** `int8_t scales[QK_K/16] = scales[16]` bytes
(`ggml-common.h:365`). Each sub-block has one signed 8-bit scale; no
packing needed. See F06.

### 10.5 Q3_K hmask pattern

Q3_K stores 3-bit weights as 2-bit low parts (`qs[QK_K/4]` = 64 bytes,
4 elements per byte, 2 bits each) plus a 1-bit high part in
`hmask[QK_K/8]` = 32 bytes (1 bit per element). The recovered 3-bit
value is `(q3_byte & 3) - (hmask_bit ? 0 : 4)`, producing a signed
4-bit value in `{-4, -3, -2, -1, 0, 1, 2, 3}` (8 levels). The 4-bit
scale (unpacked from `scales[12]`) then multiplies this. See
`quants.c:651-672`.

### 10.6 I-quant (importance quantization) — LUT-driven formats

I-quants encode 8-element (or 4-element) groups of weights as a single
index into a precomputed codebook grid. The grid stores *signed 8-byte
patterns* of weights; the index plus a sign mask reconstructs the
original 8 weights. The principle: instead of quantizing each weight
independently, quantize a *group* of 8 weights jointly, choosing the
codebook entry that best fits the group.

* **IQ2_XXS** — uses `iq2xxs_grid[256]` (256 entries of `uint64_t`).
  Each `uint16_t` in `qs[]` holds a 9-bit grid index + 7-bit sign index.
  Per-32-element block has one 4-bit LS (level scale) packed in the high
  nibble of the second `qs` word (`quants.c:929-945`).
* **IQ2_XS** — uses `iq2xs_grid[512]` (512 entries). Per-32-element
  block has two 4-bit LS values packed in `scales[QK_K/32]` = 8 bytes
  (`quants.c:948-996`).
* **IQ2_S** — uses `iq2s_grid[1024]` (1024 entries, a superset of
  IQ2_XS). Per-32-element block has two 4-bit LS values in `scales[8]`,
  and a separate `qh[QK_K/32] = 8` bytes for high bits of the grid
  index (`quants.c:998-1048`).
* **IQ3_XXS / IQ3_S** — 3-bit I-quants. Use `iq3xxs_grid[256]` /
  `iq3s_grid[512]` of `uint32_t` (4 signed bytes per entry, two grid
  lookups per 8 weights). `quants.c:1050-1148`.
* **IQ1_S / IQ1_M** — 1-bit I-quants using `iq1s_grid[2048]`. Each
  group of 8 weights is encoded as a single grid index (11-bit for
  IQ1_S, packed with a 3-bit LS in the `qh` word). Includes a delta-
  encoded zero-offset (`IQ1S_DELTA = 0.125`, `IQ1M_DELTA = 0.125` at
  `ggml-common.h:1132-1133`). `quants.c:1150-1252`.
* **IQ4_NL** — non-linear 4-bit quant using `kvalues_iq4nl[16]` (16
  non-uniformly-spaced int8 values: `{-127,-104,-83,-65,-49,-35,-22,
  -10,1,13,25,38,53,69,89,113}` at `ggml-common.h:1120-1122`). Each
  nibble indexes into this table. Block size 32, like simple Q4_NL.
* **IQ4_XS** — K-quant super-block of 256 elements using the same
  `kvalues_iq4nl` LUT. Per-32-element sub-block has a 6-bit scale
  packed in `scales_l[QK_K/64]` + `scales_h (uint16_t)`
  (`quants.c:1283-1327`).

The grids encode *signed, non-uniform* weight patterns chosen by
importance sampling during offline quantization. The grid is read-only
at inference time; only the index and sign mask are stored per group.

### 10.7 Activation formats and the `vec_dot_type` indirection

Every weight format declares an activation format it expects, via
`type_traits_cpu[type].vec_dot_type`. The matmul path converts src1
from F32 to that format once up-front (`ggml-cpu.c:1322-1355`,
audited in ARTX01 §5.5).

The reason for the indirection is algebraic. The matmul computes
`Σ weight[i] * activation[i]`, and different weight formats decompose
the multiplication differently:

* **Q4_0 dot with Q8_0** (`quants.c:225`): `Σ (nibble-8) * q8 * d *
  d`. The scale `d` lives once per block on each side; the kernel
  computes `Σ (nibble-8)*q8` as int32, then multiplies by `d_x*d_y` as
  float.
* **Q4_1 dot with Q8_1** (`quants.c:262`): `d_x*d_y * Σ nibble*q8 +
  m_x * s_y` where `s_y = d_y * Σ q8` is precomputed in Q8_1. The
  min-offset term would require a second int32 reduction (Σ q8) per
  block; Q8_1 precomputes it as `s` to avoid the second reduction.
* **Q4_K dot with Q8_K** (`quants.c:696`): `d_x*d_y * Σ scale[j] *
  Σ_{i in j} nibble*q8 - dmin_x*d_y * Σ min[j] * Σ_{i in j} q8`. The
  inner `Σ_{i in j} q8` per sub-block (16 elements) is precomputed in
  `Q8_K.bsums[16]`, avoiding 16 redundant reductions per super-block.

So `vec_dot_type` exists because each weight family has its own
algebraic structure that benefits from a different precomputed
activation layout. Sharing one activation format across all weight
types would lose these optimizations. See F02, F11, F12.

### 10.8 Ternary quantization (TQ1_0 / TQ2_0)

For BitNet b1.58 and TriLMs, weights are constrained to {-1, 0, +1}.

* **TQ2_0** — 2 bits per element, trivially encoding 3 levels. Block
  size 256, `qs[QK_K/4] = 64` bytes (4 elements per byte, 2 bits each),
  fp16 `d`. Value = `((qs >> shift) & 3) - 1`. 2.0625 bpw
  (`ggml-common.h:284-288`).
* **TQ1_0** — mixed-radix packing. Each byte of `qs[]` encodes 5
  ternary values (3^5 = 243 ≤ 256), with `qh[]` encoding the remaining
  4 values per byte (3^4 = 81 ≤ 256). Block layout: `qs[(QK_K - 4 *
  QK_K/64)/5] = qs[48]`, `qh[QK_K/64] = qh[4]`, fp16 `d`. Achieves
  1.6875 bpw — closer to the 1.585 information-theoretic minimum
  (`ggml-common.h:276-281`). The vecdot at `quants.c:481-531` is
  scalar and uses a `pow3[6]` lookup; it is the slowest vecdot in the
  tree.

### 10.9 Microscaling formats (MXFP4 / NVFP4)

OCP MX-compliant 4-bit formats. Both use the E2M1 lookup table
`kvalues_mxfp4[16] = {0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12}`
(`ggml-common.h:1126-1129`) — these are E2M1 float values ×2 (the ×2
convention matches the OCP MX spec). The high bit of the nibble is the
sign; the low 3 bits index `{0,1,2,3,4,6,8,12}`.

* **MXFP4** — block size 32. One E8M0 byte scale per block (8-bit
  biased exponent, value = `2^(e-127)`). The scale is converted to
  FP32 via `ggml_e8m0_to_fp32_half` (`ggml-impl.h:477`), which returns
  `0.5 * 2^(e-127)` to fold in the ×2 E2M1 convention
  (`ggml-impl.h:475-495`). Quantizer at `ggml-quants.c:350-382`.
* **NVFP4** — block size 64, with 4 sub-blocks of 16 elements. Each
  sub-block has its own UE4M3 scale (4 bytes total, one per 16-element
  sub-block). UE4M3 is unsigned 4-exp + 3-mantissa, value = `0.5 *
  (1 + man/8) * 2^(exp-7)` (the ×0.5 again folds in the E2M1 ×2
  convention, `ggml-impl.h:502-515`). Quantizer at
  `ggml-quants.c:384-417`.

The vecdot for MXFP4 at `quants.c:298-327` is straightforward: one
scale per block. The NVFP4 vecdot at `quants.c:330-363` is more
complex: it loops over 4 sub-blocks, each with its own UE4M3 scale,
mapping onto 2 Q8_0 activation blocks (since NVFP4's 64-element block
spans 2 Q8_0 blocks).

### 10.10 `ggml_quantize_init` — runtime initialization for I-quants

IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M, IQ3_XXS, IQ3_S require runtime
initialization before they can be quantized (`ggml.c:7873-7889`). The
init builds `kmap_q2xs` (reverse-lookup table from quantized pattern to
grid index) and `kneighbors_q2xs` (nearest-neighbor list for patterns
not on the grid). Both are populated by `iq2xs_init_impl` /
`iq3xs_init_impl` in `ggml-quants.c:3108-3260` (OpenMP-parallelized).

The dequantize / vecdot paths do **not** use these runtime tables; they
use only the static grids in `ggml-common.h`. So inference-only
workloads (load pre-quantized weights, run forward pass) never need
`ggml_quantize_init`. Only the offline quantizer (called from
`ggml_quantize_chunk` at `ggml.c:7913-7988`) requires it.

`ggml_quantize_requires_imatrix` (`ggml.c:7905-7911`) declares that
IQ2_XXS, IQ2_XS, IQ1_S additionally require an *importance matrix*
(per-channel variance estimate from calibration data) at quantization
time. Without an imatrix, these formats cannot be quantized at all.
This is the strictest constraint in the format taxonomy.

### 10.11 BF16 / F16 / F32 — non-quantized formats

For completeness: BF16 (`blck_size=1`, `type_size=2`, no scale, no
min) and F16 (`blck_size=1`, `type_size=2`) are stored as raw IEEE
754 bytes. Their `vec_dot_type` equals themselves; the matmul path
does no activation conversion. Their `from_float` is the FP32→FP16/BF16
converter (`ggml_cpu_fp32_to_fp16` / `ggml_cpu_fp32_to_bf16`). F32 is
the identity format. These are not "quantization" in any meaningful
sense, but they participate in the same `type_traits_cpu` table.

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

Every `vec_dot_*` kernel accumulates into one or more int32 or float
accumulators and horizontally reduces at the end. The reduction order
depends on the kernel (1-accumulator scalar reference, 8-accumulator
NEON, 16-accumulator AVX-512 — see ARTX02/ARTX04/ARTX05). Per-block
results are summed across blocks in source order. The result is
deterministic per (ISA, thread count) but differs at the ULP level
across ISAs or thread counts. This is inherited from ARTX01 §11.1; the
quant layer adds no new reassociation concerns.

### 11.2 K-quant scale unpacking determinism

The 6-bit / 4-bit scale unpacking at `quants.c:675-680, 736-741, 816-
821` is deterministic and bit-exact. Every backend unpacks scales the
same way; the on-disk layout is part of the format ABI.

### 11.3 I-quant LUT determinism

The `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`,
`iq3s_grid`, `iq1s_grid` tables are `static const` in `ggml-common.h`
and are byte-identical across all backends (CPU/CUDA/Metal/Vulkan all
include `ggml-common.h`). So a model quantized by one backend loads
into another and produces bit-identical dequantized weights, modulo
the per-ISA vecdot reassociation (§11.1).

### 11.4 Q8_K `bsums` precomputation

`quantize_row_q8_K_ref` at `ggml-quants.c:2795-2801` computes
`bsums[j] = sum(qs[j*16..(j+1)*16])` as `int16_t`. The sum of 16
int8 values can range from -2048 to +2048, which fits in `int16_t`
(range -32768..+32767). However, the cumulative `Σ bsums[j]` over 16
sub-blocks can range from -32768 to +32768 — *exactly* `int16_t`
overflow boundary. The K-quant vecdot uses `bsums` as int32 in the
multiplication (`y[i].bsums[j] * sc[j] >> 4` at `quants.c:587`), so
the per-sub-block sum is safe. The cumulative sum is never materialized
in `bsums`. No correctness issue, but a subtle invariant: any code that
sums `bsums[]` across sub-blocks into an int16 must handle overflow.

### 11.5 Q1_0 / Q2_0 activation ratio

The generic Q1_0 vecdot at `quants.c:148-166` reads `y[i*4 + k]` for
`k = 0..3`. If the caller passes a Q8_0 activation buffer that is not
sized to `4 * nb` (where `nb = n / QK1_0`), it reads out of bounds.
The matmul path in `ggml-cpu.c:1322-1355` allocates `wdata` with
`nbw1 = ggml_row_size(vec_dot_type, ne10)`, which for `vec_dot_type =
Q8_0` and Q1_0 weights yields `ne10 / 32 * 34` bytes per row — i.e.
exactly `4 * (ne10/128) * sizeof(block_q8_0)` when `ne10` is a multiple
of 128. The contract holds because `ggml_blck_size(Q1_0) = 128` and
the matmul path requires `ne10 % blck_size(src0_type) == 0`
(`ggml-cpu.c:1282`).

### 11.6 MXFP4 / NVFP4 scale conventions

The `kvalues_mxfp4` table stores E2M1 values ×2 (e.g., E2M1 code 4 = 2.0
becomes 4 in the table). Both `ggml_e8m0_to_fp32_half` and
`ggml_ue4m3_to_fp32` return half of their nominal value (`* 0.5f` at
`ggml-impl.h:514, 477-495`) to compensate. The product `scale * kvalue`
therefore equals the true E2M1 value × the true scale. Convention is
consistent within `ggml-impl.h` but unusual; an implementation that
forgets the ×0.5 will produce weights that are 2× too large. Documented
only by the comment at `ggml-impl.h:475` ("Useful with MXFP4
quantization since the E0M2 values are doubled").

### 11.7 IQ1_M nibble-spread scale

`block_iq1_m` (`ggml-common.h:433-438`) has no top-level `d` field.
The 16-bit fp16 super-block scale is reconstructed at runtime from four
scale bytes: `scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) |
((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000)` (`quants.c:1218`). Each
scale byte also holds a 3-bit LS for one sub-block. So each byte does
double duty: top nibble contributes one nibble to the super-block
scale; bottom 6 bits hold two 3-bit LS values. An implementation that
mismasks either field will produce nonsense weights.

### 11.8 IQ1_S / IQ1_M delta encoding

IQ1_S and IQ1_M add a `IQ1S_DELTA = 0.125` / `IQ1M_DELTA = 0.125`
correction term to compensate for the 1-bit quantization bias. The
vecdot at `quants.c:1183-1187` computes `sumi + IQ1S_DELTA * sumi1`
where `sumi1` accumulates the delta contributions. The constant 0.125
is hard-coded; any change to the delta requires re-quantization of
all IQ1_S / IQ1_M models.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                              | Where                                    | Notes                                                                  |
| ----------------------------------------- | ---------------------------------------- | ---------------------------------------------------------------------- |
| `vec_dot_type` indirection                | `ggml-cpu.c:214-415`                     | Lets each weight family pick the activation format that minimizes per-block work. |
| 6-bit scale packing (K-quants)            | `ggml-common.h:90, 335`                  | `K_SCALE_SIZE = 12` packs 16 6-bit scales into 12 bytes (vs 16 bytes if unpacked). |
| Precomputed `bsums` (Q8_K)                | `ggml-common.h:374`                      | Eliminates per-sub-block reduction in the K-quant vecdot hot path.     |
| Precomputed `s = d*Σqs` (Q8_1)            | `ggml-common.h:263`                      | Eliminates the min-offset reduction in Q4_1/Q5_1 vecdot.               |
| E2M1 codebook LUT (MXFP4/NVFP4)           | `ggml-common.h:1126-1129`                | 4-bit nibble → int8 lookup replaces FP arithmetic in vecdot.           |
| IQ codebook grids                         | `ggml-common.h:560, 627, 758, 1017, 1052, 1135` | 8-weight joint quantization replaces 8 independent quantizations. |
| TQ1_0 mixed-radix packing (3^5/byte)      | `ggml-common.h:276-281`                  | 1.6875 bpw vs 2.0625 bpw for TQ2_0; closer to the 1.585 minimum.      |
| `static_assert`-enforced block sizes      | every struct in `ggml-common.h`          | Compile-time guarantee of the on-disk ABI.                             |
| `from_float = NULL` for inference-only IQ | `ggml-cpu.c:337, 343, 368, 374`          | Prevents accidental runtime quantization of formats that need offline init. |
| Q1_0 / Q2_0 minimal-block                 | `ggml-common.h:180, 187`                 | Smaller block-size overhead (1/128 or 1/64 scale bytes per weight) than Q4_0 (1/32). |
| NVFP4 per-sub-block UE4M3 scale           | `ggml-common.h:223-227`                  | Finer scale granularity than MXFP4's per-block E8M0.                   |
| `ggml_quantize_init` deferred-build       | `ggml.c:7873-7889`                       | Runtime LUT build only for quantizers; inference-only loads skip it.   |

### 12.2 Optimizations *not* present

* **No runtime kernel swap.** Once the .so is loaded, the kernel is
  fixed (ARTX01-F11). The quant layer has no "select best kernel per
  shape" mechanism; the only shape-aware switch is the `nrc=2` vs
  `nrc=1` flag for ARM I8MM.
* **No fused dequantize + vecdot.** Every vecdot kernel re-decodes
  the block layout from scratch on each call. There is no cache of
  decoded weights across calls (the repack layer in
  `arch/<isa>/repack.cpp` provides an interleaved layout but not a
  decoded one).
* **No JIT.** Every kernel is statically compiled. There is no
  mechanism to generate a specialized kernel for a known block size or
  known scale pattern.
* **No FP8 activations.** All activations are Q8_0, Q8_1, Q8_K, F16,
  BF16, or F32. No `vec_dot_type = FP8` exists.

---

## 13. Architectural Strengths

1. **`static_assert`-enforced block layout ABI.** Every `block_*`
   struct in `ggml-common.h` is followed by a `static_assert` that
   pins its `sizeof`. Any change to the struct breaks the build across
   every backend simultaneously. This is the single most important
   correctness property of the format layer.

2. **Two-tier traits table.** Splitting `type_traits` (storage facts,
   shared across backends) from `type_traits_cpu` (compute facts,
   backend-specific) lets the storage ABI stay stable while each
   backend picks its own kernels. Adding a new format means one entry
   in each table.

3. **`vec_dot_type` indirection.** Each weight format declares which
   activation format it expects. This lets the matmul path pre-convert
   activations once, and lets each weight family use the algebraic
   decomposition that minimizes per-block work (F02, F11, F12).

4. **K-quant 6-bit scale packing.** `K_SCALE_SIZE = 12` packs 16
   6-bit scales (Q3_K) or 8 scales + 8 mins (Q4_K, Q5_K) into 12
   bytes. The unpacking is a fixed bit shuffle that any backend can
   replicate. Saves 4-8 bytes per super-block vs naive 8-bit storage.

5. **I-quant codebook grids as `static const`.** The 6 IQ grids total
   ~50 KB but encode ~10^6 possible weight patterns. The dequantize
   path is a single indexed load; the cost is borne entirely at
   quantization time.

6. **MXFP4 / NVFP4 reuse `kvalues_mxfp4`.** Both 4-bit float formats
   share the same E2M1 lookup. The differences (E8M0 vs UE4M3 scale,
   1 vs 4 sub-blocks) are confined to the scale path.

7. **`from_float = NULL` as a contract.** Setting `from_float` to NULL
   for inference-only IQ formats is a *compile-time-strength* signal:
   any code that tries to runtime-quantize IQ2_XXS will fail at the
   type-traits lookup. This prevents silent misuse.

8. **TQ1_0 mixed-radix packing.** Packing 5 ternary values per byte
   (3^5 = 243 ≤ 256) is a non-obvious optimization that beats naive
   2-bit-per-value (TQ2_0) by 18% on storage. The cost is a more
   complex vecdot, but for memory-bound ternary inference the
   storage win dominates.

---

## 14. Architectural Weaknesses

### W1 — IQ1_M nibble-spread scale is a maintenance hazard

**Evidence:** `ggml-common.h:433-438` (struct),
`quants.c:1216-1218` (reconstruction).

**Impact:** The 16-bit fp16 super-block scale is reconstructed from
four scale bytes' top nibbles via a 4-step shift-and-or. Each scale
byte also holds a 3-bit LS for a sub-block. A masked-bit bug anywhere
in the reconstruction produces silently wrong weights. No compile-time
protection; only runtime test coverage catches this.

### W2 — TQ1_0 vecdot is the slowest in the tree

**Evidence:** `quants.c:481-531`. Scalar loop with `pow3[6]` lookup
and 5-deep inner nesting. No SIMD acceleration in the generic path;
per-ISA overrides exist only for x86 (`arch/x86/quants.c:1376`) and
ARM (`arch/arm/quants.c:1397`).

**Impact:** TQ1_0 inference on ISAs without a hand-tuned kernel
(LoongArch, PowerPC, s390, WASM, RISC-V) runs at scalar speed.

### W3 — `from_float = NULL` blocks runtime quantization

**Evidence:** `ggml-cpu.c:337, 343, 368, 374` (IQ2_XXS, IQ2_XS, IQ1_S,
IQ1_M), `ggml-cpu.c:349-365` (IQ3_XXS, IQ3_S, IQ2_S commented out).

**Impact:** These formats cannot be quantized at runtime by the
generic API. Users must pre-quantize offline using
`ggml_quantize_chunk` (which calls the special-purpose
`quantize_iq2_xxs` / `quantize_iq3_xxs` / etc. functions). The
`from_float` contract is thus split: simple formats expose
`from_float` for runtime use; IQ formats expose only
`from_float_ref` (which may also be NULL).

### W4 — No runtime kernel swap; `type_traits_cpu` is `static const`

Inherited from ARTX01-F11. A tuned vecdot kernel cannot be installed
at runtime; the only extension point is `extra_buffer_type` (ARTX01-
F04), which requires a full buffer-type registration.

### W5 — IQ3_XXS / IQ3_S / IQ2_S `from_float` is commented out

**Evidence:** `ggml-cpu.c:349-365`:
```c
[GGML_TYPE_IQ3_XXS] = {
    // NOTE: from_float for iq3 and iq2_s was removed because these quants require initialization in ggml_quantize_init
    //.from_float               = quantize_row_iq3_xxs,
    .vec_dot                  = ggml_vec_dot_iq3_xxs_q8_K,
```

**Impact:** These formats *do* have a quantizer (`quantize_iq3_xxs`
exists in `ggml-quants.c`), but the type-traits entry is commented
out. So runtime quantization through `type_traits_cpu[type].from_float`
returns NULL, forcing callers to know to use `quantize_iq3_xxs`
directly. The base `type_traits[].from_float_ref` is non-NULL for
IQ3_XXS (`ggml.c:829`) but NULL for IQ3_S and IQ2_S — inconsistent.

### W6 — Q1_0 / Q2_0 activation ratio is implicit

**Evidence:** `quants.c:148-166, 199-217`. The Q1_0 vecdot reads
`y[i*4 + k]`; the Q2_0 vecdot reads `y[i*2 + k]`. The 4× / 2× ratio is
implicit in the loop bounds and not enforced by any type-traits field.
A caller that mis-sizes the activation buffer will read out of bounds.

### W7 — `kvalues_iq4nl` is misnamed

**Evidence:** `ggml-common.h:1119-1122`:
```c
// TODO: fix name to kvalues_iq4_nl
GGML_TABLE_BEGIN(int8_t, kvalues_iq4nl, 16)
```

The TODO has been present for at least one release cycle. Cosmetic but
indicates technical debt in the naming convention.

### W8 — No quantization-quality report

The quant layer has no mechanism to report the per-block error
introduced by quantization. The `quantize_*` functions either succeed
or assert-fail. There is no `quantize_with_error` API. This makes it
hard to compare quantization quality across formats at runtime.

### W9 — `IQ1S_DELTA` / `IQ1M_DELTA` are hard-coded

**Evidence:** `ggml-common.h:1132-1133`. Both are `0.125f`. Changing
them requires re-quantizing all IQ1_S / IQ1_M models. No versioning
mechanism in the format header.

### W10 — BF16 / F16 / F32混 in the same `type_traits_cpu` table

**Evidence:** `ggml-cpu.c:215-220, 221-226, 394-399`. The non-
quantized types share the table with quantized types, even though
their `from_float` is just an FP converter and their `vec_dot` is a
plain F16/BF16 dot. This is a minor abstraction leak; a separate
"float-format traits" table would be cleaner.

---

## 15. GwenLand Mapping

| GwenLand module    | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| ------------------ | ---------------------------------------- | ---- | --------- |
| `glproc`           | **ADOPT** | `static_assert`-enforced block structs | Guarantees on-disk ABI stability; same trick works in any backend. |
| `glproc`           | **ADOPT** | Two-tier traits table (`type_traits` storage + `type_traits_cpu` compute) | Clean separation of storage ABI from kernel choice. |
| `glproc`           | **ADOPT** | `vec_dot_type` indirection | Each weight format picks its optimal activation format. |
| `glproc`           | **ADOPT** | K-quant `K_SCALE_SIZE = 12` 6-bit packing | Saves 4-8 bytes per super-block; the unpacking is a fixed shuffle. |
| `glproc`           | **ADOPT** | I-quant codebook LUTs as `static const` | ~50 KB buys ~10^6 patterns; quant-time cost only. |
| `glproc`           | **ADOPT** | `from_float = NULL` contract for inference-only IQ | Compile-time signal that prevents misuse. |
| `glproc`           | **ADOPT** | MXFP4 E8M0 + NVFP4 UE4M3 scale paths | OCP-compliant; lets GwenLand load MX-compliant models unmodified. |
| `glproc`           | **ADAPT** | TQ1_0 mixed-radix packing | Keep the packing; provide a SIMD vecdot for at least x86 and ARM. |
| `glproc`           | **ADAPT** | IQ1_M nibble-spread scale | Keep the format; add a unit test that round-trips every scale nibble. |
| `glproc`           | **ADAPT** | `kvalues_iq4nl` non-linear LUT | Adopt; rename to `kvalues_iq4_nl` (fix the TODO at ggml-common.h:1119). |
| `glcuda`           | **ADOPT** | Identical block-layout structs | Same `ggml-common.h` includes compile in CUDA; zero porting effort. |
| `glmetal`          | **ADOPT** | Identical block-layout structs | Same as above; `GGML_COMMON_IMPL_METAL` macro already exists. |
| `glvulkan`         | **ADOPT** | Identical block-layout structs | Same; the LUT tables can be uploaded as SSBOs. |
| `GATE`             | **ADOPT** | `ggml_quantize_init` deferred-build pattern | Init only the formats actually being quantized; skip for inference-only loads. |
| `GATE`             | **ADAPT** | `ggml_quantize_requires_imatrix` flag | Extend to a per-format "requires X" trait (imatrix, calibration data, etc.). |
| `GATE`             | **REJECT**| Commenting out `from_float` for IQ3_XXS/IQ3_S/IQ2_S | Use `from_float = NULL` consistently (or a separate `from_float_quantize_only` field) so the contract is uniform. |

---

## 16. Recommendations

### R1 — ADOPT block-layout structs as GwenLand's on-disk weight ABI
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `glproc`, `glcuda`, `glmetal`, `glvulkan` should all include
the same `ggml-common.h` block-layout definitions. The `static_assert`s
pin the ABI. No extension is needed; this is a direct copy.

### R2 — ADOPT `vec_dot_type` indirection
**Priority:** Critical
**Difficulty:** S
**Dependencies:** R1
Every weight format in GwenLand's `gl_type_traits` table should declare
its expected activation format. The matmul path should pre-convert
src1 to that format once. Same contract as `type_traits_cpu.vec_dot_type`.

### R3 — ADOPT K-quant 6-bit scale packing
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Replicate `K_SCALE_SIZE = 12` and the `kmask1/kmask2/kmask3` unpacking
exactly. Saves 4-8 bytes per super-block; the unpacking is deterministic
and cheap.

### R4 — ADOPT I-quant codebook grids as static data
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Copy `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`,
`iq3s_grid`, `iq1s_grid`, `kmask_iq2xs`, `ksigns_iq2xs` verbatim. ~50 KB
of static data; the grids are part of the format ABI.

### R5 — ADOPT `from_float = NULL` for inference-only formats
**Priority:** High
**Difficulty:** XS
**Dependencies:** R2
Any format that requires offline importance sampling (IQ2_XXS, IQ2_XS,
IQ1_S, IQ1_M, and — per W5 — IQ3_XXS, IQ3_S, IQ2_S) should set
`from_float = NULL` in the traits table. The matmul path already
handles `from_float = NULL` for activations because activations are
always converted from F32 via the `vec_dot_type`'s `from_float`, never
the weight type's. So setting `from_float = NULL` on a weight type
blocks only weight-side runtime quantization, which is the intent.

### R6 — ADOPT MXFP4 E8M0 + NVFP4 UE4M3 scale paths
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Implement `gl_e8m0_to_fp32_half` and `gl_ue4m3_to_fp32` matching
`ggml-impl.h:477, 502`. Replicate the `* 0.5f` convention to match
the `kvalues_mxfp4` ×2 LUT.

### R7 — ADAPT TQ1_0 mixed-radix packing with SIMD vecdot
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1, R4
Keep the TQ1_0 packing scheme (1.6875 bpw). Provide a SIMD vecdot for
at least x86 AVX2 and ARM NEON; the generic scalar path at
`quants.c:481` is the slowest in the tree.

### R8 — ADAPT IQ1_M with explicit scale-reconstruction test
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1, R4
Adopt the IQ1_M format. Add a round-trip test that verifies every
combination of (super-block scale nibbles, per-sub-block LS) decodes
correctly. The 4-step shift-and-or at `quants.c:1218` is fragile.

### R9 — REJECT the commented-out `from_float` pattern
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** R5
Do not comment out `from_float` entries (as `ggml-cpu.c:349-365` does
for IQ3_XXS, IQ3_S, IQ2_S). Use `from_float = NULL` explicitly, and
document why. The commented-out form is invisible to grep and creates
the inconsistency noted in W5.

### R10 — ADOPT `ggml_quantize_init` deferred-build pattern
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R5
Build the IQ quantization LUTs (kmap, kneighbors) lazily, only when
the corresponding format is actually being quantized. Inference-only
loads skip the build entirely. Use a critical section to make the
build thread-safe.

### R11 — ADOPT `ggml_quantize_requires_imatrix` flag
**Priority:** Low
**Difficulty:** XS
**Dependencies:** R5
Add a per-format boolean `requires_imatrix` to the traits table.
Formats that need it: IQ2_XXS, IQ2_XS, IQ1_S. The matmul / quantize
path can check this before requiring an imatrix argument.

### R12 — ADAPT `kvalues_iq4nl` naming
**Priority:** Low
**Difficulty:** XS
**Dependencies:** R1
Rename to `kvalues_iq4_nl` (fix the upstream TODO at
`ggml-common.h:1119`). Trivial cleanup; aligns with the `IQ4_NL` type
name.

---

## 17. Findings

### Finding ARTX06-F01

```
Finding ID:           ARTX06-F01
Category:             ADOPT
Engine:               CPU (shared with CUDA, Metal, Vulkan)
Component:            Block layout ABI
Source File:          ggml/src/ggml-common.h
Function:             block_q4_0, block_q4_K, block_iq2_xxs, ... (28 structs)
Lines:                180-460
Summary:              Every quant format's on-disk layout is pinned by a
                      static_assert(sizeof(block_X) == expected) immediately
                      after the struct definition, shared across all backends.
Observation:          ggml-common.h is included by the CPU backend (quants.c),
                      by the CUDA backend (via GGML_COMMON_DECL_CUDA), by the
                      Metal backend (via GGML_COMMON_DECL_METAL), by the SYCL
                      backend (via GGML_COMMON_DECL_SYCL), and by the reference
                      path (via GGML_COMMON_IMPL_C). Every backend sees the
                      same struct definitions and the same static_asserts.
                      A change to any block layout breaks the build across
                      every backend simultaneously, forcing a coordinated
                      ABI bump.
Evidence:             ggml-common.h:185 (Q1_0 assert), :192 (Q2_0), :199 (Q4_0),
                      :212 (Q4_1), :219 (MXFP4), :227 (NVFP4), :235 (Q5_0),
                      :249 (Q5_1), :256 (Q8_0), :269 (Q8_1), :281 (TQ1_0),
                      :288 (TQ2_0), :309 (Q2_K), :321 (Q3_K), :338 (Q4_K),
                      :356 (Q5_K), :368 (Q6_K), :376 (Q8_K), :385 (IQ2_XXS),
                      :393 (IQ2_XS), :402 (IQ2_S), :411 (IQ3_XXS), :422 (IQ3_S),
                      :430 (IQ1_S), :438 (IQ1_M), :452 (IQ4_NL), :460 (IQ4_XS).
Architectural Impact: GwenLand's glproc/glcuda/glmetal/glvulkan can share a
                      single block-layout header. The static_assert is the
                      correctness guarantee that makes the on-disk format
                      portable across backends.
Correctness Impact:   None. The asserts verify a property that is already true;
                      they do not change behavior.
Optimization Type:    None (ABI guarantee).
GwenLand Target:      multiple (glproc, glcuda, glmetal, glvulkan)
Recommendation:       ADOPT. Copy ggml-common.h verbatim; do not invent a new
                      block-layout header.
Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX06-F02

```
Finding ID:           ARTX06-F02
Category:             ADOPT
Engine:               CPU
Component:            vec_dot_type indirection
Source File:          ggml/src/ggml-cpu/ggml-cpu.c, ggml/include/ggml-cpu.h
Function:             type_traits_cpu[] (vec_dot_type field), ggml_vec_dot_t
Lines:                ggml-cpu.c:214-415; ggml-cpu.h:114-122
Summary:              Every weight format declares a vec_dot_type — the
                      activation format its vec_dot kernel expects — letting
                      each weight family pick the activation layout that
                      minimizes per-block work.
Observation:          The type_traits_cpu entry for each weight type has a
                      vec_dot_type field. The matmul path reads it at
                      ggml-cpu.c:1272 and uses it to drive activation
                      conversion (ggml-cpu.c:1322-1355) and to select the
                      vec_dot kernel (ggml-cpu.c:1181). The activation
                      formats in use are Q8_0 (for Q1_0/Q2_0/Q4_0/Q5_0/MXFP4/
                      NVFP4/IQ4_NL/Q8_0), Q8_1 (for Q4_1/Q5_1/Q8_1), Q8_K
                      (for all K-quants, all IQ-quants, both TQ-quants), F16
                      (for F16), BF16 (for BF16), and F32 (for F32). The
                      split exists because each weight family's dot product
                      algebra benefits from a different precomputed
                      activation layout (see F11, F12).
Evidence:             ggml-cpu.c:230 (Q1_0→Q8_0), :236 (Q2_0→Q8_0), :242
                      (Q4_0→Q8_0), :252 (Q4_1→Q8_1), :262 (Q5_0→Q8_0), :268
                      (Q5_1→Q8_1), :274 (Q8_0→Q8_0), :283 (Q8_1→Q8_1), :289
                      (MXFP4→Q8_0), :295 (NVFP4→Q8_0), :301 (Q2_K→Q8_K), :307
                      (Q3_K→Q8_K), :313 (Q4_K→Q8_K), :323 (Q5_K→Q8_K), :329
                      (Q6_K→Q8_K), :339 (IQ2_XXS→Q8_K), :345 (IQ2_XS→Q8_K),
                      :352 (IQ3_XXS→Q8_K), :358 (IQ3_S→Q8_K), :364 (IQ2_S→
                      Q8_K), :370 (IQ1_S→Q8_K), :376 (IQ1_M→Q8_K), :382
                      (IQ4_NL→Q8_0), :388 (IQ4_XS→Q8_K), :397 (BF16→BF16),
                      :403 (TQ1_0→Q8_K), :409 (TQ2_0→Q8_K).
Architectural Impact: GwenLand's gl_type_traits should have a vec_dot_type
                      field. The matmul path should pre-convert src1 to
                      vec_dot_type once, before chunked execution.
Correctness Impact:   None. The indirection is a dispatch mechanism; the
                      arithmetic per format is unchanged.
Optimization Type:    Indirect dispatch + precomputed activation layout.
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. The vec_dot_type field is a one-line addition to
                      the traits table; the matmul-path integration is the
                      activation conversion loop already audited in ARTX01 §5.5.
Priority:             Critical
Difficulty:           S
Dependencies:         R1 (block layout ABI)
Confidence:           High
```

### Finding ARTX06-F03

```
Finding ID:           ARTX06-F03
Category:             QUANTIZATION
Engine:               CPU
Component:            from_float NULL contract for inference-only IQ formats
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             type_traits_cpu[]
Lines:                336-378, 348-365
Summary:              IQ2_XXS, IQ2_XS, IQ1_S, IQ1_M set from_float=NULL;
                      IQ3_XXS, IQ3_S, IQ2_S have from_float commented out.
                      These formats require offline importance sampling and
                      runtime init via ggml_quantize_init().
Observation:          The matmul path uses from_float only on activations
                      (always converted via vec_dot_type.from_float, never
                      weight-type.from_float). Setting weight-side
                      from_float=NULL therefore blocks only runtime weight
                      quantization, which is the intent for IQ formats that
                      need an importance matrix or a grid-lookup init. The
                      commented-out form for IQ3_XXS/IQ3_S/IQ2_S is functionally
                      equivalent but is invisible to grep and creates
                      inconsistency (W5).
Evidence:             ggml-cpu.c:337 (IQ2_XXS NULL), :343 (IQ2_XS NULL), :349
                      (IQ3_XXS commented), :356 (IQ3_S commented), :362 (IQ2_S
                      commented), :368 (IQ1_S NULL), :374 (IQ1_M NULL). The
                      init flow is at ggml.c:7873-7889 (ggml_quantize_init)
                      which calls iq2xs_init_impl / iq3xs_init_impl.
                      ggml_quantize_requires_imatrix at ggml.c:7905-7911
                      declares IQ2_XXS, IQ2_XS, IQ1_S as needing an imatrix.
Architectural Impact: GwenLand should adopt the from_float=NULL contract for
                      any format that cannot be runtime-quantized. The
                      contract is enforced by the matmul path's existing NULL
                      check (it simply never invokes weight-side from_float).
Correctness Impact:   None. The NULL is a deliberate "do not call" signal.
Optimization Type:    None (contract enforcement).
GwenLand Target:      glproc, GATE
Recommendation:       ADOPT. Use from_float=NULL uniformly; reject the
                      commented-out form (R9).
Priority:             High
Difficulty:           XS
Dependencies:         R2 (vec_dot_type indirection)
Confidence:           High
```

### Finding ARTX06-F04

```
Finding ID:           ARTX06-F04
Category:             ADOPT
Engine:               CPU
Component:            K-quant super-block structure
Source File:          ggml/src/ggml-common.h
Function:             block_q2_K, block_q3_K, block_q4_K, block_q5_K, block_q6_K
Lines:                89-90, 298-368
Summary:              All five K-quants use QK_K=256 with sub-blocks of 16
                      (Q2_K/Q3_K/Q6_K) or 32 (Q4_K/Q5_K) elements. Scales
                      are packed 4-bit (Q2_K), 6-bit (Q3_K/Q4_K/Q5_K), or
                      8-bit (Q6_K) into a fixed byte array sized by
                      K_SCALE_SIZE=12 or QK_K/16.
Observation:          The super-block structure trades a slightly higher
                      decode cost (bit unpacking) for much lower scale
                      overhead per weight. Q4_K at 4.5 bpw beats Q4_0 at
                      4.5 bpw on quality because the 8 sub-block scales
                      track weight variance within the super-block, whereas
                      Q4_0 has a single scale for 32 elements. The 6-bit
                      scale packing in K_SCALE_SIZE=12 bytes (96 bits) holds
                      either 16 6-bit scales (Q3_K) or 8 scales + 8 mins
                      (Q4_K, Q5_K) via a fixed bit shuffle.
Evidence:             ggml-common.h:89 (QK_K=256), :90 (K_SCALE_SIZE=12),
                      :298-309 (block_q2_K: scales[16], qs[64], d, dmin),
                      :315-321 (block_q3_K: hmask[32], qs[64], scales[12], d),
                      :327-338 (block_q4_K: d, dmin, scales[12], qs[128]),
                      :344-356 (block_q5_K: d, dmin, scales[12], qh[32], qs[128]),
                      :362-368 (block_q6_K: ql[128], qh[64], scales[16], d).
Architectural Impact: GwenLand should adopt QK_K=256 and K_SCALE_SIZE=12
                      verbatim. The bit shuffle for 6-bit unpacking is part
                      of the format ABI; any implementation must replicate
                      the kmask1/kmask2/kmask3 logic at quants.c:709-711.
Correctness Impact:   None. The layout is deterministic and bit-exact across
                      backends.
Optimization Type:    Bit-packed scale storage (saves 4-8 bytes per super-block
                      vs naive 8-bit storage).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. The packing is a measurable storage win and the
                      unpacking is a fixed shuffle.
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX06-F05

```
Finding ID:           ARTX06-F05
Category:             ADOPT
Engine:               CPU
Component:            Q3_K hmask + 6-bit scale unpacking
Source File:          ggml/src/ggml-cpu/quants.c
Function:             ggml_vec_dot_q3_K_q8_K_generic
Lines:                617-694
Summary:              Q3_K stores 3-bit weights as 2-bit low parts (qs) plus
                      1-bit high parts (hmask); the high bit subtracts 4 when
                      clear, producing a signed {-4..+3} value. The 6-bit
                      per-sub-block scales are unpacked from scales[12] via a
                      4-way 32-bit shuffle (kmask1=0x03030303, kmask2=0x0f0f0f0f).
Observation:          The unpacking at quants.c:675-680 takes 12 bytes of
                      scales + an intermediate uint32_t tmp = auxs[2] and
                      produces 16 6-bit signed scales stored in auxs[0..3]
                      as int8_t. The bit shuffle is non-obvious — it pulls
                      high bits from the third word and spreads them across
                      the low and high nibbles of the other three words.
                      Any implementation that gets the shuffle wrong produces
                      silently wrong weights. The hmask subtraction (a[l]
                      -= (hm[l] & m ? 0 : 4)) is also non-obvious; it makes
                      the recovered 3-bit value signed without an explicit
                      sign extension.
Evidence:             quants.c:651-672 (hmask loop), :675-680 (scale unpack),
                      :681-690 (scale-multiply + accumulate).
Architectural Impact: GwenLand must replicate the bit shuffle exactly. The
                      kmask constants are part of the format ABI.
Correctness Impact:   None if implemented correctly. A bug in the shuffle
                      produces silently wrong scales.
Optimization Type:    Bit-packed scale + bit-packed weight storage.
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Copy the unpacking verbatim from quants.c:675-680.
Priority:             Medium
Difficulty:           M
Dependencies:         R3 (K-quant 6-bit packing)
Confidence:           High
```

### Finding ARTX06-F06

```
Finding ID:           ARTX06-F06
Category:             ADOPT
Engine:               CPU
Component:            Q6_K simplified scale (no min, 8-bit signed)
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-cpu/quants.c
Function:             block_q6_K, ggml_vec_dot_q6_K_q8_K_generic
Lines:                ggml-common.h:362-368; quants.c:851-904
Summary:              Q6_K is the only K-quant with no min/zero-point. Its
                      8-bit signed scales[QK_K/16] directly multiply the
                      (q-32) recovered 6-bit signed weights. Simpler and
                      faster than Q4_K/Q5_K which pack both 6-bit scales
                      and 6-bit mins.
Observation:          Q6_K stores 6-bit weights as a 4-bit low part (ql) and
                      a 2-bit high part (qh), combined as
                      ((ql & 0xF) | ((qh >> N) & 3) << 4) - 32 to produce a
                      signed 6-bit value in {-32..+31}. The 8-bit signed
                      scale then multiplies this directly. No min offset
                      means no dmin, no mins[], no second reduction in the
                      vecdot. The vecdot at quants.c:851-904 is correspondingly
                      simpler than Q4_K (quants.c:696-769): one scale-multiply
                      per sub-block, no min subtraction.
Evidence:             ggml-common.h:362-368 (struct), quants.c:872-898 (vecdot
                      body — note absence of dmin/sumi subtraction vs Q4_K
                      at quants.c:763-765).
Architectural Impact: Q6_K is the highest-quality K-quant below Q8_K and is
                      the simplest K-quant vecdot. GwenLand should implement
                      it first as the reference K-quant kernel.
Correctness Impact:   None.
Optimization Type:    Simplified scale path (no min offset).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Q6_K is the cleanest K-quant to use as a
                      reference implementation.
Priority:             Medium
Difficulty:           S
Dependencies:         R3
Confidence:           High
```

### Finding ARTX06-F07

```
Finding ID:           ARTX06-F07
Category:             ADOPT
Engine:               CPU
Component:            I-quant codebook LUT grids
Source File:          ggml/src/ggml-common.h
Function:             iq2xxs_grid, iq2xs_grid, iq2s_grid, iq3xxs_grid,
                      iq3s_grid, iq1s_grid, ksigns_iq2xs, kmask_iq2xs
Lines:                509-575, 627-770, 1017-1133
Summary:              I-quants encode 8-element (or 4-element) groups of
                      weights as a single index into a precomputed codebook
                      grid. Six grids total ~50 KB of static const data and
                      encode ~10^6 possible weight patterns. The dequantize
                      path is a single indexed load; quantization cost is
                      borne entirely at quantization time.
Observation:          Each grid entry is a uint64_t (8 int8 values) or
                      uint32_t (4 int8 values) representing a signed byte
                      pattern. The vecdot looks up the grid entry, then
                      applies a per-group sign mask from ksigns_iq2xs[128]
                      (a 7-bit index → 8 sign bits). Per-32-element blocks
                      have a 4-bit LS (level scale) that multiplies the grid
                      output. The grids are constructed offline by importance
                      sampling — they encode the most-fitted 8-weight
                      patterns for each format's bit budget.
Evidence:             ggml-common.h:560 (iq2xxs_grid[256] uint64_t), :627
                      (iq2xs_grid[512]), :758 (iq2s_grid[1024]), :1017
                      (iq3xxs_grid[256] uint32_t), :1052 (iq3s_grid[512]),
                      :1135 (iq1s_grid[2048]), :509 (kmask_iq2xs[8] = {1,2,4,
                      8,16,32,64,128}), :513 (ksigns_iq2xs[128]). Usage at
                      quants.c:934 (iq2xxs_grid + aux8[l]), :973 (iq2xs_grid
                      + (q2[l] & 511)), :1026 (iq2s_grid + ...), :1077
                      (iq3xxs_grid + q3[2*l+0]), :1120 (iq3s_grid + ...),
                      :1176 (iq1s_grid + (qs[l] | ...)).
Architectural Impact: GwenLand can copy the grids verbatim. They are part of
                      the format ABI. The dequantize path is uniform across
                      all IQ formats: lookup grid → apply sign mask → multiply
                      by LS.
Correctness Impact:   None. The grids are static const; identical across
                      backends.
Optimization Type:    LUT-based weight reconstruction (replaces per-weight
                      arithmetic with one indexed load).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Copy the six grids + kmask + ksigns verbatim.
Priority:             High
Difficulty:           M
Dependencies:         R1
Confidence:           High
```

### Finding ARTX06-F08

```
Finding ID:           ARTX06-F08
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            IQ1_M nibble-spread super-block scale
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-cpu/quants.c
Function:             block_iq1_m, ggml_vec_dot_iq1_m_q8_K_generic
Lines:                ggml-common.h:432-438; quants.c:1193-1252
Summary:              block_iq1_m has no top-level d field; the 16-bit fp16
                      super-block scale is reconstructed at runtime from four
                      scale bytes' top nibbles via a 4-step shift-and-or.
                      Each scale byte also holds a 3-bit LS for one sub-block.
Observation:          The reconstruction at quants.c:1218 is:
                        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0)
                                  | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
                      Each sc[i] byte holds 4 bits of the super-block scale
                      (top nibble) plus 4 bits of LS data (bottom nibble
                      holds two 3-bit LS values, with overlap). This is the
                      densest scale packing in the format tree — IQ1_M at
                      1.75 bpw would be 1.8125 bpw if the scale were stored
                      as a separate ggml_half. But the cost is fragile bit
                      manipulation; a mismask anywhere produces nonsense
                      weights. The format has no compile-time protection
                      against reconstruction bugs.
Evidence:             ggml-common.h:433-438 (struct — note absence of ggml_half
                      d), quants.c:1216-1218 (reconstruction), :1239-1240 (LS
                      extraction from same sc[] bytes).
Architectural Impact: GwenLand should adopt IQ1_M only with a comprehensive
                      round-trip test. The 4-step shift-and-or is correct
                      but is a maintenance hazard. Alternative: introduce an
                      IQ1_M_V2 with a separate ggml_half d (costs 0.0625 bpw).
Correctness Impact:   None if implemented correctly. The reconstruction is
                      deterministic; a bug would be caught by a round-trip
                      test.
Optimization Type:    Bit-packed scale storage (saves 2 bytes per super-block
                      vs separate ggml_half).
GwenLand Target:      glproc
Recommendation:       MONITOR. Adopt IQ1_M for compatibility; add a unit test
                      that round-trips every (scale nibble, LS nibble)
                      combination. Consider an IQ1_M_V2 with separate d if
                      bit-packing proves error-prone.
Priority:             Medium
Difficulty:           S
Dependencies:         R1, R4
Confidence:           High
```

### Finding ARTX06-F09

```
Finding ID:           ARTX06-F09
Category:             ADOPT
Engine:               CPU
Component:            MXFP4 (E8M0) vs NVFP4 (UE4M3) microscaling scales
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-impl.h,
                      ggml/src/ggml-cpu/quants.c
Function:             block_mxfp4, block_nvfp4, ggml_e8m0_to_fp32_half,
                      ggml_ue4m3_to_fp32, ggml_vec_dot_mxfp4_q8_0_generic,
                      ggml_vec_dot_nvfp4_q8_0_generic
Lines:                ggml-common.h:214-227; ggml-impl.h:477-540;
                      quants.c:298-363
Summary:              MXFP4 uses a single E8M0 byte scale per 32-element block
                      (8-bit biased exponent, value = 2^(e-127), no mantissa).
                      NVFP4 uses four UE4M3 bytes per 64-element block (one
                      per 16-element sub-block, 4 exp + 3 mantissa bits, value
                      = (1 + m/8) * 2^(e-7)). Both share kvalues_mxfp4 = E2M1
                      values ×2.
Observation:          The two formats address different scale-granularity
                      tradeoffs. MXFP4's E8M0 is a pure power-of-two scale —
                      cheap to convert (one shift) but no mantissa precision.
                      NVFP4's UE4M3 has 3 mantissa bits per sub-block scale,
                      giving finer control at the cost of 4× more scale bytes
                      per block. Both fold the E2M1 ×2 convention into the
                      scale converter (* 0.5f at ggml-impl.h:514, 488). The
                      NVFP4 vecdot (quants.c:330-363) loops over 4 sub-blocks,
                      each mapping onto half of a Q8_0 activation block;
                      MXFP4's vecdot (quants.c:298-327) is a single 32-element
                      loop. NVFP4 is the more complex implementation.
Evidence:             ggml-common.h:214-219 (block_mxfp4: 1 E8M0 byte +
                      qs[16]), :221-227 (block_nvfp4: d[4] UE4M3 bytes +
                      qs[32]), :1126-1129 (kvalues_mxfp4 = {0,1,2,3,4,6,8,12,
                      0,-1,-2,-3,-4,-6,-8,-12}). ggml-impl.h:477-495
                      (ggml_e8m0_to_fp32_half with *0.5f), :502-515
                      (ggml_ue4m3_to_fp32 with *0.5f). quants.c:316 (MXFP4
                      uses GGML_E8M0_TO_FP32_HALF), :347 (NVFP4 uses
                      ggml_ue4m3_to_fp32).
Architectural Impact: GwenLand should adopt both formats with their respective
                      scale converters. The *0.5f convention must be preserved
                      to match the kvalues_mxfp4 ×2 LUT. NVFP4's per-sub-block
                      scale loop is more complex but enables higher quality at
                      the same 4-bit weight width.
Correctness Impact:   None. The scale converters are deterministic.
Optimization Type:    Microscaling per-block / per-sub-block scale.
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. Both formats are OCP MX-compliant; sharing the
                      format ABI with the OCP spec maximizes interop.
Priority:             High
Difficulty:           M
Dependencies:         R1, R6
Confidence:           High
```

### Finding ARTX06-F10

```
Finding ID:           ARTX06-F10
Category:             ADOPT
Engine:               CPU
Component:            TQ1_0 mixed-radix ternary packing
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-cpu/quants.c
Function:             block_tq1_0, ggml_vec_dot_tq1_0_q8_K_generic
Lines:                ggml-common.h:275-281; quants.c:481-531
Summary:              TQ1_0 packs 5 ternary values per byte (3^5=243 ≤ 256)
                      for most weights, with 4 values per byte (3^4=81 ≤ 256)
                      for the remainder. Achieves 1.6875 bpw, closer to the
                      1.585 information-theoretic minimum than TQ2_0's 2.0625
                      bpw. The vecdot uses a pow3[6] lookup and is the slowest
                      in the tree.
Observation:          The block layout splits 256 weights into 48 bytes of 5-
                      per-byte packing (240 weights) + 4 bytes of 4-per-byte
                      packing (16 weights) = 64 bytes of weight data, plus
                      fp16 d = 54 bytes total. The vecdot at quants.c:481-531
                      is a 5-deep nested scalar loop that multiplies each
                      weight by a power of 3 (pow3[l]) to recover the ternary
                      value, then by 3/256 to extract the high bit. This is
                      the most arithmetic-intensive vecdot in the tree.
                      Per-ISA overrides exist for x86 (arch/x86/quants.c:1376)
                      and ARM (arch/arm/quants.c:1397); other ISAs fall
                      through to the generic scalar path.
Evidence:             ggml-common.h:275-281 (struct with comment "1.6875 bpw"),
                      quants.c:493 (pow3 table), :497-528 (nested unpacking
                      loops). Per-ISA overrides: arch/x86/quants.c:1376-1505,
                      arch/arm/quants.c:1397-1570.
Architectural Impact: GwenLand should adopt TQ1_0 for storage density. For
                      ISAs without a hand-tuned kernel, the generic scalar
                      vecdot is the bottleneck; a SIMD vecdot (at least for
                      x86 AVX2 and ARM NEON) is recommended.
Correctness Impact:   None.
Optimization Type:    Mixed-radix bit packing (5 ternary values per byte).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT the format. Provide SIMD vecdots for at least x86
                      and ARM (R7).
Priority:             Medium
Difficulty:           L
Dependencies:         R1, R7
Confidence:           High
```

### Finding ARTX06-F11

```
Finding ID:           ARTX06-F11
Category:             ADOPT
Engine:               CPU
Component:            Q8_1 s-field precomputation (min-offset folding)
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-cpu/quants.c,
                      ggml/src/ggml-quants.c
Function:             block_q8_1, ggml_vec_dot_q4_1_q8_1_generic,
                      quantize_row_q8_1_ref
Lines:                ggml-common.h:258-269; quants.c:262-296;
                      ggml-quants.c:302-348
Summary:              block_q8_1 stores d (delta) AND s = d * sum(qs[i]) as
                      two precomputed fp16 values. This lets the Q4_1/Q5_1
                      vecdot fold the min-offset term into a single scalar
                      multiply (m_x * s_y) instead of computing a second
                      per-block int32 reduction.
Observation:          The Q4_1 dot product algebra is:
                        Σ weight[i] * activation[i]
                      = Σ (nibble_x * d_x + m_x) * (q8_y * d_y)
                      = d_x * d_y * Σ nibble_x * q8_y + m_x * d_y * Σ q8_y
                      The second sum Σ q8_y is the same for every Q4_1 block
                      that uses this Q8_1 activation block. By precomputing
                      s_y = d_y * Σ q8_y at quantization time and storing it
                      in the activation block, the vecdot avoids a second
                      int32 reduction per block. The vecdot at quants.c:262-
                      296 implements exactly this: it computes sumi = Σ
                      nibble*q8 once, then sums float results as
                      d_x*d_y*sumi + m_x*s_y. The Q5_1 vecdot (quants.c:408-
                      449) uses the same trick.
Evidence:              ggml-common.h:258-269 (block_q8_1: union {struct{d,s;}
                      ds;}), quants.c:292 (sumf += ... + m_x * s_y),
                      ggml-quants.c:326-348 (quantize_row_q8_1_ref computes
                      s = d * sum(qs)).
Architectural Impact: GwenLand should adopt the s-field precomputation for
                      any weight format with a min offset (Q4_1, Q5_1, and
                      any future format with affine quantization).
Correctness Impact:   None. s is deterministic; the precomputation is exact.
Optimization Type:    Precomputed activation field (eliminates per-block
                      reduction).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. The s-field is a 2-byte-per-block cost that
                      eliminates a 32-deep int32 reduction in the vecdot.
Priority:             Medium
Difficulty:           S
Dependencies:         R2
Confidence:           High
```

### Finding ARTX06-F12

```
Finding ID:           ARTX06-F12
Category:             ADOPT
Engine:               CPU
Component:            Q8_K bsums precomputation (per-sub-block sums)
Source File:          ggml/src/ggml-common.h, ggml/src/ggml-cpu/quants.c,
                      ggml/src/ggml-quants.c
Function:             block_q8_K, ggml_vec_dot_q4_K_q8_K_generic,
                      quantize_row_q8_K_ref
Lines:                ggml-common.h:371-376; quants.c:565-769;
                      ggml-quants.c:2768-2805
Summary:              block_q8_K stores qs[256] (int8) AND bsums[16] (int16)
                      where bsums[j] = sum(qs[j*16..(j+1)*16]). This lets
                      every K-quant vecdot fold the min-offset term into a
                      single sumi * mins[j/2] multiplication instead of
                      computing a per-sub-block int32 reduction.
Observation:          The K-quant dot product algebra for Q4_K is:
                        Σ weight[i] * activation[i]
                      = Σ_j scale[j] * Σ_{i in j} nibble*q8
                        - Σ_j min[j] * Σ_{i in j} q8
                      The second inner sum is per-sub-block (16 elements)
                      and is the same for every Q4_K block that uses this
                      Q8_K activation block. By precomputing bsums[j] at
                      quantization time, the vecdot replaces 8 per-sub-block
                      int32 reductions with 8 int16 reads. The Q4_K vecdot
                      at quants.c:744 implements this: sumi += y[i].bsums[j]
                      * mins[j/2]. Q2_K, Q5_K use the same pattern.
                      (Q3_K and Q6_K have no min and do not use bsums, but
                      the Q8_K activation format always carries them for
                      uniformity.)
Evidence:              ggml-common.h:371-376 (block_q8_K: float d, qs[256],
                      bsums[16]), quants.c:587 (Q2_K uses bsums), :744 (Q4_K
                      uses bsums), :824 (Q5_K uses bsums). Q3_K vecdot at
                      quants.c:617-694 does NOT use bsums (no min). Q6_K
                      vecdot at quants.c:851-904 does NOT use bsums. Q8_K
                      quantizer at ggml-quants.c:2795-2801 computes bsums.
Architectural Impact: GwenLand should adopt bsums precomputation for any
                      weight format with per-sub-block min offsets (Q2_K,
                      Q4_K, Q5_K). The 32-byte-per-block cost (16 int16
                      values) is amortized across all weight blocks that
                      use the activation.
Correctness Impact:   None. bsums is deterministic; the int16 range is
                      sufficient (sum of 16 int8 values fits in int16).
Optimization Type:    Precomputed activation field (eliminates per-sub-block
                      reduction).
GwenLand Target:      glproc, glcuda, glmetal, glvulkan
Recommendation:       ADOPT. The bsums precomputation is the key reason
                      K-quants use a different activation format (Q8_K) than
                      simple quants (Q8_0/Q8_1).
Priority:             High
Difficulty:           S
Dependencies:         R2, R3
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the Q8_K `bsums[16]` int16 overflow boundary (sum of
  16 int8 values can be ±2048, fits in int16) is ever exceeded in
  practice. Static analysis confirms the per-sub-block sum is safe;
  the cumulative sum across sub-blocks is never materialized in
  `bsums`. Requires runtime test with adversarial inputs to confirm.

* **U2**. Whether the `IQ1S_DELTA = 0.125` / `IQ1M_DELTA = 0.125`
  constants are optimal. They are hard-coded in `ggml-common.h:1132-
  1133` with no documentation of how they were derived. Requires
  offline quality sweep.

* **U3**. Whether the IQ3_XXS / IQ3_S / IQ2_S `from_float` entries
  (commented out at `ggml-cpu.c:349-365`) were disabled because the
  quantizer is broken, because it requires init, or for some other
  reason. The comment says "require initialization in
  ggml_quantize_init" but IQ3_XXS / IQ3_S / IQ2_S all have non-NULL
  `from_float_ref` in the base `type_traits[]` (`ggml.c:829, 837,
  845`) — implying the quantizer exists and works after init. The
  commented-out form may be a leftover from a refactor. Requires git
  history review.

* **U4**. Whether the MXFP4 / NVFP4 `* 0.5f` convention in
  `ggml-impl.h:514, 488` is the right design. The convention couples
  the scale converter to the LUT layout; changing one without the
  other produces silently 2× wrong weights. A cleaner design would
  store E2M1 values at their true magnitude in the LUT and remove
  the `* 0.5f`. Requires upstream discussion.

* **U5**. Whether TQ1_0's 5-deep nested scalar vecdot at
  `quants.c:481-531` is ever the bottleneck in a real BitNet b1.58
  inference workload. The x86 and ARM overrides exist; other ISAs
  fall through to scalar. Requires profiling on LoongArch / PowerPC
  / s390 / WASM / RISC-V hardware.

* **U6**. Whether the 6-bit scale unpacking shuffle for Q3_K
  (`quants.c:675-680`) and Q4_K/Q5_K (`quants.c:736-741`) can be
  replaced with a simpler layout without breaking the on-disk ABI.
  The current shuffle is a fixed property of the format; any
  alternative layout would be a new format. Requires design study.

* **U7**. Whether the `iq2_data[gindex].map` / `neighbours` runtime
  tables (built by `ggml_quantize_init`) can be shared across
  multiple IQ formats that use the same grid (e.g., IQ2_XS and
  IQ2_S both use `iq2xs_grid[512]`). Currently each format builds
  its own copy. Requires inspection of `iq2_data_index()` at
  `ggml-quants.c:3108-3113`.

* **U8**. Whether the Q1_0 / Q2_0 activation ratio (4× / 2×) is
  documented anywhere outside the source. The matmul path enforces
  it implicitly via `ne10 % blck_size(src0->type) == 0`
  (`ggml-cpu.c:1282`), but a caller that bypasses the matmul path
  could miss the constraint. Requires API doc review.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/include/ggml.h`                               | `enum ggml_type`                               | 389–433       |
| R02       | `ggml/include/ggml.h`                               | `struct ggml_type_traits` (base)               | 2884–2892     |
| R03       | `ggml/include/ggml-cpu.h`                           | `struct ggml_type_traits_cpu`, `ggml_vec_dot_t`| 114–122       |
| R04       | `ggml/src/ggml-common.h`                            | `QK_K = 256`, `K_SCALE_SIZE = 12`              | 89–90         |
| R05       | `ggml/src/ggml-common.h`                            | `block_q1_0` … `block_q8_1`                    | 180–269       |
| R06       | `ggml/src/ggml-common.h`                            | `block_tq1_0`, `block_tq2_0`                   | 275–288       |
| R07       | `ggml/src/ggml-common.h`                            | `block_q2_K` … `block_q6_K`, `block_q8_K`     | 298–376       |
| R08       | `ggml/src/ggml-common.h`                            | `block_iq2_xxs` … `block_iq4_xs`               | 381–460       |
| R09       | `ggml/src/ggml-common.h`                            | `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`       | 560, 627, 758 |
| R10       | `ggml/src/ggml-common.h`                            | `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid`        | 1017, 1052, 1135 |
| R11       | `ggml/src/ggml-common.h`                            | `kmask_iq2xs`, `ksigns_iq2xs`, `ksigns64`      | 509, 513, 524 |
| R12       | `ggml/src/ggml-common.h`                            | `kvalues_iq4nl`, `kvalues_mxfp4` (= `kvalues_fp4`) | 1120, 1126 |
| R13       | `ggml/src/ggml-common.h`                            | `IQ1S_DELTA`, `IQ1M_DELTA`                     | 1132–1133     |
| R14       | `ggml/src/ggml-cpu/quants.h`                        | public CPU quant API                           | 15–106        |
| R15       | `ggml/src/ggml-cpu/quants.c`                        | `quantize_row_q4_0` … `quantize_row_q8_K_generic` | 25–123     |
| R16       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q4_0_q8_0_generic`               | 225–259       |
| R17       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q4_1_q8_1_generic`               | 262–296       |
| R18       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_mxfp4_q8_0_generic`              | 298–327       |
| R19       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_nvfp4_q8_0_generic`              | 330–363       |
| R20       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_tq1_0_q8_K_generic`              | 481–531       |
| R21       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_tq2_0_q8_K_generic`              | 533–563       |
| R22       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q2_K_q8_K_generic`               | 565–615       |
| R23       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q3_K_q8_K_generic`               | 617–694       |
| R24       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q4_K_q8_K_generic`               | 696–769       |
| R25       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q5_K_q8_K_generic`               | 771–849       |
| R26       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_q6_K_q8_K_generic`               | 851–904       |
| R27       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq2_xxs_q8_K_generic`            | 906–946       |
| R28       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq2_xs_q8_K_generic`             | 948–996       |
| R29       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq2_s_q8_K_generic`              | 998–1048      |
| R30       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq3_xxs_q8_K_generic`            | 1050–1092     |
| R31       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq3_s_q8_K_generic`              | 1094–1148     |
| R32       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq1_s_q8_K_generic`              | 1150–1191     |
| R33       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq1_m_q8_K_generic`              | 1193–1252     |
| R34       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq4_nl_q8_0_generic`             | 1254–1281     |
| R35       | `ggml/src/ggml-cpu/quants.c`                        | `ggml_vec_dot_iq4_xs_q8_K_generic`             | 1283–1327     |
| R36       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `type_traits_cpu[]`                            | 214–415       |
| R37       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_get_type_traits_cpu`                     | 417–419       |
| R38       | `ggml/src/ggml.c`                                   | `type_traits[]` (base)                         | 631–945       |
| R39       | `ggml/src/ggml.c`                                   | `ggml_get_type_traits`                         | 947–950       |
| R40       | `ggml/src/ggml.c`                                   | `ggml_blck_size`, `ggml_type_size`, `ggml_row_size` | 1326–1335 |
| R41       | `ggml/src/ggml.c`                                   | `ggml_quantize_init`                           | 7873–7889     |
| R42       | `ggml/src/ggml.c`                                   | `ggml_quantize_free`                           | 7891–7903     |
| R43       | `ggml/src/ggml.c`                                   | `ggml_quantize_requires_imatrix`               | 7905–7911     |
| R44       | `ggml/src/ggml.c`                                   | `ggml_quantize_chunk`                          | 7913–7988     |
| R45       | `ggml/src/ggml-quants.c`                            | `quantize_row_mxfp4_ref`                       | 350–382       |
| R46       | `ggml/src/ggml-quants.c`                            | `quantize_row_nvfp4_ref`                       | 384–417       |
| R47       | `ggml/src/ggml-quants.c`                            | `quantize_row_q8_K_ref` (bsums precomputation) | 2768–2805     |
| R48       | `ggml/src/ggml-quants.c`                            | `iq2xs_init_impl`, `iq3xs_init_impl`           | 3108–3260     |
| R49       | `ggml/src/ggml-quants.c`                            | `quantize_iq4_xs`                              | 5115–5133     |
| R50       | `ggml/src/ggml-impl.h`                              | `ggml_e8m0_to_fp32_half`                       | 477–495       |
| R51       | `ggml/src/ggml-impl.h`                              | `ggml_ue4m3_to_fp32`, `ggml_fp32_to_ue4m3`     | 502–540       |
| R52       | `ggml/src/ggml-cpu/arch-fallback.h`                 | `_generic` aliases                             | 21–22, 119–120, 162–163, 252–253, 301–302 |
| R53       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_forward_mul_mat` (activation conversion) | 1272–1355 |
