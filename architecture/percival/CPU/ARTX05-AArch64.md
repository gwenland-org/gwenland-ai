# ARTX05 — AArch64 Advanced SIMD (DOTPROD / I8MM / SVE / SVE2 / SME / SME2) Quantized Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (AArch64-advanced kernel layer), `GATE` (kernel-selection contract)

---

## 1. Executive Summary

The AArch64 advanced-SIMD quantized kernels in llama.cpp live in
`arch/arm/quants.c` (4319 lines), `arch/arm/repack.cpp` (5156 lines),
`arch/arm/cpu-feats.cpp` (103 lines), `vec.cpp` / `vec.h` (SVE F32/F16
dot), `ggml-cpu-impl.h:307-321` (the `ggml_vdotq_s32` DOTPROD macro
covered in ARTX04-F02), and `kleidiai/` (2911 lines of wrapper around
the third-party KleidiAI library). The "advanced" tiers are everything
above baseline NEON: DOTPROD (`__ARM_FEATURE_DOTPROD`), I8MM
(`__ARM_FEATURE_MATMUL_INT8`), SVE (`__ARM_FEATURE_SVE`), SVE2
(`__ARM_FEATURE_SVE2`), and SME/SME2 (`__ARM_FEATURE_SME` /
`__ARM_FEATURE_SME2`). Baseline NEON is audited in ARTX04 and is *not*
re-audited here except to the extent it forms the fallback rung of an
advanced kernel's preprocessor ladder.

The single most important observation is that **the per-block vecdot
ladder is a four-tier compile-time dispatch**: I8MM 2-row → SVE →
DOTPROD (selected kernels only) → NEON baseline. The I8MM tier is the
only tier that accepts `nrc == 2` (two weight rows × two activation
cols per call) and is gated by `__ARM_FEATURE_MATMUL_INT8`. It produces
four independent float32 results per block via a single
`vmmlaq_s32` (NEON) or `svmmla_s32` (SVE) chain on lane-interleaved
`vzip1q_s64`/`vzip2q_s64` data. The SVE tier (without I8MM) falls back
to `svdot_s32` (1×4 dot of 16 int8s into 4 int32s) and is a true
per-block VLA *in form* — but every kernel wraps its body in a
`switch (vector_length)` that asserts `false` on VL ∉ {128, 256, 512}.
The DOTPROD tier (`vdotq_s32`) appears only in NVFP4 (line 840) and a
handful of TQ1_0/TQ2_0 paths; it does **not** appear as a multi-row
tile for any standard Q4/Q8/K-quant vecdot.

The second key observation is that **SME / SME2 are entirely delegated
to KleidiAI**. grep across the whole `ggml-cpu/` tree finds
`__ARM_FEATURE_SME` in exactly one non-KleidiAI file:
`ggml-cpu.c:3803` (the `ggml_cpu_has_sme()` detection function). All
actual SME/SME2 matmul execution lives inside the bundled third-party
KleidiAI library, dispatched through a wrapper at `kleidiai/kernels.cpp`
that selects between SME2, SME, I8MM, and DOTPROD variants of the
pre-packaged RHS layout. There is no upstream-written SME outer-product
kernel.

The third observation is that **the 22 batched GEMV/GEMM entry points
in `repack.cpp` are hand-written inline assembly using `.inst 0x…`
encodings** for `sdot`/`smmla`/`sdot.lane`. These bypass the compiler's
intrinsic-availability window (useful for older clangs) and let the
author hand-pipeline 4×4 / 4×8 / 8×8 tiles, but they introduce a hard
maintenance cost. The SVE GEMV/GEMM kernels are additionally gated on
`ggml_cpu_get_sve_cnt() == QK8_0` (i.e., == 16 bytes / 128-bit SVE);
non-128-bit SVE hardware falls through to the `_generic` scalar path.

For GwenLand, the decisions worth **ADOPT**ing are the I8MM 2-row×2-col
`vzip`+`vmmlaq_s32` tile (F01), the one-shot `svcntb()`-cached SVE VL
discovery (F03), the KleidiAI `extra_buffer_type` plugin mechanism with
its weight-header + dual-slot layout (F07, F12), and the SVE2 widened
F16 FMA helper (F05). The decisions worth **REJECT**ing are the
`assert(false)` on unsupported SVE VLs (F02) and the absence of an
SVE+I8MM 2-row path for Q4_0/Q4_1/Q8_0 (F08). The decisions worth
**MONITOR**ing are the dead `has_sme2` field on Linux (F11) and the
heavy reliance on inline assembly (F10).

---

## 2. Purpose

Provide the AArch64 advanced-SIMD quantized-kernel layer for `glproc`.
This layer is responsible for:

* `vec_dot` kernels for every supported quant format on hardware with
  DOTPROD, I8MM, SVE, SVE2, or SME features (compiled-in via
  `__ARM_FEATURE_*` macros). The I8MM tier additionally accepts
  `nrc == 2` for 2-row × 2-col tile consumption; the type-traits table
  at `ggml-cpu.c:214-415` sets `nrows = 2` for Q4_0/Q4_1/Q8_0/Q4_K/Q6_K
  only when `__ARM_FEATURE_MATMUL_INT8` is defined (see ARTX01-F03).
* 22 batched GEMV / GEMM entry points in `repack.cpp` — hand-written
  inline-assembly tiles for Q4_0/Q4_K/Q5_K/Q6_K/IQ4_NL/MXFP4/Q8_0 across
  4×4, 4×8, 8×8 column tiles, gated on DOTPROD/I8MM/SVE.
* KleidiAI integration (`kleidiai/kleidiai.cpp`, `kleidiai/kernels.cpp`)
  that overrides `compute_forward(MUL_MAT)` and `compute_forward(GET_ROWS)`
  for Q4_0, Q8_0, F32, F16 RHS tensors when the pre-packed
  `kleidiai_weight_header` is present and the kernel chain returns a
  non-empty slot set.

It is **not** responsible for: baseline NEON vecdot (ARTX04), x86 or
RISC-V vecdot, the type-traits table itself (ARTX01), graph scheduling
(ARTX01), elementwise ops (ARTX06), or AMX (separate ARTX).

---

## 3. Source Files

| File                                          | Lines  | Role                                                                                  |
| --------------------------------------------- | ------ | ------------------------------------------------------------------------------------- |
| `ggml/src/ggml-cpu/arch/arm/quants.c`         | 4319   | Per-block `vec_dot` kernels; 4-tier MATMUL_INT8 / SVE / DOTPROD / NEON ladder         |
| `ggml/src/ggml-cpu/arch/arm/repack.cpp`       | 5156   | 22 inline-asm batched GEMV/GEMM (4×4, 4×8, 8×8) — gated on DOTPROD/I8MM/SVE           |
| `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`    | 103    | `ggml_backend_cpu_aarch64_score` — multi-binary score, HWCAP2 probe; aarch64-only     |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h`           | 307-321| `ggml_vdotq_s32` DOTPROD fallback macro (covered in ARTX04-F02; reused by advanced)   |
| `ggml/src/ggml-cpu/vec.h`                     | 17-44  | `ggml_sve_f16_fma_widened` SVE2 vs SVE1 widened F16→F32 FMA helper                    |
| `ggml/src/ggml-cpu/vec.h`                     | 142-317| `ggml_vec_dot_f16_unroll` SVE path (uses the widened helper)                          |
| `ggml/src/ggml-cpu/vec.cpp`                   | 11-110 | `ggml_vec_dot_f32` SVE path — 8-accumulator VLA dot with `svwhilelt_b32` tail         |
| `ggml/src/ggml-cpu/ggml-cpu.c`                | 728-737| `ggml_init_arm_arch_features` caches `sve_cnt = svcntb()` at init                     |
| `ggml/src/ggml-cpu/ggml-cpu.c`                | 3794-3816 | `ggml_cpu_get_sve_cnt`, `ggml_cpu_has_sme`, `ggml_cpu_has_sme2` (detection only)   |
| `ggml/src/ggml-cpu/kleidiai/kleidiai.h`       | 17     | Public `ggml_backend_cpu_kleidiai_buffer_type()` entry                                |
| `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`     | 1783   | Wrapper: SMCU detection, weight header, slot chain, `compute_forward`, `repack`       |
| `ggml/src/ggml-cpu/kleidiai/kernels.cpp`      | 1131   | Static kernel tables: SME2/SME/I8MM/DOTPROD variants for Q4_0/Q8_0/F32/F16 RHS        |
| `ggml/src/ggml-cpu/kleidiai/kernels.h`        | 100    | Structs: `kernel_info`, `lhs_packing_info`, `rhs_packing_info`, `ggml_kleidiai_kernels`|

> The third-party KleidiAI library itself (the `kai_*` functions
> referenced from `kernels.cpp`) is bundled out-of-tree. This audit
> covers the wrapper and the kernel-selection tables, not the
> `kai_run_matmul_*` library bodies.

---

## 4. Architecture Overview

```
              ┌─────────────────────────────────────────────────────────────────┐
              │  type_traits_cpu[type].vec_dot  (ggml-cpu.c:214-415)            │
              │   nrows == 2  for Q4_0/Q4_1/Q8_0/Q4_K/Q6_K                      │
              │                when __ARM_FEATURE_MATMUL_INT8 is defined        │
              │   nrows == 1  otherwise                                         │
              └─────────────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼─────────────────────────────┐
                ▼                                                 ▼
   ┌──────────────────────────────────┐         ┌──────────────────────────────────┐
   │ arch/arm/quants.c  (this audit)  │         │ kleidiai/kleidiai.cpp            │
   │ ───────────────────────          │         │ ──────────────────────            │
   │ 4-tier #if ladder per vecdot:    │         │ extra_buffer_type                │
   │                                  │         │   .supports_op(MUL_MAT|GET_ROWS) │
   │  1. __ARM_FEATURE_MATMUL_INT8    │         │   for Q4_0/Q8_0/F32/F16 RHS      │
   │     if (nrc==2) { vmmlaq_s32 ... │         │   ├─ requires pre-packed KLAI    │
   │       vst1_f32(s); vst1_f32(s+bs)│         │   │  weight header                 │
   │       return; }                  │         │   ├─ slot[0] = primary (SME/I8MM)│
   │                                  │         │   ├─ slot[1] = fallback (DOTPROD)│
   │  2. __ARM_FEATURE_SVE            │         │   └─ if nrc>1: hybrid threads    │
   │     switch(vector_length) {      │         │      split by sme_thread_cap      │
   │       case 128/256/512: svdot_s32│         │                                   │
   │       default: assert(false)     │         │ compute_forward(MUL_MAT)         │
   │     }                            │         │   ├─ pack LHS (parallel by mr)   │
   │                                  │         │   ├─ ggml_barrier                │
   │  3. __ARM_FEATURE_DOTPROD (some) │         │   ├─ chunk loop with atomic      │
   │     vdotq_s32                    │         │   │  counter (like ARTX01-F06)  │
   │                                  │         │   └─ run_kernel_ex(...)          │
   │  4. __ARM_NEON (ARTX04)          │         │                                   │
   │     ggml_vdotq_s32 3-op emulation│         │ compute_forward(GET_ROWS)        │
   └──────────────────────────────────┘         │   rhs_info.to_float(...)         │
                ▲                               └──────────────────────────────────┘
                │
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ arch/arm/repack.cpp (this audit)                                        │
   │ ────────────────────────                                                │
   │ 22 batched GEMV/GEMM kernels, hand-written inline assembly:             │
   │  - 4x4:  Q4_0 / IQ4_NL / MXFP4 / Q8_0   (gated DOTPROD)                 │
   │  - 4x8:  Q4_0 / Q8_0                    (gated DOTPROD)                 │
   │  - 8x4:  Q4_K / Q5_K / Q6_K             (gated DOTPROD)                 │
   │  - 8x8:  Q4_0 / Q4_K / Q5_K / Q6_K      (gated SVE && cnt==16)          │
   │   uses .inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]             │
   │   uses .inst 0x451f9872  // smmla z18.s, z3.b, z31.b (SVE I8MM)         │
   │   if gate not taken → falls through to _generic                         │
   │                                                                         │
   │ decode_q_Kx8_6bit_scales (line 29) — shared helper, gated               │
   │   (I8MM || DOTPROD) for K-quant scale/min decode                        │
   └─────────────────────────────────────────────────────────────────────────┘
                ▲
                │
   ┌─────────────────────────────────────────────────────────────────────────┐
   │ arch/arm/cpu-feats.cpp (this audit)                                     │
   │  ggml_backend_cpu_aarch64_score:                                        │
   │   score = 1;                                                            │
   │   if (DOTPROD compiled && !has_dotprod) return 0; score += 1<<1;        │
   │   if (FP16_VA  compiled && !has_fp16_va) return 0; score += 1<<2;       │
   │   if (SVE      compiled && !has_sve)     return 0; score += 1<<3;       │
   │   if (I8MM     compiled && !has_i8mm)    return 0; score += 1<<4;       │
   │   if (SVE2     compiled && !has_sve2)    return 0; score += 1<<5;       │
   │   if (SME      compiled && !has_sme)     return 0; score += 1<<6;       │
   │   return score;   // no GGML_USE_SME2 macro checked (F11)               │
   │   // has_sme2 field populated (Apple only) but never consulted          │
   └─────────────────────────────────────────────────────────────────────────┘
```

Key design points:

* **Compile-time tier selection inside one `.so`**. The 4-tier ladder
  is resolved at compile time per-kernel via `#if`. The runtime
  decision happens once at `.so` load time in `cpu-feats.cpp`'s score
  function — same multi-binary model as x86 (ARTX02 §4) and baseline
  ARM (ARTX04 §4). There is no per-call runtime dispatch within a
  kernel.
* **I8MM is the only `nrc == 2` tier**. DOTPROD, SVE (alone), and
  baseline all assert `nrc == 1`. The 2-row consumption is gated on the
  type-traits table's `nrows` field, which itself is gated on
  `__ARM_FEATURE_MATMUL_INT8`.
* **DOTPROD and I8MM are distinct paths, not nested**. DOTPROD is the
  v8.2a `vdotq_s32` (16 int8 × 16 int8 → 4 int32 lanes, 4 dot products
  per instruction). I8MM is the v8.6a `vmmlaq_s32` (8 int8 × 8 int8 →
  4 int32 matrix-multiply, 32 dot products per instruction, 8× more
  arithmetic per instruction than DOTPROD). Hardware that has DOTPROD
  but not I8MM (e.g., Cortex-A78, Neoverse N1/V1 with DOTPROD enabled)
  gets only the 1-row DOTPROD path; hardware with I8MM (Cortex-X2,
  Neoverse V2, every Apple M-series, every v9 core) gets the 2-row
  I8MM path.
* **SME is delegated to KleidiAI, not written upstream**. The only
  `__ARM_FEATURE_SME` symbol outside `kleidiai/` is the
  `ggml_cpu_has_sme()` detection function (`ggml-cpu.c:3803`). All
  actual SME/SME2 outer-product matmul execution lives inside the
  bundled KleidiAI library.
* **KleidiAI uses an `extra_buffer_type` plugin**. It overrides
  `supports_op(MUL_MAT | GET_ROWS)` and `compute_forward(...)` for
  Q4_0/Q8_0/F32/F16 RHS tensors when a pre-packed `kleidiai_weight_header`
  is present. This is the same plugin architecture as AMX (ARTX01-F04).

---

## 5. Execution Flow

### 5.1 Per-block vecdot (the `quants.c` path)

1. `ggml_compute_forward_mul_mat_one_chunk` (`ggml-cpu.c:1164`) walks
   tiles `(iir0, iir1)` and for each weight row calls
   `vec_dot(qk, &s, sizeof(float), src0_row, nb01, src1_row, nb11, nrows)`.
2. `vec_dot` is a function pointer from `type_traits_cpu[type].vec_dot`
   (`ggml-cpu.c:1181-1182`). For Q4_0/Q4_1/Q8_0/Q4_K/Q6_K on an I8MM
   build, `nrows = 2`; otherwise `nrows = 1`.
3. The vecdot function checks `nrc` and dispatches:
   - I8MM build with `nrc == 2` (Q4_0/Q4_1/Q8_0/Q4_K/Q6_K):
     runs the `vmmlaq_s32` tile, writes 4 results via `vst1_f32(s)` +
     `vst1_f32(s+bs)` and returns. **No fallthrough** to lower tiers.
   - SVE build with `nrc == 1`: runs `switch (vector_length)` over
     {128, 256, 512}; writes 1 result.
   - DOTPROD build (selected kernels only, e.g. NVFP4 at line 840):
     runs `vdotq_s32` chain.
   - NEON baseline: runs `ggml_vdotq_s32` 3-op emulation (ARTX04-F02).

### 5.2 Batched GEMV/GEMM (the `repack.cpp` path)

1. `ggml_compute_forward_mul_mat` checks `n > GGML_N_ROWS` and the
   repack callback to decide whether to call the batched kernel.
2. `ggml_gemv_q4_0_8x8_q8_0` (e.g., `repack.cpp:339`) enters, asserts
   shape, then on `__ARM_FEATURE_SVE` checks
   `ggml_cpu_get_sve_cnt() == QK8_0` (== 16 bytes).
3. If the gate is satisfied: executes a single `__asm__ __volatile__`
   block that contains the column loop, block loop, all sdot/smmla
   instructions, and the result store.
4. If the gate is not satisfied: falls through to
   `ggml_gemv_q4_0_8x8_q8_0_generic(...)` — the scalar reference.

### 5.3 KleidiAI path (the `kleidiai/kleidiai.cpp` path)

1. `ggml_backend_cpu_kleidiai_buffer_type()` registers a
   `extra_buffer_type` that overrides `supports_op` and
   `compute_forward` for Q4_0/Q8_0/F32/F16 RHS tensors.
2. At tensor allocation: `init_tensor` runs `repack(tensor, data, …)`,
   which writes a `kleidiai_weight_header` (magic = `KLAI`, version = 1)
   followed by up to 2 packed-RHS slots (SME primary + non-SME fallback
   when SMCUs detected and `nth_total > sme_thread_cap`).
3. At MUL_MAT time: `compute_forward` collects the kernel chain
   (`kleidiai_collect_kernel_chain`), assigns threads to slots based on
   `sme_thread_cap`, packs LHS in parallel by `mr`-aligned chunks,
   synchronizes via `ggml_barrier`, then runs the chunk loop using
   `ggml_threadpool_chunk_set` / `_add` (same atomic-counter scheme as
   ARTX01-F06) and calls `slot.kernel->run_kernel_ex(...)` per chunk.
4. At GET_ROWS time: `compute_forward_get_rows` uses
   `rhs_info.to_float(packed_base, row_idx, nc, out, ...)` to dequantize
   on the fly — useful for embedding-table lookup.

---

## 6. Data Layout

The advanced tiers introduce three layout transformations beyond the
baseline block format:

1. **I8MM 2-row × 2-col interleaving** (`quants.c:362-372` for Q4_0,
   `quants.c:1203-1213` for Q8_0, `quants.c:2431-2442` for Q4_K SVE+I8MM).
   Two weight rows `(x0_l, x0_h, x1_l, x1_h)` and two activation rows
   `(y0_l, y0_h, y1_l, y1_h)` are interleaved with
   `vzip1q_s64`/`vzip2q_s64` into four `int8x16_t` lanes `(l0,l1,l2,l3)`
   and `(r0,r1,r2,r3)` such that each 8-byte lane-pair contains
   `[x0_byte, x1_byte]` for weights (or `[y0_byte, y1_byte]` for
   activations). `vmmlaq_s32(acc, l, r)` then produces a 4-element
   `int32x4_t` where lanes 0/1/2/3 are partial sums for the 4
   combinations `(x0×y0, x0×y1, x1×y0, x1×y1)` — i.e., a 2×2 output
   tile per `vmmlaq` instruction. Four `vmmlaq` chains (l0×r0, l1×r1,
   l2×r2, l3×r3) accumulate the four 16-byte halves of a Q8_0 block.

2. **SVE 2-row × 2-col interleaving (Q4_K, Q6_K only)** — same
   pattern but using `svzip1_s64`/`svzip2_s64` on SVE vectors, with
   `svmmla_s32` (SVE I8MM) producing the 4-lane partial-sum tile.
   Predicates `svptrue_pat_b8(SV_VL16)` (for 128-bit SVE) and
   `svptrue_pat_b8(SV_VL32)` (for 256/512-bit SVE) gate the load/ops.

3. **KleidiAI pre-packed RHS layout** — opaque. The wrapper stores a
   `kleidiai_weight_header` followed by one or two packed-RHS slots.
   The slot offsets and sizes are stored in the header. The layout
   itself is determined by the KleidiAI library's
   `kai_get_rhs_packed_size_*` and `kai_get_rhs_packed_stride_*`
   functions, which are not in scope for this audit.

The remaining layouts — per-block `block_q4_0` / `block_q8_0` /
`block_q4_K` / `block_q8_K` / `block_q6_K` — are unchanged from
baseline. The advanced tiers read the same on-disk format as baseline
and only transform the in-register layout.

---

## 7. Memory Layout

Advanced-tier kernels introduce three memory-pattern shifts relative to
baseline:

1. **2-row × 2-col tile consumption** means the caller must provide
   two weight rows and two activation columns per `vec_dot` call. The
   caller (`ggml_compute_forward_mul_mat_one_chunk` at
   `ggml-cpu.c:1164`) accomplishes this by walking weight rows in
   pairs (`ir += 2`) and activation columns in pairs (via the `bs`
   stride). This halves the per-call overhead but requires the caller
   to align `nb` and `nr` to 2 — otherwise the I8MM tier is bypassed
   (the `nrc==1` fallback runs).

2. **KleidiAI LHS packing** allocates a per-op `wdata` buffer that
   holds the pre-quantized/packed LHS for every batch. The size is
   computed by `tensor_traits::work_size` (`kleidiai.cpp:535+`) by
   summing `lhs_info->packed_size_ex(m, k, …)` across all slots. This
   can be substantial — for a 4096×4096 MUL_MAT with batch=1 and `mr=4`,
   it is ~16 MB of packed int8.

3. **KleidiAI RHS packing** happens eagerly at `init_tensor` time
   (`kleidiai.cpp:1429+`). The packed RHS replaces the original tensor
   data in-place (the buffer was allocated with
   `get_alloc_size` accounting for the packed size). This is a one-time
   cost but means the tensor's data pointer changes interpretation:
   callers that bypass the `extra_buffer_type` and read the raw bytes
   will see the `KLAI` magic, not Q4_0 blocks.

The I8MM inline-asm batched kernels in `repack.cpp` also consume a
specific interleaved layout produced by `ggml_quantize_mat_q8_0_4x4` /
`_4x8` (`repack.cpp:51, 119`) — these functions interleave 4 or 8 rows
of Q8_0 blocks into a single `block_q8_0x4` / `block_q8_0x8` for
column-major access. This interleaving is set up by the repack callback
at tensor allocation time and is invisible to the vecdot path.

---

## 8. Parallelism Strategy

Parallelism at the advanced-tier level is *identical to baseline* at the
chunk level (ARTX01-F06, ARTX04): the matmul path uses
`ggml_compute_forward_mul_mat_one_chunk` with dynamic chunk stealing
via `ggml_threadpool_chunk_add`. The advanced tiers participate only
by being faster per chunk.

The one exception is **KleidiAI's hybrid SME + non-SME threading**
(`kleidiai.cpp:1100-1184`). When `sme_thread_cap > 0` (SMCUs detected)
and `nth_total > sme_thread_cap`, KleidiAI assigns `sme_thread_cap`
threads to the SME slot and the remaining threads to the non-SME
fallback slot (typically DOTPROD or I8MM). Each slot gets its own
`runtime[i].assigned_threads`, `thread_begin`, `thread_end`. Threads
pick their slot by `ith_total ∈ [thread_begin, thread_end)`. Within a
slot, the chunk loop uses the same atomic-counter scheme as ARTX01-F06.

This is the only place in the CPU backend where two different kernel
implementations run concurrently on the same MUL_MAT op. It is made
possible by KleidiAI's `RHS_REPACK_SHARED` mode: both slots consume
the same packed RHS (or two slot-specific packed RHSes laid out
consecutively in the buffer). The non-SME slot must use a kernel that
consumes the same data layout — KleidiAI validates this in
`kleidiai_collect_kernel_chain_common` (`kleidiai.cpp:453-484`) by
checking `lhs_type`, `rhs_type`, and `op_type` match.

---

## 9. SIMD / GPU Strategy

### 9.1 DOTPROD

* `vdotq_s32(acc, a, b)` — int8×int8→int32 lane dot, 4 dot products per
  instruction. Used in `quants.c` only for NVFP4 (`line 840`),
  TQ1_0/TQ2_0 (lines 1417, 1461, 1523, 1551, 1592, 1628, 1662), and
  indirectly through the `ggml_vdotq_s32` macro (which expands to
  `vdotq_s32` under `__ARM_FEATURE_DOTPROD`).
* DOTPROD is the **first non-baseline tier that does not require
  I8MM**. Hardware: Cortex-A78, Neoverse N1/V1 (with DOTPROD), Apple
  A11+, every v8.4a+ core.
* DOTPROD does **not** have a 2-row variant for any vecdot. The 2-row
  pattern is I8MM-exclusive.

### 9.2 I8MM (NEON `vmmlaq_s32`)

* `vmmlaq_s32(acc, a, b)` — int8×int8→int32 matrix multiply, 8×4 dot
  products per instruction (8 columns × 4 row-output lanes). Used in
  the 2-row × 2-col paths for Q4_0/Q4_1/Q8_0/Q4_K (NEON-only
  variant)/Q6_K (NEON-only variant).
* Hardware: Cortex-X2, Neoverse V2, Apple M1+ (Apple implements I8MM
  but reports it via sysctl, not HWCAP2_I8MM), every v8.6a+ / v9-a core.
* Layout: 2 weight rows × 2 activation cols interleaved via
  `vzip1q_s64`/`vzip2q_s64` to form 4 `int8x16_t` pairs consumed by
  4 `vmmlaq` chains. Output is a `float32x4_t` containing
  `[s00, s01, s10, s11]`, stored via `vst1_f32(s, low)` +
  `vst1_f32(s+bs, high)`.

### 9.3 SVE (`svdot_s32`, `svmmla_s32`)

* `svdot_s32(acc, a, b)` — SVE int8 dot product, predicated, VLA. Used
  in the SVE (no I8MM) tier for every standard vecdot.
* `svmmla_s32(acc, a, b)` — SVE I8MM matrix multiply. Used only in the
  SVE+I8MM tier for Q4_K (`quants.c:2443`) and Q6_K
  (`quants.c:3148-3149`). **Not** used for Q4_0/Q4_1/Q8_0.
* Vector-length selection: `switch (vector_length = ggml_cpu_get_sve_cnt()*8)`
  with cases 128/256/512 and `default: assert(false)`. The cached
  `sve_cnt = svcntb()` is initialized once at `ggml_cpu_init` time
  (`ggml-cpu.c:731-733`).
* `svwhilelt_b32(np2, n)` is used for the F32 vecdot tail (`vec.cpp:80`),
  providing a true predicated VLA tail. The integer vecdots do **not**
  use predicated tails — they require `n` to be block-aligned (asserted
  at every vecdot entry).

### 9.4 SVE2

* Used in exactly one function: `ggml_sve_f16_fma_widened`
  (`vec.h:23-38`). SVE2 enables `svmlalb_f32`/`svmlalt_f32` (widened
  F16×F16→F32 FMA, 2 instructions per pair). SVE1 fallback uses
  `svtrn1_f16`/`svtrn2_f16` to split even/odd, then `svcvt_f32_f16_x`
  + `svmla_f32_x` (6 instructions per pair, 3× the op count).
* No quantized int8 kernel uses SVE2-specific instructions. The SVE
  integer path works identically under SVE1 and SVE2.

### 9.5 SME / SME2

* Used **only** inside the KleidiAI integration. KleidiAI's
  `kai_run_matmul_*_sme2_mopa` and `kai_run_matmul_*_sme_mopa`
  functions are called from `kleidiai/kernels.cpp` for Q4_0, Q8_0, F32,
  and F16 RHS tensors when `CPU_FEATURE_SME` or `CPU_FEATURE_SME2` is
  in the feature mask.
* No upstream-written SME outer-product kernel exists. The SME path is
  a "black box" that delegates to the third-party library.
* Detection: `cpu-feats.cpp:43` sets `has_sme = !!(hwcap2 & HWCAP2_SME)`
  on Linux; `cpu-feats.cpp:56-58` uses sysctl on Apple. SME2 is
  detected on Apple only (`cpu-feats.cpp:60-62`); on Linux, the
  `has_sme2` field is left `false` (F11).

### 9.6 Inline-assembly `.inst` opcodes

* `repack.cpp` uses raw `.inst 0x…` encodings for `sdot vN.4s, vM.16b,
  vK.4b[lane]` (e.g., `0x4f9fe18a` at line 1905) and `smmla zN.s,
  zM.b, zK.b` (e.g., `0x451f9872` at line 2817). This bypasses the
  compiler's intrinsic-availability window and lets the author hand-
  pipeline 4×4 / 4×8 / 8×8 tiles.
* KleidiAI's kernels themselves also use this approach internally (out
  of scope for this audit).

---

## 10. Quantization Strategy

The advanced tiers do not introduce new quantization formats. They
accelerate the existing Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/
Q6_K/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_NL/IQ4_XS/Q1_0/
Q2_0/MXFP4/NVFP4/TQ1_0/TQ2_0 set by changing the inner-loop arithmetic
from `ggml_vdotq_s32` (3-op emulation) to `vdotq_s32` (1 op) /
`vmmlaq_s32` (1 op, 8× arithmetic per call) / `svdot_s32` (1 op, VLA) /
`svmmla_s32` (1 op, VLA + 8× arithmetic).

The one quantization-specific decision is **which dtypes get the 2-row
I8MM path**. The type-traits table sets `nrows = 2` only for Q4_0,
Q4_1, Q8_0, Q4_K, Q6_K (and Q5_K does *not* have an I8MM 2-row path
despite being a K-quant). The other quants (Q5_0, Q5_1, Q2_K, Q3_K,
Q5_K, IQ series, TQ1_0, TQ2_0, MXFP4, NVFP4, Q1_0, Q2_0) get only
1-row paths even on I8MM hardware.

This is a measured decision: the 2-row tile requires 4 scale factors
per block (`(x0×y0, x0×y1, x1×y0, x1×y1)` scales, computed at line
354-359 for Q4_0). For quants with complex scale encoding (Q3_K,
Q5_K), the scale-decode cost may exceed the I8MM savings, so the
2-row path is reserved for the formats where scale decode is cheap.

---

## 11. Correctness Analysis

### 11.1 I8MM 2-row reassociation

The 2-row I8MM tile accumulates the dot product of `(x0_l, x0_h)` with
`(y0_l, y0_h)` in a different order than the 1-row baseline path:

- Baseline: `vdotq_s32(0, x0_l, y0_l) → p0; vdotq_s32(p0, x0_h, y0_h) → p1`
  produces a `int32x4_t` of 4 lanes, reduced via `vaddvq_s32(p1)`.
- I8MM: `vmmlaq_s32(0, l0, r0) → acc0; vmmlaq_s32(acc0, l1, r1) → acc1;
  vmmlaq_s32(acc1, l2, r2) → acc2; vmmlaq_s32(acc2, l3, r3) → acc3`
  produces a `int32x4_t` of 4 lanes containing 4 *different* dot
  products (one per `(x_row, y_col)` pair). The final sum for `(x0×y0)`
  is `acc3[0] + acc3[1]` (after the `vextq_f32`/`vzip1q_f32` reduction
  at lines 378-382).

Both produce the same total sum, but the intermediate lane assignments
differ. The `ggml_vdotq_s32` fallback's comment at
`ggml-cpu-impl.h:309` warns about this: *"do not use when individual
lane values matter."* The I8MM tile's reduction at lines 378-382 is
correct, but any future code that reads the per-lane values (rather
than the final reduced scalar) would silently produce different results
between baseline and I8MM builds.

### 11.2 SVE vector-length crash

The `assert(false && "Unsupported vector length")` at 8 sites
(`quants.c:523, 1349, 1938, 2203, 2560, 2784, 3166, 3485`) means that
on hardware with SVE VL ∉ {128, 256, 512}, the kernel crashes in
debug builds and exhibits undefined behavior in release builds. SVE
VL=384 (a hypothetical non-power-of-2 implementation) is not supported.
QEMU-user and FVP models with non-standard VLs are also unsupported.
This is a correctness landmine for any future SVE implementation.

### 11.3 SVE predicate correctness

Each SVE case uses a different predicate: `svptrue_b8()` (all-active),
`svptrue_pat_b8(SV_VL16)` (16-lane), `svptrue_pat_b8(SV_VL32)` (32-lane),
`svptrue_pat_b32(SV_VL4)`, etc. The predicate must match the case's VL.
The 128-bit case uses `svptrue_b8()` (16-lane predicate, matches 128-bit
SVE) but the 256/512 cases use `svptrue_pat_b8(SV_VL16)` /
`svptrue_pat_b8(SV_VL32)` to explicitly activate only the relevant
lanes. This is correct but fragile: if a future maintainer adds a
384-bit case, they must remember to use `svptrue_pat_b8(SV_VL48)`.

### 11.4 KleidiAI dual-slot correctness

The KleidiAI dual-slot scheme (`kleidiai.cpp:453-484`) validates that
the fallback slot has the same `lhs_type`, `rhs_type`, and `op_type` as
the primary slot. This is necessary but not sufficient: the two slots
must also produce the same numerical result. The SME kernel uses
outer-product accumulation in 32-bit float; the DOTPROD/I8MM fallback
uses int8×int8→int32 → float conversion. The two paths will produce
slightly different results due to the SME kernel's intermediate float
accumulation vs the I8MM kernel's int32 accumulation. This means a
MUL_MAT with `nth_total > sme_thread_cap` will produce ULP-level
non-determinism across runs depending on which slot processed which
chunks.

---

## 12. Optimization Analysis

The advanced tiers' primary optimizations are:

1. **Per-instruction throughput**: `vmmlaq_s32` does 8× the arithmetic
   of `vdotq_s32` per instruction (8 dot products per 4-lane output vs
   1). The 2-row × 2-col tile halves the per-call overhead (one scale
   decode + one `vmmlaq` chain produces 4 outputs).
2. **VLA SVE**: 256-bit and 512-bit SVE hardware processes 2× / 4× more
   elements per instruction than 128-bit NEON. The 128/256/512 switch
   captures this, though it requires the hardware to implement one of
   those exact VLs.
3. **Inline-assembly pipelining**: `repack.cpp`'s GEMV/GEMM kernels
   hand-pipeline loads and arithmetic to hide latency. The
   `ggml_quantize_mat_q8_0_4x4`/`_4x8` interleaving ensures 4 or 8
   activation rows are contiguous in memory for vectorized load.
4. **SME streaming-mode outer product**: KleidiAI's SME kernels use
   the 128×16 (or wider) outer-product tile, the largest single-
   instruction arithmetic block on AArch64.
5. **Kernel fusion (none)**: The advanced tiers do not fuse with
   adjacent ops. There is no `MUL_MAT + ADD` (residual) fusion at the
   advanced-tier level. KleidiAI's `compute_forward_get_rows` is a
   form of fusion (dequantize + index) but only for GET_ROWS.
6. **Software prefetch (none)**: No advanced-tier kernel uses `prfm`.
   The inline-asm kernels rely on out-of-order hardware prefetch.

---

## 13. Architectural Strengths

1. **The I8MM 2-row × 2-col tile is the canonical pattern.** Every
   standard Q4/Q8/K-quant that has cheap scale decode (Q4_0, Q4_1,
   Q8_0, Q4_K, Q6_K) gets a 2-row I8MM path. The `vzip1q_s64`/
   `vzip2q_s64` interleaving is clean and the `vmmlaq_s32` chain
   produces the 2×2 output tile directly. GwenLand should ADOPT this
   pattern.

2. **The SVE F32 vecdot (`vec.cpp:11-110`) is a clean VLA design.** It
   uses `ggml_cpu_get_sve_cnt()` to size the step, `svwhilelt_b32` for
   the predicated tail, and 8 independent accumulators for ILP. This
   is the model SVE kernel in the codebase.

3. **The SVE2 widened F16 FMA helper (`vec.h:18-44`) is the cleanest
   SVE1→SVE2 fallback.** Two paths, one clear tradeoff (2 instrs vs 6
   instrs per pair), no conditional compilation gymnastics. This is
   how all SVE2 fallbacks should be structured.

4. **KleidiAI's hybrid SME+non-SME threading model is novel and
   pragmatic.** Real SME hardware has fewer SMCUs than total cores
   (e.g., 2 SMCUs on Apple M4 vs 10 P-cores). Assigning SME threads
   only to SMCUs and the rest to a non-SME fallback maximizes total
   throughput. This is the only such design in the CPU backend.

5. **Multi-binary dispatch is consistent.** `cpu-feats.cpp`'s score
   function returns 0 when a required feature is missing, ensuring the
   .so is never loaded on incompatible hardware. The score weights
   (DOTPROD=2, FP16_VA=4, SVE=8, I8MM=16, SVE2=32, SME=64) are
   power-of-two, so any superset build strictly outscores any subset
   build.

6. **KleidiAI's weight-header scheme is portable.** The `KLAI` magic +
   version + slot offsets allows the packed layout to be upgraded
   in-place and validated at runtime. This is a clean migration story
   for KleidiAI's evolving pack format.

---

## 14. Architectural Weaknesses

### W1 — SVE VL is not truly VLA

**Evidence**: `quants.c:523, 1349, 1938, 2203, 2560, 2784, 3166, 3485`
— `default: assert(false && "Unsupported vector length");`.

**Impact**: SVE hardware with VL ∉ {128, 256, 512} crashes. SVE's
design promise is "write once, run on any VL"; this code breaks that
promise. Future hardware (or current emulators) with VL=384 or VL=2048
will not work. The `ggml_vec_dot_f32` SVE path (`vec.cpp:11-110`) shows
the correct VLA pattern (no switch, just predicated tails), but the
quantized integer paths do not follow it.

**Why it's hard to fix**: The quantized paths use block-aligned
predicates that depend on the exact VL. A true VLA implementation
would need to handle arbitrary VL/block-size ratios, which is
non-trivial for K-quants (block size 256) and IQ quants (variable
block sizes).

### W2 — SVE+I8MM 2-row path missing for Q4_0/Q4_1/Q8_0

**Evidence**: `quants.c:315-386` (Q4_0 I8MM 2-row, NEON `vmmlaq_s32`
only). No `#if defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_MATMUL_INT8)`
2-row block for Q4_0/Q4_1/Q8_0 exists in `quants.c`.

**Impact**: On SVE-capable + I8MM hardware (e.g., Neoverse V2, which
has both SVE 128-bit and I8MM), the Q4_0/Q4_1/Q8_0 vecdot takes the
NEON `vmmlaq_s32` path even though an SVE `svmmla_s32` path could
potentially run at 256-bit width on a 256-bit SVE implementation. On
128-bit SVE hardware, this is not a regression (svmmla_s32 and
vmmlaq_s32 produce the same per-instruction throughput at 128 bits).

**Why it's hard to fix**: The Q4_K SVE+I8MM path (`quants.c:2360-2568`)
is 208 lines of dense SVE code with predicate manipulation and scale
decoding. Replicating this for Q4_0/Q4_1/Q8_0 is significant work for
marginal gain on 128-bit SVE hardware.

### W3 — DOTPROD has no 2-row path

**Evidence**: `quants.c:840-856` (NVFP4 DOTPROD), `quants.c:1417-1471`
(TQ1_0 DOTPROD). All DOTPROD paths process one row at a time. The
type-traits table sets `nrows = 2` only under
`__ARM_FEATURE_MATMUL_INT8`, never under `__ARM_FEATURE_DOTPROD`.

**Impact**: DOTPROD-only hardware (Cortex-A78, Neoverse N1 with
DOTPROD) does not get the 2-row tile benefit. For these cores, the
I8MM 2-row path is unavailable (no I8MM) and the DOTPROD 1-row path
leaves 50% of the per-call overhead on the table.

### W4 — `has_sme2` field is dead on Linux

**Evidence**: `cpu-feats.cpp:31` declares `has_sme2`; line 60-62
populates it on Apple only. No Linux code path reads `HWCAP2_SME2`
(the define is in `kleidiai.cpp:30` but not in `cpu-feats.cpp`).
`ggml_backend_cpu_aarch64_score` (line 69-99) does not check
`GGML_USE_SME2` because no such macro exists.

**Impact**: On Linux hardware with SME2 (e.g., future Neoverse V3 /
Cortex-X4 with FEAT_SME2), the `has_sme2` flag is always `false`. The
score function would not penalize a SME2-compiled .so for lack of
SME2 — but since SME2 implies SME (per Arm ARM), an SME-compiled .so
would still score correctly. The bigger issue is the dead field: a
future maintainer reading `has_sme2` would assume it's populated on
all platforms.

### W5 — Inline-assembly kernels in `repack.cpp` are unmaintainable

**Evidence**: `repack.cpp:339-428` (Q4_0 8x8 GEMV, 90 lines of asm),
`repack.cpp:2728-3160` (Q4_0 8x8 GEMM, 432 lines of asm). Each kernel
encodes register allocation, predicate setup, and instruction order
by hand.

**Impact**: Bugs in the asm (wrong register, wrong predicate) are
silent and hard to debug. Porting to a new dtype requires writing a
new 100-400 line asm block. The kernels are also compiler-agnostic —
they bypass clang's SVE intrinsic availability — but pay for it with
zero compiler optimization.

### W6 — SVE batched GEMV/GEMM gated on `ggml_cpu_get_sve_cnt() == QK8_0`

**Evidence**: `repack.cpp:360` (`if (ggml_cpu_get_sve_cnt() == QK8_0)`),
`repack.cpp:2750` (same gate for GEMM).

**Impact**: 256-bit and 512-bit SVE hardware falls through to the
`_generic` scalar path for the 8x8 batched GEMV/GEMM. The hand-written
SVE asm was authored for 128-bit SVE only. On 256-bit SVE hardware
(e.g., some future Neoverse), this means the batched GEMV/GEMM is
*slower* than on 128-bit SVE hardware, because the scalar fallback
runs instead of the SVE asm. This is a regression risk.

### W7 — DOTPROD / I8MM detection is build-time only, not runtime

**Evidence**: `cpu-feats.cpp:73-96` checks the compile-time macros
`GGML_USE_DOTPROD`, `GGML_USE_MATMUL_INT8`, etc. against the runtime
HWCAP bits. If a feature is compiled in but absent at runtime, the
score returns 0 (the .so is not loaded). If a feature is present at
runtime but not compiled in, the score returns 1 (baseline .so is
loaded).

**Impact**: A single .so cannot dynamically choose between DOTPROD
and I8MM kernels at runtime based on which is faster for a given
shape. The KleidiAI integration does some of this (it picks between
SME, I8MM, and DOTPROD kernels at runtime via `kai_*` selection), but
the upstream vecdot path cannot.

### W8 — KleidiAI SME/non-SME hybrid produces ULP-level non-determinism

**Evidence**: `kleidiai.cpp:1131-1171` — threads split between SME
and non-SME slots; chunks are assigned via atomic counter (non-
deterministic).

**Impact**: A MUL_MAT with `nth_total > sme_thread_cap` produces
slightly different results across runs depending on which threads
processed which chunks via which kernel. This is the same ULP-level
non-determinism noted in ARTX01-F06, but compounded by the use of two
different kernel implementations with different accumulation orders.

### W9 — Q5_K has no I8MM 2-row path

**Evidence**: `quants.c:2866` asserts `nrc == 1` unconditionally for
Q5_K. The type-traits table (ARTX01-F03) does not set `nrows = 2` for
Q5_K even under I8MM.

**Impact**: Q5_K is the only "cheap-scale" K-quant that does not get
the I8MM 2-row benefit. The Q5_K scale decode is similar to Q4_K
(which does get the 2-row path), so the omission appears to be an
oversight rather than a measured decision.

### W10 — `ggml_init_arm_arch_features` is called once but not thread-safe

**Evidence**: `ggml-cpu.c:731-733` initializes `sve_cnt = svcntb()`.
The function is called from `ggml_cpu_init` (line 3818+) under a
critical section, but `sve_cnt` is a plain `int` field on a static
struct (no `std::atomic`). Reads via `ggml_cpu_get_sve_cnt()` are not
synchronized.

**Impact**: In practice this is fine because `ggml_cpu_init` is called
before any tensor ops, but it's a latent data race if any code path
calls `svcntb()` from multiple threads.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc`        | **ADOPT** | I8MM 2-row × 2-col `vzip1q_s64`/`vmmlaq_s32` tile | Clean, canonical, 8× arithmetic per instruction. (F01) |
| `glproc`        | **ADOPT** | One-shot `svcntb()` cached in static struct | Cheap runtime VL discovery, single function call site. (F03) |
| `glproc`        | **ADOPT** | SVE2 widened F16 FMA helper (`ggml_sve_f16_fma_widened`) | Clean SVE1→SVE2 fallback pattern, 3× fewer instructions. (F05, F09) |
| `glproc`        | **ADOPT** | KleidiAI `extra_buffer_type` plugin with KLAI weight header | Plugin architecture for vendor SDK integration. (F07, F12) |
| `glproc`        | **ADOPT** | KleidiAI hybrid SME+non-SME threading | Maximizes throughput on partial-SME hardware (e.g., Apple M4). (F12) |
| `glproc`        | **ADAPT** | SVE VLA switch (128/256/512) | Keep the switch but replace `assert(false)` with a graceful fallback to NEON at 128-bit equivalent. (F02) |
| `glproc`        | **REJECT**| `assert(false)` on unsupported SVE VL | Crash on non-standard VL is a correctness landmine. (F02) |
| `glproc`        | **REJECT**| Absence of SVE+I8MM 2-row path for Q4_0/Q4_1/Q8_0 | Marginal gain on 128-bit SVE, but blocking on 256-bit SVE. (F08) |
| `glproc`        | **REJECT**| Q5_K has no I8MM 2-row path | Appears to be an oversight; Q4_K has one. (W9) |
| `glproc`        | **MONITOR**| Inline-asm `.inst` opcodes in `repack.cpp` | Maintainability cost; revisit when compiler intrinsics are universally available. (F10) |
| `glproc`        | **MONITOR**| `has_sme2` dead field on Linux | Latent bug for future SME2-only features. (F11) |
| `glproc`        | **MONITOR**| KleidiAI dual-slot ULP non-determinism | Acceptable for inference; problematic for differential testing. (W8) |
| `glproc`        | **DEFER** | SME outer-product kernels (write upstream) | SME is fully delegated to KleidiAI today; revisit if GwenLand wants to drop the KleidiAI dependency. (F06) |
| `GATE`          | **ADOPT** | `nrows = 2` flag in type-traits for I8MM 2-row | Already covered by ARTX01-F03; reaffirmed here. (F01) |
| `GATE`          | **ADOPT** | KleidiAI's `compute_forward_get_rows` fusion | Dequantize-on-index is a clean fusion for embedding tables. (F07) |

---

## 16. Recommendations

### R1 — ADOPT I8MM 2-row × 2-col tile pattern
**Priority:** Critical
**Difficulty:** M
**Dependencies:** ARTX01-F03 (type-traits table)
GwenLand's `glproc` should implement the `vzip1q_s64`/`vzip2q_s64` +
`vmmlaq_s32` chain for Q4_0/Q4_1/Q8_0/Q4_K/Q6_K. The pattern is in
`quants.c:362-382` (Q4_0), `quants.c:1203-1223` (Q8_0),
`quants.c:2580+` (Q4_K NEON I8MM), `quants.c:3176+` (Q6_K NEON I8MM).
Each tile produces 4 outputs per call (2 weight rows × 2 activation
cols). The caller must set `nrows = 2` in the type-traits entry.

### R2 — ADOPT one-shot SVE VL discovery via `svcntb()` cached at init
**Priority:** High
**Difficulty:** XS
**Dependencies:** none
Replicate `ggml_init_arm_arch_features` (`ggml-cpu.c:731-733`) +
`ggml_cpu_get_sve_cnt` (`ggml-cpu.c:3794-3800`). Cache `svcntb()` once
in a static struct, expose via a single function. Avoids repeated
`svcntb()` calls in hot loops (which would be no-ops on most hardware
but a code-cleanliness issue).

### R3 — ADAPT SVE VLA switch with graceful fallback
**Priority:** High
**Difficulty:** M
**Dependencies:** R2
Keep the `switch (vector_length)` pattern but replace
`default: assert(false)` with a fallback to the NEON baseline path.
This preserves the optimized 128/256/512 paths while not crashing on
non-standard VLs. The fallback can call the existing NEON baseline
vecdot directly.

### R4 — ADOPT SVE2 widened F16 FMA helper
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R2
Replicate `ggml_sve_f16_fma_widened` (`vec.h:18-44`). The SVE2 path
uses `svmlalb_f32`/`svmlalt_f32` (2 instrs per pair); the SVE1 fallback
uses `svtrn1_f16`/`svtrn2_f16` + `svcvt_f32_f16_x` + `svmla_f32_x`
(6 instrs per pair). Clean 3× speedup on SVE2 hardware.

### R5 — ADOPT KleidiAI-style `extra_buffer_type` plugin for vendor SDKs
**Priority:** High
**Difficulty:** L
**Dependencies:** ARTX01-F04 (extra-buffer-type hook)
GwenLand will likely want to integrate vendor SDKs (KleidiAI, AMX,
SpacemiT). The `extra_buffer_type` plugin pattern with a weight-header
scheme (magic + version + slot offsets) is the clean way to do this.
The KLAI header at `kleidiai.cpp:392-419` is the model.

### R6 — ADOPT KleidiAI's hybrid SME+non-SME threading
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R5
When SME hardware has fewer SMCUs than total cores, assign only
`sme_thread_cap` threads to the SME slot and the rest to a non-SME
fallback. The runtime-slot assignment at `kleidiai.cpp:1131-1184` is
the reference. This maximizes throughput on partial-SME hardware
(Apple M4: 2 SMCUs / 10 P-cores).

### R7 — REJECT absence of SVE+I8MM 2-row path for Q4_0/Q4_1/Q8_0
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1, R3
On 256-bit SVE + I8MM hardware (e.g., future Neoverse V3 with 256-bit
SVE), the Q4_0/Q4_1/Q8_0 vecdot should use `svmmla_s32` at 256-bit
width instead of falling back to NEON `vmmlaq_s32` at 128-bit width.
The Q4_K SVE+I8MM path (`quants.c:2360-2568`) is the reference.

### R8 — REJECT Q5_K missing I8MM 2-row path
**Priority:** Low
**Difficulty:** M
**Dependencies:** R1
Add the I8MM 2-row path for Q5_K. The Q5_K scale decode is similar to
Q4_K; the Q4_K 2-row path can be adapted. Set `nrows = 2` for Q5_K in
the type-traits table under `__ARM_FEATURE_MATMUL_INT8`.

### R9 — MONITOR inline-asm `.inst` opcodes in `repack.cpp`
**Priority:** Low
**Difficulty:** N/A
Revisit when compiler SVE intrinsics are universally available (clang
14+, gcc 12+). Replace the inline asm with intrinsic-based C++ to
improve maintainability. The hand-pipelined schedules may need to be
preserved via `#pragma unroll` and explicit register hints.

### R10 — MONITOR `has_sme2` dead field on Linux
**Priority:** Low
**Difficulty:** XS
**Dependencies:** R5
Add `#if !defined(HWCAP2_SME2) #define HWCAP2_SME2 (1UL << 37) #endif`
to `cpu-feats.cpp` and populate `has_sme2 = !!(hwcap2 & HWCAP2_SME2)`
in the Linux constructor. Add a `GGML_USE_SME2` CMake macro and a
score-function check. Even if no current code uses SME2 directly, the
field should not lie.

### R11 — DEFER upstream SME outer-product kernels
**Priority:** Low
**Difficulty:** XL
**Dependencies:** R5, R6
GwenLand can rely on KleidiAI for SME execution. Writing upstream SME
kernels is high-effort (streaming-mode entry/exit, tile register
allocation, SVL-aware outer-product scheduling) for limited gain over
the KleidiAI integration. Revisit only if GwenLand wants to drop the
KleidiAI dependency.

---

## 17. Findings

### Finding ARTX05-F01

```
Finding ID:           ARTX05-F01
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            I8MM 2-row × 2-col vecdot tile
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0 (and 5 sibling I8MM 2-row vecdots)
Lines:                315-385 (Q4_0); 608-680 (Q4_1); 1168-1226 (Q8_0);
                      2360-2568 (Q4_K SVE+I8MM); 2569+ (Q4_K NEON I8MM);
                      2984-3174 (Q6_K SVE+I8MM); 3175+ (Q6_K NEON I8MM)
Summary:              The I8MM tier consumes nrc==2 (2 weight rows × 2 activation
                      cols) per call, interleaving data via vzip1q_s64/vzip2q_s64
                      and producing 4 float32 outputs per block via vmmlaq_s32
                      (NEON) or svmmla_s32 (SVE) chains.
Observation:          Each I8MM vecdot entry begins with `if (nrc == 2) { ... return; }`
                      before falling through to the 1-row SVE/NEON path. Inside
                      the nrc==2 block, two weight rows (vx0, vx1) and two
                      activation rows (vy0, vy1) are loaded, unpacked to int8,
                      and interleaved with `vzip1q_s64`/`vzip2q_s64` to form four
                      int8x16_t pairs (l0,l1,l2,l3 for weights; r0,r1,r2,r3 for
                      activations) such that each 8-byte lane pair contains
                      [x0_byte, x1_byte] or [y0_byte, y1_byte]. Four vmmlaq_s32
                      chains accumulate the four 16-byte halves. The final
                      float32x4_t (lanes = [s00, s01, s10, s11]) is reduced via
                      vextq_f32 + vzip1q_f32 and stored with vst1_f32(s, low) +
                      vst1_f32(s+bs, high). The 2-row pattern is gated on
                      __ARM_FEATURE_MATMUL_INT8 only; DOTPROD has no 2-row
                      variant.

                      The 4-lane output corresponds to the 2×2 outer-product
                      partial sums: lane 0 = (x0×y0), lane 1 = (x0×y1), lane 2 =
                      (x1×y0), lane 3 = (x1×y1). Each lane is the sum of 4
                      vmmlaq outputs (one per 16-byte half-block), so the total
                      dot product for (x0×y0) = lane 0 + lane 2 (after the
                      vextq+vzip reduction at lines 378-382).

Evidence:             quants.c:315 (#if defined(__ARM_FEATURE_MATMUL_INT8));
                      quants.c:316 (if nrc==2); quants.c:362-372 (vzip1q_s64
                      interleave); quants.c:374-375 (4 vmmlaq_s32 chain);
                      quants.c:378-382 (vextq_f32 + vzip1q_f32 reduction +
                      vst1_f32 store).

Architectural Impact: The I8MM 2-row tile is the highest-throughput per-block
                      vecdot pattern in the codebase. Each vmmlaq_s32 does 8
                      int8 dot products per instruction (vs 1 for vdotq_s32),
                      and the 2-row consumption halves per-call scale-decode
                      overhead. The pattern is clean and reproducible across
                      dtypes.

Correctness Impact:   The 2-row path produces the same total sums as the 1-row
                      baseline path, but with different per-lane grouping (see
                      §11.1). The reduction at lines 378-382 is correct. Any
                      future code that reads individual lanes (rather than the
                      reduced scalar) would silently produce different results
                      between baseline and I8MM builds.

Optimization Type:    SIMD (2-row × 2-col tile via vmmlaq_s32).

GwenLand Target:      glproc

Recommendation:       ADOPT. Implement the vzip+vmmlaq_s32 chain for Q4_0/Q4_1/
                      Q8_0/Q4_K/Q6_K in glproc. Set nrows=2 in the type-traits
                      table under __ARM_FEATURE_MATMUL_INT8.

Priority:             Critical
Difficulty:           M
Dependencies:         ARTX01-F03 (type-traits table)
Confidence:           High
```

### Finding ARTX05-F02

```
Finding ID:           ARTX05-F02
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            SVE vector-length switch
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0 (SVE branch), ggml_vec_dot_q8_0_q8_0
                      (SVE branch), and 6 other SVE vecdots
Lines:                397-525 (Q4_0 SVE); 1239-1349 (Q8_0 SVE); 1705-1938
                      (Q2_K SVE); 2070-2203 (Q3_K SVE); 2376-2560 (Q4_K SVE+I8MM);
                      2745-2784 (Q5_K SVE — no I8MM); 2995-3166 (Q6_K SVE+I8MM);
                      3371-3485 (Q6_K SVE-only)
Summary:              Every SVE vecdot wraps its body in switch(vector_length)
                      with cases 128/256/512 and default: assert(false && "Unsupported
                      vector length"). SVE hardware with VL outside {128, 256, 512}
                      crashes.
Observation:          The pattern is `const int vector_length =
                      ggml_cpu_get_sve_cnt()*8; switch (vector_length) { case 128:
                      ...; case 256: ...; case 512: ...; default: assert(false
                      && "Unsupported vector length"); }`. The cached sve_cnt is
                      bytes-per-SVE-register (returned by svcntb()); multiplied
                      by 8 to get bits. The 8 sites are at quants.c:523, 1349,
                      1938, 2203, 2560, 2784, 3166, 3485.

                      The switch is used because each VL requires different
                      predicate patterns (svptrue_pat_b8(SV_VL16) for 128-bit,
                      SV_VL32 for 256/512-bit) and different load widths. A
                      truly VLA implementation would use svptrue_b8() (all-
                      active) and let the hardware size the operation — but the
                      block-aligned nature of the quantized kernels (block size
                      32 or 256) means the VLA pattern would need to handle
                      arbitrary VL/block ratios, which is non-trivial.

                      The F32 SVE vecdot at vec.cpp:11-110 is the model VLA
                      implementation (no switch, predicated tail via
                      svwhilelt_b32). The integer quantized paths do not follow
                      this pattern.

Evidence:             quants.c:395 (const int vector_length =
                      ggml_cpu_get_sve_cnt()*8); quants.c:398 (switch);
                      quants.c:522-524 (default: assert(false)).
                      quants.c:1705-1938 (Q2_K SVE switch with same default).
                      quants.c:2070-2203 (Q3_K SVE switch).
                      vec.cpp:11-110 (F32 SVE — model VLA, no switch).

Architectural Impact: SVE's design promise is "write once, run on any VL." This
                      code breaks that promise. Future hardware with VL=384 or
                      VL=2048 (both architecturally valid) will crash. QEMU-user
                      with non-default VL also crashes. The 8 assert(false) sites
                      are correctness landmines.

Correctness Impact:   In debug builds: process abort on unsupported VL. In
                      release builds (NDEBUG): undefined behavior — the function
                      falls through the switch without setting sumf, returning
                      uninitialized memory. Either way, incorrect.

Optimization Type:    SIMD (compile-time VL selection).

GwenLand Target:      glproc

Recommendation:       ADAPT. Keep the switch for the 128/256/512 fast paths but
                      replace default: assert(false) with a fallback to the NEON
                      baseline vecdot. This preserves the optimized paths while
                      not crashing on non-standard VLs. The fallback can call
                      the existing NEON baseline vecdot directly.

Priority:             High
Difficulty:           M
Dependencies:         ARTX04-F02 (ggml_vdotq_s32 baseline macro)
Confidence:           High
```

### Finding ARTX05-F03

```
Finding ID:           ARTX05-F03
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            SVE vector-length discovery
Source File:          ggml/src/ggml-cpu/ggml-cpu.c
Function:             ggml_init_arm_arch_features, ggml_cpu_get_sve_cnt
Lines:                728-737 (init); 3794-3800 (getter)
Summary:              SVE vector length is discovered once at ggml_cpu_init time
                      via svcntb() and cached in a static struct
                      ggml_arm_arch_features.sve_cnt. The cached value is exposed
                      via ggml_cpu_get_sve_cnt() (returns bytes-per-SVE-register).
Observation:          ggml_init_arm_arch_features (line 731-733) is a one-line
                      function: `ggml_arm_arch_features.sve_cnt = svcntb();`.
                      It is called from ggml_cpu_init under a critical section.
                      ggml_cpu_get_sve_cnt (line 3794-3800) returns the cached
                      value, or 0 if __ARM_FEATURE_SVE is not defined.

                      The cached value is consulted in 7 vecdot kernels
                      (quants.c:395, 1236, 2357, 2982, and 4 more via svcntb()
                      directly), 2 vec.cpp sites (vec.cpp:22, vec.h:323, 708),
                      2 repack.cpp sites (repack.cpp:360, 2750), 1 kleidiai.cpp
                      site (kleidiai.cpp:209), and 1 ggml-cpu.cpp site for
                      feature reporting (line 591-593).

                      svcntb() is a compiler builtin that compiles to a single
                      CNTB instruction. Caching it avoids redundant CNTB
                      instructions in hot loops (though CNTB is a 1-cycle
                      instruction on most hardware, so the perf benefit is
                      negligible). The real value is API cleanliness: callers
                      get a single function rather than having to wrap
                      svcntb() in #ifdef __ARM_FEATURE_SVE.

Evidence:             ggml-cpu.c:728-737 (init function);
                      ggml-cpu.c:3794-3800 (getter);
                      ggml-cpu.c:90 (struct field declaration);
                      quants.c:395 (callsite: `vector_length =
                      ggml_cpu_get_sve_cnt()*8`);
                      kleidiai.cpp:209 (callsite: `ggml_cpu_get_sve_cnt() ==
                      QK8_0`).

Architectural Impact: Centralized SVE VL discovery is a clean pattern. A single
                      init call, a single getter, no per-kernel #ifdef
                      gymnastics. This is the right way to expose a runtime
                      property of the hardware to the kernel layer.

Correctness Impact:   None. svcntb() returns the same value for the lifetime of
                      the process (SVE VL is fixed at EL0 by the kernel).

Optimization Type:    None (caching a 1-cycle instruction).

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the init-once + getter pattern in
                      glproc. Expose via a single function
                      gl_cpu_get_sve_cnt() returning bytes-per-SVE-register.

Priority:             High
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX05-F04

```
Finding ID:           ARTX05-F04
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            DOTPROD vs I8MM distinct paths
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0 (I8MM branch), ggml_vec_dot_nvfp4_q8_0
                      (DOTPROD branch), ggml_vec_dot_tq1_0_q8_K (DOTPROD branch)
Lines:                315-386 (I8MM Q4_0 2-row); 840-856 (DOTPROD NVFP4);
                      1417-1471 (DOTPROD TQ1_0); 1461-1471 (DOTPROD TQ1_0 inner);
                      1523-1550 (DOTPROD TQ2_0); 1628-1660 (DOTPROD TQ2_0 inner)
Summary:              DOTPROD (vdotq_s32, 1×4 dot per instr) and I8MM
                      (vmmlaq_s32, 8×4 dot per instr) are distinct compile-time
                      paths. DOTPROD appears only in NVFP4/TQ1_0/TQ2_0 vecdots
                      and never as a 2-row tile. I8MM appears only as a 2-row
                      tile for Q4_0/Q4_1/Q8_0/Q4_K/Q6_K.
Observation:          The 4-tier ladder is MATMUL_INT8 → SVE → DOTPROD (some
                      kernels) → NEON. DOTPROD and I8MM are NOT nested: a kernel
                      that has DOTPROD does not automatically get I8MM. The
                      selection is purely at .so build time via
                      __ARM_FEATURE_DOTPROD vs __ARM_FEATURE_MATMUL_INT8.

                      DOTPROD-only hardware (Cortex-A78, Neoverse N1 with
                      DOTPROD) gets the 1-row vdotq_s32 path. I8MM hardware
                      (Cortex-X2, Neoverse V2, Apple M1+, every v8.6a+/v9-a
                      core) gets the 2-row vmmlaq_s32 path. Hardware with both
                      DOTPROD and I8MM (which is all I8MM hardware, since I8MM
                      requires DOTPROD as a prerequisite per Arm ARM) takes the
                      I8MM path — the DOTPROD path is unreachable.

                      The DOTPROD path for NVFP4 (line 840-856) is a 1-row
                      pattern: `vdotq_s32(vdupq_n_s32(0), q4_lo_0, q8_lo_0)`
                      produces a 4-lane int32x4_t reduced via vpaddq_s32 +
                      vaddvq_s32. No 2-row DOTPROD variant exists.

Evidence:             quants.c:840 (#if defined(__ARM_FEATURE_DOTPROD) in NVFP4);
                      quants.c:852-856 (vdotq_s32 calls);
                      quants.c:315 (#if defined(__ARM_FEATURE_MATMUL_INT8) in
                      Q4_0);
                      quants.c:374 (vmmlaq_s32 chain);
                      quants.c:302-306 (assertion: nrc==1 unless MATMUL_INT8 —
                      DOTPROD-only build asserts nrc==1).

Architectural Impact: The DOTPROD-only path is a measured decision: DOTPROD
                      hardware lacks the 8× arithmetic density of I8MM, so a
                      2-row DOTPROD tile would have half the per-call overhead
                      but no per-instruction gain. The 2-row pattern is
                      I8MM-exclusive because I8MM's vmmlaq_s32 is the only
                      instruction that can fill the 2×2 output tile in a single
                      op.

Correctness Impact:   None. Both paths produce identical total sums.

Optimization Type:    SIMD (compile-time ISA selection).

GwenLand Target:      glproc

Recommendation:       ADOPT the 4-tier ladder with the understanding that DOTPROD
                      is a 1-row-only tier. Document that DOTPROD hardware gets
                      no 2-row benefit; I8MM hardware is required for nrows=2.

Priority:             Medium
Difficulty:           S
Dependencies:         ARTX04-F02 (ggml_vdotq_s32 macro)
Confidence:           High
```

### Finding ARTX05-F05

```
Finding ID:           ARTX05-F05
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            SVE2 widened F16 FMA helper
Source File:          ggml/src/ggml-cpu/vec.h
Function:             ggml_sve_f16_fma_widened
Lines:                17-44
Summary:              The helper performs F16×F16→F32 widened FMA using two
                      accumulators (lo/hi). Under __ARM_FEATURE_SVE2 it uses
                      svmlalb_f32/svmlalt_f32 (2 instrs per pair). Under SVE1 it
                      falls back to svtrn1_f16/svtrn2_f16 + svcvt_f32_f16_x +
                      svmla_f32_x (6 instrs per pair, 3× the op count).
Observation:          The function signature is `ggml_sve_f16_fma_widened(
                      svfloat32_t *acc_lo, svfloat32_t *acc_hi, svfloat16_t x,
                      svfloat16_t y)`. It widens F16×F16 to F32 and accumulates
                      into two F32 accumulators (lo = even lanes, hi = odd
                      lanes).

                      SVE2 path (line 23-25):
                        *acc_lo = svmlalb_f32(*acc_lo, x, y);  // bottom-half widening FMA
                        *acc_hi = svmlalt_f32(*acc_hi, x, y);  // top-half widening FMA
                      2 instructions, no extra work.

                      SVE1 fallback (line 27-37):
                        svfloat16_t x_even = svtrn1_f16(x, x);  // even lanes duplicated
                        svfloat16_t x_odd  = svtrn2_f16(x, x);  // odd lanes duplicated
                        svfloat16_t y_even = svtrn1_f16(y, y);
                        svfloat16_t y_odd  = svtrn2_f16(y, y);
                        svbool_t pg = svptrue_b32();
                        *acc_lo = svmla_f32_x(pg, *acc_lo,
                            svcvt_f32_f16_x(pg, x_even), svcvt_f32_f16_x(pg, y_even));
                        *acc_hi = svmla_f32_x(pg, *acc_hi,
                            svcvt_f32_f16_x(pg, x_odd), svcvt_f32_f16_x(pg, y_odd));
                      6 instructions (2 trn + 2 cvt + 2 FMA) plus 2 cvt hidden
                      in the svmla operands = 8 instructions total.

                      The SVE2 path is the only SVE2-specific code in the entire
                      CPU backend (grep __ARM_FEATURE_SVE2 returns 1 hit in
                      vec.h:23). No quantized int8 kernel uses SVE2-specific
                      instructions.

Evidence:             vec.h:17-44 (full function); vec.h:152-209 (callsite in
                      ggml_vec_dot_f16_unroll); vec.h:23 (#if defined
                      (__ARM_FEATURE_SVE2)).

Architectural Impact: SVE2 brings a clean 3-4× speedup for F16 widened FMA. The
                      pattern is the cleanest SVE1→SVE2 fallback in the codebase:
                      two paths, clear tradeoff, no conditional compilation
                      gymnastics. This is the model for how all SVE2 fallbacks
                      should be structured.

Correctness Impact:   Both paths produce the same widened F32 accumulators. The
                      SVE1 fallback duplicates even/odd lanes via trn1/trn2 and
                      then converts to F32 — equivalent to the SVE2 path's
                      widening FMA, just with more instructions.

Optimization Type:    SIMD (SVE2 widening FMA vs SVE1 fallback).

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the SVE2/SVE1 fallback pattern for any
                      widened FMA operation in glproc.

Priority:             Medium
Difficulty:           S
Dependencies:         ARTX05-F03 (SVE VL discovery)
Confidence:           High
```

### Finding ARTX05-F06

```
Finding ID:           ARTX05-F06
Category:             MISSING_FEATURE
Engine:               CPU
Component:            SME / SME2 kernel coverage
Source File:          ggml/src/ggml-cpu/ggml-cpu.c, ggml/src/ggml-cpu/kleidiai/kernels.cpp
Function:             ggml_cpu_has_sme, ggml_cpu_has_sme2 (detection only);
                      kai_run_matmul_*_sme2_mopa / kai_run_matmul_*_sme_mopa
                      (KleidiAI execution)
Lines:                ggml-cpu.c:3802-3816 (detection);
                      kleidiai/kernels.cpp:315, 706, 928, 1043, 1081, 1101,
                      1118 (kernel table entries)
Summary:              SME/SME2 are detected in ggml-cpu.c and used exclusively
                      inside the KleidiAI integration. No upstream-written SME
                      kernel exists. All SME execution is delegated to the
                      bundled third-party KleidiAI library via the
                      extra_buffer_type plugin.
Observation:          grep __ARM_FEATURE_SME across ggml/src/ggml-cpu/ returns 8
                      hits in kleidiai/kernels.cpp (kernel table entries for SME
                      and SME2 variants) and 1 hit in ggml-cpu.c:3803 (the
                      ggml_cpu_has_sme detection function). No hits in quants.c,
                      repack.cpp, vec.cpp, vec.h, ops.cpp, or ggml-cpu.c's
                      kernel paths.

                      The KleidiAI kernel tables (kernels.cpp:315-703 for Q4_0;
                      705-924 for Q8_0; 927-1031 for F32) define SME2 and SME
                      variants that call kai_run_matmul_*_sme2_mopa (SME2
                      outer-product accumulate) and kai_run_matmul_*_sme_mopa
                      (SME outer-product accumulate). These functions are
                      implemented in the bundled KleidiAI library (out of scope
                      for this audit).

                      The CPU_FEATURE_SME / CPU_FEATURE_SME2 flags are set in
                      kleidiai.cpp:262-274 based on ggml_cpu_has_sme() and
                      (Apple-only) sysctl for FEAT_SME2. The kernel selection
                      at kleidiai.cpp:278-280 picks the first matching kernel
                      from the table.

Evidence:             grep __ARM_FEATURE_SME in ggml/src/ggml-cpu/: 1 hit in
                      ggml-cpu.c (detection), 8 hits in kleidiai/kernels.cpp
                      (kernel table entries), 0 hits elsewhere.
                      ggml-cpu.c:3802-3808 (ggml_cpu_has_sme);
                      ggml-cpu.c:3810-3816 (ggml_cpu_has_sme2);
                      kleidiai/kernels.cpp:315 (#if defined(__ARM_FEATURE_SME));
                      kleidiai/kernels.cpp:755, 977 (CPU_FEATURE_SME2);
                      kleidiai/kernels.cpp:808, 1030 (CPU_FEATURE_SME).

Architectural Impact: SME is the largest single-instruction arithmetic block on
                      AArch64 (128×16 outer product per FMOPA). Delegating it
                      entirely to KleidiAI means llama.cpp has no SME expertise
                      in-house. This is a reasonable decision — KleidiAI is
                      Arm's reference implementation — but it creates a hard
                      dependency on the third-party library for SME hardware.

                      For GwenLand, the question is whether to (a) also delegate
                      to KleidiAI (ADOPT the wrapper) or (b) write upstream SME
                      kernels. Option (a) is faster to ship; option (b) avoids
                      the dependency.

Correctness Impact:   None. SME execution correctness is the responsibility of
                      the KleidiAI library.

Optimization Type:    SIMD (SME outer-product matmul, delegated).

GwenLand Target:      glproc

Recommendation:       DEFER upstream SME kernels. ADOPT the KleidiAI wrapper
                      pattern (F07, F12) for SME execution. Revisit only if
                      GwenLand wants to drop the KleidiAI dependency.

Priority:             Low
Difficulty:           XL
Dependencies:         ARTX05-F07 (KleidiAI wrapper)
Confidence:           High
```

### Finding ARTX05-F07

```
Finding ID:           ARTX05-F07
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            KleidiAI integration
Source File:          ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
Function:             ggml::cpu::kleidiai::extra_buffer_type::supports_op,
                      ::compute_forward, ::compute_forward_get_rows, ::repack
Lines:                534-820 (tensor_traits); 1336-1426 (get_rows);
                      1429-1610 (repack); 1681-1761 (extra_buffer_type)
Summary:              KleidiAI registers an extra_buffer_type that claims
                      MUL_MAT and GET_ROWS for Q4_0/Q8_0/F32/F16 RHS tensors
                      when a pre-packed kleidiai_weight_header is present. It
                      overrides compute_forward to call KleidiAI's
                      run_kernel_ex, with hybrid SME+non-SME threading when
                      SMCUs are detected.
Observation:          supports_op (line 1683-1724) returns true only when:
                      (a) op is MUL_MAT or GET_ROWS;
                      (b) src0 type is Q4_0, Q8_0, or F32 (F16 for MUL_MAT
                      only);
                      (c) src0 buffer is the KleidiAI buffer type (i.e., the
                      KLAI weight header is present);
                      (d) src1 type is F32 (or I32 for GET_ROWS);
                      (e) ggml_n_dims(src0) == 2;
                      (f) the kernel chain returns slot_total > 0.

                      compute_forward (line 880-1334) implements the full MUL_MAT
                      pipeline: LHS packing (parallel by mr), ggml_barrier,
                      chunked RHS execution via slot.kernel->run_kernel_ex. The
                      chunk loop uses ggml_threadpool_chunk_set/_add (same
                      atomic-counter scheme as ARTX01-F06).

                      compute_forward_get_rows (line 1336-1426) uses
                      rhs_info.to_float(packed_base, row_idx, nc, out, ...)
                      to dequantize a single row on the fly — useful for
                      embedding-table lookup.

                      repack (line 1429-1610) is called from init_tensor. It
                      writes the KLAI header followed by 1 or 2 packed-RHS
                      slots. For Q8_0 RHS, it also re-quantizes from the
                      per-block Q8_0 format to a per-row scale + int8 layout
                      (line 1459-1503) — this is a precision change (per-row
                      scale vs per-block scale) that may differ slightly from
                      the original Q8_0 quantization.

Evidence:             kleidiai.cpp:1683-1724 (supports_op);
                      kleidiai.cpp:880-1334 (compute_forward MUL_MAT);
                      kleidiai.cpp:1336-1426 (compute_forward_get_rows);
                      kleidiai.cpp:1429-1610 (repack);
                      kleidiai.cpp:1459-1503 (Q8_0 re-quantization);
                      kleidiai.cpp:1241-1257 (run_chunk lambda);
                      kleidiai.cpp:1311-1326 (chunk loop with atomic counter).

Architectural Impact: KleidiAI is the most sophisticated extra_buffer_type in
                      the CPU backend. It demonstrates the plugin pattern's full
                      power: a third-party SDK can claim specific ops, manage
                      its own weight layout, run its own threading model, and
                      still integrate with ggml's chunk-stealing parallelism.
                      This is the model for any vendor SDK integration in
                      GwenLand.

                      The Q8_0 re-quantization at line 1459-1503 is a subtle
                      precision change: KleidiAI's Q8_0 kernel expects per-row
                      scales, but ggml's Q8_0 format uses per-block scales. The
                      repack re-quantizes by computing a per-row max-abs and
                      re-scaling all blocks to that max. This may produce
                      slightly different results than the upstream Q8_0 vecdot
                      path, which preserves per-block scales.

Correctness Impact:   The Q8_0 re-quantization introduces ULP-level differences
                      vs the upstream Q8_0 vecdot. The MUL_MAT results are not
                      bit-identical between KleidiAI and upstream paths. This
                      is acceptable for inference but problematic for
                      differential testing.

Optimization Type:    SIMD (delegated to KleidiAI) + threading (hybrid SME+
                      non-SME via F12).

GwenLand Target:      glproc, GATE

Recommendation:       ADOPT the extra_buffer_type plugin pattern with weight-
                      header scheme for vendor SDK integration. MONITOR the Q8_0
                      re-quantization precision change — document it explicitly
                      and provide a "preserve per-block scale" mode for
                      differential testing.

Priority:             High
Difficulty:           L
Dependencies:         ARTX01-F04 (extra-buffer-type hook), ARTX05-F06 (SME
                      delegation), ARTX05-F12 (dual-slot layout)
Confidence:           High
```

### Finding ARTX05-F08

```
Finding ID:           ARTX05-F08
Category:             MISSING_FEATURE
Engine:               CPU
Component:            SVE+I8MM 2-row vecdot for Q4_0/Q4_1/Q8_0
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0, ggml_vec_dot_q4_1_q8_1,
                      ggml_vec_dot_q8_0_q8_0
Lines:                315-386 (Q4_0 I8MM 2-row, NEON only);
                      608-680 (Q4_1 I8MM 2-row, NEON only);
                      1168-1226 (Q8_0 I8MM 2-row, NEON only)
Summary:              Q4_0, Q4_1, and Q8_0 have an I8MM 2-row path that uses
                      NEON vmmlaq_s32 only. There is no SVE+I8MM 2-row variant
                      (svmmla_s32) for these dtypes, unlike Q4_K and Q6_K which
                      do have SVE+I8MM 2-row paths.
Observation:          The preprocessor structure for Q4_0 is:
                        #if defined(__ARM_FEATURE_MATMUL_INT8)
                          if (nrc == 2) { ... vmmlaq_s32 (NEON) ... return; }
                        #endif
                        #if defined(__ARM_FEATURE_SVE)
                          ... svdot_s32 (SVE 1-row) ...
                        #elif defined(__ARM_NEON)
                          ... baseline ...
                        #endif
                      The SVE branch (svdot_s32) handles nrc==1 only. There is
                      no `#if defined(__ARM_FEATURE_SVE) && defined
                      (__ARM_FEATURE_MATMUL_INT8)` block for Q4_0/Q4_1/Q8_0 that
                      would use svmmla_s32 for the 2-row tile.

                      Q4_K (line 2360-2568) and Q6_K (line 2984-3174) DO have
                      such a block: `#if defined(__ARM_FEATURE_SVE) && defined
                      (__ARM_FEATURE_MATMUL_INT8)` with svmmla_s32 calls (e.g.,
                      quants.c:2443 for Q4_K, quants.c:3148-3149 for Q6_K).

                      On 128-bit SVE hardware, vmmlaq_s32 (NEON) and svmmla_s32
                      (SVE 128-bit) have the same per-instruction throughput, so
                      the omission is not a regression. On 256-bit SVE hardware
                      (e.g., future Neoverse V3 with 256-bit SVE), svmmla_s32
                      would process 2× more elements per instruction than
                      vmmlaq_s32 (which is fixed at 128-bit NEON width). On
                      512-bit SVE, the gap is 4×.

Evidence:             quants.c:315-386 (Q4_0 I8MM 2-row, NEON vmmlaq_s32 at
                      line 374);
                      quants.c:391-525 (Q4_0 SVE 1-row, svdot_s32 at line 434);
                      quants.c:2360 (Q4_K SVE+I8MM 2-row block);
                      quants.c:2443 (svmmla_s32 call in Q4_K SVE+I8MM);
                      quants.c:2984 (Q6_K SVE+I8MM 2-row block);
                      quants.c:3148 (svmmla_s32 call in Q6_K SVE+I8MM).
                      grep svmmla_s32 in quants.c: 4 hits, all in Q4_K/Q6_K
                      SVE+I8MM blocks.

Architectural Impact: On 128-bit SVE hardware (current mainstream: Cortex-X2,
                  Neoverse V2, Apple M1-M4), this is a non-issue — vmmlaq_s32
                  and svmmla_s32 have identical throughput at 128-bit. On
                  256-bit+ SVE hardware (future), Q4_0/Q4_1/Q8_0 vecdot will
                  leave 2-4× throughput on the table.

Correctness Impact:   None. The NEON vmmlaq_s32 path is correct.

Optimization Type:    SIMD (missed SVE I8MM opportunity).

GwenLand Target:      glproc

Recommendation:       REJECT the omission. Write SVE+I8MM 2-row variants for
                      Q4_0/Q4_1/Q8_0 modeled on the Q4_K SVE+I8MM path. This is
                      high-effort (208 lines of dense SVE for Q4_K) but necessary
                      for 256-bit SVE hardware. Priority is Low for current
                      hardware, Medium for future hardware.

Priority:             Low
Difficulty:           L
Dependencies:         ARTX05-F01 (I8MM 2-row pattern), ARTX05-F02 (SVE VL
                      switch)
Confidence:           High
```

### Finding ARTX05-F09

```
Finding ID:           ARTX05-F09
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            SVE F16 widened accumulation pattern
Source File:          ggml/src/ggml-cpu/vec.h
Function:             ggml_vec_dot_f16_unroll (SVE branch)
Lines:                142-317 (full function); 152-209 (SVE branch)
Summary:              The SVE F16 vecdot uses 8 widened F32 accumulators
                      (sum_0_0_lo, sum_0_0_hi, sum_0_1_lo, sum_0_1_hi, sum_1_0_lo,
                      sum_1_0_hi, sum_1_1_lo, sum_1_1_hi) — 2 rows × 2 halves ×
                      lo/hi. Each pair is updated via ggml_sve_f16_fma_widened
                      (F05).
Observation:          The function computes 2 dot products in parallel (one per
                      row in x[0..1]) using GGML_VEC_DOT_UNROLL=2. For each row,
                      it maintains 4 F32 accumulators (lo/hi for each of 2
                      half-blocks). The 8-accumulator design provides ILP: each
                      accumulator is an independent dependency chain.

                      The loop (line 168-182) processes 2*ggml_f16_epr F16
                      elements per iteration (2 SVE registers worth). Each
                      iteration calls ggml_sve_f16_fma_widened 4 times (2 rows ×
                      2 halves).

                      The tail (line 184-201) handles leftovers via predicated
                      loads with svwhilelt_b16(np2, n) — a true VLA tail.

                      The final reduction (line 203-209) combines the 8
                      accumulators into 2 scalar floats via svadd_f32_x and
                      ggml_sve_sum_f32x2.

Evidence:             vec.h:142 (function signature);
                      vec.h:152 (#if defined(__ARM_FEATURE_SVE));
                      vec.h:159-166 (8 accumulators);
                      vec.h:168-182 (main loop with 4 ggml_sve_f16_fma_widened
                      calls per iter);
                      vec.h:184-201 (predicated tail);
                      vec.h:203-209 (reduction).

Architectural Impact: The 8-accumulator pattern is the SVE equivalent of the
                      8-accumulator F32 vecdot in vec.cpp:27-34. It provides ILP
                      and pairs naturally with the lo/hi split in
                      ggml_sve_f16_fma_widened. The predicated tail via
                      svwhilelt_b16 is the correct VLA pattern (contrast with
                      the integer vecdots' assert(false) on non-standard VLs —
                      F02).

Correctness Impact:   None. The 8 accumulators are reduced in a deterministic
                      order, so the result is bit-stable across runs (unlike the
                      chunked parallel matmul which has ULP non-determinism).

Optimization Type:    SIMD (8-accumulator VLA SVE F16 dot).

GwenLand Target:      glproc

Recommendation:       ADOPT. The 8-accumulator + predicated-tail pattern is the
                      model for SVE F16 dot in glproc.

Priority:             Medium
Difficulty:           M
Dependencies:         ARTX05-F05 (ggml_sve_f16_fma_widened)
Confidence:           High
```

### Finding ARTX05-F10

```
Finding ID:           ARTX05-F10
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            Inline-assembly batched GEMV/GEMM in repack.cpp
Source File:          ggml/src/ggml-cpu/arch/arm/repack.cpp
Function:             ggml_gemv_q4_0_8x8_q8_0, ggml_gemm_q4_0_8x8_q8_0, and 20
                      sibling GEMV/GEMM entry points
Lines:                339-428 (Q4_0 8x8 GEMV, 90 lines asm);
                      1826-2305 (Q4_0 4x4 GEMM, 480 lines asm);
                      2728-3160 (Q4_0 8x8 GEMM, 432 lines asm);
                      and 19 more entry points (see §3)
Summary:              22 batched GEMV/GEMM kernels are hand-written inline
                      assembly using .inst 0x… encodings for sdot/smmla/sdot-
                      lane. The SVE variants are gated on
                      ggml_cpu_get_sve_cnt() == QK8_0 (i.e., 128-bit SVE only);
                      non-128-bit SVE falls through to _generic.
Observation:          The inline asm blocks use raw .inst encodings like:
                        .inst 0x4f9fe18a  // sdot v10.4s, v12.16b, v31.4b[0]
                        .inst 0x451f9872  // smmla z18.s, z3.b, z31.b
                      The comment after each .inst documents the instruction,
                      but the encoding is what the assembler sees. This bypasses
                      the compiler's intrinsic-availability window — useful when
                      the target compiler doesn't yet support the intrinsic
                      (e.g., older clang versions).

                      The SVE GEMV/GEMM kernels are gated on
                      `if (ggml_cpu_get_sve_cnt() == QK8_0)` (repack.cpp:360,
                      2750). QK8_0 is 32 (block size), but
                      ggml_cpu_get_sve_cnt() returns bytes-per-SVE-register,
                      which is 16 for 128-bit SVE. So the gate is `16 == 16` —
                      true only for 128-bit SVE. On 256-bit SVE (cnt=32) or
                      512-bit SVE (cnt=64), the gate fails and the function
                      falls through to _generic.

                      The DOTPROD GEMV/GEMM kernels (e.g., Q4_0 4x4 at
                      repack.cpp:1826) are gated on __ARM_FEATURE_DOTPROD
                      and __ARM_FEATURE_MATMUL_INT8 (for 4x8 and 8x8 variants).
                      They do not check SVE cnt — they use NEON registers
                      (v0-v31) and NEON sdot instructions.

Evidence:             repack.cpp:339-428 (Q4_0 8x8 GEMV);
                      repack.cpp:360 (gate: ggml_cpu_get_sve_cnt() == QK8_0);
                      repack.cpp:365-422 (inline asm block with .inst encodings);
                      repack.cpp:1826-2305 (Q4_0 4x4 GEMM);
                      repack.cpp:1846 (gate: __ARM_FEATURE_DOTPROD);
                      repack.cpp:1905-1952 (.inst 0x4f9fe18a sdot encodings);
                      repack.cpp:2728-3160 (Q4_0 8x8 GEMM);
                      repack.cpp:2750 (gate: ggml_cpu_get_sve_cnt() == QK8_0);
                      repack.cpp:2817-2822 (.inst 0x451f9872 smmla encodings).

Architectural Impact: The hand-written asm achieves maximum throughput (hand-
                      pipelined loads + arithmetic) at the cost of maintainability.
                      Each kernel is 90-480 lines of asm with manual register
                      allocation. Bugs are silent and hard to debug. Porting to a
                      new dtype requires writing a new asm block.

                      The 128-bit-SVE-only gate (repack.cpp:360, 2750) means the
                      SVE GEMV/GEMM kernels are unreachable on 256-bit+ SVE
                      hardware. On such hardware, the function falls through to
                      _generic — a scalar reference. This is a regression: the
                      SVE asm would still work (SVE is VLA), but the gate
                      artificially restricts it.

Correctness Impact:   None. The asm is correct on supported configurations. The
                      _generic fallback is correct on unsupported configurations.

Optimization Type:    SIMD (hand-pipelined inline asm tiles) + tiling (4x4/4x8/
                      8x8 column tiles).

GwenLand Target:      glproc

Recommendation:       MONITOR. The hand-written asm is a maintenance liability
                      but delivers maximum throughput on supported hardware.
                      Revisit when compiler SVE intrinsics are universally
                      available (clang 14+, gcc 12+). Replace the asm with
                      intrinsic-based C++ to improve maintainability. Also
                      relax the 128-bit-SVE-only gate to allow 256/512-bit SVE
                      (the asm is VLA-compatible).

Priority:             Low
Difficulty:           L
Dependencies:         ARTX05-F03 (SVE VL discovery)
Confidence:           High
```

### Finding ARTX05-F11

```
Finding ID:           ARTX05-F11
Category:             CORRECTNESS_SHORTCUT
Engine:               CPU
Component:            has_sme2 field dead on Linux; no GGML_USE_SME2 macro
Source File:          ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
Function:             aarch64_features constructor, ggml_backend_cpu_aarch64_score
Lines:                31 (has_sme2 field); 33-66 (constructor); 69-99 (score)
Summary:              The has_sme2 field is populated on Apple (sysctl at line
                      60-62) but never on Linux (no HWCAP2_SME2 read). The score
                      function checks GGML_USE_SME but has no GGML_USE_SME2
                      branch — the has_sme2 field is never consulted.
Observation:          cpu-feats.cpp line 11-21 defines fallbacks for HWCAP2_SVE2,
                      HWCAP2_I8MM, HWCAP2_SME — but NOT for HWCAP2_SME2. The
                      Linux constructor (line 34-43) sets has_sme = !!(hwcap2 &
                      HWCAP2_SME) but does not set has_sme2. The Apple
                      constructor (line 56-62) sets has_sme and has_sme2 via
                      sysctl. The score function (line 69-99) checks
                      GGML_USE_DOTPROD, GGML_USE_FP16_VECTOR_ARITHMETIC,
                      GGML_USE_SVE, GGML_USE_MATMUL_INT8, GGML_USE_SVE2,
                      GGML_USE_SME — but NOT GGML_USE_SME2 (no such macro exists
                      in the codebase).

                      The has_sme2 field is therefore dead on Linux and dead in
                      the score function on all platforms. The KleidiAI
                      integration (kleidiai.cpp:265) reads HWCAP2_SME2 directly
                      on Linux (with its own #define HWCAP2_SME2 (1UL << 37) at
                      line 30), bypassing cpu-feats.cpp.

                      The likely reason: SME2 was added to the Arm ARM after
                      SME, and at the time of writing no upstream code path
                      uses SME2 specifically (KleidiAI's SME2 kernels are
                      selected at the KleidiAI library level, not via
                      GGML_USE_SME2). But the dead has_sme2 field is a latent
                      bug: a future maintainer reading it would assume it's
                      populated on all platforms.

Evidence:             cpu-feats.cpp:11-21 (HWCAP2_* fallbacks — no SME2);
                      cpu-feats.cpp:31 (has_sme2 field declaration);
                      cpu-feats.cpp:34-43 (Linux constructor — no has_sme2
                      assignment);
                      cpu-feats.cpp:60-62 (Apple constructor — has_sme2 via
                      sysctl);
                      cpu-feats.cpp:69-99 (score function — no GGML_USE_SME2
                      check);
                      kleidiai.cpp:30 (#define HWCAP2_SME2 (1UL << 37) —
                      KleidiAI bypasses cpu-feats.cpp);
                      kleidiai.cpp:265 (KleidiAI reads HWCAP2_SME2 directly).

Architectural Impact: Latent bug for future SME2-specific features. If GwenLand
                      adds a GGML_USE_SME2 macro and an SME2-only kernel, the
                      score function would need updating and the has_sme2 field
                      would need Linux population. As-is, the field is a
                      misleading API surface.

Correctness Impact:   None today. SME2 hardware (which is also SME hardware) is
                      correctly detected via has_sme and routed to KleidiAI's
                      SME2 kernels via KleidiAI's own HWCAP2_SME2 read. The dead
                      field is a maintenance hazard, not a correctness bug.

Optimization Type:    None (detection bug).

GwenLand Target:      glproc

Recommendation:       MONITOR. Add HWCAP2_SME2 fallback to cpu-feats.cpp, set
                      has_sme2 on Linux, and add a GGML_USE_SME2 score branch.
                      Even if no current code uses SME2 directly, the field
                      should not lie. Priority is Low because KleidiAI bypasses
                      the issue.

Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX05-F12

```
Finding ID:           ARTX05-F12
Category:             ADOPT
Engine:               CPU
Component:            KleidiAI weight-header + dual-slot layout
Source File:          ggml/src/ggml-cpu/kleidiai/kleidiai.cpp
Function:             kleidiai_weight_header (struct), kleidiai_collect_kernel_chain,
                      compute_forward (hybrid SME+non-SME threading)
Lines:                57-60 (constants); 62-70 (context struct);
                      392-419 (weight header struct + validators);
                      453-484 (kernel chain collection);
                      1100-1184 (hybrid thread assignment);
                      1241-1331 (chunk loop with run_kernel_ex)
Summary:              KleidiAI stores pre-packed RHS weights behind a
                      kleidiai_weight_header (magic='KLAI', version=1,
                      slot_count, offsets[2], sizes[2]). Up to 2 slots are
                      populated: SME primary + non-SME fallback when SMCUs are
                      detected and nth_total > sme_thread_cap. Threads are
                      assigned to slots based on sme_thread_cap; each slot runs
                      its own kernel via run_kernel_ex.
Observation:          The weight header (line 392-398) is:
                        struct kleidiai_weight_header {
                          uint32_t magic;       // 0x4b4c4149 = "KLAI"
                          uint16_t version;     // 1
                          uint16_t slot_count;  // 1 or 2
                          uint64_t offsets[GGML_KLEIDIAI_MAX_KERNEL_SLOTS]; // 2
                          uint64_t sizes  [GGML_KLEIDIAI_MAX_KERNEL_SLOTS]; // 2
                        };
                      GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2 (line 57). The header
                      is written at repack time (line 1439-1441) and validated
                      at runtime via kleidiai_is_weight_header_valid (line
                      408-419).

                      The kernel chain (kleidiai_collect_kernel_chain_common,
                      line 453-484) collects up to 2 kernels: the primary
                      (selected by feature mask) and a fallback (selected by
                      feature mask minus SME/SME2). The fallback is included
                      only if it has the same lhs_type, rhs_type, and op_type
                      as the primary, and only if the primary is an SME-family
                      kernel.

                      The hybrid thread assignment (line 1100-1184) computes:
                        sme_cap = min(sme_thread_cap, nth_total);
                        runtime[sme_slot].assigned_threads = sme_cap;
                        threads_remaining = nth_total - sme_cap;
                        // distribute remaining threads across fallback slots
                      Each thread then picks its slot by ith_total ∈
                      [thread_begin, thread_end).

                      The chunk loop (line 1311-1326) uses
                      ggml_threadpool_chunk_add (atomic counter, same as
                      ARTX01-F06) to dynamically assign chunks within each
                      slot. Both slots run concurrently: SME threads process
                      chunks via the SME kernel, non-SME threads process
                      chunks via the fallback kernel.

Evidence:             kleidiai.cpp:57 (GGML_KLEIDIAI_MAX_KERNEL_SLOTS = 2);
                      kleidiai.cpp:58-60 (PACK_MAGIC, PACK_VERSION,
                      PACK_ALIGN);
                      kleidiai.cpp:392-398 (header struct);
                      kleidiai.cpp:408-419 (validator);
                      kleidiai.cpp:453-484 (kernel chain collection);
                      kleidiai.cpp:1100-1184 (hybrid thread assignment);
                      kleidiai.cpp:1241-1257 (run_chunk lambda);
                      kleidiai.cpp:1311-1326 (chunk loop);
                      kleidiai.cpp:282-300 (SME thread cap population).

Architectural Impact: The dual-slot scheme is the only place in the CPU backend
                      where two different kernel implementations run concurrently
                      on the same MUL_MAT op. It maximizes throughput on partial-
                      SME hardware (e.g., Apple M4 with 2 SMCUs / 10 P-cores).
                      Without it, either all threads would use SME (serializing
                      on the 2 SMCUs) or no threads would use SME (wasting the
                      2 SMCUs).

                      The weight-header scheme is portable: magic + version
                      allows the packed layout to be upgraded in-place and
                      validated at runtime. This is a clean migration story for
                      KleidiAI's evolving pack format.

Correctness Impact:   The two slots produce slightly different results due to
                      different accumulation orders (SME: float outer-product;
                      I8MM: int32 then convert to float). A MUL_MAT with
                      nth_total > sme_thread_cap will produce ULP-level non-
                      determinism across runs depending on which slot processed
                      which chunks. Acceptable for inference; problematic for
                      differential testing.

Optimization Type:    Threading (hybrid SME+non-SME) + persistent threads (per-
                      slot thread assignment for the duration of the MUL_MAT).

GwenLand Target:      glproc, GATE

Recommendation:       ADOPT. The dual-slot scheme with weight header is the
                      model for integrating a vendor SDK that has a partial-
                      hardware feature (like SME on Apple M4). Replicate the
                      header struct, the kernel chain collection, and the
                      hybrid thread assignment.

Priority:             High
Difficulty:           L
Dependencies:         ARTX01-F04 (extra-buffer-type hook), ARTX05-F06 (SME
                      delegation), ARTX05-F07 (KleidiAI wrapper)
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the I8MM 2-row path produces bit-identical results to
  the 1-row baseline path for the same input. The arithmetic is
  equivalent (same total sum) but the per-lane grouping differs (see
  §11.1). Static analysis confirms the reduction at lines 378-382 is
  correct, but ULP-level differences from different FMADD ordering
  cannot be ruled out without execution. (Same as ARTX01-U2.)

* **U2**. Whether the KleidiAI Q8_0 re-quantization (kleidiai.cpp:1459-
  1503) produces measurably different MUL_MAT results than the
  upstream Q8_0 vecdot path. The re-quantization changes per-block
  scales to per-row scales, which is a precision change. Requires
  differential execution to quantify.

* **U3**. Whether the SVE 256/512-bit cases in the integer vecdots
  (quants.c:443-521 for Q4_0) actually produce correct results on
  256/512-bit SVE hardware. The predicate patterns (svptrue_pat_b8
  (SV_VL16) for 256-bit, svptrue_pat_b8(SV_VL32) for 512-bit) are
  hand-written and have not been validated by this audit on real
  256/512-bit SVE hardware. Requires hardware testing.

* **U4**. Whether the KleidiAI dual-slot hybrid threading actually
  outperforms single-slot on Apple M4 (2 SMCUs / 10 P-cores). The
  design is sound, but the overhead of two concurrent kernel
  implementations (different register pressure, different cache
  footprint) may negate the throughput gain. Requires profiling on
  M4 hardware.

* **U5**. Whether the inline-asm GEMV/GEMM kernels in repack.cpp
  outperform equivalent C++ intrinsic implementations on current
  compilers. The hand-pipelined schedules may be matched or exceeded
  by clang/gcc auto-scheduling on SVE intrinsics. Requires
  benchmarking.

* **U6**. Whether the `ggml_cpu_get_sve_cnt() == QK8_0` gate
  (repack.cpp:360, 2750) is intentional (the asm was authored for
  128-bit SVE only) or a bug (the asm is VLA-compatible and would
  work on 256/512-bit SVE). Requires reading the original PR or
  testing the asm on 256-bit SVE hardware.

* **U7**. Whether the DOTPROD path for NVFP4/TQ1_0/TQ2_0 is faster
  than the I8MM 2-row path would be on the same hardware. The
  NVFP4/TQ1_0/TQ2_0 vecdots do not have an I8MM 2-row variant, even
  on I8MM hardware. Requires benchmarking.

* **U8**. Whether the KleidiAI SME/SME2 kernels are correct on all
  supported hardware (Apple M4, future Neoverse V3). The kernels are
  a black box (third-party library); this audit cannot validate their
  correctness. Requires KleidiAI's own test suite.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_0_q8_0` (I8MM 2-row branch)  | 315–386       |
| R02       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_0_q8_0` (SVE branch)         | 391–525       |
| R03       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_1_q8_1` (I8MM 2-row branch)  | 608–680       |
| R04       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q8_0_q8_0` (I8MM 2-row branch)  | 1168–1226     |
| R05       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_K_q8_K` (SVE+I8MM 2-row)     | 2360–2568     |
| R06       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_K_q8_K` (NEON I8MM 2-row)    | 2569+         |
| R07       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q6_K_q8_K` (SVE+I8MM 2-row)     | 2984–3174     |
| R08       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q6_K_q8_K` (NEON I8MM 2-row)    | 3175+         |
| R09       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_nvfp4_q8_0` (DOTPROD branch)    | 840–856       |
| R10       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_tq1_0_q8_K` (DOTPROD)           | 1417–1471     |
| R11       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | SVE `default: assert(false)` (8 sites)        | 523, 1349, 1938, 2203, 2560, 2784, 3166, 3485 |
| R12       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `vmmlaq_s32` callsites                        | 374, 669, 1215 |
| R13       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `svmmla_s32` callsites                        | 2443, 3148, 3149 |
| R14       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `svdot_s32` callsites                         | 434, 435, 437, 438, 474, 476, 515, 517, 1265–1294 |
| R15       | `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`          | `ggml_backend_cpu_aarch64_score`              | 69–99         |
| R16       | `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`          | `aarch64_features` constructor                | 33–66         |
| R17       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_init_arm_arch_features`                 | 728–737       |
| R18       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_get_sve_cnt`                        | 3794–3800     |
| R19       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_cpu_has_sme`, `ggml_cpu_has_sme2`       | 3802–3816     |
| R20       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_vdotq_s32` (DOTPROD fallback macro)     | 307–321       |
| R21       | `ggml/src/ggml-cpu/vec.h`                           | `ggml_sve_f16_fma_widened`                    | 17–44         |
| R22       | `ggml/src/ggml-cpu/vec.h`                           | `ggml_vec_dot_f16_unroll` (SVE branch)        | 142–317       |
| R23       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_f32` (SVE branch)               | 11–110        |
| R24       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemv_q4_0_8x8_q8_0` (SVE inline asm)    | 339–428       |
| R25       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemm_q4_0_8x8_q8_0` (SVE+I8MM asm)      | 2728–3160     |
| R26       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemm_q4_0_4x4_q8_0` (DOTPROD asm)       | 1826–2305     |
| R27       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `decode_q_Kx8_6bit_scales` (K-quant helper)   | 29–48         |
| R28       | `ggml/src/ggml-cpu/kleidiai/kleidiai.h`             | `ggml_backend_cpu_kleidiai_buffer_type`       | 13            |
| R29       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `init_kleidiai_context`                       | 194–317       |
| R30       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `detect_num_smcus` (SMIDR_EL1 / Apple M4)     | 96–174        |
| R31       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `kleidiai_weight_header` struct               | 392–398       |
| R32       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `kleidiai_collect_kernel_chain_common`        | 453–484       |
| R33       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `tensor_traits::work_size`                    | 535–820       |
| R34       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `tensor_traits::compute_forward` (MUL_MAT)    | 880–1334      |
| R35       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `tensor_traits::compute_forward_get_rows`     | 1336–1426     |
| R36       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `tensor_traits::repack`                       | 1429–1610     |
| R37       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `extra_buffer_type::supports_op`              | 1683–1724     |
| R38       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | `extra_buffer_type::get_tensor_traits`        | 1726–1760     |
| R39       | `ggml/src/ggml-cpu/kleidiai/kleidiai.cpp`           | hybrid SME+non-SME thread assignment          | 1100–1184     |
| R40       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `gemm_gemv_kernels[]` (Q4_0 table)            | 314–703       |
| R41       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `gemm_gemv_kernels_q8[]` (Q8_0 table)         | 705–924       |
| R42       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `ggml_kleidiai_kernels_f32[]` (F32 table)     | 927–1031      |
| R43       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `ggml_kleidiai_select_kernels` (dispatch)     | 1040–1076     |
| R44       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `ggml_kleidiai_select_kernels_q4_0`           | 1078–1096     |
| R45       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `ggml_kleidiai_select_kernels_q8_0`           | 1098–1113     |
| R46       | `ggml/src/ggml-cpu/kleidiai/kernels.cpp`            | `ggml_kleidiai_select_kernels_f32`            | 1115–1131     |
| R47       | `ggml/src/ggml-cpu/kleidiai/kernels.h`              | `kernel_info`, `lhs_packing_info`, `rhs_packing_info`, `ggml_kleidiai_kernels` | 26–95 |
| R48       | `ggml/src/ggml-cpu/kleidiai/kernels.h`              | `cpu_feature` enum (DOTPROD/I8MM/SVE/SME/SME2) | 9–16         |
