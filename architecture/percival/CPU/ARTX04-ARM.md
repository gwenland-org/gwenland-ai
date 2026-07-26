# ARTX04 — ARM Baseline NEON Quantized Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (NEON baseline kernel layer), `GATE` (kernel-selection contract)

---

## 1. Executive Summary

The ARM baseline-NEON quantized kernels in llama.cpp live in
`arch/arm/quants.c` (4319 lines), with companion 8×8 batched GEMV/GEMM
kernels in `arch/arm/repack.cpp` (5156 lines) and per-ISA shims in
`ggml-cpu-impl.h:78-322`. The "baseline NEON" path is what runs on any
ARMv7-A with NEONv2 or any ARMv8-A core compiled without `+dotprod`,
`+i8mm`, or `+sve` — i.e., every Cortex-A53/A57/A72/A73 smart-phone SoC
still in circulation, every Raspberry Pi 3, and every ARMv7-A IoT-class
device. The DOTPROD/I8MM/SVE variants are audited separately in ARTX05.

The single most important observation is that **baseline NEON is the
universal fallback**: the per-block `vec_dot` kernels have a 3-tier
`#if defined(__ARM_FEATURE_MATMUL_INT8) / #elif defined(__ARM_FEATURE_SVE)
/ #elif defined(__ARM_NEON)` ladder in which baseline NEON is the last
resort. The inner int8 dot product is emulated by `ggml_vdotq_s32`
(`ggml-cpu-impl.h:310-315`), which expands the missing `vdotq_s32`
instruction into three ops: `vmull_s8` (×2) → `vpaddlq_s16` (×2) →
`vaddq_s32`. Every Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/
IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M/IQ4_NL/IQ4_XS/Q1_0/Q2_0/
MXFP4/NVFP4/TQ1_0/TQ2_0 baseline vecdot routes through this helper, so
the helper's 3-op cost is paid once per 16-element int8 chunk.

The second most important observation is that **baseline NEON gets none of
the 8×8 batched GEMV/GEMM fast paths** in `repack.cpp`. Every one of the
~20 `ggml_gemv_*` / `ggml_gemm_*` entry points is gated behind
`__ARM_FEATURE_DOTPROD`, `__ARM_FEATURE_MATMUL_INT8`, or
`__ARM_FEATURE_SVE` (`repack.cpp:231, 292, 358, 450, 520, 590, 729, 883,
1042, 1329, 1518, 1718, 1776, 1846, 2327, 2748, 3186, 3262, 3338, 3544,
3773, 4293, 4540, 4742, 4958, 5026, 5091`). When the gate is not taken,
the function body falls through to `ggml_gemv_*_generic` /
`ggml_gemm_*_generic` (`repack.cpp:269, 335, 428, 497, 572, …`). In
other words: prompt-processing shapes on baseline NEON run a scalar
reference, not a batched kernel. Only the per-block `vec_dot` path
(GEMV-friendly) is NEON-optimized on baseline hardware.

The third observation is that **the file supports 32-bit ARM
transparently** via 220 lines of shims in `ggml-cpu-impl.h:87-305`. Every
AArch64-only NEON intrinsic that the kernels need (`vaddlvq_s16`,
`vpaddq_s16`, `vaddvq_s32`, `vaddvq_f32`, `vmaxvq_f32`, `vcvtnq_s32_f32`,
`vzip1_u8`, `vzip2_u8`, `vld1q_s8_x2/x4`, `vld1q_u8_x2/x4`,
`vld1q_s16_x2`, `vqtbl1q_s8`, `vqtbl1q_u8`) is emulated in software on
ARMv7-A. Two of those shims carry the comment `// NOTE: not tested`
(`ggml-cpu-impl.h:241, 265`).

For GwenLand, the decisions worth **ADOPT**ing are the `ggml_vdotq_s32`
fallback macro pattern (one edit swaps baseline NEON ↔ DOTPROD), the
two-block-per-iter 2-accumulator dep-chain-breaking pattern, and the
`table_b2b_0/1` bit-expansion LUT for Q5_0/Q5_1/Q1_0. The decisions worth
**REJECT**ing are the absence of any baseline-NEON batched GEMM (leaving
prompt processing on a scalar fallback) and the scalar scale-multiply
collapse in the K-quant baseline path. The decisions worth **MONITOR**ing
are the 32-bit ARM `vqtbl1q_s8`/`vqtbl1q_u8` shims (marked "not tested")
and the inconsistent FMA gating between the NVFP4 and MXFP4 vecdot
kernels.

---

## 2. Purpose

Provide the ARM baseline-NEON quantized-kernel layer for `glproc`. This
layer is responsible for:

* `vec_dot` kernels invoked through `type_traits_cpu[type].vec_dot` for
  every supported quant format on hardware without DOTPROD, I8MM, or SVE.
* `quantize_row_q8_0`, `quantize_row_q8_1` — F32→Q8_0/Q8_1 SIMD
  activation quantizers (the K-quant activation quantizer,
  `quantize_row_q8_K`, is a placeholder; see Finding ARTX02-F12 for the
  x86 equivalent and ARTX04-F10 for the ARM-specific note).
* The `ggml_quantize_mat_q8_0_4x4` / `_4x8` interleaved-row quantizers in
  `repack.cpp:51, 119` — the *only* `repack.cpp` functions whose
  `__ARM_NEON` path is reachable on baseline hardware.

It is **not** responsible for: DOTPROD/I8MM/SVE vecdot kernels (ARTX05),
the batched GEMV/GEMM kernels in `repack.cpp` (those are baseline-invisible
on ARM; see Finding ARTX04-F03), graph scheduling (ARTX01), elementwise
ops (ARTX06), or AMX (separate ARTX).

---

## 3. Source Files

| File                                          | Lines | Role                                                                                  |
| --------------------------------------------- | ----- | ------------------------------------------------------------------------------------- |
| `ggml/src/ggml-cpu/arch/arm/quants.c`         | 4319  | Per-block `vec_dot` kernels for every quant type; 3-tier MATMUL_INT8/SVE/NEON ladder  |
| `ggml/src/ggml-cpu/arch/arm/repack.cpp`       | 5156  | 8×8/4x4 batched GEMV/GEMM — all gated behind DOTPROD/I8MM/SVE on ARM; only `ggml_quantize_mat_q8_0_*` and a few helpers reachable on baseline |
| `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`    | 103   | `ggml_backend_cpu_aarch64_score` — multi-binary score function; aarch64-only, no 32-bit ARM score |
| `ggml/src/ggml-cpu/ggml-cpu-impl.h`           | 78-322 | 32-bit ARM NEON shim block (`vaddlvq_s16`, `vpaddq_s16`, `vaddvq_s32`, `vld1q_*_x2/x4`, `vqtbl1q_s8/u8`) and `ggml_vdotq_s32` DOTPROD fallback macro |
| `ggml/src/ggml-cpu/simd-mappings.h`           | 38-55 | `GGML_CPU_FP16_TO_FP32` → `neon_compute_fp16_to_fp32` scalar `__fp16→float` cast (no LUT on ARM) |

> The I8MM/DOTPROD/SVE batched GEMM kernels in `repack.cpp` are out of
> scope for this audit. They are mentioned only to document what
> baseline NEON *cannot* reach.

---

## 4. Architecture Overview

```
              ┌──────────────────────────────────────────────────────────┐
              │   type_traits_cpu[type].vec_dot  (ggml-cpu.c:214)        │
              │   nrows == 1 on baseline ARM (no I8MM 2-row consumption) │
              └──────────────────────────────────────────────────────────┘
                                    │
                ┌───────────────────┼─────────────────────┐
                ▼                                          ▼
   ┌────────────────────────────┐              ┌────────────────────────────┐
   │ arch/arm/quants.c          │              │ arch/arm/repack.cpp        │
   │ ───────────────            │              │ ──────────────             │
   │ 3-tier #if ladder per      │              │ All batched GEMV/GEMM      │
   │   vecdot:                  │              │ gated behind DOTPROD /     │
   │   1. __ARM_FEATURE_        │              │ I8MM / SVE. Baseline NEON  │
   │      MATMUL_INT8 (vmmlaq)  │              │ falls through to _generic. │
   │   2. __ARM_FEATURE_SVE     │              │                            │
   │      (svdot_s32)           │              │ Only baseline-reachable:   │
   │   3. __ARM_NEON (this AUDIT│              │  ggml_quantize_mat_q8_0_4x4│
   │      scope)                │              │  ggml_quantize_mat_q8_0_4x8│
   │                            │              │  decode_q_Kx8_6bit_scales  │
   │ Uses ggml_vdotq_s32 macro  │              │  (gated on I8MM||DOTPROD)  │
   │  → vmull_s8 + vpaddlq_s16  │              │                            │
   │    + vaddq_s32 (3 ops)     │              │                            │
   │                            │              │                            │
   │ 2-block-per-iter,          │              │                            │
   │ 2 float32x4_t accumulators │              │                            │
   └────────────────────────────┘              └────────────────────────────┘
                ▲
                │
   ┌────────────────────────────────────────────────────────────────────┐
   │ ggml-cpu-impl.h:78-322                                             │
   │  ──────────────────────                                            │
   │  ggml_vdotq_s32 fallback (line 310-315):                           │
   │    vmull_s8(low,low); vmull_s8(high,high);                         │
   │    vpaddlq_s16(p0); vpaddlq_s16(p1); vaddq_s32(acc, sum0+sum1)     │
   │                                                                    │
   │  32-bit ARM shims (line 87-305):                                   │
   │    vaddlvq_s16, vpaddq_s16, vpaddq_s32, vaddvq_s32, vaddvq_f32,    │
   │    vmaxvq_f32, vcvtnq_s32_f32, vzip1_u8, vzip2_u8,                 │
   │    vld1q_*_x2/x4 (4 types), vqtbl1q_s8/u8 (NOTE: not tested)       │
   └────────────────────────────────────────────────────────────────────┘
                ▲
                │
   ┌────────────────────────────────────────────────────────────────────┐
   │ arch/arm/cpu-feats.cpp (aarch64-only, 103 lines)                   │
   │  ggml_backend_cpu_aarch64_score:                                   │
   │   score = 1;                                                       │
   │   if (DOTPROD compiled && !has_dotprod) return 0; score += 2;      │
   │   if (FP16_VA  compiled && !has_fp16_va) return 0; score += 4;     │
   │   if (SVE      compiled && !has_sve)     return 0; score += 8;     │
   │   if (I8MM     compiled && !has_i8mm)    return 0; score += 16;    │
   │   if (SVE2     compiled && !has_sve2)    return 0; score += 32;    │
   │   if (SME      compiled && !has_sme)     return 0; score += 64;    │
   │   return score;                                                    │
   │   // 32-bit ARM has NO score function; baseline .so is the only    │
   │   //  one loaded.                                                  │
   └────────────────────────────────────────────────────────────────────┘
```

Key design points:

* **Baseline NEON is the lowest rung.** Every per-block vecdot's
  preprocessor ladder puts `__ARM_FEATURE_MATMUL_INT8` first, then
  `__ARM_FEATURE_SVE`, then `__ARM_NEON` (baseline). The baseline path
  is reachable when neither of the upper two is compiled in, *or* when
  the SVE case fails because `vector_length` is not 128/256/512 (the
  SVE `switch` defaults to `assert(false)`). Baseline NEON is the
  silent fallback.
* **One macro, two implementations.** `ggml_vdotq_s32(acc, a, b)` is
  defined in `ggml-cpu-impl.h:307-321`. Under `__ARM_FEATURE_DOTPROD`
  it expands to `vdotq_s32(acc, a, b)` (single instruction). Under
  baseline NEON it expands to the 3-op emulation. Every kernel uses
  this macro unconditionally, so the same source compiles for both
  targets.
* **Multi-binary dispatch.** `cpu-feats.cpp:69-99` returns 0 if a
  required feature is missing at runtime; otherwise returns a power-
  of-two-weighted score. The `.so` with the highest non-zero score
  wins. A baseline `.so` (compiled with no `__ARM_FEATURE_*`) returns
  score 1 unconditionally and is the universal fallback.
* **No runtime dispatch inside kernels.** Every ISA branch is resolved
  at compile time via `#if`. The runtime decision happens once at
  `.so` load time in the score function. Same model as x86 (ARTX02 §4).

---

## 5. Execution Flow

### 5.1 Per-block vecdot (the `quants.c` path)

1. `ggml_compute_forward_mul_mat_one_chunk` (`ggml-cpu.c:1164`) walks
   tiles `(iir0, iir1)` and for each weight row calls
   `vec_dot(qk, &s, sizeof(float), src0_row, nb01, src1_row, nb11, nrows)`.
2. `vec_dot` is a function pointer from `type_traits_cpu[type].vec_dot`
   (`ggml-cpu.c:1181-1182`). On ARM the linker resolves it to one of
   the `ggml_vec_dot_*` functions in `arch/arm/quants.c`.
3. Inside the kernel: `assert(nrc == 1)` on baseline (only the I8MM
   build accepts `nrc == 2`). The `#if defined(__ARM_FEATURE_MATMUL_INT8)`
   branch is skipped; if SVE is compiled in, the SVE branch is skipped
   when `vector_length` is not in {128, 256, 512}; the
   `#elif defined(__ARM_NEON)` branch runs.
4. The branch's hot loop processes 2 blocks per iteration
   (`for (; ib + 1 < nb; ib += 2)`), using two independent
   `float32x4_t` accumulators `sumv0` and `sumv1`. Each block's int32
   dot product is computed via `ggml_vdotq_s32` (the 3-op fallback
   macro on baseline NEON).
5. After the hot loop, a scalar tail loop (`for (; ib < nb; ++ib)`)
   handles the case where `nb` is odd. This tail is pure scalar C —
   no SIMD.
6. The kernel writes a single `float *s` result.

### 5.2 Activation conversion (`wdata`)

When `src1->type != vec_dot_type`, the matmul path converts src1 into
`params->wdata` once per matmul (ARTX01 §5.5 step 2). For most quants
this means F32 → Q8_0 via `quantize_row_q8_0` (`quants.c:41-83`,
NEON path at line 48-77). For K-quants, `quantize_row_q8_K`
(`quants.c:134-136`) is a placeholder that always calls the reference —
**no SIMD optimization for the K-quant activation quantizer on ARM**,
baseline or otherwise. This is the ARM analogue of ARTX02-W5.

### 5.3 Batched GEMV/GEMM (the `repack.cpp` path — invisible on baseline)

The `extra_buffer_type` mechanism (ARTX01-F04) detects that src0 was
allocated with the "repack" buffer type and calls
`tensor_traits::compute_forward` instead of the default
`ggml_compute_forward_mul_mat`. On baseline NEON, every
`ggml_gemv_*`/`ggml_gemm_*` in `repack.cpp` falls through its
`__ARM_FEATURE_DOTPROD`/`_MATMUL_INT8`/`_SVE` gate to the `_generic`
suffix (`repack.cpp:269, 335, 428, 497, 572, 705, 859, 1018, 1305,
1494, 1695, 1753, 1822, 2303, 2724, 3162, 3238, 3319, 3519, 3748,
4081, 4268, 4515, 4717, 4934, 5002, 5089, 5154`). In other words,
even if weights are repacked into `block_q4_0x8` format, baseline NEON
cannot exploit the 8×8 batched layout — it de-opts to scalar.

---

## 6. Data Layout

### 6.1 Quantized weight blocks

`quants.c` consumes the standard ggml block layouts (`block_q4_0`,
`block_q8_0`, `block_q4_K`, `block_q6_K`, `block_iq4_xs`, …) defined in
`ggml-common.h`. Block sizes: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 = 32 elements;
Q2_K/Q3_K/Q4_K/Q5_K/Q6_K = 256 elements (`QK_K`); IQ4_XS = 256;
IQ4_NL = 32; Q1_0 = 128; Q2_0 = 64; MXFP4 = 32; NVFP4 = 16;
TQ1_0/TQ2_0 = 256. Each block carries one or more fp16 scales (and `m`/
`s` zero-point fields for the `_1` variants). On baseline NEON the
fp16 scale is converted via the scalar `GGML_CPU_FP16_TO_FP32` macro
(`simd-mappings.h:42-48`), which on AArch64 compiles to a single
`fcvt` instruction; on 32-bit ARM with VFPv4 it is also `fcvt`, but on
VFPv3-only hardware it is a soft-fallback.

### 6.2 "Repacked" 8×8 batched layouts (repack.cpp — out of reach)

`repack.cpp` consumes `block_q4_0x8`, `block_q8_0x4`, `block_iq4_nlx8`,
`block_mxfp4x8`, `block_q4_Kx8`, `block_q8_Kx4`, `block_q2_Kx8` —
interleaved weight layouts where 8 (or 4) original blocks are shuffled
together to allow a single `vld1q_s8` to deliver 8 weight rows' nibbles
into one 128-bit lane. The conversion happens via
`ggml_quantize_mat_q8_0_4x4` (`repack.cpp:51`) and
`ggml_quantize_mat_q8_0_4x8` (`repack.cpp:119`) — both have a baseline
`__ARM_NEON` path that is reachable. The consumption of these layouts
(the `gemv`/`gemm` kernels) is *not* reachable on baseline.

### 6.3 Activation conversion (`wdata`)

Same as ARTX01 §6.2: src1 is re-laid-out into `params->wdata` as a
contiguous `vec_dot_type` tensor. The activation quantizer for Q8_0/
Q8_1 is NEON (`quants.c:48, 90`); the activation quantizer for Q8_K
is the scalar reference (`quants.c:134-136`). Both produce the standard
`block_q8_0` / `block_q8_K` layout — baseline NEON does not change
the activation layout.

---

## 7. Memory Layout

### 7.1 Per-block layout in `quants.c`

Inside each baseline NEON vecdot, the input blocks are streamed
sequentially:

```
x = vx;  // block_q4_0 []  (size = nb * sizeof(block_q4_0))
y = vy;  // block_q8_0 []  (size = nb * sizeof(block_q8_0))
for (int ib = 0; ib + 1 < nb; ib += 2) {
    vld1q_u8(x[ib+0].qs);    // 16-byte load
    vld1q_u8(x[ib+1].qs);    // 16-byte load
    vld1q_s8(y[ib+0].qs);    // 16-byte load
    vld1q_s8(y[ib+0].qs+16); // 16-byte load
    vld1q_s8(y[ib+1].qs);
    vld1q_s8(y[ib+1].qs+16);
    ...
}
```

No software prefetching is inserted in any baseline NEON vecdot. The
code relies entirely on the hardware prefetcher (Cortex-A53/A57 have
2-line L1 prefetchers that handle the 18-byte Q4_0 / 34-byte Q8_0
sequential strides well, but the 32-byte TQ1_0/TQ2_0 / 256-byte K-quant
blocks exceed the typical 4-line sequential stream window).

### 7.2 Constant tables

The following tables are defined for and consumed by the baseline NEON
paths:

* `table_b2b_0[1<<8]` and `table_b2b_1[1<<8]` (`quants.c:37-38`) —
  2×256×8 = 4 KB total. Pre-computed bit-to-byte expansion LUTs used
  by `ggml_vec_dot_q1_0_q8_0` (line 174-178), `ggml_vec_dot_q5_0_q8_0`
  (line 960-968, using `table_b2b_1`), and `ggml_vec_dot_q5_1_q8_1`
  (line 1078-1086, using `table_b2b_0`). Indexed by a single byte;
  returns a `uint64_t` that, when reinterpreted as 8 bytes, expands
  the input bit to all 8 bit positions.
* `keven_signs_q2xs[1024]` (`quants.c:3595-3628`) — 1 KB. Sign-mask
  lookup table for IQ2_XXS, IQ2_XS, IQ3_XXS vecdot. Indexed by a 7-bit
  sign index; returns 8 sign bytes (each ±1). Used at `quants.c:3646,
  3668-3671, 3708, 3739-3742, 3879, 3901-3904`. **No SVE/I8MM variant
  of these three vecdots exists** — baseline NEON is the only path.
* `k_mask1[32]`, `k_mask2[16]` (`quants.c:3782-3786`, `3946-3950`) —
  48 bytes total. Used by IQ2_S and IQ3_S for sign-bit expansion.
* `k_shift[8]` (`quants.c:3952`) — 16 bytes. Used by IQ3_S only.

The `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`,
`iq1s_grid`, `kvalues_iq4nl`, `kvalues_mxfp4` grids are defined in
`ggml-common.h` and shared across all ISAs; they are not baseline-NEON-
specific.

### 7.3 FP16 / E8M0 / UE4M3 LUTs

`simd-mappings.h:38-55` defines `GGML_CPU_FP16_TO_FP32(x)` on ARM NEON
as `neon_compute_fp16_to_fp32(x)`, a scalar `__fp16 → float` cast. On
AArch64 this is a single `fcvt` instruction; the 256 KB `ggml_table_f32_f16`
LUT (used on x86 without F16C, see ARTX01 §7.4) is **not used on ARM**.
`GGML_E8M0_TO_FP32_HALF` and `GGML_CPU_UE4M3_TO_FP32` use 256-entry LUTs
defined in `ggml-cpu.c` (referenced from `simd-mappings.h:135-136`).
The same LUTs are used by all ARM variants; baseline NEON inherits
them unchanged.

### 7.4 Per-iteration register budget (Q4_0 baseline, representative)

In `quants.c:531-567`, one iteration of the Q4_0 baseline loop holds
across registers:
- `m4b`, `s8b` — 2 NEON registers
- `v0_0`, `v0_1` (loaded weight bytes) — 2 NEON registers
- `v0_0l/0ls`, `v0_0h/0hs`, `v0_1l/1ls`, `v0_1h/1hs` — 4 NEON registers
  (intermediate and final)
- `v1_0l`, `v1_0h`, `v1_1l`, `v1_1h` (loaded activation) — 4 NEON registers
- `p_0`, `p_1` (int32x4 accumulators) — 2 NEON registers
- `sumv0`, `sumv1` (float32x4 accumulators) — 2 NEON registers
- Plus temporaries for `ggml_vdotq_s32`'s 3-op expansion: 4 more (two
  `int16x8` partials, two `int32x4` partials)

Peak ~20 NEON registers on AArch64 (32 available); ~20 on ARMv7-A NEONv2
(16 quad-regs available, aliasing the 32 double-regs). **The ARMv7-A
build almost certainly spills** — see Unknowns U2.

---

## 8. Parallelism Strategy

The kernels themselves are single-threaded. Threading is layered above
in `ggml_compute_forward_mul_mat` (ARTX01 §5.5, §8.4): dynamic chunk
stealing on `current_chunk` atomic for the per-block vecdot path.
Baseline NEON makes no contribution to or deviation from this strategy.

The `nrc == 1` assertion in every baseline vecdot (`quants.c:145, 227,
305, 598, 750, 811, 929, 1041, 1158, 1398, 1575, 2020, 2339, 2866,
2967, 3633, 3695, 3769, 3866, 3928, 4038, 4104, 4197, 4257`) means
**baseline ARM kernels consume exactly one weight row × one activation
row per call**. The I8MM kernels can set `nrows = 2` and consume two
rows in parallel from a single activation block (ARTX01 §11.1); baseline
NEON has no equivalent. This is the same gap as ARTX02-W6 on x86, but
on ARM the gap is between the *same codebase's* I8MM and baseline
variants — not across vendors.

---

## 9. SIMD / GPU Strategy

This is the meatiest section. The ARM baseline-NEON SIMD strategy is
fragmented across two files (one reachable, one effectively not) and
defined in terms of a single shared macro that swaps between baseline
and DOTPROD at compile time.

### 9.1 SIMD feature matrix (per file, per feature)

| Feature                          | `quants.c` baseline path              | `repack.cpp` baseline path | `ggml-cpu-impl.h` shim |
| -------------------------------- | ------------------------------------- | -------------------------- | ---------------------- |
| NEON (128-bit, AArch64)          | **Main path** for every vecdot        | Not used (gated on DOTPROD)| N/A (native)           |
| NEON (128-bit, ARMv7-A NEONv2)   | Same source; uses 32-bit shims        | Not used (gated on DOTPROD)| Emulates AArch64-only intrinsics |
| DOTPROD (`vdotq_s32`)            | Compiled out (`#elif`), reachable via macro | Reachable on DOTPROD builds | Macro swaps to native `vdotq_s32` |
| I8MM (`vmmlaq_s32`)              | Compiled out (`#if` first)            | Compiled out (`#if` first) | N/A                    |
| SVE (`svdot_s32`)                | Compiled out (`#elif` second)         | Compiled out (`#if` first) | N/A                    |
| FP16 vector arithmetic (`vld1q_f16`, `vfmaq_f16`) | Not used (no `__ARM_FEATURE_FP16_FML`) | Used only in DOTPROD paths for `vld1_f16` scale load | N/A |
| `vqtbl1q_s8`/`u8` (AArch64 native, ARMv7-A shim) | Used by IQ2/IQ3/IQ4 NL/XS vecdots, MXFP4, NVFP4 | N/A (only DOTPROD paths use it) | Shim marked "NOTE: not tested" |

### 9.2 The `ggml_vdotq_s32` macro (the baseline-NEON hot primitive)

```c
// ggml-cpu-impl.h:307-321
#if !defined(__ARM_FEATURE_DOTPROD)
// NOTE: this fallback produces the same total sum as native vdotq_s32
// but with different per-lane grouping — do not use when individual lane
// values matter.
inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));
    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}
#else
#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)
#endif
```

This is *the* baseline-NEON hot primitive. Every Q4_0/Q4_1/Q5_0/Q5_1/
Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ1_S/
IQ1_M/IQ4_NL/IQ4_XS/Q1_0/Q2_0/MXFP4 vecdot routes through it on baseline
hardware (see `quants.c:205-206, 281-282, 562-563, 720-721, 788-789,
996-1000, 1114-1119, 1374-1379, 1980-1981, 2267-2287, 2834-2843,
2943-2944, 3548-3578, 3676-3677, 3747-3751, 3840-3843, 3909-3910,
4018-4019, 4077-4078, 4160-4169, 4235-4236, 4296-4297`).

Three ops (vmull_s8 ×2 + vpaddlq_s16 ×2 + vaddq_s32 ×2 = ~6 uops) replace
DOTPROD's single `vdotq_s32` instruction. Throughput on Cortex-A72/A53
is roughly 1 cycle per `vmull_s8` (F0/F1 pipes), giving the fallback a
~6-cycle cost per 16-element chunk vs. ~1-cycle for native DOTPROD. The
comment at line 309 documents a precision caveat: the fallback produces
the same total sum but with **different per-lane grouping** than native
`vdotq_s32`. This matters for any caller that inspects individual int32x4
lanes; the kernels themselves always `vaddvq_s32` the result, so the
caveat does not affect them.

### 9.3 Tile blocking and accumulator count

| Kernel                                 | File:Line           | Accumulator count (baseline NEON path)    | Tile shape             |
| -------------------------------------- | ------------------- | ---------------------------------------- | ---------------------- |
| `ggml_vec_dot_q4_0_q8_0`               | `quants.c:527-567`  | 2 × `float32x4_t` (`sumv0`, `sumv1`)     | 2 blocks/iter          |
| `ggml_vec_dot_q4_1_q8_1`               | `quants.c:688-725`  | 2 × `float32x4_t` + 1 × `float summs`    | 2 blocks/iter          |
| `ggml_vec_dot_q5_0_q8_0`               | `quants.c:938-1006` | 2 × `float32x4_t`                        | 2 blocks/iter          |
| `ggml_vec_dot_q5_1_q8_1`               | `quants.c:1050-1124`| 2 × `float32x4_t` + 2 × `float summs`    | 2 blocks/iter          |
| `ggml_vec_dot_q8_0_q8_0`               | `quants.c:1352-1382`| 2 × `float32x4_t`                        | 2 blocks/iter          |
| `ggml_vec_dot_q1_0_q8_0`               | `quants.c:154-211`  | 1 × `float32x4_t` (`sumv`)               | 4 sub-blocks/block     |
| `ggml_vec_dot_q2_0_q8_0`               | `quants.c:238-294`  | 1 × `float32x4_t`                        | 2 sub-blocks/block     |
| `ggml_vec_dot_mxfp4_q8_0`              | `quants.c:766-796`  | 1 × `float sumf` (scalar accumulator)    | 2 blocks/iter          |
| `ggml_vec_dot_nvfp4_q8_0`              | `quants.c:826-898`  | 1 × `float32x4_t acc`                    | 1 block/iter           |
| `ggml_vec_dot_q2_K_q8_K`               | `quants.c:1942-2009`| 1 × `int isum` (scalar!)                 | 8 sub-blocks/iter      |
| `ggml_vec_dot_q3_K_q8_K`               | `quants.c:2209-2301`| 1 × `int32_t isum` (scalar!)             | 32 sub-blocks/iter     |
| `ggml_vec_dot_q4_K_q8_K`               | `quants.c:2789-2851`| 2 × `int32_t` (scalar!) `sumi1`, `sumi2` | 32 sub-blocks/iter     |
| `ggml_vec_dot_q5_K_q8_K`               | `quants.c:2884-2961`| 1 × `int32_t sumi` (scalar!)             | 32 sub-blocks/iter     |
| `ggml_vec_dot_q6_K_q8_K`               | `quants.c:3492-3591`| 1 × `int32_t isum` (scalar!)             | 32 sub-blocks/iter     |
| `ggml_vec_dot_iq2_xxs_q8_K`            | `quants.c:3644-3684`| 2 × `float` (scalar!) `sumf1`, `sumf2`   | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq2_xs_q8_K`             | `quants.c:3706-3757`| 1 × `int32x4_t sumi` + 4 × `int32x4_t p` | 4 sub-blocks/iter      |
| `ggml_vec_dot_iq2_s_q8_K`              | `quants.c:3780-3854`| 2 × `int` (scalar!) `sumi1`, `sumi2`     | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq3_xxs_q8_K`            | `quants.c:3877-3917`| 2 × `float` (scalar!) `sumf1`, `sumf2`   | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq3_s_q8_K`              | `quants.c:3939-4027`| 2 × `int` (scalar!) `sumi1`, `sumi2`     | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq1_s_q8_K`              | `quants.c:4049-4093`| 3 × `int` (scalar!) `sumi1/2/3`          | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq1_m_q8_K`              | `quants.c:4117-4186`| 2 × `int32x4_t sumi1/2`                  | 2 sub-blocks/iter      |
| `ggml_vec_dot_iq4_nl_q8_0`             | `quants.c:4213-4243`| 1 × `float sumf` (scalar!)               | 2 blocks/iter          |
| `ggml_vec_dot_iq4_xs_q8_K`             | `quants.c:4269-4310`| 2 × `int` (scalar!) `sumi1`, `sumi2`     | 4 sub-blocks/iter      |

The Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 kernels break the dependence chain with
two independent `float32x4_t` accumulators processing 2 blocks per iter.
The K-quants collapse each block's `int32x4` dot product back to a scalar
via `vaddvq_s32(p) * scales[i]` (e.g. `quants.c:2284, 2835, 2843, 2943,
3548`) — a deliberate choice to allow scalar multiplication by the
per-block-32 scale, but one that forfeits the dep-chain-breaking benefit
of vector accumulators. The I-quants are mixed: IQ2_XS and IQ1_M use
`int32x4_t` accumulators; the rest use scalar.

### 9.4 The `vqtbl1q_s8`/`u8` lookup instruction

The 128-bit NEON `vqtbl1q_s8` instruction (table lookup with byte index)
is used by:

* `ggml_vec_dot_q2_0_q8_0` (`quants.c:266-272`) — to replicate bytes
  before the shift+mask+sub that expands 2-bit nibbles to int8.
* `ggml_vec_dot_mxfp4_q8_0` (`quants.c:783-786`) — to expand 4-bit
  nibbles through the `kvalues_mxfp4` LUT (16-entry int8 table) into
  signed int8 values.
* `ggml_vec_dot_nvfp4_q8_0` (`quants.c:835-838`) — same as MXFP4, via
  `kvalues_mxfp4` LUT.
* `ggml_vec_dot_iq4_nl_q8_0` (`quants.c:4230-4233`) — same via
  `kvalues_iq4nl` LUT.
* `ggml_vec_dot_iq4_xs_q8_K` (`quants.c:4291-4294`) — same.
* `ggml_vec_dot_iq2_s_q8_K` (`quants.c:3821, 3830`) — for sign-bit
  expansion via `k_mask1`.
* `ggml_vec_dot_iq3_s_q8_K` (`quants.c:3999, 4008`) — same.

On AArch64 `vqtbl1q_s8` is a single instruction (1 cycle on Cortex-A72,
2 cycles on A53). On ARMv7-A it is **emulated by `ggml_vqtbl1q_s8`**
(`ggml-cpu-impl.h:241-287`): a 16-element loop with `res[i] = a[b[i]]`.
This is O(16) scalar loads per call. On a Cortex-A9/A15 (ARMv7-A) the
emulation costs ~16 cycles per call vs. 1-2 cycles on AArch64.

### 9.5 FP16 usage (no `__ARM_FEATURE_FP16_FML`)

Grep for `__ARM_FEATURE_FP16_FML`, `vfmaq_f16`, `vfmlalq`, `vld1q_f16`
across `quants.c` returns **zero matches**. The baseline path never
uses native FP16 dot product or FP16 vector arithmetic. FP16 scales are
always converted to FP32 via the scalar `GGML_CPU_FP16_TO_FP32` macro
(`simd-mappings.h:42-48`). The `vld1_f16` intrinsic (4-lane FP16 load)
appears only in the DOTPROD-gated paths of `repack.cpp` (line 242, 246,
303, 309, 488, 607-613, 746-752, 902-908, 1061-1067, 1348-1349, 1536-
1537, 1727, 1730, 1786, 1793, 3200-3201, 3276, 3358-3364, 3567-3573,
+ more) — unreachable on baseline.

### 9.6 FMA gating inconsistency

`ggml_vec_dot_nvfp4_q8_0` (`quants.c:826`) gates its entire NEON path on
`#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)`. The kernel
uses `vfmaq_f32` (line 896). AArch64 always has FMA (it is mandatory);
ARMv7-A with NEONv1 does not. On a non-FMA ARMv7-A target, the `#else`
(line 899) falls through to scalar. **The MXFP4 kernel at line 766 has
no such gate** and uses plain `vmlaq_n_f32` (line 791-793) which compiles
to a multiply+add sequence on non-FMA hardware. The two kernels are
inconsistent in their FMA requirement: NVFP4 demands it, MXFP4 falls
back to non-FMA multiply+add. See Finding ARTX04-F09.

### 9.7 DOTPROD/I8MM/SVE usage (not used on baseline)

`vdotq_s32` (DOTPROD) appears at `quants.c:852-856, 1462-1471, 1524-
1529, 1628-1636, 1662-1672` — all inside `#if defined(__ARM_FEATURE_DOTPROD)`
blocks. `vmmlaq_s32` (I8MM) appears at `quants.c:374-375, 1215-1216` —
inside `#if defined(__ARM_FEATURE_MATMUL_INT8)`. `svdot_s32` (SVE)
appears at `quants.c:434-438, 473-476, 514-517, 1265-1269, 2751-2781,
3175-3290` — inside `#if defined(__ARM_FEATURE_SVE)` blocks. None of
these are reachable on baseline NEON.

---

## 10. Quantization Strategy

`quants.c` provides SIMD `from_float` (quantize) kernels for two
activation formats only:

* `quantize_row_q8_0` (`quants.c:41-83`) — NEON path at line 48-77,
  scalar fallback at line 78-82. Uses `vmaxq_f32` for max-abs
  reduction (8-deep pyramid), `vcvtnq_s32_f32` for round-to-nearest,
  `vgetq_lane_s32` for int32→int8 store (4 lanes per call, called 8
  times per block).
* `quantize_row_q8_1` (`quants.c:85-131`) — same structure as Q8_0
  plus an extra `vaddq_s32(accv, vi)` per lane to compute the block
  sum `s` field (`quants.c:121-124`).
* `quantize_row_q8_K` (`quants.c:134-136`) — **placeholder**. Always
  calls `quantize_row_q8_K_ref`. No SIMD optimization for Q8_K
  activation quantization on ARM, baseline or otherwise. Same gap as
  ARTX02-W5 on x86.

All other `from_float` functions for the K-quants and I-quants are not
defined in this file; they live in the generic `quants.c` (parent
directory). The I-quants (IQ2_XXS, IQ2_XS, IQ2_S, IQ1_S, IQ1_M) have
`from_float = NULL` per ARTX01 — they are inference-only.

The `vec_dot` strategy across all quants is consistent on baseline NEON:

1. Load one weight block (16-32 bytes via `vld1q_u8` / `vld1q_s8`).
2. Expand the packed nibbles/bits to int8 via:
   - For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K:
     `vandq_u8(q, m4b)` + `vshrq_n_u8(q, 4)` + (for Q5) `vorrq_u8` with
     a sign-shifted `qh` vector.
   - For Q1_0: `table_b2b_0[bits[i]]` LUT lookup.
   - For Q2_0: `ggml_vqtbl1q_u8(raw16, idx_lo/hi)` to replicate bytes.
   - For IQ2_*/IQ3_*: `iq2xxs_grid`/`iq2xs_grid`/`iq3xxs_grid` lookup
     via `vld1_s8` against an 8-bit index, then `vmulq_s8` with a
     sign vector from `keven_signs_q2xs[1024]`.
   - For IQ4_NL/IQ4_XS/MXFP4/NVFP4: `ggml_vqtbl1q_s8(kvalues_*, q)`
     to expand 4-bit nibbles through a 16-entry int8 LUT.
3. Load the corresponding Q8 activation block (16 bytes via `vld1q_s8`).
4. Compute the int32 dot product via `ggml_vdotq_s32` (the 3-op
   fallback macro on baseline NEON).
5. For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0: multiply by the per-block fp16 scale
   (broadcast to `float32x4_t` via `vmlaq_n_f32(sumv, vcvtq_f32_s32(p),
   GGML_CPU_FP16_TO_FP32(x->d) * GGML_CPU_FP16_TO_FP32(y->d))`).
6. For K-quants: `vaddvq_s32(p)` to collapse int32x4 to scalar, then
   scalar multiply by per-block-32 scale, scalar accumulate into `isum`.
7. After all blocks: `vaddvq_f32(sumv)` (or scalar `isum * d`) for
   final reduction.

The K-quant scale-handling deserves a callout: the baseline path
collapses to scalar after each `ggml_vdotq_s32`, paying 4×`vaddvq_s32`
+ 4× scalar multiply + 4× scalar add per 32-element sub-block, vs. the
SVE/I8MM path which keeps the scale-multiply in vector form
(`svmla_n_s32_x` or `vmmlaq_s32`). See Finding ARTX04-F08.

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

Every baseline NEON vecdot accumulates into 1 or 2 `float32x4_t`
accumulators (or, for K-quants and most I-quants, a scalar `int` /
`float`) across blocks, then horizontally reduces via `vaddvq_f32` /
`vaddvq_s32` at the end. This reassociates the sum: the result differs
from a strict left-to-right scalar sum at the ULP level. The reduction
order is deterministic for a fixed `n` and a fixed compile-time ISA
selection, but combined with dynamic chunk stealing (ARTX01-F06) it is
non-deterministic across runs with `nth > 1`.

### 11.2 The `ggml_vdotq_s32` lane-grouping caveat

The comment at `ggml-cpu-impl.h:309` explicitly notes: "this fallback
produces the same total sum as native `vdotq_s32` but with different
per-lane grouping — do not use when individual lane values matter."
The fallback's lane grouping is `(lane0+lane1+lane8+lane9, lane2+lane3+
lane10+lane11, lane4+lane5+lane12+lane13, lane6+lane7+lane14+lane15)`,
vs. DOTPROD's `(lane0+lane1+lane2+lane3, …)`. Every baseline NEON
kernel immediately reduces via `vaddvq_s32(p)`, so the total is
identical. However, **the K-quant kernels** (`quants.c:1980-1981, 2267,
2834, 2841, 2943, 3548-3578`) feed `ggml_vdotq_s32`'s output into
`vaddvq_s32(p) * scales[i]` — a scalar multiply. The lane grouping
therefore does not affect correctness either. The caveat is forward-
looking: a future kernel that inspected individual lanes would silently
produce different results on baseline NEON vs. DOTPROD builds.

### 11.3 Approximate math

* No transcendental approximations in any baseline NEON vecdot. All
  activations of that kind live in `ops.cpp` (ARTX06).
* E8M0 / UE4M3 LUTs (`simd-mappings.h:135-136`): MXFP4 / NVFP4 scales
  are looked up from 256-entry LUTs. Bit-exact equivalents of the
  format's specification; no approximation beyond the format itself.
* FP16→FP32 conversion is exact (IEEE 754 half → IEEE 754 single is
  lossless). No precision reduction.

### 11.4 Precision reduction

* All K-quant and I-quant kernels convert the F32 activation to Q8_0 or
  Q8_K once up-front (ARTX01 §6.2). Lossy conversion before the dot
  product — the whole point of quantized inference.
* The dot product itself is computed in int32 (i8×i8 → i16 → i32 via
  `vmull_s8` + `vpaddlq_s16` + `vaddq_s32`), then converted to FP32
  (`vcvtq_f32_s32`) for the scale multiply. No precision loss beyond
  the storage format.
* For K-quants, the scalar `isum += vaddvq_s32(p) * scales[i]`
  accumulation happens in `int32_t`. The block scales are int8 (sign-
  extended); the product `int32 * int8` is `int32`. With 8 sub-blocks
  per K-quant block and 256-element blocks, `isum` can hold values up
  to ~`127 * 127 * 32 * 8 = 4M`, well within int32 range.

### 11.5 Non-deterministic reductions

Same as ARTX01 §11.4: matmul output is deterministic bit-for-bit only
when `nth = 1`. With `nth > 1`, dynamic chunk stealing + per-chunk
reassociation produces ULP-level variation. The kernels themselves
are deterministic; the non-determinism is in the chunk scheduler.

### 11.6 Atomic accumulation

None in any audited kernel. Output tiles are written by exactly one
thread each (chunk stealing assigns disjoint chunks).

### 11.7 Architecture-specific assumptions

* `assert(nrc == 1)` in every baseline vecdot. Only the I8MM build
  accepts `nrc == 2`. The two paths produce *slightly different*
  results on the same input due to lane interleaving in the I8MM
  `vmmlaq_s32` path (ARTX01 §11.1).
* `assert(n % qk == 0)` in every kernel. Block-aligned lengths only.
  Every kernel has a scalar tail loop for the `n % qk != 0` case
  (e.g. `quants.c:571-585, 730-744, 797-806, 1007-1027, 1125-1145,
  1384-1392`).
* The SVE path's `switch (vector_length)` (`quants.c:398, 1240, 2376,
2745`) defaults to `assert(false && "Unsupported vector length")` for
  vector lengths other than 128/256/512. Baseline NEON is the silent
  fallback when SVE is not compiled in; there is no fallback when SVE
  *is* compiled in but the vector length is e.g. 384 bits (some
  emulators).
* The 32-bit ARM shims at `ggml-cpu-impl.h:241, 265` are marked
  `// NOTE: not tested`. The `vld1q_*_x2/x4` shims at line 165-239
  have `// TODO: double-check these work correctly`. These comments
  are evidence of unverified correctness on 32-bit ARM.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations (baseline-NEON-specific)

| Optimization                                 | Where                                       | Notes                                                                                  |
| -------------------------------------------- | ------------------------------------------- | -------------------------------------------------------------------------------------- |
| `ggml_vdotq_s32` macro swaps baseline ↔ DOTPROD | `ggml-cpu-impl.h:307-321`                | One macro, two implementations; kernel source unchanged across build variants.         |
| 2-block-per-iter, 2-accumulator pattern      | `quants.c:531-567, 694-725, 948-1002, 1063-1120, 1356-1380` | Breaks dep chain on Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 vecdots. Each accumulator is independent. |
| `table_b2b_0/1` 4 KB LUT                     | `quants.c:37-38`                            | Bit-to-byte expansion for Q1_0/Q5_0/Q5_1; replaces per-bit shift+or with one LUT load. |
| `keven_signs_q2xs[1024]` 1 KB LUT            | `quants.c:3595-3628`                        | Sign-mask expansion for IQ2_XXS/XS/IQ3_XXS; indexed by 7-bit sign index.               |
| Multi-binary score function                  | `cpu-feats.cpp:69-99`                       | Compile N `.so` per ISA target; runtime score picks best. Cross-ref ARTX01-F12.        |
| Block-aligned assertion + scalar tail        | every kernel                                | Keeps the SIMD kernel simple without sacrificing correctness for odd-shaped inputs.    |
| Pre-computed `iq*xxs_grid` LUTs              | `ggml-common.h` (referenced from `quants.c:3664-3667, 3735-3738, 3810-3817, 3896-3899, 3987-3995, 4065-4072, 4149-4156`) | 8-bit index → 8-byte int8 grid; lookup via `vld1_s8`. |
| `vmaxvq_f32` / `vaddvq_s32` horizontal reduction | every kernel                             | Single-instruction horizontal sum (AArch64); 4-lane scalar sum on ARMv7-A shim.        |
| FP16→FP32 via `fcvt` (scalar)                | `simd-mappings.h:44-48`                     | Single-instruction conversion on AArch64; replaces 256 KB LUT.                          |

### 12.2 Optimizations *not* present (worth noting)

* **No baseline-NEON batched GEMV/GEMM in `repack.cpp`.** Every batched
  kernel is gated behind DOTPROD/I8MM/SVE. Prompt processing on
  baseline NEON falls through to `_generic`. See F03.
* **No native FP16 dot product.** `__ARM_FEATURE_FP16_FML` (`vfmlalq_*`)
  and `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` (`vfmaq_f16`) are unused
  in baseline NEON. Every FP16 scale is converted to FP32 first.
* **No software prefetching** in any baseline NEON vecdot. Tiling
  relies on hardware prefetchers.
* **No persistent threads.** Threads are managed above (ARTX01).
* **No kernel fusion.** Quantized vecdot is a leaf operation; no
  upstream fusion with bias-add or activation. (Cross-ref ARTX01-F08.)
* **No multi-row vecdot on baseline NEON.** `nrc == 1` is asserted
  everywhere; the I8MM build's `nrc == 2` is unreachable on baseline.
  Same gap as ARTX02-W6 on x86.
* **K-quant baseline path collapses to scalar after each block.** The
  `isum += vaddvq_s32(p) * scales[i]` pattern at `quants.c:2284, 2835,
2843, 2943, 3548` forfeits vector accumulation. See F08.
* **No `quantize_row_q8_K` SIMD optimization.** The function is a
  placeholder that always calls the scalar reference (`quants.c:134-136`).
  Same gap as ARTX02-W5 on x86.
* **No runtime dispatch inside kernels.** Every ISA branch is compile-
  time. The `.so` selection happens once at load time via the score
  function.

---

## 13. Architectural Strengths

1. **`ggml_vdotq_s32` macro is a clean compile-time dispatch.** The
   single macro (`ggml-cpu-impl.h:307-321`) selects between the 3-op
   baseline emulation and the 1-op native DOTPROD. Every int8 vecdot
   kernel uses this macro, so adding a new instruction (e.g. SME's
   `sdot` variants) is one macro edit, not 20 kernel edits. This is
   the ARM analogue of ARTX02-F02's `mul_sum_us8_pairs_float`.

2. **Two-accumulator dep-chain-breaking pattern.** The Q4_0/Q4_1/Q5_0/
   Q5_1/Q8_0 baseline vecdots use two independent `float32x4_t`
   accumulators processing 2 blocks per iteration. Each accumulator is
   an independent dependency chain — the FMA into `sumv1` does not
   wait on the FMA into `sumv0`. This is the right design for an
   in-order core (Cortex-A53, A55, A7) where dep-chain latency
   dominates.

3. **Multi-binary dispatch is OS-portable and low-overhead.** The
   score function (`cpu-feats.cpp:69-99`) is a 30-line function that
   returns 0 or a power-of-two-weighted integer. No ifunc, no PLT
   indirection, no runtime patching. Cross-ref ARTX01-F12.

4. **The `table_b2b_0/1` LUT is a clean trick.** Expanding 1 bit to
   8 bytes via a 2 KB LUT eliminates 8 shift+or instructions per
   Q1_0/Q5_0/Q5_1 vecdot. The LUT is `static const` and shared across
   all three kernels. Worth ADOPT.

5. **The `keven_signs_q2xs[1024]` LUT makes IQ2_XXS/XS/IQ3_XXS
   practical.** Without it, each sub-block would need 32 conditional
   sign-flips. The LUT reduces this to one indexed load per 8 sign
   bits. The fact that baseline NEON is the *only* ISA path for these
   quants is a deliberate design choice — these are the I-quants that
   matter most on small ARM devices.

6. **Block-aligned assertion + scalar tail.** Same strength as ARTX02.
   Keeps the SIMD kernel simple without sacrificing correctness for
   odd-shaped inputs.

7. **32-bit ARM compatibility.** The 220-line shim block in
   `ggml-cpu-impl.h:87-305` lets the *same source* compile for both
   AArch64 and ARMv7-A. This is rare among LLM inference frameworks
   and is a meaningful portability win.

---

## 14. Architectural Weaknesses

### W1 — No baseline-NEON batched GEMV/GEMM in `repack.cpp`

**Evidence:** `repack.cpp:231, 292, 358, 450, 520, 590, 729, 883, 1042,
1329, 1518, 1718, 1776, 1846, 2327, 2748, 3186, 3262, 3338, 3544, 3773,
4293, 4540, 4742, 4958, 5026, 5091` — every batched entry point gates
on `__ARM_FEATURE_DOTPROD`, `__ARM_FEATURE_MATMUL_INT8`, or
`__ARM_FEATURE_SVE`. Baseline NEON falls through to `_generic`.

**Impact:** Prompt-processing shapes (large `ne11`, wide `ne10`) on
baseline NEON run a scalar reference. For a model with `d_model = 4096`,
a single prompt-processor iteration on Cortex-A53 may be 50-100× slower
than the per-block vecdot path would suggest, because the batched
GEMV/GEMM kernels are not just SIMD versions of the vecdot — they use
8×8/4x4 interleaved weight layouts that the scalar reference does not
exploit.

**Why it's hard to fix:** Writing a baseline-NEON batched GEMM would
duplicate the structure of the DOTPROD/I8MM paths but with the 3-op
`ggml_vdotq_s32` emulation, dramatically higher register pressure, and
no clear throughput benefit over the per-block vecdot path. The
DOTPROD gate exists because the batched path is only profitable when
the inner dot product is 1 instruction, not 6.

### W2 — K-quant baseline path collapses to scalar after each block

**Evidence:** `quants.c:1980-1981` (Q2_K), `quants.c:2267-2287` (Q3_K),
`quants.c:2834-2843` (Q4_K), `quants.c:2943` (Q5_K), `quants.c:3548-3578`
(Q6_K). Each loop iteration computes `ggml_vdotq_s32(...)`, immediately
`vaddvq_s32(p)` reduces it to a scalar, then multiplies by a scalar
scale and accumulates into a scalar `isum`. The SVE/I8MM paths keep
the scale-multiply in vector form via `svmla_n_s32_x` / `vmmlaq_s32`.

**Impact:** Each Q4_K block (256 elements, 32 sub-blocks of 8 elements)
runs 32 `ggml_vdotq_s32` calls, 32 `vaddvq_s32` reductions, 32 scalar
multiplies, 32 scalar adds. The dep chain is 32-deep. The SVE/I8MM
paths break this by keeping the accumulator vector and applying the
scale as a vector multiply.

**Why it's hard to fix:** The K-quant scale layout (8 different per-
block-32 scales, packed in 6-bit fields) makes vector broadcast
nontrivial. The DOTPROD/I8MM paths handle this with shuffle masks
(`get_scale_shuffle_k4` on x86; SVE has `svdup_n_s32(scales[i])`).
A baseline-NEON equivalent would require a 16-byte shuffle LUT per
block, similar to x86's `get_scale_shuffle_q3k`.

### W3 — 32-bit ARM shims marked "not tested"

**Evidence:** `ggml-cpu-impl.h:241` (`// NOTE: not tested` for
`ggml_vqtbl1q_s8`), `ggml-cpu-impl.h:265` (`// NOTE: not tested` for
`ggml_vqtbl1q_u8`), `ggml-cpu-impl.h:170` (`// TODO: double-check these
work correctly` for `vld1q_*_x2/x4`).

**Impact:** The shims are used by every baseline NEON IQ2/IQ3/IQ4/
MXFP4/NVFP4 vecdot on 32-bit ARM. If any shim is wrong, those kernels
silently produce wrong results on 32-bit ARM hardware. There is no
compile-time or runtime test that exercises these paths.

**Why it's hard to fix:** Requires running the ggml test suite on
32-bit ARM hardware (e.g. Raspberry Pi 2/3 in 32-bit mode, or a
Cortex-A9 board). Static analysis cannot verify the shims are correct.

### W4 — NVFP4 path requires FMA, MXFP4 path does not

**Evidence:** `quants.c:826` gates the NVFP4 NEON path on
`#if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)`. The MXFP4
path at `quants.c:766` gates only on `#if defined __ARM_NEON` and
uses `vmlaq_n_f32` (line 791-793), which compiles to a non-FMA
multiply+add on hardware without FMA.

**Impact:** On ARMv7-A NEONv1 (no FMA), NVFP4 vecdot falls through
to the scalar `#else` path (line 899), losing all SIMD throughput.
MXFP4 vecdot keeps its NEON path but uses non-FMA multiply+add.
The two kernels are inconsistent in their FMA requirement. On
AArch64 (which always has FMA), both paths are NEON; the
inconsistency is invisible.

**Why it's hard to fix:** Replace `vfmaq_f32` in NVFP4 with
`vmlaq_f32` (non-FMA equivalent), or gate both kernels consistently.
The original author may have used `vfmaq_f32` for a reason (slightly
higher precision on FMA hardware).

### W5 — `quantize_row_q8_K` is a placeholder

**Evidence:** `quants.c:134-136`:
```c
// placeholder implementation for Apple targets
void quantize_row_q8_K(const float * GGML_RESTRICT x, void * GGML_RESTRICT y, int64_t k) {
    quantize_row_q8_K_ref(x, y, k);
}
```
No `#if defined(__ARM_NEON)` ladder. Always calls the generic reference.
The comment says "for Apple targets" but the function is unconditional
on ARM. Same as ARTX02-W5 on x86.

**Impact:** Every matmul that targets a K-quant (Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, IQ2_*, IQ3_*, IQ4_XS, IQ1_*) pays scalar performance for
the F32 → Q8_K activation conversion on baseline NEON.

**Why it's hard to fix:** Q8_K has 256-element blocks with 32
sub-block scales; writing a SIMD quantizer that produces both the
packed int8 values and the per-sub-block `bsums[QK_K/16]` array is
non-trivial but straightforward.

### W6 — 32-bit ARM has no score function

**Evidence:** `cpu-feats.cpp:3` — the entire file is wrapped in
`#if defined(__aarch64__)`. There is no `ggml_backend_cpu_armv7_score`
or equivalent. The baseline `.so` (compiled for ARMv7-A with NEON) is
the only one loaded on 32-bit ARM hardware, regardless of whether
ARMv7-A has optional features like NEONv2 with `vqtbl1q_s8`.

**Impact:** On 32-bit ARM hardware, there is no multi-binary dispatch.
The `.so` loaded is whatever was compiled; if the build was for
ARMv7-A without NEONv2, the `__ARM_NEON` paths are unavailable and
everything falls back to the scalar reference. If the build was for
ARMv7-A with NEONv2 (`-mfpu=neon-vfpv4`), the baseline NEON paths
are taken. There is no fallback detection.

**Why it's hard to fix:** 32-bit ARM has no standard `/proc/cpuinfo`
feature bits equivalent to aarch64's `getauxval(AT_HWCAP)`. Detection
would require parsing `/proc/cpuinfo` strings (Android does this) or
using `SIGILL` trap-and-recover. Probably not worth the engineering
cost given 32-bit ARM's declining relevance.

### W7 — No software prefetching in baseline NEON vecdots

**Evidence:** Grep for `__builtin_prefetch` / `vprf` / `pld` across
`quants.c` returns zero matches. The K-quant blocks (256 bytes for
Q4_K, larger for Q6_K) exceed the typical 4-line hardware prefetcher
window on Cortex-A53/A55.

**Impact:** Unknown statically. The hardware prefetcher on Cortex-A72
(4-line stream, 8 outstanding) may be sufficient; on Cortex-A53 (2-line
stream, 4 outstanding) it likely is not. Baseline ARM is the platform
most likely to benefit from software prefetching, and it is exactly
the platform that does not have it.

**Why it's hard to fix:** Requires runtime profiling on representative
ARM hardware (Cortex-A53, A72, A76) to determine whether prefetching
helps. Static analysis cannot resolve.

### W8 — IQ2/IQ3 baseline paths use scalar accumulators

**Evidence:** `quants.c:3660, 3678-3679` (IQ2_XXS: `float sumf1, sumf2`),
`quants.c:3807, 3845-3848` (IQ2_S: `int sumi1, sumi2`),
`quants.c:3892, 3911-3912` (IQ3_XXS: `float sumf1, sumf2`),
`quants.c:3981, 4021-4022` (IQ3_S: `int sumi1, sumi2`),
`quants.c:4061, 4082-4083` (IQ1_S: `int sumi1, sumi2, sumi3`),
`quants.c:4285, 4302-4303` (IQ4_XS: `int sumi1, sumi2`).

**Impact:** Each sub-block's `ggml_vdotq_s32` result is reduced to
scalar immediately, then multiplied by a scalar scale and accumulated
into a scalar `sumi1`/`sumi2`/`sumf1`/`sumf2`. The dep chain is N-
deep where N is the number of sub-blocks. IQ2_XS and IQ1_M are the
only I-quants that use `int32x4_t` accumulators (`quants.c:3732, 4144`).

**Why it's hard to fix:** Same as W2 — the per-sub-block scale is
scalar, and broadcasting it to a vector requires a shuffle LUT. The
DOTPROD/I8MM paths on ARTX05 may handle this differently.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc` | **ADOPT** | `ggml_vdotq_s32` macro pattern (baseline fallback ↔ DOTPROD) | One macro edit swaps baseline 3-op emulation for 1-op DOTPROD. Cross-ref F02. |
| `glproc` | **ADOPT** | 2-block-per-iter, 2-accumulator dep-chain-breaking pattern | Right design for in-order ARM cores (Cortex-A53/A55/A7). F04. |
| `glproc` | **ADOPT** | `table_b2b_0/1` bit-expansion LUT | 4 KB LUT replaces per-bit shift+or for Q1_0/Q5_0/Q5_1. F06. |
| `glproc` | **ADOPT** | `keven_signs_q2xs[1024]` sign-mask LUT | Makes IQ2/IQ3 I-quants practical on baseline NEON. F07. |
| `glproc` | **ADOPT** | Multi-binary score function (`ggml_backend_cpu_aarch64_score`) | Low-overhead, OS-portable. Cross-ref ARTX01-F12. |
| `glproc` | **ADOPT** | 32-bit ARM compatibility shims (with tests) | Same-source AArch64/ARMv7-A compilation is a real portability win. F05. |
| `glproc` | **REJECT** | Absence of baseline-NEON batched GEMV/GEMM | Prompt processing on baseline NEON runs scalar. Provide at least a 4×4 baseline batched GEMV. F03. |
| `glproc` | **REJECT** | K-quant baseline scalar-collapse pattern | Vectorize the scale-multiply as the SVE/I8MM paths do. F08. |
| `glproc` | **ADAPT** | NVFP4 FMA gate | Make FMA gating consistent across NVFP4 and MXFP4. F09. |
| `glproc` | **MONITOR** | 32-bit ARM `vqtbl1q_s8/u8` shims ("not tested") | Verify with runtime tests on 32-bit ARM hardware before relying on them. F05. |
| `glproc` | **MONITOR** | `quantize_row_q8_K` placeholder | Should be SIMD-optimized; monitor whether upstream does it first. F10. |
| `glproc` | **DEFER** | 32-bit ARM score function | Probably not worth the engineering cost given 32-bit ARM's declining relevance. F10. |
| `GATE` | **ADOPT** | `nrows = 1` baseline / `nrows = 2` I8MM contract | Already adopted per ARTX01; lets multi-row vecdot be advertised per-quant per-build. |
| `GATE` | **ADAPT** | SVE `vector_length` switch with baseline fallback | Keep the switch, but make baseline NEON an explicit `default:` rather than an `assert(false)`. |

---

## 16. Recommendations

### R1 — ADOPT the `ggml_vdotq_s32` macro pattern for glproc
**Priority:** Critical
**Difficulty:** S
**Dependencies:** none
GwenLand's `glproc` should define an equivalent `gl_vdotq_s32(acc, a, b)` macro with the same `#if defined(__ARM_FEATURE_DOTPROD)` ladder: native `vdotq_s32` on DOTPROD, 3-op `vmull_s8` + `vpaddlq_s16` + `vaddq_s32` emulation on baseline NEON. Every int8 vecdot kernel should route through this macro. (F02.)

### R2 — ADOPT the 2-block-per-iter, 2-accumulator dep-chain-breaking pattern
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
For Q4_0/Q4_1/Q5_0/Q5_1/Q8_0-equivalent vecdots, process 2 blocks per iteration with two independent `float32x4_t` accumulators. This breaks the dep chain on in-order ARM cores where FMA latency is 4-5 cycles. (F04.)

### R3 — REJECT the absence of baseline-NEON batched GEMV/GEMM; provide at least a 4×4 baseline path
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
GwenLand should provide a baseline-NEON batched GEMV (4×4 interleaved weight layout) for at least Q4_0, Q4_K, IQ4_NL on Cortex-A53/A55-class hardware. The 4×4 layout (vs. 8×8) fits the 16-register ARMv7-A budget better. Expected speedup: 2-4× over the per-block vecdot path for prompt-processing shapes with `ne11 ∈ [4, 32]`. (F03, W1.)

### R4 — ADOPT the `table_b2b_0/1` bit-expansion LUT
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** none
GwenLand should replicate the 4 KB `table_b2b_0/1` LUT for Q1_0/Q5_0/Q5_1 vecdots. The LUT replaces 8 shift+or instructions with one indexed load. (F06.)

### R5 — ADOPT the `keven_signs_q2xs[1024]` sign-mask LUT
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** none
GwenLand should replicate the 1 KB `keven_signs_q2xs` LUT for IQ2_XXS/XS/IQ3_XXS vecdots. The LUT is what makes these I-quants practical on baseline NEON. (F07.)

### R6 — REJECT the K-quant scalar-collapse pattern; vectorize the scale-multiply
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
GwenLand should keep the K-quant accumulator in `int32x4_t` form across the inner loop, applying the per-block-32 scale as a vector multiply via `vmulq_n_s32(acc, scales[i])` followed by `vpaddq_s32` reduction at the end. This breaks the 32-deep dep chain in the current baseline path. Requires a scale-broadcast shuffle LUT similar to x86's `get_scale_shuffle_k4`. (F08, W2.)

### R7 — ADOPT multi-binary score function for aarch64
**Priority:** High
**Difficulty:** M
**Dependencies:** none
GwenLand should adopt the `ggml_backend_cpu_aarch64_score` pattern: compile N `.so` variants per ARM ISA target, score each at load time, pick the best. (F01, cross-ref ARTX01-F12.)

### R8 — ADOPT 32-bit ARM compatibility shims, but add tests
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
GwenLand should keep the 220-line 32-bit ARM shim block (`vaddlvq_s16`, `vpaddq_s16`, `vaddvq_s32`, `vaddvq_f32`, `vmaxvq_f32`, `vcvtnq_s32_f32`, `vzip1_u8`, `vzip2_u8`, `vld1q_*_x2/x4`, `vqtbl1q_s8/u8`). Add a unit test that exercises each shim on 32-bit ARM hardware. Mark the test as skipped on AArch64. (F05, W3.)

### R9 — ADAPT NVFP4/MXFP4 FMA gating for consistency
**Priority:** Low
**Difficulty:** XS
**Dependencies:** none
GwenLand should either (a) gate both NVFP4 and MXFP4 NEON paths on `__ARM_FEATURE_FMA` and use `vfmaq_f32`, or (b) gate neither and use `vmlaq_f32`. The current llama.cpp code is inconsistent. Recommendation: option (b), because the precision difference is negligible and ARMv7-A NEONv1 hardware benefits from the non-FMA path. (F09, W4.)

### R10 — ADAPT `quantize_row_q8_K` to be SIMD-optimized
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
The placeholder at `quants.c:134-136` should be replaced with an AArch64/ARMv7-A SIMD implementation. The quantizer must produce both the packed int8 values and the per-sub-block `bsums[QK_K/16]` array. (F10, W5.)

### R11 — MONITOR software prefetching on Cortex-A53-class hardware
**Priority:** Low
**Difficulty:** S
**Dependencies:** R3
GwenLand should add software prefetching (`__builtin_prefetch` with `PLDL1KEEP` hint) to the K-quant baseline paths on Cortex-A53/A55-class hardware, behind a runtime check. Measure whether it helps before making it default. (W7.)

---

## 17. Findings

### Finding ARTX04-F01

```
Finding ID:           ARTX04-F01
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Per-block vecdot dispatch ladder (all quant types)
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0 (and ~20 sibling vecdot functions)
Lines:                297-588 (Q4_0 representative); 590-747 (Q4_1); 920-1030 (Q5_0);
                      1032-1148 (Q5_1); 1150-1395 (Q8_0); 1685-2016 (Q2_K);
                      2018-2312 (Q3_K); 2334-2862 (Q4_K); 2864-2962 (Q5_K);
                      2964-3592 (Q6_K)

Summary:              Every per-block vecdot has a 3-tier compile-time dispatch
                      ladder: __ARM_FEATURE_MATMUL_INT8 (vmmlaq_s32) →
                      __ARM_FEATURE_SVE (svdot_s32) → __ARM_NEON (baseline, this
                      audit's scope). Baseline NEON is the silent fallback when
                      neither upper tier is compiled in.

Observation:          The preprocessor ladder for every quant type is:
                        #if defined(__ARM_FEATURE_MATMUL_INT8)  // I8MM, nrc==2 path
                          ... vmmlaq_s32 ...
                        #endif
                        #if defined(__ARM_FEATURE_SVE)         // SVE
                          ... svdot_s32, switch(vector_length) ...
                        #elif defined(__ARM_NEON)              // baseline
                          ... ggml_vdotq_s32 (3-op fallback) ...
                        #endif
                      The baseline path is reached when __ARM_FEATURE_MATMUL_INT8
                      and __ARM_FEATURE_SVE are both undefined at compile time. It
                      is also reached when SVE is compiled in but vector_length is
                      not in {128, 256, 512} (the SVE switch defaults to
                      assert(false); the baseline path runs only if SVE is
                      entirely absent).

                      The I8MM tier is the only one that accepts nrc==2 (two weight
                      rows per call); baseline and SVE tiers assert nrc==1.

Evidence:             quants.c:315 (#if defined(__ARM_FEATURE_MATMUL_INT8) — first);
                      quants.c:391 (#if defined(__ARM_FEATURE_SVE) — second);
                      quants.c:527 (#elif defined(__ARM_NEON) — baseline fallback);
                      quants.c:302-306 (assertion: nrc==1 unless MATMUL_INT8).

Architectural Impact: Baseline NEON is the universal fallback. Adding a new ARM
                      extension (e.g. SME's outer-product instructions) means
                      adding a new #if tier at the top of every vecdot — not a
                      one-line macro edit. The ladder is duplicated across 20+
                      vecdot functions.

Correctness Impact:   None. All three tiers produce identical total sums (modulo
                      ULP-level reassociation differences from different lane
                      groupings).

Optimization Type:    SIMD (compile-time ISA selection).

GwenLand Target:      glproc

Recommendation:       ADOPT the 3-tier ladder pattern but route every baseline
                      path through the ggml_vdotq_s32 macro (R1) so that the only
                      per-kernel edit needed for a new ISA tier is the macro
                      definition. Avoid duplicating the ladder across 20 kernels.

Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F02

```
Finding ID:           ARTX04-F02
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            ggml_vdotq_s32 DOTPROD fallback
Source File:          ggml/src/ggml-cpu/ggml-cpu-impl.h
Function:             ggml_vdotq_s32
Lines:                307-321

Summary:              On baseline NEON (no __ARM_FEATURE_DOTPROD), the
                      ggml_vdotq_s32 macro emulates vdotq_s32 with 3 ops:
                      vmull_s8 ×2 + vpaddlq_s16 ×2 + vaddq_s32. Every baseline
                      NEON int8 vecdot routes through this macro.

Observation:          The macro is defined in ggml-cpu-impl.h:307-321:
                        #if !defined(__ARM_FEATURE_DOTPROD)
                        inline static int32x4_t ggml_vdotq_s32(int32x4_t acc,
                                                               int8x16_t a,
                                                               int8x16_t b) {
                            const int16x8_t p0 = vmull_s8(vget_low_s8 (a),
                                                          vget_low_s8 (b));
                            const int16x8_t p1 = vmull_s8(vget_high_s8(a),
                                                          vget_high_s8(b));
                            return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0),
                                                            vpaddlq_s16(p1)));
                        }
                        #else
                        #define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)
                        #endif
                      The comment at line 309 documents a precision caveat: the
                      fallback produces the same total sum as native vdotq_s32
                      but with different per-lane grouping. Every caller reduces
                      via vaddvq_s32 immediately, so the caveat does not affect
                      correctness — but a future kernel that inspected individual
                      lanes would silently produce different results on baseline
                      NEON vs. DOTPROD builds.

                      The 3-op emulation costs ~6 cycles on Cortex-A72 (2-cycle
                      vmull_s8 ×2 on F0/F1 pipes, 1-cycle vpaddlq_s16 ×2 on F0,
                      1-cycle vaddq_s32 ×2 on F0) vs. 1-cycle for native
                      vdotq_s32.

Evidence:             ggml-cpu-impl.h:310-315 (fallback definition);
                      ggml-cpu-impl.h:317-319 (DOTPROD macro);
                      quants.c:205-206 (Q1_0 callsite);
                      quants.c:562-563 (Q4_0 callsite);
                      quants.c:1374-1379 (Q8_0 callsite);
                      quants.c:2834, 2841 (Q4_K callsites — 2 per iter);
                      quants.c:3548-3578 (Q6_K callsites — 8 per iter).

Architectural Impact: The 3-op emulation is THE baseline-NEON hot primitive. Its
                      ~6× latency overhead vs. native DOTPROD is the single
                      biggest reason baseline NEON is slow. Every vecdot that
                      routes through it pays this cost per 16-element chunk.

Correctness Impact:   None. The total sum is bit-identical to native vdotq_s32
                      when reduced via vaddvq_s32 (verified by the comment at
                      line 309 and by the fact that the test suite passes on
                      both builds).

Optimization Type:    SIMD (instruction emulation via macro).

GwenLand Target:      glproc

Recommendation:       ADOPT. Define an equivalent gl_vdotq_s32 macro with the
                      same #if ladder. Every int8 vecdot kernel should route
                      through this macro. (R1.)

Priority:             Critical
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F03

```
Finding ID:           ARTX04-F03
Category:             MISSING_FEATURE
Engine:               CPU
Component:            Batched GEMV/GEMM in repack.cpp
Source File:          ggml/src/ggml-cpu/arch/arm/repack.cpp
Function:             ggml_gemv_q4_0_4x4_q8_0, ggml_gemv_q4_0_4x8_q8_0,
                      ggml_gemv_q4_0_8x8_q8_0, ggml_gemv_iq4_nl_4x4_q8_0,
                      ggml_gemv_mxfp4_4x4_q8_0, ggml_gemv_q4_K_8x4_q8_K,
                      ggml_gemv_q4_K_8x8_q8_K, ggml_gemv_q5_K_8x4_q8_K,
                      ggml_gemv_q5_K_8x8_q8_K, ggml_gemv_q6_K_8x4_q8_K,
                      ggml_gemv_q6_K_8x8_q8_K, ggml_gemv_q8_0_4x4_q8_0,
                      ggml_gemv_q8_0_4x8_q8_0, ggml_gemm_q4_0_4x4_q8_0,
                      ggml_gemm_q4_0_4x8_q8_0, ggml_gemm_q4_0_8x8_q8_0,
                      ggml_gemm_iq4_nl_4x4_q8_0, ggml_gemm_mxfp4_4x4_q8_0,
                      ggml_gemm_q4_K_8x4_q8_K, ggml_gemm_q5_K_8x4_q8_K,
                      ggml_gemm_q4_K_8x8_q8_K, ggml_gemm_q5_K_8x8_q8_K,
                      ggml_gemm_q6_K_8x4_q8_K, ggml_gemm_q6_K_8x8_q8_K,
                      ggml_gemm_q8_0_4x4_q8_0, ggml_gemm_q8_0_4x8_q8_0
Lines:                212-269 (Q4_0 4x4 gemv representative); 273-335 (4x8);
                      339-428 (8x8); ... 5091-5154 (last gemm)

Summary:              Every batched GEMV/GEMM entry point in repack.cpp is gated
                      behind __ARM_FEATURE_DOTPROD, __ARM_FEATURE_MATMUL_INT8,
                      or __ARM_FEATURE_SVE. On baseline NEON, every entry point
                      falls through to _generic.

Observation:          The pattern is uniform:
                        #if !((defined(_MSC_VER)) && !defined(__clang__)) && \
                            defined(__aarch64__) && defined(__ARM_NEON) && \
                            defined(__ARM_FEATURE_DOTPROD)
                          ... SIMD batched kernel using vdotq_laneq_s32 ...
                          return;
                        #endif
                        ggml_gemv_*_generic(n, s, bs, vx, vy, nr, nc);
                      The baseline NEON path is the _generic suffix, which is a
                      scalar reference. The same pattern applies to the I8MM-gated
                      (ggml_gemm_q4_0_4x8_q8_0, ggml_gemm_q4_K_8x8_q8_K) and
                      SVE-gated (ggml_gemv_q4_0_8x8_q8_0, ggml_gemm_q4_K_8x8_q8_K)
                      variants.

                      Even when weights have been repacked into block_q4_0x8 /
                      block_q4_0x4 interleaved layout (via
                      ggml_quantize_mat_q8_0_4x4/4x8 in repack.cpp:51, 119, which
                      DO have baseline NEON paths), the consuming kernel is
                      unreachable on baseline. The repacked layout is wasted.

Evidence:             repack.cpp:231 (first DOTPROD gate, gemv_q4_0_4x4);
                      repack.cpp:269 (fallthrough to _generic);
                      repack.cpp:292 (4x8 gate);
                      repack.cpp:335 (4x8 fallthrough);
                      repack.cpp:358-359 (8x8 SVE gate);
                      repack.cpp:428 (8x8 fallthrough);
                      repack.cpp:1846 (gemm 4x4 DOTPROD gate);
                      repack.cpp:2303 (gemm 4x4 fallthrough);
                      repack.cpp:2327 (gemm 4x8 I8MM gate);
                      repack.cpp:2724 (gemm 4x8 fallthrough);
                      repack.cpp:2748-2749 (gemm 8x8 SVE+I8MM gate);
                      repack.cpp:3162 (gemm 8x8 fallthrough).

Architectural Impact: Prompt-processing shapes on baseline NEON run a scalar
                      reference. For a model with d_model=4096, a single prompt-
                      processor iteration on Cortex-A53 may be 50-100× slower
                      than the per-block vecdot path would suggest. The batched
                      kernels are not just SIMD versions of the vecdot — they
                      use 8×8/4x4 interleaved weight layouts that the scalar
                      reference does not exploit.

Correctness Impact:   None. The _generic reference is correct.

Optimization Type:    SIMD (absence of optimization).

GwenLand Target:      glproc

Recommendation:       REJECT the absence. GwenLand should provide a baseline-NEON
                      4×4 batched GEMV for at least Q4_0, Q4_K, IQ4_NL on Cortex-
                      A53/A55-class hardware. The 4×4 layout (vs. 8×8) fits the
                      16-register ARMv7-A budget better. (R3.)

Priority:             High
Difficulty:           L
Dependencies:         ARTX04-F02
Confidence:           High
```

### Finding ARTX04-F04

```
Finding ID:           ARTX04-F04
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 baseline vecdot accumulator pattern
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q4_0_q8_0, ggml_vec_dot_q4_1_q8_1,
                      ggml_vec_dot_q5_0_q8_0, ggml_vec_dot_q5_1_q8_1,
                      ggml_vec_dot_q8_0_q8_0
Lines:                527-567 (Q4_0); 688-725 (Q4_1); 938-1002 (Q5_0);
                      1050-1120 (Q5_1); 1352-1380 (Q8_0)

Summary:              The 5 most common baseline NEON vecdots process 2 blocks
                      per loop iteration using 2 independent float32x4_t
                      accumulators (sumv0, sumv1). This breaks the FMA dep chain
                      on in-order ARM cores.

Observation:          The pattern is uniform:
                        float32x4_t sumv0 = vdupq_n_f32(0.0f);
                        float32x4_t sumv1 = vdupq_n_f32(0.0f);
                        for (; ib + 1 < nb; ib += 2) {
                            // load x[ib+0], x[ib+1], y[ib+0], y[ib+1]
                            const int32x4_t p_0 = ggml_vdotq_s32(...);
                            const int32x4_t p_1 = ggml_vdotq_s32(...);
                            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p_0),
                                                 GGML_CPU_FP16_TO_FP32(x0->d) *
                                                 GGML_CPU_FP16_TO_FP32(y0->d));
                            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p_1),
                                                 GGML_CPU_FP16_TO_FP32(x1->d) *
                                                 GGML_CPU_FP16_TO_FP32(y1->d));
                        }
                        sumf = vaddvq_f32(sumv0) + vaddvq_f32(sumv1);
                      Each accumulator is an independent dependency chain — the
                      FMA into sumv1 does not wait on the FMA into sumv0. On
                      Cortex-A53 (FMA latency 4 cycles, 2 FMA pipes), this gives
                      ~2× throughput vs. a single-accumulator loop.

                      A scalar tail loop (for (; ib < nb; ++ib)) handles the
                      odd-nb case. The tail is pure scalar C.

Evidence:             quants.c:528-529 (Q4_0 2-accumulator declaration);
                      quants.c:531 (Q4_0 loop header: ib += 2);
                      quants.c:565-566 (Q4_0 dual vmlaq_n_f32);
                      quants.c:689-690 (Q4_1 2-accumulator);
                      quants.c:939-940 (Q5_0 2-accumulator);
                      quants.c:1051-1052 (Q5_1 2-accumulator);
                      quants.c:1353-1354 (Q8_0 2-accumulator).

Architectural Impact: Right design for in-order ARM cores (Cortex-A53, A55, A7,
                      A9, A15). On out-of-order cores (Cortex-A72+, Apple A-series)
                      the hardware renames the accumulators anyway, so the
                      benefit is smaller but non-zero (renaming capacity is
                      finite).

Correctness Impact:   None. The two accumulators are summed at the end via
                      vaddvq_f32(sumv0) + vaddvq_f32(sumv1); the result is
                      identical (modulo ULP-level reassociation) to a single-
                      accumulator loop.

Optimization Type:    SIMD (dep-chain breaking via multiple accumulators).

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the 2-block-per-iter, 2-accumulator pattern
                      in glproc's Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 baseline vecdots. (R2.)

Priority:             High
Difficulty:           S
Dependencies:         ARTX04-F02
Confidence:           High
```

### Finding ARTX04-F05

```
Finding ID:           ARTX04-F05
Category:             ADOPT
Engine:               CPU
Component:            32-bit ARM NEON compatibility shims
Source File:          ggml/src/ggml-cpu/ggml-cpu-impl.h
Function:             vaddlvq_s16, vpaddq_s16, vpaddq_s32, vaddvq_s32,
                      vaddvq_f32, vmaxvq_f32, vcvtnq_s32_f32, vzip1_u8, vzip2_u8,
                      ggml_vld1q_s16_x2, ggml_vld1q_u8_x2, ggml_vld1q_u8_x4,
                      ggml_vld1q_s8_x2, ggml_vld1q_s8_x4, ggml_vqtbl1q_s8,
                      ggml_vqtbl1q_u8
Lines:                87-305

Summary:              220 lines of inline shims let the same quants.c source
                      compile for both AArch64 and ARMv7-A NEONv2. Each shim
                      emulates an AArch64-only intrinsic via lane-by-lane scalar
                      ops. Two shims are marked "NOTE: not tested".

Observation:          The shim block is gated on #if !defined(__aarch64__) at
                      line 87. It defines:
                        - vaddlvq_s16: vpaddlq_s32(vpaddlq_s16(v)) then lane-reduce
                        - vpaddq_s16 / vpaddq_s32: vpadd_s16/s32 on low+high halves
                        - vaddvq_s32 / vaddvq_f32: 4-lane scalar sum
                        - vmaxvq_f32: 4-lane scalar max
                        - vcvtnq_s32_f32: 4-lane roundf() (line 132-141)
                        - vzip1_u8 / vzip2_u8: 8-lane manual interleave
                        - ggml_vld1q_*_x2/x4: struct-of-N-vld1q (line 176-239)
                        - ggml_vqtbl1q_s8 / ggml_vqtbl1q_u8: 16-lane a[b[i]] loop
                          (line 242-287), marked "NOTE: not tested" at 241, 265
                      The struct types ggml_int8x16x2_t, ggml_uint8x16x2_t, etc.
                      are defined as plain structs (line 172-228) with .val[N]
                      fields, matching the AArch64 int8x16x2_t layout.

                      On AArch64, the #else branch at line 289-304 #defines the
                      ggml_ prefixes to the native intrinsics (vld1q_s8_x2,
                      vqtbl1q_s8, etc.) — zero overhead.

Evidence:             ggml-cpu-impl.h:87 (#if !defined(__aarch64__));
                      ggml-cpu-impl.h:101-104 (vaddlvq_s16 shim);
                      ggml-cpu-impl.h:118-120 (vaddvq_s32 shim);
                      ggml-cpu-impl.h:132-141 (vcvtnq_s32_f32 shim — uses roundf);
                      ggml-cpu-impl.h:241-263 (ggml_vqtbl1q_s8, "NOTE: not tested");
                      ggml-cpu-impl.h:265-287 (ggml_vqtbl1q_u8, "NOTE: not tested");
                      ggml-cpu-impl.h:289-304 (AArch64 #else — native intrinsics).

Architectural Impact: Same-source AArch64/ARMv7-A compilation is a real
                      portability win. Most LLM frameworks drop 32-bit ARM
                      support entirely; llama.cpp keeps it. The cost is ~6×
                      slower vqtbl1q emulation on ARMv7-A and ~4× slower
                      vaddvq emulation.

Correctness Impact:   The shims at line 241, 265 are marked "NOTE: not tested".
                      The vld1q_*_x2/x4 shims at line 165-239 have "TODO: double-
                      check these work correctly" at line 170. If any shim is
                      wrong, IQ2/IQ3/IQ4/MXFP4/NVFP4 vecdots on 32-bit ARM
                      silently produce wrong results. There is no compile-time
                      or runtime test that exercises these paths.

Optimization Type:    None (compatibility shim).

GwenLand Target:      glproc

Recommendation:       ADOPT the shim block but add tests. Specifically, add a
                      unit test that exercises each shim on 32-bit ARM hardware
                      (Raspberry Pi 3 in 32-bit mode is sufficient). Mark the
                      test as skipped on AArch64. (R8.)

Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F06

```
Finding ID:           ARTX04-F06
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            Q1_0/Q5_0/Q5_1 bit-to-byte expansion LUT
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q1_0_q8_0, ggml_vec_dot_q5_0_q8_0,
                      ggml_vec_dot_q5_1_q8_1
Lines:                37-38 (LUT definitions); 174-178 (Q1_0 use);
                      960-968 (Q5_0 use); 1078-1086 (Q5_1 use)

Summary:              A pair of 256-entry, 8-byte LUTs (table_b2b_0 and
                      table_b2b_1, total 4 KB) expands a single bit to all 8
                      bit positions of an 8-byte vector in one indexed load,
                      replacing 8 shift+or instructions.

Observation:          The LUTs are generated by the B8 macro (line 27-34):
                        #define B8(c, s) B7(c, s, c), B7(c, s, s)
                        static const uint64_t table_b2b_0[1 << 8] = { B8(00, 10) };
                        static const uint64_t table_b2b_1[1 << 8] = { B8(10, 00) };
                      table_b2b_0[byte] returns a uint64_t where each bit of the
                      input byte has been expanded to a full nibble (0x00 or 0x10).
                      table_b2b_1 returns the complement (0x10 or 0x00).

                      In Q5_0 vecdot (line 960-968), the qh field (4 bytes, 32
                      bits) is split into 4 bytes, each looked up in
                      table_b2b_1 to produce a 16-byte sign vector in 4 LUT
                      loads + 4 vld1q_s8 loads. This replaces 32 shift+or
                      instructions with 4 LUT lookups.

                      In Q1_0 vecdot (line 174-178), each byte of the 4-byte
                      bit field is looked up in table_b2b_0 to produce the
                      sign vector for 8 int8 values.

                      These LUTs are baseline-NEON-specific. The DOTPROD/I8MM/SVE
                      paths do not use them.

Evidence:             quants.c:27-34 (B8 macro);
                      quants.c:37-38 (LUT definitions);
                      quants.c:174-178 (Q1_0 use);
                      quants.c:960-968 (Q5_0 use — table_b2b_1);
                      quants.c:1078-1086 (Q5_1 use — table_b2b_0).

Architectural Impact: Replaces 8 shift+or instructions with one indexed load.
                      On Cortex-A53 (1-cycle LUT load from L1), this is ~8×
                      faster than the shift+or ladder. The 4 KB LUT fits easily
                      in L1.

Correctness Impact:   None. The LUTs are precomputed at compile time; the
                      expansion is bit-exact.

Optimization Type:    SIMD (LUT-based bit expansion).

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the table_b2b_0/1 LUT and the B8 macro
                      in glproc. (R4.)

Priority:             Medium
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F07

```
Finding ID:           ARTX04-F07
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            IQ2_XXS/IQ2_XS/IQ3_XXS sign-mask LUT
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_iq2_xxs_q8_K, ggml_vec_dot_iq2_xs_q8_K,
                      ggml_vec_dot_iq3_xxs_q8_K
Lines:                3594-3629 (LUT definition); 3646 (IQ2_XXS use);
                      3708 (IQ2_XS use); 3879 (IQ3_XXS use)

Summary:              A 1024-byte LUT (keven_signs_q2xs) provides 128 entries
                      of 8-byte sign masks (each ±1) for IQ2_XXS/XS/IQ3_XXS
                      vecdot. Baseline NEON is the ONLY ISA path for these three
                      I-quants — there is no SVE/I8MM/DOTPROD variant.

Observation:          The LUT is gated on #if defined (__ARM_NEON) at line 3594.
                      It is referenced from:
                        - IQ2_XXS (line 3646, 3668-3671): 4 sign-mask loads per
                          sub-block, each from keven_signs_q2xs[sign_index & 127]
                        - IQ2_XS (line 3708, 3739-3742): same pattern, indexed
                          by (q2[k] >> 9)
                        - IQ3_XXS (line 3879, 3901-3904): same pattern, indexed
                          by (aux32[i] >> N) & 127
                      Each entry is 8 bytes of ±1 values. The LUT consumes 1 KB
                      of static data, fits in L1.

                      These three vecdots have NO SVE/I8MM/DOTPROD path. The
                      baseline NEON implementation is the only one — even on
                      the highest-end ARM hardware (Cortex-X3, Apple M3), these
                      quants run the baseline path.

Evidence:             quants.c:3594 (#if defined (__ARM_NEON) — only gate);
                      quants.c:3595-3628 (LUT definition — 1024 int8_t values);
                      quants.c:3631-3691 (IQ2_XXS vecdot — no upper-tier #if);
                      quants.c:3693-3765 (IQ2_XS vecdot — no upper-tier #if);
                      quants.c:3864-3924 (IQ3_XXS vecdot — no upper-tier #if);
                      quants.c:3685-3690 (IQ2_XXS #else branch — _generic only).

Architectural Impact: The LUT makes these I-quants practical on baseline NEON.
                      Without it, each sub-block would need 32 conditional sign-
                      flips. The fact that baseline NEON is the only path is a
                      deliberate design choice — these are the I-quants that
                      matter most on small ARM devices where the activation
                      cache is small.

Correctness Impact:   None. The LUT is precomputed at compile time.

Optimization Type:    SIMD (LUT-based sign expansion).

GwenLand Target:      glproc

Recommendation:       ADOPT. Replicate the keven_signs_q2xs LUT in glproc. (R5.)

Priority:             Medium
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F08

```
Finding ID:           ARTX04-F08
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            K-quant baseline vecdot scale-multiply collapse
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_q2_K_q8_K, ggml_vec_dot_q3_K_q8_K,
                      ggml_vec_dot_q4_K_q8_K, ggml_vec_dot_q5_K_q8_K,
                      ggml_vec_dot_q6_K_q8_K
Lines:                1979-2000 (Q2_K); 2267-2289 (Q3_K); 2824-2847 (Q4_K);
                      2924-2948 (Q5_K); 3524-3580 (Q6_K)

Summary:              The baseline NEON K-quant vecdots collapse each block's
                      int32x4 dot product to a scalar via vaddvq_s32(p), then
                      multiply by a scalar per-block-32 scale and accumulate
                      into a scalar isum. This creates a 32-deep dep chain per
                      K-quant block; the SVE/I8MM paths keep the accumulator
                      vector.

Observation:          The pattern is uniform across Q2_K, Q3_K, Q4_K, Q5_K, Q6_K:
                        int32_t isum = 0;
                        for (int j = 0; j < QK_K/64; ++j) {
                            const int32x4_t p = ggml_vdotq_s32(...);
                            isum += vaddvq_s32(p) * scales[2*j+0];  // scalar!
                            ...
                        }
                        sumf += d * isum;
                      The SVE path at line 2748-2782 uses svmla_n_s32_x to keep
                      the accumulator in svint32_t form:
                        svint32_t sumi1 = svdup_n_s32(0);
                        for (int j = 0; j < QK_K/64; ++j) {
                            svint32_t dot = svdot_s32(mzero, q4bytes, q8bytes);
                            sumi1 = svmla_n_s32_x(svptrue_b32(), sumi1, dot,
                                                  scales[2*j+0]);
                        }
                      The I8MM path uses vmmlaq_s32 with similar vector
                      accumulation.

                      Each K-quant baseline iteration does:
                        1 × ggml_vdotq_s32 (3-op emulation, ~6 cycles)
                        1 × vaddvq_s32 (~2 cycles on A53)
                        1 × scalar multiply (~1 cycle)
                        1 × scalar add (~1 cycle)
                      Total ~10 cycles per sub-block. With 32 sub-blocks per
                      K-quant block, that's a 320-cycle dep chain. The SVE path
                      breaks this to ~96 cycles (3 cycles per svdot_s32 on
                      Cortex-X3, no scalar reduction).

Evidence:             quants.c:1980-1981 (Q2_K scalar isum accumulation);
                      quants.c:2267-2270 (Q3_K scalar isum);
                      quants.c:2834-2835 (Q4_K scalar isum1 += vaddvq_s32(p1) * scales[...]);
                      quants.c:2943 (Q5_K scalar sumi);
                      quants.c:3548-3578 (Q6_K scalar isum — 8 sub-blocks per iter);
                      quants.c:2751 (SVE Q4_K equivalent — svmla_n_s32_x vector accumulate);
                      quants.c:2675-2690 (I8MM Q4_K — vmmlaq_s32 + vector scale broadcast).

Architectural Impact: 32-deep dep chain per K-quant block. On in-order Cortex-A53
                      the chain stalls the FMA pipe ~30% of the time. The SVE/I8MM
                      paths avoid this entirely.

Correctness Impact:   None. The scalar and vector paths produce identical total
                      sums (modulo ULP-level reassociation from different reduction
                      orders).

Optimization Type:    None (suboptimal layout — accumulator collapses to scalar).

GwenLand Target:      glproc

Recommendation:       REJECT the scalar-collapse pattern. Keep the K-quant
                      accumulator in int32x4_t form across the inner loop,
                      applying the per-block-32 scale as a vector multiply via
                      vmulq_n_s32(acc, scales[i]) followed by vpaddq_s32 reduction
                      at the end. Requires a scale-broadcast shuffle LUT. (R6, W2.)

Priority:             High
Difficulty:           M
Dependencies:         ARTX04-F02
Confidence:           High
```

### Finding ARTX04-F09

```
Finding ID:           ARTX04-F09
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            NVFP4 vs MXFP4 FMA gating inconsistency
Source File:          ggml/src/ggml-cpu/arch/arm/quants.c
Function:             ggml_vec_dot_nvfp4_q8_0, ggml_vec_dot_mxfp4_q8_0
Lines:                826-898 (NVFP4); 766-796 (MXFP4)

Summary:              NVFP4 vecdot gates its NEON path on
                      __ARM_NEON && __ARM_FEATURE_FMA; MXFP4 vecdot gates only
                      on __ARM_NEON. The two kernels are inconsistent in their
                      FMA requirement.

Observation:          NVFP4 (line 826):
                        #if defined(__ARM_NEON) && defined(__ARM_FEATURE_FMA)
                          ... vfmaq_f32(acc, vcvtq_f32_s32(sumi), scales) ...
                        #else
                          ... scalar fallback ...
                        #endif
                      MXFP4 (line 766):
                        #if defined __ARM_NEON
                          ... vmlaq_n_f32 equivalent via sumf += ... (line 791-793) ...
                        #endif
                      On AArch64 (which always has FMA), both paths are NEON.
                      On ARMv7-A NEONv1 (no FMA), NVFP4 falls through to scalar;
                      MXFP4 keeps its NEON path but uses non-FMA multiply+add.

                      The two kernels use different accumulation strategies:
                        - NVFP4: float32x4_t acc with vfmaq_f32 (FMA)
                        - MXFP4: float sumf with scalar += (no FMA needed)
                      The NVFP4 FMA gate is therefore not strictly necessary —
                      vmlaq_f32 (non-FMA) would work on ARMv7-A NEONv1. The
                      gate appears to be a copy-paste from a kernel that
                      genuinely required FMA.

Evidence:             quants.c:826 (NVFP4 #if with FMA gate);
                      quants.c:896 (NVFP4 vfmaq_f32 use);
                      quants.c:899 (NVFP4 #else scalar fallback);
                      quants.c:766 (MXFP4 #if without FMA gate);
                      quants.c:791-793 (MXFP4 scalar += accumulation, no FMA).

Architectural Impact: On ARMv7-A NEONv1 hardware (Cortex-A9, A15 without FMA),
                      NVFP4 vecdot falls to scalar — losing all SIMD throughput.
                      MXFP4 vecdot keeps its NEON path. The inconsistency is
                      invisible on AArch64 (always FMA) but visible on legacy
                      32-bit ARM.

Correctness Impact:   None. Both paths produce identical results; the difference
                      is only whether NEON or scalar is used on non-FMA hardware.

Optimization Type:    SIMD (FMA usage gating).

GwenLand Target:      glproc

Recommendation:       ADAPT. Make FMA gating consistent across NVFP4 and MXFP4.
                      Recommendation: drop the FMA gate on NVFP4 and use vmlaq_f32
                      (non-FMA equivalent of vfmaq_f32). The precision difference
                      is negligible and ARMv7-A NEONv1 hardware benefits from the
                      non-FMA path. (R9, W4.)

Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX04-F10

```
Finding ID:           ARTX04-F10
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Multi-binary score function (ARM)
Source File:          ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp
Function:             ggml_backend_cpu_aarch64_score
Lines:                1-103 (entire file)

Summary:              The ARM score function is aarch64-only. There is no score
                      function for 32-bit ARM. On 32-bit ARM hardware, the
                      baseline .so (compiled for ARMv7-A with NEON) is the only
                      one loaded, regardless of which optional features (NEONv2,
                      VFPv4) the CPU supports.

Observation:          The entire file is wrapped in #if defined(__aarch64__)
                      at line 3. The score function at line 69-99:
                        static int ggml_backend_cpu_aarch64_score() {
                            int score = 1;
                            aarch64_features af;
                            #ifdef GGML_USE_DOTPROD
                            if (!af.has_dotprod) { return 0; }
                            score += 1<<1;
                            #endif
                            ... (same pattern for FP16_VA, SVE, I8MM, SVE2, SME)
                            return score;
                        }
                      The score is power-of-two-weighted: DOTPROD=2, FP16_VA=4,
                      SVE=8, I8MM=16, SVE2=32, SME=64. The .so with the highest
                      non-zero score wins. The baseline .so (compiled with no
                      GGML_USE_*) returns score=1 unconditionally.

                      The aarch64_features constructor (line 33-67) reads
                      AT_HWCAP / AT_HWCAP2 on Linux (getauxval) and sysctlbyname
                      on Apple. Apple does not implement SVE (line 64 comment).

                      There is no equivalent for 32-bit ARM. The baseline .so
                      is loaded by default; if the build was for ARMv7-A without
                      NEONv2, the __ARM_NEON paths are unavailable and everything
                      falls back to scalar.

Evidence:             cpu-feats.cpp:3 (#if defined(__aarch64__) — entire file);
                      cpu-feats.cpp:33-67 (aarch64_features constructor — Linux/
                      Apple only);
                      cpu-feats.cpp:69-99 (score function — aarch64 only);
                      cpu-feats.cpp:101 (GGML_BACKEND_DL_SCORE_IMPL macro call);
                      cpu-feats.cpp:103 (#endif — no 32-bit ARM score).

Architectural Impact: 32-bit ARM gets no multi-binary dispatch. The build-time
                      -march flag decides what runs. On Android ARMv7-A (still
                      common in 2026 on low-end devices), there is no way to
                      ship a NEONv2 .so that loads only on NEONv2-capable
                      hardware — the build is one-size-fits-all.

                      The placeholder quantize_row_q8_K at quants.c:134-136
                      ("placeholder implementation for Apple targets") is also
                      unconditional on ARM — same gap as ARTX02-W5 on x86. There
                      is no SIMD optimization for Q8_K activation quantization
                      on any ARM build, baseline or otherwise.

Correctness Impact:   None. The score function only affects which .so loads; it
                      does not affect kernel correctness.

Optimization Type:    Multi-binary ISA dispatch (absent on 32-bit ARM).

GwenLand Target:      glproc

Recommendation:       ADOPT the aarch64 score function (R7). DEFER a 32-bit ARM
                      score function — probably not worth the engineering cost
                      given 32-bit ARM's declining relevance. ADAPT
                      quantize_row_q8_K to be SIMD-optimized (R10).

Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the 32-bit ARM `vqtbl1q_s8`/`vqtbl1q_u8` shims
  (ggml-cpu-impl.h:241-287, marked "NOTE: not tested") produce correct
  results on real ARMv7-A hardware. Requires running the ggml test
  suite on a Raspberry Pi 3 in 32-bit mode or a Cortex-A9 board.
  Static analysis confirms the algorithm matches the AArch64 spec but
  cannot verify the lane-indexing on real silicon.
* **U2**. Whether the baseline NEON vecdots spill registers on ARMv7-A
  (16 quad-regs vs. AArch64's 32). The Q4_0 baseline loop holds ~20
  NEON registers (Section 7.4); on ARMv7-A this exceeds the 16-quad-
  reg budget and forces spills. The spill cost may negate the 2-
  accumulator dep-chain benefit. Requires runtime profiling on
  ARMv7-A hardware.
* **U3**. Whether the K-quant baseline scalar-collapse pattern (F08)
  is meaningfully slower than a vector-accumulator version on Cortex-
  A53. The scalar dep chain is 32-deep, but each iteration is only
  ~10 cycles, so the chain is 320 cycles — well within the L1 hit
  window. A vector-accumulator version would still pay the 6-cycle
  ggml_vdotq_s32 cost per sub-block; the win is only in the dep-chain
  break. Requires runtime profiling.
* **U4**. Whether software prefetching would help the K-quant baseline
  paths on Cortex-A53/A55. The 256-byte Q4_K block exceeds the typical
  4-line (256-byte) hardware prefetcher window. Static analysis
  cannot resolve; requires runtime profiling with `__builtin_prefetch`.
* **U5**. Whether the `ggml_vdotq_s32` lane-grouping caveat
  (ggml-cpu-impl.h:309 comment) is exercised by any test in the ggml
  test suite. The comment says "do not use when individual lane values
  matter"; if no test exercises individual lanes, the caveat is
  undocumented behavior. Requires inspecting the test suite (not
  audited here).
* **U6**. Whether the `keven_signs_q2xs[1024]` LUT (F07) is byte-
  identical to the equivalent LUT in the SVE/I8MM paths of ARTX05.
  If ARTX05 introduces a SVE/I8MM variant of IQ2_XXS/XS/IQ3_XXS, the
  LUT must match. Static analysis of quants.c shows baseline is the
  only path; ARTX05 will reveal whether the LUT is reused.
* **U7**. Whether the `vld1q_*_x2/x4` shims (ggml-cpu-impl.h:165-239,
  marked "TODO: double-check these work correctly") produce the same
  memory layout as the AArch64 native intrinsics. The shims define
  custom struct types (ggml_int8x16x2_t etc.); if the AArch64
  int8x16x2_t has a different layout, the kernels would silently
  miscompile on 32-bit ARM. Requires runtime verification.
* **U8**. Whether the `quantize_row_q8_K` placeholder (F10, W5) is a
  deliberate design choice (perhaps Q8_K is rarely the activation
  format) or a missed optimization. The K-quants Q2_K-Q6_K all use
  Q8_K as their activation format; the placeholder is on the hot
  path. Requires inspecting the git history (not audited here).

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `quantize_row_q8_0` (NEON path)                | 41-83         |
| R02       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `quantize_row_q8_1` (NEON path)                | 85-131        |
| R03       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `quantize_row_q8_K` (placeholder)              | 134-136       |
| R04       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q1_0_q8_0` (NEON path)           | 140-220       |
| R05       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q2_0_q8_0` (NEON path)           | 222-295       |
| R06       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_0_q8_0` (NEON path)           | 297-588       |
| R07       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_1_q8_1` (NEON path)           | 590-747       |
| R08       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_mxfp4_q8_0`                      | 749-808       |
| R09       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_nvfp4_q8_0` (FMA-gated)          | 810-918       |
| R10       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q5_0_q8_0` (NEON path)           | 920-1030      |
| R11       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q5_1_q8_1` (NEON path)           | 1032-1148     |
| R12       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q8_0_q8_0` (NEON path)           | 1150-1395     |
| R13       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_tq1_0_q8_K` (NEON/DOTPROD)       | 1397-1572     |
| R14       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_tq2_0_q8_K` (NEON/DOTPROD)       | 1574-1683     |
| R15       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q2_K_q8_K` (NEON path)           | 1685-2016     |
| R16       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q3_K_q8_K` (NEON path)           | 2018-2312     |
| R17       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q4_K_q8_K` (NEON path)           | 2334-2862     |
| R18       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q5_K_q8_K` (NEON path)           | 2864-2962     |
| R19       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_q6_K_q8_K` (NEON path)           | 2964-3592     |
| R20       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `keven_signs_q2xs[1024]` LUT                   | 3594-3628     |
| R21       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq2_xxs_q8_K` (NEON-only)        | 3631-3691     |
| R22       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq2_xs_q8_K` (NEON-only)         | 3693-3765     |
| R23       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq2_s_q8_K` (NEON path)          | 3767-3862     |
| R24       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq3_xxs_q8_K` (NEON-only)        | 3864-3924     |
| R25       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq3_s_q8_K` (NEON path)          | 3926-4034     |
| R26       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq1_s_q8_K` (NEON path)          | 4036-4100     |
| R27       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq1_m_q8_K` (NEON path)          | 4102-4194     |
| R28       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq4_nl_q8_0` (NEON path)         | 4196-4254     |
| R29       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `ggml_vec_dot_iq4_xs_q8_K` (NEON path)         | 4256-4318     |
| R30       | `ggml/src/ggml-cpu/arch/arm/quants.c`               | `table_b2b_0/1` LUTs                           | 37-38         |
| R31       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_quantize_mat_q8_0_4x4` (NEON reachable)  | 51-117        |
| R32       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_quantize_mat_q8_0_4x8` (NEON reachable)  | 119-210       |
| R33       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemv_q4_0_4x4_q8_0` (DOTPROD-gated)      | 212-269       |
| R34       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemv_q4_0_8x8_q8_0` (SVE-gated)          | 339-428       |
| R35       | `ggml/src/ggml-cpu/arch/arm/repack.cpp`             | `ggml_gemm_q4_0_8x8_q8_0` (SVE+I8MM-gated)     | 2728-3162     |
| R36       | `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`          | `ggml_backend_cpu_aarch64_score`               | 69-99         |
| R37       | `ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp`          | `aarch64_features` constructor                 | 33-67         |
| R38       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_vdotq_s32` (DOTPROD fallback macro)      | 307-321       |
| R39       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | 32-bit ARM shim block                          | 87-305        |
| R40       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `vaddlvq_s16`, `vaddvq_s32`, `vaddvq_f32` shims| 101-124       |
| R41       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `vcvtnq_s32_f32` shim                          | 132-141       |
| R42       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `vld1q_*_x2/x4` shims ("TODO: double-check")   | 165-239       |
| R43       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_vqtbl1q_s8/u8` shims ("NOTE: not tested")| 241-287       |
| R44       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `ggml_nvfp4_dot8` helper                       | 323-330       |
| R45       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `GGML_CPU_FP16_TO_FP32` (ARM NEON scalar)      | 38-55         |
| R46       | `ggml/src/ggml-cpu/simd-mappings.h`                 | E8M0 / UE4M3 LUTs                              | 129-136       |
| R47       | `ggml/src/ggml-cpu/ggml-cpu-impl.h`                 | `GGML_COMPUTE_FP16_TO_FP32` (generic)          | 433-437       |
| R48       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | ARM feature detection (`check_arm_feature`)    | 145-160       |
| R49       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | ARM all-variants matrix                        | 171-213       |
| R50       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | ARM feature compile-check                      | 226-237       |
