# ARTX03 — AMD Zen (Zen 4 / Zen 5 / AVX-512 + AVX-VNNI + BF16) Quantized Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glproc` (AMD Zen kernel layer + multi-binary dispatch), `GATE` (kernel-selection contract)

---

## 1. Executive Summary

The AMD Zen quantized-kernel story in llama.cpp is, in one sentence: **there
is no AMD Zen quantized-kernel story.** Every line of x86 SIMD code audited
in ARTX02 (the IceLake audit) runs unchanged on Zen 4 and Zen 5, because
llama.cpp makes zero AMD-specific kernel decisions. The vendor bit
`is_amd` is set in `cpu-feats.cpp:135` and consulted only to gate seven
legacy AMD feature flags (ABM, SSE4a, XOP, TBM, MMXEXT, 3DNOWEXT, 3DNOW) at
`cpu-feats.cpp:68-77`. No score calculation, no kernel-selection branch,
no compile-time `#if`, and no runtime check anywhere in `ggml/src/ggml-cpu/`
reads `is_amd` for any decision that affects a kernel path.

The result is a codebase that treats Zen 4 and Zen 5 as "AVX-512 + AVX-VNNI
+ BF16 capable IceLake-like" CPUs. This is mostly correct — Zen 4 and Zen 5
do implement the relevant AVX-512 subsets — but it ignores the single most
important micro-architectural difference between Zen 4/5 and Intel's
IceLake/Sapphire Rapids: **Zen 4 and Zen 5 implement AVX-512 on a 256-bit
data path**, so every 512-bit `__m512` instruction is decoded into two
256-bit uops internally. On Intel, the same instructions execute on a
native 512-bit data path. The throughput is therefore similar (Zen actually
sustains 256-bit VNNI at one instruction per cycle, just like Intel), but
the 512-bit instructions cost twice the decode/fetch bandwidth and trigger
mild downclocking on Zen without providing any throughput benefit.

Three concrete consequences fall out of this:

1. **The `tinyBLAS_Q0_AVX` template** (`llamafile/sgemm.cpp:1351`) is the
   *only* audited AMD-friendly design. It uses `__m256`/`__m256i` width
   even on AVX-512 builds, but harvests the 32-register file (via
   `VECTOR_REGISTERS == 32` at `sgemm.cpp:67`) to pick wider tile shapes
   (4×4, 4×3, 3×4). The int8 dot product inside it dispatches to
   `_mm256_dpbusd_epi32` (AVX-512 VNNI + VL at 256-bit width) on Zen 4/Zen 5
   via the `updot` helper at `sgemm.cpp:1754-1764`. This is exactly the
   right pattern for Zen: native 256-bit data path + 32 registers + VNNI
   acceleration at 256-bit width. Worth **ADOPT**.

2. **The 8×8 batched GEMM in `repack.cpp`** (line 663+, 16 × `__m512`
   accumulators) and the `tinyBLAS<16, __m512, …>` / `tinyBLAS<32, __m512,
   __m512bh, …>` templates in `llamafile/sgemm.cpp:3727, 3790` are
   *suboptimal* on Zen 4/Zen 5. They issue 512-bit instructions that split
   into two 256-bit uops. There is no compile-time switch to prefer the
   256-bit equivalents (`_mm256_dpbusd_epi32` / `_mm256_dpbf16_ps` /
   `_mm256_fmadd_ph`) on Zen, even though those equivalents are already
   implemented as template specializations in `sgemm.cpp:154-159` and are
   available via AVX-512 + VL on every Zen 4/Zen 5 CPU.

3. **The `zen4` build variant** is misnamed. `ggml/src/CMakeLists.txt:396`
   defines `ggml_add_cpu_backend_variant(zen4 … AVX512_BF16)`. Zen 4
   hardware does **not** have AVX-512 BF16 — only Zen 5 added it. The
   score function (`cpu-feats.cpp:309-311`) returns 0 for the `zen4`
   variant on actual Zen 4 silicon because `!is.AVX512_BF16()` fails. So
   Zen 4 falls back to the `icelake` variant, and the `zen4` variant is
   effectively a Zen 5 binary. There is no Zen 4-specific `.so`. This is
   not a crash bug, but it is a configuration and naming defect that
   obscures what the build matrix actually delivers to AMD hardware.

For GwenLand, the decisions worth **ADOPT**ing are: the `tinyBLAS_Q0_AVX`
pattern (32 registers + 256-bit data path + 256-bit VNNI), the 3-way VNNI
helper ladder (already noted in ARTX02-F02), and the multi-binary dispatch
scheme (ARTX02-F09). The decisions worth **REJECT**ing are: ignoring the
256-bit Zen data path when picking 512-bit vs 256-bit instructions, the
misnamed `zen4` variant, and the inflated Zen 4 score from the redundant
`AVX_VNNI` bit. The decisions worth **MONITOR**ing are: AVX-512 FP16 in
`vec.cpp:ggml_vec_dot_f16` (it IS used, contradicting ARTX02-F08's scope,
but only at 512-bit width), and the 256-bit `_mm256_dpbf16_ps`
specialization in `sgemm.cpp:156-159` (exists but never instantiated on
the BF16 GEMM template).

---

## 2. Purpose

Document the AMD Zen 4 / Zen 5 quantized-kernel paths and the AMD-specific
decisions (or their absence) in llama.cpp. This audit answers four
questions posed in the brief:

* Does llama.cpp have any AMD-specific kernel paths, or does it treat
  Zen 4/Zen 5 as "AVX-512 + AVX-VNNI + BF16 capable IceLake-like"?
  → **The latter.** Zero AMD-specific kernel paths exist. See F01, F10.
* Is the 512-bit AVX-512 path the same on Zen 4 as on IceLake? Does the
  code acknowledge Zen 4's 256-bit data path?
  → **Same code, no acknowledgment.** See F03, F05, F08.
* Is `is_amd` used anywhere beyond setting `is_amd = true`?
  → **Yes — but only to gate seven legacy AMD feature flags** (ABM, SSE4a,
  XOP, TBM, MMXEXT, 3DNOWEXT, 3DNOW). See F01. No kernel decision consults
  it.
* Is there any Zen-specific tuning (e.g., 256-bit preferred over 512-bit
  on Zen 4)?
  → **No Zen-specific tuning exists**, with one happy accident: the
  `tinyBLAS_Q0_AVX` template happens to use 256-bit width + 32 registers,
  which is correct for Zen — but it does so because the template was
  written before AVX-512 was added to the AVX variant ladder, not because
  of any Zen-aware decision. See F06.

This audit is **not** responsible for: IceLake-specific findings (covered
in ARTX02 F01-F12), AMX tile-based matmul (the `amx/` subdirectory, which
is Intel-only and never compiles in on AMD), ARM kernels (ARTX04),
elementwise ops (ARTX06), or graph scheduling (ARTX01).

---

## 3. Source Files

| File                                          | Lines | Role                                                                                |
| --------------------------------------------- | ----- | ----------------------------------------------------------------------------------- |
| `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`    | 327   | CPUID + multi-binary score function; vendor detection; AVX-VNNI/BF16/FP16/AMX bits  |
| `ggml/src/ggml-cpu/arch/x86/quants.c`         | 4108  | Per-block vecdot kernels (audited in ARTX02; AMD inherits)                          |
| `ggml/src/ggml-cpu/arch/x86/repack.cpp`       | 6407  | 8×8 batched GEMV/GEMM with `__m512` 512-bit path; helpers at lines 115-176          |
| `ggml/src/ggml-cpu/llamafile/sgemm.cpp`       | 4059  | tinyBLAS GEMM templates; `tinyBLAS_Q0_AVX` (256-bit, AMD-friendly) at 1351-1794     |
| `ggml/src/ggml-cpu/vec.cpp`                   | 614   | F32 / F16 / BF16 vecdot paths; BF16 native dot at 148-158; F16 native FMA at 264-378 |
| `ggml/src/ggml-cpu/simd-mappings.h`           | 1318  | `__AVX512FP16__` macros (native `_mm512_fmadd_ph`) at 493-578; F16C / LUT fallbacks |
| `ggml/src/CMakeLists.txt`                     | —     | x86 build variant definitions; `zen4` variant at line 396                           |
| `ggml/src/ggml-cpu/amx/amx.cpp`               | 249   | AMX plugin — conditional on `__AMX_INT8__ && __AVX512VNNI__`; compiles out on AMD   |

> Note: `arch/x86/quants.c` and `arch/x86/repack.cpp` are the same files
> audited in ARTX02. AMD inherits every line of those files unchanged.
> This audit references them only where the AMD execution path differs
> (or fails to differ) from the IceLake execution path.

---

## 4. Architecture Overview

```
                ┌─────────────────────────────────────────────────────────┐
                │  ggml/src/CMakeLists.txt:378-402                        │
                │  x86 variant matrix (built when GGML_CPU_ALL_VARIANTS): │
                │    x64, sse42, sandybridge, ivybridge, piledriver,      │
                │    haswell, skylakex, cannonlake, cascadelake,          │
                │    icelake, cooperlake, zen4, alderlake, sapphirerapids │
                │  Each variant compiled with -mavx* + GGML_AVX512* defs  │
                └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼  (load-time score selection)
                ┌─────────────────────────────────────────────────────────┐
                │  cpu-feats.cpp:263-323  ggml_backend_cpu_x86_score()    │
                │  Reads CPUID, returns 0 if any required feature is      │
                │  missing, otherwise returns a power-of-2-weighted sum.  │
                │  Does NOT consult is_amd.                               │
                └─────────────────────────────────────────────────────────┘
                                      │
                                      ▼  (highest-score .so is loaded)
                ┌─────────────────────────────────────────────────────────┐
                │  On actual AMD silicon, the loaded .so is:              │
                │   • Zen 4 (e.g. Ryzen 9 7950X, EPYC 9654):              │
                │       — zen4 variant FAILS (no AVX512_BF16 bit)         │
                │       — falls back to icelake variant                   │
                │   • Zen 5 (e.g. Ryzen 9 9950X, EPYC 9755):              │
                │       — zen4 variant loads (has AVX512_BF16)            │
                │       — runs the SAME code as on Cooper Lake / SPR      │
                └─────────────────────────────────────────────────────────┘
                                      │
            ┌─────────────────────────┼─────────────────────────────┐
            ▼                         ▼                             ▼
   ┌────────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
   │ arch/x86/quants.c  │    │ arch/x86/repack.cpp  │    │ llamafile/sgemm.cpp  │
   │ per-block vecdot   │    │ 8×8 batched GEMM     │    │ tinyBLAS templates   │
   │ 256-bit __m256 acc │    │ 16× __m512 acc       │    │                      │
   │ VNNI helper @105   │    │ 512-bit uops split   │    │ tinyBLAS_Q0_AVX      │
   │ 256-bit VNNI path  │    │ into 2× 256-bit on   │    │ 256-bit __m256 acc   │
   │ taken on Zen 4/5   │    │ Zen 4/5              │    │ 32 regs (VR=32)      │
   │                    │    │                      │    │ 256-bit VNNI path    │
   │ NO is_amd branch   │    │ NO is_amd branch     │    │ → AMD-friendly!      │
   └────────────────────┘    └──────────────────────┘    └──────────────────────┘
                                      │
                                      ▼  (also used by F16/BF16 matmuls)
                ┌─────────────────────────────────────────────────────────┐
                │  vec.cpp:                                              │
                │   ggml_vec_dot_f32  (line 11)  — __m512 if AVX512F      │
                │   ggml_vec_dot_bf16 (line 139) — _mm512_dpbf16_ps if    │
                │                                   __AVX512BF16__ (Zen 5)│
                │                                 — _mm512_mul_ps + add   │
                │                                   if __AVX512F__ (Zen 4)│
                │   ggml_vec_dot_f16  (line 264) — _mm512_fmadd_ph if     │
                │                                   __AVX512FP16__ (Zen 5,│
                │                                   via simd-mappings.h)  │
                └─────────────────────────────────────────────────────────┘
```

Key design points:

* **No AMD-specific branch anywhere.** Grep for `is_amd`, `__zn`, `ZEN`,
  `zen4`, `zen5`, `AVX10`, `family`, `0x19`, `0x1a` across
  `ggml/src/ggml-cpu/` returns hits only in `cpu-feats.cpp` (the legacy
  feature gates at lines 68-77) and in the CMake `zen4` variant name. The
  kernel-selection logic, the score function, and every kernel template
  treat AMD and Intel identically.
* **Multi-binary dispatch is the only AMD-relevant mechanism.** The
  loader picks the best-matching `.so` via `ggml_backend_cpu_x86_score`
  (`cpu-feats.cpp:263`). For AMD, the relevant variants in priority order
  are: `icelake` (Zen 4), `zen4` (Zen 5). The `sapphirerapids` variant
  requires AMX_INT8 (Intel-only) and is never loaded on AMD. See F02.
* **No `ifunc`, no runtime patching.** Once the `.so` is loaded, every
  kernel call is a direct call. The cost of the multi-binary scheme is
  paid entirely at load time; hot-path overhead is zero. Cross-ref
  ARTX01-F12, ARTX02-F09.
* **The 256-bit data path of Zen 4/Zen 5 is undocumented in the code.**
  No comment, no `#if`, no architecture-aware tuning. The only implicit
  acknowledgment is the `tinyBLAS_Q0_AVX` template's choice to use
  `__m256` width (a choice made for AVX2 compatibility, not for Zen).
  See F03, F06.

---

## 5. Execution Flow

### 5.1 Build-time variant selection

`ggml/src/CMakeLists.txt:371-402` defines 14 x86 variants when
`GGML_CPU_ALL_VARIANTS` is set. The variants are listed in ascending
capability order; the loader picks the highest-scoring one. The relevant
variant lines for AMD are:

```
line 390: icelake   SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI
line 396: zen4      SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16
```

The `zen4` variant is a strict superset of `icelake` plus `AVX512_BF16`.
There is no separate `zen5` variant. The `cooperlake` variant (line 395)
is similar to `zen4` but lacks `AVX512_VBMI` — so Cooper Lake loads
`cooperlake`, Zen 5 loads `zen4`, and Zen 4 (which has AVX512_VBMI but
not AVX512_BF16) loads `icelake`. See Finding F02.

### 5.2 Load-time score function

`ggml_backend_cpu_x86_score` (`cpu-feats.cpp:263-323`) computes a
power-of-two-weighted integer. Each `#ifdef GGML_*` block adds a bit if
the CPUID bit is set, or returns 0 if a required bit is missing.

Approximate scores (base 1 + sum of bit weights):

| Variant loaded     | CPU example                          | Score |
| ------------------ | ------------------------------------ | ----- |
| `icelake`          | IceLake-client i7-1065G7             | 1473  |
| `icelake`          | **Zen 4** (Ryzen 9 7950X, EPYC 9654) | 1537  |
| `zen4`             | **Zen 5** (Ryzen 9 9950X, EPYC 9755) | 2049  |
| `cooperlake`       | Cooper Lake Xeon                     | 1731  |
| `sapphirerapids`   | Sapphire Rapids / Granite Rapids     | 4033  |

Zen 4 outscores IceLake-client by 64 (the `AVX_VNNI` bit, `1<<6`). This
inflates the score without reflecting any real capability advantage on
AMD — see F09. The `AVX_VNNI` bit on AMD is redundant with
`AVX512_VNNI + AVX512_VL`, which subsumes AVX-VNNI at 256-bit width via
`_mm256_dpbusd_epi32`.

### 5.3 Kernel dispatch (per matmul)

The dispatch path is identical to ARTX02 §5. The type-traits table
(`ggml-cpu.c:214`) routes `vec_dot` calls to the linker-resolved function
in `arch/x86/quants.c` or `vec.cpp`. The 8×8 batched GEMM is invoked only
when `GGML_USE_CPU_REPACK` is enabled at build time
(`repack.cpp:4821-4835` registers the buffer type only under that macro).
`llamafile_sgemm` is invoked when `GGML_USE_LLAMAFILE` is enabled.

On AMD, all three paths execute the same compiled code as on Intel.
There is no runtime CPU-detection branch inside any kernel.

### 5.4 Hot-path instruction sequence (Zen 4, Q4_0 vecdot)

On a Zen 4 CPU running the `icelake` variant `.so`:

1. `type_traits_cpu[GGML_TYPE_Q4_0].vec_dot` resolves to
   `ggml_vec_dot_q4_0_q8_0` (`arch/x86/quants.c:701`).
2. The `#if defined(__AVX2__)` branch runs (line 718). The helper
   `mul_sum_i8_pairs_float` (`quants.c:122`) is called per block.
3. Because `__AVXVNNIINT8__` is undefined (no variant enables it), the
   `#else` branch runs (line 127-133): `abs+sign` trick, then
   `mul_sum_us8_pairs_float`.
4. `mul_sum_us8_pairs_float` (`quants.c:105`) hits the first `#if`
   branch: `__AVX512VNNI__ && __AVX512VL__` are both defined (icelake
   variant sets `__AVX512VNNI__`), so `_mm256_dpbusd_epi32` runs at
   256-bit width.
5. On Zen 4, this is a single 256-bit uop, executing on the native
   256-bit data path. **No uop split, no downclocking penalty.** This is
   the efficient path.

The same code path on IceLake-client also uses `_mm256_dpbusd_epi32` —
same instruction, same width. Zen 4 and IceLake run identical code with
near-identical throughput. The 256-bit width of the vecdot helper is
*correct* for Zen, but it is correct by accident: the helper was designed
for the AVX2 baseline, and the AVX-512 VNNI path slots into the same
256-bit shape.

### 5.5 Hot-path instruction sequence (Zen 5, BF16 GEMM)

On a Zen 5 CPU running the `zen4` variant `.so`:

1. `llamafile_sgemm` (`sgemm.cpp:3699`) accepts a `BF16 × BF16 → F32`
   matmul.
2. `__AVX512BF16__` is defined, so line 3788-3795 instantiates
   `tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, float>`.
3. The `madd<__m512bh, __m512bh, __m512>` template specialization
   (`sgemm.cpp:151-155`) calls `_mm512_dpbf16_ps` — a 512-bit BF16 dot
   product, 32 BF16 lanes per pair of `__m512bh` inputs.
4. On Zen 5's 256-bit data path, this 512-bit instruction **splits into
   two 256-bit uops**. Each uop produces 16 lanes of FP32 output.
5. The 256-bit equivalent `_mm256_dpbf16_ps` is implemented as a
   template specialization at `sgemm.cpp:156-159`, but the BF16 GEMM
   dispatch at line 3790 instantiates only the 512-bit template. The
   256-bit specialization is **dead code** on the BF16 GEMM path.

This is the central AMD-specific suboptimality: 512-bit instructions on
Zen 5 do not buy throughput, but they cost decode bandwidth and trigger
mild frequency reduction. The 256-bit equivalents exist in the codebase
but are not selected.

---

## 6. Data Layout

### 6.1 Block layouts (unchanged from ARTX02)

AMD inherits the same `block_q4_0`, `block_q8_0`, `block_q4_K`,
`block_q8_K`, `block_iq4_xs`, etc. layouts defined in `ggml-common.h`.
There are no AMD-specific block layouts.

### 6.2 "Repacked" 8×8 layouts

The `block_q4_0x8`, `block_q8_0x4`, `block_iq4_nlx8`, `block_mxfp4x8`,
`block_q4_Kx8`, `block_q8_Kx4`, `block_q2_Kx8` interleaved layouts
(`repack.cpp:4528-4570`) are not AMD-specific. They are consumed only when
`GGML_USE_CPU_REPACK` is enabled. The repack buffer type's `supports_op`
(`repack.cpp:4773-4776`) checks the buffer type and op type only, not the
CPU vendor.

### 6.3 Activation conversion (`wdata`)

Same as ARTX02 §6.3. The `quantize_row_q8_K` placeholder
(`quants.c:505-507`) is a scalar fallback on AMD just as on Intel
(cross-ref ARTX02-F12). For Q8_0 activation quantization, the AVX2 path
at `quants.c:302-398` runs — same code on both vendors.

---

## 7. Memory Layout

### 7.1 Per-block layout in `quants.c`

Same as ARTX02 §7.1. Sequential block strides, unaligned loads
(`_mm256_loadu_si256`), no software prefetch on the AVX2 path. The SSSE3
fallback at `quants.c:782, 800` has `_mm_prefetch(..., _MM_HINT_T0)`, but
the AVX2 path drops it. Zen 4 and Zen 5 have capable hardware
prefetchers, but the absence of software prefetch is a vendor-neutral
gap, not AMD-specific.

### 7.2 8×8 batched layout in `repack.cpp`

Same as ARTX02 §7.2: 16 `__m512` accumulators live in registers across
the inner loop. On Zen 4 / Zen 5 (32 AVX-512 architectural registers
available), the 16 accumulators fit comfortably. The uop-split concern
(F03) is not a register-pressure issue — it is a decode/fetch bandwidth
issue.

### 7.3 `tinyBLAS_Q0_AVX` layout

The `tinyBLAS_Q0_AVX` template (`sgemm.cpp:1351`) uses `__m256i` and
`__m256` exclusively. On AVX-512 builds (`VECTOR_REGISTERS == 32` at
line 67), `mnpack` (line 1376-1459) selects wider tile shapes (4×4,
4×3, 3×4, 3×3) than on 16-register AVX2 builds. The accumulators are
`__m256` per (m, n) cell of the tile — up to 4×4 = 16 accumulator
registers, plus 4 input registers and a handful of constants. Total:
~24 registers, comfortably within 32. This is the AMD-friendly layout.

### 7.4 Precomputed tables

`ggml_table_f32_f16[1<<16]` (256 KB), `ggml_table_f32_e8m0_half[1<<8]`
(1 KB), `ggml_table_f32_ue4m3[1<<8]` (1 KB), `ggml_table_gelu_f16[1<<16]`
(128 KB), `ggml_table_gelu_quick_f16[1<<16]` (128 KB) — all vendor-
neutral. On AMD, F16C is always available (since AVX implies F16C), so
the FP16 conversion goes through `_cvtsh_ss` (`simd-mappings.h:61`), not
the 256 KB LUT. Same as Intel.

---

## 8. Parallelism Strategy

The kernels themselves are single-threaded. Threading is layered above
in `ggml_compute_forward_mul_mat` (ARTX01 §5.5, §8.4). The chunk
scheduler, the per-node barrier, and the NUMA-aware chunking fallback
are all vendor-neutral.

One AMD-relevant note: AMD EPYC 9xxx series are typically multi-chiplet,
multi-NUMA-node parts. The NUMA-aware chunking fallback at
`ggml-cpu.c:1413-1417` (ARTX01-F09) switches to one-chunk-per-thread
when `ggml_is_numa()` is true. On a 12-chiplet EPYC 9654, this means
each chiplet's L3 cache holds a disjoint slice of the weights, and the
chunk-per-thread split aligns chunks to chiplet boundaries (assuming
NUMA affinity is set correctly). This is the correct behavior for AMD
multi-chiplet parts — but it is not an AMD-specific code path. It is the
generic NUMA path that happens to be important on AMD.

The `nrc == 1` assertion in every x86 vecdot (cross-ref ARTX02-F10)
applies to AMD as well. AMD has no equivalent of ARM I8MM's multi-row
consumption. The `nrows` in `type_traits_cpu` is 1 for every x86 quant
on every variant.

---

## 9. SIMD / GPU Strategy

This section is the core of the AMD audit.

### 9.1 SIMD feature matrix on AMD

| Feature                | Zen 4 (loads `icelake` .so)            | Zen 5 (loads `zen4` .so)               |
| ---------------------- | --------------------------------------- | --------------------------------------- |
| AVX2 + FMA + F16C      | yes                                     | yes                                     |
| AVX-VNNI (256-bit)     | yes (CPUID bit set, but redundant)      | yes (same)                              |
| AVX-512 F+DQ+BW+VL+CD  | yes                                     | yes                                     |
| AVX-512 VBMI           | yes                                     | yes                                     |
| AVX-512 VNNI           | yes                                     | yes                                     |
| AVX-512 BF16           | **no** (Zen 4 lacks it)                 | yes                                     |
| AVX-512 FP16           | no (variant doesn't set `__AVX512FP16__`)| yes (variant sets `__AVX512FP16__` via `-mavx512f` cascade) |
| AMX_INT8               | no                                      | no                                      |

> Caveat: the table reflects what the loaded `.so` *compiles with*, not
> what the silicon supports. The `zen4` variant does not explicitly set
> `-mavx512fp16` in `CMakeLists.txt:336-355` — only `-mavx512f`,
> `-mavx512cd`, `-mavx512vl`, `-mavx512dq`, `-mavx512bw`,
> `-mavx512vbmi`, `-mavx512vnni`, `-mavx512bf16`. Whether
> `__AVX512FP16__` is defined depends on the compiler: clang defines it
> transitively when `-mavx512f` is set on some toolchains, gcc does not.
> The FP16 path in `vec.cpp:ggml_vec_dot_f16` is therefore only
> conditionally taken on Zen 5 — see Unknowns U4.

### 9.2 The three compile-time ladders (vendor-neutral, but AMD-relevant)

Each int8 dot-product helper has a 3-way compile-time ladder. The branch
taken on AMD is bolded.

**`quants.c:105-119` `mul_sum_us8_pairs_float` (256-bit width):**

| `#if` branch                            | Instruction                | AMD?                                   |
| --------------------------------------- | -------------------------- | -------------------------------------- |
| `__AVX512VNNI__ && __AVX512VL__`        | `_mm256_dpbusd_epi32`      | **taken on Zen 4 and Zen 5**           |
| `__AVXVNNI__`                           | `_mm256_dpbusd_avx_epi32`  | taken only on Alder Lake (Intel)       |
| `#else`                                 | `_mm256_maddubs_epi16` + `madd` | taken on AVX2-only builds (Zen 3, Haswell) |

**`repack.cpp:152-161` `mul_sum_us8_pairs_acc_int32x8` (256-bit width):**
Identical ladder to the above. **Same branch taken on AMD.**

**`repack.cpp:123-131` `mul_sum_us8_pairs_acc_int32x16` (512-bit width):**

| `#if` branch             | Instruction                | AMD?                                                  |
| ------------------------ | -------------------------- | ----------------------------------------------------- |
| `__AVX512VNNI__`         | `_mm512_dpbusd_epi32`      | **taken on Zen 4 and Zen 5 — splits into 2 uops**     |
| `#else`                  | `_mm512_maddubs_epi16` + `madd` | would be taken on Skylake-X (no VNNI)            |

The 256-bit ladder is AMD-optimal. The 512-bit ladder is AMD-suboptimal.
The 8×8 batched GEMM in `repack.cpp:663+` uses the 512-bit ladder
exclusively when `__AVX512BW__ && __AVX512DQ__` are defined. See F03.

### 9.3 BF16 paths on AMD

| File           | Path                                          | Instruction        | Width   | AMD execution                     |
| -------------- | --------------------------------------------- | ------------------ | ------- | --------------------------------- |
| `vec.cpp:148`  | `__AVX512BF16__` branch in `ggml_vec_dot_bf16` | `_mm512_dpbf16_ps` | 512-bit | Zen 5: 2 uops; Zen 4: falls to `__AVX512F__` branch |
| `vec.cpp:160`  | `__AVX512F__` branch                          | convert + `_mm512_mul_ps` + `add` | 512-bit | **Zen 4**: 512-bit FP32 emulation, 2 uops each |
| `vec.cpp:172`  | `__AVX2__` / `__AVX__` branch                 | convert + `_mm256_mul_ps` + `add` | 256-bit | Zen 3 and earlier (no AVX-512)    |
| `sgemm.cpp:151-155` | `madd<__m512bh, __m512bh, __m512>` template  | `_mm512_dpbf16_ps` | 512-bit | Zen 5 BF16 GEMM: 2 uops per FMA   |
| `sgemm.cpp:156-159` | `madd<__m256bh, __m256bh, __m256>` template  | `_mm256_dpbf16_ps` | 256-bit | **never instantiated** — dead on AMD |

The 256-bit BF16 specialization exists in `sgemm.cpp` but is not
instantiated by any dispatch site. The BF16 GEMM at `sgemm.cpp:3788-3795`
instantiates only the 512-bit template. See F05.

### 9.4 FP16 paths on AMD

| File               | Path                                      | Instruction        | Width   | AMD execution                                       |
| ------------------ | ----------------------------------------- | ------------------ | ------- | --------------------------------------------------- |
| `vec.cpp:357`      | `GGML_F16_VEC_FMA` macro                  | `_mm512_fmadd_ph`  | 512-bit | Zen 5: 2 uops (if `__AVX512FP16__` is defined)      |
| `vec.cpp:357`      | `GGML_F16_VEC_FMA` macro (fallback)       | `_mm512_fmadd_ps`  | 512-bit | Zen 4: convert-then-FMA at 512-bit, 2 uops per FMA  |
| `sgemm.cpp:363-365`| `load<ggml_fp16_t> → __m512`              | `_mm512_cvtph_ps`  | 512-bit | All Zen: 2 uops                                     |
| `quants.c`         | per-block FP16 scale broadcast            | `GGML_CPU_FP16_TO_FP32` (scalar `_cvtsh_ss`) | scalar | All Zen: scalar per block, broadcast via `_mm256_set1_ps` |

`vec.cpp:ggml_vec_dot_f16` uses `GGML_F16_VEC_FMA` from
`simd-mappings.h:493-578`. When `__AVX512FP16__` is defined, the macro
resolves to `_mm512_fmadd_ph` — a 512-bit native FP16 FMA, 32 FP16
lanes per `__m512h` accumulator. When `__AVX512FP16__` is not defined
(e.g. on the `icelake` variant that loads on Zen 4), the macro falls back
to `_mm512_fmadd_ps` after converting FP16→FP32 via `_mm512_cvtph_ps`.

This **contradicts ARTX02-F08** in scope: ARTX02-F08 claimed "No AVX-512
FP16 dot products anywhere in the audited files." That was correct for
the three files ARTX02 audited (`quants.c`, `repack.cpp`, `sgemm.cpp`).
But `vec.cpp` and `simd-mappings.h` DO use AVX-512 FP16 when defined.
The F16×F16 vecdot path on Zen 5 (with the `zen4` variant .so, assuming
`__AVX512FP16__` is defined by the compiler) executes
`_mm512_fmadd_ph` — which on Zen 5's 256-bit data path splits into two
256-bit uops. The 256-bit `_mm256_fmadd_ph` (also part of AVX-512 FP16 +
VL) would be more efficient, but `vec.cpp` does not use it. See F08.

### 9.5 VNNI / AVX-VNNI / AVX-VNNI_INT8 usage on AMD

| Instruction                | ISA                       | AMD execution                                                     |
| -------------------------- | ------------------------- | ----------------------------------------------------------------- |
| `_mm256_dpbusd_epi32`      | AVX-512 VNNI + VL         | **Zen 4 / Zen 5: native, 256-bit, 1 uop, 1-cycle throughput**    |
| `_mm256_dpbusd_avx_epi32`  | AVX-VNNI (no AVX-512)     | Never reached on AMD (Zen 4+ take the AVX-512 branch first)       |
| `_mm256_dpbssd_epi32`      | AVX-VNNI_INT8             | Never reached (no variant enables `__AVXVNNIINT8__`)              |
| `_mm512_dpbusd_epi32`      | AVX-512 VNNI              | **Zen 4 / Zen 5: 2 uops (split)**; 2-cycle effective throughput   |
| `_mm512_dpbf16_ps`         | AVX-512 BF16              | **Zen 5: 2 uops (split)**; 2-cycle effective throughput           |

The 256-bit VNNI instruction is the AMD-optimal choice and is what the
per-block vecdot helper actually uses. The 512-bit VNNI instruction is
what the 8×8 batched GEMM uses, and it is AMD-suboptimal.

### 9.6 AMX (Intel-only, correctly excluded on AMD)

The `amx/` subdirectory is conditionally compiled under
`#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)` (`amx/amx.cpp:19`).
The `sapphirerapids` variant (line 401) is the only x86 variant that
defines `__AMX_INT8__`. On AMD:

1. The `sapphirerapids` variant `.so` returns score=0
   (`cpu-feats.cpp:318`: `if (!is.AMX_INT8()) { return 0; }`) because AMD
   CPUs do not have the AMX_TILE / AMX_INT8 CPUID bits.
2. The `.so` is never loaded; `amx/amx.cpp` produces an empty
   translation unit.
3. The `ggml::cpu::amx::extra_buffer_type` is never registered.

The exclusion is correct. The architectural consequence is that AMD has
no equivalent to Intel's tile-matrix-multiply unit. The only path to
high-throughput int8 matmul on AMD is AVX-512 VNNI at 256-bit width
(one `_mm256_dpbusd_epi32` per cycle per core, 8 int32 multiplies per
instruction = 8 int32 MACs/cycle/core). Intel AMX delivers ~64 int32
MACs/cycle/core (TMM0..TMM7, 16-row × 64-byte tiles). The 8× throughput
asymmetry is real and unaddressed in the codebase. See F07.

---

## 10. Quantization Strategy

The quant format list, block sizes, scale handling, and zero-point
encoding are all vendor-neutral (cross-ref ARTX02 §10, ARTX06 for the
full format catalog). AMD inherits:

* Q4_0/Q4_1/Q5_0/Q5_1/Q8_0: 32-element blocks, fp16 scale.
* Q2_K/Q3_K/Q4_K/Q5_K/Q6_K: 256-element blocks, 6-bit fp16 scale + 4-bit
  sub-block scales.
* IQ2_XXS / IQ2_XS / IQ2_S / IQ3_XXS / IQ3_S / IQ1_S / IQ1_M / IQ4_XS:
  inference-only (no runtime `from_float`).
* MXFP4 / NVFP4: 32-element blocks with E8M0 / UE4M3 per-block scales,
  looked up from 256-entry LUTs (`simd-mappings.h:127-139`).

The activation conversion path (`quantize_row_q8_0`, `quantize_row_q8_1`,
`quantize_row_q8_K`) is the same on AMD as on Intel. The `quantize_row_q8_K`
placeholder (`quants.c:505-507`) is a scalar fallback on AMD as well
(cross-ref ARTX02-F12).

The `from_float` function pointer in `type_traits_cpu[GGML_TYPE_BF16]`
(`ggml-cpu.c:396`) points to `ggml_cpu_fp32_to_bf16`. The `vec_dot`
pointer points to `ggml_vec_dot_bf16` in `vec.cpp:139`. On Zen 5 (with
the `zen4` variant .so that defines `__AVX512BF16__`), the BF16 vecdot
uses `_mm512_dpbf16_ps` at 512-bit width. On Zen 4 (with the `icelake`
variant .so that defines `__AVX512F__` only), the BF16 vecdot uses the
FP32-emulation path at 512-bit width (`vec.cpp:160-169`). Neither path
uses the 256-bit `_mm256_dpbf16_ps` (which requires `__AVX512BF16__` and
is available on Zen 5).

The `from_float` for `GGML_TYPE_F16` (`ggml-cpu.c:222`) points to
`ggml_cpu_fp32_to_fp16`. The `vec_dot` points to `ggml_vec_dot_f16` in
`vec.cpp:264`. On Zen 5 with `__AVX512FP16__` defined, the F16 vecdot
uses `_mm512_fmadd_ph` at 512-bit width. Otherwise it converts to FP32
and uses `_mm512_fmadd_ps`. Neither uses the 256-bit `_mm256_fmadd_ph`.

---

## 11. Correctness Analysis

### 11.1 Floating-point reassociation

Same as ARTX02 §11.1. Every vecdot kernel in `quants.c` and `vec.cpp`
accumulates into one or more vector accumulators and horizontally reduces
at the end. The reduction order differs from a strict left-to-right
scalar sum at the ULP level. Deterministic for a fixed `n`, fixed
compile-time ISA, and fixed `nth=1`. Non-deterministic across runs when
`nth > 1` due to dynamic chunk stealing (ARTX01-F06).

### 11.2 The `zen4` variant naming defect (correctness-relevant)

`CMakeLists.txt:396` defines:

```
ggml_add_cpu_backend_variant(zen4  SSE42 AVX F16C FMA AVX2 BMI2 AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
```

The `AVX512_BF16` flag causes `cpu-feats.cpp:309-311` to return 0 on
actual Zen 4 hardware (which lacks AVX-512 BF16). The variant is
therefore never loaded on Zen 4. The fallback is the `icelake` variant.
This is **not a correctness bug** — the `icelake` variant is a strict
subset of what Zen 4 supports, and it runs correctly. But the variant
name is misleading, and the build matrix delivers fewer AMD-tuned
binaries than the variant list suggests. See F02.

### 11.3 Approximate math

Same as ARTX02 §11.2. No AMD-specific approximate math. The E8M0 / UE4M3
LUTs and the GELU f16 LUTs are vendor-neutral.

### 11.4 Precision reduction

Same as ARTX02 §11.3. The BF16 / F16 vecdot paths accumulate in FP32
(per Intel/AMD ISA specification for `_mm512_dpbf16_ps` and
`_mm512_fmadd_ph`). No precision reduction beyond the storage format.

### 11.5 Non-deterministic reductions

Same as ARTX01 §11.4. Matmul output is deterministic bit-for-bit only
when `nth = 1`. Vendor-neutral.

### 11.6 Atomic accumulation

None in any audited kernel. Output tiles are written by exactly one
thread each. Vendor-neutral.

### 11.7 Architecture-specific assumptions

* `assert(nrc == 1)` in every x86 vecdot (cross-ref ARTX02-F10). Applies
  to AMD.
* `assert(n % qk == 0)` in every K-quant vecdot. Applies to AMD.
* AVX2 `packs` lane-shuffle permute constant
  (`quants.c:364, 463`). Applies to AMD.
* The OS XSAVE check FIXME (`cpu-feats.cpp:264`) applies to AMD. Modern
  Linux/macOS/Windows kernels enable XSAVE for AVX-512 on AMD Zen 4/Zen 5
  just as on Intel. The FIXME is vendor-neutral.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations (AMD-relevant subset)

| Optimization                                | Where                                       | AMD relevance                                                                |
| ------------------------------------------- | ------------------------------------------- | ---------------------------------------------------------------------------- |
| Multi-binary ISA dispatch                   | `cpu-feats.cpp:263-323`                     | Yes — picks `icelake` (Zen 4) or `zen4` (Zen 5). Cross-ref ARTX02-F09.       |
| 256-bit AVX-512 VNNI in vecdot helper       | `quants.c:106-118`, `repack.cpp:152-161`    | **AMD-optimal** — native 256-bit uop, no split. See F04.                     |
| `tinyBLAS_Q0_AVX` 256-bit + 32 registers    | `sgemm.cpp:1351-1794`                       | **AMD-friendly by design** — 256-bit data path + 32 accumulators. See F06.   |
| 8×8 batched GEMM with 16 × `__m512`          | `repack.cpp:663-1096`                       | AMD-suboptimal — 512-bit uops split. See F03.                                |
| BF16 native dot (`_mm512_dpbf16_ps`)        | `sgemm.cpp:151-155, 3790`; `vec.cpp:152-155`| AMD-suboptimal on Zen 5 — 512-bit uop splits; 256-bit version unused. See F05. |
| FP16 native FMA (`_mm512_fmadd_ph`)         | `vec.cpp:357` via `simd-mappings.h:503`     | AMD-suboptimal on Zen 5 — 512-bit uop splits; 256-bit version unused. See F08. |
| FMA via `_mm256_fmadd_ps` / `_mm512_fmadd_ps` | `quants.c:738, 1339, 2113, 2505`         | 256-bit FMA on vecdot; 512-bit FMA on batched GEMM.                          |
| Type-traits function pointer                | `ggml-cpu.c:214`                            | Vendor-neutral. Cross-ref ARTX01-F03.                                         |
| Cache-aligned atomics                       | `ggml-cpu.c:489-491`                        | Vendor-neutral. Cross-ref ARTX01.                                            |
| NUMA-aware chunking                         | `ggml-cpu.c:1413-1417`                      | **Important on AMD multi-chiplet EPYC.** Cross-ref ARTX01-F09.               |

### 12.2 Optimizations *not* present (AMD-specific gaps)

* **No 256-bit-preferred compile-time switch.** When `__AVX512F__` is
  defined, the code uses 512-bit instructions in `repack.cpp` 8×8 GEMM
  and in `llamafile/sgemm.cpp` F32/BF16 GEMM templates. There is no
  `#if defined(__AMD_ZEN_4__)` or `#if defined(__AVX10_256__)` branch
  that would prefer the 256-bit equivalents. See F03.
* **No AMD vendor detection in any kernel.** `is_amd` is set in
  `cpu-feats.cpp:135` and consulted only for legacy feature gates.
  No kernel reads it. See F01.
* **No AMD-specific tile-multiply unit.** AMX is Intel-only. AMD's
  highest-throughput int8 path is AVX-512 VNNI at 256-bit width, ~8×
  slower than Intel AMX. The codebase does not document or compensate
  for this asymmetry. See F07.
* **No AVX-VNNI_INT8 variant.** No x86 variant in `CMakeLists.txt:378-402`
  enables `__AVXVNNIINT8__`. The signed-VNNI path at `quants.c:123-126`
  and `repack.cpp:166-167` is dead in the standard build matrix.
  Cross-ref ARTX02-F04 — this is vendor-neutral but worth re-noting
  because Zen 5 was rumored to add AVX-VNNI_INT8 (it does not; only
  Granite Rapids does).
* **No software prefetching on x86 vecdot kernels** (except the legacy
  SSSE3 Q4_0 path). The `tinyBLAS_Q0_AVX` template also has no prefetch.
  Cross-ref ARTX02-W7.
* **No persistent threads, no async execution, no kernel fusion** in
  the kernel layer. Vendor-neutral; cross-ref ARTX01.

---

## 13. Architectural Strengths

1. **`tinyBLAS_Q0_AVX` is the correct pattern for Zen.** The template
   uses `__m256` width throughout (no 512-bit uop splits), harvests the
   32-register file via `VECTOR_REGISTERS == 32` to pick wider tile
   shapes (4×4, 4×3, 3×4), and dispatches the int8 dot product through
   the 3-way VNNI ladder to `_mm256_dpbusd_epi32` on Zen 4/Zen 5. This
   is exactly the right design for a 256-bit-data-path AVX-512 CPU.
   Worth **ADOPT** in `glproc`. See F06.

2. **The 256-bit VNNI helper ladder is AMD-optimal by accident.**
   `mul_sum_us8_pairs_float` (`quants.c:105-119`) and its
   `repack.cpp:152-161` sibling both pick the 256-bit `_mm256_dpbusd_epi32`
   instruction on Zen 4/Zen 5 because `__AVX512VNNI__ + __AVX512VL__`
   are defined and are checked first in the `#if` ladder. The choice
   is correct for Zen, but it is correct because the helper was designed
   for AVX2-width vecdots, not because of any Zen-aware tuning. Either
   way, the result is the right instruction. Cross-ref ARTX02-F02, F03.

3. **Multi-binary dispatch correctly excludes AMX on AMD.** The
   `sapphirerapids` variant returns score=0 on AMD (no AMX_INT8 bit),
   so the AMX plugin's `tensor_traits` is never registered. No AMD user
   accidentally triggers an AMX `#UD` fault. The exclusion is silent
   but correct.

4. **NUMA-aware chunking matters on AMD EPYC.** The one-chunk-per-thread
   fallback (`ggml-cpu.c:1413-1417`) is critical for multi-chiplet EPYC
   parts. The codebase inherits this from the vendor-neutral NUMA path
   (ARTX01-F09), but its value on AMD is high — without it, a 12-chiplet
   EPYC 9654 would suffer cross-chiplet cache traffic on every matmul.

5. **The `icelake` variant falls back gracefully on Zen 4.** When the
   `zen4` variant fails to load (no AVX-512 BF16), the `icelake` variant
   is the next-best match. The `icelake` variant is a strict subset of
   Zen 4's capabilities, so it runs correctly. The misnamed `zen4`
   variant is a configuration defect, not a crash bug. See F02.

---

## 14. Architectural Weaknesses

### W1 — No AMD-specific kernel decisions

**Evidence:** Grep for `is_amd`, `__zn`, `ZEN`, `zen4`, `zen5`, `AVX10`,
`family`, `0x19`, `0x1a` across `ggml/src/ggml-cpu/` returns hits only
in `cpu-feats.cpp:68-77` (legacy feature gates) and the CMake `zen4`
variant name. Zero kernel decisions consult AMD vendor info.

**Impact:** AMD and Intel run identical code. Where the optimal
instruction width differs (512-bit on Intel native, 256-bit on Zen
native), the code uniformly picks 512-bit. The 256-bit `_mm256_*`
equivalents exist as template specializations in `sgemm.cpp:156-159`
but are not instantiated.

**Why it's hard to fix:** Requires either (a) compile-time macros
recognizing Zen (`-march=znver4`, `-march=znver5`) and a corresponding
variant in CMakeLists, or (b) runtime `is_amd` checks plumbed into
kernel dispatch. Option (a) is cleaner but requires the build matrix to
add `zen4-256` and `zen5-256` variants. Option (b) breaks the
multi-binary dispatch model.

### W2 — The `zen4` variant is misnamed and misconfigured

**Evidence:** `CMakeLists.txt:396` defines the `zen4` variant with
`AVX512_BF16`. Zen 4 hardware lacks AVX-512 BF16 (added in Zen 5). The
score function (`cpu-feats.cpp:309-311`) returns 0 for this variant on
actual Zen 4 silicon.

**Impact:** Zen 4 falls back to the `icelake` variant. The `zen4`
variant is effectively a Zen 5 binary. The build matrix advertises a
Zen 4 variant that never loads on Zen 4. This obscures the actual
AMD delivery: one binary (`icelake`) for Zen 4, one binary (`zen4`,
misnamed) for Zen 5.

**Why it's hard to fix:** Renaming `zen4` to `zen5` would break
existing user expectations and downstream packager scripts. Splitting
into `zen4` (without BF16) and `zen5` (with BF16) is the correct fix
but doubles the build matrix entry count.

### W3 — 512-bit instructions on Zen 4/Zen 5 split into two 256-bit uops

**Evidence:** `repack.cpp:663+` (8×8 batched GEMM, 16 × `__m512`
accumulators, `_mm512_dpbusd_epi32`); `sgemm.cpp:3727` (F32 GEMM,
`tinyBLAS<16, __m512, …>`); `sgemm.cpp:3790` (BF16 GEMM,
`tinyBLAS<32, __m512, __m512bh, …>`); `vec.cpp:152-155` (BF16 vecdot,
`_mm512_dpbf16_ps`); `vec.cpp:357` (F16 vecdot, `_mm512_fmadd_ph`).

**Impact:** On Zen 4/Zen 5's 256-bit data path, every 512-bit
instruction is decoded into two 256-bit uops. The throughput is at best
equal to the 256-bit equivalent (no benefit from 512-bit width), and
the cost is doubled decode/fetch bandwidth plus mild downclocking.
Where the 256-bit equivalent exists (`_mm256_dpbf16_ps`,
`_mm256_fmadd_ph`, `_mm256_dpbusd_epi32`), the code does not select it.

**Why it's hard to fix:** Each kernel needs a compile-time branch
preferring 256-bit width on Zen. The 256-bit BF16 / FP16 / VNNI
instructions all require AVX-512 + VL, which is already defined on
every AMD variant. The work is in adding the branches and the
alternative tile shapes, not in new instructions.

### W4 — AVX-VNNI fallback path is dead on AMD

**Evidence:** `quants.c:110-113` and `repack.cpp:154-155` define an
`#elif defined(__AVXVNNI__)` branch using `_mm256_dpbusd_avx_epi32`.
This branch is reached only when `__AVXVNNI__` is defined but
`__AVX512VNNI__ + __AVX512VL__` are not. The only AMD CPUs with
AVX-VNNI are Zen 4 and Zen 5, which both also have AVX-512 VNNI + VL.
So the AVX-VNNI fallback is never taken on AMD. The only CPU that
actually uses it is Alder Lake (Intel).

**Impact:** The fallback code is correct but unreachable on AMD. The
Alder Lake `.so` is the only consumer. Not a bug, but a notable
observation: AVX-VNNI is Intel's "AVX-512 VNNI for non-AVX-512 CPUs"
feature, and AMD's Zen 4/5 chose to implement AVX-512 instead of
AVX-VNNI alone.

**Why it's hard to fix:** Not a bug. The fallback exists for Alder
Lake. Leave it alone.

### W5 — AVX-512 FP16 path in `vec.cpp` contradicts ARTX02-F08's scope

**Evidence:** `vec.cpp:357` uses `GGML_F16_VEC_FMA` from
`simd-mappings.h:503` (`_mm512_fmadd_ph` when `__AVX512FP16__` is
defined). ARTX02-F08 claimed "No AVX-512 FP16 dot products anywhere in
the audited files" — that was correct for the three files ARTX02
audited (`quants.c`, `repack.cpp`, `sgemm.cpp`), but `vec.cpp` and
`simd-mappings.h` were not in ARTX02's scope.

**Impact:** On Zen 5 (with the `zen4` variant `.so` and a compiler that
defines `__AVX512FP16__`), the F16 vecdot uses native FP16 FMA at
512-bit width. On Zen 5's 256-bit data path, this splits into two uops.
The 256-bit `_mm256_fmadd_ph` would be more efficient but is not used.
ARTX02-F08's recommendation to add an AVX-512 FP16 dot product should
be amended: the dot product exists in `vec.cpp`, but only at 512-bit
width; the 256-bit variant is missing.

**Why it's hard to fix:** Requires adding a 256-bit FP16 path to
`vec.cpp` and a compile-time switch to prefer it on Zen. The
`simd-mappings.h` macros would need a 256-bit variant.

### W6 — Score function inflates Zen 4 score by 64 vs IceLake-client

**Evidence:** `cpu-feats.cpp:293-296` adds `1<<6 = 64` for AVX-VNNI.
Zen 4 has AVX-VNNI (CPUID bit set) and AVX-512 VNNI + VL. IceLake-client
has AVX-512 VNNI + VL but NOT AVX-VNNI. The AVX-VNNI bit on Zen 4 is
redundant — `_mm256_dpbusd_epi32` (the AVX-512 VNNI + VL instruction)
subsumes `_mm256_dpbusd_avx_epi32` (the AVX-VNNI instruction). Both
produce the same result. But the score adds 64 for the redundant bit.

**Impact:** Zen 4 outscores IceLake-client by 64 in the multi-binary
selection. Since both fall back to the same `icelake` variant binary
(Zen 4 lacks AVX-512 BF16 so the `zen4` variant fails), the inflated
score is harmless in practice. But it misrepresents the capability
gap: Zen 4 and IceLake-client have equivalent VNNI throughput, not a
64-point difference.

**Why it's hard to fix:** Would require the score function to consult
`is_amd` and skip the `AVX_VNNI` bit on CPUs that have `AVX512_VNNI +
AVX512_VL`. Or, simpler: define the `AVX_VNNI` score bit as
`AVX_VNNI && !(AVX512_VNNI && AVX512_VL)`. Cross-ref F09.

### W7 — No `zen5` variant; no AVX-512 FP16 variant

**Evidence:** `CMakeLists.txt:378-402` defines no `zen5` variant. The
`zen4` variant (line 396) is the closest match for Zen 5 hardware but
does not explicitly enable `-mavx512fp16` (lines 336-355 enable only
F/CD/VL/DQ/BW/VBMI/VNNI/BF16). Whether `__AVX512FP16__` is defined
depends on the compiler: clang sometimes defines it transitively from
`-mavx512f` on certain toolchains, gcc does not.

**Impact:** The F16 vecdot path in `vec.cpp:ggml_vec_dot_f16` may or
may not use native FP16 FMA on Zen 5, depending on compiler. This is a
non-deterministic build outcome. See Unknowns U4.

**Why it's hard to fix:** Add an explicit `zen5` variant with
`-mavx512fp16` and the corresponding `GGML_AVX512_FP16` definition.
Or unify by always enabling `-mavx512fp16` when `-mavx512f` is set
(but this would produce SIGILL on CPUs that lack AVX-512 FP16, e.g.
IceLake-client, Sapphire Rapids without FP16).

### W8 — No 256-bit BF16 / FP16 GEMM dispatch

**Evidence:** `sgemm.cpp:156-159` defines the 256-bit BF16 dot product
template specialization `madd<__m256bh, __m256bh, __m256>`, but
`llamafile_sgemm`'s BF16 case at `sgemm.cpp:3788-3795` instantiates
only the 512-bit `tinyBLAS<32, __m512, __m512bh, …>` template. The
256-bit specialization is never instantiated. Same for FP16: no 256-bit
`tinyBLAS` template exists.

**Impact:** On Zen 5, the BF16 GEMM uses 512-bit `_mm512_dpbf16_ps`
which splits into two 256-bit uops. The 256-bit `_mm256_dpbf16_ps`
would execute as a single uop on Zen 5's native 256-bit data path. The
latter is strictly better on Zen 5 but is not selectable.

**Why it's hard to fix:** Add a `#if defined(__AMD_ZEN5__)` branch
(or runtime `is_amd` check) that instantiates `tinyBLAS<16, __m256,
__m256bh, …>` instead. Requires a Zen-detection mechanism that
currently does not exist.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glproc` | **ADOPT** | `tinyBLAS_Q0_AVX` 256-bit + 32-register pattern | The one AMD-friendly design in the codebase. 256-bit data path + 32 architectural registers + 256-bit VNNI. F06. |
| `glproc` | **ADOPT** | 256-bit VNNI helper ladder (`mul_sum_us8_pairs_float`) | Already ADOPT per ARTX02-F02; reaffirm for AMD. The 256-bit `_mm256_dpbusd_epi32` is the AMD-optimal instruction. |
| `glproc` | **ADAPT** | Multi-binary dispatch | Keep the scheme, but fix the `zen4` variant naming and add a real `zen4` variant without AVX-512 BF16. Add a `zen5` variant with AVX-512 BF16 + FP16. F02. |
| `glproc` | **REJECT** | 512-bit-only batched GEMM on Zen | On Zen 4/Zen 5, prefer 256-bit `_mm256_dpbusd_epi32` in the 8×8 batched GEMM, or fall back to `tinyBLAS_Q0_AVX`. F03. |
| `glproc` | **REJECT** | 512-bit-only BF16 GEMM on Zen 5 | Use the 256-bit `_mm256_dpbf16_ps` template specialization that already exists. F05. |
| `glproc` | **REJECT** | 512-bit-only FP16 vecdot on Zen 5 | Add a 256-bit FP16 path. The 256-bit `_mm256_fmadd_ph` is part of AVX-512 FP16 + VL. F08. |
| `glproc` | **MONITOR** | AVX-VNNI fallback (`_mm256_dpbusd_avx_epi32`) | Dead on AMD (Zen 4+ have AVX-512). Keep for Alder Lake. F04. |
| `glproc` | **MONITOR** | AVX-VNNI_INT8 (`_mm256_dpbssd_epi32`) | Cross-ref ARTX02-F04. Never enabled in any x86 variant. Granite Rapids only. |
| `glproc` | **MONITOR** | `is_amd` vendor detection | Currently used only for legacy feature gates. Could be extended to drive 256-bit-preferred kernel selection in glproc. F01. |
| `glproc` | **DEFER** | AMX tile-based matmul | Intel-only. AMD has no equivalent. Do not try to emulate AMX on AMD; use 256-bit AVX-512 VNNI instead. F07. |
| `GATE` | **ADOPT** | Type-traits `nrows` parameter | Cross-ref ARTX02. Vendor-neutral. Allows future multi-row vecdot. |
| `GATE` | **ADAPT** | `llamafile_sgemm` integration | Make selection plan-time, with vendor-aware width choice (256-bit on Zen, 512-bit on Intel). Cross-ref ARTX01-R5. |
| `GATE` | **MONITOR** | NUMA-aware chunking | Important on AMD EPYC multi-chiplet. Already adopted per ARTX01-F09. |

---

## 16. Recommendations

### R1 — ADOPT the `tinyBLAS_Q0_AVX` 256-bit + 32-register pattern
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glproc` should provide an equivalent 256-bit GEMM template that uses `__m256`/`__m256i` width throughout, dispatches the int8 dot product through a 3-way VNNI ladder (256-bit AVX-512 VNNI first, AVX-VNNI fallback, scalar maddubs+madd), and harvests the 32-register file when AVX-512 is enabled to pick wider tile shapes (4×4, 4×3, 3×4). This is the AMD-optimal design and also a reasonable design for Intel IceLake-client (where 256-bit width avoids the downclocking penalty that 512-bit triggers). (F06.)

### R2 — REJECT the 512-bit-only batched GEMM on Zen; provide a 256-bit variant
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
For the 8×8 batched GEMM (the `repack.cpp:663+` pattern), GwenLand should provide a 256-bit variant selected when running on AMD Zen 4/Zen 5. The 256-bit variant uses `_mm256_dpbusd_epi32` (already in the helper ladder) and 32 × `__m256` accumulators (vs. 16 × `__m512`). The tile shape can stay 16 rows × 8 cols; only the SIMD width changes. Expected: equivalent throughput to the 512-bit version on Zen (because Zen's 512-bit splits), with lower decode pressure and no downclocking. (F03.)

### R3 — ADOPT 256-bit BF16 dot product for Zen 5
**Priority:** High
**Difficulty:** S
**Dependencies:** R1
The 256-bit `_mm256_dpbf16_ps` template specialization already exists at `sgemm.cpp:156-159`. GwenLand should add a dispatch branch that instantiates `tinyBLAS<16, __m256, __m256bh, …>` on Zen 5 instead of the 512-bit `tinyBLAS<32, __m512, __m512bh, …>`. Same for `ggml_vec_dot_bf16` in `vec.cpp` — add a 256-bit path before the 512-bit `__AVX512BF16__` branch. (F05.)

### R4 — ADOPT 256-bit FP16 dot product for Zen 5
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R3
GwenLand should provide a 256-bit FP16 vecdot path using `_mm256_fmadd_ph` (AVX-512 FP16 + VL). The current `vec.cpp:ggml_vec_dot_f16` uses `_mm512_fmadd_ph` at 512-bit width, which splits on Zen 5. Add a `simd-mappings.h` 256-bit FP16 macro block and a `vec.cpp` 256-bit branch. This also corrects the scope of ARTX02-F08: the FP16 dot product exists in `vec.cpp`, but only at 512-bit width. (F08.)

### R5 — ADAPT the multi-binary dispatch: rename `zen4` to `zen5`, add a real `zen4`
**Priority:** High
**Difficulty:** S
**Dependencies:** none
GwenLand should split the `zen4` variant into two: a real `zen4` variant without `AVX512_BF16` (loads on Zen 4 silicon) and a `zen5` variant with `AVX512_BF16` (loads on Zen 5 silicon). The current `zen4` variant never loads on Zen 4 because Zen 4 lacks AVX-512 BF16. Also add `AVX512_FP16` to the `zen5` variant explicitly (via `-mavx512fp16`). (F02, F10.)

### R6 — ADAPT the score function: de-duplicate AVX_VNNI when AVX512_VNNI + VL is present
**Priority:** Low
**Difficulty:** XS
**Dependencies:** R5
Change `cpu-feats.cpp:293-296` so the `AVX_VNNI` score bit is added only when `AVX_VNNI` is set AND `AVX512_VNNI + AVX512_VL` is NOT set. This removes the spurious 64-point inflation on Zen 4 and Cooper Lake. (F09.)

### R7 — ADOPT `is_amd` as a first-class kernel-selection signal
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R5
GwenLand's `glproc` should extend `is_amd` beyond legacy feature gates. The score function should record the vendor in a struct field accessible to kernel dispatch. Kernel templates that have 256-bit and 512-bit variants should consult vendor + microarchitecture to pick the optimal width. The cleanest design is a `glproc_cpu_traits` struct populated at load time with vendor, family, model, and a `prefer_256bit_avx512` boolean. (F01.)

### R8 — DEFER AMX-equivalent int8 tile-multiply on AMD
**Priority:** Low
**Difficulty:** XL
**Dependencies:** none
AMD has no AMX-equivalent tile-matrix-multiply unit. The highest-throughput int8 matmul on AMD is AVX-512 VNNI at 256-bit width (~8 int32 MACs/cycle/core vs. Intel AMX's ~64). GwenLand should not attempt to emulate AMX on AMD; instead, document the asymmetry and provide a 256-bit VNNI path that scales with core count. For very large int8 GEMM, AMD's advantage is core count (96-core EPYC 9654 vs. 56-core Xeon Platinum 8480+), not per-core throughput. (F07.)

### R9 — MONITOR AVX-VNNI_INT8 and AVX10
**Priority:** Low
**Difficulty:** S
**Dependencies:** none
AVX-VNNI_INT8 (`_mm256_dpbssd_epi32`) is plumbed but never enabled in any x86 variant. AVX10.1 / AVX10.2 are not yet referenced in the codebase. GwenLand should monitor whether future AMD Zen 6 or Intel Clearwater Forest parts introduce new int8 / FP16 dot product instructions, and add corresponding helper-ladder branches. (Cross-ref ARTX02-F04.)

### R10 — ADOPT the NUMA-aware chunking fallback for AMD EPYC
**Priority:** Medium
**Difficulty:** M
**Dependencies:** none
Already adopted per ARTX01-F09. Reaffirm for AMD: multi-chiplet EPYC parts (9xxx, 90xx series) benefit significantly from the one-chunk-per-thread fallback when `ggml_is_numa()` is true. GwenLand should keep this fallback and consider extending it to per-chiplet chunking (one chunk per L3 slice, not per NUMA node). (Cross-ref ARTX01-F09.)

---

## 17. Findings

### Finding ARTX03-F01

```
Finding ID:           ARTX03-F01
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Vendor detection
Source File:          ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
Function:             cpuid_x86 constructor; ggml_backend_cpu_x86_score
Lines:                132-136 (vendor detection); 68-77 (legacy feature gates); 263-323 (score)
Summary:              is_amd is set from the "AuthenticAMD" vendor string but is consulted only
                      to gate seven legacy AMD feature flags (ABM, SSE4a, XOP, TBM, MMXEXT,
                      3DNOWEXT, 3DNOW). No kernel decision, no score differentiation, and no
                      compile-time branch anywhere in ggml/src/ggml-cpu/ reads is_amd.
Observation:          The vendor string is captured at cpu-feats.cpp:126-136. The is_amd flag
                      is set true when vendor == "AuthenticAMD" (line 134-135). The only
                      consumers of is_amd are the seven legacy feature gates at lines 68-77
                      (ABM, SSE4a, XOP, TBM, MMXEXT, 3DNOWEXT, 3DNOW) — all AMD-specific
                      extensions that predate Zen and are irrelevant to AVX-512 / VNNI / BF16
                      kernel selection.

                      The score function (lines 263-323) does not consult is_amd. Every
                      GGML_AVX512* and GGML_AVX_VNNI score branch reads only the CPUID bit
                      (e.g. is.AVX512_BF16() at line 310), not the vendor. The result is that
                      Zen 4 and Zen 5 are scored identically to Intel CPUs with the same
                      CPUID bits.

                      Grep across ggml/src/ggml-cpu/ for is_amd, __zn, ZEN, zen4, zen5, AVX10,
                      family, 0x19, 0x1a returns no kernel-level hits. The only AMD-aware
                      code in the entire CPU backend is the seven legacy feature gates.
Evidence:             cpu-feats.cpp:132-136 (is_amd set from vendor string);
                      cpu-feats.cpp:68-77 (only consumers — legacy feature gates);
                      cpu-feats.cpp:263-323 (score function, no is_amd reference).
Architectural Impact: AMD Zen 4 / Zen 5 receive zero vendor-specific tuning. The codebase
                      treats them as "AVX-512 + AVX-VNNI + BF16 capable IceLake-like" CPUs.
                      Where the optimal instruction width differs (512-bit on Intel native,
                      256-bit on Zen native), the code uniformly picks 512-bit. The 256-bit
                      _mm256_* equivalents exist as template specializations but are not
                      instantiated.
Correctness Impact:   None. Vendor-neutral code is correct on AMD.
Optimization Type:    None (absence of vendor-aware optimization).
GwenLand Target:      glproc
Recommendation:       ADAPT. Extend is_amd beyond legacy feature gates. GwenLand's glproc
                      should populate a glproc_cpu_traits struct at load time with vendor,
                      family, model, and a prefer_256bit_avx512 boolean. Kernel templates
                      with 256-bit and 512-bit variants should consult this struct to pick
                      the optimal width.
Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX03-F02

```
Finding ID:           ARTX03-F02
Category:             LAYOUT_SUBOPTIMAL
Engine:               CPU
Component:            x86 build variant matrix
Source File:          ggml/src/CMakeLists.txt
Function:             GGML_CPU_ALL_VARIANTS block
Lines:                395-396 (cooperlake and zen4 variant definitions); 390 (icelake)
Summary:              The zen4 variant is built with AVX512_BF16 enabled, but Zen 4 hardware
                      does NOT have AVX-512 BF16 (only Zen 5 added it). The score function
                      returns 0 for the zen4 variant on actual Zen 4 silicon, so Zen 4 falls
                      back to the icelake variant. The zen4 variant is effectively a Zen 5
                      binary. There is no Zen 4-specific .so.
Observation:          Line 396:
                        ggml_add_cpu_backend_variant(zen4  SSE42 AVX F16C FMA AVX2 BMI2
                                                     AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16)
                      The AVX512_BF16 flag causes cpu-feats.cpp:309-311 to test
                      is.AVX512_BF16(), which is false on Zen 4 silicon. The function
                      returns 0. The variant is never loaded on Zen 4.

                      The fallback is the icelake variant (line 390), which is a strict
                      subset of Zen 4 capabilities (no AVX512_BF16). Zen 4 runs the icelake
                      binary correctly.

                      On Zen 5, AVX512_BF16 IS present, so the zen4 variant loads. Zen 5
                      runs the zen4 binary — which is identical to the cooperlake binary
                      (line 395) except for the addition of AVX512_VBMI. There is no Zen
                      5-specific code in the zen4 variant; it is just a different set of
                      compile-time macros.

                      Net effect: there is one AMD-tuned variant in the build matrix (zen4,
                      misnamed), and it loads only on Zen 5. Zen 4 runs the Intel-targeted
                      icelake variant. Neither variant makes any AMD-specific kernel decision.
Evidence:             CMakeLists.txt:396 (zen4 variant definition with AVX512_BF16);
                      cpu-feats.cpp:309-311 (score returns 0 if AVX512_BF16 missing);
                      CMakeLists.txt:390 (icelake variant — Zen 4 fallback).
Architectural Impact: The build matrix advertises a zen4 variant that never loads on Zen 4
                      hardware. The actual AMD delivery is: icelake binary for Zen 4, zen4
                      binary (misnamed) for Zen 5. No AMD-specific kernel decisions exist
                      in either binary.
Correctness Impact:   None. The icelake variant is a strict subset of Zen 4 capabilities
                      and runs correctly. The zen4 variant is correct on Zen 5.
Optimization Type:    None (configuration / naming defect).
GwenLand Target:      glproc
Recommendation:       ADAPT. Split the zen4 variant into two: a real zen4 variant without
                      AVX512_BF16 (loads on Zen 4 silicon), and a zen5 variant with
                      AVX512_BF16 + AVX512_FP16 (loads on Zen 5 silicon). Update the variant
                      list at CMakeLists.txt:395-397 accordingly.
Priority:             High
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX03-F03

```
Finding ID:           ARTX03-F03
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            8x8 batched GEMM (repack.cpp) and tinyBLAS 512-bit templates (sgemm.cpp)
Source File:          ggml/src/ggml-cpu/arch/x86/repack.cpp; ggml/src/ggml-cpu/llamafile/sgemm.cpp
Function:             gemm_q4_b32_8x8_q8_0_lut_avx (repack); tinyBLAS<16, __m512> / tinyBLAS<32, __m512, __m512bh>
Lines:                repack.cpp:663-1096 (16 × __m512 accumulators); repack.cpp:123-131 (512-bit VNNI helper);
                      sgemm.cpp:3726-3731 (F32 GEMM); sgemm.cpp:3788-3795 (BF16 GEMM)
Summary:              The 512-bit AVX-512 path is identical on Zen 4/Zen 5 and on Intel
                      IceLake/Sapphire Rapids. The code does not acknowledge Zen 4/Zen 5's
                      256-bit data path, which causes every 512-bit instruction to split
                      into two 256-bit uops internally. No compile-time switch prefers the
                      256-bit equivalents on Zen.
Observation:          Zen 4 and Zen 5 implement AVX-512 on a 256-bit data path. The
                      front-end decodes each 512-bit instruction (e.g. _mm512_dpbusd_epi32,
                      _mm512_dpbf16_ps, _mm512_fmadd_ph, _mm512_fmadd_ps) into two 256-bit
                      uops. The throughput is therefore at best equal to the 256-bit
                      equivalent (no benefit from 512-bit width). The cost is doubled
                      decode/fetch bandwidth and a mild frequency reduction (downclocking).

                      The 8x8 batched GEMM in repack.cpp:663-1096 uses 16 × __m512
                      accumulators and _mm512_dpbusd_epi32 (via the helper at line 123-131).
                      On Zen 4/Zen 5, each _mm512_dpbusd_epi32 splits into two _mm256_dpbusd
                      uops. The 16 accumulators occupy 16 of the 32 AVX-512 architectural
                      registers — register file is fine; the bottleneck is decode/fetch
                      bandwidth.

                      The tinyBLAS F32 GEMM at sgemm.cpp:3726-3731 instantiates
                      tinyBLAS<16, __m512, __m512, ...> on any AVX-512F build. The BF16 GEMM
                      at sgemm.cpp:3788-3795 instantiates tinyBLAS<32, __m512, __m512bh, ...>
                      on any AVX-512 BF16 build. On Zen 5, both run with 512-bit uops that
                      split.

                      The 256-bit equivalents exist: _mm256_dpbusd_epi32 (in the helper at
                      repack.cpp:152-161 and quants.c:106-118), _mm256_dpbf16_ps (template
                      specialization at sgemm.cpp:156-159, never instantiated), and
                      _mm256_fmadd_ph (part of AVX-512 FP16 + VL, not used). The codebase
                      has the pieces but does not assemble them into a Zen-preferred path.
Evidence:             repack.cpp:663-1096 (16 × __m512 acc block, conditional on
                      __AVX512BW__ && __AVX512DQ__);
                      repack.cpp:123-131 (_mm512_dpbusd_epi32 in 512-bit helper);
                      sgemm.cpp:3726-3731 (F32 GEMM 512-bit template instantiation);
                      sgemm.cpp:3788-3795 (BF16 GEMM 512-bit template instantiation);
                      sgemm.cpp:156-159 (_mm256_dpbf16_ps specialization, never instantiated).
Architectural Impact: Zen 4 and Zen 5 execute 512-bit instructions at 2x decode/fetch cost
                      with no throughput benefit. For prompt-processing GEMM (large m, n),
                      this is a measurable throughput loss vs. a 256-bit-native implementation.
                      The gap is approximately 10-30% on memory-bound shapes (where decode
                      bandwidth matters) and ~0% on compute-bound shapes (where the data path
                      is the bottleneck either way).
Correctness Impact:   None. 512-bit and 256-bit instructions produce identical results; the
                      difference is execution efficiency.
Optimization Type:    SIMD vectorization (proposed: 256-bit-preferred variant on Zen).
GwenLand Target:      glproc
Recommendation:       REJECT the absence; provide 256-bit variants for the 8x8 batched GEMM
                      and the tinyBLAS F32/BF16 templates, selected when running on AMD Zen.
                      Use the existing _mm256_dpbusd_epi32 / _mm256_dpbf16_ps instructions.
                      The tinyBLAS_Q0_AVX pattern (F06) is the model: 256-bit width + 32
                      registers.
Priority:             High
Difficulty:           L
Dependencies:         ARTX03-F01 (is_amd as kernel-selection signal)
Confidence:           High
```

### Finding ARTX03-F04

```
Finding ID:           ARTX03-F04
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            AVX-VNNI fallback in VNNI helper ladder
Source File:          ggml/src/ggml-cpu/arch/x86/quants.c; ggml/src/ggml-cpu/arch/x86/repack.cpp
Function:             mul_sum_us8_pairs_float; mul_sum_us8_pairs_acc_int32x8
Lines:                quants.c:110-113; repack.cpp:154-155
Summary:              The AVX-VNNI fallback branch (#elif defined(__AVXVNNI__)) using
                      _mm256_dpbusd_avx_epi32 is dead code on AMD. The only AMD CPUs with
                      AVX-VNNI are Zen 4 and Zen 5, which both also have AVX-512 VNNI + VL
                      and take the first #if branch instead. AVX-VNNI is reached only on
                      Intel Alder Lake (the only x86 variant that enables GGML_AVX_VNNI
                      without GGML_AVX512).
Observation:          The VNNI helper ladder is:
                        #if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                            _mm256_dpbusd_epi32       // Zen 4, Zen 5, IceLake-client, Sapphire Rapids
                        #elif defined(__AVXVNNI__)
                            _mm256_dpbusd_avx_epi32   // Alder Lake only
                        #else
                            _mm256_maddubs_epi16 + _mm256_madd_epi16  // AVX2-only (Zen 3, Haswell)
                        #endif
                      On AMD, the AVX-VNNI branch is unreachable because every AMD CPU that
                      has AVX-VNNI (Zen 4, Zen 5) also has AVX-512 VNNI + VL. The
                      AVX-VNNI fallback exists for Intel Alder Lake P-cores, which have
                      AVX-VNNI at 256-bit width but no AVX-512.

                      This is not a bug. The fallback is correct for Alder Lake. The
                      observation is that AVX-VNNI as a standalone ISA extension (without
                      AVX-512) is an Intel-only feature; AMD chose to implement AVX-512
                      VNNI + VL instead, which subsumes AVX-VNNI.
Evidence:             quants.c:106-118 (3-way ladder, AVX-VNNI branch at 110-113);
                      repack.cpp:152-161 (same ladder, AVX-VNNI branch at 154-155);
                      CMakeLists.txt:398 (alderlake variant — only variant with AVX_VNNI
                      and not AVX512).
Architectural Impact: None on AMD. The AVX-VNNI fallback is dead code on AMD but live on
                      Alder Lake. The helper ladder is correct.
Correctness Impact:   None.
Optimization Type:    None (correct fallback for the one CPU that needs it).
GwenLand Target:      glproc
Recommendation:       MONITOR. Keep the AVX-VNNI fallback for Alder Lake. GwenLand should
                      preserve the 3-way ladder. No action needed for AMD.
Priority:             Low
Difficulty:           XS
Dependencies:         ARTX02-F02, ARTX02-F03
Confidence:           High
```

### Finding ARTX03-F05

```
Finding ID:           ARTX03-F05
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            BF16 dot product path (vec.cpp and sgemm.cpp)
Source File:          ggml/src/ggml-cpu/vec.cpp; ggml/src/ggml-cpu/llamafile/sgemm.cpp
Function:             ggml_vec_dot_bf16; madd<__m512bh, __m512bh, __m512> template;
                      madd<__m256bh, __m256bh, __m256> template (never instantiated)
Lines:                vec.cpp:148-158 (__AVX512BF16__ branch using _mm512_dpbf16_ps);
                      sgemm.cpp:151-160 (both BF16 madd specializations);
                      sgemm.cpp:3788-3795 (BF16 GEMM dispatch — instantiates only 512-bit)
Summary:              The BF16 vecdot in vec.cpp uses _mm512_dpbf16_ps (512-bit) on Zen 5
                      (via the zen4 variant .so). The 256-bit _mm256_dpbf16_ps template
                      specialization exists at sgemm.cpp:156-159 but is never instantiated
                      by any dispatch site. On Zen 5's 256-bit data path, the 512-bit
                      instruction splits into two 256-bit uops, wasting decode bandwidth
                      without throughput benefit.
Observation:          The BF16 vecdot ladder in vec.cpp:148-195 is:
                        #if defined(__AVX512BF16__)   → _mm512_dpbf16_ps (512-bit, 32 BF16 lanes)
                        #elif defined(__AVX512F__)    → convert + _mm512_mul_ps + _mm512_add_ps (512-bit FP32 emulation)
                        #elif defined(__AVX2__)       → convert + _mm256_mul_ps + _mm256_add_ps (256-bit FP32 emulation)
                      On Zen 4 (icelake variant .so, no __AVX512BF16__), the __AVX512F__
                      branch runs: 512-bit FP32 emulation, 2 uops per FMA, no native BF16.
                      On Zen 5 (zen4 variant .so, __AVX512BF16__ defined), the
                      __AVX512BF16__ branch runs: 512-bit native BF16 dot, 2 uops per FMA.

                      In sgemm.cpp:151-160, both 512-bit and 256-bit BF16 madd specializations
                      are defined. The 256-bit _mm256_dpbf16_ps specialization (line 156-159)
                      is reachable only if a tinyBLAS template is instantiated with __m256bh
                      as the vector type. The BF16 GEMM dispatch at sgemm.cpp:3788-3795
                      instantiates tinyBLAS<32, __m512, __m512bh, ...> — only the 512-bit
                      template. The 256-bit template is never instantiated.

                      On Zen 5's 256-bit data path, _mm512_dpbf16_ps splits into two
                      _mm256_dpbf16 uops. Each uop produces 16 lanes of FP32 output. The
                      256-bit _mm256_dpbf16_ps would execute as a single uop on Zen 5's
                      native 256-bit data path — strictly better.
Evidence:             vec.cpp:148-158 (__AVX512BF16__ branch, _mm512_dpbf16_ps);
                      sgemm.cpp:151-160 (both 512-bit and 256-bit BF16 madd specializations);
                      sgemm.cpp:3788-3795 (BF16 GEMM dispatch — 512-bit only).
Architectural Impact: On Zen 5, BF16 matmuls (including BF16 activation quantization for
                      LLM inference with BF16 weights) pay 2x decode/fetch cost with no
                      throughput benefit. The 256-bit alternative exists in the codebase
                      but is not selected.
Correctness Impact:   None. _mm512_dpbf16_ps and _mm256_dpbf16_ps produce identical int32
                      results (per AMD/Intel ISA spec); only the SIMD width differs.
Optimization Type:    SIMD vectorization (proposed: 256-bit BF16 path on Zen).
GwenLand Target:      glproc
Recommendation:       REJECT the absence; add a 256-bit BF16 dispatch path. In sgemm.cpp,
                      add a branch that instantiates tinyBLAS<16, __m256, __m256bh, ...>
                      when running on AMD Zen. In vec.cpp, add a 256-bit __AVX512BF16__
                      branch before the 512-bit branch (or use _mm256_dpbf16_ps directly
                      when is_amd).
Priority:             High
Difficulty:           S
Dependencies:         ARTX03-F01, ARTX03-F03
Confidence:           High
```

### Finding ARTX03-F06

```
Finding ID:           ARTX03-F06
Category:             ADOPT
Engine:               CPU
Component:            tinyBLAS_Q0_AVX template (Q4_0/Q5_0/Q5_1/Q8_0 GEMM)
Source File:          ggml/src/ggml-cpu/llamafile/sgemm.cpp
Function:             tinyBLAS_Q0_AVX (class template)
Lines:                1351-1794 (class); 1376-1459 (mnpack tile selection); 1754-1764
                      (updot VNNI dispatch)
Summary:              tinyBLAS_Q0_AVX is the one AMD-friendly design in the codebase. It
                      uses __m256 / __m256i width throughout (no 512-bit uop splits),
                      harvests the 32-register file via VECTOR_REGISTERS == 32 to pick
                      wider tile shapes (4x4, 4x3, 3x4, 3x3), and dispatches the int8 dot
                      product through a 3-way VNNI ladder to _mm256_dpbusd_epi32 on Zen 4/
                      Zen 5. This is exactly the right pattern for a 256-bit-data-path
                      AVX-512 CPU.
Observation:          The template at sgemm.cpp:1351-1794 uses __m256 and __m256i exclusively.
                      There are no __m512 / __m512i instructions anywhere in the class. The
                      updot helper at line 1754-1764 has the 3-way VNNI ladder:
                        #if defined(__AVX512VNNI__) && defined(__AVX512VL__)
                            _mm256_dpbusd_epi32       // Zen 4, Zen 5 (256-bit, 1 uop, native)
                        #elif defined(__AVXVNNI__)
                            _mm256_dpbusd_avx_epi32   // Alder Lake
                        #else
                            _mm256_madd_epi16 + _mm256_maddubs_epi16  // AVX2-only
                      The mnpack function (line 1376-1459) picks tile shapes based on
                      remaining (m, n) and VECTOR_REGISTERS. On AVX-512 builds
                      (VECTOR_REGISTERS == 32, line 67), the wider tile shapes are enabled:
                      4x4, 4x3, 3x4, 3x3, 4x2, 2x4. On AVX2 builds (VECTOR_REGISTERS == 16),
                      the tile shapes collapse to 4x2, 2x4, 3x2, 2x3.

                      On Zen 4 / Zen 5, this template:
                        1. Uses 256-bit data path (no uop splits).
                        2. Uses 32 architectural registers (AVX-512 doubles the register file).
                        3. Uses 256-bit _mm256_dpbusd_epi32 (1 uop, 1-cycle throughput).
                      All three are AMD-optimal. The 32-register benefit is the one
                      advantage Zen gets from AVX-512 (vs. AVX2 which has 16 registers),
                      and the template harvests it without paying the 512-bit decode cost.
Evidence:             sgemm.cpp:1351-1794 (tinyBLAS_Q0_AVX class);
                      sgemm.cpp:66-70 (VECTOR_REGISTERS macro — 32 on AVX-512F);
                      sgemm.cpp:1376-1459 (mnpack — wider tiles when VR==32);
                      sgemm.cpp:1754-1764 (updot — 3-way VNNI ladder, 256-bit width).
Architectural Impact: The Q4_0/Q5_0/Q5_1/Q8_0 GEMM path (selected via llamafile_sgemm at
                      sgemm.cpp:3935-4019) is the AMD-optimal path. This is the one place
                      in the codebase where the design happens to be correct for Zen's
                      256-bit data path.
Correctness Impact:   None. The 256-bit VNNI instruction produces identical results to the
                      512-bit version (per ISA spec).
Optimization Type:    SIMD (256-bit VNNI + 32-register tile selection).
GwenLand Target:      glproc
Recommendation:       ADOPT. Replicate the tinyBLAS_Q0_AVX pattern in glproc: __m256 width
                      throughout, 3-way VNNI ladder, 32-register-aware tile selection. Use
                      this as the default Q4_0/Q5_0/Q5_1/Q8_0 GEMM on AMD, and as a
                      fallback on Intel when 512-bit downclocking is undesirable.
Priority:             Critical
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX03-F07

```
Finding ID:           ARTX03-F07
Category:             MISSING_FEATURE
Engine:               CPU
Component:            AMX tile-matrix-multiply unit (Intel-only); AMD has no equivalent
Source File:          ggml/src/ggml-cpu/amx/amx.cpp; ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
Function:             ggml::cpu::amx::extra_buffer_type (conditional compilation);
                      ggml_backend_cpu_x86_score (AMX_INT8 branch)
Lines:                amx.cpp:19 (#if defined(__AMX_INT8__) && defined(__AVX512VNNI__));
                      amx.cpp:144-249 (extra_buffer_type, compiles out on AMD);
                      cpu-feats.cpp:317-320 (AMX_INT8 score branch — returns 0 on AMD)
Summary:              AMX is Intel-only. The sapphirerapids variant (CMakeLists.txt:401)
                      is the only x86 variant that defines __AMX_INT8__. On AMD, the AMX
                      plugin compiles to an empty TU and the sapphirerapids variant returns
                      score=0. AMD has no equivalent tile-matrix-multiply unit; the only
                      path to high-throughput int8 matmul on AMD is AVX-512 VNNI at 256-bit
                      width, ~8x slower per core than Intel AMX.
Observation:          The AMX plugin at amx/amx.cpp:19-249 is wrapped in
                      #if defined(__AMX_INT8__) && defined(__AVX512VNNI__). The
                      sapphirerapids variant (CMakeLists.txt:401) is the only x86 variant
                      that defines __AMX_INT8__ (via -mamx-int8 at CMakeLists.txt:361).
                      On AMD:
                        1. The sapphirerapids .so returns score=0 (cpu-feats.cpp:318
                           fails: !is.AMX_INT8()).
                        2. The .so is never loaded.
                        3. amx/amx.cpp produces an empty translation unit.
                        4. The ggml::cpu::amx::extra_buffer_type is never registered.
                      The exclusion is correct. The architectural consequence is that AMD's
                      highest-throughput int8 matmul is AVX-512 VNNI at 256-bit width:
                      one _mm256_dpbusd_epi32 per cycle per core, 8 int32 MACs per
                      instruction = 8 int32 MACs/cycle/core. Intel AMX delivers ~64 int32
                      MACs/cycle/core (TMM0..TMM7, 16-row × 64-byte tiles). The 8x per-core
                      asymmetry is real and unaddressed.

                      For LLM inference, this means:
                        - On Intel Sapphire Rapids / Granite Rapids: AMX is the preferred
                          path for Q4_0/Q4_K/IQ4_NL GEMM (selected via the repack buffer
                          type + AMX tensor_traits).
                        - On AMD Zen 4 / Zen 5: AVX-512 VNNI at 256-bit width is the only
                          path. The same Q4_0/Q4_K/IQ4_NL GEMM runs through the repack
                          8x8 batched GEMM (if GGML_USE_CPU_REPACK is enabled) or through
                          the per-block vecdot in quants.c. AMD's advantage is core count
                          (96-core EPYC 9654 vs. 56-core Xeon 8480+), not per-core throughput.
Evidence:             amx/amx.cpp:19 (conditional compilation gate);
                      amx/amx.cpp:144-249 (extra_buffer_type, compiles out);
                      cpu-feats.cpp:317-320 (AMX_INT8 score branch — returns 0 on AMD);
                      CMakeLists.txt:401 (sapphirerapids variant — only AMX_INT8 variant).
Architectural Impact: AMD and Intel have asymmetric int8 matmul throughput. GwenLand should
                      not assume AMD and Intel are equivalent for Q4_0/Q4_K GEMM. The
                      scheduler should prefer AMX on Intel and 256-bit VNNI on AMD, with
                      different per-core throughput expectations.
Correctness Impact:   None. The exclusion is correct — AMD never executes AMX instructions.
Optimization Type:    None (architectural asymmetry documentation).
GwenLand Target:      glproc, GATE
Recommendation:       DEFER. Do not attempt to emulate AMX on AMD. Document the asymmetry.
                      For very large int8 GEMM on AMD, scale with core count (multi-chiplet
                      EPYC) rather than per-core throughput. Provide a 256-bit VNNI path
                      (F06) as the AMD equivalent of AMX.
Priority:             Low
Difficulty:           XL
Dependencies:         ARTX03-F06
Confidence:           High
```

### Finding ARTX03-F08

```
Finding ID:           ARTX03-F08
Category:             SIMD_STRATEGY
Engine:               CPU
Component:            F16 vecdot (vec.cpp) — AVX-512 FP16 path
Source File:          ggml/src/ggml-cpu/vec.cpp; ggml/src/ggml-cpu/simd-mappings.h
Function:             ggml_vec_dot_f16; GGML_F16_VEC_FMA macro
Lines:                vec.cpp:264-378 (ggml_vec_dot_f16);
                      simd-mappings.h:493-578 (__AVX512FP16__ macro block, _mm512_fmadd_ph
                      at line 503)
Summary:              The F16 vecdot in vec.cpp DOES use native AVX-512 FP16 FMA
                      (_mm512_fmadd_ph) when __AVX512FP16__ is defined, contradicting the
                      scope of ARTX02-F08 (which audited only quants.c / repack.cpp /
                      sgemm.cpp and found no FP16 usage). On Zen 5 (with the zen4 variant
                      .so and a compiler that defines __AVX512FP16__), this 512-bit
                      instruction splits into two 256-bit uops. The 256-bit _mm256_fmadd_ph
                      would be more efficient on Zen 5 but is not used.
Observation:          ARTX02-F08 stated: "No AVX-512 FP16 dot products anywhere in the
                      audited files." That was correct for the three files ARTX02 audited
                      (quants.c, repack.cpp, sgemm.cpp). However, vec.cpp and simd-mappings.h
                      were not in ARTX02's scope.

                      simd-mappings.h:493-578 defines an __AVX512FP16__ block that maps
                      GGML_F16_VEC_FMA to _mm512_fmadd_ph (line 503). vec.cpp:357 calls
                      GGML_F16_VEC_FMA inside the F16 vecdot loop. When __AVX512FP16__ is
                      defined, the F16 vecdot uses native FP16 FMA at 512-bit width, 32 FP16
                      lanes per __m512h accumulator, 4 accumulators (GGML_F16_ARR = 128/32).

                      On Zen 5 (with the zen4 variant .so), __AVX512FP16__ MAY be defined —
                      this depends on the compiler. The CMakeLists.txt:336-355 does not
                      explicitly enable -mavx512fp16 for the zen4 variant. Clang sometimes
                      defines __AVX512FP16__ transitively from -mavx512f on certain
                      toolchains; GCC does not. The result is a non-deterministic build
                      outcome. See Unknowns U4.

                      When __AVX512FP16__ IS defined and Zen 5 runs the zen4 .so:
                        - vec.cpp:357 uses _mm512_fmadd_ph (512-bit FP16 FMA, 2 uops on Zen 5)
                        - The 256-bit _mm256_fmadd_ph (part of AVX-512 FP16 + VL) is not used

                      When __AVX512FP16__ is NOT defined (e.g. on the icelake variant that
                      loads on Zen 4):
                        - simd-mappings.h:533-577 fallback block maps GGML_F16_VEC_FMA to
                          _mm512_fmadd_ps (512-bit FP32 FMA after FP16->FP32 conversion)
                        - The FP16->FP32 conversion uses _mm512_cvtph_ps at simd-mappings.h:544

                      In both cases, the vecdot runs at 512-bit width. On Zen 4/Zen 5, this
                      means 2 uops per FMA. The 256-bit alternatives (_mm256_fmadd_ph or
                      _mm256_cvtph_ps + _mm256_fmadd_ps) are not used.
Evidence:             vec.cpp:264-378 (ggml_vec_dot_f16);
                      simd-mappings.h:493-578 (__AVX512FP16__ block, _mm512_fmadd_ph at 503);
                      simd-mappings.h:533-578 (fallback block, _mm512_fmadd_ps at 547);
                      CMakeLists.txt:336-355 (zen4 variant does not explicitly enable
                      -mavx512fp16).
Architectural Impact: On Zen 5, the F16 vecdot pays 2x decode/fetch cost with no throughput
                      benefit. The 256-bit _mm256_fmadd_ph would execute as a single uop on
                      Zen 5's native 256-bit data path. For LLM inference with F16 weights,
                      F16 vecdot is the hot path — the gap is measurable.
Correctness Impact:   None. _mm512_fmadd_ph and _mm256_fmadd_ph produce identical FP32
                      results (per ISA spec); only the SIMD width differs.
Optimization Type:    SIMD vectorization (proposed: 256-bit FP16 path on Zen).
GwenLand Target:      glproc
Recommendation:       ADAPT. Add a 256-bit FP16 path to vec.cpp and simd-mappings.h. Use
                      _mm256_fmadd_ph on Zen 5. Also: explicitly enable -mavx512fp16 in the
                      zen4/zen5 variant CMakeLists to make the build deterministic. This
                      finding corrects the scope of ARTX02-F08: the FP16 dot product exists
                      in vec.cpp, but only at 512-bit width.
Priority:             Medium
Difficulty:           M
Dependencies:         ARTX03-F01, ARTX03-F05
Confidence:           High
```

### Finding ARTX03-F09

```
Finding ID:           ARTX03-F09
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            Score function (AVX_VNNI bit redundancy)
Source File:          ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp
Function:             ggml_backend_cpu_x86_score
Lines:                293-296 (AVX_VNNI score branch); 313-316 (AVX512_VNNI score branch)
Summary:              The score function adds 1<<6 = 64 for AVX_VNNI and 1<<10 = 1024 for
                      AVX512_VNNI. On Zen 4 and Zen 5, both bits are set, so the score
                      includes both contributions — but AVX_VNNI is redundant when
                      AVX512_VNNI + AVX512_VL are present (the 256-bit _mm256_dpbusd_epi32
                      subsumes _mm256_dpbusd_avx_epi32). Zen 4 outscores IceLake-client by
                      64 due to this redundant bit, without reflecting any real capability
                      difference.
Observation:          The AVX_VNNI CPUID bit (leaf 7, sub-leaf 1, EAX[4]) indicates
                      support for _mm256_dpbusd_avx_epi32 — the AVX-VNNI encoding at
                      256-bit width, available without AVX-512. The AVX512_VNNI CPUID bit
                      (leaf 7, sub-leaf 0, ECX[11]) indicates support for
                      _mm512_dpbusd_epi32 and (with AVX512_VL) _mm256_dpbusd_epi32 — the
                      AVX-512 VNNI encoding.

                      On a CPU with both AVX_VNNI and AVX512_VNNI + AVX512_VL (e.g. Zen 4,
                      Zen 5, Sapphire Rapids, Granite Rapids), the AVX-VNNI encoding is
                      never used — the AVX-512 VNNI + VL encoding is strictly more capable
                      (same 256-bit width, plus 512-bit width when desired).

                      Despite this, the score function adds 64 for AVX_VNNI independently
                      of AVX512_VNNI. On Zen 4:
                        score = 1 (base) + 1 (FMA) + 2 (F16C) + 4 (SSE42) + 8 (BMI2) +
                                16 (AVX) + 32 (AVX2) + 64 (AVX_VNNI) + 128 (AVX512) +
                                256 (AVX512_VBMI) + 1024 (AVX512_VNNI)
                              = 1536
                      On IceLake-client:
                        score = 1 + 1 + 2 + 4 + 8 + 16 + 32 + 0 + 128 + 256 + 1024
                              = 1472
                      Difference: 64 (the redundant AVX_VNNI bit). Zen 4 and IceLake-client
                      have equivalent VNNI throughput (both use _mm256_dpbusd_epi32 at
                      256-bit width), but the score misrepresents a 64-point gap.

                      In practice, this is harmless: Zen 4 falls back to the icelake variant
                      anyway (because the zen4 variant requires AVX512_BF16, which Zen 4
                      lacks — see F02). But the inflated score misrepresents the capability
                      gap and could cause confusion in the variant-selection logic if
                      future variants are added.
Evidence:             cpu-feats.cpp:293-296 (AVX_VNNI score = 1<<6 = 64);
                      cpu-feats.cpp:313-316 (AVX512_VNNI score = 1<<10 = 1024);
                      cpu-feats.cpp:79-83 (AVX_VNNI and AVX512_VNNI are independent CPUID
                      bits).
Architectural Impact: Zen 4 and Zen 5 scores are inflated by 64 vs. Intel equivalents
                      without reflecting any capability difference. Harmless in the
                      current build matrix (the inflated score does not change which .so is
                      loaded), but misleading.
Correctness Impact:   None.
Optimization Type:    None (score function design defect).
GwenLand Target:      glproc
Recommendation:       ADAPT. Change the AVX_VNNI score branch to:
                        #ifdef GGML_AVX_VNNI
                            if (!is.AVX_VNNI()) { return 0; }
                            #ifndef GGML_AVX512_VNNI
                                score += 1<<6;   // only count if AVX512_VNNI is not also required
                            #endif
                        #endif
                      This removes the redundant 64-point contribution on Zen 4/Zen 5 and
                      Cooper Lake without affecting Alder Lake (which has AVX_VNNI but not
                      AVX512_VNNI).
Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX03-F10

```
Finding ID:           ARTX03-F10
Category:             BACKEND_DESIGN
Engine:               CPU
Component:            AMD variant matrix (absence of Zen 4 / Zen 5 specific binaries)
Source File:          ggml/src/CMakeLists.txt
Function:             GGML_CPU_ALL_VARIANTS block (x86)
Lines:                378-402 (all x86 variants); 396 (zen4 variant — the only AMD-named
                      variant)
Summary:              There is no Zen 4-specific or Zen 5-specific binary in the standard
                      build matrix. The only AMD-named variant is zen4 (line 396), which
                      (per F02) is misnamed — it requires AVX512_BF16 and therefore loads
                      only on Zen 5. Zen 4 runs the Intel-targeted icelake variant. Neither
                      variant makes any AMD-specific kernel decision; both run the same code
                      as their Intel equivalents.
Observation:          The 14 x86 variants defined at CMakeLists.txt:378-402 are:
                        x64, sse42, sandybridge, ivybridge, piledriver, haswell, skylakex,
                        cannonlake, cascadelake, icelake, cooperlake, zen4, alderlake,
                        sapphirerapids
                      The zen4 variant is the only AMD-named variant. piledriver (line 384)
                      is an older AMD variant (Bulldozer-era, AVX+FMA, no AVX2), but it is
                      irrelevant to modern Zen hardware.

                      On Zen 4 silicon, the variant loaded is icelake (the highest-scoring
                      variant that does not require AVX512_BF16 or AMX_INT8). On Zen 5
                      silicon, the variant loaded is zen4 (the highest-scoring variant
                      overall, since AVX512_BF16 is present and AMX_INT8 is not).

                      Neither icelake nor zen4 contains any AMD-specific kernel code. Both
                      are compiled with the same flags as their Intel equivalents (icelake
                      for IceLake-client; zen4 for Cooper Lake + AVX512_VBMI). The AMD
                      kernel path is therefore identical to the Intel kernel path for the
                      same CPUID feature set.

                      This is the central architectural observation of this audit: AMD Zen
                      4 / Zen 5 do not have a dedicated kernel path in llama.cpp. They run
                      Intel-targeted binaries (icelake for Zen 4, a Cooper Lake + VBMI
                      superset for Zen 5).
Evidence:             CMakeLists.txt:378-402 (x86 variant list — only one AMD-named variant
                      at line 396);
                      CMakeLists.txt:390 (icelake — fallback for Zen 4);
                      CMakeLists.txt:396 (zen4 — loaded on Zen 5);
                      Grep across ggml/src/ggml-cpu/ for is_amd / __zn / ZEN returns no
                      kernel-level hits (only cpu-feats.cpp:68-77 legacy gates and the
                      CMake variant name).
Architectural Impact: AMD users get binaries tuned for Intel microarchitecture. The
                      256-bit data path of Zen 4 / Zen 5 is not acknowledged. The 512-bit
                      instructions in repack.cpp and sgemm.cpp split into two 256-bit uops
                      on Zen, paying decode/fetch cost without throughput benefit. The
                      256-bit alternatives exist as template specializations but are not
                      instantiated.
Correctness Impact:   None. The Intel-targeted binaries run correctly on AMD (the ISA is
                      the same).
Optimization Type:    None (absence of vendor-specific binaries).
GwenLand Target:      glproc
Recommendation:       ADAPT. GwenLand should add real zen4 and zen5 variants to its build
                      matrix. The zen4 variant should NOT require AVX512_BF16 (Zen 4 lacks
                      it). The zen5 variant should include AVX512_BF16 and explicitly enable
                      AVX512_FP16. Both variants should prefer 256-bit width in the
                      repack.cpp 8x8 batched GEMM and the sgemm.cpp tinyBLAS templates
                      (per F03, F05, F08). This requires extending is_amd beyond legacy
                      feature gates (per F01).
Priority:             High
Difficulty:           L
Dependencies:         ARTX03-F01, ARTX03-F02, ARTX03-F03
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the 512-bit `__m512` instructions in `repack.cpp:663+`
  cause a measurable throughput loss on Zen 4 / Zen 5 vs. a 256-bit
  equivalent. The 256-bit equivalent is not present in the codebase for
  the 8×8 batched GEMM, so a direct comparison requires implementing it.
  Requires runtime profiling on Zen 4 / Zen 5 hardware. Static analysis
  can only confirm the uop split, not the throughput consequence.

* **U2**. Whether the `zen4` variant naming defect (F02) has caused
  real-world performance loss on Zen 4 hardware. Zen 4 falls back to the
  `icelake` variant, which uses the same 256-bit VNNI helper as a
  hypothetical `zen4` variant would. The 8×8 batched GEMM and the BF16
  GEMM paths differ between `icelake` (no `__AVX512BF16__`, no native
  BF16) and a hypothetical `zen4` (would define `__AVX512BF16__` if Zen 4
  had it — but Zen 4 doesn't, so this is moot). Net: the naming defect
  is harmless on Zen 4 because Zen 4 lacks AVX-512 BF16 anyway. The
  actual loss is on Zen 5, where the `zen4` variant loads but uses
  512-bit BF16/FP16 instructions that split. Requires runtime profiling
  on Zen 5.

* **U3**. Whether the AVX-VNNI fallback (`_mm256_dpbusd_avx_epi32`) at
  `quants.c:110-113` is ever reached in practice. It is reached only on
  Alder Lake P-cores (the only AMD-relevant... no, the only x86 variant
  with AVX_VNNI but not AVX512). Whether Alder Lake users actually run
  the `alderlake` variant .so is a deployment question. Requires
  surveying downstream packagers (Debian, Fedora, Arch, etc.).

* **U4**. Whether `__AVX512FP16__` is defined when the `zen4` variant is
  compiled. The CMakeLists.txt:336-355 does not explicitly enable
  `-mavx512fp16`. Whether the macro is defined depends on the compiler:
  Clang sometimes defines `__AVX512FP16__` transitively from `-mavx512f`
  on certain toolchains; GCC does not. If the macro is NOT defined, the
  F16 vecdot on Zen 5 falls back to the FP32 convert path
  (`simd-mappings.h:533-578`). This is a non-deterministic build
  outcome. Requires building the `zen4` variant with both GCC and Clang
  and checking `__AVX512FP16__`.

* **U5**. Whether the `sapphirerapids` variant (with AMX) ever loads on
  AMD hardware. AMD CPUs do not have AMX_TILE or AMX_INT8 CPUID bits, so
  the score returns 0. The exclusion is correct. But whether any AMD
  user has accidentally built with `-march=native` on an EPYC and ended
  up with AMX instructions compiled in is a separate question (would
  cause `#UD` at runtime). Requires surveying build configurations.

* **U6**. Whether the `tinyBLAS_Q0_AVX` template's 4×4 tile shape (with
  32 registers) causes register spilling on Zen 5. The template uses up
  to 16 accumulator registers (4×4 tile) + 4 input registers + ~4
  constant registers + ~4 temporary registers ≈ 28 registers. This
  fits within 32 but leaves little headroom. Whether the compiler spills
  is not determinable without inspecting the compiler's register
  allocation output. Requires `objdump` on the compiled .so.

* **U7**. Whether Zen 5's reported AVX-512 downclocking is significant
  enough to justify preferring 256-bit width unconditionally. AMD has
  stated that Zen 5's AVX-512 downclocking is smaller than Intel's
  (Zen 5 keeps the 256-bit data path of Zen 4, so 512-bit instructions
  split but do not require the heavy 512-bit FPU power-up that Intel
  needs). If the downclocking is negligible, the 512-bit instructions
  on Zen 5 are merely wasted decode bandwidth, not a frequency penalty.
  Requires runtime measurement of core frequency under 512-bit workload
  on Zen 5.

* **U8**. Whether the OS XSAVE check FIXME (`cpu-feats.cpp:264`) matters
  on AMD. Modern Linux/macOS/Windows kernels enable XSAVE for AVX-512 on
  AMD Zen 4/Zen 5 just as on Intel. The FIXME is vendor-neutral. Whether
  any deployed AMD system runs an OS old enough to lack AVX-512 XSAVE
  support is a deployment question. Requires surveying OS versions on
  AMD EPYC deployments.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines              |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------------ |
| R01       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `cpuid_x86` constructor (vendor detection)     | 112-178            |
| R02       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `is_amd` / `is_intel` flags                    | 134-136, 180-181   |
| R03       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | Legacy AMD feature gates (ABM, SSE4a, XOP, …)  | 68-77              |
| R04       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `AVX_VNNI()` / `AVX512_VNNI()` / `AVX512_BF16()` / `AVX512_FP16()` / `AMX_INT8()` | 79-88 |
| R05       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | `ggml_backend_cpu_x86_score`                   | 263-323            |
| R06       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | AVX_VNNI score branch                          | 293-296            |
| R07       | `ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp`          | AVX512 / AVX512_BF16 / AVX512_VNNI / AMX_INT8 score branches | 297-320 |
| R08       | `ggml/src/CMakeLists.txt`                           | x86 variant matrix (`GGML_CPU_ALL_VARIANTS`)   | 378-402            |
| R09       | `ggml/src/CMakeLists.txt`                           | `icelake` variant (Zen 4 fallback)             | 390                |
| R10       | `ggml/src/CMakeLists.txt`                           | `zen4` variant (misnamed — loads on Zen 5)     | 396                |
| R11       | `ggml/src/CMakeLists.txt`                           | `sapphirerapids` variant (Intel AMX only)      | 401                |
| R12       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `mul_sum_us8_pairs_float` (256-bit VNNI helper)| 105-119            |
| R13       | `ggml/src/ggml-cpu/arch/x86/quants.c`               | `mul_sum_i8_pairs_float` (signed VNNI helper)  | 122-134            |
| R14       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `mul_sum_us8_pairs_acc_int32x16` (512-bit VNNI)| 123-131            |
| R15       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `mul_sum_us8_pairs_acc_int32x8` (256-bit VNNI) | 151-161            |
| R16       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `mul_sum_i8_pairs_acc_int32x8` (256-bit signed VNNI) | 165-175     |
| R17       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | 8×8 batched GEMM, 16 × `__m512` accumulators   | 663-1096           |
| R18       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `VECTOR_REGISTERS` macro (32 on AVX-512F)      | 66-70              |
| R19       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `madd<__m512bh, __m512bh, __m512>` (512-bit BF16) | 151-155         |
| R20       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `madd<__m256bh, __m256bh, __m256>` (256-bit BF16, never instantiated) | 156-159 |
| R21       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `load<ggml_bf16_t>` (BF16 → `__m512bh`)        | 372-385            |
| R22       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `tinyBLAS_Q0_AVX` class (AMD-friendly design)  | 1351-1794          |
| R23       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `tinyBLAS_Q0_AVX::mnpack` (32-register tile)   | 1376-1459          |
| R24       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `tinyBLAS_Q0_AVX::updot` (256-bit VNNI dispatch) | 1754-1764        |
| R25       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | F32 GEMM instantiation (512-bit)               | 3726-3731          |
| R26       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | BF16 GEMM instantiation (512-bit only)         | 3788-3795          |
| R27       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | Q8_0 / Q4_0 / Q5_0 GEMM instantiation (tinyBLAS_Q0_AVX) | 3935-4019 |
| R28       | `ggml/src/ggml-cpu/llamafile/sgemm.cpp`             | `llamafile_sgemm` (entry)                      | 3699-4058          |
| R29       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_f32`                             | 11-137             |
| R30       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_bf16` (`__AVX512BF16__` branch)  | 139-262            |
| R31       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_bf16` 512-bit BF21 dot           | 148-158            |
| R32       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_bf16` 512-bit FP32 emulation     | 160-169            |
| R33       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_f16`                             | 264-378            |
| R34       | `ggml/src/ggml-cpu/vec.cpp`                         | `ggml_vec_dot_f16` FMA loop (uses `GGML_F16_VEC_FMA`) | 352-359      |
| R35       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `__AVX512FP16__` macro block (native `_mm512_fmadd_ph`) | 493-578    |
| R36       | `ggml/src/ggml-cpu/simd-mappings.h`                 | FP16 fallback block (`_mm512_fmadd_ps` after cvtph) | 533-578      |
| R37       | `ggml/src/ggml-cpu/simd-mappings.h`                 | `GGML_CPU_FP16_TO_FP32` (F16C path)            | 56-63              |
| R38       | `ggml/src/ggml-cpu/amx/amx.cpp`                     | AMX conditional compilation gate                | 19                 |
| R39       | `ggml/src/ggml-cpu/amx/amx.cpp`                     | `ggml::cpu::amx::extra_buffer_type`            | 144-249            |
| R40       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `type_traits_cpu[GGML_TYPE_F16]`               | 221-226            |
| R41       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `type_traits_cpu[GGML_TYPE_BF16]`              | 394-399            |
| R42       | `ggml/src/ggml-cpu/ggml-cpu.c`                      | `ggml_compute_forward_mul_mat` (NUMA chunking) | 1413-1417          |
| R43       | `ggml/src/ggml-cpu/arch/x86/repack.cpp`             | `ggml_backend_cpu_repack_buffer_type` registration | 4821-4835       |
| R44       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | x86 architecture flag setup (non-MSVC)         | 304-368            |
| R45       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | `GGML_USE_CPU_REPACK` opt-in                   | 574-576            |
| R46       | `ggml/src/ggml-cpu/CMakeLists.txt`                  | `GGML_USE_LLAMAFILE` opt-in                    | 80-86              |
