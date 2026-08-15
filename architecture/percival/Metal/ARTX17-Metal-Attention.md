# ARTX17 — Metal Attention Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux (ARTX17)
**Target GwenLand module:** `glmetal` (attention kernels), `GATE` (mask-policy planner)

---

## 1. Executive Summary

The Metal attention layer of llama.cpp is implemented by a *two-kernel
Flash-Attention family* — `kernel_flash_attn_ext` (prefill / tile path,
`simdgroup_half8x8` outer-product QK and PV) and `kernel_flash_attn_ext_vec`
(decode / vec path, `simd_shuffle_down` intra-row reduction, multi-workgroup
combine) — plus three attention-adjacent helpers (`kernel_flash_attn_ext_pad`
for KV-cache tail padding, `kernel_flash_attn_ext_blk` for mask-prune
pre-pass, `kernel_flash_attn_ext_vec_reduce` for cross-workgroup result
combine), four RoPE variants (`kernel_rope_norm`, `_neox`, `_multi`,
`_vision`) with always-on YaRN length extrapolation, a standalone
`kernel_soft_max_f32[_4]` with ALiBi bias + attention-sink cap, the
SSM/Mamba kernels `kernel_ssm_conv_f32_f32[_4][_batched]` and
`kernel_ssm_scan_f32`, and the KV-cache-write kernels
`kernel_set_rows_f` / `kernel_set_rows_q32`.

A central dispatch routine (`ggml_metal_op_flash_attn_ext` in
`ggml-metal-ops.cpp:2650`) chooses between the tile and vec kernels
via the heuristic `ne01 < 20 && ne00 % 32 == 0` (decode→vec, prefill→tile),
then orchestrates the pad/blk pre-passes and the optional vec-reduce
post-pass as a 2–3 kernel pipeline inside one node. There is **no**
separate `diag_mask_inf` kernel — Metal relies on a precomputed F16 mask
tensor passed via `op->src[3]`, exactly as on CUDA but without the
standalone fallback kernel. The mask-prune `flash_attn_ext_blk` pre-pass
classifies each `C × Q` mask block into one of three states (skip, active,
all-zero), allowing the main FA kernel to elide fully-masked KV tiles —
this is Metal's analog of CUDA's `flash_attn_mask_to_KV_max`.

The FA tile/vec templates are instantiated for **8 KV dtypes only**:
F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. **No K-quants, no
IQ-quants, no MXFP4, no FP8** are supported in the FA family at this
commit. (The audit prompt mentioned `kernel_flash_attn_ext_q4_K`,
`_q6_K`, `_iq4_nl` — these do **not** exist; the corresponding dtype
cases fall through to `ggml_metal_device_supports_op` returning `false`
at `ggml-metal-device.m:1250-1266`.) MLA-style `DK != DV` shapes
(`dk192_dv128`, `dk320_dv256`, `dk576_dv512`) **are** supported via
template parameters, but without the `V_is_K_view` aliasing trick that
CUDA uses — Metal always loads V from its own buffer.

For GwenLand, the decisions worth **ADOPT**ing are: the two-kernel
(tile/vec) taxonomy keyed on `ne01 < 20`, the per-(dtype,DK,DV) template
explosion with string-keyed lazy pipeline cache, the mask-prune
`flash_attn_ext_blk` three-state pre-pass, the KV-tail `flash_attn_ext_pad`
pre-pass with `-MAXHALF` mask fill, the always-on YaRN integration inside
every RoPE variant, the attention-sink `src[4]` baked into both FA and
soft_max, and the `set_rows` quantize-on-write pattern. The decisions
worth **REJECT**ing are the absence of an FP8 KV path, the absence of
K-quant / IQ-quant KV support, the absence of a `V_is_K_view` MLA
shortcut, the dead `nsg` autotuner (commented out at
`ggml-metal-ops.cpp:2819-2834`), and the absence of an
`ROPE+VIEW+SET_ROWS` fusion (CUDA has it; Metal does not).

---

## 2. Purpose

Provide a Metal implementation of the attention family of `ggml_op`s:

* `GGML_OP_FLASH_ATTN_EXT` — FlashAttention-2 with online softmax,
  optional ALiBi bias, optional logit-softcap (Gemma), optional
  attention-sink cap (Gemma), and a precomputed F16 mask tensor
  (causal / sliding-window / arbitrary pattern).
* `GGML_OP_SOFT_MAX` — standalone two-pass F32 softmax with optional
  F16 mask, ALiBi slope, and attention-sink `src[2]` per-head max cap.
  Used by the legacy (non-FA) attention path and by sampling.
* `GGML_OP_ROPE` / `GGML_OP_ROPE_BACK` — four RoPE variants
  (`norm`/`neox`/`multi`/`vision`) with always-on YaRN length
  extrapolation and an optional per-channel frequency factor
  (`src[2]`).
* `GGML_OP_SSM_CONV` — Mamba short convolution (1D depthwise), with
  optional batched kernel for prefill.
* `GGML_OP_SSM_SCAN` — Mamba-1/2 SSM scan (state recurrence).
* `GGML_OP_SET_ROWS` — scatter rows into a destination tensor by
  integer index; the primary KV-cache-append mechanism (used after
  RoPE to write the new K/V row into the cache).
* `GGML_OP_DIAG` — construct a diagonal matrix (NOT `DIAG_MASK_INF`).

It is **not** responsible for: graph construction, mask construction
(handled by the model code; Metal has no `diag_mask_inf` kernel),
kernel selection across backends (handled by the scheduler — ARTX22),
or pipeline compilation (handled in `ggml-metal-device.m`, ARTX15).
This file owns the kernel bodies and the per-op encoder setup; the
host-side dispatch and pipeline-name selection are in
`ggml-metal-ops.cpp` (audited here as the dispatch layer) and
`ggml-metal-device.cpp` (audited in ARTX15).

---

## 3. Source Files

| File                                       | Lines  | Role                                                                  |
| ----------------------------------------- | ------ | -------------------------------------------------------------------- |
| `ggml/src/ggml-metal/ggml-metal.metal`    | 11 218 | All Metal kernels; this audit covers FA (lines 6175-7827), RoPE (4550-4866), soft_max (1950-2161), ssm_conv (2172-2326), ssm_scan (2330-2520), set_rows (9842-9935), diag (9936-9954). |
| `ggml/src/ggml-metal/ggml-metal-ops.cpp`  | 4 864  | Per-op encoder functions: `ggml_metal_op_flash_attn_ext` (lines 2650-3078), `ggml_metal_op_soft_max` (1300-1388), `ggml_metal_op_rope` (3547-3641), `ggml_metal_op_ssm_conv` (1390-1461), `ggml_metal_op_ssm_scan` (1463-1540). Also the FA `use_vec` heuristic (2526-2534) and three `extra_*` size helpers (2536-2648). |
| `ggml/src/ggml-metal/ggml-metal-impl.h`   | 1 222  | Function-constant offsets (`FC_FLASH_ATTN_EXT_*`, `FC_ROPE`, `FC_SSM_CONV`); per-op kernel-args structs (`ggml_metal_kargs_flash_attn_ext`, `_vec`, `_pad`, `_blk`, `_vec_reduce`, `_rope`, `_soft_max`, `_ssm_conv`, `_ssm_scan`, `_set_rows`, `_diag`); tile-size constants `OP_FLASH_ATTN_EXT_NQPSG=8`, `OP_FLASH_ATTN_EXT_NCPSG=64`, `OP_FLASH_ATTN_EXT_VEC_NQPSG=1`, `OP_FLASH_ATTN_EXT_VEC_NCPSG=32`. |
| `ggml/src/ggml-metal/ggml-metal-device.cpp` | 2 161 | Pipeline-name → pipeline-state cache: `ggml_metal_library_get_pipeline_flash_attn_ext[_vec|_pad|_blk|_vec_reduce]` (1309-1549), `ggml_metal_library_get_pipeline_rope` (1719-1761), `ggml_metal_library_get_pipeline_soft_max` (453-478), `ggml_metal_library_get_pipeline_ssm_conv[_batched]` (480-538), `ggml_metal_library_get_pipeline_ssm_scan` (539+). |
| `ggml/src/ggml-metal/ggml-metal-device.m` | 1 917  | `ggml_metal_device_supports_op` for `GGML_OP_FLASH_ATTN_EXT` (1229-1267): whitelists DK ∈ {32,40,48,64,72,80,96,112,128,192,256,320,512,576} and KV dtype ∈ {F32,F16,Q8_0,Q4_0,Q4_1,Q5_0,Q5_1,BF16}. |

> **Note on the constants header.** `ggml-metal-impl.h` is the single
> source of truth for the tile-size constants that flow from the host
> (which picks the pipeline name) into the kernel (which reads them as
> `function_constant`). `OP_FLASH_ATTN_EXT_NQPSG = 8` means 8 queries
> per threadgroup in the tile kernel; `OP_FLASH_ATTN_EXT_NCPSG = 64`
> means 64 KV items per simdgroup. The vec kernel uses 1 query / 32 KV
> items per simdgroup. These are compile-time `#define`s, not
> autotunable parameters.

---

## 4. Architecture Overview

```
       ┌────────────────────────────────────────────────────────────────┐
       │  ggml-metal-ops.cpp : ggml_metal_op_flash_attn_ext             │
       │  ├─ ggml_metal_op_flash_attn_ext_use_vec(op)                   │
       │  │    heuristic: ne01 < 20 && ne00 % 32 == 0                   │
       │  ├─ parse op_params: scale, max_bias, logit_softcap            │
       │  ├─ compute m0, m1, n_head_log2 (ALiBi)                        │
       │  ├─ allocate bid_pad, bid_blk, bid_tmp from dst->data tail     │
       │  └─ branch on use_vec:                                         │
       └────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┴─────────────────────┐
        ▼                                           ▼
┌────────────────────────────────┐    ┌────────────────────────────────┐
│  TILE PATH (half8x8)           │    │  VEC PATH (half4 + shuffle)    │
│  nqptg=8, ncpsg=64             │    │  nqptg=1, ncpsg=32             │
│  nsg ∈ {4, 8} (ne00 >= 512 ? 8)│    │  nsg ∈ {1, 2, 4} (from ne11)   │
│  nwg = 1 (single workgroup)    │    │  nwg = 32 (fixed)              │
│                                │    │                                │
│  Pre-passes (optional):        │    │  Pre-passes (optional):        │
│  1. flash_attn_ext_pad         │    │  1. flash_attn_ext_pad         │
│  2. flash_attn_ext_blk         │    │                                │
│                                │    │  Post-pass (if nwg > 1):       │
│  Main: kernel_flash_attn_ext   │    │  3. flash_attn_ext_vec_reduce  │
│       (1 dispatch)             │    │                                │
│                                │    │  Main: kernel_flash_attn_ext_  │
│                                │    │       vec (1 dispatch)         │
└────────────────────────────────┘    └────────────────────────────────┘
                              │
                              ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  Template instantiation table (ggml-metal.metal:7050-7779)     │
       │  8 dtypes × 15 (DK,DV) pairs × 2 kernels = ~240 entries        │
       │  Plus 80 vec entries                                            │
       │  Each entry: kernel_flash_attn_ext_<dtype>_dk<DK>_dv<DV>       │
       │  Pipeline name also encodes: mask/sinks/bias/scap/kvpad/bcm/   │
       │  ns10/ns20/nsg (+ nwg for vec) — total ~10 binary dimensions   │
       └────────────────────────────────────────────────────────────────┘
                              │
                              ▼
       ┌────────────────────────────────────────────────────────────────┐
       │  Per-kernel inside the .metal:                                  │
       │  ├─ kernel_flash_attn_ext → kernel_flash_attn_ext_impl<...NSG> │
       │  │   (template impl, lines 6353-6961)                          │
       │  ├─ kernel_flash_attn_ext_vec (template, 7199-7827)            │
       │  ├─ kernel_flash_attn_ext_pad (6180-6243)                      │
       │  ├─ kernel_flash_attn_ext_blk (6252-6305)                      │
       │  └─ kernel_flash_attn_ext_vec_reduce (7787-7827)               │
       └────────────────────────────────────────────────────────────────┘
```

Key design points:

* **Two-kernel taxonomy, not three.** Unlike CUDA's `VEC / TILE /
  MMA_F16` (ARTX11-F01), Metal has only `VEC` and `TILE`. Apple's
  `simdgroup_multiply_accumulate` *is* Metal's tensor-core analog — it
  maps to the GPU's matrix-multiply unit on Apple7+ hardware — so there
  is no separate "MMA" path. The vec kernel uses `simd_shuffle_down`
  instead of simdgroup_matrix because at `nqptg=1` (single query) the
  QK^T outer product degenerates to a vector dot product, and shuffles
  are cheaper than simdgroup matrix setup.
* **Per-(dtype, DK, DV) template explosion.** 8 KV dtypes × 15
  (DK,DV) shape pairs = 120 instantiations of `kernel_flash_attn_ext`,
  plus 80 instantiations of `kernel_flash_attn_ext_vec`. Each is a
  distinct compiled pipeline; the host caches them lazily under a
  string name that also encodes 10 binary function-constant dimensions
  (`has_mask`, `has_sinks`, `has_bias`, `has_scap`, `has_kvpad`,
  `bc_mask`, `ns10`, `ns20`, `nsg`, `nwg`). The theoretical cache size
  is in the thousands; in practice the dtype+shape dimensions dominate
  and the binary dimensions are mostly zero.
* **Causal / sliding-window mask is consumed but never constructed
  here.** The mask is a precomputed F16 tensor of shape
  `[seq_len, n_tokens, 1, n_batch]` passed via `op->src[3]`. Metal has
  no `diag_mask_inf` kernel — the model code (or the graph builder) is
  responsible for materialising the mask before FA runs. The
  `kernel_diag_f32` at `ggml-metal.metal:9936` is `GGML_OP_DIAG`
  (diagonal-matrix constructor), a completely different op.
* **MLA shapes supported, MLA aliasing not.** The `(DK,DV)` shape
  pairs `(192,128)`, `(320,256)`, `(576,512)` cover Deepseek-V3,
  MiMo-V2.5, and Mistral Small 4 — all MLA models. But Metal has no
  `V_is_K_view` shortcut; V must be a separate buffer. The host
  asserts `ne11 == ne21` and `ne12 == ne22` at `ggml-metal-ops.cpp:2675-2676`
  but does not check `K->data == V->data`.
* **Function-constant-driven branch elimination.** The 10 binary
  dimensions of the pipeline name (`has_mask`, `has_sinks`, etc.) are
  passed as `MTLFunctionConstantValues` at compile time, so the
  compiler eliminates the dead branches per specialization. The kernel
  reads them as `constant bool FC_flash_attn_ext_has_mask
  [[function_constant(FC_FLASH_ATTN_EXT + 0)]]` at `ggml-metal.metal:6307`.

---

## 5. Execution Flow

### 5.1 Top-level entry

`ggml_metal_op_flash_attn_ext` (`ggml-metal-ops.cpp:2650`) is invoked
from the per-node dispatch switch (`ggml-metal-ops.cpp:446`).

### 5.2 Parameter parsing and buffer setup

1. Read `scale`, `max_bias`, `logit_softcap` from `op->op_params[0..2]`
   (`ggml-metal-ops.cpp:2686-2688`). If `logit_softcap != 0`, divide
   `scale` by it (`:2691`) so the kernel can apply
   `logit_softcap * tanhf(KQ)` after the matmul.
2. Compute `n_head_log2 = 1 << floor(log2(n_head))`, then
   `m0 = 2^(-max_bias/n_head_log2)`, `m1 = 2^(-(max_bias/2)/n_head_log2)`
   (`:2700-2703`). These are the ALiBi slope bases; the per-head
   exponent is computed in-kernel via `pow(base, exph)`.
3. Allocate three sub-buffers carved out of the *tail* of `dst->data`:
   `bid_pad` (KV-cache tail padding), `bid_blk` (mask block-state
   buffer), `bid_tmp` (vec-path multi-workgroup combine buffer). Sizes
   computed by `ggml_metal_op_flash_attn_ext_extra_pad/blk/tmp`
   (`:2536-2648`). This is the same "extra data appended to dst" trick
   CUDA uses (ARTX11-F06) — one allocation, no scratch pool.
4. Branch on `ggml_metal_op_flash_attn_ext_use_vec(op)`.

### 5.3 Tile path (prefill)

`ggml-metal-ops.cpp:2724-2890`:

1. Set `nqptg = OP_FLASH_ATTN_EXT_NQPSG = 8`, `ncpsg = OP_FLASH_ATTN_EXT_NCPSG = 64`.
2. **Optional KV-pad pre-pass** (`has_kvpad = ne11 % ncpsg != 0`):
   launch `kernel_flash_attn_ext_pad` (`ggml-metal.metal:6180`) which
   copies the trailing partial chunk of K, V, and mask into `bid_pad`,
   zero-filling K/V past the end and writing `-MAXHALF` into the mask
   tail. Grid: `(ncpsg, max(ne12, ne32), max(ne13, ne33))`, 32 threads
   per threadgroup. Calls `ggml_metal_op_concurrency_reset` after to
   insert a host-side barrier so the main kernel does not overlap with
   the pad write.
3. **Optional mask-prune pre-pass** (`has_mask`): launch
   `kernel_flash_attn_ext_blk` (`ggml-metal.metal:6252`). This kernel
   classifies each `C × Q` mask block into 0 (fully masked, skip), 1
   (active, has at least one non-`-MAXHALF` value), or 2 (all zero
   mask — special-cased because zero is a *valid* mask value but the
   block can be elided in the softmax-multiply because multiplying by
   zero contributes nothing). The result is stored as `int8_t` in
   `bid_blk`. Grid: `(nblk0, nblk1, ne32*ne33)`, 32 threads per TG.
   Another `concurrency_reset` follows.
4. **Main FA dispatch**:
   `nsg = ne00 >= 512 ? 8 : 4` (`:2835`, the hardcoded heuristic — see
   Finding ARTX17-F12). Compute `smem = FATTN_SMEM(nsg)` via the macro
   `:2817`:
   `GGML_PAD((nqptg*(ne00 + 2*GGML_PAD(ne20,64) + 2*(2*ncpsg)) + is_q*(16*32*nsg)) * sizeof(half), 16)`.
   Build the args struct `ggml_metal_kargs_flash_attn_ext` (32 fields).
   Look up pipeline by name
   `kernel_flash_attn_ext_<dtype>_dk<DK>_dv<DV>_mask=.._sinks=.._bias=.._scap=.._kvpad=.._bcm=.._ns10=.._ns20=.._nsg=..`.
   Dispatch `(ceil(ne01/nqptg), ne02, ne03, 32, nsg, 1)`.

### 5.4 Vec path (decode)

`ggml-metal-ops.cpp:2891-3074`:

1. Set `nqptg = 1`, `ncpsg = 32`, `nhptg = 1` (one head per TG).
2. Assert `ne10 >= ne20` (K head dim ≥ V head dim; the vec kernel
   reuses the K dequant path for V when they match, see
   `:2948`).
3. **Optional KV-pad pre-pass** (same as tile path; uses the same
   `kernel_flash_attn_ext_pad`).
4. **No mask-prune pre-pass** — the comment at `:2605` says "this
   optimization is not useful for the vector kernels", but the buffer
   is still reserved (`extra_blk` always returns non-zero when
   `has_mask`) to avoid graph reallocations if the heuristic changes.
5. Compute `nsg = 1` and `nwg = 32` (fixed):
   ```
   nwg = 32; nsg = 1;
   while (2*nwg*nsg*ncpsg < ne11 && nsg < 4) nsg *= 2;
   ```
   So for short KV caches (ne11 < 64) nsg stays at 1; for ne11 ≥ 64
   nsg becomes 2; for ne11 ≥ 128 nsg becomes 4.
6. Compute `smem = FATTN_SMEM(nsg)` via the macro `:2957`:
   `GGML_PAD(((GGML_PAD(ne00, 128) + 4*ncpsg + 2*GGML_PAD(ne20, 128)) * nsg) * sizeof(half), 16)`.
7. Look up pipeline by name
   `kernel_flash_attn_ext_vec_<dtype>_dk<DK>_dv<DV>_mask=.._sink=.._bias=.._scap=.._kvpad=.._ns10=.._ns20=.._nsg=.._nwg=..`.
8. **If `nwg == 1`**: write directly to `dst`. Single dispatch.
9. **If `nwg > 1`**: write partial results + `(S, M)` meta to
   `bid_tmp`. Then dispatch `kernel_flash_attn_ext_vec_reduce`
   (`ggml-metal.metal:7787`) to combine the 32 partials via online
   softmax rescaling: `m = simd_max(M_i)`, `ms_i = exp(M_i - m)`,
   `dst = Σ ms_i * partial_i / Σ ms_i * S_i`. Grid: `(nrows, 1, 1)`,
   `32*nwg` threads per TG.

### 5.6 RoPE execution

`ggml_metal_op_rope` (`ggml-metal-ops.cpp:3547`):

1. Read 15 op_params: `n_past`, `n_dims`, `mode`, `n_ctx` (unused,
   GLM-only), `n_ctx_orig`, `freq_base`, `freq_scale`, `ext_factor`,
   `attn_factor`, `beta_fast`, `beta_slow`, `sect_0..3` (mRoPE
   section sizes).
2. Switch on `mode` (parsed in `ggml-metal-device.cpp:1727-1732`):
   * `GGML_ROPE_TYPE_NEOX` → `kernel_rope_neox_<T>`
   * `GGML_ROPE_TYPE_MROPE | IMROPE` → `kernel_rope_multi_<T>`
     (mrope sector layout t/h/w/e; imrope interleaves sectors mod 3)
   * `GGML_ROPE_TYPE_VISION` → `kernel_rope_vision_<T>` (2D mRoPE
     layout, only sect_0 + sect_1; uses `args.n_dims` not
     `args.n_dims/2` for the x1 stride)
   * else (NeoX bit clear) → `kernel_rope_norm_<T>` (interleaved
     layout)
3. `T` is `float` or `half` based on `op->src[0]->type`. **No BF16
   RoPE** — only f32 and f16 instantiations exist
   (`ggml-metal.metal:4855-4865`).
4. Dispatch `(ne01, ne02, ne03, nth, 1, 1)` where
   `nth = min(1024, ne00)`.

### 5.7 Softmax execution

`ggml_metal_op_soft_max` (`ggml-metal-ops.cpp:1300`):

1. Read `scale`, `max_bias` from `op->op_params[0..1]`.
2. Compute `n_head_log2`, `m0`, `m1` (same as FA path).
3. Pick `nth ∈ {32, 64, 128, ...}`: start at 32, double while
   `nth < ne00/4 && nth * ne01 * ne02 * ne03 < 256` (the second
   condition caps thread count when there are many rows). The `/4`
   variant is for the `_4` vectorized kernel.
4. Look up pipeline `kernel_soft_max_<src1_type>[_4]` where
   `src1_type` is the mask dtype (F16 or F32). Note: even when there
   is no mask, the kernel still reads `src1` (aliased to `src0`) — the
   `src1 != src0` check at `ggml-metal.metal:1971` is the runtime
   mask-present test.
5. Dispatch `(ne01, ne02, ne03, nth, 1, 1)`.

### 5.8 SSM execution

`ggml_metal_op_ssm_conv` (`:1390`): if `ne1 > 1` (prefill), pick a
`BATCH_SIZE` ∈ {2, 8, 16, 32, 64, 128, 256} from a power-of-2 ladder
and dispatch `kernel_ssm_conv_f32_f32_batched[_4]` with
`(ne01, ceil(ne1/BATCH_SIZE), ne02, BATCH_SIZE, 1, 1)`. Otherwise
(decode) dispatch `kernel_ssm_conv_f32_f32[_4]` with
`(ne01, ne1, ne02, 1, 1, 1)`.

`ggml_metal_op_ssm_scan` (`:1463`): single dispatch of
`kernel_ssm_scan_f32` with `(d_inner, n_head, n_seqs*n_seq_tokens, 32, 1, 1)`.

---

## 6. Data Layout

### 6.1 KV cache

The FA kernels assume (and the host asserts) the canonical ggml
attention layout for K and V:

| Dim | Meaning                | Stride variable        |
| --- | ---------------------- | ---------------------- |
| 0   | `head_dim` (DKQ or DV) | `nb11` (K), `nb21` (V) |
| 1   | `seq_len` (KV cache)   | `nb12` (K), `nb22` (V) |
| 2   | `n_kv_heads`           | `nb13` (K), `nb23` (V) |
| 3   | `n_batch` (sequences)  | (implicit in `nb13`)   |

The args struct (`ggml-metal-impl.h:365-398`) carries `ne_12_2` and
`ne_12_3` as the K=V-shared `ne[2]` and `ne[3]` — the host asserts
`ne12 == ne22` and `ne13 == ne23` at `ggml-metal-ops.cpp:2675-2676`,
so the kernel uses these fields to index both K and V. Strides `ns10`
and `ns20` are precomputed as `nb11/nb10` and `nb21/nb20` (the
innermost stride ratio, in elements) — these are passed as int32
function constants (`FC_FLASH_ATTN_EXT + 20/21`) so the kernel can
compute K/V addresses with integer arithmetic.

The host asserts `op->src[0]->type == GGML_TYPE_F32` (Q is always
F32) and `op->src[1]->type == op->src[2]->type` (K and V share a
dtype) at `:2671-2672`. Output `op->type` is always F32.

### 6.2 Q layout

Q is F32, shape `[head_dim, n_tokens, n_q_heads, n_batch]`. The tile
kernel loads Q into threadgroup memory once at the start as `half4`
(via `(q4_t) q4[i]` cast at `ggml-metal.metal:6452`) — even though Q
is F32 on the device, the threadgroup copy is F16 to halve the shmem
footprint. This is the same F32→F16 precision trade-off as the
`mul_mm` legacy path (ARTX16-W6).

The vec kernel does the same: `sq4[i] = (q4_t) q4[i]` at
`ggml-metal.metal:7290`, where `q4_t` is `half4` (for F16/BF16/Q4_0
K) or `float4` (for F32 K).

### 6.3 Mask layout

Mask is F16, shape `[seq_len, n_tokens, 1, n_batch]`. The host asserts
`!op->src[3] || op->src[3]->type == GGML_TYPE_F16` at
`ggml-metal-ops.cpp:2678` and `!op->src[3] || op->src[3]->ne[1] >=
op->src[0]->ne[1]` at `:2679-2680` (mask must be at least n_queries
big). The kernel indexes it as `pm2[jj] = (device const half2 *)
(mask + (iq1+jj*NSG+sgitg)*args.nb31 + (iq2%ne32)*nb32 +
(iq3%ne33)*nb33)` — note the `iq2%ne32` and `iq3%ne33` modulo
operations implement broadcast across Q heads (if `ne32 < n_q_heads`,
the same mask plane is reused).

A second bounds-check flag `bc_mask = op->src[3] && op->src[3]->ne[1]
% 8 != 0` (`ggml-metal-device.cpp:1416`) is passed as a function
constant. When `bc_mask` is true, the kernel does an extra bounds
check `(iq1 + j) < args.ne31 ? pm2[jj][tiisg] : half2(-MAXHALF,
-MAXHALF)` at `ggml-metal.metal:6556` to avoid reading past the mask
row when the threadgroup's Q tile extends past the mask's n_queries.

### 6.4 Sinks

`op->src[4]` (when non-null) is a per-head F32 sink value (Gemma-style
attention sink). Shape `[n_q_heads]` (1D). The kernel reads it as
`((device const float *) sinks)[iq2]` at `ggml-metal.metal:6919` and
`7563`, treating it as an extra "always-present" logit that
participates in the online softmax max/sum. This is fused into the FA
kernel itself — there is no separate sink-application pass.

### 6.5 RoPE position layout

`op->src[1]` is the position tensor, `int32_t`, shape `[n_tokens,
n_q_heads, ...]` for norm/neox (one position per token), or
`[n_tokens * 4, n_q_heads, ...]` for mrope/vision (four positions per
token: t/h/w/e). The host asserts `ne10 % ne02 == 0 && ne10 >= ne02`
at `ggml-metal-ops.cpp:3561-3562`.

`op->src[2]` (optional) is a per-channel frequency factor, F32, shape
`[n_dims/2]`. Used by Deepseek-style RoPE with non-uniform frequency
scaling.

---

## 7. Memory Layout

### 7.1 Output (`dst` / KQV)

F32, shape `[DV, n_tokens, n_q_heads, n_batch]`. The tile kernel
writes directly to `dst` (single workgroup per output tile). The vec
kernel writes to `bid_tmp` (F32, `nwg * (DV + 2)` floats per row) when
`nwg > 1`, then `kernel_flash_attn_ext_vec_reduce` reads the partials
and writes the final F32 result to `dst`. When `nwg == 1` the vec
kernel writes directly to `dst`.

### 7.2 "Extra data" tail of `dst->data`

The three sub-buffers `bid_pad`, `bid_blk`, `bid_tmp` are carved out
of the *tail* of `dst->data`:

```
dst->data  ┌────────────────────────────┐
           │ F32 output (n_tokens *      │
           │   n_q_heads * DV * n_batch) │
           ├────────────────────────────┤
bid_pad    │ KV-cache tail padding       │  (extra_pad)
           │   K-pad + V-pad + mask-pad  │
           ├────────────────────────────┤
bid_blk    │ mask block-state buffer     │  (extra_blk)
           │   int8_t[nblk0*nblk1*ne32*  │
           │          ne33]              │
           ├────────────────────────────┤
bid_tmp    │ vec-path partial results    │  (extra_tmp)
           │   F32[ne01_max * ne02 *     │
           │   ne03 * nwg * (DV + 2)]    │
           └────────────────────────────┘
```

Sizes are computed up-front by
`ggml_metal_op_flash_attn_ext_extra_pad/blk/tmp`
(`:2536-2648`). The backend allocator uses these to size `dst->data`
once at graph-plan time; no per-call scratch allocation. This is the
same "extra data appended to dst" trick as CUDA (ARTX11-F06), and it
is critical for avoiding allocation overhead in the hot path.

### 7.3 Shared memory

| Kernel                 | smem size (bytes)                                       | Source                         |
| ---------------------- | ------------------------------------------------------- | ------------------------------ |
| `flash_attn_ext` (tile) | `FATTN_SMEM(nsg)`, typically 4–16 KiB                  | `ggml-metal-ops.cpp:2817`     |
| `flash_attn_ext_vec`   | `FATTN_SMEM(nsg)`, typically 1–4 KiB                   | `ggml-metal-ops.cpp:2957`     |
| `flash_attn_ext_pad`   | 0 (no shmem parameter)                                  | `ggml-metal.metal:6180`       |
| `flash_attn_ext_blk`   | 0 (no shmem parameter)                                  | `ggml-metal.metal:6252`       |
| `flash_attn_ext_vec_reduce` | 0 (no shmem parameter)                             | `ggml-metal.metal:7787`       |
| `soft_max[_4]`         | `32 * sizeof(float) = 128`                              | `ggml-metal-device.cpp:475`   |
| `rope_*`               | 0 (no shmem parameter)                                  | `ggml-metal.metal:4595+`      |
| `ssm_conv_*`           | 0 (no shmem parameter)                                  | `ggml-metal.metal:2172+`      |
| `ssm_scan_f32`         | `sgptg * NW + 2*sgptg` floats (≤ 256 B)                 | `ggml-metal.metal:2353-2355`  |

The largest allocation is the FA-tile kernel at ~16 KiB for `nsg=8`,
which is well within the 32 KiB Metal limit on Apple7+ hardware. The
vec kernel is much smaller (1–4 KiB) because it uses a per-simdgroup
layout instead of a per-threadgroup tile.

### 7.4 Threadgroup tile layout (`flash_attn_ext` tile)

`ggml-metal.metal:6402-6416` defines a 6-region layout in a single
`threadgroup half * shmem_f16` buffer:

| Region | Symbol | Offset (half-elements)                  | Size                          |
| ------ | ------ | --------------------------------------- | ----------------------------- |
| Q      | `sq`/`sq4` | `0`                                  | `Q * DK`                      |
| O      | `so`/`so4` | `Q*DK`                              | `Q * PV` (PV = pad(DV,64))    |
| S scratch | `ss`/`ss2` | `Q*T` (T = DK + 2*PV)             | `Q * 2*SH` (SH = 2*C)         |
| K scratch | `sk`/`sk4x4` | `sgitg*(4*16*KV) + Q*T + Q*TS`   | `4*16*KV` per simdgroup       |
| V scratch | `sv`/`sv4x4` | same as sk (overlapped)          | `4*16*KV` per simdgroup       |
| Mask   | `sm2`  | `Q*T + 2*C`                             | `C` half2 elements            |

`sk` and `sv` share the same offset — they are loaded at different
times in the KV loop (K for QK^T, then V for PV), so the overlap is
safe. This layout is identical to the one audited in ARTX16 §6.5; the
detail specific to attention is the `Mask` region, which holds one
`C`-element half2 tile of the mask for the current KV iteration.

### 7.5 Threadgroup tile layout (`flash_attn_ext_vec`)

`ggml-metal.metal:7265-7269`:

| Region | Symbol | Offset (half-elements)            | Size                |
| ------ | ------ | --------------------------------- | ------------------- |
| Q      | `sq4`  | `0`                               | `PK` (pad(DK,128))  |
| S scratch | `ss`/`ss4` | `sgitg*SH + NSG*PK`        | `SH = 4*C` per simdgroup |
| Mask   | `sm`   | `sgitg*SH + 2*C + NSG*PK`        | `C` half elements   |
| O      | `so4`  | `2*sgitg*PV + NSG*PK + NSG*SH`   | `PV = pad(DV,128)` per simdgroup |

The vec kernel's shmem is `NSG * (PK + SH + PV) * sizeof(half)` ≈
`NSG * (DK + 4*C + DV) * 2` bytes. For DK=DV=128, NSG=4, C=32:
`4 * (128 + 128 + 128) * 2 = 3072` bytes.

---

## 8. Parallelism Strategy

### 8.1 Three grid axes

| Kernel           | Grid X                              | Grid Y          | Grid Z                |
| ---------------- | ----------------------------------- | --------------- | --------------------- |
| `flash_attn_ext` (tile) | `ceil(ne01/nqptg)` = `ceil(n_tokens/8)` | `ne02` (n_q_heads) | `ne03` (n_batch)   |
| `flash_attn_ext_vec` | `ceil(ne01/nqptg)` = `n_tokens` (nqptg=1) | `ceil(ne02/nhptg)` = `n_q_heads` | `ne03 * nwg` |
| `flash_attn_ext_pad` | `ncpsg` (64 or 32)              | `max(ne12, ne32)` | `max(ne13, ne33)` |
| `flash_attn_ext_blk` | `nblk0 = ceil(ne30/ncpsg)`       | `nblk1 = ceil(ne01/nqptg)` | `ne32*ne33` |
| `flash_attn_ext_vec_reduce` | `nrows = ne1*ne2*ne3`     | 1               | 1                     |
| `soft_max`       | `ne01`                              | `ne02`          | `ne03`                |
| `rope_*`         | `ne01`                              | `ne02`          | `ne03`                |
| `ssm_conv` (decode) | `ne01`                           | `ne1`           | `ne02`                |
| `ssm_conv_batched` | `ne01`                           | `ceil(ne1/BS)`  | `ne02`                |
| `ssm_scan`       | `d_inner`                          | `n_head`        | `n_seqs*n_seq_tokens` |
| `set_rows_*`     | `ceil(ne01/tpg.y)`                 | `ne02`          | `ne03`                |

The vec kernel's Z axis carries `nwg` (32 workgroups per output row)
— they cooperate via the `bid_tmp` buffer and the
`flash_attn_ext_vec_reduce` post-pass.

### 8.2 Within-threadgroup parallelism

* **Tile kernel**: `nsg ∈ {4, 8}` simdgroups per TG (128 or 256
  threads). Each simdgroup owns `NQ = Q/NSG` queries (1 or 2) and
  cooperates on the K/V tile loads via shmem. The QK^T outer product
  uses `simdgroup_multiply_accumulate(mqk, mq, mk, mqk)` with 4 K-tiles
  + 4 Q-tiles per inner iteration (when `DK % 16 == 0`, else 1+1).
  The PV matmul uses 1-4 V-tiles per iteration.
* **Vec kernel**: `nsg ∈ {1, 2, 4}` simdgroups per TG. Each simdgroup
  owns the full Q row (single query) and a disjoint slice of the KV
  cache. Intra-row reduction uses `simd_shuffle_down` (5-step tree at
  `:7518-7550`) — no shmem for the reduction.
* **Pad/blk/reduce kernels**: single simdgroup (32 threads) per TG.

### 8.3 Cross-workgroup cooperation (vec path)

The vec kernel's `nwg = 32` workgroups per output row each compute a
partial `(S_i, M_i, O_i)` triplet. They write to disjoint regions of
`bid_tmp` (no atomics). The `flash_attn_ext_vec_reduce` kernel then
reads all 32 partials and produces the final output via:

```
m = simd_max(M_i over i=0..31)
ms_i = exp(M_i - m)
S = Σ ms_i * S_i
dst = (Σ ms_i * partial_i) / S
```

This is the same online-softmax combine as CUDA's
`flash_attn_combine_results`, but with `nwg = 32` hardcoded instead of
the CUDA `parallel_blocks` autotuned value. The reduce kernel uses
`32 * nwg = 1024` threads per TG, with each `tiisg` (lane in the
simdgroup) owning one of the `nwg = 32` workgroups' partials.

### 8.4 Cross-encoder parallelism

The pad, blk, and main FA dispatches in the tile path each call
`ggml_metal_op_concurrency_reset(ctx)` after launching. This forces
the next dispatch to wait for the previous to complete (no concurrent
encoder overlap). The comment at `:2804` says this is needed because
the pad and blk kernels write to `bid_pad` / `bid_blk` which the main
kernel reads — the overlap tracker (ARTX15) cannot see this
intra-node dependency, so the host inserts an explicit barrier.

---

## 9. SIMD / GPU Strategy

### 9.1 `simdgroup_matrix<float, 8>` and `simdgroup_multiply_accumulate`

The tile kernel uses `simdgroup_half8x8` for Q, K, V tile storage in
shmem, and `simdgroup_float8x8` for the QK^T and PV accumulators.
This is the same API as the `mul_mm` legacy path (ARTX16 §9.1). Each
`simdgroup_multiply_accumulate` performs an 8×8 × 8×8 → 8×8 matrix
multiply-accumulate in the simdgroup's register file, mapping to the
GPU's matrix-multiply unit on Apple7+ hardware.

### 9.2 QK^T matmul (tile kernel)

`ggml-metal.metal:6608-6645`:

```
FOR_UNROLL (short cc = 0; cc < NC; ++cc) {
    qk8x8_t mqk = make_filled_simdgroup_matrix<qk_t, 8>(0.0f);
    if (DK % 16 != 0) {
        // 1-ma + 1-mq per inner iteration
        for (short i = 0; i < DK8; ++i) {
            simdgroup_load(mk, pk + 8*i, NS10, 0, true); // transpose
            simdgroup_load(mq, pq + 8*i, DK);
            simdgroup_multiply_accumulate(mqk, mq, mk, mqk);
        }
    } else {
        // 2-ma + 2-mq per inner iteration (unrolled)
        for (short i = 0; i < DK8/2; ++i) {
            simdgroup_load(mq[0], pq + 0*8 + 16*i, DK);
            simdgroup_load(mq[1], pq + 1*8 + 16*i, DK);
            simdgroup_load(mk[0], pk + 0*8 + 16*i, NS10, 0, true);
            simdgroup_load(mk[1], pk + 1*8 + 16*i, NS10, 0, true);
            simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
            simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
        }
    }
    simdgroup_store(mqk, ps, SH, 0, false);
}
```

`NC = (C/8)/NSG` — each simdgroup owns `NC` 8-column KQ tiles within
the `C = 64` KV items per iteration. With `nsg=4`, `NC = 2`; with
`nsg=8`, `NC = 1`.

### 9.3 PV matmul (tile kernel)

`ggml-metal.metal:6765-6909`: two paths based on `DV`:

* **DV ≤ 64**: 1-vs-tile per inner iteration, with `NO = PV8/NSG`
  output tiles. Each iteration loads 1 `s8x8_t` (softmax P row) and 2
  `v8x8_t` (V tile) and does 2 `simdgroup_multiply_accumulate` calls.
* **DV > 64**: 2-vs-tiles + 4-mv-tiles per inner iteration, with `NO =
  PV8/NSG` output tiles. Each iteration loads 2 `s8x8_t` and 4
  `v8x8_t` and does 4 `simdgroup_multiply_accumulate` calls.

The DV>64 path doubles the inner-loop work to amortize the larger V
tile load. Both paths store the result via `simdgroup_store(lo[ii],
sot, PV, 0, false)`.

### 9.4 Vec kernel KQ reduction via `simd_shuffle_down`

`ggml-metal.metal:7518-7550`:

```
if (NE > 1)  lo[ii] += simd_shuffle_down(lo[ii], 16);
if (NE > 2)  lo[ii] += simd_shuffle_down(lo[ii],  8);
if (NE > 4)  lo[ii] += simd_shuffle_down(lo[ii],  4);
if (NE > 8)  lo[ii] += simd_shuffle_down(lo[ii],  2);
if (NE > 16) lo[ii] += simd_shuffle_down(lo[ii],  1);
```

`NE = 4` (template default) so only the first 3 shuffles execute. The
reduction is a 3-step tree: 16→8→4 lanes cooperate to produce the
final dot-product. This is the same pattern as `mul_mv_ext` (ARTX16-F06)
but specialized for the FA-vec case where each thread holds `NE=4`
elements of the Q row.

### 9.5 Online softmax in the tile kernel

`ggml-metal.metal:6717-6760`:

```
FOR_UNROLL (short jj = 0; jj < NQ; ++jj) {
    const short j = jj*NSG + sgitg;
    const float m = M[jj];
    float2 s2 = ss2[j*SH/2 + tiisg] * args.scale;
    if (FC_flash_attn_ext_has_scap) s2 = args.logit_softcap * precise::tanh(s2);
    if (blk_cur != 2) {
        if (FC_flash_attn_ext_has_bias) s2 += s2_t(sm2[j*SH + tiisg]) * slope;
        else                              s2 += s2_t(sm2[j*SH + tiisg]);
    }
    M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
    const float  ms  = exp(m  - M[jj]);
    const float2 vs2 = exp(s2 - M[jj]);
    S[jj] = S[jj]*ms + simd_sum(vs2[0] + vs2[1]);
    ss2[j*SH/2 + tiisg] = vs2;  // P matrix
    // rescale O
    for (short i = tiisg; i < DV4; i += NW) so4[j*PV4 + i] *= ms;
}
```

Note `precise::tanh` (MSL's precise-math namespace) for the logit
softcap. `blk_cur == 2` (all-zero mask block) skips the mask add
entirely because adding zero is a no-op — but the block-prune pre-pass
already guarantees the block was classified as 2 only when `mmin ==
0 && mmax == 0`, so the skip is safe.

### 9.6 RoPE scalar per thread

Each RoPE kernel is scalar per thread: `for (int i0 = 2*tiitg; i0 <
args.ne0; i0 += 2*tptg.x)`. Each thread computes one `(cos, sin)` pair
via `rope_yarn(theta, freq_scale, corr_dims, i0, ext_factor,
attn_factor, &cos_theta, &sin_theta)` (`ggml-metal.metal:4560-4578`)
and applies the 2×2 rotation to one `(x0, x1)` pair. The four RoPE
variants differ only in:

* **norm** (interleaved): `x0 = src[0], x1 = src[1]` (adjacent in
  memory).
* **neox** (split-half): `x0 = src[0], x1 = src[n_dims/2]` (split
  halves).
* **multi** (mRoPE): same as neox layout, but `theta_base` is
  selected from one of 4 position IDs (`pos[i2 + ne02 * {0,1,2,3}]`)
  based on which sector `ic % sect_dims` falls in. `imrope` variant
  interleaves sectors mod 3.
* **vision**: 2D-only mRoPE (uses `sect_0 + sect_1`, not all 4
  sectors). Stride is `args.n_dims` (not `n_dims/2`) because the
  vision layout is `[n_dims, 2, ...]` not `[n_dims/2, n_dims/2, ...]`.

YaRN length extrapolation is always-on: `rope_yarn_corr_dims` is
computed at the top of every kernel call
(`ggml-metal.metal:4609, 4662, 4715, 4798`), and `rope_yarn` applies
the ramp mixing + magnitude scaling per element when `ext_factor !=
0.0f`. When `ext_factor == 0.0f` the YaRN code is a no-op (theta is
not mixed, mscale is not applied).

### 9.7 Softmax two-pass

`kernel_soft_max` (`ggml-metal.metal:1950-2053`) is a standard
two-pass (max, then sum-exp) softmax. The first pass computes
`lmax = max(psrc0[i00]*scale + slope*pmask[i00])` over the row; the
second pass computes `lsum = Σ exp(... - max_val)` and writes
`pdst[i00] = exp(...)`. Final pass divides by `sum`.

The attention-sink `src[2]` is integrated as: `lmax = max(lmax,
psrc2[i02])` (initialized to sink before the row scan), and `sum +=
exp(psrc2[i02] - max_val)` after the row scan. This adds the sink
value as an extra "always-present" logit.

The `_4` variant (`:2056-2161`) uses `float4` loads/stores for 4×
fewer iterations when `ne00 % 4 == 0`.

### 9.8 Capability gating

* `simdgroup_matrix` (tile kernel) requires `MTLGPUFamilyApple7`.
* `bfloat` types (BF16 KV) require `GGML_METAL_HAS_BF16` (Apple6+ /
  Metal3+).
* FA `supports_op` requires `has_simdgroup_mm` (Apple7+).
* RoPE has no capability gating — runs on every Metal device.

---

## 10. Quantization Strategy

### 10.1 Supported KV dtypes

The FA family is instantiated for exactly 8 KV dtypes:

| Dtype | FA-tile | FA-vec | Notes |
| ----- | ------- | ------ | ----- |
| F32   | ✓       | ✓      | Uses `dequantize_f32` (identity) and `float4x4` thread storage. |
| F16   | ✓       | ✓      | Uses `dequantize_f16` and `half4x4` thread storage. |
| BF16  | ✓       | ✓      | Gated on `GGML_METAL_HAS_BF16`. Uses `bfloat4x4`. |
| Q4_0  | ✓       | ✓      | `nl_k = nl_v = 2`, `block_q4_0` device storage. |
| Q4_1  | ✓       | ✓      | `nl_k = nl_v = 2`, `block_q4_1`. |
| Q5_0  | ✓       | ✓      | `nl_k = nl_v = 2`, `block_q5_0`. |
| Q5_1  | ✓       | ✓      | `nl_k = nl_v = 2`, `block_q5_1`. |
| Q8_0  | ✓       | ✓      | `nl_k = nl_v = 2`, `block_q8_0`. |

**Unsupported** (despite being mentioned in the audit prompt):

* `Q4_K`, `Q5_K`, `Q6_K` — K-quants. Not instantiated.
* `Q2_K`, `Q3_K` — K-quants. Not instantiated.
* `IQ1_S`, `IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`,
  `IQ4_NL`, `IQ4_XS` — importance-measured quants. Not instantiated.
* `MXFP4` — MX FP4. Not instantiated in FA (only in `mul_mv` / `mul_mm`).
* `F8_E4M3`, `F8_E5M2` — FP8. Not instantiated. Apple Silicon has no
  native FP8 matmul instruction; supporting FP8 KV would require
  software dequant to F16 before the matmul.

`ggml_metal_device_supports_op` returns `false` for any FA op whose
`op->src[1]->type` is not in the 8-dtype whitelist
(`ggml-metal-device.m:1250-1266`).

### 10.2 Dequantize-on-load

The FA-tile kernel's quantized-K branch (`ggml-metal.metal:6652-6713`)
and quantized-V branch (`:6843-6907`) use a `deq_k` / `deq_v` function
pointer template parameter. Each thread dequantizes one 4×4 sub-tile
(`block_q4_0` etc.) into threadgroup shmem (`sk4x4[4*ty + tx]`), then
the simdgroup loads it via `simdgroup_load(mk, sk + 16*k + 0*8,
4*16, 0, true)` (note `true` = transpose). The dequantize is
per-thread, not per-simdgroup — same pattern as `mul_mm` (ARTX16 §10.2).

The comment at `:6653` and `:6844` says **"TODO: this is the quantized
K cache branch - not optimized yet"** — the quantized-K path uses
shmem staging and a full `simdgroup_barrier(mem_threadgroup)` per
inner iteration, whereas the F16/F32 K path reads directly from device
memory with no shmem staging. This is a known performance gap; the F16
path is the fast path, the quantized-K path is the slow path.

### 10.3 Quantize-on-write (`set_rows_q32`)

`kernel_set_rows_q32` (`ggml-metal.metal:9841-9870`) is the
quantize-on-write KV-cache-append kernel. It takes F32 src0, an int32
or int64 row-index src1, and writes quantized dst. The `quantize_func`
template parameter is one of `quantize_q8_0`, `quantize_q4_0`,
`quantize_q4_1`, `quantize_q5_0`, `quantize_q5_1`, `quantize_iq4_nl`
(see instantiations at `:9921-9934`). Each thread processes one
32-element block: `quantize_func(src_row + 32*ind, dst_row[ind])`.

This is the KV-cache-append path: the model produces a new K/V row as
F32, RoPE is applied (still F32), then `set_rows_q32` quantizes and
writes into the cache.

---

## 11. Correctness Analysis

### 11.1 FlashAttention-2 online softmax

Both the tile and vec kernels implement the standard online softmax:

```
m_new = max(m_old, max(KQ_new))
s_new = s_old * exp(m_old - m_new) + sum(exp(KQ_new - m_new))
VKQ_new = VKQ_old * exp(m_old - m_new) + sum_i(KQ_new[i] * exp(KQ_new[i] - m_new) * V[i])
```

The result is bit-exact only in infinite precision; in F32 it differs
from the textbook form at the ULP level due to reassociation of the
sum across KV iterations.

### 11.2 `-FLT_MAX/2` sentinel (no `KQ_MAX_OFFSET` shift)

Unlike CUDA's FA kernels (ARTX11-F05), Metal does **not** add a
`FATTN_KQ_MAX_OFFSET = 3·log(2)` shift to `KQ_max`. Instead, it
initialises `M = -FLT_MAX/2` (`ggml-metal.metal:6477, 7311`) so that
`exp(-FLT_MAX/2 - any_finite_logit)` underflows to 0 harmlessly
without producing NaN. The final division `dst = VKQ / S` is guarded
by `S == 0.0 ? 0.0 : 1.0/S` (`:6944, 7628, 7810`) so the all-masked
case produces 0 output instead of NaN.

There is also no `SOFTMAX_FTZ_THRESHOLD` bit-hack (ARTX11-F05). The
Metal kernels rely on the natural FTZ behavior of `expf` on Apple
Silicon (which flushes denormals to zero by default). This is simpler
than CUDA's approach but means the kernels may produce different ULP
results than CUDA on the same inputs.

### 11.3 Mask-prune block-state semantics

`kernel_flash_attn_ext_blk` classifies each `C × Q` mask block into
three states:

* **0** (skip): `mmax <= -MAXHALF`. The block is fully masked; the
  main kernel `continue`s past it (`:6548`).
* **1** (active): `mmax > -MAXHALF && !(mmin == 0 && mmax == 0)`. The
  block has at least one non-`-inf` value; the main kernel processes
  it normally.
* **2** (all-zero): `mmin == 0 && mmax == 0`. The block is all zeros
  (a valid mask value). The main kernel skips the mask-add (`blk_cur
  != 2` at `:6731`) because adding zero is a no-op — but it still
  computes the QK^T and PV matmuls because the softmax contribution
  of a zero-masked block is not zero (it's `exp(KQ) * V`).

State 2 is an optimisation: it avoids the mask load and add for blocks
that are known to be zero. It is correct only because the mask values
are *added* to KQ, not multiplied.

### 11.4 GQA divisibility

The host computes `gqa_ratio = Q->ne[2] / K->ne[2]` implicitly via
the `ne_12_2` field. The kernel computes `ikv2 = iq2 / (ne02 / ne_12_2)`
(`ggml-metal.metal:6437, 7277`) to map a Q head index to its KV head
index. There is no explicit assert that `ne02 % ne_12_2 == 0` in the
kernel — the host relies on the model code to produce divisible
shapes. If the shapes are non-divisible, `ikv2` will be silently
wrong (integer division truncates).

### 11.5 Causal / sliding-window mask

Causal masking is **outside** the FA kernels: the model code produces
an F16 mask tensor with `-inf` (or `-MAXHALF`) in masked positions,
then passes it as `op->src[3]`. The FA kernels add the mask to KQ
*before* the softmax max-reduction (`:6731-6737`):

```
if (blk_cur != 2) {
    if (FC_flash_attn_ext_has_bias) s2 += s2_t(sm2[j*SH + tiisg]) * slope;
    else                              s2 += s2_t(sm2[j*SH + tiisg]);
}
```

The `slope` is `1.0f` for non-ALiBi masks, and the ALiBi slope for
ALiBi masks. If the mask value is `-MAXHALF`, the corresponding
`exp(KQ - KQ_max)` underflows to 0 and the row-sum is unaffected.

There is **no** `flash_attn_sliding_window` parameter. Sliding-window
attention is realised by passing a precomputed F16 mask tensor with
the window pattern baked in. The mask-prune `flash_attn_ext_blk`
pre-pass skips fully-masked tiles, which helps for sliding-window
patterns with a small window over a long cache — but the dense mask
must still be materialised by the caller.

### 11.6 RoPE numerical precision

`rope_yarn` uses `cos` / `sin` (single-precision transcendental) on
the device (`ggml-metal.metal:4573-4574`). Same precision as CUDA
(ARTX11 §11.7). No F64 path. The YaRN magnitude scaling multiplies by
`1.0f + 0.1f * log(1.0f / freq_scale)` (`:4571`), also F32.

### 11.7 Non-determinism

* **Vec path with `nwg > 1`** is non-deterministic across runs at the
  ULP level, because the `flash_attn_ext_vec_reduce` kernel
  reassociates the partial sums across workgroups. The order of
  summation depends on the order in which workgroups complete, which
  is non-deterministic.
* **Tile path with `nsg > 1`** is deterministic per-threadgroup
  (single workgroup per output tile), but the per-threadgroup result
  depends on `nsg`, which is a function of `ne00`. Two runs with the
  same `ne00` produce the same result; two runs with different `ne00`
  may produce ULP-different results.
* **Concurrent encoder dispatch** (ARTX15): when `use_concurrency` is
  true, Metal may overlap dispatches that write to disjoint buffers.
  Each dispatch is internally deterministic; the cross-dispatch order
  does not affect results because the overlap tracker prevents
  conflicts. The pad/blk/main/reduce dispatches in the FA path are
  *not* overlapped (the host inserts `concurrency_reset` between
  them — see §8.4).

### 11.8 Atomic accumulation

None in the FA kernels. Output tiles are written by exactly one
threadgroup each (tile path) or combined via the reduce kernel (vec
path). The reduce kernel uses `simd_sum` (intra-simdgroup) and
disjoint output regions (inter-simdgroup), no atomics.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                | Where                                       | Notes |
| ------------------------------------------- | ------------------------------------------- | ----- |
| Two-kernel FA dispatch (tile vs vec)        | `ggml-metal-ops.cpp:2526-2534`              | Heuristic: `ne01 < 20 && ne00 % 32 == 0`. |
| Per-(dtype,DK,DV) template instantiation   | `ggml-metal.metal:7050-7779`                | 8 dtypes × 15 shapes × 2 kernels = 240 instantiations. |
| Function-constant branch elimination        | `ggml-metal.metal:6307-6321`                | 10 binary FCs eliminate dead branches per specialization. |
| Mask-prune pre-pass (`flash_attn_ext_blk`)  | `ggml-metal.metal:6252-6305`                | Three-state (skip/active/all-zero) block classifier. |
| KV-tail pad pre-pass (`flash_attn_ext_pad`) | `ggml-metal.metal:6180-6243`                | Copies partial KV chunk into padded buffer with `-MAXHALF` mask fill. |
| Vec-path multi-workgroup combine            | `ggml-metal.metal:7787-7827`                | `nwg=32` partials reduced via online-softmax rescaling. |
| Online softmax (FlashAttention-2)           | Both kernels                                | `M`, `S`, `O` per-thread / per-simdgroup. |
| `simdgroup_multiply_accumulate` outer product | `ggml-metal.metal:6608-6909`              | 2-ma + 2-mq (DK%16==0) or 1-ma + 1-mq (else) per inner iteration. |
| `simd_shuffle_down` reduction (vec)         | `ggml-metal.metal:7518-7550`                | 3-step tree for `NE=4`. |
| Always-on YaRN in RoPE                      | `ggml-metal.metal:4560-4578`                | `rope_yarn` helper called per element in all 4 RoPE variants. |
| Attention sinks baked into FA               | `ggml-metal.metal:6914-6932, 7561-7577`     | `src[4]` per-head sink added to softmax max/sum. |
| ALiBi slope precomputed on host             | `ggml-metal-ops.cpp:2702-2703`              | `m0`, `m1` bases passed as args; per-head `pow` in kernel. |
| Logit-softcap (Gemma) via `precise::tanh`   | `ggml-metal.metal:6727`                     | `s2 = logit_softcap * precise::tanh(s2)`. |
| `set_rows_q32` quantize-on-write            | `ggml-metal.metal:9841-9870`                | 6 quant funcs × 2 idx dtypes = 12 instantiations. |
| Extra-data tail in `dst->data`              | `ggml-metal-ops.cpp:2715-2722`              | Pad/blk/tmp carved from dst tail; no scratch pool. |
| `soft_max_4` vectorized variant             | `ggml-metal.metal:2056-2161`                | `float4` loads when `ne00 % 4 == 0`. |
| SSM conv batched kernel for prefill         | `ggml-metal-ops.cpp:1425-1447`              | Power-of-2 batch size ladder from 2 to 256. |
| `_t4` dequantize variants for FA-vec        | `ggml-metal.metal:7669-7779`                | `float4` output for FA-vec dequant path. |

### 12.2 Optimizations *not* present

* **No `V_is_K_view` MLA shortcut.** Unlike CUDA (ARTX11-F04), Metal
  does not alias V to K's pointer for MLA models. V is always loaded
  from its own buffer. This costs ~2× the memory bandwidth for MLA
  models (Deepseek-V3, MiMo-V2.5, Mistral Small 4).
* **No FP8 KV path.** Apple Silicon has no native FP8 matmul, but a
  software dequant-to-F16 path would still save memory bandwidth. Not
  implemented.
* **No K-quant / IQ-quant KV path.** Only Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
  are supported in FA. K-quants (Q4_K etc.) and IQ-quants (IQ4_NL
  etc.) require a dequant-to-F16 conversion before FA, defeating the
  bandwidth benefit.
* **No fused `ROPE+VIEW+SET_ROWS`.** CUDA has this fusion
  (ARTX11-F07); Metal does not. Each RoPE call is a separate dispatch,
  and each KV-cache append is a separate `set_rows` dispatch.
* **No `nsg` autotuner.** The commented-out code at
  `ggml-metal-ops.cpp:2819-2834` shows the intent: compute `nsgmax`
  from `max_theadgroup_memory_size` and pick the largest `nsg` that
  fits. The heuristic is hardcoded `ne00 >= 512 ? 8 : 4`.
* **No double-buffering** in the FA-tile K loop. The K/V tile is
  loaded, then a `threadgroup_barrier`, then compute. No overlap.
* **No software prefetch.** All loads are demand loads.
* **No persistent kernel.** Each FA launch is one-shot.
* **No graph-level FA fusion.** The plan-time optimizer (ARTX15-F11)
  does not include any FA-related pattern. The closest is the
  `set_rows` quantize-on-write, which is fusion of the KV-cache-append
  but not of attention itself.
* **No BF16 RoPE.** Only `f32` and `f16` RoPE instantiations exist.
  BF16 RoPE would require adding `bfloat` template instantiations and
  a `bfloat`-typed `cos`/`sin` path (or upcast to F32 for the
  transcendentals).
* **No `diag_mask_inf` kernel.** Metal relies on the model code to
  materialise the mask before FA runs. CUDA has a standalone
  `diagmask.cu` kernel for the legacy path; Metal does not.

---

## 13. Architectural Strengths

1. **Two-kernel FA taxonomy is the right granularity for Apple
   Silicon.** The tile kernel uses `simdgroup_matrix` (the Apple
   tensor-core analog); the vec kernel uses `simd_shuffle_down`. The
   `ne01 < 20` heuristic is simple and covers the decode case
   (`ne01 = 1`) and the small-batch prefill case. There is no need
   for a third "MMA" kernel because `simdgroup_multiply_accumulate`
   already maps to the matrix-multiply unit.

2. **Mask-prune `flash_attn_ext_blk` three-state classifier.** The
   third state (all-zero) is a Metal-specific optimisation that
   avoids the mask-add for blocks known to be zero. This is a small
   but real win for models with sparse attention patterns (e.g.
   Mistral's sliding window with a 4096-token window over a 32k cache).

3. **KV-tail `flash_attn_ext_pad` pre-pass.** Splitting the
   KV-tail-padding into a separate kernel keeps the main FA kernel
   simple (no bounds-checking on the KV dimension inside the hot
   loop). The pad buffer is carved from `dst->data` tail, so no
   scratch allocation.

4. **Always-on YaRN in RoPE.** The `rope_yarn` helper is shared
   across all four RoPE variants, so length-extrapolation fixes
   propagate for free. This is cleaner than CUDA's separate `yarn`
   code path (which is also shared, but the Metal version is more
   compact).

5. **Attention sinks baked into FA.** The `src[4]` per-head sink is
   fused into the FA kernel itself, not a separate softmax-cap pass.
   This is the right design: the sink participates in the online
   softmax max/sum, so fusing it avoids a separate reduction.

6. **`set_rows_q32` quantize-on-write.** Fusing the quantize into
   the KV-cache-append kernel avoids a separate F32→quant conversion
   dispatch. The 6 quant-func × 2 idx-dtype instantiation table
   covers all common cases.

7. **Extra-data tail in `dst->data`.** The pad/blk/tmp sub-buffers
   carved from the dst tail avoid per-call scratch allocation. This
   is the same trick as CUDA (ARTX11-F06) and is critical for
   avoiding allocator overhead in the hot path.

8. **Per-(dtype,DK,DV) template explosion with lazy pipeline
   cache.** 240 FA-tile + 80 FA-vec instantiations is a lot, but the
   lazy cache means only the shapes actually used get compiled. The
   string-keyed cache also encodes the 10 binary function-constant
   dimensions, so each specialization is fully dead-branch-eliminated.

9. **SSM conv batched kernel.** The power-of-2 batch size ladder
   (2, 8, 16, 32, 64, 128, 256) for prefill is a clean way to
   amortize threadgroup dispatch overhead across multiple tokens.

---

## 14. Architectural Weaknesses

### W1 — No FP8 / K-quant / IQ-quant KV support

**Evidence:** `ggml-metal-device.m:1250-1266` whitelists only F32,
F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0. No K-quants, no IQ-quants,
no MXFP4, no FP8.

**Impact:** Models that store KV cache in Q4_K (a common
memory-saving choice for long contexts) must use F16 KV on Metal,
paying 2× the memory bandwidth. Models that store KV in FP8 (some
vLLM-exported Llama-3 variants) are not supported at all on Metal.

### W2 — No `V_is_K_view` MLA shortcut

**Evidence:** No `V_is_K_view` flag, no `view_src` aliasing in the FA
kernels. The MLA shapes (DK,DV) = (192,128), (320,256), (576,512) are
supported via template parameters, but V is always loaded from its
own buffer.

**Impact:** For MLA models (Deepseek-V3, MiMo-V2.5, Mistral Small 4,
GLM-4.7-Flash), Metal pays 2× the memory bandwidth for KV loads
compared to CUDA's `V_is_K_view` shortcut.

### W3 — Dead `nsg` autotuner

**Evidence:** `ggml-metal-ops.cpp:2819-2834` — the autotuner code is
commented out. The heuristic is `nsg = ne00 >= 512 ? 8 : 4`
(`:2835`).

**Impact:** Suboptimal for head sizes near 256 or 384. Also ignores
`ne11` (KV cache length), which affects how many KV iterations each
simdgroup processes. See Finding ARTX17-F12.

### W4 — No fused `ROPE+VIEW+SET_ROWS`

**Evidence:** `ggml_metal_op_rope` (`:3547`) and `set_rows`
(`:9841`) are completely separate dispatches. No fusion pattern in
the plan-time optimizer (ARTX15-F11) connects them.

**Impact:** Each decode step pays two extra kernel launches (RoPE +
set_rows for K, RoPE + set_rows for V = 4 launches). At ~5 µs launch
overhead per kernel, this is ~20 µs of pure overhead per token at
decode.

### W5 — Quantized-K / quantized-V branch is "not optimized yet"

**Evidence:** Comments at `ggml-metal.metal:6653` and `:6844`:
"TODO: this is the quantized K cache branch - not optimized yet".
The quantized-K path uses shmem staging + `simdgroup_barrier` per
inner iteration; the F16/F32 K path reads directly from device memory
with no shmem staging.

**Impact:** Quantized KV cache is slower than F16 KV cache on Metal,
which defeats part of the purpose of quantizing the KV cache. CUDA's
FA-vec kernel has hand-tuned per-dtype vecdot kernels using `dp4a`
(ARTX11-F02); Metal has no such specialisation.

### W6 — No `diag_mask_inf` kernel

**Evidence:** No `diag_mask` or `mask_inf` symbol in the .metal file.
The only `kernel_diag_f32` is `GGML_OP_DIAG` (diagonal-matrix
constructor), a different op.

**Impact:** Metal relies entirely on the model code to materialise
the mask. For the legacy (non-FA) attention path, this means the
mask must be constructed on the CPU or via a separate ADD op. CUDA's
`diagmask.cu` provides a one-shot causal-mask kernel that is faster
than the general ADD path.

### W7 — No BF16 RoPE

**Evidence:** Only `f32` and `f16` RoPE instantiations exist
(`ggml-metal.metal:4855-4865`). No `bfloat` template instantiation.

**Impact:** Models with BF16 activations (e.g. some Gemma variants)
must use F16 or F32 RoPE, paying a cast. Small but non-zero.

### W8 — `nwg = 32` hardcoded for vec path

**Evidence:** `ggml-metal-ops.cpp:2970` `nwg = 32;`. No autotuning
based on KV cache length or GPU SM count.

**Impact:** For very short KV caches (e.g. `ne11 = 64`), 32
workgroups each process only 2 KV items — the overhead of the
`flash_attn_ext_vec_reduce` post-pass may exceed the savings from
parallelism. For very long KV caches (e.g. `ne11 = 32768`), 32
workgroups may be too few to saturate the GPU.

### W9 — No graph-level FA fusion

**Evidence:** `ggml_metal_graph_optimize` (ARTX15-F11) does not
include any FA-related pattern. The closest is the `set_rows`
quantize-on-write, which is fusion of the KV-cache-append but not of
attention itself.

**Impact:** Each transformer layer pays separate kernel launches for
the residual add and the post-attention RMS_NORM. With ~80 layers
(Llama-3-70B) and ~5 µs launch overhead per kernel, this is ~800 µs
of pure overhead per token at decode.

### W10 — Per-head `pow` for ALiBi slope

**Evidence:** `ggml-metal.metal:6488, 7329` `slope = pow(base,
exph)`. Each threadblock computes the slope via a transcendental,
even though the slope depends only on `iq2` (head index).

**Impact:** `pow` is ~10-20 cycles on Apple Silicon. For a 32-head
model with 32 threadgroups per head, that's 32 redundant `pow` calls
per layer. A per-head slope lookup table (precomputed on the host)
would eliminate this. Small but non-zero.

### W11 — Template instantiation count is enormous

**Evidence:** 240 FA-tile + 80 FA-vec = 320 template instantiations.
Each is a separate compiled pipeline. The Metal library compile time
can be minutes; the lazy cache mitigates runtime cost but not
build-time cost.

**Impact:** Slow first-build. Iterating on the FA kernel source
requires recompiling all 320 instantiations, which is painful for
development. A smaller set of "canonical" shapes with a runtime
fallback would be more maintainable.

### W12 — `MAXHALF` undefined in source

**Evidence:** `MAXHALF` is used at `ggml-metal.metal:6235, 6275,
6276, 6290, 6519, 6556, 6576, 6584, 7354, 7371` but is not defined
in the .metal file, in `ggml-metal-impl.h`, or in any other audited
source file. It must come from `<metal_stdlib>` (Apple's metal
standard library), but its definition is not documented in the
audited source.

**Impact:** A reader cannot determine the value of `MAXHALF` from the
source alone. It is presumably `65504.0h` (the maximum half value),
but this should be explicit.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glmetal`       | **ADOPT** | Two-kernel FA dispatch (tile vs vec) | Right granularity for Apple Silicon; `simdgroup_matrix` is the tensor-core analog. |
| `glmetal`       | **ADOPT** | Mask-prune `flash_attn_ext_blk` three-state classifier | The all-zero state is a Metal-specific optimisation; clean and effective. |
| `glmetal`       | **ADOPT** | KV-tail `flash_attn_ext_pad` pre-pass | Clean separation; pad buffer carved from dst tail. |
| `glmetal`       | **ADOPT** | Always-on YaRN in RoPE | Single `rope_yarn` helper shared across all 4 variants; fixes propagate for free. |
| `glmetal`       | **ADOPT** | Attention sinks baked into FA | `src[4]` per-head sink fused into online softmax; no separate cap pass. |
| `glmetal`       | **ADOPT** | `set_rows_q32` quantize-on-write | 6 quant funcs × 2 idx dtypes; clean fusion of quantize + scatter. |
| `glmetal`       | **ADOPT** | Extra-data tail in `dst->data` | Pad/blk/tmp carved from dst tail; no scratch pool. |
| `glmetal`       | **ADAPT** | Per-(dtype,DK,DV) template explosion | Keep the pattern but reduce the shape set; 320 instantiations is too many for fast builds. |
| `glmetal`       | **ADAPT** | `nsg` heuristic | Re-enable the autotuner; pick largest `nsg` that fits `max_theadgroup_memory_size` and `ne11/ncpsg`. |
| `glmetal`       | **ADAPT** | `nwg = 32` for vec path | Autotune based on KV length and GPU SM count. |
| `glmetal`       | **ADAPT** | `use_vec` heuristic | `ne01 < 20` is hand-tuned; consider a shape-aware policy. |
| `glmetal`       | **REJECT**| Absence of FP8 KV path | Add a software dequant-to-F16 path for FP8 KV. |
| `glmetal`       | **REJECT**| Absence of K-quant / IQ-quant KV path | Add Q4_K, Q6_K, IQ4_NL FA-vec specializations. |
| `glmetal`       | **REJECT**| Absence of `V_is_K_view` MLA shortcut | Alias V to K's pointer for MLA models. |
| `glmetal`       | **REJECT**| Dead `nsg` autotuner | Re-enable or remove the commented-out code. |
| `glmetal`       | **REJECT**| No fused `ROPE+VIEW+SET_ROWS` | Add the fusion; CUDA has it. |
| `glmetal`       | **MONITOR**| `precise::tanh` for logit softcap | Watch for precision issues; `precise::tanh` is slower than `tanh` but more accurate. |
| `glmetal`       | **DEFER** | BF16 RoPE | Only relevant if GwenLand uses BF16 activations for RoPE input. |
| `GATE`          | **ADOPT** | Mask-prune three-state classifier | Generalise to a `mask_block_state` op for any backend. |
| `GATE`          | **ADAPT** | `FLASH_ATTN_EXT` op signature | Add `sliding_window` parameter; drop dense-mask requirement. |
| `GATE`          | **ADOPT** | Fused `ROPE+SET_ROWS` pattern | Extend to `ROPE+FA` if/when a persistent kernel is added. |

---

## 16. Recommendations

### R1 — ADOPT two-kernel FA dispatch taxonomy
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glmetal` should define `gl_fa_kernel_kind ∈ {TILE, VEC}`
and a `gl_fa_use_vec(op)` function. The `ne01 < 20` heuristic is a
reasonable starting point; tune per-device. The tile kernel uses
`simdgroup_multiply_accumulate`; the vec kernel uses
`simd_shuffle_down`. No third "MMA" kernel is needed on Apple Silicon.

### R2 — ADOPT FlashAttention-2 online-softmax contract
**Priority:** Critical
**Difficulty:** S
**Dependencies:** R1
Implement `(M, S, O)` per-thread / per-simdgroup accumulators with
`M = -FLT_MAX/2` initialisation (Metal's approach) or the `3·log(2)`
shift (CUDA's approach). Both work; Metal's is simpler. Document the
choice and the `S == 0 ? 0 : 1/S` guard.

### R3 — ADOPT mask-prune three-state pre-pass
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Replicate `kernel_flash_attn_ext_blk` as `glmetal_mask_blk_classify`.
Three states: skip, active, all-zero. The all-zero state is the
Metal-specific optimisation that avoids the mask-add for zero blocks.
Run before the main FA kernel when `has_mask`.

### R4 — REJECT absence of FP8 / K-quant / IQ-quant KV; ADOPT paths
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
Add FP8 (E4M3, E5M2) FA-vec specializations via software dequant-to-F16.
Add Q4_K, Q6_K, IQ4_NL FA-vec specializations via per-block dequant
(similar to the existing Q4_0 path). For FA-tile, add an F8→F16 / Q4_K→F16
conversion pre-pass (similar to CUDA's `to_fp16_cuda` in `launch_fattn`).

### R5 — REJECT absence of `V_is_K_view` MLA shortcut; ADOPT
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
When `DK != DV` and `V` aliases `K` (same buffer, same offset), set
`v_h2 = k_h2` and reverse the K-load iteration so the same shmem tile
can be re-used as V after a transpose. Saves a smem buffer and a load
path for MLA models.

### R6 — ADOPT fused `ROPE+SET_ROWS`; extend to `ROPE+FA`
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Add a fused `ROPE+SET_ROWS` kernel for the KV-cache-append path
(CUDA has this; Metal does not). Saves one kernel launch per RoPE'd
KV row at decode. Extending to `ROPE+FA` is harder on Metal (no
persistent kernel), so DEFER that.

### R7 — ADAPT `nsg` autotuner
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
Re-enable the commented-out code at `ggml-metal-ops.cpp:2819-2834`.
Compute `nsgmax` from `max_theadgroup_memory_size` and the
`FATTN_SMEM(nsg)` formula. Pick the largest `nsg ≤ nsgmax` that also
satisfies `nsg ≤ ne11/ncpsg` (don't allocate more simdgroups than
there are KV blocks).

### R8 — ADOPT `set_rows_q32` quantize-on-write pattern
**Priority:** Medium
**Difficulty:** S
**Dependencies:** none
Replicate the `set_rows_q32` pattern: F32 src0, int32/int64 row index
src1, quantized dst. 6 quant funcs × 2 idx dtypes = 12
instantiations. Each thread processes one block. Fuses quantize +
scatter.

### R9 — ADAPT per-(dtype,DK,DV) template explosion
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Keep the template-explosion pattern but reduce the shape set. Support
only the canonical DK ∈ {64, 128, 256, 512} and DV ∈ {64, 128, 256,
512} plus the MLA shapes (192,128), (320,256), (576,512). Fall back
to a runtime-looped kernel for unusual shapes. This cuts the
instantiation count from 320 to ~80.

### R10 — ADOPT attention-sink `src[4]` in FA op signature
**Priority:** Medium
**Difficulty:** XS
**Dependencies:** R1
Add `src[4]` as an optional per-head sink tensor to the
`FLASH_ATTN_EXT` op. Fuse it into the online softmax max/sum. This
matches Metal's design and avoids a separate softmax-cap pass.

### R11 — ADOPT `sliding_window` op parameter
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Add `int32_t sliding_window` to `FLASH_ATTN_EXT`'s `op_params`. In
the FA kernel, use it to compute `k_VKQ_max = min(k_VKQ_max, j +
sliding_window)` per Q row, avoiding the dense-mask materialisation.
Generalise the `flash_attn_ext_blk` pre-pass to compute this bound.

### R12 — DEFER BF16 RoPE
**Priority:** Low
**Difficulty:** XS
**Dependencies:** none
Only relevant if GwenLand uses BF16 activations for RoPE input. Add
`bfloat` template instantiations for the four RoPE variants if needed.

---

## 17. Findings

### Finding ARTX17-F01

```
Finding ID:           ARTX17-F01
Category:             BACKEND_DESIGN
Engine:               Metal
Component:            Flash-Attention top-level dispatch
Source File:          ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             ggml_metal_op_flash_attn_ext_use_vec / ggml_metal_op_flash_attn_ext
Lines:                2526-2534, 2650-3078
Summary:              A single heuristic selects one of two FA kernels
                      (TILE / VEC) at runtime, keyed on Q-token count
                      (ne01) and head-dim divisibility (ne00 % 32 == 0).
                      No autotuner; the threshold is hardcoded.
Observation:          ggml_metal_op_flash_attn_ext_use_vec returns true
                      when ne01 < 20 && ne00 % 32 == 0. The dispatch
                      then branches into the tile path (lines 2724-2890)
                      or the vec path (lines 2891-3074). Unlike CUDA's
                      three-way VEC/TILE/MMA_F16 split (ARTX11-F01), Metal
                      has only two paths because simdgroup_multiply_accumulate
                      IS Metal's tensor-core analog — there is no separate
                      "MMA" kernel to dispatch to. The vec kernel uses
                      simd_shuffle_down for intra-row reduction; the tile
                      kernel uses simdgroup_matrix 8x8 outer products.
Evidence:             ggml-metal-ops.cpp:2526-2534 (use_vec heuristic),
                      2724 (tile branch), 2891 (vec branch).
Architectural Impact: Clean two-way taxonomy. The ne01 < 20 threshold
                      is hand-tuned; may not generalise to future Apple
                      Silicon generations. Adding a new (DK,DV) shape
                      requires adding template instantiations in
                      ggml-metal.metal, not touching the dispatcher.
Correctness Impact:   None. The dispatcher only selects kernels.
Optimization Type:    None (this is a dispatch policy, not an optimization).
GwenLand Target:      glmetal
Recommendation:       ADOPT the taxonomy, ADAPT the heuristic to be
                      table-driven (per-device thresholds).
Priority:             High
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F02

```
Finding ID:           ARTX17-F02
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_flash_attn_ext (tile path)
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_flash_attn_ext_impl
Lines:                6353-6961
Summary:              FA-tile kernel uses simdgroup_multiply_accumulate
                      for QK^T and PV matmuls, with 4-K-tile + 4-Q-tile
                      (or 2+2 when DK%16!=0) outer products per inner
                      iteration. Threadgroup size 32*nsg (128 or 256).
Observation:          The kernel allocates Q (DK*Q), O (PV*Q), S-scratch
                      (SH*Q), K-scratch (4*16*KV per simdgroup), V-scratch
                      (overlapped with K), and Mask (C half2) in threadgroup
                      memory. The QK^T matmul (lines 6608-6645) loads K
                      from device (F16/F32) or from shmem (quantized) and
                      accumulates into simdgroup_float8x8 mqk. The PV
                      matmul (lines 6765-6909) has two paths: DV<=64 uses
                      1-vs-tile per iteration; DV>64 uses 2-vs-tiles + 4-
                      mv-tiles per iteration. The online softmax (lines
                      6717-6760) rescales O by exp(m_old - m_new) per
                      iteration.
Evidence:             ggml-metal.metal:6402-6416 (shmem layout), 6608-6645
                      (QK^T), 6765-6909 (PV), 6717-6760 (online softmax).
Architectural Impact: This is the canonical Metal FA tile kernel. The
                      nsg ∈ {4,8} choice multiplies the per-TG work by
                      nsg but also multiplies the shmem by nsg (for K/V
                      scratch). The hardcoded nsg = ne00 >= 512 ? 8 : 4
                      heuristic (ARTX17-F12) underuses the GPU for
                      mid-size heads.
Correctness Impact:   Standard FA-2 algorithm. Deterministic per
                      (nsg, DK, DV) for a fixed input.
Optimization Type:    SIMD / tiling / blocking.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Replicate the 4-ma + 4-mq (or 2+2) outer-
                      product pattern. Document the DV<=64 vs DV>64 split.
Priority:             Critical
Difficulty:           L
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F03

```
Finding ID:           ARTX17-F03
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_flash_attn_ext_vec (vec path) + vec_reduce
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_flash_attn_ext_vec, kernel_flash_attn_ext_vec_reduce
Lines:                7199-7827
Summary:              Decode-time FA kernel uses 32 workgroups per output
                      row, each computing a partial (S,M,O) triplet, then
                      a separate reduce kernel combines them via online-
                      softmax rescaling. Intra-row reduction via
                      simd_shuffle_down (5-step tree).
Observation:          The vec kernel parallelises the KV-iteration space
                      across nwg=32 workgroups. Each workgroup processes
                      a disjoint slice of the KV cache (ic0 = iwg*NSG +
                      sgitg; ic0 += NWG*NSG). The partial results are
                      written to bid_tmp as float4 O[DV4*NWG] + float
                      S,M[2*NWG] per row. The reduce kernel (lines 7787-
                      7827) reads all 32 partials, computes m = simd_max
                      (M_i), ms_i = exp(M_i - m), S = Σ ms_i * S_i, dst =
                      (Σ ms_i * partial_i) / S. The intra-row KQ reduction
                      uses simd_shuffle_down at 16/8/4/2/1 strides (lines
                      7518-7550), gated by NE comparisons so only the
                      needed shuffles execute.
Evidence:             ggml-metal.metal:7234-7240 (NWG/NSG macros), 7334
                      (KV loop with iwg stride), 7518-7550 (shuffle tree),
                      7620-7642 (partial result write), 7787-7827 (reduce
                      kernel); ggml-metal-ops.cpp:2970 (nwg=32 hardcoded),
                      3056-3072 (reduce dispatch).
Architectural Impact: The nwg=32 fixed value over-parallelises short KV
                      caches (each WG does almost nothing) and under-
                      parallelises long ones. CUDA autotunes parallel_blocks
                      from cudaOccupancyMaxActiveBlocksPerMultiprocessor;
                      Metal does not.
Correctness Impact:   The reduce kernel reassociates across workgroups,
                      producing ULP-level non-determinism across runs.
                      For nwg=1 (short KV cache) the result is deterministic.
Optimization Type:    Parallel reduction + online-softmax combine.
GwenLand Target:      glmetal
Recommendation:       ADOPT the pattern, ADAPT nwg to be autotuned from
                      KV length and GPU SM count.
Priority:             High
Difficulty:           M
Dependencies:         ARTX17-F01
Confidence:           High
```

### Finding ARTX17-F04

```
Finding ID:           ARTX17-F04
Category:             MISSING_FEATURE
Engine:               Metal
Component:            FA KV dtype support
Source File:          ggml/src/ggml-metal/ggml-metal-device.m, ggml/src/ggml-metal/ggml-metal.metal
Function:             ggml_metal_device_supports_op (GGML_OP_FLASH_ATTN_EXT)
Lines:                device.m:1250-1266; metal.metal:7050-7779 (instantiations)
Summary:              FA supports only 8 KV dtypes (F32, F16, BF16, Q4_0,
                      Q4_1, Q5_0, Q5_1, Q8_0). No K-quants (Q4_K, Q5_K,
                      Q6_K, Q2_K, Q3_K), no IQ-quants (IQ4_NL, IQ4_XS,
                      IQ2_*, IQ3_*), no MXFP4, no FP8 (E4M3, E5M2). The
                      audit prompt's mention of kernel_flash_attn_ext_q4_K,
                      _q6_K, _iq4_nl is incorrect — these do not exist.
Observation:          The supports_op whitelist at device.m:1250-1266
                      enumerates the 8 supported dtypes. The template
                      instantiations at metal.metal:7050-7779 cover exactly
                      these 8 dtypes × 15 (DK,DV) shape pairs × 2 kernels
                      (tile+vec) = 240+80 = 320 entries. No Q4_K, Q6_K, or
                      IQ4_NL entry exists. The model code must use F16 KV
                      on Metal even when Q4_K KV would be acceptable on
                      CPU/CUDA.
Evidence:             device.m:1250-1266 (whitelist), metal.metal:7050-7779
                      (instantiations — grep for "kernel_flash_attn_ext_"
                      yields 8 unique dtype prefixes: f32, f16, bf16, q4_0,
                      q4_1, q5_0, q5_1, q8_0).
Architectural Impact: Models that store KV cache in Q4_K (a common memory-
                      saving choice) must use F16 KV on Metal, paying 2×
                      memory bandwidth. FP8 KV models (vLLM exports) are
                      unsupported.
Correctness Impact:   None. The supports_op check fails loudly; no silent
                      fallback.
Optimization Type:    None (missing optimisation).
GwenLand Target:      glmetal
Recommendation:       REJECT this gap. Add Q4_K, Q6_K, IQ4_NL FA-vec
                      specializations via per-block dequant (similar to
                      existing Q4_0 path). Add FP8 FA-vec via software
                      dequant-to-F16.
Priority:             High
Difficulty:           L
Dependencies:         ARTX17-F01
Confidence:           High
```

### Finding ARTX17-F05

```
Finding ID:           ARTX17-F05
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_flash_attn_ext_blk (mask-prune pre-pass)
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             kernel_flash_attn_ext_blk
Lines:                metal.metal:6252-6305; ops.cpp:2775-2802
Summary:              A pre-pass kernel classifies each C×Q mask block
                      into one of three states (0=skip, 1=active, 2=all-
                      zero) stored as int8_t. The main FA kernel reads
                      the state and skips fully-masked blocks, and avoids
                      the mask-add for all-zero blocks.
Observation:          The pre-pass is launched only when has_mask is true.
                      Grid: (nblk0=ceil(ne30/ncpsg), nblk1=ceil(ne01/nqptg),
                      ne32*ne33), 32 threads per TG. Each thread reads one
                      half of the mask (mask_src[ii*NW]) and computes mmin
                      and mmax via simd_min / simd_max. The classification:
                      if mmax <= -MAXHALF → state 0 (skip); else if mmin==0
                      && mmax==0 → state 2 (all-zero); else → state 1
                      (active). The main FA kernel (metal.metal:6540-6567)
                      reads blk[ic0] at the start of each KV iteration
                      and continues past state-0 blocks, skips the mask-
                      add for state-2 blocks, and processes state-1 blocks
                      normally.
Evidence:             metal.metal:6252-6305 (kernel body), 6540-6567 (main
                      kernel reads blk[]), 6731 (blk_cur != 2 check);
                      ops.cpp:2775-2802 (dispatch).
Architectural Impact: For sliding-window attention with a small window
                      over a long cache, this prunes the KV iteration
                      space by ~seq_len/window_size. The all-zero state
                      is a Metal-specific optimisation not present in
                      CUDA's flash_attn_mask_to_KV_max (which is binary
                      skip/active only).
Correctness Impact:   None. State 2 is correct only because the mask is
                      added to KQ (not multiplied); adding zero is a
                      no-op.
Optimization Type:    Kernel fusion (mask scan + main FA) via a two-
                      kernel pipeline.
GwenLand Target:      glmetal, GATE
Recommendation:       ADOPT. Generalise the three-state classifier to a
                      GATE-level mask_block_state op usable by any backend.
Priority:             High
Difficulty:           S
Dependencies:         ARTX17-F01
Confidence:           High
```

### Finding ARTX17-F06

```
Finding ID:           ARTX17-F06
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_flash_attn_ext_pad (KV-tail padding pre-pass)
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             kernel_flash_attn_ext_pad
Lines:                metal.metal:6180-6243; ops.cpp:2737-2773, 2905-2941
Summary:              When the KV cache length (ne11) is not divisible by
                      ncpsg (64 for tile, 32 for vec), a separate kernel
                      copies the trailing partial chunk of K, V, and mask
                      into a padded buffer, zero-filling K/V past the end
                      and writing -MAXHALF into the mask tail. The main FA
                      kernel then runs on the pad buffer for that iteration.
Observation:          The pad pre-pass is launched when has_kvpad = ne11 %
                      ncpsg != 0. Grid: (ncpsg, max(ne12,ne32), max(ne13,
                      ne33)), 32 threads per TG. Each thread copies one
                      byte of K, V, or mask from the source to the pad
                      buffer. For positions past icp (= ne11 % ncpsg), K
                      and V are zero-filled and mask is set to -MAXHALF.
                      The main FA kernel (metal.metal:6500-6535) detects
                      the last partial chunk (ic + C > ne11) and redirects
                      k, v, mask pointers to the pad buffer, resetting
                      ic=0 so the iteration runs over the padded data.
Evidence:             metal.metal:6180-6243 (kernel body), 6500-6535 (main
                      kernel pad redirect); ops.cpp:2737-2773 (tile-path
                      dispatch), 2905-2941 (vec-path dispatch).
Architectural Impact: Splitting the KV-tail-padding into a separate kernel
                      keeps the main FA kernel simple (no bounds-checking
                      on the KV dimension inside the hot loop). The pad
                      buffer is carved from dst->data tail, so no scratch
                      allocation.
Correctness Impact:   None. The pad buffer is correctly initialised;
                      -MAXHALF mask values produce exp(-inf) = 0 in the
                      softmax, contributing nothing.
Optimization Type:    Kernel fusion (KV-tail pad + main FA) via a two-
                      kernel pipeline.
GwenLand Target:      glmetal
Recommendation:       ADOPT. Clean separation; avoids bounds-checking in
                      the hot loop.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX17-F01
Confidence:           High
```

### Finding ARTX17-F07

```
Finding ID:           ARTX17-F07
Category:             MISSING_FEATURE
Engine:               Metal
Component:            Causal / diag-mask path
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             (no diag_mask_inf kernel exists)
Lines:                N/A (absence)
Summary:              Metal has no diag_mask_inf kernel. The mask is
                      precomputed externally (in the model code) and
                      passed as op->src[3]. The kernel_diag_f32 in the
                      .metal file is GGML_OP_DIAG (diagonal-matrix
                      constructor), a different op. CUDA has a standalone
                      diagmask.cu; Metal does not.
Observation:          A grep for "diag_mask|mask_inf|DIAG_MASK" in the
                      ggml-metal directory yields no matches. The
                      kernel_diag_f32 at metal.metal:9936-9954 implements
                      GGML_OP_DIAG: dst[i0] = i0 == i1 ? src0[i0] : 0.0f.
                      This is a diagonal-matrix constructor, not a causal-
                      mask kernel. The FA kernels consume the mask as a
                      precomputed F16 tensor via op->src[3].
Evidence:             grep "diag_mask|mask_inf|DIAG_MASK" → no matches;
                      metal.metal:9936-9954 (kernel_diag_f32 is OP_DIAG);
                      ops.cpp:446 (GGML_OP_FLASH_ATTN_EXT dispatch — no
                      diag_mask case in the switch).
Architectural Impact: Metal relies entirely on the model code to
                      materialise the mask. For the legacy (non-FA)
                      attention path, this means the mask must be
                      constructed on the CPU or via a separate ADD op.
Correctness Impact:   None. The mask is correct by construction; the
                      FA kernel applies it correctly.
Optimization Type:    None (missing optimisation).
GwenLand Target:      glmetal
Recommendation:       ADOPT a diag_mask_inf kernel for the legacy path
                      (matches CUDA's diagmask.cu). For the FA path, the
                      precomputed-mask design is fine.
Priority:             Low
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F08

```
Finding ID:           ARTX17-F08
Category:             GPU_KERNEL
Engine:               Metal
Component:            RoPE kernel family
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-device.cpp
Function:             kernel_rope_norm, kernel_rope_neox, kernel_rope_multi, kernel_rope_vision
Lines:                metal.metal:4550-4866; device.cpp:1719-1761
Summary:              Four RoPE variants (norm/neox/multi/vision) with
                      always-on YaRN length extrapolation, each templated
                      on <f32, f16>. No BF16. No separate rope_yarn_*
                      kernel — YaRN is integrated into all four variants
                      via the rope_yarn helper. The vision variant has a
                      2D mRoPE layout (sect_0 + sect_1 only) for vision
                      transformers.
Observation:          The four variants differ only in the (x0, x1)
                      stride and the theta_base selection:
                      - norm: interleaved, x1 = src[1], theta = pos[i2]
                      - neox: split-half, x1 = src[n_dims/2], theta = pos[i2]
                      - multi: neox layout, theta from one of 4 pos IDs
                        based on sector (t/h/w/e)
                      - vision: 2D mRoPE, x1 = src[n_dims], theta from
                        pos[i2] or pos[i2+ne02] based on sector
                      The rope_yarn helper (metal.metal:4560-4578) applies
                      ramp mixing + magnitude scaling when ext_factor !=
                      0.0f, and is a no-op otherwise. The is_back flag
                      (FC_rope_is_back) negates sin_theta for the backward
                      pass.
Evidence:             metal.metal:4594-4645 (norm), 4647-4698 (neox),
                      4700-4781 (multi), 4783-4848 (vision), 4560-4578
                      (rope_yarn), 4855-4865 (instantiations);
                      device.cpp:1727-1744 (mode-based pipeline selection).
Architectural Impact: Always-on YaRN means length-extrapolation fixes
                      propagate to all variants for free. The vision
                      variant is Metal-specific (no CUDA equivalent in
                      ARTX11). The absence of BF16 RoPE is a small gap.
Correctness Impact:   Standard RoPE math. YaRN magnitude scaling is F32
                      precision (~24 bits).
Optimization Type:    None (per-thread scalar math; no SIMD).
GwenLand Target:      glmetal
Recommendation:       ADOPT the four-variant taxonomy with always-on
                      YaRN. Add BF16 instantiations if needed.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F09

```
Finding ID:           ARTX17-F09
Category:             GPU_KERNEL
Engine:               Metal
Component:            kernel_soft_max (standalone softmax)
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             kernel_soft_max, kernel_soft_max_4
Lines:                metal.metal:1950-2161; ops.cpp:1300-1388
Summary:              Standalone F32 softmax with optional F16 mask,
                      ALiBi bias, and attention-sink cap (src[2] per-head
                      max). Two-pass (max, sum-exp). Optional _4 variant
                      uses float4 loads when ne00 % 4 == 0. Cross-simdgroup
                      reduction via 32-float shmem when tptg.x > 32.
Observation:          The kernel reads scale, max_bias, m0, m1, n_head_log2
                      from the args struct. ALiBi slope is computed per-
                      threadblock via pow(base, exph) where base = h <
                      n_head_log2 ? m0 : m1 and exph = h < n_head_log2 ?
                      h+1 : 2*(h-n_head_log2)+1 (metal.metal:1978-1985).
                      The attention-sink src[2] (psrc2) is integrated as:
                      lmax initialised to psrc2[i02] (line 1988), then
                      sum += exp(psrc2[i02] - max_val) after the row scan
                      (line 2045). The _4 variant uses float4 loads and
                      dot4 reduction for 4x fewer iterations.
Evidence:             metal.metal:1950-2053 (kernel_soft_max), 2056-2161
                      (kernel_soft_max_4), 1978-1985 (ALiBi), 1988 (sink
                      init), 2045 (sink add); ops.cpp:1300-1388 (dispatch).
Architectural Impact: The standalone softmax is for the legacy (non-FA)
                      attention path and for sampling. The FA kernels
                      have their own online-softmax baked in. The
                      attention-sink integration is duplicated: once in
                      kernel_soft_max, once in the FA kernels.
Correctness Impact:   Standard softmax. The sink integration is correct:
                      the sink value is treated as an extra "always-present"
                      logit.
Optimization Type:    Two-pass reduction + optional float4 vectorization.
GwenLand Target:      glmetal
Recommendation:       ADOPT. The attention-sink integration should be
                      documented as a shared pattern between soft_max and
                      FA.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F10

```
Finding ID:           ARTX17-F10
Category:             CORRECTNESS_SHORTCUT
Engine:               Metal
Component:            FA online-softmax numerical stability
Source File:          ggml/src/ggml-metal/ggml-metal.metal
Function:             kernel_flash_attn_ext_impl, kernel_flash_attn_ext_vec
Lines:                6474-6477, 6919-6927, 7311, 7561-7570, 7628, 7810
Summary:              Metal's FA kernels use -FLT_MAX/2 as the KQ_max
                      sentinel and a S==0 ? 0 : 1/S final-division guard.
                      No FATTN_KQ_MAX_OFFSET shift (as in CUDA) and no
                      SOFTMAX_FTZ_THRESHOLD bit-hack. The approach is
                      simpler but may produce different ULP results than
                      CUDA on the same inputs.
Observation:          The M accumulator is initialised to -FLT_MAX/2
                      (metal.metal:6477, 7311) so that exp(-FLT_MAX/2 -
                      any_finite_logit) underflows to 0 harmlessly. The
                      sink value is also initialised to -FLT_MAX/2 when
                      tiisg != 0 (line 6919, 7563) to avoid contributing
                      to the simd_max from non-lane-0 threads. The final
                      division is guarded by S == 0.0 ? 0.0 : 1.0/S (lines
                      6944, 7628, 7810) so the all-masked case produces 0
                      output instead of NaN. There is no equivalent of
                      CUDA's FATTN_KQ_MAX_OFFSET = 3*log(2) shift (which
                      lifts the VKQ dynamic range by 8x to avoid F16
                      overflow) — Metal's VKQ accumulator is always F32,
                      so F16 overflow is not a concern.
Evidence:             metal.metal:6477 (M init), 6919 (sink init), 6944
                      (S==0 guard), 7311 (vec M init), 7628 (vec S==0
                      guard), 7810 (reduce S==0 guard).
Architectural Impact: The -FLT_MAX/2 approach is simpler than CUDA's
                      shift+FTZ-threshold approach. It works because F32
                      exp underflows gracefully. The downside is that
                      Metal's FA may produce different ULP results than
                      CUDA on the same inputs — not a correctness issue,
                      but a cross-backend reproducibility issue.
Correctness Impact:   The result is mathematically equivalent to the
                      textbook softmax(KQ)·V. The S==0 guard prevents NaN
                      in the all-masked case.
Optimization Type:    None (numerical stability workaround).
GwenLand Target:      glmetal
Recommendation:       ADOPT. Document the -FLT_MAX/2 sentinel and the
                      S==0 guard. Note the cross-backend ULP difference
                      vs CUDA.
Priority:             Medium
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F11

```
Finding ID:           ARTX17-F11
Category:             GPU_KERNEL
Engine:               Metal
Component:            KV-cache-append (set_rows) + SSM kernels
Source File:          ggml/src/ggml-metal/ggml-metal.metal, ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             kernel_set_rows_f, kernel_set_rows_q32, kernel_ssm_conv_f32_f32[_4][_batched], kernel_ssm_scan_f32
Lines:                metal.metal:9841-9935 (set_rows), 2172-2326 (ssm_conv), 2330-2520 (ssm_scan); ops.cpp:1390-1540 (ssm dispatch)
Summary:              KV-cache append uses kernel_set_rows_f (F32/F16/BF16
                      dst) or kernel_set_rows_q32 (quantized dst, 6 quant
                      funcs). Mamba SSM uses 4 ssm_conv variants (scalar,
                      float4, batched, batched_4) and one ssm_scan kernel.
                      No fused ROPE+SET_ROWS (unlike CUDA).
Observation:          set_rows_q32 takes F32 src0, int32/int64 row-index
                      src1, and writes quantized dst via the quantize_func
                      template parameter. Each thread processes one
                      32-element block. The 12 instantiations cover Q8_0,
                      Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL × {int32, int64}
                      index types (metal.metal:9921-9934). ssm_conv has 4
                      variants: scalar (1 thread/token), float4 (vectorized),
                      batched (BATCH_SIZE threads/token, prefill), batched_4
                      (vectorized + batched). The batch size is picked from
                      a power-of-2 ladder {2, 8, 16, 32, 64, 128, 256}
                      based on ne1 (ops.cpp:1427-1434). ssm_scan is a single
                      kernel with shmem-staged partial sums.
Evidence:             metal.metal:9841-9870 (set_rows_q32), 9872-9901
                      (set_rows_f), 9921-9934 (instantiations), 2172-2201
                      (ssm_conv scalar), 2203-2232 (ssm_conv_4), 2238-2281
                      (ssm_conv_batched), 2330+ (ssm_scan); ops.cpp:1427-
                      1434 (batch size ladder).
Architectural Impact: set_rows is the primary KV-cache-append mechanism.
                      The quantize-on-write fusion saves a separate F32→
                      quant conversion dispatch. The SSM kernels cover
                      Mamba-1/2 but are F32-only (no Tensor Core use);
                      ARTX11-W11 notes the same gap on CUDA.
Correctness Impact:   None. Standard scatter + quantize.
Optimization Type:    Kernel fusion (quantize + scatter) for set_rows;
                      vectorization + batched dispatch for ssm_conv.
GwenLand Target:      glmetal
Recommendation:       ADOPT set_rows_q32 quantize-on-write. ADOPT the ssm
                      conv batched dispatch pattern. DEFER SSM Tensor
                      Core path until GwenLand needs Mamba performance.
Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX17-F12

```
Finding ID:           ARTX17-F12
Category:             SIMD_STRATEGY
Engine:               Metal
Component:            FA-tile nsg heuristic
Source File:          ggml/src/ggml-metal/ggml-metal-ops.cpp
Function:             ggml_metal_op_flash_attn_ext (tile path)
Lines:                2817-2837
Summary:              The FA-tile nsg (simdgroups per threadgroup) is
                      hardcoded as nsg = ne00 >= 512 ? 8 : 4. A more
                      sophisticated autotuner that picks the largest nsg
                      fitting in max_theadgroup_memory_size is commented
                      out. The heuristic ignores KV cache length (ne11)
                      and GPU SM count.
Observation:          The commented-out code (lines 2819-2834) computes
                      nsgmax by doubling from 2 until FATTN_SMEM(nsgmax)
                      exceeds max_theadgroup_memory_size, then halving.
                      The actual code (line 2835) is just `int32_t nsg =
                      ne00 >= 512 ? 8 : 4`. This means:
                      - For DK=128 (typical Llama): nsg=4 → 128 threads/TG
                      - For DK=256 (Llama-3): nsg=4 → 128 threads/TG
                      - For DK=512 (Deepseek): nsg=8 → 256 threads/TG
                      - For DK=576 (Deepseek MLA): nsg=8 → 256 threads/TG
                      The heuristic does not consider ne11 (KV cache
                      length), which affects how many KV iterations each
                      simdgroup processes. For ne11=64 and nsg=8, each
                      simdgroup processes only 1 KV iteration — the
                      parallelism overhead may exceed the compute.
Evidence:             ops.cpp:2817-2837 (FATTN_SMEM macro + commented
                      autotuner + hardcoded heuristic).
Architectural Impact: Suboptimal for mid-size heads (256, 384) and for
                      short KV caches. The autotuner would pick nsg=8 for
                      DK=256 if the shmem budget allows, potentially
                      doubling throughput.
Correctness Impact:   None. nsg only affects performance, not correctness.
Optimization Type:    None (suboptimal heuristic).
GwenLand Target:      glmetal, GATE
Recommendation:       ADAPT. Re-enable the autotuner. Pick the largest
                      nsg ≤ nsgmax that also satisfies nsg ≤ ne11/ncpsg.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX17-F01
Confidence:           High
```

---

## 18. Unknowns

* **U1.** Whether the `-FLT_MAX/2` sentinel produces bit-identical
  results to CUDA's `FATTN_KQ_MAX_OFFSET = 3·log(2)` shift on the
  same inputs. Both are mathematically equivalent but reassociate the
  sum differently. Requires cross-backend differential testing.
  Static analysis cannot resolve this.
* **U2.** Whether the `nsg = ne00 >= 512 ? 8 : 4` heuristic is
  optimal for Apple M5 / Apple M6 hardware. The heuristic was tuned
  on earlier Apple Silicon; M5 may have different simdgroup-matrix
  throughput characteristics. Requires runtime profiling on M5.
* **U3.** Whether the `nwg = 32` fixed value for the vec path is
  optimal for KV cache lengths from 64 to 32768. The reduce kernel
  overhead may dominate for short caches; the parallelism may be
  insufficient for long caches. Requires profiling across the KV-
  length range.
* **U4.** Whether the `ne01 < 20` use_vec threshold is optimal. The
  threshold was likely tuned for a specific model (Llama-3-8B?) and
  may not generalise. For models with very wide Q heads (e.g. GLM-
  4.7-Flash with 96 Q heads), the threshold may need to be lower.
  Requires per-model profiling.
* **U5.** The value of `MAXHALF` — it is used but not defined in the
  audited source. Presumably `65504.0h` (max half value) from
  `<metal_stdlib>`, but this is not documented. A GwenLand engineer
  should confirm by inspecting the metal_stdlib header or running a
  small test kernel.
* **U6.** Whether the `flash_attn_ext_blk` mask-prune pre-pass is
  net-positive for typical attention patterns. For dense causal masks
  (no sliding window), every block is state 1 (active), and the pre-
  pass is pure overhead. For sliding-window masks with a small window,
  the pre-pass prunes many blocks. The break-even point depends on
  the mask density and the GPU's pre-pass vs main-kernel cost ratio.
  Requires profiling with realistic mask patterns.
* **U7.** Whether the FA-vec kernel's quantized-K/V path (which uses
  shmem staging + `simdgroup_barrier` per inner iteration) is
  competitive with the F16 path. The comment at metal.metal:6653
  says "not optimized yet" — unclear how slow it is relative to F16.
  Requires profiling Q4_0 vs F16 KV on the same model.
* **U8.** Whether the SSM scan kernel's shmem-staged partial sums
  (sgptg * NW + 2*sgptg floats) fit within the 32 KiB shmem limit
  for the largest Mamba configurations (d_state=256, n_head=128).
  Requires shape-specific analysis.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext_use_vec`         | 2526-2534     |
| R02       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext` (dispatcher)    | 2650-3078     |
| R03       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext_extra_pad`       | 2536-2580     |
| R04       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext_extra_blk`       | 2582-2619     |
| R05       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_flash_attn_ext_extra_tmp`       | 2621-2648     |
| R06       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_soft_max`                       | 1300-1388     |
| R07       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_rope`                           | 3547-3641     |
| R08       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_ssm_conv`                       | 1390-1461     |
| R09       | `ggml/src/ggml-metal/ggml-metal-ops.cpp`            | `ggml_metal_op_ssm_scan`                       | 1463-1540     |
| R10       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext` (entry)                | 6991-7016     |
| R11       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_impl` (template)        | 6353-6961     |
| R12       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_vec`                    | 7199-7648     |
| R13       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_pad`                    | 6180-6243     |
| R14       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_blk`                    | 6252-6305     |
| R15       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_flash_attn_ext_vec_reduce`             | 7787-7827     |
| R16       | `ggml/src/ggml-metal/ggml-metal.metal`              | FA-tile template instantiations                | 7050-7178     |
| R17       | `ggml/src/ggml-metal/ggml-metal.metal`              | FA-vec template instantiations                 | 7669-7779     |
| R18       | `ggml/src/ggml-metal/ggml-metal.metal`              | `FA_TYPES` / `FA_TYPES_F32` / `FA_TYPES_BF` macros | 7021-7046, 7653-7667 |
| R19       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_rope_norm` / `_neox` / `_multi` / `_vision` | 4595-4848 |
| R20       | `ggml/src/ggml-metal/ggml-metal.metal`              | `rope_yarn` / `rope_yarn_corr_dims` helpers    | 4553-4592     |
| R21       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_soft_max` / `kernel_soft_max_4`        | 1950-2161     |
| R22       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_ssm_conv_f32_f32[_4][_batched]`        | 2172-2326     |
| R23       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_ssm_scan_f32`                          | 2330-2520     |
| R24       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_set_rows_f` / `kernel_set_rows_q32`    | 9841-9935     |
| R25       | `ggml/src/ggml-metal/ggml-metal.metal`              | `kernel_diag_f32` (OP_DIAG, not OP_DIAG_MASK)  | 9936-9954     |
| R26       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `FC_FLASH_ATTN_EXT_*` function-constant offsets | 91-95       |
| R27       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `OP_FLASH_ATTN_EXT_NQPSG/NCPSG/VEC_*` constants | 109-113     |
| R28       | `ggml/src/ggml-metal/ggml-metal-impl.h`             | `ggml_metal_kargs_flash_attn_ext[_vec|_pad|_blk|_vec_reduce|_rope|_soft_max|_ssm_conv|_ssm_scan|_set_rows|_diag]` | 304-1013 |
| R29       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_flash_attn_ext[_vec|_pad|_blk|_vec_reduce]` | 1309-1549 |
| R30       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_rope`         | 1719-1761     |
| R31       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_soft_max`     | 453-478       |
| R32       | `ggml/src/ggml-metal/ggml-metal-device.cpp`         | `ggml_metal_library_get_pipeline_ssm_conv[_batched]` | 480-538  |
| R33       | `ggml/src/ggml-metal/ggml-metal-device.m`           | `ggml_metal_device_supports_op` (FLASH_ATTN_EXT) | 1229-1267   |
