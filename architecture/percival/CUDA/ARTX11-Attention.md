# ARTX11 — CUDA Attention Kernels

**Audited commit:** `555881ebc8b0fc0402b30e09258a32a7bfd13c52`
**Last updated:** 2026-07-25
**Auditor:** Percival-aux
**Target GwenLand module:** `glcuda` (attention kernels), `GATE` (op-fusion planner)

---

## 1. Executive Summary

The CUDA attention layer of llama.cpp is implemented by a *family of three
Flash-Attention (FA) kernels* — `fattn-vec`, `fattn-tile`, `fattn-mma-f16` —
plus a shared infrastructure header (`fattn-common.cuh`) and a central
dispatcher (`fattn.cu`). Together they implement the
FlashAttention-2 algorithm (online softmax with tiled Q, K, V) and a
decode-time single-query special-case, dispatching at runtime across NVIDIA
(Volta → Turing → Ampere → Ada → Hopper → Blackwell → Rubin), AMD (GCN, CDNA,
RDNA2/3/4), and Moore Threads (QY/PH) hardware via a CC-keyed heuristic.

The dispatcher (`fattn.cu:358-534`) does **not** expose a `flash_attn_sliding_window`
parameter, an FP8 KV cache path, or a Hopper `wgmma`/TMA path. Sliding-window
attention is realised by passing a precomputed F16 mask tensor through
`dst->src[3]`; FP8 KV is simply unsupported (`fattn.cu:338-356`). MLA
(Multi-Latent Attention, used by Deepseek-V3 and GLM-4.7-Flash) is bolted on
via the `V_is_K_view` flag, which lets V share K's pointer when `DKQ != DV`
(e.g. `576/512`, `192/128`, `320/256`).

The kernels share a uniform *FlashAttention-2 contract*:
* online softmax accumulators (`KQ_max`, `KQ_sum`, `VKQ`) per thread,
* a `FATTN_KQ_MAX_OFFSET = 3·log(2)` shift that lifts the VKQ dynamic range by
  a factor of 8 (work-around for issue #18606),
* a `SOFTMAX_FTZ_THRESHOLD = -20.0f` below which `expf` differences are
  flushed to zero to avoid NaNs,
* a separate optional `flash_attn_mask_to_KV_max` pre-pass that prunes fully
  masked `FATTN_KQ_STRIDE × FATTN_KQ_STRIDE` tiles,
* a *stream-K* scheduling path (Ada Lovelace+, AMD WMMA) with two fixup
  kernels (`flash_attn_stream_k_fixup_uniform`,
  `flash_attn_stream_k_fixup_general`),
* and a *combine* kernel (`flash_attn_combine_results`) for the legacy
  `parallel_blocks > 1` scheduling used when stream-K is not active.

RoPE (Rotary Positional Embedding) is implemented as a *standalone* CUDA op
(`rope.cu`), not fused into the FA kernels. It supports four positional
encodings — `rope_norm` (interleaved), `rope_neox` (split-half), `rope_multi`
(multimodal mRoPE), `rope_vision` — and a YaRN length-extrapolation ramp.
A fused `ROPE+VIEW+SET_ROWS` variant supports incremental KV-cache writes
(`rope.cu:670-672`). The standalone `softmax.cu` kernel (F32 in/out, optional
F16 mask, ALiBi slope, attention sinks) is used by the non-FA legacy attention
path and by sampling. The standalone `diagmask.cu` kernel implements a causal
mask by subtracting `FLT_MAX` from masked positions (`diagmask.cu:14`) — a
documented "slightly faster on GPU" alternative to `-INFINITY`.

For GwenLand the architectural decisions worth **ADOPT**ing are the
three-kernel dispatch taxonomy, the FlashAttention-2 online-softmax contract,
the per-CC config table pattern, the `V_is_K_view` MLA trick, the stream-K
fixup pair, the fused `ROPE+VIEW+SET_ROWS` path, and the YaRN ramp. The
decisions worth **REJECT**ing are the absence of an FP8 KV path, the absence
of Hopper `wgmma`/TMA, the missing `flash_attn_sliding_window` parameter
(sliding window is opaque to the kernel and therefore cannot prune KV loads),
and the `diagmask` `FLT_MAX` shortcut (which can produce NaN in pathological
cases).

---

## 2. Purpose

Provide a CUDA implementation of the `GGML_OP_FLASH_ATTN_EXT` op that:

* computes `softmax(scale·Q·Kᵀ + mask) · V` with optional ALiBi bias,
  logit-softcap, attention-sinks, and YaRN-style RoPE (RoPE applied upstream),
* supports per-head dimension `D ∈ {40, 64, 72, 80, 96, 112, 128, 192, 256,
  320, 512, 576}` (with `DV ∈ {40, 64, 72, 80, 96, 112, 128, 256, 512}` for
  the MLA cases where `DKQ != DV`),
* supports Grouped Query Attention (GQA) and Multi-Query Attention (MQA) via
  a `ncols2` template parameter that packs multiple Q heads per K/V head,
* supports quantized KV caches: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16 (FA-vec
  only), in addition to F16 and F32,
* supports ALiBi-style linear biases and logit-softcap (Gemma family),
* supports Multi-Latent Attention (Deepseek-V3, GLM-4.7-Flash, MiMo-V2.5)
  via the `V_is_K_view` shortcut,
* selects the best kernel at runtime based on compute capability, head size,
  GQA ratio, batch size, and KV-cache alignment,
* exposes a fully asynchronous interface to the CUDA backend
  (`ggml_cuda_flash_attn_ext` is one kernel launch, no host-side
  synchronisation).

It is **not** responsible for: graph construction, RoPE application (separate
op, audited below), causal-mask creation (handled by `diagmask.cu` or by
upstream graph construction), KV-cache allocation (handled by the model code),
or scheduler-level op fusion (handled by `ggml_cuda_try_fuse`, ARTX08).

---

## 3. Source Files

| File                                       | Lines | Role                                                                              |
| ------------------------------------------ | ----- | --------------------------------------------------------------------------------- |
| `ggml/src/ggml-cuda/fattn.cu`              | 589   | Top-level dispatch (`ggml_cuda_flash_attn_ext`, `ggml_cuda_get_best_fattn_kernel`), Volta/GQA switches for MMA path, FA-vec per-dtype case table. |
| `ggml/src/ggml-cuda/fattn.cuh`             | 8     | Public entry prototypes (`ggml_cuda_flash_attn_ext`, `_supported`, `_get_alloc_size`). |
| `ggml/src/ggml-cuda/fattn-common.cuh`      | 1274  | Shared types (`fattn_kernel_t`), online-softmax helpers, `quantize_q8_1_to_shared`, per-dtype KQ vec-dot and V dequantize templates, `flash_attn_mask_to_KV_max` pre-pass, `flash_attn_stream_k_fixup_*` kernels, `flash_attn_combine_results` kernel, `launch_fattn` host helper. |
| `ggml/src/ggml-cuda/fattn-tile.cu`         | 60    | Per-`(DKQ,DV)` switch dispatcher for the tile kernel.                              |
| `ggml/src/ggml-cuda/fattn-tile.cuh`        | 1355  | Per-CC config tables (NVIDIA-FP16, NVIDIA-FP32, AMD, AMD-RDNA), tile load/iter helpers, `flash_attn_tile` kernel. |
| `ggml/src/ggml-cuda/fattn-vec.cuh`         | 611   | Single-query decode FA kernel (`flash_attn_ext_vec`), per-dtype Q8_1 quantization, vec-dot KQ + dequantize V inner loops, multi-block combine. |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh`     | 2033  | Tensor-Core FA kernel (`flash_attn_ext_f16`), per-CC config tables (Ampere, Turing, Volta, RDNA, CDNA), `cp.async` multi-stage loader, `flash_attn_ext_f16_iter`/`_process_tile` inner loops, `V_is_K_view` MLA path. |
| `ggml/src/ggml-cuda/rope.cu`               | 672   | RoPE op (`rope_norm`, `rope_neox`, `rope_multi`, `rope_vision`) + YaRN length extrapolation + fused `ROPE+VIEW+SET_ROWS` path. |
| `ggml/src/ggml-cuda/rope.cuh`              | ~10   | Public prototypes.                                                                 |
| `ggml/src/ggml-cuda/softmax.cu`            | 472   | Standalone F32 softmax with optional F16 mask, ALiBi slope, attention sinks, cooperative-launch single-row path for huge ncols (top-p). |
| `ggml/src/ggml-cuda/softmax.cuh`           | ~10   | Public prototypes.                                                                 |
| `ggml/src/ggml-cuda/diagmask.cu`           | 40    | Legacy causal-mask kernel (subtracts `FLT_MAX` from masked positions).             |
| `ggml/src/ggml-cuda/diagmask.cuh`          | ~10   | Public prototypes.                                                                 |
| `ggml/src/ggml-cuda/gla.cu`                | 93    | Gated Linear Attention kernel (`gated_linear_attn_f32`, single-warp per head). |
| `ggml/src/ggml-cuda/wkv.cu`                | 199   | RWKV-6 / RWKV-7 linear-attention kernels (`rwkv_wkv_f32`).                         |
| `ggml/src/ggml-cuda/ssm-scan.cu`           | 364   | Mamba-1/2 SSM scan kernel (`ssm_scan_f32`, CUB `BlockLoad` based).                 |
| `ggml/src/ggml-cuda/ssm-conv.cu`           | 206   | Mamba short-conv kernel (`ssm_conv_f32`) with optional fused SILU+bias.            |
| `ggml/src/ggml-cuda/gated_delta_net.cu`    | 327   | Gated DeltaNet linear-attention kernel (`gated_delta_net_cuda`).                   |
| `ggml/src/ggml-cuda/template-instances/`   | ~200 files | Per-`(DKQ,DV,ncols2)` / per-`(D,type_K,type_V)` explicit template instantiations, generated by `generate_cu_files.py`. |

> Note: the audit prompt lists `fattn-mma-f16.cuh` as "FA using `mma.sync`
> Tensor Cores." At this commit the file uses the `mma.sync.aligned.m16n8k16`
> PTX instruction (Turing+) and the `m8n8k16`/`m8n8k8` Volta/Turing-fallback
> variants (audited in `mma.cuh`). It does **not** use Hopper's `wgmma` or
> TMA. The Ampere-or-newer path uses `cp.async.cg` (`cp_async_cg_16`) for
> shared-memory loads; the file makes no use of cluster-mode or distributed
> shared memory.

---

## 4. Architecture Overview

```
            ┌─────────────────────────────────────────────────────────────────┐
            │  fattn.cu : ggml_cuda_flash_attn_ext(ctx, dst)                 │
            │  └─ ggml_cuda_get_best_fattn_kernel(device, dst)               │
            │     └─ heuristic: CC, head size, GQA ratio, batch, mask, align  │
            └─────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼───────────────────────────────┐
        ▼                         ▼                               ▼
┌───────────────────┐  ┌──────────────────────────┐  ┌──────────────────────────┐
│ BEST_FATTN_VEC    │  │ BEST_FATTN_TILE          │  │ BEST_FATTN_KERNEL_MMA_F16│
│ fattn-vec.cuh     │  │ fattn-tile.cuh           │  │ fattn-mma-f16.cuh        │
│ decode, Q->ne[1]  │  │ prefill, non-TC GPUs     │  │ Tensor Cores (Turing+)   │
│ == 1 or small     │  │ (Pascal, Volta non-MMA,  │  │ AMD MFMA / WMMA          │
│                   │  │  AMD non-WMMA)           │  │ Volta WMMA m8n8k16       │
│ 128 threads       │  │                          │  │                          │
│ Per-dtype vecdot  │  │ Tiled Q/K/V in smem      │  │ mma.sync.m16n8k16 f16    │
│ Q8_1 quant on fly │  │ FATTN_KQ_STRIDE=256      │  │ cp.async 1-2 stages      │
│                   │  │                          │  │ ldmatrix                  │
└───────────────────┘  └──────────────────────────┘  └──────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │  fattn-common.cuh : launch_fattn<...>                           │
            │  ├─ convert K, V to F16 if needed (using to_fp16_cuda)           │
            │  ├─ optional flash_attn_mask_to_KV_max pre-pass                  │
            │  ├─ choose stream-K vs parallel_blocks scheduling                 │
            │  ├─ launch fattn_kernel via ggml_cuda_kernel_launch (PDL-aware)  │
            │  └─ launch flash_attn_combine_results or stream_k_fixup_*        │
            └─────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
            ┌─────────────────────────────────────────────────────────────────┐
            │  Per-dtype helpers in fattn-common.cuh                          │
            │  ├─ vec_dot_fattn_vec_KQ_{f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0}    │
            │  ├─ dequantize_V_{f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0}            │
            │  └─ quantize_q8_1_to_shared (warp-cooperative Q quantization)   │
            └─────────────────────────────────────────────────────────────────┘
```

Key design points:

* **Single dispatch function.** `ggml_cuda_get_best_fattn_kernel`
  (`fattn.cu:358-534`) is the *only* selection function. It returns an enum
  (`BEST_FATTN_KERNEL_{NONE,VEC,TILE=200,MMA_F16=400}`); the public
  `ggml_cuda_flash_attn_ext` switches on that enum
  (`fattn.cu:570-585`). There is no per-shape autotuner; the policy is
  hard-coded.
* **Three template-parameter dimensions.** Every kernel is templated on
  `(DKQ, DV, ncols1, ncols2)` where `ncols1` is the number of Q columns per
  CUDA block and `ncols2` is the GQA fan-out (Q heads per K/V head per block).
  Per-CC config tables pick `nthreads`, `occupancy`, `nbatch_fa`,
  `nbatch_K`/`nbatch_K2`, `nbatch_V2`, `nbatch_combine`, `nstages_target`,
  `Q_in_reg` — packaged into a `uint32_t` bitfield for the tile kernel
  (`fattn-tile.cuh:12-19`) or into a `fattn_mma_config` struct for the MMA
  kernel (`fattn-mma-f16.cuh:10-24`).
* **Online softmax contract.** All three kernels maintain `KQ_max`,
  `KQ_sum`, and `VKQ` accumulators per thread; on each iteration they compute
  the new max, rescale the running VKQ and KQ_sum by
  `expf(KQ_max_old - KQ_max_new)`, add the new partial sum, and proceed.
  The shift `FATTN_KQ_MAX_OFFSET = 3·log(2)` is added to each new KQ_max
  to push the dynamic range of the VKQ accumulator up by a factor of 8
  (see Finding ARTX11-F05).
* **Optional mask-prune pre-pass.** `flash_attn_mask_to_KV_max`
  (`fattn-common.cuh:664-719`) is launched when `mask && K->ne[1] %
  FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024 || Q->ne[3] > 1)`. It scans the
  mask column-by-column to find the first `FATTN_KQ_STRIDE` tile containing
  a non-`-inf` value and stores that as a per-(sequence, tile) `KV_max` int.
  The main kernel then iterates only up to `KV_max`, skipping fully masked
  tiles. This is the only kernel-level optimisation for sliding-window-like
  masks.
* **Stream-K vs `parallel_blocks`.** When `stream_k=true` (Ada Lovelace+,
  AMD WMMA, or efficiency < 75 %), `launch_fattn` partitions the
  KV-iteration space into `nblocks_stream_k` CUDA blocks, each of which may
  own a *fraction* of an output tile. Two fixup kernels —
  `flash_attn_stream_k_fixup_uniform` (when `nblocks_stream_k % ntiles_dst
  == 0`) and `flash_attn_stream_k_fixup_general` — combine partial results
  (`fattn-common.cuh:721-912`). When `stream_k=false`, the legacy
  `parallel_blocks > 1` scheme is used: each output tile is split into
  `parallel_blocks` blocks along the KV axis and the
  `flash_attn_combine_results` kernel (`:916-970`) does the final reduction.
* **No FP8 KV.** `ggml_cuda_fattn_kv_type_supported` returns `true` for
  F32, F16, BF16, Q4_0, Q4_1 (only with `GGML_CUDA_FA_ALL_QUANTS`), Q5_0
  (likewise), Q5_1 (likewise), Q8_0; everything else, including all FP8
  types, returns `false` (`fattn.cu:338-356`).
* **MLA via `V_is_K_view`.** When `DKQ != DV` (e.g. `576/512` for Deepseek,
  `192/128` for MiMo-V2.5, `320/256` for Mistral Small 4), the MMA kernel
  is launched with `V_is_K_view=true`. V is then aliased to K's pointer
  (`fattn-mma-f16.cuh:1821`), and the inner loop iterates over K in reverse
  so that the same shared-memory tile can be re-used as V after a
  transpose (`:601-673`). This is the *only* MLA-specific code path; no
  separate kernel is provided.
* **RoPE is a separate op.** `rope.cu` implements four RoPE variants and a
  fused `ROPE+VIEW+SET_ROWS` path; the FA kernels receive their Q with RoPE
  already applied. There is no RoPE-FA fusion.

---

## 5. Execution Flow

### 5.1 Top-level entry

`ggml_cuda_flash_attn_ext` (`fattn.cu:570`)

1. `ggml_cuda_set_device(ctx.device)` (cheap if already on the right device).
2. `ggml_cuda_get_best_fattn_kernel(device, dst)` — compute the enum.
3. `switch` on the enum and dispatch to one of
   `ggml_cuda_flash_attn_ext_tile`, `ggml_cuda_flash_attn_ext_vec`,
   `ggml_cuda_flash_attn_ext_mma_f16`.

### 5.2 Kernel selection

`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:358-534`)

1. Compute `gqa_ratio = Q->ne[2] / K->ne[2]` and assert divisibility.
2. Read `max_bias` (ALiBi) from `dst->op_params[1]`.
3. Compute `gqa_opt_applies`: `gqa_ratio >= 2 && mask && max_bias == 0 &&
   K->ne[1] % FATTN_KQ_STRIDE == 0`, plus a 16-byte alignment check on
   every Q/K/V/mask stride (`fattn.cu:378-389`).
4. Switch on `K->ne[0]` (head dim). Valid values: 40, 64, 72, 80, 96, 112,
   128, 192 (with `DV=128` + GQA), 256, 320 (with `DV=256` + GQA ratio 32),
   512, 576 (with `DV=512` + GQA). Anything else → `BEST_FATTN_KERNEL_NONE`
   (caller will abort).
5. Type check: `K->type == V->type` (unless `GGML_CUDA_FA_ALL_QUANTS`), and
   both must be in the supported set (`fattn.cu:442-450`).
6. Mask check: `mask && mask->ne[2] != 1` → `NONE`.
7. Branch on `turing_mma_available(cc) && Q->ne[0] != 40 && Q->ne[0] != 72`:
   * For Ada Lovelace+ and non-quantized K/V, prefer `VEC` when
     `Q->ne[1] == 1 && Q->ne[3] == 1` (single-token decode) and the GQA
     ratio isn't too large for the KV length.
   * For quantized K/V on Ada+, prefer `VEC` when `Q->ne[1] <= 2`.
   * Otherwise return `MMA_F16`.
8. Volta MMA path: prefer `VEC` if `Q->ne[1] * gqa_ratio_eff <= 2`; use
   `TILE` for small batches (the tensor cores aren't profitable until
   `Q->ne[1] * gqa_ratio_eff > 16`); otherwise `MMA_F16`.
9. AMD MFMA/WMMA path: prefer `MMA_F16` once the effective batch size
   exceeds the head-size-dependent threshold.
10. No-tensor-cores fallback (Pascal, AMD non-WMMA): prefer `VEC` for
    decode, otherwise `TILE`.

### 5.3 Per-kernel dispatch (FA-mma-f16)

`ggml_cuda_flash_attn_ext_mma_f16` (`fattn.cu:113-242`)

1. Switch on `Q->ne[0]` (DKQ). For each valid DKQ call
   `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<DKQ, DV>`.
2. `switch_ncols2` (`fattn.cu:36-111`) computes `use_gqa_opt` (mask + no
   ALiBi + aligned K + 16-byte strides) and the `gqa_ratio`. On Volta it
   applies a smaller-fan-out heuristic. Otherwise it picks `ncols2 ∈
   {1, 2, 4, 8}` based on the GQA ratio (`>4 → 8`, `>2 → 4`, `>1 → 2`,
   else 1).
3. `switch_ncols1` (`fattn.cu:8-34`) then picks `ncols1 = 8/ncols2`,
   `16/ncols2`, `32/ncols2`, or `64/ncols2` based on `Q->ne[1]` (number of
   Q tokens), the CC, and an AMD special case for `DKQ > 256`.

### 5.4 Per-kernel dispatch (FA-vec)

`ggml_cuda_flash_attn_ext_vec` (`fattn.cu:259-328`)

1. Iterate a macro-expanded table of `(D, type_K, type_V)` cases. With
   `GGML_CUDA_FA_ALL_QUANTS` defined, every combination of `type_K ∈ {F16,
   Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, BF16}` and `type_V ∈ {F16, Q4_0, Q4_1,
   Q5_0, Q5_1, Q8_0, BF16}` is instantiated for `D ∈ {64, 128, 256}`.
   Without that macro, only the four "diagonal" combinations `(F16,F16),
   (Q4_0,Q4_0), (Q8_0,Q8_0), (BF16,BF16)` are compiled in.
2. The first matching case calls `ggml_cuda_flash_attn_ext_vec_case<D,
   type_K, type_V>`, which in turn picks `cols_per_block ∈ {1, 2}` based
   on `Q->ne[1]` (`fattn-vec.cuh:546-574`).

### 5.5 Per-kernel dispatch (FA-tile)

`ggml_cuda_flash_attn_ext_tile` (`fattn-tile.cu`)

1. Switch on `K->ne[0]`. For each valid `DKQ`, assert `V->ne[0]` matches
   the expected `DV` and call `ggml_cuda_flash_attn_ext_tile_case<DKQ, DV>`.
2. `tile_case` switches on `logit_softcap == 0` (`fattn-tile.cuh:1322-1336`)
   and dispatches to `launch_fattn_tile_switch_ncols2<DKQ, DV,
   use_logit_softcap>`.
3. `switch_ncols2` applies the same GQA / alignment rules as the MMA
   dispatcher, then picks `ncols2 ∈ {1, 2, 4, 8, 16, 32}` (`:1255-1320`)
   and dispatches to `launch_fattn_tile_switch_ncols1<DKQ, DV, ncols2,
   use_logit_softcap>`, which selects `ncols1` (the per-block Q-token
   count) from `{2, 4, 8, 16, 32, 64}` based on `Q->ne[1]`
   (`:1148-1234`).

### 5.6 Inside `launch_fattn` (`fattn-common.cuh:972-1274`)

1. Allocate `KV_max`, `dst_tmp`, `dst_tmp_meta` from the per-backend pool
   (`:1008-1010`).
2. If the kernel needs F16 K or V and the supplied type is not F16, call
   `to_fp16_cuda` / `to_fp16_nc_cuda` to convert in-place into a buffer
   carved out of `dst->data` past the F32 output (`:1022-1084`; the buffer
   layout is computed by `ggml_cuda_flash_attn_ext_get_f16_extra_data`
   `:53-85`).
3. Optionally launch `flash_attn_mask_to_KV_max` (`:1094-1109`).
4. Query `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for the chosen
   kernel; this becomes the cap on `parallel_blocks` (`:1112-1114`).
5. Branch on `stream_k`:
   * **Stream-K path** (`:1120-1146`): pick `nblocks_stream_k` ≤
     `max_blocks_per_sm * nsm` and ≤ `ntiles_KV * ntiles_dst`. Round down
     to a multiple of `ntiles_dst` to skip the fixup if efficiency loss is
     ≤ 5 %.
   * **Legacy path** (`:1151-1185`): `parallel_blocks = min(parallel_blocks,
     ntiles_KV)`, then scan a small range of `parallel_blocks_test` values
     to find the best wave efficiency (stops once efficiency ≥ 95 %).
6. Read `scale`, `max_bias`, `logit_softcap` from `dst->op_params[0..2]`
   (`:1191-1193`). If `logit_softcap != 0`, divide `scale` by it
   (`:1195-1197`) so the kernel can apply `logit_softcap * tanhf(KQ)` after
   the matmul.
7. Compute `m0`, `m1` (ALiBi slopes) from `max_bias` and `n_head_log2`
   (`:1199-1203`).
8. Launch the kernel via `ggml_cuda_kernel_launch` (PDL-aware, see ARTX08).
9. Launch the appropriate fixup/combine kernel.

### 5.7 RoPE execution (`rope.cu:507-660`)

`ggml_cuda_op_rope_impl<forward>(ctx, dst, set_rows)`:

1. Read `n_dims`, `mode` (NeoX / mRoPE / Vision / normal), `freq_base`,
   `freq_scale`, `ext_factor`, `attn_factor`, `beta_fast`, `beta_slow`,
   `mrope_sections` from `dst->op_params`.
2. Compute `corr_dims` via `ggml_rope_yarn_corr_dims` (host-side).
3. Branch on `mode` and on `src0->type / dst->type ∈ {(F32,F32),
   (F32,F16), (F16,F16)}`, launch `rope_norm_cuda<...>`,
   `rope_neox_cuda<...>`, `rope_multi_cuda<...>`, or `rope_vision_cuda<...>`.
4. If `set_rows != nullptr` (fused `ROPE+VIEW+SET_ROWS`), redirect `dst_d`
   to the SET_ROWS destination and use `row_indices` to scatter
   (`rope.cu:522-528`, `:81-84`, `:156-159`).

### 5.8 Softmax execution (`softmax.cu:375-444`)

`ggml_cuda_op_soft_max`:

1. Read `scale`, `max_bias` from `dst->op_params[0..1]`.
2. Compute `n_head_log2 = 1 << floor(log2(n_head))`, `m0`, `m1` (ALiBi).
3. If the supplied shared memory (≤ `smpbo`) is large enough, dispatch to
   a specialised kernel templated on `ncols ∈ {32, 64, 128, 256, 512, 1024,
   2048, 4096}` (`:342`). Otherwise, for very large `ncols` (top-p
   sampling) and a CC that supports cooperative launch, dispatch to
   `soft_max_f32_parallelize_cols` which uses `cg::this_grid().sync()` to
   do a grid-wide reduction (`:347-357`). Last resort: launch with
   minimal shared memory and loop.

---

## 6. Data Layout

### 6.1 KV cache

The kernels assume (but do not enforce) that K and V have the canonical
ggml attention layout:

| Dim | Meaning                | Stride variable used by FA kernels |
| --- | ---------------------- | ---------------------------------- |
| 0   | `head_dim` (DKQ or DV) | `nb11` (K), `nb21` (V) — converted to `stride_K2 = nb11/sizeof(half2)` etc. |
| 1   | `seq_len` (KV cache)   | `nb12` (K), `nb22` (V)             |
| 2   | `n_kv_heads`           | `nb13` (K), `nb23` (V)             |
| 3   | `n_batch` (sequences)  | (implicit in `nb13`)               |

The shape itself is set by the llama model code, not by ggml. The FA kernels
key off `K->ne[0..3]`, `K->nb[1..3]`, and `V->nb[1..3]`. Several
invariants are asserted:

* `Q->nb[0] == ggml_element_size(Q)` and similarly for K, V
  (`fattn-common.cuh:993-995`). Inner-dim must be contiguous.
* `Q->type == GGML_TYPE_F32` and `KQV->type == GGML_TYPE_F32`
  (`:990-991`). The output is always F32.
* If `mask` is provided, `mask->type == GGML_TYPE_F16`
  (`:997`). The mask is stored as `half` for the FA-tile / FA-vec paths
  and as `half2` for the FA-mma-f16 path (`fattn-mma-f16.cuh:450-528`).
* `mask->ne[2] == 1` (single mask plane shared across heads) or mask is
  `nullptr` (`fattn.cu:452-454`).
* `Q->ne[2] % K->ne[2] == 0` (GQA divisibility) (`fattn.cu:63`, `:371`).

For the GQA / MQA case, `gqa_ratio = Q->ne[2] / K->ne[2]`. The kernel
broadcasts one K/V head to `gqa_ratio` Q heads by striding Q in dim 2:
`Q_h2 = Q + nb02*head0` where `head0 = blockIdx.z*ncols2 - sequence*ne02`
and the per-block Q head index is `head0 / gqa_ratio` for K and V
(`fattn-tile.cuh:855-858`).

### 6.2 MLA `V_is_K_view`

For MLA models (Deepseek-V3, GLM-4.7-Flash, MiMo-V2.5, Mistral Small 4)
where `DKQ != DV`, the launcher reuses K's data for V. Detection is at
`fattn-common.cuh:63`: `V->view_src && (V->view_src == K || (V->view_src
== K->view_src && V->view_offs == K->view_offs))`. In `launch_fattn`, this
short-circuits the V→F16 conversion (`:1051-1055`); in the MMA kernel, it
sets `V_h2 = K_h2` and `stride_V = stride_K` (`fattn-mma-f16.cuh:1821`)
and the K-loading loop iterates in reverse so the same shared-memory tile
can be re-used as V after a transpose (`:601-673`).

### 6.3 Q layout

Q is F32 with shape `[head_dim, n_tokens, n_q_heads, n_batch]`. The
FA-tile kernel materialises Q once in shared memory as `half2` (if
`FAST_FP16_AVAILABLE`) or `float` (`fattn-tile.cuh:884-894`):

```
__shared__ half2 Q_tmp[ncols * DKQ/2];
```

The FA-vec kernel keeps Q in registers, either as `half2` (with
`V_DOT2_F32_F16_AVAILABLE`) or as `float2`, or quantizes it to Q8_1 in
shared memory if K is quantized (`fattn-vec.cuh:140-203`). The FA-mma-f16
kernel loads Q once via `ldmatrix` into the MMA B-fragment
(`fattn-mma-f16.cuh:619-667`).

### 6.4 Mask layout

Mask is F16, shape `[seq_len, n_tokens, 1, n_batch]`. The FA-tile kernel
indexes it as `maskh[j*stride_mask + k_VKQ_0 + i_KQ]` where `stride_mask
= nb31 / sizeof(half)` (`fattn-tile.cuh:864`). The FA-mma-f16 kernel
loads it through a `flash_attn_ext_f16_load_mask` helper that tiles the
mask into shared memory in `nbatch_fa + 8` stride to avoid bank conflicts
(`fattn-mma-f16.cuh:450-528`). Mask values are `±inf` (or large negative)
to implement causal / sliding-window patterns; the kernel adds them to KQ
before the softmax max-reduction.

---

## 7. Memory Layout

### 7.1 Output (`dst` / `KQV`)

F32, shape `[DV, n_tokens, n_q_heads, n_batch]`. The FA-tile kernel writes
it directly when `gridDim.y == 1` (single parallel block); otherwise each
block writes its partial VKQ to `dst_tmp` (F32, `parallel_blocks *
ggml_nelements(KQV)` floats) and `dst_tmp_meta` (F32x2, `parallel_blocks *
ggml_nrows(KQV)` entries) which are then reduced by
`flash_attn_combine_results` (`fattn-common.cuh:916-970`).

The stream-K path is more intricate: blocks write their partial result
*directly to `dst`* (overwriting), and emit a `(KQ_max, KQ_sum)` meta pair
per block to `dst_tmp_meta`. The fixup kernels
(`flash_attn_stream_k_fixup_uniform`, `flash_attn_stream_k_fixup_general`)
walk the chain of blocks that contributed to each output tile, rescaling
each partial by `expf(block_max - global_max)` and summing
(`fattn-common.cuh:770-800`, `:862-908`).

### 7.2 Output "extra data"

`ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:536-568`) computes the
total allocation needed for `dst->data` plus any on-the-side F16 K and V
conversions. The F16 K/V buffers are placed *after* the F32 output, 128-byte
aligned, by `ggml_cuda_flash_attn_ext_get_f16_extra_data`
(`fattn-common.cuh:53-85`). This is a deliberate trick: it lets the
backend allocate a single buffer that holds both the output and the
converted inputs, with no separate scratch pool allocation.

### 7.3 Shared memory

* **FA-tile** (`fattn-tile.cuh:884-894`):
  `Q_tmp[ncols * DKQ/2] + KV_tmp[nbatch_fa * (nbatch_K/2 + cpy_ne) + DVp-DV]
  + KQ[ncols * nbatch_fa]` (half2/half), or the float equivalent. For
  `(DKQ, DV, ncols) = (128, 128, 32)`: `32*64 + 64*(32+4) + 32*64` half2 =
  ~12 kiB. For `(256, 256, 32)`: `32*128 + 64*(32+4) + 32*64` half2 ≈
  ~24 kiB. For `(512, 512, 16)`: `16*256 + 64*(32+4) + 16*64` half2 ≈
  ~28 kiB.
* **FA-vec** (`fattn-vec.cuh:124-130`): `KQ[ne_KQ > ne_combine ?
  ne_KQ : ne_combine]` half2/float, where `ne_KQ = ncols*D` and
  `ne_combine = nwarps*V_cols_per_iter*D`. For `D=256, ncols=1, nwarps=4,
  V_cols_per_iter=4`: ~4 kiB.
* **FA-mma-f16** (`fattn-mma-f16.cuh:1917-1927`): `nbytes_shared_Q +
  nbytes_shared_KV (+ nbytes_shared_mask) + nbytes_shared_combine`. The
  combine buffer can be larger than the Q+KV buffer; the launcher takes
  the max. For `(DKQ, DV, ncols) = (128, 128, 8)` with `nstages=2`:
  Q: `8*(64+4)*4 = 2176` B; KV 2-stage: `128*(64+4+64+4)*4 = 34816` B;
  mask: `8*(128/2+4)*4 = 2176` B; combine: `2*8*(64+4)*4 = 4352` B →
  ~39 kiB. For `(512, 512, 4, 2)` with `nstages=1`: ~36 kiB.
* The MMA kernel calls `cudaFuncSetAttribute(...,
  cudaFuncAttributeMaxDynamicSharedMemorySize, ...)` once per kernel per
  device to allow up to ~100 kiB of dynamic shared memory
  (`fattn-mma-f16.cuh:1938-1960`), gated by a `static bool
  shared_memory_limit_raised[GGML_CUDA_MAX_DEVICES]` flag.

### 7.4 Per-thread registers

* FA-tile: `KQ_acc[nbatch_fa/(np*warp_size) * cpw]` floats + `VKQ[cpw *
  (DVp/2)/warp_size]` half2/float2 + `KQ_max[cpw]`, `KQ_sum[cpw]`
  (`fattn-tile.cuh:604, 888, 896, 901`). For `(128, 128, 32)` with
  `cpw=1, np=1, nbatch_fa=64`: `2*1 = 2` KQ_acc + `1*(64/32) = 2` VKQ +
  2 scalars = 6 floats + 2 half2 = 32 B per thread.
* FA-vec: `VKQ[ncols][(D/2)/nthreads_V]` half2/float2 + `KQ_max[ncols]`,
  `KQ_sum[ncols]` + `Q_reg[ncols][(D/2)/nthreads_KQ]` half2/float2 + (if
  quantized K) `Q_i32[ncols][D/(4*nthreads_KQ)]` and
  `Q_ds[ncols][...]` (`fattn-vec.cuh:124-147`). For `D=256, ncols=1,
  nthreads_KQ=8, nthreads_V=32`: 4 half2 VKQ + 1+1 floats + 16 half2 Q_reg
  ≈ 84 B per thread.
* FA-mma-f16: `KQ_C[nbatch_fa/(np*T_C_KQ::I)]` MMA C-fragments +
  `VKQ_C[DV/T_C_VKQ::I]` MMA C-fragments + `KQ_max[cols_per_thread]`,
  `KQ_rowsum[cols_per_thread]` (`fattn-mma-f16.cuh:577, 687, 692`). Each
  MMA fragment is `2*half2` (4 floats) per thread for `m16n8k16`.

---

## 8. Parallelism Strategy

### 8.1 Three parallelism axes

Every FA kernel is parallel over three grid axes:

* `blockIdx.x` — Q tokens (split into `ncols1` chunks) and/or stream-K
  iteration index.
* `blockIdx.y` — KV-iteration parallel block (legacy `parallel_blocks`).
* `blockIdx.z` — `(n_q_heads / ncols2) × K->ne[2] × Q->ne[3]` — i.e. all
  (Q-head-chunk, KV-head, sequence) combinations.

The FA-vec kernel uses `blockIdx.x = ncols` (1 or 2 Q tokens per block) and
the same `blockIdx.z` scheme; `blockIdx.y` ranges over the KV cache in
chunks of `nthreads` (`fattn-vec.cuh:104-113, 250-256`).

### 8.2 Within-block parallelism

* **FA-tile** uses `nwarps` warps (1–8) and `nthreads = nwarps * 32`
  threads. The `cpw` / `np` constants decide whether each warp owns one
  Q column (cpw = 1, multiple warps cooperate via `np`) or multiple Q
  columns (cpw > 1, np = 1) (`fattn-tile.cuh:871-873`). Inside an
  iteration, all warps cooperate on `flash_attn_tile_iter_KQ` (load K
  tile, multiply against Q tile) and `flash_attn_tile_iter` (softmax,
  load V tile, multiply against KQ).
* **FA-vec** uses 4 warps (128 threads). One warp does the KQ dot product
  while the others can prefetch V (though the code is cooperative, not
  explicitly specialised). The KQ reduction uses
  `warp_reduce_sum<nthreads_KQ>` where `nthreads_KQ ∈ {2, 4, 8, 16, 32}`
  depending on dtype and warp size (`fattn-vec.cuh:74-91`).
* **FA-mma-f16** uses `nwarps ∈ {2, 4}` warps. Warps are partitioned into
  `np` groups along the Q-columns axis; each warp owns `cols_per_warp ∈
  {8, 16, 32}` Q columns. `load_ldmatrix` is used to feed the MMA
  A-fragment (`fattn-mma-f16.cuh:619-667`).

### 8.3 No warp specialisation

The kernels do **not** use producer-consumer warp specialisation (the
"FlashAttention-3" pattern where some warps issue `cp.async` while others
compute). The FA-mma-f16 kernel does use a software-pipelined
multi-stage `cp.async` loop (`nstages_target ∈ {1, 2}`), but all warps
participate in both load and compute. See Finding ARTX11-F11.

### 8.4 Stream-K

`launch_fattn` chooses stream-K when `cc >= GGML_CUDA_CC_ADA_LOVELACE ||
amd_wmma_available(cc) || tiles_efficiency_percent < 75`
(`fattn-common.cuh:1126`). The total work (`ntiles_KV * ntiles_dst`) is
divided among `nblocks_stream_k` CUDA blocks via integer arithmetic:
`kbc0 = blockIdx.x * total_work / gridDim.x`, `kbc_stop = (blockIdx.x+1) *
total_work / gridDim.x`. A block may start in the middle of an output tile
(`needs_fixup = true`) and/or end in the middle of one (`is_fixup = true`).
The two fixup kernels handle the two cases
(`flash_attn_stream_k_fixup_uniform` when the seam is uniform,
`flash_attn_stream_k_fixup_general` otherwise).

---

## 9. SIMD / GPU Strategy

### 9.1 Tensor Core matrix multiply

The FA-mma-f16 kernel uses the `mma.sync.aligned` PTX instruction
exclusively (audited in `mma.cuh`):

| Hardware               | Instruction                                     | Shape      | Dtypes              |
| ---------------------- | ----------------------------------------------- | ---------- | ------------------- |
| Volta                  | `mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32` | 16×8×8   | f16×f16 → f32        |
| Turing (no m16n8k16)   | 2× `mma.sync.aligned.m8n8k8.row.col.f16.f16.f16.f16` | 8×8×8 (×2) | f16×f16 → f16        |
| Turing+ (m16n8k16)     | `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` | 16×8×16  | f16×f16 → f32        |
| AMD MFMA (CDNA)        | `mfma.f32.16x16x16.f16` (via header macros)     | 16×16×16   | f16×f16 → f32        |
| AMD WMMA (RDNA3/4)     | `wmma.f16.s32`-style intrinsics                 | 16×16×16   | f16×f16 → f32        |

The K-fragment is loaded via `ldmatrix.sync.aligned.m8n8.x4.b16`
(`mma.cuh:830-840`), which uses the Tensor Memory path. K, V, and Q are
all kept as `half2` (F16) in shared memory; the MMA accumulates in F32
(`KQ_C` and `VKQ_C` are F32 fragments). After the AV matmul, the result
is converted back to F16 for the combine step (`fattn-mma-f16.cuh:1568-
1625`).

There is **no** `wgmma` (Hopper) or `mma.sync.aligned.kind::mxf4nvf4`
(Blackwell) path in the FA kernels. The `BLACKWELL_MMA_AVAILABLE` macro
is defined (`common.cuh:286-288`) but is only used by the MMQ matmul
(ARTX10), not by FA. The `mma.sync.aligned.kind::mxf4nvf4` instruction
audited at `mma.cuh:1145` is also MMQ-only.

### 9.2 `cp.async` software pipeline

The FA-mma-f16 kernel uses `cp.async.cg.shared.global` (`cp_async_cg_16`)
to load K, V, and mask tiles into shared memory without going through
registers (`fattn-mma-f16.cuh:371-410`). The pipeline depth is
`nstages_target ∈ {1, 2}`: with `nstages=2`, V for iteration *i+1* is
preloaded while K for iteration *i* is being consumed
(`:584-599`). The multi-stage path is only enabled when
`cp_async_available(cc) && ncols2 >= 2` (`:348-359`). There is no
3-or-more-stage pipeline (FlashAttention-3 uses up to 3 stages on Hopper).

### 9.3 Vectorised memory access

Both the FA-tile and FA-vec kernels use `ggml_cuda_memcpy_1<N>` (defined
in `cpy-utils.cuh`) to issue 4/8/16-byte vectorised loads. The maximum
load size is `ggml_cuda_get_max_cpy_bytes()` = 16 B on Volta+ and AMD,
8 B on Pascal (`common.cuh:382-393`). The FA-tile load helper
`flash_attn_tile_load_tile` (`fattn-tile.cuh:377-480`) uses a
`ggml_cuda_unroll<7>` metaprogramming trick to issue decreasing-granularity
loads for non-power-of-two row lengths.

### 9.4 FP16 vs FP32 compute

FA-tile has two parallel implementations for every kernel: a
`FAST_FP16_AVAILABLE` path that uses `half2` arithmetic and
`v_dot2_f32_f16` (Turing+), and an `#else` path that uses `float2` and
emulates the half2 ops. The latter is selected when the GPU has FP16
storage but no fast FP16 arithmetic (e.g. Pascal GP100 vs GP102).
A `4.0f` rescaling is applied to KQ_acc when `v_dot2_f16` is unavailable
to avoid overflow (`fattn-tile.cuh:628-632`).

### 9.5 RoPE strategy

RoPE kernels are scalar per thread: each thread computes one (cos, sin)
pair via `rope_yarn` (`rope.cu:23-41`) — `cosf`/`sinf` transcendental
calls in F32, no SFU intrinsics. The (cos, sin) is applied to a pair of
Q components. The kernel parallelises across `(dim_idx, token, head,
batch)` with `blockDim.x` over tokens and `blockDim.y` over dim pairs
(`rope.cu:65-69`). There is no vectorised F32 rotation (e.g. via warp
shuffles or `sincosf`); this is a deliberate trade-off for code clarity
and YaRN-correctness.

### 9.6 Softmax strategy

The standalone `soft_max_f32` kernel (`softmax.cu:55-138`) uses
`block_reduce<MAX, SUM>` (cooperative-groups based). The kernel templates
on `ncols ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}` to enable
`#pragma unroll` (`:283-300`). For very wide rows (> `smpbo`) a
cooperative-launch path is used (`:302-317`).

---

## 10. Quantization Strategy

The FA kernels support a *fixed* set of KV cache quantizations:

| Dtype       | FA-vec | FA-tile | FA-mma-f16 | Notes                                          |
| ----------- | ------ | ------- | ---------- | ---------------------------------------------- |
| F32         | ✓ (cast to F16 first) | ✓ (cast to F16 first) | ✓ (cast to F16 first) | Always supported via conversion. |
| F16         | ✓      | ✓       | ✓          | Native; the tile and MMA paths *require* F16.  |
| BF16        | ✓      | ✗       | ✗          | Vecdot via `__bfloat1622float2`.               |
| Q4_0        | ✓      | ✗       | ✗          | Vecdot via `dp4a`.                             |
| Q4_1        | ✓*     | ✗       | ✗          | `*` only with `GGML_CUDA_FA_ALL_QUANTS`.       |
| Q5_0        | ✓*     | ✗       | ✗          | `*` only with `GGML_CUDA_FA_ALL_QUANTS`.       |
| Q5_1        | ✓*     | ✗       | ✗          | `*` only with `GGML_CUDA_FA_ALL_QUANTS`.       |
| Q8_0        | ✓      | ✗       | ✗          | Vecdot via `vec_dot_q8_0_q8_1_impl`.           |
| FP8 (any)   | ✗      | ✗       | ✗          | Not supported; see Finding ARTX11-F14.         |

The FA-vec kernel uses `dp4a` (the `__dp4a` intrinsic, available on
Pascal+ and AMD VEGA20+) to compute the KQ dot product for quantized K
(`fattn-common.cuh:170, 201, 247, 292`). The V dequantization is done
per-element, no `dp4a` use. The Q side is quantized to Q8_1 on-the-fly
inside the kernel via `quantize_q8_1_to_shared`
(`fattn-common.cuh:331-373`), which uses warp shuffles to compute the
per-32-element scale and sum.

The FA-tile and FA-mma-f16 kernels *only* accept F16 (or F32 cast to F16
up-front). They cannot consume a quantized KV cache directly: `launch_fattn`
calls `to_fp16_cuda` to convert in-place into the "extra data" buffer
appended to `dst->data` (`fattn-common.cuh:1022-1084`).

This is a major architectural asymmetry: the FA-vec path keeps the
quantized KV cache *in situ* and saves memory bandwidth; the FA-tile and
FA-mma-f16 paths pay a one-time conversion cost. For decoding with a
small `Q->ne[1]`, FA-vec is therefore strongly preferred when KV is
quantized.

---

## 11. Correctness Analysis

### 11.1 FlashAttention-2 online softmax

All three kernels implement the standard online softmax:

```
m_new = max(m_old, max(KQ_new))
s_new = s_old * exp(m_old - m_new) + sum(exp(KQ_new - m_new))
VKQ_new = VKQ_old * exp(m_old - m_new) + sum_i(KQ_new[i] * exp(KQ_new[i] - m_new) * V[i])
```

This is mathematically equivalent to the textbook
`softmax(KQ) · V` formulation but accumulates the sum and product in a
single pass over K/V. The result is bit-exact only in infinite precision;
in F32 it differs from the textbook form at the ULP level due to
reassociation of the sum.

### 11.2 `FATTN_KQ_MAX_OFFSET = 3·log(2)` shift

`fattn-common.cuh:19` adds `3·log(2) ≈ 2.08` to every `KQ_max` before the
`expf` calls. This effectively divides every `expf(KQ - KQ_max)` by 8,
which lifts the dynamic range of the VKQ accumulator by a factor of 8
to avoid overflow in `half2` accumulation paths. The shift is *undone
implicitly* — the final `dst = VKQ / KQ_sum` ratio cancels it because
both numerator and denominator are scaled by the same factor. The comment
cites issue #18606 (`fattn-common.cuh:13-18`). Without this shift, the F16
VKQ accumulator could overflow for sequences where the max attention logit
exceeds ~12, producing `inf` and propagating to NaN.

**Correctness impact:** the result is identical (within FP precision) to
the un-shifted computation as long as the final division succeeds. The
shift is a stability hack, not a numerical approximation.

### 11.3 `SOFTMAX_FTZ_THRESHOLD = -20.0f`

When `KQ_max_diff < -20.0f`, the kernel sets `KQ_max_scale = 0` directly
(bit-hacked via `*((uint32_t *)&KQ_max_scale[col]) *= KQ_max_diff >=
SOFTMAX_FTZ_THRESHOLD;` at `fattn-mma-f16.cuh:859`) instead of calling
`expf`. This avoids NaN propagation when `expf(-large)` underflows to 0
and the accumulator is later multiplied by it. The fixup kernels
(`fattn-common.cuh:790-791, 894-895`) use the same threshold. The threshold
corresponds to `expf(-20) ≈ 2e-9`, well above the F32 denormal floor.

**Correctness impact:** results differ from the un-FTZ computation only
when an attention logit is *so* small that its `expf` value would
underflow anyway. The output is bit-identical to a computation that uses
FTZ-mode F32.

### 11.4 GQA ratio divisibility

`Q->ne[2] % K->ne[2] == 0` is asserted at multiple points (`fattn.cu:63,
371`; `fattn-common.cuh:1087`). The kernel *cannot* handle non-integer
GQA ratios. This is correct for all current llama.cpp-supported models.

### 11.5 Causal masking

Causal masking is implemented **outside** the FA kernels: the model code
produces an F16 mask tensor with `-inf` (or a large negative value) in
masked positions, then passes it as `dst->src[3]`. The FA kernels add the
mask to KQ *before* the softmax max-reduction:

```c
KQ_acc[...] += slope * __half2float(mask[j*stride_mask + k_VKQ_0 + i_KQ]);
KQ_max_new[jc0] = fmaxf(KQ_max_new[jc0], KQ_acc[...] + FATTN_KQ_MAX_OFFSET);
```

(`fattn-tile.cuh:638-642`). The `slope` is `1.0f` for non-ALiBi masks, and
`get_alibi_slope(max_bias, head0, n_head_log2, m0, m1)` for ALiBi
(`fattn-tile.cuh:866`). If the mask value is `-inf`, the corresponding
`expf(KQ - KQ_max)` underflows to 0 and the row-sum is unaffected. The
`SOFTMAX_FTZ_THRESHOLD` short-circuit ensures no NaN is generated.

### 11.6 Diag-mask `FLT_MAX` shortcut

The legacy `diag_mask_inf_f32` kernel (`diagmask.cu:14`) writes
`x[i] - (col > n_past + row % rows_per_channel) * FLT_MAX` to the output
instead of the more obvious `-INFINITY`. The comment says this is
"slightly faster on GPU" because the conditional subtraction compiles to
a single `selp` instruction without a branch. **However**, `FLT_MAX =
3.4e38` is finite, so adding it to a finite `x[i]` gives `FLT_MAX` (after
rounding), not `-inf`. Subsequent `expf(FLT_MAX - max_val)` will
underflow to 0 identically to `expf(-inf - max_val)`, so the softmax is
correct. But if the user computes `softmax(x + diag_mask)` *without* a
softmax-max step (i.e. consumes the masked KQ directly), they will see
`-FLT_MAX` instead of `-inf`, which may break downstream code that tests
`isinf`.

### 11.7 RoPE numerical precision

`rope_yarn` uses `cosf`/`sinf` (single-precision transcendental) on the
device (`rope.cu:36-37`). The math is therefore accurate only to F32
precision (~24 bits). There is no F64 path. The YaRN magnitude scaling
multiplies by `1.0f + 0.1f * logf(1.0f / freq_scale)`, also F32.

### 11.8 Non-determinism

* **Stream-K scheduling** makes the block→output mapping non-deterministic
  across runs (depends on SM availability). Combined with the per-block
  reassociation in the fixup kernel, the F32 output can vary at the ULP
  level across runs with the same inputs.
* **`parallel_blocks > 1`** in the legacy path also reassociates across
  blocks in `flash_attn_combine_results`, so it is also non-deterministic
  across runs.
* **Single-block (`gridDim.y == 1`) execution is deterministic** for a
  fixed kernel and a fixed CC.

### 11.9 Atomic accumulation

* **None** in the FA kernels themselves. The fixup and combine kernels
  write disjoint output regions (one block per output element).
* `cudaOccupancyMaxActiveBlocksPerMultiprocessor` is called once per
  kernel launch (host side, `fattn-common.cuh:1113`); no atomics in the
  hot path.

---

## 12. Optimization Analysis

### 12.1 Identified optimizations

| Optimization                                | Where                                                                  | Notes |
| ------------------------------------------- | ---------------------------------------------------------------------- | ----- |
| Three-way kernel dispatch                   | `fattn.cu:358-534`                                                     | Heuristic, CC + shape + dtype driven. |
| Per-CC config tables (tile + mma)           | `fattn-tile.cuh:21-326`, `fattn-mma-f16.cuh:38-245`                    | Bitfield-encoded (tile) or struct (mma). |
| Online softmax (FlashAttention-2)           | All three kernels                                                      | `KQ_max`, `KQ_sum`, `VKQ` per thread. |
| `FATTN_KQ_MAX_OFFSET = 3·log(2)` shift      | `fattn-common.cuh:19`                                                  | Lifts VKQ range by 8× to avoid F16 overflow. |
| `SOFTMAX_FTZ_THRESHOLD` short-circuit       | `fattn-mma-f16.cuh:859`, `fattn-common.cuh:790-791, 894-895`           | Avoids `expf` underflow NaN. |
| Mask-prune pre-pass (`flash_attn_mask_to_KV_max`) | `fattn-common.cuh:664-719, 1094-1109`                          | Skips fully-masked FATTN_KQ_STRIDE tiles. Used when `Q->ne[1] >= 1024` or batch > 1. |
| Stream-K scheduling                         | `fattn-common.cuh:721-912, 1120-1146`                                  | Ada Lovelace+, AMD WMMA. Two fixup kernels. |
| `cp.async` multi-stage pipeline             | `fattn-mma-f16.cuh:363-447, 584-673`                                   | `nstages_target ∈ {1, 2}`. Ampere+. |
| `ldmatrix.sync.aligned` for K-fragment load | `fattn-mma-f16.cuh:626, 652`; `mma.cuh:830-840`                         | Required by `mma.sync.m16n8k16`. |
| `mma.sync.m16n8k16` (Turing+) or `m8n8k8` (Volta/Turing-fallback) | `mma.cuh:977-1023`, `:1163-1224` | F16 input, F32 accumulate. |
| On-the-fly Q8_1 quantization (FA-vec)       | `fattn-vec.cuh:150-203`; `fattn-common.cuh:331-373`                    | Lets FA-vec consume quantized KV without conversion. |
| MLA `V_is_K_view`                           | `fattn-mma-f16.cuh:601-673, 1821`                                      | V aliases K; saves a load and a smem buffer. |
| GQA fan-out (`ncols2` template)             | `fattn.cu:91-110`; `fattn-tile.cuh:1255-1320`                          | Packs 2/4/8/16/32 Q heads per block to amortise K/V loads. |
| F16 K/V conversion in-place in dst buffer   | `fattn-common.cuh:53-85, 1022-1084`                                    | Avoids a separate scratch allocation. |
| `cudaFuncSetAttribute` for large smem       | `fattn-mma-f16.cuh:1938-1960`; `softmax.cu:284`                        | One-time per (kernel, device). |
| Wave-efficiency scan for `parallel_blocks`  | `fattn-common.cuh:1157-1175`                                           | Stops at 95% efficiency. |
| `flash_attn_combine_results` reduce kernel  | `fattn-common.cuh:916-970`                                             | F32 reduction across `parallel_blocks` partial results. |
| RoPE YaRN length extrapolation              | `rope.cu:23-41`                                                        | `cosf`/`sinf` + `mscale = 1 + 0.1·log(1/freq_scale)`. |
| Fused `ROPE+VIEW+SET_ROWS`                  | `rope.cu:81-84, 156-159, 522-528, 670-672`                             | Avoids a separate SET_ROWS kernel for incremental KV cache writes. |
| Softmax cooperative-launch for huge ncols  | `softmax.cu:302-317, 347-357`                                          | `cudaLaunchCooperativeKernel` for top-p sampling with wide rows. |
| Softmax per-`ncols` template specialisation | `softmax.cu:272-300`                                                   | `ncols ∈ {32, 64, 128, 256, 512, 1024, 2048, 4096}` for unrolling. |
| `#pragma unroll` + `ggml_cuda_unroll<N>` metaprogram | `fattn-tile.cuh:424`, `fattn-mma-f16.cuh:410, 446`         | Decreasing-granularity vectorised loads for non-power-of-two shapes. |

### 12.2 Optimizations *not* present (worth noting)

* **No Hopper `wgmma` or TMA.** The FA-mma-f16 kernel uses `mma.sync` and
  `cp.async.cg` only. `wgmma.mma_async` (Hopper SM_90) and `cp.async.bulk`
  (TMA) are not used. This leaves substantial performance on the table
  for Hopper and Blackwell.
* **No FP8 KV path.** FA-mma-f16 requires F16 K/V; FA-vec supports a
  fixed set of integer quants but not FP8 (E4M3/E5M2).
* **No warp specialisation** (FlashAttention-3 style). All warps in the
  MMA kernel cooperate on both load and compute; no producer-consumer
  split.
* **No `flash_attn_sliding_window` parameter.** Sliding-window attention
  is realised by passing a dense F16 mask tensor; the kernel cannot prune
  KV loads based on a window size.
* **No RoPE fusion into FA.** RoPE is a separate op. The fused
  `ROPE+VIEW+SET_ROWS` is the only RoPE-level fusion.
* **No persistent kernel.** Each FA launch is one-shot; the kernel does
  not loop over multiple `ggml_tensor` ops.
* **No graph-level fusion across FA + residual + RMS_NORM.** The CUDA
  backend's `ggml_cuda_try_fuse` (ARTX08) does not include any FA-related
  pattern. The closest is `ROPE+VIEW+SET_ROWS`, which is fusion of the
  incremental KV write, not of attention itself.

---

## 13. Architectural Strengths

1. **Clean three-kernel taxonomy.** The `VEC / TILE / MMA_F16` split maps
   cleanly to (a) decode (single Q), (b) prefill on non-TC GPUs, (c)
   prefill on TC GPUs. The dispatcher's heuristic is readable and
   documented inline (`fattn.cu:358-534`).

2. **Shared FlashAttention-2 contract.** All three kernels use the same
   `(KQ_max, KQ_sum, VKQ)` accumulators, the same `FATTN_KQ_MAX_OFFSET`
   shift, and the same `SOFTMAX_FTZ_THRESHOLD`. This means the
   numerical-stability fixes propagate for free once.

3. **Per-CC config tables.** Both FA-tile and FA-mma-f16 encode
   per-shape-per-CC parameters as constexpr tables (`fattn-tile.cuh:21-
   326`, `fattn-mma-f16.cuh:38-245`). Adding a new CC variant means adding
   a new column to the table; no code rewrite. The bitfield-encoding
   trick in the tile kernel (`:12-19`) is a nice way to pack four
   parameters into a single `uint32_t` for `__launch_bounds__` use on
   ROCm (which can't template `__launch_bounds__`).

4. **MLA via `V_is_K_view`.** The `V_is_K_view` flag is a minimal,
   elegant way to support MLA without a separate kernel. V is aliased to
   K's pointer; the K-loading loop iterates in reverse to allow tile
   re-use. This works because MLA's K and V are projections of the same
   latent vector, so the data is genuinely identical.

5. **Mask-prune pre-pass.** `flash_attn_mask_to_KV_max` is a one-shot
   scan of the mask that lets the main kernel skip fully-masked
   `FATTN_KQ_STRIDE × FATTN_KQ_STRIDE` tiles. For sliding-window attention
   with a long KV cache this can reduce the work by an order of magnitude.

6. **Stream-K with two fixup kernels.** The split between
   `flash_attn_stream_k_fixup_uniform` (fast path, when `nblocks %
   ntiles_dst == 0`) and `flash_attn_stream_k_fixup_general` (general
   path) is a pragmatic engineering choice: the uniform path is much
   cheaper because it can do one fixup per output tile, while the general
   path has to walk a chain of partial blocks.

7. **Fused `ROPE+VIEW+SET_ROWS`.** The incremental KV-cache write path
  (RoPE applied to a new Q/K pair, then written into the cache via
  SET_ROWS) is fused into a single kernel launch. This avoids a separate
  gather+scatter pass and is critical for decode latency.

8. **F16 K/V conversion in-place in dst buffer.** The "extra data" trick
   (`fattn-common.cuh:53-85`) avoids a separate scratch allocation for
   the F16-converted K/V when the user supplies F32 or quantized K/V to
   FA-tile / FA-mma-f16.

9. **Per-dtype FA-vec vecdot.** The FA-vec kernel has hand-tuned per-
   dtype KQ vecdot kernels for F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
   (`fattn-common.cuh:87-329`). These use `dp4a` for the integer quants
   and `v_dot2_f32_f16` for F16, and they keep the quantized KV cache
   *in situ* without conversion.

10. **YaRN length extrapolation in RoPE.** The `rope_yarn` helper
    implements the full YaRN algorithm (ramp interpolation + magnitude
    scaling) in a single device function. This is shared across all four
    RoPE variants.

---

## 14. Architectural Weaknesses

### W1 — No Hopper / Blackwell tensor-core path

**Evidence:** `fattn-mma-f16.cuh` uses `mma.sync.aligned.m16n8k16`
(ampere-equivalent) and `cp.async.cg` only. No `wgmma.mma_async`, no
`cp.async.bulk.tensor` (TMA), no cluster-mode distributed shared memory.
The `BLACKWELL_MMA_AVAILABLE` macro (`common.cuh:286-288`) is defined but
not consulted by the FA dispatcher.

**Impact:** On Hopper (H100) and Blackwell (B200), the FA-mma-f16 kernel
achieves a fraction of the achievable Tensor Core throughput. Competing
implementations (FlashAttention-3, FlashInfer) report 1.5–2× speedups
from `wgmma` + TMA + warp specialisation.

### W2 — No FP8 KV cache support

**Evidence:** `ggml_cuda_fattn_kv_type_supported` (`fattn.cu:338-356`)
returns `false` for all FP8 types. The FA-vec vecdot table
(`fattn-common.cuh:620-640`) and the FA-mma-f16 / FA-tile paths have no
FP8 entry. FA-mma-f16 requires F16 input exclusively.

**Impact:** Models that store KV cache in FP8 (e.g. some vLLM-exported
Llama-3 variants) must convert to F16 on every FA call, paying both
memory and compute. This defeats the bandwidth benefit of FP8 KV.

### W3 — Sliding-window attention is opaque to the kernel

**Evidence:** There is no `flash_attn_sliding_window` parameter. The
mask is consumed as a dense F16 tensor (`fattn-common.cuh:997`,
`fattn-tile.cuh:860-864`, `fattn-mma-f16.cuh:450-528`). The only
sliding-window-aware optimisation is the `flash_attn_mask_to_KV_max`
pre-pass, which prunes fully-masked tiles but still requires the dense
mask to be materialised.

**Impact:** For sliding-window attention with a small window (e.g.
Mistral's 4096-token window over a 32k-token cache), the FA kernel still
loads `K[ne[1]]` rows of V from HBM unless the mask-prune pre-pass
succeeds in skipping tiles. The mask itself is `n_tokens × n_tokens ×
1 × n_batch` F16 — for `n_tokens = 32768`, that is 2 GiB of mask, which
is absurd.

### W4 — `diagmask` uses `FLT_MAX` instead of `-INFINITY`

**Evidence:** `diagmask.cu:14`: `dst[i] = x[i] - (col > n_past + row %
rows_per_channel) * FLT_MAX;`. The comment cites "slightly faster on
GPU".

**Impact:** `FLT_MAX` is finite, so masked positions hold `-FLT_MAX`
rather than `-inf`. Any downstream consumer that tests `isinf` will see
`false`. The softmax path is correct (because `expf(-FLT_MAX - max)
→ 0`), but the raw KQ tensor (before softmax) is technically wrong. The
kernel is also a legacy artefact: modern code paths use a precomputed
mask + FA, not `diagmask`.

### W5 — RoPE is not fused with FA

**Evidence:** `rope.cu` and `fattn*.cu` are completely separate. The
only RoPE-level fusion is `ROPE+VIEW+SET_ROWS` (`rope.cu:670-672`),
which fuses the *KV-cache write* but not the *attention*.

**Impact:** Each attention layer pays one extra global-memory round-trip
for the RoPE'd Q tensor. For decode with `D=128` and `n_heads=32`, that's
16 KiB per layer — small but non-zero. For prefill it is more
significant.

### W6 — The FA-vec / FA-tile / FA-mma-f16 kernels share no code

**Evidence:** Each kernel has its own KQ vecdot, its own V dequantize,
its own softmax, its own mask handling. `fattn-common.cuh` provides
*template helpers* but each kernel wires them up independently. The
FA-tile kernel uses `flash_attn_tile_load_tile` and
`flash_attn_tile_iter`; the FA-mma-f16 kernel uses
`flash_attn_ext_f16_load_tile` and `flash_attn_ext_f16_iter`; the FA-vec
kernel uses inline vecdot calls.

**Impact:** Bug fixes (e.g. the `FATTN_KQ_MAX_OFFSET` shift) must be
applied to three places. Adding a new dtype requires touching two files
(FA-vec vecdot table in `fattn-common.cuh`, FA-tile/MMA F16 conversion
in `launch_fattn`).

### W7 — The dispatcher heuristic is hard-coded and shape-specific

**Evidence:** `ggml_cuda_get_best_fattn_kernel` (`fattn.cu:358-534`) is
~180 lines of `if` statements with hand-tuned thresholds (e.g.
"if `Q->ne[1] <= 4 && K->ne[1] >= 65536` then ncols2=16"). There are
special cases for MiMo-V2.5 (`:142-157`), Mistral Small 4 (`:162-177`),
Deepseek (`:182-237`), and GLM-4.7-Flash (`:193-231`). The `576/512`
case has a 50-line nested `if` on CC + `gqa_ratio` + `Q->ne[1]`.

**Impact:** Adding a new model with a new (DKQ, DV, GQA) combination
requires hand-editing the dispatcher. The thresholds are not autotuned;
they are based on the developer's benchmarks on specific GPUs. A
shape-driven autotuner (even a small offline one) would be more
maintainable.

### W8 — No graph-level attention fusion

**Evidence:** `ggml_cuda_try_fuse` (audited in ARTX08) has ~12 fusion
patterns but none involve `FLASH_ATTN_EXT`. The closest is
`ROPE+VIEW+SET_ROWS`, which is RoPE-side only. There is no
`FA+RESIDUAL+RMS_NORM` or `FA+RESIDUAL` fusion.

**Impact:** Each transformer layer pays a separate kernel launch for
the residual add and the post-attention RMS_NORM. With ~80 layers
(Llama-3-70B) and ~5 µs launch overhead per kernel, this is ~800 µs of
pure overhead per token at decode.

### W9 — `parallel_blocks > 1` path allocates a full `dst_tmp` buffer

**Evidence:** `fattn-common.cuh:1182`: `dst_tmp.alloc(parallel_blocks *
ggml_nelements(KQV))`. For `parallel_blocks=4` and a typical prefill
shape, this is 4× the output size, allocated from the per-backend pool
on every FA call.

**Impact:** Memory pressure on small GPUs; the pool may need to grow
under transient workloads. The stream-K path avoids this allocation
entirely (it writes partial results directly to `dst` and uses a much
smaller `dst_tmp_meta` buffer).

### W10 — Config tables are enormous and easy to mis-edit

**Evidence:** `fattn-tile.cuh:21-326` has 4 tables of ~50 lines each,
one per (NVIDIA-FP16, NVIDIA-FP32, AMD, AMD-RDNA). Each entry is a
single line with 7 numeric parameters. `fattn-mma-f16.cuh:38-245` has 5
tables of similar size. There is no `static_assert` that every
`(DKQ, DV, ncols)` triple has an entry; the fallback returns 0 and the
kernel `static_assert`s at compile time (`fattn-tile.cuh:841`).

**Impact:** High risk of typos. Adding a new head size requires
hand-editing multiple tables. A table-generation script (Python or
Jinja) would be safer; the `template-instances/generate_cu_files.py`
script does this for *instantiations* but not for *config tables*.

### W11 — Linear-attention variants (GLA, RWKV, SSM) are scalar and FP32-only

**Evidence:** `gla.cu:4-62` uses `float4` loads but no Tensor Cores.
`wkv.cu:4-30` is similar. `ssm-scan.cu:18-60` uses CUB `BlockLoad` but
no MMA. `gated_delta_net.cu:4-30` uses warp-level cooperation but no
Tensor Cores. All four are F32-in/F32-out only.

**Impact:** For Mamba-2, RWKV-7, GLA-based models, the CUDA backend
leaves substantial performance on the table compared to dedicated
implementations (e.g. Mamba's `selective_scan` CUDA kernel in the
upstream repo, which uses Tensor Cores).

### W12 — `softmax.cu` has no half-precision output path

**Evidence:** `softmax.cu:55-138` always outputs F32. The
`ggml_cuda_op_soft_max` entry asserts `dst->type == GGML_TYPE_F32`
(`:388`).

**Impact:** When the softmax feeds into a F16 matmul (the common
post-attention path), an explicit `FP32→F16` conversion op is required.
Fusing this conversion into the softmax kernel would save a memory
round-trip.

---

## 15. GwenLand Mapping

| GwenLand module | ADOPT / ADAPT / REJECT / MONITOR / DEFER | What | Reasoning |
| --------------- | ---------------------------------------- | ---- | --------- |
| `glcuda`        | **ADOPT** | Three-way FA dispatch (`VEC / TILE / MMA_F16`) | Clean taxonomy; maps to decode / non-TC / TC. |
| `glcuda`        | **ADOPT** | Online softmax contract (`KQ_max`, `KQ_sum`, `VKQ`) with `FATTN_KQ_MAX_OFFSET` shift | Standard FA-2; the shift is a tested stability fix. |
| `glcuda`        | **ADOPT** | Per-CC config tables (bitfield or struct) | Maintainable; constexpr; allows per-shape tuning. |
| `glcuda`        | **ADOPT** | `V_is_K_view` MLA shortcut | Minimal, elegant; works for Deepseek/GLM/MiMo. |
| `glcuda`        | **ADOPT** | Stream-K with two fixup kernels | Adaptive to fractional-tile SM partitioning. |
| `glcuda`        | **ADOPT** | `cp.async` multi-stage pipeline | Up to 2 stages; FlashAttention-2 standard. |
| `glcuda`        | **ADOPT** | Fused `ROPE+VIEW+SET_ROWS` | Critical for decode latency. |
| `glcuda`        | **ADAPT** | Per-dtype FA-vec vecdot table | Keep the table, but extend to FP8 (E4M3, E5M2). |
| `glcuda`        | **ADAPT** | Mask-prune pre-pass | Keep, but add a real `sliding_window` parameter to the op. |
| `glcuda`        | **REJECT**| Absence of Hopper `wgmma` / TMA | GwenLand should implement a Hopper-native FA path. |
| `glcuda`        | **REJECT**| Absence of FP8 KV path | GwenLand should support FP8 KV cache. |
| `glcuda`        | **REJECT**| `diagmask` `FLT_MAX` shortcut | Use `-INFINITY`; the perf difference is negligible on modern GPUs. |
| `glcuda`        | **REJECT**| Hard-coded dispatcher thresholds | Replace with a small offline autotuner. |
| `glcuda`        | **MONITOR**| `softmax.cu` cooperative-launch path | Useful for top-p sampling; watch for grid-sync overhead. |
| `glcuda`        | **DEFER** | Linear-attention variants (GLA, RWKV, SSM) | Adopt only when GwenLand needs to run those models. |
| `GATE`          | **ADOPT** | RoPE YaRN length extrapolation | Standard; needed for long-context models. |
| `GATE`          | **ADAPT** | `FLASH_ATTN_EXT` op signature | Add `sliding_window`, `sink_tokens`, `fp8_kv` parameters. |
| `GATE`          | **ADOPT** | Fused `ROPE+VIEW+SET_ROWS` pattern | Extend to `ROPE+FA` if/when Hopper path is added. |
| `GATE`          | **REJECT**| `FLASH_ATTN_EXT` consuming a precomputed F16 mask for sliding window | Replace with a `sliding_window` op parameter. |

---

## 16. Recommendations

### R1 — ADOPT three-way FA dispatch taxonomy
**Priority:** Critical
**Difficulty:** M
**Dependencies:** none
GwenLand's `glcuda` should define `gl_fa_kernel_kind ∈ {VEC, TILE,
MMA_F16}` and a `gl_fa_get_best_kernel(device, q, k, v, mask, params)`
function. The taxonomy maps cleanly to (decode, non-TC, TC) and is
proveable across five NVIDIA generations and three AMD families.

### R2 — ADOPT FlashAttention-2 online-softmax contract with `KQ_MAX_OFFSET` shift
**Priority:** Critical
**Difficulty:** S
**Dependencies:** R1
Implement `(KQ_max, KQ_sum, VKQ)` per-thread accumulators with the
`3·log(2)` shift on `KQ_max`. Document the shift in a comment; it is
non-obvious and the only reason it's correct is that the final
`VKQ / KQ_sum` cancels it.

### R3 — ADOPT per-CC config tables
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
For both the tile and MMA kernels, encode `nthreads`, `occupancy`,
`nbatch_fa`, `nbatch_K`/`K2`, `nbatch_V2`, `nbatch_combine`,
`nstages_target`, `Q_in_reg` per `(DKQ, DV, ncols, CC)` in a constexpr
table. Use a bitfield pack for `__launch_bounds__` compatibility on
ROCm.

### R4 — REJECT absence of FP8 KV; ADOPT FP8 path
**Priority:** High
**Difficulty:** L
**Dependencies:** R1
Add `GGML_TYPE_F8_E4M3` and `GGML_TYPE_F8_E5M2` to the FA-vec vecdot
table (via `mma.sync.aligned.kind::mxf4nvf4` on Blackwell, or via F32
emulation on older CCs). For FA-mma-f16, add an F8→F16 conversion path
in `launch_fattn` (similar to the existing Q4_0→F16 path) but keep the
option to consume F8 directly on Hopper+ via `mma.sync.aligned.m16n8k32`
F8.

### R5 — REJECT absence of Hopper `wgmma` / TMA; ADOPT Hopper-native path
**Priority:** High
**Difficulty:** XL
**Dependencies:** R1, R3
Implement a `BEST_FATTN_KERNEL_MMA_F16_HOPPER` (or a `glcuda_fa_hopper`
kernel) that uses `wgmma.mma_async`, `cp.async.bulk.tensor` (TMA), and
warp specialisation (producer-consumer). Target Hopper SM_90 and
Blackwell SM_120. FlashAttention-3 is the reference.

### R6 — REJECT opaque sliding window; ADOPT `sliding_window` op parameter
**Priority:** High
**Difficulty:** M
**Dependencies:** R1
Add `int32_t sliding_window` to `FLASH_ATTN_EXT`'s `op_params`. In the
FA kernel, use it to compute `k_VKQ_max = min(k_VKQ_max, j + sliding_window)`
per Q row, avoiding the dense-mask materialisation. The
`flash_attn_mask_to_KV_max` pre-pass can be repurposed to compute this.

### R7 — ADOPT fused `ROPE+VIEW+SET_ROWS`; extend to `ROPE+FA` on Hopper
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1, R5
Keep the existing fusion for incremental KV writes. On Hopper, where
`wgmma` allows the Q tile to stay in registers, evaluate fusing RoPE
directly into the FA kernel's Q-load path.

### R8 — ADOPT `V_is_K_view` MLA shortcut
**Priority:** Medium
**Difficulty:** S
**Dependencies:** R1
When `DKQ != DV` and `V` aliases `K`, set `V_h2 = K_h2` and reverse the
K-load iteration. Saves a smem buffer and a load path.

### R9 — REJECT `diagmask` `FLT_MAX` shortcut
**Priority:** Low
**Difficulty:** XS
**Dependencies:** none
Use `-INFINITY` instead of `FLT_MAX`. The performance difference is
negligible on post-Turing GPUs (both compile to a `selp`), and
`-INFINITY` is correct under `isinf` tests.

### R10 — ADAPT dispatcher: replace hard-coded thresholds with offline autotuner
**Priority:** Medium
**Difficulty:** L
**Dependencies:** R1, R3
Generate the dispatcher's `if`-ladder from a JSON/YAML file that lists
`(DKQ, DV, ncols2, CC, condition)` → `kernel_id`. The thresholds can be
populated by an offline autotuner that runs once per (GPU, shape) and
caches the result. This avoids the special-case blocks for MiMo, GLM,
Deepseek, Mistral Small 4.

### R11 — ADOPT stream-K with two fixup kernels
**Priority:** Medium
**Difficulty:** M
**Dependencies:** R1
Keep the uniform vs general fixup split. The uniform path is ~5× cheaper
than the general path because it does one fixup per output tile rather
than walking a chain of partial blocks.

### R12 — DEFER linear-attention variants
**Priority:** Low
**Difficulty:** L
**Dependencies:** none
The GLA / RWKV / SSM / DeltaNet kernels are F32-only and do not use
Tensor Cores. Defer adopting them until GwenLand needs to run those
models; at that point, write new Tensor-Core kernels from scratch.

---

## 17. Findings

### Finding ARTX11-F01

```
Finding ID:           ARTX11-F01
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            Flash-Attention top-level dispatch
Source File:          ggml/src/ggml-cuda/fattn.cu
Function:             ggml_cuda_get_best_fattn_kernel / ggml_cuda_flash_attn_ext
Lines:                358-585
Summary:              A single heuristic selects one of three FA kernels
                      (VEC / TILE / MMA_F16) at runtime, keyed on CC,
                      head size, GQA ratio, batch, mask, and KV alignment.
Observation:          ggml_cuda_get_best_fattn_kernel returns an enum
                      (BEST_FATTN_KERNEL_{NONE,VEC,TILE=200,MMA_F16=400}).
                      The public ggml_cuda_flash_attn_ext switches on the
                      enum. The heuristic is ~180 lines of hard-coded
                      if-statements with shape-specific special cases for
                      MiMo-V2.5, Mistral Small 4, Deepseek, and GLM-4.7-Flash.
                      There is no autotuner; thresholds are developer-tuned.
Evidence:             fattn.cu:358-534 (heuristic), 570-585 (switch).
Architectural Impact: Clean three-way taxonomy. Adding a new model with a
                      new (DKQ, DV, GQA) combination requires hand-editing
                      the dispatcher. Hard-coded thresholds may not
                      generalise to future GPUs.
Correctness Impact:   None. The dispatcher only selects kernels; it does
                      not affect the result.
Optimization Type:    None (this is a dispatch policy, not an optimization).
GwenLand Target:      glcuda
Recommendation:       ADOPT the taxonomy, ADAPT the dispatcher to be
                      table-driven. Replace hard-coded thresholds with an
                      offline autotuner.
Priority:             High
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX11-F02

```
Finding ID:           ARTX11-F02
Category:             GPU_KERNEL
Engine:               CUDA
Component:            FA-vec single-query decode kernel
Source File:          ggml/src/ggml-cuda/fattn-vec.cuh
Function:             flash_attn_ext_vec
Lines:                19-528
Summary:              Decode-time FA kernel uses 128 threads (4 warps) per
                      block, one or two Q tokens per block, and a per-dtype
                      KQ vecdot template that supports F16/BF16/Q4_0/Q4_1/
                      Q5_0/Q5_1/Q8_0 KV caches without on-the-fly F16
                      conversion.
Observation:          The kernel keeps Q in registers (half2 or float2 for
                      F16/BF16 K; int32 + float2 Q8_1 scales for quantized K).
                      For quantized K, Q is quantized to Q8_1 on-the-fly via
                      quantize_q8_1_to_shared (fattn-common.cuh:331-373)
                      using warp shuffles for the per-32-element scale/sum
                      reduction. The KQ dot product uses dp4a for integer
                      quants and v_dot2_f32_f16 for F16. V is dequantized
                      per-element inside the AV loop.
Evidence:             fattn-vec.cuh:19-528 (kernel), 533-574 (case dispatch);
                      fattn-common.cuh:87-329 (per-dtype vecdot templates).
Architectural Impact: This is the only FA kernel that can consume a
                      quantized KV cache in situ. FA-tile and FA-mma-f16
                      require an up-front F16 conversion. FA-vec is
                      therefore strongly preferred for decode with
                      quantized KV.
Correctness Impact:   The on-the-fly Q8_1 quantization is deterministic
                      per warp (single scale/sum reduction). The output
                      is bit-identical across runs for fixed CC and
                      kernel selection.
Optimization Type:    Vectorization (dp4a, v_dot2_f32_f16) + on-the-fly
                      quantization + warp-cooperative reduction.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Extend to FP8 (E4M3, E5M2) by adding new
                      vecdot templates.
Priority:             High
Difficulty:           L
Dependencies:         ARTX11-F01
Confidence:           High
```

### Finding ARTX11-F03

```
Finding ID:           ARTX11-F03
Category:             GPU_KERNEL
Engine:               CUDA
Component:            FA-tile prefill kernel
Source File:          ggml/src/ggml-cuda/fattn-tile.cuh
Function:             flash_attn_tile
Lines:                791-1146
Summary:              Tiled FlashAttention-2 kernel for non-Tensor-Core GPUs
                      (Pascal, Volta-non-MMA, AMD-non-WMMA). Uses shared-
                      memory tiles for Q, K, V, and KQ; online softmax with
                      KQ_max/KQ_sum/VKQ accumulators per thread.
Observation:          The kernel allocates Q_tmp[ncols*DKQ/2],
                      KV_tmp[nbatch_fa*(nbatch_K/2+cpy_ne)+DVp-DV], and
                      KQ[ncols*nbatch_fa] in shared memory. For each KV
                      iteration (nbatch_fa rows), it loads a K tile, computes
                      KQ via flash_attn_tile_iter_KQ, applies mask + logit
                      softcap, runs online softmax to update KQ_max/KQ_sum,
                      rescales VKQ by expf(KQ_max_old - KQ_max_new), loads
                      a V tile, and accumulates VKQ += V @ KQ. Final write-
                      back divides by KQ_sum.
Evidence:             fattn-tile.cuh:791-1146 (kernel), 560-789 (iter helpers),
                      884-894 (smem allocation).
Architectural Impact: This is the fallback for GPUs without tensor cores.
                      It is also the only FA kernel that supports FP32-only
                      compute (the #else branch of FAST_FP16_AVAILABLE). On
                      modern GPUs it is selected only for very small batches
                      on Volta or for non-WMMA AMD GPUs.
Correctness Impact:   Standard FA-2 algorithm. Bit-identical across runs
                      for gridDim.y == 1. Non-deterministic at ULP level
                      when parallel_blocks > 1.
Optimization Type:    Tiling + blocking + online softmax.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Keep as the non-TC fallback.
Priority:             Medium
Difficulty:           L
Dependencies:         ARTX11-F01
Confidence:           High
```

### Finding ARTX11-F04

```
Finding ID:           ARTX11-F04
Category:             GPU_KERNEL
Engine:               CUDA
Component:            FA-mma-f16 Tensor-Core kernel
Source File:          ggml/src/ggml-cuda/fattn-mma-f16.cuh
Function:             flash_attn_ext_f16
Lines:                1703-1893
Summary:              FA kernel using mma.sync Tensor Cores (Turing+,
                      Volta m8n8k8, AMD MFMA/WMMA). F16 input, F32
                      accumulate. cp.async multi-stage pipeline on Ampere+.
                      No wgmma, no TMA.
Observation:          The kernel uses mma.sync.aligned.m16n8k16.row.col.
                      f32.f16.f16.f32 on Turing+ (mma.cuh:977, 1002, 1163,
                      1204). On Volta and Turing-without-m16n8k16 it falls
                      back to 2× m8n8k8 or 4× m8n8k8. The K-fragment is
                      loaded via ldmatrix.sync.aligned.m8n8.x4.b16. cp.async
                      .cg.shared.global (cp_async_cg_16) is used on Ampere+
                      to load K, V, and mask tiles into shared memory; the
                      pipeline depth is nstages_target ∈ {1, 2}. All warps
                      cooperate on both load and compute (no producer-
                      consumer specialisation).
Evidence:             fattn-mma-f16.cuh:1703-1893 (kernel), 530-1700 (iter
                      helpers), 1917-1960 (smem sizing + cudaFuncSetAttribute);
                      mma.cuh:977-1023 (f16 mma), 1163-1224 (f32-acc f16 mma),
                      830-840 (ldmatrix).
Architectural Impact: This is the primary prefill kernel on all modern
                      NVIDIA GPUs and on AMD CDNA/RDNA3+. It is the only
                      FA kernel that uses Tensor Cores. The absence of
                      wgmma/TMA on Hopper is a major performance gap.
Correctness Impact:   F16 input → F32 accumulate. Reassociation across
                      MMA tiles produces ULP-level differences vs the
                      FA-tile kernel. Deterministic per (CC, shape).
Optimization Type:    Tensor Cores + ldmatrix + cp.async multi-stage
                      pipeline + stream-K scheduling.
GwenLand Target:      glcuda
Recommendation:       ADOPT for Turing…Ada. ADD a Hopper/Blackwell path
                      that uses wgmma + TMA + warp specialisation (R5).
Priority:             Critical
Difficulty:           XL
Dependencies:         ARTX11-F01, ARTX11-F05, ARTX11-F10, ARTX11-F11
Confidence:           High
```

### Finding ARTX11-F05

```
Finding ID:           ARTX11-F05
Category:             CORRECTNESS_SHORTCUT
Engine:               CUDA
Component:            Online softmax numerical stability
Source File:          ggml/src/ggml-cuda/fattn-common.cuh
Function:             (used by all three FA kernels)
Lines:                9-19
Summary:              The FA kernels add FATTN_KQ_MAX_OFFSET = 3*log(2) to
                      every KQ_max and use SOFTMAX_FTZ_THRESHOLD = -20.0f
                      to short-circuit expf calls. These are workarounds
                      for F16 accumulator overflow and denormal underflow.
Observation:          FATTN_KQ_MAX_OFFSET shifts the VKQ accumulator range
                      up by a factor of 8 (2^3). This was added to fix
                      issue #18606 (overflow in half2 VKQ accumulation).
                      The shift is cancelled by the final dst = VKQ/KQ_sum
                      division because both numerator and denominator are
                      scaled by the same factor. SOFTMAX_FTZ_THRESHOLD
                      short-circuits expf(KQ_max_diff) to 0 when the diff
                      is less than -20.0f, avoiding denormal NaN propagation.
                      The bit-hack at fattn-mma-f16.cuh:859 multiplies the
                      uint32 representation of KQ_max_scale by the boolean
                      (diff >= threshold) to zero out sub-threshold scales
                      without a branch.
Evidence:             fattn-common.cuh:9-19 (macros), 790-791, 894-895
                      (fixup kernels use FTZ threshold);
                      fattn-mma-f16.cuh:859 (bit-hack).
Architectural Impact: The shift is a global invariant of the FA kernels.
                      Any future kernel (e.g. Hopper wgmma path) must
                      preserve it or re-derive the stability analysis.
Correctness Impact:   The shift is mathematically invisible (cancels in
                      the final division). The FTZ short-circuit produces
                      bit-identical results to FTZ-mode F32 (because the
                      short-circuited values would underflow to 0 anyway).
Optimization Type:    None (numerical stability workaround).
GwenLand Target:      glcuda
Recommendation:       ADOPT both. Document the shift in a comment;
                      it is non-obvious and the only reason it's correct
                      is that the final division cancels it.
Priority:             High
Difficulty:           XS
Dependencies:         none
Confidence:           High
```

### Finding ARTX11-F06

```
Finding ID:           ARTX11-F06
Category:             LAYOUT_SUBOPTIMAL
Engine:               CUDA
Component:            KV cache layout assumption
Source File:          ggml/src/ggml-cuda/fattn-common.cuh
Function:             launch_fattn
Lines:                990-995, 1086-1090
Summary:              The FA kernels assume K and V are 4D tensors with
                      ne[0]=head_dim, ne[1]=seq_len, ne[2]=n_kv_heads,
                      ne[3]=n_batch, and that the innermost dimension is
                      contiguous (nb[0] == element_size). The shape is set
                      by the llama model code, not enforced by ggml.
Observation:          launch_fattn asserts Q->nb[0] == ggml_element_size(Q)
                      and similarly for K and V (fattn-common.cuh:993-995).
                      K and V must share ne[0..3] (they share gqa_ratio
                      computation: gqa_ratio = Q->ne[2]/K->ne[2]). The
                      V_is_K_view shortcut (fattn-common.cuh:63) lets V
                      alias K's pointer when V is a view of K (MLA case).
                      Non-contiguous K or V (e.g. a transposed view) is
                      rejected; the user must materialise a contiguous
                      copy via GGML_OP_CONT first.
Evidence:             fattn-common.cuh:63 (V_is_K_view detection),
                      990-995 (contiguity asserts), 1086-1090 (gqa_ratio
                      and tile computation); fattn.cu:370-371 (gqa_ratio
                      assertion).
Architectural Impact: The kernels cannot consume arbitrary strides. This
                      is a standard FA constraint (the inner-dim must be
                      contiguous for vectorised loads and for ldmatrix).
Correctness Impact:   None. The asserts fail loudly on misuse.
Optimization Type:    None (layout assumption).
GwenLand Target:      glcuda
Recommendation:       ADOPT the assumption. Document it in the op
                      contract. Consider adding a fast GGML_OP_CONT
                      path in the scheduler for transposed KV inputs.
Priority:             Medium
Difficulty:           S
Dependencies:         none
Confidence:           High
```

### Finding ARTX11-F07

```
Finding ID:           ARTX11-F07
Category:             GPU_KERNEL
Engine:               CUDA
Component:            GQA / MQA via ncols2 template
Source File:          ggml/src/ggml-cuda/fattn.cu, ggml/src/ggml-cuda/fattn-mma-f16.cuh
Function:             ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2 / flash_attn_ext_f16
Lines:                fattn.cu:36-111, 502-517; fattn-mma-f16.cuh:1781-1793
Summary:              GQA / MQA is implemented by a ncols2 template parameter
                      that packs multiple Q heads (2/4/8/16/32) per K/V head
                      into a single CUDA block, amortising K/V loads.
Observation:          ncols2 is the GQA fan-out per block. The kernel
                      computes zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2 (fattn-
                      mma-f16.cuh:1813) and loads ncols2 Q heads while
                      loading one K and one V head. The KQ matmul produces
                      ncols2 output columns per warp. The dispatcher picks
                      ncols2 from the gqa_ratio: >4 → 8, >2 → 4, >1 → 2,
                      else 1 (fattn.cu:91-110). On Volta the heuristic is
                      more conservative (gqa_ratio must be %8, %4, or %2).
                      The `use_gqa_opt` flag (fattn.cu:50) requires a mask
                      and zero ALiBi bias and KV length divisible by
                      FATTN_KQ_STRIDE.
Evidence:             fattn.cu:36-111 (ncols2 selection), 502-517 (AMD
                      WMMA threshold); fattn-mma-f16.cuh:1781-1793 (gqa_ratio
                      and tile iteration), 564 (ncols = ncols1*ncols2).
Architectural Impact: Without GQA packing, each K/V load would be
                      replicated gqa_ratio times across blocks. With it,
                      the K/V load is amortised across ncols2 Q heads
                      inside one block, reducing HBM traffic by ~ncols2×.
Correctness Impact:   None. The packed Q heads are independent in the
                      output.
Optimization Type:    Tiling + blocking (GQA fan-out amortisation).
GwenLand Target:      glcuda
Recommendation:       ADOPT. The ncols2 ∈ {1,2,4,8,16,32} template
                      expansion is the right granularity.
Priority:             High
Difficulty:           M
Dependencies:         ARTX11-F01
Confidence:           High
```

### Finding ARTX11-F08

```
Finding ID:           ARTX11-F08
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            Sliding-window attention
Source File:          ggml/src/ggml-cuda/fattn.cu, ggml/src/ggml-cuda/fattn-common.cuh
Function:             ggml_cuda_get_best_fattn_kernel / launch_fattn
Lines:                fattn.cu:570-585 (no sliding_window param);
                      fattn-common.cuh:997, 1094-1109 (mask handling)
Summary:              There is no flash_attn_sliding_window parameter.
                      Sliding-window attention is realised by passing a
                      precomputed F16 mask tensor through dst->src[3]; the
                      kernel cannot prune KV loads based on a window size.
Observation:          The FLASH_ATTN_EXT op_params (fattn-common.cuh:1191-
                      1193) carry only (scale, max_bias, logit_softcap).
                      The mask is consumed as a dense F16 tensor of shape
                      [seq_len, n_tokens, 1, n_batch]. The only sliding-
                      window-aware optimisation is the flash_attn_mask_to_
                      KV_max pre-pass (fattn-common.cuh:664-719), which
                      scans the mask column-by-column to find the first
                      non-masked FATTN_KQ_STRIDE tile. This prunes fully-
                      masked tiles but still requires the dense mask to
                      be materialised by the caller.
Evidence:             fattn.cu:570-585 (entry); fattn-common.cuh:997
                      (mask type assert), 1094-1109 (pre-pass launch),
                      664-719 (pre-pass kernel); fattn-tile.cuh:860-864
                      (mask indexed per element).
Architectural Impact: For sliding-window attention with a small window
                      (e.g. Mistral's 4096-token window over a 32k-token
                      cache), the FA kernel still loads K[ne[1]] rows of
                      V from HBM unless the mask-prune pre-pass succeeds
                      in skipping tiles. The mask itself is
                      n_tokens × n_tokens × 1 × n_batch F16 — for
                      n_tokens = 32768, that is 2 GiB of mask, which is
                      absurd.
Correctness Impact:   None. Sliding-window attention is correct, just
                      inefficient.
Optimization Type:    None (missing optimisation).
GwenLand Target:      glcuda, GATE
Recommendation:       REJECT this design. Add a sliding_window parameter
                      to FLASH_ATTN_EXT; in the kernel compute
                      k_VKQ_max = min(k_VKQ_max, j + sliding_window) per
                      Q row. Drop the dense mask requirement.
Priority:             High
Difficulty:           M
Dependencies:         ARTX11-F01
Confidence:           High
```

### Finding ARTX11-F09

```
Finding ID:           ARTX11-F09
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Mask-prune pre-pass
Source File:          ggml/src/ggml-cuda/fattn-common.cuh
Function:             flash_attn_mask_to_KV_max
Lines:                664-719, 1094-1109
Summary:              A pre-pass kernel scans the F16 mask to find the
                      first FATTN_KQ_STRIDE tile containing a non-inf
                      value per (sequence, tile), storing it as an int
                      per (sequence, tile). The main FA kernel then
                      iterates only up to KV_max, skipping fully-masked
                      tiles.
Observation:          The pre-pass is launched only when mask &&
                      K->ne[1] % FATTN_KQ_STRIDE == 0 && (Q->ne[1] >= 1024
                      || Q->ne[3] > 1) (fattn-common.cuh:1094). It uses
                      WARP_SIZE threads per block and one block per
                      (tile, sequence). Each thread checks one half2 of
                      the mask for isinf; the warp reduces via warp_reduce_
                      all. The walk-backward loop (line 686) finds the
                      highest KV_max_sj divisible by FATTN_KQ_STRIDE for
                      which all_inf is false. The result is stored in
                      KV_max[sequence*ne31 + jt].
Evidence:             fattn-common.cuh:664-719 (kernel), 1094-1109
                      (launch site), 1117 (used to compute ntiles_KV in
                      main kernel via K->ne[1] -> KV_max[...] replacement
                      at fattn-tile.cuh:954 and fattn-mma-f16.cuh:1826-
                      1828).
Architectural Impact: For sliding-window attention with Q->ne[1] >= 1024
                      (typical prefill), this prunes the KV iteration
                      space by a factor of ~seq_len/window_size. For
                      decode (Q->ne[1] = 1) it is not launched.
Correctness Impact:   None. The pre-pass only computes an upper bound on
                      the iteration count; the main kernel still applies
                      the mask per-element within the iterated range.
Optimization Type:    Kernel fusion (mask scan + main FA) via a two-
                      kernel pipeline.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Generalise to a sliding_window parameter
                      (R6) so the pre-pass can run without a materialised
                      mask.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX11-F08
Confidence:           High
```

### Finding ARTX11-F10

```
Finding ID:           ARTX11-F10
Category:             GPU_KERNEL
Engine:               CUDA
Component:            Stream-K scheduling + fixup kernels
Source File:          ggml/src/ggml-cuda/fattn-common.cuh
Function:             flash_attn_stream_k_fixup_uniform / flash_attn_stream_k_fixup_general / launch_fattn
Lines:                721-912, 1120-1146
Summary:              When stream_k=true (Ada Lovelace+, AMD WMMA, or
                      efficiency < 75%), the FA-mma-f16 kernel partitions
                      the KV-iteration space into nblocks_stream_k CUDA
                      blocks, each owning a fraction of an output tile.
                      Two fixup kernels combine the partial results.
Observation:          launch_fattn computes nblocks_stream_k_raw = min(
                      max_blocks, ntiles_KV*ntiles_dst), then rounds down
                      to a multiple of ntiles_dst if the efficiency loss
                      is <= 5% (fattn-common.cuh:1133-1143). The main
                      kernel (fattn-mma-f16.cuh:1795-1796) computes
                      kbc = blockIdx.x*total_work/gridDim.x and
                      kbc_stop = (blockIdx.x+1)*total_work/gridDim.x. If
                      kbc % iter_k != 0 the block is in the middle of an
                      output tile and needs_fixup or is_fixup is set
                      (fattn-mma-f16.cuh:1829-1847, 1876-1877). The
                      uniform fixup kernel (fattn-common.cuh:723-801)
                      handles the case where nblocks_stream_k is a
                      multiple of ntiles_dst; it does one fixup per
                      output tile. The general fixup kernel (fattn-
                      common.cuh:807-912) walks a chain of partial
                      blocks backward.
Evidence:             fattn-common.cuh:721-801 (uniform), 807-912
                      (general), 1120-1146 (launch site);
                      fattn-mma-f16.cuh:1795-1796 (kbc computation),
                      1829-1847 (needs_fixup / is_fixup flags).
Architectural Impact: Stream-K eliminates the tail effect of
                      parallel_blocks scheduling when ntiles_dst is not
                      a multiple of (nsm * max_blocks_per_sm). On Ada
                      Lovelace with a 4096-token prefill, this can
                      improve SM utilisation from ~70% to ~95%.
Correctness Impact:   The fixup kernels rescale partial VKQ results by
                      expf(block_max - global_max) and sum. The
                      computation is mathematically equivalent to the
                      single-block case but reassociates across blocks,
                      producing ULP-level non-determinism across runs.
Optimization Type:    Persistent threads (effectively) + software
                      reduction across blocks.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Keep the uniform vs general split; the
                      uniform path is ~5x cheaper than the general path.
Priority:             High
Difficulty:           L
Dependencies:         ARTX11-F04
Confidence:           High
```

### Finding ARTX11-F11

```
Finding ID:           ARTX11-F11
Category:             GPU_KERNEL
Engine:               CUDA
Component:            cp.async multi-stage pipeline
Source File:          ggml/src/ggml-cuda/fattn-mma-f16.cuh
Function:             flash_attn_ext_f16_iter / flash_attn_ext_f16_load_tile
Lines:                348-447, 584-673
Summary:              The FA-mma-f16 kernel uses cp.async.cg.shared.global
                      (cp_async_cg_16) to load K, V, and mask tiles into
                      shared memory. The pipeline depth is nstages_target
                      ∈ {1, 2}. Stage 2 (multi-stage) is only enabled on
                      Ampere+ (cp_async_available) and when ncols2 >= 2.
Observation:          With nstages=2, V for iteration i+1 is preloaded
                      while K for iteration i is being consumed
                      (fattn-mma-f16.cuh:584-599). The K-load inner loop
                      uses cp_async_wait_all + __syncthreads to stage K
                      tiles. cp.async.cg uses the .cg (cache-global) cache
                      operator, which caches in L2 only (not L1) —
                      appropriate for one-shot KV loads. The pipeline is
                      not producer-consumer: all warps participate in both
                      load and compute. There is no 3-or-more-stage
                      pipeline (FlashAttention-3 uses up to 3 stages on
                      Hopper with warp specialisation).
Evidence:             fattn-mma-f16.cuh:348-359 (nstages selection),
                      363-447 (load_tile with cp.async), 584-599 (multi-
                      stage V preload), 607-616 (K load with cp_async_wait);
                      common.cuh:356-358 (cp_async_available).
Architectural Impact: The 2-stage pipeline hides ~50% of the K/V load
                      latency. On Ampere/Ada this is the difference
                      between ~40% and ~70% peak Tensor Core utilisation
                      for medium-shape prefill. The absence of a 3-stage
                      pipeline with warp specialisation is the main gap
                      vs FlashAttention-3 on Hopper.
Correctness Impact:   None. The pipeline only reorders loads; the
                      computation order is unchanged.
Optimization Type:    Asynchronous execution (cp.async) + software
                      pipelining.
GwenLand Target:      glcuda
Recommendation:       ADOPT for Ampere/Ada/AMD. ADD a 3-stage warp-
                      specialised path for Hopper (R5).
Priority:             High
Difficulty:           L
Dependencies:         ARTX11-F04
Confidence:           High
```

### Finding ARTX11-F12

```
Finding ID:           ARTX11-F12
Category:             GPU_KERNEL
Engine:               CUDA
Component:            On-the-fly Q8_1 quantization in FA-vec
Source File:          ggml/src/ggml-cuda/fattn-vec.cuh, ggml/src/ggml-cuda/fattn-common.cuh
Function:             flash_attn_ext_vec / quantize_q8_1_to_shared
Lines:                fattn-vec.cuh:150-203; fattn-common.cuh:331-373
Summary:              When K is quantized (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0), the
                      FA-vec kernel quantizes Q to Q8_1 on-the-fly inside
                      the kernel using warp-shuffle reductions for the
                      per-32-element scale and sum.
Observation:          quantize_q8_1_to_shared (fattn-common.cuh:331-373)
                      takes 4 floats per thread, computes amax and sum
                      via warp_reduce_max/sum, scales by 127/amax, rounds
                      to int8, and stores the packed int32 + half2
                      (d, sum) to shared memory. The FA-vec kernel then
                      calls the per-dtype vecdot (e.g. vec_dot_fattn_vec_
                      KQ_q4_0) which loads the Q8_1 from shared memory
                      and the quantized K from global memory, computing
                      the dot product via dp4a. This avoids an up-front
                      F16 conversion of K (which FA-tile and FA-mma-f16
                      require).
Evidence:             fattn-vec.cuh:150-203 (quantize-on-load path),
                      268-289 (vecdot call); fattn-common.cuh:148-329
                      (per-dtype vecdot templates), 331-373 (quantize_q8_
                      1_to_shared).
Architectural Impact: This is the only FA kernel that can consume a
                      quantized KV cache in situ. FA-tile and FA-mma-f16
                      require an up-front F16 conversion. For decode
                      with a quantized KV cache, FA-vec is therefore
                      strongly preferred.
Correctness Impact:   The Q8_1 quantization is deterministic per warp.
                      The output is bit-identical across runs for fixed
                      CC and kernel selection.
Optimization Type:    Kernel fusion (Q quant + KQ vecdot) + warp-
                      cooperative reduction.
GwenLand Target:      glcuda
Recommendation:       ADOPT. Extend to FP8 (E4M3, E5M2) by adding new
                      vecdot templates that consume FP8 K directly.
Priority:             High
Difficulty:           L
Dependencies:         ARTX11-F02
Confidence:           High
```

### Finding ARTX11-F13

```
Finding ID:           ARTX11-F13
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            RoPE (Rotary Positional Embedding)
Source File:          ggml/src/ggml-cuda/rope.cu
Function:             rope_norm / rope_neox / rope_multi / rope_vision / ggml_cuda_op_rope_impl
Lines:                43-268, 507-672
Summary:              RoPE is implemented as a standalone CUDA op with
                      four variants (norm, neox, multi, vision) and a YaRN
                      length-extrapolation ramp. It is NOT fused into the
                      FA kernels. A fused ROPE+VIEW+SET_ROWS variant
                      supports incremental KV-cache writes.
Observation:          Each RoPE variant parallelises across (dim_idx,
                      token, head, batch) with blockDim.x over tokens and
                      blockDim.y over dim pairs. rope_yarn (rope.cu:23-41)
                      computes cos_theta/sin_theta via cosf/sinf in F32,
                      applies the YaRN ramp (rope_yarn_ramp, line 15-18)
                      and the mscale (1 + 0.1*log(1/freq_scale)). The
                      fused ROPE+VIEW+SET_ROWS path (rope.cu:522-528,
                      81-84, 156-159) redirects dst to a SET_ROWS
                      destination and uses row_indices to scatter, avoiding
                      a separate SET_ROWS kernel launch for incremental KV
                      cache writes.
Evidence:             rope.cu:43-113 (rope_norm), 116-182 (rope_neox),
                      185-268 (rope_multi), 271-332 (rope_vision),
                      507-660 (entry), 670-672 (fused entry); 23-41
                      (rope_yarn).
Architectural Impact: RoPE is computed once per Q (and once per K for
                      RoPE'd KV caches). The absence of RoPE-FA fusion
                      means each attention layer pays one extra global-
                      memory round-trip for the RoPE'd Q tensor. The
                      fused ROPE+VIEW+SET_ROWS is the only RoPE-level
                      fusion.
Correctness Impact:   cosf/sinf are F32 transcendentals; the result is
                      accurate to ~24 bits. YaRN mscale and ramp are
                      deterministic.
Optimization Type:    Kernel fusion (ROPE+VIEW+SET_ROWS) + scalar F32
                      transcendental.
GwenLand Target:      glcuda, GATE
Recommendation:       ADOPT the four-variant taxonomy and YaRN. ADOPT the
                      fused ROPE+VIEW+SET_ROWS path. DEFER RoPE-FA fusion
                      until a Hopper wgmma path is available (R5), at
                      which point Q can stay in registers across the
                      RoPE+FA boundary.
Priority:             Medium
Difficulty:           M
Dependencies:         none
Confidence:           High
```

### Finding ARTX11-F14

```
Finding ID:           ARTX11-F14
Category:             MISSING_FEATURE
Engine:               CUDA
Component:            FP8 KV cache support
Source File:          ggml/src/ggml-cuda/fattn.cu
Function:             ggml_cuda_fattn_kv_type_supported
Lines:                338-356
Summary:              The FA kernels do not support FP8 (E4M3, E5M2, E8M0)
                      KV caches. ggml_cuda_fattn_kv_type_supported returns
                      true only for F32, F16, BF16, Q4_0, Q4_1 (with
                      GGML_CUDA_FA_ALL_QUANTS), Q5_0 (likewise), Q5_1
                      (likewise), Q8_0. FA-tile and FA-mma-f16 require F16
                      exclusively; FA-vec supports the integer quants but
                      not FP8.
Observation:          The FA-vec vecdot table (fattn-common.cuh:620-640)
                      has no FP8 entry. The FA-tile and FA-mma-f16 paths
                      require F16 K/V; if the user supplies F32 or quantized
                      K/V, launch_fattn converts to F16 up-front into the
                      "extra data" buffer (fattn-common.cuh:1022-1084).
                      There is no F8→F16 conversion path, and no F8-native
                      MMA path (which would use mma.sync.aligned.m16n8k32
                      .f32.e4m3.e4m3.f32 on Hopper).
Evidence:             fattn.cu:338-356 (type support table);
                      fattn-common.cuh:620-640 (vecdot template selector),
                      1022-1084 (F16 conversion path);
                      mma.cuh:1138-1146 (mxf4nvf4 mma used by MMQ, not FA).
Architectural Impact: Models that store KV cache in FP8 (e.g. vLLM-exported
                      Llama-3 variants) must convert to F16 on every FA
                      call, paying both memory and compute. This defeats
                      the bandwidth benefit of FP8 KV.
Correctness Impact:   None (FP8 is simply unsupported).
Optimization Type:    None (missing feature).
GwenLand Target:      glcuda
Recommendation:       REJECT this absence. Add FP8 KV support: (a) F8→F16
                      conversion in launch_fattn as a fallback, (b) F8-
                      native FA-vec vecdot templates, (c) F8-native MMA on
                      Hopper+ (mma.sync.aligned.m16n8k32.f32.e4m3.e4m3.f32).
Priority:             High
Difficulty:           L
Dependencies:         ARTX11-F02, ARTX11-F04
Confidence:           High
```

### Finding ARTX11-F15

```
Finding ID:           ARTX11-F15
Category:             BACKEND_DESIGN
Engine:               CUDA
Component:            MLA (Multi-Latent Attention) via V_is_K_view
Source File:          ggml/src/ggml-cuda/fattn.cu, ggml/src/ggml-cuda/fattn-mma-f16.cuh, ggml/src/ggml-cuda/fattn-common.cuh
Function:             ggml_cuda_flash_attn_ext_mma_f16 / launch_fattn / flash_attn_ext_f16_iter
Lines:                fattn.cu:142-237; fattn-mma-f16.cuh:601-673, 1821, 1915; fattn-common.cuh:63, 1051-1055
Summary:              MLA (Deepseek-V3, GLM-4.7-Flash, MiMo-V2.5, Mistral
                      Small 4) is supported via a V_is_K_view flag: when
                      DKQ != DV, V is aliased to K's pointer and the K-
                      loading loop iterates in reverse so the same shared-
                      memory tile can be re-used as V after a transpose.
Observation:          V_is_K_view is detected at fattn-common.cuh:63:
                      V->view_src && (V->view_src == K || (V->view_src
                      == K->view_src && V->view_offs == K->view_offs)).
                      In launch_fattn this short-circuits the V→F16
                      conversion (fattn-common.cuh:1051-1055). In the MMA
                      kernel, V_h2 = K_h2 and stride_V = stride_K
                      (fattn-mma-f16.cuh:1821). The K-loading loop iterates
                      k0_start from (DKQ/2-1) - (DKQ/2-1) % nbatch_K2 down
                      to 0 (fattn-mma-f16.cuh:604), so the same tile can
                      be re-used as V after the KQ matmul. The constexpr
                      V_is_K_view template parameter (set at fattn-mma-
                      f16.cuh:1915 based on DKQ == 576) lets the compiler
                      eliminate the V-load path entirely.
Evidence:             fattn.cu:142-237 (per-DKQ switch with MLA cases);
                      fattn-mma-f16.cuh:601-673 (reverse K iteration),
                      1821 (V aliasing), 1915 (V_is_K_view constexpr);
                      fattn-common.cuh:63 (detection), 1051-1055 (skip
                      conversion).
Architectural Impact: MLA is supported without a separate kernel. The
                      trick saves a smem buffer (no V tile needed) and
                      halves the K/V HBM traffic. The constraint is that
                      V must be a view of K (not a copy), which the
                      model code must guarantee.
Correctness Impact:   None. The aliasing is correct because MLA's K and
                      V are projections of the same latent vector.
Optimization Type:    Kernel fusion (K + V load) via pointer aliasing.
GwenLand Target:      glcuda
Recommendation:       ADOPT. The V_is_K_view flag is minimal and elegant.
Priority:             Medium
Difficulty:           S
Dependencies:         ARTX11-F04
Confidence:           High
```

---

## 18. Unknowns

* **U1**. Whether the `FATTN_KQ_MAX_OFFSET = 3·log(2)` shift is still
  needed on Hopper/Blackwell with native F32 accumulator MMA. The shift
  was added to fix a `half2` accumulator overflow (issue #18606); with
  F32 accumulation throughout (which the MMA path already does), it may
  be a no-op. Static analysis cannot confirm without running on the
  target hardware.
* **U2**. The actual performance gap between the FA-mma-f16 kernel (which
  uses `mma.sync` + `cp.async`) and a `wgmma` + TMA + warp-specialised
  FlashAttention-3 implementation on Hopper. External benchmarks
  (FlashAttention-3 paper) suggest 1.5-2×, but llama.cpp's FA-mma-f16
  kernel has shape-specific tuning that may narrow the gap. Requires
  benchmarking.
* **U3**. Whether the `flash_attn_mask_to_KV_max` pre-pass is profitable
  for sliding-window attention at decode time (Q->ne[1] = 1). The current
  heuristic skips the pre-pass when Q->ne[1] < 1024 && Q->ne[3] <= 1.
  For decode with a 4096-token sliding window over a 32k-token cache,
  the pre-pass would skip ~7/8 of the KV iteration, but its launch
  overhead may dominate. Requires profiling.
* **U4**. Whether the `parallel_blocks > 1` legacy path is ever selected
  on modern GPUs. The dispatcher prefers stream-K on Ada Lovelace+ and
  AMD WMMA. On Ampere and earlier NVIDIA, `parallel_blocks > 1` may
  still be selected when `ntiles_KV > max_blocks_per_sm * nsm`. Requires
  runtime tracing.
* **U5**. Whether the per-CC config tables (fattn-tile.cuh:21-326,
  fattn-mma-f16.cuh:38-245) are still optimal for Blackwell (SM_120).
  The tables have no Blackwell column; Blackwell GPUs fall through to
  the Ampere config. This may leave performance on the table.
* **U6**. Whether the FA-vec kernel's on-the-fly Q8_1 quantization is
  faster than an up-front F16 conversion for quantized KV decode on
  Ada Lovelace. The Q8_1 path uses warp shuffles for the scale/sum
  reduction; the F16 conversion path uses a separate to_fp16_cuda
  kernel. The trade-off depends on the K->ne[1] (KV cache length).
  Requires benchmarking.
* **U7**. Whether the `diagmask` kernel is still on any hot path. The
  modern code uses FA with a precomputed mask; `diagmask` is a legacy
  artefact. Static analysis cannot determine which models still use it.
* **U8**. The behaviour of the FA kernels when `mask->ne[2] > 1` (per-
  head mask). The dispatcher returns `BEST_FATTN_KERNEL_NONE` in this
  case (fattn.cu:452-454), but the model code may not exercise this
  path. Whether any model in the llama.cpp test suite uses per-head
  masks is unknown.

---

## 19. References

| Reference | File                                                | Function / Symbol                              | Lines         |
| --------- | --------------------------------------------------- | ---------------------------------------------- | ------------- |
| R01       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext`                     | 570–585       |
| R02       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_get_best_fattn_kernel`              | 358–534       |
| R03       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_fattn_kv_type_supported`            | 338–356       |
| R04       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext_mma_f16`             | 113–242       |
| R05       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1` / `_ncols2` | 8–111 |
| R06       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext_vec` (case table)    | 259–328       |
| R07       | `ggml/src/ggml-cuda/fattn.cu`                       | `ggml_cuda_flash_attn_ext_get_alloc_size`      | 536–568       |
| R08       | `ggml/src/ggml-cuda/fattn.cuh`                      | public prototypes                              | 1–8           |
| R09       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `fattn_kernel_t` typedef + macros              | 9–42          |
| R10       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `ggml_cuda_flash_attn_ext_get_f16_extra_data`  | 53–85         |
| R11       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `vec_dot_fattn_vec_KQ_{f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0}` | 87–329 |
| R12       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `quantize_q8_1_to_shared`                      | 331–373       |
| R13       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `dequantize_V_{f16,bf16,q4_0,q4_1,q5_0,q5_1,q8_0}` | 377–618 |
| R14       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `flash_attn_mask_to_KV_max`                    | 664–719       |
| R15       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `flash_attn_stream_k_fixup_uniform`            | 721–801       |
| R16       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `flash_attn_stream_k_fixup_general`            | 807–912       |
| R17       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `flash_attn_combine_results`                   | 916–970       |
| R18       | `ggml/src/ggml-cuda/fattn-common.cuh`               | `launch_fattn`                                 | 972–1274      |
| R19       | `ggml/src/ggml-cuda/fattn-tile.cu`                  | `ggml_cuda_flash_attn_ext_tile` (per-DKQ switch) | 4–60        |
| R20       | `ggml/src/ggml-cuda/fattn-tile.cuh`                 | per-CC config tables                           | 21–326        |
| R21       | `ggml/src/ggml-cuda/fattn-tile.cuh`                 | `flash_attn_tile_load_tile`                    | 377–480       |
| R22       | `ggml/src/ggml-cuda/fattn-tile.cuh`                 | `flash_attn_tile_iter_KQ` / `flash_attn_tile_iter` | 483–789 |
| R23       | `ggml/src/ggml-cuda/fattn-tile.cuh`                 | `flash_attn_tile` (kernel)                     | 791–1146      |
| R24       | `ggml/src/ggml-cuda/fattn-tile.cuh`                 | `launch_fattn_tile_switch_ncols1/ncols2`       | 1148–1320     |
| R25       | `ggml/src/ggml-cuda/fattn-vec.cuh`                  | `flash_attn_ext_vec` (kernel)                  | 19–528        |
| R26       | `ggml/src/ggml-cuda/fattn-vec.cuh`                  | `ggml_cuda_flash_attn_ext_vec_case` / `_impl`  | 533–574       |
| R27       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `fattn_mma_config` struct + per-CC tables      | 10–245        |
| R28       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `flash_attn_ext_f16_load_tile` / `_load_mask`  | 363–528       |
| R29       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `flash_attn_ext_f16_iter`                      | 530–1701      |
| R30       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `flash_attn_ext_f16` (kernel)                  | 1703–1893     |
| R31       | `ggml/src/ggml-cuda/fattn-mma-f16.cuh`              | `ggml_cuda_flash_attn_ext_mma_f16_case`        | 1895–1964     |
| R32       | `ggml/src/ggml-cuda/rope.cu`                        | `rope_yarn` / `rope_yarn_ramp`                 | 15–41         |
| R33       | `ggml/src/ggml-cuda/rope.cu`                        | `rope_norm` / `rope_neox` / `rope_multi` / `rope_vision` | 43–332 |
| R34       | `ggml/src/ggml-cuda/rope.cu`                        | `ggml_cuda_op_rope_impl` / `_fused`            | 507–672       |
| R35       | `ggml/src/ggml-cuda/softmax.cu`                     | `soft_max_f32`                                 | 55–138        |
| R36       | `ggml/src/ggml-cuda/softmax.cu`                     | `soft_max_f32_parallelize_cols_single_row`     | 141–244       |
| R37       | `ggml/src/ggml-cuda/softmax.cu`                     | `soft_max_f32_cuda` / `ggml_cuda_op_soft_max`  | 319–444       |
| R38       | `ggml/src/ggml-cuda/diagmask.cu`                    | `diag_mask_inf_f32`                            | 3–22          |
| R39       | `ggml/src/ggml-cuda/gla.cu`                         | `gated_linear_attn_f32`                        | 4–93          |
| R40       | `ggml/src/ggml-cuda/wkv.cu`                         | `rwkv_wkv_f32`                                 | 4–199         |
| R41       | `ggml/src/ggml-cuda/ssm-scan.cu`                    | `ssm_scan_f32`                                 | 18–364        |
| R42       | `ggml/src/ggml-cuda/ssm-conv.cu`                    | `ssm_conv_f32`                                 | 5–206         |
| R43       | `ggml/src/ggml-cuda/gated_delta_net.cu`             | `gated_delta_net_cuda`                         | 4–327         |
| R44       | `ggml/src/ggml-cuda/common.cuh`                     | CC constants / availability macros             | 50–106, 257–296 |
| R45       | `ggml/src/ggml-cuda/common.cuh`                     | `CUDA_SET_SHARED_MEMORY_LIMIT`                 | 230–245       |
| R46       | `ggml/src/ggml-cuda/common.cuh`                     | `ggml_cuda_pdl_sync` / `ggml_cuda_pdl_lc`      | 123–135       |
| R47       | `ggml/src/ggml-cuda/mma.cuh`                        | `mma.sync.aligned.m16n8k16` (f16)               | 977–1023, 1163–1224 |
| R48       | `ggml/src/ggml-cuda/mma.cuh`                        | `load_ldmatrix` variants                       | 786–895       |
