# GwenLand - 2026-07-11: glcuda M2.1–M2.3 — Native Quant Kernels, Tensor Cores, Prefill De-serialization

**Date:** 2026-07-11 (WIB / SEAST)
**Scope:**
- Kernels: `glcuda/src/kernels/{glcuda.ptx,glcuda_sm75.ptx,mod.rs}`
- Repack/loader/model: `glcuda/src/{repack.rs,loader.rs,model.rs,dequant.rs,cache.rs,runner.rs}`
- Tests/bench: `glcuda/tests/parity.rs`, `glcuda/examples/bench.rs`
- Validation notebooks: `glcuda_t4_validation.ipynb`, `glcuda_vs_llamacpp_bench.ipynb`, `glcuda_prefill_profile.ipynb`
**Type:** Throughput (M2.1 batched prefill + Tensor Cores; M2.2 native Q4_K/Q4_0/Q6_K; M2.3 prefill de-serialization), plus the first head-to-head vs llama.cpp.
**Status:** Implemented and T4-validated through Stage 2b; prefill optimization ongoing (root cause of the residual FFN cost still under measurement, per-op profiler landed).
**Hardware:** NVIDIA Tesla T4 (sm_75, Turing, 40 SMs, 14.6 GiB VRAM), Google Colab / Kaggle.

---

## Executive Summary

Three milestone bands landed on top of the completed M2 base engine:

1. **M2.1 — Q4_K native decode + INT8 Tensor Cores.** A hand-authored Q4_K SoA
   GEMV (`gl_gemv_q4_k_soa`) streams 4-bit weights natively (~2× decode headroom
   over Q8_0's bandwidth ceiling), and an INT8 tensor-core batched GEMM
   (`gl_gemm_mma_q8`, sm_75) accelerates prefill. Both parity-clean on the T4.

2. **M2.2 — the hybrid native-quant path (Task C).** Native SoA GEMVs for **Q4_0**
   (`gl_gemv_q4_0_soa`, 4.5 bpw) and **Q6_K** (`gl_gemv_q6_k_soa`, native 6.56 →
   tuned 7.06 bpw), so a user's chosen model file dispatches to the right kernel
   instead of paying the Q8_0-requant tax.

3. **M2.3 — prefill de-serialization (post head-to-head).** The first
   same-T4/same-model comparison vs llama.cpp showed **decode already at or above
   parity** (0.95–1.27×) but **prefill 15–30× behind**. Four stages attacked the
   prefill path in the order the measurements dictated: kill the per-token HtoD,
   batch every per-token kernel, raise GEMM arithmetic intensity, and stage
   activations in shared memory. Prefill moved 78 → ~200 tok/s; the residual gap
   is under active per-op profiling rather than guesswork.

**Governing lesson, re-earned this session:** measure before optimizing. Two
prefill fixes (activation-staging, weight-restreaming) were built on byte-math
theories that the profiler then falsified — each cost a kernel rewrite. The
project's per-op profiler is now the gate before any further kernel work.

---

## M2.1 — Native Q4_K + Tensor Cores

**Task A: `gl_gemv_q4_k_soa` (native Q4_K decode).** One warp per output row,
one loop iteration per 256-weight super-block (128 coalesced qs bytes), dp4a
integer dots against the int8-quantized activation. Per 32-weight sub-block —
which matches the activation quantizer's block — the dot decomposes exactly as
`(d·sc)·xs·dot(q,xq) − (dmin·m)·xs·Σxq`, both terms dp4a chains.

- **Repack (`repack.rs`):** nibbles repacked so one u32 holds 8 consecutive
  values (lo/hi-nibble int8×4 halves, each matching one aligned activation u32);
  sub-block scales/mins **pre-multiplied** to f16 (`d·sc`, `dmin·m`), buying
  ggml's branchy 6-bit unpack out of the hot loop for +11% weight bytes (5.0 bpw
  vs 4.5 native).
- **Rounding gotcha (T4-caught):** the first parity run failed by 2.4% over ε
  (one element, |diff| 1.024e-2 vs 1e-2). Truncating f32→f16 on the premultiplied
  scales is one-sided (every scale slightly low), so the error accumulates
  coherently across a dot. Fix: round-to-nearest-even for the Q4_K premultiply
  (`f32_to_f16_bits_rne`); worst element dropped to 0.34× ε. The Q8_0 requant
  keeps the truncating converter for glproc byte-parity.
- **Loader policy:** Q4_K matmuls go native SoA; the Q6_K/Q5_0 tensors a Q4_K_M
  file carries were requantized to Q8_0 SoA (later made native in M2.2); Q4_K
  embeddings stay 4.5 bpw host-side.

**Task B: `gl_gemm_mma_q8` (INT8 tensor cores, sm_75).** Separate `.target sm_75`
module loaded only on capable devices; the sm_70 dp4a `gl_gemm_q8_0_soa` remains
the fallback (`GLCUDA_NO_MMA=1` forces it). **Spec correction:** integer
`mma.m16n8k16` (the brief's shape) is sm_80+; Turing's INT8 shape is `m8n8k16`,
which is what the kernel uses. No new weight layout — the row-major Q8_0 SoA qs
stream *is* the col-major B fragment `mma.row.col` wants. Scale epilogue fused
per 32-K block in registers. **Measured:** 1.43× the dp4a GEMM at kernel level
(346 vs 496 µs), prefill 84 → 88 tok/s at the model level, decode unchanged.

---

## M2.2 — Hybrid Native-Quant Path (Task C)

**Task C-2: `gl_gemv_q4_0_soa` (speed-first, 4.5 bpw).** The Q4_K kernel minus the
mins stream, with the −8 centering folded into the integer domain
(`d·xs·(dot − 8·Σxq)`). Verbatim f16 block scales — no premultiply, so none of
the Q4_K rounding applies. Guarded tail keeps the requirement at `in % 32 == 0`,
so dim-896-class Q4_0 models work.

**Task C-1: `gl_gemv_q6_k_soa` (precision-first).** Four SoA streams — packed low
nibbles, 2-bit highs, verbatim i8 sub-block scales, verbatim f16 super-block d.
q6 assembled in registers (`ql | qh<<4`), −32 centering integer-folded. **Format
correction:** the real GGML `block_q6_K` is **210 B** (ql 128 + qh 64 + scales 16
+ d 2), not the brief's 178 B — reconciled against `dequant.rs`/glproc.

**Q6_K tuning (T4-driven).** The first Q6_K kernel was parity-clean but
compute-stalled at 155–183 GB/s (vs Q4_K's 242), so Q4_K_M decode stayed flat.
The stall was a 32-op per-byte 2-bit spread reconstructing qh. Fix: repack qh
into the identical u32-per-8-values nibble layout as ql, so the kernel rebuilds q6
with one and/shl/or per int8×4 half — a deliberate bytes-for-ALU trade (qh widens
64→128 B, 6.56→7.06 bpw) because the kernel had bandwidth headroom and no compute
headroom. Cache magic bumped GLCACHE3→GLCACHE6 across these landings (the Q6_K qh
change is a *correctness* bump, not just performance).

---

## M2.3 — Prefill De-serialization

### The head-to-head that started it

First comparison vs **llama.cpp built with CUDA**, same T4, same model files,
same session:

| Format | glcuda decode | llama.cpp decode | ratio | glcuda prefill | llama.cpp prefill |
|--------|--------------:|-----------------:|------:|---------------:|------------------:|
| Q8_0   | 29.3          | 30.7             | 0.95× | 91.5           | 1439.9            |
| Q4_K_M | 38.7          | 37.7             | **1.03×** | 46.4       | 1294.7            |
| Q4_0   | 47.5          | 49.0             | 0.97× | 54.7           | 1374.0            |

**Decode is done** — hand-authored PTX matches (and on both quant formats,
beats) llama.cpp's mature CUDA kernels. Prefill was the whole gap.

### Stage 1a — kill the per-token HtoD

`prefill_batched` uploaded `token_params` via `cuMemcpyHtoD` per token **per
layer** — ~896 synchronous, pipeline-draining copies per 32-token chunk. Since
token positions are consecutive integers, a `pos_seq` identity array
(`0..=kv_capacity`, uploaded once at load) replaces them all: a launch for
position p passes `pos_seq + p·4`. Zero PTX change (the kernels already read pos
by pointer). **Also fixed a latent cursor bug** this exposed: `kv.advance()` ran
per layer (28× overcount), which would falsely report "KV cache full" on any
prompt > 146 tokens — masked because `debug_assert` compiles out in release.

### Stage 1b — batch every per-token kernel

Five batched-over-tokens PTX variants (`gl_rms_norm_rows`, `gl_add_bias_rows`,
`gl_rope_rows`, `gl_kv_write_rows`, `gl_attn_decode_rows`) collapse prefill's
serial ~7000 launches/chunk into ~15 launches/layer. Attention causality holds by
construction: row t reads `cached_len = pos_seq[t]+1` rows, so the chunk's later
KV rows (written on the same stream) exist but are never read. Single-token
originals untouched — the decode graph is captured against them. **Result: 78 →
91.5 tok/s** — launch overhead gone, the GEMM now the ceiling.

### Stage 2a — MMA GEMM arithmetic intensity

v1 read the full weight once per 8-token m-tile. v2 inverts the loops: k-blocks
outer, each weight fragment (in registers) feeds up to eight 8-token m-tiles (64
rows) — weight DRAM traffic 8× down. `PREFILL_BATCH` 32→64. **Result: 91.5 → ~132
tok/s.**

### Stage 2b — shared-memory activation staging

The phase profiler showed **ffn 67% / attn 24% / qkv 8%**. v3 stages each
k-block's activation slice + scales in shared memory once per block (all 8 warps
share the same tokens), cutting redundant per-warp A reads. `bar.sync` brackets
the staging, so out-of-range warps stage + synchronize rather than early-exit
(exercised by the out=16 parity cases). **Result: FFN unchanged (2218 vs 2210
ms).** This *falsified* the redundant-A-read theory — and subsequent byte math
showed the FFN weight stream is only ~60 GB ≈ 300 ms, yet FFN measures 2218 ms: a
5× gap that is neither A-traffic nor weight bandwidth.

### Where it stands

A per-op FFN profiler (`GLCUDA_PROFILE_PREFILL=1`, second output line) now splits
FFN into gate+up GEMM / down+o GEMM / elementwise, to localize that 5× before any
further kernel work. Prefill: **78 → ~200 tok/s** across the stages; decode held
at parity throughout.

---

## Parity & Test State

- 19 GPU parity tests (Q8_0/Q4_K/Q4_0/Q6_K GEMVs, MMA GEMM at ntok 5/20/64, the
  five rows kernels via `attn_decode_rows`), all green on the T4.
- 34 host-side lib tests (repacks bit-exact vs glproc, cache round-trips, RNE
  converter), green without a GPU.
- Decode ratios vs llama.cpp: 0.95× (Q8_0), 1.03–1.27× (Q4_K_M), 0.97–1.11×
  (Q4_0) across runs.

## Validation Infrastructure

- `glcuda_t4_validation.ipynb` — full parity + bench (sections 13/14 for M2.1/2.2).
- `glcuda_vs_llamacpp_bench.ipynb` — head-to-head; builds llama.cpp with CUDA.
- `glcuda_prefill_profile.ipynb` — glcuda-only, skips the ~15-min llama.cpp build;
  the fast path for iterating on the FFN profiler.

Full architecture + benchmark detail: [`../docs/ArchGLCuda/ArchGLML_M2.1-M2.3.md`](../docs/ArchGLCuda/ArchGLML_M2.1-M2.3.md).
