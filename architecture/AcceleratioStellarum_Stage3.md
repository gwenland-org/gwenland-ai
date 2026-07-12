# Acceleratio Stellarum — Stage 3 Execution-Graph Redesign

> *"The fastest stars are not those that burn brighter, but those that travel a shorter path."*

Design record for the glcuda prefill execution-graph redesign. Kernel-level
optimization (Stages 2a–2c.1) is complete; this stage asks whether the
execution schedule itself — not any kernel — is the limiting factor, and
answers yes.

Status: **Phase A implemented** (this commit). Phase B pending T4 validation
of Phase A. All performance figures below are analytical predictions until
glbench archives say otherwise.

---

## 1. Execution graphs

**Current (chunk-first, PREFILL_BATCH-token chunks):**

```
for chunk c in ceil(S / B):
    embed chunk rows (n small HtoD copies)
    for layer l in 28:
        QKV GEMMs      <- W_qkv[l] from DRAM   \
        rope / kv-write / attention (batched)   | every layer's weights
        o-proj GEMM    <- W_o[l]                | cross DRAM once PER
        gate/up GEMMs  <- W_gu[l]               | CHUNK: W x ceil(S/B)
        down GEMM      <- W_dn[l]              /
logits (last token only)
```

**Target (layer-first, prompt resident):**

```
stage all S embeddings host-side -> ONE HtoD ([S,d])
for layer l in 28:
    QKV GEMM over ALL S rows        <- W_qkv[l] ONCE
    rope + kv-write over ALL S rows    (kernels already batched, Stage 1b)
    attention over ALL S rows          (causal per row via pos_seq - op unchanged)
    o-proj / gate/up / down GEMMs over ALL S rows  <- each weight ONCE
logits (last token only)
```

Launches per 596-token prompt: ~3,900 -> ~420. Embed HtoD: 596 small copies
-> one ~8.5 MB copy per chunk.

## 2. Why chunk-first is not mathematically necessary

The only cross-token dependency in a transformer layer is the causal
attention prefix — a dependency **within** a layer. The inter-layer
dependency is per-token. Any schedule that (a) completes layer *l*'s KV
writes for rows <= t before row t's attention at layer *l*, and (b)
completes layer *l* for a token before layer *l+1* for that token, computes
the identical function. Layer-first satisfies both; the attention kernel
already enforces causality per row (`cached_len = pos_seq[t] + 1`, later
rows written but never read). The chunk exists for one historical reason:
fixed `[B, width]` workspace slabs. It is an artifact, not mathematics.

## 3. Communication analysis — and the one honest dependency

- Weight traffic: current **Θ(W · ceil(S/B))** ≈ 64 GB at S=596, B=64,
  W≈6.9 GB. Target **Θ(W · ceil(S/N))** with resident ubatch N — Θ(W) for
  S <= N.
- **Stated dependency:** the graph reorder is the *enabler*, not the whole
  win. Ten back-to-back 64-row GEMM calls on the same 144 MB weight get no
  L2 reuse (L2 = 4 MB). Realizing Θ(W) requires the GEMM node's **contract**
  to change: "accepts all N resident rows; each weight fragment visits every
  row before eviction." The graph makes rows *available*; the contract makes
  traffic collapse. Hence the two-phase migration (§8).

## 4. Compute and arithmetic intensity

FLOPs identical — the schedule changes no MAC. T4 roofline ridge:
130 TOPS / ~300 GB/s ≈ **~430 int8-ops/byte**. Chunk-64 delivers
2×64 = 128 ops/weight-byte — structurally memory-bound; no kernel can save
it. N=512 delivers 1,024 ops/byte — past the ridge, compute-bound, the only
regime where the tensor-core path can pay for itself. Minimum ubatch to
cross the ridge ≈ 256.

## 5. Activation residency

Per-token workspace ≈ 248 KB (7B shapes, dominated by the [·,18944] FFN
hiddens): 512 tok = 127 MB, 1024 = 254 MB, 2048 = 509 MB, 4096 = 1.02 GB.
All fit the T4 (16 GB, model 7.6 GB); policy is **N = min(S, N_max)** with
N_max guarded by the existing load-time footprint check. Phase A sets
N_max = 512.

## 6. Scalability

Disappearing bottlenecks: weight re-streaming (80% of the current profile),
per-chunk embed HtoD, ~90% of launches. Emerging, in order: (1) GEMM
compute efficiency (the desirable ceiling); (2) attention's O(S²) term past
~2k tokens when the KV prefix spills L2 — Stage 2c.2's tiled online-softmax
attack, orthogonal to this graph; (3) hidden-slab residency capping N_max.

## 7. Comparison of execution philosophy

Modern high-performance inference systems generally converge toward
layer-first or large-microbatch execution for prefill, because it improves
weight reuse and arithmetic intensity: llama.cpp schedules prefill as a
layer-wise graph over an n_ubatch (default 512) token microbatch and reaches
~1400 tok/s on the same T4/GGUF; CUTLASS/cuBLASLt GEMMs assume the caller
presents all rows at once (weight/output-stationary dataflow is their design
premise); Candle and Burn delegate prefill to full-batch GEMM backends;
FlashAttention applies the same stream-the-reused-operand-once philosophy
inside the attention node. The convergence follows from the two lower
bounds (weights >= W once; attention >= the FlashAttention IO bound), which
is the theoretical complement to the field survey.

## 8. Migration strategy — one step, one measurement

- **Phase A (this commit):** graph swap, GEMM contract unchanged.
  PREFILL_BATCH 64 -> 512; embeddings staged host-side, one HtoD per chunk;
  `gemm_rows` issues the MMA GEMM in 64-row sub-slabs, so weight traffic is
  **unchanged by design** — Phase A isolates graph-correctness risk from
  the traffic win. Expected: correctness-neutral, small win (launches,
  HtoD).
- **Phase B (after Phase A validates on T4):** GEMM contract v2 — weights
  streamed once per resident chunk. The Θ(W) collapse lands here.
- **Phase C (separate project):** Stage 2c.2 tiled online-softmax GQA
  attention.

## 9. Risk analysis

- Numerics: per-output dot order unchanged by scheduling -> GEMM outputs
  bit-compatible; attention op untouched. Gated by the glproc oracle tests.
- Activation memory: +~110 MiB VRAM at N=512 (7B); existing footprint check
  guards it.
- Decode: untouched path (separate single-token workspace + captured
  graph; pf-slab resizing only shifts bump offsets fixed at load, before
  capture). Verified by the standing decode A/B anyway.
- Cursor/chunk-boundary bugs (the historical 28x kv.advance overcount
  class): covered by a new 530-token cross-chunk parity test.

## 10. Predicted performance (Analytical Prediction — to be validated by glbench)

At S≈596 on the T4, with Phase B landed and attention unchanged:
prefill in the **900–1400 tok/s band** (from 362.6). This is a prediction,
not a result; the falsification path is `glbench compare` between the
Phase-A and Phase-B archives plus the `[prefill split]` profile, which
should show FFN/QKV buckets dropping toward the 6.9 GB / ~200 GB/s floor
and attention becoming the dominant share. If measurement disagrees, the
model above is wrong somewhere specific — and the instrumentation to locate
it already exists.
