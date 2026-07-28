# gljax — Overall Architecture

**Series:** gljax (Sanctum Visibilia) — ARTX01 … ARTX16
**Status:** Design complete, **implementation not started**
**Compiled:** 2026-07-27
**Total:** 16 documents, ~15,700 lines

> **What gljax is:** a pure-Rust XLA/PJRT client for LLM inference, targeting TPU v5e, A100, H100,
> and CPU — with zero Python, zero ML-framework dependencies, and no kernels of its own.

---

# 0. ⛔ Reality Check — read this first

**Nothing in this series is implemented.**

* `gljax/` contains architecture documents only. There is no `src/`, no `Cargo.toml`.
* `gljax` is **not a member** of the root workspace (`glcore`, `glproc`, `glcuda`, `glvulkan`,
  `glmetal`, `glbench`, `glcli`, `glictus-caliburni`, `packages/{core,mcp}`).
* `glserve` (ARTX16) does not exist either.

The series specifies a complete engine. It is a design, and every wave gate in it is a claim to be
tested, not a result. **Where a document reports a number, that number came from published research
or from a *different* GwenLand engine — never from gljax.**

What *does* exist in-repo and is referenced: `glcore::tokenizer` (✅ **14 vocabulary families
exact** against reference vectors, enforced on every build — ARTX13 §0.4; it was measured wrong and
rewritten, see §0.2.2),
`glbench` (KL divergence, validation harnesses), `glproc`/`glcuda` (measured CPU/GPU results, cited
as method not as evidence — ARTX08 Scope), and `architecture/Pridwen-proposal-v5.md` (GQ4A/GQ2A).

---

# 1. The Map

```text
                        ┌──────────────────────────────────────┐
   ECOSYSTEM            │  ARTX16  Distributed Serving         │  glserve, OpenAI API,
                        │          (glserve, TP+PP, HA)        │  multi-node, routing
                        └──────────────────┬───────────────────┘
                                           │
   ┌───────────────────────────────────────┴───────────────────────────────────┐
   │  ARTX13 Tokenization  →  ARTX14 Sampling  →  ARTX15 Structured Generation │
   │  text → tokens           logits → tokens      grammar-masked logits        │
   └───────────────────────────────────────┬───────────────────────────────────┘
                                           │
   ┌───────────────────────────────────────┴───────────────────────────────────┐
   │  ARTX12  Model Compatibility & Runtime Conformance                        │
   │  Part A: what loads      Part B: how correctness is proven                │
   └───────────────────────────────────────┬───────────────────────────────────┘
                                           │
   CORE ENGINE                             │
   ┌───────────────────────────────────────┴───────────────────────────────────┐
   │ ARTX08 Matrix    ARTX09 Attention   ARTX10 Quantized   ARTX11 Speculative │
   │ compute          & memory           runtime            inference          │
   └───────────────────────────────────────┬───────────────────────────────────┘
                                           │
   FOUNDATION                              │
   ┌───────────────────────────────────────┴───────────────────────────────────┐
   │ ARTX01 PJRT FFI → ARTX02 IR → ARTX03 ops → ARTX04 runtime/checkpoint      │
   │        → ARTX05 static KV + bucketing → ARTX06 TP+MoE → ARTX07 batching   │
   └───────────────────────────────────────────────────────────────────────────┘
```

| # | Document | Owns | Lines |
|---|---|---|---|
| **01** | [PJRT + StableHLO Rust Research](ARTX01-pjrt-stablehlo-rust-research.md) | PJRT C API FFI, dynamic plugin loading, StableHLO MLIR text emission, precision policy, FP64 oracle | 1376 |
| **02** | [IR Design](ARTX02-ir-design-funcbuilder-tracecx-ssa.md) | `MlirEmitter`, `FuncBuilder`, `TraceCx`, `SsaValue`, `Tensor`, `PrecisionPolicy` | 1818 |
| **03** | [ops/ Layer](ARTX03-ops-layer-llm-implementations.md) | softmax, rms_norm, rope_neox, gqa_attention, swiglu_ffn, gather_embed, moe_ffn | 1290 |
| **04** | [runtime/ + checkpoint/](ARTX04-runtime-and-checkpoint.md) | `Session`, `CompileCache` (SHA256), safetensors + `.gllm` loaders | 1509 |
| **05** | [Static KV Cache + Bucketing](ARTX05-kv-cache.md) | `dynamic_update_slice` + buffer donation, 5 seq buckets, `Session::generate()` | 861 |
| **06** | [Multi-Device TP + MoE](ARTX06-multi-device-tensor-parallel-moe.md) | `DeviceMesh`, Shardy/SDY sharding, `all_reduce`, expert all-to-all | 929 |
| **07** | [Continuous Batching](ARTX07-continuous-batching-and-dynamic-sequence-multiplexing.md) | `KvSlotManager`, `StaticKvSlab`, iteration-level scheduling, chunked prefill | 825 |
| **08** | [Matrix Compute](ARTX08-matrix-compute-architecture.md) | Plan/Lower/Execute; **gljax owns no GEMM kernel**; structural fusion | 1070 |
| **09** | [Attention & Memory](ARTX09-attention-and-memory-architecture.md) | FlashAttention reachability, KV access, prefix cache, RadixAttention | 573 |
| **10** | [Quantized Runtime](ARTX10-quantized-runtime-architecture.md) | FP8/INT8/INT4/GQ4A, emission dispatch, capability probe, quantized KV | 760 |
| **11** | [Speculative Inference](ARTX11-speculative-decoding.md) | Draft/verify under static shapes, cross-vocab, `Architecture` descriptor | 864 |
| **12** | [Model Compat & Conformance](ARTX12-model-compatibility-and-runtime-conformance.md) | **A:** GGUF/GPTQ/AWQ/SmoothQuant/FP8, capability matrix · **B:** 4-tier oracle harness | 1467 |
| **13** | [Tokenization](ARTX13-tokenization-architecture.md) | `Tokenizer` trait, vocab loading, **incremental detokenization**, cross-vocab | 493 |
| **14** | [Sampling & Logits](ARTX14-sampling-and-logits-processing.md) | Sampler chain, **device/host split**, penalties, grammar-mask seam | 426 |
| **15** | [Structured Generation](ARTX15-structured-generation.md) | LLGuidance, mask pipeline, jump-forward, **speculation rollback** | 415 |
| **16** | [Distributed Serving](ARTX16-distributed-serving.md) | `glserve`, OpenAI API + SSE, TP+PP, fault tolerance, observability | 1075 |

---

# 2. ⭐ The Five Principles That Emerged

None of these was planned. Each was arrived at independently in two or more documents, which is why
they are worth naming.

## P1 — gljax produces StableHLO. It owns no kernels.

**Stated:** ARTX08. **Tested and held:** ARTX09 (FlashAttention), ARTX10 (quantized GEMM),
ARTX14 (sorting-free sampler), ARTX15 (grammar masks).

Every layer that could plausibly have justified a kernel was examined and declined:

| Temptation | Verdict |
|---|---|
| GEMM microkernel (BLIS/CUTLASS-style) | ⛔ Would need Pallas *and* CUDA; competes with cuBLASLt on its own silicon; rewritten every accelerator generation (Hopper `wgmma` → Blackwell `tcgen05.mma`) |
| FlashAttention | ✅ **Reachable without a kernel on GPU** — XLA's `CudnnFusedMHARewriter` pattern-matches the natural graph. ⛔ Not reachable on TPU (Splash Attention is Pallas) |
| Quantized GEMM | ✅ StableHLO has quantized types; ⚠️ whether the backend *fuses* them is a plugin property, not gljax's |
| Sorting-free sampler | ⛔ FlashInfer's kernel; gljax splits at top-K instead |

## P2 — Emit standard IR, let the backend decide, **measure whether it did**

**Independently derived in:** ARTX09 §3.2 (attention rewrite verification), ARTX10 §5 (quantization
capability probe), ARTX14 §5.1 (does `top_k` lower well?).

```text
1. Emit portable IR — no custom call, no backend branch
2. The backend may or may not apply its optimized path
3. MEASURE — a near-miss is silent, never an error
4. Keep the unoptimized path always reachable as the floor
```

⚠️ **Recommendation carried forward:** hoist this into one `gljax/src/probe/` with a single result
type, disk cache, and versioning scheme — rather than two parallel implementations drifting apart.
ARTX09 §8 raised it; it should be settled when ARTX10's `quant/probe.rs` is written.

## P3 — Static shapes are the organizing constraint, and each feature adds a cache-key dimension

```text
ARTX05   key = (seq_bucket, dtype, device)
ARTX07   + batch_size            ← slot-count buckets
ARTX11   + gamma, + arch_hash    ← speculation depth, model architecture
ARTX14   + K                     ← top-K reduction width
```

⚠️ **These multiply.** ARTX16 §4.2 records XLA/TPU warmup at 20–30 minutes cold, ~5 minutes warm.
5 seq buckets × 4 slot buckets × 3 γ values × 2 architectures is 120 artifacts. **Bucket-grid design
is a first-class capacity decision**, not a tuning afterthought.

The same constraint is what excluded dynamic draft trees (ARTX11 §0.2) and PagedAttention (ARTX07),
and what forced ARTX14's device/host sampling split.

## P4 — The bug class is **silent wrong output**, and it recurs everywhere

GwenLand shipped it once: a Q6_K dequant used naive linear nibble order, corrupting `ffn_down` in
every layer. Shapes correct, no error, output fluent. Found by an end-to-end run, traced backwards.

Every document since has found the same shape in its own domain:

| Document | The silent failure |
|---|---|
| ARTX11 §7 | GQA block grouping instead of interleaved — identical shapes, corrupted attention |
| ARTX12 §A3.2 | AWQ's **interleaved** 4-bit packing (`0x86427531`) read as natural order |
| ARTX12 §A3.3 | SmoothQuant's smoothing vector folded into the preceding norm — no tensor fingerprint at all |
| ARTX13 §0.2 | `glcore::tokenizer`'s merge logic — untested, and merge order is where tokenizers silently disagree |
| ARTX15 §4.1 | Grammar state advanced γ times while only `n` tokens were accepted |

⚠️ This is why ARTX12 exists at all, and why its `SupportLevel::Verified` requires evidence enforced
by a test (§A5). **A support table that can drift silently is worse than none.**

## P5 — Refuse rather than approximate

ARTX12 §A4 (unknown architecture), §A6 (every validation refuses), ARTX13 §3 (unknown chat template),
ARTX15 §2 (unsupported schema keyword → 400 at admission).

The rule: when the failure mode of guessing is *silent wrongness*, guessing is never the safer
default — even when refusing is less convenient.

---

# 3. What Blocks What

```text
IMPLEMENTATION CRITICAL PATH (nothing below can start before this)
  ARTX01 PJRT FFI → ARTX02 IR → ARTX03 ops → ARTX04 Session → ARTX05 generate()

HARD GATES discovered during the series
  ┌─ ARTX08 A8.α  (precision_config / preferred_element_type plumbing)
  │     └─► blocks ARTX10 entirely — quantization is a numerics contract
  │
  ├─ ARTX12 Part B T0/T1  (correctness harness, no device needed)
  │     └─► blocks ARTX11 — a 2nd model + new KV pattern + new sampling = 3 unknowns
  │
  ├─ ARTX13 A13.0  (tokenizer hardening: reference token-ID parity)
  │     └─► blocks all of ARTX13, and ARTX11's cross-vocab claims inherit its uncertainty
  │
  ├─ ARTX14 A14.5  (distribution sampling + gather for verify)
  │     └─► ARTX11's lossless guarantee is UNTESTABLE without it
  │
  ├─ ARTX11 A11.0  (Architecture descriptor — the ARTX03 retrofit)
  │     └─► blocks multi-architecture support; ARTX03 is Qwen2-shaped in 7 ways
  │
  ├─ ARTX15 rollback verification  (does LLGuidance expose checkpoint/rollback?)
  │     └─► until answered, grammar + speculation are MUTUALLY EXCLUSIVE
  │
  └─ ARTX16 §2.1  (PJRT multi-host coordination FFI — never designed in ARTX01/06)
        └─► blocks ARTX16 A9.3 multi-node entirely
```

⭐ **The cheapest high-value work in the entire series is ARTX12 Part B tiers T0** — pure host-side
Rust, no PJRT, no device, no model — which catches the whole structural bug class (transposed
dimension numbers, GQA grouping, attention scale, fusion offsets, `lm_head` axis). It can be written
alongside ARTX02/ARTX03 rather than after them.

## 3.1 Suggested build order

```text
1.  ARTX01 → ARTX02 → ARTX03 → ARTX04 → ARTX05        get one token out
    (write ARTX12 Part B T0 tests in parallel — they need no device)
2.  ARTX08 A8.α                                        make numerics stateable
3.  ARTX13 A13.0 + A13.1                               text in, text out
4.  ARTX14 A14.1 + A14.2                               real sampling
5.  ARTX16 A9.1 + A9.2                                 it serves HTTP
    ── minimum viable engine ──
6.  ARTX07                                             concurrency
7.  ARTX11 A11.0 (Architecture) → ARTX12 Part A        more models, correctly
8.  ARTX09 → ARTX10 → ARTX11 → ARTX15                  performance + features
9.  ARTX16 A9.3                                        multi-node (needs new FFI)
```

---

# 4. Key Numbers

⚠️ **All from published research or other GwenLand engines. None measured on gljax.**

## Hardware ridge points (ARTX08 §A8.1)

| Device | Peak BF16 | HBM BW | Ridge point |
|---|---|---|---|
| TPU v5e | 197 TFLOP/s | 819 GB/s | ≈ 241 FLOP/byte |
| A100 80GB | 312 TFLOP/s | 2,039 GB/s | ≈ 153 |
| H100 SXM | ~989 TFLOP/s | 3,350 GB/s | ≈ 295 |

⭐ **Low-batch decode runs at 0.5–2 FLOP/byte — 200–600× below the ridge.** This single fact drives
ARTX08 (weight-only quantization is the decode lever), ARTX10 (bandwidth, not compute),
ARTX11 (speculation converts idle compute into throughput), and ARTX07 (batching raises intensity).

## Per-iteration traffic at ARTX07's 64 slots

| Flow | Bytes | Source |
|---|---|---|
| KV read (BF16, S=2048) | **201 MB** ↓ | ARTX09 §4.3 |
| Logits, full-vocab host sampling | **38.9 MB** ↓ | ARTX14 §0.1 |
| Logits, top-K split (K=512) | **256 KB** ↓ | ARTX14 §2.2 — **148× less** |
| Grammar mask upload (bit-packed) | **1.19 MB** ↑ | ARTX14 §3.3 |

## Attention score tensor (ARTX09 §2)

| Config | Size |
|---|---|
| Decode (`S_q`=1) | 458 KB — negligible |
| Prefill unchunked, S=8192 | ⛔ 15.0 GB — does not fit v5e |
| Prefill **chunked** (512), S=8192 | 939 MB — **16× less** |

⭐ ARTX07's chunked prefill, adopted for latency, turns out to remove the O(S²) term for free.

## Derived tolerances (ARTX12 §3)

```text
BF16 unit roundoff u = 2⁻⁸ ≈ 3.91e-3;  FP32 u = 2⁻²⁴ ≈ 5.96e-8
per-matmul  δ ≈ 2·u_bf16 + K·u_fp32  ≈ 1e-2   (does NOT scale with K)
logits      δ ≈ √L · δ_matmul        ≈ 4e-2 at 24–32 layers
```

⚠️ Input rounding dominates accumulation by 16–150×. **A test that starts failing as K grows means
the accumulator is not FP32.**

---

# 5. Deferred, and Why

| Item | Blocked on | Where |
|---|---|---|
| PagedAttention | Static-shape thesis; needs a per-backend ragged kernel | ARTX07 |
| Prefill/decode disaggregation | Multi-host FFI; ARTX07 already chose chunked prefill (the competing answer) | ARTX16 §3.3 |
| Dynamic draft trees (EAGLE-2) | Verified-position count varies per step → recompilation | ARTX11 §0.2 |
| Medusa / Hydra / EAGLE heads | Require **training**; gljax is inference-only, no gradient path | ARTX11 §2.1 |
| MXFP8 / block-scaled | Becomes mandatory only if gljax targets consumer Blackwell | ARTX10 §3.3 |
| Physical KV sharing (prefix dedup) | Needs paging; gljax takes the **compute-reuse** half instead | ARTX09 §5.2 |
| Multi-LoRA serving | Not scoped; ⚠️ requires per-adapter KV isolation or tenant context bleeds | ARTX16 §10 |
| FP8 on TPU v5e | **Unconfirmed** — v5e publishes BF16 and INT8, not FP8 | ARTX10 §3.2 |

---

# 6. The Open Questions That Matter Most

Ranked by how much downstream work rests on them.

1. ✅ **ANSWERED — it did not, and the answer was worse than the question.** Scored against
   llama.cpp's reference vectors, **not one vocabulary was correct**; the worst got a third of its
   inputs wrong while every round-trip test passed. Closed by rewriting rather than hardening.
   `glcore::tokenizer` is now at **14 vocabulary families exact**, enforced on every build.
   ⚠️ Two carry a stated caveat that ARTX13 §5's overlap claims still inherit — see ARTX13 §0.4.
   *(ARTX13 §0.2.2, §0.4)*
2. ⭐ **Does `CudnnFusedMHARewriter` fire for GQA, and for Gemma?** ARTX11's query pre-attention
   scalar and QK-norm insert ops into the matched region. If it does not fire, prefill attention
   memory is bounded only by chunk size. *(ARTX09 §7.1)*
3. ⭐ **Does any PJRT plugin actually fuse `uniform_dequantize → dot_general`?** If none does,
   quantization for gljax remains a **storage** format — a legitimate measured outcome, not a
   failure. *(ARTX10 A12.2)*
4. ⭐ **Does LLGuidance expose checkpoint/rollback?** Without it, structured generation and
   speculative decoding cannot both be enabled. *(ARTX15 §6.1)*
5. **Is ARTX01's TPU-MXU-accumulates-in-BF16 claim wrong?** Published TPU docs say FP32. If so,
   ARTX01's long-context FP32-cast guidance solves a non-problem. Settle with the FP64 oracle, not by
   editing either document. *(ARTX08 §A8.3)*
6. **How expensive is the PJRT multi-host coordination FFI?** Entirely undesigned, and it gates all
   multi-node work. *(ARTX16 §2.1)*

---

# 7. Document Conventions

* **Filenames:** `ARTXnn-lowercase-kebab.md`, zero-padded to two digits so lexicographic order
  matches series order (the repo's `architecture/GateCostModel/` precedent).
* **⚠️ DESIGN DECISION** marks a choice with alternatives and a rationale. Each document ends with an
  appendix table summarizing them, including reversibility.
* **⛔** marks a correction to a premise, a hard blocker, or a known-dangerous pattern.
* **⭐** marks the load-bearing finding in a section.
* **Sources** at the end of each document, split into web sources and repo-internal references.
* Every number is attributed. Numbers from other GwenLand engines are cited as *method*, never as
  evidence about gljax — a rule established in ARTX08's Scope after CPU-tier verdicts were initially
  misapplied to cloud hardware.

---

# 8. Reading Paths

**"I want to implement this."**
→ §3.1's build order. Start ARTX01, and write ARTX12 Part B T0 tests alongside — they need no device.

**"Why does gljax not just use vLLM's approach?"**
→ ARTX08 §2 (no kernels, and why), ARTX07 §"Why PagedAttention Is Not Used", ARTX10 §2.2 (every
production quantized stack uses custom kernels; gljax deliberately does not).

**"What makes gljax different?"**
→ §2's five principles. Short version: a portable StableHLO producer that measures whether the
backend took the fast path, rather than a kernel library that guarantees it.

**"Is this going to be fast?"**
→ Honestly: unmeasured. §4's numbers bound what is *possible*; ARTX10 §2.2 states plainly that gljax
should not plan to match vLLM's quantized throughput. The bet is portability and correctness, with
performance obtained where the backend already provides it.

**"What's most likely to go wrong?"**
→ §2's P4 and §6's open questions. Of the three load-bearing assumptions this once listed, **the
tokenizer has been measured** — it was wrong, and is now 14 families exact. That leaves the attention
rewrite (unverified) and the quantization fusion (unprobed).

⚠️ The tokenizer is worth reading as the *calibration* for the other two, not as reassurance. It was
the assumption that looked safest — an existing, working, from-scratch implementation with passing
tests — and it was the one that turned out to be wrong in every vocabulary. Neither of the remaining
two has even that much evidence behind it.
