# ARTX8 — Matrix Compute Architecture

**Series:** ARTX8
**Status:** Draft — research-grounded
**Depends On:** ARTX2 (IR: `FuncBuilder`, `emit_dot_general`, `PrecisionPolicy`), ARTX3 (ops/ layer)
**Related:** ARTX5 (bucketing), ARTX6 (tensor-parallel sharding), ARTX7 (continuous batching)
**Next:** [ARTX09 — Attention & Memory Architecture](ARTX09-attention-and-memory-architecture.md)
**Research grounded:** 2026-07-27 (sources at the end of Research Findings)

> ⚠️ **Series-numbering note.** ARTX7 previously pointed to "ARTX8 — Distributed Serving" as its
> successor. This document claims the ARTX8 slot for Matrix Compute, so **distributed serving moved
> to ARTX16**. ARTX7 has been updated to match (`Next:` pointer, "Future Work — ARTX16" heading, and
> its seven `ARTX8+` deferral markers). No other document in `gljax/architecture/` referenced ARTX8.

---

# Scope — gljax is cloud-first

Per ARTX1, gljax targets **TPU v5e, A100, H100, and CPU** — but the CPU plugin's stated role there is
the **FP64 reference oracle**: ARTX1 §3.1's support matrix lists it as "No restrictions. Ideal for
reference/oracle runs," and §3.4 builds the oracle pattern on it. CPU is a *correctness* backend for
gljax, not a performance target.

⛔ **This has a hard consequence for what counts as evidence in this document.** GwenLand's CPU
engine (`glproc`) has an extensive body of measured optimization results in
`gl-agent-skills/cpu-skills/rejected-optimizations.md`. **Those verdicts do not apply here, and are
not cited in this document as evidence about gljax.** That file states its own scope explicitly:

> This list is **per hardware tier**. A future AVX-512 desktop tier or an ARM NEON port re-opens
> questions *for that tier only* […] never by editing this list's verdicts for the i3 tier.

The *judgment patterns* generalize (a probe is not a production measurement; an isolated kernel win
may not survive integration). The *verdicts* do not. They were measured on a 2-core Tiger Lake
i3-1115G4 with 8 GB of DDR4-2667 at a **29.4 GB/s** ceiling — roughly **1/28th** of an H100's HBM3
bandwidth, with a completely different compute-to-bandwidth ratio and no MXU or Tensor Core at all.

The clearest illustration that these tiers genuinely invert: glproc's native Q4_K quantized compute
path **lost 33%** on that CPU and was rejected — while `glcuda`'s native Q4_K kernel **wins** on GPU,
because GPU bandwidth is far more expensive relative to compute. Same repo, same quantization
format, opposite verdicts. Importing the CPU conclusion into a cloud-accelerator document would have
argued for exactly the wrong thing.

Every performance claim below is grounded in gljax's actual target hardware or in published
cloud-accelerator measurements.

---

# Executive Summary

This document specifies gljax's matrix compute architecture. The single most important
architectural fact, and the one that determines everything else:

> **gljax does not own a GEMM kernel, and must not acquire one.**

The brief's constraints — no Triton DSL, no custom CUDA codegen, no auto-tuning, no runtime kernel
search, compiler-friendly, backend-agnostic, maps efficiently to StableHLO/PJRT — are not a list of
things gljax happens to skip. Taken together they *define* gljax's position in the stack: gljax is
a **producer of `stablehlo.dot_general`**, and XLA is the compiler that turns that into a kernel
(cuBLASLt, Triton, or a Mosaic/MXU program, chosen by XLA's own autotuner). ARTX1 already committed
to this by making gljax a pure-Rust, plugin-only PJRT client.

Therefore gljax's "matrix compute architecture" is **not** a kernel library. It is three separated
host-side concerns:

| Concern | Owns | Does NOT own |
|---|---|---|
| **Plan** | Classifying each contraction (GEMM / GEMV / batched / ragged), choosing dtype + accumulation, labeling regime | Any IR, any buffer |
| **Lower** | Emitting `dot_general` / `ragged_dot` with correct dimension numbers, precision, and accumulation type | Tiling, packing, scheduling |
| **Execute** | Nothing — hands off to PJRT | Everything (XLA owns it) |

Everything BLIS, CUTLASS, and oneDNN do — packing, register tiling, cache blocking, microkernels,
thread decomposition — happens *below* gljax's floor. gljax studies those designs not to reimplement
them but to understand **what makes XLA's chosen kernel fast**, and then to hand XLA shapes, layouts,
and dtypes that let it pick the good path.

Three findings in this document contradict premises in the brief or in sibling documents, and are
flagged inline where they occur:

1. **"LLM inference is overwhelmingly GEMM-bound" is half true.** Prefill is GEMM-bound; decode is
   GEMV-bound and therefore *bandwidth*-bound — on an H100, single-request decode sits roughly
   200–600× below the ridge point. Designing the whole matrix layer around GEMM would optimize the
   phase that is not the bottleneck for low-batch inference.
   See [A8.1](#wave-a81--matrix-computation-fundamentals).
2. **ARTX1 §3.2 states TPU MXU accumulates in BF16. Published TPU architecture says FP32.** This
   changes the mixed-precision guidance for attention on TPU. See
   [A8.3](#wave-a83--hardware-acceleration).
3. **ARTX2's `emit_dot_general` hardcodes `precision_config = [DEFAULT, DEFAULT]`** and drops
   `preferred_element_type` entirely, while `infer_dot_general_shape` sets output dtype = lhs dtype.
   For a BF16 model this silently requests BF16-in/BF16-out with unspecified accumulation — the one
   knob that most affects both numerics and speed is currently not reachable. See
   [A8.4](#wave-a84--matrix-lowering) and [API Design](#api-design).

---

# Research Findings

## Wave A8.1 — Matrix Computation Fundamentals

### The primitive taxonomy

| Form | Shape | Arithmetic intensity | LLM occurrence |
|---|---|---|---|
| **GEMV** | `[1,K] × [K,N]` | ≈ 1 op/byte, *independent of size* | Every decode-step projection |
| **GEMM** | `[M,K] × [K,N]`, M ≫ 1 | ≈ O(M) op/byte | Every prefill projection |
| **Batched GEMM** | `[B,M,K] × [B,K,N]` | O(M) per batch element | Attention scores/AV across heads |
| **Strided batched** | same, one buffer + stride | same | Same, when heads are one tensor |
| **Ragged / grouped** | `[m,k] × [g,k,n]` + group sizes | varies per group | MoE expert FFN |

`dot_general` expresses all five. Batch dimensions, contracting dimensions, and free dimensions are
all explicit, so "batched GEMM" is not a separate op in gljax — it is `dot_general` with non-empty
batching dims. This is why ARTX2 was right to expose `dot_general` rather than a `matmul` primitive,
and `Tensor::matmul` is correctly just a convenience wrapper over it.

### Arithmetic intensity and the roofline — and its limit

The roofline model plots achievable FLOP/s against arithmetic intensity (AI = FLOPs ÷ bytes moved).
Below the ridge point the workload is bandwidth-bound; above it, compute-bound.

The decisive property: **matrix-vector multiplication has AI ≈ 1 regardless of dimension.** Each
weight element is loaded once and used for exactly one multiply-add. No amount of kernel engineering
changes that ratio. Published analyses put prefill AI at roughly 200–400 op/byte versus decode at
roughly 1 — with GPU utilization falling to 20–40% during decode.

⛔ **Correction to the brief's framing.** The brief asks to "focus on why modern LLM inference is
overwhelmingly GEMM-bound." That is true of *prefill*, and false of *low-batch decode* — including
on gljax's own target hardware. Ridge points for the accelerators ARTX1 targets:

| Device | Peak BF16 | HBM bandwidth | **Ridge point** (FLOP/byte) |
|---|---|---|---|
| **TPU v5e** | 197 TFLOP/s | 819 GB/s (HBM2e, 16 GB) | ≈ **241** |
| **A100 80GB SXM** | 312 TFLOP/s | 2,039 GB/s (HBM2e) | ≈ **153** |
| **H100 SXM** | ~989 TFLOP/s | 3,350 GB/s (HBM3) | ≈ **295** |

Single-request decode runs at an arithmetic intensity of roughly **0.5–2 FLOP/byte** — placing it
**200–600× below** the H100 ridge point. The consequence is stark enough to be quotable: during
low-batch decode the matrix unit is idle the overwhelming majority of the time, waiting on weights
to arrive from HBM. A device with ~989 TFLOP/s of BF16 compute is being used as a memory controller.
And per the property above, that gap belongs to the *operation*, not the implementation — no kernel,
library, or compiler closes it.

⛔ **Second-order caution: roofline gives an upper bound, not a promise.** The model assumes a kernel
can actually sustain the throughput its intensity implies. Real hardware does not reach the ceiling
even when compute-bound — published figures put H100 and B200 at roughly **80–85% of claimed peak
FLOPs** in practice, with TPUs closer to **95%**. Any ARTX8 decision argued from "roofline says this
moves fewer bytes" must also state *which kernel* is assumed to sustain the resulting intensity, and
be confirmed by measurement on the target device before it is believed.

### Consequence for ARTX8

Batch size is the lever that moves decode from GEMV toward GEMM: N requests sharing one weight read
multiply the arithmetic intensity by roughly N, sliding the workload rightward along the
bandwidth roofline. ARTX7's continuous batching is therefore not only a throughput feature — it is
*the* mechanism that raises decode's arithmetic intensity toward the ridge point. **ARTX7 and ARTX8
are the same optimization viewed from two ends.**

⚠️ The batch size at which a given stage actually crosses into compute-bound is **model- and
shape-dependent, and must be measured** — it depends on hidden dim, vocab size, KV-cache traffic,
and how much of the read is weights versus cache. Published crossover claims vary widely because
they measure different stages. HBM capacity also caps the usable batch before the ridge point is
reached. The matrix layer should therefore *expose* the regime per stage
(see [`matrix/roofline.rs`](#module-responsibilities)) rather than assume a global crossover.

---

## Wave A8.2 — Efficient GEMM Architectures

Studied for architectural ideas, explicitly **not** for implementation.

### The GotoBLAS / BLIS decomposition

BLIS refactors the GotoBLAS algorithm as **five loops around a micro-kernel**: three outer loops
around a macro-kernel plus two packing routines, with the macro-kernel itself being two more loops
around the micro-kernel. The micro-kernel is a loop over rank-1 (outer-product) updates, written in
assembly or vector intrinsics; the five surrounding loops are plain C. Exposing those loops in C
(rather than GotoBLAS's assembly macro-kernel) is what buys BLIS its portability, and gives multiple
loop levels at which thread parallelism can be introduced.

**Packing** — copying panels of A and B into contiguous, micro-kernel-shaped buffers — is the load-
bearing idea. It converts strided, TLB-hostile access into linear streams that feed the micro-kernel
at full rate.

### The CUTLASS decomposition

CUTLASS mirrors the same idea onto the GPU execution hierarchy: **threadblock tile → warp tile →
register tile**, with a shared-memory staging buffer double-buffered against the global-memory load
of the next tile, and register fragments double-buffered against the current MMA. CUTLASS 3.x
composes a kernel from a **collective mainloop + collective epilogue**. Modern Hopper kernels go
further with warp specialization (a producer warp group driving TMA loads, consumer warp groups
driving MMA + epilogue).

### The idea worth extracting

Strip the hardware detail and BLIS, CUTLASS, oneDNN, and IREE all encode the same three-part
structure:

```text
1. A fixed-shape innermost compute unit    (micro-kernel / MMA fragment / MXU pass)
2. A memory staging discipline that keeps that unit fed   (packing / smem pipeline / tensor.pack)
3. A loop nest that tiles the problem to fit the memory hierarchy
```

⛔ **gljax owns none of the three.** Restating that as a design rule: every optimization in this
section is expressible only *below* the StableHLO boundary. Attempting them in gljax means either
emitting custom_call (excluded by the brief) or hand-tiling the traced graph — which fights XLA's
own tiling and layout assignment, and materializes copy ops.

**Why that is not a close call for a cloud-first engine**, in order of decisiveness:

1. **The competition is cuBLASLt and the XLA TPU compiler.** These are the reference implementations
   for their own silicon, written by the vendors who designed the MMA units, and they already reach
   roughly 80–85% of peak on H100 and ~95% on TPU. A gljax microkernel would need to beat that to
   justify existing at all.
2. **One kernel becomes N kernels.** There is no portable place to put a microkernel below
   StableHLO: TPU needs Pallas/Mosaic, NVIDIA needs CUDA or Triton, CPU needs per-ISA intrinsics.
   ARTX1 committed gljax to being a pure-Rust, plugin-only PJRT client precisely to avoid owning
   that matrix of toolchains, and ARTX7 made the same call when it deferred PagedAttention.
3. **XLA already searches the space.** XLA:GPU autotunes across cuBLAS, cuBLASLt, cuDNN Graph GEMMs,
   and Triton per dot. A gljax kernel layer would be a second, worse search sitting on top of a
   better one.
4. **The hardware target moves every ~18 months.** Hopper `wgmma` (warpgroup, 128 threads) →
   Blackwell `tcgen05.mma` (back to 32-thread warp scope) is a full rewrite of the innermost loop;
   TPU MXU went 128×128 → 256×256 at v6e. Vendor libraries absorb that churn. A gljax kernel layer
   would inherit it permanently.

The generalizable engineering lesson from GwenLand's CPU work still applies as *method*, not as
verdict: an isolated kernel benchmark is not a production measurement, because a real pipeline is a
mix of compute-bound and bandwidth-bound stages and a technique tuned for one regime need not
transfer to the other. That is a reason to demand end-to-end measurement of any kernel-level claim —
on the target device. It is not, on its own, evidence about TPU or GPU behavior.

---

## Wave A8.3 — Hardware Acceleration

| Unit | Shape / capability | Accumulation | Reachable from StableHLO? |
|---|---|---|---|
| **TPU MXU** (≤ v5p) | 128×128 systolic array = 16,384 MACs/cycle; `bf16[8,128] @ bf16[128,128] → f32[8,128]` | **FP32** | Yes — every `dot_general` |
| **TPU MXU** (v6e Trillium+) | 256×256, 4× ops/cycle | FP32 | Yes |
| **NVIDIA Tensor Core** (Hopper `wgmma`) | `m64nNk16`, N ∈ {8..256} step 8; warpgroup of 4 warps | FP32 | Indirectly (cuBLASLt/Triton via XLA) |
| **NVIDIA Tensor Core** (Blackwell `tcgen05.mma`) | `64×N×16` and `128×N×16`, N ≤ 256; back to 32-thread warp scope; 2–4× Hopper | FP16 or FP32 by mode | Indirectly |
| **Intel AMX** | 8 tile registers, 16 rows × 64 B = 1 KB/tile; 512 MAC/cycle BF16, 1024 INT8 | FP32 | Indirectly (CPU plugin / oneDNN) |
| **AVX-512 / VNNI** | 512-bit lanes; `vpdpbusd` int8 dot | INT32 | Indirectly |
| **AVX2** | 256-bit lanes | FP32 / INT32 | Indirectly |
| **ARM NEON / SVE / SME** | SVE is vector-length-agnostic; **SME** adds 2-D tile registers + outer-product instructions (`FMOPA`, `BFMOPA`, `UMOPA`) | FP32 | Indirectly |
| **RISC-V RVV** | Vector-length-agnostic, same VLA family as SVE | varies | Indirectly |

### The common abstraction

Every one of these is a **fixed-shape, mixed-precision, accumulate-in-wider-type outer-product
engine**. The convergence is striking: TPU MXU, Tensor Cores, AMX, and ARM SME independently landed
on 2-D tile registers fed by an outer-product unit that multiplies in a narrow type and accumulates
in a wide one.

That is the *only* hardware abstraction gljax needs to represent, and it maps to exactly two
decisions per matmul:

```text
1. What narrow type do the inputs use?     (bf16 / f16 / f8 / int8)
2. What wide type does accumulation use?   (f32 — essentially always)
```

Both are expressible in StableHLO. Neither is currently reachable through gljax's API (see A8.4).

### ⛔ Correction to ARTX1 §3.2

ARTX1 line 423 states:

> **TPU v5e MXU (Matrix Multiply Unit)**: BF16 input, **BF16 accumulate**. At long sequence lengths,
> cumulative error in attention score computation is noticeably higher than A100.

Published TPU architecture documentation contradicts this: **all multiplies take bfloat16 inputs,
but all accumulations are performed in FP32**, and the MXU's documented signature is
`bf16[8,128] @ bf16[128,128] -> f32[8,128]`.

If the FP32-accumulate description is correct, then ARTX1's derived guidance — "on TPU, for very
long sequences (>8k tokens), you may need to explicitly cast scores to FP32 before the softmax
reduce" — is solving a problem that does not exist at the MXU level, and the extra `convert` ops it
recommends are pure cost. **This is a documentation conflict, not a measurement**, and ARTX1's own
rule applies: it should be settled by running a long-context attention block on a real TPU plugin
and comparing against the FP64 CPU oracle (ARTX1 §3.4 already specifies that oracle), not by editing
either document from theory. Until then ARTX8 treats FP32 accumulation as the default assumption on
all backends, because it is what both the TPU and NVIDIA documentation describe.

---

## Wave A8.4 — Matrix Lowering

### How the stacks lower a matmul

| Stack | Path |
|---|---|
| **StableHLO → XLA** | `stablehlo.dot_general` → HLO `dot` → backend-specific rewrite |
| **XLA:GPU** | `dot` → cuBLAS / cuBLASLt / cuDNN Graph GEMM / **Triton-generated kernel**, selected by XLA's autotuner |
| **XLA:TPU** | `dot` → MXU program via the TPU compiler |
| **MLIR Linalg** | `linalg.matmul` — stays `linalg.matmul` through tiling and fusion; then vectorization → loops |
| **IREE** | `linalg.matmul` → `linalg.mmt4d` via data-tiling, with `tensor.pack` / `tensor.unpack` materializing the tiled layout → micro-kernels |

The interesting divergence: **IREE materializes packing as IR** (`tensor.pack` → `mmt4d`), whereas
XLA keeps packing inside the library/kernel it selects and instead reasons about **layouts**
(minor-to-major orderings propagated through the graph, with conflicts materialized as copy ops).

gljax targets PJRT, so it inherits XLA's model, not IREE's: **gljax never expresses packing. It
expresses shapes and layouts, and XLA's layout assignment decides the rest.** The practical corollary
is a negative one — gljax must avoid emitting transposes or reshapes that force layout conflicts,
because those become materialized copies. XLA fusion performance is documented to regress severely
when a fusion's input/output tensors carry different layouts.

### The precision knobs, and gljax's current gap

`stablehlo.dot_general` carries two mutually-exclusive numeric controls:

* **`precision_config`** — `DEFAULT` | `HIGH` | `HIGHEST`. Tells the backend how hard to work at
  emulating a wider dtype (e.g. emulating f32 on hardware that only does bf16 matmul).
* **`algorithm`** (`DotAlgorithm`) — an explicit, named numeric contract:
  `bf16_bf16_f32`, `f32_f32_f32`, `tf32_tf32_f32_x3`, `bf16_bf16_f32_x3`, `bf16_bf16_f32_x6`, …
  The `_xN` suffix means N passes emulating higher precision (bf16_3x decomposes each input into 3
  bf16 components, does 3 dots, accumulates in f32).

Plus `preferred_element_type`, which *recommends* the accumulation type (explicitly a
recommendation, not a guarantee).

⛔ **Current state in ARTX2** (`stablehlo/ops.rs`, `emit_dot_general`, ~line 603):

```rust
e.line(r#"precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]"#.to_string());
```

Hardcoded. No parameter, no `algorithm`, no `preferred_element_type`. And `infer_dot_general_shape`
(ARTX2 ~line 1258) ends with:

```rust
Shape::new(out_dims, lhs.dtype) // output dtype = lhs dtype
```

So a BF16 model emits BF16-in → BF16-out with `DEFAULT` precision and no stated accumulation type.
On hardware whose MXU/Tensor Core natively accumulates in FP32 this is *probably* fine — the wide
accumulator is used internally and the result is rounded on write-back. But "probably fine, decided
by the backend, unstatable by us" is exactly the wrong property for a project whose ARTX1 built an
FP64 oracle to validate numerics. **The knob that most affects both numerics and speed is currently
not reachable.** Closing that is Wave A8.α below and the single highest-value change in this
document.

### MoE: `ragged_dot`

StableHLO has a first-class op for grouped/ragged contraction: `ragged_dot(lhs, rhs, group_sizes)`
with `ragged_dot_dimension_numbers`. Mode 1 signature: `[b,m,k], [g,b,k,n], [b,g] -> [b,m,n]`, where
the ragged dimension is an lhs non-contracting dim.

This is directly relevant to ARTX3's `ops/moe.rs` and ARTX6's expert sharding, both of which
currently plan a padded batched `dot_general` with a fixed capacity factor. `ragged_dot` expresses
variable expert token counts without capacity padding. **Not adopted in this document** — see
[Future Extensions](#future-extensions) — because it needs per-plugin support verification first,
and ARTX5/ARTX7's whole static-shape thesis makes a ragged dimension a non-trivial interaction, not
a drop-in.

---

## Wave A8.5 — Kernel Fusion

The critical distinction, and the one that determines what gljax builds:

| Fusion class | Who performs it | gljax's job |
|---|---|---|
| **Structural** — concatenating separate weight matrices into one so a single larger GEMM replaces several small ones (QKV, gate+up) | **gljax**, at checkpoint-load + trace time | **Build it.** XLA cannot: the weights arrive as separate arrays; nothing in the graph licenses merging them. |
| **Epilogue** — `dot + bias`, `dot + logistic + multiply` (SwiGLU), `dot + activation` | **XLA**, automatically | **Do nothing.** ARTX1 §7 already notes XLA fuses `dot + logistic + multiply` via epilogue fusion on both GPU and TPU with no custom call. Hand-rolling this is wasted work that also risks defeating the pass. |
| **Prologue** — `rms_norm → dot` | **XLA**, partially | Emit the natural graph; verify, don't force. |
| **Whole-block** — flash attention (QK + mask + softmax + AV in one kernel) | Backend library via `custom_call` | Out of scope (ARTX1 §7 records `@flash_attention` as the mechanism; excluded by ARTX8's constraints). |

### Why structural fusion is worth doing

ARTX1 §7 already recommends emitting gate and up as a single `[D, 2·FFN]` weight matmul then
splitting — "this halves the number of matmul kernel launches and improves memory access patterns
(one large GEMM vs two smaller)." ARTX8 promotes that from a recommendation to the matrix layer's
default, and extends it to QKV.

The mechanism is the arithmetic-intensity story from A8.1 again: three `[1,D]×[D,N]` GEMVs each
stream their own weight matrix with AI≈1. One `[1,D]×[D,3N]` GEMV moves the *same bytes* but as one
dispatch over one contiguous region — which matters for three separate reasons on cloud
accelerators:

* **Dispatch overhead amortizes.** Three kernel launches become one. On a memory-bound op whose
  actual work is short, launch and synchronization overhead is a real fraction of the step.
* **The shape gets large enough to tile well.** A `[D, 3N]` output gives the backend more columns to
  distribute across an MXU pass or a CUTLASS threadblock tile than `[D, N]` does. Very narrow GEMMs
  underutilize a 128×128 (or 256×256) systolic array.
* **One contiguous HBM stream instead of three.** For a bandwidth-bound op, sequential access across
  a single region is the access pattern the memory system is built for.

⚠️ **On published fusion speedups — attribute carefully.** The strongest recent numbers
(QKV fusion 2.51–2.56× prefill / 1.61–1.79× decode; gate+up fusion 2.64–2.68× prefill) come from a
study on the **Tenstorrent Tensix** architecture, whose explicit on-chip SRAM management makes
fusion far more valuable than on GPU/TPU, where XLA already fuses epilogues and the caches are
implicit. **Those numbers must not be quoted as gljax's expected gain.** They establish that the
fusion axis is real and worth building; the magnitude on gljax's targets is unmeasured.

### Fusions this document adopts

| Fusion | Class | Rationale |
|---|---|---|
| **QKV** — concat `[D,Hq·Dh]`, `[D,Hkv·Dh]`, `[D,Hkv·Dh]` → one `[D, (Hq+2·Hkv)·Dh]` | Structural | 3 dispatches → 1. GQA-aware: Q and KV column counts differ. |
| **Gate+Up** — concat `[D,F]`,`[D,F]` → `[D,2F]` | Structural | Already recommended by ARTX1 §7. |
| **Bias / SwiGLU / activation** | Epilogue | **Explicitly not built.** XLA's job. |
| **RMSNorm → dot** | Prologue | **Explicitly not built.** Verify XLA does it. |

⚠️ Structural fusion has a hard constraint from ARTX6: concatenated weights must remain shardable.
A `[D, (Hq+2·Hkv)·Dh]` QKV matrix column-sharded across a TP mesh must split along head boundaries,
not arbitrary column boundaries. The fusion layer must therefore record the segment offsets it used,
so `tp/linear.rs` can shard segment-wise. This is why fusion metadata is part of the plan
([`MatmulPlan::segments`](#api-design)) and not thrown away after emission.

---

## Sources

- [Prefill Is Compute-Bound. Decode Is Memory-Bound.](https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/) — prefill/decode roofline regimes, AI 200–400 vs ~1.
- [Roofline fundamentals for LLM inference](https://github.com/harshuljain13/llm-inference-at-scale/blob/master/content/01_gpu_hardware/01.2_roofline_model/roofline_fundamentals.md) — ridge point, GEMV AI ≈ 1 independent of dimension.
- [BLIS: BLAS and So Much More](https://www.siam.org/publications/siam-news/articles/blis-blas-and-so-much-more/) / [BLIS FAQ](https://github.com/flame/blis/blob/master/docs/FAQ.md) — five loops around the micro-kernel, packing, portability vs GotoBLAS assembly macro-kernel.
- [Efficient GEMM in CUDA — CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html) and [CUTLASS 3.0 GEMM API](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/gemm_api_3x.html) — threadblock/warp/register tile hierarchy, collective mainloop + epilogue.
- [Deep Dive on CUTLASS Ping-Pong GEMM Kernel](https://pytorch.org/blog/cutlass-ping-pong-gemm-kernel/) — producer/consumer warp specialization.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) — MXU 128×128 = 16,384 MACs/cycle, `bf16[8,128] @ bf16[128,128] -> f32[8,128]`.
- [BFloat16: The secret to high performance on Cloud TPUs](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) — bf16 multiply, **FP32 accumulate**.
- [NVIDIA Tensor Core Evolution: Volta to Blackwell](https://newsletter.semianalysis.com/p/nvidia-tensor-core-evolution-from-volta-to-blackwell) and [Blackwell SM100 GEMMs](https://docs.nvidia.com/cutlass/4.3.4/media/docs/cpp/blackwell_functionality.html) — `wgmma` m64nNk16, `tcgen05.mma` shapes, accumulation modes.
- [AI Acceleration using Intel AMX/TMUL](https://cdrdv2-public.intel.com/784471/784471_AI%20Acceleration%20using%20Intel%20AMX-TMUL.pdf) — 8 tile registers, 1 KB/tile, 512 BF16 / 1024 INT8 MAC per cycle.
- [Arm Scalable Matrix Extension (SME) Introduction](https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction) — 2-D tile registers, outer-product `FMOPA`/`BFMOPA`/`UMOPA`, VLA.
- [StableHLO specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) — `dot_general`, `precision_config`, `DotAlgorithm`, `ragged_dot`.
- [JAX issue #23797 — dot algorithm spec](https://github.com/jax-ml/jax/issues/23797) — `bf16_bf16_f32`, `tf32_tf32_f32_x3`, `bf16_bf16_f32_x6`; algorithm and precision mutually exclusive.
- [XLA:GPU Architecture Overview](https://openxla.org/xla/gpu_architecture) and [XLA:GPU Emitters](https://openxla.org/xla/emitters) — cuBLAS/cuBLASLt/cuDNN/Triton selection with autotuning; fusion "hero" op.
- [Shapes and layout | OpenXLA](https://www.tensorflow.org/performance/xla/shapes) and [XLA discussion #766](https://github.com/openxla/xla/discussions/766) — minor-to-major layouts, conflicts materialized as copies, fusion regression on mixed layouts.
- [MLIR Linalg dialect](https://mlir.llvm.org/docs/Dialects/Linalg/) and [IREE data-tiling walkthrough](https://iree.dev/community/blog/2025-08-25-data-tiling-walkthrough/) — `linalg.matmul` → `linalg.mmt4d`, `tensor.pack`/`unpack`.
- [Operator Fusion for LLM Inference on the Tensix Architecture](https://arxiv.org/pdf/2606.09879) — QKV 2.51–2.56× prefill / 1.61–1.79× decode, gate+up 2.64–2.68× prefill. **Tensix-specific; see the attribution warning above.**
- [Introducing Grouped GEMM APIs in cuBLAS](https://developer.nvidia.com/blog/introducing-grouped-gemm-apis-in-cublas-and-more-performance-updates/) — grouped GEMM for MoE.
- [All About Rooflines](https://jax-ml.github.io/scaling-book/roofline/) — ridge-point method; achievable vs peak FLOPs (H100/B200 ~80–85%, TPU ~95%).
- [TPUv5e: The New Benchmark in Cost-Efficient Inference](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) and [TPU v5e | Google Cloud](https://docs.cloud.google.com/tpu/docs/v5e) — 197 BF16 TFLOP/s, 819 GB/s HBM2e, 16 GB, 4 MXUs per TensorCore.
- [NVIDIA A100 vs H100 comparison](https://www.bestgpusforai.com/gpu-comparison/a100-vs-h100) — A100 312 BF16 TFLOP/s @ 2,039 GB/s HBM2e; H100 @ 3,350 GB/s HBM3.
- [Roofline model for LLM inference, Part 2](https://github.com/monamishra95/roofline-model-llm-inference/blob/main/PART2.md) — H100 ridge point ≈ 295 FLOP/byte; batch-1 decode AI ≈ 0.5–2, i.e. 200–600× below the roofline.

> **Deliberately not cited:** GwenLand's `glproc` CPU optimization results
> (`gl-agent-skills/cpu-skills/rejected-optimizations.md` and the associated measurements). See
> [Scope](#scope--gljax-is-cloud-first) for why those verdicts are out of scope for this document.

---

# Design Rationale

## Why a matrix layer exists at all

If gljax emits no kernels, why not just call `Tensor::matmul` everywhere and let XLA sort it out?

Because four decisions genuinely belong to gljax, and today they are scattered or absent:

1. **Which contractions get structurally fused** (QKV, gate+up) — XLA cannot do this; it must happen
   before or during tracing, and it must record segment offsets for ARTX6 sharding.
2. **What numeric contract each matmul declares** — currently hardcoded to `DEFAULT` with no
   accumulation type, unreachable from model code.
3. **Which dimension numbers to use** — `ops/attention.rs`, `ops/ffn.rs`, `ops/moe.rs`, and
   `tp/linear.rs` each construct `DotDimensionNumbers` by hand today. That is four places to get
   batching/contracting dims wrong.
4. **What regime each matmul is in** — needed by ARTX7 (does batching help here?) and by any future
   benchmark that wants to explain *why* a stage is slow rather than just how slow.

A thin layer that owns exactly those four things, and nothing else, is the smallest structure that
removes the duplication without adding a kernel abstraction.

## Why "plan / lower / execute" and not "kernel / dispatch"

The brief's success criteria ask to "clearly separate planning, lowering, and execution." That maps
cleanly onto what is actually separable here:

```text
PLAN     pure host-side data. No IR, no Tensor, no builder borrow.
         Testable with zero PJRT. Serializable (useful for cache keys + debugging).
   ↓
LOWER    plan + operands → SSA value. The only place `dot_general` is emitted in gljax.
   ↓
EXECUTE  XLA. gljax contributes nothing.
```

This is the same ownership split ARTX7 used for `KvSlotManager` (logic) vs `StaticKVSlab` (storage),
and for the same reason: the half that has no backend dependency can be unit-tested exhaustively,
and the half that does stays thin enough to audit.

## Why no auto-tuning

Excluded by the brief, and correctly so — but worth recording *why*, since "the backend already does
it" is the real reason and it is easy to forget:

* **XLA:GPU already autotunes**, choosing among cuBLAS, cuBLASLt, cuDNN, and Triton per dot.
  A gljax-level search would sit on top of an existing search and mostly measure noise.
* ARTX5/ARTX7's compile cache means each shape is compiled once and reused thousands of times.
  A *compile-time* sweep is compatible with that; a *runtime* search is not — it would invalidate
  the "compile once, execute many" invariant that ARTX7 is built on.
* A gljax-level autotuner would have to time candidates through the PJRT boundary, where each sample
  includes transfer, launch, and synchronization cost that the intra-XLA autotuner measures without.
  It would be a noisier instrument making the same decision, one layer further from the kernel.

---

# Proposed Architecture

```text
model code  (Tensor::matmul, Tensor::matmul_with)
     │
     ▼
ops/{attention,ffn,moe}.rs          ARTX3 — semantic layer
     │
     ▼
┌─────────────────────────────────────────────────────┐
│  matrix/                          ARTX8 — NEW       │
│                                                     │
│   spec.rs      Contraction kind + operand shapes    │
│      │         (pure data)                          │
│      ▼                                              │
│   plan.rs      MatmulPlan: dnums + numerics +       │
│      │         fusion segments  (pure data)         │
│      ├────────► roofline.rs   regime label          │
│      │                        (analysis only)       │
│      ▼                                              │
│   fusion.rs    structural weight concat + offsets   │
│      │                                              │
│      ▼                                              │
│   lower.rs     ONLY place dot_general is emitted    │
└──────┬──────────────────────────────────────────────┘
       ▼
graph/builder.rs  FuncBuilder::dot_general            ARTX2
       ▼
stablehlo/ops.rs  emit_dot_general                    ARTX2
       ▼
   XLA:  fusion → layout assignment → library selection / autotune
       ▼
   PJRT execute
```

Everything inside the box is host-side Rust with no PJRT dependency except `lower.rs`, which needs
only `FuncBuilder`.

## Wave breakdown

| Wave | Scope | Gate |
|---|---|---|
| **A8.α** | Plumb numerics: parameterize `emit_dot_general`'s `precision_config`, add `algorithm` + `preferred_element_type`, fix output-dtype inference. **Prerequisite for everything else.** | FP64 oracle (ARTX1 §3.4) agreement unchanged or improved on one attention block |
| **A8.1** | `spec.rs` + `plan.rs` + `roofline.rs` — planning only, no emission changes | Unit tests; every existing `dot_general` call site reproduces its current dnums through the planner |
| **A8.2** | `lower.rs` — route all emission through one place; migrate `ops/` call sites | Emitted MLIR byte-identical to pre-migration for an unfused model |
| **A8.3** | `fusion.rs` — QKV + gate/up structural fusion, segment offsets for ARTX6 | Numerics match unfused within oracle tolerance; ARTX6 TP sharding still valid |

⚠️ A8.α is not optional sequencing. Landing fusion (A8.3) before the numerics are stateable means any
numeric regression it causes is undiagnosable — you cannot tell a fusion bug from an accumulation
difference when accumulation is whatever the backend picked.

---

# API Design

Constraint: *keep APIs small and stable*. The public surface added by ARTX8 is **one struct, one
enum, one method, and one free function.** Everything else is internal.

## Public surface

```rust
// src/matrix/mod.rs

/// Numeric contract for a single contraction.
/// `Default` reproduces gljax's current behavior exactly (precision DEFAULT, no algorithm).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DotNumerics {
    /// precision_config = [DEFAULT, DEFAULT]. What gljax emits today.
    #[default]
    Default,
    /// precision_config = [HIGHEST, HIGHEST]. Backend emulates wider dtype.
    Highest,
    /// Explicit algorithm, e.g. Bf16Bf16F32. Mutually exclusive with precision_config
    /// per the StableHLO spec — the emitter enforces this.
    Algorithm(DotAlgorithm),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DotAlgorithm {
    Bf16Bf16F32,
    Bf16Bf16F32X3,
    Bf16Bf16F32X6,
    Tf32Tf32F32,
    Tf32Tf32F32X3,
    F32F32F32,
    F64F64F64,
}

/// Options for a single matmul. All fields have defaults; `MatmulOpts::default()`
/// is byte-identical to today's `Tensor::matmul`.
#[derive(Debug, Clone, Default)]
pub struct MatmulOpts {
    pub numerics: DotNumerics,
    /// Recommended accumulation type. `None` = leave to backend (current behavior).
    pub accumulate: Option<DType>,
}

impl Tensor {
    /// Unchanged from ARTX2. Equivalent to `matmul_with(rhs, MatmulOpts::default())`.
    pub fn matmul(&self, rhs: &Tensor) -> Tensor;

    /// The one new method.
    pub fn matmul_with(&self, rhs: &Tensor, opts: MatmulOpts) -> Tensor;
}

/// Structural fusion of N weight matrices sharing one activation.
/// Returns the fused result plus the segment boundaries needed to split it
/// (and needed by ARTX6 to shard the fused weight along head boundaries).
pub fn fused_projection(
    x: &Tensor,
    weights: &[&Tensor],
    opts: MatmulOpts,
) -> (Tensor, Vec<Segment>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment { pub start: usize, pub len: usize }
```

That is the whole public API. `Tensor::matmul`'s signature does not change; every existing call site
keeps compiling and emitting identical MLIR.

## Internal types

```rust
// src/matrix/spec.rs — pure data, no builder, no IR

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Contraction {
    /// [1, K] × [K, N] — decode projection. AI ≈ 1.
    Gemv,
    /// [M, K] × [K, N], M > 1 — prefill projection. AI ≈ O(M).
    Gemm,
    /// [B, M, K] × [B, K, N] — attention across heads.
    BatchedGemm { batch_rank: usize },
    /// [m, k] × [g, k, n] + group_sizes — MoE. Reserved; see Future Extensions.
    Ragged { groups: usize },
}

#[derive(Debug, Clone)]
pub struct MatmulSpec {
    pub lhs: Shape,
    pub rhs: Shape,
    pub kind: Contraction,
}

impl MatmulSpec {
    /// Classify from shapes alone. Total, no failure mode.
    pub fn classify(lhs: &Shape, rhs: &Shape, dnums: &DotDimensionNumbers) -> Contraction;
    /// FLOPs for this contraction (2·M·N·K·batch).
    pub fn flops(&self) -> u64;
    /// Bytes that must move, assuming each operand is read once.
    pub fn bytes(&self) -> u64;
    /// FLOPs ÷ bytes.
    pub fn arithmetic_intensity(&self) -> f64;
}
```

```rust
// src/matrix/plan.rs

#[derive(Debug, Clone)]
pub struct MatmulPlan {
    pub spec: MatmulSpec,
    pub dnums: DotDimensionNumbers,
    pub numerics: DotNumerics,
    pub accumulate: Option<DType>,
    pub out_shape: Shape,
    /// Non-empty only for structurally fused projections. Preserved for ARTX6 sharding.
    pub segments: Vec<Segment>,
}
```

```rust
// src/matrix/roofline.rs — analysis only. Never changes emission.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Regime { BandwidthBound, ComputeBound, Balanced }

#[derive(Debug, Clone, Copy)]
pub struct DeviceCeilings { pub peak_flops: f64, pub peak_bw_bytes_per_s: f64 }

/// Label a plan's regime against a device's ridge point.
/// Reference ridge points: TPU v5e ~241, A100 ~153, H100 ~295 FLOP/byte.
///
/// ⚠️ This is an UPPER BOUND, not a prediction. Roofline assumes the selected kernel
/// sustains the throughput its intensity implies; real kernels reach ~80-85% of peak
/// on H100 and ~95% on TPU even when compute-bound. Use this to EXPLAIN a measurement,
/// never to substitute for one, and never to select a code path (see roofline.rs's
/// module invariant).
pub fn classify(plan: &MatmulPlan, dev: DeviceCeilings) -> Regime;
```

---

# Module Responsibilities

| Module | Owns | Must NOT know |
|---|---|---|
| `matrix/spec.rs` | Shape → contraction classification; FLOPs, bytes, AI | `FuncBuilder`, `Tensor`, PJRT, dtypes beyond byte size |
| `matrix/plan.rs` | Assembling dnums + numerics + output shape into one plan | How anything is emitted |
| `matrix/roofline.rs` | Regime labeling against device ceilings | Emission; it may never influence what is emitted |
| `matrix/fusion.rs` | Weight concatenation, segment offsets, split-after-dot | Numerics policy, dnums construction |
| `matrix/lower.rs` | The **only** call to `FuncBuilder::dot_general` in gljax | Why a plan looks the way it does |

The invariant that makes this worth having: **`lower.rs` is the single choke point.** After A8.2, a
grep for `dot_general` in gljax outside `matrix/lower.rs`, `graph/builder.rs`, and
`stablehlo/ops.rs` should return nothing. That is the property that makes A8.α's numerics change a
one-line policy edit rather than a four-file migration.

⚠️ **`roofline.rs` must never feed back into emission.** It is a labeling and explanation tool. The
moment a regime label changes what gets emitted, gljax has a runtime kernel-selection heuristic —
which the brief excludes, and which this repo has three measurements saying it would get wrong.

---

# Folder Structure

```text
src/
├── stablehlo/
│   ├── types.rs        (ARTX2) + DotAlgorithm, DotPrecision enums
│   └── ops.rs          (ARTX2) emit_dot_general — MODIFIED by A8.α
├── graph/
│   └── builder.rs      (ARTX2) FuncBuilder::dot_general — MODIFIED by A8.α
├── matrix/             ← NEW (ARTX8)
│   ├── mod.rs          pub use; MatmulOpts, DotNumerics, fused_projection
│   ├── spec.rs         Contraction, MatmulSpec, classify, flops/bytes/AI
│   ├── plan.rs         MatmulPlan
│   ├── fusion.rs       structural weight concat + Segment
│   ├── lower.rs        plan + operands → SsaValue  (sole dot_general call site)
│   └── roofline.rs     Regime, DeviceCeilings, classify
├── ops/                (ARTX3) — migrated to call matrix::
│   ├── attention.rs    → matrix::fused_projection for QKV
│   ├── ffn.rs          → matrix::fused_projection for gate+up
│   └── moe.rs          → matrix::lower (Ragged reserved)
├── tp/                 (ARTX6) — consumes MatmulPlan::segments for sharding
└── runtime/            (ARTX4/5/7)
```

Five new files. No new dependencies.

---

# Execution Flow

```text
TRACE TIME (host, single-threaded, no device)

  ops/ffn.rs: swiglu_ffn(x, w_gate, w_up, w_down)
      │
      ├─► matrix::fused_projection(x, &[w_gate, w_up], opts)
      │       │
      │       ├─► fusion.rs: concat w_gate[D,F] ++ w_up[D,F] → [D,2F]
      │       │              segments = [{0,F}, {F,F}]
      │       │
      │       ├─► spec.rs: classify([1,D], [D,2F]) → Gemv     (decode)
      │       │            classify([S,D], [D,2F]) → Gemm     (prefill)
      │       │
      │       ├─► plan.rs: MatmulPlan { dnums, numerics, accumulate, segments }
      │       │
      │       └─► lower.rs: FuncBuilder::dot_general(...) → %gate_up [.., 2F]
      │
      ├─► slice %gate_up by segments → %gate, %up
      ├─► %gate.silu() * %up            ← XLA fuses this epilogue. gljax emits it plainly.
      └─► matrix::lower(down)           → %out

COMPILE TIME (XLA, once per bucket — ARTX5/ARTX7 compile cache)

  StableHLO module
      → fusion passes        (epilogue: dot + logistic + multiply → one kernel)
      → layout assignment    (minor-to-major propagation; conflicts → copy ops)
      → library selection    (GPU: cuBLAS/cuBLASLt/cuDNN/Triton, autotuned)
      → compiled executable → CompileCache

RUN TIME (PJRT)

  execute(compiled, inputs)     ← gljax contributes nothing here
```

The middle box is the point of the whole document: **compile time is where matrix performance is
decided, and gljax's only influence on it is the module it hands over.** Shapes, layouts, dtypes,
and how many separate dots there are. Nothing else.

---

# Memory Layout

## What gljax controls

| Thing | Controlled by | Notes |
|---|---|---|
| Logical weight shape (`[D,N]` vs `[N,D]`) | gljax (checkpoint loader + trace) | Determines contracting dims. Pick one convention and hold it. |
| Whether QKV / gate+up are one array or three | gljax (`fusion.rs`) | The structural fusion decision. |
| dtype of weights on device | gljax (checkpoint loader) | bf16 default per ARTX2's `PrecisionPolicy`. |
| Physical layout (minor-to-major) | **XLA** | Layout assignment. gljax does not set it. |
| Packing into micro-kernel panels | **XLA / the selected library** | Never expressed in gljax IR. |

## The rule that follows

> Emit the natural graph. Do not insert transposes or reshapes to "help" the backend.

XLA propagates layouts through the graph and materializes conflicts as physical copy operations. A
transpose added by gljax to make a matmul "look right" is at best a no-op that layout assignment
elides, and at worst a materialized copy plus a fusion that now has mismatched input/output layouts —
a documented severe-regression pattern in XLA:GPU, where fusion kernels whose I/O tensors carry
different layouts get bad data locality and later fusion passes do not correct for it.

The asymmetry that makes this rule one-directional: layout assignment can *remove* a redundant
transpose gljax emitted, but it cannot *undo* a layout conflict gljax created between two ops it
decided to place. Emitting less is strictly safer than emitting more.

## Fused weight layout

```text
QKV fused weight, GQA (Hq query heads, Hkv KV heads, head_dim Dh):

  [D, Hq·Dh + Hkv·Dh + Hkv·Dh]
   └───────┬──────┘└────┬────┘└────┬────┘
        segment 0   segment 1  segment 2
           (Q)         (K)        (V)

  segments = [{0, Hq·Dh}, {Hq·Dh, Hkv·Dh}, {Hq·Dh + Hkv·Dh, Hkv·Dh}]
```

⚠️ ARTX6 constraint: a column-sharded TP split of this matrix must fall on head boundaries within
each segment, never across a segment boundary. `MatmulPlan::segments` exists precisely so
`tp/linear.rs` can enforce that rather than rediscovering it.

Concatenation happens **once at checkpoint load**, not per trace — the fused array is what lives on
device. This costs `Hkv·Dh·D·2` extra bytes of host memory transiently during load and zero extra
bytes on device.

---

# Kernel Pipeline

Read this as "who does what," since gljax's row is deliberately short.

| Stage | Owner | gljax's contribution |
|---|---|---|
| Contraction classification | gljax | `spec.rs` |
| Structural fusion | gljax | `fusion.rs` |
| Numeric contract | gljax | `DotNumerics` → `precision_config` / `algorithm` |
| Dimension numbers | gljax | `plan.rs` |
| **Epilogue fusion** | XLA | emit the plain graph |
| **Layout assignment** | XLA | avoid conflicts |
| **Tiling / blocking** | XLA + library | none |
| **Packing** | library | none |
| **Micro-kernel / MMA selection** | library / hardware | none |
| **Thread & block decomposition** | library / hardware | none |
| **Autotuning** | XLA | none |
| Execution | PJRT | none |

---

# Pseudocode

## Contraction classification

```rust
// matrix/spec.rs
pub fn classify(lhs: &Shape, rhs: &Shape, dnums: &DotDimensionNumbers) -> Contraction {
    let batch_rank = dnums.lhs_batching.len();
    if batch_rank > 0 {
        return Contraction::BatchedGemm { batch_rank };
    }
    // The single free (non-batch, non-contracting) dim of lhs is M.
    let m = lhs.dims.iter().enumerate()
        .filter(|(i, _)| !dnums.lhs_contracting.contains(i))
        .map(|(_, &d)| d)
        .product::<usize>();
    if m == 1 { Contraction::Gemv } else { Contraction::Gemm }
}

pub fn arithmetic_intensity(&self) -> f64 {
    self.flops() as f64 / self.bytes() as f64
}
```

## Structural fusion

```rust
// matrix/fusion.rs
pub fn fused_projection(
    x: &Tensor, weights: &[&Tensor], opts: MatmulOpts,
) -> (Tensor, Vec<Segment>) {
    assert!(!weights.is_empty());
    // All weights must share the contracting dim (D) and dtype.
    let d = weights[0].shape().dims[0];
    assert!(weights.iter().all(|w| w.shape().dims[0] == d));
    assert!(weights.iter().all(|w| w.dtype() == weights[0].dtype()));

    let mut segments = Vec::with_capacity(weights.len());
    let mut offset = 0usize;
    for w in weights {
        let len = w.shape().dims[1];
        segments.push(Segment { start: offset, len });
        offset += len;
    }

    // Concat along the output (column) dim. In practice this is done ONCE at
    // checkpoint load; at trace time the fused array is already a single weight.
    let w_fused = concat_columns(weights);              // [D, offset]
    let out = matrix::lower::emit(x, &w_fused, opts);   // one dot_general
    (out, segments)
}

// Caller splits:
//   let (gate_up, segs) = fused_projection(x, &[w_gate, w_up], opts);
//   let gate = gate_up.slice_dim(-1, segs[0]);
//   let up   = gate_up.slice_dim(-1, segs[1]);
//   let h    = gate.silu() * up;     // ← XLA fuses. Do not hand-roll.
```

## Lowering (the sole emission point)

```rust
// matrix/lower.rs
pub fn emit(lhs: &Tensor, rhs: &Tensor, opts: MatmulOpts) -> Tensor {
    let dnums = plan::default_dnums(lhs.shape(), rhs.shape());
    let plan  = MatmulPlan::build(lhs.shape(), rhs.shape(), dnums, opts);

    let mut b = lhs.builder().borrow_mut();
    let v = b.dot_general_with(
        lhs.value(), rhs.value(),
        &plan.dnums,
        plan.numerics,        // → precision_config OR algorithm, never both
        plan.accumulate,      // → preferred_element_type, omitted if None
    );
    drop(b);
    Tensor::new(v, lhs.builder_rc())
}
```

## A8.α — the numerics plumbing fix

```rust
// stablehlo/ops.rs — emit_dot_general, MODIFIED
//
// BEFORE (ARTX2 ~line 603): hardcoded, unreachable.
//   e.line(r#"precision_config = [#stablehlo<precision DEFAULT>, ...]"#.to_string());
//
// AFTER:
match numerics {
    DotNumerics::Default => e.line(
        r#"precision_config = [#stablehlo<precision DEFAULT>, #stablehlo<precision DEFAULT>]"#.into()),
    DotNumerics::Highest => e.line(
        r#"precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]"#.into()),
    // Spec: algorithm and precision_config are MUTUALLY EXCLUSIVE.
    DotNumerics::Algorithm(alg) => e.line(format!("algorithm = {}", alg.mlir_str())),
}
if let Some(acc) = accumulate {
    e.line(format!("preferred_element_type = {}", acc.mlir_str()));
}
```

```rust
// graph/builder.rs — infer_dot_general_shape, MODIFIED
//
// BEFORE: Shape::new(out_dims, lhs.dtype)   // always lhs dtype
// AFTER:  accumulate.unwrap_or(lhs.dtype)   // honor preferred_element_type
Shape::new(out_dims, accumulate.unwrap_or(lhs.dtype))
```

---

# Tradeoffs

| Decision | Gain | Cost | Why accepted |
|---|---|---|---|
| No kernel layer | Zero per-backend kernel maintenance; portable across every PJRT plugin | Cannot beat cuBLASLt/MXU on any specific shape | Beating vendor libraries on their own silicon is not the goal, and each new accelerator generation would re-open the whole innermost loop |
| Single `lower.rs` choke point | Numerics/policy changes are one-line | One more indirection between `ops/` and `FuncBuilder` | The four-call-site duplication it removes is a real, current bug surface |
| Structural fusion by construction | Fewer, larger dots; one contiguous HBM stream; better MXU/threadblock occupancy | Fused weights complicate TP sharding; needs `segments` metadata | ARTX1 §7 already recommends it for gate/up; XLA cannot do it (weights arrive as separate arrays) |
| Roofline as labeling only | Explains measurements without becoming a heuristic | Doesn't automatically pick anything | Roofline is an upper bound; real kernels hit 80–95% of it, and the gap is exactly where an automated decision would go wrong |
| `MatmulOpts::default()` == today's behavior | Zero-risk migration; existing MLIR unchanged | Two ways to call matmul | Keeps the API additive; the constraint was small *and stable* |
| Deferring `ragged_dot` | No plugin-support risk taken now | MoE keeps capacity-factor padding for now | Interacts non-trivially with ARTX5/7 static shapes; needs its own wave |

---

# Rejected Alternatives

Each of these was considered and rejected with a reason, not merely omitted.

### 1. A BLIS-style micro-kernel layer in gljax

**Rejected.** For gljax's cloud targets a microkernel is not even expressible in portable code — it
means Pallas/Mosaic for TPU and CUDA or Triton for NVIDIA, contradicting ARTX1's pure-Rust
plugin-only commitment. It would have to beat cuBLASLt and the XLA TPU compiler on their own
silicon, and be rewritten each accelerator generation (Hopper `wgmma` → Blackwell `tcgen05.mma`
changed the innermost loop's thread scope outright). See [A8.2](#wave-a82--efficient-gemm-architectures).

### 2. Emitting `custom_call` to hand-written kernels (Pallas / CUDA / Triton)

**Rejected by the brief's constraints** (no Triton DSL, no custom CUDA codegen), and independently by
portability: a custom_call must be registered per backend, so one kernel becomes N kernels. ARTX1 §7
records `@flash_attention` as the mechanism if this is ever revisited — as a deliberate, scoped,
one-backend-at-a-time decision, exactly like ARTX7 deferred PagedAttention.

### 3. Runtime kernel search / autotuning in gljax

**Rejected by the brief**, and redundant: XLA:GPU already autotunes across cuBLAS/cuBLASLt/cuDNN/
Triton. A runtime search also breaks ARTX7's "compile once, execute many" invariant.

### 4. Expressing packing in gljax IR (IREE `mmt4d` / `tensor.pack` style)

**Rejected.** That is IREE's model, not XLA's. XLA reasons about layouts, and packing lives inside
the selected library. Emitting pack-shaped IR would fight layout assignment and materialize copies.

### 5. Hand-rolling epilogue fusions (bias, SwiGLU) as single ops

**Rejected.** XLA already fuses `dot + logistic + multiply` on both GPU and TPU (ARTX1 §7). Building
it in gljax duplicates a compiler pass and risks producing a graph shape the pass no longer
recognizes.

### 6. A quantized (int4/int8) weight compute path

**Rejected for this document; not rejected permanently — and it looks *favourable* on gljax's
targets.** Quantized weights pay off exactly when bandwidth binds before the kernel's compute
ceiling, and A8.1 shows low-batch decode on TPU v5e / A100 / H100 sits 200–600× below the ridge
point — deeply bandwidth-bound, which is the regime where cutting weight bytes wins. GwenLand's own
GPU engine (`glcuda`) confirms the direction: its native Q4_K kernel wins there.

It is deferred here only because it is a *separate concern* from this document's scope: it needs a
plugin-support survey (which quantized dtypes each PJRT plugin accepts), a checkpoint-format
decision, and its own measurements on real devices. It should not be smuggled in as a side effect of
the matrix layer. See [Future Extensions](#future-extensions).

### 7. A `Matmul` trait with per-backend implementations

**Rejected.** Classic over-engineering for this problem: there is exactly one implementation
(`dot_general`) and the backend variation is handled by XLA, below gljax's floor. A trait here would
have one implementor forever.

---

# Future Extensions

| Extension | Blocked on |
|---|---|
| **`ragged_dot` for MoE** — removes capacity-factor padding for expert FFN | Per-plugin support verification; interaction with ARTX5/ARTX7 static-shape bucketing |
| **Quantized dot (fp8 / int8)** — `DotAlgorithm` already has the vocabulary; A8.1 shows decode is deeply bandwidth-bound, the regime where cutting weight bytes pays | Plugin support survey (which quantized dtypes each PJRT plugin accepts); checkpoint-format decision; measurement on a real TPU/GPU device |
| **Compile-time algorithm sweep** — try `bf16_bf16_f32` vs `_x3` vs `_x6` once per bucket, cache the winner | A8.α landed; must stay compile-time (a runtime search breaks ARTX7's invariant) |
| **Flash attention via `custom_call`** | Explicit decision to take on per-backend kernels; ARTX1 §7 has the mechanism |
| **HLO-after-optimization assertions** — verify XLA actually fused what we assumed | PJRT hook for dumping post-optimization HLO |
| **Batch-size ridge-point reporting** — feed `roofline.rs` regime labels into ARTX7's scheduler telemetry so "does batching help this stage?" is answerable | ARTX7 Wave A7.2 landed |

---

# Success Criteria Check

| Criterion (from the brief) | How this architecture meets it |
|---|---|
| Compiler-friendly | Emits only standard `dot_general`; expresses no tiling, packing, or scheduling that XLA would have to undo |
| Backend-agnostic | Zero backend-specific code paths; hardware differences are absorbed by the `DotAlgorithm` / accumulation contract |
| Maps efficiently to StableHLO/PJRT | `dot_general` is the native primitive; structural fusion reduces dispatch count; layout is left to layout assignment |
| Extensible without breaking APIs | `MatmulOpts` is additive with a `Default` equal to current behavior; `Contraction::Ragged` reserved |
| Avoids over-engineering | 5 new files, 1 new public method, 1 new free function, 0 new dependencies, 0 traits |
| Separates planning / lowering / execution | `plan.rs` (pure data) → `lower.rs` (sole emitter) → PJRT (gljax contributes nothing) |
| Treats GEMM as the primary primitive | Accepted **with the A8.1 correction**: GEMM is the primary *primitive*, but for single-stream decode it is not the primary *bottleneck* — GEMV bandwidth is. `roofline.rs` exists to keep that distinction visible instead of assumed |

---

# Summary

gljax's matrix compute architecture is a **planning and lowering layer, not a kernel library.** It
owns four decisions — structural fusion, numeric contract, dimension numbers, and regime labeling —
and delegates everything below `stablehlo.dot_general` to XLA, which already has cuBLASLt, cuDNN,
Triton, and the TPU compiler behind it.

The highest-value work in this document is the smallest: **A8.α**, plumbing `precision_config`,
`algorithm`, and `preferred_element_type` through `emit_dot_general`. Today gljax cannot state what
numerics it wants from a matmul, in a project that built an FP64 oracle specifically to validate
numerics. Everything else in ARTX8 is structure; that one is a gap.
