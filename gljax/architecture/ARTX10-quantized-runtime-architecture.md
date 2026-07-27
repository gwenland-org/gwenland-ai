# ARTX10 — Quantized Runtime Architecture (FP8 / INT8 / INT4 / GQ4A)

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX8 (matrix compute — **binding**), ARTX09 (attention & memory — quantized KV), ARTX16 §6 (serving quantization posture), ARTX12 (runtime conformance — **gating**)
**Introduces:** `gljax/src/quant/`
**Next:** [ARTX11 — Speculative Inference](ARTX11-speculative-decoding.md)
**Research grounded:** 2026-07-27 (sources at end)

---

# 0. The Question This Document Was Asked to Settle

Wave A12.5 in the brief is titled **"Kernel Dispatch."** ARTX8's central holding is:

> **gljax does not own a GEMM kernel, and must not acquire one.**

Those appear to collide. The decision was deferred to research. **§5 settles it**, and the answer is
neither "yes, build kernels" nor "no, forget quantization" — it turns on an asymmetry that the
research makes sharp:

> **The quantization that helps decode is bandwidth-saving weight-only quantization. That only pays
> if the weights stay packed *through* the matmul. Whether they do is a property of the PJRT plugin,
> not of gljax — and gljax cannot know it without probing.**

So ARTX10's architecture is built around a **capability probe**, not a compile-time assumption.
ARTX8's no-kernel rule holds unchanged; what gljax gains is the ability to *emit* quantized StableHLO
and to *measure* whether the backend honoured it.

## 0.1 Scope

**In:** quantization survey · runtime architecture · backend abstraction · dispatch API · memory
layout · pseudocode.

**Out (per the brief):** quantization-aware training · calibration pipelines · training algorithms ·
model conversion tools.

⚠️ That out-of-scope list has a consequence worth naming: **AWQ, GPTQ, and SmoothQuant are
calibration algorithms.** They are studied here for the *artifact* they produce — what gljax must be
able to load and execute — never for how the artifact is produced. §1.3 reads them as format
specifications, not as methods.

## 0.2 Prerequisites that genuinely block

| Blocker | Why |
|---|---|
| **ARTX8 Wave A8.α** | `precision_config` is hardcoded and `preferred_element_type` is dropped entirely. Quantization is *entirely* a statement about numeric contracts. Without A8.α gljax cannot say what it wants. |
| **ARTX12 T0+T2** | Quantization changes numerics by design. Without a correctness harness there is no way to distinguish "quantized as intended" from "silently corrupted" — the exact bug class that already shipped here once as the Q6_K dequant-order defect. |

---

# 1. Wave A12.1 — Quantization Fundamentals

## 1.1 The two axes that actually matter

Almost every practical decision falls out of one table:

| Scheme | What is quantized | Win | Helps which phase |
|---|---|---|---|
| **W4A16 / W8A16** (weight-only) | Weights only; activations stay BF16 | **Bandwidth** — fewer weight bytes from HBM | **Decode** |
| **W8A8** (weight + activation) | Both | **Compute** — INT8/FP8 MMA at higher rate | **Prefill** |
| **KV quantization** | The KV cache | **Capacity** — longer context / more slots | Both, at long context |

⭐ **The load-bearing consequence, straight from ARTX8's measurement.** ARTX8 established that
low-batch decode runs at arithmetic intensity 0.5–2 FLOP/byte against ridge points of 241 (TPU v5e),
153 (A100), 295 (H100) — 200–600× below compute-bound. Decode is starved for *bytes*, not for FLOPs.

Therefore:

* **Weight-only quantization is the one that helps decode.** Halving weight bytes roughly halves
  decode time in the bandwidth-bound regime.
* **W8A8 helps prefill, and barely touches decode.** TPU v5e does 393 INT8 TOPS against 197 BF16
  TFLOP/s — a genuine **~2×** compute uplift — but doubling compute on a workload running at 0.4% of
  its compute ceiling changes nothing.

⚠️ This is why the brief's source list needs re-ranking for gljax's purposes: SmoothQuant is a W8A8
method and therefore a *prefill* optimization; AWQ and GPTQ are weight-only and therefore *decode*
optimizations. For an interactive serving engine, the second pair matters more.

## 1.2 Number formats

### FP8 — E4M3 and E5M2

| Format | Bits | Range | Typical use |
|---|---|---|---|
| `f8E4M3FN` | 4 exp / 3 mant | narrower | **weights, activations, KV** (precision-favoured) |
| `f8E5M2` | 5 exp / 2 mant | wider | gradients (training) |

StableHLO defines both plus the `FNUZ` variants (`f8E4M3FNUZ`, `f8E5M2FNUZ`, `f8E4M3B11FNUZ`).
For inference, **E4M3 is the relevant one** — and for the KV cache specifically, E4M3 dominates
because K and V dynamic range is bounded by the softmax that consumes them.

### Block-scaled integer formats — and a convergence worth naming

⭐ Three independently-developed families landed on the *same shape*: narrow values plus a shared
per-block scale.

| Format | Value | Block | Scale |
|---|---|---|---|
| **llama.cpp Q4_K** | 4-bit | 256-element superblock ÷ eight 32-element blocks | 6-bit per-block scale + 16-bit superblock scale & min |
| **OCP MXFP8** | FP8 E4M3 | 32 elements | E8M0 (pure power-of-2, no sign, no mantissa, 2⁻¹²⁷…2¹²⁷) |
| **Pridwen GQ4A** | 4-bit, zero-centered `dequant(c) = c − 8` | per-block | per-block `scale_i` |
| **Pridwen GQ2A** | 2-bit, asymmetric | per-block | `min_i` / `scale_i` + `super_scale` |

⚠️ **GwenLand's own Pridwen format is a member of the microscaling family.** GQ2A's
`super_scale` over per-block `min_i`/`scale_i` is structurally the same hierarchy as Q4_K's
superblock-over-block scales. That is not a coincidence — it is the shape that block quantization
converges on. Practically it means **Pridwen is not exotic**: anything that can execute a Q4_K-shaped
or MX-shaped dequant can execute Pridwen.

⚠️ It also means the reverse: Pridwen inherits the family's core problem — **the dequant is
arithmetic that must happen somewhere**, and where it happens decides whether the format saves
bandwidth or merely saves disk (§5).

### StableHLO's own quantized types

StableHLO has first-class quantization, inheriting its type expression from MLIR's Quant dialect and
following the LiteRT quantization spec:

```text
real_value = scale × (quantized_value − zero_point)
```

with **per-tensor** and **per-axis** schemes, the ops `uniform_quantize` / `uniform_dequantize`, and
— importantly for gljax — the per-axis scheme was added **to `dot_general` specifically**.

⚠️ So gljax *can* express a quantized matmul in portable StableHLO without any custom kernel. Whether
that expression is fast is §5's question.

## 1.3 The calibration algorithms, read as format specifications

Per §0.1, these are surveyed for what gljax must *load*, not for how to run them.

| Method | Mechanism | Artifact gljax must consume |
|---|---|---|
| **GPTQ** | OBQ-derived; Cholesky of the inverse Hessian; layer-wise, arbitrary fixed order; compensates residual error into not-yet-quantized weights. 3–4 bit weight-only (OPT-175B at 3-bit on one 80 GB A100) | INT4/INT3 weights + per-group scales (+ zero-points if asymmetric) |
| **AWQ** | Activation-aware: finds salient channels from a small calibration set's *activation* distribution, applies a per-channel scale so naive INT4 rounding suffices. Faster to calibrate than Hessian methods | INT4 weights + **per-channel scales**, group size typically 128 |
| **SmoothQuant** | Migrates quantization difficulty from activations to weights offline via `s_j = max(\|X_j\|)^α / max(\|W_j\|)^(1−α)`; training-free; enables W8A8 | INT8 weights + INT8 activation scales + **the smoothing vector folded into the preceding norm** |

⚠️ **SmoothQuant's artifact has a structural consequence the other two do not.** Its smoothing factor
is mathematically folded into the *previous* layer's normalization weights. A gljax checkpoint loader
that assumes RMSNorm weights are the model's original weights will load a SmoothQuant checkpoint and
produce fluent garbage — ARTX12 failure class exactly. It is a **loader** concern, not a kernel one.

⚠️ **GPTQ vs AWQ, for gljax's purposes, differ only in group granularity.** Both emit "INT4 weights +
scales at some granularity." From the runtime's view they are one format with a parameter. gljax does
not need two code paths; it needs `group_size` and `symmetric: bool`.

---

# 2. Wave A12.2 — Serving Architectures

## 2.1 Where dequantization can happen

There are exactly three places, and the choice determines everything:

```text
┌── (A) LOAD TIME, HOST ────────────────────────────────────────┐
│  Q4 on disk → dequant on CPU → BF16 on device                 │
│  HBM traffic: BF16.  Saves: disk + network.  Saves HBM: NO    │
│  gljax support: TRIVIAL — this is ARTX16 §6.1 today            │
└───────────────────────────────────────────────────────────────┘
┌── (B) IN-GRAPH, DEVICE ───────────────────────────────────────┐
│  Q4 in HBM → uniform_dequantize → BF16 → dot_general          │
│  HBM traffic: Q4 for the WEIGHT READ.  Saves HBM: YES*        │
│  * only if XLA FUSES the dequant into the dot's prologue      │
│  gljax support: emit standard StableHLO. Fusion NOT guaranteed│
└───────────────────────────────────────────────────────────────┘
┌── (C) INSIDE THE KERNEL ──────────────────────────────────────┐
│  Q4 stays packed into the MMA; dequant in registers           │
│  HBM traffic: Q4.  Saves HBM: YES, guaranteed                 │
│  gljax support: NONE — requires a custom kernel (ARTX8 ⛔)     │
└───────────────────────────────────────────────────────────────┘
```

⚠️ **(A) is what gljax does today and it saves no HBM bandwidth.** ARTX16 §6.1 already reached this
conclusion and stated it plainly: *"Pridwen is a storage/distribution format for gljax, not a compute
format."* ARTX10 does not overturn that — it identifies **(B)** as the path that could, and specifies
how to find out whether a given plugin delivers it.

⚠️ **(B) is the entire prize, and it is conditional.** If XLA fuses `uniform_dequantize → dot_general`
into the matmul's prologue, the weight read from HBM is Q4-sized and decode gets the bandwidth win.
If XLA instead materializes a BF16 tensor first, (B) degenerates to (A) plus extra work — *worse*
than doing nothing. There is no middle outcome, and the difference is invisible from gljax's side
without measurement.

## 2.2 What the reference implementations do

| System | Approach | Transferable to gljax? |
|---|---|---|
| **llama.cpp** | Custom per-ISA kernels; weights stay packed; dequant in registers → path (C) | ❌ Owns kernels |
| **TensorRT-LLM** | Custom CUDA/CUTLASS kernels per quantization scheme → path (C) | ❌ Owns kernels |
| **vLLM** | Kernel per scheme (Marlin, Machete, …) → path (C) | ❌ Owns kernels |
| **DeepSpeed** | Custom kernels + blocked KV → path (C) | ❌ Owns kernels |
| **CUTLASS** | *Is* the kernel library others build on | ❌ Is the thing ARTX8 declines to own |

⛔ **Every production quantized-serving stack takes path (C).** That is the honest headline of this
survey and it must not be softened: gljax is choosing a path the ecosystem does not use, because
ARTX1 chose to be a portable plugin-only client and ARTX8 confirmed the trade.

⚠️ The consequence is a **bounded expectation**: gljax should not plan to match vLLM's quantized
throughput. It should plan to get whatever (B) yields on each plugin, measure it, and fall back to
(A) — which is always correct and never slower than BF16 — when (B) does not materialize.

---

# 3. Wave A12.3 — Mixed Precision Execution

## 3.1 The precision ladder

```text
Weights      Q4 / Q8 / FP8 / BF16        ← where the bandwidth win lives
Activations  BF16 (default) / FP8        ← where the compute win lives
Accumulation FP32, always                ← ARTX8: MXU and Tensor Cores both accumulate FP32
KV cache     BF16 / FP8-E4M3 / INT8      ← where the capacity win lives  (§4)
Logits       FP32                        ← softmax stability
```

⚠️ **DESIGN DECISION — accumulation is FP32 unconditionally, at every quantization level.**
ARTX8 recorded that both the TPU MXU and NVIDIA Tensor Cores multiply narrow and accumulate in FP32,
and ARTX12 §3.2 derived the error model on exactly that assumption (`δ ≈ 2·u_input + K·u_fp32`).
Quantizing the accumulator would invalidate ARTX12's entire tolerance table and reintroduce a
K-dependent error term. There is no quantization scheme worth that.

## 3.2 FP8's lowering is not a dtype swap

⚠️ Recorded in ARTX16 §6.3 and re-confirmed: XLA's FP8 Dot lowering **casts the FP8 inputs up, applies
input scales, runs the Dot at the wider type, computes an output scale via a reduction, and casts
back down — with the whole sequence fused** so the wide Dot is never materialized.

Three consequences that must not be discovered late:

1. **FP8 requires calibrated scales.** Those come from a calibration pipeline, which is out of scope
   (§0.1). gljax consumes scales from the checkpoint; it never produces them.
2. **The win depends on the fusion firing**, structurally the same conditionality as path (B) in §2.1.
3. **TPU v5e FP8 support is unconfirmed.** v5e's published figures are 197 BF16 TFLOP/s and 393 INT8
   TOPS; FP8 does not appear among them. **Do not assume FP8 portability across gljax's targets.**

## 3.3 Where the hardware actually rewards narrowness

| Device | BF16 | Narrow | Ratio |
|---|---|---|---|
| **TPU v5e** | 197 TFLOP/s | **393 INT8 TOPS** | ~2.0× |
| **Blackwell (consumer)** | 51 TFLOP/s | **202 TFLOP/s MXFP8 block-scaled** | **3.95×** |

⚠️ The Blackwell row carries a detail worth flagging for any future block-scaled-format work: on consumer Blackwell the
FP32-accumulate throughput of legacy warp-MMA is halved, while the **block-scaled** tensor
instruction is not throttled — making block-scaled MXFP8 MMA *the only way* to get full-rate tensor
throughput together with FP32 accumulation there. If gljax ever targets that tier, block-scaled
formats stop being an optimization and become the default path.

⚠️ But per §1.1, both columns are **compute** ratios. They price prefill. Decode's win is bandwidth
and does not appear in this table at all.

---

# 4. Wave A12.4 — Quantized KV Cache

## 4.1 This reopens an ARTX7 Non-Goal, deliberately

ARTX7 listed "KV compression" under Non-Goals. ARTX10 reopens it because the motivation is now
concrete: ARTX16 §9.4 flagged TPU v5e's 16 GB HBM per chip as the binding constraint, and ARTX11 §3.3
**doubled the KV budget** by adding a draft model's slab. Halving KV bytes is the most direct relief
available.

```text
ARTX7 slab:  [max_slots, n_kv_heads, max_seq_len, head_dim] × 2 (K,V) × n_layers

BF16 → FP8:  50% of the bytes
             → 2× max_slots, or 2× context, at the same HBM
```

## 4.2 ⛔ The long-context cliff

FP8 E4M3 KV quantization is usually described as "minimally degrading." That description is
**workload-dependent in a way that matters enormously**:

> On long-context needle-in-a-haystack tasks, FP8 KV accuracy has been measured regressing from
> **91% (BF16) to 13%** — attributed to hardware-level precision loss on large contraction
> dimensions. A **two-level accumulation** strategy recovers it to **89%**.

⚠️ A 78-point collapse on exactly the workload KV quantization is *for* (long context) is not a
rounding-error caveat. Two rules follow:

1. **Quantized KV must never be a silent default.** It is an explicit opt-in with a documented
   accuracy risk.
2. **The acceptance gate is a long-context retrieval task, not perplexity.** Perplexity on short
   sequences would show this as fine. ARTX12's harness must add a needle-in-a-haystack case before
   §4 ships.

## 4.3 ⚠️ Interaction with ARTX11 — the composition tax

Quantized KV degrades speculative decoding's acceptance rate:

> Mean accepted tokens per cycle drops **0.3–0.8** for FP8 E4M3 and **0.5–1.5** for INT8 per-token.

Run that through ARTX11 §1.3's model. At α=0.8, γ=6, c=0.1, ARTX11 predicts ~2.47×. Losing ~0.5
accepted tokens per cycle corresponds to roughly α≈0.72, which drops the speedup to ~2.1× — **a ~15%
loss of speculation's benefit, bought in exchange for 2× KV capacity.**

⚠️ **DESIGN DECISION — quantized KV and speculative decoding are jointly configured, not
independently.** Neither feature may be enabled without the policy layer knowing whether the other
is on. Enabling both blindly can leave a deployment slower *and* less accurate than enabling
neither.

## 4.4 Layout

⚠️ **DESIGN DECISION — per-tensor scales in v1; the slab shape is unchanged.**

```rust
pub struct QuantizedKvSlab {
    /// [max_slots, n_kv_heads_local, max_seq_len, head_dim] — SAME dims as ARTX7,
    /// narrower dtype. Every ARTX5 addressing rule carries over unmodified.
    kv_k: Vec<PjRtBuffer>,   // f8E4M3FN
    kv_v: Vec<PjRtBuffer>,
    /// Per-tensor scale, per layer, K and V separately. BF16.
    k_scale: Vec<f32>,
    v_scale: Vec<f32>,
    dtype: KvDtype,
}

pub enum KvDtype { Bf16, Fp8E4M3, Int8 }
```

Per-tensor rather than per-token because it keeps `dynamic_update_slice` at `[slot, :, pos, :]`
working **byte-identically** to ARTX5 — only the element width changes. Per-token or per-head scales
(a few KB for an 8k context) are more accurate and are what the accuracy-sensitive deployments use,
but they add a second, differently-shaped buffer that ARTX7's slab and ARTX11's rewind logic would
both have to learn about. ⚠️ Note that vLLM currently supports only per-tensor scalar scaling for
E4M3 — gljax is not behind the ecosystem here.

---

# 5. ⭐ Wave A12.5 — Kernel Dispatch: The Decision

## 5.1 The evidence

| Finding | Source |
|---|---|
| StableHLO has first-class quantized types, `uniform_quantize`/`uniform_dequantize`, per-tensor and per-axis, per-axis added to `dot_general` | StableHLO quantization spec |
| `stablehlo-legalize-quant-to-math` **decomposes** quantized ops into integer arithmetic plus CHLO broadcasts for scale multiply/divide and zero-point add | StableHLO passes |
| That pass's stated purpose: *"useful for systems that **do not support quantization natively**"* | StableHLO passes |
| PyTorch/XLA's quantized ops are labelled an **"Experimental feature"** | PyTorch/XLA docs |
| XLA's FP8 Dot casts up → scales → wide Dot → rescales, **fused** | XLA FP8 RFC |
| Every production quantized stack (llama.cpp, vLLM, TRT-LLM, DeepSpeed) uses custom kernels | §2.2 |

## 5.2 ⚠️ DESIGN DECISION — "dispatch" means **emission** dispatch, and ARTX8's rule stands

> gljax owns no kernels. `quant/dispatch.rs` chooses **which StableHLO expression to emit** for a
> given (weight format, activation dtype, backend) triple. It never chooses a kernel, because gljax
> has none to choose among.

Three candidate emissions per quantized matmul:

```text
E1  HOST DEQUANT      weights arrive BF16; emit an ordinary dot_general
                      → always correct, always available, zero HBM saving

E2  IN-GRAPH DEQUANT  emit uniform_dequantize → dot_general
                      → saves HBM **iff** XLA fuses the dequant into the dot prologue

E3  QUANTIZED DOT     emit dot_general on quantized operand types directly
                      → saves HBM **iff** the plugin lowers it natively rather than
                        falling back to legalize-quant-to-math emulation
```

## 5.3 The probe — because the answer is a plugin property, not a gljax property

⛔ **gljax cannot know statically whether E2 or E3 pays off.** The same emitted module may be fused
natively by one plugin and decomposed to emulation by another, and `legalize-quant-to-math`'s own
documentation says that decomposition exists precisely for backends without native support.

⚠️ **DESIGN DECISION — a startup capability probe, cached, with mandatory fallback to E1.**

```rust
// gljax/src/quant/probe.rs

pub struct QuantCapability {
    pub emission: Emission,        // E1 | E2 | E3 — the winner
    /// Measured bytes/s of the weight read. THE number that decides it:
    /// if E2/E3 did not reduce HBM traffic, they did not fuse.
    pub measured_weight_bw: f64,
    pub speedup_vs_e1: f64,
}

/// Run ONCE per (plugin, device, weight format) at startup. Cached on disk
/// beside the ARTX4 CompileCache — the answer cannot change without one of
/// those three changing.
pub fn probe(client: &PjRtClient, fmt: WeightFormat) -> QuantCapability {
    let baseline = bench_emission(client, fmt, Emission::HostDequant);
    let mut best = (Emission::HostDequant, baseline);

    for candidate in [Emission::InGraphDequant, Emission::QuantizedDot] {
        // compile() may legitimately fail — a plugin need not accept quantized types.
        let Ok(t) = try_bench_emission(client, fmt, candidate) else { continue };
        // ⚠️ Require a MARGIN, not merely "not worse". A tie means the dequant
        // did not fuse and we are paying for complexity we did not buy.
        if t.speedup_vs(&baseline) > PROBE_MARGIN {   // 1.15
            if t.better_than(&best.1) { best = (candidate, t); }
        }
    }
    QuantCapability::from(best)
}
```

⚠️ **`PROBE_MARGIN = 1.15` is deliberate, and it is this repo's culture encoded as a constant.**
GwenLand's CPU engine has three separate documented cases where an optimization measured strongly in
isolation and **neutral in production** — the standing lesson being that a flat result means the
technique is not the bottleneck. A quantized path that ties with BF16 has not fused; shipping it
would add a format, a probe, and a failure mode in exchange for nothing.

⚠️ **E1 must always remain reachable.** It is correct on every plugin, needs no capability, and is
never slower than BF16 serving because it *is* BF16 serving. Quantization is an accelerator, never a
dependency — the same rule ARTX11 §6.4 applied to speculation.

## 5.4 What this settles about Pridwen

ARTX16 §6.1 concluded Pridwen GQ4A/GQ2A is "a storage/distribution format, not a compute format."
ARTX10 refines rather than reverses that:

```text
ARTX16:   Pridwen → BF16 at load. Saves disk and network. Never saves HBM.
ARTX10:  Pridwen → E2/E3 IF the probe says the plugin fuses it.
         Otherwise → E1, which is exactly ARTX16's behaviour.
```

⚠️ Pridwen's block structure (§1.2) is expressible as StableHLO per-axis quantization when
`group_size` maps onto a `quantized_dimension` slice. GQ2A's `super_scale` hierarchy is **not**
directly expressible — StableHLO's uniform scheme is single-level. GQ2A therefore requires either an
extra in-graph multiply (an E2 variant) or host-side flattening of the two-level scales into
one level at load.

---

# 6. Runtime Architecture

```text
Checkpoint (safetensors / .gllm / GGUF)
      │
      ▼
quant/loader.rs      parse scheme + scales; ⚠️ detect SmoothQuant-folded norms (§1.3)
      │
      ▼
quant/format.rs      WeightFormat  { Bf16 | Fp8E4M3 | Int4Group{..} | Pridwen{..} }
      │
      ▼
quant/probe.rs       ONCE per (plugin, device, format) → QuantCapability   (§5.3)
      │
      ▼
quant/dispatch.rs    WeightFormat + QuantCapability → Emission
      │
      ▼
matrix/lower.rs      (ARTX8) the SOLE dot_general emission point — extended, not bypassed
      │
      ▼
XLA → PJRT
```

⚠️ **DESIGN DECISION — quantization enters through ARTX8's existing choke point, not around it.**
ARTX8 established `matrix/lower.rs` as the only place `dot_general` is emitted, and stated the
invariant that a grep for `dot_general` outside it should return nothing. ARTX10 extends
`MatmulOpts` with a weight format; it does not open a second emission path. Breaking that invariant
would undo ARTX8's most useful structural property.

```rust
// ARTX8's MatmulOpts, extended. Default is unchanged → BF16, byte-identical MLIR.
pub struct MatmulOpts {
    pub numerics: DotNumerics,          // ARTX8
    pub accumulate: Option<DType>,      // ARTX8
    pub weight_format: WeightFormat,    // ARTX10 — defaults to Bf16
}
```

---

# 7. Backend Abstraction

⚠️ **DESIGN DECISION — the abstraction is a capability *record*, not a trait with per-backend
implementations.**

ARTX8's rejected-alternative #7 declined a `Matmul` trait because it would have exactly one
implementor. The same reasoning applies here with more force: gljax has **one** backend interface
(PJRT) and backends differ only in *what they support*, never in *how gljax talks to them*.

```rust
// gljax/src/quant/backend.rs

/// What a specific plugin+device pair can actually do. Populated by probe(),
/// cached on disk. NOT a trait — there is nothing to dispatch dynamically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendQuantSupport {
    pub plugin: String,               // "tpu" | "cuda" | "cpu"
    pub device_kind: String,          // "TPU v5e" | "H100" | ...

    /// Compiles at all — says nothing about speed.
    pub accepts_quantized_types: bool,
    pub accepts_fp8: bool,

    /// ⚠️ THE field that matters. Compiling is not fusing.
    pub fuses_dequant_into_dot: bool,

    pub best_emission: Emission,
    pub measured_weight_bw_gbps: f64,
    pub probe_version: u32,           // invalidates the cache on plugin upgrade
}
```

⚠️ The `accepts_*` / `fuses_*` split is the whole point. A plugin that *accepts* quantized types but
decomposes them via `legalize-quant-to-math` will compile successfully and run **slower** than BF16.
A boolean "supports quantization" would report that as success.

---

# 8. Memory Layout

## 8.1 Weights

```text
BF16 (baseline)
  [K, N] × 2 bytes

INT4 group-quantized (GPTQ / AWQ / Pridwen GQ4A), group_size = G along K
  values : [K, N] × 4 bits          = K·N/2 bytes
  scales : [K/G, N] × 2 bytes
  zeros  : [K/G, N] × 4 bits        (asymmetric only)

  Example K=2048, N=6144, G=128:
    BF16     25.2 MB
    INT4     6.29 MB values + 0.20 MB scales   = 6.49 MB   → 3.88× smaller

FP8 E4M3
  [K, N] × 1 byte + per-tensor or per-axis scale   → 2.0× smaller
```

⚠️ **The scale tensor's layout decides whether E2/E3 is expressible at all.** StableHLO per-axis
quantization requires scales along one `quantized_dimension`. A checkpoint whose groups run along `K`
while gljax contracts along `K` maps cleanly; a checkpoint grouped along `N` needs a transpose —
which ARTX8 warns becomes a materialized copy under XLA layout assignment. **Check the group axis at
load, and reject rather than silently transpose.**

## 8.2 Total budget with everything on

```text
HBM = W_target(fmt) + W_draft(fmt)                    ARTX11 §3.3
    + kv_slab(target, dtype_kv)                       ARTX10 §4
    + kv_slab(draft,  dtype_kv)
    + activations(max_slots, bucket)                  ARTX7
    + compiled artifacts × |buckets| × |γ| × |arch|    ARTX5/7/11
```

⚠️ The artifact term is the one that surprises. ARTX5 has 5 sequence buckets; ARTX7 multiplies by
slot buckets; ARTX11 multiplies by the γ ladder; ARTX11 §4.3 adds `arch_hash`. Quantization does not
multiply it further — `WeightFormat` is fixed per model, not per request — but the compile *time*
already documented at 20–30 minutes cold (ARTX16 §4.2) is multiplied by everything before it.

---

# 9. Pseudocode

## 9.1 Emission dispatch

```rust
// gljax/src/quant/dispatch.rs

pub fn emit_quantized_matmul(
    x: &Tensor, w: &QuantizedWeight, cap: &QuantCapability, opts: &MatmulOpts,
) -> Tensor {
    match cap.emission {
        // E1 — weights already dequantized on the host at load time.
        // Always available. This is ARTX16 §6.1's behaviour, preserved as the floor.
        Emission::HostDequant => matrix::lower::emit(x, &w.as_bf16(), opts),

        // E2 — dequantize in-graph, then an ordinary dot.
        // Pays off ONLY if XLA fuses the dequant into the dot's prologue.
        Emission::InGraphDequant => {
            let w_bf16 = ops::uniform_dequantize(&w.packed, &w.scales, w.zero_points.as_ref());
            matrix::lower::emit(x, &w_bf16, opts)
        }

        // E3 — dot_general directly on quantized operand types.
        // Pays off ONLY if the plugin lowers it natively rather than via
        // legalize-quant-to-math emulation.
        Emission::QuantizedDot => {
            matrix::lower::emit_quantized(
                x, &w.packed,
                QuantParams { scales: &w.scales, zero_points: w.zero_points.as_deref(),
                              quantized_dimension: w.group_axis },
                opts,
            )
        }
    }
}
```

## 9.2 The probe

```rust
// Weight-read bandwidth is the discriminator — not wall time.
// A fused dequant reads Q4 bytes; an unfused one reads BF16 bytes.
// The ratio is directly observable and is not confounded by compute.
fn bench_emission(client: &PjRtClient, fmt: WeightFormat, e: Emission) -> ProbeResult {
    let m = build_probe_module(fmt, e);          // one representative decode matmul
    let exe = client.compile(&m)?;
    warmup(&exe, PROBE_WARMUP);                  // exclude first-call compile/alloc
    let t = median_of(PROBE_REPEATS, || exe.execute(&inputs));

    ProbeResult {
        emission: e,
        seconds: t,
        // If this lands near the BF16 figure, the dequant did not fuse.
        weight_bw_gbps: (fmt.packed_bytes() as f64 / t) / 1e9,
    }
}
```

## 9.3 Quantized KV write

```rust
// The slab's SHAPE is unchanged from ARTX5/ARTX7 — only element width differs,
// so every addressing rule carries over untouched.
pub fn write_kv_quantized(
    slab: &mut QuantizedKvSlab, slot: SlotId, layer: usize, pos: usize,
    k: &Tensor, v: &Tensor,
) {
    let kq = ops::uniform_quantize(k, slab.k_scale[layer], slab.dtype);
    let vq = ops::uniform_quantize(v, slab.v_scale[layer], slab.dtype);
    // Identical to ARTX5 §2 — dynamic_update_slice at a runtime scalar index.
    dynamic_update_slice(&mut slab.kv_k[layer], &kq, &[slot.0, 0, pos, 0]);
    dynamic_update_slice(&mut slab.kv_v[layer], &vq, &[slot.0, 0, pos, 0]);
}
```

---

# 10. Tradeoffs

| Decision | Gain | Cost | Verdict |
|---|---|---|---|
| No custom kernels (ARTX8 upheld) | Portability across every PJRT plugin; no toolchain matrix | **Will not match vLLM/TRT-LLM quantized throughput** | ✅ Bounded expectation, stated honestly |
| Capability probe over static assumption | The answer is a plugin property; measurement is the only way to get it | Startup cost; a cache to invalidate | ✅ Only correct option |
| `PROBE_MARGIN = 1.15` | Rejects paths that compile but do not fuse | May reject a genuine small win | ✅ A tie means it did not fuse |
| E1 always reachable | Correct everywhere; never slower than BF16 | Three code paths instead of one | ✅ Quantization is an accelerator, not a dependency |
| Weight-only prioritized over W8A8 | Decode is bandwidth-bound (ARTX8) | Leaves the ~2×/3.95× compute uplift on the table for prefill | ✅ Matches the measured bottleneck |
| FP32 accumulation always | Preserves ARTX12's error model and tolerance table | Forgoes narrow-accumulator speed | ✅ Not negotiable |
| Per-tensor KV scales in v1 | ARTX5 addressing byte-identical | Less accurate than per-token | ✅ Matches vLLM's current E4M3 support |
| Quantized KV opt-in, never default | 2× KV capacity when chosen | 91%→13% long-context cliff if chosen blindly | ✅ Gated on a retrieval test, not perplexity |
| KV quant + spec decode jointly configured | Prevents a slower-and-less-accurate combination | Policy coupling between two features | ✅ ~15% of speculation's benefit is at stake |
| Capability record, not a trait | One PJRT interface; backends differ in support, not in protocol | Adding a backend means editing a struct | ✅ ARTX8 rejected-alternative #7 |
| Quantization enters via `matrix/lower.rs` | Preserves ARTX8's single-choke-point invariant | `MatmulOpts` grows a field | ✅ |

## 10.1 When quantization is a net loss

1. **The probe returns E1** — no HBM saving; you gained disk compression and nothing else.
2. **W8A8 on a decode-dominated workload** — pays compute you are not spending (§1.1).
3. **Quantized KV at long context without a retrieval gate** — §4.2's cliff.
4. **Quantized KV plus speculation, unconsidered** — §4.3's composition tax.
5. **Group axis mismatched to the contraction axis** — a materialized transpose per matmul (§8.1).

---

# 11. Wave Plan + What Comes Next

| Wave | Scope | Gate |
|---|---|---|
| **A12.0** | `quant/format.rs` + `loader.rs`: parse GPTQ/AWQ/Pridwen/FP8 metadata; ⚠️ detect SmoothQuant-folded norms | ARTX12 T0: dequantized weights bit-match a reference dequant |
| **A12.1** | `probe.rs` + `backend.rs`: capability probe, disk cache, E1 fallback | Probe returns a *stable* verdict across 3 runs on one device |
| **A12.2** | `dispatch.rs` + `matrix/lower.rs` extension: E2 and E3 emission | ⭐ Measured weight-read bandwidth confirms fusion, or the path is rejected |
| **A12.3** | FP8 activation path (W8A8) | Prefill throughput gain measured; decode confirmed unchanged |
| **A12.4** | `QuantizedKvSlab`, FP8-E4M3 KV | ⭐ **Needle-in-a-haystack retrieval**, not perplexity (§4.2) |
| **A12.5** | Joint policy: KV quant × speculation (§4.3) | Combined config never slower than either alone |

⚠️ **A12.2's gate is the document's decisive experiment.** If measured weight-read bandwidth on E2/E3
does not drop toward the packed size, then no plugin gljax targets fuses the dequant, and the honest
conclusion is that **quantization for gljax remains a storage format** — ARTX16 §6.1's position,
confirmed by measurement rather than assumed. That is a legitimate outcome and should be recorded as
such, not worked around.

## 11.1 Where this leads

§1.3 surfaced a bug class this document cannot close on its own: **checkpoints carry
algorithm-specific transformations that are invisible from tensor shapes.** SmoothQuant folds a
smoothing vector into the preceding norm, AWQ bakes per-channel scales into the weights, GPTQ may
store weights in a permuted order. Each is silent, and each produces fluent wrong output rather than
an error.

That is **ARTX12 — Model Compatibility & Runtime Conformance**, which pairs the checkpoint-variant
matrix with the correctness harness that verifies it. ARTX11 §4's `Architecture` descriptor (seven
Qwen2-specific assumptions found baked into ARTX3) belongs to the same problem and is settled there.

**Deferred beyond the ARTX08–ARTX16 arc:** MXFP8 / block-scaled formats (§3.3 — becomes mandatory if
gljax targets consumer Blackwell); prefill/decode disaggregation (ARTX16 §3.3, still blocked on
multi-host); dynamic draft trees (ARTX11 §0.2, blocked on relaxing static shapes).

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | **ARTX8's no-kernel rule stands; "dispatch" = emission dispatch** | gljax has no kernels to dispatch among; it chooses which StableHLO to emit | Hard |
| D2 | Capability **probe** at startup, cached, versioned | Whether dequant fuses is a plugin property; `legalize-quant-to-math` exists precisely for backends without native support | Hard |
| D3 | `PROBE_MARGIN = 1.15`, not "not worse" | A tie means the dequant did not fuse; this repo has three documented isolated-win/production-neutral cases | Trivial |
| D4 | **E1 (host dequant) always reachable** | Correct on every plugin; never slower than BF16 because it *is* BF16 | N/A |
| D5 | Weight-only prioritized over W8A8 | ARTX8: decode is bandwidth-bound at 200–600× below ridge; W8A8 prices compute | Medium |
| D6 | **FP32 accumulation at every quantization level** | ARTX8's hardware finding; ARTX12's tolerance table derives from it | Hard |
| D7 | Quantization enters through `matrix/lower.rs` | Preserves ARTX8's single-emission-point invariant | Hard |
| D8 | Backend support is a **record**, not a trait | One PJRT protocol; backends differ in capability only (ARTX8 rej-alt #7) | Medium |
| D9 | `accepts_*` and `fuses_*` are separate fields | Compiling is not fusing; a single boolean would report emulation as success | Trivial |
| D10 | GPTQ and AWQ share one runtime path | From the runtime's view both are "INT4 + scales at granularity G" | Trivial |
| D11 | **SmoothQuant detection is a loader concern** | Its smoothing vector is folded into the preceding norm; a naive loader yields fluent garbage | Trivial |
| D12 | Reject mismatched group axis; never silently transpose | ARTX8: a transpose becomes a materialized copy under layout assignment | Trivial |
| D13 | Quantized KV is **opt-in, never default** | 91% → 13% long-context retrieval collapse | Trivial |
| D14 | KV acceptance gate is **retrieval**, not perplexity | Short-sequence perplexity does not surface the cliff | Trivial |
| D15 | Per-tensor KV scales in v1 | Keeps ARTX5 addressing byte-identical; matches vLLM's current E4M3 support | Medium |
| D16 | **KV quant and speculation jointly configured** | Quantized KV costs 0.3–0.8 accepted tokens/cycle ≈ 15% of speculation's benefit | Medium |
| D17 | Pridwen is E1 by default, E2/E3 only if probed | Refines ARTX16 §6.1 rather than reversing it | Trivial |
| D18 | GQ2A's two-level scales flattened at load | StableHLO's uniform scheme is single-level | Medium |
| D19 | FP8 portability not assumed | TPU v5e publishes BF16 and INT8, not FP8 | N/A |
| D20 | A12.2 may legitimately conclude "storage format only" | If no target plugin fuses, that is a measurement, not a failure | N/A |

---

# Appendix B — Format Quick Reference

```text
Emission paths      E1 host dequant · E2 in-graph dequant · E3 quantized dot
StableHLO quant     real_value = scale × (quantized_value − zero_point)
                    per-tensor | per-axis(quantized_dimension); uniform_{quantize,dequantize}
FP8                 f8E4M3FN (inference) · f8E5M2 (gradients) · FNUZ variants
MXFP8               E4M3 values · 32-element blocks · E8M0 scale (2⁻¹²⁷…2¹²⁷) · FP32 accum
Q4_K                256 superblock ÷ 8×32 blocks · 4-bit · 6-bit block scale · 16-bit super scale+min
GQ4A                4-bit · dequant(c) = c − 8 · per-block scale
GQ2A                2-bit asymmetric · per-block min+scale · super_scale
SmoothQuant         s_j = max(|X_j|)^α / max(|W_j|)^(1−α)   ⚠️ folded into the preceding norm
Narrow uplift       TPU v5e 393 INT8 TOPS vs 197 BF16 TFLOP/s (~2×)
                    Blackwell consumer 202 vs 51 TFLOP/s MXFP8 block-scaled (3.95×)
```

---

# Sources

- [StableHLO Quantization | OpenXLA](https://openxla.org/stablehlo/quantization) — LiteRT spec, MLIR Quant dialect, per-tensor/per-axis, `uniform_quantize`/`uniform_dequantize`, per-axis added to `dot_general`.
- [StableHLO Passes | OpenXLA](https://openxla.org/stablehlo/generated/stablehlo_passes) — `stablehlo-legalize-quant-to-math`: decomposes quantized ops to integer arithmetic + CHLO broadcasts; *"useful for systems that do not support quantization natively."*
- [Quantized Operations for XLA (Experimental) — PyTorch/XLA](https://docs.pytorch.org/xla/release/r2.7/perf/quantized_ops.html) — maturity level of the XLA quantized path.
- [RFC: FP8 in XLA](https://github.com/openxla/xla/discussions/22) — FP8 Dot lowering: cast up → scale → wide Dot → rescale, fused.
- [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/pdf/2210.17323) — OBQ-derived, inverse-Hessian Cholesky, arbitrary fixed order, 3–4 bit, OPT-175B at 3-bit on one 80 GB A100.
- [AWQ: Activation-aware Weight Quantization](https://arxiv.org/pdf/2306.00978) and [AWQ concepts](https://leimao.github.io/blog/AWQ-Activation-Aware-Weight-Quantization/) — salient channels from activation stats, per-channel scaling then naive INT4 rounding.
- [SmoothQuant](https://arxiv.org/abs/2211.10438) and [Intel Neural Compressor: SmoothQuant](https://intel.github.io/neural-compressor/latest/docs/source/smooth_quant.html) — `s_j = max(|X_j|)^α / max(|W_j|)^(1−α)`, W8A8, training-free, difficulty migrated activations→weights.
- [MXFP8 — NVIDIA Transformer Engine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/mxfp8/mxfp8.html) and [Chasing 6+ TB/s: an MXFP8 quantizer on Blackwell](https://blog.fal.ai/chasing-6-tb-s-an-mxfp8-quantizer-on-blackwell/) — OCP MX, 32-element blocks, E8M0 scales, 202 vs 51 TFLOP/s on consumer Blackwell.
- [Quantized KV Cache — vLLM](https://docs.vllm.ai/en/v0.9.2/features/quantization/quantized_kvcache.html) and [The State of FP8 KV-Cache and Attention Quantization in vLLM](https://vllm.ai/blog/2026-04-22-fp8-kvcache) — E4M3 for KV, per-tensor scaling support.
- [KV cache quantization: what FP8/INT8 K and V actually buy you, and where they break](https://dev.to/tech_nuggets/kv-cache-quantization-what-fp8int8-k-and-v-actually-buy-you-and-where-they-break-4fnl) — **91% → 13%** long-context needle-in-a-haystack regression, two-level accumulation recovering to 89%; speculative-decoding acceptance drop of 0.3–0.8 (FP8 E4M3) and 0.5–1.5 (INT8 per-token).
- [Quantization Techniques | llama.cpp](https://deepwiki.com/ggml-org/llama.cpp/7.3-quantization-techniques) and [Which Quantization Should I Use?](https://arxiv.org/html/2601.14277v1) — Q4_K: 256-element superblocks ÷ eight 32-element blocks, 6-bit block scales, 16-bit superblock scale and min.
- [How to Think About TPUs](https://jax-ml.github.io/scaling-book/tpus/) and [TPUv5e](https://newsletter.semianalysis.com/p/tpuv5e-the-new-benchmark-in-cost) — 197 BF16 TFLOP/s, 393 INT8 TOPS.

**Repo-internal:** `architecture/Pridwen-proposal-v5.md` (GQ4A zero-centered `c − 8` + per-block
scale; GQ2A asymmetric `min_i`/`scale_i` + `super_scale`); `ARTX8` (no-kernel rule, decode
bandwidth-bound, single emission point, rejected-alternative #7); `ARTX16 §6` (Pridwen as storage
format, FP8 lowering, TPU v5e FP8 unconfirmed); `ARTX12 §3` (FP32-accumulate error model);
`ARTX11 §3.3, §4.3` (draft model doubles the KV budget; `Architecture` descriptor).
