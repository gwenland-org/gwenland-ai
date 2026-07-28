# ARTX10 v2 — Quantized Runtime

**Status:** ⛔ **REDESIGNED.** v1 was built on
`stablehlo.uniform_dequantize → dot_general`. That path does not exist — not
"is slow", not "is unfused": *does not parse*. v1 is preserved at
[`ARTX10-quantized-runtime-architecture-v1-superseded.md`](ARTX10-quantized-runtime-architecture-v1-superseded.md).

**Evidence:** [`gljax/probes/pjrt_cpu_quant_probe.py`](../probes/pjrt_cpu_quant_probe.py)
— reproducible, records its own results, re-runnable against any PJRT plugin.
Every claim below marked ⭐ was measured on this machine, not read.

---

# 0. The one-paragraph version

On the PJRT CPU plugin there is **no configuration in which quantized weights
stay quantized through a `dot_general`**. Either they are function arguments,
in which case XLA materialises the full f32 weight into scratch on every call
(measured: 4.91 MB of arguments, **17.45 MB of temp**, for a `[896, 4864]`
weight), or they are constants, in which case XLA constant-folds the
dequantisation at compile time and the executable carries dense f32 weights
(measured: `%constant.6 = f32[896,4864] constant({…})`). The first is *worse*
than an f32 model for a bandwidth-bound decode — 4.64x slower, measured. The
second saves nothing.

⭐⭐ **But there is a third way, and it works: never dequantise more than a
tile.** Emitting a `stablehlo.while` over the contracting dimension — slice,
dequantise, dot, accumulate — is **3.58x faster than the naive quantized path**
and costs **1.30x** an f32 baseline while using **2.86x less memory** (§3.1).
Quantization stops being catastrophic and becomes a memory-for-time trade whose
sign depends on whether the model fits.

⭐ **ARTX10 v2 therefore defaults to host-side dequant, offers tile streaming as
the quantized path, and picks between them with a cost model rather than a
capability flag** — because the right answer genuinely differs per model.

---

# 1. Research findings

## Q1 — `uniform_dequantize`: ⛔ dead, and worse than the issue reports

[openxla/xla#9291](https://github.com/openxla/xla/issues/9291) is **still
open**, filed 2024-02-07, with **no subsequent comments** — two and a half
years of silence. Its reported failure is a *translation* one:
`'mhlo.uniform_quantize' op can't be translated to XLA HLO`. The sibling
[pytorch/xla#6567](https://github.com/pytorch/xla/issues/6567) shows the same
class from a different frontend.

⭐ **The probe found something stronger.** Fed to jaxlib 0.10.2's MLIR context,
the module does not even parse:

```
`!"quant"<"uniform<i8:f32, 3.900000e-03>">` type created with unregistered dialect
```

The `quant` dialect is **not registered on this path at all**, while an
otherwise identical f32 module parses cleanly (probe case `(k)`, the parser
control). This matters for planning: it is not a lowering gap that a future XLA
release closes from the compiler side — the type cannot currently be *spelled*
in the context the plugin parses with.

## Q2 — StableHLO Quantizer output: ⚠️ **unconfirmed, and not depended on**

The Quantizer exists and was presented at
[OpenXLA Dev Lab 2024](https://opensource.googleblog.com/2024/05/openxla-dev-lab-2024-building-grouundbreaking-systems-together.html)
as "framework and device-agnostic". StableHLO quantization follows the LiteRT
specification and **inherits its type expression from MLIR's Quant dialect**
(per-tensor and per-axis).

⛔ **No published example of its emitted IR was found.** The official
[quantization page](https://openxla.org/stablehlo/quantization) does not mention
the Quantizer at all. If it emits Quant-dialect types then §Q1 applies to it
verbatim — but that is an inference, not a source, and nothing in this design
rests on it.

⭐ The same page documents the escape hatch, and it is load-bearing here: the
`stablehlo-legalize-quant-to-math` pass *"converts StableHLO operations on
quantized types into equivalent operations on integer types… useful for systems
that do not support quantization natively."* ARTX10 v2 emits that arithmetic
**directly**, never constructing the quantized type it would have to be
legalised away from.

## Q3 — What actually runs: ✅ answered by probe, not by literature

The literature search was inconclusive. [PyTorch/XLA's quantized
ops](https://docs.pytorch.org/xla/master/quantized_ops.html) advertise blockwise
W4A16/W8A16 but **do not document the IR they lower to**; activation
quantization (W8A8, W4A8) is marked *not supported*.
[zml](https://github.com/zml/zml)'s README does not mention quantization at all,
so it is not evidence of anything.

⭐ So the question was settled empirically. Every non-quant-dialect pattern
compiles **and produces numerically correct output**:

| pattern | verdict | max abs err |
|---|---|---|
| f32 `dot_general` (control) | RUNS-OK | 3.8e-06 |
| `convert(i8→f32) → dot` | RUNS-OK | 1.5e-05 |
| per-axis scale (`convert·broadcast·multiply → dot`) | RUNS-OK | 2.3e-05 |
| ⭐ **blockwise, 32-element blocks** | RUNS-OK | 1.5e-05 |
| ⭐ **two-level block × superblock scales** | RUNS-OK | 2.3e-05 |
| f16 block scale (GQ4A's own dtype) | RUNS-OK | 1.9e-05 |
| `int4` as a real PJRT buffer | RUNS-OK | 2.3e-05 |

⭐ Independently: asked to scale an `i8` weight and matmul it, **JAX itself
emits exactly this pattern** — `stablehlo.convert → broadcast_in_dim →
multiply → dot_general` — and never `uniform_dequantize`. The pattern is not a
workaround; it is what the ecosystem already produces.

**Verdict on the candidate list:** (a) ⛔ fails to parse · (b) ✅ works · (c) ✅
works · (d) GPU-only, unprobed · (e) not PJRT-native.

## Q4 — PJRT CPU plugin: ⛔ **compiles it, never keeps it quantized**

⭐ This is the finding that decides the architecture. A `[896, 4864]` weight —
a Qwen2.5-0.5B `ffn_up` shape — three deliveries, figures from the compiled
executable's own `memory_analysis()`:

| delivery | argument | temp | what the optimised HLO shows |
|---|---:|---:|---|
| weights as **arguments** | 4.91 MB | ⛔ **17.45 MB** | `%fused_computation.1 (f32[28,4864], s8[896,4864]) -> f32[896,4864]` — the full f32 weight is built into scratch, then the dot reads it |
| weights as **constants** | 0.00 MB | 0.02 MB | `%constant.6 = f32[896,4864] constant({…})` — the dequantisation was constant-folded at compile time |
| f32 weights (reference) | 17.44 MB | 0.02 MB | — |

17.45 MB is exactly `896 × 4864 × 4`. Nothing is being approximated here: the
dequantised matrix is fully materialised, per call.

⭐ **Consequences, stated plainly:**

* **Arguments** — host memory drops 17.4 → 4.4 MB, but each forward pass now
  *writes* 17.4 MB and *reads* 17.4 MB where an f32 model only reads it. For
  decode, which is bandwidth-bound by construction, this is a **regression**.
* **Constants** — no runtime cost, and **no memory saving**: the quantization
  has been compiled away.
* `int4` is **not bit-packed**: `int4` and `int8` both measure **1.00 byte per
  element**. The narrow dtype buys nothing on its own.

⚠️ **Scope of the claim:** jaxlib 0.10.2 CPU plugin, one shape, batch 1, no
donated buffers, no XLA flag overrides. §9 lists what would change it.

## Q5 — GPU: ⚠️ pattern confirmed, ⛔ **no numbers**

XLA has `xla_gpu_experimental_enable_subchannel_dequantisation_fusion`, fusing
`[x,z]param → [x,y,z]broadcast → [x*y,z]bitcast → multiply → dot`
([xla.proto](https://github.com/openxla/xla/blob/main/xla/xla.proto)) — the same
shape §Q3 measured, and **not** `uniform_dequantize`. Its own documentation
warns: *"performance can be worse, because some block sizes / split-k > 1 is not
considered for subchannel dequant fusions."*

⛔ **No FLOP or bandwidth measurement against an FP16 baseline was found.** The
flag is named `xla_gpu_*`, implying no CPU counterpart — an inference from
naming, not a citation. **GPU remains unprobed**; the bring-up machine has none.

## Q6 — GQ4A: ⛔ not a quantized *type*, ✅ exactly expressible as *math*

The [StableHLO spec](https://raw.githubusercontent.com/openxla/stablehlo/main/docs/spec.md)
permits quantized element types `si2…si64` / `ui2…ui64` with **per-tensor** (one
scale) or **per-axis** (one scale per slice along **one** `quantization_dimension`).

GQ4A is 4.3125 bpw: an f16 scale per **32-element block** plus an f32 superblock
scale per **256**. For a `[in, out]` weight that needs `(in/32) × out` scales.
Per-axis yields `in` scales *or* `out` scales — **never `(in/32) × out`**.

⛔ **GQ4A cannot be represented as a StableHLO quantized type, and no future
per-axis extension fixes it** — the requirement is two-dimensional, and per-axis
is one-dimensional by definition. GQ2A (three levels) is further away still.

⭐ But it maps exactly onto explicit math, and the probe ran it end-to-end:
`convert → reshape → broadcast_in_dim → multiply → reshape → dot_general`, with
a second `multiply` for the superblock scale. Two-level scales, f16 block
scales, 32-element blocks — all correct to ~2e-05.

---

# 2. Viable paths, ranked by confidence

| # | Path | Measured | Verdict |
|---|---|---|---|
| **1** | **Host-side dequant** → dense f32 → plain `dot_general` | 848 µs · 17.46 MB | ✅ **Default.** Fastest, works on every backend including unprobed ones, saves no memory |
| **2** | ⭐ **Tile streaming** — `while` over the contracting dim | **1100 µs · 6.10 MB** | ✅ **The quantized path.** 1.30x the time for 2.86x less memory. Chosen by cost model (§5) |
| **3** | Blockwise IR over the whole weight, as **arguments** | 3932 µs · 22.36 MB | ⛔ **Never by default.** 4.64x slower than f32 *and* more total memory than path 2. Kept only as the comparison baseline, and as the shape a genuinely-fusing backend would want |
| **4** | Blockwise IR, weights as **constants** | folded to f32 | ⛔ **Never.** Same memory as path 1, longer compile, no benefit |
| **5** | `!quant.uniform` / `uniform_quantize` | PARSE-FAIL | ⛔ **Unrunnable.** Do not emit |
| **6** | Custom call to a vendor INT8 GEMM | — | ⛔ **Violates P1**; gljax owns no kernels |

⭐ Paths 1 and 2 are both correct and both defensible; **which one wins is a
property of the deployment, not of the backend.** That is why §5 replaces v1's
capability enum with a cost model.

---

# 3. Chosen architecture

```
                    ┌──────────────────────────────────┐
   .gllm / GGUF ───►│  WeightSource  (host, ARTX04)    │
                    │  GQ4A · GQ2A · Q4_K · INT8 · FP8 │
                    └───────────────┬──────────────────┘
                                    │
                     ┌──────────────▼───────────────┐
                     │  QuantPlan::decide()          │
                     │  consults CapabilityProbe once│
                     └───┬───────────────────────┬───┘
             MATERIALISE │                       │ EMIT_BLOCKWISE
                (default)▼                       ▼ (probe said Fused)
        ┌────────────────────────┐   ┌─────────────────────────────┐
        │ host dequant → f32/bf16│   │ upload int quants + scales  │
        │ upload one dense buffer│   │ as separate arguments; emit │
        │ emit plain dot_general │   │ convert·reshape·bcast·mul·dot│
        └────────────┬───────────┘   └──────────────┬──────────────┘
                     └───────────┬──────────────────┘
                                 ▼
                   ┌──────────────────────────────┐
                   │ NumericGate (P4)             │
                   │ blockwise vs f32 oracle;     │
                   │ diverge ⇒ fall back, loudly  │
                   └──────────────────────────────┘
```

**The default is MATERIALISE**, and the quantized paths are reached only from a
measurement.

## 3.1 ⭐⭐ Tile streaming — the reason a quantized path exists at all

§Q4 measured that dequantising a *whole* weight is catastrophic. The fix is not
to give up on quantized IR but to **never dequantise more than a tile**:
slice the reduction dimension, dequantise one slice, dot it, accumulate.
Measured, decode-shaped matvec `[1,896] × [896,4864]`, block 32, best-of-30
([`tile_streaming_probe.py`](../probes/tile_streaming_probe.py)):

| variant | time | vs f32 | temp | argument |
|---|---:|---:|---:|---:|
| f32 weights (reference) | **848 µs** | 1.00× | 0.02 MB | 17.44 MB |
| quant, whole-weight dequant | 3932 µs | ⛔ **4.64×** | 17.45 MB | 4.91 MB |
| quant, reduction tiles of 128 | 1108 µs | 1.31× | 2.51 MB | 4.91 MB |
| ⭐ **quant, reduction tiles of 32** | **1100 µs** | **1.30×** | 1.19 MB | 4.91 MB |

⭐ **3.58× faster than the naive quantized path**, and scratch falls
17.45 → 1.19 MB. Quantization goes from catastrophic to a ~30 % tax.

⭐⭐ **The trade is now real and its sign depends on the model:**

```
   f32          17.46 MB working set,   848 us
   tiled quant   6.10 MB working set,  1100 us
                 2.86x less memory  for  1.30x the time
```

A win when the model does not fit in RAM; a loss when it does. **That is a cost
model's decision, not a capability flag's** — see §5.

### ⛔ Two implementation constraints the measurement forced

1. **The loop must survive into the compiled program.** An *unrolled* tile loop
   (a Rust/Python `for` emitting one dot per tile) measured **17.43 MB — no
   improvement at all.** XLA fuses it straight back into a whole-weight
   dequant. Only a real `stablehlo.while` (what `lax.scan` lowers to) holds.
   ⚠️ Any emitter that unrolls has silently reverted to the broken path.
2. **Tile the reduction dimension, not the output dimension.** Output tiling
   measured 9.26 MB against reduction tiling's 1.19 MB, because under reduction
   tiling the accumulator stays `[m, n]` and only `[tile_k, n]` is ever live.

⚠️ Measured at **batch 1** only. Prefill amortises a materialised weight across
many tokens and may invert the ranking — §9 question 3, now the most valuable
open item rather than a speculative one.

This inverts v1, which emitted quantized IR by default and treated fallback as
the exception.

---

# 4. `EmissionDispatch` changes

v1 dispatched on **weight dtype**. v2 dispatches on a **measured plugin
behaviour**, because §Q4 showed dtype does not predict what the backend does
with it.

```rust
/// How a quantized weight reaches the compiled program.
///
/// ⛔ Not a user preference. Chosen by [`CapabilityProbe`] from measurement,
/// and `Materialise` is the answer whenever there is no evidence — P5.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightDelivery {
    /// Dequantise on the host at load; upload one dense buffer.
    ///
    /// Costs full weight memory. Costs nothing at runtime. Works on every
    /// backend, including ones nobody has probed.
    Materialise { as_dtype: DType },

    /// Upload integer quants and scales as separate arguments; emit
    /// `convert · reshape · broadcast_in_dim · multiply · dot_general` over
    /// the **whole** weight.
    ///
    /// ⛔ Measured 4.64x slower than f32 on the CPU plugin (§Q4). Retained
    /// only as the thing `EmitTileStream` is compared against, and as the
    /// shape a backend that genuinely fuses would want. Never a default.
    EmitBlockwise { block: u32, levels: ScaleLevels },

    /// ⭐ Emit a `stablehlo.while` over the contracting dimension: slice,
    /// dequantise one tile, dot, accumulate. The dequantised weight never
    /// exists in full.
    ///
    /// Measured 3.58x faster than `EmitBlockwise` and 1.30x slower than f32,
    /// at 2.86x less memory (§3.1). This is the only quantized delivery worth
    /// selecting on a backend that does not fuse.
    ///
    /// ⛔ `tile_k` must divide the contracting dim AND be a multiple of
    /// `block`. ⛔ The emitter must produce a real while-loop — unrolling it
    /// measured *zero* improvement, because XLA folds it back.
    EmitTileStream { block: u32, tile_k: u32, levels: ScaleLevels },
}

/// GQ4A has two levels: f16 per 32 elements, f32 per 256. GQ2A adds a third.
/// Modelled explicitly because each level is one more `broadcast_in_dim` +
/// `multiply` in the graph, and therefore one more place a backend can break
/// the fusion chain this whole path depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleLevels {
    /// One scale per block. Q8_0, Q4_0.
    Block { dtype: DType },
    /// Block scale x superblock scale. GQ4A, Q4_K.
    BlockSuper { block_dtype: DType, super_dtype: DType, super_span: u32 },
}

pub struct EmissionDispatch {
    delivery: WeightDelivery,
    /// ⛔ Retained so a bug report can state which probe result produced this
    /// choice. A dispatch decision without provenance is unauditable, and
    /// v1's central assumption survived two years partly because nothing
    /// recorded where it came from.
    provenance: ProbeVerdict,
}

impl EmissionDispatch {
    /// Emit the weight operand of one `dot_general`; returns the SSA value the
    /// dot contracts against.
    pub fn weight_operand(
        &self,
        f: &mut FuncBuilder,      // ARTX02
        w: WeightHandle,
        shape: [usize; 2],        // [in, out]
    ) -> Value {
        match self.delivery {
            WeightDelivery::Materialise { .. } => f.param(w.dense),

            WeightDelivery::EmitBlockwise { block, levels } => {
                let [k, n] = shape;
                let nb = k / block as usize;
                debug_assert_eq!(nb * block as usize, k, "block must divide the contracting dim");

                let q = f.convert(f.param(w.quants), DType::F32);
                // [k, n] -> [nb, block, n], so a per-block scale broadcasts
                // along the block axis only. This reshape is the whole reason
                // GQ4A is expressible at all (§Q6).
                let mut acc = f.reshape(q, &[nb, block as usize, n]);

                for scale in levels.operands(&w) {
                    let s = f.convert(f.param(scale.handle), DType::F32);
                    // dims [0, 2] = (block-index, output) — the block axis is
                    // the broadcast one.
                    let s = f.broadcast_in_dim(s, &[nb, block as usize, n], &[0, 2]);
                    acc = f.multiply(acc, s);
                }
                f.reshape(acc, &[k, n])
            }
        }
    }
}
```

⚠️ **`EmitBlockwise` adds compile-cache key dimensions (P3).** `block`,
`levels`, and the derived `nb` all change the graph, and they multiply against
`seq_bucket` and `batch_size`. Warmup is already 20–30 minutes cold; a model
with per-layer block sizes would be untenable. `QuantPlan` must canonicalise to
the fewest distinct `(block, levels)` pairs the format permits, and refuse a
model that needs more than a configured ceiling.

---

# 5. `CapabilityProbe` spec

v1 asked *"does this plugin support quantized ops?"* — a yes/no feature query.
⭐ §Q4 showed that question is useless: the CPU plugin **compiles the blockwise
pattern perfectly, computes the right answer, and still produces a worse
result**. v2 asks *"what does this plugin do with it"*, and answers by compiling
and reading the compiler's own memory analysis.

```rust
/// What a plugin actually does with a blockwise dequantisation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProbeVerdict {
    /// The dequantised weight never appears as a buffer — the multiply fused
    /// into the dot. The only verdict that justifies `EmitBlockwise`.
    Fused,

    /// The full dequantised weight is built into scratch on every call.
    /// ⭐ Measured on the PJRT CPU plugin, jaxlib 0.10.2: temp = 17.45 MB for
    /// a [896, 4864] weight, i.e. exactly k*n*4.
    MaterialisesPerCall { temp_bytes: u64 },

    /// The compiler constant-folded the dequantisation away. Numerically fine;
    /// the executable now carries dense f32 weights, so nothing was saved.
    ConstantFolded,

    /// The pattern did not compile.
    Unsupported { detail: String },

    /// Never probed. ⛔ `QuantPlan` treats this exactly as
    /// `MaterialisesPerCall` — P5, refuse rather than approximate.
    Unknown,
}

pub struct CapabilityProbe<'c> {
    client: &'c PjrtClient,   // ARTX01
}

impl CapabilityProbe<'_> {
    /// Compile a blockwise module and classify the result.
    ///
    /// ⚠️ Uses a **representative** shape, never a toy one. A small matrix can
    /// fuse where a real layer materialises, so a 4x8 probe would report
    /// `Fused` and be wrong about every weight in the model. Default: the
    /// largest FFN weight present.
    pub fn probe_blockwise(&self, shape: [usize; 2], block: u32) -> ProbeVerdict {
        let module = build_blockwise_probe(shape, block);
        let exe = match self.client.compile(&module) {
            Ok(e) => e,
            Err(e) => return ProbeVerdict::Unsupported { detail: e.to_string() },
        };
        let mem = exe.memory_analysis();
        let dense = (shape[0] * shape[1] * 4) as u64;

        // The dequantised matrix is k*n*f32. If it appears in scratch the
        // backend built it; if the arguments vanished it folded it.
        if mem.argument_bytes == 0 {
            ProbeVerdict::ConstantFolded
        } else if mem.temp_bytes >= dense {
            ProbeVerdict::MaterialisesPerCall { temp_bytes: mem.temp_bytes }
        } else {
            ProbeVerdict::Fused
        }
    }
}
```

⭐ **The Python probe is the specification.**
[`gljax/probes/pjrt_cpu_quant_probe.py`](../probes/pjrt_cpu_quant_probe.py)
already produces these verdicts against a real plugin, and it runs **without
gljax existing** — which is how the next plugin gets classified before a line of
Rust is written for it. Keep it in step with the Rust, not the other way round.

⚠️ Verdicts cache per `(plugin_id, plugin_version, shape, block)`. **A plugin
upgrade invalidates them.** This is the direct lesson of §Q1: v1's central
assumption went unre-checked for two years.

---

# 6. Quantized KV cache

⛔ **Deferred — and not for want of a design.**

v1 planned to `uniform_quantize` K and V before writing them to the cache. That
op does not exist (§Q1). The natural replacement — store int8 with per-head
scales and dequantise in-graph on read — hits §Q4's wall in its worst possible
place: the KV cache is read **every step, for every layer**, so materialising it
back to f32 each time converts a memory saving into a bandwidth cost on the
single hottest tensor in decode.

⚠️ There is a reason to want it anyway, specific to gljax: ARTX05 records that
a Qwen3-1.7B full-context KV cache is **8.75 GiB against an 8 GB machine**, so
`max_seq_len` is already clamped to 2048. KV quantization is the one place where
halving bytes buys *context length* rather than throughput — the goal is right,
the mechanism is absent.

**Re-open when** `CapabilityProbe` returns `Fused` on any target, or a PJRT
plugin exposes a quantized cache primitive. The interface exists now so nothing
downstream has to change later:

```rust
/// KV cache element format.
///
/// ⛔ Only `F32`/`BF16` are reachable today. The quantized variant is
/// constructible but `KvCache::new` refuses it unless the probe returned
/// `Fused` — P5, and P4's automatic-fallback rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvFormat {
    F32,
    BF16,
    /// int8 quants plus one f32 scale per (layer, head, block) tile.
    Int8Blockwise { block: u32 },
}

impl KvCache {
    pub fn new(spec: KvSpec, verdict: ProbeVerdict) -> Result<Self, QuantError> {
        match (spec.format, verdict) {
            (KvFormat::Int8Blockwise { .. }, ProbeVerdict::Fused) => Ok(Self::quantized(spec)),
            (KvFormat::Int8Blockwise { .. }, v) => Err(QuantError::RefusedByProbe {
                what: "quantized KV cache",
                verdict: v,
                // ⛔ Always say what would change the answer. A refusal with no
                // exit is indistinguishable from a bug.
                remedy: "re-run CapabilityProbe on this plugin; needs Fused",
            }),
            _ => Ok(Self::dense(spec)),
        }
    }
}
```

---

# 7. P4 — the numeric gate

Correctness is never inferred from the probe. `EmitBlockwise` is enabled only
after its output has been compared against the `Materialise` path on the same
weights.

```rust
/// Run both deliveries on one layer and compare. Called once, at load.
///
/// ⚠️ Compares against the *host-dequantised f32 path*, not against an
/// absolute tolerance. The two compute the same arithmetic, so they should
/// agree to f32 rounding; a real divergence means the backend reassociated or
/// mis-lowered something — precisely P4's bug class.
pub fn gate(layer: &Layer, probe: ProbeVerdict) -> WeightDelivery {
    if probe != ProbeVerdict::Fused {
        return WeightDelivery::Materialise { as_dtype: DType::F32 };
    }
    let dense     = run_materialised(layer);
    let blockwise = run_blockwise(layer);
    let rel = max_rel_err(&dense, &blockwise);

    // ⭐ The probe measured 1.5e-05 to 2.3e-05 across five shapes on this
    // pattern. 1e-3 sits two orders above that: loose enough never to fire on
    // rounding, tight enough that a mis-lowering cannot hide under it.
    if rel < 1e-3 {
        layer.blockwise_delivery()
    } else {
        tracing::warn!(rel, layer = layer.name, "blockwise dequant diverged; falling back to f32");
        WeightDelivery::Materialise { as_dtype: DType::F32 }
    }
}
```

---

# 8. What ARTX10 v2 does **not** do

| Not done | Why |
|---|---|
| Emit `!quant.uniform`, `uniform_quantize`, `uniform_dequantize` | §Q1 — dialect unregistered; the module does not parse |
| Represent GQ4A as a StableHLO quantized type | §Q6 — needs `(in/32) × out` scales; per-axis is one-dimensional. A mismatch in kind, not a gap to be filled |
| Write a CUDA / Triton / PTX quantized kernel | P1 — gljax owns no kernels |
| Enable `EmitBlockwise` by default | §Q4 — on the only plugin measured it is a bandwidth **regression**. P5 |
| Bake quantized weights as constants | §Q4 — XLA folds them to dense f32: same memory as host dequant, longer compile |
| Rely on `int4` for memory savings | §Q4 — `int4` and `int8` both measure 1.00 byte/element; not bit-packed |
| Ship a quantized KV cache | §6 — mechanism absent, and the failure mode lands on the hottest tensor in decode |
| Claim any GPU speedup | §Q5 — the fusion flag exists, **no benchmark numbers were found**, no GPU was available |

---

# 9. Open questions

1. ⛔ **Does the GPU subchannel fusion actually avoid materialisation?** §Q5
   confirms the pattern, not the payoff, and XLA's own note says it can be
   slower. **Probe required — and it is the same probe**: run
   `pjrt_cpu_quant_probe.py` against a CUDA plugin and read `temp_bytes`. Until
   then GPU quantization is `Unknown`, which `QuantPlan` treats as
   `Materialise`.
2. ⚠️ **Do donated buffers change the CPU verdict?** The measurement used
   ordinary arguments. If donation lets XLA dequantise in place, the temp cost
   changes character. Untested, and cheap to test.
3. ⭐ **Does batch size move the line?** Measured at batch 1. Prefill amortises
   a materialised weight across many tokens, so `MaterialisesPerCall` may be
   acceptable for **prefill** and not for **decode** — which would make delivery
   a *per-phase* choice, not a per-model one. Worth flagging that glproc reached
   the same per-stage conclusion independently for its CPU kernels; when two
   engines converge on it, it is probably structural.
4. ⚠️ **What does the StableHLO Quantizer emit?** §Q2 unresolved. Only matters
   if gljax ever ingests a Quantizer-produced module, which it does not today.
5. 🕐 **GQ2A** (2.625 bpw, three scale levels) is not designed here. It extends
   `ScaleLevels` with a third term — one more `multiply`, one more chance to
   break a fusion that is not happening anyway. Revisit after question 1.
