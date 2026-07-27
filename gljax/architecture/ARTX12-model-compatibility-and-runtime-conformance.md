# ARTX12 — Model Compatibility & Runtime Conformance

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded (Part A §A1–§A7 + Part B §1–§13)
**Depends on:** ARTX2 (IR / `dot_general` emission), ARTX3 (ops layer), ARTX8 (matrix compute), ARTX10 §1.3 (checkpoint-borne transformations), ARTX11 §4 (`Architecture` descriptor)
**Related:** ARTX1 §3.4 (FP64 oracle), ARTX5 (KV cache shapes), ARTX16 (serving)
**Introduces:** `gljax/tests/correctness/` + `gljax/src/oracle/`
**Next:** [ARTX13 — Tokenization Architecture](ARTX13-tokenization-architecture.md)
**Research grounded:** 2026-07-27 (sources at end)

---

> ## ⚠️ Document status
>
> ARTX12 covers two halves of one problem — *"which models does gljax claim to support, and what
> evidence backs the claim?"*
>
> | | Scope | Sections |
> |---|---|---|
> | **Part A — Model Compatibility** | GGUF · Safetensors · GPTQ · AWQ · SmoothQuant · FP8 variants · checkpoint compatibility · runtime capability matrix | **§A1–§A7** |
> | **Part B — Runtime Conformance** | The MatMul correctness harness: 4 oracle tiers, derived tolerances, failure-mode map | **§1–§13** |
>
> The two halves answer one question — *"which models does gljax claim to support, and what evidence
> backs the claim?"* Part A defines what must be loaded correctly; Part B defines how correctness is
> demonstrated. §A5's capability matrix is where they meet: a `Verified` entry requires Part B
> evidence, enforced by a test.
>
> Part A's motivation is recorded in [ARTX10 §1.3](ARTX10-quantized-runtime-architecture.md):
> checkpoints carry algorithm-specific transformations (SmoothQuant folds a smoothing vector into the
> preceding norm; AWQ bakes per-channel scales into the weights; GPTQ may permute weight order) that
> are invisible from tensor shapes and produce fluent wrong output rather than an error.
>
> **Numbering:** Part A uses §A1–§A7; Part B keeps its original §1–§13. They are independent.

---

# Part A — Model Compatibility

# A1. The Bug Class Part A Exists to Catch

Part B catches wrong *math*. Part A catches wrong *interpretation of bytes* — and the two are
different problems with the same symptom.

> **A checkpoint is not just weights. It is weights plus a set of transformations already applied to
> them, recorded nowhere in the tensor shapes.**

| Transformation | What it did to the weights | What a naive loader sees |
|---|---|---|
| **SmoothQuant** | Folded `s_j = max(\|X_j\|)^α / max(\|W_j\|)^(1−α)` into the **preceding norm's** weights | Normal norm weights of the right shape |
| **AWQ** | Baked per-channel scales into the weights; packed 4-bit values **interleaved** | An `int32` tensor of the right shape |
| **GPTQ w/ act_order** | Permuted the input-feature order; `g_idx` records the mapping | An `int32` tensor of the right shape |
| **FP8** | Scaled activations/weights to the FP8 range; scales live beside them | An `f8E4M3FN` tensor of the right shape |

⛔ **Every row produces fluent wrong output if misread. None produces an error.**

## A1.1 This repo has already paid for this lesson once

GwenLand's Q6_K dequantization used a naive **linear** nibble order where the format specifies a
different one. It silently corrupted `ffn_down.weight` in every layer, shapes were correct, nothing
raised, and the output stayed grammatical. It was found by an end-to-end run noticing the text was
wrong, then traced backwards.

⚠️ **AWQ's packing is the identical trap.** AutoAWQ packs eight 4-bit integers into one 32-bit
integer using an **interleaved** pattern — `0x1`–`0x8` are stored as `0x86427531`, not `0x87654321` —
chosen for unpack performance. A loader that assumes natural order reads plausible-looking garbage.

**Part A's job is to make every such assumption explicit, checked, and refused when unverifiable.**

---

# A2. Container Formats

| Format | Carries | gljax status |
|---|---|---|
| **safetensors** | Tensors + a JSON header; config in a sibling `config.json` | ARTX04 — supported |
| **`.gllm`** | GwenLand's own package | ARTX04 — supported |
| **GGUF** | Tensors **and** metadata **and** tokenizer vocabulary, in one file | ⬜ **not yet** |

## A2.1 GGUF

Four sections: **header** (magic, version, section counts) → **metadata KV** → **tensor info** (name,
dims, type, offset per tensor) → **tensor data**.

* Metadata is a **typed key-value** store using a `namespace.property` convention
  (`general.architecture`, `llama.context_length`). This was the deliberate improvement over earlier
  formats' fixed untyped hyperparameter lists: new metadata can be added without breaking existing
  readers.
* **Alignment** comes from `general.alignment`, defaulting to **32 bytes**, with `0x00` padding to
  the next multiple.
* **Version 3** is current and added optional big-endian support.
* Quantization type is **per tensor**, not per file.

⚠️ **DESIGN DECISION — GGUF support is read-only, and gljax parses it directly rather than depending
on `glcore`'s parser.** ARTX13 §0.2 already flagged that depending on `glcore` may pull GGUF parsing
and quantization kernels into gljax's build. A read-only GGUF reader is a few hundred lines against a
stable, well-documented format; inheriting an entire crate to get it is the worse trade.

⚠️ Per-tensor quantization types mean a single GGUF can mix formats — GwenLand's own measurements
found Qwen2.5-1.5B-q4_k_m is **75.7% Q4_K + 24.3% Q6_K by weight**. A loader that reads
`general.file_type` and assumes it applies uniformly is wrong.

---

# A3. Quantized Checkpoint Layouts

## A3.1 GPTQ

```text
qweight : [in_dim / pack_size, out_dim]     int32, 8×4-bit packed
qzeros  : [in_dim / group_size, out_dim / pack_size]
scales  : [in_dim / group_size, out_dim]
g_idx   : [in_dim]                          present only with act_order
```

`g_idx` maps each input feature to its group: the scale for input feature `i` is
`scales[g_idx[i]]`, and likewise for `qzeros`.

⚠️ **`g_idx` is the act_order flag in disguise.** Its presence means the input features were
**reordered** during quantization. A loader that ignores it applies the right scales to the wrong
channels — shapes valid, output fluent, weights wrong. ⛔ **If `g_idx` is present and not honoured,
refuse to load.**

## A3.2 AWQ

Three safetensors entries per linear layer — packed weights, zero-points, scales — with

```text
weight = scale × (qweight − qzero),   per group of 128
```

⚠️ And the interleaved packing of §A1.1. **The unpack order is not inferable from the file**; it is a
property of the producing tool. gljax must key it off `quantization_config` in `config.json` and
**refuse an unrecognized producer** rather than guessing.

## A3.3 SmoothQuant

⛔ The one with no tensor-level fingerprint at all. Its smoothing vector is folded into the
**preceding layer's normalization weights**, so the checkpoint contains ordinary-looking norm weights
that are *not* the model's original norm weights.

⚠️ **DESIGN DECISION — SmoothQuant is detected from `config.json`'s `quantization_config` only, and
an unrecognized W8A8 config is refused.** There is no safe fallback: loading a SmoothQuant checkpoint
as if it were plain INT8 produces a model that runs and is wrong.

## A3.4 FP8

Per ARTX10 §3.2: XLA's FP8 path casts up, applies **calibrated scales**, runs the Dot wider, and
rescales — all fused. The scales come from the checkpoint. Variants: `f8E4M3FN`, `f8E5M2`, and the
`FNUZ` forms; scale granularity may be per-tensor or per-channel.

⚠️ A checkpoint's FP8 *variant* must be read, not assumed — `f8E4M3FN` and `f8E4M3FNUZ` have
different exponent bias and NaN handling, and mistaking one for the other shifts every weight.

---

# A4. Architecture Detection

ARTX11 §4 introduced the `Architecture` descriptor after finding **seven** Qwen2-specific assumptions
baked into ARTX03. Part A owns populating it from a checkpoint.

```rust
// gljax/src/compat/detect.rs

pub fn detect(src: &CheckpointSource) -> Result<ModelIdentity, CompatError> {
    let arch = match src {
        // GGUF: general.architecture — authoritative, typed.
        CheckpointSource::Gguf(f) => f.metadata_str("general.architecture")?,
        // HF: config.json model_type.
        CheckpointSource::Safetensors { config, .. } => config.model_type.as_str(),
        CheckpointSource::Gllm(p) => p.manifest().architecture.as_str(),
    };
    // ⚠️ REGISTRY LOOKUP, never inference. An unknown architecture is an
    // error, not a reason to fall back to "probably like Qwen2".
    arch::registry::lookup(arch)
        .ok_or_else(|| CompatError::UnknownArchitecture(arch.to_string()))
}
```

⚠️ **DESIGN DECISION — unknown architectures are refused, never approximated.**
The tempting fallback ("it has the same tensor names as Llama, treat it as Llama") is exactly how
Gemma's seven differences (ARTX11 §4.2) would be silently ignored — `(1 + weight)` norms loaded as
plain norms, GeGLU run as SwiGLU. Each is fluent-wrong.

---

# A5. ⭐ The Runtime Capability Matrix

The deliverable that ties Part A to the rest of the series: **what gljax claims to support, and what
evidence backs each claim.**

```rust
// gljax/src/compat/matrix.rs

#[derive(Debug, Clone, Serialize)]
pub struct SupportEntry {
    pub architecture: &'static str,        // "qwen2" | "qwen3" | "gemma3" | "gemma4"
    pub container: ContainerFormat,        // Safetensors | Gguf | Gllm
    pub weight_format: WeightFormat,       // Bf16 | Gptq{..} | Awq{..} | Fp8{..} | Pridwen{..}
    pub level: SupportLevel,
    /// ⭐ What was actually run. Empty = the claim is unbacked.
    pub evidence: Vec<Evidence>,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub enum SupportLevel {
    /// Loads, runs, and passes Part B's harness + a perplexity check vs a reference.
    Verified,
    /// Loads and runs; no reference comparison yet.
    Loads,
    /// Parsing exists; never executed.
    Parsed,
    /// Explicitly known-unsupported, with a reason.
    Unsupported(&'static str),
}

#[derive(Debug, Clone, Serialize)]
pub enum Evidence {
    /// Part B's oracle tiers (§1–§13 below).
    HarnessTier { tier: &'static str, commit: String },
    /// Perplexity vs llama.cpp on a named corpus.
    PerplexityParity { reference: String, delta_pct: f64 },
    /// Token-ID parity for the tokenizer (ARTX13 A13.0).
    TokenizerParity { reference: String },
}
```

⚠️ **DESIGN DECISION — `SupportLevel::Verified` requires `evidence`, enforced by a test.**

```rust
#[test]
fn every_verified_entry_has_evidence() {
    for e in matrix::ALL.iter().filter(|e| matches!(e.level, SupportLevel::Verified)) {
        assert!(!e.evidence.is_empty(),
            "{} / {:?} / {:?} claims Verified with no evidence", e.architecture, e.container, e.weight_format);
    }
}
```

⭐ This is the mechanism that stops the support matrix from becoming aspirational. A README table can
drift from reality silently; a table that fails CI when a claim is unbacked cannot.

⚠️ The matrix is also **served**: ARTX16's `/v1/models` and `/health` expose the entry for the loaded
model, so an operator can see at runtime whether the thing they deployed is `Verified` or merely
`Loads`.

---

# A6. Load-Time Validation

```text
open checkpoint
   ▼
V1  container parses; magic + version recognized
   ▼
V2  architecture resolves in the registry            ⛔ unknown → refuse (§A4)
   ▼
V3  quantization_config recognized                   ⛔ unknown producer → refuse (§A3.2, §A3.3)
   ▼
V4  every expected tensor present, correct shape
   ▼
V5  g_idx honoured if present                        ⛔ present + unhandled → refuse (§A3.1)
   ▼
V6  tokenizer vocab_size vs model vocab_size         (ARTX13 §2.3)
   ▼
V7  capability matrix lookup → warn if not Verified
   ▼
load
```

⚠️ **DESIGN DECISION — every validation failure refuses the load; none downgrades to a warning.**
Except V7, which is genuinely informational. The others all guard transformations whose failure mode
is silent wrongness, and this series' repeated lesson is that silent wrongness costs more to find
than a refused load costs to fix.

---

# A7. Part A Wave Plan

| Wave | Scope | Gate |
|---|---|---|
| **A12.A1** | `compat/detect.rs` + `arch/registry.rs` population | Qwen2/Qwen3/Gemma3/Gemma4 resolve; an unknown `model_type` refuses |
| **A12.A2** | GGUF reader (read-only, §A2.1) | Tensor bytes byte-identical to `glcore`'s reader on the same file |
| **A12.A3** | GPTQ + AWQ loaders (§A3.1–A3.2) | ⭐ Dequantized weights bit-match a reference (AutoGPTQ / AutoAWQ) — **the interleave check** |
| **A12.A4** | SmoothQuant + FP8 detection (§A3.3–A3.4) | Recognized configs load; unrecognized W8A8 refuses |
| **A12.A5** | `compat/matrix.rs` + the evidence test (§A5) | CI fails on any `Verified` entry lacking evidence |
| **A12.A6** | Validation pipeline (§A6) wired into `Session::new` | Each of V1–V6 has a negative test that refuses |

⚠️ **A12.A3's gate is the one that matters most.** Bit-matching a reference dequant is the only check
that catches an interleave or `g_idx` misread; a shape check, a load test, and even a short
generation would all pass with the weights scrambled.

## A7.1 Open questions

1. **Which AWQ producers exist in the wild, and do they all use the same interleave?** §A3.2 assumes
   AutoAWQ's; others may differ.
2. **Does `.gllm` need a `quantization_config` equivalent?** Pridwen's GQ4A/GQ2A are declared in the
   manifest, but SmoothQuant-style folded transformations have no field yet.
3. **How is the matrix maintained as models are added** — hand-written, or generated from test runs?
   Generated is better and harder.

---

# Part B — Runtime Conformance

*(Originally authored standalone as the MatMul Algorithm Correctness Harness. Sections §1–§13 below
are Part B's own numbering.)*

---

# Reality Check

Per ARTX16's opening: **`gljax` has no code and is not a workspace member.** ARTX12 specifies a test
harness for an engine that does not yet exist.

That is not wasted work — a correctness harness written *before* the implementation is a
specification of what "correct" means, and it is the only ordering that avoids writing tests that
merely ratify whatever the code already does. But it fixes the sequencing:

| Test tier | Blocked on |
|---|---|
| **T0** pure-Rust f64 reference | Nothing. Buildable today. |
| **T1** StableHLO interpreter conformance | ARTX2 (`FuncBuilder` emitting MLIR text) |
| **T2** FP64 CPU-plugin oracle | ARTX1 (PJRT FFI) + ARTX4 (`Session`) |
| **T3** cross-engine differential | ARTX5 (`Session::generate()`) |

Bullets 1–2 of the brief (classification, dimension numbers) are **T0 and testable the day
`matrix/spec.rs` exists**. Bullet 8 (logits vs glproc) is T3 and is the last thing to land.

## Scope decisions taken for this document

| Question | Decision |
|---|---|
| Deliverable | **Design document.** Rust below is implementation-ready but not yet compiled. |
| glproc as reference | **Demoted to a divergence detector, not a correctness oracle** (§2.4). Ground truth moves to T0–T2 plus llama.cpp. |
| Harness scope | **gljax only.** The oracle layer is not generalized into an engine-agnostic trait. glproc's own precision investigation stays a separate effort. |

---

# 1. The Bug Class This Harness Exists to Catch

A correctness harness is only worth its maintenance cost if it targets a bug class that is both
**real** and **invisible to ordinary testing**. For matmul in an LLM engine that class is sharply
defined:

> **Shape-valid, type-valid, non-crashing errors that produce fluent but wrong output.**

Every bug in this class shares three properties: the tensor shapes are right, no assertion fires,
and the model still emits grammatical text. Perplexity moves; nothing else does.

**GwenLand has already been bitten by exactly this, twice, in shipping code:**

1. **The Q6_K dequant order bug.** `glcore` used a naive linear nibble order instead of the correct
   one, silently corrupting `ffn_down.weight` in **every layer**. Shapes were correct, no error was
   raised, and output was fluent — it was caught only by an end-to-end run that noticed the text was
   wrong, then traced back. (Resolved; routed through `glproc`'s dequant.)
2. **glproc's unexplained precision gap.** A confirmed ~46% perplexity gap versus llama.cpp on
   identical GGUF weights (36.12 vs 24.78). Repack accounts for ~9 points; SIMD and `fast_exp` were
   both ruled out by a forced-scalar run that did *not* help. **~33 percentage points remain
   unexplained**, and the investigation narrowed the candidates to *architectural formula matching*
   — RMSNorm epsilon placement and reduction order, softmax numerics, or RoPE convention. **Still
   open.**

⚠️ Item 2 is the reason for this document's most important design decision (§2.4). An engine with a
33-point unexplained numerical defect cannot serve as the correctness oracle for another engine.

## 1.1 The specific failure modes in scope

| # | Failure | Why ordinary tests miss it | Section |
|---|---|---|---|
| F1 | Transposed / swapped `dot_general` dimension numbers | Square matrices give the right output *shape* | §5 |
| F2 | GQA group mapping uses block layout where the checkpoint used interleaved | Identical shapes; attention is silently corrupted | §7 |
| F3 | Attention scale uses `1/sqrt(hidden_dim)` instead of `1/sqrt(head_dim)` | Valid shape; only softmax temperature is wrong | §8 |
| F4 | Fused gate/up segments split at the wrong offset | Both halves have the right shape (ARTX8 §fusion) | §9 |
| F5 | `lm_head` contracts the wrong axis on a tied-embedding model | `[V,D]` vs `[D,V]` both "work" if you transpose somewhere else | §10 |
| F6 | Accumulation happens in BF16 rather than FP32 | Output is plausible, precision quietly halves | §6 |
| F7 | Batch dims mis-assigned in batched attention `dot_general` | Heads get silently mixed | §5, §8 |

---

# 2. Oracle Architecture

⚠️ **DESIGN DECISION — four oracle tiers, each answering a different question.**
A single oracle cannot distinguish "wrong algorithm" from "different precision" from "different
model". Separating them is what makes a failure *diagnostic* rather than merely red.

```text
T0  Pure-Rust f64 reference          "Is the algorithm right?"
      │  no PJRT, no MLIR, runs in `cargo test`
      ▼
T1  StableHLO reference interpreter  "Did we emit the MLIR we think we did?"
      │  no device, spec-conformant
      ▼
T2  FP64 CPU plugin (ARTX1 §3.4)     "Does the real compiler+runtime agree?"
      │  real PJRT, real XLA
      ▼
T3  Cross-engine differential        "Do we diverge from another engine, and where?"
       NOT a correctness oracle — see §2.4
```

## 2.1 T0 — pure-Rust f64 reference

The naive triple loop, in `f64`, with no optimization whatsoever. Its only virtues are that it is
obviously correct by inspection and that it has no dependencies.

```rust
// gljax/src/oracle/reference.rs
// Deliberately naive. Do NOT optimize this file — its value is auditability.

/// Reference `dot_general` in f64. Mirrors the StableHLO spec directly:
/// output dims = batch dims, then lhs free dims, then rhs free dims.
pub fn dot_general_f64(
    lhs: &TensorF64, rhs: &TensorF64, dnums: &DotDimensionNumbers,
) -> TensorF64 {
    let batch: Vec<usize> = dnums.lhs_batching.iter().map(|&i| lhs.dims[i]).collect();
    let lhs_free = free_dims(&lhs.dims, &dnums.lhs_batching, &dnums.lhs_contracting);
    let rhs_free = free_dims(&rhs.dims, &dnums.rhs_batching, &dnums.rhs_contracting);
    let contract: Vec<usize> =
        dnums.lhs_contracting.iter().map(|&i| lhs.dims[i]).collect();

    let out_dims: Vec<usize> = batch.iter().chain(&lhs_free).chain(&rhs_free).copied().collect();
    let mut out = TensorF64::zeros(&out_dims);

    for b in index_space(&batch) {
        for m in index_space(&lhs_free) {
            for n in index_space(&rhs_free) {
                // Sequential f64 accumulation. No FMA, no reassociation,
                // no pairwise summation — the reference must be boring.
                let mut acc = 0.0f64;
                for k in index_space(&contract) {
                    acc += lhs.at(&scatter(&b, &m, &k, LhsAxes(dnums)))
                         * rhs.at(&scatter(&b, &n, &k, RhsAxes(dnums)));
                }
                out.set(&concat3(&b, &m, &n), acc);
            }
        }
    }
    out
}
```

⚠️ **DESIGN DECISION — the reference must not be optimized, ever.**
The moment `reference.rs` uses FMA, blocking, or parallel reduction it shares failure modes with the
thing it validates. A reference that is slow is fine; the harness runs it on small shapes (§4.3).

## 2.2 T1 — StableHLO reference interpreter

This is the tier the brief did not include and which is the strongest available check on
`dot_general` *semantics*.

OpenXLA ships a reference interpreter (`stablehlo-translate --interpret`) whose stated requirement is
**1:1 correspondence with the StableHLO specification**; it covers 91 of the 96 specified ops, uses a
`Check` dialect to compare runtime values against expected values, and ships a testdata suite of
~3,000 files with gold results intended for vendor integration testing.

What this buys gljax specifically: it validates **the MLIR text gljax emits**, independent of any
plugin, any device, and any XLA optimization. If gljax's `emit_dot_general` writes dimension numbers
that do not mean what gljax thinks they mean, the interpreter disagrees with T0 — and the failure
localizes to *emission*, not to math.

```rust
// gljax/tests/correctness/t1_interpreter.rs

/// Emit a module through the real FuncBuilder, run it in the StableHLO
/// interpreter, compare to the T0 f64 reference.
fn interpret_and_compare(case: &MatmulCase) -> Result<Comparison, HarnessError> {
    let mlir = build_matmul_module(case);            // ARTX2 FuncBuilder
    let got  = stablehlo_interpret(&mlir, &case.inputs)?;
    let want = reference::dot_general_f64(&case.lhs, &case.rhs, &case.dnums);
    Ok(compare(&got, &want, case.tolerance()))
}
```

⚠️ **DESIGN DECISION — the interpreter is invoked as an external binary, not linked.**
Linking MLIR/LLVM into gljax's test build would contradict ARTX1's pure-Rust, zero-heavy-dependency
posture. `stablehlo-translate` is invoked as a subprocess and the tier is **skipped with a clear
message** when the binary is absent, so `cargo test` still works on a machine without it.

```rust
#[test]
fn t1_dot_general_matches_spec() {
    let Some(bin) = stablehlo_translate_path() else {
        eprintln!("SKIP t1: stablehlo-translate not found (set STABLEHLO_TRANSLATE)");
        return;
    };
    // ...
}
```

## 2.3 T2 — FP64 CPU-plugin oracle

ARTX1 §3.4 already specifies this: compile an FP64 program, run it on the CPU plugin, compare. It is
the only tier that exercises the real PJRT path, the real XLA compiler, and gljax's real buffer
plumbing at once.

⚠️ Constrained by hardware, per ARTX1 §3.1: FP64 is full on the CPU plugin and CUDA (at ~1/32 BF16
throughput), **absent on TPU v5e**. T2 is a CPU-and-CUDA tier and must be `#[cfg]`-gated, never run
in a TPU CI lane.

⚠️ T2's diagnostic value is precisely the *delta from T1*: if T1 passes and T2 fails, the emitted
MLIR is right and something in the compile/execute path is wrong — donation aliasing, layout, or
buffer plumbing. That two-tier split is what turns "the numbers are wrong" into a bisected answer.

## 2.4 T3 — cross-engine differential, and why glproc is **not** an oracle

The brief's bullet 8 reads: *"Integration: compare logits vs glproc FP32 (expected relative L2 <
0.05)."*

⛔ **glproc cannot serve as a correctness oracle, for two independent reasons.**

**Reason 1 — glproc has a large, confirmed, unexplained numerical defect.** §1 above: ~33 percentage
points of a ~46% perplexity gap versus llama.cpp remain unexplained and are believed to be
*architectural formula* differences in RMSNorm, softmax, or RoPE. Those are shared-by-construction
concerns: gljax's ARTX3 implements the same three ops. If gljax agrees with glproc to within 5%, the
most likely explanations are that both are right **or that both are wrong in the same way** — and the
test cannot distinguish them. A test that passes when both engines are broken is worse than no test,
because it confers confidence.

**Reason 2 — the two engines run different numerical pipelines.** glproc serves Q8_0-repacked
quantized weights on CPU in FP32; gljax serves BF16 on an accelerator. A logits difference between
them is dominated by *quantization error*, not by matmul correctness. The measurement does not
isolate the variable the test claims to test.

⚠️ **DESIGN DECISION — T3 is retained, renamed, and re-purposed.**

```text
T3 is a DIVERGENCE DETECTOR, not a correctness oracle.

  Purpose:  answer "where do these two engines first disagree?"
  Not:      answer "which one is right?"
  Gate:     a divergence REPORT, never a pass/fail assertion in CI.
```

It earns its place as a **layer-bisection tool**: run both engines on the same tokens, capture the
residual stream after each layer, and report the first layer where relative divergence exceeds the
expected quantization floor. That localizes a disagreement to a layer and an op — which is exactly
the tool glproc's own open investigation needs, and exactly what a raw end-to-end L2 number cannot
give.

**Ground truth moves up a tier.** Where an external absolute reference is needed, use **llama.cpp**,
which is already cloned locally as a sibling directory (`C:\Users\reyha\Documents\JinXSuper-Projects\
llama.cpp`, commit `910196f`) and is the reference glproc's gap was measured against in the first
place. It is not perfect either, but it is the reference the rest of the ecosystem calibrates to, and
crucially it is *independent* of GwenLand's own implementations.

---

# 3. Tolerance Derivation

⚠️ **DESIGN DECISION — every tolerance in this harness is derived, not chosen.**
A magic constant in a numerical test is a future false pass or a future flake. The derivations below
are short enough to re-run when a dtype or dimension changes.

## 3.1 The error model

Standard floating-point unit roundoff `u` (half of machine epsilon):

| Format | Machine ε | Unit roundoff `u` |
|---|---|---|
| BF16 | 2⁻⁷ ≈ 7.81e-3 | 2⁻⁸ ≈ **3.91e-3** |
| FP32 | 2⁻²³ ≈ 1.19e-7 | 2⁻²⁴ ≈ **5.96e-8** |
| FP64 | 2⁻⁵² ≈ 2.22e-16 | 2⁻⁵³ ≈ **1.11e-16** |

The classical GEMM error bound (Higham; the criterion LAPACK is judged by) is

```text
|fl(AB)ᵢⱼ − (AB)ᵢⱼ|  ≤  f(n) · ε · (|A||B|)ᵢⱼ
```

with `f(n)` required not to exceed linear growth for a well-behaved implementation.

**The mixed-precision refinement that matters here**, from ARTX8 §A8.3: both the TPU MXU and NVIDIA
Tensor Cores multiply in BF16 but **accumulate in FP32**. So the two error sources have very
different weights:

```text
For a dot product of length K, bf16 inputs, fp32 accumulation:

  input rounding      ≈ 2·u_bf16          (one rounding per operand)
  accumulation        ≈ K·u_fp32          (sequential summation)

  δ_rel  ≈  2·u_bf16 + K·u_fp32
```

## 3.2 Per-matmul tolerance

```text
K = 896   (Qwen2.5-0.5B):  2(3.91e-3) + 896(5.96e-8)  = 7.81e-3 + 5.3e-5 ≈ 7.9e-3
K = 2048  (Qwen3-1.7B):    7.81e-3 + 1.2e-4                            ≈ 7.9e-3
K = 8192  (70B-class):     7.81e-3 + 4.9e-4                            ≈ 8.3e-3
```

⚠️ **The input rounding dominates by a factor of ~16–150×.** Accumulation length is almost
irrelevant while accumulation is FP32. Two consequences:

* **A single per-matmul tolerance of `1e-2` is correct across every model size gljax targets.** It
  does not need to scale with K.
* **If a test starts failing when K grows, the accumulator is not FP32.** That is failure mode F6,
  and this tolerance's flatness is what makes it detectable.

## 3.3 End-to-end tolerance — the brief's 0.05 checks out

Treating per-layer errors as independent (they are not exactly, but this is the standard estimate),
error through `L` layers grows as `√L`:

```text
δ_logits ≈ √L · δ_matmul

L = 24  (Qwen2.5-0.5B):  √24  × 8e-3 = 3.9e-2
L = 28  (Qwen3-1.7B):    √28  × 8e-3 = 4.2e-2
L = 32  (7B-class):      √32  × 8e-3 = 4.5e-2
L = 80  (70B-class):     √80  × 8e-3 = 7.2e-2
```

⚠️ **The brief's `relative L2 < 0.05` is well-calibrated for models up to ~32 layers** — it sits just
above the predicted 3.9–4.5e-2 with modest headroom. It is *not* a safe constant for an 80-layer
model, where the prediction already exceeds it. The harness therefore computes the threshold rather
than hardcoding it:

```rust
// gljax/src/oracle/tolerance.rs
pub const U_BF16: f64 = 3.906_25e-3;   // 2^-8
pub const U_FP32: f64 = 5.960_46e-8;   // 2^-24

/// Expected relative error for one bf16-input / fp32-accumulate matmul.
pub fn matmul_tolerance(k: usize) -> f64 {
    2.0 * U_BF16 + (k as f64) * U_FP32
}

/// Expected relative error at the logits after `n_layers`, with a safety factor.
/// SAFETY_FACTOR covers non-independence of per-layer errors and normalization
/// effects; 1.5 is chosen so the 24-32 layer case lands near the brief's 0.05.
pub const SAFETY_FACTOR: f64 = 1.5;

pub fn logits_tolerance(n_layers: usize, k: usize) -> f64 {
    SAFETY_FACTOR * (n_layers as f64).sqrt() * matmul_tolerance(k)
}
```

## 3.4 Tolerance table

| Comparison | dtype path | Tolerance | Basis |
|---|---|---|---|
| T0 f64 reference vs gljax **f64 oracle mode** | f64 → f64 | `1e-12` | Testing *algorithm*, not precision. Anything looser hides a real bug. |
| T0 f64 reference vs gljax **bf16** | bf16 → f32 acc | `matmul_tolerance(K)` ≈ `1e-2` | §3.2 |
| T1 interpreter vs T0 | f64 → f64 | `1e-12` | Both are f64 spec implementations |
| T2 FP64 plugin vs T0 | f64 → f64 | `1e-10` | Slightly looser: XLA may reassociate |
| T3 gljax bf16 vs glproc fp32-quantized | mixed | **no gate** | §2.4 — report only |

⚠️ The `1e-12` rows are the load-bearing ones. Running gljax in `PrecisionPolicy::f64_oracle()` mode
(ARTX2 §8) removes precision from the equation entirely, so any disagreement is an **algorithm**
disagreement. **This is the single most valuable configuration in the harness** — most tests should
run here, and the bf16 tolerance tests exist mainly to catch F6.

---

# 4. Comparison Metrics

## 4.1 ⛔ Why relative L2 on raw logits is the wrong gate

The brief specifies relative L2 on logits. It is a fine *tripwire* and a poor *gate*, for two
reasons:

**Softmax is shift-invariant.** `softmax(z + c·1) = softmax(z)` exactly. A uniform additive offset in
the logits changes relative L2 arbitrarily while changing the model's output **not at all**. A gate
on raw-logit L2 can fail on a difference that provably does not exist downstream.

**L2 does not map monotonically to output difference.** A 3% error concentrated on the top-1 logit
can flip the argmax and change every generated token. A 3% error spread across 151,000 tail logits
changes nothing. One number cannot distinguish them.

⚠️ **DESIGN DECISION — gate on top-1 agreement and KL; keep centered relative L2 as a tripwire.**

```rust
// gljax/src/oracle/metrics.rs
pub struct LogitsComparison {
    /// Fraction of positions where argmax agrees. PRIMARY GATE.
    pub top1_agreement: f64,
    /// Fraction of positions where the top-5 sets agree (order-insensitive).
    pub top5_overlap: f64,
    /// KL(softmax(want) || softmax(got)), mean over positions. PRIMARY GATE.
    /// Shift-invariant by construction — the metric the brief's L2 wanted to be.
    pub kl_divergence: f64,
    /// Relative L2 AFTER mean-centering each position. Tripwire only.
    pub centered_rel_l2: f64,
    /// Raw relative L2, reported for continuity with the brief. Not a gate.
    pub raw_rel_l2: f64,
}

pub fn centered_rel_l2(got: &[f32], want: &[f32]) -> f64 {
    let (gm, wm) = (mean(got), mean(want));
    let num: f64 = got.iter().zip(want)
        .map(|(g, w)| { let d = (*g as f64 - gm) - (*w as f64 - wm); d * d })
        .sum::<f64>().sqrt();
    let den: f64 = want.iter()
        .map(|w| { let d = *w as f64 - wm; d * d })
        .sum::<f64>().sqrt();
    num / den
}
```

⚠️ `glbench/src/kl_divergence.rs` already exists in this repo and already implements the KL
comparison. ARTX12 should **reuse it rather than reimplement**, and the harness's job is to feed it
gljax logits.

## 4.2 Gate thresholds

| Metric | Gate | Rationale |
|---|---|---|
| `top1_agreement` | **≥ 0.99** vs T0/T2 oracle | Below this, generated text diverges visibly |
| `kl_divergence` | **≤ 1e-3** vs T0/T2 oracle | Distribution-level, shift-invariant |
| `centered_rel_l2` | `≤ logits_tolerance(L, K)` | §3.3, computed not hardcoded |
| `raw_rel_l2` | reported, never gated | §4.1 |

## 4.3 Test input design

⚠️ **DESIGN DECISION — adversarial shapes, not square, not random-only.**

```rust
/// Every dimension DISTINCT and non-power-of-two where possible.
/// A transposed dnums bug (F1) survives square matrices and dies here.
pub const ADVERSARIAL_SHAPES: &[(usize, usize, usize)] = &[
    (3, 5, 7),        // tiny, all prime, all distinct — the F1 detector
    (1, 896, 4864),   // GEMV, Qwen2.5-0.5B ffn_gate_up
    (17, 2048, 6144), // GEMM, non-power-of-2 M
    (8, 2048, 2048),  // square K=N — the case that HIDES F1; included on purpose
];
```

Input value patterns, each targeting a different failure:

| Pattern | Catches |
|---|---|
| **Index-encoded** — `A[i][k] = i*1000 + k` | F1, F5, F7 — a wrong axis produces obviously wrong magnitudes |
| **One-hot rows** | Isolates a single output element; exact by hand |
| **Distinguishable head signature** — KV head `h` filled with constant `h+1` | **F2** (§7) — the single highest-value pattern |
| Random `N(0,1)` | General numerical agreement |
| Large-magnitude (`~1e4`) mixed with small (`~1e-4`) | Catastrophic cancellation, accumulation width (F6) |

⚠️ Index-encoded inputs are worth more than random inputs for structural bugs. With
`A[i][k] = i*1000 + k`, a transposed contraction does not merely produce a different number — it
produces a number in a visibly wrong *range*, which is debuggable from the failure message alone.

---

# 5. Test Area 1 — GEMM vs GEMV Classification

**Target:** ARTX8 `matrix/spec.rs` — `MatmulSpec::classify`, `flops`, `bytes`,
`arithmetic_intensity`.

**Tier:** T0. Pure host-side logic; no PJRT, no MLIR, no device. Fastest tests in the harness.

```rust
// gljax/tests/correctness/t0_classification.rs

#[test]
fn gemv_when_all_lhs_free_dims_collapse_to_one() {
    // Decode: [1, 896] × [896, 4864]
    let spec = MatmulSpec::classify(
        &Shape::new(vec![1, 896], DType::BF16),
        &Shape::new(vec![896, 4864], DType::BF16),
        &dnums_contract(&[1], &[0]),
    );
    assert_eq!(spec, Contraction::Gemv);
}

#[test]
fn gemm_when_m_exceeds_one() {
    // Prefill: [512, 896] × [896, 4864]
    assert_eq!(
        MatmulSpec::classify(
            &Shape::new(vec![512, 896], DType::BF16),
            &Shape::new(vec![896, 4864], DType::BF16),
            &dnums_contract(&[1], &[0]),
        ),
        Contraction::Gemm
    );
}

#[test]
fn batched_when_batching_dims_present() {
    // Attention scores: [B, H, S, D] × [B, H, D, S]
    let spec = MatmulSpec::classify(
        &Shape::new(vec![4, 14, 128, 64], DType::BF16),
        &Shape::new(vec![4, 14, 64, 128], DType::BF16),
        &DotDimensionNumbers {
            lhs_batching: vec![0, 1], rhs_batching: vec![0, 1],
            lhs_contracting: vec![3], rhs_contracting: vec![2],
        },
    );
    assert_eq!(spec, Contraction::BatchedGemm { batch_rank: 2 });
}

/// ⚠️ The subtle one. A rank-3 decode activation [B, 1, D] must classify as
/// Gemv even though rank > 2 — classification is about the PRODUCT of free
/// dims, not the rank.
#[test]
fn rank3_decode_activation_is_still_gemv() {
    assert_eq!(
        MatmulSpec::classify(
            &Shape::new(vec![8, 1, 2048], DType::BF16),   // 8 slots, 1 token
            &Shape::new(vec![2048, 6144], DType::BF16),
            &dnums_contract(&[2], &[0]),
        ),
        Contraction::Gemv
    );
}
```

⚠️ **The rank-3 case is the one that will actually break.** ARTX7's continuous batching makes decode
activations `[max_slots, 1, D]`, not `[1, D]`. A `classify` that keys on `dims[0] == 1` instead of
`product(free_dims) == 1` gets this wrong, and the consequence is a mislabeled regime in
`roofline.rs` — wrong diagnosis, not wrong output. Low severity, high confusion cost.

**Property tests** (these need no oracle at all — they are self-checking invariants):

```rust
proptest! {
    /// FLOPs must be 2·M·N·K·batch for every classification.
    #[test]
    fn flops_matches_closed_form(spec in arb_matmul_spec()) {
        prop_assert_eq!(spec.flops(), 2 * spec.m() * spec.n() * spec.k() * spec.batch());
    }

    /// AI = flops/bytes, and for GEMV it must be ≈1 regardless of size (ARTX8 §A8.1).
    #[test]
    fn gemv_arithmetic_intensity_is_scale_invariant(k in 128usize..16384) {
        let spec = gemv_spec(k, 4 * k);
        prop_assert!((spec.arithmetic_intensity() - 1.0).abs() < 0.5,
            "GEMV AI must stay ~1 at K={k}, got {}", spec.arithmetic_intensity());
    }
}
```

---

# 6. Test Area 2 — `dot_general` Dimension Numbers

**Target:** ARTX2 `emit_dot_general` + `infer_dot_general_shape`; ARTX8 `matrix/lower.rs`.

**Tiers:** T0 (values) + T1 (emitted MLIR semantics).

## 6.1 Shape inference

```rust
/// Output dims must be exactly: batch dims, then lhs free, then rhs free —
/// in that order. StableHLO spec ordering; getting it wrong transposes output.
#[test]
fn output_dim_order_is_batch_lhsfree_rhsfree() {
    let out = infer_dot_general_shape(
        &Shape::new(vec![4, 7, 64], DType::F32),    // [B, M, K]
        &Shape::new(vec![4, 64, 11], DType::F32),   // [B, K, N]
        &DotDimensionNumbers {
            lhs_batching: vec![0], rhs_batching: vec![0],
            lhs_contracting: vec![2], rhs_contracting: vec![1],
        },
    );
    assert_eq!(out.dims, vec![4, 7, 11]);   // B, M, N — all distinct on purpose
}

#[test]
#[should_panic(expected = "contracting dim size mismatch")]
fn mismatched_contracting_dims_panic() {
    infer_dot_general_shape(
        &Shape::new(vec![7, 64], DType::F32),
        &Shape::new(vec![32, 11], DType::F32),      // 64 != 32
        &dnums_contract(&[1], &[0]),
    );
}
```

## 6.2 ⚠️ The output-dtype defect ARTX8 flagged

ARTX8 recorded that `infer_dot_general_shape` ends with `Shape::new(out_dims, lhs.dtype)` — output
dtype is unconditionally the lhs dtype, and `preferred_element_type` is not emitted at all. That is
failure mode **F6**, and it is directly testable once ARTX8 Wave A8.α lands:

```rust
/// After A8.α: requesting f32 accumulation must produce an f32 output shape.
/// Before A8.α: this test FAILS, and that failure is the point — it is the
/// executable statement of the defect.
#[test]
fn preferred_element_type_governs_output_dtype() {
    let out = infer_dot_general_shape_with(
        &Shape::new(vec![8, 2048], DType::BF16),
        &Shape::new(vec![2048, 6144], DType::BF16),
        &dnums_contract(&[1], &[0]),
        Some(DType::F32),                            // preferred_element_type
    );
    assert_eq!(out.dtype, DType::F32, "accumulation dtype must reach the output shape");
}

/// The emitted MLIR must actually carry the attribute — not just the Rust shape.
#[test]
fn emitted_mlir_contains_preferred_element_type() {
    let mlir = build_matmul_module(&case_with_accumulate(DType::F32));
    assert!(mlir.contains("preferred_element_type"),
        "A8.α regression: accumulation type dropped during emission");
}
```

## 6.3 The transposition test — failure mode F1

```rust
/// ⚠️ THE test for F1. Index-encoded inputs + all-distinct dims.
/// With A[i][k] = i*100 + k, a swapped contraction is off by orders of
/// magnitude, not by a rounding error.
#[test]
fn transposed_dnums_produce_different_values_not_just_different_shapes() {
    let a = TensorF64::index_encoded(&[3, 5]);      // M=3, K=5
    let b = TensorF64::index_encoded(&[5, 7]);      // K=5, N=7

    let correct = reference::dot_general_f64(&a, &b, &dnums_contract(&[1], &[0]));

    // A plausible typo: contract lhs dim 0 instead of dim 1.
    // Shapes happen to be legal here; values are not.
    let wrong_is_rejected = std::panic::catch_unwind(|| {
        reference::dot_general_f64(&a, &b, &dnums_contract(&[0], &[0]))
    });
    assert!(wrong_is_rejected.is_err(), "3 != 5 must be caught by shape validation");

    // And the correct result must match a hand-computed element.
    // out[0][0] = Σ_k a[0][k]·b[k][0] = Σ_k (0*100+k)·(k*100+0)
    let expected: f64 = (0..5).map(|k| (k as f64) * ((k * 100) as f64)).sum();
    assert!((correct.at(&[0, 0]) - expected).abs() < 1e-12);
}
```

---

# 7. Test Area 3 — GQA Expand / Broadcast

⚠️ **This is the highest-value section in ARTX12.** Failure mode F2 is the one that most closely
resembles the Q6_K bug that already shipped: right shapes, no error, fluent wrong output.

## 7.1 Why it is dangerous

In GQA, `n_q_heads` query heads share `n_kv_heads` key/value heads, with group size
`G = n_q_heads / n_kv_heads`. The mapping from query head to KV head has **two plausible
conventions that produce identically-shaped tensors**:

```text
n_q_heads = 8, n_kv_heads = 2, G = 4

Interleaved / repeat_interleave  (HuggingFace standard, what checkpoints expect):
   q head:  0  1  2  3  4  5  6  7
   kv head: 0  0  0  0  1  1  1  1      ← kv = q / G

Block / tile  (the plausible wrong one):
   q head:  0  1  2  3  4  5  6  7
   kv head: 0  1  0  1  0  1  0  1      ← kv = q % n_kv_heads
```

The grouping is **a convention fixed at training time**: a checkpoint trained with contiguous
interleaved groups must be served the same way, and swapping the pattern at inference **silently
corrupts attention**. HuggingFace standardized on `repeat_interleave`-style contiguous groups.

Both produce `[B, 8, S, D]`. Both run. Only one is right.

## 7.2 The test

```rust
// gljax/tests/correctness/t0_gqa_expand.rs

/// ⚠️ THE F2 detector. Each KV head carries a unique constant signature, so
/// after expansion every query head's source is directly readable.
#[test]
fn gqa_expand_uses_interleaved_grouping() {
    const N_Q: usize = 8;
    const N_KV: usize = 2;
    const G: usize = N_Q / N_KV;      // 4
    const S: usize = 3;
    const D: usize = 4;

    // KV head h is filled entirely with the value (h + 1).
    let mut kv = TensorF64::zeros(&[1, N_KV, S, D]);
    for h in 0..N_KV {
        for s in 0..S { for d in 0..D { kv.set(&[0, h, s, d], (h + 1) as f64); } }
    }

    let expanded = ops::gqa_expand_f64(&kv, N_Q);
    assert_eq!(expanded.dims, vec![1, N_Q, S, D]);

    for q in 0..N_Q {
        let expected = (q / G + 1) as f64;                 // INTERLEAVED
        let block_wrong = (q % N_KV + 1) as f64;           // the bug we're hunting
        let got = expanded.at(&[0, q, 0, 0]);

        assert_eq!(got, expected,
            "q head {q} must read kv head {} (interleaved). \
             Got {got}; block-grouping would give {block_wrong}.",
            q / G);
    }
}

/// The two conventions must be provably distinguishable by this fixture —
/// otherwise the test above proves nothing.
#[test]
fn interleaved_and_block_grouping_actually_differ_for_this_fixture() {
    let (n_q, n_kv) = (8usize, 2usize);
    let g = n_q / n_kv;
    let differs = (0..n_q).any(|q| q / g != q % n_kv);
    assert!(differs, "fixture cannot distinguish the two conventions — pick other dims");
}
```

⚠️ **The second test is not redundant.** With `n_q_heads = n_kv_heads` (MHA) or `n_kv_heads = 1`
(MQA), the two conventions coincide and the first test passes vacuously. Asserting that the fixture
*can* discriminate is what stops a future dim change from silently disarming the test.

## 7.3 Degenerate configurations

```rust
#[test] fn mha_is_identity_expansion()   { /* n_kv == n_q → expand is a no-op */ }
#[test] fn mqa_broadcasts_single_head()  { /* n_kv == 1  → every q head reads head 0 */ }

/// Qwen2.5-0.5B: n_q=14, n_kv=2, G=7. ⚠️ ARTX16 §3.1 notes this model cannot
/// run TP>2. The harness pins the real config so a refactor cannot quietly
/// assume power-of-two head counts.
#[test]
fn qwen25_05b_real_head_config() {
    assert_gqa_grouping(/* n_q */ 14, /* n_kv */ 2);
}
```

---

# 8. Test Area 4 — Attention Score Correctness

**Target:** ARTX3 `ops/attention.rs` — `gqa_attention`, `causal_mask`.

## 8.1 Shape and scale

```rust
/// QK^T: [B,H,S,D] × [B,H,S,D] contracting D → [B,H,S,S].
/// ⚠️ Note rhs_contracting = 3, not 2 — K is NOT pre-transposed.
/// Getting this wrong yields [B,H,S,D] which fails to compose with V,
/// so this one at least fails loudly. Included because it pins the convention.
#[test]
fn qk_transpose_dimension_numbers() {
    let out = infer_dot_general_shape(
        &Shape::new(vec![2, 14, 17, 64], DType::F32),
        &Shape::new(vec![2, 14, 17, 64], DType::F32),
        &DotDimensionNumbers {
            lhs_batching: vec![0, 1], rhs_batching: vec![0, 1],
            lhs_contracting: vec![3], rhs_contracting: vec![3],
        },
    );
    assert_eq!(out.dims, vec![2, 14, 17, 17]);
}

/// ⚠️ F3. The scale is 1/sqrt(head_dim) — NOT 1/sqrt(hidden_dim),
/// NOT 1/sqrt(n_heads·head_dim). All three are dimensionally plausible and
/// only one is right; the wrong ones just change softmax temperature, which
/// produces confident nonsense rather than a crash.
#[test]
fn attention_scale_uses_head_dim_not_hidden_dim() {
    const HEAD_DIM: usize = 64;
    const N_HEADS: usize = 14;
    const HIDDEN:  usize = HEAD_DIM * N_HEADS;   // 896

    let scores = ops::attention_scores_f64(&q_ones(), &k_ones(), HEAD_DIM);

    // With Q=K=1, raw QK^T = head_dim, so scaled score = head_dim/sqrt(head_dim)
    //                                                  = sqrt(head_dim) = 8.0
    let expected = (HEAD_DIM as f64).sqrt();
    let wrong_hidden = HEAD_DIM as f64 / (HIDDEN as f64).sqrt();   // ≈ 2.14

    assert!((scores.at(&[0, 0, 0, 0]) - expected).abs() < 1e-12,
        "expected sqrt(head_dim)={expected}; 1/sqrt(hidden) would give {wrong_hidden}");
}
```

## 8.2 Causal mask

```rust
/// Masked positions must be -inf BEFORE softmax, and exactly zero AFTER.
/// A large-negative sentinel (-1e9) instead of -inf leaks a small probability
/// into the future — invisible at short context, corrupting at long context.
#[test]
fn causal_mask_is_neg_inf_and_softmaxes_to_exact_zero() {
    let masked = ops::causal_mask_f64(&scores_ones(5, 5));
    for i in 0..5 { for j in (i + 1)..5 {
        assert_eq!(masked.at(&[0, 0, i, j]), f64::NEG_INFINITY,
            "position ({i},{j}) is in the future and must be -inf, not a sentinel");
    }}

    let probs = ops::softmax_f64(&masked, 3);
    for i in 0..5 {
        for j in (i + 1)..5 {
            assert_eq!(probs.at(&[0, 0, i, j]), 0.0, "future leaked probability");
        }
        let row: f64 = (0..=i).map(|j| probs.at(&[0, 0, i, j])).sum();
        assert!((row - 1.0).abs() < 1e-12, "row {i} does not sum to 1");
    }
}
```

⚠️ **Metamorphic property — the strongest attention test available without an oracle:**

```rust
/// Appending future tokens must NOT change the output at earlier positions.
/// This is causality itself, and it catches mask bugs, KV-offset bugs, and
/// RoPE position bugs at once — with no reference implementation needed.
#[test]
fn causality_prefix_invariance() {
    let short = ops::gqa_attention_f64(&tokens[..8]);
    let long  = ops::gqa_attention_f64(&tokens[..16]);
    for pos in 0..8 {
        assert_close_f64(short.row(pos), long.row(pos), 1e-12,
            "position {pos} changed when future tokens were appended");
    }
}
```

---

# 9. Test Area 5 — FFN Matmuls (gate, up, down)

**Target:** ARTX3 `ops/ffn.rs`; ARTX8 `matrix/fusion.rs`.

```rust
/// Shape chain: x[B,S,D] → gate[B,S,F], up[B,S,F] → down → [B,S,D].
#[test]
fn swiglu_shape_chain() {
    const D: usize = 896;
    const F: usize = 4864;                       // Qwen2.5-0.5B, deliberately not 4*D
    let out = ops::swiglu_ffn_shapes(&[2, 17, D], D, F);
    assert_eq!(out, vec![2, 17, D]);
}

/// ⚠️ F4. ARTX8 fuses gate+up into one [D, 2F] matmul, then splits by segment.
/// A wrong split offset gives two correctly-shaped [B,S,F] tensors with the
/// halves swapped or straddling the boundary. Nothing crashes.
#[test]
fn fused_gate_up_split_offsets_are_exact() {
    const D: usize = 8;
    const F: usize = 5;

    // Column j of w_gate holds value (j+1); column j of w_up holds -(j+1).
    // After fusion the sign identifies which half a column came from.
    let (fused, segments) = matrix::fuse_columns(&[&w_gate_signed(D, F), &w_up_signed(D, F)]);
    assert_eq!(fused.dims, vec![D, 2 * F]);
    assert_eq!(segments, vec![Segment { start: 0, len: F }, Segment { start: F, len: F }]);

    for j in 0..F {
        assert!(fused.at(&[0, j])         > 0.0, "gate segment column {j} sign flipped");
        assert!(fused.at(&[0, F + j])     < 0.0, "up segment column {j} sign flipped");
    }
}

/// Fusion must be numerically exact, not merely close — it is a data movement,
/// not a computation.
#[test]
fn fused_gate_up_equals_separate_matmuls_bitwise_in_f64() {
    let (fused_out, segs) = ops::swiglu_fused_f64(&x, &w_gate, &w_up);
    let gate_sep = reference::dot_general_f64(&x, &w_gate, &dnums_contract(&[2], &[0]));
    let up_sep   = reference::dot_general_f64(&x, &w_up,   &dnums_contract(&[2], &[0]));

    assert_bitwise_eq(&fused_out.slice_seg(segs[0]), &gate_sep);
    assert_bitwise_eq(&fused_out.slice_seg(segs[1]), &up_sep);
}

/// SwiGLU applies SiLU to GATE only, then multiplies by UP.
/// Swapping them is shape-identical and silently wrong.
#[test]
fn silu_applies_to_gate_not_up() {
    let gate = TensorF64::from(&[1.0, 2.0]);
    let up   = TensorF64::from(&[3.0, 4.0]);
    let got  = ops::swiglu_activation_f64(&gate, &up);

    let silu = |x: f64| x / (1.0 + (-x).exp());
    assert!((got.at(&[0]) - silu(1.0) * 3.0).abs() < 1e-12);
    assert!((got.at(&[1]) - silu(2.0) * 4.0).abs() < 1e-12);

    // Prove the swap is detectable by this fixture.
    assert!((silu(1.0) * 3.0 - silu(3.0) * 1.0).abs() > 1e-6,
        "fixture cannot distinguish gate/up swap");
}
```

---

# 10. Test Area 6 — `lm_head` / Vocab Projection

**Target:** ARTX3 `ops/embedding.rs` + the final projection.

⚠️ ARTX8 noted `lm_head` is disproportionate on small models — for Qwen2.5-0.5B the vocab
(151,936) does not shrink with the model, and ARTX1 §7 records that weight tying (sharing the
embedding table with the LM head) is expressed as the **same SSA value** used in both a gather and a
final `dot_general`. Both facts create failure mode **F5**.

```rust
#[test]
fn lm_head_projects_to_vocab() {
    const D: usize = 896;
    const V: usize = 151_936;
    let out = infer_dot_general_shape(
        &Shape::new(vec![2, 17, D], DType::F32),
        &Shape::new(vec![D, V], DType::F32),
        &dnums_contract(&[2], &[0]),
    );
    assert_eq!(out.dims, vec![2, 17, V]);
}

/// ⚠️ F5. A tied embedding table is stored [V, D] (gather over V), but the
/// lm_head matmul must contract over D. Contracting the wrong axis of a tied
/// table is the classic tied-weights bug.
#[test]
fn tied_embedding_lm_head_contracts_d_not_v() {
    const D: usize = 8;
    const V: usize = 5;
    let table = TensorF64::index_encoded(&[V, D]);         // [V, D] as stored

    // Correct: contract lhs dim 2 (D) with table dim 1 (D) → [.., V]
    let out = reference::dot_general_f64(
        &hidden_f64(&[1, 1, D]), &table,
        &DotDimensionNumbers {
            lhs_batching: vec![], rhs_batching: vec![],
            lhs_contracting: vec![2], rhs_contracting: vec![1],
        },
    );
    assert_eq!(out.dims, vec![1, 1, V], "lm_head must produce vocab-sized logits");

    // Contracting rhs dim 0 (V=5) against D=8 must be rejected outright.
    let bad = std::panic::catch_unwind(|| {
        reference::dot_general_f64(&hidden_f64(&[1, 1, D]), &table,
            &DotDimensionNumbers {
                lhs_batching: vec![], rhs_batching: vec![],
                lhs_contracting: vec![2], rhs_contracting: vec![0],
            })
    });
    assert!(bad.is_err(), "D != V must be caught; pick D != V in this fixture");
}

/// Weight tying must share ONE SSA value, not copy the table.
/// A copy is numerically identical and doubles memory — invisible to a
/// value-comparison test, visible only here.
#[test]
fn tied_weights_share_one_ssa_value() {
    let module = build_tied_embedding_model();
    assert_eq!(count_weight_params(&module, "model.embed_tokens.weight"), 1,
        "tied embedding was materialized twice");
}
```

---

# 11. Test Area 7 — Integration / Divergence Report

Per §2.4 this is **not** a pass/fail correctness gate. It is a layer-bisection report.

```rust
// gljax/tests/correctness/t3_divergence.rs

pub struct DivergenceReport {
    pub per_layer: Vec<LayerDivergence>,
    /// First layer whose divergence exceeds the expected quantization floor.
    pub first_anomaly: Option<usize>,
    pub logits: LogitsComparison,
}

pub struct LayerDivergence {
    pub layer: usize,
    pub residual_rel_l2: f64,
    /// Expected floor from quantization alone, NOT from a correctness defect.
    pub expected_floor: f64,
    pub anomalous: bool,
}
```

⚠️ **DESIGN DECISION — divergence is judged against a *quantization floor*, not against zero.**
glproc serves Q8_0-repacked weights; gljax serves BF16. A per-layer difference is *expected*. The
report flags a layer only when its divergence jumps materially above the smooth accumulation the
quantization difference predicts — a **step**, not a level.

```text
Expected (quantization only):        Anomalous (a real bug at layer 12):
  rel_l2                               rel_l2
    │              ╱                     │           ┌────────
    │         ╱                          │           │
    │    ╱                               │      ╱────┘
    └──────────── layer                  └──────────── layer
                                                    ↑ step = the bug
```

```rust
#[test]
#[ignore = "requires both engines + a real model; run manually"]
fn divergence_report_gljax_vs_glproc() {
    let report = run_divergence(&model_path(), &TOKENS);

    // NOT a gate — printed for a human, per §2.4.
    println!("{}", report.render_table());

    if let Some(layer) = report.first_anomaly {
        println!("⚠️  First anomalous divergence at layer {layer}. \
                  This localizes a disagreement; it does NOT say which engine is wrong. \
                  glproc has a known ~33pp unexplained precision gap vs llama.cpp.");
    }
}
```

## 11.1 The gated integration test uses llama.cpp, not glproc

```rust
/// Gated end-to-end check against an INDEPENDENT reference.
/// llama.cpp is already cloned as a sibling directory (commit 910196f) and is
/// the reference glproc's own gap was measured against.
#[test]
#[ignore = "requires llama.cpp + GGUF; run in the nightly lane"]
fn logits_agree_with_llamacpp() {
    let got  = gljax_logits(&model, &TOKENS);
    let want = llamacpp_logits(&gguf, &TOKENS);
    let cmp  = compare_logits(&got, &want);

    // Gates per §4.2 — top-1 and KL, not raw L2.
    assert!(cmp.top1_agreement >= 0.99, "top-1 agreement {:.4}", cmp.top1_agreement);
    assert!(cmp.kl_divergence  <= 1e-3, "KL {:.6}", cmp.kl_divergence);

    // Threshold computed from §3.3, not hardcoded.
    let tol = logits_tolerance(model.n_layers, model.hidden_dim);
    assert!(cmp.centered_rel_l2 <= tol,
        "centered rel-L2 {:.4} exceeds derived tolerance {tol:.4}", cmp.centered_rel_l2);
}
```

⚠️ Comparing gljax-BF16 against llama.cpp-quantized still mixes numerical pipelines. The cleanest
form of this test runs **gljax in FP64 oracle mode against llama.cpp in F16/F32 GGUF**, removing
quantization from both sides. That is the configuration to build first.

---

# 12. Harness Structure

```text
gljax/
├── src/
│   └── oracle/                    ← shipped, not test-only: T2 needs it at runtime
│       ├── mod.rs
│       ├── reference.rs           T0 naive f64 dot_general. NEVER optimize.
│       ├── tolerance.rs           §3 — derived tolerances, no magic numbers
│       ├── metrics.rs             §4 — top1, KL, centered rel-L2
│       └── fixtures.rs            adversarial shapes + input patterns (§4.3)
│
└── tests/correctness/
    ├── mod.rs                     shared helpers, tier gating, skip messages
    ├── t0_classification.rs       §5  — GEMM/GEMV, property tests
    ├── t0_dnums.rs                §6  — dimension numbers, dtype, transposition
    ├── t0_gqa_expand.rs           §7  — ⚠️ highest-value file
    ├── t0_attention.rs            §8  — QK^T, scale, causal mask, prefix invariance
    ├── t0_ffn.rs                  §9  — gate/up/down, fusion offsets, SiLU placement
    ├── t0_lm_head.rs              §10 — vocab projection, tied weights
    ├── t1_interpreter.rs          §2.2 — StableHLO conformance (skips if absent)
    ├── t2_fp64_plugin.rs          §2.3 — PJRT CPU oracle (cfg-gated)
    └── t3_divergence.rs           §11 — report + llama.cpp gate (#[ignore])
```

⚠️ **DESIGN DECISION — `oracle/` lives in `src/`, not `tests/`.**
T2 runs the FP64 oracle through a real `Session`, and ARTX16's `/debug/oracle` endpoint needs the same
code. Test-only placement would force duplication. It is feature-gated (`feature = "oracle"`,
default off) so production builds do not carry it.

## 12.1 CI lanes

| Lane | Tiers | Runtime | Trigger |
|---|---|---|---|
| `fast` | T0 | seconds | every commit |
| `spec` | T0 + T1 | ~1 min | every PR (skips without `stablehlo-translate`) |
| `oracle` | T0 + T1 + T2 | minutes | pre-merge; CPU/CUDA only, never TPU |
| `nightly` | all + T3 | tens of minutes | scheduled; needs a real model |

⚠️ **T0 must never require a model, a device, or a network.** That property is what keeps the `fast`
lane on every commit, and it is the lane that catches F1–F5 — the entire structural bug class.

---

# 13. Wave Plan + What Comes After

| Wave | Scope | Blocked on |
|---|---|---|
| **A10.1** | `oracle/{reference,tolerance,metrics,fixtures}.rs` + `t0_classification.rs` + `t0_dnums.rs` | ARTX8 `matrix/spec.rs` |
| **A10.2** | `t0_{gqa_expand,attention,ffn,lm_head}.rs` | ARTX3 ops layer |
| **A10.3** | `t1_interpreter.rs` | ARTX2 `FuncBuilder` emitting parseable MLIR |
| **A10.4** | `t2_fp64_plugin.rs` | ARTX1 FFI + ARTX4 `Session` |
| **A10.5** | `t3_divergence.rs` + llama.cpp gate | ARTX5 `Session::generate()` |

⚠️ A10.1 and A10.2 are **T0 — no PJRT, no device, no model**. They are the cheapest high-value work
in the entire ARTX series and can be written alongside ARTX2/ARTX3 rather than after them.

## What ARTX11 should cover

**Recommendation: ARTX11 — Speculative Decoding under Static Shapes**, the item ARTX16 §10 ranked
first before correctness was pulled forward. The argument is unchanged: ARTX8 measured decode at
200–600× below the roofline ridge point, so the matrix unit is idle through most of decode, and
speculative decoding is the technique that converts that idle compute into throughput. EAGLE-3 is the
production standard (merged in vLLM, SGLang, TensorRT-LLM; acceptance 0.75–0.85; 2–6×), and the
static-shape interaction — bucketing a dynamic draft tree — is a genuinely open design problem for
gljax rather than a port.

⚠️ ARTX12 is a **precondition** for it, not merely a predecessor. Speculative decoding changes the
KV write pattern (variable accepted length per step) and adds a second model. Landing that on an
engine with no matmul correctness harness would mean debugging two interacting sources of wrongness
at once.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | Four oracle tiers (T0–T3), each answering a different question | A single oracle cannot separate "wrong algorithm" from "different precision" from "different model" | Hard |
| D2 | T0 reference is naive f64 and **must never be optimized** | An optimized reference shares failure modes with its subject | Trivial |
| D3 | **Add T1: StableHLO reference interpreter** (not in the brief) | 1:1 spec correspondence, 91/96 ops, ~3k gold testdata; validates emission independent of device | Trivial |
| D4 | T1 invoked as a subprocess, skipped when absent | Linking MLIR/LLVM contradicts ARTX1's dependency posture | Trivial |
| D5 | T2 is CPU/CUDA-only, `cfg`-gated | TPU v5e has no usable FP64 (ARTX1 §3.1) | N/A |
| D6 | **glproc demoted from oracle to divergence detector** | ~33pp of its ~46% PPL gap vs llama.cpp is unexplained and suspected architectural; and it runs a different (quantized) numerical pipeline | Medium |
| D7 | llama.cpp is the external absolute reference | Independent of GwenLand's own implementations; already cloned as a sibling dir | Medium |
| D8 | Every tolerance is **derived**, none hardcoded | A magic constant is a future false pass | Trivial |
| D9 | Per-matmul tolerance `1e-2` and **does not scale with K** | bf16 input rounding (2·2⁻⁸) dominates fp32 accumulation (K·2⁻²⁴) by 16–150× | Trivial |
| D10 | A test that fails only as K grows indicates a non-FP32 accumulator | Direct corollary of D9 — makes F6 detectable | N/A |
| D11 | The brief's `rel L2 < 0.05` is **confirmed** for ≤32 layers, and computed thereafter | √L·δ gives 3.9–4.5e-2 at 24–32 layers; 7.2e-2 at 80 | Trivial |
| D12 | **Gate on top-1 agreement + KL**, not raw relative L2 | Softmax is shift-invariant, so raw-logit L2 can fail on a difference with zero downstream effect | Trivial |
| D13 | Reuse `glbench/src/kl_divergence.rs` | Already implemented in this repo | Trivial |
| D14 | Adversarial shapes: distinct, prime, non-power-of-two | Square matrices hide transposed dimension numbers (F1) | Trivial |
| D15 | Index-encoded inputs preferred over random for structural bugs | A wrong axis produces a visibly wrong magnitude, debuggable from the message alone | Trivial |
| D16 | Every discriminating fixture ships a **"can this fixture discriminate?"** meta-test | MHA/MQA degenerate cases silently disarm the GQA test | Trivial |
| D17 | Most tests run in `PrecisionPolicy::f64_oracle()` at `1e-12` | Removes precision from the equation, so a failure is an *algorithm* failure | Trivial |
| D18 | `oracle/` in `src/` behind a default-off feature, not in `tests/` | T2 and ARTX16's `/debug/oracle` both need it at runtime | Trivial |
| D19 | T0 lane requires no model, device, or network | Keeps the whole structural bug class (F1–F5) on every-commit CI | Hard |
| D20 | ARTX12 gates ARTX11 (speculative decoding) | Spec-decode changes the KV write pattern and adds a model; debugging that without a matmul harness means two unknowns at once | Medium |

---

# Appendix B — Failure Mode → Test Map

| Mode | Description | Caught by | Tier |
|---|---|---|---|
| F1 | Transposed / swapped dimension numbers | §6.3 `transposed_dnums_produce_different_values` | T0 |
| F2 | GQA block grouping instead of interleaved | §7.2 `gqa_expand_uses_interleaved_grouping` | T0 |
| F3 | Attention scale from hidden_dim not head_dim | §8.1 `attention_scale_uses_head_dim_not_hidden_dim` | T0 |
| F4 | Fused gate/up split at wrong offset | §9 `fused_gate_up_split_offsets_are_exact` | T0 |
| F5 | `lm_head` contracts wrong axis on tied weights | §10 `tied_embedding_lm_head_contracts_d_not_v` | T0 |
| F6 | BF16 accumulation instead of FP32 | §6.2 + D9/D10 (tolerance grows with K) | T0/T2 |
| F7 | Batch dims mis-assigned in batched attention | §5 `batched_when_batching_dims_present`, §8.1 | T0 |
| — | Causality / mask / RoPE position errors | §8.2 `causality_prefix_invariance` | T0 |
| — | Compile/execute path defects (donation, layout) | T2 passing where T1 fails | T2 |
| — | Cross-engine architectural divergence | §11 layer-bisection report | T3 |

---

# Sources

- [StableHLO Interpreter | OpenXLA](https://openxla.org/stablehlo/interpreter_status) and [Interpreter Design](https://openxla.org/stablehlo/reference) — `stablehlo-translate --interpret`, 1:1 spec correspondence, 91/96 ops, `Check` dialect, ~3k gold testdata files.
- [StableHLO Specification](https://github.com/openxla/stablehlo/blob/main/docs/spec.md) — `dot_general` output dim ordering (batch, lhs free, rhs free), `precision_config`, `preferred_element_type`.
- [Testing Linear Algebra Software — N. Higham](https://nhigham.com/wp-content/uploads/2023/10/high97t.pdf) — methodology for numerical test design.
- [Guaranteed DGEMM Accuracy While Using Reduced Precision Tensor Cores](https://arxiv.org/pdf/2511.13778) — the `|fl(AB)ᵢⱼ − (AB)ᵢⱼ| ≤ f(n)·ε·(|A||B|)ᵢⱼ` bound as the LAPACK/Higham gold standard; `f(n)` must not exceed linear growth.
- [Machine epsilon](https://en.wikipedia.org/wiki/Machine_epsilon) and [Numerical Precision in ONNX and AI Inference](https://www.emmtrix.com/wiki/Numerical_Precision_in_ONNX_and_AI_Inference) — BF16 ε = 2⁻⁷ ≈ 7.81e-3, unit roundoff u = ε/2 = 2⁻⁸.
- [Understanding GQA](https://simondong1.github.io/gqa.html) and [Grouped Query Attention (GQA) — oneDNN](https://uxlfoundation.github.io/oneDNN/dev_guide_graph_gqa.html) — `repeat_interleave` contiguous grouping is the HuggingFace standard; swapping the pattern at inference **silently corrupts attention**.
- [Rethinking KL Divergence in Knowledge Distillation for LLMs](https://arxiv.org/html/2404.02657v2) — KL over softmax as the distribution-level comparison metric for logits.
- [Numerical stability analysis of large language models](https://arxiv.org/pdf/2503.10251) — error accumulation through transformer layers.

**Repo-internal:** `memory/project_glproc_precision_gap_vs_llamacpp.md` (the ~46% gap, ~33pp
unexplained, llama.cpp at `C:\Users\reyha\Documents\JinXSuper-Projects\llama.cpp` commit `910196f`);
`memory/project_gllm_e2e_garbage_output.md` (the Q6_K dequant-order precedent);
`glproc/tests/kernel_parity.rs` (existing `assert_close` tolerance conventions, 1e-6..1e-4);
`glbench/src/kl_divergence.rs`, `glbench/src/validation/{numerical,parity}.rs` (reusable comparison
infrastructure).
