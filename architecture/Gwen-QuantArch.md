# Pridwen: A Co-Designed Mixed-Precision Quantization Framework for Shard-Based LLM Inference
**Codename:** Pridwen — The Impenetrable Shield  
**Status:** Pre-spec proposal v4  
**Author:** JinXSuper  
**Date:** 2026-07-21  
**Crate target:** `glictus-caliburni` (DType extension) + `glproc` (kernel path)

---

## 1. Positioning

G-Quant is a **Systems-Level Mixed-Precision Quantization Framework** — not a novel quantization algorithm. It functions as an optimizing compilation middle-layer that:

- Defines the **Quantization Architecture** (binary storage format, block structure, dequantization behavior)
- Defines the **Assignment Policy** (optimization objective for tensor precision selection)
- Enforces strict programmatic separation between the two

This separation enables future research on assignment strategies without requiring changes to the underlying binary storage format, while also allowing new architecture generations to reuse existing policies.

G-Quant is co-designed with the GLLM shard format (`.gllm`) — enabling per-tensor heterogeneous precision as a first-class citizen. This is the primary differentiator from GGUF, which has no native Assignment Engine: GGUF users manually select one quant type applied globally, with no automatic per-tensor sensitivity-aware assignment.

---

## 0. Codename

**Pridwen** — perisai Raja Arthur dalam legenda Welsh/Breton. Dalam mitologi Arthurian, Pridwen adalah perisai yang tidak bisa ditembus, melindungi Arthur dalam setiap pertempuran.

Dalam konteks GwenLand:

```
Ictus Caliburni  — membawa weights         (the sword)
Pridwen          — melindungi kualitas      (the shield)  
GATE             — mengoptimasi eksekusi    (the strategy)
```

Pridwen melindungi kualitas model dari kehancuran akibat quantization — walau weights di-compress ke 2-bit, kualitas tidak boleh collapse.

**Prior art acknowledged:** dMX, PrismQuant, mlx-optiq apply similar decoupling concepts. G-Quant's differentiation is co-design with a shard-based format and pure Rust zero-dep implementation.

---

## 2. Naming Convention

```
GQ<bit><Architecture>_<Policy>
```

| Component | Description |
|---|---|
| **GQ** | GwenLand Quantization format identifier |
| **bit** | Target quantization bit-width |
| **Architecture** | Block structure, metadata layout, scale representation, weight packing, dequantization algorithm, kernel requirements |
| **Policy** | Tensor assignment strategy used during model conversion |

### Format Examples

```
GQ2A_PFP   — 2-bit, Architecture A, Precision-First Policy
GQ2A_SFP   — 2-bit, Architecture A, Speed-First Policy
GQ2A_CPP   — 2-bit, Architecture A, Combined Policy (recommended)

GQ4A_PFP   — 4-bit, Architecture A, Precision-First Policy
GQ4A_CPP   — 4-bit, Architecture A, Combined Policy

GQ1A_PFP   — 1-bit, Architecture A, Precision-First Policy (requires QAT)
```

---

## 3. Architecture A — Block Structure

Architecture defines the binary storage specification. It does not determine how tensors are assigned during conversion.

### 3.1 GQ4A — Foundation (compete Q4_K_M)

```
SuperBlock (256 weights):
┌─────────────────────────────────────────┐
│ super_scale: f16          (2 bytes)     │
│ block[0..7]:                            │
│   scale_delta: i8         (1 byte each) │  ← 8 bytes total
│   weights[32]: 4-bit      (16 bytes)    │  ← 128 bytes total
├─────────────────────────────────────────┤
│ Total: 2 + 8 + 128 = 138 bytes          │
│ Effective bpw: 4.3125                   │
└─────────────────────────────────────────┘
```

Scale reconstruction:
```
scale_i     = super_scale × (scale_delta_i / 127.0)
weight_f32  = scale_i × dequant(weight_int4)
```

### 3.2 GQ2A — Primary Innovation

```
SuperBlock (256 weights):
┌─────────────────────────────────────────┐
│ super_scale: f16          (2 bytes)     │
│ super_min:   f16          (2 bytes)     │  ← asymmetric support
│ block[0..15]:                           │
│   scale_delta: i4 packed  (8 bytes)     │  ← finer granularity
│   min_delta:   i4 packed  (8 bytes)     │
│   weights[16]: 2-bit      (4 bytes)     │  ← 64 bytes total
├─────────────────────────────────────────┤
│ Total: 2 + 2 + 8 + 8 + 64 = 84 bytes   │
│ Effective bpw: 2.625                    │
└─────────────────────────────────────────┘
```

`super_min` handles asymmetric weight distributions — critical for quality preservation at 2-bit. Absent in GGML Q2_K per-superblock.

### 3.3 GQ1A — Ternary (requires QAT)

```
SuperBlock (256 weights):
┌─────────────────────────────────────────┐
│ group_scale: f16          (2 bytes)     │
│ weights[256]: ternary                   │
│   packed 4 per byte (2-bit slot)        │  ← 64 bytes
│   encoding: 0b00 = -1                   │
│             0b01 =  0                   │
│             0b10 = +1                   │
├─────────────────────────────────────────┤
│ Total: 2 + 64 = 66 bytes                │
│ Deployed bpw: 2.0625                    │
│ Information-theoretic bpw: 1.58         │
└─────────────────────────────────────────┘
```

⚠️ GQ1A requires QAT (Quantization-Aware Training). PTQ to ternary on models <3B = catastrophic collapse.

### 3.4 Memory Alignment Requirements

All superblock structures must align to **32-byte boundaries** — matching x86/ARM cache line sizes and GPU warp memory coalescing requirements. Quantized weights are stored in contiguous memory regions, separate from metadata blocks, to prevent scattered loads and bandwidth waste.

---

## 4. Assignment Policies

Policies define the optimization objective used by the Assignment Engine during model conversion. Policies influence tensor precision selection, escape-hatch decisions, and heterogeneous precision assignment without modifying the underlying storage architecture.

### PFP — Precision-First Policy

**Optimization objective:** Minimize quantization error, prioritize capability preservation.

**Phase 1 implementation (naive):** Hardcoded sensitivity table — sensitive layers assigned to highest available format, rest at base bit-width.

**Expected characteristics:**
- Lower quantization error
- Better capability retention
- Higher storage overhead
- Potentially lower inference throughput

### SFP — Speed-First Policy

**Optimization objective:** Maximize execution throughput and memory bandwidth efficiency.

**Phase 1 implementation (naive):** Hardcoded sensitivity table — only EXTREME sensitivity layers get escape hatch, everything else at minimum bit-width.

**Why latency_cost is NOT in Phase 1:**  
SFP requires per-format latency profiling on target hardware (dequant cycles per block, cache pressure per format). This data does not exist yet. Incorporating unvalidated latency estimates into the assignment objective would produce worse results than the naive heuristic. Latency-aware assignment is deferred to Phase 3 after empirical profiling via glbench.

**Expected characteristics:**
- Higher throughput (intent — not guaranteed until measured)
- Lower memory bandwidth usage
- Increased quantization error may occur
- Capability degradation depends on model characteristics

### CPP — Combined Precision Policy

**Optimization objective:** Balance precision preservation and execution performance.

**Phase 1 implementation (naive):** Sensitivity table from quantization degradation research — layer type → format assignment. This is the reference implementation that will be refined as empirical data arrives.

**Why MCKP is NOT in Phase 1:**  
The MCKP formulation requires two inputs that do not exist yet:
1. `sensitivity(layer_i, format_i)` — quantization error per layer per format, requires calibration run per model
2. `latency_cost(layer_i, format_i)` — hardware-specific dequant overhead, requires profiling on i3-1115G4

Without real data, an MCKP solver optimizes invented numbers and can produce assignments worse than the naive heuristic. MCKP is the correct long-term formulation and will be implemented in Phase 3 once empirical data is available from Phase 1 and Phase 2 runs.

**Expected characteristics:**
- Balanced capability retention
- Balanced throughput
- Flexible heterogeneous precision assignment
- Recommended general-purpose conversion policy

---

## 5. Default Layer Sensitivity Assignment (CPP Phase 1 Reference)

Based on quantization degradation research. This table is the Phase 1 naive implementation — it will be revised as empirical PPL measurements arrive from real model runs.

```
Layer Type              Sensitivity   GQ2A_CPP assign   GQ4A_CPP assign
─────────────────────   ───────────   ───────────────   ───────────────
token_embd              EXTREME       GQ4A (escape)     GQ4A
output (LM head)        EXTREME       GQ4A (escape)     GQ4A
output_norm             EXTREME       F32  (always)     F32
attn_norm / ffn_norm    HIGH          F16  (always)     F16
attn_q / attn_k         HIGH          GQ4A (escape)     GQ4A
attn_v / attn_output    MEDIUM-HIGH   GQ2A              GQ4A
ffn_gate / ffn_up       MEDIUM        GQ2A              GQ4A
ffn_down                MEDIUM-LOW    GQ2A              GQ4A
```

**Effective bpw GQ2A_CPP** (Qwen2.5-0.5B reference):
- ~15% weights at GQ4A escape hatch = 4.3125 bpw
- ~85% weights at GQ2A = 2.625 bpw
- **Weighted average: ~2.87 bpw** — above 2.8 cliff threshold ✅

---

## 6. Incoherence Processing (QuIP#-inspired — Phase 2 research question)

Incoherence processing is a candidate technique for GQ2A quality improvement, not a committed feature.

**The idea (from QuIP#):**
```
Offline (glconv):
  W_rotated = H × W × H^T     (H = random Hadamard matrix, seeded)
  quantize W_rotated → GQ2A
  store rotation_seed: u32     (4 bytes per tensor)

Online (glproc dequant):
  W_approx = H^T × dequant(W_rotated_GQ2A) × H
```

Hadamard rotation spreads outlier weights uniformly across the matrix. Post-rotation distribution approaches Gaussian — significantly more friendly for uniform quantization.

**Why this is NOT committed to Phase 1:**

1. **QuIP# is CUDA + PyTorch** — their implementation has no zero-dep Rust equivalent. Fast Hadamard Transform (FHT) must be implemented from scratch. FHT is O(n log n) in theory, but actual overhead on i3-1115G4 is unknown. If dequant-time inverse transform is too slow, it eliminates the speed benefit of GQ2A entirely.

2. **Quality benefit unvalidated for GQ2A specifically** — QuIP# proves benefit for their E8 Lattice VQ format. Whether FHT preprocessing materially improves GQ2A's simpler uniform quantization on Qwen2.5-0.5B weights is an open empirical question.

**Phase 2 plan:** Implement GQ2A first without incoherence processing. Measure PPL baseline. Then add FHT and measure delta. If improvement is significant and overhead is acceptable → commit. If not → reject and document (same pattern as rejected-optimizations.md in glproc).

---

## 7. Assignment Engine — Phase Roadmap

```
Phase 1 (now):
  Assignment Engine = hardcoded sensitivity table
  Input:  tensor name → lookup table → format
  No solver, no profiling, no calibration data needed

Phase 2 (after GQ2A baseline):
  Sensitivity scoring = per-layer PPL delta measurement
  Input:  calibration run on real model → sensitivity scores
  Still greedy assignment, but scores are empirical not hardcoded

Phase 3 (after Phase 2 data):
  MCKP formulation becomes viable:
    - sensitivity(layer_i, format_i) from Phase 2 calibration
    - latency_cost(layer_i, format_i) from glbench profiling
  PFP / SFP / CPP differentiation becomes meaningful
```

---

## 8. Integration with GLLM Format

G-Quant extends the `DType` enum in `glictus-caliburni`:

```rust
pub enum DType {
    // Existing (from GGUF compatibility)
    F32, F16, Q8_0, Q4_K, Q5_0, Q6_K,

    // G-Quant native
    GQ4A = 0x0200,
    GQ2A = 0x0201,
    GQ1A = 0x0202,
    // Architecture B, C reserved: 0x0300, 0x0400
}
```

Per-tensor dtype in `gllm.json`:
```json
{
  "layers": [
    { "name": "token_embd",      "dtype": "GQ4A" },
    { "name": "blk.0.attn_q",   "dtype": "GQ4A" },
    { "name": "blk.0.ffn_gate", "dtype": "GQ2A" },
    { "name": "blk.0.ffn_down", "dtype": "GQ2A" }
  ]
}
```

This heterogeneous per-tensor assignment is impossible in GGUF without hacks — it is a first-class citizen in GLLM.

---

## 9. Conversion Pipeline (glconv extension)

```
glconv model.gguf output/ --quant GQ2A --policy CPP
```

Phase 1 pipeline:
```
1. Parse GGUF                              (existing glconv)
2. Dequant GGUF weights → F32             (glcore dequant path)
3. Run sensitivity table lookup            (Phase 1: hardcoded CPP table)
4. Quantize each tensor → assigned format
5. Write to GLLM package, dtype per-tensor
6. Self-validate                           (existing GllmPackage::open pattern)
```

Steps 3 will be replaced by empirical scoring in Phase 2, and MCKP in Phase 3.

---

## 10. Inference Kernel Requirements (glproc extension)

```rust
// glproc/src/kernels/gquant.rs (new)

// GQ4A scalar reference
pub fn dequant_gq4a(block: &GQ4ABlock, out: &mut [f32; 256]);

// GQ4A AVX2 fast path
#[cfg(target_feature = "avx2")]
pub fn dequant_gq4a_avx2(block: &GQ4ABlock, out: &mut [f32; 256]);

// GQ2A scalar reference  
pub fn dequant_gq2a(block: &GQ2ABlock, out: &mut [f32; 256]);

// GQ2A AVX2 fast path
#[cfg(target_feature = "avx2")]
pub fn dequant_gq2a_avx2(block: &GQ2ABlock, out: &mut [f32; 256]);
```

All kernels verified against scalar reference per `testing-standards.md`.

---

## 11. Design Disclaimer

Policy names indicate **optimization intent** rather than guaranteed runtime behavior.

They must **not** be interpreted as guarantees of:
- Perplexity
- Throughput or latency
- Memory consumption
- Downstream benchmark performance
- Reasoning capability

Actual behavior depends on: model architecture, calibration dataset, tensor distribution, hardware platform, inference backend, and implementation details.

**Empirical evaluation is required for every model and deployment scenario.**

---

## 12. Known Unknowns (explicit)

These are open questions that will be answered through empirical work — not assumed:

| Unknown | Answered in | Method |
|---|---|---|
| GQ4A PPL vs Q4_K_M delta | Phase 1 | glbench perplexity run |
| GQ2A baseline PPL (no FHT) | Phase 2 | glbench perplexity run |
| FHT overhead on i3-1115G4 | Phase 2 | glbench latency profiling |
| FHT quality delta for GQ2A | Phase 2 | PPL before vs after FHT |
| Per-layer sensitivity scores | Phase 2 | calibration run on Qwen2.5-0.5B |
| latency_cost per format | Phase 3 | glbench dequant kernel profiling |
| MCKP vs greedy quality delta | Phase 3 | A/B comparison on real model |
| GQ2A_CPP vs Q4_K_M real tok/s | Phase 2 | glbench decode benchmark |

---

## 13. Expected Results

Phase 1 target (GQ4A_CPP vs Q4_K_M):
- PPL: within 1% of Q4_K_M baseline
- Size: ~450MB vs 463MB (minor reduction)
- Decode: comparable — GQ4A dequant overhead vs Q4_K_M to be measured

Phase 2+ results: **TBD — requires empirical measurement.** No tok/s projections until Phase 1 latency data exists.

---

## 14. Implementation Phases

### Phase 1 — GQ4A (Foundation)
- GQ4A block structure + scalar dequant + AVX2 fast path
- glconv: `--quant GQ4A --policy CPP` with hardcoded sensitivity table
- Validate: PPL parity vs Q4_K_M on Qwen2.5-0.5B via glbench
- Output: Proven baseline, real latency data for Phase 2 planning

### Phase 2 — GQ2A baseline (Primary)
- GQ2A block structure + scalar dequant + AVX2 fast path
- glconv: `--quant GQ2A --policy CPP`
- Validate: PPL measurement without FHT first
- Validate: FHT implementation + measure overhead + measure PPL delta
- Decision gate: commit or reject FHT based on empirical data
- Output: First real GQ2A numbers — paper-worthy if quality holds

### Phase 3 — Assignment Engine (Research)
- Empirical sensitivity scoring via calibration run
- Hardware latency profiling → `latency_cost` table
- MCKP solver implementation + A/B vs greedy
- PFP / SFP / CPP properly differentiated with real data
- Output: Full framework demonstrable, policies meaningful

### Phase 4 — GQ1A (Research track)
- Ternary format + QAT integration with gltrain
- Prerequisite: gltrain un-park, compute resources
- Output: Potential standalone paper

---

## 15. Open Questions

1. **Calibration data source** — Phase 2 sensitivity scoring needs ~128 samples. C4 subset or custom GwenLand calibration set?
2. **Hadamard seed storage** — if FHT committed in Phase 2: per-tensor in `gllm.json` or embedded in binary layer file?
3. **Architecture B roadmap** — candidate: FP4/MXFP4 support for GPU path (glcuda). Timing: after Phase 2 complete.
4. **GQ4A exact compatibility with Q4_K_M** — do we want byte-level compatible superblock, or diverge for better GLLM alignment?

---

## 16. Paper Angle

> **"Pridwen: A Co-Designed Mixed-Precision Quantization Framework for Shard-Based LLM Inference"**

**Claimed contributions:**
1. Formal separation of Quantization Architecture from Assignment Policy as a programmatic API
2. First quantization framework co-designed with a shard-based weight format (GLLM)
3. Per-tensor heterogeneous precision as a first-class citizen enabling lazy expert loading (MoE on 8GB RAM)
4. GQ2A: empirically validated 2-bit PTQ for small models via asymmetric superblock (FHT pending validation)
5. Open-source pure Rust zero-dep implementation in the GwenLand ecosystem

**Target venue:** MLSys, EMNLP Systems Track, or arXiv standalone.

**Explicitly NOT claimed:**
- Novel quantization algorithm
- Guaranteed perplexity or throughput improvements
- First framework to decouple policy from format (prior art: dMX, PrismQuant, mlx-optiq acknowledged)
- MCKP solver (Phase 3, not yet implemented)
- Incoherence processing (Phase 2 research question, not yet validated)