# Pridwen: A Co-Designed Mixed-Precision Quantization Framework for Shard-Based LLM Inference
**Codename:** Pridwen — The Impenetrable Shield  
**Status:** Pre-spec proposal v5  
**Author:** JinXSuper  
**Date:** 2026-07-22  
**Crate target:** `glictus-caliburni` (DType extension) + `glproc` (kernel path)

**Revision note (v4 → v5):** This revision resolves all 6 blocking and 7 non-blocking
issues found in the v4 analysis report (2026-07-22). The most structurally significant
change is splitting what v4 called "Phase N" into two independently-numbered tracks —
**Phase N** (Implementation Phases, §14 — what gets built) and **Stage N** (Assignment
Engine sophistication, §7 — how tensor assignment gets smarter) — which v4 conflated
under a single "Phase" label used inconsistently across §4/§6/§7/§9/§12. Every
cross-reference below has been re-verified against this split. See §17 for the full
changelog.

**Addendum (2026-07-22, before GQ2A implementation):** §3.2 gained an explicit
bit-packing subsection (scale_delta/min_delta/weights byte layout) that the original
v5 text omitted — it stated field *sizes* but not internal bit order, which Phase 1's
GQ4A AVX2 kernel showed cannot be safely assumed by an implementer (a mismatched
guess there produced a real, silently-wrong-answer bug before being caught by
parity testing). Written before any GQ2A code exists, so there is no implementation
to reconcile against — this is filling a spec gap, not fixing a discovered bug.

---

## 1. Positioning

G-Quant is a **Systems-Level Mixed-Precision Quantization Framework** — not a novel quantization algorithm. It functions as an optimizing compilation middle-layer that:

- Defines the **Quantization Architecture** (binary storage format, block structure, dequantization behavior)
- Defines the **Assignment Policy** (optimization objective for tensor precision selection)
- Enforces strict programmatic separation between the two, **for Policy changes that do not require new storage layouts**

This separation enables future research on assignment strategies without requiring changes to the underlying binary storage format, while also allowing new architecture generations to reuse existing policies. This is a *design goal*, not an absolute law: a small number of quality-improving techniques (e.g. incoherence processing, §6) genuinely require a new field in the block structure to store per-tensor side data. Where that happens, the framework treats it as a new **Architecture sub-variant** (e.g. GQ2A-R, §3.3) rather than pretending it fits inside the Policy layer — see §6 for why this boundary case does not undermine the separation principle for the common case (PFP/SFP/CPP all target the same Architecture A block layouts and never require a format change).

G-Quant is co-designed with the GLLM shard format (`.gllm`) — enabling per-tensor heterogeneous precision as a first-class citizen. This is the primary differentiator from GGUF, which has no native Assignment Engine: GGUF users manually select one quant type applied globally, with no automatic per-tensor sensitivity-aware assignment. **Note:** the Phase 1 reference policy (GQ4A_CPP) does not itself exercise heterogeneous assignment — see §5's note — so this differentiator is demonstrated starting at GQ2A_CPP (Phase 2), not from the first shipped policy.

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

Architecture is normally a single letter (`A`), but a letter may carry a suffix to mark a
**sub-variant**: same superblock family, same bit-width, one additive field for an
optional feature. `GQ2A-R` (§3.3) is the only sub-variant defined so far — it is GQ2A
plus a `rotation_seed` field for incoherence processing (§6). A sub-variant is always
opt-in and decodable independently: a decoder that only understands GQ2A never needs to
understand GQ2A-R, and vice versa — they are different `DType` codes (§8), not a runtime
flag on the same code.

### Format Examples

```
GQ2A_PFP     — 2-bit, Architecture A, Precision-First Policy
GQ2A_SFP     — 2-bit, Architecture A, Speed-First Policy
GQ2A_CPP     — 2-bit, Architecture A, Combined Policy (recommended)
GQ2A-R_CPP   — 2-bit, Architecture A sub-variant R (rotated/incoherence-processed), Combined Policy

GQ4A_PFP     — 4-bit, Architecture A, Precision-First Policy
GQ4A_CPP     — 4-bit, Architecture A, Combined Policy

GQ1A_PFP     — 1-bit (deployed: 2.0625 bpw, see §3.4), Architecture A, Precision-First Policy (requires QAT)
```

---

## 3. Architecture A — Block Structure

Architecture defines the binary storage specification. It does not determine how tensors are assigned during conversion.

### 3.1 GQ4A — Foundation

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

**This is a GwenLand-native layout, not byte-compatible with llama.cpp's Q4_K_M**
(which packs 256 weights into 144 bytes / 4.5 bpw using 6-bit per-block scale and min
pairs, not a 1-byte signed delta). GQ4A trades some of Q4_K_M's per-block scale
resolution for a simpler, smaller superblock; §12 tracks the resulting PPL delta as an
open empirical question. See §15 for why byte-compatibility was not pursued.

**Weight code range:** `weight_int4` is stored as an **unsigned 4-bit code in `[0, 15]`**,
remapped to a signed, zero-centered range at dequant time via `dequant(c) = c − 8`,
giving `dequant(weight_int4) ∈ [-8, 7]` (16 evenly spaced integer levels, asymmetric
around zero by one step — the same convention GGML's Q4_0 uses).

Scale reconstruction:
```
scale_i     = super_scale × (scale_delta_i / 127.0)
dequant(c)  = c − 8                                    // c ∈ [0,15] → [-8,7]
weight_f32  = scale_i × dequant(weight_int4)
```
`scale_delta_i` is an `i8` in `[-128, 127]`; dividing by `127.0` (the max positive `i8`
magnitude) normalizes it to approximately `[-1.008, 1.0]`, giving `scale_i` a small
signed adjustment around `super_scale` per block.

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

**Weight code range:** `weight_int2` is an unsigned 2-bit code in `[0, 3]` (4 levels — the
maximum a 2-bit field can address without wasting encoding space, unlike GQ1A's ternary
scheme in §3.4 which deliberately uses only 3 of 4 states).

**Reconstruction (asymmetric, min + scale per block):**
```
scale_i    = super_scale × (1.0 + scale_delta_i / 7.0)   // scale_delta_i ∈ [-8,7] (i4)
min_i      = super_min   × (1.0 + min_delta_i   / 7.0)   // min_delta_i   ∈ [-8,7] (i4)
weight_f32 = min_i + scale_i × (weight_int2 / 3.0)
```
`scale_delta_i` and `min_delta_i` are each a **signed 4-bit** value in `[-8, 7]`,
normalized by `7.0` (the max positive `i4` magnitude — mirroring GQ4A's `i8`/`127.0`
convention at 4-bit width) to a `[-1.14, 1.0]` multiplicative adjustment around
`super_scale`/`super_min` respectively. `weight_int2 / 3.0` maps the 2-bit code linearly
onto `[0.0, 1.0]` (4 evenly spaced fractions: 0, 1/3, 2/3, 1), so the fully-reconstructed
weight ranges over `[min_i, min_i + scale_i]` per block — this is what makes GQ2A
asymmetric: unlike GQ4A's zero-centered `dequant(c) = c − 8`, GQ2A's 2-bit codes do not
need to represent negative values directly, because `min_i` (which can itself be
negative, since `super_min` and `min_delta_i` are signed/unconstrained) supplies the
lower bound of the block's actual weight range.

**Bit-packing (added in this revision — the byte diagram above states field sizes but
not internal bit layout, which an implementation cannot derive on its own without
risking the same interleaved-vs-split-nibble ambiguity GQ4A's AVX2 kernel hit in
Phase 1):**

GQ2A has three independently-packed streams per superblock, each with its own natural
packing density — they are **not** packed the same way as each other or as GQ4A:

```
scale_delta (16 x i4 -> 8 bytes):
  Same convention as GQ4A's 4-bit weight packing (Pridwen v5 §3.1): byte k holds
  block 2k in bits [0:4) (low nibble) and block 2k+1 in bits [4:8) (high nibble).
  i.e. scale_delta_byte[k] = (scale_delta[2k] & 0xF) | (scale_delta[2k+1] << 4)
  scale_delta[i] is stored as its raw i4 two's-complement bit pattern (not offset
  like GQ4A's u4 weight codes) — decode by sign-extending the nibble: a nibble
  value >= 8 (bit 3 set) represents -16 + nibble, matching a standard 4-bit
  two's-complement range of [-8, 7].

min_delta (16 x i4 -> 8 bytes):
  Identical packing to scale_delta above, in a separate 8-byte region immediately
  following scale_delta's 8 bytes (per the block diagram's field order).

weights (256 x u2 -> 64 bytes):
  Four 2-bit codes per byte, low-to-high, sequential (not interleaved like GQ4A's
  4-bit weights, since GQ4A interleaves exactly 2 values per byte — with 4 values
  per byte here, sequential low-to-high is both the natural bit layout and the one
  that keeps a whole 16-weight block's codes contiguous across exactly 4 bytes,
  which the AVX2 kernel needs for a clean per-block load):
  byte k holds weight 4k in bits [0:2), weight 4k+1 in bits [2:4), weight 4k+2 in
  bits [4:6), weight 4k+3 in bits [6:8). i.e.
    weights_byte[k] = weight_int2[4k] | (weight_int2[4k+1] << 2)
                    | (weight_int2[4k+2] << 4) | (weight_int2[4k+3] << 6)
  Block b's 16 weights therefore occupy weights_byte[4b .. 4b+4) (4 contiguous
  bytes per block, 16 blocks x 4 bytes = 64 bytes total, matching the block
  diagram).
```

Field order within the 84-byte superblock (offsets, matching the diagram above):
`super_scale` (0..2), `super_min` (2..4), `scale_delta` (4..12), `min_delta`
(12..20), `weights` (20..84).

### 3.3 GQ2A-R — Incoherence-Processed Sub-Variant (Architecture-level, Phase 2 candidate)

**Reclassified from v4's §6 "Policy-adjacent" framing.** Incoherence processing
(QuIP#-inspired Hadamard rotation) requires storing a `rotation_seed: u32` per tensor so
the online dequant path can apply the correct inverse rotation. That is binary-format
side data, not an assignment decision — §1 restricts the "Policy changes never require
format changes" guarantee to the common case precisely because of this feature, and
GQ2A-R exists so the guarantee can stay true for GQ2A itself: a decoder or Policy that
has never heard of incoherence processing keeps using plain GQ2A (§3.2) unchanged.

```
SuperBlock (256 weights) — identical to GQ2A (§3.2), plus:
┌─────────────────────────────────────────┐
│ ...all GQ2A fields unchanged (84 bytes) │
│ rotation_seed: u32        (4 bytes)     │  ← NEW: per-tensor, stored once
├─────────────────────────────────────────┤
│ Per-superblock: 84 bytes (same as GQ2A) │
│ Per-tensor: +4 bytes (one seed, not     │
│   one per superblock — see below)       │
└─────────────────────────────────────────┘
```

`rotation_seed` is **per-tensor, not per-superblock** — the Hadamard matrix is applied
once to the whole weight matrix before quantization (§6), so one seed reconstructs the
inverse rotation for every superblock in that tensor. It is stored once per tensor
(e.g. as a manifest-level field on the tensor's `gllm.json` entry, or as a small
fixed-size header immediately preceding that tensor's superblock stream — **exact
placement is Open Question §15 Q2, unresolved as of this revision**), not duplicated
per-superblock. This keeps GQ2A-R's *effective bpw* within noise of plain GQ2A's 2.625
bpw for any tensor of realistic size (a 4-byte one-time cost against a multi-megabyte
tensor).

Reconstruction: identical to GQ2A's formula (§3.2) for `weight_f32` per block, followed
by the inverse Hadamard transform applied once over the whole reconstructed tensor:
`W_approx = H^T × W_gq2a_dequantized × H` (see §6 for the full offline/online
pseudocode).

**Status:** candidate, not committed — §6's decision gate still applies. If Phase 2's
empirical PPL delta does not justify FHT's overhead, GQ2A-R is never shipped and this
subsection is dropped from a future revision; GQ2A (§3.2) is unaffected either way.

### 3.4 GQ1A — Ternary (requires QAT)

```
SuperBlock (256 weights):
┌─────────────────────────────────────────┐
│ group_scale: f16          (2 bytes)     │
│ weights[256]: ternary                   │
│   packed 4 per byte (2-bit slot)        │  ← 64 bytes
│   encoding: 0b00 = -1                   │
│             0b01 =  0                   │
│             0b10 = +1                   │
│             0b11 = RESERVED (invalid)   │
├─────────────────────────────────────────┤
│ Total: 2 + 64 = 66 bytes                │
│ Deployed bpw: 2.0625                    │
│ Information-theoretic bpw: 1.58         │
└─────────────────────────────────────────┘
```

**`0b11` is reserved and must never be written by a GQ1A encoder.** A decoder that
encounters `0b11` in a weight slot **must treat the file as corrupt and error out**
(fail loud, per the crate's existing `GllmError`/checksum conventions) rather than
silently substituting one of the three valid values — silent substitution would corrupt
model weights without any detectable signal, which is worse than refusing to load. The
"1-bit" in the format name refers to the **information-theoretic** content (`log2(3) ≈
1.585` bits per ternary symbol, listed as "Information-theoretic bpw" above), not the
**deployed** on-disk bit-width, which is 2.0625 bpw because a 2-bit slot is the smallest
addressable packing this architecture uses. Where this proposal or downstream code needs
to talk about GQ1A's *actual storage cost*, "2.0625 bpw" is the correct figure to cite,
not "1-bit" or "1.58 bpw."

⚠️ GQ1A requires QAT (Quantization-Aware Training). PTQ (post-training quantization) to
ternary on models under roughly 3B parameters is **expected, based on published
quantization-degradation research on ternary/1-bit schemes (e.g. BitNet-family PTQ
ablations), to cause severe capability loss** — this has not been separately verified for
GwenLand's target models and is tracked as a Known Unknown (§12).

### 3.5 Memory Alignment Requirements

**Superblock *payloads themselves* are not required to be a multiple of any fixed byte
count** — GQ4A (138 bytes), GQ2A (84 bytes), and GQ1A (66 bytes) are all deliberately
tight, unpadded layouts, and padding them to 32 or 64 bytes would inflate effective bpw
for no correctness benefit (dequant reads the whole superblock sequentially regardless of
where it starts within a cache line). What *is* required to be 64-byte aligned (matching
real x86/ARM cache line size — v4 incorrectly stated 32 bytes) is the **start offset of
each tensor's superblock stream within its layer file** — i.e. tensor data segments, not
individual superblocks, begin on a 64-byte boundary. This matches
[`TENSOR_ALIGNMENT`](../glictus-caliburni/src/constants.rs) (64 bytes), already defined
in the crate for exactly this purpose, so no new alignment constant is introduced by
G-Quant. A sequential dequant loop over one tensor's superblocks will cross cache-line
boundaries mid-superblock for GQ2A/GQ4A/GQ1A (since none of their sizes divide 64 evenly)
— this is expected and acceptable, since the loop is streaming/sequential rather than
randomly indexing into individual superblocks.

---

## 4. Assignment Policies

Policies define the optimization objective used by the Assignment Engine during model conversion. Policies influence tensor precision selection, escape-hatch decisions, and heterogeneous precision assignment without modifying the underlying storage architecture (§1's guarantee — PFP/SFP/CPP all target Architecture A's existing GQ4A/GQ2A layouts, never GQ2A-R, which is selected independently as an Architecture choice, not a Policy one).

### PFP — Precision-First Policy

**Optimization objective:** Minimize quantization error, prioritize capability preservation.

**Stage 1 implementation (naive):** Hardcoded sensitivity table — sensitive layers assigned to highest available format, rest at base bit-width.

**Expected characteristics:**
- Lower quantization error
- Better capability retention
- Higher storage overhead
- Potentially lower inference throughput

### SFP — Speed-First Policy

**Optimization objective:** Maximize execution throughput and memory bandwidth efficiency.

**Stage 1 implementation (naive):** Hardcoded sensitivity table — only EXTREME sensitivity layers get escape hatch, everything else at minimum bit-width.

**Why latency_cost is NOT in Stage 1:**  
SFP requires per-format latency profiling on target hardware (dequant cycles per block, cache pressure per format). This data does not exist yet. Incorporating unvalidated latency estimates into the assignment objective would produce worse results than the naive heuristic. Latency-aware assignment is deferred to **Stage 3** (Assignment Engine sophistication, §7) after empirical profiling via glbench — which in terms of the Implementation roadmap (§14) happens no earlier than **Phase 3**.

**Expected characteristics:**
- Higher throughput (intent — not guaranteed until measured)
- Lower memory bandwidth usage
- Increased quantization error may occur
- Capability degradation depends on model characteristics

### CPP — Combined Precision Policy

**Optimization objective:** Balance precision preservation and execution performance.

**Stage 1 implementation (naive):** Sensitivity table from quantization degradation research — layer type → format assignment. This is the reference implementation that will be refined as empirical data arrives.

**Why MCKP is NOT in Stage 1:**  
The MCKP formulation requires two inputs that do not exist yet:
1. `sensitivity(layer_i, format_i)` — quantization error per layer per format, requires calibration run per model
2. `latency_cost(layer_i, format_i)` — hardware-specific dequant overhead, requires profiling on i3-1115G4

Without real data, an MCKP solver optimizes invented numbers and can produce assignments worse than the naive heuristic. MCKP is the correct long-term formulation and will be implemented in **Stage 3** once empirical data is available from **Stage 1 and Stage 2** work (§7). Stage 2's calibration work happens during Implementation **Phase 2**/early **Phase 3** (§14); Stage 3's MCKP solver work happens during Implementation **Phase 3** (§14) — see §7 for the full Stage roadmap and how it maps onto Phase numbers.

**Expected characteristics:**
- Balanced capability retention
- Balanced throughput
- Flexible heterogeneous precision assignment
- Recommended general-purpose conversion policy

---

## 5. Default Layer Sensitivity Assignment (CPP Stage 1 Reference)

Based on quantization degradation research. This table is the Stage 1 naive implementation (§7) — it will be revised as empirical PPL measurements arrive from real model runs.

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

**Note on GQ4A_CPP — degenerate case, not a demonstration of heterogeneous assignment.**
The `GQ4A_CPP assign` column is uniformly `GQ4A` for every quantized tensor (norms use
F32/F16 regardless of which quantized format is chosen elsewhere, so they don't count as
heterogeneity). This means GQ4A_CPP, the Phase 1 reference policy (§14), makes no
per-tensor format *decision* at all — it is CPP applied to a one-format universe. This is
expected and not a defect: GQ4A_CPP exists to validate the Architecture A block format
and Stage 1 pipeline plumbing (§9) in isolation, before GQ2A introduces a second format to
actually choose between. **The heterogeneous per-tensor assignment described in §1 as
G-Quant's primary differentiator first becomes real starting with GQ2A_CPP** (Phase 2),
whose column above does mix GQ4A-escape and GQ2A assignments. A reader should not expect
Phase 1's shipped output to exhibit multi-format tensors within one package.

**Effective bpw GQ2A_CPP** (Qwen2.5-0.5B reference):
- ~15% weights at GQ4A escape hatch = 4.3125 bpw
- ~85% weights at GQ2A = 2.625 bpw
- **Weighted average: ~2.87 bpw**

This is above a **hypothesized** "~2.8 bpw cliff" below which small-model quality is
expected to degrade sharply — this threshold is carried over informally from general
quantization-scaling intuition and has **not been measured for GwenLand's target models**.
It is not a validated GwenLand result and should not be read as one; tracked as a Known
Unknown in §12 (row "2.8 bpw cliff threshold validity").

---

## 6. Incoherence Processing (QuIP#-inspired — Architecture-level candidate, GQ2A-R)

Incoherence processing is a candidate **Architecture sub-variant** (GQ2A-R, §3.3) for GQ2A quality improvement, not a committed feature, and — per the reclassification in §1/§3.3 — not a Policy concern, since it requires an additive field in the block/tensor layout.

**The idea (from QuIP#):**
```
Offline (glconv):
  W_rotated = H × W × H^T     (H = random Hadamard matrix, seeded)
  quantize W_rotated → GQ2A-R
  store rotation_seed: u32     (4 bytes per tensor — see §3.3 for exact placement status)

Online (glproc dequant):
  W_approx = H^T × dequant(W_rotated_GQ2A-R) × H
```

Hadamard rotation spreads outlier weights uniformly across the matrix. Post-rotation distribution approaches Gaussian — significantly more friendly for uniform quantization.

**Why this is NOT committed yet:**

1. **QuIP# is CUDA + PyTorch** — their implementation has no zero-dep Rust equivalent. Fast Hadamard Transform (FHT) must be implemented from scratch. FHT is O(n log n) in theory, but actual overhead on i3-1115G4 is unknown. If dequant-time inverse transform is too slow, it eliminates the speed benefit of GQ2A entirely.

2. **Quality benefit unvalidated for GQ2A specifically** — QuIP# proves benefit for their E8 Lattice VQ format. Whether FHT preprocessing materially improves GQ2A's simpler uniform quantization on Qwen2.5-0.5B weights is an open empirical question.

**Implementation Phase 2 plan (§14):** Implement GQ2A first without incoherence processing. Measure PPL baseline. Then add FHT and GQ2A-R, and measure delta. If improvement is significant and overhead is acceptable → commit GQ2A-R as a shipped `DType`. If not → reject and document (same pattern as rejected-optimizations.md in glproc), and §3.3/this section are removed from the next revision.

---

## 7. Assignment Engine — Stage Roadmap

**Renamed from v4's "Phase Roadmap"** — this progression tracks how sophisticated the
Assignment Engine's tensor-format decisions get, which is orthogonal to §14's
Implementation Phases (what gets *built*, in what order). A single Implementation Phase
can span multiple Stages of Assignment Engine work; see the mapping note at the end of
this section.

```
Stage 1 (now):
  Assignment Engine = hardcoded sensitivity table
  Input:  tensor name → lookup table → format
  No solver, no profiling, no calibration data needed

Stage 2 (after GQ2A baseline):
  Sensitivity scoring = per-layer PPL delta measurement
  Input:  calibration run on real model → sensitivity scores
  Still greedy assignment, but scores are empirical not hardcoded

Stage 3 (after Stage 2 data):
  MCKP formulation becomes viable:
    - sensitivity(layer_i, format_i) from Stage 2 calibration
    - latency_cost(layer_i, format_i) from glbench profiling
  PFP / SFP / CPP differentiation becomes meaningful
```

**Stage-to-Phase mapping (informal, subject to change as Phase 2/3 work is scoped):**
Stage 1 work ships as part of Implementation Phase 1 (§14). Stage 2 work (calibration,
empirical scoring) begins once Phase 2's GQ2A baseline exists, and completes during
Phase 3. Stage 3 work (MCKP solver, latency profiling) is entirely Phase 3 work. Stage
numbers are never used interchangeably with Phase numbers elsewhere in this document —
every remaining "Phase N" reference refers exclusively to §14's Implementation Phases.

---

## 8. Integration with GLLM Format

G-Quant extends the `DType` enum in `glictus-caliburni`. **Note on the enum's real
shape:** `DType` itself (`manifest/types.rs`) is a plain, discriminant-free Rust enum —
the binary `u16` codes live separately, as named constants in `constants::dtype_codes`
(e.g. `FP32 = 0x0001`, `Q4_K = 0x0012`, `Q8_K = 0x0021`, `I32 = 0x0030` — grouped in
small per-family ranges), and `DType::from_code(code: u16)` maps a code to a variant via
an explicit `match`. G-Quant follows this exact existing pattern rather than attaching
`= 0x0200`-style discriminants directly to the enum (which is not how the current code
is written). New codes are chosen to sit in unused space well above the highest existing
code (`I32 = 0x0030`), avoiding any collision:

```rust
// constants::dtype_codes — new constants, same module/pattern as the existing ones
pub const GQ4A:  u16 = 0x0200;
pub const GQ2A:  u16 = 0x0201;
pub const GQ2AR: u16 = 0x0202; // GQ2A-R sub-variant (§3.3) — separate code, not a flag on GQ2A
pub const GQ1A:  u16 = 0x0203;
// Architecture B, C reserved: 0x0300, 0x0400

// manifest::types::DType — new variants, no explicit discriminants (matches existing style)
pub enum DType {
    // ... existing variants unchanged (F32, F16, Bf16, Fp8E4m3, Fp8E5m2,
    // Q4_0, Q4_1, Q4K, Q4Km, Q4Ks, Q5_0, Q6K, Q8_0, Q8K, I32, Unknown) ...

    // G-Quant native
    GQ4A,
    GQ2A,
    GQ2AR, // GQ2A-R sub-variant (§3.3)
    GQ1A,
}

// DType::from_code — new match arms, same pattern as the existing ones
match code {
    // ... existing arms ...
    dtype_codes::GQ4A  => Ok(Self::GQ4A),
    dtype_codes::GQ2A  => Ok(Self::GQ2A),
    dtype_codes::GQ2AR => Ok(Self::GQ2AR),
    dtype_codes::GQ1A  => Ok(Self::GQ1A),
    // ...
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

This heterogeneous per-tensor assignment is impossible in GGUF without hacks — it is a
first-class citizen in GLLM (demonstrated starting at GQ2A_CPP, §5).

---

## 9. Conversion Pipeline (glconv extension)

```
glconv model.gguf output/ --quant GQ2A --policy CPP
```

Phase 1 pipeline:
```
1. Parse GGUF                              (existing glconv)
2. Dequant GGUF weights → F32              (glcore dequant path)
3. Run sensitivity table lookup            (Stage 1: hardcoded CPP table, §7)
4. Quantize each tensor → assigned format  (glictus-caliburni::converter — new
                                             encode-direction module; see §10 for
                                             why this lives in the format crate,
                                             not glproc)
5. Write to GLLM package, dtype per-tensor
6. Self-validate                           (existing GllmPackage::open pattern)
```

Step 3 will be replaced by empirical scoring in Stage 2, and MCKP in Stage 3 (§7) — both
land no earlier than Implementation Phase 2/3 (§14).

---

## 10. Kernel & Encoder Requirements

### 10.1 Encoders (glictus-caliburni — format layer)

The **encode direction** (F32 weights → packed GQ4A/GQ2A/GQ1A superblocks, §9 step 4) is
new surface area not specified in earlier drafts of this proposal. It belongs in
`glictus-caliburni`'s converter module (alongside the existing GGUF→GLLM tensor mapping
in `converter.rs`), not in `glproc`, for the same reason the `DType` enum itself lives in
`glictus-caliburni` (§8): choosing block boundaries, scales, and per-tensor format
assignment is a **format/conversion-time decision**, not a per-inference compute
kernel — it runs once per model at `glconv` time, not once per forward pass. This mirrors
the crate's existing boundary: `glictus-caliburni` stays the zero-workspace-dep format
crate by default, and only opts into `glcore`/`glproc` behind the existing `converter`
and `glproc-backend` Cargo features (already used for the GGUF-reading and inference
paths respectively) — no new feature-gating pattern is introduced.

```rust
// glictus-caliburni/src/converter/gquant.rs (new; behind the existing `converter` feature)

pub fn quantize_gq4a(weights: &[f32]) -> Vec<GQ4ABlock>;
pub fn quantize_gq2a(weights: &[f32]) -> Vec<GQ2ABlock>;
// GQ2A-R (§3.3): same encoder plus the FHT forward pass and rotation_seed
// generation — deferred until §6's decision gate resolves.
pub fn quantize_gq1a(weights: &[f32]) -> Vec<GQ1ABlock>; // QAT-trained weights only, §3.4
```

### 10.2 Dequant kernels (glproc — compute layer)

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

// GQ2A-R: dequant_gq2a (above) plus one inverse-FHT pass over the full
// tensor — deferred until §6's decision gate resolves.
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

**Empirical evaluation is required for every model and deployment scenario.** Every
specific number quoted elsewhere in this document (§5's "~2.87 bpw" and "2.8 cliff",
§13's PPL/size targets) is a **target or hypothesis to be tested**, not a result — see the
inline notes at each of those locations, and §12's Known Unknowns table for what remains
unmeasured as of this revision.

---

## 12. Known Unknowns (explicit)

These are open questions that will be answered through empirical work — not assumed:

| Unknown | Answered in | Method |
|---|---|---|
| GQ4A PPL vs Q4_K_M delta | Phase 1 | glbench perplexity run |
| GQ2A baseline PPL (no FHT) | Phase 2 | glbench perplexity run |
| FHT overhead on i3-1115G4 | Phase 2 | glbench latency profiling |
| FHT quality delta for GQ2A (GQ2A vs GQ2A-R) | Phase 2 | PPL before vs after FHT |
| Per-layer sensitivity scores | Phase 2–3 (Stage 2, §7) | calibration run on Qwen2.5-0.5B |
| latency_cost per format | Phase 3 (Stage 3, §7) | glbench dequant kernel profiling |
| MCKP vs greedy quality delta | Phase 3 (Stage 3, §7) | A/B comparison on real model |
| GQ2A_CPP vs Q4_K_M real tok/s | Phase 2 | glbench decode benchmark |
| GQ1A PTQ collapse severity on <3B models | Phase 4 | calibration run, PPL vs QAT baseline (§3.4 currently states this as an expectation from general research, not a GwenLand-measured result) |
| 2.8 bpw cliff threshold validity | Phase 2 | PPL sweep across bpw values on Qwen2.5-0.5B (§5 currently uses this only as a rough target, not a validated cutoff) |
| GQ2A-R `rotation_seed` storage location (manifest field vs binary header) | Phase 2, before GQ2A-R ships | design decision, not empirical — see §15 Q2 |

---

## 13. Expected Results

Phase 1 target (GQ4A_CPP vs Q4_K_M) — **target, not a guarantee; see §11**:
- PPL: within 1% of Q4_K_M baseline
- Size: ~450MB vs 463MB (minor reduction)
- Decode: comparable — GQ4A dequant overhead vs Q4_K_M to be measured

Phase 2+ results: **TBD — requires empirical measurement.** No tok/s projections until Phase 1 latency data exists.

---

## 14. Implementation Phases

*(Unchanged track from v4 — this is the "Phase N" every other section now refers to
exclusively; see §7 for the separate, renamed "Stage N" Assignment Engine track.)*

### Phase 1 — GQ4A (Foundation)
- GQ4A block structure + scalar dequant + AVX2 fast path + encoder (§10.1)
- glconv: `--quant GQ4A --policy CPP` with hardcoded sensitivity table (Stage 1, §7)
- Validate: PPL parity vs Q4_K_M on Qwen2.5-0.5B via glbench
- Output: Proven baseline, real latency data for Phase 2 planning

### Phase 2 — GQ2A baseline (Primary)
- GQ2A block structure + scalar dequant + AVX2 fast path + encoder (§10.1)
- glconv: `--quant GQ2A --policy CPP`
- Validate: PPL measurement without FHT first
- Validate: FHT implementation + GQ2A-R (§3.3) + measure overhead + measure PPL delta
- Decision gate: commit or reject GQ2A-R based on empirical data (§6)
- Begin Stage 2 calibration work (§7)
- Output: First real GQ2A numbers — paper-worthy if quality holds

### Phase 3 — Assignment Engine (Research)
- Complete Stage 2 empirical sensitivity scoring via calibration run (§7)
- Stage 3: hardware latency profiling → `latency_cost` table (§7)
- Stage 3: MCKP solver implementation + A/B vs greedy (§7)
- PFP / SFP / CPP properly differentiated with real data
- Output: Full framework demonstrable, policies meaningful

### Phase 4 — GQ1A (Research track)
- Ternary format + QAT integration with gltrain
- Prerequisite: gltrain un-park, compute resources
- Validate: PTQ collapse severity claim from §3.4 (currently unverified for GwenLand models, §12)
- Output: Potential standalone paper

---

## 15. Open Questions

1. **Calibration data source** — Stage 2 sensitivity scoring (§7) needs ~128 samples. C4 subset or custom GwenLand calibration set?
2. **`rotation_seed` storage (GQ2A-R, §3.3)** — if GQ2A-R is committed in Phase 2 (§6 decision gate): per-tensor field in `gllm.json`, or a small fixed header embedded in the binary layer file immediately before that tensor's superblock stream? Both are viable; this document does not pick one yet. Tracked in §12.
3. **Architecture B roadmap** — candidate: FP4/MXFP4 support for GPU path (glcuda). Timing: after Phase 2 complete. (Note: this is a distinct reserved code range, `0x0300`, from GQ2A-R's `0x0202` — GQ2A-R is a same-generation Architecture A sub-variant, not part of Architecture B.)

*(v4's Open Question 4, "GQ4A exact compatibility with Q4_K_M," is removed in this
revision — §3.1 already answers it: GQ4A's 138-byte layout is not byte-compatible with
Q4_K_M's 144-byte layout, and the tradeoff rationale is stated inline there. This was not
actually an open question in v4; it was already implicitly settled by the block structure
given in the same section.)*

---

## 16. Paper Angle

> **"Pridwen: A Co-Designed Mixed-Precision Quantization Framework for Shard-Based LLM Inference"**

**Claimed contributions:**
1. Formal separation of Quantization Architecture from Assignment Policy as a programmatic API, with an explicit, documented boundary case (incoherence processing, §1/§3.3/§6) where the separation requires an Architecture sub-variant rather than a Policy change
2. First quantization framework co-designed with a shard-based weight format (GLLM)
3. Per-tensor heterogeneous precision as a first-class citizen enabling lazy expert loading (MoE on 8GB RAM) — demonstrated starting at GQ2A_CPP (§5)
4. GQ2A: 2-bit PTQ for small models via asymmetric superblock, **to be empirically validated in Phase 2** (not yet measured as of this revision — GQ2A-R/FHT quality delta is a separate, later question, §6)
5. Open-source pure Rust zero-dep implementation in the GwenLand ecosystem

**Target venue:** MLSys, EMNLP Systems Track, or arXiv standalone.

**Explicitly NOT claimed:**
- Novel quantization algorithm
- Guaranteed perplexity or throughput improvements
- First framework to decouple policy from format (prior art: dMX, PrismQuant, mlx-optiq acknowledged)
- MCKP solver (Stage 3, not yet implemented — §7)
- Incoherence processing / GQ2A-R (Phase 2 candidate, §6 decision gate not yet resolved, not yet validated)
- Any specific bpw/PPL cliff threshold as validated (§5's "2.8 bpw cliff" is a working hypothesis, §12)

---

## 17. Changelog (v4 → v5)

**Blocking fixes:**
- **#1** — Added explicit GQ2A asymmetric reconstruction formula (§3.2) and defined
  `dequant(weight_int4)`'s exact signed range for GQ4A (§3.1).
- **#2** — Corrected the cache-line claim (64 bytes, not 32) and rescoped the alignment
  requirement to tensor data segment start offsets rather than individual superblock
  sizes, which was contradicted by GQ4A/GQ2A/GQ1A's actual (non-64-divisible) byte totals
  (§3.5, renumbered from v4's §3.4).
- **#5** — Added an explicit note in §5 that GQ4A_CPP is a degenerate, homogeneous case
  of CPP and does not itself demonstrate heterogeneous per-tensor assignment; that claim
  is now scoped to GQ2A_CPP onward (also reflected in §1 and §16 claim 3).
- **#6** — Split v4's single overloaded "Phase N" label into two tracks: **Phase N**
  (§14, Implementation Phases — unchanged) and **Stage N** (§7, renamed from "Phase
  Roadmap" to "Stage Roadmap" — Assignment Engine sophistication). Updated every
  cross-reference in §4, §6, §9, §12 to cite the correct track, and added an explicit
  Stage-to-Phase mapping note at the end of §7.
- **#7** — Reclassified incoherence processing from a Policy-adjacent technique to an
  Architecture-level sub-variant, **GQ2A-R** (new §3.3), carrying the `rotation_seed`
  field GQ2A itself lacks. Updated §1's separation claim to state the guarantee applies
  to Policy changes that don't require new storage layouts, and to name this as the one
  documented exception. Updated §2 (naming convention now documents sub-variants), §8
  (new `GQ2AR = 0x0202` code, `GQ1A` renumbered to `0x0203`), §10 (GQ2A-R encoder/kernel
  notes), §14 Phase 2, and §15 Q2/Q3.
- **#9** — Added §10.1 specifying that the quantize-direction encoder lives in
  `glictus-caliburni`'s converter module (behind the existing `converter` feature),
  mirroring where `DType` itself lives, and clarified why this differs from the
  dequant-direction kernels' home in `glproc` (§10.2). Updated §9 step 4 to reference it.

**Non-blocking fixes:**
- **#3** — Removed v4's Open Question 4 ("byte-compatible with Q4_K_M?"); §3.1 already
  answers it (diverged, 138 vs 144 bytes) and the removal is explained inline in §15.
- **#4** — GQ1A's `0b11` state is now explicitly documented as reserved/invalid, with a
  fail-loud (error, not silent substitution) requirement on decode (§3.4).
- **#8** — §16 claim 4 reworded from present-tense "empirically validated" to
  conditional/future "to be empirically validated in Phase 2," matching §12's Known
  Unknowns framing.
- **#10** — §3.4's PTQ-collapse claim softened from stated fact to "expected, based on
  published quantization-degradation research," with a corresponding row added to §12.
- **#11** — §5's "2.8 cliff threshold" reframed as a hypothesis (with the ✅ checkmark
  removed) and added to §12 as an unmeasured Known Unknown.
- **#12** — §13's Phase 1 PPL/size targets now carry an inline "(target, not a guarantee;
  see §11)" pointer, and §11 itself now references §13/§5 by name.
- **#13** — §3.3 (new GQ2A-R section) forward-references §15 Q2 for the unresolved
  `rotation_seed` storage-location question, and §12 carries a corresponding row.

**Not changed (verified correct in the v4 analysis, left as-is):**
- bpw arithmetic: GQ4A 4.3125, GQ2A 2.625, GQ1A 2.0625 deployed / 1.58 information-theoretic.
- Naming convention `GQ<bit><Arch>_<Policy>` (extended, not replaced, to cover sub-variants).
- `DType` numbering range `0x0200+` (no collision with the existing enum up to `I32 = 0x0030`).
- Architecture/compute boundary: format-layer concerns (`DType`, manifest) in
  `glictus-caliburni`, compute-layer dequant kernels in `glproc`, both gated behind the
  crate's existing optional-feature pattern.
