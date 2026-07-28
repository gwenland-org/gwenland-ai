# Optimization Taxonomy

Each category has a name, mechanism, examples from this investigation, and expected production impact based on observed evidence.

---

## Category 1 — Remove Work

**Mechanism:** Eliminate computations entirely. The work is genuinely redundant — the same result was computed multiple times and only one instance was needed.

**Expected production impact:** High. Savings transfer unconditionally to production because the work is gone in every regime, not just the one the kernel was optimized for.

**Why it works:** There is no regime (compute-bound or bandwidth-bound) where eliminating real redundancy can make things worse. The reduction in both compute and memory traffic is proportional to the redundancy factor.

**Examples from this investigation:**

| Example | Redundancy removed | Measured production gain |
|---|---|---|
| RoPE table cache | 384× redundant sin/cos per decode step | +1.4–5.1% decode, +6.8% prefill |

**Contrasted with:** Category 2 (Accelerate Existing Work), which does the same computations faster. Category 2 is regime-dependent.

---

## Category 2 — Accelerate Existing Work

**Mechanism:** Perform the same computations faster — wider SIMD, better instruction selection, more parallelism in the hot loop.

**Expected production impact:** Depends on bottleneck regime at the stage being optimized. If the stage is compute-bound: possible gain. If bandwidth-bound or indeterminate: likely neutral.

**Why it does not always work:** A stage running at 68% of the bandwidth ceiling is not purely compute-bound. The bottleneck switches. A kernel that runs 2× faster at peak GMAC/s will be limited by memory bandwidth before it can double production throughput.

**Examples from this investigation:**

| Example | Technique | Isolated gain | Production gain |
|---|---|---|---|
| VNNI-512 | 256-bit → 512-bit vecdot | +20–26% GMAC/s | +0.3–1.1% (noise) |
| Row Tile | 8-accumulator qdot | +2× GMAC/s | +0.8–1.1% (noise) |

**When it can work:**
- The stage is confirmed compute-bound (ceiling efficiency approaching 100%, arithmetic intensity high, bandwidth utilization low).
- The stage occupies a large share of total time (P-04 bound is meaningful).
- The optimization is dispatched only to stages that are actually compute-bound (not globally).

---

## Category 3 — Reduce Memory Traffic

**Mechanism:** Reduce the number of bytes read from DRAM per operation. Techniques: fused kernels (read once, compute multiple ops), quantization, caching, tiling for cache reuse.

**Expected production impact:** High for bandwidth-bound stages; neutral for compute-bound stages.

**Why:** The bandwidth ceiling on this machine is 31 GB/s. lm_head is already at 73.5% of the ceiling and classified bandwidth-bound. Any reduction in bytes read by lm_head directly reduces its time.

**Examples from this investigation:**

| Example | Technique | Status |
|---|---|---|
| Q8_0 repack (GATE selection) | Avoid re-quantizing Q4_K at every step; serve already-repacked Q8_0 | Active in production; GATE calibrates which format wins |
| Fused SwiGLU (ffn_gate_up) | Single kernel fuses gate + up + activation | Active in production (kernel: "Q8_0 fused-swiglu integer-dot") |

**Candidates not yet attempted:**
- Per-layer dequant cache (load each layer once per step instead of twice, saving ~2× dequant bandwidth).
- Source: `Experimental/NewExperiment.md`.

---

## Category 4 — Reduce Startup Cost

**Mechanism:** Move one-time costs out of the per-token path. Caching, precomputation, or lazy initialization.

**Expected production impact:** Zero effect on per-token throughput; visible effect on session startup latency.

**Examples from this investigation:**

| Example | One-time cost | Per-token cost |
|---|---|---|
| GATE calibration | ~2.5 seconds at load | Zero (confirmed: decode tok/s unchanged vs pre-GATE) |
| RoPE table cache | Table computed once at model load | Eliminated per-step recomputation |

**Note:** RoPE cache is both Category 1 (removes redundant work per token) and Category 4 (precomputes the table once at load).

---

## Category 5 — Precision / Format Change

**Mechanism:** Change the representation of weights or activations (e.g., Q4_K → Q8_0, f32 → f16) to trade precision for speed or bandwidth.

**Expected production impact:** Mixed. Lower-precision formats are faster to stream (less bandwidth) but may harm PPL/output quality.

**Examples from this investigation:**

| Example | Change | PPL cost | Speed gain |
|---|---|---|---|
| Q4_K → Q8_0 repack | Wider integers, simpler kernel | +9% of PPL gap vs llama.cpp | Kernel speed justified GATE selecting it |

**Risk:** Format change is not precision-neutral. Measure PPL before and after.

Source: `notes/issues/glproc-precision-gap-vs-llamacpp.md`.
