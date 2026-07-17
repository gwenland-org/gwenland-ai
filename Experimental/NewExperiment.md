# NewExperiment — Math / Performance Whitepaper Backlog

A scratchpad for proposing, evaluating, and tracking **mathematical or systems
optimizations** for GwenLand's CPU-only training path. Each idea here is a
candidate that can graduate into a Linear issue (`GWEN-XXX`) + a real spec under
`.kiro/specs/` once it survives the "Reality check" below.

> Rule of thumb: an optimization only counts if it makes a **measured**
> wall-clock or RAM number better **without** corrupting the loss/gradients.
> Approximations that change the math must prove the error is acceptable.

---

## 0. The problem we are actually optimizing

Target hardware (non-negotiable): **i3 11th gen, 8 GB RAM, no GPU**, mmap-based
OOM-safe layered loader.

Measured baseline (Qwen3-1.7B Q8_0, `LayeredTrainingLoop`, batch_size=1):

| Metric | Value | Source |
|---|---|---|
| Wall time / optimiser step | **~200–320 s** (short fixture) | GWEN-219 dry-run |
| Peak RSS | **~1.04 GB** | GWEN-219 dry-run |
| Trainable LoRA params | 8.72 M | GWEN-219 dry-run |
| Step-1 loss (real attention, real text) | ~2.77 | GWEN-220 Wave 4 |
| Vocab (capped) | 8192 (full ≈152k) | runtime |

**Where the time goes (the real bottleneck):** each step re-**dequantizes all 28
Q8_0 layers from the mmap, twice** (forward boundaries + gradient-checkpoint
recompute). It is **dequant + memory-bandwidth bound, not multiply bound.** Step
cost also scales with **sequence length²** (attention) and with sequence length
(projections), so longer samples → slower steps.

This is the surface every whitepaper below should attack. Speeding up "the
multiply" specifically is a dead end (see Rejected ideas).

---

## Template — copy this per idea

```
### <short title>
- **Status:** proposed | prototyping | measured | accepted | rejected
- **Targets:** dequant cost | memory bandwidth | seq-len² | RAM | step count | other
- **Claim (1 sentence):**
- **The math:** (the actual formula / transform, with where the error lives)
- **Why it could help here:** (tie to the measured bottleneck above)
- **Risk to correctness:** (does it change the loss/gradient? bound the error)
- **Cheap experiment:** (smallest test that proves/kills it — target metric + how to measure)
- **Result:** (fill after measuring: before → after, error vs baseline)
```

---

## Candidate backlog

### Per-layer dequant cache ("dequant once, reuse within step")
- **Status:** proposed
- **Targets:** dequant cost
- **Claim:** The forward pass and the reverse recompute each dequantize the same
  28 layers — caching a layer's F32 weights for the duration of one step removes
  half the dequant work (~up to ~2× step speedup) for a bounded RAM bump.
- **The math:** none — exact. Pure memoization of `dequant(W_q)` within a step.
- **Why it could help here:** dequant is the dominant cost; we currently pay it
  2× per step. One layer of F32 weights ≈ tens of MB, well under the 8 GB budget.
- **Risk to correctness:** zero (bit-identical weights).
- **Cheap experiment:** cache the *current* layer only (1 layer live at a time,
  preserves the GWEN-216 invariant), measure step wall-time before/after on the
  6-layer synthetic GGUF. Target: ~30–50 % step-time drop.
- **Result:** _TBD_

### Sequence-length budget for validation runs
- **Status:** proposed
- **Targets:** seq-len² + step count
- **Claim:** Truncating training samples to N tokens for *validation* makes the
  loss-trend smoke test minutes instead of an hour, with no code-path change.
- **The math:** attention is O(L²); halving L ≈ 4× cheaper attention.
- **Risk to correctness:** none for *validation* (we are checking the loop, not
  shipping the adapter). Would be wrong for a *real* training run.
- **Cheap experiment:** add a `max_seq_len` knob, run the Wave-4 trend at L=128.
- **Result:** _TBD_

### Mixed-precision dequant (F16 working weights)
- **Status:** proposed
- **Targets:** memory bandwidth
- **Claim:** Dequantizing Q8_0 → F16 instead of F32 halves the bytes moved per
  matmul; on a bandwidth-bound CPU step that can be a real win.
- **The math:** none structurally; numerical error = F16 rounding (~1e-3), needs
  a loss-delta check vs F32 baseline.
- **Risk to correctness:** small but real — must bound loss drift over N steps.
- **Cheap experiment:** F16 forward, compare step-1 loss vs the 2.77 F32 number.
- **Result:** _TBD_

### (Your idea) Taylor / log-space "multiply → add"
- **Status:** rejected — see Rejected ideas
- Kept here so it is not re-proposed.

---

## Rejected ideas (and why — so we don't loop back)

### Taylor series / log trick to turn multiplications into additions
- **Why rejected:**
  1. **Not the bottleneck.** Steps are dequant/bandwidth bound; the multiplies
     are not the slow part.
  2. **Hardware already fuses it.** CPUs do fused multiply-add (FMA): a multiply
     and add in one ~1-cycle instruction. Removing the multiply saves nothing.
  3. **The detour is *more* expensive.** `log`/`exp` (or summing series terms)
     cost tens of cycles each — far more than the single multiply avoided.
  4. **It breaks the math.** Truncated Taylor = approximation error injected into
     every projection → corrupted gradients, the exact faithfulness GWEN-220 is
     verifying.
- **Salvageable kernel:** the only place a series/exp identity is even relevant
  is the **softmax `exp`** in attention — but that is already a tiny fraction of
  step cost, so not worth it. Park it unless profiling ever says otherwise.

---

## How an idea graduates

1. Fill the template + run the **Cheap experiment**; record before→after numbers.
2. If it wins and correctness holds → open `GWEN-XXX`, write a `.kiro/specs/`
   spec with waves + gates (same format as GWEN-220).
3. Implement on a throwaway branch first; never regress the GWEN-216 one-layer
   RAM invariant or the GWEN-219 multi-tensor LoRA routing.
