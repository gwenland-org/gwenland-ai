# Architecture Decision — Pridwen Phase 2
**Date:** 2026-07-22  
**Decision by:** JinXSuper

---

## Context

Pridwen Phase 1 notes (pridwen-p1-notes.md, BLOCKER entry 2026-07-22T03:00:00Z)
identified that `glcore::dequantize` does not support Q4_K and Q5_0.

Result: real Qwen2.5-0.5B conversion only reached ~8.6% of tensors (25/291).
The bulk of FFN/attention weights (Q4_K + Q5_0 sourced) passed through
unconverted. End-to-end GQ4A_CPP on a real model is blocked until this is resolved.

---

## Decision

**Option B is chosen: give `converter` feature a reason-argued glproc dependency.**

### Rationale

`glproc` already has battle-tested, AVX2-optimized dequant kernels for Q4_K,
Q5_0, and all other GGUF quantized dtypes — proven through Veritas Prima sprints,
83 tests green, worst error 2.83×10⁻⁴.

Implementing Q4_K/Q5_0 dequant in `glcore` (Option A) would be:
- **Reinventing the wheel** — duplicating proven code that already exists
- **Unmaintained duplication** — two dequant paths for the same dtype diverge over time
- **Not actually cleaner** — DRY violation is worse than the boundary extension

### Precedent in existing codebase

```
glictus-caliburni/glproc-backend feature → depends on glproc (runtime path)
glictus-caliburni/converter feature      → depends on glproc (conversion path) ← THIS
```

Same pattern, different phase of the pipeline. The boundary is:
- format layer (glictus-caliburni) = owns structs, manifest, serialization
- compute layer (glproc) = owns dequant kernels, inference

`converter` using glproc's dequant kernels does not move compute logic
into the format layer — it uses compute from the compute layer, which is correct.

### The "reason" for reason-argued dependency

> glproc is the authoritative, tested, optimized dequant implementation
> for all quantized GGUF dtypes in the GwenLand ecosystem.
> No other crate should duplicate this work.

---

## Implementation (Phase 2 task for Claude Code)

Add to `glictus-caliburni/Cargo.toml`:
```toml
[features]
converter = ["dep:glcore", "dep:glproc", "gquant"]  # glproc added here
```

Update `glictus-caliburni/src/converter.rs`:
- Replace `glcore::dequantize(tensor)` calls with glproc dequant kernels
  for Q4_K and Q5_0 sourced tensors
- All other dtypes (F32/F16/BF16/Q8_0/Q6_K) keep existing glcore path

Expected result after fix:
- Real Qwen2.5-0.5B: ~291/291 tensors reach GQ4A encoder (not 25/291)
- Conversion warnings drop from 219 to ~0
- glbench PPL comparison becomes meaningful

---

## Notes for Claude Code

- Do NOT implement this in Phase 1 — Phase 1 is already complete as-is
- This is the first task of Phase 2, before GQ2A block structure work begins
- After fix: re-run `glconv qwen2.5-0.5b-instruct-q4_k_m.gguf out/ --quant GQ4A --policy CPP`
  and record new dtype tally + package size in notes/pridwen-p1-notes.md
- Then proceed with GQ2A implementation
