# Read the Architecture First

> **Domain:** before-coding
> **Applies to:** every crate, every task
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I have read the ground-truth doc(s) for the crate I am about to touch (table below).
- [ ] I have verified the crate/module structure with `cargo metadata` or by reading `Cargo.toml` — not assumed it.
- [ ] I have checked [`../../ROADMAP.md`](../../ROADMAP.md) so my change fits the current milestone.
- [ ] I have confirmed nothing I plan to do is on a rejected-optimizations list (`../cpu-skills/rejected-optimizations.md`).
- [ ] I am not treating anything under [`../../Experimental/`](../../Experimental/README.md) as a spec.

## Context

GwenLand's architecture decisions are *measured*, not aesthetic — many "obvious
improvements" were tried and reverted with numbers attached. An agent that codes
from general LLM-engine intuition instead of this repo's documents will
re-introduce known regressions. Reading first is cheaper than reverting.

## Rules

1. **Ground truth by domain — read before editing:**

   | You are touching | Read first |
   |---|---|
   | `glcuda` (anything) | [`architecture/ArchGLML_X2.md`](../../architecture/ArchGLML_X2.md) — the glcuda M2 ground truth |
   | `glproc` / CPU path | [`architecture/ArchGLLM_X5.md`](../../architecture/ArchGLLM_X5.md) (M1.5 bridge/SIMD/threading spec, final & locked) and [`docs/Mensura_Veritatis.md`](../../docs/Mensura_Veritatis.md) (the measured glproc knowledge base) |
   | Overall engine story / benchmarks | [`architecture/ArchGLML.md`](../../architecture/ArchGLML.md) |
   | glcuda validation claims | [`docs/ArchGLCuda/ArchGLML_Done.md`](../../docs/ArchGLCuda/ArchGLML_Done.md) |
   | Anything | [`ROADMAP.md`](../../ROADMAP.md), [`CONTRIBUTING.md`](../../CONTRIBUTING.md) |

2. **Never assume crate structure.** The workspace members are declared in the
   root [`Cargo.toml`](../../Cargo.toml): `glcore`, `glproc`, `glcuda`,
   `glvulkan`, `glmetal`, `glbench`, `glcli`, `glictus-caliburni`, and
   `packages/{core,gltui,mcp}`. Verify a module exists before importing it.
3. **Every number in the architecture docs was measured.** Do not "correct" a
   measured number from theory. If your theory disagrees with a measurement,
   the measurement wins until you re-measure.
4. **Session history lives in `changelog/`.** Before re-attempting something,
   grep `changelog/` — if it was tried and reverted, the entry says why.
5. **`Experimental/` is research, not spec.** Ideas there graduate through
   `NewExperiment.md` → GWEN-XXX issue → real spec. Never implement directly
   from an Experimental document.
6. If a document this file points to does not exist at that path, STOP and
   report the broken pointer — do not guess a substitute.

## ✅ Correct Pattern

```text
Task: "speed up glproc FFN"
1. Read ArchGLLM_X5.md + docs/Mensura_Veritatis.md
2. Read cpu-skills/rejected-optimizations.md  → L2 tiling is banned, etc.
3. grep changelog/ for "FFN"                  → find prior attempts + numbers
4. Only THEN form a plan, citing the docs you read
```

## ❌ Anti-Pattern (Never Do This)

```text
Task: "speed up glproc FFN"
1. "FFN matmuls usually benefit from cache tiling, let me add L2 tiling"
   → re-introduces a rejected optimization that measured zero benefit
     because decode is bandwidth-bound, not cache-bound.
```

## GwenLand-Specific Notes

- The primary CPU reference machine is a Tiger Lake **i3-1115G4** (2P/4T,
  AVX2, 8 GB DDR4-2667 dual-channel). Optimizations are judged on this tier —
  results from a big desktop CPU do not transfer.
- CPU decode is **FFN-bound** (biggest bucket), the opposite of the GPU
  prefill profile where attention was the single biggest bucket. Don't carry
  conclusions across engines.

## Related Skills

- [check-existing-tests.md](check-existing-tests.md)
- [wave-confirmation-gates.md](wave-confirmation-gates.md)
- [../cpu-skills/rejected-optimizations.md](../cpu-skills/rejected-optimizations.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
