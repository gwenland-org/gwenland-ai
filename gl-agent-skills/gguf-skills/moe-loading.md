# MoE Loading (`_exps` tensors)

> **Domain:** gguf-skills
> **Applies to:** `glproc` MoE path ([`moe.rs`](../../glproc/src/moe.rs), `split_experts` in [`loader.rs`](../../glproc/src/loader.rs)), [`glictus-caliburni/`](../../glictus-caliburni/) (experimental GLLM shard format)
> **Last updated:** 2026-07-17
>
> ⚠️ **Upstream drift warning:** the `_exps` stacked-expert tensor convention
> comes from **ggml / llama.cpp**, and our reading of it is written from
> their conventions — not from an authoritative spec. If upstream changes or
> clarifies expert-tensor packing, this skill and the loader assumption below
> must be revisited immediately.

## BEFORE YOU START

- [ ] ⚠️ I know the **#1 hazard in this domain**: the `_exps` stacking-order layout is an **UNVERIFIED ASSUMPTION** — marked `_EXPS_LAYOUT_ASSUMPTION` in `glproc/src/loader.rs`. Grep it before touching anything.
- [ ] I understand the failure mode is **silent**: a wrong expert order passes shape checks, runs fine, and produces *fluent garbage*.
- [ ] I know what IS verified: the MoE compute path at Qwen3 scale — 128 experts, top-8 routing, structural skip confirmed (~8/128 experts touched per token).

## Context

MoE GGUFs pack all experts of a layer into single stacked `…_exps` tensors.
`split_experts` slices them per expert, cross-checking declared dims against
`expert_count` metadata and byte lengths — so a wrong *shape* cannot load.
But a wrong **stacking order** that satisfies those checks still loads: every
expert gets internally-consistent-looking weights that belong to a different
expert. The model runs, the router routes, and the output is plausible
nonsense. That's why the assumption is loudly marked and why verification
against a real dumped file is the gate for trusting this path.

## Rules

1. **The layout assumption stays marked until verified.** Do not remove or
   rename `_EXPS_LAYOUT_ASSUMPTION` markers until someone has inspected a
   real Qwen3-MoE GGUF's bytes (dump a small expert slice, compare against
   llama.cpp's dequant output for the same tensor) and recorded the result
   in the changelog. Verification, not vibes, closes it.
2. **Any change to `split_experts` keeps the cross-checks:** declared
   dimensions × expert_count × dtype block math must equal the stacked
   tensor's byte length exactly. These checks have real detection value —
   never weaken them to admit an odd file.
3. **MoE correctness claims are scoped:** "compute path verified at Qwen3
   scale" means routing math, top-k selection, and expert FFN evaluation —
   it does **not** certify the expert *identity* mapping (rule 1). Keep that
   distinction in any status you write.
4. **Structural skip is the perf model:** only routed experts (top-8 of 128)
   are touched per token. Loading/paging strategy must preserve that — the
   RAM win of MoE on an 8 GB box *is* the skip.
5. **Expert-level threading is forbidden** by the pool's non-reentrancy
   (`ThreadPool::run`, see
   [`../cpu-skills/threading-model.md`](../cpu-skills/threading-model.md)) —
   experts are evaluated within the existing worker structure, not
   per-expert workers.
6. **`glictus-caliburni` (GLLM shard format) is experimental:** shard-based
   weights + lazy expert loading targeting the 8 GB budget
   (`format.rs`/`shard.rs`/`router.rs`/`loader.rs`). Treat its format as
   unstable; nothing in the product path may hard-depend on it, and its
   design questions go through `Experimental/`-style validation before the
   format is called real.
7. **A real-file fixture beats a synthetic one here:** synthetic MoE tests
   validate the math; only a real GGUF validates the layout. Until a small
   real fixture is feasible, every MoE result carries the rule-1 caveat.

## ✅ Correct Pattern

```rust
// loader.rs — the honest marker pattern (keep it this loud):
// _EXPS_LAYOUT_ASSUMPTION: experts assumed stacked contiguously in expert-
// major order per llama.cpp convention. NOT verified against real file
// bytes. Wrong order passes shape checks and produces fluent garbage.
// Verify before trusting MoE output; see gl-agent-skills/gguf-skills/moe-loading.md.
let per_expert = split_experts(stacked, expert_count, dims)?; // cross-checked
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "MoE output looks coherent, so the layout must be right" — fluent garbage
   is precisely the failure mode; coherence proves nothing here.

❌ Deleting the _EXPS_LAYOUT_ASSUMPTION comment during a refactor because
   "the code moved" — move the marker with the code.

❌ Spawning a thread per routed expert — pool is not reentrant; measured
   alternative (moe_threads bench) exists, read it first.
```

## GwenLand-Specific Notes

- [`docs/Mensura_Veritatis.md`](../../docs/Mensura_Veritatis.md) carries the
  full evidence table for this path (search `_EXPS_LAYOUT_ASSUMPTION`) —
  including which claims are `[M] Measured` vs `Evidence Required`. Keep
  that document's status in sync when the verification lands.
- Router/expert telemetry (`experts_touched`) exists so glbench can see the
  structural skip — MoE perf work must keep feeding it.

## Related Skills

- [format-parsing.md](format-parsing.md)
- [../cpu-skills/threading-model.md](../cpu-skills/threading-model.md)
- [../before-coding/read-architecture-first.md](../before-coding/read-architecture-first.md)
