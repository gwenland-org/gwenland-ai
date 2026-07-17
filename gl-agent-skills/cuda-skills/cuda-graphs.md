# CUDA Graphs (Decode)

> **Domain:** cuda-skills
> **Applies to:** `glcuda` — graph capture/launch in [`driver.rs`](../../glcuda/src/driver.rs) / [`runner.rs`](../../glcuda/src/runner.rs)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know Stage 1+2 of graph decode is **already implemented** — I'm not re-introducing it.
- [ ] I have internalized the measured conclusion: the remaining cost is **inter-kernel data-dependency serialization**, NOT launch overhead.
- [ ] If my plan says "reduce launch overhead", I stop now — that lever is spent.

## Context

The decode step is captured into a CUDA graph and replayed per token, which
already removed per-kernel launch overhead from the critical path. Profiling
after that landed showed the residual gap comes from the *structure* of the
work: each kernel in the token loop depends on the previous one's output, so
the GPU serializes through a chain of small kernels with dead time at each
edge. That is a fusion problem, not a launch problem — and it bounds which
optimizations can possibly pay.

## Rules

1. **Do not optimize launch overhead further.** Tried, measured, exhausted —
   graphs already own that cost. Any proposal whose mechanism is "fewer/faster
   launches" on the decode path is rejected by default.
2. **The lever is kernel fusion** — reducing the number of dependency edges in
   the token graph (e.g. norm+matmul, dequant+dot, gate+up+SwiGLU). Fusion
   proposals follow [kernel-design.md](kernel-design.md) Rule 4: parity
   against the unfused baseline first, production A/B second.
3. **Graph capture assumes fixed shapes and fixed buffer addresses.** All
   buffers in the captured region come from the bump allocator and never
   move ([memory-management.md](memory-management.md)). Anything
   shape-dynamic (prompt-length-dependent work) stays *outside* the captured
   decode step.
4. **Per-token varying scalars go through pinned/device memory updated before
   replay** — never by re-capturing the graph per token. Re-capture is a
   load-time / config-change event only.
5. **Graph APIs are optional driver symbols** — resolve per
   [dynamic-loading.md](dynamic-loading.md) Rule 4 and keep the
   individual-launch path alive as the degraded mode on old drivers.
6. **Changing the decode step = re-validating the graph:** parity suite AND a
   check that graph replay output matches individual-launch output for the
   same seed/input.

## ✅ Correct Pattern

```text
Proposal shape that is allowed to proceed:
"Fuse dequant+dot for the FFN down-projection: removes 2 dependency edges
 per layer from the decode graph. Plan: parity vs unfused (TOL_MATMUL,
 incl. dim 896) → bench.rs A/B on T4 production decode → gate report."
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Batch kernel launches / use streams to overlap decode kernels" —
   decode kernels are data-dependent; there is nothing to overlap, and
   launches are already amortized by the graph.

❌ Re-capturing the graph every token to handle a changing scalar.

❌ Allocating a fresh buffer inside the captured region (breaks capture,
   breaks the zero-alloc contract).
```

## GwenLand-Specific Notes

- Keep perspective on the ceiling: decode already runs at ~88 % of memory
  bandwidth on the T4. Fusion buys back dependency-edge dead time, but the
  bandwidth roofline caps the total win — size expectations accordingly
  before spending days on a fused kernel.
- The individual-launch fallback path is not dead code — it's the degraded
  mode for older drivers *and* the reference for validating graph replay.
  Keep it working and tested.

## Related Skills

- [kernel-design.md](kernel-design.md)
- [memory-management.md](memory-management.md)
- [dynamic-loading.md](dynamic-loading.md)
