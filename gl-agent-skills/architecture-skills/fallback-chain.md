# The Fallback Chain

> **Domain:** architecture-skills
> **Applies to:** `glcore` runtime; every backend's availability reporting
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the order: **glcuda → glvulkan → glmetal → glproc**, and that `glproc` is the floor — always available, never allowed to become conditional.
- [ ] "Engine unavailable" in my change is a *reported state*, not an error, not a panic.
- [ ] The chain is wired so it works with stub engines (that's why the stubs exist) — my change must not require conditional compilation to keep the chain intact.

## Context

The runtime tries engines in preference order and settles on the first one
that is available *and* can serve the request. This is the mechanism behind
two promises at once: "the same tree builds and runs anywhere" (a GPU-less
laptop silently lands on glproc) and "a GPU is optional, never required."
The stubs (`glvulkan`, `glmetal`) return not-implemented precisely so the
chain logic is real code today, not `cfg`-gated future code.

## Rules

1. **Chain order is fixed:** glcuda → glvulkan → glmetal → glproc. Changing
   preference order is an architecture decision (spec + sign-off), not a
   tweak.
2. **glproc is the unconditional floor.** It must initialize on any machine
   the project supports (pure Rust, scalar fallback included). No change may
   make glproc's availability conditional on a CPU feature — `SimdStrategy`
   degrades to scalar instead.
3. **Unavailability is normal and cheap:** missing driver, stub engine,
   unsupported dtype → the engine reports `available: false` (or a clean
   `GlError` at init), the runtime moves on. No panics, no process exit, no
   retry loops.
4. **Every fallback decision is logged with its reason** ("glcuda: driver
   not found → trying glvulkan"). Silent fallback is forbidden — a user on
   an unexpectedly slow path must be able to see why from the log, and
   glbench must be able to record which engine actually ran.
5. **Selection happens once per session/request — not per op.** The chain
   picks *an engine*; it never splits a forward pass across engines
   ([backend-independence.md](backend-independence.md) Rule 2).
6. **An explicit `--engine` request does not silently fall back.** If the
   user names an engine and it can't serve, that's an error with the reason
   — fallback is for *automatic* selection only.
7. **New engines join the chain by implementing `GlEngine`** and being
   registered in the runtime — the chain code itself should not need
   engine-specific branches.

## ✅ Correct Pattern

```text
Auto-select on a GPU-less Windows laptop:
  glcuda:   LoadLibraryA("nvcuda.dll") fails → log "glcuda unavailable: no
            driver" → next
  glvulkan: stub → "not yet implemented" → next
  glmetal:  stub/off-platform → next
  glproc:   init OK → selected. Log: "engine: glproc (cpu)".
Result: user runs; glbench sessions record engine=glproc.
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ hard requirement where a fallback belongs:
let engine = CudaEngine::new().expect("CUDA required");

// ❌ silent fallback on an explicit request:
// user passed --engine glcuda
Err(_) => runtime.select_best(),   // user thinks they measured CUDA; they
                                   // benchmarked the CPU. Fail loudly instead.

// ❌ per-op scavenging across the chain:
if cuda_result.is_err() { vulkan.infer(input) } // chain picks ONE engine
```

## GwenLand-Specific Notes

- The fallback chain is why **stub crates must keep compiling** at every
  commit — deleting a stub or making it `cfg`-conditional breaks the "no
  conditional compilation" property the runtime was designed around.
- Fallback reasons are benchmarking data: a glbench session on the "wrong"
  engine is only interpretable because the session records which engine ran
  and why (see [`../bench-skills/glbench-usage.md`](../bench-skills/glbench-usage.md)).

## Related Skills

- [backend-independence.md](backend-independence.md)
- [inference-first.md](inference-first.md)
- [../cuda-skills/dynamic-loading.md](../cuda-skills/dynamic-loading.md)
