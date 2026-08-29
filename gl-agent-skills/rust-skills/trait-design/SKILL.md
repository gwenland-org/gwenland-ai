# Trait Design

> **Domain:** rust-skills
> **Applies to:** `glcore` (trait definitions), every backend (implementations)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I have read [`glcore/src/engine_trait.rs`](../../glcore/src/engine_trait.rs) — the actual `GlEngine` trait, not my mental model of it.
- [ ] If I'm changing `GlEngine` itself: I understand every backend implements it, and it must stay **object-safe** (`Box<dyn GlEngine>` in the runtime).
- [ ] Dynamic dispatch stays at the engine boundary; hot loops use static dispatch.

## Context

The entire gl-stack hangs off one trait: `GlEngine: Send + Sync`. The runtime
holds engines as `Box<dyn GlEngine>` and owns zero compute — so the trait is
the *only* coupling between backends and the rest of the system. A careless
trait change ripples into four engines at once; a non-object-safe change
breaks the runtime outright.

## Rules

1. **`GlEngine` must remain object-safe.** No generic methods, no
   `impl Trait` in argument or return position on required methods. This is
   why `stream` takes `&(dyn Fn(u32, &str) + Send)` and not `impl Fn` — keep
   that pattern for any new callback.
2. **New trait methods get a default implementation** whenever a backend can
   conform by doing nothing (the model: `telemetry()` defaults to `None`,
   `stream()` defaults to wrapping `infer`). A required method with no default
   forces simultaneous edits to every backend — avoid unless truly mandatory.
3. **Absence ≠ zero.** Optional data (like telemetry) returns `Option`; the
   caller must treat `None` as "not measured", never as `0`. Design new
   telemetry-like methods the same way: **pull-based snapshots** — the engine
   never learns who asked.
4. **Dynamic dispatch at the boundary only.** `dyn GlEngine` is fine per
   request. Inside an engine's hot loop, never dispatch through `dyn` per
   token or per block — use enum dispatch (`match SimdStrategy { ... }`) or
   monomorphized functions. The `glproc` kernel-bridge pattern is the
   reference: one `match` on a strategy enum selecting concrete `unsafe fn`s.
5. **Zero-cost means measured-zero.** If an abstraction claims to be free,
   the benchmark before/after must show it (see
   [`../bench-skills/measurement-discipline.md`](../bench-skills/measurement-discipline.md)).
6. Trait-level docs explain *contract*, not implementation: what the runtime
   may assume (e.g. `init` before `load_model` before `infer`; `shutdown`
   frees GPU memory).
7. Backends implement `GlEngine` and expose **nothing else** to the runtime.
   A backend-specific public API that the runtime "just this once" calls
   directly violates backend independence.

## ✅ Correct Pattern

```rust
// New optional capability: default keeps all four backends conforming.
pub trait GlEngine: Send + Sync {
    // ...existing methods...

    /// KV-cache occupancy snapshot; `None` = not tracked (NOT "empty").
    fn kv_stats(&self) -> Option<KvStats> {
        None
    }
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
pub trait GlEngine: Send + Sync {
    // ❌ generic method — GlEngine is no longer object-safe,
    //    Box<dyn GlEngine> in the runtime stops compiling:
    fn infer_with<F: Fn(u32)>(&self, input: InferInput, cb: F) -> Result<InferOutput, GlError>;
}

// ❌ per-token dyn dispatch in a hot loop:
for _ in 0..max_tokens {
    let logits = (kernel as &dyn Kernel).forward(&x); // vtable call per token
}
```

## GwenLand-Specific Notes

- `Send + Sync` on `GlEngine` is load-bearing: engines cross thread
  boundaries (TUI, server use). A backend holding non-`Sync` raw pointers
  (CUDA streams) must justify its `unsafe impl Send/Sync` with the invariant —
  see [unsafe-rules.md](unsafe-rules.md).
- The streaming default replays tokens *after* inference. It exists so
  backends work before implementing true streaming — do not "fix" a backend's
  latency by making the default smarter; implement `stream` in that backend.

## Related Skills

- [../architecture-skills/glcore-rules.md](../architecture-skills/glcore-rules.md)
- [../architecture-skills/backend-independence.md](../architecture-skills/backend-independence.md)
- [memory-safety.md](memory-safety.md)
