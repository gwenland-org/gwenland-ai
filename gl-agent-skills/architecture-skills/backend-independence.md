# Backend Independence

> **Domain:** architecture-skills
> **Applies to:** `glproc`, `glcuda`, `glvulkan`, `glmetal` (and any future engine)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] My change keeps every backend's `Cargo.toml` free of other backends.
- [ ] Anything I want to "share" between two backends is going through glcore as *format/type knowledge*, not as compute.
- [ ] Test-only cross-references are in the right direction (GPU crates may dev-depend on glproc as the parity oracle — production code may not).

## Context

Each backend is a complete, standalone engine: it implements `GlEngine`, owns
its kernels, its memory model, and its numerics. That's what makes the stack
tractable — an engine can be added, rewritten, or deleted without touching the
others, and a bug in one cannot corrupt another. The price is deliberate
duplication: four engines will each have their own matmul. That duplication
is a feature; "DRY across backends" is the standing temptation this skill
exists to kill.

## Rules

1. **A backend never imports another backend.** Not code, not constants, not
   "just the layout struct" — if two backends need it, it's format/type
   knowledge and moves to glcore ([glcore-rules.md](glcore-rules.md)).
2. **All cross-engine interaction goes through glcore's runtime.** There is
   no engine-to-engine handoff, shared state, or "ask the CPU engine to do
   this bit" at runtime. An engine that can't do an op reports itself
   unavailable/unsupported; the *runtime* decides what runs where.
3. **The one sanctioned exception:** GPU crates use `glproc` as the **parity
   oracle in tests** (dev-dependency). Production `[dependencies]` on another
   backend is forbidden; a `[dev-dependencies]` on glproc for validation is
   the pattern.
4. **Duplication across backends is intentional.** Each engine's kernel is
   tuned to its hardware model (AVX2 vs PTX vs SPIR-V) and validated
   independently. Do not build "shared kernel abstractions" spanning
   engines — that recreates the framework-coupling this project was built to
   escape.
5. **Backends expose `GlEngine` and nothing else** to the rest of the
   system. Public helper APIs consumed by the CLI/runtime directly are a
   boundary leak.
6. **Feature flags don't bend the rules:** no
   `#[cfg(feature = "cuda")] use glcuda::…` inside another engine.

## ✅ Correct Pattern

```toml
# glcuda/Cargo.toml
[dependencies]
glcore = { path = "../glcore" }    # the only production dependency direction

[dev-dependencies]
glproc = { path = "../glproc" }    # parity oracle for tests — sanctioned
```

## ❌ Anti-Pattern (Never Do This)

```toml
# ❌ glvulkan/Cargo.toml
[dependencies]
glcuda = { path = "../glcuda" }    # "to reuse its repack logic"
```

```rust
// ❌ runtime-level engine chimera:
if !cuda.supports(op) {
    cpu_engine.run_op(op)?;  // engines don't hand off ops; the runtime
                             // picks ONE engine per request
}
```

## GwenLand-Specific Notes

- Conclusions don't transfer even when code could: CPU decode is FFN-bound,
  GPU prefill was attention-bound; Q4_K native compute lost on CPU while
  INT8 MMA won on GPU prefill. Independent engines exist precisely because
  the *right answer differs per hardware*.
- glmetal/glvulkan bring-up must copy glcuda's *shape* (loader-based API,
  arena memory, parity ladder) by **pattern**, not by import — see
  [`../vulkan-skills/spirv-writing.md`](../vulkan-skills/spirv-writing.md).

## Related Skills

- [glcore-rules.md](glcore-rules.md)
- [fallback-chain.md](fallback-chain.md)
- [../rust-skills/testing-standards.md](../rust-skills/testing-standards.md)
