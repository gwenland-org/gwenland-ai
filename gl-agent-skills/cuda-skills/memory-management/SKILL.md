# CUDA Memory Management

> **Domain:** cuda-skills
> **Applies to:** `glcuda` — [`buffer.rs`](../../glcuda/src/buffer.rs), [`kv_cache.rs`](../../glcuda/src/kv_cache.rs), [`loader.rs`](../../glcuda/src/loader.rs), [`runner.rs`](../../glcuda/src/runner.rs)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I understand the M2 contract: **zero `cuMemAlloc` after init** — all VRAM comes from the bump allocator, allocated once.
- [ ] I know the VRAM-leak test exists and runs serially (`--test-threads=1`); my change must keep it green.
- [ ] Weights stream from the mmap'd file on the host side — no full host-RAM staging copy.

## Context

glcuda's memory model is deliberately primitive: one big VRAM arena, bumped
during model load, reused forever. This is what makes decode-time behavior
predictable (no allocator jitter, no fragmentation, no async-alloc surprises)
and what lets the leak test assert an exact allocation count. "Just allocate a
temp buffer" is the most common way to silently break the M2 contract.

## Rules

1. **All device memory comes from the bump allocator at init/load time.**
   Per-request or per-token `cuMemAlloc` is forbidden. If a new kernel needs
   scratch space, size it at load (worst case over the model config) and
   reserve it in the arena.
2. **Zero device allocations after init — verified, not assumed.** The leak /
   allocation-count checks in the test suite are part of the Definition of
   Done. If your feature genuinely needs a new buffer, update the expected
   counts *and say so in the PR*, don't loosen the assertion.
3. **Buffer reuse is explicit.** Buffers are owned by the backend
   (`buffer.rs`), named, and reused across steps; aliasing two logical
   tensors onto one buffer requires a comment proving their lifetimes don't
   overlap within a step.
4. **KV cache is pre-sized from context length** at load (`kv_cache.rs`) and
   indexed by cursor — never grown mid-generation (same policy as glproc).
5. **Host→device transfers happen at load, not in the decode loop.** Decode
   reads weights already resident in VRAM. Anything that would upload per
   token belongs in prefill/load redesign, not inline.
6. **Frees happen in `shutdown()`**, completely: modules unloaded, arena
   released, context destroyed. "The process is exiting anyway" is not a free.
7. Stream-ordered allocation (`cuMemAllocAsync`) is **not** the current
   model. Introducing it is an architecture change: it needs an
   `architecture/` spec update and JinXSuper's sign-off first, because it
   invalidates the leak-count contract.

## ✅ Correct Pattern

```rust
// At load: reserve everything the forward pass will ever need.
let scratch_logits = arena.alloc::<f32>(cfg.vocab_size)?;      // once
let scratch_attn   = arena.alloc::<f32>(cfg.max_ctx * cfg.n_heads)?; // worst case

// Per token: reuse. No allocation on this path.
runner.launch_lm_head(&weights.lm_head, &hidden, &scratch_logits)?;
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ per-step device allocation — breaks the zero-alloc contract AND
// the leak test's expected counts:
fn decode_step(&self) -> Result<Token, GlError> {
    let tmp = self.device.alloc::<f32>(self.vocab)?; // every token!
    ...
}

// ❌ "temporary" host staging of the full weight file:
let all_bytes = std::fs::read(path)?; // 7 GB in host RAM; the mmap loader
                                      // exists precisely to avoid this
```

## GwenLand-Specific Notes

- Measured reality on the T4: decode is **bandwidth-bound at 88 % of memory
  bandwidth**. Memory layout changes (SoA repack in `repack.rs`, coalescing)
  are where the wins are; allocator cleverness is not — the allocator is
  intentionally boring.
- VRAM budget mirrors the RAM philosophy: fit the model, the KV cache, and a
  fixed scratch set; there is no eviction/paging story and none should be
  added casually.

## Related Skills

- [kernel-design.md](kernel-design.md)
- [cuda-graphs.md](cuda-graphs.md)
- [../rust-skills/memory-safety.md](../rust-skills/memory-safety.md)
