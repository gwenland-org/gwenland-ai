# Descriptor Sets & Buffer Binding

> **Domain:** vulkan-skills
> **Applies to:** `glvulkan` (bring-up rules; the crate is currently a stub)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I read [spirv-writing.md](spirv-writing.md) — binding layouts are declared there, in the shader source headers.
- [ ] I know the memory model this must mirror: glcuda's bump arena — **allocate at load, bind forever, zero descriptor churn per token**.

## Context

Descriptor management is where Vulkan backends usually bleed CPU time and
complexity. GwenLand's inference shape makes the simple thing correct: the set
of buffers is *static after model load* (weights, KV cache, a fixed scratch
set), so descriptor sets can be built once and reused for the whole session.

## Rules

1. **Static binding model:** all storage buffers come from arena allocations
   at load time; descriptor sets are allocated and written **once** after
   load. Per-token descriptor allocation/update is forbidden — same contract
   as glcuda's zero-alloc rule.
2. **Per-token variability goes through push constants** (step index, seq
   length, scales pointer offsets), not through rebinding buffers. Keep the
   push-constant block ≤ 128 bytes (the portable guaranteed minimum).
3. **One descriptor-set layout per kernel family**, documented next to the
   shader source; set 0 = weights (read-only), set 1 = activations/scratch,
   set 2 = KV cache. Don't invent per-kernel ad-hoc layouts.
4. **Read-only weight buffers are declared `readonly`** in the shader and
   non-writable in the layout — let the driver and validation layers enforce
   what the architecture promises.
5. **Descriptor pools are sized exactly at load** from the model config; pool
   exhaustion at runtime is a bug in the sizing math, never a reason to grow
   dynamically mid-session.
6. **Validation layers on in tests, off in production.** The GPU test harness
   enables `VK_LAYER_KHRONOS_validation` and fails on validation errors;
   production init must not require the layers to be installed.

## ✅ Correct Pattern

```text
Load time:  arena.alloc all buffers → build descriptor sets (once)
Per token:  vkCmdBindDescriptorSets(same sets) + vkCmdPushConstants(step data)
            → dispatch. No vkAllocateDescriptorSets, no vkUpdateDescriptorSets.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ vkUpdateDescriptorSets inside the decode loop to point at "this token's"
   buffer — per-token state belongs in push constants / fixed ring offsets.

❌ VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT + free/realloc churn —
   the buffer population is static; churn is pure overhead and fragmentation.

❌ Requiring validation layers at production init (fails on user machines
   without the SDK).
```

## GwenLand-Specific Notes

- KV cache is pre-sized from context length and cursor-indexed (the
  project-wide policy) — the descriptor for it never changes; the cursor is a
  push constant.
- If a future model config genuinely needs a buffer count that varies (MoE
  expert sets), that design goes through an `architecture/` spec first, like
  every allocation-model change.

## Related Skills

- [spirv-writing.md](spirv-writing.md)
- [pipeline-barriers.md](pipeline-barriers.md)
- [../cuda-skills/memory-management.md](../cuda-skills/memory-management.md)
