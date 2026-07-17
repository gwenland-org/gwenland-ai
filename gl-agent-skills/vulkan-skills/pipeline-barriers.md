# Pipeline Barriers & Synchronization

> **Domain:** vulkan-skills
> **Applies to:** `glvulkan` (bring-up rules; the crate is currently a stub)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the decode dependency structure: a *chain* — each kernel reads the previous kernel's output. There is no hidden parallelism to unlock at batch=1.
- [ ] I have read [`../cuda-skills/cuda-graphs.md`](../cuda-skills/cuda-graphs.md): on CUDA, the measured bottleneck was inter-kernel dependency serialization, not launch overhead. The same physics applies here.

## Context

Vulkan gives you explicit synchronization and therefore explicit ways to get
it wrong in both directions: over-synchronize (full-pipeline barriers between
every dispatch → dead GPU time) or under-synchronize (missing hazard →
wrong numbers that only parity tests catch). GwenLand's decode is a linear
chain of small dispatches, so the sync story must be boring, correct, and
recorded once — not re-derived per dispatch at runtime.

## Rules

1. **Compute-to-compute hazards use precise barriers:**
   `VkMemoryBarrier2` with `SHADER_STORAGE_WRITE → SHADER_STORAGE_READ` on
   the compute stage — not `ALL_COMMANDS`/`MEMORY_READ|WRITE` sledgehammers.
   Use Synchronization2; don't mix the legacy API in the same code path.
2. **Batch the whole token step into one command buffer**, pre-recorded where
   possible (fixed shapes + static descriptors make this legal — the Vulkan
   analog of CUDA-graph replay). Re-recording per token is the fallback, not
   the design.
3. **No queue ownership transfers in v1:** one compute queue, one family.
   Async transfer queues, cross-queue semaphores, and timeline-semaphore
   pipelining are extensions that need measured motivation first.
4. **Host readback happens once per token** (logits or sampled id), through a
   persistently mapped host-visible buffer + the appropriate
   `HOST_READ` barrier; no `vkQueueWaitIdle` in the token loop.
5. **`vkQueueWaitIdle`/`vkDeviceWaitIdle` are shutdown/debug tools only.**
   In-loop waits serialize CPU and GPU and hide the real timing — production
   waits use fences/timeline semaphores on the step granularity.
6. **Every barrier carries a comment naming the hazard** (which buffer, which
   producer → consumer). An uncommented barrier is treated as cargo cult and
   will be challenged in review.
7. **Correctness arbiter is parity, not vibes:** a sync bug shows up as
   nondeterministic parity failures. Any intermittent parity failure on
   glvulkan is treated as a missing/wrong barrier until proven otherwise.

## ✅ Correct Pattern

```text
Record once (per model config):
  for each layer kernel K1→K2→…→Kn:
      vkCmdDispatch(Ki)
      // hazard: Ki writes activations[buf A] read by Ki+1
      MemoryBarrier2(COMPUTE_SHADER, STORAGE_WRITE → COMPUTE_SHADER, STORAGE_READ)
Per token:
  update push constants → submit prerecorded buffer → wait fence (step level)
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ vkCmdPipelineBarrier(ALL_COMMANDS → ALL_COMMANDS) after every dispatch
   "to be safe" — over-sync; measurable dead time on a chain of small kernels.

❌ vkQueueWaitIdle after each dispatch to "make timing simple".

❌ Omitting the barrier between FFN write and attention read because "it
   worked on my NVIDIA card" — caches differ per vendor; Mali/AMD will
   produce wrong numbers, intermittently.
```

## GwenLand-Specific Notes

- Do not import CUDA's implicit-stream mental model: Vulkan dispatches on one
  queue may overlap unless you say otherwise. On glcuda, ordering came for
  free on a stream; here every producer→consumer edge needs its barrier.
- Expect the same endgame as glcuda: once sync is minimal and correct, the
  remaining dependency-chain dead time is attacked by **kernel fusion**, not
  by cleverer barriers.

## Related Skills

- [descriptor-sets.md](descriptor-sets.md)
- [spirv-writing.md](spirv-writing.md)
- [../cuda-skills/cuda-graphs.md](../cuda-skills/cuda-graphs.md)
