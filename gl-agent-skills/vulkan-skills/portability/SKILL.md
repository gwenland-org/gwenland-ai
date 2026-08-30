# Vulkan Portability (NVIDIA / AMD / Intel / ARM Mali)

> **Domain:** vulkan-skills
> **Applies to:** `glvulkan` (bring-up rules; the crate is currently a stub)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I remember why glvulkan exists: it is the **cross-vendor** backend. A kernel that only works on NVIDIA belongs in glcuda; NVIDIA-only Vulkan is pointless here.
- [ ] Anything device-dependent (subgroup size, shared-memory size, timestamp support) is queried at init, never assumed.

## Context

glvulkan's entire reason to exist is hardware GwenLand can't otherwise reach:
AMD and Intel GPUs, and — matching the project's modest-hardware mission —
integrated GPUs and ARM Mali. That inverts the usual priority: the portable
baseline path is the product; vendor-specific fast paths are optional
extensions that must never become load-bearing.

## Rules

1. **Baseline device target:** Vulkan 1.2 + compute queue + `std430` storage
   buffers, nothing else required. Every optional feature
   (subgroup ops, shader float16/int8, timestamps) is capability-queried and
   has a working fallback path.
2. **Subgroup size is a runtime value.** 32 on NVIDIA/Intel, 64 on AMD GCN
   (32/64 on RDNA), variable on Mali. Kernels either use
   `VK_EXT_subgroup_size_control`-style explicit sizing with a portable
   fallback, or avoid subgroup ops entirely.
3. **Memory-type selection is explicit per vendor topology:** discrete GPUs
   (DEVICE_LOCAL + staging uploads) vs UMA/integrated (HOST_VISIBLE |
   DEVICE_LOCAL, zero-copy). On UMA machines — the modest-hardware target —
   do **not** copy weights into a "device" heap that is the same physical
   RAM; that doubles the footprint the mmap loader exists to avoid.
4. **Workgroup shared-memory usage stays within the portable minimum**
   (16 KiB guaranteed; query for more). A tile size tuned to 48 KiB of
   NVIDIA smem is a vendor path, not the baseline.
5. **No vendor-ID branching for correctness.** Branch on *queried
   capabilities*, never `vendorID == 0x10DE`. Vendor-ID checks are allowed
   only for known-driver-bug workarounds, each with a link/comment.
6. **Parity on every vendor you can reach.** A kernel is not "done" when it
   passes on one card; CI-reachable vendors + the loud-SKIP pattern document
   exactly which vendors a change was actually validated on — state it in
   the PR.
7. **Float behavior differs across vendors** (FMA contraction, flush-to-zero
   on some mobile parts). Per-op tolerances are shared with the other GPU
   backends; if a vendor needs a wider tolerance, that's a finding to raise,
   not a constant to quietly widen.

## ✅ Correct Pattern

```rust
let caps = device.query_capabilities();
let kernel = if caps.subgroup_size_control && caps.subgroup_arithmetic {
    kernels.reduce_subgroup(caps.subgroup_size) // fast path, sized at runtime
} else {
    kernels.reduce_shared_memory()              // portable baseline, always present
};
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ gl_SubgroupSize assumed 32 in shader math       → wrong sums on AMD.
❌ Staging-copy of all weights on an Intel iGPU     → doubles RAM use on UMA.
❌ if vendor == NVIDIA { correct path } else { … }  → capability queries exist.
❌ "Validated on my RTX card" as the only test note for a portability backend.
```

## GwenLand-Specific Notes

- Mali and Intel iGPUs are closer to the project's mission (8 GB laptop) than
  a discrete card is — treat integrated-GPU behavior as first-class when
  making design trade-offs, not as an afterthought.
- The fallback chain protects users, not kernels: a glvulkan init failure on
  any vendor must land softly in glproc
  ([`../architecture-skills/fallback-chain.md`](../architecture-skills/fallback-chain.md)).

## Related Skills

- [spirv-writing.md](spirv-writing.md)
- [pipeline-barriers.md](pipeline-barriers.md)
- [../architecture-skills/fallback-chain.md](../architecture-skills/fallback-chain.md)
