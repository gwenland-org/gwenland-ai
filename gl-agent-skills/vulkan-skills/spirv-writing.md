# SPIR-V Compute Shader Authoring

> **Domain:** vulkan-skills
> **Applies to:** `glvulkan` — ⚠️ currently a **stub** ([`glvulkan/src/lib.rs`](../../glvulkan/src/lib.rs)); these rules govern its bring-up
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know glvulkan is a stub today: every `GlEngine` method returns `not_implemented`. Bring-up is an `engine/glvulkan-…` branch effort, not a drive-by patch.
- [ ] I have read the glcuda template first — [`../cuda-skills/`](../cuda-skills/) — because glvulkan copies its shape: loader-based API access, hand-authored kernels shipped as data, parity vs glproc.
- [ ] I will not add heavyweight shader-toolchain dependencies to the build.

## Context

glvulkan is the cross-vendor GPU backend (NVIDIA / AMD / Intel / ARM Mali).
The GwenLand way is the same as glcuda's: kernels are **artifacts committed to
the repo** (SPIR-V binaries or assembly, with readable source alongside), the
Vulkan loader is opened at runtime, and every kernel is validated against the
glproc oracle. The zero-dependency build promise applies here exactly as it
does for CUDA: `cargo build` on a machine with no Vulkan SDK must succeed.

## Rules

1. **No build-time shader compilation.** Committed SPIR-V blobs (plus their
   human-readable source — WGSL/GLSL/`.spvasm` — in the same directory) are
   the source of truth. A `build.rs` invoking `glslc`/`naga` is forbidden;
   regeneration is a documented manual step in the kernel directory's README.
2. **One kernel = one entry point = one documented dispatch geometry**
   (workgroup size, push-constant block, bindings) in a header comment of the
   source file — the SPIR-V twin of the PTX header rule.
3. **Workgroup size is explicit** (`local_size_x` etc.), a multiple of the
   subgroup size where used, and never assumed to be 32 — subgroups are 32 on
   NVIDIA but 64 on AMD GCN and vary on mobile. Anything subgroup-dependent
   must read the size at runtime or avoid subgroup ops (see
   [portability.md](portability.md)).
4. **Buffer layouts are `std430`**, explicit, and shared with the Rust side
   via one commented struct definition on each side. No `std140` for storage
   buffers, no implicit padding assumptions.
5. **Floating-point behavior:** no `fast-math`-style relaxations that change
   numerics silently; precision choices must keep kernels inside the same
   per-op tolerances used for glcuda parity.
6. **Every kernel lands with:** glproc parity test (per-op tolerance, ragged
   shapes like dim 896 included), a bench entry, and a loud SKIP when no
   Vulkan device is present — the glcuda test pattern verbatim.

## ✅ Correct Pattern

```text
glvulkan/src/kernels/
├── rmsnorm.comp        # readable source, header documents bindings + geometry
├── rmsnorm.spv         # committed artifact actually loaded at runtime
└── README.md           # exact regeneration command + toolchain version
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ build.rs that shells out to glslc — build now requires the Vulkan SDK,
   breaking "builds on a machine with no GPU vendor anything".

❌ A kernel that hardcodes subgroup size 32 — wrong on AMD/Mali.

❌ Shipping only .spv with no readable source — unreviewable, unmaintainable.
```

## GwenLand-Specific Notes

- Decode will be bandwidth-bound here too. The glcuda experience transfers:
  design decode kernels around bytes moved, prefill kernels around GEMM
  throughput ([`../cuda-skills/kernel-design.md`](../cuda-skills/kernel-design.md)).
- The stub exists *so that* the runtime fallback chain works without
  conditional compilation — bring-up must preserve that property at every
  intermediate commit.

## Related Skills

- [descriptor-sets.md](descriptor-sets.md)
- [portability.md](portability.md)
- [../cuda-skills/ptx-writing.md](../cuda-skills/ptx-writing.md)
