# Dynamic Driver Loading

> **Domain:** cuda-skills
> **Applies to:** `glcuda` — [`ffi.rs`](../../glcuda/src/ffi.rs), [`driver.rs`](../../glcuda/src/driver.rs)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I understand the invariant: **glcuda compiles and runs on machines with no NVIDIA anything** — no toolkit at build time, no driver at run time.
- [ ] I am not about to add a link-time CUDA dependency (`-lcuda`, a `cuda-sys`-style crate, a build.rs probe).
- [ ] Every new driver API I call goes through the runtime symbol table, with a missing-symbol failure path.

## Context

The whole tree builds with plain `cargo build` on a GPU-less laptop because
glcuda never links the CUDA driver: it opens `nvcuda.dll` (Windows) /
`libcuda.so.1` (Linux) at runtime with `LoadLibraryA` / `dlopen(RTLD_NOW)` and
resolves the Driver API symbols it needs. If the library or a symbol is
missing, the engine reports `available: false` and the runtime falls back to
the CPU engine. This is a hard architectural invariant, not an optimization.

## Rules

1. **Driver API only, resolved at runtime.** No CUDA *Runtime* API
   (`cudart`), no cuBLAS/cuDNN, no build-time linking of any NVIDIA library,
   ever.
2. **Adding a driver function = adding it to the FFI symbol table** in
   `ffi.rs`: typed fn pointer, resolved once at load, with the resolution
   failure handled (old drivers may lack newer symbols).
3. **Missing driver is a normal state, not an error state:** the outcome is
   `EngineSpec { available: false }` + a logged reason + fallback to glproc.
   `panic!`, `abort`, or `unwrap()` on driver absence is forbidden.
4. **Version-gate optional symbols.** If a symbol only exists in newer
   drivers (graph APIs etc.), resolve it as `Option<fn>` and degrade the
   feature, not the engine.
5. **All FFI calls stay inside `driver.rs`'s safe wrappers.** Kernel/runner
   code never touches raw `CUresult`s; wrappers convert to
   `Result<_, GlError>` with the CUDA error code and the operation name in
   the message.
6. **SAFETY comments on the FFI boundary** follow
   [`../rust-skills/unsafe-rules.md`](../rust-skills/unsafe-rules.md): each
   states the driver-contract invariant (context current on this thread,
   pointer validity, NUL-terminated names).
7. **Library names are exactly** `nvcuda.dll` and `libcuda.so.1`. Don't add
   search heuristics over user paths; if the driver isn't where drivers live,
   the machine doesn't have one.

## ✅ Correct Pattern

```rust
// Resolving an optional, newer-driver symbol:
let cu_graph_launch: Option<CuGraphLaunchFn> = table.try_resolve("cuGraphLaunch");
// Later:
match cu_graph_launch {
    Some(f) => self.launch_via_graph(f)?,
    None => self.launch_kernels_individually()?, // degrade the feature,
                                                 // not the engine
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ build-time linking — breaks "builds anywhere":
#[link(name = "cuda")]
extern "C" { fn cuInit(flags: u32) -> i32; }

// ❌ driver absence treated as a bug:
let lib = load_cuda_library().expect("CUDA driver required"); // kills CPU users

// ❌ raw FFI leaking out of driver.rs into a kernel launcher:
unsafe { (ffi::CU_LAUNCH)(kernel_handle, grid, block, params) }; // no wrapper,
                                                                 // no GlError
```

## GwenLand-Specific Notes

- This pattern is the template for **glvulkan and glmetal too**: loader-based
  API access, `available: false` on absence, fallback chain intact. See
  [`../architecture-skills/fallback-chain.md`](../architecture-skills/fallback-chain.md).
- The PTX-not-cubin choice is part of the same invariant: the driver JIT
  compiles our kernels at load, so we ship no architecture-specific binaries
  and need no toolkit (see [ptx-writing.md](ptx-writing.md)).
- Tests: host-side glcuda tests must pass on a GPU-less machine; GPU tests
  skip loudly (see [`../rust-skills/testing-standards.md`](../rust-skills/testing-standards.md)).

## Related Skills

- [../rust-skills/unsafe-rules.md](../rust-skills/unsafe-rules.md)
- [../architecture-skills/fallback-chain.md](../architecture-skills/fallback-chain.md)
- [ptx-writing.md](ptx-writing.md)
