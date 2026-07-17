# PTX Writing

> **Domain:** cuda-skills
> **Applies to:** `glcuda` — [`glcuda/src/kernels/glcuda.ptx`](../../glcuda/src/kernels/glcuda.ptx), [`glcuda_sm75.ptx`](../../glcuda/src/kernels/glcuda_sm75.ptx)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I read [`architecture/ArchGLML_X2.md`](../../architecture/ArchGLML_X2.md) — the glcuda ground truth.
- [ ] I know which PTX file I'm editing: `glcuda.ptx` (baseline, any sm) vs `glcuda_sm75.ptx` (Turing tensor-core kernels).
- [ ] My editor will save **pure ASCII with LF line endings** — no smart quotes, no em-dashes, no CRLF.

## Context

glcuda's kernels are **hand-authored PTX** — there is no `nvcc`, no CUDA C to
regenerate from. The PTX file *is* the source. `ptxas` (invoked by the driver
JIT at load) is unforgiving and its errors are cryptic, so most classic
kernel-dev safety nets don't exist here: discipline in the file replaces them.

## Rules

1. **Pure ASCII, LF endings, always.** A single stray em-dash in a comment
   makes `ptxas` reject the module before parsing one instruction. This has
   actually happened. If your tooling "prettifies" quotes/dashes, fix the
   tooling before editing PTX.
2. **Unique register/variable names per scope.** `ptxas` accepts some
   shadowing that then miscompiles or fails late; a duplicate declaration in
   one kernel body cost us a debugging session. Prefix locals by kernel
   section (`%f_acc0`, `%r_col`, …) and never re-declare a name in the same
   function.
3. **No duplicate declarations in one PTX function.** Grep the function body
   for the name before adding it.
4. **Every kernel is host-tested before device-trusted:** first the CPU-side
   reference (`glproc` implementation), then the parity test
   (`cargo test -p glcuda --test parity -- --test-threads=1`) on real
   hardware, within the per-op tolerance from the architecture spec.
5. **Coalesced memory access is the default layout assumption.** Adjacent
   threads read adjacent addresses; strided access needs a comment justifying
   it. Decode is bandwidth-bound (measured 88 % of T4 bandwidth) — an
   uncoalesced weight read shows up immediately as lost tok/s.
6. **Warp size = 32. Always.** Shuffle reductions, predication, and tile
   shapes assume 32 lanes; do not parameterize it away.
7. **Every kernel gets a header comment**: what it computes, expected launch
   geometry (block/grid), which buffers it reads/writes, and its tolerance
   class.
8. **Target the file's floor, not your card.** `glcuda.ptx` must stay loadable
   on the project's minimum sm; anything needing `sm_75+` instructions
   (`mma.sync`, dp4a variants) lives in `glcuda_sm75.ptx` behind the runtime
   capability check.

## ✅ Correct Pattern

```ptx
// gl_rmsnorm_f32: y = x * rsqrt(mean(x^2)+eps) * w
// launch: 1 block / row, 256 threads; reads x,w; writes y (f32)
// tolerance class: TOL_NORM (see ArchGLML_X2.md)
.visible .entry gl_rmsnorm_f32(...)
{
    .reg .f32 %f_acc<4>;      // accumulator lanes — unique prefix
    .reg .f32 %f_val;
    // coalesced: tid.x walks contiguous f32s of the row
    ...
}
```

## ❌ Anti-Pattern (Never Do This)

```ptx
// “fast path” — ❌ smart quotes + em-dash: ptxas rejects the whole module
.reg .f32 %f1;
...
.reg .f32 %f1;        // ❌ duplicate declaration in the same function body
ld.global.f32 %f2, [%rd_base + %r_tid * 4096];  // ❌ stride-4096 per lane,
                                                // uncoalesced, no comment
```

## GwenLand-Specific Notes

- The PTX is loaded and JIT-compiled at runtime by the driver — a PTX error
  surfaces as a module-load `GlError::Engine`, not a build failure. CI cannot
  catch it without a GPU; that's why host-first testing (Rule 4) is mandatory.
- Kernel names are `gl_`-prefixed and resolved by string from the Rust side
  (`glcuda/src/kernels/`); renaming a kernel means updating the Rust lookup in
  the same commit.

## Related Skills

- [kernel-design.md](kernel-design.md)
- [tensor-cores.md](tensor-cores.md)
- [../before-coding/check-existing-tests.md](../before-coding/check-existing-tests.md)
