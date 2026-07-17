# Unsafe Rules

> **Domain:** rust-skills
> **Applies to:** `glproc` (SIMD), `glcuda`/`glvulkan`/`glmetal` (FFI), `glcore` (mmap views)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] My use of `unsafe` falls into an **allowed category** (Rule 1) — otherwise stop.
- [ ] I can state the exact invariant that makes the block sound, in one or two sentences, and I will write it as a `// SAFETY:` comment.
- [ ] A safe wrapper will be the public face of this code; callers never see `unsafe`.

## Context

GwenLand needs `unsafe` in exactly three places: CPU SIMD intrinsics, GPU
driver FFI, and reinterpreting mmap'd bytes as typed slices. Everything else
in a pure-Rust engine can and must be safe. Unsound `unsafe` here is
security-relevant — the bytes being reinterpreted come from **untrusted model
files** (this is also the #1 in-scope class in [`SECURITY.md`](../../SECURITY.md)).

## Rules

1. **`unsafe` is allowed ONLY for:**
   - SIMD intrinsics (`core::arch::x86_64`) behind CPU-feature dispatch;
   - FFI to the CUDA/Vulkan/Metal drivers (calls, raw handles, `Send`/`Sync`
     impls for wrapper types);
   - checked reinterpretation of mmap bytes into `#[repr(C)]`/POD slices.
   Anything else — including "the bounds check was slow" — is not a category,
   it's a proposal that needs profiling proof first.
2. **Every `unsafe` block/fn carries a `// SAFETY:` comment** stating the
   invariant it relies on and *where that invariant is established*. No
   comment, no merge.
3. **`#[target_feature(enable = "avx2")]` functions are `unsafe` to call** —
   the caller's obligation is proving the CPU has the feature. In GwenLand
   that proof is the `SimdStrategy` runtime detection: intrinsic kernels are
   reachable **only** through the strategy `match`. Never call an
   AVX2/AVX-512 kernel directly.
4. **Safe wrappers, always.** The `pub` API of a kernel module takes safe
   slices, validates lengths/alignment once, then enters `unsafe`. Callers of
   the wrapper must not be able to cause UB with any input.
5. **Validate before trusting model bytes**: length, alignment, and bounds
   are checked with `GlError` returns *before* the cast. `unsafe` never does
   the checking itself.
6. **No `unsafe` for performance without a measured win** on the production
   path (not a probe — see
   [`measurement-discipline.md`](../bench-skills/measurement-discipline.md)).
   "Removes a bounds check" without a number is a rejected diff.
7. FFI wrapper types get `unsafe impl Send`/`Sync` only with a SAFETY comment
   explaining the driver's actual threading contract.
8. Keep `unsafe` blocks minimal — wrap the intrinsic call, not the whole
   function body.

## ✅ Correct Pattern

```rust
/// Safe wrapper: validates once, then dispatches to the proven-available ISA.
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    match SimdStrategy::detect() {
        // SAFETY: SimdStrategy::detect() returned Avx2, which guarantees
        // is_x86_feature_detected!("avx2") held on this CPU; lengths are
        // equal per the assert above.
        SimdStrategy::Avx2 => unsafe { avx2::dot_f32(a, b) },
        SimdStrategy::Scalar => scalar::dot_f32(a, b),
    }
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ no SAFETY comment, no feature proof — UB (SIGILL) on non-AVX2 CPUs:
let s = unsafe { avx2::dot_f32(a, b) };

// ❌ unsafe doing the validation "inside": a short malicious GGUF tensor
// walks past the end of the mmap:
unsafe {
    let w = std::slice::from_raw_parts(ptr, expected_len); // len never checked
}

// ❌ "performance" unsafe with no measurement:
let x = unsafe { *values.get_unchecked(i) }; // saved nothing, risked UB
```

## GwenLand-Specific Notes

- AVX-512 has its own policy: the strategy layer deliberately declines
  AVX-512F on Tiger Lake even though the CPU reports it (thermal throttling).
  Do not "enable what the CPU supports" — the dispatch table is policy, not
  detection. See [`../cpu-skills/avx2-simd.md`](../cpu-skills/avx2-simd.md).
- glcuda's FFI resolves driver symbols at runtime (`dlopen`/`LoadLibrary`).
  Missing driver = `available: false` + fallback, never a panic — see
  [`../cuda-skills/dynamic-loading.md`](../cuda-skills/dynamic-loading.md).

## Related Skills

- [memory-safety.md](memory-safety.md)
- [../cpu-skills/avx2-simd.md](../cpu-skills/avx2-simd.md)
- [../cuda-skills/dynamic-loading.md](../cuda-skills/dynamic-loading.md)
