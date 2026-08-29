# AVX2 SIMD

> **Domain:** cpu-skills
> **Applies to:** `glproc` — [`kernels/`](../../glproc/src/kernels/), [`attention.rs`](../../glproc/src/attention.rs), `simd_strategy`
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I read [`../before-coding/read-architecture-first.md`](../before-coding/read-architecture-first.md) and [rejected-optimizations.md](rejected-optimizations.md).
- [ ] Reference CPU: Tiger Lake **i3-1115G4** (2P/4T). AVX2 = yes. **AVX-512F = policy-banned. AVX-512VNNI-512 = declined by the engine on purpose.**
- [ ] My kernel will be reachable only through the `SimdStrategy` dispatch, with a scalar reference implementation alongside.

## Context

glproc's hot kernels (matmul/dot, dequant, softmax, V-accumulation) are
AVX2 intrinsics selected at runtime by `SimdStrategy`. The ISA policy is the
part agents get wrong: Tiger Lake *reports* AVX-512, and the engine still says
no — 512-bit execution downclocks/heats this 15 W part, and 256-bit VNNI
covers the integer-dot need. The dispatch table encodes measured policy, not
CPU capability.

## Rules

1. **AVX2 (256-bit) is the ceiling on this tier.** Never emit AVX-512F.
   The engine detecting-yet-declining AVX-512 is intentional — do not
   "fix" the strategy selection to use it.
2. **Every SIMD kernel has a scalar twin** — the correctness reference the
   tests compare against, and the `SimdStrategy::Scalar` fallback. New
   intrinsic kernel without a scalar twin = incomplete PR.
3. **V-accumulation pattern** (this bought +35 % decode): accumulate in
   registers across the loop (multiple independent accumulators to hide FMA
   latency), reduce **once** at the end. Never accumulate through memory,
   never horizontal-reduce inside the loop.
4. **Horizontal reduction is the closing move only:** `haddps`-style /
   shuffle-and-add sequences at loop exit. A `hadd` inside the hot loop is a
   review rejection.
5. **Dispatch, don't detect, in kernels.** Kernels are `unsafe fn` +
   `#[target_feature(enable = "avx2")]`, called only from the `SimdStrategy`
   match ([`../rust-skills/unsafe-rules.md`](../rust-skills/unsafe-rules.md)).
   No `is_x86_feature_detected!` inside kernel bodies.
6. **Handle ragged tails explicitly.** Real dims (896, 1536) aren't always
   multiples of 8 f32 lanes × unroll factor; every kernel has a tail path,
   and the tests include a ragged shape.
7. **Alignment is not assumed:** mmap'd GGUF tensor data has format-defined
   alignment — use unaligned loads (`loadu`) unless a repack step guarantees
   alignment, and document that guarantee where you rely on it.
8. Changes claiming speed must show production `glbench` numbers on the
   i3-1115G4 class, per [`../bench-skills/measurement-discipline.md`](../bench-skills/measurement-discipline.md).

## ✅ Correct Pattern

```rust
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    // 4 independent accumulators hide FMA latency (V-accumulation).
    let (mut s0, mut s1, mut s2, mut s3) = (zero(), zero(), zero(), zero());
    for chunk in 0..n / 32 {
        s0 = fmadd(load(a, chunk, 0), load(b, chunk, 0), s0);
        s1 = fmadd(load(a, chunk, 8), load(b, chunk, 8), s1);
        s2 = fmadd(load(a, chunk, 16), load(b, chunk, 16), s2);
        s3 = fmadd(load(a, chunk, 24), load(b, chunk, 24), s3);
    }
    let sum = horizontal_reduce(add(add(s0, s1), add(s2, s3))); // once, at exit
    sum + scalar_tail(a, b, n - n % 32)                          // ragged tail
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ AVX-512 "because the CPU supports it" — thermal throttle on Tiger Lake:
#[target_feature(enable = "avx512f")]
unsafe fn dot_f32_512(...) { ... }

// ❌ single accumulator + horizontal add inside the loop:
for chunk in chunks {
    acc = fmadd(a, b, acc);
    total += hsum(acc);      // serializes on FMA latency AND shuffles per iter
}
```

## GwenLand-Specific Notes

- The threaded-attention win came from SIMD V-accumulation + vectorized
  softmax + threading across heads — but threading heads has its own gate
  (needs ≥ 4 KV heads; see [threading-model.md](threading-model.md)).
- The decode "share %" of a kernel is not comparable across models — compare
  GMAC/s, not percentage points, when judging a kernel across configs.

## Related Skills

- [rejected-optimizations.md](rejected-optimizations.md)
- [threading-model.md](threading-model.md)
- [../rust-skills/unsafe-rules.md](../rust-skills/unsafe-rules.md)
