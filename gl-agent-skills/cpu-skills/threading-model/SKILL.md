# Threading Model

> **Domain:** cpu-skills
> **Applies to:** `glproc` — [`threading.rs`](../../glproc/src/threading.rs), threaded kernels, load-time parallelism in [`loader.rs`](../../glproc/src/loader.rs)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the measured facts on the i3-1115G4 (2 physical / 4 logical): **compute thread knee = 3** — not 2 ("physical cores only"), not 4 ("use all logical").
- [ ] I know `ThreadPool::run` is **NOT reentrant** — no nested pool dispatch, ever.
- [ ] I am not about to "correct" the load-path worker count in `loader.rs` (`num_cpus::get().clamp(1, 8)`, capped by layer count) — it is intentional.

## Context

Threading numbers here were all measured, and they're counter-intuitive twice
over. Compute kernels peak at **3 threads** on a 2P/4T machine: the fourth
logical thread fights its sibling for ports on a saturated core, while 2
threads leave one core's second issue slot idle. Meanwhile the **load path
legitimately uses more workers** — mmap page-in + requantization is
memory-stall-heavy, exactly where SMT pays. Two different jobs, two different
correct answers; agents keep trying to unify them, and the unification is
always a regression.

## Rules

1. **Compute-kernel parallelism sizes to the measured knee (3 on the
   reference box)**, derived from the machine profile — never hardcode "all
   cores" or "physical cores" logic into kernels.
2. **`ThreadPool::run` is not reentrant.** A worker must never call back into
   the pool (this is why expert-per-worker MoE threading was *not* done).
   Nesting deadlocks or serializes; if a kernel needs two parallel levels,
   flatten the work items instead.
3. **The loader's worker count stays.** Parallel layer loading
   (mmap + requant) benefits from SMT; its `num_cpus`-based sizing is
   deliberate and separately tuned from compute threading. Don't fold it
   into the compute-knee policy.
4. **Threaded attention has a model gate: ≥ 4 KV heads.** Threading across
   KV heads *loses* on models with 2 KV heads (Qwen2.5-0.5B). The gate is
   config-driven — never remove it because a bigger model showed a win.
5. **Partition work statically per token step** (contiguous row ranges per
   worker). No work-stealing, no per-token spawn: threads are pooled and
   persistent, work assignment is deterministic — determinism is part of the
   parity story.
6. **No thread does allocation in the token loop** — scratch is
   per-worker, pre-allocated ([`../rust-skills/memory-safety.md`](../rust-skills/memory-safety.md)).
7. **Topology experiments require production A/B on the reference box.**
   The "fix B topology threading" change looked right and cost **-23 %
   decode** — it is on the rejected list. Read
   [rejected-optimizations.md](rejected-optimizations.md) before proposing
   affinity/topology cleverness.

## ✅ Correct Pattern

```rust
// Compute: knee-sized, static split, pooled threads.
let n = self.pool.compute_threads();          // = measured knee (3 on 2P/4T)
self.pool.run(n, |worker| {
    let rows = row_range_for(worker, n, total_rows); // contiguous, deterministic
    kernel::ffn_rows(&weights, &x, &mut scratch[worker], rows);
});
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ "use everything" — 4th logical thread regresses compute on 2P/4T:
let n = num_cpus::get();
self.pool.run(n, ...);

// ❌ nested dispatch — ThreadPool::run is not reentrant:
self.pool.run(n, |w| {
    self.pool.run(2, |inner| ...); // deadlock/serialization
});

// ❌ removing the KV-head gate because a 7B model liked threading:
// 0.5B (2 KV heads) measurably loses — the gate exists for it.
```

## GwenLand-Specific Notes

- Single-core memory-level parallelism is a real ceiling: the attention
  "anomaly" was eventually explained as the **line-fill-buffer limit of one
  core walking cold, strided KV** — not a kernel bug. Some "threading wins"
  are actually MLP wins (more outstanding misses), which is why they don't
  appear on warm data. Diagnose before optimizing.
- Thread counts and knees are *this hardware tier's* numbers. On another
  machine, re-measure; don't port the constants, port the method.

## Related Skills

- [rejected-optimizations.md](rejected-optimizations.md)
- [memory-bandwidth.md](memory-bandwidth.md)
- [avx2-simd.md](avx2-simd.md)
