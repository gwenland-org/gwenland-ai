# Gate Integration — Execution-Policy Gates

> **Domain:** architecture-skills
> **Applies to:** runtime engine selection, kernel/strategy dispatch in every engine
> **Last updated:** 2026-07-17
>
> ⚠️ **Scope note:** there is no `GATE.md` spec in this repository today. This
> skill documents the gating *pattern* as it actually exists in the code —
> the measured-policy gates listed below. If a formal GATE algorithm spec
> lands later, it supersedes this file and this file must be updated to
> reference it, not paraphrase it.

## BEFORE YOU START

- [ ] I understand what a "gate" is here: a **runtime decision point that chooses an execution path from measured policy**, not from capability alone.
- [ ] I am not removing/bypassing an existing gate because "the hardware supports more".
- [ ] Any new gate I add is driven by config/measurement, logged, and testable on hardware that takes either branch.

## Context

GwenLand is full of places where the naive rule "use the best thing the
hardware reports" is measurably wrong. The codebase's answer is explicit
gates: small, named decision points where *policy* (built from measurements)
overrides *capability* (what the hardware claims). Gates are the executable
form of the project's knowledge base — deleting one deletes a measurement.

## Rules

1. **The existing gates and their reasons — never bypass without sign-off:**

   | Gate | Decision | Why (measured) |
   |------|----------|----------------|
   | Engine selection | fallback chain order, availability | see [fallback-chain.md](fallback-chain.md) |
   | `SimdStrategy` | AVX2 yes; **AVX-512 detected-but-declined**; scalar floor | 512-bit thermal/downclock loss on the reference tier |
   | Threaded attention | threads across KV heads **only if ≥ 4 KV heads** | 2-KV-head models measurably lose |
   | Compute thread count | knee = 3 on 2P/4T, not all cores | 4th logical thread regresses compute |
   | Quant path | Q4_K repacked → Q8_0 at load; no native Q4_K compute | native Q4_K −33 % |
   | PTX module tier | `glcuda_sm75.ptx` only on sm_75+ capability query | baseline must load on older sm |
   | CUDA graph APIs | used when driver has the symbols, else per-kernel launch | old-driver degradation, not failure |

2. **Gates decide from config + measurement, never from vibes:** each gate
   reads model config (KV heads, dims), machine profile (ISA, driver,
   capability queries), or a measured constant — and its threshold cites its
   source in a comment.
3. **Every gate decision is observable:** log which branch was taken and
   why, and expose it via telemetry where relevant — glbench sessions must
   be able to record the taken path, or A/B results become uninterpretable.
4. **Both branches stay tested.** A gate whose "else" branch bit-rots is a
   time bomb; the loud-SKIP pattern covers hardware you can't reach.
5. **Gates are cheap and out of the hot loop:** decide at init/load/step
   granularity, never per element. A per-token `if` on an invariant config
   value should be hoisted.
6. **Adding a gate = adding its evidence:** the PR carries the measurement
   that justifies the threshold, and the threshold lands in the gate table
   above (update this file in the same PR).
7. **Removing or inverting a gate is a policy change**, not a cleanup:
   requires JinXSuper's sign-off plus a fresh production measurement — same
   bar as the rejected-optimizations list.

## ✅ Correct Pattern

```rust
// Gate: threaded attention needs enough KV heads to amortize thread cost.
// Measured: 2-KV-head models (Qwen2.5-0.5B) lose with threading; ≥4 wins.
// See gl-agent-skills/cpu-skills/threading-model.md.
let attn_threads = if cfg.n_kv_heads >= MIN_KV_HEADS_FOR_THREADING {
    pool.compute_threads()
} else {
    1
};
trace!("attention threading: {} ({} kv heads)", attn_threads, cfg.n_kv_heads);
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ capability-as-policy — undoes the AVX-512 measurement:
if is_x86_feature_detected!("avx512f") { use_avx512() }

// ❌ un-logged gate — invisible in benchmarks, undiagnosable in the field:
let n = if fast_path_ok { 3 } else { 1 }; // which ran? nobody knows

// ❌ magic threshold with no provenance:
if cfg.n_kv_heads >= 3 { ... } // why 3? measured where? cite or don't merge
```

## GwenLand-Specific Notes

- Gates are per **hardware tier**: the AVX-512 decline is Tiger-Lake-tier
  policy. A future tier may gate differently — by *adding* profile-keyed
  policy, never by deleting the existing tier's entry.
- The `--engine` CLI flag is a user override of the selection gate — and
  deliberately does *not* fall back silently
  ([fallback-chain.md](fallback-chain.md) Rule 6). User overrides make gates
  observable end-to-end; keep that property for any new gate you expose.

## Related Skills

- [fallback-chain.md](fallback-chain.md)
- [../cpu-skills/rejected-optimizations.md](../cpu-skills/rejected-optimizations.md)
- [../cpu-skills/threading-model.md](../cpu-skills/threading-model.md)
