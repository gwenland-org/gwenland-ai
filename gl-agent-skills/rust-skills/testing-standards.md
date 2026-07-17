# Testing Standards

> **Domain:** rust-skills
> **Applies to:** all crates; parity suites in `glcuda` (and future GPU backends)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I read [`../before-coding/check-existing-tests.md`](../before-coding/check-existing-tests.md) and ran the baseline.
- [ ] I know where my test belongs: in-module unit test vs `tests/` integration test vs parity suite.
- [ ] For anything numerical: I know the tolerance I'm asserting and *why* that number.

## Context

The gl-stack's correctness story is layered: unit tests pin kernel behavior,
integration tests pin end-to-end model behavior, and **parity tests pin every
GPU backend to `glproc`, the numerical ground truth**, tensor-by-tensor with
explicit per-operation tolerances. Tests here encode measured reality — several
exist purely because a specific real-world shape or file once broke us.

## Rules

1. **Placement:**
   - Unit tests: `#[cfg(test)] mod tests` in the same file as the code.
   - Integration tests: the crate's `tests/` directory (e.g.
     `glcuda/tests/parity.rs`).
   - Cross-engine parity: in the GPU crate, comparing against `glproc`.
2. **Naming states behavior, not implementation:**
   `q8_0_dot_handles_non_multiple_of_block_dim`, not `test_dot_2`. A failing
   test's name alone should say what contract broke.
3. **Numerical assertions use explicit named tolerances** (`const TOL_…`),
   one per operation class, with a comment saying where the number comes from.
   Never bare `assert_eq!` on floats; never an unexplained magic epsilon.
4. **Regression tests are permanent** and named/commented after the real case
   that motivated them (e.g. dim = 896 from Qwen2.5-0.5B breaking a block
   loop; the TheBloke-TinyLlama norm-corruption GGUF). Deleting one requires
   explicit permission.
5. **Hardware-gated tests must SKIP loudly, pass, and say so** — the glcuda
   pattern: no CUDA device → print `SKIP: no CUDA device` and return Ok. They
   must never fail on a GPU-less machine, and never silently pass without
   printing that they skipped.
6. **Serial suites stay serial:** parity/VRAM-leak suites document
   `-- --test-threads=1` at the top of the file; tests touching process-global
   state likewise.
7. **Test data must be tiny and synthetic** where possible. Real GGUF files
   are `.gitignore`d (`*.gguf`) — a test needing a real model must skip (with
   a message) when the file is absent, keyed off a documented env var or
   path probe.
8. Every bug fix ships with a test that fails on the pre-fix code. Every new
   kernel ships with: a correctness test vs the scalar reference, an
   edge-shape test, and (GPU) a parity entry.

## ✅ Correct Pattern

```rust
/// Tolerance for Q8_0 integer-dot vs scalar reference.
/// Derived from worst-case rounding across a 4096-dim dot; see
/// architecture/ArchGLLM_X5.md §tolerances.
const TOL_Q8_DOT: f32 = 1e-3;

#[test]
fn q8_0_dot_matches_scalar_reference_at_dim_896() {
    // dim 896 (Qwen2.5-0.5B) is deliberately NOT a multiple of 32-lane tiles.
    let (w, x) = synthetic_q8_case(896, seed = 7);
    let fast = q8_0_dot(&w, &x);
    let reference = scalar::q8_0_dot(&w, &x);
    assert!((fast - reference).abs() < TOL_Q8_DOT,
            "fast={fast} ref={reference}");
}

#[test]
fn forward_pass_parity() {
    let Some(dev) = CudaDevice::try_open() else {
        eprintln!("SKIP: no CUDA device");
        return;
    };
    // ... tensor-by-tensor comparison against glproc within per-op TOLs
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
assert_eq!(out, 0.734212); // ❌ float eq — flaky across ISAs by design

#[test]
fn test_kernel() { ... }   // ❌ name says nothing

// ❌ silencing a hardware-gated failure instead of skipping loudly:
#[ignore]                  // now nobody ever runs it, even on GPU machines
fn forward_pass_parity() { ... }

let tol = 0.1;             // ❌ unexplained, and wide enough to hide real bugs
```

## GwenLand-Specific Notes

- A green suite on your machine proves less than you think: GPU tests skip
  without hardware, and `cfg`-gated code compiles away on the wrong OS. State
  *which* machine/OS your green run came from in the PR.
- Sampler tests must pin the RNG seed — sampling paths are only testable as
  deterministic functions of (logits, seed, params).

## Related Skills

- [../before-coding/check-existing-tests.md](../before-coding/check-existing-tests.md)
- [error-handling.md](error-handling.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
