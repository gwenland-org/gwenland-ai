# Check Existing Tests — Before AND After

> **Domain:** before-coding
> **Applies to:** every code change in every crate
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I ran the relevant test suites BEFORE changing anything and recorded the baseline result.
- [ ] I know which tests cover the code I'm about to touch (grep the test files).
- [ ] I know which tests are hardware-gated (GPU tests skip without a CUDA device) so a "pass" on my machine isn't over-read.

## Context

The CPU engine (`glproc`) is the **numerical ground truth** that every GPU
backend is validated against, tensor-by-tensor, within explicit per-operation
tolerances. A change that silently shifts glproc's numerics doesn't just break
glproc — it invalidates the parity baseline for every other engine. The test
suite is the contract; run it before you touch the code so you can tell *your*
breakage from pre-existing breakage.

## Rules

1. **Baseline first.** Run the suite before your first edit. If something is
   already red, report it — do not fix-and-bundle it into your change.
2. The standard check set (from the workspace root):
   ```bash
   cargo test -p glcore -p glproc
   cargo test -p gltui
   cargo test -p glcuda --lib                       # host-side, GPU-less OK
   cargo fmt --all
   cargo clippy --workspace --all-targets -- -D warnings
   ```
3. **GPU parity suites need real hardware and serial execution:**
   ```bash
   cargo test -p glcuda --test parity -- --test-threads=1
   ```
   `--test-threads=1` is mandatory there — the VRAM-leak check is perturbed by
   concurrent allocations. glcuda's GPU tests print `SKIP` and pass on machines
   without a CUDA device; a green run on a GPU-less box proves **nothing**
   about kernel correctness.
4. Tests touching process-global state (panic hooks, env vars) also run with
   `-- --test-threads=1`.
5. **Never weaken a test to make it pass** — no widening a parity tolerance, no
   `#[ignore]`, no deleting assertions — without explicit permission and a
   written justification in the PR/changelog note.
6. **After the change:** rerun the same set, plus a new test that fails without
   your fix (for bug fixes) or covers the new path (for features).
7. If a test fails after your change: STOP, report the failure verbatim, and
   wait for instruction (see [wave-confirmation-gates.md](wave-confirmation-gates.md)).
   Do not iterate blindly toward green.

## ✅ Correct Pattern

```rust
// Bug fix PR: the fix comes WITH a regression test that fails on the old code.
#[test]
fn q8_0_dot_handles_non_multiple_of_block_dim() {
    // dim = 896 is a real-world case (Qwen2.5-0.5B) that broke the naive
    // block loop; keep it as a permanent regression guard.
    let out = q8_0_dot(&weights_dim_896(), &activations());
    assert!((out - reference_scalar_dot()).abs() < TOL_Q8_DOT);
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// Parity test started failing after "optimizing" a kernel, so:
const TOL_Q8_DOT: f32 = 1e-1; // was 1e-3  ← NO. You just redefined "correct"
                              // to fit your bug and poisoned the baseline
                              // every GPU backend validates against.
```

## GwenLand-Specific Notes

- `Cargo.lock` is committed — use `--locked` when you need reproducible deps.
- Watch for `cfg`-gated paths: code under `#[cfg(unix)]` /
  `#[cfg(target_os = ...)]` compiles cleanly on the *wrong* platform because
  it's skipped there. If you touch a gated path, build it on a platform that
  actually compiles it (see CONTRIBUTING's `MADV_DONTNEED`/macOS incident).
- CI runs on GitHub Actions (`.github/workflows/ci.yml`); a local green is not
  a substitute for CI green on Linux.

## Related Skills

- [read-architecture-first.md](read-architecture-first.md)
- [wave-confirmation-gates.md](wave-confirmation-gates.md)
- [../rust-skills/testing-standards.md](../rust-skills/testing-standards.md)
