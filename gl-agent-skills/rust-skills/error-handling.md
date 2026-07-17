# Error Handling

> **Domain:** rust-skills
> **Applies to:** all crates (`glcore`, `glproc`, `glcuda`, `glvulkan`, `glmetal`, `glbench`, `glcli`, `packages/*`)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the project error type: **`GlError`** in [`glcore/src/error.rs`](../../glcore/src/error.rs) — NOT a crate-local error enum, NOT `anyhow`.
- [ ] The code path I'm writing is classified: production path (CLI → runtime → engine → kernels) vs test/bench/tooling.
- [ ] I am not about to add `.unwrap()` / `.expect()` / `panic!` to a production path.

## Context

GwenLand parses **untrusted model files** (GGUF, safetensors, tokenizer JSON)
and probes hardware that may not exist (no GPU, no AVX-512). Every one of those
is an expected runtime condition, not a programmer bug — so it must surface as
a typed `GlError` the runtime can react to (e.g. fall back to another engine),
never as a panic that kills the process mid-load.

## Rules

1. **All fallible paths return `Result<T, GlError>`.** `GlError` is the unified
   error type shared by every crate; do not invent per-crate error enums.
2. **NO `unwrap()`/`expect()` on production paths.** Use `?`, or an explicit
   `match` when you need to react. Allowed exceptions:
   - tests and benches;
   - locks, where poisoning already means a panicked thread;
   - genuinely infallible cases with a proof in a comment — and prefer
     restructuring so the proof isn't needed.
3. **Pick the precise variant.** `Io` (auto via `#[from]`), `Parse` for
   malformed model files, `ShapeMismatch { expected, got }` for tensor shape
   bugs, `UnsupportedDtype` for dtypes a path can't handle, `Engine` for
   init/load/hardware failures. Don't stuff everything into `Engine(String)`.
4. **Error messages carry evidence.** A `Parse` error from the GGUF parser
   must say *what* was malformed and *where* (offset/tensor name), because the
   user's only lever is "which file do I re-download".
5. **Log rejections — never silent fallback.** When an engine declines
   (hardware missing, dtype unsupported) or the runtime falls back down the
   chain, that decision must be observable (log/trace), not swallowed.
6. New `panic!`/`unreachable!` in engine crates requires a comment proving the
   branch is impossible, not just unlikely.
7. Adding a `GlError` variant is allowed when no existing variant fits —
   update `glcore/src/error.rs` with doc comments matching the existing style.

## ✅ Correct Pattern

```rust
use glcore::error::GlError;

fn load_norm_weights(t: &TensorView) -> Result<Vec<f32>, GlError> {
    if t.dtype != DType::F32 {
        return Err(GlError::UnsupportedDtype(format!(
            "norm tensor {} is {:?}, expected F32", t.name, t.dtype
        )));
    }
    t.shape_checked(&[self.dim])
        .map_err(|got| GlError::ShapeMismatch { expected: vec![self.dim], got })?;
    Ok(t.as_f32_slice()?.to_vec())
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// Untrusted GGUF metadata → panic on a malformed file:
let n_heads = metadata.get("llama.attention.head_count").unwrap(); // ❌ crash

// Silent fallback — user thinks they're on CUDA, they're on CPU:
let engine = cuda.init().ok().unwrap_or_else(|| cpu); // ❌ no log, no reason

// Lazy variant choice — undebuggable:
return Err(GlError::Engine("bad file".into())); // ❌ what file? bad how?
```

## GwenLand-Specific Notes

- There are **legacy `unwrap()`s** in `glcore/src/format/*` and `glcuda/*`.
  They are debt, not precedent: never add new ones; if your change touches a
  line with one, convert it to `?`/`GlError` in the same PR.
- "Engine unavailable" is a *normal* outcome (CUDA driver absent, wrong
  quant for a backend). It flows through `EngineSpec::available` and
  `Result`, and the runtime's fallback chain handles it — see
  [`../architecture-skills/fallback-chain.md`](../architecture-skills/fallback-chain.md).

## Related Skills

- [unsafe-rules.md](unsafe-rules.md)
- [testing-standards.md](testing-standards.md)
- [../gguf-skills/format-parsing.md](../gguf-skills/format-parsing.md)
