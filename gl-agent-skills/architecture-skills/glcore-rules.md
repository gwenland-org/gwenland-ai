# glcore Rules — Zero Inference Compute

> **Domain:** architecture-skills
> **Applies to:** [`glcore/`](../../glcore/)
> **Last updated:** 2026-07-28

## BEFORE YOU START

- [ ] I can state glcore's boundary: **shared foundation + orchestration, zero inference compute.**
- [ ] The thing I'm adding to glcore is used by (or defined for) *more than one* backend — otherwise it belongs in that backend.
- [ ] I am not adding a tensor-math kernel, however small, to glcore.

## Context

glcore is what every engine depends on and what no engine may be depended on
*by*. It owns the things that must be identical across backends — file
formats, the tokenizer, tensor types, the `GlEngine` trait, `GlError`, the
runtime that selects engines, telemetry types. The moment a matmul lands in
glcore, backends start "borrowing" it, numerics stop being per-engine, and
the parity story (each engine validated against glproc) collapses into
circularity.

## Rules

1. **glcore contains:** GGUF/safetensors parsers, the BPE tokenizer, `Tensor`
   types and dtype plumbing, `GlEngine` + `EngineSpec` + telemetry types,
   `GlError`, and the `Runtime` (select + route). That list grows rarely and
   deliberately.
2. **glcore must NEVER contain:** matmul, attention, FFN/activation math,
   sampling kernels, dequant *compute* kernels, SIMD intrinsics, GPU API
   calls — any code whose output is model numerics. If you find inference
   compute in glcore, that's a refactor task: move it to the backend(s).
3. **The runtime owns no compute either.** It selects an engine, routes the
   request, and reports the decision. "Just do the small fallback math in
   the runtime" is how rule 2 dies.
4. **Reference *dequantization* for parsing/validation is the one nuance:**
   byte-layout decoding needed to *load* and *verify* formats lives with the
   parsers, but hot-path dequant kernels used during inference belong to the
   engines. When in doubt: if it runs per-token, it is not glcore's.
5. **Dependency direction is law:** backends depend on glcore; glcore
   depends on no backend, ever, including behind a feature flag or
   `cfg(test)`. (Dev-dependencies included.)
6. **Changes to shared types are cross-backend events.** Touching `Tensor`,
   `GlError`, or `GlEngine` means checking every implementor in the same PR
   — see [`../rust-skills/trait-design.md`](../rust-skills/trait-design.md).

## ✅ Correct Pattern

```text
Need: both glproc and glcuda must repack Q4_K → Q8_0.
✅ glcore: block layout structs + validated byte accessors (format knowledge).
✅ each engine: its own repack/compute kernel using those accessors
   (glproc SIMD version, glcuda device version) — numerics stay per-engine
   and parity between them stays meaningful.
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ in glcore/src/tensor.rs — "tiny" shared math:
impl Tensor {
    pub fn matmul(&self, other: &Tensor) -> Tensor { ... } // compute in glcore
}

// ❌ in glcore/Cargo.toml:
[dependencies]
glproc = { path = "../glproc" }   // reversed dependency — forbidden even
                                  // "temporarily"
```

## GwenLand-Specific Notes

- The tokenizer and parsers being in glcore is deliberate: tokenization and
  file decoding must be bit-identical regardless of engine, and they run
  once per request/load — not per token.
- **Tokenizer work specifically:** read
  [`glcore/src/tokenizer/README.md`](../../glcore/src/tokenizer/README.md)
  first — it documents the merge engine, the pre-tokenizer's regex-arm
  grouping (not model-family grouping — a table keyed on the wrong axis is
  how 13/24 entries went wrong once), and the pre-token cache's correctness
  gate. `glcore/tests/tokenizer_parity.rs` scores every claimed vocabulary
  family against llama.cpp's reference vectors on every build; a family this
  crate cannot express is refused at load, never approximated.
- Telemetry stays **pull-based** (`GlEngine::telemetry()` snapshot): glcore
  defines the types, engines fill them, glbench reads them. glcore never
  pushes callbacks into engines.

## Related Skills

- [backend-independence.md](backend-independence.md)
- [fallback-chain.md](fallback-chain.md)
- [../rust-skills/trait-design.md](../rust-skills/trait-design.md)
