# glcore

**The shared vocabulary.** Every other GL crate speaks these types.

## What level does it work at?

**All three, and that is the point** — glcore is where bytes become tensors and
text becomes tokens, so it is the only crate that spans the whole range:

| Level | What glcore does |
|---|---|
| **Byte** | `format/` — GGUF and safetensors parsers. Reads headers, validates lengths and alignment, and reinterprets mmap'd bytes as typed slices. |
| **Tensor** | `tensor.rs` — `Tensor`, `DType`, dequantisation to f32 for the dtypes it owns. |
| **Token** | `tokenizer/` — a from-scratch BPE/SentencePiece tokenizer. **14 GGUF vocabulary families verified exact** against reference vectors. |
| **Contract** | `engine_trait.rs` — the `GlEngine` trait every backend implements, and `runtime.rs`, the `Runtime` front-ends drive. |

glcore **does not run inference**. It defines what an engine is; `glproc` and
`glcuda` are engines.

## What is in here

```
format/      GGUF + safetensors parsing
tensor/      Tensor, DType, dequant
tokenizer/   BPE + SentencePiece, 14 vocab families
engine_trait GlEngine, InferInput, InferOutput, EngineSpec
runtime/     Runtime — owns tokenization, holds one Box<dyn GlEngine>
hash/        SHA-256, the workspace's ONE implementation
telemetry/   the vocabulary engines report their internals in
gate/        GATE protocol boilerplate
stopping/    StoppingCriteria
trace/       TokenTrace, TraceConfig
```

~7,700 lines.

## Dependencies

Five, all boring:

```
thiserror  memmap2  byteorder  serde  serde_json
```

Fifteen crates in the full tree. **Zero external ML dependencies** — the
"Inference First" rule (`gl-agent-skills/architecture-skills/inference-first.md`)
is enforced here first, because everything downstream inherits it.

`hash.rs` is hand-written SHA-256 rather than the `sha2` crate. That is
deliberate: `glbench` forbids external crates entirely and its archive
integrity cannot be feature-gated, so a `sha2` edge here would reach every
crate in the workspace. It is tested against the published FIPS-180-4 vectors,
including the one-million-`a` case — a hand-written digest that is not tested
against them is just a hash-shaped function.

## Gotchas worth knowing before you touch it

- **`Vocab::from_hf_json` once dropped `added_tokens`**, which silently broke
  every modern `tokenizer.json`. Fixed 2026-07-29. The "14 vocab families
  exact" claim covers the **GGUF path**; the HF-JSON loader has far thinner
  coverage.
- **Q6_K dequant lived here and was wrong** (naive linear nibble order),
  corrupting `ffn_down.weight` in every layer of a real Q4_K_M model for
  months. Q4_K/Q5_0/Q6_K now route through `glproc`'s kernels instead. See
  `architecture/gl-stack-audit-2026-07/ARTX2-Quant.md`.
- `format::dequantize()` deliberately **excludes** Q4_K/Q5_0 — dequant lives in
  glproc (Pridwen Phase 2 ADR).

## Build

```bash
cargo test -p glcore
```
