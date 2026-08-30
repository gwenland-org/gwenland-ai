# GGUF Format Parsing

> **Domain:** gguf-skills
> **Applies to:** [`glcore/src/format/gguf.rs`](../../glcore/src/format/gguf.rs) (and `safetensors.rs` by analogy)
> **Last updated:** 2026-07-17
>
> ⚠️ **Upstream drift warning:** GGUF is defined by the upstream
> **ggml / llama.cpp** project, not by GwenLand. Everything in this skill —
> byte layouts, metadata keys, version numbers, quant types — reflects the
> spec as of the date above, and ggml evolves (new versions, new types, new
> keys). When a real file disagrees with this skill or our parser, check the
> official ggml spec/source **first**, then update the parser *and* this
> skill together. Never "fix" the parser to match one odd file.

## BEFORE YOU START

- [ ] I treat every GGUF as **untrusted input** — this parser is the repo's #1 security surface ([`SECURITY.md`](../../SECURITY.md)).
- [ ] Zero external dependencies: the parser is std + mmap only. No gguf crates, no serde for the binary format.
- [ ] Every offset/length I read from the file gets bounds-checked before use.

## Context

The parser is from-scratch, mmap-based, and zero-copy: it decodes the header
and metadata KV section, builds a tensor directory (name, dtype, shape,
offset), and hands out borrowed views into the mapping. Nothing is trusted:
lengths, counts, offsets, and alignment all come from attacker-controllable
bytes, and the parser is the wall between those bytes and `unsafe` slice
casts.

## Rules

1. **Bounds-check before every access.** Any offset/length arithmetic uses
   checked math (`checked_add`/`checked_mul`) and validates against the
   mapped size *before* slicing. Overflow or out-of-range →
   `GlError::Parse` naming the tensor/key and offset — never a panic, never
   UB ([`../rust-skills/unsafe-rules.md`](../rust-skills/unsafe-rules.md)).
2. **Sanity-cap unbounded counts.** Tensor count, KV count, string lengths,
   array lengths are attacker-supplied — reject absurd values with a clear
   error instead of attempting a 2⁶³-entry allocation.
3. **Zero-copy by default:** tensor data is returned as borrowed views into
   the mmap with lifetimes tied to it. Materialize only small tensors and
   explicit repack outputs ([`../rust-skills/memory-safety.md`](../rust-skills/memory-safety.md)).
4. **Alignment comes from the file's `general.alignment` (default 32)** —
   compute the data-section base from the spec, don't hardcode paddings.
   Misaligned tensor base = `Parse` error, not a silent `loadu` shrug at the
   parser layer (kernels choose `loadu`; the *parser* enforces the spec).
5. **Unknown ≠ invalid:** unknown metadata keys are skipped (preserved for
   `gwen info` display where feasible); unknown dtype/quant IDs produce
   `UnsupportedDtype` with the numeric ID — the file may be from a newer
   ggml (see the drift warning), so say that in the message.
6. **Version handling is explicit:** supported GGUF version(s) are checked
   up front; an unsupported version is a clean error naming both versions.
   Don't guess-parse across versions.
7. **Metadata is validated against tensors:** declared dims/dtype must be
   consistent with byte lengths (the `split_experts` cross-check pattern).
   Inconsistency is a hard `Parse` error — this has caught real broken
   exports.
8. **Every parser bug becomes a synthetic regression fixture** — a tiny
   crafted header/file in the tests, not a 500 MB real model
   ([`../rust-skills/testing-standards.md`](../rust-skills/testing-standards.md)).

## ✅ Correct Pattern

```rust
let end = t.offset
    .checked_add(t.byte_len)
    .ok_or_else(|| GlError::Parse(format!("tensor {}: offset overflow", t.name)))?;
if end > map.len() {
    return Err(GlError::Parse(format!(
        "tensor {}: [{}..{}] exceeds file size {}", t.name, t.offset, end, map.len()
    )));
}
let bytes = &map[t.offset..end]; // safe: bounds proven above
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ trusting attacker math — offset+len can wrap or exceed the map:
let bytes = &map[t.offset..t.offset + t.byte_len];

// ❌ pre-allocating from an untrusted count:
let mut tensors = Vec::with_capacity(header.tensor_count as usize); // 2^63 → OOM

// ❌ "fixing" the parser so one weird file loads:
// if that file violates the ggml spec, the file is broken (see the
// TheBloke-TinyLlama norm-corruption case) — reject it, don't bend to it.
```

## GwenLand-Specific Notes

- **Broken GGUFs exist in the wild** (the TheBloke TinyLlama re-export with
  corrupted norm weights is our canonical case). The parser's job includes
  *detecting* them where possible; garbage output from a well-formed-but-
  miscooked file is a model problem, and `gwen info` + validation output is
  how users find out.
- The parser feeds both engines *and* `glictus-caliburni`'s converter —
  layout knowledge lives here once, in glcore
  ([`../architecture-skills/glcore-rules.md`](../architecture-skills/glcore-rules.md)).

## Related Skills

- [quantization-types.md](quantization-types.md)
- [moe-loading.md](moe-loading.md)
- [../rust-skills/error-handling.md](../rust-skills/error-handling.md)
