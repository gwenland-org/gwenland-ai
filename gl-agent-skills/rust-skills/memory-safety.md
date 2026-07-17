# Memory Safety & the RAM Budget

> **Domain:** rust-skills
> **Applies to:** `glcore` (loader/tensors), `glproc` (hot paths), all backends
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know the RAM budget: the reference machine has **8 GB total**; the loader is mmap-based zero-copy precisely so weights are *not* duplicated into heap.
- [ ] My change does not clone, collect, or buffer a full-size weight tensor.
- [ ] Hot-path allocations: I checked whether the runner's existing scratch buffers already cover my need.

## Context

GwenLand's core promise is running real models on an 8 GB machine. That works
because weights live in the **mmap'd file** and are read through borrowed
slices — the OS pages them in and out. One careless `.to_vec()` on an FFN
weight matrix adds hundreds of MB of resident heap and breaks the promise more
surely than any perf regression. Memory discipline here *is* correctness.

## Rules

1. **Never hold an extra full-size copy of weights.** Borrow from the mmap
   (`&[u8]` → typed slice views). Materializing is allowed only for small
   tensors (norms, biases) or explicit, documented repack steps.
2. **Zero allocation in the decode loop.** The runner pre-allocates scratch
   buffers, KV cache, and logits once; per-token code reuses them. Adding a
   `Vec::new()`/`format!`/`Box::new` inside the token loop needs explicit
   justification.
3. **Borrow, don't refcount, in hot paths.** Prefer `&[f32]`/`&mut [f32]`
   arguments over `Arc<Vec<f32>>`. `Arc` is for ownership across threads at
   setup time, not for per-call sharing.
4. **KV cache grows by policy, not ad hoc.** It is pre-sized from context
   length; kernels index into it via a cursor. Do not resize or reallocate it
   mid-generation.
5. **Slices over raw indexing.** Use `chunks_exact`, `split_at_mut`, and
   iterator patterns that eliminate bounds checks safely before reaching for
   `unsafe` (see [unsafe-rules.md](unsafe-rules.md)).
6. **Lifetimes make the mmap dependency explicit.** Types that view into the
   mapped file carry a lifetime tied to the mapping. Do not "fix" a lifetime
   error by copying the data out — restructure ownership instead.
7. Dropping/`shutdown()` must actually release: GPU buffers freed, mappings
   dropped. Leak checks in glcuda's tests are part of the contract.

## ✅ Correct Pattern

```rust
/// Borrows the tensor bytes straight out of the mmap — zero copy.
fn ffn_weight<'m>(&self, map: &'m Mmap, t: &TensorMeta) -> Result<&'m [i8], GlError> {
    let bytes = map
        .get(t.offset..t.offset + t.byte_len)
        .ok_or_else(|| GlError::Parse(format!("tensor {} out of file bounds", t.name)))?;
    Ok(bytemuck_cast_checked(bytes)?)
}

// Per-token: reuse the pre-allocated scratch, no allocation.
self.scratch.logits.fill(0.0);
kernel::lm_head(weights, &hidden, &mut self.scratch.logits);
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ materializes a full weight matrix — blows the 8 GB budget:
let w: Vec<f32> = tensor.iter_dequant().collect();

// ❌ allocation inside the decode loop:
for step in 0..max_tokens {
    let mut probs = vec![0f32; vocab_size]; // fresh 600 KB+ every token
    ...
}

// ❌ "fixing" a borrow error by cloning the KV cache slice:
let keys = self.kv.keys_for_layer(l).to_vec();
```

## GwenLand-Specific Notes

- Page-cache advice is OS-specific and has already bitten us once:
  `MADV_DONTNEED` under `#[cfg(unix)]` broke macOS builds — gate such code
  `#[cfg(target_os = "linux")]` and compile it on a platform that exercises it.
- Raw-mmap **lazy layer paging is a rejected design** — it is incompatible
  with the Q8_0 repack path. Do not reintroduce it
  (see [`../cpu-skills/rejected-optimizations.md`](../cpu-skills/rejected-optimizations.md)).
- Repacked tensors (e.g. Q4_K→Q8_0) are an *explicit* exception to rule 1:
  the repack buffer replaces the original as the working copy — never keep
  both hot.

## Related Skills

- [unsafe-rules.md](unsafe-rules.md)
- [../cpu-skills/memory-bandwidth.md](../cpu-skills/memory-bandwidth.md)
- [../gguf-skills/format-parsing.md](../gguf-skills/format-parsing.md)
