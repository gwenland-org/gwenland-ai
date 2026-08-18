# Implementing the Stummañ Checkpoint Family

> **Domain:** stumman-m2-skills
> **Applies to:** `stumman/src/checkpoint/` (does not exist yet)
> **Scope:** `CPLora` (**FULL**), `CPFull`, `CPSharded`, `CPIncremental` (all
> **STUB**, real formats), `PLGgufMerge` (**STUB**, separate trait)
> **Prerequisite reading:** `stumman/M2_RESEARCH.md` §3, §7-D, §7-E, §8.5 — the
> full research and architecture record this skill distills.
> **Last updated:** 2026-08-17

## BEFORE YOU START

- [ ] I know `CP` is the approved prefix and now means **Checkpoint**, not
      Compiler (`gwenland-naming-convention/SKILL.md`, decided 2026-08-17 —
      Compiler moved to `CM`, and `CP` was reassigned because zero types in the
      repo used it). `CPLora`, not `LoraCheckpoint`.
- [ ] I know `PLGgufMerge` is `PL` (Pipeline), **not** `CP` — see Rule 1.
- [ ] I have read `glcore/src/format/safetensors.rs` in full. The **reader**
      already exists there, from scratch, with an mmap-backed parser and
      offset-bounds checking. There is no writer anywhere in the repo — that is
      the one new format primitive `CPLora` needs.
- [ ] Before writing `PLGgufMerge` specifically: I have read
      `glcore/src/format/gguf.rs` in full (`GgufFile::open`, `GgufDType`,
      `dequantize`) — the **reader** and dtype enum already exist there too,
      and `gl-agent-skills/gguf-skills/quantization-types.md` +
      `gl-agent-skills/gguf-skills/dequant-path.md` for the quantization rules
      this pipeline must not reinvent.
- [ ] I have read `glictus-caliburni/src/plugin.rs` and
      `glictus-caliburni/src/manifest/validator.rs` — both patterns are adopted
      here nearly verbatim.
- [ ] I have read `gljax/src/checkpoint/safetensors.rs`'s `bind_safetensors` —
      it already implements exactly the shape-mismatch-including-transposed
      check this skill's Rule 4 requires, on a real bug class (gljax hit a
      transposed-weight bug from HF checkpoints in production).

## Context

The task brief's five checkpoint types (`LoraCheckpoint`, `FullCheckpoint`,
`ShardedCheckpoint`, `IncrementalCheckpoint`, `GgufMerge`) do not share one
interface, because one of them cannot round-trip and the other four can. Get
that split right first (Rule 1); everything else follows from it. Rules 2–8
cover the shared `CheckpointStore` family; Rules 9–12 give each individual
stub its real, researched format — a stub whose format is invented on the spot
by whoever implements it later inherits nothing from this research pass.

## Rules — architecture (all types)

1. **Split into `CheckpointStore` (round-trips) and `Exporter` (one-way).**
   `CPLora`/`CPFull`/`CPSharded`/`CPIncremental` implement `CheckpointStore`
   (has `load()`). `PLGgufMerge` implements `Exporter` (no `load()` — GGUF
   requantization is lossy and one-way by construction). Forcing `GgufMerge`
   behind `CheckpointStore` means giving it a `load()` that can never be
   implemented, which is exactly the "meaningless method on a stub" shape
   `gl-agent-skills` forbids. This is *why* it is `PLGgufMerge`, not `CPGgufMerge`
   — the prefix records the architectural split, not just a naming preference.

2. **A checkpoint is a directory of independent segments under one manifest,
   not one opaque file.** Six categories exist and have different
   save/deploy-relevance:

   | Segment | Resume? | Deploy? |
   |---|---|---|
   | Adapter state (A/B/...) | yes | yes |
   | Optimizer state (m, v) | yes | no |
   | Scheduler state | yes | no |
   | Training progress (step, RNG) | yes | no |
   | Metadata / config | yes | yes |

   `save_adapter_only()` writes the adapter + metadata files.
   `save_full(include_optimizer: bool)` writes everything. Loading a segment is
   independent — a deploy path never has to parse optimizer-state bytes it
   does not need. `gltrain/src/train/adamw_state.rs` already does the sidecar
   half of this (`{stem}_adamw.safetensors`); generalize that instinct, don't
   reinvent it.

3. **`TensorId` cannot be a checkpoint key.** `stumman/src/tensor/tensor.rs`'s
   own doc comment: *"Do not persist or compare IDs across process restarts...
   Checkpoints must key on parameter names, never on these."* Every tensor
   entry in every segment is keyed by the parameter's **name**
   (`TPParameter::name()`), full stop. Optimizer state (keyed by `TensorId` in
   memory, per `adamw.md` Rule 4) gets re-keyed to name at exactly the
   save/load boundary and nowhere else.

4. **Validate transposed shapes, not just element counts, and report every
   error at once.** `[896, 4864]` and `[4864, 896]` have the same element count
   and would load silently under a size-only check — this is a real bug class
   (gljax hit it against real HF weights; `gljax/src/checkpoint/safetensors.rs`'s
   `bind_safetensors` doc comment explains why it checks shape, not just size).
   Copy that function's error-reporting shape too: collect *every* missing key,
   *every* shape mismatch (including transposed), and *every* length
   disagreement in one pass, plus unused-checkpoint-key warnings — one pass
   that reports everything is what tells a caller "one key off" from "wrong
   model entirely". Return a `VLValidation { errors: Vec<String>, warnings:
   Vec<String> }`, following `glictus-caliburni/src/manifest/validator.rs`'s
   `ValidationResult` shape exactly (major mismatch = error, minor = warning).

5. **Write the safetensors format from scratch, matching the existing reader.**
   Format (verified against the official spec): 8 bytes little-endian `u64`
   header length, then that many bytes of UTF-8 JSON
   (`{"name": {"dtype", "shape", "data_offsets": [begin, end)}}`, offsets
   relative to the start of the byte buffer, `__metadata__` reserved for a
   flat **string→string** map only), then the raw tensor bytes with no holes.
   `glcore/src/format/safetensors.rs` is the reader and the correctness oracle:
   a round-trip test (`write` then `SafetensorsFile::open` then compare) is
   the primary correctness check for the writer, and it is a check nothing
   else in the repo can perform, which is the reason to write this in-tree
   instead of reaching for a crate.

6. **Reserve a field for "generated from this seed" alongside "here are the
   bytes".** This is not speculative — it is required by VeRA (M3, see
   `lora.md` Rule 5/6): its frozen `A`/`B` pair is explicitly *not* meant to be
   stored, only its RNG seed. A tensor-entry schema of `{name, dtype, shape,
   data_offsets}` alone cannot express that and would need a breaking format
   change to add it later. Add the variant now (e.g. an entry can be
   `Stored{data_offsets}` or `Generated{seed, distribution}`), even though only
   `Stored` is used until VeRA lands.

7. **Every `CheckpointStore` implementor gets a `VLCheckpointFormat`
   capability record**, mirroring `VLAdapterCapability`/`VLOptimizerCapability`
   in `lora.md`/`adamw.md`: `id`, `status`, `round_trips: bool` (always `true`
   for `CheckpointStore` implementors, which is exactly why `PLGgufMerge` is
   not one), `segments: &'static [&'static str]`, `source`.

8. **Registry pattern.** `CheckpointRegistry<B>` (or two registries, one per
   trait) follows `glictus-caliburni/src/plugin.rs`'s `PluginRegistry` —
   refuse duplicate id, `Option` for `resolve`, `Result` for `require`,
   `with_builtins()` preloads all five. This is now the third registry in this
   family (adapters, optimizers, checkpoints) — if a fourth is ever needed,
   extract the shared shape instead of copying it a fourth time.

## Rules — per-type formats

9. **`CPLora` (FULL) — concrete layout.** A directory, e.g.
   `checkpoint_000500/`:

   ```text
   checkpoint_000500/
     manifest.json            # format version, adapter type ("lora"), r, alpha,
                               # rslora, d_in/d_out per site, step, base model id
     adapter.safetensors      # {"lora_layer_0_a": ..., "lora_layer_0_b": ..., ...}
     optimizer.safetensors    # OPTIONAL sidecar: {"lora_layer_0_a.m": ..., ".v": ...}
   ```

   `manifest.json`'s format version follows the same major/minor split as
   `glictus-caliburni/src/manifest/metadata.rs`'s `is_compatible_with` — a
   major mismatch is an error, a minor one a warning, never the reverse. Tensor
   names inside `adapter.safetensors` must match `TPParameter::name()`
   (`lora_a`, `lora_b` per `LRLora`), not a synthetic index — see Rule 3.

10. **`CPFull` (stub) — same segments, all parameters instead of just the
    adapter.** Its real blocker is not the checkpoint format, it is that
    stumman has no model tree yet to enumerate ("all params" requires a
    `Module` that owns the base model, and M2 only has `ABLinear` + adapters
    over an externally-supplied base weight). The stub's capability record
    should say exactly that (`reason: "needs a full Module tree; the segment
    format itself is CPLora's, unchanged"`), not invent an unrelated blocker.

11. **`CPSharded` (stub) — a pure layout over the same manifest, not a new
    format.** HuggingFace's convention: shard files
    `model-00001-of-00006.safetensors`, plus an index
    `model.safetensors.index.json`:

    ```json
    { "metadata": { "total_size": 28966928384 },
      "weight_map": { "lora_layer_0_a": "model-00001-of-00006.safetensors", "...": "..." } }
    ```

    Default max shard size 5 GB (configurable). `CPSharded`'s `load()` reads
    the index first, then only the shard(s) a given `parameters_mut()` call
    actually needs — that partial-loading property is the entire reason this
    type exists, and a stub that just concatenates shards on load defeats it.
    Implement as "the same manifest, plus an index file", not a parallel format
    — reuse `CPLora`'s tensor-entry schema verbatim inside each shard.

12. **`CPIncremental` (stub) — genuinely low-value here; the capability record
    should say so, not imply otherwise.** Check-N-Run's differential mode
    (NSDI '22, arXiv:2010.08679) saves space because recommendation-model
    embedding tables are *sparsely* updated between checkpoints; the paper
    itself notes limited applicability outside that shape. **Every LoRA
    adapter parameter gets a gradient every step**, so a presence-delta ("which
    tensor changed") saves nothing for this crate's primary workload — only a
    *value*-delta plus compression would, which is substantially more work for
    a smaller win. If built anyway, the honest shape is:

    ```text
    checkpoint_000520_delta/
      manifest.json     # base_checkpoint: "checkpoint_000500", chain_depth: 1
      delta.safetensors # full per-element VALUE deltas (not presence flags),
                         # named identically to the base's tensors
    ```

    with a **chain-depth cap** (reconstruction cost grows with chain length)
    and a documented recovery rule: past the cap, or on any chain-link
    corruption, fall back to the nearest full checkpoint rather than attempting
    partial reconstruction. Capability record: `reason: "near-zero benefit for
    dense LoRA gradients (Check-N-Run's saving is embedding-table sparsity,
    absent here); implement only if a real workload needs it"`.

13. **`PLGgufMerge` (stub, `Exporter`) — the pipeline, in order.** Given a
    `CPLora` checkpoint and a base model:

    1. Load the adapter checkpoint (`CPLora::load`).
    2. Load the base model. If it is already GGUF: `glcore::format::gguf::GgufFile::open`
       + `.dequantize(info)` per tensor — **reuse this reader, do not
       reparse the format.**
    3. For each adapted site: `merged = W0 + scale·(A @ B)` — this is exactly
       `LRLora::merge_into`'s math against a dense `Tensor<B>`; call it, don't
       reimplement it.
    4. **Requantize** each merged tensor to the target GGUF quant type. This
       calls into glproc's existing quant kernels — read
       `gl-agent-skills/gguf-skills/quantization-types.md` and
       `gl-agent-skills/gguf-skills/dequant-path.md` first, and check
       `gl-agent-skills/cpu-skills/rejected-optimizations.md` before assuming
       any particular quant type is fast on this tier (native Q4_K was built
       and measured 33% *slower*, compute-bound, not memory-bound — a
       surprising, already-paid-for lesson, do not rediscover it).
    5. Write the merged GGUF. There is **no GGUF writer anywhere in the repo**
       — `glcore/src/format/gguf.rs` is read-only. This is genuinely new work:
       header (`magic 0x46554747`, `version 3`, tensor_count, kv_count),
       metadata KV section, tensor info (name ≤ **64 bytes**, dims ≤ 4,
       `ggml_type`, offset), then the data section aligned to
       `general.alignment` (default **32**, must be a multiple of 8),
       `0x00`-padded between tensors.

    `PLGgufMerge` has no `load()` — trying to round-trip a requantized GGUF
    back into a resumable training checkpoint would silently lose precision
    with no error, which is worse than not offering the method at all.

## ✅ Correct Pattern

```rust
pub trait CheckpointStore {
    fn save(&self, path: &Path, segments: &[CheckpointSegment]) -> Result<()>;
    fn load(&self, path: &Path) -> Result<LoadedCheckpoint>;
    fn validate(&self, path: &Path, against: &Signature) -> Result<VLValidation>;
    fn capability(&self) -> &'static VLCheckpointFormat;
}

// GgufMerge does NOT implement CheckpointStore -- there is no sound `load()`.
pub trait Exporter {
    fn export(&self, adapter: &dyn Adapter<B>, base: &Path, out: &Path) -> Result<()>;
}
impl Exporter for PLGgufMerge {
    fn export(&self, adapter: &dyn Adapter<B>, base: &Path, out: &Path) -> Result<()> {
        // 1. load base via glcore::format::gguf::GgufFile (existing reader)
        // 2. merge via Adapter::merge_into (existing, e.g. LRLora)
        // 3. requantize via glproc's existing kernels
        // 4. write via a NEW gguf writer (does not exist yet -- this is the work)
    }
}
```

## ❌ Anti-Pattern (Never Do This)

```rust
// ❌ CP for the export pipeline -- implies a load() that cannot exist
pub struct CPGgufMerge;

// ❌ one opaque blob mixing model, optimizer, and training state
fn save(path: &Path) { /* single file, everything serialized together */ }
// -- a deploy path now has to parse optimizer bytes it will never use

// ❌ validating only element count, missing the transposed-shape bug class
fn validate(a: &[usize], b: &[usize]) -> bool { a.iter().product::<usize>() == b.iter().product() }

// ❌ keying optimizer state on TensorId in the SAVED file
{ "12": { "m": [...], "v": [...] } }   // "12" is a process-local ID, meaningless on reload

// ❌ a tensor-entry schema with no room for "generated, not stored"
struct TensorEntry { name: String, dtype: DType, shape: Vec<usize>, data_offsets: [usize; 2] }
// -- VeRA's checkpoint cannot be expressed without a breaking change later

// ❌ CPIncremental's stub implying it's a near-term win when the research says otherwise
reason: "not implemented yet",   // true but misleading -- omits that it's low-value here

// ❌ CPSharded's stub loading every shard regardless of which params were asked for
fn load(&self, path: &Path) -> Result<LoadedCheckpoint> {
    for shard in all_shards { load_entire_shard(shard)?; }  // defeats the point of sharding
}

// ❌ PLGgufMerge reimplementing quantization instead of calling glproc's kernels
fn requantize_q4_k(&self, data: &[f32]) -> Vec<u8> { /* hand-rolled nibble packing */ }
```

## GwenLand-Specific Notes

- `CPLora`'s manifest should carry a format version and go through the same
  major/minor validation split as
  `glictus-caliburni/src/manifest/metadata.rs`'s `is_compatible_with` — major
  mismatch is an error, minor is a warning, not the other way around.
- The GGUF alignment/name-length/tensor-info rules in Rule 13 step 5 are
  verified against the official ggml spec, not inferred — if
  `general.alignment` is absent when reading a *base* model, the default is
  **32**, not 0 and not unaligned.

## Related Skills

- [adamw.md](adamw.md) — `state_tensors`/`load_state`, the optimizer-segment source
- [lora.md](lora.md) — VeRA's seed-not-bytes requirement (Rule 6 here), transposed-shape stakes (Rule 4 here), `LRLora::merge_into`'s math (Rule 13 here)
- [../stumman-naming/SKILL.md](../stumman-naming/SKILL.md) — the `CP`/`PL` prefix decisions and full M2 name map
- [../gguf-skills/quantization-types.md](../gguf-skills/quantization-types.md), [../gguf-skills/dequant-path.md](../gguf-skills/dequant-path.md) — required before touching `PLGgufMerge`'s requantization step
- [../cpu-skills/rejected-optimizations.md](../cpu-skills/rejected-optimizations.md) — do not re-probe a quant-type speed assumption already measured false
- [../architecture-skills/inference-first.md](../architecture-skills/inference-first.md) — Rule 6 (no dependency for something writable from scratch) is why the safetensors writer is in-tree
