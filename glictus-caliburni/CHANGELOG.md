# Changelog — glictus-caliburni

GLLM (GwenLand Language Model Format), codename *Ictus Caliburni*.
All notable changes to this crate. Dates are WIB/SEAST.

## [0.1.161] — 2026-07-19 · ARTX07-lite: GGUF → GLLM converter

- **`converter` module + `glconv` binary**, feature-gated behind
  `converter` (pulls `glcore` for GGUF parsing) — the default build stays
  zero-workspace-dep. Usage: `glconv <input.gguf> <out_dir> [--model-id]`.
- Pipeline per ARTX7: GGUF parse → tensor grouping (`token_embd`/
  `output_norm`/`output` → shared; `blk.N.*` → `layer_NNN` with prefix
  stripped) → `write_unit_file` per unit → manifest + `checksums.sha256`
  → **self-validation**: re-open via `GllmPackage::open` + cross-check
  every layer + full integrity sweep.
- Metadata mapping is architecture-prefix aware (`{arch}.block_count`,
  …); `head_count_kv` falls back to `num_heads`; `vocab_size` from
  `{arch}.vocab_size` → tokenizer token count → `token_embd` shape.
- **New dtypes `Q5_0` (0x0015) and `Q6_K` (0x0018)** — real Q4_K_M GGUFs
  carry Q5_0 fallback rows and Q6_K output heads.
- Declared lite-scope deviations: RoPE tables not materialized
  (derivable), unmapped tensors → shared with warning (not projector),
  tokenizer NOT packaged (spec open question — warning emitted), GGUF
  fastest-first dims reversed to row-major shapes.
- Real-model test is opt-in via `GWENLAND_TEST_GGUF` (skips loudly).
- Also: cleared new clippy-1.95 lint debt in `glcore` (tokenizer/gguf).

## [0.1.160] — 2026-07-19 · ARTX04 Waves 2–4: tensor index & layer I/O

- **`layer_io` (new module):** binary tensor index codec per ARTX04 —
  `name_len u16` + UTF-8 name + `shape_len u8` + u32 dims + dtype code +
  `offset`/`size` u64. Dims widen to the manifest's `u64` on read; write
  rejects `DType::Unknown`, dims > u32, rank > 255.
- **`LayerFile::read`:** one buffered pass — header, full index, 64-byte
  aligned `data_offset`, overflow-safe bounds check of every tensor against
  file size. `absolute_range(name)` returns seek/mmap-ready ranges.
- **`write_unit_file`:** complete unit-file writer (converter/fixture path)
  with auto-computed, per-tensor 64-byte-aligned offsets.
- **`GllmPackage::read_layer_file` / `cross_check_layer`:** binary index vs
  manifest metadata consistency check (both directions), opt-in — not run
  during `open()`.
- Declared deviations from ARTX04 text: index starts at offset 16 (hybrid
  header), offsets are region-relative (ARTX03 semantics), not absolute.
- 120 tests total.

## [0.1.159] — 2026-07-19 · ARTX04 Wave 1: hybrid execution-unit header

- Resolved the `LayerHeader` (12-byte, ARTX01/04) vs `ExecutionUnitHeader`
  (16-byte, ARTX02) conflict — **hybrid layout decided by JinXSuper:** one
  universal 16-byte header for all unit files; ARTX04's `tensor_count` u32
  occupies bytes 8..12 (ex-reserved), reserved shrinks to 12..16.
- `LayerHeader` deleted; `LayerFile` carries `ExecutionUnitHeader`.
- ⚠️ Recorded caveat: u16-LE vs major/minor-u8 version encodings are only
  byte-identical for v1.0 — must be re-decided before any minor bump
  (`notes/gllm-layerheader-vs-executionunitheader.md`).

## [0.1.158] — 2026-07-19 · ARTX03 Waves 2–5: manifest, validator, integration

- **`manifest::metadata`:** `FormatVersion` (semver parse, compatibility
  check: same-major loadable, minor mismatch warns), `ModelMetadata` with
  GQA/MoE/RoPE helpers.
- **`manifest` (top level):** full `gllm.json` structs — `SharedManifest`,
  `LayerManifest`, `ProjectorManifest`, `DevicePlacement`, `CustomMetadata`,
  `GllmManifest` with context-wrapped parse errors.
- **`manifest::validator`:** semantic rules V01–V17, collecting **all**
  errors + warnings without short-circuiting.
- **`GllmPackage::open`** now: discover → parse manifest → validate (fatal →
  `ManifestValidationError`) → verify `shared.gllm` against the manifest
  checksum. `verifier_from_manifest()` covers packages without
  `checksums.sha256`.
- ARTX01 `types/manifest.rs` superseded into a shim; old placeholder types
  (`Architecture`, `SharedComponent`, `LayerEntry`, `ProjectorEntry`)
  dropped.
- Open problems filed in `notes/gllm-*.md` (DevicePlacement cuda-index loss,
  GllmPackageMeta redundancy).

## [0.1.157] — 2026-07-19 · ARTX03 Wave 1: manifest core types

- **`manifest::types` (new canonical home):** `DType` (manifest serde names,
  `#[serde(other)] Unknown` forward-compat, exact `bytes_per_element()`
  vs `approx_bytes_per_element()` split), `TensorEntry` (`Vec<u64>` shape,
  `validate()`), `ExtensionUri` (validated string newtype; parts are
  methods).
- `types/tensor.rs` + `types/extension.rs` became re-export shims; `Shape`
  dropped (unused). 8 new manifest error variants.

## [0.1.156] — 2026-07-19 · ARTX02: package layer

- **Wave 1 `package`:** `PackageLayout::discover` — manifest/shared/layers/
  projector/checksums path resolution, numeric layer sort, loud errors on
  non-numeric layer names. ZIP `.gllm` archives detected but deferred to the
  mmap work (ARTX06).
- **Wave 2 `execution_unit`:** 16-byte GLLM header parse/serialize,
  header-only `ExecutionUnit::open`. ARTX01 manifest-level `ExecutionUnit`
  renamed `ExecutionUnitMeta`.
- **Wave 3 `checksum`:** streamed `sha256_file` (64 KiB chunks),
  `sha256sum`-format `checksums.sha256` parser, `ChecksumVerifier` with
  non-short-circuiting `verify_all`. No `hex` crate — manual encoding.
- **Wave 4 `shared`:** `SharedComponents` structural validation with
  optional checksum verify.
- **Wave 5:** `GllmPackage` top-level handle — lazy, cached `open_layer`;
  `verify_integrity`. ARTX01 `GllmPackage` renamed `GllmPackageMeta`.

## [0.1.155] — 2026-07-18 · ARTX01: boilerplate

- Initial crate: constants (magic `0x474C4C4D`, dtype codes, 64-byte
  alignment), `GllmError`, placeholder types, object-safe `GllmRuntime` /
  `LayerPlugin` / `GllmConverter` traits. Zero workspace deps by design;
  edition 2024. 14 tests.
