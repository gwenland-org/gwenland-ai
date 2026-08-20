# glictus-caliburni

> **Ictus Caliburni** — GLLM (GwenLand Language Model Format)
> *Designed for the Impossible.*

Part of the GwenLand AI ecosystem.

## What is GLLM?

GLLM is a layer-native, correctness-first binary format for LLM inference.
Unlike GGUF (flat namespace) or Safetensors (no architecture metadata),
GLLM treats each transformer layer as an independent execution unit.

| Format | Layer Boundaries | Architecture Metadata | Checksums | mmap-native |
|--------|-----------------|----------------------|-----------|-------------|
| GGUF | ❌ | Partial | ❌ | ✅ |
| Safetensors | ❌ | ❌ | ❌ | ✅ |
| **GLLM** | **✅** | **✅** | **✅** | **✅** |

## Package Structure

```
model.gllm/
├── gllm.json           ← Manifest (single source of truth)
├── GLLMShared.gllm         ← Token embeddings, output head, norms
├── GLLMTensorLayer-0000.gllm      ← One file per layer
├── GLLMTensorLayer-0001.gllm
├── ...
├── GLLMProj.gllm      ← Optional multimodal projector
└── checksums.sha256    ← Optional aggregated checksums
```

A package can also be an **uncompressed** ZIP archive with the same layout
inside it (files stored, not deflated, so they stay directly mmap-able). Only
the directory form is actually readable today — see "What does not exist yet"
below.

## Design Principles

1. **Storage Follows Execution** — `GLLMTensorLayer-0000.gllm` through `GLLMTensorLayer-0079.gllm`
2. **Metadata Is Executable** — `gllm.json` is a machine-readable contract
3. **Fail Fast, Fail Loud** — SHA-256 per file, detected at load time
4. **Extensibility Over Generality** — URI-based plugin system

## Status

ARTX01 (types/traits) + ARTX02 (package-level abstractions) + ARTX03
(manifest parsing & validation) + ARTX04 (layer binary format: hybrid 16-byte
header, tensor index codec, `LayerFile::read`, manifest cross-check) +
ARTX07-lite (GGUF → GLLM converter) + ARTX08 (layer-type extension registry,
in-process only — no `dlopen`, see `src/plugin.rs`) + ARTX09 (format
versioning hardening) + **ARTX05/06 (CPU runtime)**: a layer-sequential
executor that maps one layer at a time, hands it to a backend, and unmaps it,
keeping the working set at roughly one layer plus the shared components —
plus ARTX10 Wave 1, an execution backend adapter over glproc's per-op
kernels.

What exists today:

- `constants` — magic bytes, format version, dtype codes, alignment
- `error` — `GllmError` / `GllmResult`
- `manifest` — `GllmManifest` (full `gllm.json` parsing), `ModelMetadata`,
  `FormatVersion`, `TensorEntry`, `DType`, `ExtensionUri`,
  `ManifestValidator` (rules V01–V17, errors + warnings)
- `types` — `Device`, `DeviceMap`, `ExecutionUnitMeta`, `GllmPackageMeta`,
  `LayerFile` (ARTX01 leftovers; tensor/extension/manifest are shims into
  `manifest`)
- `traits` — `GllmRuntime`, `LayerPlugin`, `GllmConverter` (definitions;
  `runtime` below is the first working implementation)
- `package` — `GllmPackage::open` = discover + parse manifest + validate +
  checksum-verify `GLLMShared.gllm`; lazy per-layer opens; integrity sweep
- `execution_unit` — 16-byte file header parse/serialize, header-only `open`
- `layer_io` — tensor index reading over a parsed layer file
- `checksum` — streamed SHA-256, `checksums.sha256` parsing, verifier
- `shared` — `SharedComponents` structural validation for `GLLMShared.gllm`
- `plugin` — the URI-based extension-loading contract
- `gquant` — Pridwen G-Quant (`DType::GQ4A`) block definitions, no encoder
- `converter` — GGUF → GLLM conversion, including the G-Quant re-encode policy
  (`gquant_policy`); ships the `glconv` binary
- `runtime` — the CPU execution path: `mmap` (validated, page-cache-backed
  layer mapping), `backend` (the `ExecutionBackend` trait), `cpu` /
  `glproc_backend` (the glproc adapter), `kv_cache`, `scheduler`, `device`,
  `gllm_engine`. `distributed` is present as a module but has no working
  multi-device path yet.

What does **not** exist yet:

- **GPU or hybrid execution.** The `runtime` module structure has room for it
  (`ExecutionBackend` is the seam), but the GPU path is deliberately deferred
  until it can follow glcuda's runtime dynamic-loading pattern rather than
  pull in a build-time CUDA dependency.
- **ZIP-archive packages.** `PackageFormat::ZipArchive` is detected at
  discovery time and explicitly rejected — `"ZIP-archive packages are
  detected but not yet readable (ZIP central-directory parsing lands with the
  mmap work, ARTX06)"` — rather than silently mishandled. Only directory
  packages are readable today.
- **An embedded tokenizer.** ARTX1 OQ3's `GLLMTokenizer.gllm` unit is decided
  but not yet emitted by the converter, which is why `glbench --engine gllm`
  has to synthesize token ids instead of encoding real prompt text (see
  [`glbench/README.md`](../glbench/README.md#benchmarking-gllm)).

Known open problems are tracked in [`../notes/`](../notes/) — each note
carries its own `status` field (most opened during ARTX03–ARTX05 are already
`resolved`; check the file before assuming an open problem is still open).

## Dependencies

Unconditionally: `serde`, `serde_json`, `sha2`, `thiserror`, and `memmap2` —
the last is a thin OS mmap wrapper for the ARTX05/06 runtime, not an ML
dependency. `glcore` and `glproc` are pulled in only behind the optional
`converter` and `glproc-backend` features, so the format/runtime library
itself stays workspace-dependency-free by default. Per GwenLand's zero-ML-
dependency rule: no ML framework, in any feature combination, ever.

## Architecture Spec

Based on the Mensa Rotunda ARTX series. This crate implements ARTX01–ARTX10
(ARTX07 as its "lite" GGUF-converter scope, ARTX10 as its Wave 1 CPU-only
scope) plus the layer-type extension registry from ARTX08. ARTX11 (GLLM as a
benchmarkable engine) lives in `glbench`, not here — see its README.
