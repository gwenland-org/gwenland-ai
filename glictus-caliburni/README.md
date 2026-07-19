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
├── shared.gllm         ← Token embeddings, output head, norms
├── layer_000.gllm      ← One file per layer
├── layer_001.gllm
├── ...
├── projector.gllm      ← Optional multimodal projector
└── checksums.sha256    ← Optional aggregated checksums
```

## Design Principles

1. **Storage Follows Execution** — `layer_000.gllm` through `layer_079.gllm`
2. **Metadata Is Executable** — `gllm.json` is a machine-readable contract
3. **Fail Fast, Fail Loud** — SHA-256 per file, detected at load time
4. **Extensibility Over Generality** — URI-based plugin system

## Status

ARTX01 (types/traits boilerplate) + ARTX02 (package-level abstractions) +
ARTX03 (manifest parsing & validation) + ARTX04 (layer binary format:
hybrid 16-byte header with tensor_count, tensor index codec,
`LayerFile::read`, `write_unit_file`, manifest cross-check).

What exists today:

- `constants` — magic bytes, format version, dtype codes, alignment
- `error` — `GllmError` / `GllmResult`
- `manifest` — `GllmManifest` (full `gllm.json` parsing), `ModelMetadata`,
  `FormatVersion`, `TensorEntry`, `DType`, `ExtensionUri`,
  `ManifestValidator` (rules V01–V17, errors + warnings)
- `types` — `Device`, `DeviceMap`, `ExecutionUnitMeta`, `GllmPackageMeta`,
  `LayerFile` (ARTX01 leftovers; tensor/extension/manifest are shims into
  `manifest`)
- `traits` — `GllmRuntime`, `LayerPlugin`, `GllmConverter` (definitions only;
  no implementations ship in this crate)
- `package` — `GllmPackage::open` = discover + parse manifest + validate +
  checksum-verify `shared.gllm`; lazy per-layer opens; integrity sweep
- `execution_unit` — 16-byte file header parse/serialize, header-only `open`
- `checksum` — streamed SHA-256, `checksums.sha256` parsing, verifier
- `shared` — `SharedComponents` structural validation for `shared.gllm`

What does **not** exist yet: reading tensor *data* (only locating it),
quantization block interpretation (runtime plugin scope), runtime execution
(ARTX05), and mmap + ZIP-archive reading (ARTX06). The traits describe the
intended contract — they are not backed by working code in this crate.
Known open problems are tracked in [`../notes/`](../notes/).

## Dependencies

Zero external ML dependencies, per GwenLand philosophy — only `serde`,
`serde_json`, `sha2`, and `thiserror`.

## Architecture Spec

Based on the Mensa Rotunda ARTX series. This crate implements ARTX01
(format overview); ARTX02–ARTX11 remain to be built on top of it.
