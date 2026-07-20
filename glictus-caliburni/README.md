# glictus-caliburni

> **Ictus Caliburni** â€” GLLM (GwenLand Language Model Format)
> *Designed for the Impossible.*

Part of the GwenLand AI ecosystem.

## What is GLLM?

GLLM is a layer-native, correctness-first binary format for LLM inference.
Unlike GGUF (flat namespace) or Safetensors (no architecture metadata),
GLLM treats each transformer layer as an independent execution unit.

| Format | Layer Boundaries | Architecture Metadata | Checksums | mmap-native |
|--------|-----------------|----------------------|-----------|-------------|
| GGUF | âŒ | Partial | âŒ | âœ… |
| Safetensors | âŒ | âŒ | âŒ | âœ… |
| **GLLM** | **âœ…** | **âœ…** | **âœ…** | **âœ…** |

## Package Structure

```
model.gllm/
â”œâ”€â”€ gllm.json           â† Manifest (single source of truth)
â”œâ”€â”€ GLLMShared.gllm         â† Token embeddings, output head, norms
â”œâ”€â”€ GLLMTensorLayer-0000.gllm      â† One file per layer
â”œâ”€â”€ GLLMTensorLayer-0001.gllm
â”œâ”€â”€ ...
â”œâ”€â”€ GLLMProj.gllm      â† Optional multimodal projector
â””â”€â”€ checksums.sha256    â† Optional aggregated checksums
```

## Design Principles

1. **Storage Follows Execution** â€” `GLLMTensorLayer-0000.gllm` through `GLLMTensorLayer-0079.gllm`
2. **Metadata Is Executable** â€” `gllm.json` is a machine-readable contract
3. **Fail Fast, Fail Loud** â€” SHA-256 per file, detected at load time
4. **Extensibility Over Generality** â€” URI-based plugin system

## Status

ARTX01 (types/traits boilerplate) + ARTX02 (package-level abstractions) +
ARTX03 (manifest parsing & validation) + ARTX04 (layer binary format:
hybrid 16-byte header with tensor_count, tensor index codec,
`LayerFile::read`, `write_unit_file`, manifest cross-check) + ARTX07-lite
(GGUF â†’ GLLM converter: `converter` module + `glconv` bin behind the
`converter` feature â€” the only feature that pulls a workspace dep,
`glcore`, for GGUF parsing; the default build stays zero-workspace-dep).

What exists today:

- `constants` â€” magic bytes, format version, dtype codes, alignment
- `error` â€” `GllmError` / `GllmResult`
- `manifest` â€” `GllmManifest` (full `gllm.json` parsing), `ModelMetadata`,
  `FormatVersion`, `TensorEntry`, `DType`, `ExtensionUri`,
  `ManifestValidator` (rules V01â€“V17, errors + warnings)
- `types` â€” `Device`, `DeviceMap`, `ExecutionUnitMeta`, `GllmPackageMeta`,
  `LayerFile` (ARTX01 leftovers; tensor/extension/manifest are shims into
  `manifest`)
- `traits` â€” `GllmRuntime`, `LayerPlugin`, `GllmConverter` (definitions only;
  no implementations ship in this crate)
- `package` â€” `GllmPackage::open` = discover + parse manifest + validate +
  checksum-verify `GLLMShared.gllm`; lazy per-layer opens; integrity sweep
- `execution_unit` â€” 16-byte file header parse/serialize, header-only `open`
- `checksum` â€” streamed SHA-256, `checksums.sha256` parsing, verifier
- `shared` â€” `SharedComponents` structural validation for `GLLMShared.gllm`

What does **not** exist yet: reading tensor *data* (only locating it),
quantization block interpretation (runtime plugin scope), runtime execution
(ARTX05), and mmap + ZIP-archive reading (ARTX06). The traits describe the
intended contract â€” they are not backed by working code in this crate.
Known open problems are tracked in [`../notes/`](../notes/).

## Dependencies

Zero external ML dependencies, per GwenLand philosophy â€” only `serde`,
`serde_json`, `sha2`, and `thiserror`.

## Architecture Spec

Based on the Mensa Rotunda ARTX series. This crate implements ARTX01
(format overview); ARTX02â€“ARTX11 remain to be built on top of it.
