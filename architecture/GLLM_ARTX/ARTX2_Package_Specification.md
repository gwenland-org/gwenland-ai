# ARTX2 — Package Specification

## Architecture Specification

**Document Name:** ARTX2_—_Package_Specification.md  
**Codename:** Mensa Rotunda  
**Tagline:** Designed for the Impossible.  
**Version:** 1.0.0-draft  
**Date:** 2026-07-10  
**Status:** Draft  

---

## Revision History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0.0-draft | 2026-07-10 | GLLM Core Team | Extracted from ArchGLLMFormat.md master document |

---

## Table of Contents

- [Package Structure](#package-structure)
  - [Directory Layout](#directory-layout)
  - [Archive Format](#archive-format)
- [Execution Units](#execution-units)
- [Shared Components](#shared-components)
  - [Typical Shared Tensors](#typical-shared-tensors)
- [Checksums](#checksums)
  - [Integrity Model](#integrity-model)
  - [Verification Flow](#verification-flow)
- [Related Documents](#related-documents)

---

# Package Structure

## Directory Layout

```
model.gllm/
├── gllm.json
├── shared.gllm
├── layer_000.gllm
├── layer_001.gllm
├── ...
├── layer_079.gllm
├── projector.gllm
└── checksums.sha256
```

## Archive Format

GLLM packages may be distributed as uncompressed ZIP archives (`.gllm` extension) or as directories. The ZIP format is used for portability; the directory format is used for runtime execution. The runtime extracts the manifest and checksums from the ZIP header without decompressing layer files.

> **Rationale:** ZIP stores file offsets in the central directory, allowing the runtime to locate `gllm.json` and individual layer files via `seek`. Uncompressed stored files within the ZIP are directly mmap-able.

```mermaid
graph TD
    A[GLLM Package] --> B[Directory]
    A --> C[ZIP Archive]
    B --> D[gllm.json]
    B --> E[shared.gllm]
    B --> F[layer_*.gllm]
    C --> G[Local File Header]
    C --> H[Central Directory]
    C --> I[Stored Files]
```

For manifest schema details, see ARTX3: Manifest Specification. For layer file binary format, see ARTX4: Layer Specification. For runtime loading behavior, see ARTX5: Runtime Architecture and ARTX6: Memory Model.

---

# Execution Units

An execution unit is any file that the runtime can load, verify, and map independently. This includes:

- `shared.gllm`
- `layer_NNN.gllm`
- `projector.gllm`
- Adapter layer files (future)

Each execution unit has:

- A file path (relative to manifest)
- A SHA-256 checksum
- A layer type URI (for layer files)
- A tensor index (names, shapes, offsets, dtypes)

```mermaid
graph LR
    A[Execution Unit] --> B[Header]
    A --> C[Tensor Index]
    A --> D[Tensor Data]
    B --> E[Magic: GLLM]
    B --> F[Version]
    B --> G[Flags]
    C --> H[Tensor Count]
    C --> I[Tensor Entries]
    D --> J[Contiguous Buffers]
```

---

# Shared Components

Shared components are tensors used by multiple layers or the final output projection. They are stored in `shared.gllm`.

## Typical Shared Tensors

| Tensor | Shape | Purpose |
|--------|-------|---------|
| `token_embeddings` | [V, D] | Input embedding lookup |
| `output_norm.weight` | [D] | Final layer normalization |
| `output_head.weight` | [V, D] | Logit projection (often tied to embeddings) |
| `rope_cos` | [C, R] | Precomputed RoPE cosine table |
| `rope_sin` | [C, R] | Precomputed RoPE sine table |

> **Rationale:** Separating shared components prevents duplication in layer files and enables the runtime to keep them permanently mapped while layers are swapped.

---

# Checksums

## Integrity Model

GLLM uses SHA-256 for all integrity verification. Checksums are applied at three levels:

1. **Per-File Checksum:** Every execution unit file has a SHA-256 checksum in the manifest.
2. **Manifest Checksum:** The manifest itself may be accompanied by a `gllm.json.sha256` file.
3. **Aggregated Checksums:** The optional `checksums.sha256` file contains all file checksums for offline verification.

## Verification Flow

```mermaid
sequenceDiagram
    participant R as Runtime
    participant M as Manifest
    participant F as File System
    R->>M: Parse manifest
    R->>F: Open shared.gllm
    R->>R: Compute SHA-256
    R->>M: Compare with manifest
    alt Checksum Mismatch
        R->>R: Raise IntegrityError
    else Checksum Match
        R->>F: mmap shared.gllm
    end
    R->>F: Open layer_000.gllm
    R->>R: Compute SHA-256
    R->>M: Compare with manifest
    R->>F: mmap layer_000.gllm
```

> **Rationale:** Per-file checksums enable the runtime to detect corruption at the granularity of execution units. A corrupted layer file does not invalidate the entire package.

---

## Related Documents

| Document | Title | Description |
|----------|-------|-------------|
| ARTX1 | GLLM Overview | Entry-point overview of the GLLM format, vision, and ecosystem |
| **ARTX2** | **Package Specification** | **This document** |
| ARTX3 | Manifest Specification | gllm.json schema, metadata, versioning, and extension registry |
| ARTX4 | Layer Specification | Binary layer file format, tensor layouts, and extension layer types |
| ARTX5 | Runtime Architecture | Execution model, scheduler, CPU/GPU/hybrid runtime, and failure recovery |
| ARTX6 | Memory Model | Address space layout, memory lifecycle, prefetch strategy, and KV cache |
| ARTX7 | Converter Architecture | Conversion pipeline, parsers, validation, and error handling |
| ARTX8 | Extension System | Plugin architecture, MLA, Mamba, MoE, and custom layer types |
| ARTX9 | Compatibility & Versioning | Format versioning, source compatibility, and known limitations |
| ARTX10 | Distributed Runtime | Multi-node execution, pipeline/tensor parallelism, and recovery |
| ARTX11 | Benchmarks | Benchmark suite, methodology, and measurement standards |

---

*This document is part of the GLLM Architecture Specification Series. The master document is ArchGLLMFormat.md.*
