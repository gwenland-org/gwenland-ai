# ARTX3 — Manifest Specification

## Architecture Specification

**Document Name:** ARTX3_—_Manifest_Specification.md  
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

- [Manifest](#manifest)
  - [Schema Overview](#schema-overview)
  - [Manifest Design Decisions](#manifest-design-decisions)
- [Metadata](#metadata)
  - [Model Metadata](#model-metadata)
  - [Custom Metadata](#custom-metadata)
- [Versioning](#versioning)
  - [Format Version](#format-version)
  - [Version Negotiation](#version-negotiation)
  - [Layer Type Versioning](#layer-type-versioning)
- [Extension Points](#extension-points)
  - [Extension URI Scheme](#extension-uri-scheme)
  - [Extension Registration](#extension-registration)
  - [Custom Layer Types](#custom-layer-types)
- [Related Documents](#related-documents)

---

# Manifest

The manifest is a JSON document with a strict schema. It is the only file the runtime must parse before loading any tensors.

## Schema Overview

```json
{
  "gllm_version": "1.0.0",
  "model_id": "org.gwenland.mensa-rotunda-70b",
  "architecture": "transformer",
  "parameters": 70000000000,
  "quantization": "Q4_K_M",
  "metadata": {
    "vocab_size": 32000,
    "context_length": 8192,
    "embedding_length": 8192,
    "num_layers": 80,
    "num_heads": 64,
    "head_count_kv": 8,
    "rope_dims": 128,
    "rope_freq_base": 10000.0
  },
  "tokenizer": {
    "file": "GLLMTokenizer.gllm",
    "checksum": "sha256:9f8e7d...",
    "model": "bpe",
    "pre": "qwen2",
    "vocab_size": 32000,
    "bos_id": 1,
    "eos_id": 2,
    "add_bos": true
  },
  "shared": {
    "file": "GLLMShared.gllm",
    "checksum": "sha256:abc123...",
    "tensors": [
      {
        "name": "token_embeddings",
        "shape": [32000, 8192],
        "dtype": "Q4_K",
        "offset": 0,
        "size": 131072000
      }
    ]
  },
  "layers": [
    {
      "index": 0,
      "file": "GLLMTensorLayer-0000.gllm",
      "checksum": "sha256:def456...",
      "type": "gllm:transformer/standard@v1",
      "tensors": [
        {
          "name": "attn_q.weight",
          "shape": [8192, 8192],
          "dtype": "Q4_K",
          "offset": 0,
          "size": 16777216
        },
        {
          "name": "attn_k.weight",
          "shape": [1024, 8192],
          "dtype": "Q4_K",
          "offset": 16777216,
          "size": 2097152
        }
      ],
      "device": "cuda:0"
    }
  ],
  "projector": {
    "file": "GLLMProj.gllm",
    "checksum": "sha256:ghi789...",
    "type": "gllm:projector/linear@v1"
  },
  "extensions": [
    "gllm:transformer/standard@v1",
    "gllm:projector/linear@v1"
  ]
}
```

## Manifest Design Decisions

1. **Single JSON File.** The manifest is a single document to ensure atomic parsing. No includes, no fragments.
2. **Tensor Metadata Only.** The manifest lists tensor names, shapes, dtypes, offsets, and sizes. It does not contain tensor data.
3. **Checksum per Execution Unit.** Every file referenced by the manifest has a SHA-256 checksum. The runtime verifies the checksum before mapping the file.
4. **Device Map Embedded.** The manifest may specify a default device per layer. The runtime may override this based on available hardware.
5. **Extension Registry.** The manifest lists all layer type URIs used in the package. The runtime loads plugins for these URIs before execution.

---

# Metadata

## Model Metadata

Model-level metadata is stored in the manifest `metadata` object. It includes architecture hyperparameters required for correct execution.

Required fields:

| Field | Type | Description |
|-------|------|-------------|
| `vocab_size` | int | Size of the vocabulary |
| `context_length` | int | Maximum sequence length |
| `embedding_length` | int | Hidden dimension size |
| `num_layers` | int | Number of layers in the model |
| `num_heads` | int | Number of attention heads |
| `head_count_kv` | int | Number of key/value heads (for GQA) |

Optional fields:

| Field | Type | Description |
|-------|------|-------------|
| `rope_dims` | int | RoPE dimensionality |
| `rope_freq_base` | float | RoPE frequency base |
| `rope_scaling` | object | RoPE scaling configuration (type, factor) |
| `expert_count` | int | Number of experts (MoE) |
| `expert_used_count` | int | Number of active experts per token |
| `sliding_window` | int | Sliding window attention size |
| `attention_bias` | bool | Whether attention uses bias |

## Tokenizer Descriptor

The manifest's `tokenizer` object points at `GLLMTokenizer.gllm` and carries
just enough to identify and verify it. The vocabulary, merges, token types and
chat template live **in the unit, not here** — inlining a 152 000-entry
vocabulary would defeat the manifest's guarantee of being parseable without
loading tensors.

| Field | Type | Description |
|-------|------|-------------|
| `file` | string | Unit filename, normally `GLLMTokenizer.gllm` |
| `checksum` | string | `"sha256:<hex>"` over the unit |
| `model` | string | Tokenizer algorithm, e.g. `bpe`, `spm` |
| `pre` | string | Pre-tokenizer dialect, e.g. `qwen2`, `llama3` |
| `vocab_size` | int | Must equal `metadata.vocab_size` |
| `bos_id`, `eos_id` | int | Special token ids |
| `add_bos` | bool | Whether BOS is prepended by default |
| `padding_id` | int | Optional padding token id |

`tokenizer.vocab_size` duplicating `metadata.vocab_size` is intentional: the
two are produced from different sources during conversion, and a mismatch
between them is a converter bug worth failing on rather than a value worth
deriving. Validators MUST reject a package where they disagree.

The `tokenizer` object is REQUIRED for any package intended for text
generation. A package MAY omit it — an embedding-only or vision-encoder
package has no vocabulary — in which case the runtime MUST refuse tokenization
requests rather than fall back to an external tokenizer.

For the unit's contents and the rationale for embedding it, see ARTX2: Package
Specification, "Tokenizer Unit".

## Custom Metadata

The manifest may contain a `custom_metadata` object for user-defined key-value pairs. These are not interpreted by the runtime but are preserved for tooling and provenance.

---

# Versioning

## Format Version

The GLLM format version follows semantic versioning: `MAJOR.MINOR.PATCH`.

- **MAJOR:** Incompatible structural changes (e.g., manifest schema revision, layer file header change).
- **MINOR:** Backward-compatible additions (e.g., new dtype codes, new optional manifest fields).
- **PATCH:** Clarifications and corrections to the specification.

## Version Negotiation

The runtime reads the `gllm_version` field from the manifest. If the runtime supports the major version but not the minor version, it may proceed with a warning. If the major version is unsupported, the runtime must refuse to load the package.

## Layer Type Versioning

Layer type URIs include a version suffix (e.g., `@v1`). The runtime plugin system matches the exact version. A runtime with a `v2` plugin for `gllm:transformer/standard` cannot execute a `v1` layer unless a compatibility shim is registered.

For source format compatibility, see ARTX9: Compatibility & Versioning. For converter version handling, see ARTX7: Converter Architecture.

---

# Extension Points

## Extension URI Scheme

Layer types are identified by URIs of the form:

```
gllm:<category>/<name>@<version>
```

Examples:

- `gllm:transformer/standard@v1`
- `gllm:transformer/moe@v1`
- `gllm:mamba/standard@v1`
- `gllm:projector/linear@v1`
- `gllm:adapter/lora@v1`

## Extension Registration

The manifest lists all extension URIs in the `extensions` array. The runtime loads the corresponding plugin for each URI before executing the package. Plugins are shared libraries or runtime modules that implement:

- Tensor layout parsing for the layer type
- Forward pass execution logic
- Memory allocation strategy for the layer type

```mermaid
graph TD
    A[Manifest] --> B[Extension List]
    B --> C[gllm:transformer/standard@v1]
    B --> D[gllm:mamba/standard@v1]
    C --> E[Plugin Loader]
    D --> E
    E --> F[Runtime Plugin Registry]
    F --> G[Execution Engine]
```

## Custom Layer Types

Users may define custom layer types by registering a new URI and providing a runtime plugin. The GLLM specification does not restrict custom URIs, but recommends the `vendor:category/name@version` convention for third-party extensions.

For plugin interface details, see ARTX8: Extension System. For layer file binary format, see ARTX4: Layer Specification.

---

## Related Documents

| Document | Title | Description |
|----------|-------|-------------|
| ARTX1 | GLLM Overview | Entry-point overview of the GLLM format, vision, and ecosystem |
| ARTX2 | Package Specification | Physical and logical package structure, archive format, execution units |
| **ARTX3** | **Manifest Specification** | **This document** |
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
