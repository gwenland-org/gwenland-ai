# ARTX9 — Compatibility & Versioning

## Architecture Specification

**Document Name:** ARTX9_Compatibility_Versioning.md  
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

- [Versioning](#versioning)
  - [Format Version](#format-version)
  - [Version Negotiation](#version-negotiation)
  - [Layer Type Versioning](#layer-type-versioning)
- [Compatibility](#compatibility)
  - [Source Format Versions](#source-format-versions)
  - [Target Format Versions](#target-format-versions)
- [Known Limitations](#known-limitations)
- [Related Documents](#related-documents)

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

For manifest version fields, see ARTX3: Manifest Specification, Section 3. For plugin versioning, see ARTX8: Extension System, Section 5.3.

---

# Compatibility

## Source Format Versions

The converter supports the following source format versions:

| Source Format | Supported Versions | Notes |
|---------------|-------------------|-------|
| GGUF | 3.0+ | Full support for all GGML quantization types |
| Safetensors | All | Requires external metadata for architecture info |
| PyTorch | 1.9+ | Requires `torch.load` compatibility |
| ONNX | 1.10+ | Limited to static transformer graphs |

## Target Format Versions

The converter generates packages conforming to the GLLM format version specified by the `--format-version` flag. The default is the latest stable version.

For converter architecture details, see ARTX7: Converter Architecture, Section 7.

---

# Known Limitations

1. **Single-Threaded Layer Execution:** The current runtime executes layers sequentially within a single thread. Inter-layer parallelism is not supported.
2. **Limited GPU Kernel Fusion:** Only basic fusion (norm+GEMM) is implemented. Full attention fusion is pending.
3. **No Dynamic Batching:** The runtime processes one sequence at a time. Batch inference requires multiple runtime instances.
4. **Converter Coverage:** The converter only supports GGUF as a primary source. Safetensors and PyTorch support require manual metadata specification.
5. **Windows Support:** The runtime is developed and tested on Linux. Windows support is untested.

For future work addressing these limitations, see ARTX1: GLLM Overview, Section 3.4.

---

## Related Documents

| Document | Title | Description |
|----------|-------|-------------|
| ARTX1 | GLLM Overview | Entry-point overview of the GLLM format, vision, and ecosystem |
| ARTX2 | Package Specification | Physical and logical package structure, archive format, execution units |
| ARTX3 | Manifest Specification | gllm.json schema, metadata, versioning, and extension registry |
| ARTX4 | Layer Specification | Binary layer file format, tensor layouts, and extension layer types |
| ARTX5 | Runtime Architecture | Execution model, scheduler, CPU/GPU/hybrid runtime, and failure recovery |
| ARTX6 | Memory Model | Address space layout, memory lifecycle, prefetch strategy, and KV cache |
| ARTX7 | Converter Architecture | Conversion pipeline, parsers, validation, and error handling |
| ARTX8 | Extension System | Plugin architecture, MLA, Mamba, MoE, and custom layer types |
| **ARTX9** | **Compatibility & Versioning** | **This document** |
| ARTX10 | Distributed Runtime | Multi-node execution, pipeline/tensor parallelism, and recovery |
| ARTX11 | Benchmarks | Benchmark suite, methodology, and measurement standards |

---

*This document is part of the GLLM Architecture Specification Series. The master document is ArchGLLMFormat.md.*
