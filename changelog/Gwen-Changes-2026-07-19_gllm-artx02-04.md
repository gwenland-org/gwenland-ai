# GwenLand - 2026-07-19: glictus-caliburni — GLLM ARTX02–ARTX04 (Package, Manifest, Layer Binary)

**Date:** 2026-07-19 (WIB / SEAST)
**Scope:**
- Crate: `glictus-caliburni/` v0.1.155 → v0.1.160 (11 commits on `feat/gllm-artx02-package`)
- New modules: `package`, `execution_unit`, `checksum`, `shared`, `manifest/{types,metadata,validator}`, `layer_io`
- Docs: crate `CHANGELOG.md`, `README.md` status; problem notes in `notes/gllm-*.md`
**Type:** Format library build-out — the `.gllm` model packaging format, from placeholder types to a readable/writable binary format.
**Status:** Implemented; 120 tests green (from 14); `cargo clippy --all-targets -- -D warnings` clean. Dev box (i3-1115G4, Windows 11). No real converted model yet — converter is ARTX07.

---

## Executive Summary

Three ARTX specs landed back-to-back on one branch, taking GLLM from "types
that describe a format" to "a format you can actually write and read back":

- **ARTX02 (package):** `PackageLayout::discover`, 16-byte unit header,
  streamed SHA-256 verification, `SharedComponents`, and the lazy
  `GllmPackage` handle.
- **ARTX03 (manifest):** full `gllm.json` parsing into typed structs, a
  semantic validator (rules V01–V17, all findings collected), wired into
  `GllmPackage::open` — a package with a lying manifest no longer opens.
- **ARTX04 (layer binary):** tensor index codec, `LayerFile::read` (locate +
  bounds-check every tensor without reading data), `write_unit_file`, and a
  binary-vs-manifest metadata cross-check.

## Problem → Root Cause → Fix (the one real conflict)

**Problem:** ARTX04's spec defines a 12-byte layer header (major/minor u8
version, tensor_count at offset 8), but ARTX02 had already shipped a
universal 16-byte header (u16 LE version, 8 reserved bytes) used by every
runtime path and fixture.

**Root cause:** the ARTX spec series was written before implementation
started; ARTX02's megaprompt and ARTX04's document disagree about the same
bytes.

**Fix (decided by JinXSuper): hybrid.** Keep the universal 16-byte header,
fold ARTX04's `tensor_count` into bytes 8..12 (formerly reserved), reserved
shrinks to 12..16, tensor index starts at 16. For v1.0 the two version
encodings are byte-identical (`[0x01, 0x00]`), so nothing already written
breaks. Recorded caveat: they are NOT identical for v1.1+ — the encoding must
be re-decided before any minor format bump
(`notes/gllm-layerheader-vs-executionunitheader.md`).

## Declared spec deviations

- ZIP-archive packages: detected, reading deferred to ARTX06 (mmap; no `zip`
  dep under the zero-dep rule).
- No `hex` crate (manual encoding); no new deps at all beyond ARTX01's set.
- Tensor index offsets are region-relative (ARTX03 `TensorEntry` semantics),
  not ARTX04's "offset in file"; index starts at 16, not 12.
- `ExtensionUri`/`FormatVersion` are serde-transparent — deserialization does
  not validate; `parse()` and the manifest validator do.
- ARTX01 names superseded: `ExecutionUnit`→`ExecutionUnitMeta`,
  `GllmPackage`→`GllmPackageMeta`, `LayerHeader` deleted, `Shape` dropped.

## Open problems (notes/)

- `gllm-deviceplacement-cuda-index.md` — manifest enum only knows
  `cuda:0/1`; higher indices collapse to `Unknown`, original string lost.
- `gllm-gllmpackagemeta-redundancy.md` — traits still take the now-redundant
  `GllmPackageMeta`; clean up when ARTX05 changes `GllmRuntime` anyway.
