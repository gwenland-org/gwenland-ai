---
title: GllmPackageMeta jadi redundan setelah ARTX03
status: open
severity: low
found: 2026-07-19
blocking: none (cleanup)
files:
  - glictus-caliburni/src/types/package.rs
  - glictus-caliburni/src/traits/runtime.rs
  - glictus-caliburni/src/traits/converter.rs
---

# Problem

`GllmPackageMeta` (eks-`GllmPackage` ARTX01 di
[types/package.rs](../glictus-caliburni/src/types/package.rs)) adalah wrapper
`{root, manifest, format}` dengan helper path. Setelah ARTX03, `GllmPackage`
yang asli sudah memuat manifest + layout + validation — semua informasi
`GllmPackageMeta` adalah subset-nya. Yang masih memakai `GllmPackageMeta`:

- `GllmRuntime::load_package(&GllmPackageMeta)` di traits/runtime.rs
- `GllmConverter::convert/validate` di traits/converter.rs
- tests `object_safety.rs` + `spec_artx01.rs` (expected_layer_filename)

# Plan

1. Di ARTX05 (runtime execution), saat signature `GllmRuntime` memang harus
   berubah, ganti parameter trait ke `&GllmPackage` dan hapus
   `GllmPackageMeta`.
2. `expected_layer_filename` sudah ada padanannya:
   `manifest::format_layer_filename` — test ARTX01 tinggal dialihkan.
3. Jangan hapus sekarang: object-safety suite mengunci signature trait, dan
   mengubah trait tanpa kebutuhan runtime nyata = churn tanpa nilai.
