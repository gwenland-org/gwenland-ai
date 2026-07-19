---
title: Dua definisi binary header yang bersaing di glictus-caliburni
status: open
severity: high
found: 2026-07-19
blocking: ARTX04 (layer binary format)
files:
  - glictus-caliburni/src/types/layer.rs
  - glictus-caliburni/src/execution_unit.rs
  - glictus-caliburni/src/constants.rs
---

# Problem

Crate `glictus-caliburni` sekarang punya **dua definisi header file binary yang
saling bertentangan**, dua-duanya hidup dan dites:

| | `LayerHeader` (ARTX01) | `ExecutionUnitHeader` (ARTX02) |
|---|---|---|
| File | [types/layer.rs](../glictus-caliburni/src/types/layer.rs) | [execution_unit.rs](../glictus-caliburni/src/execution_unit.rs) |
| Ukuran | 12 byte | 16 byte |
| Magic | u32 `0x474C4C4D` | 4 byte `b"GLLM"` (nilai sama) |
| Versi | major u8 + minor u8 | single u16 little-endian |
| Ekstra | flags u16 + tensor_count u32 | flags u16 + 8 byte reserved (wajib nol) |

`ExecutionUnitHeader` adalah yang benar-benar dipakai runtime path
(`ExecutionUnit::open`, `SharedComponents::open`, `GllmPackage::open`).
`LayerHeader` tidak dibaca dari disk oleh siapapun — hanya dikonstruksi
in-memory dan divalidasi di test `spec_artx01::test_layer_header_validate`.

# Kenapa belum dibereskan

Rekonsiliasi butuh keputusan spec ARTX04 (layer binary format): apakah layer
file punya sub-header sendiri (tensor_count, dst) SETELAH 16-byte
ExecutionUnitHeader, atau `LayerHeader` memang mati.

# Plan

1. Saat mengerjakan ARTX04, putuskan: kemungkinan besar `LayerHeader` ARTX01
   dihapus dan diganti "layer section header" yang duduk setelah 16-byte
   header universal (tensor_count pindah ke situ; index tensor per ARTX04).
2. Migrasi `flags` bitmask (`types/layer.rs::flags`) ke tempat baru atau hapus.
3. Update `constants.rs` — `GLLM_VERSION_MAJOR/MINOR` (u8+u8) menyisakan model
   versi lama; ExecutionUnitHeader pakai u16 tunggal. Satukan ke satu sumber.
4. Test `test_layer_header_validate` di `tests/spec_artx01.rs` di-update /
   diganti test format ARTX04 (regression test ARTX01 boleh mati di sini
   karena kontraknya memang diganti spec — declare di gate).
