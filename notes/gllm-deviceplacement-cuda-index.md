---
title: DevicePlacement manifest cuma kenal cuda:0 dan cuda:1
status: open
severity: medium
found: 2026-07-19
blocking: ARTX05 (runtime execution) kalau mau multi-GPU > 2
files:
  - glictus-caliburni/src/manifest/mod.rs
  - glictus-caliburni/src/types/execution.rs
---

# Problem

Spec ARTX03 mendefinisikan `DevicePlacement` sebagai enum serde dengan varian
hardcoded `cuda:0` / `cuda:1`, dan saya implement persis begitu di
[manifest/mod.rs](../glictus-caliburni/src/manifest/mod.rs). Konsekuensinya:

- `"cuda:2"`, `"cuda:3"`, `"vulkan:1"`, `"rank:0/cuda:0"` di manifest jatuh ke
  `DevicePlacement::Unknown` — **string aslinya hilang** (enum, bukan newtype),
  jadi runtime tidak bisa recover device mana yang diminta.
- Redundan dengan `types::execution::Device` (ARTX01) yang justru sudah bisa
  parse `cuda:N` / `vulkan:N` arbitrer + `Remote{rank}`.

Ini deviasi yang saya ikuti dari spec text, bukan bug implementasi.

# Plan

1. Usulan ke spec: ganti `DevicePlacement` jadi string newtype transparan
   (pola sama seperti `ExtensionUri`) dengan method `to_device() ->
   Option<Device>` yang delegasi ke `Device::from_str` — arbitrer index
   selamat, serde tetap plain string, dan satu-satunya parser device ada di
   satu tempat.
2. Kalau spec mau tetap enum, minimal tambahkan varian `Other(String)` via
   `#[serde(untagged)]` supaya string tidak hilang.
3. Sampai diputuskan: JANGAN pakai `effective_device()` untuk placement
   multi-GPU nyata; treat `Unknown` sebagai "tanya manifest string mentah"
   (yang sekarang belum disimpan — makanya ini masalah).
