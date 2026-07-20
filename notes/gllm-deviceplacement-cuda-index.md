---
title: DevicePlacement manifest cuma kenal cuda:0 dan cuda:1
status: resolved
severity: medium
found: 2026-07-19
resolved: 2026-07-20
resolution: opsi 2 (varian Other(String)) — dipilih JinXSuper saat ARTX05 Phase 4
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

# Bukti tambahan dari seri spec lengkap (2026-07-19)

Setelah ZIP spec dibaca: ARTX6 §Device Mapping memakai skema **berbeda lagi** —
objek `device_map` dengan range (`{"default": "cuda:0", "layers": {"0-39":
"cuda:0", "40-79": "cuda:1"}}`), dan ARTX10 memakai string `rank:0/cuda:0`
untuk distributed. Tiga representasi device yang saling tidak kompatibel di
satu seri spec (enum ARTX3, range-map ARTX6, rank-string ARTX10) —
memperkuat usulan newtype string transparan + satu parser terpusat
(`Device::from_str` sudah bisa `rank:`-style via `Device::Remote`).

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

# Resolusi (2026-07-20, ARTX05 Phase 4)

Diambil **opsi 2**: `DevicePlacement` jadi enum `#[serde(untagged)]` dengan dua
varian — `Known(KnownDevicePlacement)` untuk yang ARTX03 sebut eksplisit, dan
`Other(String)` yang menyimpan string apa pun secara verbatim. Varian
`Unknown` yang membuang string **dihapus**.

- `"cuda:2"` → `Other("cuda:2")` → `to_device()` = `Some(Device::Cuda(2))`.
  Selamat, dan resolusinya lewat parser yang sama dengan `cuda:0`.
- `"rank:0/cuda:0"` (ARTX10) → string selamat, `to_device()` = `None`. Runtime
  fallback ke CPU dengan `overridden: Some(...)` supaya bisa dilaporkan apa
  yang di-drop — bukan menebak.
- Wire-compatible: `"cpu"`, `"cuda:0"`, `"cuda:1"`, `"metal"`, `"vulkan"`
  deserialize dan serialize persis seperti sebelumnya (ada test).
- Satu-satunya parser device tetap `Device::from_str` — `to_device()` cuma
  delegasi, tidak ada dialek kedua.

Konstanta `DevicePlacement::CPU/CUDA0/CUDA1` disediakan supaya call site tidak
perlu menulis `Known(KnownDevicePlacement::Cpu)`.

Usulan newtype transparan (opsi 1) **tidak** diambil: lebih bersih secara
teori tapi mengubah tipe publik yang sudah dipakai ARTX02–04 di tengah
implementasi runtime. Kalau spec ARTX03 nanti direvisi, migrasi ke newtype
masih terbuka — `as_str()`/`to_device()` sudah jadi permukaan yang sama.
