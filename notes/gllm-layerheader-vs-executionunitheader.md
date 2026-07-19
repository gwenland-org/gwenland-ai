---
title: Dua definisi binary header yang bersaing di glictus-caliburni
status: RESOLVED — hybrid layout (keputusan JinXSuper, 2026-07-19)
severity: high
found: 2026-07-19
resolved: 2026-07-19
blocking: none (was ARTX04)
files:
  - glictus-caliburni/src/execution_unit.rs
  - glictus-caliburni/src/types/layer.rs
  - glictus-caliburni/src/constants.rs
---

# Problem (historis)

Crate punya dua definisi header binary yang bertentangan: `LayerHeader` ARTX01
(12 byte: magic u32 + major/minor u8 + flags u16 + tensor_count u32) vs
`ExecutionUnitHeader` ARTX02 (16 byte: magic + version u16 LE + flags +
8 byte reserved). Spec ARTX04 kemudian memakai layout 12-byte untuk layer
file — bertabrakan dengan yang sudah landed di runtime path.

# Resolusi: HYBRID (dipilih JinXSuper saat review ARTX04)

Satu header universal 16 byte untuk SEMUA unit file (shared/layer/projector),
menggabungkan keduanya:

| Offset | Size | Isi | Asal |
|---|---|---|---|
| 0 | 4 | Magic `b"GLLM"` | keduanya (nilai identik) |
| 4 | 2 | Version u16 LE | ARTX02 — byte-identical dengan major/minor u8 ARTX04 untuk v1.0 (`[0x01, 0x00]`) |
| 6 | 2 | Flags (bitmask `types::layer::flags`) | keduanya |
| 8 | 4 | Tensor count u32 LE | ARTX04 (menempati bekas reserved) |
| 12 | 4 | Reserved, wajib nol | ARTX02 |
| 16 | — | Tensor index mulai | **deviasi dari ARTX04** (spec bilang offset 12) |

Implementasi: `ExecutionUnitHeader` sekarang membawa `tensor_count`
(`new_v1_with_tensors`); `LayerHeader` DIHAPUS; `LayerFile.header` memakai
`ExecutionUnitHeader`; test `spec_artx01::test_layer_header_validate`
di-retarget ke kontrak hybrid. Landed v0.1.159.

# Sisa yang perlu diingat untuk implementasi tensor index (ARTX04 lanjutan)

- Tensor index entry per ARTX04: `name_len u16 + name + shape_len u8 +
  shape u32[] + dtype u16 + offset u64 + size u64` — offset field relatif ke
  region tensor data, dan region data aligned 64 byte
  (`constants::TENSOR_ALIGNMENT`).
- Shape di binary = u32 per dim; manifest `TensorEntry.shape` = `Vec<u64>` →
  widen saat parse binary → manifest type.
- `constants::GLLM_VERSION_MAJOR/MINOR` (u8+u8) masih ada sebagai konstanta
  ARTX01; representasi runtime adalah `GLLM_CURRENT_VERSION: u16 = 1`.
  Kalau suatu saat versi minor naik, pastikan konvensi byte (LE u16 vs
  major,minor) diputuskan ulang — untuk 1.0 dua-duanya `[0x01, 0x00]`,
  tapi u16 LE utk "1.1" = `[0x01+..]`... TIDAK identik lagi (LE: minor
  jadi high byte). Keputusan versi 1.1+ harus eksplisit.
