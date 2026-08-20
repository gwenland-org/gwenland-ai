# ARTX2 — Quant: Audit Pola Bug Q6_K di Seluruh Format

**Bagian dari:** Mensura Veritatis v3 — audit korektnes pasca-fix Q6_K
**Status:** v1 · Basis evidence: audit kode (Explore agent, sesi 2026-07-23) + fix Q6_K (PR #16, merged)
**Legenda:** `[M]` diukur/dibaca langsung dari kode · `[ER]` Evidence Required · `⛔` bug live, belum di-fix

## Kelas bug yang sedang diaudit

`glcore::format::gguf::dequant_q6_k` memakai urutan nibble linear naif; `glproc::kernels::dequant::q6_k::scalar` memakai layout GGML dua-half yang benar. Dua implementasi independen dari format bit yang sama, diam-diam beda tafsir, tidak ada test yang membandingkan keduanya secara langsung — itu yang membuat bug ini korupsi `ffn_down.weight` di 24/24 layer tanpa terdeteksi selama investigasi garbage-output yang panjang (lihat `notes/issues/gllm-e2e-garbage-output.md`, resolved).

Dokumen ini menjawab: **di mana lagi pola ini ada di codebase, dan apakah masing-masing sudah aman atau masih berisiko?**

---

## Peta lengkap per format

### Q6_K — bug asli, **hanya sebagian ter-fix**

| Implementasi | Lokasi | Status |
|---|---|---|
| Dequant-ke-F32 (benar, faithful GGML) | `glproc/src/kernels/dequant/q6_k/scalar.rs:30-55` | ✅ ground truth |
| Integer-dot (`qdot`) | `glproc/src/kernels/qdot/q6_k/scalar.rs:16-55` | ⚠️ re-derive inline, **tidak** import dari kolom di atas |
| `glcore` (native GGUF-side) | `glcore/src/format/gguf.rs:544-572` | ⛔ **masih salah, masih ada, masih reachable** |
| `glcuda` (dua salinan) | `glcuda/src/dequant.rs:103-135` (`dequant_q6_k`), `:196-222` (`q6_k_row_into`) | ✅ benar, tapi 2 salinan independen lagi (tidak panggil glproc) |
| `packages/core` (dua salinan) | `packages/core/src/convert/dequant.rs:816-870` (`dequant_q6_k_standard`), `:879+` (`_euler`) | ⛔ **pakai formula linear-naif yang SAMA SALAHNYA** dengan bug `glcore` — dan didokumentasikan (baris 795-807) *seolah-olah* itu GGML yang benar |

**Total: 7 implementasi independen dari bit-unpacking Q6_K di seluruh workspace ini** (glproc dequant, glproc qdot, glcore, glcuda×2, packages/core×2).

#### ⛔ Bug masih reachable setelah fix PR #16 — tiga lokasi

Fix Q6_K hanya membypass `glcore`'s jalur salah di **dua** call site produksi (`glproc/src/loader.rs:186-193`, `glictus-caliburni/src/converter.rs:162-182`). Tidak menghapus/memperbaiki `dequant_q6_k` itu sendiri. Tiga tempat lain masih memanggilnya:

1. **`glictus-caliburni/examples/diff_dump.rs:235-244, :437-444, :620-629`** — tool diagnostik yang dipakai untuk MENEMUKAN bug Q6_K, ironisnya masih men-fallback ke `gguf.dequantize(info)` (jalur `glcore` yang salah) untuk Q6_K spesifik — cuma Q4_K/Q5_0 yang di-special-case. Setiap perbandingan "GGUF-dequant vs package-F32" yang melibatkan tensor Q6_K lewat tool ini masih memakai referensi yang salah sebagai "ground truth". `[M]`
2. **`packages/core/src/convert/dequant.rs`** — kalau jalur convert crate ini pernah reachable dari `glcli`/`glbench` untuk model real, ini korupsi Q6_K yang identik dengan bug asli, di tempat yang sama sekali berbeda. `[ER]` — belum diverifikasi apakah crate ini punya jalur eksekusi produksi yang benar-benar dipanggil.

**Rekomendasi:** (a) fix `diff_dump.rs` untuk route Q6_K ke `glproc` juga (3 lokasi, low-effort, tool ini akan dipakai lagi untuk audit format lain), (b) audit apakah `packages/core`'s convert path reachable, kalau ya — fix yang sama persis.

---

### Q4_K — pola yang benar, dijadikan referensi

- Dequant: `glproc/src/kernels/dequant/q4_k/scalar.rs`.
- Integer-dot (`qdot/q4_k/scalar.rs:48`): **`use crate::kernels::dequant::q4_k::scalar::{decode_scales, scale_min, BLOCK_BYTES};`** — betulan import, bukan re-derive. `[M]`
- `glcore` menolak eksplisit (`gguf.rs:462-465`, `UnsupportedDtype`, "dequant lives in glproc") — tidak ada implementasi kompetitor.
- **Ini contoh yang benar** dari cara menghindari kelas bug Q6_K: satu ground truth, satu consumer yang mengimpornya. Q6_K seharusnya di-refactor mengikuti pola ini persis.

### Q5_0 — aman secara kebetulan, bukan secara desain

- Dequant: `glproc/src/kernels/dequant/q5_0/scalar.rs`.
- Integer-dot (`qdot/q5_0/scalar.rs:12-27`): **tidak** import, re-derive inline formula yang sama (`lo=(byte&0x0F)|((qh>>i)&1)<<4`, dst.) secara terpisah — dibandingkan baris demi baris dengan versi dequant, saat ini identik secara bentuk. `[M]`
- `glcore` menolak eksplisit (`gguf.rs:462-469`) — tidak ada implementasi kompetitor.
- **Status: sama seperti Q6_K sebelum ketahuan** — dua implementasi independen yang kebetulan setuju, tidak ada test yang menegakkan itu tetap begitu. Risiko drift di masa depan (mis. refactor salah satu tanpa update yang lain) tidak nol.

### Q4_0 — dead code + implementasi ganda, tapi jalur produksi aman

- Dequant: `glproc/src/kernels/dequant/q4_0/{scalar,avx2,avx512}.rs` — **dead code**, nol caller in-crate selain definisinya sendiri. `[M]`
- Tidak ada `qdot/q4_0` sama sekali — `QuantFormat` enum (`qdot/mod.rs:129-163`) tidak punya varian Q4_0.
- Jalur produksi nyata: `glproc/src/loader.rs:186-193,283` — Q4_0 selalu jatuh ke `_ => gguf.dequantize(info)`, yaitu **`glcore`'s implementasi** (`gguf.rs:508-526`), bukan `glproc`'s kernel sendiri.
- Dibandingkan langsung: `glproc`'s dequant (`q4_0/scalar.rs:10-15`) dan `glcore`'s (`gguf.rs:514-521`) **identik secara logika** by inspection. `[M]` — tapi **tidak ada test otomatis** yang menegaskan itu.
- **Catatan:** kernel `glproc` untuk Q4_0 pada dasarnya tidak pernah dieksekusi di jalur produksi — worth menghapusnya atau (lebih baik) mengganti `glcore`'s implementasi jadi memanggilnya, biar cuma ada satu.

### Q8_0 — aman (tidak ada bit-packing nontrivial untuk beda tafsir)

- Dequant + qdot ada, tidak saling import, tapi formatnya cuma `(byte as i8) as f32 * scale` — tidak ada nibble/bit-shuffle yang bisa disalahtafsirkan. Risiko duplikasi rendah secara intrinsik, beda dengan Q4_K/Q5_0/Q6_K.
- Catatan kecil: `dequant/q8_0/scalar.rs`'s `dequant_block` dan `run` mengimplementasikan konversi identik dua kali dalam file yang sama (bukan `run` memanggil `dequant_block` per blok) — kosmetik, bukan risiko korektnes.

### Q8_K — bukan dtype on-disk, aman by construction

Cuma format aktivasi (`qdot/q8_k/`), dikonsumsi hanya oleh `row_dot_q8k`-nya Q4_K. Tidak ada tensor GGUF yang datang sebagai Q8_K, jadi tidak ada risiko dequant-ganda.

### Q4_1 — tidak diimplementasikan sama sekali

Tidak ada `GgufDType::Q4_1`, tidak ada file kernel. Bukan bug, cuma catatan cakupan: model apa pun yang memakai Q4_1 (jarang di dunia nyata dibanding Q4_K) tidak bisa dimuat sama sekali oleh gl-stack. `[ER]` — belum diverifikasi apakah ada model relevan yang memakai ini.

---

## Pola lintas-format: titik fusi adalah tempat yang belum diaudit

Riset industri (lihat [ARTX4-Benchmark.md](ARTX4-Benchmark.md)) menegaskan: bug validasi kuantisasi paling sering muncul bukan di kernel dequant tunggal, tapi di **titik fusi** (dequant+multiply+aktivasi digabung jadi satu kernel). SwiGLU sendiri (bukan bagian dequant, tapi konsumen langsungnya) punya **8+ salinan independen** dari rumus yang sama (`glproc/src/threading.rs` ×4, `qdot/q4_k/swiglu.rs` ×3, `kernels/ops/silu/scalar.rs` ×1) — detail lengkap di [ARTX1-Arsitektur.md](ARTX1-Arsitektur.md) §4. Titik-titik fusi Q4_K/Q5_0/Q6_K × SwiGLU-fused (`qdot/q4_k/swiglu.rs`) adalah kombinasi risiko tertinggi yang belum diaudit numerik secara spesifik di sesi ini — sudah ada test paritas internal (self-consistency), tapi belum dibandingkan ke ground truth non-fused di luar model sintetis.

---

## Rekomendasi, diurutkan berdasarkan rasio biaya/manfaat

1. **[Murah, konkret]** Fix `diff_dump.rs`'s 3 fallback Q6_K yang masih salah (sekarang, karena tool ini akan dipakai lagi buat audit RoPE/format lain).
2. **[Murah, konkret]** Tambah regression test yang membandingkan `qdot::q5_0::row_dot` (integer-dot) terhadap `dequant::q5_0::scalar::dequant_block` + dot manual, pada data acak — menutup risiko "aman by coincidence" yang sama seperti Q6_K sebelum ketahuan.
3. **[Sedang]** Verifikasi apakah `packages/core`'s Q6_K path reachable dari jalur produksi manapun; kalau ya, fix sama seperti PR #16.
4. **[Sedang]** Refactor `qdot::q6_k::row_dot` supaya import `scale_min`/nibble-unpack dari `dequant::q6_k::scalar`, mengikuti pola Q4_K yang sudah benar — menutup kemungkinan divergensi di masa depan, bukan cuma memperbaiki yang sekarang.
5. **[Riset]** Audit numerik titik fusi Q4_K/Q5_0/Q6_K × SwiGLU pada bobot model nyata (bukan cuma test sintetis) — lihat [ARTX4-Benchmark.md](ARTX4-Benchmark.md) untuk metodologi (KL-divergence per-format).

---

## Evidence log

```
[M] Explore agent research report, sesi 2026-07-23 (91 tool call, dicross-check manual
    untuk klaim Q6_K/Q4_K yang sudah diketahui dari fix PR #16)
[M] glictus-caliburni/src/converter.rs:162-182 (fix PR #16, commit 3fa9e64)
[M] glictus-caliburni/examples/diff_dump.rs:235-244,437-444,620-629
[M] packages/core/src/convert/dequant.rs:795-807,816-870,879+,1267-1310
[M] glproc/src/kernels/{dequant,qdot}/{q4_0,q4_k,q5_0,q6_k,q8_0,q8_k}/*.rs
[M] glcuda/src/dequant.rs:103-135,139-146,196-222
[M] glcore/src/format/gguf.rs:459-572
```

**Terkait:** [ARTX1-Arsitektur.md](ARTX1-Arsitektur.md) (pola bug yang sama, sisi arsitektur model) · [ARTX3-Format.md](ARTX3-Format.md) (dtype table `.gllm` vs implementasi) · [ARTX4-Benchmark.md](ARTX4-Benchmark.md) (metodologi validasi yang bisa nangkep kelas bug ini lebih cepat)
