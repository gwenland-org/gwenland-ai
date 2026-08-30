# ARTX3 — Format: `.gllm` Package, Spesifikasi vs Implementasi

**Bagian dari:** Mensura Veritatis v3 — audit korektnes pasca-fix Q6_K
**Status:** v1 · Basis evidence: perbandingan `architecture/GLLM_ARTX/ARTX2-4` (spesifikasi) vs kode nyata (sesi 2026-07-23)
**Legenda:** `[M]` diukur/dibaca langsung · `[R]` klaim dari spesifikasi `GLLM_ARTX` · `[C]` spec vs implementasi berbeda · `[ER]` Evidence Required

## Catatan penting soal penomoran

Repo ini sudah punya seri `ARTX1-11` di `architecture/GLLM_ARTX/` (Overview, Package Spec, Manifest Spec, Layer Spec, Runtime Architecture, dst — ditulis 2026-07-19, dasar untuk implementasi `glictus-caliburni`). Dokumen ini (`gl-stack-audit-2026-07/ARTX3-Format.md`) **bukan** revisi seri itu — ini AUDIT terpisah yang membandingkan spec itu terhadap apa yang benar-benar terimplementasi, dengan disiplin evidence Mensura Veritatis ([R] vs [M], bukan asumsi). Penomoran ARTX di folder ini independen dari seri `GLLM_ARTX`.

---

## 1. Format arsip ZIP — dispesifikasikan, tidak diimplementasikan

### Klaim spesifikasi `[R]`
`architecture/GLLM_ARTX/ARTX2_Package_Specification.md:85-101` (§Archive Format): *"GLLM packages may be distributed as uncompressed ZIP archives (`.gllm` extension) or as directories... The runtime extracts the manifest and checksums from the ZIP header without decompressing layer files."* Diagram eksplisit menunjukkan dua jalur: Directory dan ZIP Archive, sama-sama valid.

### Realita terukur `[M]`
- `GllmPackage::open` dan seluruh jalur baca (`layer_io`, `mmap.rs`) hanya pernah beroperasi pada direktori sepanjang investigasi sesi ini — setiap package yang dibuat/dibaca (`gllm-qwen05b-f32-debug`, `-f32-fixed`, `-gq4a-fixed`) adalah direktori berisi `gllm.json` + `GLLMShared.gllm` + `GLLMTensorLayer-NNNN.gllm`.
- Tidak ada dependency `zip` di `glictus-caliburni/Cargo.toml` — konsisten dengan aturan dependency-bar ketat proyek ini (`CONTRIBUTING.md` §Dependencies) dan sudah dicatat sebelumnya di memory sesi lain: *"task brief's ZIP+flat-tensors[] spec was fictional; zero-dep rule enforced, no zip/serde_json added"* (`project_glbench_quant_info_wave1`).

### `[C]` Kesimpulan
Format ZIP adalah **aspirasi arsitektural yang tidak pernah dibangun**, bukan fitur yang regresi. Tidak ada bahaya korupsi di sini (tidak ada kode yang mengklaim mendukungnya lalu gagal diam-diam) — tapi dokumentasi spec ini menyesatkan kalau dibaca sebagai "apa yang ada sekarang" tanpa konteks ini. **Rekomendasi:** tandai §Archive Format di `ARTX2_Package_Specification.md` sebagai "planned, not implemented" secara eksplisit, atau hapus sampai benar-benar dibangun — status "spesifikasi" tanpa penanda ini membuat siapa pun yang membaca spec doang (bukan kode) mengira dukungan ZIP sudah ada.

---

## 2. `GLLMTokenizer.gllm` — didesain, dinyatakan open, tidak pernah ditulis

### Klaim spesifikasi `[R]`
`ARTX2_Package_Specification.md:107-114` (§Execution Units) mendaftarkan `GLLMTokenizer.gllm` sebagai salah satu dari 4 execution unit yang runtime bisa muat.

### Realita terukur `[M]`
- Memory sesi lain (`project_gllm_tokenizer_oq3`): *"DECIDED: embed as `GLLMTokenizer.gllm` unit... converter does not emit it yet"* — keputusan desain sudah dibuat, implementasi belum.
- Dikonfirmasi ulang langsung di sesi ini: setiap `glconv` run (baik `--quant F32` maupun `--quant GQ4A --policy CPP`) mencetak peringatan eksplisit: `"tokenizer metadata present in GGUF but NOT packaged — GLLM tokenizer packaging is an open spec question (ARTX1 OQ3)"`.
- Konsekuensi langsung: `run_package_e2e.rs` (driver E2E yang dipakai sepanjang investigasi Q6_K) harus memuat tokenizer dari **GGUF asli**, bukan dari package `.gllm` — setiap pemakaian `.gllm` yang benar-benar text-in/text-out saat ini diam-diam bergantung pada file GGUF sumber tetap ada di samping package-nya. Package `.gllm` **tidak self-contained**.

### Status
Bukan bug korektnes (tidak ada yang mengklaim tokenizer ada lalu salah) — murni gap fitur yang sudah jujur ditandai di kode (pesan warning eksplisit). Dicatat di sini karena mempengaruhi klaim "portabilitas" `.gllm` sebagai format standalone.

---

## 3. Tabel dtype — implementasi sudah melampaui spec dasar, secara sah

### Klaim spesifikasi `[R]`
`ARTX4_Layer_Specification.md:67-84` (§Data Type Codes) mendaftarkan: FP32, FP16, BF16, FP8_E4M3, FP8_E5M2, Q4_0, Q4_1, Q4_K, Q4_K_M, Q4_K_S, Q8_0, Q8_K, I32. **Tidak ada Q5_0, tidak ada Q6_K.**

### Realita terukur `[M]`
`DType` enum (`glictus-caliburni/src/manifest/types.rs:19+`) punya `Q5_0` dan `Q6K` sebagai varian valid, dipakai secara ekstensif — dan model uji sepanjang sesi ini (`qwen2.5-0.5b-instruct-q4_k_m.gguf`) punya tensor Q5_0 (mayoritas attn/ffn_gate/ffn_up) dan Q6_K (`ffn_down` di banyak layer) sebagai dtype sumber GGUF asli. `GQ4A`/`GQ2A` (kode `0x0201`/`0x0202`, ekstensi Pridwen) juga ada, jelas di luar tabel dasar ini.

### `[C]` Kesimpulan
Ini **drift yang sah**, bukan bug — implementasi berkembang lebih cepat dari dokumen spec dasarnya (Q5_0/Q6_K perlu didukung begitu format mixed-precision Q4_K_M nyata dipakai; GQ4A/GQ2A adalah pekerjaan Pridwen yang lebih baru dari `ARTX4` aslinya). Tidak ada bahaya di sini selama tabel kode (`0x0010`, dst.) tidak bentrok antara dua dtype berbeda — belum diverifikasi tidak ada tabrakan kode `[ER]`. **Rekomendasi:** update `ARTX4_Layer_Specification.md`'s tabel supaya spec dan kode tidak diam-diam menyimpang lebih jauh; ini murni pekerjaan dokumentasi, bukan kode.

---

## 4. Model integritas checksum + validator — **ini yang sudah benar**

### Klaim spesifikasi `[R]`
`ARTX2_Package_Specification.md:210-246` (§Checksums, §Integrity Model, §Verification Flow) — setiap unit eksekusi punya checksum, diverifikasi saat load.

### Realita terukur `[M]`
- 17 aturan validator (V01-V17) di `glictus-caliburni/src/manifest/validator.rs`, mencakup: kompatibilitas versi (V01/V02), field identitas (V03/V04), sanity metadata termasuk konsistensi MoE (V05), kesesuaian jumlah layer (V06), indeks 0-based/sekuensial/tanpa celah (V07), nama file harus cocok indeks (V08), format checksum per-layer (V09), tensor entries (V14), URI ekstensi valid (V11), checksum shared (V10/V13), batas `parameters` 2 triliun (V17), KV heads ≤ query heads (V16).
- Dipakai nyata sepanjang sesi ini: `pkg.verify_integrity()` dipanggil di setiap test conversion (`convert_quant_f32_assigns_f32_to_every_tensor`, `quant_f32_diagnostic_dequantizes_every_real_tensor`, dst.) dan **selalu mengembalikan kosong** (tidak ada pelanggaran) untuk package yang dihasilkan converter — konsisten dengan `GllmRuntime::open`'s `verify_on_load` opsional yang memanggil `cross_check_layer` per layer.
- Ini adalah lapisan pertahanan yang **tidak pernah gagal menangkap** apa pun sepanjang investigasi Q6_K — masuk akal, karena bug Q6_K adalah salah tafsir NILAI bit (byte-level identik, makna semantiknya yang salah), sesuatu yang checksum secara desain tidak bisa mendeteksi (checksum memverifikasi "byte ini tidak berubah sejak ditulis", bukan "byte ini berarti apa yang seharusnya"). **Ini bukan kegagalan validator** — ini batas jenis bug yang bisa ditangkap kelas alat ini sama sekali. Perbandingan numerik lintas-implementasi (lihat [ARTX4-Benchmark.md](ARTX4-Benchmark.md)) adalah kelas alat yang berbeda, dibutuhkan untuk kelas bug yang berbeda.

### Insight
> Validator manifest (`V01-V17`) dan checksum menjawab pertanyaan **"apakah package ini konsisten secara struktural dengan dirinya sendiri?"** Bug Q6_K butuh pertanyaan yang sama sekali berbeda: **"apakah nilai di dalam package ini benar secara semantik dibanding sumbernya?"** Dua pertanyaan itu butuh dua kelas alat yang sepenuhnya berbeda — repo ini kuat di yang pertama, nol di yang kedua sebelum sesi ini.

---

## 5. `GLLMProj.gllm` (projector, multimodal) — slot ada, jalur tulis belum diverifikasi

`PROJECTOR_FILENAME` (`constants.rs:15`), field opsional di `package.rs:50`, disebut di `plugin.rs:278` untuk "linear projector for multimodal packages" — tapi `converter.rs`'s komentar sendiri (baris ~13) menyatakan tensor non-standar dirutekan "ke nama aslinya (dengan warning), bukan ke `GLLMProj.gllm`". `[ER]` — belum diverifikasi ada jalur produksi yang benar-benar menulis file ini untuk model multimodal nyata; di luar cakupan sesi ini (tidak ada model multimodal diuji).

---

## Ringkasan tabel spec-vs-implementasi

| Fitur | Spec bilang | Implementasi nyata | Kelas gap |
|---|---|---|---|
| Arsip ZIP | Didukung | Tidak pernah dibangun | Fitur belum ada (jujur, tak berbahaya) |
| `GLLMTokenizer.gllm` | Execution unit resmi | Diputuskan, belum ditulis converter | Fitur belum ada (ditandai eksplisit di kode) |
| Tabel dtype (`ARTX4`) | 13 kode, tanpa Q5_0/Q6_K | Q5_0/Q6_K/GQ4A/GQ2A semua ada & dipakai | Drift dokumentasi, bukan bug |
| Checksum + V01-V17 | Menjamin integritas | Bekerja persis seperti dispec | ✅ Sesuai — tapi bukan alat buat kelas bug Q6_K |
| `GLLMProj.gllm` | Execution unit resmi | Slot ada, jalur tulis `[ER]` | Belum diverifikasi |

---

## Evidence log

```
[R] architecture/GLLM_ARTX/ARTX2_Package_Specification.md:85-114,210-246
[R] architecture/GLLM_ARTX/ARTX4_Layer_Specification.md:67-85

[M] glictus-caliburni/src/manifest/validator.rs (V01-V17, dibaca langsung)
[M] glictus-caliburni/src/manifest/types.rs:19+ (DType enum)
[M] glictus-caliburni/src/constants.rs:15, package.rs:50, plugin.rs:278
[M] glconv real run, sesi 2026-07-23 (warning tokenizer, dua kali: --quant F32 dan --quant GQ4A)
[M] pkg.verify_integrity() dipanggil di setiap test conversion sesi ini, selalu kosong

[R] Memory: project_gllm_tokenizer_oq3, project_glbench_quant_info_wave1
```

**Terkait:** [ARTX2-Quant.md](ARTX2-Quant.md) (dtype table drift terkait langsung ke bug Q6_K) · [ARTX4-Benchmark.md](ARTX4-Benchmark.md) (kelas alat yang sebenarnya dibutuhkan untuk menangkap bug semantik, bukan struktural)
