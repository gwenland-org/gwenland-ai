# ARTX1 — Arsitektur: RoPE, Fungsi Aktivasi FFN, Dimensi Head

**Bagian dari:** Mensura Veritatis v3 — audit korektnes pasca-fix Q6_K
**Status:** v1 · Basis evidence: pembacaan kode langsung (sesi 2026-07-23) + `docs/Mensura_Veritatis.md` v1
**Legenda:** `[M]` diukur/dibaca langsung dari kode · `[R]` klaim dari dokumen lain · `[ER]` Evidence Required · `[C]` Research Conflict

## Kenapa dokumen ini ada

Fix Q6_K (`fix/gllm-q6k-dequant-corruption`, PR #16, merged) memperbaiki satu contoh dari kelas bug yang lebih besar: **dua implementasi independen dari hal yang sama, diam-diam beda tafsir, tidak ada yang mengecek keduanya setuju.** Dokumen ini mengaudit tempat lain di codebase yang punya bentuk risiko sama, di sisi *arsitektur model* (bukan format bit quant — itu di [ARTX2-Quant.md](ARTX2-Quant.md)): RoPE style, RoPE scaling, dan fungsi aktivasi FFN.

Semua temuan di sini adalah **silent-wrong**, bukan crash — kelas bug yang sama seperti Q6_K: model tetap jalan, mengeluarkan output yang terlihat valid secara bentuk, tapi salah secara numerik.

---

## 1. RoPE style di-hardcode NeoX di jalur `.gllm`

### Ringkasan
`glproc::runner::Runner` (jalur GGUF asli, `gwen run`) mendukung **dua** gaya RoPE dan memilihnya per arsitektur. `glictus-caliburni::GlprocBackend` (jalur `.gllm`) hanya mengimplementasikan **satu**, tanpa cara memilih yang lain.

### Temuan
- `glproc/src/model.rs:5-10` — enum `RopeStyle { Norm, Neox }`, didokumentasikan: `Norm` = "original llama", `Neox` = "qwen2, phi, gemma...". `[M]`
- `glproc/src/loader.rs:685-688` — pemilihan otomatis dari string arsitektur GGUF: `"llama" | "llama2" | "minicpm" => RopeStyle::Norm, _ => RopeStyle::Neox`. `[M]`
- `glproc/src/runner.rs:132-147` (`fn rope`) — menerapkan kedua gaya lewat `match style`. `[M]`
- `glictus-caliburni/src/runtime/glproc_backend.rs:176-192` (`fn rope_neox`) — **hanya NeoX**, tanpa parameter style. Doc comment fungsi ini sendiri mengakui ini duplikasi tangan: *"Mirrors `glproc::runner::rope`, which is private to that module; duplicated rather than exposed because it is 8 lines of arithmetic, not a kernel."* `[M]`
- `AttnShape` (`glproc_backend.rs:66-79`) — struct konfigurasi attention `.gllm`, **tidak punya field style sama sekali**. `[M]`
- `ModelMetadata` (`glictus-caliburni/src/manifest/metadata.rs:108-158`) — skema manifest `.gllm`, juga tidak menyimpan arsitektur atau RoPE style apa pun; hanya `rope_dims`/`rope_freq_base`/`rope_scaling`. `[M]`

### Konsekuensi
Model apa pun ber-arsitektur `llama`/`llama2`/`minicpm` yang dikonversi ke `.gllm` dan dijalankan lewat `GlprocBackend` akan mendapat RoPE **NeoX**, padahal seharusnya **Norm** — salah secara diam-diam, tidak error, tidak crash. Qwen2/Qwen2.5 (model uji sepanjang investigasi Q6_K) kebetulan memang NeoX, jadi gap ini **tidak pernah termanifestasi** di pengujian manapun sejauh ini — persis pola yang sama dengan bug Q6_K sebelum ketahuan (butuh model/kondisi spesifik untuk memicu).

### Insight
> Bug ini secara struktural identik dengan bug key_length yang sudah tercatat sejak investigasi garbage-output (§15 poin 4, `Pridwen-proposal-v5.md`): **glproc's native loader membaca konfigurasi per-arsitektur dengan benar; jalur konversi `.gllm` tidak pernah menyalin informasi arsitektur itu ke manifest sama sekali.** Ini bukan kejadian terisolasi — ini gejala bahwa `ModelMetadata` dirancang terlalu sempit terhadap sebaran arsitektur asli GGUF sejak awal.

### Fix yang disarankan (belum dikerjakan)
Tambah field arsitektur (atau langsung `rope_style: RopeStyle`) ke `ModelMetadata`, isi dari `general.architecture` GGUF di `converter.rs` (mirror logika `glproc/src/loader.rs:685-688` persis, idealnya lewat fungsi yang di-share, bukan disalin tangan ketiga kalinya), thread ke `AttnShape`.

---

## 2. `rope_scaling` (YaRN/linear) ada di skema, tidak pernah dipakai

### Ringkasan
Manifest `.gllm` punya slot untuk RoPE scaling. Parsernya jalan. Tidak ada satu pun kode yang benar-benar mengalikan frekuensi/posisi dengan faktor itu.

### Temuan
- `RopeScaling` struct (`glictus-caliburni/src/manifest/metadata.rs:98-106`, field `scaling_type: String`, `factor: f64`) ada sebagai field `ModelMetadata.rope_scaling: Option<RopeScaling>` (`metadata.rs:135-137`). `[M]`
- `converter.rs:278` — **di-hardcode `None`**. Tidak ada `gguf.get_meta` untuk key `{arch}.rope.scaling.type`/`{arch}.rope.scaling.factor` di mana pun di codebase (grep repo-wide, nol hasil). `[M]`
- Field ini **tidak pernah dibaca** di jalur manapun: grep `.rope_scaling` (akses field, bukan definisi) hanya menemukan 3 situs assignment-ke-`None` (`converter.rs:278`, `test_helpers.rs:67`, `glproc_backend.rs:404`), nol situs pembacaan. `[M]`
- Signature `glproc::runner::rope` (`runner.rs:132`) dan `glictus-caliburni`'s `rope_neox` (`glproc_backend.rs:181`) **sama-sama** hanya menerima `pos, head_dim, freq_base[, style]` — tidak ada parameter scaling di kedua jalur, bukan cuma `.gllm`. `[M]`
- Satu-satunya kode yang menyentuh `rope_scaling` adalah test fixture yang membuktikan *parser*-nya jalan (`test_helpers.rs:156`, `{"rope_scaling": {"type": "linear", "factor": 2.0}}`) — bukan buktinya dipakai. `[M]`
- `glproc/src/loader.rs` hanya membaca `rope.freq_base`, tidak ada key scaling. `glcuda/src/loader.rs` sama. `[M]`

### Konsekuensi
**Ini bukan gap `.gllm`-spesifik — ini gap seluruh gl-stack.** Model apa pun yang dilatih/di-tune dengan RoPE scaling (YaRN, linear, dynamic NTK — makin umum untuk model long-context, lihat [ARTX4-Benchmark.md](ARTX4-Benchmark.md) §riset industri) akan dijalankan dengan RoPE **tidak ter-scale** di setiap engine `gl*`, tanpa peringatan apa pun. Untuk konteks pendek dalam batas pretraining asli model, ini sering tidak kentara (RoPE unscaled masih "masuk akal" secara lokal); begitu prompt melewati panjang pretraining asli, degradasi kualitas kemungkinan besar diam-diam, bukan crash.

### Status
`[ER]` — belum diukur dampaknya pada model nyata (belum ada model YaRN yang diuji di repo ini). Ini murni temuan kode-baca: fitur di-skema-kan, tidak pernah diimplementasi, tidak ada peringatan run-time bila sebuah GGUF sebenarnya mendeklarasikan rope scaling dan itu dibuang diam-diam saat konversi.

---

## 3. `attention.key_length` — dikonfirmasi ulang, masih terbuka

### Ringkasan
Sudah tercatat sejak investigasi garbage-output (`Pridwen-proposal-v5.md` §15 poin 4), dikonfirmasi ulang sesi ini: masih ada, masih belum di-fix.

### Temuan
- `glproc/src/loader.rs:676` — `meta_u64(gguf, &arch, "attention.key_length").unwrap_or((dim / n_heads) as u64)`. `[M]`
- `glcuda/src/loader.rs:195` — pola identik, ditulis independen. `[M]`
- `glictus-caliburni/src/manifest/metadata.rs:190-198` (`ModelMetadata::head_dim`) — **hanya** `embedding_length / num_heads`, tidak ada field `key_length` sama sekali di struct, dan `converter.rs` tidak pernah mengekstrak `{arch}.attention.key_length` dari GGUF sumber (grep `converter.rs`, nol hasil). `[M]`

### Konsekuensi
Model dengan `head_dim` non-uniform (di mana `key_length` GGUF override `embedding_length/num_heads`) akan dihitung benar oleh `glproc`'s native loader, tapi salah oleh jalur konversi `.gllm`. Qwen2.5-0.5B (model uji) tidak punya key `key_length`, jadi kedua jalur kebetulan setuju — gap ini tidak pernah termanifestasi di pengujian manapun, sama seperti RoPE style di §1.

---

## 4. Fungsi aktivasi FFN — SwiGLU adalah satu-satunya, di-hardcode unconditional

### Ringkasan
Tidak ada GELU atau ReLU di mana pun di codebase. SwiGLU dipanggil tanpa syarat di setiap jalur FFN, tanpa cek arsitektur.

### Temuan
- Grep case-insensitive `gelu` dan `relu` di `glproc/src` dan `glictus-caliburni/src`: **nol hasil untuk keduanya**. `[M]`
- Tidak ada dispatch kondisional (`match arch { "gemma" => gelu, _ => silu }`) di mana pun. `[M]`
- Formula inti: `glproc/src/kernels/ops/silu/scalar.rs:3-10` — `*g = x / (1.0 + fast_exp(-x)) * u`. `[M]`
- Call site produksi: `glproc/src/runner.rs:1073` (decode), `:1451` (batched prefill), `glproc/src/moe.rs:345` (MoE expert), `glictus-caliburni/src/runtime/glproc_backend.rs:345` (`.gllm`, komentar baris 339 eksplisit: `// --- feed-forward block (dense SwiGLU) ---`). `[M]`
- **Reimplementasi inline tambahan** dari formula yang sama (bukan panggil `silu_mul`, tapi menyalin ulang `g/(1+exp(-g))*u` secara manual): `glproc/src/threading.rs:631` (`par_matvec_swiglu`), `:896,908,917` (`par_matmul_swiglu`, varian batched/tail-loop) — **4 salinan inline independen lagi** dari rumus satu baris yang sama. `[M]`
- `glproc/src/kernels/qdot/q4_k/swiglu.rs:48,76,92` — tiga varian SwiGLU terfusi Q4_K lagi, didokumentasikan divalidasi paritas satu sama lain lewat test (baris ~245-271). `[M]`

### Konsekuensi
Untuk **Qwen2/Qwen2.5** (satu-satunya keluarga model yang diuji intensif di repo ini), SwiGLU memang benar — ini bukan bug untuk model uji manapun sejauh ini. Tapi setiap model GELU-based (beberapa varian GPT, BERT-turunan, dst.) yang dimuat lewat engine manapun di gl-stack akan **dihitung dengan SwiGLU**, bukan error "unsupported activation" — silent-wrong dalam bentuk paling murni.

### Insight yang menghubungkan §4 ke §1-3
Pola yang berulang di keempat temuan dokumen ini: **konfigurasi arsitektural (RoPE style, RoPE scaling, key_length, fungsi aktivasi) dibaca dengan benar di jalur GGUF-native (`glproc::loader`), tapi jalur `.gllm` (`glictus-caliburni::converter`) tidak pernah dirancang untuk membawa informasi arsitektural itu ke manifest sejak awal — `ModelMetadata` cuma punya bidang numerik generik (dims, head count, eps), tidak ada bidang "arsitektur apa ini".** Setiap gap di atas punya akar yang sama; memperbaikinya satu-satu akan terus bocor sampai `ModelMetadata` benar-benar membawa cukup info arsitektural untuk merekonstruksi keputusan yang `glproc::loader` buat dari string `general.architecture`.

### Rekomendasi lintas-temuan
Bukan 4 fix terpisah (RoPE style, RoPE scaling, key_length, activation) — satu fix struktural: **converter.rs merekam `general.architecture` (dan turunannya: rope style, activation kind) ke `ModelMetadata` sebagai fakta eksplisit, bukan diam-diam diasumsikan ulang di sisi runtime.** Ini juga jadi prasyarat alami kalau `.gllm` suatu saat mau mendukung model non-Qwen (llama, gemma, dst.) — sampai sekarang jalur `.gllm` implisit **hanya benar untuk keluarga Qwen2/Qwen2.5/Qwen3-dense**, meski tidak ada validator yang menegakkan batasan itu.

---

## Evidence log

```
[M] Dibaca langsung, sesi 2026-07-23:
    glproc/src/model.rs:5-10, runner.rs:132-147, loader.rs:676,685-688
    glictus-caliburni/src/runtime/glproc_backend.rs:66-79,176-192,345,404
    glictus-caliburni/src/manifest/metadata.rs:98-106,108-158,135-137,190-198
    glictus-caliburni/src/converter.rs:278
    glictus-caliburni/src/test_helpers.rs:67,156
    glproc/src/threading.rs:631,896,908,917
    glproc/src/kernels/ops/silu/scalar.rs:3-10
    glproc/src/kernels/qdot/q4_k/swiglu.rs:48,76,92
    glproc/src/moe.rs:345
    glcuda/src/loader.rs:195

[R] Pridwen-proposal-v5.md §15 poin 4 (key_length flagged sebelumnya, belum di-fix)
```

**Terkait:** [ARTX2-Quant.md](ARTX2-Quant.md) (pola bug yang sama, sisi format bit) · [ARTX4-Benchmark.md](ARTX4-Benchmark.md) (kenapa RoPE scaling makin relevan di industri sekarang)
