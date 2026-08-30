# ARTX4 — Benchmark & Profiling: Apa yang Perlu Diukur Sekarang

**Bagian dari:** Mensura Veritatis v3 — audit korektnes pasca-fix Q6_K
**Status:** v1 · Basis evidence: `glbench` (dibaca langsung), `Pridwen-proposal-v5.md` §12, riset web 2026-07-23
**Legenda:** `[M]` diukur/dibaca langsung dari kode · `[R]` riset eksternal/dokumen lain · `[ER]` Evidence Required

## Kenapa sekarang, bukan sebelumnya

Setiap angka PPL/kualitas Pridwen sebelum sesi ini **tidak bisa dipercaya** — bug garbage-output (root cause: Q6_K, lihat [ARTX2-Quant.md](ARTX2-Quant.md)) membuat `glbench ppl` mengembalikan PPL ~18 juta, angka yang tidak berarti apa-apa selain "modelnya rusak total". Sekarang bug itu resolved (PR #16, merged), setiap item di bawah ini **baru pertama kali bisa dieksekusi dengan berarti**.

---

## 1. Yang sudah direncanakan Pridwen sendiri — sekarang unblocked

`Pridwen-proposal-v5.md` §12 "Known Unknowns" mendaftarkan 10 pertanyaan empiris. Status tiap satu, setelah fix Q6_K:

| Unknown | Metode terencana | Status sekarang |
|---|---|---|
| GQ4A PPL vs Q4_K_M delta | `glbench ppl` | **Bisa dijalankan, belum dijalankan** |
| GQ2A baseline PPL (tanpa FHT) | `glbench ppl` | **Bisa dijalankan, belum dijalankan** |
| GQ2A_CPP vs Q4_K_M tok/s nyata | `glbench` decode benchmark | **Bisa dijalankan** — tapi lihat §4, runtime `.gllm` masih lambat (mmap/unmap per-layer per-token, dicatat sejak investigasi awal, belum diperbaiki) |
| `scale_delta`/`min_delta` saturation rate GQ2A | Ukur di tensor kalibrasi nyata | Belum ada tooling — perlu skrip baru, bukan sekadar re-run `glbench` |
| Per-layer sensitivity scores | Kalibrasi run Qwen2.5-0.5B | Phase 2-3 scope, belum dimulai |
| 2.8 bpw cliff threshold | PPL sweep lintas bpw | Butuh GQ1A dulu (Phase 4, belum ada) — GQ4A (4.3125 bpw) dan GQ2A (2.625 bpw) sudah cukup untuk sweep parsial |
| FHT overhead / GQ2A-R quality delta | Latency profiling + PPL before/after | GQ2A-R belum diimplementasi (Phase 2 decision gate belum dilewati) |
| MCKP vs greedy | A/B comparison | Phase 3 scope |
| latency_cost per format | `glbench` dequant kernel profiling | Lihat §3 — mekanismenya belum ada di `glbench` |
| GQ1A PTQ collapse severity | Kalibrasi + PPL vs QAT | Phase 4, butuh `gltrain` (belum diputuskan, lihat memory `project_gltrain_plan`) |

**Yang benar-benar bisa dieksekusi minggu ini, tanpa kerja tambahan:** GQ4A PPL vs Q4_K_M, GQ2A baseline PPL. Semua yang lain butuh tooling baru dulu (lihat §3) atau menunggu fase lain.

---

## 2. PPL saja tidak cukup — riset industri 2026 `[R]`

### KL-divergence vs referensi full-precision
`llama.cpp`'s tool perplexity mendukung `--kl-divergence-base path/to/logits.kld`, merekam logit lengkap dari versi FP16 lalu mengukur KL-divergence versi terkuantisasi terhadapnya per-token — bukan cuma cross-entropy agregat. Konvensi komunitas per Februari-April 2026: setiap model GGUF publik dibenchmark PPL **dan** KL-divergence, bukan PPL saja.

**Kenapa ini penting buat kita spesifik:** KL-divergence per-token adalah persis apa yang `diff_dump.rs` lakukan secara ad-hoc untuk menemukan bug Q6_K (bandingkan logits Path A vs Path B) — tapi itu one-off diagnostic script, bukan metrik `glbench` yang reusable. Kalau metrik ini sudah ada sebagai bagian rutin dari `glbench run`, bug Q6_K kemungkinan besar akan terdeteksi jauh lebih cepat (KL-divergence yang meledak akan langsung mencolok, dibanding harus menunggu seseorang curiga ke output teks yang "kelihatan garbage").

### Reasoning/task benchmark turun lebih cepat dari PPL
Pada Q3 quantization, akurasi matematika (GSM8K-style) turun **~3x lebih besar** dari penurunan PPL. Model bisa punya PPL yang "kelihatan oke" sementara kemampuan reasoning-nya sudah kolaps. Dimensi benchmark standar 2026: coding (HumanEval), reasoning matematis (MATH/GSM8K), kepatuhan instruksi (IFEval) — bukan cuma PPL.

**Implikasi buat Pridwen:** klaim GQ2A (2.625 bpw)/GQ1A (2.0625 bpw) "dalam target PPL" tidak cukup untuk klaim kualitas menyeluruh kalau tidak pernah divalidasi terhadap reasoning task. `glbench` saat ini **tidak punya task-based eval sama sekali** — hanya PPL (`glbench/src/ppl.rs`) dan token-parity (`glbench validate`).

### Studi pembanding yang relevan langsung
Paper "Which Quantization Should I Use? A Unified Evaluation of llama.cpp Quantization on Llama-3.1-8B-Instruct" (arXiv:2601.14277, Jan 2026) — 1 model tetap, semua format GGUF resmi, PPL + GSM8K/HellaSwag/IFEval/MMLU/TruthfulQA + throughput CPU + ukuran, kontrol hardware/harness ketat. **Ini blueprint metodologi yang tepat buat studi GQ4A/GQ2A/Q4_K_M yang Pridwen §13 rencanakan** — hanya perlu menambah task-based eval selain PPL, mengikuti bentuk studi ini persis.

---

## 3. `glbench` — tooling yang ada, dan gap-nya

### Subcommand yang ada `[M]`
- `run` — satu engine+model, satu workload (prefill/decode/end_to_end/stress), timing + validasi.
- `ab` — banding N model sekuensial di bawah workload identik.
- `compare` — diff dua session JSON arsip.
- `validate` — token-level parity vs oracle (default `glproc`) — **bukan** tensor/weight-level, dan bukan KL-divergence.
- `scale` — sweep token budget, klasifikasi linear/sub-linear/saturating.
- `inspect`/`export` — render ulang, tanpa pengukuran baru.
- `quant-info` — **hanya** tally jumlah tensor per dtype `.gllm` (GQ4A/GQ2A/F32/dst.) + persentase coverage. Tidak ada error/waktu per format.
- `ppl` — perplexity WikiText-2 lewat runtime `.gllm`, feature-gated `gllm-bench`, doc comment-nya sendiri masih menyebut "known open correctness bug" — **sekarang stale**, perlu diupdate mengikuti resolusi Q6_K.

### Dead code yang sudah ditulis, tinggal disambung `[M]`
`glbench/src/comparison/{quantization,engine,hardware}.rs` — tiga fungsi (`compare_quantization`, `compare_engines`, `compare_hardware`) sudah ditulis, `compare_quantization`'s doc comment eksplisit menyebut "Quantization-vs-quantization comparison (e.g. Q8_0 vs Q4_K_M)" — **persis kebutuhan kita** — tapi **nol caller di mana pun**, tidak di-wire ke subcommand `main.rs` manapun. Semua tiga beroperasi pada `BenchmarkSession` di mana `.engine.quantization` adalah **satu label string untuk seluruh model**, bukan breakdown per-tensor/per-blok — jadi bahkan kalau disambung, granularitasnya masih "seluruh model pakai format X" vs "seluruh model pakai format Y", bukan "berapa error yang disumbang Q6_K secara spesifik di dalam satu model campuran Q4_K_M".

### Kesimpulan: tidak ada mekanisme atribusi per-format sama sekali
Tidak ada di `glbench` manapun cara menjawab "berapa banyak error/waktu yang disumbang tensor Q6_K secara spesifik, di dalam satu model yang formatnya campuran" — pertanyaan yang persis dibutuhkan untuk mendeteksi kelas bug Q6_K secara sistematis (bukan kebetulan seperti kemarin).

---

## 4. Prasyarat non-benchmark: performa runtime `.gllm` masih jadi penghalang

Dicatat sejak investigasi awal (`Pridwen-proposal-v5.md` §14 Phase 2, blocking gap): setiap layer `.gllm` di-mmap dan di-unmap ulang dari disk **setiap token**, bukan sekali per load. Diverifikasi ulang tersirat di sesi Q6_K ini — log runtime menunjukkan `"unmapped 24"` setiap token sepanjang setiap run `run_package_e2e`/`diff_dump`. Belum ada perbaikan atas ini. **Ini bukan gap benchmark — ini gap performa yang akan membuat benchmark tok/s versus Q4_K_M jadi tidak representatif** (`.gllm` akan kalah jauh secara tidak adil karena overhead I/O, bukan karena GQ4A/GQ2A secara inheren lebih lambat). PPL bisa diukur sekarang; tok/s belum bisa diukur secara adil sampai ini diperbaiki.

---

## 5. Rekomendasi konkret, diurutkan

1. **[Sekarang juga, murah]** Jalankan `glbench ppl` GQ4A_CPP dan GQ2A_CPP vs Q4_K_M asli (via `glproc` engine) — nomor pertama yang sudah bisa dipercaya sejak Pridwen dimulai. Update doc comment `ppl.rs` yang masih menyebut bug lama.
2. **[Sedang]** Tambah KL-divergence (bukan cuma cross-entropy PPL) sebagai metrik `glbench ppl` opsional — bandingkan logit .gllm vs logit `glproc` per-token pada sequence yang sama. Ini menutup gap "kelas alat yang menangkap bug semantik" yang dicatat di [ARTX3-Format.md](ARTX3-Format.md) §4.
3. **[Sedang]** Sambungkan `compare_quantization`/`compare_engines`/`compare_hardware` (dead code) ke subcommand `glbench` nyata — langkah pertama menuju atribusi per-format, meski granularitasnya masih per-model bukan per-tensor.
4. **[Riset, lebih besar]** Tambah minimal 1-2 task-based eval (GSM8K atau setara, ukuran kecil dulu) — PPL saja tidak cukup untuk klaim kualitas GQ2A/GQ1A per riset industri di atas.
5. **[Prasyarat]** Perbaiki mmap/unmap per-token `.gllm` runtime sebelum klaim tok/s apa pun dipublikasikan — angka sekarang akan menyesatkan (bukan mengukur GQ4A/GQ2A, mengukur overhead I/O).

---

## Evidence log

```
[R] Pridwen-proposal-v5.md §12 "Known Unknowns", §13 "Expected Results", §14 Phase 2 blocking gap

[M] glbench/src/main.rs (subcommand dispatch, dibaca langsung)
[M] glbench/src/ppl.rs (doc comment ppl.rs:6-13, stale post-fix)
[M] glbench/src/comparison/{quantization,engine,hardware}.rs (dead code, nol caller)
[M] glbench/src/quant_info.rs (tally-only, tidak ada error/waktu attribution)
[M] run_package_e2e/diff_dump log sesi ini: "unmapped 24" per token, konsisten dgn
    Pridwen-proposal-v5.md §14 Phase 2's catatan blocking gap

[R] WebSearch 2026-07-23:
    "local LLM inference engine quantization correctness validation benchmark
     best practices 2026" — Q4 sweet spot, Q5/Q6 barely-worth-it, reasoning
     degrades faster than PPL at low bit-width
    "perplexity benchmark standard quantized LLM 2026 llama.cpp wikitext KL
     divergence" — llama.cpp --kl-divergence-base, konvensi WikiText-2
    "Which Quantization Should I Use? A Unified Evaluation of llama.cpp
     Quantization on Llama-3.1-8B-Instruct" (arXiv:2601.14277, Jan 2026)
    "quantization kernel unit testing bit-exact parity cross-implementation
     consistency LLM inference" — bug sering di titik fusi operator, bukan
     kernel tunggal (relevan ke ARTX2-Quant.md §Pola lintas-format)
    "RoPE scaling YaRN long context support inference engines 2026" —
     vLLM/SGLang aktif menambah dukungan; relevan ke ARTX1-Arsitektur.md §2
```

**Terkait:** [ARTX1-Arsitektur.md](ARTX1-Arsitektur.md) · [ARTX2-Quant.md](ARTX2-Quant.md) · [ARTX3-Format.md](ARTX3-Format.md) · `docs/Mensura_Veritatis.md` (v1, pendahulu metodologis dokumen ini)
