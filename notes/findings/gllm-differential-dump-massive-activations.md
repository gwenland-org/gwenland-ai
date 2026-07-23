---
title: "diff_dump.rs: divergence layer-by-layer terlokalisir ke satu dimensi — fenomena \"massive activations\""
status: resolved
severity: high
found: 2026-07-23
files:
  - glictus-caliburni/examples/diff_dump.rs
  - glproc/src/runner.rs
references:
  - https://www.alphaxiv.org/audio/2402.17762
  - https://arxiv.org/html/2605.08504
  - https://arxiv.org/pdf/2410.10781
  - https://arxiv.org/abs/2603.05498
  - https://arxiv.org/html/2508.03616v1
---

# Problem

Setelah 3 bug real ditemukan+diperbaiki tanpa menyelesaikan gejala garbage
(lihat [gllm-e2e-garbage-output.md](../issues/gllm-e2e-garbage-output.md)),
dibutuhkan perbandingan numerik langsung antara jalur known-good
(`glproc::runner::Runner`) dan jalur yang dicurigai rusak (`GlprocBackend`)
— bukan cuma baca kode.

# Tooling yang dibangun

`glictus-caliburni/examples/diff_dump.rs` — jalankan kedua jalur di token
yang sama persis, dump L2 norm + max abs diff hidden state per layer, plus
top-5 logits akhir.

Infrastruktur pendukung (diagnostic-only, mengikuti pola zero-cost-when-off
yang sudah ada di `Runner` — `trace_on`/`prof`):
- `Runner::set_capture_hidden(bool)` + `hidden_dump() -> &[Vec<f32>]` —
  capture residual stream setiap layer.
- `Runner::forward_chunk_into` — wrapper publik untuk `step_chunk` (jalur
  batched prefill privat). **Penting**: generasi asli SELALU memproses
  posisi 0 lewat `step_chunk`, walau prompt cuma 1 token — `step`/
  `forward_into` cuma pernah dipakai untuk posisi decode asli (>= 1). Diverifikasi
  keduanya memberi output byte-identik untuk 1 token, jadi bukan sumber bug,
  cuma cara yang benar untuk menguji jalur yang benar-benar dipakai posisi 0.
- `CapturingBackend` (dalam contoh, bukan produksi) — decorator
  `ExecutionBackend` yang membungkus `GlprocBackend` asli dan merekam output
  tiap layer. Nol perubahan ke `glproc_backend.rs` — `ExecutionBackend`
  memang seam yang bisa ditukar.

# Temuan: divergence meledak di layer 2, terlokalisir ke SATU dimensi

Token 1, posisi 0. Embedding, config (rms_eps, rope_freq_base, layout GQA),
dan nilai bobot representatif (`blk.0.attn_q.weight`, dequant ulang dari
GGUF asli) semua **identik persis** antar kedua jalur.

Tabel norm per layer: Path A (`Runner`) dan Path B (`GlprocBackend`) dekat
di layer 0-1 (norm ~5-8, max_diff ~3 — kecil, wajar). Lalu Path A
**meledak**: 8.25 → 698.78 (layer 2) → plateau stabil ~1500 sampai layer 20
→ **collapse** tiba-tiba ke 48 di layer 21 → konvergen lagi ~82-87 di layer
23. Path B tetap moderat sepanjang waktu (9 → ~30 → ~80).

Dump per-dimensi menemukan: **dimensi 62** sendirian menyumbang hampir
SELURUH norm total Path A dari layer 2-20 (nilai dim 62 ≈ norm keseluruhan
vektor, keduanya ~1500). Path B, di dimensi yang sama, tetap kecil (0.1-17)
sepanjang range itu. Di layer 21 kedua jalur pindah "dimensi top" ke 490 dan
mengecil bersama.

# Riset web: ini fenomena nyata, bukan bug signature

Pola ini match persis dengan **"massive activations"** — fenomena
terdokumentasi dan aktif diriset di interpretability LLM (Sun et al. 2024 +
beberapa follow-up 2025-2026, lihat references di frontmatter):
- Sejumlah kecil dimensi spesifik (bukan seluruh vektor) tiba-tiba melonjak
  di satu "emergence layer", lalu **tetap stabil** di layer-layer tengah
  berikutnya — ini perilaku SEHAT model yang benar, bukan sinyal bug.
- Berkorelasi kuat dengan **attention sink**: ketika satu head tidak punya
  konteks berguna untuk diambil, model mengarahkan massa attention ke token
  tanpa isi informasi nyata — muncul sebagai massive activation. Test kita
  (satu token terisolasi di posisi 0, `cached_len=1`, tidak ada apa pun
  selain dirinya sendiri untuk di-attend) adalah skenario yang persis
  memicu ini.

Kesimpulan pada titik ini: divergensi kemungkinan besar bukan artefak
presisi kuantisasi (nilai bobot sudah terverifikasi identik — massive
activation adalah properti nilai bobot yang dipelajari model berinteraksi
dengan mekanisme tertentu, bukan noise angka). **Ini kesimpulan yang
kemudian dibalik oleh temuan berikutnya** — lihat
[gllm-f32-replay-reversal.md](gllm-f32-replay-reversal.md).
