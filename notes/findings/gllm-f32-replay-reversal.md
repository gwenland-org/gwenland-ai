---
title: "REVERSAL: replay F32 independen match ke GlprocBackend, bukan ke glproc::runner::Runner"
status: open
severity: critical
found: 2026-07-23
blocking: "notes/issues/gllm-e2e-garbage-output.md — mengubah total premis investigasi"
files:
  - glictus-caliburni/examples/diff_dump.rs
  - glproc/src/runner.rs
---

# Problem

[gllm-differential-dump-massive-activations.md](gllm-differential-dump-massive-activations.md)
menyimpulkan divergensi kemungkinan bukan artefak kuantisasi karena nilai
bobot sudah terverifikasi identik. Tapi perbandingan itu tetap mencampur
komputasi native-quantized Path A (GGUF Q4_K_M asli — `attn_q.weight` layer
0 misalnya berdtype Q5_0) dengan komputasi F32 murni Path B. Perlu
dipisahkan: apakah divergensi datang dari NILAI (sudah disingkirkan) atau
dari JALUR KOMPUTASI native-quantized `glproc` itu sendiri (bridge/repack
ke Q8_0 untuk SIMD cepat — dilacak juga di catatan performa quant lama)?

# Metode: replay F32 dari nol, independen dari kedua implementasi

Ditulis `replay_full_layer_f32` di `diff_dump.rs` — reimplementasi rumus
transformer standar dari NOL (RMSNorm → matvec Q/K/V → bias → [shortcut: V
saja, karena `cached_len==1` membuat softmax satu skor selalu `1.0`, jadi
RoPE/Q/K terbukti secara matematis tidak berpengaruh ke attn_out] → wo →
residual → RMSNorm → SwiGLU FFN → residual), pakai HANYA fungsi kernel
publik `glproc::kernels::*` (rms_norm_into, matvec, silu_mul) — bukan kernel
native-quantized/bridge apa pun, dan bukan salinan kode dari
`GlprocBackend`/`Runner` manapun.

Diberi makan bobot layer 2 (dequant ulang dari GGUF asli, sudah
diverifikasi byte-identik dengan package) + output real layer 1 Path A
sebagai input.

**Di layer 2**: replay F32 ini ≈ output real Path A (native-quantized)
— norm 700.9 vs 698.8, beda ~0.3%, sewajarnya noise float biasa.

**Lalu diulang di layer 0** (input = embedding, sudah terbukti byte-identik
antar kedua jalur) — hasilnya membalik semuanya:

```
F32-replay first8  = [0.0047937585, -0.04873758, -0.005444631, 0.023335831, ...]
Path B REAL first8 = [0.0047937585, -0.04873758, -0.005444631, 0.023335831, ...]   <- IDENTIK ke replay
Path A REAL first8 = [0.3573556,    0.06527804,  0.036348894, -0.03157193,  ...]   <- semua yang lain beda
```

**Replay F32 independen match PERSIS (banyak digit desimal) ke output real
`GlprocBackend`** — BUKAN ke `glproc::runner::Runner`. Pakai nilai bobot
yang sudah diverifikasi benar dan rumus textbook, `GlprocBackend` (yang
diasumsikan rusak sepanjang investigasi ini) mereproduksi matematika F32
ideal dengan sempurna di layer 0. Komputasi native-quantized (bridge Q5_0)
milik `glproc::runner::Runner` sendiri yang menyimpang dari matematika yang
benar.

# Verifikasi tambahan: bukan artefak posisi-0

Diulang dengan 3 token berurutan (id sama diulang 3x, KV cache tumbuh nyata
posisi 0→1→2, `cached_len` 1→2→3) untuk menyingkirkan "mungkin ini cuma
quirk attention trivial di posisi 0". **Pola divergensi identik muncul di
ketiga posisi** — mengonfirmasi ini bukan soal posisi/KV-cache sama sekali,
murni perbedaan komputasi per-token-per-layer, ada sejak layer 0, tidak
bergantung posisi sequence.

# Interpretasi ulang layer 2's "massive activation match"

Kecocokan replay F32 dengan Path A REAL di layer 2 (dari catatan
sebelumnya) sekarang lebih masuk akal dijelaskan sebagai: di layer 2,
apa pun masalah numerik yang ada di jalur native-quantized Path A sudah
teramplifikasi ke skala yang sangat besar sehingga perbedaan presisi lebih
lanjut jadi relatif kecil — BUKAN bukti bahwa komputasi Path A "pada
dasarnya benar".

# Implikasi — kenapa status ini masih "open", bukan "resolved"

Ini SATU token, SATU model. Sebelum sepenuhnya diyakini:

1. **Prioritas tertinggi**: jalankan `glproc::runner::Runner` (`gwen run`/
   `GlprocEngine`) langsung di file GGUF Q4_K_M yang PERSIS sama ini,
   prompt nyata, dan cek APAKAH BENAR outputnya koheren. Klaim "known-good,
   coherent output" untuk `Runner` mungkin tidak pernah diverifikasi ulang
   untuk file/kuantisasi spesifik ini.
2. Kalau `Runner` ternyata JUGA tidak koheren di file ini → bug pindah
   total ke `glproc`'s kernel native-quantized (Q5_0/Q4_K bridge/qdot),
   bukan glictus-caliburni sama sekali.
3. Kalau `Runner` TETAP koheren → re-audit `replay_full_layer_f32` untuk
   kemungkinan salah transkripsi (meski kecocokan byte-per-byte
   many-decimal-places dengan `GlprocBackend` REAL membuat ini kecil
   kemungkinannya — sebuah replay yang salah rumus tidak akan kebetulan
   cocok sedetail itu).

**JANGAN mulai kerja perbaikan di `GlprocBackend`/glictus-caliburni sampai
poin 1 di atas diverifikasi.** Lihat
[gllm-e2e-garbage-output.md](../issues/gllm-e2e-garbage-output.md) untuk
status investigasi keseluruhan.
