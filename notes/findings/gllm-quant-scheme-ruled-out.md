---
title: "Kuantisasi (GQ4A/GQ2A/native GGUF) disingkirkan sebagai penyebab garbage output"
status: resolved
severity: high
found: 2026-07-23
resolution: "diagnostic --quant F32 dibangun di glconv, package murni F32 tetap garbage; --quant None (native GGUF quant) malah gagal load"
files:
  - glictus-caliburni/src/converter.rs
  - glictus-caliburni/src/bin/glconv.rs
---

# Problem

Sebelum sesi ini, kandidat utama penyebab garbage output adalah
ketidakbenaran dequantisasi GQ4A/GQ2A (kuantisasi native Pridwen). Perlu
kontrol group yang bersih: package `.gllm` dengan NOL kuantisasi sama sekali
untuk memastikan apakah GQ4A/GQ2A memang penyebabnya.

# `--quant F32` — diagnostic baru

Ditambahkan `QuantTarget::F32` ke `converter.rs` (ditandai
`/// Diagnostic only` di kode) — setiap tensor dipaksa dequant ke F32 asli,
apa pun dtype sumber GGUF-nya (Q4_K/Q5_0/Q6_K/Q8_0/dst), tanpa encoder
GQ4A/GQ2A tersentuh sama sekali. Beda dari `QuantTarget::None` (default) yang
cuma mempertahankan dtype asli GGUF tanpa dequant.

`glconv <gguf> <out> --quant F32` mencetak peringatan wajib:
`[warn] --quant F32 produces uncompressed packages for diagnostic use only`.

Hasil pada `qwen2.5-0.5b-instruct-q4_k_m.gguf`: package 2.52 GB (vs 491 MB
sumber, ~5.1x — wajar untuk F32 tanpa kompresi). Re-run `run_package_e2e`:
**tetap garbage** — `葳ं pu Sessions蕉吸引)test bu比利EdgeInsets`, karakter
sama dengan output GQ4A_CPP.

**Kesimpulan: kuantisasi GQ4A/GQ2A bukan penyebab.** Package tanpa kuantisasi
sama sekali pun tetap menghasilkan garbage.

# `--quant None` (default, tanpa flag) — malah gagal load, bukan garbage

Untuk kelengkapan, dites juga mode DEFAULT (`glconv` tanpa `--quant` sama
sekali) — package mempertahankan dtype native GGUF: 121 F32, 133 Q5_0,
12 Q4_K, 12 Q6_K, 13 Q8_0 (skema Q4_K_M asli llama.cpp, utuh).

Percobaan load lewat `run_package_e2e` **gagal total, bukan garbage**:

```
load .gllm package: Parse("tensor \"token_embeddings\": Unsupported dtype:
\"Q5_0 (tensor token_embeddings) — GLLM Wave 1 CPU path computes in f32 only\"")
```

Ini mengonfirmasi (sudah dicurigai sejak smoke test `quant-info` Wave 1):
`GllmEngine`/`GlprocBackend` cuma bisa dequant GQ4A/GQ2A (kernel Pridwen)
atau F32/F16/BF16 (`glcore::decode_tensor`) — dtype native GGUF (Q4_K/Q5_0/
Q6_K/Q8_0) sama sekali tidak punya jalur dequant di sisi glictus-caliburni.
Package `--quant None` karena itu tidak bisa dipakai sebagai test case untuk
bug garbage sama sekali — dia mati sebelum sempat menjalankan satu layer
pun.

# Implikasi penting: confound presisi itu struktural, bukan kebetulan

Temuan `--quant None` gagal-load ini mengonfirmasi sesuatu yang penting buat
investigasi lanjutan (`diff_dump.rs`): runtime `.gllm` **cuma bisa** hitung
di F32, titik — tidak ada cara membuat `GlprocBackend` menghitung di format
quantized (itu di luar scope Wave 1-nya, bukan celah untuk ditutup).
`glproc::runner::Runner` sebaliknya PUNYA kernel native quantized (bridge/
qdot) dan memakainya secara default. Jadi perbandingan presisi yang benar-
benar sama byte itu cuma bisa dicapai dengan memaksa **Path A** (glproc)
juga hitung F32 murni — bukan mengubah Path B. Detail lanjutan ada di
[gllm-f32-replay-reversal.md](gllm-f32-replay-reversal.md).
