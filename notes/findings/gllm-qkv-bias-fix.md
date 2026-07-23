---
title: "GlprocBackend tidak pernah fetch/apply attention QKV bias"
status: resolved
severity: high
found: 2026-07-23
resolved: 2026-07-23
resolution: "tambah ATTN_Q_BIAS/K_BIAS/V_BIAS + optional_tensor + bias-add sebelum RoPE"
blocking: "TIDAK menyelesaikan notes/issues/gllm-e2e-garbage-output.md — bug real, tapi bukan penyebab utama"
files:
  - glictus-caliburni/src/runtime/glproc_backend.rs
---

# Problem

Diff sistematis operasi-per-operasi antara `glproc::runner::Runner::step`
(dianggap known-good saat itu) dan `GlprocBackend::execute_layer` menemukan
satu divergensi struktural yang jelas: `GlprocBackend::execute_layer` tidak
pernah fetch tensor bias (`attn_q.bias`, `attn_k.bias`, `attn_v.bias`) dan
tidak pernah menambahkannya ke Q/K/V — padahal `glproc::runner::Runner`
melakukannya tepat setelah matvec Q/K/V, sebelum RoPE (`runner.rs` ~847-861).

Qwen2 (arsitektur model referensi) membawa bias ini di setiap layer — QwenV2
dikenal sebagai salah satu keluarga model yang pakai attention bias (beda
dari kebanyakan model turunan LLaMA). Dikonfirmasi 3 cara independen:
1. `glproc`'s loader (`loader.rs:638-640`) baca tensor ini tanpa syarat per layer.
2. `glictus-caliburni/src/plugin.rs:221-222` — komentar dokumentasi eksplisit:
   "Biases (attn_q.bias and friends) are present in Qwen2 but absent in many
   other models" + test `standard_transformer_accepts_a_real_qwen2_layer`.
3. Grep langsung ke manifest package hasil konversi nyata: semua 24 layer
   punya `attn_q.bias`/`attn_k.bias`/`attn_v.bias`.

`GlprocBackend` cuma fetch 9 tensor (`ATTN_NORM`, `ATTN_Q/K/V`,
`ATTN_OUTPUT`, `FFN_NORM`, `FFN_GATE/UP/DOWN`) — tidak ada konstanta bias
sama sekali di file itu sebelum fix ini.

# Kenapa lolos dari semua test sebelumnya

Fixture test `glproc_backend.rs` (`fixture_layer_tensors`) tidak pernah
menyertakan tensor bias — jadi jalur kode "bias hilang" ini punya nol
cakupan test di kedua arah (tidak pernah membuktikan benar maupun salah).

# Resolusi

Ditambahkan ke `glproc_backend.rs`:
- Konstanta `ATTN_Q_BIAS`/`ATTN_K_BIAS`/`ATTN_V_BIAS` = `"attn_q.bias"` dst
  (titik, bukan underscore — diverifikasi langsung ke manifest package nyata
  sebelum di-hardcode, karena tebakan salah akan diam-diam tidak pernah
  match).
- `optional_tensor()`: tensor absen → `None` (skip diam-diam — banyak
  arsitektur memang tidak punya bias ini), tensor ada tapi gagal decode →
  tetap propagate error (optional cuma soal keberadaan, bukan "abaikan
  kegagalan").
- Bias-add persis di antara matvec Q/K/V dan RoPE, sama urutan seperti
  `Runner::step`.
- 2 test baru: `execute_layer_applies_attention_bias_when_present` (bias
  ada → output beda dari tanpa bias) dan
  `execute_layer_missing_bias_tensors_do_not_error_and_match_explicit_zero_bias`
  (bias absen ≡ bias eksplisit nol, bukan sekadar "tidak error").

277 test glictus-caliburni (dari 275), semua hijau, clippy bersih.

# Verifikasi E2E — TIDAK menyelesaikan gejala

Re-run `run_package_e2e` setelah fix: output **berubah**
(`葳ं pu Sessions蕉吸引)test bu比利EdgeInsets` → `. —iulp']6 --_,alaria皲
wrong2!?-ematbel kut;`) — bukti fix ini beneran berpengaruh secara numerik —
tapi **tetap tidak koheren**. ini bug real ke-3 yang ditemukan dan
diperbaiki tanpa menyelesaikan gejala utama; lihat
[gllm-e2e-garbage-output.md](../issues/gllm-e2e-garbage-output.md) untuk
status investigasi lengkap.
