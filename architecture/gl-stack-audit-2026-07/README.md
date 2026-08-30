# Mensura Veritatis v3

Audit korektnes gl-stack, dipicu oleh fix bug Q6_K dequant (PR #16, merged 2026-07-23 —
lihat `notes/issues/gllm-e2e-garbage-output.md`, resolved). Kelas bug itu: dua
implementasi independen dari hal yang sama, diam-diam beda tafsir, tidak ada test
yang membandingkan keduanya. Dokumen-dokumen di sini mengaudit di mana lagi pola itu
berisiko, dan apa yang perlu diukur sekarang setelah runtime `.gllm` akhirnya
menghasilkan output koheren untuk pertama kalinya.

Mengikuti disiplin evidence `docs/Mensura_Veritatis.md` v1: `[M]` = diukur/dibaca
langsung, `[R]` = klaim dari riset/dokumen lain, `[ER]` = Evidence Required (bukan
spekulasi), `[C]` = spec vs implementasi berbeda. Bukan revisi seri `architecture/GLLM_ARTX/`
(ARTX1-11) yang sudah ada — itu spesifikasi desain; seri ini audit apakah kenyataan
kodenya cocok.

| Dokumen | Isi |
|---|---|
| [ARTX1-Arsitektur.md](ARTX1-Arsitektur.md) | RoPE style hardcoded NeoX di `.gllm`, `rope_scaling` tidak pernah dipakai, gap `attention.key_length`, SwiGLU-only tanpa GELU |
| [ARTX2-Quant.md](ARTX2-Quant.md) | Peta 7 implementasi Q6_K independen di seluruh workspace, 3 lokasi bug masih reachable pasca-fix, audit format lain (Q4_K/Q5_0/Q4_0/Q8_0/Q8_K/Q4_1) |
| [ARTX3-Format.md](ARTX3-Format.md) | `.gllm` package: ZIP archive dispec tapi tak diimplementasi, tokenizer belum ter-package, drift tabel dtype, kenapa checksum/validator tidak bisa nangkep bug Q6_K |
| [ARTX4-Benchmark.md](ARTX4-Benchmark.md) | Pridwen §12 Known Unknowns yang sekarang unblocked, riset industri (KL-divergence, task-eval, studi pembanding), gap `glbench` (dead-code comparison fns, nol atribusi per-format) |

Status per 2026-07-23: riset/audit selesai, rekomendasi tercatat di tiap dokumen,
belum ada implementasi/fix yang dikerjakan dari sini (kecuali Q6_K sendiri, yang
sudah merged sebelum dokumen ini ditulis).
