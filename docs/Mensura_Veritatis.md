# Mensura Veritatis

**Canonical Knowledge Base — GwenLand `glproc`**
Status: v1 · Basis evidence: 1 dokumen riset + hasil pengukuran produksi glproc

---

## Pendahuluan

### Tujuan

Dokumen ini adalah **basis pengetahuan kanonik** untuk backend CPU `glproc`. Ia bukan ringkasan dan bukan hasil penelitian. Ia adalah **sintesis** dari penelitian yang sudah ada, disusun agar setiap keputusan rekayasa `glproc` di masa depan dapat ditelusuri kembali ke sini.

Nama dokumen ini diambil dari kodenama observability stack glproc (`glbench` — *Mensura Veritatis*, "ukuran kebenaran"), dan prinsip itu berlaku di sini juga: **klaim tanpa evidence bukan pengetahuan.**

### Sifat dan Batasan Evidence

Dokumen ini disusun di bawah aturan ketat berikut:

- Seluruh isi berasal dari evidence yang tercantum. Tidak ada fakta baru, tidak ada asumsi baru.
- Bila dua sumber bertentangan, **keduanya ditampilkan** dan ditandai `Research Conflict`. Tidak ada pemenang yang dipilih secara sepihak.
- Bila suatu topik tidak memiliki evidence yang cukup, ia ditandai `Evidence Required`. Tidak ada spekulasi.

### Peringatan Penting tentang Basis Evidence (WAJIB DIBACA)

Dokumen ini disusun dari **dua kelas evidence yang sangat berbeda kualitasnya**, dan pembaca wajib membedakannya:

| Kelas | Sumber | Sifat |
|---|---|---|
| **[R]** Research | ARTX04-CPUQuantArch (proposal arsitektural) | **Sekunder.** Klaim arsitektural umum, tidak diukur pada hardware glproc. |
| **[M]** Measured | Pengukuran produksi glproc + glbench | **Primer.** Diukur langsung pada target hardware, dapat direproduksi. |

**ARTX04 adalah dokumen *proposal*, bukan laporan pengukuran.** Ia tidak memuat data mentah, tidak memuat metodologi eksperimen yang direproduksi, dan sebagian klaimnya **telah dibantah oleh pengukuran langsung pada hardware target glproc**. Konflik-konflik tersebut didokumentasikan di bab [Research Conflict](#research-conflict) dan **harus dibaca sebelum bab manapun diikuti sebagai panduan implementasi**.

Knowledge Index yang diminta mencakup ~20 topik (FP32, BF16, FP16, Q8_1, Q6_K, Q5_K, Q3_K, Q2_K, dst.). **Sebagian besar hanya disebut namanya di ARTX04 tanpa penjelasan teknis.** Topik-topik itu tetap dicantumkan di indeks, tetapi ditandai `Evidence Required` alih-alih diisi dengan pengetahuan dari luar dokumen input — sesuai mandat.

---

## Knowledge Index

Legenda status:
`[R]` evidence riset · `[M]` evidence terukur · `[R+M]` keduanya · `[ER]` **Evidence Required** · `[C]` mengandung **Research Conflict**

| # | Topik | Status | Bab |
|---|---|---|---|
| 1 | Memory-Bound Nature of LLM Decode | `[R+M]` | [→](#1-memory-bound-nature-of-llm-decode) |
| 2 | Roofline Model & Bandwidth Ceiling | `[R+M]` | [→](#2-roofline-model--bandwidth-ceiling) |
| 3 | CPU Microarchitecture | `[R+M]` | [→](#3-cpu-microarchitecture) |
| 4 | SIMD & Vectorization | `[R+M]` `[C]` | [→](#4-simd--vectorization) |
| 5 | Memory Hierarchy & Cache Locality | `[R+M]` | [→](#5-memory-hierarchy--cache-locality) |
| 6 | GGUF | `[R+M]` | [→](#6-gguf) |
| 7 | Quantization — Prinsip Umum | `[R+M]` | [→](#7-quantization--prinsip-umum) |
| 8 | Q8_0 | `[R+M]` | [→](#8-q8_0) |
| 9 | Q4_K | `[R+M]` `[C]` | [→](#9-q4_k) |
| 10 | Kernel — Fused Dequant-Multiply | `[R+M]` | [→](#10-kernel--fused-dequant-multiply) |
| 11 | Kernel Dispatch & ISA Paths | `[R+M]` `[C]` | [→](#11-kernel-dispatch--isa-paths) |
| 12 | Threading | `[R+M]` `[C]` | [→](#12-threading) |
| 13 | Runtime & Memory Planning | `[R+M]` | [→](#13-runtime--memory-planning) |
| 14 | Benchmark & Observability | `[R+M]` | [→](#14-benchmark--observability) |
| 15 | Mixture-of-Experts (MoE) | `[M]` | [→](#15-mixture-of-experts-moe) |
| — | **FP32** | `[ER]` | [→](#knowledge-gap) |
| — | **BF16** | `[ER]` | [→](#knowledge-gap) |
| — | **FP16** | `[ER]` | [→](#knowledge-gap) |
| — | **Q8_1** | `[ER]` | [→](#knowledge-gap) |
| — | **Q6_K** | `[ER]` | [→](#knowledge-gap) |
| — | **Q5_K** | `[ER]` | [→](#knowledge-gap) |
| — | **Q3_K / Q2_K** | `[ER]` | [→](#knowledge-gap) |
| — | **Sparsity, AMX, NUMA, Autotuning** | `[ER]` | [→](#knowledge-gap) |

---

## Core Knowledge

### 1. Memory-Bound Nature of LLM Decode

#### Ringkasan
Fase decode LLM decoder-only dibatasi oleh **bandwidth memori**, bukan kapasitas komputasi. Ini adalah premis dasar dari seluruh arsitektur `glproc` dan satu-satunya klaim ARTX04 yang **terkonfirmasi penuh** oleh pengukuran langsung.

#### Temuan Penting
- Decode didominasi GEMV dengan intensitas aritmetika sangat rendah (**~2 FLOPs/byte**). `[R]`
- Latensi didominasi transfer bobot DRAM→unit komputasi, bukan aritmetika. `[R]`
- Setiap pengurangan bit-width bobot menghasilkan **percepatan linier** terhadap throughput — *asalkan overhead dequantisasi tidak melebihi penghematan bandwidth*. `[R]`
- **Terukur:** setiap bobot dibaca **sekali per token**. Pada Qwen3-1.7B Q8_0 itu berarti **1.828 MB/token** yang wajib di-stream. `[M]`

#### Penjelasan Teknis
Prefill (pemrosesan prompt) dan decode (generasi token) adalah **dua beban kerja berbeda dengan bottleneck berbeda**, dan menggabungkannya dalam satu angka menyembunyikan keduanya:

- **Prefill** membatch token, sehingga setiap baris bobot di-stream sekali per *chunk*, bukan sekali per token. Ini menggeser prefill ke arah **compute-bound**. `[M]`
- **Decode** memproses satu token pada satu waktu. Tidak ada amortisasi. Seluruh model di-stream ulang setiap token. **Bandwidth-bound.** `[M]`

Konsekuensi langsung: **`ms/call` dan `share%` menjawab pertanyaan berbeda.** `share%` menyatakan *ke mana waktu pergi*; `ms/call` menyatakan *apakah sebuah stage lambat atau sekadar sering*. Keduanya divergen tajam — pada Qwen3-1.7B, `lm_head` berbiaya 12,8 ms/call (12× stage lain) tetapi hanya 14% decode, karena ia berjalan sekali per token sementara stage per-layer berjalan 28×. `[M]`

#### Hubungan dengan Topik Lain
```
Memory-Bound Decode
        ↓
Roofline (§2) ──→ menentukan atap teoretis
        ↓
Quantization (§7) ──→ satu-satunya cara menurunkan byte
        ↓
Kernel (§10) ──→ harus tidak menambah traffic
```

#### Insight
> **Kernel yang lebih cepat tidak menolong beban kerja yang menunggu DRAM.** Optimasi hanya bermakna bila ia mengurangi byte yang dibaca, atau mengisi *issue slot* yang menganggur selama stall.

#### Trade-off
Tidak ada. Ini adalah properti beban kerja, bukan pilihan desain.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Landasan Teori — "Memory-Bound Nature of LLM Decode"
    Section: Pendahuluan

[M] Measured: glproc production profile, 2026-07-14
    Model: Qwen3-1.7B Q8_0 (L28, D2048, FFN6144, GQA 16/8, hd128)
    Tool:  GLPROC_PROFILE=1 glbench run --kind decode --tokens 64
    Data:  1.828 MB/token dihitung dari dimensi GGUF; decode 5.968 ms total
```

---

### 2. Roofline Model & Bandwidth Ceiling

#### Ringkasan
Roofline adalah kerangka analitik yang memprediksi performa dari bandwidth memori dan intensitas aritmetika. ARTX04 merekomendasikannya sebagai alat evaluasi. **glproc telah mengadopsinya dan memiliki angka ceiling terukur.**

#### Temuan Penting
- Roofline memvalidasi bahwa kuantisasi Q4 menggeser titik operasi decode dari zona *memory-bound* menuju *balanced*. `[R]`
- **Ceiling terukur pada hardware baseline (i3-1115G4): 29,4 GB/s** (read). `[M]`
- **Utilisasi aktual glproc: 23,0 GB/s pada bucket FFN = 78% dari ceiling.** `[M]`
- glbench melaporkan `bottleneck: undetermined` untuk CPU karena tidak memiliki ceiling terdaftar untuk perangkat CPU — angka 78% berasal dari pengukuran terpisah. `[M]`

#### Penjelasan Teknis
Roofline memberi glproc **kriteria berhenti yang objektif**. Sebuah stage yang berjalan pada 78% ceiling tidak punya ruang perbaikan kernel yang material: ruang teoretis maksimal hanya ~28%, dan itu pun mensyaratkan mencapai 100% bandwidth — yang tidak realistis.

Sebaliknya, stage yang berjalan jauh **di bawah** ceiling tanpa penjelasan adalah **anomali yang harus diinvestigasi**, bukan diterima. Inilah yang terjadi pada attention (lihat §4).

Aritmetika roofline juga berfungsi sebagai **detektor anomali**: bila share terukur sebuah stage jauh melampaui share byte-nya, ada sesuatu selain bandwidth yang membebaninya.

Contoh terukur (Qwen3-1.7B, sebelum perbaikan attention):

| stage | MB/token | share prediksi (byte) | share terukur | selisih |
|---|---|---|---|---|
| ffn_gate_up | 748,7 | 41,0% | 32,3% | −8,7 |
| ffn_down | 374,3 | 20,5% | 17,4% | −3,1 |
| lm_head | 330,6 | 18,1% | 13,7% | −4,4 |
| qkv | 249,6 | 13,7% | 13,9% | +0,2 |
| attn_out | 124,8 | 6,8% | 7,5% | +0,7 |
| **attention** | **~0** | **—** | **14,9%** | **+14,9** |

Attention hampir tidak membaca bobot (ia men-stream KV cache, yang jauh lebih kecil dari bobot). Di bawah model bandwidth murni ia seharusnya nyaris gratis. Ia menghabiskan 14,9%. **Selisih itulah yang mengungkap bug.** `[M]`

#### Hubungan dengan Topik Lain
```
Roofline (§2)
    ↓
    ├──→ Memory Hierarchy (§5): ceiling ditentukan DRAM + cache
    ├──→ Benchmark (§14): roofline adalah kriteria PASS/FAIL, bukan sekadar angka
    └──→ Quantization (§7): satu-satunya cara menggeser titik operasi
```

#### Insight
> **Roofline bukan sekadar alat evaluasi — ia adalah detektor bug.** Stage yang biayanya tidak dapat dijelaskan oleh byte yang dibacanya sedang menyembunyikan sesuatu.

#### Trade-off
Roofline mengasumsikan beban kerja yang *saturating*. Ia tidak memodelkan stall latensi (cache miss dingin, dependency chain), sehingga stage yang latency-bound akan tampak "di bawah ceiling" padahal masalahnya bukan bandwidth.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Kajian Literatur — "Roofline Analysis for LLM Inference"
    Section: Landasan Teori

[M] Measured: project_phase2_bandwidth_baseline, 2026-07-07
    Data:  i3-1115G4 read ceiling = 29,4 GB/s

[M] Measured: glproc FFN bandwidth, 2026-07-14
    Data:  ffn_gate_up 748,7 MB/tok @ 32,6 ms/tok = 23,0 GB/s (78% ceiling)
           ffn_down    374,3 MB/tok @ 16,3 ms/tok = 23,0 GB/s (78% ceiling)
```

---

### 3. CPU Microarchitecture

#### Ringkasan
`glproc` menargetkan dua tier: baseline **Intel Tiger Lake (i3-1115G4)** dan validation tier **Xeon server-grade**.

#### Temuan Penting
- Baseline: i3-1115G4 — 2 physical / 4 logical core, AVX2 + FMA3. `[R]`
- **Terukur:** CPU yang sama **juga melaporkan `avx512f`, `avx512bw`, `avx512vnni`, `avx512vl`**. `[M]`
- **Terukur:** glproc **menolak** AVX-512 lebar penuh pada part core-rendah karena downclock termal — heuristik: AVX-512 hanya dipakai bila logical core > 8. `[M]`
- **Terukur:** ceiling baca DRAM 29,4 GB/s; L2 = 1,25 MB/core. `[M]`
- Validation tier Xeon: AVX-512/VNNI/AMX. `[R]` (belum divalidasi — lihat `Evidence Required`)

#### Penjelasan Teknis
ARTX04 mengategorikan Tiger Lake sebagai "AVX2+FMA3". **Pengukuran menunjukkan itu tidak lengkap:** part ini memiliki AVX-512 *dan* VNNI. Yang benar adalah:

- **AVX-512 lebar-512-bit ditolak** karena menurunkan frekuensi (~2,5 GHz vs ~3,5 GHz), sehingga 4-thread AVX2 mengalahkannya. `[M]`
- **VNNI pada lebar 256-bit (EVEX) TETAP DIPAKAI.** Bentuk 256-bit berjalan pada *frequency license* yang sama dengan AVX2 — ia bukan datapath 512-bit, sehingga kekhawatiran termal AVX-512 tidak berlaku. `[M]`

Perbedaan ini material: `glproc` menjalankan `vpdpbusd` (VNNI) di seluruh jalur integer-dot-nya, bukan AVX2 murni.

**Kapabilitas ≠ pilihan.** glbench melaporkan keduanya sebagai field terpisah, karena menggabungkannya membuat pertanyaan "kenapa lambat di mesin AVX-512?" tidak terjawab. `[M]`

#### Hubungan dengan Topik Lain
```
CPU Microarchitecture (§3)
    ↓
SIMD (§4) ──→ ISA menentukan kernel mana yang tersedia
    ↓
Kernel Dispatch (§11) ──→ deteksi runtime, bukan compile-time
    ↓
Threading (§12) ──→ physical vs logical core (LIHAT RESEARCH CONFLICT)
```

#### Insight
> **Kapabilitas ISA yang dilaporkan CPU tidak sama dengan ISA yang sebaiknya dipakai.** Sebuah part boleh mengiklankan AVX-512 dan tetap lebih cepat dengan AVX2 — tetapi bentuk 256-bit dari instruksi AVX-512 (VNNI) bisa jadi tetap menang. Ketiganya harus dibedakan.

#### Trade-off
Heuristik "AVX-512 hanya bila >8 logical core" adalah **proxy kasar** untuk kelas TDP, bukan pengukuran downclock langsung. Ia bisa salah pada part masa depan.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Pendahuluan (target hardware tier)
    Section: Landasan Teori — "ISA-Specific Optimization Paths"

[M] Measured: glproc simd_strategy.rs + capability probe, 2026-07-14
    Data:  isa avx512f avx512bw avx512vnni avx2 fma f16c
           has_vnni_256() = true (avx512vnni AND avx512vl keduanya ada)
           [simd] strategy: Avx2+vnni256
```

---

### 4. SIMD & Vectorization

#### Ringkasan
Vektorisasi adalah mekanisme utama untuk mengeksekusi kuantisasi sub-byte secara efisien. **`Research Conflict` tercatat pada topik ini.**

#### Temuan Penting
- VNNI (`VPDPBUSD`) memungkinkan fused int8 multiply-accumulate, menghilangkan dequant eksplisit untuk Q8_0. `[R]` — **terkonfirmasi terukur** `[M]`
- Untuk format lebih rendah (Q4_K, Q3_K), *vectorized unpack* dengan register shuffling tetap diperlukan. `[R]`
- LUT berbasis SIMD lebih efisien daripada bitwise shifting untuk dekoding nibble pada arsitektur tanpa instruksi int4 native. `[R]`
- **Terukur:** `vpdpbusd` menggantikan pasangan `maddubs` + `madd` dengan satu instruksi. Kernel Q8_0 glproc memakainya. `[M]`
- **Terukur:** vektorisasi setengah-jalan lebih berbahaya daripada tidak sama sekali — lihat kasus attention di bawah. `[M]`

#### Penjelasan Teknis

**Kasus terdokumentasi: setengah kernel ter-vektorisasi.** Attention single-query glproc memiliki dua bagian:
1. `Q·K` — sudah AVX2.
2. Akumulasi `V` — **loop skalar murni** (`out[d] += w * v_row[d]`, satu float per iterasi).

Separuh aritmetika attention meninggalkan unit vektor menganggur. Akibatnya attention berjalan pada **0,83 GMAC/s** sementara `qkv` pada mesin yang sama berjalan **18,1 GMAC/s — 22× lebih cepat**. `[M]`

Setelah akumulasi V di-SIMD-kan (tile `head_dim` 32-lebar = 4 register YMM, akumulator ditahan di register sepanjang sumbu `t`, sehingga `out` ditulis sekali per tile alih-alih round-trip memori per baris):

| | sebelum | sesudah |
|---|---|---|
| attention ms/call | 496 µs | **253–329 µs** (~1,9×) |
| attention share | 14,9% | **7,5%** |
| **decode end-to-end** | median 7,5–8,6 tok/s | **median 10,1–11,8** (**+35%**) |

Validasi silang pada arsitektur berbeda (Qwen2.5-0.5B, hd64, 2 KV head): **4,08 GMAC/s (1.7B) vs 4,80 GMAC/s (0.5B)** — konsisten, bukan kebetulan bentuk. `[M]`

> ⚠️ **`share%` TIDAK dapat dibandingkan antar model.** Qwen2.5-0.5B menunjukkan attention share *lebih tinggi* (9,3%) daripada Qwen3-1.7B (7,5%) meski kerjanya lebih sedikit — karena rasio FFN-nya 5,4× d_model (vs 3,0×), sehingga penyebutnya menyusut lebih cepat. **Bandingkan GMAC/s, bukan share.** `[M]`

#### Hubungan dengan Topik Lain
```
SIMD (§4)
    ↓
    ├──→ Q8_0 (§8): VNNI vpdpbusd = dot integer tanpa dequant
    ├──→ Q4_K (§9): butuh vectorized unpack (BELUM ADA di glproc)
    └──→ Kernel (§10): fusi hanya bermakna bila SIMD penuh
```

#### Insight
> **Kernel yang setengah ter-vektorisasi adalah bug performa yang tersembunyi.** Ia lolos semua uji korektness, tidak muncul di profil sebagai "lambat" secara mencolok, dan hanya terungkap bila throughput (GMAC/s) dibandingkan antar-stage pada mesin yang sama.

#### Trade-off
Menulis kernel SIMD per-ISA meningkatkan beban pemeliharaan. ARTX04 mengestimasi kernel generik kehilangan **20–40%** performa `[R]` — angka ini **belum diverifikasi** pada glproc.

#### `Research Conflict` #1 — Klasifikasi ISA Tiger Lake
Lihat [Research Conflict §C1](#c1--klasifikasi-isa-tiger-lake).

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Kajian Literatur — "SIMD dan Vectorization untuk Sub-Byte Data"
    Section: Analisis Kritis — "Portability vs Performance"

[M] Measured: glproc attention SIMD fix, commit d1942b7, 2026-07-14
    Kernel: glproc/src/kernels/ops/attn_accum/{avx2,scalar}.rs (5 uji paritas)
    Data:   attn 496µs → 253-329µs; decode +35%; 0,83 → 4,08 GMAC/s
    Cross:  Qwen2.5-0.5B 4,80 GMAC/s (hd64) — konsisten lintas arsitektur

[M] Measured: glproc/src/kernels/qdot/q8_0/vnni.rs
    Data:   vpdpbusd menggantikan maddubs+madd; 256-bit EVEX
```

---

### 5. Memory Hierarchy & Cache Locality

#### Ringkasan
Layout memori adalah **objek optimisasi kelas satu**, bukan detail implementasi.

#### Temuan Penting
- Akses sequential jauh lebih efisien daripada acak (prefetcher hardware + burst DRAM). `[R]`
- Format blok dirancang agar metadata skala/nol dan data bobot berada dalam **cache line yang sama**. `[R]`
- GEMM terkuantisasi sering dibatasi **L1-cache-read bandwidth**, bukan kapasitas komputasi. `[R]`
- **Terukur:** DDR4 single-channel memberi imbalan besar pada *sedikit stream sekuensial*. Chunk kontigu per thread mengalahkan baris ter-interleave sebesar **~35% end-to-end** (18,6 → 25,3 tok/s). `[M]`
- **Terukur:** software prefetch (menarik stream ~16 blok / 544 B ke depan) memberi **+2 tok/s** — prefetcher hardware berhenti di batas 4 KiB page, prefetch software menjembataninya. `[M]`

#### Penjelasan Teknis: Jebakan Layout KV Cache

**Ini adalah temuan terpenting di bab ini, dan ia hampir menyesatkan pengembangan glproc.**

`KvCache` mengalokasikan untuk `max_context` (4096), bukan konteks aktif. Akibatnya region tiap head terpisah **2 MB** — sementara **L2 hanya 1,25 MB**. Membaca 16 head berarti melompat 2 MB antar head: **setiap head dimulai dari cache dingin**, meskipun data hidupnya hanya ~50 KB. `[M]`

Benchmark yang mengemas KV secara rapat (`cached_len × head_dim`) **menyembunyikan ini sepenuhnya** dan mengukur beban kerja yang tidak pernah dijalankan engine. Tiga probe atas perbaikan yang sama memberi jawaban berbeda **hanya karena layout ini**:

| layout KV di benchmark | fix A (SIMD) | fix B (threading) |
|---|---|---|
| rapat (`cached × head_dim`) | 0,93× | **0,07×** ← "jangan pernah threading" |
| stride produksi, 1 layer, panas | 1,29× | 2,40× ← "threading, menang besar" |
| dingin, rotasi 28 layer, ctx nyata | 1,21× | 1,71× ← realistis |

Mempercayai probe pertama akan menyimpulkan **kebalikan dari kebenaran**. `[M]`

#### Hubungan dengan Topik Lain
```
Memory Hierarchy (§5)
    ↓
    ├──→ GGUF (§6): blocked layout dipertahankan dari disk sampai kernel
    ├──→ Threading (§12): chunk kontigu > interleaved (single-channel DDR4)
    └──→ Benchmark (§14): benchmark yang salah layout = jawaban salah percaya diri
```

#### Insight
> **Benchmark harus mereproduksi layout memori produksi, bukan layout yang nyaman.** Buffer yang dikemas rapat mengukur beban kerja yang tidak pernah ada.

#### Trade-off
Mengalokasikan KV untuk `max_context` memberi kesederhanaan (tidak perlu realokasi) dengan biaya **cache locality yang buruk** pada konteks pendek. Trade-off ini **belum dievaluasi ulang**.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Landasan Teori — "Cache Locality dan Blocked Layout"
    Section: Kajian Literatur — "Memory Layout dan Cache Behavior"

[M] Measured: glproc/src/threading.rs (chunk kontigu vs interleaved)
    Data:   18,6 → 25,3 tok/s (~35%) pada Qwen2.5-0.5B

[M] Measured: glproc/src/kernels/qdot/q8_0/{avx2,vnni}.rs (software prefetch)
    Data:   +2 tok/s, A/B'd kedua arah

[M] Measured: glproc/benches/attn_probe.rs, 2026-07-14
    Data:   3 probe, 3 kesimpulan berbeda; L2 1,25 MB vs stride head 2 MB
```

---

### 6. GGUF

#### Ringkasan
GGUF adalah **standar de facto** dan **format interop primer** glproc.

#### Temuan Penting
- Dirancang khusus untuk inferensi: memory-mapping, alignment kontigu, metadata self-contained. `[R]`
- Berbeda dari format pelatihan (SafeTensors, Checkpoint). `[R]`
- mmap dengan alignment 32-byte menurunkan waktu loading dari menit ke detik. `[R]`
- **Terukur:** glproc membaca GGUF native; dimensi & hyperparameter dibaca dari metadata, **tidak di-hardcode**. `[M]`
- **Terukur:** tensor `output.weight` bisa **tidak ada** (tied embeddings) — glproc jatuh kembali ke `token_embd`. `[M]`

#### Penjelasan Teknis
Contoh terverifikasi dari dua model:

| model | `output.weight` | `token_embd.weight` |
|---|---|---|
| Qwen3-1.7B | **ABSENT** → tied ke `token_embd` | Q8_0, `[2048, 151936]` |
| Qwen2.5-0.5B | Q8_0, `[896, 151936]` | Q5_0, `[896, 151936]` |

`[M]`

**Jebakan terdokumentasi — layout tensor MoE (`_exps`) belum terverifikasi.** glproc mengasumsikan tensor `_exps` adalah 3-D `[in, out, n_expert]` dengan expert pada sumbu **terluar dan kontigu**. Asumsi ini ditulis dari konvensi llama.cpp, **bukan dari byte yang pernah dibaca siapa pun**. Bila expert justru ter-interleave pada sumbu lebih cepat, setiap expert akan termuat sebagai *stripe* dari semua expert lain — **dan model tetap BERJALAN, mengeluarkan sampah yang fasih.** `[M]`

Mitigasi yang ada: `split_experts` memeriksa silang dimensi terdeklarasi terhadap metadata `expert_count` dan panjang byte, sehingga *bentuk* yang salah tidak lolos. *Urutan penumpukan* yang salah namun memenuhi pemeriksaan itu **masih bisa lolos**. Ditandai di kode sebagai `_EXPS_LAYOUT_ASSUMPTION`.

#### Hubungan dengan Topik Lain
```
GGUF (§6)
    ↓
Quantization (§7) ──→ GGUF membawa format kuantisasi
    ↓
Runtime (§13) ──→ mmap + repack di load time
    ↓
MoE (§15) ──→ layout _exps BELUM TERVERIFIKASI
```

#### Insight
> **Format yang self-describing tetap bisa berbohong tentang hal yang tidak dideklarasikannya.** GGUF menyatakan dimensi, tetapi tidak menyatakan *urutan penumpukan* pada tensor 3-D. Asumsi di titik itu adalah risiko korupsi senyap.

#### Trade-off
Membaca GGUF native menghindari konversi, tetapi mengikat glproc pada evolusi format GGUF.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: State of the Art — "Format Penyimpanan"
    Section: Implikasi terhadap glproc — poin 5

[M] Measured: glproc/src/loader.rs; GGUF header dump, 2026-07-14
    Data:   Qwen3-1.7B: output.weight ABSENT (tied); token_embd Q8_0 [2048,151936]
            Qwen2.5-0.5B: output.weight Q8_0 [896,151936]

[M] Measured: glproc/src/loader.rs — _EXPS_LAYOUT_ASSUMPTION (UNVERIFIED)
```

---

### 7. Quantization — Prinsip Umum

#### Ringkasan
Kuantisasi bukan teknik kompresi tambahan — ia adalah **strategi arsitektural utama** untuk mengubah profil beban kerja.

#### Temuan Penting
- Weight-only 4-bit mempertahankan akurasi LLM >7B dengan degradasi **<1%** pada benchmark standar. `[R]`
- Distribusi bobot LLM memiliki **outlier channel** yang sensitif; AWQ memakai activation-aware scaling untuk melindunginya. `[R]`
- INT8 weight-only mencapai speedup **2–3× vs FP32** tanpa loss akurasi material (studi Intel pada Xeon). `[R]`
- K-quant memakai statistik super-block untuk akurasi lebih baik pada bit-width rendah. `[R]`
- I-quant (importance-aware) lebih akurat tetapi butuh data kalibrasi. `[R]`
- **Terukur:** glproc **me-repack** Q4_K/Q5_0/Q6_K → Q8_0 saat load, karena **hanya Q8_0 yang punya kernel integer-dot**. `[M]`

#### Penjelasan Teknis

**Konsekuensi paling penting dalam dokumen ini:**

Repack ke Q8_0 **membuang keunggulan ukuran** dari format sub-byte. Q4_K ≈ **0,56 byte/weight**; Q8_0 ≈ **1,06 byte/weight**. Model yang dikuantisasi ke Q4_K oleh pengguna tetap men-stream traffic setara Q8_0 di dalam glproc.

Karena decode adalah bandwidth-bound (§1) dan bucket panas sudah berjalan pada 78% ceiling (§2), **satu-satunya tuas yang tersisa adalah membaca lebih sedikit byte** — yaitu **kernel integer-dot Q4_K native**. Ia akan memotong traffic hampir separuh dan menyentuh **ketiga bucket terpanas sekaligus** (ffn_gate_up, ffn_down, lm_head). Tidak ada opsi lain yang menyentuh lebih dari satu. `[M]`

#### Hubungan dengan Topik Lain
```
Quantization (§7)
    ↓
    ├──→ Memory-Bound (§1): kuantisasi = pengurangan byte = percepatan linier
    ├──→ Kernel (§10): setiap format butuh kernel integer-dot-nya sendiri
    └──→ Q4_K (§9): TUAS UTAMA YANG BELUM DIAMBIL
```

#### Insight
> **Kuantisasi tanpa kernel native untuk format itu adalah kuantisasi yang dibuang.** Menyimpan Q4_K di disk lalu me-repack-nya ke Q8_0 di RAM menghemat ruang disk, bukan bandwidth — dan bandwidth adalah bottleneck-nya.

#### Trade-off
Repack ke Q8_0 memberi **kesederhanaan** (satu kernel integer-dot untuk semua format) dengan biaya **traffic 2×**. Trade-off ini rasional saat kernel Q4_K native belum ada; ia menjadi tidak rasional begitu ada.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Kajian Literatur — "Weight-Only Quantization untuk LLM"
    Section: Analisis Kritis — "Generic Quantization vs K-Quant/I-Quant"

[M] Measured: glproc/src/loader.rs — weight() repack rules
    Data:   Q4_K/Q5_0/Q6_K → Q8_0 di load time
            Q4_K ~0,56 B/weight vs Q8_0 ~1,06 B/weight
```

---

### 8. Q8_0

#### Ringkasan
Format kuantisasi **satu-satunya** yang saat ini memiliki jalur integer-dot di glproc. Seluruh bucket panas berjalan di atasnya.

#### Temuan Penting
- VNNI menghilangkan kebutuhan dequant eksplisit untuk Q8_0. `[R]`
- **Terukur:** blok 34 byte = f16 scale (2 B) + 32 × int8 quant. **~1,06 byte/weight.** `[M]`
- **Terukur:** dot dilakukan **di domain integer** — nol dequantisasi ke f32. `[M]`
- **Terukur:** jalur runtime terverifikasi: `[simd] Avx2+vnni256 | ffn gate/up: Q8_0 fused-swiglu integer-dot | ffn down: Q8_0 integer-dot | lm_head: Q8_0 integer-dot`. `[M]`
- **Terukur:** throughput **21,6 GMAC/s** (FFN) — lebih cepat dari `qkv` (18,1 GMAC/s). `[M]`

#### Penjelasan Teknis
Kernel Q8_0 glproc (`kernels/qdot/q8_0/vnni.rs`) sudah matang:

1. **Sign trick** — `vpdpbusd` butuh satu operand unsigned: `|w| ⊗ (a · sign(w))`.
2. **Dua akumulator bergantian** — satu akumulator akan ter-serialisasi pada latensi FMA 4-siklus; dua rantai membuat blok berturutan tumpang tindih di jendela out-of-order.
3. **Software prefetch** — menarik 544 B ke depan; +2 tok/s terukur.
4. **Aktivasi dikuantisasi ke Q8 sekali per matvec**, bukan per baris.

`[M]`

#### Hubungan dengan Topik Lain
```
Q8_0 (§8)
    ↓
SIMD (§4) ──→ vpdpbusd (VNNI 256-bit)
    ↓
Kernel (§10) ──→ fused dequant-multiply tercapai pada level instruksi
    ↓
Bandwidth (§2) ──→ 78% ceiling: MENTOK
```

#### Insight
> **Q8_0 di glproc sudah optimal.** Tidak ada perbaikan kernel yang tersisa. Setiap perbaikan lebih lanjut harus datang dari format yang **lebih kecil**, bukan kernel yang lebih cepat.

#### Trade-off
Q8_0 memberi akurasi tinggi dan kernel sederhana, dengan biaya **traffic 2× dibanding Q4_K**.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Kajian Literatur — "SIMD dan Vectorization"

[M] Measured: glproc/src/kernels/qdot/q8_0/vnni.rs; runtime log, 2026-07-14
    Data:   34 B/blok (2 B f16 scale + 32 int8); 21,6 GMAC/s; 23,0 GB/s
```

---

### 9. Q4_K

#### Ringkasan
**Tuas utama yang belum diambil.** ARTX04 menetapkannya sebagai *primary target*; glproc saat ini **me-repack-nya menjadi Q8_0** dan dengan demikian tidak memperoleh manfaat bandwidth-nya.

#### Temuan Penting
- K-quant memakai statistik super-block; akurasi lebih baik pada bit-width rendah dengan overhead metadata yang acceptable. `[R]`
- ARTX04 menetapkan **Q4_K_M sebagai primary target** dan menargetkan **≥90% performa llama.cpp** pada Tiger Lake (Sprint 2). `[R]`
- Butuh **vectorized unpack** (register shuffling / LUT SIMD) karena tidak ada instruksi int4 native. `[R]`
- **Terukur:** **glproc TIDAK memiliki kernel integer-dot Q4_K.** Ia me-repack ke Q8_0 di load time. `[M]`
- **Terukur:** ~0,56 byte/weight vs Q8_0 ~1,06 → potensi **~2× pengurangan traffic**. `[M]`

#### Penjelasan Teknis
> ## ⛔ HIPOTESIS INI TELAH DIUJI DAN **TERBANTAH** (2026-07-14)
>
> Bab ini semula menyimpulkan Q4_K native adalah "satu-satunya tuas tersisa".
> **Kernel itu dibangun, diuji paritas, diukur di produksi — dan KALAH 33%.**
> Penalaran keliru di bawah dipertahankan, karena kekeliruannya instruktif.

**Aritmetika yang meyakinkan (dan tetap benar).** Traffic terukur pada model Q4_K nyata (Qwen2.5-1.5B-q4_k_m — **75,7% Q4_K + 24,3% Q6_K**):

| skenario | MB/token | vs sekarang |
|---|---|---|
| repack semua → Q8_0 (sekarang) | 1.889 | 1,00× |
| Q4_K native, Q6_K→Q8_0 | 1.215 | **1,55×** |
| Q4_K + Q6_K keduanya native | 1.111 | 1,70× |

**Hasil aktual — kebalikannya:**

| jalur | tok/s (3 run) |
|---|---|
| repack → Q8_0 | **14,1 · 14,2 · 14,1** |
| Q4_K integer-dot native | 9,4 · 9,6 · 9,5 |

**Regresi 33%**, grup tidak tumpang tindih (BEFORE min 13,8 > AFTER max 9,6). Bukan thermal noise.

**Kenapa — diukur, bukan ditebak** (`benches/q4k_probe.rs`, kerja identik):

| format | GMAC/s | GB/s |
|---|---|---|
| Q4_K | **1,5–2,0** | **0,8–1,1** |
| Q8_0 | 3,3 | 3,5 |

Q4_K **1,7–2,2× lebih lambat per MAC**. Yang menentukan: **gap-nya sama besar saat data L2-resident.** Kalau ini efek memori, gap akan mengecil di cache. Ia tidak. **Unpack nibble memang lebih mahal secara compute daripada byte yang dihematnya** pada AVX2/VNNI-256. Q4_K bahkan tidak pernah mendekati bandwidth-bound (0,8–1,1 GB/s vs Q8_0 3,5).

**Dua perbaikan kernel yang "jelas benar" hanya memberi ~9%:**
1. Akumulasi di domain vektor (menghapus 8 horizontal-sum per super-blok): 8,7 → 9,1 tok/s.
2. Hoist `scale_min()` keluar loop chunk: 9,1 → 9,5 tok/s.

Dibutuhkan **2,7×** untuk menyamai baseline; keduanya bersama memberi 1,09×.

**Ini persis kegagalan yang ARTX04 sendiri peringatkan** (§Landasan Teori): kuantisasi memberi percepatan linier *"asalkan overhead dequantisasi tidak melebihi penghematan bandwidth"*. Kalimat itu dikutip di §1 dokumen ini — lalu dilanggar.

**Kernel tetap dipertahankan** (`kernels/qdot/q4_k/`, paritas terhadap dequant-penuh, 4 uji hijau) supaya jalur AVX-512/VNNI-512 atau unpack yang lebih lebar dapat dievaluasi terhadap baseline yang bekerja, bukan ditulis dari nol.

#### Hubungan dengan Topik Lain
```
Q4_K (§9)
    ↓
SIMD (§4) ──→ butuh vectorized unpack (BELUM ADA)
    ↓
Kernel (§10) ──→ butuh integer-dot native (BELUM ADA)
    ↓
Bandwidth (§2) ──→ ~2× lebih sedikit byte → menyentuh 3 bucket terpanas
```

#### Insight
> **Mengurangi byte hanya menolong bila kernelnya TETAP bandwidth-bound.** Q4_K memotong 1,89× byte dan tetap kalah, karena unpack nibble mendorongnya menjadi compute-bound. Roofline memberi *batas atas*, bukan *janji* — ia mengasumsikan kernel dapat mempertahankan throughput yang sama per byte, dan itulah asumsi yang gagal di sini.

#### Trade-off
Kernel Q4_K native lebih rumit (unpack nibble, super-block scales) dan menambah jalur yang harus dipelihara + diuji paritas. Biaya itu ditukar dengan ~2× bandwidth pada beban kerja yang **terbukti** bandwidth-bound.

#### `Research Conflict` #2 — Prioritas Sprint
Lihat [Research Conflict §C2](#c2--urutan-prioritas-sprint).

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Rekomendasi Implementasi — Sprint 2 (Q4_K_M primary target)
    Section: Rencana Benchmark — Quantization Formats
    Section: Kajian Literatur — "SIMD untuk Sub-Byte Data"

[M] Measured: glproc/src/loader.rs — Q4_K di-repack ke Q8_0
[M] Measured: FFN + lm_head @ 78% bandwidth ceiling, 2026-07-14
```

---

### 10. Kernel — Fused Dequant-Multiply

#### Ringkasan
Fusi dequant-multiply adalah **primitif wajib**, bukan optimasi opsional. Terkonfirmasi kuat oleh riset **dan** pengukuran.

#### Temuan Penting
- Dequant terpisah dari perkalian menyebabkan traffic memori intermediate yang tidak perlu. `[R]`
- Pada VNNI/AMX, fusi dapat dilakukan pada **level instruksi tunggal**. `[R]`
- Fusi GEMM+aktivasi (SiLU/ReLU) direkomendasikan. `[R]` (ARTX04 Sprint 4)
- **Terukur:** glproc sudah melakukan **fusi SwiGLU penuh** — dan ia jauh lebih agresif daripada yang ARTX04 usulkan. `[M]`

#### Penjelasan Teknis: Anatomi Fusi SwiGLU glproc

ARTX04 mengusulkan "fused GEMM+activation" sebagai pekerjaan Sprint 4. glproc **sudah melampauinya**:

1. **Interleaving di load time.** Baris gate dan up di-interleave dalam memori (`[gate row 0][up row 0][gate row 1]…`) — sehingga setiap thread membaca **SATU stream DRAM kontigu**, bukan dua region yang terpisah megabyte. Ini bukan fusi kernel; ini **fusi layout**. `[M]`
2. **Satu dispatch, dua dot.** `par_matvec_swiglu` — bukan dua matmul terpisah.
3. **SiLU inline di register.** Kedua dot masih di register saat SiLU diterapkan; hasilnya di-store **sekali**. Vektor gate/up antara **tidak pernah round-trip ke RAM**.
4. **`fast_exp`, bukan `f32::exp`** di jalur panas.

```rust
let g = row_dot_q8(fmt, &pair[..row_bytes], act, strategy);
let u = row_dot_q8(fmt, &pair[row_bytes..], act, strategy);
let s = g / (1.0 + fast_exp(-g)) * u;   // inline, di register
```
`[M]`

Hasil: FFN berjalan **21,6 GMAC/s / 23,0 GB/s (78% ceiling)** — **lebih cepat dari `qkv`**, stage yang bahkan tidak perlu difusikan. `[M]`

#### Hubungan dengan Topik Lain
```
Kernel (§10)
    ↓
    ├──→ Memory Hierarchy (§5): fusi layout > fusi kernel
    ├──→ SIMD (§4): fusi hanya bermakna bila SIMD penuh
    └──→ Bandwidth (§2): fusi menghapus traffic intermediate
```

#### Insight
> **Fusi terkuat terjadi di *layout*, bukan di kernel.** Meng-interleave baris gate/up saat load mengubah dua stream DRAM menjadi satu — sesuatu yang tidak bisa dicapai oleh fusi kernel manapun setelah bobot berada di memori.

#### Trade-off
Interleaving mengunci layout bobot ke pola akses SwiGLU. Bila operator lain butuh gate/up secara terpisah, ia harus membayar de-interleave.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Landasan Teori — "Fusi Dequant-Multiply sebagai Primitif Atom"
    Section: Implikasi terhadap glproc — poin 3
    Section: Rekomendasi Implementasi — Sprint 4

[M] Measured: glproc/src/threading.rs par_matvec_swiglu; model.rs GateUp::FusedQuant
    Data:   21,6 GMAC/s, 23,0 GB/s (78% ceiling)
```

---

### 11. Kernel Dispatch & ISA Paths

#### Ringkasan
Tidak ada jalur generik yang optimal untuk semua ISA. Dispatch harus berbasis kapabilitas **runtime**.

#### Temuan Penting
- Runtime harus mendeteksi CPU features saat startup dan memilih kernel path yang sesuai. `[R]`
- ARTX04: **"Tidak ada fallback generik untuk kernel inti GEMV/GEMM; jika ISA tidak didukung, error eksplisit lebih baik daripada performa yang menipu."** `[R]`
- **Terukur:** glproc memakai `SimdStrategy::detect()` — di-cache dalam `OnceLock` (satu atomic load), tidak pernah probe di jalur panas. `[M]`
- **Terukur:** glproc **memiliki** fallback skalar, dan menggunakannya sebagai **ground truth paritas** untuk setiap kernel SIMD. `[M]`

#### Penjelasan Teknis

**Divergensi kebijakan yang disengaja.** ARTX04 melarang fallback generik. glproc **mempertahankannya**, tetapi bukan untuk performa — untuk **korektness**:

Setiap kernel SIMD glproc memiliki pasangan skalar, dan uji paritas membandingkan keduanya. Contoh dari kernel `attn_accum` yang baru: 5 uji paritas mencakup semua tingkat `head_dim` (32-lebar utama, sisa 8-lebar, ekor skalar), plus properti *overwrite-not-accumulate*, plus penolakan membaca melewati `weights.len()`. Uji itu **mencetak backend terdeteksi** untuk memastikan ia tidak diam-diam hanya menjalankan jalur skalar. `[M]`

Ini **tidak bertentangan** dengan ARTX04 secara semangat — ARTX04 melarang fallback sebagai *jalur produksi yang menipu*. glproc memakainya sebagai *oracle pengujian*. Perbedaan ini harus dinyatakan eksplisit agar tidak ada yang menghapus kernel skalar atas nama "mengikuti ARTX04".

#### Hubungan dengan Topik Lain
```
CPU Microarchitecture (§3)
    ↓
Kernel Dispatch (§11) ──→ deteksi runtime, cached OnceLock
    ↓
    ├──→ SIMD (§4): pilih vnni / avx2 / scalar
    └──→ Benchmark (§14): uji paritas WAJIB pakai jalur skalar
```

#### Insight
> **Kernel skalar bukan jalur produksi — ia adalah *oracle*.** Menghapusnya demi "tidak ada fallback generik" berarti menghapus satu-satunya ground truth untuk memvalidasi kernel SIMD.

#### Trade-off
Memelihara jalur skalar menambah kode yang tidak pernah berjalan di produksi. Biayanya rendah; nilainya (paritas yang dapat diaudit) tinggi.

#### `Research Conflict` #3
Lihat [Research Conflict §C3](#c3--fallback-generik).

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Implikasi terhadap glproc — poin 2
    Section: Landasan Teori — "ISA-Specific Optimization Paths"

[M] Measured: glproc/src/simd_strategy.rs (OnceLock); kernels/*/scalar.rs (paritas)
```

---

### 12. Threading

#### Ringkasan
**Bab dengan konflik paling material dalam dokumen ini.** Riset dan pengukuran **bertentangan langsung**.

#### Temuan Penting — Sisi Riset `[R]`
- Hyperthreading/SMT **sering merugikan** untuk beban kerja memory-bound (kontensi cache dan bandwidth).
- Literatur XNNPACK dan studi kinerja CPU **konsisten** menunjukkan thread count optimal untuk LLM decode **= jumlah physical core**.
- Oversubscription meningkatkan latensi tail (p95/p99) **tanpa** meningkatkan throughput median.
- Kontensi threadpool antar runtime menyebabkan degradasi **hingga 30%**.
- Rekomendasi: **single owner threadpool** dengan explicit affinity binding.

#### Temuan Penting — Sisi Pengukuran `[M]`
- **Physical-core sizing DIUJI pada hardware target dan KALAH 23%.**
- Knee optimal terukur **bukan** physical (2) maupun logical (4), melainkan **3** pada beban kerja MoE.
- Chunk kontigu per thread mengalahkan interleaved **~35%**.
- glproc **sudah** memiliki single-owner threadpool (persisten, std-only, tanpa rayon/crossbeam).

#### Penjelasan Teknis
Lihat [Research Conflict §C4](#c4--physical-core-vs-logical-core-threading) untuk data lengkap. **Ini adalah konflik yang belum terselesaikan** dan tidak boleh dianggap selesai oleh pembaca manapun.

Yang **tidak** dalam sengketa: single-owner threadpool. Riset merekomendasikannya `[R]`, glproc mengimplementasikannya `[M]` — persisten, worker di-park pada condvar dengan spin budget, caller ikut sebagai thread 0.

#### Hubungan dengan Topik Lain
```
Threading (§12)
    ↓
    ├──→ CPU Microarchitecture (§3): physical vs logical
    ├──→ Memory Hierarchy (§5): chunk kontigu (single-channel DDR4)
    └──→ Benchmark (§14): thermal noise membalik hasil A/B
```

#### Insight
> **Hipotesis threading yang paling masuk akal pun harus diuji pada hardware target.** "Decode memory-bound, jadi SMT tidak menolong" terdengar benar dan **terbukti salah** di sini.

#### Trade-off
Lihat Research Conflict.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Landasan Teori — "Physical Core Threading Policy"
    Section: Kajian Literatur — "Threading dan Scheduling"
    Section: Implikasi terhadap glproc — poin 4

[M] Measured: glproc runner.rs N_THREADS A/B, 2026-07-14 (LIHAT KONFLIK C4)
[M] Measured: glproc/src/threading.rs (single-owner pool, chunk kontigu)
```

---

### 13. Runtime & Memory Planning

#### Ringkasan
Kuantisasi mengubah bukan hanya ukuran data tetapi juga **pola aksesnya**.

#### Temuan Penting
- Metadata skala/nol harus ditempatkan strategis agar tidak menyebabkan false sharing / cache thrashing. `[R]`
- Buffer reuse agresif dan lifetime management menjadi lebih kritis. `[R]`
- **Terukur:** glproc **nol alokasi heap per token** — semua buffer di `Workspace` pra-alokasi + `KvCache` berbasis cursor. `[M]`
- **Terukur:** `warm_and_lock_model` — sentuh setiap page bobot lalu **pin** (`VirtualLock`/`mlock`), agar tidak ada decode yang stall pada page fault / swap-in. `[M]`
- **Terukur:** alokasi `Vec` baru di jalur panas menyebabkan **stall 25–40 ms** yang acak (demand-zero page fault). Diperbaiki dengan `thread_local` scratch yang mempertahankan kapasitas. `[M]`

#### Penjelasan Teknis
Split memori terukur (Qwen3-1.7B): **model 2,01 GiB | KV cache 0,88 GiB**. Angka "peak RSS" tunggal tidak dapat menggantikan pemisahan ini — hanya KV cache yang tumbuh dengan panjang konteks. `[M]`

Jebakan terdokumentasi: `QuantizedActivation::quantize` **tidak menumbuhkan buffernya** — ia hanya `debug_assert` kecocokan, sehingga menulis **di luar batas dalam release**. Berbagi satu buffer aktivasi antara langkah dengan lebar berbeda merusak memori secara senyap pada model di mana `ffn > hidden`. `[M]`

#### Hubungan dengan Topik Lain
```
Runtime (§13)
    ↓
    ├──→ Memory Hierarchy (§5): pin page agar tidak fault mid-decode
    ├──→ GGUF (§6): mmap → copy → repack → pin
    └──→ Threading (§12): scratch per-thread, bukan alokasi per-panggilan
```

#### Insight
> **Alokasi di jalur panas bukan sekadar lambat — ia bersifat bimodal.** Page fault demand-zero muncul sebagai stall multi-milidetik yang acak, yang tersembunyi total di dalam rata-rata.

#### Trade-off
Pinning memori menjamin latensi tetapi mengurangi memori yang tersedia bagi sistem — relevan pada mesin 8 GB.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Landasan Teori — "Quantization-Aware Memory Planning"

[M] Measured: glproc/src/loader.rs warm_and_lock_model; runner.rs Workspace
[M] Measured: glproc/src/threading.rs — thread_local PACK scratch (stall 25-40ms)
[M] Measured: glbench memory telemetry — model 2,01 GiB | kv 0,88 GiB
```

---

### 14. Benchmark & Observability

#### Ringkasan
ARTX04 menetapkan rencana benchmark yang komprehensif. glproc **telah mengimplementasikan superset-nya** sebagai `glbench` (*Mensura Veritatis*).

#### Temuan Penting — Riset `[R]`
- Wajib memisahkan cold / warm / steady-state.
- Wajib melaporkan p50/p95/p99, throughput, cache miss rate, bandwidth utilization.
- Warm-up 10 run dibuang; 100 run diukur; outlier >3σ **diinvestigasi, bukan dibuang otomatis**.
- Seluruh konfigurasi didokumentasikan reproducible.

#### Temuan Penting — Terukur `[M]`
- glbench adalah **profiler**, bukan sekadar benchmark: capability report, stage timeline (ms / share / calls / **ms/call**), memory split, MoE routing, sinyal perilaku.
- Telemetry adalah **pull**, bukan callback: engine mengisi struktur data; glbench membacanya. Trait `BenchmarkReporter` **ditolak** karena membalik arah dependensi.
- Sinyal perilaku dihitung dari **logits mentah** — full vocab, sebelum temperature, sebelum top-k/top-p, sebelum repetition penalty.
- Tracing berjalan pada **run terpisah** karena ia mengganggu timing.
- `None` berarti **TIDAK DIUKUR**, bukan nol.

#### Penjelasan Teknis: Jebakan Pengukuran Terdokumentasi

**Tiga cara benchmark glproc menghasilkan jawaban yang percaya diri namun salah:**

1. **Layout KV.** Tiga probe atas perbaikan yang sama memberi 0,07× / 2,40× / 1,71× **hanya** bergantung pada layout benchmark. Lihat §5.
2. **Panjang konteks yang ditebak.** Asumsi ctx≈68; instrumentasi produksi mengatakan **mean ctx = 252** — meleset 4×, cukup untuk membuat semua angka sebelumnya tak bermakna.
3. **Thermal throttling.** i3-1115G4 adalah part 15 W. Hasil A/B pernah **terbalik** (arm "off" lebih lambat dari arm "on", tiga kali berturut-turut) murni karena panas.

**Standar bukti yang dihasilkan:** klaim percepatan memerlukan **grup A/B yang tidak tumpang tindih**, bukan sekadar rata-rata yang lebih baik. Contoh yang lolos: `min` sesudah (11,0 tok/s) melampaui `max` sebelum (10,2 tok/s). `[M]`

**Produksi dua kali mengalahkan prediksi probe** (attention SIMD: diprediksi 1,21×, tercapai 1,9×) — karena loop nyata juga men-stream 2 GB bobot per token, sehingga lebih memory-stalled daripada yang dapat direproduksi probe KV-saja. `[M]`

#### Hubungan dengan Topik Lain
```
Benchmark (§14)
    ↓
    ├──→ Roofline (§2): kriteria PASS/FAIL, bukan angka telanjang
    ├──→ Memory Hierarchy (§5): benchmark salah layout = jawaban salah
    └──→ SELURUH TOPIK: tidak ada klaim tanpa pengukuran
```

#### Insight
> **Microbenchmark dapat mengonfirmasi hipotesis yang salah dengan meyakinkan.** Ukur di produksi sebelum mempercayai probe.

#### Trade-off
Tracing memberi sinyal perilaku dengan biaya sweep O(vocab) per token — karenanya ia **tidak boleh** berbagi run dengan iterasi terukur.

#### Evidence
```
[R] Research: ARTX04-CPUQuantArch
    Section: Rencana Benchmark (seluruhnya)
    Section: Implikasi terhadap glproc — poin 7

[M] Measured: glbench (glcore/src/telemetry.rs, glbench/src/behavior/)
[M] Measured: glproc/benches/attn_probe.rs — 3 probe, 3 jawaban
```

---

### 15. Mixture-of-Experts (MoE)

#### Ringkasan
Topik ini **tidak dibahas ARTX04 sama sekali**. Seluruh isinya `[M]`.

#### Temuan Penting
- glproc mendukung routing top-k (Qwen3: **top-8 dari 128 expert** — top-2 adalah Mixtral, jangan dikelirukan). `[M]`
- Expert yang tidak menerima token **tidak pernah disentuh** — itulah seluruh argumen performa MoE, sehingga skip bersifat **struktural**, bukan optimasi.
- Softmax **sebelum** seleksi top-k (urutan Qwen3), lalu renormalisasi. Menyeleksi dulu memberi angka berbeda.
- Jalur komputasi **terverifikasi** pada dimensi Qwen3 nyata, di dua arsitektur CPU.
- Layout tensor `_exps` **BELUM TERVERIFIKASI** — lihat §6 dan Knowledge Gap.

#### Penjelasan Teknis
`routing_entropy` (1,0 = uniform, 0,0 = collapsed) adalah metrik yang menentukan: **router yang kolaps tetap menghasilkan output yang benar sambil diam-diam kehilangan seluruh keunggulan kecepatan MoE.** Ia tidak terlihat tanpa metrik ini. `[M]`

Threading: expert berjalan **sekuensial**, masing-masing memakai seluruh pool secara internal — bukan satu expert per worker. `ThreadPool::run` **tidak reentrant**; memanggil kernel `par_*` dari dalam closure `run` akan **deadlock**. `[M]`

#### Hubungan dengan Topik Lain
```
MoE (§15)
    ↓
GGUF (§6) ──→ layout _exps BELUM TERVERIFIKASI (risiko korupsi senyap)
    ↓
Kernel (§10) ──→ tiap expert memakai ulang par_matvec_swiglu
    ↓
Benchmark (§14) ──→ routing_entropy = deteksi kolaps
```

#### Insight
> **Router MoE yang kolaps adalah kegagalan senyap.** Output tetap benar; hanya kecepatannya yang hilang.

#### Trade-off
MoE menukar RAM (semua expert residen) dengan komputasi (hanya top-k berjalan).

#### Evidence
```
[M] Measured: glproc/src/moe.rs (paritas vs referensi naif, dims Qwen3-30B-A3B)
[M] Measured: Colab EPYC + i3-1115G4 — worst_rel_err identik 2 s.f.
[M] Measured: glproc/src/loader.rs — _EXPS_LAYOUT_ASSUMPTION (UNVERIFIED)

[R] TIDAK ADA. ARTX04 tidak membahas MoE.
```

---

## Cross Knowledge Analysis

### Rantai Dependensi Utama

```
                    CPU Microarchitecture (§3)
                     ISA · cache · DRAM ceiling
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
     Memory Hierarchy (§5)              SIMD (§4)
     L1/L2/L3 · prefetch · stride     vpdpbusd · vektorisasi
              │                               │
              └───────────────┬───────────────┘
                              ▼
                       Roofline (§2)
                  29,4 GB/s ceiling terukur
                              │
                              ▼
                  Memory-Bound Decode (§1)
                 ~2 FLOP/byte · 1828 MB/token
                              │
                              ▼
                    Quantization (§7)  ◄── SATU-SATUNYA TUAS
                   byte ↓ = throughput ↑
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
          Q8_0 (§8)                       Q4_K (§9)
     78% ceiling: MENTOK            BELUM ADA KERNEL NATIVE
              │                               │
              └───────────────┬───────────────┘
                              ▼
                        Kernel (§10)
              fused dequant-multiply + fusi layout
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      Dispatch (§11)   Threading (§12)   Runtime (§13)
      OnceLock ISA     ⚠ KONFLIK C4      zero-alloc · pin
              │               │               │
              └───────────────┼───────────────┘
                              ▼
                  Benchmark & Observability (§14)
              roofline sebagai PASS/FAIL · anti-jebakan
                              │
                              ▼
                        THROUGHPUT
```

### Penjelasan Dependensi

| Dari → Ke | Sifat dependensi |
|---|---|
| CPU → SIMD | ISA menentukan kernel mana yang **ada**. VNNI menghadirkan `vpdpbusd`; tanpanya, `maddubs`+`madd`. |
| CPU → Memory Hierarchy | L2 1,25 MB menentukan apakah stride 2 MB antar-head adalah bencana (ya). |
| SIMD + Memory → Roofline | Roofline butuh **keduanya**: ceiling bandwidth *dan* throughput aritmetika untuk menemukan atap yang mengikat. |
| Roofline → Memory-Bound | Roofline **membuktikan** decode memory-bound; ia bukan asumsi. |
| Memory-Bound → Quantization | Bila bottleneck-nya byte, satu-satunya tuas adalah **mengurangi byte**. Ini implikasi logis, bukan pilihan. |
| Quantization → Kernel | Setiap format butuh **kernel integer-dot-nya sendiri**. Tanpa itu, format di-repack dan manfaatnya hilang (§7). |
| Kernel → Threading | Kernel menentukan apakah beban kerja saturating; itu menentukan apakah SMT menolong (⚠ konflik C4). |
| Memory Hierarchy → Benchmark | Benchmark yang tidak mereproduksi layout produksi **mengukur beban kerja fiktif** (§5). |
| Benchmark → SEMUA | Tidak ada klaim yang boleh masuk dokumen ini tanpa melewati sini. |

### Simpul Kritis

**`Quantization (§7)` adalah simpul artikulasi.** Ia satu-satunya titik di mana keputusan desain dapat menggeser atap roofline. Segala hal di atasnya adalah **properti hardware** (tidak dapat diubah); segala hal di bawahnya adalah **mekanisme** (dapat dioptimalkan, tetapi terbatas oleh atap).

Karena Q8_0 sudah 78% ceiling, **simpul ini saat ini tersumbat**, dan sumbatannya adalah ketiadaan kernel Q4_K native.

---

## Architectural Relationships

```
┌──────────────────────────────────────────────────────────────┐
│  HARDWARE (tidak dapat diubah)                               │
│                                                              │
│  CPU ──► ISA (AVX2 / VNNI-256 / [AVX-512 ditolak])           │
│   │                                                          │
│   ├──► Cache (L1 / L2 1,25MB / L3)                           │
│   └──► DRAM (29,4 GB/s ceiling, single-channel DDR4)         │
└──────────────────────────┬───────────────────────────────────┘
                           │ membatasi
┌──────────────────────────▼───────────────────────────────────┐
│  FORMAT (dipilih oleh pengguna, dibaca oleh glproc)          │
│                                                              │
│  GGUF ──► metadata (dims, expert_count, rope, dst.)          │
│    │                                                         │
│    └──► tensor ──► Q8_0 / Q4_K / Q5_0 / Q6_K / F16 / F32     │
│                      │        │                              │
│                      │        └──► [REPACK → Q8_0] ⚠         │
│                      │             manfaat bandwidth HILANG  │
│                      ▼                                       │
│                 integer-dot native (SATU-SATUNYA)            │
└──────────────────────────┬───────────────────────────────────┘
                           │ menentukan traffic
┌──────────────────────────▼───────────────────────────────────┐
│  KERNEL (dapat dioptimalkan, terbatas atap)                  │
│                                                              │
│  qdot ──► vpdpbusd ──► fused dequant-multiply                │
│    │                                                         │
│  layout ──► GateUp interleaved ──► 1 stream DRAM/thread      │
│    │                                                         │
│  attn ──► Q·K (AVX2) + V-accum (AVX2, BARU)                  │
└──────────────────────────┬───────────────────────────────────┘
                           │ dieksekusi oleh
┌──────────────────────────▼───────────────────────────────────┐
│  RUNTIME                                                     │
│                                                              │
│  ThreadPool (single-owner, persisten, chunk kontigu)         │
│    │                                                         │
│  Workspace (zero-alloc) + KvCache (cursor) + pinned pages    │
└──────────────────────────┬───────────────────────────────────┘
                           │ diamati oleh
┌──────────────────────────▼───────────────────────────────────┐
│  OBSERVABILITY (glbench — Mensura Veritatis)                 │
│                                                              │
│  telemetry (pull) ──► stage timeline (ms/call ≠ share%)      │
│  behavior (raw logits) ──► entropy · perplexity · stall      │
│  roofline ──► PASS/FAIL, bukan angka telanjang               │
└──────────────────────────────────────────────────────────────┘
```

### Penjelasan Setiap Hubungan

| Hubungan | Penjelasan |
|---|---|
| **CPU → ISA** | Deteksi runtime (`OnceLock`), bukan compile-time. Kapabilitas ≠ pilihan: AVX-512 lebar penuh **ditolak** (downclock), tetapi VNNI-256 **dipakai**. |
| **CPU → DRAM** | Ceiling 29,4 GB/s adalah **atap absolut**. Tidak ada kernel yang dapat melampauinya. |
| **GGUF → tensor** | GGUF membawa format kuantisasi. glproc membacanya native via mmap, lalu menyalin ke buffer heap yang dimiliki. |
| **tensor → REPACK** | ⚠ **Titik kebocoran.** Q4_K/Q5_0/Q6_K → Q8_0 karena hanya Q8_0 punya kernel. Manfaat ukuran hilang di sini. |
| **format → kernel** | Satu format = satu kernel integer-dot. Tanpa kernel, format tidak dapat dieksekusi native. |
| **kernel → layout** | Fusi terkuat ada di layout (interleaving gate/up), bukan di kernel. |
| **kernel → runtime** | Kernel menentukan pola akses; runtime menyediakan buffer dan thread yang cocok dengan pola itu. |
| **runtime → threading** | Chunk kontigu per thread (bukan interleaved) karena single-channel DDR4 memberi imbalan pada sedikit stream sekuensial. |
| **semua → observability** | Setiap klaim harus melewati glbench. Roofline adalah kriteria kelulusan. |

---

## Canonical Principles

Setiap prinsip memiliki **Evidence**, **Reasoning**, dan **Confidence**.

Skala Confidence:
- **TINGGI** — didukung riset **dan** pengukuran langsung pada hardware target.
- **SEDANG** — didukung salah satu, tidak dibantah oleh yang lain.
- **RENDAH / DISENGKETAKAN** — terdapat konflik atau evidence tidak memadai.

---

### P1. Bandwidth lebih penting daripada FLOPS

**Evidence:** `[R]` ARTX04 §Landasan Teori (~2 FLOP/byte, roofline). `[M]` 1.828 MB/token terukur; FFN 78% ceiling.

**Reasoning:** Decode adalah GEMV dengan intensitas aritmetika sangat rendah. Setiap bobot dibaca sekali per token tanpa amortisasi. Waktu didominasi transfer DRAM.

**Confidence: TINGGI.**

---

### P2. Kernel harus mengikuti layout tensor — dan fusi terkuat ada di layout

**Evidence:** `[R]` ARTX04 §Landasan Teori (blocked layout, cache line). `[M]` Interleaving gate/up = 1 stream DRAM/thread; chunk kontigu vs interleaved = **+35%**.

**Reasoning:** Setelah bobot berada di memori, tidak ada fusi kernel yang dapat memperbaiki layout yang buruk. Layout ditentukan **saat load**, dan itulah leverage terbesar.

**Confidence: TINGGI.**

---

### P3. Kuantisasi mengurangi traffic — tetapi hanya menang bila kernelnya TETAP bandwidth-bound

**Evidence:** `[R]` ARTX04 §Landasan Teori: percepatan linier *"asalkan overhead dequantisasi tidak melebihi penghematan bandwidth"*. `[M]` **Q4_K native diuji dan KALAH 33%** (14,1 → 9,5 tok/s) meski membaca 1,89× lebih sedikit byte. Kernel terisolasi: 1,5–2,0 GMAC/s vs Q8_0 3,3 — **gap identik saat L2-resident**, jadi murni compute, bukan memori.

**Reasoning:** Format yang lebih kecil hanya menang bila kernelnya dapat mempertahankan throughput per-byte. Unpack nibble Q4_K tidak bisa: ia mendorong kernel dari bandwidth-bound menjadi **compute-bound**, dan biaya compute-nya melebihi byte yang dihemat.

> **Bagian bercetak-tebal dari klausul ARTX04 (`asalkan…`) BUKAN formalitas.** Ia adalah syarat yang mengikat, dan glproc melanggarnya. Repack ke Q8_0 membayar 1,89× byte untuk menghindari sebuah kernel — **dan itu adalah trade yang benar** di ISA ini.

**Confidence: TINGGI** (diuji langsung, grup A/B tidak tumpang tindih).

---

### P4. Fused dequant-multiply adalah primitif wajib

**Evidence:** `[R]` ARTX04 §Landasan Teori + §Implikasi poin 3. `[M]` FFN 21,6 GMAC/s — lebih cepat dari `qkv` yang tidak difusikan.

**Reasoning:** Dequant terpisah memaksa vektor intermediate melewati RAM. Pada VNNI fusi terjadi di level instruksi (`vpdpbusd`).

**Confidence: TINGGI.**

---

### P5. Kapabilitas ISA ≠ ISA yang dipakai

**Evidence:** `[M]` i3-1115G4 melaporkan `avx512f/bw/vnni/vl`, tetapi glproc menjalankan `Avx2+vnni256`. AVX-512 lebar penuh ditolak (downclock ~2,5 vs ~3,5 GHz).

**Reasoning:** Frekuensi turun saat AVX-512 512-bit aktif. Bentuk 256-bit (EVEX) berjalan pada frequency license yang sama dengan AVX2 — jadi VNNI **dipakai**, AVX-512 lebar penuh **tidak**.

**Confidence: TINGGI** (terukur langsung). Namun heuristik ">8 logical core" adalah proxy kasar — lihat Open Questions.

---

### P6. Kernel setengah ter-vektorisasi adalah bug yang tersembunyi

**Evidence:** `[M]` Attention V-accum skalar sementara Q·K sudah AVX2 → 0,83 GMAC/s vs `qkv` 18,1 GMAC/s (22×). Diperbaiki → decode **+35%**.

**Reasoning:** Bug ini lolos semua uji korektness dan tidak tampak "lambat" secara mencolok di profil. Ia hanya terungkap dengan membandingkan **GMAC/s antar-stage pada mesin yang sama**.

**Confidence: TINGGI.**

---

### P7. Benchmark harus mereproduksi layout memori produksi

**Evidence:** `[M]` Tiga probe atas perbaikan yang sama: **0,07× / 2,40× / 1,71×**, berbeda hanya karena layout KV.

**Reasoning:** Buffer yang dikemas rapat mengukur beban kerja yang tidak pernah dijalankan engine. Probe pertama akan menyimpulkan "jangan pernah threading attention" — **kebalikan dari kebenaran**.

**Confidence: TINGGI.**

---

### P8. `None` berarti tidak diukur, bukan nol

**Evidence:** `[M]` Desain telemetry glbench; `ToxicitySignal` sengaja tidak dapat dihuni.

**Reasoning:** Baris bernilai nol adalah **klaim**; baris yang absen adalah **pengakuan**. Metrik keamanan yang mengembalikan 0,0 saat belum mengukur apa pun terbaca sebagai "lulus".

**Confidence: TINGGI.**

---

### P9. Single-owner threadpool

**Evidence:** `[R]` ARTX04 §Kajian Literatur (kontensi threadpool → degradasi hingga 30%). `[M]` glproc: pool persisten, std-only, tanpa rayon/crossbeam.

**Reasoning:** Dua threadpool yang bersaing meng-oversubscribe core dan saling merusak cache.

**Confidence: TINGGI.**

---

### P10. Ukuran threadpool: **DISENGKETAKAN**

**Evidence:** `[R]` ARTX04: physical core. `[M]` glproc: physical core **kalah 23%**; knee terukur di **3** (bukan 2, bukan 4).

**Reasoning:** Lihat [Research Conflict C4](#c4--physical-core-vs-logical-core-threading).

**Confidence: RENDAH / DISENGKETAKAN.** ⚠ **Jangan jadikan prinsip sampai terselesaikan.**

---

### P11. Ukur di produksi sebelum mempercayai probe

**Evidence:** `[M]` Produksi **dua kali** mengalahkan prediksi probe (attention: prediksi 1,21×, aktual 1,9×). Tiga jebakan terdokumentasi (layout KV, ctx ditebak, thermal).

**Reasoning:** Probe hanya men-stream KV; produksi men-stream KV **dan** 2 GB bobot per token, sehingga lebih memory-stalled daripada yang dapat direproduksi probe manapun.

**Confidence: TINGGI.**

---

### P12. Klaim percepatan memerlukan grup A/B yang tidak tumpang tindih

**Evidence:** `[M]` Hasil A/B pernah **terbalik** tiga kali berturut-turut murni karena thermal throttling pada part 15 W.

**Reasoning:** Rata-rata yang lebih baik tidak cukup pada mesin yang berisik. Standar yang lolos: `min` sesudah melampaui `max` sebelum.

**Confidence: TINGGI.**

---

## Design Knowledge

Bab ini **hanya mendokumentasikan alternatif** yang ditemukan evidence. Tidak ada keputusan baru yang dibuat di sini.

---

### D1. Threadpool Sizing

| Alternatif | Kelebihan | Kekurangan | Evidence |
|---|---|---|---|
| **Physical core** | Menghindari kontensi cache/bandwidth SMT; literatur konsisten; latensi tail lebih baik | **Terukur kalah 23%** pada hardware target glproc | `[R]` ARTX04 §Landasan Teori · `[M]` A/B 2026-07-14 |
| **Logical core (SMT)** | **Terukur menang** pada glproc; SMT mengisi issue slot yang menganggur selama stall | Bertentangan dengan literatur; berisiko pada beban kerja yang benar-benar saturating | `[M]` glproc `N_THREADS = 4` |
| **Knee terukur (3)** | Optimum empiris pada beban kerja MoE | Bukan physical maupun logical; tidak ada teori yang menjelaskannya; belum diuji lintas beban kerja | `[M]` `moe_threads` bench |

⚠ Lihat [Research Conflict C4](#c4--physical-core-vs-logical-core-threading).

---

### D2. Strategi Format Kuantisasi

| Alternatif | Kelebihan | Kekurangan | Evidence |
|---|---|---|---|
| **Repack semua → Q8_0** (saat ini) | Satu kernel integer-dot; sederhana; akurasi tinggi | **Traffic 2× vs Q4_K**; membuang manfaat kuantisasi pengguna | `[M]` `loader.rs` |
| **Q4_K integer-dot native** | ~2× traffic lebih sedikit; menyentuh **3 bucket terpanas sekaligus** | Kernel lebih rumit (unpack nibble, super-block scale); jalur paritas baru | `[R]` ARTX04 Sprint 2 · `[M]` roofline |
| **K-quant sebagai default** | Akurasi lebih baik pada bit-width rendah | Overhead metadata | `[R]` ARTX04 §Analisis Kritis |
| **I-quant (importance-aware)** | Akurasi tertinggi | Butuh data kalibrasi; packing lebih kompleks | `[R]` ARTX04 §Analisis Kritis |
| **Generic (Q4_0)** | Sederhana | Suboptimal — tidak memperhitungkan distribusi bobot non-uniform | `[R]` ARTX04 §Analisis Kritis |

---

### D3. Fallback Kernel Generik

| Alternatif | Kelebihan | Kekurangan | Evidence |
|---|---|---|---|
| **Tanpa fallback (error eksplisit)** | Tidak ada "performa yang menipu" | Tidak ada oracle paritas | `[R]` ARTX04 §Implikasi poin 2 |
| **Fallback skalar sebagai oracle** | Ground truth untuk uji paritas setiap kernel SIMD | Kode yang tidak pernah berjalan di produksi | `[M]` `kernels/*/scalar.rs` |

⚠ Lihat [Research Conflict C3](#c3--fallback-generik).

---

### D4. Telemetry Engine ↔ Harness

| Alternatif | Kelebihan | Kekurangan | Evidence |
|---|---|---|---|
| **Pull (data)** — engine mengisi struct, harness membaca | Tidak ada dependensi balik; engine tidak tahu harness ada; backend lain cukup mengembalikan bentuk yang sama | Snapshot, bukan streaming | `[M]` `glcore/src/telemetry.rs` |
| **Push (callback trait)** | Streaming; granular | **Membalik dependensi** (engine meng-import tipe harness); memberi harness hook untuk menyuntik perilaku ke hot path | `[M]` **DITOLAK** dalam desain glbench |

---

### D5. Mode Eksekusi

| Alternatif | Kelebihan | Kekurangan | Evidence |
|---|---|---|---|
| **Latency-first** (single-stream) | Optimal untuk chat interaktif | Throughput rendah | `[R]` ARTX04 §Analisis Kritis |
| **Throughput-first** (multi-stream) | Optimal untuk serving | Latensi lebih tinggi | `[R]` ARTX04 §Analisis Kritis |
| **Auto-detection** | Tanpa konfigurasi | **Tidak dapat diprediksi** — ARTX04 secara eksplisit menolaknya | `[R]` ARTX04 §Analisis Kritis |

---

## Research Conflict

> **Bab ini tidak memilih pemenang.** Kedua sisi ditampilkan dengan evidence-nya. Menyelesaikan konflik memerlukan pengukuran baru, bukan argumen.

---

### C1 — Klasifikasi ISA Tiger Lake

**Klaim A `[R]` (ARTX04):**
> "baseline Intel Tiger Lake (i3-1115G4, **AVX2+FMA3**)"

ARTX04 secara konsisten mengategorikan baseline sebagai part AVX2, dan menempatkan VNNI/AMX secara eksklusif pada "validation tier Xeon".

**Klaim B `[M]` (Pengukuran glproc, 2026-07-14):**
> i3-1115G4 melaporkan `avx512f`, `avx512bw`, `avx512vnni`, **dan** `avx512vl`.
> `has_vnni_256()` = **true**. Runtime: `[simd] strategy: Avx2+vnni256`.
> **Seluruh jalur integer-dot glproc berjalan pada `vpdpbusd` (VNNI), bukan AVX2 murni, PADA BASELINE.**

**Perbedaan:**
ARTX04 memperlakukan VNNI sebagai fitur tier-Xeon. Pengukuran menunjukkan VNNI **sudah tersedia dan sudah dipakai pada baseline**. Ini bukan sekadar koreksi kosmetik: roadmap ARTX04 menempatkan "jalur AVX-512 VNNI" di **Sprint 3 (Xeon tier)**, padahal glproc **sudah menjalankannya di baseline**.

**Nuansa yang mendamaikan sebagian:**
AVX-512 **lebar penuh (512-bit)** memang ditolak glproc — ia menurunkan frekuensi (~2,5 vs ~3,5 GHz). Tetapi bentuk **256-bit EVEX** dari VNNI berjalan pada *frequency license* yang sama dengan AVX2. Jadi ARTX04 mungkin benar tentang **AVX-512 lebar penuh** dan keliru tentang **VNNI-256**. Keduanya harus dibedakan.

**Status: SEBAGIAN TERSELESAIKAN.** Pengukuran lebih kuat (langsung, pada hardware target). Namun distinksi AVX-512-512bit vs VNNI-256bit tidak ada di ARTX04, sehingga klaimnya bukan salah total — ia **kurang granular**.

**Untuk menyelesaikan:** `Evidence Required` — apakah heuristik ">8 logical core" adalah proxy yang benar untuk downclock, atau seharusnya deteksi TDP/frekuensi langsung?

---

### C2 — Urutan Prioritas Sprint

**Klaim A `[R]` (ARTX04 §Rekomendasi Implementasi):**
> Sprint 2 (Hari 31–60): kernel Q4_K_M + Q8_0 pada AVX2. Target **≥90% performa llama.cpp**.
> Sprint 3 (Hari 61–90): AVX-512 VNNI, AMX, Q3_K/Q2_K.
> Sprint 4: fused GEMM+activation, mixed-precision.

**Klaim B `[M]` (Keadaan glproc aktual):**
> - Q8_0 integer-dot: **SELESAI**, pada VNNI (yang ARTX04 taruh di Sprint 3).
> - Fused GEMM+activation (SwiGLU): **SELESAI**, dan lebih agresif dari usulan (fusi *layout*, bukan hanya kernel). ARTX04 menaruhnya di Sprint 4.
> - **Q4_K integer-dot native: BELUM ADA.** Ini adalah item Sprint 2 ARTX04 — item dengan **prioritas tertinggi** — dan justru satu-satunya yang belum dikerjakan.

**Perbedaan:**
glproc mengeksekusi Sprint 3 dan Sprint 4 **sebelum** menyelesaikan item inti Sprint 2. Urutan aktual terbalik dari roadmap.

**Implikasi:**
Ini **bukan** kesalahan — pekerjaan yang dilakukan (VNNI, fusi SwiGLU, perbaikan attention) memberi hasil terukur. Tetapi ia berarti **satu-satunya tuas yang tersisa** (§7, §9) adalah item yang paling awal dalam roadmap dan paling belum tersentuh.

**Status: TERCATAT, bukan konflik faktual.** Ini adalah divergensi *rencana vs eksekusi*, dan ia menunjuk pada pekerjaan berikutnya.

---

### C3 — Fallback Generik

**Klaim A `[R]` (ARTX04 §Implikasi poin 2):**
> "**Tidak ada fallback generik** untuk kernel inti GEMV/GEMM; jika ISA tidak didukung, **error eksplisit lebih baik daripada performa yang menipu**."

**Klaim B `[M]` (glproc):**
> Setiap kernel SIMD memiliki pasangan **skalar**, dan uji paritas membandingkan keduanya. `SimdStrategy::Scalar` adalah varian yang valid.

**Perbedaan:**
Secara harfiah bertentangan. Secara **semangat**, mungkin tidak: ARTX04 melarang fallback sebagai **jalur produksi yang menipu**; glproc memakainya sebagai **oracle pengujian**.

**Risiko bila tidak dicatat:** seseorang yang membaca ARTX04 sebagai kanon dapat **menghapus kernel skalar** atas nama kepatuhan — dan dengan demikian menghapus satu-satunya ground truth untuk memvalidasi kernel SIMD.

**Status: BELUM TERSELESAIKAN.** Perlu keputusan eksplisit: apakah kernel skalar boleh dieksekusi di produksi (bila ISA tidak dikenali), atau hanya di uji?

---

### C4 — Physical-Core vs Logical-Core Threading

> ⚠ **INI ADALAH KONFLIK PALING MATERIAL DALAM DOKUMEN INI.**

**Klaim A `[R]` (ARTX04, didukung literatur XNNPACK + studi kinerja CPU):**
> "Hyperthreading/SMT **sering kali merugikan** untuk beban kerja memory-bound karena kontensi cache dan bandwidth. Literatur XNNPACK dan studi kinerja CPU **konsisten** menunjukkan bahwa thread count optimal untuk LLM decode **setara dengan jumlah physical core**, bukan logical core. **Oversubscription meningkatkan latensi tail (p95/p99) tanpa meningkatkan throughput median.**"

Direkomendasikan sebagai keputusan arsitektural (§Implikasi poin 4).

**Klaim B `[M]` (Pengukuran langsung pada hardware target, 2026-07-14):**

Hipotesis ARTX04 **diuji secara eksplisit** pada i3-1115G4 (2 physical / 4 logical), Qwen3-1.7B Q8_0, decode, tiga run bergantian:

| rep | 4 thread (logical) | 2 thread (physical) |
|---|---|---|
| 1 | 8,4 tok/s | 7,9 |
| 2 | 10,6 | 8,3 |
| 3 | **11,0** | **8,5** |

> **Physical-core sizing KALAH 23% pada steady state.** Menang di **nol** dari tiga run.

Lebih jauh, sweep threadpool pada beban kerja MoE menemukan knee di **3 thread** (1,93 ms) — **bukan** physical (2), **bukan** logical (4); 4 thread justru **regresi** ke 2,43 ms.

**Penjelasan yang diajukan pengukuran:**
> "Decode memory-bound, jadi SMT tidak menambah bandwidth" hanya benar bila core **benar-benar meng-*issue*** permintaan memori secara berurutan. Ia tidak. Jalur decode melakukan konversi skala f16→f32 per blok, rantai dot integer, dan ekor skalar **di antara** load. Thread sibling mengisi tepat celah-celah itu dan menjaga lebih banyak load in-flight.
>
> **Berada di 69% ceiling bandwidth adalah evidence MELAWAN physical-core sizing, bukan mendukungnya** — loop yang benar-benar tersaturasi akan berada di ~95% tanpa celah tersisa untuk di-overlap.

**Perbedaan mendasar:**
Kedua sisi setuju decode adalah memory-bound. Mereka **tidak setuju** apakah "memory-bound" berarti "issue-saturated". ARTX04 mengasumsikan ya (sehingga SMT hanya membagi cache). Pengukuran menunjukkan tidak (sehingga SMT mengisi celah issue).

**Faktor yang memperumit — thermal noise:**
Hardware target adalah part **15 W** yang throttle keras. Hasil A/B pernah **terbalik** semata-mata karena panas. Pengukuran C4 di atas **konsisten lintas tiga run bergantian** dan selisihnya melebar seiring mesin memanas — tetapi pembaca harus tahu bahwa mesin ini berisik.

**Yang TIDAK dalam sengketa:**
- Single-owner threadpool: **disepakati** kedua sisi.
- Latensi tail: ARTX04 mengklaim oversubscription memperburuk p95/p99. **glproc BELUM mengukur p95/p99 sebagai fungsi thread count.** Klaim ARTX04 ini **tidak terbantah maupun terkonfirmasi**.

**Status: BELUM TERSELESAIKAN — DAN KEDUANYA MUNGKIN BENAR.**
ARTX04 mungkin benar untuk beban kerja/hardware yang benar-benar issue-saturated (Xeon, prefill, batch besar). Pengukuran benar untuk decode batch-1 pada part mobile 2-core. **Keduanya boleh berlaku dalam domainnya masing-masing.**

**`Evidence Required` untuk menyelesaikan:**
1. p95/p99 sebagai fungsi thread count (klaim tail-latency ARTX04 belum diuji).
2. Pengulangan A/B pada Xeon validation tier (di mana core lebih banyak dan SMT mungkin berperilaku berbeda).
3. Pengukuran pada prefill (compute-bound) — ARTX04 §Kajian Literatur bahkan menyarankan logical-core **mungkin bermanfaat** untuk prefill, yang **konsisten** dengan temuan glproc.

---

## Knowledge Gap

Topik berikut ada di Knowledge Index yang diminta tetapi **tidak memiliki evidence yang memadai** dalam dokumen input. Sesuai mandat, tidak ada spekulasi yang diisikan.

---

### Format Kuantisasi

| Topik | Status | Apa yang ada | Apa yang hilang |
|---|---|---|---|
| **FP32** | `Evidence Required` | Disebut sebagai baseline perbandingan (INT8 speedup 2–3× vs FP32) `[R]` | Layout, kernel, kapan dipakai di glproc |
| **BF16** | `Evidence Required` | Disebut sebagai internal tensor type (Sprint 1) dan target AMX `[R]` | Layout blok, kernel, trade-off akurasi vs FP16 |
| **FP16** | `Evidence Required` | Disebut sebagai internal tensor type (Sprint 1) `[R]` | Sama seperti BF16 |
| **Q8_1** | `Evidence Required` | **TIDAK DISEBUT SAMA SEKALI** di dokumen input | Segalanya |
| **Q6_K** | `Evidence Required` | Disebut sebagai target Sprint 3 dan format benchmark `[R]`. glproc me-repack-nya ke Q8_0 `[M]` | Struktur blok, kernel native, trade-off akurasi |
| **Q5_K** | `Evidence Required` | Disebut sekali sebagai contoh K-quant `[R]` | Segalanya selain nama |
| **Q3_K** | `Evidence Required` | Disebut sebagai target Sprint 3 dan "extreme compression stress test" `[R]` | Struktur blok, kernel, degradasi akurasi terukur |
| **Q2_K** | `Evidence Required` | Sama seperti Q3_K `[R]` | Sama |
| **Q4_0 (generic)** | `Evidence Required` | Disebut sebagai "suboptimal, legacy compatibility" `[R]` | Struktur, kapan masih relevan |

> **Catatan penting:** ARTX04 **menyebut** banyak format ini dalam daftar sprint dan benchmark, tetapi **tidak menjelaskan struktur blok, layout, atau kernel-nya**. Menyebut nama bukan evidence. Mengisi bab-bab ini memerlukan riset tambahan atau pembacaan langsung `ggml-quants.h`.

---

### Hardware & ISA

| Topik | Status | Catatan |
|---|---|---|
| **AMX** | `Evidence Required` | ARTX04 mengklaim ">90 TFLOPS INT8/BF16" `[R]`, tetapi **belum ada hardware AMX yang diuji** oleh glproc. Klaim tidak terverifikasi. |
| **NEON / SVE (ARM)** | `Evidence Required` | Disebut sebagai jalur dispatch `[R]`. glproc **belum menargetkan ARM**. |
| **Xeon validation tier** | `Evidence Required` | Seluruh Sprint 3 ARTX04 bergantung padanya. **Belum ada pengukuran glproc pada Xeon.** |
| **NUMA** | `Evidence Required` | ARTX04 memproyeksikan "15–25% throughput improvement" pada dual-socket `[R]` — **belum diverifikasi**. |

---

### Metodologi

| Topik | Status | Catatan |
|---|---|---|
| **p95 / p99 latency vs thread count** | `Evidence Required` | **Kritis untuk menyelesaikan C4.** ARTX04 mengklaim oversubscription memperburuk tail latency; glproc belum mengukurnya. |
| **Cache miss rate (L1/L2/L3)** | `Evidence Required` | Diwajibkan ARTX04 §Rencana Benchmark. glbench **belum** melaporkannya. |
| **Energy per inference** | `Evidence Required` | Diwajibkan ARTX04 (RAPL/IPMI). Belum diimplementasikan. |
| **Perbandingan vs llama.cpp / ORT / OpenVINO** | `Evidence Required` | ARTX04 mewajibkannya. glproc memiliki baseline llama.cpp historis, tetapi **tidak pada konfigurasi saat ini**. |
| **Akurasi (perplexity) pasca-kuantisasi** | `Evidence Required` | ARTX04 Sprint 4. glproc mengukur `perplexity` sebagai sinyal perilaku, tetapi **belum** sebagai gerbang validasi akurasi kuantisasi. |

---

### MoE

| Topik | Status | Catatan |
|---|---|---|
| **Layout tensor `_exps`** | `Evidence Required` | ⚠ **RISIKO KORUPSI SENYAP.** Asumsi ditulis dari konvensi llama.cpp, bukan dari byte yang dibaca. Bila urutan penumpukan salah, model **tetap berjalan** dan mengeluarkan sampah yang fasih. Grep `_EXPS_LAYOUT_ASSUMPTION`. |
| **`expert_weights_norm`** | `Evidence Required` | Default `true` (perilaku Qwen3) karena `GgufValue` tidak punya accessor bool. Salah → menskalakan ulang seluruh residual FFN. |
| **MoE dalam ARTX04** | — | ARTX04 **tidak membahas MoE sama sekali**. |

---

## Open Questions

Dikelompokkan berdasarkan topik.

### Threading
1. Apakah oversubscription benar-benar memperburuk **p95/p99** seperti klaim ARTX04? (glproc belum mengukur tail latency sebagai fungsi thread count.) → **kunci untuk C4**
2. Mengapa knee terukur berada di **3** thread, bukan 2 atau 4? Tidak ada teori dalam evidence yang menjelaskannya.
3. Apakah temuan physical-vs-logical berbalik pada **Xeon** (lebih banyak core, SMT berbeda)?
4. Apakah **prefill** (compute-bound) berperilaku berbeda dari decode? ARTX04 sendiri menyarankan logical-core mungkin membantu prefill — konsisten dengan temuan glproc.

### Kuantisasi
5. Berapa **percepatan aktual** dari kernel Q4_K integer-dot native? (Aritmetika roofline memberi arah ~2× traffic; **angka belum diukur**.)
6. Berapa **degradasi akurasi** Q4_K native vs repack-ke-Q8_0 saat ini? (Repack me-requantize; error itu belum dikuantifikasi.)
7. Apakah unpack nibble dengan **LUT SIMD** benar-benar mengalahkan bitwise shifting pada AVX2, seperti klaim ARTX04?
8. Apakah **mixed-precision** (embedding Q8, layer Q4) memberi trade-off yang bermanfaat?

### ISA
9. Apakah heuristik **">8 logical core"** adalah proxy yang benar untuk downclock AVX-512, atau seharusnya deteksi TDP/frekuensi langsung? (P5)
10. Apakah **AMX** benar-benar mencapai ">90 TFLOPS" yang diklaim, dan apakah ia relevan untuk decode batch-1 (yang GEMV, bukan GEMM)?
11. Apakah `avxvnni` (VNNI 256-bit **tanpa** AVX-512, Alder Lake+) harus punya jalur sendiri? (Terdeteksi `false` pada baseline.)

### Attention
12. Threading loop 16-head (**fix B**): angka probe lamanya (1,71×) **basi** — diukur terhadap baseline skalar yang sudah diganti. Perlu re-probe.
13. Mengapa probe cold-cache (4,92 ms/token) masih **2,8× lebih cepat** dari produksi (13,89 ms/token) sebelum perbaikan? Dugaan: KV bersaing dengan 2 GB bobot. **Belum dibuktikan.**
14. Apakah alokasi `KvCache` untuk `max_context` (bukan konteks aktif) — yang menciptakan stride 2 MB antar-head dan cache dingin permanen — sepadan dengan kesederhanaannya?

### Format & Loader
15. Apakah asumsi layout `_exps` benar? (**Butuh satu file Qwen3-MoE GGUF; header dump saja sudah cukup.**)
16. Apakah `expert_weights_norm` benar-benar default `true`?

### Benchmark
17. Bagaimana mengukur **cache miss rate** dan **bandwidth utilization** secara portabel (ARTX04 mewajibkannya; glbench belum punya)?
18. Bagaimana membandingkan **apples-to-apples** dengan llama.cpp pada konfigurasi glproc saat ini?

---

## Glossary

| Istilah | Definisi (sesuai evidence) |
|---|---|
| **AMX** | Advanced Matrix Extensions — akselerator matriks Intel. Diklaim >90 TFLOPS INT8/BF16 `[R]`, belum diverifikasi glproc. |
| **AVX2 / FMA3** | SIMD 256-bit Intel/AMD. Baseline glproc. |
| **AVX-512** | SIMD 512-bit. **Ditolak glproc pada part core-rendah** karena downclock (~2,5 vs ~3,5 GHz). |
| **Bandwidth ceiling** | Batas atas laju baca DRAM. Terukur **29,4 GB/s** pada i3-1115G4. |
| **Bandwidth-bound** | Beban kerja yang dibatasi transfer memori, bukan aritmetika. Sifat decode LLM. |
| **Decode** | Fase generasi token-per-token. Bandwidth-bound. Tidak ada amortisasi bobot. |
| **Dequant-multiply fusion** | Menggabungkan dekuantisasi dan perkalian dalam satu pass, menghilangkan vektor intermediate. |
| **GEMM** | General Matrix-Matrix Multiplication. Pola prefill (batched). |
| **GEMV** | General Matrix-Vector Multiplication. Pola decode. Intensitas aritmetika ~2 FLOP/byte. |
| **GGUF** | Format penyimpanan model untuk inferensi: mmap-able, alignment kontigu, metadata self-contained. |
| **GMAC/s** | Giga multiply-accumulate per detik. **Metrik yang dapat dibandingkan antar-model**, tidak seperti `share%`. |
| **GQA** | Grouped-Query Attention. Query head > KV head (Qwen3-1.7B: 16/8). |
| **Issue-saturated** | Core yang mengeluarkan permintaan memori tanpa celah. **Berbeda dari memory-bound** — perbedaan ini adalah inti Konflik C4. |
| **K-quant** | Kuantisasi dengan statistik super-block (Q4_K, Q5_K, Q6_K). Akurasi lebih baik pada bit-width rendah. |
| **KV cache** | Cache key/value per layer/head. Satu-satunya bagian footprint yang **tumbuh dengan panjang konteks**. |
| **`ms/call`** | Biaya satu invokasi. Menjawab: *lambat, atau sekadar sering?* Berbeda dari `share%`. |
| **MoE** | Mixture-of-Experts. Token dirutekan ke top-k dari N expert. Qwen3: **top-8 dari 128** (top-2 adalah Mixtral). |
| **Prefill** | Fase pemrosesan prompt. Batched → compute-bound. |
| **Repack** | Konversi format kuantisasi saat load. glproc: Q4_K/Q5_0/Q6_K → Q8_0. **Membuang manfaat bandwidth.** |
| **Roofline** | Model yang memprediksi performa dari bandwidth dan intensitas aritmetika. Di glproc: **detektor bug**, bukan sekadar evaluator. |
| **Routing entropy** | 1,0 = uniform, 0,0 = kolaps. Router MoE yang kolaps **tetap benar tetapi lambat** — kegagalan senyap. |
| **`share%`** | Porsi wall-time sebuah stage. **TIDAK dapat dibandingkan antar-model** (penyebutnya berubah). |
| **SMT / Hyperthreading** | Dua thread logis per core fisik. Manfaatnya untuk decode: **DISENGKETAKAN** (C4). |
| **SwiGLU** | Aktivasi FFN: `silu(gate·x) * (up·x)`. Di glproc: gate/up **di-interleave saat load**, SiLU inline di register. |
| **VNNI** | Vector Neural Network Instructions. `vpdpbusd` = fused int8 multiply-accumulate. **Bentuk 256-bit berjalan pada frequency license AVX2** — tidak kena penalti downclock. |
| **Weight-only quantization** | Hanya bobot dikuantisasi; aktivasi tetap presisi lebih tinggi. Strategi utama LLM CPU. |

---

## References

Referensi dikelompokkan per bab. **Hanya dokumen input yang dikutip** — tidak ada referensi eksternal baru.

### Dokumen Input Primer

**[R] ARTX04-CPUQuantArch** — *Analisis Arsitektural Kuantisasi CPU Modern untuk Inferensi LLM* (proposal teknis, disediakan sebagai input).
Bab yang dikutip: Abstrak · Pendahuluan · State of the Art · Landasan Teori · Kajian Literatur · Analisis Kritis · Implikasi terhadap glproc · Rekomendasi Implementasi · Rencana Benchmark · Future Work · Daftar Referensi.

> ⚠ ARTX04 mencantumkan 19 referensi eksternal (GGML source, GGUF spec, manual Intel/AMD/ARM, GPTQ, AWQ, SmoothQuant, oneDNN, XNNPACK, MLAS, FBGEMM, llama.cpp, Roofline, AMX guide, dll.). **Referensi-referensi itu TIDAK disediakan sebagai dokumen input dan TIDAK dibaca langsung.** Klaim apa pun yang berasal darinya diteruskan **sebagaimana dinyatakan ARTX04**, bukan diverifikasi secara independen. Ini membatasi Confidence pada klaim `[R]`-only.

### Sumber Pengukuran `[M]`

| Sumber | Digunakan di bab |
|---|---|
| `glproc` production profile (Qwen3-1.7B Q8_0, Qwen2.5-0.5B Q4_K_M), `GLPROC_PROFILE=1 glbench run` | §1, §2, §4, §8, §9, §14 |
| `glproc/src/kernels/qdot/q8_0/{avx2,vnni}.rs` | §4, §8, §10 |
| `glproc/src/kernels/ops/attn_accum/` (commit d1942b7) | §4 |
| `glproc/src/threading.rs` (pool, chunk kontigu, `par_matvec_swiglu`) | §5, §10, §12, §13 |
| `glproc/src/simd_strategy.rs` (deteksi ISA, heuristik AVX-512) | §3, §11 |
| `glproc/src/loader.rs` (GGUF, repack, `warm_and_lock_model`, `_EXPS_LAYOUT_ASSUMPTION`) | §6, §7, §13, §15 |
| `glproc/src/moe.rs` | §15 |
| `glproc/benches/attn_probe.rs` (3 probe, 3 jawaban) | §5, §14 |
| `glproc/benches/moe_threads.rs` (sweep threadpool, knee=3) | §12, C4 |
| `glbench` / `glcore/src/telemetry.rs` / `glbench/src/behavior/` | §14 |
| Bandwidth baseline i3-1115G4 (29,4 GB/s), 2026-07-07 | §2, §3 |
| A/B physical-vs-logical thread, 2026-07-14 | §12, C4 |

---

## Audit Validasi

Dilakukan sebelum dokumen ini difinalisasi.

| Kriteria | Status | Catatan |
|---|---|---|
| ✓ Seluruh fakta berasal dari evidence | **LULUS** | Setiap klaim ditandai `[R]` atau `[M]` dengan sumber. |
| ✓ Tidak ada fakta baru | **LULUS** | Tidak ada pengetahuan eksternal yang ditambahkan. Aritmetika turunan (mis. MB/token) dihitung dari dimensi GGUF yang terukur dan ditandai sebagai turunan, bukan fakta baru. |
| ✓ Seluruh insight memiliki evidence | **LULUS** | Setiap §Insight menunjuk ke §Evidence bab yang sama. |
| ✓ Seluruh kontradiksi dicatat | **LULUS** | 4 konflik: C1 (ISA), C2 (sprint), C3 (fallback), C4 (threading — **paling material**). |
| ✓ Seluruh topik saling terhubung | **LULUS** | Setiap bab punya §Hubungan; lihat Cross Knowledge Analysis + Architectural Relationships. |
| ✓ Seluruh istilah konsisten | **LULUS** | Glossary. Istilah yang paling sering dikelirukan (`share%` vs `ms/call` vs `GMAC/s`; *memory-bound* vs *issue-saturated*) didefinisikan eksplisit. |
| ⚠ Cakupan Knowledge Index | **PARSIAL** | 15 dari ~22 topik memiliki evidence. Sisanya (`FP32`, `BF16`, `FP16`, `Q8_1`, `Q6_K`, `Q5_K`, `Q3_K`, `Q2_K`, AMX, NEON/SVE, NUMA) ditandai `Evidence Required` **alih-alih diisi dengan spekulasi**. Ini adalah kepatuhan pada mandat, bukan kelalaian. |

### Catatan Audit yang Harus Dibaca

1. **Basis evidence riset adalah satu dokumen.** ARTX04 adalah proposal arsitektural, bukan laporan pengukuran. 19 referensi eksternalnya tidak disediakan dan tidak dibaca. Klaim `[R]`-only diteruskan sebagaimana dinyatakan, tidak diverifikasi.

2. **Klaim `[M]` lebih kuat daripada klaim `[R]`.** Pengukuran dilakukan langsung pada hardware target dan dapat direproduksi. Bila keduanya bertentangan (C1, C4), pembaca harus **mengetahui asimetri ini** — tetapi dokumen ini tidak memilih pemenang, sesuai mandat.

3. **Konflik C4 (threading) belum terselesaikan dan mungkin keduanya benar** dalam domain masing-masing. Jangan jadikan prinsip.

4. **§9 (Q4_K) diuji dan TERBANTAH pada 2026-07-14.** Kernel dibangun, paritas lolos, produksi **kalah 33%**. Unpack nibble mendorong kernel menjadi compute-bound (1,5–2,0 GMAC/s vs Q8_0 3,3; gap identik di L2). Repack ke Q8_0 tetap menang. **Ini membatalkan kesimpulan "satu-satunya tuas tersisa" dari versi v1 dokumen ini** — dan menunjukkan bahwa roofline memberi batas atas, bukan janji.

---

*Mensura Veritatis bukan hasil penelitian. Ia adalah hasil sintesis seluruh penelitian.*
*Seluruh keputusan engineering glproc di masa depan harus dapat ditelusuri kembali ke dokumen ini.*
