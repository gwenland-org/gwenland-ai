# Observations

Facts only. No speculation. No causality claims without evidence.

---

## Hardware Baseline

- Machine: Intel i3-1115G4, 2 physical / 4 logical cores, 2995 MHz (no throttle observed in any run).
- RAM: 8 GB total, 2.0–3.9 GB available at measurement time (varies by session state).
- Measured memory read bandwidth: 23.3–31.8 GB/s (varies session to session; glbench probes it at run start).
  - This 19% session-to-session swing is a documented property of this machine.
- Model file: 491,400,032 bytes (0.46 GiB). Q4_K_M GGUF.
- Loaded weight size: 669,476,864 bytes (0.63 GiB, after Q4_K→Q8_0 GATE repack).
- KV cache at session init: 100,663,296 bytes (96 MB).

Source: `benchmarks/full-bottleneck-e2e.json`, `benchmarks/post-gate-benchmark-report.md`.

---

## Production Throughput (Post-GATE, most reliable readings)

| Phase | glproc | llama.cpp (llama-bench r=5) | Gap |
|---|---|---|---|
| Decode | 37.1 tok/s | 47.36 ± 2.31 tok/s | glproc −24.8% |
| Prefill | 121.3 tok/s | 194.93 ± 6.18 tok/s | glproc −41.6% |

Source: `benchmarks/post-gate-benchmark-report.md`.

Pre-GATE baseline (same session, Veritas Secunda): decode 36.7–39.1 tok/s, prefill 128.5–135.5 tok/s.
Post-GATE decode (37.1) is within the pre-GATE range. Prefill (121.3) is slightly below the range floor — a single-run reading, not confirmed as regression.

---

## Roofline Analysis (full-bottleneck-e2e run)

Measured bandwidth ceiling: 31.43 GB/s.
Decode throughput: 36.6 tok/s.
Ceiling efficiency: 57.3% (decode running at 57% of the bandwidth ceiling).
glbench classification: compute_bound.

Stage breakdown for decode:

| Stage | Share | GB/s | Ceiling% | Verdict |
|---|---|---|---|---|
| ffn_gate_up | 35.5% | 22.4 | 68.2% | indeterminate |
| lm_head | 22.4% | 23.1 | 73.5% | bandwidth-bound |
| ffn_down | 20.0% | 19.8 | 68.2% | indeterminate |
| qkv | 9.3% | 10.1 | 28.5% | indeterminate |
| attn_out | 6.2% | 11.7 | 37.2% | indeterminate |
| attention | 5.9% | 4.2 | 6.0% (wait) | indeterminate |
| sampler | 0.6% | — | — | unknown |

Stage breakdown for prefill:

| Stage | Share | Arithmetic intensity (FLOP/byte) | Verdict |
|---|---|---|---|
| ffn_gate_up | 50.3% | 60.2 | not-bandwidth-bound |
| ffn_down | 20.6% | 67.5 | not-bandwidth-bound |
| qkv | 5.9% | 55.9 | not-bandwidth-bound |
| attn_out | 5.8% | 44.4 | not-bandwidth-bound |
| attention | 3.2% | 67.0 | not-bandwidth-bound |
| fixup | 5.5% | — | unknown |
| serial | 2.6% | — | unknown |

Source: `benchmarks/full-bottleneck-e2e.json`.

---

## VNNI-512 Investigation

VNNI-512 replaces the 256-bit AVX2 vecdot inner loop with a 512-bit AVX-512 VNNI inner loop.

### Isolated (glbench stage-level telemetry)

VMAC/s improvement was not directly measured in glbench's stage telemetry.
glbench baseline ceiling efficiency: 56.4% (baseline) vs 75.7% (candidate) — a 34% jump in ceiling efficiency.
The candidate ceiling was measured at 23.3 GB/s vs baseline 31.1 GB/s — the bandwidth probe ran on a different session state and returned a lower ceiling, making the efficiency jump an artifact of a lower denominator.

### Production (5-iteration A/B)

Run 1:
- Baseline decode: 35.8 ± 0.82 tok/s
- Candidate decode: 35.9 ± 0.38 tok/s
- Delta: +0.1 tok/s (+0.3%, within noise)

Run 2 (repeat):
- Baseline decode: 35.8 ± 0.73 tok/s
- Candidate decode: 36.2 ± 2.50 tok/s
- Delta: +0.4 tok/s (+1.1%, within noise given candidate std_dev 2.50)

Prefill: mixed-sign across runs (candidate prefill 118.6 vs baseline 120.4 in run 1; 102.8 vs 124.9 in run 2 — candidate run 2 prefill was noisy, a 3.7 GB/s prefill spike visible in cold iterations).

Production result: neutral. No sustained decode or prefill improvement across both repeats.

SIMD path: both baseline and candidate executed on `avx2` path (confirmed by `backend.simd_path = "avx2"` in all JSON telemetry). VNNI-512 kernel was compiled and selectable but the default production path remained AVX2.

Source: `benchmarks/vnni512-ab-baseline.json`, `benchmarks/vnni512-ab-candidate.json`, `benchmarks/vnni512-ab-baseline-r2.json`, `benchmarks/vnni512-ab-candidate-r2.json`.

---

## Row Tile Investigation

Row-tiled qdot uses 8 independent accumulator chains across output rows, following llama.cpp's ARTX02-F05 architectural pattern (Percival audit).

### Isolated

Row tile produced approximately 2× GMAC/s in isolated probe. This is the strongest isolated result of all three kernel investigations.

### Production (5-iteration A/B)

Run 1:
- Baseline decode: 36.8 ± 0.85 tok/s
- Candidate decode: 37.2 ± 0.67 tok/s
- Delta: +0.4 tok/s (+1.1%, marginal)

Run 2 (repeat):
- Baseline decode: 35.2 ± 0.30 tok/s
- Candidate decode: 35.4 ± 0.34 tok/s
- Delta: +0.3 tok/s (+0.8%, within noise)

Stage-level breakdown (row-tile candidate vs baseline, run 1):

| Stage | Baseline ms | Candidate ms | Delta |
|---|---|---|---|
| ffn_down | 711.6 ms | 653.4 ms | −8.2% (improved) |
| lm_head | 810.7 ms | 894.5 ms | +10.3% (regressed) |
| ffn_gate_up | 1258.4 ms | 1257.4 ms | flat |

ffn_down improved (+8.2%). lm_head regressed (−10.3%). lm_head share is 22–25% of decode time; ffn_down share is 18–20%. The lm_head regression cancelled the ffn_down gain at the production level.

Prefill: marginally improved in candidate vs baseline (122.3 vs 115.2 in run 1; 128.7 vs 131.5 in run 2 — mixed sign across runs).

Source: `benchmarks/row-tile-ab-baseline.json`, `benchmarks/row-tile-ab-candidate.json`, `benchmarks/row-tile-ab-baseline-r2.json`, `benchmarks/row-tile-ab-candidate-r2.json`.

---

## RoPE Table Cache Investigation

RoPE table cache pre-computes sin/cos into a table once per model load, eliminating 384 redundant `sin_cos()` recomputations per decode step.

### Production (5-iteration A/B)

Run 1:
- Baseline decode: 36.4 ± 1.06 tok/s
- Candidate decode: 36.9 ± 0.23 tok/s
- Delta: +0.5 tok/s (+1.4%)

Run 2 (repeat):
- Baseline decode: 34.8 ± 1.31 tok/s
- Candidate decode: 36.6 ± 0.98 tok/s
- Delta: +1.8 tok/s (+5.1%)

Stage-level improvement (candidate vs baseline, run 1):

| Stage | Baseline ms | Candidate ms | Delta |
|---|---|---|---|
| qkv | 344.0 ms | 244.3 ms | −29.0% (improved) |
| fixup | 99.2 ms | 12.0 ms | −87.9% (improved — fixup includes rope apply) |

The `fixup` stage (which contains RoPE application) shrank from 99 ms to 12 ms. The qkv stage also improved. These are real, directional changes in the same direction across both runs.

Prefill also improved: baseline 120.1 → candidate 128.2 tok/s (run 1), 125.4 → 129.4 (run 2).

This is the only kernel-class change in this investigation that produced a consistent production improvement in both runs.

The RoPE fix eliminated 384× redundant recomputation of sin/cos per decode step — genuinely redundant work, not just the same work done faster.

Source: `benchmarks/rope-ab-baseline.json`, `benchmarks/rope-ab-candidate.json`, `benchmarks/rope-ab-baseline-r2.json`, `benchmarks/rope-ab-candidate-r2.json`, `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## Bias/Residual SIMD Investigation

The bias-add and residual-add SIMD path produced a measured 2.81× isolated speedup.

The absolute time occupied by this path was measured at 0.085% of decode wall-clock time.

At 37 tok/s decode, 0.085% of one token's time is approximately 0.23 ms total. 2.81× speedup of 0.23 ms saves 0.15 ms per token. This is below the measurement noise floor (glbench std_dev on this machine is typically 0.6–1.0 tok/s, corresponding to tens of milliseconds per 128 tokens).

Result: not pursued. The isolated speedup is real; the production impact is unmeasurable.

Source: `notes/issues/glproc-throughput-gap-vs-llamacpp.md`.

---

## GATE Calibration Overhead

GATE Wave A/B/C (kernel selection calibration at session init) adds approximately 2.5 seconds of startup latency.

This was measured directly by timestamping log lines from process start. The design estimate was ~150 ms for the common case; the actual measurement is approximately 17× higher due to dual-load cost (both Q4_K_M native and Q8_0 candidates are loaded once to calibrate).

GATE overhead does not appear in per-token throughput. Decode and prefill tok/s remain within the pre-GATE baseline range.

Source: `benchmarks/post-gate-benchmark-report.md`.

---

## Thread Scaling (glbench thread-scale)

On this machine (2 physical / 4 logical cores, Qwen2.5-0.5B):

| Threads | Decode tok/s | Scaling efficiency |
|---|---|---|
| 1 | 16.2 | 100% |
| 2 | 26.9 | 83% |
| 4 | 29.5 | 46% |

Sub-linear scaling, consistent with this model's 2 KV heads limiting multi-thread attention utilization.

Source: `glbench/RESEARCH_REQUIREMENTS.md` (thread-scale feature entry).

---

## Precision Gap vs llama.cpp

PPL measured on WikiText-2 (256-token non-overlapping chunks):

| Configuration | PPL |
|---|---|
| llama.cpp native Q4_K | 24.78 ± 3.69 |
| glproc native Q4_K | 32.91 |
| glproc Q8_0-repack (production default) | 36.12 |
| glproc native Q4_K + forced scalar | 36.33 |

Q4_K→Q8_0 repack accounts for ~9% of the total gap. Scalar vs SIMD does not explain the remaining gap (forcing scalar made things worse, not better). Root cause of the remaining ~33 PPL points is not yet identified — narrowed to RMSNorm, RoPE, or softmax formula mismatch against llama.cpp.

Tokenization mismatch was ruled out (token IDs identical, verified byte-for-byte).
BOS handling was ruled out (both engines respect the model's `add_bos_token: false`).

Source: `notes/issues/glproc-precision-gap-vs-llamacpp.md`.
