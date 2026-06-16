# GwenLand Benchmark Report — 2026-06-03 22:00

## Environment

| Field | Value |
|---|---|
| Machine | Intel Core i3 All-in-One, no discrete GPU |
| OS | Windows 11 Home |
| Tool | Criterion 0.5 (100 samples per function, statistically validated) |
| Crate | `gwen-core` — `cargo bench -p gwen-core` |
| Profile | dev (criterion default) |
| Report | `gwen-cli/target/criterion/report/index.html` |

All values are **real measured output from Criterion**. No invented numbers.  
Units: ps = picoseconds · ns = nanoseconds · µs = microseconds · ms = milliseconds.

---

## 1. `estimate_tokens` — Token Count Estimation

**Algorithm:** `text.len() / 4` — single integer division, zero allocation.  
**Complexity claim:** O(1).

| Input Size | Mean | Median | Verdict |
|---|---|---|---|
| 11 chars | **684 ps** | 635 ps | ✓ O(1) |
| 1,000 chars | **652 ps** | 633 ps | ✓ O(1) |
| 100,000 chars | **631 ps** | 626 ps | ✓ O(1) |

Delta across 9,090× input growth: **53 ps** (noise floor). Confirmed constant.  
Target was `< 2 ns`. Actual: **~0.65 ns**. Exceeds target by 3×.

---

## 2. `detect_budget_from_model` — Model Context Budget Lookup

**Algorithm:** Sequential `contains()` scan over model name string.  
**Complexity claim:** O(n patterns) — falls through all arms on unknown model.

| Model | Mean | Median |
|---|---|---|
| `llama-3-8b-instruct` (small_7b) | **90 ns** | 88 ns |
| `codellama-13b-instruct` (medium_13b) | **82 ns** | 80 ns |
| `mistral-large-instruct-2407` (large_mistral) | **137 ns** | 135 ns |
| Unknown model (worst case) | **150 ns** | 147 ns |

Target was `< 600 ns worst-case`. Actual worst case: **150 ns**. 4× headroom.

---

## 3. `estimate_context` — Multi-File Context Assembly

**Algorithm:** Iterates over `FileEntry` slice, falls back to `size_bytes / 4` for missing paths.  
**Complexity claim:** O(n files).

| File Count | Mean | Median | Throughput |
|---|---|---|---|
| 10 files | **224 µs** | 221 µs | ~44,600 files/sec |
| 50 files | **1.26 ms** | 1.18 ms | ~39,600 files/sec |
| 100 files | **2.38 ms** | 2.24 ms | ~42,000 files/sec |

Target was `< 3 ms for 50 files`. Actual: **1.26 ms**. 2.4× headroom.  
Scaling from 10 → 100 files: ~10.6× time increase for 10× files — confirms linear.

---

## 4. `extract_query_terms` — NLP Query Tokenization

**Algorithm:** Split + stopword filter + dedup.  
**Complexity claim:** O(n words).

| Query | Words | Mean | Median |
|---|---|---|---|
| `"authenticate error"` (short) | 2 | **656 ns** | 631 ns |
| Production 403 query (medium) | ~13 | **1.95 µs** | 1.91 µs |
| Async refactor pipeline query (long) | ~28 | **5.60 µs** | 5.53 µs |

Target was `< 5 µs for realistic query`. Actual for medium: **1.95 µs**.  
Long query (28 words): **5.60 µs** — within acceptable range at 2× word count.

---

## 5. `extract_relevant_windows` — TF-Score Windowing Engine

**Algorithm:** TF scoring pass → boundary expansion → merge overlapping windows.  
**Complexity claim:** O(lines × terms). Most expensive hot-path per file.

| Lines | Mean | Median | Throughput |
|---|---|---|---|
| 100 lines | **59 µs** | 58 µs | ~1.7 MB/s |
| 500 lines | **277 µs** | 275 µs | ~1.8 MB/s |
| 1,000 lines | **593 µs** | 565 µs | ~1.7 MB/s |

Target was `< 500 µs for 500 lines`. Actual: **277 µs**. 1.8× headroom.  
Scaling 100 → 1,000 lines: ~10× time for 10× lines — confirms linear.

---

## 6. `estimate_vram` — VRAM Requirement Estimator

**Algorithm:** Pure arithmetic (multiply/add/divide). Zero allocations.  
**Complexity claim:** O(1) regardless of model size.

| Model Config | Mean | Median | Verdict |
|---|---|---|---|
| 7B · LoRA r8 · seq 1024 | **8.82 ns** | 8.44 ns | ✓ O(1) |
| 13B · LoRA r16 · seq 1024 | **8.66 ns** | 8.32 ns | ✓ O(1) |
| 70B · LoRA r8 · seq 2048 | **8.57 ns** | 8.36 ns | ✓ O(1) |

Delta across 10× parameter size: **0.25 ns** (noise). Confirmed constant.  
Target was `< 50 ns`. Actual: **~8.7 ns**. Exceeds target by 5.7×.

---

## 7. `estimate_time` — Training Time Estimator

**Algorithm:** Floating-point division: `(tokens × epochs × 6) / (tflops × 1e12)`.  
**Complexity claim:** O(1).

| Device | TFLOPS | Mean | Median |
|---|---|---|---|
| CPU (baseline) | ~0.1 | **8.77 ns** | 8.44 ns |
| NVIDIA T4 | ~65 | **7.61 ns** | 7.34 ns |
| NVIDIA A100 | ~312 | **8.49 ns** | 8.27 ns |
| NVIDIA RTX 4090 | ~330 | **6.70 ns** | 6.59 ns |

Target was `< 20 ns`. All devices: **6.7–8.8 ns**. Exceeds target by 2.3–3×.  
Variation between devices is scheduling noise, not algorithmic — all are single FP divides.

---

## Summary

| Function | Complexity | Target | Actual (mean) | Status |
|---|---|---|---|---|
| `estimate_tokens` | O(1) | < 2 ns | **~0.65 ns** | ✓ PASS |
| `detect_budget_from_model` | O(patterns) | < 600 ns | **~150 ns** | ✓ PASS |
| `estimate_context` (50 files) | O(n) | < 3 ms | **1.26 ms** | ✓ PASS |
| `extract_query_terms` (medium) | O(n words) | < 5 µs | **1.95 µs** | ✓ PASS |
| `extract_relevant_windows` (500 lines) | O(lines×terms) | < 500 µs | **277 µs** | ✓ PASS |
| `estimate_vram` | O(1) | < 50 ns | **~8.7 ns** | ✓ PASS |
| `estimate_time` | O(1) | < 20 ns | **~7–9 ns** | ✓ PASS |

All 7 benchmarks pass. All O(1) claims verified with sub-nanosecond or single-digit nanosecond results that do not grow with input size.

---

## Notes

- `estimate_context` and `extract_relevant_windows` scale linearly as claimed — verified by 10× input producing ~10× latency.
- `extract_query_terms` long query (28 words) lands at 5.60 µs, slightly above the 5 µs target for the short/medium tier. Acceptable — the target was stated for "realistic natural-language query" (medium, ~13 words).
- Hyperfine cold-start benchmark (`bench_coldstart.ps1`) is written but not yet run — requires a release build of `gwen-tui`. Cold-start results will be appended to a future report.
- Full HTML report with violin plots and regression charts: `gwen-cli/target/criterion/report/index.html`
