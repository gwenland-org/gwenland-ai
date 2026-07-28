# Bottleneck Model

The glproc pipeline for a single transformer block, broken into stages with their measured bottleneck classification.

---

## Machine Parameters

All measurements are from the i3-1115G4 reference machine.

- Peak measured read bandwidth: 31.43 GB/s (varies 23.3–31.8 GB/s session to session).
- CPU: 2 physical / 4 logical cores, AVX2 + AVX-512 VNNI.
- Model: Qwen2.5-0.5B Q4_K_M loaded as Q8_0. Total weight bytes: 0.63 GiB.

Roofline definition used by glbench:
- If `GB/s ≥ 90% of ceiling`: bandwidth-bound.
- If `arithmetic_intensity > 50 FLOP/byte` AND `GB/s < 5% of ceiling`: not-bandwidth-bound (compute-bound candidate).
- Otherwise: indeterminate.

---

## Decode Pipeline

In decode, one new token is generated. The sequence length is 1 for each transformer forward pass (GEMV-regime, not GEMM).

| Stage | Share | GB/s | Ceiling% | Arith. intensity | Verdict |
|---|---|---|---|---|---|
| ffn_gate_up | 35.5% | 22.4 | 68.2% | 1.88 FLOP/byte | Indeterminate |
| lm_head | 22.4% | 23.1 | 73.5% | 1.88 FLOP/byte | Bandwidth-bound |
| ffn_down | 20.0% | 19.8 | 63.1% | 1.88 FLOP/byte | Indeterminate |
| qkv | 9.3% | 10.1 | 28.5% | 2.09 FLOP/byte | Indeterminate |
| attn_out | 6.2% | 11.7 | 37.2% | 1.88 FLOP/byte | Indeterminate |
| attention | 5.9% | 4.2 | 13.4% | 2.09 FLOP/byte | Indeterminate (stall suspected) |
| sampler | 0.6% | — | — | — | Unknown |

Source: `benchmarks/full-bottleneck-e2e.json`.

**Key observations:**
- lm_head is the only stage with a hard bandwidth-bound classification. Any optimization that reduces bytes read by lm_head has potential.
- ffn_gate_up dominates time share (35.5%) but is indeterminate — not clearly limited by either compute or bandwidth.
- attention stage bandwidth utilization (4.2 GB/s) is anomalously low for its time share (5.9%). One glbench hypothesis: "stalled on something other than memory traffic." Source: `benchmarks/simd-check.json`.
- Overall ceiling efficiency: 57.3%. The pipeline is not saturating bandwidth.

---

## Prefill Pipeline

In prefill, all prompt tokens are processed in parallel (GEMM-regime). Sequence length = 220 tokens in these measurements.

| Stage | Share | GB/s | Arith. intensity | Verdict |
|---|---|---|---|---|
| ffn_gate_up | 50.3% | 1.83 | 60.2 FLOP/byte | Not-bandwidth-bound |
| ffn_down | 20.6% | 2.24 | 67.5 FLOP/byte | Not-bandwidth-bound |
| qkv | 5.9% | 1.85 | 55.9 FLOP/byte | Not-bandwidth-bound |
| attn_out | 5.8% | 1.47 | 44.4 FLOP/byte | Not-bandwidth-bound |
| attention | 3.2% | 0.91 | 67.0 FLOP/byte | Not-bandwidth-bound |
| ffn_downq | 6.1% | — | — | Unknown |
| fixup | 5.5% | — | — | Unknown |
| serial | 2.6% | — | — | Unknown |

Source: `benchmarks/full-bottleneck-e2e.json`.

**Key observations:**
- All measured stages in prefill are "not-bandwidth-bound". Arithmetic intensity is 44–67 FLOP/byte vs decode's 1.88–2.09 FLOP/byte.
- Prefill is compute-bound. The leverage is compute throughput, not memory traffic reduction.
- ffn_gate_up dominates (50.3% of prefill time).
- The `fixup` stage is 5.5% of prefill time and uncharacterized. RoPE cache brought it to 0.7% (from 5.5% in pre-cache runs, derived from the `fixup` ms reduction in rope-ab measurements).

---

## Bottleneck Regimes by Phase

| Phase | Dominant regime | Lever |
|---|---|---|
| Decode | Mixed (bandwidth-adjacent, indeterminate) | Memory traffic reduction, work elimination |
| Prefill | Compute-bound | Compute throughput per cycle (SIMD width, ILP) |

This is why an optimization targeting compute throughput (VNNI-512, Row Tile) may help prefill and be neutral on decode — or help decode only in a specific stage that is compute-bound while leaving the rest unchanged.

---

## Stage Ownership Map

Each stage maps to glproc source code:

| Stage | Code location | Kernel |
|---|---|---|
| qkv | `glproc/src/attention.rs` | QKV projection |
| attention | `glproc/src/attention.rs` | Scaled dot-product |
| attn_out | `glproc/src/attention.rs` | Output projection |
| ffn_gate_up | `glproc/src/engine.rs` (dispatch), `glproc/src/kernels/` | "Q8_0 fused-swiglu integer-dot" |
| ffn_down | `glproc/src/engine.rs`, `glproc/src/kernels/` | "Q8_0 integer-dot" |
| lm_head | `glproc/src/model.rs` | "Q8_0 integer-dot" |
| fixup | `glproc/src/runner.rs` | RoPE apply, norm |
| serial | `glproc/src/runner.rs` | Embedding lookup, non-parallelizable serial ops |
| sampler | `glproc/src/sampler.rs` | Token sampling |

---

## Unknown Stages

The following stages have no bandwidth measurement in glbench telemetry (`bytes_read: null`):
- `fixup` (prefill and decode)
- `ffn_downq` (prefill)
- `serial` (prefill and decode)
- `sampler` (decode)

These stages contribute 8–14% of total decode time and ~13–14% of prefill time combined. Their bottleneck classification is unknown. They are not currently instrumented for memory traffic.
