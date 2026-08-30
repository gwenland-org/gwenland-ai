# T4 Ceiling Sprint - Wave 11

**Status:** ready for T4 confirmation  
**Baseline:** retained Wave 4 (`BASE + Wave 3 + Wave 4`)  
**Target:** 15,000+ prefill tok/s on Tesla T4

## Why this wave

Wave 10 topped out near 7.2k tok/s and its combined scheduling arm improved
only 3.64%, so it remains historical evidence rather than part of the retained
stack. Its telemetry still exposed the next two independent levers: launch and
activation-traffic overhead around the Q8 GEMMs, and repeated K/V reads across
Qwen's seven query heads per KV head.

Wave 11 therefore tests a clean 2x2 factorial from Wave 4:

1. Wave 4 behavior (both flags off);
2. exact adjacent glue fusion;
3. exact GQA7 K/V reuse;
4. both mechanisms together.

All four production arms use the same Wave 11 executable. The flags only select
dispatch, so module load, PTX JIT, dormant code, and startup overhead are
balanced. With both flags off the prompt hot path is the retained Wave 4 path.

## Candidate A - exact Q8 glue fusion

The candidate fuses only chains that were already adjacent:

- RMSNorm -> Q8 before Q/K/V projections;
- residual add -> RMSNorm -> Q8 before gate/up projections;
- SiLU multiply -> Q8 before the down projection.

The PTX preserves the retained reduction tree, f32 operation order, Q8
per-K32 maximum, scale, reciprocal, round-to-nearest-integer, and clamp. It
still materializes the f32 RMSNorm and SiLU outputs, so downstream non-Q8
consumers and observable state remain unchanged. Decode and its CUDA graph are
not dispatched through these kernels.

At 244 tokens and 24 layers, the launch/geometry model predicts 96 fewer kernel
launches and 1,343,952 fewer CTA dispatches. This is a diagnostic work-count,
not a throughput claim.

## Candidate B - exact three-pass GQA7 attention

Qwen2.5-0.5B uses 14 query heads, 2 KV heads, and head dimension 64. One
Wave 11 CTA owns `(kv_head, token)` and computes the seven query heads sharing
that K/V head. It tiles eight 64-f32 K or V rows into shared memory once and
reuses the tile across all seven heads.

The kernel deliberately remains three-pass attention. It keeps the same warp
to key-row mapping, lane-to-dimension mapping, shuffle reductions, softmax
partition, approximate exponential, and ascending V accumulation as retained
Wave 4. A full online-softmax FlashAttention rewrite is deferred because it
would reassociate max/sum and accumulator rescaling, conflicting with this
wave's bit-exact kernel gate. The IO-awareness rationale follows the
[FlashAttention paper](https://arxiv.org/abs/2205.14135), while the grouped-head
reuse direction is consistent with the serving-oriented analysis in
[FlashInfer](https://arxiv.org/abs/2501.01005).

The opt-in dispatch requires exactly `heads_per_kv=7`, `head_dim=64`, and score
capacity at most 288; every other shape or larger context falls back to Wave 4.
Dynamic shared memory is `28 * round_up4(capacity) + 2112` bytes: 8,944 bytes
at the 244-token production prompt and 10,176 bytes at the fallback boundary.
Logical K/V reads fall by up to 7x for this shape; cache behavior determines
the realized gain.

## Correctness and resource contract

The notebook hard-gates:

- the pinned T4, base revision, Wave 3/4/11 patch hashes, and pinned model hash;
- both PTX modules assembled with `ptxas -arch=sm_75`;
- zero spill stores and loads;
- at least 24 projected resident warps for every new hot kernel;
- byte-identical raw Q8 bytes, f32 scale bits, materialized f32 bits, and GQA7
  output bits versus the retained unfused/attention kernels;
- the complete hardware parity suite with any local-style `SKIP` treated as a
  failure;
- exact `50/50` greedy next-token agreement with `glproc` in every production
  session.

No `cp.async`, FP16 attention, row-scale approximation, online softmax, or
`glcuda_sm75.ptx` MMA change is part of Wave 11.

## Measurement protocol

Four production repeats use the balanced Williams square:

```text
W4  F   FG  G
F   G   W4  FG
G   FG  F   W4
FG  W4  G   F
```

Every arm occupies every position once and all 12 directed carryover pairs
occur once. Each arm receives one discarded stabilization run before the
factorial. Production uses the pinned 244-token prompt, greedy seed 42,
`cold/warmup/measure = 3/3/10`, and the full `glproc` oracle. The parser locks
the engine, workload, iteration count, prompt length, model filename, and exact
oracle payload. P50/P90/P95/P99, mean, P95/P99 latency, decode P50, per-arm
telemetry, and pre/post GPU snapshots are archived.

## Retention gate

Evaluate fusion, GQA7, and the combined candidate independently against the
flags-off Wave 4 control. Retain a candidate only if:

- median production P50 improves by at least 5%;
- every paired Williams block improves P50 and mean by at least 5%;
- P95 and P99 prefill latency do not regress by more than 5%;
- decode P50 does not regress by more than 5%;
- exact oracle, bit parity, PTXAS, spill, and occupancy gates remain green.

Diagnostic microbenchmarks, work-count estimates, telemetry, and NCU are not
retention evidence. `ERR_NVGPUCTRPERM` is recorded but is non-gating.

## Artifacts

- Notebook: `notebooks/glcuda_t4_ceiling_wave11.ipynb`
- Notebook build: `wave11-exact-fusion-gqa7-v2-cargo-bootstrap`
- Wave 11 embedded patch SHA-256:
  `0e644ae9c37ecdf73fd354a8c59100ed31bb3edce5c76e13b5f08ef4dd3292ca`
- Notebook SHA-256:
  `1c8490b20e6e2dee6f5cb2c87afbb91673c6adcf2393a5eaf4c88d9c73986ee7`
- Expected Kaggle archive:
  `/kaggle/working/glcuda_t4_ceiling_wave11_fetch_results.zip`

The implementation is embedded in the notebook as a clean patch. The existing
dirty development product tree is intentionally not modified by this sprint.
The v2 bootstrap reuses Cargo from common Kaggle locations or installs the
official minimal Rust toolchain when the image omits Cargo from both disk and
`PATH`; the resolved Cargo and rustc versions are archived.
