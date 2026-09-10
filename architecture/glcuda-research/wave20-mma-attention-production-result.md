# glcuda Wave 20 - fused compensated-MMA attention production result

- notebook: `wave20-production-v2`
- notebook SHA-256:
  `90f900c79a8d2ed603a865c6b4671253555c17573aac775aafed0fdb380562d2`
- embedded Wave 20 patch SHA-256:
  `26eee3ada2b71f3ad3505a1fff603caf48626676dc66555cc0fdb5af8de4fd64`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- model SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- decision runs: Kaggle versions 2 and 3, identical notebook bytes
- production protocol per run: four position-balanced repeats, five cold
  iterations, five warmups and ten measured prefills per arm
- device gates per run: 62/62 `glcuda` library tests and 33/33 parity tests

## Question and registered gate

Wave 19 proved that four compensated f16 Tensor Core products can preserve the
existing `1e-5` attention tolerance while making QK about 2.31x faster in an
isolated score-matrix diagnostic. That result did not establish a production
win because the diagnostic materialised scores in global memory.

Wave 20 asks the production question. The candidate must:

1. fuse compensated MMA QK, softmax and AV without a global score matrix;
2. preserve causal masking, GQA head mapping, padded Q row strides, chunked
   positions and short tail tiles;
3. pass `ptxas` with zero spills and have at least one resident block per SM;
4. match the existing device attention oracle at `1e-5`;
5. use the same release binary and differ from qk4 only by the
   `GLCUDA_ATTN_MMA4=1` opt-in;
6. pass exact 50/50 token-oracle validation in every production session;
7. keep every paired decode delta at or above -5% and the median session-tail
   regression at or below +5%; and
8. improve median production prefill by at least 5%, with every paired repeat
   positive, to clear the retention bar.

The two-arm order is `base -> candidate`, `candidate -> base`, `candidate ->
base`, `base -> candidate`. Each arm therefore occupies each position twice.
The fixed prompt produces 244 prompt tokens. Each decision run contains 80 raw
prefill samples; the final claim covers 160 samples and eight paired repeats.

## Fused production design

`gl_attn_mma4_fused_f32` launches one 128-thread CTA for each query-head and
16-query-row tile. Warp zero computes QK in 16x8x8 sm_75 MMA fragments. Each f32
operand is split into high and low f16 components and accumulated as
HH + HL + LH + LL, the precision-preserving form admitted by Wave 19.

The CTA keeps a fixed 4,096-byte Q high/low tile and its causal score tile in
shared memory. Four warps then own four query rows each and complete softmax and
AV in the same CTA. No score matrix is written to global memory. The production
244-token launch uses 15,616 bytes of dynamic shared memory, for 19,712 bytes
including the static Q tile. Capacities above 640 retain the qk4 fallback.

The candidate remains an explicit opt-in in the experimental patch so the A/B
can audit dispatch. The unconfigured arm announces `qk4`; the candidate arm
announces `mma4-fused`.

## Correctness and resources

The new device parity test covers three shapes:

- production: 14 Q heads, two KV heads, position base 0, 244 tokens, packed Q;
- ragged/chunked: four Q heads, two KV heads, position base 5, 17 tokens and an
  11-float Q row pad; and
- tail: two Q heads, one KV head, position base 7, three tokens and a
  five-float Q row pad.

Both decision runs passed all 33 device parity tests. Every production session
also matched the glproc token oracle exactly, 50/50.

| kernel | registers | dynamic shared at 244 | spills | active blocks/SM |
|---|---:|---:|---:|---:|
| qk4 default | 63 | 1,012 B | 0 | 8 |
| fused compensated MMA4 | 48 | 15,616 B | 0 | 3 |

The candidate uses fewer registers but trades residency for the shared score
tile. Three resident CTAs per SM are sufficient at the production grid, and
the reduction in QK work more than repays that occupancy cost.

## Reproduced production timing

The reported value for each arm is the median of its four session P50s. A
session P50 comes from ten measured 244-token prefills after five cold and five
warmup iterations.

| run | qk4 tok/s | MMA4 tok/s | qk4 median | MMA4 median | production delta | paired deltas |
|---|---:|---:|---:|---:|---:|---|
| Kaggle v2 | 8,968.0 | 10,119.4 | 27.210 ms | 24.112 ms | **+12.84%** | +11.01%, +11.84%, +14.34%, +12.26% |
| Kaggle v3 | 9,052.4 | 10,365.3 | 26.956 ms | 23.541 ms | **+14.50%** | +12.64%, +14.43%, +14.58%, +14.80% |

Across the two identical decision runs:

- midpoint production throughput is **9,010.2 -> 10,242.4 tok/s**;
- the ratio of those midpoints is **+13.67%**;
- the median of all eight paired deltas is **+13.49%**;
- all eight paired deltas are positive, spanning **+11.01% to +14.80%**;
- midpoint median latency is **27.083 -> 23.826 ms**, a **12.02%** reduction;
- the median-of-session-max tail improves by 12.19% and 12.12%; and
- midpoint decode throughput is 196.5 -> 203.9 tok/s, a +3.74% cross-check,
  although decode was not the optimization target.

Absolute clocks moved slightly between fresh T4 sessions, but both paired
production deltas clear the 5% bar by more than 2.5x and every individual pair
has double-digit headroom.

## Decision

**PRODUCTION RETENTION PASS.** The fused compensated-MMA attention candidate is
correct, spill-free and measurably faster end to end. It clears every registered
gate in two fresh runs with identical notebook bytes. Wave 19's isolated QK
speedup transfers to a reproduced **+13.67% production prefill gain** at the
pinned 244-token workload.

This research branch reconstructs the historical performance stack inside the
notebook, so this wave records the production-qualified patch and does not
silently transplant that stack into the current product tree. The patch is
licensed for integration and a default flip; qk4 remains the capacity fallback.

The measured MMA attention path reaches about 10.24k tok/s on the current GEMM
stack. It does not by itself reach 15k. The earlier vendor-GEMM scenario remains
the separate lever required to cross that target; any combined number is still
a projection until measured after integration.
