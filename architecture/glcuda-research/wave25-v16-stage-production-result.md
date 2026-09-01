# glcuda Wave 25 - 16-byte cooperative-stage production result

- executed notebook: `wave25-v16-stage-v2`
- final notebook SHA-256:
  `c4685365a69a3b81f5cec635231e218c48a7f6895bf1314d459631d952eed534`
- embedded Wave 25 patch SHA-256:
  `6e956d09c37230c873720d867561940f5d47419f6f816296518fb465eddcb1de`
- reconstructed sm_75 PTX file SHA-256:
  `26e1e312623eb5c8639f1357dc502d9f638e980f9b8c3f3719c68251834f5380`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- Kaggle decision version: 1
- model SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- result archive SHA-256:
  `b6b7d4933f7e9614579ab2d3bfd93fd74630633d8759d135881ee11bc7034c6f`

## Question and causal scope

Wave 25 asks whether doubling each cooperative activation and weight payload
copy from 8 to 16 bytes reduces memory-instruction issue cost in the retained
K32 B-stage GEMM. Two lanes issue `ld.global.v2.u64` and
`st.shared.v2.u64` instead of four lanes issuing scalar u64 copies.

The candidate preserves the K32-major weight image, scalar activation and
weight scale loads, N64/N128 launch geometry, 9,728-byte shared image, two
barriers, 16 MMA instructions, dequantisation and f32 accumulation order,
guards, stores and ragged-tail behavior. It is isolated behind
`GLCUDA_GEMM_V16=1`. Both production arms retain Wave 20 `mma4-fused`
attention; the rejected Wave 22, Wave 23 and Wave 24 paths remain disabled.

## Harness history and declared deviations

The first local host-test attempt exposed a lifetime error in the new Rust
structural-test helper. After explicit authorization, the closure was changed
to a nested `arithmetic_suffix` function. The candidate PTX stayed byte
identical at the SHA above, and the repaired stack passed 66/66 tests.

A pre-upload static audit then rejected notebook v1, SHA-256
`9e2fadf92a412853bb56ba7c5f11f6c3f39f8a94419794cec28933cc9a71c545`:
its stabilization call requested one warmup and one measured iteration, while
the frozen protocol permits exactly one discarded launch. The harness-only v2
repair changed that call to 0 cold, 0 warmup and 1 measured iteration. It also
asserted the exact direct-screen dimensions, recomputed the hard-screen ratio
from the recorded microseconds, and exposed MAD in the report. The embedded
38,930-byte Wave 25 patch remained byte identical. Only v2 was uploaded and
executed.

`rustfmt --check` remains inapplicable to the cumulative historical
`kernels/mod.rs` and `runner.rs` reconstruction because earlier embedded
patches are not rustfmt-clean; rewriting them would change the evidence stack.
The new direct example passes its exact-file rustfmt check. The first broad
Kaggle output download also hit an SSL EOF while traversing exported Cargo
cache files; a selective retry fetched the unchanged result ZIP and execution
log, and the ZIP passed CRC validation.

## Correctness and resources

The final image passed every registered prerequisite:

- all 14 embedded patches matched their manifest SHA-256 values, applied
  cleanly and left `git diff --check` clean;
- 66/66 `glcuda` library tests and 33/33 serial release CUDA parity tests;
- bit-exact f32 output at both production GEMM shapes and the K160 diagnostic;
- exact 50/50 token-oracle agreement in all eight production sessions;
- `mma4-fused` attention in both arms and `bstage-v16` only in the candidate;
- exactly 5 cold and 10 measured positive samples in every session; and
- no `FAILED.json` in the CRC-valid result archive.

| entry | registers | static shared | stack | spill stores | spill loads |
|---|---:|---:|---:|---:|---:|
| retained K32 B-stage | 49 | 9,728 B | 0 B | 0 B | 0 B |
| 16-byte stage | 50 | 9,728 B | 0 B | 0 B | 0 B |

The driver reports four active candidate blocks/SM at 256 threads and two at
512 threads, so the extra register does not cross the frozen residency tier.

The direct gate passed, but its direction was mixed:

| shape | retained | V16 | retained/V16 | V16 throughput delta | bit-exact |
|---|---:|---:|---:|---:|---|
| `ffn_gate_up` 9728x896x244 | 595.716 us | 604.674 us | 0.9852x | -1.48% | yes |
| `ffn_down` 896x4864x244 | 197.599 us | 195.538 us | 1.0105x | +1.05% | yes |
| ragged N64 K160 128x160x17 | 5.447 us | 5.495 us | 0.9913x | -0.87% | yes |

Both production ratios remain above the frozen 0.95x hard-stop threshold, so
the production gate was allowed to run.

## Production metrics

The fixed 244-token workload used one discarded stabilization invocation per
arm, then four position-balanced pairs in orders A/B, B/A, B/A and A/B. Every
production arm/session contained 5 cold, 5 warmup and 10 measured iterations.
The table reports the median of the four session statistics for each arm.

| arm | P50 tok/s | P90 tok/s | P99 tok/s | latency P50/P90/P99 | MAD | session max | cold P50/P90 | decode tok/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| retained K32 | 10,271.5 | 10,388.2 | 10,444.0 | 23.756/24.215/24.400 ms | 0.151 ms | 24.421 ms | 46.863/48.359 ms | 200.7 |
| 16-byte stage | 10,386.1 | 10,447.9 | 10,578.9 | 23.494/23.728/23.805 ms | 0.125 ms | 23.814 ms | 47.045/48.778 ms | 202.5 |

- primary ratio of session-P50 medians: **+1.116%**;
- within-pair deltas: **-0.925%, +1.265%, +1.479%, -0.794%**;
- median within-pair delta: **+0.235%**;
- only two of four pairs are positive; worst pair: **-0.925%**;
- median session-P50 latency delta: **-1.106%**;
- median session-maximum latency delta: **-2.485%**;
- every paired decode delta remains above the -5% floor; and
- all 40 cold and 80 measured timing records are finite and positive.

## Decision and interpretation

**WEAK / DO NOT RETAIN.** The +1.116% primary result is below the frozen +5%
retention bar, and two of four paired repeats are negative. The retained K32
B-stage kernel remains the production default, and no candidate product code
or dispatch change is promoted. Because the result is not `RETAIN`, the
protocol does not require a second byte-identical Kaggle run.

Wider cooperative copies preserve correctness, resource limits and residency,
but the direct shapes move in opposite directions and the median paired gain
is only +0.235%. That is not evidence of a durable end-to-end lever. Future T4
GEMM work should not revisit payload-copy width on this retained K32 schedule.
