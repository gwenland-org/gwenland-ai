# Wave 80 - fused compensated-MMA AV production result

## Deliverable

Measure Wave 78's opt-in fused compensated-MMA AV attention path through the
real `glbench` prefill path on two independently allocated Tesla T4 workers.
The comparison holds the true Q8_0 model, 244-token prompt, seed, sampling,
GEMM stack, warmup and oracle fixed; the candidate differs only by
`GLCUDA_ATTN_MMA4_AV=1`.

## Reproduction contract

- Kaggle kernel: `jinxsuperdev/glcuda-wave-80-hard-t4-mma-av-production`;
- valid versions: 3 and 4;
- machine shape: `NvidiaTeslaT4`;
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04;
- source revision: `48ae7c7ea53e299efca02afb8c996f938dec16bf`;
- embedded patch SHA-256:
  `da156a5ebc4b9e6b4c89181120f27511092ac1650d4eadffe022a316b34e367c`;
- submitted notebook SHA-256:
  `6608d9f7d009a7f3071427a5720ab052ea8c5004e7462ee329d2bf141434545f`;
- model: Qwen2.5-0.5B-Instruct Q8_0, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`;
- per version: ten alternating pairs, 20 sessions, five appearances in each
  position per arm;
- per session: one cold, five warmup and ten measured prefills;
- workload: 244 prompt tokens, greedy, seed 42, one generated token;
- Q8 correctness: first token must match glproc; the contiguous 50-token
  decode prefix is recorded.

Both valid versions passed full-module assembly, 33/33 serial CUDA parity and
the Wave 79 direct fused-attention gate before production measurement.

## Production result

| allocation | retained P50 | candidate P50 | median gain | paired median | positive pairs |
|---:|---:|---:|---:|---:|---:|
| v3 | 10,492.8 tok/s | 11,198.9 tok/s | +6.73% | +5.36% | 9/10 |
| v4 | 11,187.4 tok/s | 11,556.9 tok/s | +3.30% | +4.70% | 9/10 |
| midpoint | 10,840.1 tok/s | **11,377.9 tok/s** | **+4.96%** | - | 18/20 |

The direction reproduces: both independent allocations show a positive median
and nearly identical paired-median improvement. The midpoint latency falls
from 22.534 ms to 21.451 ms, a 4.80% reduction.

The numerical signal also improves. Retained register-Q matches a contiguous
29/50-token glproc prefix in every session. The candidate matches 50/50 in
every session. Both therefore meet the predeclared Q8 first-token requirement;
the candidate additionally removes the later autoregressive divergence.

## Retention decision

**REJECT RETENTION.** The candidate misses the gate that was fixed before the
runs. Version 3 contains one -10.36% pair and a decode-control regression below
-5%. Version 4 contains one -0.65% pair and its median session-maximum tail is
+14.27%. Post-hoc relaxation would invalidate the decision, so the candidate
remains opt-in despite the reproduced median production speedup.

This is production `glbench` evidence, not a probe. It still does not reach the
goal: the two-allocation candidate midpoint is 11,377.9 tok/s, leaving 3,622.1
tok/s or another 31.83%. Attention alone cannot close that gap; the next sprint
must return to the dominant GEMM/FFN path.

## Deviations and failed setup versions

- Version 1 correctly stopped after Kaggle allocated a P100. The metadata used
  an ignored `accelerator` field.
- Version 2 used the corrected hard T4 `machine_shape`, but incorrectly applied
  the Q4-era exact 50/50 glproc gate to the true Q8 model. It stopped on the
  retained baseline's known 29/50 prefix before the candidate ran.
- Versions 3 and 4 use the established Wave 57 Q8 correctness contract and are
  the only production results included in the decision.

## Evidence

- `benchmarks/glcuda-t4-wave80-mma-av-production-v3.zip`, SHA-256
  `2bc1cc7424dba44d8f5c7968152104d81c079f5ecdcb01f30ed1596f96eafbe9`;
- `benchmarks/glcuda-t4-wave80-mma-av-production-v4.zip`, SHA-256
  `ac60daea208efebf35aa6d005db3d8e38f92873803b5cca98cd82a3658443241`;
- `benchmarks/glcuda-t4-wave80-mma-av-production.json`;
- `notebooks/glcuda_t4_wave80_mma_av_production.ipynb`;
- `scripts/build_wave80.py`.

Each valid archive contains all 20 `glbench` sessions, raw measurements,
dispatch logs, compiler output, parity output, direct-gate output, parsed
records and a production-success marker.
