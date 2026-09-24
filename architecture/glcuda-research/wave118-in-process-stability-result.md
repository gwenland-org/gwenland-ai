# Wave 118 - in-process production stability result

## Result

Wave 118 ran the retained and Wave 111 deferred-residual paths through one
loaded Q8_0 model, one glcuda runner, and one Tesla T4 primary context. The
harness switched the single causal flag between synchronized production
prefill calls instead of launching a fresh process per arm.

Two independent invocations used reversed ABBA order. Each invocation
collected 100 samples per arm, for 400 total production samples. Every sample
used the exact 244-token prompt and matched the same glproc oracle token.

| arm | samples | median | range |
| --- | ---: | ---: | ---: |
| retained | 200 | 12,729.4 tok/s | 12,351.2-13,253.6 |
| Wave 111 | 200 | **12,793.0 tok/s** | 12,376.1-13,275.8 |

Paired log-speedup analysis:

- median speedup: **1.00432x** (+0.43%);
- P10/P90: 0.98101x / 1.02477x;
- bootstrap 95% CI: 0.99908x-1.00814x;
- invocation A median: 1.00142x;
- invocation B median: 1.00688x.

The diagnostic gate passed: both invocations agreed in direction and the
confidence interval excluded a 1% regression. This result explains the mild
sign reversals in Wave 116 as resolution/noise around a very small effect; it
does not turn the effect into a production-sized win.

## Correctness and environment

- Tesla T4, compute capability 7.5, driver 580.159.04;
- pinned Q8_0 model, 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`;
- 67 glcuda host tests passed;
- full serial CUDA parity passed with zero failures;
- 400/400 measured outputs matched the oracle;
- production dispatch selected exact fusion, Grid2D, N128 B-stage, N16
  prefetch, and MMA4 register-Q/AV attention as required;
- no PTX changed; ptxas reported zero spills for the existing SM75 entries.

## Decision

Close Wave 111 as **correct but below production reach**. Keep it opt-in and
do not promote it as the default. Its measured +0.43% cannot materially close
the gap from 12,793.0 to 15,000 tok/s.

The 15,000 tok/s target remains unachieved. At 244 tokens it requires 16.267
ms; the Wave 118 candidate median is approximately 19.073 ms, leaving about
2.806 ms or 14.72% of candidate latency to remove. The next optimization must
span a dominant compute stage rather than another launch-only micro-fusion.

Raw archive:
`benchmarks/glcuda-t4-wave118-in-process-stability-results.zip`

Archive SHA-256:
`30354dcc3682d99d908f3d1945c3ee20b03b0765c2430b8bd01202d169efeeb3`
