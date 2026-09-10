# Wave 102 - production dispatch-gate diagnostic

## Result

Wave 102 reached the requested Tesla T4, reconstructed the retained source,
passed 65 host tests, repeated the direct N16-prefetch gate twice, built
`glbench`, verified the byte-locked Q8_0 model, and completed both one-iteration
stabilization runs with 50/50 oracle agreement.

The run stopped before all 20 production sessions because the harness expected
the candidate to report three GEMM path labels. Its actual dispatch was the
correct replacement set:

- baseline: `bstage-n16-m32`, `bstage-n16`;
- candidate: `bstage-n16-m32`, `bstage-n16-prefetch`.

For the covered `4864 x 896 x 244` projection, prefetch replaces retained N16;
the two labels are mutually exclusive. The checker incorrectly required the
candidate to retain `bstage-n16` in addition to its replacement.

This is a benchmark-harness rejection, not a candidate rejection. No production
throughput result was produced.

## Evidence reached before the stop

- direct run 1: 172.339 us to 140.127 us, 1.2299x, bit-exact;
- direct run 2: 165.884 us to 131.701 us, 1.2595x, bit-exact;
- retained/candidate occupancy: three active blocks per SM;
- host tests: 65 passed, zero failed;
- model: Qwen2.5-0.5B-Instruct Q8_0, 675,710,816 bytes;
- model SHA-256:
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`;
- stabilization oracle: 50/50 for both arms.

The one-iteration stabilization readings (6,058.0 and 6,363.6 tok/s) include
cold-start behavior, have no warmup, and have no variance estimate. They are
not production metrics and are excluded from the retention decision.

## Repair

Wave 103 must make the candidate dispatch set a replacement, regenerate the
notebook, and rerun the unchanged 10-pair production protocol. No kernel or
retention threshold should change.

Raw archive:
`benchmarks/glcuda-t4-wave102-production-dispatch-gate-miss.zip`

Archive SHA-256:
`963b47007196a18683d052b5a286396fb59c5e0da8a64db58dff427ec5941049`

The fastest verified experimental production result remains 11,377.9 tok/s;
the 15,000 tok/s target remains open.
