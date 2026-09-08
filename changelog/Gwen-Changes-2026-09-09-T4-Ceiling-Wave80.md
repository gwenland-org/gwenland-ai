# T4 Ceiling Wave 80 - fused MMA AV production gate

## Result

Two independent Tesla T4 production `glbench` allocations measured the opt-in
compensated-MMA AV attention path against retained register-Q attention on the
same Q8_0 model and 244-token workload. Candidate P50 medians improved by 6.73%
and 3.30%; paired medians improved by 5.36% and 4.70%.

The candidate midpoint is 11,377.9 tok/s, up 4.96% from the retained midpoint
of 10,840.1 tok/s. It also produced a 50/50 contiguous glproc decode prefix in
every session versus retained's stable 29/50.

## Decision

The path remains opt-in. Each allocation contained one negative paired prefill,
and the second allocation exceeded the predeclared session-maximum tail gate.
The production improvement is reproduced, but the retention gate is not.

The 15,000 tok/s target remains open by 3,622.1 tok/s (31.83%). Further work
must target the dominant GEMM/FFN path rather than treating attention as enough.
