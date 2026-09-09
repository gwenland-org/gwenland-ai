# Wave 103 - N16 register-prefetch production result

## Result

Wave 103 completed the full Q8_0 production protocol on Tesla T4: ten
position-balanced pairs, twenty independent `glbench` sessions, one cold
iteration, five warmups, ten measured iterations per session, and `glproc`
oracle verification.

| arm | median session P50 | median latency P50 | median session max |
| --- | ---: | ---: | ---: |
| retained Wave 80 stack | 11,700.4 tok/s | 20.854 ms | 22.117 ms |
| N16 prefetch candidate | **12,232.7 tok/s** | **19.955 ms** | 23.408 ms |

The candidate gained 532.4 tok/s, or 4.55%, by the ratio of session-P50
medians. The median paired delta was +4.14%, and nine of ten pairs were
positive. One pair was -0.99%. Candidate median session-max latency was 5.84%
higher than retained, beyond the predeclared 5% tail ceiling.

## Gate decision

**REJECT for retention.** The candidate failed two predeclared requirements:
every paired throughput delta must be positive, and median session-max latency
must remain within 5%. Decode and oracle gates passed.

The 12,232.7 tok/s value is a valid production measurement and the fastest
experimental configuration measured so far, but it is not a retained/default
configuration. It remains opt-in.

## Supporting evidence

- direct run minimum speedup: 1.2281x across two bit-exact runs;
- resources: retained 72 registers, candidate 80, both 9,728 bytes shared and
  zero spills;
- observed dispatch:
  - retained: `bstage-n16-m32`, `bstage-n16`;
  - candidate: `bstage-n16-m32`, `bstage-n16-prefetch`;
- oracle: 50/50 decode-prefix tokens in every production session;
- Q8_0 model SHA-256:
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.

The generated result inherited the numeric field `"wave": 80` from the Wave
80 production template. Kernel identity, version 2, source revision, dispatch,
and raw records establish this as Wave 103; the stale numeric label does not
alter the measurements.

## Target gap

The fastest measured experimental number is now 12,232.7 tok/s, 2,767.3 tok/s
or 22.62% below 15,000 tok/s. N16 gate/up prefetch is directionally useful but
cannot close that gap alone. The next optimization wave must attack another
production-dominant projection while leaving this rejected candidate opt-in.

Raw archive:
`benchmarks/glcuda-t4-wave103-n16-prefetch-production-v2.zip`

Archive SHA-256:
`b4679a2b5ef1ffd5b97f27fcf54f93db552c60003ebc88ce76f19aba05a47f5d`
