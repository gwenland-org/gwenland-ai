# glcuda Wave 17 - NumStages=2 on the B-stage GEMM

- notebook: wave17-pipeline-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- patch: be0e5e239699fd3ac1735bb93239da7c694564815852559321da122fc01d2ed4
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 32 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.16s

## What this tests

Wave 16B ablated the kernel production runs and found the exposed
global-load latency to be the largest non-arithmetic thing in it:
**25.9%** unhidden staging, against 2.4% for barriers, 2.3% for the f32
epilogue, and 9.0% for tensor-core issue.

The pipelined kernel issues each k-block's four global loads one iteration
early, so their latency lands under the previous block's arithmetic - what
CUTLASS's sm_75 int8 example spends `NumStages = 2` on. The stage is four
registers per thread, not a second shared buffer, so the shared footprint
and the 6-blocks-per-SM occupancy tier are exactly the retained kernel's.

Pre-registered: **>= 1.10x** to carry into production,
**< 1.05x** is not worth its maintenance, and bit-exactness
is checked on the device before any timing is read.

## Measured

| shape | out x in | retained us | pipelined us | ratio | spread | share of the 25.9% | end-to-end | bit-exact |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `ffn_gate_up` | 9728x896 | 384.74 | 370.80 | 1.038x | 4.2% | 14% | +1.7% | True |
| `ffn_down` | 896x4864 | 196.42 | 190.11 | 1.033x | 1.0% | 12% | +1.5% | True |

- **REJECT - worst shape 1.033x is under the 1.05x that would make the kernel worth its own maintenance. The exposed staging latency was measured, but prefetching it one block early did not collect it - so it was not the thing waiting.**

## How to read this

- `share of the 25.9%` is how much of Wave 16B's measured unhidden staging
  this actually collected. It is the number that says whether the ablation's
  mechanism was real, and it can embarrass the wave as easily as confirm it.
- `end-to-end` applies Amdahl at the FFN's 45.6% share of prefill. It is an
  isolated-kernel projection and bounds what the wave can buy; the
  interleaved production A/B against the retained kernel remains the only
  retention authority, at four repeats and a tail judged across sessions.
- Ratios are medians of counterbalanced runs inside one invocation.
  Absolute microseconds are not evidence on this machine.