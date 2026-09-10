# Wave 104 - event-timed CUDA stage profile

## Result

Wave 104 collected five Tesla T4 sessions per arm with
`GLCUDA_TELEMETRY=1`, Q8_0 model and prompt locks, exact dispatch checks, and
50/50 `glproc` oracle verification. The telemetry describes the last measured
prefill iteration in each session and records all eight expected CUDA stages.

The retained Wave 80 stack provides the authoritative hotspot ranking:

| CUDA stage | median | share | share range across sessions |
| --- | ---: | ---: | ---: |
| FFN gate/up | 8.653 ms | **33.62%** | 33.47-33.79% |
| FFN down | 6.119 ms | **23.78%** | 23.58-23.91% |
| attention | 4.218 ms | 16.39% | 16.17-16.44% |
| FFN elementwise | 2.847 ms | 11.06% | 10.77-11.68% |
| QKV | 1.772 ms | 6.89% | 6.86-7.01% |
| attention out | 1.220 ms | 4.74% | 4.72-4.77% |
| attention norm | 0.676 ms | 2.63% | 2.49-2.64% |
| KV write | 0.232 ms | 0.90% | 0.83-0.93% |

The median sum of retained stage times was 25.737 ms. Profiled wall time was
29.574 ms, versus 20.854 ms in the unprofiled Wave 103 production result on a
different allocation. Absolute profiled throughput is therefore diagnostic
only; the stable within-session stage proportions are the evidence used here.

## Candidate observation and limitation

N16 prefetch reduced median gate/up time from 8.653 to 8.159 ms, but unrelated
candidate stages drifted upward and candidate profiled wall time was slower.
The five-pair alternating schedule gives each arm both positions but, because
the pair count is odd, assigns positions 3/2 rather than equally. Consequently
the generated `position_balanced: true` field is imprecise and the cross-arm
stage delta is not a retention result. Wave 103's unprofiled 20-session result
remains authoritative for the candidate decision.

The direct gate independently passed again with at least 1.2607x speedup and
bit-exact output.

## Decision for Wave 105

Test the existing complete-register-prefetch N16 kernel on the exact FFN-down
shape `out=896, in=4864, ntok=244` behind a separate opt-in. Wave 88 restricted
the candidate to gate/up, so the longest-K production shape remains untested.
FFN-down is now 23.78% of profiled CUDA time, while the kernel implementation
already exists and preserves the K32 accumulation order.

The direct gate must compare retained and candidate on the same allocations,
remain bit-exact, preserve zero spills and driver occupancy, and reach at least
1.10x twice before any production run. Gate/up stays opt-in and its Wave 103
retention rejection is not overridden.

The fastest valid production measurement remains the rejected experimental
12,232.7 tok/s configuration. Reaching 15,000 still requires another 22.62%.

Raw archive: `benchmarks/glcuda-t4-wave104-cuda-stage-profile-v1.zip`

Archive SHA-256:
`e896e3f24d796b4d7e74644748331ecdd136613e24c2ca42464d17fc6971b2b5`
