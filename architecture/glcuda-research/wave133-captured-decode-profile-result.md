# Wave 133 - captured decode graph profile result

## Outcome

Wave 133 adds an opt-in diagnostic profiler for the logical stages inside the
captured decode graph. On a Tesla T4, two instrumented runs produced the same
stage ordering: `ffn_gate_up` was largest, followed by `attention`,
`ffn_down`, and `qkv`.

The profiler is not production retention authority. Adding 218 timed event
nodes made decode 2.0524x slower, from a control P50 of 205.50 tok/s to an
instrumented P50 of 100.13 tok/s. The stage results identify where to run the
next isolated A/B, but their percentages are not production wall-time shares.

The counterbalanced controls measured a P50-of-session-medians of 15,025.26
tok/s prefill and 205.50 tok/s decode. The prefill target remains met, while
decode remains below its ceiling.

## Production controls

Four independent `glbench` processes ran in control-profile-profile-control
order. Every process used a 244-token prompt, 64 decode tokens, two warmups,
ten measured iterations, deterministic sampling, and a 50-token `glproc`
oracle comparison.

| Order | Arm | Prefill P50 | Decode P50 | Oracle |
|---:|---|---:|---:|---:|
| 1 | control | 15,049.01 tok/s | 206.49 tok/s | 50/50 |
| 2 | profiled | 15,048.45 tok/s | 99.91 tok/s | 50/50 |
| 3 | profiled | 15,078.99 tok/s | 100.35 tok/s | 50/50 |
| 4 | control | 15,001.52 tok/s | 204.51 tok/s | 50/50 |

The profiler flag was the only arm difference. Control telemetry contained no
decode stages; both profiled runs contained all ten expected stages.

## Stage ranking

The table reports the median across the two profiled runs. Per-token values
divide the accumulated stage time by the 640 measured decode tokens in each
session.

| Rank | Logical stage | Median total | Instrumented ms/token |
|---:|---|---:|---:|
| 1 | `ffn_gate_up` | 1,179.16 ms | 1.8424 |
| 2 | `attention` | 1,039.45 ms | 1.6241 |
| 3 | `ffn_down` | 918.23 ms | 1.4347 |
| 4 | `qkv` | 677.99 ms | 1.0594 |
| 5 | `attn_prepare` | 512.78 ms | 0.8012 |
| 6 | `attn_out` | 495.15 ms | 0.7737 |
| 7 | `lm_head` | 385.09 ms | 0.6017 |
| 8 | `ffn_input` | 300.25 ms | 0.4691 |
| 9 | `ffn_activation` | 266.19 ms | 0.4159 |
| 10 | `attn_input` | 224.24 ms | 0.3504 |

Each layer stage contains 24 event intervals per token, while `lm_head`
contains one. Event-node overhead therefore inflates the layer buckets and
prevents treating these values as production attribution. The important
signal is the repeatable ordering of the large buckets, not the absolute
duration or derived share.

## Implementation and failed first attempt

The first hardware run correctly failed its telemetry contract: ordinary
`cuEventRecord` calls made during stream capture become graph dependency edges,
not event-record nodes, so `cuEventElapsedTime` had no timestamps to read.

The retained implementation resolves `cuEventRecordWithFlags` dynamically and
uses `CU_EVENT_RECORD_EXTERNAL` only for the opt-in captured-graph profiler.
This creates real event-record nodes. The symbol remains optional, preserving
the requirement that `glcuda` builds and falls back normally on machines with
no CUDA driver or an older driver. The production graph is unchanged when
`GLCUDA_PROFILE_DECODE_STAGES` is absent.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- Host library tests: 75 passed.
- Serial CUDA parity: 36 passed.
- Standard CUDA Graph replay: 4 passed.
- Instrumented CUDA Graph replay: 4 passed and logged the profiler path.
- Four production sessions: 50/50 oracle agreement each.
- Wave 129 compiler-resource gate: zero stack and zero spill.
- Result archive SHA-256:
  `a1131218ff56b439d095d66cc78840e65feb63a1a0d8a1ce41440487c5f28df5`.

## Next decision

Wave 134 should isolate the decode `ffn_gate_up` projection first. It is the
largest bucket in both profiles, and its stacked output streams roughly twice
the weights of `ffn_down`. Any candidate still needs an uninstrumented
counterbalanced production A/B; the Wave 133 profile alone cannot retain it.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave133-decode-graph-profile.json`.
- Four raw sessions:
  `benchmarks/glcuda-t4-wave133-glbench-{1..4}-*.json`.
- Full logs and raw records:
  `benchmarks/glcuda-t4-wave133-decode-graph-profile-results.zip`.
- Reproducible notebook:
  `notebooks/glcuda_t4_wave133_decode_graph_profile.ipynb`.
