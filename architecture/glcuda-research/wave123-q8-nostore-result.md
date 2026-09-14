# Wave 123 - Q8 no-store and stacked gate/up

## Outcome

Wave 123 retained the Q8 no-store glue path and contiguous stacked gate/up
dispatch. The production Tesla T4 A/B improved in both counterbalanced
invocations, with all candidate deltas positive and the oracle token unchanged.

## Production result

The combined decision statistic used 40 samples per arm:

- retained median: 19.103201 ms = 12,772.7 tok/s;
- candidate median: 18.333203 ms = 13,309.2 tok/s;
- speedup: 1.0420x (+4.20%);
- oracle token: 3323 for both arms;
- 15,000 tok/s target: not reached.

The two invocation summaries independently agreed:

| Invocation | Retained | Candidate | Speedup |
|---|---:|---:|---:|
| A | 12,774.2 tok/s | 13,367.3 tok/s | 1.0464x |
| B | 12,736.9 tok/s | 13,199.6 tok/s | 1.0363x |

## Profile guidance

The instrumented profile is diagnostic only and is not the production
throughput authority. It places the remaining time primarily in the two FFN
GEMMs:

| Stage | Time | Share of enclosing GPU interval |
|---|---:|---:|
| `ffn_gate_up` | 4.864064 ms | 25.47% |
| `ffn_down` | 4.287040 ms | 22.45% |
| `attention` | 2.923424 ms | 15.31% |
| `qkv` | 1.308448 ms | 6.85% |
| `ffn_silu_quant` | 1.238368 ms | 6.48% |

At the combined production median, 15,000 tok/s requires 16.267 ms, leaving a
2.067 ms gap. The next candidate therefore needs a material FFN GEMM change;
another isolated glue optimization cannot close the target alone.

## Validation

- GPU: Tesla T4, sm_75, driver 580.159.04.
- Model: 675,710,816 bytes, SHA-256
  `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e`.
- CUDA parity: 36 passed.
- Library tests: 67 passed.
- Build and PTX assembly completed with no spills reported for the changed
  kernels.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave123-q8-nostore.json`.
- Evidence archive: `benchmarks/glcuda-t4-wave123-q8-nostore-results.zip`.
- Archive SHA-256:
  `377446d3cf91b9867c61208d3a0ce3bdf5c0e421fd5eb593f56313fcbf54b058`.
