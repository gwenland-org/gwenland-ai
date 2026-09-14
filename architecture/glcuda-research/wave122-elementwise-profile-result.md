# Wave 122 - elementwise-stage production profile

## Outcome

Wave 122 repaired the coarse elementwise bucket by splitting it into named
production stages. The run was observation-only: it did not retain or reject an
optimization. It produced a valid event-timed Tesla T4 profile for the current
best Wave 118 deferred-residual stack.

## Result

- GPU prefill: 19.259329 ms over 244 tokens = 12,669.2 tok/s.
- Stage sum: 17.935136 ms, 0.931x of the enclosing GPU interval.
- Attention bucket: 6.068544 ms.
- FFN bucket: 11.270976 ms.
- Glue subtotal: 2.369120 ms.
- Largest non-GEMM glue stage: `ffn_silu_quant` at 1.546464 ms.

| Stage | Time | Share of GPU total |
|---|---:|---:|
| `ffn_gate_up` | 4.952800 ms | 25.72% |
| `ffn_down` | 4.297024 ms | 22.31% |
| `attention` | 2.894112 ms | 15.03% |
| `ffn_silu_quant` | 1.546464 ms | 8.03% |
| `qkv` | 1.328032 ms | 6.90% |

## Interpretation

The 15,000 tok/s target for a 244-token prefill requires at most 16.267 ms.
Wave 122 is therefore about 2.992 ms short. Perfectly removing the largest glue
stage would not be enough by itself; the next candidate needs either a broad FFN
GEMM win or a glue fusion that also reduces GEMM launch/scratch cost.

## Evidence

- Summary: `benchmarks/glcuda-t4-wave122-elementwise-profile.json`.
- Evidence archive: `benchmarks/glcuda-t4-wave122-elementwise-profile-results.zip`.
- Archive SHA-256: `d94fdfd52ff04d9d599d9459cb0bd0e5622cb9053447c406c9916a23dec79756`.
