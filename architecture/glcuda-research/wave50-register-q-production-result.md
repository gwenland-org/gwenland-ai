# glcuda Wave 50 - N16 GEMM and register-Q production result

Wave 50 closes the requested +1,000 tok/s sprint on one Tesla T4. The retained
production arm combines Wave 27's N16 B-stage GEMM with Wave 48's
register-resident-Q attention schedule.

## Reproduction contract

- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- model SHA-256: `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- prompt: 244 tokens
- protocol: six balanced three-arm permutations, with 5 cold, 5 warmup and 10
  measured prefills per arm
- correctness: 66/66 host tests, 33/33 CUDA parity tests and a 50/50 exact token
  oracle in every production session
- raw evidence: [`benchmarks/glcuda-t4-wave50-production.json`](../../benchmarks/glcuda-t4-wave50-production.json)
- reproducer: [`notebooks/glcuda_t4_wave50_regq_attention.ipynb`](../../notebooks/glcuda_t4_wave50_regq_attention.ipynb)

## Production result

| arm | P50 tok/s | P50 latency | delta from previous | cumulative delta |
|---|---:|---:|---:|---:|
| pre-N16 retained stack | 10,146.2 | 24.049 ms | - | - |
| N16 GEMM | 10,774.4 | 22.647 ms | +628.2 tok/s | +628.2 tok/s |
| N16 + register-Q | **11,252.8** | **21.684 ms** | +478.5 tok/s | **+1,106.6 tok/s** |

All six paired cumulative gains were positive; the worst was +1,085.1 tok/s.
The final arm is +10.91% over the same-session pre-N16 baseline. Relative to
Wave 20's separately measured 10,242.4 tok/s midpoint it is +1,010.4 tok/s,
but that cross-wave difference is context only, not the controlled claim.

## Resource and direct gates

| kernel | registers | static shared | spills | blocks/SM |
|---|---:|---:|---:|---:|
| retained MMA4 attention | 48 | 4,096 B | 0 | 3 |
| register-Q MMA4 attention | 64 | 0 B | 0 | 4 |

At the production shape, the register-Q kernel measured 149.362 us versus
194.238 us for retained MMA4, a 1.3004x direct speedup, and was bit-exact to it.
The N16 entries compiled at 72/64 registers with 9,728 B static shared memory
and zero spills.

## Integration decision

**PRODUCTION RETENTION PASS.** `GLCUDA_GEMM_N16=1` and
`GLCUDA_ATTN_MMA4_REGQ=1` select the retained paths. The shipped source excludes
the rejected fused-SwiGLU, K128 and independent-MMA-chain experiments. The four
retained PTX entry bodies are byte-identical to the measured Wave 50 snapshot;
only comments delimiting removed rejected entries differ.
