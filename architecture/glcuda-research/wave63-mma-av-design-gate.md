# Wave 63 - compensated-MMA AV design gate

## Outcome

Wave 63 rebases the remaining path to 15,000 Q8 prefill tok/s after the valid
Wave 62 rejection of N32/M32 GEMM. It changes no product kernel or default.
The next experiment is frozen around one new mechanism: replace the scalar-FMA
AV half of the retained `mma4-regq` attention kernel with compensated f16
Tensor Core products while keeping QK, softmax, cache layout, and the causal
contract unchanged.

## Production gap

The apples-to-apples Wave 57 T4 result is the current production reference:

- GwenLand Q8_0 median: 11,499.7 tok/s, 21.218 ms;
- llama.cpp Q8_0 median: 11,453.5 tok/s, 21.304 ms; and
- target latency at 244 tokens and 15,000 tok/s: 16.267 ms.

The target therefore needs 4.951 ms, or 23.3%, removed from the unprofiled
production median. Wave 58's traced current stack accounts for 20.260 ms:

| stage | time | share |
|---|---:|---:|
| FFN gate/up | 6.408 ms | 31.6% |
| FFN down | 4.441 ms | 21.9% |
| FFN elementwise | 2.515 ms | 12.4% |
| attention core | 4.064 ms | 20.1% |
| QKV | 1.293 ms | 6.4% |
| attention output GEMM | 0.873 ms | 4.3% |
| attention norm + KV write | 0.667 ms | 3.3% |

The retained fused-Q8 glue already combines attention/FFN RMSNorm with
activation quantization and combines SwiGLU with quantization. Repeating that
fusion is not a new lever. Fusing only the attention-output quantizer removes
about 21 MB of f32 reads and 24 launches, but cannot close a 4.951 ms gap.

The FFN alone would need to fall from 10.849 ms to at most 6.856 ms if every
other stage stayed fixed: a 1.58x speedup. If attention falls by 2x, it saves
2.032 ms and the required FFN speedup falls to about 1.22x. The admissible
route is therefore compound; neither a small glue fusion nor another wider
GEMM tile can honestly claim a path to 15k by itself.

## Why AV is the next large attention lever

The retained kernel already maps QK to compensated
`mma.sync.aligned.m16n8k8` and keeps causal scores in shared memory. After
softmax, however, each warp still computes four query rows with scalar f32
FMAs over the full value history. The mathematical shape owned by one CTA is
already a GEMM:

```
P[16, cached_len] * V[cached_len, 64] -> O[16, 64]
```

This maps directly to eight N8 output fragments per K8 step. It does not need
a score matrix in global memory, a split-K workspace, a new allocation during
inference, or a different KV layout.

Plain f16 is not admissible. Wave 19 showed that three compensated products
miss the retained attention tolerance by 7.46x, while four products
`HH + HL + LH + LL` pass with 4.1x headroom. Wave 64 must apply the same
four-product numeric discipline independently to the softmax-weight by value
product; QK's result cannot be assumed to transfer.

## Frozen Wave 64 feasibility experiment

Wave 64 is diagnostic only. It must not alter the production dispatcher.

1. Build an isolated SM75 AV kernel for the exact production fixture:
   244 tokens, 14 query heads, two KV heads, head dimension 64. It consumes
   the retained normalized causal score rows and f32 V cache.
2. Split both operands into f16 high and low components in registers and issue
   all four `m16n8k8` products. K tails are zero-predicated; output remains
   packed f32.
3. Compare against the retained scalar AV using the same score matrix and V
   values. The numeric gate is max absolute error at or below `1e-5`, with
   finite outputs and no tolerance relaxation.
4. Assemble with `ptxas -v`. Zero stack, zero spill stores, and zero spill
   loads are mandatory. Record registers, static/dynamic shared memory, and
   driver-reported active blocks per SM.
5. Run a position-balanced direct A/B with at least 10 warmups, 100 measured
   iterations, and five repeats. Every repeat must be correct. Median AV
   speedup must be at least 1.50x to justify integration work.
6. Include cached-length tails 1, 17, 241, and 244 so a favorable aligned-only
   result cannot pass.

Any compiler, resource, numeric, or timing failure stops Wave 64. A feasibility
pass licenses a later fused production candidate; it is not a production
retention claim.

## Production integration boundary

A later integration wave may replace only the AV section inside
`gl_attn_mma4_regq_fused_f32`. It must preserve the retained compensated QK
instruction order, causal softmax values, register-Q schedule, grid, fallback,
and score-capacity rules. One CTA-wide barrier between completed softmax rows
and cooperative MMA AV is expected and must be counted explicitly.

Production retention still requires the real `glbench` Q8_0 prefill path,
same model and 244-token prompt, position-balanced pairs, full CUDA parity,
teacher-forced logits characterization, and no decode/tail regression beyond
the existing gates. The 15,000 tok/s objective is achieved only by a measured
production median at or above 15,000; this design document makes no such
claim.

## Closed directions respected

- Wave 62 N32/M32 remains rejected; block count is not treated as occupancy.
- Plain f16 and three-product compensated attention remain rejected.
- No CUTLASS dependency or opaque library result is introduced.
- No per-request device allocation, split-K scratch, or host-sized f32 weight
  materialization is introduced.
- Fused SwiGLU, K128 superstaging, accumulator-chain splitting, wider copy,
  multi-stream prefill, and N32 geometry are not restacked.

