# glcuda Wave 19 - compensated-f16 MMA QK feasibility

- notebook: `wave19-mma-qk-v3`
- notebook SHA-256: `bd224b853b3d4e9c01173fc00aa76026aa1ecffb7de43dc19208512d5fcaa7c1`
- embedded Wave 19 patch SHA-256:
  `7f2116e3e4e44ae5124f590457ee005f218ef546558b5b268e43a77162db198e`
- GPU: Tesla T4, compute capability 7.5
- decision runs: Kaggle versions 2 and 3
- timing: 10 warmups, 100 measured launches, five paired forward/reverse repeats
- device gates per decision run: 61/61 `glcuda` library tests

## Question and registered gate

Wave 18 measured QK at 65.6-65.7% of the retained qk4 attention kernel. Plain
f16 Tensor Core operands were already rejected in Wave 15: at the production
shape they changed the final attention output by as much as `1.227e-2`, while
the retained attention parity tolerance is `1e-5`.

Wave 19 tests whether two f16 components can retain f32-class accuracy while
still making Turing Tensor Cores useful:

```text
x = hi + lo
QK = Qhi*Khi + Qhi*Klo + Qlo*Khi + Qlo*Klo
```

The gate was fixed before the device run:

1. the three-product form, without `Qlo*Klo`, is a negative control;
2. the four-product form must keep final attention max absolute error at or
   below the existing `1e-5` tolerance;
3. both device arms must materialise the same causal score matrix;
4. the candidate must spill nothing and must not cross an occupancy tier;
5. paired-median QK speedup must be at least `1.5x`;
6. passing licenses a fused-production experiment only. It does not change
   dispatch.

The diagnostic uses
`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`, the sm_75-supported f16
MMA shape documented by NVIDIA. One warp covers 16 query rows by eight key
rows. Eight K=8 chunks cover the production head dimension of 64, and four MMA
products per chunk implement the compensated dot product.

## Numeric gate

The host oracle uses the production fixture: 244 tokens, 14 query heads, two KV
heads and head dimension 64.

| variant | max abs attention error | RMS relative | existing `1e-5` gate |
|---|---:|---:|---:|
| three products: HH + HL + LH | `7.460e-5` | `7.148e-6` | fail |
| four products: HH + HL + LH + LL | `2.466e-6` | `3.242e-7` | pass |

The split reconstructed 251,349 of 251,392 fixture operands bit for bit
(99.983%). The 43 remaining operands are why the low-low term cannot be
discarded: three products miss the retained tolerance by 7.46x, while four
products have about 4.1x headroom.

## Device parity and resources

Both decision runs produced the same parity result:

| measurement | result |
|---|---:|
| MMA4 versus qk4 causal score max abs | `1.311e-6` |
| qk4 attention versus f32 oracle max abs | `1.490e-7` |
| MMA4 attention versus f32 oracle max abs | `2.682e-7` |
| retained attention tolerance | `1.000e-5` |

`ptxas -v` and the CUDA occupancy query also reproduced:

| kernel | registers | shared | spills | active blocks/SM |
|---|---:|---:|---:|---:|
| qk4 score diagnostic | 34 | 0 B | 0 | 16 |
| compensated MMA4 | 43 | 4,096 B | 0 | 16 |

The candidate therefore pays nine registers and one barrier to stage the
two-component Q tile, but does not lose a block-residency tier on T4.

## Reproduced timing

Each repeat measures both orders and contributes one paired ratio. Version 2's
notebook initially printed the ratio of independently sorted medians; the raw
paired observations remain valid, but that summary value (`2.4488x`) is not the
decision statistic. Version 3 fixes the reducer to `median(ratio_samples)`.

| run | qk4 median | MMA4 median | paired-median speedup | repeat range |
|---|---:|---:|---:|---:|
| Kaggle v2 | 192.930 us | 78.784 us | `2.3237x` | `2.2027x-2.4488x` |
| Kaggle v3 | 201.863 us | 87.344 us | `2.3100x` | `2.2349x-2.6312x` |

Absolute clocks moved between sessions, but the paired-median speedup
reproduced within about 0.6%. Every one of the ten paired repeats cleared the
registered `1.5x` bar.

## Decision

**FEASIBILITY PASS; NOT RETAINED IN PRODUCTION.** Four-product compensated-f16
MMA preserves the existing attention tolerance and clears the Wave 18 QK target
by a large margin in a same-contract, QK-only device benchmark. Three products
are rejected. Plain f16 remains rejected.

This wave deliberately does not add the kernel to the product module or change
the attention dispatcher. The diagnostic materialises scores in global memory;
the retained attention kernel keeps scores in shared memory and immediately
runs softmax and AV. A speedup over the diagnostic score contract does not prove
that conversion, tiling, causal tails and synchronisation still net-win after
fusion.

If the `2.31x` QK ratio transferred without loss, Wave 18's Amdahl model would
remove about 2.56 ms from a 25.03 ms prefill. That projects about 10.9k tok/s on
the current GEMMs, or about 17.1k tok/s after the separate vendor-GEMM
projection. Those numbers are scenario math, not measured production results.

The next admissible experiment is a fused 16-query by 8-key attention tile that
keeps the qk4 fallback, preserves causal tails, performs softmax and AV without a
global score round trip, and is decided by a same-session production prefill
A/B.
