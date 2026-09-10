# Wave 81 - N16/M32 wide-grid direct result

## Outcome

Wave 81 tested whether the retained paired-M32 N16 entry should replace the
wide M64 schedule for the dominant gate and up projections. The candidate was
bit-exact and raised residency, but was **7.09% slower**. The predeclared direct
gate rejected it before production `glbench`.

## Reproduction contract

- Kaggle kernel: `jinxsuperdev/glcuda-wave-81-m32-wide-production`, version 1;
- Tesla T4, sm_75, driver 580.159.04;
- shape: one gate or up projection, 4864x896x244;
- ten warmups, 100 measured launches and five counterbalanced repeats;
- source patch SHA-256:
  `68cc5e6e1672787d03563a1782208eda97b9755d09a3612958c4c863d5ab2c16`;
- result archive SHA-256:
  `175c1b8cd1de5c8cb3bf701ccc7c5fb0a8a296485d459b41fa0e2820055dfaee`.

The notebook initially exceeded Kaggle's one-megabyte source limit. Wave 85
changed only the transport envelope from raw Base64 to deterministic gzip plus
Base64. The decoded patch and candidate SHA remained unchanged.

## Correctness and resources

- full serial CUDA parity: 33/33, with no skip;
- direct output: bit-exact, no mismatched element;
- both entries: zero stack and zero spills.

| schedule | registers | barriers | active blocks/SM |
|---|---:|---:|---:|
| retained M64/N16 wide | 72 | 1 | 3 |
| paired M32/N16 | 64 | 1 | 4 |

## Direct measurement

| retained wide | candidate M32 | retained/candidate | candidate delta |
|---:|---:|---:|---:|
| 161.380 us | 173.699 us | 0.9291x | -7.09% |

The candidate met the numerical and resource gates but missed the required
1.10x speedup decisively. Doubling the N-axis CTA count duplicates activation
staging and scheduling work; the extra residency did not repay it. This also
confirms that blocks-per-SM alone is not the useful objective for this GEMM.

## Decision

**REJECT BEFORE PRODUCTION.** No production throughput number is claimed.
The rejected dispatch and harness were removed from the product tree; the
self-contained notebook and evidence archive preserve reproducibility. Future
work must not force the retained M32 entry onto already-covered wide grids.

The verified production number remains Wave 80's experimental midpoint of
11,377.9 tok/s. The 15,000 tok/s goal remains open.
