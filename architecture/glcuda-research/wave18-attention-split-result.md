# glcuda Wave 18 - Attention pass split

- notebook: `attn-split-v4`
- notebook SHA-256: `20972feb0af2494b3c56d0fb85bc513f12087bed690b16cddb9613f279cf887`
- GPU: Tesla T4, compute capability 7.5
- Kaggle runs: versions 8 and 9, identical notebook bytes
- timing: 10 warmups, 3 repeats x 100 measured launches per arm
- device gates per run: 32/32 parity tests and 61/61 library tests

## Why the first qk4 result was rejected

The initial stop-0 qk4 probe took 651.2 us against 575.4 us for the real
kernel, a 13.2% mismatch. The rows control differed by only 1.0%. A probe that
does not reproduce the kernel it models cannot attribute that kernel's time, so
the raw qk4 phase split was discarded.

The suspected occupancy cliff was then measured and falsified. Both the real
qk4 kernel and both probe images use 63 registers, spill nothing, and permit
eight resident 128-thread blocks per SM. The rows kernel and probe both use 36
registers and permit the same eight blocks per SM. Reusing `%r47` and `%p7` in
the diagnostic probe therefore does not cross an occupancy tier; it merely
removes the unnecessary named probe registers.

Two decision-depth runs of that probe still disagreed when the stop-0 probe and
real kernel were timed in separate batches: one missed by 11.3%, while the
other matched within 0.5%. The validity gate was order-biased even though the
three-arm phase split itself was counterbalanced.

Wave 18 v4 makes validity a distinct paired A/B inside every repeat, ordered
`probe -> real -> real -> probe`. The phase split remains separately
counterbalanced. This keeps clock and thermal drift from attaching to one side
of the validity comparison.

## Reproduced measurement

| run | qk4 stop-0 / real | qk4 QK | qk4 softmax | qk4 AV | rows stop-0 / real | rows QK | rows / qk4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kaggle v8 | 616.9 / 620.4 us (-0.6%) | 65.7% | 3.5% | 30.7% | 795.2 / 796.3 us (-0.1%) | 74.9% | 1.283x |
| Kaggle v9 | 622.1 / 621.6 us (+0.1%) | 65.6% | 4.0% | 30.4% | 791.6 / 793.4 us (-0.2%) | 74.0% | 1.277x |

The shipped qk4 path is therefore directly attributable: **QK 65.7%, softmax
3.8%, AV 30.5%** on average across the two fresh runs. The QK share reproduced
within 0.1 percentage point. Rows remains a useful path-specific cross-check at
74.5% QK and about 1.28x qk4's wall time, but its phase shares do not transfer
to qk4.

## Decision

**CONFIRMED.** The paired stop-0 probe reproduces the shipped qk4 kernel in two
identical decision-depth runs, with no spill or resource-tier mismatch. The
qk4 phase split is evidence; the earlier unpaired qk4 measurements remain
rejected.

At the measured 27.5% attention share of a 25.03 ms prefill, qk4 QK is about
4.52 ms, or 18.1% of the whole prefill. Softmax is about 0.26 ms, or 1.0%, so
online-softmax remains an architectural choice rather than the speed lever.

The 15k conclusion is an Amdahl projection, not a measured production result:
the existing vendor-GEMM projection reaches 16.84 ms / 14,501 tok/s; a further
1.5x QK speedup removes about 1.51 ms and projects 15.33 ms / about 15.9k tok/s.
That makes combined GEMM and QK work sufficient on paper, with 1.5x QK still an
optimistic target that must be measured by a later wave.
