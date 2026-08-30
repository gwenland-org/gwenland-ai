# glcuda Wave 15C audit - why did four chains buy only 1.17x?

- notebook: wave15c-audit-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- Wave 15C patch: f50b800b4209495d1941cd2ab8ce3e4ad685ddaa71685d4b465d4f5a101b3ee9
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.56s

## A + C. GQA7's own pass split, and the floor before any arithmetic

The 71% QK share every Wave 15 ratio rests on was measured on the ROW
kernel. GQA7 reads K from shared memory and runs seven heads per CTA, so
this is the first time its own split has been measured.

| stage | us | share of kernel |
|---|---:|---:|
| tile loop + both barriers, no scores | 67.70 | 12.5% |
| + QK | 343.79 | 63.1% |
| + softmax | -0.14 | 0.0% |
| + PV | 131.06 | 24.4% |
| whole probe | 542.41 | 100% |
| real kernel | 558.68 | probe matches: True |

- **QK share: GQA7 63.1% vs the row kernel's 71%** (delta -7.9 points)
- Wave 15B's 1.117x kernel ratio therefore implies QK itself moved **1.20x** using GQA7's own share, against the 1.17x I reported from the row kernel's

## B. Occupancy sweep: are ILP and TLP substitutes?

The kernel is untouched. Only the dynamic shared-memory request is padded,
so resident blocks per SM fall and strictly nothing else changes.

| blocks/SM | warps/SM | pad bytes | 1 chain us | 4 chains us | 4-chain ratio | spread |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 28 | 418 | 726.50 | 500.65 | 1.456x | 1.8% |
| 6 | 24 | 1978 | 710.02 | 491.72 | 1.444x | 1.3% |
| 5 | 20 | 4163 | 852.21 | 544.71 | 1.565x | 0.5% |
| 4 | 16 | 7440 | 847.74 | 542.26 | 1.563x | 0.9% |
| 3 | 12 | 12901 | 1262.14 | 764.67 | 1.650x | 0.2% |
| 2 | 8 | 23824 | 1241.53 | 756.15 | 1.642x | 0.3% |

## What this settles

- GQA7's QK share is 63%, close enough to the row kernel's 71% that the derived ratios stand
- the pre-arithmetic floor is only 12%, so barriers are not the wall
- the four-chain advantage GROWS as occupancy falls (1.456x at 7 blocks -> 1.642x at 2), so ILP and TLP are substitutes: at full occupancy the chains arrive to a job the resident warps have already done

Every number is a median of counterbalanced runs inside one invocation.
Absolute microseconds are not evidence on this machine: it drifted 5-19%
between sessions during Wave 14. An audit retains nothing and decides
nothing on its own - it exists so the next wave is chosen from a mechanism
instead of from a plausible story.