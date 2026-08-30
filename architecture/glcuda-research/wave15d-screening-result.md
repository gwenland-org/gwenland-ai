# glcuda Wave 15D screening - trading shared memory for occupancy

- notebook: wave15d-tile-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- Wave 15D patch: b48888eb976a5860a97790a26dc626894cde1db46d8d164213015fa1b2745c98
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 30 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.41s

## 1. Occupancy, as the DRIVER reports it

Wave 15C computed this from bytes, ignored the shared-memory allocation
granule, and mislabelled every point on its sweep axis. This asks
`cuOccupancyMaxActiveBlocksPerMultiprocessor` instead.

| entry | dynamic smem | blocks/SM (driver) | warps/SM |
|---|---:|---:|---:|
| `rows` | 1012 | 8 | 32 |
| `rows_qk4` | 1012 | 8 | 32 |
| `gqa7` | 8944 | 7 | 28 |
| `gqa7_qk2` | 8944 | 7 | 28 |
| `gqa7_qk4` | 8944 | 7 | 28 |
| `gqa7_t4` | 7920 | 8 | 32 |

- **tier gained: True** - the whole wave rests on this, and it is now a
  driver answer rather than my arithmetic
- for scale, Wave 15C measured one tier DOWN costing 30% on an identical kernel

## 2. Eight-row tile against four-row

| shape | ctx | 8-row us | 4-row us | ratio | spread | end-to-end | bit-exact |
|---|---:|---:|---:|---:|---:|---:|---|
| full_grid | 244 | 538.42 | 592.67 | 0.905x | 1.0% | -3.5% | True |
| one_group | 244 | 280.00 | 311.87 | 0.898x | 0.6% | -3.8% | True |
| long_ctx | 288 | 831.23 | 660.36 | 1.247x | 1.4% | +7.4% | True |

- **STOP - the tier is real but worth only 0.905x; the extra barriers ate it**

## How to read this

- The four-row tile changes the shared-memory budget and nothing else: same
  arithmetic, same launch geometry, same rows per warp in the same order,
  which is why bit-exactness is the correctness gate rather than a tolerance.
- The cost side is twice as many tiles and twice as many barriers. Wave 15C
  priced the entire tile loop plus both barriers at 12.5% of the kernel, so
  the trade only has to beat roughly that.
- Ratios are medians of counterbalanced runs inside one invocation. Absolute
  microseconds are not evidence on this machine.
- This is an isolated screening. It bounds what the wave can buy; the
  interleaved production A/B against retained GQA7 remains the only retention
  authority.