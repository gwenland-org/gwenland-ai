# glcuda Wave 15B screening - how many QK chains does GQA7 want?

- notebook: wave15b-chains-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- Wave 15B patch: 9cc0ea7558c78ac92560f3737acb73df7037de46cccab4f753f126484b7d4a91
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 29 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.26s

## 1. Resource evidence, before any inference from timing

Wave 15A inferred an occupancy story from a P95 number and was wrong. This
is the measurement that inference should have rested on. It runs first and
is archived whatever it says.

| entry | regs | static smem | dynamic smem | spills | blocks/SM | warps/SM | occupancy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gl_attn_decode_rows_f32` | 36 | 0 | 1012 | 0/0 | 8 | 32 | 100% |
| `gl_attn_rows_qk4_f32` | 63 | 0 | 1012 | 0/0 | 8 | 32 | 100% |
| `gl_attn_decode_rows_gqa7_f32` | 48 | 0 | 8944 | 0/0 | 7 | 28 | 88% |
| `gl_attn_gqa7_qk2_f32` | 48 | 0 | 8944 | 0/0 | 7 | 28 | 88% |
| `gl_attn_gqa7_qk4_f32` | 48 | 0 | 8944 | 0/0 | 7 | 28 | 88% |

- GQA7 family resident blocks/SM: {'gl_attn_decode_rows_gqa7_f32': 7, 'gl_attn_gqa7_qk2_f32': 7, 'gl_attn_gqa7_qk4_f32': 7}
- **same occupancy tier: True** - the factorial is only valid if this is true,
  because otherwise chain count is confounded with occupancy.

## 2. Chain count, isolated

| shape | CTAs | ctx | 1 chain us | 2 us | 4 us | 2x ratio | 4x ratio | spread 2/4 | bit-exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| full_grid | 488 | 244 | 552.43 | 534.29 | 494.57 | 1.034x | 1.117x | 0.4%/0.4% | True |
| one_group | 244 | 244 | 282.77 | 304.27 | 283.03 | 0.929x | 0.999x | 0.5%/0.5% | True |
| long_ctx | 488 | 288 | 856.20 | 664.52 | 581.83 | 1.288x | 1.472x | 0.7%/1.8% | True |

- projected end to end at the production shape: 2 chains +1.1%, 4 chains +3.8%
- screening gates: {"bit_exact": true, "zero_spills": true, "same_resource_tier": true, "isolated_ratio": true}
- **PROCEED to production with chains=4 (1.117x isolated, +3.8% projected end to end)**

## How to read this

- All three arms read the same eight-row K tile and launch the same grid,
  so chain count is the only thing that moves. The resource table above is
  what makes that claim checkable rather than asserted.
- Ratios are medians of counterbalanced runs inside one invocation.
  Absolute microseconds are not evidence on this machine: it drifted 5-19%
  between sessions during Wave 14.
- The end-to-end column is Amdahl at the 34.6% attention share. It bounds
  what a wave can buy. The production gate is separate and is judged
  against retained GQA7 with median tail across sessions - never with P95
  inside one group of ten, which is what produced Wave 15A's spurious veto.