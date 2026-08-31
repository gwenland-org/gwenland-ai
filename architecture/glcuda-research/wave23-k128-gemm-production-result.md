# glcuda Wave 23 - K128 B-stage GEMM production result

- notebook: `wave23-k128-gemm-v1`
- final notebook SHA-256:
  `29c5b9d4d20d00ca74843bfff653d8cf7e6ab35164db0b3c6a84bef8a70b9658`
- embedded Wave 23 patch SHA-256:
  `90927231489e073259796131f3d04971f5281a2596fc843a116ad6c51fca2ad6`
- reconstructed base revision: `3bce8dd7b8aaa2765855ab927c611b54981f9241`
- GPU: Tesla T4, compute capability 7.5, driver 580.159.04
- Kaggle decision version: 2
- model SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`

## Question and causal scope

Wave 23 asks whether reducing four retained K32 stage/compute barrier pairs to
one K128 superstage can collect the exposed B-stage mainloop cost measured in
Wave 16. The candidate preserves the Wave 12 prepacked weight image, launch
geometry, MMA operands, integer accumulation, per-K32 dequantisation and f32
FMA order. It changes only shared staging and is opt-in through
`GLCUDA_GEMM_K128=1`.

Both production arms retain Wave 20 compensated-MMA attention and leave the
Wave 22 fused-SwiGLU candidate disabled. Unsupported K shapes fall back to the
retained K32 B-stage kernel.

## Validation

The reconstructed historical stack passed locally:

- `cargo check -p glcuda --example wave23_k128_gemm --locked`;
- 65/65 `cargo test -p glcuda --lib --locked` tests;
- structural assertions for the four disjoint shared stages, two barriers,
  16 static MMA instructions and ascending K32 reader strides;
- Python syntax and `nbformat` validation for every notebook cell;
- byte-exact decompression of the embedded 33,265-byte Wave 23 patch; and
- `git apply --reverse --check` plus `git diff --check`.

Kaggle version 1 assembled the candidate successfully, then stopped at two
attention-policy assertions because the parity harness incorrectly exported
`GLCUDA_ATTN_MMA4=1` to tests that explicitly require the unconfigured QK4
default. Both failures reported `left: Mma4, right: Qk4`; neither exercised the
K128 GEMM. Version 2 removed that unrelated attention flag from the parity
process and fixed notebook cell schema only. Production still enabled Wave 20
attention in both arms, and the embedded kernel patch remained byte-identical.

Kaggle version 2 passed every registered correctness gate:

- 65/65 `glcuda` library tests;
- 33/33 full CUDA parity tests with the K128 candidate selected;
- bit-exact f32 output at both production GEMM shapes and the K160 tail;
- exact 50/50 token oracle in all eight production sessions;
- retained `mma4-fused` attention in both production arms; and
- `bstage-k128` dispatch only in the candidate arm.

The resource image is legal but crosses the retained occupancy tier:

| entry | registers | static shared | active blocks/SM | stack | spill stores | spill loads |
|---|---:|---:|---:|---:|---:|---:|
| retained K32 B-stage | 49 | 9,728 B | historical 6-block tier | 0 B | 0 B | 0 B |
| K128 B-stage, 256 threads | 64 | 38,912 B | 1 | 0 B | 0 B | 0 B |
| K128 B-stage, 512 threads | 64 | 38,912 B | 1 | 0 B | 0 B | 0 B |

The direct bit-exact screen already rejects the mechanism on both production
shapes:

| shape | retained | K128 | retained/K128 | K128 throughput delta | bit-exact |
|---|---:|---:|---:|---:|---|
| `ffn_gate_up` 9728x896x244 | 418.681 us | 460.165 us | 0.9099x | -9.01% | yes |
| `ffn_down` 896x4864x244 | 217.699 us | 325.457 us | 0.6689x | -33.11% | yes |
| ragged K160 128x160x17 | 4.871 us | 4.760 us | 1.0232x | +2.32% | yes |

The K160 diagnostic confirms the shorter final superstage is correct, but it
does not represent the production matrices and cannot rescue the candidate.

## Production metrics

The fixed 244-token workload used four position-balanced pairs in orders A/B,
B/A, B/A and A/B, with 5 cold, 5 warmup and 10 measured iterations per arm.

| arm | median tok/s | median latency | median MAD | median max | decode tok/s |
|---|---:|---:|---:|---:|---:|
| retained K32 | 10,097.7 | 24.167 ms | 0.093 ms | 24.666 ms | 198.8 |
| K128 superstage | 8,681.1 | 28.108 ms | 0.114 ms | 28.411 ms | 201.5 |

- median throughput delta: **-14.03%**;
- paired throughput deltas: **-15.49%, -14.06%, -13.44%, -13.59%**;
- worst paired delta: **-15.49%**;
- median session-max latency delta: **+15.18%**;
- every paired decode delta remained above the -5% floor; and
- all token-oracle comparisons were exact.

## Decision and interpretation

**REJECT: -14.03%, with all four production pairs negative and tail latency
15.18% worse.** The retained K32 B-stage kernel remains the production default.
No product kernel or dispatch change is promoted.

The result answers the Wave 16 staging question: reducing barrier count is not
worth collapsing the retained six-block occupancy tier to one resident block.
The penalty is strongest on `ffn_down`, whose many K blocks were supposed to
benefit most, so the experiment falsifies the barrier-amortisation mechanism
rather than merely missing the +5% retention threshold. Future GEMM work must
preserve occupancy or change the arithmetic engine; larger shared superstages
should not be revisited on T4.

Because the decision is REJECT, the protocol does not require a second
byte-identical confirmation run.
