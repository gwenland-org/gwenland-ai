# glcuda Wave 16 probe - which part of the GEMM mainloop is the 12x gap?

- notebook: wave16-gemm-probe-v2
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- patch: a21668f960934afb3cb0914aa2c71bfef96e7b24dfefbe3f579c85df4f2fb70a
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 32 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.96s

## Why this probe exists

Attention is finished: QK is 63% of the attention kernel and attention is
34.6% of prefill, so a free QK caps at +32.6% end to end. Meanwhile prefill
is **5.85x behind llama.cpp** and the healthiest FFN GEMM runs at **8.2% of
the T4's int8 peak** - about **33 us of tensor-core work against ~398 us
measured**. Not bandwidth: Wave 14 walked the weight image across the L2
for a -0.9% change.

The mainloop stages, barriers, issues 16 `mma.sync`, barriers again, every
**32 K-elements**. CUTLASS's sm_75 int8 example spends `NumStages = 2` to
stop exactly that serialisation; MMQ reaches the same end with a 256-K,
39 KB tile. Before building either, this removes the pieces one at a time.

⛔ Every ablated arm computes the WRONG ANSWER by construction. These are
times, never outputs, and none of this is on the inference path.

## Measured

| shape | out x in | full us | no math | no barriers | no staging | no epilogue | **no A loads** | GMAC/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ffn_gate_up` | 9728x896 | 361.47 | 224.78 | 354.48 | 271.73 | 351.47 | **352.93** | 5884 |
| `ffn_down` | 896x4864 | 208.68 | 132.54 | 185.88 | 152.76 | 207.62 | **207.10** | 5096 |

### What each removal buys

| shape | floor | barriers | staging | epilogue | **A loads** |
|---|---:|---:|---:|---:|---:|
| `ffn_gate_up` | 62.2% of the kernel | +1.9% | +24.8% | +2.8% | **+2.4%** |
| `ffn_down` | 63.5% of the kernel | +10.9% | +26.8% | +0.5% | **+0.8%** |

## The prediction this was built to test

`ffn_down` carries **5.4x** the barriers of `ffn_gate_up` for an
identical MAC count. If barrier density is the mechanism, removing barriers
must help it proportionally more.

- gate_up saves **+1.9%**, down saves **+10.9%** -> ratio **5.65x** against 5.4x the barriers
- **A-OPERAND LOADS ARE NOT THE WALL - removing them entirely buys only 2%. The 26.7% left in the math block is something else again, and the geometry rewrite has NO measured case behind it.**

## How to read this

- Every number is a median of counterbalanced runs inside one invocation.
  Absolute microseconds are not evidence on this machine; it drifted 5-19%
  between sessions during Wave 14.
- The floor column is staging plus barriers with the math block skipped. It
  bounds what any amount of arithmetic tuning could ever recover.
- A probe retains nothing and decides nothing on its own. It exists so the
  next wave is chosen from a mechanism instead of from a plausible story -
  which in Wave 15 killed four candidate explanations for the price of one
  session each.