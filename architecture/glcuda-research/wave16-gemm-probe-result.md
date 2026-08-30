# glcuda Wave 16 probe - which part of the GEMM mainloop is the 12x gap?

- notebook: wave16-gemm-probe-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- patch: 8003242678da50e8400f4414db819223d6511ec734ddcf04921ddcd639160abe
- runs: 3 (medians), each counterbalanced forward and reverse
- parity gate: test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 12.40s

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

| shape | out x in | barriers/CTA | full us | no math | no barriers | no staging | GMAC/s | spread |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `ffn_gate_up` | 9728x896 | 56 | 472.09 | 160.12 | 458.39 | 343.04 | 4505 | 4.2% |
| `ffn_down` | 896x4864 | 304 | 225.47 | 63.03 | 206.45 | 168.42 | 4716 | 10.0% |

### What each removal buys

| shape | staging+barriers alone (floor) | removing barriers | removing staging |
|---|---:|---:|---:|
| `ffn_gate_up` | 33.9% of the kernel | +2.9% | +27.3% |
| `ffn_down` | 28.0% of the kernel | +8.4% | +25.3% |

## The prediction this was built to test

`ffn_down` carries **5.4x** the barriers of `ffn_gate_up` for an
identical MAC count. If barrier density is the mechanism, removing barriers
must help it proportionally more.

- gate_up saves **+2.9%**, down saves **+8.4%** -> ratio **2.91x** against 5.4x the barriers
- **BARRIER DENSITY CONFIRMED - down saves 2.9x what gate_up saves against 5.4x the barriers. Pipelining the mainloop is the lever, and NumStages=2 is the shape of the fix.**

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