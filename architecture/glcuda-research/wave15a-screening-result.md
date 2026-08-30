# glcuda Wave 15A screening - four independent QK chains per warp

- notebook: wave15a-screen-v1
- GPU: 0, Tesla T4, 7.5, 15360, 580.159.04
- Wave 13B patch (retained baseline): 51aed3b9a31adb8d902257c605b79181bc1973ef5ab123a5036148a0e6ca878c
- Wave 15A patch: 6e74a7dfc484fef9671f024f5424deee19e8cb49aaf6703507a84b19314494b5
- runs: 3 (medians), each counterbalanced over both orders
- parity gate: test result: ok. 28 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 11.41s

## The pre-registered target

Attention is 34.6% of prefill and QK is 71% of
attention, so +24% end to end needs attention **2.27x**, which needs
QK **4.71x**. Fixed before this kernel existed.

## Measured

| shape | CTAs | retained us | four-chain us | ratio | spread | implied QK | end-to-end | bit-exact |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| full_grid | 3416 | 776.67 | 575.88 | 1.349x | 0.5% | 1.57x | +9.8% | True |
| single_cta | 1 | 52.88 | 30.95 | 1.709x | 1.9% | 2.40x | +16.8% | True |
| long_ctx | 3416 | 3334.34 | 2566.60 | 1.299x | 0.4% | 1.48x | +8.7% | True |

- **REJECT - 1.35x is under the 1.50x that would make the kernel worth its own maintenance**

## How to read this

- `ratio` is retained/four-chain measured **inside one invocation**, both
  orders averaged. Absolute microseconds are not evidence on this machine:
  it drifted 5-19% between sessions during Wave 14.
- `implied QK` is what QK itself must have done to move the whole kernel by
  the measured ratio, given the 71% share. It is a derivation, not a
  measurement, and it is only ever as good as that share.
- `end-to-end` applies Amdahl at the 34.6% attention share. It is an
  isolated-kernel projection and bounds what the wave can buy; the
  interleaved same-session A/B in the engine remains the only retention
  authority, and its bar is unchanged at a 5% median.
- `single_cta` is the cleanest test of the ILP argument: with one CTA there
  is no other work to hide the dependent shfl chain behind, so if the
  chains help anywhere they help most there.