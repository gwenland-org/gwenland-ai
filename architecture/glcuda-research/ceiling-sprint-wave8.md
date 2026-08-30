# T4 Ceiling Sprint — Wave 8

## Status

Prepared for T4 measurement. Local host gates pass; real-hardware PTX assembly, bit parity, and production retention remain gated by the Kaggle notebook.

## Why Wave 7 was not retained

Wave 7 measured a strong prefill gain, but changed the activation scale contract from one scale per K32 block to one scale per row. Kernel-reference tests passed against that new contract, while production greedy parity against `glproc` fell to 9/50 tokens. The performance number is therefore invalid as a shippable optimization.

Wave 8 keeps the useful scheduling hypothesis and removes the numerical change.

## Single causal lever

The retained `gl_quantize_q8` launches one 32-thread CTA for every K32 activation block. For a 244x896 activation this is 6,832 CTAs. Wave 8 adds `gl_quantize_q8_rowcta`, launched as one 256-thread CTA per token row:

- warp `w` owns K32 block `w`;
- every warp repeats at block `w + 8`, `w + 16`, and so on;
- each warp uses the same five-step max reduction, lane-0 broadcast, `amax / 127`, reciprocal, round-to-nearest-even conversion, clamp, byte store, and f32 scale store as the retained kernel;
- there is no shared memory and no CTA barrier;
- the 244x896 case falls from 6,832 CTAs to 244 CTAs (28x fewer).

The candidate is opt-in through `GLCUDA_Q8_ROWCTA=1`. Dispatch requires `rows > 1`, so single-row decode and CUDA graph capture keep the old kernel and launch geometry.

## Clean candidate composition

The Kaggle candidate is constructed from:

1. baseline revision `3bce8dd7b8aaa2765855ab927c611b54981f9241`;
2. retained Wave 3 patch;
3. retained Wave 4 patch;
4. the Wave 8-only patch.

Rejected Wave 5, Wave 6, and Wave 7 patches are not applied. The sm_75 MMA module and retained dynamic-shared attention kernel must remain byte-identical between arms.

## Gates

Wave 8 may be retained only when all of these hold on a Tesla T4:

- both PTX modules assemble for `sm_75` with zero spills;
- the new quantizer remains in the projected 32-resident-warp tier;
- old and row-CTA quantizers produce bit-identical quant bytes and f32 scale bits, including zero blocks and rounding ties;
- the full hardware parity suite runs without a CUDA skip;
- both baseline and candidate preserve exact greedy next-token parity against `glproc`;
- every paired production session improves prefill P50 and mean by at least 5%;
- P95 prefill latency and decode throughput do not regress by more than 5%.

The isolated quantizer microbenchmark is diagnostic only.

## Local evidence

- Clean BASE+Wave3+Wave4+Wave8 candidate: `cargo check -p glcuda --locked` passed.
- Clean library suite: 43 passed.
- Clean parity binary: 22 passed locally, but the machine has no CUDA device, so this does not count as hardware parity.
- PTX structural gate: ASCII, LF-only, five down-shuffles, one index broadcast, no shared memory, no barriers, and no `cp.async`.
- Wave 8 patch SHA-256: `03746e114f8228f5aebfe43399b29e179c17c805f084fafd754f9d0cc878fb0e`.

