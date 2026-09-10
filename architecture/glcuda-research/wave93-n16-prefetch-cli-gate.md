# Wave 93 - N16 register-prefetch CLI gate

## Outcome

Wave 93 adds `glcuda/examples/wave93_n16_prefetch.rs`, a self-checking direct
T4 harness for the isolated Wave 88 candidate. It compares the retained N16
entry with `gl_gemm_mma_q8_bstage_n16_prefetch` at the exact production call
shape `4864x896x244`.

The harness:

- requires an SM75 device and the complete opt-in flag stack;
- rejects an unexpected M32 dispatch;
- queries retained and candidate occupancy through the CUDA driver and
  requires at least three active 256-thread blocks/SM for both;
- builds one deterministic Q8 weight image and one activation image shared by
  both arms;
- checks every f32 output bit before reading timing;
- runs five forward/reverse counterbalanced timing pairs, with ten warmups and
  100 launches per observation; and
- exits non-zero unless candidate speedup is at least 1.10x.

`cargo check -p glcuda --example wave93_n16_prefetch --locked` and
`cargo test -p glcuda --lib --locked` passed locally; the latter reports 65/65
tests. The extra PTX EOF blank line found at the prior commit gate was removed,
and `git diff --check` is clean.

## Boundary

No T4 result exists yet. Wave 94 must package the current source state into a
self-contained notebook, compile both PTX modules with `ptxas -v`, enforce the
80-register/9,728-byte/zero-spill resource gate, run the CLI twice, and archive
all raw logs. A direct failure stops before production measurement.

The fastest measured experimental production result remains 11,377.9 tok/s;
15,000 tok/s is not yet proven.
