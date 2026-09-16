# Wave 127: Wave88 launcher wiring repair gate

## Failure being repaired

Wave 126 stopped at the host compile gate. Reconstruction placed the four
zero-valued fused-epilogue parameters in the generic N16 launcher even though
the corresponding parameter array lives in the isolated Wave88 N16-prefetch
launcher. Rust therefore reported `E0425` for `up_wqs`, `up_wsc`, `y_scales`,
and `fused`.

## Frozen repair

Move that one local tuple into `gemm_mma_q8_bstage_n16_prefetch`, immediately
before its FFI parameter array. No PTX, arithmetic, dispatch, launch geometry,
feature guard, buffer allocation, or benchmark workload changes are permitted.

The retained Wave88 call passes four zeroes to select its original f32 output
path. The fused call supplies its real up-weight and Q8-output pointers.

## Gate

1. `cargo test -p glcuda --lib --locked` returns to 67/67.
2. `cargo check -p glcuda --all-targets --locked` compiles both Wave126
   harnesses and every retained target.
3. `git diff --check` passes and the generic N16 launcher contains no fused
   parameters.

Wave 127 ends at this host-wiring gate. Device resource, parity, and production
measurement remain Wave 128 work and may begin only after this repair is green.
