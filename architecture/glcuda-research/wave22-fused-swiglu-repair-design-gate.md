# glcuda Wave 22 - fused SwiGLU PTX repair design gate

## Frozen question

Does the Wave 21 fused-SwiGLU candidate clear its existing compile, resource,
exact-parity and production gates after repairing only the illegal direct use
of `%ctaid.x` as a `shl` operand?

Wave 21 failed before producing a cubin:

```text
glcuda_wave21_sm75.ptx, line 83; error: Special register argument not allowed
for instruction 'shl'
```

Wave 22 changes that instruction sequence from:

```ptx
shl.b32 %r11, %ctaid.x, 6;
```

to:

```ptx
mov.u32 %r11, %ctaid.x;
shl.b32 %r11, %r11, 6;
```

The move reuses the destination register already declared and live at that
point. It adds no named register and changes no computed value.

## Frozen causal scope

The Wave 21 kernel image, launch geometry, shared-memory layout, MMA order,
dequantisation, SiLU sequence, Q8 quantisation, output layout, fallback guards,
dispatch logging and production workload are otherwise byte-identical.

Both production arms retain Wave 20 fused MMA attention. They differ only by
`GLCUDA_FUSED_SWIGLU=1`.

## Gates, in order

1. Reconstruct the SHA-verified Wave 3 through Wave 21 stack and apply the
   incremental Wave 22 repair.
2. Host-check the diagnostic and pass all 64 `glcuda` library tests. A
   structural regression assertion forbids the old illegal operand form.
3. Assemble the retained and candidate sm_75 modules with `ptxas -v`. Every
   production entry must have zero stack, zero spill stores and zero spill
   loads. Any compiler rejection terminates the wave.
4. Ask the CUDA driver for the candidate's occupancy at the actual 512-thread,
   zero-dynamic-shared launch. At least one active block per SM is required.
5. Require bit-exact Q8 bytes and f32 scale bits versus the retained two-GEMM
   plus glue path at full-64, ragged-17 and production-244 shapes.
6. Pass the complete CUDA parity suite serially.
7. Run four position-balanced production pairs in orders A/B, B/A, B/A, A/B,
   with 5 cold, 5 warmup and 10 measured iterations per arm. Every session
   must match the glproc token oracle exactly, 50/50.
8. Retain only if median production prefill improves by at least 5%, every
   pair is positive, every paired decode delta is at least -5%, and the median
   session-max latency regression is at most +5%.

A retention result requires a second run of byte-identical notebook content.
A compile, resource, parity, oracle or hard regression failure stops the wave;
it is not repaired in place.
