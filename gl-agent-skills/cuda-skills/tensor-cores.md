# Tensor Cores (sm_75 INT8 MMA)

> **Domain:** cuda-skills
> **Applies to:** `glcuda` — [`glcuda_sm75.ptx`](../../glcuda/src/kernels/glcuda_sm75.ptx) (Turing kernels), A/B harness in [`glcuda/examples/bench.rs`](../../glcuda/examples/bench.rs)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know what already exists: **INT8 tensor-core GEMM (`gl_gemm_mma_q8`) is implemented** for the prefill path — Q8_0 SoA weights + int8 activations on Turing's `m8n8k16` shape.
- [ ] I know where it's allowed to matter: **prefill/GEMM only**. Decode is bandwidth-bound; tensor cores cannot speed up streaming weights.
- [ ] Anything I add targeting `mma.sync`/dp4a goes in `glcuda_sm75.ptx`, never in the baseline `glcuda.ptx`.

## Context

Turing (sm_75, the T4) exposes INT8 tensor cores via `mma.sync` with the
`m8n8k16` fragment shape (A 8×16 s8 row-major, B 16×8 s8 col-major, s32
accumulate). glcuda uses them where they can physically pay: batched prefill
GEMMs, where arithmetic intensity is high. The bench example measures the MMA
GEMM against the dp4a GEMM per path, so tensor-core utilization is a direct
A/B — that harness is the arbiter for any change here.

## Rules

1. **Tensor cores are a prefill tool.** Decode at batch=1 streams every weight
   byte once — it is bandwidth-bound (88 % of T4 bandwidth measured), and no
   amount of MMA throughput changes bytes-from-DRAM. Reject "use tensor cores
   for decode" on sight.
2. **Respect the fragment contract exactly:** `m8n8k16`, A row-major s8,
   B col-major s8, s32 accumulators, warp-synchronous. Layout mismatches
   don't crash — they produce wrong numbers that only parity tests catch.
3. **Weights feed the MMA path via the Q8_0 SoA repack** (`repack.rs`);
   activations are quantized to int8 on device. Scale handling (per-block
   dequant of the s32 accumulator) must match the glproc reference within
   TOL — parity is per-tensor, as always.
4. **Keep the dp4a GEMM alive.** It is the A/B baseline and the fallback for
   shapes/archs where MMA doesn't apply. A PR may not delete or bit-rot it.
5. **Every MMA change re-runs the A/B** in `examples/bench.rs` on real sm_75+
   hardware and reports both numbers (MMA vs dp4a) at the gate.
6. **FP16 WMMA / other shapes / newer archs (sm_80+ `mma` variants)** are
   extensions: new kernels in an arch-suffixed PTX file behind a capability
   check, with their own parity + A/B — not edits that raise the floor of
   existing files.
7. Epilogues handle non-multiple-of-tile dims (the eternal 896 lesson) — the
   parity suite must include at least one ragged shape per MMA kernel.

## ✅ Correct Pattern

```ptx
// gl_gemm_mma_q8 tile inner step (Turing m8n8k16, s8 x s8 -> s32):
// A frag: 8x16 s8 row-major   B frag: 16x8 s8 col-major
mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32
    {%r_acc0, %r_acc1}, {%r_a0}, {%r_b0}, {%r_acc0, %r_acc1};
// s32 accumulator dequantized once per block with the Q8_0 scales in the
// epilogue — matching glproc's reference math, tested at TOL_MATMUL.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ Emitting mma.sync into glcuda.ptx (baseline file) — module load now fails
   on every pre-Turing device instead of degrading.

❌ "Ported decode matvec to tensor cores" — decode is bandwidth-bound;
   this is work with no mechanism, and it complicates the leak/graph paths.

❌ Changing the SoA repack layout without updating BOTH the MMA kernel's
   fragment loads and the dp4a kernel — the two must read the same layout,
   or the A/B compares different math.
```

## GwenLand-Specific Notes

- The A/B harness note in `examples/bench.rs` (§2f) is the source of truth
  for how MMA vs dp4a is compared — extend it rather than writing a separate
  ad-hoc probe (probes mislead; see
  [`../bench-skills/measurement-discipline.md`](../bench-skills/measurement-discipline.md)).
- sm_75 kernels are selected at runtime by device capability query; a T4
  gets `glcuda_sm75.ptx` in addition to the baseline module. Selection logic
  lives on the Rust side — keep PTX files self-contained per arch tier.

## Related Skills

- [ptx-writing.md](ptx-writing.md)
- [kernel-design.md](kernel-design.md)
- [../gguf-skills/quantization-types.md](../gguf-skills/quantization-types.md)
