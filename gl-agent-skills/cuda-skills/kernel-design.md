# CUDA Kernel Design

> **Domain:** cuda-skills
> **Applies to:** `glcuda/src/kernels/` (PTX + Rust launch code)
> **Last updated:** 2026-07-17

## BEFORE YOU START

- [ ] I know which phase my kernel serves: **decode** (bandwidth-bound, batch=1) or **prefill** (GEMM/compute-bound) — they have different design rules.
- [ ] I checked the measured profile before "optimizing": on the T4, `attn_decode_rows` was the single biggest prefill bucket (~39 %), FFN GEMMs ~49 % combined.
- [ ] Launch geometry is written down in the kernel header (see [ptx-writing.md](ptx-writing.md)).

## Context

Decode on glcuda already runs at **88 % of the T4's memory bandwidth** — i.e.
within ~12 % of the physical ceiling for weight-streaming decode. That number
disciplines all kernel work: decode kernels win by moving fewer bytes (layout,
quantization, fusion), never by "more occupancy". Prefill is the opposite —
batched GEMM with real arithmetic intensity — and is where classic
occupancy/tiling thinking applies.

## Rules

1. **Classify before designing.** Decode kernel → optimize bytes moved per
   token (coalescing, SoA layout, quantized weights, fusion). Prefill kernel →
   optimize math throughput (tiles, tensor cores, shared-memory reuse).
2. **Block sizing:** start from multiples of the warp (32); typical glcuda
   kernels use 128–256 threads/block. Deviations get a comment with the
   reason (register pressure, shared-mem tile shape).
3. **Occupancy is a means, not a score.** Chasing 100 % occupancy on a
   bandwidth-bound kernel is cargo cult — measure tok/s, not occupancy. A
   kernel already at the bandwidth ceiling cannot be "occupancy'd" faster.
4. **One kernel, one job — until the profiler says fuse.** Fusion is the main
   lever for reducing inter-kernel dependency stalls (see
   [cuda-graphs.md](cuda-graphs.md)), but fused kernels are validated against
   the *unfused* parity baseline before the unfused path is touched.
5. **Reductions use warp shuffles** (`shfl.sync`) before shared memory, and
   shared memory before global atomics. Global atomics in the token path need
   explicit justification.
6. **Shapes are runtime inputs, not constants.** Real models bring awkward
   dims (896, GQA head ratios); kernels must handle non-multiple-of-tile
   dimensions with an epilogue, and the parity suite must include such a
   shape.
7. **Every new kernel ships with:** parity test vs `glproc` at spec tolerance,
   an entry in the bench example so its cost is visible, and a header
   documenting launch geometry.
8. **Measure on production decode/prefill**, not standalone kernel probes —
   probe results in this project have ranged 0.07×–2.40× vs reality.

## ✅ Correct Pattern

```text
Plan for a new fused RMSNorm+QKV kernel:
1. Baseline: bench current unfused pair on T4 (production path, not probe).
2. Write fused PTX; header documents geometry (grid = rows, block = 256).
3. Parity: fused output vs glproc within TOL_NORM/TOL_MATMUL, incl. dim 896.
4. A/B in glcuda/examples/bench.rs; keep only if production tok/s improves.
5. Gate report with both numbers.
```

## ❌ Anti-Pattern (Never Do This)

```text
❌ "Increased block size to 1024 for better occupancy" on a decode kernel
   that is bandwidth-bound — no mechanism for a win, adds register pressure.

❌ Tile size hardcoded to 64 with no epilogue: silently corrupt for dim 896
   (real Qwen2.5-0.5B shape) — exactly what parity shapes exist to catch.

❌ Trusting a standalone microbench of one kernel to justify a change on the
   full decode path.
```

## GwenLand-Specific Notes

- Reference measured numbers (T4, Qwen2.5-7B-Q8_0): decode 29.2 tok/s @ 88 %
  bandwidth; prefill 73 tok/s via batched GEMM. A kernel PR claiming a win
  states its numbers against these baselines, same hardware class.
- Per-arch instructions split by file: baseline kernels in `glcuda.ptx`,
  `sm_75+` (dp4a / `mma.sync`) in `glcuda_sm75.ptx` behind a runtime
  capability check — never emit sm_75 instructions into the baseline file.
- Anything measured on a different GPU class (laptop GPU, A100) is a
  different regime — label it, don't extrapolate.

## Related Skills

- [ptx-writing.md](ptx-writing.md)
- [tensor-cores.md](tensor-cores.md)
- [../bench-skills/measurement-discipline.md](../bench-skills/measurement-discipline.md)
