# glcuda

**The CUDA inference engine.** Same contract as `glproc`, different silicon.

## What level does it work at?

**Kernel and device-memory level.** Like `glproc`, glcuda thinks in blocks and
lanes rather than tokens — but the lanes are CUDA threads and the memory is
VRAM.

| Level | Example |
|---|---|
| **Kernel** | fused attention, quantised matmul, RMSNorm, RoPE |
| **Device memory** | `RSDeviceBuffer`-style ownership, VRAM lifetime, H2D/D2H staging |
| **Driver FFI** | `driver.rs` — raw `cudarc`-free handles, loaded at runtime |
| **Layer** | the same forward pass `glproc` runs, on device |

## Status, measured

The retained Tesla T4 production stack reaches **11,252.8 prefill tok/s** on
the pinned 244-token Qwen2.5-0.5B workload. That is +1,106.6 tok/s (+10.91%)
over its same-session pre-N16 baseline. The result passed 33/33 CUDA parity
tests and a 50/50 exact token oracle in every measured session.

The production configuration is explicit so benchmarks can still run clean
A/B arms:

```text
GLCUDA_FORCE_Q8=1
GLCUDA_GRID2D=1
GLCUDA_FUSE_Q8_GLUE=1
GLCUDA_NTILE128=1
GLCUDA_BSTAGE=1
GLCUDA_GEMM_N16=1
GLCUDA_ATTN_MMA4=1
GLCUDA_ATTN_MMA4_REGQ=1
```

Wave 78's compensated-MMA AV production candidate is deliberately opt-in on
top of that retained stack:

```text
GLCUDA_ATTN_MMA4_AV=1
```

It is not a retained/default path until T4 device parity and production
`glbench` A/B both pass.

See the [Wave 50 production report](../architecture/glcuda-research/wave50-register-q-production-result.md)
for the controlled protocol and raw evidence. A longer Wave 56 confirmation
ran ten near-position-balanced repetitions (30 production sessions): the
baseline median was 9,997.3 tok/s and the retained stack reached 11,173.1
tok/s, a **+1,175.8 tok/s (+11.76%)** gain. All ten paired gains exceeded
+1,000 tok/s; the worst was +1,018.8 tok/s, and all 30 sessions passed their
50/50 oracle. See the [ten-repeat report](../architecture/glcuda-research/wave56-ten-repeat-reproducibility.md)
and [raw evidence](../benchmarks/glcuda-t4-wave56-repro10.json).

In the pinned same-T4 Wave 55 head-to-head, GwenLand measured **11,454.9
prefill tok/s** versus llama.cpp's **10,640.4 tok/s**, a 7.65% (+814.5 tok/s)
lead across six position-balanced pairs. The effective ChatML prompt was
244/244 token-ID exact and every GwenLand session passed its 50/50 oracle. See the
[Wave 55 comparison](../architecture/glcuda-research/wave55-llamacpp-head-to-head.md)
and [compact raw evidence](../benchmarks/glcuda-t4-wave55-h2h.json). llama.cpp
uses its native Q4_K path while GwenLand repacks to Q8_0, so this is a matched
model, prompt shape, GPU, and workload comparison rather than identical
quantized bytes. `architecture/ArchGLML_X2.md` remains the ground truth for
this crate.

Wave 57's same-file Q8_0 comparison observed **11,499.7 vs 11,453.5 tok/s**
(+0.40%, effectively tied; four of six pairs positive). Its correctness gate
is unresolved: all six CUDA sessions matched only the first 29/50 oracle
tokens. The rerun relaxed that gate to first-token agreement after the first
failure; this is a protocol deviation, not a fix. See the
[qualified Wave 57 report](../architecture/glcuda-research/wave57-q8-llamacpp-head-to-head.md)
and [raw observations](../benchmarks/glcuda-t4-wave57-q8-h2h.json).

## Dependencies

**Zero direct.** Fifteen crates in the tree, all inherited from `glcore`.

The CUDA driver is loaded at **runtime**, never linked. That is why depending on
glcuda on a CPU-only machine is harmless: it self-probes at `init()` and reports
`capabilities().available == false`, and the adapter simply treats the engine as
unavailable.

## `unsafe`

Present and justified: FFI to the CUDA driver — calls, raw handles, and
`Send`/`Sync` impls for wrapper types. Every one carries a `// SAFETY:` comment
explaining the driver's actual threading contract.

## Testing on a machine without a GPU

Host-side tests run anywhere:

```bash
cargo test -p glcuda --lib
```

Parity tests need real hardware **and** serial execution:

```bash
cargo test -p glcuda --test parity -- --test-threads=1
```

`--test-threads=1` is mandatory there — the VRAM-leak check is perturbed by
concurrent allocations.

⚠️ **A green run on a GPU-less box proves nothing about kernel correctness.**
GPU tests print `SKIP: no CUDA device` and pass. That is deliberate (they must
never fail on a laptop), but do not read it as validation. State which machine
your green run came from.
