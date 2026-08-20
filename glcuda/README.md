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

~5,900 lines.

## Status, measured

**Decode is at llama.cpp parity on a T4.** That is a measured result, not a
target.

Prefill was profiled 2026-07-12 and is the open front:

| Bucket | Share of prefill |
|---|---|
| `attn_decode_rows` (attention core) | **39% — the single biggest bucket** |
| FFN GEMMs (combined) | 49% |

`architecture/ArchGLML_X2.md` is the ground truth for this crate.

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
