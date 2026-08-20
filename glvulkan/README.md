# glvulkan

## ⛔ This is a stub. It does not run inference.

**59 lines.** It implements `glcore::GlEngine` and reports
`capabilities().available == false`. That is all it does.

```rust
// the whole of what this crate promises today
assert!(!engine.capabilities().available);
```

There is no Vulkan backend. There are no kernels, no device memory management,
no shader/pipeline code. If you are looking for an engine that actually
computes something, use:

| Crate | Hardware | Status |
|---|---|---|
| [`glproc`](../glproc) | CPU (AVX2 / AVX-512 / VNNI) | real, and the numerical oracle |
| [`glcuda`](../glcuda) | NVIDIA | real, decode at llama.cpp parity on a T4 |

## Why it exists at all

The engine `match` in `glbench::engine::adapter::build_engine` is the only
place in the workspace that names concrete engine types. Keeping a compiling
stub here means adding a real Vulkan backend later is **one match arm**, with the
trait boundary already proven to hold — rather than a refactor that touches the
adapter, the runtime, and every caller at once.

It also keeps the workspace honest: `available: false` is a measured fact the
adapter reports, not a silent absence a caller has to guess at.

## What "done" would look like

Anything claiming to replace this stub needs, at minimum:

- kernels for the ops `glproc` implements (qdot, dequant, attention, RoPE, RMSNorm)
- an entry in `glcuda`-style parity tests, validating **tensor-by-tensor
  against `glproc`** within explicit per-operation tolerances
- `SAFETY:` comments on every FFI block, per `gl-agent-skills/rust-skills/unsafe-rules.md`

Until then, this README is the accurate description of the crate.

## Build

```bash
cargo test -p glvulkan   # 1 test: stub_reports_unavailable
```
