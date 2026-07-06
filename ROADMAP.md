# GwenLand AI — Engine Roadmap (Revised)

## Milestone M1 — CPU Baseline

**Goal:** Deliver a fully working CPU inference engine (`glproc`).

### Issues

- **GWEN-??? — glcore Workspace Scaffold**
  - Setup Cargo workspace monorepo: `glcore`, `glproc`, `glcuda`, `glvulkan`, `glmetal`, `glcli`, `gltui`
  - Define shared tensor types in `glcore`
  - Done: `cargo build` succeeds across all crates

- **GWEN-??? — Define Engine Trait**
  - Define common engine interface in `glcore`:
    ```rust
    fn init()
    fn load_model(path: &str)
    fn infer(input: Input) -> Output
    fn stream(input: Input) -> Stream
    fn shutdown()
    fn capabilities() -> EngineSpec
    ```
  - Done: `glproc` compiles against the trait

- **GWEN-??? — GGUF Parser (from scratch)**
  - Pure Rust GGUF v1/v2/v3 parser in `glcore/src/format/gguf.rs`
  - mmap-based loading, zero-copy tensor access
  - Done: can load a GGUF model and read tensor metadata

- **GWEN-??? — Safetensors Parser (from scratch)**
  - Pure Rust safetensors parser in `glcore/src/format/safetensors.rs`
  - JSON header parse + mmap tensor data
  - Done: can load a safetensors file and read tensors

- **GWEN-??? — Tokenizer (BPE, from scratch)**
  - BPE tokenizer in `glcore/src/tokenizer.rs`
  - Load vocab from GGUF or standalone tokenizer.json
  - Done: encode/decode round-trip matches reference tokenizer output

- **GWEN-??? — glproc Skeleton**
  - Create `glproc/` folder structure: `loader.rs`, `matmul.rs`, `attention.rs`, `kv_cache.rs`, `sampler.rs`, `runtime.rs`
  - Register glproc against engine trait
  - Done: glproc implements all trait methods (stubs ok)

- **GWEN-??? — Implement Scalar MatMul**
  - Baseline scalar matmul in `glproc/src/matmul.rs`
  - No SIMD, no parallelism — correctness first
  - Done: output matches numpy reference within float tolerance

- **GWEN-??? — Implement Minimal Attention**
  - Scaled dot-product attention in `glproc/src/attention.rs`
  - Support causal mask
  - Done: output matches reference within float tolerance

- **GWEN-??? — Implement KV Cache**
  - KV cache in `glproc/src/kv_cache.rs`
  - Done: sequential generation reuses cached K/V correctly

- **GWEN-??? — Implement Sampler**
  - `glproc/src/sampler.rs`: Greedy, Top-K, Top-P, Temperature
  - Done: greedy output is deterministic and correct

- **GWEN-??? — Runtime Engine Manager**
  - `runtime/src/manager.rs`: select engine, route request, expose unified API
  - Runtime MUST NOT implement compute logic
  - Done: runtime can init glproc and route an infer request

- **GWEN-??? — End-to-End CLI Inference**
  - `glcli`: `gwen run model.gguf --prompt "Hello"`
  - Done: generates coherent text output on CPU

---

## Milestone M2 — GPU Engines

**Goal:** Add hardware acceleration across NVIDIA, AMD/Intel, Apple Silicon.

### Issues

- **GWEN-??? — Implement glcuda**
  - Full glcuda engine: loader, matmul (CUDA kernels), attention, kv_cache, sampler
  - Implements engine trait
  - Done: same model produces parity output vs glproc

- **GWEN-??? — Implement glvulkan**
  - Full glvulkan engine via Vulkan compute shaders
  - Cross-vendor: AMD, Intel, NVIDIA
  - Done: same model produces parity output vs glproc

- **GWEN-??? — Implement glmetal**
  - Full glmetal engine via Metal Performance Shaders
  - Apple Silicon (M1/M2/M3/M4)
  - Done: same model produces parity output vs glproc

- **GWEN-??? — Hardware Detection**
  - Runtime detects available hardware at startup
  - Done: correctly identifies CUDA/Vulkan/Metal/CPU availability

- **GWEN-??? — Runtime Fallback System**
  - Fallback chain: `glcuda → glvulkan → glmetal → glproc`
  - Engines MUST NOT self-fallback — runtime owns this
  - Done: if glcuda fails, runtime silently falls back to next engine

- **GWEN-??? — Unified Engine Selection**
  - Runtime selects best available engine automatically
  - Manual override via CLI flag: `--engine glproc`
  - Done: `gwen run model.gguf` picks the best engine automatically

---

## Milestone M3 — Parity & Stability

**Goal:** Consistent, reliable behavior across all engines.

### Issues

- **GWEN-??? — Output Parity Testing**
  - Test suite comparing all engine outputs vs glproc reference
  - Done: all engines pass parity within defined tolerance

- **GWEN-??? — Numerical Tolerance System**
  - Define and enforce floating point tolerance per op (matmul, attention, sampler)
  - Done: tolerance thresholds documented and enforced in CI

- **GWEN-??? — Benchmark Suite**
  - Measure TPS (tokens/sec), latency (TTFT), memory usage per engine
  - Done: benchmark report generated for each engine on reference hardware

- **GWEN-??? — Crash Isolation**
  - Engine crash must not bring down runtime or other engines
  - Done: glcuda panic → runtime falls back gracefully, no crash

- **GWEN-??? — Engine Switching Stress Test**
  - Run 100+ inference requests with random engine switching
  - Done: zero panics, output consistent

---

## Milestone M4 — Ecosystem

**Goal:** GwenLand AI becomes an extensible, community-ready platform.

### Issues

- **GWEN-??? — Engine Plugin Standard**
  - Finalize engine trait as stable public API
  - Done: external crate can implement the trait and register as engine

- **GWEN-??? — External Engine Support**
  - Runtime can load external engine plugins at runtime
  - Done: third-party engine loaded and passes parity test

- **GWEN-??? — gltui — Terminal UI**
  - Interactive TUI in `gltui/`: model picker, prompt input, streaming output, engine status
  - Done: `gwen tui` launches interactive session

- **GWEN-??? — Documentation Freeze**
  - Full API docs, architecture guide, PLANNING.md finalized
  - Done: docs published at gwenland.dev

- **GWEN-??? — Contributor Onboarding Guide**
  - How to add a new engine, how to run tests, how to benchmark
  - Done: new contributor can add a stub engine following the guide

- **GWEN-??? — Packaging & Distribution**
  - Binary releases for Windows/macOS/Linux via GitHub Actions
  - Done: `gwen` binary downloadable from gwenland.dev

---

# Exit Criteria

## M1
CPU inference fully operational — `gwen run model.gguf` generates text.

## M2
Same model runs on CPU and GPU engines with parity output.

## M3
Consistent multi-engine behavior, benchmarks published.

## M4
New engines can be added without modifying the runtime. gltui ships.
