# ARTX1 — gljax Research Report: PJRT + StableHLO Rust Implementation

**Date**: July 2026  
**Author**: Research brief for GwenLand / gwenland-ml  
**Scope**: All technical groundwork needed to implement `gljax` — a pure Rust XLA/PJRT client for ML inference on TPU v5e, A100, H100, and CPU.

---

## Table of Contents

1. [PJRT C API — FFI from Rust](#1-pjrt-c-api--ffi-from-rust)
2. [StableHLO IR Emission from Rust](#2-stablehlo-ir-emission-from-rust)
3. [Precision Handling](#3-precision-handling)
4. [Multi-Device Sharding](#4-multi-device-sharding)
5. [Plugin Ecosystem + Build Strategy](#5-plugin-ecosystem--build-strategy)
6. [Compiled Artifact Persistence](#6-compiled-artifact-persistence)
7. [LLM-Specific Ops — Implementation Guide](#7-llm-specific-ops--implementation-guide)
8. [Prior Art + Lessons Learned](#8-prior-art--lessons-learned)
9. [Risks + Open Questions](#9-risks--open-questions)
10. [Prioritized Reading List](#10-prioritized-reading-list)

---

## 1. PJRT C API — FFI from Rust

### 1.1 Header Location

Canonical header: `xla/pjrt/c/pjrt_c_api.h`  
Repo: https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h  
Companion docs: `xla/pjrt/c/docs/pjrt_integration_guide.md`  
Changelog: `xla/pjrt/c/CHANGELOG.md` — read this before pinning a plugin version.

### 1.2 Key Structs

| C Type | Role |
|---|---|
| `PJRT_Api` | The vtable. Obtained by calling `GetPjrtApi()` from the loaded plugin .so |
| `PJRT_Client` | Owns device list, compile queue, and buffer allocator |
| `PJRT_Device` | A single hardware device (one GPU, one TPU chip) |
| `PJRT_Buffer` | A device-resident tensor; holds memory on-device |
| `PJRT_LoadedExecutable` | Compiled + loaded program ready for execution |
| `PJRT_Executable` | The underlying compiled artifact (serializable separately) |
| `PJRT_Program` | Input to compile — holds StableHLO MLIR text or HLO proto bytes |
| `PJRT_Error` | Owned error object returned by any failing API call |
| `PJRT_Event` | Async completion handle for execute + transfer operations |

All structs are **opaque from the caller's side**. You never allocate them directly.
Every API call takes a single `*_Args` struct. You fill it, call the function pointer from `PJRT_Api`, and check the returned `PJRT_Error*`.

```c
// Representative call pattern
PJRT_Client_Create_Args create_args;
memset(&create_args, 0, sizeof(create_args));
create_args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
PJRT_Error* err = api->PJRT_Client_Create(&create_args);
// create_args.client is now valid if err == NULL
```

### 1.3 Dynamic Plugin Loading

Never link against a PJRT plugin at compile time. Use `dlopen` (Linux/macOS) or `LoadLibrary` (Windows) to load the plugin .so at runtime, find `GetPjrtApi`, and call it to get the vtable.

```rust
use libloading::{Library, Symbol};

type GetPjrtApiFn = unsafe extern "C" fn() -> *const PJRT_Api;

pub struct PjrtPlugin {
    _lib: Library,           // Keep alive — dropping unloads the .so
    pub api: *const PJRT_Api,
}

// Safety: PJRT_Api is a vtable of C function pointers.
// Safe to share across threads after construction.
unsafe impl Send for PjrtPlugin {}
unsafe impl Sync for PjrtPlugin {}

pub fn load_plugin(path: &std::path::Path) -> Result<PjrtPlugin, Error> {
    let lib = unsafe { Library::new(path) }
        .map_err(|e| Error::PluginLoad(e.to_string()))?;
    let get_api: Symbol<GetPjrtApiFn> = unsafe { lib.get(b"GetPjrtApi\0") }
        .map_err(|_| Error::SymbolNotFound("GetPjrtApi"))?;
    let api = unsafe { get_api() };
    if api.is_null() {
        return Err(Error::NullApi);
    }
    Ok(PjrtPlugin { _lib: lib, api })
}
```

Store the `PjrtPlugin` (and thus `Library`) in your top-level `Session` struct. It must live for the entire program lifetime.


### 1.4 Plugin Sources and Versions

**zml/pjrt-artifacts** (https://github.com/zml/pjrt-artifacts) distributes prebuilt plugins:

| Plugin | Platform | Approx Size | Notes |
|---|---|---|---|
| `pjrt_c_api_cpu_plugin.so` | Linux, macOS, Windows | ~80MB | Always use for CI |
| `pjrt_c_api_cuda_plugin.so` | Linux x86_64 | ~1.8GB | Requires CUDA 12 runtime |
| `pjrt_c_api_rocm_plugin.so` | Linux x86_64 | ~1.5GB | AMD GPU, ROCm 6.x |
| `libtpu.so` | Linux x86_64 | ~200MB | TPU v5e/v5p |

Tags are in the form `v0.X.Y` and each release message documents the exact openxla/xla commit SHA. The version that matters for ABI compatibility is `PJRT_Api::pjrt_api_version.{major,minor}`. As of mid-2026, XLA HEAD is approximately API minor version 58. ZML's pinned artifacts track JAX 0.4.x releases.

**Recommendation**: Pin a specific `pjrt-artifacts` tag. Do not auto-upgrade plugins. Treat them like a compiler dependency.

### 1.5 Struct Size Versioning — ABI Safety

PJRT uses a Vulkan-derived struct-size versioning pattern. Every `*_Args` struct has:

```c
size_t struct_size;  // MUST be set to PJRT_STRUCT_SIZE(Type, last_field)
void*  priv;         // MUST be NULL unless told otherwise
```

New fields are appended at the end when the API evolves. The library checks: if caller `struct_size < library's minimum`, the call returns an error. If caller `struct_size > library's`, the library ignores the extra bytes.

In Rust:
1. `bindgen` generates `PJRT_*_STRUCT_SIZE` constants from the macro expansion.
2. Always zero-initialize every args struct with `std::mem::zeroed()` or `Default::default()`.
3. Always set `struct_size` before any other field.
4. At startup, check `(*api).pjrt_api_version.major_version`. If it differs from what you bound against at compile time, refuse to proceed.

```rust
unsafe fn make_client_create_args() -> PJRT_Client_Create_Args {
    let mut args = std::mem::zeroed::<PJRT_Client_Create_Args>();
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    args
}
```

### 1.6 Error Handling

`PJRT_Error*` is an **owned** pointer. Non-null means an error occurred. You **must** call `PJRT_Error_Destroy` or you leak memory. Extract the message before destroying:

```rust
pub fn check(api: *const PJRT_Api, err: *mut PJRT_Error) -> Result<(), PjrtError> {
    if err.is_null() { return Ok(()); }
    unsafe {
        let mut msg_args = std::mem::zeroed::<PJRT_Error_Message_Args>();
        msg_args.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
        msg_args.error = err;
        ((*api).PJRT_Error_Message.unwrap())(&mut msg_args);
        let s = std::slice::from_raw_parts(
            msg_args.message as *const u8, msg_args.message_size
        );
        let msg = std::str::from_utf8_unchecked(s).to_owned();

        let mut destroy_args = std::mem::zeroed::<PJRT_Error_Destroy_Args>();
        destroy_args.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroy_args.error = err;
        ((*api).PJRT_Error_Destroy.unwrap())(&mut destroy_args);

        Err(PjrtError(msg))
    }
}
```

Wrap every single API call with this. Never propagate a `PJRT_Error*` up the call stack — always convert at the call site.

### 1.7 Thread Safety

`PJRT_Client` is **thread-safe** for concurrent compile and execute calls. The internal device queues handle concurrent dispatch. `PJRT_Buffer` objects are **not** thread-safe — a buffer may only be used by one thread at a time unless you explicitly synchronize via `PJRT_Event_Await`. Pattern: create one client at startup, hold it in an `Arc`, give buffers to individual worker threads that await their events before returning.


---

## 2. StableHLO IR Emission from Rust

### 2.1 Minimal Valid MLIR Module

```mlir
module @my_model {
  func.func @main(
      %arg0: tensor<2x4xf32>,
      %arg1: tensor<4x8xf32>
  ) -> tensor<2x8xf32> {
    %0 = stablehlo.dot_general %arg0, %arg1,
        contracting_dims = [1] x [0]
        : (tensor<2x4xf32>, tensor<4x8xf32>) -> tensor<2x8xf32>
    return %0 : tensor<2x8xf32>
  }
}
```

Rules:
- One `module` block; name is arbitrary but aids debugging.
- Entry point is `func.func @main` by default (PJRT looks for `main`).
- SSA values are `%name` — every op result needs a unique name in its block.
- Tensor types: `tensor<D0 x D1 x ... x dtype>`.
  - dtypes: `f32`, `f64`, `bf16`, `f16`, `i32`, `i64`, `i8`, `i1`.
  - Dynamic dims: `tensor<?x512xf32>`.
- All ops are `stablehlo.*` prefixed. Do not mix `mhlo.*` — it is unstable.

### 2.2 Complete Op Syntax for gljax

**`stablehlo.dot_general`** — the workhorse for all matmuls and attention QKV:
```mlir
%result = stablehlo.dot_general %lhs, %rhs,
    batching_dims = [0, 1] x [0, 1],
    contracting_dims = [3] x [2]
    : (tensor<B x H x S x D x f32>, tensor<B x H x D x S x f32>)
      -> tensor<B x H x S x S x f32>
```

**`stablehlo.add`, `stablehlo.multiply`** — elementwise, shapes must match exactly:
```mlir
%r = stablehlo.add %a, %b : tensor<B x D x f32>
%r = stablehlo.multiply %a, %b : tensor<B x D x f32>
```

**`stablehlo.broadcast_in_dim`** — expand into a larger tensor:
```mlir
// %input: tensor<2x4xf32> -> tensor<2x8x4xf32>
// dims = [0, 2] means: input dim 0 -> output dim 0, input dim 1 -> output dim 2
%result = stablehlo.broadcast_in_dim %input, dims = [0, 2]
    : (tensor<2x4xf32>) -> tensor<2x8x4xf32>
```

**`stablehlo.reduce`** — general reduction (used for softmax, RMSNorm):
```mlir
%sum = stablehlo.reduce(%input init: %zero) across dimensions = [2]
    : (tensor<B x H x S x f32>, tensor<f32>) -> tensor<B x H x S x f32> {
  ^bb0(%a: f32, %b: f32):
    %r = stablehlo.add %a, %b : f32
    stablehlo.return %r : f32
}
```

**`stablehlo.reduce_window`** — windowed reduction (rarely needed for LLMs directly):
```mlir
%result = stablehlo.reduce_window(%input init: %zero)
    window_dimensions = [1, 3]
    window_strides = [1, 1]
    : (tensor<4x6xf32>, tensor<f32>) -> tensor<4x4xf32> {
  ^bb0(%a: f32, %b: f32):
    %r = stablehlo.maximum %a, %b : f32
    stablehlo.return %r : f32
}
```

**`stablehlo.gather`** — embedding lookup and KV read:
```mlir
%result = stablehlo.gather %operand, %start_indices,
    dimension_numbers = <
      offset_dims = [1],
      collapsed_slice_dims = [0],
      start_index_map = [0],
      index_vector_dim = 1
    >,
    slice_sizes = [1, 128]
    : (tensor<32000 x 128 x f32>, tensor<B x 1 x i32>)
      -> tensor<B x 128 x f32>
```

**`stablehlo.scatter`** — KV cache write:
```mlir
%result = stablehlo.scatter(%base, %scatter_indices, %updates)
    scatter_dimension_numbers = <
      update_window_dims = [1, 2],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [0],
      index_vector_dim = 1
    >,
    unique_indices = false
    : (tensor<S x H x D x f32>, tensor<B x 1 x i32>, tensor<B x H x D x f32>)
      -> tensor<S x H x D x f32> {
  ^bb0(%old: f32, %new: f32):
    stablehlo.return %new : f32
}
```

**`stablehlo.custom_call`** — for XLA-registered fused kernels:
```mlir
%result = stablehlo.custom_call @rms_norm(%input, %weight)
    {backend_config = "{\"epsilon\":1e-6}"}
    : (tensor<B x D x f32>, tensor<D x f32>) -> tensor<B x D x f32>
```

**`stablehlo.while`** — autoregressive generation loop:
```mlir
%out:2 = stablehlo.while(%iter0 = %tokens, %iter1 = %pos)
    : tensor<B x S x i32>, tensor<i32> {
  // Condition region
  ^bb0(%t: tensor<B x S x i32>, %p: tensor<i32>):
    %cond = stablehlo.compare LT, %p, %max_pos
        : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.return %cond : tensor<i1>
} do {
  // Body region
  ^bb0(%t: tensor<B x S x i32>, %p: tensor<i32>):
    // ... forward pass producing %new_tokens, %new_pos ...
    stablehlo.return %new_tokens, %new_pos : tensor<B x S x i32>, tensor<i32>
}
```

**`stablehlo.transpose`**, **`stablehlo.reshape`**, **`stablehlo.slice`**, **`stablehlo.concatenate`**:
```mlir
%t = stablehlo.transpose %x, dims = [0, 2, 1]
    : (tensor<2x4x8xf32>) -> tensor<2x8x4xf32>

%r = stablehlo.reshape %x
    : (tensor<16xf32>) -> tensor<2x8xf32>

// slice syntax: [start:limit:stride] per dimension
%s = stablehlo.slice %x [0:1:1, 2:6:1]
    : (tensor<2x8xf32>) -> tensor<1x4xf32>

%c = stablehlo.concatenate %a, %b, dim = 1
    : (tensor<2x4xf32>, tensor<2x4xf32>) -> tensor<2x8xf32>
```

**`stablehlo.convert`** — explicit dtype cast, the only way to change precision:
```mlir
%bf16 = stablehlo.convert %f32_val
    : (tensor<B x D x f32>) -> tensor<B x D x bf16>
```


### 2.3 Text Emitter in Rust

Build a simple string-builder IR. No MLIR C API needed for text emission.

```rust
pub struct MlirEmitter {
    buf: String,
    indent: usize,
    ssa_counter: usize,
}

impl MlirEmitter {
    pub fn fresh(&mut self) -> String {
        let n = self.ssa_counter;
        self.ssa_counter += 1;
        format!("%v{n}")
    }

    pub fn emit_line(&mut self, line: impl std::fmt::Display) {
        for _ in 0..self.indent { self.buf.push_str("  "); }
        self.buf.push_str(&line.to_string());
        self.buf.push('\n');
    }

    pub fn emit_add(&mut self, lhs: &str, rhs: &str, ty: &str) -> String {
        let out = self.fresh();
        self.emit_line(format!("{out} = stablehlo.add {lhs}, {rhs} : {ty}"));
        out
    }

    pub fn emit_dot_general(
        &mut self, lhs: &str, rhs: &str,
        batch_lhs: &[usize], batch_rhs: &[usize],
        contract_lhs: &[usize], contract_rhs: &[usize],
        input_ty: (&str, &str), output_ty: &str,
    ) -> String {
        let out = self.fresh();
        let bd = format_dims(batch_lhs, batch_rhs);
        let cd = format_dims(contract_lhs, contract_rhs);
        self.emit_line(format!(
            "{out} = stablehlo.dot_general {lhs}, {rhs},\n\
             {ind}  batching_dims = {bd},\n\
             {ind}  contracting_dims = {cd}\n\
             {ind}  : ({}, {}) -> {output_ty}",
            input_ty.0, input_ty.1,
            ind = "    ".repeat(self.indent),
        ));
        out
    }
    
    pub fn finish(self) -> String { self.buf }
}
```

This pattern is exactly what fusebox uses. It's straightforward to debug: `println!("{mlir_text}")` and paste into `stablehlo-opt` for validation.

### 2.4 Validation Before Passing to PJRT

Two options:

1. **Offline: `stablehlo-opt --verify-diagnostics input.mlir`**  
   Ships with LLVM/MLIR builds. Use as a subprocess in tests and CI. Not available on the hot path.

2. **Runtime: PJRT compile call**  
   `PJRT_Client_Compile` rejects invalid programs with a descriptive `PJRT_Error` containing the MLIR verifier output. This is the de-facto validation mechanism in production. On error, log the full MLIR text (gated behind a `GLJAX_DUMP_MLIR=1` env var) for debugging.

There is no stable PJRT C API for "validate only." Accept compile-time errors as your validation signal and ensure your MLIR dump path is always reachable.

### 2.5 StableHLO Versioning

As of mid-2026:
- **5-year backward compatibility**: portable artifacts from today are deserializable by any libStableHLO released before mid-2031.
- **2-year forward compatibility**: new libStableHLO can consume artifacts from older versions within 2 years.
- Compatibility applies to the **MLIR bytecode** serialization format (via `stablehlo.serialize`), NOT to the text format.
- Text format is for human readability and PJRT compilation input. It is stable in practice but not guaranteed.
- For PJRT compilation input: text is fine (it is ephemeral — compiled and discarded).
- For stored artifacts: use bytecode (see Section 6).

**Recommendation**: Emit text for the PJRT compile path. If you ever need to store pre-compiled StableHLO for replay/debugging, serialize to bytecode via the StableHLO C API or by calling `stablehlo-translate --serialize`.

---

## 3. Precision Handling

### 3.1 FP64 Backend Support Matrix

| Backend | FP64 Native | Notes |
|---|---|---|
| CPU plugin | Full | No restrictions. Ideal for reference/oracle runs. |
| A100 (CUDA) | Full | Native FP64 CUDA cores. ~1/32 of BF16 throughput. |
| H100 (CUDA) | Full | Same ratio. H100 SXM5 has better FP64/FP32 ratio than A100. |
| TPU v5e | **No** | No FP64 hardware. XLA will error or software-emulate (unreliable). |
| TPU v5p | Partial | Better than v5e, but not production-grade FP64. |

**Implication for gljax**: The FP64 oracle use case (cross-checking glproc's FP32 output) only works on CPU and CUDA backends. Gate it behind `#[cfg(feature = "cuda")]` or `#[cfg(feature = "cpu-oracle")]`. Never pass FP64 programs to TPU plugins.

### 3.2 BF16: A100 vs TPU v5e Differences

Both natively support BF16 at high throughput. The critical difference is **accumulation behavior**:

- **A100 Tensor Cores**: BF16 input, **FP32 accumulate** internally before writing back. This is important: dot products are exact to FP32 precision even with BF16 inputs.
- **TPU v5e MXU (Matrix Multiply Unit)**: BF16 input, **BF16 accumulate**. At long sequence lengths, cumulative error in attention score computation is noticeably higher than A100.

**Practical implication for attention**:
- On A100: use BF16 weights and activations throughout. The Tensor Core FP32 accumulation keeps softmax numerically stable.
- On TPU: for very long sequences (>8k tokens), you may need to explicitly cast scores to FP32 before the softmax reduce. Profile accuracy impact before deciding.

### 3.3 Mixed Precision in StableHLO

There is no implicit casting. `stablehlo.convert` is the only mechanism and must be explicit at every dtype boundary:

```mlir
// Upcast-accumulate pattern for attention scores on TPU:
%q_f32  = stablehlo.convert %q_bf16  : (tensor<B x H x S x D x bf16>) -> tensor<B x H x S x D x f32>
%k_f32  = stablehlo.convert %k_bf16  : (tensor<B x H x S x D x bf16>) -> tensor<B x H x S x D x f32>
%scores = stablehlo.dot_general %q_f32, %k_f32, ...
          : (tensor<B x H x S x D x f32>, tensor<B x H x S x D x f32>)
            -> tensor<B x H x S x S x f32>
%probs_bf16 = stablehlo.convert %probs_f32
              : (tensor<B x H x S x S x f32>) -> tensor<B x H x S x S x bf16>
```

This pattern is more expensive on TPU (extra convert ops) but required for numerical correctness at long context. On A100 it is unnecessary because Tensor Core handles FP32 accumulation transparently.

### 3.4 FP64 Oracle Pattern for Validating glproc

Use the CPU plugin. Write a standalone `oracle_forward` function in StableHLO FP64, compile it once at test startup, and run it on the CPU plugin:

```mlir
// oracle.mlir
module @oracle {
  func.func @attention_f64(
      %q: tensor<B x H x S x D x f64>,
      %k: tensor<B x H x S x D x f64>,
      %v: tensor<B x H x S x D x f64>
  ) -> tensor<B x H x S x D x f64> {
    // QK^T
    %scores = stablehlo.dot_general %q, %k,
        batching_dims = [0, 1] x [0, 1],
        contracting_dims = [3] x [3]
        : (tensor<B x H x S x D x f64>, tensor<B x H x S x D x f64>)
          -> tensor<B x H x S x S x f64>
    // Scale by 1/sqrt(D) expressed as a constant
    %scale = stablehlo.constant dense<0.125> : tensor<f64>
    // ... broadcast, multiply, softmax, AV product ...
    return %out : tensor<B x H x S x D x f64>
  }
}
```

In your Rust test harness:
1. Cast glproc's FP32 inputs to FP64 before copying to PJRT CPU buffer.
2. Run oracle_forward.
3. Cast PJRT FP64 output back to FP32.
4. Compare element-wise to glproc's FP32 output. Assert max absolute error < 1e-4.

This confirms glproc's numerical correctness without needing a Python reference.


---

## 4. Multi-Device Sharding

### 4.1 PJRT Multi-Device API

Enumerate devices after client creation:

```rust
// After PJRT_Client_Create:
let mut devices_args = zeroed::<PJRT_Client_Devices_Args>();
devices_args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
devices_args.client = client;
check(api, ((*api).PJRT_Client_Devices.unwrap())(&mut devices_args))?;
// devices_args.devices: *mut *mut PJRT_Device
// devices_args.num_devices: usize
```

To execute across multiple devices, compile once and call `PJRT_LoadedExecutable_Execute` with `num_devices` input/output buffer arrays. The executable distributes work based on sharding annotations embedded during compilation. Without sharding annotations, XLA replicates the full computation to every device (useful for data-parallel batch inference).

For device-specific buffer allocation, call `PJRT_Client_BufferFromHostBuffer` with a specific `device` pointer.

### 4.2 Shardy Dialect — State as of 2026

[Shardy](https://openxla.org/shardy) (SDY) is now the **production sharding system** for XLA/JAX, having replaced GSPMD.

Key facts:
- Born from the merger of Google's GSPMD and DeepMind's PartIR teams.
- Integrated into XLA and used by JAX by default since late 2024.
- Has its own MLIR dialect (`sdy.*`) that annotates StableHLO programs.
- Dialect-agnostic — defines `ShardingRuleOpInterface` for any MLIR dialect.
- Shardy handles sharding **propagation** (fills in un-annotated ops) and **SPMD partitioning** (expands single-device program to multi-device).

Example annotations in StableHLO MLIR:

```mlir
// Define a 2D device mesh: 4 devices arranged as 2 x 2
sdy.mesh @mesh = <["x"=2, "y"=2]>

func.func @main(
    // Shard the embedding dimension across axis "x"
    %weights: tensor<4096 x 4096 x bf16>
        {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}
) -> tensor<4096 x 4096 x bf16> {
    ...
}
```

**Maturity caveat as of mid-2026**: Shardy is production-ready when used through JAX (which handles annotation generation). Direct Rust emission of Shardy annotations in MLIR text is possible but requires careful study of the SDY dialect spec. The `ryft-mlir` crate (see Section 8) provides Rust bindings that simplify this.

**Recommendation for gljax v1**: Skip Shardy. Use PJRT's built-in data-parallel replication: replicate weights to all devices, scatter batch across devices, gather outputs. Zero annotation needed. Add tensor-parallel Shardy sharding in v2.

### 4.3 Tensor Parallel FFN Pattern

For a SwiGLU FFN with column-parallel gate+up (shard on output dim) and row-parallel down (shard on input dim) across N devices:

```mlir
// Each device executes this sub-program.
// W_gate, W_up each: [D, FFN/N]  (column-parallel: output dim sharded)
// W_down: [FFN/N, D]             (row-parallel: input dim sharded)

%gate_partial = stablehlo.dot_general %x, %w_gate,
    contracting_dims = [1] x [0]
    : (tensor<B x D x f32>, tensor<D x FFN_N x f32>) -> tensor<B x FFN_N x f32>

%up_partial = stablehlo.dot_general %x, %w_up,
    contracting_dims = [1] x [0]
    : (tensor<B x D x f32>, tensor<D x FFN_N x f32>) -> tensor<B x FFN_N x f32>

%gate_act = stablehlo.custom_call @silu(%gate_partial) : ... -> tensor<B x FFN_N x f32>

%ffn_partial = stablehlo.multiply %gate_act, %up_partial : tensor<B x FFN_N x f32>

%down_partial = stablehlo.dot_general %ffn_partial, %w_down,
    contracting_dims = [1] x [0]
    : (tensor<B x FFN_N x f32>, tensor<FFN_N x D x f32>) -> tensor<B x D x f32>

// AllReduce sums the partial D-dim results from all devices
%out = stablehlo.all_reduce %down_partial,
    replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1 x 4 x i64>
    : (tensor<B x D x f32>) -> tensor<B x D x f32> {
  ^bb0(%a: f32, %b: f32):
    %r = stablehlo.add %a, %b : f32
    stablehlo.return %r : f32
}
```

XLA lowers `stablehlo.all_reduce` to NCCL on CUDA, ICI on TPU — you do not call NCCL directly.

### 4.4 MoE Expert Parallel

For 128 experts across N GPUs (each GPU owns 128/N experts):

The core ops needed:
1. `stablehlo.top_k` (custom or via sort+slice) for router top-2 selection
2. `stablehlo.all_to_all` to route tokens to their assigned expert's device
3. Batched expert GEMMs as `stablehlo.dot_general` with batch dim = expert index
4. `stablehlo.all_to_all` to route results back
5. Weighted sum of expert outputs

**Prior art**: MaxText (`layers/moe.py`) implements this in JAX. Capture its StableHLO output via `jax.export(...).mlir_module()` and read it as ground truth for your emitter. The `all_to_all` patterns in MaxText are the reference for TPU MoE.

**Warning**: Variable expert load (different numbers of tokens per expert) requires padding + masking. For v1, use static top-k=2 with a fixed capacity factor (e.g., 1.25x tokens per expert), padding to that capacity. This avoids dynamic shapes entirely.

### 4.5 NCCL vs XLA Collectives

Do not use NCCL directly. `stablehlo.all_reduce`, `stablehlo.all_gather`, `stablehlo.reduce_scatter`, and `stablehlo.all_to_all` are lowered by the PJRT plugin to:
- NCCL on CUDA backends (loaded by the plugin, not by you)
- ICI (Inter-Chip Interconnect) on TPU backends
- Shared memory / OpenMP on CPU backends

This hardware abstraction is the core value of PJRT. Your StableHLO is hardware-agnostic.


---

## 5. Plugin Ecosystem + Build Strategy

### 5.1 zml/pjrt-artifacts

Full platform support matrix:

| Plugin | OS | Arch | CUDA/Driver Req | Size |
|---|---|---|---|---|
| CPU | Linux, macOS, Windows | x86_64, aarch64 | None | ~80MB |
| CUDA 12 | Linux | x86_64 | CUDA 12.x, Driver ≥535 | ~1.8GB |
| ROCm 6 | Linux | x86_64 | ROCm 6.x | ~1.5GB |
| TPU (libtpu) | Linux | x86_64 | GCP TPU VM only | ~200MB |

The CPU plugin is the only one suitable for universal CI. CUDA/ROCm/TPU require hardware runners.

### 5.2 Dynamic dlopen Pattern (Full Rust)

```rust
use libloading::{Library, Symbol};
use std::sync::OnceLock;

// === Types from pjrt_c_api.h (generated by bindgen) ===
use crate::sys::{PJRT_Api, PJRT_Client_Create_Args, PJRT_Client_Create_Args_STRUCT_SIZE};

static PLUGIN: OnceLock<PjrtPlugin> = OnceLock::new();

pub struct PjrtPlugin {
    _lib: Library,
    pub api: *const PJRT_Api,
}
unsafe impl Send for PjrtPlugin {}
unsafe impl Sync for PjrtPlugin {}

pub fn init_plugin(path: &std::path::Path) -> Result<(), Error> {
    PLUGIN.get_or_try_init(|| {
        let lib = unsafe { Library::new(path) }?;
        type Sym = unsafe extern "C" fn() -> *const PJRT_Api;
        let sym: Symbol<Sym> = unsafe { lib.get(b"GetPjrtApi\0") }?;
        let api = unsafe { sym() };
        anyhow::ensure!(!api.is_null(), "GetPjrtApi returned null");
        Ok(PjrtPlugin { _lib: lib, api })
    }).map(|_| ())
}

pub fn plugin() -> &'static PjrtPlugin {
    PLUGIN.get().expect("PJRT plugin not initialized")
}
```

### 5.3 libtpu.so for TPU v5e

**How to obtain**:

Option A — pip wheel (for local cross-compilation or container builds):
```bash
pip install libtpu-nightly
python -c "import libtpu; print(libtpu.__file__)"
# Copy the .so from the package directory
```

Option B — GCP TPU VM (pre-installed):
```bash
ls /lib/libtpu.so          # TPU v4/v5
ls /usr/lib/libtpu.so      # alternate location on newer images
```

**License**: Apache 2.0 as of 2025. Compatible with gljax's zero-ML-framework-dep philosophy.

**Compatibility**: `libtpu.so` implements the PJRT C API and is the exact plugin used by JAX and PyTorch/XLA on TPUs. It is the same ABI as the CPU/CUDA plugins; `GetPjrtApi()` works identically.

**Practical dev workflow for solo dev without TPU hardware**:
1. Develop and unit-test locally using CPU plugin.
2. CI runs CPU plugin always. GPU runs behind `--features cuda` on a GPU runner.
3. TPU integration test: rent a `v5e-4` (4 chips) on GCP on-demand (~$6-12/hr). Use preemptible ("spot") for long benchmark runs (~60% cheaper). Script the test to terminate the VM when done. Estimate: $20-50 per integration test session.

### 5.4 Build Reproducibility

No `build.rs` monster needed. Plugins are purely runtime dependencies.

```toml
# Cargo.toml
[features]
default = []
cpu = []        # CI-safe: uses CPU PJRT plugin
cuda = []       # requires CUDA 12 GPU runner
tpu = []        # requires GCP TPU VM

[dependencies]
libloading = "=0.8.5"    # exact pin

[build-dependencies]
# nothing
```

Resolve plugin paths via environment variables, with documented defaults:

```rust
pub fn resolve_plugin_path(backend: Backend) -> std::path::PathBuf {
    match backend {
        Backend::Cpu  => env_or("PJRT_PLUGIN_CPU",  "pjrt_c_api_cpu_plugin.so"),
        Backend::Cuda => env_or("PJRT_PLUGIN_CUDA", "pjrt_c_api_cuda_plugin.so"),
        Backend::Tpu  => env_or("PJRT_PLUGIN_TPU",  "/lib/libtpu.so"),
    }
}
```

For CI, set `PJRT_PLUGIN_CPU` in the workflow env. For local dev, document the path in `README.md`. No Bazel, no build.rs download scripts.

### 5.5 CI Strategy

Exactly mirrors the glcuda pattern:

```yaml
# .github/workflows/ci.yml
jobs:
  test-cpu:
    runs-on: ubuntu-latest
    env:
      PJRT_PLUGIN_CPU: /path/to/pjrt_c_api_cpu_plugin.so
    steps:
      - run: cargo test --features cpu

  test-cuda:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' && contains(github.ref, 'refs/heads/main')
    env:
      PJRT_PLUGIN_CUDA: /path/to/pjrt_c_api_cuda_plugin.so
    steps:
      - run: cargo test --features cuda
```

Gate GPU/TPU tests behind push-to-main or explicit workflow dispatch. Never run them on every PR — latency and cost are prohibitive.


---

## 6. Compiled Artifact Persistence

### 6.1 PJRT Executable Serialization API

```c
PJRT_Executable_Serialize_Args serialize_args = {
    .struct_size = PJRT_Executable_Serialize_Args_STRUCT_SIZE,
    .executable  = my_pjrt_executable,  // NOT LoadedExecutable — see below
};
PJRT_Error* err = api->PJRT_Executable_Serialize(&serialize_args);
// serialize_args.serialized_executable = *const c_char  (owned bytes)
// serialize_args.serialized_bytes_size = usize
// You own the bytes. Free with PJRT_SerializedExecutable_Destroy.
```

Note the distinction:
- `PJRT_LoadedExecutable` — executable currently loaded on a device, ready to run.
- `PJRT_Executable` — the underlying compiled artifact that can be serialized.
- Get the `PJRT_Executable` from a `PJRT_LoadedExecutable` via `PJRT_LoadedExecutable_GetExecutable`.

**Stability caveat**: `PJRT_Executable_Serialize` output is **plugin-specific and version-specific**. A `.pjrt` artifact compiled by CUDA plugin v0.4.1 may not load in v0.5.0. There is no cross-plugin portability. The serialized format is XLA's native compiled executable (essentially an ELF + embedded XLA metadata), not a portable format.

**Recommendation**: Treat `.pjrt` artifacts as a compile cache, not a distribution format. Key the cache on (plugin version, model weights hash, input shapes hash). On cache miss, recompile from StableHLO.

### 6.2 AOT Compilation Workflow

```
[Rust model definition]
        ↓
[gljax IR / FuncBuilder]
        ↓
[emit StableHLO MLIR text]
        ↓
[PJRT_Client_Compile → PJRT_LoadedExecutable]   ← compilation happens here
        ↓
[PJRT_LoadedExecutable_GetExecutable → PJRT_Executable]
        ↓
[PJRT_Executable_Serialize → bytes → write to disk]
```

Loading from cache:
```
[read bytes from disk]
        ↓
[PJRT_Executable_DeserializeAndLoad → PJRT_LoadedExecutable]
        ↓  
[ready to execute]
```

XLA performs all compilation inside the plugin. Your code never calls into LLVM or the XLA compiler directly — you pass StableHLO text to `PJRT_Client_Compile` and get back a compiled executable. The compilation pipeline (StableHLO → HLO → LLVM IR → machine code) is entirely inside the plugin .so.

### 6.3 Compile Cache Design

Cache key should be a SHA256 hash of:
1. Plugin path + plugin version string (from `PJRT_Api_Version`)
2. Model architecture identifier (crate version + model config hash)
3. Input shape signature (dtypes and shapes of all `main` function arguments)
4. Weight tensor shapes (not values — weight updates don't require recompile)
5. Compilation flags (e.g., optimization level, device ordinal)

Weight value changes do NOT invalidate the compiled executable. You can load new weights at runtime into existing buffer shapes without recompiling. This is the "compile once, serve many weights" pattern used by ZML.

Shape changes (different batch size, different sequence length) **do** require recompilation unless you use dynamic shapes (see Section 9).

```rust
pub struct CompileKey {
    pub plugin_version: (u64, u64),     // (major, minor)
    pub mlir_hash: [u8; 32],            // SHA256 of StableHLO text
    pub shape_sig: Vec<ShapeDesc>,      // (dtype, dims) per input
}

impl CompileKey {
    pub fn cache_path(&self, cache_dir: &Path) -> PathBuf {
        let key = hex::encode(sha256(&bincode::serialize(self).unwrap()));
        cache_dir.join(format!("{key}.pjrt"))
    }
}
```

### 6.4 Compile Cache Invalidation Summary

| Change Type | Requires Recompile? |
|---|---|
| Weight values updated (fine-tune, LoRA) | No |
| Weight shapes changed (different model size) | Yes |
| Input batch size changed | Yes (unless using dynamic shapes) |
| Input sequence length changed | Yes (unless padded to fixed max) |
| Plugin version bumped | Yes (serialized format not portable) |
| StableHLO IR changed | Yes (always hash the MLIR text) |


---

## 7. LLM-Specific Ops — Implementation Guide

### 7.1 RMSNorm

**(a) StableHLO representation**:
```mlir
// RMSNorm: out = x / sqrt(mean(x^2) + eps) * weight
// x: [B, S, D], weight: [D]
func.func @rms_norm(%x: tensor<B x S x D x f32>, %w: tensor<D x f32>,
                    %eps: tensor<f32>) -> tensor<B x S x D x f32> {
  // Square
  %x2 = stablehlo.multiply %x, %x : tensor<B x S x D x f32>
  
  // Mean over D
  %sum = stablehlo.reduce(%x2 init: %zero) across dimensions = [2]
      : (tensor<B x S x D x f32>, tensor<f32>) -> tensor<B x S x f32> { ... }
  %d_inv = stablehlo.constant dense<0.00390625> : tensor<f32>  // 1/D
  %mean  = stablehlo.multiply %sum_bc, %d_inv_bc : tensor<B x S x f32>
  
  // Add epsilon, sqrt, reciprocal
  %eps_bc  = stablehlo.broadcast_in_dim %eps, dims = [] : ... -> tensor<B x S x f32>
  %mean_e  = stablehlo.add %mean, %eps_bc : tensor<B x S x f32>
  %rsqrt   = stablehlo.rsqrt %mean_e : tensor<B x S x f32>
  
  // Broadcast back and multiply
  %rsqrt_bc = stablehlo.broadcast_in_dim %rsqrt, dims = [0, 1]
              : (tensor<B x S x f32>) -> tensor<B x S x D x f32>
  %normed    = stablehlo.multiply %x, %rsqrt_bc : tensor<B x S x D x f32>
  
  // Scale by weight
  %w_bc = stablehlo.broadcast_in_dim %w, dims = [2]
          : (tensor<D x f32>) -> tensor<B x S x D x f32>
  %out = stablehlo.multiply %normed, %w_bc : tensor<B x S x D x f32>
  return %out : tensor<B x S x D x f32>
}
```

**(b) custom_call alternative**: `@rms_norm` is registered as a custom call in XLA on both GPU and TPU and will be pattern-matched if you emit the above sequence. XLA's `simplify_qdq` and `norm_runner` passes fuse the reduce+rsqrt+multiply into a single kernel automatically on both GPU and TPU. You do not need to emit the custom call yourself.

**(c) Performance pitfalls**:
- **TPU**: The `reduce` across D must be across the fastest-varying dimension. Ensure D is the innermost dimension in your tensor layout. If D is not the last dimension in memory, add a `stablehlo.transpose` before the reduce.
- **GPU**: The fused kernel is launched by XLA. Do NOT manually split the reduce into multiple ops — it prevents fusion.
- **Epsilon placement**: Place epsilon add **after** the mean, not inside the sqrt argument. Some sources put it as `1/(sqrt(mean + eps))`. This is numerically equivalent but the fused kernel pattern-matcher may not recognize an alternative placement.

### 7.2 RoPE NeoX Variant (Qwen2/Qwen3 Style)

**(a) StableHLO representation**:
```mlir
// RoPE NeoX: rotate pairs (q[..., 0::2], q[..., 1::2]) using (cos, sin)
// q:   [B, H, S, D]
// cos: [S, D/2]  (precomputed)
// sin: [S, D/2]

// Split q into even and odd halves
%q_even = stablehlo.slice %q [0:B:1, 0:H:1, 0:S:1, 0:D_2:1]
          : (tensor<B x H x S x D x f32>) -> tensor<B x H x S x D_2 x f32>
%q_odd  = stablehlo.slice %q [0:B:1, 0:H:1, 0:S:1, D_2:D:1]
          : (tensor<B x H x S x D x f32>) -> tensor<B x H x S x D_2 x f32>

// Broadcast cos/sin: [S, D/2] -> [B, H, S, D/2]
%cos_bc = stablehlo.broadcast_in_dim %cos, dims = [2, 3]
          : (tensor<S x D_2 x f32>) -> tensor<B x H x S x D_2 x f32>
%sin_bc = stablehlo.broadcast_in_dim %sin, dims = [2, 3]
          : (tensor<S x D_2 x f32>) -> tensor<B x H x S x D_2 x f32>

// Rotate: out_even = q_even * cos - q_odd * sin
//         out_odd  = q_even * sin + q_odd * cos
%rot_even = stablehlo.subtract
    (stablehlo.multiply %q_even, %cos_bc),
    (stablehlo.multiply %q_odd,  %sin_bc)
    : tensor<B x H x S x D_2 x f32>
%rot_odd  = stablehlo.add
    (stablehlo.multiply %q_even, %sin_bc),
    (stablehlo.multiply %q_odd,  %cos_bc)
    : tensor<B x H x S x D_2 x f32>

// Concatenate back
%q_rot = stablehlo.concatenate %rot_even, %rot_odd, dim = 3
         : (tensor<B x H x S x D_2 x f32>, tensor<B x H x S x D_2 x f32>)
           -> tensor<B x H x S x D x f32>
```

**(b) custom_call alternative**: No standard registered RoPE custom call in XLA. The above op sequence is compact enough that XLA's elementwise fusion handles it well. Do not use custom_call for RoPE.

**(c) Performance pitfalls**:
- **Static position offsets**: The `cos`/`sin` tensors must be precomputed and passed as constants or weights, not recomputed inside the loop. Precompute the full RoPE table up to max_seq_len at model load time.
- **TPU**: The slice/concatenate pattern is efficient on TPU only if D is a multiple of 128 (TPU tile size). For Qwen2-7B (head_dim=128), this is fine. For head_dim=64 or unusual sizes, add padding to 128 and slice the excess at the output.

### 7.3 Causal Attention Mask

**(a) StableHLO representation** (static seq len preferred):
```mlir
// Generate a causal mask as a constant tensor [S, S] of -inf/-0
// XLA can fold this to a constant if S is static.
%mask = stablehlo.constant dense<[[-0.0, -inf, -inf],
                                   [-0.0, -0.0, -inf],
                                   [-0.0, -0.0, -0.0]]> : tensor<3 x 3 x f32>

// Add to scores before softmax
%masked_scores = stablehlo.add %scores, %mask_bc : tensor<B x H x S x S x f32>
```

For static S, XLA folds the mask add into the softmax kernel. This is the MaxText pattern.

**(b) custom_call alternative**: Flash Attention (`@flash_attention`) is registered as a custom call in XLA for both GPU (via cuDNN / Pallas) and TPU (via Pallas). This fuses QKV matmuls + masking + softmax + AV into a single kernel. To use it:

```mlir
%out = stablehlo.custom_call @flash_attention(%q, %k, %v)
    {backend_config = "{\"is_causal\":true, \"scale\":0.125}"}
    : (tensor<B x H x S x D x bf16>, tensor<B x H x S x D x bf16>,
       tensor<B x H x S x D x bf16>) -> tensor<B x H x S x D x bf16>
```

The exact `backend_config` schema is XLA version-dependent. The call name may also be `@__cudnn$fmhaCausalMask` on GPU. **Prefer the explicit op sequence for portability** and rely on XLA's automatic fusion for performance. Only reach for custom_call Flash Attention if you profile and find the auto-fused path is slower.

**(c) Performance pitfalls**:
- Static S is critical on TPU — it enables the XLA compiler to generate fixed-shape TPU programs which are 3-10x faster than dynamic-shape equivalents.
- On GPU, static S enables cuDNN Flash Attention which has strict shape requirements.

### 7.4 GQA (Grouped Query Attention)

**(a) StableHLO representation** — repeat KV heads to match Q head count:
```mlir
// Q: [B, H_q, S, D], K/V: [B, H_kv, S, D], where H_q = N * H_kv
// Repeat K and V N times along the head dimension

// stablehlo has no native repeat. Use broadcast + reshape:
// K: [B, H_kv, S, D] -> reshape to [B, H_kv, 1, S, D]
//                     -> broadcast to [B, H_kv, N, S, D]
//                     -> reshape to [B, H_kv*N, S, D]

%k_expanded = stablehlo.reshape %k
    : (tensor<B x H_kv x S x D x bf16>) -> tensor<B x H_kv x 1 x S x D x bf16>
%k_bc = stablehlo.broadcast_in_dim %k_expanded, dims = [0, 1, 2, 3, 4]
    : (tensor<B x H_kv x 1 x S x D x bf16>) -> tensor<B x H_kv x N x S x D x bf16>
%k_rep = stablehlo.reshape %k_bc
    : (tensor<B x H_kv x N x S x D x bf16>) -> tensor<B x H_q x S x D x bf16>
// Same for V. Then proceed with standard MHA.
```

**(b) custom_call alternative**: The Flash Attention custom call natively supports GQA via `num_query_heads` / `num_kv_heads` config fields, avoiding the explicit repeat. Worth using if you use Flash Attention anyway.

**(c) Performance pitfalls**:
- The broadcast+reshape expand is memory-bound — K and V data is duplicated in SRAM.
- On TPU with large H_kv and long S, the memory expansion can cause HBM bandwidth saturation. Flash Attention's GQA mode avoids this by loading KV tiles N times without materializing the expansion.

### 7.5 SwiGLU FFN

**(a) StableHLO representation**:
```mlir
// SwiGLU: out = (gate * silu(gate)) * up_proj applied to x, then down
// gate_up projection: x -> [gate, up] via a single matmul with 2*FFN columns,
// then split.

// Or separately:
%gate_preact = stablehlo.dot_general %x, %w_gate, contracting_dims=[2]x[0]
    : (tensor<B x S x D x f32>, tensor<D x FFN x f32>) -> tensor<B x S x FFN x f32>
%up_preact = stablehlo.dot_general %x, %w_up, contracting_dims=[2]x[0]
    : (tensor<B x S x D x f32>, tensor<D x FFN x f32>) -> tensor<B x S x FFN x f32>

// SiLU: x * sigmoid(x)
%sigmoid_gate = stablehlo.logistic %gate_preact : tensor<B x S x FFN x f32>
%silu_gate = stablehlo.multiply %gate_preact, %sigmoid_gate : tensor<B x S x FFN x f32>

// Gated product
%gated = stablehlo.multiply %silu_gate, %up_preact : tensor<B x S x FFN x f32>

// Down projection
%out = stablehlo.dot_general %gated, %w_down, contracting_dims=[2]x[0]
    : (tensor<B x S x FFN x f32>, tensor<FFN x D x f32>) -> tensor<B x S x D x f32>
```

**(b) custom_call alternative**: XLA fuses `dot + logistic + multiply` into a single kernel on both GPU and TPU via its epilogue fusion. No custom call needed.

**(c) Performance pitfalls**:
- Emit `gate` and `up` as a single fused `[D, 2*FFN]` weight matmul, then split. This halves the number of matmul kernel launches and improves memory access patterns (one large GEMM vs two smaller).
- `stablehlo.logistic` is the correct op for sigmoid. Do not approximate with `tanh` or piecewise linear — XLA knows to fuse logistic with multiply.

### 7.6 MoE Top-k Routing + Expert Dispatch

**(a) StableHLO representation** (simplified top-2 static routing):
```mlir
// Router: x -> logits -> top-2 indices + weights
%logits = stablehlo.dot_general %x, %w_router, contracting_dims=[2]x[0]
    : (tensor<B x S x D x f32>, tensor<D x E x f32>) -> tensor<B x S x E x f32>

// Top-2 (XLA has stablehlo.top_k in newer versions, otherwise sort+slice)
// ... produces %indices: [B, S, 2], %weights: [B, S, 2] ...

// Dispatch: gather tokens for each expert
// Use stablehlo.gather with expert index as start_index
// ... produces %expert_inputs: [E, capacity, D] ...

// Expert compute: batched GEMM
// For each expert e, compute W1[e], W2[e] GEMMs
// These can be expressed as a single batched dot_general
%expert_out = stablehlo.dot_general %expert_inputs, %expert_w1,
    batching_dims = [0] x [0],  // batch over expert dimension
    contracting_dims = [2] x [1]
    : (tensor<E x cap x D x f32>, tensor<E x D x FFN_E x f32>)
      -> tensor<E x cap x FFN_E x f32>

// Reassemble: weighted sum via scatter
// ... %output: [B, S, D] ...
```

**(b) custom_call alternative**: `@moe_dispatch` and `@moe_combine` are custom calls in some XLA builds (especially TPU). Check MaxText's MoE source for the exact call signature used on v5e.

**(c) Performance pitfalls**:
- **Load imbalance**: top-2 routing without capacity constraints causes some experts to overflow. Always use a fixed capacity factor (e.g., 1.25) and a `stablehlo.pad` to normalize expert token counts before the batched GEMM.
- **TPU**: The `all_to_all` for cross-device dispatch is bandwidth-limited on inter-chip ICI links. Profile this first before scaling to multi-host MoE.

### 7.7 Paged KV Cache vs Static KV Cache

**Recommendation for gljax v1: use static KV cache.**

Static KV cache: allocate `[max_seq_len, num_heads, head_dim]` tensors at start, write each token's K and V at position `[pos]` via `stablehlo.scatter`, read via `stablehlo.slice` with static bounds.

Advantages:
- Entirely static shapes → XLA can fully optimize.
- Simple to implement: `scatter` one position per step.
- Compatible with compiled `.pjrt` artifact caching.

Paged KV cache: dynamically allocate fixed-size "pages" of KV entries, chain pages via a page table.

Disadvantages for current gljax:
- Requires dynamic shapes or significant padding overhead.
- Complicates the XLA compilation model significantly.
- Not needed until you're running concurrent request batches with highly variable lengths (serving workload, not single-request inference).

Defer paged KV to when gljax has a working serving path. Static KV is sufficient for benchmarking and single-session inference.

### 7.8 Token Embedding Lookup

**(a) StableHLO representation**:
```mlir
// embedding_table: [vocab_size, d_model]
// token_ids: [batch, seq_len] (i32)

%embeds = stablehlo.gather %embedding_table, %token_ids,
    dimension_numbers = <
      offset_dims = [2],              // output dim 2 is the embedding dim
      collapsed_slice_dims = [0],     // vocab dim is collapsed (we select one row)
      start_index_map = [0],          // token_id indexes into dim 0 of table
      index_vector_dim = 2            // each token_id is a scalar in the last dim
    >,
    slice_sizes = [1, D_MODEL]
    : (tensor<V x D x f32>, tensor<B x S x 1 x i32>) -> tensor<B x S x D x f32>
```

**(b) custom_call alternative**: None needed. XLA's gather lowering is efficient.

**(c) Performance pitfalls**:
- Embedding tables are accessed randomly (token indices are not sequential). This is an irregular memory access pattern — make sure the table is in HBM, not streamed through SRAM.
- On TPU, large vocabularies (250k+ tokens, Qwen3 uses 151k) require careful tiling. XLA handles this but be aware of the compile time overhead for very large embedding gathers.
- Weight tying (sharing embedding table with LM head): express as the same SSA value used in both gather and a final dot_general. XLA will not double-store the weights.


---

## 8. Prior Art + Lessons Learned

### 8.1 fusebox (Rust, Erik Kaunismäki, Mar 2026)

Source: Erik's blog + associated code (he's affiliated with ZML). fusebox runs SmolLM2-135M via StableHLO + PJRT CPU plugin in pure Rust.

**What it got right**:
- Text-based StableHLO emission from Rust — no MLIR C API needed for simple models. Validates that the text-emit → PJRT compile path works end-to-end.
- Static weight loading from safetensors into PJRT CPU buffers. Useful reference for the `checkpoint/` module.
- Minimal, readable code (~2k LOC). Good starting point for understanding the plumbing.
- Shows the exact `GetPjrtApi` → `PJRT_Client_Create` → `PJRT_Client_Compile` → `Execute` loop.

**What's missing for production**:
- Single-device CPU only. No GPU, no TPU, no multi-device.
- No dynamic generation loop — single forward pass only, no KV cache.
- No mixed precision / BF16 support.
- No distributed sharding (Shardy, tensor parallel).
- No compiled artifact persistence / compile cache.
- No attention (SmolLM2-135M uses attention but fusebox emits a simplified version, not the full GQA path).
- No safetensors streaming for large models — loads entire model into RAM first.

**Verdict**: Use fusebox as a validation scaffold to confirm your PJRT bindings work. Do not use it as an architecture reference for gljax beyond the basic plumbing layer.

### 8.2 ZML (Zig, March 2026 v2 release)

ZML is a production inference stack: Zig + OpenXLA + MLIR + Bazel. v2 (March 2026) supports NVIDIA, AMD, Google TPU, and AWS Trainium from a single codebase.

**Architectural decisions worth copying**:

1. **Compile/weight separation**: ZML separates the "compile the computation graph" step from the "load the weights" step. The compiled executable is shape-parameterized but weight-value-agnostic. Weights are loaded as PJRT buffers at serving time and passed as arguments. This enables the "compile once, hot-reload weights" pattern. Copy this into gljax's `runtime/Session`.

2. **`ShapeProvider` pattern**: ZML's tracing system traces through the model with symbolic shapes, collecting ops without actually computing values. The Rust equivalent is a proc-macro `#[derive(Module)]`-style trait that produces a `TraceCx` + `FuncBuilder` from a model struct definition. Your `graph/` module should implement this.

3. **Buffers as typed wrappers**: ZML wraps `PJRT_Buffer*` in a typed `Tensor<dtype, shape>` struct. This catches shape mismatches at Rust compile time for known-static shapes. Implement this in gljax's `tensor/` module.

4. **Plugin abstraction layer**: ZML defines a thin hardware-agnostic `Platform` trait over PJRT plugins. All model code is written against `Platform`. This is exactly what gljax needs so that a Qwen3-7B forward pass doesn't know if it's running on TPU or A100.

**What not to copy**:
- Bazel build system. Cargo is sufficient.
- Zig-specific metaprogramming patterns. Rust proc-macros cover the same ground.
- ZML's tight coupling to their internal GLLM format. gljax uses safetensors.

### 8.3 xla-rs (Laurent Mazare)

xla-rs provides Rust bindings for XLA's C++ API (not the PJRT C API). It links against libtensorflow_framework and the full XLA C++ library.

**Why gljax is NOT using it**:
1. **Compile-time linking**: requires pre-built XLA C++ libraries (~2GB+), making it incompatible with gljax's zero-compiled-dependency philosophy.
2. **C++ ABI fragility**: Rust FFI into C++ namespaces is inherently fragile. The PJRT C API exists precisely to avoid this.
3. **Not maintained for PJRT v2**: xla-rs targets the older XlaBuilder C++ API, not the current PJRT plugin model.
4. **Dependency chain**: brings in protobuf-sys, TF framework symbols, and other transitive deps — antithetical to gljax's design.

**What its FFI approach reveals about PJRT binding pain**:
- xla-rs had to add extensive `#[allow(non_snake_case)]` and `#[allow(dead_code)]` suppressions for bindgen output. Expect the same for PJRT C API bindings.
- Shape/type management for XlaBuilder is verbose. PJRT's buffer-centric API is actually simpler — you don't manage shape metadata explicitly in PJRT client code.
- Error handling via `StatusOr<T>` in C++ is messier than PJRT's `PJRT_Error*` pattern.

### 8.4 MaxText (Google, JAX)

MaxText is Google's reference LLM implementation on TPU in JAX. It runs Llama2, Mistral, Gemma, and Mixtral.

**StableHLO patterns worth studying** (via `jax.export`):
- `layers/attentions.py`: GQA + causal masking pattern with Flash Attention custom call.
- `layers/normalizations.py`: RMSNorm via `jax.lax.reduce` — examine how JAX lowers this to StableHLO.
- `layers/linears.py`: SwiGLU and ColumnParallel/RowParallel patterns.
- `layers/moe.py`: MoE dispatch with `jax.lax.all_to_all` — the canonical TPU MoE reference.

To dump StableHLO from MaxText:
```python
import jax
exported = jax.export(jax.jit(model.forward))(sample_inputs)
print(exported.mlir_module())  # prints StableHLO MLIR text
```

This gives you ground-truth MLIR for every op combination you need to implement. Paste it into gljax's test fixtures as reference output.

### 8.5 Existing Rust PJRT Crates (2026)

Two relevant crates found:

**`ryft` / `ryft-pjrt` / `ryft-xla-sys` / `ryft-mlir`** (lib.rs, published ~April 2026):
- `ryft-xla-sys`: low-level `-sys` bindings for XLA/MLIR/PJRT. Handles native artifact building/downloading. Has feature flags for CUDA 12, ROCm, etc.
- `ryft-pjrt`: high-level, ownership-aware Rust bindings for PJRT clients, buffers, and execution.
- `ryft-mlir`: high-level Rust bindings for MLIR and dialects including StableHLO and Shardy.
- `ryft` top-level: JAX-inspired tracing, AD, and JIT compilation crate.

**Recommendation**: Study `ryft-xla-sys` to understand what bindgen output and build-time wiring look like in practice. Do NOT take it as a dependency (brings in XLA C++ artifacts, contradicts gljax's dynamic loading philosophy). Use it as a reference for your own `gljax-sys` crate design.

The fact that `ryft` exists validates the approach — but its scope is broader (it includes AD and JIT, which gljax explicitly excludes). gljax's value proposition is the tight integration with gwenland-ml's IR and the zero-compiled-dep policy.


---

## 9. Risks + Open Questions

### 9.1 PJRT C API ABI Stability

**Historical record**: The PJRT C API has had breaking changes between openxla releases. The `CHANGELOG.md` in `xla/pjrt/c/` documents these. The struct-size versioning mechanism is designed to catch mismatches at runtime, but it cannot protect you if a function's *semantics* change while its signature stays the same.

**Known categories of breakage**:
- New required `*_Args` fields added with no safe default (zero is not always safe).
- Function pointer slots reordered in `PJRT_Api` struct (would break if you access by offset, but you access by name via bindgen so this is safe).
- Deprecated functions removed from the vtable (new plugin, old caller pattern).

**Mitigation**:
1. Pin to a specific `pjrt-artifacts` tag. Don't auto-update.
2. Read `CHANGELOG.md` on every upgrade. It's thorough.
3. Check major/minor version at startup and refuse if mismatched.
4. Maintain a `PJRT_API_VERSION_TESTED = (0, 58)` constant in your code. Add a compile-time assert (or startup check) that panics if the runtime version diverges from your tested version.

### 9.2 TPU v5e Availability for Solo Dev

**Pricing (approximate, mid-2026)**:
- On-demand `v5e-4` (4 chips): ~$3.50/chip-hour = ~$14/hour
- Preemptible `v5e-4`: ~$1.40/chip-hour = ~$5.60/hour (can be reclaimed with 30s notice)
- `v5e-8` (8 chips): double the above

**Practical workflow**:
1. Develop locally with CPU plugin. Test all correctness properties on CPU.
2. For TPU integration tests, write a `scripts/run_tpu_test.sh` that:
   - Creates a preemptible TPU VM via `gcloud compute tpus tpu-vm create ...`
   - Copies the gljax test binary via `gcloud compute scp`
   - Runs the test binary with `PJRT_PLUGIN_TPU=/lib/libtpu.so`
   - Deletes the VM immediately on success or failure
3. Budget: ~$20 per TPU integration test session (30-60 min).
4. Run TPU tests manually on feature branches, not on every PR.

**Alternative**: Use Google's [TPU Research Cloud (TRC)](https://sites.research.google/trc/) for free TPU access. Apply as a research project. Turnaround is typically 1-2 weeks.

### 9.3 StableHLO Dynamic Shapes — Do You Need Them?

**Short answer**: No, not for gljax v1. Use padding + static shapes.

**Detailed analysis**:

Static approach (recommended for v1):
- Pad all sequences to `max_seq_len` (e.g., 2048 or 4096).
- Use a causal attention mask to zero out padding positions.
- Compile one executable per `(batch_size, seq_len, model_config)` tuple.
- Cache compiled executables keyed on this tuple.

Advantages:
- Full XLA optimization (static shapes enable the best kernel scheduling).
- Simple compile cache (shape is the key).
- Parity with MaxText's default approach.

Disadvantages:
- Compile multiple artifacts for different seq_len buckets (e.g., 128, 256, 512, 1024, 2048).
- Wasted compute on padding tokens.

Dynamic shapes (defer to v2 or later):
- Requires `tensor<?x512xf32>` types and `stablehlo.set_dimension_size` ops.
- XLA compilation with dynamic shapes is significantly more complex and slower.
- PJRT's handling of dynamic shapes requires shape inference at execute time.
- The StableHLO dynamism spec (RFC 20230704) is still evolving as of 2026.

**Recommendation**: Implement a static shape bucketing strategy. On the first request of a new shape, compile and cache. Subsequent requests with the same shape hit the cache. Use 6-8 pre-warmed bucket sizes at startup.

### 9.4 MLIR Text vs Protobuf

**Emit text (recommended)**. Here's why:

MLIR text format:
- Human-readable and debuggable.
- PJRT's `PJRT_Client_Compile` accepts it directly (format = "mlir" in `PJRT_Program`).
- No protobuf dependency in your crate.
- Easy to dump for debugging: write to a file, open in any text editor.
- Text parsing is done by the XLA compiler inside the plugin — not your code.

StableHLO protobuf:
- More compact wire format.
- Covered by the 5-year compatibility guarantee (bytecode serialization, not text).
- Required for stored artifacts intended for long-term replay (not compilation input).
- Requires linking against or calling the StableHLO C API for serialization.

**For the `stablehlo/` emitter module**: emit text. For the `serialization/` module (stored artifacts): use bytecode/protobuf via `stablehlo-translate --serialize` invoked as a subprocess, or via the StableHLO C API if you add a `stablehlo-sys` build dependency (acceptable for serialization only).

The text format has been stable in practice for years. PJRT backends parse it via MLIR's own parser which is very robust. There is no documented case of a production breakage caused by MLIR text format changes.

---

## 10. Prioritized Reading List

### Tier 1 — Read Before Writing a Single Line of gljax

1. **`xla/pjrt/c/pjrt_c_api.h`** (openxla/xla, main branch)  
   The primary source of truth. Read every struct and every function pointer. Pay attention to the `_Args` patterns and `struct_size` usage.  
   https://github.com/openxla/xla/blob/main/xla/pjrt/c/pjrt_c_api.h

2. **PJRT Integration Guide** (`xla/pjrt/c/docs/pjrt_integration_guide.md`)  
   Explains the caller contract, the expected call sequence for compile+execute, and error handling ownership rules.  
   https://github.com/openxla/xla/blob/main/xla/pjrt/c/docs/pjrt_integration_guide.md

3. **StableHLO Spec** (openxla.org/stablehlo/spec)  
   The authoritative op reference. Read the type system section, then the op definitions for all ops in Section 2.2 of this report. This is your ground truth for op semantics.  
   https://openxla.org/stablehlo/spec

4. **StableHLO Compatibility and Versioning**  
   Understand the VHLO dialect and the 5-year/2-year guarantee before designing your serialization layer.  
   https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md

5. **`pjrt_c_api/CHANGELOG.md`** (openxla/xla)  
   Read the history to understand what has broken before and why. This informs your version-checking strategy.  
   https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md

### Tier 2 — Read While Implementing

6. **fusebox blog post** (Erik Kaunismäki / ZML, March 2026)  
   Fastest path to understanding the end-to-end Rust + PJRT + StableHLO plumbing. Read alongside the source code.  
   https://erikkaum.com/blog/zml (search for the March 2026 post on ZML Rust / fusebox)

7. **ZML source code** (zml/zml, GitHub)  
   Focus on: `zml/src/backend.zig` (Platform abstraction), `zml/src/tensor.zig` (buffer management), `zml/src/mlir.zig` (StableHLO emission). Zig is readable if you know C.  
   https://github.com/zml/zml

8. **MaxText `layers/` directory** (google-deepmind/maxtext, GitHub)  
   Reference LLM op implementations in JAX. Use `jax.export` to dump their StableHLO for every op you need to implement. Treat the dumps as test fixtures.  
   https://github.com/google-deepmind/maxtext/tree/main/MaxText/layers

9. **Shardy documentation** (openxla.org/shardy)  
   Read the `sharding_representation.md` and `dialect_agnostic_sharding.md` docs to understand how Shardy annotations are expressed in MLIR text. Needed before implementing `distributed/`.  
   https://openxla.org/shardy

10. **`ryft-xla-sys` crate source** (crates.io / lib.rs)  
    Study as a reference for how to structure your own `gljax-sys` crate (bindgen setup, feature flags, plugin path resolution). Do not take as a dependency.  
    https://lib.rs/crates/ryft-xla-sys

### Tier 3 — Reference While Debugging / Optimizing

11. **XLA GPU Architecture Overview** (openxla.org/xla/gpu_architecture)  
    Explains the PJRT runtime execution path on GPU, HLO scheduling, and where compilation time is spent.  
    https://openxla.org/xla/gpu_architecture

12. **"How to Think About TPUs"** (jax-ml scaling book)  
    Explains TPU MXU tiling constraints, BF16 accumulation behavior, and why static shapes matter. Essential before writing any TPU-specific tuning.  
    https://jax-ml.github.io/scaling-book/tpus/

13. **PJRT C++ API Overview** (openxla.org/xla/pjrt/cpp_api_overview)  
    Clarifies the relationship between C and C++ APIs. Useful for understanding the semantics of functions that are more naturally explained in C++ terms.  
    https://openxla.org/xla/pjrt/cpp_api_overview

14. **StableHLO Dynamism RFC** (openxla/stablehlo, rfcs/)  
    Read before designing any dynamic-shape feature to understand the design decisions and limitations.  
    https://github.com/openxla/stablehlo/blob/main/rfcs/20230704-dynamism-101.md

15. **JAX discussion: "Calling pre-compiled JAX code from C++"** (GitHub jax-ml/jax #22184)  
    Community discussion that documents the exact PJRT C API pattern for deserialize+load+execute. Contains working C++ code that maps directly to what gljax needs in Rust.  
    https://github.com/jax-ml/jax/discussions/22184

---

## Appendix: Recommended gljax Module Responsibilities

```
src/
├── sys/           # Raw bindgen output for pjrt_c_api.h (gljax-sys subcrate)
├── pjrt/
│   ├── plugin.rs  # dlopen, GetPjrtApi, version check
│   ├── client.rs  # PJRT_Client lifecycle, device enumeration
│   ├── buffer.rs  # PJRT_Buffer, host<->device transfer
│   ├── compile.rs # PJRT_Client_Compile, PJRT_Program construction
│   ├── exec.rs    # PJRT_LoadedExecutable_Execute, event await
│   └── error.rs   # PJRT_Error extraction + destroy wrapper
├── stablehlo/
│   ├── emitter.rs # MlirEmitter: text buffer + SSA counter
│   ├── types.rs   # TensorType, DType, ShapeDesc formatting
│   └── ops.rs     # emit_dot_general, emit_reduce, emit_gather, etc.
├── graph/
│   ├── builder.rs # FuncBuilder: traces ops, owns MlirEmitter
│   ├── value.rs   # SsaValue: typed handle into the trace
│   └── trace.rs   # TraceCx: context for a single forward pass trace
├── ops/           # High-level ops (rms_norm, rope, attention, swiglu, moe, ...)
├── precision/     # PrecisionPolicy: decides dtype per op class
├── runtime/
│   ├── session.rs # Session: owns plugin + client + compile cache
│   └── cache.rs   # CompileKey, disk cache, invalidation logic
├── distributed/
│   ├── mesh.rs    # DeviceMesh: device topology description
│   └── shard.rs   # ShardingSpec: Shardy annotation builder (v2)
├── checkpoint/
│   ├── safetensors.rs  # safetensors reader → PJRT buffer
│   └── gllm.rs         # GLLM format reader
└── serialization/
    ├── mlir.rs    # dump/load StableHLO MLIR text
    └── pjrt.rs    # PJRT_Executable_Serialize / DeserializeAndLoad
```

---

*End of ARTX1 — gljax PJRT + StableHLO Research Report*  
*Next: ARTX2 — gljax IR Design: FuncBuilder, TraceCx, and SSA Value System*
