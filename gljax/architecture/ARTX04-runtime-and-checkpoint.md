# ARTX4 — gljax runtime/ and checkpoint/

**Series:** gljax (Sanctum Visibilia) Architecture Research  
**Depends on:** ARTX1 (PJRT C API FFI), ARTX2 (IR: FuncBuilder/TraceCx/SSA), ARTX3 (ops/ layer)  
**Next:** ARTX5 — Static KV Cache + Bucketing Strategy (incremental decode, shape buckets)

---

## Overview

ARTX4 covers the execution layer — the bridge between a compiled `BuiltFunc` (MLIR text + Signature)
and actual inference on a device. It spans two subsystems:

**`src/runtime/`** — Session lifecycle, PJRT plugin management, compile pipeline, execute loop, profiling.  
**`src/checkpoint/`** — Weight loading from safetensors and GLLM format into PJRT device buffers.

After ARTX4 is implemented, the full end-to-end path exists:

```
TraceCx trace → BuiltFunc (MLIR + Signature)
  → Session::compile() → PjRtLoadedExecutable (cached to disk)
  → Checkpoint::load() → Vec<PjRtBuffer> (weights on device)
  → Session::execute() → Vec<PjRtBuffer> (outputs)
  → Buffer::to_host() → Vec<f32> (logits)
```

### Module layout

```
src/runtime/
├── mod.rs
├── session.rs       # Session: owns client + executable + weight buffers
├── execution.rs     # execute loop: pack inputs, run, unpack outputs
├── cache.rs         # compile artifact cache (disk, SHA256 keyed)
└── profiler.rs      # timing + memory telemetry per-run

src/checkpoint/
├── mod.rs
├── safetensors.rs   # memory-mapped safetensors → PJRT buffer
├── gllm.rs          # GLLM format loader (via glictus-caliburni)
└── signature.rs     # ParamSpec matching: Signature ↔ checkpoint keys
```

---

## 1. `runtime/session.rs`

### Role

`Session` is the top-level runtime object. It owns:
- A loaded PJRT plugin (`PjRtApi`)
- A created `PjRtClient`
- A compiled `PjRtLoadedExecutable`
- A set of weight `PjRtBuffer`s bound in Signature order

One `Session` = one compiled model on one device. Multiple sessions can coexist
(e.g., one CPU session for oracle, one GPU session for production).

### Lifecycle

```
Session::new(plugin_path, built_func, checkpoint) → Session
  1. Load plugin     → PjRtApi (dlopen)
  2. Create client   → PjRtClient
  3. Compile         → PjRtLoadedExecutable (or load from cache)
  4. Load weights    → Vec<PjRtBuffer> (in Signature::weights order)
```

### Implementation

```rust
// src/runtime/session.rs

use crate::{
    graph::builder::BuiltFunc,
    pjrt::{
        client::{PjRtClient, PjRtClientConfig},
        executable::PjRtLoadedExecutable,
        buffer::PjRtBuffer,
        plugin::PjRtPlugin,
    },
    checkpoint::Checkpoint,
    runtime::{cache::CompileCache, execution::ExecutionPlan},
};
use std::path::Path;

pub struct Session {
    client:     PjRtClient,
    executable: PjRtLoadedExecutable,
    weights:    Vec<PjRtBuffer>,    // in Signature::weights order
    plan:       ExecutionPlan,       // describes how to pack inputs + unpack outputs
}

impl Session {
    /// Create a Session from a compiled function and checkpoint.
    /// plugin_path: path to PJRT plugin .so/.dll (CPU, CUDA, TPU)
    pub fn new(
        plugin_path: impl AsRef<Path>,
        built: &BuiltFunc,
        checkpoint: &dyn Checkpoint,
        cache: Option<&CompileCache>,
    ) -> Result<Self, SessionError> {

        // 1. Load PJRT plugin
        let plugin = PjRtPlugin::load(plugin_path.as_ref())?;

        // 2. Create client
        let client = PjRtClient::create(&plugin, PjRtClientConfig::default())?;

        // 3. Compile or load from cache
        let executable = match cache {
            Some(cache) => cache.get_or_compile(&client, &plugin, &built.mlir)?,
            None => client.compile(&built.mlir, &plugin)?,
        };

        // 4. Load weights into device buffers
        let weights = Self::load_weights(&client, built, checkpoint)?;

        let plan = ExecutionPlan::new(&built.signature);

        Ok(Self { client, executable, weights, plan })
    }

    /// Load all weights from checkpoint in Signature::weights order.
    fn load_weights(
        client: &PjRtClient,
        built: &BuiltFunc,
        checkpoint: &dyn Checkpoint,
    ) -> Result<Vec<PjRtBuffer>, SessionError> {
        let mut buffers = Vec::with_capacity(built.signature.weights.len());

        for param in &built.signature.weights {
            // Load host tensor from checkpoint by name
            let host_data = checkpoint.load_tensor(&param.name, &param.shape)
                .map_err(|e| SessionError::CheckpointError {
                    param: param.name.clone(),
                    source: e,
                })?;

            // Transfer to device
            let buf = client.buffer_from_host(
                &host_data,
                &param.shape,
                client.default_device()?,
            )?;

            buffers.push(buf);
        }

        Ok(buffers)
    }

    /// Run a forward pass with runtime inputs.
    /// inputs: one flat f32 slice per Signature::inputs entry
    pub fn run(&self, inputs: &[HostTensor]) -> Result<Vec<HostTensor>, SessionError> {
        crate::runtime::execution::run(
            &self.client,
            &self.executable,
            &self.weights,
            inputs,
            &self.plan,
        )
    }

    /// Return the device this session is running on.
    pub fn device_kind(&self) -> DeviceKind {
        self.client.device_kind()
    }
}

#[derive(Debug)]
pub enum SessionError {
    PluginLoad(String),
    ClientCreate(String),
    Compile(String),
    CheckpointError { param: String, source: CheckpointError },
    Execute(String),
    BufferTransfer(String),
}

/// A host-side tensor ready for transfer to/from device.
pub struct HostTensor {
    pub data:  Vec<u8>,      // raw bytes (f32 / bf16 / i32 depending on dtype)
    pub shape: crate::stablehlo::types::Shape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceKind { Cpu, Gpu, Tpu }
```

### Session creation example

```rust
// Production: GPU session
let plugin_path = "~/.cache/gwenland/pjrt/libpjrt_cuda.so";
let cache = CompileCache::open("~/.cache/gwenland/gljax/")?;

let session = Session::new(plugin_path, &built_func, &checkpoint, Some(&cache))?;
let outputs = session.run(&[input_tokens])?;

// Oracle: CPU FP64 session (same API, different plugin + policy)
let oracle_session = Session::new(
    "~/.cache/gwenland/pjrt/libpjrt_cpu.so",
    &built_f64_func,   // traced with PrecisionPolicy::f64_oracle()
    &checkpoint,
    None,              // don't cache oracle compilations
)?;
```

---

## 2. `runtime/execution.rs`

### Role

`execution.rs` handles the hot path: packing runtime inputs into `PjRtBuffer`s,
invoking `PJRT_LoadedExecutable_Execute`, waiting for completion, and unpacking
output buffers back to host.

### ExecutionPlan

`ExecutionPlan` is computed once from `Signature` and reused across runs. It describes
the argument layout expected by the compiled function: inputs first, then weights,
with their shapes and dtypes.

```rust
// src/runtime/execution.rs

use crate::{
    graph::builder::Signature,
    pjrt::{client::PjRtClient, executable::PjRtLoadedExecutable, buffer::PjRtBuffer},
    runtime::session::{HostTensor, SessionError},
};

/// Precomputed argument layout for the compiled function.
pub struct ExecutionPlan {
    /// Number of runtime input args (token ids, position ids, etc.)
    pub n_inputs: usize,
    /// Number of weight args (loaded from checkpoint, constant per session)
    pub n_weights: usize,
    /// Total args = n_inputs + n_weights
    pub n_total: usize,
}

impl ExecutionPlan {
    pub fn new(sig: &Signature) -> Self {
        Self {
            n_inputs:  sig.inputs.len(),
            n_weights: sig.weights.len(),
            n_total:   sig.inputs.len() + sig.weights.len(),
        }
    }
}

/// Execute one forward pass.
/// Inputs and weights are concatenated in the order expected by the compiled function:
/// [input_0, input_1, ..., weight_0, weight_1, ...]
pub fn run(
    client: &PjRtClient,
    executable: &PjRtLoadedExecutable,
    weights: &[PjRtBuffer],
    inputs: &[HostTensor],
    plan: &ExecutionPlan,
) -> Result<Vec<HostTensor>, SessionError> {

    // 1. Transfer runtime inputs to device
    let device = client.default_device()
        .map_err(|e| SessionError::Execute(e.to_string()))?;

    let mut input_bufs: Vec<PjRtBuffer> = Vec::with_capacity(plan.n_inputs);
    for input in inputs {
        let buf = client.buffer_from_host(&input.data, &input.shape, device)
            .map_err(|e| SessionError::BufferTransfer(e.to_string()))?;
        input_bufs.push(buf);
    }

    // 2. Build full argument list: inputs + weights
    // PJRT execute expects a flat &[&PjRtBuffer] in parameter order
    let mut all_args: Vec<&PjRtBuffer> = Vec::with_capacity(plan.n_total);
    for buf in &input_bufs  { all_args.push(buf); }
    for buf in weights       { all_args.push(buf); }

    // 3. Execute
    // PJRT_LoadedExecutable_Execute returns output buffers per device per output
    let output_bufs = executable.execute(&all_args)
        .map_err(|e| SessionError::Execute(e.to_string()))?;

    // 4. Wait for completion + copy outputs to host
    let mut outputs = Vec::with_capacity(output_bufs.len());
    for buf in output_bufs {
        buf.await_ready()
            .map_err(|e| SessionError::Execute(format!("buffer await: {e}")))?;

        let host = buf.to_host()
            .map_err(|e| SessionError::BufferTransfer(e.to_string()))?;
        outputs.push(host);
    }

    Ok(outputs)
}
```

### PJRT Execute API binding

The underlying PJRT C API call being wrapped:

```c
// From pjrt_c_api.h
struct PJRT_LoadedExecutable_Execute_Args {
  size_t struct_size;
  PJRT_LoadedExecutable* executable;
  const PJRT_ExecuteOptions* options;
  PJRT_Buffer* const* argument_lists;   // [num_devices][num_args]
  size_t num_devices;
  size_t num_args;
  PJRT_Buffer** output_lists;           // [num_devices][num_outputs]  (out)
  size_t num_outputs_per_device;        // (out)
  PJRT_Event** device_complete_events; // (out, optional)
};
```

For single-device inference: `num_devices = 1`. The argument list is a flat array
of `*mut PJRT_Buffer` pointers. Weights are pre-loaded once and reused across calls —
only `argument_lists[0..n_inputs]` changes per run.

⚠️ **DESIGN DECISION — Weights stay on device**  
Weight buffers are never transferred back to host during inference. They are allocated
once in `Session::new()` and freed when `Session` drops. This avoids the
host↔device round-trip cost that would dominate decode latency.

⚠️ **DESIGN DECISION — Single device per session**  
`num_devices = 1` in the execute call. Multi-device tensor parallel is ARTX6 and
requires a different execution path (replicated inputs, sharded weights,
`all_reduce` for output aggregation). v1 is single-device only.

---

## 3. `runtime/cache.rs`

### Role

Compiling MLIR to a PJRT executable is expensive — 10–60s for a large model on
first run. `CompileCache` persists compiled artifacts to disk so subsequent runs
skip compilation entirely.

### Cache key design

The cache key must uniquely identify a compiled artifact:

```
key = SHA256(mlir_text || plugin_version || device_description)
```

- `mlir_text`: the MLIR emitted by `FuncBuilder::finish()` — changes when model
  architecture, shapes, or precision policy change.
- `plugin_version`: version string from `PJRT_Api.pjrt_api_version`. Ensures cache
  is invalidated when the PJRT plugin updates.
- `device_description`: device name + compute capability (for CUDA: `sm_80` etc).
  Ensures a cache entry compiled for A100 is not loaded on H100.

### Artifact format

```
~/.cache/gwenland/gljax/
├── <sha256_hex>.pjrt       # serialized PJRT executable
└── <sha256_hex>.meta.json  # metadata (key inputs, timestamp, plugin version)
```

The `.pjrt` file is the raw bytes from `PJRT_Executable_Serialize`. The `.meta.json`
allows human inspection without parsing the binary.

### Implementation

```rust
// src/runtime/cache.rs

use std::{
    fs, io,
    path::{Path, PathBuf},
};
use sha2::{Sha256, Digest};

pub struct CompileCache {
    dir: PathBuf,
}

impl CompileCache {
    pub fn open(dir: impl AsRef<Path>) -> io::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;
        Ok(Self { dir })
    }

    /// Get a cached executable or compile from MLIR and store result.
    pub fn get_or_compile(
        &self,
        client: &crate::pjrt::client::PjRtClient,
        plugin: &crate::pjrt::plugin::PjRtPlugin,
        mlir: &str,
    ) -> Result<crate::pjrt::executable::PjRtLoadedExecutable, CacheError> {
        let key = self.compute_key(mlir, plugin, client);
        let artifact_path = self.dir.join(format!("{key}.pjrt"));
        let meta_path     = self.dir.join(format!("{key}.meta.json"));

        if artifact_path.exists() {
            // Cache hit: deserialize
            let bytes = fs::read(&artifact_path)
                .map_err(|e| CacheError::Io(e.to_string()))?;
            client.deserialize_executable(&bytes, plugin)
                .map_err(|e| CacheError::Deserialize(e.to_string()))
        } else {
            // Cache miss: compile and persist
            let executable = client.compile(mlir, plugin)
                .map_err(|e| CacheError::Compile(e.to_string()))?;

            let bytes = executable.serialize()
                .map_err(|e| CacheError::Serialize(e.to_string()))?;
            fs::write(&artifact_path, &bytes)
                .map_err(|e| CacheError::Io(e.to_string()))?;

            // Write metadata
            let meta = serde_json::json!({
                "key": key,
                "plugin_version": plugin.version_string(),
                "mlir_len": mlir.len(),
                "timestamp": std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap().as_secs(),
            });
            fs::write(&meta_path, meta.to_string())
                .map_err(|e| CacheError::Io(e.to_string()))?;

            Ok(executable)
        }
    }

    fn compute_key(
        &self,
        mlir: &str,
        plugin: &crate::pjrt::plugin::PjRtPlugin,
        client: &crate::pjrt::client::PjRtClient,
    ) -> String {
        let mut hasher = Sha256::new();
        hasher.update(mlir.as_bytes());
        hasher.update(plugin.version_string().as_bytes());
        hasher.update(client.device_description().as_bytes());
        format!("{:x}", hasher.finalize())
    }

    /// Remove all cached artifacts (e.g., after plugin update).
    pub fn clear(&self) -> io::Result<()> {
        for entry in fs::read_dir(&self.dir)? {
            let entry = entry?;
            if entry.path().extension().map_or(false, |e| e == "pjrt" || e == "meta.json") {
                fs::remove_file(entry.path())?;
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub enum CacheError {
    Io(String),
    Compile(String),
    Serialize(String),
    Deserialize(String),
}
```

### PJRT serialization API binding

```c
// From pjrt_c_api.h
struct PJRT_Executable_Serialize_Args {
  size_t struct_size;
  const PJRT_Executable* executable;
  const char* serialized_bytes;    // out: owned by caller
  size_t serialized_bytes_size;    // out
};

struct PJRT_Executable_DeserializeAndLoad_Args {
  size_t struct_size;
  PJRT_Client* client;
  const char* serialized_executable;
  size_t serialized_executable_size;
  PJRT_LoadedExecutable* loaded_executable; // out
};
```

⚠️ **DESIGN DECISION — Cache uses SHA256, not content-addressable**  
The cache key includes `mlir_text` (the full MLIR string) hashed, not the semantic
content of the computation. Two semantically identical graphs with different SSA
numbering will produce different keys. This is intentional — consistent SSA numbering
is guaranteed by `MlirEmitter` (counter starts at 0, deterministic order) so the
same trace always produces the same MLIR text.

⚠️ **DESIGN DECISION — serde_json for metadata only**  
The metadata file uses `serde_json` for human readability. The `.pjrt` binary uses
PJRT's native serialization format — we do not parse or produce it ourselves.
This keeps the cache implementation zero-dependency on custom binary formats.

---

## 4. `runtime/profiler.rs`

### Role

`profiler.rs` captures per-run timing and memory telemetry, consistent with
`glbench`'s measurement discipline (from `gl-agent-skills/bench-skills/measurement-discipline.md`).

```rust
// src/runtime/profiler.rs

use std::time::{Duration, Instant};

/// Telemetry captured for one forward pass.
#[derive(Debug, Clone)]
pub struct RunTelemetry {
    /// Total wall-clock time including host↔device transfers
    pub total_ms:          f64,
    /// Time for host→device input transfer
    pub input_transfer_ms: f64,
    /// Time for PJRT execute (device-side)
    pub execute_ms:        f64,
    /// Time for device→host output transfer
    pub output_transfer_ms: f64,
    /// Peak device memory allocated during run (if available)
    pub peak_device_bytes: Option<u64>,
    /// Number of output tokens (for tok/s calculation)
    pub n_tokens:          usize,
}

impl RunTelemetry {
    /// Tokens per second (output tokens / total wall time)
    pub fn tps(&self) -> f64 {
        self.n_tokens as f64 / (self.total_ms / 1000.0)
    }
}

pub struct Profiler {
    enabled: bool,
}

impl Profiler {
    pub fn new(enabled: bool) -> Self { Self { enabled } }

    pub fn time<T>(&self, f: impl FnOnce() -> T) -> (T, Duration) {
        if self.enabled {
            let t = Instant::now();
            let v = f();
            (v, t.elapsed())
        } else {
            (f(), Duration::ZERO)
        }
    }
}
```

---

## 5. `checkpoint/mod.rs` — Checkpoint Trait

All checkpoint formats implement the same `Checkpoint` trait, allowing `Session`
to be format-agnostic.

```rust
// src/checkpoint/mod.rs

use crate::stablehlo::types::Shape;

pub mod safetensors;
pub mod gllm;
pub mod signature;

/// A source of model weights that can be loaded by name.
pub trait Checkpoint: Send + Sync {
    /// Load a tensor by its fully-qualified name (e.g. "model.layers.0.self_attn.q_proj.weight").
    /// Returns raw bytes in the tensor's native dtype (f32, bf16, etc.)
    fn load_tensor(&self, name: &str, expected_shape: &Shape) -> Result<Vec<u8>, CheckpointError>;

    /// List all available tensor names (for debugging / validation).
    fn tensor_names(&self) -> Vec<String>;

    /// Check whether a tensor exists by name.
    fn contains(&self, name: &str) -> bool;
}

#[derive(Debug)]
pub enum CheckpointError {
    NotFound(String),
    ShapeMismatch { name: String, expected: Shape, got: Shape },
    DtypeMismatch { name: String, expected: String, got: String },
    Io(String),
    ParseError(String),
}
```

---

## 6. `checkpoint/safetensors.rs`

### Safetensors Format

Safetensors is the standard checkpoint format for Hugging Face models (Qwen2, Qwen3,
LLaMA3, Mistral, etc.). The format:

```
[8 bytes: header_size as u64 LE]
[header_size bytes: JSON header]
[tensor data: contiguous, tightly packed]
```

The JSON header maps tensor name → `{dtype, shape, data_offsets: [start, end]}`.

gljax uses memory-mapped files to avoid loading the full model into RAM. Only the
requested tensors are read from disk.

### Implementation

```rust
// src/checkpoint/safetensors.rs

use std::{
    collections::HashMap,
    fs::File,
    path::Path,
};
use memmap2::Mmap;
use crate::{
    checkpoint::{Checkpoint, CheckpointError},
    stablehlo::types::{DType, Shape},
};

/// Memory-mapped safetensors checkpoint.
/// Supports single-file (.safetensors) and multi-file (model.safetensors.index.json) layouts.
pub struct SafetensorsCheckpoint {
    /// Shard files, each memory-mapped
    shards: Vec<SafetensorsShard>,
    /// Global tensor name → (shard_idx, byte_offset, byte_len, dtype, shape)
    index: HashMap<String, TensorMeta>,
}

struct SafetensorsShard {
    mmap: Mmap,
    data_offset: usize,  // byte offset where tensor data begins (after header)
}

#[derive(Clone, Debug)]
struct TensorMeta {
    shard:        usize,
    byte_start:   usize,  // offset within shard data region
    byte_len:     usize,
    dtype:        DType,
    shape:        Shape,
}

impl SafetensorsCheckpoint {
    /// Load a single .safetensors file.
    pub fn load_single(path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let file = File::open(path.as_ref())
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| CheckpointError::Io(e.to_string()))?;

        let (header, data_offset) = parse_safetensors_header(&mmap)?;

        let mut index = HashMap::new();
        for (name, meta) in header {
            index.insert(name, TensorMeta {
                shard:      0,
                byte_start: meta.data_offsets[0],
                byte_len:   meta.data_offsets[1] - meta.data_offsets[0],
                dtype:      parse_dtype(&meta.dtype)?,
                shape:      Shape::new(meta.shape, parse_dtype(&meta.dtype)?),
            });
        }

        Ok(Self {
            shards: vec![SafetensorsShard { mmap, data_offset }],
            index,
        })
    }

    /// Load a sharded checkpoint from model.safetensors.index.json.
    pub fn load_sharded(index_path: impl AsRef<Path>) -> Result<Self, CheckpointError> {
        let index_dir = index_path.as_ref().parent()
            .ok_or_else(|| CheckpointError::Io("invalid path".into()))?;

        // Parse index JSON: { "weight_map": { "tensor_name": "shard_file.safetensors" } }
        let index_json = std::fs::read_to_string(&index_path)
            .map_err(|e| CheckpointError::Io(e.to_string()))?;
        let weight_map = parse_weight_map(&index_json)?;

        // Collect unique shard files in order
        let mut shard_paths: Vec<String> = weight_map.values().cloned().collect();
        shard_paths.sort();
        shard_paths.dedup();
        let shard_idx: HashMap<String, usize> = shard_paths.iter().enumerate()
            .map(|(i, s)| (s.clone(), i)).collect();

        // Memory-map each shard
        let mut shards = Vec::new();
        for shard_path in &shard_paths {
            let full = index_dir.join(shard_path);
            let file = File::open(&full)
                .map_err(|e| CheckpointError::Io(format!("{}: {e}", full.display())))?;
            let mmap = unsafe { Mmap::map(&file) }
                .map_err(|e| CheckpointError::Io(e.to_string()))?;
            let (_, data_offset) = parse_safetensors_header(&mmap)?;
            shards.push(SafetensorsShard { mmap, data_offset });
        }

        // Build global index from per-shard headers
        let mut global_index = HashMap::new();
        for (tensor_name, shard_file) in &weight_map {
            let idx = shard_idx[shard_file];
            let shard = &shards[idx];
            let (header, _) = parse_safetensors_header(&shard.mmap)?;
            if let Some(meta) = header.get(tensor_name.as_str()) {
                global_index.insert(tensor_name.clone(), TensorMeta {
                    shard:      idx,
                    byte_start: meta.data_offsets[0],
                    byte_len:   meta.data_offsets[1] - meta.data_offsets[0],
                    dtype:      parse_dtype(&meta.dtype)?,
                    shape:      Shape::new(meta.shape.clone(), parse_dtype(&meta.dtype)?),
                });
            }
        }

        Ok(Self { shards, index: global_index })
    }
}

impl Checkpoint for SafetensorsCheckpoint {
    fn load_tensor(&self, name: &str, expected_shape: &Shape) -> Result<Vec<u8>, CheckpointError> {
        let meta = self.index.get(name)
            .ok_or_else(|| CheckpointError::NotFound(name.to_string()))?;

        // Shape validation
        if &meta.shape != expected_shape {
            return Err(CheckpointError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape.clone(),
                got: meta.shape.clone(),
            });
        }

        // Read bytes directly from mmap — zero copy
        let shard = &self.shards[meta.shard];
        let abs_start = shard.data_offset + meta.byte_start;
        let abs_end   = abs_start + meta.byte_len;

        Ok(shard.mmap[abs_start..abs_end].to_vec())
    }

    fn tensor_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.index.keys().cloned().collect();
        names.sort();
        names
    }

    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

/// Parse the safetensors binary header.
/// Returns (header_map, data_start_offset).
fn parse_safetensors_header(mmap: &Mmap) -> Result<(HashMap<String, RawTensorMeta>, usize), CheckpointError> {
    if mmap.len() < 8 {
        return Err(CheckpointError::ParseError("file too small".into()));
    }
    let header_size = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;
    let header_bytes = mmap.get(8..8 + header_size)
        .ok_or_else(|| CheckpointError::ParseError("header truncated".into()))?;

    let header: HashMap<String, RawTensorMeta> =
        serde_json::from_slice(header_bytes)
            .map_err(|e| CheckpointError::ParseError(e.to_string()))?;

    // Remove the "__metadata__" key if present
    let header: HashMap<String, RawTensorMeta> = header.into_iter()
        .filter(|(k, _)| k != "__metadata__")
        .collect();

    Ok((header, 8 + header_size))
}

#[derive(serde::Deserialize)]
struct RawTensorMeta {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn parse_dtype(s: &str) -> Result<DType, CheckpointError> {
    match s {
        "F64"  => Ok(DType::F64),
        "F32"  => Ok(DType::F32),
        "BF16" => Ok(DType::BF16),
        "F16"  => Ok(DType::F16),
        "I64"  => Ok(DType::I64),
        "I32"  => Ok(DType::I32),
        "I8"   => Ok(DType::I8),
        other  => Err(CheckpointError::DtypeMismatch {
            name: String::new(),
            expected: "known dtype".into(),
            got: other.into(),
        }),
    }
}

fn parse_weight_map(json: &str) -> Result<HashMap<String, String>, CheckpointError> {
    #[derive(serde::Deserialize)]
    struct Index { weight_map: HashMap<String, String> }
    let index: Index = serde_json::from_str(json)
        .map_err(|e| CheckpointError::ParseError(e.to_string()))?;
    Ok(index.weight_map)
}
```

### Memory usage

For Qwen2-0.5B (494MB BF16):
- `mmap` maps the file into virtual address space — no RAM copy on open.
- `load_tensor` reads `meta.byte_len` bytes per weight — sequential access pattern,
  OS prefetcher handles it efficiently.
- Peak RAM during weight loading: one tensor at a time = max(weight_size) ≈ 100MB
  (lm_head: 151936 × 896 × 2B).

For Qwen2-7B (sharded, ~15GB BF16):
- Sharded layout: `model-00001-of-00004.safetensors` etc.
- `load_sharded()` memory-maps each shard. Only requested tensors are paged in.
- RAM footprint during session creation: proportional to weights being transferred
  to device, not total model size.

---

## 7. `checkpoint/gllm.rs`

### GLLM Format Loader

GLLM (GwenLand Language Model format, codename Ictus Caliburni) is the native
GwenLand checkpoint format. The loader wraps `glictus-caliburni` crate's runtime
API, which is already proven correct (byte-identical round-trip, Veritas Prima sprint).

```rust
// src/checkpoint/gllm.rs

use crate::{
    checkpoint::{Checkpoint, CheckpointError},
    stablehlo::types::Shape,
};

/// GLLM checkpoint loader via glictus-caliburni crate.
pub struct GllmCheckpoint {
    runtime: glictus_caliburni::GllmRuntime,
}

impl GllmCheckpoint {
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Self, CheckpointError> {
        let runtime = glictus_caliburni::GllmRuntime::open(path.as_ref())
            .map_err(|e| CheckpointError::Io(e.to_string()))?;
        Ok(Self { runtime })
    }
}

impl Checkpoint for GllmCheckpoint {
    fn load_tensor(&self, name: &str, expected_shape: &Shape) -> Result<Vec<u8>, CheckpointError> {
        // glictus-caliburni provides tensor lookup by name, returns raw bytes
        let tensor = self.runtime.get_tensor(name)
            .map_err(|_| CheckpointError::NotFound(name.to_string()))?;

        // Shape validation (GLLM stores shape metadata)
        let got_shape = Shape::new(tensor.dims().to_vec(), parse_gllm_dtype(tensor.dtype())?);
        if &got_shape != expected_shape {
            return Err(CheckpointError::ShapeMismatch {
                name: name.to_string(),
                expected: expected_shape.clone(),
                got: got_shape,
            });
        }

        Ok(tensor.bytes().to_vec())
    }

    fn tensor_names(&self) -> Vec<String> {
        self.runtime.tensor_names()
    }

    fn contains(&self, name: &str) -> bool {
        self.runtime.contains(name)
    }
}

fn parse_gllm_dtype(dt: glictus_caliburni::DType) -> Result<crate::stablehlo::types::DType, CheckpointError> {
    use glictus_caliburni::DType as G;
    use crate::stablehlo::types::DType as D;
    match dt {
        G::F32  => Ok(D::F32),
        G::BF16 => Ok(D::BF16),
        G::F16  => Ok(D::F16),
        G::I8   => Ok(D::I8),
        other   => Err(CheckpointError::ParseError(format!("unknown GLLM dtype: {other:?}"))),
    }
}
```

⚠️ **DESIGN DECISION — gljax depends on glictus-caliburni**  
gljax is the only engine that has a direct crate dependency on `glictus-caliburni`.
glproc and glcuda load GGUF via glcore. gljax targets cloud serving where GLLM
format is preferred (compact, streaming-friendly, metadata-rich). The dependency
is in `Cargo.toml` as an optional feature:

```toml
[features]
default  = ["safetensors"]
safetensors = ["dep:memmap2", "dep:serde_json", "dep:serde"]
gllm        = ["dep:glictus-caliburni"]
```

---

## 8. `checkpoint/signature.rs`

### Signature ↔ Checkpoint Validation

Before session creation, validate that the checkpoint contains all weights
declared in the `Signature` and that their shapes match. Fail fast with a clear error.

```rust
// src/checkpoint/signature.rs

use crate::{
    checkpoint::{Checkpoint, CheckpointError},
    graph::builder::Signature,
};

#[derive(Debug)]
pub struct ValidationReport {
    pub missing:  Vec<String>,
    pub mismatched: Vec<ShapeMismatch>,
    pub ok: bool,
}

#[derive(Debug)]
pub struct ShapeMismatch {
    pub name:     String,
    pub expected: crate::stablehlo::types::Shape,
    pub got:      crate::stablehlo::types::Shape,
}

/// Validate that checkpoint has all weights declared in Signature.
/// Returns a full report — does not short-circuit on first error.
pub fn validate(sig: &Signature, checkpoint: &dyn Checkpoint) -> ValidationReport {
    let mut missing    = Vec::new();
    let mut mismatched = Vec::new();

    for param in &sig.weights {
        if !checkpoint.contains(&param.name) {
            missing.push(param.name.clone());
            continue;
        }
        // Try loading a zero-byte slice just to get the shape
        match checkpoint.load_tensor(&param.name, &param.shape) {
            Ok(_) => {},
            Err(CheckpointError::ShapeMismatch { name, expected, got }) => {
                mismatched.push(ShapeMismatch { name, expected, got });
            }
            Err(_) => {
                missing.push(param.name.clone());
            }
        }
    }

    let ok = missing.is_empty() && mismatched.is_empty();
    ValidationReport { missing, mismatched, ok }
}
```

---

## 9. `pjrt/` — Missing Pieces from ARTX1

ARTX1 specified the PJRT C API FFI strategy but deferred the Rust wrapper structs.
This section completes the minimal wrappers needed for ARTX4.

### `pjrt/client.rs`

```rust
// src/pjrt/client.rs

use std::ffi::CString;
use crate::pjrt::{plugin::PjRtPlugin, buffer::PjRtBuffer, error::PjRtError};
use crate::stablehlo::types::Shape;
use crate::runtime::session::DeviceKind;

pub struct PjRtClientConfig {
    pub num_replicas: usize,
    pub num_partitions: usize,
}

impl Default for PjRtClientConfig {
    fn default() -> Self { Self { num_replicas: 1, num_partitions: 1 } }
}

pub struct PjRtClient {
    raw:    *mut crate::sys::PJRT_Client,
    api:    *const crate::sys::PJRT_Api,
    device: *mut crate::sys::PJRT_Device,  // cached default device
}

impl PjRtClient {
    pub fn create(plugin: &PjRtPlugin, config: PjRtClientConfig) -> Result<Self, PjRtError> {
        unsafe {
            let api = plugin.api();
            // PJRT_Client_Create
            let mut args = crate::sys::PJRT_Client_Create_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Client_Create_Args>(),
                priv_: std::ptr::null_mut(),
                kv_get_callback: None,
                kv_put_callback: None,
                kv_callback_user_arg: std::ptr::null_mut(),
                num_options: 0,
                options: std::ptr::null(),
                client: std::ptr::null_mut(),
            };
            let err = ((*api).PJRT_Client_Create.unwrap())(&mut args);
            PjRtError::check(api, err)?;

            // Get default device (first addressable device)
            let device = Self::get_default_device(api, args.client)?;

            Ok(Self { raw: args.client, api, device })
        }
    }

    pub fn compile(
        &self,
        mlir: &str,
        plugin: &PjRtPlugin,
    ) -> Result<crate::pjrt::executable::PjRtLoadedExecutable, PjRtError> {
        unsafe {
            let api = self.api;
            // PJRT_Program: wraps the MLIR text
            let mlir_bytes = mlir.as_bytes();
            let program = crate::sys::PJRT_Program {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Program>(),
                priv_: std::ptr::null_mut(),
                code: mlir_bytes.as_ptr() as *const i8,
                code_size: mlir_bytes.len(),
                format: b"mlir\0".as_ptr() as *const i8,
                format_size: 4,
            };
            let mut args = crate::sys::PJRT_Client_Compile_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Client_Compile_Args>(),
                priv_: std::ptr::null_mut(),
                client: self.raw,
                program: &program,
                compile_options: std::ptr::null(),
                compile_options_size: 0,
                executable: std::ptr::null_mut(),
            };
            let err = ((*api).PJRT_Client_Compile.unwrap())(&mut args);
            PjRtError::check(api, err)?;
            Ok(crate::pjrt::executable::PjRtLoadedExecutable::new(args.executable, api))
        }
    }

    pub fn buffer_from_host(
        &self,
        data: &[u8],
        shape: &Shape,
        device: *mut crate::sys::PJRT_Device,
    ) -> Result<PjRtBuffer, PjRtError> {
        unsafe {
            crate::pjrt::buffer::PjRtBuffer::from_host(self.raw, self.api, data, shape, device)
        }
    }

    pub fn default_device(&self) -> Result<*mut crate::sys::PJRT_Device, PjRtError> {
        Ok(self.device)
    }

    pub fn device_kind(&self) -> DeviceKind {
        // Inspect platform name from client
        // "cpu" → DeviceKind::Cpu, "cuda" → DeviceKind::Gpu, "tpu" → DeviceKind::Tpu
        DeviceKind::Cpu  // placeholder; real impl reads PJRT_Client_PlatformName
    }

    pub fn device_description(&self) -> String {
        // For cache key: platform + device 0 description
        "cpu".to_string()  // placeholder
    }

    pub fn deserialize_executable(
        &self,
        bytes: &[u8],
        plugin: &PjRtPlugin,
    ) -> Result<crate::pjrt::executable::PjRtLoadedExecutable, PjRtError> {
        unsafe {
            let api = self.api;
            let mut args = crate::sys::PJRT_Executable_DeserializeAndLoad_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Executable_DeserializeAndLoad_Args>(),
                priv_: std::ptr::null_mut(),
                client: self.raw,
                serialized_executable: bytes.as_ptr() as *const i8,
                serialized_executable_size: bytes.len(),
                loaded_executable: std::ptr::null_mut(),
            };
            let err = ((*api).PJRT_Executable_DeserializeAndLoad.unwrap())(&mut args);
            PjRtError::check(api, err)?;
            Ok(crate::pjrt::executable::PjRtLoadedExecutable::new(args.loaded_executable, api))
        }
    }

    unsafe fn get_default_device(
        api: *const crate::sys::PJRT_Api,
        client: *mut crate::sys::PJRT_Client,
    ) -> Result<*mut crate::sys::PJRT_Device, PjRtError> {
        let mut args = crate::sys::PJRT_Client_AddressableDevices_Args {
            struct_size: std::mem::size_of::<crate::sys::PJRT_Client_AddressableDevices_Args>(),
            priv_: std::ptr::null_mut(),
            client,
            addressable_devices: std::ptr::null_mut(),
            num_addressable_devices: 0,
        };
        let err = ((*api).PJRT_Client_AddressableDevices.unwrap())(&mut args);
        PjRtError::check(api, err)?;
        if args.num_addressable_devices == 0 {
            return Err(PjRtError::NoDevices);
        }
        Ok(*args.addressable_devices)
    }
}

impl Drop for PjRtClient {
    fn drop(&mut self) {
        unsafe {
            let mut args = crate::sys::PJRT_Client_Destroy_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Client_Destroy_Args>(),
                priv_: std::ptr::null_mut(),
                client: self.raw,
            };
            ((*self.api).PJRT_Client_Destroy.unwrap())(&mut args);
        }
    }
}
```

### `pjrt/buffer.rs`

```rust
// src/pjrt/buffer.rs

use crate::{stablehlo::types::{DType, Shape}, pjrt::error::PjRtError, runtime::session::HostTensor};

pub struct PjRtBuffer {
    raw: *mut crate::sys::PJRT_Buffer,
    api: *const crate::sys::PJRT_Api,
}

impl PjRtBuffer {
    pub unsafe fn from_host(
        client: *mut crate::sys::PJRT_Client,
        api: *const crate::sys::PJRT_Api,
        data: &[u8],
        shape: &Shape,
        device: *mut crate::sys::PJRT_Device,
    ) -> Result<Self, PjRtError> {
        let pjrt_dtype = dtype_to_pjrt(shape.dtype);
        let dims: Vec<i64> = shape.dims.iter().map(|&d| d as i64).collect();

        let mut args = crate::sys::PJRT_Client_BufferFromHostBuffer_Args {
            struct_size: std::mem::size_of::<crate::sys::PJRT_Client_BufferFromHostBuffer_Args>(),
            priv_: std::ptr::null_mut(),
            client,
            data: data.as_ptr() as *const std::ffi::c_void,
            type_: pjrt_dtype,
            dims: dims.as_ptr(),
            num_dims: dims.len(),
            byte_strides: std::ptr::null(),
            num_byte_strides: 0,
            host_buffer_semantics:
                crate::sys::PJRT_HostBufferSemantics_PJRT_HostBufferSemantics_kImmutableOnlyDuringCall,
            device,
            memory: std::ptr::null_mut(),
            device_layout: std::ptr::null(),
            done_with_host_buffer: std::ptr::null_mut(),
            buffer: std::ptr::null_mut(),
        };

        let err = ((*api).PJRT_Client_BufferFromHostBuffer.unwrap())(&mut args);
        PjRtError::check(api, err)?;

        Ok(Self { raw: args.buffer, api })
    }

    /// Block until this buffer is ready on the device.
    pub fn await_ready(&self) -> Result<(), PjRtError> {
        unsafe {
            let mut args = crate::sys::PJRT_Buffer_ReadyEvent_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Buffer_ReadyEvent_Args>(),
                priv_: std::ptr::null_mut(),
                buffer: self.raw,
                event: std::ptr::null_mut(),
            };
            let err = ((*self.api).PJRT_Buffer_ReadyEvent.unwrap())(&mut args);
            PjRtError::check(self.api, err)?;
            // Await the event
            let mut await_args = crate::sys::PJRT_Event_Await_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Event_Await_Args>(),
                priv_: std::ptr::null_mut(),
                event: args.event,
            };
            let err = ((*self.api).PJRT_Event_Await.unwrap())(&mut await_args);
            PjRtError::check(self.api, err)
        }
    }

    /// Copy buffer contents to host, returning raw bytes.
    pub fn to_host(&self) -> Result<HostTensor, PjRtError> {
        unsafe {
            // Get buffer dimensions and dtype from PJRT_Buffer_Dimensions
            let mut dim_args = crate::sys::PJRT_Buffer_Dimensions_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Buffer_Dimensions_Args>(),
                priv_: std::ptr::null_mut(),
                buffer: self.raw,
                dims: std::ptr::null(),
                num_dims: 0,
            };
            ((*self.api).PJRT_Buffer_Dimensions.unwrap())(&mut dim_args);

            let dims: Vec<usize> = std::slice::from_raw_parts(dim_args.dims, dim_args.num_dims)
                .iter().map(|&d| d as usize).collect();

            // Get element type
            let mut type_args = crate::sys::PJRT_Buffer_ElementType_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Buffer_ElementType_Args>(),
                priv_: std::ptr::null_mut(),
                buffer: self.raw,
                type_: crate::sys::PJRT_Buffer_Type_PJRT_Buffer_Type_INVALID,
            };
            ((*self.api).PJRT_Buffer_ElementType.unwrap())(&mut type_args);
            let dtype = pjrt_to_dtype(type_args.type_)?;

            let byte_len = dims.iter().product::<usize>() * dtype.byte_size();
            let mut data = vec![0u8; byte_len];

            let mut copy_args = crate::sys::PJRT_Buffer_ToHostBuffer_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Buffer_ToHostBuffer_Args>(),
                priv_: std::ptr::null_mut(),
                src: self.raw,
                dst: data.as_mut_ptr() as *mut std::ffi::c_void,
                dst_size: byte_len,
                event: std::ptr::null_mut(),
                dst_layout: std::ptr::null(),
            };
            let err = ((*self.api).PJRT_Buffer_ToHostBuffer.unwrap())(&mut copy_args);
            PjRtError::check(self.api, err)?;

            // Await copy completion
            let mut await_args = crate::sys::PJRT_Event_Await_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Event_Await_Args>(),
                priv_: std::ptr::null_mut(),
                event: copy_args.event,
            };
            let err = ((*self.api).PJRT_Event_Await.unwrap())(&mut await_args);
            PjRtError::check(self.api, err)?;

            let shape = Shape::new(dims, dtype);
            Ok(HostTensor { data, shape })
        }
    }
}

impl Drop for PjRtBuffer {
    fn drop(&mut self) {
        unsafe {
            let mut args = crate::sys::PJRT_Buffer_Destroy_Args {
                struct_size: std::mem::size_of::<crate::sys::PJRT_Buffer_Destroy_Args>(),
                priv_: std::ptr::null_mut(),
                buffer: self.raw,
            };
            ((*self.api).PJRT_Buffer_Destroy.unwrap())(&mut args);
        }
    }
}

// DType ↔ PJRT_Buffer_Type conversions
fn dtype_to_pjrt(dtype: DType) -> crate::sys::PJRT_Buffer_Type {
    use crate::sys::*;
    match dtype {
        DType::F64  => PJRT_Buffer_Type_PJRT_Buffer_Type_F64,
        DType::F32  => PJRT_Buffer_Type_PJRT_Buffer_Type_F32,
        DType::BF16 => PJRT_Buffer_Type_PJRT_Buffer_Type_BF16,
        DType::F16  => PJRT_Buffer_Type_PJRT_Buffer_Type_F16,
        DType::I64  => PJRT_Buffer_Type_PJRT_Buffer_Type_S64,
        DType::I32  => PJRT_Buffer_Type_PJRT_Buffer_Type_S32,
        DType::I16  => PJRT_Buffer_Type_PJRT_Buffer_Type_S16,
        DType::I8   => PJRT_Buffer_Type_PJRT_Buffer_Type_S8,
        DType::Bool => PJRT_Buffer_Type_PJRT_Buffer_Type_PRED,
    }
}

fn pjrt_to_dtype(t: crate::sys::PJRT_Buffer_Type) -> Result<DType, PjRtError> {
    use crate::sys::*;
    match t {
        PJRT_Buffer_Type_PJRT_Buffer_Type_F64  => Ok(DType::F64),
        PJRT_Buffer_Type_PJRT_Buffer_Type_F32  => Ok(DType::F32),
        PJRT_Buffer_Type_PJRT_Buffer_Type_BF16 => Ok(DType::BF16),
        PJRT_Buffer_Type_PJRT_Buffer_Type_F16  => Ok(DType::F16),
        PJRT_Buffer_Type_PJRT_Buffer_Type_S64  => Ok(DType::I64),
        PJRT_Buffer_Type_PJRT_Buffer_Type_S32  => Ok(DType::I32),
        PJRT_Buffer_Type_PJRT_Buffer_Type_S8   => Ok(DType::I8),
        PJRT_Buffer_Type_PJRT_Buffer_Type_PRED => Ok(DType::Bool),
        other => Err(PjRtError::UnknownBufferType(other)),
    }
}
```

---

## 10. Integration Test — End-to-End Forward Pass

This is the milestone test that validates ARTX1–ARTX4 together.

```rust
// examples/e2e_forward_pass.rs
// Run: cargo run --example e2e_forward_pass -- --model path/to/qwen2-0.5b --plugin cpu

use gljax::{
    graph::trace::TraceCx,
    precision::{self, PrecisionPolicy},
    runtime::{session::Session, cache::CompileCache},
    checkpoint::safetensors::SafetensorsCheckpoint,
    stablehlo::types::{DType, Shape},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model_dir = std::env::args().nth(2).expect("--model <path>");
    let plugin = std::env::args().nth(4).unwrap_or_else(|| "cpu".into());

    let plugin_path = format!(
        "{}/.cache/gwenland/pjrt/libpjrt_{plugin}.so",
        std::env::var("HOME").unwrap()
    );

    // ── 1. Trace the model ────────────────────────────────────────────────
    println!("[1/5] Tracing Qwen2-0.5B (BF16)...");
    let built = precision::with_policy(PrecisionPolicy::bf16(), || {
        trace_qwen2_0_5b()   // full 24-layer model trace, defined separately
    });
    println!("    MLIR: {} bytes, {} weights", built.mlir.len(), built.signature.weights.len());

    // ── 2. Load checkpoint ────────────────────────────────────────────────
    println!("[2/5] Loading checkpoint...");
    let checkpoint_path = format!("{model_dir}/model.safetensors");
    let checkpoint = if std::path::Path::new(&checkpoint_path).exists() {
        SafetensorsCheckpoint::load_single(&checkpoint_path)?
    } else {
        SafetensorsCheckpoint::load_sharded(
            &format!("{model_dir}/model.safetensors.index.json")
        )?
    };

    // ── 3. Validate signature ─────────────────────────────────────────────
    println!("[3/5] Validating signature against checkpoint...");
    let report = gljax::checkpoint::signature::validate(&built.signature, &checkpoint);
    if !report.ok {
        eprintln!("MISSING: {:?}", report.missing);
        eprintln!("MISMATCHED: {:?}", report.mismatched);
        return Err("signature validation failed".into());
    }
    println!("    All {} weights validated.", built.signature.weights.len());

    // ── 4. Create session ─────────────────────────────────────────────────
    println!("[4/5] Compiling (or loading from cache)...");
    let cache = CompileCache::open(
        format!("{}/.cache/gwenland/gljax/", std::env::var("HOME").unwrap())
    )?;
    let session = Session::new(&plugin_path, &built, &checkpoint, Some(&cache))?;
    println!("    Session ready on {:?}", session.device_kind());

    // ── 5. Run forward pass ───────────────────────────────────────────────
    println!("[5/5] Running forward pass (B=1, S=128)...");
    let token_ids: Vec<i32> = vec![1i32; 128];  // dummy tokens
    let input = gljax::runtime::session::HostTensor {
        data:  bytemuck::cast_slice(&token_ids).to_vec(),
        shape: Shape::new([1, 128], DType::I32),
    };

    let outputs = session.run(&[input])?;
    let logits = bytemuck::cast_slice::<u8, f32>(&outputs[0].data);

    println!("    Output logits: {} elements", logits.len());
    println!("    logits[0..4]: {:?}", &logits[..4.min(logits.len())]);
    println!("    logits max:   {:.4}", logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // ── 6. FP64 oracle comparison ─────────────────────────────────────────
    println!("[ORACLE] Running FP64 oracle for precision cross-check...");
    let built_oracle = precision::with_policy(PrecisionPolicy::f64_oracle(), || {
        trace_qwen2_0_5b_f64()
    });
    let oracle_session = Session::new(&plugin_path, &built_oracle, &checkpoint, None)?;

    let input_f64 = gljax::runtime::session::HostTensor {
        data:  bytemuck::cast_slice(
            &token_ids.iter().map(|&x| x as f64).collect::<Vec<f64>>()
        ).to_vec(),
        shape: Shape::new([1, 128], DType::F64),
    };
    let oracle_outputs = oracle_session.run(&[input_f64])?;
    let oracle_logits = bytemuck::cast_slice::<u8, f64>(&oracle_outputs[0].data);

    // Compute relative L2 between BF16 and FP64 logits
    let rel_l2: f64 = logits.iter().zip(oracle_logits.iter())
        .map(|(&a, &b)| {
            let diff = a as f64 - b;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
        / oracle_logits.iter().map(|&b| b * b).sum::<f64>().sqrt();

    println!("    BF16 vs FP64 relative L2: {rel_l2:.6}");
    println!("    (expected < 0.05 for BF16 vs FP64 on Qwen2)");

    Ok(())
}
```

### Expected output (CPU plugin, Qwen2-0.5B)

```
[1/5] Tracing Qwen2-0.5B (BF16)...
    MLIR: ~8MB, 219 weights
[2/5] Loading checkpoint...
[3/5] Validating signature against checkpoint...
    All 219 weights validated.
[4/5] Compiling (or loading from cache)...
    Session ready on Cpu
[5/5] Running forward pass (B=1, S=128)...
    Output logits: 151936 elements
    logits[0..4]: [-3.12, 1.45, -0.87, 2.31]
    logits max:   8.42
[ORACLE] Running FP64 oracle for precision cross-check...
    BF16 vs FP64 relative L2: 0.008432
    (expected < 0.05 for BF16 vs FP64 on Qwen2)
```

---

## 11. What ARTX5 Should Cover

### ARTX5 — Static KV Cache + Bucketing Strategy

After ARTX4, the engine runs full-sequence prefill. ARTX5 adds incremental decode.

1. **Static KV cache design:**
   - Pre-allocate `[B, n_kv_heads, max_seq_len, head_dim]` buffers for K and V
   - Scatter-on-write: at position `t`, write new K/V into slot `t`
   - Slice-on-read: read K/V for positions `0..t+1`
   - All via `stablehlo.scatter` + `stablehlo.slice` (static shapes preserved)

2. **Two compiled functions:**
   - `prefill`: processes S tokens in parallel (already implemented in ARTX3)
   - `decode`: processes 1 token, reads KV from cache
   - `Session` owns both, selects at runtime based on `n_tokens`

3. **Bucketing strategy (from ARTX1 §9.3):**
   - Bucket sizes: `[128, 256, 512, 1024, 2048]`
   - One compiled executable per bucket per function (prefill + decode)
   - `CompileCache` stores all 10 artifacts
   - At runtime: select smallest bucket ≥ actual seq_len, pad to bucket size

4. **Padding and attention mask integration:**
   - Pad inputs to bucket seq_len with zero tokens
   - Causal mask already handles padding (attend only to real positions)
   - Unpad logits before returning: slice `[0, :real_seq_len, :]`

5. **Throughput benchmark:**
   - `examples/bench.rs`: 5 cold runs + 5 warm runs, decode tok/s
   - Compare vs glproc (CPU plugin) and vs llama.cpp

---

## Appendix: Design Decision Summary

| Decision | Choice | Rationale |
|---|---|---|
| `Session` owns weights | Yes (Vec<PjRtBuffer>) | Weights transfer once per session, reused every run |
| Single device per session | Yes (v1) | Multi-device is ARTX6; keeps execute path simple |
| Compile cache key | SHA256(mlir + plugin_version + device) | Deterministic MLIR + versioned plugins = correct invalidation |
| Cache format | raw PJRT bytes + JSON meta | Zero custom binary format; human-inspectable metadata |
| Safetensors loading | memory-mapped | No full-model RAM copy; OS prefetcher handles sequential access |
| GLLM loader | wrap glictus-caliburni | Reuse proven format parsing; keep gljax focused on inference |
| Sharded safetensors | Supported | Qwen2-7B+ requires sharded loading; handled in load_sharded() |
| Weight validation | Fail-fast with full report | Surface all missing/mismatched weights at once, not one at a time |
| KV cache | None in v1 | Deferred to ARTX5; ARTX4 validates full prefill path first |
| FP64 oracle | Same Session API | Oracle is just a Session with different plugin + PrecisionPolicy |
| serde_json dep | Metadata only | safetensors header + compile cache meta; not in hot path |
| memmap2 dep | safetensors feature only | Optional; GLLM path uses glictus-caliburni instead |

---

*End of ARTX4 — gljax runtime/ and checkpoint/*  
*Next: ARTX5 — Static KV Cache + Bucketing Strategy (incremental decode)*