# ARTX6 — gljax Multi-Device Tensor Parallel + MoE Expert Sharding

**Series:** gljax (Sanctum Visibilia) Architecture Research  
**Depends on:** ARTX1–ARTX5 (PJRT FFI, IR, ops/, runtime/, checkpoint/, static KV cache)  
**Next:** ARTX7 — Continuous Batching + Dynamic Sequence Multiplexing

---

## Overview

ARTX5 delivers single-device autoregressive generation with a static KV cache and bucketing strategy.
ARTX6 extends the compute graph to span **multiple devices** via two orthogonal mechanisms:

1. **Tensor Parallelism (TP)** — split individual weight matrices across devices. Each device
   holds a shard and communicates partial results via collective ops (`all_reduce`, `all_gather`).
   No token routing needed; every device sees every token.

2. **Expert Parallelism (EP)** — for MoE models (e.g. Qwen2-57B-A14B, DeepSeek-V2), assign
   entire experts to specific devices. Tokens are routed to the correct device via
   `all_to_all`, computed locally, then gathered back.

Both mechanisms emit **StableHLO collective ops** (`stablehlo.all_reduce`, `stablehlo.all_gather`,
`stablehlo.all_to_all`) inside the same MLIR function that ARTX1–ARTX5 already produce.
PJRT compiles and executes the multi-device program; gljax does not implement any custom
communication runtime.

The core challenge: collective ops require static `replica_groups` at compile time, but
device count and parallelism degree are runtime configuration. Solution: **compile one
program per (TP degree, EP degree) pair**, cache by SHA256 (extends CompileCache from ARTX4),
and select at runtime based on available devices.

---

## What changes from ARTX5

| ARTX5 | ARTX6 |
|---|---|
| Single-device only | Multi-device TP + EP |
| `Tensor` has shape + dtype + value | `Tensor` additionally carries `mesh`, `placement`, `sharding` metadata |
| `MlirEmitter` emits compute ops only | `MlirEmitter` also emits collective ops (`all_reduce`, `all_gather`, `all_to_all`, etc.) |
| `moe_ffn` is a placeholder stub | Full MoE with TopK routing, All-to-All dispatch, expert group placement |
| 10 compiled artifacts (5 buckets × 2) | `10 × TP_degrees × EP_configs` compiled artifacts in CompileCache |
| `Session` takes single device | `Session` takes `DeviceMesh` |

---

## Wave A6.1 — Distributed Foundation

**Goal:** Establish the primitives that Waves A6.2 and A6.3 build on.
No ML logic yet — pure mesh/sharding/collective infrastructure.

### `distributed/mesh.rs`

Represents the logical N-dimensional device grid.

```rust
pub struct DeviceMesh {
    axes: Vec<MeshAxis>,       // ordered axes, e.g. [tp:4, ep:2]
    devices: Vec<PjRtDeviceId>, // flat row-major device list, len = product(axis.size)
    local_device: PjRtDeviceId,
}

impl DeviceMesh {
    pub fn new(axes: Vec<MeshAxis>, devices: Vec<PjRtDeviceId>) -> Result<Self>;
    pub fn shape(&self) -> Vec<usize>;          // [4, 2] for tp=4,ep=2
    pub fn rank(&self) -> usize;                // number of axes (2 here)
    pub fn size(&self) -> usize;                // total devices (8 here)
    pub fn axis(&self, name: &str) -> Option<&MeshAxis>;
    pub fn devices(&self) -> &[PjRtDeviceId];
    pub fn contains(&self, device: PjRtDeviceId) -> bool;
    pub fn coordinates(&self, device: PjRtDeviceId) -> Option<Vec<usize>>; // e.g. [1, 0]
    pub fn local_device(&self) -> PjRtDeviceId;
}
```

**Design note:** `DeviceMesh` is the single source of truth for topology.
All `replica_groups` are derived from it — never hardcoded elsewhere.

### `distributed/axis.rs`

```rust
pub struct MeshAxis {
    name: String,   // "tp", "ep", "dp"
    size: usize,
    index: usize,   // position in mesh.axes
}

impl MeshAxis {
    pub fn new(name: impl Into<String>, size: usize, index: usize) -> Self;
    pub fn name(&self) -> &str;
    pub fn size(&self) -> usize;
    pub fn index(&self) -> usize;
}
```

### `distributed/sharding.rs`

Describes how a tensor is distributed across a mesh axis.

```rust
pub enum ShardingSpec {
    /// Tensor is fully replicated on all devices in the group.
    Replicated,
    /// Tensor is split along `tensor_dim` across `mesh_axis`.
    Tiled { mesh_axis: String, tensor_dim: usize },
    /// Partial result — each device holds a partial sum/max/etc.
    /// Requires a collective to produce the full value.
    Partial { mesh_axis: String, reduction: ReductionKind },
    /// User-managed sharding; gljax emits no automatic collectives.
    Manual,
}

impl ShardingSpec {
    pub fn replicated() -> Self;
    pub fn tiled(mesh_axis: impl Into<String>, tensor_dim: usize) -> Self;
    pub fn partial(mesh_axis: impl Into<String>, reduction: ReductionKind) -> Self;
    pub fn manual() -> Self;

    pub fn is_replicated(&self) -> bool;
    pub fn is_tiled(&self) -> bool;
    pub fn mesh_axes(&self) -> Vec<&str>;
    pub fn validate(&self, mesh: &DeviceMesh, tensor_shape: &Shape) -> Result<()>;
}

pub enum ReductionKind { Sum, Max, Min, Prod }
```

### `distributed/replica_groups.rs`

Translates `DeviceMesh` + `MeshAxis` into the `dense<[[0,1],[2,3]]>` format
that StableHLO collective ops require.

```rust
pub struct ReplicaGroups {
    groups: Vec<Vec<u32>>,   // e.g. [[0,1],[2,3]] for tp=2, ep=2
}

impl ReplicaGroups {
    /// Groups where each group = all devices sharing the same coordinates
    /// on axes OTHER than `axis_name` (i.e. devices that communicate together).
    pub fn new(groups: Vec<Vec<u32>>) -> Self;
    pub fn from_mesh(mesh: &DeviceMesh, axis_name: &str) -> Result<Self>;
    pub fn all(mesh: &DeviceMesh) -> Self;       // one group containing all devices
    pub fn axis(mesh: &DeviceMesh, name: &str) -> Result<Self>;
    pub fn custom(groups: Vec<Vec<u32>>) -> Self;
    pub fn groups(&self) -> &[Vec<u32>];

    /// Emit as MLIR dense integer attribute text:
    /// `dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>`
    pub fn to_mlir_attr(&self) -> String;
}
```

**Example:** `DeviceMesh { tp:2, ep:2 }` with devices `[0,1,2,3]` (row-major):
- `ReplicaGroups::axis(mesh, "tp")` → `[[0,2],[1,3]]` (devices sharing same ep index)
- `ReplicaGroups::axis(mesh, "ep")` → `[[0,1],[2,3]]` (devices sharing same tp index)

### `distributed/collective.rs`

High-level collective emitter. Calls `MlirEmitter` internally — never writes MLIR text directly.

```rust
pub struct CollectiveEmitter<'a> {
    emitter: &'a mut MlirEmitter,
}

impl<'a> CollectiveEmitter<'a> {
    /// all_reduce: sum/max/min partial tensors across replica group.
    pub fn all_reduce(
        &mut self,
        operand: SsaValue,
        groups: &ReplicaGroups,
        reduction: ReductionKind,
    ) -> Result<SsaValue>;

    /// all_gather: concatenate sharded tensors along gather_dim.
    pub fn all_gather(
        &mut self,
        operand: SsaValue,
        groups: &ReplicaGroups,
        gather_dim: usize,
    ) -> Result<SsaValue>;

    /// reduce_scatter: reduce + scatter result shards.
    pub fn reduce_scatter(
        &mut self,
        operand: SsaValue,
        groups: &ReplicaGroups,
        scatter_dim: usize,
        reduction: ReductionKind,
    ) -> Result<SsaValue>;

    /// all_to_all: used for MoE token dispatch/combine.
    pub fn all_to_all(
        &mut self,
        operand: SsaValue,
        groups: &ReplicaGroups,
        split_dim: usize,
        concat_dim: usize,
        split_count: usize,
    ) -> Result<SsaValue>;

    /// broadcast: send from rank 0 of group to all others.
    pub fn broadcast(
        &mut self,
        operand: SsaValue,
        groups: &ReplicaGroups,
    ) -> Result<SsaValue>;

    /// collective_permute: pairwise send/recv by source_target_pairs.
    pub fn collective_permute(
        &mut self,
        operand: SsaValue,
        source_target_pairs: &[(u32, u32)],
    ) -> Result<SsaValue>;

    /// barrier: synchronization point across all devices.
    pub fn barrier(&mut self, groups: &ReplicaGroups) -> Result<()>;
}
```

### `distributed/placement.rs`

Combines `DeviceMesh` + `ShardingSpec` into a single placement descriptor attached to `Tensor`.

```rust
pub struct Placement {
    mesh: Arc<DeviceMesh>,
    sharding: ShardingSpec,
}

impl Placement {
    pub fn replicated(mesh: Arc<DeviceMesh>) -> Self;
    pub fn sharded(mesh: Arc<DeviceMesh>, sharding: ShardingSpec) -> Self;
    pub fn manual(mesh: Arc<DeviceMesh>) -> Self;
    pub fn device(device: PjRtDeviceId) -> Self;   // single-device, no mesh
    pub fn mesh(&self) -> &DeviceMesh;
}
```

---

## Wave A6.2 — Tensor Parallelism

**Goal:** Megatron-style column/row parallel linear layers + TP-aware attention and FFN.
Wraps existing ARTX3 ops (`gqa_attention`, `swiglu_ffn`) with sharding + collective emit.

### Megatron-TP primer (why column+row works)

For `Z = XAB` (two linear layers):
- **Column-parallel A**: split `A` along output dim → each device computes `Y_i = X * A_i`
  → **no communication needed** between A and B
- **Row-parallel B**: split `B` along input dim, inputs already sharded → partial `Z_i = Y_i * B_i`
  → **all_reduce(sum)** at the end to get full `Z`

Net result: 1 all_reduce per MLP block, 1 all_reduce per attention block.
Communication volume: `batch * seq_len * hidden_dim` per all_reduce.

### `tp/linear.rs`

```rust
pub struct ColumnParallelLinear {
    mesh_axis: String,    // which axis to shard across (usually "tp")
    gather_output: bool,  // if true, emit all_gather at end (default: false for chained TP)
}

impl ColumnParallelLinear {
    pub fn new(mesh_axis: impl Into<String>, gather_output: bool) -> Self;

    /// Emits: matmul(input, weight_shard) → output_shard
    /// weight dim [in, out] → shard along out → [in, out/tp]
    /// If gather_output: emit all_gather along out dim.
    pub fn emit(
        &self,
        cx: &TraceCx,
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}

pub struct RowParallelLinear {
    mesh_axis: String,
    input_is_parallel: bool,  // if true, skip scatter (input already sharded)
}

impl RowParallelLinear {
    pub fn new(mesh_axis: impl Into<String>, input_is_parallel: bool) -> Self;

    /// Emits: matmul(input_shard, weight_shard) → partial_output
    /// weight dim [in, out] → shard along in → [in/tp, out]
    /// Always emits all_reduce(sum) to produce full output.
    pub fn emit(
        &self,
        cx: &TraceCx,
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}
```

### `tp/attention.rs`

QKV projection = column-parallel (each device handles `n_heads/tp` heads).
Output projection = row-parallel.

```rust
pub struct TensorParallelAttention {
    mesh_axis: String,
    n_heads: usize,
    n_kv_heads: usize,  // GQA: kv heads also sharded, must be divisible by tp
    head_dim: usize,
}

impl TensorParallelAttention {
    pub fn new(
        mesh_axis: impl Into<String>,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Self;

    pub fn emit(
        &self,
        cx: &TraceCx,
        hidden: Tensor,
        weights: &AttentionWeights,   // Wq, Wk, Wv, Wo shards
        kv_cache: &KvCacheSlice,
        position: usize,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}

/// Emits column-parallel QKV projection.
/// Q: [hidden, n_heads*head_dim] → shard → [hidden, n_heads/tp * head_dim]
/// K/V: [hidden, n_kv_heads*head_dim] → shard → [hidden, n_kv_heads/tp * head_dim]
pub struct TensorParallelQKV;
impl TensorParallelQKV {
    pub fn emit(
        cx: &TraceCx,
        hidden: Tensor,
        wq: Tensor, wk: Tensor, wv: Tensor,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<(Tensor, Tensor, Tensor)>;
}

/// Emits row-parallel output projection + all_reduce.
pub struct TensorParallelOutput;
impl TensorParallelOutput {
    pub fn emit(
        cx: &TraceCx,
        attn_out: Tensor,
        wo: Tensor,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}
```

**GQA constraint:** `n_kv_heads` must be divisible by `tp_degree`.
Validated at emit time. If not satisfied → `Err(InvalidSharding)`.

### `tp/ffn.rs`

SwiGLU FFN = w1/w3 column-parallel, w2 row-parallel. Zero all_reduce between w1/w3 and w2.

```rust
pub struct TensorParallelMLP {
    mesh_axis: String,
}

impl TensorParallelMLP {
    pub fn new(mesh_axis: impl Into<String>) -> Self;

    /// Standard dense FFN (2-layer + activation).
    pub fn emit(
        &self,
        cx: &TraceCx,
        hidden: Tensor,
        w_up: Tensor, w_down: Tensor,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}

pub struct TensorParallelSwiGLU {
    mesh_axis: String,
}

impl TensorParallelSwiGLU {
    pub fn new(mesh_axis: impl Into<String>) -> Self;

    /// w1: column-parallel gate proj  [hidden, intermediate/tp]
    /// w3: column-parallel up proj    [hidden, intermediate/tp]
    /// w2: row-parallel  down proj    [intermediate/tp, hidden]
    /// Communication: 1× all_reduce at w2 output.
    pub fn emit(
        &self,
        cx: &TraceCx,
        hidden: Tensor,
        w1: Tensor, w2: Tensor, w3: Tensor,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;
}
```

### `tp/planner.rs` (optional, lightweight)

Not auto-sharding. Just validates that the model config is compatible with
the requested TP degree and centralizes the "what gets sharded how" logic.

```rust
pub struct TensorParallelPlanner {
    tp_degree: usize,
    mesh_axis: String,
}

impl TensorParallelPlanner {
    pub fn plan(&self, model_config: &ModelConfig) -> Result<TpPlan>;
    pub fn validate(&self, model_config: &ModelConfig) -> Result<()>;
}

pub struct TpPlan {
    pub attention_groups: ReplicaGroups,
    pub ffn_groups: ReplicaGroups,
    pub tp_degree: usize,
}
```

---

## Wave A6.3 — MoE Expert Sharding

**Goal:** Replace the `moe_ffn` placeholder stub from ARTX3 with a full
Expert Parallel implementation. Emits `stablehlo.all_to_all` for token dispatch/combine.

### MoE execution flow (per layer)

```
hidden [B, seq, d_model]
    │
    ▼
TopKRouter              → topk_weights [B*seq, top_k]
                        → topk_ids    [B*seq, top_k]   (logical expert IDs)
    │
    ▼
TokenDispatcher::dispatch   → all_to_all → tokens reach the device owning each expert
    │
    ▼
ExpertGroup::lookup         → each device runs its local experts (grouped GEMM)
    │
    ▼
TokenDispatcher::combine    → all_to_all → outputs return to originating device
    │
    ▼
TokenDispatcher::restore_order → reorder + weight-sum → hidden [B, seq, d_model]
```

### `moe/router.rs`

```rust
pub struct TopKRouter {
    n_experts: usize,
    top_k: usize,
    /// Expert capacity factor. capacity = ceil(top_k * n_tokens / n_experts * capacity_factor).
    /// None = no capacity limit (used for decode, batch=1).
    capacity_factor: Option<f32>,
}

impl TopKRouter {
    pub fn new(n_experts: usize, top_k: usize, capacity_factor: Option<f32>) -> Self;

    /// Emits: gate_proj(hidden) → softmax → topk → (weights, ids)
    pub fn route(
        &self,
        cx: &TraceCx,
        hidden: Tensor,      // [B*seq, d_model]
        gate_weight: Tensor, // [d_model, n_experts]
    ) -> Result<RouterOutput>;

    /// Per-device token capacity given current batch size.
    pub fn capacity(&self, n_tokens: usize) -> usize;
}

pub struct RouterOutput {
    pub weights: Tensor,  // [B*seq, top_k], fp32
    pub ids: Tensor,      // [B*seq, top_k], i32 (logical expert IDs)
}
```

### `moe/dispatch.rs`

Handles the All-to-All token routing. Static shapes required by PJRT:
tokens are **padded to capacity** before dispatch.

```rust
pub struct TokenDispatcher {
    n_experts: usize,
    n_devices: usize,  // = ep_degree
    capacity: usize,   // tokens per expert per device (static, compile-time)
    top_k: usize,
}

impl TokenDispatcher {
    /// Phase 1: sort + pad tokens by expert, then all_to_all to target devices.
    /// Input:  hidden [B*seq, d_model], router_output
    /// Output: dispatched [ep_degree, capacity, d_model] (each device's view)
    pub fn dispatch(
        &self,
        cx: &TraceCx,
        hidden: Tensor,
        router: &RouterOutput,
        placement: &ExpertPlacement,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<DispatchedTokens>;

    /// Phase 2: all_to_all results back, restore to [B*seq, d_model].
    pub fn combine(
        &self,
        cx: &TraceCx,
        expert_outputs: Tensor,
        dispatch_meta: &DispatchedTokens,
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;

    /// Reorder + weighted sum of top_k expert outputs per token.
    pub fn restore_order(
        &self,
        cx: &TraceCx,
        combined: Tensor,
        router_weights: Tensor,
        dispatch_meta: &DispatchedTokens,
    ) -> Result<Tensor>;
}

pub struct DispatchedTokens {
    pub token_indices: Tensor,   // original positions, for restore_order
    pub expert_mask: Tensor,     // [n_tokens, n_experts] bool, which tokens went where
    pub capacity: usize,
}
```

### `moe/experts.rs`

```rust
pub struct ExpertGroup {
    n_experts_total: usize,
    n_experts_local: usize,   // = n_experts_total / ep_degree
    expert_dim: usize,        // intermediate size per expert
}

impl ExpertGroup {
    pub fn new(n_experts_total: usize, ep_degree: usize, expert_dim: usize) -> Result<Self>;

    /// Assign experts to this device based on placement strategy.
    pub fn place(&self, device_rank: usize, strategy: &ExpertPlacement) -> Vec<usize>;

    /// Given a logical expert ID, return which device owns it.
    pub fn lookup(&self, expert_id: usize, strategy: &ExpertPlacement) -> usize;

    pub fn count(&self) -> usize;  // n_experts_local
}
```

### `moe/placement.rs`

Distinct from `distributed/placement.rs` — this is about **expert layout strategy**,
not tensor placement on a mesh.

```rust
pub enum ExpertPlacement {
    /// Experts 0..n/ep on device 0, n/ep..2n/ep on device 1, etc.
    Contiguous,
    /// Expert i → device (i % ep_degree). Balances expert load.
    RoundRobin,
    /// Custom mapping: expert_id → device_rank.
    Balanced(HashMap<usize, usize>),
}

impl ExpertPlacement {
    pub fn contiguous(n_experts: usize, ep_degree: usize) -> Self;
    pub fn round_robin(n_experts: usize, ep_degree: usize) -> Self;
    pub fn balanced(mapping: HashMap<usize, usize>) -> Self;

    pub fn device_for(&self, expert_id: usize) -> usize;
    pub fn experts_on(&self, device_rank: usize) -> Vec<usize>;
}
```

**Default:** `Contiguous` for simplicity. `RoundRobin` can be toggled
via `Session` config if expert load imbalance is detected (future work).

### `moe/moe.rs`

Top-level MoE block. Replaces the `moe_ffn` placeholder from ARTX3.

```rust
pub struct MoE {
    router: TopKRouter,
    dispatcher: TokenDispatcher,
    experts: ExpertGroup,
    placement: ExpertPlacement,
    /// ep axis name in the mesh (usually "ep")
    ep_axis: String,
}

impl MoE {
    pub fn new(
        config: &MoeConfig,
        mesh: &DeviceMesh,
        ep_axis: impl Into<String>,
    ) -> Result<Self>;

    /// Full MoE forward pass emit.
    /// Replaces placeholder `moe_ffn` in ops/moe.rs.
    pub fn emit(
        &self,
        cx: &TraceCx,
        hidden: Tensor,         // [B*seq, d_model]
        gate_weight: Tensor,    // [d_model, n_experts]
        expert_weights: Tensor, // [n_experts_local, d_model, expert_dim*2] (SwiGLU)
        groups: &ReplicaGroups,
        collective: &mut CollectiveEmitter,
    ) -> Result<Tensor>;

    pub fn validate(&self, mesh: &DeviceMesh) -> Result<()>;
}
```

---

## Tensor API Extension

Add distributed metadata to `Tensor`. Non-breaking: all fields are `Option`,
default to `None` for single-device tensors from ARTX1–ARTX5.

```rust
pub struct Tensor {
    // existing fields (ARTX2)
    shape: Shape,
    dtype: DType,
    value: SsaValue,
    graph: Rc<RefCell<FuncBuilder>>,

    // NEW (ARTX6)
    mesh: Option<Arc<DeviceMesh>>,
    placement: Option<Placement>,
    sharding: Option<ShardingSpec>,
}

impl Tensor {
    // existing
    pub fn shape(&self) -> &Shape;
    pub fn dtype(&self) -> DType;
    pub fn layout(&self) -> Layout;

    // new
    pub fn mesh(&self) -> Option<&DeviceMesh>;
    pub fn placement(&self) -> Option<&Placement>;
    pub fn sharding(&self) -> Option<&ShardingSpec>;

    pub fn with_mesh(self, mesh: Arc<DeviceMesh>) -> Self;
    pub fn with_sharding(self, sharding: ShardingSpec) -> Self;
    pub fn with_placement(self, placement: Placement) -> Self;
}
```

---

## MlirEmitter Extension

Add collective op emission to `MlirEmitter`. These are called by `CollectiveEmitter` only —
Wave A6.2 and A6.3 code must not call these directly.

```rust
impl MlirEmitter {
    pub fn emit_all_reduce(
        &mut self,
        operand: SsaValue,
        replica_groups_attr: &str, // pre-serialized MLIR dense attr
        reduction: ReductionKind,
        channel_id: u32,
    ) -> Result<SsaValue>;

    pub fn emit_all_gather(
        &mut self,
        operand: SsaValue,
        replica_groups_attr: &str,
        gather_dim: usize,
        channel_id: u32,
    ) -> Result<SsaValue>;

    pub fn emit_reduce_scatter(
        &mut self,
        operand: SsaValue,
        replica_groups_attr: &str,
        scatter_dim: usize,
        reduction: ReductionKind,
        channel_id: u32,
    ) -> Result<SsaValue>;

    pub fn emit_all_to_all(
        &mut self,
        operand: SsaValue,
        replica_groups_attr: &str,
        split_dim: usize,
        concat_dim: usize,
        split_count: usize,
        channel_id: u32,
    ) -> Result<SsaValue>;

    pub fn emit_collective(
        &mut self,
        kind: CollectiveKind,
        operand: SsaValue,
        params: CollectiveParams,
    ) -> Result<SsaValue>;
}
```

**Channel ID discipline:** Each call site must use a unique, monotonically increasing
`channel_id` per compiled function. `MlirEmitter` tracks next available ID internally
via a `channel_counter: u32` field.

---

## StableHLO Collective Ops — Reference Syntax

These are the exact MLIR text patterns that `MlirEmitter` must emit.

### `stablehlo.all_reduce` (sum)
```mlir
%result = "stablehlo.all_reduce"(%operand) ({
  ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
    %sum = stablehlo.add %arg0, %arg1 : tensor<bf16>
    stablehlo.return %sum : tensor<bf16>
}) {
  replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  use_global_device_ids
} : (tensor<4096xbf16>) -> tensor<4096xbf16>
```

### `stablehlo.all_gather` (TP gather along dim 1)
```mlir
%result = "stablehlo.all_gather"(%operand) {
  all_gather_dim = 1 : i64,
  replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>,
  use_global_device_ids
} : (tensor<4096x1024xbf16>) -> tensor<4096x2048xbf16>
```

### `stablehlo.all_to_all` (MoE token dispatch)
```mlir
%result = "stablehlo.all_to_all"(%operand) {
  split_dimension = 0 : i64,
  concat_dimension = 0 : i64,
  split_count = 4 : i64,
  replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>,
  channel_handle = #stablehlo.channel_handle<handle = 3, type = 1>
} : (tensor<64x4096xbf16>) -> tensor<64x4096xbf16>
```

---

## Folder Structure

```
gljax/src/
├── distributed/              ← NEW (Wave A6.1)
│   ├── mod.rs
│   ├── mesh.rs
│   ├── axis.rs
│   ├── sharding.rs
│   ├── replica_groups.rs
│   ├── collective.rs
│   └── placement.rs
│
├── tp/                       ← NEW (Wave A6.2)
│   ├── mod.rs
│   ├── linear.rs
│   ├── attention.rs
│   ├── ffn.rs
│   └── planner.rs
│
├── moe/                      ← NEW (Wave A6.3)
│   ├── mod.rs
│   ├── router.rs
│   ├── dispatch.rs
│   ├── experts.rs
│   ├── placement.rs          ← expert placement (distinct from distributed/placement.rs)
│   └── moe.rs
│
├── tensor/
│   └── mod.rs                ← EXTENDED: add mesh/placement/sharding fields
│
├── stablehlo/
│   ├── emitter.rs            ← EXTENDED: add emit_all_reduce/gather/to_all
│   └── ops.rs
│
├── ops/
│   └── moe.rs                ← UPDATED: replace placeholder with MoE::emit()
│
└── runtime/
    └── session.rs            ← UPDATED: Session takes DeviceMesh, CompileCache
                                         keyed by (bucket, tp_degree, ep_config)
```

---

## Key Design Decisions

### 1. Collective ops emitted at trace time, not runtime
gljax follows the XLA/PJRT model: all collective ops (`all_reduce`, `all_to_all`, etc.)
are part of the compiled StableHLO program. PJRT executes the multi-device program
atomically. There is no separate communication runtime in gljax.

### 2. `distributed/placement.rs` ≠ `moe/placement.rs`
These are intentionally two separate files with different concerns:
- `distributed/placement.rs`: where a **tensor** lives on a mesh (replicated/sharded/manual)
- `moe/placement.rs`: which **device** owns each **expert** (contiguous/round_robin/balanced)

Merging them would conflate tensor-level and expert-level abstractions.

### 3. Static shapes for All-to-All
PJRT requires static shapes at compile time. `TokenDispatcher` pads all token
tensors to `capacity` before dispatch. Capacity = `ceil(top_k * n_tokens / n_experts * factor)`.
For decode (batch=1), `capacity=1` → minimal padding overhead.

### 4. CompileCache key extension
Extend ARTX4's SHA256 key to include:
```
key = sha256(model_weights_hash | bucket_size | tp_degree | ep_degree | ep_strategy)
```
This ensures TP=4/EP=2 and TP=2/EP=4 produce different cached artifacts.

### 5. `CollectiveEmitter` owns `channel_id` assignment
All wave A6.2/A6.3 code goes through `CollectiveEmitter`, never calling
`MlirEmitter::emit_all_reduce` etc. directly. This ensures monotonically
increasing channel IDs across all collectives in a function.

### 6. GQA TP constraint validated early
`n_kv_heads % tp_degree == 0` is validated in `TensorParallelAttention::new()`
(not at emit time) so misconfiguration is caught before any MLIR is generated.

---

## Test Plan

### A6.1 — Distributed Foundation (target: ~25 tests)
```
distributed::mesh::tests::new_valid
distributed::mesh::tests::new_device_count_mismatch
distributed::mesh::tests::coordinates_roundtrip
distributed::mesh::tests::local_device
distributed::axis::tests::new_and_getters
distributed::sharding::tests::replicated
distributed::sharding::tests::tiled_valid
distributed::sharding::tests::tiled_dim_out_of_bounds
distributed::sharding::tests::partial_sum
distributed::sharding::tests::validate_mesh_axis_missing
distributed::replica_groups::tests::from_mesh_tp_axis
distributed::replica_groups::tests::from_mesh_ep_axis
distributed::replica_groups::tests::all_devices
distributed::replica_groups::tests::to_mlir_attr_2x2
distributed::replica_groups::tests::to_mlir_attr_1x4
distributed::collective::tests::emit_all_reduce_sum
distributed::collective::tests::emit_all_gather_dim1
distributed::collective::tests::emit_reduce_scatter
distributed::collective::tests::emit_all_to_all
distributed::collective::tests::channel_ids_monotonic
distributed::placement::tests::replicated
distributed::placement::tests::sharded
distributed::placement::tests::manual
```

### A6.2 — Tensor Parallel (target: ~20 tests)
```
tp::linear::tests::column_parallel_emit_mlir
tp::linear::tests::row_parallel_emit_mlir
tp::linear::tests::column_then_row_no_intermediate_collective
tp::attention::tests::qkv_split_heads
tp::attention::tests::gqa_kv_heads_divisible
tp::attention::tests::gqa_kv_heads_not_divisible_err
tp::attention::tests::output_proj_all_reduce
tp::ffn::tests::swiglu_column_row_pattern
tp::ffn::tests::swiglu_single_all_reduce
tp::planner::tests::validate_compatible_config
tp::planner::tests::validate_incompatible_heads
```

### A6.3 — MoE Expert Sharding (target: ~20 tests)
```
moe::router::tests::topk_route_emit
moe::router::tests::capacity_batch1
moe::router::tests::capacity_batch8
moe::dispatch::tests::dispatch_emit_all_to_all
moe::dispatch::tests::combine_emit_all_to_all
moe::dispatch::tests::restore_order
moe::experts::tests::contiguous_placement
moe::experts::tests::round_robin_placement
moe::experts::tests::lookup_device
moe::placement::tests::contiguous_experts_on
moe::placement::tests::round_robin_experts_on
moe::placement::tests::device_for_contiguous
moe::placement::tests::device_for_round_robin
moe::moe::tests::emit_full_forward
moe::moe::tests::validate_ep_degree_mismatch
```

### Tensor Extension (target: ~5 tests)
```
tensor::tests::with_mesh_roundtrip
tensor::tests::with_sharding_roundtrip
tensor::tests::default_no_mesh
tensor::tests::with_placement_roundtrip
```

**Total target: ~70 new tests** (ARTX1–ARTX5 baseline stays green)