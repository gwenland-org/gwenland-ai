# ARTX16 — gljax Distributed Serving

**Series:** gljax (Sanctum Visibilia) Architecture Research
**Status:** Draft — research-grounded
**Depends on:** ARTX1–ARTX8 (full gljax stack)
**Introduces:** `glserve` crate (new)
**Next:** — end of the ARTX08–ARTX16 arc; deferred items listed in §10
**Research grounded:** 2026-07-27 (sources at end)

---

# Reality Check — read before planning work from this document

Two facts about the current repo shape the sequencing of everything below.

**1. `gljax` has no code.** `gljax/` contains architecture documents only, and `gljax` is **not a
member of the root `Cargo.toml` workspace** (members are `glcore`, `glproc`, `glcuda`, `glvulkan`,
`glmetal`, `glbench`, `glcli`, `glictus-caliburni`, `packages/{core,mcp}`). ARTX1–ARTX8 are design,
not implementation.

**2. `glserve` does not exist.** No crate, no directory, no prior art in-tree.

ARTX16 therefore designs the serving layer for an engine that is not yet built. That is legitimate —
it is what the ARTX series is for — but it means **no wave in this document is startable until
`Session::generate()` from ARTX5 actually runs.** Nothing here is blocked on ARTX16 research; it is
blocked on ARTX1–ARTX5 implementation.

⚠️ **DESIGN DECISION — glserve is a separate crate, not a module of gljax.**
`gljax` stays a library with zero HTTP/async dependencies (no `tokio`, no `axum`, no `hyper`). This
preserves ARTX7's synchronous single-threaded scheduler as the *engine's* execution model, and keeps
gljax embeddable (glcli, glbench, tests) without dragging in a web stack. `glserve` depends on
`gljax`; never the reverse.

**Port 1136 is the established GwenLand convention** — the legacy `packages/tui` serve command, the
Tauri GUI's SSE endpoint, and `general.default_port = 1136` in the config schema all use it.
`glserve` inherits it rather than inventing a new default.

---

# 1. Serving Architecture Overview

## 1.1 The layering

```text
HTTP client (curl / OpenAI SDK / GUI / glbench --http)
      │  POST /v1/chat/completions   {"stream": true}
      ▼
┌──────────────────────────────────────────────────────────┐
│ glserve  (tokio + axum)                                  │
│                                                          │
│  api/chat.rs      parse OpenAI request → InferenceReq    │
│      ▼                                                   │
│  router.rs        pick a replica (§5)                    │
│      ▼                                                   │
│  pool.rs          SessionPool: N worker threads,         │
│                   one per replica, mpsc inbox            │
│      ▼                     ▲ token stream (mpsc)         │
│ ─── thread boundary ───────┼──────────────────────────── │
│      ▼                     │                             │
│  SessionWorker (blocking, one OS thread per replica)     │
│      │                     │                             │
│      ▼                     │                             │
│   ARTX7 Scheduler ─────────┘                             │
│      │  collect / retire / form_batch /                  │
│      │  schedule_decode / schedule_prefill_chunks        │
│      ▼                                                   │
│   gljax Session  (ARTX4/ARTX5)                           │
│      ▼                                                   │
│   PJRT execute                                           │
└──────────────────────────────────────────────────────────┘
```

⚠️ **DESIGN DECISION — the async/sync boundary is a thread, not an async runtime inside the engine.**
ARTX7 locked the scheduler as synchronous and single-threaded, and ARTX7's Non-Goals list "Async
runtime" explicitly. `glserve` honours that: each replica gets **one dedicated OS thread** running a
blocking `SessionWorker` loop. Axum handlers communicate with it over channels only.

```text
axum handler (async)  ──mpsc::Sender<InferenceReq>──►  SessionWorker (blocking)
axum handler (async)  ◄──mpsc::Receiver<TokenEvent>──  SessionWorker (blocking)
```

The alternative — making the engine `async` — would require every PJRT call to be cancel-safe and
would push `tokio` into `gljax`. Rejected.

## 1.2 Request lifecycle

```text
1. HTTP POST arrives                        axum
2. Deserialize + validate                   api/chat.rs
3. Tokenize prompt                          gljax tokenizer (host side)
4. Admission control: queue depth check     router.rs → 429 if full (§5.4)
5. Send InferenceReq to a replica inbox     pool.rs
6. SessionWorker admits into ARTX7 queue    scheduler.collect_new_requests()
7. Slot allocated, prefill chunks scheduled ARTX7 KvSlotManager + policy
8. Each decode iteration emits one token    → TokenEvent on the reply channel
9. Handler wraps each TokenEvent as SSE     stream.rs
10. finish_reason → final chunk → [DONE]    stream.rs
11. Slot freed, telemetry recorded          ARTX7 retire + metrics.rs
```

## 1.3 Core types

```rust
// glserve/src/types.rs

/// Engine-facing request. Deliberately NOT the OpenAI wire type —
/// api/ converts, so the engine never sees HTTP concerns.
pub struct InferenceReq {
    pub id: RequestId,
    pub prompt_tokens: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    /// Where the worker sends tokens back.
    pub reply: mpsc::Sender<TokenEvent>,
    /// Dropped by the handler on client disconnect → worker cancels the request.
    pub cancel: CancelToken,
    pub arrived_at: Instant,
}

pub enum TokenEvent {
    /// First token — carries measured TTFT for telemetry.
    First { token: u32, text: String, ttft: Duration },
    Token { token: u32, text: String },
    Done  { finish_reason: FinishReason, usage: Usage },
    Error { message: String },
}

pub enum FinishReason { Stop, Length, Cancelled, Error }

pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: Option<usize>,
    pub stop: Vec<String>,
    pub seed: Option<u64>,
}
```

## 1.4 OpenAI-compatible endpoints

| Endpoint | Method | Notes |
|---|---|---|
| `/v1/chat/completions` | POST | Primary. `stream: true` → SSE; `false` → single JSON |
| `/v1/completions` | POST | Legacy text completion. Thin shim over the same path |
| `/v1/models` | GET | Lists the loaded model id + `owned_by: "gwenland"` |
| `/health` | GET | Liveness/readiness (§4.2) |
| `/metrics` | GET | Prometheus text exposition (§7) |
| `/debug/oracle` | POST | FP64 oracle, non-production (§6.4) |

## 1.5 SSE streaming

The OpenAI streaming contract is: each event is a line beginning `data: ` containing a
`chat.completion.chunk` JSON object whose `choices[].delta` carries the incremental content; the
stream terminates with the literal `data: [DONE]`.

```rust
// glserve/src/stream.rs
use axum::response::sse::{Event, KeepAlive, Sse};

pub fn stream_chat(
    rx: mpsc::Receiver<TokenEvent>,
    meta: ChunkMeta,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let body = ReceiverStream::new(rx).map(move |ev| {
        let chunk = match ev {
            TokenEvent::First { text, .. } | TokenEvent::Token { text, .. } =>
                ChatChunk::delta(&meta, &text),
            TokenEvent::Done { finish_reason, .. } =>
                ChatChunk::finish(&meta, finish_reason),
            TokenEvent::Error { message } =>
                ChatChunk::error(&meta, &message),
        };
        Ok(Event::default().data(serde_json::to_string(&chunk).unwrap()))
    })
    // OpenAI's terminator is the literal string [DONE], not JSON.
    .chain(stream::once(async { Ok(Event::default().data("[DONE]")) }));

    Sse::new(body).keep_alive(KeepAlive::default())
}
```

⚠️ **DESIGN DECISION — keep-alive is on, default interval.**
A long prefill (ARTX7 chunks a 4096-token prompt into 8 executions) can exceed a proxy's idle
timeout before the first token appears. Axum's `KeepAlive` emits SSE comment frames that OpenAI
clients ignore but proxies count as traffic. Cheap insurance; no protocol change.

⚠️ **DESIGN DECISION — client disconnect cancels the request.**
When the SSE response is dropped, `rx` closes. The `SessionWorker` observes the closed channel on
its next send, marks the request `Cancelled`, and calls `KvSlotManager::free(slot)`. Without this,
an abandoned browser tab holds a KV slot for its full `max_tokens`. ARTX7 already has the
`Cancelled` state and the `free()` path — this is wiring, not new mechanism.

## 1.6 How ARTX7's continuous batching integrates

The integration point is narrow by design. `SessionWorker` is a loop around ARTX7's scheduler:

```rust
// glserve/src/pool.rs — the blocking worker
fn run(mut self) {
    loop {
        // 1. Drain the inbox WITHOUT blocking — new HTTP arrivals.
        while let Ok(req) = self.inbox.try_recv() {
            self.scheduler.push(req.into_artx7_request());
        }

        // 2. One ARTX7 iteration (this is the iteration-level scheduling point).
        self.scheduler.collect_new_requests();
        self.scheduler.retire_finished();
        self.scheduler.form_batch();
        self.scheduler.schedule_decode();
        self.scheduler.schedule_prefill_chunks();
        let step = self.executor.execute();

        // 3. Fan generated tokens back out to their HTTP handlers.
        for (req_id, tok) in step.tokens {
            self.emit(req_id, tok);
        }

        // 4. Nothing to do at all → park briefly rather than spin.
        if self.scheduler.is_idle() && self.inbox.is_empty() {
            self.inbox.recv_timeout(IDLE_PARK).ok().map(|r| {
                self.scheduler.push(r.into_artx7_request())
            });
        }
    }
}
```

⚠️ **DESIGN DECISION — `try_recv` in the hot loop, blocking `recv_timeout` only when idle.**
ARTX7's scheduler is work-conserving: if any request has runnable work the device must stay busy. A
blocking receive in the hot path would stall decode waiting for HTTP. A pure spin would burn a core
when idle. The split gives both properties.

---

# 2. Multi-Node Deployment

## 2.1 ⛔ The real blocker is not pipeline parallelism — it is multi-host PJRT

Before any PP design matters, note what ARTX6 did **not** cover: ARTX6 built `DeviceMesh`, Shardy/SDY
sharding annotations, `stablehlo.all_reduce`, and MoE all-to-all — all of it **within one process,
one PJRT client, one host**.

Multi-node requires a facility ARTX1 never bound: the **PJRT distributed coordination service**. In
the JAX/XLA world this is `jax.distributed.initialize(coordinator_address, num_processes,
process_id)`, which starts a coordination service on process 0 that all other processes connect to.
Its jobs are process discovery, topology exchange, health checking (so all processes shut down if
any one dies), and backing distributed checkpointing. At the PJRT layer this surfaces as a
**key-value store interface** passed into client creation — the C API exposes primitives such as
`PJRT_KeyValueTryGet`, and `PJRT_Client_Create` accepts options carrying node identity.

**What gljax must add before multi-node is possible:**

| Requirement | Where it lands | Status |
|---|---|---|
| KV-store callback FFI (`PJRT_KeyValue{Get,TryGet,Put}` function pointers) | `gljax/src/pjrt/kv_store.rs` (new) | Not designed in ARTX1 |
| `node_id` / `num_nodes` in `PJRT_Client_Create` options | extend ARTX1 §1 client creation | Not designed |
| A coordination service implementation (or a binding to XLA's) | `glserve` or a sidecar | Not designed |
| Global device ordering across hosts | extend ARTX6 `DeviceMesh` | ARTX6 is single-host |

⚠️ **DESIGN DECISION — multi-node is a Wave A9.3, gated behind single-node working end-to-end.**
Sections 2.2–2.5 below specify the design so it is not re-derived later, but the honest sequencing is:
**single-node TP (§9.2) is the production target for gljax v1.** A single 8×H100 node runs a
70B-class model in BF16 comfortably; multi-node is for models that genuinely do not fit.

## 2.2 Why TP within node, PP across nodes

This is the settled industry answer and gljax has no reason to deviate.

* **Tensor parallelism** splits every layer across devices, so *every* layer boundary requires an
  `AllReduce`. That is only affordable over NVLink (intra-node) or TPU ICI. vLLM's guidance is
  explicit: set `--tensor-parallel-size` to the number of GPUs *in the box*, because fast intra-node
  interconnect makes per-layer communication cheap.
* **Pipeline parallelism** splits the model by *layer ranges*, so the cross-node link carries exactly
  **one activation hand-off per stage boundary**, not one collective per layer. vLLM: set
  `--pipeline-parallel-size` to the number of nodes.

```text
16 devices, 2 nodes × 8 devices:

  TP degree = 8  (within node, over NVLink/ICI)
  PP degree = 2  (across nodes, over DCN/Ethernet)

  Node 0: layers  0..N/2   ← devices 0-7,  TP group 0
             │ activation hand-off (one tensor per microbatch)
             ▼
  Node 1: layers N/2..N    ← devices 8-15, TP group 1
```

**Why pure TP fails across nodes**, concretely: TP's AllReduce runs twice per transformer layer
(after attention output projection, after FFN down projection). For a 32-layer model that is 64
cross-node collectives per forward pass, each one a full-size activation tensor, each one a
synchronization barrier. DCN bandwidth is one to two orders of magnitude below NVLink and its
latency is far worse. The collectives stop being overlappable and the pipeline stalls on the wire.

## 2.3 ⚠️ Pipeline parallelism in StableHLO — the premise needs a correction

The brief proposes: *"Pipeline parallelism in StableHLO: `stablehlo.send` / `stablehlo.recv` for
activation passing between pipeline stages."* The ops exist and the intent is right, but **this is
not a well-trodden path, and for a from-scratch Rust client it is the highest-risk item in ARTX16.**
The evidence:

1. **`stablehlo.send`/`recv`'s documented channel types are host-oriented.** `DEVICE_TO_HOST` is
   valid only with `send`; `HOST_TO_DEVICE` only with `recv`; `is_host_transfer` gates the behaviour.
   Device-to-device pipelining is a different use of the same op and is much less exercised.
2. **PyTorch/XLA's PP support is still an RFC** (openxla/pytorch-xla issue #6347), and its own
   framing notes the PP send/recv ops must live in a *different graph* from the model execution
   graph so GSPMD can insert collectives into the latter — i.e. it is not a simple in-graph edge.
3. **MaxText — Google's own reference JAX LLM implementation — has an open issue asking how to
   implement 1F1B pipelining in JAX** (AI-Hypercomputer/maxtext issue #752). If it were routine it
   would not be an open question there.
4. **NVIDIA built an entire separate library, JaxPP, to get MPMD pipeline parallelism in JAX**,
   splitting computations into multiple independently-jitted SPMD modules dispatched to different
   devices. That is the shape of the real solution, and it is not one MLIR op.
5. **JAX's own supported routes are (a) XLA's SPMD-based PP behind a flag, or (b) manual
   `psend`/`precv`** — and the documentation for (b) carries explicit deadlock warnings: no single
   send/recv's source-target pairs may contain a cycle, and a fake data dependency must be inserted
   to sequentialize send/recv pairs.

⚠️ **DESIGN DECISION — gljax's PP unit is a whole compiled stage program, not an in-graph send/recv edge.**

Rather than one module containing send/recv edges, compile **one StableHLO module per pipeline
stage**, and let the *host* move activations between stages:

```text
Stage 0 module:  (tokens, kv_0)      -> (hidden, kv_0')     layers 0..N/2
Stage 1 module:  (hidden, kv_1)      -> (logits, kv_1')     layers N/2..N
```

Host on node 0 executes stage 0, transfers `hidden` to node 1, node 1 executes stage 1. This is the
MPMD shape JaxPP arrived at, it needs no in-graph send/recv, and it composes with everything gljax
already has: each stage module is an ordinary ARTX5 bucketed artifact in the ARTX4 `CompileCache`,
and each stage's TP sharding is ordinary ARTX6.

**Cost:** the activation crosses the host boundary (device→host→network→host→device) instead of
device→device. For a `[microbatch, 1, D]` decode activation at bf16 that is small — `D=8192`,
microbatch 32 → 512 KB. For prefill chunks it is `[microbatch, chunk, D]` — a 512-token chunk at the
same dims is 256 MB, which is *not* small and must be measured before committing.

**Escape hatch:** if the host round-trip proves too expensive for prefill, the in-graph send/recv
route remains available as a later optimization, on one backend at a time. The MLIR shape it would
take, recorded here so the option is not lost:

```mlir
// Stage 0 tail — hand the activation to stage 1.
// channel_id must be unique and MATCHED between the two stage modules;
// channel_type DEVICE_TO_DEVICE (not the host-transfer variants).
"stablehlo.send"(%hidden, %token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  is_host_transfer = false
} : (tensor<32x1x8192xbf16>, !stablehlo.token) -> !stablehlo.token

// Stage 1 head — receive it.
%hidden, %token_out = "stablehlo.recv"(%token) {
  channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>,
  is_host_transfer = false
} : (!stablehlo.token) -> (tensor<32x1x8192xbf16>, !stablehlo.token)
```

⚠️ The `!stablehlo.token` threading is what orders the transfers. Getting the token chain wrong
produces a deadlock, not a compile error. This is precisely why it is the escape hatch and not the
default.

## 2.4 Pipeline bubbles — and why the training literature misleads here

⛔ **Correction worth stating plainly: 1F1B and interleaved-1F1B are *training* schedules.** They
exist to interleave forward and backward passes and to bound activation memory for the backward
pass. **Inference has no backward pass.** Importing 1F1B terminology into an inference document
imports a solution to a problem gljax does not have.

For pure-forward pipelining with `p` stages and `m` microbatches, the pipeline drains in
`(m + p − 1)` stage-times, so:

```text
bubble fraction = (p − 1) / (m + p − 1)
```

| p (stages) | m (microbatches) | Bubble |
|---|---|---|
| 2 | 1 | 50% |
| 2 | 8 | 11% |
| 2 | 32 | 3.1% |
| 4 | 8 | 27% |
| 4 | 32 | 8.8% |

⚠️ **DESIGN DECISION — the microbatch source is ARTX7's batch, not a new mechanism.**
ARTX7 already produces a batch of concurrent requests each iteration, and already chunks long
prefills. Those are the microbatches. `m` is therefore *the number of concurrently served requests*,
which means **PP efficiency is a function of load**.

⚠️ **The consequence that must not be buried: PP is a throughput technique that costs
single-request latency.** With `p` stages, one decode token for one lone request traverses all `p`
stages sequentially — latency `p × t_stage` with the other `p−1` stages idle. At `m=1` the bubble is
`(p−1)/p`: 50% waste at PP=2. PP only pays when the pipeline is full. A deployment that adds PP to
reduce latency has misunderstood it.

This interacts directly with ARTX8's finding that decode is 200–600× below the roofline ridge point:
PP does not raise arithmetic intensity at all — it partitions the same bandwidth-bound work across
more machines. **PP is for models that do not fit, not for models that are slow.**

## 2.5 Deployment topology

```text
TP degree = devices per node        (NVLink / ICI — cheap collectives)
PP degree = number of nodes         (DCN — one hand-off per boundary)
total devices = TP × PP
```

Constraints inherited from ARTX6: TP degree must divide `n_kv_heads` (GQA constraint, ARTX6 §"GQA TP
constraint validated early"). PP degree must divide `n_layers`, or the layer split is uneven and the
slowest stage sets the pipeline rate.

---

# 3. KV Cache in Distributed Setting

## 3.1 KV cache under TP

ARTX7's slab is `[max_slots, n_kv_heads, max_seq_len, head_dim]` per layer. Under TP the **head
dimension shards**, because attention heads are already independent:

```text
Single device (ARTX7):
  kv_k[layer]: [max_slots, n_kv_heads,          max_seq_len, head_dim]

TP degree T (ARTX16):
  kv_k[layer]: [max_slots, n_kv_heads / T,      max_seq_len, head_dim]   per device
```

Each device owns `n_kv_heads / T` KV heads for **all** slots and **all** its layers. This is the
natural consequence of ARTX6's column-sharded QKV projection: a device that computes K and V for a
head is the device that should store them. **No KV collective is needed** — attention output is
already all-reduced by ARTX6's existing `tp/attention.rs`.

⚠️ **DESIGN DECISION — KV sharding is a slab constructor parameter, not a new module.**
`StaticKvSlab::new()` gains `n_kv_heads_local = n_kv_heads / tp_degree`. `KvSlotManager` is
**completely unchanged** — it owns `SlotId` logic and never knew the head count in the first place
(ARTX7 Design Principle #3). This is the separation earning its keep.

⚠️ GQA constraint: `n_kv_heads % tp_degree == 0` must hold. Qwen2.5-0.5B has `n_kv_heads = 2` — it
**cannot** run TP=8. Small models are TP-limited by their KV head count, and `glserve` must reject
such a config at startup rather than fail at trace time.

## 3.2 KV cache under PP

Each pipeline stage owns the KV cache for **only its own layers**:

```text
Node 0 (layers 0..15):   kv_k[0..15],  kv_v[0..15]
Node 1 (layers 16..31):  kv_k[16..31], kv_v[16..31]
```

**No cross-stage KV transfer ever occurs.** A layer's KV cache is read only by that layer's
attention, and that layer lives entirely on one stage. Only the *activation* crosses the boundary
(§2.3). This is what makes PP cheap on the wire and is the main reason it survives DCN.

Combined TP+PP memory, per device:

```text
kv_bytes_per_device =
    2                        (K and V)
  × max_slots
  × (n_kv_heads / TP)
  × max_seq_len
  × head_dim
  × dtype_bytes
  × (n_layers / PP)
```

## 3.3 Prefill-decode disaggregation — not for gljax v1

Splitwise (ISCA 2024) and DistServe run prefill and decode on **separate machines**, because they are
different workloads: prefill is compute-bound, decode is bandwidth-bound (exactly ARTX8's finding).
Reported gains are large — 2–7× throughput.

⚠️ **DESIGN DECISION — deferred, for three independent reasons.**

1. **ARTX7 already chose the competing answer.** Chunked prefill (Sarathi-Serve) and phase
   disaggregation (Splitwise) are two solutions to *the same problem*: long prefills stalling
   decodes. ARTX7 adopted chunked prefill and built its whole scheduler around it. Adding
   disaggregation is not an increment; it is a different scheduler.
2. **It requires KV cache transfer between machines** — the one thing §3.2 notes PP carefully
   avoids. That means a KV transport layer, which means the multi-host infrastructure of §2.1 that
   does not exist yet.
3. **It is a multi-machine throughput optimization.** gljax v1's target (§9) is one node. On one
   node there is nothing to disaggregate across.

Revisit after multi-host works and there is a measured prefill/decode interference problem that
chunked prefill does not solve.

---

# 4. Fault Tolerance + Health

## 4.1 PJRT failure model — what is actually recoverable

An honest taxonomy, because "restart the Session" is not always sufficient:

| Failure | Detectable as | Recovery |
|---|---|---|
| Bad request (shape/dtype mismatch) | `PJRT_Error` from execute, `INVALID_ARGUMENT` | Fail that request, keep serving |
| Transient execute failure | `PJRT_Error` `INTERNAL` from execute | Retry once; if it recurs, treat as device failure |
| Device failure (GPU XID hardware class) | execute errors persist; device unusable | **Process restart.** See below |
| Plugin/driver crash | process death | Supervisor restarts (systemd / k8s) |
| OOM at slab allocation | error at `Session::new()` | Fail fast at startup, do not start serving |

⚠️ **DESIGN DECISION — device failure is handled by process exit, not in-process recovery.**

The reasoning is empirical, not defeatist: once a PJRT client is initialized with a device it cannot
be re-pointed at another; and GPU hardware faults (the NVIDIA XID classes that require node
isolation) leave the device in an inoperable state needing a reset that a user-space process cannot
perform. In-process "recovery" would mean tearing down the client, the compile cache, the weights,
and all device buffers — which is a process restart with extra bug surface and no weight-load saving.

`glserve` therefore:

```rust
// glserve/src/health.rs
pub enum WorkerHealth {
    Healthy,
    Degraded { consecutive_errors: u32 },
    Failed   { reason: String },
}
```

* A worker that hits `DEVICE_FAILURE_THRESHOLD` consecutive execute errors marks itself `Failed`.
* `Failed` workers are removed from the router's rotation (§5) — a multi-replica deployment keeps
  serving on its remaining replicas.
* If **all** workers are `Failed`, `/health` returns `503` and the process exits non-zero so the
  supervisor restarts it.

⚠️ Every in-flight request on a failed worker gets `TokenEvent::Error` and a proper SSE termination.
Requests must never hang on a dead worker.

## 4.2 Health endpoints

Liveness and readiness must be distinguishable — this is what lets a load balancer drain a node
without killing it.

```rust
// GET /health        → 200 {"status":"ok"} while any worker is Healthy; 503 otherwise
// GET /health/live   → 200 as long as the process is running (never checks devices)
// GET /health/ready  → 200 only when weights are loaded, buckets warmed, and
//                      at least one worker is Healthy AND not draining
```

⚠️ **DESIGN DECISION — readiness is false during bucket warmup.**
ARTX5 compiles 10 artifacts (5 buckets × {prefill, decode}), and ARTX7 multiplies that by the
slot-count buckets. XLA/TPU warmup on this scale is documented at 20–30 minutes cold, ~5 minutes
with a warm on-disk cache. A load balancer must not route traffic during that window. Warmup
progress is exposed so operators can see it moving:

```json
{"status":"warming","buckets_compiled":6,"buckets_total":10,"elapsed_s":412}
```

## 4.3 Graceful drain

```text
SIGTERM received
  ▼
1. /health/ready → 503 immediately   (LB stops sending new work)
2. Router rejects new requests → 503 with Retry-After
3. In-flight requests continue to completion or drain_timeout
4. At drain_timeout: remaining requests get finish_reason "stop" +
   a truncation note, SSE closed cleanly
5. Sessions dropped, PJRT client destroyed, exit 0
```

⚠️ **DESIGN DECISION — bounded drain, default 30 s, and clients are told the truth.**
A request with `max_tokens: 4096` can outlive any reasonable drain window. Cutting the SSE stream
without a terminating chunk leaves clients hanging. Emitting a proper `finish_reason` is the honest
behaviour even though the generation is incomplete.

---

# 5. Request Routing + Load Balancing

## 5.1 Replica model

⚠️ **DESIGN DECISION — a replica is a full model copy, and replicas do not share KV cache.**

```text
Replica = one gljax Session
        = one full copy of the weights
        = its own StaticKvSlab + KvSlotManager + ARTX7 scheduler
        = TP × PP devices
```

Replica-level parallelism *is* data parallelism, and it is the simplest scaling axis: no collectives
between replicas, no shared state, trivially fault-isolated.

```text
8 devices, model fits on 2:
   Option A: TP=8, 1 replica    → lowest single-request latency
   Option B: TP=2, 4 replicas   → highest aggregate throughput
```

⚠️ Option B is usually correct for serving, and it follows from ARTX8: decode is bandwidth-bound, so
4 independent replicas each streaming their own weights use total system bandwidth better than one
replica whose TP collectives serialize. Prefer more replicas until the model no longer fits.

## 5.2 SessionPool

```rust
// glserve/src/pool.rs
pub struct SessionPool {
    workers: Vec<WorkerHandle>,
    policy: RoutingPolicy,
}

pub struct WorkerHandle {
    pub id: ReplicaId,
    inbox: mpsc::Sender<InferenceReq>,
    /// Updated by the worker each scheduler iteration. Read by the router.
    stats: Arc<WorkerStats>,
    health: Arc<AtomicWorkerHealth>,
}

/// Lock-free — the router reads these on every request.
pub struct WorkerStats {
    pub queue_depth:  AtomicUsize,
    pub active_slots: AtomicUsize,
    pub free_slots:   AtomicUsize,
}

impl SessionPool {
    pub fn route(&self, req: InferenceReq) -> Result<(), RouteError>;
    pub fn drain(&self, timeout: Duration);
}
```

## 5.3 Routing policy

```rust
pub enum RoutingPolicy {
    RoundRobin,
    /// Default. Fewest active slots wins; ties broken by queue depth.
    LeastLoaded,
    /// Reserved for ARTX12+ prefix caching — needs a prefix-affinity index.
    SessionAffinity,
}
```

⚠️ **DESIGN DECISION — `LeastLoaded` on *active slots*, not on request count.**
ARTX7's KV slots are the real scarce resource. A replica with 3 long-running generations and one
with 3 nearly-finished ones look identical by request count but have very different capacity to
absorb a new prefill. `free_slots` is the number the router should actually optimize.

⚠️ Prefix-affinity routing is deliberately *not* v1: ARTX7 lists prefix cache in its Non-Goals, so
there is no prefix cache for affinity to exploit. Adding the routing policy before the cache exists
would be routing on a benefit that does not exist.

## 5.4 Backpressure

```text
Request arrives
  ▼
Any worker with free_slots > 0?          ──yes──► route there
  │ no
  ▼
Any worker with queue_depth < max_queue? ──yes──► enqueue (will wait)
  │ no
  ▼
429 Too Many Requests + Retry-After
```

⚠️ **DESIGN DECISION — bounded queues and a real 429, never an unbounded wait.**
An unbounded queue converts an overload into unbounded latency, which every client experiences as a
timeout with no signal about what happened. A 429 with `Retry-After` is actionable — clients back off
and autoscalers see a signal. `max_queue_depth` defaults to `2 × total_slots`: enough to cover slot
turnover, not enough to hide a capacity problem.

---

# 6. Quantization in Serving

## 6.1 Pridwen GQ4A/GQ2A — dequantize at load, on the host

GwenLand's Pridwen quantization architecture (`architecture/Pridwen-proposal-v5.md`) defines block-
structured 4-bit and 2-bit formats — `GQ4A` with zero-centered `dequant(c) = c − 8` and per-block
`scale_i`, `GQ2A` asymmetric with `min_i`/`scale_i` per block and a `super_scale`.

⚠️ **DESIGN DECISION — GQ4A/GQ2A weights are dequantized to BF16 on the host at checkpoint load. No
on-device dequant, no `custom_call`, in gljax v1.**

The reasoning is ARTX8's, applied:

* ARTX8 established that gljax emits only standard StableHLO and owns no kernels. An on-device
  dequant would be either (a) a `custom_call` — which means a per-backend kernel (Pallas for TPU,
  CUDA for GPU) and the toolchain matrix ARTX1 was designed to avoid — or (b) expressed in
  StableHLO ops, which materializes a full BF16 tensor anyway and saves nothing.
* **The bandwidth benefit only exists if the weights stay packed in HBM.** Dequantizing on device
  and then doing a BF16 dot moves the same BF16 bytes as loading BF16 weights would, plus the
  dequant work. There is no win without a fused quantized kernel, and a fused quantized kernel is
  exactly what ARTX8 declined to own.

So the honest v1 position: **Pridwen is a storage/distribution format for gljax, not a compute
format.** It reduces checkpoint size on disk and over the network, not HBM traffic at runtime.

```rust
// glserve/src/config.rs — surfaced so operators know what they're getting
pub enum WeightFormat {
    /// safetensors BF16 → device BF16. No conversion.
    Bf16,
    /// Pridwen GQ4A/GQ2A on disk → dequantized to BF16 on host at load.
    /// Saves disk and network. Does NOT save HBM or bandwidth.
    PridwenDequantToBf16,
}
```

⚠️ Load-time cost must be measured: dequantizing a 70B GQ4A checkpoint on the host is a large
single-threaded pass unless parallelized, and it lands directly in startup latency alongside the
20–30 minute bucket warmup (§4.2).

## 6.2 BF16 is the serving default

Per ARTX2's `PrecisionPolicy::bf16()` default and ARTX8's finding that both TPU MXU and NVIDIA
Tensor Cores multiply in BF16 and accumulate in FP32. Nothing in serving changes this.

## 6.3 FP8 — future, and the mechanism is not what it sounds like

StableHLO does define FP8 types (`f8E4M3FN`, `f8E5M2`, plus `FNUZ` variants), and H100 has FP8 Tensor
Cores.

⚠️ But XLA's FP8 lowering is worth understanding before planning around it: the documented approach
casts FP8 inputs up, applies input scales, runs the Dot at the wider type, computes an output scale
via a reduction, and casts back down — **with the whole sequence fused** so the wider Dot is not
actually materialized. FP8 in XLA is therefore a *fusion pattern with scaling*, not a drop-in dtype
swap. It also needs calibrated scales, which is a checkpoint-conversion concern, not a serving one.

⚠️ **TPU v5e FP8 support is unconfirmed.** v5e's published numbers are 197 BF16 TFLOP/s and 393 INT8
TOPS; FP8 is not among them in the sources reviewed. Do not assume portability of an FP8 path across
gljax's targets without checking per-plugin.

**Verdict: FP8 is ARTX12+ material**, and it depends on ARTX8's Wave A8.α (the `DotAlgorithm` /
`preferred_element_type` plumbing) landing first — without it gljax cannot state the numeric contract
an FP8 path requires.

## 6.4 FP64 oracle in serving

ARTX1 §3.1/§3.4 established the CPU plugin as the FP64 reference oracle.

⚠️ **DESIGN DECISION — `/debug/oracle` exists, is off by default, and is never the same process as a
production replica.**

```rust
// Enabled only with --enable-debug-endpoints. Returns 404 otherwise.
// POST /debug/oracle  { "prompt": "...", "layers": [0, 1] }
//   → per-layer activations from an FP64 CPU-plugin run
```

FP64 on an A100 runs at roughly 1/32 of BF16 throughput and TPU v5e has no usable FP64 at all, so an
oracle request colocated with production traffic would be a self-inflicted denial of service. It is
a correctness tool for a debug deployment.

---

# 7. Observability

## 7.1 Per-request telemetry

```rust
// glserve/src/metrics.rs
pub struct RequestTelemetry {
    pub request_id: RequestId,
    pub replica: ReplicaId,
    pub queued_for: Duration,          // admission → first schedule
    pub ttft: Duration,                // arrival → first token  (the headline number)
    pub prefill_time: Duration,        // summed across ARTX7 chunks
    pub decode_time: Duration,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub tps: f64,                      // generated / decode_time
    pub finish_reason: FinishReason,
    pub bucket: (usize, usize),        // (slot bucket, seq bucket) — cache-hit analysis
}
```

⚠️ **DESIGN DECISION — record the ARTX7 bucket on every request.**
ARTX7's compile cache only pays off if real traffic stays inside the configured
`(batch_size, sequence_bucket)` grid. Logging the bucket per request turns "is our bucket grid right?"
from a guess into a histogram. This is the single most gljax-specific telemetry field and it costs
two `usize`s.

## 7.2 Prometheus metrics

⚠️ **DESIGN DECISION — mirror vLLM's metric semantics under a `gljax:` prefix.**
Operators already have vLLM dashboards, alert rules, and PromQL runbooks. Matching names and label
semantics makes those portable; inventing new ones makes every operator rewrite them for no gain.

| Metric | Type | vLLM analogue |
|---|---|---|
| `gljax:num_requests_running` | Gauge | `vllm:num_requests_running` |
| `gljax:num_requests_waiting` | Gauge | `vllm:num_requests_waiting` |
| `gljax:kv_cache_usage_perc` | Gauge | `vllm:kv_cache_usage_perc` |
| `gljax:time_to_first_token_seconds` | Histogram | `vllm:time_to_first_token_seconds` |
| `gljax:inter_token_latency_seconds` | Histogram | `vllm:inter_token_latency_seconds` |
| `gljax:e2e_request_latency_seconds` | Histogram | `vllm:e2e_request_latency_seconds` |
| `gljax:request_prefill_time_seconds` | Histogram | `vllm:request_prefill_time_seconds` |
| `gljax:request_decode_time_seconds` | Histogram | `vllm:request_decode_time_seconds` |
| `gljax:prompt_tokens_total` | Counter | `vllm:prompt_tokens_total` |
| `gljax:generation_tokens_total` | Counter | `vllm:generation_tokens_total` |
| `gljax:request_success_total{finish_reason}` | Counter | `vllm:request_success_total` |

**gljax-specific additions** with no vLLM equivalent:

| Metric | Type | Why |
|---|---|---|
| `gljax:active_slots{replica}` | Gauge | ARTX7 slot occupancy — the real capacity signal |
| `gljax:compile_cache_hits_total` | Counter | A rising miss rate means traffic left the bucket grid |
| `gljax:compile_cache_misses_total` | Counter | **A miss in steady state is an incident** — it means a 20–30 min compile |
| `gljax:bucket_padding_waste_ratio` | Histogram | Padded tokens ÷ real tokens (ARTX5/ARTX7 grid tuning) |
| `gljax:worker_health{replica,state}` | Gauge | §4.1 |
| `gljax:pipeline_bubble_ratio` | Gauge | PP only — measured `(p−1)/(m+p−1)` against live `m` |

⚠️ `compile_cache_misses_total` deserves an alert, not just a dashboard. Every other metric here
degrades gracefully; a compile miss stalls a request for minutes.

## 7.3 glbench HTTP mode

⚠️ **DESIGN DECISION — yes, add an HTTP mode to glbench, and it measures a different thing than the
in-process mode.**

`glbench` is already a profiler with pull-based engine telemetry. An HTTP mode measures the
**serving stack**: queueing, admission, SSE framing, and multi-request contention — none of which
the in-process path exercises.

```bash
glbench serve --url http://localhost:1136 \
              --concurrency 32 --duration 60s \
              --prompt-len 512 --output-len 128 \
              --report ttft,tpot,throughput,p50,p95,p99
```

⚠️ The two modes must not be compared to each other. In-process measures the engine; HTTP measures
the engine plus the server. Reporting an HTTP number as an engine regression would be the same
category error ARTX8's Scope section warns about.

---

# 8. glserve crate structure

```text
glserve/
├── Cargo.toml              deps: gljax, tokio, axum, tower, tower-http,
│                                 serde, serde_json, prometheus, tracing, clap
└── src/
    ├── main.rs             CLI parse → Config → build pool → serve
    ├── lib.rs              pub use for integration tests
    │
    ├── config.rs           Config, CLI args, TOML file, validation
    ├── types.rs            InferenceReq, TokenEvent, SamplingParams, FinishReason
    │
    ├── api/
    │   ├── mod.rs          Router assembly
    │   ├── chat.rs         POST /v1/chat/completions
    │   ├── completions.rs  POST /v1/completions
    │   ├── models.rs       GET  /v1/models
    │   ├── openai.rs       Wire types: ChatRequest, ChatChunk, Usage, ErrorBody
    │   └── debug.rs        POST /debug/oracle  (feature-gated)
    │
    ├── router.rs           RoutingPolicy, admission control, 429 backpressure
    ├── pool.rs             SessionPool, WorkerHandle, SessionWorker (blocking loop)
    ├── stream.rs           SSE construction, [DONE], keep-alive, disconnect→cancel
    ├── metrics.rs          Prometheus registry, RequestTelemetry, /metrics
    ├── health.rs           WorkerHealth, /health{,/live,/ready}, drain state machine
    │
    └── dist/               Wave A9.3 — multi-node (§2.1). Empty in v1.
        ├── mod.rs
        ├── coordinator.rs  PJRT distributed coordination service binding
        └── stage.rs        PP stage program orchestration (§2.3)
```

⚠️ **DESIGN DECISION — `api/openai.rs` holds the wire types and nothing else.**
OpenAI's schema changes without warning. Isolating it in one file means a schema change touches one
file, and the engine-facing `types.rs` never moves. `api/chat.rs` is the only place that translates
between them.

---

# 9. Deployment Configs

## 9.1 Development — single device

```toml
[server]
port = 1136
[model]
path = "models/qwen2.5-0.5b"
format = "bf16"
[parallel]
tp = 1
pp = 1
replicas = 1
[runtime]
max_slots = 4
buckets = [128, 512]        # fewer buckets → fast warmup, more padding waste
max_queue_depth = 8
```

⚠️ Two buckets, not five: ARTX5's full grid costs 20–30 minutes of cold compile. Dev iteration wants
warmup measured in minutes.

## 9.2 Production, single node — **the gljax v1 target**

```toml
[parallel]
tp = 8          # 8 devices, NVLink/ICI
pp = 1          # no cross-node hop
replicas = 1
[runtime]
max_slots = 64
buckets = [128, 256, 512, 1024, 2048]
max_queue_depth = 128
```

Alternative on the same hardware, per §5.1:

```toml
[parallel]
tp = 2
pp = 1
replicas = 4    # 4 independent Sessions × 2 devices
```

⚠️ Prefer the replicas variant whenever the model fits in `tp=2`. Independent replicas avoid TP
collectives entirely and fault-isolate; ARTX8's bandwidth-bound decode finding says four independent
weight streams beat one collective-serialized one.

## 9.3 Production, multi-node — Wave A9.3, blocked on §2.1

```toml
[parallel]
tp = 8          # within node
pp = 4          # across 4 nodes
replicas = 1    # 32 devices total
[dist]
coordinator = "10.0.0.1:1137"
num_nodes = 4
node_id = 0     # per-node override
```

⛔ **Not startable until the PJRT coordination-service FFI exists** (§2.1). Listed so the config
shape is settled, not because it can be deployed.

## 9.4 Cloud specifics

**GCP TPU v5e pod** — 16 GB HBM per chip at 819 GB/s is the binding constraint. A v5e-8 slice gives
128 GB HBM total; subtract weights and ARTX7's slab. Multi-host TPU pods use the same coordination
service as multi-host GPU, so §2.1 gates this too. TPU has no FP64, so `/debug/oracle` must be
disabled (§6.4).

**Vast.ai / rented A100** — no NVLink guarantee between rented GPUs in some listings. ⚠️ Verify
topology (`nvidia-smi topo -m`) before choosing TP degree; TP over PCIe is a very different
performance regime than TP over NVLink, and a config that assumed NVLink will silently underperform
rather than fail.

---

# 10. What ARTX12 Should Cover

**Recommendation: ARTX12 — Speculative Decoding under Static Shapes.**

The argument follows directly from ARTX8's central measurement. Decode runs at an arithmetic
intensity of 0.5–2 FLOP/byte against ridge points of 241 (TPU v5e), 153 (A100), and 295 (H100) — the
matrix unit is idle the overwhelming majority of decode time, waiting on HBM. Speculative decoding is
the technique that **converts that idle compute into throughput**: a draft model proposes k tokens,
the target model verifies all k in one forward pass at the same bandwidth cost as verifying one.

It is also the item where gljax's constraints make the design genuinely non-trivial rather than a
port of someone else's:

* Draft trees are **dynamic** — variable accepted length per step. gljax is **static-shape**
  (ARTX5/ARTX7). Bucketing a draft tree is an open design problem, not a solved one.
* Verification is a `[k, D]` forward pass — a *small GEMM* rather than a GEMV, which is exactly the
  arithmetic-intensity shift ARTX8 identified as the lever.
* It interacts with ARTX7's slot accounting: accepted tokens advance the KV position by a variable
  amount, which the static slab addressing (`dynamic_update_slice` at `[slot, :, pos, :]`) must
  handle without a shape change.

The state of the art is settled enough to build against: EAGLE-3 is the production standard, merged
into vLLM, SGLang, and TensorRT-LLM, with acceptance rates of roughly 0.75–0.85 on chat-style
workloads and reported 2–6× speedups.

**Runners-up, and why they rank lower:**

| Candidate | Why not first |
|---|---|
| **Quantized serving (FP8)** | Blocked on ARTX8 Wave A8.α; XLA's FP8 path is a scaled fusion pattern needing calibration (§6.3); TPU v5e support unconfirmed |
| **Prefill/decode disaggregation** | Blocked on multi-host (§2.1), and ARTX7 already chose the competing answer (§3.3) |
| **Prefix caching / RadixAttention** | Real win for multi-turn chat, but ARTX7 listed it as a Non-Goal and it needs a KV-sharing design the static slab does not currently permit |
| **Online distillation** | Training-adjacent; gljax is inference-only and has no gradient path |

⚠️ **A candidate the brief did not list, and which may deserve to outrank all of these: a
correctness/evaluation harness.** ARTX1–ARTX16 specify a complete engine, and the only correctness
mechanism anywhere in the series is ARTX1's FP64 oracle for individual ops. There is no end-to-end
answer to *"does this engine produce the right tokens for a real model?"* — no perplexity harness, no
reference-output comparison, no model-coverage matrix. GwenLand has already been bitten by exactly
this class of bug in a sibling engine (a silent dequant-order corruption that produced fluent
garbage, caught only by an end-to-end perplexity run). If ARTX12 is chosen for *risk reduction*
rather than *performance*, this is the one.

---

# Appendix A — Design Decision Summary

| # | Decision | Rationale | Reversible? |
|---|---|---|---|
| D1 | `glserve` is a separate crate; `gljax` keeps zero async/HTTP deps | Preserves ARTX7's sync scheduler; keeps gljax embeddable | Hard — dependency direction |
| D2 | One blocking OS thread per replica; channels across the async boundary | ARTX7 Non-Goal: no async runtime in the engine | Hard |
| D3 | Port 1136 | Established GwenLand convention (GUI, legacy TUI, config default) | Trivial |
| D4 | SSE keep-alive on by default | Long prefill can exceed proxy idle timeouts | Trivial |
| D5 | Client disconnect cancels the request and frees the KV slot | Abandoned tabs otherwise hold slots for full `max_tokens` | Trivial |
| D6 | `try_recv` hot loop, `recv_timeout` when idle | Work-conserving without spinning | Trivial |
| D7 | Multi-node gated behind PJRT coordination-service FFI | ARTX1/ARTX6 are single-host; the FFI does not exist | N/A — sequencing |
| D8 | TP within node, PP across nodes | Industry-settled; TP AllReduce is unaffordable on DCN | Hard |
| D9 | **PP unit = one compiled module per stage, host-mediated transfer** (not in-graph send/recv) | In-graph device-to-device PP is an RFC in PyTorch/XLA, an open issue in MaxText, and needed a whole separate library (JaxPP) in JAX | Medium — send/recv remains an escape hatch |
| D10 | Bubble math uses `(p−1)/(m+p−1)`; 1F1B/interleaved are training schedules | Inference has no backward pass | N/A — correction |
| D11 | ARTX7's batch supplies the microbatches | No new mechanism needed; makes PP efficiency load-dependent | Trivial |
| D12 | KV shards on the head dim under TP; `KvSlotManager` unchanged | ARTX7 Design Principle #3 (ownership vs storage) earning its keep | Trivial |
| D13 | No cross-stage KV transfer under PP | Each layer's KV is read only by that layer | N/A — property |
| D14 | Prefill/decode disaggregation deferred | ARTX7 chose chunked prefill, the competing answer; needs multi-host | Medium |
| D15 | Device failure → process exit, not in-process recovery | PJRT client cannot be re-pointed; GPU hard faults need a device reset | Medium |
| D16 | Readiness false during bucket warmup, with progress exposed | 20–30 min cold compile must not receive traffic | Trivial |
| D17 | Bounded drain (30 s default) with honest `finish_reason` | `max_tokens: 4096` can outlive any drain window | Trivial |
| D18 | Replica = full model copy; no shared KV | Simplest scaling axis; fault-isolated | Hard |
| D19 | `LeastLoaded` routes on free KV slots, not request count | Slots are the scarce resource (ARTX7) | Trivial |
| D20 | Bounded queue + 429 `Retry-After` | Unbounded queues convert overload into unbounded latency | Trivial |
| D21 | **Pridwen GQ4A/GQ2A dequantized to BF16 on host at load** | ARTX8: no custom kernels; on-device dequant saves no HBM traffic without a fused quantized kernel | Medium |
| D22 | BF16 serving default | ARTX2 `PrecisionPolicy`; MXU/Tensor Core native | Trivial |
| D23 | FP8 deferred to ARTX12+ | Needs A8.α plumbing + calibration; TPU v5e support unconfirmed | N/A — sequencing |
| D24 | `/debug/oracle` off by default, never colocated with production | FP64 is ~1/32 BF16 on A100 and absent on TPU v5e | Trivial |
| D25 | Prometheus names mirror vLLM under a `gljax:` prefix | Operator dashboards and runbooks port unchanged | Trivial |
| D26 | Log ARTX7 bucket per request; alert on compile-cache misses | A steady-state miss is a multi-minute stall, i.e. an incident | Trivial |
| D27 | `glbench` gains an HTTP mode, not compared against in-process | They measure different systems | Trivial |
| D28 | `api/openai.rs` isolates all wire types | OpenAI's schema changes without notice | Trivial |

---

# Appendix B — Wave Plan

| Wave | Scope | Gate |
|---|---|---|
| **A9.1** | `glserve` skeleton: config, `SessionPool` with 1 replica, `/v1/chat/completions` non-streaming, `/health`, `/v1/models` | A single prompt returns correct tokens over HTTP, matching an in-process `Session::generate()` run |
| **A9.2** | SSE streaming, cancel-on-disconnect, `/metrics`, backpressure + 429, graceful drain, multi-replica `LeastLoaded` | `glbench serve` at concurrency 32 with no dropped or hung streams; slots return to free after client disconnect |
| **A9.3** | Multi-node: PJRT coordination-service FFI, `dist/coordinator.rs`, PP stage orchestration | **Blocked on §2.1.** Two-node PP produces token-identical output to single-node |

⚠️ A9.1 and A9.2 are blocked only on ARTX5's `Session::generate()` existing. A9.3 is blocked on new
PJRT FFI work that no ARTX document has yet specified.

---

# Sources

- [Distributed Inference with vLLM](https://vllm.ai/blog/2025-02-17-distributed-inference) and [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/) — TP within node / PP across nodes; TP size = GPUs per node, PP size = node count.
- [jax.distributed.initialize](https://docs.jax.dev/en/latest/_autosummary/jax.distributed.initialize.html) and [Introduction to multi-controller JAX](https://docs.jax.dev/en/latest/multi_process.html) — coordinator address, `num_processes`, `process_id`; coordination service does discovery, topology exchange, health checking.
- [PJRT C API changelog](https://github.com/openxla/xla/blob/main/xla/pjrt/c/CHANGELOG.md) and [PJRT C++ Device API Overview](https://openxla.org/xla/pjrt/cpp_api_overview) — `PJRT_KeyValueTryGet`, KV-store interface in client creation.
- [StableHLO Specification](https://openxla.org/stablehlo/spec) — `send`/`recv`, `channel_handle`, `DEVICE_TO_HOST` / `HOST_TO_DEVICE`, `is_host_transfer`; FP8 types `f8E4M3FN`, `f8E5M2`, FNUZ variants.
- [RFC: Pipeline parallelism for PyTorch/XLA (issue #6347)](https://github.com/pytorch/xla/issues/6347) — PP send/recv must live in a separate graph from the model execution graph.
- [MaxText issue #752 — How to implement 1F1B pipeline parallelism in Jax?](https://github.com/AI-Hypercomputer/maxtext/issues/752) — still an open question in Google's reference JAX LLM implementation.
- [JaxPP (NVIDIA)](https://github.com/NVIDIA/jaxpp) and [Scaling Deep Learning Training with MPMD Pipeline Parallelism](https://arxiv.org/pdf/2412.14374) — MPMD PP as multiple independently-jitted SPMD modules.
- [JAX GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html) — XLA SPMD-based PP behind a flag; manual `psend`/`precv` with cycle/deadlock warnings.
- [Pipeline-Parallelism: Distributed Training via Model Partitioning](https://siboehm.com/articles/22/pipeline-parallel-training) and [Megatron scaling blog](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/) — GPipe bubble `(p−1)/m`; 1F1B and interleaved schedules are forward+backward training constructs.
- [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://www.researchgate.net/publication/382806162_Splitwise_Efficient_Generative_LLM_Inference_Using_Phase_Splitting) and [Prefill-decode disaggregation](https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation) — 2–7× throughput; compute grew 3.43× A100→H100 while bandwidth grew 1.64×.
- [Chat Completions streaming events | OpenAI API](https://developers.openai.com/api/reference/resources/chat/subresources/completions/streaming-events) — `data:` framing, `choices[].delta`, `[DONE]` terminator.
- [axum::response::sse](https://docs.rs/axum/latest/axum/response/struct.Sse.html) and [KeepAlive](https://docs.rs/axum/latest/axum/response/sse/struct.KeepAlive.html) — `Sse::new(stream).keep_alive(...)`, default 15 s interval.
- [vLLM Metrics design](https://docs.vllm.ai/en/stable/design/metrics/) and [Monitoring vLLM in Production](https://akrisanov.com/vllm-metrics/) — `vllm:` metric names and semantics.
- [RFC: FP8 in XLA](https://github.com/openxla/xla/discussions/22) — FP8 Dot lowering casts up, applies scales, runs wider Dot, rescales, all fused.
- [Story of Two GPUs: Characterizing the Resilience of Hopper H100 and Ampere A100](https://arxiv.org/pdf/2503.11901) and [From Detection to Recovery: LLM Pre-training with 504 GPUs](https://arxiv.org/pdf/2605.09370) — XID classification; hardware-class errors require node isolation, application-class allow retry.
- [EAGLE-3 speculative decoding](https://www.spheron.network/blog/eagle-3-speculative-decoding-gpu-cloud/) and [Speculative Decoding 2026](https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/) — production standard in vLLM/SGLang/TensorRT-LLM; acceptance 0.75–0.85; 2–6×.

**Repo-internal:** `architecture/Pridwen-proposal-v5.md` (GQ4A/GQ2A block format), root `Cargo.toml`
(workspace membership), `changelog/Gwen-Changes-2026-05-31_23-00.md` (`general.default_port = 1136`).
