# glserve

**OpenAI-compatible HTTP serving for `gljax`** (ARTX16).

## What level does it work at?

**Request level.** One HTTP request in, one completion out. glserve owns no
kernels, no tensors and no tokenizer — it is a translation layer between the
OpenAI wire format and `gljax::runtime::Session`.

```
POST /v1/chat/completions  →  glserve  →  gljax Session  →  PJRT plugin
```

## Endpoints

| Route | Method |
|---|---|
| `/v1/chat/completions` | POST |
| `/v1/models` | GET |
| `/health` | GET |

Default bind: **`127.0.0.1:1136`** — loopback only. Port 1136 is the
established GwenLand convention, inherited rather than reinvented.

```bash
cargo run -p glserve
curl localhost:1136/health
```

## This is local, not cloud

Worth saying plainly, because "HTTP server" reads as "network service": glserve
binds loopback and **makes no outbound connections at all**. There is no HTTP
client, no TLS stack, and no downloader anywhere in the workspace — verified by
`cargo tree`, not assumed.

Its purpose is the same as `ollama serve` or `llama-server`: give *other
programs on the same machine* — an editor plugin, a GUI, a Python script — a way
to reach the engine without linking Rust.

## ⛔ Status: v1, and currently unused

**Nothing in this repository depends on glserve.** It is a workspace member and
a leaf. The three consumers its own module docs cite for the port-1136
convention are all gone or absent:

| Cited consumer | Reality |
|---|---|
| `packages/tui` serve command | retired to `.abandoned/gltui/` (2026-07-18) |
| Tauri GUI SSE endpoint | not in this repository |
| `general.default_port` config schema | only in `.kiro/specs/`, legacy candle/mistralrs specs |

It works — it was built in the ARTX06-16 sprint and verified end-to-end with
`curl` — but it is waiting for a consumer.

**Also v1 in scope**, per its own docs: one model, one worker, no hot-swap.
ARTX16's full design (multi-replica routing, `SessionPool`, continuous batching,
Prometheus, multi-node) targets an engine that did not run when that document
was written. `gljax` runs now (Gate A5), but **ARTX7 — continuous batching,
`KvSlotManager`, the scheduler — still does not exist**, so the scheduler this
crate would otherwise loop around is simplified away.

## Dependency cost — the largest in the workspace

| | external crates |
|---|---|
| glserve's subtree | **95** |
| ...of which **only** glserve pulls | **36** |
| whole workspace without glserve | 74 |
| whole workspace with it | 110 |

The 36 are the async HTTP stack: `tokio`, `axum`, `hyper`, `tower`, `mio`,
`socket2`, `bytes`, `futures-*`, `tracing`.

**None of it reaches the inference path.** `glcore`, `glproc`, `glcuda`,
`gljax`, `glbench` and `glcli` pull zero of these — `cargo build -p glbench`
never compiles `tokio`. The cost is paid only by `cargo build --workspace` and
by building glserve itself.

If that becomes a problem, the fix is to `exclude` glserve from the root
workspace the way `gltrain` and `stumman` already are — not to delete it.

## Direction, kept from ARTX16

**glserve depends on gljax; never the reverse.** `gljax` has zero HTTP and zero
async dependencies, and keeping the server a separate crate is what preserves
that. A cycle fails `cargo check`, which is the enforcement.
