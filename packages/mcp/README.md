# gwenland-mcp

**Model Context Protocol server for the GwenLand stack.**

## What level does it work at?

**Tool-call level.** The coarsest grain in the workspace: an agent asks for a
whole operation ("benchmark this model", "run inference") and gets a structured
result. One MCP call can be seconds of engine work.

```
agent  →  JSON-RPC over stdio  →  gwenland-mcp  →  GL engines
```

## Tools

| Tool | What it does |
|---|---|
| `load` | Load a model into the runtime |
| `infer` | Run inference and return the completion |
| `benchmark` | Run a `glbench` session |
| `train` | Start a training run |
| `train_status` | Poll a running training job |
| `publish` | Publish a model artifact |

~1,350 lines.

## Dependencies

**Two direct: `serde`, `serde_json`.** Eleven crates in the tree — the smallest
footprint of any crate here that does real work.

That is not an accident. MCP speaks **JSON-RPC over stdio**, not HTTP, so there
is no async runtime, no server framework and no socket anywhere in this crate.
Compare `glserve` at 95 crates for the same "let something else drive the
engine" job over a different transport.

## Running it

```bash
cargo run -p gwenland-mcp
```

It reads JSON-RPC from stdin and writes to stdout, which is what an MCP client
expects. Anything the crate wants to say to a human goes to **stderr** — writing
a log line to stdout corrupts the protocol stream.

## Tests

```bash
cargo test -p gwenland-mcp
```

Four cover the registry itself: every descriptor has a name and an input schema,
every dispatchable tool is listed, an unknown tool name is refused, and a known
tool reports bad arguments rather than panicking. Those are the invariants an
agent depends on — a tool that panics on malformed input takes the session down.
