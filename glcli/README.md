# glcli

**The command-line front end.** Binary name: `gwen`.

## What level does it work at?

**Token level.** This is the highest-level crate in the engine stack: a prompt
goes in, tokens come out, and everything below the `Runtime` boundary is
someone else's concern.

```
gwen run  →  glcore::Runtime  →  Box<dyn GlEngine>  →  glproc / glcuda
             (owns tokenization)
```

glcli holds **no inference logic at all**. It parses arguments, builds a
`Runtime`, and prints what comes back. 328 lines.

## Commands

| Command | What it does |
|---|---|
| `gwen run` | Load a model, run a prompt, stream tokens |
| `gwen info` | Report the machine and which engines are available |
| `gwen tui` | ⛔ **stub — see below** |

```bash
cargo run -p glcli -- run --model model.gguf --prompt "hello"
cargo run -p glcli -- info
```

## ⛔ `gwen tui` is stale, and its advice is wrong

The subcommand exists and prints:

```
gwen tui: not wired to the GL engines yet — coming in M2.
Meanwhile, run the standalone TUI with: cargo run -p gltui
```

**That second line no longer works.** `gltui` was retired to `.abandoned/gltui/`
on 2026-07-18 — it never called the GL engines (its `CoreBridge` was a stub) and
it is not a workspace member, so `cargo run -p gltui` fails.

The subcommand is kept only because removing a CLI verb is a breaking change to
anyone's scripts. Treat it as unimplemented; the message needs updating.

## Dependencies

**One direct: `clap`.** Which costs more than it looks — 33 crates in the tree,
against 15–16 for every engine crate. `clap` alone is roughly double the
dependency footprint of `glproc`.

That is a deliberate exception rather than an oversight: `glbench` hand-rolls
its argument parser precisely to avoid this, but a user-facing CLI wants `clap`'s
help text, error messages, and completions, and glcli is not on the inference
hot path.

## Build

```bash
cargo test -p glcli
cargo build --release -p glcli   # binary: target/release/gwen
```
