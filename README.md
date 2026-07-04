# GwenLand

GwenLand is a local-first AI toolkit written in Rust. It ships as a single, lightweight binary (under 50 MB) that provides an end-to-end workflow for running and experimenting with local models.

A single command-line tool covers the whole loop: pull a model, fine-tune it, serve it locally, and chat with it. Inference and data processing run entirely on your own machine—your data never leaves it.

## Key Features

- **Fetch**: Download models from HuggingFace and manage local registry.
- **Train (Experimental)**: Fine-tune models using LoRA directly on a native Rust/Candle backend.
- **Serve**: Spin up a local inference server.
- **Chat**: Interact with models in your terminal.
- **Tools**: Built-in dataset validation, GGUF conversion, evaluation, and security scanning (PII, toxicity).

## Status & Expectations (No Overpromising)

GwenLand is in **pre-release**.
- Flags, file formats, and APIs are subject to change without warning.
- The native `gwen train` backend is highly experimental. While the layer-streaming LoRA objective converges, it is currently an approximation and the generated adapters are not yet ready to drop straight into inference. It is suitable for experimenting and hacking, but **not** for real-world training runs yet.

GwenLand targets modest hardware (e.g., 8GB RAM machines without GPUs) using an mmap-based layer loader, but it also natively supports CUDA and Metal if you have the hardware.

## Quick Start

You will need a recent Rust toolchain (edition 2024, Rust 1.85+).

```bash
# Build the binary
cargo build --release -p gwenland-tui

# Alias it for convenience
alias gwen="$PWD/target/release/gwenland"

# Check your environment
gwen doctor

# Start the workflow
gwen fetch -m tinyllama/TinyLlama-1.1B -q q4_k_m
gwen serve tinyllama-1.1b-q4_k_m
gwen chat
```

## Documentation

Everything is built as a Cargo workspace. Head over to [`gwen-cli/`](gwen-cli/) and read its [README](gwen-cli/README.md) for full documentation, build steps, and command references.
The session-by-session changelog is kept in [`gwen-cli/changelog/`](gwen-cli/changelog/).

## License

MIT with the Commons Clause (see [LICENSE.txt](LICENSE.txt)). Free for personal and research use; commercial use requires a separate agreement.
