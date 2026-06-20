# GwenLand

An AI toolkit that runs entirely on your own machine. It's written in Rust, ships as one binary under 50 MB, and doesn't send your data anywhere.

A single command-line tool covers the whole loop: pull a model, fine-tune it, serve it locally, and chat with it. Dataset tools and HuggingFace Hub access come built in.

This is still pre-release, so flags, file formats, and APIs can change without warning. The native `gwen train` backend is the rough edge — it converges, but adapters it produces aren't ready to drop straight into inference, so don't lean on it for real training yet.

## Why Rust instead of Python

Most local-AI tooling is Python on top of more Python. That's great in a notebook, but it drags along a heavy runtime and a big install. GwenLand goes the other way: native code, no interpreter, no garbage-collector pauses, and a stripped binary small enough to carry around. You trade a little convenience for speed and a tiny footprint.

## What it does

- `gwen fetch` — download a model from HuggingFace and pick the quantization
- `gwen train` — fine-tune with LoRA on the native Rust/Candle backend
- `gwen serve` — start a local inference server
- `gwen chat` — talk to a local model in your terminal
- `gwen benchmark` — measure cold start, inference, layer load, and memory
- `gwen hub` — list, pull, push, and prune HuggingFace models
- `gwen dataset` — validate, convert, and split JSONL datasets
- `gwen scan` — check models and datasets for PII, toxicity, and prompt injection
- `gwen convert` — turn a GGUF file into SafeTensors
- `gwen eval` — score a model against a validation set
- `gwen config` and `gwen doctor` — manage settings and sanity-check your setup

## Where the code lives

Everything is under `gwen-cli/`. Its [README](gwen-cli/README.md) has the build steps and the full command reference, and `gwen-cli/changelog/` keeps the per-session notes.

## License

MIT with the Commons Clause, in [LICENSE](LICENSE). Free for personal and research use; commercial use needs a separate agreement.
