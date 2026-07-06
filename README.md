# GwenLand

GwenLand is a local-first AI toolkit. One Rust binary, under 50 MB, that takes you through the whole loop: fetch a model, fine-tune it, serve it, and chat with it. Inference runs on your machine and nothing leaves it.

It's still pre-release, so expect flags and file formats to move around. The native `gwen train` backend especially is experimental: the layer-streaming LoRA objective converges, but it's still an approximation (mean-pool forward, capped vocab, one projection per layer), so the adapters it makes aren't ready for inference yet. Fine for experimenting, not for real training runs. The `changelog/` folder tracks what's landed.

GwenLand is built for modest hardware. The machine we target is an 11th-gen i3 with 8 GB of RAM and no GPU, running Linux, backed by an mmap-based loader that streams one model layer into memory at a time so it doesn't blow the RAM budget. A GPU is optional, not required.

## Installing

You'll need a recent Rust toolchain (edition 2024, so Rust 1.85 or newer). From the workspace root:

```bash
cargo build --release -p gltui
```

The stripped binary lands at `target/release/gwenland`. It's named `gwenland`, but its command name is `gwen`, which is how this README refers to it. Alias it so the examples work as written:

```bash
alias gwen="$PWD/target/release/gwenland"
```

Or just run it through cargo while you're hacking on it:

```bash
cargo run -p gltui -- doctor
```

### Using a GPU

The default build is CPU-only and compiles anywhere without a CUDA toolkit or the macOS SDK. To use a GPU you don't rebuild anything — set an environment variable and Candle picks it up at runtime:

```bash
CANDLE_CUDA=1  gwen serve qwen3-8b-q4_0    # NVIDIA
CANDLE_METAL=1 gwen serve qwen3-8b-q4_0    # Apple Silicon
```

## A quick run-through

The whole pipeline, start to finish:

```bash
gwen doctor                                   # check your environment first
gwen fetch -m tinyllama/TinyLlama-1.1B -q q4_k_m
gwen train -m tinyllama/TinyLlama-1.1B -d ./data.jsonl --epochs 3   # experimental
gwen serve tinyllama-1.1b-q4_k_m              # serves SSE on port 1136
gwen chat                                     # chat with it in the terminal
```

## Commands

- `gwen fetch` — download a model from HuggingFace. Resumes interrupted downloads and checks the SHA-256.
- `gwen train` — fine-tune a LoRA adapter on the native Candle backend.
- `gwen serve` — start the local inference server (SSE, served by candle-transformers, no subprocess).
- `gwen chat` — a streaming terminal chat with history.
- `gwen run` — one-shot inference on a local GGUF file.
- `gwen eval` — score a model on a validation set.
- `gwen benchmark` — cold start, inference, layer load, and memory numbers.
- `gwen hub` — list, pull, push, info, and prune on HuggingFace.
- `gwen dataset` — validate, convert, and split JSONL datasets.
- `gwen scan` — flag PII, toxicity, and prompt injection in models and datasets.
- `gwen convert` — GGUF to SafeTensors.
- `gwen config` — read and write your settings.
- `gwen doctor` — check the storage layout, runtime, and dependencies.
- `gwen update` — update to the latest release.

Every command also takes a few global flags: `--json` for machine-readable output (NDJSON with `--non-interactive`), `--non-interactive` for scripts and agents (no TUI or prompts, and it turns on automatically when stdout isn't a terminal), `--dry-run` to validate without doing anything, and `--yes` to auto-confirm prompts.

A few examples:

```bash
gwen fetch -m mistralai/Mistral-7B-v0.1 -q q5_k_m
gwen fetch --from https://example.com/model.gguf --to /data/models

gwen train -m tinyllama/TinyLlama-1.1B -d ./data.jsonl --dry-run     # estimate cost first
gwen train --auto-merge --base-model ./qwen3.gguf --dataset ./data.jsonl

gwen serve qwen3-8b-q4_0 --port 8080
gwen serve qwen3-8b-q4_0 --dry-run

gwen dataset validate ./data.jsonl
gwen scan ./model.gguf
```

## Where your files go

Everything lives under one folder in your home directory, `~/.gwenland/`. Set `GWEN_HOME` if you want it somewhere else. The folders are created as needed:

- `config/` — your `config.json`
- `models/` — downloaded models and the `models.json` registry
- `crash-logs/` — readable crash reports
- `cache/` — internal cache and a `tmp/` for partial downloads and updates
- `eval_results/` — output from `gwen eval`

This replaced the old `~/.config/gwen/` layout, and there's no automatic migration. If you used a pre-1.0 build, just re-run `gwen fetch <model>` to repopulate; your old data is left alone.

If `gwen` ever crashes — a panic, or a lower-level fault like a segfault — it writes a readable report to `~/.gwenland/crash-logs/`. Set `RUST_BACKTRACE=1` for a full trace, and run `gwen doctor` to confirm the storage folders exist and are writable.

## Project structure

The repo is a Cargo workspace (resolver 2, edition 2024):

- `packages/core` — `gwenland-core`, where the real work happens: inference, training, benchmarks, storage, diagnostics. It's a library crate.
- `packages/gltui` — `gltui`, the `gwenland` CLI and its ratatui TUI. This is the binary.
- `changelog/` — per-session notes.

The release profile is tuned for size: `opt-level = "z"`, fat LTO, one codegen unit, `panic = "abort"`, symbols stripped. That's how the binary stays under 50 MB.

## Working on it

```bash
cargo check --workspace

# Core tests run single-threaded on purpose: a few of them touch process-global
# state (the panic hook, the GWEN_HOME test env) and would race otherwise.
cargo test -p gwenland-core --lib -- --test-threads=1

cargo bench -p gwenland-core
```

Each change gets a note under `changelog/`, and new ideas for the CPU training path get triaged in [NewExperiment.md](NewExperiment.md) before they turn into a tracked issue and a spec. See [CONTRIBUTING.md](CONTRIBUTING.md) for the rest.

## Privacy

Training and inference run on your machine. The only network calls are the model and dataset downloads and pushes you ask for with `gwen fetch` and `gwen hub`. [PRIVACY.md](PRIVACY.md) has the details.

## License

MIT with the Commons Clause — see [LICENSE.txt](LICENSE.txt). Free for personal and research use; commercial use needs a separate agreement.
