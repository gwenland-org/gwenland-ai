# GwenLand

> **"Speed is Everything, but Precise is more than Everything."**

AI all-in-one toolkit. Local-first. &lt;50MB. Privacy-first. Zero Python.

A single Rust CLI for the full LLM lifecycle — **fetch → train → serve → chat** — with dataset tooling, a safety scanner, and HuggingFace Hub integration built in. Every model runs on your machine. No data leaves it.

> ⚠️ **Unstable / pre-release.** GwenLand is under active development; APIs, file formats, and CLI flags may change without notice. The native `gwen train` backend in particular is **experimental** — the layer-streaming LoRA objective converges but is still an approximation (mean-pool forward, capped vocab, one projection per layer), so trained adapters are **not yet drop-in for inference**. Don't rely on it for production training. See the [changelog](changelog/) for what's landed and what's coming.

---

## Why Rust

| Others | GwenLand |
|---|---|
| Python for accessibility | Rust for precision |
| Abstractions on abstractions | Lean, direct, no runtime overhead |
| &lt;50MB? Impossible in Python | &lt;50MB. Stripped. Precise. |

GwenLand is designed to be useful on commodity hardware. The reference target is an
**i3 (11th gen), 8 GB RAM, no GPU**, served by an mmap-based, OOM-safe layered model
loader that streams one layer into RAM at a time. GPUs are optional, not required.

---

## Install

GwenLand builds from source with a stable Rust toolchain (edition 2024).

```bash
# clone, then from the gwen-cli/ workspace root:
cargo build --release -p gwenland-tui

# the stripped binary lands at:
#   target/release/gwenland
```

The binary is named `gwenland`; throughout this README it is invoked as **`gwen`**
(its CLI name). Symlink or alias it for convenience:

```bash
alias gwen="$PWD/target/release/gwenland"
```

Or run it through Cargo while developing:

```bash
cargo run -p gwenland-tui -- doctor
```

### GPU acceleration (optional)

The default build is CPU-only and compiles everywhere without a CUDA toolkit or macOS
SDK. To opt into GPU at runtime, set an environment variable — Candle picks it up
dynamically:

```bash
CANDLE_CUDA=1  gwen serve qwen3-8b-q4_0     # NVIDIA
CANDLE_METAL=1 gwen serve qwen3-8b-q4_0     # Apple Silicon
```

---

## Quickstart

The full local pipeline, end to end:

```bash
# 0. Sanity-check your environment
gwen doctor

# 1. Pull a quantised model from HuggingFace
gwen fetch -m tinyllama/TinyLlama-1.1B -q q4_k_m

# 2. Fine-tune a LoRA adapter on a JSONL dataset  (experimental)
gwen train -m tinyllama/TinyLlama-1.1B -d ./data.jsonl --epochs 3

# 3. Serve the model locally over SSE
gwen serve tinyllama-1.1b-q4_k_m            # POST /gwenland/chat on :1136

# 4. Chat with it in the terminal
gwen chat
```

---

## Commands

| Command | What it does |
|---|---|
| `gwen fetch` | Download models from HuggingFace with quantisation selection (resumable, checksum-verified) |
| `gwen train` | Fine-tune via LoRA — native Rust/Candle backend |
| `gwen serve` | Spawn a local inference server (SSE, candle-transformers, no subprocess) |
| `gwen chat` | Streaming TUI chat with conversation history |
| `gwen run` | One-shot native inference on a local GGUF model |
| `gwen eval` | Evaluate a model on a validation dataset |
| `gwen benchmark` | Cold-start, inference, layer-load, and memory benchmarks |
| `gwen hub` | HuggingFace Hub — list, pull, push, info, prune |
| `gwen dataset` | Validate, convert, and split JSONL training datasets |
| `gwen scan` | Safety scanner — PII, toxicity, prompt injection |
| `gwen convert` | GGUF → SafeTensors dequantisation |
| `gwen config` | Manage user configuration |
| `gwen doctor` | Environment health check (storage layout, runtime, deps) |
| `gwen update` | Self-update to the latest release |

### Global flags

Every subcommand accepts these:

| Flag | Effect |
|---|---|
| `--json` | Structured JSON output (NDJSON when combined with `--non-interactive`) |
| `--non-interactive` | Agent/script mode — no TUI, spinners, or prompts (auto-enabled when stdout is not a TTY) |
| `--dry-run` | Pre-flight validation only — no side effects |
| `--yes` | Auto-confirm all `[Y/n]` prompts |

### Examples

```bash
# Fetch — multiple models, direct URL, or recover a corrupt download
gwen fetch -m mistralai/Mistral-7B-v0.1 -q q5_k_m
gwen fetch --from https://example.com/model.gguf --to /data/models
gwen fetch -m tinyllama/TinyLlama-1.1B --cache-clear

# Train — estimate cost first, then the one-shot train→export→merge pipeline
gwen train -m tinyllama/TinyLlama-1.1B -d ./data.jsonl --dry-run
gwen train --auto-merge --base-model ./qwen3.gguf --dataset ./data.jsonl
gwen train export-adapter ...      # export a LoRA adapter from a checkpoint
gwen train merge-adapter   ...      # merge adapter SafeTensors into a GGUF base

# Serve — pick a port, or just validate the request
gwen serve qwen3-8b-q4_0 --port 8080
gwen serve qwen3-8b-q4_0 --dry-run

# Datasets & safety
gwen dataset validate ./data.jsonl
gwen scan ./model.gguf
```

---

## Storage layout

All state lives under a single home dotfile root, **`~/.gwenland/`** (override with the
`GWEN_HOME` environment variable). Every accessor self-heals — directories are created on
demand.

```
~/.gwenland/
  config/         — config.json (user + engine settings)
  models/         — downloaded models + models.json registry
  crash-logs/     — human-readable crash-<timestamp>.txt reports
  cache/          — internal cache + tmp/ for partial downloads & self-updates
  eval_results/   — gwen eval output JSON
```

> **Note:** this is a breaking change from the old `~/.config/gwen/` layout with **no
> auto-migration**. Pre-1.0 users upgrading just re-run `gwen fetch <model>` to repopulate
> models; old data at `~/.config/gwen/` is left untouched.

On any panic or OS-level fault (segfault, illegal instruction, abort) — from the CLI, TUI,
or GUI — GwenLand writes a readable crash report to `~/.gwenland/crash-logs/`. Set
`RUST_BACKTRACE=1` for a full trace. `gwen doctor` verifies this directory structure is
present and writable.

---

## Project structure

```
gwen-cli/                — Rust workspace (resolver 2, edition 2024)
  packages/core/         — gwenland-core: inference, training, benchmark, storage,
                           diagnostics — all the real logic (lib crate)
  packages/tui/          — gwenland-tui: the `gwenland` CLI + ratatui TUI binary
  packages/gui/src-tauri — Tauri 2 desktop shell (shares core's crash reporting)
  changelog/             — full per-session change history
  benchmark/             — dequant/quant math experiments (sibling of gwen-cli/)
```

The release profile is tuned for size: `opt-level = "z"`, fat LTO, single codegen unit,
`panic = "abort"`, symbols stripped — that's how the binary stays under 50 MB.

---

## Development

```bash
# Build / check the whole workspace
cargo check --workspace

# Run the core test suite (single-threaded: some tests touch process-global state
# like std::panic::set_hook and the GWEN_HOME test env)
cargo test -p gwenland-core --lib -- --test-threads=1

# Benchmarks
cargo bench -p gwenland-core
```

Each change session is recorded under [changelog/](changelog/) (`Gwen-Changes-<date>.md`),
and new optimisation candidates for the CPU training path are triaged in
[`NewExperiment.md`](../NewExperiment.md) before they graduate into a tracked `GWEN-XXX`
issue and a `.kiro/specs/` spec.

---

## Privacy

Inference and training run entirely on your machine. The only network calls are explicit
model/dataset downloads and pushes you initiate via `gwen fetch` / `gwen hub`. See
[PRIVACY.md](PRIVACY.md) for details.

---

## License

MIT with Commons Clause. See [LICENSE.txt](LICENSE.txt) for details.
Free for personal and research use. Commercial use requires a separate agreement.
