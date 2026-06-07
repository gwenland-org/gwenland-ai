# GwenLand - EXPERIMENTAL

> **"Speed is Everything, but Precise is more than Everything."**

**AI all-in-one toolkit. Local-first. &lt;50MB. Privacy-first.**

While others use Python, we use Rust. Not because it's easy — because it's precise.

A unified CLI for the full LLM lifecycle — **fetch → train → serve → chat** — with dataset tooling and HuggingFace Hub integration built in. All inference runs locally. No data leaves your machine.

> Version: `1.0.0` — Actively developed

---

## The Philosophy

| Others | GwenLand |
|---|---|
| "We use Python for accessibility!" | "We use Rust for precision." |
| Abstractions on abstractions | Lean, direct, no runtime overhead |
| Convenience over correctness | Correctness *is* the convenience |
| &lt;50MB? Impossible in Python. | &lt;50MB. Stripped. Precise. |

**Speed is Everything. But Precise is more than Everything.**

GwenLand is built on the belief that an AI toolkit should be as rigorous as the models it trains. Every byte counts. Every millisecond counts. Every guarantee the type system gives you counts.

---

## Features

| Category | What it does |
|---|---|
| **Fetch** | Download models from HuggingFace with quantisation selection |
| **Train** | Fine-tune via LoRA/QLoRA — Python (Unsloth/TRL) or native Rust/Candle backend |
| **Eval** | Evaluate model performance on a validation dataset |
| **Serve** | Spawn a local mistral.rs inference server with auto-pull |
| **Chat** | Streaming TUI chat with conversation history |
| **Hub** | Full HuggingFace Hub integration — list, pull, push, info, prune for models and datasets |
| **Dataset** | Validate, convert, and split JSONL training datasets |
| **Scan** | Safety scanner — PII, toxicity, prompt injection, bias, balance checks |
| **Convert** | GGUF → SafeTensors with standard or Euler dequantisation |
| **Benchmark** | Cold-start, inference, convert pipeline, and memory benchmarks |
| **Config** | Manage GwenLand configuration (TOML-based, cross-platform paths) |
| **Doctor** | Environment health check — CUDA, VRAM, Python deps |

---

## Installation

### Prerequisites

- Rust toolchain (stable, edition 2024)
- `git` CLI in PATH (for version embedding via `vergen`)
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) binary in PATH or `~/.cargo/bin/` (required for `gwen serve` and `gwen chat`)
- `HF_TOKEN` environment variable or OS keyring entry for private HuggingFace models

### Build from source

```sh
git clone https://github.com/JinXSuper/gwenland
cd gwenland
cargo build --release -p gwenland-tui
```

The binary lands at `target/release/gwenland`. Alias `gwen` is created by `gwenland setup`.

#### Optional feature flags

```sh
# CPU-only native Rust/Candle training
cargo build --release -p gwenland-tui --features gwenland-core/candle

# CUDA-accelerated Candle (requires CUDA toolkit)
cargo build --release -p gwenland-tui --features gwenland-core/cuda

# Bundle mistral.rs core directly into the binary
cargo build --release -p gwenland-tui --features gwenland-core/bundled

# NVIDIA GPU VRAM detection without nvidia-smi
cargo build --release -p gwenland-tui --features gwenland-core/nvidia
```

### npm (wrapper)

```sh
npm install -g @jinxsuper/gwenland
```

The postinstall script downloads the pre-built binary for your platform.

---

## Usage

```
gwenland — AI all-in-one toolkit. Local-first, <50MB, privacy-first.
Alias: gwen → gwenland (added by `gwenland setup`)

Usage: gwenland [OPTIONS] <COMMAND>

Commands:
  doctor     Check environment health (CUDA, VRAM, Python deps)
  fetch      Download base model from HuggingFace
  train      Fine-tune a model (LoRA/QLoRA)
  eval       Evaluate model performance on a validation dataset
  serve      Start local inference server
  chat       Chat with local model (TUI)
  hub        HuggingFace Hub integration (model list, pull, push, info, prune)
  dataset    Dataset management (validate/convert/split)
  scan       Safety scanner for models and datasets
  convert    Convert model format (GGUF ↔ SafeTensors)
  benchmark  Benchmark GwenLand runtime
  config     Manage GwenLand user configuration
  update     Self-update GwenLand to the latest release

Global Options:
      --json              Structured JSON output
  -n, --non-interactive   Agent/script mode — no TUI, no spinners, no interactive prompts
      --dry-run           Pre-flight validation only — no side effects
  -y, --yes               Auto-confirm all [Y/n] prompts
  -V, --version           Print version information and exit
  -h, --help              Print help
```

### Quick start

```sh
# 1. Check your environment
gwen doctor

# 2. Download a model
gwen fetch -m mistralai/Mistral-7B-v0.1 -q q4_k_m

# 3. Validate your training dataset
gwen dataset validate --input data.jsonl --strict

# 4. Dry-run a training job (checks VRAM, disk, config — no training)
gwen train --dry-run -c config.yaml -m mistralai/Mistral-7B-v0.1 -d data.jsonl

# 5. Train
gwen train -c config.yaml -m mistralai/Mistral-7B-v0.1 -d data.jsonl -o ./output

# 6. Serve the model locally
gwen serve --model mistralai/Mistral-7B-v0.1

# 7. Chat
gwen chat
```

---

## Command Reference

### `gwen fetch`

Downloads a model from HuggingFace Hub into the local cache.

```sh
gwen fetch -m mistralai/Mistral-7B-v0.1 -q q4_k_m
```

- Quantisation (`-q`): `4bit`, `8bit`, `16bit`, `fp16`, `fp32`
- Required in `--non-interactive` mode (no interactive quant picker)

---

### `gwen train`

Fine-tunes a model using LoRA/QLoRA. GwenLand supports two backends — and both are intentional:

- **Python backend** (default): Runs Unsloth + TRL via a bundled `base_train.py` script. Progress is streamed as JSON lines to a live TUI panel.
- **Native Rust/Candle backend** (`--features gwenland-core/candle`): Pure-Rust LoRA training with gradient accumulation and SafeTensors checkpoints every 500 steps. No Python. No subprocess. Precise.

```sh
# With YAML config
gwen train -c config.yaml -m mistralai/Mistral-7B-v0.1 -d dataset.jsonl -o ./output

# Dry-run — VRAM estimate + full pre-flight, no training
gwen train --dry-run -c config.yaml -m mistralai/Mistral-7B-v0.1 -d dataset.jsonl
```

**Config file (`config.yaml`) fields:**

| Field | Default | Description |
|---|---|---|
| `epochs` | `3` | Number of training epochs |
| `batch_size` | `1` | Per-step batch size |
| `grad_accum` | `16` | Gradient accumulation steps |
| `lr` | `1e-4` | Learning rate |
| `max_seq_len` | `1024` | Maximum token sequence length |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha |
| `qlora` | `false` | Enable 4-bit QLoRA |

**Training TUI keybinds:**

| Key | Action |
|---|---|
| `P` | Pause / resume training |
| `L` | Toggle full log view |
| `S` | Toggle status overlay |
| `Q` | Detach TUI (training continues) |

**Dry-run output example:**

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Config           (loaded)                 ✓
  Dataset          dataset.jsonl            ✓  (4,998 valid samples)
  Base model       Mistral 7B               ✓
  Output dir       ./output                 ✓  (142 GB free)

  VRAM Breakdown
  ────────────────────────────────────────────────
  Base model           7B @ 4-bit              4.2 GB
  LoRA adapters        r=8, target: q,v_proj   0.1 GB
  Activations          batch=1, seq=1024       1.0 GB
  Optimizer            AdamW 8-bit             1.7 GB
  Safety buffer        +20%                    1.4 GB
  ────────────────────────────────────────────────
  Total estimated                              8.4 GB
  Available VRAM   RTX 3090               24.0 GB   ✓ fits!

  Training Estimate
  ────────────────────────────────────────────────
  Epochs               3
  Steps                14994
  Est. time            ~2h 18m  (based on T4 baseline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ✓ Ready to train. Remove --dry-run to start.
```

---

### `gwen serve`

Launches mistral.rs as a local inference server. Auto-pulls the model from HuggingFace if not cached locally.

```sh
gwen serve --model mistralai/Mistral-7B-v0.1 --port 1136
```

- Default port: `1136`
- Endpoint: `http://localhost:1136/gwenland/chat`
- JSON mode: `gwenland serve ... --json` → prints `{status, model, port, pid}`
- Dry-run: validates mistral.rs binary, model cache, and port availability — no process spawned

---

### `gwen chat`

Streaming TUI chat connected to the local server.

```sh
gwen chat --model mistralai/Mistral-7B-v0.1

# Pipe mode — non-interactive, plain text
echo "Explain LoRA in one sentence" | gwen chat --non-interactive

# NDJSON stream for agents and scripts
echo "Hello" | gwen chat --non-interactive --json
```

**TUI layout:**

```
┌─────────────────────────────────────────────────────┐
│ gwen chat                                           │
│ You    Hello, what can you do?                      │
│ Gwen   I can help with code, questions, writing…    │
│ You    Write me a Rust function to reverse a string │
│ Gwen   Sure! Here's a simple implementation:▌       │
├─────────────────────────────────────────────────────┤
│ > _                                                 │
│  Ctrl+C exit   Ctrl+L clear   ↑↓ scroll             │
└─────────────────────────────────────────────────────┘
```

**TUI keybinds:**

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Ctrl+C` | Exit |
| `Ctrl+L` | Clear history |
| `↑↓ / PgUp / PgDn` | Scroll history |

---

### `gwen hub`

Full HuggingFace Hub integration — models and datasets.

```sh
# Login (stores token in OS keyring — never written to any config file)
gwen hub model login --token hf_xxx

# List your models
gwen hub model list

# Pull a model
gwen hub model pull owner/repo-name

# Push a fine-tuned model
gwen hub model push owner/my-model ./output

# Show model info
gwen hub model info mistralai/Mistral-7B-v0.1

# Delete local cache
gwen hub model prune owner/repo-name

# Dataset operations (same subcommands)
gwen hub dataset pull owner/dataset-name
```

Token resolution order: OS keyring → `HF_TOKEN` env → hf-hub cache file. The token is **never written to any GwenLand config file**.

---

### `gwen dataset`

Validates, converts, and splits JSONL training datasets.

```sh
# Validate (ChatML format)
gwen dataset validate --input data.jsonl --strict

# Split into train/val
gwen dataset split --input data.jsonl --ratio 0.9
```

Expected JSONL format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

---

### `gwen scan`

Safety scanner for datasets and models.

```sh
gwen scan dataset --input data.jsonl --check safety,pii,injection,bias,balance
```

Exit codes: `0` clean, `1` issues found.

---

### `gwen convert`

Converts GGUF models to SafeTensors — no extra crate dependencies, no Python, no runtime surprises. The GGUF parser is hand-written in ~300 lines of pure Rust. The SafeTensors writer skips the `safetensors` crate entirely.

```sh
# Standard linear dequantisation
gwen convert gguf model.gguf

# Euler cosine-projection dequantisation
gwen convert gguf model.gguf --euler
```

Supports GGUF v1/v2/v3. Outputs `.safetensors` alongside the source file. Warns when >20% of weights fall outside the Euler sweet-spot `[-0.309, 0.309]`.

---

### `gwen benchmark`

Benchmarks GwenLand runtime performance.

```sh
# Run all suites
gwen benchmark --full

# Individual suites
gwen benchmark --cold-start
gwen benchmark --inference    # requires Ollama running locally
gwen benchmark --convert
```

Suites: cold-start latency (10 runs, median), Ollama token throughput, GGUF dequantisation ns/element, process memory. No criterion, no framework — plain `std::time::Instant`.

---

### `gwen eval`

Evaluates model performance on a validation dataset.

```sh
gwen eval --model mistralai/Mistral-7B-v0.1 --dataset val.jsonl
```

---

### `gwen doctor`

Checks environment health: CUDA presence, VRAM, Python version, required dependencies.

```sh
gwen doctor
```

---

## Global Flags

All flags apply to every subcommand without repetition.

| Flag | Short | Description |
|---|---|---|
| `--json` | | Structured JSON output; combine with `--non-interactive` for NDJSON streams |
| `--non-interactive` | `-n` | No TUI, no spinners, no prompts. Auto-enabled when stdout is not a TTY |
| `--dry-run` | | Pre-flight validation only — no side effects, no spawns, no network calls |
| `--yes` | `-y` | Auto-confirm all `[Y/n]` prompts |

Auto-detection: piping stdout (`gwen chat | jq .`) automatically enables `--non-interactive`. No flag needed.

---

## Configuration

Config is stored as TOML, managed cross-platform via the `directories` crate. No JSON, no YAML, no ambiguity.

| Platform | Path |
|---|---|
| Linux | `~/.config/gwen/config.toml` |
| macOS | `~/Library/Application Support/gwen/config.toml` |
| Windows | `AppData\Roaming\gwen\config.toml` |

Override all paths with the `GWEN_HOME` environment variable.

```sh
gwen config get general.last_used_model
gwen config set ai.token_budget 8192
```

**Config keys:**

| Key | Default | Description |
|---|---|---|
| `general.last_used_model` | `""` | Model last used by `gwen serve` |
| `general.default_port` | `1136` | Default inference server port |
| `ai.compression` | `true` | Context relevance windowing |
| `ai.token_budget` | `4096` | Max tokens per context window |
| `ai.strategy` | `"tfidf"` | Context compression strategy |

---

## Privacy

GwenLand is fully local by default. Precise about what stays on your machine.

| Data | Location | Transmitted? |
|---|---|---|
| Conversation history | `~/.config/gwen/history.jsonl` | Never |
| Session error logs | `~/.cache/gwen/sessions/` | Never |
| File contents (context injection) | Read at inference time | Only to local mistral.rs |
| Config | `~/.config/gwen/config.toml` | Never |
| HF token | OS keyring only | Never written to disk by GwenLand |

See [PRIVACY.md](gwen-cli/PRIVACY.md) for full details.

---

## Project Structure

```
GwenLand/
└── gwen-cli/
    ├── packages/
    │   ├── core/          # gwenland-core — all ML logic, storage, engine, platform
    │   │   └── src/
    │   │       ├── benchmark/     # Cold-start, inference, convert, memory benchmarks
    │   │       ├── convert/       # GGUF parser, dequantisation, SafeTensors writer
    │   │       ├── dataset/       # Validation, splitting, scanning
    │   │       ├── diagnostics/   # VRAM/time estimator
    │   │       ├── engine/        # Chat, tokenizer, windowing, GwenMode
    │   │       ├── platform/      # Hub (model/dataset), serve, hardware
    │   │       ├── storage/       # Config (TOML), registry, history, session, paths
    │   │       └── train/         # Config, LoRA layer, training loop, VRAM estimator
    │   ├── tui/           # gwenland-tui — CLI entry point, all subcommands, TUI panels
    │   │   └── src/
    │   │       ├── commands/      # One file per subcommand
    │   │       ├── panes/         # ratatui pane components
    │   │       └── tui/           # TUI panels (train panel, etc.)
    │   └── gui/           # GUI — planned for Cycle 4
    ├── tests/
    │   └── fixtures/      # CI test data (sample_100.jsonl, ci_train_config.yaml)
    └── .github/workflows/ # CI: build, E2E pipeline, E2E serve+chat
```

**Crate boundary rule:** All ML logic, storage, platform ops, and hardware detection live in `gwenland-core`. `gwenland-tui` calls only typed functions — it never references `candle_core`, `hf_hub`, `tokenizers`, `sysinfo`, or `reqwest` directly in command handlers. Precision at the module boundary.

---

## CI

Three GitHub Actions workflows run on push to `main` and all PRs — Ubuntu, macOS, Windows:

| Workflow | What it tests |
|---|---|
| `build.yml` | `cargo build --release` across all platforms |
| `e2e-pipeline.yml` | fetch → dataset validate → scan → train `--dry-run` → hub push `--dry-run` |
| `e2e-serve.yml` | Serve subprocess teardown, SSE stream contract, SIGTERM handling |

---

## Version

```
gwenland 1.0.0 (abc1234)
```

Format: `gwenland {VERSION} ({git-short-SHA})`. SHA is embedded at compile time via `vergen` — precise provenance for every binary.

---

## Links

- Homepage: [jinxsuper.vercel.app](https://jinxsuper.vercel.app)
- Documentation: [gwenland.vercel.app](https://gwenland.vercel.app)
- Repository: [github.com/JinXSuper/gwenland](https://github.com/JinXSuper/gwenland)
- Issues: [github.com/JinXSuper/gwenland/issues](https://github.com/JinXSuper/gwenland/issues)

---

## License

See `LICENSE.txt` in the repository root.

---

*Speed is Everything. But Precise is more than Everything.*
