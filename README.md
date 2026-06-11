# GwenLand

> **"Speed is Everything, but Precise is more than Everything."**

AI all-in-one toolkit. Local-first. &lt;50MB. Privacy-first.

A unified CLI for the full LLM lifecycle — **fetch → train → serve → chat** — with dataset tooling and HuggingFace Hub integration built in. All inference runs locally. No data leaves your machine.

---

## Why Rust

| Others | GwenLand |
|---|---|
| Python for accessibility | Rust for precision |
| Abstractions on abstractions | Lean, direct, no runtime overhead |
| &lt;50MB? Impossible in Python | &lt;50MB. Stripped. Precise. |

---

## Commands

| Command | What it does |
|---|---|
| `gwen fetch` | Download models from HuggingFace with quantisation selection |
| `gwen train` | Fine-tune via LoRA — native Rust/Candle backend |
| `gwen serve` | Spawn a local inference server |
| `gwen chat` | Streaming TUI chat with conversation history |
| `gwen benchmark` | Cold-start, inference, layer-load, and memory benchmarks |
| `gwen hub` | HuggingFace Hub — list, pull, push, info, prune |
| `gwen dataset` | Validate, convert, and split JSONL training datasets |
| `gwen scan` | Safety scanner — PII, toxicity, prompt injection |
| `gwen convert` | GGUF → SafeTensors dequantisation |
| `gwen eval` | Evaluate model on a validation dataset |
| `gwen config` | Manage configuration |
| `gwen doctor` | Environment health check |

---

## Structure

```
gwen-cli/          — Rust workspace (core library + TUI binary + GUI)
  packages/core/   — All inference, training, benchmark, and storage logic
  packages/tui/    — CLI binary (gwenland)
  packages/gui/    — Tauri desktop app
  changelog/       — Full per-session change history
```

---

## License

MIT with Commons Clause. See [LICENSE](LICENSE) for details.
Free for personal and research use. Commercial use requires a separate agreement.
