// commands/run.rs — `gwen run` — native local inference.
//
// Dispatches to one of three execution paths:
//   --dry-run       → print pre-flight table, exit 0
//   --interactive   → REPL chat loop (stdin/stdout)
//   default         → single-shot prompt from --prompt or stdin
//
// All sampling parameters (temperature, top-p, repeat-penalty, max-tokens)
// are forwarded directly to the inference runner. The --auto-stop flag wires
// into MemoryGuard so generation halts gracefully if RAM exceeds the threshold
// rather than causing an OOM kill.

use clap::Args;
use std::io::{self, BufRead, Write};

use gwenland_core::engine::inference::runner::{print_dry_run, run_inference, InferenceConfig};
use gwenland_core::engine::inference::sampler::SamplerConfig;
use gwenland_core::engine::memory_guard::MemoryGuard;

// ── CLI args ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Run inference on a local model",
    long_about = "Run inference on a local GGUF model using native Rust (candle-transformers).\n\
                  No Ollama. No Python. No external process.\n\n\
                  Model resolution order:\n  \
                    1. Exact path (./model.gguf, /abs/path.gguf)\n  \
                    2. ~/.config/gwen/models/<name>.gguf\n  \
                    3. ~/.config/gwen/models/<name>-q4_0.gguf\n  \
                    4. Error — run `gwen fetch <name>` first\n\n\
                  Examples:\n  \
                    gwen run qwen3:8b --prompt \"hello\"\n  \
                    gwen run qwen3:8b --interactive\n  \
                    gwen run ./models/custom.gguf --temperature 0.5\n  \
                    gwen run qwen3:8b --dry-run"
)]
pub struct RunArgs {
    /// Model name or path (e.g. qwen3:8b, ./models/custom.gguf)
    #[arg(required = true, value_name = "MODEL")]
    pub model: String,

    /// Prompt to run (omit for stdin or --interactive)
    #[arg(short = 'p', long, value_name = "TEXT")]
    pub prompt: Option<String>,

    /// Interactive chat loop (REPL mode)
    #[arg(short = 'i', long)]
    pub interactive: bool,

    /// Max tokens to generate
    #[arg(long, default_value = "512", value_name = "N")]
    pub max_tokens: usize,

    /// Sampling temperature — 0.0 = greedy, higher = more creative
    #[arg(long, default_value = "0.7", value_name = "F")]
    pub temperature: f32,

    /// Top-p nucleus sampling threshold [0.0, 1.0]
    #[arg(long, default_value = "0.9", value_name = "F")]
    pub top_p: f32,

    /// Repetition penalty — 1.0 = disabled, higher = less repetition
    #[arg(long, default_value = "1.1", value_name = "F")]
    pub repeat_penalty: f32,

    /// RAM safety threshold (0–100%). Stop if system RAM exceeds this.
    #[arg(short = 'A', long, default_value = "90", value_name = "PCT")]
    pub auto_stop: u8,

    /// Dry run — show what would run, do not execute
    #[arg(long)]
    pub dry_run: bool,
}

// ── entry point ────────────────────────────────────────────────────────────────

pub fn run_run_cmd(args: RunArgs) {
    // Validate auto-stop early so the user gets a clear message before loading weights.
    if args.auto_stop < 10 {
        eprintln!(
            "error: --auto-stop {} is dangerously low — minimum is 10%",
            args.auto_stop
        );
        std::process::exit(1);
    }

    let sampler = SamplerConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        repeat_penalty: args.repeat_penalty,
        max_tokens: args.max_tokens,
    };

    let base_cfg = InferenceConfig {
        model: args.model.clone(),
        // Derive tokenizer model id from the model name.
        // Users running local paths should set GWEN_TOKENIZER env to override.
        model_id_for_tokenizer: std::env::var("GWEN_TOKENIZER").unwrap_or(args.model.clone()),
        prompt: String::new(),
        sampler: sampler.clone(),
        auto_stop_pct: args.auto_stop,
        show_banner: true,
    };

    if args.dry_run {
        print_dry_run(&base_cfg);
        std::process::exit(0);
    }

    if args.interactive {
        run_interactive_loop(base_cfg);
    } else {
        let prompt = get_prompt(args.prompt);
        let cfg = InferenceConfig { prompt, ..base_cfg };
        match run_inference(&cfg, None) {
            Ok(result) => {
                if result.memory_stopped {
                    std::process::exit(1);
                }
            }
            Err(e) => {
                // `{:#}` prints the full anyhow cause chain on one line, so the
                // actionable reason (e.g. the resolve_model_path hint) is shown
                // rather than just the top-level "failed to load model".
                eprintln!("error: {:#}", e);
                std::process::exit(1);
            }
        }
    }
}

// ── interactive REPL ────────────────────────────────────────────────────────────

fn run_interactive_loop(base_cfg: InferenceConfig) {
    let stdin = io::stdin();
    let mut history: Vec<String> = Vec::new();

    eprintln!("  ❖ Interactive mode — type your message and press Enter. Ctrl+C to exit.\n");

    loop {
        print!("> ");
        let _ = io::stdout().flush();

        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => break, // EOF (Ctrl+D)
            Ok(_) => {}
            Err(e) => {
                eprintln!("error reading input: {}", e);
                break;
            }
        }

        let input = line.trim().to_string();
        if input.is_empty() {
            continue;
        }

        // Build prompt from history + new user message.
        history.push(input.clone());
        let prompt = build_chat_prompt(&history);

        let cfg = InferenceConfig {
            prompt,
            show_banner: false, // banner already shown on first run
            ..base_cfg.clone()
        };

        match run_inference(&cfg, None) {
            Ok(result) => {
                // Add the assistant response to history for next turn.
                history.push(result.generated_text.clone());
                if result.memory_stopped {
                    eprintln!("  ⚠ Memory threshold reached. Exiting interactive mode.");
                    break;
                }
            }
            Err(e) => {
                eprintln!("error: {:#}", e);
                break;
            }
        }
    }
}

// ── helpers ────────────────────────────────────────────────────────────────────

fn get_prompt(prompt_arg: Option<String>) -> String {
    if let Some(p) = prompt_arg {
        return p;
    }
    // Read from stdin (piped mode).
    let mut buf = String::new();
    let _ = io::stdin().lock().read_line(&mut buf);
    buf.trim().to_string()
}

/// Build a ChatML-style prompt from alternating user/assistant history.
fn build_chat_prompt(history: &[String]) -> String {
    let mut out = String::new();
    for (i, msg) in history.iter().enumerate() {
        let role = if i % 2 == 0 { "user" } else { "assistant" };
        out.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, msg));
    }
    out.push_str("<|im_start|>assistant\n");
    out
}
