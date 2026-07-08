//! `gwen` — the GwenLand AI command-line interface.
//!
//! * `gwen run <model> --prompt "Hello"` — one-shot inference
//! * `gwen run <model>` — interactive REPL
//! * `gwen info <model>` — print model metadata (GGUF or safetensors)
//! * `gwen tui` — launch the terminal UI (stub in M1)

use std::io::Write as _;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use glcore::engine_trait::InferInput;
use glcore::format::gguf::{GgufFile, GgufValue};
use glcore::format::safetensors::SafetensorsFile;
use glcore::runtime::Runtime;
use glcore::GlError;
use glproc::GlprocEngine;

#[derive(Parser)]
#[command(name = "gwen", version, about = "GwenLand AI — local inference, pure Rust")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run inference on a model (one-shot with --prompt, REPL without)
    Run {
        /// Path to a .gguf or .safetensors model file
        model: String,
        /// Prompt text; omit to enter interactive REPL mode
        #[arg(long)]
        prompt: Option<String>,
        /// Maximum number of tokens to generate
        #[arg(long, default_value_t = 256)]
        max_tokens: usize,
        /// Sampling temperature (0 = greedy)
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// Top-k sampling cutoff (0 = disabled)
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// Top-p (nucleus) sampling cutoff (1.0 = disabled)
        #[arg(long, default_value_t = 0.95)]
        top_p: f32,
        /// Repetition penalty over the last 64 tokens (1.0 = disabled)
        #[arg(long, default_value_t = 1.1)]
        repeat_penalty: f32,
        /// Encode the prompt as raw completion text, skipping the chat
        /// template even for chat models
        #[arg(long)]
        raw: bool,
    },
    /// Print model metadata from a GGUF or safetensors file
    Info {
        /// Path to a .gguf or .safetensors model file
        model: String,
    },
    /// Launch the terminal UI
    Tui,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.command {
        Commands::Run {
            model,
            prompt,
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            raw,
        } => cmd_run(
            &model,
            prompt.as_deref(),
            max_tokens,
            temperature,
            top_k,
            top_p,
            repeat_penalty,
            raw,
        ),
        Commands::Info { model } => cmd_info(&model),
        Commands::Tui => cmd_tui(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn cmd_run(
    model: &str,
    prompt: Option<&str>,
    max_tokens: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    raw: bool,
) -> Result<(), GlError> {
    let mut runtime = Runtime::new(Box::new(GlprocEngine::new()))?;
    runtime.set_raw_prompt(raw);
    eprintln!("loading {model} ...");
    runtime.load(model)?;
    eprintln!("model loaded.");

    let config = InferInput {
        token_ids: Vec::new(), // filled by Runtime from the prompt
        max_new_tokens: max_tokens,
        temperature,
        top_k,
        top_p,
        repeat_penalty,
    };

    match prompt {
        Some(p) => {
            stream_answer(&runtime, p, config)?;
        }
        None => {
            eprintln!("interactive mode — empty line or Ctrl+C to exit\n");
            loop {
                eprint!("> ");
                let _ = std::io::stderr().flush();
                let mut line = String::new();
                if std::io::stdin().read_line(&mut line)? == 0 {
                    break; // EOF
                }
                let line = line.trim();
                if line.is_empty() {
                    break;
                }
                stream_answer(&runtime, line, config.clone())?;
            }
        }
    }
    runtime.shutdown();
    Ok(())
}

/// Stream one generation to stdout, token by token, then report prefill
/// and generation speed separately — a single blended tok/s hides the
/// real decode rate behind prompt-processing time.
fn stream_answer(runtime: &Runtime, prompt: &str, config: InferInput) -> Result<(), GlError> {
    let out = runtime.stream(prompt, config, |piece| {
        print!("{piece}");
        let _ = std::io::stdout().flush();
    })?;
    println!();
    if out.tokens_generated == 0 {
        return Ok(());
    }
    let tps = |tokens: usize, ms: f64| {
        if ms > 0.0 {
            tokens as f64 / (ms / 1000.0)
        } else {
            0.0
        }
    };
    let prefill_tps = tps(out.prompt_tokens, out.prefill_ms);
    let gen_tps = tps(out.tokens_generated, out.generation_ms);
    eprintln!(
        "[benchmark] prefill: {} tokens @ {prefill_tps:.2} tok/s | \
         generation: {} tokens @ {gen_tps:.2} tok/s",
        out.prompt_tokens, out.tokens_generated
    );
    eprintln!(
        "-- {} tokens in {:.2}s ({gen_tps:.2} tok/s generation) --",
        out.tokens_generated,
        out.generation_ms / 1000.0
    );
    Ok(())
}

fn cmd_info(model: &str) -> Result<(), GlError> {
    let lower = model.to_ascii_lowercase();
    if lower.ends_with(".gguf") {
        info_gguf(model)
    } else if lower.ends_with(".safetensors") {
        info_safetensors(model)
    } else {
        Err(GlError::Parse(
            "unknown model extension (expected .gguf or .safetensors)".into(),
        ))
    }
}

fn info_gguf(path: &str) -> Result<(), GlError> {
    let g = GgufFile::open(path)?;
    println!("format:        GGUF v{}", g.header.version);
    println!("tensors:       {}", g.header.tensor_count);
    println!("metadata keys: {}", g.header.metadata_kv_count);
    println!();

    let mut keys: Vec<&String> = g.metadata.keys().collect();
    keys.sort();
    for key in keys {
        if let Some(v) = g.get_meta(key) {
            println!("{key} = {}", brief_value(v));
        }
    }

    println!();
    let shown = g.tensors.len().min(12);
    for t in &g.tensors[..shown] {
        println!(
            "tensor {:40} {:?} dims={:?}",
            t.name, t.dtype, t.dimensions
        );
    }
    if g.tensors.len() > shown {
        println!("... and {} more tensors", g.tensors.len() - shown);
    }
    Ok(())
}

/// Render a metadata value compactly — long arrays (vocabularies) elided.
fn brief_value(v: &GgufValue) -> String {
    match v {
        GgufValue::Array(items) if items.len() > 8 => {
            format!("[array of {} values]", items.len())
        }
        GgufValue::String(s) if s.len() > 80 => format!("{:?}...", &s[..80]),
        other => format!("{other:?}"),
    }
}

fn info_safetensors(path: &str) -> Result<(), GlError> {
    let st = SafetensorsFile::open(path)?;
    println!("format:  safetensors");
    println!("tensors: {}", st.tensors.len());
    println!();
    let mut names = st.tensor_names();
    names.sort();
    for name in names.iter().take(24) {
        if let Some(meta) = st.tensors.get(*name) {
            println!("tensor {:40} {} shape={:?}", name, meta.dtype, meta.shape);
        }
    }
    if names.len() > 24 {
        println!("... and {} more tensors", names.len() - 24);
    }
    Ok(())
}

fn cmd_tui() -> Result<(), GlError> {
    // M1 stub: gltui still speaks to the legacy server backend. It gets
    // rewired onto Runtime in M2.
    eprintln!("gwen tui: not wired to the GL engines yet — coming in M2.");
    eprintln!("Meanwhile, run the standalone TUI with: cargo run -p gltui");
    Ok(())
}
