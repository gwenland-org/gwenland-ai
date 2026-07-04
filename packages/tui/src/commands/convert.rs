/// `gwen convert` — GGUF ↔ SafeTensors format conversion.
///
/// All conversion math lives in gwen-core. This file owns only:
///   - CLI arg definitions (clap)
///   - Progress line printing (per-tensor)
///   - Final summary rendering
///   - Error formatting and exit codes
///
/// This keeps the crate boundary clean: gwen-tui never touches the binary
/// format internals, only calls typed functions from gwenland_core::convert.
use clap::{Args, Subcommand};
use gwenland_core::convert::{convert_gguf, DequantMode, TensorProgress};
use std::path::PathBuf;

// ── Top-level arg struct ──────────────────────────────────────────────────────

#[derive(Args, Debug)]
#[command(
    about = "Convert model format (GGUF ↔ SafeTensors)",
    long_about = "Convert model files between GGUF and SafeTensors formats.\n\n\
                  Examples:\n  \
                    gwen convert gguf ./models/qwen3-8b.gguf\n  \
                    gwen convert gguf ./models/qwen3-8b.gguf --euler\n  \
                    gwen convert st   ./models/qwen3-8b.safetensors\n\n\
                  Dequant modes (--euler):\n  \
                    default   Linear dequant: W = X*scale + zero_point\n  \
                    --euler   GwenTensor Euler projection: W = cos(θ)*δ_b/φ\n  \
                              Output bounded to [-0.618, 0.618]."
)]
pub struct ConvertArgs {
    #[command(subcommand)]
    pub action: ConvertCommands,
}

#[derive(Subcommand, Debug)]
pub enum ConvertCommands {
    /// Convert a GGUF file to SafeTensors format
    Gguf(GgufArgs),
    /// Convert SafeTensors to GGUF (coming soon)
    St(StArgs),
}

// ── gguf subcommand args ──────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct GgufArgs {
    /// Path to the source .gguf file
    pub path: PathBuf,

    /// Use Euler/GwenTensor cosine projection instead of linear dequant.
    /// Output weights bounded to [-0.618, 0.618]; sweet spot [-0.309, 0.309].
    #[arg(long)]
    pub euler: bool,
}

// ── st subcommand args ────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct StArgs {
    /// Path to the source .safetensors file
    pub path: PathBuf,
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

/// Entry point called from main.rs. Dispatches to the appropriate subcommand.
pub fn run_convert_cmd(args: ConvertArgs) {
    match args.action {
        ConvertCommands::Gguf(a) => run_gguf(a),
        ConvertCommands::St(a)   => run_st(a),
    }
}

// ── gguf → safetensors impl ───────────────────────────────────────────────────

/// ANSI escape code for "Gwen Orange" (#FF8C42 / RGB 255,140,66).
/// Used for tensor names in progress lines so they're visually distinct from
/// the counter and timing metadata on the same line.
const GWEN_ORANGE: &str = "\x1b[38;2;255;140;66m";
const RESET: &str = "\x1b[0m";

fn run_gguf(args: GgufArgs) {
    if !args.path.exists() {
        eprintln!("error: file not found: '{}'", args.path.display());
        std::process::exit(1);
    }

    let mode = if args.euler {
        DequantMode::Euler
    } else {
        DequantMode::Standard
    };

    // `convert_gguf` calls this closure once per tensor. We print immediately
    // on each call rather than buffering so the user sees real-time progress.
    let progress_cb = |p: TensorProgress| {
        print_tensor_line(&p);
    };

    let result = match convert_gguf(&args.path, mode, progress_cb) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    // Print Euler sweet-spot warning before the summary box so it visually
    // precedes the "Conversion Complete" header rather than appearing after.
    if let Some(pct) = result.euler_warning {
        println!(
            "\x1b[33m⚠ Euler mode: {:.0}% weights outside GwenTensor sweet spot [-0.309, 0.309]\x1b[0m",
            pct
        );
        println!("  Consider --mode standard for general-purpose conversion.");
    }

    print_summary(&result);
}

/// Print one progress line for a converted tensor.
///
/// Format:
///   Converting [3/42] <ORANGE>token_embd.weight<RESET> (Q4_0, 4096×32000) ... ✓ 124ms
fn print_tensor_line(p: &TensorProgress) {
    // Build the shape string: join dimensions with "×" (×, U+00D7).
    let shape_str: Vec<String> = p.shape.iter().map(|d| d.to_string()).collect();
    let shape_display = shape_str.join("\u{00D7}");

    println!(
        "Converting [{}/{}] {}{}{} ({}, {}) ... \u{2713} {}ms",
        p.index,
        p.total,
        GWEN_ORANGE,
        p.name,
        RESET,
        p.dtype,
        shape_display,
        p.elapsed_ms,
    );
}

/// Print the final summary box after all tensors are converted.
fn print_summary(result: &gwenland_core::convert::ConvertResult) {
    let sep = "\u{2501}".repeat(32); // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    let mode_str = match result.mode {
        DequantMode::Standard => "Standard",
        DequantMode::Euler    => "Euler",
    };

    // Format elapsed: use "3.2s" for >= 1s, "420ms" for sub-second.
    let time_str = if result.elapsed_secs >= 1.0 {
        format!("{:.1}s", result.elapsed_secs)
    } else {
        format!("{:.0}ms", result.elapsed_secs * 1000.0)
    };

    println!("\u{2726} Conversion Complete");
    println!("{}", sep);
    println!("  {:<20} {}", "Tensors converted", result.tensors_converted);
    println!("  {:<20} {}", "Mode", mode_str);
    println!("  {:<20} {}", "Output", result.output_path.display());
    println!("  {:<20} {}", "Time", time_str);
    println!("{}", sep);
}

// ── st stub ───────────────────────────────────────────────────────────────────

fn run_st(_args: StArgs) {
    println!("coming soon");
}
