//! `glconv` — GGUF → GLLM package converter CLI (ARTX07-lite + Pridwen v5
//! Phase 1 `--quant`/`--policy`).
//!
//! Usage: `glconv <input.gguf> <output_dir> [--model-id ID] [--quant GQ4A] [--policy CPP]`
//! Build with: `cargo build -p glictus-caliburni --features converter --bin glconv`

use std::path::PathBuf;
use std::process::ExitCode;

use glictus_caliburni::converter::{ConvertOptions, QuantPolicy, QuantTarget, convert};

fn usage() -> ! {
    eprintln!(
        "usage: glconv <input.gguf> <output_dir> [--model-id ID] [--quant GQ4A] [--policy CPP]"
    );
    std::process::exit(2);
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(input) = args.next() else { usage() };
    let Some(out_dir) = args.next() else { usage() };
    let mut opts = ConvertOptions::default();
    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--model-id" => match args.next() {
                Some(id) => opts.model_id = Some(id),
                None => usage(),
            },
            "--quant" => match args.next().as_deref() {
                Some("GQ4A") => opts.quant = QuantTarget::Gq4a,
                _ => {
                    eprintln!("--quant only supports GQ4A in Phase 1 (Pridwen v5 §14)");
                    usage();
                }
            },
            "--policy" => match args.next().as_deref() {
                Some("CPP") => opts.policy = QuantPolicy::Cpp,
                _ => {
                    eprintln!("--policy only supports CPP in Phase 1 (Pridwen v5 §7 Stage 1)");
                    usage();
                }
            },
            _ => usage(),
        }
    }

    match convert(&PathBuf::from(&input), &PathBuf::from(&out_dir), &opts) {
        Ok(report) => {
            println!(
                "OK: {} -> {} ({} layers, {} shared tensors)",
                input, out_dir, report.num_layers, report.shared_tensors
            );
            for w in &report.warnings {
                println!("warning: {w}");
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
