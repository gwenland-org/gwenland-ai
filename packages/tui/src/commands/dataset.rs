use clap::{Args, Subcommand};
use gwenland_core::dataset::convert::{run_convert, OutputFormat};
use gwenland_core::dataset::info::run_info;
use gwenland_core::dataset::split::{run_split, SplitOptions};
use gwenland_core::dataset::validate::{run_validation, Severity, ValidateOptions};
use std::path::PathBuf;

#[derive(Args, Debug)]
#[command(
    about = "Dataset management (validate/convert/split)",
    long_about = "Manage JSONL training datasets: validate format, convert between schemas,\n\
                  split into train/val/test partitions, and print statistics.\n\n\
                  Supported input formats: ChatML (default), Alpaca, ShareGPT.\n\
                  Output formats for convert: gwenstyle, chatml, alpaca.\n\n\
                  Examples:\n  \
                    gwen dataset validate -i data.jsonl\n  \
                    gwen dataset validate -i data.jsonl --strict --fix\n  \
                    gwen dataset convert -i data.jsonl -o out.jsonl --format chatml\n  \
                    gwen dataset split -i data.jsonl --train 0.9 --val 0.1\n  \
                    gwen dataset split -i data.jsonl --train 0.8 --val 0.1 --test 0.1 --seed 42\n  \
                    gwen dataset info -i data.jsonl\n  \
                    gwen dataset info -i data.jsonl -m mistralai/Mistral-7B-v0.1"
)]
pub struct DatasetArgs {
    #[command(subcommand)]
    pub action: DatasetCommands,
}

#[derive(Subcommand, Debug)]
pub enum DatasetCommands {
    /// Validate a JSONL dataset file
    Validate(ValidateArgs),
    /// Convert a dataset between formats (gwenstyle, chatml, alpaca, sharegpt→*)
    Convert(ConvertArgs),
    /// Split a dataset into train / val / test files
    Split(SplitArgs),
    /// Print statistics about a dataset file
    Info(InfoArgs),
}

// ── Validate ─────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ValidateArgs {
    /// Input JSONL file path (e.g. -i data.jsonl)
    #[arg(short = 'i', long, value_name = "FILE")]
    pub input: PathBuf,

    /// Treat warnings as errors — exit code 1 if any warnings exist
    #[arg(long)]
    pub strict: bool,

    /// Output machine-readable JSON report instead of human-readable text
    #[arg(long)]
    pub json: bool,

    /// Auto-fix fixable issues and write to <basename>.fixed.jsonl alongside the original
    #[arg(long)]
    pub fix: bool,

    /// With --fix: overwrite the original file instead of creating .fixed.jsonl
    #[arg(long, requires = "fix")]
    pub inplace: bool,
}

// ── Convert ───────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct ConvertArgs {
    /// Input JSONL file path
    #[arg(short = 'i', long)]
    pub input: PathBuf,

    /// Output JSONL file path
    #[arg(short = 'o', long)]
    pub output: PathBuf,

    /// Output format: gwenstyle | chatml | alpaca
    #[arg(long = "format")]
    pub format: String,
}

// ── Split ─────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct SplitArgs {
    /// Input JSONL file path
    #[arg(short = 'i', long)]
    pub input: PathBuf,

    /// Training split ratio (e.g. 0.9)
    #[arg(long)]
    pub train: f32,

    /// Validation split ratio (e.g. 0.1)
    #[arg(long)]
    pub val: f32,

    /// Optional test split ratio (e.g. 0.1)
    #[arg(long)]
    pub test: Option<f32>,

    /// Random seed for reproducibility; if omitted a random seed is generated and printed
    #[arg(long)]
    pub seed: Option<u64>,

    /// Overwrite output files if they already exist
    #[arg(short = 'w', long)]
    pub overwrite: bool,
}

// ── Info ──────────────────────────────────────────────────────────────────────

#[derive(Args, Debug)]
pub struct InfoArgs {
    /// Input JSONL file path
    #[arg(short = 'i', long)]
    pub input: PathBuf,

    /// HF model ID for exact token counting (e.g. mistralai/Mistral-7B)
    #[arg(short = 'm', long)]
    pub model: Option<String>,
}

// ── Dispatch ──────────────────────────────────────────────────────────────────

pub async fn run_dataset(args: DatasetArgs) {
    match args.action {
        DatasetCommands::Validate(a) => run_validate(a).await,
        DatasetCommands::Convert(a) => run_convert_cmd(a),
        DatasetCommands::Split(a) => run_split_cmd(a),
        DatasetCommands::Info(a) => run_info_cmd(a),
    }
}

// ── validate impl ─────────────────────────────────────────────────────────────

async fn run_validate(args: ValidateArgs) {
    let opts = ValidateOptions {
        strict: args.strict,
        fix: args.fix,
        inplace: args.inplace,
    };

    let result = match run_validation(&args.input, &opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    if args.json {
        let issues_json: Vec<serde_json::Value> = result
            .issues
            .iter()
            .map(|i| {
                let mut obj = serde_json::json!({
                    "line": i.line,
                    "severity": match i.severity {
                        Severity::Error   => "error",
                        Severity::Warning => "warning",
                    },
                    "code": i.code,
                    "message": i.message,
                });
                if let Some(fixable) = i.fixable {
                    obj["fixable"] = serde_json::Value::Bool(fixable);
                }
                obj
            })
            .collect();

        let output = serde_json::json!({
            "total":    result.total,
            "valid":    result.valid,
            "errors":   result.error_count,
            "warnings": result.warning_count,
            "ready":    result.ready,
            "issues":   issues_json,
        });
        println!("{}", serde_json::to_string_pretty(&output).unwrap_or_default());
    } else {
        print_validate_output(&result, args.fix, args.inplace, &args.input);
    }

    if !result.ready {
        std::process::exit(1);
    }
}

fn print_validate_output(
    result: &gwenland_core::dataset::validate::ValidationResult,
    fix: bool,
    inplace: bool,
    input_path: &PathBuf,
) {
    let sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    let use_color = atty::is(atty::Stream::Stdout);
    let (green, yellow, red, reset) = color_codes(use_color);

    println!("{}", sep);
    println!("  {:<18} {}", "Total samples", format_count(result.total));
    println!("  {:<18} {}  {}✓{}", "Valid", format_count(result.valid), green, reset);

    if result.error_count > 0 {
        println!("  {:<18} {}  {}✗{}", "Errors", result.error_count, red, reset);
    } else {
        println!("  {:<18} {}", "Errors", 0);
    }
    if result.warning_count > 0 {
        println!("  {:<18} {}  {}⚠{}", "Warnings", result.warning_count, yellow, reset);
    } else {
        println!("  {:<18} {}", "Warnings", 0);
    }

    const MAX_INLINE: usize = 20;
    if !result.issues.is_empty() {
        println!();
        for issue in result.issues.iter().take(MAX_INLINE) {
            let (prefix, color) = match issue.severity {
                Severity::Error   => ("✗", red),
                Severity::Warning => ("⚠", yellow),
            };
            let fix_tag = if issue.fixable == Some(true) {
                format!("  {}[fixable]{}", yellow, reset)
            } else {
                String::new()
            };
            println!(
                "  {color}Line {:>6}{reset}:  {} {}{}",
                issue.line, prefix, issue.message, fix_tag,
                color = color, reset = reset,
            );
        }
        let remaining = result.issues.len().saturating_sub(MAX_INLINE);
        if remaining > 0 {
            println!(
                "  ... and {} more {} (run --json for full list)",
                remaining,
                if remaining == 1 { "issue" } else { "issues" }
            );
        }
    }

    println!("{}", sep);

    if result.ready {
        println!("  {}✓ Dataset ready for training.{}", green, reset);
        println!(
            "    Run: gwen train -i {} -m jinxsuperdev/gwen1.0-code-mini",
            input_path.display()
        );
    } else {
        println!(
            "  {}Dataset NOT ready for training.{} Fix {} error{} first.",
            red, reset,
            result.error_count,
            if result.error_count == 1 { "" } else { "s" }
        );
        let fixable_warnings = result
            .issues
            .iter()
            .filter(|i| i.severity == Severity::Warning && i.fixable == Some(true))
            .count();
        if fixable_warnings > 0 && !fix {
            println!(
                "  Run with --fix to auto-correct {} warning{}.",
                fixable_warnings,
                if fixable_warnings == 1 { "" } else { "s" }
            );
        }
        if fix {
            let out_label = if inplace {
                input_path.display().to_string()
            } else {
                let stem = input_path.file_stem().and_then(|s| s.to_str()).unwrap_or("dataset");
                let parent = input_path.parent().unwrap_or(std::path::Path::new("."));
                parent.join(format!("{}.fixed.jsonl", stem)).display().to_string()
            };
            println!("  Fixed rows written to: {}", out_label);
        }
    }
}

// ── convert impl ──────────────────────────────────────────────────────────────

fn run_convert_cmd(args: ConvertArgs) {
    let fmt = match OutputFormat::from_str(&args.format) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    let result = match run_convert(&args.input, &args.output, &fmt) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    let sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    let use_color = atty::is(atty::Stream::Stdout);
    let (green, yellow, _, reset) = color_codes(use_color);

    for w in &result.warnings {
        println!("  {}{}{}", yellow, w, reset);
    }

    println!("{}", sep);
    println!("  {:<18} {}", "Written", format_count(result.written));
    if result.skipped > 0 {
        println!("  {:<18} {}", "Skipped (corrupt)", result.skipped);
    }
    println!("  {:<18} {}", "Output", args.output.display());
    println!("{}", sep);
    println!("  {}✓ Conversion complete.{}", green, reset);
}

// ── split impl ────────────────────────────────────────────────────────────────

fn run_split_cmd(args: SplitArgs) {
    let opts = SplitOptions {
        input: args.input,
        train: args.train,
        val: args.val,
        test: args.test,
        seed: args.seed,
        overwrite: args.overwrite,
    };

    let result = match run_split(&opts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    let use_color = atty::is(atty::Stream::Stdout);
    let (green, yellow, _, reset) = color_codes(use_color);

    for w in &result.warnings {
        println!("  {}{}{}", yellow, w, reset);
    }

    println!("{}✓ Split complete (seed: {}){}", green, result.seed_used, reset);
    println!("  {:<32} {} samples", result.train_path.display(), format_count(result.train_count));
    println!("  {:<32} {} samples", result.val_path.display(), format_count(result.val_count));
    if let (Some(tp), Some(tc)) = (result.test_path, result.test_count) {
        println!("  {:<32} {} samples", tp.display(), format_count(tc));
    }
}

// ── info impl ─────────────────────────────────────────────────────────────────

fn run_info_cmd(args: InfoArgs) {
    let model = args.model.as_deref();
    let info = match run_info(&args.input, model) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(1);
        }
    };

    let sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";
    let use_color = atty::is(atty::Stream::Stdout);
    let (_, yellow, _, reset) = color_codes(use_color);

    for w in &info.warnings {
        println!("  {}{}{}", yellow, w, reset);
    }

    println!("{}", sep);
    println!("  {:<18} {}", "Samples", format_count(info.total));
    println!(
        "  {:<18} {:.0} tokens  ({})",
        "Avg input len", info.avg_input_tokens, info.token_label
    );
    println!(
        "  {:<18} {:.0} tokens  ({})",
        "Avg output len", info.avg_output_tokens, info.token_label
    );
    println!("  {:<18} {:.1}%", "Think ratio", info.think_ratio);

    if info.categories.is_empty() {
        println!("  {:<18} (none)", "Categories");
    } else {
        // First line inline, subsequent lines indented to align.
        let first = &info.categories[0];
        print!("  {:<18} {}: {}", "Categories", first.0, format_count(first.1));
        for (cat, count) in info.categories.iter().skip(1) {
            print!(", {}: {}", cat, format_count(*count));
        }
        println!();
    }

    println!("{}", sep);
}

// ── helpers ───────────────────────────────────────────────────────────────────

fn color_codes(use_color: bool) -> (&'static str, &'static str, &'static str, &'static str) {
    if use_color {
        ("\x1b[32m", "\x1b[33m", "\x1b[31m", "\x1b[0m")
    } else {
        ("", "", "", "")
    }
}

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}
