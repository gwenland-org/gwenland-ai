// @INFO: Command definition and execution for `gwen doctor`.
// @EDITABLE: Yes, this file should be expanded if the CLI arguments for doctor change.

use clap::Args;
use gwenland_core::doctor::{run_all_checks, CheckStatus};
use std::path::PathBuf;

// @INFO: CLI Arguments structure for `gwen doctor`.
// Defines the `--safe`, `--force`, and `--output` flags. 
// @EDITABLE: Yes, you can add more arguments for the doctor command here.
#[derive(Args, Debug)]
#[command(
    about = "Check environment health (CUDA, VRAM, Python deps)",
    long_about = "Check system environment health: CUDA/ROCm drivers, VRAM, Python installation,\n\
                  mistralrs-server availability, and all GwenLand dependencies.\n\n\
                  Exit code 0 = all checks pass (or all failures auto-fixed with --force).\n\
                  Exit code 1 = one or more checks failed.\n\n\
                  Examples:\n  \
                    gwen doctor                  # interactive check with fix suggestions\n  \
                    gwen doctor --safe           # read-only, no fixes, no network calls\n  \
                    gwen doctor --force          # auto-apply all available fixes\n  \
                    gwen doctor -o report.json   # save full JSON report alongside stdout output"
)]
pub struct DoctorArgs {
    /// Read-only check — zero side effects (no fixes, no spawns, no network calls).
    /// Conflicts with --force.
    #[arg(short = 's', long, conflicts_with = "force")]
    pub safe: bool,

    /// Auto-apply all available fixes without prompts. Conflicts with --safe.
    #[arg(short = 'f', long, conflicts_with = "safe")]
    pub force: bool,

    /// Write a full JSON report to FILE in addition to stdout output (e.g. -o report.json)
    #[arg(short = 'o', long, value_name = "FILE")]
    pub output: Option<PathBuf>,

    /// Specific GGUF file(s) to probe for training readiness instead of scanning the models dir.
    /// Accepts one or more paths (repeat the flag or separate with spaces).
    /// When omitted, all GGUFs in the default models directory are scanned automatically.
    #[arg(long = "model", value_name = "GGUF", num_args = 1..)]
    pub model: Vec<PathBuf>,
}

// @INFO: The main runner for the doctor command in TUI.
// It executes the checks from core and formats the output for the terminal.
// Includes graceful degradation for non-TTY terminals via `atty` and optionally writes a JSON report.
// @EDITABLE: Yes, update this function if you want to change the visual formatting or add new display metrics.
pub async fn run_doctor(args: DoctorArgs) {
    let results = run_all_checks(args.safe, args.force, args.model).await;
    
    let use_color = atty::is(atty::Stream::Stdout);
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    let mut total_fails = 0;
    
    for r in &results {
        let (icon, color) = match r.status {
            CheckStatus::Pass => ("✓", "\x1b[32m"),
            CheckStatus::Fail => {
                if r.fix_applied && r.fix_succeeded {
                    ("✓", "\x1b[32m")
                } else {
                    total_fails += 1;
                    ("✗", "\x1b[31m")
                }
            },
            CheckStatus::Warning => ("⚠", "\x1b[33m"),
            CheckStatus::NotApplicable => ("—", "\x1b[90m"),
        };
        
        let reset = "\x1b[0m";
        let c_icon = if use_color { format!("{}{}{}", color, icon, reset) } else { icon.to_string() };
        let c_name = format!("{:10}", r.name);
        let c_value = format!("{:15}", r.value);
        
        if args.force && r.fix_applied {
             if r.fix_succeeded {
                 println!("  {} {} {} (✓ fixed)", c_name, c_value, c_icon);
             } else {
                 println!("  {} {} {} (✗ fix failed)", c_name, c_value, c_icon);
             }
        } else {
             if let Some(sugg) = &r.suggestion {
                 if !args.safe && r.status != CheckStatus::Pass && r.status != CheckStatus::NotApplicable {
                     println!("  {} {} {}  → {}", c_name, c_value, c_icon, sugg);
                 } else {
                     println!("  {} {} {}", c_name, c_value, c_icon);
                 }
             } else {
                 println!("  {} {} {}", c_name, c_value, c_icon);
             }
        }
    }
    
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    
    if total_fails > 0 {
        if !args.force && !args.safe {
            println!("  {} issues found. Run `gwen doctor -f` to auto-fix.", total_fails);
        } else if args.force {
            println!("  {} issues failed to fix.", total_fails);
        } else {
            println!("  {} issues found.", total_fails);
        }
    } else {
        println!("  All clear! Ready for deployment.");
    }
    
    if let Some(out_path) = args.output {
        let mut total = 0;
        let mut passed = 0;
        let mut warnings = 0;
        let mut not_applicable = 0;
        let mut failed = 0;
        let mut fixed = 0;
        let mut fix_failed = 0;
        
        for r in &results {
            total += 1;
            match r.status {
                CheckStatus::Pass => passed += 1,
                CheckStatus::Warning => warnings += 1,
                CheckStatus::NotApplicable => not_applicable += 1,
                CheckStatus::Fail => {
                    if r.fix_applied {
                        if r.fix_succeeded {
                            fixed += 1;
                            passed += 1;
                        } else {
                            failed += 1;
                            fix_failed += 1;
                        }
                    } else {
                        failed += 1;
                    }
                }
            }
        }
        
        let report = serde_json::json!({
            "mode": if args.force { "force" } else if args.safe { "safe" } else { "default" },
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "checks": results,
            "summary": {
                "total": total,
                "passed": passed,
                "warnings": warnings,
                "not_applicable": not_applicable,
                "failed": failed,
                "fixed": fixed,
                "fix_failed": fix_failed
            }
        });
        
        if let Ok(json) = serde_json::to_string_pretty(&report) {
            let _ = std::fs::write(out_path, json);
        }
    }
    
    if total_fails > 0 {
        std::process::exit(1);
    }
}
