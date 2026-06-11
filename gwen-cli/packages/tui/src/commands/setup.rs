// @INFO: Command definition and execution for `gwen setup`.
// One-command bootstrap: scaffolds config, runs all checks with force, prints manual action summary.

use gwenland_core::doctor::{run_all_checks, CheckStatus};
use gwenland_core::storage::config::GwenConfig;
use gwenland_core::storage::paths::config_toml_path;

pub async fn run_setup(mode: gwenland_core::engine::GwenMode) {
    // Step 1: scaffold config.toml if missing (GwenConfig::load() handles migration of old config.json)
    let toml_path = config_toml_path();

    if !toml_path.exists() {
        let cfg = GwenConfig::default();
        match cfg.save() {
            Ok(_) => {
                if !mode.non_interactive {
                    println!("✓ created ~/.config/gwen/config.toml");
                }
            }
            Err(e) => {
                eprintln!("error: could not write config.toml: {}", e);
                std::process::exit(1);
            }
        }
    }

    // Step 2: run all checks with force=true, safe=false
    let results = run_all_checks(false, true).await;

    // Step 3: collect unresolved Fail items (Warning and auto-fixed are excluded)
    let unresolved: Vec<_> = results
        .iter()
        .filter(|r| r.status == CheckStatus::Fail && !r.fix_applied)
        .collect();

    if mode.non_interactive {
        if mode.json {
            let obj = serde_json::json!({
                "ok": unresolved.is_empty(),
                "unresolved": unresolved.iter().map(|r| serde_json::json!({
                    "name": r.name,
                    "value": r.value,
                })).collect::<Vec<_>>(),
            });
            println!("{}", obj);
        }
        std::process::exit(if unresolved.is_empty() { 0 } else { 1 });
    }

    let sep = "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━";

    println!("{}", sep);

    if unresolved.is_empty() {
        println!("  ✓ Setup complete! Environment is ready.");
        println!("    Run: gwen fetch -m jinxsuperdev/gwen1.0-code-mini");
        println!("{}", sep);
        std::process::exit(0);
    }

    println!(
        "  Setup complete! {} item{} need manual action.",
        unresolved.len(),
        if unresolved.len() == 1 { "" } else { "s" }
    );
    println!();

    for r in &unresolved {
        println!("  {}   {}", r.name, r.value);
        print_guides(&r.name);
        println!();
    }

    println!("{}", sep);
    std::process::exit(1);
}

fn print_guides(name: &str) {
    match name {
        "python" => {
            let install_cmd = match std::env::consts::OS {
                "macos" => "brew install python@3.11",
                "windows" => "winget install Python.Python.3.11",
                _ => "sudo apt install python3.11",
            };
            println!("           → {}", install_cmd);
            println!("           → https://python.org/downloads");
        }
        "cuda" => {
            let note = match std::env::consts::OS {
                "macos" => "CUDA is not supported on macOS.",
                _ => "Not required for CPU-only inference.",
            };
            println!("           → {}", note);
            println!("           → https://developer.nvidia.com/cuda-downloads");
        }
        _ => {}
    }
}
