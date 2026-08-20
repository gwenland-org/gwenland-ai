// @INFO: Command-line entry point for gwen-core to run and verify the workspace scanner.
// @EDITABLE
use std::env;
use gwenland_core::scanner::scan_workspace;
use gwenland_core::ignore_rules::load_ignore_rules;
use gwenland_core::diagnostics::benchmark::run_benchmark;
use gwenland_core::context_tree::{build_context_tree, tree_to_json};
use gwenland_core::tokenizer::{auto_estimate_context, print_token_report};

fn main() {
    // Get path from command line arguments or use current directory as fallback
    let args: Vec<String> = env::args().collect();
    let is_benchmark = args.iter().any(|arg| arg == "--benchmark");
    let is_tree = args.iter().any(|arg| arg == "--tree");
    let is_tokens = args.iter().any(|arg| arg == "--tokens");

    // Filter out flag arguments to find the path argument, if any
    let flags = ["--benchmark", "--tree", "--tokens"];
    let path_arg = args.iter().skip(1).find(|arg| !flags.contains(&arg.as_str()));
    let scan_path = if let Some(path_str) = path_arg {
        std::path::PathBuf::from(path_str)
    } else {
        env::current_dir().expect("Failed to get current directory")
    };

    println!("GwenLand Scanner");
    println!("Target path: {}", scan_path.display());

    if is_benchmark {
        run_benchmark(&scan_path, 5);
        return;
    }

    if is_tree {
        let ignore_rules = load_ignore_rules(&scan_path);
        let ctx = build_context_tree(&scan_path, &ignore_rules);
        println!("{}", tree_to_json(&ctx));
        return;
    }

    if is_tokens {
        let ignore_rules = load_ignore_rules(&scan_path);
        let result = scan_workspace(&scan_path, Some(&ignore_rules));
        let estimate = auto_estimate_context(&result.files, "");
        print_token_report(&estimate);
        return;
    }

    println!("Scanning... \n");

    let ignore_rules = load_ignore_rules(&scan_path);
    let result = scan_workspace(&scan_path, Some(&ignore_rules));

    println!("=================== SCAN RESULTS ===================");
    println!("Total Files Indexed:      {}", result.total_files);
    println!("Total Workspace Size:     {:.3} MB ({} bytes)", result.total_size_bytes as f64 / 1_048_576.0, result.total_size_bytes);
    println!("Scan Time:                {} ms", result.scan_duration_ms);
    println!("====================================================");

    // List some of the files for verification
    let max_display = 15;
    let files_to_show = std::cmp::min(result.files.len(), max_display);
    if files_to_show > 0 {
        println!("\nFirst {} files:", files_to_show);
        for entry in result.files.iter().take(files_to_show) {
            let type_str = if entry.is_binary { "BINARY" } else { "TEXT" };
            let ext_str = entry.extension.as_deref().unwrap_or("none");
            println!(
                "  - {} ({} bytes, type: {}, ext: {})",
                entry.path, entry.size_bytes, type_str, ext_str
            );
        }
        if result.files.len() > max_display {
            println!("  ... and {} more files.", result.files.len() - max_display);
        }
    } else {
        println!("No files found in workspace.");
    }
}

