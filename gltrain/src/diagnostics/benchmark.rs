// @INFO: GwenLand Benchmark Harness
// This module measures workspace scanning performance over multiple runs,
// comparing the average duration against a specified performance budget.

// @EDITABLE
use std::path::Path;
use std::time::Instant;
use colored::Colorize;
use crate::platform::scanner::scan_workspace;
use crate::storage::ignore_rules::load_ignore_rules;

/// The target performance budget for workspace scanning in milliseconds.
pub const BUDGET_MS: u64 = 500;

/// Report containing performance metrics gathered from the benchmark runs.
#[derive(Debug, Clone)]
pub struct BenchmarkReport {
    pub runs: u32,
    pub min_ms: u64,
    pub max_ms: u64,
    pub avg_ms: u64,
    pub total_files: usize,
    pub passed: bool,  // true if avg_ms < BUDGET_MS
}

/// Runs the workspace scanner multiple times to gauge average performance.
/// Prints a formatted summary using the `colored` crate.
pub fn run_benchmark(root: &Path, runs: u32) -> BenchmarkReport {
    let ignore = load_ignore_rules(root);
    let mut durations = Vec::new();
    let mut total_files = 0;

    println!("{}", format!("Starting benchmark: {} runs on {}", runs, root.display()).bold().cyan());

    for i in 1..=runs {
        let start = Instant::now();
        let result = scan_workspace(root, Some(&ignore));
        let duration = start.elapsed().as_millis() as u64;

        durations.push(duration);
        total_files = result.total_files;

        println!("  Run #{:2}: {} ms (found {} files)", i, duration, total_files);
    }

    let min_ms = *durations.iter().min().unwrap_or(&0);
    let max_ms = *durations.iter().max().unwrap_or(&0);
    let sum_ms: u64 = durations.iter().sum();
    let avg_ms = if runs > 0 { sum_ms / (runs as u64) } else { 0 };
    let passed = avg_ms < BUDGET_MS;

    println!("\n{}", "=================== BENCHMARK REPORT ===================".bold().yellow());
    println!("Total Runs:       {}", runs);
    println!("Total Files:      {}", total_files);
    println!("Min Duration:     {} ms", min_ms);
    println!("Max Duration:     {} ms", max_ms);

    let avg_str = format!("{} ms", avg_ms);
    let budget_str = format!("{} ms", BUDGET_MS);

    if passed {
        println!("Average Duration: {} (Budget: {})", avg_str.green().bold(), budget_str.dimmed());
        println!("{}", "STATUS: PASSED".green().bold());
    } else {
        println!("Average Duration: {} (Budget: {})", avg_str.red().bold(), budget_str.bold());
        println!("{}", "STATUS: FAILED (Over budget)".red().bold());
    }
    println!("{}", "========================================================".bold().yellow());

    BenchmarkReport {
        runs,
        min_ms,
        max_ms,
        avg_ms,
        total_files,
        passed,
    }
}
