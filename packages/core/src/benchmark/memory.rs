/// Process memory baseline measurement.
///
/// Reuses the same platform-detection logic as `eval/metrics.rs`
/// (`sample_peak_memory_mb`). It is re-implemented here rather than calling
/// across module boundaries because:
///   1. The eval module is not `pub`-exported from lib.rs (it uses `pub mod`
///      but the internal `sample_peak_memory_mb` function is `fn`, not `pub fn`).
///   2. Duplicating a 20-line function is less coupling than making internal
///      eval helpers public just to share them with benchmark.
///   3. If the memory measurement strategy diverges (e.g. benchmark wants
///      VmPeak instead of VmRSS) the two can evolve independently.
///
/// Platform strategy is identical to eval/metrics.rs — see that file for the
/// full rationale. Brief summary:
///   Linux  → /proc/self/status VmRSS (current RSS, updated per allocation)
///   Other  → sysinfo process memory() (WorkingSetSize on Windows, phys_footprint on macOS)
///   Error  → 0.0 (non-fatal; benchmark report shows "0.0 MB" which signals the issue)
use super::MemoryResult;

/// Sample the current process's resident memory and return a `MemoryResult`.
///
/// Called once at benchmark start so the report can show how much RAM
/// GwenLand uses at idle — a key claim in the project's "< 50 MB baseline"
/// positioning. Measuring at benchmark start (before any heavy work) gives
/// the cleanest idle-footprint reading.
pub fn sample_memory() -> MemoryResult {
    MemoryResult {
        baseline_mb: read_rss_mb(),
    }
}

// ── Platform-split implementation ─────────────────────────────────────────────

fn read_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        // /proc/self/status is guaranteed to exist on Linux ≥ 2.6.
        // VmRSS (resident set size) is the most accurate measure of actual
        // physical memory in use. VmPeak (peak virtual memory) includes
        // memory-mapped shared libraries which inflates the number dramatically.
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    // Format: "VmRSS:   12345 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return kb / 1024.0;
                        }
                    }
                }
            }
        }
        0.0
    }

    #[cfg(not(target_os = "linux"))]
    {
        // sysinfo is already a dep of gwen-core (used in platform/hardware.rs).
        // `refresh_processes()` in sysinfo 0.30 takes no arguments — there is
        // no selective-PID refresh API in this version. The full refresh is
        // acceptable here because memory sampling happens only once per
        // benchmark run, not in a hot loop.
        use sysinfo::{Pid, System};
        let mut sys = System::new();
        sys.refresh_processes();
        let pid = Pid::from(std::process::id() as usize);
        if let Some(proc) = sys.process(pid) {
            // memory() returns bytes on all sysinfo-supported platforms.
            return proc.memory() as f64 / (1024.0 * 1024.0);
        }
        0.0
    }
}
