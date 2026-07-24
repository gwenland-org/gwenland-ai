//! CPU utilization measurement — how much of the measured phase this process
//! actually spent running, vs. waiting.
//!
//! [Vetted per `RESEARCH_REQUIREMENTS.md`'s 8 mandatory questions]
//! 1. What problem: `analysis::bottleneck` already classifies decode as
//!    memory-bound / compute-bound / undetermined from throughput vs. the
//!    bandwidth ceiling — but that says nothing about whether the *CPU*
//!    itself was pinned the whole time or mostly idle waiting on something
//!    else (a lock, I/O, a page fault). A "compute_bound" verdict with low
//!    CPU utilization would itself be a finding worth seeing.
//! 2. Who benefits: anyone chasing a "throughput is low but the bottleneck
//!    classification doesn't explain why" investigation.
//! 3. Production/research use: process CPU-time accounting (`getrusage`,
//!    `/proc/[pid]/stat`, `GetProcessTimes`) is standard OS-level practice,
//!    not novel.
//! 4. How calculated: (process CPU-seconds consumed during the measured
//!    phase) / (wall-clock seconds elapsed) / (logical core count) * 100 —
//!    an aggregate percentage for the whole phase, not a time series (no
//!    background sampling thread, matching the "average memory" scoping
//!    decision in `measurement::memory`).
//! 5. Reproducible: yes, pure OS counters, no randomness.
//! 6. Actionable: yes — low utilization alongside a "compute_bound" verdict
//!    directly contradicts that verdict and points at something else
//!    (scheduler contention, I/O wait) instead.
//! 7. Lightweight: two point-in-time reads (before/after the measured
//!    phase), same shape as `probe_mhz`'s existing before/after pattern.
//! 8. Philosophy: read-only, reports a fact (percentage), not a verdict —
//!    `analysis` is free to build a classification on top of it later.
//!
//! # Platform scope, stated honestly
//!
//! **Linux**: `/proc/self/stat` fields 14 (`utime`) and 15 (`stime`), in
//! clock ticks. Converting ticks to seconds needs the kernel's configured
//! tick rate (`sysconf(_SC_CLK_TCK)`), which this crate cannot call without
//! `libc` (the zero-dependency rule). **This uses the standard `USER_HZ =
//! 100` assumption** that `ps`/`top`/most `/proc`-parsing tools rely on
//! without calling `sysconf` either — true for essentially every x86_64
//! Linux distribution, but unlike every other OS-observed fact in this
//! crate, not independently confirmed against the specific machine's own
//! kernel. Flagged here rather than presented as equally certain to the
//! byte-counted memory figures.
//!
//! **Windows**: `wmic process where ProcessId=<pid> get
//! KernelModeTime,UserModeTime`, via the same
//! [`crate::environment::cpu::wmic_field_where`] helper `measurement::memory`
//! uses. **Verified empirically** (not assumed from documentation alone,
//! after `PeakWorkingSetSize`'s KB-not-bytes surprise): a controlled 3-second
//! busy-loop's wmic-reported kernel+user time, divided by `Get-Process`'s
//! independently-computed `TotalProcessorTime.TotalSeconds` for the same
//! PID, converged on 10,000,000 units/second — confirming the documented
//! 100-nanosecond unit is correct here (unlike `PeakWorkingSetSize`, which
//! documentation alone would also have gotten wrong).
//!
//! Both platforms report `None` — never a guessed percentage — when the
//! underlying read fails.

/// Process CPU time consumed so far (user + system), in seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProcessCpuTime {
    pub total_secs: f64,
}

/// Read this process's cumulative CPU time since it started.
pub fn process_cpu_time() -> Option<ProcessCpuTime> {
    #[cfg(target_os = "linux")]
    {
        linux_cpu_time()
    }
    #[cfg(target_os = "windows")]
    {
        windows_cpu_time()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// `/proc/self/stat` fields 14/15 (`utime`/`stime`), space-separated,
/// after the `)` that closes field 2 (the process name, which may itself
/// contain spaces or parens) — the standard, documented way to parse this
/// file safely.
#[cfg(target_os = "linux")]
fn linux_cpu_time() -> Option<ProcessCpuTime> {
    const USER_HZ: f64 = 100.0;
    let text = std::fs::read_to_string("/proc/self/stat").ok()?;
    let after_paren = text.rsplit_once(')')?.1;
    let fields: Vec<&str> = after_paren.split_whitespace().collect();
    // Field 2 is ")" itself (already consumed); fields[0] here is original
    // field 3 (state). utime is original field 14 -> fields[11]; stime is
    // field 15 -> fields[12].
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    Some(ProcessCpuTime { total_secs: (utime + stime) / USER_HZ })
}

#[cfg(target_os = "windows")]
fn windows_cpu_time() -> Option<ProcessCpuTime> {
    const HUNDRED_NS_PER_SEC: f64 = 10_000_000.0;
    let pid = std::process::id();
    let condition = format!("ProcessId={pid}");
    let kernel: f64 = crate::environment::cpu::wmic_field_where("process", Some(&condition), "KernelModeTime")?
        .parse()
        .ok()?;
    let user: f64 = crate::environment::cpu::wmic_field_where("process", Some(&condition), "UserModeTime")?
        .parse()
        .ok()?;
    Some(ProcessCpuTime { total_secs: (kernel + user) / HUNDRED_NS_PER_SEC })
}

/// Aggregate utilization over one phase: how much of `logical_cores`' worth
/// of continuous execution this process actually used, as a percentage.
/// `100.0` means fully pinning one core the entire time; `logical_cores *
/// 100.0` is the theoretical ceiling if every core were saturated the whole
/// phase (this function does not clamp to that ceiling — a caller seeing a
/// figure near or above it has probably measured a phase too short for the
/// sampling granularity to be meaningful, which is itself worth surfacing
/// rather than silently clamping away).
pub fn utilization_pct(cpu_secs_delta: f64, wall_secs_delta: f64, logical_cores: usize) -> Option<f64> {
    if wall_secs_delta <= 0.0 || logical_cores == 0 {
        return None;
    }
    Some(cpu_secs_delta / wall_secs_delta / logical_cores as f64 * 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn process_cpu_time_is_observable_and_nonnegative() {
        let t = process_cpu_time();
        assert!(t.is_some(), "CPU time should be observable on this platform");
        assert!(t.unwrap().total_secs >= 0.0);
    }

    #[test]
    fn utilization_pct_full_single_core_saturation() {
        // 1 CPU-second consumed over 1 wall-second on a 1-core budget = 100%.
        let pct = utilization_pct(1.0, 1.0, 1).unwrap();
        assert!((pct - 100.0).abs() < 1e-6);
    }

    #[test]
    fn utilization_pct_divides_by_core_count() {
        // 2 CPU-seconds over 1 wall-second on a 4-core budget = 50% of the
        // machine's total capacity, even though it's 200% of one core.
        let pct = utilization_pct(2.0, 1.0, 4).unwrap();
        assert!((pct - 50.0).abs() < 1e-6);
    }

    #[test]
    fn utilization_pct_zero_wall_time_is_none() {
        assert!(utilization_pct(1.0, 0.0, 4).is_none());
    }

    #[test]
    fn utilization_pct_zero_cores_is_none() {
        assert!(utilization_pct(1.0, 1.0, 0).is_none());
    }

    #[test]
    fn utilization_pct_idle_process_is_near_zero() {
        let pct = utilization_pct(0.01, 1.0, 4).unwrap();
        assert!(pct < 1.0, "got {pct}");
    }
}
