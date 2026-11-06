//! Memory-usage measurement helpers.
//!
//! glbench does not link a GPU SDK, so device-memory peak comes from the engine
//! (if it reports one) rather than a driver query here. This module holds the
//! host-side facts std can observe — the model file size (the resident weight
//! footprint a fully-loaded engine holds) and this *process's* own RSS. Kept
//! separate from [`crate::environment::memory`] because that is the *snapshot*
//! of the whole machine's memory at probe time; this is the *measurement*
//! helper the runner calls to fill [`crate::core::metrics::MeasurementSet::peak_memory_bytes`]
//! — a field the schema already carried (JSON export, round-trip tests) but
//! nothing populated until this.
//!
//! # Why peak, not "average"
//!
//! The OS kernel already tracks each process's high-water-mark RSS
//! (`VmHWM` on Linux, `PeakWorkingSetSize` on Windows) — reading it after the
//! measured phase is one syscall-equivalent, exact, and needs no sampling
//! loop. A true time-averaged RSS would need a background thread polling
//! throughout the run, which is real added complexity and overhead for a
//! number peak already mostly substitutes for in practice (a benchmark that
//! loads a model once and then decodes has one RSS ramp, not an oscillating
//! series) — deliberately not built here; see `RESEARCH_REQUIREMENTS.md`'s
//! "Average memory usage" row for this reasoning recorded against the
//! backlog entry.

/// Convert a byte count to gibibytes for display.
pub fn bytes_to_gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// This process's peak resident-set size (high-water mark) in bytes, if the
/// OS exposes it. `None` — never a guess — on platforms without a known
/// mechanism, mirroring every other probe in `environment/`.
///
/// Deliberately reads the *current* process's own memory, not the engine's:
/// glbench and the engine share one address space (in-process `GlEngine`,
/// not a subprocess), so the process-wide watermark already is the
/// benchmark's memory footprint.
pub fn peak_rss_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        linux_status_field("VmHWM")
    }
    #[cfg(target_os = "windows")]
    {
        windows_peak_working_set_bytes()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Read one `Key:  N kB` line from `/proc/self/status` — the same file
/// format `environment::memory` reads from `/proc/meminfo`, just a
/// per-process file instead of a system-wide one.
#[cfg(target_os = "linux")]
fn linux_status_field(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        if let Some((k, val)) = line.split_once(':') {
            if k.trim() == key {
                let kb: u64 = val.trim().trim_end_matches(" kB").trim().parse().ok()?;
                return Some(kb * 1024);
            }
        }
    }
    None
}

#[cfg(target_os = "windows")]
fn windows_peak_working_set_bytes() -> Option<u64> {
    crate::environment::windows::peak_working_set_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_gib() {
        assert!((bytes_to_gib(1u64 << 30) - 1.0).abs() < 1e-9);
    }

    #[test]
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    fn peak_rss_is_observable_on_a_running_process() {
        // The test process itself has some nonzero RSS on Linux/Windows —
        // this is the closest thing to an integration test this pure probe
        // function gets without mocking the OS.
        let peak = peak_rss_bytes();
        assert!(peak.is_some(), "peak RSS should be observable on this platform");
        assert!(peak.unwrap() > 0, "a running process has nonzero RSS");
    }
}
