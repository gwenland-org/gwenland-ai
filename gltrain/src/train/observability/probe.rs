//! Stummañ Deskiñ: the platform probes for axes 5, 6 and 7.
//!
//! This is the only file in the observability sub-system that touches the
//! operating system, and it is the only one that can return nothing useful.
//!
//! # Nothing here returns a `Result`
//!
//! Every probe returns `Option`, and every failure path produces `None`. That
//! is deliberate and it is not laziness about error handling: an unreadable
//! `/sys/class/thermal/thermal_zone0/temp` is not an error in a training run,
//! it is a machine without that sensor. Propagating it as `Err` would force
//! every call site to decide between crashing a training run over a missing
//! thermometer and writing the same `unwrap_or_default()` everywhere.
//!
//! What matters is that `None` is never rendered as a healthy number. The
//! detectors in [`super::anomaly`] fire on nothing when a reading is absent,
//! and the live line prints `n/a`.
//!
//! # Platform coverage, stated plainly
//!
//! | Axis | Linux | Windows | other |
//! |---|---|---|---|
//! | RSS, peak RSS | `/proc/self/status` | `K32GetProcessMemoryInfo` | none |
//! | available RAM | `/proc/meminfo` | `GlobalMemoryStatusEx` | none |
//! | CPU temperature | `thermal_zone*` | none | none |
//! | CPU frequency | `cpufreq` | none | none |
//! | CPU usage | `/proc/self/stat` | process times | none |
//! | threads, ctx switches, faults | `/proc/self/status` | none | none |
//!
//! Windows exposes no per-process temperature or frequency without WMI or a
//! driver, so those stay `None` there rather than being faked from a timer.

use super::axes::{VLHardwareSnapshot, VLMemorySnapshot, VLSystemSnapshot};

/// Rolling state the probes need to turn absolute counters into per-step deltas.
///
/// Counters like context switches and page faults are monotonic since process
/// start. What a step wants is what happened *during* it, so the previous
/// reading has to be kept.
#[derive(Debug, Clone, Default)]
pub struct VLProbeState {
    /// RSS at the previous step.
    prev_rss: Option<u64>,
    /// Largest RSS seen.
    peak_rss: Option<u64>,
    /// Frequency observed on the first probed step.
    freq_baseline_mhz: Option<f32>,
    /// Previous voluntary context switch count.
    prev_ctx_switches: Option<u64>,
    /// Previous major fault count.
    prev_major_faults: Option<u64>,
    /// Previous minor fault count.
    prev_minor_faults: Option<u64>,
    /// Previous process CPU time, in seconds.
    prev_cpu_seconds: Option<f64>,
}

impl VLProbeState {
    /// Fresh state, before any step has been probed.
    pub fn new() -> Self {
        Self::default()
    }

    /// The frequency baseline, once one has been established.
    pub fn freq_baseline_mhz(&self) -> Option<f32> {
        self.freq_baseline_mhz
    }

    /// Read memory, folding this step's reading into the rolling peak.
    ///
    /// `wall_seconds` is unused here but taken so all three probes share a
    /// shape; see [`VLProbeState::hardware`].
    pub fn memory(&mut self) -> VLMemorySnapshot {
        let rss = read_rss_bytes();
        let delta = match (rss, self.prev_rss) {
            (Some(now), Some(prev)) => Some(now as i64 - prev as i64),
            // The first step has nothing to subtract. Reporting 0 would look
            // like "allocated nothing", which is the opposite of the truth.
            _ => None,
        };
        if let Some(now) = rss {
            self.peak_rss = Some(self.peak_rss.map_or(now, |p| p.max(now)));
            self.prev_rss = Some(now);
        }
        VLMemorySnapshot {
            heap_rss_bytes: rss,
            heap_delta_bytes: delta,
            peak_rss_bytes: self.peak_rss,
            available_ram_bytes: read_available_ram_bytes(),
            alloc_count_delta: None,
        }
    }

    /// Read hardware. `wall_seconds` is the wall time this step took, used to
    /// turn a CPU-time delta into a utilisation percentage.
    pub fn hardware(&mut self, wall_seconds: f64) -> VLHardwareSnapshot {
        let freq = read_cpu_freq_mhz();
        if self.freq_baseline_mhz.is_none() {
            self.freq_baseline_mhz = freq;
        }
        // Only claim a throttle when both numbers exist. A machine that cannot
        // report its frequency is not a throttling machine.
        let throttle = match (freq, self.freq_baseline_mhz) {
            (Some(now), Some(base)) if base > 0.0 => now < base * super::anomaly::THROTTLE_FRACTION,
            _ => false,
        };

        let cpu_usage_pct = read_process_cpu_seconds().and_then(|now| {
            let pct = self.prev_cpu_seconds.and_then(|prev| {
                if wall_seconds > 0.0 {
                    Some((100.0 * (now - prev) / wall_seconds) as f32)
                } else {
                    None
                }
            });
            self.prev_cpu_seconds = Some(now);
            pct
        });

        VLHardwareSnapshot {
            cpu_temp_celsius: read_cpu_temp_celsius(),
            cpu_freq_mhz: freq,
            cpu_freq_baseline_mhz: self.freq_baseline_mhz,
            throttle_detected: throttle,
            cpu_usage_pct,
        }
    }

    /// Read system counters, as per-step deltas.
    pub fn system(&mut self) -> VLSystemSnapshot {
        let raw = read_system_counters();

        let delta = |now: Option<u64>, prev: &mut Option<u64>| -> Option<u64> {
            let now = now?;
            let out = prev.map(|p| now.saturating_sub(p));
            *prev = Some(now);
            out
        };

        VLSystemSnapshot {
            thread_count: raw.threads,
            voluntary_ctx_switches: delta(raw.ctx_switches, &mut self.prev_ctx_switches),
            page_faults_major: delta(raw.major_faults, &mut self.prev_major_faults),
            page_faults_minor: delta(raw.minor_faults, &mut self.prev_minor_faults),
        }
    }
}

/// Raw, absolute system counters as the OS reports them.
#[derive(Debug, Clone, Copy, Default)]
struct RawCounters {
    threads: Option<u32>,
    ctx_switches: Option<u64>,
    major_faults: Option<u64>,
    minor_faults: Option<u64>,
}

// ── Linux ────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn proc_status_field(key: &str) -> Option<u64> {
    let text = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in text.lines() {
        // `continue`, not `?`: a `?` here would return `None` for the whole
        // function on the first line that is not the one being looked for,
        // so only a key that happened to be on line 1 could ever be found.
        let Some(rest) = line.strip_prefix(key).and_then(|r| r.strip_prefix(':')) else {
            continue;
        };
        return rest.split_whitespace().next()?.parse().ok();
    }
    None
}

/// Read the whitespace-separated fields of `/proc/self/stat` that follow the
/// `comm` field.
///
/// Split after the last `)` rather than tokenizing the line: `comm` is the
/// executable name in parentheses and may itself contain spaces and
/// parentheses, which shifts every field index if it is not skipped.
#[cfg(target_os = "linux")]
fn proc_stat_fields() -> Option<Vec<String>> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    Some(
        stat.rsplit_once(')')?
            .1
            .split_whitespace()
            .map(str::to_string)
            .collect(),
    )
}

#[cfg(target_os = "linux")]
fn read_rss_bytes() -> Option<u64> {
    // VmRSS is in kibibytes.
    proc_status_field("VmRSS").map(|kb| kb * 1024)
}

#[cfg(target_os = "linux")]
fn read_available_ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            return rest
                .split_whitespace()
                .next()?
                .parse::<u64>()
                .ok()
                .map(|kb| kb * 1024);
        }
    }
    None
}

#[cfg(target_os = "linux")]
fn read_cpu_temp_celsius() -> Option<f32> {
    // thermal_zone0 is not always the CPU package, but it is the only zone
    // present on every machine that has any. Millidegrees.
    let raw = std::fs::read_to_string("/sys/class/thermal/thermal_zone0/temp").ok()?;
    raw.trim().parse::<f32>().ok().map(|milli| milli / 1000.0)
}

#[cfg(target_os = "linux")]
fn read_cpu_freq_mhz() -> Option<f32> {
    let raw =
        std::fs::read_to_string("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq").ok()?;
    raw.trim().parse::<f32>().ok().map(|khz| khz / 1000.0)
}

#[cfg(target_os = "linux")]
fn read_process_cpu_seconds() -> Option<f64> {
    // After `comm`, index 11 is utime and 12 is stime, in clock ticks.
    let fields = proc_stat_fields()?;
    let utime: f64 = fields.get(11)?.parse().ok()?;
    let stime: f64 = fields.get(12)?.parse().ok()?;
    // USER_HZ is 100 on every Linux this crate would run on.
    Some((utime + stime) / 100.0)
}

#[cfg(target_os = "linux")]
fn read_system_counters() -> RawCounters {
    // Faults come from `stat`, not `status`: `/proc/self/status` has no
    // MajFlt or MinFlt line at all, so reading them from there would return
    // `None` on every Linux machine and quietly disable MAJOR_PAGE_FAULT —
    // the one system anomaly rated critical.
    let stat = proc_stat_fields();
    let field = |i: usize| -> Option<u64> { stat.as_ref()?.get(i)?.parse().ok() };
    RawCounters {
        threads: proc_status_field("Threads").map(|t| t as u32),
        ctx_switches: proc_status_field("voluntary_ctxt_switches"),
        // After `comm`, index 9 is majflt and index 7 is minflt.
        major_faults: field(9),
        minor_faults: field(7),
    }
}

// ── Windows ──────────────────────────────────────────────────────────────

/// The prefix of `PROCESS_MEMORY_COUNTERS` this crate reads.
///
/// Declared here rather than pulled from the `windows` crate: gltrain's whole
/// dependency list is glcore, glproc, anyhow and thiserror, and adding a
/// multi-megabyte binding crate to read one integer is a poor trade. The
/// layout is fixed ABI and has not changed since Windows 2000.
#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Default)]
struct ProcessMemoryCounters {
    cb: u32,
    page_fault_count: u32,
    peak_working_set_size: usize,
    working_set_size: usize,
    quota_peak_paged_pool_usage: usize,
    quota_paged_pool_usage: usize,
    quota_peak_non_paged_pool_usage: usize,
    quota_non_paged_pool_usage: usize,
    pagefile_usage: usize,
    peak_pagefile_usage: usize,
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct MemoryStatusEx {
    length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_page_file: u64,
    avail_page_file: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_extended_virtual: u64,
}

#[cfg(target_os = "windows")]
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct FileTime {
    low: u32,
    high: u32,
}

#[cfg(target_os = "windows")]
extern "system" {
    fn GetCurrentProcess() -> isize;
    fn K32GetProcessMemoryInfo(
        process: isize,
        counters: *mut ProcessMemoryCounters,
        cb: u32,
    ) -> i32;
    fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    fn GetProcessTimes(
        process: isize,
        creation: *mut FileTime,
        exit: *mut FileTime,
        kernel: *mut FileTime,
        user: *mut FileTime,
    ) -> i32;
}

/// SAFETY for every `unsafe` block below: each call passes a pointer to a
/// correctly-sized, correctly-aligned local of the exact `#[repr(C)]` type the
/// API expects, with the `cb`/`length` field set to that type's size as the
/// API requires. `GetCurrentProcess` returns a pseudo-handle that needs no
/// closing and is always valid. Every call's return code is checked, and a
/// failure yields `None` rather than reading the uninitialised buffer.
#[cfg(target_os = "windows")]
fn read_rss_bytes() -> Option<u64> {
    let mut counters = ProcessMemoryCounters {
        cb: std::mem::size_of::<ProcessMemoryCounters>() as u32,
        ..Default::default()
    };
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            &mut counters,
            std::mem::size_of::<ProcessMemoryCounters>() as u32,
        )
    };
    (ok != 0).then_some(counters.working_set_size as u64)
}

#[cfg(target_os = "windows")]
fn read_available_ram_bytes() -> Option<u64> {
    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    (ok != 0).then_some(status.avail_phys)
}

/// Windows exposes no per-process CPU temperature without WMI or a kernel
/// driver. Reporting `None` is the honest answer; deriving a number from
/// anything else here would be inventing one.
#[cfg(target_os = "windows")]
fn read_cpu_temp_celsius() -> Option<f32> {
    None
}

/// Likewise for current core frequency: `CallNtPowerInformation` reports a
/// nominal value that does not track boost or throttle, so it would make
/// `THERMAL_THROTTLE` permanently silent while looking like a real reading.
#[cfg(target_os = "windows")]
fn read_cpu_freq_mhz() -> Option<f32> {
    None
}

#[cfg(target_os = "windows")]
fn read_process_cpu_seconds() -> Option<f64> {
    let (mut creation, mut exit, mut kernel, mut user) = (
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
        FileTime::default(),
    );
    let ok = unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut creation,
            &mut exit,
            &mut kernel,
            &mut user,
        )
    };
    if ok == 0 {
        return None;
    }
    // FILETIME counts 100-nanosecond intervals.
    let to_seconds = |t: FileTime| ((t.high as u64) << 32 | t.low as u64) as f64 * 1e-7;
    Some(to_seconds(kernel) + to_seconds(user))
}

/// Thread and fault counters need `NtQuerySystemInformation` or a toolhelp
/// snapshot walk, neither of which is worth an FFI surface for a diagnostic.
#[cfg(target_os = "windows")]
fn read_system_counters() -> RawCounters {
    RawCounters::default()
}

// ── Everything else ──────────────────────────────────────────────────────

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_rss_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_available_ram_bytes() -> Option<u64> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_cpu_temp_celsius() -> Option<f32> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_cpu_freq_mhz() -> Option<f32> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_process_cpu_seconds() -> Option<f64> {
    None
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
fn read_system_counters() -> RawCounters {
    RawCounters::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The contract every probe shares: it answers or it says nothing. On a
    /// machine with no sensors at all this test still passes, which is the
    /// point — it asserts the absence of a panic, not the presence of a value.
    #[test]
    fn probing_never_panics_on_any_platform() {
        let mut state = VLProbeState::new();
        for _ in 0..3 {
            let _ = state.memory();
            let _ = state.hardware(0.01);
            let _ = state.system();
        }
    }

    /// The first step has no previous reading, so a delta would be a fiction.
    #[test]
    fn the_first_memory_probe_reports_no_delta() {
        let mut state = VLProbeState::new();
        assert_eq!(state.memory().heap_delta_bytes, None);
    }

    /// Peak is monotonic by construction, and a second probe can supply a
    /// delta because a first reading now exists.
    #[test]
    fn the_peak_never_decreases_and_a_delta_appears_on_the_second_probe() {
        let mut state = VLProbeState::new();
        let first = state.memory();
        let second = state.memory();

        if let (Some(p1), Some(p2)) = (first.peak_rss_bytes, second.peak_rss_bytes) {
            assert!(p2 >= p1, "peak went backwards: {p1} then {p2}");
            assert!(
                second.heap_delta_bytes.is_some(),
                "a delta is available once there is a previous reading"
            );
        }
    }

    /// No baseline exists before the first hardware probe, and once one is
    /// taken it must not drift — it is what throttling is measured against.
    #[test]
    fn the_frequency_baseline_is_taken_once_and_then_held() {
        let mut state = VLProbeState::new();
        assert_eq!(state.freq_baseline_mhz(), None);
        state.hardware(0.01);
        let baseline = state.freq_baseline_mhz();
        for _ in 0..3 {
            state.hardware(0.01);
            assert_eq!(state.freq_baseline_mhz(), baseline, "baseline drifted");
        }
    }

    /// A machine that cannot report its frequency must never be accused of
    /// throttling. On a platform that can, the first probe compares the
    /// baseline against itself and so cannot fire either.
    #[test]
    fn the_first_hardware_probe_never_claims_a_throttle() {
        let mut state = VLProbeState::new();
        assert!(!state.hardware(0.01).throttle_detected);
    }

    /// Counters are deltas, so the first probe has nothing to subtract from.
    #[test]
    fn the_first_system_probe_reports_no_counter_deltas() {
        let mut state = VLProbeState::new();
        let first = state.system();
        assert_eq!(first.voluntary_ctx_switches, None);
        assert_eq!(first.page_faults_major, None);
        assert_eq!(first.page_faults_minor, None);
    }

    /// Where RSS is readable at all it must be a plausible figure. A process
    /// running a test binary occupies more than a page and less than a
    /// terabyte; anything outside that means the units are wrong.
    #[test]
    fn a_readable_rss_is_in_a_plausible_range() {
        if let Some(rss) = read_rss_bytes() {
            assert!(rss > 4096, "RSS {rss} is implausibly small — wrong units?");
            assert!(
                rss < 1 << 40,
                "RSS {rss} is implausibly large — wrong units?"
            );
        }
    }
}
