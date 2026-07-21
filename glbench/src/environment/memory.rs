//! System memory facts, probed with std + OS files only.

/// Observed system memory.
#[derive(Debug, Clone, Default)]
pub struct MemoryInfo {
    /// Total physical RAM in bytes, if the OS exposes it.
    pub total_bytes: Option<u64>,
    /// Available RAM in bytes at probe time, if the OS exposes it.
    pub available_bytes: Option<u64>,
}

impl MemoryInfo {
    /// Probe the current machine.
    pub fn probe() -> MemoryInfo {
        let mut info = MemoryInfo::default();
        info.probe_os();
        info
    }

    #[cfg(target_os = "linux")]
    fn probe_os(&mut self) {
        if let Ok(text) = std::fs::read_to_string("/proc/meminfo") {
            for line in text.lines() {
                if let Some((key, val)) = line.split_once(':') {
                    // Values are in kB.
                    let kb: Option<u64> = val.trim().trim_end_matches(" kB").trim().parse().ok();
                    match key.trim() {
                        "MemTotal" => self.total_bytes = kb.map(|k| k * 1024),
                        "MemAvailable" => self.available_bytes = kb.map(|k| k * 1024),
                        _ => {}
                    }
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    fn probe_os(&mut self) {
        // Both wmic fields are reported in KiB; convert to bytes to match
        // this struct's unit. `wmic_field` returns None on any failure —
        // missing binary, non-zero exit, empty output — so an unreadable
        // counter stays None rather than a guess.
        self.total_bytes = super::cpu::wmic_field("OS", "TotalVisibleMemorySize")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|kb| kb * 1024);
        self.available_bytes = super::cpu::wmic_field("OS", "FreePhysicalMemory")
            .and_then(|s| s.parse::<u64>().ok())
            .map(|kb| kb * 1024);
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn probe_os(&mut self) {
        let _ = &self.total_bytes;
    }
}
