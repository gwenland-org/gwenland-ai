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

    #[cfg(not(target_os = "linux"))]
    fn probe_os(&mut self) {
        let _ = &self.total_bytes;
    }
}
