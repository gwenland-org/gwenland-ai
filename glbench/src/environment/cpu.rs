//! CPU facts, probed with the standard library only.
//!
//! There is no portable std API for CPU model/frequency, so this reports what
//! std *can* see (logical core count) and fills the rest from OS-specific files
//! where they exist (Linux `/proc/cpuinfo`). On platforms without those files
//! the extra fields are left `None` — glbench records what it can observe and
//! is honest about the rest.

use std::thread;

/// Observed CPU facts.
#[derive(Debug, Clone, Default)]
pub struct CpuInfo {
    /// Logical processor count (`std::thread::available_parallelism`).
    pub logical_cores: usize,
    /// Model name string, if the OS exposes one.
    pub model: Option<String>,
    /// Nominal/observed clock in MHz, if the OS exposes one.
    pub mhz: Option<f64>,
}

impl CpuInfo {
    /// Probe the current machine.
    pub fn probe() -> CpuInfo {
        let logical_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut info = CpuInfo { logical_cores, model: None, mhz: None };
        info.probe_os();
        info
    }

    #[cfg(target_os = "linux")]
    fn probe_os(&mut self) {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in text.lines() {
                if let Some((key, val)) = line.split_once(':') {
                    let (key, val) = (key.trim(), val.trim());
                    match key {
                        "model name" if self.model.is_none() => {
                            self.model = Some(val.to_string());
                        }
                        "cpu MHz" if self.mhz.is_none() => {
                            self.mhz = val.parse().ok();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn probe_os(&mut self) {
        // Windows/macOS expose this only via APIs outside std; the dependency
        // rule forbids pulling those crates in, so these stay None.
        let _ = &self.model;
    }
}
