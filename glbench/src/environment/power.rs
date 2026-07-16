//! Package energy measurement — Joules per token, measured, never estimated.
//!
//! Reads Intel RAPL (and AMD's compatible implementation) through the Linux
//! `powercap` sysfs interface: `/sys/class/powercap/intel-rapl:*/energy_uj`,
//! a monotonically increasing microjoule counter per package. Plain `std` file
//! reads — no crates, no drivers, honoring the zero-dependency rule.
//!
//! # What this is NOT
//!
//! - **Not a TDP estimate.** TDP-based "power" figures run 40%+ off from
//!   measured draw; if the counter is not readable, this module reports
//!   `None` — *not measured* — rather than substituting a spec number.
//! - **Not available on Windows or macOS.** RAPL on Windows requires a signed
//!   kernel driver for MSR access, which a zero-dependency user-space tool
//!   cannot honestly provide. On those platforms `EnergyMeter::start` returns
//!   `None` and the report simply carries no energy figure. (The v2 decision
//!   record sketched a UAC elevation prompt; elevation alone does not grant
//!   MSR access from user space, so it would be theater — omitted on purpose.)
//! - **Whole-package, not process-scoped.** RAPL counts everything on the
//!   package, including other processes. On an otherwise idle machine that is
//!   the inference cost; on a busy one it is an upper bound. The reader is
//!   told the number is package-level; scoping it further would be a claim
//!   the counter cannot back.
//!
//! The counter wraps at `max_energy_range_uj`; a single benchmark run is far
//! shorter than a wrap period, so one wrap correction is sufficient.

use std::fs;

/// Where powercap exposes RAPL domains. Each `intel-rapl:N` directory is one
/// package; sub-domains (`:N:M`, core/uncore/dram) are ignored — the package
/// counter already includes them.
const POWERCAP_DIR: &str = "/sys/class/powercap";

/// A running energy measurement across all readable RAPL packages.
#[derive(Debug)]
pub struct EnergyMeter {
    /// One (energy_uj at start, wrap ceiling) per package counter.
    start_uj: Vec<(u64, Option<u64>)>,
    /// The counter file paths, in the same order.
    paths: Vec<String>,
}

impl EnergyMeter {
    /// Begin measuring. `None` when no RAPL package counter is readable —
    /// non-Linux platforms, missing permissions (the files are often
    /// root-readable only), or hardware without RAPL.
    pub fn start() -> Option<EnergyMeter> {
        let entries = fs::read_dir(POWERCAP_DIR).ok()?;
        let mut paths = Vec::new();
        for e in entries.flatten() {
            let name = e.file_name().to_string_lossy().into_owned();
            // Package domains only: "intel-rapl:0", not "intel-rapl:0:1".
            if name.starts_with("intel-rapl:") && name.matches(':').count() == 1 {
                paths.push(format!("{POWERCAP_DIR}/{name}"));
            }
        }
        paths.sort();

        let mut start_uj = Vec::new();
        let mut readable = Vec::new();
        for p in paths {
            if let Some(uj) = read_u64(&format!("{p}/energy_uj")) {
                let ceiling = read_u64(&format!("{p}/max_energy_range_uj"));
                start_uj.push((uj, ceiling));
                readable.push(p);
            }
        }
        if readable.is_empty() {
            return None;
        }
        Some(EnergyMeter { start_uj, paths: readable })
    }

    /// Stop measuring and return the Joules consumed since `start`, summed
    /// over packages. `None` if any counter became unreadable mid-run —
    /// a partial sum would silently under-report.
    pub fn stop(self) -> Option<f64> {
        let mut total_uj: u64 = 0;
        for (path, (start, ceiling)) in self.paths.iter().zip(self.start_uj) {
            let now = read_u64(&format!("{path}/energy_uj"))?;
            let delta = if now >= start {
                now - start
            } else {
                // The counter wrapped once: distance to the ceiling, plus the
                // new value. Without a known ceiling the delta is unknowable.
                ceiling?.checked_sub(start)? + now
            };
            total_uj = total_uj.checked_add(delta)?;
        }
        Some(total_uj as f64 / 1e6)
    }
}

/// Read a file containing one decimal integer, tolerating trailing whitespace.
fn read_u64(path: &str) -> Option<u64> {
    fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_powercap_yields_none_not_zero() {
        // On Windows/macOS (and most CI) there is no /sys/class/powercap:
        // the meter must decline to exist rather than report 0 J.
        if !std::path::Path::new(POWERCAP_DIR).exists() {
            assert!(EnergyMeter::start().is_none());
        }
    }

    #[test]
    fn read_u64_parses_counter_format() {
        // The sysfs format: digits plus a trailing newline.
        let dir = std::env::temp_dir().join("glbench_rapl_test");
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("energy_uj");
        std::fs::write(&f, "123456789\n").unwrap();
        assert_eq!(read_u64(f.to_str().unwrap()), Some(123456789));
        std::fs::write(&f, "not a number").unwrap();
        assert_eq!(read_u64(f.to_str().unwrap()), None);
    }
}
