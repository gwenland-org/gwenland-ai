//! CPU facts, probed with the standard library only.
//!
//! There is no portable std API for CPU model/frequency, so this reports what
//! std *can* see (logical core count) and fills the rest from OS-specific files
//! where they exist (Linux `/proc/cpuinfo`). On platforms without those files
//! the extra fields are left `None` — glbench records what it can observe and
//! is honest about the rest.

use std::thread;

/// Which SIMD instruction sets the CPU *supports*.
///
/// Support is not use. An engine may decline an ISA it could run — glproc
/// deliberately rejects AVX-512 on low-core parts because it downclocks below
/// AVX2's effective throughput. So this records the machine's capability, and
/// the engine's actual choice arrives separately via
/// [`glcore::telemetry::BackendTelemetry::simd_path`]. Reporting only one of
/// the two would make a "why is this slow on an AVX-512 box" question
/// unanswerable.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IsaSupport {
    pub avx2: bool,
    pub fma: bool,
    pub f16c: bool,
    pub avx512f: bool,
    pub avx512bw: bool,
    /// AVX-512 VNNI (`VPDPBUSD`) — the int8 dot-product accelerator.
    pub avx512_vnni: bool,
    /// AVX-VNNI, the 256-bit VNNI on parts with no AVX-512 (Alder Lake+).
    pub avx_vnni: bool,
}

impl IsaSupport {
    /// Probe via `std::arch` feature detection — no external crate, per the
    /// crate's zero-new-dependency rule.
    pub fn probe() -> IsaSupport {
        #[cfg(target_arch = "x86_64")]
        {
            IsaSupport {
                avx2: std::arch::is_x86_feature_detected!("avx2"),
                fma: std::arch::is_x86_feature_detected!("fma"),
                f16c: std::arch::is_x86_feature_detected!("f16c"),
                avx512f: std::arch::is_x86_feature_detected!("avx512f"),
                avx512bw: std::arch::is_x86_feature_detected!("avx512bw"),
                avx512_vnni: std::arch::is_x86_feature_detected!("avx512vnni"),
                avx_vnni: std::arch::is_x86_feature_detected!("avxvnni"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            IsaSupport::default()
        }
    }

    /// The supported ISAs, highest-value first, for display.
    pub fn names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        for (on, name) in [
            (self.avx512f, "avx512f"),
            (self.avx512bw, "avx512bw"),
            (self.avx512_vnni, "avx512vnni"),
            (self.avx_vnni, "avxvnni"),
            (self.avx2, "avx2"),
            (self.fma, "fma"),
            (self.f16c, "f16c"),
        ] {
            if on {
                v.push(name);
            }
        }
        v
    }
}

/// Observed CPU facts.
#[derive(Debug, Clone, Default)]
pub struct CpuInfo {
    /// Logical processor count (`std::thread::available_parallelism`).
    pub logical_cores: usize,
    /// Physical core count, if it can be determined. Distinct from
    /// `logical_cores` on SMT parts — and the distinction matters, because the
    /// optimal thread count for a memory-bound decode loop tracks neither one
    /// reliably (measured knee on an i3-1115G4 was 3, between physical 2 and
    /// logical 4). Recording both lets that be analyzed rather than assumed.
    pub physical_cores: Option<usize>,
    /// Model name string, if the OS exposes one.
    pub model: Option<String>,
    /// Nominal/observed clock in MHz, if the OS exposes one.
    pub mhz: Option<f64>,
    /// SIMD instruction sets the CPU supports (not necessarily what runs).
    pub isa: IsaSupport,
}

impl CpuInfo {
    /// Probe the current machine.
    pub fn probe() -> CpuInfo {
        let logical_cores = thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        let mut info = CpuInfo {
            logical_cores,
            physical_cores: None,
            model: None,
            mhz: None,
            isa: IsaSupport::probe(),
        };
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
                        // Physical cores per socket. Repeats once per logical
                        // CPU, so take the first and do not sum — summing would
                        // multiply by the SMT factor.
                        "cpu cores" if self.physical_cores.is_none() => {
                            self.physical_cores = val.parse().ok();
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    #[cfg(not(target_os = "linux"))]
    fn probe_os(&mut self) {
        // Windows/macOS expose model/MHz only via APIs outside std, and the
        // crate's dependency rule forbids pulling those in — so they stay None.
        // Physical cores are left None rather than guessed: halving the logical
        // count assumes SMT, and a wrong number here would be reported as an
        // observed fact. glbench records what it can see and is honest about
        // the rest.
        let _ = &self.model;
    }
}
