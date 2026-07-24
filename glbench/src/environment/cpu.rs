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
    /// Sustained sequential read bandwidth, GB/s — **measured on this machine**,
    /// not looked up from a vendor table.
    ///
    /// This is the ceiling every other bandwidth figure is judged against.
    /// Without it, a stage reporting "23 GB/s" is uninterpretable: it could be
    /// 78% of the machine (nothing left to win) or 30% (something is wrong), and
    /// those call for opposite decisions. See [`super::bandwidth`].
    pub read_bandwidth_gbs: Option<f64>,
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
            // Measured, not assumed. Costs ~1s of streaming reads at startup,
            // paid once per session — cheap against the alternative of every
            // efficiency number in the report being uninterpretable.
            read_bandwidth_gbs: super::bandwidth::measure_read_gbs(),
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

    #[cfg(target_os = "windows")]
    fn probe_os(&mut self) {
        // `wmic` is deprecated (and absent on some stripped-down Windows
        // installs), but it is still present on the reference machine
        // (Windows 11) and every other Windows box this crate has been run
        // on. `wmic_field` returns None on any failure — missing binary,
        // non-zero exit, empty output — so a machine without it degrades to
        // exactly the "not observed" state this module already uses for
        // macOS, rather than a hard error.
        self.model = wmic_field("cpu", "Name");
        self.physical_cores = wmic_field("cpu", "NumberOfCores").and_then(|s| s.parse().ok());
        self.mhz = wmic_field("cpu", "CurrentClockSpeed").and_then(|s| s.parse().ok());
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    fn probe_os(&mut self) {
        // macOS and anything else expose model/MHz only via APIs outside std
        // (or, on macOS, a `sysctl` binary this crate has not been asked to
        // shell out to), and the crate's dependency rule forbids pulling
        // libraries in — so they stay None. Physical cores are left None
        // rather than guessed: halving the logical count assumes SMT, and a
        // wrong number here would be reported as an observed fact. glbench
        // records what it can see and is honest about the rest.
        let _ = &self.model;
    }
}

/// The current CPU clock, MHz, without a full [`CpuInfo::probe`] (which pays
/// for `IsaSupport::probe` and the ~1s bandwidth measurement neither the
/// start nor end thermal reading needs). Cheap enough to call twice in one
/// session — once before the measured iterations, once after — to detect a
/// clock drop between them. `None` wherever [`CpuInfo::mhz`] would be `None`.
pub fn probe_mhz() -> Option<f64> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
        for line in text.lines() {
            if let Some((key, val)) = line.split_once(':') {
                if key.trim() == "cpu MHz" {
                    return val.trim().parse().ok();
                }
            }
        }
        None
    }
    #[cfg(target_os = "windows")]
    {
        wmic_field("cpu", "CurrentClockSpeed").and_then(|s| s.parse().ok())
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// Run `wmic <class> get <field> /format:list` and pull `field`'s value out
/// of the `KEY=VALUE` output. `None` on any failure — `wmic` missing, a
/// non-zero exit, or a field absent from the output — never a guessed value.
///
/// `wmic`'s `/format:list` output is plain ASCII when captured through
/// `std::process::Command` (verified against the real binary: no UTF-16
/// decoding needed, unlike what a console might display), with stray blank
/// lines and doubled `\r` around each `KEY=VALUE` line.
///
/// `pub(crate)` and shared: [`super::memory`] queries the `OS` wmic class
/// through the same helper rather than duplicating the process-spawning and
/// parsing logic for one more class.
#[cfg(target_os = "windows")]
pub(crate) fn wmic_field(class: &str, field: &str) -> Option<String> {
    wmic_field_where(class, None, field)
}

/// As [`wmic_field`], with an optional `where <condition>` clause inserted
/// before `get` — e.g. `wmic process where ProcessId=1234 get
/// PeakWorkingSetSize /format:list`, for a single-process query instead of a
/// system-wide one. [`super::super::measurement::memory`] uses this form;
/// [`wmic_field`] stays the zero-argument case every other caller wants,
/// rather than every call site building its own `Vec<&str>` of args.
#[cfg(target_os = "windows")]
pub(crate) fn wmic_field_where(class: &str, condition: Option<&str>, field: &str) -> Option<String> {
    let mut args: Vec<&str> = vec![class];
    if let Some(cond) = condition {
        args.push("where");
        args.push(cond);
    }
    args.push("get");
    args.push(field);
    args.push("/format:list");

    let output = std::process::Command::new("wmic").args(&args).output().ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    let prefix = format!("{field}=");
    for line in text.lines() {
        let line = line.trim();
        if let Some(value) = line.strip_prefix(&prefix) {
            let value = value.trim();
            if !value.is_empty() {
                return Some(value.to_string());
            }
        }
    }
    None
}
