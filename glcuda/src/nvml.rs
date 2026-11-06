//! NVML: what the card is doing while it works.
//!
//! The CUDA driver API says what a device is; NVML says what state it is in:
//! clocks, temperature, power, why a clock is being held down, the energy
//! counter. Loaded at runtime like the driver itself, from the library every
//! NVIDIA driver installs for `nvidia-smi`, so it adds nothing to the build.
//!
//! Every query is optional. Drivers and boards differ in what they expose, and
//! an unsupported counter reads as absent, never as zero.

use std::ffi::{c_char, c_void, CStr};

use crate::ffi::{load_library, sym_opt};

const NVML_SUCCESS: i32 = 0;
const CLOCK_SM: u32 = 1;
const CLOCK_MEM: u32 = 2;
const TEMPERATURE_GPU: u32 = 0;
const TEXT_LEN: usize = 96;

/// Clock-limit reason bits, `nvmlClocksEventReasons`.
const REASONS: [(u64, &str); 9] = [
    (0x1, "gpu_idle"),
    (0x2, "applications_clocks"),
    (0x4, "sw_power_cap"),
    (0x8, "hw_slowdown"),
    (0x10, "sync_boost"),
    (0x20, "sw_thermal_slowdown"),
    (0x40, "hw_thermal_slowdown"),
    (0x80, "hw_power_brake_slowdown"),
    (0x100, "display_clock"),
];

/// The names of the reasons set in `bits`.
pub fn reason_names(bits: u64) -> Vec<&'static str> {
    REASONS.iter().filter(|(bit, _)| bits & bit != 0).map(|&(_, name)| name).collect()
}

type Device = *mut c_void;
type GetU32 = unsafe extern "C" fn(Device, *mut u32) -> i32;
type GetClock = unsafe extern "C" fn(Device, u32, *mut u32) -> i32;
type GetU64 = unsafe extern "C" fn(Device, *mut u64) -> i32;
type GetText = unsafe extern "C" fn(Device, *mut c_char, u32) -> i32;

#[repr(C)]
#[derive(Default)]
struct Utilization {
    gpu: u32,
    memory: u32,
}

#[repr(C)]
#[derive(Default)]
struct Memory {
    total: u64,
    free: u64,
    used: u64,
}

/// Facts that hold for the life of the process.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NvmlStatic {
    pub driver_version: Option<String>,
    pub vbios_version: Option<String>,
    pub power_limit_mw: Option<u32>,
    pub max_sm_clock_mhz: Option<u32>,
    pub max_mem_clock_mhz: Option<u32>,
    pub pcie_gen: Option<u32>,
    pub pcie_width: Option<u32>,
    pub pcie_max_gen: Option<u32>,
    pub pcie_max_width: Option<u32>,
}

/// The card's state at one instant.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct NvmlSnapshot {
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    pub temperature_c: Option<u32>,
    pub power_mw: Option<u32>,
    pub gpu_util_pct: Option<u32>,
    pub mem_util_pct: Option<u32>,
    pub vram_used_bytes: Option<u64>,
    /// `P0` is 0, the fastest state.
    pub pstate: Option<u32>,
    /// Why clocks are where they are; see [`reason_names`].
    pub clock_reasons: Option<u64>,
}

/// An open NVML session bound to one device.
pub struct Nvml {
    device: Device,
    shutdown: Option<unsafe extern "C" fn() -> i32>,
    driver_version: Option<unsafe extern "C" fn(*mut c_char, u32) -> i32>,
    vbios_version: Option<GetText>,
    power_limit: Option<GetU32>,
    clock: Option<GetClock>,
    max_clock: Option<GetClock>,
    temperature: Option<GetClock>,
    power: Option<GetU32>,
    energy: Option<GetU64>,
    pstate: Option<GetU32>,
    utilization: Option<unsafe extern "C" fn(Device, *mut Utilization) -> i32>,
    memory: Option<unsafe extern "C" fn(Device, *mut Memory) -> i32>,
    reasons: Option<GetU64>,
    pcie_gen: Option<GetU32>,
    pcie_width: Option<GetU32>,
    pcie_max_gen: Option<GetU32>,
    pcie_max_width: Option<GetU32>,
}

// SAFETY: NVML is documented thread-safe, and the device handle is an opaque
// token that is only ever passed back to it.
unsafe impl Send for Nvml {}
unsafe impl Sync for Nvml {}

impl Nvml {
    /// Open NVML for the device at `pci_bus_id`, the identity CUDA and NVML
    /// share: their device orderings can differ on a multi-GPU machine. Falls
    /// back to device 0 when the id is unknown. `None` when the library or the
    /// device is unavailable.
    pub fn open(pci_bus_id: Option<&str>) -> Option<Nvml> {
        let lib = load_library(&[c"libnvidia-ml.so.1", c"libnvidia-ml.so", c"nvml.dll"])?;

        let init: unsafe extern "C" fn() -> i32 =
            sym_opt(lib, &[b"nvmlInit_v2\0", b"nvmlInit\0"])?;
        // SAFETY: takes no arguments; repeated calls are reference counted.
        if unsafe { init() } != NVML_SUCCESS {
            return None;
        }

        let mut device: Device = std::ptr::null_mut();
        let by_pci: Option<unsafe extern "C" fn(*const c_char, *mut Device) -> i32> =
            sym_opt(lib, &[b"nvmlDeviceGetHandleByPciBusId_v2\0"]);
        let by_index: Option<unsafe extern "C" fn(u32, *mut Device) -> i32> =
            sym_opt(lib, &[b"nvmlDeviceGetHandleByIndex_v2\0"]);
        let id = pci_bus_id.and_then(|id| std::ffi::CString::new(id).ok());
        // SAFETY: the id is NUL-terminated and the out pointer is a live local.
        let found = match (by_pci, id.as_ref()) {
            (Some(get), Some(id)) => (unsafe { get(id.as_ptr(), &mut device) }) == NVML_SUCCESS,
            _ => false,
        };
        // SAFETY: the out pointer is a live local.
        let found = found
            || by_index.is_some_and(|get| (unsafe { get(0, &mut device) }) == NVML_SUCCESS);
        if !found {
            return None;
        }

        Some(Nvml {
            device,
            shutdown: sym_opt(lib, &[b"nvmlShutdown\0"]),
            driver_version: sym_opt(lib, &[b"nvmlSystemGetDriverVersion\0"]),
            vbios_version: sym_opt(lib, &[b"nvmlDeviceGetVbiosVersion\0"]),
            power_limit: sym_opt(lib, &[b"nvmlDeviceGetEnforcedPowerLimit\0"]),
            clock: sym_opt(lib, &[b"nvmlDeviceGetClockInfo\0"]),
            max_clock: sym_opt(lib, &[b"nvmlDeviceGetMaxClockInfo\0"]),
            temperature: sym_opt(lib, &[b"nvmlDeviceGetTemperature\0"]),
            power: sym_opt(lib, &[b"nvmlDeviceGetPowerUsage\0"]),
            energy: sym_opt(lib, &[b"nvmlDeviceGetTotalEnergyConsumption\0"]),
            pstate: sym_opt(lib, &[b"nvmlDeviceGetPerformanceState\0"]),
            utilization: sym_opt(lib, &[b"nvmlDeviceGetUtilizationRates\0"]),
            memory: sym_opt(lib, &[b"nvmlDeviceGetMemoryInfo\0"]),
            reasons: sym_opt(
                lib,
                &[
                    b"nvmlDeviceGetCurrentClocksEventReasons\0",
                    b"nvmlDeviceGetCurrentClocksThrottleReasons\0",
                ],
            ),
            pcie_gen: sym_opt(lib, &[b"nvmlDeviceGetCurrPcieLinkGeneration\0"]),
            pcie_width: sym_opt(lib, &[b"nvmlDeviceGetCurrPcieLinkWidth\0"]),
            pcie_max_gen: sym_opt(lib, &[b"nvmlDeviceGetMaxPcieLinkGeneration\0"]),
            pcie_max_width: sym_opt(lib, &[b"nvmlDeviceGetMaxPcieLinkWidth\0"]),
        })
    }

    fn get_u32(&self, f: Option<GetU32>) -> Option<u32> {
        let mut value = 0u32;
        // SAFETY: f came from this library, the device is live, and the out
        // pointer is a live local.
        (unsafe { f?(self.device, &mut value) } == NVML_SUCCESS).then_some(value)
    }

    fn get_u64(&self, f: Option<GetU64>) -> Option<u64> {
        let mut value = 0u64;
        // SAFETY: as in `get_u32`.
        (unsafe { f?(self.device, &mut value) } == NVML_SUCCESS).then_some(value)
    }

    fn get_keyed(&self, f: Option<GetClock>, key: u32) -> Option<u32> {
        let mut value = 0u32;
        // SAFETY: as in `get_u32`; `key` is a constant of the enum f expects.
        (unsafe { f?(self.device, key, &mut value) } == NVML_SUCCESS).then_some(value)
    }

    fn text(fill: impl FnOnce(*mut c_char, u32) -> i32) -> Option<String> {
        let mut buffer = [0 as c_char; TEXT_LEN];
        if fill(buffer.as_mut_ptr(), TEXT_LEN as u32) != NVML_SUCCESS {
            return None;
        }
        // SAFETY: NVML wrote a NUL-terminated string into the buffer on success.
        let text = unsafe { CStr::from_ptr(buffer.as_ptr()) };
        Some(text.to_string_lossy().trim().to_string()).filter(|s| !s.is_empty())
    }

    /// Facts that do not change during a run.
    pub fn static_info(&self) -> NvmlStatic {
        NvmlStatic {
            // SAFETY: the buffer and its length come from `text`, and the
            // device is live.
            driver_version: self
                .driver_version
                .and_then(|f| Self::text(|p, n| unsafe { f(p, n) })),
            // SAFETY: as above.
            vbios_version: self
                .vbios_version
                .and_then(|f| Self::text(|p, n| unsafe { f(self.device, p, n) })),
            power_limit_mw: self.get_u32(self.power_limit),
            max_sm_clock_mhz: self.get_keyed(self.max_clock, CLOCK_SM),
            max_mem_clock_mhz: self.get_keyed(self.max_clock, CLOCK_MEM),
            pcie_gen: self.get_u32(self.pcie_gen),
            pcie_width: self.get_u32(self.pcie_width),
            pcie_max_gen: self.get_u32(self.pcie_max_gen),
            pcie_max_width: self.get_u32(self.pcie_max_width),
        }
    }

    /// The card's state now.
    pub fn snapshot(&self) -> NvmlSnapshot {
        let utilization = self.utilization.and_then(|f| {
            let mut value = Utilization::default();
            // SAFETY: as in `get_u32`.
            (unsafe { f(self.device, &mut value) } == NVML_SUCCESS).then_some(value)
        });
        let memory = self.memory.and_then(|f| {
            let mut value = Memory::default();
            // SAFETY: as in `get_u32`.
            (unsafe { f(self.device, &mut value) } == NVML_SUCCESS).then_some(value)
        });
        NvmlSnapshot {
            sm_clock_mhz: self.get_keyed(self.clock, CLOCK_SM),
            mem_clock_mhz: self.get_keyed(self.clock, CLOCK_MEM),
            temperature_c: self.get_keyed(self.temperature, TEMPERATURE_GPU),
            power_mw: self.get_u32(self.power),
            gpu_util_pct: utilization.as_ref().map(|u| u.gpu),
            mem_util_pct: utilization.as_ref().map(|u| u.memory),
            vram_used_bytes: memory.as_ref().map(|m| m.used),
            pstate: self.get_u32(self.pstate),
            clock_reasons: self.get_u64(self.reasons),
        }
    }

    /// Energy the card has used since the driver loaded, millijoules. Two
    /// readings bracket a run exactly, which integrating sampled power cannot.
    pub fn energy_mj(&self) -> Option<u64> {
        self.get_u64(self.energy)
    }
}

impl Drop for Nvml {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown {
            // SAFETY: pairs with the init in `open`; errors are of no use here.
            unsafe {
                shutdown();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_bits_are_named_and_unknown_bits_are_dropped() {
        assert_eq!(reason_names(0), Vec::<&str>::new());
        assert_eq!(reason_names(0x1 | 0x20), ["gpu_idle", "sw_thermal_slowdown"]);
        assert_eq!(reason_names(1 << 40), Vec::<&str>::new());
    }
}
