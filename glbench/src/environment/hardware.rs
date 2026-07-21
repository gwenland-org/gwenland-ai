//! The environment and hardware snapshots — the aggregate "where did this run"
//! record stamped into every session.
//!
//! [`HardwareSnapshot`] is the physical machine (CPU, GPU, memory, storage).
//! [`EnvironmentSnapshot`] wraps it with runtime/build facts. Both are pure
//! data with a JSON projection; the analysis layer reads them for ceilings.

use crate::core::schema::ToJson;
use crate::environment::cpu::CpuInfo;
use crate::environment::gpu::GpuInfo;
use crate::environment::memory::MemoryInfo;
use crate::environment::runtime::RuntimeInfo;
use crate::environment::storage::StorageInfo;
use crate::environment::thermal::ThermalSnapshot;
use crate::export::json::Json;

/// A snapshot of the physical machine at benchmark time.
#[derive(Debug, Clone)]
pub struct HardwareSnapshot {
    /// CPU facts.
    pub cpu: CpuInfo,
    /// GPU facts (empty on a CPU-only run; filled by the engine adapter).
    pub gpu: GpuInfo,
    /// System memory facts.
    pub memory: MemoryInfo,
    /// Model-file storage facts.
    pub storage: StorageInfo,
    /// CPU clock at the start and (once [`Self::with_end_mhz`] is called)
    /// end of the session, for thermal-throttle detection.
    pub thermal: ThermalSnapshot,
}

impl HardwareSnapshot {
    /// Probe CPU/memory/storage now. GPU facts are attached separately by the
    /// engine adapter (only the engine knows its device), via [`Self::with_gpu`].
    pub fn probe(model_path: &str) -> HardwareSnapshot {
        let cpu = CpuInfo::probe();
        let thermal = ThermalSnapshot { start_mhz: cpu.mhz, end_mhz: None, avg_mhz: None };
        HardwareSnapshot {
            cpu,
            gpu: GpuInfo::default(),
            memory: MemoryInfo::probe(),
            storage: StorageInfo::probe(model_path),
            thermal,
        }
    }

    /// Attach GPU facts reported by the active engine.
    pub fn with_gpu(mut self, gpu: GpuInfo) -> Self {
        self.gpu = gpu;
        self
    }

    /// Record the end-of-session CPU clock reading. Called once, after the
    /// measured iterations finish — a second, independent
    /// [`crate::environment::cpu::probe_mhz`] call, not a re-run of the
    /// (expensive) full [`CpuInfo::probe`].
    pub fn with_end_mhz(mut self, end_mhz: Option<f64>) -> Self {
        self.thermal.end_mhz = end_mhz;
        self
    }

    /// Record the mean of the per-iteration clock readings taken during the
    /// measured phase (see [`crate::environment::thermal::average_mhz`]).
    pub fn with_avg_mhz(mut self, avg_mhz: Option<f64>) -> Self {
        self.thermal.avg_mhz = avg_mhz;
        self
    }
}

impl ToJson for HardwareSnapshot {
    fn to_json(&self) -> Json {
        Json::obj([
            (
                "cpu",
                Json::obj([
                    ("logical_cores", Json::n(self.cpu.logical_cores as f64)),
                    (
                        "physical_cores",
                        opt_num(self.cpu.physical_cores.map(|n| n as f64)),
                    ),
                    ("model", opt_str(self.cpu.model.as_deref())),
                    ("mhz", opt_num(self.cpu.mhz)),
                    // Measured ceiling — every efficiency figure in the report
                    // is a fraction of this, so it must travel with the archive.
                    ("read_bandwidth_gbs", opt_num(self.cpu.read_bandwidth_gbs)),
                    // What the CPU *supports*. What the engine actually chose
                    // is a different fact, and lives in telemetry.backend.
                    (
                        "isa",
                        Json::obj([
                            ("avx2", Json::Bool(self.cpu.isa.avx2)),
                            ("fma", Json::Bool(self.cpu.isa.fma)),
                            ("f16c", Json::Bool(self.cpu.isa.f16c)),
                            ("avx512f", Json::Bool(self.cpu.isa.avx512f)),
                            ("avx512bw", Json::Bool(self.cpu.isa.avx512bw)),
                            ("avx512_vnni", Json::Bool(self.cpu.isa.avx512_vnni)),
                            ("avx_vnni", Json::Bool(self.cpu.isa.avx_vnni)),
                        ]),
                    ),
                ]),
            ),
            (
                "gpu",
                Json::obj([
                    ("name", opt_str(self.gpu.name.as_deref())),
                    ("backend", opt_str(self.gpu.backend.as_deref())),
                    ("compute", opt_str(self.gpu.compute.as_deref())),
                    (
                        "total_memory_bytes",
                        opt_num(self.gpu.total_memory_bytes.map(|b| b as f64)),
                    ),
                    ("peak_bandwidth_gbs", opt_num(self.gpu.peak_bandwidth_gbs)),
                    ("peak_compute_tops", opt_num(self.gpu.peak_compute_tops)),
                ]),
            ),
            (
                "memory",
                Json::obj([
                    (
                        "total_bytes",
                        opt_num(self.memory.total_bytes.map(|b| b as f64)),
                    ),
                    (
                        "available_bytes",
                        opt_num(self.memory.available_bytes.map(|b| b as f64)),
                    ),
                ]),
            ),
            (
                "storage",
                Json::obj([(
                    "model_file_bytes",
                    opt_num(self.storage.model_file_bytes.map(|b| b as f64)),
                )]),
            ),
            (
                "thermal",
                Json::obj([
                    ("start_mhz", opt_num(self.thermal.start_mhz)),
                    ("end_mhz", opt_num(self.thermal.end_mhz)),
                    ("avg_mhz", opt_num(self.thermal.avg_mhz)),
                    ("throttled", Json::Bool(self.thermal.throttled())),
                ]),
            ),
            // Forward-compat placeholder for gljax / Sanctum Visibilia.
            // Always null: no TPU detection exists yet, and this field must
            // never read as "checked, none found" until it actually is.
            ("tpu", Json::Null),
        ])
    }
}

/// The full environment: hardware plus runtime/build facts.
#[derive(Debug, Clone)]
pub struct EnvironmentSnapshot {
    /// The physical machine.
    pub hardware: HardwareSnapshot,
    /// The glbench build and host OS.
    pub runtime: RuntimeInfo,
}

impl EnvironmentSnapshot {
    /// Probe the environment for a run against `model_path`.
    pub fn probe(model_path: &str) -> EnvironmentSnapshot {
        EnvironmentSnapshot {
            hardware: HardwareSnapshot::probe(model_path),
            runtime: RuntimeInfo::probe(),
        }
    }
}

impl ToJson for EnvironmentSnapshot {
    fn to_json(&self) -> Json {
        Json::obj([
            ("hardware", self.hardware.to_json()),
            (
                "runtime",
                Json::obj([
                    ("os", Json::s(self.runtime.os.clone())),
                    ("arch", Json::s(self.runtime.arch.clone())),
                    ("glbench_version", Json::s(self.runtime.glbench_version.clone())),
                    ("build_profile", Json::s(self.runtime.build_profile)),
                ]),
            ),
        ])
    }
}

/// Encode an optional string field as the value or JSON null.
fn opt_str(v: Option<&str>) -> Json {
    match v {
        Some(s) => Json::s(s),
        None => Json::Null,
    }
}

/// Encode an optional number field as the value or JSON null.
fn opt_num(v: Option<f64>) -> Json {
    match v {
        Some(n) => Json::n(n),
        None => Json::Null,
    }
}
