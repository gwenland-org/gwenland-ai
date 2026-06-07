use serde::Serialize;
use sysinfo::System;
use wgpu::{Instance, InstanceDescriptor, Backends};

#[derive(Serialize)]
pub enum Arch {
    X86_64,
    Aarch64,
    Unknown,
}

#[derive(Serialize, Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuType {
    Dedicated,
    Integrated,
    Unknown,
}

#[derive(Serialize, Clone, Debug)]
pub struct GpuProfile {
    pub name: String,
    pub vram_gb: f64,
    pub vendor: String,
    pub gpu_type: GpuType,
}

#[derive(Serialize)]
pub struct HardwareProfile {
    pub total_ram_gb: f64,
    pub available_ram_gb: f64,
    pub cpu_count: usize,
    pub cpu_brand: String,
    pub arch: Arch,
    pub gpus: Vec<GpuProfile>, // Null if GPU Not Loaded
}

pub fn detect_gpus() -> Vec<GpuProfile> {
    let instance = Instance::new(InstanceDescriptor {
        backends: Backends::all(),
        ..Default::default()
    });

    let adapters = instance.enumerate_adapters(Backends::all());
    let mut gpus = Vec::new();
    let mut seen = std::collections::HashSet::new();

    for adapter in adapters {
        let info = adapter.get_info();

        if info.name.contains("Microsoft Basic Render Driver") {
            continue;
        }

        let gpu_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => GpuType::Dedicated,
            wgpu::DeviceType::IntegratedGpu => GpuType::Integrated,
            _ => GpuType::Unknown,
        };

        let gpu_type_str = match gpu_type {
            GpuType::Dedicated => "Dedicated",
            GpuType::Integrated => "Integrated",
            GpuType::Unknown => "Unknown",
        };

        let key = (info.name.clone(), gpu_type_str);
        if !seen.insert(key) {
            continue;
        }

        let vendor = match info.vendor {
            0x8086 => "Intel",
            0x10DE => "NVIDIA",
            0x1002 => "AMD",
            0x1414 => "Microsoft",
            _ => "Unknown",
        }.to_string();

        gpus.push(GpuProfile {
            name: info.name,
            vendor,
            vram_gb: 0.0,
            gpu_type,
        });
    }

    gpus
}

pub fn profile() -> HardwareProfile {
    let mut sys = System::new_all();
    sys.refresh_all();

    let arch = match std::env::consts::ARCH {
        "x86_64" => Arch::X86_64,
        "aarch64" => Arch::Aarch64,
        _ => Arch::Unknown,
    };

    HardwareProfile {
        total_ram_gb: sys.total_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        available_ram_gb: sys.available_memory() as f64 / 1024.0 / 1024.0 / 1024.0,
        cpu_count: sys.cpus().len(),
        cpu_brand: sys.cpus()[0].brand().to_string(),
        arch,
        gpus: detect_gpus(),
    }
}

// GPU busy_percent via /sys/class/drm:
// Ubuntu 20.04+    → Intel i915 ✓, AMD amdgpu ✓, NVIDIA nouveau ✓
// Debian 11+       → same
// Arch Linux       → same (rolling, always latest)
// Fedora 35+       → same
// openSUSE Leap    → same
// NixOS            → same if kernel ≥5.10 + correct driver
// NVIDIA proprietary → /sys unreliable, use nvml feature flag instead
// WSL2             → /sys/class/drm may not exist → graceful None

#[derive(Serialize, Clone, Debug)]
pub struct UsageSample {
    pub ram_used_bytes: u64,
    pub ram_total_bytes: u64,
    pub cpu_usage_percent: f64,      // global average across all cores
    pub gpus: Vec<GpuUsageSample>,
}

#[derive(Serialize, Clone, Debug)]
pub struct GpuUsageSample {
    pub name: String,
    pub gpu_type: GpuType,
    pub busy_percent: Option<f64>,   // GPU core utilization
    pub vram_used_bytes: Option<u64>,
    pub vram_total_bytes: Option<u64>,
}

pub struct LiveMonitor {
    sys: System,
    gpus: Vec<GpuProfile>,
}

impl LiveMonitor {
    pub fn new() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();
        let gpus = detect_gpus();
        Self { sys, gpus }
    }

    pub fn sample(&mut self) -> UsageSample {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        
        let gpus = sample_gpu_usage(&self.gpus);
        
        UsageSample {
            ram_used_bytes: self.sys.used_memory(),
            ram_total_bytes: self.sys.total_memory(),
            cpu_usage_percent: self.sys.global_cpu_info().cpu_usage() as f64,
            gpus,
        }
    }
}

fn sample_gpu_usage(gpus: &[GpuProfile]) -> Vec<GpuUsageSample> {
    let mut samples = Vec::new();
    for (_idx, gpu) in gpus.iter().enumerate() {
        let mut busy_percent = None;
        let mut vram_used_bytes = None;
        let mut vram_total_bytes = None;

        #[cfg(feature = "nvidia")]
        {
            if gpu.vendor == "NVIDIA" {
                if let Ok(nvml) = nvml_wrapper::Nvml::init() {
                    if let Ok(device) = nvml.device_by_index(_idx as u32) {
                        if let Ok(rates) = device.utilization_rates() {
                            busy_percent = Some(rates.gpu as f64);
                        }
                        if let Ok(mem) = device.memory_info() {
                            vram_used_bytes = Some(mem.used);
                            vram_total_bytes = Some(mem.total);
                        }
                        samples.push(GpuUsageSample {
                            name: gpu.name.clone(),
                            gpu_type: gpu.gpu_type,
                            busy_percent,
                            vram_used_bytes,
                            vram_total_bytes,
                        });
                        continue;
                    }
                }
            }
        }

        // Intel / AMD Integrated, AMD Dedicated, or NVIDIA nouveau (open source driver):
        // Try reading in order: card0, card1
        let cards = [0, 1];
        for card in &cards {
            let busy_path = format!("/sys/class/drm/card{}/device/gpu_busy_percent", card);
            if let Ok(content) = std::fs::read_to_string(&busy_path) {
                if let Ok(val) = content.trim().parse::<f64>() {
                    busy_percent = Some(val);

                    // Also try to read VRAM info for this card
                    let vram_used_path = format!("/sys/class/drm/card{}/device/mem_info_vram_used", card);
                    let vram_total_path = format!("/sys/class/drm/card{}/device/mem_info_vram_total", card);
                    if let (Ok(used_str), Ok(total_str)) = (
                        std::fs::read_to_string(&vram_used_path),
                        std::fs::read_to_string(&vram_total_path),
                    ) {
                        vram_used_bytes = used_str.trim().parse::<u64>().ok();
                        vram_total_bytes = total_str.trim().parse::<u64>().ok();
                    }
                    break;
                }
            }
        }

        samples.push(GpuUsageSample {
            name: gpu.name.clone(),
            gpu_type: gpu.gpu_type,
            busy_percent,
            vram_used_bytes,
            vram_total_bytes,
        });
    }
    samples
}

pub fn check_disk_space(path: &std::path::Path) -> Option<(u64, u64)> {
    let disks = sysinfo::Disks::new_with_refreshed_list();
    
    // Find the disk that contains the path
    let mut best_match: Option<&sysinfo::Disk> = None;
    let mut longest_prefix = 0;

    for disk in disks.list() {
        let mount = disk.mount_point();
        if path.starts_with(mount) {
            let len = mount.as_os_str().len();
            if len > longest_prefix {
                longest_prefix = len;
                best_match = Some(disk);
            }
        }
    }

    best_match.map(|disk| (disk.total_space(), disk.available_space()))
}