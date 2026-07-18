use serde::{Deserialize, Serialize};

/// Device target untuk execution
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    Cpu,
    Cuda(u32),              // cuda:N
    Vulkan(u32),            // vulkan:N
    Metal,                  // Apple Metal
    Remote { rank: u32, device: Box<Device> }, // rank:N/cuda:M
}

impl Device {
    /// Parse a device string, returning `None` when unrecognised.
    ///
    /// Kept as an inherent method (shadowing `FromStr::from_str`) because the
    /// ARTX01 spec calls `Device::from_str(..) -> Option<Device>`; the
    /// `FromStr` impl below delegates here for idiomatic `str::parse()` use.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        if s == "cpu" { return Some(Self::Cpu); }
        if s == "metal" { return Some(Self::Metal); }
        if let Some(idx) = s.strip_prefix("cuda:") {
            return idx.parse().ok().map(Self::Cuda);
        }
        if let Some(idx) = s.strip_prefix("vulkan:") {
            return idx.parse().ok().map(Self::Vulkan);
        }
        None
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Vulkan(_) | Self::Metal)
    }
}

/// A sensible default so `DeviceMap` can derive `Default`.
impl Default for Device {
    fn default() -> Self { Self::Cpu }
}

impl std::str::FromStr for Device {
    type Err = crate::error::GllmError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        // Explicit inherent-method path: plain `Self::from_str` would resolve
        // back to this trait impl and recurse forever.
        Device::from_str(s).ok_or_else(|| {
            crate::error::GllmError::ValidationError(format!("unknown device: {s}"))
        })
    }
}

/// Execution unit — smallest independently loadable unit
#[derive(Debug, Clone)]
pub struct ExecutionUnit {
    pub id: ExecutionUnitId,
    pub file: String,
    pub checksum: String,
    pub device: Device,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutionUnitId {
    Shared,
    Layer(usize),
    Projector,
}

/// Device map — layer index → device assignment
#[derive(Debug, Clone, Default)]
pub struct DeviceMap {
    pub assignments: std::collections::HashMap<usize, Device>,
    pub default_device: Device,
}

impl DeviceMap {
    pub fn all_cpu() -> Self {
        Self {
            assignments: Default::default(),
            default_device: Device::Cpu,
        }
    }

    pub fn device_for_layer(&self, index: usize) -> &Device {
        self.assignments.get(&index).unwrap_or(&self.default_device)
    }
}
