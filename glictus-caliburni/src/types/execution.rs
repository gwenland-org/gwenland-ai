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
    ///
    /// ARTX10's `"rank:N/cuda:M"` device-map strings parse to
    /// [`Self::Remote`] — recognised as a *shape*, not something this crate
    /// can execute on: no runtime here opens a rank connection, so a caller
    /// getting `Some(Device::Remote { .. })` back still cannot schedule work
    /// on it. See [`DeviceMapResolver`](crate::runtime::device::DeviceMapResolver),
    /// which resolves a `Remote` placement to a CPU fallback specifically
    /// tagged as a distributed one, rather than a generic "device
    /// unavailable" fallback.
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
        if let Some(rest) = s.strip_prefix("rank:") {
            let (rank_str, device_str) = rest.split_once('/')?;
            let rank = rank_str.parse().ok()?;
            let device = Self::from_str(device_str)?;
            return Some(Self::Remote { rank, device: Box::new(device) });
        }
        None
    }

    pub fn is_gpu(&self) -> bool {
        matches!(self, Self::Cuda(_) | Self::Vulkan(_) | Self::Metal)
    }

    /// Whether this device names a remote rank (ARTX10 distributed device map).
    pub fn is_remote(&self) -> bool {
        matches!(self, Self::Remote { .. })
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

/// Manifest-level metadata about an execution unit (ARTX01).
///
/// Renamed from `ExecutionUnit` in ARTX02: the file-handle type in
/// [`crate::execution_unit`] now owns that name; this struct is the
/// manifest's *description* of a unit, not an opened file.
#[derive(Debug, Clone)]
pub struct ExecutionUnitMeta {
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
