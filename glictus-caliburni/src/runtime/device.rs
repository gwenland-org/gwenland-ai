//! Device placement resolution (ARTX05 AD-06, ARTX06 Â§"Device Mapping").
//!
//! Three sources can name a device for a layer, in descending priority:
//! the layer's own manifest entry, a range assignment in the device map, and
//! the runtime's default. Whatever wins is then checked against the devices
//! actually present; a device that is named but unavailable falls back to CPU
//! with a warning rather than failing the pass.

use std::collections::HashMap;

use crate::error::{GllmError, GllmResult};
use crate::manifest::{DevicePlacement, LayerManifest};
use crate::types::execution::Device;

/// Where a resolved placement came from â€” recorded so a fallback can say
/// *what* it overrode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSource {
    /// The layer's own `device` field in the manifest.
    LayerManifest,
    /// A `"START-END"` range in the device map.
    RangeAssignment,
    /// The device map's default.
    MapDefault,
    /// The runtime config's device.
    RuntimeDefault,
    /// CPU, substituted because the resolved device was unavailable.
    CpuFallback,
}

impl DeviceSource {
    /// Whether this source represents a fallback rather than an honoured
    /// request.
    pub fn is_fallback(self) -> bool {
        matches!(self, DeviceSource::CpuFallback)
    }
}

/// A resolved placement plus how it was arrived at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDevice {
    /// The device to execute on.
    pub device: Device,
    /// Which rule produced it.
    pub source: DeviceSource,
    /// The placement that was requested but could not be honoured, if any.
    ///
    /// `Some` exactly when `source` is [`DeviceSource::CpuFallback`], so a
    /// caller can log *what* was overridden instead of a bare "fell back".
    pub overridden: Option<DevicePlacement>,
}

impl ResolvedDevice {
    /// Whether this placement is a CPU fallback from something else.
    pub fn is_fallback(&self) -> bool {
        self.source.is_fallback()
    }
}

/// Default and per-range device assignments, as an ARTX06 device map.
#[derive(Debug, Clone)]
pub struct DeviceMapConfig {
    /// Device for layers no rule assigns.
    pub default_device: DevicePlacement,
    /// Explicit per-layer assignments, expanded from ranges.
    pub layer_overrides: HashMap<u32, DevicePlacement>,
}

impl Default for DeviceMapConfig {
    fn default() -> Self {
        DeviceMapConfig {
            default_device: DevicePlacement::CPU,
            layer_overrides: HashMap::new(),
        }
    }
}

impl DeviceMapConfig {
    /// A map placing everything on `default_device`.
    pub fn new(default_device: DevicePlacement) -> Self {
        DeviceMapConfig {
            default_device,
            layer_overrides: HashMap::new(),
        }
    }

    /// Assign every layer in a `"START-END"` range (inclusive).
    ///
    /// Ranges are expanded eagerly so lookup is a hash hit rather than a scan,
    /// and so an invalid range is rejected here rather than mid-pass.
    pub fn assign_range(&mut self, range: &str, device: DevicePlacement) -> GllmResult<()> {
        let (start, end) = parse_range(range)?;
        for i in start..=end {
            self.layer_overrides.insert(i, device.clone());
        }
        Ok(())
    }

    /// Build from ARTX06's `{"default": ..., "layers": {"0-39": ...}}` shape.
    pub fn from_ranges(
        default_device: DevicePlacement,
        ranges: &[(&str, DevicePlacement)],
    ) -> GllmResult<Self> {
        let mut map = DeviceMapConfig::new(default_device);
        for (range, device) in ranges {
            map.assign_range(range, device.clone())?;
        }
        Ok(map)
    }

    /// The placement assigned to a layer by a range, if any.
    pub fn placement_for(&self, layer_index: u32) -> Option<&DevicePlacement> {
        self.layer_overrides.get(&layer_index)
    }
}

/// Parse `"START-END"` (inclusive) into its bounds.
///
/// A bare `"7"` is accepted as the single-layer range `7-7`. Returns
/// [`GllmError::InvalidLayerRange`] on malformed input or `START > END`.
pub fn parse_range(s: &str) -> GllmResult<(u32, u32)> {
    let trimmed = s.trim();
    let invalid = || GllmError::InvalidLayerRange(s.to_string());

    let (start, end) = match trimmed.split_once('-') {
        Some((a, b)) => (
            a.trim().parse::<u32>().map_err(|_| invalid())?,
            b.trim().parse::<u32>().map_err(|_| invalid())?,
        ),
        None => {
            let only = trimmed.parse::<u32>().map_err(|_| invalid())?;
            (only, only)
        }
    };

    if start > end {
        return Err(invalid());
    }
    Ok((start, end))
}

/// Resolves layer placements against the devices actually available.
#[derive(Debug, Clone)]
pub struct DeviceMapResolver {
    available: Vec<Device>,
}

impl DeviceMapResolver {
    /// Build a resolver over the devices present on this machine.
    ///
    /// CPU is always appended: it is the fallback of last resort, and a
    /// resolver without it could not honour AD-06's final rule.
    pub fn new(available: impl IntoIterator<Item = Device>) -> Self {
        let mut available: Vec<Device> = available.into_iter().collect();
        if !available.contains(&Device::Cpu) {
            available.push(Device::Cpu);
        }
        DeviceMapResolver { available }
    }

    /// A resolver for a machine with no accelerators.
    pub fn cpu_only() -> Self {
        DeviceMapResolver {
            available: vec![Device::Cpu],
        }
    }

    /// Devices this resolver considers usable.
    pub fn available(&self) -> &[Device] {
        &self.available
    }

    /// Whether a device is present.
    pub fn is_available(&self, device: &Device) -> bool {
        self.available.contains(device)
    }

    /// Resolve one layer's device using the AD-06 priority chain:
    /// layer manifest â†’ range assignment â†’ map default â†’ runtime default,
    /// then CPU if the winner is unavailable or unparseable.
    pub fn resolve(
        &self,
        layer_index: u32,
        layer_manifest: Option<&LayerManifest>,
        map_config: &DeviceMapConfig,
        runtime_default: &DevicePlacement,
    ) -> ResolvedDevice {
        let (placement, source) = layer_manifest
            .and_then(|m| m.device.as_ref())
            .map(|p| (p, DeviceSource::LayerManifest))
            .or_else(|| {
                map_config
                    .placement_for(layer_index)
                    .map(|p| (p, DeviceSource::RangeAssignment))
            })
            .unwrap_or_else(|| {
                // The map's default outranks the runtime's only when it says
                // something other than CPU; an unset map defaults to CPU and
                // must not silently override an explicit runtime device.
                if map_config.default_device == DevicePlacement::CPU {
                    (runtime_default, DeviceSource::RuntimeDefault)
                } else {
                    (&map_config.default_device, DeviceSource::MapDefault)
                }
            });

        match placement.to_device() {
            Some(device) if self.is_available(&device) => ResolvedDevice {
                device,
                source,
                overridden: None,
            },
            // Named a real device we don't have, or a string we can't parse
            // (ARTX10's "rank:0/cuda:0"). Both are recoverable: run on CPU.
            _ => ResolvedDevice {
                device: Device::Cpu,
                source: DeviceSource::CpuFallback,
                overridden: Some(placement.clone()),
            },
        }
    }
}

impl Default for DeviceMapResolver {
    fn default() -> Self {
        Self::cpu_only()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ExtensionUri, KnownDevicePlacement};

    fn layer_with_device(index: u32, device: Option<DevicePlacement>) -> LayerManifest {
        LayerManifest {
            index,
            file: crate::manifest::format_layer_filename(index),
            checksum: format!("sha256:{}", "0".repeat(64)),
            layer_type: ExtensionUri::parse("gllm:transformer/standard@v1")
                .expect("well-formed extension URI"),
            tensors: Vec::new(),
            device,
        }
    }

    fn cuda(n: u32) -> DevicePlacement {
        DevicePlacement::Other(format!("cuda:{n}"))
    }

    // --- parse_range -------------------------------------------------------

    #[test]
    fn parses_inclusive_ranges() {
        assert_eq!(parse_range("0-39").unwrap(), (0, 39));
        assert_eq!(parse_range("40-79").unwrap(), (40, 79));
        assert_eq!(parse_range("5-5").unwrap(), (5, 5));
    }

    #[test]
    fn parses_a_bare_index_as_a_single_layer_range() {
        assert_eq!(parse_range("7").unwrap(), (7, 7));
    }

    #[test]
    fn tolerates_surrounding_whitespace() {
        assert_eq!(parse_range("  0 - 39 ").unwrap(), (0, 39));
    }

    #[test]
    fn rejects_malformed_and_inverted_ranges() {
        for bad in ["", "-", "a-b", "0-", "-39", "39-0", "1-2-3", "0..39"] {
            assert!(
                parse_range(bad).is_err(),
                "{bad:?} should not parse"
            );
        }
        assert!(matches!(
            parse_range("39-0").unwrap_err(),
            GllmError::InvalidLayerRange(_)
        ));
    }

    // --- DeviceMapConfig ---------------------------------------------------

    #[test]
    fn range_assignment_expands_to_every_layer_in_it() {
        let mut map = DeviceMapConfig::default();
        map.assign_range("2-4", cuda(0)).unwrap();

        assert!(map.placement_for(1).is_none());
        for i in 2..=4 {
            assert_eq!(map.placement_for(i), Some(&cuda(0)), "layer {i}");
        }
        assert!(map.placement_for(5).is_none());
    }

    #[test]
    fn later_ranges_win_where_they_overlap() {
        let map = DeviceMapConfig::from_ranges(
            DevicePlacement::CPU,
            &[("0-9", cuda(0)), ("5-9", cuda(1))],
        )
        .unwrap();
        assert_eq!(map.placement_for(4), Some(&cuda(0)));
        assert_eq!(map.placement_for(5), Some(&cuda(1)), "overlap: last wins");
    }

    #[test]
    fn an_invalid_range_is_rejected_at_build_time() {
        let err = DeviceMapConfig::from_ranges(DevicePlacement::CPU, &[("9-0", cuda(0))])
            .unwrap_err();
        assert!(matches!(err, GllmError::InvalidLayerRange(_)));
    }

    // --- resolve: priority chain -------------------------------------------

    #[test]
    fn layer_manifest_device_outranks_everything() {
        let resolver = DeviceMapResolver::new([Device::Cuda(0), Device::Cuda(1)]);
        let mut map = DeviceMapConfig::new(cuda(1));
        map.assign_range("0-9", cuda(1)).unwrap();

        let layer = layer_with_device(3, Some(cuda(0)));
        let r = resolver.resolve(3, Some(&layer), &map, &cuda(1));

        assert_eq!(r.device, Device::Cuda(0), "manifest wins");
        assert_eq!(r.source, DeviceSource::LayerManifest);
        assert!(!r.is_fallback());
    }

    #[test]
    fn range_assignment_wins_when_the_layer_says_nothing() {
        let resolver = DeviceMapResolver::new([Device::Cuda(1)]);
        let map =
            DeviceMapConfig::from_ranges(DevicePlacement::CPU, &[("0-9", cuda(1))]).unwrap();

        let layer = layer_with_device(5, None);
        let r = resolver.resolve(5, Some(&layer), &map, &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cuda(1));
        assert_eq!(r.source, DeviceSource::RangeAssignment);
    }

    #[test]
    fn falls_through_to_the_runtime_default() {
        let resolver = DeviceMapResolver::cpu_only();
        let r = resolver.resolve(0, None, &DeviceMapConfig::default(), &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cpu);
        assert_eq!(r.source, DeviceSource::RuntimeDefault);
        assert!(r.overridden.is_none());
    }

    #[test]
    fn a_non_cpu_map_default_outranks_the_runtime_default() {
        let resolver = DeviceMapResolver::new([Device::Cuda(0)]);
        let map = DeviceMapConfig::new(cuda(0));
        let r = resolver.resolve(0, None, &map, &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cuda(0));
        assert_eq!(r.source, DeviceSource::MapDefault);
    }

    // --- resolve: fallback -------------------------------------------------

    #[test]
    fn unavailable_device_falls_back_to_cpu_and_says_what_it_dropped() {
        let resolver = DeviceMapResolver::cpu_only(); // no GPU present
        let layer = layer_with_device(0, Some(cuda(0)));
        let r = resolver.resolve(0, Some(&layer), &DeviceMapConfig::default(), &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cpu);
        assert_eq!(r.source, DeviceSource::CpuFallback);
        assert!(r.is_fallback());
        assert_eq!(
            r.overridden,
            Some(cuda(0)),
            "the dropped request must be reportable"
        );
    }

    #[test]
    fn an_unparseable_device_string_falls_back_rather_than_guessing() {
        // ARTX10 rank strings don't resolve to a Device yet.
        let resolver = DeviceMapResolver::cpu_only();
        let placement = DevicePlacement::Other("rank:0/cuda:0".into());
        let layer = layer_with_device(0, Some(placement.clone()));
        let r = resolver.resolve(0, Some(&layer), &DeviceMapConfig::default(), &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cpu);
        assert_eq!(r.source, DeviceSource::CpuFallback);
        assert_eq!(r.overridden, Some(placement));
    }

    #[test]
    fn a_high_cuda_index_resolves_when_present() {
        // The whole point of preserving the raw string: cuda:7 used to be
        // unrecoverable, and would have fallen back to CPU here.
        let resolver = DeviceMapResolver::new([Device::Cuda(7)]);
        let layer = layer_with_device(0, Some(cuda(7)));
        let r = resolver.resolve(0, Some(&layer), &DeviceMapConfig::default(), &DevicePlacement::CPU);

        assert_eq!(r.device, Device::Cuda(7));
        assert_eq!(r.source, DeviceSource::LayerManifest);
        assert!(!r.is_fallback());
    }

    #[test]
    fn named_variants_resolve_through_the_same_path() {
        let resolver = DeviceMapResolver::new([Device::Cuda(0)]);
        let layer = layer_with_device(0, Some(DevicePlacement::CUDA0));
        let r = resolver.resolve(0, Some(&layer), &DeviceMapConfig::default(), &DevicePlacement::CPU);
        assert_eq!(r.device, Device::Cuda(0));

        let metal = DevicePlacement::Known(KnownDevicePlacement::Metal);
        assert_eq!(metal.to_device(), Some(Device::Metal));
    }

    // --- resolver construction ---------------------------------------------

    #[test]
    fn cpu_is_always_available_even_if_unlisted() {
        let resolver = DeviceMapResolver::new([Device::Cuda(0)]);
        assert!(
            resolver.is_available(&Device::Cpu),
            "CPU is the fallback of last resort and must always be present"
        );
        assert_eq!(resolver.available().len(), 2);
    }

    #[test]
    fn cpu_is_not_duplicated_when_listed() {
        let resolver = DeviceMapResolver::new([Device::Cpu, Device::Cuda(0)]);
        assert_eq!(resolver.available().len(), 2);
    }

    #[test]
    fn default_resolver_is_cpu_only() {
        let r = DeviceMapResolver::default();
        assert_eq!(r.available(), &[Device::Cpu]);
        assert!(!r.is_available(&Device::Cuda(0)));
    }
}
