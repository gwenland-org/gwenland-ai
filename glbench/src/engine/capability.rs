//! Known hardware ceilings, keyed by device name.
//!
//! glbench does not link a GPU SDK, so it cannot query peak bandwidth from the
//! driver. Instead it keeps a small table of published figures for the devices
//! the project actually validates on. This is reference data, not a measurement
//! — a device not in the table simply yields `None` and the ceiling analysis
//! declines to compute an efficiency, which is the honest outcome.

/// Published peak specs for a device.
#[derive(Debug, Clone, Copy)]
pub struct DeviceCeiling {
    /// Peak memory bandwidth, GB/s.
    pub peak_bandwidth_gbs: f64,
    /// Peak INT8 compute, TOPS (dense).
    pub peak_int8_tops: f64,
}

/// Look up published ceilings by substring match on the device name. Matching
/// is lenient (case-insensitive `contains`) because driver-reported names vary
/// ("Tesla T4", "NVIDIA T4", ...).
pub fn lookup(device_name: &str) -> Option<DeviceCeiling> {
    let n = device_name.to_ascii_lowercase();
    // Table intentionally small: the devices this project benchmarks on.
    const TABLE: &[(&str, DeviceCeiling)] = &[
        ("t4", DeviceCeiling { peak_bandwidth_gbs: 320.0, peak_int8_tops: 130.0 }),
        ("a100", DeviceCeiling { peak_bandwidth_gbs: 1555.0, peak_int8_tops: 624.0 }),
        ("v100", DeviceCeiling { peak_bandwidth_gbs: 900.0, peak_int8_tops: 62.0 }),
        ("l4", DeviceCeiling { peak_bandwidth_gbs: 300.0, peak_int8_tops: 242.0 }),
        ("rtx 3090", DeviceCeiling { peak_bandwidth_gbs: 936.0, peak_int8_tops: 284.0 }),
        ("rtx 4090", DeviceCeiling { peak_bandwidth_gbs: 1008.0, peak_int8_tops: 660.0 }),
    ];
    TABLE
        .iter()
        .find(|(key, _)| n.contains(key))
        .map(|(_, c)| *c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_t4_variants() {
        assert!(lookup("Tesla T4").is_some());
        assert!(lookup("NVIDIA T4").is_some());
    }

    #[test]
    fn unknown_is_none() {
        assert!(lookup("SomeFutureGPU 9000").is_none());
    }
}
