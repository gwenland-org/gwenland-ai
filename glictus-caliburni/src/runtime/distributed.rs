//! Distributed runtime — forward-compatibility placeholder (ARTX10 Wave 2).
//!
//! ARTX10 describes multi-host/multi-process model parallelism: a device map
//! whose placements are `"rank:N/cuda:M"` instead of a plain local device,
//! pipeline parallelism (rank *N* holds a contiguous layer range, activations
//! `send`/`recv` to rank *N+1*), tensor parallelism (a single layer split
//! across ranks via all-gather/reduce-scatter), and checkpoint-based failure
//! recovery.
//!
//! **None of that is implemented here.** This module is the shape ARTX10
//! specifies, with nothing behind it — no NCCL, no RCCL, no Gloo, no
//! `send`/`recv`, no cross-process anything. Standing up an actual
//! communication backend is deferred to a later initiative (internally
//! "Sanctum Visibilia"). What exists today:
//!
//! - [`Device::Remote`](crate::types::execution::Device::Remote) parses
//!   `"rank:N/cuda:M"` device-map strings (see
//!   [`Device::from_str`](crate::types::execution::Device::from_str)).
//! - [`DeviceMapResolver`](super::device::DeviceMapResolver) recognises a
//!   resolved [`Device::Remote`] placement and falls back to CPU, tagged as
//!   [`DeviceSource::DistributedUnavailable`](super::device::DeviceSource::DistributedUnavailable)
//!   rather than a generic "device missing" fallback.
//! - [`RankTopology`] (below), which reads *how many ranks a device map
//!   implies* and *which layer ranges belong to which rank* — a static
//!   summary a future distributed runtime would need, computed today purely
//!   for visibility (logging, `gwen doctor`-style diagnostics) since nothing
//!   consumes it to actually schedule cross-rank work.
//!
//! A manifest with rank placements loads and runs correctly today: every
//! layer executes locally on CPU, exactly as if the ranks were never there.
//! It is slower than real pipeline parallelism would be (no work is actually
//! distributed) but it is not wrong — every layer still runs, in order,
//! with a real result.

use std::collections::BTreeMap;

use crate::manifest::GllmManifest;
use crate::types::execution::Device;

/// One rank's contiguous slice of layers, as implied by a device map's
/// `"rank:N/..."` placements.
///
/// Layer indices, not a `"START-END"` string: the manifest's ranges are
/// already expanded by [`DeviceMapConfig`](super::device::DeviceMapConfig)
/// before this summary is built, so a rank's layers need not be contiguous
/// in the manifest even though ARTX10's pipeline-parallel diagram assumes
/// they are — a topology that turned out non-contiguous is exactly the kind
/// of thing worth being able to observe rather than silently coalescing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RankAssignment {
    /// Layer indices assigned to this rank, in ascending order.
    pub layers: Vec<u32>,
    /// The local device string this rank names for its layers (e.g.
    /// `"cuda:0"`) — informational only; nothing here connects to it.
    pub local_device: String,
}

/// Summary of the ranks a manifest's per-layer device placements imply.
///
/// Built once from a manifest, read-only after that — a future distributed
/// runtime would use this to know which ranks to connect to and what each
/// owns; today nothing does, and building this is the entire scope of
/// ARTX10 Wave 2's "distributed types as forward-compat placeholder."
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RankTopology {
    ranks: BTreeMap<u32, RankAssignment>,
}

impl RankTopology {
    /// Derive a topology from a manifest's per-layer device placements.
    ///
    /// Layers with no placement, or a placement that is not
    /// `Device::Remote` (plain `"cpu"`, `"cuda:0"`, an unparseable string),
    /// are not part of any rank — this only reports what the manifest
    /// explicitly asked to run remotely.
    pub fn from_manifest(manifest: &GllmManifest) -> Self {
        let mut ranks: BTreeMap<u32, RankAssignment> = BTreeMap::new();
        for layer in &manifest.layers {
            let Some(placement) = &layer.device else { continue };
            let Some(Device::Remote { rank, device }) = placement.to_device() else { continue };
            let entry = ranks.entry(rank).or_default();
            entry.layers.push(layer.index);
            // First writer wins: ARTX10 assumes one local device per rank,
            // so a manifest naming two different devices for the same rank
            // is describing something this summary cannot represent — not
            // this module's job to validate (that belongs with the
            // manifest validator, should this ever stop being a stub).
            if entry.local_device.is_empty() {
                entry.local_device = format!("{device:?}");
            }
        }
        for assignment in ranks.values_mut() {
            assignment.layers.sort_unstable();
        }
        RankTopology { ranks }
    }

    /// Number of distinct ranks the manifest names.
    ///
    /// `0` for an ordinary local manifest — the common case, and the reason
    /// this is a count rather than an `Option`: "no distributed ranks" is a
    /// valid, frequent topology, not a missing one.
    pub fn rank_count(&self) -> usize {
        self.ranks.len()
    }

    /// Whether any layer names a distributed rank placement.
    pub fn is_distributed(&self) -> bool {
        !self.ranks.is_empty()
    }

    /// The layer assignment for one rank, if the manifest names it.
    pub fn rank(&self, rank: u32) -> Option<&RankAssignment> {
        self.ranks.get(&rank)
    }

    /// Every rank index the manifest names, ascending.
    pub fn rank_ids(&self) -> Vec<u32> {
        self.ranks.keys().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::fixtures::minimal_manifest_json;

    fn manifest_with_devices(num_layers: u32, devices: &[(u32, &str)]) -> GllmManifest {
        let json = minimal_manifest_json(num_layers);
        let mut manifest = GllmManifest::from_str(&json).unwrap();
        for &(index, device) in devices {
            manifest.layers[index as usize].device =
                Some(crate::manifest::DevicePlacement::Other(device.to_string()));
        }
        manifest
    }

    #[test]
    fn a_manifest_with_no_rank_placements_is_not_distributed() {
        let manifest = manifest_with_devices(3, &[(0, "cpu"), (1, "cuda:0")]);
        let topo = RankTopology::from_manifest(&manifest);
        assert!(!topo.is_distributed());
        assert_eq!(topo.rank_count(), 0);
        assert_eq!(topo.rank_ids(), Vec::<u32>::new());
    }

    #[test]
    fn groups_layers_by_rank_from_device_map_style_placements() {
        // Mirrors ARTX10's example: layers 0-1 on rank 0, layer 2 on rank 1.
        let manifest = manifest_with_devices(
            3,
            &[
                (0, "rank:0/cuda:0"),
                (1, "rank:0/cuda:0"),
                (2, "rank:1/cuda:0"),
            ],
        );
        let topo = RankTopology::from_manifest(&manifest);

        assert!(topo.is_distributed());
        assert_eq!(topo.rank_count(), 2);
        assert_eq!(topo.rank_ids(), vec![0, 1]);
        assert_eq!(topo.rank(0).unwrap().layers, vec![0, 1]);
        assert_eq!(topo.rank(1).unwrap().layers, vec![2]);
        assert!(topo.rank(2).is_none());
    }

    #[test]
    fn a_mix_of_local_and_remote_placements_only_counts_the_remote_ones() {
        let manifest = manifest_with_devices(3, &[(0, "cpu"), (1, "rank:0/cuda:0"), (2, "cuda:0")]);
        let topo = RankTopology::from_manifest(&manifest);

        assert_eq!(topo.rank_count(), 1);
        assert_eq!(topo.rank(0).unwrap().layers, vec![1]);
    }

    #[test]
    fn an_unparseable_device_string_is_not_a_rank() {
        let manifest = manifest_with_devices(1, &[(0, "not-a-device")]);
        let topo = RankTopology::from_manifest(&manifest);
        assert!(!topo.is_distributed());
    }

    #[test]
    fn empty_manifest_layers_yield_an_empty_topology() {
        let manifest = manifest_with_devices(0, &[]);
        let topo = RankTopology::from_manifest(&manifest);
        assert_eq!(topo.rank_count(), 0);
        assert!(!topo.is_distributed());
    }
}
