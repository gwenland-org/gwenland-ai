//! Stummañ Pik: checkpoint persistence.
//!
//! # Two traits, not five
//!
//! The obvious shape is one `Checkpoint` trait with five implementors. It does
//! not survive the question "what does `load()` mean for each of them":
//!
//! | | [`CheckpointStore`] | [`Exporter`] |
//! |---|---|---|
//! | Members | [`CPLora`], [`CPFull`], [`CPSharded`], [`CPIncremental`] | [`PLGgufMerge`] |
//! | Round-trips | yes, bit-exact | **no**, lossy by construction |
//! | Has `load()` | yes | **meaningless** |
//! | Purpose | resume training | produce a deployment artifact |
//!
//! `PLGgufMerge` reads an adapter plus a base model, computes `W0 + scale*A@B`,
//! and **requantizes** to a GGUF type. Nothing can resume from its output. Put
//! it behind `CheckpointStore` and it must carry a `load()` that can never be
//! implemented, which is the "meaningless method on a stub" shape this repo's
//! skills forbid. That is also why it is named `PL` (pipeline) and not `CP`:
//! the prefix records the split rather than a naming preference.
//!
//! The other four genuinely do share one interface, because they are all
//! layouts over the same logical content: a named tensor bundle plus typed
//! metadata.
//!
//! # A checkpoint is a directory, not a file
//!
//! Segments have different lifetimes and different consumers. Adapter state is
//! needed to resume *and* to deploy; optimizer state is needed only to resume
//! and is 1-2x the adapter's size; training progress is tiny. Keeping them in
//! separate files means a deploy path never parses optimizer bytes it will
//! never use, and `gltrain`'s own prior art already landed here with its
//! `{stem}_adamw.safetensors` sidecar.
//!
//! # Names are the primary key, everywhere
//!
//! `TensorId` is process-global and explicitly not persistable, so every tensor
//! in every segment is keyed by [`crate::nn::TPParameter::name`]. Optimizer
//! state, which is keyed by `TensorId` in memory, is re-keyed to name at
//! exactly the save/load boundary and nowhere else.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use crate::optim::VLNamedTensor;
use std::collections::BTreeMap;
use std::path::Path;

pub mod export_gguf;
pub mod full;
pub mod incremental;
pub mod json;
pub mod lora_ckpt;
pub mod manifest;
pub mod safetensors;
pub mod sharded;

pub use export_gguf::PLGgufMerge;
pub use full::CPFull;
pub use incremental::CPIncremental;
pub use lora_ckpt::CPLora;
pub use manifest::{
    ENVersionCompatibility, VLFormatVersion, VLManifest, VLValidation, FORMAT_VERSION,
    MANIFEST_FILE,
};
pub use safetensors::ENTensorEntry;
pub use sharded::CPSharded;

/// Which part of a training run a segment holds.
///
/// `EN` because a closed set of variants is this type's whole job. The variants
/// are exactly the categories whose "needed to resume?" and "needed to deploy?"
/// answers disagree, which is why they are separate files at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ENSegment {
    /// Adapter tensors. Needed to resume and to deploy.
    Adapter,
    /// Optimizer moments. Needed to resume, never to deploy.
    Optimizer,
}

impl ENSegment {
    /// The file this segment is written to inside a checkpoint directory.
    pub fn file_name(&self) -> &'static str {
        match self {
            ENSegment::Adapter => "adapter.safetensors",
            ENSegment::Optimizer => "optimizer.safetensors",
        }
    }
}

/// One checkpoint's full contents, in memory.
///
/// Not generic over a backend: by the time state is being written it is host
/// data, and a file has no backend. This is what lets [`CheckpointStore`] be a
/// backend-free trait.
#[derive(Debug, Clone, PartialEq)]
pub struct VLCheckpoint {
    /// What the directory says about itself.
    pub manifest: VLManifest,
    /// Segment contents, keyed by segment. `BTreeMap` for deterministic
    /// iteration, so a checkpoint written twice is byte-identical.
    pub segments: BTreeMap<ENSegment, Vec<VLNamedTensor>>,
}

impl VLCheckpoint {
    /// A checkpoint holding only a manifest.
    pub fn new(manifest: VLManifest) -> Self {
        Self {
            manifest,
            segments: BTreeMap::new(),
        }
    }

    /// Attach a segment. Replaces one already present under the same key.
    pub fn with_segment(mut self, kind: ENSegment, tensors: Vec<VLNamedTensor>) -> Self {
        self.segments.insert(kind, tensors);
        self
    }

    /// A segment's tensors, if present.
    pub fn segment(&self, kind: ENSegment) -> Option<&[VLNamedTensor]> {
        self.segments.get(&kind).map(Vec::as_slice)
    }

    /// One named tensor from a segment.
    pub fn tensor(&self, kind: ENSegment, name: &str) -> Option<&VLNamedTensor> {
        self.segment(kind)?.iter().find(|t| t.name == name)
    }

    /// A named tensor from a segment, erroring when absent.
    pub fn require_tensor(&self, kind: ENSegment, name: &str) -> Result<&VLNamedTensor> {
        self.tensor(kind, name).ok_or_else(|| {
            GlTrainError::Checkpoint(format!(
                "{} has no tensor named '{name}'",
                kind.file_name()
            ))
        })
    }
}

/// Machine-readable facts about a checkpoint layout.
///
/// Mirrors [`crate::nn::VLAdapterCapability`] and
/// [`crate::optim::VLOptimizerCapability`].
#[derive(Debug, Clone)]
pub struct VLCheckpointFormat {
    /// Registry key, e.g. `"lora"`.
    pub id: &'static str,
    /// Implemented, or a stub with a reason and the milestone that owns it.
    pub status: ENSkillStatus,
    /// Whether `load()` reproduces what `save()` was given, bit for bit.
    ///
    /// Always `true` for a [`CheckpointStore`], which is exactly why
    /// [`PLGgufMerge`] is not one.
    pub round_trips: bool,
    /// File names this layout writes.
    pub segments: &'static [&'static str],
    /// Where the format came from.
    pub source: &'static str,
}

/// A checkpoint layout that round-trips.
///
/// Traits take no prefix (naming rule 2). Not generic over a backend: see
/// [`VLCheckpoint`]. Object-safe, because [`CheckpointRegistry`] hands out
/// `Box<dyn CheckpointStore>`.
pub trait CheckpointStore {
    /// Write `ckpt` into the directory at `dir`, creating it if needed.
    fn save(&self, dir: &Path, ckpt: &VLCheckpoint) -> Result<()>;

    /// Read back what [`CheckpointStore::save`] wrote.
    fn load(&self, dir: &Path) -> Result<VLCheckpoint>;

    /// Check a directory against what the caller expects, collecting every
    /// problem in one pass.
    fn validate(&self, dir: &Path, against: &VLManifest) -> Result<VLValidation>;

    /// This layout's capability record.
    fn capability(&self) -> &'static VLCheckpointFormat;
}

/// A one-way conversion to a deployment artifact.
///
/// Deliberately has no `load()`. Round-tripping a requantized GGUF back into a
/// resumable training checkpoint would lose precision with no error, which is
/// worse than not offering the method.
pub trait Exporter {
    /// Read an adapter checkpoint plus a base model, and write the merged,
    /// requantized artifact to `out`.
    fn export(&self, ckpt_dir: &Path, base_model: &Path, out: &Path) -> Result<()>;

    /// This pipeline's capability record. `round_trips` is always `false`.
    fn capability(&self) -> &'static VLCheckpointFormat;
}

type BuildFn = fn() -> Box<dyn CheckpointStore>;

/// Maps checkpoint-layout ids to their capability record and constructor.
///
/// Copies `glictus-caliburni/src/plugin.rs`'s `PluginRegistry`, the same way
/// [`crate::nn::AdapterRegistry`] and [`crate::optim::OptimizerRegistry`] do:
/// registration refuses on a duplicate id, `resolve` returns an `Option` and
/// `require` an error, and `with_builtins` preloads what the crate ships.
///
/// # Why there is no matching `ExporterRegistry`
///
/// A registry exists to resolve among alternatives, and there is exactly one
/// [`Exporter`]. Adding a second registry to hold one entry would also make
/// this the fourth copy of the same shape in the crate, which the checkpoint
/// skill says to extract rather than repeat. [`PLGgufMerge`] is constructed
/// directly and introspected through its `CAPABILITY`.
pub struct CheckpointRegistry {
    entries: BTreeMap<&'static str, (&'static VLCheckpointFormat, BuildFn)>,
}

impl Default for CheckpointRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CheckpointRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Every checkpoint layout this crate ships: one real, three stubs.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        // Built-in ids are distinct by construction, so these cannot collide.
        let builtins: [(&'static VLCheckpointFormat, BuildFn); 4] = [
            (lora_ckpt::CAPABILITY, || Box::new(CPLora)),
            (full::CAPABILITY, || Box::new(CPFull)),
            (sharded::CAPABILITY, || Box::new(CPSharded)),
            (incremental::CAPABILITY, || Box::new(CPIncremental)),
        ];
        for (cap, build) in builtins {
            r.register(cap, build)
                .expect("built-in checkpoint ids are unique");
        }
        r
    }

    /// Add a layout. Refuses a duplicate id.
    pub fn register(
        &mut self,
        capability: &'static VLCheckpointFormat,
        build: BuildFn,
    ) -> Result<()> {
        if self.entries.contains_key(capability.id) {
            return Err(GlTrainError::InvalidOp(format!(
                "checkpoint format '{}' is already registered; an id maps to exactly one \
                 implementation",
                capability.id
            )));
        }
        self.entries.insert(capability.id, (capability, build));
        Ok(())
    }

    /// Look up a capability record. `None` means nothing claims this id.
    pub fn resolve(&self, id: &str) -> Option<&'static VLCheckpointFormat> {
        self.entries.get(id).map(|(cap, _)| *cap)
    }

    /// Look up a capability record, erroring when absent.
    pub fn require(&self, id: &str) -> Result<&'static VLCheckpointFormat> {
        self.resolve(id).ok_or_else(|| {
            GlTrainError::InvalidOp(format!(
                "unknown checkpoint format '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })
    }

    /// Build a layout. A stub constructs and fails on `save`/`load`.
    pub fn build(&self, id: &str) -> Result<Box<dyn CheckpointStore>> {
        let (_, build) = self.entries.get(id).ok_or_else(|| {
            GlTrainError::InvalidOp(format!(
                "unknown checkpoint format '{id}'; registered: {}",
                self.ids().join(", ")
            ))
        })?;
        Ok(build())
    }

    /// Every registered id, sorted.
    pub fn ids(&self) -> Vec<&'static str> {
        self.entries.keys().copied().collect()
    }

    /// Ids that actually save and load.
    pub fn implemented_ids(&self) -> Vec<&'static str> {
        self.entries
            .iter()
            .filter(|(_, (cap, _))| cap.status.is_full())
            .map(|(id, _)| *id)
            .collect()
    }
}

/// Compare two shapes, and say whether they are a permutation of one another.
///
/// `[896, 4864]` and `[4864, 896]` have the same element count and would load
/// silently under a size-only check. gljax hit exactly this against real HF
/// weights, which is why its `bind_safetensors` names the transposed case in
/// the error rather than reporting a generic mismatch.
pub(crate) fn is_transposed(a: &[usize], b: &[usize]) -> bool {
    if a.len() != b.len() || a == b {
        return false;
    }
    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable();
    sb.sort_unstable();
    sa == sb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_register_all_four_checkpoint_layouts() {
        let r = CheckpointRegistry::with_builtins();
        assert_eq!(r.ids(), vec!["full", "incremental", "lora", "sharded"]);
    }

    /// Only CPLora saves and loads on M2.
    #[test]
    fn only_the_lora_layout_reports_itself_as_implemented() {
        let r = CheckpointRegistry::with_builtins();
        assert_eq!(r.implemented_ids(), vec!["lora"]);
    }

    #[test]
    fn checkpoint_registry_refuses_a_duplicate_id() {
        let mut r = CheckpointRegistry::with_builtins();
        assert!(r.register(lora_ckpt::CAPABILITY, || Box::new(CPLora)).is_err());
    }

    #[test]
    fn require_names_the_registered_ids_when_the_lookup_fails() {
        let r = CheckpointRegistry::with_builtins();
        assert!(r.resolve("lora").is_some());
        assert!(r.resolve("laura").is_none());
        let msg = r.require("laura").unwrap_err().to_string();
        assert!(msg.contains("lora"), "{msg}");
    }

    /// Every `CheckpointStore` round-trips by definition. This is the property
    /// that puts `PLGgufMerge` behind a different trait.
    #[test]
    fn every_checkpoint_store_claims_to_round_trip_and_the_exporter_does_not() {
        let r = CheckpointRegistry::with_builtins();
        for id in r.ids() {
            assert!(
                r.resolve(id).unwrap().round_trips,
                "'{id}' implements CheckpointStore but does not claim to round-trip"
            );
        }
        assert!(
            !export_gguf::CAPABILITY.round_trips,
            "a requantizing export cannot round-trip"
        );
    }

    /// The transposed-shape case is the one a size-only check misses, and it
    /// is a bug class this repo has actually hit.
    #[test]
    fn is_transposed_catches_a_permutation_but_not_an_unrelated_shape() {
        assert!(is_transposed(&[896, 4864], &[4864, 896]));
        assert!(!is_transposed(&[896, 4864], &[896, 4864]), "identical is not transposed");
        assert!(!is_transposed(&[2, 3], &[6]), "different rank");
        assert!(!is_transposed(&[2, 3], &[2, 4]), "different elements");
        assert!(is_transposed(&[2, 3, 4], &[4, 2, 3]), "any permutation counts");
    }

    #[test]
    fn segments_map_to_distinct_file_names() {
        assert_eq!(ENSegment::Adapter.file_name(), "adapter.safetensors");
        assert_eq!(ENSegment::Optimizer.file_name(), "optimizer.safetensors");
        assert_ne!(
            ENSegment::Adapter.file_name(),
            ENSegment::Optimizer.file_name()
        );
    }
}
