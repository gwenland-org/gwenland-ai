//! Stummañ Pik: sharded checkpoint. **STUB, with the real layout.**
//!
//! # A layout over the same manifest, not a parallel format
//!
//! HuggingFace's convention, which this follows: numbered shard files plus an
//! index that maps every tensor name to the file holding it.
//!
//! ```text
//! checkpoint_000500/
//!   manifest.json
//!   model-00001-of-00006.safetensors
//!   ...
//!   model.safetensors.index.json
//! ```
//!
//! ```json
//! { "metadata": { "total_size": 28966928384 },
//!   "weight_map": { "lora_a": "model-00001-of-00006.safetensors" } }
//! ```
//!
//! Each shard reuses [`crate::checkpoint::CPLora`]'s tensor-entry schema
//! verbatim. Only the index file is new.
//!
//! # The partial load is the whole point
//!
//! `load` reads the index first, then opens only the shards the requested
//! parameters actually live in. A stub that concatenated every shard on load
//! would produce correct tensors while defeating the single reason this type
//! exists, which is why this one refuses instead.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use std::path::Path;

use super::manifest::{VLManifest, VLValidation, MANIFEST_FILE};
use super::{CheckpointStore, VLCheckpoint, VLCheckpointFormat};

const STUB_REASON: &str =
    "needs an index writer and a name-to-shard partial loader; concatenating every shard on load \
     would defeat the partial-loading property that is the only reason to shard";
const STUB_MILESTONE: &str = "M3";

/// Default maximum bytes per shard.
///
/// HuggingFace's default, and configurable there. Recorded so a later
/// implementation does not pick a different one by accident and produce
/// checkpoints other tools split differently.
pub const DEFAULT_MAX_SHARD_BYTES: u64 = 5 * 1024 * 1024 * 1024;

/// The index file name, alongside the shards.
pub const INDEX_FILE: &str = "model.safetensors.index.json";

/// CPSharded's capability record.
pub static CAPABILITY: &VLCheckpointFormat = &VLCheckpointFormat {
    id: "sharded",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    round_trips: true,
    segments: &[
        MANIFEST_FILE,
        INDEX_FILE,
        "model-NNNNN-of-NNNNN.safetensors",
    ],
    source: "HuggingFace sharded-checkpoint convention",
};

/// Multi-file checkpoint. Constructs and introspects; refuses to save or load.
pub struct CPSharded;

/// The shard file name for `index` of `total`, in HuggingFace's format.
pub fn shard_name(index: usize, total: usize) -> String {
    format!("model-{index:05}-of-{total:05}.safetensors")
}

fn unsupported<T>() -> Result<T> {
    Err(GlTrainError::Unsupported {
        skill: "sharded",
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    })
}

impl CheckpointStore for CPSharded {
    fn save(&self, _dir: &Path, _ckpt: &VLCheckpoint) -> Result<()> {
        unsupported()
    }

    fn load(&self, _dir: &Path) -> Result<VLCheckpoint> {
        unsupported()
    }

    fn validate(&self, _dir: &Path, _against: &VLManifest) -> Result<VLValidation> {
        unsupported()
    }

    fn capability(&self) -> &'static VLCheckpointFormat {
        CAPABILITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpsharded_save_and_load_return_unsupported() {
        let dir = std::path::Path::new("unused");
        assert!(matches!(
            CPSharded.load(dir),
            Err(GlTrainError::Unsupported {
                skill: "sharded",
                ..
            })
        ));
    }

    /// The names are five-digit, one-based, and include the total. Getting the
    /// padding wrong produces files no other tool finds.
    #[test]
    fn shard_names_follow_the_huggingface_convention() {
        assert_eq!(shard_name(1, 6), "model-00001-of-00006.safetensors");
        assert_eq!(shard_name(12, 12), "model-00012-of-00012.safetensors");
    }

    /// A stub that loaded every shard would be correct and useless. Recording
    /// that in the reason is what stops the next wave shipping it.
    #[test]
    fn cpsharded_names_partial_loading_as_the_property_at_stake() {
        let ENSkillStatus::Stub { reason, .. } = CAPABILITY.status else {
            panic!("CPSharded must be a stub");
        };
        assert!(reason.contains("partial"), "{reason}");
    }

    #[test]
    fn the_default_shard_size_is_five_gigabytes() {
        assert_eq!(DEFAULT_MAX_SHARD_BYTES, 5_368_709_120);
    }
}
