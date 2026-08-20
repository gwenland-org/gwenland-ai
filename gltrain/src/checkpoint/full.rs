//! Stummañ Pik: full-model checkpoint. **STUB.**
//!
//! # The format is not the blocker
//!
//! `CPFull` writes exactly [`crate::checkpoint::CPLora`]'s segments over every
//! parameter of a [`crate::nn::Module`] instead of only an adapter's two. The
//! tensor-entry schema, the manifest, and the safetensors writer are all
//! reused unchanged.
//!
//! What is missing is a **model tree to enumerate**. "All parameters" needs a
//! `Module` that owns a base model, and M2 has `ABLinear` plus adapters over a
//! base weight supplied from outside. There is nothing yet whose
//! `parameters()` returns a whole network, so `save` would have nothing to
//! walk.
//!
//! Naming a format problem here would be wrong and would send the next wave to
//! rewrite serialization that is already correct.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use std::path::Path;

use super::manifest::{VLManifest, VLValidation, MANIFEST_FILE};
use super::{CheckpointStore, VLCheckpoint, VLCheckpointFormat};

const STUB_REASON: &str =
    "needs a full Module tree to enumerate; the segment format itself is CPLora's, unchanged";
const STUB_MILESTONE: &str = "M3";

/// CPFull's capability record.
pub static CAPABILITY: &VLCheckpointFormat = &VLCheckpointFormat {
    id: "full",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    round_trips: true,
    segments: &[MANIFEST_FILE, "model.safetensors", "optimizer.safetensors"],
    source: "same segment split as CPLora (M2_RESEARCH.md 7-E)",
};

/// Whole-model checkpoint. Constructs and introspects; refuses to save or load.
pub struct CPFull;

fn unsupported<T>() -> Result<T> {
    Err(GlTrainError::Unsupported {
        skill: "full",
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    })
}

impl CheckpointStore for CPFull {
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
    fn cpfull_save_and_load_return_unsupported() {
        let dir = std::path::Path::new("unused");
        let m = VLManifest::for_lora(
            &crate::nn::adapter::lora::VLLoraConfig {
                r: 1,
                alpha: 1.0,
                rslora: false,
                d_in: 1,
                d_out: 1,
            },
            0,
        );
        assert!(matches!(
            CPFull.save(dir, &VLCheckpoint::new(m.clone())),
            Err(GlTrainError::Unsupported { skill: "full", .. })
        ));
        assert!(CPFull.load(dir).is_err());
        assert!(CPFull.validate(dir, &m).is_err());
    }

    /// The blocker is the model tree, not the file format. Saying otherwise
    /// would send the next wave to rewrite serialization that already works.
    #[test]
    fn cpfull_names_the_module_tree_as_its_blocker_not_the_format() {
        let ENSkillStatus::Stub { reason, .. } = CAPABILITY.status else {
            panic!("CPFull must be a stub");
        };
        assert!(reason.contains("Module tree"), "{reason}");
        assert!(reason.contains("CPLora"), "{reason}");
    }

    #[test]
    fn cpfull_still_claims_to_round_trip_because_it_is_a_checkpoint_store() {
        assert!(CAPABILITY.round_trips);
    }
}
