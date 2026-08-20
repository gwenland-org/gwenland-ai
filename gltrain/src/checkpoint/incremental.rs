//! Stummañ Pik: incremental checkpoint. **STUB, and low-value here. Say so.**
//!
//! # The honest reason this is last
//!
//! Check-N-Run (NSDI 2022, arXiv:2010.08679) saves space because
//! recommendation models update their embedding tables **sparsely** between
//! checkpoints, so a presence-delta ("which tensors changed") is mostly empty.
//! The paper itself notes the limited applicability outside that shape.
//!
//! **Every LoRA adapter parameter receives a gradient every step.** A
//! presence-delta over this crate's primary workload saves nothing at all.
//! Only a *value*-delta plus compression would help, which is substantially
//! more work for a smaller win than the other three layouts offer.
//!
//! A capability record saying only "not implemented yet" would be true and
//! misleading: it would omit that the win is near-zero here, and someone would
//! spend a wave finding that out.
//!
//! # If it is built anyway, this is the honest shape
//!
//! ```text
//! checkpoint_000520_delta/
//!   manifest.json      base_checkpoint: "checkpoint_000500", chain_depth: 1
//!   delta.safetensors  full per-element VALUE deltas, named identically to
//!                      the base's tensors
//! ```
//!
//! With a **chain-depth cap**, because reconstruction cost grows with chain
//! length, and a documented recovery rule: past the cap, or on any broken
//! link, fall back to the nearest full checkpoint rather than attempting a
//! partial reconstruction.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use std::path::Path;

use super::manifest::{VLManifest, VLValidation, MANIFEST_FILE};
use super::{CheckpointStore, VLCheckpoint, VLCheckpointFormat};

const STUB_REASON: &str =
    "near-zero benefit for dense LoRA gradients (Check-N-Run's saving is embedding-table \
     sparsity, absent here); every adapter parameter changes every step, so only a value-delta \
     plus compression would help. Implement only if a real workload needs it";
const STUB_MILESTONE: &str = "unscheduled";

/// How many deltas may chain before a full checkpoint is required.
///
/// Reconstruction walks the chain, so cost grows with its length, and every
/// link is another file that can be lost.
pub const DEFAULT_MAX_CHAIN_DEPTH: usize = 8;

/// CPIncremental's capability record.
pub static CAPABILITY: &VLCheckpointFormat = &VLCheckpointFormat {
    id: "incremental",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    round_trips: true,
    segments: &[MANIFEST_FILE, "delta.safetensors"],
    source: "Check-N-Run, NSDI 2022, arXiv:2010.08679",
};

/// Delta checkpoint. Constructs and introspects; refuses to save or load.
pub struct CPIncremental;

fn unsupported<T>() -> Result<T> {
    Err(GlTrainError::Unsupported {
        skill: "incremental",
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    })
}

impl CheckpointStore for CPIncremental {
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
    fn cpincremental_save_and_load_return_unsupported() {
        let dir = std::path::Path::new("unused");
        assert!(matches!(
            CPIncremental.load(dir),
            Err(GlTrainError::Unsupported {
                skill: "incremental",
                ..
            })
        ));
    }

    /// "Not implemented yet" would be true and misleading. The record has to
    /// carry the measured reason, or someone spends a wave rediscovering it.
    #[test]
    fn cpincremental_records_that_the_win_is_near_zero_here_not_merely_that_it_is_unbuilt() {
        let ENSkillStatus::Stub {
            reason, milestone, ..
        } = CAPABILITY.status
        else {
            panic!("CPIncremental must be a stub");
        };
        assert!(reason.contains("near-zero"), "{reason}");
        assert!(reason.contains("sparsity"), "{reason}");
        assert!(
            !reason.starts_with("not implemented"),
            "the reason must say why it is not worth building: {reason}"
        );
        assert_eq!(
            milestone, "unscheduled",
            "no milestone owns this; claiming one would imply it is planned"
        );
    }

    /// Reconstruction walks every link, and each one is another file that can
    /// go missing, so the cap has to be small rather than merely present: an
    /// effectively uncapped chain is how a delta scheme turns one lost file
    /// into an unrecoverable run.
    #[test]
    fn the_chain_depth_cap_is_small() {
        assert_eq!(DEFAULT_MAX_CHAIN_DEPTH, 8);
    }
}
