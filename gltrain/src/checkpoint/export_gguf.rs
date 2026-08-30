//! Stummañ Pik: merge an adapter into a GGUF base model. **STUB, [`Exporter`].**
//!
//! # Why this is `PL` and not `CP`
//!
//! It is not a checkpoint. It reads an adapter plus a base model, folds the
//! delta in, **requantizes**, and writes a deployment artifact. That is one-way
//! and lossy by construction: nothing can resume training from the output.
//! A `CP` name would imply a `load()` that cannot be implemented, so this
//! implements [`Exporter`], which deliberately has none. Round-tripping a
//! requantized GGUF back into a training checkpoint would lose precision with
//! no error at all, which is worse than not offering the method.
//!
//! # The pipeline, in order
//!
//! 1. Load the adapter checkpoint with
//!    [`crate::checkpoint::CheckpointStore::load`] on
//!    [`crate::checkpoint::CPLora`].
//! 2. Load the base model. If it is GGUF, use `glcore::format::gguf::GgufFile`
//!    and `.dequantize(info)` per tensor. **Reuse that reader; do not reparse
//!    the format.**
//! 3. Per adapted site, `merged = W0 + scale*(A @ B)`. That is exactly
//!    [`crate::nn::Adapter::merge_into`]'s math against a dense tensor. Call
//!    it rather than reimplementing it.
//! 4. **Requantize** each merged tensor to the target GGUF type, through
//!    glproc's existing quant kernels. Read `gguf-skills/quantization-types.md`
//!    and `gguf-skills/dequant-path.md` first, and check
//!    `cpu-skills/rejected-optimizations.md` before assuming any quant type is
//!    fast on this tier: native Q4_K was built and measured **33% slower**,
//!    compute-bound rather than memory-bound. That lesson is already paid for.
//! 5. Write the merged GGUF.
//!
//! # Step 5 is the actual blocker
//!
//! `glcore/src/format/gguf.rs` is **read-only**. There is no GGUF writer
//! anywhere in this repository, and that is genuinely new work rather than a
//! training concern: header (magic `0x46554747`, version 3, tensor count, kv
//! count), the metadata KV section, tensor info (name at most **64 bytes**, at
//! most 4 dims, `ggml_type`, offset), then a data section aligned to
//! `general.alignment` (default **32**, and it must be a multiple of 8), with
//! `0x00` padding between tensors.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::ENSkillStatus;
use std::path::Path;

use super::{Exporter, VLCheckpointFormat};

const STUB_REASON: &str =
    "there is no GGUF writer anywhere in the repo (glcore's gguf.rs is read-only), and the merged \
     tensors still need requantizing through glproc's quant kernels";
const STUB_MILESTONE: &str = "M4";

/// GGUF magic, `"GGUF"` little-endian.
pub const GGUF_MAGIC: u32 = 0x4655_4747;

/// GGUF version this pipeline would write.
pub const GGUF_VERSION: u32 = 3;

/// Default `general.alignment` when a base model does not state one.
///
/// 32, not 0 and not unaligned. Verified against the ggml spec rather than
/// inferred, and it must always be a multiple of 8.
pub const DEFAULT_ALIGNMENT: u32 = 32;

/// Maximum bytes in a GGUF tensor name.
pub const MAX_TENSOR_NAME_BYTES: usize = 64;

/// Maximum dimensions in a GGUF tensor.
pub const MAX_TENSOR_DIMS: usize = 4;

/// PLGgufMerge's capability record.
///
/// `round_trips: false` is the field that puts this behind [`Exporter`] rather
/// than [`crate::checkpoint::CheckpointStore`].
pub static CAPABILITY: &VLCheckpointFormat = &VLCheckpointFormat {
    id: "gguf_merge",
    status: ENSkillStatus::Stub {
        reason: STUB_REASON,
        milestone: STUB_MILESTONE,
    },
    round_trips: false,
    segments: &["merged.gguf"],
    source: "ggml GGUF spec v3",
};

/// One-way export: adapter plus base model, merged and requantized to GGUF.
pub struct PLGgufMerge;

impl Exporter for PLGgufMerge {
    fn export(&self, _ckpt_dir: &Path, _base_model: &Path, _out: &Path) -> Result<()> {
        Err(GlTrainError::Unsupported {
            skill: "gguf_merge",
            reason: STUB_REASON,
            milestone: STUB_MILESTONE,
        })
    }

    fn capability(&self) -> &'static VLCheckpointFormat {
        CAPABILITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gguf_merge_export_returns_unsupported() {
        let p = std::path::Path::new("unused");
        assert!(matches!(
            PLGgufMerge.export(p, p, p),
            Err(GlTrainError::Unsupported {
                skill: "gguf_merge",
                ..
            })
        ));
    }

    /// The property that decides which trait this belongs to. A requantizing
    /// export cannot reproduce its input, so it cannot be a CheckpointStore.
    #[test]
    fn gguf_merge_declares_that_it_does_not_round_trip() {
        assert!(!CAPABILITY.round_trips);
    }

    /// The blocker is a missing GGUF writer, not anything about training. A
    /// reason naming the wrong thing would misdirect the wave that picks it up.
    #[test]
    fn gguf_merge_names_the_absent_gguf_writer_as_its_blocker() {
        let ENSkillStatus::Stub { reason, .. } = CAPABILITY.status else {
            panic!("PLGgufMerge must be a stub");
        };
        assert!(reason.contains("GGUF writer"), "{reason}");
        assert!(reason.contains("read-only"), "{reason}");
    }

    /// Verified against the ggml spec rather than guessed. An alignment of 0,
    /// or one that is not a multiple of 8, produces a file ggml rejects.
    #[test]
    fn the_gguf_constants_match_the_spec() {
        assert_eq!(GGUF_MAGIC, 0x4655_4747);
        assert_eq!(&GGUF_MAGIC.to_le_bytes(), b"GGUF");
        assert_eq!(GGUF_VERSION, 3);
        assert_eq!(DEFAULT_ALIGNMENT, 32);
        assert_eq!(DEFAULT_ALIGNMENT % 8, 0, "alignment must be a multiple of 8");
        assert_eq!(MAX_TENSOR_NAME_BYTES, 64);
        assert_eq!(MAX_TENSOR_DIMS, 4);
    }

    /// `PLGgufMerge` must not appear in the CheckpointRegistry: that registry
    /// hands out `Box<dyn CheckpointStore>`, and this type deliberately is not
    /// one. A compile error would catch it, but the id could still collide.
    #[test]
    fn gguf_merge_is_not_registered_as_a_checkpoint_store() {
        let r = crate::checkpoint::CheckpointRegistry::with_builtins();
        assert!(
            r.resolve("gguf_merge").is_none(),
            "an exporter must not be resolvable as a checkpoint store"
        );
    }
}
