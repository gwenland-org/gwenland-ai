use std::path::Path;

use crate::error::GllmResult;
use crate::types::package::GllmPackageMeta;

/// Converter trait — transforms a source format into a GLLM package.
///
/// ## Contract
///
/// [`convert`](Self::convert) writes a complete package into `output_dir`
/// (manifest + shared component + one file per layer) and returns the parsed
/// result. It is responsible for computing the per-file SHA-256 checksums the
/// manifest records — a package that ships without them defeats the format's
/// integrity guarantee.
///
/// [`validate`](Self::validate) re-reads what was written and verifies it:
/// checksums match, every layer referenced by the manifest exists, and the
/// layer count agrees with the model metadata. Converters are expected to call
/// it on their own output rather than trusting the write path.
pub trait GllmConverter {
    /// Source format name (e.g. `"gguf"`, `"safetensors"`).
    fn source_format(&self) -> &str;

    /// Convert the source model into a GLLM package under `output_dir`.
    fn convert(&self, source: &Path, output_dir: &Path) -> GllmResult<GllmPackageMeta>;

    /// Validate a converted package.
    fn validate(&self, package: &GllmPackageMeta) -> GllmResult<()>;
}
