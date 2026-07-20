use crate::error::GllmResult;
use crate::manifest::{DType, ExtensionUri, TensorEntry};
use crate::types::execution::Device;

/// Describes one layer-type extension (ARTX8).
///
/// A plugin says **what a layer type is** — which tensors it requires, how
/// they must be shaped, how much memory it needs — and nothing about how to
/// compute it.
///
/// ## Why there is no `execute`
///
/// ARTX8's sketch gave plugins an `execute(inputs, outputs)` method. That is
/// deliberately absent here: this crate orchestrates and validates, and every
/// tensor operation goes through
/// [`ExecutionBackend`](crate::runtime::ExecutionBackend) into glproc/glcuda
/// (ARTX05 AD-02). A plugin that could execute would be a second, competing
/// compute path inside a crate that is supposed to have none — and kernels
/// belong to the engines, which tune and validate them per hardware model.
///
/// So a plugin's job is to answer, before any execution begins: *is this
/// layer file structurally valid for the type it claims to be?*
///
/// ## Object safety
///
/// Must remain object-safe: plugins live in a
/// [`PluginRegistry`](crate::plugin::PluginRegistry) as
/// `Box<dyn LayerPlugin>` and are resolved once per layer at load time.
/// Resolution is a load-time cost, never a per-token one — do not dispatch
/// through `dyn` inside a decode loop.
pub trait LayerPlugin: Send + Sync {
    /// This plugin's extension URI, e.g. `gllm:transformer/moe@v1`.
    ///
    /// Two plugins claiming one URI is a registration error, not a race —
    /// see [`PluginRegistry::register`](crate::plugin::PluginRegistry::register).
    fn uri(&self) -> &ExtensionUri;

    /// Tensor names a layer of this type must contain.
    ///
    /// Names are the layer-local ones written by the converter
    /// (`attn_q.weight`, not `blk.0.attn_q.weight`).
    fn required_tensors(&self) -> &[&'static str];

    /// Check a layer's tensor index against this type's layout.
    ///
    /// The default implementation reports every missing required tensor at
    /// once, rather than failing on the first — a converter bug usually drops
    /// a whole family of tensors, and one name per run makes that tedious to
    /// find. Override to add type-specific checks (expert counts, latent
    /// dimensions), calling this first.
    fn validate_layout(&self, index: &[TensorEntry]) -> GllmResult<()> {
        let missing: Vec<&str> = self
            .required_tensors()
            .iter()
            .copied()
            .filter(|name| !index.iter().any(|t| t.name == *name))
            .collect();

        if missing.is_empty() {
            Ok(())
        } else {
            Err(crate::error::GllmError::TensorEntryInvalid(format!(
                "{}: missing required tensor(s): {}",
                self.uri(),
                missing.join(", ")
            )))
        }
    }

    /// Bytes this layer occupies once resident on `device`.
    ///
    /// The default sums the tensor index, which is correct for any layer
    /// whose weights are simply mapped. Override for types that need scratch
    /// beyond their weights — Mamba's recurrent state, for instance.
    fn memory_requirement_bytes(&self, index: &[TensorEntry], _device: &Device) -> u64 {
        index.iter().map(|t| t.size).sum()
    }

    /// Whether this plugin can handle tensors of `dtype`.
    ///
    /// A `false` is a normal outcome — the loader reports the layer as
    /// unsupported and the caller picks another backend, rather than failing
    /// the package.
    fn supports_dtype(&self, dtype: DType) -> bool;
}
