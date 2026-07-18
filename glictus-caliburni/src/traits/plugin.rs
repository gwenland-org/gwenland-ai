use crate::error::GllmResult;
use crate::types::execution::Device;
use crate::types::tensor::{DType, TensorEntry};

/// Plugin trait — implemented per layer-type extension.
///
/// Registration is URI-based: a plugin declares
/// `"gllm:transformer/standard@v1"` via [`uri`](Self::uri), and a manifest's
/// [`LayerEntry::layer_type`](crate::types::manifest::LayerEntry::layer_type)
/// selects it. An unmatched URI is
/// [`GllmError::UnknownExtension`](crate::error::GllmError::UnknownExtension) —
/// never a silent no-op.
///
/// ## Object safety
///
/// Must remain object-safe: plugins live in a registry as
/// `Box<dyn LayerPlugin>`, resolved once per layer at load time. Resolution is
/// a load-time cost, not a per-token one — never dispatch through `dyn` inside
/// a decode loop.
pub trait LayerPlugin: Send + Sync {
    /// Parse the layer's tensor index and return the tensors this plugin
    /// requires, erroring if a mandatory tensor is absent.
    fn parse_tensors(&self, index: &[TensorEntry]) -> GllmResult<Vec<TensorEntry>>;

    /// Memory this layer needs once resident on `device`.
    fn memory_requirement_bytes(&self, tensors: &[TensorEntry], device: &Device) -> u64;

    /// Execute the forward pass.
    ///
    /// `inputs` is the activation from the previous layer; `outputs` receives
    /// the activation for the next one. Implementations should treat
    /// `outputs` as scratch to be overwritten, not appended to.
    fn execute(
        &self,
        inputs: &[f32],
        outputs: &mut Vec<f32>,
        tensors: &[TensorEntry],
    ) -> GllmResult<()>;

    /// Whether this plugin can handle tensors of `dtype`.
    ///
    /// A `false` here is a normal outcome — the loader reports the layer as
    /// unsupported and the caller picks another backend.
    fn supports_dtype(&self, dtype: DType) -> bool;

    /// This plugin's extension URI.
    fn uri(&self) -> &str;
}
