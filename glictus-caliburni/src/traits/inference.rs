use crate::error::GllmResult;
use crate::types::execution::Device;
use crate::types::package::GllmPackageMeta;

/// Core inference trait — implemented by CPU/CUDA/Vulkan/Metal backends.
///
/// This is the *token-level* API: tokens in, logits out. It is distinct from
/// [`GllmRuntime`](crate::runtime::GllmRuntime) (ARTX05), which orchestrates
/// layer-sequential execution (map → execute → unmap) one forward pass at a
/// time and knows nothing about tokens or sampling. A backend may eventually
/// implement this trait *on top of* a `GllmRuntime`, once tokenizer packaging
/// (ARTX1 OQ3) is settled.
///
/// Philosophy: *"Correctness first. Load what you need. Execute what's valid."*
///
/// ## Contract
///
/// Callers must observe this ordering:
/// 1. [`load_package`](Self::load_package) — validates the manifest and
///    per-file checksums. Fails loud on a corrupt package (ARTX01 principle
///    "Fail Fast, Fail Loud"): a checksum mismatch surfaces here, never
///    later as garbage logits.
/// 2. [`infer`](Self::infer) / [`stream`](Self::stream) — only valid once a
///    package is loaded; implementations return
///    [`GllmError::MissingExecutionUnit`](crate::error::GllmError::MissingExecutionUnit)
///    otherwise.
/// 3. [`unload`](Self::unload) — frees device memory. Safe to call when
///    nothing is loaded, and safe to call twice.
///
/// ## Object safety
///
/// This trait MUST remain object-safe — runtimes are held as
/// `Box<dyn GllmInference>`. That is why [`stream`](Self::stream) takes
/// `&mut dyn FnMut` rather than a generic `F: FnMut`. Keep that pattern for
/// any new callback-taking method.
pub trait GllmInference: Send + Sync {
    /// Load a GLLM package — validate checksums, parse manifest.
    fn load_package(&mut self, package: &GllmPackageMeta) -> GllmResult<()>;

    /// Execute inference on the loaded package, returning logits.
    fn infer(&mut self, input_ids: &[u32]) -> GllmResult<Vec<f32>>;

    /// Stream tokens one by one.
    ///
    /// The callback returns `false` to request that generation stop early.
    ///
    /// The default implementation runs [`infer`](Self::infer) and emits the
    /// single greedy (argmax) token from the returned logits — enough for a
    /// backend to be usable before it implements true streaming, and nothing
    /// more. It is deliberately not a sampler and not a generation loop.
    /// Do not make this default smarter to paper over a backend's latency —
    /// implement `stream` in that backend instead.
    fn stream(
        &mut self,
        input_ids: &[u32],
        callback: &mut dyn FnMut(u32) -> bool,
    ) -> GllmResult<()> {
        let logits = self.infer(input_ids)?;
        let argmax = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32);
        if let Some(token) = argmax {
            callback(token);
        }
        Ok(())
    }

    /// Unload the package and free resources.
    fn unload(&mut self);

    /// Devices this runtime can execute on.
    fn supported_devices(&self) -> &[Device];

    /// Runtime name (e.g. `"glproc-cpu"`, `"glcuda-nvidia"`).
    fn name(&self) -> &str;
}
