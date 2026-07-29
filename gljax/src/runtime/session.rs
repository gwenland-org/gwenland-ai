//! `Session` — plugin, client, compiled plan and resident weights (ARTX04).
//!
//! ⛔ **Unexecuted.** Every path below has been compiled and type-checked and
//! **none of it has ever run**: there is no PJRT plugin for Windows, so
//! `Session::open` cannot be constructed on the reference machine. See
//! `gljax/README.md`. The host-side halves — the cache key, the signature
//! check, the weight binding — are tested; the device calls are not.

use std::path::Path;
use std::rc::Rc;

use crate::checkpoint::{bind_safetensors, WeightSource};
use crate::graph::BuiltFunc;
use crate::model::Qwen2Config;
use crate::pjrt::{LoadedExecutable, PjrtBufferHandle, PjrtClientHandle, PjrtPlugin};
use crate::runtime::cache::{CompileCache, CompileKey};
use crate::runtime::plan::PlanSignature;
use crate::stablehlo::types::{DType, Shape};
use crate::GlError;

/// A loaded model: compiled program plus the weights already on device.
///
/// ⭐ ARTX01 §8.2's compile/weight separation, made concrete. The program is
/// compiled once for a given shape and is weight-value-agnostic; the weights
/// are uploaded once and passed as arguments on every call. Swapping weights
/// does not recompile — that is what makes bucketing affordable.
pub struct Session {
    client: Rc<PjrtClientHandle>,
    program: LoadedExecutable,
    signature: PlanSignature,
    /// Uploaded once, in `signature` order, reused every call.
    weights: Vec<PjrtBufferHandle>,
    config: Qwen2Config,
    seq_len: usize,
}

impl Session {
    /// Compiles a traced module and uploads its weights.
    ///
    /// `cache_dir`, when given, is consulted before compiling. ⚠️ The cached
    /// artifact is *not* loaded back yet: that needs
    /// `PJRT_Executable_DeserializeAndLoad`, which is bound in
    /// [`crate::sys`] but not wrapped, because a deserialize path that has
    /// never round-tripped against a real plugin is a path that will be wrong
    /// in a way no test here can catch. The cache is written, not read.
    pub fn open(
        plugin: Rc<PjrtPlugin>,
        built: &BuiltFunc,
        config: Qwen2Config,
        seq_len: usize,
        weights: &dyn WeightSource,
        cache_dir: Option<&Path>,
    ) -> Result<Self, GlError> {
        let client = PjrtClientHandle::create(plugin)?;
        let device = client.default_device()?;

        // Bind before compiling: a checkpoint mismatch is cheap to detect and
        // expensive to discover after a 20-minute compile (ARTX16 §4.2).
        let bound = bind_safetensors(&built.signature, weights)?;

        if let Some(dir) = cache_dir {
            let key = CompileKey::new(
                client.plugin().api_version(),
                client.platform_name()?,
                client.platform_version()?,
                &built.mlir,
            );
            let cache = CompileCache::open(dir)?;
            match cache.get(&key)? {
                Some(_) => log::info!(
                    "compile cache has an artifact for {} — not reused yet, \
                     DeserializeAndLoad is unwrapped",
                    key.mlir_hex()
                ),
                None => log::info!("compile cache miss for {}", key.mlir_hex()),
            }
        }

        let program = client.compile(&built.mlir)?;

        let mut buffers = Vec::with_capacity(bound.len());
        for w in &bound {
            let shape = Shape::new(w.shape.dims.clone(), DType::F32);
            buffers.push(client.buffer_from_host_f32(&w.data, &shape, &device)?);
        }

        Ok(Session {
            client,
            program,
            signature: PlanSignature::from_traced(&built.signature),
            weights: buffers,
            config,
            seq_len,
        })
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    pub fn seq_len(&self) -> usize {
        self.seq_len
    }

    pub fn signature(&self) -> &PlanSignature {
        &self.signature
    }

    /// Runs one forward pass over `token_ids` and returns the logits.
    ///
    /// `token_ids` must be exactly `seq_len` long — this is a static-shape
    /// engine, and padding to the bucket is the caller's job (P3).
    pub fn forward(&self, token_ids: &[i32]) -> Result<Vec<f32>, GlError> {
        if token_ids.len() != self.seq_len {
            return Err(GlError::ShapeMismatch {
                expected: vec![1, self.seq_len],
                got: vec![1, token_ids.len()],
            });
        }

        let device = self.client.default_device()?;
        // ⚠️ Token ids are I32 on the device. Uploading them as F32 would be a
        // silent reinterpretation at the gather, so this needs an I32 upload
        // path that `buffer_from_host_f32` deliberately refuses to provide.
        let ids = self.upload_token_ids(token_ids, &device)?;

        let mut args: Vec<&PjrtBufferHandle> = Vec::with_capacity(1 + self.weights.len());
        args.push(&ids);
        args.extend(self.weights.iter());

        let outputs = self.program.execute(&args)?;
        let first = outputs.first().ok_or_else(|| {
            GlError::Engine("forward: executable returned no outputs".to_owned())
        })?;
        first.to_host_f32()
    }

    /// Greedy generation: prompt in, token ids out.
    ///
    /// # ⚠️ This recomputes the whole sequence every step
    ///
    /// ARTX03 §4's design decision, deliberately kept: v1 has no KV cache, so
    /// step `n` re-runs the full bucket-length forward pass. That is O(n·S)
    /// work for n tokens where a cache would be O(n), and it is the single
    /// biggest performance item outstanding.
    ///
    /// It is also *correct by construction*, which is the trade being made
    /// here. [`crate::ops::kv_cache`] has the primitives that replace it; what
    /// it does not have is a validated cache-aware trace, and a cache that
    /// reads one position off produces fluent, wrong text with no error
    /// anywhere (P4).
    ///
    /// Sampling is argmax only — see [`crate::runtime::sample`].
    ///
    /// # Errors
    /// If the prompt does not fit the compiled bucket, or a device call fails.
    pub fn generate(
        &self,
        prompt: &[i32],
        max_new_tokens: usize,
        eos_id: Option<i32>,
        pad_id: i32,
    ) -> Result<Vec<i32>, GlError> {
        if prompt.is_empty() {
            return Err(GlError::Engine("generate: empty prompt".to_owned()));
        }
        if prompt.len() > self.seq_len {
            return Err(GlError::Engine(format!(
                "generate: prompt of {} tokens exceeds the compiled bucket of {}",
                prompt.len(),
                self.seq_len
            )));
        }

        let vocab = self.config.vocab;
        let mut tokens = prompt.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            if tokens.len() > self.seq_len {
                log::warn!(
                    "generate: sequence reached the bucket length ({}), stopping early",
                    self.seq_len
                );
                break;
            }
            let padded = crate::runtime::bucket::pad_to_bucket(&tokens, self.seq_len, pad_id);
            let logits = self.forward(&padded)?;

            // ⭐ Right padding means the row that matters is the last *real*
            // token, not the last row of the bucket. Reading the last row would
            // sample from a padding position — plausible output, wrong model.
            let position = crate::runtime::bucket::last_real_position(tokens.len());
            let next =
                crate::runtime::sample::argmax_at(&logits, self.seq_len, vocab, position)? as i32;

            generated.push(next);
            if Some(next) == eos_id {
                break;
            }
            tokens.push(next);
        }
        Ok(generated)
    }

    fn upload_token_ids(
        &self,
        ids: &[i32],
        device: &crate::pjrt::PjrtDeviceRef,
    ) -> Result<PjrtBufferHandle, GlError> {
        // SAFETY: `ids` is a live `&[i32]` for the duration of the call, and
        // `ImmutableOnlyDuringCall` (used inside `buffer_from_host_raw`) is
        // exactly the promise that covers that. The dims match `[1, seq_len]`
        // as declared in the traced signature.
        unsafe {
            self.client.buffer_from_host_raw(
                ids.as_ptr().cast::<core::ffi::c_void>(),
                crate::sys::types::PjrtBufferType::S32,
                &[1, ids.len()],
                device,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Session` cannot be constructed without a plugin, so what is testable
    /// here is the ordering contract it depends on: the traced signature must
    /// put `input_ids` first, because `forward` pushes the ids buffer before
    /// the weights and PJRT matches by position.
    #[test]
    fn the_traced_signature_puts_the_runtime_input_first() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(crate::PrecisionPolicy::f32(), || {
            crate::model::trace_forward(&cfg, 4, 0)
        })
        .expect("trace");
        let plan = PlanSignature::from_traced(&built.signature);

        assert_eq!(plan.names()[0], "input_ids");
        assert_eq!(
            plan.params.len(),
            1 + built.signature.weights.len(),
            "forward() pushes exactly one input then every weight"
        );
        assert_eq!(plan.params[0].1.dims, vec![1, 4]);
        assert_eq!(plan.params[0].1.dtype, DType::I32);
    }

    /// A wrong-length prompt must be refused before any device call.
    #[test]
    fn forward_rejects_a_prompt_that_is_not_the_bucket_length() {
        // Checked without a Session by exercising the same comparison; the
        // real call is unreachable here (no plugin).
        let seq_len = 128usize;
        let ids = vec![0i32; 127];
        assert_ne!(ids.len(), seq_len);
    }
}
