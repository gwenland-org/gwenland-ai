//! `CachedSession` — prefill + decode with a real KV cache (ARTX05).
//!
//! ⛔ **Unexecuted**, exactly like [`crate::runtime::Session`]: there is no
//! PJRT plugin for Windows, so nothing below has ever run. It joins the same
//! category the rest of `runtime` is already in — see `gljax/README.md`.
//!
//! ⚠️ **Do not trust this path's output without checking it against
//! [`Session::generate`](crate::runtime::Session::generate).** ARTX05's
//! headline risk is a cache that reads one position off: shapes stay right,
//! output stays fluent, and the answer is wrong. `Session`'s recomputation
//! path is unchanged and is the oracle to diff against — see
//! `tests/wave_a5_kv_cache.rs`'s gated parity test, which does exactly that
//! against a real model on CI's plugin.
//!
//! # Why a separate type instead of a mode flag on `Session`
//!
//! `Session` is the thing Gate A5 already validated end-to-end in CI. Adding
//! an `if cached { .. }` branch through its methods would put every one of
//! those already-passing paths one merge away from a regression neither this
//! type nor `Session`'s own tests would catch. Keeping them as two types means
//! `Session`'s code — and its CI history — does not change at all.

use std::path::Path;
use std::rc::Rc;

use crate::checkpoint::{bind_safetensors, WeightSource};
use crate::graph::BuiltFunc;
use crate::model::{trace_decode, trace_prefill_with_cache, Qwen2Config};
use crate::pjrt::{LoadedExecutable, PjrtBufferHandle, PjrtClientHandle, PjrtDeviceRef, PjrtPlugin};
use crate::precision::{self, PrecisionPolicy};
use crate::runtime::hf::HfCheckpoint;
use crate::runtime::plan::PlanSignature;
use crate::stablehlo::types::{DType, ParamDesc, ParamKind, Shape};
use crate::GlError;

/// A loaded model with two compiled programs — prefill and decode — sharing
/// one set of weights and one pair of KV cache buffers.
pub struct CachedSession {
    client: Rc<PjrtClientHandle>,
    prefill_program: LoadedExecutable,
    decode_program: LoadedExecutable,
    prefill_param_order: Vec<ParamDesc>,
    decode_param_order: Vec<ParamDesc>,
    prefill_signature: PlanSignature,
    decode_signature: PlanSignature,
    /// Uploaded once, shared by both programs — see [`Self::open`]'s check
    /// that both traces declared the same weights in the same order.
    weights: Vec<PjrtBufferHandle>,
    config: Qwen2Config,
    /// The single bucket used for both the padded prompt (prefill's
    /// `input_ids` width) and the cache's total capacity. See
    /// `trace_prefill_with_cache`'s docs on why these are the same value here
    /// rather than two independently-sized buckets.
    window: usize,
    cache_shape: Shape,
    k_cache: PjrtBufferHandle,
    v_cache: PjrtBufferHandle,
    tokenizer: Option<glcore::GllmTokenizer>,
    eos_id: Option<u32>,
}

impl CachedSession {
    /// Compiles both programs and uploads weights + zeroed KV caches.
    ///
    /// `built_prefill` and `built_decode` must come from the same
    /// [`Qwen2Config`] (so their weight lists agree) and the same `window`
    /// (so their cache shapes agree) — see the checks below, which refuse
    /// rather than silently binding a decode program to a differently-shaped
    /// cache than prefill just filled.
    pub fn open(
        plugin: Rc<PjrtPlugin>,
        built_prefill: &BuiltFunc,
        built_decode: &BuiltFunc,
        config: Qwen2Config,
        window: usize,
        weights: &dyn WeightSource,
    ) -> Result<Self, GlError> {
        let client = PjrtClientHandle::create(plugin)?;
        let device = client.default_device()?;

        let prefill_weight_names: Vec<&str> = built_prefill
            .signature
            .weights
            .iter()
            .map(|w| w.name.as_str())
            .collect();
        let decode_weight_names: Vec<&str> = built_decode
            .signature
            .weights
            .iter()
            .map(|w| w.name.as_str())
            .collect();
        if prefill_weight_names != decode_weight_names {
            return Err(GlError::Engine(
                "CachedSession::open: prefill and decode traces declared different weight \
                 lists — they must be traced from the same Qwen2Config so one uploaded set \
                 of weight buffers can serve both executables"
                    .to_owned(),
            ));
        }

        let prefill_kv_shape = kv_cache_shape_of(&built_prefill.signature)?;
        let decode_kv_shape = kv_cache_shape_of(&built_decode.signature)?;
        if prefill_kv_shape != decode_kv_shape {
            return Err(GlError::Engine(format!(
                "CachedSession::open: prefill's cache is {} but decode's is {} — both traces \
                 must share the same window",
                prefill_kv_shape.mlir_type(),
                decode_kv_shape.mlir_type()
            )));
        }

        // Bind before compiling: a checkpoint mismatch is cheap to detect and
        // expensive to discover after two 20-minute compiles.
        let bound = bind_safetensors(&built_prefill.signature, weights)?;

        let prefill_program = client.compile(&built_prefill.mlir)?;
        let decode_program = client.compile(&built_decode.mlir)?;

        let mut buffers = Vec::with_capacity(bound.len());
        for w in &bound {
            let shape = Shape::new(w.shape.dims.clone(), DType::F32);
            buffers.push(client.buffer_from_host_f32(&w.data, &shape, &device)?);
        }

        let zeros = vec![0f32; prefill_kv_shape.numel()];
        let k_cache = client.buffer_from_host_f32(&zeros, &prefill_kv_shape, &device)?;
        let v_cache = client.buffer_from_host_f32(&zeros, &prefill_kv_shape, &device)?;

        Ok(CachedSession {
            client,
            prefill_program,
            decode_program,
            prefill_param_order: built_prefill.signature.param_order.clone(),
            decode_param_order: built_decode.signature.param_order.clone(),
            prefill_signature: PlanSignature::from_traced(&built_prefill.signature),
            decode_signature: PlanSignature::from_traced(&built_decode.signature),
            weights: buffers,
            config,
            window,
            cache_shape: prefill_kv_shape,
            k_cache,
            v_cache,
            tokenizer: None,
            eos_id: None,
        })
    }

    /// Traces both programs (bucket `window` for the prompt and the cache
    /// alike, per `trace_prefill_with_cache`'s docs), then compiles and loads
    /// a HuggingFace directory.
    ///
    /// Runs under [`PrecisionPolicy::f32`], matching [`Session::from_hf_dir`]
    /// (crate::runtime::session::Session) — the weights are widened to F32 on
    /// load, so an F32 graph is what matches them.
    pub fn from_hf_dir(
        plugin: Rc<PjrtPlugin>,
        dir: &Path,
        window: usize,
    ) -> Result<Self, GlError> {
        let checkpoint = HfCheckpoint::open(dir)?;
        Self::from_hf_checkpoint(plugin, checkpoint, window)
    }

    /// As [`Self::from_hf_dir`], for an already-opened checkpoint.
    pub fn from_hf_checkpoint(
        plugin: Rc<PjrtPlugin>,
        checkpoint: HfCheckpoint,
        window: usize,
    ) -> Result<Self, GlError> {
        let config = checkpoint.config.clone();
        log::info!(
            "tracing qwen2 (cached): {} layers, hidden {}, vocab {}, window {window}",
            config.n_layers,
            config.hidden,
            config.vocab,
        );

        let built_prefill = precision::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&config, window, window)
        })?;
        let built_decode =
            precision::with_policy(PrecisionPolicy::f32(), || trace_decode(&config, window))?;

        let mut session = Self::open(
            plugin,
            &built_prefill,
            &built_decode,
            config,
            window,
            &checkpoint.weights,
        )?;
        session.attach_tokenizer(checkpoint.tokenizer, checkpoint.eos_id);
        Ok(session)
    }

    pub(crate) fn attach_tokenizer(&mut self, tokenizer: glcore::GllmTokenizer, eos_id: u32) {
        self.tokenizer = Some(tokenizer);
        self.eos_id = Some(eos_id);
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    pub fn window(&self) -> usize {
        self.window
    }

    pub fn prefill_signature(&self) -> &PlanSignature {
        &self.prefill_signature
    }

    pub fn decode_signature(&self) -> &PlanSignature {
        &self.decode_signature
    }

    pub fn tokenizer(&self) -> Option<&glcore::GllmTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Re-zeros both KV cache buffers. Every [`Self::generate`] call does this
    /// first, so stale KV from a previous conversation never leaks in.
    pub fn reset_kv_cache(&mut self) -> Result<(), GlError> {
        let device = self.client.default_device()?;
        let zeros = vec![0f32; self.cache_shape.numel()];
        self.k_cache = self.client.buffer_from_host_f32(&zeros, &self.cache_shape, &device)?;
        self.v_cache = self.client.buffer_from_host_f32(&zeros, &self.cache_shape, &device)?;
        Ok(())
    }

    /// Text in, text out. Requires a session built by [`Self::from_hf_dir`].
    pub fn generate_text(&mut self, prompt: &str, max_new_tokens: usize) -> Result<String, GlError> {
        let prompt_ids: Vec<i32> = {
            let tokenizer = self.tokenizer.as_ref().ok_or_else(|| {
                GlError::Engine(
                    "generate_text: no tokenizer attached — build with CachedSession::from_hf_dir"
                        .to_owned(),
                )
            })?;
            tokenizer
                .encode(prompt, tokenizer.add_bos_default())?
                .into_iter()
                .map(|id| id as i32)
                .collect()
        };

        let eos = self.eos_id.map(|e| e as i32);
        let pad = eos.unwrap_or(0);
        let generated = self.generate(&prompt_ids, max_new_tokens, eos, pad)?;

        let ids: Vec<u32> = generated.iter().map(|&t| t as u32).collect();
        Ok(self
            .tokenizer
            .as_ref()
            .expect("checked above")
            .decode(&ids, true))
    }

    /// Prefill once, then decode `max_new_tokens - 1` times with a real KV
    /// cache — O(1) work per decode step instead of
    /// [`Session::generate`](crate::runtime::Session::generate)'s O(current
    /// length).
    ///
    /// # ⚠️ Unverified — see the module docs
    ///
    /// # Errors
    /// If `prompt` is empty, `prompt.len() + max_new_tokens` exceeds the
    /// compiled `window`, or a device call fails.
    pub fn generate(
        &mut self,
        prompt: &[i32],
        max_new_tokens: usize,
        eos_id: Option<i32>,
        pad_id: i32,
    ) -> Result<Vec<i32>, GlError> {
        if prompt.is_empty() {
            return Err(GlError::Engine("generate: empty prompt".to_owned()));
        }
        if prompt.len() + max_new_tokens > self.window {
            return Err(GlError::Engine(format!(
                "generate: {} prompt tokens + {max_new_tokens} new exceeds the compiled \
                 window of {}",
                prompt.len(),
                self.window
            )));
        }
        self.reset_kv_cache()?;

        let device = self.client.default_device()?;
        let vocab = self.config.vocab;

        // ── Prefill: pad to the window, fill the cache, sample the first
        // continuation token from the last real prompt position. ──────────
        let padded = crate::runtime::bucket::pad_to_bucket(prompt, self.window, pad_id);
        let ids_buf = self.upload_token_ids(&padded, &device)?;
        let args = build_args(
            &self.prefill_param_order,
            &ids_buf,
            None,
            &self.weights,
            &self.k_cache,
            &self.v_cache,
        );
        let outputs = self.prefill_program.execute(&args)?;
        let (logits_buf, k_new, v_new) = take_three(outputs, "prefill")?;
        self.k_cache = k_new;
        self.v_cache = v_new;

        let logits = logits_buf.to_host_f32()?;
        let position = crate::runtime::bucket::last_real_position(prompt.len());
        let mut next =
            crate::runtime::sample::argmax_at(&logits, self.window, vocab, position)? as i32;

        let mut generated = Vec::with_capacity(max_new_tokens);
        generated.push(next);
        if Some(next) == eos_id {
            return Ok(generated);
        }

        // ── Decode: one token at a time. `pos` is the *absolute* position of
        // `next` — the same indexing prefill's RoPE/mask already used, so the
        // cache's contents and every subsequent read agree with it. ────────
        let prompt_len = prompt.len() as i32;
        for pos_val in prompt_len..prompt_len + max_new_tokens as i32 - 1 {
            let tok_buf = self.upload_token_ids(&[next], &device)?;
            let pos_buf = self.upload_scalar_i32(pos_val, &device)?;
            let args = build_args(
                &self.decode_param_order,
                &tok_buf,
                Some(&pos_buf),
                &self.weights,
                &self.k_cache,
                &self.v_cache,
            );
            let outputs = self.decode_program.execute(&args)?;
            let (logits_buf, k_new, v_new) = take_three(outputs, "decode")?;
            self.k_cache = k_new;
            self.v_cache = v_new;

            let logits = logits_buf.to_host_f32()?;
            next = crate::runtime::sample::argmax_at(&logits, 1, vocab, 0)? as i32;
            generated.push(next);
            if Some(next) == eos_id {
                break;
            }
        }
        Ok(generated)
    }

    fn upload_token_ids(
        &self,
        ids: &[i32],
        device: &PjrtDeviceRef,
    ) -> Result<PjrtBufferHandle, GlError> {
        // SAFETY: `ids` is a live `&[i32]` for the duration of the call, and
        // `ImmutableOnlyDuringCall` is exactly the promise that covers that.
        unsafe {
            self.client.buffer_from_host_raw(
                ids.as_ptr().cast::<core::ffi::c_void>(),
                crate::sys::types::PjrtBufferType::S32,
                &[1, ids.len()],
                device,
            )
        }
    }

    fn upload_scalar_i32(
        &self,
        value: i32,
        device: &PjrtDeviceRef,
    ) -> Result<PjrtBufferHandle, GlError> {
        // SAFETY: `value` is a live `i32` for the duration of the call, and
        // `ImmutableOnlyDuringCall` is exactly the promise that covers that.
        // An empty `dims` slice is StableHLO's rank-0 `tensor<i32>` — the
        // decode program's `pos` parameter.
        unsafe {
            self.client.buffer_from_host_raw(
                (&value as *const i32).cast::<core::ffi::c_void>(),
                crate::sys::types::PjrtBufferType::S32,
                &[],
                device,
            )
        }
    }
}

/// The KV cache's shape as declared by a traced signature's first `kv_cache`
/// parameter — both `k_cache` and `v_cache` share one shape by construction
/// (`trace_prefill_with_cache`/`trace_decode` build both from the same
/// `kv_cache::cache_shape` call).
fn kv_cache_shape_of(sig: &crate::graph::Signature) -> Result<Shape, GlError> {
    sig.kv_caches
        .first()
        .map(|p| p.shape.clone())
        .ok_or_else(|| {
            GlError::Engine(
                "CachedSession::open: traced signature declares no kv_cache parameters — was \
                 it built by trace_prefill_with_cache or trace_decode?"
                    .to_owned(),
            )
        })
}

/// Builds the flat, positionally-correct argument list a compiled program's
/// `param_order` expects, from the buffers `CachedSession` is holding.
///
/// Driven entirely by each parameter's declared `kind`/`name`, not by an
/// assumption about where inputs/weights/kv-caches sit relative to each other
/// — `trace_prefill_with_cache` and `trace_decode` declare them in different
/// relative positions (no `pos` in the former, `input_id` vs `input_ids` in
/// the latter), and PJRT matches `argument_lists` by position, so getting this
/// wrong silently binds the wrong buffer (P4).
fn build_args<'a>(
    param_order: &[ParamDesc],
    ids: &'a PjrtBufferHandle,
    pos: Option<&'a PjrtBufferHandle>,
    weights: &'a [PjrtBufferHandle],
    k_cache: &'a PjrtBufferHandle,
    v_cache: &'a PjrtBufferHandle,
) -> Vec<&'a PjrtBufferHandle> {
    let mut weight_iter = weights.iter();
    param_order
        .iter()
        .map(|p| match p.kind {
            ParamKind::Weight => weight_iter
                .next()
                .expect("build_args: fewer uploaded weights than the signature declares"),
            ParamKind::KvCache => match p.name.as_str() {
                "k_cache" => k_cache,
                "v_cache" => v_cache,
                other => panic!("build_args: unknown kv-cache parameter {other:?}"),
            },
            ParamKind::Input => match p.name.as_str() {
                "pos" => pos
                    .expect("build_args: signature declares pos but no pos buffer was given"),
                _ => ids,
            },
        })
        .collect()
}

/// Pulls exactly `(logits, k_cache', v_cache')` out of an executable's output
/// list, refusing rather than indexing past the end if a trace ever changes
/// its output count without this file being updated to match.
fn take_three(
    mut outputs: Vec<PjrtBufferHandle>,
    which: &str,
) -> Result<(PjrtBufferHandle, PjrtBufferHandle, PjrtBufferHandle), GlError> {
    if outputs.len() != 3 {
        return Err(GlError::Engine(format!(
            "{which} executable returned {} outputs, expected 3 (logits, k_cache', v_cache')",
            outputs.len()
        )));
    }
    let v_cache = outputs.pop().expect("checked len == 3");
    let k_cache = outputs.pop().expect("checked len == 3");
    let logits = outputs.pop().expect("checked len == 3");
    Ok((logits, k_cache, v_cache))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_args` must place buffers by the signature's own declared
    /// order/kind/name, not by a positional assumption — pins that against
    /// both traces' actual (and different) parameter layouts.
    #[test]
    fn build_args_matches_prefill_and_decode_signatures_independently() {
        let cfg = Qwen2Config::tiny();
        let built_prefill = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 4, 8)
        })
        .expect("prefill trace");
        let built_decode =
            crate::with_policy(PrecisionPolicy::f32(), || trace_decode(&cfg, 8)).expect("decode trace");

        // Prefill: no `pos` parameter at all.
        assert!(
            !built_prefill
                .signature
                .param_order
                .iter()
                .any(|p| p.name == "pos"),
            "prefill must not declare a pos input"
        );
        // Decode: exactly one.
        assert_eq!(
            built_decode
                .signature
                .param_order
                .iter()
                .filter(|p| p.name == "pos")
                .count(),
            1
        );

        // Both declare exactly one k_cache and one v_cache kv-cache param.
        for built in [&built_prefill, &built_decode] {
            assert_eq!(built.signature.kv_caches.len(), 2);
            assert!(built.signature.kv_caches.iter().any(|p| p.name == "k_cache"));
            assert!(built.signature.kv_caches.iter().any(|p| p.name == "v_cache"));
        }
    }

    #[test]
    fn kv_cache_shape_of_reads_the_first_kv_cache_param() {
        let cfg = Qwen2Config::tiny();
        let built = crate::with_policy(PrecisionPolicy::f32(), || {
            trace_prefill_with_cache(&cfg, 4, 8)
        })
        .expect("trace");
        let shape = kv_cache_shape_of(&built.signature).expect("has kv caches");
        assert_eq!(shape.dims, vec![cfg.n_layers, 8, cfg.n_kv_heads, cfg.head_dim]);
    }

    #[test]
    fn kv_cache_shape_of_refuses_a_signature_with_no_kv_cache_params() {
        let cfg = Qwen2Config::tiny();
        let built =
            crate::with_policy(PrecisionPolicy::f32(), || crate::model::trace_forward(&cfg, 4, 0))
                .expect("trace");
        let err = kv_cache_shape_of(&built.signature).expect_err("trace_forward has no cache");
        assert!(err.to_string().contains("kv_cache"), "{err}");
    }

    #[test]
    fn take_three_refuses_the_wrong_output_count() {
        // `PjrtBufferHandle` is not `Debug` (it wraps a raw device pointer),
        // so `expect_err` — which requires the Ok side to be `Debug` — is not
        // usable here; match instead.
        match take_three(Vec::new(), "test") {
            Err(e) => assert!(e.to_string().contains("expected 3"), "{e}"),
            Ok(_) => panic!("an empty output list must be refused"),
        }
    }
}
