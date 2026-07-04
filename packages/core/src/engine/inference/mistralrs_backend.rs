// engine/inference/mistralrs_backend.rs — mistral.rs inference backend implementation.
//
// Provides `MistralRsBackend`, a production-ready inference engine using the
// `mistralrs` crate. Gated behind `#[cfg(feature = "mistralrs-backend")]` to
// avoid compile-time dependency when not in use.
//
// Requirements: 3.1–3.8, 6.1–6.7, 7.1–7.4, 11.1, 11.2, 11.4

#[cfg(feature = "mistralrs-backend")]
mod impl_mistralrs {
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::sync::{Arc, Mutex};

    use anyhow::{Context, Result};
    use futures_util::{Stream, StreamExt};

    use super::super::{
        arch_detect::detect_architecture, backend::InferenceBackend, params::InferParams,
    };

    struct MistralRsBackendInner {
        model: Option<Arc<mistralrs::Model>>,
        model_path: Option<PathBuf>,
    }

    /// Inference backend powered by mistral.rs.
    ///
    /// All methods are thread-safe (`Send + Sync`). The `Arc<Mutex<…>>` pattern
    /// allows `&self` receivers while protecting mutable state (the loaded model).
    pub struct MistralRsBackend {
        inner: Arc<Mutex<MistralRsBackendInner>>,
    }

    impl MistralRsBackend {
        pub fn new() -> Self {
            Self {
                inner: Arc::new(Mutex::new(MistralRsBackendInner {
                    model: None,
                    model_path: None,
                })),
            }
        }
    }

    /// Bridge an async future to sync, using the current tokio runtime if
    /// present (block_in_place) or creating a temporary one as fallback.
    fn block_async<F, T>(fut: F) -> Option<T>
    where
        F: std::future::Future<Output = T> + Send + 'static,
        T: Send + 'static,
    {
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            Some(tokio::task::block_in_place(|| handle.block_on(fut)))
        } else {
            tokio::runtime::Runtime::new().ok()?.block_on(async { Some(fut.await) })
        }
    }

    /// Convert `InferParams` into `mistralrs::SamplingParams`.
    fn to_sampling_params(params: &InferParams) -> mistralrs::SamplingParams {
        // SamplingParams has no Default impl — use neutral() as the base.
        let mut sp = mistralrs::SamplingParams::neutral();
        sp.temperature = Some(params.temperature as f64);
        sp.top_k = params.top_k;                                      // Option<usize>
        sp.top_p = Some(params.top_p as f64);
        sp.max_len = Some(params.max_tokens);
        sp.repetition_penalty = params.repetition_penalty;            // Option<f32>
        sp.stop_toks = if params.stop_sequences.is_empty() {
            None
        } else {
            // StopTokens::Seqs accepts Vec<String>
            Some(mistralrs::StopTokens::Seqs(params.stop_sequences.clone()))
        };
        sp
    }

    impl InferenceBackend for MistralRsBackend {
        fn load_model(&self, model_path: &Path) -> Result<()> {
            let _arch = detect_architecture(model_path)
                .context("failed to detect model architecture")?;

            let path_str = model_path.to_string_lossy().to_string();
            let path_buf = model_path.to_path_buf();

            let file_str = path_buf.to_string_lossy().to_string();
            let model = block_async(async move {
                mistralrs::GgufModelBuilder::new(path_str, vec![file_str])
                    .with_force_cpu()
                    .build()
                    .await
            })
            .context("no async runtime available")?
            .context("failed to build mistralrs model")?;

            let mut inner = self.inner.lock().unwrap();
            inner.model = Some(Arc::new(model));
            inner.model_path = Some(model_path.to_path_buf());
            Ok(())
        }

        fn infer(&self, prompt: &str, params: &InferParams) -> Result<String> {
            let stream = self.stream_infer(prompt, params)?;
            let tokens = block_async(async move {
                let v: Vec<String> = stream.collect().await;
                v
            })
            .context("failed to collect stream tokens")?;
            Ok(tokens.join(""))
        }

        fn stream_infer(
            &self,
            prompt: &str,
            params: &InferParams,
        ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>> {
            // Clone Arc out from under the mutex before async work.
            let model = self
                .inner
                .lock()
                .unwrap()
                .model
                .clone()
                .context("no model loaded; call load_model first")?;

            let sampling_params = to_sampling_params(params);
            let max_tokens = params.max_tokens;
            let stop_sequences = params.stop_sequences.clone();

            // Build request via RequestBuilder so we can inject custom SamplingParams.
            // TextMessages::take_sampling_params() always returns deterministic() —
            // we must use RequestBuilder::set_sampling() to override it.
            let request = mistralrs::RequestBuilder::from(
                mistralrs::TextMessages::new()
                    .add_message(mistralrs::TextMessageRole::User, prompt),
            )
            .set_sampling(sampling_params);

            let stream = async_stream::stream! {
                match model.stream_chat_request(request).await {
                    Ok(mut response_stream) => {
                        let mut token_count = 0usize;
                        while let Some(response) = response_stream.next().await {
                            match response {
                                mistralrs::Response::Chunk(chunk) => {
                                    if let Some(content) = chunk
                                        .choices
                                        .first()
                                        .and_then(|ch| ch.delta.content.as_ref())
                                    {
                                        if token_count >= max_tokens {
                                            break;
                                        }
                                        let should_stop = stop_sequences
                                            .iter()
                                            .any(|seq| content.contains(seq.as_str()));
                                        yield content.clone();
                                        token_count += 1;
                                        if should_stop {
                                            break;
                                        }
                                    }
                                }
                                // Done(ChatCompletionResponse) — tuple variant in 0.8.1
                                mistralrs::Response::Done(_) => break,
                                // Error variants — log and stop, never panic
                                mistralrs::Response::InternalError(e) => {
                                    eprintln!("mistralrs internal error: {e}");
                                    break;
                                }
                                mistralrs::Response::ValidationError(e) => {
                                    eprintln!("mistralrs validation error: {e}");
                                    break;
                                }
                                mistralrs::Response::ModelError(msg, _) => {
                                    eprintln!("mistralrs model error: {msg}");
                                    break;
                                }
                                _ => {}
                            }
                        }
                    }
                    Err(e) => {
                        eprintln!("mistralrs stream_chat_request failed: {e}");
                    }
                }
            };

            Ok(Box::pin(stream))
        }

        fn unload(&self) -> Result<()> {
            let mut inner = self.inner.lock().unwrap();
            inner.model = None;
            inner.model_path = None;
            Ok(())
        }

        fn name(&self) -> &'static str {
            "mistralrs"
        }
    }

    impl Default for MistralRsBackend {
        fn default() -> Self {
            Self::new()
        }
    }
}

#[cfg(feature = "mistralrs-backend")]
pub use impl_mistralrs::MistralRsBackend;

#[cfg(not(feature = "mistralrs-backend"))]
pub struct MistralRsBackend;

#[cfg(not(feature = "mistralrs-backend"))]
impl MistralRsBackend {
    pub fn new() -> Self {
        MistralRsBackend
    }
}

#[cfg(not(feature = "mistralrs-backend"))]
impl Default for MistralRsBackend {
    fn default() -> Self {
        Self::new()
    }
}
