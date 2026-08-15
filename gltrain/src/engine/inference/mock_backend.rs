// engine/inference/mock_backend.rs — MockBackend test double.
// Only compiled in test builds or with the "test-utils" feature.

use std::path::Path;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use anyhow::Result;
use futures_util::Stream;
use async_stream::stream;
use super::backend::InferenceBackend;
use super::params::InferParams;

/// A configurable test double for InferenceBackend.
///
/// Tracks call counts and allows configuring:
/// - A fixed token list to yield from stream_infer / infer
/// - Whether load_model should fail
#[derive(Clone)]
pub struct MockBackend {
    inner: Arc<Mutex<MockBackendInner>>,
}

struct MockBackendInner {
    pub tokens: Vec<String>,
    pub fail_on_load: bool,
    pub load_count: usize,
    pub unload_count: usize,
    pub loaded: bool,
}

impl MockBackend {
    pub fn new(tokens: Vec<String>) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MockBackendInner {
                tokens,
                fail_on_load: false,
                load_count: 0,
                unload_count: 0,
                loaded: false,
            })),
        }
    }

    pub fn with_fail_on_load(tokens: Vec<String>) -> Self {
        let m = Self::new(tokens);
        m.inner.lock().unwrap().fail_on_load = true;
        m
    }

    pub fn load_count(&self) -> usize { self.inner.lock().unwrap().load_count }
    pub fn unload_count(&self) -> usize { self.inner.lock().unwrap().unload_count }
    pub fn is_loaded(&self) -> bool { self.inner.lock().unwrap().loaded }
}

impl InferenceBackend for MockBackend {
    fn load_model(&self, _path: &Path) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        if g.fail_on_load {
            anyhow::bail!("MockBackend: configured to fail on load");
        }
        g.load_count += 1;
        g.loaded = true;
        Ok(())
    }

    fn infer(&self, _prompt: &str, params: &InferParams) -> Result<String> {
        let g = self.inner.lock().unwrap();
        let tokens: Vec<_> = g.tokens.iter()
            .take(params.max_tokens)
            .cloned()
            .collect();
        Ok(tokens.join(""))
    }

    fn stream_infer(
        &self,
        _prompt: &str,
        params: &InferParams,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>> {
        let g = self.inner.lock().unwrap();
        let tokens: Vec<_> = g.tokens.iter()
            .take(params.max_tokens)
            .cloned()
            .collect();
        // Check stop_sequences
        let stop = params.stop_sequences.clone();
        let out = stream! {
            for tok in tokens {
                if stop.contains(&tok) {
                    break;
                }
                yield tok;
            }
        };
        Ok(Box::pin(out))
    }

    fn unload(&self) -> Result<()> {
        let mut g = self.inner.lock().unwrap();
        g.unload_count += 1;
        g.loaded = false;
        Ok(())
    }

    fn name(&self) -> &'static str { "mock" }
}
