// engine/inference/registry.rs — Backend registry for runtime selection.
//
// `BackendRegistry` is the central store for all registered [`InferenceBackend`]
// implementations. Backends are inserted by name at startup (or by tests) and
// retrieved by the chat/inference pipeline at inference time.
//
// ── Registration order and backward compatibility ───────────────────────────
//
//   `BackendRegistry::new()` always registers every compiled-in backend so
//   that `selector::select_backend` can look them up by name at runtime.
//   Adding a new backend here is the *only* place that needs to change —
//   no other module needs to know which backends exist.
//
//   Backend name → key used in InferenceConfig.backend / CLI --backend flag:
//     "mistralrs"   — only when built with --features mistralrs-backend
//     "candle-ggqr" — only when built with --features candle-backend
//
//   `MistralRsBackend` remains the default when `auto` is requested and
//   both features are compiled in (see selector.rs).
//
// Requirements: 4.1–4.5, 11.1–11.5, 21.1, 21.2

use std::collections::HashMap;

use super::backend::InferenceBackend;
#[cfg(feature = "mistralrs-backend")]
use super::mistralrs_backend::MistralRsBackend;
#[cfg(feature = "candle-backend")]
use super::candle_ggqr::GgqrCandleBackend;

/// Central registry of named inference backends.
///
/// Backends are registered under a string key (e.g. `"candle-ggqr"`,
/// `"mistralrs"`) and retrieved by name at inference time.  The registry owns
/// the backends via `Box<dyn InferenceBackend>`, so each backend lives for as
/// long as the registry itself.
///
/// # Thread safety
///
/// `BackendRegistry` itself is not `Sync` — it is intended to be constructed
/// once on the main thread and then accessed via an `Arc<Mutex<…>>` if shared
/// across threads.
///
/// # Example
///
/// ```rust,ignore
/// let mut registry = BackendRegistry::new();
/// registry.register("my_backend", Box::new(MyBackend::new()));
/// let backend = registry.get("my_backend").expect("backend not found");
/// ```
pub struct BackendRegistry {
    backends: HashMap<String, Box<dyn InferenceBackend>>,
}

impl BackendRegistry {
    /// Create a registry pre-populated with every compiled-in backend.
    ///
    /// Registered backends (conditional on Cargo features):
    ///
    /// | Feature flag          | Backend name   | Type                 |
    /// |-----------------------|----------------|----------------------|
    /// | `mistralrs-backend`   | `"mistralrs"`  | `MistralRsBackend`   |
    /// | `candle-backend`      | `"candle-ggqr"`| `GgqrCandleBackend`  |
    ///
    /// Requirements: 11.1, 11.3, 21.1
    pub fn new() -> Self {
        #[allow(unused_mut)]
        let mut r = Self {
            backends: HashMap::new(),
        };

        // Register mistralrs when compiled in.  It is listed first so that
        // `list_available()` (which sorts lexicographically) still puts it
        // after "candle-ggqr", which is intentional — `"auto"` in the
        // selector prefers mistralrs explicitly by name, not by sort order.
        #[cfg(feature = "mistralrs-backend")]
        {
            eprintln!("gwen-registry: registering backend 'mistralrs'");
            r.register("mistralrs", Box::new(MistralRsBackend::new()));
        }

        // Register the GGQR-Candle backend when compiled in.
        // Requirement: 21.1
        #[cfg(feature = "candle-backend")]
        {
            eprintln!("gwen-registry: registering backend 'candle-ggqr'");
            r.register("candle-ggqr", Box::new(GgqrCandleBackend::new()));
        }

        r
    }

    /// Register a backend under the given `name`.
    ///
    /// If a backend with the same `name` was already registered it is silently
    /// replaced.
    ///
    /// # Arguments
    ///
    /// * `name`    – The lookup key (should match [`InferenceBackend::name`]).
    /// * `backend` – Boxed trait object to store in the registry.
    pub fn register(&mut self, name: &str, backend: Box<dyn InferenceBackend>) {
        self.backends.insert(name.to_string(), backend);
    }

    /// Look up a backend by name.
    ///
    /// Returns `Some(&dyn InferenceBackend)` when `name` is registered, or
    /// `None` otherwise.  The caller borrows the backend for as long as the
    /// registry is alive.
    ///
    /// # Arguments
    ///
    /// * `name` – The backend identifier passed to [`register`].
    ///
    /// [`register`]: BackendRegistry::register
    pub fn get(&self, name: &str) -> Option<&dyn InferenceBackend> {
        self.backends.get(name).map(|b| b.as_ref())
    }

    /// Return a sorted list of all registered backend names.
    ///
    /// The list is sorted lexicographically so that callers receive a
    /// deterministic ordering regardless of insertion order.
    pub fn list_available(&self) -> Vec<String> {
        let mut names: Vec<String> = self.backends.keys().cloned().collect();
        names.sort();
        names
    }
}

impl Default for BackendRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::Path;
    use std::pin::Pin;

    use anyhow::Result;
    use futures_util::Stream;

    use crate::engine::inference::params::InferParams;

    /// Minimal no-op backend used only inside this test module.
    struct MockBackend;

    impl InferenceBackend for MockBackend {
        fn load_model(&self, _model_path: &Path) -> Result<()> {
            Ok(())
        }

        fn infer(&self, _prompt: &str, _params: &InferParams) -> Result<String> {
            Ok(String::new())
        }

        fn stream_infer(
            &self,
            _prompt: &str,
            _params: &InferParams,
        ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>> {
            Ok(Box::pin(futures_util::stream::empty()))
        }

        fn unload(&self) -> Result<()> {
            Ok(())
        }

        fn name(&self) -> &'static str {
            "mock"
        }
    }

    // ── register + get returns Some ───────────────────────────────────────────

    #[test]
    fn register_then_get_returns_some() {
        let mut registry = BackendRegistry::new();
        registry.register("mock", Box::new(MockBackend));
        assert!(registry.get("mock").is_some());
    }

    // ── get on unregistered name returns None ─────────────────────────────────

    #[test]
    fn get_unregistered_returns_none() {
        let registry = BackendRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    // ── list_available contains registered names ──────────────────────────────

    #[test]
    fn list_available_contains_registered_name() {
        let mut registry = BackendRegistry::new();
        registry.register("mock", Box::new(MockBackend));
        assert!(registry.list_available().contains(&"mock".to_string()));
    }

    // ── list_available does NOT contain unregistered names ────────────────────

    #[test]
    fn list_available_excludes_unregistered_name() {
        let mut registry = BackendRegistry::new();
        registry.register("mock", Box::new(MockBackend));
        assert!(!registry.list_available().contains(&"candle".to_string()));
    }

    // ── list_available is sorted ──────────────────────────────────────────────

    #[test]
    fn list_available_is_sorted() {
        let mut registry = BackendRegistry::new();
        // Insert in reverse alphabetical order to verify sorting.
        registry.register("zebra", Box::new(MockBackend));
        registry.register("alpha", Box::new(MockBackend));
        registry.register("mock", Box::new(MockBackend));
        let names = registry.list_available();
        let is_sorted = names.windows(2).all(|w| w[0] <= w[1]);
        assert!(is_sorted, "list_available() should return names in sorted order");
    }

    // ── new registry contains exactly the compiled-in backends ───────────────
    //
    // The expected set of backends depends on which feature flags are active.
    // We test each combination so the assertion is always precise rather than
    // leaving the "empty otherwise" case unchecked.

    #[test]
    fn new_registry_contains_compiled_in_backends() {
        let registry = BackendRegistry::new();
        let available = registry.list_available();

        // Each feature implies its backend is present.
        #[cfg(feature = "mistralrs-backend")]
        assert!(
            available.contains(&"mistralrs".to_string()),
            "expected 'mistralrs' in {:?}",
            available
        );
        #[cfg(feature = "candle-backend")]
        assert!(
            available.contains(&"candle-ggqr".to_string()),
            "expected 'candle-ggqr' in {:?}",
            available
        );

        // When neither feature is active the registry must be empty.
        #[cfg(not(any(feature = "mistralrs-backend", feature = "candle-backend")))]
        assert!(available.is_empty(), "expected empty registry, got {:?}", available);
    }

    // ── candle-ggqr is retrievable by exact name ──────────────────────────────
    //
    // Requirement: 11.3 — the backend must be reachable via BackendRegistry::get.

    #[test]
    #[cfg(feature = "candle-backend")]
    fn candle_ggqr_backend_is_retrievable() {
        let registry = BackendRegistry::new();
        let backend = registry.get("candle-ggqr");
        assert!(backend.is_some(), "candle-ggqr should be registered when feature is active");
        assert_eq!(backend.unwrap().name(), "candle-ggqr");
    }

    // ── default() delegates to new() ─────────────────────────────────────────

    #[test]
    fn default_registry_matches_new_registry() {
        let from_new     = BackendRegistry::new().list_available();
        let from_default = BackendRegistry::default().list_available();
        assert_eq!(from_new, from_default, "default() and new() should produce identical registries");
    }
}
