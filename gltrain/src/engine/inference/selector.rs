// engine/inference/selector.rs — Backend name resolution and model path expansion.
//
// This is the single authoritative place for the "which backend wins" logic.
// All other modules take a resolved `&str` name and look it up in
// `BackendRegistry`; they never contain priority tables or feature-flag checks.
//
// ── Backend selection priority (for `backend = "auto"`) ────────────────────
//
//   1. "mistralrs"   — if built with --features mistralrs-backend
//   2. "candle-ggqr" — if built with --features candle-backend
//   3. Error         — no inference backend compiled in
//
// Explicit backend names ("mistralrs", "candle-ggqr", "candle") are passed
// through as-is after a feature-availability check so existing configs and
// CLI invocations continue to work unchanged (Requirement 21.2).
//
// ── Backward-compatibility note ─────────────────────────────────────────────
//
// The legacy name "candle" (without the "-ggqr" suffix) is accepted as an
// alias for "candle-ggqr" so that config files written before the backend was
// renamed continue to work.  No silent migration is performed; the returned
// name is always the canonical form that `BackendRegistry` uses as its key.
//
// Requirements: 4.6, 4.7, 11.4, 11.5, 15.1–15.5, 21.2, 22.1

use std::path::PathBuf;

use crate::engine::inference::config::InferenceConfig;
use crate::error::GwenError;

/// Resolve the backend name and absolute model path from `config`.
///
/// # Steps
///
/// 1. Reject an empty `model_path`.
/// 2. Expand a leading `~` to the user's home directory.
/// 3. Require a `.gguf` file extension.
/// 4. Resolve the backend name (see module-level priority table).
/// 5. Log the resolved pair so callers can trace which backend was chosen.
///
/// # Errors
///
/// | Error variant              | When                                              |
/// |----------------------------|---------------------------------------------------|
/// | `InferenceBackend`         | `model_path` is empty / unset                     |
/// | `ModelLoad`                | file extension is not `.gguf`                     |
/// | `BackendNotAvailable`      | named backend not compiled in                     |
/// | `ArchitectureNotSupported` | `"auto"` but no inference backend is compiled in  |
///
/// # Returns
///
/// `(&'static str, PathBuf)` — canonical backend name + absolute model path.
pub fn select_backend(config: &InferenceConfig) -> Result<(&'static str, PathBuf), GwenError> {
    // ── 1. Require a non-empty model path ────────────────────────────────────
    let raw = config.model_path.to_string_lossy();
    if raw.is_empty() || config.model_path == PathBuf::from("") {
        return Err(GwenError::InferenceBackend(
            "no model_path set in InferenceConfig".to_string(),
        ));
    }

    // ── 2. Expand leading ~ ──────────────────────────────────────────────────
    let expanded = expand_tilde(&config.model_path);

    // ── 3. Require .gguf extension ───────────────────────────────────────────
    match expanded.extension().and_then(|e| e.to_str()) {
        Some("gguf") => {}
        _ => {
            return Err(GwenError::ModelLoad(format!(
                "model path does not have a .gguf extension: {}",
                expanded.display()
            )));
        }
    }

    // ── 4. Resolve backend name ───────────────────────────────────────────────
    let backend = resolve_backend(&config.backend)?;

    // ── 5. Diagnostic log ────────────────────────────────────────────────────
    //
    // Logged to stderr so it appears in the terminal without interfering with
    // token streaming on stdout.  Uses the same `eprintln!` convention as the
    // rest of the candle-ggqr module (no `log` crate dependency required).
    eprintln!(
        "gwen-selector: resolved backend='{}' model='{}'",
        backend,
        expanded.display()
    );

    Ok((backend, expanded))
}

/// Map a raw backend name string to the canonical key used by `BackendRegistry`.
///
/// Accepts:
/// - `"mistralrs"`   — explicit, requires `mistralrs-backend` feature
/// - `"candle-ggqr"` — explicit, requires `candle-backend` feature
/// - `"candle"`      — legacy alias for `"candle-ggqr"` (backward compat)
/// - `"auto"`        — selects the highest-priority compiled-in backend
/// - anything else   — treated as `"auto"` (forward-compat for future backends)
///
/// Requirements: 11.4, 11.5, 21.2
fn resolve_backend(requested: &str) -> Result<&'static str, GwenError> {
    match requested {
        // ── Explicit mistralrs ────────────────────────────────────────────────
        "mistralrs" => {
            #[cfg(feature = "mistralrs-backend")]
            { return Ok("mistralrs"); }
            #[cfg(not(feature = "mistralrs-backend"))]
            { return Err(GwenError::BackendNotAvailable { backend: "mistralrs".to_string() }); }
        }

        // ── Explicit candle-ggqr (canonical) or "candle" (legacy alias) ───────
        "candle-ggqr" | "candle" => {
            #[cfg(feature = "candle-backend")]
            { return Ok("candle-ggqr"); }
            #[cfg(not(feature = "candle-backend"))]
            { return Err(GwenError::BackendNotAvailable { backend: "candle-ggqr".to_string() }); }
        }

        // ── Auto / unknown — pick the best available backend ──────────────────
        //
        // Priority: mistralrs > candle-ggqr > error.
        // Unknown names fall through to "auto" behaviour so future backend
        // names in config files don't hard-error on older builds.
        _ => {
            #[cfg(feature = "mistralrs-backend")]
            { return Ok("mistralrs"); }

            #[cfg(all(not(feature = "mistralrs-backend"), feature = "candle-backend"))]
            { return Ok("candle-ggqr"); }

            #[cfg(not(any(feature = "mistralrs-backend", feature = "candle-backend")))]
            {
                return Err(GwenError::ArchitectureNotSupported(
                    "no inference backend is compiled in — rebuild with \
                     --features candle-backend or --features mistralrs-backend"
                        .to_string(),
                ));
            }
        }
    }
}

fn expand_tilde(path: &PathBuf) -> PathBuf {
    let s = path.to_string_lossy();
    if s.starts_with("~/") || s == "~" {
        if let Some(home) = dirs::home_dir() {
            let rest = s.strip_prefix("~/").unwrap_or("");
            return home.join(rest);
        }
    }
    path.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::inference::config::InferenceConfig;
    use std::path::PathBuf;

    fn cfg_with_path(p: &str) -> InferenceConfig {
        let mut c = InferenceConfig::default();
        c.model_path = PathBuf::from(p);
        c
    }

    fn cfg_with_path_and_backend(p: &str, backend: &str) -> InferenceConfig {
        let mut c = cfg_with_path(p);
        c.backend = backend.to_string();
        c
    }

    // 1. no_path — empty model_path → InferenceBackend error
    #[test]
    fn no_path() {
        let mut c = InferenceConfig::default();
        c.model_path = PathBuf::from("");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::InferenceBackend(_)),
            "expected InferenceBackend, got {:?}",
            err
        );
    }

    // 2. non_gguf — path with wrong extension → ModelLoad error
    #[test]
    fn non_gguf() {
        let c = cfg_with_path("/tmp/model.safetensors");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::ModelLoad(_)),
            "expected ModelLoad, got {:?}",
            err
        );
    }

    // 3. tilde_expand — path starting with ~ should expand to home directory
    #[test]
    fn tilde_expand() {
        let c = cfg_with_path("~/models/test.gguf");
        if let Some(home) = dirs::home_dir() {
            let result = select_backend(&c).unwrap();
            assert_eq!(result.1, home.join("models/test.gguf"));
        }
        // If no home dir, the test is vacuously OK (CI without $HOME)
    }

    // 4. candle_ggqr_explicit — requesting "candle-ggqr" by canonical name
    //    returns "candle-ggqr" when the feature is active, or BackendNotAvailable.
    #[test]
    fn candle_ggqr_explicit() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "candle-ggqr");
        #[cfg(feature = "candle-backend")]
        {
            let (backend, _) = select_backend(&c).unwrap();
            assert_eq!(backend, "candle-ggqr");
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            let err = select_backend(&c).unwrap_err();
            assert!(matches!(err, GwenError::BackendNotAvailable { .. }));
        }
    }

    // 4b. candle_legacy_alias — the old name "candle" maps to "candle-ggqr"
    //     for backward compatibility with existing config files.
    #[test]
    fn candle_legacy_alias() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "candle");
        #[cfg(feature = "candle-backend")]
        {
            let (backend, _) = select_backend(&c).unwrap();
            assert_eq!(backend, "candle-ggqr", "'candle' should resolve to 'candle-ggqr'");
        }
        #[cfg(not(feature = "candle-backend"))]
        {
            let err = select_backend(&c).unwrap_err();
            assert!(matches!(err, GwenError::BackendNotAvailable { .. }));
        }
    }

    // 5. mistralrs_no_feature_err — mistralrs without feature → BackendNotAvailable
    #[test]
    #[cfg(not(feature = "mistralrs-backend"))]
    fn mistralrs_no_feature_err() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "mistralrs");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::BackendNotAvailable { .. }),
            "expected BackendNotAvailable, got {:?}",
            err
        );
    }

    // 6. mistralrs_with_feature_ok — mistralrs with feature → "mistralrs"
    #[test]
    #[cfg(feature = "mistralrs-backend")]
    fn mistralrs_with_feature_ok() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "mistralrs");
        let (backend, _) = select_backend(&c).unwrap();
        assert_eq!(backend, "mistralrs");
    }

    // 7. auto_fallback_candle_ggqr — auto without mistralrs feature falls back
    //    to "candle-ggqr" when the candle-backend feature is active.
    #[test]
    #[cfg(all(not(feature = "mistralrs-backend"), feature = "candle-backend"))]
    fn auto_fallback_candle_ggqr() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "auto");
        let (backend, _) = select_backend(&c).unwrap();
        assert_eq!(backend, "candle-ggqr");
    }

    // 7b. auto_no_backends_errors — auto with no backend features returns
    //     ArchitectureNotSupported so the error message is actionable.
    #[test]
    #[cfg(not(any(feature = "mistralrs-backend", feature = "candle-backend")))]
    fn auto_no_backends_errors() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "auto");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::ArchitectureNotSupported(_)),
            "expected ArchitectureNotSupported, got {:?}", err
        );
    }

    // 8. auto_prefers_mistralrs — auto with mistralrs feature → "mistralrs"
    #[test]
    #[cfg(feature = "mistralrs-backend")]
    fn auto_prefers_mistralrs() {
        let c = cfg_with_path_and_backend("/tmp/model.gguf", "auto");
        let (backend, _) = select_backend(&c).unwrap();
        assert_eq!(backend, "mistralrs");
    }

    // 9. relative_gguf_ok — relative .gguf path is accepted
    #[test]
    fn relative_gguf_ok() {
        let c = cfg_with_path("models/llama.gguf");
        let result = select_backend(&c);
        assert!(result.is_ok(), "relative .gguf should be accepted");
    }

    // 10. no_extension_err — path with no extension → ModelLoad error
    #[test]
    fn no_extension_err() {
        let c = cfg_with_path("/tmp/model");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::ModelLoad(_)),
            "expected ModelLoad, got {:?}",
            err
        );
    }

    // 11. empty_stop_sequences_ok — default params (empty stop_sequences) don't affect selection
    #[test]
    fn empty_stop_sequences_ok() {
        let mut c = cfg_with_path("/tmp/model.gguf");
        c.params.stop_sequences = vec![];
        let result = select_backend(&c);
        assert!(result.is_ok(), "empty stop_sequences should not cause error");
    }

    // 12. path_without_gguf_err — path ending in .bin → ModelLoad error
    #[test]
    fn path_without_gguf_err() {
        let c = cfg_with_path("/tmp/model.bin");
        let err = select_backend(&c).unwrap_err();
        assert!(
            matches!(err, GwenError::ModelLoad(_)),
            "expected ModelLoad, got {:?}",
            err
        );
    }
}
