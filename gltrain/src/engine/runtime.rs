// engine/runtime.rs — Native-only runtime descriptor.
//
// Cycle 6 removed all external runtime detection (Ollama, llama.cpp, LM Studio).
// GwenLand now runs inference directly via candle-transformers; no subprocess,
// no HTTP proxy to an external daemon, no Python. This module is kept as the
// single authoritative place that names the active execution backend so that
// call sites can be updated in one place if a new backend is ever added.

/// The active inference backend. Currently only `Native` (candle-transformers)
/// is supported. The enum exists to keep the dispatch surface open without
/// requiring callers to hard-code a string.
#[derive(Debug, Clone, PartialEq)]
pub enum DetectedRuntime {
    /// Pure-Rust inference via candle-transformers. No external process required.
    Native,
}

/// Always succeeds — native inference is always available because candle is a
/// compile-time dependency, not a runtime service that can be missing.
pub fn detect_runtime() -> DetectedRuntime {
    eprintln!("  ❖ runtime: native (candle-transformers)");
    DetectedRuntime::Native
}
