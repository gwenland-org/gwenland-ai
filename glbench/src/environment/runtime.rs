//! Runtime/build facts about the glbench process itself, for reproducibility.

/// Facts about the running glbench build and host OS.
#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    /// Target OS (`std::env::consts::OS`).
    pub os: String,
    /// Target architecture (`std::env::consts::ARCH`).
    pub arch: String,
    /// glbench version string.
    pub glbench_version: String,
    /// Whether this is a debug or release build (debug assertions on = debug).
    pub build_profile: &'static str,
}

impl RuntimeInfo {
    /// Probe the current process.
    pub fn probe() -> RuntimeInfo {
        RuntimeInfo {
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
            glbench_version: crate::core::schema::GLBENCH_VERSION.to_string(),
            build_profile: if cfg!(debug_assertions) { "debug" } else { "release" },
        }
    }
}
