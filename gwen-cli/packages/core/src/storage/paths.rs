// @INFO: Single source of truth for ALL GwenLand filesystem paths.
//        Use GwenPaths::* everywhere. Never hardcode ~/.config, AppData, etc.
// @DANGER: DO NOT call dirs::* or directories::* from anywhere else in the codebase.
//          All path resolution must go through this module.

use directories::ProjectDirs;
use std::path::PathBuf;

// ── core resolver ─────────────────────────────────────────────────────────────

fn project_dirs() -> ProjectDirs {
    // qualifier="dev", organization="JinXSuper", application="gwen"
    // Linux   → ~/.config/gwen/  +  ~/.cache/gwen/
    // macOS   → ~/Library/Application Support/gwen/  +  ~/Library/Caches/gwen/
    // Windows → AppData\Roaming\gwen\  +  AppData\Local\gwen\cache\
    ProjectDirs::from("dev", "JinXSuper", "gwen")
        .expect("cannot determine platform config directories")
}

// ── GwenPaths ─────────────────────────────────────────────────────────────────

pub struct GwenPaths;

impl GwenPaths {
    /// Platform-correct config directory (e.g. ~/.config/gwen on Linux).
    pub fn config_dir() -> PathBuf {
        // Honour GWEN_HOME override for CI and portable installs.
        if let Ok(v) = std::env::var("GWEN_HOME") {
            return PathBuf::from(v);
        }
        project_dirs().config_dir().to_path_buf()
    }

    /// Path to the primary config file.
    pub fn config_file() -> PathBuf {
        Self::config_dir().join("config.toml")
    }

    /// Path to the model registry file.
    pub fn models_file() -> PathBuf {
        Self::config_dir().join("models.toml")
    }

    /// Directory where downloaded models are stored.
    pub fn models_dir() -> PathBuf {
        Self::config_dir().join("models")
    }

    /// Platform-correct cache directory.
    pub fn cache_dir() -> PathBuf {
        if let Ok(v) = std::env::var("GWEN_HOME") {
            return PathBuf::from(v).join("cache");
        }
        project_dirs().cache_dir().to_path_buf()
    }

    /// Directory for eval result JSON files (machine output — stays JSON).
    pub fn eval_results_dir() -> PathBuf {
        Self::config_dir().join("eval_results")
    }

    /// Temporary directory used during self-updates and partial downloads.
    pub fn tmp_dir() -> PathBuf {
        Self::cache_dir().join("tmp")
    }

    /// Chat history file.
    pub fn history_file() -> PathBuf {
        Self::config_dir().join("history.jsonl")
    }

    /// Session log directory.
    pub fn session_dir() -> PathBuf {
        Self::cache_dir().join("sessions")
    }
}

// ── backwards-compat shim ─────────────────────────────────────────────────────
// Called by diagnostics/doctor.rs and any module not yet migrated.
// Will be removed once all callers are updated.

#[inline]
pub fn gwen_config_dir() -> PathBuf {
    GwenPaths::config_dir()
}

/// Absolute path to the primary config file (TOML format).
#[inline]
pub fn config_toml_path() -> PathBuf {
    GwenPaths::config_file()
}

/// Absolute path to the model registry file (TOML format).
#[inline]
pub fn models_toml_path() -> PathBuf {
    GwenPaths::models_file()
}
