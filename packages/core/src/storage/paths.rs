// @INFO: Single source of truth for all GwenLand filesystem paths.
//        Use GwenPaths::* everywhere. Never hardcode ~/.config, AppData, etc.

use std::path::{Path, PathBuf};

const GWENLAND_DIR: &str = ".gwenland";
const GWEN_HOME_ENV: &str = "GWEN_HOME";

fn ensure_dir(path: &Path) {
    let _ = std::fs::create_dir_all(path);
}

fn home_root() -> PathBuf {
    if let Ok(v) = std::env::var(GWEN_HOME_ENV) {
        return PathBuf::from(v);
    }

    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(GWENLAND_DIR)
}

pub struct GwenPaths;

impl GwenPaths {
    /// Root GwenLand storage directory: ~/.gwenland/.
    pub fn root_dir() -> PathBuf {
        let path = home_root();
        ensure_dir(&path);
        path
    }

    /// User and engine configuration directory: ~/.gwenland/config/.
    pub fn config_dir() -> PathBuf {
        let path = Self::root_dir().join("config");
        ensure_dir(&path);
        path
    }

    /// Path to the primary JSON config file.
    pub fn config_file() -> PathBuf {
        Self::config_dir().join("config.json")
    }

    /// Directory where downloaded models and the model registry are stored.
    pub fn models_dir() -> PathBuf {
        let path = Self::root_dir().join("models");
        ensure_dir(&path);
        path
    }

    /// Path to the model registry file.
    pub fn models_file() -> PathBuf {
        Self::models_dir().join("models.json")
    }

    /// Directory for human-readable crash reports.
    pub fn crash_logs_dir() -> PathBuf {
        let path = Self::root_dir().join("crash-logs");
        ensure_dir(&path);
        path
    }

    /// Directory for internal cache artifacts.
    pub fn cache_dir() -> PathBuf {
        let path = Self::root_dir().join("cache");
        ensure_dir(&path);
        path
    }

    /// Directory for eval result JSON files.
    pub fn eval_results_dir() -> PathBuf {
        let path = Self::root_dir().join("eval_results");
        ensure_dir(&path);
        path
    }

    /// Temporary directory used during self-updates and partial downloads.
    pub fn tmp_dir() -> PathBuf {
        let path = Self::cache_dir().join("tmp");
        ensure_dir(&path);
        path
    }

    /// Chat history file.
    pub fn history_file() -> PathBuf {
        Self::root_dir().join("history.jsonl")
    }

    /// Session log directory.
    pub fn session_dir() -> PathBuf {
        let path = Self::root_dir().join("sessions");
        ensure_dir(&path);
        path
    }
}

#[inline]
pub fn root_dir() -> PathBuf {
    GwenPaths::root_dir()
}

#[inline]
pub fn config_dir() -> PathBuf {
    GwenPaths::config_dir()
}

#[inline]
pub fn models_dir() -> PathBuf {
    GwenPaths::models_dir()
}

#[inline]
pub fn crash_logs_dir() -> PathBuf {
    GwenPaths::crash_logs_dir()
}

#[inline]
pub fn gwen_config_dir() -> PathBuf {
    GwenPaths::config_dir()
}

#[inline]
pub fn config_json_path() -> PathBuf {
    GwenPaths::config_file()
}

#[inline]
pub fn models_json_path() -> PathBuf {
    GwenPaths::models_file()
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::ffi::OsString;
    use std::path::Path;
    use std::sync::{Mutex, MutexGuard};

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    pub(crate) struct GwenHomeGuard {
        old: Option<OsString>,
        _lock: MutexGuard<'static, ()>,
    }

    impl Drop for GwenHomeGuard {
        fn drop(&mut self) {
            if let Some(old) = self.old.take() {
                unsafe {
                    std::env::set_var(super::GWEN_HOME_ENV, old);
                }
            } else {
                unsafe {
                    std::env::remove_var(super::GWEN_HOME_ENV);
                }
            }
        }
    }

    pub(crate) fn set_gwen_home(path: &Path) -> GwenHomeGuard {
        let lock = ENV_LOCK.lock().expect("GWEN_HOME test lock poisoned");
        let old = std::env::var_os(super::GWEN_HOME_ENV);
        unsafe {
            std::env::set_var(super::GWEN_HOME_ENV, path);
        }
        GwenHomeGuard { old, _lock: lock }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_dirs_resolve_under_gwenland_root() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = test_support::set_gwen_home(temp.path());

        let root = GwenPaths::root_dir();
        assert_eq!(root, temp.path());
        assert_eq!(GwenPaths::config_dir(), root.join("config"));
        assert_eq!(GwenPaths::models_dir(), root.join("models"));
        assert_eq!(GwenPaths::crash_logs_dir(), root.join("crash-logs"));

        assert!(!GwenPaths::config_dir().to_string_lossy().contains(".config/gwen"));
        assert!(!GwenPaths::models_dir().to_string_lossy().contains(".config/gwen"));
        assert!(!GwenPaths::crash_logs_dir().to_string_lossy().contains(".config/gwen"));
    }

    #[test]
    fn path_dirs_are_created_on_first_call() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = test_support::set_gwen_home(temp.path());

        let config = GwenPaths::config_dir();
        let models = GwenPaths::models_dir();
        let crash_logs = GwenPaths::crash_logs_dir();

        assert!(config.is_dir());
        assert!(models.is_dir());
        assert!(crash_logs.is_dir());
    }

    #[test]
    fn file_paths_use_new_layout() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = test_support::set_gwen_home(temp.path());

        assert_eq!(
            GwenPaths::config_file(),
            temp.path().join("config").join("config.json")
        );
        assert_eq!(
            GwenPaths::models_file(),
            temp.path().join("models").join("models.json")
        );
    }
}
