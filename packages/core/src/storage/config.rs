// @INFO: Single source of truth for ~/.config/gwen/config.toml.
//        On first access, auto-migrates the legacy config.json if present.
// @DANGER: Never store hf_token here — use OS keyring (platform::hub_model).
// @EDITABLE: Add new fields to the appropriate section struct; serde defaults handle missing keys.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

// ── section structs ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct GeneralConfig {
    pub last_used_model: String,
    pub default_port: u16,
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            last_used_model: String::new(),
            default_port: 1136,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(default)]
pub struct AiConfig {
    pub compression: bool,
    pub token_budget: u32,
    pub strategy: String,
}

impl Default for AiConfig {
    fn default() -> Self {
        Self {
            compression: true,
            token_budget: 4096,
            strategy: "tfidf".to_string(),
        }
    }
}

/// [auth] section intentionally holds no token — keyring only.
/// Reserved for future non-secret auth preferences.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default)]
pub struct AuthConfig {}

// ── root config ───────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default)]
pub struct GwenConfig {
    pub general: GeneralConfig,
    pub ai: AiConfig,
    pub auth: AuthConfig,
}

impl GwenConfig {
    fn toml_path() -> std::path::PathBuf {
        crate::storage::paths::GwenPaths::config_file()
    }

    fn json_path() -> std::path::PathBuf {
        crate::storage::paths::GwenPaths::config_dir().join("config.json")
    }

    /// Load from disk.
    /// If config.toml is missing but config.json exists, migrates automatically.
    /// Returns default on missing/corrupt (never errors out).
    pub fn load() -> Self {
        let toml_path = Self::toml_path();

        // Auto-migrate legacy config.json → config.toml
        if !toml_path.exists() {
            let json_path = Self::json_path();
            if json_path.exists() {
                if let Ok(migrated) = Self::migrate_from_json(&json_path) {
                    if migrated.save().is_ok() {
                        let _ = std::fs::remove_file(&json_path);
                        eprintln!("✦ Config migrated to TOML format.");
                        return migrated;
                    }
                }
            }
            return Self::default();
        }

        std::fs::read_to_string(&toml_path)
            .ok()
            .and_then(|s| toml::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist to disk, creating parent dirs as needed.
    pub fn save(&self) -> Result<()> {
        let path = Self::toml_path();
        std::fs::create_dir_all(crate::storage::paths::GwenPaths::config_dir())
            .context("cannot create gwen config dir")?;
        let toml_str = toml::to_string_pretty(self).context("cannot serialise GwenConfig")?;
        std::fs::write(&path, toml_str).context("cannot write config.toml")?;
        Ok(())
    }

    /// Read a config value by dotted key (e.g. "general.last_used_model").
    /// Returns the value as a String (numbers and bools are coerced).
    pub fn get(&self, key: &str) -> Result<String> {
        match key {
            "general.last_used_model" => Ok(self.general.last_used_model.clone()),
            "general.default_port"    => Ok(self.general.default_port.to_string()),
            "ai.compression"          => Ok(self.ai.compression.to_string()),
            "ai.token_budget"         => Ok(self.ai.token_budget.to_string()),
            "ai.strategy"             => Ok(self.ai.strategy.clone()),
            _ => anyhow::bail!("unknown config key: {}", key),
        }
    }

    /// Write a config value by dotted key, parsing the string to the correct type.
    pub fn set(&mut self, key: &str, value: &str) -> Result<()> {
        match key {
            "general.last_used_model" => {
                self.general.last_used_model = value.to_string();
            }
            "general.default_port" => {
                self.general.default_port = value
                    .parse::<u16>()
                    .context("default_port must be a u16 (0–65535)")?;
            }
            "ai.compression" => {
                self.ai.compression = value
                    .parse::<bool>()
                    .context("compression must be true or false")?;
            }
            "ai.token_budget" => {
                self.ai.token_budget = value
                    .parse::<u32>()
                    .context("token_budget must be an unsigned integer")?;
            }
            "ai.strategy" => {
                self.ai.strategy = value.to_string();
            }
            _ => anyhow::bail!("unknown config key: {}", key),
        }
        Ok(())
    }

    // ── migration ─────────────────────────────────────────────────────────────

    fn migrate_from_json(json_path: &std::path::Path) -> Result<Self> {
        #[derive(serde::Deserialize, Default)]
        struct LegacyJson {
            #[serde(default)]
            last_used_model: String,
            #[serde(default)]
            default_port: Option<u16>,
        }

        let raw = std::fs::read_to_string(json_path)
            .context("cannot read legacy config.json")?;
        let legacy: LegacyJson = serde_json::from_str(&raw).unwrap_or_default();

        let mut cfg = Self::default();
        if !legacy.last_used_model.is_empty() {
            cfg.general.last_used_model = legacy.last_used_model;
        }
        if let Some(port) = legacy.default_port {
            cfg.general.default_port = port;
        }
        Ok(cfg)
    }
}

// ── convenience helpers (used by serve, chat, hub_model) ─────────────────────

/// Read just the last_used_model field. Returns None if empty.
pub fn read_last_used_model() -> Option<String> {
    let m = GwenConfig::load().general.last_used_model;
    if m.is_empty() { None } else { Some(m) }
}

/// Update only last_used_model, preserving other fields.
pub fn save_last_used_model(model_id: &str) -> Result<()> {
    let mut cfg = GwenConfig::load();
    cfg.general.last_used_model = model_id.to_string();
    cfg.save()
}

/// Convenience wrapper for get — loads config fresh each call.
pub fn get(key: &str) -> Result<String> {
    GwenConfig::load().get(key)
}

/// Convenience wrapper for set — loads, mutates, saves atomically.
pub fn set(key: &str, value: &str) -> Result<()> {
    let mut cfg = GwenConfig::load();
    cfg.set(key, value)?;
    cfg.save()
}
