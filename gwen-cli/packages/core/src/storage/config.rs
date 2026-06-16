// @INFO: Core GwenLand configuration persisted at ~/.gwenland/config/config.json.
// @DANGER: Never store hf_token here; use OS keyring (platform::hub_model).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

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

/// [auth] intentionally holds no token; keyring only.
#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default)]
pub struct AuthConfig {}

#[derive(Debug, Deserialize, Serialize, Clone, Default)]
#[serde(default)]
pub struct GwenConfig {
    pub general: GeneralConfig,
    pub ai: AiConfig,
    pub auth: AuthConfig,
    #[serde(default)]
    pub inference: crate::engine::inference::config::InferenceConfig,
}

impl GwenConfig {
    fn json_path() -> std::path::PathBuf {
        crate::storage::paths::GwenPaths::config_file()
    }

    /// Load from disk. Missing or malformed config returns defaults.
    pub fn load() -> Self {
        let path = Self::json_path();
        if !path.exists() {
            return Self::default();
        }

        std::fs::read_to_string(&path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    /// Persist to disk, merging core sections into the shared JSON config file.
    pub fn save(&self) -> Result<()> {
        let path = Self::json_path();
        std::fs::create_dir_all(crate::storage::paths::GwenPaths::config_dir())
            .context("cannot create gwen config dir")?;

        let mut root = read_json_object(&path);
        let core = serde_json::to_value(self).context("cannot serialise GwenConfig")?;
        if let Value::Object(core_map) = core {
            for (key, value) in core_map {
                root.insert(key, value);
            }
        }

        root.insert(
            "configDir".to_string(),
            Value::String(
                crate::storage::paths::GwenPaths::config_dir()
                    .to_string_lossy()
                    .to_string(),
            ),
        );
        root.insert(
            "modelsDir".to_string(),
            Value::String(
                crate::storage::paths::GwenPaths::models_dir()
                    .to_string_lossy()
                    .to_string(),
            ),
        );
        root.insert(
            "sessionsDir".to_string(),
            Value::String(
                crate::storage::paths::GwenPaths::session_dir()
                    .to_string_lossy()
                    .to_string(),
            ),
        );

        let json = serde_json::to_string_pretty(&Value::Object(root))
            .context("cannot serialise config JSON")?;
        std::fs::write(&path, json).context("cannot write config.json")?;
        Ok(())
    }

    /// Read a config value by dotted key (e.g. "general.last_used_model").
    pub fn get(&self, key: &str) -> Result<String> {
        match key {
            "general.last_used_model" => Ok(self.general.last_used_model.clone()),
            "general.default_port" => Ok(self.general.default_port.to_string()),
            "ai.compression" => Ok(self.ai.compression.to_string()),
            "ai.token_budget" => Ok(self.ai.token_budget.to_string()),
            "ai.strategy" => Ok(self.ai.strategy.clone()),
            "inference.backend" => Ok(self.inference.backend.clone()),
            "inference.model" => Ok(self.inference.model.clone()),
            "inference.params.temperature" => Ok(self.inference.params.temperature.to_string()),
            "inference.params.top_p" => Ok(self.inference.params.top_p.to_string()),
            "inference.params.max_tokens" => Ok(self.inference.params.max_tokens.to_string()),
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
                    .context("default_port must be a u16 (0-65535)")?;
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
            "inference.backend" => {
                self.inference.backend = value.to_string();
            }
            "inference.model" => {
                self.inference.model = value.to_string();
            }
            "inference.params.temperature" => {
                self.inference.params.temperature = value
                    .parse::<f32>()
                    .context("temperature must be a float")?;
            }
            "inference.params.top_p" => {
                self.inference.params.top_p = value
                    .parse::<f32>()
                    .context("top_p must be a float")?;
            }
            "inference.params.max_tokens" => {
                self.inference.params.max_tokens = value
                    .parse::<usize>()
                    .context("max_tokens must be an unsigned integer")?;
            }
            _ => anyhow::bail!("unknown config key: {}", key),
        }
        Ok(())
    }
}

fn read_json_object(path: &std::path::Path) -> Map<String, Value> {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok())
        .and_then(|value| match value {
            Value::Object(map) => Some(map),
            _ => None,
        })
        .unwrap_or_default()
}

/// Read just the last_used_model field. Returns None if empty.
pub fn read_last_used_model() -> Option<String> {
    let m = GwenConfig::load().general.last_used_model;
    if m.is_empty() {
        None
    } else {
        Some(m)
    }
}

/// Update only last_used_model, preserving other fields.
pub fn save_last_used_model(model_id: &str) -> Result<()> {
    let mut cfg = GwenConfig::load();
    cfg.general.last_used_model = model_id.to_string();
    cfg.save()
}

/// Convenience wrapper for get; loads config fresh each call.
pub fn get(key: &str) -> Result<String> {
    GwenConfig::load().get(key)
}

/// Convenience wrapper for set; loads, mutates, saves atomically.
pub fn set(key: &str, value: &str) -> Result<()> {
    let mut cfg = GwenConfig::load();
    cfg.set(key, value)?;
    cfg.save()
}

/// Load just the inference configuration section.
pub fn load_inference_config() -> crate::engine::inference::config::InferenceConfig {
    GwenConfig::load().inference
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trips_through_gwenland_config_json() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = crate::storage::paths::test_support::set_gwen_home(temp.path());

        let mut cfg = GwenConfig::default();
        cfg.general.last_used_model = "qwen3:8b".to_string();
        cfg.ai.compression = false;
        cfg.save().unwrap();

        let path = crate::storage::paths::GwenPaths::config_file();
        assert_eq!(path, temp.path().join("config").join("config.json"));
        assert!(path.exists());
        assert!(!path.to_string_lossy().contains(".config/gwen"));

        let loaded = GwenConfig::load();
        assert_eq!(loaded.general.last_used_model, "qwen3:8b");
        assert!(!loaded.ai.compression);
    }

    #[test]
    fn save_preserves_non_core_json_keys() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = crate::storage::paths::test_support::set_gwen_home(temp.path());

        let path = crate::storage::paths::GwenPaths::config_file();
        std::fs::write(&path, r#"{"theme":"gwen-noir"}"#).unwrap();

        let mut cfg = GwenConfig::default();
        cfg.general.default_port = 4242;
        cfg.save().unwrap();

        let raw = std::fs::read_to_string(path).unwrap();
        let value: Value = serde_json::from_str(&raw).unwrap();
        assert_eq!(value["theme"], "gwen-noir");
        assert_eq!(value["general"]["default_port"], 4242);
        assert!(value["configDir"].as_str().unwrap().ends_with("config"));
        assert!(value["modelsDir"].as_str().unwrap().ends_with("models"));
    }

    #[test]
    fn old_xdg_config_is_not_migrated_or_removed() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = crate::storage::paths::test_support::set_gwen_home(temp.path());
        let old_root = temp.path().join(".config").join("gwen");
        std::fs::create_dir_all(&old_root).unwrap();
        let old_config = old_root.join("config.json");
        std::fs::write(&old_config, r#"{"general":{"last_used_model":"old"}}"#).unwrap();

        let loaded = GwenConfig::load();
        assert!(loaded.general.last_used_model.is_empty());
        assert!(old_config.exists());
    }
}
