// @INFO: Model registry persisted under ~/.gwenland/models/.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelEntry {
    pub id: String,
    pub source: String,
    pub format: String,
    pub quant: String,
    pub size_bytes: u64,
    pub downloaded_at: String,
    pub sha256: String,
    pub path: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
struct RegistryFile {
    version: u32,
    models: Vec<ModelEntry>,
}

pub struct ModelRegistry {
    models: Vec<ModelEntry>,
    registry_path: PathBuf,
}

impl ModelRegistry {
    /// Load from ~/.gwenland/models/models.json. Missing means empty registry.
    pub fn load() -> Result<Self> {
        let registry_path = super::paths::GwenPaths::models_dir().join("models.json");

        if !registry_path.exists() {
            return Ok(Self {
                models: Vec::new(),
                registry_path,
            });
        }

        match std::fs::read_to_string(&registry_path) {
            Ok(content) => match serde_json::from_str::<RegistryFile>(&content) {
                Ok(parsed) => Ok(Self {
                    models: parsed.models,
                    registry_path,
                }),
                Err(_) => Self::rebuild_from_disk(),
            },
            Err(_) => Self::rebuild_from_disk(),
        }
    }

    /// Persist to disk atomically (write to .tmp, then rename).
    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.registry_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file_data = RegistryFile {
            version: 1,
            models: self.models.clone(),
        };
        let json = serde_json::to_string_pretty(&file_data)?;

        let tmp_path = self.registry_path.with_extension("json.tmp");
        std::fs::write(&tmp_path, json)?;
        std::fs::rename(&tmp_path, &self.registry_path)?;
        Ok(())
    }

    /// Add or update a model entry.
    pub fn upsert(&mut self, entry: ModelEntry) {
        if let Some(existing) = self.models.iter_mut().find(|m| m.id == entry.id) {
            *existing = entry;
        } else {
            self.models.push(entry);
        }
    }

    /// Remove by id. Returns true if found and removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let initial_len = self.models.len();
        self.models.retain(|m| m.id != id);
        self.models.len() < initial_len
    }

    /// Find by id or by source (e.g. "mistralai/Mistral-7B-v0.1").
    pub fn find(&self, query: &str) -> Option<&ModelEntry> {
        self.models
            .iter()
            .find(|m| m.id == query || m.source == query)
    }

    /// List all entries.
    pub fn list(&self) -> &[ModelEntry] {
        &self.models
    }

    /// Rebuild registry by scanning models/ directory for metadata.json files.
    pub fn rebuild_from_disk() -> Result<Self> {
        let registry_path = super::paths::GwenPaths::models_dir().join("models.json");
        let models_dir = super::paths::GwenPaths::models_dir();

        let mut models = Vec::new();

        if models_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&models_dir) {
                for entry_result in entries {
                    if let Ok(entry) = entry_result {
                        let path = entry.path();
                        if path.is_dir() {
                            let metadata_path = path.join("metadata.json");
                            if metadata_path.exists() {
                                if let Ok(content) = std::fs::read_to_string(&metadata_path) {
                                    if let Ok(metadata) =
                                        serde_json::from_str::<serde_json::Value>(&content)
                                    {
                                        let id = path
                                            .file_name()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .to_string();

                                        if id.starts_with('.') {
                                            continue;
                                        }

                                        let source = metadata["source"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        let format = metadata["format"]
                                            .as_str()
                                            .unwrap_or("gguf")
                                            .to_string();
                                        let quant = metadata["quant"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        let size_bytes = metadata["size_bytes"]
                                            .as_u64()
                                            .or_else(|| metadata["size"].as_u64())
                                            .unwrap_or(0);
                                        let downloaded_at = metadata["downloaded_at"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        let sha256 = metadata["sha256"]
                                            .as_str()
                                            .unwrap_or("")
                                            .to_string();
                                        let model_gguf_path = path.join("model.gguf");

                                        models.push(ModelEntry {
                                            id,
                                            source,
                                            format,
                                            quant,
                                            size_bytes,
                                            downloaded_at,
                                            sha256,
                                            path: model_gguf_path,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!("models.json was corrupt; rebuilt from disk");

        Ok(Self {
            models,
            registry_path,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_entry(path: PathBuf) -> ModelEntry {
        ModelEntry {
            id: "qwen3".to_string(),
            source: "Qwen/Qwen3".to_string(),
            format: "gguf".to_string(),
            quant: "q4_k_m".to_string(),
            size_bytes: 123,
            downloaded_at: "2026-06-16T14:32:07+07:00".to_string(),
            sha256: "abc".to_string(),
            path,
        }
    }

    #[test]
    fn registry_round_trips_through_gwenland_models_dir() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = crate::storage::paths::test_support::set_gwen_home(temp.path());

        let model_path = crate::storage::paths::GwenPaths::models_dir().join("qwen3.gguf");
        let mut registry = ModelRegistry::load().unwrap();
        registry.upsert(sample_entry(model_path.clone()));
        registry.save().unwrap();

        let registry_path = temp.path().join("models").join("models.json");
        assert_eq!(registry.registry_path, registry_path);
        assert!(registry_path.exists());
        assert!(!registry_path.to_string_lossy().contains(".config/gwen"));

        let loaded = ModelRegistry::load().unwrap();
        let entry = loaded.find("qwen3").unwrap();
        assert_eq!(entry.path, model_path);
        assert_eq!(entry.source, "Qwen/Qwen3");
    }

    #[test]
    fn old_xdg_registry_is_not_migrated_or_removed() {
        let temp = tempfile::tempdir().unwrap();
        let _guard = crate::storage::paths::test_support::set_gwen_home(temp.path());
        let old_root = temp.path().join(".config").join("gwen");
        std::fs::create_dir_all(&old_root).unwrap();
        let old_registry = old_root.join("models.json");
        std::fs::write(&old_registry, r#"{"version":1,"models":[]}"#).unwrap();

        let registry = ModelRegistry::load().unwrap();
        assert!(registry.list().is_empty());
        assert!(old_registry.exists());
    }
}
