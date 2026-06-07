// @INFO: Model Registry — persisted to ~/.config/gwen/models.toml.
//        On first load, auto-migrates legacy models.json if present.
// @EDITABLE: Add new fields to ModelEntry as needed.

use std::path::PathBuf;
use serde::{Serialize, Deserialize};
use anyhow::Result;

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
    /// Load from disk. If models.toml missing but models.json exists, migrates.
    /// Empty registry on file missing; attempts rebuild from disk on corrupt.
    pub fn load() -> Result<Self> {
        let registry_path = super::paths::GwenPaths::models_file();
        let json_path = super::paths::GwenPaths::config_dir().join("models.json");

        // Auto-migrate legacy models.json → models.toml
        if !registry_path.exists() && json_path.exists() {
            if let Ok(migrated) = Self::migrate_from_json(&json_path, &registry_path) {
                eprintln!("✦ Model registry migrated to TOML format.");
                return Ok(migrated);
            }
        }

        if !registry_path.exists() {
            return Ok(Self {
                models: Vec::new(),
                registry_path,
            });
        }

        match std::fs::read_to_string(&registry_path) {
            Ok(content) => {
                match toml::from_str::<RegistryFile>(&content) {
                    Ok(parsed) => Ok(Self {
                        models: parsed.models,
                        registry_path,
                    }),
                    Err(_) => Self::rebuild_from_disk(),
                }
            }
            Err(_) => Self::rebuild_from_disk(),
        }
    }

    /// Persist to disk atomically (write to .tmp, then rename).
    pub fn save(&self) -> Result<()> {
        let file_data = RegistryFile {
            version: 1,
            models: self.models.clone(),
        };
        let toml_str = toml::to_string_pretty(&file_data)?;

        let tmp_path = self.registry_path.with_extension("toml.tmp");
        std::fs::write(&tmp_path, toml_str)?;
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
        self.models.iter().find(|m| m.id == query || m.source == query)
    }

    /// List all entries.
    pub fn list(&self) -> &[ModelEntry] {
        &self.models
    }

    /// Rebuild registry by scanning models/ directory for metadata.json files.
    pub fn rebuild_from_disk() -> Result<Self> {
        let registry_path = super::paths::GwenPaths::models_file();
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
                                    if let Ok(metadata) = serde_json::from_str::<serde_json::Value>(&content) {
                                        let id = path.file_name().unwrap_or_default().to_string_lossy().to_string();

                                        if id.starts_with('.') {
                                            continue;
                                        }

                                        let source = metadata["source"].as_str().unwrap_or("").to_string();
                                        let format = metadata["format"].as_str().unwrap_or("gguf").to_string();
                                        let quant = metadata["quant"].as_str().unwrap_or("").to_string();
                                        let size_bytes = metadata["size_bytes"].as_u64()
                                            .or_else(|| metadata["size"].as_u64())
                                            .unwrap_or(0);
                                        let downloaded_at = metadata["downloaded_at"].as_str().unwrap_or("").to_string();
                                        let sha256 = metadata["sha256"].as_str().unwrap_or("").to_string();
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

        eprintln!("models.toml was corrupt — rebuilt from disk");

        Ok(Self {
            models,
            registry_path,
        })
    }

    // ── migration ─────────────────────────────────────────────────────────────

    fn migrate_from_json(json_path: &std::path::Path, toml_path: &std::path::Path) -> Result<Self> {
        let content = std::fs::read_to_string(json_path)?;

        // Legacy models.json was either a bare Vec<Value> (fetch.rs wrote it)
        // or the structured RegistryFile. Try structured first.
        let models: Vec<ModelEntry> = if let Ok(rf) = serde_json::from_str::<RegistryFile>(&content) {
            rf.models
        } else if let Ok(entries) = serde_json::from_str::<Vec<serde_json::Value>>(&content) {
            // Bare array written by old fetch.rs — best-effort parse.
            entries.into_iter().filter_map(|v| {
                Some(ModelEntry {
                    id: v["model"].as_str()?.to_string(),
                    source: v["model"].as_str().unwrap_or("").to_string(),
                    format: "gguf".to_string(),
                    quant: String::new(),
                    size_bytes: 0,
                    downloaded_at: String::new(),
                    sha256: String::new(),
                    path: PathBuf::from(v["path"].as_str().unwrap_or("")),
                })
            }).collect()
        } else {
            Vec::new()
        };

        if let Some(parent) = toml_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let registry = Self { models, registry_path: toml_path.to_path_buf() };
        registry.save()?;
        // Remove legacy file after successful write
        let _ = std::fs::remove_file(json_path);
        Ok(registry)
    }
}
