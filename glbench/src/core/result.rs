//! Session metadata: the identifying header of a benchmark run.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::schema::{field_f64, field_str, FromJson, ToJson, GLBENCH_VERSION, SCHEMA_VERSION};
use crate::export::json::Json;

/// Identifying facts about a session: what it is called, when it ran, and which
/// tool/schema produced it.
#[derive(Debug, Clone)]
pub struct SessionMetadata {
    /// Human label, e.g. `"qwen7b-glcuda-q8"`. Defaults from engine+model.
    pub label: String,
    /// Unix epoch seconds when the session was created.
    pub created_unix: u64,
    /// glbench version that produced this session.
    pub glbench_version: String,
    /// Archive schema version.
    pub schema_version: u32,
}

impl SessionMetadata {
    /// Build metadata stamped with the current time and tool versions.
    pub fn new(label: impl Into<String>) -> SessionMetadata {
        SessionMetadata {
            label: label.into(),
            created_unix: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            glbench_version: GLBENCH_VERSION.to_string(),
            schema_version: SCHEMA_VERSION,
        }
    }
}

impl ToJson for SessionMetadata {
    fn to_json(&self) -> Json {
        Json::obj([
            ("label", Json::s(self.label.clone())),
            ("created_unix", Json::n(self.created_unix as f64)),
            ("glbench_version", Json::s(self.glbench_version.clone())),
            ("schema_version", Json::n(self.schema_version as f64)),
        ])
    }
}

impl FromJson for SessionMetadata {
    fn from_json(v: &Json) -> Result<Self, String> {
        Ok(SessionMetadata {
            label: field_str(v, "label")?,
            created_unix: field_f64(v, "created_unix")? as u64,
            glbench_version: field_str(v, "glbench_version")?,
            schema_version: field_f64(v, "schema_version")? as u32,
        })
    }
}
