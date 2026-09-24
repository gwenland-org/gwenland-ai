//! Session metadata: the identifying header of a benchmark run.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::core::mode::ENSessionMode;
use crate::core::schema::{field_f64, field_str, FromJson, ToJson, GLBENCH_VERSION, SCHEMA_VERSION};
use crate::export::json::Json;

/// Whether the headline throughput came from the production path or from a
/// deliberately instrumented diagnostic run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENMeasurementMode {
    /// No engine-side timing instrumentation was active. Throughput can be
    /// used as production evidence when the rest of the benchmark gate passes.
    Production,
    /// Engine-side timing instrumentation was active. Stage timing is useful,
    /// but the headline throughput is diagnostic and not production evidence.
    Instrumented,
    /// A legacy archive did not record whether instrumentation was active.
    Unknown,
}

impl ENMeasurementMode {
    /// Stable identifier used in the archive and human-readable reports.
    pub fn as_str(self) -> &'static str {
        match self {
            ENMeasurementMode::Production => "production",
            ENMeasurementMode::Instrumented => "instrumented",
            ENMeasurementMode::Unknown => "unknown",
        }
    }

    fn from_str(value: &str) -> Option<ENMeasurementMode> {
        match value {
            "production" => Some(ENMeasurementMode::Production),
            "instrumented" => Some(ENMeasurementMode::Instrumented),
            "unknown" => Some(ENMeasurementMode::Unknown),
            _ => None,
        }
    }
}

/// Reproducibility facts for the engine configuration that selected a kernel
/// path. This fingerprints configuration, not the resolved kernel suite:
/// hardware-dependent dispatch still belongs in engine telemetry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLDispatchProvenance {
    /// Self-describing hash of the engine name, glbench version, and recorded
    /// engine overrides.
    pub config_fingerprint: String,
    /// Whitelisted engine environment overrides, sorted by name. The explicit
    /// allowlist avoids leaking unrelated process environment into archives.
    pub engine_overrides: Vec<(String, String)>,
}

impl ToJson for VLDispatchProvenance {
    fn to_json(&self) -> Json {
        Json::obj([
            ("config_fingerprint", Json::s(self.config_fingerprint.clone())),
            (
                "engine_overrides",
                Json::Arr(
                    self.engine_overrides
                        .iter()
                        .map(|(name, value)| {
                            Json::obj([
                                ("name", Json::s(name.clone())),
                                ("value", Json::s(value.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
        ])
    }
}

impl FromJson for VLDispatchProvenance {
    fn from_json(v: &Json) -> Result<Self, String> {
        let overrides = v
            .get("engine_overrides")
            .and_then(Json::as_arr)
            .ok_or_else(|| "dispatch.engine_overrides is not an array".to_string())?;
        let mut engine_overrides = Vec::with_capacity(overrides.len());
        for (index, entry) in overrides.iter().enumerate() {
            let name = entry
                .get("name")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("dispatch.engine_overrides[{index}].name is not a string"))?;
            let value = entry
                .get("value")
                .and_then(Json::as_str)
                .ok_or_else(|| format!("dispatch.engine_overrides[{index}].value is not a string"))?;
            engine_overrides.push((name.to_string(), value.to_string()));
        }
        Ok(VLDispatchProvenance {
            config_fingerprint: field_str(v, "config_fingerprint")?,
            engine_overrides,
        })
    }
}

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
    /// What this session measured (v3). A v1 archive has no such field and is
    /// read as [`ENSessionMode::InferenceOnly`] (D-20).
    pub session_mode: ENSessionMode,
    /// Which machine produced it, when the operator chose to record one.
    ///
    /// `None` by default and never auto-probed: a hostname is identifying
    /// information, and `glbench` archives are files users hand to each other.
    /// `EnvironmentSnapshot` already carries everything analysis needs about
    /// the machine without naming it.
    pub host_identifier: Option<String>,
    /// Which optional collection passes ran, e.g. `"bits+weights"`. `None` when
    /// only the default measurements were taken.
    pub collection_profile: Option<String>,
    /// Whether throughput is production evidence or diagnostic-only. New runs
    /// always set this explicitly; legacy archives read as `Unknown`.
    pub measurement_mode: ENMeasurementMode,
    /// Minimum dispatch provenance. `None` only for legacy/programmatically
    /// assembled sessions that did not pass through the engine adapter.
    pub dispatch: Option<VLDispatchProvenance>,
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
            session_mode: ENSessionMode::InferenceOnly,
            host_identifier: None,
            collection_profile: None,
            measurement_mode: ENMeasurementMode::Production,
            dispatch: None,
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
            ("session_mode", Json::s(self.session_mode.as_str())),
            (
                "host_identifier",
                self.host_identifier.clone().map(Json::s).unwrap_or(Json::Null),
            ),
            (
                "collection_profile",
                self.collection_profile.clone().map(Json::s).unwrap_or(Json::Null),
            ),
            ("measurement_mode", Json::s(self.measurement_mode.as_str())),
            (
                "dispatch",
                self.dispatch.as_ref().map(ToJson::to_json).unwrap_or(Json::Null),
            ),
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
            // Absent in a v1 archive. Defaulting rather than erroring is the
            // whole of D-20's reader-compatibility promise; an unrecognised
            // value is a different matter and is rejected.
            session_mode: match v.get("session_mode").and_then(|m| m.as_str()) {
                None => ENSessionMode::InferenceOnly,
                Some(s) => ENSessionMode::from_str(s)
                    .ok_or_else(|| format!("unknown session_mode '{s}'"))?,
            },
            host_identifier: v.get("host_identifier").and_then(|h| h.as_str()).map(String::from),
            collection_profile: v
                .get("collection_profile")
                .and_then(|c| c.as_str())
                .map(String::from),
            measurement_mode: match v.get("measurement_mode").and_then(Json::as_str) {
                None => ENMeasurementMode::Unknown,
                Some(value) => ENMeasurementMode::from_str(value)
                    .ok_or_else(|| format!("unknown measurement_mode '{value}'"))?,
            },
            dispatch: match v.get("dispatch") {
                Some(value) if !matches!(value, Json::Null) => {
                    Some(VLDispatchProvenance::from_json(value)?)
                }
                _ => None,
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_metadata_is_explicitly_production() {
        assert_eq!(SessionMetadata::new("run").measurement_mode, ENMeasurementMode::Production);
    }

    #[test]
    fn legacy_metadata_reads_as_unknown_not_production() {
        let Json::Obj(mut value) = SessionMetadata::new("run").to_json() else {
            unreachable!()
        };
        value.remove("measurement_mode");
        let back = SessionMetadata::from_json(&Json::Obj(value)).unwrap();
        assert_eq!(back.measurement_mode, ENMeasurementMode::Unknown);
    }

    #[test]
    fn dispatch_provenance_round_trips() {
        let mut metadata = SessionMetadata::new("run");
        metadata.measurement_mode = ENMeasurementMode::Instrumented;
        metadata.dispatch = Some(VLDispatchProvenance {
            config_fingerprint: "sha256-128:0123".into(),
            engine_overrides: vec![("GLCUDA_TELEMETRY".into(), "1".into())],
        });

        let back = SessionMetadata::from_json(&metadata.to_json()).unwrap();
        assert_eq!(back.measurement_mode, ENMeasurementMode::Instrumented);
        assert_eq!(back.dispatch, metadata.dispatch);
    }
}
