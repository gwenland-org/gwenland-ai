//! Stummañ Pik: the checkpoint manifest and its validator.
//!
//! A checkpoint is a directory of independent segments under one manifest, so
//! the manifest is the file that says what the other files are. It carries the
//! format version, the adapter's configuration, and the training step.
//!
//! # Major is an error, minor is a warning
//!
//! Copied from `glictus-caliburni/src/manifest/metadata.rs`'s
//! `is_compatible_with`, which settled the same question: a different major
//! version means the layout changed and the file cannot be read; a different
//! minor means a field may be missing but the layout holds. Getting these the
//! wrong way round would either refuse readable files or accept unreadable
//! ones, and only one of those fails loudly.

use crate::error::{GlTrainError, Result};
use crate::nn::adapter::lora::VLLoraConfig;

use super::json::{parse, Json};

/// The format version this build writes.
pub const FORMAT_VERSION: VLFormatVersion = VLFormatVersion { major: 1, minor: 0 };

/// A `major.minor` checkpoint format version.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VLFormatVersion {
    /// Layout-breaking changes.
    pub major: u32,
    /// Additive changes.
    pub minor: u32,
}

impl VLFormatVersion {
    /// A version.
    pub fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }

    /// How a checkpoint at this version relates to `runtime`.
    pub fn is_compatible_with(&self, runtime: &VLFormatVersion) -> ENVersionCompatibility {
        if self.major != runtime.major {
            ENVersionCompatibility::Incompatible
        } else if self.minor != runtime.minor {
            ENVersionCompatibility::MinorMismatch
        } else {
            ENVersionCompatibility::Compatible
        }
    }
}

impl std::fmt::Display for VLFormatVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// Outcome of a checkpoint-vs-runtime format version check.
///
/// `EN` because a closed set of variants is this type's whole job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ENVersionCompatibility {
    /// Same major and minor.
    Compatible,
    /// Same major, different minor. Loadable, with a warning.
    MinorMismatch,
    /// Different major. Refuse.
    Incompatible,
}

/// Errors and warnings from one validation pass.
///
/// Follows `glictus-caliburni/src/manifest/validator.rs`'s `ValidationResult`
/// exactly. The split matters: a caller has to be able to tell "one key off"
/// from "wrong model entirely", and that is the difference between a warning
/// list and an error list.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct VLValidation {
    /// Fatal. The checkpoint cannot be loaded.
    pub errors: Vec<String>,
    /// Non-fatal. Loadable, but worth saying out loud.
    pub warnings: Vec<String>,
}

impl VLValidation {
    /// An empty result.
    pub fn new() -> Self {
        Self::default()
    }

    /// True when nothing fatal was found.
    pub fn ok(&self) -> bool {
        self.errors.is_empty()
    }

    /// True when anything non-fatal was found.
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }

    /// Record a fatal problem.
    pub fn error(&mut self, msg: impl Into<String>) {
        self.errors.push(msg.into());
    }

    /// Record a non-fatal problem.
    pub fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Collapse to a `Result`, keeping the warnings on the success path.
    ///
    /// Every error is reported at once, joined, rather than only the first:
    /// "three tensors missing and two transposed" and "one key off" are
    /// different situations and a caller should not have to re-run to find out
    /// which it has.
    pub fn into_result(self) -> Result<Vec<String>> {
        if self.errors.is_empty() {
            Ok(self.warnings)
        } else {
            Err(GlTrainError::Checkpoint(format!(
                "checkpoint validation failed with {} error(s):\n  {}",
                self.errors.len(),
                self.errors.join("\n  ")
            )))
        }
    }
}

/// What a checkpoint directory says about itself.
///
/// `VL` because it is a plain data bag with derived traits only.
#[derive(Debug, Clone, PartialEq)]
pub struct VLManifest {
    /// Format version of the files in this directory.
    pub format_version: VLFormatVersion,
    /// Registry id of the adapter, e.g. `"lora"`.
    pub adapter_type: String,
    /// Rank.
    pub r: usize,
    /// Alpha.
    pub alpha: f32,
    /// `alpha/sqrt(r)` instead of `alpha/r`.
    pub rslora: bool,
    /// Input dimension of the adapted layer.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,
    /// Training step this checkpoint was taken at.
    pub step: usize,
    /// Which base model this adapter was trained against, if known.
    ///
    /// `Option` because there is no honest default. An adapter merged onto the
    /// wrong base produces garbage with no error, so an absent id is a warning
    /// at validation time rather than something to invent.
    pub base_model_id: Option<String>,
    /// Whether `optimizer.safetensors` is present.
    pub has_optimizer_state: bool,
}

/// The file a manifest lives in, inside a checkpoint directory.
pub const MANIFEST_FILE: &str = "manifest.json";

impl VLManifest {
    /// A manifest describing a LoRA adapter at `step`.
    pub fn for_lora(config: &VLLoraConfig, step: usize) -> Self {
        Self {
            format_version: FORMAT_VERSION,
            adapter_type: "lora".to_string(),
            r: config.r,
            alpha: config.alpha,
            rslora: config.rslora,
            d_in: config.d_in,
            d_out: config.d_out,
            step,
            base_model_id: None,
            has_optimizer_state: false,
        }
    }

    /// The adapter config this manifest describes.
    pub fn lora_config(&self) -> VLLoraConfig {
        VLLoraConfig {
            r: self.r,
            alpha: self.alpha,
            rslora: self.rslora,
            d_in: self.d_in,
            d_out: self.d_out,
        }
    }

    /// Serialize to JSON text.
    pub fn to_json(&self) -> String {
        let mut fields = vec![
            (
                "format_version".to_string(),
                Json::s(self.format_version.to_string()),
            ),
            ("adapter_type".to_string(), Json::s(&*self.adapter_type)),
            ("r".to_string(), Json::n(self.r as f64)),
            ("alpha".to_string(), Json::n(self.alpha as f64)),
            ("rslora".to_string(), Json::Bool(self.rslora)),
            ("d_in".to_string(), Json::n(self.d_in as f64)),
            ("d_out".to_string(), Json::n(self.d_out as f64)),
            ("step".to_string(), Json::n(self.step as f64)),
            (
                "has_optimizer_state".to_string(),
                Json::Bool(self.has_optimizer_state),
            ),
        ];
        // Written as `null` rather than omitted, so a reader can tell "this
        // build did not record a base model" from "this field is from a newer
        // minor version I do not know about".
        fields.push((
            "base_model_id".to_string(),
            match &self.base_model_id {
                Some(id) => Json::s(&**id),
                None => Json::Null,
            },
        ));
        Json::Obj(fields.into_iter().collect()).to_compact()
    }

    /// Parse from JSON text.
    ///
    /// Every missing or wrong-typed required field is an error naming the
    /// field, never a default: a manifest missing `r` would otherwise load as
    /// rank 0 and fail much later inside the adapter constructor.
    pub fn from_json(text: &str) -> Result<Self> {
        let doc = parse(text)
            .map_err(|e| GlTrainError::Checkpoint(format!("{MANIFEST_FILE}: {e}")))?;

        let field = |k: &str| -> Result<&Json> {
            doc.get(k).ok_or_else(|| {
                GlTrainError::Checkpoint(format!("{MANIFEST_FILE}: missing field '{k}'"))
            })
        };
        let usize_field = |k: &str| -> Result<usize> {
            field(k)?.as_usize().ok_or_else(|| {
                GlTrainError::Checkpoint(format!(
                    "{MANIFEST_FILE}: field '{k}' is not a non-negative integer"
                ))
            })
        };
        let bool_field = |k: &str| -> Result<bool> {
            field(k)?.as_bool().ok_or_else(|| {
                GlTrainError::Checkpoint(format!("{MANIFEST_FILE}: field '{k}' is not a bool"))
            })
        };

        let version_text = field("format_version")?.as_str().ok_or_else(|| {
            GlTrainError::Checkpoint(format!(
                "{MANIFEST_FILE}: 'format_version' must be a \"major.minor\" string"
            ))
        })?;
        let format_version = parse_version(version_text)?;

        Ok(Self {
            format_version,
            adapter_type: field("adapter_type")?
                .as_str()
                .ok_or_else(|| {
                    GlTrainError::Checkpoint(format!(
                        "{MANIFEST_FILE}: 'adapter_type' is not a string"
                    ))
                })?
                .to_string(),
            r: usize_field("r")?,
            alpha: field("alpha")?.as_f64().ok_or_else(|| {
                GlTrainError::Checkpoint(format!("{MANIFEST_FILE}: 'alpha' is not a number"))
            })? as f32,
            rslora: bool_field("rslora")?,
            d_in: usize_field("d_in")?,
            d_out: usize_field("d_out")?,
            step: usize_field("step")?,
            // Optional fields tolerate absence: that is what makes adding one
            // a minor-version change rather than a major one.
            base_model_id: doc
                .get("base_model_id")
                .and_then(Json::as_str)
                .map(str::to_string),
            has_optimizer_state: doc
                .get("has_optimizer_state")
                .and_then(Json::as_bool)
                .unwrap_or(false),
        })
    }

    /// Check this manifest against what the caller expects.
    ///
    /// Collects **every** disagreement in one pass rather than returning at the
    /// first. A caller needs to distinguish "one field off" from "wrong model
    /// entirely", and that is only visible with the whole list in hand.
    pub fn validate_against(&self, expected: &VLManifest) -> VLValidation {
        let mut v = VLValidation::new();

        match self.format_version.is_compatible_with(&expected.format_version) {
            ENVersionCompatibility::Incompatible => v.error(format!(
                "format version {} cannot be read by a runtime at {}: the major version differs, \
                 so the layout is not the same",
                self.format_version, expected.format_version
            )),
            ENVersionCompatibility::MinorMismatch => v.warn(format!(
                "format version {} differs in minor from this runtime's {}: readable, but a field \
                 may be missing or unread",
                self.format_version, expected.format_version
            )),
            ENVersionCompatibility::Compatible => {}
        }

        if self.adapter_type != expected.adapter_type {
            v.error(format!(
                "adapter type is '{}', expected '{}'",
                self.adapter_type, expected.adapter_type
            ));
        }
        // Every geometry field is fatal on its own: each one changes the shape
        // of a stored tensor, so loading past it cannot produce a usable
        // adapter.
        for (name, got, want) in [
            ("r", self.r, expected.r),
            ("d_in", self.d_in, expected.d_in),
            ("d_out", self.d_out, expected.d_out),
        ] {
            if got != want {
                v.error(format!("{name} is {got}, expected {want}"));
            }
        }
        // Alpha and rslora change only the scale, so a mismatch is loadable but
        // gives a numerically different adapter. That is a warning, and a loud
        // one: the two are indistinguishable from the tensors alone.
        if (self.alpha - expected.alpha).abs() > f32::EPSILON {
            v.warn(format!(
                "alpha is {}, expected {}: the adapter will scale differently",
                self.alpha, expected.alpha
            ));
        }
        if self.rslora != expected.rslora {
            v.warn(format!(
                "rslora is {}, expected {}: scaling is alpha/{} either way, so the delta differs",
                self.rslora,
                expected.rslora,
                if self.rslora { "sqrt(r)" } else { "r" }
            ));
        }
        // An adapter merged onto the wrong base produces garbage with no error
        // anywhere, so an unknown base is worth saying out loud.
        match (&self.base_model_id, &expected.base_model_id) {
            (Some(got), Some(want)) if got != want => v.error(format!(
                "checkpoint was trained against base model '{got}', not '{want}'"
            )),
            (None, _) => v.warn(
                "checkpoint does not record which base model it was trained against; merging it \
                 onto the wrong one produces garbage with no error"
                    .to_string(),
            ),
            _ => {}
        }
        v
    }
}

fn parse_version(text: &str) -> Result<VLFormatVersion> {
    let bad = || {
        GlTrainError::Checkpoint(format!(
            "{MANIFEST_FILE}: 'format_version' must be \"major.minor\", got {text:?}"
        ))
    };
    let (major, minor) = text.split_once('.').ok_or_else(bad)?;
    Ok(VLFormatVersion {
        major: major.parse().map_err(|_| bad())?,
        minor: minor.parse().map_err(|_| bad())?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> VLManifest {
        VLManifest {
            format_version: FORMAT_VERSION,
            adapter_type: "lora".to_string(),
            r: 4,
            alpha: 8.0,
            rslora: false,
            d_in: 16,
            d_out: 32,
            step: 500,
            base_model_id: Some("qwen2.5-0.5b".to_string()),
            has_optimizer_state: true,
        }
    }

    #[test]
    fn a_manifest_round_trips_through_json() {
        let m = manifest();
        let back = VLManifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn a_manifest_without_a_base_model_id_round_trips() {
        let mut m = manifest();
        m.base_model_id = None;
        let back = VLManifest::from_json(&m.to_json()).unwrap();
        assert_eq!(back, m);
        assert!(back.base_model_id.is_none());
    }

    /// A manifest missing `r` must not load as rank 0 and fail much later
    /// inside the adapter constructor with an unrelated message.
    #[test]
    fn a_manifest_missing_a_required_field_names_that_field() {
        let mut m = manifest();
        m.r = 4;
        let text = m.to_json().replace(r#""r":4,"#, "");
        let err = VLManifest::from_json(&text).unwrap_err().to_string();
        assert!(err.contains("'r'"), "error should name the field: {err}");
    }

    #[test]
    fn a_manifest_with_a_wrong_typed_field_is_rejected() {
        let text = manifest().to_json().replace(r#""r":4"#, r#""r":"four""#);
        assert!(VLManifest::from_json(&text).is_err());
        let text = manifest()
            .to_json()
            .replace(r#""format_version":"1.0""#, r#""format_version":1"#);
        assert!(VLManifest::from_json(&text).is_err());
    }

    /// Major means the layout changed, so the file cannot be read. Minor means
    /// a field may be missing but the layout holds. Reversing these would
    /// either refuse readable files or accept unreadable ones.
    #[test]
    fn version_compatibility_makes_major_an_error_and_minor_a_warning() {
        let rt = VLFormatVersion::new(1, 0);
        assert_eq!(
            VLFormatVersion::new(1, 0).is_compatible_with(&rt),
            ENVersionCompatibility::Compatible
        );
        assert_eq!(
            VLFormatVersion::new(1, 3).is_compatible_with(&rt),
            ENVersionCompatibility::MinorMismatch
        );
        assert_eq!(
            VLFormatVersion::new(2, 0).is_compatible_with(&rt),
            ENVersionCompatibility::Incompatible
        );
    }

    #[test]
    fn validation_reports_a_major_version_difference_as_an_error() {
        let mut old = manifest();
        old.format_version = VLFormatVersion::new(2, 0);
        let v = old.validate_against(&manifest());
        assert!(!v.ok(), "a major mismatch must be fatal");
        assert!(v.errors.iter().any(|e| e.contains("major")), "{v:?}");
    }

    #[test]
    fn validation_reports_a_minor_version_difference_as_a_warning() {
        let mut newer = manifest();
        newer.format_version = VLFormatVersion::new(1, 7);
        let v = newer.validate_against(&manifest());
        assert!(v.ok(), "a minor mismatch must stay loadable: {v:?}");
        assert!(v.has_warnings());
    }

    /// One pass, every problem. A caller needs to tell "one field off" from
    /// "wrong model entirely", and re-running to discover the second error is
    /// exactly the workflow this avoids.
    #[test]
    fn validation_reports_every_disagreement_in_one_pass() {
        let mut wrong = manifest();
        wrong.r = 8;
        wrong.d_in = 64;
        wrong.d_out = 128;
        wrong.adapter_type = "dora".to_string();
        wrong.alpha = 16.0;

        let v = wrong.validate_against(&manifest());
        assert!(!v.ok());
        assert_eq!(v.errors.len(), 4, "r, d_in, d_out, adapter_type: {v:?}");
        assert!(v.warnings.iter().any(|w| w.contains("alpha")), "{v:?}");
    }

    /// Geometry is fatal, scale is a warning: `r`/`d_in`/`d_out` change stored
    /// tensor shapes, `alpha`/`rslora` change only the number they are
    /// multiplied by.
    #[test]
    fn scale_differences_warn_where_geometry_differences_fail() {
        let mut scaled = manifest();
        scaled.alpha = 32.0;
        scaled.rslora = true;
        let v = scaled.validate_against(&manifest());
        assert!(v.ok(), "a scale difference is loadable: {v:?}");
        assert_eq!(v.warnings.len(), 2, "{v:?}");
    }

    #[test]
    fn a_missing_base_model_id_is_a_warning_and_a_wrong_one_is_an_error() {
        let mut unknown = manifest();
        unknown.base_model_id = None;
        assert!(unknown.validate_against(&manifest()).ok());
        assert!(unknown.validate_against(&manifest()).has_warnings());

        let mut wrong = manifest();
        wrong.base_model_id = Some("llama-3-8b".to_string());
        assert!(!wrong.validate_against(&manifest()).ok());
    }

    #[test]
    fn into_result_keeps_warnings_on_the_success_path_and_joins_errors() {
        let mut v = VLValidation::new();
        v.warn("careful");
        assert_eq!(v.clone().into_result().unwrap(), vec!["careful".to_string()]);

        v.error("first");
        v.error("second");
        let msg = v.into_result().unwrap_err().to_string();
        assert!(msg.contains("first") && msg.contains("second"), "{msg}");
        assert!(msg.contains("2 error(s)"), "{msg}");
    }

    #[test]
    fn a_manifest_built_from_a_lora_config_keeps_its_geometry() {
        let cfg = VLLoraConfig {
            r: 8,
            alpha: 16.0,
            rslora: true,
            d_in: 128,
            d_out: 256,
        };
        let m = VLManifest::for_lora(&cfg, 42);
        assert_eq!(m.lora_config(), cfg);
        assert_eq!(m.step, 42);
        assert_eq!(m.adapter_type, "lora");
    }
}
