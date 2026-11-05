//! D-10: a `null` with no availability entry is a defect.
//!
//! This is the check that keeps [`crate::core::availability`] honest. Without
//! it the sparse map is optional documentation that drifts out of date the
//! first time someone adds a field; with it, "we forgot to say why this is
//! null" is a test failure naming the exact path.
//!
//! # Both directions
//!
//! 1. Every `null` in the emitted JSON must have an entry in `availability`.
//! 2. Every entry in `availability` must point at a path that is `null` (or
//!    absent). A status of `unsupported` on a field that carries a number is
//!    the same class of lie in the other direction, and it is the one a
//!    copy-pasted annotation produces.
//!
//! # Where it runs
//!
//! At session **finalisation** — the write path — so a malformed archive is
//! never written. Deliberately not at export: by then the archive already
//! exists and the check has nothing left to prevent. That ordering is asserted
//! by a test rather than left to convention.

use crate::core::availability::{normalize_path, VLAvailabilityMap};
use crate::export::json::Json;
use crate::validation::integrity::{Severity, ValidationReport};

/// How strictly a D-10 violation is treated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ENNullSemantics {
    /// A violation is an error and the archive is not written. The default:
    /// the invariant is only worth having if it can stop something.
    #[default]
    Strict,
    /// A violation is a warning and the archive is written anyway. For reading
    /// and re-annotating archives produced by an older build.
    Lenient,
}

impl ENNullSemantics {
    /// Parse the `--null-semantics` flag value.
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ENNullSemantics> {
        Some(match s {
            "strict" => ENNullSemantics::Strict,
            "lenient" => ENNullSemantics::Lenient,
            _ => return None,
        })
    }

    /// The severity a violation is reported at under this policy.
    pub fn severity(self) -> Severity {
        match self {
            ENNullSemantics::Strict => Severity::Error,
            ENNullSemantics::Lenient => Severity::Warning,
        }
    }
}

/// The check name every finding from this module carries.
const CHECK: &str = "availability";

/// Collect the normalised dotted path of every `null` in `value`.
///
/// Array indices collapse (`steps[0]` and `steps[7]` both become `steps[]`), so
/// one entry can explain a thousand elements. The `availability` block itself is
/// skipped — it is the annotation, not the annotated.
pub fn null_paths(value: &Json) -> Vec<String> {
    let mut out = Vec::new();
    walk_nulls(value, "", &mut out);
    out.sort();
    out.dedup();
    out
}

fn walk_nulls(value: &Json, path: &str, out: &mut Vec<String>) {
    match value {
        // The root itself being null has no path to record.
        Json::Null if !path.is_empty() => out.push(path.to_string()),
        Json::Null => {}
        Json::Obj(map) => {
            for (key, child) in map {
                // The availability map is the explanation, not a thing needing
                // one. Walking into it would demand an entry explaining an
                // entry.
                if path.is_empty() && key == "availability" {
                    continue;
                }
                let child_path =
                    if path.is_empty() { key.clone() } else { format!("{path}.{key}") };
                walk_nulls(child, &child_path, out);
            }
        }
        Json::Arr(items) => {
            // Index dropped rather than recorded: the collapsed form is the
            // whole point, and emitting `steps[0]`…`steps[999]` would defeat it.
            let child_path = format!("{path}[]");
            for item in items {
                walk_nulls(item, &child_path, out);
            }
        }
        _ => {}
    }
}

/// Look up a normalised dotted path and report whether a status on it explains
/// at least one real gap — a `null`, or a field that is absent.
///
/// For a plain path that is simply "null or absent". For an array column it is
/// "null or absent in *some* element": a column can legitimately be mixed. The
/// CUDA launch profile is the case that forced this — kernel entries carry a
/// launch `config`, the CUDA Graph replay entry cannot (the driver does not
/// expose its internal launches), so `entries[].config` is null in one element
/// and a value in the rest. Direction 1 demands an entry for that null; the
/// entry must therefore be allowed to stand next to the elements that do carry
/// a value. A status on a column with *no* gap anywhere is still the
/// copy-pasted lie Direction 2 exists to catch.
fn has_a_gap(value: &Json, path: &str) -> bool {
    if path.is_empty() {
        return matches!(value, Json::Null);
    }
    // Split off the first segment, keeping any `[]` suffix with it.
    let (head, rest) = match path.split_once('.') {
        Some((h, r)) => (h, Some(r)),
        None => (path, None),
    };
    let (key, is_array) = match head.strip_suffix("[]") {
        Some(k) => (k, true),
        None => (head, false),
    };

    let child = match value.get(key) {
        None | Some(Json::Null) => return true,
        Some(child) => child,
    };
    if is_array {
        let Some(items) = child.as_arr() else {
            return true;
        };
        // An empty column has no element that could carry a value.
        return items.is_empty()
            || items.iter().any(|item| has_a_gap(item, rest.unwrap_or("")));
    }
    has_a_gap(child, rest.unwrap_or(""))
}

/// Run the D-10 check over a serialised session and its availability map.
///
/// Takes the map separately rather than re-parsing it out of the JSON so the
/// caller can check a session *before* deciding to write it.
pub fn check_value(
    value: &Json,
    availability: &VLAvailabilityMap,
    policy: ENNullSemantics,
) -> ValidationReport {
    let mut report = ValidationReport::default();
    let severity = policy.severity();

    // Direction 1: every null needs an entry.
    for path in null_paths(value) {
        if !availability.contains_key(&path) {
            report.push(
                severity,
                CHECK,
                format!(
                    "'{path}' is null with no availability entry — say why it has no value \
                     (measured/estimated/derived/unsupported/unavailable/not_applicable/\
                     not_observed/does_not_exist)"
                ),
            );
        }
    }

    // Direction 2: every entry must explain a real gap — its path is null or
    // absent, or for an array column, null or absent in at least one element.
    for (path, entry) in availability {
        if !has_a_gap(value, path) {
            report.push(
                severity,
                CHECK,
                format!(
                    "availability says '{path}' is {}, but that field carries a value",
                    entry.status.as_str()
                ),
            );
        }
    }

    report
}

/// Run the D-10 check over an archive's JSON text.
///
/// This is the `glbench validate --availability <archive>` entry point, for
/// archives written by an older build. Parses the `availability` block out of
/// the document itself.
pub fn check(session_json: &str) -> ValidationReport {
    let value = match crate::export::json::parse(session_json) {
        Ok(v) => v,
        Err(e) => {
            let mut report = ValidationReport::default();
            report.push(Severity::Error, CHECK, format!("archive is not valid JSON: {e}"));
            return report;
        }
    };
    let availability = match crate::core::availability::from_json(value.get("availability")) {
        Ok(a) => a,
        Err(e) => {
            let mut report = ValidationReport::default();
            report.push(Severity::Error, CHECK, format!("availability block: {e}"));
            return report;
        }
    };
    check_value(&value, &availability, ENNullSemantics::Strict)
}

/// Normalise a path the way the walker does, for callers building a map by hand.
pub fn path_key(path: &str) -> String {
    normalize_path(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::availability::{self, ENAvailability};

    #[test]
    fn a_null_with_no_entry_is_an_error_naming_the_exact_path() {
        let value = Json::obj([(
            "environment",
            Json::obj([("hardware", Json::obj([("gpu_bandwidth_gbs", Json::Null)]))]),
        )]);
        let report = check_value(&value, &VLAvailabilityMap::new(), ENNullSemantics::Strict);

        assert!(!report.passed(), "an unexplained null must fail the session");
        assert_eq!(report.findings.len(), 1);
        assert!(
            report.findings[0]
                .message
                .contains("environment.hardware.gpu_bandwidth_gbs"),
            "finding must name the exact path, got: {}",
            report.findings[0].message
        );
    }

    #[test]
    fn the_same_null_passes_once_it_carries_an_entry() {
        let value = Json::obj([(
            "environment",
            Json::obj([("hardware", Json::obj([("gpu_bandwidth_gbs", Json::Null)]))]),
        )]);
        let mut map = VLAvailabilityMap::new();
        availability::set(
            &mut map,
            "environment.hardware.gpu_bandwidth_gbs",
            ENAvailability::Unsupported,
        )
        .unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(report.passed(), "findings: {:?}", report.findings);
    }

    /// The mirror case: an annotation claiming a field is absent when it is not.
    #[test]
    fn a_status_on_a_field_that_carries_a_value_is_an_error() {
        let value = Json::obj([("measurements", Json::obj([("decode_ms", Json::n(4000.0))]))]);
        let mut map = VLAvailabilityMap::new();
        availability::set(&mut map, "measurements.decode_ms", ENAvailability::Unsupported).unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(!report.passed());
        assert!(
            report.findings[0].message.contains("measurements.decode_ms"),
            "got {}",
            report.findings[0].message
        );
        assert!(report.findings[0].message.contains("carries a value"));
    }

    /// An entry pointing at a field that does not exist at all is allowed: the
    /// `training` slot of an inference-only session is exactly that case.
    #[test]
    fn a_status_on_an_absent_field_is_allowed() {
        let value = Json::obj([("metadata", Json::obj([("label", Json::s("x"))]))]);
        let mut map = VLAvailabilityMap::new();
        availability::set(&mut map, "training", ENAvailability::NotApplicable).unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(report.passed(), "findings: {:?}", report.findings);
    }

    #[test]
    fn array_element_nulls_collapse_to_one_finding_for_the_whole_column() {
        let steps = Json::Arr(vec![
            Json::obj([("tokens", Json::Null), ("loss", Json::n(0.4))]),
            Json::obj([("tokens", Json::Null), ("loss", Json::n(0.3))]),
            Json::obj([("tokens", Json::Null), ("loss", Json::n(0.2))]),
        ]);
        let value = Json::obj([("training", Json::obj([("steps", steps)]))]);

        let paths = null_paths(&value);
        assert_eq!(paths, vec!["training.steps[].tokens".to_string()]);

        let report = check_value(&value, &VLAvailabilityMap::new(), ENNullSemantics::Strict);
        assert_eq!(
            report.findings.len(),
            1,
            "three null elements must produce one finding, not three"
        );
    }

    #[test]
    fn one_entry_explains_every_element_of_an_array_column() {
        let steps = Json::Arr(vec![
            Json::obj([("tokens", Json::Null)]),
            Json::obj([("tokens", Json::Null)]),
        ]);
        let value = Json::obj([("training", Json::obj([("steps", steps)]))]);
        let mut map = VLAvailabilityMap::new();
        availability::set(&mut map, "training.steps[].tokens", ENAvailability::NotApplicable)
            .unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(report.passed(), "findings: {:?}", report.findings);
    }

    /// A mixed column — null in some elements, a value in others — must be
    /// writable: Direction 1 demands an entry for the null elements, so
    /// Direction 2 cannot reject that same entry. Regression test for the
    /// 2026-11-05 T4 audit, where every `--profile stages` glcuda run was
    /// refused because the CUDA Graph replay entry has no launch config.
    #[test]
    fn a_mixed_array_column_accepts_the_entry_its_nulls_require() {
        let steps = Json::Arr(vec![
            Json::obj([("tokens", Json::Null)]),
            Json::obj([("tokens", Json::n(512.0))]),
        ]);
        let value = Json::obj([("training", Json::obj([("steps", steps)]))]);
        let mut map = VLAvailabilityMap::new();
        availability::set(&mut map, "training.steps[].tokens", ENAvailability::NotApplicable)
            .unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(report.passed(), "findings: {:?}", report.findings);

        // And without the entry, the null element is still an error.
        let bare = check_value(&value, &VLAvailabilityMap::new(), ENNullSemantics::Strict);
        assert!(!bare.passed());
    }

    /// If *every* element carries a number, a status over the column explains
    /// nothing — and must be caught.
    #[test]
    fn an_array_column_status_is_rejected_when_every_element_holds_a_value() {
        let steps = Json::Arr(vec![
            Json::obj([("tokens", Json::n(256.0))]),
            Json::obj([("tokens", Json::n(512.0))]),
        ]);
        let value = Json::obj([("training", Json::obj([("steps", steps)]))]);
        let mut map = VLAvailabilityMap::new();
        availability::set(&mut map, "training.steps[].tokens", ENAvailability::NotApplicable)
            .unwrap();

        let report = check_value(&value, &map, ENNullSemantics::Strict);
        assert!(!report.passed());
        assert!(report.findings[0].message.contains("carries a value"));
    }

    #[test]
    fn the_availability_block_itself_is_not_walked_for_nulls() {
        // A long-form entry is an object; nothing inside it needs explaining.
        let value = Json::obj([
            ("measurements", Json::obj([("decode_ms", Json::n(1.0))])),
            (
                "availability",
                Json::obj([("some.path", Json::obj([("status", Json::s("estimated"))]))]),
            ),
        ]);
        assert!(null_paths(&value).is_empty());
    }

    #[test]
    fn lenient_mode_downgrades_the_same_violation_to_a_warning() {
        let value = Json::obj([("telemetry", Json::Null)]);

        let strict = check_value(&value, &VLAvailabilityMap::new(), ENNullSemantics::Strict);
        assert!(!strict.passed());
        assert_eq!(strict.findings[0].severity, Severity::Error);

        let lenient = check_value(&value, &VLAvailabilityMap::new(), ENNullSemantics::Lenient);
        // Same finding, but it no longer blocks the write.
        assert!(lenient.passed());
        assert_eq!(lenient.findings.len(), 1);
        assert_eq!(lenient.findings[0].severity, Severity::Warning);
    }

    #[test]
    fn check_parses_an_archive_and_finds_its_own_availability_block() {
        let text = Json::obj([
            ("telemetry", Json::Null),
            ("availability", Json::obj([("telemetry", Json::s("unavailable"))])),
        ])
        .to_pretty();
        assert!(check(&text).passed());

        let unexplained = Json::obj([("telemetry", Json::Null)]).to_pretty();
        assert!(!check(&unexplained).passed());
    }

    #[test]
    fn malformed_json_is_reported_rather_than_silently_passing() {
        let report = check("{ not json");
        assert!(!report.passed());
        assert!(report.findings[0].message.contains("not valid JSON"));
    }

    #[test]
    fn null_semantics_flag_values_parse_and_reject_anything_else() {
        assert_eq!(ENNullSemantics::from_str("strict"), Some(ENNullSemantics::Strict));
        assert_eq!(ENNullSemantics::from_str("lenient"), Some(ENNullSemantics::Lenient));
        assert_eq!(ENNullSemantics::from_str("relaxed"), None);
        assert_eq!(ENNullSemantics::default(), ENNullSemantics::Strict);
    }

    #[test]
    fn path_key_normalizes_the_same_way_the_walker_does() {
        assert_eq!(path_key("training.steps[7].tokens"), "training.steps[].tokens");
    }
}
