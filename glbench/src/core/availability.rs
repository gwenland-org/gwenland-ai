//! Null semantics: why a field in an archive has no value (D-09, D-10).
//!
//! An archive full of `null`s is unreadable, because `null` conflates eight
//! genuinely different situations. The engine could not produce the number; the
//! platform has no such counter; the event never happened; the field is
//! meaningless for this kind of session. A consumer that cannot tell those
//! apart either over-trusts a missing value or throws the whole archive away.
//!
//! # Shape: a sparse exception map, not a per-field wrapper
//!
//! One `availability` block per session maps a dotted field path to its status,
//! and lists **only the fields that are not plainly measured**:
//!
//! ```json
//! "availability": {
//!   "environment.hardware.gpu.peak_bandwidth_gbs": "unsupported",
//!   "training": "not_applicable",
//!   "analysis.roofline.ceiling_gbs": {
//!     "status": "estimated",
//!     "note": "device capability table, engine::capability::lookup"
//!   }
//! }
//! ```
//!
//! Wrapping every value as `{value, status}` was the alternative. It was
//! rejected because it changes the shape of every field for a benefit that only
//! ever matters on the exception path: here `peak_bandwidth_gbs` is still a
//! number (or `null`) at its existing path, so every reader written against v1
//! keeps working and only the *explanation* is new. It also collapses repeated
//! array elements — `training.steps[].tokens` states the status once for a
//! thousand steps rather than a thousand times.
//!
//! # The invariant that makes it work (D-10)
//!
//! A `null` with no entry in this map is a **defect**, not documentation that
//! someone forgot to write. [`crate::validation::availability`] walks the
//! emitted JSON and fails the session on any unexplained `null` — and on the
//! mirror case, a status attached to a field that actually carries a value.
//! Without that check the map degrades into optional prose.

use std::collections::BTreeMap;

use crate::export::json::Json;

/// Why a field has no value.
///
/// These are not interchangeable. `Unsupported` says the platform *cannot*
/// produce the number; `Unavailable` says it could have and this run did not
/// collect it; `NotObserved` says the instrument was watching and the event
/// simply did not occur. A consumer deciding whether to re-run, change machine,
/// or give up needs exactly that distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ENAvailability {
    /// An instrument produced this value.
    Measured,
    /// Modelled from measurements; the note names the model.
    Estimated,
    /// Computed from other measured fields; the note names the formula.
    Derived,
    /// The platform or runtime cannot produce it.
    Unsupported,
    /// It could exist, but was not collected this run.
    Unavailable,
    /// Meaningless for this session type.
    NotApplicable,
    /// The event did not occur in the observation window.
    NotObserved,
    /// The architectural feature is absent.
    DoesNotExist,
}

impl ENAvailability {
    /// Stable wire identifier, `snake_case`.
    pub fn as_str(self) -> &'static str {
        match self {
            ENAvailability::Measured => "measured",
            ENAvailability::Estimated => "estimated",
            ENAvailability::Derived => "derived",
            ENAvailability::Unsupported => "unsupported",
            ENAvailability::Unavailable => "unavailable",
            ENAvailability::NotApplicable => "not_applicable",
            ENAvailability::NotObserved => "not_observed",
            ENAvailability::DoesNotExist => "does_not_exist",
        }
    }

    /// Parse the wire identifier back. `None` on an unknown string — an archive
    /// from a future build may carry a status this one has never heard of, and
    /// guessing would be worse than saying so.
    /// Inherent `Option`-returning parser rather than a `FromStr` impl,
    /// matching [`crate::core::workload::WorkloadKind::from_str`]: the call
    /// sites want `Option`, not a `Result` with an error type.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ENAvailability> {
        Some(match s {
            "measured" => ENAvailability::Measured,
            "estimated" => ENAvailability::Estimated,
            "derived" => ENAvailability::Derived,
            "unsupported" => ENAvailability::Unsupported,
            "unavailable" => ENAvailability::Unavailable,
            "not_applicable" => ENAvailability::NotApplicable,
            "not_observed" => ENAvailability::NotObserved,
            "does_not_exist" => ENAvailability::DoesNotExist,
            _ => return None,
        })
    }

    /// Whether this status is required to carry a note.
    ///
    /// `Estimated` and `Derived` both assert that a number *was* produced, by a
    /// model or a formula. Which one is the entire content of the claim, so a
    /// bare `"estimated"` says nothing a reader can check.
    pub fn requires_note(self) -> bool {
        matches!(self, ENAvailability::Estimated | ENAvailability::Derived)
    }
}

/// One entry in the availability map: a status, optionally with a note.
///
/// The JSON form is a bare string when there is no note (`"unsupported"`) and
/// an object when there is (`{"status": "estimated", "note": "..."}`). The
/// string form is sugar for the object; parsers accept both, and the writer
/// emits the short one whenever it can.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VLAvailabilityEntry {
    /// Why the field has no value.
    pub status: ENAvailability,
    /// The model or formula behind an `Estimated`/`Derived` status. Required
    /// for those two, optional for the rest.
    pub note: Option<String>,
}

impl VLAvailabilityEntry {
    /// JSON projection: bare string when there is no note, object when there is.
    pub fn to_json(&self) -> Json {
        match &self.note {
            None => Json::s(self.status.as_str()),
            Some(note) => Json::obj([
                ("status", Json::s(self.status.as_str())),
                ("note", Json::s(note.clone())),
            ]),
        }
    }

    /// Parse either form back.
    pub fn from_json(v: &Json) -> Result<VLAvailabilityEntry, String> {
        // Short form: a bare status string.
        if let Some(s) = v.as_str() {
            let status = ENAvailability::from_str(s)
                .ok_or_else(|| format!("unknown availability status '{s}'"))?;
            return Ok(VLAvailabilityEntry { status, note: None });
        }
        // Long form: {"status": ..., "note": ...}.
        let obj = v
            .as_obj()
            .ok_or_else(|| "availability entry is neither a string nor an object".to_string())?;
        let status_str = obj
            .get("status")
            .and_then(|s| s.as_str())
            .ok_or_else(|| "availability entry object has no 'status' string".to_string())?;
        let status = ENAvailability::from_str(status_str)
            .ok_or_else(|| format!("unknown availability status '{status_str}'"))?;
        let note = obj.get("note").and_then(|n| n.as_str()).map(String::from);
        Ok(VLAvailabilityEntry { status, note })
    }
}

/// The sparse map itself, keyed by dotted JSON path with array indices
/// normalised to `[]` (see [`normalize_path`]).
///
/// A `BTreeMap` rather than a `HashMap`: the map is part of the archive, and
/// the archive's digest is computed over its serialisation. Sorted keys make
/// that serialisation a deterministic function of the content.
pub type VLAvailabilityMap = BTreeMap<String, VLAvailabilityEntry>;

/// Record a status with no note.
///
/// Returns `Err` for `Estimated`/`Derived`, which are required to name their
/// model or formula — use [`set_with_note`] for those. Enforcing it at the
/// insertion point is the only place it can be caught before the archive is
/// written; a validator running afterwards can only report the omission.
pub fn set(
    map: &mut VLAvailabilityMap,
    path: &str,
    status: ENAvailability,
) -> Result<(), String> {
    if status.requires_note() {
        return Err(format!(
            "availability '{}' at '{path}' requires a note naming its model or formula; \
             use set_with_note",
            status.as_str()
        ));
    }
    map.insert(
        normalize_path(path),
        VLAvailabilityEntry { status, note: None },
    );
    Ok(())
}

/// Record a status together with the note explaining it.
///
/// Returns `Err` on an empty note: a blank string satisfies the type but not the
/// reason the note is required.
pub fn set_with_note(
    map: &mut VLAvailabilityMap,
    path: &str,
    status: ENAvailability,
    note: &str,
) -> Result<(), String> {
    if note.trim().is_empty() {
        return Err(format!("availability note at '{path}' is empty"));
    }
    map.insert(
        normalize_path(path),
        VLAvailabilityEntry { status, note: Some(note.to_string()) },
    );
    Ok(())
}

/// Normalise a dotted path so every element of an array shares one key:
/// `steps[0].tokens` and `steps[7].tokens` both become `steps[].tokens`.
///
/// This is what lets one entry explain a thousand steps. It also means the map
/// cannot say "step 3 specifically is unavailable" — deliberately: a status
/// that varies per element is a property of the data, and belongs in the data.
pub fn normalize_path(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut in_index = false;
    for c in path.chars() {
        match c {
            '[' => {
                in_index = true;
                out.push('[');
            }
            ']' => {
                in_index = false;
                out.push(']');
            }
            _ if in_index => {} // drop the index digits
            _ => out.push(c),
        }
    }
    out
}

/// JSON projection of the whole map.
pub fn to_json(map: &VLAvailabilityMap) -> Json {
    let mut obj = BTreeMap::new();
    for (path, entry) in map {
        obj.insert(path.clone(), entry.to_json());
    }
    Json::Obj(obj)
}

/// Parse the map back. An absent or `null` block is an empty map, which is what
/// a v1 archive has (D-20) — not an error.
pub fn from_json(v: Option<&Json>) -> Result<VLAvailabilityMap, String> {
    let Some(v) = v else {
        return Ok(VLAvailabilityMap::new());
    };
    if matches!(v, Json::Null) {
        return Ok(VLAvailabilityMap::new());
    }
    let obj = v
        .as_obj()
        .ok_or_else(|| "'availability' is not an object".to_string())?;
    let mut map = VLAvailabilityMap::new();
    for (path, entry) in obj {
        let parsed = VLAvailabilityEntry::from_json(entry)
            .map_err(|e| format!("availability['{path}']: {e}"))?;
        map.insert(path.clone(), parsed);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_round_trips_through_its_wire_identifier() {
        let all = [
            ENAvailability::Measured,
            ENAvailability::Estimated,
            ENAvailability::Derived,
            ENAvailability::Unsupported,
            ENAvailability::Unavailable,
            ENAvailability::NotApplicable,
            ENAvailability::NotObserved,
            ENAvailability::DoesNotExist,
        ];
        for status in all {
            assert_eq!(
                ENAvailability::from_str(status.as_str()),
                Some(status),
                "{} did not round-trip",
                status.as_str()
            );
        }
    }

    #[test]
    fn an_unknown_status_string_is_rejected_rather_than_guessed() {
        assert_eq!(ENAvailability::from_str("probably_fine"), None);
    }

    #[test]
    fn set_refuses_estimated_and_derived_without_a_note() {
        let mut map = VLAvailabilityMap::new();
        for status in [ENAvailability::Estimated, ENAvailability::Derived] {
            let err = set(&mut map, "analysis.roofline.ceiling_gbs", status).unwrap_err();
            assert!(err.contains("requires a note"), "got {err}");
        }
        assert!(map.is_empty(), "a rejected insert must not land in the map");
    }

    #[test]
    fn set_with_note_refuses_a_blank_note() {
        let mut map = VLAvailabilityMap::new();
        let err = set_with_note(&mut map, "a.b", ENAvailability::Estimated, "   ").unwrap_err();
        assert!(err.contains("empty"), "got {err}");
        assert!(map.is_empty());
    }

    #[test]
    fn set_accepts_every_status_that_does_not_require_a_note() {
        let mut map = VLAvailabilityMap::new();
        for status in [
            ENAvailability::Measured,
            ENAvailability::Unsupported,
            ENAvailability::Unavailable,
            ENAvailability::NotApplicable,
            ENAvailability::NotObserved,
            ENAvailability::DoesNotExist,
        ] {
            set(&mut map, status.as_str(), status).unwrap();
        }
        assert_eq!(map.len(), 6);
    }

    #[test]
    fn array_indices_collapse_so_one_entry_explains_every_element() {
        assert_eq!(normalize_path("training.steps[0].tokens"), "training.steps[].tokens");
        assert_eq!(normalize_path("training.steps[997].tokens"), "training.steps[].tokens");
        assert_eq!(normalize_path("a[0].b[12].c"), "a[].b[].c");
        // Paths with no index are untouched.
        assert_eq!(normalize_path("environment.hardware.gpu.name"), "environment.hardware.gpu.name");
    }

    #[test]
    fn set_normalizes_the_path_it_stores() {
        let mut map = VLAvailabilityMap::new();
        set(&mut map, "training.steps[3].tokens", ENAvailability::NotApplicable).unwrap();
        assert!(map.contains_key("training.steps[].tokens"));
        assert!(!map.contains_key("training.steps[3].tokens"));
    }

    #[test]
    fn a_note_free_entry_serializes_as_a_bare_string() {
        let entry = VLAvailabilityEntry { status: ENAvailability::Unsupported, note: None };
        assert_eq!(entry.to_json(), Json::s("unsupported"));
    }

    #[test]
    fn an_entry_with_a_note_serializes_as_an_object() {
        let entry = VLAvailabilityEntry {
            status: ENAvailability::Estimated,
            note: Some("device capability table".to_string()),
        };
        let json = entry.to_json();
        assert_eq!(json.get("status").unwrap().as_str(), Some("estimated"));
        assert_eq!(json.get("note").unwrap().as_str(), Some("device capability table"));
    }

    #[test]
    fn both_entry_forms_parse_back_to_the_same_value() {
        for entry in [
            VLAvailabilityEntry { status: ENAvailability::Unavailable, note: None },
            VLAvailabilityEntry {
                status: ENAvailability::Derived,
                note: Some("total_ms - sum(stage_ms)".to_string()),
            },
        ] {
            let back = VLAvailabilityEntry::from_json(&entry.to_json()).unwrap();
            assert_eq!(back, entry);
        }
    }

    #[test]
    fn the_whole_map_round_trips_through_json() {
        let mut map = VLAvailabilityMap::new();
        set(&mut map, "training", ENAvailability::NotApplicable).unwrap();
        set(&mut map, "environment.hardware.gpu.peak_bandwidth_gbs", ENAvailability::Unsupported).unwrap();
        set_with_note(
            &mut map,
            "analysis.roofline.ceiling_gbs",
            ENAvailability::Estimated,
            "engine::capability::lookup",
        )
        .unwrap();

        let back = from_json(Some(&to_json(&map))).unwrap();
        assert_eq!(back, map);
    }

    #[test]
    fn an_absent_or_null_block_reads_as_an_empty_map_not_an_error() {
        assert!(from_json(None).unwrap().is_empty());
        assert!(from_json(Some(&Json::Null)).unwrap().is_empty());
    }

    #[test]
    fn an_unparseable_entry_names_the_path_it_failed_on() {
        let json = Json::obj([("some.path", Json::s("not_a_real_status"))]);
        let err = from_json(Some(&json)).unwrap_err();
        assert!(err.contains("some.path"), "got {err}");
        assert!(err.contains("not_a_real_status"), "got {err}");
    }
}
