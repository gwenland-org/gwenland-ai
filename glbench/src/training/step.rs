//! [`VLTrainingStep`] — one training step, in glbench's archived form.
//!
//! # Why this is a separate type from stumman's
//!
//! `stumman::VLTrainingStep` is the *wire* type: what the observer receives
//! inside `train_step`, shaped by what is cheap to collect there. This one is
//! the *archive* type: what a v2 session carries, shaped by what a reader needs
//! years later.
//!
//! They are field-for-field identical today, and copying rather than
//! re-exporting looks redundant because of that. It is not. A re-export would
//! make stumman's struct part of glbench's archive schema, so any field stumman
//! added for its own reasons would silently become a schema change — and
//! `SCHEMA_VERSION` would have to move for a reason no glbench reader cares
//! about. The conversion in [`From`] is the seam where that decision gets made
//! deliberately, and it is one line per field.

use crate::core::schema::ToJson;
use crate::export::json::Json;

/// One training step, as archived facts.
#[derive(Debug, Clone, PartialEq)]
pub struct VLTrainingStep {
    /// Global step index, zero-based.
    pub index: usize,
    /// Epoch this step belongs to.
    pub epoch: usize,
    /// The loss the step returned.
    pub loss: f32,

    /// Nanoseconds in the forward pass.
    pub forward_ns: u64,
    /// Nanoseconds in backward, including `finish_step`.
    pub backward_ns: u64,
    /// Nanoseconds in the optimizer update.
    pub optimizer_ns: u64,
    /// Nanoseconds from the top of the step to the end of the update.
    /// Excludes the observer's own cost.
    pub total_ns: u64,

    /// Trainable parameters that received a gradient. **Not** the size of the
    /// gradient store — stumman's tape returns activations too, and folding
    /// those in would make this number describe the graph rather than the
    /// update. See `stumman::VLTrainingStep::grad_count`.
    pub grad_count: usize,
    /// Total gradient elements across those parameters.
    pub grad_elements: usize,
    /// Global L2 norm over the parameter gradients, non-finite values excluded.
    pub grad_l2_norm: f64,
    /// NaN gradient elements.
    pub grad_nan: usize,
    /// Infinite gradient elements.
    pub grad_inf: usize,

    /// Base learning rate at this step, read from the optimizer.
    pub lr: f64,
}

impl From<&stumman::VLTrainingStep> for VLTrainingStep {
    fn from(s: &stumman::VLTrainingStep) -> VLTrainingStep {
        VLTrainingStep {
            index: s.index,
            epoch: s.epoch,
            loss: s.loss,
            forward_ns: s.forward_ns,
            backward_ns: s.backward_ns,
            optimizer_ns: s.optimizer_ns,
            total_ns: s.total_ns,
            grad_count: s.grad_count,
            grad_elements: s.grad_elements,
            grad_l2_norm: s.grad_l2_norm,
            grad_nan: s.grad_nan,
            grad_inf: s.grad_inf,
            lr: s.lr,
        }
    }
}

impl ToJson for VLTrainingStep {
    fn to_json(&self) -> Json {
        Json::obj([
            ("index", Json::n(self.index as f64)),
            ("epoch", Json::n(self.epoch as f64)),
            ("loss", Json::n(self.loss as f64)),
            ("forward_ns", Json::n(self.forward_ns as f64)),
            ("backward_ns", Json::n(self.backward_ns as f64)),
            ("optimizer_ns", Json::n(self.optimizer_ns as f64)),
            ("total_ns", Json::n(self.total_ns as f64)),
            ("grad_count", Json::n(self.grad_count as f64)),
            ("grad_elements", Json::n(self.grad_elements as f64)),
            ("grad_l2_norm", Json::n(self.grad_l2_norm)),
            ("grad_nan", Json::n(self.grad_nan as f64)),
            ("grad_inf", Json::n(self.grad_inf as f64)),
            ("lr", Json::n(self.lr)),
            // F-05: stumman M2 trains one linear layer with no tokenizer, so
            // these have no subject. Emitted as null with a `not_applicable`
            // status rather than omitted or zeroed — D-04's whole point.
            ("tokens", Json::Null),
            ("sync_ms", Json::Null),
        ])
    }
}

/// Dotted paths this type emits as `null`, for the session's availability map.
///
/// Returned rather than hard-coded at the call site so the list cannot drift
/// away from [`ToJson`] — the two are edited together or the D-10 check fails.
pub const NULL_PATHS: [&str; 2] = ["tokens", "sync_ms"];

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> VLTrainingStep {
        VLTrainingStep {
            index: 3,
            epoch: 1,
            loss: 0.4127,
            forward_ns: 1_000,
            backward_ns: 2_000,
            optimizer_ns: 500,
            total_ns: 3_600,
            grad_count: 2,
            grad_elements: 16,
            grad_l2_norm: 0.87,
            grad_nan: 0,
            grad_inf: 0,
            lr: 1e-3,
        }
    }

    #[test]
    fn the_json_projection_carries_every_measured_field() {
        let json = sample().to_json();
        for field in [
            "index",
            "epoch",
            "loss",
            "forward_ns",
            "backward_ns",
            "optimizer_ns",
            "total_ns",
            "grad_count",
            "grad_elements",
            "grad_l2_norm",
            "grad_nan",
            "grad_inf",
            "lr",
        ] {
            assert!(json.get(field).is_some(), "missing {field}");
            assert!(
                !matches!(json.get(field), Some(Json::Null)),
                "{field} must carry a value, not null"
            );
        }
    }

    /// F-05: the token-denominated fields have no subject at M2. They must be
    /// present and null — never omitted, never zero.
    #[test]
    fn fields_with_no_subject_at_m2_are_null_rather_than_zero() {
        let json = sample().to_json();
        for path in NULL_PATHS {
            assert!(
                matches!(json.get(path), Some(Json::Null)),
                "{path} must be null, not absent and not 0"
            );
        }
    }

    /// The declared null list must match what the projection actually emits, or
    /// the session's availability map annotates the wrong set and D-10 fails.
    #[test]
    fn the_declared_null_paths_match_what_the_projection_emits() {
        let json = sample().to_json();
        let obj = json.as_obj().unwrap();
        let actual: Vec<&str> = obj
            .iter()
            .filter(|(_, v)| matches!(v, Json::Null))
            .map(|(k, _)| k.as_str())
            .collect();
        let mut declared: Vec<&str> = NULL_PATHS.to_vec();
        declared.sort_unstable();
        assert_eq!(actual, declared);
    }
}
