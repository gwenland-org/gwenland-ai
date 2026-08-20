//! [`VLAdapterObservation`] — what was being trained, and how much of it.
//!
//! The point of a LoRA run is that you update a small fraction of the weights.
//! A training archive that records the loss curve but not that fraction cannot
//! answer the first question anyone asks of it — *how much capacity did this
//! actually have?* — so the parameter counts and their ratio are recorded next
//! to the numbers they explain.
//!
//! Read from `VLTrainerConfig`, which glbench hands to stumman in the first
//! place. Nothing here is inferred.

use crate::core::schema::ToJson;
use crate::export::json::Json;

/// Bytes per f32 parameter.
const BYTES_PER_F32: u64 = 4;

/// The adapter a training run used.
#[derive(Debug, Clone, PartialEq)]
pub struct VLAdapterObservation {
    /// Adapter family. `"lora"` is the only one stumman M2 implements.
    pub kind: String,
    /// LoRA rank.
    pub rank: usize,
    /// LoRA alpha. Effective scaling is `alpha / rank`.
    pub alpha: f32,
    /// Effective scaling, `alpha / rank`, derived. Archived because it is what
    /// actually multiplies the update, and a reader should not have to redo the
    /// division to compare two runs.
    pub scaling: f32,
    /// Input dimension of the adapted layer.
    pub d_in: usize,
    /// Output dimension.
    pub d_out: usize,

    /// Trainable parameters: `r*(d_in + d_out)` for a LoRA `A`/`B` pair.
    pub trainable_parameters: usize,
    /// Parameters in the frozen base layer, `d_in * d_out`.
    pub base_parameters: usize,
    /// `trainable / base`. The headline number a LoRA run exists to justify.
    pub parameter_ratio: f64,
    /// Bytes the trainable parameters occupy at f32.
    pub trainable_bytes: u64,
}

impl VLAdapterObservation {
    /// Describe a LoRA adapter from the config the run was given.
    pub fn lora(d_in: usize, d_out: usize, rank: usize, alpha: f32) -> VLAdapterObservation {
        // A LoRA pair is A[d_in, r] and B[r, d_out].
        let trainable = rank * (d_in + d_out);
        let base = d_in * d_out;
        VLAdapterObservation {
            kind: "lora".to_string(),
            rank,
            alpha,
            // Guarded: rank 0 is not a usable adapter, but it must not divide
            // by zero on the way to saying so.
            scaling: if rank > 0 { alpha / rank as f32 } else { 0.0 },
            d_in,
            d_out,
            trainable_parameters: trainable,
            base_parameters: base,
            parameter_ratio: if base > 0 { trainable as f64 / base as f64 } else { 0.0 },
            trainable_bytes: trainable as u64 * BYTES_PER_F32,
        }
    }
}

impl ToJson for VLAdapterObservation {
    fn to_json(&self) -> Json {
        Json::obj([
            ("kind", Json::s(self.kind.clone())),
            ("rank", Json::n(self.rank as f64)),
            ("alpha", Json::n(self.alpha as f64)),
            ("scaling", Json::n(self.scaling as f64)),
            ("d_in", Json::n(self.d_in as f64)),
            ("d_out", Json::n(self.d_out as f64)),
            ("trainable_parameters", Json::n(self.trainable_parameters as f64)),
            ("base_parameters", Json::n(self.base_parameters as f64)),
            ("parameter_ratio", Json::n(self.parameter_ratio)),
            ("trainable_bytes", Json::n(self.trainable_bytes as f64)),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ratios are one division over exact integer counts.
    const TOL_ADAPTER: f64 = 1e-12;

    #[test]
    fn a_lora_pair_counts_both_matrices() {
        // r=8 over a 512x512 layer: A is 512x8, B is 8x512.
        let a = VLAdapterObservation::lora(512, 512, 8, 16.0);
        assert_eq!(a.trainable_parameters, 8 * (512 + 512));
        assert_eq!(a.base_parameters, 512 * 512);
        assert_eq!(a.trainable_bytes, 8192 * 4);
    }

    /// The headline claim of a LoRA run, so it gets an exact check.
    #[test]
    fn the_parameter_ratio_is_the_fraction_actually_being_trained() {
        let a = VLAdapterObservation::lora(512, 512, 8, 16.0);
        // 8192 / 262144 = 1/32
        assert!(
            (a.parameter_ratio - 0.03125).abs() < TOL_ADAPTER,
            "got {}",
            a.parameter_ratio
        );
        assert!(a.parameter_ratio < 1.0, "a LoRA adapter trains less than the base");
    }

    #[test]
    fn scaling_is_alpha_over_rank() {
        let a = VLAdapterObservation::lora(64, 64, 4, 8.0);
        assert!((a.scaling - 2.0).abs() < 1e-6, "got {}", a.scaling);

        // The common alpha == rank convention gives scaling 1.
        let a = VLAdapterObservation::lora(64, 64, 4, 4.0);
        assert!((a.scaling - 1.0).abs() < 1e-6);
    }

    #[test]
    fn a_higher_rank_trains_proportionally_more() {
        let low = VLAdapterObservation::lora(256, 256, 2, 2.0);
        let high = VLAdapterObservation::lora(256, 256, 8, 8.0);
        assert_eq!(high.trainable_parameters, low.trainable_parameters * 4);
        assert!(high.parameter_ratio > low.parameter_ratio);
    }

    /// Rank 0 is not a usable adapter, but describing one must not panic or
    /// produce NaN on the way to reporting it.
    #[test]
    fn a_degenerate_adapter_reports_zeros_rather_than_dividing_by_zero() {
        let a = VLAdapterObservation::lora(64, 64, 0, 1.0);
        assert_eq!(a.trainable_parameters, 0);
        assert!(a.scaling.is_finite());
        assert!(a.parameter_ratio.is_finite());

        let a = VLAdapterObservation::lora(0, 0, 4, 4.0);
        assert_eq!(a.base_parameters, 0);
        assert!(a.parameter_ratio.is_finite(), "zero base must not give NaN");
    }

    #[test]
    fn the_json_projection_has_no_nulls_to_explain() {
        let json = VLAdapterObservation::lora(128, 256, 4, 8.0).to_json();
        let obj = json.as_obj().unwrap();
        assert!(
            obj.values().all(|v| !matches!(v, Json::Null)),
            "every adapter field is known from the config; none may be null"
        );
        assert_eq!(obj.len(), 10);
    }
}
